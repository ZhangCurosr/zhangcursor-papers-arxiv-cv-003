# Uncertainty-Guided Adverse Weather Restoration via Gated Transformer Network

Zheke Jin <sup>2</sup>, Yuning Cui <sup>2</sup>, Tianle Jin <sup>2</sup>, Alois Knoll <sup>2</sup>, Fellow, IEEE, Hu Cao <sup>1∗</sup>, Member, IEEE

Abstract—Restoring images degraded by adverse weather remains challenging due to spatially heterogeneous degradations. Many existing weather-specific restoration models rely on weather-agnostic global aggregation, naive cross-scale fusion, and deterministic objectives, which struggle to handle heterogeneous degradations in all-in-one adverse-weather settings. To address these limitations, we propose an Uncertainty-guided Adverse-weather Restoration Network (UAR-Net), a weatherspecific AiO framework that integrates a gated transformer with balanced multi-scale skip connections. Specifically, we employ Gated Dual-scale Transformer Blocks (GDTB) to jointly model selective global interactions and multi-scale local structures, a progressive Balanced Multi-scale Skip Connection (BMSC) for balanced multi-scale feature integration, and an Uncertainty-Aware Refinement Head (URH) that performs artifact removal, detail enhancement, and predictive uncertainty estimation. The model is supervised by a Brightness-Aware Energy Loss (BAE-Loss) to encourage accurate reconstruction with well-calibrated uncertainty. Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple adverseweather benchmarks. The codes will open source upon acceptance.

Index Terms—Image Restoration, Gated Transformer, Balanced Multi-scale Skip Connection, Uncertainty-Aware Refinement Head.

## I. INTRODUCTION

MAGE restoration aims to reconstruct high-quality images from degraded observations and is fundamental to modern computer vision. This is especially critical in safety-related applications such as autonomous driving [1]–[3] and intelligent surveillance [4], where rain, snow, and haze can obscure critical scene content, blur structural boundaries, and reduce visibility, thereby degrading the performance of downstream tasks like object detection, tracking [5]–[7] and semantic segmentation [8]. Consequently, robust adverse-weather restoration has become a prerequisite for reliable perception in realworld systems rather than a purely aesthetic enhancement.

Early adverse-weather restoration methods typically target a single degradation and rely on physical models or hand-crafted priors [9]–[11], which often struggle in complex or real-world scenarios. With deep learning, data-driven models have significantly improved performance for specific tasks such as deraining [12], [13], dehazing [14], [15], and desnowing [16]. However, these methods are designed for individual weather conditions, requiring separate models for different degradations, which limits scalability and robustness in unconstrained environments. Recent advances in adverseweather image restoration have shifted from task-specific models to unified All-in-One (AiO) frameworks tailored for multiple weather degradations within a single network [17], [18]. Early AiO methods rely on shared representations but have limited capacity for global context modeling. More recent transformer-based AiO approaches leverage hierarchical architectures to capture multi-scale features and long-range dependencies, achieving improved restoration performance across diverse weather conditions [19]–[21]. However, most existing approaches aggregate global context in a uniform and weather-agnostic manner, even though different degradations rely on global information differently. This often leads to suboptimal context representations and weak emphasis on severely degraded regions. Meanwhile, skip connections are often implemented as simple concatenation or addition, which may directly propagate heavily corrupted and variable shallow features across weather conditions. This can amplify degradation-specific noise and weaken effective cross-scale feature interaction, hindering robust restoration. Finally, most AiO models adopt deterministic objectives with point-wise predictions. In a unified setting with diverse weather conditions and degradation levels, ignoring predictive uncertainty often leads to over-confident and unstable restorations in ambiguous or severely degraded regions.

![](images/fd27716d6fd9ea9b100163db818ed116e29b2e1e4ee1975b7f881b5363c03cb3.jpg)

![](images/fe8cb41e199afd036166005126c1d9af3438347eaa27239dafa0345dda060a0e.jpg)

![](images/b800d03082e274868e33a22fc3c382bb6fbe274ac1f2d19c2f56c803d56452e2.jpg)

![](images/7d87c312c05d4373dfcfbd63b735008c1962ddc74f23a45889dda280d0f5cb92.jpg)  
Fig. 1: Quantitative comparison on four adverse-weather benchmarks, including Snow100K-S/L, Outdoor-Rain, and Raindrop. UAR-Net (Ours) achieves consistently strong PSNR and SSIM performance across all datasets, ranking favorably against recent SOTA methods and showing stable improvements under different adverse weather conditions.

To address these issues, we propose UAR-Net, a unified adverse-weather restoration network that integrates GDTB with BMSC and URH. Each GDTB combines Selective Gated Attention (SGA) and a Dual-Scale Gated Feed-Forward (DGFF) [19], where SGA applies sinusoidal reweighting to queries and keys together with data-dependent gating to selectively emphasize degradation-related long-range interactions, while DGFF provides complementary local enhancement. Beyond the backbone, we introduce BMSC, which progressively integrates multi-level encoder features instead of performing one-shot concatenation or addition. This design jointly considers multi-scale information [22], [23]. By controlling crossscale information, BMSC reduces the impact of corrupted shallow features and provides more balanced guidance for stable decoding. Building upon the coarse prediction, we introduce URH, which performs fine-grained correction while explicitly estimating pixel-wise uncertainty. The refinement is supervised by the proposed BAE-Loss, leading to robust restoration in severely degraded regions. As shown in Fig. 1, extensive experiments demonstrate that our proposed method achieves SOTA performance across multiple benchmarks, consistently outperforming recent unified models [19], [21], [24]. The main contributions of this work are summarized below:

• We propose UAR-Net, which adopts GDTB with queryconditioned gating to selectively regulate global context aggregation, enabling weather-aware and adaptive feature modeling under diverse adverse conditions.

• We introduce BMSC that replaces direct skip concatenation with progressive cross-scale integration, producing cleaner and better-balanced skip representations and enabling more stable and effective cross-scale information transfer for decoding.

• We propose URH together with BAE-Loss to refine visual details and explicitly model pixel-wise uncertainty, resulting in sharper reconstructions and more reliable uncertainty estimates under severe degradations.

• Extensive experiments on multiple adverse-weather benchmarks demonstrate that our method achieves SOTA performance compared with recent unified restoration models.

## II. RELATED WORK

This section reviews representative approaches to adverseweather image restoration, covering both task-specific methods and unified AiO frameworks.

## A. Task-specific Adverse-Weather Restoration

Early deep learning approaches mainly focus on restoring images degraded by a single type of adverse weather, such as rain streaks, snow particles, or raindrops. For rain streak removal, representative methods aim to separate rain structures from background textures using high-frequency decomposi tion [25] and recurrent context aggregation [26]. Subsequent works improve robustness through uncertainty-guided learning [27]. More recently, transformer-based architectures have been introduced to better capture long-range dependencies in deraining [28]. For snow removal, existing methods therefore often incorporate explicit snow modeling [29] or dense multiscale architectures [30], with additional semantic or contextual priors to improve robustness under heavy snow conditions [16]. For raindrop removal, representative approaches formulate the task as attention-guided image-to-image translation [31] or adopt strong residual learning frameworks for robust inpainting [32], while recent models further enhance contextual reasoning using transformer-based designs [33].

## B. All-in-One Adverse-Weather Restoration

AiO restoration seeks to address multiple adverse-weather degradations within a single unified framework. Early AiO approaches employ recurrent architectures to jointly handle different degradations such as rain and haze [26], or explore architecture-search-based designs for multi-task restoration [4]. Recent advances are largely driven by transformerbased models, which leverage hierarchical architectures and global self-attention to model diverse degradations more effectively. Representative works include histogram-based selfattention for intensity-aware restoration [19], degradationaware transformers [20], gradient-conditioned attention with explicit priors [24], and Morton-order scanning with dual degradation estimation [21]. However, in the AiO setting existing approaches still face notable challenges. Global context is often aggregated in a uniform manner, even though different weather conditions rely on long-range information in distinct ways. And multi-scale feature fusion remains unreliable, as shallow representations exhibit highly variable reliability across adverse-weather scenarios.

## III. METHOD

## A. Overview

As shown in Fig. 2, UAR-Net is a Transformer-based framework for adverse-weather image restoration. Both the encoder and decoder are built from stacked GDTBs for hierarchical feature extraction and reconstruction. Each GDTB integrates SGA, which introduces query-conditioned gating into linear attention to selectively regulate long-range context aggregation, followed by a dual-scale gated feed-forward module for local detail enhancement. To facilitate effective cross-scale interaction, BMSC progressively aggregates features from multiple encoder stages into a balanced skip representation. To retain low-frequency priors and facilitate residual learning, the supplementary skip connections [19] are introduced to highlight degradation residuals through average pooling, pointwise convolution and depthwise convolution. Finally, URH refines the coarse output and jointly predicts the restored image and uncertainty, supervised by the proposed BAE-Loss.

![](images/dc1cd47901c3051aa027f41ff6bf11c836c74e83f0ec5be182b03c9483059fff.jpg)  
Fig. 2: Overall architecture of the proposed UAR-Net. The network follows a U-Netstyle encoderdecoder design built upon GDTB. Balanced Multi-scale Skip Connection (BMSC) progressively integrates multi-level encoder features to form a balanced skip representation. In addition, a Supplementary Skip connection composed of average pooling, $1 \times 1$ convolution, and $3 \times 3$ depth-wise convolution injects low-level structural cues with minimal computational overhead. The decoder produces a coarse feature representation that is further refined by URH. Through parallel mean and variance prediction heads, the network outputs the final restored image together with a pixel-wise uncertainty map, which is supervised by the proposed BAE-Loss and a Correlation Loss to enforce structural consistency.

![](images/ce01565f2f43329e7c9ed1ace11bb0461606a6c9fe5dd4db677faf420b813ff9.jpg)  
Fig. 3: Structure of the Selective Gated Attention (SGA). The query projection is split into an attention query and a gating branch. The gating branch applies a sigmoid activation to generate a data-dependent gate, which multiplicatively modulates the linear attention output, enabling selective and content-adaptive global context aggregation.

## B. Gated Dual-Scale Transformer Blocks (GDTB)

The query projection is split into an attention query and a gating branch. The gating branch applies a sigmoid activation to generate a data-dependent gate, which multiplicatively modulates the linear attention output, enabling selective and content-adaptive global context aggregation.

a) From Softmax Attention to Linear Attention: Given query, key, and value matrices $Q , K , V \in \mathbb { R } ^ { N \times d }$ , standard self-attention is defined as

$$
\operatorname { A t t n } ( Q , K , V ) = \operatorname { S o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) V ,\tag{1}
$$

which explicitly constructs the attention matrix in $\mathbb { R } ^ { N \times N }$ and therefore incurs $\mathcal { O } ( N ^ { 2 } )$ time and memory complexity, 풍풆�풆풍�<sub>becoming prohibitive for high-resolution inputs. The attention</sub> mechanism was generalized by allowing arbitrary similarity functions between queries and keys [34]:

$$
\mathrm { A t t e n t i o n } ( Q , K , V ) _ { i } = { \frac { \sum _ { j = 1 } ^ { N } \sin ( Q _ { i } , K _ { j } ) V _ { j } } { \sum _ { j = 1 } ^ { N } \sin ( Q _ { i } , K _ { j } ) } } ,\tag{2}
$$

where sim $\left( \cdot , \cdot \right)$ denotes a customizable similarity function. When choosing

$$
\mathrm { s i m } ( q , k ) = \mathrm { e x p } \Bigg ( \frac { q ^ { \top } k } { \sqrt { d } } \Bigg ) ,
$$

Eq. (2) reduces to conventional softmax attention. To obtain a decomposable similarity function, one can adopt a kernel $\omega : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R } ^ { + }$ that admits a feature map $\phi ( \cdot )$ such that

$$
\mathrm { s i m } ( x , y ) = \omega ( x , y ) = \phi ( x ) ^ { \top } \phi ( y ) .\tag{3}
$$

Substituting Eq. (3) into Eq. (2) yields

Attention(

$$
, K , V ) _ { i } = \frac { \sum _ { j = 1 } ^ { N } \phi ( Q _ { i } ) ^ { \top } \phi ( K _ { j } ) V _ { j } } { \sum _ { j = 1 } ^ { N } \phi ( Q _ { i } ) ^ { \top } \phi ( K _ { j } ) } .\tag{4}
$$

By exploiting the distributive and associative properties of matrix multiplication, Eq. (4) can be rewritten as

Attention

$$
( Q , K , V ) _ { i } = \frac { \phi ( Q _ { i } ) ^ { \top } \Big ( \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) V _ { j } \Big ) } { \phi ( Q _ { i } ) ^ { \top } \Big ( \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) \Big ) } .\tag{5}
$$

This reformulation avoids explicit computation of all pairwise dot-products $Q _ { i } ^ { \top } K _ { j }$ . Instead, it relies on two global summaries, $\begin{array} { r } { \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) \ ' V _ { j } } \end{array}$ and $\textstyle \sum _ { j = 1 } ^ { N } \phi ( K _ { j } )$ , which can be computed once and shared across all queries. As a result, both time and memory complexity are reduced from $\mathcal { O } ( N ^ { 2 } )$ to $\mathcal { O } ( N )$ for a fixed head dimension. In practice, Linear Attention commonly adopts

$$
\phi ( x ) = \mathrm { E L U } ( x ) + 1
$$

to ensure non-negativity and stable optimization.

b) Selective Gated Attention: In image restoration, selfattention is computationally expensive on high-resolution feature maps, while Window-based or sparse variants limit global context modeling. Linear attention demonstrates promise in global context modeling while maintaining linear complexity [35]. However the AiO setting requires a single model to handle multiple weather degradations, both self-attention and linear attention tend to aggregate global information uniformly. Such uniform aggregation can mix different degradation patterns and weaken degradation-related cues, limiting the models ability to adapt to different weather conditions. Motivated by this observation, we adopt Selective Gated Attention (SGA) as the core attention mechanism in GDTB, as shown in Fig. 2. We build SGA on a sinusoidal feature mapping applied to queries and keys [36], where nearby $\mathrm { { _ { o r } } }$ similar tokens receive higher responses, implicitly encouraging locality in global aggregation. This property is particularly beneficial under different weather conditions, as it helps preserve locally coherent structures (e.g., rain streaks or snowflakes) while preventing distant and unrelated regions from being mixed together, enabling more adaptive use of global context across diverse weather scenarios.

To further enhance the reweighting capability of linear attention, we adopt a sinusoidal modulation of query and key features [36]. Specifically, the similarity between a query at position i and a key at position $j$ is modulated by a cosine function of their relative positions:

$$
s ( Q _ { i } , K _ { j } ) = Q _ { i } ^ { \prime } ( K _ { j } ^ { \prime } ) ^ { \top } \cos \left( \frac { \pi } { 2 } \cdot \frac { i - j } { M } \right) ,\tag{6}
$$

where $Q ^ { \prime } \ = \ \operatorname { R e L U } ( Q ) , \ K ^ { \prime } \ = \ \operatorname { R e L U } ( K )$ , and $M \ \geq \ N$ is a normalization constant. Using the trigonometric identity cos(a − b) = cos a cos b + sin a sin $\mathrm { . } b ,$ Eq. (6) can be decomposed as

$$
\begin{array} { r } { Q _ { i } ^ { \prime } ( K _ { j } ^ { \prime } ) ^ { \top } \cos \left( \frac { \pi ( i - j ) } { 2 M } \right) = \left( Q _ { i } ^ { \prime } \cos \frac { \pi i } { 2 M } \right) \left( K _ { j } ^ { \prime } \cos \frac { \pi j } { 2 M } \right) ^ { \top } } \\ { + \left( Q _ { i } ^ { \prime } \sin \frac { \pi i } { 2 M } \right) \left( K _ { j } ^ { \prime } \sin \frac { \pi j } { 2 M } \right) } \end{array}\tag{7}
$$

Accordingly, we define the sinusoidally reweighted queries and keys as

$$
\begin{array} { r } { Q _ { i } ^ { \mathrm { c o s } } = Q _ { i } ^ { \prime } \cos \left( \frac { \pi i } { 2 M } \right) , \quad Q _ { i } ^ { \mathrm { s i n } } = Q _ { i } ^ { \prime } \sin \left( \frac { \pi i } { 2 M } \right) , } \\ { K _ { j } ^ { \mathrm { c o s } } = K _ { j } ^ { \prime } \cos \left( \frac { \pi j } { 2 M } \right) , \quad K _ { j } ^ { \mathrm { s i n } } = K _ { j } ^ { \prime } \sin \left( \frac { \pi j } { 2 M } \right) . } \end{array}
$$

With these definitions, the attention output at position i can be expressed as

$$
O _ { i } = \frac { \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { c o s } } ( K _ { j } ^ { \mathrm { c o s } } ) ^ { \top } V _ { j } + \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { s i n } } ( K _ { j } ^ { \mathrm { s i n } } ) ^ { \top } V _ { j } } { \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { c o s } } ( K _ { j } ^ { \mathrm { c o s } } ) ^ { \top } + \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { s i n } } ( K _ { j } ^ { \mathrm { s i n } } ) ^ { \top } } .\tag{8}
$$

This formulation preserves the linear computational complexity of kernelized attention while introducing a structured reweighting mechanism through sinusoidal modulation. The sine and cosine components act as complementary channels that encode relative positional relationships and selectively reweight long-range interactions, thereby enhancing expressiveness without sacrificing efficiency.

The attention output at position i is given by

$$
O _ { i } = \frac { \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { c o s } } ( K _ { j } ^ { \mathrm { c o s } } ) ^ { \top } V _ { j } } { \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { c o s } } ( K _ { j } ^ { \mathrm { c o s } } ) ^ { \top } } + \frac { \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { s i n } } ( K _ { j } ^ { \mathrm { s i n } } ) ^ { \top } V _ { j } } { \sum _ { j = 1 } ^ { N } Q _ { i } ^ { \mathrm { s i n } } ( K _ { j } ^ { \mathrm { s i n } } ) ^ { \top } } ,\tag{9}
$$

where N denotes the number of tokens, $Q , K , V \in \mathbb { R } ^ { N }$ ×d are the query, key, and value matrices with head dimension $d ,$ and $Q _ { i } ^ { \mathrm { c o s } } , \bar { Q } _ { i } ^ { \mathrm { s i n } } \in \mathbb { R } ^ { 1 \times d }$ (resp. $K _ { j } ^ { \mathrm { c o s } } , K _ { j } ^ { \mathrm { s i n } } \in \mathbb { R } ^ { 1 \times d } )$ denote the sinusoidally reweighted query and key features. By avoiding explicit construction of the $N \times N$ attention matrix, the computational complexity is reduced from $\mathcal { O } ( N ^ { 2 } )$ to O(N).

To enable more selective utilization of global context, SGA further incorporates a gating mechanism. As illustrated in Fig. 3, the query features are projected and expanded to jointly generate both attention queries and gating signals. Formally, given the input feature $X _ { i }$ at position i, we compute

$$
[ Q _ { i } G _ { i } ] = X _ { i } W _ { q } , K _ { i } = X _ { i } W _ { k } , V _ { i } = X _ { i } W _ { v } ,\tag{10}
$$

where $W _ { q } ~ \in ~ \mathbb { R } ^ { d \times 2 d }$ and $W _ { k } , W _ { v } \ \in \ \mathbb { R } ^ { d \times d }$ are learnable projection matrices. The expanded query projection is split channel-wise into two equal parts: $Q _ { i } ~ \in ~ \mathbb { R } ^ { d }$ is used for attention computation, while $G _ { i } ~ \in ~ \mathbb { R } ^ { d }$ serves as a gating signal. The attention output is then modulated as

$$
{ \tilde { O } } _ { i } = \sigma ( G _ { i } ) \odot O _ { i } ,\tag{11}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and ⊙ represents element-wise multiplication. This gating introduces datadependent non-linearity into the attention pathway [37], breaking the purely linear transformation of the value and output projections and thereby enhancing the expressive capacity of attention modeling. The gate also acts as a selective filter, suppressing less informative tokens while emphasizing relevant ones, leading to more focused global context modeling for diverse adverse-weather degradations. Together, this leads to more focused global context modeling, which is beneficial for diverse adverse-weather degradations.

c) Dual-scale Gated Feed-Forward: To enhance lo-.cal feature modeling under spatially heterogeneous adverseweather degradations, GDTB adopts DGFF [19] by employing parallel depth-wise convolution branches with different receptive fields. Fig. 4 illustrates the detailed structure of the Dual-Scale Gated Feed-Forward (DGFF) module. DGFF enhances local feature modeling by capturing complementary spatial patterns at different receptive fields and adaptively fusing them through a gating mechanism, providing effective multi-scale local refinement within each GDTB.

## C. Balanced Multi-Scale Skip Connection (BMSC)

Instead of directly adding or concatenating encoder features to decoder features at the same resolution, we introduce BMSC as a more robust skip pathway for adverse-weather image restoration. BMSC adopts a top-down multi-scale fusion strategy inspired by FPN, progressively integrating encoder features from deep to shallow into a single balanced skip representation. This design is particularly important under diverse weather conditions, where shallow features may be unevenly corrupted by different degradations (e.g., rain streaks, snowflakes, or haze). By explicitly controlling cross-scale aggregation, BMSC suppresses unreliable shallow responses while preserving complementary fine details and high-level semantics from deeper layers. As illustrated in Fig. 5, the integration is implemented using a predictor–corrector (P–C) scheme [38] followed by a lightweight refinement, providing clean and stable skip guidance for the decoder across different weather scenarios.

<sub>TABLE</sub> <sub>I:</sub> <sub>Classical</sub> <sub>linear</sub> <sub>multistep</sub> <sub>methods</sub> <sub>(Adams–Bashforth</sub> <sub>and</sub> <sub>Adams–Moulton)</sub>upplementary
<table><tr><td colspan="3">Explicit: Adams-Bashforth (AB)</td></tr><tr><td>Step</td><td>Order</td><td>Equation</td></tr><tr><td>1</td><td>1</td><td> $y _ { n + 1 } = y _ { n } + \delta F _ { n }$ </td></tr><tr><td>2</td><td>2</td><td> $\begin{array} { r } { y _ { n + 2 } = y _ { n + 1 } + \frac { \delta } { 2 } ( 3 F _ { n + 1 } - F _ { n } ) } \end{array}$ </td></tr><tr><td>3</td><td>3</td><td> $\begin{array} { r } { y _ { n + 3 } = y _ { n + 2 } + \frac { \delta } { 1 2 } ( 2 3 F _ { n + 2 } - 1 6 F _ { n + 1 } + 5 F _ { n } ) } \end{array}$ </td></tr><tr><td>4</td><td>4</td><td> $\begin{array} { r } { y _ { n + 4 } = y _ { n + 3 } + \frac { \delta } { 2 4 } ( 5 5 F _ { n + 3 } - 5 9 F _ { n + 2 } + 3 7 F _ { n + 1 } - 9 F _ { n } ) } \end{array}$ </td></tr><tr><td colspan="3">Implicit: Adams-Moulton (AM)</td></tr><tr><td>Step</td><td>Order</td><td>Equation</td></tr><tr><td>1</td><td>2</td><td> $\begin{array} { r } { y _ { n + 1 } = y _ { n } + \frac { \delta } { 2 } ( F _ { n } + F _ { n + 1 } ) } \end{array}$ </td></tr><tr><td>2</td><td>3</td><td> $\begin{array} { r } { y _ { n + 2 } = y _ { n + 1 } + \frac { \delta } { 1 2 } ( 5 F _ { n + 2 } + 8 F _ { n + 1 } - F _ { n } ) } \end{array}$ </td></tr><tr><td>3</td><td>4</td><td> $\begin{array} { r } { y _ { n + 3 } = y _ { n + 2 } + \frac { \delta } { 2 4 } ( 9 F _ { n + 3 } + 1 9 F _ { n + 2 } - 5 F _ { n + 1 } + F _ { n } ) } \end{array}$ </td></tr></table>

![](images/003ce5734bb1bf81dbfec9bd91a9afd9268f80c7add73d9367d506d7a52893d6.jpg)  
Fig. 4: Structure of DGFF. DGFF applies a $1 \times 1$ projection, expands features via PixelShuffle, and uses two depth-wise convolution branches $( 3 \times 3$ and $5 \times 5 )$ to capture local patterns at different receptive fields. A gating operation adaptively fuses the two branches, followed by PixelUnshuffle and a final $1 \times 1$ projection to produce the output.

a) Multi-scale Integration: We view the progressive skip fusion process from the perspective of numerical integration. Table I summarizes the explicit Adams–Bashforth (AB) and implicit Adams–Moulton (AM) schemes [38], which motivate the predictor–corrector update used in our BMSC. When the number of integration steps is fixed, implicit methods generally provide higher accuracy and better numerical stability than explicit ones. However, implicit schemes require the fusion direction at the current step, which is unknown before completing the update. The P–C strategy provides a practical compromise: it first predicts the next state using an explicit scheme and then refines it using an implicit correction. Let $\{ X _ { n } \} _ { n = 1 } ^ { L }$ denote encoder features extracted at different depths, where $n \ = \ 1$ corresponds to the shallowest level and $n \ = \ L$ to the deepest (latent) level. BMSC integrates these features from deep to shallow in a fixed order and progressively constructs a balanced skip feature $\left\{ Y _ { n } \right\}$ , where n indexes the integration step. At each step, the encoder feature is aligned to the current skip representation through an operator $g ( \cdot )$ composed of convolution and interpolation, ensuring matched spatial resolution and channel dimension. To enable a principled integration, we model the evolution of the �balanced skip feature as a continuous-time dynamical system:

$$
\dot { Y } ( t ) = F \big ( t , Y ( t ) \big ) = f \big ( Y ( t ) + g ( X ( t ) ) \big ) - Y ( t ) ,\tag{12}
$$

where $f ( \cdot )$ 풍풆�풆풍�denotes a lightweight fusion operator implemented 풍풂풕풆 풕as element-wise addition followed by a ReLU(·) activation. The decay term $- Y ( t )$ serves as a stabilizing mechanism that prevents uncontrolled accumulation of corrupted information, which is particularly important when integrating noisy shallow P-C P-C P-C<sub>features.</sub> <sub>We</sub> <sub>use</sub> <sub>the</sub> <sub>continuous</sub> <sub>formulation</sub> <sub>as</sub> <sub>a</sub> <sub>conceptual</sub> model to explain how the balanced skip feature evolves during multi-scale fusion. In implementation, this evolution is realized by a sequence of discrete fusion steps. At the n-th step, the fusion direction is evaluated as $F ( Y _ { n } , X _ { n } )$ . Using a step size δ, an explicit Euler discretization provides a first prediction:

$$
\bar { Y } _ { n + 1 } = Y _ { n } + \delta \cdot F ( Y _ { n } , X _ { n } ) ,\tag{13}
$$

where $\bar { Y } _ { n + 1 }$ denotes the predicted balanced skip feature. To improve robustness under severe adverse-weather corruption, we further apply a P–C update:

$$
Y _ { n + 1 } = Y _ { n } + { \frac { \delta } { 2 } } { \Big ( } F ( Y _ { n } , X _ { n } ) + F ( { \bar { Y } } _ { n + 1 } , X _ { n + 1 } ) { \Big ) } ,\tag{14}
$$

which can be viewed as a second-order Adams–Moulton correction. This two-point update reduces sensitivity to noisy updates and promotes smoother cross-scale information propagation. More generally, BMSC can be interpreted as a linear multistep integration scheme:

$$
Y _ { n + 1 } = Y _ { n } + \delta \sum _ { j = 0 } ^ { K - 1 } \beta _ { j } F ( Y _ { n - j } , X _ { n - j } ) ,\tag{15}
$$

Here, K denotes the number of integration steps and $\beta _ { j }$ are fixed multistep coefficients. In practice, we adopt a fourstep scheme $( K \ : = \ : 4 )$ , illustrated in Fig. 6. The AB step predicts the next balanced skip feature from recent fusion directions, while the AM step refines it using the newly estimated direction. The balanced feature is constructed at the intermediate encoder resolution (Level-3) to balance spatial detail and semantic robustness. After integration, a lightweight refinement is applied, and the resulting balanced feature is resized and injected into all decoder stages as shared skip guidance, enabling consistent cross-scale information transfer.

![](images/5f54fe87864a2e1322757c60185fb382a7eb3a2c76cf88301e4ccb9ab8ea5839.jpg)  
Fig. 5: Illustration of the proposed Balanced Multi-Scale Skip Connection (BMSC). Encoder features from multiple levels are progressively integrated via a predictor–corrector (P–C) scheme to form a balanced skip representation, which is lightly refined before being fed to the decoder as stable cross-scale guidance.

![](images/7ba572b5e096e501406c1d514b3a6adcdca6e9a7f3190ae5974b777b69d46c30.jpg)  
Fig. 6: Structure of the predictor–corrector (P–C) module and the Fusion Direction Block (FDB). The alignment function g(·) aligns features via convolution and interpolation, while the fusion function f(·) is implemented as element-wise addition followed by a ReLU(·).

Fig. 7 compares encoder features and balanced skip features at different scales. Compared to encoder features, balanced features exhibit reduced noise responses and more coherent spatial structures, suggesting improved cross-scale information integration.

b) Refinement: After integration, the balanced skip feature is further refined before being passed to the decoder. In this work, we adopt an efficient attention module to capture global context. The refined skip feature thus serves as a clean and stable multi-scale guidance for decoding under diverse adverse weather conditions.

To efficiently model long-range dependencies in the refinement stage, we adopt vHeat [39], a physics-inspired attention mechanism that reformulates global information aggregation as a heat diffusion process. Unlike conventional self-attention, which explicitly computes pairwise token interactions with quadratic complexity, vHeat propagates information smoothly across the feature map via diffusion, enabling efficient and stable global context modeling for high-resolution features.

Heat diffusion formulation. The design of vHeat is motivated by the classical two-dimensional heat equation, which describes how temperature diffuses over space and time:

$$
{ \frac { \partial u } { \partial t } } = k \left( { \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } } + { \frac { \partial ^ { 2 } u } { \partial y ^ { 2 } } } \right) ,\tag{16}
$$

where $u ( x , y , t )$ denotes the temperature at spatial location $( x , y )$ and time t, and k is the thermal diffusivity. This equation characterizes a smooth and global propagation process, where information naturally spreads from each location to the entire spatial domain. By transforming the heat equation into the frequency domain, the diffusion process admits a closed-form solution:

$$
\tilde { u } ( \omega _ { x } , \omega _ { y } , t ) = \tilde { f } ( \omega _ { x } , \omega _ { y } ) \exp \big ( - k ( \omega _ { x } ^ { 2 } + \omega _ { y } ^ { 2 } ) t \big ) ,\tag{17}
$$

where $\tilde { f } ( \cdot )$ is the frequency-domain representation of the input signal. This formulation reveals that heat diffusion corresponds to a frequency-dependent attenuation, where different frequency components are modulated according to the diffusion strength.

Heat Conduction Operator (HCO). Building upon this observation, the Heat Conduction Operator (HCO) refine feature maps in a fully differentiable manner. Given an input feature map $U _ { 0 } \in \mathbb { R } ^ { H \times W \times C }$ , HCO is defined as

$$
U _ { t } = \mathrm { I D C T 2 D } \big ( \mathrm { D C T 2 D } ( U _ { 0 } ) \cdot \exp \big ( - k ( \omega _ { x } ^ { 2 } + \omega _ { y } ^ { 2 } ) t \big ) \big ) _ { \mathrm { ~ } } .\tag{18}
$$

where DCT2D(·) and IDCT2D(·) denote the 2D Discrete Cosine Transform and its inverse, respectively. The DCT provides an efficient approximation of the Fourier transform under Neumann boundary conditions, making it well suited for image-like feature maps. In this formulation, global context aggregation is achieved through frequency-domain diffusion rather than explicit token-to-token interaction. As a result, HCO maintains a global receptive field while avoiding the quadratic cost of self-attention, achieving a computational complexity of $\mathcal { O } ( N ^ { 1 . 5 } )$ for an N-pixel feature map.

Adaptive diffusion and refinement. To enable content-aware refinement, the diffusion strength k is not fixed but predicted dynamically using learnable Frequency Value Embeddings (FVEs). This allows the model to adaptively control the extent of diffusion according to the input content, balancing global structure propagation and local detail preservation.

Overall, vHeat provides an efficient and interpretable alternative to self-attention for the refinement stage. By modeling feature interactions as a diffusion process, it enables smooth global information propagation, stable optimization, and scalability to high-resolution inputs, making it particularly suitable for fine-grained image restoration.

The effectiveness of this refinement design is further validated through ablation studies in Section V-B, where we compare different refinement strategies and key architectural components.

![](images/3174913d7935d5714ab958f27f175a7872dd5e1903c2c7e8ba50c194002e2701.jpg)  
Fig. 7: Visualization of encoder features (enc level1–level3, top row) and the corresponding balanced skip features (balanced level1–level3, bottom row) fed into the decoder. Compared to raw encoder features, balanced features exhibit reduced noise and more coherent spatial structures, indicating effective cross-scale information integration by BMSC.

## D. Uncertainty-Aware Refinement Head (URH)

Many restoration frameworks generate a coarse prediction followed by a refinement head [19], [21], [24], [40]. However, under severe degradations, coarse outputs often exhibit blurred boundaries and inconsistencies, limiting refinement effectiveness [41]. Moreover, most refinement modules are shallow and single-scale, making it difficult to handle large artifacts or ambiguous structures, and treating refinement as deterministic often leads to over-confident and unstable predictions. To address these issues, we propose URH, which performs finegrained correction with uncertainty modeling. URH adopts a compact four-stage U-Net with GDTB blocks to enhance multi-scale details, while retaining standard skip connections since cross-scale fusion is already handled by earlier stages.

URH produces a probabilistic output via two parallel heads that predict the per-pixel mean $\mu ( x )$ and variance $\sigma ^ { 2 } ( x )$ Rather than serving as a strictly calibrated uncertainty estimate, the predicted variance acts as a task-driven signal that reflects the relative difficulty of restoration across spatial regions. In particular, severely degraded areas (e.g., heavy snow or haze) tend to exhibit higher variance, while clean or wellobserved regions show lower variance. This design enables uncertainty-aware refinement, where the variance highlights ambiguous regions and modulates the refinement process. As a result, the model reduces over-confident predictions in difficult areas and achieves more stable and robust restoration under heterogeneous degradations.

a) Brightness-Aware Energy Loss: Most existing adverse-weather restoration methods rely on point-wise $\ell _ { 1 }$ or $\ell _ { 2 }$ losses, which work well in lightly degraded regions but often produce over-confident predictions in severely corrupted or structurally ambiguous areas. This issue becomes more pronounced when handling diverse weather conditions and degradation levels, as the resulting ambiguity and uncertainty increase during restoration. Since adverse-weather restoration is inherently a dense regression problem with spatially varying ambiguity, explicitly modeling pixel-wise uncertainty is crucial for robust and stable prediction. To this end, we supervise URH using the proposed BAE-Loss, inspired by heteroscedastic uncertainty modeling in dense prediction tasks [42]. URH predicts a per-pixel Gaussian distribution parameterized by a mean image $\mu ( x )$ and a variance map. To evaluate the quality of such probabilistic predictions, we adopt an energy-based scoring rule, which is a strictly proper and non-local metric for multivariate probabilistic forecasts [43]. Given a ground-truth image $z _ { n }$ and M Monte Carlo samples $z _ { n , i } i = 1 ^ { \mathbf { \overline { { M } } } }$ drawn from $\mathcal { N } ( \mu ( \boldsymbol { x } _ { n } ) , \Sigma ( \boldsymbol { x } _ { n } ) )$ (using 1000 Monte Carlo samples), the Energy Score is approximated as

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { E S } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| z _ { n , i } - z _ { n } \|  } } \\ & { \quad \quad  - \frac { 1 } { 2 ( M - 1 ) } \sum _ { i = 1 } ^ { M - 1 } \| z _ { n , i } - z _ { n , i + 1 } \| ) } \end{array}\tag{19}
$$

where the distance is computed using a pixel-wise $\ell _ { 1 }$ norm, which provides stable gradients and is better suited to highresolution image restoration under heavy degradation. Owing to its non-local nature, this formulation encourages the predicted distribution to place probability mass near the ground truth rather than matching it at a single point, leading to more robust uncertainty estimation. In addition to uncertainty modeling, adverse-weather images often exhibit global brightness shifts caused by haze, snow accumulation, or illumination changes. To alleviate this issue, we incorporate a brightnessaware regression term [44]. Let $f ( x )$ denote the predicted mean image and y the ground truth. The brightness-aware regression loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G T } } ( f ( x ) , y ) = W \left\| f ( x ) - y \right\| _ { 1 } + ( 1 - W ) \left\| \frac { \mu _ { y } } { \mu _ { f ( x ) } } f ( x ) - y \right\| _ { 1 } , } \end{array}\tag{20}
$$

where $\mu _ { f ( x ) }$ and $\mu _ { y }$ denote the mean brightness of the predicted image $f ( x )$ and the ground truth y, respectively. The adaptive weight $W \in [ 0 , 1 ]$ is computed based on the brightness discrepancy between $f ( x )$ and y using a Bhattacharyyadistance-based measure. Specifically, we model the brightness statistics of the ground truth and prediction as Gaussian distributions:

$$
p \sim { \mathcal N } ( \mu _ { y } , \sigma _ { y } ^ { 2 } ) , \quad q \sim { \mathcal N } ( \mu _ { f ( x ) } , \sigma _ { f ( x ) } ^ { 2 } ) ,\tag{21}
$$

where $\mu _ { y }$ and $\mu _ { f ( x ) }$ denote the mean brightness of y and $f ( x )$ and $\sigma _ { y } ^ { 2 }$ and $\sigma _ { f ( x ) } ^ { 2 }$ denote the corresponding variances. The Bhattacharyya distance between $p$ and $q$ is computed as:

$$
D _ { B } ( p \| q ) = \frac { 1 } { 4 } \frac { ( \mu _ { y } - \mu _ { f ( x ) } ) ^ { 2 } } { \sigma _ { y } ^ { 2 } + \sigma _ { f ( x ) } ^ { 2 } } + \frac { 1 } { 2 } \ln \left( \frac { \sigma _ { y } ^ { 2 } + \sigma _ { f ( x ) } ^ { 2 } } { 2 \sigma _ { y } \sigma _ { f ( x ) } } \right) .\tag{22}
$$

The adaptive weight $W$ is obtained by clipping $D _ { B }$ to the range [0, 1]. This formulation adaptively balances the original prediction and its brightness-aligned counterpart according to global illumination consistency. Finally, we combine the energy-based uncertainty loss and the brightness-aware regression term to form the proposed Brightness-Aware Energy Loss:

$$
\mathcal { L } _ { \mathrm { B A E } } = \left( 1 - w _ { \mathrm { p r o b } } ( t ) \right) \mathcal { L } _ { \mathrm { G T } } + w _ { \mathrm { p r o b } } ( t ) \mathcal { L } _ { \mathrm { E S } } ,\tag{23}
$$

where $w _ { \mathrm { p r o b } } ( t ) ~ \in ~ [ 0 , 1 ]$ is an annealing weight that gradually increases during training. This design allows URH to first focus on stable brightness-aware reconstruction and progressively incorporate probabilistic supervision, resulting in sharper refinements and better-calibrated uncertainty under severe and ambiguous degradations.

## E. Total Loss

The final training objective further incorporates a correlation loss ${ \mathcal L } _ { \mathrm { c o r } }$ [45]. consistency:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B A E } } + \mathcal { L } _ { \mathrm { c o r } } . } \end{array}\tag{24}
$$

The correlation loss is defined based on the Pearson correlation coefficient between the restored image $I _ { \mathrm { H Q } }$ and the ground truth $I _ { \mathrm { G T } } { \mathrm { : } }$

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathrm { c o r } } ( I _ { \mathrm { H Q } } , I _ { \mathrm { G T } } ) = \frac { 1 } { 2 } \left( 1 - \rho ( I _ { \mathrm { H Q } } , I _ { \mathrm { G T } } ) \right) , } \\ & { \quad \displaystyle \rho ( I _ { \mathrm { H Q } } , I _ { \mathrm { G T } } ) = \frac { \sum _ { i = 1 } ^ { N } ( I _ { i , \mathrm { H Q } } - \bar { I } _ { \mathrm { H Q } } ) ( I _ { i , \mathrm { G T } } - \bar { I } _ { \mathrm { G T } } ) } { N \sigma ( I _ { \mathrm { H Q } } ) \sigma ( I _ { \mathrm { G T } } ) } . } \end{array}\tag{25}
$$

This loss encourages the restored image to preserve global structural and intensity consistency with the ground truth, complementing the pixel-wise supervision and uncertainty modeling in BAE-Loss.

## IV. EXPERIMENTAL SETTING

## A. Training Details

Our model is implemented in PyTorch and trained from scratch on four NVIDIA H100 GPUs for a total of 300,000 iterations. We adopt a progressive learning strategy with five training stages. At stage k, a patch size $s _ { k } .$ , a per-GPU mini-batch size $b _ { k } .$ , and a training length of $T _ { k }$ iterations are used, where $^ { S } k { \in } \{ 1 , . . . , 5 \}$ = {128, 160, 256, 320, 360}, $b _ { k \in \{ 1 , \ldots , 5 \} } ~ = ~ \{ 8 , 5 , 2 , 1 , 1 \}$ , and $T _ { k \in \{ 1 , \dots , 5 \} } = \{ 9 2 , 0 0 0 , 8 4 , 0 0 0 , 5 6 , 0 0 0 , 3 6 , 0 0 0 , 3 2 , 0 0 0 \}$ , with $\textstyle \sum _ { k } { \dot { T } } _ { k } = { \dot { 3 } } 0 0 , 0 0 0$ . This schedule progressively increases the effective patch size while reducing the batch size, enabling the network to learn higher-resolution content without exceeding GPU memory limits. We use the AdamW optimizer with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , which is kept constant for the first 92,000 iterations and then decayed to $1 \times 1 0 ^ { - 6 }$ using a cosine annealing schedule over the remaining iterations. The main architectural hyperparameters are as follows: the numbers of blocks at the four encoderdecoder stages are $L _ { i \in \{ 1 , 2 , 3 , 4 \} } \ = \ \{ 4 , 4 , 6 , 8 \}$ , the base channel dimension is $C = 3 6 .$ , the channel expansion factor in DGFF is $r = 2 . 6 6 7$ and the numbers of attention heads at the four stages are $\{ 1 , 2 , 4 , 8 \}$ . For data augmentation, random horizontal and vertical flips are applied during training.

## B. Datasets

Snow100K [29] contains 100K synthetic snowy images generated from clean outdoor scenes with different snow densities and particle sizes. Following common practice, we use 9,000 images for training. For testing, we adopt three subsets: Snow100K-S (small-particle snow), Snow100K-L (largeparticle snow), and Snow100K-Real (real snowy scenes).

Raindrop [31] provides 1,319 real-world image pairs degraded by adherent raindrops. We use 1,069 pairs for training and 249 pairs for testing. This dataset focuses on localized, non-uniform occlusions that obscure important image regions.

Outdoor-Rain [46] consists of 9,000 synthetic images with combined rain streaks and fog, simulating complex atmospheric degradations. It complements the above datasets by introducing mixed rainhaze conditions.

During training, we merge Snow100K, Raindrop, and Outdoor-Rain into a unified multi-weather training set that covers both synthetic and real degradations. For evaluation, we report results on Snow100K-S/L, the Raindrop test set, and the Outdoor-Rain Test1 split.

## C. Evaluation Metrics

We adopt two standard full-reference metrics to evaluate restoration quality: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

a) Peak Signal-to-Noise Ratio (PSNR): PSNR is derived from the Mean Squared Error (MSE) between the restored image $I _ { h q }$ and ground truth $I _ { g t }$

$$
\mathrm { M S E } = \frac { 1 } { H W } \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } \big ( I _ { h q } ( i , j ) - I _ { g t } ( i , j ) \big ) ^ { 2 } ,\tag{26}
$$

$$
\mathrm { P S N R } = 1 0 \cdot \log _ { 1 0 } \left( \frac { M A X ^ { 2 } } { \mathrm { M S E } } \right) ,\tag{27}
$$

where H and W are height and width, and MAX is the maximum pixel value (e.g., 255 for 8-bit images). Higher PSNR means smaller pixel-wise error.

b) Structural Similarity Index (SSIM): SSIM measures perceptual similarity in terms of luminance, contrast, and structure:

$$
\mathrm { S S I M } ( I _ { h q } , I _ { g t } ) = \frac { ( 2 \mu _ { h q } \mu _ { g t } + C _ { 1 } ) ( 2 \sigma _ { h q , g t } + C _ { 2 } ) } { ( \mu _ { h q } ^ { 2 } + \mu _ { g t } ^ { 2 } + C _ { 1 } ) ( \sigma _ { h q } ^ { 2 } + \sigma _ { g t } ^ { 2 } + C _ { 2 } ) } ,\tag{28}
$$

where $\mu _ { h q } , \mu _ { g t }$ are mean intensities, $\sigma _ { h q } ^ { 2 } , \sigma _ { g t } ^ { 2 }$ are variances, and $\sigma _ { h q , g t }$ is the covariance between $I _ { h q }$ and $I _ { g t } . \ C _ { 1 }$ and $C _ { 2 }$ are small constants for numerical stability. SSIM ranges from −1 to 1, with larger values indicating better structural similarity.

c) Learned Perceptual Image Patch Similarity (LPIPS): LPIPS [47] is a learned perceptual metric that aligns image similarity with human visual perception more closely than traditional pixel-wise measures. Instead of computing differences in the RGB space, LPIPS evaluates perceptual similarity by measuring distances between deep feature representations extracted from a fixed pretrained convolutional network, such as AlexNet or VGG. Given a restored image $I _ { h q }$ and the ground truth $I _ { g t } ,$ , both images are forwarded through the network, and let ϕ<sub>l</sub>(·) denote the activation at the l-th layer. Following [47], the perceptual distance is computed as a learned weighted feature difference:

$$
\begin{array} { r l } & { \mathrm { L P I P S } ( I _ { h q } , I _ { g t } ) = \displaystyle \sum _ { l } \frac { 1 } { H _ { l } W _ { l } } \sum _ { h , w } \left\| w _ { l } \odot \left( \hat { \phi } _ { l } ( I _ { h q } ) _ { h , w } \right. } \\ & { \left. \qquad - \hat { \phi } _ { l } ( I _ { g t } ) _ { h , w } \right) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{29}
$$

where $\hat { \phi } _ { l } ( \cdot )$ denotes channel-wise normalized feature maps, w<sub>l</sub> are learned per-channel weights, and $H _ { l }$ and $W _ { l }$ are the spatial dimensions at layer l. By computing distances in deep feature space, LPIPS captures perceptual differences related to texture, structure, and semantic consistency that are often overlooked by pixel-wise metrics, with lower values indicating higher perceptual similarity.

d) Q-Align: Q-Align [48] is a no-reference perceptual quality metric proposed to align machine-predicted visual scores with human subjective judgments. Unlike conventional no-reference IQA models that directly regress continuous scores, Q-Align emulates the human rating process by predicting discrete, text-defined quality levels (e.g., bad, poor, fair, good, excellent) using a large multimodal model. During inference, the probabilities of these rating levels are extracted and converted into a final quality score via weighted averaging, analogous to the computation of mean opinion scores (MOS) in subjective studies. By leveraging semantic reasoning and discrete-level supervision, Q-Align demonstrates strong robustness and cross-dataset generalization, making it particularly suitable for evaluating perceptual quality under complex and previously unseen degradations.

e) Multi-scale Image Quality (MUSIQ): MUSIQ [49] is a no-reference image quality assessment (NR-IQA) metric that predicts perceptual image quality directly from a single input image, without requiring a reference. It explicitly models image quality across multiple spatial scales, reflecting the fact that human perception jointly considers local details and global composition. MUSIQ extracts patchbased representations from the image at multiple resolutions and employs a Transformer to aggregate quality-related cues through self-attention. By jointly encoding spatial location and scale information, MUSIQ effectively captures both finegrained distortions and large-scale structural degradations. The final quality score is obtained by regressing from a global representation that summarizes the multi-scale features. Higher MUSIQ values indicate better perceptual image quality.

## V. EXPERIMENTS

Following previous work [19], UAR-Net is evaluated on standard benchmarks for adverse-weather restoration [29], [31], [46].

## A. Experimental Results and Comparisons

We evaluate UAR-Net against a wide range of representative adverse-weather image restoration methods, including both unified and task-specific approaches. Specifically, Table II reports comparisons with unified models [4], [19], [21], [24], [40], [45], [50]–[52]. In addition, task-specific comparisons on snow, rain, and raindrop removal are presented in Table III.

As shown in Tables II and III, UAR-Net consistently achieves SOTA performance across all benchmarks. On unified evaluation (Table II), UAR-Net outperforms Histoformer [19] by an average PSNR margin of +0.78 dB, with notable gains on Snow100K-S (+0.93 dB), Snow100K-L (+0.61 dB), Outdoor-Rain (+1.32 dB), and Raindrop (+0.26 dB), and further surpasses MODEM [21] and HOGformer [24] by clear margins. Task-specific comparisons show consistent improvements as well: UAR-Net achieves the best PSNR and SSIM on both Snow100K-S and Snow100K-L (Table III(a)), improves PSNR from 33.10 to 33.40 on Outdoor-Rain (Table III(b)), and yields a +0.31 dB PSNR gain on raindrop removal (Table III(c)). demonstrating strong robustness to both large-scale and localized adverse-weather degradations.

a) Perceptual quality evaluation: Beyond distortionbased metrics, we further evaluate perceptual quality using both full-reference and no-reference metrics, including LPIPS [47], Q-Align [48], and MUSIQ [49]. The results are reported in Table IV. UAR-Net consistently achieves the lowest LPIPS scores and the highest Q-Align and MUSIQ scores across all datasets, indicating that our method not only reduces pixel-wise errors but also produces more perceptually pleasing and natural results. The consistent improvements on both full-reference and no-reference metrics suggest that UAR-Net better balances distortion reduction and perceptual fidelity, which is crucial for real-world adverse-weather restoration.

b) T-SNE feature visualization: Fig. 8 presents a t-SNE visualization of the encoder features from MODEM and our method. MODEM shows scattered feature distributions with noticeable overlap across different weather conditions, indicating limited feature separability. In contrast, our method yields more compact intra-class clusters and clearer inter-class separation, suggesting more condition-aware and disentangled representations. This improved feature organization reflects the effectiveness of the gated attention and balanced multi-scale skip design in reducing interference across adverse-weather conditions.

TABLE II: Quantitative comparison on four adverse-weather benchmarks. Values are PSNR (dB) and SSIM. Best results are bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Snow100K-S</td><td colspan="2">Snow100K-L</td><td colspan="2">Outdoor-Rain</td><td colspan="2">Raindrop</td><td colspan="2">Avg.</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>All-in-One [4]</td><td></td><td></td><td>28.33</td><td>0.8820</td><td>24.71</td><td>0.8980</td><td>31.12</td><td>0.9268</td><td></td><td></td></tr><tr><td>Trans Weather [50]</td><td>32.51</td><td>0.9341</td><td>29.31</td><td>0.8879</td><td>28.83</td><td>0.9000</td><td>30.17</td><td>0.9157</td><td>30.20</td><td>0.9094</td></tr><tr><td>Restormer [40]</td><td>36.02</td><td>0.9579</td><td>30.36</td><td>0.9068</td><td>30.03</td><td>0.9215</td><td>32.18</td><td>0.9408</td><td>32.21</td><td>0.9317</td></tr><tr><td>Chen et al. [51]</td><td>34.42</td><td>0.9469</td><td>30.22</td><td>0.9071</td><td>29.27</td><td>0.9147</td><td>31.81</td><td>0.9309</td><td>31.43</td><td>0.9249</td></tr><tr><td>WGWSNet [52]</td><td>34.31</td><td>0.9460</td><td>30.16</td><td>0.9007</td><td>29.32</td><td>0.9207</td><td>32.38</td><td>0.9378</td><td>31.54</td><td>0.9263</td></tr><tr><td>WeatherDiff64 [45]</td><td>35.83</td><td>0.9566</td><td>30.09</td><td>0.9041</td><td>29.64</td><td>0.9312</td><td>30.71</td><td>0.9312</td><td>31.57</td><td>0.9308</td></tr><tr><td>PromptIR [53]</td><td>36.88</td><td>0.9643</td><td>31.34</td><td>0.9200</td><td>30.80</td><td>0.9229</td><td>32.20</td><td>0.9359</td><td>32.80</td><td>0.9357</td></tr><tr><td>DiffUIR-L [54]</td><td></td><td></td><td>30.64</td><td>0.9082</td><td>30.89</td><td>0.9231</td><td>31.90</td><td>0.9368</td><td></td><td></td></tr><tr><td>Histoformer [19]</td><td>37.41</td><td>0.9656</td><td>32.16</td><td>0.9261</td><td>32.08</td><td>0.9389</td><td>33.06</td><td>0.9441</td><td>33.68</td><td>0.9437</td></tr><tr><td>MODEM [21]</td><td>38.08</td><td>0.9673</td><td>32.52</td><td>0.9292</td><td>33.10</td><td>0.9410</td><td>33.01</td><td>0.9434</td><td>34.18</td><td>0.9452</td></tr><tr><td>HOGformer [24]</td><td>37.93</td><td>0.9685</td><td>32.41</td><td>0.9297</td><td>32.89</td><td>0.9460</td><td>32.72</td><td>0.9452</td><td>33.99</td><td>0.9474</td></tr><tr><td>Ours</td><td>38.34</td><td>0.9697</td><td>32.77</td><td>0.9324</td><td>33.40</td><td>0.9491</td><td>33.32</td><td>0.9487</td><td>34.46</td><td>0.9500</td></tr></table>

TABLE III: Quantitative comparison on adverse-weather removal tasks.

(a) Snow removal
<table><tr><td>Method</td><td colspan="2">Snow-S</td><td>Snow-L</td></tr><tr><td></td><td>PSNR SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>SPANet [55]</td><td>29.92</td><td>0.8260</td><td>23.70 0.7930</td></tr><tr><td>JSTASR [56]</td><td>31.40</td><td>0.9012</td><td>25.32 0.8076</td></tr><tr><td>RESCAN [26]</td><td>31.51</td><td>0.9032</td><td>26.08 0.8108</td></tr><tr><td>DesnowNet [29]</td><td>32.33</td><td>0.9500</td><td>27.17 0.8983</td></tr><tr><td>DDMSNet [16]</td><td>34.34</td><td>0.9445</td><td>28.85 0.8772</td></tr><tr><td>ConvIR [57]</td><td>37.98</td><td>0.9686</td><td>32.11 0.9300</td></tr><tr><td>MODEM [21]</td><td>38.08</td><td>0.9673</td><td>32.52 0.9292</td></tr><tr><td>Ours</td><td>38.34</td><td>0.9697 32.77</td><td>0.9324</td></tr></table>

(b) Rain removal
<table><tr><td>Method</td><td>PSNR</td><td>SSIM</td></tr><tr><td>CycleGAN [58] pix2pix [59]</td><td>17.62 19.09</td><td>0.6560 0.7100</td></tr><tr><td>HRĠAN [46]</td><td>21.56</td><td>0.8550</td></tr><tr><td>PCNet [60]</td><td>26.19</td><td>0.9015</td></tr><tr><td>MPRNet [61]</td><td>28.03</td><td>0.9192</td></tr><tr><td>NAFNet [17]</td><td>29.59</td><td>0.9027</td></tr><tr><td>MODEM [21]</td><td>33.10</td><td>0.9410</td></tr><tr><td>Ours</td><td>33.40</td><td>0.9491</td></tr></table>

(c) Raindrop removal
<table><tr><td>Method</td><td>PSNR</td><td>SSIM</td></tr><tr><td>pix2pix [59]</td><td>28.02</td><td>0.8547</td></tr><tr><td>DuRN [32]</td><td>31.24</td><td>0.9259</td></tr><tr><td>RaindropAttn [62]</td><td>31.44</td><td>0.9263</td></tr><tr><td>AttentiveGAN [31]</td><td>31.59</td><td>0.9170</td></tr><tr><td>MAXIM [33]</td><td>31.87</td><td>0.9352</td></tr><tr><td>AST [63]</td><td>30.57</td><td></td></tr><tr><td>MODEM [21]</td><td></td><td>0.9333</td></tr><tr><td></td><td>33.01</td><td>0.9434</td></tr><tr><td>Ours</td><td>33.32</td><td>0.9487</td></tr></table>

TABLE IV: Comparison of perceptual metrics. LPIPS is fullreference (↓), while Q-Align and MUSIQ are no-reference metrics (↑).
<table><tr><td>Method</td><td></td><td>Snow100K-L Snow100K-S</td><td></td><td>Outdoor</td><td>Raindrop</td></tr><tr><td rowspan="4">PPS</td><td rowspan="4">WeatherDiff [45] Histoformer [19] MODEM [21]</td><td>0.0982</td><td>0.0541</td><td>0.0887</td><td>0.0615</td></tr><tr><td>0.0919</td><td>0.0445</td><td>0.0778</td><td>0.0672</td></tr><tr><td>0.0880</td><td>0.0407</td><td>0.0699</td><td>0.0650</td></tr><tr><td>0.0799</td><td>0.0366</td><td>0.0650</td><td>0.0610</td></tr><tr><td rowspan="4">-Ain Ours</td><td>WeatherDiff [45]</td><td>3.4531</td><td>3.5293</td><td>3.8691</td><td>4.0000</td></tr><tr><td rowspan="3">Histoformer [19] MODEM [21]</td><td>3.7207</td><td>3.7598</td><td>4.1445</td><td>4.0156</td></tr><tr><td>3.7324</td><td>3.7695</td><td>4.1875</td><td>4.0664</td></tr><tr><td>4.0407</td><td>4.0177</td><td>4.4015</td><td>4.2514</td></tr><tr><td rowspan="4">MU</td><td rowspan="4">WeatherDiff [45] Histoformer [19] MODEM [21] Ours</td><td>62.6267</td><td>63.1278</td><td>67.4814</td><td>69.3608</td></tr><tr><td>64.2526</td><td>64.2581</td><td>67.7461</td><td>68.4852</td></tr><tr><td>64.2438</td><td>64.2853</td><td>68.2926</td><td>69.7925</td></tr><tr><td>66.2562</td><td>66.3938</td><td>70.9819</td><td>71.4844</td></tr></table>

TABLE V: Ablation study on the proposed components.
<table><tr><td rowspan="2">Exp.</td><td colspan="4">Factors</td><td colspan="2">Avg.</td></tr><tr><td>GDTB</td><td>BMSC</td><td>BAE</td><td>URH</td><td>PSNR</td><td>SSIM</td></tr><tr><td>1</td><td>×</td><td>×</td><td>×</td><td>×</td><td>33.76</td><td>0.9447</td></tr><tr><td>2</td><td>√</td><td>×</td><td>×</td><td>×</td><td>33.85</td><td>0.9453</td></tr><tr><td>3</td><td>√</td><td>√</td><td>×</td><td>×</td><td>34.00</td><td>0.9465</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>×</td><td>34.21</td><td>0.9480</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>34.46</td><td>0.9500</td></tr></table>

## B. Ablation Studies

We present ablation studies on the proposed components and several key design choices of the framework.

![](images/132aa11829a6d94d70fbaeeeafb6e490e79678a20b12f8a1eb6501b2c8a61b14.jpg)

![](images/6e55737e3a9e25f9d0eb4ee57a59135afc5618789dae66b8889ce2e667e45390.jpg)  
Fig. 8: T-SNE visualization of features learned by MO-DEM [21] (left) and our method (right). Compared with MODEM, our method yields more compact intra-class feature clusters and clearer inter-class separation among different adverse-weather conditions, indicating improved conditionaware feature disentanglement and representation robustness.

a) Ablation study on the proposed components: The results are summarized in Table V. Starting from the histogramtransformer baseline [19], replacing it with GDTB improves the average performance by +0.09 dB PSNR and +0.0006 SSIM. Adding BMSC further increases the performance to 34.00 dB PSNR and 0.9465 SSIM. Introducing BAE-Loss brings additional gains, reaching 34.21 dB PSNR and 0.9480 SSIM. Finally, incorporating URH yields the full model with 34.46 dB PSNR and 0.9500 SSIM, outperforming the baseline by about +0.70 dB PSNR and +0.0053 SSIM. These results demonstrate that GDTB, BMSC, BAE-Loss, and URH contribute positively and complement each other.

TABLE VI: Controlled ablations on sinusoidal reweighting and gating. S-S: Snow100K-S, S-L: Snow100K-L.

(a) Sinusoidal reweighting (with URH & BMSC)
<table><tr><td rowspan="2">Rew.</td><td colspan="2">S-S</td><td colspan="2">S-L</td><td colspan="2">Outdoor-Rain</td><td colspan="2">Raindrop</td></tr><tr><td>P</td><td>S</td><td>P</td><td>S</td><td>P</td><td>S</td><td>P</td><td>S</td></tr><tr><td>X</td><td>37.72</td><td>.9671</td><td>32.21</td><td>.9268</td><td>32.39</td><td>.9420</td><td>32.73</td><td>.9430</td></tr><tr><td>√</td><td>37.85</td><td>.9675</td><td>32.39</td><td>.9282</td><td>32.33</td><td>.9416</td><td>32.68</td><td>.9431</td></tr></table>

(b) Gating mechanism (with URH)
<table><tr><td rowspan="2">Gate</td><td colspan="2">S-S</td><td colspan="2">S-L</td><td colspan="2">Outdoor-Rain</td><td colspan="2">Raindrop</td></tr><tr><td>P</td><td>S</td><td>P</td><td>S</td><td>P</td><td>S</td><td>P</td><td>S</td></tr><tr><td>×</td><td>38.18</td><td>.9688</td><td>32.63</td><td>.9307</td><td>32.87</td><td>.9454</td><td>33.06</td><td>.9455</td></tr><tr><td>√</td><td>38.34</td><td>.9697</td><td>32.77</td><td>.9324</td><td>33.40</td><td>.9491</td><td>33.32</td><td>.9487</td></tr></table>

TABLE VII: Ablation on the integration and refinement modules inside BMSC.
<table><tr><td rowspan="2">Exp.</td><td colspan="2">Factors</td><td colspan="2">Avg.</td></tr><tr><td>Integration</td><td>Refinement</td><td>PSNR</td><td>SSIM</td></tr><tr><td>1</td><td>Avg.</td><td>vHeat</td><td>33.82</td><td>0.9455</td></tr><tr><td>2</td><td>LMF</td><td>CosFormer</td><td>33.93</td><td>0.9459</td></tr><tr><td>3</td><td>LMF</td><td>Nonlocal</td><td>33.94</td><td>0.9461</td></tr><tr><td>4</td><td>LMF</td><td>vHeat</td><td>34.00</td><td>0.9465</td></tr></table>

b) Ablation on sinusoidal reweighting and gating: Table VI presents controlled ablations on sinusoidal feature reweighting and the gating mechanism under their respective settings. With URH and BMSC enabled (Table VI(a)), sinusoidal reweighting yields consistent but modest improvements, providing about +0.05 dB PSNR gain on average, which indicates its effectiveness in enhancing locality-aware feature interactions. In contrast, when evaluated with URH enabled (Table VI(b)), the gating mechanism leads to more substantial gains, improving the average PSNR by about +0.27 dB along with consistent SSIM improvements, highlighting its critical role in adaptively modulating feature responses under diverse weather degradations. Based on these observations, both sinusoidal reweighting and gating are adopted in the final model.

c) Ablation on integration and refinement inside BMSC: Table VII shows an ablation on the integration and refinement choices inside BMSC, again under a fixed setting without URH and BAE-Loss. The results indicate that using linear multistep fusion (LMF) instead of simple averaging leads to better average performance: with the same vHeat refinement, LMF achieves about +0.18 dB higher PSNR and a small SSIM gain. Under LMF, vHeat also performs slightly better than CosFormer [36] and Nonlocal [64]. Based on these observations, we adopt LMF for integration and vHeat for refinement in our model.

d) Balanced feature size in BMSC: In the BMSC, the balanced feature is the intermediate resolution used by the linear multistep fusion to combine multi-scale encoder features. As shown in Table VIII, we test three choices for this resolution, H × W, $H / 2 \times W / 2$ , and $H / 4 \times W / 4$ , under a simplified setting without URH and BAE-Loss (trained with $\ell _ { 1 }$ loss). All three options give very similar average PSNR and SSIM, but $H / 4 \times W / 4$ slightly outperforms the others and is also cheaper to compute because of the lower spatial size. Therefore, we use $H / 4 \times W / 4$ as the default balanced feature size in all subsequent experiments.

TABLE VIII: Ablation on the balanced feature size (without URH and BAE-Loss, trained with $\ell _ { 1 }$ loss).
<table><tr><td>Exp.</td><td>Balanced Size</td><td>PSNR</td><td>SSIM</td></tr><tr><td>1</td><td> $H \times W$ </td><td>33.90</td><td>0.9458</td></tr><tr><td>2</td><td> $H / 2 \times W / 2$ </td><td>33.90</td><td>0.9457</td></tr><tr><td>3</td><td> $H / 4 \times W / 4$ </td><td>34.00</td><td>0.9465</td></tr></table>

TABLE IX: Ablation on the annealing step of BAE-Loss.
<table><tr><td>Exp.</td><td>Annealing Step</td><td>PSNR</td><td>SSIM</td></tr><tr><td>1</td><td>230k</td><td>34.12</td><td>0.9469</td></tr><tr><td>2</td><td>250k</td><td>34.14</td><td>0.9471</td></tr><tr><td>3</td><td>270k</td><td>34.17</td><td>0.9478</td></tr><tr><td>4</td><td>300k</td><td>34.21</td><td>0.9480</td></tr></table>

e) The impact of the annealing step in BAE-Loss: According to Table IX, increasing the annealing step from 230k to 300k slightly but consistently improves the average PSNR/SSIM (from 34.16/0.9476 to 34.21/0.9480). This shows that removing the regression loss too early hurts performance, and it is better to keep it almost throughout training. Hence, we set the annealing step to 300k in all subsequent experiments.

## C. Complexity Analysis

Table X reports the computational complexity and average restoration performance of representative unified methods. UAR-Net achieves the highest average PSNR among all compared models, outperforming MODEM and Histoformer under the same evaluation setting. Although the compared methods differ in computational cost, UAR-Net consistently delivers superior restoration quality, indicating stronger representation capacity for unified adverse-weather restoration. As further illustrated in Fig. 9, UAR-Net lies on a more favorable accuracy–complexity trade-off curve, achieving higher restoration accuracy under comparable computational budgets.

![](images/43cda6b6dbdf150efce5eaf1d5efe938381459b664f5b21b0a4c3608d0483618.jpg)  
Fig. 9: Accuracy–complexity trade-off (Avg. PSNR vs. FLOPs) on 128 × 128 input patches.

TABLE X: Model complexity and average restoration performance on a single H100 GPU with 128 × 128 input patches.
<table><tr><td>Method</td><td>Histoformer [19]</td><td>MODEM [21]</td><td>Ours</td></tr><tr><td>FLOPs (GMacs)</td><td>23.27</td><td>29.08</td><td>35.44</td></tr><tr><td>Avg. PSNR (dB)</td><td>33.67</td><td>34.18</td><td>34.43</td></tr></table>

## VI. CONCLUSION

In this paper, we proposed UAR-Net, an Uncertainty-guided Adverse- weather Restoration Network. It integrates GDTB, BMS, and URH to better handle multi-scale structures and residual artifacts, and employs BAE-Loss to jointly learn accurate reconstructions and pixel-wise uncertainty. Extensive experiments demonstrate SOTA PSNR/SSIM across multiple adverse-weather benchmarks.

## REFERENCES

[1] V. Muat, I. Fursa, P. Newman, F. Cuzzolin, and A. Bradley, “Multiweather city: Adverse weather stacking for autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 2906–2915. 1

[2] H. Cao, G. Chen, J. Xia, G. Zhuang, and A. Knoll, “Fusion-based feature attention gate component for vehicle detection based on event camera,” IEEE Sensors Journal, vol. 21, no. 21, pp. 24 540–24 548, 2021. 1

[3] Y. Almalioglu, M. Turan, N. Trigoni, and A. Markham, “Deep learningbased robust positioning for all-weather autonomous driving,” Nature machine intelligence, vol. 4, no. 9, pp. 749–760, 2022. 1

[4] R. Li, R. T. Tan, and L.-F. Cheong, “All in one bad weather removal using architectural search,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2020, pp. 3175– 3185. 1, 2, 9, 10

[5] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” Advances in neural information processing systems, vol. 28, 2015. 1

[6] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European conference on computer vision (ECCV). Springer, 2020, pp. 213–229. 1

[7] M. Hong, S. Cheng, H. Huang, H. Fan, and S. Liu, “You only look around: Learning illumination-invariant feature for low-light object detection,” Advances in Neural Information Processing Systems, vol. 37, pp. 87 136–87 158, 2024. 1

[8] J. Iqbal, R. Hafiz, and M. Ali, “Fogadapt: Self-supervised domain adaptation for semantic segmentation of foggy images,” Neurocomputing, vol. 501, pp. 844–856, 2022. 1

[9] K. He, J. Sun, and X. Tang, “Single image haze removal using dark channel prior,” IEEE transactions on pattern analysis and machine intelligence, vol. 33, no. 12, pp. 2341–2353, 2010. 1

[10] C. O. Ancuti and C. Ancuti, “Single image dehazing by multi-scale fusion,” IEEE Transactions on Image Processing, vol. 22, no. 8, pp. 3271–3282, 2013. 1

[11] D. Berman, S. Avidan et al., “Non-local image dehazing,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2016, pp. 1674–1682. 1

[12] J. Chen, C.-H. Tan, J. Hou, L.-P. Chau, and H. Li, “Robust video content alignment and compensation for rain removal in a cnn framework,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018, pp. 6286–6295. 1

[13] K. Jiang, Z. Wang, P. Yi, C. Chen, B. Huang, Y. Luo, J. Ma, and J. Jiang, “Multi-scale progressive fusion network for single image deraining,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2020, pp. 8346–8355. 1

[14] L. Li, Y. Dong, W. Ren, J. Pan, C. Gao, N. Sang, and M.-H. Yang, “Semisupervised image dehazing,” IEEE Transactions on Image Processing, vol. 29, pp. 2766–2779, 2019. 1

[15] H. Wu, Y. Qu, S. Lin, J. Zhou, R. Qiao, Z. Zhang, Y. Xie, and L. Ma, “Contrastive learning for compact single image dehazing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2021, pp. 10 551–10 560. 1

[16] K. Zhang, R. Li, Y. Yu, W. Luo, and C. Li, “Deep dense multi-scale network for snow removal using semantic and depth priors,” IEEE Transactions on Image Processing, vol. 30, pp. 7419–7431, 2021. 1, 2, 10

[17] L. Chen, X. Chu, X. Zhang, and J. Sun, “Simple baselines for image restoration,” in European conference on computer vision (ECCV). Springer, 2022, pp. 17–33. 1, 10

[18] C. Mou, Q. Wang, and J. Zhang, “Deep generalized unfolding networks for image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2022, pp. 17 399– 17 410. 1

[19] S. Sun, W. Ren, X. Gao, R. Wang, and X. Cao, “Restoring images in adverse weather conditions via histogram transformer,” in European Conference on Computer Vision (ECCV). Springer, 2024, pp. 111–129. 1, 2, 4, 7, 9, 10, 12

[20] R. Zhu, Z. Tu, J. Liu, A. C. Bovik, and Y. Fan, “Mwformer: Multiweather image restoration using degradation-aware transformers,” IEEE Transactions on Image Processing, 2024. 1, 2

[21] H. Wang, Q. Hu, and X. Guo, “MODEM: A morton-order degradation estimation mechanism for adverse weather image recovery,” in Conference on Neural Information Processing Systems (NeurIPS), 2025. 1, 2, 7, 9, 10, 12

[22] J. Pang, K. Chen, J. Shi, H. Feng, W. Ouyang, and D. Lin, “Libra rcnn: Towards balanced learning for object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019, pp. 821–830. 2

[23] H. Cao, G. Chen, H. Zhao, D. Jiang, X. Zhang, Q. Tian, and A. Knoll, “Sdpt: Semantic-aware dimension-pooling transformer for image segmentation,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, pp. 15 934–15 946, 2024. 2

[24] J. Wu, Z. Yang, Z. Wang, and Z. Jin, “Beyond degradation conditions: All-in-one image restoration via hog transformers,” in Association for the Advancement of Artificial Intelligence Conference on Artificial Intelligence (AAAI), 2026. 2, 7, 9, 10

[25] X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding, and J. Paisley, “Removing rain from single images via a deep detail network,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2017, pp. 3855–3863. 2

[26] X. Li, J. Wu, Z. Lin, H. Liu, and H. Zha, “Recurrent squeezeand-excitation context aggregation net for single image deraining,” in European conference on computer vision (ECCV), 2018, pp. 254–269. 2, 10

[27] R. Yasarla and V. M. Patel, “Uncertainty guided multi-scale residual learning-using a cycle spinning cnn for single image de-raining,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019, pp. 8405–8414. 2

[28] X. Chen, H. Li, M. Li, and J. Pan, “Learning a sparse transformer network for effective image deraining,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2023, pp. 5896–5905. 2

[29] Y.-F. Liu, D.-W. Jaw, S.-C. Huang, and J.-N. Hwang, “Desnownet: Context-aware deep network for snow removal,” IEEE Transactions on Image Processing, vol. 27, no. 6, pp. 3064–3073, 2018. 2, 8, 9, 10

[30] P. Li, M. Yun, J. Tian, Y. Tang, G. Wang, and C. Wu, “Stacked dense networks for single-image snow removal,” Neurocomputing, vol. 367, pp. 152–163, 2019. 2

[31] R. Qian, R. T. Tan, W. Yang, J. Su, and J. Liu, “Attentive generative adversarial network for raindrop removal from a single image,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018, pp. 2482–2491. 2, 8, 9, 10

[32] X. Liu, M. Suganuma, Z. Sun, and T. Okatani, “Dual residual networks leveraging the potential of paired operations for image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019, pp. 7007–7016. 2, 10

[33] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li, “Maxim: Multi-axis mlp for image processing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2022, pp. 5769–5780. 2, 10

[34] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Transformers are rnns: Fast autoregressive transformers with linear attention,” in International conference on machine learning (ICML). PMLR, 2020, pp. 5156–5165. 3

[35] Y. Ai, H. Huang, T. Wu, Q. Fan, and R. He, “Breaking complexity barriers: High-resolution image restoration with rank enhanced linear attention,” arXiv preprint arXiv:2505.16157, 2025. 4

[36] Z. Qin, W. Sun, H. Deng, D. Li, Y. Wei, B. Lv, J. Yan, L. Kong, and Y. Zhong, “cosformer: Rethinking softmax in attention,” in International Conference on Learning Representations (ICLR), 2022. 4, 11

[37] Z. Qiu, Z. Wang, B. Zheng, Z. Huang, K. Wen, S. Yang, R. Men, L. Yu, F. Huang, S. Huang et al., “Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free,” Conference on Neural Information Processing Systems (NeurIPS), 2025. 4

[38] Q. He, X. Min, K. Wang, and T. He, “Fuseunet: A multi-scale feature fusion method for u-like networks,” in International Conference on Machine Learning (ICML), 2025. 5

[39] Z. Wang, Y. Liu, Y. Tian, Y. Liu, Y. Wang, and Q. Ye, “Building vision models upon heat conduction,” in Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025, pp. 9707–9717. 6

[40] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2022, pp. 5728–5739. 7, 9, 10

[41] X. Qin, Z. Zhang, C. Huang, C. Gao, M. Dehghan, and M. Jagersand, “Basnet: Boundary-aware salient object detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019, pp. 7479–7489. 7

[42] Y. Cheng, A. Knoll, and H. Cao, “Urnet: uncertainty-aware refinement network for event-based stereo depth estimation,” Visual Intelligence, vol. 3, no. 1, p. 18, 2025. 7

[43] T. Gneiting, L. I. Stanberry, E. P. Grimit, L. Held, and N. A. Johnson, “Assessing probabilistic forecasts of multivariate quantities, with an application to ensemble predictions of surface winds,” Test, vol. 17, no. 2, pp. 211–235, 2008. 7

[44] J. Liao, S. Hao, R. Hong, and M. Wang, “Gt-mean loss: A simple yet effective solution for brightness mismatch in low-light image enhancement,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 6112–6121. 7

[45] O. Ozdenizci and R. Legenstein, “Restoring vision in adverse weather<sup>¨</sup> conditions with patch-based denoising diffusion models,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 8, pp. 10 346–10 357, 2023. 8, 9, 10

[46] R. Li, L.-F. Cheong, and R. T. Tan, “Heavy rain image restoration: Integrating physics model and conditional adversarial learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019, pp. 1633–1642. 8, 9, 10

[47] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018, pp. 586–595. 9

[48] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun et al., “Q-align: Teaching lmms for visual scoring via discrete text-defined levels,” arXiv preprint arXiv:2312.17090, 2023. 9

[49] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “Musiq: Multiscale image quality transformer,” in Proceedings of the IEEE/CVF international conference on computer vision (ICCV), 2021, pp. 5148– 5157. 9

[50] J. M. J. Valanarasu, R. Yasarla, and V. M. Patel, “Transweather: Transformer-based restoration of images degraded by adverse weather conditions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2022, pp. 2353–2363. 9, 10

[51] W.-T. Chen, Z.-K. Huang, C.-C. Tsai, H.-H. Yang, J.-J. Ding, and S.-Y. Kuo, “Learning multiple adverse weather removal via two-stage knowledge learning and multi-contrastive regularization: Toward a unified model,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2022, pp. 17 653–17 662. 9, 10

[52] Y. Zhu, T. Wang, X. Fu, X. Yang, X. Guo, J. Dai, Y. Qiao, and X. Hu, “Learning weather-general and weather-specific features for image restoration under multiple adverse weather conditions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2023, pp. 21 747–21 758. 9, 10

[53] V. Potlapalli, S. W. Zamir, S. Khan, and F. Khan, “Promptir: Prompting for all-in-one image restoration,” in Conference on Neural Information Processing Systems (NeurIPS), 2023. 10

[54] D. Zheng, X.-M. Wu, S. Yang, J. Zhang, J.-F. Hu, and W.-s. Zheng, “Selective hourglass mapping for universal image restoration based on diffusion model,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 10

[55] T. Wang, X. Yang, K. Xu, S. Chen, Q. Zhang, and R. W. Lau, “Spatial attentive single-image deraining with a high quality real rain dataset,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2019, pp. 12 270–12 279. 10

[56] W.-T. Chen, H.-Y. Fang, J.-J. Ding, C.-C. Tsai, and S.-Y. Kuo, “Jstasr: Joint size and transparency-aware snow removal algorithm based on modified partial convolution and veiling effect removal,” in European conference on computer vision (ECCV). Springer, 2020, pp. 754–770. 10

[57] Y. Cui, W. Ren, X. Cao, and A. Knoll, “Revitalizing convolutional network for image restoration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9423–9438, 2024. 10

[58] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision (ICCV), 2017, pp. 2223–2232. 10

[59] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2017, pp. 1125–1134. 10

[60] K. Jiang, Z. Wang, P. Yi, C. Chen, Z. Wang, X. Wang, J. Jiang, and C.-W. Lin, “Rain-free and residue hand-in-hand: A progressive coupled network for real-time image deraining,” IEEE Transactions on Image Processing, vol. 30, pp. 7404–7418, 2021. 10

[61] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, “Multi-stage progressive image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2021, pp. 14 821–14 831. 10

[62] Y. Quan, S. Deng, Y. Chen, and H. Ji, “Deep learning for seeing through window with raindrops,” in Proceedings of the IEEE/CVF international conference on computer vision (ICCV), 2019, pp. 2463–2471. 10

[63] S. Zhou, D. Chen, J. Pan, J. Shi, and J. Yang, “Adapt or perish: Adaptive sparse transformer with attentive feature refinement for image restoration,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), 2024, pp. 2952–2963. 10

[64] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018, pp. 7794–7803. 11