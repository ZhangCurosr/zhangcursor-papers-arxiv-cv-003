# On the Necessity of Attention–FFN Split in Vision Transformers

Junhyeok Kim <sup>1</sup> Jinyeong Kim <sup>1</sup> Jae Wan Park <sup>1</sup> Seong Jae Hwang <sup>1</sup>

## Abstract

The standard Transformer architecture relies on a rigid pattern that alternates Attention and Feed-Forward Network (FFN) layers. Despite its widespread adoption, the inductive bias imposed by this strict separation has not been systematically examined. In this work, we investigate the necessity of the Attention–FFN dichotomy in Vision Transformers (ViTs). To facilitate this analysis, we introduce the AttenFeed module, a unified component that integrates the functional properties of both Attention and FFN. Based on this module, we devise the unified Vision Transformer (uViT), which replaces the conventional alternating Attention–FFN structure with a sequence of AttenFeed modules. We then use uViT as a control group that relaxes the Attention–FFN dichotomy of the standard ViT and systematically compare the two models across multiple datasets and model scales. Our experiments reveal that the Attention–FFN dichotomy can hinder performance at smaller model scales due to the rigid parameter allocation of ViTs. The AttenFeed module and uViT serve as new analytical tools for understanding the Attention–FFN structure and offer theoretical insights into the heuristically designed architecture of conventional ViTs.

## 1. Introduction

The Transformer architecture (Vaswani et al., 2017) has demonstrated exceptional performance across diverse domains, establishing itself as a universal architecture (Devlin et al., 2019; Achiam et al., 2023; Dosovitskiy, 2020; Radford et al., 2021; Peebles & Xie, 2023; Baevski et al., 2020; Jumper et al., 2021). Although originally developed for machine translation, it is highly encouraging that the Transformer exhibits robust performance even in disparate fields. In particular, Vision Transformer (ViT) (Dosovitskiy, 2020)

has achieved results surpassing those of Convolutional Neural Networks (CNNs) (LeCun et al., 1998; Krizhevsky et al., 2012; He et al., 2016), which had long served as the de facto standard in the computer vision domain.

This success of ViT is largely attributed to the attention mechanism. Unlike CNNs, which structurally enforce a locality inductive bias (Scherer et al., 2010; Schmidhuber, 2015; Bengio et al., 2017), attention-based models are required to learn this property directly from data. While this absence of structural constraints poses an additional learning challenge, the lack of bias grants the model the freedom to learn superior mechanisms from scratch. Consequently, ViTs not only outperform CNNs in capturing long-range dependencies (d’Ascoli et al., 2022; Edelman et al., 2022b) but also exhibit advantageous emergent properties such as register tokens (Darcet et al., 2023) that are absent in traditional CNNs.

As inductive bias has played a pivotal role in the advancement of Transformers, some research have been dedicated to understanding the biases intrinsic to the architecture. Briefly, it has been observed that Transformers exhibit a structural bias toward symmetric functions (Lavie et al., 2024) and tend to concentrate attention on a sparse subset of tokens (Edelman et al., 2022a). Despite these findings, the inductive bias imposed by the most fundamental structural choice of the Transformer, alternation of Attention and Feed-Forward Network (FFN), remains largely unexplored.

Moreover, recent studies on Mixture-of-Experts (MoE) provide evidence that this “baked-in” alternating structure of Attention and FFN may not be indispensable (Luo et al., 2025; Zhao et al., 2025; He et al., 2025; 2024). For instance, SkipGPT (Zhao et al., 2025) trained a router to conditionally drop attention or FFN blocks, revealing that consecutive interactions (e.g., Attention-Attention or FFN-FFN) are occasionally more effective than the standard alternating pattern. These studies motivate an investigation into when the Attention–FFN structure in Transformers is necessary and when it may be redundant.

Given these observations, understanding the inductive bias imposed by the repeating Attention–FFN structure of the Transformer becomes crucial. Therefore, we aim to analyze this “Attention–FFN dichotomy” in ViT. To achieve this, (i) we formulate the AttenFeed module, a single component capable of exhibiting the properties of both Attention and FFN. (ii) Then we build the unified Vision Transformer (uViT) composed of a sequence of AttenFeed modules, thereby removing the strict Attention–FFN dichotomy. (iii) Lastly, we compare uViT with ViT to analyze under what circumstances and to what extent the conventional Attention–FFN dichotomy facilitates the learning process.

![](images/cbc06f59c5304b1fcc828e96f661187cd084e68249474c278a9747b03ed2c329.jpg)  
(a) standard Transformer

![](images/a13f1042cd97e0a10d6a6cd54ddd28a9f3fef45f414caa7974176e550a637dd2.jpg)  
(b) unified Transformer  
Figure 1. Architecture illustrations. (a) shows a layer of the standard Transformer, and (b) shows a layer of the unified Transformer, a control architecture for studying the Attention–FFN dichotomy.

To formulate the AttenFeed module, we start from the interpretation of UMoE (Yang et al., 2025) that views the standard Attention as an FFN with no activation function. By incorporating a non-linear activation function into standard attention, we enable its interpretation as a conventional FFN. In Section 4.1, we mathematically demonstrate that the AttenFeed module is a generalized form of both Attention and FFN. Then, in Sections 4.2 to 4.4, we experimentally show that the AttenFeed module simultaneously possesses the properties of both Attention and FFN. These results confirm that AttenFeed relaxes the Attention–FFN dichotomy, subsuming the functionalities of both.

Next, we design uViT, an architecture composed of a sequence of AttenFeed modules (Figure 1b). Since the Atten-Feed module relaxes the structural inductive bias imposed by the Attention–FFN separation in conventional ViTs, we hypothesize that uViT exhibits weaker inductive bias than standard ViTs. We empirically validate this conjecture in Section 5.1. Consequently, uViT serves as a control group for investigating the Attention–FFN dichotomy of ViTs, as it subsumes the functional properties of both Attention and FFN while relaxing the strict structural constraints.

Finally, we pretrain both ViT and uViT on various datasets and compare their performance in Section 5.2. The results show that while uViT significantly outperforms ViT at smaller model scales, the performance gap between the two architectures narrows and becomes marginal as the model capacity increases. The experiments in Section 5 confirm that the inductive bias imposed by the Attention–FFN dichotomy in standard ViTs actually hinders performance at smaller scales.

The significance of our proposed AttenFeed module extends beyond the mere integration of standard Attention and FFN. Rather, it aims to open a new avenue for theoretically and empirically investigating the heuristically established alternating Attention–FFN structure of Transformers (Wang et al., 2025; Yu et al., 2023). We anticipate that the AttenFeed module and uViT will facilitate a more systematic investigation into the architectural inductive biases that have long underpinned the success of ViT.

Our main contributions are summarized as follows:

◦ Formulation of the AttenFeed Module. We propose the AttenFeed module, a unified component mathematically and empirically proven to generalize the properties of both standard Attention and FFNs.

◦ Introduction of uViT. We design the unified Vision Transformer (uViT) composed entirely of AttenFeed modules, effectively removing the strict Attention– FFN dichotomy of standard Transformers.

◦ Analysis of Inductive Bias at Scale. Through experiments using uViT, we reveal that the conventional Attention–FFN dichotomy acts as a restrictive inductive bias that impedes effective learning at smaller model scales.

## 2. Related Work

Inductive bias serves as a fundamental architectural constraint that guides a model toward a specific solution space. While these biases often accelerate early-stage training, they can arguably act as a bottleneck, limiting the performance ceiling when data becomes sufficiently large (Bachmann et al., 2023; Dosovitskiy, 2020; Neyshabur et al., 2017; Sutton, 2019). For instance, the Vision Transformer (ViT) (Dosovitskiy, 2020) relaxes the image-specific inductive biases inherent in Convolutional Neural Networks (CNNs), such as translation equivariance and locality (Scherer et al., 2010; Schmidhuber, 2015; Bengio et al., 2017).

As inductive bias has emerged as a key concept in understanding Transformer architectures, a growing body of literature has sought to characterize the specific biases that underpin Transformer performance. Transformers exhibit a structural bias toward symmetric functions (Lavie et al., 2024) and tend to concentrate attention on a sparse subset of tokens (Edelman et al., 2022a). From a task-specific perspective, the pairwise computation of self-attention inherently favors tasks requiring relational reasoning (Sanford et al., 2023). Furthermore, compared to State-Space Models (SSMs) (Gu & Dao, 2024), the structural bias of Transformers makes them more suitable for the copy task (Jelassi et al., 2024).

Building upon these insights, subsequent research has focused on actively manipulating these biases to enhance training efficiency and performance (Xu et al., 2021; Xiao et al., 2021; d’Ascoli et al., 2022). For instance, Swin Transformer (Liu et al., 2021) reintroduces locality and translation invariance through a hierarchical architecture and shifted window attention mechanism. Conversely, MLP-Mixer (Tolstikhin et al., 2021) and ResMLP (Touvron et al., 2023) minimize the inductive bias of token mixing by replacing self-attention with vanilla MLPs, demonstrating superior performance over standard Transformers on larger datasets. Beyond structural modifications, several studies (Peruzzo et al., 2024; An et al., 2025) propose methods for injecting inductive biases during the training phase to enable more efficient training.

Despite these advancements, to the best of our knowledge, there is no research exploring the necessity of the rigid functional separation between Attention and FFN. While GAU (Hua et al., 2022) attempts to merge GLU and Attention, it was primarily designed to compensate for the performance limitations of linear attention rather than to minimize inductive bias. We propose a unified module that generalizes both components, thereby enabling, for the first time, a systematic investigation of the Attention–FFN dichotomy.

## 3. Method

The primary objective of this paper is to investigate when the inductive bias imposed by Attention–FFN structure becomes beneficial. To facilitate this investigation, we devise an architecture designed to relax this structural separation. Specifically, in Section 3.1, we briefly review the conventional Vision Transformer (ViT) block. Next, in Section 3.2, we introduce the AttenFeed module, a component that conceptually unifies Attention and FFN. Finally, in Section 3.3, we present the overall unified Vision Transformer (uViT) architecture, designed to relax the explicit decoupling of Attention and FFN in ViT.

## 3.1. Preliminary

A conventional ViT takes a [CLS] token and T image patch tokens as input. Let $\mathbf { X } ^ { \ell } \in \mathbb { R } ^ { ( 1 + T ) \times d }$ denote the input token sequence for layer $\ell \in \{ 1 , \ldots , L \}$ , and $\mathbf { x } _ { i } ^ { \ell } \in \mathbb { R } ^ { d }$ represent the $i ^ { \mathrm { { t h } } }$ token in the sequence. A standard ViT block $( { \mathrm { F i g } } .$ ure 1a) comprises an alternating structure of multi-head Attention and FFNs, formulated as follows (Dosovitskiy, 2020; Ferrando et al., 2024):

$$
\mathbf { x } _ { i } ^ { \mathrm { m i d } , \ell } = \mathbf { x } _ { i } ^ { \ell - 1 } + \sum _ { h } \mathrm { A t t n } ^ { \ell , h } ( \mathrm { L N } _ { 1 } ^ { \ell } ( \mathbf { X } ^ { \ell - 1 } ) ) _ { i } ,\tag{1}
$$

$$
\mathbf { x } _ { i } ^ { \ell } = \mathbf { x } _ { i } ^ { \mathrm { m i d } , \ell } + \mathrm { F F N } ^ { \ell } ( \mathrm { L N } _ { 2 } ^ { \ell } ( \mathbf { x } _ { i } ^ { \mathrm { m i d } , \ell } ) ) ,
$$

where $\mathrm { L N } _ { 1 } ^ { \ell }$ and $\mathrm { L N _ { 2 } ^ { \ell } }$ are layer normalizations.

The attention head in a Transformer is typically formulated $\mathrm { a s } ^ { 1 } \mathrm { : }$

$$
\mathrm { A t t n } ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } ) _ { i } = \sum _ { j } a _ { i , j } ^ { \ell , h } { \mathbf { x } } _ { j } ^ { \ell - 1 } { \mathbf { W } } _ { V } ^ { \ell , h } { \mathbf { W } } _ { O } ^ { \ell , h } ,\tag{2}
$$

where $\mathbf { W } _ { V } ^ { \ell , h } \ \in \ \mathbb { R } ^ { d \times d _ { h } }$ and $\mathbf { W } _ { O } ^ { \ell , h } ~ \in ~ \mathbb { R } ^ { d _ { h } \times d }$ , and $\textit { h } \in$ $\{ 1 , \ldots , H \}$ denotes the head index. Following the terminology of (Elhage et al., 2021), we refer to the computation of the attention weights $a _ { i , j } ^ { \ell , h }$ as the QK circuit, and the transformation $\mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \mathbf { W } _ { O } ^ { \tilde { \ell } , h }$ as the OV circuit<sup>2</sup>.

The Attention layer is followed by a position-wise FFN:

$$
\mathrm { F F N } ^ { \ell } ( \mathbf { x } _ { i } ^ { \mathrm { m i d } , \ell } ) = \sigma ( \mathbf { x } _ { i } ^ { \mathrm { m i d } , \ell } \mathbf { W } _ { \mathrm { i n } } ^ { \ell } ) \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } ,\tag{3}
$$

where $\sigma$ is a non-linear activation function, and $\mathbf { W } _ { \mathrm { i n } } ^ { \ell } \in$ $\mathbb { R } ^ { d \times d _ { \mathrm { F F N } } } , \mathbf { W } _ { \mathrm { o u t } } ^ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { F F N } } \times d }$

## 3.2. AttenFeed Module

UMoE (Yang et al., 2025) interprets the OV circuit in Equation (2) as an FFN without an activation function. Inspired by this interpretation, we apply minimal modifications to the standard attention mechanism to endow it with FFN-like behavior. Specifically, we simply introduce a non-linear activation function σ into the value projection to formulate the AttenFeed module as follows:

$$
\mathrm { A t t e n F e e d } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } : = \sum _ { j } a _ { i , j } ^ { \ell , h } \sigma ( \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } ) \mathbf { W } _ { O } ^ { \ell , h } .\tag{4}
$$

Unless otherwise mentioned, we use GELU (Hendrycks, 2016) for σ.

This modification now allows the OV circuit to be interpreted as a standard FFN by explicitly incorporating a nonlinear activation. At the same time, as this requires minimal alterations to the conventional attention mechanism, we expect the module to retain the intrinsic properties of the standard attention. In other words, while $\mathbf { W } _ { V }$ and $\mathbf { W } _ { O }$ can be interpreted as the value and output matrices of standard attention, respectively, the module can also be viewed as a standard FFN by redefining them as $\mathbf { W } _ { V } \ : = \ \mathbf { W } _ { \mathrm { i n } }$ and $\mathbf { W } _ { O } : = \mathbf { W } _ { \mathrm { o u t } }$ given in Equation (3). We show in Section 4.1 that the AttenFeed module serves as a mathematical generalization of Attention and FFN, and further empirically validate this finding in Sections 4.2 to 4.4.

## 3.3. Unified Vision Transformer

Based on the AttenFeed module, we propose the unified Vision Transformer (uViT) (Figure 1a), where the update

Table 1. Configurations of the ViT and uViT models. $L , H$ and D denote the number of layers, heads, and model dimension, respectively.
<table><tr><td>Name</td><td>Params.</td><td> $( L , H , D )$ </td><td>MLPs</td></tr><tr><td>ViT-S</td><td>21M</td><td>(12, 6, 384)</td><td>√</td></tr><tr><td>uViT-S</td><td>20M</td><td>(12, 8, 640)</td><td>X</td></tr><tr><td>ViT-B</td><td>86M</td><td>(12, 12, 768)</td><td>√</td></tr><tr><td>uViT-B</td><td>80M</td><td>(12, 16, 1280)</td><td>X</td></tr><tr><td>ViT-L</td><td>303M</td><td>(24, 16, 1024)</td><td>V</td></tr><tr><td>uViT-L</td><td>289M</td><td>(28, 20, 1600)</td><td>X</td></tr></table>

rule is given by:

$$
\mathbf { x } _ { i } ^ { \ell } = \mathbf { x } _ { i } ^ { \ell - 1 } + \sum _ { h } \mathrm { A t t e n F e e d } ^ { \ell , h } ( \mathrm { L N } ^ { \ell } ( \mathbf { X } ^ { \ell - 1 } ) ) _ { i } .\tag{5}
$$

Thus, uViT replaces the coupled Attention–FFN structure of the standard Transformer with a single AttenFeed module per layer. We argue that this design alleviates structural constraints and the associated inductive bias ofthe standard Transformer in Section 5.1.

Structurally, uViT is equivalent to removing the FFN from a standard Transformer and introducing a parameter-free nonlinear activation function, which results in approximately three times fewer parameters (Geva et al., 2021). To ensure a fair comparison, we align the total parameter count of uViT with that of the standard baseline by proportionally increasing its number of layers and model dimension. The detailed model configurations are summarized in Table 1.

## 4. AttenFeed as a Generalization of Attention and FFN

In Section 4.1, we mathematically demonstrate that the AttenFeed module generalizes both Attention and FFN. Furthermore, in Section 4.2 and Section 4.3, we empirically show that the AttenFeed module exhibits the distinct characteristics of both Attention and FFN, respectively. Finally, in Section 4.4, we reveal that the parameters $\mathbf { W } _ { V }$ and $\mathbf { W } _ { O }$ within the AttenFeed module converge to intermediate values between those of standard attention and standard FFN. Ultimately, these findings indicate that the AttenFeed module mitigates the inductive biases inherent in Attention–FFN separation, effectively inheriting the properties ofboth.

## 4.1. Mathematical Description

To claim that the AttenFeed module generalizes both Attention and FFN, we present the following propositions. The detailed proofs are provided in Section A.

Proposition 4.1. Ifan AttenFeed head AttenFeed<sup>ℓ,h</sup> is selffocused, then the head’s output approximates a positionwise $F F N \colon A t t e n F e e d ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } ) _ { i } \approx F F N ( { \mathbf { x } } _ { i } ^ { \ell - 1 } ) .$

Proposition 4.2. Assume the pre-activation inputs to the GELU function are sufficiently small, such that $\lVert \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \rVert \stackrel { \cdot } { \to } ~ 0$ for all j. Under this condition, the AttenFeed head approximates a standard attention head: $A t t e n F e e d ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } ) \approx A t t n ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } )$ .

Proposition 4.3. Let an AttenFeed head AttenFeed<sup>ℓ,h</sup> follow the conditions ofProposition 4.1. Then, there exists a set ofparameterizations for the AttenFeed head such that $A t t e n F e e d ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } ) \approx A t t n ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } )$ .

Proposition 4.1 implies that when the attention of the AttenFeed module is self-focused, it becomes equivalent to an FFN. Proposition 4.2 demonstrates that the AttenFeed module can approximate standard attention. Proposition 4.3 more rigorously shows that when the AttenFeed module is self-focused, it approximates standard attention under looser conditions than those required in Proposition 4.2. Consequently, these propositions mathematically prove that the AttenFeed module is a generalization of both Attention and FFN.

## 4.2. FFN-like Behavior of the AttenFeed Module

To empirically validate Proposition 4.1, we examine whether the AttenFeed module exhibits the functional properties of an FFN. Specifically, we focus on the established observation that FFNs play a critical role in mitigating rank collapse within Transformers (Dong et al., 2021; Jha & Reagen, 2026; Noci et al., 2022; Park & Kim, 2022). Rank collapse refers to a phenomenon where the latent representations of different tokens become increasingly correlated as the network depth increases. This leads to “token uniformity,” where the model loses its expressivity because the features of distinct tokens become nearly indistinguishable. Models composed solely of attention layers are therefore known to exhibit a reduction in the rank of their representations (Dong et al., 2021).

As shown in Equation (4), the AttenFeed module introduces a GELU activation into the attention mechanism. This modification endows the conventional attention mechanism with FFN-like properties (Section 4.1). Thus, to empirically verify that AttenFeed module possesses FFN-like behavior, we compare the representational rank of uViT against a ViT variant in which the FFN is removed, resulting in a model composed solely of attention layers. We denote this FFNremoved ViT as xViT. Given that FFNs are known to play a pivotal role in preventing rank collapse, we hypothesize that xViT will exhibit a relatively lower rank. Conversely, we expect uViT to maintain a higher rank, as its AttenFeed modules are designed to integrate FFN-like functionalities.

Settings. First, we train xViT on a training set. Then, we freeze the model and record all the outputs of the [CLS] token on the associated validation set. Next, we measure the rank by performing Singular Value Decomposition (SVD) and counting the number of eigenvalues that account for 95% of the variance (Roy & Vetterli, 2007; Denil et al., 2013) (for details, see Section C.1). This rank value is then divided by the maximum possible rank to present the normalized rank (Yunis et al., 2024).<sup>3</sup> Subsequently, we insert a GELU activation into the attention module of xViT to transform it into uViT, and repeat the identical procedure to measure its normalized rank. We use CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) as train datasets.

Table 2. Performance and representation rank comparison on CIFAR10 and CIFAR100. The results demonstrate that uViT (composed of AttenFeed modules) consistently maintains a higher normalized rank and achieves superior accuracy compared to xViT (ViT without FFNs), confirming that the AttenFeed module effectively inherits the rank-preserving properties of FFNs.
<table><tr><td colspan="3"></td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td></tr><tr><td>Model</td><td> $( L , H , D )$ </td><td>Params.</td><td>Test acc.</td><td> $\mathbf { R a n k } _ { \mathrm { ( p r e - h e a d ) } }$ </td><td> $\mathbf { R a n k } _ { ( \mathrm { p o s t - h e a d } ) }$ </td><td>Test acc.</td><td> ${ \mathrm { R a n k } } _ { \mathrm { ( p r e - h e a d ) } }$ </td><td>Rank(post-head)</td></tr><tr><td>xViT uViT</td><td>(6, 4, 128)</td><td>0.43M</td><td>78.34%</td><td>0.375</td><td>0.70</td><td>50.94%</td><td>0.406</td><td>0.25</td></tr><tr><td rowspan="2">xViT</td><td rowspan="2"></td><td rowspan="2"></td><td>84.00%</td><td>0.383</td><td>0.70</td><td>58.56%</td><td>0.508</td><td>0.28</td></tr><tr><td>78.59%</td><td>0.356</td><td>0.70</td><td>52.73%</td><td>0.463</td><td>0.29</td></tr><tr><td>uViT</td><td>(6, 4, 160)</td><td>0.67M</td><td>85.16%</td><td>0.369</td><td>0.70</td><td>59.80%</td><td>0.506</td><td>0.32</td></tr><tr><td rowspan="2">xViT uViT</td><td rowspan="2">(12, 4, 128)</td><td rowspan="2">0.83M</td><td>80.13%</td><td>0.430</td><td>0.70</td><td>52.82%</td><td>0.586</td><td>0.29</td></tr><tr><td>85.60%</td><td>0.445</td><td>0.80</td><td>58.69%</td><td>0.586</td><td>0.30</td></tr></table>

Results. The experimental results are summarized in Table 6. We define pre-head rank and post-head rank as the normalized ranks of representations immediately preceding and following the classification head. When comparing xViT and uViT under identical configurations, uViT consistently exhibits a higher output rank, which correlates with improved test accuracy. These trends remain robust across various datasets (Section C). Given that the only structural disparity between uViT and xViT is the presence of the GELU activation, these findings suggest that the non-linear function imparts FFN-like, rank-preserving properties to the AttenFeed module.

## 4.3. Attention-like Behavior of AttenFeed Module

To empirically validate Proposition 4.2 and 4.3, we examine whether the AttenFeed module exhibits attention-like behavior through attention-based segmentation. Attention maps in ViTs visualize the salient image regions that the model focuses on when making predictions (Dosovitskiy, 2020; Gao et al., 2021; Darcet et al., 2023). These maps can therefore be leveraged to obtain segmentation masks for object regions (Helbling et al., 2025; Chefer et al., 2021). Thus, we assess whether the attention maps produced by ViT can also be utilized to generate meaningful segmentation masks.

Table 3. Segmentation performance using the attention maps from ViT and uViT. The results demonstrate that the AttenFeed module effectively generates meaningful segmentation masks, performing comparably to standard attention modules.
<table><tr><td rowspan="2">Model</td><td colspan="3">ImageNet-Seg.</td><td colspan="2">Pascal-VOC (Single class)</td></tr><tr><td>Acc.</td><td>mIoU</td><td>mAP</td><td>Acc. mIoU</td><td>mAP</td></tr><tr><td>ViT-S uViT-S</td><td>0.772 0.699</td><td>0.629 0.538</td><td>0.821 0.806</td><td>0.745 0.550 0.689 0.490</td><td>0.853 0.847</td></tr><tr><td>ViT-B</td><td>0.725</td><td>0.569</td><td>0.801</td><td>0.707 0.480</td><td>0.844</td></tr><tr><td>uViT-B</td><td>0.680</td><td>0.516</td><td>0.775</td><td>0.682 0.449</td><td>0.828</td></tr><tr><td>ViT-L uViT-L</td><td>0.666 0.617 0.447</td><td>0.500</td><td>0.765 0.767 0.631</td><td>0.677 0.422 0.384</td><td>0.819 0.821</td></tr></table>

Settings. First, we pretrain ViT-S/B/L and uViT-S/B/L on the ImageNet-1k dataset (Deng et al., 2009). To ensure a fair comparison, all models are trained under identical conditions (please refer to Section B for further details). After pretraining, we freeze the models and perform image segmentation. Following (Helbling et al., 2025; Chefer et al., 2021), we extract the attention map of the [CLS] token from the final layer for a given input image. Regions with values exceeding the mean attention score are binarized to produce the segmentation mask. We evaluate the segmentation performance of these attention-based masks on ImageNet-Segmentation (Guillaumin et al., 2014) and PASCAL-VOC (Everingham et al., 2010) datasets.

Results. The results are presented in Table 3. The attention map segmentation performance of uViT is slightly lower than that of ViT. This is expected, since the AttenFeed module in uViT must simultaneously act as both attention and FFN (Section 4.2), unlike the attention module in ViT. Nevertheless, the performance gap is not substantial, with uViT-L even achieving a higher mAP than ViT-L.

![](images/75385771ae6d92094cc22e5fd14ad982bcd1d2bbc8d40a10974b8fec7c97a259.jpg)  
Figure 2. Qualitative comparison of attention maps between ViT and uViT.

Furthermore, we present qualitative segmentation results in Figure 2. The first column shows the original input images, whereas the second and third columns display the attention maps of ViT and uViT, respectively. As shown in the $\mathrm { f i g \mathrm { - } }$ ure, uViT effectively captures salient object regions in the input image, suggesting that the AttenFeed module exhibits attention-like properties. Additional qualitative results are provided in Section C.3.

## 4.4. Comparing Parametric Spaces of uViT and ViT

As discussed in Section 3.2 and Section A, $\mathbf { W } _ { V }$ in the Atten-Feed module can be interpreted as either $\mathbf { W } _ { V }$ in the attention mechanism of a standard transformer or as $\mathbf { W } _ { \mathrm { i n } }$ in the FFN. Similarly, $\mathbf { W } _ { O }$ in the AttenFeed module can be interpreted as either $\mathbf { W } _ { O }$ or $\mathbf { W _ { \mathrm { o u t } } }$ in the standard transformer. Therefore, if our assumption that the AttenFeed module generalizes both Attention and FFN is valid, the parameters $\mathbf { W } _ { V }$ and $\mathbf { W } _ { O }$ of the AttenFeed module are expected to lie in an intermediate region between the corresponding parameters of the Attention and FFN modules in a standard Transformer. In this section, we empirically verify this hypothesis.

Settings. We visualize the parameter distributions of the ViT-B and uViT-B models used in Section 4.3. The analysis is conducted from two perspectives:

◦ Visualizing the value distributions of $\mathbf { W } _ { V }$ and $\mathbf { W } _ { \mathrm { i n } }$ in ViT, and $\mathbf { W } _ { V }$ in uViT, at a specific layer (Figure 3a).

◦ Visualizing the value distributions of $\mathbf { W } _ { O }$ and $\mathbf { W _ { \mathrm { o u t } } }$ in ViT, and $\mathbf { W } _ { O }$ in uViT, at a specific layer (Figure 3b).

The distribution of each matrix element is represented by connecting the midpoints of histogram bins to form a continuous density profile.

Results. As illustrated in Figure 3, the values of $\mathbf { W } _ { V }$ and $\mathbf { W } _ { O }$ in the AttenFeed module occupy an intermediate range between those of the standard attention and FFN components. These results suggest that the parameter space of the AttenFeed module does not deviate significantly from the regimes of standard attention and FFN, supporting the functional generalization of the AttenFeed module. Additional results are provided in Section C.

Finding of Section 4: AttenFeed module mathematically generalizes standard Attention and FFN, and empirically exhibits properties of both attention and FFN.

## 5. Impact of Inductive Bias in ViT

Unified Vision Transformer (uViT) consists of a sequence of AttenFeed modules, each capable of functioning as both Attention and FFN. Consequently, unlike the standard ViT, uViT is expected to be liberated from the inductive biases inherent in the conventional Attention–FFN dichotomy. In Section 5.1, we experimentally demonstrate that uViT indeed possesses weaker inductive bias than ViT. Subsequently, in Section 5.2, we compare the pretraining performance of uViT and ViT, showing that the Attention–FFN dichotomy in ViT has a detrimental effect on performance when the model size is small.

![](images/6d7238f0d41979ad4ba36f9626c00838dcf411c5081bad2524c8f822ab47c98c.jpg)  
(a) Mean, Std Comparison of �<sub>�</sub> and �<sub>��</sub>

![](images/67c402596398726dd6d9574031d20f92c51cc0ebf9a616b5ccd49c209a6a76d2.jpg)  
(b) Mean, Std Comparison of �<sub>�</sub> and �<sub>���</sub>  
Figure 3. Weight value distributions of uViT and standard ViT. The results show that uViT parameters occupy a transitional space between standard attention and FFN regimes, providing empirical evidence for the unified nature of the AttenFeed module.

## 5.1. Assessing Inductive Bias via Transfer Learning

uViT relaxes the strict separaration of Attention and FFN architecture in ViT. As a result, uViT is expected to exhibit less inductive bias than ViT and therefore holds the potential to learn more generalized representations (Dosovitskiy, 2020). To examine this hypothesis, we compare the transfer learning performance of pre-trained uViT and ViT on downstream datasets. If uViT indeed possesses less inductive bias than ViT, it should demonstrate better transfer performance than ViT.

Settings. We further train ViT-S/B/L and uViT-S/B/L, which were pre-trained on ImageNet-1k as described in Section 4.3, on the SVHN, CIFAR10, CIFAR100, and STL10 datasets (Netzer et al., 2011; Krizhevsky, 2009; Coates et al., 2011). For a fair comparison, all training hyperparameters were kept strictly identical (see Section B). To evaluate the flexibility of the output representations of pre-trained models, we report the performance of training only the classification head (linear probing). In addition, to assess the flexibility of the model’s internal parameters, we present full fine-tuning results.

Results. The experimental results are presented in Table 4. uViT outperformed ViT across all evaluated scenarios for linear probing, and for full fine-tuning, uViT achieved superior performance in the majority of cases. These results indicate that both the output representations and the internal parameters of uViT possess greater flexibility when adapting to unseen datasets. These results provide strong evidence that uViT exhibits less inductive bias than ViT.

Finding of Section 5.1: uViT exhibits less inductive bias imposed by the Attention–FFN dichotomy than ViT.

## 5.2. The Role of Inductive Bias at Scale

Section 4 shows that the AttenFeed module in uViT relaxes the Attention–FFN dichotomy in ViT, and Section 5.1 demonstrates that this leads to reduced inductive bias compared to ViT. Taken together, these findings indicate that uViT mitigates the inductive bias inherent in the Attention– FFN dichotomy. Leveraging uViT as a control group for investigating the role of the Attention–FFN separation, we compare the pre-training performance of ViT and uViT to examine under what conditions the Attention–FFN dichotomy is beneficial. Our results show that the benefit of the Attention–FFN dichotomy in ViT diminishes as the model size decreases.

Table 4. Transfer learning performance comparison between ViT and uViT. The results include Top-1 accuracy (%) for linear probing and full fine-tuning across four downstream datasets (SVHN, CIFAR10, CIFAR100, and STL10).
<table><tr><td rowspan="2">Model</td><td colspan="4">Linear Probing (%)</td><td colspan="4">Full Fine-tuning (%)</td></tr><tr><td>SVHN</td><td>CIFAR10</td><td>CIFAR100</td><td>STL10</td><td>SVHN</td><td>CIFAR10</td><td>CIFAR100</td><td>STL10</td></tr><tr><td>ViT-S</td><td>40.03</td><td>89.94</td><td>80.84</td><td>58.64</td><td>93.03</td><td>75.68</td><td>51.09</td><td>52.13</td></tr><tr><td>uViT-S</td><td>56.57</td><td>95.40</td><td>90.99</td><td>72.67</td><td>91.76</td><td>73.77</td><td>47.77</td><td>50.63</td></tr><tr><td>ViT-B</td><td>47.91</td><td>91.73</td><td>73.10</td><td>96.93</td><td>91.08</td><td>73.58</td><td>44.73</td><td>50.04</td></tr><tr><td>uViT-B</td><td>60.84</td><td>93.75</td><td>77.85</td><td>97.36</td><td>91.70</td><td>74.98</td><td>47.41</td><td>51.56</td></tr><tr><td>ViT-L</td><td>50.69</td><td>93.23</td><td>76.05</td><td>97.63</td><td>87.35</td><td>70.10</td><td>43.90</td><td>47.66</td></tr><tr><td>uViT-L</td><td>58.80</td><td>94.41</td><td>79.16</td><td>97.54</td><td>93.33</td><td>74.28</td><td>46.20</td><td>53.35</td></tr></table>

Table 5. Pre-training performance across different model scales. uViT shows a significant performance advantage over ViT at the Small scale, while the gap narrows at the Base scale. <sup>†</sup>We doubled the batch size for the experiments, since convergence of uViT was unstable under the default setting.
<table><tr><td rowspan="2">Model</td><td colspan="3">Test Acc. (%)</td></tr><tr><td>ImageNet-1k</td><td>Places365</td><td>iNaturalist</td></tr><tr><td>ViT-S</td><td>53.64</td><td>44.63</td><td>52.82</td></tr><tr><td>uViT-S</td><td>67.11</td><td>50.25</td><td>66.81</td></tr><tr><td>ViT-B</td><td>74.22</td><td>53.37</td><td>76.28</td></tr><tr><td>uViT-B</td><td>74.90</td><td>54.45</td><td>78.26</td></tr><tr><td>ViT-L</td><td>77.33</td><td>54.75</td><td>85.39†</td></tr><tr><td>uViT-L</td><td>77.18</td><td>55.14</td><td>84.49†</td></tr></table>

Settings. We pre-train the uViT-S/B/L and ViT-S/B/L on three larger-scale datasets: ImageNet-1k, Places365 (Zhou et al., 2018), and iNaturalist 2021 (Van Horn et al., 2021). The training procedure for all experiments is identical to the setup described in Section 4.3.

Results. The experimental results are presented in Table 5. When the model size is Small, uViT significantly outperforms ViT. However, as the model scales from Small to Base, this performance gap diminishes substantially. Since ViT and uViT were trained under identical conditions, these performance differences arise from the structural dichotomy of Attention–FFN. Therefore, we interpret this diminishing performance gap as an indication that the Attention–FFN dichotomy in ViT becomes counterproductive at a smaller model scales.

Hypothesis on the Result. We attribute the disadvantage of the Attention–FFN dichotomy at smaller scales to the rigid parameter allocation of ViT. Under the limited parameter budgets of small models, allocating parameters efficiently becomes critical. However, the strict separation between Attention and FFN forces parameters into fixed functional roles, which can lead to structural inefficiencies. For example, the standard architecture cannot dynamically allocate more capacity toward attention due to the fixed parameter budget assigned to the FFN, even when modeling token relationships is crucial for the dataset. In contrast, uViT provides architectural flexibility that allows limited parameters to be utilized more adaptively.

Finding of Section 5.2: Attention–FFN dichotomy in ViT becomes detrimental at a smaller model.

## 6. Conclusion

In this work, we revisit one of the most fundamental yet under-examined design choices in Vision Transformers: the rigid alternation between Attention and Feed-Forward Networks (FFNs). To analyze the necessity of this Attention– FFN dichotomy, we propose the AttenFeed module, a unified component that mathematically and empirically generalizes both standard attention and FFNs. Building upon this formulation, we introduce the unified Vision Transformer (uViT), which replaces the alternating structure with a sequence of AttenFeed modules, thereby relaxing the structural inductive bias imposed by explicit functional separation. Through theoretical analysis and extensive experiments, we demonstrate that the conventional Attention– FFN dichotomy acts as a restrictive inductive bias at smaller model scales. The limitations of our study are discussed in Section D.

## Impact Statement

The models we analyze are pre-trained on large-scale image datasets (ImageNet-1k, Places365, iNaturalist-2021) known to contain geographic, taxonomic, and demographic imbalances; any downstream use of the resulting representations inherits these biases and should be subject to the fairness and robustness practices already established for pre-trained vision backbones. Beyond this, we are not aware of specific societal consequences of this work that require additional discussion.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

An, J., Wang, D., Guo, P., Luo, J., and Schwing, A. On inductive biases that enable generalization in diffusion transformers. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Andriushchenko, M., Varre, A. V., Pillaud-Vivien, L., and Flammarion, N. Sgd with large step sizes learns sparse features. In International Conference on Machine Learning, pp. 903–925. PMLR, 2023.

Bachmann, G., Anagnostidis, S., and Hofmann, T. Scaling mlps: A tale of inductive bias. Advances in Neural Information Processing Systems, 36:60821–60840, 2023.

Baevski, A., Zhou, Y., Mohamed, A., and Auli, M. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

Bengio, Y., Goodfellow, I., Courville, A., et al. Deep learning, volume 1. MIT press Cambridge, MA, USA, 2017.

Bowling, S. R., Khasawneh, M. T., Kaewkuekool, S., and Cho, B. R. A logistic approximation to the cumulative normal distribution. Journal of industrial engineering and management, 2(1):114–127, 2009.

Chefer, H., Gur, S., and Wolf, L. Transformer interpretability beyond attention visualization. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 782–791, 2021.

Clark, K., Khandelwal, U., Levy, O., and Manning, C. D. What does bert look at? an analysis of bert’s attention. arXiv preprint arXiv:1906.04341, 2019.

Coates, A., Ng, A., and Lee, H. An analysis of singlelayer networks in unsupervised feature learning. In Proceedings of the fourteenth international conference on

artificial intelligence and statistics, pp. 215–223. JMLR Workshop and Conference Proceedings, 2011.

Cordonnier, J.-B., Loukas, A., and Jaggi, M. On the relationship between self-attention and convolutional layers. In International Conference on Learning Representations (ICLR), 2020. URL https://openreview.net/ forum?id=HJlnC1rKPB.

Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 702–703, 2020.

Darcet, T., Oquab, M., Mairal, J., and Bojanowski, P. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.

d’Ascoli, S., Touvron, H., Leavitt, M. L., Morcos, A. S., Biroli, G., and Sagun, L. Convit: Improving vision transformers with soft convolutional inductive biases. Journal of Statistical Mechanics: Theory and Experiment, 2022(11):114005, nov 2022. ISSN 1742-5468. doi: 10.1088/1742-5468/ac9830. URL https://doi. org/10.1088/1742-5468/ac9830.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Denil, M., Shakibi, B., Dinh, L., Ranzato, M., and De Freitas, N. Predicting parameters in deep learning. Advances in neural information processing systems, 26, 2013.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference ofthe North American chapter ofthe associationfor computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In International conference on machine learning, pp. 2793–2803. PMLR, 2021.

Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Edelman, B. L., Goel, S., Kakade, S., and Zhang, C. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, pp. 5793–5831. PMLR, 2022a.

Edelman, B. L., Goel, S., Kakade, S., and Zhang, C. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, pp. 5793–5831. PMLR, 2022b.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman, A. The PASCAL visual object classes (VOC) challenge. International Journal ofComputer Vision, 88(2):303–338, 2010. doi: 10.1007/ s11263-009-0275-4.

Ferrando, J., Sarti, G., Bisazza, A., and Costa-Jussà, M. R. A primer on the inner workings of transformer-based language models. arXiv preprint arXiv:2405.00208, 2024.

Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International conference on machine learning, pp. 10323–10337. PMLR, 2023.

Gao, W., Wan, F., Pan, X., Peng, Z., Tian, Q., Han, Z., Zhou, B., and Ye, Q. Ts-cam: Token semantic coupled attention map for weakly supervised object localization. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 2886–2895, 2021.

Geva, M., Schuster, R., Berant, J., and Levy, O. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, 2021.

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. In First conference on language modeling, 2024.

Guillaumin, M., Küttel, D., and Ferrari, V. ImageNet autoannotation with segmentation propagation. International Journal ofComputer Vision, 110(3):328–348, 2014. doi: 10.1007/s11263-014-0713-9.

Han, S., Kim, S., Byeon, G., Yoon, J., and Hong, S. Zebra: Leveraging diagonal attention pattern for vision transformer accelerator. In 2025 Design, Automation & Test in Europe Conference (DATE), pp. 1–7. IEEE, 2025.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

He, S., Sun, G., Shen, Z., and Li, A. What matters in transformers? not all attention is needed. arXiv preprint arXiv:2406.15786, 2024.

He, S., Ge, T., Sun, G., Tian, B., Wang, X., and Yu, D. Router-tuning: A simple and effective approach for dynamic depth. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 1925–1938, 2025.

Helbling, A., Meral, T. H. S., Hoover, B., Yanardag, P., and Chau, D. H. Conceptattention: Diffusion transformers learn highly interpretable features. arXiv preprint arXiv:2502.04320, 2025.

Hendrycks, D. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Hua, W., Dai, Z., Liu, H., and Le, Q. Transformer quality in linear time. In International conference on machine learning, pp. 9099–9117. PMLR, 2022.

Huben, R. and Morris, V. Attention-only transformers and implementing mlps with attention heads. arXiv preprint arXiv:2309.08593, 2023.

Jelassi, S., Brandfonbrener, D., Kakade, S. M., and Malach, E. Repeat after me: Transformers are better than state space models at copying. arXiv preprint arXiv:2402.01032, 2024.

Jha, N. K. and Reagen, B. Nerve: Nonlinear eigenspectrum dynamics in llm feed-forward networks. In The Fourteenth International Conference on Learning Representations (ICLR), 2026.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., et al. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583–589, 2021. doi: 10.1038/s41586-021-03819-2.

Krizhevsky, A. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. URL https://www.cs.toronto.edu/ \~kriz/learning-features-2009-TR.pdf.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Lavie, I., Gur-Ari, G., and Ringel, Z. Towards understanding inductive bias in transformers: A view from infinity. arXiv preprint arXiv:2402.05173, 2024.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5.726791.

Lee, M. Gelu activation function in deep learning: a comprehensive mathematical analysis and performance. arXiv preprint arXiv:2305.12073, 2023.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Luo, X., Wang, W., and Yan, X. Diffskip: Differential layer skipping in large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 7221–7231, 2025.

Luo, Y., Song, C., Han, X., Chen, Y., Xiao, C., Meng, X., Deng, L., Wei, J., Liu, Z., and Sun, M. Sparsing law: Towards large language models with greater activation sparsity. arXiv preprint arXiv:2411.02335, 2024.

Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., Ng, A. Y., et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervisedfeature learning, volume 2011, pp. 4. Granada, 2011.

Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N. Exploring generalization in deep learning. Advances in neural information processing systems, 30, 2017.

Noci, L., Anagnostidis, S., Biggio, L., Orvieto, A., Singh, S. P., and Lucchi, A. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. Advances in Neural Information Processing Systems, 35: 27198–27211, 2022.

Park, N. and Kim, S. How do vision transformers work? arXiv preprint arXiv:2202.06709, 2022.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Peng, Z., Qi, L., Shi, Y., and Gao, Y. A theoretical explanation of activation sparsity through flat minima and adversarial robustness. arXiv preprint arXiv:2309.03004, 2023.

Peruzzo, E., Sangineto, E., Liu, Y., De Nadai, M., Bi, W., Lepri, B., and Sebe, N. Spatial entropy as an inductive bias for vision transformers. Machine Learning, 113(9): 6945–6975, 2024. doi: 10.1007/s10994-024-06570-7.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Roy, O. and Vetterli, M. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pp. 606–610. IEEE, 2007.

Sanford, C., Hsu, D. J., and Telgarsky, M. Representational strengths and limitations of transformers. Advances in Neural Information Processing Systems, 36:36677– 36707, 2023.

Scherer, D., Müller, A., and Behnke, S. Evaluation of pooling operations in convolutional architectures for object recognition. In International conference on artificial neural networks, pp. 92–101. Springer, 2010.

Schmidhuber, J. Deep learning in neural networks: An overview. Neural Networks, 61:85–117, 2015. doi: 10. 1016/j.neunet.2014.09.003.

Sun, M., Chen, X., Kolter, J. Z., and Liu, Z. Massive activations in large language models. arXiv preprint arXiv:2402.17762, 2024.

Sutton, R. The bitter lesson. Incomplete Ideas (blog), 13(1): 38, 2019.

Szatkowski, F., B˛edkowski, P., Devoto, A., Dubinski, J.,´ Minervini, P., Piórczynski, M., Scardapane, S., and´ Wójcik, B. Universal properties of activation sparsity in modern large language models. arXiv preprint arXiv:2509.00454, 2025.

Tolstikhin, I. O., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., et al. Mlp-mixer: An all-mlp architecture for vision. Advances in neural information processing systems, 34:24261–24272, 2021.

Touvron, H., Bojanowski, P., Caron, M., Cord, M., El-Nouby, A., Grave, E., Izacard, G., Joulin, A., Synnaeve, G., Verbeek, J., et al. ResMLP: Feedforward networks for image classification with data-efficient training. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):5314–5321, 2023. doi: 10.1109/TPAMI. 2022.3206148.

Van Horn, G., Cole, E., Beery, S., Sisinni, K., Shepard, A., and Mac Aodha, O. Benchmarking representation learning for natural world image collections. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12884–12893, 2021.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. arXiv preprint arXiv:1905.09418, 2019.

Wang, P., Lu, Y., Yu, Y., Pai, D., Qu, Q., and Ma, Y. Attention-only transformers via unrolled subspace denoising. arXiv preprint arXiv:2506.03790, 2025.

Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollár, P., and Girshick, R. Early convolutions help transformers see better. Advances in neural information processing systems, 34:30392–30400, 2021.

Xu, Y., Zhang, Q., Zhang, J., and Tao, D. Vitae: Vision transformer advanced by exploring intrinsic inductive bias. Advances in neural information processing systems, 34:28522–28535, 2021.

Yang, Y., Wang, C., and Li, J. Umoe: Unifying attention and ffn with shared experts. arXiv preprint arXiv:2505.07260, 2025.

Yu, Y., Buchanan, S., Pai, D., Chu, T., Wu, Z., Tong, S., Haeffele, B., and Ma, Y. White-box transformers via sparse rate reduction. Advances in Neural Information Processing Systems, 36:9422–9457, 2023.

Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., and Yoo, Y. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6023–6032, 2019.

Yunis, D., Patel, K. K., Wheeler, S., Savarese, P. H. P., Vardi, G., Livescu, K., Maire, M., and Walter, M. Rank minimization, alignment and weight decay in neural networks. In High-dimensional Learning Dynamics 2024: The Emergence ofStructure and Reasoning, 2024.

Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

Zhang, Z., Song, Y., Yu, G., Han, X., Lin, Y., Xiao, C., Song, C., Liu, Z., Mi, Z., and Sun, M. ReLU<sup>2</sup> wins: Discovering efficient activation functions for sparse llms. arXiv preprint arXiv:2402.03804, 2024.

Zhao, A., Ye, F., Fan, Y., Tong, J., Fei, Z., Su, H., and Shen, X. Skipgpt: Dynamic layer pruning reinvented with token awareness and module decoupling. arXiv preprint arXiv:2506.04179, 2025.

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(6):1452–1464, 2018. doi: 10.1109/TPAMI.2017.2723009.

Zhou, D., Kang, B., Jin, X., Yang, L., Lian, X., Jiang, Z., Hou, Q., and Feng, J. Deepvit: Towards deeper vision transformer. arXiv preprint arXiv:2103.11886, 2021.

## A. Mathematical Descriptions

We now demonstrate the theoretically generalized formulation of AttenFeed modules encompassing both standard attention and FFN components, thus minimizing the inductive biases.

## A.1. FFN-like Behavior.

First, we examine the asymptotic behavior of the module when the attention mechanism becomes highly localized. We demonstrate that when each token attends exclusively to itself, the spatial mixing vanishes, and the module collapses into a standard, position-wise FFN.

Proposition A.1. If the h-th AF head at layer ℓ is self-focused, then the head’s output approximates a position-wise FFN: $A F ^ { \ell , \overline { { h } } } ( { \mathbf { X } } ^ { \ell - 1 } ) _ { i } \approx F F N ( { \mathbf { x } } _ { i } ^ { \ell - 1 } )$ .

Proof. First, we must show that AttenFeed module follows the condition of self-focus, such that $a _ { i , i } ^ { \ell , h } \to 1$ and $a _ { i , j } ^ { \ell , h } \to 0$ for all $j \neq i .$ . The possibility of this condition is empirically established by various studies(Clark et al., 2019; Voita et al., 2019; Zhou et al., 2021; Han et al., 2025; Cordonnier et al., 2020) on Transformer models called diagonal dominance, and as AttenFeed module is based on Transformer, the condition is given. Recall the definition of our module for a single token i as the sum of attention-weighted activation units:

$$
\underset { j } { \mathrm { A F } } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } = \sum _ { j } a _ { i , j } ^ { \ell , h } \mathrm { G E L U } \left( \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h } .\tag{6}
$$

We decompose this summation into the diagonal term ${ ( j = i ) }$ and the off-diagonal terms $( j \neq i )$

$$
\begin{array} { r } { \mathrm { A F } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } = \underbrace { a _ { i , i } ^ { \ell , h } \mathrm { G E L U } \left( \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h } } _ { \mathrm { D i a g o n a l T e r m } } } \end{array}\tag{7}
$$

$$
+ \underbrace { \sum _ { j \neq i } a _ { i , j } ^ { \ell , h } \mathbf { G } \mathbf { E } \mathbf { L } \mathbf { U } \left( \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h } } _ { \mathrm { O f f - d i a g o n a l ~ T e r m } } .\tag{8}
$$

Applying the diagonal dominance condition where $a _ { i , i } ^ { \ell , h }$ ≈ 1 and $a _ { i , j } ^ { \ell , h } \approx 0$ for $j \neq i ,$ the cross-term vanishes:

$$
\mathbf { A } \mathbf { F } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } \approx 1 \cdot \mathrm { G E L U } \left( \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h } + \sum _ { j \ne i } 0 \cdot \left( \dots \right)\tag{9}
$$

$$
\begin{array} { r } { \mathbf { \Omega } = \mathrm { G E L U } \left( \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h } . } \end{array}\tag{10}
$$

This resulting expression is exactly the definition of a standard, token-wise GELU FFN applied to $\mathbf { x } _ { i } ^ { \ell - 1 }$ , proving that the AttenFeed module naturally reduces to a standard FFN structure. □

## A.2. Attention-like Behavior.

To mathematically justify our module’s behavior in reproducing the standard attention module, we utilize the argument that an FFN layer with SiLU activation is mathematically identical to the attention head of the standard module (Huben & Morris, 2023). For more details on this, refer to Theorem 4 of the cited study. Our argument can be summed up with the following proposition.

Proposition A.2. Assume the pre-activation inputs to the GELUfunction are sufficiently small, such that $\| \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \| \to 0$ for all j. Under this condition, the AF head approximates a standard attention head: $A F ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } ) \approx A t t n ^ { \ell , h } \widetilde { \left( { \mathbf { X } } ^ { \ell - 1 } \right) }$

Proof. Let the pre-activation vector be $\mathbf z _ { j } = \mathbf x _ { j } ^ { \ell - 1 } \mathbf W _ { V } ^ { \ell , h }$ . Recent studies on activation sparsity demonstrate that modern architectures exhibit highly sparse, input-dependent activation patterns even with non-strict activation functions like GELU (Szatkowski et al., 2025; Andriushchenko et al., 2023; Peng et al., 2023; Zhang et al., 2024; Luo et al., 2024). Consequently, the components of $\mathbf { z } _ { j }$ can be partitioned into two distinct functional regimes: a sparse subset of large positive activations and a dense majority of near-zero values. We analyze the behavior of the GELU function across these two regimes:

◦ Sparse active regime $( z \gg 0 ) \mathrm { : }$ : The function approaches its linear asymptote, GEL $\mathbf { U } ( z ) \approx z \ ( \mathrm { L e e } , 2 0 2 3 )$ . These few large values dominantly drive the output of the layer. The massive activation phenomenon also falls into this category (Sun et al., 2024).

◦ Dense inactive regime $( z \approx 0 ) { \ : } :$ Following recent observations on activation sparsity (Frantar & Alistarh, 2023), the activation values in this region are sufficiently small and inherently negligible. Thus, we can directly approximate the function as near-zero, i.e., GELU(z) ≈ 0.

For the inactive regime, since both z and GELU(z) are approximately zero, replacing GELU(z) with z introduces near-zero absolute error. Assuming large negative activations are effectively suppressed by the sparsity pattern, we can approximate the activation function as an identity mapping over the entire vector without significant loss of accuracy:

$$
\mathrm { G E L U } ( \mathbf { z } _ { j } ) \approx \mathbf { z } _ { j } .\tag{11}
$$

Substituting this linear approximation back into the AttenFeed module formulation yields:

$$
\boldsymbol { \mathrm { A F } } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } \approx \sum _ { j } a _ { i , j } ^ { \ell , h } \left( \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h } = \boldsymbol { \mathrm { A t t n } } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) .\tag{12}
$$

This resulting expression is mathematically identical to the formal definition of the standard attention mechanism given in Equation 2. Thus, driven by the properties of activation sparsity, the AttenFeed module structurally converges to standard attention. □ □

Proposition A.3. Let an AF head $A F ^ { \ell , h }$ follow the conditions of Theorem 4.1. Then, there exists a set of parameterizations for the AF head such that $A F ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } ) \approx A t t n ^ { \ell , h } ( { \mathbf { X } } ^ { \ell - 1 } )$ ).

Proof. From Theorem 4.1, we can redefine AttenFeed modules as FFN with GELU activation. To utilize the mentioned argument, we must justify the cases where GELU function approximates SiLU function. This can be done with a simple minimax approximation. We define the native Sigmoid gate as σ and establish the following minimax optimization problem to find the optimal scaling factor $\lambda ^ { * }$ that maps the Gaussian CDF Φ to the standard Sigmoid:

$$
\lambda ^ { * } = \underset { \lambda } { \arg \operatorname* { m i n } } \left( \underset { x \in \mathbb { R } } { \operatorname* { m a x } } \left| \Phi ( \lambda x ) - \sigma ( x ) \right| \right) .\tag{13}
$$

Based on the minimax theorem, such $\lambda ^ { * }$ exists (numerically, $\lambda ^ { * } \approx 1 . 7 0 2$ (Bowling et al., 2009)) and can be parameterized into AttenFeed modules from the perspective of FFN. Without loss of generality, we define $\mathbf { W } _ { \mathrm { o u t } } ^ { \ell , h , k } : = \lambda ^ { * } \mathbf { W } _ { O } ^ { \ell , h } \vert k , : ] \in \mathbb { R } ^ { d }$ and $\mathbf { W } _ { \mathrm { i n } } ^ { \ell , h , k } : = \mathbf { W } _ { V } ^ { \ell , h } [ : , k ] \in \mathbb { R } ^ { d }$ where $k \in \{ 1 , \ldots , d _ { h } \}$ , and rewrite the AttenFeed module as follows:

$$
\mathbf { A } \mathbf { F } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } = \mathbf { G E L U } \left( \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } \right) \mathbf { W } _ { O } ^ { \ell , h }\tag{14}
$$

$$
\approx \sum _ { k = 1 } ^ { d _ { h } } \mathrm { S i L U } \left( \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { \mathrm { i n } } ^ { \ell , h , k } \right) \mathbf { W } _ { \mathrm { o u t } } ^ { \ell , h , k } .\tag{15}
$$

Under the assumption of diagonal dominance, the conditions of Theorem 4 in (Huben & Morris, 2023) are satisfied. Thus, the following holds:<sup>4</sup>

$$
\mathrm { S i L U } \left( \mathbf { x } _ { i } ^ { \ell - 1 } \mathbf { W } _ { \mathrm { i n } } ^ { \ell , h , k } \right) \mathbf { W } _ { \mathrm { o u t } } ^ { \ell , h , k } = \sum _ { j } a _ { i , j } ^ { \ell , h , k } \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { \mathrm { i n } } ^ { \ell , h , k } \mathbf { W } _ { \mathrm { o u t } } ^ { \ell , h , k } .\tag{16}
$$

This leads to the conclusion that the head of AttenFeed modules act as a standard attention head modules that has $d _ { h }$ heads. This can be concluded with the following equation:

$$
\mathsf { A F } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } = \sum _ { k } \sum _ { j } a _ { i , j } ^ { \ell , h , k } \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { \mathrm { i n } } ^ { \ell , h , k } \mathbf { W } _ { \mathrm { o u t } } ^ { \ell , h , k }\tag{17}
$$

$$
= \sum _ { k } \mathrm { A t t n } ^ { \ell , h , k } ( { \mathbf { X } } ^ { \ell - 1 } )\tag{18}
$$

$$
\begin{array} { r } { = \mathrm { A t t n } ^ { \ell , h } ( { \mathbf { X } ^ { \ell - 1 } } ) . } \end{array}\tag{19}
$$

## A.3. Mixture of FFN-like Behavior.

While AttenFeed modules can function as either standard attention or standard FFNs, they also manifest a novel emergent behavior: a spatially weighted mixture of FFNs. As shown in Equation (20), the extent to which the non-linear FFN outputs from each token position are integrated into the residual stream is dynamically governed by the QK circuit of the attention mechanism.

$$
\mathsf { A F } ^ { \ell , h } ( \mathbf { X } ^ { \ell - 1 } ) _ { i } = \sum _ { j } a _ { i , j } ^ { \ell , h } \sigma ( \mathbf { x } _ { j } ^ { \ell - 1 } \mathbf { W } _ { V } ^ { \ell , h } ) \mathbf { W } _ { O } ^ { \ell , h }
$$

$$
= \sum _ { j } a _ { i , j } ^ { \ell , h } \mathrm { F F N } ( \mathbf { x } _ { j } ^ { \ell - 1 } ) .\tag{20}
$$

(21)

## B. Training Details

We evaluate the proposed unified Vision Transformer (uViT) against the standard Vision Transformer (ViT) baseline at two scales: (i) a toy-scale regime on small classification datasets used for our rank analyses (Section 4.2), and (ii) a large-scale pre-training regime on ImageNet-1k, Places365, and iNaturalist-2021. In both regimes, every hyper-parameter that is not intrinsic to the architectural difference between ViT and uViT is held fixed, so that all observed differences can be attributed to the architectural change rather than to the optimization recipe.

## B.1. Toy-Scale Training

Datasets and preprocessing. Toy-scale experiments are conducted on CIFAR-10, CIFAR-100, STL-10, and SVHN. All images are resized to a common spatial resolution (32×32 for CIFAR-10/100 and SVHN, 96×96 for STL-10) and patchified with a patch size of 4. Training images are augmented with a random crop (padding 4 for CIFAR-10/100, STL-10 and padding 2 for SVHN), random horizontal flip (except SVHN, where flipping alters semantics), and RandAugment (Cubuk et al., 2020) (CIFAR-10/100, STL-10 only). Evaluation images are only resized and normalized using each dataset’s mean and standard deviation. Identical augmentation pipelines are applied to ViT and uViT.

Optimization. All toy-scale models are trained for 100 epochs with a global batch size of 128 on a single GPU. We use the AdamW (Loshchilov & Hutter, 2017) optimizer with a peak learning rate of $5 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 3 }$ , warmed up linearly from peak/W to the peak over $W = 1 0$ epochs and then decayed by a cosine annealing schedule to $1 0 ^ { - 5 }$ over the remaining 190 epochs. Loss is cross-entropy over hard labels. Dropout is fixed at 0.1 throughout. Random seeds, data splits, dataloader ordering, optimizer, learning-rate schedule, dropout, and augmentation are identical across ViT, uViT, and noFFN within each dataset, ensuring that only the architectural choice varies between the compared models.

## B.2. Large-Scale Pre-training

Datasets and preprocessing. Large-scale pre-training is performed on ImageNet-1k, Places365-standard, and iNaturalist-2021 at an input resolution of 224×224 with a patch size of 16. For training, we apply a standard strong-augmentation pipeline: RANDOMRESIZEDCROP(224, scale = (0.08, 1.0)), random horizontal flip with probability 0.5, and RandAugment with magnitude 9, followed by normalization with the ImageNet mean and standard deviation. At the collate stage, each mini-batch is randomly transformed by either MixUp $( \alpha _ { \mathrm { m i x u p } } = 1 . 0 )$ (Zhang et al., 2017) or CutMix $( \alpha _ { \mathrm { c u t m i x } } = 1 . 0 )$ (Yun et al., 2019), yielding soft labels. Evaluation uses a single center crop: images are first resized to 256 pixels (short-side) and then center-cropped to 224×224. The same training and evaluation pipeline is used for ViT and uViT on every dataset.

Optimization. Both ViT and uViT are trained with the AdamW optimizer for 300 epochs. We adopt a reference recipe of peak learning rate $3 \times 1 0 ^ { - 3 }$ at a reference global batch of 4096, and use the linear scaling rule (Goyal et al., 2017):

$$
\begin{array} { r } { \mathrm { l r } = 3 \times 1 0 ^ { - 3 } \cdot \frac { B _ { \mathrm { g l o b a l } } } { 4 0 9 6 } , } \end{array}\tag{22}
$$

to adapt to the actual global batch size $B _ { \mathrm { g l o b a l } }$ . A linear warm-up over the first 10,000 optimization steps is followed by a cosine decay to zero over the remainder of training. Weight decay is 0.3, gradients are clipped to a global $\ell _ { 2 }$ norm of 1.0, and training is conducted in mixed precision (FP16 autocast with a dynamic loss scaler). The loss is cross-entropy against the soft targets produced by MixUp/CutMix.

Distributed training. All large-scale runs use PyTorch DISTRIBUTEDDATAPARALLEL over a single multi-GPU node with the NCCL backend, launched via torchrun. A DistributedSampler ensures that every GPU sees a disjoint shard of the training set each epoch. Random seeds are offset by the process rank so that augmentation is decorrelated across GPUs while remaining reproducible. Per-GPU batch sizes are chosen so that $B _ { \mathrm { g l o b a l } }$ is identical between matched ViT/uViT runs, and the linear scaling rule above then yields the same effective learning-rate trajectory for both models.

## C. Experiment Result Details

## C.1. Rank Measurement Procedure

The experiments in Section 4.2 and Section C.2 rely on a representation rank that is estimated from finite samples rather than from the exact matrix rank. This subsection details the mathematical procedure and clarifies the two rank quantities reported in the tables.

Effective rank via truncated variance. Let $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { d }$ denote the [CLS] token representations produced by a frozen model at a fixed layer when applied to the N samples of the evaluation set, and let $\mathbf { Z } \in \mathbb { R } ^ { N \times d }$ be their row-stacked matrix. Applying singular value decomposition (SVD) to Z yields non-increasing singular values $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq \sigma _ { k } \geq 0$ with $k = \operatorname* { m i n } ( N , d )$ . Because the total variance carried by Z is $\textstyle \sum _ { i = 1 } ^ { k } \sigma _ { i } ^ { 2 }$ , the proportion of variance captured by the top-r singular directions is

$$
\eta ( r ) = \frac { \sum _ { i = 1 } ^ { r } \sigma _ { i } ^ { 2 } } { \sum _ { i = 1 } ^ { k } \sigma _ { i } ^ { 2 } } .\tag{23}
$$

Following Roy & Vetterli (2007); Denil et al. (2013), we define the effective rank at a variance-explanation threshold $\tau \in ( 0 , 1 ]$ as the smallest number of singular directions whose cumulative variance reaches τ:

$$
r _ { \tau } ( \mathbf { Z } ) \ = \ \operatorname* { m i n } \{ r \in \{ 1 , \ldots , k \} \ : \ \eta ( r ) \geq \tau \ \} .\tag{24}
$$

This estimator is a soft analogue of the exact matrix rank that discards singular directions carrying negligible variance and is therefore robust to numerical and sampling noise. We fix $\tau = 0 . 9 5$ throughout, following the convention used in prior works on rank analysis of deep networks.

Normalized rank. The raw effective rank $r _ { \tau } ( \mathbf { Z } )$ is bounded above by $k = \operatorname* { m i n } ( N , d )$ , which depends on the feature dimension of the representation under analysis. To enable a fair comparison between representations with different feature dimensions (e.g., between the embedding features of a model and its logit outputs), we report the normalized rank (Yunis et al., 2024)

$$
\rho _ { \tau } ( { \bf Z } ) = \frac { r _ { \tau } ( { \bf Z } ) } { \mathrm { m i n } ( N , d ) } = \frac { r _ { \tau } ( { \bf Z } ) } { d } ,\tag{25}
$$

where the last equality uses the fact that $N \gg$ d for every evaluation set considered in this paper. The normalized rank takes values in (0, 1]: a value close to 1 indicates that essentially every singular direction of the feature space carries a non-negligible amount of variance, whereas a value close to 0 indicates that the representation has collapsed into a low-dimensional subspace.

Pre-head and post-head rank. We apply the procedure above at two distinct points of the forward pass in order to characterize both the inputs and the outputs of the classification head. Denoting the final LayerNorm immediately before the classifier by LN(·) and the classification head by Head(·):

◦ The pre-head rank is the normalized rank of the representation that enters the classification head, i.e. the output of the final LayerNorm, $\mathbf { z } _ { i } ^ { \mathrm { p r e } } = \mathrm { L N } ( \mathbf { h } _ { i } ^ { L } ) [ \mathrm { C L S } ] \in \mathbb { R } ^ { d }$ , where $\mathbf { h } _ { i } ^ { L }$ is the last-layer token sequence for sample i. The feature dimension is the embedding dimension d, so $\rho _ { \tau } ( { \bf Z } ^ { \mathrm { p r e } } ) = r _ { \tau } ( { \bf Z } ^ { \mathrm { p r e } } ) / d .$

◦ The post-head rank is the normalized rank of the representation that leaves the classification head, i.e. the logits, $\mathbf { z } _ { i } ^ { \mathrm { p o s t } } = \mathrm { H e a d } \left( \mathbf { z } _ { i } ^ { \mathrm { p r e } } \right) \in \mathbb { R } ^ { C }$ , where C is the number of classes. The feature dimension is now C, so $\rho _ { \tau } ( \mathbf { Z } ^ { \mathrm { p o s t } } ) = r _ { \tau } ( \mathbf { Z } ^ { \mathrm { p o s t } } ) / C$

The two quantities probe complementary aspects of the learned representation. The pre-head rank measures how much of the embedding space is actively used by the backbone before classification; a low pre-head rank indicates that the backbone has collapsed distinct tokens into an overly narrow subspace, so that samples from different classes become difficult to separate linearly. The post-head rank, in contrast, measures how effectively the head spreads class-discriminative information across the available logit dimensions; a post-head rank well below 1 indicates that the logit output effectively operates on fewer than C degrees of freedom, which is a direct symptom of confounded classes. Reporting both rank statistics thus allows us to localize rank collapse either to the backbone or to the classification stage.

## C.2. Rank Preservation of AttenFeed Module

In Section 4.2, we empirically demonstrated on CIFAR-10 and CIFAR-100 that the AttenFeed module inherits the rankpreserving behavior of FFNs. To verify that this finding is not specific to a particular dataset, we repeat the same experimental protocol on two additional small-scale datasets, SVHN (Netzer et al., 2011) and STL-10 (Coates et al., 2011), which differ considerably from CIFAR in visual statistics (digit crops and high-resolution natural images, respectively). For each dataset, xViT (a ViT variant in which the FFN is removed) and uViT are trained under identical conditions at three matched (L, H, D) configurations, and the normalized ranks of the [CLS] token representations immediately preceding and following the classification head (pre-head and post-head) are measured via SVD with the 95% variance threshold. The results are summarized in Table 6.

Table 6. Performance and representation rank comparison on SVHN and STL10. uViT consistently matches or exceeds xViT in pre-head rank and test accuracy, confirming that the rank-preserving behavior of the AttenFeed module generalizes beyond CIFAR-10/100.
<table><tr><td colspan="3"></td><td colspan="3">SVHN</td><td colspan="3">STL10</td></tr><tr><td>Model</td><td>(L, H, D)</td><td>Params.</td><td>Test acc.</td><td>Rank(pre-head)</td><td>Rank(post-head)</td><td>Test acc.</td><td>Rank(pre-head)</td><td>Rank(post-head)</td></tr><tr><td>xViT</td><td>(6, 4, 128)</td><td>0.43M</td><td>94.70%</td><td>0.320</td><td>0.8</td><td>67.15%</td><td>0.367</td><td>0.6</td></tr><tr><td rowspan="2">uViT xViT</td><td rowspan="2"></td><td rowspan="2">0.67M</td><td>94.78%</td><td>0.320</td><td>0.8</td><td>69.79%</td><td>0.421</td><td>0.7</td></tr><tr><td>94.87%</td><td>0.269</td><td>0.8</td><td>66.51%</td><td>0.325</td><td>0.7</td></tr><tr><td rowspan="2">uViT</td><td rowspan="2">(6, 4, 160)</td><td rowspan="2"></td><td>95.00%</td><td>0.294</td><td>0.8</td><td>69.60%</td><td>0.406</td><td>0.7</td></tr><tr><td>94.85%</td><td>0.336</td><td>0.8</td><td>67.31%</td><td>0.406</td><td></td></tr><tr><td>xViT uViT</td><td>(12, 4, 128)</td><td>0.83M</td><td>95.04%</td><td>0.352</td><td>0.8</td><td>70.51%</td><td>0.430</td><td>0.7 0.7</td></tr></table>

Across every configuration, uViT attains a pre-head rank that is equal to or higher than that of xViT, and this higher rank is consistently accompanied by higher test accuracy. The advantage is especially pronounced on STL-10, where uViT improves the pre-head rank by up to 0.081 and the test accuracy by up to 3.20 percentage points over xViT. On SVHN, where both models nearly saturate due to the relative simplicity of the task, the rank and accuracy margins are smaller but still favor uViT. These observations replicate the CIFAR-10/100 trend reported in the main text and reinforce our conclusion that the GELU activation in the AttenFeed module imparts FFN-like rank-preserving properties independent of the target dataset.

## C.3. Attention Map Comparison of uViT and ViT

To complement the quantitative segmentation scores reported in Section 4.3, we present additional qualitative comparisons of the attention maps produced by ViT-B and uViT-B in the figure below. Across a wide variety of object categories and scene compositions, uViT consistently attends to the salient object regions identified by ViT, with only minor deviations in spatial extent. This qualitative agreement extends the main-text finding to a broader set of images and further supports Propositions 4.2 and 4.3.

Image  
ViT  
uViT  
Image  
ViT  
uViT  
![](images/ec3cd5c08f30ce764ca4c52b2d07654ae1a1bb8a33f51c41c1ab1ed4e87925f4.jpg)  
Figure 4. Qualitative comparison of attention maps between ViT and uViT. uViT-B reliably highlights the same salient object regions as ViT-B, supporting the attention-like behavior of the AttenFeed module.

## C.4. Parametric Space Comparison of uViT and ViT

Section 4.4 in the main text shows that, for the Base-scale models, the values of $\mathbf { W } _ { V }$ and $\mathbf { W } _ { O }$ in the AttenFeed module lie in an intermediate regime between the corresponding parameters of standard attention $( \mathbf { W } _ { V } , \mathbf { W } _ { O } )$ and those of the FFN $( \mathbf { W } _ { \mathrm { i n } } , \mathbf { W } _ { \mathrm { o u t } } )$ in ViT. Here, we examine whether this transitional behavior also holds at the Small and Large scales. The layer selection, plotting procedure, and normalization are kept identical to the main-text protocol; the only difference is the mode scale. The results are shown in the two figures below, corresponding to the Small and Large configurations, respectively.

At both scales, the uViT weight distributions are systematically wider than those of ViT’s attention parameters $( \mathbf { W } _ { V } , \mathbf { W } _ { O } )$ yet narrower than those of ViT’s FFN parameters $( \mathbf { W } _ { \mathrm { i n } } , \mathbf { W } _ { \mathrm { o u t } } )$ , reproducing the intermediate-regime pattern observed for the Base model. The consistency of this pattern across Small, Base, and Large models indicates that the unified character of the AttenFeed module is a scale-agnostic property of the architecture rather than an artifact of a specific capacity. This provides additional empirical support for our claim that the AttenFeed module simultaneously inherits the parametric tendencies of both the attention and FFN components of a standard Transformer.

![](images/27347279f7c94163f3fa14ba26f779317477fcf6c0a29bc49e9f7fade002c247.jpg)

![](images/8193298b29978b41dd67525356d888a7065615c9abec30a76957e6f0260624be.jpg)

(a) Mean, Std Comparison of $\mathbf { W } _ { V }$ and ${ \bf { W _ { i n } } }$  
![](images/a6022ef1fff5e3a06148322c1ebce9dcc7e1d41b25c3127f9f97b2068c7dca39.jpg)

![](images/6e14cb241cb6fe5f80f646964913243ee27eddfb6a79ae46a3fc49b0ae5cdb64.jpg)

![](images/ee5b5f49e4d8aa25bf1afa076274b241c6f60b02bc4b9a9277e907c75fa98196.jpg)

![](images/ca9b33f04b792d5052bd53499354254f2898b4f7f36d8fcd8e35e0ac6117092a.jpg)

![](images/cad4c2334d4b4255c9c91961ab430086f51bc010007ebabaa25d6102a21fd0a6.jpg)  
(b) Mean, Std Comparison of ${ \bf W } _ { O }$ and $\mathbf { W _ { o u t } }$

![](images/69add8bf6679b77e88ad93239ebd7fc6ed23a3c5fd26703532b3e70305bd80a6.jpg)  
Figure 5. Weight value distributions of uViT-S and ViT-S. The results show that uViT parameters occupy a transitional space between standard attention and FFN regimes at the Small scale, providing empirical evidence for the unified nature of the AttenFeed module.

![](images/aa922ad6373d206a67bb1eafdb7e8443cd14d3244df442c2419c69deec4d468f.jpg)

![](images/07bf70dee5bd7e76531ab386c60da21bb1e599b807ce44bf9832787dd4879e78.jpg)

![](images/92c37227ca98298d71e339b184b814516396187e7327eae5f7d7c56eeb7a3776.jpg)  
(a) Mean, Std Comparison of $\mathbf { W } _ { V }$ and ${ \bf { W _ { i n } } }$

![](images/bcd4530f3a4a4b24e96a2ef51ec2cc39246331739ddd1458af2612b2c8550030.jpg)

![](images/9ddb8d05a5e2e8e5ccb6c14a7cb626ae2bd015b052281037b463ce6b7b3646f7.jpg)

![](images/09cb1110c5ef41a3715c7d376a01e0a7723eb8eb51fa98b731adc27b9cb33183.jpg)

![](images/b55d64174ce24303523443477bf39908a5860dca194f16301336f50a570a5b0a.jpg)  
(b) Mean, Std Comparison of ${ \bf W } _ { O }$ and $\mathbf { W _ { o u t } }$

![](images/dbab035a03bca86c482087659d1e66e45977254d3d3a516de45671062a9c6940.jpg)  
Figure 6. Weight value distributions of uViT-L and ViT-L. The same transitional pattern holds at the Large scale, indicating that the unified nature of the AttenFeed module is not scale-specific.

## D. Limitation

While this work provides theoretical and empirical insights into the Attention-FFN dichotomy through the proposed AttenFeed module and uViT architecture, we acknowledge a few limitations that warrant future investigation.

First, our empirical analysis is currently confined to the computer vision domain. Although the Transformer was originally designed for natural language processing and has since become a universal architecture across various modalities, our experiments strictly evaluate Vision Transformers (ViTs) on image datasets. The inductive biases imposed by the Attention-FFN separation may manifest differently in other domains, such as language modeling or audio processing. Validating the necessity of the Attention-FFN dichotomy and the efficacy of the AttenFeed module in these non-vision domains remains an important direction for future work.

Second, to ensure a strictly fair comparison between uViT and the standard ViT, all models in our experiments were trained under identical conditions using a standard ViT training recipe. However, since uViT fundamentally alters the conventional network structure by unifying Attention and FFN into a single sequence of AttenFeed modules, it may possess a distinct optimal optimization landscape. Developing a specialized training recipe tailored to uViT—such as exploring different regularization techniques, learning rate schedules, or data augmentations—could potentially unlock further performance gains and alter the scaling behaviors observed in our current findings.