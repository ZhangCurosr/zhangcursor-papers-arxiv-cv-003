# MCL: Meta Convolution Layer

Naim Reza, Md Al Amin, and Ho Yub Jung

Abstract—Dynamic convolution enhances convolutional neural networks (CNNs) by adapting kernels to input content, but it expresses the effective kernel as a linear mixture of a small number of basis kernels, which limits expressivity and complicates optimization as the mixture size grows. In this work, we revisit dynamic convolution from a functional perspective and propose the Meta Convolution Layer (MCL), which directly models the convolutional kernel as an input-conditioned function W(x) realized via a high-order polynomial expansion. Leveraging nested residual blocks inspired by deep polynomial networks, MCL implements a structured polynomial meta-network that generates a single input-adaptive kernel, thereby decoupling representational power from the explicit number of mixture kernels and alleviating training instability. MCL is a plug-in addition with standard convolutions and can be seamlessly integrated into both CNN and transformer backbones. Experimental evaluation shows that adding MCL improves the Top-1 accuracy of Resnet-18, Resnet-50 and ResNet-101 by 6.61%, 3.42% and 3.05% on the ImageNet dataset. Moreover, the proposed method significantly boosts the accuracy of Resnet and Wide-Resnet variants on CIFAR-10 and CIFAR-100 datasets. Additionally, the proposed method outperforms previous methods on fine-grained visual classification tasks using Swin and ViT backbones. These results demonstrate that high-order polynomial kernel generation is a powerful and scalable alternative to linear mixture based dynamic convolution.

Index Terms—Meta Convolution Layer, Dynamic Convolution, Input-adaptive Kernels, Polynomial Networks, CNNs, Transformers, Image Classification.

## I. INTRODUCTION

Deep convolutional neural networks (CNNs) have achieved remarkable success across a wide range of visual recognition tasks, including large-scale image classification, object detection and fine-grained recognition [1]–[3]. This success is largely attributed to the representational power of convolutional layers, which learn hierarchical feature representations by repeatedly applying spatially local, channel-wise shared kernels to input feature maps. In the standard design, each convolutional layer uses a single static kernel that is shared across all input samples, and the expressiveness of the model is mainly increased by stacking more layers or widening the network. While effective, such static kernels cannot adapt to the diverse and input-dependent patterns present in natural images.

To enhance the adaptability of CNNs, a large body of work has focused on feature recalibration through attention mechanisms. Channel attention and spatial attention modules, such as squeeze-and-excitation (SE) blocks [4], convolutional block attention modules (CBAM) [5], efficient channel attention (ECA) [6], and related variants, learn input-dependent importance weights to re-scale feature channels or spatial locations, thereby emphasizing informative responses while suppressing less relevant ones. These mechanisms significantly improve the performance of baseline CNNs with modest parameter overhead. However, they operate on feature activations while keeping the convolutional kernels themselves fixed, and thus cannot fully exploit the potential of input-adaptive kernel generation.

Dynamic convolution takes a more direct approach by learning input-dependent kernels. Instead of using a single static kernel per layer, dynamic convolution approximates the effective kernel as a linear mixture of n candidate kernels, each modulated by an attention coefficient predicted from the input [7], [8]. This design enables the network to instantiate different effective kernels for different inputs, improving the representation capacity especially for lightweight models. Subsequent works generalize this idea along different dimensions of the kernel space. For example, CondConv [9] and DY-Conv [7] learn input-conditioned mixtures of multiple kernels, while Omni-dimensional Dynamic Convolution (ODConv) [10] extends the dynamic mechanism to simultaneously modulate kernel number, spatial size, input channel and output channel dimensions. KernelWarehouse [11] further rethinks dynamic convolution by partitioning kernels into a shared “warehouse” and assembling them with learnable attention, enabling parameter sharing across layers and larger effective kernel sets under a fixed parameter budget.

Despite these advances, existing dynamic convolution methods share a common limitation, the effective kernel is still expressed as a linear combination of a finite set of basis kernels. In practice, the number of kernels n is often constrained to a small value (typically n < 10) to control parameter growth and computational cost. Increasing n improves expressivity but it also scales parameters and FLOPs linearly, and makes optimization difficult, often requiring careful temperature scheduling for the attention softmax and specialized initialization schemes to stabilize training [7], [8], [11]. Even with advanced designs such as KernelWarehouse [11] or ODConv [10], the underlying combination rule remains linear, which may limit the ability to capture high-order, input-dependent interactions in large and wide networks.

In parallel, polynomial neural networks [12] and related functional approximators [13] have shown that high-order polynomial expansions can provide powerful and compact representations when implemented with appropriate structure. Deep Polynomial Networks (Π-Nets) [12] introduce nested coupled polynomial (NCP) and NCP-Skip decompositions to realize $\bar { 2 } ^ { N }$ -th order polynomial expansions using only N residual-like blocks, providing a principled way to model rich interactions with controlled parameter growth. These works suggest that structured polynomial expansions, when embedded in residual architectures, can approximate complex functions efficiently and remain amenable to gradient-based optimization.

Motivated by these studies, we revisit dynamic convolution from a functional perspective and consider the convolutional kernel as an explicit input-conditioned function W(x) instead of a linear mixture of a few static kernels. We propose Meta Convolution Layer (MCL), which models W(x) via a high-order polynomial expansion implemented through nested residual blocks following the NCP-Skip decomposition. Instead of aggregating n kernels with attention, MCL uses a polynomial meta-network to directly generate a single inputadaptive kernel that captures high-order interactions in the feature space. This design mitigates the difficulty of jointly optimizing many kernels, decouples expressivity from the explicit kernel count $n ,$ and naturally scales to deeper and wider backbones.

At a high level, MCL bridges dynamic convolution and deep polynomial networks. It preserves the plug-and-play nature of dynamic convolution modules while replacing linear mixtures with structured polynomial expansions of the kernel. We validate MCL across multiple regimes, including CIFAR [14], ImageNet [15], and fine-grained visual classification (FGVC) benchmarks with both CNN and transformer backbones, demonstrating consistent improvements over strong baselines and existing dynamic convolution methods.

The main contributions of this work are summarized as follows:

• We propose the Meta Convolution Layer (MCL), which directly estimates the convolution kernel via a high-order polynomial expansion of input features and generates a single input-adaptive kernel.

• We show that the proposed MCL with a polynomial adaptive kernel is easy to train without hyperparameter tuning. This is the main advantage over the previous attentionbased linear kernel mixtures where softmax temperature (τ) scheduling is required for stable optimization.

• We demonstrate that MCL can be seamlessly integrated into standard CNN backbones and transformer architectures, consistently improving performance on CIFAR, ImageNet, and multiple FGVC benchmarks, while offering favorable accuracy–parameter trade-offs compared to prior dynamic and attention-based modules.

• We provide empirical analyses on polynomial order via kernel size and meta-block depth, parameter efficiency, and computational cost, highlighting the scalability and practical applicability of MCL as a general plug-in module alongside static convolutions.

## II. RELATED WORKS

a) Dynamic Convolution and Attention over Kernels.: Dynamic convolution generalizes standard convolution by making the kernel input-dependent. Early approaches such as Dynamic Convolution [7] and CondConv [9] construct the effective kernel as a linear combination of multiple candidate kernels, where the mixture weights are produced by a lightweight attention function conditioned on the input feature map. DY-Conv [8] further refines this paradigm by adopting more flexible attention formulations and regularization strategies to improve training stability. These designs have been particularly effective for compact architectures such as MobileNetV2 [2], [16], [17] and shallow ResNet [18] variants, achieving improved accuracy with moderate increases in parameters and FLOPs.

Subsequent research extends kernel attention to richer factorized spaces. ODConv [10] introduces omni-dimensional attentions along kernel number, spatial size, input channels, and output channels, enabling more expressive modulation and yielding consistent improvements on large-scale benchmarks such as ImageNet [15] and MS-COCO [19]. Complementary analyses revisit dynamic convolution through matrix decomposition to study low-rank structure, parameter efficiency, and regularization effects [8]. KernelWarehouse (KW) [11] further decouples kernel construction from kernel usage by organizing kernels into a shared warehouse of kernel cells and assembling layer-specific kernels via input-dependent scalar attentions, allowing the effective mixture size to scale beyond the typical n < 10 regime under a fixed parameter budget. Nevertheless, KW and related methods still rely on linear mixtures of basis elements and often require careful initialization and temperature scheduling to maintain stable attention distributions during optimization.

b) Feature Recalibration and Attention in CNNs.: Feature recalibration modules improve CNNs by modulating activations while keeping kernels fixed. Squeeze-and-Excitation (SE) [4] introduces channel-wise attention via global pooling and a lightweight bottleneck to re-scale channels, while CBAM [5] augments this with sequential channel and spatial attention. ECA [6] simplifies channel attention using local cross-channel interactions without dimensionality reduction. Related variants such as CGC and WeightNet explore alternative parameterizations and normalization strategies for attention over channels or weights [20], [21]. Although these modules are complementary to dynamic convolution, they do not explicitly generate input-conditioned kernels. In contrast, our approach targets kernel generation itself: rather than enriching feature responses with fixed kernels, we directly increase kernel expressivity via an explicit functional form.

c) Polynomial Neural Networks and Dynamic Parameterization.: Polynomial neural networks offer a principled mechanism to capture high-order interactions. Deep Polynomial Networks (Π-Nets) address the combinatorial growth of classical polynomial models by using nested coupled polynomial (NCP) and NCP-Skip decompositions, realizing $2 ^ { \hat { N } }$ -th order expansions with only N residual-like modules [12]. Building on these insights, we design MCL to generate convolutional kernels through NCP-Skip style recursions in the kernel space, where the meta-network maps the current feature map to the kernel weights $\mathbf { W } ( x )$ . This contrasts with standard dynamic convolution, which constrains $\mathbf { W } ( x )$ to linear mixtures of a few kernels, and relates to broader dynamic parameterization schemes such as HyperNetworks and one-shot weight generation [22], [23].

Beyond generating a single kernel, some other work goes even further. These works represent things like images or scenes as continuous functions, instead of storing them as fixed grids of values. Implicit neural representations (INRs) do this by treating images, videos, or 3D scenes as functions that map coordinates to values, as done in Neural Radiance Fields [24] and SIREN [25]. Recent work has also applied this idea to image classification, closing some of the gap with regular pixel-based methods [26]. INRs and MCL are similar in one way. Both replace a fixed, explicit representation with a function that is evaluated only when needed. But they differ in what that function takes as input. INRs learn a function of spatial coordinates, built for one specific scene. MCL instead learns a function of the input feature map itself, and this function produces a single kernel that is shared across every spatial position in that input. HyperNetwork research has also grown beyond the original per-layer embedding approach of Ha et al. [22]. Some newer work uses graph-based hypernetworks that can generate weights for many different target architectures [27]. Other work uses transformer-based hypernetworks that condition on a small support set of examples for fewshot learning [28]. More recently, researchers have shown that pretrained foundation models can further improve how transformer-based hypernetworks generate weights [29]. These approaches share MCL’s goal of avoiding a large explicit weight tensor, but typically generate weights for entire layers or networks from a compact conditioning signal, whereas MCL restricts weight generation to a single convolutional kernel conditioned directly on the local input feature map.

d) Transformers and Fine-Grained Visual Recognition.: Transformer backbones such as ViT and Swin become dominant for fine-grained recognition [30], [31], recent FGVC methods incorporate specialized token or part-based mechanisms [32]–[34]. Our experiments integrate MCL into ViT and Swin for FGVC datasets and compare against representative approaches (e.g., IELT, MP-FGVC, ACC-ViT, MPSA) [32]–[35], demonstrating that polynomially generated kernels provide consistent gains beyond purely convolutional settings.

Adaptive mechanisms are also central to Vision Transformers themselves. Self-attention generates input-dependent mixing weights over tokens through a softmax, which, similar to dynamic convolution, expresses its output as a linear combination of existing elements rather than as a directly generated function of the input. Other adaptive components used in recent ViT and Swin variants, such as dynamic position bias and lightweight adapter modules, similarly adjust existing weights or biases based on the input rather than generating a new kernel directly. MCL differs from these mechanisms in that it produces an explicit input-dependent convolutional kernel through a polynomial expansion, rather than reweighting a fixed set of tokens or parameters. In our experiments, MCL is inserted as an additional module alongside standard self-attention layers in Swin-B and ViT-B/16, indicating that polynomial kernel generation is complementary to, rather than a replacement for, transformer-native adaptive mechanisms.

## III. METHODOLOGY

In this section, we will outline dynamic convolution and then describe the proposed meta convolution method in detail.

## A. Motivation

For a convolution layer, let $\boldsymbol { x } \in \mathbb { R } ^ { c \times h \times w }$ and $y \in \mathbb { R } ^ { f \times h } \mathrm { ~ } ;$ ×w having c feature channels and $f$ output channels respectively, where h×w denotes the feature size. A traditional convolution, $y = \mathbf { W } \otimes x ,$ , uses a static kernel $\mathbf { W } \in \mathbb { R } ^ { f \times c \times k \times k }$ consisting of f filters of spatial size $k \ \times \ k .$ . Dynamic convolution [7], [8] approximates this W using aggregation of a linear mixture of $n$ number of same sized kernels, $\mathbf { W _ { 1 } } , \ldots , \mathbf { W _ { n } } ,$ weighted by n number of input-dependent attention scores, $\alpha _ { 1 } , \ldots , \alpha _ { n }$ , generated by an attention module $\phi ( x )$ . Thus, dynamic convolution is formulated as follows:

$$
y = \sum _ { i = 1 } ^ { n } \alpha _ { i } \mathbf { W } _ { i } \otimes x + \sum _ { i = 1 } ^ { n } \alpha _ { i } \mathbf { b } _ { i }\tag{1}
$$

where, b is the bias vector and b $\in \mathbb { R } ^ { f }$ . Although, this nonlinear representation of the kernel can learn richer representation but it requires a large number of kernels to achieve this. Moreover, jointly optimizing many kernels is difficult and creates training instability by requiring near-uniform attention at the initial stage of training, which is facilitated by using a softmax annealing temperature τ as $\begin{array} { r } { \alpha _ { i } = \frac { e x p ( z _ { i } / \tau ) } { \sum _ { i } e x p ( z _ { j } / \tau ) } } \end{array}$ , where, $z _ { i }$ is the output of the final FC layer of the attention module $\phi ( x )$

These shortcomings motivate us to utilize a polynomial expansion to estimate a single kernel to learn high-dimensional representation rather than using multiple kernels. Two practical factors further distinguish MCL from prior dynamic convolution methods. First, MCL incorporates a normalizing scaling factor $\gamma$ in Eq. (11), which ensures stable variance at initialization and is absent in prior linear mixture approaches. Second, MCL omits softmax-weighted kernel aggregation, thereby avoiding the large zero-gradient regions that typically hinder optimization in both dynamic convolution and attention-based architectures.

## B. Meta kernel via Polynomial expansion

To address the limitations of dynamic convolution in jointly optimizing multiple kernels, we propose to estimate the convolution kernel W directly via a high-order polynomial expansion from input x, rather than as a weighted mixture of kernels. Specifically, we replace the linear mixture formulation of Eq. 1, with a formulation where W is a function of the input x, parameterized via a structured polynomial expansion.

Inspired by the Nested Coupled Polynomial (NCP) and NCP-Skip decomposition introduced in the Π-Net framework [12], we define $\mathbf { W } ( x )$ recursively through a set of latent features $\{ x _ { n } \} _ { n = 1 } ^ { N }$ encoding polynomial interactions up to order N. A sequence of recursive hidden states $x _ { n } \in \mathbb { R } ^ { k }$ can be defined as:

$$
x _ { 1 } = ( A _ { 1 } ^ { T } x ) \odot ( B _ { 1 } ^ { T } b _ { 1 } )\tag{2}
$$

$$
x _ { n } = \bigl ( A _ { n } ^ { T } x \bigr ) \odot \bigl ( S _ { n } ^ { T } x _ { n - 1 } + B _ { n } ^ { T } b _ { n } \bigr ) + x _ { n - 1 } ; \mathrm { f o r ~ n = 2 , . . . , N }\tag{3}
$$

where, $A _ { n } ~ \in ~ \mathbb { R } ^ { c \times k }$ projects the input to a latent space, $S _ { n } ~ \in ~ \mathbb { R } ^ { k \times k }$ captures recursive dependencies across orders, $B _ { n } \in \mathbb { R } ^ { \omega \times k }$ and $b _ { n } \in \mathbb { R } ^ { \omega }$ define the bias transformations and ⊙ denotes the Hadamard product. The matrix multiplication can represent various linear operations including convolutions. Note that all the components of Eq. (2) and Eq. (3) are differentiable and can be implemented within a residual block. Eq. (3) suggests that the output of a residual block would be a quadratic expansion of the polynomial. Stacking N such polynomials results in an overall $2 ^ { N }$ order of expansion. Therefore, we can simply stack N number of residual block sequentially to achieve our desired order of expansion of the polynomial. We refer the reader to [12] for more insights on this subject matter.

![](images/a764b7eb16e214be38661367fe1dddea9dc4ff1c7aee244a83ec359ec4118337.jpg)  
Fig. 1: Schematic diagram of the proposed MCL method.

Let x ∈ R<sup>c×h×w</sup> denote the input feature map and $\mathbf { W } ( x ) \in$ $\mathbb { R } ^ { f \times c \times k \times k }$ the generated convolution kernel. Eq. (2)–(3) define the polynomial recursion through linear projections, Hadamard products, and residual addition. In practice, we instantiate each linear map $( \mathbf { e } . \mathbf { g } . , A _ { n } ^ { \top } ( \cdot )$ and $S _ { n } ^ { \top } ( \cdot ) )$ with a standard Conv–BN– ReLU operator, as shown in Fig. 2. Concretely, for order $n ,$ we implement the projections in Eq. (2)–(3) as,

$$
\tilde { A } _ { n } ^ { \top } ( x ) \triangleq \sigma ( \mathrm { B N } ( \mathrm { C o n v } _ { A _ { n } } ( x ) ) ) ,\tag{4}
$$

$$
\tilde { S } _ { n } ^ { \top } ( x _ { n - 1 } ) \triangleq \sigma ( \mathrm { B N } ( \mathrm { C o n v } _ { S _ { n } } ( x _ { n - 1 } ) ) ) ,\tag{5}
$$

$$
\tilde { B } _ { n } ^ { \top } ( b _ { n } ) \triangleq \sigma ( \mathrm { B N } ( \mathrm { C o n v } _ { B _ { n } } ( b _ { n } ) ) )\tag{6}
$$

where $\sigma ( \cdot )$ denotes ReLU. Substituting these into Eq. (2)–(3) yields the implementation-level recursion

$$
x _ { 1 } = \tilde { A } _ { 1 } ^ { \top } ( x ) \odot \tilde { B } _ { 1 } ^ { \top } ( b _ { 1 } ) ,\tag{7}
$$

$$
x _ { n } = \tilde { A } _ { n } ^ { \top } ( x ) \odot \ \left( \tilde { S } _ { n } ^ { \top } ( x _ { n - 1 } ) + \tilde { B } _ { n } ^ { \top } ( b _ { n } ) \right) + x _ { n - 1 }\tag{8}
$$

for $n = 2 \ldots N _ { : }$ , this formulation of the meta-blocks realizes the same multiplicative and residual topology as Eq. (2)– (3), while each “matrix” operator is implemented by a small convolutional sub-network. Importantly, if $\sigma$ is replaced by the identity, the implementation reduces exactly to the pure polynomial form in Eq. (2)–(3).

After stacking N meta-blocks, the final state $x _ { N }$ is linearly projected and reshaped to realize the final meta-polynomial kernel,

$$
\mathbf { W } ( x ) = { \mathrm { r e s h a p e } } ( C x _ { N } + \beta )\tag{9}
$$

where, $C$ and $\beta$ are linear transformations. This formulation estimates W as a single input-adaptive kernel capturing high-order interactions via nested polynomial transformations, providing an efficient and expressive alternative to traditional dynamic convolutions. With ReLU activation retained in Eq. (4)–(6), W(x) becomes a piecewise-polynomial function that makes polynomial interactions gated by piecewiselinear activations. This hybrid design is consistent with prior polynomial-network practice that explicitly study settings both with and without activations, and report that keeping standard nonlinearities can mitigate overfitting and stabilize training [12], [13]. All projection operators, $\{ A _ { n } , S _ { n } , B _ { n } , b _ { n } \} _ { n = 1 } ^ { N }$ that parameterize the meta-polynomial recursion, and $C , \beta$ that parameterize the final kernel projection are learnable.

## C. Meta Convolution Layer

Given the input feature x and its associated meta–polynomial kernel $\mathbf { W } ( x )$ from Eq. (9), we compute a normalizing scaling factor γ as follows, analogous to Xavier and He-initialization [36], [37], to ensure equal input and output variance at initialization:

$$
\gamma = \sqrt { \frac { 2 } { c \times k ^ { 2 } + f } }\tag{10}
$$

where, $c , k ,$ and $f$ denote the input channel size, metakernel size, and output channel size, respectively. Note that $\gamma$ is a fixed architectural constant fully determined by these dimensions. Omitting this normalization can cause training instability, as the unnormalized meta-kernel outputs would have variance that scales with the network dimensions.

$$
y = \gamma \mathbf { W } ( x ) \otimes x + \mathbf { B }\tag{11}
$$

where $\mathbf { B } \in \mathbb { R } ^ { f }$ is a learnable bias, and $\otimes$ denotes convolution. Following this, we use batch normalization and activation function like traditional convolution as shown in Fig. 1.

![](images/3d63b4a4acac595fdd4983ba1819ea2b14b41e6427cf7090fc334c906a19c3fe.jpg)  
Fig. 2: Schematic Diagram of (a) Basic Meta-Block and (b) Compact Meta-Block. The bias term is omitted for clarity.

This formulation is consistent with the dynamic convolution notation in Eq. (1), but with a single input-dependent kernel $\mathbf { W } ( x )$ generated by a polynomial meta-network instead of a mixture of multiple static kernels. The learnable parameters are optimized end-to-end with the backbone network and thus easy to optimize.

## D. Optimization Behavior and Stability

Here, we discuss why MCL remains stable to optimize despite generating a high-order polynomial kernel.

The scaling factor γ in Eq. (10) serves the same purpose as Xavier and He initialization for standard convolution. Without γ, the variance of $\mathbf { W } ( x )$ grows with the product of input channels, kernel size, and output channels, so the gradient magnitude at the convolution in Eq. (11) would scale with these dimensions and destabilize training at initialization, particularly for larger meta-kernel sizes k. Fixing γ as in Eq. (10) removes this dependency and keeps the expected output variance constant across layer widths. Each meta-block in Eq. (7)-(8) also adds the previous hidden state back onto its transformed output, the same way a residual block does. Because of this, the gradient going back to each hidden state always has a direct, unblocked path, no matter how the multiplicative Hadamard term inside the block behaves. This is why the polynomial order can grow as $2 ^ { N }$ while the number of steps only grows linearly with N. Stacking more meta-blocks does not make the vanishing or exploding gradient problem worse, unlike typical deep multiplicative chains.

Finally, the attention mixture in Eq. (1) used by prior dynamic convolution methods needs a softmax over the kernel logits. When this softmax becomes close to one-hot, the gradient for the unused kernels drops to nearly zero, so those kernels stop learning. MCL does not use softmax over kernels, so this problem does not happen. Instead, the ReLU activations in Eq. (4)-(6) pass each polynomial term through in a piecewiselinear way, so a gradient is either passed through fully or blocked, not slowly reduced to zero like with softmax. This matches the stable training we see in the standard deviations reported in Table I and the convergence curves in the appendix.

## E. Implementation

Unlike prior dynamic convolution approaches that replace the $3 \times 3$ operators throughout the backbone, we integrate only one MCL at the end of the final feature-extraction stage. Since stacking N meta-blocks within a single MCL results in a $2 ^ { N }$ -order polynomial expansion, a single MCL suffices to model high-order interactions while keeping the backbone unchanged. The output of MCL is then forwarded to the final fully connected classifier layer for prediction. The plug-andplay property of MCL means that it does not itself introduce training instability when attached to any backbone. MCL is not designed to resolve pre-existing optimization fragility in the host architecture, such as the difficulty of training ViTs from scratch on small datasets.

To instantiate each meta-block, we adopt the residual-style configuration in Fig. 2(a) and denote the MCL kernel size by k and the polynomial expansion order by n. While larger k and higher-order expansions $( n > 1 )$ improve accuracy, they also increase parameter cost as later shown in appendix A. To control this overhead, we employ the compact meta-block in Fig. 2(b), where the expensive $3 \times 3$ operation is replaced with 1 × 1 convolutions. In all experiments, we set $n = 2$ (two meta-blocks), yielding a $4 ^ { \mathrm { t h } }$ -order expansion, and use symmetric meta-kernels (e.g. 3 × 3 and $5 \times 5 )$ . To preserve the same spatial resolution of the backbone feature at the output of MCL, we fix stride to 1 and set padding to $0 / 1 / 2$ for $k = 1 / 3 / 5$ , respectively.

## IV. EXPERIMENTS

We evaluate the proposed Meta Convolution Layer (MCL) on CIFAR [14], ImageNet-1K [15], and four fine-grained visual classification (FGVC) benchmarks to assess accuracy gains, scalability, and transferability.

## A. Image Classification on CIFAR Dataset

We study MCL on CIFAR-10/100 [14] using ResNet-20/56/110 [18] and WideResNet-16-10/28-10 [38]. All models are trained for 200 epochs with SGD (momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 } )$ , batch size 128, and a cosine learning-rate schedule from 0.1. We apply standard CIFAR augmentation, random crop with 4-pixel padding, horizontal flip, normalization, and additionally report results with MixUp [39] as stronger data augmentation. Unless stated otherwise, we set the polynomial order to n = 2 and vary the meta-kernel size k, where, k followed by (b) denotes the compact meta-block variant.

TABLE I: Classification accuracy (%) on CIFAR-10 and CIFAR-100 using ResNet and WideResNet backbones with and without the proposed MCL. We report Top-1 accuracy ± STD of 3 independent runs for different meta-kernel sizes k and training settings.
<table><tr><td rowspan="2">Model</td><td colspan="6"></td><td colspan="6"></td></tr><tr><td>Baseline  $\mathtt { B a s e l i n e + M C L }$ </td><td></td><td> ${ \mathrm { B a s e l i n e } } + { \mathrm { A u g } }$  一</td><td></td><td> $\overline { { { \bf B a s e l i n e + A u g + M C L } } }$ </td><td></td><td>Baseline</td><td> $_ \mathrm { B a s e l i n e + M C L }$ </td><td> $\mathrm { B a s e l i n e + A u g }$ </td><td></td><td> $\overline { { \mathrm { { B a s e l i n e + A u g + M C L } } } }$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>k=3</td><td>k=3(b)</td><td> $k { = } 5 ( b )$ </td><td></td><td></td><td></td><td> $k { = } 1$ </td><td>k=3</td><td> $\scriptstyle { k = 3 ( b ) }$ </td><td> $k { = } 5 ( b )$ </td></tr><tr><td>ResNet20</td><td>91.43</td><td> $\overline { { 9 3 . 5 2 \pm 0 . 0 7 } }$ </td><td>92.33</td><td> $9 4 . 5 3 \pm 0 . 1 8$ </td><td> $\mathbf { 9 5 . 3 2 \pm 0 . 0 8 }$   $9 4 . 6 6 \pm 0 . 0 9$ </td><td> $9 5 . 2 0 \pm 0 . 0 5$ </td><td>69.95</td><td> $\overline { { 7 0 . 4 1 \pm 0 . 0 5 } }$ </td><td>68.89</td><td> $7 2 . 8 3 \pm 0 . 1 9$ </td><td> $7 7 . 9 7 \pm 0 . 2 3$ </td><td> $\overline { { 7 7 . 3 2 \pm 0 . 2 7 } }$ </td><td> $\overline { { { \bf 7 8 . 5 8 \pm 0 . 2 3 } } }$ </td></tr><tr><td>ResNet56</td><td>93.03</td><td> $9 4 . 3 2 \pm 0 . 0 8$ </td><td>94.31</td><td> $9 5 . 8 1 \pm 0 . 1 3$   $9 6 . 1 6 \pm 0 . 0 0$ </td><td> $9 6 . 0 9 \pm 0 . 1 2$   $9 5 . 7 7 \pm 0 . 0 4$ </td><td> ${ \bf 9 6 . 1 2 \pm 0 . 0 8 }$ </td><td>71.97</td><td> $7 3 . 5 4 \pm 0 . 2 4$ </td><td>75.26</td><td>78.26 ± 0.16</td><td> $8 0 . 3 7 \pm 0 . 0 8$ </td><td>79.66 ± 0.18</td><td> $\mathbf {8 . 2 5 \pm 0 . 0 5 }$ </td></tr><tr><td>ResNet110 WideResNet16-10</td><td>93.57 95.44</td><td> $9 4 . 7 2 \pm 0 . 1 8$   $9 6 . 0 1 \pm 0 . 0 4$ </td><td>94.90 96.80</td><td> $9 7 . 7 1 \pm 0 . 0 0$ </td><td> $\mathbf { 9 6 . 5 7 \pm 0 . 0 6 }$   $9 6 . 3 2 \pm 0 . 0 7$   $9 7 . 7 8 \pm 0 . 0 4$ </td><td> $9 6 . 4 3 \pm 0 . 1 5$   $\mathbf { 9 7 . 8 0 \pm 0 . 0 7 }$ </td><td>74.80 78.41</td><td> $7 5 . 0 0 \pm 0 . 1 8$   $7 9 . 9 6 \pm 0 . 2 6$ </td><td>78.11 82.86</td><td> $7 9 . 7 3 \pm 0 . 1 4$   $8 4 . 6 2 \pm 0 . 0 8$ </td><td> $8 1 . 8 1 \pm 0 . 1 9$   $8 5 . 1 0 \pm 0 . 2 5$ </td><td> $8 0 . 9 3 \pm 0 . 1 1$   ${ \bf 8 5 . 6 2 \pm 0 . 1 1 }$ </td><td> $8 5 . 4 4 \pm 0 . 0 4$ </td></tr><tr><td>WideResNet28-10</td><td>96.00</td><td> $9 6 . 2 5 \pm 0 . 0 2$ </td><td>97.41</td><td> $\mathbf { 9 8 . 0 5 \ : \pm 0 . 1 1 }$ </td><td> $9 7 . 9 3 \pm 0 . 0 1$ </td><td> $9 7 . 6 5 \pm 0 . 0 3$   $9 7 . 9 8 \pm 0 . 0 3$   $9 7 . 9 8 \pm 0 . 1 0$ </td><td>80.75</td><td> $8 0 . 8 7 \pm 0 . 2 3$ </td><td>83.50</td><td> $8 5 . 2 6 \pm 0 . 0 9$ </td><td> $8 6 . 0 0 \pm 0 . 1 3$ </td><td> $8 5 . 9 0 \pm 0 . 0 3$ </td><td> $\mathbf { 8 6 . 2 0 \overset { - } { \pm } 0 . 0 9 }$ </td></tr></table>

TABLE II: Performance comparison on the ImageNet dataset using ResNet18 backbone.

![](images/cd4fc06d4c98098a84078c4fe41fcf35873f752ff5597380cd761d8eb0126a86.jpg)  
Fig. 3: This figure illustrates the comparison between accuracy vs parameters (M) for varying meta-kernel sizes of MCL on the CIFAR-100 dataset.

<table><tr><td>Models</td><td>Params</td><td>Top-1 (%)</td><td>Top-5 (%)</td></tr><tr><td>ResNet18</td><td>11.69M</td><td>70.25</td><td>89.38</td></tr><tr><td>+ SE</td><td>11.78M</td><td>70.98 (↑0.73)</td><td>90.03 (↑0.65)</td></tr><tr><td>+ CBAM</td><td>11.78M</td><td>71.01 (↑0.76)</td><td>89.85 (↑0.47)</td></tr><tr><td>+ ECA</td><td>11.69M</td><td>70.60 (↑0.35)</td><td>89.68 (↑0.30)</td></tr><tr><td>+ CGC</td><td>11.69M</td><td>71.60 (↑1.35)</td><td>90.35 (↑0.97)</td></tr><tr><td>+ WeightNet</td><td>11.93M</td><td>71.56 (↑1.31)</td><td>90.38 (↑1.00)</td></tr><tr><td>+ DCD</td><td>14.70M</td><td>72.33 (↑2.08)</td><td>90.65 (↑1.27)</td></tr><tr><td>+ CondConv (8×)</td><td>81.35M</td><td>71.99 (↑1.74)</td><td>90.27 (↑0.89)</td></tr><tr><td>+ DY-Conv (4×)</td><td>45.47M</td><td>72.76 (↑2.51)</td><td>90.79 (↑1.41)</td></tr><tr><td>+ ODConv (4×)</td><td>44.90M</td><td>73.97 (↑3.72)</td><td>91.35 (↑1.97)</td></tr><tr><td>+ KW (4×)</td><td>45.86M</td><td>74.16 (↑3.91)</td><td>91.42 (↑2.04)</td></tr><tr><td> $+ { \mathrm { ~ M C L ~ } } ( k = 1 )$ </td><td>21.13M</td><td>72.04 (↑1.79)</td><td>90.75 (↑1.37)</td></tr><tr><td> $+ { \bf \ M C L } \left( k = 3 \right)$ </td><td>36.63M</td><td>75.55 (↑5.30)</td><td>92.38 (↑3.00)</td></tr><tr><td>+ MCL (k = 5)</td><td>158.0M</td><td>76.86 (↑6.61)</td><td>92.98 (↑3.60)</td></tr></table>

a) Results.: Table I summarizes the results on CIFAR-10 and CIFAR-100 across different backbones and kernel sizes of MCL. Adding MCL on top of the baseline consistently improves performance in all configurations. On CIFAR-10, attaching MCL with $k = 3$ already yields noticeable gains. For example, the Top-1 accuracy of ResNet-20 increases from 91.43% to 93.52%, while ResNet-110 improves from 93.57% to 94.72%. When MCL is combined with stronger data augmentation (baseline + Aug + MCL), the gains are further amplified, achieving up to 2.61% points improvement on ResNet-20 and 2.59% points on ResNet-110.

The improvement is even more pronounced on CIFAR-100. For ResNet-20, incorporating MCL with full augmentation increases Top-1 accuracy from 69.95% to 78.58%, corresponding to an 8.63% points gain. WideResNet-28-10 exhibits a similar trend, with accuracy improving from 80.75% to 86.20% under the same setting. Across all backbones, MCL consistently yields strong gains on CIFAR-100, highlighting the benefit of high-order input-adaptive kernels in more challenging multi-class scenarios.

Figure 3 provides a comparison of accuracy versus parameter count for different meta-kernel sizes on CIFAR-100.

TABLE III: Performance comparison on ImageNet dataset using ResNet50 and ResNet101 backbone network.
<table><tr><td>Model</td><td colspan="3">ResNet50</td><td colspan="3">ResNet101</td></tr><tr><td></td><td>Params</td><td>Top-1 (%)</td><td>Top-5 (%)</td><td>Params</td><td>Top-1 (%)</td><td>Top-5 (%)</td></tr><tr><td>Baseline</td><td>25.56M</td><td>76.23</td><td>93.01</td><td>44.55M</td><td>77.41</td><td>93.67</td></tr><tr><td>+ SE</td><td>28.07M</td><td>77.31 (↑1.08)</td><td>93.63 (↑0.62)</td><td>49.29M</td><td>78.42 (↑1.01)</td><td>94.15 (↑0.48)</td></tr><tr><td>+ CBAM</td><td>28.07M</td><td>77.46 (↑1.23)</td><td>93.59 (↑0.58)</td><td>49.30M</td><td>78.50 (↑1.09)</td><td>94.20 (↑0.53)</td></tr><tr><td>+ ECA</td><td>25.56M</td><td>77.34 (↑1.11)</td><td>93.64 (↑0.63)</td><td>44.55M</td><td>78.60 (↑1.19)</td><td>94.34 (↑0.67)</td></tr><tr><td>+ ODConv</td><td>90.67M</td><td>78.52 (↑2.29)</td><td>94.01 (↑1.00)</td><td>90.44M</td><td>79.27 (↑1.86)</td><td>94.47 (↑0.80)</td></tr><tr><td>+ MCL (k = 1)</td><td>177.0M</td><td>79.25 (↑3.02)</td><td>94.44 (↑1.43)</td><td>196.0M</td><td>80.20 (↑2.79)</td><td>94.77 (↑1.10)</td></tr><tr><td>+ MCL (k = 3)</td><td>424.0M</td><td>79.65 (↑3.42)</td><td>94.46 (↑1.45)</td><td>443.0M</td><td>80.46 (↑3.05)</td><td>94.97 (↑1.30)</td></tr></table>

As k increases, MCL progressively improves accuracy while exhibiting a controllable increase in parameter count. This trade-off curve shows that MCL occupies a favorable region, moderate increases in model capacity (e.g., k = 3) already deliver substantial performance gains.

## B. Image Classification on ImageNet

We further evaluate scalability on ImageNet-1K [15]. We adopt ResNet-18/50/101 [18] and compare against representative feature recalibration modules (SE [4], CBAM [5], ECA [6]) and dynamic/kernel modulation baselines (CGC [20], WeightNet [21], DCD [8], CondConv [9], DY-Conv [7], ODConv [10], KW [11]). All models are trained for 100 epochs with SGD, using a mini-batch size of 256, momentum 0.9, and weight decay $5 ~ \times ~ 1 0 ^ { - 4 }$ The learning rate is initially set to 0.1 and decayed to zero with a cosine schedule over the course of training. Training images are resized to $2 5 6 ~ \times ~ 2 5 6 .$ followed by random 224 × 224 crops and random horizontal flips, as well as per-channel mean–std normalization. At evaluation time, we report single-crop Top-1 and Top-5 accuracy using the 224 × 224 center crop. Following previous dynamic convolution studies, we only employ MixUp [39] regularization and label-smoothing for ImageNet, in order to maintain clean and fair performance comparisons.

For MCL, we insert the meta-convolution layer before the fully-connected layer and the backbone network remains unchanged. We consider different meta-kernel sizes k with the polynomial expansion order $n = 2$ and we used the compact meta-block design on ImageNet to reduce the parameter cost.

a) Results.: Training stability is further evidenced by the standard deviations reported across 3 independent runs in Table I, and convergence curves for all evaluated backbones are provided in Appendix A.4. Table II reports ImageNet results using ResNet-18 [18] as the backbone. Standard attention and kernel modulation modules like SE [4], CBAM [5], ECA [6], CGC [20], WeightNet [21], DCD [8] provide moderate gains over the baseline, with improvements typically within 2% of Top-1 accuracy. Dynamic convolution approaches such as CondConv [9], DyConv [7], ODConv [10], and KW [11] yield larger improvements, with the best prior configuration (KW (4×)) achieving 74.16% Top-1 accuracy [11].

By contrast, MCL delivers substantially larger gains. With $k = 1$ , MCL already improves ResNet-18 from 70.25% to 72.04% Top-1 accuracy. Increasing the kernel size to $k = 3$ and k = 5 further boosts performance to 75.55% and 76.86%, respectively. The best configuration, MCL with $k \ = \ 5 ,$ achieves a 6.61% Top-1 improvement and a 3.60% Top-5 improvement over the baseline, surpassing all compared attention and dynamic convolution methods by a clear margin. The MCL configuration with k=3 reaches 75.55% with 36.63M parameters, comparable in cost to prior dynamic convolution baselines yet outperforming them by a clear margin. The configuration with k=5 is included to characterize the upper bound of the accuracy and parameter curve. The gain over a parameter-matched ResNet-50 at 76.23% with 25.6M parameters confirms that the improvement stems from the polynomial meta-kernel rather than parameter scaling alone. As shown in Figure 3, moderate kernel sizes already occupy a favorable accuracy and parameter region. These results indicate that directly learning a single high-order polynomial meta-kernel is more effective than aggregating multiple static kernels via attention, especially when the input resolution and category diversity are high. A detailed comparison of inference latency, throughput, and peak memory usage against prior dynamic convolution methods is provided in Appendix A.3.

Table III presents the ImageNet results on ResNet-50 and ResNet-101. Similar to the ResNet-18 case, adding SE, CBAM, or ECA to the baseline models yields modest 1% Top-1 gains. ODConv provides larger improvements, achieving 78.52% and 79.27% Top-1 accuracy on ResNet-50 and ResNet-101, respectively. MCL further improves over these strong baselines, with k = 1, it reaches 79.25% and 80.20%, and with $k \ = \ 3 ,$ it achieves 79.65% and 80.46% Top-1 accuracy for ResNet-50 and ResNet-101 respectively. In both backbones, MCL yields around 3% improvement over the vanilla ResNet and about 1% gain over ODConv, confirming that the proposed meta-polynomial kernel generalizes well

TABLE IV: Details of the FGVC datasets used in our experiments.
<table><tr><td>Dataset</td><td>Class</td><td>Train</td><td>Test</td></tr><tr><td>NABirds [40]</td><td>555</td><td>23929</td><td>24633</td></tr><tr><td>CUB-200-2011 [41]</td><td>200</td><td>5994</td><td>5794</td></tr><tr><td>Oxford 102 Flowers [42]</td><td>102</td><td>2040</td><td>6179</td></tr><tr><td>Oxford-IIIT Pet [43]</td><td>37</td><td>3680</td><td>3669</td></tr></table>

across deeper architectures.

While MCL k = 3 yields the largest accuracy gains reported in Table III, this comes at a substantial increase in parameter count relative to the baseline. We note that the meta-kernel size k is a controllable design choice rather than a fixed cost: $\mathbf { M C L } k \ = \ 1$ already recovers a large fraction of this improvement at a much smaller parameter overhead, as shown for ResNet-18 in Table II, where MCLk = 1 reaches 72.04% Top-1 accuracy with 21.13M parameters, compared to 158.0M parameters for $k = 5$ . Practitioners operating under tighter parameter budgets can therefore select $k \ = \ 1$ rather than k = 3 or k = 5. This accuracy-parameter trade-off is further characterized in Figure 3 and Table VII. Moreover, as shown in Table X (Appendix), the run-time cost of MCL, measured in latency and throughput, scales considerably more gently than its parameter count, since the additional parameters are concentrated in a single meta-network evaluated once per forward pass rather than distributed across repeated operations throughout the backbone.

## C. Fine-Grained Visual Classification with MCL

Following standard practice for FGVC, all transformer experiments fine-tune ImageNet-pretrained backbones. MCL consistently provides accuracy gains without introducing additional training instability, regardless of the host architecture. In contrast to Transformers, MCL demonstrates particular effectiveness on smaller datasets. We evaluate transfer learning on NABirds [40], CUB-200-2011 [41], Oxford-102 Flowers [42], and Oxford-IIIT Pets [43]. These datasets are widely adopted for FGVC evaluation with pretrained networks precisely because of their smaller scale, and they provide a useful testbed for demonstrating MCL’s effectiveness in dataconstrained scenarios where large architectures such as Vision Transformers struggle to train from scratch. We fine-tune ImageNet-pretrained Swin-B [31] and ViT-B/16 [30] with and without MCL, inserting a single module before the classifier.

Since Swin-B keeps its internal feature maps in a spatial, channel-last format like $H \times W \times C$ rather than a flattened token sequence, MCL’s convolutional operators are applied after permuting this tensor to a standard channel-first layout, and the output is permuted back before the classification head. For ViT-B/16, the backbone instead outputs a flat sequence of patch tokens rather than a spatial feature map. To apply MCL, which requires convolutional operators over a spatial input, the patch token sequence is reshaped into its corresponding spatial grid before being passed through MCL, and the resulting output is flattened back into a token sequence before the classification head. This is analogous to the permutation used for Swin-B, but requires reshaping from a flat sequence rather than permuting an already-spatial tensor.

TABLE V: Performance comparison of the proposed MCL on FGVC task using commonly used FGVC datasets.
<table><tr><td>Dataset</td><td>Method</td><td>Backbone</td><td>Accuracy (%)</td></tr><tr><td rowspan="5">NABirds</td><td>IELT</td><td>ViT-B/16</td><td>90.8</td></tr><tr><td>MP-FGVC</td><td>ViT-B/16</td><td>91.0</td></tr><tr><td>ACC-ViT</td><td>ViT-B/16</td><td>91.4</td></tr><tr><td>MPSA</td><td>Swin-B</td><td>92.5</td></tr><tr><td>MCL (k = 1)</td><td>Swin-B</td><td>92.8</td></tr><tr><td rowspan="5">CUB-200-2011</td><td>MP-FGVC</td><td>ViT-B/16</td><td>91.8</td></tr><tr><td>ACC-ViT</td><td>ViT-B/16</td><td>91.8</td></tr><tr><td>DACL</td><td>ResNet50-ViT</td><td>92.0</td></tr><tr><td>MPSA</td><td>Swin-B</td><td>92.8</td></tr><tr><td>MCL (k = 1)</td><td>Swin-B</td><td>92.7</td></tr><tr><td rowspan="5">Oxford 102 Flowers</td><td>InterAct</td><td>DenseNet-161</td><td>96.4</td></tr><tr><td>ViT</td><td>ViT-B/16</td><td>99.4</td></tr><tr><td>CvT</td><td>CvT-21</td><td>99.5</td></tr><tr><td>IELT</td><td>ViT-B/16</td><td>99.6</td></tr><tr><td>MCL (k = 1)</td><td>Swin-B</td><td>99.6</td></tr><tr><td rowspan="5">Oxford-IIIT Pet</td><td>InterAct</td><td></td><td></td></tr><tr><td></td><td>DenseNet-161</td><td>93.5</td></tr><tr><td>ViT CvT</td><td>ViT-B/16</td><td>93.8</td></tr><tr><td>IELT</td><td>CvT-21 ViT-B/16</td><td>94.7 95.2</td></tr><tr><td>MCL (k = 1)</td><td>Swin-B</td><td>95.9</td></tr></table>

TABLE VI: Performance of the proposed MCL across various FGVC benchmarks using different transformer-based architectures.
<table><tr><td>Backbone</td><td>Method</td><td>CUB</td><td>NABirds</td><td>Flowers</td><td>Pet</td></tr><tr><td rowspan="2">Swin-B</td><td>Baseline</td><td>91.93</td><td>91.99</td><td>99.41</td><td>95.09</td></tr><tr><td>MCL (k = 1) Improvement</td><td>92.72 0.79</td><td>92.83 0.84</td><td>99.64 0.24</td><td>95.91 0.82</td></tr><tr><td rowspan="2">ViT-B/16</td><td>Baseline</td><td>88.90</td><td></td><td></td><td></td></tr><tr><td>MCL (k = 1)</td><td></td><td>87.55</td><td>97.06</td><td>94.44</td></tr><tr><td rowspan="2"></td><td></td><td>90.35</td><td>89.38</td><td>99.17</td><td>94.55</td></tr><tr><td>Improvement</td><td>1.45</td><td>1.83</td><td>2.11</td><td>0.11</td></tr></table>

We follow the commonly adopted fine-tuning schedule for FGVC, using 5 epochs of warm-up followed by cosineannealed learning rates. The initial learning rate is set to $5 \times 1 0 ^ { - 4 }$ for all the datasets along with a SGD optimizer using momentum of 0.9 and a weight decay of $1 \times 1 0 ^ { - 4 }$ . The input image is first scaled up to the size of $5 1 0 \times 5 1 0$ for Swin Transformer and 550×550 for ViT. After that random cropping is applied to attain fixed-sized image of size 384×384 for Swin Transformer and 448 × 448 for ViT during training. Center crop is used at the test time to obtain the same sized images. Random flipping and random Gaussian blur are applied during the training process for all the datasets. All the models are trained for 50 epochs with a batch size of 16 for all the datasets presented in Table IV. Unless stated otherwise, we use k = 1 and n = 2.

a) Results on FGVC.: Tables V and VI summarize the performance of MCL on NABirds, CUB-200-2011, Oxford-102 Flowers and Oxford-IIIT Pets. When combined with Swin-B, MCL consistently improves over the strong Swin-B baselines on all four datasets. For instance, the Top-1 accuracy on CUB-200-2011 increases from 91.93% to 92.72%, and on NABirds from 91.99% to 92.83%, while on Oxford-102 Flowers and Oxford-IIIT Pets, MCL achieves 99.64% and 95.91%, respectively. Moreover, the proposed method demonstrates comparative or improved performance over stateof-the-art methods like IELT [33], MP-FGVC [35], MPSA [34], ACC-ViT [32], DACL [44], InterAct [45] and CvT [46]. These gains, although modest in absolute terms due to the high baseline performance, demonstrate that MCL remains effective even in regimes where the backbone already captures rich finegrained cues.

![](images/59dcfd032ff9c001963206a5e6eda9020e12f2d7f58114c67f5a3af526a3f187.jpg)  
Fig. 4: Visual comparison of the proposed method using Grad-Cam visualization.

On ViT-B/16, the improvements are even more pronounced. Across the four FGVC datasets, MCL yields gains of up to 2.11% on Oxford-102 Flowers over the baseline ViT-B/16. The average improvement across all datasets is substantial for both Swin-B and ViT-B/16, confirming that the meta-polynomial kernel can be seamlessly integrated into transformer-based architectures and provides consistent benefits in fine-grained recognition tasks.

b) Visual comparison.: To better understand the qualitative effect of MCL, we visualize class-discriminative regions using Grad-CAM [47] on representative samples from the four FGVC datasets. Although MCL and competing methods achieve similar quantitative performance on some datasets, the spatial activation patterns differ noticeably, suggesting that the polynomial meta-kernel produces qualitatively distinct feature responses even when top-1 accuracy margins are small. As illustrated in Figure 4, models equipped with MCL attend more tightly to object regions and capture semantically meaningful parts, while baselines without MCL often exhibit more diffuse or background-focused activations. For example, in bird and flower categories, MCL encourages the network to focus on characteristic patterns such as heads, wings, or petals rather than surrounding background, which aligns with the observed improvements in classification accuracy. These visualizations suggest that the polynomial meta-kernel enables more structured and task-relevant feature responses.

c) Computational cost.: We further quantify the efficiency of MCL in the FGVC setting by comparing parameter counts and FLOPs against representative transformer-based FGVC methods. The results are presented in Table VII. On ViT-B/16 with 448 × 448 inputs, MCL increases the baseline from 86.4M/78.5G to 109.0M/84.8G (Params/FLOPs). On Swin-B with 384 × 384 inputs, MCL raises the baseline from 87.9M/47.2G to 127.0M/52.9G. While MCL introduces additional parameters due to the meta-kernel parameterization, the FLOPs overhead remains moderate relative to the underlying backbone.

TABLE VII: Comparison of computational cost of the proposed method while using transformer architectures.
<table><tr><td>Method</td><td>Backbone</td><td>Input Size</td><td>Param (M)</td><td>FLOPs (G)</td></tr><tr><td>ViT</td><td>ViT-B/16</td><td>448</td><td>86.4</td><td>78.5</td></tr><tr><td>IELT</td><td>ViT-B/16</td><td>448</td><td>93.5</td><td>73.2</td></tr><tr><td>TransFG</td><td>ViT-B/16</td><td>448</td><td>86.4</td><td>130.2</td></tr><tr><td>RAMS-Trans</td><td>ViT-B/16</td><td>448</td><td>86.4</td><td>157.4</td></tr><tr><td>ACC-ViT</td><td>ViT-B/16</td><td>448</td><td>87.0</td><td>162.9</td></tr><tr><td>Swin</td><td>Swin-B</td><td>384</td><td>87.9</td><td>47.2</td></tr><tr><td>MPSA</td><td>Swin-B</td><td>384</td><td>97.4</td><td>49.2</td></tr><tr><td>ViT-Net</td><td>Swin-B</td><td>448</td><td>92.2</td><td>65.6</td></tr><tr><td>MCL (k = 1)</td><td>Swin-B</td><td>384</td><td>127.0</td><td>52.9</td></tr><tr><td>MCL (k = 1)</td><td>ViT-B/16</td><td>448</td><td>109.0</td><td>84.8</td></tr></table>

More importantly, Table VII indicates that MCL achieves favorable compute efficiency when compared to several FGVCspecific transformer pipelines. In particular, the FLOPs of MCL-equipped Swin-B (52.9G) remain close to the base Swin-B (47.2G) and are substantially lower than alternatives that rely on heavier architectural modifications or multistage processing like, TransFG [48] and RAMS-Trans [49]. Similarly, for ViT-B/16, MCL incurs only a small FLOPs increase over the baseline while avoiding the significantly higher compute budgets reported by some prior FGVC methods. These results suggest that MCL provides a practical accuracy– efficiency trade-off, delivering consistent gains in fine-grained recognition without resorting to computationally expensive FGVC-specific designs.

## V. CONCLUSION

We introduced the MCL, a dynamic convolution mechanism that generates an input-conditioned convolution kernel via a structured high-order polynomial expansion. Unlike prior dynamic convolution approaches that rely on linear mixtures of multiple basis kernels, MCL produces a single adaptive kernel without kernel-mixture attention, eliminating the need for temperature scheduling and alleviating the difficulty of jointly optimizing multiple basis kernels and their input-dependent mixing weights as the mixture size grows.

Experiments on ImageNet-1K and CIFAR demonstrate consistent improvements over vanilla networks and competing methods, and additional results on fine-grained datasets show that the approach generalizes to transformer backbones as well. Overall, MCL offers a practical and scalable alternative for input-adaptive parameterization, and future work may further reduce overhead, explore deeper placement strategies across network stages, and extend evaluation to broader downstream tasks such as object detection and semantic segmentation. In this work, MCL is added only at the end of the backbone, which fits classification since the module only needs to refine one final representation. Detection and segmentation instead need features at many spatial resolutions, so MCL could instead be added at multiple stages of the backbone so it can generate kernels at different resolutions. We have not tested this multi-stage setting and leave it for future work. Additionally, while larger kernel sizes offer favorable accuracy gains, they also introduce non-trivial parameter overhead. Reducing this cost through parameter-efficient techniques such as tensor factorization could further improve the scalability of the method.

## ACKNOWLEDGMENT

This study was supported by research fund from Chosun University, 2026.

## REFERENCES

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, pp. 84 – 90, 2012. [Online]. Available: https://api.semanticscholar.org/CorpusID:195908774

[2] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” ArXiv, vol. abs/1704.04861, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:12670695

[3] Y. Cui, Y. Song, C. Sun, A. G. Howard, and S. J. Belongie, “Large scale fine-grained categorization and domain-specific transfer learning,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4109–4118, 2018. [Online]. Available: https://api.semanticscholar.org/CorpusID:43993788

[4] J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, “Squeeze-andexcitation networks,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7132–7141, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:140309863

[5] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional block attention module,” in Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

[6] Q. Wang, B. Wu, P. F. Zhu, P. Li, W. Zuo, and Q. Hu, “Eca-net: Efficient channel attention for deep convolutional neural networks,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11 531–11 539, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:203902337

[7] Y. Chen, X. Dai, M. Liu, D. Chen, L. Yuan, and Z. Liu, “Dynamic convolution: Attention over convolution kernels,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11 027–11 036, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:208910380

[8] Y. Li, Y. Chen, X. Dai, M. Liu, D. Chen, Y. Yu, L. Yuan, Z. Liu, M. Chen, and N. Vasconcelos, “Revisiting dynamic convolution via matrix decomposition,” ArXiv, vol. abs/2103.08756, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:232240418

[9] B. Yang, G. Bender, Q. V. Le, and J. Ngiam, “Condconv: Conditionally parameterized convolutions for efficient inference,” in Neural Information Processing Systems, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:202775981

[10] C. Li, A. Zhou, and A. Yao, “Omni-dimensional dynamic convolution,” ArXiv, vol. abs/2209.07947, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:251647798

[11] C. Li and A. Yao, “Kernelwarehouse: Rethinking the design of dynamic convolution,” ArXiv, vol. abs/2406.07879, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:270391975

[12] G. Chrysos, S. Moschoglou, G. Bouritsas, J. Deng, Y. Panagakis, and S. Zafeiriou, “Deep polynomial neural networks,” CoRR, vol. abs/2006.13026, 2020. [Online]. Available: https://arxiv.org/abs/2006.13026

[13] G. Chrysos, B. Wang, J. Deng, and V. Cevher, “Regularization of polynomial networks for image recognition,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[14] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, 05 2012.

[15] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[16] M. Sandler, A. G. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4510–4520, 2018. [Online]. Available: https://api.semanticscholar.org/CorpusID:4555207

[17] A. G. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, “Searching for mobilenetv3,” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314–1324, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:146808333

[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” 2015.

[19] T.-Y. Lin, M. Maire, S. J. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in European Conference on Computer Vision, 2014. [Online]. Available: https://api.semanticscholar.org/CorpusID:14113767

[20] X. Lin, L. Ma, W. Liu, and S.-F. Chang, “Context-gated convolution,” ArXiv, vol. abs/1910.05577, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:204509548

[21] N. Ma, X. Zhang, J. Huang, and J. Sun, “Weightnet: Revisiting the design space of weight networks,” CoRR, vol. abs/2007.11823, 2020. [Online]. Available: https://arxiv.org/abs/2007.11823

[22] D. Ha, A. M. Dai, and Q. V. Le, “Hypernetworks,” in International Conference on Learning Representations, 2017. [Online]. Available: https://openreview.net/forum?id=rkpACe1lx

[23] L. Bertinetto, J. F. Henriques, J. Valmadre, P. H. S. Torr, and A. Vedaldi, “Learning feed-forward one-shot learners,” CoRR, vol. abs/1606.05233, 2016. [Online]. Available: http://arxiv.org/abs/1606.05233

[24] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[25] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, “Implicit neural representations with periodic activation functions,” Advances in neural information processing systems, vol. 33, pp. 7462– 7473, 2020.

[26] A. Gielisse and J. van Gemert, “End-to-end implicit neural representations for classification,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18 728–18 737.

[27] B. Knyazev, M. Drozdzal, G. W. Taylor, and A. Romero Soriano, “Parameter prediction for unseen deep architectures,” Advances in Neural Information Processing Systems, vol. 34, pp. 29 433–29 448, 2021.

[28] A. Zhmoginov, M. Sandler, and M. Vladymyrov, “Hypertransformer: Model generation for supervised and semi-supervised few-shot learning,” in International Conference on Machine Learning. PMLR, 2022, pp. 27 075–27 098.

[29] J. Gu and S. Yeung, “Foundation models secretly understand neural network weights: Enhancing hypernetwork architectures with foundation models,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 58 565–58 577.

[30] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[31] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9992–10 002.

[32] Z.-C. Zhang, Z.-D. Chen, Y. Wang, X. Luo, and X.-S. Xu, “A vision transformer for fine-grained classification by reducing noise and enhancing discriminative information,” Pattern Recognition, vol. 145, p. 109979, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0031320323006775

[33] Q. Xu, J. Wang, B. Jiang, and B. Luo, “Fine-grained visual classification via internal ensemble learning transformer,” IEEE Transactions on Multimedia, vol. 25, pp. 9015–9028, 2023.

[34] J. Wang, Q. Xu, B. Jiang, B. Luo, and J. Tang, “Multi-granularity part sampling attention for fine-grained visual classification,” IEEE Transactions on Image Processing, vol. 33, pp. 4529–4542, 2024.

[35] X. Jiang, H. Tang, J. Gao, X. Du, S. He, and Z. Li, “Delving into multimodal prompting for fine-grained visual classification,” Proceedings of the AAAI Conference on Artificial Intelligence,

vol. 38, no. 3, pp. 2570–2578, Mar. 2024. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/28034

[36] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, Y. W. Teh and M. Titterington, Eds., vol. 9. Chia Laguna Resort, Sardinia, Italy: PMLR, 13–15 May 2010, pp. 249–256. [Online]. Available: https://proceedings.mlr.press/v9/glorot10a.html

[37] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on imagenet classification,” in 2015 IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1026–1034.

[38] S. Zagoruyko and N. Komodakis, “Wide residual networks,” 2017.

[39] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, “Cutmix: Regularization strategy to train strong classifiers with localizable features,” 2019.

[40] G. Van Horn, S. Branson, R. Farrell, S. Haber, J. Barry, P. Ipeirotis, P. Perona, and S. Belongie, “Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 595–604.

[41] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” Aug. 2023.

[42] M.-E. Nilsback and A. Zisserman, “Automated flower classification over a large number of classes,” in 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, 2008, pp. 722–729.

[43] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, “Cats and dogs,” in IEEE Conference on Computer Vision and Pattern Recognition, 2012.

[44] H. Zhu, W. Ke, D. Li, J. Liu, L. Tian, and Y. Shan, “Dual crossattention learning for fine-grained visual categorization and object reidentification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 4692– 4702.

[45] L. Xie, L. Zheng, J. Wang, A. Yuille, and Q. Tian, “Interactive: Interlayer activeness propagation,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 270–279.

[46] H. Wu, B. Xiao, N. Codella, M. Liu, X. Dai, L. Yuan, and L. Zhang, “Cvt: Introducing convolutions to vision transformers,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 22–31.

[47] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.

[48] J. He, J.-N. Chen, S. Liu, A. Kortylewski, C. Yang, Y. Bai, and C. Wang, “Transfg: A transformer architecture for fine-grained recognition,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 1, pp. 852–860, Jun. 2022. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/19967

[49] Y. Hu, X. Jin, Y. Zhang, H. Hong, J. Zhang, Y. He, and H. Xue, “Ramstrans: Recurrent attention multi-scale transformer for fine-grained image recognition,” in Proceedings of the 29th ACM International Conference on Multimedia, ser. MM ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 4239–4248. [Online]. Available: https://doi.org/10.1145/3474085.3475561

## A. ABLATION STUDY ON META-KERNEL SIZE k AND ORDER OF EXPANSION n

This appendix analyzes the sensitivity of MCL to two key design factors: the meta-kernel size k and the polynomial expansion order n. Tables VIII and IX report Top-1 accuracy and parameter counts on CIFAR-100 using a ResNet-20 backbone under two meta-block realizations.

## A.1. Basic Meta-Block

Table VIII shows that increasing the expansion order n generally improves performance for moderate kernels. In particular, for $k \in \{ 1 , 3 \}$ , accuracy increases monotonically as n grows, indicating that higher-order interactions contribute positively to discriminative capacity. For $k \ = \ 3 ,$ the gains are consistent yet modest, suggesting that a moderate spatial receptive field is already expressive and benefits primarily from additional higher-order modulation.

In contrast, enlarging the kernel to $k = 5$ under the basic meta-block substantially increases the parameter count while providing limited additional accuracy. This behavior suggests that, without further structural constraints, scaling k can quickly become an inefficient way to improve representation. Overall, the basic meta-block favors moderate kernel sizes, e.g., $k = 3 .$ , with small-to-medium expansion orders.

TABLE VIII: Classification accuracy of ResNet-20 backbone network using MCL with varying meta-kernel size k and order of polynomial expansion $2 ^ { n }$ on the CIFAR-100 dataset. Here, MCL is constructed with only the basic meta-block.
<table><tr><td></td><td colspan="2"> $k = 1$ </td><td colspan="2"> $k = 3$ </td><td colspan="2"> $k = 5$ </td></tr><tr><td></td><td>#param</td><td>Top-1</td><td>#param</td><td>Top-1</td><td>#param</td><td>Top-1</td></tr><tr><td> $n = 1$ </td><td>0.35M</td><td>71.94</td><td>3.63M</td><td>77.69</td><td>24.35M</td><td>77.41</td></tr><tr><td> $n = 2$ </td><td>0.42M</td><td>72.86</td><td>3.70M</td><td>77.88</td><td>24.42M</td><td>77.92</td></tr><tr><td> $n = 3$ </td><td>0.49M</td><td>73.66</td><td>3.78M</td><td>78.19</td><td>24.49M</td><td>77.94</td></tr></table>

## A.2. Compact Meta-Block

Table IX demonstrates that the Compact Meta-Block markedly improves parameter efficiency, especially when using larger kernels. Compared with the Basic Meta-Block design, the Compact Meta-Block substantially reduces the cost of spatial modeling, allowing $k = 5$ to become viable without incurring prohibitive parameters. Notably, the bestperforming Compact Meta-Block configuration is achieved at a moderate expansion order, typically $n = 2 ,$ , while further increasing n does not necessarily yield additional gains and may even degrade accuracy for certain kernel sizes. This observation suggests that, under the Compact Meta-Block constraints, excessive expansion order can introduce optimization difficulty or reduce effective capacity due to over-constrained intermediate representations.

## A.3. Latency and Memory Overhead Comparison

Table X presents an inference-time comparison of the proposed MCL against the baseline backbone and representative dynamic convolution baselines, namely $\mathrm { K W } ( 4 \times )$ [11] and

TABLE IX: Classification accuracy of ResNet-20 backbone network using MCL with varying meta-kernel size k and order of polynomial expansion $2 ^ { n }$ on the CIFAR-100 dataset. Here, MCL is constructed with only the compact meta-block.
<table><tr><td></td><td colspan="2"> $k = 1$ </td><td colspan="2"> $k = 3$ </td><td colspan="2"> $k = 5$ </td></tr><tr><td></td><td>#param</td><td> $\overline { { \mathrm { T o p } { - } 1 } }$ </td><td>#param</td><td>Top-1</td><td>#param</td><td> $\overline { { \mathrm { T o p } { - } 1 } }$ </td></tr><tr><td> $n = 1$ </td><td>0.28M</td><td>70.93</td><td>0.59M</td><td>76.13</td><td>2.49M</td><td>77.83</td></tr><tr><td> $n = 2$ </td><td>0.28M</td><td>70.38</td><td>0.60M</td><td>76.57</td><td>2.49M</td><td>78.31</td></tr><tr><td> $n = 3$ </td><td>0.29M</td><td>70.64</td><td>0.60M</td><td>75.86</td><td>2.50M</td><td>78.07</td></tr></table>

ODConv(4×) [10], on the ImageNet test set. The reported latency, throughput, and peak memory metrics are obtained using a batch size of 128, and each value is computed over 50 batches to provide a stable estimate of runtime behavior. As indicated in the table, lower latency and peak memory are preferred, whereas higher throughput is desirable.

For ResNet18, the baseline model achieves $5 . 2 3 \pm 1 . 5 6$ ms latency, 24472.18 images/sec throughput, and 910.82 MB peak memory. Relative to this reference, MCL introduces only a moderate inference overhead: MCL(k = 1) attains $7 . 6 8 \pm 1 . 8 6$ ms and 16659.78 images/sec with 992.38 MB peak memory, while MCL $\left( k \ = \ 3 \right)$ yields $6 . 7 5 \pm 2 . 1 3$ ms and 18957.37 images/sec with 1909.15 MB peak memory. In contrast, $\mathrm { K W } ( 4 \times )$ and ODConv(4×) incur substantially larger runtime costs, reaching $9 1 . 6 7 \pm 0 . 5 6$ ms and $3 5 . 0 4 \pm 1 . 7 1$ ms latency, respectively, with markedly lower throughput. These results show that, on a lightweight backbone, MCL remains much closer to the efficiency profile of the static baseline than prior kernel-aggregation methods.

A similar trend is observed on ResNet50. The baseline records 10.24±1.82 ms latency, 12503.45 images/sec throughput, and 1622.69 MB peak memory. $\mathbf { M C L } ( k \ = \ 1 )$ and MCL $, ( k = 3 )$ increase latency only modestly to 11.73 ± 2.12 ms and $1 1 . 5 4 \pm 1 . 5 9 ~ \mathrm { m s } ,$ , respectively, while sustaining high throughput of 10908.65 and 11087.52 images/sec. By comparison, KW(4×) and ODConv(4×) exhibit much heavier computational overhead, with latencies of $2 8 0 . 4 3 \pm 1 . 7 3$ ms and $1 2 5 . 3 2 \pm 0 . 2 4$ ms and correspondingly low throughput of 456.44 and 1021.37 images/sec.

Overall, MCL delivers input-adaptive convolution with a considerably smaller runtime penalty than methods that rely on aggregating multiple kernels through attention. This observation is consistent with the core design of MCL, which directly generates a single input-conditioned meta-kernel rather than performing expensive multi-kernel composition.

## A.4. Optimization Behavior and Run-to-Run Stability

In addition to the ablations on meta-kernel size k and expansion order n, we further examine the optimization behavior of MCL to directly address the claim made in the main text that the proposed formulation is easier to optimize than prior attention-based dynamic convolution methods. The convergence plots in Fig. 5 provide empirical evidence of a stable training procedure.

First, across ImageNet, MCL exhibits stable and wellbehaved convergence for all evaluated backbones. For both $k = 1$ and $k = 3 ,$ , the test error decreases rapidly during the early stage of training and continues to improve smoothly over later epochs for ResNet18, ResNet50, and ResNet101. Importantly, we do not observe unstable oscillations or optimization collapse as network depth increases, which suggests that the proposed meta-polynomial kernel generation remains trainable even when integrated into larger backbone architectures. This observation is consistent with the design rationale of MCL in Sections III-B and III-E, where the adaptive kernel is learned end-to-end through a single meta-network without requiring auxiliary optimization heuristics such as temperature annealing.

![](images/d76e1c8248e5095b201bae8bbdfb376c2361b322f0ab5c16bb80660b427965a1.jpg)

![](images/c34e99fedc291eb6ea81fef6136f84b3126292c58610afb2675d358e30cb86d0.jpg)

![](images/4f235189eb03263686447733f0b04a0097e354720a29025ec7813a28d90cfa45.jpg)

![](images/5fe4afde48c0912fb3481177e1a9ce4ae9312621357cd27c183cd3245de0ca42.jpg)  
Fig. 5: Convergence behavior of MCL on ImageNet and CIFAR-100 under different meta-kernel sizes. The first two panels show the test-error trajectories of ResNet18, ResNet50, and ResNet101 equipped with MCL on ImageNet for $k = 1$ and $k = 3 ,$ respectively. The last two panels show the corresponding CIFAR-100 results for ResNet20, ResNet56, ResNet110, WRN16-10, and WRN28-10. For CIFAR-100, solid lines denote the mean of three independent runs and the shaded bands indicate the standard deviation. Across datasets, architectures, and kernel settings, MCL exhibits smooth convergence and low run-to-run variability, supporting the claim that the proposed layer is stable and easy to optimize in practice.

TABLE X: Latency and memory overhead comparison of the proposed method on ImageNet test set. ↓ indicates lower is better and ↑ indicates higher is better.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Latency (ms)↓</td><td rowspan=1 colspan=1>Throughput(images/sec)↑</td><td rowspan=1 colspan=1>Peak Mem(MB)↓</td></tr><tr><td rowspan=1 colspan=1>ResNet18</td><td rowspan=1 colspan=1> $5 . 2 3 \pm 1 . 5 6$ </td><td rowspan=1 colspan=1>24472.18</td><td rowspan=1 colspan=1>910.82</td></tr><tr><td rowspan=1 colspan=1>+ KW(4×) ${ } + \mathrm { O D C o n v } ( 4 \times )$ </td><td rowspan=1 colspan=1> $9 1 . 6 7 \pm 0 . 5 6$  $3 5 . 0 4 \pm 1 . 7 1$ </td><td rowspan=1 colspan=1>1396.303653.18</td><td rowspan=1 colspan=1>2736.416376.81</td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { \mathbf { M C L } } ( k = 1 )$ + MCL(k = 3)</td><td rowspan=1 colspan=1> $7 . 6 8 \pm 1 . 8 6$  $6 . 7 5 \pm 2 . 1 3$ </td><td rowspan=1 colspan=1>16659.7818957.37</td><td rowspan=1 colspan=1>992.381909.15</td></tr><tr><td rowspan=1 colspan=1>ResNet50</td><td rowspan=1 colspan=1> $1 0 . 2 4 \pm 1 . 8 2$ </td><td rowspan=1 colspan=1>12503.45</td><td rowspan=1 colspan=1>1622.69</td></tr><tr><td rowspan=1 colspan=1>+ KW(4×) ${ } + \mathrm { O D C o n v } ( 4 \times )$ </td><td rowspan=1 colspan=1> $2 8 0 . 4 3 \pm 1 . 7 3$  $1 2 5 . 3 2 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1>456.441021.37</td><td rowspan=1 colspan=1>3707.657547.10</td></tr><tr><td rowspan=1 colspan=1> $+ \mathrm { \mathbf { M C L } } ( k = 1 )$  $+ \mathbf { M C L } ( k = 3 )$ </td><td rowspan=1 colspan=1> $1 1 . 7 3 \pm 2 . 1 2$  $1 1 . 5 4 \pm 1 . 5 9$ </td><td rowspan=1 colspan=1>10908.6511087.52</td><td rowspan=1 colspan=1>3293.9222196.76</td></tr></table>

Second, the CIFAR-100 results indicate that this optimization behavior is also stable across independent runs. In Fig. 5, the solid curves for CIFAR-100 denote the mean test error over three independent runs, and the shaded regions represent the corresponding standard deviation. Across ResNet20, ResNet56, ResNet110, WRN16-10, and WRN28- 10, the variance bands remain relatively narrow throughout training for both $k \ = \ 1$ and $k \ = \ 3$ . This low run-torun variability suggests that MCL is not overly sensitive to random initialization or stochasticity in optimization, and that the observed performance gains are reproducible rather than arising from isolated favorable runs.

## A.5. Discussion

Taken together, Tables VIII–IX highlight a clear trade-off. Increasing n is beneficial up to a small value, after which returns diminish and may become unstable depending on the meta-block structure. Increasing k can improve accuracy, but is only practical when paired with a parameter-efficient realization such as the Compact Meta-Block. In practice, a strong accuracy–efficiency balance is obtained with $k \mathit { \Theta } = \mathit { 3 }$ and a small expansion order, e.g., $n \in \{ 2 , 3 \}$ , for the Basic Meta-Block design, while the Compact Meta-Block enables larger kernels, e.g., k = 5, to achieve competitive accuracy with substantially reduced parameter overhead.