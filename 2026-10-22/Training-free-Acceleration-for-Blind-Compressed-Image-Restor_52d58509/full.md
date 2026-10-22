# Training-free Acceleration for Blind Compressed Image Restoration via Activation-Calibrated Singular Value Decomposition

Suhang Yao, Xin Li, Xijun Wang, Yufei Wang, Yeying Jin, and Zhibo Chen

Abstract—Transformer-based models achieve strong performance on blind compressed image restoration (CIR) under unknown compression settings, but their densely stacked projection operators make deployment costly. We study training-free acceleration for such models through an empirical, operatorlevel analysis. Specifically, we profile where computation is concentrated, characterize channel-wise activation distributions under compressed inputs, and examine the singular-value decay of representative projection weights. These measurements identify the recurring projections in the attention, feed-forward, convolutional, and prompt-conditioning pathways as effective low-rank replacement targets that preserve the original inference graph. They further establish truncated SVD (TSVD) as a principled training-free baseline for blind CIR and reveal why a uniform low-rank objective is mismatched to the channel-imbalanced feature distributions of compressed images. Accordingly, we formulate Activation-Calibrated Singular Value Decomposition (ASVD) for this restoration setting, deriving low-rank factors entirely offline from pretrained weights and lightweight activation statistics gathered from only a few compressed calibration images. By weighting the decomposition with channel-wise activation magnitudes, our ASVD assigns larger penalties to errors on more active channels, requiring neither retraining nor online calibration and leaving the restoration pipeline intact. When instantiated on PromptCIR at rank 64, ASVD consistently improves PSNR-B over TSVD across four standard benchmarks under identical parameter and MAC budgets. At this operating point, the factorized model cuts parameters by 59% (34.75M → 14.34M) and computation by 16.1% (258.79 → 217.16 GMACs for a 256 × 256 input), enabling acceleration without retraining.

Index Terms—Singular value decomposition, activation calibration, low-rank approximation, model compression, inference acceleration, blind compressed image restoration

## I. INTRODUCTION

Lossy image compression is indispensable for large-scale visual storage and transmission. JPEG remains one of the most widely deployed standards thanks to its simplicity and broad compatibility, yet aggressive compression inevitably introduces blocking, ringing, and texture loss. Blind compressed image restoration (CIR) targets this problem by recovering faithful images from compressed inputs without access to the compression quality factor or other degradation parameters.

Deep restoration models have substantially advanced blind CIR. Transformer-based architectures are particularly effective, since self-attention and hybrid attention blocks capture the long-range dependencies that purely local models struggle to represent [1]. Representative backbones such as Uformer [2], Restormer [3], and HAT [4] have performed well across a range of low-level vision tasks. For blind CIR, PromptCIR [5] shows that prompt-based conditioning improves robustness when compression artifacts are unknown and spatially varying.

The architectural choices that raise restoration quality also complicate deployment. Modern transformer-based CIR models repeatedly apply high-dimensional projection operators in their attention, feed-forward, convolutional-projection, and promptconditioning pathways. Because these operators recur across many scales and layers, they dominate both parameters and computation. Our profiling of PromptCIR shows that the attention, feed-forward, and convolutional-projection operators alone account for 99.43% of parameters and 99.89% of convolutional and linear MACs in the 128 × 128 profiling setting, which makes them the primary targets for acceleration. For resource-constrained or latency-sensitive pipelines, reducing these projections directly improves efficiency while keeping the surrounding restoration process intact.

We approach acceleration from this operator-level evidence rather than from a fixed compression recipe. An effective method should target the repeatedly used projections, preserve the pretrained restoration pipeline, avoid task-specific retraining, and account for channel-dependent behavior under compressed inputs. We therefore examine activation statistics and singular-value spectra before deciding how to replace these operators. The spectral analysis points to truncated singular value decomposition (TSVD) as a natural trainingfree baseline: linear projections and reshaped convolutional projections are written as matrices, decomposed offline, and replaced by two sequential low-rank projections. Direct TSVD, however, treats all channels uniformly under an unweighted matrix objective. This assumption is restrictive for restoration, since compressed-image features exhibit markedly nonuniform channel magnitudes tied to artifacts, textures, and promptconditioned information.

Guided by these observations, we design Activation-Calibrated Singular Value Decomposition (ASVD) as a projection-level acceleration method for transformer-based blind CIR. The method first expresses the selected linear and convolutional projections in a unified matrix form. TSVD then provides a closed-form, training-free low-rank baseline grounded in the Eckart–Young theorem [6]. Building on this baseline, our ASVD uses lightweight input and output activation statistics collected from a small set of compressed calibration images to weight the decomposition. The resulting low-rank factors are produced entirely offline and inserted back into the pretrained network, requiring no ground-truth restored images, no retraining, no prompt redesign, and no online calibration.

We instantiate the framework on PromptCIR, a representative prompt-based blind CIR model, and evaluate it on standard blind CIR benchmarks.

At rank 64, ASVD improves PSNR-B over TSVD across all four benchmarks while maintaining comparable PSNR and SSIM under identical parameter and MAC budgets.

The main contributions are summarized as follows:

• We present a training-free, projection-level low-rank acceleration formulation for transformer-based blind CIR. Operator profiling and singular-value analysis identify the dominant replacement targets and establish TSVD as an architecture-preserving baseline.

• We design ASVD to address the channel imbalance observed in compressed-image feature activations. Our decomposition draws on lightweight statistics from compressed calibration inputs to emphasize activation-sensitive channels, without paired targets or retraining.

• We instantiate the method on PromptCIR and evaluate it against the original backbone and TSVD. The results show consistent PSNR-B improvements over TSVD at rank 64 under the same parameter and MAC budgets, while leaving the surrounding restoration pipeline unchanged.

## II. RELATED WORK

## A. Compressed Image Restoration

Compressed image restoration (CIR) targets artifacts caused by quantization and transform loss. Early approaches relied on transform-domain priors or dual-domain modeling [7], using handcrafted assumptions to recover information lost to compression.

CNN-based methods later pushed CIR performance further. ARCNN [8] demonstrated the effectiveness of convolutional networks for artifact reduction, while dual-domain designs such as D3 [9] and DMCNN [10] exploited complementary representations in the pixel and transform domains, following the residual learning paradigm [11]–[14]. Attention-enhanced CNN restorers further improved representation capacity through channel and non-local modeling [15], [16], while quantizationguided designs injected JPEG side information [17].

More recently, transformer-based architectures have become popular restoration backbones, owing to the global modeling capability of self-attention [18]. Representative models such as SwinIR [19], Uformer [2], Restormer [3], and HAT [4], [20] handle long-range dependencies and complex artifact patterns effectively, while Stripformer [21] and NAFNet [22] improve efficiency through structured attention or simplified operators. State-space restoration backbones such as MambaIR [23] offer another route to efficient long-range modeling. This progression has been evaluated across diverse restoration benchmarks and datasets [1], [24]–[27]. However, the gains of these backbones come largely from stacking high-dimensional projection and attention operators, whose cost grows rapidly with feature width and token length, making efficient deployment a recurring concern.

The blind CIR setting poses additional challenges, because the compression artifacts are unknown and spatially varying. To improve robustness, prior work incorporates auxiliary cues or representation learning strategies, including quality factor-aware modeling [28] and contrastive representation learning [29]. More recently, prompt-based conditioning has emerged as an effective way to handle diverse degradations, with prompt-conditioned transformers proving effective in both all-in-one and blind CIR settings, including PromptIR [30], PromptCIR [5], ProRes [31], PromptRR [32], Promptin-Prompt [33], DA-CLIP [34], and visual or multimodal prompt learning methods [35]–[37]. Related compressed-image restoration and super-resolution efforts, including UCIP [38], MambaCSR [39], and CODiff [40], further show the importance of compression-aware conditioning. Such prompt conditioning introduces additional projection pathways that interact with the backbone, further enlarging the operator budget that any acceleration method must address.

## B. Model Compression and Acceleration

Model efficiency has been studied extensively through pruning, quantization, and low-rank approximation [41]–[44]. Among these, low-rank approximation offers a principled and training-free route to reducing computation, grounded in the classical Eckart–Young theorem [6]. Classical CNN acceleration methods showed that convolutional redundancy can be exploited through low-rank or tensor factorization [45]–[47]. Recent works revisit truncated SVD for large neural networks and propose truncation-aware [48] or importance-weighted and Fisher-weighted variants [49] to improve approximation fidelity. A shared insight is that raw weight-reconstruction error is an incomplete objective, since channels contribute unequally to the layer output; this motivates activation- or importance-aware reweighting of the decomposition objective.

In parallel, training-free acceleration techniques for iterative generative models exploit feature reuse and operator-level approximation. Representative examples include DeepCache [50], block-wise caching [51], and channel-aware SVD acceleration for diffusion transformers [52]. Other cache-based or trainingfree variants further explore layer caching, token-wise feature reuse, fast-forward caching, and delta updates [53]–[57]. Although developed mainly for diffusion models, these approaches reinforce calibration and operator-level approximation as practical tools for training-free acceleration.

![](images/9adac4cc6e80e9799be57b3b7b660c4af71c2de4de6107d5bbc842ddda822f57.jpg)

![](images/fbce1c2b2c5ebccb1959a0b5e86cdbb7a94b70bdee7670932508e9177a58bf17.jpg)

(c) Rank for 95% spectral energy across representative layers  
![](images/beda428a3a5c2abf07df460a959ec02144803b2772f4af9be21540a1bced59fc.jpg)  
Fig. 1. Empirical motivation for projection-level ASVD. (a) Parameter distribution and convolutional and linear MAC distribution across operator groups in PromptCIR, evaluated at an input resolution of 128 × 128. (b) Output activations are channel-imbalanced: a relatively small fraction of channels often accounts for 50% of activation energy, while the fraction required for 90% energy varies by operator family. (c) The rank required to preserve 95% spectral energy also varies across operators; bars show the mean over representative layers and error bars show the min–max range.

We develop a training-free acceleration scheme for PromptCIR-based blind CIR that calibrates projection factorization using activation magnitudes from compressed images. The resulting diagonal scaling provides a lightweight alternative to truncation-aware whitening [48] and Fisher-weighted objec tives [49]. Whereas channel-aware decomposition for diffusion transformers supports cache correction [52], our factors directly replace spatial and prompt-conditioning projections, preserving the restoration pipeline without retraining.

## III. MOTIVATION AND DESIGN RATIONALE

We derive the projection-level SVD design from an empirical analysis of transformer-based blind CIR. The analysis is organized around three questions: which operators dominate the cost, whether their weights admit a compact low-rank approximation, and whether compressed-image activations justify channel-aware weighting. The answers determine our method design: projection operators define the replacement unit, TSVD supplies the training-free baseline, and ASVD supplies activation-aware weighting.

## A. Projection Operators as the Replacement Unit

Prompt-based transformer restoration models contain many projection operators, including query-key-value projections, output projections, feed-forward expansions and reductions, point-wise convolutional projections, and prompt-conditioning projections. These modules are invoked repeatedly across the encoder, bottleneck, and decoder stages. They share a unified matrix representation: a linear layer is already a matrix, and a convolution kernel can be reshaped into a two-dimensional operator. This common representation localizes the replacement while preserving the architecture.

The cost profile in Fig. 1(a) shows that this choice is not merely convenient algebraically but also meaningful for efficiency. In the profiled PromptCIR model, attention projections, feed-forward projections, and convolutional-projection layers together account for 99.43% of parameters and 99.89% of convolutional and linear MACs at $1 2 8 \times 1 2 8$ , so replacement can concentrate on these operators rather than on the non-dominant modules. Prompt-conditioning operators are far smaller in aggregate, yet we retain them within the unified formulation because they govern degradation-conditioned feature modulation. Replacing selected projections is also far less intrusive than redesigning attention blocks. The surrounding normalization, nonlinearity, skip-connection, and prompt-conditioning logic can remain unchanged, while each dense projection is realized by two low-rank factors. This locality is essential in an offline replacement setting, where the compressed model must reproduce the pretrained restoration mapping as closely as possible.

## B. Spectral Evidence for SVD-Based Replacement

After identifying the replacement unit, we examine whether SVD is a reasonable low-rank mechanism for these operators. SVD satisfies the deployment constraints imposed by the pretrained restoration model: it can be computed offline from pretrained weights, yields the optimal rank-r approximation under the standard Frobenius objective, and applies uniformly to linear projections and reshaped convolution kernels. It preserves the surrounding inference graph, since each original projection is replaced by two sequential low-rank projections with the same tensor interface.

The singular-value analysis in Fig. 1(c) shows why SVD is an appropriate starting point but not a complete design. Representative projection operators exhibit measurable spectral concentration, but its degree varies across operator families: the mean rank needed to preserve 95% of the spectral energy, defined as the sum of squared singular values, ranges from 47.93% for prompt linear projections to 86.47% for GDFN projections. This evidence supports SVD-based replacement while indicating that rank must be analyzed explicitly. First, TSVD is a principled baseline because it tests how much of the pretrained projection behavior a bounded-rank subspace can capture. Second, because uniform low-rank replacement is not equally effective for all projections, rank and operator selection should be evaluated explicitly rather than fixed globally.

## C. Channel Statistics Under Compressed Inputs

TSVD provides the low-rank baseline, but its objective is uniform over channels. For blind CIR, this assumption deserves scrutiny because the network must preserve spatially precise structures while suppressing compression artifacts. As compressed images pass through a restoration transformer, different channels carry different amounts of artifact, texture, and prompt-conditioned information. We therefore inspect the activation statistics of high-cost projections using a small compressed calibration set.

The observed activation distributions are markedly nonuniform. For several window-attention, multi-Dconv head transposed attention (MDTA), gated-Dconv feed-forward network (GDFN), and output-projection families, 50% of the output activation energy, defined as the sum of squared per-channel activation magnitudes, concentrates in only 17.39%–23.76% of the channels, whereas prompt-related and some overlapattention projections are less concentrated. Such heterogeneity means that a single unweighted objective treats all channels alike even when their activation magnitudes differ sharply under compressed inputs, which makes uniform channel treatment restrictive. If a high-activation channel is poorly approximated, the resulting error can propagate through the restoration features far more strongly than an error on a weakly activated channel.

Our ASVD uses this observation as calibration information rather than as a training signal. The collected statistics neither optimize a restoration loss, nor rely on ground-truth restored images, nor modify the model online. Instead, they define diagonal scaling factors that give activation-dominant input and output channels greater influence within a deterministic offline replacement procedure.

## D. From Analysis to ASVD

These observations define a two-step replacement strategy. Projection-level replacement targets the dominant cost centers while keeping the pretrained restoration pipeline intact. The spectral results support TSVD as a closed-form low-rank baseline for these matrix-representable operators. The activation results then motivate ASVD, our factor-generation refinement that introduces activation-derived channel weights so that channels more active under compressed inputs gain greater influence in the weighted approximation objective. At a fixed rank, TSVD and ASVD share the same inference-time structure, parameter count, and MACs; comparing them therefore isolates the effect of activation-calibrated factor generation under an identical deployment budget.

## IV. METHODOLOGY

## A. Overview

The analysis in Sec. III leads to an operator-replacement formulation: selected projections are replaced by two low-rank factors generated offline from pretrained weights and, in our ASVD variant, lightweight activation statistics. The surrounding restoration pipeline is left unchanged, and no retraining, prompt redesign, or online updating is introduced.

As illustrated in Fig. 2, TSVD directly decomposes the pretrained projection weights as a closed-form baseline, whereas our ASVD first applies channel-wise scaling, decomposes the scaled weights, and then maps the factors back to the original operator space. At a fixed rank, the two variants share the same inference-time structure and complexity; only factor generation differs.

## B. Unified Projection Representation

Transformer-based CIR models use heterogeneous projection operators. A linear layer is parameterized by $W \in \mathbb { R } ^ { o \times i }$ , where i and o denote the input and output channel dimensions. For a convolution kernel $\bar { K ^ { \prime } } \in \mathbb { R } ^ { o \times i \times k _ { h } \times k _ { w } }$ , we use the matrix form

$$
W = { \mathrm { r e s h a p e } } ( K ) \in \mathbb { R } ^ { o \times ( i k _ { h } k _ { w } ) } .\tag{1}
$$

Eq. (1) lets linear layers and convolutional projections share the same decomposition procedure; after factorization, each operator is restored to its original layer type.

## C. Problem Formulation

Let f(·) denote a pretrained transformer-based blind CIR model. Guided by the cost profile in Sec. III-A, we select a set of projection operators $s$ from the attention, feed-forward, convolutional projection, and prompt-conditioning pathways. Each operator $m \in { \mathcal { S } }$ is represented by a weight matrix $W ^ { ( m ) }$ Given a rank budget r, the goal is to construct

$$
W ^ { ( m ) } \approx \hat { W } ^ { ( m ) } = \hat { W } _ { a } ^ { ( m ) } \hat { W } _ { b } ^ { ( m ) } , \qquad \mathrm { r a n k } ( \hat { W } ^ { ( m ) } ) \leq r ,\tag{2}
$$

while leaving all non-selected modules unchanged. The effective rank is bounded by the operator shape,

$$
r _ { \mathrm { e f f } } = \operatorname* { m i n } ( r , r _ { \mathrm { m a x } } ) ,\tag{3}
$$

where $r _ { \operatorname* { m a x } } = \operatorname* { m i n } ( o , i )$ for a linear projection, and $r _ { \operatorname* { m a x } } =$ min $\left( o , i k _ { h } k _ { w } \right)$ for a matricized convolution. Thus, a shared r is an upper bound rather than a common retained-rank fraction. It provides a controlled comparison of TSVD and ASVD without layer-wise rank optimization.

![](images/627b8ca4c4c439fa7d894f3184af2bc9876ef7cf4c99b279d1e3ac344e9e0bce.jpg)  
Fig. 2. Overview of our ASVD. We select high-cost projection operators in a transformer-based blind CIR model, collect lightweight per-channel activation statistics on a small calibration set, and form diagonal scaling factors. Truncated SVD is applied to the scaled weights, and the low-rank factors are mapped back via inverse scaling, yielding factorized replacements that preserve the original inference flow.

## D. Complexity of Low-rank Replacement

The weight and MAC counts follow directly from the matrix shape; the expressions below exclude unchanged bias terms. For a dense projection $W \in \mathbb { R } ^ { o \times i }$ , the original parameter count is oi. Replacing it with two factors $W _ { a } \in \mathbb { R } ^ { o \times r _ { \mathrm { e f f } } }$ and $W _ { b } \in \mathbb { R } ^ { r _ { \mathrm { e f f } } \times i }$ changes the count to

$$
P _ { \mathrm { l r } } = r _ { \mathrm { e f f } } ( o + i ) .\tag{4}
$$

The same expression gives the per-location multiply-accumulate cost of a linear projection. For a convolution kernel reshaped as $W \in \mathbb { R } ^ { o \times ( i k _ { h } \bar { k } _ { w } ) }$ , the low-rank parameter count becomes

$$
P _ { \mathrm { l r } } = r _ { \mathrm { e f f } } \big ( o + i k _ { h } k _ { w } \big ) .\tag{5}
$$

Low-rank replacement reduces computation when the rank is small enough to offset the factorization overhead. The ablation study reports the resulting parameter and MAC counts.

## E. Implementation of Factorized Operators

The low-rank factors are inserted as ordinary projection layers without intermediate nonlinearity or normalization. For a linear projection, $W _ { b }$ first maps the input feature from i channels to $r _ { \mathrm { e f f } }$ channels, and $W _ { a }$ then maps the intermediate feature from $r _ { \mathrm { e f f } }$ channels to o channels. If the original projection has a bias term, it is retained after the second factor to preserve the affine form.

For a convolutional projection, the reshaped factor $W _ { b } \in$ $\mathbb { R } ^ { r _ { \mathrm { e f f } } \times ( i k _ { h } k _ { w } ) }$ is implemented as a convolution from i channels to $r _ { \mathrm { e f f } }$ channels with kernel size $k _ { h } \times k _ { w } ,$ and the second factor $W _ { a } \in \mathbb { R } ^ { o \times r _ { \mathrm { e f f } } }$ as a $1 \times 1$ convolution from $r _ { \mathrm { e f f } }$ channels to o channels. The original spatial stride, padding, and dilation are attached to the first factor, while the second factor mixes the intermediate channels. This preserves the input-output tensor shape and leaves residual connections, attention reshaping, prompt fusion, and output heads unchanged.

Because TSVD and ASVD produce the same two-factor graph at the same rank, their difference is attributable to factor generation rather than runtime architecture.

## F. TSVD as the Training-free Low-rank Baseline

For a selected weight matrix W, the singular value decomposition is

$$
W = U \Sigma V ^ { \top } ,\tag{6}
$$

where the singular values are sorted in descending order. Here and below, the subscript r denotes truncation to the effective rank $r _ { \mathrm { e f f } }$ in Eq. (3). TSVD retains these leading components:

$$
\boldsymbol { W } \approx \boldsymbol { W _ { r } } = U _ { r } \boldsymbol { \Sigma _ { r } } \boldsymbol { V _ { r } } ^ { \top } .\tag{7}
$$

We form the two factors as

$$
W _ { a } = U _ { r } \Sigma _ { r } , \qquad W _ { b } = V _ { r } ^ { \top } ,\tag{8}
$$

so that $W _ { r } \ : = \ : W _ { a } W _ { b }$ , and the original dense projection is replaced by two sequential low-rank projections. This baseline is closed-form, deterministic, and training-free; however, its weight-centric objective ignores the channel magnitudes that compressed inputs induce.

## G. Proposed Activation-Calibrated SVD

1) Activation-calibrated objective: Our ASVD reshapes the decomposition objective using the channel-wise activation magnitudes observed in Sec. III-C. For a projection matrix W, we define diagonal scaling matrices

$$
D _ { l } = \mathrm { d i a g } ( d _ { l } ) , \qquad D _ { r } = \mathrm { d i a g } ( d _ { r } ) .\tag{9}
$$

The scaling vectors $d _ { l }$ and $d _ { r }$ are derived from the output and input activation statistics, respectively, with dimensions matching the rows and columns of W. For $E = W - { \hat { W } }$ and input $x ,$ the expected squared output perturbation is

$$
\mathbb { E } \big [ \| E x \| _ { 2 } ^ { 2 } \big ] = \mathrm { t r } ( E C _ { x } E ^ { \top } ) ,
$$

where $C _ { x } = \mathbb { E } [ x x ^ { \top } ]$ . Input activation magnitudes approximate input-channel importance, while output magnitudes weight errors in active output channels. Using these lightweight diagonal weights, ASVD minimizes

$$
\operatorname* { m i n } _ { \operatorname { r a n k } ( \hat { W } ) \leq r } \left\| D _ { l } ( W - \hat { W } ) D _ { r } \right\| _ { F } ^ { 2 } .\tag{10}
$$

Algorithm 1 Offline ASVD factor generation   
Require: Pretrained CIR model ${ \overline { { f , } } }$ selected projection set $s ,$   
compressed calibration set ${ \mathcal { C } } ,$ rank budget r   
1: Run $f$ on $\mathcal { C }$ and collect input and output channel magni  
tudes for each $m \in { \mathcal { S } }$   
2: for each selected projection m $\in S$ do   
3: Convert its weight to matrix form $W ^ { ( m ) }$   
4: Form diagonal scaling matrices ${ D } _ { l } ^ { ( m ) }$ and ${ D } _ { r } ^ { ( m ) }$ from   
the collected channel magnitudes   
5: Compute $\tilde { W } ^ { ( m ) } = D _ { l } ^ { ( m ) } \breve { W } ^ { ( m ) } D _ { r } ^ { ( m ) }$   
6: Retain the largest $r _ { \mathrm { e f f } }$ singular values of $\tilde { W } ^ { ( m ) }$ and their   
singular vectors   
7: Map the factors back according to Eq. (12)   
8: Replace the original dense projection with the two low   
rank factors   
9: end for   
10: return Compressed model with all other modules un  
changed

Let $\tilde { W } ~ = ~ D _ { l } W D _ { \ l }$ <sub>r</sub> and $Z ~ = ~ D _ { l } { \hat { W } } D _ { \eta }$ . Invertible scaling preserves rank, so Eq. (10) is equivalent to

$$
\operatorname* { m i n } _ { \operatorname { r a n k } ( Z ) \leq r } \| \tilde { W } - Z \| _ { F } ^ { 2 } .
$$

Its truncated-SVD minimizer is

$$
\tilde { W } _ { r } = \tilde { U } _ { r } \tilde { \Sigma } _ { r } \tilde { V } _ { r } ^ { \top } ,\tag{11}
$$

and mapping the factors back gives

$$
\begin{array} { r l } & { \hat { W } _ { a } = D _ { l } ^ { - 1 } \tilde { U } _ { r } \tilde { \Sigma } _ { r } , } \\ & { \hat { W } _ { b } = \tilde { V } _ { r } ^ { \top } D _ { r } ^ { - 1 } . } \end{array}\tag{12}
$$

The product $\hat { W } = \hat { W } _ { a } \hat { W } _ { b }$ minimizes the weighted projection error in Eq. (10) and replaces the original projection with two sequential factors. The scaling is absorbed during offline factor generation. We examine the contribution of each weighting side in Sec. V-D. After factor generation, ASVD executes in the same form as TSVD and adds no inference-time computation at a fixed rank.

In Algorithm 1, rank- $- r _ { \mathrm { e f f } }$ SVD means the truncated product $\tilde { U } _ { r _ { \mathrm { e f f } } } \tilde { \Sigma } _ { r _ { \mathrm { e f f } } } \tilde { V } _ { r _ { \mathrm { e f f } } } ^ { \top }$ . For an $o \times i$ matrix, the two mapped factors have sizes $o \times r _ { \mathrm { e f f } }$ and $r _ { \mathrm { e f f } } \times i ,$ respectively.

2) Offline calibration: The scaling vectors $( d _ { l } , d _ { r } )$ are estimated by forwarding a small compressed calibration set through the pretrained model and accumulating per-channel mean absolute activation magnitudes at selected projection inputs and outputs. For an input activation tensor $X ^ { ( m ) }$ and output activation tensor $Y ^ { ( m ) }$ of projection $m ,$ , we compute the per-channel statistics

$$
\begin{array} { r l } & { a _ { r , c } ^ { ( m ) } = \mathbb { E } _ { \mathcal { C } , h , w } \left[ | X _ { c , h , w } ^ { ( m ) } | \right] , } \\ & { a _ { l , c } ^ { ( m ) } = \mathbb { E } _ { \mathcal { C } , h , w } \left[ | Y _ { c , h , w } ^ { ( m ) } | \right] . } \end{array}\tag{13}
$$

where the expectation is taken over the calibration set C and the spatial dimensions. For each side $s \in \{ l , r \}$ , we form strictly positive scaling entries as

$$
d _ { s , c } ^ { ( m ) } = \operatorname* { m a x } \biggl ( a _ { s , c } ^ { ( m ) } , \eta \operatorname* { m a x } _ { j } a _ { s , j } ^ { ( m ) } \biggr ) + \epsilon ,\tag{14}
$$

where $\eta \ : = \ : 1 0 ^ { - 3 }$ and $\epsilon = 1 0 ^ { - 6 }$ . The relative floor limits inverse amplification on weak channels, and the additive term keeps the scaling invertible even when all channel statistics are zero. Statistics and factor generation use FP32. Channels with larger activation magnitudes retain larger weights in Eq. (10). For convolutional projections, each input-channel statistic is repeated across the corresponding $k _ { h } k _ { w }$ entries of the matricized operator. Calibration requires no ground-truth restored images, since it estimates feature magnitudes rather than a restoration loss. By default, ASVD uses 16 randomly sampled compressed images (all 14 images for ICB), and its sensitivity to the calibration sample size is evaluated in Table III.

## H. Instantiation on PromptCIR

We instantiate both variants on PromptCIR [5], whose residual hybrid attention groups and transformer-style feedforward blocks contain extensive projections across the encoder, bottleneck, decoder, and prompt-conditioning pathways. Selected operators are replaced offline, while surrounding modules, prompts, normalization layers, nonlinearities, and inference procedures remain unchanged.

## V. EXPERIMENTS

## A. Experimental Setup

We evaluate the framework on four blind compressed image restoration benchmarks: LIVE1 [58], BSDS500 [59], ICB [26], and DIV2K [25]. Low-quality inputs are generated by JPEG compression with quality factors uniformly sampled from [10, 70], which spans mild to severe degradation.

We compare against JPEG, FBCNN [28], Contrastive JPEG [29], and prompt-based transformer frameworks including PromptIR [30] and PromptCIR [5]. All PromptCIR variants use identical ensemble inference and evaluation settings.

ASVD uses 16 calibration images by default (all 14 images for ICB). Calibration inputs are $1 2 8 \times 1 2 8$ JPEG images generated from the corresponding benchmark with QFs sampled uniformly from 10 to 70. Only compressed-input activations are used for factor generation; no retraining is required.

PSNR, SSIM, and PSNR-B are evaluated on RGB images and averaged over each dataset. PSNR uses the overall RGB mean squared error, whereas PSNR-B follows the channel-wise decibel averaging in SwinIR [19].

Parameter counts and MACs are reported per model for a $2 5 6 \times 2 5 6$ input, including convolution, linear projections, and attention matrix multiplication.

## B. Comparison with Representative Methods

Table I reports quantitative comparisons on the four benchmarks. Both TSVD and ASVD substantially reduce the parameter count, with restoration fidelity increasing as the rank budget grows.

At rank 64, ASVD improves PSNR-B over TSVD by 0.12– 0.24 dB across all four benchmarks, with comparable PSNR and SSIM at identical parameter and MAC budgets. The PSNR differences between the two methods are below 0.10 dB. At ranks 96 and 128, both methods approach the restoration performance of the original model.

![](images/db2037ff5584a083fc34d70e5afb5e7f12e109a8eced7d6497964addf61b2540.jpg)

TABLE I  
QUANTITATIVE COMPARISON ON COMPRESSED IMAGE RESTORATION BENCHMARKS. RESULTS ARE PSNR / SSIM / PSNR-B (↑). BEST AND SECOND-BEST VALUES AT THE DISPLAYED PRECISION ARE BOLD AND UNDERLINED. PARAMETER COUNTS ARE PER MODEL; PROMPTCIR AND ITS FACTORIZED VARIANTS USE THE SAME ENSEMBLE INFERENCE PROTOCOL.
<table><tr><td>Method</td><td>Rank</td><td>Params. (M)</td><td>LIVE1</td><td>BSDS500</td><td>ICB</td><td>DIV2K</td></tr><tr><td>JPEG</td><td></td><td></td><td>30.02/0.872/28.18</td><td>30.58/0.876/28.29</td><td>33.87/0.842/33.05</td><td>31.71/0.885/30.02</td></tr><tr><td>FBCNN [28]</td><td></td><td>71.92</td><td>32.08/0.904/31.44</td><td>32.41/0.903/31.45</td><td>35.90/0.865/36.33</td><td>33.96/0.918/33.41</td></tr><tr><td>Contrastive JPEG [29]</td><td></td><td>11.11</td><td>32.26/0.907/31.70</td><td>32.75/0.908/31.78</td><td>35.93/0.865/36.35</td><td>34.13/0.919/33.62</td></tr><tr><td>PromptIR [30]</td><td></td><td>36.54</td><td>32.34/0.907/31.75</td><td>32.67/0.907/31.67</td><td>36.35/0.868/36.75</td><td>34.38/0.922/33.81</td></tr><tr><td>PromptCIR [5]</td><td></td><td>34.75</td><td>32.50/0.909/31.91</td><td>32.23/0.906/31.35</td><td>36.49/0.866/36.88</td><td>34.25/0.919/33.71</td></tr><tr><td rowspan="3">PromptCIR-TSVD</td><td>64</td><td>14.34</td><td>31.75/0.898/30.89</td><td>31.71/0.896/30.62</td><td>34.91/0.850/35.03</td><td>33.19/0.907/32.44</td></tr><tr><td>96</td><td>18.44</td><td>32.42/0.908/31.92</td><td>32.24/0.905/31.47</td><td>36.36/0.864/36.78</td><td>34.15/0.918/33.64</td></tr><tr><td>128</td><td>21.80</td><td>32.47/0.909/31.90</td><td>32.25/0.906/31.38</td><td>36.44/0.865/36.84</td><td>34.22/0.918/33.69</td></tr><tr><td rowspan="3">PromptCIR-ASVD</td><td>64</td><td>14.34</td><td>31.67/0.899/31.04</td><td>31.74/0.898/30.84</td><td>34.96/0.853/35.27</td><td>33.10/0.907/32.56</td></tr><tr><td>96</td><td>18.44</td><td>32.43/0.908/31.88</td><td>32.26/0.906/31.48</td><td>36.36/0.864/36.76</td><td>34.15/0.918/33.69</td></tr><tr><td>128</td><td>21.80</td><td>32.48/0.909/31.89</td><td>32.26/0.906/31.40</td><td>36.45/0.865/36.83</td><td>34.22/0.918/33.67</td></tr></table>

ASVD reduces the parameter count by 59% at rank 64. At rank 96, it retains a 47% parameter reduction while keeping PSNR within 0.13 dB of PromptCIR across all four benchmarks. On BSDS500, ranks 96 and 128 yield a small PSNR improvement of approximately 0.03 dB. Such improvements are possible because minimizing projection error does not impose a fixed ordering on the final restoration error. The corresponding computational savings and measured latency are reported in Table VII.

## C. Qualitative and Feature-level Analysis

Fig. 3 compares the low-quality input, the ASVD output, the PromptCIR output, and the ground-truth reference. In the zoomed region, ASVD suppresses blocking artifacts while preserving object boundaries and local edge transitions.

Fig. 4 visualizes the absolute activation differences between the original PromptCIR backbone and the factorized models at encoder level 1 and decoder level 1. Under rank 64, ASVD yields lower-magnitude and more spatially coherent feature errors on the high-activation channels, consistent with the proposed activation-weighted factor generation.

## D. Ablation Study

a) Effect of rank: Table II reports the rank ablation on DIV2K. Increasing the rank generally improves restoration fidelity. ASVD achieves its largest PSNR gain over TSVD at rank 32 (0.87 dB). At ranks 16 and 64, it improves PSNR-B with PSNR differences below 0.12 dB; the two methods approach one another at higher ranks.

b) Effect of calibration sample size: Table III evaluates DIV2K at rank 64 over five random calibration samplings, with nested subsets and fixed test inputs. Results are reported as mean ± sample standard deviation. The PSNR standard deviation falls from 0.192 dB with one image to 0.032 dB with 16 images. Performance approximately plateaus around 8–16 images; increasing the sample count further does not produce a monotonic gain. These results support lightweight calibration and the default choice of 16 images.

Fig. 3. Visual comparison among the low-quality input (LQ), the ASVD output, the PromptCIR output, and the ground-truth reference (GT). Red boxes mark the zoomed region.  
TABLE II  
EFFECT OF RANK ON DIV2K. RESULTS ARE PSNR / SSIM / PSNR-B (↑).
<table><tr><td>Rank</td><td>TSVD</td><td>ASVD</td></tr><tr><td>8</td><td>28.26/0.854/28.21</td><td>28.34/0.846/28.39</td></tr><tr><td>16</td><td>30.35/0.879/30.05</td><td>30.23/0.874/30.11</td></tr><tr><td>32</td><td>30.82/0.889/30.61</td><td>31.69/0.892/31.36 33.10/0.907/32.56</td></tr><tr><td>64</td><td>33.19/0.907/32.44</td><td>34.15/0.918/33.69</td></tr><tr><td>96 128</td><td>34.15/0.918/33.64 34.22/0.918/33.69</td><td>34.22/0.918/33.67</td></tr></table>

c) Sensitivity to calibration quality range: We vary the calibration QF range among [10, 30], [40, 70], and [10, 70], holding the sampled image identities fixed across the three ranges within each seed. Each model uses 16 calibration images at rank 64 and is evaluated at fixed test QFs over three random samplings. Table IV shows that the spread of mean PSNR across calibration ranges is at most 0.015 dB on LIVE1 and 0.045 dB on DIV2K for test QFs 10–70. At the out-of-range test QF of 90, the corresponding spreads remain 0.036 and 0.053 dB. The small changes indicate low sensitivity to the tested calibration quality distributions.

![](images/b474931936884c72c9c9f5ec5c69fd2c8d0aaad26373fca543e517aaef80fbc1.jpg)

![](images/4a910164da580e66d995aac4388ff560367345754f8d7d8fad44e3de3f0b7c60.jpg)  
(b) enc1: base vs ASVD

(a) enc1: base vs TSVD  
![](images/56bdc24032fb84bea2cf904613e8c16027b21aa9f7738204f928d0320249d9b2.jpg)

![](images/877c8209c1214ae45a8ad201a924e7539bc411547ef1110af623d18bf1ba2a87.jpg)  
(c) dec1: base vs TSVD  
(d) dec1: base vs ASVD  
Fig. 4. Feature difference visualization on DIV2K at rank 64, using our ASVD instantiated on the PromptCIR backbone. Top row: encoder stage (enc1). Bottom row: decoder stage (dec1). Left: absolute feature error between the base model and TSVD. Right: absolute feature error between the base model and ASVD. Errors are computed on high-activation channels only. ASVD exhibits lower-magnitude and more spatially coherent error responses.

TABLE III  
CALIBRATION SAMPLE SIZE ON DIV2K (ASVD, RANK 64). MEAN ± SAMPLE STANDARD DEVIATION OVER FIVE SEEDS.
<table><tr><td>Images</td><td>PSNR</td><td>SSIM</td><td>PSNR-B</td></tr><tr><td>1</td><td> $3 2 . 8 6 1 { \scriptstyle \pm 0 . 1 9 2 }$ </td><td> $0 . 9 0 5 3 { \pm } 0 . 0 0 1 9$ </td><td> $3 2 . 3 1 4 { \pm } 0 . 1 5 7$ </td></tr><tr><td>2</td><td> $3 3 . 0 0 6 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 9 0 5 8 { \pm } 0 . 0 0 0 8$ </td><td> $3 2 . 4 2 9 { \scriptstyle \pm 0 . 0 9 0 }$ </td></tr><tr><td>4</td><td>33.064±0.059</td><td> $0 . 9 0 6 5 { \scriptstyle \pm 0 . 0 0 0 8 }$ </td><td> $3 2 . 4 9 5 { \pm } 0 . 0 4 4$ </td></tr><tr><td>8</td><td> $3 3 . 0 7 3 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 9 0 6 7 { \scriptstyle \pm 0 . 0 0 0 6 }$ </td><td> $3 2 . 5 1 8 { \pm } 0 . 0 2 6$ </td></tr><tr><td>16</td><td> $3 3 . 0 6 2 { \pm } 0 . 0 3 2$ </td><td> $0 . 9 0 7 0 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $3 2 . 5 1 8 { \pm } 0 . 0 2 8$ </td></tr><tr><td>32</td><td> $3 3 . 0 4 1 { \pm } 0 . 0 4 3$ </td><td> $0 . 9 0 7 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $3 2 . 5 1 1 { \pm } 0 . 0 3 4$ </td></tr></table>

d) Contribution of the weighting sides: Table V compares unweighted TSVD, input-only, output-only, and bilateral weighting on DIV2K at ranks 32 and 64. An input-only variant using the square root of the mean absolute magnitude is also included. All calibrated variants use 16 images and three seeds. Common-reference output NMSE is evaluated on a common set of 16 low-quality images, excluding the calibration identities and resized to $1 2 8 \times 1 2 8$

At rank 32, bilateral weighting has the highest mean PSNR, SSIM, and PSNR-B among the tested variants. At rank 64, input-only weighting performs better, with the square-root variant attaining the highest restoration metrics and the lowest output NMSE. Thus, the output-side proxy is useful under the tighter rank budget but does not uniformly improve upon input only scaling. The output NMSE and final restoration metrics need not induce the same ordering, because the former measures local projection errors rather than the nonlinear network output.

e) Uniform versus spectrum-based rank allocation: A shared rank cap isolates the effect of calibrated factor generation, but does not imply equal compressibility of different operators. We therefore compare uniform rank 64 with nonuniform ranks selected by a common cumulative squaredsingular-value energy threshold under a matched MAC budget. The search uses pretrained weights, not test restoration scores. Both allocations use the same DIV2K calibration set for all four benchmarks, so this controlled comparison differs from the dataset-specific calibration in Table I.

TABLE IV  
SENSITIVITY TO THE CALIBRATION QF RANGE (RANK 64, 16 IMAGES). PSNR MEAN ± SAMPLE STANDARD DEVIATION OVER THREE SEEDS. QF 90 IS OUTSIDE ALL CALIBRATION RANGES.
<table><tr><td>Dataset</td><td>Test QF</td><td>[10, 30]</td><td>[40, 70]</td><td>[10,70]</td></tr><tr><td rowspan="5">LIVE1</td><td>10</td><td> $2 7 . 4 9 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $2 7 . 4 8 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $2 7 . 4 9 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>30</td><td> $3 1 . 2 8 0 { \pm } 0 . 0 0 8$ </td><td> $3 1 . 2 8 4 { \pm } 0 . 0 1 2$ </td><td> $3 1 . 2 8 4 { \pm } 0 . 0 1 3$ </td></tr><tr><td>50</td><td> $3 2 . 9 2 7 { \pm } 0 . 0 1 4$ </td><td> $3 2 . 9 3 5 { \pm } 0 . 0 1 2$ </td><td> $3 2 . 9 3 4 { \pm } 0 . 0 1 6$ </td></tr><tr><td>70</td><td> $3 4 . 4 4 7 { \pm } 0 . 0 3 1$ </td><td> $3 4 . 4 5 9 { \pm } 0 . 0 2 5$ </td><td> $3 4 . 4 5 3 { \pm } 0 . 0 3 5$ </td></tr><tr><td>90</td><td> $3 6 . 5 9 5 { \scriptstyle \pm 0 . 0 7 0 }$ </td><td> $3 6 . 6 3 0 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $3 6 . 6 0 5 { \scriptstyle \pm 0 . 0 7 7 }$ </td></tr><tr><td rowspan="5">DIV2K</td><td>10</td><td> $2 9 . 1 3 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $2 9 . 1 2 8 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $2 9 . 1 3 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td>30</td><td> $3 2 . 9 0 8 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $3 2 . 9 0 8 { \pm } 0 . 0 2 1$ </td><td>32.922±0.019</td></tr><tr><td>50</td><td> $3 4 . 2 8 9 { \pm } 0 . 0 4 5$ </td><td> $3 4 . 2 9 7 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $3 4 . 3 1 6 { \pm } 0 . 0 3 3$ </td></tr><tr><td>70</td><td> $3 5 . 5 5 8 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td> $3 5 . 5 8 9 { \scriptstyle \pm 0 . 0 4 7 }$ </td><td> $3 5 . 6 0 2 { \scriptstyle \pm 0 . 0 4 9 }$ </td></tr><tr><td>90</td><td> $3 7 . 2 2 1 { \pm } 0 . 1 5 0$ </td><td> $3 7 . 2 4 9 { \pm } 0 . 1 0 0$ </td><td> $3 7 . 2 7 4 { \pm } 0 . 1 1 4$ </td></tr></table>

Table VI shows PSNR gains of 0.008–0.409 dB from spectrum-based allocation, with mixed SSIM and PSNR-B changes. The combined MAC budgets of the two models are 434.3195G and 434.3178G, respectively, but the parameter totals increase from 28.67M to 42.37M. Non-uniform allocation can therefore improve fidelity at matched arithmetic cost, while uniform ranks offer a smaller model. Neither allocation is claimed to be globally optimal.

f) Sensitivity to the scaling floor: On DIV2K at rank 64, varying the relative floor coefficient in Eq. (14) by 0.1×, 1×, and 10× changes PSNR by at most 0.00016 dB, SSIM by 0.000025, and PSNR-B by 0.00267 dB relative to the default. All resulting factors are finite. Restoration performance is therefore insensitive to these tested floor variations.

g) Efficiency analysis: Table VII reports model complexity and measured forward latency. At rank 64, the parameter count decreases from 34.75M to 14.34M and MACs from 258.79G to 217.16G. TSVD and ASVD have the same arithmetic cost at a fixed rank.

Forward latency is measured on an NVIDIA GeForce RTX 3090 in FP32 with batch size one and $2 5 6 \times 2 5 6$ inputs. We average three runs of 200 synchronized forward passes after 50 warm-up passes. Timing covers a single model without tiling or ensemble inference. Rank-64 ASVD reduces latency from 292.28 to 238.42 ms, yielding a 1.23× speedup.

## VI. DISCUSSION

The empirical results support the analysis-driven acceleration scheme. The profiling and spectral evidence in Fig. 1 identify projection operators as the dominant source of cost and show that their weights contain exploitable low-rank structure. Table I demonstrates the effectiveness of activation-calibrated factor generation. At rank 64, ASVD improves PSNR-B over TSVD on all four benchmarks under identical parameter and MAC budgets. This connects the motivation to evaluation: TSVD tests compact projection approximations, while the channel imbalance in Fig. 1(b) motivates activation-calibrated factor selection.

TABLE V  
WEIGHTING-SIDE ABLATION ON DIV2K. CALIBRATED RESULTS ARE MEAN ± SAMPLE STANDARD DEVIATION OVER THREE SEEDS; TSVD IS DETERMINISTIC. OUTPUT NMSE (↓) IS MEASURED ON COMMON REFERENCE INPUTS.
<table><tr><td>Rank</td><td>Weighting</td><td>PSNR</td><td>SSIM</td><td>PSNR-B</td><td>Output NMSE</td></tr><tr><td rowspan="5">32</td><td>TSVD</td><td> $3 0 . 8 2 1 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 8 8 9 1 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $3 0 . 6 1 1 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 8 0 2 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td></tr><tr><td>Input only</td><td> $3 1 . 4 5 3 { \pm } 0 . 1 6 5$ </td><td> $0 . 8 8 9 3 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $3 1 . 1 3 5 { \pm } 0 . 1 6 0$ </td><td> $0 . 0 7 2 1 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td></tr><tr><td>Output only</td><td> $3 0 . 2 8 2 { \pm } 0 . 1 7 8$ </td><td> $0 . 8 8 2 5 { \pm } 0 . 0 0 0 2$ </td><td> $3 0 . 3 5 0 { \pm } 0 . 0 9 0$ </td><td> $0 . 0 7 7 4 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td></tr><tr><td>Bilateral (ASVD)</td><td> $3 1 . 6 6 1 { \pm } 0 . 1 3 5$ </td><td> $0 . 8 9 1 9 { \pm } 0 . 0 0 1 3$ </td><td> $3 1 . 3 1 0 { \pm } 0 . 1 0 1$ </td><td> $0 . 0 7 3 7 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td></tr><tr><td>Input only  $( \alpha = 0 . 5 )$ </td><td>31.523±0.075</td><td> $0 . 8 9 1 6 { \pm } 0 . 0 0 0 9$ </td><td> $3 1 . 2 2 2 { \pm } 0 . 0 4 8$ </td><td> $0 . 0 6 9 3 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td></tr><tr><td rowspan="5">64</td><td>TSVD</td><td> $3 3 . 1 9 4 { \pm } 0 . 0 0 0$ </td><td> $0 . 9 0 7 2 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $3 2 . 4 3 7 { \pm } 0 . 0 0 0$ </td><td> $0 . 0 2 7 1 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td></tr><tr><td>Input only</td><td> $3 3 . 3 0 7 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 0 9 6 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td><td> $3 2 . 6 4 7 { \pm } 0 . 0 2 1$ </td><td> $0 . 0 2 3 2 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td></tr><tr><td>Output only</td><td> $3 2 . 7 0 7 { \scriptstyle \pm 0 . 0 8 4 }$ </td><td> $0 . 9 0 2 8 { \pm } 0 . 0 0 0 2$ </td><td> $3 2 . 1 5 0 { \pm } 0 . 0 7 2$ </td><td> $0 . 0 2 6 7 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td></tr><tr><td>Bilateral (ASVD)</td><td> $3 3 . 0 8 0 { \pm } 0 . 0 3 0$ </td><td> $0 . 9 0 6 8 { \pm } 0 . 0 0 0 5$ </td><td> $3 2 . 5 2 7 { \pm } 0 . 0 3 0$ </td><td> $0 . 0 2 4 3 { \pm } 0 . 0 0 0 1$ </td></tr><tr><td>Input only  $( \alpha = 0 . 5 )$ </td><td> $3 3 . 4 1 1 { \pm } 0 . 0 0 3$ </td><td> $0 . 9 1 0 0 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $3 2 . 6 7 6 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 0 2 2 4 { \pm } 0 . 0 0 0 0$ </td></tr></table>

TABLE VI

RANK ALLOCATION AT MATCHED MACS. RESULTS ARE PSNR / SSIM / PSNR-B (↑). BOTH METHODS USE THE SAME FIXED DIV2K CALIBRATION SET.
<table><tr><td>Dataset</td><td>Uniform 64</td><td>Spectrum-based</td></tr><tr><td>LIVE1</td><td>31.69/0.899/31.03</td><td>31.74/0.899/30.97</td></tr><tr><td>BSDS500</td><td>31.71/0.898/30.81</td><td>31.72/0.898/30.69</td></tr><tr><td>ICB</td><td>34.79/0.847/35.12</td><td>35.20/0.853/35.40</td></tr><tr><td>DIV2K</td><td>33.10/0.907/32.56</td><td>33.28/0.908/32.61</td></tr></table>

TABLE VII

COMPLEXITY AND 256 × 256 FORWARD LATENCY. PARAMETERS AND MACS ARE PER MODEL; MACS INCLUDE ATTENTION MATRIX PRODUCTS. LATENCIES ARE IN MS; SPEEDUP IS ASVD RELATIVE TO THE ORIGINAL.
<table><tr><td>Rank</td><td>Params. (M)</td><td>MACs (G)</td><td>TSVD (ms)</td><td>ASVD (ms)</td><td>Speedup (×)</td></tr><tr><td>32</td><td>9.79</td><td>164.15</td><td>208.26</td><td>213.19</td><td>1.37</td></tr><tr><td>64</td><td>14.34</td><td>217.16</td><td>215.72</td><td>238.42</td><td>1.23</td></tr><tr><td>96</td><td>18.44</td><td>254.38</td><td>228.53</td><td>240.56</td><td>1.21</td></tr><tr><td>128</td><td>21.80</td><td>263.19</td><td>244.83</td><td>246.16</td><td>1.19</td></tr></table>

The rank ablation in Table II further clarifies the role of activation calibration. At very small ranks, the factorized operators have limited capacity, and the choice of retained subspace determines most of the restoration error. The weighted objective favors preservation of active channels. ASVD achieves its largest DIV2K PSNR gain over TSVD at rank 32 and consistently improves PSNR-B at rank 64, with comparable PSNR and SSIM. As the rank grows, both decompositions recover a larger fraction of the pretrained operator, and the gap between TSVD and ASVD naturally narrows. This trend distinguishes ASVD from runtime augmentation: it improves the offline factor selection under the same rank and complexity budget. The feature visualizations in Fig. 4 provide representation-level evidence, with ASVD producing lower and more coherent errors on high-activation channels.

The calibration-size study in Table III supports the use of a small compressed calibration set. The statistics used by our ASVD are per-channel activation magnitudes rather than task loss gradients or reconstructed-image supervision; they simply measure which input and output channels respond more strongly when the pretrained model processes compressed images. The multi-seed results indicate that a small set can stabilize these coarse channel weights, with performance approximately plateauing around 8–16 images. The quality-range study also shows low sensitivity to the tested calibration distributions. This stability supports blind CIR, where the model restores compressed inputs without receiving their JPEG quality factors.

The operator-level formulation in Fig. 2 keeps the accelerated model close to the original restoration backbone. Each selected projection is replaced locally by two factors, while attention reshaping, prompt interaction, normalization, nonlinear activation, skip connections, and output reconstruction remain unchanged. This locality avoids conflating low-rank replacement with architectural redesign and enables a controlled comparison: at a fixed rank, TSVD and ASVD share the same parameter count, MACs, tensor interfaces, and inference graph. Consequently, their observed difference is attributable to activation-calibrated factor generation rather than topology or runtime computation.

The choice of the rank-64 operating point follows the same evidence in Tables II and VII. It reduces parameters by 58.75% and MACs by 16.09% relative to the original PromptCIR backbone. Rank 96 reduces parameters by 46.93% but MACs by only 1.70%; rank 128 exceeds the original MAC count by 1.70%. Higher ranks are useful for demonstrating the convergence of TSVD and ASVD and for retaining restoration fidelity, but offer less arithmetic reduction. Rank 64 provides a compact model with measurable PSNR-B improvements over TSVD, while rank 96 offers restoration performance close to the original backbone. At rank 96, the PSNR difference from the original model ranges from −0.13 to +0.03 dB, illustrating the complementary higher-fidelity operating regime.

The method also offers a favorable deployment profile. All factor generation is performed offline, and the compressed model retains the original PromptCIR topology, prompts, normalization layers, and restoration head. ASVD is therefore complementary to pruning and quantization: it reduces the algebraic cost of dense projections while leaving precision reduction and sparsity-based acceleration available for deployment.

## VII. CONCLUSION

We presented a training-free framework for accelerating transformer-based blind compressed image restoration. An empirical analysis of prompt-based restoration transformers shows that projection operators dominate the parameter and MAC cost, that their singular-value spectra exhibit operatordependent low-rank structure, and that their compressed-image activations are strongly channel-imbalanced. These observations support TSVD as a principled training-free baseline and explain why a uniform low-rank approximation can fall short under aggressive rank budgets. Building on this rationale, we formulated projection-level replacement with TSVD as the closed-form baseline and designed ASVD as our activationcalibrated refinement that incorporates lightweight activation statistics into the decomposition objective. The resulting factors are generated entirely offline from compressed calibration inputs while preserving the original restoration architecture and inference flow. Experiments on four standard blind CIR benchmarks show that ASVD consistently improves PSNR-B over TSVD at rank 64 under identical parameter and MAC budgets, while maintaining comparable PSNR and SSIM. These results indicate that activation-aware factor generation can compress restoration transformers without retraining data, task specific supervision, or online adaptation.

The main evaluation uses a shared rank cap and one promptbased backbone; a spectrum-based allocation study additionally examines non-uniform ranks under a matched MAC budget. Promising future directions include further optimization of operator-wise or layer-wise rank allocation guided by spectral and activation profiles, combining ASVD with quantization and pruning for further compression, and extending it to other transformer-based restoration tasks.

## REFERENCES

[1] Anas M Ali, Bilel Benjdira, Anis Koubaa, Walid El-Shafai, Zahid Khan, and Wadii Boulila, “Vision transformers in image restoration: A survey,” Sensors, vol. 23, no. 5, pp. 2385, 2023.

[2] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li, “Uformer: A general u-shaped transformer for image restoration,” in CVPR, 2022, pp. 17683–17693.

[3] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in CVPR, 2022, pp. 5728–5739.

[4] Xiangyu Chen, Xintao Wang, Wenlong Zhang, Xiangtao Kong, Yu Qiao, Jiantao Zhou, and Chao Dong, “Hat: Hybrid attention transformer for image restoration,” arXiv preprint arXiv:2309.05239, 2023.

[5] Bingchen Li, Xin Li, Yiting Lu, Ruoyu Feng, Mengxi Guo, Shijie Zhao, Li Zhang, and Zhibo Chen, “Promptcir: Blind compressed image restoration with prompt learning,” in CVPR, 2024, pp. 6442–6452.

[6] Carl Eckart and Gale Young, “The approximation of one matrix by another of lower rank,” Psychometrika, vol. 1, no. 3, pp. 211–218, 1936.

[7] Xianming Liu, Xiaolin Wu, Jiantao Zhou, and Debin Zhao, “Data-driven sparsity-based restoration of jpeg-compressed images in dual transform pixel domain,” in CVPR, 2015, pp. 5171–5178.

[8] Chao Dong, Yubin Deng, Chen Change Loy, and Xiaoou Tang, “Compression artifacts reduction by a deep convolutional network,” in ICCV, 2015, pp. 576–584.

[9] Zhangyang Wang, Ding Liu, Shiyu Chang, Qing Ling, Yingzhen Yang, and Thomas S Huang, “D3: Deep dual-domain based fast restoration of jpeg-compressed images,” in CVPR, 2016, pp. 2764–2772.

[10] Xiaoshuai Zhang, Wenhan Yang, Yueyu Hu, and Jiaying Liu, “Dmcnn: Dual-domain multi-scale convolutional neural network for compression artifacts removal,” in ICIP. IEEE, 2018, pp. 390–394.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.

[12] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang, “Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising,” IEEE transactions on image processing, vol. 26, no. 7, pp. 3142–3155, 2017.

[13] Xiaojiao Mao, Chunhua Shen, and Yu-Bin Yang, “Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections,” NeurIPS, vol. 29, 2016.

[14] Ying Tai, Jian Yang, Xiaoming Liu, and Chunyan Xu, “Memnet: A persistent memory network for image restoration,” in ICCV, 2017, pp. 4539–4547.

[15] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu, “Image super-resolution using very deep residual channel attention networks,” in ECCV, 2018, pp. 286–301.

[16] Yulun Zhang, Kunpeng Li, Kai Li, Bineng Zhong, and Yun Fu, “Residual non-local attention networks for image restoration,” arXiv preprint arXiv:1903.10082, 2019.

[17] Max Ehrlich, Larry Davis, Ser-Nam Lim, and Abhinav Shrivastava, “Quantization guided jpeg artifact correction,” in ECCV. Springer, 2020, pp. 293–309.

[18] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” NIPS, vol. 30, 2017.

[19] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte, “Swinir: Image restoration using swin transformer,” in ICCV, 2021, pp. 1833–1844.

[20] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong, “Activating more pixels in image super-resolution transformer,” in CVPR, 2023, pp. 22367–22377.

[21] Fu-Jen Tsai, Yan-Tsung Peng, Yen-Yu Lin, Chung-Chi Tsai, and Chia-Wen Lin, “Stripformer: Strip transformer for fast image deblurring,” in ECCV. Springer, 2022, pp. 146–162.

[22] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun, “Simple baselines for image restoration,” in ECCV. Springer, 2022, pp. 17–33.

[23] Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia, “Mambair: A simple baseline for image restoration with state-space model,” in ECCV. Springer, 2024, pp. 222–241.

[24] Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, et al., “Lsdir: A large scale dataset for image restoration,” in CVPR, 2023, pp. 1775–1787.

[25] Eirikur Agustsson and Radu Timofte, “Ntire 2017 challenge on single image super-resolution: Dataset and study,” in CVPR workshops, 2017, pp. 126–135.

[26] Rawzor, “Image compression benchmark,” [Online]. Available: http: //imagecompression.info/.

[27] Ren Yang, Radu Timofte, Xin Li, Qi Zhang, Lin Zhang, Fanglong Liu, Dongliang He, Fu Li, He Zheng, Weihang Yuan, et al., “Aim 2022 challenge on super-resolution of compressed image and video: Dataset, methods and results,” in ECCV. Springer, 2022, pp. 174–202.

[28] Jiaxi Jiang, Kai Zhang, and Radu Timofte, “Towards flexible blind jpeg artifacts removal,” in ICCV, 2021, pp. 4997–5006.

[29] Xi Wang, Xueyang Fu, Yurui Zhu, and Zheng-Jun Zha, “Jpeg artifacts removal via contrastive representation learning,” in ECCV. Springer, 2022, pp. 615–631.

[30] Vaishnav Potlapalli, Syed Waqas Zamir, Salman H Khan, and Fahad Shahbaz Khan, “Promptir: Prompting for all-in-one image restoration,” NeurIPS, vol. 36, pp. 71275–71293, 2023.

[31] Jiaqi Ma, Tianheng Cheng, Guoli Wang, Qian Zhang, Xinggang Wang, and Lefei Zhang, “Prores: Exploring degradation-aware visual prompt for universal image restoration,” arXiv preprint arXiv:2306.13653, 2023.

[32] Tao Wang, Wanglong Lu, Kaihao Zhang, Wenhan Luo, Tae-Kyun Kim, Tong Lu, Hongdong Li, and Ming-Hsuan Yang, “Promptrr: Diffusion models as prompt generators for single image reflection removal,” arXiv preprint arXiv:2402.02374, 2024.

[33] Zilong Li, Yiming Lei, Chenglong Ma, Junping Zhang, and Hongming Shan, “Prompt-in-prompt learning for universal image restoration,” arXiv preprint arXiv:2312.05038, 2023.

[34] Ziwei Luo, Fredrik K Gustafsson, Zheng Zhao, Jens Sjolund, and ¨ Thomas B Schon, “Controlling vision-language models for universal¨ image restoration,” arXiv preprint arXiv:2310.01018, 2023.

[35] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim, “Visual prompt tuning,” in ECCV. Springer, 2022, pp. 709–727.

[36] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan, “Maple: Multi-modal prompt learning,” in CVPR, 2023, pp. 19113–19122.

[37] Kihyuk Sohn, Huiwen Chang, Jose Lezama, Luisa Polania, Han Zhang,´ Yuan Hao, Irfan Essa, and Lu Jiang, “Visual prompt tuning for generative transfer learning,” in CVPR, 2023, pp. 19840–19851.

[38] Xin Li, Bingchen Li, Yeying Jin, Cuiling Lan, Hanxin Zhu, Yulin Ren, and Zhibo Chen, “Ucip: A universal framework for compressed image super-resolution using dynamic prompt,” in ECCV. Springer, 2024, pp. 107–125.

[39] Yulin Ren, Xin Li, Mengxi Guo, Bingchen Li, Shijie Zhao, and Zhibo Chen, “Mambacsr: Dual-interleaved scanning for compressed image super-resolution with ssms,” arXiv preprint arXiv:2408.11758, 2024.

[40] Jinpei Guo, Zheng Chen, Wenbo Li, Yong Guo, and Yulun Zhang, “Compression-aware one-step diffusion model for jpeg artifact removal,” in ICCV, 2025, pp. 14930–14939.

[41] Tailin Liang, John Glossner, Lei Wang, Shaobo Shi, and Xiaotong Zhang, “Pruning and quantization for deep neural network acceleration: A survey,” Neurocomputing, vol. 461, pp. 370–403, 2021.

[42] Song Han, Jeff Pool, John Tran, and William Dally, “Learning both weights and connections for efficient neural network,” NeurIPS, vol. 28, 2015.

[43] Song Han, Huizi Mao, and William J Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,” arXiv preprint arXiv:1510.00149, 2015.

[44] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang, “Learning efficient convolutional networks through network slimming,” in ICCV, 2017, pp. 2736–2744.

[45] Emily L Denton, Wojciech Zaremba, Joan Bruna, Yann LeCun, and Rob Fergus, “Exploiting linear structure within convolutional networks for efficient evaluation,” NeurIPS, vol. 27, 2014.

[46] Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman, “Speeding up convolutional neural networks with low rank expansions,” arXiv preprint arXiv:1405.3866, 2014.

[47] Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan Oseledets, and Victor Lempitsky, “Speeding-up convolutional neural networks using fine-tuned cp-decomposition,” arXiv preprint arXiv:1412.6553, 2014.

[48] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang, “Svd-llm: Truncation-aware singular value decomposition for large language model compression,” arXiv preprint arXiv:2403.07378, 2024.

[49] Viktoriia Chekalina, Daniil Moskovskiy, Daria Cherniuk, Maxim Kurkin, Andrey Kuznetsov, and Evgeny Frolov, “Generalized fisher-weighted svd: Scalable kronecker-factored fisher approximation for compressing large language models,” arXiv preprint arXiv:2505.17974, 2025.

[50] Xinyin Ma, Gongfan Fang, and Xinchao Wang, “Deepcache: Accelerating diffusion models for free,” in CVPR, 2024, pp. 15762–15772.

[51] Felix Wimbauer, Bichen Wu, Edgar Schoenfeld, Xiaoliang Dai, Ji Hou, Zijian He, Artsiom Sanakoyeu, Peizhao Zhang, Sam Tsai, Jonas Kohler, et al., “Cache me if you can: Accelerating diffusion models through block caching,” in CVPR, 2024, pp. 6211–6220.

[52] Zhiyuan Chen, Keyi Li, Yifan Jia, Le Ye, and Yufei Ma, “Accelerating diffusion transformer via increment-calibrated caching with channelaware singular value decomposition,” in CVPR, 2025, pp. 18011–18020.

[53] Pratheba Selvaraju, Tianyu Ding, Tianyi Chen, Ilya Zharkov, and Luming Liang, “Fora: Fast-forward caching in diffusion transformer acceleration,” arXiv preprint arXiv:2407.01425, 2024.

[54] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang, “Learning to-cache: Accelerating diffusion transformer via layer caching,” NeurIPS, vol. 37, pp. 133282–133304, 2024.

[55] Yushi Huang, Zining Wang, Ruihao Gong, Jing Liu, Xinjie Zhang, Jinyang Guo, Xianglong Liu, and Jun Zhang, “Harmonica: Harmonizing training and inference for better feature caching in diffusion transformer acceleration,” arXiv preprint arXiv:2410.01723, 2024.

[56] Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen, “∆-DiT: A training-free acceleration method tailored for diffusion transformers,” arXiv preprint arXiv:2406.01125, 2024.

[57] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang, “Accelerating diffusion transformers with token-wise feature caching,” arXiv preprint arXiv:2410.05317, 2024.

[58] H. R. Sheikh, Z. Wang, L. Cormack, and A. C. Bovik, “LIVE image quality assessment database release 2,” [Online]. Available: https://live. ece.utexas.edu/research/quality/, 2005.

[59] David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik, “A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics,” in ICCV. IEEE, 2001, vol. 2, pp. 416–423.

![](images/758441e3a040f8605105753ebcb6da4d7e83df816d8d626cb5170cb1c17bde21.jpg)

Suhang Yao is currently a first-year graduate student at the University of Science and Technology of China (USTC). His research interests focus on lowlevel vision, including image restoration, image superresolution, and preference optimization, as well as world models.

![](images/491468dadc345ef50c400528f4fa8973f4f06afeef36e8ea604e34317d9561cc.jpg)

Xin Li (M’21) received the B.Sc. and Ph.D. degrees from the Department of Electronic Engineering and Information Science, University of Science and Technology of China, in 2019 and 2024, respectively, where he is now an associate researcher. His research interests include image/video compression, restoration, quality assessment, AIGC, and AI agents. He has more than 50 publications and won the IEEE VCIP 2023 Best Paper Award, ACM SIGMM China Outstanding Doctoral Dissertation Award, ByteDance Scholarship, and multiple first prizes in international competitions, including CLIC2022, CLIC2024, and CVPR NTIRE 2024. He has been involved in the standard committee MPEG AG5. He has served as an Area Chair of ICLR, NeurIPS, ACM Multimedia, and BMVC. He has also served as a regular reviewer of IEEE Transactions on Pattern Analysis and Machine Intelligence, IEEE Transactions on Image Processing, IEEE Transactions on Circuits and Systems for Video Technology, IEEE Transactions on Multimedia, and IEEE Journal on Selected Areas in Communications.

![](images/1200282ed6dcb59dcc956adc16415eb7127f93a5fac792625c1c512d9f2ec8e1.jpg)

Xijun Wang is currently a second-year graduate student at the University of Science and Technology of China (USTC). His research interests focus on lowlevel vision, including video restoration, video superresolution, as well as world models.

![](images/d648fae5cf4b7f7a0620cef191cb696c4b464ed095621b7260c627e328ce76e5.jpg)

Yufei Wang received the B.Eng. degree from the Uni versity of Electronic Science and Technology of China (UESTC), China, in 2020, and the Ph.D. degree from Nanyang Technological University (NTU), Singapore, in 2025. He is currently a co-founder of SparcAI Inc., where he leads research on 3D/4D generation, video, and world models. His research interests include computer vision, generative AI, 3D vision, and com putational photography. He has published dozens of papers in venues such as CVPR, ICCV, ECCV, NeurIPS,

ICML, TPAMI, and IJCV. He received the Best Ph.D. Thesis Award and was the sole recipient of NTU’s Research Excellence Award in STEM in 2025.

![](images/cf2bace9bd338134982f38fbde45e658052011bddda7c7ca2dd8509c826f7cc6.jpg)

Yeying Jin received the B.Eng. degree in electronic in formation engineering from the University of Electronic Science and Technology of China, Chengdu, China, in 2017, and the M.S. and Ph.D. degrees in electrical and computer engineering from the National University of Singapore (NUS), Singapore, in 2018 and 2024, respectively. She is currently a Staff Researcher with Tencent, Singapore, leading research on world models, agentic AI, and AIGC, and an Adjunct Faculty Member with the Department of Electrical and Computer

Engineering, NUS. Her research interests include game world models, agentic video generation, and coding agents. In 2023, she was a Research Intern with Adobe Research, San Jose, CA, USA. Her doctoral research was supported by the AI Singapore Ph.D. Fellowship, and she received the ICML Gold Reviewer Award. She has authored or coauthored more than 70 papers and founded the Workshop on Low-Level Vision Frontiers (LoViF). She serves as a reviewer for IEEE TPAMI, IJCV, TIP, TCSVT, and TMM.

![](images/234535c0c59e14a364b9ba82ddeaa4b2521e6c380ebab4cc17797db058d5eaec.jpg)

Zhibo Chen (M’01–SM’11) received the B.Sc. and Ph.D. degrees from the Department of Electrical Engi neering, Tsinghua University, in 1998 and 2003, respec tively. He is now a full professor at the University of Science and Technology of China. His research interests focus on investigating artificial intelligence techniques for advanced visual signal generation, representation, processing and coding, as well as other interdisciplinary research fields. He has more than 200 publications and over 100 granted patent applications. Some of his standard proposals have been adopted in MPEG/VCEG on video coding and ITU-T P.1202 on video quality assessment. He served as Chair of the Technical Committee of IEEE CASS Visual Signal Processing and Communications (VSPC) from 2021 to 2024, Senior Associate Editor of IEEE Transactions on Circuits and Systems for Video Technology, and Corresponding Guest Editor of IEEE JETCAS and IEEE OJCAS. He is currently also a member of the IEEE SP Image, Video, and Multidimensional Signal Processing (IVMSP) Technical Committee, the IEEE CASS Multimedia Systems and Applications (MSA) Technical Committee, and the IEEE CASS Circuits and Systems Education and Outreach (CASEO) Technical Committee.