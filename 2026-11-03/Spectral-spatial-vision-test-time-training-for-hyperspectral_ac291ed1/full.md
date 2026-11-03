# Spectral-spatial vision test-time training for hyperspectral image classification

Xiang Zhao<sup>a,\*</sup>

<sup>a</sup>College of Mathematical Sciences, Harbin Engineering University, 145 Nantong Street, Nangang District, Harbin 150001, China

Abstract. Hyperspectral image (HSI) classification requires the representation of contiguous spectral signatures and spatial context under limited labeled supervision. Vision test-time training (TTT) offers an alternative to fixed-weight token mixing by performing self-supervised inner-model updates on the input sequence, but its application to centerpixel HSI classification remains insufficiently characterized. We present ViTTT-HSI, a spectral–spatial adaptation of the vision test-time training (ViT<sup>3</sup>) architecture. A spectral-aware embedding stem extracts ordered-band information using spectral convolution and forms spatial tokens without reducing their spatial resolution. Two ViT<sup>3</sup>-based TTT encoder blocks perform patch-conditioned token mixing, followed by the fusion of center-pixel and neighborhoodaverage features for classification. We evaluate the framework on Indian Pines, Pavia University, Salinas, and Houston 2013 using ten random seeds and dataset-specific training fractions of 1% or 5%. The mean overall accuracies are 97.6390%, 98.8753%, 99.3292%, and 98.4283%, respectively. Ablations reveal a substantial contribution from the TTT inner update on Indian Pines and a smaller positive contribution on Pavia University, while the relative benefits of the spectral stem and feature fusion depend on the dataset. The results demonstrate the feasibility of using patch-conditioned vision TTT for spectral–spatial HSI classification and highlight the role of the inner adaptation mechanism.

Keywords: hyperspectral image classification, test-time training, spectral–spatial representation, center-pixel classifi cation, remote sensing.

\*Corresponding author: Xiang Zhao, 2021241331@hrbeu.edu.cn

## 1 Introduction

Hyperspectral images (HSIs) record a dense sequence of spectral measurements for each spatial location, providing discriminative signatures for identifying vegetation, built surfaces, and other land-cover materials. Their classification is a fundamental step in remote sensing applications such as agricultural monitoring, environmental assessment, and urban mapping. Compared with conventional three-channel images, HSI cubes contain substantially more spectral channels and often exhibit pronounced correlations between neighboring bands. At the same time, reliable pixel annotations can be costly to obtain, making accurate classification under limited supervision a persistent research challenge.<sup>1,</sup> <sup>2</sup>

A classifier based solely on the spectrum of a target pixel may be sensitive to local spectral variation and similarities between distinct materials. Incorporating neighboring pixels provides complementary spatial evidence, but also requires the model to accommodate boundaries and heterogeneous patches. Deep convolutional networks have addressed this issue by learning local spectral and spatial features through 3-D and 2-D operators.<sup>3,</sup> <sup>4</sup> More recently, Transformers have offered flexible sequence-level interactions for spectral and spatial feature learning, as illustrated by SpectralFormer and the spectral–spatial feature tokenization Transformer (SSFTT).<sup>5,</sup> <sup>6</sup> Selective state-space models have also been used for hyperspectral representation learning, with S<sup>2</sup>Mamba combining spatial and spectral sequence processing.

These advances offer increasingly expressive spectral–spatial encoders. A complementary question is whether the token-mixing operation itself can adapt to the particular neighborhood being classified. Test-time training (TTT) addresses sequence modeling through temporary, inputdependent inner-model updates: features from the current sequence form key–value training pairs, the inner model is updated by an unlabeled objective, and the adapted model then processes query features.<sup>8</sup> In vision, Han et al. developed ViT<sup>3</sup> and studied how inner-training objectives, optimization settings, gated units, and depthwise convolutional inner models affect visual token mixing.<sup>9</sup> This approach provides a patch-conditioned representation without requiring class labels at inference.

Applying a visual TTT architecture to HSI center-pixel recognition requires changes at the input and prediction interfaces. First, the spectral axis carries a meaningful band ordering that is absent from standard red, green, and blue (RGB) patch embeddings. Second, the target is the label of the center pixel rather than a single label for the entire image. Third, spatial positions should remain explicitly associated with their original neighborhoods so that center information can be recovered after token interaction. These requirements motivate a spectral-aware embedding and a center-oriented prediction head around a ViT<sup>3</sup>-based spatial encoder.

In this work, we construct ViTTT-HSI, a spectral–spatial adaptation of vision TTT for HSI center-pixel classification. A spectral convolutional stem first extracts ordered-band responses and creates one token per spatial position. Two encoder blocks then apply patch-conditioned TTT mixing while retaining the complete spatial token grid. Finally, the normalized center token is combined with an average representation of the neighborhood for the class prediction. The TTT inner models and their update rules follow ViT<sup>3</sup>;<sup>9</sup> the hyperspectral embedding, token organization, and center–global prediction pathway form the task-specific adaptation studied here.

The main contributions are summarized as follows. (1) We establish a vision TTT-based spectral–spatial framework for HSI center-pixel classification, maintaining spatial correspondence throughout the token-processing stages. (2) We introduce a spectral-aware embedding pathway that exploits local spectral continuity while generating spatially aligned features suitable for ViT<sup>3</sup> mixing. (3) We evaluate the framework across four public HSI datasets and ten random seeds, and examine the roles of the TTT inner update, input embedding, center–global feature aggregation, spatial context size, and spatially separated sampling. The controlled inner-update comparisons provide a direct assessment of the patch-conditioned learning mechanism.

## 2 Related work

## 2.1 Spectral–spatial feature learning in hyperspectral classification

HSI classification methods must combine rich spectral descriptions with spatial relationships. Earlier deep spectral–spatial architectures typically encode a neighborhood centered at the target pixel and assign the resulting feature vector to the center label. The spectral–spatial residual network (SSRN) uses consecutive spectral and spatial residual blocks in a 3-D convolutional architecture.<sup>3</sup> HybridSN subsequently couples 3-D convolutional feature extraction with 2-D convolution to learn joint representations at different stages.<sup>4</sup> These methods demonstrate the utility of treating local spectral structure and image-space context as complementary sources of information.

A local convolutional receptive field, however, is not the only means of representing dependencies across an HSI patch. Motivated by attention-based sequence modeling,<sup>10</sup> SpectralFormer represents spectral signatures as grouped embeddings and incorporates spectral information flow across its Transformer layers.<sup>5</sup> SSFTT employs spectral–spatial feature tokenization before Transformer encoding, thereby constructing compact tokens from convolutional spatial–spectral features.<sup>6</sup> More recently, the dual selective fusion Transformer network (DSFormer) has used selec tive fusion of spatial–spectral features and contextual tokens to capture information across different receptive fields.<sup>11</sup> HyperTransXNet has explored local–global dynamics through hybrid spectral– spatial processing and dynamic token interactions.<sup>12</sup> The corresponding design space encompasses both the representation of spectral bands and the organization of spatial neighborhoods.

## 2.2 State-space and alternative token mixers

Attention-based methods have encouraged parallel work on sequence-processing operators with different computational trade-offs. The selective state-space model Mamba provides an inputdependent state transition mechanism for sequential representations.<sup>13</sup> In the HSI domain, S<sup>2</sup>Mamba constructs complementary spatial and spectral processing pathways and combines their outputs through a spatial–spectral mixture gate.<sup>7</sup> Spatial-frequency joint learning Mamba (SFMamba) further explores spatial and frequency-domain information in hyperspectral classification.<sup>14</sup> These approaches, together with Transformer and hybrid convolutional neural network (CNN) architectures, constitute relevant baselines for evaluating another form of token interaction.

A central distinction of TTT-based mixing is that the current input sequence is used to construct temporary inner-model parameters rather than only to produce activations through a fixed mixing operator. Thus, TTT permits the model state used for query processing to depend on the content of the observed patch. In this work, the practical benefit of this mechanism is examined through comparisons with both a removed-mixer variant and a frozen-inner-model variant.

## 2.3 Test-time training versus test-time adaptation

In conventional test-time adaptation (TTA), a trained classifier can update selected model param eters using unlabeled target-domain observations, often to address corruption or distribution shift. HyperTTA, for example, studies hyperspectral classification under several forms of image degradation and adapts normalization parameters using confidence-aware entropy minimization.<sup>15</sup> Its purpose is to improve performance when test inputs depart from the source-domain distribution.

Test-time training in the sense of Sun et al.<sup>8</sup> and ViT<sup>39</sup> serves a different architectural role. An unlabeled inner objective is evaluated on key–value representations of the present sequence; the resulting fast weights process queries, while shared outer parameters remain unchanged during inference. In ViTTT-HSI, fast weights are reinitialized from their learned initial values for every patch. There is no persistent adaptation across patches, and no target class annotation participates in the inner update. The present study therefore focuses on TTT as an input-conditioned spatial token mixer for center-pixel classification rather than as a mechanism for adapting to external domain shifts.

## 3 Methodology

## 3.1 Problem formulation and framework overview

Let $\mathbf { I } \in \mathbb { R } ^ { H \times W \times C }$ represent an HSI cube, where C is the number of spectral bands. For a labeled center coordinate $p ,$ an odd-sized neighborhood $\mathbf { X } _ { p } \in \mathbb { R } ^ { C \times P \times P }$ is extracted. The goal is to predict the center label $y _ { p } \in \{ 1 , \ldots , K \}$ without using ground-truth labels from other pixels in the patch

during inner adaptation. Figure 1 summarizes the architecture. For a batch of B patches, the embedding stage produces $\bar { \mathbf { T } } ^ { ( 0 ) } \in \mathbb { R } ^ { B \times N \times D }$ , where $N = P ^ { 2 }$ and $D = 6 4$ . The spatial tokens pass through $L = 2$ encoder blocks, are normalized, and then feed a center–global classifier.

![](images/c7cef4ddb782746102d0aa5bd2901b87746fa1f71c443d0cd7e2cf490609c4ff.jpg)  
B: batch size C: spectral bands K: classes LN: LayerNorm CPE: convolutional positional encoding  
Fig 1 Overall architecture of ViTTT-HSI. (a) A centered hyperspectral patch is encoded without spatial downsampling, processed by two TTT encoder blocks, and classified using a center–global feature head. (b) Each encoder block uses residual convolutional positional encoding (CPE), a pre-normalized TTT mixer inherited from $\mathrm { V i T ^ { 3 } , ^ { 9 } }$ and a pre-normalized feed-forward network. (c) The center token and spatial average are concatenated for center-pixe prediction. C, D, N, and K denote spectral bands, token dimension, token count, and classes.

## 3.2 Spectral-aware feature embedding

Because the spectral axis has an intrinsic ordering, the embedding stem applies a convolution along spectral bands, without convolving over spatial coordinates in its first stage. A batch $\mathbf { X \in }$ $\mathbb { R } ^ { B \times \mathbf { \bar { C } } \times \mathbf { \bar { P } } \times P }$ is reshaped to $\mathbb { R } ^ { B \times 1 \times C \times P \times P }$ . An eight-channel 3-D convolution with kernel (7, 1, 1), stride (4, 1, 1), and spectral padding 3 is followed by a Gaussian error linear unit (GELU) and adaptive average pooling to 16 spectral bins. This operation yields $\mathbb { R } ^ { B \times 8 \times 1 6 \times P \times P }$

Folding the channel and pooled spectral axes produces 128 channels at each spatial position. A pointwise 1 × 1 convolution maps these features to D = 64 channels, followed by batch normalization and GELU. A spatial depthwise $3 \times 3$ convolution and another GELU further refine local spatial context. The resulting $\mathbf { F } \mathbf { \bar { \Pi } } \in \mathbb { R } ^ { B \times D \times P \times P }$ is flattened only along its spatial axes:

$$
\begin{array} { r } { \mathbf { T } ^ { ( 0 ) } = \mathrm { F l a t t e n } _ { H W } ( \mathrm { S t e m } ( \mathbf { X } ) ) ^ { \top } \in \mathbb { R } ^ { B \times P ^ { 2 } \times D } . } \end{array}\tag{1}
$$

The superscript ⊤ in Eq. (1) indicates the channel-to-feature-axis permutation, not a transposition of the full batch tensor. No spatial token is discarded and no class token is introduced. The linearstem ablation replaces spectral convolution and pooling with a direct channel projection while retaining the remaining spatial embedding pathway.

## 3.3 TTT-based spatial encoder

Each encoder block first introduces a residual convolutional positional encoding, $\widehat { \textbf { T } } = \textbf { T } +$ Flatten ${ \bf \nabla } _ { H W } ( \mathrm { D W C o n v _ { 3 \times 3 } ( T ) } )$ . The subsequent pre-normalized token mixer and feed-forward sublayer are residual:

$$
\mathbf { U } = \widehat { \mathbf { T } } + \mathrm { T T T } ( \mathrm { L N } ( \widehat { \mathbf { T } } ) ) ,\tag{2}
$$

$$
\mathbf { T } ^ { \prime } = \mathbf { U } + \operatorname { F F N } ( \operatorname { L N } ( \mathbf { U } ) ) .\tag{3}
$$

Here, DWConv denotes depthwise convolution, LN denotes layer normalization, and FFN denotes a feed-forward network. The feed-forward network uses two linear transformations with dimensions 64 → 128 → 64, GELU, and dropout with probability 0.1.

The TTT mixer is the original $\mathrm { { V i T ^ { 3 } } }$ dual-branch construction.<sup>9</sup> A joint projection produces distinct query (Q), key (K), and value (V) views for a simplified SwiGLU branch and a depthwise convolution branch. Four gated heads each have dimension $d = D / 4 = 1 6$ , while the convolution branch has 16 channels. Their projected output dimensions are 64 and 16, respectively, and a final linear projection maps the 80 concatenated features to 64 dimensions.

For each gated head, denote the learnable inner initialization by $\boldsymbol { \Theta } _ { 0 } = \{ \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } \}$ . The inner network is

$$
f _ { \Theta } ( \mathbf { z } ) = ( \mathbf { z } \mathbf { W } _ { 1 } ) \odot \mathrm { S i L U } ( \mathbf { z } \mathbf { W } _ { 2 } ) .\tag{4}
$$

Here, SiLU denotes the sigmoid linear unit. For N spatial tokens, a compact expression for the negative dot-product inner objective is

$$
\mathcal { L } _ { \mathrm { i n n e r } } ( \Theta ; { \bf K } , { \bf V } ) = - \frac { s } { N } \sum _ { i = 1 } ^ { N } \langle f _ { \Theta } ( { \bf K } _ { i } ) , { \bf V } _ { i } \rangle , \qquad s = 1 / 3 .\tag{5}
$$

The convolution inner model uses the corresponding spatial dot-product objective with a $3 \times 3$ depthwise kernel. In the implementation, gradients are computed analytically without explicitly materializing the scalar objective, and each parameter gradient is rescaled by its norm along the implementation-defined axes:

$$
\widetilde { \mathbf { G } } = \frac { \mathbf { G } } { 1 + \Vert \mathbf { G } \Vert } , \quad \Theta _ { p } ^ { * } = \Theta _ { 0 } - \eta \widetilde { \mathbf { G } } _ { p } , \quad \eta = 1 .\tag{6}
$$

Equation (6) represents the one-step update for the current patch p. The gated and convolutional gradients are computed separately and normalized over their respective tensor axes. This operation

(a) Spectral-aware stem  
![](images/9d2ee80c1622a4102b4aa505f1216c1634ebd93cb2428a4379703b7ad9498c17.jpg)

![](images/99c20d399646b6da049d5885bf8cb9129c549d62deec81d0ffb2e9383efacbd5.jpg)  
Fresh learned initialization for every patch and forward pass.  
ğ = g/(1 +|g|) Batch axis omitted. Inner updates use K and V without class labels.

Fig 2 Implementation details of the spectral-aware stem and $\mathrm { V i T ^ { 3 } }$ -based TTT mixer. (a) Spectral convolution, pooling, channel folding, and spatial depthwise convolution produce 64-dimensional spatial tokens. (b) Four 16-dimensional simplified swish-gated linear unit (SwiGLU) heads and a separate 16-channel depthwise convolution branch each perform a one-step, unlabeled key–value inner update and then evaluate query features using the patch-specific fast weights. Both branch designs and their analytical update mechanism follow $\mathrm { V i T ^ { 3 } . }$ <sup>9</sup> Fast weights do not persist between patches.

is repeated independently for each inner branch and each input patch. The resulting temporary weights produce query outputs, $\mathbf { O } _ { p } = f _ { \boldsymbol { \Theta } _ { p } ^ { \ast } } ( \mathbf { Q } _ { p } )$ ; the outer-network initializations remain shared across patches and are not overwritten at inference. Crucially, this inner update uses no centerpixel labels or test-set annotations.

## 3.4 Center–global head and outer optimization

After two encoder blocks, a final layer normalization produces $\mathbf { Z } \in \mathbb { R } ^ { B \times N \times D }$ . Let $i _ { c } = \lfloor P / 2 \rfloor P +$ $\lfloor P / 2 \rfloor$ be the center-token index under zero-based indexing. We define

$$
\mathbf { f } _ { c } = \mathbf { Z } _ { : , i _ { c } , : } ,
$$

$$
\mathbf { f } _ { g } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { Z } _ { : , i , : } ,\tag{7}
$$

$$
\widehat { \mathbf { y } } = \mathrm { L i n e a r } ( \mathrm { D r o p o u t } ( \mathrm { L N } ( [ \mathbf { f } _ { c } ; \mathbf { f } _ { g } ] ) ) ) .\tag{8}
$$

The global average includes the center token. The classifier receives 128 features and produces K class logits. Outer-network parameters, including the initial inner weights, are optimized end-to end using standard supervised cross-entropy on training center-pixel labels.

## 4 Experimental setup

## 4.1 Datasets and sampling protocol

Four widely used HSI benchmarks are considered: Indian Pines (16 classes), Pavia University (9 classes), Salinas (16 classes), and Houston 2013 (15 classes). For each class, a seed-controlled random partition of labeled center pixels is used. Indian Pines and Houston 2013 each allocate 5% of labeled pixels to training and another $5 \%$ to validation; the rest is reserved for testing. Pavia University and Salinas each allocate 1% to training and 1% to validation. Dataset-specific rounding and a minimum of one sample per class are implemented where applicable. The ten seeds are 3405–3414. These experiments use random pixel splits rather than the official Houston 2013 fixed training/test division. Training-pixel spectra alone determine the per-band normalization statistics.

The input patch sizes are 29 for Indian Pines, Pavia University, and Houston 2013, and 27 for Salinas. All sampling protocols are applied to labeled center pixels, with disjoint training, validation, and test center-pixel sets. A separate spatial-block experiment further examines classification with spatially separated samples.

## 4.2 Implementation and checkpoint selection

The model has embedding dimension 64, encoder depth 2, four gated TTT heads and one convolution TTT branch, a feed-forward expansion factor of 2, 16 pooled spectral bins, and dropout 0.1. Training uses PyTorch 2.2.0, a single NVIDIA RTX 3090, AdamW with initial learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay 0.01, cosine annealing, batch size 32, and gradient-norm clipping at 1.0. A maximum of 300 epochs is allowed, with early-stopping patience 50.

The primary checkpoint criterion is the highest validation overall accuracy (OA). When multiple epochs attain the same maximum number of correct validation predictions, the model with the lower validation cross-entropy is retained. Early-stopping patience resets only when validation

OA strictly increases. Test labels are not used for checkpoint selection or hyperparameter updates. Reported results are the mean and sample standard deviation across ten seeds, unless stated otherwise.

## 4.3 Evaluation criteria and baselines

We report overall accuracy (OA), average accuracy (AA, the mean per-class recall), Cohen’s kappa, macro-F1, weighted-F1, and per-class recall. The comparison methods are SFMamba,<sup>14</sup> Hyper-TransXNet,<sup>12</sup> S<sup>2</sup>Mamba,<sup>7</sup> DSFormer,<sup>11</sup> and SSFTT.<sup>6</sup> All methods follow the same class-stratified random-sampling protocol, using seeds 3405–3414 and the training and validation fractions specified in Sec. 4. Each competing method uses the best-performing hyperparameter configuration specified in its original paper or official open-source implementation. Results are summarized over ten runs as the mean and standard deviation.

## 5 Results and discussion

## 5.1 Overall and class-wise classification performance

Tables 1–4 present class-wise recall, overall accuracy (OA), average accuracy (AA), Cohen’s kappa, macro-F1, and weighted-F1 for the five comparison methods and ViTTT-HSI. Each entry reports the mean and standard deviation over ten random seeds. The mean OA values of ViTTT-HSI are 97.6390% on Indian Pines, 98.8753% on Pavia University, 99.3292% on Salinas, and 98.4283% on Houston 2013.

On Indian Pines, ViTTT-HSI exceeds the strongest comparison method, DSFormer, by 1.8470 percentage points in mean OA. The corresponding improvements over the highest competing mean OA are 0.3315 percentage points on Pavia University, 0.3636 on Salinas, and 0.2728 on Houston 2013. The class-wise results further characterize performance across land-cover categories. For example, on Indian Pines, ViTTT-HSI achieves recalls of 85.38% and 72.78% for classes 7 and 9, respectively, while other methods show lower mean recall on these challenging classes. On Pavia University, the method achieves 99.23% recall for class 7 and 99.56% for class 9. Salinas and Houston 2013 also show strong recall across most classes. The detailed per-class comparison allows the OA and AA differences to be interpreted alongside class-specific performance.

Table 1 Class-wise recall and overall performance on Indian Pines (%, ten-seed mean ± standard deviation). All metrics, including kappa, are expressed as percentages. Class IDs follow the reference labels; bold marks the highest mean in each row.
<table><tr><td>Metric</td><td colspan="2">SFMamba HyperTransXNet</td><td> $\mathbf { S } ^ { \mathrm { 2 } } \mathbf { M } \mathbf { a m b a }$ </td><td>DSFormer</td><td>SSFTT</td><td>ViTTT-HSI</td></tr><tr><td>Class 01</td><td> $7 1 . 1 9 \pm 2 5 . 2 1$ </td><td> $6 7 . 8 6 \pm 2 0 . 0 5$ </td><td> $0 . 4 8 \pm 1 . 0 0$ </td><td> $7 9 . 0 5 \pm 1 8 . 8 7$ </td><td> $5 9 . 0 5 \pm 2 8 . 1 4$ </td><td> ${ \bf 9 6 . 4 3 \pm 2 . 3 1 }$ </td></tr><tr><td>Class 02</td><td>91.73 ± 3.88</td><td> $9 3 . 6 8 \pm 2 . 9 1$ </td><td> $9 3 . 1 7 \pm 2 . 5 7$ </td><td> $9 2 . 6 5 \pm 1 . 9 1$ </td><td> $9 1 . 7 6 \pm 2 . 5 0$ </td><td> ${ \bf 9 6 . 0 1 } \pm 2 . 2 1$ </td></tr><tr><td>Class 03</td><td> $9 2 . 4 4 \pm 3 . 4 6$ </td><td> $9 3 . 1 4 \pm 2 . 9 7$ </td><td> $9 3 . 1 4 \pm 3 . 5 5$ </td><td> ${ \bf 9 6 . 8 8 \pm 2 . 3 3 }$ </td><td> $9 4 . 5 0 \pm 2 . 2 3 $ </td><td> $9 6 . 7 4 \pm 1 . 4 8$ </td></tr><tr><td>Class 04</td><td> $9 3 . 0 0 \pm 5 . 4 7$ </td><td> $9 5 . 1 6 \pm 4 . 0 1$ </td><td> $9 1 . 7 4 \pm 6 . 7 2$ </td><td> $9 3 . 8 0 \pm 4 . 0 7$ </td><td> $9 3 . 9 4 \pm 7 . 0 6$ </td><td> ${ \bf 9 8 . 2 2 \pm 1 . 7 1 }$ </td></tr><tr><td>Class 05</td><td> $9 2 . 4 8 \pm 4 . 7 8$ </td><td> $9 3 . 5 2 \pm 3 . 0 4$ </td><td> $9 1 . 6 8 \pm 3 . 5 2$ </td><td> $9 3 . 8 2 \pm 2 . 5 8$ </td><td> $9 3 . 3 6 \pm 3 . 0 1$ </td><td> ${ \bf 9 3 . 9 3 \pm 3 . 3 2 }$ </td></tr><tr><td>Class 06</td><td> $9 8 . 3 4 \pm 1 . 1 5$ </td><td> $9 7 . 0 7 \pm 2 . 2 2$ </td><td> $9 9 . 5 9 \pm 0 . 2 8$ </td><td> $9 8 . 8 6 \pm 1 . 0 5$ </td><td> $9 8 . 6 3 \pm 0 . 9 6$ </td><td> ${ \bf 9 9 . 6 0 \pm 0 . 2 8 }$ </td></tr><tr><td>Class 07</td><td> $4 8 . 8 5 \pm 3 5 . 4 4$ </td><td> $3 0 . 0 0 \pm 2 5 . 3 7$ </td><td></td><td>0.77 ± 1.62 73.08 ± 18.13</td><td> $3 2 . 6 9 \pm 3 1 . 4 7$ </td><td> ${ \bf 8 5 . 3 8 \pm 1 5 . 8 9 }$ </td></tr><tr><td>Class 08</td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td><td> $9 9 . 9 3 \pm 0 . 1 6$ </td><td>100.00 ± 0.00</td><td> $9 9 . 8 1 \pm 0 . 5 9$ </td><td> $9 8 . 1 4 \pm 1 . 6 5$ </td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td></tr><tr><td>Class 09</td><td> $4 2 . 7 8 \pm 1 8 . 7 1$ </td><td> $4 9 . 4 4 \pm 2 3 . 4 9$ </td><td></td><td>0.00 ± 0.00 54.44 ± 19.39</td><td> $3 1 . 6 7 \pm 2 4 . 8 5$ </td><td> $7 2 . 7 8 \pm 2 5 . 3 2$ </td></tr><tr><td>Class 10</td><td> $8 9 . 6 7 \pm 3 . 5 1 $ </td><td> $8 9 . 7 9 \pm 2 . 9 9$ </td><td> $9 1 . 6 7 \pm 3 . 6 2$ </td><td> $9 0 . 5 0 \pm 3 . 4 2$ </td><td> $9 1 . 5 1 \pm 2 . 7 2$ </td><td> ${ \bf 9 4 . 7 7 \pm 2 . 7 0 }$ </td></tr><tr><td>Class 11</td><td> $9 6 . 2 3 \pm 1 . 7 3$ </td><td> $9 6 . 8 9 \pm 1 . 9 0$ </td><td> $9 7 . 2 2 \pm 1 . 5 9$ </td><td> $9 7 . 4 9 \pm 1 . 4 8$ </td><td> $9 5 . 9 6 \pm 2 . 2 8$ </td><td> ${ \bf 9 8 . 8 9 \pm 1 . 1 2 }$ </td></tr><tr><td>Class 12</td><td> $8 8 . 1 4 \pm 3 . 4 4$ </td><td> $9 0 . 0 0 \pm 4 . 0 1$ </td><td> $9 5 . 9 8 \pm 2 . 6 0$ </td><td> $9 3 . 8 1 \pm 2 . 1 3$ </td><td> $8 5 . 4 0 \pm 7 . 0 2$ </td><td> ${ \bf 9 6 . 0 4 \pm 1 . 8 1 }$ </td></tr><tr><td>Class 13</td><td> $9 7 . 8 9 \pm 2 . 5 5$ </td><td> $9 5 . 3 5 \pm 6 . 0 1$ </td><td> $9 9 . 4 6 \pm 0 . 9 2 $ </td><td> $9 8 . 9 7 \pm 0 . 8 6$ </td><td> $9 9 . 0 8 \pm 1 . 5 1 $ </td><td> ${ \bf 9 9 . 7 8 \pm 0 . 3 8 }$ </td></tr><tr><td>Class 14</td><td> $9 8 . 0 1 \pm 1 . 4 1$ </td><td> $9 6 . 9 8 \pm 2 . 1 4$ </td><td> $9 8 . 1 3 \pm 1 . 2 5$ </td><td> $9 8 . 6 7 \pm 0 . 8 4$ </td><td> $9 7 . 8 8 \pm 1 . 7 4$ </td><td> ${ \bf 9 9 . 9 5 \pm 0 . 0 9 }$ </td></tr><tr><td>Class 15</td><td> $9 4 . 1 7 \pm 3 . 3 5$ </td><td> $9 0 . 2 3 \pm 4 . 4 0$ </td><td> $8 9 . 0 8 \pm 6 . 3 9$ </td><td> $9 8 . 5 1 \pm 1 . 3 8$ </td><td> $9 3 . 9 9 \pm 4 . 4 7$ </td><td> ${ \bf 9 9 . 2 0 \pm 2 . 0 6 }$ </td></tr><tr><td>Class 16</td><td> ${ \pm } \mathrm { \ } 6 . 3 0 \pm 6 . 3 6$ </td><td> $8 3 . 7 3 \pm 9 . 8 2$ </td><td> $9 2 . 1 7 \pm 7 . 3 2$ </td><td> $9 5 . 0 6 \pm 3 . 9 1$ </td><td> $8 9 . 2 8 \pm 1 0 . 9 3$ </td><td> $9 5 . 1 8 \pm 5 . 9 0$ </td></tr><tr><td>OA</td><td> $9 4 . 1 0 \pm 1 . 0 6$ </td><td> $9 4 . 2 2 \pm 0 . 8 1$ </td><td> $9 4 . 5 4 \pm 0 . 6 5$ </td><td> $9 5 . 7 9 \pm 0 . 6 4$ </td><td> $9 4 . 0 4 \pm 0 . 9 0$ </td><td> ${ \bf 9 7 . 6 4 \pm 0 . 5 8 }$ </td></tr><tr><td>AA</td><td> $8 6 . 8 9 \pm 3 . 0 1$ </td><td> $8 5 . 1 7 \pm 2 . 1 1$ </td><td> $7 7 . 1 4 \pm 0 . 9 8$ </td><td> $9 0 . 9 6 \pm 1 . 3 6$ </td><td> $8 4 . 1 8 \pm 3 . 6 8$ </td><td> ${ \bf 9 5 . 1 8 \pm 1 . 8 3 }$ </td></tr><tr><td>Kappa</td><td> $9 3 . 2 7 \pm 1 . 2 1$ </td><td> $9 3 . 4 1 \pm 0 . 9 2$ </td><td> $9 3 . 7 7 \pm 0 . 7 4$ </td><td> $9 5 . 2 0 \pm 0 . 7 3$ </td><td> $9 3 . 2 0 \pm 1 . 0 2$ </td><td> ${ \bf 9 7 . 3 1 \pm 0 . 6 6 }$ </td></tr><tr><td>Macro-F1</td><td> $8 7 . 9 0 \pm 2 . 9 2$ </td><td> $8 6 . 8 5 \pm 2 . 6 2$ </td><td> $7 7 . 0 7 \pm 0 . 8 1$ </td><td> $9 2 . 4 7 \pm 1 . 4 8$ </td><td> $8 4 . 7 4 \pm 3 . 6 4$ </td><td> ${ \bf 9 5 . 4 7 \pm 1 . 6 9 }$ </td></tr><tr><td>Weighted-F1</td><td> $9 4 . 0 1 \pm 1 . 0 8$ </td><td> $9 4 . 1 3 \pm 0 . 8 0$ </td><td> $9 4 . 0 9 \pm 0 . 6 6$ </td><td> $9 5 . 7 5 \pm 0 . 6 4$ </td><td> $9 3 . 9 1 \pm 0 . 9 1 $ </td><td> ${ \bf 9 7 . 6 3 \pm 0 . 5 9 }$ </td></tr></table>

Table 2 Class-wise recall and overall performance on Pavia University (%, ten-seed mean ± standard deviation). All metrics, including kappa, are expressed as percentages. Class IDs follow the reference labels; bold marks the highest mean in each row.
<table><tr><td>Metric</td><td>SFMamba HyperTransXNet</td><td></td><td> $\mathbf { S } ^ { \mathrm { 2 } } \mathbf { M } \mathbf { a m b a }$ </td><td>DSFormer</td><td>SSFTT</td><td></td><td>ViTTT-HSI</td></tr><tr><td>Class 01</td><td> $9 7 . 0 5 \pm 1 . 4 4$ </td><td></td><td></td><td> $9 7 . 4 1 \pm 1 . 5 3 ~ 9 8 . 6 7 \pm 0 . 9 4 ~ 9 7 . 7 9 \pm 1 . 2 2 ~ 9 7 . 0 9 \pm 1 . 4 6 ~ 9 8 . 3 0 \pm 0 . 9 4$ </td><td></td><td></td><td></td></tr><tr><td>Class 02</td><td> $9 9 . 7 8 \pm 0 . 1 4$ </td><td></td><td></td><td> $9 9 . 5 0 \pm 0 . 4 3 ~ 9 9 . 8 2 \pm 0 . 1 6 ~ 9 9 . 8 2 \pm 0 . 1 1 ~ 9 9 . 7 9 \pm 0 . 2 0 ~ 9 9 . 7 3 \pm 0 . 2 9$ </td><td></td><td></td><td></td></tr><tr><td>Class 03</td><td> $9 4 . 1 6 \pm 1 . 6 5$ </td><td></td><td> $9 1 . 4 4 \pm 5 . 7 0 ~ 9 4 . 2 1 \pm 3 . 4 6 ~ 9 4 . 5 3 \pm 2 . 7 6 ~ 9 0 . 1 1 \pm 4 . 1 0 ~ 9 5 . 1 1 \pm 5 . 7 5$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Class 04</td><td> $9 5 . 5 9 \pm 1 . 5 2$ </td><td></td><td></td><td> $9 2 . 3 2 \pm 3 . 1 0 \ 9 6 . 8 6 \pm 1 . 0 7 \ 9 6 . 5 6 \pm 1 . 2 1 \ 9 4 . 9 0 \pm 2 . 7 8$ </td><td></td><td></td><td> ${ \bf 9 8 . 0 6 \pm 0 . 9 5 }$ </td></tr><tr><td>Class 05</td><td> $9 9 . 7 9 \pm 0 . 1 8$ </td><td></td><td></td><td> $9 9 . 2 4 \pm 0 . 8 0 \ 9 9 . 9 8 \pm 0 . 0 5 \ 9 9 . 9 9 \pm 0 . 0 2 \ 9 9 . 8 5 \pm 0 . 3 1 \ 9 9 . 9 6 \pm 0 . 1 0$ </td><td></td><td></td><td></td></tr><tr><td>Class 06</td><td> ${ \bf 9 9 . 3 1 \pm 0 . 6 9 }$ </td><td></td><td></td><td></td><td> $9 8 . 3 1 \pm 0 . 9 2 \ 9 8 . 5 2 \pm 1 . 1 4 \ 9 9 . 0 1 \pm 0 . 9 9 \ 9 8 . 7 5 \pm 1 . 7 2 \ 9 8 . 8 6 \pm 2 . 3 7$ </td><td></td><td></td></tr><tr><td>Class 07</td><td> $9 4 . 0 6 \pm 5 . 3 9$ </td><td></td><td></td><td></td><td> $9 3 . 0 3 \pm 3 . 8 6 9 3 . 4 4 \pm 7 . 2 9 9 5 . 7 9 \pm 6 . 9 4 9 9 . 0 1 \pm 0 . 7 3 9 9 . 2 3 \pm 1 . 8 7$ </td><td></td><td></td></tr><tr><td>Class 08</td><td> $9 6 . 3 0 \pm 1 . 8 3$ </td><td></td><td></td><td></td><td> $9 6 . 5 0 \pm 2 . 3 5 ~ 9 8 . 4 8 \pm 1 . 6 4 ~ 9 5 . 3 5 \pm 2 . 5 8 ~ 8 9 . 8 8 \pm 4 . 8 6$ </td><td></td><td> $9 7 . 7 5 \pm 1 . 3 7$ </td></tr><tr><td>Class 09</td><td> $9 1 . 8 6 \pm 6 . 1 7$ </td><td></td><td></td><td></td><td> $9 2 . 6 9 \pm 3 . 8 9 9 3 . 0 8 \pm 5 . 3 2 9 5 . 4 6 \pm 5 . 9 1 9 2 . 1 7 \pm 6 . 5 4 9 9 . 5 6 \pm 0 . 6 6$ </td><td></td><td></td></tr><tr><td>OA</td><td> $9 8 . 0 7 \pm 0 . 3 8$ </td><td></td><td></td><td> $9 7 . 5 1 \pm 0 . 5 1 \ 9 8 . 5 4 \pm 0 . 3 9 \ 9 8 . 3 1 \pm 0 . 5 4 \ 9 7 . 3 8 \pm 0 . 5 3 \ 9 8 . 8 8 \pm { \bf 0 . 4 9 }$ </td><td></td><td></td><td></td></tr><tr><td>AA</td><td> $9 6 . 4 3 \pm 1 . 0 9$ </td><td></td><td></td><td> $9 5 . 6 1 \pm 0 . 9 6 9 7 . 0 1 \pm 1 . 2 0 9 7 . 1 4 \pm 1 . 3 8 9 5 . 7 3 \pm 1 . 0 0 9 8 . 5 1 \pm 0 . 7 2$ </td><td></td><td></td><td></td></tr><tr><td>Kappa</td><td> $9 7 . 4 5 \pm 0 . 5 0$ </td><td></td><td></td><td></td><td> $9 6 . 6 9 \pm 0 . 6 8 9 8 . 0 7 \pm 0 . 5 2 9 7 . 7 6 \pm 0 . 7 2 9 6 . 5 3 \pm 0 . 7 0 9 8 . 5 1 \pm 0 . 6 5$ </td><td></td><td></td></tr><tr><td>Macro-F1</td><td> $9 6 . 6 5 \pm 0 . 8 4$ </td><td></td><td></td><td></td><td> $9 5 . 9 2 \pm 0 . 9 1 9 7 . 5 3 \pm 0 . 9 8 9 7 . 3 5 \pm 1 . 1 6 9 5 . 9 4 \pm 1 . 0 1 9 8 . 4 7 \pm { \bf 0 . 7 6 }$ </td><td></td><td></td></tr><tr><td>Weighted-F1</td><td> $9 8 . 0 7 \pm 0 . 3 8$ </td><td></td><td></td><td></td><td> $9 7 . 4 9 \pm 0 . 5 2 \ 9 8 . 5 3 \pm 0 . 4 0 \ 9 8 . 3 1 \pm 0 . 5 5 \ 9 7 . 3 6 \pm 0 . 5 4 \ 9 8 . 8 7 \pm 0 . 5 0$ </td><td></td><td></td></tr></table>

Table 3 Class-wise recall and overall performance on Salinas (%, ten-seed mean ± standard deviation). All metrics, including kappa, are expressed as percentages. Class IDs follow the reference labels; bold marks the highest mean in each row.
<table><tr><td>Metric</td><td>SFMamba HyperTransXNet</td><td></td><td> $\mathbf { S } ^ { \mathrm { 2 } } \mathbf { M } \mathbf { a m b a }$ </td><td>DSFormer</td><td></td><td>SSFTT</td><td>ViTTT-HSI</td></tr><tr><td>Class 01</td><td> $9 8 . 7 0 \pm 2 . 7 9$ </td><td> $9 9 . 3 6 \pm 0 . 8 2$ </td><td> ${ \bf 9 9 . 9 0 \pm 0 . 2 4 }$ </td><td></td><td> $9 7 . 6 4 \pm 5 . 0 7 \ 9 9 . 8 4 \pm 0 . 3 0 \ 9 9 . 8 5 \pm 0 . 4 1$ </td><td></td><td></td></tr><tr><td>Class 02</td><td> $9 9 . 9 9 \pm 0 . 0 3$ </td><td> $9 9 . 9 6 \pm 0 . 1 1 $ </td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td><td></td><td> ${ \bf 1 0 0 . 0 0 \pm 0 . 0 0 ~ 9 9 . 9 1 \pm 0 . 1 6 ~ 9 9 . 9 9 \pm 0 . 0 2 }$ </td><td></td><td></td></tr><tr><td>Class 03</td><td> $9 8 . 6 9 \pm 3 . 8 0$ </td><td> $9 9 . 1 6 \pm 1 . 1 5$ </td><td> ${ \bf 9 9 . 9 8 \pm 0 . 0 4 }$ </td><td></td><td>99.78 ± 0.53 99.98 ± 0.07 99.53 ± 0.50</td><td></td><td></td></tr><tr><td>Class 04</td><td> $9 8 . 9 5 \pm 2 . 1 3$ </td><td> $9 7 . 5 6 \pm 2 . 6 0$ </td><td> $9 8 . 6 6 \pm 1 . 2 4$ </td><td> $9 9 . 4 3 \pm 0 . 3 3 ~ 9 9 . 0 2 \pm 0 . 7 0 ~ 9 9 . 3 9 \pm 1 . 3 4$ </td><td></td><td></td><td></td></tr><tr><td>Class 05</td><td> $9 8 . 6 9 \pm 1 . 9 2$ </td><td> $9 8 . 5 5 \pm 1 . 0 2$ </td><td> $9 8 . 6 4 \pm 2 . 2 0$ </td><td> $9 8 . 9 1 \pm 1 . 6 0 ~ 9 9 . 0 2 \pm 1 . 2 3 ~ 9 8 . 0 1 \pm 0 . 8 3$ </td><td></td><td></td><td></td></tr><tr><td>Class 06</td><td> $9 9 . 8 9 \pm 0 . 2 0$ </td><td> $9 9 . 8 3 \pm 0 . 3 6$ </td><td> $9 9 . 8 3 \pm 0 . 5 3 $ </td><td>100.00 ± 0.00 99.89 ± 0.19 99.90 ± 0.23</td><td></td><td></td><td></td></tr><tr><td>Class 07</td><td> $9 9 . 9 2 \pm 0 . 0 9$ </td><td> $9 9 . 6 4 \pm 0 . 7 1 $ </td><td> $9 9 . 9 7 \pm 0 . 0 5$ </td><td> $9 9 . 9 9 \pm { \bf 0 . 0 1 9 9 . 7 8 } \pm 0 . 5 8 9 9 . 8 6 \pm 0 . 2 9$ </td><td></td><td></td><td></td></tr><tr><td>Class 08</td><td> $9 2 . 5 7 \pm 2 . 7 8$ </td><td> $9 5 . 9 1 \pm 2 . 5 5$ </td><td> $9 6 . 2 3 \pm 1 . 4 4$ </td><td> $9 6 . 8 8 \pm 0 . 7 2 ~ 9 8 . 3 6 \pm 0 . 8 3 ~ 9 9 . 1 2 \pm 0 . 6 4$ </td><td></td><td></td><td></td></tr><tr><td>Class 09</td><td> $9 9 . 9 8 \pm 0 . 0 3$ </td><td> $9 9 . 7 9 \pm 0 . 3 0$ </td><td>100.00 ± 0.00</td><td> ${ \bf 1 0 0 . 0 0 \pm 0 . 0 0 ~ 9 9 . 9 7 \pm 0 . 0 4 ~ 9 9 . 9 8 \pm 0 . 0 6 }$ </td><td></td><td></td><td></td></tr><tr><td>Class 10</td><td> $9 7 . 9 3 \pm 1 . 5 2$ </td><td> $9 9 . 0 8 \pm 0 . 7 7$ </td><td> $9 8 . 9 5 \pm 1 . 1 8$ </td><td></td><td>98.32 ± 1.14 98.76 ± 1.25 99.44 ± 0.87</td><td></td><td></td></tr><tr><td>Class 11</td><td> $9 9 . 0 1 \pm 0 . 7 4$ </td><td> $9 9 . 8 3 \pm 0 . 2 5$ </td><td> $9 9 . 5 2 \pm 0 . 0 9$ </td><td></td><td> $9 9 . 9 4 \pm 0 . 1 5 ~ 9 9 . 6 6 \pm 0 . 6 0 ~ 9 9 . 3 4 \pm 1 . 0 9$ </td><td></td><td></td></tr><tr><td>Class 12</td><td> $9 9 . 2 5 \pm 1 . 6 7$ </td><td> $9 9 . 3 8 \pm 1 . 1 1 $ </td><td> ${ \bf 9 9 . 9 7 \pm 0 . 0 6 }$ </td><td></td><td> $9 9 . 8 8 \pm 0 . 1 2 ~ 9 9 . 7 6 \pm 0 . 4 4 ~ 9 9 . 8 6 \pm 0 . 1 9$ </td><td></td><td></td></tr><tr><td>Class 13</td><td> $9 9 . 6 5 \pm 0 . 4 6$ </td><td> $9 9 . 4 3 \pm 1 . 1 7$ </td><td> $9 9 . 6 2 \pm 1 . 1 2$ </td><td></td><td>99.97 ± 0.08 96.34 ± 7.32 98.15 ± 2.77</td><td></td><td></td></tr><tr><td>Class 14</td><td> $9 8 . 9 1 \pm 0 . 9 7$ </td><td> $9 8 . 4 8 \pm 1 . 7 2$ </td><td> $9 8 . 9 5 \pm 1 . 0 2$ </td><td></td><td> $9 9 . 3 9 \pm { \bf 0 . 6 0 ~ 9 7 . 7 6 } \pm 2 . 8 0 ~ 9 7 . 6 7 \pm 2 . 0 8$ </td><td></td><td></td></tr><tr><td>Class 15</td><td> $9 2 . 1 9 \pm 2 . 1 9$ </td><td> $9 4 . 5 0 \pm 2 . 6 9$ </td><td> $9 4 . 0 6 \pm 3 . 3 3$ </td><td></td><td> $9 5 . 3 6 \pm 2 . 1 8 ~ 9 7 . 5 3 \pm 1 . 6 0 ~ 9 8 . 9 3 \pm 0 . 7 3$ </td><td></td><td></td></tr><tr><td>Class 16</td><td> $9 8 . 8 4 \pm 1 . 9 7$ </td><td> ${ \bf 9 9 . 5 8 \pm 0 . 7 5 }$ </td><td> $9 9 . 4 9 \pm 0 . 5 2 $ </td><td></td><td> $9 9 . 0 4 \pm 2 . 4 9 ~ 9 8 . 4 8 \pm 0 . 8 8 ~ 9 8 . 2 9 \pm 3 . 5 5$ </td><td></td><td></td></tr><tr><td>OA</td><td> $9 6 . 9 6 \pm 0 . 4 0$ </td><td> $9 8 . 0 2 \pm 0 . 6 0$ </td><td> $9 8 . 1 8 \pm 0 . 6 9$ </td><td></td><td> $9 8 . 4 1 \pm 0 . 4 2 ~ 9 8 . 9 7 \pm 0 . 2 8 ~ 9 9 . 3 3 \pm 0 . 2 7 ~$ </td><td></td><td></td></tr><tr><td>AA</td><td> $9 8 . 3 2 \pm 0 . 4 2$ </td><td> $9 8 . 7 5 \pm 0 . 3 4$ </td><td> $9 8 . 9 9 \pm 0 . 4 1$ </td><td></td><td>99.03 ± 0.41 99.00 ± 0.57 99.21 ± 0.44</td><td></td><td></td></tr><tr><td>Kappa</td><td> $9 6 . 6 2 \pm 0 . 4 4$ </td><td> $9 7 . 8 0 \pm 0 . 6 7$ </td><td> $9 7 . 9 7 \pm 0 . 7 7$ </td><td></td><td> $9 8 . 2 3 \pm 0 . 4 6 ~ 9 8 . 8 5 \pm 0 . 3 2$ </td><td></td><td> ${ \bf 9 9 . 2 5 \pm 0 . 3 0 }$ </td></tr><tr><td>Macro-F1</td><td> $9 8 . 2 3 \pm 0 . 4 6$ </td><td> $9 8 . 6 7 \pm 0 . 3 9$ </td><td> $9 8 . 9 8 \pm 0 . 4 0$ </td><td></td><td> $9 8 . 9 7 \pm 0 . 4 0 \ 9 9 . 0 0 \pm 0 . 5 1$ </td><td></td><td> ${ \bf 9 9 . 1 9 \pm 0 . 4 5 }$ </td></tr><tr><td>Weighted-F1</td><td> $9 6 . 9 6 \pm 0 . 3 9$ </td><td> $9 8 . 0 2 \pm 0 . 6 0$ </td><td> $9 8 . 1 8 \pm 0 . 6 9$ </td><td></td><td> $9 8 . 4 1 \pm 0 . 4 2 \ 9 8 . 9 7 \pm 0 . 2 8$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> ${ \bf 9 9 . 3 3 \pm 0 . 2 7 }$ </td></tr></table>

Table 4 Class-wise recall and overall performance on Houston 2013 (%, ten-seed mean ± standard deviation). All metrics, including kappa, are expressed as percentages. Class IDs follow the reference labels; bold marks the highest mean in each row.
<table><tr><td>Metric</td><td>SFMamba HyperTransXNet</td><td></td><td>S²Mamba</td><td>DSFormer</td><td>SSFTT</td><td></td><td>ViTTT-HSI</td></tr><tr><td>Class 01</td><td> $9 7 . 9 6 \pm 1 . 0 8$ </td><td> $9 7 . 6 3 \pm 1 . 6 0$ </td><td> $9 7 . 2 4 \pm 1 . 0 2$ </td><td> $9 8 . 1 8 \pm 1 . 2 9$ </td><td></td><td> $9 8 . 2 8 \pm 1 . 6 3$ </td><td> ${ \bf 9 8 . 4 1 \pm 1 . 5 0 }$ </td></tr><tr><td>Class 02</td><td> $9 8 . 9 7 \pm 0 . 9 7$ </td><td> $9 8 . 3 2 \pm 1 . 5 5$ </td><td> $9 8 . 7 1 \pm 0 . 5 3$ </td><td> $9 9 . 2 3 \pm 0 . 4 7$ </td><td> ${ \bf 9 9 . 4 7 \pm 0 . 7 6 }$ </td><td></td><td> $9 9 . 2 0 \pm 0 . 4 9$ </td></tr><tr><td>Class 03</td><td> $9 9 . 2 5 \pm 0 . 4 3$ </td><td> $9 9 . 3 6 \pm 0 . 4 4$ </td><td> $9 9 . 6 5 \pm 0 . 3 0$ </td><td> $9 8 . 8 2 \pm 1 . 2 7$ </td><td> $9 9 . 5 7 \pm 0 . 2 9$ </td><td></td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td></tr><tr><td>Class 04</td><td> $9 8 . 8 4 \pm 1 . 0 8$ </td><td> $9 7 . 0 7 \pm 1 . 9 7$ </td><td> $9 8 . 4 4 \pm 0 . 6 5$ </td><td> ${ \bf 9 9 . 2 3 \pm 0 . 5 5 }$ </td><td> $9 8 . 6 4 \pm 0 . 5 0$ </td><td></td><td> $9 9 . 0 9 \pm 1 . 7 7$ </td></tr><tr><td>Class 05</td><td> $9 9 . 9 6 \pm 0 . 0 9$ </td><td> $9 9 . 9 6 \pm 0 . 1 0$ </td><td></td><td>99.99 ± 0.03 100.00 ± 0.00 100.00 ± 0.00</td><td></td><td></td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td></tr><tr><td>Class 06</td><td> $9 0 . 3 4 \pm 5 . 6 7$ </td><td> $9 3 . 1 7 \pm 4 . 6 6$ </td><td> $8 3 . 5 5 \pm 4 . 0 0$ </td><td>93.89 ± 5.43</td><td> $9 4 . 5 7 \pm 5 . 9 5$ </td><td></td><td> ${ \pm } 7 . 3 4 \pm 2 . 4 7$ </td></tr><tr><td>Class 07</td><td> $9 8 . 1 1 \pm 1 . 0 9$ </td><td>95.95 ± 2.03</td><td> ${ \bf 9 8 . 9 8 \pm 0 . 7 1 }$ </td><td> $9 8 . 4 8 \pm 1 . 2 9$ </td><td> $9 7 . 4 3 \pm 2 . 7 9$ </td><td></td><td> $9 7 . 9 4 \pm 0 . 8 4$ </td></tr><tr><td>Class 08</td><td> $9 3 . 3 6 \pm 2 . 2 8$ </td><td>92.83 ± 2.06</td><td> $8 4 . 3 8 \pm 4 . 5 0$ </td><td> $9 4 . 5 0 \pm 1 . 6 4$ </td><td> $9 3 . 9 2 \pm 3 . 8 6$ </td><td></td><td>94.64 ± 3.18</td></tr><tr><td>Class 09</td><td> $9 5 . 3 6 \pm 2 . 2 3$ </td><td>94.80 ± 2.36</td><td>91.27 ± 2.54</td><td> $9 5 . 3 6 \pm 1 . 9 2$ </td><td> $9 6 . 7 5 \pm 1 . 3 6$ </td><td></td><td>97.31 ± 2.08</td></tr><tr><td>Class 10</td><td> $9 9 . 2 3 \pm 0 . 7 5$ </td><td>99.10 ± 1.55</td><td> $9 6 . 6 2 \pm 1 . 4 4$ </td><td> $9 9 . 2 5 \pm 0 . 5 4$ </td><td> $9 9 . 3 2 \pm 0 . 7 5$ </td><td></td><td> ${ \bf 9 9 . 8 4 \pm 0 . 2 3 }$ </td></tr><tr><td>Class 11</td><td> $9 8 . 6 1 \pm 1 . 1 6$ </td><td>97.76 ± 1.76</td><td> ${ \bf 9 8 . 9 1 \pm 1 . 2 4 }$ </td><td>98.88 ± 1.21</td><td> $9 8 . 2 8 \pm 1 . 7 1 $ </td><td></td><td> $9 8 . 7 4 \pm 1 . 3 0$ </td></tr><tr><td>Class 12</td><td> $9 7 . 0 8 \pm 1 . 2 4$ </td><td>95.98 ± 1.16</td><td>93.10 ± 3.55</td><td>97.12 ± 1.16</td><td> ${ \bf 9 8 . 7 0 \pm 0 . 6 5 }$ </td><td></td><td> $9 7 . 9 7 \pm 2 . 1 2$ </td></tr><tr><td>Class 13</td><td> $9 5 . 8 9 \pm 3 . 4 4$ </td><td></td><td> $9 2 . 5 3 \pm 3 . 1 2 \ 4 4 . 8 0 \pm 3 2 . 1 7$ </td><td></td><td>97.40 ± 2.53 96.43 ± 3.32</td><td></td><td> $9 6 . 3 6 \pm 2 . 2 2$ </td></tr><tr><td>Class 14</td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td><td></td><td> $9 9 . 3 0 \pm 1 . 0 4 \ 1 0 0 . 0 0 \pm 0 . 0 0$ </td><td></td><td>99.61 ± 0.48 100.00 ± 0.00</td><td></td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm 0 . 0 0 }$ </td></tr><tr><td>Class 15</td><td> $9 9 . 9 7 \pm 0 . 1 1 $ </td><td></td><td>99.85 ± 0.23 100.00 ± 0.00</td><td>99.95 ± 0.11</td><td> $9 9 . 9 8 \pm 0 . 0 5$ </td><td></td><td> $9 9 . 9 7 \pm 0 . 1 1 $ </td></tr><tr><td>OA</td><td> $9 7 . 7 6 \pm 0 . 2 7$ </td><td> $9 7 . 0 2 \pm 0 . 6 2$ </td><td> $9 4 . 4 0 \pm 1 . 2 8$ </td><td> $9 8 . 0 8 \pm 0 . 2 6$ </td><td> $9 8 . 1 6 \pm 0 . 5 9$ </td><td></td><td> ${ \bf 9 8 . 4 3 \pm 0 . 6 0 }$ </td></tr><tr><td>AA</td><td> $9 7 . 5 3 \pm 0 . 4 1$ </td><td> $9 6 . 9 1 \pm 0 . 6 6$ </td><td> $9 2 . 3 8 \pm 2 . 4 2$ </td><td> $9 7 . 9 9 \pm 0 . 5 0 $ </td><td> $9 8 . 0 9 \pm 0 . 7 4$ </td><td></td><td> ${ \bf 9 8 . 4 5 \pm 0 . 5 8 }$ </td></tr><tr><td>Kappa</td><td> $9 7 . 5 8 \pm 0 . 2 9$ </td><td> $9 6 . 7 8 \pm 0 . 6 7$ </td><td> $9 3 . 9 4 \pm 1 . 3 9$ </td><td> $9 7 . 9 2 \pm 0 . 2 9$ </td><td> $9 8 . 0 1 \pm 0 . 6 4$ </td><td></td><td> ${ \bf 9 8 . 3 0 \pm 0 . 6 4 }$ </td></tr><tr><td>Macro-F1</td><td> $9 7 . 7 1 \pm 0 . 2 7$ </td><td> $9 6 . 9 6 \pm 0 . 6 3$ </td><td> $9 2 . 6 8 \pm 2 . 8 8$ </td><td> $9 8 . 1 1 \pm 0 . 3 5$ </td><td> $9 8 . 1 4 \pm 0 . 6 7$ </td><td></td><td> ${ \bf 9 8 . 4 4 \pm 0 . 6 0 }$ </td></tr><tr><td>Weighted-F1</td><td> $9 7 . 7 5 \pm 0 . 2 7$ </td><td> $9 7 . 0 2 \pm 0 . 6 2$ </td><td> $9 3 . 9 5 \pm 1 . 7 4$ </td><td> $9 8 . 0 8 \pm 0 . 2 7$ </td><td> $9 8 . 1 5 \pm 0 . 5 9$ </td><td></td><td> ${ \bf 9 8 . 4 3 \pm 0 . 6 0 }$ </td></tr></table>

![](images/e7ed1a5a997df0fe4a0641f012a28a60e4c68ea5b10c1b32c95ed4d433ef9f1b.jpg)

## 5.2 Classification maps and error visualization

Figures 3–6 compare six classifiers using consistent colors within each dataset. Predictions are paired with binary error maps (white: incorrect; black: other pixels). ViTTT-HSI uses the validation-selected checkpoint from seed 3405 (Sec. 4.2).

(a)  
GT  
![](images/19e16a163e7e28e2a27cbd33beb05f2d45498dc6a5254d0561063701e9010503.jpg)  
(b)  
Error maps White: errors Black: other  
SFMamba

![](images/6132cc000b0008234cfc954a2a3f5635b8a07f7a5f7ef7b6d0113f28fcf51f66.jpg)

(c)  
HyperTransXNet  
![](images/6db731d35fc75a639637f3cb3ab4cfdea7be93d714fc211fad103f3ce6ec2263.jpg)

(d)  
S²Mamba  
![](images/8a08dbf13163df75135fe1c56e22ab454a461ec1a5b3234dd799fc1d4b518214.jpg)

![](images/d794a571d7eca56edf774a0462bf9e3e269d370c4ed4a6776d0642bdc3222662.jpg)

(e)  
DSFormer  
![](images/ad290d5f5922dd0376029f55c7a6953a2e929e7a707087bfc277a587d5214e62.jpg)

![](images/a8e9414fc186de6af71994ff6c8a488bcd6eeed9f1063ec5c68758823b7dde0e.jpg)  
(f)  
ViTTT-HSI

(c)  
(g)  
(b)  
![](images/9f8ea0205db658c5638c3e9681386303e5b88437f77964ed05bd813131f0bc6d.jpg)  
Fig 3 Classification maps on Indian Pines: (a) ground truth; (b) SFMamba; (c) HyperTransXNet; (d) $\mathbf { S } ^ { 2 } \mathbf { M }$ amba; (e) DSFormer; (f) SSFTT; (g) ViTTT-HSI. Each model’s prediction appears above its error map.  
(e)  
(f)  
SSFTT  
(g)

(a)  
(d)  
SFMamba  
GT  
HyperTransXNet  
S²Mamba  
![](images/20e8e21ac916fbd2e86d42ff7d4978438076f575a135d2aba20c405a65d0b25d.jpg)

ViTTT-HSI  
DSFormer  
![](images/077e82cab7265900bb1fa406b3dd4c45101957a84ee4576ee57bd4afc2d49959.jpg)

![](images/f8b141860984d3d8a4b5eba519d971a484d07dd68a6c1fcfe42b03272cecc11d.jpg)

![](images/83f2b30af260e02d2c4c0fd51b4733423a41761773b4e0a603c2c88ba28c093e.jpg)

![](images/a72afc0ceb8a860a121fea8938bb824f9c1ef89185006595f1079030899102f0.jpg)

![](images/8c67b98681a16cbd08e814d0d53fa0e6bb95ec169f871e871855452e09d4cd96.jpg)  
Fig 4 Classification maps on Pavia University. Panel order and the error-map convention follow Fig. 3.

(a)  
GT  
(b)  
SFMamba  
HyperTransXNet  
(c)  
(d)  
S²Mamba  
(e)  
DSFormer  
(f)  
SSFTT  
(g)  
ViTTT-HSI  
![](images/dd071aa8e3cef2919cb032299e540cb7641ad96c587e2220b41e59f646d06527.jpg)  
Fig 5 Classification maps on Salinas. Panel order and the error-map convention follow Fig. 3.

(a)  
GT  
(b)  
SFMamba  
(c)  
HyperTransXNet  
(d)  
S²Mamba  
(e)  
DSFormer  
(f)  
SSFTT  
(g)  
ViTTT-HSI  
![](images/08d3b5c6829c73ce84f0441827152fa82b52f859673c0236f99ce9dedfc5a4d9.jpg)  
Fig 6 Classification maps on Houston 2013. Panel order and the error-map convention follow Fig. 3. All maps are rotated $9 0 ^ { \circ }$ counterclockwise for readability.

## 5.3 Training dynamics and validation convergence

The loss and accuracy histories provide a complementary view of optimization under the specified training and checkpoint-selection protocol. Figures 7–9 plot the epoch-wise mean training loss, validation cross-entropy, and validation OA for all four datasets. The curves show rapid early improvements followed by a more gradual refinement of the validation metrics. The two types of validation measurement serve different purposes: OA determines the highest accuracy used in checkpoint selection, while cross-entropy resolves ties between epochs attaining that level. The plotted means use the runs that have reached the respective epoch, because individual runs stop at different epochs with an early-stopping patience of 50 epochs.

![](images/e3f7e6747e0c5b133d1a2d35cde36bae41f7d075f0dd8e8f54f6645f4150448a.jpg)  
(c) Salinas

![](images/76a193c7f4ed1d4461eb8f3503c349c168dd1c4abaf8222e08f440de65f7391c.jpg)  
(d) Houston 2013

![](images/8c15502f92b5f9b62517bad89c2e58b9022c64423c83e81f1045323ac4101c6d.jpg)

![](images/1f7df12597b354bca06175edae2694d6a0686e55e3690ae3c969c0cacdb65ce0.jpg)  
Fig 7 Training cross-entropy across seeds 3405–3414: (a) Indian Pines; (b) Pavia University; (c) Salinas; (d) Houston 2013. Lines and bands show the mean and ± one sample standard deviation of the available runs at each epoch; stopped runs are not extended.

(c) Salinas  
![](images/5b2efcdd665fa66bacd9de9928992bb985f7f75d13a12b2fe1f5444e53fca32d.jpg)

![](images/e5e1724ff123d7123b22f20985b33e0896a55b0780d736fddf1a8fd2e010a638.jpg)

![](images/f5058a6fa79770e24b386142bd1960bdb8658681a172f11b1645adb2c69be9e0.jpg)

![](images/b0eb7440897fc857a78809efbaeab8c0ec65da7901d4529cca6abdb2dc046b0f.jpg)

Fig 8 Validation cross-entropy across seeds 3405–3414. The line/band convention and panels (a)–(d) follow Fig. 7; stopped runs are not extended.  
![](images/88d9c0bec6b352f053fc473cf84a886c824001a3390ab448e5a05588cd12a3f2.jpg)  
(c) Salinas

![](images/d837fe534448d3006583869b75b45cdc45e711ebefedafb79d75303851c31472.jpg)

![](images/f3a55a4585e347634126c873436b82202815a95a6cc910eb02efc777f8e9689a.jpg)

(d) Houston 2013  
![](images/68359181b094c74edeaedf55047c3044961f9e918c5ea4bb70e6bc61094c2312.jpg)  
Fig 9 Validation overall accuracy across seeds 3405–3414. The line/band convention and panels (a)–(d) follow Fig. 7; stopped runs are not extended.

## 5.4 Effect of spectral embedding and TTT

Table 5 provides a factorial ablation of spectral versus linear embedding and the presence of the TTT mixer on Indian Pines and Pavia University. The TTT mixer improves Indian Pines classification with either embedding, and the spectral stem produces the highest OA and AA on Indian Pines. Pavia University also benefits from TTT, while linear embedding with TTT gives a slightly higher mean OA than spectral embedding with TTT. The two datasets therefore exhibit different sensitivities to the embedding pathway.

## 5.5 Role of the TTT inner update

Table 6 compares the full mixer against a model without the mixer and a frozen-inner version that retains the inner-model architecture but skips fast-weight updates. Full TTT improves Indian Pines OA by 3.5555 percentage points over the frozen-inner version, with a smaller gain of 0.2478 percentage points on Pavia University. These controlled comparisons isolate the benefit of patchspecific fast-weight updates more clearly than removing the mixer altogether: the learned innermodel architecture is retained in the frozen variant, whereas only its data-conditioned update is omitted.

## 5.6 Center–globalfeatures and patch-size analysis

The center–global classifier is assessed against center-only and global-only heads in Table 7. The center feature is especially important: global-only classification is markedly weaker on both datasets. Combining center and global features improves Indian Pines OA over center-only classification, while the center-only head is slightly better on Pavia University. The result is consistent with the different contributions of center-pixel evidence and neighborhood context across scenes.

Table 8 evaluates patch sizes 11, 21, and 29 under both full TTT and a no-TTT setting, using three seeds per configuration. The full TTT configuration obtains higher mean OA than the no-TTT setting at each evaluated patch size. The observed gain is largest at $P = 2 9$ on Indian Pines and P = 21 on Pavia University, indicating that the interaction between patch context and inner adaptation depends on the dataset.

Table 5 Embedding and TTT ablation on Indian Pines and Pavia University (%, ten seeds).
<table><tr><td>Dataset</td><td>Variant</td><td>OA</td><td>AA</td></tr><tr><td>Indian</td><td>Linear, no TTT</td><td> $9 3 . 7 6 \pm 0 . 8 8$ </td><td> $8 9 . 4 5 \pm 3 . 1 3$ </td></tr><tr><td>Indian</td><td>Spectral, no TTT</td><td> $9 4 . 1 9 \pm 0 . 9 5$ </td><td> $9 0 . 9 1 \pm 2 . 1 6$ </td></tr><tr><td>Indian</td><td>Linear, TTT</td><td> $9 7 . 4 2 \pm 0 . 4 6$ </td><td> $9 4 . 8 5 \pm 1 . 8 9$ </td></tr><tr><td>Indian</td><td>Spectral, TTT</td><td> $9 7 . 6 4 \pm 0 . 5 8$ </td><td> $9 5 . 1 8 \pm 1 . 8 3$ </td></tr><tr><td>Pavia</td><td>Linear, no TTT</td><td> $9 8 . 4 2 \pm 0 . 3 8$ </td><td> $9 7 . 7 2 \pm 0 . 6 4$ </td></tr><tr><td>Pavia</td><td>Spectral, no TTT</td><td> $9 8 . 4 6 \pm 0 . 3 5$ </td><td> $9 7 . 7 8 \pm 0 . 6 9$ </td></tr><tr><td>Pavia</td><td>Linear, TTT</td><td> $9 8 . 9 7 \pm 0 . 3 5$ </td><td> $9 8 . 6 7 \pm 0 . 5 6$ </td></tr><tr><td>Pavia</td><td>Spectral, TTT</td><td> $9 8 . 8 8 \pm 0 . 4 9$ </td><td> $9 8 . 5 1 \pm 0 . 7 2$ </td></tr></table>

Table 6 Ablation of the TTT inner update (%, mean ± standard deviation; ten seeds).
<table><tr><td>Dataset</td><td>Mixer setting</td><td>OA</td><td>AA</td></tr><tr><td>Indian</td><td>No TTT</td><td> $9 4 . 1 9 \pm 0 . 9 5$ </td><td> $9 0 . 9 1 \pm 2 . 1 6$ </td></tr><tr><td>Indian</td><td>Frozen inner</td><td> $9 4 . 0 8 \pm 0 . 6 5$ </td><td> $9 0 . 1 8 \pm 2 . 0 6$ </td></tr><tr><td>Indian</td><td>Full</td><td> $9 7 . 6 4 \pm 0 . 5 8$ </td><td> $9 5 . 1 8 \pm 1 . 8 3$ </td></tr><tr><td>Pavia</td><td>No TTT</td><td> $9 8 . 4 6 \pm 0 . 3 5$ </td><td> $9 7 . 7 8 \pm 0 . 6 9$ </td></tr><tr><td>Pavia</td><td>Frozen inner</td><td> $9 8 . 6 3 \pm 0 . 3 1$ </td><td> $9 8 . 1 1 \pm 0 . 5 2 $ </td></tr><tr><td>Pavia</td><td>Full</td><td> $9 8 . 8 8 \pm 0 . 4 9$ </td><td> $9 8 . 5 1 \pm 0 . 7 2$ </td></tr></table>

Table 7 Classification-head ablation (%, ten-seed OA mean ± standard deviation).
<table><tr><td>Dataset</td><td>Head</td><td>OA</td></tr><tr><td>Indian</td><td>Center</td><td> $9 7 . 4 3 \pm 0 . 4 0$ </td></tr><tr><td>Indian</td><td>Global</td><td> $9 5 . 0 4 \pm 0 . 6 9$ </td></tr><tr><td>Indian</td><td>Center + Global</td><td> $9 7 . 6 4 \pm 0 . 5 8$ </td></tr><tr><td>Pavia</td><td>Center</td><td> $9 8 . 9 0 \pm 0 . 4 1$ </td></tr><tr><td>Pavia</td><td>Global</td><td> $9 4 . 7 1 \pm 0 . 7 1$ </td></tr><tr><td>Pavia</td><td>Center + Global</td><td> $9 8 . 8 8 \pm 0 . 4 9$ </td></tr></table>

## 5.7 Spatially separated evaluation

A further evaluation on Pavia University uses a spatial-block split with patch size 11 and three seeds. As summarized in Table 9, full TTT outperforms the no-TTT variant in this more spatially separated setup. The full TTT configuration achieves 90.16% OA, exceeding the no-TTT configuration by 1.40 percentage points in this spatial-block experiment.

## 5.8 Computational complexity and efficiency

Table 10 presents the parameter count, computational cost, and training and inference times of ViTTT-HSI on an NVIDIA RTX 3090. Computational cost in billions of floating-point operations (GFLOPs) is estimated for one input patch with a batch size of one, counting the main forward computations of both TTT inner adaptation and query evaluation at two floating-point operations per multiply–accumulate. Training time covers the epoch loop, including validation and checkpoint writing. Testing time is measured on the entire test loader after warm-up, with device synchronization.

Across all four datasets, the model uses approximately 0.09 million trainable parameters. The measured test-set runtime varies with the number of pixels evaluated and the dataset-specific patch size.

Table 8 Patch-size ablation on Indian Pines and Pavia University (%, mean OA over three seeds). The last column is the Full TTT minus No TTT difference in percentage points.
<table><tr><td>Dataset</td><td>P</td><td>Full TTT</td><td>No TTT</td><td>Difference</td></tr><tr><td>Indian</td><td>11</td><td>97.34</td><td>95.30</td><td>+2.04</td></tr><tr><td>Indian</td><td>21</td><td>97.24</td><td>95.17</td><td>+2.07</td></tr><tr><td>Indian</td><td>29</td><td>97.69</td><td>94.80</td><td>+2.89</td></tr><tr><td>Pavia</td><td>11</td><td>97.46</td><td>97.36</td><td>+0.11</td></tr><tr><td>Pavia</td><td>21</td><td>98.74</td><td>98.21</td><td>+0.53</td></tr><tr><td>Pavia</td><td>29</td><td>98.72</td><td>98.50</td><td>+0.22</td></tr></table>

Table 9 Spatial-block evaluation on Pavia University using patch size 11 (%, mean ± standard deviation across three seeds).
<table><tr><td>Variant</td><td>OA AA</td></tr><tr><td>Full</td><td> $9 0 . 1 6 \pm 1 . 0 4$   $8 7 . 3 9 \pm 2 . 6 5$ </td></tr><tr><td>No TTT</td><td> $8 8 . 7 6 \pm 0 . 8 5$   $8 5 . 5 6 \pm 6 . 9 2$ </td></tr></table>

Table 10 Model complexity and seed-3405 runtime of ViTTT-HSI. GFLOPs denote the estimated major forward arithmetic operations per input patch; M denotes millions of parameters, and runtime is reported in seconds.
<table><tr><td>Dataset</td><td>Parameters (M)</td><td>GFLOPs</td><td>Train (s)</td><td>Test (s)</td></tr><tr><td>Indian</td><td>0.0923</td><td>0.167</td><td>139.1</td><td>4.04</td></tr><tr><td>Pavia</td><td>0.0914</td><td>0.165</td><td>42.2</td><td>11.15</td></tr><tr><td>Salinas</td><td>0.0923</td><td>0.145</td><td>93.3</td><td>20.44</td></tr><tr><td>Houston 2013</td><td>0.0922</td><td>0.166</td><td>219.5</td><td>4.33</td></tr></table>

## 6 Conclusion

This study adapted the ViT<sup>3</sup> visual test-time training mechanism to center-pixel hyperspectral classification through a spectral-aware embedding, spatial-token-preserving TTT encoders, and a center–global classifier. Four public benchmarks and ten random seeds supported competitive classification performance under the specified low-label random-pixel protocols. Controlled ablations indicate that the patch-conditioned TTT inner update contributes to the classification gains, especially on Indian Pines, while the benefits of the spectral stem and global feature pooling are dependent on the dataset. The results establish an experimental basis for applying patch-conditioned visual TTT to spectral–spatial center-pixel classification.

## Disclosures

The authors declare that there are no financial interests, commercial affiliations, or other potential conflicts of interest that could have influenced the objectivity of this research or the writing of this paper.

## Code and Data Availability

The Indian Pines, Pavia University, Salinas, and Houston 2013 datasets used in this study are available through the HSI Datasets repository on Hugging Face.<sup>16</sup> The source code and reproduction scripts for ViTTT-HSI will be released in the project GitHub repository.

## Acknowledgments

We thank OpenAI ChatGPT for assistance with English-language editing and LaTeX preparation. The authors take full responsibility for the scientific content, accuracy, and conclusions of this manuscript.

## References

1 M. E. Paoletti et al., “Deep learning classifiers for hyperspectral imaging: A review,” ISPRS Journal ofPhotogrammetry and Remote Sensing 158, 279–317 (2019). doi:10.1016/j.isprsjprs.2019.09.006.

2 S. Jia et al., “A survey: Deep learning for hyperspectral image classification with few labeled samples,” Neurocomputing 448, 179–204 (2021). doi:10.1016/j.neucom.2021.03.035.

3 Z. Zhong et al., “Spectral–spatial residual network for hyperspectral image classification: A 3-D deep learning framework,” IEEE Transactions on Geoscience and Remote Sensing 56(2), 847–858 (2018). doi:10.1109/TGRS.2017.2755542.

4 S. K. Roy et al., “HybridSN: Exploring 3-D–2-D CNN feature hierarchy for hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters 17(2), 277–281 (2020). doi:10.1109/LGRS.2019.2918719.

5 D. Hong et al., “SpectralFormer: Rethinking hyperspectral image classification with Transformers,” IEEE Transactions on Geoscience and Remote Sensing 60, 1–15 (2022). doi:10.1109/TGRS.2021.3130716.

6 L. Sun et al., “Spectral–spatial feature tokenization transformer for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing 60, 1–14 (2022). doi:10.1109/TGRS.2022.3144158.

7 G. Wang et al., “S<sup>2</sup>Mamba: A spatial-spectral state space model for hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing 63, 1–13 (2025). doi:10.1109/TGRS.2025.3530993.

8 Y. Sun et al., “Learning to (learn at test time): RNNs with expressive hidden states,” in Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research 267, 57503–57522 (2025). https://proceedings.mlr.press/v267/sun25h.html.

9 D. Han et al., “ViT<sup>3</sup>: Unlocking test-time training in vision,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 51–61 (2026). https://openaccess.thecvf.com/content/CVPR2026/html/ Han ViT3 Unlocking Test-Time Training in Vision CVPR 2026 paper.html.

10 A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, 30, 5998–6008 (2017). https://papers.nips.cc/paper/7181-attention-is-all-you-need.

11 Y. Xu et al., “Dual selective fusion transformer network for hyperspectral image classification,” Neural Networks 187, 107311 (2025). doi:10.1016/j.neunet.2025.107311.

12 X. Dai et al., “HyperTransXNet: Learning both global and local dynamics with a dual dynamic token mixer for hyperspectral image classification,” Remote Sensing 17(14), 2361 (2025). doi:10.3390/rs17142361.

13 A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in Proceedings ofthe Conference on Language Modeling (COLM), (2024). https://openreview.net/forum?id=tEYskw1VY2.

14 Z. Meng, L. Yue, and F. Zhao, “Spatial-frequency joint learning Mamba for hyperspectral image classification,” IEEE Geoscience and Remote Sensing Letters 23, 1–5 (2026). doi:10.1109/LGRS.2026.3652369.

15 X. Yue et al., “Beyond distribution shifts: Adaptive hyperspectral image classification at test time,” Neurocomputing 678, 133188 (2026). doi:10.1016/j.neucom.2026.133188.

16 Tanishq165, “HSI Datasets.” Hugging Face dataset repository. Accessed 30 October 2026. https://huggingface.co/datasets/Tanishq165/HSI Datasets.

Xiang Zhao received his B.Sc. degree in mathematics and applied mathematics from Harbin Engineering University, Harbin, China, in 2025. He is currently pursuing an M.Sc. degree in mathematics at the College of Mathematical Sciences, Harbin Engineering University. His research interests include remote sensing image processing, hyperspectral image classification, and deep learning algorithms, particularly graph neural networks and attention mechanisms.