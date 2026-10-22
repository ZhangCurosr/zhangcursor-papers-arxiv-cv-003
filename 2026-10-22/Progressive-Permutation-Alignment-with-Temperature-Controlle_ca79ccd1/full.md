# Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening

Yuta Horioka<sup>⋆</sup> , Naoya Chiba<sup>⋆</sup> , and Yuki Uranishi

The University of Osaka

Abstract. In implicit neural representations (INRs), 3D signals are encoded directly in network weights, making stable weight-space operations essential. However, permutation symmetry causes independently trained yet functionally equivalent models to appear distant in weight space, often inducing large loss barriers under linear interpolation. Existing neuron alignment methods either commit early to discrete matches and become trapped in inconsistent local optima or rely on computationally expensive gradient optimization over continuous relaxations, both of which are unreliable in challenging cases such as NeRF. We adopt progressive alignment, a layer-wise strategy that follows inter-layer dependencies from input to output. At each layer, a correlation matrix is estimated and aligned with a discrete anchor obtained by the Hungarian algorithm, after which temperature-controlled spectral annealing gradually sharpens the correlation structure toward a permutation matrix. This update is deterministic and lightweight, avoiding the inner-loop gradient optimization required by continuous relaxation methods. A final refinement stage further optimizes the network layer-by-layer via weight matching, starting from the aligned solution. Experiments on MLPs, CNNs, and Neural Radiance Fields (NeRFs) show that the proposed method reliably recovers neuron correspondences and achieves stable alignment across architectures. In practical experiments aligning independently trained NeRF models, our method reduces the loss barriers under weight interpolation compared with existing approaches.

## 1 Introduction

Implicit Neural Representations (INRs) encode continuous signals directly in network weights. NeRF represents 3D scenes as neural weights, allowing models to function as data representations [2,3,6,27]. Along with advances such as Model Soups and task arithmetic [15, 35], direct manipulation in weight space, such as interpolation and merging, has become an important research topic [13, 24, 30].

However, neural networks exhibit permutation symmetry. In Multi-Layer Perceptrons (MLPs), the units in each hidden layer can be arbitrarily permuted without changing the network’s function, provided the weights of adjacent layers are correspondingly transformed. Therefore, numerous sets of weights exist that represent the same function, and simple weight averaging or linear interpolation often leads to performance degradation, resulting in a high loss barrier along the interpolation path [8].

![](images/a9fbb0232246cc5a0ea46416584b7e43af1c5f51a66005cfe60879cef440fa82.jpg)  
Fig. 1: Overview of Progressive Alignment with Temperature-Controlled Spectral Sharpening. Two independently trained neural fields, $\mathcal { N } _ { A }$ and $\mathcal { N } _ { B }$ exhibit permutation symmetry across the hidden layers. Our method aligns them in a progressive, layer-wise manner following the inter-layer dependency from input to output. At each layer, a correlation matrix is computed and gradually sharpened via temperature-controlled spectral projection.

Prior work has shown the existence of Linear Mode Connectivity (LMC), which maps independently trained models into the same loss basin via appropriate permutations, enabling interpolation without loss barriers [8]. LMC has since been extended to more complex architectures [33], multi-model settings [16], model merging [23, 36], and weight-space generative modeling [13].

Previous studies have attempted alignment in the weight space by estimating the unit correspondences layer by layer. Git Re-Basin proposed weight matching, which solves a linear assignment problem for each layer, and showed that linear interpolation is enabled by moving models into the same basin [1]. In contrast, OT4P, focusing on the fact that permutation matrices are a subset of orthogonal matrices, proposed a gradient-based framework by parameterizing orthogonal matrices using Lie algebras [12].

While efective, existing approaches face limitations for challenging targets, such as NeRF. Weight matching commits to discrete permutations early, which may propagate errors across layers. OT4P performs continuous global optimization but requires costly iterative gradient updates and can struggle in complex settings such as NeRF.

The permutation of each layer depends on its neighboring layers, revealing a clear inter-layer dependency structure. Several prior works adopted sequential layer-wise alignment strategies to stabilize optimization [23, 32, 34, 36], suggesting that alignment should proceed from input to output.

We propose progressive alignment, which advances alignment sequentially from input to output while respecting inter-layer dependencies, combined with temperature-controlled spectral annealing. At each layer, a correlation matrix is aligned to a Hungarian anchor and progressively sharpened toward a permutation matrix through spectral annealing. This bridges continuous relaxation and discrete matching without iterative optimization.

We demonstrate that the proposed method enables reliable alignment even for Neural Radiance Fields (NeRFs) [22], where weights encode 3D scenes in practice. This facilitates smoother interpolation and improves the usability of NeRFs as weight-space representations [3, 6, 13, 24, 27]. Synthetic permutation recovery experiments can be solved by most existing methods. However, in practical scenarios, such as independently trained NeRF models, the correlation structure becomes highly ambiguous and noisy. In such cases, early discrete decisions made by weight matching often propagate errors across layers, whereas the proposed spectral annealing method gradually sharpens alignment while preserving global consistency.

The main contributions of this study are as follows:

We formulate Permutation Alignment as a sequential problem with interlayer dependencies and propose progressive alignment, which advances the alignment step-by-step from input to output.

– We introduce a temperature-controlled spectral projection that combines Hungarian anchors with spectral annealing, avoiding premature fixation to local optima.

– We introduce a two-stage strategy that performs sequential layer-wise weight matching using the proposed alignment as a robust initialization, enabling highly stable optimization.

– We apply the proposed method to NeRFs, a challenging alignment setting, and demonstrate reliable alignment and reduced loss barriers for independently trained models.

## 2 Related Work

The permutation symmetries of neural networks have been studied in model merging, mode connectivity, and weight-space learning. However, their role in INRs remains unclear. Here, we review the permutation alignment methods and INR weight representations.

## 2.1 Permutation Symmetries and Mode Connectivity of Neural Networks

Neural networks are invariant to the permutations of hidden units, resulting in exponentially many equivalent parameterizations [5]. This symmetry underlies the geometry of loss landscape and mode connectivity [7,9–11]. Entezari et al. [8] conjectured that appropriate permutation alignment places independently trained models within a single connected basin, enabling the LMC. This idea has since been extended to large-scale architectures and multi-model settings [16, 33], as well as training-time alignment strategies [18].

## 2.2 Permutation Alignment of Neural Networks

Representative studies that explicitly address permutation alignment include methods that estimate neuron correspondences using Optimal Transport [31] and Git Re-Basin [1]. In particular, Git Re-Basin proposed weight matching, which estimates optimal permutations by constructing a correlation matrix for each layer and solving a Linear Assignment Problem. Because permutations are fixed discretely using the Hungarian algorithm at each update, incorrect correspondences may propagate across layers, and the results can depend on the update order and initialization. Additionally, the layer update order is typically determined randomly, which can lead to variance in the results depending on the initial conditions and random seed.

To avoid the dificulties of discrete permutation optimization, several studies relaxed permutation matrices to continuous spaces. A common approach optimizes the Birkhof polytope, which is the convex hull of permutation matrices, using doubly stochastic relaxations and Sinkhorn regularization [4, 21]. In the context of a model re-basin, Sinkhorn operators combined with implicit diferentiation enable gradient-based end-to-end optimization [26]. Beyond convex relaxations, OT4P [12] extends the search space to the orthogonal group by parameterizing orthogonal matrices via Lie algebras and performs gradient-based optimization on the orthogonal manifold.

Orthogonal to permutation optimization itself, it has been observed that even after alignment, linear interpolation may sufer from Variance Collapse of activation. Methods such as REPAIR [17] address this issue by recalibrating the activation statistics after merging.

Progressive layer-wise alignment, which advances matching from input to output, has gained attention for federated learning and model merging. FedMA [34] se quentially matches and integrates neurons starting from the early layers. ZipIt! [32] performs step-by-step merging while propagating the transformation matrices forward through the network. Similarly, ProDistill [36] and PLeaS [23] adopted progressive layer-wise distillation and alignment to improve the scalability and performance of large models. These works collectively indicate that respecting directional inter-layer dependencies can stabilize alignment and merging.

Spectral decomposition has also been used to resolve permutation ambiguities, for example, in transformer head matching via permutation-invariant spectra [28]. The eigenvalue and singular value spectra can provide symmetry-aware signals for alignment.

## 2.3 Weight Representations of INRs

INRs, such as NeRF [22] and DeepSDF [25] encode signals directly in network weights, connecting to prior work that treats weights as data [14,29]. Methods such as inr2vec and nf2vec [6, 27], Graph Meta-Networks [3], and Spatial LLaNA [2] learn compact representations of INR weights for downstream tasks. However, these approaches are sensitive to permutation symmetries, and misalignment often leads to high loss barriers or representation collapse [24]. Symmetry-aware architectures, including graph-based processing [19], DEEP-ALIGN [24], and Universal Neural Functionals [37], attempt to address this issue at the architectural level. Nevertheless, the stable post-hoc alignment of independently trained INRs remains challenging.

To directly address symmetry, alternative approaches design permutationequivariant architectures. Graph-based processing of networks [19], inferencebased permutation prediction using Deep Weight Space Networks (DEEP-ALIGN) [24], and Universal Neural Functionals (UNFs) [37] aim to incorporate symmetry handling into model design itself. Nevertheless, aligning independently optimized INRs in a stable and scalable manner remains challenging.

## 3 Proposed Method

In this paper, we propose an alignment method for the weight space of neural networks. Our approach introduces a temperature parameter τ into the correspondence matrix to enable continuous relaxation and performs alignment progressively following the order of layers (Fig. 1).

While adopting a progressive layer-wise framework, the proposed method incorporates a temperature-controlled spectral projection from the unconstrained space of the correlation matrices to the permutation matrices. Specifically, we applied temperature scaling to the singular values of the correlation matrix. For numerical robustness, we solely rely on SVD for this spectral projection process.

This strategy enables the method to avoid the local optima that conventional approaches are prone to while exploring globally consistent alignments more eficiently than stochastic gradient-based optimization methods.

## 3.1 Problem Formulation

First, we formulate the alignment problem of the neural networks. Given two neural networks $\mathcal { N } _ { A }$ and $\mathcal { N } _ { B }$ , let $\bar { W } _ { l } ^ { ( A ) }$ and $W _ { l } ^ { ( B ) }$ denote the weight matrices at layers $l \in \{ 1 , \ldots , L \}$ , respectively. Our objective is to estimate permutation matrices P that reorder the hidden neurons of $\mathcal { N } _ { B }$ such that they become functionally aligned with $\mathcal { N } _ { A }$ , thereby minimizing the distance between the two models in the weight space.

The global objective function L is defined as the sum of the squared diferences between the aligned weights at each layer:

$$
\operatorname* { m i n } _ { \{ P _ { l } \} _ { l = 1 } ^ { L - 1 } } \mathcal { L } = \sum _ { l = 1 } ^ { L } \left\| W _ { l } ^ { ( A ) } - P _ { l } W _ { l } ^ { ( B ) } P _ { l - 1 } ^ { \top } \right\| _ { F } ^ { 2 } ,\tag{1}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. For simplicity of exposition, the bias terms are omitted, but the formulation can be straightforwardly extended to incorporate them.

Each $P _ { l }$ is a permutation matrix, and $P _ { 0 }$ and $P _ { L }$ , corresponding to the input and output layers, respectively, are fixed to the identity matrix I. Because the permutation at each layer is structurally coupled with that of the adjacent layers, this optimization problem becomes a discrete combinatorial problem with exponential complexity. In general, this is NP-hard [1, 12].

## 3.2 Progressive Alignment Based on Correlation Maximization

Optimizing all permutation matrices $\{ P l \} _ { l = 1 } ^ { L - 1 }$ jointly is computationally intractable owing to the combinatorial explosion inherent in discrete permutation searches. Although approaches such as OT4P [12] relax parameters to continuous values and apply gradient-based optimization, they may fail to produce satisfactory results in challenging alignment scenarios, particularly for models such as NeRF. To balance computational eficiency and optimization stability, we adopt a progressive optimization strategy that decomposes the problem into layer-wise subproblems and solves them iteratively.

When updating the permutation matrix $P _ { l }$ at layer l, fixing the adjacent permutation matrices $P _ { l - 1 }$ and $P _ { l + 1 }$ reduces the number of terms in the objective function (Eq. 1) that depend on $P _ { l }$ , yielding the following correlation maximization problem:

$$
\operatorname* { m a x } _ { P _ { l } } \mathrm { T r } ( P _ { l } ^ { \top } C _ { l } ) ,\tag{2}
$$

where the structure of the correlation matrix $C _ { l }$ reflects the neuron-wise alignment strength. Under fixed neighboring permutations, $C _ { l }$ is computed from the current weights as follows:

$$
\begin{array} { r } { C _ { l } = W _ { l } ^ { ( A ) } ( P _ { l - 1 } W _ { l } ^ { ( B ) } ) ^ { \top } + ( W _ { l + 1 } ^ { ( A ) } ) ^ { \top } P _ { l + 1 } W _ { l + 1 } ^ { ( B ) } . } \end{array}\tag{3}
$$

The derivation is provided in the Supplementary Material. Intuitively, $C _ { l }$ represents the local consistency at layer l based on its connections to the preceding and succeeding layers. Its spectral structure reflects the distribution of the alignment strength within layer l. Solving this optimization problem corresponds to reordering the correspondences at layer l to simultaneously maximize consistency with both adjacent layers.

In contrast, Git Re-Basin [1] solves this subproblem in a random layer order by addressing a linear assignment problem and immediately updates to a discrete permutation matrix,

$$
P _ { l } = \underset { P \in \mathcal { P } _ { n } } { \arg \operatorname* { m a x } } \mathrm { T r } ( P ^ { \top } C _ { l } ) .\tag{4}
$$

However, this constitutes a premature and irreversible decision. As a result, incorrect correspondences determined in the early stages may propagate to subsequent layers, potentially causing the entire network to converge to a poor local optimum. (A concrete numerical example on a small-scale case is provided in the Supplementary Material.)

![](images/152149d588dc789143e58c1444a630973085b24ae8290dcff927cf197194559a.jpg)  
Fig. 2: Temperature-Controlled Spectral Sharpening for Layer-wise Permutation $\mathrm { U p } \cdot$ date. At layer $l ,$ given weights $W _ { l } ^ { ( \mathbf { \bar { A } } ) }$ and $W _ { l } ^ { \left( B \right) ^ { \bullet } }$ , a correlation matrix $C _ { l }$ that captures the neuron-wise similarity under fixed neighboring permutations is computed. The Hungarian algorithm first produces an anchor permutation $\hat { P } _ { l }$ that provides a discrete reference alignment. Temperature-controlled spectral sharpening is applied to the aligned correlation matrix $\hat { P } _ { l } ^ { \top } C _ { l }$ , where spectral values are exponentiated by a temperature parameter τ.

## 3.3 Temperature-Controlled Spectral Annealing

Inspired by the OT4P approach [12], we address the aforementioned local optimum issue by proposing an update rule based on temperature-controlled spectral annealing. Starting with the correlation matrix $C _ { l }$ , the proposed method gradually drives the solution toward a permutation matrix. An overview of the update procedure is shown in Fig. 2, and the complete algorithm is presented in Alg. 1. The key feature of our method is that instead of relying on gradient-based optimization, it directly controls the spectral structure of the correlation matrix. By adjusting the temperature parameter, the solution is progressively sharpened toward a discrete permutation matrix in a deterministic manner.

Our method starts with the correlation matrix $C _ { l } \in \mathbb { R } ^ { n \times n }$ at each layer. First, we solve the linear assignment problem

$$
\hat { P } _ { l } = \underset { P \in \mathcal { P } _ { n } } { \arg \operatorname* { m a x } } \mathrm { T r } ( P ^ { \top } C _ { l } ) ,\tag{5}
$$

using the Hungarian algorithm to obtain an anchor permutation matrix $\hat { P } _ { l }$ that is most consistent with the correlation structure. Next, we align the correlation matrix with the anchor permutation,

$$
M _ { l } = \hat { P } _ { l } ^ { \top } C _ { l } ,\tag{6}
$$

and apply temperature-controlled spectral annealing to $M _ { l }$ . Let the singular value decomposition (SVD) of the matrix $M \in \mathbb { R } ^ { n \times n }$ be given by $M = U \pmb { \Sigma } V ^ { \top }$ where $\pmb { \Sigma } = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { n } )$ is a diagonal matrix of singular values.

In this study, we define a spectral operator that applies a temperature parameter $\tau \in [ 0 , 1 ]$ to the singular values as $\begin{array} { r } { M ^ { \langle \tau \rangle } = U \pmb { \Sigma } ^ { \tau } V ^ { \top } } \end{array}$ where $\begin{array} { r l } { \pmb { \Sigma } ^ { \tau } = } \end{array}$ $\mathrm { d i a g } ( \sigma _ { 1 } ^ { \tau } , \dots , \sigma _ { n } ^ { \tau } )$

This operation preserves the orthogonal bases while applying the spectral function $\sigma \mapsto \sigma ^ { \tau }$ to the singular values. As τ decreases, the singular spectrum is flattened, efectively projecting the unconstrained correlation matrix toward the orthogonal group. By strictly employing SVD, our approach ensures numerical stability and avoids the complex values that can arise when computing the eigendecomposition of asymmetric matrices.

To ensure a smooth transition from the continuous spectral relaxation to a discrete permutation, we introduce a linear interpolation mechanism. Using the obtained anchor $\hat { P } _ { l }$ and the original correlation matrix $C _ { l } .$ we update the permutation matrix by linearly interpolating between the spectrally annealed correlation matrix and the identity matrix according to the current temperature parameter $\tau _ { k } \in \mathcal { T }$ , as follows:

$$
P _ { l } ^ { ( \mathrm { n e w } ) } = \hat { P } _ { l } \left[ \tau _ { k } \left( \hat { P } _ { l } ^ { \top } C _ { l } \right) ^ { \langle \tau _ { k } \rangle } + ( 1 - \tau _ { k } ) I \right] .\tag{7}
$$

Here, $\mathcal { T } = \{ \tau _ { 1 } , \dots , \tau _ { K } \}$ denotes the temperature schedule that decreases from 1.0 to $0 . 0 ,$ , and I represents the identity matrix. This linear interpolation explicitly anchors the continuous relaxation. As $\tau _ { k }$ approaches $0 ,$ the matrix inside the bracket converges to the identity matrix I, forcing the overall update $P _ { l } ^ { ( \mathrm { n e w } ) }$ to smoothly and deterministically land on the discrete anchor permutation $\hat { P } _ { l }$ This mechanism prevents the exploration from remaining overly broad in the continuous unconstrained space.

By iteratively performing progressive alignment along the layer order while gradually decreasing $\tau ,$ the correlation matrix $C _ { l }$ becomes increasingly concentrated around the anchor permutation $\hat { P } _ { l }$ , tending toward a discrete correspondence. Because our method performs only a single update per temperature value, it does not require iterative gradient-based optimization as in OT4P, enabling fast and deterministic estimation of correspondences. During this annealing process, the continuous structure of the correlation matrix is preserved in the early stages, allowing the exploration of global consistency. As τ decreases, the high-confidence modes are progressively emphasized, ultimately leading to convergence to discrete neuron correspondences.

## 4 Experiment

In this section, we present two experimental results using traditional evaluation of neural network alignment and NeRF alignment to evaluate the alignment capability and efectiveness of the proposed method. All experiments were conducted on a computer with an NVIDIA GeForce RTX 5090 GPU.

Algorithm 1 Progressive Alignment with Temperature-Controlled Spectral Sharpening Sharpening

```latex
Require: Models $\mathcal { N } _ { A } , \mathcal { N } _ { B }$ , Temperature schedule T, Max iterations M
1: Initialize $P _ { l } = I$ for all layers l
2: Generate schedule $\mathcal { T } = \{ \hat { \tau } _ { 1 } , \dots , \tau _ { K } \}$ from 1.0 down to 0.0
3: $\mathbf { f o r } \ \tau \in \mathcal { T }$ do
<sup>4:</sup> <sub>5:</sub> for Layer iteration $m = 1$ to M do
for layer $l = 1$ to L − 1 do
6: Compute correlation matrix $C _ { l }$ using current $P _ { l - 1 } , P _ { l + }$ 1
7: Obtain discrete anchor: $\hat { P } _ { l } \gets$ Hungarian(C<sub>l</sub>)
8: Align correlation matrix: $M _ { l } \gets \hat { P } _ { l } ^ { \top } C _ { l }$
9: Compute SVD: U, Σ, $V ^ { \top }  \operatorname { S V D } ( M _ { l } )$
10: Apply spectral flattening: $M _ { l } ^ { \langle \tau \rangle } \gets U \pmb { \Sigma } ^ { \tau } V ^ { \top }$
11: Update with anchored continuous relaxation: $P _ { l } \gets \hat { P } _ { l } \left[ \tau M _ { l } ^ { \langle \tau \rangle } + ( 1 - \tau ) I \right]$
12: end for
13: end for
14: end for
15: return Aligned model $\mathcal { N } _ { B } ^ { \prime }$ permuted by {P<sub>l</sub>}
```

## 4.1 Comparison Methods

To evaluate the performance of the proposed method, we compared it with the following approaches.

1. Naive Interpolation: Linear interpolation is performed directly on the weights without any permutation alignment.

2. Weight Matching [1]: The neuron alignment problem is formulated as a linear assignment problem and solved using greedy weight matching.

3. Progressive Weight Matching: In contrast to the original weight matching [1], which selects layers in random order for optimization, this variant optimizes the permutations sequentially from the input layer toward the output layer.

4. OT4P [12]: This method relaxes the discreteness of permutation matrices and performs optimization in a continuous space using stochastic gradient descent (SGD).

5. Progressive OT4P [12]: A modified version of standard OT4P normally optimizes all layers jointly, where gradient-based optimization is performed sequentially from the input side of the network to the output.

6. Ours w/o weight matching refinement: An ablated version of our method that relies solely on correlation-based relaxation and temperature-controlled progressive projection, omitting the final refinement stage.

7. Ours: Our complete two-stage approach. It utilizes the continuously aligned solution obtained from the spectral projection as a robust initialization, followed by iterative layer-wise weight matching to achieve highly stable optimization.

For weight matching, random correspondence estimation was performed 10 times using shufled layer orders covering all layers. For progressive variants, 100 sequential sweeps were conducted. For the proposed method, the temperature schedule was set to $\mathcal { T } = \{ 1 . 0 , 0 . 5 , 0 . 0 \}$ and a single sweep was performed for each temperature value, followed by a final sequential layer-wise refinement. For OT4P-based optimization, we used AdamW [20] as the optimizer and terminated the optimization if no improvement was observed for 10 consecutive steps. The learning rate was set to 0.05 for OT4P.

Table 1: $L _ { 1 }$ distance (converted by log(1+x)) and Precision (%) of alignment algorithms across standard architectures (MLP, CNN)
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">MLP5</td><td colspan="3">VGG11</td><td colspan="3">ResNet18</td></tr><tr><td> $L _ { 1 }$ </td><td>Dist.</td><td>Precision</td><td></td><td> $L _ { 1 } \ \mathrm { D i s t . }$ </td><td>Precision</td><td> $L _ { 1 }$ </td><td>Dist.</td><td>Precision</td></tr><tr><td>Weight Matching</td><td></td><td></td><td></td><td></td><td></td><td>0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000</td><td></td><td></td><td></td></tr><tr><td>Progressive Weight Matching 0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OT4P (τ = 0.7)</td><td></td><td></td><td></td><td></td><td></td><td>0.000 ±0.000 100.0 ±0.000 6.223 ±0.27199.74 ±0.033 6.376 ±0.298 99.29 ±0.090</td><td></td><td></td><td></td></tr><tr><td>OT4P (τ = 0.5)</td><td></td><td></td><td></td><td></td><td></td><td>0.000 ±0.000 100.0 ±0.000 7.075 ±0.201 99.64 ±0.036 5.768 ±0.273 99.45 ±0.065</td><td></td><td></td><td></td></tr><tr><td>OT4P (τ = 0.3)</td><td></td><td></td><td></td><td></td><td></td><td>0.000 ±0.000 100.0 ±0.000 6.746 ±0.225 99.74 ±0.029 5.820 ±0.611 99.42 ±0.135</td><td></td><td></td><td></td></tr><tr><td>Progressive OT4P(τ = 0.7)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.448 ±0.096 66.27 ±4.331 8.690 ±0.284 99.15 ±0.160 11.58 ±0.006 60.37 ±1.552</td></tr><tr><td>Progressive OT4P(τ = 0.5)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.705 ±0.031 51.76 ±2.286 9.722 ±0.928 97.58 ±1.999 11.59 ±0.003 58.06 ±1.131</td></tr><tr><td>Progressive OT4P(τ = 0.3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8.897 ±0.006 34.2 ±0.608 12.28 ±0.001 74.88 ±0.045 11.61 ±0.005 54.53 ±0.489</td></tr><tr><td>Proposed (Ours)  $\mathrm { w / o }$  Ref.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000</td></tr><tr><td>Proposed (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000 0.000 ±0.000 100.0 ±0.000</td></tr></table>

## 4.2 Evaluation of Alignment Accuracy on MLPs and CNNs

Following prior work [12], we evaluated whether the proposed method can accurately recover the correct neuron correspondences. As evaluation targets, we used a simple 5-layer MLP (with randomly initialized weights) and standard VGG11 and ResNet18 architectures (with the oficial ImageNet-pretrained weights provided by torchvision). We construct Model B by randomly permuting the intermediate weights of a reference Model A (Ground Truth), preserving the network’s input-output function. Each method is then applied to align Model B back to Model A. To account for permutation variability, experiments were conducted five times using diferent random seeds, and we report the mean and standard deviation.

Evaluation Metrics To quantitatively evaluate the alignment performance, we followed prior work [12] and adopted the following metrics:

$L _ { 1 }$ Distance: The $L _ { 1 }$ distance between models in the weight space of neural networks. A smaller value indicates that the original weight configuration has been recovered more accurately.

– Precision: In the single-model shufle recovery experiment, the proportion of elements in the estimated permutation matrices that exactly match the Ground Truth. A higher value indicates a more accurate identification of the original correct permutations.

Table 1 shows the recovery accuracy and post-alignment L1 distance. On the small MLP5, OT4P recovers the permutations exactly at every temperature. On both CNNs it still recovers more than 99% of the units, but it does not reach exact recovery, and a residual misalignment remains in the L1 distance. Its cost also grows with the layer width: on VGG11, whose fully connected layers have 4096 units, each run takes about 40 minutes because of gradient optimization over large orthogonal manifolds. Progressive OT4P is considerably less reliable. It optimizes each layer only once while the subsequent layers are still unaligned, so errors made in earlier layers are never corrected, and its precision varies widely with the network and $\tau$ (34–66% on MLP5, 75–99% on VGG11, and 55–60% on ResNet18). Even at its best temperature, it does not achieve exact recovery on any of the three networks. This sensitivity makes it dificult to choose an appropriate temperature in advance.

In contrast, both weight matching and our method perfectly recovered the shufled weights (Precision 100%, $L _ { 1 }$ distance ≈ 0) regardless of network depth or layer dimensionality, completing the alignment eficiently without iterative gradi ent updates (detailed runtime comparisons are provided in the Supplementary Material).

## 4.3 Alignment and LMC Evaluation on NeRF

While previous studies on neural network alignment have primarily focused on MLP and CNN, this study extends permutation alignment to Neural Radiance Fields (NeRF), a 3D representation model, for the first time and evaluates its efectiveness in this setting.

Evaluation on Random Permutation Recovery in NeRF We first evaluated the ability to identify correct correspondences in NeRF using the same experimental protocol as in the previous section. We used the NeRF Synthetic dataset (eight scenes) [22] and performed optimization with the NeRF implementation provided by nerfstudio. A pretrained reference model (Model A) and a randomly permuted version (Model B) are prepared. As in the MLP and CNN alignment experiments, Model B was aligned back to Model A using each method, and the accuracy of permutation recovery was evaluated.

Table 2 summarizes the results of the single-model permutation recovery experiment using NeRF. OT4P was sensitive to the temperature: it recovered about 99% of the units at $\tau = 0 . 5$ and 0.7 but fell into poor local optima at $\tau = 0 . 3$ (Precision 14.55%), and Progressive OT4P failed at every temperature (Precision $\leq 6 . 3 \% )$ . In contrast, both the proposed method and weight matching successfully restored the original weight configuration with 100% precision $\left( L _ { 1 } { \mathrm { ~ d i s t a n c e } } \approx 0 \right)$ demonstrating reliable alignment capability, even for 3D representation models such as NeRF.

Alignment of NeRF Models Trained from Diferent Initializations As a realistic and challenging alignment task for NeRF, we evaluated whether the proposed method can mitigate the loss barrier between models independently trained from diferent random initializations. In this setting, ground-truth permutation is unavailable. Therefore, the evaluation was conducted by comparing rendered images from linearly interpolated models with the corresponding ground-truth images. We generated interpolated models in the parameter space as follows:

Table 2: $L _ { 1 }$ distance (converted by $\log ( 1 + x ) )$ and Precision (%) on NeRF
<table><tr><td>Algorithm</td><td> $L _ { 1 }$  Dist. Precision</td></tr><tr><td>Weight Matching</td><td>0.000 100.0</td></tr><tr><td>Progressive Weight Matching</td><td>0.000 100.0</td></tr><tr><td>OT4P  $( \tau = 0 . 3 )$ </td><td>12.76 14.55</td></tr><tr><td> $\mathrm { O T 4 P } \ ( \tau = 0 . 5 )$ </td><td>6.818 99.16</td></tr><tr><td> $\mathrm { O T 4 P }$   $( \tau = 0 . 7 )$ </td><td>6.720 99.21</td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 3 )$ </td><td>12.92 1.455</td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 5 )$ </td><td>12.91 3.657</td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 7 )$ </td><td>12.90 6.259</td></tr><tr><td>Proposed (Ours)  $\mathrm { w } / \mathrm { o }$  Ref.</td><td>0.000 100.0</td></tr><tr><td>Proposed (Ours)</td><td>0.000 100.0</td></tr></table>

Table 3: Quantitative comparison of interpolated aligned models at $\lambda = 0 . 5$
<table><tr><td colspan="3">Method RMSE↓PSNR ↑</td></tr><tr><td>Naive</td><td>0.491</td><td>6.205</td></tr><tr><td>Weight Matching</td><td>0.294</td><td>11.07</td></tr><tr><td>Progressive Weight Matching</td><td>0.306</td><td>10.64</td></tr><tr><td>OT4P  $( \tau = 0 . 3 )$ </td><td>0.489</td><td>6.239</td></tr><tr><td>OT4P  $( \tau = 0 . 5 )$ </td><td>0.495</td><td>6.118</td></tr><tr><td>OT4P  $( \tau = 0 . 7 )$ </td><td>0.494</td><td>6.160</td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 3 )$ </td><td>0.493</td><td>6.170</td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 5 )$ </td><td>0.493</td><td>6.167</td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 7 )$ </td><td>0.492</td><td>6.182</td></tr><tr><td>Proposed (Ours)  $\mathrm { w / o }$  Ref.</td><td>0.199</td><td>14.22</td></tr><tr><td>Proposed (Ours)</td><td>0.199</td><td>14.40</td></tr></table>

$$
W _ { \lambda } = ( 1 - \lambda ) W _ { A } + \lambda W _ { B ^ { \prime } } ,\tag{8}
$$

where $W _ { B ^ { \prime } }$ denotes the weights of Model B after alignment. The interpolation coeficient λ was varied from 0.0 to 1.0 in increments of 0.1, and the rendering quality along the interpolation path was evaluated. In this experiment, we considered a pair of random seeds. For each scene, 10 images were randomly sampled from the test views for evaluation. The reported results correspond to the mean and variance computed over 10 test views for each of the eight scenes. In this experiment, we used $\tau = 0 . 5$ for OT4P and Progressive OT4P. Evaluation Metrics:

– RMSE (Root Mean Square Error): The root mean square error between pixel values of the ground-truth image and the rendered image. A smaller value indicates a lower rendering error.

$$
\lambda = 0 . 2 5
$$

$$
\lambda = 0 . 5 0
$$

$$
\lambda = 0 . 7 5
$$

![](images/dac2072f6334873d7d05b1960d7d4ad35aeac28f84460600304560882a8a9417.jpg)  
Fig. 3: Example of interpolation results for diferent λ values. (a) Naive, (b) Weight Matching, (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed (w/o weight matching refinement), and (g) Proposed.

– PSNR (Peak Signal-to-Noise Ratio): A standard metric for evaluating rendering quality. A higher value indicates better image quality of the interpolated model.

Figure 4 shows the PSNR and RMSE along the interpolated weight rendering results. The computational time is presented in the Supplementary Material. Naive Interpolation exhibits a sharp quality drop around λ = 0.5, indicating that the two models lie in distinct basins. Existing alignment methods partially reduce this barrier, but still degrade the midpoint. In contrast, the proposed method substantially suppressed the loss barrier. Although slight blurring remains at λ = 0.5, the overall scene structure and object contours are well preserved, demonstrating superior interpolation quality and suggesting that the aligned models reside in a flatter shared basin. Both the base method and the variant with sequential refinement outperform existing approaches by a large margin, while their average accuracy is comparable (14.22 vs. 14.40 dB PSNR at $\lambda = 0 . 5 )$ . The improvement therefore stems mainly from the temperature-controlled spectral sharpening (Supplementary Material, Tab. 2); the refinement stage mainly reduces the sensitivity to the temperature schedule (Supplementary Material, Tab. 3). Detailed evaluation metrics for each scene (RMSE, PSNR, and SSIM) and runtime comparisons are provided in the Supplementary Material.

![](images/488f15eb540fd96584fd61f87a386f47f71a178568c8f002e32a48a307b509fc.jpg)  
(a) RMSE ↓

![](images/48d7c330dca25359cb50acc60d47a6f3c20162c0ccb3a4e3869adab76406224a.jpg)  
(b) PSNR ↑  
Fig. 4: Quantitative comparison across diferent metrics.

## 5 Conclusion

In this paper, we addressed permutation symmetry in Neural Fields and INRs by proposing progressive alignment with temperature-controlled spectral sharpening. Each layer-wise update is reduced to trace maximization over a correlation matrix. Starting from this continuous representation, we combined a Hungarian anchor with temperature-τ-controlled spectral annealing to converge progressively to a discrete permutation. This strategy mitigates premature hard decisions in weight matching while avoiding the costly iterative optimization required by gradient-based relaxations, such as OT4P, yielding a deterministic and eficient alignment procedure. A subsequent sequential layer-wise refinement stage further improves joint consistency across layers. Experiments on MLPs and CNNs have demonstrated the accurate recovery of known permutations. Extending permutation alignment to NeRF, we showed reliable permutation recovery and reduced loss barriers between independently trained models, enabling smoother interpolation in the weight space. Our evaluation of independently trained models is limited to MLP-based NeRFs on NeRF Synthetic; other INRs and real scenes remain future work.

## Acknowledgements

This work was supported by JST PRESTO Grant Number JPMJPR25T6.

## References

1. Ainsworth, S., Hayase, J., Srinivasa, S.: Git re-basin: Merging models modulo permutation symmetries. In: International Conference on Learning Representations (ICLR) (2023)

2. Amaduzzi, A., Ramirez, P.Z., Lisanti, G., Salti, S., Stefano, L.D.: Spatially-aware weights tokenization for neRF-language models. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2025)

3. Ballerini, F., Ramirez, P.Z., Stefano, L.D., Salti, S.: Weight space representation learning on diverse neRF architectures. In: International Conference on Learning Representations (ICLR) (2026)

4. Birdal, T., Simsekli, U.: Probabilistic permutation synchronization using the riemannian structure of the birkhof polytope. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2019)

5. Brea, J., Simsek, B., Illing, B., Gerstner, W.: Weight-space symmetry in deep networks gives rise to permutation saddles, connected by equal-loss valleys across the loss landscape. arXiv abs/1907.02911 (2019)

6. De Luigi, L., Cardace, A., Spezialetti, R., Zama Ramirez, P., Salti, S., Di Stefano, L.: Deep learning on implicit neural representations of shapes. In: International Conference on Learning Representations (ICLR) (2023)

7. Draxler, F., Veschgini, K., Salmhofer, M., Hamprecht, F.: Essentially no barriers in neural network energy landscape. In: International Conference on Machine Learning (ICML) (2018)

8. Entezari, R., Sedghi, H., Saukh, O., Neyshabur, B.: The role of permutation invari ance in linear mode connectivity of neural networks. In: International Conference on Learning Representations (ICLR) (2022)

9. Frankle, J., Dziugaite, G.K., Roy, D.M., Carbin, M.: Linear mode connectivity and the lottery ticket hypothesis. In: International Conference on Machine Learning (ICML) (2020)

10. Freeman, C.D., Bruna, J.: Topology and geometry of half-rectified network optimization. In: International Conference on Learning Representations (ICLR) (2017)

11. Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., Wilson, A.G.: Loss surfaces, mode connectivity, and fast ensembling of dnns. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2018)

12. Guo, Y., Zhu, C., Zhu, H., Wu, T.: OT4p: Unlocking efective orthogonal group path for permutation relaxation. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2024)

13. Gupta, S., Biggs, S., Laber, M., Shafi, Z., Walters, R., Paul, A.: Deepweightflow: Re-basined flow matching for generating neural network weights. In: International Conference on Learning Representations (ICLR) (2026)

14. Ha, D., Dai, A.M., Le, Q.V.: Hypernetworks. In: International Conference on Learning Representations (ICLR) (2017)

15. Ilharco, G., Ribeiro, M.T., Wortsman, M., Schmidt, L., Hajishirzi, H., Farhadi, A.: Editing models with task arithmetic. In: International Conference on Learning Representations (ICLR) (2023)

16. Ito, A., Yamada, M., Kumagai, A.: Linear mode connectivity between multiple models modulo permutation symmetries. In: International Conference on Machine Learning (ICML) (2025)

17. Jordan, K., Sedghi, H., Saukh, O., Entezari, R., Neyshabur, B.: REPAIR: REnormalizing permuted activations for interpolation repair. In: International Conference on Learning Representations (ICLR) (2023)

18. Li, Z., Li, Z., Lin, J., Shen, T., Xiao, J., Guo, Y., Lin, T., Wu, C.: Improving model fusion by training-time neuron alignment with fixed neuron anchors. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 48(3), 2186– 2199 (2026)

19. Lim, D., Maron, H., Law, M.T., Lorraine, J., Lucas, J.: Graph metanetworks for processing diverse neural architectures. In: International Conference on Learning Representations (ICLR) (2024)

20. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR) (2019)

21. Mena, G., Belanger, D., Linderman, S., Snoek, J.: Learning latent permutations with gumbel-sinkhorn networks. In: International Conference on Learning Representations (ICLR) (2018)

22. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European Conference on Computer Vision (ECCV) (2020)

23. Nasery, A., Hayase, J., Koh, P.W., Oh, S.: Pleas - merging models with permutations and least squares. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2025)

24. Navon, A., Shamsian, A., Fetaya, E., Chechik, G., Dym, N., Maron, H.: Equivariant deep weight space alignment. In: International Conference on Machine Learning (ICML) (2024)

25. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2019)

26. Peña, F.A.G., Medeiros, H.R., Dubail, T., Aminbeidokhti, M., Granger, E., Pedersoli, M.: Re-basin via implicit sinkhorn diferentiation. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2023)

27. Ramirez, P.Z., De Luigi, L., Sirocchi, D., Cardace, A., Spezialetti, R., Ballerini, F., Salti, S., Stefano, L.D.: Deep learning on object-centric 3d neural fields. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 46(12), 9940– 9956 (2024)

28. Rinaldi, F., Capitani, G., Bonicelli, L., Crisostomi, D., Bolelli, F., FICARRA, E., Rodolà, E., Calderara, S., Porrello, A.: Update your transformer to the latest release: Re-basin of task vectors. In: International Conference on Machine Learning (ICML) (2025)

29. Schürholt, K., Kostadinov, D., Borth, D.: Self-supervised representation learning on neural network weights for model characteristic prediction. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2021)

30. Schürholt, K., Mahoney, M.W., Borth, D.: Towards scalable and versatile weight space learning. In: International Conference on Machine Learning (ICML) (2024)

31. Singh, S.P., Jaggi, M.: Model fusion via optimal transport. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2020)

32. Stoica, G., Bolya, D., Bjorner, J.B., Ramesh, P., Hearn, T., Hofman, J.: Zipit! merging models from diferent tasks without training. In: International Conference on Learning Representations (ICLR) (2024)

33. Theus, A., Cabodi, A., Anagnostidis, S., Orvieto, A., Singh, S.P., Boeva, V.: Generalized linear mode connectivity for transformers. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2025)

34. Wang, H., Yurochkin, M., Sun, Y., Papailiopoulos, D., Khazaeni, Y.: Federated learning with matched averaging. In: International Conference on Learning Repre sentations (ICLR) (2020)

35. Wortsman, M., Ilharco, G., Gadre, S.Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A.S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., Schmidt, L.: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In: International Conference on Machine Learning (ICML) (2022)

36. Xu, J., Li, J., Zhang, J.: Scalable model merging with progressive layer-wise distillation. In: International Conference on Machine Learning (ICML) (2025)

37. Zhou, A., Finn, C., Harrison, J.: Universal neural functionals. In: Annual Conference on Neural Information Processing Systems (NeurIPS) (2024)

# Supplemental Materials of Progressive Permutation Alignment with Temperature-Controlled Spectral Sharpening

## A Additional Baseline Comparisons: Activation Matching and STE

In this section, we provide an extended quantitative comparison of our proposed method against two additional baselines: Activation Matching (AM) and the Straight-Through Estimator (STE).

Table S1 presents the interpolation results at the midpoint (λ = 0.5). Our method consistently outperforms the weight-based STE across all metrics (RMSE, PSNR, and SSIM).

While Activation Matching (AM) achieves slightly better alignment metrics, it is crucial to note that AM operates under a fundamentally diferent assumption. AM is a data-dependent approach. In the context of NeRFs, it requires sampling specific camera poses and rays to compute network activations. This requirement not only incurs a massive computational overhead but also severely limits its applicability in practical scenarios where the original training data distribution is unavailable or inaccessible.

In contrast, both STE and our proposed approach are purely weight-based (activation-free), aiming to align models solely using their parameter spaces without relying on any external input data. Comparing data-dependent methods (like AM) directly with weight-only methods is inherently an unequal comparison. We include the AM results here for the sake of completeness, explicitly separating the categories to ensure a fair contextualization of our weight-based method’s performance.

Table S1: Quantitative comparison against Activation Matching (AM) and STE at the interpolation midpoint (λ = 0.5). Note that AM is a data-dependent method, whereas STE and our proposed method are strictly weight-based (activation-free).
<table><tr><td colspan="2">Method Category</td><td colspan="3">RMSE↓PSNR ↑SSIM ↑</td></tr><tr><td>AM</td><td>Data-dependent</td><td>0.161</td><td>16.22</td><td>0.775</td></tr><tr><td>STE</td><td>Weight-based</td><td>0.268</td><td>11.84</td><td>0.661</td></tr><tr><td>Ours</td><td>Weight-based</td><td>0.199</td><td>14.40</td><td>0.710</td></tr></table>

## B Ablation Study

In this section, we conduct ablation studies to evaluate the contribution of each component in our proposed method and to analyze the sensitivity of the

performance to the temperature scheduling of τ. All experiments in this section were performed using the NeRF Synthetic dataset.

## B.1 Importance of Temperature-Controlled Spectral Projection

To verify the efectiveness of our spectral sharpening and projection mechanism, we compared the variants of our proposed method. For reference, we also include the baseline methods (Naive, Weight Matching, and Progressive Weight Matching) detailed in Section 4.1 of the main manuscript:

– Ours (w/o SVD, w/o Ref.): Progressive alignment that bypasses the SVD and spectral flattening steps (Lines 9–10 in Algorithm 1). Instead of the sharpened matrix $M _ { l } ^ { \langle \tau \rangle }$ , it directly uses the raw matrix $M _ { l }$ for the continuous relaxation (Line 11: $P _ { l } \gets \hat { P } _ { l } [ \tau M _ { l } + ( 1 - \tau ) I ] )$ , and omits the final refinement stage.

– Ours $( \mathbf { w } / \mathbf { o } \ \mathbf { S V D } )$ : The variant without spectral flattening (as described above), but followed by the sequential layer-wise refinement.

– Ours $( \mathbf { w } / \mathbf { o } \ \mathbf { R e f . } )$ : The method executing Algorithm 1 completely (i.e., with spectral flattening), but omitting the final refinement stage.

– Ours: Our complete two-stage method combining the spectrally-sharpened continuous alignment (Algorithm 1) with the final sequential layer-wise refinement.

Table S2: Quantitative ablation results on the importance of Spectral Projection and Sequential Layer-wise Refinement. Average metrics across the NeRF Synthetic dataset are reported at the midpoint of interpolation (λ = 0.5).
<table><tr><td>Method</td><td>RMSE↓PSNR ↑ SSIM ↑</td><td></td><td></td></tr><tr><td>Naive</td><td>0.491</td><td>6.205</td><td>0.627</td></tr><tr><td>Weight Matching</td><td>0.294</td><td>11.07</td><td>0.653</td></tr><tr><td>Progressive Weight Matching</td><td>0.306</td><td>10.64</td><td>0.637</td></tr><tr><td>Ours (w/o SVD, w/o Ref.)</td><td>0.264</td><td>12.39</td><td>0.674</td></tr><tr><td>Ours  $( \mathrm { w } / \mathrm { o } \ \mathrm { S V D } )$ </td><td>0.248</td><td>12.70</td><td>0.669</td></tr><tr><td>Ours  $\mathrm { ( w / o \ R e f . ) }$ </td><td>0.199</td><td>14.22</td><td>0.697</td></tr><tr><td>Ours</td><td>0.199</td><td>14.40</td><td>0.710</td></tr></table>

Table S2 and Figure S1 summarize the ablation results. The consistent performance improvement observed from “Ours $\mathrm { ( w / o \ S V D , w / o \ R e f . ) ^ { \prime \prime } }$ to the complete $\mathrm { ^ { 6 } O u r s ^ { 9 } }$ underscores the complementary nature of our algorithmic components. Specifically, the substantial gain achieved by introducing spectral projection (comparing “Ours $( \mathrm { w / o \ S V D ) ^ { \prime \prime } \mathrm { \ v s . \ ^ { \ C } O u r s ^ { \prime \prime } } } )$ demonstrates its critical role: simply interpolating with the raw correlation matrix $M _ { l }$ is insuficient to navigate the complex loss landscape. The temperature-controlled spectral flattening (generating $M _ { l } ^ { \langle \tau \rangle } )$ is essential to actively suppress noise and emphasize dominant permutation structures during the progressive updates. Furthermore, the final sequential layer-wise refinement consistently improves PSNR, indicating that the spectrally-sharpened continuous solution serves as a robust initialization for discrete local optimization.

![](images/7da0a7671f369ef833dec13711ab328b01e17c83351aed7af3bf9f32fa4275ae.jpg)  
Fig. S1: Quantitative evaluation of interpolation trajectories for diferent ablation variants across $\lambda \in [ 0 , 1 ]$ . The integration of spectral projection significantly elevates the overall performance curve, while the addition of sequential layer-wise refinement further stabilizes the alignment, achieving the highest quality at the midpoint $( \lambda = 0 . 5 )$

## B.2 Efect of Temperature Scheduling for τ

Our method relies on a temperature parameter τ to control the sharpening of the spectrum. We investigated how the granularity of the τ schedule afects the rendering quality of interpolated models, and how this continuous initialization interacts with the final sequential layer-wise refinement. We compared three diferent schedules:

1. Two-step: $\tau \in \{ 1 . 0 , 0 . 0 \}$

2. Three-step: $\tau \in \{ 1 . 0 , 0 . 5 , 0 . 0 \}$

3. Five-step: $\tau \in \{ 1 . 0 , 0 . 7 5 , 0 . 5 , 0 . 2 5 , 0 . 0 \}$

The results, presented in Table S3 and Figure S2, reveal a non-monotonic relationship between the schedule granularity and the final alignment quality. While all proposed temperature schedules significantly outperform the standard Weight Matching baselines, the three-step schedule emerges as the clear optimal choice.

Specifically, the two-step binary schedule efectively reduces the search space but occasionally leads to premature fixation on sub-optimal permutations due to the abrupt hard-switch. Conversely, while one might expect a highly gradual schedule (five-step) to yield the best results, it performs similarly to the two-step approach. We hypothesize that excessive intermediate relaxation steps cause the optimization to drift, accumulating alignment errors in the continuous state before the permutations can properly solidify. The three-step schedule provides the optimal “sweet spot” of relaxation: it ofers suficient continuous exploration at $\tau = 0 . 5$ to escape local optima without introducing alignment drift, thereby achieving the highest and most stable performance across all evaluated metrics, both with and without the final sequential refinement.

Table S3: Quantitative ablation results on the temperature scheduling of τ. We evaluate diferent schedule granularities both with and without the final refinement. Average metrics across the NeRF Synthetic dataset are reported at λ = 0.5.
<table><tr><td>Schedule</td><td>RMSE↓PSNR ↑ SSIM ↑</td><td></td><td></td></tr><tr><td>Naive</td><td>0.491</td><td>6.205</td><td>0.616</td></tr><tr><td>Weight Matching</td><td>0.294</td><td>11.07</td><td>0.621</td></tr><tr><td>Progressive Weight Matching</td><td>0.306</td><td>10.64</td><td>0.601</td></tr><tr><td>Two-step  $( \tau \in \{ 1 . 0 , 0 . 0 \} ) ~ \mathrm { w / o ~ R e f } .$ </td><td>0.241</td><td>12.89</td><td>0.698</td></tr><tr><td>Two-step  $( \tau \in \{ 1 . 0 , 0 . 0 \} )$ </td><td>0.256</td><td>12.85</td><td>0.697</td></tr><tr><td>Three-step  $( \tau \in \{ 1 . 0 , 0 . 5 , 0 . 0 \} ) \mathrm { w / o }$  Ref.</td><td>0.199</td><td>14.22</td><td>0.697</td></tr><tr><td>Three-step  $( \tau \in \{ 1 . 0 , 0 . 5 , 0 . 0 \} )$ </td><td>0.199</td><td>14.40</td><td>0.710</td></tr><tr><td>Five-step  $( \tau \in \{ 1 . 0 , 0 . 7 5 , 0 . 5 , 0 . 2 5 , 0 . 0 \} ) \mathrm { ~ w / o ~ }$  Ref.</td><td>0.256</td><td>12.35</td><td>0.695</td></tr><tr><td>Five-step  $( \tau \in \{ 1 . 0 , 0 . 7 5 , 0 . 5 , 0 . 2 5 , 0 . 0 \} )$ </td><td>0.228</td><td>13.45</td><td>0.698</td></tr></table>

![](images/462582fb5cc2d53860d22b4e9bf8c22ecd234acff24286a39b3180c78542c33c.jpg)  
Fig. S2: Impact of temperature scheduling on interpolation quality across diferent λ values. The three-step schedule strikes the optimal balance between continuous exploration and discrete exploitation, outperforming both the abrupt binary hard switch (two-step) and the overly gradual approach (five-step).

## C Additional Quantitative and Qualitative Results

In this section, we provide further quantitative and qualitative evaluations to complement the findings presented in the main manuscript. We first evaluate the computational eficiency of the proposed method by reporting the execution time across diferent network architectures. This is followed by an extensive qualitative analysis, where we present the full-size versions of the comparisons moved from the main paper, rendering results from novel test viewpoints, and comprehensive results for all remaining scenes in the datasets.

## C.1 Computational Eficiency

Tables S4 and S5 summarize the mean and standard deviation of the time required for permutation alignment across various architectures. Our method provides a significant eficiency advantage, as it relies on deterministic progressive alignment via spectral projection, entirely avoiding the costly inner-loop gradient optimization required by continuous relaxation methods like OT4P [12].

Particularly for high-dimensional layers, such as the fully connected layers in VGG11, our complete method achieves alignment in under a minute (48 seconds), whereas the gradient-based methods (OT4P) require about 40 minutes per run, roughly 50 times longer. Furthermore, the results demonstrate that the addition of the final sequential layer-wise refinement (comparing “Proposed w/o Ref.” and “Proposed”) introduces only a small computational overhead (e.g., adding approximately 3 seconds for NeRF). This highlights that our approach successfully achieves the robust alignment quality characteristic of continuous relaxation methods, while maintaining the extreme computational eficiency of purely discrete solvers.

Table S4: Alignment computation time (mean and standard deviation, [sec])
<table><tr><td>Algorithm</td><td>MLP5</td><td>VGG11</td><td>ResNet18</td></tr><tr><td>Weight Matching</td><td> $0 . 5 3 8 \pm 0 . 0 8 0$ </td><td> $1 6 9 . 5 \pm 1 4 . 0 6$ </td><td> $7 . 9 3 7 \pm 0 . 5 0 1$ </td></tr><tr><td>Progressive Weight Matching</td><td> $0 . 0 5 2 \pm 0 . 0 4 7$ </td><td> $1 0 . 4 3 \pm 0 . 0 6 4$ </td><td> $0 . 7 5 2 \pm 0 . 0 9 1$ </td></tr><tr><td>OT4P  $( \tau = 0 . 7 )$ </td><td> $1 5 . 5 4 \pm 1 . 5 7 4$ </td><td> $2 3 0 7 \pm 1 2 . 1 7$ </td><td> $5 4 . 9 9 \pm 5 . 7 4 7$ </td></tr><tr><td>OT4P (τ = 0.5)</td><td> $1 6 . 3 6 \pm 0 . 3 5 8$ </td><td> $2 6 0 5 \pm 5 9 1 . 2$ </td><td> $5 3 . 0 6 \pm 0 . 0 9 7$ </td></tr><tr><td>OT4P  $( \tau = 0 . 3 )$ </td><td> $1 7 . 2 4 \pm 0 . 8 5 0$ </td><td> $2 3 8 0 \pm 1 9 . 8 6$ </td><td> $5 5 . 6 4 \pm 5 . 6 6 1$ </td></tr><tr><td>Progressive OT4P (τ = 0.7)</td><td> $1 0 . 2 4 \pm 0 . 6 2 6$ </td><td> $2 0 5 9 \pm 2 1 9 . 8$ </td><td> $4 6 . 6 6 \pm 4 . 0 8 4$ </td></tr><tr><td>Progressive OT4P  $( \tau = 0 . 5 )$ </td><td> $7 . 3 3 0 \pm 0 . 9 1 1$ </td><td> $2 1 3 7 \pm 1 0 9 . 8$ </td><td> $4 6 . 7 2 \pm 6 . 3 4 5$ </td></tr><tr><td>Progressive OT4P (τ = 0.3)</td><td> $1 4 . 4 6 \pm 0 . 4 9 7$ </td><td> $2 2 1 6 \pm 1 1 . 2 5$ </td><td> $6 0 . 3 3 \pm 3 . 4 5 0$ </td></tr><tr><td>Proposed (Ours) w/o Ref.</td><td></td><td> $0 . 7 5 3 \pm 0 . 0 5 5 3 8 . 9 5 \pm 1 4 . 4 3 1 . 6 9 8 \pm 0 . 0 3 5$ </td><td></td></tr><tr><td>Proposed (Ours)</td><td> $0 . 7 5 8 \pm 0 . 0 5 4$ </td><td>40.22 ± 14.49</td><td> $1 . 8 4 2 \pm 0 . 0 2 2$ </td></tr></table>

Table S5: Computation time for NeRF alignment (mean and variance, [sec])
<table><tr><td>Algorithm</td><td>NeRF (Shuffled Weights) NeRF (Different Initializations)</td></tr><tr><td>Weight Matching</td><td> $1 5 . 6 0 \pm 0 . 8 7 1$ </td></tr><tr><td>Progressive Weight Matching</td><td> $1 5 3 . 7 \pm 2 5 . 0 9$   $1 . 2 6 6 \pm 0 . 0 9 7$   $1 4 . 5 5 \pm 2 . 4 3 0$ </td></tr><tr><td>OT4P (τ = 0.3)</td><td> $4 8 . 4 0 \pm 2 3 . 7 5$ </td></tr><tr><td>OT4P (τ = 0.5)</td><td> $3 8 . 7 2 \pm 1 1 . 1 1$   $6 6 . 5 0 \pm 5 . 6 8 8$   $1 7 . 0 5 \pm 4 . 5 8 5$ </td></tr><tr><td>OT4P (τ = 0.7)</td><td> $6 0 . 6 3 \pm 7 . 6 2 1$   $6 6 . 0 6 \pm 1 3 . 8 3$ </td></tr><tr><td>Progressive OT4P (τ = 0.3)</td><td> $9 3 . 6 2 \pm 3 . 2 0 1$   $9 2 . 2 1 \pm 2 . 0 5 6$ </td></tr><tr><td>Progressive OT4P (τ = 0.5)</td><td> $9 3 . 6 8 \pm 1 . 0 5 2$   $8 1 . 2 9 \pm 2 . 9 8 5$ </td></tr><tr><td>Progressive OT4P (τ = 0.7)</td><td> $9 1 . 6 6 \pm 1 . 1 5 4$   $9 1 . 5 1 \pm 2 . 4 5 4$ </td></tr><tr><td>Proposed (Ours) w/o Ref.</td><td> $3 . 0 6 3 \pm 0 . 0 0 9$   $3 . 2 2 0 \pm 0 . 0 0 5$ </td></tr><tr><td>Proposed (Ours)</td><td> $3 . 1 6 5 \pm 0 . 1 4 9$   $6 . 0 0 1 \pm 0 . 0 7 5$ </td></tr></table>

## C.2 Extended Qualitative Comparisons

We provide additional quantitative and qualitative results on model interpolation for the NeRF Synthetic dataset. While the main paper presents representative examples, this supplementary material reports results for all eight scenes, including SSIM, RMSE, and PSNR curves, together with qualitative interpolation results.

![](images/b9834e4950ecd6d758ed6cc8debd79a592dbf82c0bac14c6e2215f118c0d9635.jpg)  
Fig. S3: Average SSIM over the eight scenes of the NeRF Synthetic dataset. The horizontal axis denotes the interpolation coeficient λ, and the vertical axis denotes the mean SSIM.

Figure S3 shows the average SSIM across all scenes. The proposed method preserves higher structural similarity along the interpolation path than the baselines, particularly near the middle of the interpolation path, where the interpolation quality typically degrades.

Figure S4 shows the SSIM curves for each scene. A consistent trend is observed across all eight scenes, indicating that the proposed method reduces the performance drop around the middle of the interpolation path.

Figure S5 shows the RMSE curves. The proposed method maintains a lower reconstruction error along the interpolation path.

Figure S6 shows the PSNR curves. The proposed method achieves a higher PSNR over a broad range of interpolation coeficients, with the largest diferences appearing near the middle of the interpolation path.

We next present the qualitative interpolation results for each scene. The figures show rendered images at $\lambda \in \{ 0 . 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 \}$ for diferent alignment methods. When the alignment is poor, the intermediate models often produce blurry or distorted renderings, even though both endpoints are valid.

Figure S7 shows qualitative interpolation results for the Lego scene from another test view. Naive interpolation produces severe artifacts in the intermediate models, and existing matching methods alleviate them only partially. In contrast, the proposed method preserves the object structure and appearance more consistently along the interpolation path. The variant with sequential refinement further stabilizes the transition.

A similar trend is observed in the Chair (Figure S8) and Ficus (Figure S9) scenes, where geometric consistency is particularly important. Baseline methods often produce collapsed or view-inconsistent structures around $\lambda = 0 . 5$ , suggesting imperfect alignment in the weight space. The proposed method reduces such failures and yields smoother visual transitions. For Drums (Figure S10), Hotdog (Figure S11), Mic (Figure S12), Materials (Figure S13), and Ship (Figure S14), the proposed method again exhibits the most stable interpolation behavior among the compared methods.

Overall, the qualitative results were consistent with the quantitative trends. Better alignment leads to smoother interpolation in the NeRF weight space, particularly near the middle of the interpolation path, where naive interpolation often fails. These results further support the efectiveness of progressive alignment with temperature-controlled spectral sharpening in independently trained neural fields.

![](images/1deb831818df7cb8766465279e48339350cad9b0643746955e1345f715cfd6a3.jpg)  
(a) Lego

![](images/9a9fba02fc695b709b21acfb257e13ecb0ff55e98486e34ff255c494528e2c63.jpg)  
(b) Chair

![](images/002e41de28de99794f5f8b0d9e4e8254a459b5c78d243e33f7702a336dcedc37.jpg)  
(c) Drums

![](images/bbae5b3c5c474166eedd1e14c0c142463a5c74357790bc30d50f40b9ea8c86ab.jpg)  
(d) Ficus

![](images/fcfe27368fbfc32889528bcd3f6c74d82b353041bd219c39a0705cd61b4b11d9.jpg)  
(e) Hotdog

![](images/951c42a0214faaf2a20ed6fd0217d2b9741249b18645e38726eb2ee9d648fa3b.jpg)  
(f ) Materials

![](images/84037f2ada8f5280f467aa64a56b9413a7859fae931691ab83b5f59ce613f887.jpg)  
(g) Mic

![](images/0b6d30c51019a0a2f65d96f975ce0b246343c59c8a97c57644e319a36be02ef5.jpg)  
(h) Ship  
Fig. S4: SSIM for each scene in the NeRF Synthetic dataset. The proposed method shows a smaller drop in SSIM in the intermediate region of interpolation.

![](images/f12009c9dccfe1cff55c87e126aa4017dde752253eab2d7c8dab381eb888a89d.jpg)  
(a) Lego

![](images/ca184774017a6487ee9ede9fe3299ea706120d2d4c250fcd8ce5334a71dc24e8.jpg)  
(b) Chair

![](images/a3777682fe7d0dc0979e80f3289ae7f28bb064035946e14fb4ffd491488f88ca.jpg)  
(c) Drums

![](images/e0706ad9d9b8cb8c0b55074da0012f1b7bb407b89a6d394a5abd92d9b2360434.jpg)  
(d) Ficus

![](images/005dea141990cb4bf7aca339164c839be353ed68cc9e81958a9f772594a74c8f.jpg)  
(e) Hotdog

![](images/dd422434eb0b0e8b67f450e74a3a9540e625b78b3f72337f14b9a054c70eb5f0.jpg)  
(f ) Materials

![](images/8b95673884bd2043fd754d7d85793f3e228b293d49c7c086d9ca4abcedd2b82e.jpg)  
(g) Mic

![](images/b3dab27a50201f19009fc3616b8361d9c4d6153e2325c7a459b6e9dd85676163.jpg)  
(h) Ship  
Fig. S5: RMSE along the linear interpolation path for all eight scenes of the NeRF Synthetic dataset. Lower is better.

![](images/6f11555091d9c76ef75297e4e3e9bb9d8f2869d69bce6414e19e0582c78b4573.jpg)  
(a) Lego

![](images/a994782940893fd126c1a28d6b577def7f989ef073e65478ae4c2a7a6972d32a.jpg)  
(b) Chair

![](images/93868f36de52cd7e3cbc930ea77fc1a2bca809865f4b9b770965d58c9769e30f.jpg)

![](images/c3fa9410e7f45cdd45d6aca5f9c6d86ac86be0e25635db5eb6e2b0712ceb6e39.jpg)

(c) Drums  
![](images/a3964ef3e7bf6bf50327421f93f70357645e440916d122aa3780ef7a72891c18.jpg)

(e) Hotdog  
![](images/0333d936a165c7fb41a6760d1506a8bcb3b6c6f701a637253e634dc148944578.jpg)  
(f) Materials

(d) Ficus  
![](images/6a00726d7603a58b3758e51290923220fdd2b45512f11eed520453ef24d2acdd.jpg)  
(g) Mic

![](images/823c7534dcccf6751e604cf9c9bb9c7c815d1c40de8bbf1bd294393d09751d28.jpg)  
(h) Ship  
Fig. S6: PSNR along the linear interpolation path for all eight scenes of the NeRF Synthetic dataset. Higher is better.

λ = 0.0

λ = 0.25

λ = 0.50

λ = 0.75

λ = 1.0

Ground Truth

![](images/eaea890172cc5609dfc9ce3a3ff4e23e1d1cfbacfa20e03c722063c60c27e69e.jpg)  
Fig. S7: Interpolation results at diferent values of λ for a novel test view. (a) Naive, (b) Weight Matching, (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, and (g) Proposed.

λ = 0.0

λ = 0.25

λ = 0.50

λ = 0.75

λ = 1.0

Ground Truth

![](images/8196cd910bedc556476f35fa8b66158dd5e3958167fa148e391e76d4c198a241.jpg)  
Fig. S8: Example of interpolation results for diferent λ values for the Chair scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

![](images/d1069f83bc89f255d782318ab528c995ba69ea066f98de075d8dc847576a44ef.jpg)  
Fig. S9: Example of interpolation results for diferent λ values for the Ficus scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

λ = 0.0

λ = 0.25

λ = 0.50

λ = 0.75

λ = 1.0

Ground Truth

![](images/6845103b6df77ac9b7457ca9fa8d87902bcf3d7c079c1f37ec10321101e02f29.jpg)  
Fig. S10: Example of interpolation results for diferent λ values for the Drums scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

λ = 0.0

λ = 0.25

λ = 0.50

λ = 0.75

λ = 1.0

Ground Truth

![](images/0dc8e5eeca28c3c56b15d560f11e2bf0d8ddef84174dbc64ec0650ddd6fdc02a.jpg)  
Fig. S11: Example of interpolation results for diferent λ values for the Hotdog scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

λ = 0.0

λ = 0.25

λ = 0.50

λ = 0.75

λ = 1.0

Ground Truth

![](images/e55006bc5469a1dcf57718dfeb3136ce77229a2f03191b65d50a581a59c1ae14.jpg)  
Fig. S12: Example of interpolation results for diferent λ values for the Mic scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

![](images/966197ab0982485e7ff8cf004c5a6d206abdaa5157a0f0b5bfb743bcb2f97c1d.jpg)  
Fig. S13: Example of interpolation results for diferent λ values for the Materials scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

![](images/8e8be4e1cea969f9060b40f7d9562ab0213d65f4f09f91fbe06a324ea4eec717.jpg)  
Fig. S14: Example of interpolation results for diferent λ values for the Ship scene. (a) Naive, (b) Weight Matching (c) Progressive Weight Matching, (d) OT4P, (e) Progressive OT4P, (f) Proposed w/o Ref, (g) Proposed.

## D Derivation of the Trace-Maximization Form

Consider the following standard weight-alignment objective:

$$
\operatorname* { m i n } _ { \{ P _ { l } \} } \sum _ { l = 1 } ^ { L } \left\| W _ { l } ^ { ( A ) } - P _ { l } W _ { l } ^ { ( B ) } P _ { l - 1 } ^ { \top } \right\| _ { F } ^ { 2 } , \qquad P _ { l } \mathrm { ~ i s ~ a ~ p e r m u t a t i o n ~ m a t r i x . }\tag{9}
$$

When $P _ { l - 1 }$ and $P _ { l + 1 }$ are fixed, the terms that depend on $P _ { l }$ are

$$
\left\| { \boldsymbol { W } } _ { l } ^ { ( A ) } - P _ { l } { \boldsymbol { W } } _ { l } ^ { ( B ) } P _ { l - 1 } ^ { \top } \right\| _ { F } ^ { 2 } + \left\| { \boldsymbol { W } } _ { l + 1 } ^ { ( A ) } - P _ { l + 1 } { \boldsymbol { W } } _ { l + 1 } ^ { ( B ) } P _ { l } ^ { \top } \right\| _ { F } ^ { 2 } .\tag{10}
$$

Expanding the Frobenius norms and using $\| P X Q \| _ { F } = \| X \| _ { F }$ for orthogonal $P$ and $Q .$ , the $P _ { l }$ -dependent part reduces to

$$
- 2 \mathrm { T r } \Big ( ( W _ { l } ^ { ( A ) } ) ^ { \top } P _ { l } W _ { l } ^ { ( B ) } P _ { l - 1 } ^ { \top } \Big ) - 2 \mathrm { T r } \Big ( ( W _ { l + 1 } ^ { ( A ) } ) ^ { \top } P _ { l + 1 } W _ { l + 1 } ^ { ( B ) } P _ { l } ^ { \top } \Big ) .\tag{11}
$$

Using the cyclic property of the trace, this is equivalent, up to constants, to

$$
\operatorname* { m a x } _ { P _ { l } } \mathrm { \mathrm { ~ T r } } ( P _ { l } ^ { \top } C _ { l } ) ,\tag{12}
$$

where

$$
\begin{array} { r } { C _ { l } = W _ { l } ^ { ( A ) } ( P _ { l - 1 } W _ { l } ^ { ( B ) } ) ^ { \top } + ( W _ { l + 1 } ^ { ( A ) } ) ^ { \top } P _ { l + 1 } W _ { l + 1 } ^ { ( B ) } . } \end{array}\tag{13}
$$

## E An example where Progressive Weight Matching fails

Consider a 3-layer linear network with three feature dimensions:

$$
f ( x ) = W _ { 3 } W _ { 2 } W _ { 1 } x , \qquad W _ { l } \in \mathbb { R } ^ { 3 \times 3 } .\tag{14}
$$

As a numerical example, the weights of model A are

$$
W _ { 1 } ^ { ( A ) } = \left[ \begin{array} { l l l } { 5 } & { 5 } & { - 3 } \\ { 3 } & { - 1 } & { - 2 } \\ { - 4 } & { 1 } & { - 3 } \end{array} \right] , \quad W _ { 2 } ^ { ( A ) } = \left[ \begin{array} { l l l } { 2 } & { 4 } & { 5 } \\ { - 5 } & { 1 } & { - 3 } \\ { 0 } & { - 2 } & { - 5 } \end{array} \right] , W _ { 3 } ^ { ( A ) } = \left[ \begin{array} { l l l } { - 5 } & { - 4 } & { 5 } \\ { - 2 } & { 2 } & { - 1 } \\ { - 3 } & { 2 } & { - 5 } \end{array} \right] ,\tag{15}
$$

and the weights of model B are

$$
W _ { 1 } ^ { ( B ) } = \left[ \begin{array} { l } { 3 ~ 4 ~ 5 } \\ { 2 ~ 5 ~ 3 } \\ { 1 ~ 2 ~ - 4 } \end{array} \right] , \quad W _ { 2 } ^ { ( B ) } = \left[ \begin{array} { l l l } { - 2 ~ - 1 ~ 4 } \\ { - 3 ~ 2 ~ 2 } \\ { 2 ~ 2 ~ 4 } \end{array} \right] , W _ { 3 } ^ { ( B ) } = \left[ \begin{array} { l l l } { 0 ~ - 5 ~ - 3 } \\ { 5 ~ - 4 ~ 2 } \\ { 1 ~ 4 ~ 3 } \end{array} \right] .\tag{16}
$$

The ground-truth correspondences are the identity permutations, written here in index notation as $P _ { 1 } ^ { * } = \left[ 0 1 2 \right] , \qquad P _ { 2 } ^ { * } = \left[ 0 1 2 \right]$ . Progressive Weight Matching (WM) alternately solves the optimization problem at each layer:

$$
\operatorname* { m a x } _ { P _ { l } \in \mathcal { P } } \mathrm { T r } ( P _ { l } ^ { \top } C _ { l } )\tag{17}
$$

where the correlation matrices $C _ { 1 }$ and $C _ { 2 }$ depend on the permutation matrices of the other layers, as follows:

$$
\begin{array} { r } { C _ { 1 } ( P _ { 2 } ) = W _ { 1 } ^ { ( A ) } ( W _ { 1 } ^ { ( B ) } ) ^ { \top } + ( W _ { 2 } ^ { ( A ) } ) ^ { \top } P _ { 2 } W _ { 2 } ^ { ( B ) } , } \end{array}\tag{18}
$$

$$
\begin{array} { r } { C _ { 2 } ( P _ { 1 } ) = W _ { 2 } ^ { ( A ) } ( P _ { 1 } W _ { 2 } ^ { ( B ) } ) ^ { \top } + ( W _ { 3 } ^ { ( A ) } ) ^ { \top } W _ { 3 } ^ { ( B ) } , } \end{array}\tag{19}
$$

Here, $P _ { 1 }$ and $P _ { 2 }$ are permutation matrices chosen to maximize the trace terms iteratively, reflecting the alignment between the corresponding weight matrices of models $\mathrm { A }$ and B at each layer.

Given $P _ { 1 } = P _ { 2 } = \left\lceil 0 1 2 \right\rceil$ , the correlation matrices are computed as:

$$
C _ { 1 } = \left[ - 2 0 - 1 1 \ 1 9 \right] , \qquad \operatorname { T r } ( C _ { 1 } ) = 2 4\tag{20}
$$

$$
C _ { 2 } = \left[ \begin{array} { c c c } { { - 1 } } & { { 3 3 } } & { { 3 4 } } \\ { { 9 } } & { { 3 1 } } & { { 2 } } \\ { { - 2 8 } } & { { - 5 5 } } & { { - 5 6 } } \end{array} \right] , \qquad \mathrm { T r } ( C _ { 2 } ) = - 2 6 .\tag{21}
$$

Given $P _ { 1 } = \left\lceil 0 1 2 \right\rceil$ fixed, optimizing $P _ { 2 }$ via Weight Matching on the correlation matrix $C _ { 2 }$ yields the following:

– When $P _ { 1 } = \left\lceil 0 1 2 \right\rceil$ (the correct identity permutation), the trace of $C _ { 2 }$ is −26. – However, the maximum trace is achieved at $P _ { 2 } = \left\lceil 2 1 0 \right\rceil$ , with a trace value of $3 7$

Therefore, Weight Matching updates $P _ { 2 }$ to $P _ { 2 } = { \left[ 2 1 0 \right] }$ . Under this incorrect $P _ { 2 }$ optimizing $P _ { 1 }$ yields

– When $P _ { 1 } = \left[ 0 1 2 \right]$ (the correct permutation), the trace value is $6 .$

– While the maximum trace is achieved at $P _ { 1 } = \left[ 1 0 2 \right]$ with a value of 18.

Repeating this alternating optimization leads to a local fixed point at these incorrect permutations, which difers from the true correspondence $P _ { 1 } = P _ { 2 } =$ -0 1 2. This demonstrates that the iterative Weight Matching procedure can converge to a spurious local optimum rather than the correct alignment.

On the other hand, applying the proposed method of this paper from the same initial values $P _ { 1 } = P _ { 2 } = \left\lceil 0 1 2 \right\rceil$ exhibits diferent behavior.

In each iteration, the proposed method first obtains a provisional anchor $\hat { P } _ { l }$ from the correlation matrix using the Hungarian algorithm. Subsequently, the matrix $\hat { P } _ { l } ^ { \top } C _ { l }$ is spectrally decomposed and sharpened by applying the exponent τ to its spectrum, yielding a matrix that is projected onto the orthogonal group. This procedure gradually sharpens the continuous correlation into a permutationlike orthogonal matrix. The temperature $\tau$ is gradually decreased from 1.0 to 0.0.

The sequence of iterates in this toy example is summarized below:

<table><tr><td>iter</td><td> $( P _ { 1 } , P _ { 2 } )$  (after discretization)</td></tr><tr><td>1</td><td>(0, 1, 2), (0, 1, 2)</td></tr><tr><td>2</td><td>(0, 2, 1), (1, 2,0)</td></tr><tr><td>3</td><td>(1, 0, 2), (0, 1, 2)</td></tr><tr><td>4</td><td>(1, 2, 0), (1, 2,0)</td></tr><tr><td>5</td><td>(0, 1, 2), (0, 1, 2)</td></tr><tr><td>6-8</td><td>(0, 1, 2), (0, 1, 2)</td></tr></table>

As τ decreases, the iterates gradually converge toward permutation-like structures, and the final solution recovers the correct correspondence, $P _ { 1 } = P _ { 2 } = \left[ 0 \ 1 \ 2 \right]$