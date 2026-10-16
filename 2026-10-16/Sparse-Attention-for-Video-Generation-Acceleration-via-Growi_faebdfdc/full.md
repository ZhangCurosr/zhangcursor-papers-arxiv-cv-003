# Sparse Attention for Video Generation Acceleration via Growing Sparsity and Reduced Search

Yun Dai<sup>1</sup> Yuyun Chen<sup>1</sup> Jiarui Wen<sup>1</sup> Huiping Zhuang<sup>1</sup> Cen Chen<sup>1</sup> Ziqian Zeng<sup>1</sup>\* <sup>1</sup>South China University of Technology zqzeng@scut.edu.cn

## Abstract

Diffusion Transformers (DiTs) dominate video generation but sufferfrom high inference cost due to the quadratic complexity of self-attention. Sparse attention mitigates this by skipping unnecessary computation in the attention matrix, but existing methods apply afixed sparsity ratio throughout denoising and struggle to accurately determine which computations to skip. To address these limitations, we analyze attention dynamics through attention entropy and mean attention distance, and find that attention becomes increasingly concentrated across denoising steps. Wefurther observe that blocks selected early continue to capture a large fraction of the total attention weight at later steps. These findings suggest two acceleration opportunities: the sparsity ratio should grow as denoising progresses, and earlier search results can be reused to reduce the search space. Based on these findings, we propose Growing Sparsity and Reduced Search (GSRS), a training-freeframework that retains more computation early and progressively reduces it later, while restricting subsequent searches to an early candidate set to reduce search overhead. GSRS adopts Top-p selection for adaptive sparsity control, supported by a precise evaluation scheme that computes exact attention weights within the FlashAttention computation at negligible overhead. Experiments on HunyuanVideo and Wan 2.1 validate that GSRS achieves 1.55×–2.21× speedup while maintaining generation quality comparable tofull attention.

## 1. Introduction

Diffusion Transformers (DiTs) [38] represent the dominant architecture in the field of video generation [5, 11, 16, 24], with models such as Sora [31], HunyuanVideo [20], and Wan [49] achieving state-of-the-art performance in highresolution and long-duration synthesis. Despite their success, the 3D full attention [48] has emerged as the primary computational bottleneck within this framework [55]. The

3D full attention converts video and text tokens into a single unified token sequence and apply self-attention across them, which causes inference costs to rise sharply as video resolution and temporal length increase. For instance, generating a 125-frame 720p video using HunyuanVideo on a single NVIDIA A800 GPU requires processing more than 115K tokens and takes approximately 38 minutes, with self-attention operations accounting for roughly 80% of the total latency.

Recent studies have shown that attention computation in DiTs contains substantial redundancy [6, 44, 52, 68], with the majority of token interactions making a negligible contribution to the final output. Sparse attention methods [14, 61] exploit this redundancy by partitioning the attention matrix into fixed-size blocks and selecting a subset of blocks to compute. Existing sparse attention strategies generally fall into two categories. Static methods [21, 22, 30, 51, 65] restrict block interactions to fixed windows, and dynamic methods [19, 29, 39, 43, 56, 58, 67] identify critical blocks at runtime. Critical blocks are blocks that actually contribute to the final output. These approaches achieve notable speedups, yet they share two limitations. First, a fixed sparsity ratio or selection threshold may allocate computation inefficiently across denoising steps. Second, accurately and efficiently identifying critical blocks remains challenging. Static methods rely on predefined blocks that may misalign with the ground truth. Evaluating block importance on the fly is expensive for dynamic methods, which may reduce identification accuracy to improve evaluation efficiency.

A detailed analysis of how attention behavior changes during the denoising process can guide the design of a better sparse attention strategy. To characterize this change, we measure how concentrated each query’s attention is via attention entropy [1] and how far each query attends to via mean attention distance [23]. Lower entropy indicates that attention is focused on fewer tokens, while shorter distance means attention is more localized on nearby tokens. We observe that both metrics generally decrease across denoising steps, meaning that attention focuses on fewer and closer tokens as denoising progresses. This observation suggests growing sparsity. Retaining more computation early and progressively reducing it later can improve the quality-efficiency trade-off. Meanwhile, we find that a candidate set of blocks selected early in denoising continues to capture a large fraction of the total attention weight at later steps. This finding suggests reduced search. Subsequent searches can focus on the candidate set established at an early denoising step rather than repeatedly evaluating all blocks, reducing search overhead while maintaining high coverage of critical blocks.

Based on these insights, we propose Growing Sparsity and Reduced Search (GSRS), a training-free framework that translates the above findings into two concrete acceleration strategies. To realize growing sparsity, GSRS progressively lowers selection thresholds over denoising steps, retaining more computation early and allowing greater sparsity later. To realize reduced search, GSRS establishes a candidate set of blocks at an early denoising step and restricts subsequent searches to this set, avoiding repeated evaluation of all blocks. This helps reduce search overhead while maintaining high coverage of critical blocks. Both mechanisms rely on selecting critical blocks at each search step. GSRS adopts cumulative attention probability (Top-p) [17, 25, 72] to adaptively adjust the number of retained blocks based on the actual attention concentration, which requires precise measurement of block importance to correctly select critical blocks. To support reliable $T o p \ – p$ selection, we design a precise block importance evaluation scheme fused into the FlashAttention [9] computation at negligible overhead. Together, these strategies allocate computation according to the changing attention concentration and reduce the cost of identifying critical blocks, improving the quality-efficiency trade-off.

Our contributions are summarized as follows:

• We systematically analyze DiT attention using attention entropy, mean attention distance, and how much attention weight early-selected blocks retain at later denoising steps, providing empirical motivation for growing sparsity and reduced search.

• We propose GSRS, a training-free sparse attention framework that combines growing sparsity and reduced search. To support reliable $T o p \ – p$ selection in GSRS, we design a precise block importance evaluation scheme fused into FlashAttention at negligible overhead.

• Extensive experiments on HunyuanVideo and Wan2.1 demonstrate that GSRS achieves up to 2.21× speedup while maintaining generation quality comparable to full attention, and delivers a superior quality-efficiency trade-off compared to compared sparse attention methods.

## 2. Related Works

Sparse Attention in LLMs. Sparse attention has been extensively studied in large language models to address the quadratic complexity of self-attention over long sequences. StreamingLLM [54] introduces a sliding window with attention sinks for efficient streaming inference. MInference [19] identifies dynamic sparse patterns and performs online approximate search to select critical tokens. MoBA [32] and NSA [59] further explore block-level importance estimation and hardware-aligned sparse designs for efficient long-context understanding. While these methods provide valuable insights for sparse attention research, they are designed for single-forward inference and do not account for the multi-step iterative denoising process in DiTs, limiting their direct applicability to video generation.

Sparse Attention in Video DiTs. Static methods [30, 51, 65] predefine spatial-temporal sparse patterns based on local windows or head-type classification, offering simplicity but limited adaptability to diverse attention behaviors. Dynamic methods [42, 53, 56, 63, 67, 69] perform runtime block selection through various importance estimation strategies, achieving better adaptability but struggling to balance identification accuracy with search overhead. Despite their diverse strategies, these methods do not exploit how attention behavior changes across denoising steps.

Other Acceleration Techniques for DiTs. Beyond sparse attention, several orthogonal techniques accelerate DiT inference. Caching-based methods [7, 28, 34, 35, 57, 70] reduce redundant computation by reusing intermediate activations across denoising steps. Low-bit attention quantization methods [4, 12, 26, 27, 52, 64] accelerate inference by reducing the numerical precision of attention computation. Step distillation methods [10, 13, 15, 36, 62, 71] reduce the number of sampling steps required. Bottleneck Sampling [45] employs variable resolution across different sampling stages to reduce token count during intermediate phases. These approaches are complementary to sparse attention and can be combined with our method for further speedup.

## 3. Preliminary

## 3.1. Attention and FlashAttention

In the self-attention mechanism [48], input tokens are projected into query, key, and value matrices Q, K, $\mathbf { V } \in \mathbb { R } ^ { L \times d } ,$ where L is the sequence length and d is the head dimension. The attention output is computed as:

$$
\mathbf { W } = \mathrm { s o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } \right) , \quad \mathbf { O } = \mathbf { W } \mathbf { V } ,\tag{1}
$$

which requires materializing the full $L \times L$ attention matrix, incurring $O ( L ^ { 2 } )$ time and memory cost. FlashAttention [8, 9, 40, 60] addresses this by partitioning Q, K, V into blocks and computing attention incrementally without materializing the full matrix. During this process, it maintains a per-row Log-Sum-Exp (LSE) [2] value as the global normalization factor, which we later leverage for precise block importance evaluation (§5.3). A detailed description of FlashAttention’s computation is provided in Appendix A.1.

![](images/c05d675c673afeed5adc8109336277f4685fa8f83dd01c64089fe503fae893e3.jpg)  
(a) Hunyuan: Entropy

![](images/ba15c5c49624f93988265649a09ba53993f1907f74c77851eac9995851cbc04c.jpg)  
(b) Hunyuan: MAD

![](images/50a54db9f1fb6436daaf6f99d1e9aebb6d1208a453262dc0dd3ea3ccfc59a8c3.jpg)  
(c) Wan2.1: Entropy

![](images/d6bff2eef11c5d3127f28a97450e1bc0be55ef556e08d1bfd2b0af90047b5a0c.jpg)  
(d) Wan2.1: MAD  
<sub>12</sub> <sup>14</sup>Figure 1. Quantitative analysis of attention dynamics in DiT models. The subfigures (a–d) displays the variation of attention entropy14 <sup>10</sup> 14and mean attention distance (MAD) across 10 diverse prompts, where curves of different shades represent individual samples. Tokens are <sub>20</sub>e e<sup>x</sup> 10 e e<sup>x</sup>reordered via Hilbert curves to preserve spatial-temporal adjacency before computing positional distances (Appendix A.6).

## 40a3.2. Block Sparse Attention

50 20Block sparse attention reduces computation by skipping irrelevant blocks during the FlashAttention loop. Given a <sup>0</sup> <sub>Den</sub>sparse mask $\mathbf { M } \in \{ 0 , 1 \} ^ { L \times L }$ <sup>0</sup> <sup>20</sup> <sup>40</sup><sub>Denoising</sub> <sub>Step</sub>, the sparse attention output is (a) Hunyuan:computed as:

$$
\mathbf { W } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } \odot \mathbf { M } \right) , \quad \mathbf { O } = \mathbf { W } \mathbf { V } ,\tag{2}
$$

where $\odot$ denotes element-wise masking that excludes unselected blocks from the softmax computation. In practice, these blocks are simply skipped during the FlashAttention tiling loop, avoiding both computation and memory access.

We measure attention recall as the fraction of total attention weight captured by the selected blocks:

$$
{ \mathrm { R e c a l l } } ( \mathbf { A } , \mathbf { M } ) = { \frac { \sum _ { i , j } A _ { i j } M _ { i j } } { \sum _ { i , j } A _ { i j } } } ,\tag{3}
$$

where $\mathbf { A } = \mathrm { s o f t m a x } ( \mathbf { Q K } ^ { \top } / \sqrt { d } )$ denotes the full attention weights. Higher recall indicates that the selected blocks capture more of the total attention weight.

![](images/0ce9630ab2a955a8e3fc67f86564217880a52b33adf33f882c6f93e83625fef7.jpg)  
Figure 2. Attention recall under a fixed sparse mask. A sparse mask is constructed at an early denoising step and held fixed at subsequent steps. The mask maintains high attention recall throughout subsequent steps on both HunyuanVideo and Wan2.1.

## a<sup>y 25</sup><sub>H</sub><sup>e</sup>4. Attention Dynamics Analysis

<sub>9 35</sub> 4Before presenting our method, we investigate how attention behavior changes during the denoising process. To charac-Denoising Step Denoising Stepterize these dynamics, we introduce two metrics based on <sup>c)</sup> <sup>Wan2.1:</sup> <sup>Entropy</sup> <sup>(Layer) (d)</sup> the post-softmax attention weight $w _ { i j }$ 2.1: Entropy (Head)<sub>from</sub> <sub>query</sub> <sub>i</sub> <sub>to</sub> <sub>key</sub> j. Attention entropy $H ( i )$ and mean attention distance $D ( i )$ are defined respectively as:

$$
H ( i ) = - \sum _ { j } w _ { i j } \log w _ { i j } , \quad D ( i ) = \sum _ { j } w _ { i j } \cdot | i - j | ,\tag{4}
$$

H(i) measures how concentrated the attention distribution is. Lower entropy means attention is focused on fewer tokens. $D ( i )$ measures the weighted average distance between the query i and the keys j it attends to, where $| i - j |$ denotes the distance between positions i and j in the token sequence. Smaller mean attention distance means attention is focused on nearby tokens.

We conduct systematic analysis on HunyuanVideo and Wan2.1 across multiple prompts, yielding the following key observations.

Observation 1: Attention entropy and distance generally decrease across denoising steps. As shown in Figure 1(ad), both attention entropy and mean attention mean attention distance generally decrease across denoising steps on HunyuanVideo and Wan2.1. This indicates that attention focuses on fewer and closer tokens as denoising progresses. This trend remains consistent across different model architectures and independent of specific inputs, indicating an inherent property of the denoising process. This motivates us to allocate more computation to early denoising steps and progressively increase sparsity later to improve the quality-efficiency trade-off.

Observation 2: An early candidate set of blocks maintains high attention recall at later steps. Figure 3 illustrates that blocks with high attention weights early in denoising continue to capture substantial attention weight at later steps. Blocks with low attention weights early in denoising also tend to retain low weights at later steps. To quantify this observation, we select a candidate set of blocks at an early denoising step and construct a sparse mask from this set. We then keep this mask fixed and measure its attention recall at subsequent denoising steps. As shown in Figure 2, this candidate set continues to capture a large fraction of the total attention weight on both HunyuanVideo and Wan2.1. This motivates us to restrict subsequent searches to the candidate set established early in denoising, reducing search overhead while maintaining high coverage of critical blocks.

![](images/e1c203209e0793ca69819b5581acdcf831b0b9097a05c4a59e44c6ded3ababd2.jpg)  
Figure 3. Block-level attention patterns across denoising steps. Each cell represents the attention weight between a query block and a key block. Two representative heads at steps 5, 17, 29, 37, and 49 on HunyuanVideo. Tokens are reordered via Hilbert curves [3] to preserve spatial-temporal adjacency. As attention becomes more concentrated, blocks with high attention weights early in denoising continue to capture substantial attention weight, while those with low weights tend to retain low weights at later steps.

## 5. Methodology

Guided by the analysis in Section 4, we propose GSRS, a training-free framework that applies increasing sparsity across denoising steps while reducing the search space for block selection. We visualize the workflow of GSRS in Figure 4. At the first search step, GSRS evaluates all blocks and establishes a baseline sparse mask. All subsequent searches are restricted to this baseline mask with progressively tighter Top-p thresholds (§5.1). Between search steps, the most recent mask is directly reused. Central to this framework is Top-p block selection for adaptive sparsity control (§5.2), which requires precise measurement of each block’s importance for reliable selection. To support reliable Top-p selection, we propose a precise block importance evaluation scheme fused into the FlashAttention computation at

negligible overhead (§5.3).

## 5.1. Growing Sparsity and Reduced Search

The search strategy of GSRS comprises two complementary mechanisms: growing sparsity that increases the sparsity ratio across denoising steps to retain more computation early and reduce it later, and reduced search that limits the search space to an early candidate set while preserving dynamic block selection within this set.

Growing Sparsity. As revealed by Observation 1, attention entropy and mean attention distance generally decrease across denoising steps. This motivates allocating more computation to early steps and increasing sparsity later. GSRS employs cumulative attention probability (Top-p) for block selection. Blocks are sorted by their importance in descending order and retained until their cumulative importance exceeds a threshold p. Inspired by the observation, GSRS applies a decreasing threshold $\{ p _ { t } \}$ across search steps, starting from a high $p _ { \mathrm { s t a r t } }$ and gradually lowering to $p _ { \mathrm { e n d } }$ . Early steps retain more blocks because the model’s attention is still spread widely, while later steps safely discard more blocks as attention concentrates on fewer tokens. As validated in Table 2, this decreasing threshold schedule consistently outperforms both fixed and shuffled p configurations under comparable computational budgets.

Reduced Search. Observation 2 shows that a candidate set of blocks selected early in denoising continues to capture a large fraction of the total attention weight at later steps.

![](images/795ec516d7af7db49babeca26d52047ca8f19396f084991633d495c40928c25e.jpg)  
Figure 4. Overview of GSRS. The top row illustrates growing sparsity. As denoising progresses, progressively tighter Top-p thresholds retain fewer blocks, yielding increasingly sparse masks. The bottom row illustrates reduced search. The first search step evaluates all blocks to establish a baseline sparse mask. Subsequent search steps only evaluate blocks within this baseline mask, while non-search steps directly reuse the most recent mask without re-evaluation.

GSRS uses this candidate set to limit subsequent searches while maintaining high coverage of critical blocks. Given a predefined set of search steps $\mathcal { T } = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { n } \}$ , GSRS begins with full attention warmup for the first $N _ { w }$ steps. At step $t _ { 1 } ,$ , a full-space evaluation establishes a baseline sparse mask $S _ { 1 }$ . The blocks selected by $S _ { 1 }$ form a fixed candidate set for subsequent searches. At each subsequent search step $t _ { k } \left( k \geq 2 \right)$ , GSRS re-evaluates all blocks within $S _ { 1 }$ instead of restricting selection to the previously selected mask $S _ { k - 1 }$ It then applies threshold $p _ { k }$ to obtain an updated sparse mask $S _ { k }$ , allowing previously omitted blocks within $S _ { 1 }$ to be selected again. Between search steps, the most recently updated sparse mask is directly reused without re-evaluation. This design preserves the ability to dynamically select blocks while reducing search overhead compared with evaluating all blocks at every search step, as validated in Table 3.

## 5.2. Adaptive Block Selection via Top-p.

Prior works [21, 51, 53] have shown that sparse patterns varies substantially across layers and heads. A fixed Top-k strategy assigns the same number of blocks to every head, regardless of its actual sparse pattern. We also observe this variation in our experiments and provide detailed analysis in Appendix K.4 and K.5. To address this, GSRS employs cumulative attention probability (Top-p) for block selection. Blocks are sorted by their importance in descending order and retained until their cumulative importance exceeds a threshold p (Algorithm 3). For heads where a few blocks carry most of the attention weight, Top-p retains only those few blocks. For heads where attention is distributed more evenly, more blocks are needed to reach the same threshold p. A single threshold p thus achieves adaptive sparsity across layers and heads without any manual tuning. The advantage of Top-p over fixed Top-k is validated in Section 6.3.

## 5.3. Precise Block Importance Evaluation

The Top-p selection described above requires precise measurement of each block’s importance to determine which blocks to retain and how many are needed to reach the cumulative probability threshold. In this section, we explain why obtaining this measurement within FlashAttention is challenging, and present our two-phase evaluation scheme.

Block Importance Definition. Given a query block u and a key block v, each containing B tokens, we define the block importance $W _ { u v }$ as the average attention weight that query block u assigns to key block v:

$$
\begin{array} { l } { { \displaystyle { W _ { u v } = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } w _ { i j } } } } \\ { { \displaystyle ~ = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } \exp ( z _ { i j } - L _ { i } ) . } } \end{array}\tag{5}
$$

Here, $z _ { i j } = \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } / \sqrt { d }$ is the scaled attention score between query i and key $j ,$ , and $w _ { i j }$ denotes the corresponding post-softmax attention weight. The row-wise normalization factor $L _ { i } = \mathrm { L S E } ( \mathbf { z } _ { i } )$ is computed via Eq. 12, where $\mathbf { z } _ { i } = [ z _ { i 1 } , z _ { i 2 } , \ldots , z _ { i L } ]$ contains the attention scores between query i and all key tokens. For each query block $u ,$ the block importance $W _ { u v }$ sums to 1 over all key blocks $v .$ This property is essential for Top-p selection, which relies on accumulating block importance until the sum reaches the threshold $p .$

The Challenge of Computing Precise Block Importance. Computing $W _ { u v }$ requires the global normalization factor $L _ { i }$ which depends on the scores between query i and all key tokens. However, FlashAttention processes key blocks one at a time and only produces the final $L _ { i }$ after the last key block has been processed (see Appendix A.1). This means that during the computation, we cannot yet compute exact attention weights for any individual block. Prior work [53] addresses this by reusing the $L _ { i }$ cached from a previous denoising step, denoted $L _ { i } ^ { \mathrm { { o l d } } }$ , to compute an approximate block importance:

$$
\hat { W } _ { u v } = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } \exp ( z _ { i j } - L _ { i } ^ { \mathrm { o l d } } ) ,\tag{6}
$$

However, as the attention distribution changes across denoising steps, the true normalization factor $L _ { i } ^ { \mathrm { n e w } }$ drifts away from the cached $L _ { i } ^ { \mathrm { { o l d } } }$ . This drift scales each row’s block importance by a factor of $\exp ( L _ { i } ^ { \mathrm { n e w } } - L _ { i } ^ { \mathrm { o l d } } )$ , causing them to no longer sum to 1. Since Top-p selects blocks until their importance sums exceed a threshold $p ,$ inaccurate sums directly affect the selection. When the sums exceed 1, the threshold is reached too early, causing important blocks to be incorrectly discarded.

Two-Phase Block Importance Evaluation. GSRS computes block importance in two phases without relying on cached LSE values. During FlashAttention computation, we record the log-sum-exp of attention scores for each query row within each key block. Once the current row-wise LSE becomes available, we normalize these stored statistics and average the normalized weights over query rows to obtain block importance.

Phase 1: Fused block log-sum-exp computation. During FlashAttention computation, we compute and store the logsum-exp of attention scores $G _ { i v }$ within each key block:

$$
G _ { i v } = m _ { i v } + \log \sum _ { j \in v } \exp ( z _ { i j } - m _ { i v } ) ,\tag{7}
$$

where $m _ { i v } ~ = ~ \operatorname* { m a x } _ { j \in v } z _ { i j }$ is the maximum score within key block v for query i. Computing $\exp ( z _ { i j } - m _ { i v } )$ keeps the exponent non-positive, preventing numerical overflow.

We store only one scalar $G _ { i v }$ per query row and key block, avoiding storage of the full $L \times L$ attention matrix.

Phase 2: Block importance recovery. After FlashAttention finishes and the current row-wise LSE $L _ { i }$ is available, we recover block importance as

$$
\begin{array} { l } { { \displaystyle { \cal W } _ { u v } = \frac { 1 } { B } \sum _ { i \in u } \exp ( G _ { i v } - L _ { i } ) } } \\ { { \displaystyle ~ = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } \exp ( z _ { i j } - L _ { i } ) } } \\ { { \displaystyle ~ = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } w _ { i j } . } } \end{array}\tag{8}
$$

This computation recovers the exact block importance defined in Eq. (5) without using cached LSE. A detailed derivation is provided in Appendix A.2.

Phase 1 is fused into FlashAttention, while Phase 2 requires only lightweight post-processing of the stored statistics. Together, they require no additional pass over the data and introduces negligible overhead in both latency (Figure 5) and memory (Table 6). The complete procedure is detailed in Appendix A.3.

Differences from AdaSpa and LiteAttention. AdaSpa [53] uses a fixed sparsity target and evaluates all blocks at scheduled search steps using cached LSE values. GSRS instead progressively lowers Top-p thresholds and restricts subsequent searches to $S _ { 1 }$ , reducing both attention computation and search overhead. Moreover, GSRS computes block importance from block log-sum-exp statistics and the current row-wise LSE, avoiding errors caused by cached LSE values and supporting reliable Top-p/Top-k selection. Once LiteAttention [41] skips a block, it excludes that block from both evaluation and computation for the remaining denoising steps. GSRS re-evaluates all blocks within $S _ { 1 }$ at each search step, allowing previously omitted blocks to be selected again as attention changes. See Appendix J for more details.

## 6. Experiments

## 6.1. Experimental Setup

Models and Datasets. We evaluate GSRS on two open-sourced DiT models, Wan2.1-T2V-14B [49] and HunyuanVideo-T2V-13B [20]. Both generate 720p videos with 50 denoising steps. In latent space, Wan2.1 and HunyuanVideo process 21 and 32 frames respectively, with 3,600 tokens per frame. For all experiments, we use the original dataset from VBench [18] for both models.

Metrics. We use PSNR [37], SSIM [50] and LPIPS [66] to measure pixel-level similarity between videos generated with sparse attention and full attention. For video quality evaluation, we report Subject Consistency, Imaging Quality, and the overall quality score VBench-Q from VBench [18]. To quantify the computational saving of sparse attention, we report density, defined as the ratio of actual computation performed to that of full attention. For efficiency, we report the end-to-end latency and the overall speedup.

Table 1. Quantitative evaluation of quality and efficiency. We evaluate on Wan 2.1 and HunyuanVideo. Bold and underline denote the best and second best results.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="6">Quality</td><td colspan="3">Efficiency</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>SubConsist ↑</td><td>ImageQual ↑</td><td>VBench-Q↑</td><td>Density ↓</td><td>Latency ↓</td><td>Speedup ↑</td></tr><tr><td rowspan="6">Wan 2.1</td><td>Full Attention</td><td></td><td>-</td><td>一</td><td>98.26%</td><td>73.29%</td><td>83.75%</td><td>100%</td><td>3689s</td><td>1.00×</td></tr><tr><td>SpargeAttn [63]</td><td>20.55</td><td>0.757</td><td>0.295</td><td>94.88%</td><td>72.80%</td><td>82.21%</td><td>42.00%</td><td>2732s</td><td>1.35×</td></tr><tr><td>SVG [51]</td><td>20.87</td><td>0.786</td><td>0.277</td><td>97.74%</td><td>71.99%</td><td>83.69%</td><td>40.14%</td><td>2675s</td><td>1.38×</td></tr><tr><td>SVG 2 [58]</td><td>22.83</td><td>0.847</td><td>0.214</td><td>97.80%</td><td>72.23%</td><td>83.19%</td><td>32.08%</td><td>2455s</td><td>1.50×</td></tr><tr><td>SVOO [33]</td><td>23.34</td><td>0.865</td><td>0.195</td><td>97.89%</td><td>72.42%</td><td>83.33%</td><td>32.82%</td><td>2610s</td><td>1.41×</td></tr><tr><td>GSRS</td><td>23.79</td><td>0.872</td><td>0.187</td><td>98.21%</td><td>73.22%</td><td>83.72%</td><td>30.63%</td><td>2380s</td><td>1.55×</td></tr><tr><td rowspan="6">Hunyuan</td><td>Full Attention</td><td>=</td><td>=</td><td>一</td><td>96.82%</td><td>70.71%</td><td>83.98%</td><td>100%</td><td>3057s</td><td>1.00×</td></tr><tr><td>SpargeAttn [63]</td><td>27.79</td><td>0.907</td><td>0.192</td><td>96.55%</td><td>70.65%</td><td>83.85%</td><td>33.00%</td><td>2014s</td><td>1.52×</td></tr><tr><td>SVG [51]</td><td>27.64</td><td>0.895</td><td>0.221</td><td>96.71%</td><td>68.98%</td><td>83.21%</td><td>31.65%</td><td>2066s</td><td>1.48×</td></tr><tr><td>SVG 2 [58]</td><td>29.27</td><td>0.914</td><td>0.213</td><td>96.60%</td><td>66.50%</td><td>82.24%</td><td>26.38%</td><td>1978s</td><td>1.55×</td></tr><tr><td>SVOO [33]</td><td>30.31</td><td>0.931</td><td>0.172</td><td>96.88%</td><td>69.19%</td><td>82.97%</td><td>27.12%</td><td>1975s</td><td>1.55×</td></tr><tr><td>GSRS</td><td>32.37</td><td>0.957</td><td>0.138</td><td>96.66%</td><td>70.79%</td><td>84.01%</td><td>25.06%</td><td>1825s</td><td>1.68×</td></tr></table>

Baselines. We compare GSRS against both static and dynamic sparse attention methods. For static methods, we include SVG [51]. For dynamic methods, we include SpargeAttention [63], SVG 2 [58] and SVOO [33]. All baselines are evaluated under the same hardware configuration and input settings as GSRS.

Implementation Details. We implement GSRS using Tile-Lang [46] for fused attention computation and block importance evaluation at search steps, and the CUDA kernels from Block Sparse Attention [14] for sparse attention computation at non-search steps. All experiments are conducted on a single NVIDIA A800 GPU with CUDA 12.8. Tokens are reordered using the Generalized Hilbert curve [3] with a block size of B = 128. By default, we perform search at denoising steps $\mathcal { T } = \{ 1 0 , 2 0 , 3 0 , 4 0 \}$ with Top-p thresholds $p = \{ 0 . 9 5 , 0 . 9 , 0 . 8 5 , 0 . 8 \}$ . For all experiments, the first 20% of denoising steps are executed with full attention as warmup before applying sparse computation. Experiments with different warmup lengths are provided in Appendix C.

## 6.2. Main Results

Similarity. As shown in Table 1, GSRS achieves the highest PSNR and SSIM on both models, reaching an average

PSNR of 32.75 on HunyuanVideo and 23.79 on Wan2.1, outperforming all baselines. The high similarity scores indicate that the sparse attention output of GSRS closely approximates full attention, preserving fine-grained details in the generated videos. Figure 8 further shows that GSRS also achieves a more stable distribution, indicating more consistent performance across diverse prompts.

Quality. GSRS maintains VBench-Q scores close to full attention and outperforms all evaluated sparse attention baselines on both models. GSRS achieves the highest VBench-Q among the evaluated sparse attention methods on both models, with scores of 83.72% on Wan2.1 and 84.01% on HunyuanVideo, comparable to 83.75% and 83.98% for full attention. Full VBench-Q results are provided in Appendix B.

Efficiency. GSRS achieves the lowest attention density and the highest speedup among the evaluated sparse attention methods on both models. It reduces density to 30.63% on Wan2.1 and 25.06% on HunyuanVideo, achieving end-toend inference speedups of 1.55× and 1.68×, respectively. As shown in Appendix G, GSRS also achieves the highest speedup among the evaluated methods on NVIDIA H800, reaching 1.78× on Wan2.1 and 2.21× on HunyuanVideo.

## 6.3. Ablation Study

We conduct ablation studies on HunyuanVideo to validate each design choice of GSRS. All ablations use the default configuration with search steps $\mathcal { T } = \{ 1 0 , 2 0 , 3 0 , 4 0 \}$ and cumulative probability thresholds $p = \{ 0 . 9 5 , 0 . 9 , 0 . 8 5 , 0 . 8 \}$ Experiments are evaluated on a fixed subset of the VBench dataset. VB-Q represent VBench Quality (VBench-Q). Since PSNR measures pixel-level deviation while VBench-Q better reflects perceptual quality, we consider both metrics with emphasis on VBench-Q for quality assessment.

Table 2. Effect of Top-p schedule configurations on HunyuanVideo.
<table><tr><td>p schedule</td><td>PSNR↑</td><td>VB-Q↑</td><td>Latency↓</td></tr><tr><td>{0.95, 0.90, 0.85, 0.80}</td><td>31.96</td><td>84.99</td><td>1825</td></tr><tr><td>{0.95, 0.80, 0.85, 0.90}</td><td>31.64</td><td>84.98</td><td>1850</td></tr><tr><td>{0.93, 0.93, 0.93, 0.93}</td><td>31.20</td><td>84.90</td><td>1852</td></tr><tr><td>{0.90, 0.90, 0.90, 0.90}</td><td>30.28</td><td>84.89</td><td>1774</td></tr></table>

Table 4. Ablation on search steps T on HunyuanVideo.
<table><tr><td>T</td><td>PSNR↑ VB-Q↑</td><td>Latency↓</td></tr><tr><td>{10}</td><td>32.94 84.93</td><td>2076</td></tr><tr><td>{10, 30}</td><td>32.59 84.88</td><td>1870</td></tr><tr><td>{10, 20, 30}</td><td>31.89 84.88</td><td>1845</td></tr><tr><td>{10, 30, 40}</td><td>32.59 84.89</td><td>1882</td></tr><tr><td>{10, 20, 30, 40}</td><td>31.96 84.99</td><td>1825</td></tr></table>

Decreasing p Schedule. As shown in Table 2, the decreasing schedule achieves the highest PSNR and VBench-Q among the compared configurations. The shuffled schedule lowers PSNR to 31.64 at similar latency, highlighting the benefit of decreasing thresholds over denoising steps. Compared with the fixed schedule with $p = 0 . 9 3$ , the decreasing schedule improves PSNR from 31.20 to 31.96 while reducing latency from 1852s to 1825s. These results support retaining more computation early and progressively reducing it later to improve the quality-efficiency trade-off. Appendix F provides results for additional Top-p configurations.

Search Strategies. With previous-mask search, each search step is restricted to the mask $S _ { k - 1 }$ selected at the preceding search step, so previously excluded blocks cannot be selected again. Without Reduced Search, each search step evaluates all blocks in the full attention matrix instead of restricting to the baseline mask $\scriptstyle { S _ { 1 } }$ . Table 3 shows that removing Reduced Search increases density from 25.06% to 28.62% and latency from 1825s to 1990s while maintaining comparable VBench-Q. As shown in Figure 7, reduced search lowers the total search overhead by 13.75× on HunyuanVideo, confirming that restricting searches to the baseline mask is both safe and efficient. Previous-mask search reduces latency to 1757s but lowers PSNR and VBench-Q to 30.87 and 84.80, respectively, highlighting the benefit of preserving dynamic block selection within $S _ { 1 }$

Precise Block Importance Evaluation. We replace our precise block importance evaluation scheme with an estimator that uses cached LSE values from a previous denoising step. As shown in Table 3, this reduces PSNR from 31.96 to 25.32 and VBench-Q from 84.99 to 84.04. Cached LSE values distort block importance and can alter both cumulative probability sums and block rankings, leading to inaccurate selection. These results demonstrate the effectiveness of our precise block importance evaluation scheme in supporting reliable Top-p selection. Our scheme also enables accurate Top-k block selection, as further evaluated in Appendix J.

Table 3. Ablation on search strategies and block importance evaluation on HunyuanVideo. Den. represents Density.
<table><tr><td>Config</td><td>PSNR↑</td><td>VB-Q↑</td><td></td><td>Den.↓ Latency↓</td></tr><tr><td>GSRS</td><td>31.96</td><td>84.99</td><td>25.06</td><td>1825</td></tr><tr><td>w/ Cached LSE</td><td>25.32</td><td>84.04</td><td>20.89</td><td>1730</td></tr><tr><td>w/ Previous-mask Search</td><td>30.87</td><td>84.80</td><td>23.06</td><td>1757</td></tr><tr><td>w/o Reduced Search</td><td>32.57</td><td>84.98</td><td>28.62</td><td>1990</td></tr></table>

Table 5. Comparison of Top-p and Top-k selection strategies on HunyuanVideo.
<table><tr><td>Strategy</td><td>PSNR↑ VB-Q↑</td><td>Latency↓</td></tr><tr><td>Top-p (ours)</td><td>31.96 84.99</td><td>1825</td></tr><tr><td>Top-k (25%)</td><td>29.84</td><td>84.95 1813</td></tr><tr><td>Top-k (50%)</td><td>32.67 85.03</td><td>2245</td></tr></table>

Top-p vs Top-k. Table 5 compares Top-p with Top-k under matched density. Top-k (25%) matches the average density of Top-p but only achieves a PSNR of 29.84, significantly lower than Top-p’s 31.96 at similar latency. Top-k (50%) achieves slightly higher PSNR but incurs substantially higher latency of 1304s with only marginal improvement in VBench-Q. This confirms that adaptive Top-p selection allocates computation more effectively than fixed budgets.

Search Steps. Table 4 shows that all configurations achieve high PSNR above 31.89 and stable VBench-Q scores, indicating that the generation quality is well preserved regardless of the number of search steps. Among them, the four-step configuration {10, 20, 30, 40} achieves the highest VBench-Q score while maintaining low latency, resulting in the best overall trade-off between quality and efficiency.

## 7. Conclusion

In this paper, we propose GSRS, a training-free sparse attention framework featuring grow sparsity and reduced search to accelerate video DiT inference. Combined with Top-p and precise block importance evaluation, GSRS achieves up to 2.21× speedup while maintaining generation quality comparable to full attention.

## References

[1] Ali Araabi, Vlad Niculae, and Christof Monz. Entropy–and distance-regularized attention improves low-resource neural machine translation. In Proceedings ofthe 16th Conference ofthe Associationfor Machine Translation in the Americas (Volume 1: Research Track), pages 140–153, 2024. 1

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014. 2

[3] Jakub Cervený and Contributors. Gilbert: Space-filling<sup>ˇ</sup> curve for rectangular domains of arbitrary size. https: //github.com/jakubcerveny/gilbert, 2025. Accessed: 2025-04-16. 4, 7, 17

[4] Yi-Chung Chen, Zhi-Kai Huang, and Jing-Ren Chen. Stepbaq: Stepping backward as correction for quantized diffusion models. Advances in Neural Information Processing Systems, 37:54054–54078, 2024. 2

[5] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101, 2024. 1

[6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. 1

[7] Huanpeng Chu, Wei Wu, Guanyu Feng, and Yutao Zhang. Omnicache: A trajectory-oriented global perspective on training-free cache reuse for diffusion transformer models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16302–16312, 2025. 2

[8] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024. 2

[9] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022. 2

[10] Zihan Ding, Chi Jin, Difan Liu, Haitian Zheng, Krishna Kumar Singh, Qiang Zhang, Yan Kang, Zhe Lin, and Yuchen Liu. Dollar: Few-step video generation via distillation and latent reward optimization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 17961– 17971, 2025. 2

[11] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, and Tat-Seng Chua. Dysen-vdm: Empowering dynamics-aware textto-video diffusion with llms. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7641–7653, 2024. 1

[12] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022. 2

[13] Youping Gu, Xiaolong Li, Yuhao Hu, Minqi Chen, and Bohan Zhuang. Blade: Block-sparse attention meets step distillation for efficient video generation. arXiv preprint arXiv:2508.10774, 2025. 2

[14] Jialin Guo, Haotian Tang, Shang Yang, Zhao Zhang, Zirui Liu, and Song Han. Block sparse attention. https://github. com/mit-han-lab/Block-Sparse-Attention, 2024. 1, 7, 16

[15] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2

[16] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 1

[17] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2019. 2

[18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6, 7

[19] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024. 1, 2

[20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 6

[21] Qirui Li, Guangcong Zheng, Qi Zhao, Jie Li, Bin Dong, Yiwu Yao, and Xi Li. Compact attention: Exploiting structured spatio-temporal sparsity for fast video generation. arXiv preprint arXiv:2508.12969, 2025. 1, 5

[22] Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O(n log n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025. 1

[23] Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, et al. Attention illuminates llm reasoning: The preplan-and-anchor rhythm enables fine-grained policy optimization. arXiv preprint arXiv:2510.13554, 2025. 1

[24] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 1

[25] Chaofan Lin, Jiaming Tang, Shuo Yang, Hanshuo Wang, Tian Tang, Boyu Tian, Ion Stoica, Song Han, and Mingyu Gao. Twilight: Adaptive attention sparsity with hierarchical top-p pruning. arXiv preprint arXiv:2502.02770, 2025. 2

[26] Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. Advances in Neural Information Processing Systems, 37:87766–87800, 2024. 2

[27] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings ofmachine learning and systems, 6:87–100, 2024. 2

[28] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 7353–7363, 2025. 2

[29] Liu Liu, Zheng Qu, Zhaodong Chen, Yufei Ding, and Yuan Xie. Transformer acceleration with dynamic sparse attention. arXiv preprint arXiv:2110.11299, 2021. 1

[30] Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Clear: Conv-like linearization revs pre-trained diffusion transformers up. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 2

[31] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 1

[32] Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, et al. Moba: Mixture of block attention for longcontext llms. arXiv preprint arXiv:2502.13189, 2025. 2

[33] Jiayi Luo, Jiayu Chen, Jiankun Wang, Cong Wang, Hanxin Zhu, Qingyun Sun, Chen Gao, Zhibo Chen, and Jianxin Li. Attention sparsity is input-stable: Training-free sparse attention for video generation via offline sparsity profiling and online QK co-clustering. In Proceedings of the 43rd International Conference on Machine Learning, pages 82177–82195. PMLR, 2026. 7, 18

[34] Zhengyao Lv, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K Wong. Fastercache: Training-free video diffusion model acceleration with high quality. In The Thirteenth International Conference on Learning Representations, 2025. 2

[35] Zehong Ma, Longhui Wei, Feng Wang, Shiliang Zhang, and Qi Tian. Magcache: Fast video generation with magnitudeaware cache. arXiv preprint arXiv:2506.09045, 2025. 2

[36] Weili Nie, Julius Berner, Nanye Ma, Chao Liu, Saining Xie, and Arash Vahdat. Transition matching distillation for fast video generation. arXiv preprint arXiv:2601.09881, 2026. 2

[37] OpenCV Contributors. Opencv: Open source computer vision library. https://github.com/opencv/opencv, 2025. Accessed: 2025-02-26. 6

[38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 1

[39] Yifan Pu, Zhuofan Xia, Jiayi Guo, Dongchen Han, Qixiu Li, Duo Li, Yuhui Yuan, Ji Li, Yizeng Han, Shiji Song, et al. Efficient diffusion transformer with step-wise dynamic attention mediators. In European Conference on Computer Vision, pages 424–441. Springer, 2024. 1

[40] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024. 2

[41] Dor Shmilovich, Tony Wu, Aviad Dahan, and Yuval Domb. Liteattention: A temporal sparse attention for diffusion transformers. arXiv preprint arXiv:2511.11062, 2025. 6, 21

[42] Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Jingyi Liao, Zhao Jin, Shunyu Liu, and Dacheng Tao. Vorta: Efficient video diffusion via routing sparse attention. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2

[43] Xin Tan, Yuetao Chen, Yimin Jiang, Xing Chen, Kun Yan, Nan Duan, Yibo Zhu, Daxin Jiang, and Hong Xu. Dsv: Exploiting dynamic sparsity to accelerate large-scale video dit training. arXiv preprint arXiv:2502.07590, 2025. 1

[44] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: query-aware sparsity for efficient long-context llm inference. In Proceedings of the 41st International Conference on Machine Learning, pages 47901–47911, 2024. 1

[45] Ye Tian, Xin Xia, Yuxi Ren, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Yunhai Tong, Ling Yang, and Bin Cui. Trainingfree diffusion acceleration with bottleneck sampling. arXiv preprint arXiv:2503.18940, 2025. 2

[46] TileLang. Tilelang github repository. https://github. com/tile-ai/tilelang, 2024. 7, 16

[47] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019. 23

[48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1, 2

[49] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 6

[50] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6

[51] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse video-gen: Accelerating video diffusion transformers with spatial-temporal sparsity. In International Conference on Machine Learning, pages 68208–68224. PMLR, 2025. 1, 2, 5, 7, 18

[52] Yifei Xia, Fangcheng Fu, Wentao Zhang, Jiawei Jiang, and Bin Cui. Efficient multi-task llm quantization and serving for multiple lora adapters. Advances in Neural Information Processing Systems, 37:63686–63714, 2024. 1, 2

[53] Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Training-free and adaptive sparse attention for efficient long video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15982–15993, 2025. 2, 5, 6, 21

[54] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2023. 2

[55] Shuo Xiao, Dongqing Zhu, Chaogang Tang, and Zhenzhen Huang. Combining graph contrastive embedding and multihead cross-attention transfer for cross-domain recommendation. Data Science and Engineering, 8(3):247–262, 2023. 1

[56] Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. Xattention: Block sparse attention with antidiagonal scoring. In International Conference on Machine Learning, pages 69819–69831. PMLR, 2025. 1, 2

[57] Qirui Yang, Runyu Jin, Ni Fan, Devasena Inupakutika, Bridget Davis, and Ming Zhao. Adacache: A disaggregated cache system with adaptive block size for cloud block storage. In 2023 IEEE 16th International Conference on Cloud Computing (CLOUD), pages 348–359. IEEE, 2023. 2

[58] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 7, 18

[59] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, et al. Native sparse attention: Hardwarealigned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23078– 23097, 2025. 2

[60] Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, and Tri Dao. Flashattention-4: Algorithm and kernel pipelining co-design for asymmetric hardware scaling. arXiv preprint arXiv:2603.05451, 2026. 2

[61] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020. 1

[62] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. Accvideo: Accelerating video diffusion model with synthetic dataset. arXiv preprint arXiv:2503.19462, 2025. 2

[63] Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, and Jianfei Chen. Spargeattention: Accurate and training-free sparse attention accelerating any model inference. In International Conference on Machine Learning, pages 76397–76413. PMLR, 2025. 2, 7, 18

[64] Jintao Zhang, Pengle Zhang, Jun Zhu, Jianfei Chen, et al. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. In The Thirteenth International Conference on Learning Representations, 2025. 2

[65] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. In International Conference on Machine Learning, pages 74714–74731. PMLR, 2025. 1, 2, 17

[66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 586–595. IEEE, 2018. 6

[67] Yuechen Zhang, Jinbo Xing, Bin Xia, Shaoteng Liu, Bohao PENG, Xin Tao, Pengfei Wan, Eric Lo, and Jiaya Jia. Training-free efficient video generation via dynamic token carving. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 2, 17

[68] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023. 1

[69] Tianchen Zhao, Ke Hong, Xinhao Yang, Xuefeng Xiao, Huixia Li, Feng Ling, Ruiqi Xie, SiQi Chen, Hongyu Zhu, Zhang Yichong, et al. Paroattention: Pattern-aware reordering for efficient sparse and quantized attention in visual generation models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2

[70] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Realtime video generation with pyramid attention broadcast. In The Thirteenth International Conference on Learning Representations, 2024. 2

[71] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025. 2

[72] Kan Zhu, Tian Tang, Qinyu Xu, Yile Gu, Zhichen Zeng, Rohan Kadekodi, Liangyu Zhao, Ang Li, Arvind Krishnamurthy, and Baris Kasikci. Tactic: Adaptive sparse attention with clustering and distribution fitting for long-context llms. arXiv preprint arXiv:2502.12216, 2025. 2

# Sparse Attention for Video Generation Acceleration via Growing Sparsity and Reduced Search

## A. Algorithm Details

## A.1. FlashAttention Details

Standard Softmax and Numerical Instability. Given a row of pre-softmax scores ${ \bf z } = [ z _ { 1 } , z _ { 2 } , \dots , z _ { N } ]$ , the standard softmax computes:

$$
\operatorname { s o f t m a x } ( z _ { j } ) = { \frac { \exp ( z _ { j } ) } { \sum _ { k = 1 } ^ { N } \exp ( z _ { k } ) } } .\tag{9}
$$

When the scores $z _ { j }$ are large, the exponentials $\exp ( z _ { j } )$ can overflow, leading to numerical errors. A common solution is to subtract the row-wise maximum $m = \operatorname* { m a x } _ { j } z _ { j }$ before exponentiation:

$$
\operatorname { s o f t m a x } ( z _ { j } ) = { \frac { \exp ( z _ { j } - m ) } { \sum _ { k = 1 } ^ { N } \exp ( z _ { k } - m ) } } .\tag{10}
$$

This is valid because softmax is shift-invariant: for any constant c,

$$
{ \frac { \exp ( z _ { j } - c ) } { \sum _ { k } \exp ( z _ { k } - c ) } } = { \frac { \exp ( z _ { j } ) \cdot \exp ( - c ) } { \sum _ { k } \exp ( z _ { k } ) \cdot \exp ( - c ) } } = { \frac { \exp ( z _ { j } ) } { \sum _ { k } \exp ( z _ { k } ) } } .\tag{11}
$$

Subtracting the maximum ensures that the largest exponent is $\exp ( 0 ) = 1$ , preventing overflow while producing identical results.

Log-Sum-Exp (LSE). The denominator of softmax can be compactly represented using the Log-Sum-Exp (LSE). For a score vector ${ \bf z } = [ z _ { 1 } , z _ { 2 } , \dots , z _ { N } ]$ , LSE is defined as:

$$
\mathrm { L S E } ( \mathbf { z } ) = m + \log \sum _ { j = 1 } ^ { N } \exp ( z _ { j } - m ) , \quad \mathrm { w h e r e \ } m = \operatorname* { m a x } _ { j } z _ { j } .\tag{12}
$$

With LSE, each softmax weight $w _ { j }$ for the j-th element can be recovered as:

$$
\begin{array} { r } { w _ { j } = \exp ( z _ { j } - \mathrm { L S E } ( \mathbf { z } ) ) . } \end{array}\tag{13}
$$

LSE serves as the global normalization factor in log-space. It plays a central role in FlashAttention’s incremental computation, as we describe next.

Online Softmax and Incremental Computation. Standard softmax requires two passes over the data: one to compute the maximum and denominator, and another to normalize. This is impractical when the full score vector z is not available in memory, as is the case in FlashAttention where scores are computed block by block.

Online softmax solves this by simultaneously accumulating the numerator and denominator of softmax as each block is processed, deferring the final division to the end. Suppose we have already processed blocks $1 , \ldots , t - 1$ with running statistics $m ^ { ( t - 1 ) }$ and $\ell ^ { \left( t - 1 \right) }$ . When a new block t with local scores $\mathbf { z } ^ { ( t ) }$ arrives, the update proceeds as follows:

$$
m _ { \mathrm { l o c a l } } ^ { ( t ) } = \operatorname* { m a x } _ { j } z _ { j } ^ { ( t ) } ,\tag{14}
$$

$$
m ^ { ( t ) } = \operatorname* { m a x } ( m ^ { ( t - 1 ) } , \ m _ { \mathrm { l o c a l } } ^ { ( t ) } ) ,\tag{15}
$$

$$
\ell ^ { ( t ) } = \ell ^ { ( t - 1 ) } \cdot \exp ( m ^ { ( t - 1 ) } - m ^ { ( t ) } ) + \sum _ { j } \exp ( z _ { j } ^ { ( t ) } - m ^ { ( t ) } ) .\tag{16}
$$

The key insight is that when a new maximum is encountered, all previously accumulated sums must be rescaled by $\exp ( m ^ { \ b ( t - 1 ) } - m ^ { ( t ) } )$ to remain on the same numerical basis. This rescaling is again justified by the shift-invariance of softmax.

Similarly, the partial attention output o is rescaled at each step:

$$
\mathbf { o } ^ { ( t ) } = \mathbf { o } ^ { ( t - 1 ) } \cdot \exp ( m ^ { ( t - 1 ) } - m ^ { ( t ) } ) + \sum _ { j } \exp ( z _ { j } ^ { ( t ) } - m ^ { ( t ) } ) \cdot \mathbf { v } _ { j } ^ { ( t ) } .\tag{17}
$$

After all blocks have been processed, the final output is normalized by the accumulated sum:

$$
\mathbf { o } _ { \mathrm { f i n a l } } = { \frac { \mathbf { o } ^ { ( T ) } } { \ell ^ { ( T ) } } } ,\tag{18}
$$

and the final LSE is computed as:

$$
\mathrm { L S E } = m ^ { ( T ) } + \log \ell ^ { ( T ) } .\tag{19}
$$

Why LSE Is Only Available After Completion. The running maximum $m ^ { ( t ) }$ may change as new blocks are processed. Each such change triggers a rescaling of all previously accumulated results. Therefore, the true global maximum and the true LSE are only known after the last block has been processed. Any block weight computed during the loop using intermediate statistics will be based on an incomplete normalization factor, and thus will not be precisely normalized.

This is exactly the challenge that GSRS’s two-phase correction addresses (Section 5.3): Phase 1 computes block weights using a cached LSE as an approximate reference during the kernel, and Phase 2 corrects them with the true LSE once the kernel completes.

Summary. Algorithm 1 summarizes the complete FlashAttention procedure described above.

Algorithm 1 FlashAttention with Online Softmax   
Require: Q, K, V: query, key, value matrices; B: block size; $N _ { b } = L / B \colon$ number of blocks   
Ensure: O: attention output; LSE: log-sum-exp values   
1: for each query block $u = 1 , \ldots , N _ { b }$ do   
2: Initialize $\mathbf { o } _ { i }  0 , \ell _ { i }  0 , m _ { i }  - \infty$ for each row $i \in$ Block u   
3: for each key-value block $v = 1 , \ldots , N _ { b }$ do   
4: $\mathbf { z } ^ { ( v ) } \gets \mathbf { \dot { Q } } _ { u } \mathbf { K } _ { v } ^ { \top } / \sqrt { d }$ ▷ Local scores   
5: m<sub>local</sub> ← max<sub>j</sub> z<sup>(v)</sup><sub>j</sub>   
6: $m _ { i } ^ { \mathrm { n e w } }  \mathrm { m a x } ( m _ { i } ^ { \cdot } , ~ m _ { \mathrm { l o c a l } } )$ ▷ Update maximum   
7: $\begin{array} { r } { \ell _ { i } \gets \ell _ { i } \cdot \exp ( m _ { i } - m _ { i } ^ { \mathrm { n e w } } ) + \sum _ { j } \exp ( z _ { j } ^ { ( v ) } - m _ { i } ^ { \mathrm { n e w } } ) } \end{array}$ ▷ Denominator   
8: $\begin{array} { r } { \mathbf { o } _ { i } \gets \mathbf { o } _ { i } \cdot \exp ( m _ { i } - m _ { i } ^ { \mathrm { n e w } } ) + \sum _ { j } \exp ( z _ { j } ^ { ( v ) } - m _ { i } ^ { \mathrm { n e w } } ) \cdot \mathbf { v } _ { j } ^ { ( v ) } } \end{array}$ ▷ Numerator   
9: $m _ { i } \gets m _ { i } ^ { \mathrm { n e w } }$   
10: end for   
11: $\mathbf { o } _ { i }  \mathbf { o } _ { i } / \ell _ { i }$ ▷ Final normalization   
12: $\mathrm { L S E } _ { i } \gets m _ { i } + \log \ell _ { i }$ ▷ Compute LSE   
13: end for   
14: return O, LSE

## A.2. Equivalence to the Block Importance Definition

We show that the two-phase evaluation recovers the block importance defined in Eq. (5). For query block u, let C denote the evaluated key blocks and $\kappa _ { c }$ their key tokens. This set contains all key blocks at the first search and the candidates defined by $S _ { 1 }$ at subsequent searches. For each query row $i \in u$ and key block $v \in { \mathcal { C } }$ , Phase 1 stores

$$
G _ { i v } = m _ { i v } + \log \sum _ { j \in v } \exp ( z _ { i j } - m _ { i v } ) , \qquad m _ { i v } = \operatorname* { m a x } _ { j \in v } z _ { i j } .\tag{20}
$$

Since $m _ { i v }$ is constant across keys within block v, it cancels when the expression is expanded:

$$
\begin{array} { l } { { \displaystyle G _ { i v } = m _ { i v } + \log \left( \exp ( - m _ { i v } ) \sum _ { j \in v } \exp ( z _ { i j } ) \right) } } \\ { { \displaystyle \quad = m _ { i v } - m _ { i v } + \log \sum _ { j \in v } \exp ( z _ { i j } ) } } \\ { { \displaystyle \quad = \log \sum _ { j \in v } \exp ( z _ { i j } ) . } } \end{array}\tag{21}
$$

Subtracting the block maximum before exponentiation provides numerical stability while preserving the block log-sum-exp. After all candidate blocks have been processed, FlashAttention provides the current row-wise LSE $\begin{array} { r l } { L _ { i } } & { { } = } \end{array}$ log $\begin{array} { r } { \sum _ { j \in { \mathcal { K } } _ { \mathcal { C } } } \exp ( z _ { i j } ) } \end{array}$ . Substituting the expression for $G _ { i v }$ into Phase $^ 2$ gives

$$
\begin{array} { l } { { \displaystyle W _ { u v } = \frac { 1 } { B } \sum _ { i \in u } \exp ( G _ { i v } - L _ { i } ) } } \\ { { \displaystyle \quad = \frac { 1 } { B } \sum _ { i \in u } \exp ( - L _ { i } ) \sum _ { j \in v } \exp ( z _ { i j } ) } } \\ { { \displaystyle \quad = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } \exp ( z _ { i j } - L _ { i } ) } } \\ { { \displaystyle \quad = \frac { 1 } { B } \sum _ { i \in u } \sum _ { j \in v } w _ { i j } } . } \end{array}\tag{22}
$$

This is the block importance in Eq. (5), with attention normalized over the evaluated key tokens. Consequently, $\begin{array} { r } { \sum _ { v \in \mathcal { C } } W _ { u v } = 1 } \end{array}$ providing normalized block weights for Top-p selection without cached LSE values.

## A.3. Fused Precise Block Importance Evaluation

Algorithm 2 presents the complete procedure of the fused block importance evaluation described in Section 5.3. Phase 1 is executed within the FlashAttention kernel, while Phase 2 is a lightweight post-kernel operation.

Algorithm 2 Fused Precise Block Importance Evaluation without Cached LSE   
Require: Q, K, V: query, key, value matrices; S: search candidate mask; B: block size   
Ensure: O: attention output; L: current row-wise LSE; W: block importance matrix   
1: Initialize $G _ { i v }  - \infty$ and $W _ { u v }  0$   
// Phase 1: In-kernel attention and block log-sum-exp   
2: for each query block u in parallel do   
3: Initialize $\mathbf { o } _ { i }  \mathbf { 0 } , \ell _ { i }  0 , m _ { i }  - \infty$ for $i \in u$   
4: for each key-value block $v \in S [ u ]$ do   
5: $\mathbf { Z } _ { u v }  \mathbf { \bar { Q } } _ { u } \mathbf { K } _ { v } ^ { \top } / \sqrt { d }$   
6: $\begin{array} { r } { m _ { i v }  \operatorname* { m a x } _ { j \in v } z _ { i j } , P _ { i j }  \exp ( z _ { i j } - m _ { i v } ) } \end{array}$   
7: $\begin{array} { r } { r _ { i v }  \sum _ { j \in v } P _ { i j } } \end{array}$   
8: $m _ { i } ^ { \prime } \gets \operatorname* { m a x } ( m _ { i } , m _ { i v } )$   
9: $\alpha _ { i } \gets \exp ( m _ { i } - m _ { i } ^ { \prime } ) , \beta _ { i } \gets \exp ( m _ { i v } - m _ { i } ^ { \prime } )$   
10: $\ell _ { i } \gets \alpha _ { i } \ell _ { i } + \beta _ { i } r _ { i v }$   
11: $\mathbf { o } _ { i }  \alpha _ { i } \mathbf { o } _ { i } + \beta _ { i } \sum _ { j \in v } P _ { i j } \mathbf { v } _ { j }$   
12: $m _ { i } \gets m _ { i } ^ { \prime }$   
13: Store $G _ { i v } \gets m _ { i v } + \log r _ { i v }$   
14: end for   
15: $L _ { i } \gets m _ { i } + \log \ell _ { i } , \mathbf { O } _ { i } \gets \mathbf { o } _ { i } / \ell _ { i }$   
16: Write $\mathbf { O } _ { u }$ and $L _ { i }$ for $i \in u$   
17: end for   
// Phase 2: Post-kernel block importance recovery   
18: for each query block u and candidate block $v \in S [ u ]$ do   
19: $\begin{array} { r } { W _ { u v } \stackrel { \cdot } {  } \frac { \mathrm { i } } { B } \sum _ { i \in u } \exp ( G _ { i v } - L _ { i } ) } \end{array}$   
20: end for   
21: return O, L, W

For fully masked block rows, we set $P _ { i j } = 0 , r _ { i v } = 0$ , and $G _ { i v } = - \infty ;$ the corresponding scale factor $\beta _ { i }$ is zero. The history scale $\alpha _ { i }$ is zero when no valid key has yet been processed. Rows with no valid keys return $\mathbf { O } _ { i } = \mathbf { 0 }$ and $L _ { i } = - \infty$ and contribute zero to block importance.

Memory Overhead. The primary memory overhead of GSRS comes from storing the row-level weight sums $\hat { S } _ { i v }$ during the fused evaluation kernel. For a sequence of length L with block size B and H attention heads, the storage requirement is $\begin{array} { r } { H \times \frac { L } { B } \times \frac { L } { B } \times B = H \times \frac { L ^ { 2 } } { B } } \end{array}$ elements. Compared to materializing the full attention matrix of size $H \times L \times L ,$ , this reduces memory by a factor of B. For example, with $L = 1 1 5 2 0 0 , B = 1 2 8$ , and $H = 2 4$ on HunyuanVideo, the storage is approximately 4.6 GB in bfloat16. As shown in Table 6, GSRS achieves slightly lower peak memory than full attention despite the additional storage for $\hat { S } _ { i v }$ , indicating that the memory overhead of block importance evaluation does not increase the overall memory footprint in practice.

Table 6. Peak GPU memory usage during inference on HunyuanVideo (single A800, 125 frames, 720p).
<table><tr><td>Configuration</td><td>Peak Memory (GB)</td></tr><tr><td>Full Attention</td><td>65.60</td></tr><tr><td>GSRS</td><td>62.65</td></tr></table>

Kernel Latency Analysis. Beyond memory efficiency, we evaluate the overhead of fused block importance evaluation using TileLang [46] implementations on H800 and A800 GPUs, as well as a CUDA implementation on A800. Figure 5 compares attention kernel latency with and without fused block statistics computation at varying density levels. Search steps perform both attention computation and block importance evaluation, while non-search steps perform attention computation only. For the CUDA implementation, we use the attention-only kernels from Block Sparse Attention [14] as the baseline. The smal latency gap across the tested implementations indicates that collecting block log-sum-exp statistics $G _ { i v }$ adds limited overhead to attention computation.

![](images/e5c655086b03a08910abbf78a7cce6a6a5f93429bdc06bf86501f6e8169fdd9f.jpg)

![](images/7050ce5ab0fd46b53c109d67b7cb3656e225e0fd74083730bb2cf11a43df98f1.jpg)

![](images/4bd1d2cde9e89025115c273a831a2059b01fcb5937612b02fa9684d9f0653aa1.jpg)  
Figure 5. Per-step attention kernel latency at varying density levels $( L = 1 1 5 2 0 0 , H = 2 4 , d = 1 2 8 )$

## A.4. Top-p Block Selection

Algorithm 3 describes the Top-p block selection procedure used to construct the sparse mask from the block importance matrix W.

Algorithm 3 Top-p Block Selection   
Require: W: block importance matrix of shape $[ H , N _ { q } , N _ { k } ] ; p \colon$ Top-p threshold   
Ensure: S: selected block mask   
1: for each head $h ,$ each query block u do   
2: Sort $\{ W _ { u v } \} _ { v = 1 } ^ { N _ { k } }$ in descending order, yielding permutation π   
3: Compute cumulative sum $\begin{array} { r } { C _ { v } = \sum _ { k = 1 } ^ { v } W _ { u , \pi ( k ) } } \end{array}$   
4: Find smallest $v ^ { * }$ such that $C _ { v ^ { * } } \geq p$   
5: $S [ h , u ] \gets \{ \pi ( 1 ) , \dots , \pi ( v ^ { * } ) \}$   
6: end for   
7: return S

## A.5. GSRS Pipeline

Algorithm 4 presents the complete GSRS inference pipeline, integrating warmup, fused evaluation, Growing Sparsity and Reduced Search, and mask reuse across denoising steps.

Algorithm 4 GSRS Pipeline   
Require: Per-step $\mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } ; N _ { w } \colon$ warmup steps; $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ : search steps with $t _ { 1 } \ge N _ { w } ; \{ p _ { 1 } , . . . , p _ { n } \}$ : Top-p   
thresholds; $B \colon$ block size   
Ensure: $\mathbf { O } _ { t } \colon$ attention output at each step   
1: $S _ { \mathrm { l a t e s t } }  S _ { \mathrm { f u l l } }$   
2: for each denoising step $t = 1 , \dots , T$ do   
3: if $t = t _ { 1 }$ then   
4: $\mathbf { O } _ { t } , \mathbf { \Lambda } _ { - } , \mathbf { W } \gets \operatorname { F u s e d E v a l } ( \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } , S _ { \operatorname { f u l l } } , B )$ ▷ Full attention and first search   
5: $S _ { 1 } \gets \mathrm { T o p } { \cdot } \mathrm { p } ( \mathbf { W } , p _ { 1 } )$   
6: $S _ { \mathrm { l a t e s t } }  S _ { 1 }$   
7: else if $t = t _ { k } \in \mathcal T$ for $k \geq 2$ then   
8: $\mathbf { O } _ { t } , _ { - } , \mathbf { W } \gets$ FusedE $\operatorname { a l } ( \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } , S _ { 1 } , B )$ ▷ Re-evaluate all candidates in $S _ { 1 }$   
9: $S _ { k } \gets \mathrm { T o p } { \cdot } \mathrm { p } ( \mathbf { W } , p _ { k } )$   
10: $\boldsymbol { S } _ { \mathrm { l a t e s t } }  \boldsymbol { S } _ { k }$   
11: else if $t \leq N _ { w }$ or $t < t _ { 1 }$ then   
12: $\mathbf { O } _ { t } \gets$ FullAttention $( \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } )$ ▷ Warmup and steps before the first search   
13: else   
14: $\mathbf { O } _ { t }$ ← SparseAttention $( \mathbf { Q } _ { t } , \mathbf { K } _ { t } , \mathbf { V } _ { t } , S _ { \mathrm { l a t e s t } } )$ ▷ Use the latest selected mask   
15: end if   
16: end for

The first search computes full attention and can coincide with the last warmup step. The current LSE is used within FusedEval to recover block importance and is not cached across denoising steps.

## A.6. Token Reordering via Hilbert Curve

Flattening a 3D video tensor into a 1D sequence in row-major order can separate spatiotemporally adjacent tokens, weakening locality within fixed-size blocks (Figure 6(a)). We therefore reorder tokens using the Generalized Hilbert curve [3], which preserves spatial-temporal locality and groups nearby tokens into compact blocks (Figure 6(b)). This helps block sparse attention capture local interactions.

Following prior work [3, 65, 67], we apply this reordering in all experiments.

![](images/949705c93588b334bf31090ac1972aaa208322ccac4da836311fb015fd2a37c5.jpg)  
(a) Standard

![](images/271516b82fa09ad02242dc78da9c1780b1a19b59948dddae4a12d390d8fec3b5.jpg)  
(b) Gilbert  
Figure 6. Token distribution within a single block under different ordering strategies. (a) Standard row-major ordering groups tokens from a flat spatial-temporal slab. (b) Gilbert curve ordering groups tokens from a compact spatial-temporal region, better preserving spatial-temporal adjacency.

## A.7. GSRS Runtime Breakdown

Figure 7 presents the runtime breakdown of GSRS on HunyuanVideo. Without reduced search, each search step evaluates all blocks in the attention matrix rather than restricting to the baseline mask. Evaluating all blocks not only increases the search time at each search step, but also retains more blocks due to the larger search space, leading to additional attention computation time (shown as Extra Attention in the figure). The search overhead itself (shown as Search) is dominated by intermediate matrix I/O, rescaling, and averaging, as the efficient kernel fusion described in Section 5.3 minimizes the computational cost. We define the total overhead of removing reduced search as the sum of Extra Attention and Search. As shown in the figure, reduced search lowers this total overhead by 13.75×, confirming that restricting searches to the baseline mask is essential for practical efficiency.

![](images/e6cdde1a607e97fc19c203bf4847c13cde1fbe0552c9fd3ad54a1905ad276041.jpg)  
Figure 7. Runtime breakdown on HunyuanVideo. Removing reduced search introduces substantial extra attention and search overhead. GSRS with reduced search achieves 13.75× lower overhead compared to the variant without it.

## B. Details about VBench Result

Table 7. Detailed VBench Quality dimensions. Full breakdown of VBench Quality Score across all 7 dimensions for Wan 2.1 and HunyuanVideo. Bold and underline denote the best and second best results.
<table><tr><td>Model</td><td>Method</td><td>Subject Consistency</td><td>Background Consistency</td><td>Temporal Flickering</td><td>Motion Smoothness</td><td>Aesthetic Quality</td><td>Imaging Quality</td><td>Dynamic Degree</td><td>VBench-Q</td></tr><tr><td rowspan="6">Wan 2.1</td><td>Full Attention</td><td>98.26%</td><td>98.25%</td><td>99.08%</td><td>99.21%</td><td>63.07%</td><td>73.29%</td><td>33.52%</td><td>83.75%</td></tr><tr><td>SpargeAttn [63]</td><td>94.88%</td><td>95.49%</td><td>99.07%</td><td>99.15%</td><td>61.73%</td><td>72.80%</td><td>32.95%</td><td>82.21%</td></tr><tr><td>SVG [51]</td><td>97.74%</td><td>97.90%</td><td>99.27%</td><td>99.43%</td><td>63.18%</td><td>71.99%</td><td>34.65%</td><td>83.69%</td></tr><tr><td>SVG 2 [58]</td><td>97.80%</td><td>97.47%</td><td>99.26%</td><td>99.38%</td><td>61.87%</td><td>72.23%</td><td>31.81%</td><td>83.19%</td></tr><tr><td>SVOO [33]</td><td>97.89%</td><td>97.76%</td><td>99.30%</td><td>99.40%</td><td>61.63%</td><td>72.42%</td><td>32.38%</td><td>83.33%</td></tr><tr><td>GSRS</td><td>98.21%</td><td>98.22%</td><td>99.07%</td><td>99.19%</td><td>63.09%</td><td>73.22%</td><td>33.52%</td><td>83.72%</td></tr><tr><td rowspan="6">HunyuanVideo</td><td>Full Attention</td><td>96.82%</td><td>97.07%</td><td>98.62%</td><td>99.51%</td><td>56.40%</td><td>70.71%</td><td>61.93%</td><td>83.98%</td></tr><tr><td>SpargeAttn [63]</td><td>96.55%</td><td>96.68%</td><td>98.52%</td><td>99.48%</td><td>56.25%</td><td>70.65%</td><td>63.06%</td><td>83.85%</td></tr><tr><td>SVG [51]</td><td>96.71%</td><td>96.85%</td><td>98.70%</td><td>99.52%</td><td>55.60%</td><td>68.98%</td><td>57.38%</td><td>83.21%</td></tr><tr><td>SVG 2 [58]</td><td>96.60%</td><td>96.05%</td><td>98.77%</td><td>99.49%</td><td>54.05%</td><td>66.50%</td><td>55.11%</td><td>82.24%</td></tr><tr><td>SVOO [33]</td><td>96.88%</td><td>96.39%</td><td>98.73%</td><td>99.51%</td><td>55.05%</td><td>69.19%</td><td>55.68%</td><td>82.97%</td></tr><tr><td>GSRS</td><td>96.66%</td><td>96.91%</td><td>98.58%</td><td>99.49%</td><td>56.51%</td><td>70.79%</td><td>63.06%</td><td>84.01%</td></tr></table>

## C. Effect of Warmup Length

Table 8 shows that increasing the number of warmup steps improves similarity to full attention at the cost of higher latency. We use $N _ { w } = 1 0$ to balance similarity and speedup. Meanwhile, VBench-Q remains between 83.71% and 84.01% across warmup settings (Table 9), indicating that generation quality is largely preserved even with $N _ { w } = 1$

Table 8. Ablation on the number of warmup steps $N _ { w }$ on HunyuanVideo.
<table><tr><td> $N _ { w }$ </td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Latency(s)↓</td></tr><tr><td>1</td><td>24.60</td><td>0.835</td><td>0.258</td><td>1680</td></tr><tr><td>3</td><td>27.63</td><td>0.899</td><td>0.203</td><td>1728</td></tr><tr><td>5</td><td>29.39</td><td>0.925</td><td>0.180</td><td>1746</td></tr><tr><td>8</td><td>31.00</td><td>0.944</td><td>0.161</td><td>1785</td></tr><tr><td>10</td><td>32.37</td><td>0.957</td><td>0.138</td><td>1825</td></tr><tr><td>15</td><td>33.79</td><td>0.966</td><td>0.134</td><td>1992</td></tr></table>

Table 9. Detailed VBench Quality dimensions under different warmup steps. Full breakdown of VBench Quality Score across all 7 dimensions on HunyuanVideo.
<table><tr><td>Warmup Steps</td><td>Subject Consistency</td><td>Background Consistency</td><td>Temporal Flickering</td><td>Motion Smoothness</td><td>Aesthetic Quality</td><td>Imaging Quality</td><td>Dynamic Degree</td><td>VBench-Q</td></tr><tr><td>1</td><td>96.64%</td><td>96.44%</td><td>98.55%</td><td>99.49%</td><td>56.02%</td><td>70.66%</td><td>63.06%</td><td>83.80%</td></tr><tr><td>3</td><td>96.66%</td><td>96.50%</td><td>98.59%</td><td>99.49%</td><td>55.89%</td><td>70.86%</td><td>61.36%</td><td>83.71%</td></tr><tr><td>5</td><td>96.68%</td><td>96.50%</td><td>98.60%</td><td>99.49%</td><td>56.03%</td><td>70.81%</td><td>61.93%</td><td>83.78%</td></tr><tr><td>8</td><td>96.66%</td><td>96.52%</td><td>98.61%</td><td>99.50%</td><td>55.94%</td><td>70.82%</td><td>62.50%</td><td>83.81%</td></tr><tr><td>10</td><td>96.66%</td><td>96.91%</td><td>98.58%</td><td>99.49%</td><td>56.51%</td><td>70.79%</td><td>63.06%</td><td>84.01%</td></tr><tr><td>15</td><td>96.67%</td><td>96.49%</td><td>98.62%</td><td>99.50%</td><td>55.89%</td><td>70.78%</td><td>62.50%</td><td>83.80%</td></tr></table>

## D. PSNR Distribution

Figure 8 compares PSNR distributions across VBench prompts on Wan2.1 and HunyuanVideo. GSRS achieves the highest median PSNR on both models, consistent with the average scores in Table 1. Its lower and upper quartiles are also higher than those of all baselines, showing improved similarity to full attention across diverse prompts.

![](images/d93d885884b71e78713c14ccd1f2a711ea530a80b87d7e0cb44c9a1a5b904de2.jpg)  
(a) PSNR distribution on Wan 2.1

![](images/7ad55b98e5a1a618958c2feda931886d60435cbb18ede7f94016ba75210e05e2.jpg)  
(b) PSNR distribution on HunyuanVideo  
Figure 8. PSNR distribution across VBench prompts. GSRS achieves the highest median PSNR on both Wan2.1 and HunyuanVideo.

## E. Performance Across Random Seeds

We evaluate GSRS on HunyuanVideo using five random seeds and report the mean and standard deviation of PSNR, SSIM, LPIPS, and VBench-Q. PSNR, SSIM, and LPIPS are measured against full-attention outputs generated with the same seed.

Table 10. GSRS performance across five random seeds on HunyuanVideo. Values are reported as mean ± standard deviation.
<table><tr><td>Model</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td> $\mathrm { \Delta V B e n c h - Q \uparrow }$ </td></tr><tr><td>HunyuanVideo</td><td> $3 1 . 3 5 \pm 0 . 9 7$ </td><td> $0 . 9 5 1 9 \pm 0 . 0 0 6 7$ </td><td> $0 . 1 4 7 \pm 0 . 0 1 1$ </td><td> $8 4 . 4 3 \pm 0 . 6 5$ </td></tr></table>

## F. Quality-Efficiency Trade-off

We evaluate the quality-efficiency trade-off of GSRS on HunyuanVideo by varying the Top-p thresholds across search steps.   
Table 11 reports PSNR, SSIM, LPIPS, and VBench-Q alongside inference latency and speedup.

Table 11. GSRS performance under different Top-p configurations on HunyuanVideo.
<table><tr><td>Top-p configuration</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>VBench-Q↑</td><td>Latency↓</td><td>Speedup↑</td></tr><tr><td>{0.95, 0.95, 0.95, 0.95}</td><td>32.55</td><td>0.959</td><td>0.136</td><td>84.89</td><td>1969s</td><td>1.55×</td></tr><tr><td>{0.93, 0.93, 0.93, 0.93}</td><td>31.20</td><td>0.947</td><td>0.153</td><td>84.90</td><td>1852s</td><td>1.65×</td></tr><tr><td>{0.90, 0.90, 0.90, 0.90}</td><td>30.28</td><td>0.937</td><td>0.166</td><td>84.89</td><td>1774s</td><td>1.72×</td></tr><tr><td>{0.88, 0.88, 0.88, 0.88}</td><td>29.63</td><td>0.930</td><td>0.170</td><td>84.95</td><td>1749s</td><td>1.75×</td></tr><tr><td>{0.95, 0.80, 0.85, 0.90}</td><td>31.64</td><td>0.951</td><td>0.143</td><td>84.98</td><td>1850s</td><td>1.65×</td></tr><tr><td>{0.90, 0.90, 0.95, 0.95}</td><td>30.40</td><td>0.939</td><td>0.163</td><td>84.87</td><td>1792s</td><td>1.70×</td></tr><tr><td>{0.95, 0.95, 0.90, 0.90}</td><td>32.38</td><td>0.957</td><td>0.141</td><td>84.85</td><td>1907s</td><td>1.60×</td></tr><tr><td>{0.90, 0.85, 0.80, 0.75}</td><td>30.00</td><td>0.933</td><td>0.174</td><td>84.85</td><td>1719s</td><td>1.78×</td></tr><tr><td>{0.85, 0.80, 0.75, 0.70}</td><td>28.95</td><td>0.920</td><td>0.190</td><td>84.78</td><td>1631s</td><td>1.87×</td></tr><tr><td>{0.95, 0.90, 0.85, 0.80}</td><td>31.96</td><td>0.953</td><td>0.141</td><td>84.99</td><td>1825s</td><td>1.68×</td></tr></table>

## G. Inference Speed on NVIDIA H800

We compare the inference latency and speedup of GSRS and compared sparse attention methods on a single NVIDIA H800 GPU. Speedup is measured relative to full attention on the same GPU.

Table 12. Inference latency and speedup on a single NVIDIA H800 GPU.
<table><tr><td rowspan="2">Method</td><td colspan="2">Wan2.1</td><td colspan="2">HunyuanVideo</td></tr><tr><td>Latency ↓</td><td>Speedup↑</td><td>Latency ↓</td><td>Speedup↑</td></tr><tr><td>Full Attention</td><td>2573s</td><td>1.00×</td><td>2265s</td><td>1.00×</td></tr><tr><td>SpargeAttn</td><td>1934s</td><td>1.33×</td><td>1497s</td><td>1.51×</td></tr><tr><td>SVG</td><td>1796s</td><td>1.43×</td><td>1319s</td><td>1.72×</td></tr><tr><td>SVG2</td><td>1735s</td><td>1.48×</td><td>1168s</td><td>1.94×</td></tr><tr><td>SVOO</td><td>1759s</td><td>1.46×</td><td>1143s</td><td>1.98×</td></tr><tr><td>GSRS</td><td>1443s</td><td>1.78×</td><td>1026s</td><td>2.21×</td></tr></table>

## H. Limitations

GSRS has two limitations that warrant future investigation. First, at least one full-attention warmup step is required to obtain the initial LSE for subsequent block importance evaluation. Although the warmup overhead is small relative to the total inference time, eliminating this requirement would further improve efficiency. Second, GSRS is designed and evaluated specifically for DiT-based video generation models. Whether the observed attention dynamics and the proposed mechanisms generalize to other architectures such as large language models remains an open question for future work.

## I. Broader Impacts

This work improves the efficiency of video diffusion models, reducing computational cost and energy consumption for high-quality video generation. By lowering the hardware requirements, GSRS makes video generation more accessible to researchers and creators with limited computational resources, potentially democratizing access to this technology.

However, improved efficiency may also lower the barrier to generating realistic synthetic videos at scale, which could be misused for producing misleading or deceptive content such as deepfakes. The ability to generate videos faster and cheaper amplifies both beneficial and harmful applications. We encourage responsible deployment of this technology and recommend combining video generation systems with safeguards such as content watermarking and synthetic media detection to mitigate potential misuse.

## J. Detailed Comparison with AdaSpa and LiteAttention

Search and block selection. AdaSpa [53] uses a fixed sparsity target and evaluates all blocks at scheduled search steps, whereas GSRS progressively lowers Top-p thresholds and restricts subsequent searches to the early candidate set $S _ { 1 }$ . Once LiteAttention [41] skips a block, it excludes that block from evaluation and computation for the remaining denoising steps. GSRS re-evaluates all blocks within $S _ { 1 }$ at each search step, allowing previously omitted blocks within $S _ { 1 }$ to be selected again as attention changes. This design preserves the ability to dynamically select blocks while reducing search overhead compared with evaluating all blocks at each search step.

Block importance estimation. GSRS computes block importance from block log-sum-exp statistics and the current rowwise LSE, without relying on cached LSE values. AdaSpa uses cached LSE values to estimate block importance, while LiteAttention applies a local tile-skipping condition based on the running softmax statistics. The distinction between cached and current-step normalization affects both cumulative probability thresholds and block rankings.

Let u and v denote a query block and a key block, respectively, $B = | u |$ the number of query rows, and $z _ { i j }$ the pre-softmax attention score. For a fixed query block, let C denote the evaluated key blocks and $\kappa _ { c }$ their key tokens. The current rowwise normalization factor is $\begin{array} { r } { L _ { i } = \log \sum _ { j \in \mathcal { K } _ { c } } \exp ( z _ { i j } ) } \end{array}$ . The normalized attention weight assigned to block $v \in { \mathcal { C } }$ and its query-block average are

$$
S _ { i v } = \sum _ { j \in v } \exp ( z _ { i j } - L _ { i } ) , \qquad W _ { u v } = \frac { 1 } { B } \sum _ { i \in u } S _ { i v } , \qquad \sum _ { v \in \mathcal { C } } S _ { i v } = 1 .\tag{23}
$$

At subsequent search steps, AdaSpa reuses row-wise LSE values cached from an earlier step to estimate current block importance. Denoting these cached values by $L _ { i } ^ { \mathrm { o l d } }$ , the resulting per-row block scores are

$$
\widetilde { S } _ { i v } = \sum _ { j \in v } \exp ( z _ { i j } - L _ { i } ^ { \mathrm { o l d } } ) = \alpha _ { i } S _ { i v } , \qquad \alpha _ { i } = \exp ( L _ { i } - L _ { i } ^ { \mathrm { o l d } } ) .\tag{24}
$$

Consequently, $\textstyle \sum _ { v \in { \mathcal { C } } } { \widetilde { S } } _ { i v } = \alpha _ { i }$ , which need not equal one. If such uncorrected scores are used for Top-p selection, the cumulative threshold can be reached too early or too late. AdaSpa aggregates these scores over query rows to obtain block importance. Because $\alpha _ { i }$ varies across rows, this aggregation introduces query-dependent reweighting:

$$
\widetilde { W } _ { u v } = \frac { 1 } { B } \sum _ { i \in u } \alpha _ { i } S _ { i v } , \qquad \widetilde { W } _ { u v } - \widetilde { W } _ { u w } = \frac { 1 } { B } \sum _ { i \in u } \alpha _ { i } ( S _ { i v } - S _ { i w } ) .\tag{25}
$$

For two candidate blocks v and $w ,$ this difference can have a different sign from $W _ { u v } \mathrm { ~ - ~ } W _ { u w }$ , changing their ranking and potentially affecting Top-k selection as well. Normalizing the aggregated scores by a single scalar does not undo this query-dependent reweighting.

GSRS directly computes block importance using block log-sum-exp statistics $G _ { i v }$ and the current row-wise LSE $L _ { i }$

$$
W _ { u v } = \frac { 1 } { B } \sum _ { i \in u } \exp ( G _ { i v } - L _ { i } ) .\tag{26}
$$

This yields normalized block importance within the search candidate set without requiring cached LSE values. The equivalence to the block importance definition is derived in Appendix $\mathrm { A } . 2$

LiteAttention compares each block’s row-wise maximum attention scores with the running row-wise maxima maintained during online softmax. A block is marked for skipping when its local maximum falls below the running maximum by a specified margin for every query row. This criterion uses intermediate softmax statistics and depends on block processing order, which may lead to less accurate block selection

Experimental comparison. Table 13 compares GSRS with related sparse attention methods on HunyuanVideo. To match AdaSpa without head-adaptive allocation, we replace Top-p selection in GSRS with fixed Top-k selection using a uniform 20% keep ratio for every head. This configuration removes differences in Top-p selection and head-adaptive sparsity allocation from the comparison. We also implement an Ampere-compatible LiteAttention kernel in CUDA, preserving its original skipping criterion and cross-step mask update mechanism.

Table 13. Comparison with related sparse attention methods on HunyuanVideo. GSRS (AdaSpa aligned) and AdaSpa (w/o Head Adaptive) both use a uniform 20% keep ratio.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>Speedup↑</td></tr><tr><td>MInference</td><td>22.53</td><td>0.7435</td><td>1.27×</td></tr><tr><td>XAttention</td><td>29.08</td><td>0.9099</td><td>1.36×</td></tr><tr><td>LiteAttention</td><td>21.85</td><td>0.7036</td><td>1.33×</td></tr><tr><td>AdaSpa (w/o Head Adaptive)</td><td>28.51</td><td>0.8825</td><td>1.76×</td></tr><tr><td>AdaSpa</td><td>29.07</td><td>0.8905</td><td>1.78×</td></tr><tr><td>GSRS (AdaSpa aligned)</td><td>29.66</td><td>0.9330</td><td>1.86×</td></tr></table>

Under the matched keep ratio, GSRS improves PSNR and SSIM over AdaSpa without head-adaptive allocation while achieving a higher speedup. These results show that the gains are not solely attributable to Top-p selection or head-adaptive sparsity allocation. GSRS also outperforms our Ampere implementation of LiteAttention in PSNR, SSIM, and speedup. These results support the effectiveness of combining Reduced Search with precise block importance evaluation to reduce search overhead while preserving accurate block selection.

## K. More Details about Attention Entropy and Mean Attention Distance

To characterize the attention dynamics without incurring the $O ( L ^ { 2 } )$ memory overhead of materializing full attention maps, we fuse the computation of attention entropy $H ( i )$ and mean attention distance $D ( i )$ directly into our FlashAttention-based kernels. This section provides the formal derivation of the efficient calculation used in our implementation.

## K.1. Efficient Derivation of Attention Entropy

The attention entropy for a specific query i is defined based on the softmax-normalized weights $w _ { i j } \colon$

$$
H ( i ) = - \sum _ { j = 1 } ^ { n } w _ { i j } \log _ { 2 } w _ { i j }\tag{27}
$$

In a hardware-efficient FlashAttention implementation, the weight $w _ { i j }$ is typically computed using the raw attention score $s _ { i j }$ (e.g., the scaled dot-product $Q K ^ { T } / \sqrt { d } )$ and the Log-Sum-Exp constant $\mathrm { L S E } _ { i }$

$$
w _ { i j } = 2 ^ { s _ { i j } - \mathrm { L S E } _ { i } }\tag{28}
$$

Substituting this into the entropy formula, we can transform it into a form that avoids explicit exponentiation of every element during the final reduction:

$$
H ( i ) = - \sum _ { j = 1 } ^ { n } w _ { i j } ( s _ { i j } - \mathrm { L S E } _ { i } )\tag{29}
$$

Given the normalization property $\textstyle \sum _ { j = 1 } ^ { n } w _ { i j } = 1$ , the expression simplifies to:

$$
H ( i ) = \mathrm { L S E } _ { i } - \mathbb { E } [ s _ { i } ]\tag{30}
$$

where $\begin{array} { r } { \mathbb { E } [ s _ { i } ] = \sum _ { j } w _ { i j } s _ { i j } } \end{array}$ represents the expected value of the raw attention scores for query i. This formulation allows us to compute entropy by maintaining a single running scalar $\sum w _ { i j } s _ { i j }$ during the block-wise tiling process.

## K.2. Fused Accumulation Logic

Consistent with the definition in the Section 4, the Mean Attention Distance $D ( i )$ represents the expected relative distance between the query i and its attended keys $j \colon$

$$
D ( i ) = \sum _ { j } w _ { i j } \cdot | i - j |\tag{31}
$$

In our Triton-based [47] implementation, we fuse the calculation of $H ( i )$ and $D ( i )$ into the attention kernel by maintaining two running accumulators, acc\_s and acc $\underline { { \boldsymbol { d } } } \mathrm { : \Omega }$

1. Accumulation: For each block of keys, we compute acc $\_ s \gets \sum w _ { i j } s _ { i j }$ and acc $\begin{array} { r } { \mathbf { \mathcal { d } }  \sum w _ { i j } | i - j | } \end{array}$

2. Online Rescaling: Since the softmax local maximum $m _ { i }$ and the LSE update during the kernel’s execution, we rescale $a c c \_ s$ and acc\_d at each step to ensure numerical consistency, identical to the standard output O rescaling in FlashAttention.

3. Finalization: After the final reduction, $H ( i )$ is obtained by subtracting the accumulated acc\_s from the final LSE<sub>i</sub>.

This approach ensures that capturing these diagnostics adds negligible computational overhead and zero additional memory footprint beyond the storage of the resulting metrics themselves.

## K.3. Attention Dynamics Across Prompts

To verify that the attention trends reported in Section 4 are not artifacts of specific inputs, we visualize attention entropy and mean attention distance across a diverse set of test prompts for both HunyuanVideo and Wan2.1 in Figure 9.

Despite substantial differences in generated content, all prompts exhibit the same monotonic decrease in both metrics as denoising progresses. The curves remain tightly bundled, with only minor variation in absolute values between prompts. This confirms that the progressive concentration of attention is an intrinsic property of the model architecture rather than a content-dependent phenomenon.

![](images/05cd0913383b0ce7a4a7690aaefdc3593db0595a9480c030bbf45b64ba598bf0.jpg)  
(a) Hunyuan: Double-stream [Entropy]

![](images/a8011451920128aa2ecac2cda35a2b874192dd84520fc68f84fd84d293266e1f.jpg)  
(b) Hunyuan: Double-stream [MAD]

![](images/0a485b3fc7d45c2c3e3498331df92340b52c4eb8579270474996bdaaa184b4c9.jpg)  
(c) Hunyuan: Single-stream [Entropy]

![](images/a14de6555d0ecc7ac836c527dbef4a492a5f92b33fdb3212751b9b06db88d8a6.jpg)  
(d) Hunyuan: Single-stream [MAD]

![](images/34ad6b8331087c14063c922cec4fa80d2760bdfdd6d893fa2772394aa38655eb.jpg)

![](images/460eab311ee37dfe250f2eb6c389acf40360c2a1f770f2d9bcbf5e6af1cc3994.jpg)  
(f) Wan2.1: Conditional [MAD]

(e) Wan2.1: Conditional [Entropy]  
![](images/65c5c925baec129048c110ba6dba82c17881886be982eb098c7d176760e0cece.jpg)  
(g) Wan2.1: Unconditional [Entropy]

![](images/2b261a8e5097ffc854e664820bf3b45ed350e694672ff0dc9677c0b856fbfe16.jpg)  
(h) Wan2.1: Unconditional [MAD]

Figure 9. Attention entropy and MAD profiles across diverse prompts for HunyuanVideo and Wan2.1. Each line represents a different test prompt. Both metrics exhibit a consistent monotonic decrease across all prompts, confirming that the attention concentration trend is an intrinsic model property independent of input content.

## K.4. Attention Dynamics Across Layers

Figure 10 presents layer-wise attention entropy and mean attention distance (MAD) heatmaps for HunyuanVideo and Wan2.1. Across all models, two consistent patterns emerge. First, both metrics decrease monotonically along the denoising trajectory within each layer, confirming that the progressive concentration observed at the model level (Section 4) holds at the individua layer level as well. Second, the variation across layers is substantial and persistent: certain layers maintain high entropy and long attention distance throughout denoising, while others exhibit consistently concentrated and localized attention. Notably, for Wan2.1, the conditional and unconditional branches display similar layer-wise patterns, suggesting that this variation is an architectural property rather than a conditioning artifact. These observations further support the need for layer-adaptive sparsity, as a uniform sparsity ratio would over-prune concentrated layers while under-pruning dispersed ones.

![](images/b3f3c93528a4c0939c71f02cd9ae4aebdf3c48ba9cd8d72a79449fb7057316b7.jpg)  
(a) Hunyuan [Entropy]

![](images/dd42cae14639c003d629b32856b475a6ddf9644847b0818cfaa4a8ebc19e48a9.jpg)  
(b) Hunyuan [MAD]

![](images/63c4840d3ad53f680b86598bf07ab79f974f16584e4e5ba3c20d968972fb3d81.jpg)  
(c) Wan2.1 (Conditional) [Entropy]

![](images/697bb6bf0fe58cc1b43e3a7cef736b40b7a8632dd3b1fb824e5d7a12982b15e0.jpg)  
(d) Wan2.1 (Conditional) [MAD]

![](images/3a7d2479fe4e6eacd937805f28bd5322dcadb42cbb8c74b4211ce589d76fab79.jpg)  
(e) Wan2.1 (Unconditional) [Entropy]

![](images/cd05e92026f0bc3cf26a84776f1d62ead3a942bc1fb59117007b4d882da5ecc3.jpg)  
(f) Wan2.1 (Unconditional) [MAD]

Figure 10. Layer-wise attention entropy and MAD across denoising steps. Both metrics decrease monotonically within each layer.   
Meanwhile, substantial variation exists across layers, confirming distinct attention concentration patterns at different depths.

## K.5. Attention Dynamics Across Heads

Across heads, we visualize attention entropy and mean attention distance (MAD) for representative layers in HunyuanVideo and Wan2.1 (conditional and unconditional branches). Each figure shows head-wise heatmaps over denoising steps.

Within each layer, different heads behave differently. Some heads keep more dispersed attention, while others stay more concentrated. Moreover, all heads show a similar decreasing trend as denoising progresses. The conditional and

unconditional branches of Wan2.1 also show very similar patterns, suggesting that these differences across heads are not caused by conditioning.

![](images/d275b82f60b47c4886b36820f32d10bdc8360d1502e9e1da44bb07325b9e0711.jpg)

![](images/8cc76c9c20b598fb7b0bf12ab49ebe414125ac4bb97e57243992dcc0b79fd34d.jpg)

![](images/aa5f8533a353f799c1a604194ca29cca8a1f422ef1e5d45c03824f3e9401cbcf.jpg)

![](images/4ee6c5bd041fbb097a293afa1d16f2953804eadc349bd8cce9037f0764472ce4.jpg)

![](images/6cc4b8c142f0a40e169fbaccf33b8f4c9a140fd7233f039cdf58604ec5e1d350.jpg)

![](images/6036c10cacbb5a1039990f82eda2e258856586562d2f5979cce7191440c02636.jpg)  
Figure 11. Head-level attention entropy and MAD in HunyuanVideo. Each row corresponds to one layer, showing entropy (left) and MAD (right) for all heads. While different heads exhibit diverse behaviors, all follow a consistent decreasing trend across denoising steps.

![](images/63a582655aa38b0cfbabefcaa55838ba614ba1c49dbef821b6d4482bd2aff2e2.jpg)  
Layer9 Entropy

![](images/c2f7486ff0fe11374cacdc10f5028ec1280585433d1ea6de338d315426f3533b.jpg)  
Layer9 MAD

![](images/7da6d25f079b8d3b6401486b858156e2567d801cd6c703b52d976429c6c64bc4.jpg)  
Layer10 Entropy

![](images/223d994ff4b26fca5ec15e09242fae2c60c2de4265ce032ac4444caaa90ac15e.jpg)  
Layer10 MAD

![](images/686239abacfc6dd2bed93acfcd9b47a6e983d30bea1c3dec67a06a75c6ea9309.jpg)  
Layer15 Entropy

![](images/545d5ba9a8a016035a8eba8fca503b4cc2b6b71b5eeb5acb4818fe2b9dcabbfc.jpg)  
Layer15 MAD  
Figure 12. Head-level attention entropy and MAD in Wan2.1 (conditional branch). Significant variation exists across heads within each layer, while the overall monotonic decrease over denoising is preserved.

![](images/385190ff27a1d46848a7993ea000a2d0e6351f4288d14f1ce0a1db1a18e6ffc9.jpg)  
Layer9 Entropy

![](images/c1bcf8258b16a03cb08e2550ff25eaac0afd2c49a088fe9e89db0867f682bc46.jpg)  
Layer9 MAD

![](images/7e60ff440080f9dd6efefc18a6c88ed1d13a4af48352400e0116b9975a4d9bc2.jpg)  
Layer10 Entropy

![](images/94a8d49e0b266cd598a8442bb64215c68da88d1dafeac5f6658ae8fd6929ec79.jpg)  
Layer10 MAD

![](images/db58d2c6814257c65c0a165e70bc774250a59f47c97807b074048c57bd2e56e5.jpg)  
Layer15 Entropy

![](images/0a9c95611975eeeb45b109130f286403c1155c61cdd2c4c7bb274d9c6574369b.jpg)  
Layer15 MAD  
Figure 13. Head-level attention entropy and MAD in Wan2.1 (unconditional branch). The patterns are highly consistent with the conditional branch, suggesting that head-wise variation is largely independent of conditioning.

## L. Visualization of the generated videos

We provide visualization comparison between Full Attention and GSRS on Wan 2.1 and HunyuanVideo. Results in Figure 14 and Figure 15 demonstrate that GSRS can preserve high pixel-level fidelity, achieving similar generation quality compared with the full attention. More video samples are provided in the supplementary materials.

![](images/17ac2e350f81a08d52219cb998dcbf0659a862c9b739c8e2453b6a2d349505c0.jpg)  
Figure 14. Comparion of Full Attention and GSRS on Wan 2.1 Text-to-Video generation.

![](images/1d9867d3b7f8343bda6a1c5c2e2f8666a86c88e308b42e47f53866157af8f3ac.jpg)  
Figure 15. Comparion of Full Attention and GSRS on HunyuanVideo Text-to-Video generation.