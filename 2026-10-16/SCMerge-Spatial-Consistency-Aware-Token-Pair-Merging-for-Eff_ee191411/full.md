# SCMerge: Spatial-Consistency-Aware Token Pair Merging for Efficient Semantic Segmentation

Junfeng Ren<sup>1</sup> Yi Gong<sup>1</sup> Yuan Zeng<sup>2</sup>

<sup>1</sup>Southern University of Science and Technology

<sup>2</sup>Shenzhen Technology University

junfengren3253@gmail.com

## Abstract

Vision Transformer architectures have showcased impressive performance on semantic segmentation, yet their quadratic attention complexity hinders practical applications. Recent works have focused on merging tokens to reduce sequence length and thus improve transformer efficiency. However, existing token merging methods primarily rely on feature similarity to identify similar tokens, overlooking spatial consistency, which may introduce segmentation errors in semantic segmentation. In this work, we propose SCMerge, a training-free token merging method thatjointly considersfeature similarity and spatial consistency in pairing similar tokens. The pairwise tokens are then merged within the encoder, and the resulting token sequences are reconstructed onto the original patch grid after decoding. The proposed method is flexible and can be plugged into existing vision Transformer-based semantic segmentation pipelines. Extensive experiments on the benchmark datasets ADE20K, Pascal Context, and Cityscapes show that SCMerge consistently achieves a better trade-offbetween segmentation accuracy and efficiency than baseline methods.

## 1. Introduction

Semantic segmentation has made significant progress with the recent introduction of Vision Transformers (ViTs) [8, 24, 29, 37]. However, ViTs with self-attention suffer from quadratic computational complexity and long latency, limiting their application to high-resolution images. A primary avenue for addressing this issue is to reduce token sequence length in self-attention and streamline computation [1, 17, 23, 25, 28, 30, 36]. Although most existing token reduction methods work well for image classification, they often struggle in dense prediction tasks, such as semantic segmentation. The main challenge is to reduce the token sequence length while preserving the spatial and semantic details required by

the segmentation decoder.

This work focuses on ViT-based semantic segmentation and aims to present a training-free token merging method to improve the efficiency of existing pipelines. Existing training-free token reduction methods have shown that merging similar tokens can effectively reduce the token sequence length and computational complexity. ToMe performs bipartite matching in feature space and merges similar tokens via averaging [1]. Mutual Pair Merging (MPM) employs mutual nearest-neighbor matching and combines the matched tokens, yielding computation and latency improvements without modifying segmentation decoders [26]. In addition, ALGM employs a segmentation-specific local-to-global merging schedule that combines local and global token matching [22]. Despite their potential, existing methods mainly match tokens based on feature similarity, which can cause segmentation errors when local feature content is similar but the tokens are spatially or globally different. Tokens near semantic boundaries may be incorrectly paired with interior tokens from neighboring regions, leading to boundary blurring. In addition, interior tokens from different objects can have similar features and be mistakenly identified as redundant. More recently, several works [13, 19] have proposed to address this issue by introducing spatial information or learned policies, but at the cost of additional inputs or training.

In this paper, we propose SCMerge, a training-free token merging method to reduce the token length in ViTbased semantic segmentation pipelines, aiming to achieve a better trade-off between segmentation accuracy and computational cost. To guide the merging process, We introduce a token-pairing criterion that jointly considers local feature similarity and spatial consistency. By merging only tokens that are similar in both feature and spatial consistency, this mechanism improves the efficiency–accuracy trade-off. Specifically, we first compute the cosine similarity between all token pairs. A local variance measure is then used to identify tokens that are near semantic boundaries or lie in a smooth interior region. For each candidate pair, we compute a spatialconsistency penalty that discourages merging of locally similar but boundary-sensitive tokens, as well as tokens belonging to different objects. Finally, paired tokens are merged via feature averaging, and the reduced token sequence is passed through the segmentation decoder before being mapped back to the original token grid. This design refines the token pairing process to preserve semantic boundaries and neighborhood spatial consistency, improving segmentation efficiency.

Our SCMerge does not require additional supervision, extra input modalities, or modifications to the segmentation head, allowing it to be plugged into ViT-based semantic segmentation pipelines. We evaluate the effectiveness of the proposed token merging module using ViT-based segmentation models [29] on three semantic segmentation datasets, including ADE20K [38], Pascal Context [21], and Cityscapes [5]. Experimental results demonstrate that plugging our SCMerge module into ViT-based semantic segmentation models improves the trade-off between computational complexity and segmentation accuracy. Our contributions can be summarized as follows:

• We propose a novel training-free token-pair merging method that incorporates a spatial-consistency penalty into feature-similarity-based matching. This method prevents the merging of locally similar but spatially incompatible tokens and improving the accuracy–efficiency trade-off.

• We design a local feature variation metric to identify boundary-sensitive tokens and introduce a spatialconsistency penalty, comprising a boundary-sensitive term and a neighborhood consistency term, to reduce incorrect merging near semantic boundaries and preserve fine spatial detail.

• Extensive experiments demonstrate that our method can be integrated into ViT-based semantic segmentation models to achieve a better accuracy-efficiency trade-off than baseline methods relying only on feature similarity.

## 2. Related Work

Efficient ViT Segmentation. Vision Transformerbased segmentation models, including SETR [37], Segmenter [29], DPT [24], and Mask2Former [4], have achieved promising performance in dense prediction tasks, but they suffer from high computational costs when processing high-resolution images. A possible solution is to redesign the architecture or attention mechanism. SegFormer [34] adopts a hierarchical Transformer encoder with a lightweight all-MLP decoder. Swin Transformer [18] restricts self-attention to shifted local windows, and PVT [32] progressively reduces spatial resolution to build pyramid features. FlashAttention [6] accelerates attention by reducing memory input and output between GPU high-bandwidth memory and on-chip memory through tiled computation. This work focuses on reducing redundant tokens inside a pretrained ViT encoder , preserving the compact representation through decoding, and reconstructing the dense prediction grid without changing the segmentation head.

Token Reduction in Vision Transformers. Many recent methods accelerate ViTs by reducing the token sequences passed to subsequent layers. DynamicViT [25] learns to prune uninformative tokens dynamically, while TokenLearner [28] learns a compact set of informative tokens. A-ViT [36] introduces adaptive token halting, and $\mathrm { { I A } { - } \mathrm { { R E D } ^ { 2 } } }$ [23] reduces redundancy with interpretability-aware token selection. Patch Slimming [30] and SPViT [15] study efficient token pruning, while ATS [9] samples tokens adaptively. Token Pooling [20], STViT [12], HeatViT [7], and DiffRate [3] further explore pooling, sampling, pruning, or learnable compression rates. These methods either target image classification or require learned policies, retraining, or architecture-specific modifications. Unlike classification tasks, preserving thin structures and fine boundaries is critical in dense prediction tasks. This depends on having a sufficient number of tokens for accurate reconstruction and precise predictions. This work introduces a training-free token merging method designed for dense prediction.

Training-free Token Merging. Token merging has emerged as an effective technique for improving the efficiency of ViTs. By combining similar tokens rather than discarding them, token merging methods substantially reduce computational complexity and latency in image classification and semantic segmentation. ToMe [1] is a training-free token merging method that matches featuresimilar tokens and averages them progressively inside the encoder. Several works have extended token merging with learned or adaptive mechanisms, including learned patch merging [27], efficient token merging [10], delayed spatial merging [11], and learned pruning-and-merging thresholds [2]. In addition, fusion-based or saliencydriven methods, such as token fusion [14], multi-criteria fusion [16], spectrum-preserving merging [31], learnable token merging [33], and saliency-driven merging [35] have been studied for token merging. Mutual Pair Merging (MPM) [26] forms mutual nearest-neighbor token pairs and preserves a reconstruction map before the dense prediction decoder. These methods can significantly improve ViT efficiency by averaging similar tokens, but their token grouping strategies mainly rely on feature similarity, which neglects spatial consistency when measuring token similarity. Our SCMerge addresses this limitation by using local feature variation to pair tokens in a more spatial-aware manner.

Spatial-aware Token Merging for Semantic Segmentation. Several recent methods have incorporated taskspecific or spatial information to improve token merging for semantic segmentation. CTS [19] learns a content-aware token-sharing policy to preserve semantic structures. ALGM [22] designs an adaptive local-toglobal merging schedule for segmentation backbones. ToSA [13] introduces spatial awareness into token merging so that merging decisions consider both visual feature similarity and spatial information. These methods demonstrate the importance of spatial information for semantic segmentation, but they often rely on learned policies, specialized schedules, additional spatial information, or method-specific implementations. SCMerge follows the same motivation of spatial-aware token merging, but it is training-free and can be plugged into existing ViT-based semantic segmentation pipelines.

## 3. Method

As illustrated in Figure 1, SCMerge is a training-free module that can be inserted into multiple blocks of a ViT encoder to reduce the token sequence length from N to $M \ ( M \ < \ N )$ , while maintaining a merge map for reconstructing compact segmentation head outputs onto the original token grid after decoding. Let $\mathbf { \bar { \Phi } } _ { X } = [ x _ { 1 } , \dots , x _ { N } ] \in \mathbb { R } ^ { N \times d }$ denote the image tokens at a merge layer, where $N$ is the number of input tokens and d is the feature dimension. Each token $x _ { i }$ is indexed by a representative coordinate $c _ { i }$ for local neighborhood statistics, while a merge map tracks reconstruction assignments across merging stages. At each insertion, SCMerge first computes a feature-similarity matrix $S$ to measure content similarity between tokens. It then calculates local context variation within a $3 \times 3$ token neighborhood and uses this variation to assign a spatial-consistency penalty score when forming matching pairs. The matched token pairs are merged into a reduced sequence $Y = [ y _ { 1 } , \dotsc , y _ { M } ] \in \mathbb { R } ^ { \tilde { M } \times d }$ The merged tokens are then propagated through the remaining backbone layers, while the corresponding merge map is stored. The merged tokens are then fed into the decoder and segmentation head of the original ViT-based model to produce compact prediction logits. Finally, the merge map is used to reconstruct these compact logits onto the final semantic segmentation results.

## 3.1. Preliminaries

We follow existing semantic segmentation models [29] and use a standard Vision Transformer (ViT) encoder [8] and Mask Transformer decoder as our backbone. Given an input image of size $H \times W$ and patch size $P ,$ , the image is divided into $N _ { 0 } = H / P \times W / P$ non-overlapping patches. Each patch is linearly projected into a ddimensional embedding, which is added to a learned positional embedding, resulting in the initial token sequence $X _ { 0 } ~ \in ~ \mathbb { R } ^ { N _ { 0 } \times d }$ . These tokens are then processed by a stack of Transformer blocks consisting of self-attention and feed-forward layers.

The decoder then jointly processes the tokens from the encoder together with $C$ learnable class embeddings, where $C$ denotes the number of semantic classes. After the Transformer decoder layers, the output image and class embeddings are projected and normalized, and the pairwise inner products between the image embeddings and the class embeddings produce patch-level class logits. In the original full-token model, these logits correspond to the $N _ { 0 }$ patch locations and are reshaped into an $H / P \times$ $W / P$ patch grid. They are then upsampled to the input resolution via bilinear interpolation.

SCMerge Insertion. SCMerge is inserted after selected ViT encoder blocks and operates on the current token sequence. At each insertion, SCMerge reduces the token sequence length from $N$ to $M ( M < N )$ . To support multiple merge layers, SCMerge maintains a representative coordinate for each token and a merge map. The coordinate indicates the token’s position in the sparse token layout used for variation computation, while the merge map records, for each original patch location, the token that represents it. The merge map is updated after each merge stage. The reduced token sequence is then propagated through the remaining encoder blocks and passed through the Mask Transformer decoder. The decoder architecture and parameters remain unchanged, while only the number of input tokens is reduced. After decoding, the merge map restores the reduced logits to the original patch grid, followed by bilinear upsampling to the input resolution.

## 3.2. Spatial-Consistency-Aware Token Merging

Local Feature Variation. Following existing trainingfree token merging methods [1, 26], we use cosine similarity to measure token similarity. For each normalized token feature $\bar { x } _ { i } = x _ { i } / ( \| x _ { i } \| _ { 2 } + \epsilon )$ , the pairwise token similarity is given by

$$
S _ { i j } = \bar { x } _ { i } ^ { \top } \bar { x } _ { j } , \quad i , j \in \{ 1 , \ldots , N \} ,\tag{1}
$$

$S _ { i j }$ denotes the pairwise feature similarity between tokens i and $j ,$ and the diagonal entries of $S$ are set to $- \infty$ to avoid self-matching. To estimate local variation efficiently, we form a deterministic $d _ { s }$ -channel sketch by averaging contiguous groups of the normalized feature channels. The sketches are placed on the current sparse token layout, where the channel-wise variance within each $3 \times 3$ neighborhood is averaged to obtain a scalar variation $v _ { i } .$ . The sketch is used only for local-variation estimation, while token similarity and merging retain the full-dimensional features. We then apply min–max normalization across the active tokens:

![](images/8d7e543b407d6054e15427f771dd4c3c06bcd8c4e771287b614bec2adc85de7d.jpg)  
Figure 1. Overview of SCMerge. SCMerge is a plug-and-play module that can be inserted into multiple ViT encoder blocks to reduce the token sequence length while maintaining a merge map. It combines feature similarity with a spatial-consistency penalty to guide similar token pairing and merging. The reduced token sequence remains compact throughout decoding and is reconstructed onto the original dense prediction grid only after the decoder.

$$
e _ { i } = { \frac { v _ { i } - \operatorname* { m i n } _ { k } v _ { k } } { \operatorname* { m a x } _ { k } v _ { k } - \operatorname* { m i n } _ { k } v _ { k } + \epsilon } } .\tag{2}
$$

$e _ { i } \in [ 0 , 1 ]$ represents a normalized local variation score that quantifies the feature variation around i. A low $e _ { i }$ indicates a spatially homogeneous neighborhood, while a high $e _ { i }$ indicates strong local feature variation, typically occurring near semantic boundaries, thin structures, or heterogeneous regions.

Spatial-Consistency-Aware Token Pairing. We first define a score to detect token pairs that may cross semantic boundaries, that is,

$$
B _ { i j } = \operatorname* { m a x } ( e _ { i } , e _ { j } ) .\tag{3}
$$

Since $B _ { i j }$ takes the maximum local variation of the two endpoints, it grows when either token lies in a highvariation region, such as near boundaries or thin structures. This penalty term discourages boundary tokens from being merged with interior tokens, thus reducing boundary blurring and preserving spatial detail.

Next, we define a neighborhood consistency score between candidate tokens i and j as the absolute difference of their local variation values:

$$
\begin{array} { r } { D _ { i j } = | e _ { i } - e _ { j } | , } \end{array}\tag{4}
$$

where $D _ { i j }$ measures the difference in local feature variation between tokens i and j. Even if two tokens have similar local content, a large $D _ { i j }$ implies they belong to different structural regions and should not be merged. The final spatial-consistency penalty is defined as the Euclidean norm of the boundary-sensitivity term $B _ { i j }$ and the neighborhood consistency term $D _ { i j } { \mathrm { : } }$

$$
R _ { i j } = \sqrt { B _ { i j } ^ { 2 } + D _ { i j } ^ { 2 } } .\tag{5}
$$

The merge score is defined by subtracting the spatialconsistency penalty from the feature similarity:

$$
A _ { i j } = S _ { i j } - R _ { i j } ,\tag{6}
$$

where $A _ { i j }$ is the final merge score used for mutual pair selection. A high $A _ { i j }$ indicates that tokens i and j are feature-similar and that their merge is spatially consistent (i.e., the pair is less likely to involve boundary-sensitive or locally inconsistent regions). The merge score matrix A is used to determine which pairs of tokens are similar and can be merged. For each token i, we find its most similar neighbor token j by selecting the one with the highest merge score:

$$
n ( i ) = \arg \operatorname* { m a x } _ { j \neq i } A _ { i j } .\tag{7}
$$

The most similar token is selected as the maximum $A _ { i j }$ Since the pair selection only relies on the ranking of $A _ { i j }$ $A _ { i j }$ is not constrained to be non-negative. As with the similarity matrix S, the diagonal of A is set to −∞ to avoid self-matching. As in [26], we define that a pair $( i , j )$ is selected only when the nearest-neighbor relation is reciprocal:

Algorithm 1 SCMerge at one insertion   
Input: token sequence $X = [ x _ { 1 } , \dots , x _ { N } ]$ , representative coor  
dinates, merge map   
Output: reduced sequence $Y = [ y _ { 1 } , \dots , y _ { M } ]$ , representative   
coordinates, merge map   
1: Compute the feature similarity $S _ { i j } = \bar { x } _ { i } ^ { \top } \bar { x } _ { j }$ , where $\bar { x } _ { i } =$   
$x _ { i } / ( \| x _ { i } \| _ { 2 } + \epsilon )$   
2: Compute $v _ { i }$ from the $d _ { s }$ -channel sketch over the $3 \times 3$   
neighborhood and normalize $\begin{array} { r } { e _ { i } = \frac { v _ { i } - \operatorname* { m i n } _ { k } v _ { k } } { \operatorname* { m a x } _ { k } v _ { k } - \operatorname* { m i n } _ { k } v _ { k } } } \end{array}$   
<sub>k</sub> v<sub>k</sub>+ϵ   
3: Compute the spatial-consistency penalty $\begin{array} { r l } { R _ { i j } ^ { " } } & { { } = } \end{array}$   
$\sqrt { \operatorname* { m a x } ( e _ { i } , e _ { j } ) ^ { 2 } + | e _ { i } - e _ { j } | ^ { 2 } }$ and the merge score $A _ { i j } =$   
$S _ { i j } - R _ { i j }$   
4: Find $n ( i ) =$ arg max<sub>j̸=i</sub> $A _ { i j }$ and select mutual pairs $\mathcal { P } =$   
$\{ ( i , j ) \mid n ( i ) = j , n ( j ) = i \}$   
5: Merge each pair by $y = ( x _ { i } + x _ { j } ) / 2$ and retain its repre  
sentative coordinate; keep unmatched tokens   
6: return Y , representative coordinates, and merge map

$$
( i , j ) \in { \mathcal { P } } \Longleftrightarrow n ( i ) = j { \mathrm { ~ a n d ~ } } n ( j ) = i ,\tag{8}
$$

where $\mathcal { P }$ denotes the set of selected mutual pairs. Tokens that do not belong to any selected pair remain as singletons.

Token Pair Merging. For each selected token pair (i, j), the two tokens are merged by averaging their features, producing a new token:

$$
y = { \frac { x _ { i } + x _ { j } } { 2 } } .\tag{9}
$$

The merged token retains its representative coordinate, keeping all active coordinates on the original patch grid. Unpaired tokens are kept unchanged as singletons. The resulting token sequence is $Y = [ y _ { 1 } , \dots , y _ { M } ] \in \mathbb { R } ^ { M \times d } ,$ where M denotes the total number of tokens after merging. Algorithm 1 summarizes the overall procedure of SCMerge.

## 3.3. Decoding and Reconstruction

The reduced token sequence $Y$ is passed through the segmentation decoder without spatial expansion, and the original token grid is reconstructed only after decoding for dense prediction. The segmentation decoder processes the merged token sequence Y and produces class logits $H \in \mathbb { R } ^ { \breve { M } \times C }$ , where C is the number of semantic classes. The merge map maps each merged token’s logits back to the original patch locations, reconstructing the $N _ { 0 }$ patch logits. These logits are then reshaped into the original patch grid and bilinearly upsampled to the input resolution. For the Mask Transformer decoder, the M reduced tokens are jointly processed with C class embeddings, resulting in a sequence of $M + C$ tokens. Thus, the token reduction is preserved across the entire encoder–decoder pipeline. The decoder remains unchanged in architecture and parameters, introducing no extra learnable parameters.

## 4. Experiments

This section evaluates the effectiveness and generalization of our method on semantic segmentation. We plug our token merging method into existing ViT-based semantic segmentation models and present quantitative and qualitative comparisons on three benchmark semantic segmentation datasets among full-token baseline models, models with feature-similarity-based token merging, and models with our method. Later, we also provide a comprehensive ablation study supporting our design choices.

## 4.1. Experimental Setup

Datasets and Models. We conduct experiments on three datasets: ADE20K [38], Pascal Context [21], and Cityscapes [5], which respectively cover scene parsing, object and stuff segmentation, and high-resolution street scenes. Our experiments use Segmenter [29] as a baseline full-token model. To provide a fair comparison, we retain its released encoder and Mask Transformer decoder and keep the input resolution and validation protocol of each checkpoint unchanged. We insert the token merging operation after the second and fifth encoder blocks by default.

Implementation Details. We fix the deterministic channel-sketch width to $d _ { s } ~ = ~ 4 8$ across all datasets and backbones, using zero padding only when the feature dimension is not divisible by $d _ { s } .$ . After each merge stage, the remaining tokens are scattered onto a zerofilled sparse grid at their retained representative coordinates; only token features are averaged during merging. We use zero-padded $3 \times 3$ average pooling with a fixed denominator to compute the first and second moments, and derive the variance as the centered second moment. Assignment maps are combined across merge stages and used to reconstruct the dense logits after decoding the reduced tokens.

Baselines. We use the officially released Segmenter checkpoint as the full-token baseline model and as the backbone of all training-free token merging methods: ToMe [1], MPM [26], and SCMerge. For ToMe [1], we use $r = 2 0 0$ for the 1024-token Segmenter-B/16 setting and scale r proportionally with the initial token count for other resolutions and backbones, preserving the same per-layer reduction ratio. To compare our method with MPM under a similar token budget, we configure MPM to merge the top 76% of token pairs, and refer to this configuration as MPM-top.

Evaluation Metrics. We report mIoU, the average final token count $N _ { \mathrm { f i n a l } }$ , GFLOPs, and latency. Since token reduction is input-dependent, $N _ { \mathrm { f i n a l } }$ and GFLOPs are averaged over the entire validation set. Latency is measured on an NVIDIA RTX 3090 with batch size 1 in FP32 with TF32 disabled. We time the complete model forward pass, including merging and reconstruction but excluding data loading and preprocessing. After 50 warm-up iterations, the full validation set is evaluated five times with explicit CUDA synchronization, and the mean per-image latency is reported. All methods use the same hardware and software environment. Moreover, to quantify the trade-off between latency improvement and accuracy drop, we define

Table 1. Quantitative comparison results on ADE20K, Pascal Context, and Cityscapes with Segmenter backbones.
<table><tr><td>Model/Method</td><td>mIoU</td><td> $N _ { \mathrm { f i n a l } }$ </td><td>GFLOPs</td><td>Latency (ms)</td><td>R↑</td></tr><tr><td colspan="6">ADE20K</td></tr><tr><td>Segmenter-B/16</td><td>48.48</td><td>1024.00</td><td>258.80</td><td>29.77</td><td></td></tr><tr><td>ToMe (r=200)</td><td>48.01</td><td>624.00</td><td>193.65</td><td>20.13</td><td>20.51</td></tr><tr><td>MPM</td><td>47.81</td><td>563.29</td><td>182.75</td><td>19.58</td><td>15.21</td></tr><tr><td>MPM-top (76%)</td><td>48.22</td><td>655.46</td><td>198.36</td><td>22.90</td><td>26.42</td></tr><tr><td>SCMerge</td><td>48.42</td><td>653.98</td><td>181.21</td><td>20.82</td><td>149.17</td></tr><tr><td>Segmenter-B/8</td><td>49.57</td><td>4096.00</td><td>1319.38</td><td>158.80</td><td></td></tr><tr><td>ToMe</td><td>49.15</td><td>2496.00</td><td>854.96</td><td>100.33</td><td>139.21</td></tr><tr><td>MPM</td><td>49.20</td><td>2247.40</td><td>802.61</td><td>91.81</td><td>181.05</td></tr><tr><td>MPM-top</td><td>49.26</td><td>2621.96</td><td>911.04</td><td>110.23</td><td>156.68</td></tr><tr><td>SCMerge</td><td>49.38</td><td>2639.09</td><td>908.26</td><td>107.86</td><td>268.11</td></tr><tr><td>Segmenter-L/16</td><td>51.81</td><td>1600.00</td><td>1343.32</td><td>134.13</td><td></td></tr><tr><td>ToMe</td><td>51.48</td><td>976.00</td><td>886.72</td><td>80.58</td><td>162.27</td></tr><tr><td>MPM</td><td>51.02</td><td>876.40</td><td>818.72</td><td>73.89</td><td>76.25</td></tr><tr><td>MPM-top</td><td>51.45</td><td>1020.21</td><td>919.94</td><td>89.29</td><td>124.56</td></tr><tr><td>SCMerge</td><td>51.64</td><td>1040.61</td><td>892.82</td><td>87.43</td><td>274.71</td></tr><tr><td colspan="6">Pascal Context</td></tr><tr><td>Segmenter-L/16</td><td>58.08</td><td>900.00</td><td>685.67</td><td>56.89</td><td></td></tr><tr><td>ToMe</td><td>57.74</td><td>548.00</td><td>461.34</td><td>38.69</td><td>53.53</td></tr><tr><td>MPM</td><td>57.46</td><td>486.28</td><td>418.49</td><td>34.87</td><td>35.52</td></tr><tr><td>MPM-top</td><td>57.78</td><td>567.75</td><td>470.45</td><td>41.00</td><td>52.97</td></tr><tr><td>SCMerge</td><td>57.94</td><td>581.48</td><td>456.49</td><td>39.51</td><td>124.14</td></tr><tr><td colspan="6">Cityscapes</td></tr><tr><td>Segmenter-S/16</td><td>76.35</td><td>2304.00</td><td>232.07</td><td>35.98</td><td></td></tr><tr><td>ToMe</td><td>75.43</td><td>1404.00</td><td>161.97</td><td>24.90</td><td>12.04</td></tr><tr><td>MPM</td><td>75.30</td><td>1334.67</td><td>159.70</td><td>24.42</td><td>11.01</td></tr><tr><td>MPM-top</td><td>75.52</td><td>1530.32</td><td>174.66</td><td>28.72</td><td>8.75</td></tr><tr><td>SCMerge</td><td>75.68</td><td>1565.60</td><td>162.22</td><td>26.52</td><td>14.12</td></tr><tr><td>Segmenter-L/16</td><td>79.01</td><td>2304.00</td><td>2010.69</td><td>208.92</td><td></td></tr><tr><td>ToMe</td><td>78.03</td><td>1404.00</td><td>1266.86</td><td>125.36</td><td>85.27</td></tr><tr><td>MPM</td><td>77.72</td><td>1299.06</td><td>1190.04</td><td>118.68</td><td>69.95</td></tr><tr><td>MPM-top</td><td>78.23</td><td>1498.76</td><td>1346.05</td><td>133.54</td><td>96.64</td></tr><tr><td>SCMerge</td><td>78.48</td><td>1535.59</td><td>1337.82</td><td>132.80</td><td>143.62</td></tr></table>

$$
R = \frac { \mathrm { L a t e n c y } _ { \mathrm { f u l l } } - \mathrm { L a t e n c y } _ { \mathrm { m e t h o d } } } { \mathrm { m I o U } _ { \mathrm { f u l l } } - \mathrm { m I o U } _ { \mathrm { m e t h o d } } } ,\tag{10}
$$

where the numerator denotes the latency reduction relative to the full-token baseline and the denominator denotes the corresponding mIoU decrease. A larger R indicates greater latency reduction per unit of accuracy loss. We report R jointly with the corresponding raw mIoU and latency values.

## 4.2. Quantitative Comparison

Table 1 reports quantitative comparison results between the full-token baseline and token-merging methods in terms of mIoU, $N _ { \mathrm { f i n a l } }$ , GFLOPs, latency and accuracylatency trade-off ratio R. We observe that all training-free methods can be plugged into existing segmentation models and improve their accuracy-latency trade-off. MPM achieves the lowest latency but sacrifices more segmentation accuracy than other token merging methods. Under a matched token budget, MPM-top achieves better segmentation accuracy than the original MPM but leads to higher latency. Our SCMerge, with spatial-consistency-aware token merging, achieves the best accuracy-latency trade-off among all training-free token merging methods. Specifically, on ADE20K, our SCMerge consistently achieves the highest mIoU and the highest R among the training-

![](images/3e2fb366dcbdb87ee2a402d03c115bff2120779578200884c1676d3a7488cabf.jpg)  
Figure 2. Token-reduction and spatial-consistency visualization. SCMerge selects spatial-consistency token pairs with lower boundary sensitivity $B _ { i j }$ and neighborhood consistency $D _ { i j }$ than competing methods.

![](images/9104a2ee6326c66d945bf66f8348aa44f78364efa6fbf71e51d8014cbe2bfad6.jpg)  
Figure 3. Qualitative segmentation comparison on ADE20K. Prediction and error maps show that SCMerge produces fewer segmentation errors than the baseline token-merging methods, achieving the lowest pixel error rate.

## 4.3. Qualitative Comparison

free methods, showing that its accuracy–latency advantage is maintained as token density and backbone capacity increase. The same trend transfers to Pascal Context and Cityscapes. Although MPM generally achieves lower raw latency, SCMerge consistently preserves higher segmentation accuracy and yields a higher R, indicating a more favorable accuracy–latency trade-off. Across all six model–dataset settings, SCMerge improves over MPM by 0.18–0.76 mIoU while reducing full-token latency by approximately 26%–36%, without retraining or datasetspecific fine-tuning.

In Figure 2, we visualize the token assignment maps that indicate the source patches for each merged token, along with the spatial-consistency penalty values of the selected pairs. Our SCMerge achieves the lowest average values of both $B _ { i j }$ and $D _ { i j }$ across all compared methods. This result is consistent with the intended effect of the spatial-consistency penalty, which prioritizes pairs with low boundary sensitivity and small neighborhood inconsistency. The corresponding visualization shows that the proposed criterion encourages more spatially consistent

Table 2. Quantitative ablation study of spatial-consistency penalty and post-decoder reconstruction on ADE20K with Segmenter-B/16
<table><tr><td>Variant</td><td>mIoU</td><td>Latency</td><td>R</td></tr><tr><td>Full tokens</td><td>48.48</td><td>29.77</td><td>一</td></tr><tr><td>w/o penalty and</td><td></td><td></td><td></td></tr><tr><td>Post-decoder Reconstruction</td><td>47.69</td><td>19.97</td><td>12.41</td></tr><tr><td>w/o penalty</td><td>47.56</td><td>16.63</td><td>14.28</td></tr><tr><td>w/o Post-decoder</td><td>48.44</td><td>24.01</td><td>144.00</td></tr><tr><td>Full model</td><td>48.42</td><td>20.82</td><td>149.17</td></tr></table>

token merging.

Figure 3 provides a visual comparison of the resulting segmentation predictions and the corresponding pixelwise error maps. The visualization results show that our SCMerge achieves higher segmentation accuracy and a fewer pixel-wise error rate that baseline token-merging methods: ToMe, MPM, and MPM-top, with an error rate of 7.96% compared to 9.52%, 9.66%, and 8.50%, respectively. This is consistent with the quantitative comparison results, demonstrating that spatial-consistency-aware token pairing is effective in preserving segmentation quality after token reduction.

## 4.4. Ablation

Table 2 provides the effectiveness of spatial-consistencyaware token pairing and post-decoder reconstruction. We compare our full design with three variants: 1) without both spatial-consistency penalty $R _ { i j }$ and post-decoder reconstruction, i.e., reconstructing the original token grid before the decoder (w/o penalty and Post-decoder); 2) without spatial-consistency penalty $R _ { i j }$ (w/o penalty); and 3) without post-decoder reconstruction (w/o Postdecoder). We observe that the spatial-consistency penalty improves segmentation accuracy while slightly increasing latency, whereas post-decoder reconstruction significantly reduces latency but slightly reduces accuracy. Our full design with both spatial-consistency penalty and postdecoder reconstruction achieves a better overall accuracylatency trade-off.

Figure 4 analyzes the effect of insertion depth on the accuracy–efficiency trade-off. Applying token merging at earlier encoder stages yields greater computational savings in subsequent layers but makes token matching more susceptible to incorrect pairing. In contrast, merging at later stages preserves finer spatial structures and segmentation accuracy at the cost of reduced latency gains. Therefore, token merging is performed in the early-to-middle encoder blocks.

## 4.5. Analyses

Boundary Preservation. Table 3 reports B-mIoU@5 (mIoU within 5 pixels of boundaries) and B-Merge (fraction of boundary tokens involved in merges). Our

![](images/5ba5227b46307fd89e9c4ab1d91d73e6bc5fca8482c361a8e73757028068897e.jpg)  
Figure 4. Accuracy–latency trade-off of MPM and our SCMerge on ADE20K. Labels denote zero-based encoder-block indices. Our SCMerge consistently achieves a higher mIoU and a better accuracy–latency trade-off than MPM.

Table 3. Effect of the boundary-sensitivity term $B _ { i j }$ on ADE20K.
<table><tr><td>Method</td><td>B-mIoU@5↑</td><td>B-Merge (%) ↓</td></tr><tr><td>Full tokens</td><td>35.37</td><td></td></tr><tr><td>MPM</td><td>34.22</td><td>50.33</td></tr><tr><td>MPM-top</td><td>34.59</td><td>42.55</td></tr><tr><td>SCMerge</td><td>35.29</td><td>22.55</td></tr></table>

Table 4. Effect of the neighborhood consistency term $D _ { i j }$ on ADE20K.
<table><tr><td>Method</td><td>Cross-cls (%) ↓</td><td>N-Agree (%) ↑</td></tr><tr><td>MPM</td><td>3.874</td><td>93.28</td></tr><tr><td>MPM-top</td><td>3.252</td><td>94.01</td></tr><tr><td>SCMerge</td><td>2.168</td><td>95.25</td></tr></table>

SCMerge improves B-mIoU@5 while sharply reducing B-Merge, confirming that the boundary-sensitivity term $B _ { i j }$ prevents boundary tokens from being merged, improving boundary segmentation quality.

Neighborhood Consistency. Table 4 reports the neighborhood consistency of selected token pairs. Crosscls denotes the fraction of pairs connecting tokens from different classes, while N-Agree measures the positionwise label agreement between the 3 × 3 neighborhoods of paired tokens. Our SCMerge produces significantly fewer cross-class pairs and achieves higher N-Agree, indicating that $D _ { i j }$ effectively promotes the selection of semantically consistent token pairs.

## 5. Conclusion

In this work, we presented SCMerge, a training-free token merging method that conducts token merging based on both feature similarity and spatial consistency for semantic segmentation. The method is flexible and can be plugged into existing ViT-based semantic segmentation models for improving computation efficiency with less quality drop. Extensive comparison experiments on three benchmark semantic segmentation datasets ADE20K, Pascal Context, and Cityscapes demonstrate that our SCMerge can achieve better accuracy-latency trade-off compared with existing state-of-the-art training-free token merging methods. Although our SCMerge with the spatial-consistency penalty mitigates boundary blurring, very small objects and extremely thin structures may still be smoothed due to information aggregation. In addition, since SCMerge is training-free and introduces no learnable parameters, its merging decisions are based on fixed, hand-designed criteria. This may limit its adaptability to diverse or unseen visual patterns. Addressing these limitations is a direction for our future work while preserving the plug-and-play and training-free advantages of SCMerge.

## References

[1] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In International Conference on Learning Representations, 2023.

[2] Maxim Bonnaerens and Joni Dambre. Learned thresholds token merging and pruning for vision transformers. Transactions on Machine Learning Research, 2023.

[3] Mengzhao Chen, Wenqi Shao, Peng Xu, Mingbao Lin, Kaipeng Zhang, Fei Chao, Rongrong Ji, Yu Qiao, and Ping Luo. Diffrate: Differentiable compression rate for efficient vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17164–17174, 2023.

[4] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1290–1299, 2022.

[5] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.

[6] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-efficient exact ´ attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

[7] Peiyan Dong, Mengshu Sun, Alec Lu, Yanyue Xie, Kenneth Liu, Zhenglun Kong, Xin Meng, Zhengang Li, Xue Lin, Zhenman Fang, and Yanzhi Wang. Heatvit: Hardware-efficient adaptive token pruning for vision transformers. In 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 442– 455. IEEE, 2023.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An im-

age is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[9] Mohsen Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei Jafari, Sunando Sengupta, Hamid Reza Vaezi Joze, Eric Sommerlade, Hamed Pirsiavash, and Juergen Gall. Adaptive token sampling for efficient vision transformers. In European Conference on Computer Vision, pages 396–414. Springer, 2022.

[10] Zhanzhou Feng and Shiliang Zhang. Efficient vision transformer via token merger. IEEE Transactions on Image Processing, 32:4156–4169, 2023.

[11] Jung Hwan Heo, Seyedarmin Azizi, Arash Fayyazi, and Massoud Pedram. Training-free acceleration of vits with delayed spatial merging. arXiv preprint arXiv:2303.02331, 2023.

[12] Huaibo Huang, Xiaoqiang Zhou, Jie Cao, Ran He, and Tieniu Tan. Vision transformer with super token sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22690– 22699, 2023.

[13] Hsiang-Wei Huang, Wenhao Chai, Kuang-Ming Chen, Cheng-Yen Yang, and Jenq-Neng Hwang. Tosa: Token merging with spatial awareness. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9654–9660. IEEE, 2025.

[14] Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. Token fusion: Bridging the gap between token pruning and token merging. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1383–1392, 2024.

[15] Zhenglun Kong, Peiyan Dong, Xiaolong Ma, Xin Meng, Wei Niu, Mengshu Sun, Xuan Shen, Geng Yuan, Bin Ren, Hao Tang, Minghai Qin, and Yanzhi Wang. Spvit: Enabling faster vision transformers via latency-aware soft token pruning. In European Conference on Computer Vision, pages 620–640. Springer, 2022.

[16] Sanghyeok Lee, Joonmyung Choi, and Hyunwoo J. Kim. Multi-criteria token fusion with one-step-ahead attention for efficient vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15741–15750, 2024.

[17] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. In International Conference on Learning Representations, 2022.

[18] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 10012–10022, 2021.

[19] Chenyang Lu, Daan De Geus, and Gijs Dubbelman. Content-aware token sharing for efficient semantic segmentation with vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23631–23640, 2023.

[20] Dmitrii Marin, Jen-Hao Rick Chang, Anurag Ranjan, Anish Prabhu, Mohammad Rastegari, and Oncel Tuzel. Token pooling in vision transformers for image classification. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 12–21, 2023.

[21] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 891–898, 2014.

[22] Narges Norouzi, Svetlana Orlova, Daan De Geus, and Gijs Dubbelman. Algm: Adaptive local-then-global token merging for efficient semantic segmentation with plain vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15773–15782, 2024.

[23] Bowen Pan, Rameswar Panda, Yifan Jiang, Zhangyang Wang, Rogerio Feris, and Aude Oliva. Ia-red<sup>2</sup>: Interpretability-aware redundancy reduction for vision transformers. In Advances in Neural Information Processing Systems, pages 24898–24911, 2021.

[24] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun.´ Vision transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12179–12188, 2021.

[25] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34: 13937–13949, 2021.

[26] Simon Rave, Pejman Rasti, and David Rousseau. MPM:´ Mutual pair merging for efficient vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 2998–3008, 2026.

[27] Cedric Renggli, Andr ´ e Susano Pinto, Neil Houlsby, Basil ´ Mustafa, Joan Puigcerver, and Carlos Riquelme. Learning to merge tokens in vision transformers. arXiv preprint arXiv:2202.12015, 2022.

[28] Michael S. Ryoo, A. J. Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: Adaptive space-time tokenization for videos. In Advances in Neural Information Processing Systems, pages 12786– 12797, 2021.

[29] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7262–7272, 2021.

[30] Yehui Tang, Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chao Xu, and Dacheng Tao. Patch slimming for efficient vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12165–12174, 2022.

[31] Hoai-Chau Tran, Duy M. H. Nguyen, Duy M. Nguyen, TrungTin Nguyen, Ngan Le, Pengtao Xie, Daniel Sonntag, James Y. Zou, Binh T. Nguyen, and Mathias Niepert. Accelerating transformers with spectrum-preserving token

merging. In Advances in Neural Information Processing Systems, pages 30772–30810, 2024.

[32] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 568–578, 2021.

[33] Yancheng Wang and Yingzhen Yang. Efficient visual transformer by learnable token merging. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47 (11):9597–9608, 2025.

[34] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34:12077–12090, 2021.

[35] Weiying Xie, Xiaoyu Chen, Xin Zhang, Chenhe Hao, Jitao Ma, Yunsong Li, and Leyuan Fang. Saliency-driven token merging for vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32184–32193, 2026.

[36] Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10809–10818, 2022.

[37] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H. S. Torr, and Li Zhang. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6881–6890, 2021.

[38] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.