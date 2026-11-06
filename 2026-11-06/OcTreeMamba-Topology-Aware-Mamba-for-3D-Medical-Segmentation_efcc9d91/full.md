# OcTreeMamba: Topology-Aware Mamba for 3D Medical Segmentation via Octree Sparsification

Kaixiang Mai<sup>1</sup>, Jiaqi Yu<sup>1</sup>, Yuzhu Ji<sup>1,\*</sup>, Yiqun Zhang<sup>1</sup>, Zhijie Wang<sup>2</sup>, Haijun Zhang<sup>3</sup>, Yiu-ming Cheung<sup>4</sup> <sup>1</sup>Guangdong University of Technology, Guangzhou, China, <sup>2</sup>RIKEN Center for Advanced Intelligence Project, Tokyo, Japan, <sup>3</sup>Harbin Institute of Technology, Shenzhen, China, <sup>4</sup>Hong Kong Baptist University, Hong Kong, China

Abstract—Accurate segmentation of 3D anatomical structures is crucial to medical image analysis. Recently, Mamba-based approaches have achieved promising segmentation performance in capturing long-range dependencies with linear-time computational complexity. However, heuristic selective scanning strategies inherently disrupt the intrinsic spatial hierarchy and increase computational redundancy, ignoring the inherent sparsity of 3D anatomical structures. To this end, we propose OcTreeMamba, a hierarchical framework that incorporates Octree structures into SSM, achieving 3D topology-aware long-range context feature propagation and integration. Specifically, a Structure Learner performs anatomy-adaptive structured sparsification through hierarchical gating and Gumbel-Softmax sampling, enabling differentiable selection of anatomical regions. The selected regions are subsequently organized by a Dynamic Octree Builder into a multi-level topology that preserves their hierarchical connectivity. Based on the constructed Octree, a bidirectional Tree-SSM propagates and integrates contextual features along connectivityaware leaf–parent–root and root–parent–leaf trajectories. Extensive experiments on four benchmark datasets show that our proposed OcTreeMamba achieves competitive performance w.r.t. topological continuity and computational efficiency, particularly for recovering the complex structure on vascular datasets. The code is available here.

Index Terms—Octree structure, State space model, Mamba, Dynamic scan path, Medical image segmentation

## I. INTRODUCTION

Automatic segmentation of anatomical structures, such as blood vessels, airways, tumors, and organs, is a foundation task in computer-aided medical image analysis and facilitates disease diagnosis, treatment planning and surgical navigation [1]–[3]. Deep Convolutional Neural Networks (DCNNs), particularly the U-Net [4] and its variants [5]–[10], have achieved remarkable progress in medical image segmentation. However, the locality of convolutional operations inherently limits their ability to capture long-range dependencies within confined receptive fields. Therefore, CNN-based models with residual connections [11], deformable convolutions [12], large kernels [13] and attention mechanisms [14] have been proposed for capturing long-range contextual features. However, it remains challenging for CNN models to segment anatomical structures, such as vessels and tumors, with intricate branching patterns, irregular shapes or elongated morphology. In contrast, Vision Transformer (ViT)-based methods [15]–[20] were proposed to model global contextual relations by using a dense query-based self-attention block. However, the quadratic computational and memory complexity in terms of the number of tokens becomes the bottleneck when applying it to 3D voxel grids. In addition, ViT-based models practically rely on sliding-window strategies and aggressive downsampling, which, to some extent, impedes the recovery of fine-grained structures with spatial details.

![](images/a64de9673718327959d041b182accfdfd5dc84ea22131a3101963b3605b249b9.jpg)  
Fig. 1. Motivation of OcTreeMamba. (a) Dense scanning over 3D grids disrupts local anatomical adjacency and increases computation redundancy on non-informative background regions. (b) Anatomy-aware sparse octree topology constructed in OctreeMamba aims to propagate features along foreground structures through hierarchical interactions, improving topology preservation and structure-aware segmentation while maintaining a memory-performance balance.

Recently, State Space Models (SSMs) [21], particularly the Mamba architecture [22], have been proposed for vision and natural language processing tasks. Due to efficiency for long-term feature propagation and linear scalability with the sequence length, SSM-based models have drawn significant attention [23], [24]. For medical image segmentation, Mambabased frameworks, such as U-Mamba [25], Vim [26] and VMamba [27], have been developed. To perform sequence modeling for the 3D volumetric data, these approaches practically flatten the dense spatial grids into 1D sequences.

Generally, handcrafted scanning patterns are introduced to obtain rich contextual relations with dense spatial traversal. However, the handcrafted scanning with fixed patterns practically corrupts the intrinsic spatial relations and makes it fail to capture the geometry and spatial distribution of anatomical structures effectively [25]. Moreover, adjacent voxels in the 3D non-causal volume data are widely separated when scanning the 1D sequence, causing inaccurate segmentation results with topological inconsistencies. In addition, the scanning algorithm generally propagates hidden states among voxels, without identifying whether the voxels contain valid contextual features. It thus increases a large amount of redundant computation and latency on multiple traversals for propagating features across structurally irrelevant regions, such as backgrounds and imaging noise. Furthermore, the sequential context feature propagation on dense grids leads to misalignment with the topology and sparse distribution of anatomical structures (see Fig. 1(a)).

We argue that, to achieve topology-preserving and computationally efficient sequence modeling, the scanning trajectories should: 1) align with the topologies of anatomical structures intrinsically, and 2) be adaptively sparse by filtering out noninformative nodes. Therefore, instead of introducing multiple handcrafted dense traversal trajectories, we reformulate feature propagation in SSM over a sparse tree-based topology to conform the propagation paths to the intrinsic anatomical structures (see Fig. 1(b)). To this end, we propose OcTreeMamba, a novel topology-adaptive state space model for 3D medical image segmentation. Our key technical insight is to reformulate feature propagation on a sparse yet efficient Octree topology on dense grids. Specifically, the proposed model consists of three key modules: 1) A Structure Learner is designed to perform differentiable anatomy-adaptive sparsification through hierarchical gating and Gumbel-Softmax sampling, enabling informative anatomical regions to be selected in an end-to-end manner; 2) A Dynamic Octree Builder is proposed to organize the selected regions into a multi-level Octree topology while preserving their hierarchical connectivity; 3) A bidirectional Tree-SSM is developed to facilitate efficient feature aggregation and broadcasting along connectivity-aware bidirectional tree traversal. Experimental results on four benchmark datasets demonstrate that OcTreeMamba achieves competitive segmentation accuracy and state-of-the-art topology preservation on vascular datasets, while maintaining a favorable memoryperformance balance on dense organ and lesion segmentation tasks. In summary, our main contributions are threefold:

1) We propose a novel Octree-based Mamba framework for 3D medical image segmentation. To the best of our knowledge, we are the first to integrate the hierarchical Octree structure to perform feature propagation along tree topology in SSM-based models for medical image segmentation.

2) We develop a hierarchical Octree construction mechanism for preserving the differentiability of the framework, while dynamically sparsifying the Octree to enable semanticdriven structure learning.

3) We present a topology-aware bidirectional feature propagation by adapting the Tree-SSM. It reformulates conventional scan paths as hierarchical propagation along Octree connectivity, enabling long-range dependency modeling while preserving local topology.

## II. RELATED WORK

## A. CNN-based Models

CNN-based models, e.g., U-Net and nnU-Net [4], [7], have prevailed in the field of medical image segmentation due to their ability to extract representative local features, becoming foundational architectures for various medical image analysis tasks [5], [9], [28], [29]. Generally, these architectures introduce skip connections [4] and multi-scale feature fusion [9], [10], to integrate local and global contextual features and recover spatial details from downsampled features. To process volumetric data, 3D U-Net [6] and V-Net [30] adapted 3D convolution operations to exploit inter-slice correlations in volumetric data. Subsequent works further augmented this paradigm by incorporating residual learning [31] and attention mechanisms [5], [32], enabling the network to focus on discriminative regions while suppressing background noise. However, this efficiency stems from their reliance on local convolutions, which predict voxel-wise labels based on limited receptive fields, and thus struggle to capture long-range dependencies and global semantic context.

## B. Transformer-based Models

To overcome the locality limitations of CNNs, Dosovitskiy et al. [15] introduced Vision Transformers (ViT) by using self-attention to model global dependencies, inspiring CNN-ViT hybrid architectures for medical image segmentation. Specifically, Chen et al. [17] pioneered TransUNet [17] by integrating a Transformer bottleneck into a CNN encoder-decoder architecture for global context modeling. Subsequently, Hatamizadeh et al. [33] presented UNETR, a pure Transformer framework that processes 3D volumetric patches with a ViT encoder to capture long-range dependencies. Furthermore, Hatamizadeh et al. [19] developed Swin-UNETR, incorporating shifted-window attention to reduce the cost of dense self-attention. However, applying Transformers to 3D images remains constrained by quadratic computational complexity [15], [33], [34]. Different from these approaches, OcTreeMamba avoids dense pairwise interactions over volumetric tokens by organizing informative anatomical regions into a hierarchical octree and propagating long-range structural information through the SSM, while retaining a dense convolutional pathway to preserve local details.

## C. Mamba-based Models

To capture long-range dependencies while preserving linear computational efficiency, the State Space Model (SSM) and its variants [21], [35], [36] have been introduced for both vision and natural language processing tasks. Specifically, S4 [21] was proposed by improving the original SSM using structural constraints on the state matrix and initializing it with High-Order Polynomial Projection Operator (HIPPO) [37].

![](images/0a2e293b319819aca6e327a941f54ac16cc28e9bd5ccc4ee0c14dcb7444fcb3a.jpg)  
Fig. 2. Overview of the proposed OcTreeMamba. The overall model (a) is built upon a UNet encoder-decoder framework. The Octree Encoder (b) is the core component, which consists of three modules: a topology-aware Structure Learner (c) for identifying active anatomical regions, a Dynamic Octree Builder (d) for constructing hierarchical sparse topology, and a bidirectional Tree-SSM Propagation module (e) for topology-aware long-range feature modeling.

Gu et al. [22] developed Mamba, also known as S6, by introducing the selective scan mechanism and achieving $O ( N )$ inference complexity and parallel training capability. Furthermore, considering the non-causal nature of images, Vim [26] and VMamba [27] incorporate strategies to traverse image patches via multiple heuristic scanning paths. Recently, hybrid architectures integrating Mamba with U-Net or Transformers, such as U-Mamba [25] and SegMamba-V2 [38], have been proposed for 3D medical image segmentation by adapting the handcrafted scanning patterns to the 3D space along tri-plane perspectives. However, these methods still rely on predefined dense scanning paths without learning anatomyadaptive hierarchical representations, separating adjacent 3D voxels and causing topological inconsistencies [27], [39], [40]. In contrast, OcTreeMamba selects informative anatomical regions, organizes them into a hierarchical Octree, and propagates features along topology-aware traversal paths with less redundant computation over background regions.

## III. METHOD

## A. Overview

As shown in Fig. 2, OcTreeMamba is built upon an encoder-decoder architecture. The core of the framework is the Octree Encoder, which consists of a topology-aware structure learner, a dynamic octree builder and a bidirectional Tree-SSM propagating module (Fig. 2(c)-(e)). Specifically, the topology-aware Structure Learner predicts active anatomical masks from the dense feature map, and the Dynamic Octree Builder constructs a hierarchical octree topology leveraging the predicted semantic activation masks. The Bi-directional Tree-SSM Propagation module performs topology-aware feature propagation over the constructed octree to model longrange semantic dependencies among sparse anatomical regions. The obtained features are upsampled and re-injected into the dense backbone through residual fusion, followed by downsampling operations to produce the output feature map at multiple scales. This design preserves fine-grained spatial details while integrating local and global structural context through topology-aware Tree-SSM propagation.

## B. Structure Learner

For 3D medical image data, anatomical regions for segmentation generally occupy only a limited portion of the volumetric space. In contrast, background regions and homogeneous interiors contribute little to long-range structural modeling. Directly applying state-space propagation to the dense grid not only introduces redundant computation but also weakens the structural responses of sparse foreground regions in sequential modeling. Therefore, before constructing the octree topology, we designed a structure learner to locate anatomy-relevant regions for subsequent topology-aware propagation.

Given an input feature $\mathbf { X } ^ { ( i n ) }$ , the structure learner produces activation scores at two complementary granularities, i.e., coarse- and fine-grained predictions. Specifically, the coarse predictor $\Phi ^ { ( c ) }$ produces binary activation scores $\mathbf { \bar { S } } ^ { ( c ) }$

$$
\mathbf { S } ^ { ( c ) } = \Phi ^ { ( c ) } ( \mathtt { M a x P o o l } ( \mathbf { X } ^ { ( i n ) } ) ) ,\tag{1}
$$

where $\mathbf { S } ^ { ( c ) }$ denotes the prediction scores for the inactive and active states at each spatial location. MaxPool(·) represents a max-pooling operation to reduce the feature resolution. This coarse prediction aims to provide a stable semantic prior for suppressing large irrelevant regions. In parallel, a fine predictor $\Phi ^ { ( f ) }$ operates on the original resolution and produces finegrained activation map $\mathbf { S } ^ { ( f ) }$ to capture boundary details:

$$
\mathbf { S } ^ { ( f ) } = \Phi ^ { ( f ) } ( \mathbf { X } ^ { ( i n ) } ) .\tag{2}
$$

To enable end-to-end optimization of discrete structure selection, a Gumbel-Softmax relaxation is introduced to provide differentiable approximations to the binary activation decisions. For each prediction granularity $r ~ \in ~ \{ c , f \}$ , the relaxed activation probability is computed as:

$$
\mathbf { P } _ { k } ^ { ( r ) } = \frac { \exp ( ( \mathbf { S } _ { k } ^ { ( r ) } + g ^ { ( r ) } k ) / \tau ) } { \sum _ { j } \exp ( ( \mathbf { S } _ { j } ^ { ( r ) } + g _ { j } ^ { ( r ) } ) / \tau ) } , \quad k \in \{ 0 , 1 \}\tag{3}
$$

where $\mathbf { S } _ { k } ^ { ( r ) }$ and $\mathbf { P } _ { k } ^ { ( r ) }$ denote the activation score and relaxed probability for decision k at granularity $^ { r } \cdot$ The indices $k = 0$ and $k = 1$ represent the inactive and active decisions. $g _ { k } ^ { ( r ) } \sim$ Gumbel(0, 1) denotes Gumbel noise, and τ is the temperature coefficient. By applying the Gumbel-Softmax relaxation, two binary activation masks can be derived by selecting the most probable activation state:

$$
\mathbf { M } ^ { ( r ) } = \mathbb { I } \left[ \arg \operatorname* { m a x } _ { k \in 0 , 1 } \mathbf { P } _ { k } ^ { ( r ) } = 1 \right] , \quad r \in \{ c , f \} ,\tag{4}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. $\mathbf { M } ^ { ( c ) }$ and $\mathbf { M } ^ { ( f ) }$ denote the coarse- and fine-grained binary masks, respectively. The hard binary masks are used in the forward pass, while gradients are propagated through the corresponding soft probabilities, enabling end-to-end optimization of the discrete structure.

In practice, these two binary activation masks describe structural activation w.r.t. different granularities. Intuitively, the coarse mask provides region-level semantic reliability, whereas the fine mask preserves local boundary and thin-structure responses. To enforce hierarchical consistency between the two scales, we introduce a top-down gating mechanism:

$$
\mathbf { M } ^ { ( f i n a l ) } = \mathbf { M } ^ { ( f ) } \wedge ( \mathrm { U p s a m p l e } ( \mathbf { M } ^ { ( c ) } ) ) ,\tag{5}
$$

where ∧ denotes element-wise logical intersection, which retains a voxel only when it is activated by both the fine-scale predictor and its corresponding coarse-scale parent, thereby suppressing isolated activations from local predictions. Such a multi-granularity compensation design is important for sparse and elongated anatomical structures, where both global support localization for the Octree structure and local boundary preservation are required for achieving accurate segmentation.

## C. Dynamic Octree Builder

The activation mask produced by the Structure Learner identifies active voxels through $\mathbf { M } ^ { ( f i n a l ) }$ , but these voxels remain isolated spatial locations without defining the hierarchical connections and interactions. Thus, our Dynamic Octree Builder further converts the sparse voxels into a hierarchical Octree topology. Specifically, for any active leaf node at spatial coordinate $v _ { i }$ (where $\mathbf { M } ^ { ( f i \dot { n } a l ) } ( v _ { i } ) \dot { = } 1 )$ , the Octree hierarchy is established by recursively mapping it to its ancestors via coordinate quantization:

$$
\mathcal { P } ( v _ { i } ) = \lfloor v _ { i } / 2 \rfloor , \quad \mathcal { G P } ( v _ { i } ) = \lfloor \mathcal { P } ( v _ { i } ) / 2 \rfloor ,\tag{6}
$$

where $\mathcal { P } ( v _ { i } )$ and $\mathcal G P ( v _ { i } )$ denote the parent and grandparent nodes of $v _ { i } ,$ respectively. More generally, the ancestor can be obtained by recursively quantizing the coordinates of its predecessor nodes. In this way, spatially adjacent active voxels are naturally grouped into shared parent nodes, and distant regions can interact through higher-level ancestors. In our implementation, the unstructured sparse mask is thereby transformed into a connected multi-level tree topology for the Tree-SSM message passing.

## D. Bidirectional OcTree-SSM Feature Propagation

The core insight of our proposed method is to perform feature propagation via hierarchical sparse tree traversal along the learned octree. Inspired by [40], a bidirectional hierarchical Octree-topology feature propagation mechanism is developed to model long-range dependencies through bottom-up aggregation and top-down broadcasting. Intuitively, the bottomup propagation aims to condense local responses into coarse structural states, allowing spatially separated active regions to interact through their shared ancestors. Top-down broadcasting is tailored to redistribute the contextual information to refine parent and leaf representations, while the input-conditioned channel-wise modulations selectively control information flow at each propagation stage. Consequently, long-range structural dependencies can be modelled over sparse active nodes maintaining the learned hierarchical topology.

Let V, U, and $\mathcal { G }$ denote the active leaf, parent, and grandparent node sets, respectively. For a leaf node $v ~ \in ~ \mathcal { V } , ~ x _ { v }$ denotes its local feature, $\vartheta ( v ) \in \mathcal { U }$ denotes its parent, and $\zeta ( u ) \ \in \ \mathcal { G }$ denotes the grandparent of a parent node u. We define the input-conditioned modulation used in different propagation stages. In the bottom-up direction, leaf features are first aggregated to their parent nodes:

$$
\alpha _ { v } ^ { \uparrow } = \rho \big ( \mathtt { M L P } \big ( x _ { v } \big ) \big ) , \quad h _ { v \to u } = \sum _ { v : \vartheta ( v ) = u } \alpha _ { v } ^ { \uparrow } \odot x _ { v } ,\tag{7}
$$

where $\rho ( \cdot )$ is the SiLU activation function, MLP(·)) denotes a multi-layer perception for producing parameters. This vector is used for channel-wise selective modulations rather than dense transition matrices. The superscript ↑ indicates selective modulation applied in bottom-up aggregation. Similarly, the parent states are further aggregated to grandparent nodes:

$$
\alpha _ { u } ^ { \uparrow } = \rho ( \mathtt { M L P } ( h _ { v  u } ) ) , \quad h _ { u  \eta } = \sum _ { u : \zeta ( u ) = \eta } \alpha _ { u } ^ { \uparrow } \odot h _ { v  u } .\tag{8}
$$

At the grandparent level, we apply an input-conditioned modulation to obtain the contextual state:

$$
\begin{array} { r } { \beta _ { \eta } ^ { \uparrow } = \rho \big ( \mathtt { M L P } \big ( h _ { u  \eta } \big ) \big ) , \quad \tilde { h } _ { u  \eta } = \beta _ { \eta } ^ { \uparrow } \odot h _ { u  \eta } , } \end{array}\tag{9}
$$

where $\tilde { h } _ { u  \eta }$ denotes the modulated state stored at grandparent node $\eta .$ This bottom-up propagation condenses sparse local responses into coarser structural states.

In the top-down direction, let $z _ { \eta } = \tilde { h } _ { u  \eta } [ \zeta ( u ) ]$ retrieve the contextual state of grandparent nodes. The contextual state is then broadcast back to parent nodes:

$$
\begin{array} { r } { \beta _ { u } ^ { \downarrow } = \rho \big ( \mathtt { M L P } \big ( h _ { v  u } \big ) \big ) , \quad \tilde { h } _ { \eta  u } = h _ { v  u } + \beta _ { u } ^ { \downarrow } \odot z _ { \eta } , } \end{array}\tag{10}
$$

where $\beta _ { u } ^ { \downarrow }$ gates the grandparent context $z _ { \eta }$ injected into the parent state, yielding the enhanced representation $\tilde { h } _ { \eta  u }$ . Each

TABLE I  
QUANTITATIVE VALIDATION ON IMAGECAS, TOPCOW2024, BRATS2021, AND ABDOMENCT BENCHMARK DATASETS.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Segmentation Accuracy</td><td colspan="4">Efficiency / Memory</td></tr><tr><td>Dice↑</td><td>clDice↑</td><td>HD95↓</td><td>IoU↑</td><td>TM↓</td><td>IM↓</td><td>Param.↓</td><td>TP↑</td></tr><tr><td rowspan="9">IaCAS</td><td>nnU-Net</td><td>CNN</td><td>81.54</td><td>86.60</td><td>16.03</td><td>69.16</td><td>5563.40</td><td>2706.91</td><td>30.79</td><td>42.55</td></tr><tr><td>MedNeXt</td><td>CNN</td><td>80.17</td><td>85.20</td><td>22.04</td><td>67.26</td><td>20959.74</td><td>4063.70</td><td>5.55</td><td>15.55</td></tr><tr><td>SegFormer3D</td><td>ViT</td><td>79.39</td><td>85.72</td><td>16.80</td><td>66.08</td><td>2145.73</td><td>895.16</td><td>4.49</td><td>131.88</td></tr><tr><td>UNETR++</td><td>ViT</td><td>80.42</td><td>85.80</td><td>13.67</td><td>67.55</td><td>3995.28</td><td>1441.55</td><td>42.64</td><td>38.42</td></tr><tr><td>U-Mamba_Bot</td><td>Mamba</td><td>82.11</td><td>87.17</td><td>13.19</td><td>70.00</td><td>11175.03</td><td>3173.21</td><td>42.12</td><td>18.91</td></tr><tr><td>SegMamba</td><td>Mamba</td><td>81.22</td><td>85.63</td><td>17.42</td><td>68.70</td><td>16963.61</td><td>4454.63</td><td>67.36</td><td>9.55</td></tr><tr><td>SegMamba-V2</td><td>Mamba</td><td>81.13</td><td>85.82</td><td>17.04</td><td>68.56</td><td>15080.17</td><td>5457.21</td><td>138.77</td><td>11.70</td></tr><tr><td>OcTreeMamba_Bot</td><td>CNN+SSM</td><td>82.31</td><td>87.67</td><td>12.88</td><td>70.23</td><td>5942.21</td><td>2771.79</td><td>34.78</td><td>36.57</td></tr><tr><td>OcTreeMamba_Enc</td><td>CNN+SSM</td><td>81.91</td><td>87.18</td><td>14.32</td><td>69.67</td><td>15465.01</td><td>3030.85</td><td>40.57</td><td>25.14</td></tr><tr><td rowspan="9">Top024</td><td>nnU-Net</td><td>CNN</td><td>65.57</td><td>78.61</td><td>25.13</td><td>56.97</td><td>5491.87</td><td>2630.60</td><td>30.80</td><td>44.12</td></tr><tr><td>MedNeXt</td><td>CNN</td><td>66.82</td><td>79.01</td><td>26.54</td><td>58.47</td><td>20254.12</td><td>3917.88</td><td>5.56</td><td>15.76</td></tr><tr><td>SegFormer3D</td><td>ViT</td><td>48.80</td><td>70.97</td><td>46.67</td><td>40.75</td><td>2925.81</td><td>862.44</td><td>4.50</td><td>130.13</td></tr><tr><td>UNETR++</td><td>ViT</td><td>64.96</td><td>76.97</td><td>28.60</td><td>56.00</td><td>4956.18</td><td>1440.80</td><td>42.64</td><td>37.60</td></tr><tr><td>U-Mamba_Bot</td><td>Mamba</td><td>60.34</td><td>77.81</td><td>29.68</td><td>52.89</td><td>10879.82</td><td>3084.89</td><td>42.13</td><td>19.15</td></tr><tr><td>SegMamba</td><td>Mamba</td><td>66.99</td><td>79.51</td><td>23.95</td><td>58.36</td><td>16419.50</td><td>4313.02</td><td>67.36</td><td>9.82</td></tr><tr><td>SegMamba-V2</td><td>Mamba</td><td>66.99</td><td>79.53</td><td>21.80</td><td>58.31</td><td>14640.62</td><td>5315.59</td><td>138.77</td><td>12.80</td></tr><tr><td>OcTreeMamba_Bot</td><td>CNN+SSM</td><td>66.28</td><td>79.50</td><td>21.22</td><td>57.62</td><td>6258.60</td><td>2698.48</td><td>34.79</td><td>37.58</td></tr><tr><td>OcTreeMamba_Enc</td><td>CNN+SSM</td><td>67.17</td><td>80.82</td><td>19.68</td><td>58.46</td><td>15311.03</td><td>2939.08</td><td>40.58</td><td>25.86</td></tr><tr><td rowspan="9">B2021</td><td>nnU-Net</td><td>CNN</td><td>84.52</td><td>一</td><td>9.76</td><td>77.28</td><td>4887.71</td><td>2425.92</td><td>31.20</td><td>52.77</td></tr><tr><td>MedNeXt</td><td>CNN</td><td>84.42</td><td>一</td><td>10.38</td><td>77.30</td><td>17968.72</td><td>3527.31</td><td>5.55</td><td>18.16</td></tr><tr><td>SegFormer3D</td><td>ViT</td><td>80.20</td><td></td><td>12.78</td><td>70.99</td><td>1816.21</td><td>823.51</td><td>4.53</td><td>140.08</td></tr><tr><td>UNETR++</td><td>ViT</td><td>83.75</td><td>一</td><td>10.99</td><td>76.39</td><td>4103.96</td><td>1470.90</td><td>42.65</td><td>38.81</td></tr><tr><td>U-Mamba_Bot</td><td>Mamba</td><td>84.27</td><td></td><td>10.78</td><td>77.07</td><td>9706.80</td><td>2842.64</td><td>42.12</td><td>22.90</td></tr><tr><td>SegMamba</td><td>Mamba</td><td>84.20</td><td></td><td>11.81</td><td>76.95</td><td>14708.65</td><td>3984.07</td><td>67.42</td><td>11.50</td></tr><tr><td>SegMamba-V2</td><td>Mamba</td><td>84.10</td><td>一</td><td>11.42</td><td>76.88</td><td>13183.95</td><td>4985.98</td><td>138.78</td><td>14.17</td></tr><tr><td>OcTreeMamba_Bot</td><td>CNN+SSM</td><td>84.44</td><td>一</td><td>10.82</td><td>77.17</td><td>5262.46</td><td>2501.82</td><td>35.19</td><td>42.46</td></tr><tr><td>OcTreeMamba_Enc</td><td>CNN+SSM</td><td>84.49</td><td></td><td>11.36</td><td>77.24</td><td>11586.51</td><td>2725.12</td><td>40.98</td><td>26.40</td></tr><tr><td rowspan="9">AbddncCT</td><td>nnU-Net</td><td>CNN</td><td>84.68</td><td></td><td>30.26</td><td>77.20</td><td>4803.27</td><td>2163.55</td><td>30.71</td><td>51.63</td></tr><tr><td>MedNeXt</td><td>CNN</td><td>85.71</td><td></td><td>28.73</td><td>78.41</td><td>14726.03</td><td>2848.82</td><td>5.56</td><td>22.00</td></tr><tr><td>SegFormer3D</td><td>ViT</td><td>80.08</td><td></td><td>36.47</td><td>70.92</td><td>1962.25</td><td>649.71</td><td>4.50</td><td>133.79</td></tr><tr><td>UNETR++</td><td>ViT</td><td>86.05</td><td></td><td>21.00</td><td>78.96</td><td>4538.10</td><td>1435.92</td><td>42.64</td><td>36.98</td></tr><tr><td>U-Mamba_Bot</td><td>Mamba</td><td>86.61</td><td></td><td>18.20</td><td>79.87</td><td>9234.50</td><td>2538.70</td><td>41.95</td><td>23.94</td></tr><tr><td>SegMamba</td><td>Mamba</td><td>83.83</td><td>一</td><td>31.35</td><td>76.20</td><td>14445.62</td><td>3901.28</td><td>67.36</td><td>11.06</td></tr><tr><td>SegMamba-V2</td><td>Mamba</td><td>86.07</td><td>一</td><td>26.13</td><td>79.03</td><td>12980.15</td><td>4903.85</td><td>138.77</td><td>14.08</td></tr><tr><td>OcTreeMamba_Bot</td><td>CNN+SSM</td><td>85.47</td><td></td><td>27.45</td><td>78.23</td><td>5539.05</td><td>2198.35</td><td>34.71</td><td>42.61</td></tr><tr><td>OcTreeMamba_Enc</td><td>CNN+SSM</td><td>86.07</td><td></td><td>22.11</td><td>78.98</td><td>14011.68</td><td>5438.26</td><td>40.49</td><td>25.00</td></tr></table>

active leaf v then retrieves its enhanced parent context as $z _ { u } =$ $\tilde { h } _ { v  u } [ \vartheta ( v ) ]$ ], where $\vartheta ( v )$ denotes its parent. The sparse output feature is computed as:

$$
[ \alpha _ { v } ^ { \downarrow } , \theta _ { v } ^ { \downarrow } ] = \rho ( \mathtt { M L P } ( x _ { v } ) ) , \hat { x } _ { v } = \mathrm { P r o j } \left( \theta _ { v } ^ { \downarrow } \odot \alpha _ { v } ^ { \downarrow } \odot z _ { u } \right] ) ,\tag{11}
$$

where $\alpha _ { v } ^ { \downarrow }$ and $\theta _ { v } ^ { \downarrow }$ are generated from $x _ { v } ,$ , and Proj(·) denotes the layer-normalized linear projection. The enhanced sparse feature set is denoted as $\mathbf { X } ^ { ( s p a r s e ) } = \{ \hat { x } _ { v } \mid v \in \mathcal { V } \}$

To integrate the sparse tree representations back into the dense backbone, we exploit a zero-initialized residual fusion strategy [41]. The enhanced sparse features are scattered back to the dense grid and fused with the original input:

$$
\mathbf { X } ^ { \left( o u t \right) } = \mathbf { X } ^ { \left( i n \right) } + \mathbf { W } \cdot \mathsf { U p } \big ( \mathsf { S c a t t e r } \big ( \mathbf { X } ^ { \left( s p a r s e \right) } \big ) \big ) ,\tag{12}
$$

where W denotes learnable parameters, which are initialized to 0. Such initialization ensures the module initially acts as an identity mapping. During the training process, W receives non-zero gradients when the sparse branch provides information that minimizes the loss, effectively acting as a “soft gate” to filter out early-stage noise. The fused tensor $\mathbf { X } ^ { ( o u t ) }$ , now enriched with global topological semantics, is processed by the Dense Downsampling. Finally, the fused tensor $\mathbf { X } ^ { ( o u t ) }$ enriched with topological-aware context features is used to predict voxel-wise semantic segmentation results.

## E. Loss Function

To optimize our model, we designed a hybrid loss function that simultaneously penalizes pixel-wise segmentation errors and discrete topological structural deviations:

$$
\mathcal { L } = \mathcal { L } _ { s e g } + \lambda _ { s t r u c t } \mathcal { L } _ { s t r u c t } + \lambda _ { a u x } \mathcal { L } _ { a u x } ,\tag{13}
$$

where $\lambda _ { s t r u c t }$ and $\lambda _ { a u x }$ are loss weights for the structural sparsity and auxiliary deep supervision loss terms, respectively. Specifically, $\mathcal { L } _ { s e g }$ represents the cross-entropy loss applied to the final network output. $\mathcal { L } _ { a u x }$ provides auxiliary deep supervision directly to the sparse module before the outcome is fused back into the dense backbone. This auxiliary term is crucial for the Octree-SSM module, as it forces the sparse branch to rapidly capture the intrinsic topological complexities independently, without solely relying on the gradients backpropagated from the dense decoder.

## IV. EXPERIMENTS

## A. Datasets and Evaluation Metrics

We performed experiments on four diverse 3D biomedical segmentation benchmarks, including ImageCAS [42] for coronary artery segmentation, TopCoW24 [43] for circle of willis segmentation, AbdomenCT [44] for multi-organ abdominal segmentation, and BraTS21 [45] for brain tumor segmentation. Segmentation performance is assessed using Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and 95% Hausdorff Distance (HD95). For the topologysensitive vascular datasets, centerline Dice (clDice) [46] is additionally reported. Computational efficiency is measured by the parameter count, throughput (TP, patches/s), training and inference memory usage (TM/IM, MiB).

![](images/df9f14c00a822f026c965a5b9c0fa15b77fd899f40d8d8fddf9ca3e8d8b10fda.jpg)  
Fig. 3. Quantitative comparison with state-of-the-art methods on four datasets.

## B. Implementation Details

We built our model upon the nnUNet framework and implemented two versions of our model, i.e., OcTreeMamba Bot and OcTreeMamba Enc by setting the octree encoder in the bottleneck only and full encoder stages, respectively. The models are optimized using SGD with an initial learning rate of 2e-3, a weight decay of 3e-5, and a differential polynomial learning-rate decay strategy with a power of 0.9. The learning rate of the CNN backbone is set to half that of the sparse branch. The loss weights are set to $\lambda _ { \mathrm { s t r u c t } } { = } 0 . 1$ and $\lambda _ { \mathrm { a u x } } { = } 0 . 5$ We trained OcTreeMamba Bot and OcTreeMamba Enc for 500 and 1,000 epochs, respectively. All experiments are conducted on a workstation with four NVIDIA RTX 4090 GPUs.

## C. Main Results

We compare our method with state-of-the-art models across three paradigms: (1) CNNs (including nnU-Net [7], Med-NeXt [47]); (2) Vision Transformers (UNETR++ [48], Seg-Former3D [49]); and (3) Dense SSMs (U-Mamba [25]). As shown in Table I, OcTreeMamba demonstrates better performance on topology-sensitive vascular segmentation while maintaining competitive results on tumor and multi-organ benchmarks. On the ImageCAS dataset, OcTreeMamba Bot outperforms all compared methods, improving over nnU-Net by 0.77 Dice, 1.07 clDice, and 1.07 IoU, while reducing HD95 from 16.03 to 12.88. It indicates that bottleneck-level octree modeling is particularly effective for thin coronary artery structures without substantially increasing memory cost. On the TopCoW2024 dataset, OcTreeMamba Enc achieves the best performance, improving Dice by 1.6 and reducing HD95 by 5.45 compared with nnU-Net. On BraTS2021 and AbdomenCT, OcTreeMamba shows competitive accuracy compared with CNN, Transformer, and Mamba baselines. In particular, OcTreeMamba Bot offers a better memoryperformance balance, achieving substantially lower memory cost and higher throughput than dense Mamba-based methods while preserving comparable segmentation accuracy.

The qualitative comparisons in Fig. 3 further visualize these observations. On ImageCAS and TopCoW2024, Oc-TreeMamba produces more continuous vascular branches with fewer isolated false positives and class inconsistencies. On AbdomenCT, it better preserves both large-organ boundaries and small anatomical structures, while on BraTS2021 it maintains a coherent tumor shape comparable to the strongest baselines. These results demonstrate that the proposed hierarchical sparse modeling is effective across both curvilinear vascular structures and compact volumetric targets.

## D. Ablation Study

We conducted ablation studies on the ImageCAS dataset to further analyze the effectiveness of OcTreeMamba, including the sparse traversing operator and different configurations of the Octree Encoder. Table II compares different operators within the octree-based traversing framework under both bottleneck-only and encoder-wide configurations, while Fig. 4 further evaluates the interaction between placement and learning rate scaling across different datasets. These studies aim to identify an effective accuracy-efficiency trade-off and examine how octree-based sparse modeling behaves under different architectural and optimization configurations.

a) Ablation on the proposed OcTreeMamba block: We ablate different feature aggregation strategies of OctreeMamba. As shown in Table II, by replacing the feature aggregation with a point-wise MLP or convolution, the bottleneck version of baseline models improves upon the nnU-Net baseline, indicating that the octree-based sparse representation is more effective for modelling hierarchical anatomical structures. The performance drop of (+Conv) with 73.24 Dice may suggest convolutional operations lack an adequate receptive field for modelling long-range dependencies. In contrast, Self-Attention and our Tree-SSM, achieved superior performance. Specifically, Octree+Tree-SSM achieves superior performance on regional overlap of segmentation mask (Dice: 82.31), topological completeness (clDice: 87.67), and IoU (70.23). Although Self-Attention obtains the lowest HD95, Tree-SSM provides a more balanced performance across overlap, topology structure preservation, and efficiency-related metrics.

TABLE II  
ABLATION STUDY OF DIFFERENT SETTINGS OF OCTREEMAMBA BLOCK ON THE IMAGECAS DATASET.
<table><tr><td rowspan="2">Component</td><td rowspan="2">Configuration</td><td colspan="4">Segmentation Accuracy</td><td colspan="4">Efficiency / Complexity</td></tr><tr><td>Dice↑</td><td>clDice↑</td><td>HD95↓</td><td>IoU↑</td><td>TM↓</td><td>M↓</td><td>Param.↓</td><td>TP↑</td></tr><tr><td>nnU-Net</td><td>一</td><td>81.54</td><td>86.60</td><td>16.03</td><td>69.16</td><td>5563.40</td><td>2706.91</td><td>30.79</td><td>42.55</td></tr><tr><td> ${ \mathrm { O c t r e e } } + { \mathrm { M L P } }$ </td><td>Bottleneck</td><td>81.99</td><td>87.36</td><td>12.29</td><td>69.78</td><td>5942.70</td><td>2775.52</td><td>34.95</td><td>36.66</td></tr><tr><td> $\mathrm { O c t r e e } + \mathrm { C o n v }$ </td><td>Bottleneck</td><td>81.93</td><td>87.28</td><td>12.52</td><td>69.71</td><td>5932.17</td><td>2763.18</td><td>34.03</td><td>37.07</td></tr><tr><td> $\mathrm { O c t r e e } + \mathrm { S e l f } { - } \mathrm { A t t n }$ </td><td>Bottleneck</td><td>82.20</td><td>87.66</td><td>11.70</td><td>70.08</td><td>5937.81</td><td>2768.66</td><td>34.44</td><td>36.40</td></tr><tr><td> $\mathbf { O c t r e e } + \mathbf { T r e e } { \mathbf { - S S M } }$ </td><td>Bottleneck</td><td>82.31</td><td>87.67</td><td>12.88</td><td>70.23</td><td>5942.21</td><td>2771.79</td><td>34.78</td><td>36.57</td></tr><tr><td> ${ \mathrm { O c t r e e } } + { \mathrm { M L P } }$ </td><td>Encoder</td><td>81.63</td><td>87.18</td><td>11.87</td><td>69.24</td><td>11028.60</td><td>3034.45</td><td>41.05</td><td>26.51</td></tr><tr><td> $\mathrm { O c t r e e } + \mathrm { C o n v }$ </td><td>Encoder</td><td>73.24</td><td>79.09</td><td>18.54</td><td>58.32</td><td>9227.00</td><td>3005.21</td><td>38.41</td><td>26.33</td></tr><tr><td> $\mathrm { O c t r e e } + \mathrm { S e l f } { - } \mathrm { A t t n }$ </td><td>Encoder</td><td>82.13</td><td>87.40</td><td>12.73</td><td>69.98</td><td>12075.42</td><td>3027.99</td><td>39.58</td><td>24.37</td></tr><tr><td> $\mathbf { O c t r e e } + \mathbf { T r e e } { \mathbf { - S S M } }$ </td><td>Encoder</td><td>81.77</td><td>87.14</td><td>13.59</td><td>69.48</td><td>14957.68</td><td>3042.41</td><td>40.57</td><td>24.80</td></tr></table>

![](images/3bed78526910d0bd9513c427760df08ee1c47886505dec561852711546b8f703.jpg)

Fig. 4. Ablation on different configurations of OcTreeMamba modules and learning rate strategy.  
![](images/a8a17c579a38f54a5c5aa0f127f9b4410cac8ea4d1d92bb3f731444965a27248.jpg)  
Fig. 5. Memory-performance analysis on training and inference memory cost(TM, IM) and 3D segmentation performance (Dice) on ImageCAS.

b) Ablation on different module configurations and learning rate strategy: Fig. 4 illustrates the impact of different model configurations (i.e., Bottleneck vs. Encoder) and the different learning rate ratio $( \gamma = 2 \ \nu s . \ \gamma = 5 )$ . For module configuration, distributing the sparse OcTreeMamba modules across the encoder (Encoder) yields a Dice improvement on the multi-organ AbdomenCT dataset (86.07 vs. 85.72), but it decreases performance on ImageCAS. The superiority of OcTreeMamba Enc on the AbdomenCT dataset indicates that multi-scale context is beneficial for capturing complex spatial relationships of large organs. Conversely, for the sparse vascular dataset, i.e., ImageCAS, applying sparse long-range modelling in the bottleneck layer is more effective than introducing it into high-resolution shallow layers. We believe that for curvilinear targets (i.e., ImageCAS, TopCoW2024), avoiding early-stage global interactions prevents interference from low-level background noise while maintaining higher computational efficiency. For the Learning Rate Strategy, increasing the ratio to $\gamma ~ = ~ 5$ provides metric gains on BraTS2021 under the Bottleneck setting. However, it shows performance drops on the AbdomenCT. Therefore, $\gamma \ : = \ : 2$ is adopted as the default setting in our experiment. Overall, OcTreeMamba Bot and OcTreeMamba Enc are retained as two complementary configurations for different anatomical structures and hardware constraints.

## E. Efficiency Analysis

As shown in Table I, OcTreeMamba Bot achieves a favorable memory-performance balance, maintaining competitive or superior accuracy with fewer resources than dense Mambabased models. Across the four datasets, SegMamba requires $2 . 6 { \sim } 2 . 9 \times$ higher training memory and 1.6∼1.8× higher inference memory, while delivering $3 . 7 { \sim } 3 . 9 \times$ lower throughput. SegMamba-V2 further uses about 4.0× more parameters with lower throughput. Compared with U-Mamba Bot, OcTreeMamba Bot reduces training memory by 40∼47% and provides 1.8∼2.0× higher throughput. Fig. 5 illustrates this balance. These gains come from applying sequence modeling to Octree-based sparse representations rather than dense 3D feature maps. Although SegFormer3D achieves the lowest memory cost and highest throughput, it suffers from substantial accuracy degradation on challenging datasets, especially TopCoW2024. In contrast, OcTreeMamba Bot preserves strong accuracy with only a small memory increase over nnU-Net, while OcTreeMamba Enc requires higher memory cost but provides stronger multi-scale modeling.

## V. CONCLUSION

In this paper, we propose OcTreeMamba, a novel framework for 3D medical image segmentation by introducing a hierarchical Octree structure for feature propagation along tree topology in the SSM. Specifically, a structure learner is proposed to enable differentiable, anatomy-adaptive sparsification using Gumbel-Softmax sampling and hierarchical gating. A Dynamic Octree Builder is designed to organize the selected anatomical regions into a hierarchical Octree representation. A bi-directional Tree-SSM is developed to facilitate Leaf-Parent-Root propagation along the OcTree-Topology-aware trajectory aligned with Octree connectivity. Extensive evaluation across four datasets demonstrates that OcTreeMamba achieves competitive segmentation accuracy and state-of-theart performance in preserving complex vessel topology on vascular datasets and exhibits a highly competitive memoryperformance balance for large-scale organ segmentation on dense anatomical targets while maintaining a low GPU memory consumption.

## REFERENCES

[1] G. Litjens, T. Kooi et al., “A survey on deep learning in med. image anal.” Med. Image Anal., vol. 42, pp. 60–88, 2017.

[2] A. Esteva, A. Robicquet et al., “A guide to deep learning in healthcare,” Nature medicine, vol. 25, no. 1, pp. 24–29, 2019.

[3] M. Antonelli, A. Reinke et al., “The medical segmentation decathlon,” Nature communications, vol. 13, no. 1, p. 4128, 2022.

[4] O. Ronneberger, P. Fischer et al., “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015, pp. 234–241.

[5] O. Oktay, J. Schlemper et al., “Attention u-net: Learning where to look for the pancreas,” CoRR, vol. abs/1804.03999, 2018.

[6] O. C¸ ic¸ek, A. Abdulkadir<sup>¨</sup> et al., “3d u-net: learning dense volumetric segmentation from sparse annotation,” in MICCAI, 2016, pp. 424–432.

[7] F. Isensee, Jaeger et al., “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.

[8] H. H. Lee, S. Bao et al., “3d ux-net: A large kernel volumetric convnet modernizing hierarchical transformer for medical image segmentation,” in ICLR, 2023, pp. 1–15.

[9] Z. Zhou, M. M. R. Siddiquee et al., “Unet++: A nested u-net architecture for medical image segmentation,” in MICCAI, 2018, pp. 3–11.

[10] H. Huang, L. Lin et al., “Unet 3+: A full-scale connected unet for medical image segmentation,” in ICASSP, 2020, pp. 1055–1059.

[11] K. He, X. Zhang et al., “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.

[12] J. Dai, H. Qi et al., “Deformable convolutional networks,” in ICCV, 2017, pp. 764–773.

[13] C. Peng, X. Zhang et al., “Large kernel matters–improve semantic segmentation by global convolutional network,” in CVPR, 2017, pp. 4353–4361.

[14] X. Wang, R. Girshick et al., “Non-local neural networks,” in CVPR, 2018, pp. 7794–7803.

[15] A. Dosovitskiy, L. Beyer et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[16] Z. Liu, Y. Lin et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” in ICCV, 2021, pp. 10 012–10 022.

[17] J. Chen, Y. Lu et al., “Transunet: Transformers make strong encoders for medical image segmentation,” CoRR, vol. abs/2102.04306, 2021.

[18] H. Cao, Y. Wang et al., “Swin-unet: Unet-like pure transformer for medical image segmentation,” in ECCV, 2023, pp. 205–218.

[19] A. Hatamizadeh, V. Nath et al., “Swin UNETR: swin transformers for semantic segmentation of brain tumors in MRI images,” in MICCAI, vol. 12962, 2021, pp. 272–284.

[20] A. Lin, B. Chen, J. Xu et al., “Ds-transunet: Dual swin transformer u-net for medical image segmentation,” IEEE Trans. Instrum. Meas., vol. 71, pp. 1–15, 2022.

[21] A. Gu, Modeling Sequences with Structured State Spaces. Stanford University, 2023.

[22] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” CoRR, vol. abs/2312.00752, 2023.

[23] K. Li, X. Li et al., “Videomamba: State space model for efficient video understanding,” in ECCV. Springer, 2024, pp. 237–255.

[24] E. Nguyen, K. Goel et al., “S4nd: Modeling images and videos as multidimensional signals with state spaces,” in NeurIPS, vol. 35, 2022, pp. 2846–2861.

[25] J. Ma, F. Li et al., “U-mamba: Enhancing long-range dependency for biomedical image segmentation,” CoRR, vol. abs/2401.04722, 2024.

[26] L. Zhu, B. Liao et al., “Vision mamba: Efficient visual representation learning with bidirectional state space model,” 2024.

[27] Y. Liu, Y. Tian et al., “Vmamba: Visual state space model,” in NeurIPS, vol. 37, 2024, pp. 103 031–103 063.

[28] K. Kamnitsas, C. Ledig et al., “Efficient multi-scale 3d cnn with fully connected crf for accurate brain lesion segmentation,” Med. Image Anal., vol. 36, pp. 61–78, 2017.

[29] O. Bernard, A. Lalande et al., “Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: Is the problem solved?” IEEE TMI, vol. 37, no. 11, pp. 2514–2525, 2018.

[30] F. Milletari, N. Navab et al., “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in 3DV, 2016, pp. 565–571.

[31] X. Xiao, S. Lian et al., “Weighted res-unet for high-quality retina vessel segmentation,” in ITME, 2018, pp. 327–331.

[32] D. Liu, Y. Zhang et al., “Multi-directional attention: a lightweight attention module for slender structures,” Intelligence & Robotics, vol. 5, no. 4, pp. 827–843, 2025.

[33] A. Hatamizadeh, Y. Tang et al., “Unetr: Transformers for 3d medical image segmentation,” in WACV, 2022, pp. 574–584.

[34] A. Vaswani, N. Shazeer et al., “Attention is all you need,” in NeurIPS, 2017, pp. 5998–6008.

[35] A. Gu, K. Goel et al., “Efficiently Modeling Long Sequences with Structured State Spaces,” in ICLR, 2022.

[36] A. Gu, I. Johnson et al., “Combining recurrent, convolutional, and continuous-time models with linear state space layers,” in NeurIPS, vol. 34, 2021, pp. 572–585.

[37] A. Gu, T. Dao et al., “Hippo: Recurrent memory with optimal polynomial projections,” in NeurIPS, vol. 33, 2020, pp. 1474–1487.

[38] Z. Xing, T. Ye et al., “Segmamba-v2: Long-range sequential modeling mamba for general 3d medical image segmentation,” IEEE TMI, vol. 45, no. 1, pp. 4–15, 2025.

[39] T. Huang, X. Pei et al., “Localmamba: Visual state space model with windowed selective scan,” in ECCV. Springer, 2024, pp. 12–22.

[40] Y. Xiao, L. Song et al., “MambaTree: Tree topology is all you need in state space model,” in NeurIPS, vol. 37, 2024, pp. 75 329–75 354.

[41] L. Zhang, A. Rao et al., “Adding conditional control to text-to-image diffusion models,” in ICCV, 2023, pp. 3836–3847.

[42] A. Zeng, C. Wu et al., “Imagecas: A large-scale dataset and benchmark for coronary artery segmentation based on computed tomography angiography images,” CoRR, 2022.

[43] K. Yang, F. Musio et al., “The TopCoW challenge—topology-aware circle of willis segmentation for CT and MR angiography,” NEJM AI, vol. 3, no. 8, 2026.

[44] J. Ma, Y. Zhang et al., “Abdomenct-1k: Is abdominal organ segmentation a solved problem?” IEEE TPAMI, vol. 44, no. 10, pp. 6695–6714, 2022.

[45] U. Baid, S. Ghodasara et al., “The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification,” CoRR, vol. abs/2107.02314, 2021.

[46] S. Shit, J. C. Paetzold et al., “cldice-a novel topology-preserving loss function for tubular structure segmentation,” in CVPR, 2021, pp. 16 560– 16 569.

[47] S. Roy, G. Koehler et al., “Mednext: transformer-driven scaling of convnets for medical image segmentation,” in MICCAI. Springer, 2023, pp. 405–415.

[48] A. Shaker, M. Maaz et al., “Unetr++: delving into efficient and accurate 3d medical image segmentation,” IEEE Transactions on Medical Imaging, vol. 43, no. 9, pp. 3377–3390, 2024.

[49] S. Perera, P. Navard, and A. Yilmaz, “Segformer3d: an efficient transformer for 3d medical image segmentation,” in CVPRW. IEEE, 2024, pp. 4981–4988.