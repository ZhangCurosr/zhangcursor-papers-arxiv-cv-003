# UniRestorer++: Scalable Universal Image Restoration with Multi-Granularity Degradation Modeling

Jingbo Lin, Zhilu Zhang, Fan Li, Renjing Pei, Wenbo Li, Hang Xu, Hongzhi Zhang, and Wangmeng Zuo

Abstract—Universal image restoration aims to handle diverse degradations within a single model. Despite rapid progress, a reliable scaling strategy remains underexplored. The conventional strategy of simply increasing backbone depth and width can exhibit performance saturation, with optimization conflicts across restoration objectives remaining a potential bottleneck. Degradation-aware methods allocate specialized capacity, but finer specialization requires more accurate degradation estimation and becomes vulnerable to estimation errors, especially on out-of-distribution and real-world inputs. To this end, we propose UniRestorer++, a scalable universal image restoration framework driven by multi-granularity degradation modeling. Specifically, we construct a degradation hierarchy for model scaling by progressively refining the degradation group and introducing group-specific restoration experts, thereby coupling capacity growth with degradation-specific specialization. To ensure reliable scaling, degradation-estimation-based routing selects a root-to-leaf path within the hierarchy, while granularity-estimation-based routing selects an appropriate level along that path, balancing fine-grained specialization with robustness against degradation-estimation errors. Furthermore, spatially dense degradation representation maps enable regionaware expert routing to adaptively handle spatially non-uniform corruptions. We develop the CDD-50 benchmark, encompassing 50 degradation categories ranging from single-degradation to penta-degradation. Extensive experiments show approximately log-linear PSNR trends over the evaluated model-size range, with stronger scaling gains on out-of-distribution and real-world data than vanilla MoE scaling. UniRestorer++ achieves stateof-the-art performance across comprehensive benchmarks. It also demonstrates zero-shot generalization and few-shot transfer capability on real-world images. The code and pre-trained models will be publicly available at https://github.com/mrluin/ UniRestorer plusplus.

## I. INTRODUCTION

MAGE restoration aims to recover high-quality visual content from degraded observations, supporting applications in computational photography [1], medical imaging [2], remote sensing [3], autonomous driving [4], and security surveillance [5]. Early methods were typically tailored to individual degradations, e.g., DnCNN [6] for denoising and SRCNN [7] for super-resolution. As degradations diversified and neural architectures advanced, research expanded from task-specific methods to task-agnostic restoration backbones, including MPRNet [8], SwinIR [9], Restormer [10], NAFNet [11], and X-Restormer [12]. More recently, universal image restoration has emerged to address multiple restoration tasks and diverse degradation types within a single unified model.

![](images/deb631c7403598a26a337f04c9906184b011c3e65e039dee6053a55d758de001.jpg)  
Fig. 1. Representative universal image restoration frameworks. (a) Degradation-agnostic methods use a shared backbone without explicit degradation conditioning. Degradation-aware methods guide restoration through conditioning (b) or expert-based processing (c) that routes features toward degradation-specific experts. These representative methods typically use a fixed degradation granularity, which can make them sensitive to degradationestimation errors. (d) UniRestorer++ proposes multi-granularity degradation modeling, expanding model capacity by introducing finer degradation-specific restoration experts. Degradation-estimation-based routing determines a hierarchical path, while granularity-estimation-based routing selects an expert level along that path, balancing degradation-specific specialization with robustness to degradation-estimation errors.

Existing universal restoration methods can be broadly categorized into degradation-agnostic and degradation-aware methods. Degradation-agnostic methods jointly train a shared backbone on data across all tasks (see Fig. 1(a)), but conflicting restoration objectives (such as denoising and deblurring) can hinder restoration quality. Degradation-aware methods use degradation information to guide restoration and can be further divided into conditioning-based and expert-based approaches. Conditioning-based methods use learnable prompts [13]–[16], degradation representations [17], [18], or adapters [19] to modulate a shared restoration backbone (see Fig. 1(b)). Expertbased methods allocate degradation-specific capacity through specialist models [20], [21], selective filters [22], [23], lowrank adaptation (LoRA) experts [24], routing features toward specialized pathways (see Fig. 1(c)).

Despite rapid progress in universal restoration, existing methods still exhibit a noticeable performance gap compared to specialist models, particularly when confronted with an expanding and diverse degradation space. Drawing inspiration from advances in large language models [25], [26] and largescale vision models [27]–[29], scaling up the capacity of universal restoration models represents a promising direction to close this gap and potentially surpass specialist models. However, simply increasing the depth and width of a dense restoration network yields limited gains in our experiments (see the upper part of Fig. 2), as it does not explicitly allocate capacity to different degradation subspaces, leading to conflicts among different restoration objectives. To reduce parameter interference, degradation-aware methods use conditioning or expert specialization to adapt restoration processing to different degradation subspaces. Nevertheless, the efficacy of these methods critically hinges on degradation estimation, where the finer parameter specialization required for model scaling demands higher estimation accuracy. For out-of-distribution and real-world degradations, degradation-estimation errors can activate mismatched parameters and compromise the benefit of increased specialization (see ‘Vanilla MoE Scaling’ in the lower part of Fig. 2). In this paper, we argue that a scalable universal restoration framework must ensure that capacity expansion effectively translates into both predictable performance and generalization gains. Achieving this goal fundamentally requires scaling up the model while maintaining robustness against degradation-estimation errors.

![](images/2ec26499ddedbdad9556fc5ea3ea7dff67c2fd0345c573d262362209b32f05b9.jpg)  
Dense Model Scaling Ours Hierarchy Model Scaling Vanilla MoE Model Scaling

![](images/e35d1f9faa4c9a6235fa4a02ea72619f18b34ab99364983d1c71fc8d27906613.jpg)  
Fig. 2. Model scaling evaluation on our CDD-50 and FoundIR [30] datasets. The equations give empirical $\mathrm { P S N R } = a + b \log _ { 1 0 } P _ { \mathrm { t o t a l } }$ fits for our hierarchy model scaling and vanilla MoE model scaling. See Section IV-D for more details.  
Fig. 3. Universal restoration performance on single-degradation tasks, CDD-11, CDD-50, and real-world datasets.

To address these challenges, we present UniRestorer++, a scalable universal image restoration framework driven by multi-granularity degradation modeling, as illustrated in Fig. 1(d). First, we construct a degradation hierarchy to govern the capacity scaling of universal restoration models. Each subsequent hierarchy level refines the degradation space of the preceding level into finer groups and introduces groupspecific restoration experts for the resulting subspaces. The expansion of model capacity is thereby achieved by increasing the depth of this hierarchy. Second, while deeper hierarchies provide more specialized capacity, they can also amplify the consequence of degradation-estimation errors. To enable reliable scaling, we employ a multi-granularity expert routing mechanism that balances expert specialization with robustness against degradation-estimation errors. Specifically, finegrained experts specialize in narrow degradation subspaces, whereas coarse-grained experts cover broader distributions to improve robustness under ambiguous estimates. Degradationestimation-based routing determines a root-to-leaf path within the hierarchy, while granularity-estimation-based routing selects an appropriate level along that path. For instance, an ambiguous estimate of heavy snow can be routed to a coarsegrained general-snow-removal expert, whereas a confident estimate can activate a fine-grained heavy-snow-removal expert. Third, existing methods generally use an image-level degradation representation to condition restoration, without explicitly selecting different experts for individual spatial locations. However, haze density, blur patterns, and many realworld degradations often vary spatially. To address this issue, our method predicts spatially dense degradation representation maps for region-aware expert routing, allowing different experts to be selected for distinct regions in an image.

We conduct extensive experiments across multiple evaluation settings. We develop the CDD-50 benchmark, which encompasses 50 distinct degradation categories ranging from single corruptions to complex mixtures of up to five simultaneous degradations. Experimental results demonstrate that UniRestorer++ consistently establishes state-of-the-art performance across in-distribution, out-of-distribution, and realworld degradation benchmarks. Crucially, the proposed hierarchical model scaling strategy exhibits approximately log-linear PSNR trends over the evaluated model-size range, with steeper fitted slopes than vanilla mixture-of-experts (MoE) scaling on out-of-distribution CDD-50 and FoundIR, as shown in Fig. 2. By scaling the parameter capacity up to 2.9B, we establish the largest known fidelity-oriented universal restoration model, which demonstrates fine zero-shot generalization and few-shot transfer capabilities on real-world images.

A preliminary version of this work was published in the Proceedings of ICLR 2026 [31]. Methodologically, the present manuscript extends the original multi-granularity routing mechanism into a scalable universal restoration framework by establishing a hierarchical model scaling method and generalizing global routing to spatially adaptive expert routing. Empirically, this work constructs a comprehensive CDD-50 evaluation benchmark, while providing investigations of scaling behavior, out-of-distribution generalization, and few-shot transferability.

The contributions are summarized as follows:

• We propose a hierarchy model scaling method for universal image restoration, which progressively refines the degradation partition and assigns restoration experts to

the resulting subspaces.

• We propose a multi-granularity expert routing mechanism, which balances expert specialization with robustness against degradation-estimation errors.

• We present a region-aware expert routing scheme, allowing local regions to dynamically activate distinct restoration experts for spatially non-uniform degradations.

• We establish the CDD-50 benchmark encompassing 50 degradation categories. Extensive experiments characterize the empirical log-linear scaling trend over the evaluated model-size range and demonstrate superior performance in universal restoration, real-world zero-shot generalization, and few-shot transferability.

## II. RELATED WORK

## A. Image Restoration for Specific Degradations

Early image restoration methods developed architectures tailored to individual degradations, $e . g .$ ., DnCNN [6] for denoising, SRCNN [7] for super-resolution, DeblurGAN-v2 [32] for deblurring, PReNet [33] for deraining, Dehazeformer [34] for dehazing, and DesnowNet [35] for desnowing. Advances in deep architectures produced general-purpose restoration backbones, including MPRNet [8], SwinIR [9], NAFNet [11], Restormer [10], and X-Restormer [12]. These backbones can be trained separately for different restoration tasks and provide specialist baselines.

## B. Universal Image Restoration

Overview. Universal image restoration aims to process multiple known or unknown degradation types within a single model. Existing approaches fall into two categories: degradation-agnostic and degradation-aware. Degradationagnostic methods share all model weights and learn cross-task knowledge implicitly from mixed training data, which suffer from restoration-task conflicts. Degradation-aware methods instead estimate degradation representations (DRs) and use them as priors, following the paradigm of blind image restoration [36], [37]. Within this category, methods can be further divided into conditioning-based and expert-based approaches according to how the degradation priors guide restoration.

Conditioning-based Methods. Conditioning-based degradation-aware methods [13], [14], [16]–[19] integrate prompting [38], adapters [39], or feature modulation [40] into a shared restoration backbone. Such mechanisms can support parameter-efficient fine-tuning (PEFT), but their use alone does not imply that the backbone is frozen. TransWeather [14] and AirNet [17], $e . g .$ ., learn degradation-related embeddings that modulate intermediate features for different corrupted inputs. PromptIR [13] combines prompt learning with a Restormer backbone [10], whereas IDR [18] integrates taskoriented prior hubs through learnable principal component analysis and dynamic routing. MiOIR [16] combines explicit and adaptive prompts, while OneRestore [15] learns textaligned prompts for composite-degradation restoration. However, the degradation-specific processing they induce can be sensitive to errors in the conditioning representation, especially over a wide and complex degradation space.

Expert-based Methods. Mixture-of-experts models have demonstrated conditional processing capability in multi-task learning [41]–[43]. Expert-based degradation-aware methods implement specialization at different scales, from lightweight degradation-specific modules to full restoration networks. $E . g .$ WGWSNet [23] and ADMS [22] learn specific filters on top of a shared backbone. DaAIR [44] uses a degradation-aware learner in a low-rank regime, Instruct-IPT [45] learns low-rank task-specific weight offsets, and LoRA-IR [24] uses low-rank experts. MoCE-IR [46] learns experts with different computational complexities and receptive fields. At the full-model level, GRIDS [21] groups degradations and trains restoration networks for the resulting groups. The related RestoreAgent system [20] instead uses a multimodal language model to select restoration tools and their execution order.

Despite this progress, degradation-aware approaches typically select prompts or experts at a fixed degradation granularity. Although such selection enables degradation-specific processing, it relies on degradation-estimation accuracy and is therefore vulnerable to degradation-estimation errors. Moreover, many such approaches use image-level degradation cues without explicitly selecting different experts for individual spatial locations. Spatially varying feature responses alone do not provide this form of region-aware expert selection.

## C. Scaling Universal Image Restoration Models

Larger models and training sets have improved capability and generalization in natural language processing and highlevel vision tasks. Beyond dense scaling of model depth and width, sparse MoE models enlarge model capacity while activating only a subset of parameters for each input [47]–[49].

Comprehensive model scaling remains underexplored in universal image restoration. Existing work has improved restoration capacity through higher-capacity backbones [10], [12], [50], pretraining on broader degradation spaces [51], [52], or larger high-quality datasets [53]. GenLV [54] studies scalable unified modeling across a wide spectrum of lowlevel vision tasks using three dense model variants from 39M to 211M parameters. Scaling laws have characterized empirical relationships between model capacity and performance in language models [55] and high-level vision models [27], [56]. Routed-model scaling laws additionally distinguish total capacity from active computation [55]. These studies do not directly address how parameters should be organized across heterogeneous image degradations. Consequently, whether universal image restoration exhibits a predictable scaling relationship and how parameter organization affects this relationship remain underexplored.

## III. PROPOSED METHOD

We first introduce the basic formulation of model scaling, then compare dense model scaling and vanilla MoE scaling to motivate our hierarchy model scaling strategy. We then detail the components supporting our hierarchy model scaling strategy. The overview of our method is shown in Fig. 4. We first learn dense degradation representations (DRs) and hierarchically cluster them to construct a multi-granularity degradation set. Then, we accordingly train a multi-granularity mixture-of-experts (MoE) restoration model. To keep increasingly fine-grained specialization robust to uncertain degradation estimates, the hierarchy retains experts at multiple granularities, allowing ambiguous inputs to be handled by broader degradation groups. The mechanism for selecting an appropriate granularity is introduced later. As a result, the dense DR maps and routing strategy enable adaptively assigning inputs to appropriate experts across spatial locations and throughout the network.

![](images/2778710810870b9a91b0e79abb38e138014c197217ce2cdf7781c01796a91f89.jpg)

(a) Multi-Granularity Degradation Set Construction  
![](images/fd19158624b1ad0c21d109687577599774fbbbf3f061f6a85371f492d27274c8.jpg)  
Fig. 4. Overview of UniRestorer++. Globally pooled degradation representations (DRs) are hierarchically clustered to construct a multi-granularity degradation set, while dense DR maps are retained for region-aware expert routing. At each restoration block, the degradation-estimation-based and granularity-estimationbased routers are conditioned on both dense DR priors and the current restoration feature, enabling expert routing to adapt throughout the network. Their joint decision produces an expert assignment at each spatial location.

## A. Preliminaries

Basic Formulation of Model Scaling. Let E denote the expected mean squared restoration error under a given evaluation distribution. Under ideal optimization without multitask interference, the scaling behavior of a monolithic neural network with parameter count $P$ is characterized by the floorplus-power-law relation [56]:

$$
\mathcal { E } ( P ) \simeq \mathcal { E } _ { \mathrm { f l o o r } } + c P ^ { - \alpha } ,\tag{1}
$$

where $c > 0$ denotes the reducible error coefficient, $\alpha > 0$ represents the fundamental power-law scaling exponent, and $\mathcal { E } _ { \mathrm { f l o o r } } \geq 0$ denotes the asymptotic error floor determined by intrinsic task ambiguity and evaluation constraints. The powerlaw exponent α serves as the foundational metric of parameter efficiency, quantifying the rate of error decay per order of magnitude of parameter expansion.

Mixture-of-Experts. A mixture-of-experts (MoE) architecture consists of an expert set $\{ \mathcal { F } _ { a } \} _ { a = 1 } ^ { N _ { \mathrm { e x p } } }$ and a routing network G. The expert set contains $N _ { \mathrm { e x p } }$ parallel modules, $e . g .$ , convolutional or feed-forward layers. Given an input x and its routing representation ${ \pmb v } , \mathcal { G } _ { a } ( { \pmb v } )$ denotes the contribution of the ath expert $\mathcal { F } _ { a } ( { \pmb x } )$ to the MoE output y,

$$
\pmb { y } = \sum _ { a = 1 } ^ { N _ { \mathrm { e x p } } } \mathcal { G } _ { a } ( \pmb { v } ) \mathcal { F } _ { a } ( \pmb { x } ) .\tag{2}
$$

One sparse routing formulation is a noisy top-1 router parameterized by $W _ { \mathrm { g a t e } }$ and $W _ { \mathrm { n o i s e } }$ . It applies softmax to the noisy routing logits and retains only the expert with the largest routing probability,

$$
\begin{array} { r l } & { \mathcal { G } ( \pmb { v } ) = \mathrm { T o p K } \Big ( \mathrm { S o f t m a x } \big ( { \pmb v } W _ { \mathrm { g a t e } } } \\ & { \qquad + \epsilon \odot \mathrm { S o f t p l u s } ( { \pmb v } W _ { \mathrm { n o i s e } } ) \big ) ; K \Big ) , } \end{array}\tag{3}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ is standard Gaussian noise, ⊙ denotes element-wise multiplication, TopK(·; 1) sets all but the largest element to zero, and Softplus(·) is a smooth approximation to ReLU.

## B. Scaling Universal Image Restoration Models

Dense Model Scaling Strategy. The conventional dense model scaling strategy expands the parameter count $P$ by uniformly increasing the depth and width of a shared restoration backbone. Because all degradation tasks share an unpartitioned parameter manifold, scaling P intensifies gradient interference and optimization conflicts among contradictory restoration objectives (such as high-frequency detail synthesis versus lowpass noise smoothing). Consequently, the effective scaling exponent decays rapidly as model size increases. Beyond a critical parameter scale, multi-task optimization conflicts dominate the optimization trajectory, driving the effective exponent to zero and eventually into negative values, which manifests empirically as performance saturation and subsequent collapse. Vanilla MoE Model Scaling Strategy. Sparse Mixture-of-Experts (MoE) architectures decouple parameter interference by routing visual representations to specialized sub-networks. Let $P _ { \mathrm { t o t a l } }$ denote the total parameter capacity of the network, and let $P _ { \mathrm { a c t } }$ denote the active parameter count. Under optimal expert assignment, the expected restoration error is governed by [55]:

$$
\mathcal { E } _ { \mathrm { M o E } } \simeq \mathcal { E } _ { \mathrm { f l o o r } } + c _ { 1 } P _ { \mathrm { a c t } } ^ { - \alpha _ { 1 } } P _ { \mathrm { t o t a l } } ^ { - \alpha _ { 2 } } + c _ { 2 } P _ { \mathrm { a c t } } ^ { - \alpha _ { 3 } } ,\tag{4}
$$

where $c _ { 1 } , c _ { 2 } , \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } > 0$ are structural scaling constants. Under a fixed active computation budget $P _ { \mathrm { { a c t } } }$ , the term $\alpha _ { 2 } > 0$ represents the theoretical upper-bound exponent achievable through total parameter expansion. In universal restoration, vanilla MoE scaling partitions the degradation space into increasingly fine, disjoint clusters governed by a flat router operating at a single granularity. However, as $P _ { \mathrm { t o t a l } }$ expands, the candidate expert space proliferates, rendering degradation estimation and expert selection progressively more difficult. Under ambiguous or out-of-distribution (OOD) degradations, escalating routing misassignments introduce estimation noise that degrades the theoretical capacity exponent $\alpha _ { 2 }$ into a substantially attenuated effective exponent $\alpha _ { \mathrm { e f f , v a n i l l a } } \ll \alpha _ { 2 }$ severely constraining the scaling return of total parameters.

Hierarchy Model Scaling Strategy. To mitigate the impact of degradation estimation error on the scaling exponent, we structure the degradation space into a sequence of nested partitions $\Pi _ { 0 } \prec \Pi _ { 1 } \prec \dots \prec \Pi _ { L - 1 }$ , where ≺ denotes partition refinement and L denotes the total number of hierarchy levels. Under a fixed active budget $P _ { \mathrm { a c t } } .$ , we model the effective restoration error for a scaling strategy κ ∈ {vanilla, hier} via an effective scaling exponent $\alpha _ { \mathrm { e f f } , \kappa }$

$$
\mathcal { E } _ { \kappa } ( P _ { \mathrm { t o t a l } } ) \simeq \mathcal { E } _ { \mathrm { f l o o r } } + C _ { \kappa } P _ { \mathrm { t o t a l } } ^ { - \alpha _ { \mathrm { e f f } , \kappa } } ,\tag{5}
$$

where $C _ { \kappa } > 0$ denotes the strategy-dependent effective error coefficient, and $\alpha _ { \mathrm { e f f } , \kappa } > 0$ characterizes the empirical scaling efficiency in the presence of routing dynamics. In vanilla MoE scaling, escalating routing errors under distribution shifts severely penalize the system, causing $\alpha _ { \mathrm { e f f , v a n i l l a } }$ to diminish sharply. In contrast, our hierarchy model scaling strategy enables ambiguous or OOD inputs to fall back dynamically to coarser ancestral experts. By safeguarding expert routing against estimation uncertainties across all scales, hierarchy model scaling preserves the theoretical potential of capacity growth, sustaining a high and stable scaling exponent $\alpha _ { \mathrm { e f f , h i e r } } > \alpha _ { \mathrm { e f f , v a n i l l a } }$ across both in-distribution and out-ofdistribution conditions.

For images normalized to $[ 0 , 1 ] , ~ - 1 0 \log _ { 1 0 } \mathcal { E } _ { \kappa }$ is the PSNR computed from the expected MSE. Under Eq. (5), a nonzero error floor generally makes this quantity nonlinear in $\log _ { 1 0 } P _ { \mathrm { t o t a l } }$ , its slope approaches $1 0 \alpha _ { \mathrm { e f f } , \prime }$ 5 only when the reducible error dominates the floor. The reported average PSNR instead averages image-wise logarithmic errors and is generally different from the PSNR of the expected MSE. We therefore fit the reported average PSNR directly, using the following log-linear relation as an empirical summary over the evaluated model-size range:

$$
\overline { { \mathrm { P S N R } } } _ { \kappa } ( P _ { \mathrm { t o t a l } } ) \simeq a _ { \kappa } ^ { \mathrm { f i t } } + b _ { \kappa } ^ { \mathrm { f i t } } \log _ { 1 0 } P _ { \mathrm { t o t a l } } ,\tag{6}
$$

where $\overline { { \mathrm { P S N R } } } _ { \kappa }$ denotes the reported average PSNR, $a _ { \kappa } ^ { \mathrm { f i t } }$ is the fitted intercept, and $b _ { \kappa } ^ { \mathrm { f i t } }$ measures the fitted PSNR gain per tenfold increase in total parameters. Under the negligiblefloor approximation, interpreting $b _ { \kappa } ^ { \mathrm { f i t } } \simeq 1 0 \alpha _ { \mathrm { e f f } , \kappa }$ additionally requires the $\mathrm { g a p }$ between the two PSNR aggregation conventions to remain approximately constant with model size. We use the measured PSNR and these fitted slopes to compare our hierarchy model scaling and vanilla MoE model scaling over the evaluated range (see Fig. 2 and Section IV-D).

## C. Degradation Hierarchy Construction

Dense Degradation Representation Extractor. Aiming to enable routing adaptive to spatially varying degradations, we first learn to extract dense DR maps. Inspired by DA-CLIP [57], we use text to describe both degradation type and degree and train the DR extractor by aligning DRs with their text embeddings. Using an image-level CLIP [58] representation, particularly after resizing or cropping high-resolution inputs, can discard local degradation cues. We therefore use a ResNet-18-based [59] dense DR extractor D. Given a lowquality image X, D predicts a dense DR map E with $p$ channels,

$$
\begin{array} { r } { \pmb { { \cal E } } = \mathcal { D } ( \mathbf { X } ) , } \end{array}\tag{7}
$$

where p denotes the DR dimension. We construct fine-grained degradation captions by refining each degradation type into degree intervals, $e . g .$ , for Gaussian noise, we divide $\sigma _ { \mathrm { n o i s e } } \in$ [0, 50] into [0, 15), [15, 35), and [35, 50], described as ‘small noise’, ‘medium noise’, and ‘large noise’, respectively. We encode these captions into text embeddings that serve as the supervision targets for DR learning. To provide pixel-wise supervision, we apply CutMix [60] between degraded images with different degradation types and degrees, producing spatially varying inputs and their pixel-wise type-and-degree labels. We optimize D using a pixel-wise contrastive loss that aligns each local DR with the text embedding of its corresponding degradation,

$$
\mathcal { L } _ { \mathrm { p i x } } = \frac { 1 } { | \Omega _ { d } | } \sum _ { r \in \Omega _ { d } } \ell _ { \mathrm { c o n } } \mathopen { } \mathclose \bgroup \left( E _ { r } , t _ { q ( r ) } \aftergroup \egroup \right) ,\tag{8}
$$

where $\ell _ { \mathrm { c o n } }$ denotes the contrastive loss, $q ( r )$ is the type-anddegree label at spatial location $r , \ t _ { q ( r ) }$ is its text embedding, and $\Omega _ { d }$ is the set of spatial locations in the dense DR map. The pixel-wise labels are aligned with this map for supervision. Consequently, E preserves local degradation characteristics instead of collapsing them into a single image-level representation. After training, D is applied to all $N _ { \mathrm { t r } }$ low-quality training samples, yielding dense DR maps $\{ E _ { n } \} _ { n = 1 } ^ { N _ { \mathrm { t r } } }$ . We apply global average pooling for multi-granularity degradation set construction,

$$
\begin{array} { r } { \pmb { e } _ { n } = \mathrm { G A P } ( \pmb { E } _ { n } ) \in \mathbb { R } ^ { p } , } \end{array}\tag{9}
$$

while retaining dense maps ${ \pmb E } _ { n }$ for routing.

Hierarchical Degradation Clustering. We recursively apply K-means clustering [61] to organize the pooled DRs into a tree with L levels. Let $\mathcal { S } _ { i , j }$ denote the set of training-sample indices in the j-th DR group at level $i \in \{ 0 , \ldots , L - 1 \}$ , and let $J _ { i }$ be the number of groups at that level. Thus, $\Pi _ { i } = \{ S _ { i , j } \} _ { j = 1 } ^ { J _ { i } } .$ Level 0 contains the single root group ${ \cal S } _ { 0 , 1 } = \{ 1 , \dots , { \tilde { \cal N } } _ { \mathrm { t r } } \}$ with $J _ { 0 } = 1$ , and level $L - 1$ is the finest. For each parent group $\boldsymbol { S } _ { i - 1 , m }$ , we recursively partition its samples by K-means,

$$
\begin{array} { l } { { \{ u _ { i , m , \nu } \} _ { \nu = 1 } ^ { b _ { i - 1 , m } } } } \\ { {  \mathrm { K - m e a n s } ( \{ e _ { n } \mid n \in { \cal S } _ { i - 1 , m } \} ; b _ { i - 1 , m } ) , } } \end{array}\tag{10}
$$

where $i \in \{ 1 , \ldots , L - 1 \}$ , m indexes parent groups, $b _ { i - 1 , m }$ denotes the number of resulting child groups, and $\mathbf { \Delta } \mathbf { u } _ { i , m , \nu }$ is the ν-th child centroid of parent group $\boldsymbol { S } _ { i - 1 , m }$ . Each sample is assigned to its nearest child centroid. The resulting child groups are indexed by j across level i, giving $J _ { i } \ =$ $\sum _ { m = 1 } ^ { J _ { i - 1 } } b _ { i - 1 , m }$ groups at that level. Recursively partitioning all parent groups yields non-overlapping DR groups at progressively finer granularities. These nested DR groups form the degradation hierarchy introduced in Section III-B. Extending it by one level refines the degradation partition and defines the subspaces to which new restoration experts are assigned.

## D. Multi-Granularity MoE Restoration Model

Hierarchical Restoration Experts. We implement the groupspecific restoration parameters introduced in Section III-B as low-rank experts on a shared restoration backbone ${ \mathcal { F } } _ { \mathrm { b a s e } } .$ The shared backbone captures knowledge common to the degradation space. For each DR group $\mathcal { S } _ { i , j }$ at level $i \geq 1$ , we learn a LoRA expert $\Delta \mathcal { F } _ { i , j }$ on top of the frozen base weights, allocating additional capacity to that degradation subspace. For block $s , \bar { \mathcal { F } } _ { \mathrm { b a s e } } ^ { ( s ) }$ denotes the base block, and $\Delta \mathcal { F } _ { i , j } ^ { ( s ) }$ denotes the LoRA residual of expert (i, j). We use rank-8 LoRA experts in our experiments. Given prediction Y<sup>ˆ</sup> and target Y, the loss is,

$$
\ell _ { 1 } = \| \hat { \mathbf Y } - \mathbf { Y } \| _ { 1 } .\tag{11}
$$

Coarse experts cover broad degradation subspaces and provide robust restoration under ambiguous estimations, whereas fine-grained experts specialize in narrower subspaces when degradation estimation is reliable. Adding a hierarchy level therefore expands specialized capacity without discarding the broader restoration capability represented by ancestor experts. Multi-Granularity Expert Routing. Multi-granularity expert routing includes two complementary parts: degradationestimation-based routing determines a root-to-leaf path, and granularity-estimation-based routing selects an appropriate level along that path. We abbreviate the corresponding degradation-estimation-based router and granularityestimation-based router as DER and GER, respectively. Following uncertainty learning [62], we model the DR at each spatial location as a Gaussian distribution with diagonal covariance to capture fine-grained degradation-estimation uncertainty. From $E ,$ two estimation heads predict,

$$
\begin{array} { l } { { E _ { d } = \phi _ { \mu } ( E ) , } } \\ { { E _ { g } = \phi _ { \sigma ^ { 2 } } ( E ) , } } \end{array}\tag{12}
$$

where $\phi _ { \mu }$ and $\phi _ { \sigma ^ { 2 } }$ are the mean and variance estimation heads, respectively. Both maps retain the spatial structure of E and have $p$ channels. The mean map $E _ { d }$ represents the estimated degradation, while the variance map $E _ { g }$ provides an uncertainty cue for GER.

Region-Aware Expert Routing. To enable region-aware expert routing, we preserve the spatial dimensions of the degradation priors and equip each restoration block with its own DER and GER. Let $\bar { \pmb { H } } ^ { ( s ) }$ be the input feature of the s-th block. The degradation-prior and restoration-feature branches of DER are denoted by $\hat { \mathcal { G } } _ { d , E } ^ { ( s ) }$ and $\mathcal { G } _ { d , H } ^ { ( s ) }$ , while those of GER are denoted by $\mathcal { G } _ { g , E } ^ { ( s ) }$ and $\mathcal { G } _ { g , H } ^ { ( s ) }$ . The routing logits are computed as,

$$
Z _ { d } ^ { ( s ) } = { \mathcal { G } } _ { d , E } ^ { ( s ) } ( E _ { d } ) + { \mathcal { G } } _ { d , H } ^ { ( s ) } ( H ^ { ( s ) } ) ,\tag{13}
$$

$$
Z _ { g } ^ { ( s ) } = \mathcal { G } _ { g , E } ^ { ( s ) } ( E _ { g } ) + \mathcal { G } _ { g , H } ^ { ( s ) } ( H ^ { ( s ) } ) ,\tag{14}
$$

where $Z _ { d } ^ { ( s ) }$ and $Z _ { g } ^ { ( s ) }$ are the DER and GER logits, respectively. The two branches of each router produce logits on the same spatial grid for addition. At each spatial location, DER predicts $J _ { L - 1 }$ logits for the finest-level groups, while GER predicts L logits for levels $0 , \ldots , L - 1$ . Different locations can select experts according to their local degradation characteristics. Conditioning both routers on $\pmb { H } ^ { ( s ) }$ allows expert routing to adapt throughout the network.

At each spatial location, DER applies top-1 routing to $Z _ { d } ^ { ( s ) }$ to select a finest-level DR group, while GER applies top-1 routing to $Z _ { g } ^ { ( s ) }$ to select a hierarchy level. The selected finest-level group determines a path through the hierarchy, and the selected non-root level determines the expert on that path. Selecting level 0 uses only the shared base block at that location. Let $\mathbf { \bar { \rho } } _ { M _ { i , j } ^ { \left( s \right) } }$ be the binary spatial mask assigned to expert (i, j) by the joint decision of DER and GER. Each mask is aligned with its residual feature map and broadcast across channels. At each location, exactly one non-root mask is 1 if GER selects a non-root level, and all non-root masks are 0 if GER selects the root. The output of block s is,

$$
\begin{array} { r l } & { \pmb { H } ^ { ( s + 1 ) } = \mathcal { F } _ { \mathrm { b a s e } } ^ { ( s ) } ( \pmb { H } ^ { ( s ) } ) } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { L - 1 } \sum _ { j = 1 } ^ { J _ { i } } M _ { i , j } ^ { ( s ) } \odot \Delta \mathcal { F } _ { i , j } ^ { ( s ) } ( \pmb { H } ^ { ( s ) } ) . } \end{array}\tag{15}
$$

For every block s and spatial location $( u , v )$ $\begin{array} { r l r } { \sum _ { i = 1 } ^ { L - 1 } \sum _ { j = 1 } ^ { \bar { J _ { i } } } [ M _ { i , j } ^ { ( s ) } ] _ { u , v } } & { { } \in } & { \{ 0 , 1 \} } \end{array}$ . The sum is 0 when GER selects the root and 1 otherwise, so each location uses the shared base block with at most one non-root expert residual.

As a result, our routing strategy preserves fine-grained specialization while allowing select coarser ancestor experts to alleviate the effect of degradation-estimate errors. This targets the assignment-related error in Eq. (5) and helps preserve the restoration benefit of adding specialized experts.

## E. Loss Functions

Let U denote the target map on $\Omega _ { d } ,$ where each spatial location is assigned the finest-level cluster centroid of its corresponding source sample. For a CutMix input, its spatial mixing mask determines the source sample used at each target location. We optimize the degradation distribution using the diagonal-Gaussian negative log-likelihood, omitting the parameter-independent additive constant,

![](images/e39097f25b7af94058515b2b59653e6c4b0bb62c09ec9729f18b8d9d025a3592.jpg)  
Fig. 5. Illustration of the degradation pipeline. Weather artifacts are first introduced, after which the remaining degradation types are randomly selected, shuffled, and sequentially applied.

$$
\mathcal { L } _ { d g } = \frac { 1 } { 2 | \Omega _ { d } | } \sum _ { r \in \Omega _ { d } } \sum _ { k = 1 } ^ { p } \left( \frac { \left( U _ { r , k } - E _ { d , r , k } \right) ^ { 2 } } { E _ { g , r , k } } + \log E _ { g , r , k } \right) ,\tag{16}
$$

where r indexes locations in $\Omega _ { d } ,$ , k indexes the p DR channels, and log denotes the natural logarithm. Thus, $E _ { g }$ provides a learned uncertainty cue for GER. The minimum value of 0.01 in Eq. (12) ensures a positive denominator and a well-defined logarithm.

In training restoration networks, we first train the shared base model and then train each LoRA expert on its corresponding DR group. We then initialize the DERs using the finestlevel cluster labels. The routers and LoRA experts are then jointly fine-tuned with the CutMix pairs while the shared base model remains frozen. We apply the load-balancing loss [47] $\mathcal { L } _ { \mathrm { a u x } }$ to GER to discourage collapse to a single granularity. The joint fine-tuning objective is,

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \ell _ { 1 } + \alpha \mathcal { L } _ { \mathrm { d g } } + \beta \mathcal { L } _ { \mathrm { a u x } } ,\tag{17}
$$

where α and $\beta$ balance uncertainty learning and load balancing, respectively.

## IV. EXPERIMENTS

We first describe the datasets and evaluation settings in Section IV-A, followed by the implementation details in Section IV-B. We then compare UniRestorer++ with stateof-the-art universal restoration methods in Section IV-C and analyze the scaling behavior of universal restoration models in Section IV-D. We validate the proposed components through ablation studies in Section V. Finally, we give the limitations and conclusions in Section VI.

## A. Experimental Setup

We consider four universal image restoration settings: the single-degradation setting, the triple-degradation setting, the penta-degradation setting, and the real-world evaluation setting. The details are as follows.

Single-Degradation Setting. Following PromptIR [13] and IDR [18], we train on Rain100L [63] for deraining, RE-SIDE [64] for dehazing, DFBW (DIV2K [65], Flickr2K [66], BSD [67], and WED [68]) for Gaussian color image denoising, GoPro [69] for motion deblurring, LOLv1 [70] for low-light enhancement, Snow100K [35] for desnowing, and DFBW for JPEG compression artifact removal. We evaluate on Rain100L, SOTS [64], BSD68, GoPro, LOLv1, Snow100K, and LIVE1 [71], respectively.

Table I. Degradation combinations in the five degradation-order groups of CDD-50. ‘RH’ and ‘SH’ denote mixed rain-haze and snow-haze degradations, respectively.
<table><tr><td>1st-Order</td><td>2nd-Order</td><td>3rd-Order</td><td>4th-Order</td><td>5th-Order</td></tr><tr><td>Noise (N)</td><td>I-RG</td><td>B-I-J</td><td>I-RL-J-RS</td><td>RS-RL-B-N-RG</td></tr><tr><td>Blur (B)</td><td>J-B</td><td>RS-RG-J</td><td>RS-B-J-RG</td><td>B-B-RL-N-J</td></tr><tr><td>JPEG (J)</td><td>B-N</td><td>B-N-RS</td><td>RL-RL-RG-RG</td><td>SH-I-RL-RL-RL</td></tr><tr><td>Rain (R)</td><td>RS-B</td><td>RL-RL-J</td><td>RS-RG-RS-J</td><td>H-RL-B-RS-J</td></tr><tr><td>Snow (S)</td><td>SH-RL</td><td>B-RS-RL</td><td>RH-RS-RG-J</td><td>RH-RS-J-RL-RG</td></tr><tr><td>Haze (H)</td><td>H-RG</td><td>RS-RS-RL</td><td>N-B-J-RG</td><td>J-J-I-RL-RG</td></tr><tr><td>Inpainting (I)</td><td>RS-J</td><td>B-J-RG</td><td>S-RL-J-J</td><td>RS-RS-B-B-RL</td></tr><tr><td>R-L (RL)</td><td>RG-RL</td><td>RH-RG-J</td><td>N-RG-RS-J</td><td>N-B-RS-J-RL</td></tr><tr><td>Resize (RS)</td><td>N-J</td><td>H-J-B</td><td>B-J-RL-RG</td><td>RL-I-B-I-RL</td></tr><tr><td>Ringing (RG)</td><td>R-RL</td><td>N-RL-RG</td><td>H-I-RL-J</td><td>RG-RG-RG-I-RL</td></tr></table>

Triple-Degradation Setting. Following OneRestore [15], we use CDD-11 [15] for both training and evaluation. CDD-11 contains four first-order, five second-order, and two third-order degradation subsets.

Penta-Degradation Setting. We construct training data and an evaluation benchmark for a wider degradation space.

(1) Degradation space: Starting from the Real-ESRGAN pipeline [72], we introduce rain, snow, haze, mixed rainhaze, mixed snow-haze, inpainting, and Richardson-Lucy (R-L) artifacts [73], [74]. The degradation space also includes additional variants of existing degradations: noise includes Poisson, speckle, impulse, and spatially correlated noise, while blur includes jitter, glass, motion, lens, and zoom blur. To synthesize a training sample with $n _ { d }$ degradation operations, we first select and apply one weather artifact, randomly sample the other $n _ { d } - 1$ degradations, shuffle them, and apply them sequentially. Fig. 5 illustrates the pipeline.

(2) High-quality (HQ) training data: We collect high-quality images from DIV2K [65], Flickr2K [66], DIV8K [75], OutdoorSceneTraining [76], MIT-Adobe-5K [77], the first 5,000 images from FFHQ [78], and the first 500,000 images from SA-1B [79]. We randomly crop five million 256 × 256 HQ patches and synthesize their low-quality (LQ) counterparts using the above pipeline, yielding five million paired training samples.

For evaluation, we develop CDD-50 using the clean images in Urban100 [80]. CDD-50 comprises five degradationorder groups, ranging from one to five degradations, with ten degradation combinations per group generated by SEAL [81]. The resulting 50 test settings are listed in Table I. Each test setting is evaluated with both in-distribution (in-dist.) and outof-distribution (out-of-dist.) parameters, measuring restoration on the training parameter range and generalization to unseen parameter ranges, respectively.

Real-World Setting. We use SIDD [82], RealBlur-J [83], DPDD [84], RealRain [85], RealSnow [23], WeatherStream [86], and FoundIR [30] to evaluate zero-shot generalization to real-world degradations.

Table II. Universal image restoration results in the single-degradation setting. CAR denotes compression artifact removal. PSNR (dB, ↑) and SSIM (↑) are reported in RGB space. ‘nT’ denotes a model jointly trained on n tasks (3T: Derain, Dehaze, and Denoise; 5T: Derain, Dehaze, Denoise, Deblur, and Lowlight; 7T: all seven tasks). The weighted averages of benchmarks are reported. The best performance is highlighted.
<table><tr><td>Type</td><td>Method</td><td colspan="2">Derain Rain100L</td><td colspan="2">Dehaze SOTS</td><td colspan="2">Denoise BSD68</td><td colspan="2">Deblur GoPro</td><td colspan="2">Low-light LOLv1</td><td colspan="2">Desnow Snow100K</td><td colspan="2">CAR LIVE1</td><td colspan="2">Average 3T</td><td>7T</td></tr><tr><td rowspan="5">3T</td><td>AirNet [17]</td><td>34.90</td><td>.967</td><td>27.94</td><td>.962</td><td>31.06</td><td>.872</td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td>29.29</td><td></td><td></td></tr><tr><td>PromptIR [13]</td><td>36.37</td><td>.972</td><td>30.58</td><td>.974</td><td>31.12</td><td>.873</td><td></td><td>I</td><td></td><td></td><td></td><td></td><td></td><td></td><td>31.50</td><td></td><td></td></tr><tr><td>InstructIR [87]</td><td>37.98</td><td>.978</td><td>30.22</td><td>.959</td><td>31.32</td><td>.875</td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td>31.49</td><td></td><td></td></tr><tr><td>DaAIR [44]</td><td>37.10</td><td>.978</td><td>32.30</td><td>.981</td><td>31.06</td><td>.868</td><td>一</td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td>32.51</td><td></td><td></td></tr><tr><td>MoCE-IR [46]</td><td>38.57</td><td>.984</td><td>31.34</td><td>.979</td><td>31.24</td><td>.873</td><td></td><td>一</td><td></td><td>一</td><td></td><td></td><td></td><td>一</td><td>32.73</td><td></td><td></td></tr><tr><td rowspan="5">5T</td><td>AirNet [17]</td><td>32.98</td><td>.951</td><td>21.04</td><td>.884</td><td>30.91</td><td>.882</td><td>24.35</td><td>.781</td><td>18.18</td><td>.735</td><td></td><td></td><td></td><td></td><td>23.82</td><td>24.10</td><td></td></tr><tr><td>InstructIR [87]</td><td>36.84</td><td>.973</td><td>27.10</td><td>.956</td><td>31.40</td><td>.887</td><td>29.40</td><td>.886</td><td>23.00</td><td>.836</td><td></td><td></td><td></td><td></td><td>28.99</td><td>29.19</td><td></td></tr><tr><td>DaAIR [44]</td><td>36.28</td><td>.975</td><td>31.97</td><td>.980</td><td>31.07</td><td>.878</td><td>29.51</td><td>.890</td><td>22.38</td><td>.825</td><td></td><td></td><td></td><td></td><td>32.20</td><td>30.24</td><td></td></tr><tr><td>PromptIR-DCPT [88]</td><td>37.32</td><td>.978</td><td>30.72</td><td>.977</td><td>31.32</td><td>.885</td><td>28.84</td><td>.877</td><td>23.35</td><td>.840</td><td></td><td></td><td></td><td></td><td>31.46</td><td>30.31</td><td></td></tr><tr><td>MoCE-IR [46]</td><td>38.04</td><td>.982</td><td>30.48</td><td>.974</td><td>31.34</td><td>.887</td><td>30.05</td><td>.899</td><td>23.00</td><td>.852</td><td></td><td></td><td></td><td></td><td>31.39</td><td>30.58</td><td></td></tr><tr><td rowspan="2">7T</td><td>UniRestorer [31]</td><td>40.22</td><td>.991</td><td>34.94</td><td>.971</td><td>31.39</td><td>.937</td><td>30.86</td><td>.933</td><td>24.42</td><td>.827</td><td>30.53</td><td>.925</td><td>30.31</td><td>.922</td><td>35.51</td><td>32.36</td><td>31.81</td></tr><tr><td>UniRestorer++</td><td>40.35</td><td>.991</td><td>35.51</td><td>.980</td><td>31.48</td><td>.937</td><td>31.59</td><td>.950</td><td>24.64</td><td>.831</td><td>30.66</td><td>.928</td><td>31.48</td><td>.947</td><td>35.78</td><td>32.71</td><td>32.24</td></tr></table>

Table III. Universal image restoration results on CDD-11. L, H, R, and S denote low-light, haze, rain, and snow, respectively. PSNR (dB, ↑) and SSIM (↑) are reported in RGB space. The best performance is highlighted.
<table><tr><td rowspan="2">Method</td><td rowspan="2">L</td><td rowspan="2"></td><td rowspan="2">H</td><td rowspan="2">1st-Order</td><td rowspan="2">R</td><td rowspan="2"></td><td rowspan="2">S</td><td rowspan="2"></td><td rowspan="2">L-H</td><td rowspan="2"></td><td rowspan="2">L-R</td><td rowspan="2"></td><td rowspan="2">2nd-Order L-S</td><td rowspan="2"></td><td rowspan="2">H-R</td><td rowspan="2"></td><td rowspan="2">H-S</td><td rowspan="2"></td><td rowspan="2">L-H-R</td><td rowspan="2">3rd-Order</td><td rowspan="2"></td><td rowspan="2">L-H-S</td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>AirNet [17]</td><td>24.83</td><td>.778</td><td>24.21</td><td>.951</td><td>26.55</td><td>.891</td><td>26.79</td><td>.919</td><td>23.23</td><td>.779</td><td>22.82</td><td>.710</td><td>23.29</td><td>.723</td><td>22.21</td><td>.868 23.29</td><td>.901</td><td></td><td>21.80</td><td>.708</td><td>22.24</td><td>.725</td><td>23.75</td></tr><tr><td>PromptIR [13]</td><td>26.32</td><td>.805</td><td>26.10</td><td>.969</td><td>31.56</td><td>.946</td><td>31.53</td><td>.960</td><td>24.49</td><td>.789</td><td>25.05</td><td>.771</td><td>24.51</td><td>.761</td><td>24.54</td><td>.924</td><td>23.70</td><td>.925</td><td>23.74</td><td>.752</td><td>23.33</td><td>.747</td><td>25.90</td></tr><tr><td>OneRestore [15]</td><td>26.48</td><td>.826</td><td>32.52</td><td>.990</td><td>33.40</td><td>.964</td><td>34.31</td><td>.973</td><td>25.79</td><td>.822</td><td>25.58</td><td>.799</td><td>25.19</td><td>.789</td><td>29.99</td><td>.957</td><td>30.21</td><td>.964</td><td>24.78</td><td>.788</td><td>24.90</td><td>.791</td><td>28.47</td></tr><tr><td>MoCE-IR [46]</td><td>27.26</td><td>.824</td><td>32.66</td><td>.990</td><td>34.31</td><td>.970</td><td>35.91</td><td>.980</td><td>26.24</td><td>.817</td><td>26.25</td><td>.800</td><td>26.04</td><td>.793</td><td>29.93</td><td>.964</td><td>30.19 .970</td><td></td><td>25.41</td><td>.789</td><td>25.39</td><td>.790</td><td>29.05</td></tr><tr><td>UniRestorer [31] UniRestorer++</td><td>27.71 27.93</td><td>.900 .901</td><td>34.53 34.98</td><td>.991 .992</td><td>36.07 36.28</td><td>.972 .975</td><td>36.84 36.89</td><td>.983 .983</td><td>26.88 27.23</td><td>.893 .896</td><td>26.93 27.22</td><td>.894 .898</td><td>26.73 27.05</td><td>.892 .893</td><td>32.02 32.66</td><td>.936 .940</td><td>30.82 31.19 .980</td><td>.976</td><td>25.78 25.92</td><td>.884 .883</td><td>25.87 26.11</td><td>.885 .887</td><td>30.02 30.31</td></tr></table>

Table IV. Universal image restoration results on CDD-50 under in-dist. and out-of-dist. degradation parameters. PSNR (dB, ↑) and SSIM (↑) are reported in RGB space. The best performance is highlighted.
<table><tr><td>Method</td><td colspan="2">1st-Order</td><td colspan="2">2nd-Order</td><td colspan="2">In-dist. 3rd-Order</td><td colspan="2">Degradation 4th-Order</td><td colspan="2">5th-Order</td><td colspan="2">Average</td><td colspan="2">1st-Order</td><td colspan="2">2nd-Order</td><td colspan="2">Out-of-dist. 3rd-Order</td><td colspan="2">Degradation 4th-Order</td><td colspan="2">5th-Order</td><td colspan="2">Average</td><td colspan="2"></td></tr><tr><td>AirNet [17]</td><td>28.25</td><td>.912</td><td>23.10</td><td>.808</td><td>20.50 .758</td><td>20.20</td><td></td><td>.760</td><td>19.05</td><td>.714</td><td>22.22</td><td>.790</td><td>25.60</td><td></td><td>.868</td><td>20.35</td><td>.755</td><td>18.80</td><td>.710</td><td>18.45</td><td>.690</td><td></td><td>16.70</td><td>.655</td><td>19.98</td><td>.736</td></tr><tr><td>PromptIR [13]</td><td>30.78</td><td>.943</td><td>25.24</td><td>.848</td><td>22.27</td><td>.801</td><td>22.11</td><td>.806</td><td>20.92</td><td>.761</td><td>24.26</td><td>.832</td><td></td><td>27.92</td><td>.902</td><td>22.45</td><td>.795</td><td>20.66</td><td>.751</td><td>19.92</td><td></td><td>.724</td><td>18.01</td><td>.689</td><td></td><td>21.79 .772</td></tr><tr><td>OneRestore [15]</td><td>28.28</td><td>.926</td><td>24.02</td><td>.846</td><td>22.54</td><td>.793</td><td>22.15</td><td>.792</td><td>20.72</td><td>.735</td><td>23.54</td><td>.818</td><td>26.08</td><td>.897</td><td></td><td>21.42</td><td>.788</td><td>20.19</td><td>.731</td><td>20.08</td><td>.727</td><td>17.85</td><td></td><td>.642</td><td>21.12</td><td>.757</td></tr><tr><td>MoCE-IR [46]</td><td>31.15</td><td>.948</td><td>26.36</td><td>.882</td><td>23.52</td><td>.823</td><td>23.32</td><td>.821</td><td>22.07</td><td>.775</td><td>25.28</td><td>.850</td><td>28.56</td><td></td><td>.923</td><td>23.66</td><td>.830</td><td>21.24</td><td>.776</td><td>20.57</td><td>.758</td><td>18.60</td><td></td><td>.721</td><td>22.53</td><td>.802</td></tr><tr><td>UniRestorer [31]</td><td>32.58</td><td>.959</td><td>26.98</td><td>.901</td><td>24.29</td><td>.842</td><td>23.81</td><td>.837</td><td>22.53</td><td>.792</td><td>26.04</td><td>.866</td><td></td><td>29.72</td><td>.935</td><td>23.91</td><td>.845</td><td>21.57</td><td>.781</td><td>21.13</td><td>.769</td><td>19.12</td><td></td><td>.713</td><td>23.09</td><td>.809</td></tr><tr><td>UniRestorer++</td><td>33.54</td><td>.967</td><td>27.40</td><td>.913</td><td>24.80</td><td>.855</td><td>24.14</td><td>.848</td><td>22.84</td><td>.804</td><td>26.54</td><td>.877</td><td>30.49</td><td></td><td>.943</td><td>24.07</td><td>.855</td><td>21.79</td><td>.785</td><td>21.51</td><td>.777</td><td>19.46</td><td></td><td>.708</td><td>23.46</td><td>.813</td></tr></table>

## B. Implementation Details

We use Restormer [10] as the restoration backbone. We first train the DR extractor using the pixel-wise contrastive loss in Eq. (8) and construct the multi-granularity degradation set. We then train the shared base model, followed by the rank-8 LoRA experts [24] for their associated DR groups. After expert training, we initialize the DERs using the finest-level cluster labels. Starting from this initialization, we jointly finetune the routers and LoRA experts while keeping the shared base model frozen. We set $\alpha = 0 . 1$ and $\beta = 0 . 0 1$ in Eq. (17). Unless otherwise specified, we follow the training protocol of Restormer [10]. For the single-degradation setting, the three granularity levels contain 1, 7, and 19 DR groups. For the triple-degradation setting, we use the three-level configuration {1, 4, 8}. For the penta-degradation setting, we construct the hierarchy as a binary tree. Training is performed using 64 NVIDIA V100 GPUs, each with 32 GB of memory. For the scaling experiments, we train each LoRA expert singly from scratch. Training 2.9B model with 510 LoRA experts takes approximately 23 days.

Table V. Zero-shot generalization comparison on real-world datasets. PSNR (dB, ↑) is reported in RGB space. All methods are trained in the single-degradation setting and evaluated without additional training. The best performance is highlighted.
<table><tr><td>Method</td><td>SIDD [82]</td><td>RealBlur-J [83]</td><td>DPDD [84]</td><td>RealRain [85]</td><td>RealSnow [23]</td><td>WeatherStream [86]</td><td>FoundIR [30]</td></tr><tr><td>AirNet [17]</td><td>23.65</td><td>24.92</td><td>20.17</td><td>22.56</td><td>23.63</td><td>21.68</td><td>20.56</td></tr><tr><td>PromptIR [13]</td><td>24.32</td><td>25.47</td><td>21.76</td><td>24.12</td><td>24.94</td><td>22.09</td><td>21.54</td></tr><tr><td>OneRestore [15]</td><td>11.46</td><td>18.79</td><td>20.96</td><td>22.73</td><td>20.72</td><td>19.49</td><td>22.94</td></tr><tr><td>MoCE-IR [46]</td><td>15.96</td><td>14.29</td><td>15.61</td><td>28.10</td><td>24.63</td><td>20.54</td><td>22.63</td></tr><tr><td>UniRestorer [31]</td><td>24.70</td><td>25.55</td><td>23.60</td><td>29.85</td><td>27.73</td><td>22.25</td><td>23.80</td></tr><tr><td>UniRestorer++</td><td>24.85</td><td>26.25</td><td>23.90</td><td>29.95</td><td>27.80</td><td>22.55</td><td>24.05</td></tr></table>

Table VI. Computational cost comparison with state-of-the-art universal methods in the single-degradation setting.
<table><tr><td>Method</td><td>#Params. (M)</td><td>#FLOPs (G)</td><td>Latency (s)</td></tr><tr><td>AirNet [17]</td><td>9</td><td>2414</td><td>0.548</td></tr><tr><td>PromptIR [13]</td><td>36</td><td>1266</td><td>0.355</td></tr><tr><td>InstructIR [87]</td><td>16</td><td>130</td><td>0.055</td></tr><tr><td>PromptIR-DCPT [88]</td><td>36</td><td>1266</td><td>0.355</td></tr><tr><td>MoCE-IR [46]</td><td>25</td><td>716</td><td>0.347</td></tr><tr><td>UniRestorer [31]</td><td>171</td><td>1156</td><td>0.484</td></tr><tr><td>UniRestorer++</td><td>176</td><td>1255</td><td>0.515</td></tr></table>

![](images/3b71002e373ea329f1e4706557e654beeb01565cd9bb7ef00d7224eeff8d6676.jpg)  
Fig. 6. Visual comparison across the single-degradation (GoPro deblurring), triple-degradation (CDD-11), penta-degradation (CDD-50), and real-world settings. UniRestorer++ removes degradations and recovers textures and colors, as highlighted by the red boxes and enlarged regions.

## C. Method Comparison and Results

Performance Comparison. Tables II to V compare the methods on single, composite, and real-world degradations. Our method achieves the state-of-the-art performance among the compared universal methods in every evaluated task or degradation-order group. Compared to UniRestorer, UniRestorer++ raises the seven-task average from 31.81 to 32.24 dB, reaches 30.31 dB on CDD-11 (+0.29 dB), and improves CDD-50 from 26.04/23.09 to 26.54/23.46 dB on the indist./out-of-dist. settings. These consistent gains demonstrate the effectiveness of the proposed method across increasingly complex degradation spaces. Fig. 6 compares restoration results across the four experimental settings, showing that our method can recover fine structure and texture.

Computational Cost Comparison. Table VI reports parameters, FLOPs, and latency for a 512 × 512 × 3 input on a single NVIDIA A6000 GPU, averaged over 100 inference runs with randomly sampled degradations from the single-degradation setting. Compared to UniRestorer, UniRestorer++ increases the parameter count from 171M to 176M (2.9%), FLOPs from 1156G to 1255G (8.6%), and latency from 0.484 s to 0.515 s (6.4%). Thus, the proposed routing design introduces slight inference overhead while improving restoration performance. Moreover, top-1 routing activates at most one non-root LoRA expert per spatial location in each restoration block, keeping the active expert computation approximately fixed as total capacity grows. This allows UniRestorer++ to scale from 176M to 2.9B parameters while maintaining approximately unchanged FLOPs and inference latency.

## D. Scaling Universal Restoration Models

We compare three scaling settings on CDD-50 and FoundIR: dense model scaling enlarges the depth and width of Restormer (five variants, 26M to 702M); vanilla MoE scaling increases the number of finest-level experts (six variants, 120M to 1.5B);

and our hierarchy model scaling retains experts at multiple granularities. In addition, Table VII includes baseline models with 238M parameters. On CDD-50, separate baselines are trained on each degradation order.

Hierarchy Model Scaling Strategy. In our method, we implement the group-specific parameters as LoRA experts, use the 76M Restormer as the shared base model, and increase the number of non-root binary hierarchy levels from three to eight, corresponding to 14, 30, 62, 126, 254, and 510 LoRA experts. Each added level refines the degradation partition and introduces parameters for the resulting fine-grained groups. This produces six UniRestorer++ variants with 154M, 244M, 423M, 781M, 1.5B, and 2.9B parameters.

Table VII and Fig. 2 show consistent gains as the hierarchical model grows from 154M to 2.9B parameters: average PSNR increases from 26.54 to 27.05 dB on in-dist. CDD-50, from 23.46 to 23.93 dB on out-of-dist. CDD-50, and from 26.22 to 28.17 dB on FoundIR. We fit the reported average PSNR of each strategy using the empirical form in Eq. (6), with $P _ { \mathrm { t o t a l } }$ measured in millions and $a _ { \kappa } ^ { \mathrm { { f i t } } } , b _ { \kappa } ^ { \mathrm { { f i t } } }$ fitted over the evaluated range. The fits of our hierarchy model scaling yield approximate $R ^ { 2 }$ values of 0.95, 0.96, and 0.98, respectively. Both MoE strategies show approximately log-linear trends over the evaluated range, whereas dense model scaling is non-monotonic. Their fitted slopes $b _ { \kappa } ^ { \mathrm { f i t } }$ are similar on in-dist. CDD-50, but ours is steeper on out-of-dist. CDD-50 (0.386 versus 0.133) and FoundIR (1.555 versus 0.808). Thus, our hierarchy model scaling provides larger gains from additional capacity under out-of-dist. and on real-world degradations, with its advantage widening over the shared parameter range. Our models also achieve higher PSNR than the 238M baseline models in Table VII and lie above the vanilla MoE scaling curves at comparable total parameter counts in Fig. 2.

Zero-shot Generalization. Under the single-degradation training setting, the standard-size UniRestorer++ achieves higher zero-shot performance than all compared universal methods on the real-world benchmarks (Table V). For models trained on the complete CDD-50 degradation space, the performance further improves consistently as model size increases from 154M to 2.9B (Table VII), demonstrating the benefit of hierarchical scaling for zero-shot generalization.

Table VII. Performance of hierarchically scaled UniRestorer++ variants. PSNR (dB, ↑) and SSIM (↑) are reported in RGB space. All UniRestorer++ variants are trained on the complete CDD-50 degradation space. Each degradation-order row uses its own 238M baseline trained only on the corresponding order and all improvements are computed relative to the baseline in the same row. In the real-world setting, the 237.8M baseline is trained on the complete CDD-50 degradation space containing all five orders.
<table><tr><td>Evaluation Setting</td><td>Degradation Order / Test Set</td><td colspan="2">Baseline 238M</td><td colspan="2">UniRestorer++ 154M</td><td colspan="2">UniRestorer++ 244M</td><td colspan="2">UniRestorer++ 423M</td><td colspan="2">UniRestorer++ 781M</td><td colspan="2">UniRestorer++ 1.5B</td><td colspan="2">UniRestorer++ 2.9B</td></tr><tr><td>In-dist.</td><td>1st-Order</td><td>33.39</td><td>.965</td><td>+0.15</td><td>+.002</td><td>+0.36</td><td>+.003</td><td>+0.56</td><td>+.003</td><td>+0.81</td><td>+.004</td><td>+0.90</td><td></td><td>+.005</td><td>+.006</td></tr><tr><td>Synthetic</td><td>2nd-Order</td><td>27.35</td><td>.912</td><td>+0.05</td><td>+.001</td><td>+0.15</td><td>+.002</td><td>+0.31</td><td>+.003</td><td>+0.46</td><td>+.005</td><td>+0.55</td><td>+.005</td><td>+0.94 +0.58</td><td>+.006</td></tr><tr><td>Degradation</td><td>3rd-Order</td><td>24.76</td><td>.854</td><td>+0.04</td><td>+.001</td><td>+0.10</td><td>+.002</td><td>+0.21</td><td>+.003</td><td>+0.31</td><td>+.005</td><td>+0.40</td><td>+.007</td><td>+0.44</td><td>+.008</td></tr><tr><td>in CDD-50</td><td>4th-Order</td><td>24.01</td><td>.845</td><td>+0.13</td><td>+.003</td><td>+0.19</td><td>+.003</td><td>+0.30</td><td>+.005</td><td>+0.44</td><td>+.017</td><td>+0.51</td><td>+.018</td><td>+0.55</td><td>+.019</td></tr><tr><td></td><td>5th-Order</td><td>22.78</td><td>.803</td><td>+0.06</td><td>+.001</td><td>+0.14</td><td>+.003</td><td>+0.23</td><td>+.004</td><td>+0.36</td><td>+.006</td><td>+0.43</td><td>+.008</td><td>+0.46</td><td>+.009</td></tr><tr><td></td><td>1st-Order</td><td>30.35</td><td>.942</td><td>+0.14</td><td>+.001</td><td>+0.31</td><td>+.002</td><td>+0.50</td><td>+.002</td><td>+0.71</td><td>+.003</td><td>+0.81</td><td>+.004</td><td>+0.85</td><td>+.005</td></tr><tr><td>Out-of-dist.</td><td>2nd-Order</td><td>23.90</td><td>.853</td><td>+0.17</td><td>+.002</td><td>+0.25</td><td>+.003</td><td>+0.37</td><td>+.005</td><td>+0.49</td><td>+.007</td><td>+0.58</td><td>+.009</td><td>+0.62</td><td>+.010</td></tr><tr><td>Synthetic</td><td>3rd-Order</td><td>21.62</td><td>.782</td><td>+0.17</td><td>+.003</td><td>+0.23</td><td>+.004</td><td>+0.31</td><td>+.007</td><td>+0.41</td><td>+.009</td><td>+0.50</td><td>+.012</td><td>+0.54</td><td>+.013</td></tr><tr><td>Degradation in CDD-50</td><td>4th-Order</td><td>21.46</td><td>.775</td><td>+0.05</td><td>+.002</td><td>+0.13</td><td>+.003</td><td>+0.23</td><td>+.006</td><td>+0.37</td><td>+.008</td><td>+0.44</td><td>+.011</td><td>+0.48</td><td>+.012</td></tr><tr><td></td><td>5th-Order</td><td>19.23</td><td>.703</td><td>+0.23</td><td>+.005</td><td>+0.30</td><td>+.007</td><td>+0.39</td><td>+.009</td><td>+0.51</td><td>+.012</td><td>+0.57</td><td>+.014</td><td>+0.60</td><td>+.015</td></tr><tr><td></td><td>SIDD [82]</td><td>23.94</td><td>.508</td><td>+6.48</td><td>+.232</td><td>+8.65</td><td>+.301</td><td>+11.04</td><td>+.353</td><td>+12.63</td><td>+.387</td><td>+13.74</td><td>+.411</td><td>+14.78</td><td>+.434</td></tr><tr><td></td><td>RealBlur-J [83]</td><td>26.42</td><td>.850</td><td>+1.17</td><td>+.005</td><td>+2.38</td><td>+.011</td><td>+3.16</td><td>+.028</td><td>+3.68</td><td>+.039</td><td>+4.05</td><td>+.047</td><td>+4.39</td><td>+.055</td></tr><tr><td></td><td>DPDD [84]</td><td>23.86</td><td>.731</td><td>+0.22</td><td>+.018</td><td>+0.60</td><td>+.033</td><td>+0.78</td><td>+.040</td><td>+0.89</td><td>+.045</td><td>+0.97</td><td>+.048</td><td>+1.05</td><td>+.051</td></tr><tr><td>Real-world</td><td></td><td></td><td>.894</td><td></td><td></td><td></td><td></td><td></td><td>+.028</td><td></td><td></td><td></td><td>+.034</td><td></td><td>+.036</td></tr><tr><td>Degradation</td><td>RealRain [85]</td><td>28.89</td><td></td><td>+1.36</td><td>+.016</td><td>+1.85</td><td>+.024</td><td>+2.10</td><td></td><td>+2.26</td><td>+.032</td><td>+2.37</td><td></td><td>+2.48</td><td></td></tr><tr><td></td><td>RealSnow [23]</td><td>27.40</td><td>.884</td><td>+0.40</td><td>-.004</td><td>+1.00</td><td>-.002</td><td>+1.19</td><td>+.003</td><td>+1.31</td><td>+.006</td><td>+1.40</td><td>+.009</td><td>+1.48</td><td>+.011</td></tr><tr><td></td><td>WeatherStream [86]</td><td>21.22</td><td>.726</td><td>+1.53</td><td>+.064</td><td>+1.75</td><td>+.071</td><td>+1.88</td><td>+.075</td><td>+1.96</td><td>+.078</td><td>+2.02</td><td>+.079</td><td>+2.08</td><td>+.081</td></tr><tr><td></td><td>FoundIR [30]</td><td>25.88</td><td>.797</td><td>+0.34</td><td>+.011</td><td>+0.67</td><td>+.032</td><td>+1.18</td><td>+.040</td><td>+1.65</td><td>+.048</td><td>+1.98</td><td>+.054</td><td>+2.29</td><td>+.060</td></tr></table>

Table VIII. Zero-, one-, and five-shot comparison with state-of-the-art specialist models on real-world benchmarks. Specialist results are taken from the corresponding papers or trained from scratch. PSNR (dB, ↑) and SSIM (↑) are reported in RGB space. We repeat each few-shot experiment five times and report mean ± std. The best performance is highlighted.
<table><tr><td>Test Sets</td><td>Task</td><td>Specialist Methods</td><td colspan="2">Performance of Specialists</td><td colspan="2">UniRestorer++ (2.9B, zero-shot)</td><td colspan="2">UniRestorer++ (2.9B, 1-shot)</td><td colspan="2">UniRestorer++ (2.9B, 5-shot)</td></tr><tr><td>SIDD [82]</td><td>Denoising</td><td>NAFNet [11]</td><td>40.30</td><td>.962</td><td>38.72</td><td>.942</td><td> $3 8 . 8 3 \pm . 2 1$ </td><td> $. 9 4 5 \pm . 0 0 9$ </td><td> $3 9 . 2 6 \pm . 1 3$ </td><td> $. 9 5 7 \pm . 0 0 4$ </td></tr><tr><td>RealBlur-J [83]</td><td>Motion Deblurring</td><td>X-Restormer [12]</td><td>36.27</td><td>.958</td><td>30.81</td><td>.905</td><td> $3 6 . 2 0 \pm . 1 1$ </td><td> $. 9 5 2 \pm . 0 0 5$ </td><td> ${ \bf 3 6 . 2 8 \pm . 0 5 }$ </td><td> $\mathbf { 9 5 8 } \pm . 0 0 2$ </td></tr><tr><td>DPDD [84]</td><td>Defocus Deblurring</td><td>Restormer [10]</td><td>25.98</td><td>.811 24.91</td><td>.782</td><td></td><td> $2 5 . 8 1 \pm . 2 8$ </td><td> $. 8 0 4 \pm . 0 1 5$ </td><td> ${ 2 6 . 2 8 \pm . 1 4 }$ </td><td> $. 8 1 5 \pm . 0 0 6$ </td></tr><tr><td>RealRain [85]</td><td>Deraining</td><td>Restormer [10]</td><td>43.93</td><td>.993 31.37</td><td>.930</td><td> $4 1 . 9 5 \pm . 3 7$ </td><td></td><td> $. 9 9 0 \pm . 0 0 3$ </td><td> ${ \bf 4 4 . 0 6 \pm . 1 6 }$ </td><td> $. 9 9 1 \pm . 0 0 2$ </td></tr><tr><td>RealSnow [23]</td><td>Snow Removal</td><td>Restormer [10]</td><td>30.94</td><td>.913 28.88</td><td>.895</td><td> $3 1 . 5 8 \pm . 4 2$ </td><td></td><td> $. 9 4 6 \pm . 0 1 1$ </td><td> $3 3 . 2 8 \pm . 1 9$ </td><td> $. 9 6 9 \pm . 0 0 4$ </td></tr><tr><td>WeatherStream [86]</td><td>Deweathering</td><td>Restormer (Imper.) [89]</td><td>23.15</td><td>.810 23.30</td><td>.807</td><td></td><td> $2 3 . 3 5 \pm . 2 4$ </td><td> $. 8 1 5 \pm . 0 0 9$ </td><td> $2 3 . 7 2 \pm . 1 5$ </td><td> $. 8 2 5 \pm . 0 0 6$ </td></tr></table>

Table IX. One- and five-shot performance on FoundIR [30]. PSNR (dB, ↑) and SSIM (↑) are reported in RGB space. We repeat each experiment five times and report mean ± std. The best performance is highlighted.
<table><tr><td>Method</td><td>Blur</td><td>Noise</td><td>JPEG</td><td>Haze</td><td>Rain</td><td>Raindrop</td><td>B+N</td><td>B+J</td><td>N+J</td><td>R+H</td><td>B+N+J</td></tr><tr><td rowspan="2">FoundIR [30]</td><td>25.93</td><td>38.98</td><td>34.03</td><td>22.19</td><td>34.29</td><td>29.74</td><td>23.87</td><td>29.58</td><td>40.43</td><td>21.80</td><td>27.98</td></tr><tr><td>.791</td><td>.964</td><td>.942</td><td>.849</td><td>.943</td><td>.916</td><td>.773</td><td>.854</td><td>.979</td><td>.758</td><td>.745</td></tr><tr><td>UniRestorer++</td><td> $2 5 . 8 0 \pm . 2 2$ </td><td> $4 0 . 1 0 \pm . 1 1$ </td><td> $3 4 . 8 0 \pm . 0 8$ </td><td> $2 1 . 5 0 \pm . 3 7$ </td><td> $3 4 . 8 0 \pm . 2 6$ </td><td> $3 2 . 8 0 \pm . 3 1$ </td><td> $2 4 . 9 0 \pm . 2 8$ </td><td> $2 6 . 8 0 \pm . 3 4$ </td><td> $4 0 . 2 0 \pm . 1 4$ </td><td> $2 3 . 0 0 \pm . 3 6$ </td><td> $2 7 . 1 0 \pm . 3 2$ </td></tr><tr><td>(2.9B, 1-shot)</td><td> $. 7 7 5 \pm . 0 0 6$ </td><td> $. 9 7 0 \pm . 0 0 3$ </td><td> $. 9 7 5 \pm . 0 0 2$ </td><td> $. 8 5 2 \pm . 0 0 8$ </td><td>.949 ± .005 .945 ± .006</td><td></td><td> $. 8 3 2 \pm . 0 0 7$ </td><td> $. 8 1 8 \pm . 0 0 7$ </td><td> $. 9 7 3 \pm . 0 0 2$ </td><td> $. 8 1 2 \pm . 0 1 0$ </td><td> $. 7 2 5 \pm . 0 0 9$ </td></tr><tr><td>UniRestorer++</td><td> $2 7 . 6 2 \pm . 1 4$ </td><td> ${ \bf 4 1 . 8 9 \pm . 0 4 }$ </td><td> $3 5 . 2 3 \pm . 0 2$ </td><td> $2 3 . 3 4 \pm . 2 3$ </td><td> ${ 3 7 . 1 0 \pm . 1 5 }$ </td><td> $3 7 . 2 2 \pm . 1 8$ </td><td> $2 7 . 0 2 \pm . 2 1$ </td><td> $2 5 . 9 4 \pm . 2 6$ </td><td> ${ \bf 4 1 . 0 1 \pm . 0 5 }$ </td><td> ${ \bf 2 6 . 8 1 \pm . 2 9 }$ </td><td> $2 7 . 3 0 \pm . 1 2$ </td></tr><tr><td> $( 2 . 9 \mathrm { B } , 5 \mathrm { - s h o t } )$ </td><td> ${ \bf . 8 8 1 \pm . 0 0 4 }$ </td><td> $\mathbf { 9 8 5 } \pm . 0 0 2$ </td><td>.976 ± .002</td><td> $. 8 9 4 \pm . 0 0 6$ </td><td> ${ \bf 9 6 9 } \pm { \bf . 0 0 3 }$ </td><td> $\mathbf { 9 7 9 } \pm \mathbf { . 0 0 3 }$ </td><td> $. 8 6 4 \pm . 0 0 5$ </td><td> $. 8 0 3 \pm . 0 0 6$ </td><td> $\mathbf { 9 8 5 } \pm . 0 0 2$ </td><td> ${ \bf . 8 9 1 \pm . 0 0 6 }$ </td><td> $. 7 4 2 \pm . 0 0 7$ </td></tr></table>

viates the effect of degradation-estimation errors and allows uncertain inputs to use broader experts while retaining finegrained specialization.

Few-shot Transfer. For evaluation on one- and five-shot performance, we sample the whole one and five images, respectively, from the training sets of each task. Starting from the pretrained 2.9B model, we freeze the shared backbone and fine-tune only the routers and LoRA experts. We keep the training settings as described in Section IV-B. We repeat each experiment five times and report the mean and standard deviation, as results in Tables VIII and IX. The visual comparisons are in Fig. 7. Finally, five-shot fine-tuning exceeds the specialist methods on 14 of 17 test sets. Compared with zero-shot inference, few-shot adaptation improves more on the target distribution, which differs more from the pretraining distribution (i.e., realrain).

## V. ABLATION STUDY AND ANALYSIS

Results and Analysis. Across the evaluated model sizes, hierarchical scaling consistently improves both restoration accuracy and generalization. Multi-granularity routing alle-

We use a subset of the developed degradation space to ablate DR fineness, the multi-granularity degradation set, granularityestimation-based routing, and the training losses. The synthetic degradation parameters are controlled to form in-dist. and outof-dist. evaluation sets. Separately, we ablate the dense DR map and region-aware expert routing in the seven-task singledegradation setting.

## A. Effect of the Multi-Granularity Degradation Modeling

Effect of DR Fineness. The upper part of Table X evaluates a single DER with different numbers of DR groups. Increasing the number of groups from 1 to 8 improves in-dist. PSNR from 22.06 to 24.22 dB, showing the benefit of finer specialization. Increasing the number further to 16 yields only 0.08 dB additional improvement. In contrast, out-of-dist. performance is highest with 4 groups (18.49 dB) among the single-granularity variants and does not improve as the partition becomes finer. Thus, a single fixed granularity cannot simultaneously satisfy in-dist. specialization and out-of-dist. generalization.

![](images/ead0dc6c819b9c74676c3c922432ba23a713f270e355394ea5097be8d0808dfd.jpg)  
Fig. 7. Visual comparison of zero-shot and five-shot UniRestorer++ against specialist models. Five-shot fine-tuning improves degradation removal and detail recovery, as highlighted by the red boxes.

Table X. Ablation of DR fineness and the multi-granularity degradation set. Average PSNR (dB) is reported under in-dist. and out-of-dist. degradation parameters.
<table><tr><td rowspan="2">Routers</td><td rowspan="2"># Levels</td><td rowspan="2">{ # DR Groups }</td><td colspan="2">PSNR (dB)</td></tr><tr><td>In-dist.</td><td>Out-of-dist.</td></tr><tr><td rowspan="5">DER</td><td>1</td><td>{1}</td><td>22.06</td><td>17.23</td></tr><tr><td>1</td><td>1 2 1</td><td>22.42</td><td>17.51</td></tr><tr><td>1</td><td>4 1</td><td>23.75</td><td>18.49</td></tr><tr><td>1</td><td>{8 1</td><td>24.22</td><td>18.35</td></tr><tr><td>1</td><td>{16}</td><td>24.30</td><td>18.44</td></tr><tr><td rowspan="4">DER, GER</td><td>2</td><td>{1, 8 }</td><td>24.27</td><td>18.86</td></tr><tr><td>2</td><td>{4, 8 }</td><td>24.44</td><td>19.06</td></tr><tr><td>3</td><td>{ 1, 4, 8 }</td><td>24.46</td><td>19.45</td></tr><tr><td>4</td><td>{ 1, 4, 8, 16}</td><td>24.41</td><td>19.47</td></tr></table>

![](images/16ce5203b1a152995096dc502dd05b93e088ed479ac8ff9fd6e88b5fb440d5d9.jpg)  
Fig. 8. Average selected hierarchy level and corresponding PSNR for the full and Vanilla variants as degradation parameters shift from in-dist. to out-ofdist.

Table XI. Ablation of the dense DR map and region-aware expert routing. PSNR (dB) is reported per task, and Average is the mean over the seven tasks.
<table><tr><td>Variant</td><td>Rain</td><td>Haze</td><td>Noise</td><td>Blur</td><td>Low-light</td><td>Snow</td><td>CAR Avg.</td></tr><tr><td>Baseline</td><td>40.22</td><td>34.94</td><td>31.39</td><td>30.86</td><td>24.42</td><td>30.53</td><td>30.31 31.81</td></tr><tr><td>+ Dense DR</td><td>40.31</td><td>35.09</td><td>31.44</td><td>31.02</td><td>24.52</td><td>30.59</td><td>31.27 32.03</td></tr><tr><td>+ Region-Aware Expert Routing</td><td>40.35 35.51 31.48</td><td></td><td></td><td>31.59</td><td>24.64</td><td>30.66 31.48 32.24</td><td></td></tr></table>

Table XII. Ablation of the loss functions.
<table><tr><td> $\ell _ { 1 }$ </td><td> $\mathcal { L } _ { \mathrm { d g } }$ </td><td> $\mathcal { L } _ { \mathrm { a u x } }$ </td><td colspan="2">PSNR (dB)</td></tr><tr><td></td><td></td><td></td><td>In-dist.</td><td>Out-of-dist.</td></tr><tr><td>√</td><td>x</td><td>x</td><td>24.14</td><td>18.37</td></tr><tr><td>√</td><td>√</td><td>x</td><td>24.35</td><td>18.76</td></tr><tr><td>√</td><td>√</td><td>√</td><td>24.46</td><td>19.45</td></tr></table>

Effect of Different Granularities. The lower part of Table X evaluates multi-granularity configurations using both DER and GER. The three-level configuration {1, 4, 8} achieves 24.46 dB on in-dist. data and 19.45 dB on out-of-dist. data, outperforming all single-granularity variants on both evaluation settings. Adding a fourth level with 16 groups changes the results to 24.41 and 19.47 dB, providing only a 0.02 dB outof-dist. gain while reducing in-dist. performance and requiring more experts. We therefore use {1, 4, 8} for the experiments in the triple-degradation setting.

## B. Effect and Analysis of Multi-Granularity Expert Routing

Effect of Granularity-Estimation-Based Routing. To isolate GER, we compare the full model with a Vanilla variant that always selects the finest hierarchy level. We normalize the degradation parameter range from in-dist. to out-of-dist. into [0, 1], uniformly sample 15 points, and apply the corresponding degradations to Urban100. Fig. 8 reports the average selected hierarchy level and PSNR at each point. As the degradation parameters move away from the training distribution, GER increasingly selects coarser hierarchy levels, whereas Vanilla remains restricted to the finest level. This adaptation preserves in-dist. accuracy while improving robustness under larger distribution shifts.

Visualization of Routing Results. We inspect the finestlevel DR grouping and expert assignments on CDD-11. The dominant degradation types of the eight finest-level DR groups are: (1) {Haze+Rain (H+R)}, (2) {Rain (R)}, (3) {Low-light+Rain (L+R), Low-light+Haze+Rain (L+H+R)}, (4) {Low-light+Haze (L+H)}, (5) {Haze+Snow (H+S), Snow (S)}, (6) {Haze (H)}, (7) {Low-light+Haze+Snow (L+H+S), Low-light+Snow (L+S)}, (8) {Low-light (L)}. Fig. 9 shows different finest-level group assignments and granularity preferences across degradation subsets. GER selects both coarse and fine levels, illustrating the use of restoration capacity at multiple granularities.

## C. Effect of Dense DR and Region-Aware Expert Routing

Table XI ablates the dense DR map and region-aware expert routing. The first row is the UniRestorer baseline, which uses an image-level DR for global expert routing. The second row replaces the image-level DR with the dense DR map but spatially averages it before expert routing, increasing the seven-task average PSNR from 31.81 to 32.03 dB. The third row preserves the spatial information in the dense DR map for region-aware expert routing, increasing the average to 32.24 dB. On the two representative spatially varying degradations, our design improves dehazing from 34.94 to 35.51 dB and deblurring from 30.86 to 31.59 dB. These gains support the use of the dense DR map and region-aware expert routing. Fig. 10 visualizes the responses of the dense DR map on motion blur, defocus blur, and haze, illustrating spatially varying responses to these degradations.

Dominant Degradation Type of Each Cluster  
![](images/8197eb8df5297f8cc504fd752f83f75b3c722d525f2092292d0f868d7653e159.jpg)

![](images/01ebfd38bee897fa90700e8595b8931b1bb89f0a5d9b283a4b7cea117eacdb05.jpg)

Fig. 9. Routing frequencies on CDD-11. Left: degradation-estimation-based router (DER) assignments to the eight finest-level DR groups, whose dominant degradation types are shown on the horizontal axis. Right: granularityestimation-based router (GER) assignments to the hierarchy levels. Rows correspond to the 11 CDD-11 degradation subsets.  
![](images/f1e2ad91e9b55aa4115fdb1cc36b8ac0b12acd021f52871d4cd962c624ea11f1.jpg)  
Fig. 10. Visualization of the dense DR map for real-world spatially varying degradations, i.e., motion blur (ReLoBlur [90]), defocus blur (DPDD [84]), and haze (FoundIR [30]). Gray and black indicate relatively lower and higher degradation severity, respectively.

## D. Effect of Loss Functions

Table XII evaluates the loss functions used during training. Compared with the $\ell _ { 1 }$ restoration objective alone, adding $\mathcal { L } _ { \mathrm { d g } }$ improves PSNR from 24.14 to 24.35 dB on the in-dist. data and from 18.37 to 18.76 dB on the out-of-dist. data. Adding the GER load-balancing loss $\mathcal { L } _ { \mathrm { a u x } }$ increases PSNR to 24.46 dB on the in-dist. data and 19.45 dB on the out-of-dist. data.

## VI. CONCLUSION AND LIMITATIONS

In this paper, we presented UniRestorer++, a universal image restoration framework that scales capacity through hierarchical degradation-specific experts. Dense degradation representations and multi-granularity routing enable spatially adaptive expert selection while reducing sensitivity to degradationestimation errors. We constructed CDD-50 and evaluated six variants from 154M to 2.9B parameters, observing approximately log-linear PSNR trends within this range. Our method outperforms the compared universal restoration methods and demonstrates zero-shot generalization and few-shot transfer, surpassing the compared specialist models on 14 of 17 test sets after five-shot fine-tuning. These results support hierarchical capacity expansion as a promising approach to scalable universal restoration.

A limitation is that our degradation hierarchy and expert pool are constructed offline and remain fixed during inference. Inspired by recent work on self-improving agents [91], a future direction is to extend this modular framework toward a recursively self-improving (RSI) restoration agent. Such an agent could diagnose failures, refine degradation groups, add or update experts, and revise their routing, with proposed changes assessed on held-out data. By learning from the outcomes of its own updates, such an agent could improve both restoration performance and its strategy for refining and expanding the expert hierarchy. This would extend our hierarchy model scaling strategy from predefined capacity expansion toward feedbackdriven improvement, allowing restoration capabilities to evolve with changing real-world degradations.

## REFERENCES

[1] D. Feijoo, J. C. Benito, A. Garcia, and M. V. Conde, “DarkIR: Robust low-light image restoration,” in CVPR, 2025, pp. 10 879–10 889.

[2] Z. Yang, H. Chen, Z. Qian, Y. Zhou, H. Zhang, D. Zhao, B. Wei, and Y. Xu, “Region attention transformer for medical image restoration,” in MICCAI, 2024, pp. 603–613.

[3] X. Wang, Z. Zheng, J. Shao, Y. Duan, and L.-J. Deng, “Adaptive rectangular convolution for remote sensing pansharpening,” in CVPR, 2025, pp. 17 872–17 881.

[4] Y. Lin, Z. Lin, H. Chen, P. Pan, C. Li, S. Chen, K. Wen, Y. Jin, W. Li, and X. Ding, “JarvisIR: Elevating autonomous driving perception with intelligent image restoration,” in CVPR, 2025, pp. 22 369–22 380.

[5] M. Liao, H. Dong, X. Wang, K. Ubul, Y. Shao, and Z. Yan, “GM-MoE: Low-light enhancement with gated-mechanism mixture-of-experts,” in ICCV, 2025, pp. 8766–8776.

[6] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142–3155, 2017.

[7] C. Dong, C. C. Loy, K. He, and X. Tang, “Learning a deep convolutional network for image super-resolution,” in ECCV, 2014, pp. 184–199.

[8] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, “Multi-stage progressive image restoration,” in CVPR, 2021, pp. 14 821–14 831.

[9] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “SwinIR: Image restoration using Swin Transformer,” in ICCVW, 2021, pp. 1833–1844.

[10] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in CVPR, 2022, pp. 5728–5739.

[11] L. Chen, X. Chu, X. Zhang, and J. Sun, “Simple baselines for image restoration,” in ECCV, 2022, pp. 17–33.

[12] X. Chen, Z. Li, Y. Pu, Y. Liu, J. Zhou, Y. Qiao, and C. Dong, “A comparative study of image restoration networks for general backbone network design,” in ECCV, 2024, pp. 74–91.

[13] V. Potlapalli, S. W. Zamir, S. Khan, and F. S. Khan, “PromptIR: Prompting for all-in-one image restoration,” in NeurIPS, vol. 36, 2023, pp. 71 275–71 293.

[14] J. M. J. Valanarasu, R. Yasarla, and V. M. Patel, “TransWeather: Transformer-based restoration of images degraded by adverse weather conditions,” in CVPR, 2022, pp. 2353–2363.

[15] Y. Guo, Y. Gao, Y. Lu, H. Zhu, R. W. Liu, and S. He, “OneRestore: A universal restoration framework for composite degradation,” in ECCV, 2024, pp. 255–272.

[16] X. Kong, C. Dong, and L. Zhang, “Towards effective multiple-in-one image restoration: A sequential and prompt learning strategy,” arXiv preprint arXiv:2401.03379, 2024.

[17] B. Li, X. Liu, P. Hu, Z. Wu, J. Lv, and X. Peng, “All-in-one image restoration for unknown corruption,” in CVPR, 2022, pp. 17 452–17 462.

[18] J. Zhang, J. Huang, M. Yao, Z. Yang, H. Yu, M. Zhou, and F. Zhao, “Ingredient-oriented multi-degradation learning for image restoration,” in CVPR, 2023, pp. 5825–5835.

[19] Y. Ai, H. Huang, X. Zhou, J. Wang, and R. He, “Multimodal prompt perceiver: Empower adaptiveness, generalizability and fidelity for all-inone image restoration,” in CVPR, 2024, pp. 25 432–25 444.

[20] H. Chen, W. Li, J. Gu, J. Ren, S. Chen, T. Ye, R. Pei, K. Zhou, F. Song, and L. Zhu, “RestoreAgent: Autonomous image restoration agent via multimodal large language models,” in NeurIPS, vol. 37, 2024, pp. 110 643–110 666.

[21] S. Cao, Y. Liu, W. Zhang, Y. Qiao, and C. Dong, “GRIDS: Grouped multiple-degradation restoration with image degradation similarity,” in ECCV, 2024, pp. 70–87.

[22] D. Park, B. H. Lee, and S. Y. Chun, “All-in-one image restoration for unknown degradations using adaptive discriminative filters for specific degradations,” in CVPR, 2023, pp. 5815–5824.

[23] Y. Zhu, T. Wang, X. Fu, X. Yang, X. Guo, J. Dai, Y. Qiao, and X. Hu, “Learning weather-general and weather-specific features for image restoration under multiple adverse weather conditions,” in CVPR, 2023, pp. 21 747–21 758.

[24] Y. Ai, H. Huang, and R. He, “LoRA-IR: Taming low-rank experts for efficient all-in-one image restoration,” arXiv preprint arXiv:2410.15385, 2024.

[25] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language models are few-shot learners,” in NeurIPS, vol. 33, 2020, pp. 1877–1901.

[26] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” J. Mach. Learn. Res., vol. 21, no. 140, pp. 1–67, 2020.

[27] X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer, “Scaling vision transformers,” in CVPR, 2022, pp. 12 104–12 113.

[28] Z. Liu, H. Hu, Y. Lin, Z. Yao, Z. Xie, Y. Wei, J. Ning, Y. Cao, Z. Zhang, L. Dong, F. Wei, and B. Guo, “Swin Transformer V2: Scaling up capacity and resolution,” in CVPR, 2022, pp. 12 009–12 019.

[29] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. E. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jegou, P. Labatut, and P. Bojanowski, “DINOv3,”´ Trans. Mach. Learn. Res., 2026.

[30] H. Li, X. Chen, J. Dong, J. Tang, and J. Pan, “FoundIR: Unleashing million-scale training data to advance foundation models for image restoration,” in ICCV, 2025, pp. 12 626–12 636.

[31] J. Lin, Z. Zhang, W. Li, R. Pei, H. Xu, H. Zhang, and W. Zuo, “UniRestorer: Universal image restoration via adaptively estimating image degradation at proper granularity,” in ICLR, 2026.

[32] O. Kupyn, T. Martyniuk, J. Wu, and Z. Wang, “DeblurGAN-v2: Deblurring (orders-of-magnitude) faster and better,” in ICCV, 2019, pp. 8878–8887.

[33] D. Ren, W. Zuo, Q. Hu, P. Zhu, and D. Meng, “Progressive image deraining networks: A better and simpler baseline,” in CVPR, 2019, pp. 3937–3946.

[34] Y. Song, Z. He, H. Qian, and X. Du, “Vision transformers for single image dehazing,” IEEE Trans. Image Process., vol. 32, pp. 1927–1941, 2023.

[35] Y.-F. Liu, D.-W. Jaw, S.-C. Huang, and J.-N. Hwang, “DesnowNet: Context-aware deep network for snow removal,” IEEE Trans. Image Process., vol. 27, no. 6, pp. 3064–3073, 2018.

[36] Z. Zhang, R. Wang, H. Zhang, and W. Zuo, “Self-supervised learning for real-world super-resolution from dual and multiple zoomed observations,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 3, pp. 1348–1361, 2025.

[37] Z. Zhang, R. Wang, H. Zhang, Y. Chen, and W. Zuo, “Self-supervised learning for real-world super-resolution from dual zoomed observations,” in ECCV, 2022, pp. 610–627.

[38] B. Lester, R. Al-Rfou, and N. Constant, “The power of scale for parameter-efficient prompt tuning,” in EMNLP, 2021, pp. 3045–3059.

[39] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for NLP,” in ICML, 2019, pp. 2790–2799.

[40] K. Zhang, W. Zuo, and L. Zhang, “Learning a single convolutional superresolution network for multiple degradations,” in CVPR, 2018, pp. 3262– 3271.

[41] Z. Chen, Y. Shen, M. Ding, Z. Chen, H. Zhao, E. Learned-Miller, and C. Gan, “Mod-Squad: Designing mixtures of experts as modular multitask learners,” in CVPR, 2023, pp. 11 828–11 837.

[42] Z. Chen, M. Ding, Y. Shen, W. Zhan, M. Tomizuka, E. Learned-Miller, and C. Gan, “An efficient general-purpose modular vision model via multi-task heterogeneous training,” arXiv preprint arXiv:2306.17165, 2023.

[43] J. Zhu, X. Zhu, W. Wang, X. Wang, H. Li, X. Wang, and J. Dai, “Uni-Perceiver-MoE: Learning sparse generalist models with conditional MoEs,” in NeurIPS, vol. 35, 2022, pp. 2664–2678.

[44] E. Zamfir, Z. Wu, N. Mehta, D. P. Paudel, Y. Zhang, and R. Timofte, “Efficient degradation-aware any image restoration,” arXiv preprint arXiv:2405.15475, 2024.

[45] Y. Tian, J. Han, H. Chen, Y. Xi, G. Zhang, J. Hu, C. Xu, and Y. Wang, “Instruct-IPT: All-in-one image processing transformer via weight modulation,” arXiv preprint arXiv:2407.00676v1, 2024.

[46] E. Zamfir, Z. Wu, N. Mehta, Y. Tan, D. P. Paudel, Y. Zhang, and R. Timofte, “Complexity experts are task-discriminative learners for any image restoration,” in CVPR, 2025, pp. 12 753–12 763.

[47] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” in ICLR, 2017.

[48] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “GShard: Scaling giant models with conditional computation and automatic sharding,” in ICLR, 2021.

[49] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” J. Mach. Learn. Res., vol. 23, no. 120, pp. 1–39, 2022.

[50] X. Chen, X. Wang, W. Zhang, X. Kong, Y. Qiao, J. Zhou, and C. Dong, “HAT: Hybrid attention transformer for image restoration,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 3, pp. 2676–2694, 2026.

[51] Y. Liu, J. He, J. Gu, X. Kong, Y. Qiao, and C. Dong, “DegAE: A new pretraining paradigm for low-level vision,” in CVPR, 2023, pp. 23 292– 23 303.

[52] X. Kong, J. Gu, Y. Liu, W. Zhang, X. Chen, Y. Qiao, and C. Dong, “A preliminary exploration towards general image restoration,” arXiv preprint arXiv:2408.15143, 2024.

[53] Q. Yang, D. Chen, Z. Tan, Q. Liu, Q. Chu, J. Bao, L. Yuan, G. Hua, and N. Yu, “HQ-50K: A large-scale, high-quality dataset for image restoration,” arXiv preprint arXiv:2306.05390, 2023.

[54] X. Chen, K. Zhu, Y. Pu, S. Cao, X. Li, W. Zhang, Y. Liu, Y. Qiao, J. Zhou, and C. Dong, “Exploring scalable unified modeling for general low-level vision,” arXiv preprint arXiv:2507.14801, 2025.

[55] A. Clark, D. De Las Casas, A. Guy, A. Mensch, M. Paganini, J. Hoffmann, B. Damoc, B. Hechtman, T. Cai, S. Borgeaud, G. B. Van Den Driessche, E. Rutherford, T. Hennigan, M. J. Johnson, A. Cassirer, C. Jones, E. Buchatskaya, D. Budden, L. Sifre, S. Osindero, O. Vinyals, M. Ranzato, J. Rae, E. Elsen, K. Kavukcuoglu, and K. Simonyan, “Unified scaling laws for routed language models,” in ICML, 2022, pp. 4057–4086.

[56] I. Alabdulmohsin, B. Neyshabur, and X. Zhai, “Revisiting neural scaling laws in language and vision,” in NeurIPS, vol. 35, 2022, pp. 22 300– 22 312.

[57] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjolund, and T. B. Sch ¨ on,¨ “Controlling vision-language models for universal image restoration,” in ICLR, 2024.

[58] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,

“Learning transferable visual models from natural language supervision,” in ICML, 2021, pp. 8748–8763.

[59] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016, pp. 770–778.

[60] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, “CutMix: Regularization strategy to train strong classifiers with localizable features,” in ICCV, 2019, pp. 6023–6032.

[61] M. Caron, P. Bojanowski, A. Joulin, and M. Douze, “Deep clustering for unsupervised learning of visual features,” in ECCV, 2018, pp. 139–156.

[62] J. Chang, Z. Lan, C. Cheng, and Y. Wei, “Data uncertainty learning in face recognition,” in CVPR, 2020, pp. 5710–5719.

[63] W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan, “Deep joint rain detection and removal from a single image,” in CVPR, 2017, pp. 1357–1366.

[64] B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang, “Benchmarking single-image dehazing and beyond,” IEEE Trans. Image Process., vol. 28, no. 1, pp. 492–505, 2019.

[65] E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” in CVPRW, 2017, pp. 126–135.

[66] B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, “Enhanced deep residual networks for single image super-resolution,” in CVPRW, 2017, pp. 136–144.

[67] D. R. Martin, C. C. Fowlkes, D. Tal, and J. Malik, “A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics,” in ICCV, vol. 2, 2001, pp. 416–423.

[68] K. Ma, Z. Duanmu, Q. Wu, Z. Wang, H. Yong, H. Li, and L. Zhang, “Waterloo exploration database: New challenges for image quality assessment models,” IEEE Trans. Image Process., vol. 26, no. 2, pp. 1004–1016, 2017.

[69] S. Nah, T. H. Kim, and K. M. Lee, “Deep multi-scale convolutional neural network for dynamic scene deblurring,” in CVPR, 2017, pp. 3883–3891.

[70] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep Retinex decomposition for low-light enhancement,” in BMVC, 2018, pp. 1–12.

[71] H. R. Sheikh, Z. Wang, L. Cormack, and A. C. Bovik, “LIVE image quality assessment database release 2,” https://live.ece.utexas.edu/ research/quality/subjective.htm.

[72] X. Wang, L. Xie, C. Dong, and Y. Shan, “Real-ESRGAN: Training realworld blind super-resolution with pure synthetic data,” in ICCVW, 2021, pp. 1905–1914.

[73] W. H. Richardson, “Bayesian-based iterative method of image restoration,” J. Opt. Soc. Am., vol. 62, no. 1, pp. 55–59, 1972.

[74] L. B. Lucy, “An iterative technique for the rectification of observed distributions,” Astron. J., vol. 79, no. 6, pp. 745–754, 1974.

[75] S. Gu, A. Lugmayr, M. Danelljan, M. Fritsche, J. Lamour, and R. Timofte, “DIV8K: DIVerse 8K resolution image dataset,” in ICCVW, 2019, pp. 3512–3516.

[76] X. Wang, K. Yu, C. Dong, and C. C. Loy, “Recovering realistic texture in image super-resolution by deep spatial feature transform,” in CVPR, 2018, pp. 606–615.

[77] V. Bychkovsky, S. Paris, E. Chan, and F. Durand, “Learning photographic global tonal adjustment with a database of input / output image pairs,” in CVPR, 2011, pp. 97–104.

[78] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019, pp. 4401–4410.

[79] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” in ICCV, 2023, pp. 4015–4026.

[80] J.-B. Huang, A. Singh, and N. Ahuja, “Single image super-resolution from transformed self-exemplars,” in CVPR, 2015, pp. 5197–5206.

[81] W. Zhang, X. Li, X. Chen, X. Zhang, Y. Qiao, X.-M. Wu, and C. Dong, “SEAL: A framework for systematic evaluation of real-world superresolution,” in ICLR, 2024.

[82] A. Abdelhamed, S. Lin, and M. S. Brown, “A high-quality denoising dataset for smartphone cameras,” in CVPR, 2018, pp. 1692–1700.

[83] J. Rim, H. Lee, J. Won, and S. Cho, “Real-world blur dataset for learning and benchmarking deblurring algorithms,” in ECCV, 2020, pp. 184–201.

[84] A. Abuolaim and M. S. Brown, “Defocus deblurring using dual-pixel data,” in ECCV, 2020, pp. 111–126.

[85] T. Wang, X. Yang, K. Xu, S. Chen, Q. Zhang, and R. W. H. Lau, “Spatial attentive single-image deraining with a high quality real rain dataset,” in CVPR, 2019, pp. 12 270–12 279.

[86] H. Zhang, Y. Ba, E. Yang, V. Mehra, B. Gella, A. Suzuki, A. Pfahnl, C. C. Chandrappa, A. Wong, and A. Kadambi, “WeatherStream: Light transport automation of single image deweathering,” in CVPR, 2023, pp. 13 499–13 509.

[87] M. V. Conde, G. Geigle, and R. Timofte, “InstructIR: High-quality image restoration following human instructions,” in ECCV, 2024, pp. 1–21.

[88] J. Hu, L. Jin, Z. Yao, and Y. Lu, “Universal image restoration pre-training via degradation classification,” in ICLR, 2025.

[89] X. Liu, Z. Zhang, X. Wu, C. Feng, X. Wang, L. Lei, and W. Zuo, “Learning real-world image de-weathering with imperfect supervision,” in AAAI, vol. 38, no. 4, 2024, pp. 3738–3746.

[90] H. Li, Z. Zhang, T. Jiang, P. Luo, H. Feng, and Z. Xu, “Real-world deep local motion deblurring,” in AAAI, vol. 37, no. 1, 2023, pp. 1314–1322.

[91] J. Zhang, S. Hu, C. Lu, R. T. Lange, and J. Clune, “Darwin Godel¨ Machine: Open-ended evolution of self-improving agents,” in ICLR, 2026.