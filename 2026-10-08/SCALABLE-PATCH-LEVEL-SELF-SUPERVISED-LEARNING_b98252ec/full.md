# SCALABLE PATCH-LEVEL SELF-SUPERVISED LEARNING

Maximilian Seitzer Maxime Oquab Meta FAIR seitzer@meta.com

Gabriele Trivigno Huy V. Vo

Anton´ın Vobecky´ Oriane Simeoni´

Seungeun Yi Piotr Bojanowski

## ABSTRACT

Self-supervised learning (SSL) at scale produces powerful visual representations. However, most scalable SSL methods rely on ad hoc combinations of multiple objectives and stabilization mechanisms. Taking a step back, we ask if we can design a high-performing, yet principled SSL algorithm. Starting from the multi-view assumption—stipulating that task-relevant content is captured by the information common to different views—we construct an information-theoretic objective decomposing into interpretable terms. This derivation yields JEM, a student-teacher method that learns by aligning corresponding patch representations across views, explicitly regularized by information and structure preservation losses. JEM trains stably from 300M to 7B parameters, and, to our knowledge, is the first latentspace patch-level method demonstrated at 7B scale. Across all scales, JEM reaches strong performance on both global and dense probing tasks, on segmentation benchmarks consistently surpassing the DINOv2 algorithm—an influential foundation for today’s strongest visual SSL methods. Notably, at 7B parameters, it exceeds the performance of DINOv3 on panoptic segmentation, despite being trained on 12× less data without refinement stages. These results demonstrate that we can indeed design an SSL algorithm that learns strong representations, is principled and stable.

## 1 INTRODUCTION

Self-supervised learning has become a standard route to general-purpose visual representations, and the strongest encoders now match or exceed supervised pretraining across a wide range of tasks (Oquab et al., 2024; Simeoni et al.´ , 2026). The best methods combine both image- and patch-level objectives, and have been successfully scaled (Oquab et al., 2024; Simeoni et al.´ , 2026; Fan et al., 2025; Fu et al., 2026). However, this success comes at a price: these methods have grown complex, their objectives do not always compose naturally, and additional stabilization tricks are required to keep them from degrading each other (Oquab et al., 2024; Simeoni et al.´ , 2026). Impressive as these results are, we take them as an invitation to take a step back and reconsider what a principled self-supervised objective could look like.

Many successful methods are built from the idea that different views (altered through augmentations, cropping, masking) of the same image should be described similarly (Chen et al., 2020; Caron et al., 2021; Zbontar et al., 2021). We believe that the underlying multi-view assumption—which suggests that if each view is sufficient for the downstream task, then a representation should retain the information shared between views (Fig. 1a)—can act as a core learning principle. From the multi-view assumption, we derive a multi-view InfoMax objective for student-teacher SSL with two terms: one maximizing cross-view alignment, one preventing information collapse.

We then discuss how to instantiate this objective into JEM, a novel joint-embedding multi-view method trained with a simple patch-level cross-view alignment loss, and two regularization terms, one preventing global collapse, one preserving the spatial structure of the features. We show that JEM trains stably and scales well (Fig. 1b), without additional machinery, to model sizes up to 7B parameters. JEM performs strongly on dense benchmarks, consistently surpassing a DINOv2 baseline on semantic, instance, and panoptic segmentation, while remaining competitive with the strongest models on image-level classification (Fig. 1c). At 7B scale, JEM even exceeds DINOv3’s performance on panoptic segmentation, despite being trained for a shorter time on a smaller dataset without additional refinement steps.

![](images/6ae91d52e4d4639ec284afa6c0dafdb30eb7e0bbb9b3aa2fb1e68161cd79db0e.jpg)

(a) Multi-view information diagram (more in Sec. 3.1), leading to JEM.  
![](images/0905243319fe1c18691cdcbbf20db3ff046ddbdba770fdeda2a6993108c95d76.jpg)  
(b) Mean segmentation (over six datasets) scales with model size.

![](images/05dcd6566f853a04deea1456209e52128a3960394971785b9b1e9bc1fc4c7b33.jpg)  
(c) Dense vs. global performance across SSL methods (VIT-L, IN1k).  
Figure 1: We introduce JEM, a student-teacher SSL objective grounded in information theory. Following the multi-view principle (a), it maximizes the information a student representation retains about what views share. Average scores over semantic, panoptic and instance segmentation at different scales (b) show that JEM scales effectively and consistently outperforms DINOv2. (c) JEM surpasses methods using patch- and/or image-level objectives on dense tasks while remaining competitive on classification. JEM learns strong representations, is principled and stable.

To summarize, we (1) derive a principled information-theoretic SSL objective leading to JEM, a joint-embedding multi-view method trained with a patch-level objective, that (2) trains stably across scales up to 7B parameters and (3) outperforms the DINOv2 algorithm in controlled comparisons on dense prediction tasks while remaining competitive on classification, even exceeding the DINOv3 7B foundation model on panoptic segmentation.

## 2 RELATED WORK

Self-Supervised Visual Representation Learning Modern SSL falls into four broad families. Contrastive methods (He et al., 2020; Chen et al., 2020) pull together representations of different views of the same image while pushing apart representations of other images. Redundancy-reduction methods (Zbontar et al., 2021; Bardes et al., 2022a) enforce view invariance while explicitly preventing collapse across dimensions. Self-clustering approaches such as SwAV Caron et al. (2020) and DINO (Caron et al., 2021) learn cluster assignments and align those across views. Masked prediction instead removes part of the input and predicts the missing information: MAE (He et al., 2022) predicts pixels, while I-JEPA (Assran et al., 2023) predicts latent representations of spatially localized target blocks, and iBOT (Zhou et al., 2022) and CAPI (Darcet et al., 2025) predict cluster assignments for masked patches.

Recent methods increasingly differ in whether they rely on a single objective family or combine complementary objectives. I-JEPA and CAPI are examples of single-family latent prediction approaches: I-JEPA predicts latent representations of masked target blocks, while CAPI predicts latent cluster assignments for masked patches. DINOv2 (Oquab et al., 2024), in contrast, combines global DINO self-distillation with patch-level iBOT prediction and KoLeo (Sablayrolles et al., 2019) regularization; DINOv3 (Simeoni et al.´ , 2026) even adds a further term, Gram anchoring, to counter an effect observed during long training at large scale: global performance keeps improving while dense feature maps progressively lose spatial locality. Are such combinations necessary for scaling SSL? Our work explores a principled information-theoretic objective derived from maximizing mutual information between representations of different views. Unlike methods with separate global and local objectives, our model has no class token and applies the objective exclusively to patch representations, while still matching the performance of multi-objective methods such as DINOv2.

Information theory in SSL Information-theoretic SSL spans several closely related perspectives (Shwartz-Ziv & LeCun, 2024). The original InfoMax principle (Linsker, 1988) prescribes maximizing the information the representation retains about its input, implemented for representation learning by Deep InfoMax (Hjelm et al., 2019). Other works instead maximize the information between different views of the data (van den Oord et al., 2018; Bachman et al., 2019; Tian et al.,

2020a; Henaff, 2020), yielding the multi-view InfoMax principle; we also use it as our starting point. However, instead of estimating mutual information with contrastive objectives or separate critics, we use a variational approximation which naturally fits the student-teacher paradigm common in modern SSL. Complementarily, the information bottleneck principle suggests retaining predictive information while compressing superfluous information (Tishby et al., 1999; Alemi et al., 2017). Federici et al. (2020) combine it with multi-view InfoMax by preserving shared and suppressing view-specific information. In contrast, we derive an explicit anti-collapse term that maintains view-specific information; however, our use of discrete representation targets can be seen as an architectural bottleneck. The limits of mutual information as an objective for SSL have been discussed by Tschannen et al. (2020); Tian et al. (2020b); Tsai et al. (2021). For us, the multi-view InfoMax objective informs practical loss design, acknowledging that learning good representations also relies on inductive biases from architecture and view construction. We review more related work in Sec. A.1.

## 3 METHOD

We aim to build an SSL method that is principled and stable, without any sacrifice in downstream performance. Starting from the multi-view assumption, we derive an information-theoretic objective $( \mathrm { S e c } . 3 . 1 )$ . From this objective, we derive an alignment loss (Sec. 3.2.1) and regularization terms that mitigate its collapse modes (Secs. 3.2.2 and 3.2.3). In Sec. 3.3, we then discuss the careful design of the views, which we found to be crucial for best performance. Throughout the section we provide detailed design choices that make JEM, a new joint-embedding SSL method yielding results competitive with more complex and highly tuned SSL objectives.

## 3.1 THE MULTI-VIEW INFOMAX OBJECTIVE

The multi-view assumption stipulates that any view of the data alone contains sufficient information for the downstream tasks (Sridharan & Kakade, 2008). It follows that all task-relevant information must be contained in the shared information across views. Whether the multi-view assumption holds in practice is dependent on the choice of views and the downstream tasks, and its limitations are known (Tsai et al., 2021; Shwartz-Ziv & LeCun, 2024). Still, we use it as an idealized principle that motivates a criterion for learning representations.

Setup: Given a sample $X ,$ , we apply masking, cropping, and photometric distortions to generate views. For generality, we assume an asymmetric student-teacher setup: given paired views $X _ { S } , X _ { T }$ , the teacher defines a target $T \sim p ( \cdot \mid X _ { T } )$ , and the student encoder $f$ outputs a representation R for $X _ { S }$ To predict the target $\mathbf { \bar { \rho } } _ { T }$ from $X _ { S } ,$ , a projection head maps the representation to a predictive distribution $\boldsymbol { q } ( \cdot \ | \ X _ { S } )$ ) over the target space. In practice, $T$ is a collection ofdiscrete variables as discussed later.

InfoMax objective The information we would like to capture in the representation R is the studentteacher shared mutual information (MI) $I ( X _ { S } ; X _ { T } )$ . The teacher exposes information about its view $X _ { T }$ through the target $T ,$ . By the data processing inequality, we have $I ( X _ { S } ; X _ { T } ) \ge I ( X _ { S } ; T ) \ge$ $I ( R ; T )$ . This yields the multi-view InfoMax objective of maximizing $I ( R ; T )$ , i.e., the shared information of the representation R with the target $T$ (see Fig. 1a). We can maximize this objective using a variational lower bound (Barber & Agakov, 2003), where $H ( T )$ is the entropy of the target and $D _ { \mathrm { K L } }$ is the Kullback-Leibler divergence (all derivations in Sec. A.4):

$$
I ( R ; T ) \geq H ( T ) + \mathbb { E } [ \log q ( T \mid X _ { S } ) ]\tag{1}
$$

$$
= \underbrace { I ( X _ { T } ; T ) } _ { \mathrm { ~ \normalfont ~ = ~ } \mathrm { ~ \normalfont ~  ~ } } - \underbrace { \mathbb { E } _ { X _ { T } , X _ { S } } D _ { \mathrm { K L } } ( p ( T \mid X _ { T } ) \mid \mid q ( T \mid X _ { S } ) ) } _ { \mathrm { ~ \normalfont ~ = ~ } \mathrm { ~ \normalfont ~ ( ~ X _ { T } ~ , ~ X _ { S } ~ ) ~ } } .\tag{2}
$$

$$
\mathrm { a n t i - c o l l a p s e } \qquad \mathrm { a l i g n m e n t }
$$

The lower bound showcases two elements intuitively important for any SSL algorithm: an anticollapse term, measuring the amount of information the targets T retain about the input $X _ { T }$ , and an alignment term, measuring how well the student approximates the teacher. In practice, we achieved the best performance with the teacher an EMA of the student, and therefore only optimize Eq. (2) with respect to the student parameters. Thus, the anti-collapse term $I ( X _ { T } ; T )$ on the teacher side is not directly maximized, which we resolve in Secs. 3.2.2 and 3.2.3.

Granularity of the objective The KL term in Eq. (2) is flexible regarding the granularity of the targets; for instance, we could align the two views through a single image-level target. Instead, to leverage more information, we model the targets as a collection of L elements $T = ( T ^ { ( 1 ) } , \dots , T ^ { ( L ) } )$ where ℓ indexes the targets by patch and feature layer. In this way, the training signal is both spatial and stems from features from different depths of the model. In the following, we assume that student and teacher distributions factorize over the elements. The alignment term in Eq. (2) then decomposes into a sum of element-wise alignment terms.

Preventing collapse From Eq. (2), we can see that collapse occurs when $I ( X _ { T } ; T ) = H ( T ) -$ $H ( T \mid X _ { T } )$ goes towards 0. Preventing collapse thus involves maintaining targets with high marginal entropy $H ( T )$ and low conditional entropy $H ( T \mid X _ { T } )$ . Intuitively, the targets should be spread around the data distribution, while remaining predictable for any particular input. For collections of elements, $I ( X _ { T } ; T )$ decomposes as

$$
I ( X _ { T } ; T ) = \sum _ { \ell = 1 } ^ { L } \underbrace { I ( X _ { T } ; T ^ { ( \ell ) } ) } _ { \mathrm { p e r - e l e m e n t ~ i n f o } } - \underbrace { \mathrm { T C } ( T ) } _ { \mathrm { r e d u n d a n c y } } + \underbrace { \mathrm { T C } ( T \mid X _ { T } ) } _ { 0 } ,\tag{3}
$$

where the total correlation $\mathrm { T C } ( T )$ measures the redundancy among elements and $T C ( T \mid X _ { T } ) = 0$ by assuming a factorized $p ( T \mid \dot { X } _ { T } )$ . Hence, preventing collapse can be achieved by maintaining high per-element information while limiting the uncontrolled growth of redundancy (Secs. 3.2.2 and 3.2.3, respectively).

## 3.2 JEM: A NEW JOINT-EMBEDDING MULTI-VIEW OBJECTIVE

In this section, we discuss our SSL method JEM. Motivated by maximizing the multi-view InfoMax objective, JEM uses a principled loss function consisting of an alignment and regularization terms derived from Eqs. (2) and (3). We provide an overview of our method in Fig. 2.

## 3.2.1 PATCH-WISE ALIGNMENT LOSS

For a given image, we produce a pair of overlapping student-teacher views $x _ { S } , x _ { T }$ . Both views are placed on the same patch grid so overlapping patches correspond one-to-one. We denote ${ \mathcal { O } } _ { S }$ as the set of indices of student patch-layer pairs also appearing in the teacher view, and m : $\mathcal { O } _ { S } \mapsto \mathcal { O } _ { T }$ the function mapping the student indices to the corresponding teacher patch-layer pairs with indices $\mathcal { O } _ { T }$ This choice of mapping enforces an invariance to the index of the elements. The alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ is the element-wise decomposition of the KL in Eq. (2) over ${ \mathcal { O } } _ { S } \colon$

$$
\mathcal { L } _ { \mathrm { a l i g n } } ( x _ { S } , x _ { T } ) = \frac { 1 } { | \mathcal { O } _ { S } | } \sum _ { \ell \in \mathcal { O } _ { S } } D _ { \mathrm { K L } } \left( p ( T ^ { ( m ( \ell ) ) } \mid x _ { T } ) \mid \mid q ( T ^ { ( \ell ) } \mid x _ { S } ) \right) .\tag{4}
$$

## 3.2.2 PREVENTING GLOBAL COLLAPSE

To maximize the lower-bound in Eq. (2), our goal is to keep $I ( X _ { T } ; T )$ large. As we use discrete targets, directly estimating $I ( X _ { T } ; T )$ is hard for multi-variate T. Instead, we maximize element-wise information terms $I ( X _ { T } ; T ^ { ( \ell ) } )$ , with ℓ the element index, using the decomposition in Eq. (3). We implement this with an explicit regularization loss on the student. Through EMA updates, the student acts as a proxy for the teacher. In particular, we maximize $I ( X _ { S } ; S ^ { ( \ell ) } ) { \stackrel { - } { = } } H ( S ^ { ( \ell ) } { \stackrel { - } { ) } } - H ( S ^ { ( \ell ) } \mid X _ { S } )$ where $S ^ { ( \ell ) }$ follows the K-class categorical $q ( \cdot \mid Z _ { S } ^ { ( \ell ) } )$ parametrized by the student head output $Z _ { S } ^ { ( \ell ) }$ for patch ℓ. Given head outputs $\mathbf { z } ^ { ( \ell ) }$ for a batch of N inputs x, we can compute an estimate of the element-wise mutual information as:

$$
\hat { I } ( \mathbf { x } ; \mathbf { z } ^ { ( \ell ) } ) = \underbrace { - \sum _ { j = 1 } ^ { K } \bar { q } _ { j } \log \bar { q } _ { j } } _ { \mathrm { m a r g . } \mathrm { e n t r o p y } } - \underbrace { ( - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { K } q _ { j } ( z _ { i } ^ { ( \ell ) } ) \log q _ { j } ( z _ { i } ^ { ( \ell ) } ) ) } _ { \mathrm { c o n d . } \mathrm { e n t r o p y } } ,\tag{5}
$$

where $\begin{array} { r } { \bar { q } _ { j } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } q _ { j } ( z _ { i } ^ { ( \ell ) } ) } \end{array}$ denotes the mean batch probability for class $j$ and patch ℓ. The global collapse regularization loss is then given as:

$$
\mathcal { R } _ { \mathrm { g l o b a l } } ( \mathbf { x } ) = - \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \hat { I } ( \mathbf { x } ; \mathbf { z } ^ { ( \ell ) } ) .\tag{6}
$$

![](images/9e72eed48ebea9f781812544995d0f019c5780b3ede059299042ecd860d6bd3f.jpg)  
Figure 2: From multi-view InfoMax to JEM. Maximizing the variational lower bound on the multi-view InfoMax objective in Eq. (2) yields two requirements: cross-view alignment between teacher and student distributions $p$ and $q ,$ and informative, non-collapsed targets $T \sim q ( \cdot \mid X _ { T } )$ . JEM implements this with three loss terms: the $\mathrm { K L }$ term $\mathcal { L } _ { \mathrm { a l i g n } }$ aligns student patch ℓ with matching teacher target $m ( \ell ) ; \mathcal { R } _ { \mathrm { g l o b a l } }$ maintains per-patch mutual information through marginal and conditional entropy; and $\mathcal { R } _ { \mathrm { s t r u c t } }$ seeks to control the total correlation of the targets $\mathrm { T C } ( \breve { T } )$ by preserving the geometry of teacher features of layer k in the final student representation. Please zoom for details.

However, this regularization term only partially covers the anti-collapse term $I ( X _ { T } ; T )$ in Eq. (2) due to the element-wise computation of $I ( X _ { S } ; S ^ { ( \ell ) } )$ . We therefore define an additional regularization term which covers the structural collapse mode.

## 3.2.3 PREVENTING STRUCTURAL COLLAPSE

In the last section, we derived a regularizer to maintain the first term of Eq. $( 3 ) , \textstyle \sum _ { \ell = 1 } ^ { L } I ( X _ { T } ; T ^ { ( \ell ) } )$ preventing global collapse of the targets. In this section, we address the second term, the redundancy $\mathrm { T C } ( T )$ . Intuitively, the redundancy becomes large when all $T ^ { ( \ell ) }$ encode similar information, which we term a structural collapse. For example, if each patch target encodes the same input-dependent category, the jointly-encoded information $I ( X _ { T } ; T )$ is the same as that of a single target $I ( X _ { T } ; T ^ { ( \ell ) } )$ . Maximizing the sum of element-wise information $I ( X _ { T } ; T ^ { ( \ell ) } )$ does not stop this collapse mode; thus, we design a mechanism to prevent the redundancy from growing too large.

Estimating the total correlation TC(T) of a large set of discrete variables is intractable. Instead, we encourage the representation R to retain the dependency structure contained in the input, using this as a proxy for controlling $T C ( T )$ . As the true dependencies are inaccessible, we use the similarities of representations from different layers of the teacher as a proxy; this rests on the assumption that different layers encode different levels of structure. The representation must then match these similarities after a learned non-linear transformation. Specifically, given student representation $r ,$ and teacher representation $h _ { k }$ and transformation $g _ { k }$ at layer $k ,$ the loss for a pair of elements at indices $i , j$ can be written as

$$
\mathcal { L } _ { \mathrm { s t r u c t } } ^ { ( i , j ) } ( \boldsymbol { r } , \boldsymbol { h } _ { k } ) = \left( \cos ( g _ { k } ( \boldsymbol { r } ^ { ( i ) } ) , g _ { k } ( \boldsymbol { r } ^ { ( j ) } ) ) - \cos ( h _ { k } ^ { ( m ( i ) ) } , h _ { k } ^ { ( m ( j ) ) } ) \right) ^ { 2 }\tag{7}
$$

where cos denotes the cosine similarity and $( i , j )$ are restricted to the subset of elements coming from the same layer of $\mathcal { O } _ { S } \times \mathcal { O } _ { S }$ . Intuitively, by being able to reconstruct the early layer similarities, the representation avoids the collapse to a single, redundant vector, and is still free to encode higher-level semantic structure such as similar object patches. Given a set of target teacher layers $\kappa .$ the total loss is then averaged over target layers and patch pairs:

$$
\mathcal { R } _ { \mathrm { s t r u c t } } ( x _ { S } , x _ { T } ) = \frac { 1 } { | K | | \mathcal { O } _ { S } | ( | \mathcal { O } _ { S } | - 1 ) } \sum _ { k \in \mathcal { K } } \sum _ { i \in \mathcal { O } _ { S } } \sum _ { \stackrel { j \in \mathcal { O } _ { S } } { i \neq j } } \mathcal { L } _ { \mathrm { s t r u c t } } ^ { ( i , j ) } ( r , h _ { k } ) .\tag{8}
$$

## 3.3 VIEW CONSTRUCTION

While often relegated to implementation details, the algorithm used to construct views of the input is a critical design choice in SSL (Chen et al., 2020; Tian et al., 2020b; He et al., 2022). For images, one obvious way to create views is cropping (Bachman et al., 2019; Chen et al., 2020), commonly combined with photometric transformations (Chen et al., 2020; Grill et al., 2020). The resulting crops overlap to varying degrees, leading to context-invariant representations. Additionally, Caron et al. (2020; 2021) exploit smaller local crops, which induce local-to-global correspondences. Maskingbased methods, on the other hand, typically use fully overlapping student and teacher crops (He et al., 2022; Assran et al., 2022), where masking controls which patches are visible to the encoder.

From the perspective of the multi-view assumption, cropping and masking are complementary mechanisms for controlling the information shared between views. In this work, we treat them in a unified manner. We build several student views which partially overlap with the teacher view and are, in addition, masked, such that the views differ both in spatial support and patch visibility. The alignment loss of Eq. (4) is applied to all overlapping patches, whether masked or visible. Note that traditional masked prediction is a special case of this construction, obtained with a single student view and the student/teacher crops coinciding. Interestingly, we found that the method performs well without the use of masks, showing that patch-level SSL is possible using only cropping (see Sec. 4.2).

Concretely, we implement this design by sampling shifted but overlapping student and teacher crops aligned on a common patch grid, as shown in Fig. 3. We integrate local crops (Caron et al., 2020) by sampling smaller student than teacher crops, obtaining local-to-global prediction pairs. For a fraction of student views, we sample and apply a patch mask. Student and teacher views are also independently transformed with a standard set of photometric transformations (Grill et al., 2020). Further implementation details are provided in Sec. A.3.2.

![](images/dca995a390a4b9c7a65822a88edd9a28f8cba91a2502f9058074023847233604.jpg)  
Figure 3: View construction.

## 4 EXPERIMENTS

This section presents the experimental results validating JEM. We first provide our implementation details (Sec. 4.1) and ablate the individual components (Sec. 4.2). We evaluate the quality of the resulting features against models from the literature (Sec. 4.3) and study scaling behaviour (Sec. 4.4).

## 4.1 IMPLEMENTATION DETAILS

Architecture Our encoder is a Vision Transformer (ViT) (Dosovitskiy et al., 2021), at three model scales, ViT-L, ViT-g and ViT-7B. All models use 16 × 16 patches, omit the class token, and prepend four learned register tokens (Darcet et al., 2024). We follow the open source implementation of Simeoni et al.´ (2026), and inherit most hyperparameters from the published DINOv3 configurations.

Training Details For each image, we sample two view sets, each containing a global student view, four local student views and one teacher view. Global and teacher views are 256 × 256 pixels and local views $1 1 2 \times 1 1 2$ . In 80% of the global and local views we mask 65% and 50% of the patches respectively; the rest of the views are left unmasked. We use K = 4096 classes for the categorical distributions. We train ViT-g for 500k iterations with AdamW, a global batch size of 3072, and a constant learning rate schedule with linear warm-up. The teacher is an exponential moving average of the student with fixed momentum 0.994. The global collapse term $\mathcal { R } _ { \mathrm { g l o b a l } }$ has a coefficient of 0.1 with a linear decay from 1 over the first 12.5k steps. The structural collapse term $\mathcal { R } _ { \mathrm { s t r u c t } }$ has a coefficient increased linearly from 0.05 to 1 over the course of training, with a linear warm-up from 0 over the first 12.5k steps. More hyperparameter details in Sec. A.3.3.

We train on the public ImageNet-1k (Russakovsky et al., 2015) and ImageNet-22k (Deng et al., 2009) datasets for comparability with prior work. For larger scale, we follow Fu et al. (2026) and reproduce the curated LVD-142M dataset (Oquab et al., 2024), denoted “A140M” (Sec. A.3.4 for details).

Evaluation Protocol We evaluate all methods with a frozen backbone and simple probing setups. For classification, we report attention probe accuracy on ImageNet-1k (Russakovsky et al., 2015). For semantic segmentation, we report linear probe mIoU on ADE20k (Zhou et al., 2019), Pascal VOC (Everingham et al., 2010), and Cityscapes (Cordts et al., 2016); for monocular depth, we report linear probe RMSE on NYUv2 (Silberman et al., 2012) and KITTI (Geiger et al., 2012). We also consider more challenging benchmarks testing combined class- and instance understanding, and report mask AP for instance segmentation on COCO (Lin et al., 2014), and PQ for panoptic segmentation on COCONut (Deng et al., 2024) and ADE20k. See Sec. A.5.1 for details.

Table 1: Ablation study. We isolate the contributions of view construction (a), the composition of the patch-level KL alignment loss (b), and the terms in the complete objective (c). We train ViT-L models on ImageNet-22k for 250k steps and report attention-probe accuracy on IN1k and segmentation mIoU on ADE20k. Tables compare the full model with variants removing (w/o) individual components.  
(a) View construction.
<table><tr><td>Configuration</td><td>IN1k ADE</td></tr><tr><td>Full JEM model</td><td>82.8 48.1</td></tr><tr><td>w/o local crops</td><td>81.2 46.2</td></tr><tr><td>w/o masking</td><td>82.3 45.3</td></tr><tr><td>w/o stud.-teach. shift</td><td>82.8 46.6</td></tr><tr><td>w/o independ. aug.</td><td>74.8 31.4</td></tr></table>

(b) KL alignment.
<table><tr><td>Configuration</td><td>IN1k ADE</td></tr><tr><td>Full JEM model</td><td>82.8 48.1</td></tr><tr><td>w/o multi-layer loss</td><td>81.7 48.3</td></tr><tr><td>w/o visible-patch loss</td><td>70.8 28.1</td></tr><tr><td>w/o masked-patch loss</td><td>81.3 46.0</td></tr><tr><td>w/o EMA</td><td>83.0 47.0</td></tr></table>

(c) Objective.
<table><tr><td>Configuration</td><td>IN1k</td><td>ADE</td></tr><tr><td>Full JEM model</td><td>82.8</td><td>48.1</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>63.1</td><td>22.4</td></tr><tr><td>w/o  $\mathcal { R } _ { \mathrm { g l o b a l } }$ </td><td>10.3</td><td>0.5</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \mathcal { R } _ { \mathrm { s t r u c t } }$ </td><td>82.3</td><td>48.0</td></tr></table>

![](images/4d02b36f69895c2fd21acb9317991329abbe2d5d404e9a35813832a997e443b7.jpg)

![](images/c43c0cb6d609436ad31902a95d3753efc3f275258c0ba6f1bfb62b585f1fe7d9.jpg)  
Figure 4: Impact of the global regularization weight on IN-1k and ADE20k performance. Models use ViT-L, IN-22k, and 250k steps.

![](images/9a898d83761b7666a5c0dd956be3c0b7aa1a2cdd1c5a8a80fe8534d938c7e23d.jpg)  
Figure 5: Collapse prevention. (Left) Teacher patch mutual information (MI) collapses when disabling $\mathcal { R } _ { \mathrm { g l o b a l } } .$ . (Right) Mean ℓ<sub>2</sub>-distance between patches and the image mean patch representation; low means loss of spatial structure. ViT-g on A140M.

## 4.2 ABLATIONS

In this section, we study the impact of the different components of JEM. In all experiments, if not stated otherwise, we train ViT-L-sized models on the ImageNet-22k dataset for 250k iterations. We evaluate on ImageNet-1k (IN1k) classification and ADE20k (ADE) semantic segmentation.

View construction (Tab. 1a) Overall, we find JEM to be robustness to individual changes in the view construction: JEM can deliver strong performance without using masked patches, local crops, and shifting student and teacher global views (i.e. fully overlapping views), vital components of other methods (Caron et al., 2021; Oquab et al., 2024). Still, these techniques are complementary and all needed to achieve good overall performance. However, removing per-view independent photometric augmentations such as random color shifts does impact JEM and greatly degrades the performance on both global and dense tasks. We believe this is because these augmentations induce an invariance to pixel-level details, enabling the model to learn semantic local features instead of a shortcut solution.

Alignment loss (Tab. 1b) Multi-layer prediction, masked-patch prediction, and an EMA teacher are not individually essential for strong performance: removing any one of them changes IN1k accuracy by at most 1.5 points and ADE20k mIoU by at most 2.1 points. This is remarkable, as masked prediction is usually seen as a key component of patch-level SSL. Similarly, EMA teachers were previously critical to enable both self-clustering (Caron et al., 2021) and latent-space masked prediction (Assran et al., 2023). However, it is important to include the visible patches in $\mathcal { L } _ { \mathrm { a l i g n } } ;$ removing them costs 20 points on ADE20k. We hypothesize this is because the loss prevents the patch representation from drifting away from spatial alignment.

Table 2: Comparison to prior SSL methods. Classification reports IN-1k attention-probe accuracy. Semantic, instance, and panoptic segmentation results are mIoU, mAP, and PQ, respectively. Depth reports RMSE (lower better). Tokens is the approximate number of tokens seen during training in trillions. ‘\*’ denotes the DINOv2 algorithm trained with the JEM setup for 125k steps on IN-1k and 500k steps on IN-22k or A140M. † denotes backbone training or fine-tuning above 256 px.
<table><tr><td></td><td></td><td></td><td></td><td>Class.</td><td>Sem. Seg.</td><td></td><td>Inst. Seg.</td><td>Pan. Seg.</td><td></td><td>Depth</td></tr><tr><td>Model</td><td>Size Dataset</td><td></td><td>Tokens (T)</td><td>IN1k</td><td></td><td>ADE VOC City.</td><td>COCO</td><td>ADE COCONut NYU KITTI</td><td></td><td></td></tr><tr><td>MAE</td><td>L IN-1k</td><td></td><td>0.40</td><td>78.4</td><td>31.6</td><td>66.1 53.1</td><td>6.0</td><td>6.5</td><td>9.7</td><td>0.489 2.903</td></tr><tr><td>MSN</td><td>L IN-1k</td><td></td><td>0.43</td><td>76.3 24.3</td><td>55.0</td><td>46.4</td><td>3.2</td><td>4.4</td><td>6.7</td><td>0.6563.696</td></tr><tr><td>data2vec 2.0</td><td>L IN-1k</td><td></td><td>0.60</td><td>79.5 27.4</td><td>52.748.6</td><td></td><td>2.1</td><td>4.8</td><td>7.1</td><td>0.512 2.901</td></tr><tr><td>CAPI</td><td>L IN-1k</td><td></td><td>2.10</td><td>82.9 31.9</td><td>66.0</td><td>50.6</td><td>10.7</td><td>12.7</td><td>14.7</td><td>0.361 2.604</td></tr><tr><td>DINOv2*</td><td>L IN-1k</td><td></td><td>0.23</td><td>83.8 45.7</td><td>83.3</td><td>66.5</td><td>12.1</td><td>18.0 22.1</td><td></td><td>0.408 2.718</td></tr><tr><td>JEM</td><td>L IN-1k</td><td></td><td>0.23</td><td>82.8 46.4</td><td>84.9</td><td>67.4</td><td>13.5 20.1</td><td>24.3</td><td></td><td>0.3962.627</td></tr><tr><td>I-JEPA</td><td>g</td><td>IN-22k</td><td>0.12</td><td>78.8</td><td>34.2 73.6</td><td>52.4</td><td>6.3</td><td>7.4</td><td>10.0</td><td>0.471 2.994</td></tr><tr><td>Franca</td><td>g</td><td>IN-22k</td><td>1.74</td><td>86.0 45.9</td><td>81.9</td><td>67.0</td><td>11.9</td><td>13.9</td><td>16.4</td><td>0.355 2.505</td></tr><tr><td>DINOv2*</td><td>g IN-22k</td><td></td><td>1.39</td><td>86.5 50.0</td><td>83.6</td><td>68.5</td><td>13.0</td><td>18.1 21.1</td><td></td><td>0.3432.494</td></tr><tr><td>JEM</td><td>g IN-22k</td><td></td><td>1.39 85.5</td><td>51.2</td><td>85.9</td><td>69.6</td><td>14.7 22.4</td><td>25.7</td><td></td><td>0.363 2.472</td></tr><tr><td colspan="9">&gt;1B, trained on large datasets</td><td></td></tr><tr><td>DINOv2*</td><td>g</td><td>A140M</td><td>1.39</td><td>86.9 50.5</td><td>83.669.4</td><td></td><td>12.7</td><td>18.7</td><td>21.4</td><td>0.3392.455</td></tr><tr><td>DINOv2 (reg.)†</td><td>g</td><td>LVD-142M</td><td>1.83</td><td>87.3 49.5</td><td>83.1</td><td>69.1</td><td>12.3</td><td>16.7</td><td>18.6</td><td>0.357 2.569</td></tr><tr><td>LingBot-Vision</td><td>g</td><td>Unk.-163M</td><td>1.86</td><td>86.5 53.6</td><td>85.9</td><td>74.1</td><td>15.2</td><td>22.2</td><td>25.0</td><td>0.2872.127</td></tr><tr><td>JEM</td><td>g A140M</td><td></td><td>1.39</td><td>85.6 52.3</td><td></td><td>86.1 71.2</td><td>14.7</td><td>22.9</td><td>26.1</td><td>0.3292.324</td></tr><tr><td>V-JEPA 2.1†</td><td>G</td><td>VM-163M</td><td>0.24</td><td>83.2 47.8</td><td>84.3</td><td>68.3</td><td>12.5</td><td>15.5</td><td>19.6</td><td>0.277 2.167</td></tr><tr><td>WebSSL</td><td>7B MC-8B</td><td></td><td>7.23</td><td>87.5 42.2</td><td>75.7</td><td>61.5</td><td>9.3</td><td>9.9</td><td>13.7</td><td>0.4602.945</td></tr><tr><td>DINOv2*</td><td>7B A140M</td><td></td><td>1.85</td><td>87.7 51.9</td><td>84.5</td><td>71.1</td><td>14.0</td><td>17.9</td><td>22.0</td><td>0.289 2.218</td></tr><tr><td>DINOv3†</td><td>7B</td><td>LVD-1689M</td><td>4.06</td><td>89.0 55.9</td><td>86.7 75.2</td><td></td><td>16.0</td><td>21.6</td><td>27.0</td><td>0.2782.242</td></tr><tr><td>JEM</td><td>7B</td><td>A140M</td><td>1.85</td><td>86.9</td><td>54.4</td><td>87.1 73.6</td><td>15.6</td><td>23.9</td><td>27.3</td><td>0.297 2.125</td></tr></table>

Objective (Tab. 1c) We ablate the three terms of our objective. First, removing the alignment loss L<sub>align</sub> (Eq. (4)) sharply reduces IN1k accuracy from 82.8 to 63.1 and ADE20k mIoU from 48.1 to 22.4, showing that the alignment loss carries a critical training signal. Second, removing global collapse regularization $\mathcal { R } _ { \mathrm { g l o b a l } }$ (Eq. (6)) is catastrophic, and leads the model to collapse, attaining 10.3 IN1k accuracy. In Fig. 4, we also examine how sensitive performance is to this regularizer’s loss weight. We observe that the concrete weight matters only in that it must be non-zero: a weight of 0 yields degenerate representations, while any value across a wide range performs comparably. Third, we find the effect of structural regularization $\mathcal { R } _ { \mathrm { s t r u c t } }$ (Eq. (8)) to be marginal at ViT-L scale (-0.5 on IN1k), but crucial when scaling the model size further. Next, we investigate its effect with larger model sizes, and find that it is important to prevent the progressive loss of spatial diversity.

Preventing collapse (Fig. 5) Now we consider larger models (ViT-g) which suffer more and faster from collapse. First, in Fig. 5 (left), we show the evolution of the per-patch MI $I ( X _ { T } ; T ^ { ( \ell ) } )$ over training when disabling global regularization $\mathcal { R } _ { \mathrm { g l o b a l } } ;$ in this case, the target distributions collapse to a uniform or constant distribution over categories. Second, in Fig. 5 (right), we show how the average patch $\ell _ { 2 } \cdot$ distance to the per-image mean patch develops over training with or without the structural regularizer. Low scores indicate that patch representations become more spatially uniform. Without regularization, increasing degradation starts after 250k iterations, leading to performance degradation. In contrast, the regularized model keeps patch representations distinct from their spatial average, thereby maintaining stronger performance on dense tasks (+2.7 ADE20k mIou).

## 4.3 COMPARISON TO OTHER SSL ALGORITHMS

In Tab. 2, we compare JEM to other SSL methods on the same data, training ViT-L on ImageNet-1k and ViT-g on ImageNet-22k. We also report results with models trained on larger, not directly comparable datasets and discuss them in Sec. 4.4. For a controlled comparison, we re-train DINOv2 models (Oquab et al., 2024) (denoted “DINOv2\*”), using the same setup as JEM (similar to training DINOv3 without gram and high-resolution finetuning steps). This baseline is relevant as DINOv2, combining global and local losses, serves as a foundation for many of today’s best SSL methods.

![](images/952cd424b4072022eb6806d4ad4720d0104aad7d3184c9012691bc22916a0c62.jpg)  
Training FLOPs

![](images/2112ceb6d34d477bb4c4608a74d61f07be210865b32621a8e35c1f7b9ed30720.jpg)  
Training FLOPs

![](images/766802e6d1d6b1076af27fa73971ee921fa6b943c43316ffa4e9af466742dfd2.jpg)  
Training FLOPs

![](images/25d2488f643d4d496d09fd7745fde2db65e12ccb05062d4b5892528edee7a428.jpg)  
Training FLOPs  
JEM DINOv2\* DINOv3 7B<sup>†</sup>  
Figure 6: Scaling model size. Performance as a function of approximate training compute, averaged over datasets for each benchmark type. DINOv2\* and JEM are trained by us for 500k steps on A140M. The dashed purple lines show DINOv3 ViT-7B.

For both model sizes, JEM outperforms all other methods on six out of nine tasks, with consistent gains across the full range of dense prediction: from coarse semantic segmentation to fine-grained instance and panoptic segmentation. On global classification, we observe a small gap to DINOv2; still a strong result as DINOv2 was heavily designed to obtain the best global performance. Interestingly, JEM scales effortlessly to ViT-g. Moreover, its advantage over DINOv2 increases at scale, e.g. with a 4 PQ pt gain on both challenging panoptic segmentation tasks; the gap is maintained at 7B (Sec. 4.4).

## 4.4 SCALING JEM

An important aspect of modern ML algorithms is their scalability with model size and data. In SSL, scaling is often challenging due to feedback dynamics of self-supervision; different model scales show different behavior (Simeoni et al.´ , 2026), and approaches that work well on small scales do not necessarily train well (or at all) beyond 1B parameters. Here we show that JEM scales well.

We train 300M (ViT-L), 1B (ViT-g), and 7B (ViT-7B) parameter models with JEM and DINOv2 on our A140M dataset for 500k steps. Fig. 6 plots task-averaged performance as a function of training FLOPs (individual benchmarks in appendix Fig. 7). We also include DINOv3 ViT-7B results (dashed line), though we note it is trained for twice as long on a 12× larger dataset and with additional refinement steps (gram and high-resolution fine-tuning). JEM trains stably even at 7B scale, with performance improving significantly when model size grows. To our knowledge, this is the first exclusively patch-level latent-space method that trains well at this scale. Compared to the DINOv2 baseline, JEM exhibits comparable or better scaling behavior and outperforms it at all model scales.

Tab. 2 also lists models above 1B parameters trained on different large datasets which are thus not directly comparable. Notably, JEM outperforms the best SSL method DINOv3 on several datasets in semantic and panoptic segmentation, as well as on depth estimation, despite DINOv3’s longer training, larger dataset and its refinement steps (gram anchoring and high resolution fine-tuning)— showing the strength of our training paradigm. In Figs. 9 and 10, we also plot qualitative PCA and patch cosine similarity results highlighting the quality of the patch features, obtained without any refinement steps.

## 5 CONCLUSION

In this work, we introduced JEM, a principled and stable patch-level joint-embedding multi-view method that learns strong representations. JEM is derived from a multi-view InfoMax objective motivated by the multi-view assumption. This objective leads to three loss terms: one aligning representations across views, one preventing global collapse and one preserving structural integrity. JEM trains stably from 300M to 7B parameters—to our knowledge, the first exclusively patch-level latent-space method demonstrated at this scale—and achieves impressive results on segmentation tasks while being competitive on classification and depth estimation. These results suggest that multi-objective SSL’s complexity is not necessary for strong performance, and establish principled patch-level objectives as a promising foundation for scalable visual representation learning.

## REFERENCES

Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information bottleneck. In ICLR, 2017. URL https://openreview.net/forum?id=HyxQzBceg.

Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, and Nicolas Ballas. Masked siamese networks for label-efficient learning. In ECCV, pp. 456–473, 2022. doi: 10.1007/978-3-031-19821-2 26. URL https: //link.springer.com/chapter/10.1007/978-3-031-19821-2\_26.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, pp. 15619–15629, 2023. doi: 10.1109/CVPR52729.2023.01499. URL https://openaccess.thecvf.com/content/CVPR2023/html/Assran\_ Self-Supervised\_Learning\_From\_Images\_With\_a\_Joint-Embedding\_ Predictive\_Architecture\_CVPR\_2023\_paper.html.

Philip Bachman, R. Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In NeurIPS, 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/hash/ ddf354219aac374f1d40b7e760ee5bb7-Abstract.html.

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML, pp. 1298– 1312, 2022. URL https://proceedings.mlr.press/v162/baevski22a.html.

Alexei Baevski, Arun Babu, Wei-Ning Hsu, and Michael Auli. Efficient self-supervised learning with contextualized target representations for vision, speech and language. In ICML, pp. 1416–1429, 2023. URL https://proceedings.mlr.press/v202/baevski23a.html.

Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics, 2025. URL https://arxiv.org/abs/2511.08544.

David Barber and Felix V. Agakov. Information maximization in noisy channels: A variational approach. In NeurIPS, 2003. URL https://proceedings.neurips.cc/paper\_files/ paper/2003/hash/a6ea8471c120fe8cc35a2954c9b9c595-Abstract.html.

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR, 2022a. URL https://openreview.net/forum? id=xm6YD62D1Ub.

Adrien Bardes, Jean Ponce, and Yann LeCun. VICRegL: Self-supervised learning of local visual features. In NeurIPS, pp. 8799–8810, 2022b. doi: 10.52202/068431-0640. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html.

John S. Bridle, Anthony J. R. Heading, and David J. C. MacKay. Unsupervised classifiers, mutual information and ‘phantom targets’. In NeurIPS, pp. 1096–1101, 1991. URL https://proceedings.neurips.cc/paper\_files/paper/1991/ hash/a8abb4bb284b5b27aa7cb790dc20f80b-Abstract.html.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/ 2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski,´ and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pp. 9650–9660, 2021. doi: 10.1109/ICCV48922.2021.00951. URL https://openaccess. thecvf.com/content/ICCV2021/html/Caron\_Emerging\_Properties\_in\_ Self-Supervised\_Vision\_Transformers\_ICCV\_2021\_paper.html.

Mathilde Caron, Neil Houlsby, and Cordelia Schmid. Location-aware self-supervised transformers for semantic segmentation. In WACV, pp. 116–126, 2024. doi: 10.1109/WACV57701.2024.00019. URL https://openaccess.thecvf.com/content/WACV2024/html/Caron\_ Location-Aware\_Self-Supervised\_Transformers\_for\_Semantic\_ Segmentation\_WACV\_2024\_paper.html.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, pp. 1597–1607, 2020. URL https: //proceedings.mlr.press/v119/chen20j.html.

Bowen Cheng, Alex Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. NeurIPS, 34:17864–17875, 2021.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR, pp. 3213–3223, 2016. doi: 10.1109/CVPR. 2016.350. URL https://openaccess.thecvf.com/content\_cvpr\_2016/html/ Cordts\_The\_Cityscapes\_Dataset\_CVPR\_2016\_paper.html.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. In ICLR, 2024. URL https://openreview.net/forum?id=2dnO3LLiJ1.

Timothee Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, and Piotr Bojanowski.´ Cluster and predict latent patches for improved masked image modeling. TMLR, 2025. URL https://openreview.net/forum?id=Ycmz7qJxUQ.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848. URL https://doi.org/10.1109/CVPR.2009.5206848.

Xueqing Deng, Qihang Yu, Peng Wang, Xiaohui Shen, and Liang-Chieh Chen. COCONut: Modernizing COCO segmentation. In CVPR, pp. 21863–21873, 2024. doi: 10.1109/CVPR52733. 2024.02065. URL https://openaccess.thecvf.com/content/CVPR2024/html/ Deng\_COCONut\_Modernizing\_COCO\_Segmentation\_CVPR\_2024\_paper.html.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. NeurIPS, 27, 2014.

Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes (VOC) challenge. IJCV, 88(2):303–338, 2010. doi: 10. 1007/s11263-009-0275-4. URL https://link.springer.com/article/10.1007/ s11263-009-0275-4.

David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, and Saining Xie. Scaling language-free visual representation learning. In ICCV, 2025. URL https://arxiv.org/abs/2504.01017.

Marco Federici, Anjan Dutta, Patrick Forre, Nate Kushman, and Zeynep Akata. Learning ro-´ bust representations via multi-view information bottleneck. In ICLR, 2020. URL https: //openreview.net/forum?id=B1xwcyHFDr.

Zelin Fu, Bin Tan, Changjiang Sun, Shaohui Liu, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Vision pretraining for dense spatial perception, 2026. URL https:// arxiv.org/abs/2607.05247.

Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In CVPR, pp. 3354–3361, 2012. doi: 10.1109/CVPR.2012.6248074. URL https://doi.org/10.1109/CVPR.2012.6248074.

Ryan Gomes, Andreas Krause, and Pietro Perona. Discriminative clustering by regularized information maximization. In NeurIPS, pp. 775–783, 2010. URL https://proceedings.neurips.cc/paper\_files/paper/2010/hash/ 42998cf32d552343bc8e460416382dca-Abstract.html.

Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre H. Richemond,´ Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko.´ Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pp. 9729–9738, 2020. doi: 10.1109/CVPR42600.2020.00975. URL https://openaccess.thecvf.com/ content\_CVPR\_2020/html/He\_Momentum\_Contrast\_for\_Unsupervised\_ Visual\_Representation\_Learning\_CVPR\_2020\_paper.html.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick.´ Masked autoencoders are scalable vision learners. In CVPR, pp. 16000–16009, 2022. doi: 10.1109/CVPR52688.2022.01553. URL https://openaccess.thecvf. com/content/CVPR2022/html/He\_Masked\_Autoencoders\_Are\_Scalable\_ Vision\_Learners\_CVPR\_2022\_paper.html.

Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In ICML, pp. 4182–4192. PMLR, 2020. URL https://proceedings.mlr.press/v119/ henaff20a.html.

R. Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In ICLR, 2019. URL https://openreview.net/forum?id= Bklr3j0cKX.

Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto, and Masashi Sugiyama. Learning discrete representations via information maximizing self-augmented training. In ICML, pp. 1558– 1567, 2017. URL https://proceedings.mlr.press/v70/hu17b.html.

Xu Ji, Joao F. Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image˜ classification and segmentation. In ICCV, pp. 9864–9873, 2019. doi: 10.1109/ICCV.2019.00996. URL https://openaccess.thecvf.com/content\_ICCV\_2019/html/ Ji\_Invariant\_Information\_Clustering\_for\_Unsupervised\_Image\_ Classification\_and\_Segmentation\_ICCV\_2019\_paper.html.

Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Dollar. Panoptic segmen-´ tation. In CVPR, pp. 9396–9405. IEEE, 2019.

Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, and Zhuowen Tu. Deeply-supervised nets. In AISTATS, pp. 562–570, 2015. URL https://proceedings.mlr.press/v38/ lee15a.html.

Yihao Li, Saeed Salehi, Lyle Ungar, and Konrad P. Kording. Does object binding naturally emerge in large pretrained vision transformers? In NeurIPS, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 050b8ff31bee2dfea65b731e71baccd5-Abstract-Conference.html.

Zhong-Yu Li, Shanghua Gao, and Ming-Ming Cheng. SERE: Exploring feature self-relation for self-supervised transformer. IEEE TPAMI, 45(12):15619–15631, 2023. doi: 10.1109/TPAMI.2023. 3309979. URL https://ieeexplore.ieee.org/document/10234504.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In´ ECCV, pp. 740–755, 2014. doi: 10.1007/978-3-319-10602-1 48. URL https://link.springer. com/chapter/10.1007/978-3-319-10602-1\_48.

Ralph Linsker. Self-organization in a perceptual network. IEEE Computer, 21(3):105–117, 1988. doi: 10.1109/2.36. URL https://ieeexplore.ieee.org/document/36.

Scott C. Lowe, Anthony Fuller, Sageev Oore, Evan Shelhamer, and Graham W. Taylor. Selfdistillation of hidden layers for self-supervised representation learning, 2026. URL https: //arxiv.org/abs/2603.15553.

Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. In ECCV, 2026. URL https://arxiv.org/abs/2603.14482.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J´ egou, Julien Mairal, Patrick Labatut, Ar-´ mand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. TMLR, 2024. URL https://openreview.net/forum?id=GLm1BA3C8p.

Pedro O. Pinheiro, Amjad Almahairi, Ryan Y. Benmalek, Florian Golemo, and Aaron Courville. Unsupervised learning of dense visual representations. In NeurIPS, pp. 4489–4500, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ hash/3000311ca56a1cb93397bc676c0b7fff-Abstract.html.

Sucheng Ren, Fangyun Wei, Samuel Albanie, Zheng Zhang, and Han Hu. DeepMIM: Deep supervision for masked image modeling. In WACV, pp. 879–888, 2025. doi: 10.1109/WACV61041.2025.00095. URL https://openaccess.thecvf.com/ content/WACV2025/html/Ren\_DeepMIM\_Deep\_Supervision\_for\_Masked\_ Image\_Modeling\_WACV\_2025\_paper.html.

Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. FitNets: Hints for thin deep nets. In ICLR, 2015. URL https://arxiv.org/ abs/1412.6550.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet large scale visual recognition challenge. IJCV, 115(3):211–252, 2015. doi: 10. 1007/s11263-015-0816-y. URL https://link.springer.com/article/10.1007/ s11263-015-0816-y.

Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, and Herve J ´ egou. Spreading vectors for´ similarity search. In ICLR, 2019.

Ravid Shwartz-Ziv and Yann LeCun. To compress or not to compress—self-supervised learning and information theory: A review. Entropy, 26(3):252, 2024. doi: 10.3390/e26030252. URL https://www.mdpi.com/1099-4300/26/3/252.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGB-D images. In ECCV, pp. 746–760, 2012. doi: 10.1007/978-3-642-33715-4 54. URL https://link.springer.com/chapter/10.1007/978-3-642-33715-4\_ 54.

Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,´ Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel¨ Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick Labatut, and Piotr Bojanowski. DINOv3.´ TMLR, 2026. URL https: //openreview.net/forum?id=2NlGyqNjns.

Karthik Sridharan and Sham M. Kakade. An information theoretic framework for multi-view learning. In Conference on Learning Theory, pp. 403–414. Omnipress, 2008. URL https: //dblp.org/rec/conf/colt/SridharanK08.

Thomas Stegmuller, Tim Lebailly, Behzad Bozorgtabar, Tinne Tuytelaars, and Jean-Philippe¨ Thiran. CrOC: Cross-view online clustering for dense visual representation learning. In CVPR, pp. 7000–7009, 2023. doi: 10.1109/CVPR52729.2023.00676. URL https: //openaccess.thecvf.com/content/CVPR2023/html/Stegmuller\_CrOC\_ Cross-View\_Online\_Clustering\_for\_Dense\_Visual\_Representation\_ Learning\_CVPR\_2023\_paper.html.

Chenxin Tao, Xizhou Zhu, Weijie Su, Gao Huang, Bin Li, Jie Zhou, Yu Qiao, Xiaogang Wang, and Jifeng Dai. Siamese image modeling for self-supervised vision representation learning. In CVPR, pp. 2132–2141, 2023. doi: 10.1109/CVPR52729.2023.00212. URL https://openaccess.thecvf.com/content/CVPR2023/html/Tao\_Siamese\_ Image\_Modeling\_for\_Self-Supervised\_Vision\_Representation\_ Learning\_CVPR\_2023\_paper.html.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, pp. 776–794, 2020a. doi: 10.1007/978-3-030-58621-8 45. URL https://link.springer. com/chapter/10.1007/978-3-030-58621-8\_45.

Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? In NeurIPS, 2020b. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ hash/4c2e5eaae9152079b9e95845750bb9ab-Abstract.html.

Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In Allerton Conf. on Communication, Control, and Computing, pp. 368–377, 1999. URL https: //arxiv.org/abs/physics/0004057.

Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Self-supervised learning from a multi-view perspective. In ICLR, 2021. URL https://openreview.net/ forum?id=-bdp\_8Itjwp.

Michael Tschannen, Josip Djolonga, Paul K. Rubenstein, Sylvain Gelly, and Mario Lucic. On mutual information maximization for representation learning. In ICLR, 2020. URL https: //openreview.net/forum?id=rkxoh24FPH.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2018. URL https://arxiv.org/abs/1807.03748.

Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. SCAN: Learning to classify images without labels. In ECCV, pp. 268–285, 2020. doi: 10.1007/978-3-030-58607-2 16. URL https://link.springer.com/chapter/ 10.1007/978-3-030-58607-2\_16.

Shashanka Venkataramanan, Valentinos Pariza, Mohammadreza Salehi, Lukas Knobel, Elias Ramzi, Spyros Gidaris, Andrei Bursuc, and Yuki M. Asano. Franca: Nested matryoshka clustering for scalable visual representation learning. In CVPR, pp. 10533–10544, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/ Venkataramanan\_Franca\_Nested\_Matryoshka\_Clustering\_for\_Scalable\_ Visual\_Representation\_Learning\_CVPR\_2026\_paper.html.

Feng Wang, Tao Kong, Rufeng Zhang, Huaping Liu, and Hang Li. Self-supervised learning by estimating twin class distributions. IEEE Transactions on Image Processing, 2023a. URL https://arxiv.org/abs/2110.07402.

Haoqing Wang, Yehui Tang, Yunhe Wang, Jianyuan Guo, Zhi-Hong Deng, and Kai Han. Masked image modeling with local multi-scale reconstruction. In CVPR, pp. 2122–2131, 2023b. doi: 10.1109/CVPR52729.2023.00211. URL https://openaccess.thecvf.com/content/ CVPR2023/html/Wang\_Masked\_Image\_Modeling\_With\_Local\_Multi-Scale\_ Reconstruction\_CVPR\_2023\_paper.html.

Xin Wen, Bingchen Zhao, Anlin Zheng, Xiangyu Zhang, and Xiaojuan Qi. Self-supervised visual representation learning with semantic grouping. In NeurIPS, pp. 16423–16438, 2022. doi: 10.52202/

068431-1195. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference. html.

Ziyang Wu, Jingyuan Zhang, Druv Pai, Xudong Wang, Chandan Singh, Jianwei Yang, Jianfeng Gao, and Yi Ma. Simplifying DINO via coding rate regularization. In ICML, pp. 68036–68059, 2025. URL https://proceedings.mlr.press/v267/wu25ar.html.

Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In CVPR, pp. 16684–16693, 2021. doi: 10.1109/CVPR46437.2021.01641. URL https://openaccess.thecvf.com/content/CVPR2021/html/Xie\_ Propagate\_Yourself\_Exploring\_Pixel-Level\_Consistency\_for\_ Unsupervised\_Visual\_Representation\_Learning\_CVPR\_2021\_paper.html.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny. Barlow twins: Self-´ supervised learning via redundancy reduction. In ICML, pp. 12310–12320, 2021. URL https://proceedings.mlr.press/v139/zbontar21a.html.

Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ADE20K dataset. IJCV, 127(3):302–321, 2019. doi: 10.1007/s11263-018-1140-0. URL https://link.springer.com/article/10. 1007/s11263-018-1140-0.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT pre-training with online tokenizer. In ICLR, 2022. URL https://openreview. net/forum?id=ydopy-e6Dg.

Adrian Ziegler and Yuki M. Asano. Self-supervised learning of object parts for semantic segmentation. In CVPR, pp. 14502–14511, 2022. doi: 10.1109/CVPR52688.2022.01410. URL https://openaccess.thecvf.com/content/CVPR2022/html/ Ziegler\_Self-Supervised\_Learning\_of\_Object\_Parts\_for\_Semantic\_ Segmentation\_CVPR\_2022\_paper.html.

## A APPENDIX

In this appendix, we provide additional analyses, implementation details, and experiments for JEM. Specifically:

• Additional discussion. In Sec. A.1, we discuss how existing self-supervised methods can be interpreted from the perspective of the lower bound of the multi-view InfoMax objective, and relate JEM to more prior work.

• Additional results. In Sec. A.2, we report scaling trends on individual benchmarks.

• Additional details. In Sec. A.3, we describe the structural collapse regularizer, view construction, training procedure, and training data.

• Derivations. In Sec. A.4, we provide the full derivations of our objective.

• Evaluation details. In Sec. A.5, we describe the evaluation protocol of each benchmark, including the implementation and reproduction of competing methods, list the evaluated model checkpoints, and detail the estimated number of training tokens for baselines.

• Training configurations. In Sec. A.6, we collect the complete architecture, training hyperparameter, and ablation settings tables.

• Qualitative examples. In Sec. A.7, we report additional qualitative feature visualizations of JEM and competing models.

## A.1 RELATIONSHIP TO EXISTING SSL PARADIGMS

In this section, we discuss how existing self-supervised methods can be interpreted from the multiview InfoMax perspective. Further, we relate JEM to prior works in which some of its key components also appear, namely: explicit entropy regularization of discrete targets, dense losses between partially overlapping views, multi-layer targets, and structural regularization, and discuss how our formulation differs.

How different methods fit the multi-view InfoMax objective Self-clustering methods such as SwAV (Caron et al., 2020), DINO (Caron et al., 2021), or iBOT (Zhou et al., 2022) use discrete categorical targets T, minimizing a cross entropy loss between student and teacher views that matches the KL divergence in Eq. (2) up to a constant. For the anti-collapse term, centering or Sinkhorn-Knopp balancing of the targets plays the role of maintaining H(T); temperature sharpening makes sure that $H ( \check { T } \mid X _ { T } )$ does not grow too large. In SimDINO (Wu et al., 2025), the targets are continuous vectors, with the minimized mean-squared-error (MSE) loss again matching the KL divergence; $H ( T )$ is maintained by maximizing the log-determinant of the covariance matrix of the head outputs, and ${ \dot { H } } ( T \mid X _ { T } = { \dot { x } } )$ is minimal for all x. Similarly, redundancy-reduction methods (Zbontar et al., 2021; Bardes et al., 2022a; Balestriero & LeCun, 2025) combine cross-view alignment with output-distribution regularization that can be seen as an indirect proxy for maintaining H(T).

A principled view of the DINO family of methods. Owing to its strong performance, the DINO family of self-distillation methods (Caron et al., 2021; Zhou et al., 2022; Oquab et al., 2024; Simeoni´ et al., 2026) has become a common foundation for many recent methods in the literature (Wu et al., 2025; Fan et al., 2025; Venkataramanan et al., 2026). As discussed above, these methods can be interpreted as instantiations of the multi-view InfoMax objective, in which the cross-entropy between student and teacher implements the alignment term, while the anti-collapse term is only maintained implicitly, through mechanisms that were introduced empirically (Caron et al., 2020; 2021). In JEM, each component of the training objective instead follows from the lower bound in Eq. (2), which simplifies the recipe of DINOv2 in three ways. (i) A single loss and single headfor thefinal layer. DINOv2 applies the DINO loss to the class token and the iBOT loss to the masked patches, with separate heads, and adds the KoLeo regularizer (Sablayrolles et al., 2019), a nearest-neighbor estimator of the differential entropy of the class-token features, which acts as a further proxy for H(T). JEM instead applies the same alignment and anti-collapse terms to all elements of T, through one head at each supervised depth. (ii) Explicit regularization instead of heuristics. Sinkhorn-Knopp balancing/centering and temperature sharpening indirectly control the marginal entropy $H ( T )$ and the conditional entropy $H ( T \mid \dot { X } _ { T } )$ , respectively. JEM replaces them with $\mathcal { R } _ { \mathrm { g l o b a l } } .$ , which optimizes both quantities explicitly, so that student and teacher can share the same constant temperature, without a teacher temperature schedule. (iii) No prototype freezing. DINO and DINOv2 freeze the last layer of the head, which holds the prototypes, during the first iterations of training to stabilize it; JEM requires no such mechanism. Moreover, the class token fits naturally into our information-theoretic derivation as one more element of $T ,$ , to which the same alignment and anti-collapse terms apply. From this perspective, DINO corresponds to the special case in which the class token is the only element of $T .$ , and iBOT and DINOv2 to adding the masked patches as further elements, each with its own loss and head.

Explicit entropy regularization of discrete assignments. For discrete targets, the InfoMax anticollapse term $\tilde { I ( X _ { T } ; T ) } = H ( T ) - H ( T \mid X _ { T } )$ corresponds to maximizing the entropy of the average prediction while minimizing the entropy of each individual prediction. Early works (Bridle et al., 1991; Gomes et al., 2010) employ explicit mutual information maximization for training unsupervised classifiers. More recently, unsupervised clustering methods have employed similar regularization techniques. SCAN (Van Gansbeke et al., 2020) maximizes the entropy of the mean prediction to avoid collapse of the cluster assignments, without an explicit term for the sharpness of individual predictions, which is only encouraged implicitly by its consistency loss between neighboring samples. IIC (Ji et al., 2019) instead directly maximizes the mutual information between the cluster assignments of two views, without explicit regularization for each input’s assignment. IMSAT (Hu et al., 2017) includes both marginal and conditional entropy terms, combined with an alignment loss to enforce augmentation invariance. Among self-clustering methods, MSN (Assran et al., 2022) optimizes the marginal entropy, but also applies Sinkhorn-Knopp balancing to the targets and resorts to temperature sharpening to control the conditional entropy.

More similar to ours, TWIST (Wang et al., 2023a) employs an alignment loss between teacher and student assignment distributions, alongside the entropy regularization terms. It is the closest imagelevel analogue of our $\mathcal { R } _ { \mathrm { g l o b a l } }$ combined with $\mathcal { L } _ { \mathrm { a l i g n } }$ . In contrast to these works, our $\mathcal { R } _ { \mathrm { g l o b a l } }$ is applied to the discrete assignments of every patch in a patch-level student-teacher framework. Moving from per-image to per-patch assignment also introduces two collapse modes that entropy terms pooled over all patches cannot control: structural collapse and positional collapse. The latter occurs when the model simply encodes the position of each patch in the grid. The marginal entropy over all patches can be increased simply by encoding the patch position. Instead, computing the marginal entropy separately for each patch position leads to estimating $I ( X _ { T } ; T $ Pos), which prevents this failure mode. In our derivation, this formulation naturally follows the decomposition in Eq. (3). Among patch-level methods, CAPI (Darcet et al., 2025) identifies the same positional collapse and alleviates it implicitly, by running the Sinkhorn-Knopp balancing of the targets separately at each patch position. On the other hand, the structural collapse failure mode entails patches within an image falling to an input-dependent constant distribution, effectively encoding the same image-level content. This motivates our structural regularizer $\mathcal { R } _ { \mathrm { s t r u c t } }$ (Sec. 3.2.3). Both of these terms have no counterpart in the image-level methods mentioned above.

Dense losses between partially overlapping views. A large family of dense SSL methods is based on masked image modeling, which predicts the content of masked patches, either in pixel (He et al., 2022), or latent space (Baevski et al., 2022; Assran et al., 2023). In pure masking methods, input and target fully overlap, whereas in our formulation we treat masking and cropping as complementary strategies to control information sharing between views. Therefore, we utilize partially overlapping views for teacher and student, additionally masking the student views. Another category of methods applies joint-embedding losses between corresponding locations of two partially overlapping crops, but without masking. Among these, PixPro (Xie et al., 2021) and VADeR (Pinheiro et al., 2020) are pixel-level contrastive methods. Several patch-level clustering-based methods also rely on partially overlapping crops, aligning the patch assignments of the overlapping region with RoIAlign (Ziegler & Asano, 2022; Wen et al., 2022) or clustering the patches of both views jointly (Stegmuller et al.¨ , 2023). VICRegL (Bardes et al., 2022b) applies a similar idea to continuous embeddings, matching each feature to its spatial nearest neighbor in the other view. A few methods combine the two families, applying masked modeling across partially overlapping views, like ours. LOCA (Caron et al., 2024) trains a student on masked query crops to predict teacher cluster assignments for a larger reference crop, where correspondences are interpolated. SIM (Tao et al., 2023) uses a decoder to predict, from the visible patches of a masked view, the features of all patches of a different view. Differently from all these prior works, we sample student and teacher crops at integer patch offsets on a shared patch lattice, so that overlapping patches correspond exactly one-to-one; we found that interpolating targets between misaligned patch grids degrades the resulting dense features. Moreover, we treat masked and visible patches uniformly, applying the alignment loss to all overlapping patches.

Multiple layers as targets. Supervising intermediate layers dates back to deeply-supervised networks (Lee et al., 2015), and to intermediate layer distillation (Romero et al., 2015). In SSL, data2vec (Baevski et al., 2022; 2023) utilizes projections to regress the top-K layers of an EMA teacher from the final layer of the student. Among masking models, DeepMIM (Ren et al., 2025) and LocalMIM (Wang et al., 2023b) add masked prediction losses on intermediate encoder layers, using as targets either pixels, image gradients, or teacher features. More recently, V-JEPA 2.1 (Mur-Labadia et al., 2026) fuses together the features of several student layers, for a shared predictor to regress the EMA-teacher features at the different layers, for both masked and visible tokens. Similarly, Bootleg (Lowe et al., 2026) extends I-JEPA by regressing, from the final student layer through a single predictor, the concatenated EMA-teacher features at the masked locations from different layers. In JEM, targets are indexed by patch and layer: each supervised student layer k predicts the categorical distribution computed by the EMA teacher at the same depth $k ,$ so that the prediction at depth k only depends on the first k layers, and the alignment and anti-collapse terms are applied at every depth. In addition, $\mathcal { R } _ { \mathrm { s t r u c t } }$ uses intermediate teacher layers as targets for thefinal student layer, but only through their similarity structure (see next paragraph).

Structural regularization and Gram anchoring. Our structural regularizer $\mathcal { R } _ { \mathrm { s t r u c t } }$ (Sec. 3.2.3) addresses the redundancy term TC(T) of the anti-collapse objective, whose uncontrolled growth leads to structural collapse, in which all patches of an image encode the same image-level content. We implement this regularizer by matching the pairwise cosine similarities between patches of a set of early-to-middle teacher layers from a projection of the features from the final student layer. Using pairwise similarities as a target has also been employed by other methods. In SSL, SERE (Li et al., 2023) extends iBOT with a patch self-relation loss, training the similarities between the final-layer patch features of the student to match those of the EMA teacher on another view. More recently, DINOv3 (Simeoni et al.´ , 2026) adopts gram anchoring, to address the different phenomenon of the local patch structure becoming increasingly noisy with long training. In DINOv3, after 1M iterations, an earlier checkpoint is used as a gram teacher, to anchor the similarity matrix of the final-layer patch features of the student. Differently, our targets come from early-to-middle teacher layers and are only matched after a learned projection, so the final representation is not forced to share the geometry of the earlier layers, but must retain the information required to recover it. Moreover, our $\mathcal { R } _ { \mathrm { s t r u c t } }$ is applied from the start of training, uniformly on visible and masked patches, as well as local crops.

## A.2 ADDITIONAL RESULTS

Fig. 7 reports the individual benchmark results underlying the task averages in Fig. 6, together with ImageNet-1k classification accuracy.

## A.3 IMPLEMENTATION DETAILS

In this section, we give additional details about the structural collapse regularizer (Sec. A.3.1), construction of the views (Sec. A.3.2), hyperparameters and training details (Sec. A.3.3), and the used training data (Sec. A.3.4).

## A.3.1 STRUCTURAL COLLAPSE REGULARIZER

We detail the structural collapse regularizer introduced in Sec. 3.2.3. For student and teacher views $x _ { S } , x _ { T }$ , we define ${ \mathcal { O } } _ { S }$ and $\mathcal { O } _ { T }$ , the sets of patch indices of student/teacher that geometrically overlap with each other, and $m \colon { \mathcal { O } } _ { S } \mapsto { \mathcal { O } } _ { T }$ , the function mapping the student indices to the spatially corresponding teacher patch. Given student representation r and teacher representation $h _ { k }$ at layer $k ,$ the loss for a pair of elements at indices $i , j$ can be written as

$$
\mathcal { L } _ { \mathrm { s t u c t } } ^ { ( i , j ) } ( \boldsymbol { r } , h _ { k } ) = \left( \cos ( g _ { k } ( \boldsymbol { r } ^ { ( i ) } ) , g _ { k } ( \boldsymbol { r } ^ { ( j ) } ) ) - \cos ( h ^ { ( m ( i ) ) } , h ^ { ( m ( j ) ) } ) \right) ^ { 2 } , \quad ( i , j ) \in \mathcal { O } _ { S } \times \mathcal { O } _ { S } ,\tag{9}
$$

where $g _ { k }$ is a learned non-linear projection for the kth teacher layer. Thus, to compute the loss, only geometrically corresponding pairs of patches are used. Given a set of target teacher layers $\kappa .$ , the

![](images/4c3b646cfcf8c67ef41c33dc34d1fb25466d83358448acfaad2b1c472effdc35.jpg)  
Figure 7: Scaling across individual benchmarks. Performance as a function of approximate training compute. DINOv2\* and JEM are trained by us on A140M for 500k steps. The purple dashed lines show DINOv3 ViT-7B. ADE20k is evaluated with both semantic segmentation (mIoU) and panoptic segmentation (PQ) probes.

total loss is then averaged over target layers and patch pairs:

$$
\mathcal { R } _ { \mathrm { s t r u c t } } ( x _ { S } , x _ { T } ) = \frac { 1 } { | K | | \mathcal { O } _ { S } | ( | \mathcal { O } _ { S } | - 1 ) } \sum _ { k \in \mathcal { K } } \sum _ { i \in \mathcal { O } _ { S } } \sum _ { \stackrel { j \in \mathcal { O } _ { S } } { i \neq j } } \mathcal { L } _ { \mathrm { s t r u c t } } ^ { ( i , j ) } ( r , h _ { k } ) .\tag{10}
$$

We use target layers [0, 8, 16] for ViT-L (24 total layers), and [0, 10, 20, 30] for ViT-g and ViT-7B (40 total layers), where layer 0 refers to the output of the patch embedding. The target features $h _ { k }$ are processed by an element-wise layernorm without affine parameters before computing the cosine similarity. We implement $g _ { k }$ with a MLP of the same size as the projection head, except that it maps to an output space of dimensionality 1024 and does not have the final linear projection to logits. The regularizer is applied both on global and local crops, and masked and unmasked patches.

Note that it is also possible to apply the regularizer only using student representations, and we successfully experimented with this option. This simplifies the implementation, as patches at the same index do directly geometrically correspond to each other. We opted for the slightly more complex teacher version as it allowed us to also apply the loss on the masked patches (which have no good similarity targets on the student), the EMA provides more stable targets, and the fact that the targets come from a different view may have an additional positive effect through enforcing a view invariance on the learned similarity structure.

## A.3.2 VIEW CONSTRUCTION

Given an image x, we sample student view $x _ { S }$ and teacher view $x _ { T }$ as follows. We first extract a super crop with pre-determined scale and random aspect ratio from the image x and resize it to a square set of pixels with the side length being a multiple of the patch size. The scale is randomly chosen such that an overlap of at least 10% is achieved between the student and teacher crops when randomly picking the scale of the teacher crop from [0.32, 1.0] of the area of the original image. Then, at random patch-grid aligned positions, we extract square student and teacher crops with a size matching the input resolution of the model. A random horizontal flip is applied to the whole image before cropping with probability 0.5, such that all views share the same orientation. Each crop is then independently transformed with standard photometric augmentations (Grill et al., 2020; Caron et al., 2021), i.e. color jittering, grayscale conversion, Gaussian blur, and solarization. Different from DINO (Caron et al., 2021), where the teacher processes the same augmented global crops as the student, teacher crops are augmented independently of their student crops.

For 80% of the student crops, we additionally sample a patch mask M that is applied to the student view, for which we use multi-block masking (Zhou et al., 2022; Oquab et al., 2024) with cyclic shifts (Darcet et al., 2025; Venkataramanan et al., 2026) for the global crops, and uniform random masking for the local crops. For global crops, we mask 65% of the patches. For local crops, we mask 50% of the patches.

## A.3.3 TRAINING DETAILS

We base the implementations and hyperparameters of both JEM and the DINOv2 baseline on the open source implementation of DINOv3 Simeoni et al.´ (2026). Our self-trained DINOv2 baseline can thus be seen as a version of DINOv3 without gram anchoring and high-resolution refinement stages.

The complete training configurations are collected in Sec. A.6. We give the backbone architectures and training hyperparameters shared by JEM and the DINOv2 baselines in Tab. 5, and their remaining training hyperparameters in Tabs. 6 and 8, respectively. We note that the listed effective learning rate $\eta _ { \mathrm { e f f } }$ is the result of square root scaling of a base learning rate $\eta _ { \mathrm { b a s e } }$ with respect to a local batch size of 64: $\begin{array} { r } { \eta _ { \mathrm { e f f } } = \sqrt { \frac { \mathrm { g l o b a l . b s } } { 6 4 } \cdot \eta _ { \mathrm { b a s e } } } . } \end{array}$

We summarize further experiment-specific settings in Tabs. 6 and 7. In particular, the JEM and DINOv2\* ViT-L models listed in Tab. 2 are trained for 125k steps; the JEM models shown in the ablation study are trained for 250k. For the “no EMA” ablation, we set the teacher momentum to 0, effectively updating the teacher parameters with the current student parameters after each update step. We additionally change the configuration of the global collapse regularizer, increasing its overall weight relative to the alignment and structural collapse regularizers as well as increasing the weight of the marginal entropy component while decreasing the weight of the conditional entropy component. We found this necessary to move the model out of the collapsed regime at the start of training. Thus, removing the EMA teacher requires a slightly different hyperparameter regime due to the EMA’s effect on the optimization dynamics, as can be expected.

The loss of JEM is a weighted combination of (i) alignment loss (Eq. (4)) on visible $( \mathcal { L } _ { \mathrm { a l i g n } } ^ { \mathrm { v i s } } )$ and masked $( \mathcal { L } _ { \mathrm { a l i g n } } ^ { \mathrm { m s k } } )$ patches, (ii) global collapse regularization loss $\mathcal { R } _ { \mathrm { g l o b a l } } \left( \mathrm { E q . } \left( 6 \right) \right)$ ) and (iii) structural regularization loss $\mathcal { R } _ { \mathrm { s t r u c t } } \left( \mathrm { E q . } \left( 8 \right) \right)$ :

$$
\mathcal { L } = w _ { \mathrm { v i s } } \cdot \mathcal { L } _ { \mathrm { a l i g n } } ^ { \mathrm { v i s } } + w _ { \mathrm { m s k } } \cdot \mathcal { L } _ { \mathrm { a l i g n } } ^ { \mathrm { m s k } } + w _ { \mathrm { g l o b a l } } \cdot \mathcal { R } _ { \mathrm { g l o b a l } } + w _ { \mathrm { s t r u c t } } \cdot \mathcal { R } _ { \mathrm { s t r u c t } } .\tag{11}
$$

In practice, we use different loss coefficients for targets from different layers (see Tab. 6), and apply the structural regularizer only to the final layer representation. All losses are applied to global and local crops. Global and local crops are weighted according to their natural ratio by default.

## A.3.4 DATA

Following the protocol established by Oquab et al. (2024), we construct A140M by retrieving from a large pool images that align with the train splits of established public datasets, including

IN-1k, IN-22k, ADE20K, etc. Our source pool comprises approximately 750M images, obtained by deduplicating an internal collection of 1.3 billion images. This process involves both selfdeduplication which remove duplicates to improve retrieval quality, and relative-deduplication which prevents data contamination from the “test” splits of the seed datasets, which are used as standard benchmarks in evaluation. Consistent with Oquab et al. (2024), we also implement NSFW filtering and facial blurring.

## A.4 DERIVATIONS

![](images/2383adbad06cc3db32ae3a6b0cdb736c285bbe76de28d7dd614fa9a423b9f0b8.jpg)  
Figure 8: Student-teacher graphical models. (a) shows the generic model with the form of $T$ not specified; (b) shows the element-wise model with additional factorization assumptions. The views $( X _ { S } , X _ { T } )$ are sampled jointly given input X. Solid arrows describe the sampling model, with $R = f ( X _ { S } )$ and $T \sim p ( \cdot \mid X _ { T } )$ ; double circles mark deterministic nodes. Dashed arrows denote the variational predictors $q ( T \mid R )$ and $q ( T ^ { ( \ell ) } \mid R ^ { ( \ell ) } )$ . In (b), ℓ indexes matched patch/layer pairs, with teacher targets relabeled accordingly and $R ^ { ( \ell ) } = f ^ { ( \ell ) } ( X _ { S } )$ . The plate represents the factorizations $\begin{array} { r } { p ( T \mid X _ { T } ) = \prod _ { \ell = 1 } ^ { L } p ( T ^ { ( \ell ) } \mid X _ { T } ) \mathrm { ~ a n d ~ } q ( T \mid R ) = \prod _ { \ell = 1 } ^ { L } q ( T ^ { ( \ell ) } \mid R ^ { ( \ell ) } ) } \end{array}$

We assume $X _ { S }$ and $X _ { T }$ are jointly generated from X. For the student, $R = f ( X _ { S } )$ , where $f$ is a deterministic function, and thus $q ( t \mid x _ { S } ) : = q ( t \mid f ( x _ { S } ) )$ . The teacher generates T from $X _ { T }$ resulting in the joint distribution $p ( x _ { S } , x _ { T } , t ) \ = \ p ( x _ { S } , x _ { T } ) p ( t \ | \ x _ { T } )$ . See Fig. 8 for graphical models.

Lower bound on mutual information (Eq. (1)) This is the lower bound from Barber & Agakov (2003). Starting with the definition of mutual information, we have:

$$
I ( R ; T ) = H ( T ) - H ( T \mid R )\tag{12}
$$

$$
= H ( T ) + \mathbb { E } _ { R , T } \log p ( T \mid R )\tag{13}
$$

$$
= H ( T ) + \mathbb { E } _ { R , T } \log q ( T \mid R ) + \mathbb { E } _ { R , T } \log { \frac { p ( T \mid R ) } { q ( T \mid R ) } }\tag{14}
$$

$$
= H ( T ) + \mathbb { E } _ { R , T } \log q ( T \mid R ) + \underbrace { \mathbb { E } _ { R } D _ { \mathrm { K L } } \left( p ( \cdot \mid R ) \parallel q ( \cdot \mid R ) \right) } _ { > 0 }\tag{15}
$$

$$
\geq H ( T ) + \mathbb { E } _ { X _ { S } , T } \log q ( T \mid X _ { S } ) .\tag{16}
$$

KL form of lower bound (Eq. (2)) Expanding the expected KL divergence:

$$
\mathbb { E } _ { X _ { S } , X _ { T } } D _ { \mathrm { K L } } \left( p ( \cdot \mid X _ { T } ) \parallel q ( \cdot \mid X _ { S } ) \right)\tag{17}
$$

$$
= \mathbb { E } _ { X _ { S } , X _ { T } } \mathbb { E } _ { T \sim p ( \cdot \vert X _ { T } ) } \left[ \log p ( T \mid X _ { T } ) - \log q ( T \mid X _ { S } ) \right]\tag{18}
$$

$$
= \mathbb { E } _ { X _ { T } } \mathbb { E } _ { T \sim p ( \cdot \mid X _ { T } ) } \log p ( T \mid X _ { T } ) - \mathbb { E } _ { X _ { S } } \mathbb { E } _ { T \sim p ( \cdot \mid X _ { T } ) } \log q ( T \mid X _ { S } )\tag{19}
$$

$$
= - H ( T \mid X _ { T } ) - \operatorname { \mathbb { E } } _ { X _ { S } , T } \log q ( T \mid X _ { S } ) .\tag{20}
$$

It follows that $\mathbb { E } _ { X _ { S } , T } \log q ( T \mid X _ { S } ) = - D _ { \mathrm { K L } } \left( p ( \cdot \mid X _ { T } ) \parallel q ( \cdot \mid X _ { S } ) \right) - H ( T \mid X _ { T } )$ . Plugging this into Eq. (1) gives

$$
I ( R ; T ) \geq H ( T ) + \mathbb { E } _ { X _ { S } , T } \log q ( T \mid X _ { S } )\tag{21}
$$

$$
= I ( X _ { T } ; T ) - D _ { \mathrm { K L } } \left( p ( \cdot \mid X _ { T } ) \parallel q ( \cdot \mid X _ { S } ) \right) .\tag{22}
$$

Element-wise decomposition of mutual information (Eq. (3)) With element-wise targets $T =$ $( T ^ { ( 1 ) } , \dots , T ^ { ( L ) } )$ , we have that

$$
H ( T ) = \sum _ { \ell = 1 } ^ { L } H ( T ^ { ( \ell ) } ) - \mathrm { T C } ( T ) ,\tag{23}
$$

where $\begin{array} { r } { \mathrm { T C } ( T ) : = D _ { \mathrm { K L } } \left( ( T ^ { ( 1 ) } , \dots , T ^ { ( L ) } ) \parallel \prod _ { \ell = 1 } ^ { L } T ^ { ( \ell ) } \right) } \end{array}$ is the total correlation measuring dependence between the elements of T. The total correlation becomes zero if and only if the variables are mutually independent; it is an extension of mutual information for more than two variables. Then,

$$
\begin{array} { l } { { \displaystyle I ( X _ { T } ; T ) = H ( T ) - H ( T \mid X _ { T } ) } } \\ { { \displaystyle \quad = \sum _ { \ell = 1 } ^ { L } H ( T ^ { ( \ell ) } ) - \mathrm { T C } ( T ) - \left( \sum _ { \ell = 1 } ^ { L } H ( T ^ { ( \ell ) } \mid X _ { T } ) - \mathrm { T C } ( T \mid X _ { T } ) \right) } } \\ { { \displaystyle \quad = \sum _ { \ell = 1 } ^ { L } H ( T ^ { ( \ell ) } ) - \sum _ { \ell = 1 } ^ { L } H ( T ^ { ( \ell ) } \mid X _ { T } ) - ( \mathrm { T C } ( T ) - \mathrm { T C } ( T \mid X _ { T } ) ) } } \\ { { \displaystyle \quad = \sum _ { \ell = 1 } ^ { L } I ( X _ { T } ; T ^ { ( \ell ) } ) - ( \mathrm { T C } ( T ) - \mathrm { T C } ( T \mid X _ { T } ) ) . } } \end{array}\tag{24}
$$

Element-wise decomposition of KL divergence (Sec. 3.2.1) We want to show how the KL alignment term in Eq. (2) decomposes for element-wise targets. Let $T = ( T ^ { ( 1 ) } , \dots , T ^ { ( L ) } )$ , and assume that the teacher and student distributions factorize conditionally over elements:

$$
p ( t \mid x _ { T } ) = \prod _ { \ell = 1 } ^ { L } p ( t ^ { ( \ell ) } \mid x _ { T } ) , \qquad q ( t \mid x _ { S } ) = \prod _ { \ell = 1 } ^ { L } q ( t ^ { ( \ell ) } \mid x _ { S } ) .\tag{25}
$$

For any fixed pair of views $( x _ { S } , x _ { T } )$ , we then have

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { K L } } ( p ( \cdot \mid x _ { T } )  q ( \cdot \mid x _ { S } ) ) = \mathbb { E } _ { T \sim p ( \cdot \mid x _ { T } ) } [ \log \frac { p ( T \mid x _ { T } ) } { q ( T \mid x _ { S } ) } ] } } \\ { ~ } \\ { { \displaystyle ~ = \mathbb { E } _ { T \sim p ( \cdot \mid x _ { T } ) } [ \sum _ { \ell = 1 } ^ { L } \log \frac { p ( T ^ { ( \ell ) } \mid x _ { T } ) } { q ( T ^ { ( \ell ) } \mid x _ { S } ) } ] } } \\ { { \displaystyle ~ = \sum _ { \ell = 1 } ^ { L } \mathbb { E } _ { T ^ { ( \ell ) } \sim p _ { \ell } ( \cdot \mid x _ { T } ) } [ \log \frac { p ( T ^ { ( \ell ) } \mid x _ { T } ) } { q ( T ^ { ( \ell ) } \mid x _ { S } ) } ] } } \\ { { \displaystyle ~ = \sum _ { \ell = 1 } ^ { L } { \cal D } _ { \mathrm { K L } } ( p ( \cdot \mid x _ { T } ) \mid q ( \cdot \mid x _ { S } ) ) . } } \end{array}\tag{26}
$$

The third equality marginalizes out all elements other than $t ^ { ( \ell ) }$ , since the logarithm depends only on that element. Taking the expectation over paired views gives

$$
\mathbb { E } _ { X _ { T } , X _ { S } } D _ { \mathrm { K L } } ( p ( \cdot \mid X _ { T } ) \parallel q ( \cdot \mid X _ { S } ) ) = \sum _ { \ell = 1 } ^ { L } \mathbb { E } _ { X _ { T } , X _ { S } } D _ { \mathrm { K L } } \left( p ( \cdot \mid X _ { T } ) \parallel q ( \cdot \mid X _ { S } ) \right) .\tag{27}
$$

## A.5 EVALUATION DETAILS

## A.5.1 BENCHMARK PROTOCOLS

We evaluate models using standard lightweight probing setups. Following Simeoni et al. ´ (2026), we compare models of different patch sizes at matched input sequence length to equalize the compute of the forward pass. All evaluations use final patch features of the backbone after the final layer norm.

Image Classification For classification, we apply the single-layer attention probing setup from Darcet et al. (2025). A learned query token is used to perform scaled multi-head attention (head dim=64) over key-value transformed features; the result is linearly projected to logits. We perform grid search on a validation set to select the best hyperparameters. Thereby, we scan the full cross product over learning rates from $( 1 \cdot 1 0 ^ { - 5 } , 2 \cdot 1 0 ^ { - 5 } , \hat { 5 } \cdot \hat { 1 } 0 ^ { - 5 } , 1 \cdot 1 0 ^ { - 4 } , 2 \cdot 1 0 ^ { - 4 } , 5 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - 3 } , 2 \cdot 1 0 ^ { - 5 } , 0 \cdot 1 0 ^ { - 6 } )$ $\mathrm { i 0 ^ { - 3 } , 5 \cdot 1 0 ^ { - 3 } , 1 \cdot 1 \bar { 0 } ^ { - 2 } } )$ and weight decays from $( 5 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - 3 } , 5 \cdot 1 0 ^ { - 2 } )$ . The probe is trained with a batch size of 1024 with AdamW for 12500 steps, with linear learning rate warm-up of 1250 steps followed by cosine decay. The applied learning rate is linearly scaled with global batch as $\mathrm { { l r } = \frac { \ g l o b a l . b s } { 2 5 6 } }$ · base lr.

For training on ImageNet, we split the original train set into training and validation sets, using 90% for training, and 10% for validation. We apply light random augmentations during training, specifically RandomResizedCrop and random horizontal flips. During evaluation, we resize the image to 256/292 pixels on the short side and take the central square crop of 224/256 pixels (87.5% zoom crop) for patch size 14/16. Models are evaluated on ImageNet val (Russakovsky et al., 2015). We report top-1 accuracy.

Semantic Segmentation Semantic segmentation is the task of assigning a class-label to every pixel of the input image. We use ADE20k (Zhou et al., 2019), PASCAL VOC (Everingham et al., 2010) and Cityscapes (Cordts et al., 2016) datasets to evaluate this task and use the mean intersection-over-union (mIoU) as the reported metric. We evaluate the task under two protocols with frozen backbones: kNN and linear probing. First, we use the kNN protocol for fast evaluations and assessment of the quality of dense patches. There, images and label maps are resized to $2 5 6 \times 2 5 6$ without cropping, using bicubic and nearest-neighbor interpolation respectively. The mIoU metric is computed on the full resized label maps. Second, for the linear probing, we follow Simeoni et al.´ (2026) and take patch tokens from the last layer after the backbone’s final LayerNorm. On top of those, we train a linear head consisting of a BatchNorm layer followed by a $1 1 \times 1$ convolution that outputs N logits, where N is the number of classes in a given dataset. The logits are bilinearly upsampled to the label resolution and trained with pixel-wise cross-entropy. We scan the full cross product of learning rates $\{ 1 \cdot 1 0 ^ { - 4 } , 3 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - 3 } \}$ and weight decays $\{ 1 \cdot 1 0 ^ { - 3 } , 1 \cdot 1 0 ^ { - 2 } \}$ . All six probes are trained together on the same frozen features. Each probe is trained with AdamW optimizer (using $\beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9 )$ and a batch size of 16 for 40,000 steps. The learning rate warms up linearly over 1,500 steps and then linearly decays to zero. During training, we resize each image so that its short side is $4 4 8 / 5 1 2 \cdot .$ s pixels for models with patch size 14/16, with s drawn uniformly from [0.5, 2.0]. We then take a random $r \times r$ square crop adapted to 1024 tokens $\mathbf { ( i . e . , 4 4 8 ^ { 2 } / 5 1 2 ^ { 2 } }$ crop for patch size 14/16) and apply random horizontal flips and photometric distortions (consisting of brightness, contrast, saturation and hue). During evaluation, we resize the image to 448/512 pixels on the short side and run $r \times r$ sliding-window inference with windows and a stride of 298/341 pixels. Logits are averaged where windows overlap and bilinearly upsampled to the original image resolution. We report the mIoU of the best probe over all hyperparameter configurations and evaluation steps.

Instance and Panoptic Segmentation Instance segmentation is the task of recognizing all pixels that belong to different object instances and their class-label. Panoptic segmentation (Kirillov et al., 2019) extends instance segmentation beyond objects: every pixel is assigned a class label and, for countable objects (things), an instance identity, while regions such as sky or road (stuff) are only labeled by class.

We evaluate instance segmentation on the COCO dataset (Lin et al., 2014), and report the standard COCO average precision metric (mAP). We evaluate panoptic segmentation on COCONut (Deng et al., 2024) and ADE20K-Panoptic (Zhou et al., 2019). For COCONut, we train on COCONut-S and evaluate on the relabeled COCO-val set with panoptic annotations; for ADE20K-Panoptic, we follow prior work (Cheng et al., 2021) and use the standard ADE20K training and validation splits. We report panoptic quality (PQ) (Kirillov et al., 2019), computed from IoU-based matching between predicted and ground-truth segments.

We use the same lightweight probe for both tasks, which operates on the last-layer patch features of the frozen backbone, taken after the final layer normalization. The probe is split into two independent modules: a patch-wise linear classifier that predicts the semantic category of each patch, as in semantic segmentation, and a patch-wise 2-layer MLP that is used to split the patches into different segments (instances). To get the masks, we compute the pairwise dot-product between the MLP outputs for different patches, whose sigmoid gives a score for the binary relation “patches belong to the same segment”; we supervise the MLP with the binary cross entropy loss against patch pairs derived from the ground truth masks. This is similar to the object probes used in Li et al. (2025).

During training, images are augmented with random resized crops to $4 4 8 \times 4 4 8 / 5 1 2 \times 5 1 2$ pixels for patch size 14/16, i.e., square crops of 1024 tokens; no other augmentation is used. Targets are defined at the patch level by taking the majority semantic and instance label within each patch. The semantic probe is trained with a cross-entropy loss, in which void patches are ignored, and a binary cross-entropy loss on the pairwise similarity predictions, with equal weights. We sweep learning rates over $\left\{ 1 \cdot 1 0 ^ { - 4 } , 3 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - 3 } \right\}$ and weight decays over $\left. 1 \cdot 1 0 ^ { - 3 } , 1 \cdot 1 0 ^ { - 2 } \right.$ , training all six probes jointly on the same frozen features. Each probe is trained with AdamW $( \beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9 )$ and a batch size of 16 for 20 epochs, with a cosine learning-rate decay to $1 0 ^ { - 6 }$ and no warm-up.

At inference, images are resized to the same square resolution without cropping. Each patch predicted as a non-void class c yields a candidate soft mask, given by its pairwise similarities to all patches multiplied by their probabilities for class c. These candidate masks are filtered using class-wise mask NMS (IoU threshold 0.7), upsampled to the original image resolution, and binarized at 0.3; regions belonging to the same “stuff” class are merged. Metrics are computed at the original image resolution. Probes are evaluated after every epoch, and we select the best combination of learning rate, weight decay and epoch by validation mAP/PQ.

Depth Estimation Monocular depth estimation is the task of predicting a dense, per-pixel metric depth map from a single RGB image. It tests whether frozen features encode 3D scene geometry and not only semantics. We evaluate on two standard benchmarks: NYU Depth v2 (Silberman et al., 2012), which contains indoor scenes at $4 8 0 \times 6 4 0$ resolution, and KITTI (Geiger et al., 2012), which contains outdoor driving scenes of about $3 7 5 \times 1 2 4 2$ pixels with sparse LiDAR ground truth. We report the root mean squared error (RMSE, in meters). It is computed per image over valid pixels and averaged over the test set. Valid pixels are those whose ground truth lies strictly inside $[ 1 0 ^ { - 3 } , d _ { \mathrm { m a x } } ]$ with $d _ { \operatorname* { m a x } } = 1 0$ m on NYUv2 and 80 m on KITTI.

To predict monocular depth, we train a linear probe on top of the frozen backbone. The last-layer patch features, taken after the final layer normalization, are bilinearly upsampled by a factor of 4 and passed through a trained batch normalization layer and a $1 \times 1$ convolution with 256 output channels. To convert the outputs to depth, we predict it by classification over bins: the 256 outputs are passed through a ReLU, shifted by 0.1 and normalized into weights over 256 bins uniformly spaced in $[ 1 0 ^ { - 3 } , \overline { { d } } _ { \operatorname* { m a x } } ]$ . The predicted depth is the weighted sum of the bin centers. During training, predictions are bilinearly upsampled to the label resolution and trained with the scale-invariant log loss (Eigen et al., 2014). Similar to the linear semantic segmentation, we sweep learning rates over $\left\{ 1 \cdot 1 0 ^ { - 4 } , 3 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - 3 } \right\}$ and weight decays over $\{ 1 ^ { \smile } \cdot 1 0 ^ { - 3 } , 1 \cdot \mathrm { { 1 0 ^ { - 2 } } } \}$ , training all six probes jointly on the same frozen features. Each probe is trained with AdamW and a batch size of 16 for 38,400 iterations, with a linear learning-rate warm-up over 12,800 iterations followed by cosine decay, with gradient norms clipped at 35.

Inputs are prepared in three steps: a fixed crop, an aspect-preserving resize, and then random square crops (training) or sliding windows (inference). On NYUv2, no fixed crop is applied: we use the full 480 × 640 frame. On KITTI, we keep the bottom 352 rows and the central 1216 columns of each frame. We then resize the image so that its short side is 448/480 pixels on NYUv2 and 308/352 pixels on KITTI for patch size 14/16. During training, we apply random rotations of up to ±2.5<sup>◦</sup> and horizontal flips, each with probability 0.5, jointly to the image and the depth map (resized with nearest-neighbor interpolation). We then take a random square crop whose side equals the resized short side $\mathrm { \overline { { ( 4 4 8 ^ { 2 } / 4 8 0 ^ { 2 } } } }$ on NYUv2, $3 0 8 ^ { 2 } / 3 5 2 ^ { 2 }$ on KITTI), and apply color augmentation (random gamma, brightness and per-channel color scaling) with probability 0.5.

At inference, the whole resized image is processed with square sliding windows of the training crop size, spanning the full image height, with strides of 298/319 pixels on NYUv2 and 205/234 pixels on KITTI for patch size 14/16. Bin logits are averaged where windows overlap, bilinearly upsampled to the ground-truth resolution, converted to depth, and averaged with the prediction for the horizontally flipped image. We report the configuration with the lowest RMSE, using the probe weights at the end of training.

## A.5.2 BASELINE DETAILS

Evaluated model checkpoints Table 3 lists the external model checkpoints evaluated in this work. Parameter counts refer to the evaluated backbone; training-only prediction heads and other auxiliary components are excluded.

Table 3: External checkpoints evaluated in Tab. 2, grouped by method and ordered by first public appearance.
<table><tr><td>Model</td><td>Citation</td><td>Arch. / Variant</td><td>Training Data</td><td>Params.</td><td>External Checkpoint</td></tr><tr><td>MAE</td><td>He et al. (2022)</td><td>ViT-L/16</td><td>IN-1k</td><td>303M</td><td>facebook/vit-mae-large</td></tr><tr><td>MSN</td><td>Assran et al. (2022)</td><td>ViT-L/16</td><td>IN-1k</td><td>303M</td><td>vitl16_600ep</td></tr><tr><td>data2vec 2.0</td><td>Baevski et al. (2023)</td><td>ViT-L/16</td><td>IN-1k</td><td>303M</td><td>large_imagenet</td></tr><tr><td>I-JEPA</td><td>Assran et al. (2023)</td><td>ViT-g/16</td><td>IN-22k</td><td>1,011M</td><td>IN22K-vit.g.16-600e</td></tr><tr><td>DINOv2 (reg.)</td><td>Oquab et al. (2024)</td><td>ViT-g/14</td><td>LVD-142M</td><td>1,136M</td><td>dinov2_vitg14_reg</td></tr><tr><td>CAPI</td><td>Darcet et al. (2025)</td><td>ViT-L/14</td><td>IN-1k</td><td>303M</td><td>capi_vitl14_in1k</td></tr><tr><td>WebSSL</td><td>Fan et al. (2025)</td><td>ViT-7B/14</td><td>MC-8B</td><td>6,450M</td><td>webssl_dino7b_full8b_224</td></tr><tr><td>Franca</td><td>Venkataramanan et al. (2026)</td><td>ViT-g/14 (224 px; no RASA)</td><td>IN-22k</td><td>1,135M</td><td>franca_vitg14_In21K.pth</td></tr><tr><td>DINOv3</td><td>Siméoni et al. (2026)</td><td>ViT-7B/16</td><td>LVD-1689M</td><td>6,716M</td><td>dinov3_vit7b16</td></tr><tr><td>V-JEPA 2.1</td><td>Mur-Labadia et al. (2026)</td><td>ViT-G/16</td><td>VM-163M</td><td>1,843M</td><td>vjepa2_1_vit_gigantic_ 384</td></tr><tr><td>LingBot- Vision</td><td>Fu et al. (2026)</td><td>ViT-g/16</td><td>Unk.-163M</td><td>1,134M</td><td>robbyant/ lingbot-vision-vit-giant</td></tr></table>

Estimation of number of training tokens Table 4 details the training token estimates in Tab. 2. We count full student patch grids before masking, including all crops and mask variants, and exclude teacher and special tokens (e.g. CLS, registers). The total number of tokens in trillions is $\begin{array} { r } { T = 1 0 ^ { - 1 2 } \sum _ { r } B _ { r } S _ { r } \dot { L } _ { r } } \end{array}$ , where $B _ { r } , S _ { r }$ , and $L _ { r }$ are the batch size, optimizer update steps, and tokens per sample for each stage and input type. For DINOv3’s high resolution fine-tuning stage, which uses a mix of resolution, we use an expected 4,570 tokens per image.

We use training configurations and checkpoint metadata where available, and published recipes otherwise. Note that this estimation is best-effort, as we cannot verify the actual training configurations from the released checkpoints. MAE and MSN assume 1,600 and 600 training epochs, respectively; WebSSL assumes eight billion seen images. I-JEPA uses only 44 completed IN-22k epochs, despite the checkpoint name IN22K-vit.g.16-600e, as indicated by the checkpoint metadata. For V-JEPA 2.1, we combine batch size, number of steps, and resolution from the published schedule with the number of image and video masks from the example configurations as these values were not otherwise published. V-JEPA video inputs also assume temporal tubelets of two frames.

Table 4: Estimated training tokens for the checkpoints in Tab. 2. V-JEPA video inputs use temporal tubelets of two frames. Student views list crop counts and spatial dimensions in pixels, with frame counts for video. Batch size counts images or clips before crop and mask replication. Multi-stage models include a total row; total token counts are in trillions $( 1 0 ^ { 1 2 } )$ . Details and assumptions are given in Sec. A.5.2.
<table><tr><td colspan="3"></td><td rowspan="2">Batch size</td><td rowspan="2">Updates Student views</td><td rowspan="2">Tokens / sample</td><td rowspan="2">Tokens (T)</td></tr><tr><td>Model</td><td></td><td>Arch. Stage / input</td></tr><tr><td>MAE</td><td>L/16</td><td>Pretraining</td><td>4,096</td><td> $\approx 5 0 0 , 4 5 6 \ 1 \times 2 2 4 ^ { 2 }$ </td><td>196</td><td>0.402</td></tr><tr><td>MSN</td><td>L/16</td><td>Pretraining</td><td>1,024</td><td> $\approx 7 5 0 , 6 8 4 1 \times 2 2 4 ^ { 2 } + 1 0 \times 9 6 ^ { 2 }$ </td><td>556</td><td>0.427</td></tr><tr><td>data2vec 2.0</td><td>L/16</td><td>Pretraining</td><td>256</td><td> $7 5 0 , 0 0 0 1 6 \mathrm { m a s k s } \times 2 2 4 ^ { 2 }$ </td><td>3,136</td><td>0.602</td></tr><tr><td>CAPI</td><td>L/14</td><td>Pretraining</td><td>16,384</td><td> $5 0 0 , 0 0 0 \ 1 \times 2 2 4 ^ { 2 }$ </td><td>256</td><td>2.097</td></tr><tr><td>DINOv2*</td><td>L/16</td><td>Pretraining</td><td>2,048</td><td> $1 2 5 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>0.231</td></tr><tr><td>JEM</td><td>L/16</td><td>Pretraining</td><td>2,048</td><td> $1 2 5 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>0.231</td></tr><tr><td>I-JEPA</td><td>g/16</td><td>Pretraining</td><td>2,048</td><td> $3 0 4 { , } 9 2 0 \ 1 \times 2 2 4 ^ { 2 }$ </td><td>196</td><td>0.122</td></tr><tr><td>Franca</td><td>g/14</td><td>Pretraining</td><td>3,072</td><td> $6 2 5 , 0 0 0 \ 2 \times 2 2 4 ^ { 2 } + 8 \times 9 8 ^ { 2 }$ </td><td>904</td><td>1.736</td></tr><tr><td>DINOv2*</td><td>g/16</td><td>Pretraining</td><td>3,072</td><td> $5 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>1.389</td></tr><tr><td>JEM</td><td>g/16</td><td>Pretraining</td><td>3,072</td><td> $5 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>1.389</td></tr><tr><td rowspan="3">DINOv2 (reg.) g/14</td><td></td><td>Pretraining</td><td>3,072</td><td> $6 2 5 , 0 0 0 \ 2 \times 2 2 4 ^ { 2 } + 8 \times 9 8 ^ { 2 }$ </td><td>904</td><td>1.736</td></tr><tr><td></td><td>High-res. Total</td><td>3,072</td><td> $1 0 , 0 0 0 \ 2 \times 5 1 8 ^ { 2 } + 8 \times 9 8 ^ { 2 }$ </td><td>3,130</td><td>0.096</td></tr><tr><td></td><td></td><td></td><td> $3 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td></td><td>1.832</td></tr><tr><td rowspan="4">LingBot-Vision g/16</td><td></td><td>Pretraining Gram anchoring</td><td>3,072 3,072</td><td> $1 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904 904</td><td>0.833 0.278</td></tr><tr><td>High-res.</td><td></td><td>3,072</td><td> $1 0 0 , 0 0 0 \ 2 \times 5 1 2 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>2,440</td><td>0.750</td></tr><tr><td>Total</td><td></td><td></td><td></td><td></td><td>1.860</td></tr><tr><td>G/16</td><td>Primary / image</td><td>2,304</td><td> $1 3 5 , 0 0 0 \ 1 \times 2 5 6 ^ { 2 }$ </td><td></td><td></td></tr><tr><td rowspan="5">V-JEPA 2.1</td><td></td><td>Primary / video</td><td>128</td><td> $1 3 5 , 0 0 0 \ 2 \ \mathrm { m a s k s } \times 1 6 \times 2 5 6 ^ { 2 }$ </td><td>256</td><td>0.080</td></tr><tr><td>Cooldown / image</td><td></td><td>2,304</td><td> $1 2 , 0 0 0 \ 1 \times 5 1 2 ^ { 2 }$ </td><td>4,096 1,024</td><td>0.071</td></tr><tr><td>Cooldown / video</td><td></td><td>128</td><td> $1 2 , 0 0 0 \ 2 \ \mathrm { m a s k s } \times 6 4 \times 3 8 4 ^ { 2 }$ </td><td>36,864</td><td>0.028 0.057</td></tr><tr><td>Total</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7B/14 Pretraining</td><td></td><td></td><td> $3 , 0 7 2 \approx 2 , 6 0 4 , 1 6 7 \ 2 \times 2 2 4 ^ { 2 } + 8 \times 9 8 ^ { 2 }$ </td><td></td><td>0.235</td></tr><tr><td>DINOv2*</td><td>7B/16 Pretraining</td><td></td><td>4,096</td><td> $5 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904 904</td><td>7.232 1.851</td></tr><tr><td>JEM</td><td>7B/16 Pretraining</td><td></td><td>4,096</td><td> $5 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>1.851</td></tr><tr><td>DINOv3</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4"></td><td>7B/16 Pretraining</td><td></td><td>4,096</td><td> $1 , 0 0 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>3.703</td></tr><tr><td></td><td>Gram refinement</td><td>4,096</td><td> $7 0 , 0 0 0 \ 2 \times 2 5 6 ^ { 2 } + 8 \times 1 1 2 ^ { 2 }$ </td><td>904</td><td>0.259</td></tr><tr><td>High-res.</td><td></td><td>2,048</td><td>10,000 Mixed resolutions</td><td>4,570</td><td>0.094</td></tr><tr><td></td><td>Total</td><td></td><td></td><td></td><td>4.056</td></tr></table>

## A.6 TRAINING CONFIGURATIONS

Table 5: Shared architectures and training hyperparameters for JEM and DINOv2 baselines. Entries spanning the three model columns are shared across all model sizes.  
(a) Backbone architectures.
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="3">Model size</td></tr><tr><td>ViT-L</td><td>ViT-g</td><td>ViT-7B</td></tr><tr><td>Backbone parameters</td><td>303.1M</td><td>1.13B</td><td>6.7B</td></tr><tr><td>Transformer blocks</td><td>24</td><td>40</td><td>40</td></tr><tr><td>Embedding dimension</td><td>1024</td><td>1536</td><td>4096</td></tr><tr><td>Attention heads</td><td>16</td><td>24</td><td>32</td></tr><tr><td>Attention-head dimension</td><td>64</td><td>64</td><td>128</td></tr><tr><td>FFN type</td><td>MLP</td><td>SwiGLU</td><td>SwiGLU</td></tr><tr><td>FFN hidden dimension</td><td>4096</td><td>4096</td><td>8192</td></tr><tr><td>Patch size Register tokens</td><td></td><td>16 × 16 4</td><td></td></tr><tr><td>Positional encoding</td><td></td><td>Axial RoPE</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>LayerScale initialization</td><td></td><td> $1 \times 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Drop-path rate</td><td>0.3</td><td>0.4</td><td>0.4</td></tr></table>

(b) Shared training hyperparameters.
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="3">Model size</td></tr><tr><td>ViT-L</td><td> $\mathbf { V i T - g }$ </td><td>ViT-7B</td></tr><tr><td>Training iterations</td><td></td><td>500k</td><td></td></tr><tr><td>Global batch size</td><td>2048</td><td>3072</td><td>4096</td></tr><tr><td>LR warm-up iterations</td><td></td><td>50k</td><td></td></tr><tr><td>Layer-wise LR decay</td><td>0.9</td><td>0.995</td><td>0.98</td></tr><tr><td>LR schedule after warm-up</td><td>Cosine decay</td><td>Constant</td><td>Constant</td></tr><tr><td>Optimizer</td><td></td><td>AdamW</td><td></td></tr><tr><td>AdamW β1</td><td></td><td>0.9</td><td></td></tr><tr><td>AdamW β2</td><td>0.999</td><td>0.99</td><td>0.99</td></tr><tr><td>Initial weight decay</td><td></td><td>0.04</td><td></td></tr><tr><td>Final weight decay</td><td>0.4</td><td>0.04</td><td>0.04</td></tr><tr><td>Weight decay schedule</td><td>Cosine</td><td>Constant</td><td>Constant</td></tr><tr><td>Teacher EMA momentum</td><td> $0 . 9 9 2 \stackrel { \mathrm { c o s } } { \longrightarrow } 1 . 0 0 . 9 9 4$ </td><td></td><td>0.994</td></tr><tr><td>Global crops per sample</td><td></td><td> $2 \times 2 5 6 ^ { 2 }$ </td><td></td></tr><tr><td>Global crop weight</td><td></td><td>0.2</td><td></td></tr><tr><td>Local crops per sample</td><td></td><td> $8 \times 1 1 2 ^ { 2 }$ </td><td></td></tr><tr><td>Local crop weight</td><td></td><td>0.8</td><td></td></tr><tr><td>Parameter / reduction precision</td><td></td><td>BF16 /FP32</td><td></td></tr></table>

Table 6: JEM training hyperparameters. Entries spanning the three model columns are shared across all model sizes. Shared architectures and training hyperparameters are given in Tab. 5. Individual ablation settings are listed in Tab. 7.
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="3">Model size</td></tr><tr><td>ViT-L</td><td>ViT-g</td><td>ViT-7B</td></tr><tr><td>Default training parameters</td><td></td><td></td><td></td></tr><tr><td>Teacher-crop scale</td><td></td><td>[0.32, 1.0]</td><td></td></tr><tr><td>Number of categories</td><td></td><td>4096</td><td></td></tr><tr><td>Student and teacher temperature</td><td></td><td>0.1</td><td></td></tr><tr><td>Global mask probability / patch ratio</td><td></td><td>0.8 / 0.65</td><td></td></tr><tr><td>Local mask probability / patch ratio</td><td></td><td>0.8 / 0.50</td><td></td></tr><tr><td>Intermediate target blocks</td><td>{8,16}</td><td>{10,20,30}</td><td>{10,20,30}</td></tr><tr><td>Struct. regularizer target blocks</td><td>{0,8,16}</td><td>{0,10,20,30}</td><td>{0,10,20,30}</td></tr><tr><td>Struct. head output dimension</td><td></td><td>1024</td><td></td></tr><tr><td>Proj./Struct hidden layers</td><td></td><td>3</td><td></td></tr><tr><td>Proj./Struct heads hidden dimension</td><td>2048</td><td>2048</td><td>4096</td></tr><tr><td>Proj. head bottleneck dimension</td><td>256</td><td>384</td><td>384</td></tr><tr><td>Masked KL weight wmasked</td><td></td><td>0.8</td><td></td></tr><tr><td>Visible KL weight wvis</td><td></td><td>0.2</td><td></td></tr><tr><td>Final layer weight</td><td></td><td>1</td><td></td></tr><tr><td>Early layer weight per-layer</td><td>0.25</td><td></td><td>0.166</td></tr><tr><td>Global collapse regularization weight wglobal</td><td></td><td> $\begin{array} { r } { 0 . 1 6 6 \qquad } \\ { 1 . 0 \ \xrightarrow { 1 2 . 5 \mathrm { k } \operatorname* { l i n c a r } } 0 . 1 } \end{array}$ </td><td></td></tr><tr><td>Structural regularizer weight wstruct schedule</td><td>0.0</td><td> $\xrightarrow { 1 2 . 5 \mathrm { k \ l i n e a r } } 0 . 0 5 \xrightarrow { \mathrm { l i n e a r } } 1 . 0$ </td><td></td></tr><tr><td>Base LR w.r.t. batch size 64</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Effective peak LR</td><td> $1 . 6 9 7 \times 1 0 ^ { - 3 }$ </td><td> $6 . 9 2 8 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Effective final LR</td><td> $5 . 6 9 7 \times 1 0 ^ { - 6 }$ </td><td> $6 . 9 2 8 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 0 0 \times 1 0 ^ { - 3 }$   $1 . 2 0 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Ablation study on IN22k (Tab. 1)</td><td></td><td></td><td></td></tr><tr><td>Training steps</td><td>250k</td><td></td><td></td></tr><tr><td>Base LR</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td></td><td></td></tr><tr><td>ViT-L on IN1k (Tab. 2)</td><td></td><td></td><td></td></tr><tr><td>Training steps</td><td>125k</td><td></td><td></td></tr><tr><td>Base LR</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td></td><td></td></tr><tr><td>Global crop weight</td><td>0.5</td><td></td><td></td></tr><tr><td>Local crop weight</td><td>0.5</td><td></td><td></td></tr></table>

Table 7: Ablation settings. Experiment-specific settings for the ablations in Tab. 1. All experiments use the shared ablation training settings in Tab. 6.  
(a) View construction (Tab. 1a).
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>w/o local crops</td><td></td></tr><tr><td>Global crop weight</td><td>1.0</td></tr><tr><td>Local crop weight</td><td>0.0</td></tr><tr><td>w/o masking</td><td></td></tr><tr><td>Global mask probability</td><td>0.0</td></tr><tr><td>Local mask probability</td><td>0.0</td></tr><tr><td>w/o stud.-teach. shift</td><td></td></tr><tr><td>Min. global crop S-T overlap</td><td>1.0</td></tr><tr><td>w/o independ. aug. Prob. color jittering</td><td>0.0</td></tr><tr><td></td><td></td></tr><tr><td>Prob. grayscale</td><td>0.0</td></tr><tr><td>Prob. Gaussian blur</td><td>0.0</td></tr><tr><td>Prob. solarization</td><td>0.0</td></tr></table>

(b) KL alignment (Tab. 1b).
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>w/o multi-layer loss Early-layer weight per-layer</td><td></td></tr><tr><td></td><td>0.0</td></tr><tr><td>w/o visible-patch loss</td><td>1.0</td></tr><tr><td>Wmasked Wvis</td><td>0.0</td></tr><tr><td></td><td></td></tr><tr><td>w/o masked-patch loss</td><td></td></tr><tr><td>Wmasked</td><td>0.0</td></tr><tr><td> $w _ { \mathrm { v i s } }$ </td><td>1.0</td></tr><tr><td>w/o EMA</td><td></td></tr><tr><td>Teacher EMA momentum</td><td>0.0</td></tr><tr><td>Marginal entropy weight</td><td>1.5</td></tr><tr><td>Conditional entropy weight</td><td>0.5</td></tr><tr><td>Wglobal post-decay</td><td>0.5</td></tr></table>

(c) Objective (Tab. 1c).
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td></td></tr><tr><td> $w _ { \mathrm { v i s } }$ </td><td>0.0</td></tr><tr><td>Wmasked</td><td>0.0</td></tr><tr><td>w/o Rglobal</td><td></td></tr><tr><td>Wglobal</td><td>0.0</td></tr><tr><td>w/o Rstruct</td><td></td></tr><tr><td> $w _ { \mathrm { s u v e t } }$ </td><td>0.0</td></tr></table>

Table 8: DINOv2 baseline training hyperparameters. Entries spanning all model columns are shared across model sizes. Shared architectures and training hyperparameters are given in Tab. 5. Loss weights are configured coefficients.
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="3">Model size</td></tr><tr><td>ViT-L</td><td>ViT-g</td><td>ViT-7B</td></tr><tr><td>Global-crop scale</td><td></td><td>[0.32, 1.00]</td><td></td></tr><tr><td>Local-crop scale</td><td></td><td>[0.05, 0.32]</td><td></td></tr><tr><td>Global mask probability</td><td></td><td>0.5</td><td></td></tr><tr><td>Mask-ratio range</td><td></td><td> $[ 0 . 1 , \ 0 . 5 ]$ </td><td></td></tr><tr><td>DINO prototypes</td><td>65,536</td><td>131,072</td><td>262,144</td></tr><tr><td>iBOT prototypes</td><td>65,536</td><td>131,072</td><td>98,304</td></tr><tr><td>DINO/iBOT heads layers</td><td></td><td>3</td><td></td></tr><tr><td>DINO head hidden dimension</td><td>2048</td><td>2048</td><td>8192</td></tr><tr><td>iBOT head hidden dimension</td><td>2048</td><td>2048</td><td>4096</td></tr><tr><td>DINO head bottleneck dimension</td><td>256</td><td>384</td><td>512</td></tr><tr><td>iBOT head bottleneck dimension</td><td>256</td><td>384</td><td>384</td></tr><tr><td>Student temperature</td><td></td><td>0.1</td><td></td></tr><tr><td>Teacher temperature</td><td colspan="3">0.04→ 0.07 over 50k iterations</td></tr><tr><td>DINO</td><td colspan="3">1.0</td></tr><tr><td>iBOT</td><td colspan="3">0.5</td></tr><tr><td>KoLeo</td><td colspan="3">0.1</td></tr><tr><td>Base LR w.r.t. batch size 64</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $6 \times 1 0 ^ { - 5 }$ </td><td> $6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Effective peak LR</td><td> $1 . 6 9 7 \times 1 0 ^ { - 3 }$ </td><td> $4 . 1 5 7 \times 1 0 ^ { - 4 }$ </td><td> $4 . 8 0 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Effective final LR</td><td> $5 . 6 5 7 \times 1 0 ^ { - 6 }$ </td><td> $4 . 1 5 7 \times 1 0 ^ { - 4 }$ </td><td> $4 . 8 0 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Last-layer freeze</td><td colspan="3">Initial 5k iterations</td></tr></table>

## A.7 QUALITATIVE EXAMPLES

We show visualizations of representations of a JEM ViT-g model trained on A140M in Fig. 9 and Fig. 10. Input images have a resolution of 896 × 896 pixels, yielding a patch grid of 56 × 56 feature vectors. PCA and cosine similarity visualizations are computed on the patch grid, and the resulting RGB image is upscaled to the input image size using nearest neighbor interpolation.

Input  
PCA  
Input  
PCA  
Input  
PCA  
Input  
PCA  
![](images/a07caf12152c191f06fba3f57304226b799ca9556f20bdb4f97367bf621a58eb.jpg)  
Figure 9: PCA visualization of final-layer patch representations of a JEM ViT-g model. For each of the 20 images, we show the input and its RGB projection onto the first three principal components computed individually on each image.

Input  
Cosine  
Input  
Cosine  
Input  
Cosine  
Input  
Cosine  
![](images/8a16df8d3052f8dac706ed5348dccc4f5c7a98aad533972fe72e02fe8f0fa360.jpg)  
Figure 10: Cosine similarity between final-layer patch representations and the center-patch representation of a JEM $\mathrm { V i T - g }$ model. Red markers identify the reference patch. The color scale interpolates between similarity values of 0 (full blue) and 1 (full yellow).