# Subspace Learning with Outlier Exposure for Out-of-Distribution Detection

Hongxu Shen<sup>a,1</sup>, Kun Fang<sup>b,1</sup>, Jiahui Zheng<sup>c</sup>, Jian Sun<sup>a</sup>, Qinghua Tao<sup>a,∗</sup>

<sup>a</sup>School of Automation, Beijing Institute of Technology, Beijing, China <sup>b</sup>The Hong Kong Polytechnic University, Hong Kong, China <sup>c</sup>Tsinghua University, Beijing, China

## Abstract

Out-of-Distribution (OoD) detection is pivotal for reliable deployments of Deep Neural Networks (DNNs), aiming to distinguish anomalous samples from the In-Distribution (InD) data in training. By introducing outliers as surrogate OoD data to finetune DNNs, Outlier Exposure (OE) has greatly advanced OoD detection in recent years. Existing OE-based research remains predominantly logit-centric, either by generating diverse outliers or by regularizing them with objectives operating at the prediction-output level. Nonetheless, these approaches leave much of the rich information in feature spaces untapped. In this work, we propose an OE-based detection framework, named RecOE, that learns appropriate feature subspaces from InD data and auxiliary outliers, such that the InD-OoD discrepancy can be efectively characterized through the learned feature subspaces. Specifically, the feature subspaces are optimized to capture the dominant variations of InD and outliers by minimizing reconstruction errors therein. Besides, a sample-level objective discourages reconstruction in the opposing subspace, while an orthogonality regularizer reduces subspace overlap, further promoting discriminative subspaces. At inference, the diference in reconstruction errors between the learned subspaces serves as the OoD detection score. Extensive experiments show the superiority of RecOE on multiple benchmarks, in which significant

performance gains are achieved over recent strong baselines with only slight extra overhead from two small-size matrices of subspace bases. The source code is available at https://github.com/shenhongxu123/RecOE.

Keywords: Out-of-distribution detection, outlier exposure, principal component analysis, subspace learning

## 1. Introduction

While deep neural networks (DNNs) have achieved vast success in realworld applications [1, 2], they are known to yield overconfident predictions on samples that deviate from the in-distribution (InD) training data [3], commonly known as Out-of-Distribution (OoD) samples. Such behavior poses substantial risks in safety-critical scenarios, including autonomous driving, medical diagnosis, and financial systems [4], making OoD detection a crucial research problem.

Existing OoD detection methods can be broadly categorized according to whether they require finetuning of a DNN. A series of post-hoc methods [5, 6, 7, 8, 9, 10, 11, 12] have been proposed with the main idea of leveraging the responses from a pretrained model to devise detection scores that capture the InD-OoD discrepancy. Another line of research introduces outliers as surrogate OoD data to finetune the DNN with regularization terms, which efectively promotes detection performance, known as Outlier Exposure (OE) [13]. OE-based methods advance OoD detection in two main aspects: outlier construction and outlier regularization. The former improves auxiliary outliers to provide surrogate OoD supervision through boundary-aware mining [14, 15], ID-outlier mixing [16], and informative extrapolation [17]. The latter develops regularization objectives to finetune the DNN for enhanced InD-outlier separation via energy-based margin regularization [6], gradient regularization of the OoD score function [18], or feature-based geometric constraints [19, 20]. Our work falls into the outlier-regularization category.

Early OE methods rely on the output logit space for both outlier generation and regularization [19]. The logit layer in DNNs primarily encodes coarse-grained knowledge associated with predictive classification probabilities. In contrast, feature spaces of DNNs encode far richer and structured information of data distributions, including hierarchical representations, geometric patterns, and semantics learned across layers. Recent feature-based OE methods promote InD–OoD separation using geometric references derived from classifier weights [19] or predefined structures [20]. Please find a detailed discussion in Section 2.1.

In this work, we propose RecOE, an OE-based OoD detection framework that exploits InD-OoD discrepancy through optimizing low-dimensional feature subspaces. We posit that distribution-associated dominant directions of feature variation provide useful structure for distinguishing InD data from outliers. Accordingly, we construct and optimize an InD-associated subspace and an auxiliary-outlier-associated subspace spanned by learnable orthonormal bases $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ , respectively. As commonly in OE-based detection, auxiliary outliers serve as a surrogate non-InD reference when the open-world OoD distribution is unavailable during training. The subspace spanned by $U _ { \mathrm { a u x } }$ therefore characterizes the structures exposed by these auxiliary training outliers, providing an important reference basis for separating InD data from OoD data.

In our RecOE framework, each subspace is constructed to capture the dominant variations of its corresponding feature distribution while those two subspaces are regularized to remain separated from each other. To this end, we jointly optimize the subspaces and the backbone, allowing the feature representations to adapt to the fitting and separation objectives. We therefore formulate two complementary learning objectives, subspace fitting and subspace separation, detailed as follows.

• Subspace Fitting. Along with Principal Component Analysis (PCA) [21], we optimize $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ by minimizing the reconstruction error of training samples on $U _ { \mathrm { i n } }$ and that of outliers on $U _ { \mathrm { a u x } }$ , such that $U _ { \mathrm { i n } }$ captures intrinsic InD patterns and $U _ { \mathrm { a u x } }$ maximally describes the outliers to characterize the potential OoD.

• Subspace Separation. The separation between the two subspaces is optimized at both sample and parameter levels. At the sample level, we deploy the margin-based hinge loss to encourage InD features with large reconstruction errors on the subspace $U _ { \mathrm { a u x } }$ , similarly for outlier features on $U _ { \mathrm { i n } }$ . This ensures $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ to retain minimal information from the opposing distribution. At the parameter level, an orthogonality regularizer is imposed to disentangle the two subspaces $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$

Notably in optimization, rather than solving an eigenvalue problem in standard PCA, the orthonormal bases can be optimized on Stiefel manifold [22] with Riemannian Adam optimizer [23, 24]. Thus, RecOE flexibly adapts to the end-to-end manner under OE-based training framework for OoD detection. We further introduce progressive-rank optimization to capture dominant feature variations at low rank before gradually expanding the subspaces to the target rank. This helps reduce interference from poorly estimated directions during early training.

![](images/004a9ba56f6bec7777c0a5075017d1f733399253c72cb5109cf2f98859fada02.jpg)  
Figure 1: Experimental exemplification on the efectiveness of RecOE, with CIFAR-10 as InD and iSUN as OoD. We interpolate the distribution of (a) reconstruction errors from a standard PCA on the pretrained DNN and (b) reconstruction contrast score by RecOE.

The two optimized subspaces yield contrasting reconstructions of InD and OoD features. Specifically, InD samples are expected to be well reconstructed by $U _ { \mathrm { i n } }$ and poorly reconstructed by $U _ { \mathrm { a u x } }$ , whereas OoD samples tend to exhibit the opposite behavior. We use the diference between the two reconstruction errors as an efective detection score, as illustrated in Figure 1.

Our main contributions are summarized as follows.

• We introduce a feature-subspace perspective for OE regularization, revealing that intrinsic low-dimensional structures of InD and outlier features provide promising discriminative supervision for OoD detection.

• We propose RecOE, a new framework that learns distribution-specific feature structures with InD-OoD separability in the subspaces. By fitting each subspace to its own distribution and suppressing information from the opposite one, their relative reconstruction naturally leads to a discriminative detection score. We introduce progressive-rank optimization to prioritize dominant feature variations early in training.

• Extensive experimental results validate the superiority of RecOE with state-of-the-art performance gains at only slight extra overhead from two small-size matrices.

## 2. Preliminaries

## 2.1. OE-based OoD Detection

In OoD detection, the key is to devise a score function $S ( { \pmb x } )$ that reflects the likelihood of a test sample x belonging to the distribution of training data (InD):

$$
{ \mathrm { c a t e g o r y ~ o f ~ } } { \textbf { \em x } } = { \left\{ \mathrm { I n D } , \quad { \mathrm { i f ~ } } S ( { \pmb x } ) \geq \tau , \right. }\tag{1}
$$

The threshold $\tau$ is chosen such that a large proportion of InD data is correctly classified. Generally, given a target model $f ( \cdot )$ , its responses w.r.t. the given sample $\mathbf { \delta } _ { \mathbf { x } , \mathbf { \delta } }$ such as logits and features, are utilized to devise the scoring function $S ( \cdot )$ . We then outline a series of representative OoD detection methods based on Outlier Exposure (OE) [13], which are closely related.

OE-based methods introduce outliers as regularization to finetune $f ( \cdot )$ with the following learning objective:

$$
\operatorname* { m i n } _ { f } \quad \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { \mathrm { t r } } } \left[ \mathcal { L } _ { \mathrm { C E } } \big ( f ( x ) , y \big ) \right] + \lambda \mathbb { E } _ { x ^ { \prime } \sim \mathcal { D } _ { \mathrm { a u x } } } \left[ \mathcal { L } _ { \mathrm { O E } } \big ( f ( x ^ { \prime } ) \big ) \right] .\tag{2}
$$

In Eq. (2), ${ \mathcal { D } } _ { \mathrm { t r } }$ and $\mathcal { D } _ { \mathrm { a u x } }$ denote the training dataset and the auxiliary outlier dataset, respectively, and $\mathcal { L } _ { \mathrm { C E } }$ denotes the cross-entropy loss. The performance of $f ( \cdot )$ on outliers is regularized through an OE regularization loss $\mathcal { L } _ { \mathrm { O E } }$ , which is typically defined as the KL divergence between the outlier predictions $f ( { \pmb x } ^ { \prime } )$ and a uniform distribution. In this way, outliers act as surrogates for OoD data, allowing $f ( \cdot )$ to be trained with the regularization term $\mathcal { L } _ { \mathrm { O E } }$ that strengthens the separation between InD and OoD samples.

A series of OE-based methods explore in diferent ways to generate diverse outliers. MixOE [16] conducts pixel-level interpolation between training samples and outliers; DivOE [17] introduces adversarial perturbations on outliers; DREAM-OOD [25] leverages difusion models in the latent space to generate photo-realistic outliers. Meanwhile, another line of OE-based research focuses on designing advanced regularization terms $\mathcal { L } _ { \mathrm { O E } }$ to finetune the DNN.

In [26], a balanced Energy regularization loss is introduced by minimizing the Energy margins between InD training data and outliers; GReg [18] imposes gradient penalties to incorporate local information of data for finetuning.

Comparison with related methods. Among OE regularization methods, approaches in feature space use auxiliary outliers to shape representation geometry. PFS [19] imposes separation constraints using the subspace spanned by classifier weights and its orthogonal complement, whereas VPC [20] uses predefined geometric references. RecOE learns separate subspaces for InD and auxiliary-outlier features, each fitted to the dominant variations of the corresponding distribution. On the other hand, subspace structure is also used in post-hoc detection with pretrained backbones. ViM [27] combines principal-subspace residuals with classification logits, while KPCA [11] measures reconstruction errors in kernel space. WDiscOOD [28] combines scores from discriminative and residual subspaces obtained through whitened linear discriminant analysis of InD features. These methods derive their subspace references from InD features while keeping the backbone fixed. RecOE uses auxiliary outliers as a proxy for OoD data and jointly optimizes both subspace bases and the backbone. This joint learning shapes the representations to preserve distribution-specific structure and promote InD-outlier separation.

## 2.2. PCA Techniques

PCA [21] has long been a fundamental tool to analyze high-dimensional data. It seeks a low-dimensional subspace that preserves the dominant variations of data and has been widely used for dimensionality reduction and feature representation analysis [11].

Given a centered data/feature vector $z \in \mathbb { R } ^ { d } \ ( \mathrm { i . e . , } \ \bar { z } = 0 )$ , PCA seeks a linear projection matrix $U \in \mathbb { R } ^ { d \times k }$ , typically with $k \ll d ,$ that maps z onto a k-dimensional principal subspace. The matrix U contains a set of orthonormal bases spanning the projected subspace, such that $U ^ { \top } U = I _ { k }$ which can also be regarded as residing in the celebrated Stiefel manifold [22]. PCA can be equivalently optimized through maximizing the projection variation or minimizing the reconstruction error onto the subspace, the latter of which is adopted in our method.

Approaching from the reconstruction-based optimization, z is projected into the subspace and then mapped, i.e., $z _ { \mathrm { r e c } } = U U ^ { \top } z$ . The discrepancy lost during this process is formally defined as the reconstruction error:

$$
e ( \boldsymbol { z } ; U ) = \lVert \boldsymbol { z } - U U ^ { \top } \boldsymbol { z } \rVert _ { 2 } ^ { 2 } ,\tag{3}
$$

which quantifies the residual energy of the feature vector lying orthogonal to the learned subspace. Small values indicate better reconstructions, i.e., the most intrinsic patterns of $_ { z }$ are well retained in this subspace. Conversely, a large reconstruction error implies that the feature contains significant variations falling outside this subspace. This inherent property is crucial for motivating the optimization and also the inference in our method as introduced in Section 3.

Notation. DNNs with standard C-class classification settings are considered. Let $\mathcal { X } \subset \mathbb { R } ^ { m }$ denote the input space and $\mathcal { Y } \subset \{ 1 , 2 , \dotsc , C \}$ denote the label space. We denote the InD training dataset and auxiliary outlier dataset by ${ \mathcal { D } } _ { \mathrm { t r } }$ and $\mathcal { D } _ { \mathrm { a u x } }$ , respectively. A DNN is defined as $f ( \cdot ) \ : = \ : h ( g ( \cdot ) )$ , where $g : \mathbb { R } ^ { m }  \mathbb { R } ^ { d }$ denotes the backbone encoder and $h : \mathbb { R } ^ { d }  \mathbb { R } ^ { C }$ denotes the classifier head. For an input sample ${ \mathbf { } } x ,$ we denote its penultimate-layer feature as $\pmb { z } = \ b { g } ( \pmb { x } ) \in \mathbb { R } ^ { d }$

## 3. Methodology

In this section, we present RecOE, an end-to-end feature-subspace learning framework for OE-based OoD detection. As illustrated in Figure 2, our framework characterizes the intrinsic patterns of InD and OoD samples through the optimization with two feature subspaces. We first elaborate the subspace modeling, and then introduce our optimization scheme.

## 3.1. Subspace Modeling

Feature Preprocessing. Following previous feature-space modeling methods [9, 19], we apply centered $\ell _ { 2 }$ normalization before learning the subspaces. Each feature is first normalized as $\widetilde { \pmb { z } } _ { i } = \pmb { z } _ { i } / \| \pmb { z } _ { i } \| _ { 2 }$ . For each distribution $j \in \{ \mathrm { i n , a u x } \}$ , we compute the mean of its normalized features in the current mini-batch $\begin{array} { r } { \dot { B } _ { j } \colon \overline { { z } } _ { j } = \frac { 1 } { | B _ { i } | } \sum _ { i \in B _ { j } } \widetilde { z } _ { i } } \end{array}$ . During training, this batch mean is used for centering and to update a distribution-specific exponential moving average (EMA) center $\mu _ { j }$ . Each EMA center is initialized from the first batch mean and subsequently updated as ${ \pmb { \mu } } _ { j }  0 . 9 9 { \pmb { \mu } } _ { j } + 0 . 0 1 { \overline { { \pmb { z } } } } _ { j }$ . The centered representation used by subspace $j$ is

$$
\widehat { z } _ { i , j } = \left\{ \begin{array} { l l } { \widetilde { z } _ { i } - \overline { { z } } _ { j } , } & { \mathrm { d u r i n g ~ t r a i n i n g , } } \\ { \widetilde { z } _ { i } - \pmb { \mu } _ { j } , } & { \mathrm { d u r i n g ~ i n f e r e n c e . } } \end{array} \right.\tag{4}
$$

![](images/520ad7d57ec5ace09d462d51824ca19c3016c93c882c4dbc3d3a0affbb6ba386.jpg)  
Figure 2: Illustration of our RecOE for OE-based OoD detection. In training, RecOE seeks separation between InD and outliers through two subspaces spanned by $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } } ,$ which are optimized not only to capture maximal information of InD data and OoD data respectively $( \mathrm { i . e . , ~ } \mathcal { L } _ { \mathrm { f i t } } )$ , but also to encode minimal information from the opposing data distribution $( \mathrm { i . e . , \ : \mathcal { L } _ { \mathrm { s e p } } } )$ , together with an orthogonality term $\left( \mathrm { i . e . , \ : \mathcal { L } _ { \mathrm { o r t h o } } } \right)$ to further refine the two subspaces. At inference, the diference of reconstruction errors in the learned two subspaces is naturally utilized as detection scores.

The center is always selected according to the subspace performing reconstruction, including when evaluating features from the opposing distribution. During inference, the EMA centers remain fixed, so centering does not depend on the composition of the test batch.

The two preprocessing operations serve complementary purposes. The $\ell _ { 2 }$ normalization removes diferences in feature magnitude, making projection and reconstruction-error measurements depend on the normalized representations rather than their original scales. Mean centering follows the principle used in PCA, removing the distribution mean before identifying principal directions of variation. Each subspace therefore models deviations around its distribution-specific center. When a feature is evaluated against the opposing distribution’s subspace, using that subspace’s center measures reconstruction relative to the corresponding distribution.

Feature Subspace. We construct two subspaces to model the dominant variations of InD features and OoD features, respectively. Since the complete open-world OoD distribution is unavailable during training, outliers are used to learn a proxy OoD subspace that captures non-InD structures observed during training. Along with the spirits in PCA, we learn each subspace by minimizing reconstruction errors for its associated distribution, while encouraging separation between the two subspaces.

We denote that the subspaces are spanned by two orthonormal bases $U _ { \mathrm { i n } } ~ \in ~ \mathbb { R } ^ { d \times k }$ and $U _ { \mathrm { a u x } } \in \mathbb { R } ^ { d \times k }$ , naturally residing on the Stiefel manifold $\operatorname { S t } ( d , k )$ , i.e., $U _ { \mathrm { i n } } ^ { \top } U _ { \mathrm { i n } } = I _ { k }$ and $U _ { \mathrm { a u x } } ^ { \top } U _ { \mathrm { a u x } } = I _ { k }$ . For an input x with feature $z = g ( \pmb { x } )$ , let $\widehat { \boldsymbol { z } } _ { j } ( \pmb { x } )$ denote its normalized feature centered using the center associated with subspace $j ,$ as defined in Feature Preprocessing. Following Eq. (3), we define

$$
E _ { j } ( \pmb { x } ) : = \left\| \widehat { \pmb z } _ { j } ( \pmb { x } ) - \pmb { U } _ { j } \pmb { U } _ { j } ^ { \top } \widehat { \pmb z } _ { j } ( \pmb { x } ) \right\| _ { 2 } ^ { 2 } , \qquad j \in \{ \mathrm { i n , a u x } \} .\tag{5}
$$

For each subspace $j , \ E _ { j } ( { \pmb x } )$ measures the squared residual that remains after orthogonal projection onto the corresponding subspace. A smaller $E _ { j } ( { \pmb x } )$ indicates that the subspace $j$ better explains the feature relative to its distribution-specific center.

Detection Score. Given the learned subspaces, we define the detection score for a test sample x as

$$
\begin{array} { r } { S ( \pmb { x } ) = E _ { \mathrm { a u x } } ( \pmb { x } ) - E _ { \mathrm { i n } } ( \pmb { x } ) . } \end{array}\tag{6}
$$

This score compares how well the two subspaces reconstruct the feature relative to their respective centers. The learning objective encourages InD features to have lower reconstruction errors in the InD subspace than in the proxy OoD subspace, yielding higher scores. Lower scores indicate greater relative compatibility with the proxy OoD subspace and are used to identify OoD samples.

## 3.2. Subspace Optimization

We now elaborate our optimization strategy to obtain the aforementioned desired feature subspaces $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ . The optimization enforces two complementary principles: each feature distribution is well aligned with its corresponding subspace, while the two learned subspaces are explicitly separated from each other.

Subspace Fitting. The two feature subspaces $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ are primarily designed to capture the intrinsic patterns of their respective data distributions, similar to the principal components in PCA. We establish the objective of minimizing the reconstruction errors on the InD training samples $\mathbf { \bar { \mathcal { D } } } _ { \mathrm { t r } } = \{ ( \mathbf { x } _ { \mathrm { t r } } ^ { ( i ) } , y _ { \mathrm { t r } } ^ { ( i ) } ) \} _ { i = 1 } ^ { N _ { \mathrm { t r } } }$ onto $U _ { \mathrm { i n } }$ and the outliers $\mathcal { D } _ { \mathrm { a u x } } = \{ \pmb { x } _ { \mathrm { a u x } } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { a u x } } }$ onto $U _ { \mathrm { a u x } }$ respectively. This design enables each subspace to encode the information most relevant to its own data. To be specific, this loss is given by:

$$
\mathcal { L } _ { \mathrm { f i t } } = \sum _ { i = 1 } ^ { N _ { \mathrm { t r } } } E _ { \mathrm { i n } } ( \pmb { x } _ { \mathrm { t r } } ^ { ( i ) } ) + \sum _ { i = 1 } ^ { N _ { \mathrm { a u x } } } E _ { \mathrm { a u x } } ( \pmb { x } _ { \mathrm { a u x } } ^ { ( i ) } ) .\tag{7}
$$

Subspace Separation. For OoD detection, we encourage each subspace to distinguish features from its corresponding distribution from those of the opposing distribution. Specifically, we penalize reconstruction errors below a prescribed margin when features are projected onto the opposing subspace, yielding the following hinge loss:

$$
\mathcal { L } _ { \mathrm { s e p } } = \sum _ { i = 1 } ^ { N _ { \mathrm { t r } } } \mathrm { m a x } \Big ( 0 , m - E _ { \mathrm { a u x } } \big ( { \pmb x } _ { \mathrm { t r } } ^ { ( i ) } \big ) \Big ) + \sum _ { i = 1 } ^ { N _ { \mathrm { a u x } } } \mathrm { m a x } \big ( 0 , m - E _ { \mathrm { i n } } \big ( { \pmb x } _ { \mathrm { a u x } } ^ { ( i ) } \big ) \big ) ,\tag{8}
$$

where m is the margin in hinge loss. Minimizing $\mathcal { L } _ { \mathrm { s e p } }$ encourages crosssubspace reconstruction errors to reach or exceed $m$ , with no further penalty once the margin is satisfied. Together with ${ \mathcal { L } } _ { \mathrm { f i t } }$ , this objective encourages each feature to be reconstructed more accurately by its corresponding subspace. We further penalize overlap between the two subspaces through $\mathcal { L } _ { \mathrm { o r t h o } } = \| \pmb { U } _ { \mathrm { i n } } ^ { \top } \pmb { U } _ { \mathrm { a u x } } \| _ { F } ^ { 2 }$ . This parameter-level regularizer complements samplelevel separation objective by discouraging shared directions between the two learned bases.

Joint Optimization. With the aforementioned schemes for subspace fitting and subspace separation, we formulate the overall optimization objective for our subspace-based OE regularization as follows:

$$
\operatorname* { m i n } _ { f ( \cdot ) , U _ { \mathrm { i n } } , U _ { \mathrm { a u x } } } \quad \mathcal { L } _ { \mathrm { C E } } + \gamma \mathcal { L } _ { \mathrm { s u b } } ,\tag{9}
$$

where $\mathcal { L } _ { \mathrm { s u b } } = \mathcal { L } _ { \mathrm { f i t } } + \mathcal { L } _ { \mathrm { s e p } } + \beta \mathcal { L } _ { \mathrm { o r t h o } }$ $\mathcal { L } _ { \mathrm { C E } }$ is the standard cross-entropy loss, and coeficients $\gamma$ and $\beta$ penalize the corresponding loss terms. During training, $\mathcal { L } _ { \mathrm { s u b } }$ jointly updates the two subspace bases and the backbone ${ \mathit { g } } ,$ allowing the feature representations to adapt to the subspace fitting and separation objectives. Meanwhile, $\mathcal { L } _ { \mathrm { C E } }$ encourages the retention of class-discriminative information. This joint optimization promotes separation between InD and OoD features within the learned representation space. The feature visualization in Figure 3 further illustrates the resulting separation between InD and OoD samples.

![](images/f430b852b27f2bafb85d5c0f1191239ed30015b9d4b09f919d9cbd3d77317160.jpg)  
Figure 3: T-SNE visualization of the learned features for InD and OoD data. RecOE achieves a clearer separation between InD (blue) and OoD (red and green) features compared to the Pretrained Model and the OE Model. By regularizing the network with subspace loss term, the network is able to output more distinguishable InD and OoD features. This distinct feature separation inherently facilitates the subsequent OoD detection based on reconstruction errors.

The parameters in optimization include the pretrained DNN $f ( \cdot )$ and the two orthonormal matrices $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ . We adopt standard stochastic gradient descent optimizer to update the network parameters in $f ( \cdot )$ , in line with the settings in existing OE-based methods, while using the Riemannian Adam optimizer [23, 24] to update the two matrices $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ constrained on the Stiefel manifold, i.e., $U _ { \mathrm { i n } } ^ { \top } U _ { \mathrm { i n } } = I _ { k }$ and $U _ { \mathrm { a u x } } ^ { \top } U _ { \mathrm { a u x } } = { \cal I } _ { k }$

## 3.3. Progressive-Rank Optimization

During optimization, the subspace bases adapt to the evolving feature representations produced by the backbone. We therefore introduce subspace capacity progressively, allowing a smaller set of directions to be optimized before additional directions are incorporated.

Let k denote the target rank and $r ^ { ( s ) }$ the shared active rank of the two subspaces at stage s. Starting from $r ^ { ( 0 ) } = \operatorname* { m a x } \{ 1 , \lceil k / 2 \rceil \}$ , we expand both subspaces every K optimization steps according to

$$
r ^ { ( s + 1 ) } = \operatorname* { m i n } \left\{ k , r ^ { ( s ) } + \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { k - r ^ { ( s ) } } { 2 } \right\rceil \right\} \right\} .\tag{10}
$$

This schedule gradually introduces the remaining directions until the target rank is reached.

At each expansion, we preserve the learned subspace and initialize addi tional directions in its orthogonal complement. The expanded basis remains orthonormal, and all directions continue to be jointly optimized. This procedure increases the representation capacity while retaining the subspace learned in preceding stages. Training uses the current-rank bases, whereas inference uses the final rank-k bases. The initialization, expansion procedure, optimization settings and complete algorithm are detailed in Appendix A.2.

## 4. Experiments

In this section, we present extensive numerical experiments to evaluate the proposed RecOE with comparisons to a series of classic and strong baselines covering post-hoc and OE-based methods in Section 4.2. We also conduct multiple ablation studies on the roles of our optimized subspaces and the mechanisms fundamental to accurate detections in Section. 4.3.

## 4.1. Experimental Setups

The main protocol in our experiments mainly follows [17], covering data, models, and training strategies.

Datasets. With the established protocols in [17, 13, 29], we experiment on CIFAR-10 [30] and ImageNet-1K [31] benchmarks. For CIFAR-10 as InD, we assess OoD detection performance using Textures [32], Places365 [33], iSUN [34], LSUN\_Crop [35], and LSUN\_Resize [35] as OoD datasets, as typically adopted in the community. Tiny\_ImageNet [36] is utilized as the auxiliary surrogate OoD dataset during OE-based training. For ImageNet, we use ImageNet-1K as the InD dataset and ImageNet-21K\_p [37] as the surrogate OoD data. The OoD data for ImageNet are iNaturalist [38], SUN [39], Places [33], and Textures [32].

Implementation. We employ WideResNet40-2 [40] and ResNet50 [41] as backbones for CIFAR and ImageNet, respectively. Following previous OE methods [17], we finetune for 10 epochs on CIFAR-10 and 4 epochs on ImageNet, respectively. For RecOE, the backbone is optimized using SGD with cosine annealing, while the matrices $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ are learned with the Riemannian Adam optimizer [23, 24].

<table><tr><td>Method</td><td>FPR95↓</td><td>AUROC↑</td><td>ID-ACC↑</td></tr><tr><td>Post-hoc methods</td><td></td><td></td><td></td></tr><tr><td>MSP</td><td>52.21</td><td>90.61</td><td>94.84</td></tr><tr><td>ODIN</td><td>32.84</td><td>90.67</td><td>94.84</td></tr><tr><td>Mahalanobis</td><td>36.27</td><td>92.69</td><td>94.84</td></tr><tr><td>Energy</td><td>32.82</td><td>91.99</td><td>94.84</td></tr><tr><td>ASH</td><td>33.01</td><td>92.11</td><td>94.84</td></tr><tr><td>PCA</td><td>69.84</td><td>74.82</td><td>94.84</td></tr><tr><td>KPCA (CoP)</td><td>40.22</td><td>91.28</td><td>94.84</td></tr><tr><td>KPCA (CoRP)</td><td>31.82</td><td>93.80</td><td>94.84</td></tr><tr><td>Mahalanobis++</td><td>19.56</td><td>95.90</td><td>94.84</td></tr><tr><td>OE-based methods</td><td></td><td></td><td></td></tr><tr><td>OE</td><td> $1 3 . 7 6 \pm 0 . 2 3$ </td><td> $9 7 . 5 3 \pm 0 . 0 3$ </td><td> $9 4 . 5 1 \pm 0 . 0 6$ </td></tr><tr><td>EnergyOE</td><td> $1 8 . 1 5 \pm 0 . 3 2$ </td><td> $9 0 . 9 0 \pm 0 . 1 8$ </td><td> $9 4 . 5 1 \pm 0 . 0 1$ </td></tr><tr><td>POEM</td><td> $1 7 . 8 1 \pm 0 . 7 8$ </td><td> $9 0 . 9 6 \pm 0 . 3 7$ </td><td> $9 4 . 9 0 \pm 0 . 0 4$ </td></tr><tr><td>MixOE</td><td> $1 3 . 5 7 \pm 0 . 2 5$ </td><td> $9 7 . 4 7 \pm 0 . 0 7$ </td><td> ${ \bf 9 4 . 9 8 \pm 0 . 0 2 }$ </td></tr><tr><td>DivOE</td><td> $1 1 . 6 6 \pm 0 . 0 4$ </td><td> $9 7 . 8 2 \pm 0 . 0 4$ </td><td> $9 4 . 6 6 \pm 0 . 1 2$ </td></tr><tr><td>GReg</td><td> $1 2 . 6 9 \pm 1 . 7 8$ </td><td> $9 7 . 0 3 \pm 0 . 5 1$ </td><td> $9 4 . 0 5 \pm 0 . 1 0$ </td></tr><tr><td>PFS</td><td> $8 . 9 5 \pm 0 . 4 0$ </td><td> $9 5 . 4 3 \pm 0 . 1 3$ </td><td> $9 4 . 3 6 \pm 0 . 1 1$ </td></tr><tr><td>VPC</td><td> $1 5 . 1 2 \pm 0 . 2 3$ </td><td> $9 6 . 5 1 \pm 0 . 1 1$ </td><td> $9 4 . 0 6 \pm 0 . 0 5$ </td></tr><tr><td>RecOE (ours)</td><td> ${ \bf 4 . 3 6 \pm 0 . 7 6 }$ </td><td> $\mathbf { 9 8 . 6 8 \pm 0 . 2 4 }$ </td><td> $9 4 . 6 5 \pm 0 . 0 6$ </td></tr></table>

Table 1: OOD detection results on CIFAR-10 (%). Results are reported as mean ± standard deviation over three trials when available.

Baselines. We compare our proposed RecOE with two main categories of classic and recent strong baselines: (i) Post-hoc methods, including MSP [5], ODIN [42], Mahalanobis [7], Energy [6], ASH [43], PCA [44], KPCA (CoP and CoRP) [11], and Mahalanobis++ [10]; (ii) OE-based methods, including standard OE [13], EnergyOE [6], POEM [15], MixOE [16], DivOE [17], GReg [18], PFS [19] and VPC[20]. Three typical metrics are employed for evaluating detection results: (i) FPR95, the false positive rate of OoD samples at a 95% true positive rate for InD test data; (ii) AUROC, the area under the receiver operating characteristic curve; and (iii) ID-ACC, the classification accuracy on InD test data. Further experimental details are provided in Appendix A.1 and Appendix A.3.

<table><tr><td>Method</td><td>FPR95↓</td><td>AUROC↑</td><td>ID-ACC↑</td></tr><tr><td>Post-hoc methods</td><td></td><td></td><td></td></tr><tr><td>MSP</td><td>75.23</td><td>76.67</td><td>68.72</td></tr><tr><td>ODIN</td><td>67.61</td><td>80.36</td><td>68.72</td></tr><tr><td>Mahalanobis</td><td>83.41</td><td>58.47</td><td>68.72</td></tr><tr><td>Energy</td><td>71.44</td><td>79.52</td><td>68.72</td></tr><tr><td>ASH</td><td>73.51</td><td>77.03</td><td>68.72</td></tr><tr><td>PCA</td><td>84.36</td><td>57.44</td><td>68.72</td></tr><tr><td>KPCA (CoP)</td><td>58.40</td><td>83.86</td><td>68.72</td></tr><tr><td>KPCA (CoRP)</td><td>48.60</td><td>87.53</td><td>68.72</td></tr><tr><td>Mahalanobis++</td><td>49.54</td><td>88.14</td><td>68.72</td></tr><tr><td>OE-based methods</td><td></td><td></td><td></td></tr><tr><td>OE</td><td> $6 1 . 9 4 \pm 0 . 0 3$ </td><td> $8 1 . 5 8 \pm 0 . 0 1$ </td><td> $7 5 . 9 0 \pm 0 . 0 0$ </td></tr><tr><td>EnergyOE</td><td> $6 6 . 9 4 \pm 0 . 0 1$ </td><td> $7 9 . 9 3 \pm 0 . 0 2$ </td><td> $7 4 . 1 9 \pm 0 . 0 8$ </td></tr><tr><td>POEM</td><td> $6 7 . 8 1 \pm 0 . 4 4$ </td><td> $7 9 . 3 6 \pm 0 . 2 4$ </td><td> $7 4 . 2 1 \pm 0 . 0 3$ </td></tr><tr><td>MixOE</td><td> $7 0 . 1 2 \pm 0 . 2 6$ </td><td> $7 8 . 8 5 \pm 0 . 0 5$ </td><td> $7 6 . 1 8 \pm 0 . 0 1$ </td></tr><tr><td>DivOE</td><td> $6 0 . 1 2 \pm 0 . 1 1$ </td><td> $8 1 . 9 6 \pm 0 . 0 0$ </td><td> $7 5 . 7 3 \pm 0 . 0 2$ </td></tr><tr><td>GReg</td><td> $6 4 . 1 1 \pm 4 . 2 7$ </td><td> $8 3 . 4 2 \pm 1 . 4 8$ </td><td> $7 3 . 0 2 \pm 0 . 5 1$ </td></tr><tr><td>PFS</td><td> $6 0 . 1 9 \pm 0 . 2 1$ </td><td> $8 3 . 2 7 \pm 0 . 1 0$ </td><td> $\mathbf { 7 6 . 3 5 \pm 0 . 0 1 }$ </td></tr><tr><td>VPC</td><td>56.50</td><td>84.70</td><td>76.11</td></tr><tr><td>RecOE (ours)</td><td> ${ \bf 3 6 . 3 2 \pm 2 . 0 4 }$ </td><td> $\mathbf { 9 1 . 0 6 \pm 0 . 2 0 }$ </td><td> $7 5 . 6 9 \pm 0 . 0 7$ </td></tr></table>

Table 2: OOD detection results on ImageNet-1K (%). Results are reported as mean ± standard deviation over three trials when available.

## 4.2. Main Comparisons

The main comparisons of our RecOE with representative OoD detection methods on the CIFAR-10 and ImageNet-1K benchmarks are presented in Table 1 and Table 2. The reported results are averaged across multiple OoD test datasets, while the results on individual datasets are provided in Appendix B. In Tables 1 and 2, post-hoc results are checkpoint-specific point estimates, as each setting is evaluated on a single fixed checkpoint without independent repeated trials. OE results are reported as mean ± standard deviation over three trials when available; published results lacking these statistics are retained as point estimates. For VPC [20], we reproduce the CIFAR-10 results and use the published ImageNet-1K results. The published ImageNet-1K setting uses five fine-tuning epochs, compared with four epochs in our experiments, and does not report standard deviations.

As illustrated, RecOE significantly outperforms the compared baselines in both FPR95 and AUROC by a substantial margin, while maintaining competitive InD classification accuracy. This joint strength on OoD robustness and InD performance is challenging for the baseline methods. For instance, our method yields substantial improvements over the very recent VPC [20] on ImageNet with FPR95 reduced by 20.18 percentage points and AUROC boosted by 6.36 percentage points, despite VPC’s readily strong performance with 56.50% on FPR95 and 84.70% on AUROC. Such improvements validate our efectiveness through subspaces optimized for InD and OoD. This result demonstrates great potential of this new perspective of proceeding outlier regularization on feature principal spaces, rather than the output logit space.

To evaluate RecOE under diferent degrees of distributional proximity, we further evaluate the tasks on far-OoD and near-OoD with the ImageNet-1K benchmark and compare with strong baselines, following protocols in [19]. Table 3 shows that RecOE achieves competitive far-OoD performance, with FPR95 of 31.46% and AUROC of 93.60%, although it trails Mahalanobis++ by 3.53 percentage points and 1.18 percentage points, respectively. On near-OoD datasets, RecOE significantly reduces FPR95 by 22.07 percentage points and improves AUROC by 6.21 percentage points upon the strongest VPC. These results demonstrate the particular efectiveness of RecOE in distinguishing OoD samples lying close to InD, which is commonly considered as more challenging settings for OoD detection.

<table><tr><td rowspan="2">Method</td><td colspan="2">Far-OoD</td><td colspan="2">Near-OoD</td></tr><tr><td>FPR95↓</td><td>AUROC↑</td><td>FPR95↓</td><td>AUROC↑</td></tr><tr><td>KPCA (CoRP)</td><td>29.66</td><td>93.75</td><td>67.85</td><td>81.33</td></tr><tr><td>Mahalanobis++</td><td>27.93</td><td>94.78</td><td>71.16</td><td>81.51</td></tr><tr><td>PFS</td><td>52.23</td><td>85.62</td><td>68.16</td><td>80.92</td></tr><tr><td>VPC</td><td>49.75</td><td>87.10</td><td>63.25</td><td>82.30</td></tr><tr><td>RecOE (ours)</td><td>31.46</td><td>93.60</td><td>41.18</td><td>88.51</td></tr></table>

Table 3: OoD detection results on the detection tasks of far-OoD and near-OoD.

Computation Cost. Our method additionally optimizes two basis matrices, $U _ { \mathrm { i n } } ~ \in ~ \mathbb { R } ^ { d \times k }$ and $U _ { \mathrm { a u x } } \in \mathbb { R } ^ { d \times k }$ , for subspace projections. Together, these matrices require storing 2dk coeficients at rank k. We compare the computational cost of RecOE with that of standard OE using WideResNet-40-2

on CIFAR-10, reporting trainable parameter counts and mean fine-tuning time per epoch. As shown in Table 4, RecOE achieves lower FPR95 and higher AUROC, with an 18.4% increase in per-epoch fine-tuning time and a 0.71% increase in trainable parameters. These results demonstrate a fa vorable trade-of between OOD detection performance and the additional training time and parameter overhead introduced by RecOE.
<table><tr><td>Method</td><td>Parameters</td><td>Time</td><td>FPR95↓</td><td>AUROC↑</td></tr><tr><td>OE</td><td>2.244M</td><td> $1 4 . 1 0 \pm 0 . 3 0$ </td><td> $1 3 . 7 6 \pm 0 . 2 3$ </td><td> $9 7 . 5 3 \pm 0 . 0 3$ </td></tr><tr><td>RecOE</td><td>2.260M</td><td> $1 6 . 7 0 \pm 1 . 1 9$ </td><td> $4 . 3 6 \pm 0 . 7 6$ </td><td> $9 8 . 6 8 \pm 0 . 2 4$ </td></tr></table>

Table 4: Training cost and detection performance on CIFAR-10 with WideResNet-40-2. RecOE parameters are counted at the target rank. Time (s/epoch) and detection metrics are mean ± standard deviation.

## 4.3. Ablation Study

To better understand the mechanisms underlying the improved detection performance of RecOE, we ablate individual loss terms on CIFAR-10 and evaluate single-subspace variants on ImageNet-1K. We further evaluate the progressive-rank optimization strategy on both CIFAR-10 and ImageNet-1K to assess its efectiveness in improving OOD detection performance.

Individual loss terms. We conduct a leave-one-out ablation study on the losses regarding optimizing our subspaces, i.e., $\mathcal { L } _ { \mathrm { s u b } } = \mathcal { L } _ { \mathrm { f i t } } + \mathcal { L } _ { \mathrm { s e p } } + \beta \mathcal { L } _ { \mathrm { o r t h o } } ,$ with results reported in Table 5. The results show that removing either ${ \mathcal { L } } _ { \mathrm { f i t } }$ or $\mathcal { L } _ { \mathrm { s e p } }$ leads to a substantial degradation in OoD detection performance, demonstrating that both strategies are essential to our subspace learning framework. Without ${ \mathcal { L } } _ { \mathrm { f i t } }$ , the learned subspaces fail to adequately capture the dominant variations of their corresponding feature distributions. In contrast, removing $\mathcal { L } _ { \mathrm { s e p } }$ prevents the model from explicitly distinguishing each distribution from the opposing subspace, leading to increased overlap between the two learned subspaces. The comparatively smaller performance drop after removing $\mathcal { L } _ { \mathrm { o r t h o } }$ indicates that $\mathcal { L } _ { \mathrm { o r t h o } }$ plays a complementary role by reducing the overlap between the two subspaces.

<table><tr><td colspan="3">Loss Terms</td><td colspan="2">OoD Detection Results</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { f i t } }$ </td><td> $\mathcal { L } _ { \mathrm { s e p } }$ </td><td> $\mathcal { L } _ { \mathrm { o r t h o } }$ </td><td>FPR95↓</td><td>AUROC↑</td></tr><tr><td>一</td><td>√</td><td>√</td><td>14.89</td><td>95.83</td></tr><tr><td>√</td><td>一</td><td>√</td><td>29.19</td><td>94.57</td></tr><tr><td>√</td><td>√</td><td>一</td><td>4.77</td><td>98.40</td></tr><tr><td>√</td><td>√</td><td>√</td><td>4.36</td><td>98.68</td></tr></table>

Table 5: Leave-one-out ablation of the subspace loss terms on CIFAR-10 (%).

Single Subspace. We evaluate the dual-subspace design on ImageNet-1K by varying the subspaces used during training and testing, with all other experimental settings unchanged. We first compare RecOE with the variants trained and tested using a single subspace. As shown in Table 6, RecOE outperformed both variants in FPR95 and AUROC. Compared with the stronger InD-only variant, RecOE reduces FPR95 by 1.86 percentage points and increases AUROC by 0.48 percentage points. The results in this ablation demonstrate the indispensability of our two subspace construction and its optimization, in which the contrast reconstruction between subspaces $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ efectively yields discriminative separation in OoD detection.

<table><tr><td>Training</td><td>Testing</td><td>FPR95↓</td><td>AUROC↑</td></tr><tr><td>InD-only</td><td>InD-only</td><td>38.18</td><td>90.58</td></tr><tr><td>Auxiliary-only</td><td>Auxiliary-only</td><td>76.61</td><td>70.06</td></tr><tr><td>Dual-subspace</td><td>InD-only</td><td>44.24</td><td>88.70</td></tr><tr><td>Dual-subspace</td><td>Auxiliary-only</td><td>39.05</td><td>88.74</td></tr><tr><td>Dual-subspace</td><td>Dual-subspace (RecOE)</td><td>36.32</td><td>91.06</td></tr></table>

Table 6: Subspace ablation on ImageNet-1K (%). Training and testing columns specify the subspaces used at each stage.

To isolate the role of reconstruction contrast at inference, we retain the trained dual-subspace model and use either subspace alone at test time. Both variants perform worse than contrast scoring, indicating that an individual reconstruction error does not fully exploit the information available in the learned subspace pair. By contrasting reconstruction errors between $U _ { \mathrm { i n } }$ and

$U _ { \mathrm { a u x } }$ , RecOE evaluates each sample’s relative compatibility with the two distributions. This comparison supports the efectiveness of jointly using both subspaces at inference for OoD detection.

Progressive-Rank Optimization.. We compare the progressive-rank strategy with a fixed-rank variant that initializes and optimizes all target-rank directions from the beginning. Both configurations use the same final subspace rank. As shown in Table 7, progressive-rank optimization achieved modest improvements in both FPR95 and AUROC on the two benchmarks, indicating that directly optimizing the full-rank subspaces is less efective. By first learning compact subspaces and then gradually introducing additional directions, progressive-rank optimization prioritizes dominant feature variations and reduces the interference caused by poorly estimated directions during early training. These results validate the efectiveness of the proposed progressive-rank pipeline in facilitating stable and discriminative subspace learning.

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-10</td><td colspan="2">ImageNet-1K</td></tr><tr><td>FPR95↓</td><td>AUROC↑</td><td>FPR95↓</td><td>AUROC↑</td></tr><tr><td>RecOE (fixed rank)</td><td>4.51</td><td>98.65</td><td>38.46</td><td>90.72</td></tr><tr><td>RecOE</td><td>4.36</td><td>98.68</td><td>36.32</td><td>91.06</td></tr></table>

Table 7: Ablation study of progressive-rank optimization on CIFAR-10 and ImageNet-1K.

## 4.4. Experiments with an Alternative Auxiliary Dataset

To further evaluate the generalizability of our method, we construct a more challenging setting in which both the InD data and the auxiliary outliers are drawn from ImageNet-1K. Specifically, 100 ImageNet-1K classes are used as InD data, while 800 disjoint classes serve as surrogate OoD data. We adopt the same test OoD datasets as in the setting where ImageNet-1K serves as the InD dataset and report the averaged results across multiple datasets. As shown in Table 8, although RecOE results in a decrease in AU-ROC from 85.78% to 80.78%, it reduces FPR95 from 69.17% to 63.57% and improves InD accuracy from 77.26% to 78.90%. These results indicate that the learned subspaces retain their discriminative capability in the challenging same-source setting, despite the trade-of in AUROC, where the InD and surrogate OoD samples are drawn from the same large-scale dataset. They further suggest that the efectiveness of RecOE does not depend solely on access to a diverse auxiliary dataset such as ImageNet-21K\_p [37].

<table><tr><td>Methods</td><td>FPR95↓</td><td>AUROC↑</td><td>ID-ACC↑</td></tr><tr><td>OE</td><td>69.17</td><td>85.78</td><td>77.26</td></tr><tr><td>RecOE</td><td>63.57</td><td>80.78</td><td>78.90</td></tr></table>

Table 8: OoD detection performance using 100 ImageNet-1K classes as InD data and 800 ImageNet-1K classes as surrogate OoD data.

## 4.5. Eficient variants of RecOE without Backbone Finetuning

In OE-based methods, finetuning is applied to the entire network $( f ( \cdot ) )$ for updating the parameters that facilitate OoD detection. In the optimization with RecOE given in Eq. (9) in the main paper, we introduce two matrices $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ in addition to the pretrained $f ( \cdot )$ . Because these learned subspaces play a central role in separating InD and OoD features, we further investigate lightweight settings in which the backbone $g ( \cdot )$ is frozen and only the subspace matrices, either alone or together with the classification head $h ,$ are optimized. For simplicity, we collectively denote the two subspace matrices by U. To further simplify the optimization, these variants employ the fixed-rank subspace learning strategy. Specifically, we consider the following two lightweight variants of RecOE:

• RecOE (U): The subspace loss $\mathcal { L } _ { \mathrm { s u b } }$ is used to only update $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ , with the entire pretrained f(·) frozen.

• RecOE (h, U): The linear head h is optimized using $\mathcal { L } _ { \mathrm { C E } }$ , whereas the two basis matrices $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ are optimized solely through $\mathcal { L } _ { \mathrm { s u b } }$ The backbone $g ( \cdot )$ remains frozen throughout training.

Table 9 reports the results obtained with ResNet50 on ImageNet-1K. Both lightweight variants substantially reduce the number of trainable parameters and training time while maintaining competitive detection performance. Specially, RecOE (U) uses 92.4% fewer trainable parameters and requires 89.7% less training time per epoch than the full RecOE. Despite the slight performance drop, the lightweight variants of our proposed RecOE still achieves advantageous results; for instance, simply updating U leads to a lower FPR95 (46.78%) than the competitive post-hoc baseline Mahalanobis++ (49.54%)

and OE-based baseline VPC (56.50%), and also attains higher AUROC than both. Nevertheless, freezing the backbone $g ( \cdot )$ results in a drop in InD classification accuracy. Despite this trade-of, such lightweight variants further demonstrate the flexibility and practicality of RecOE, enabling its deployment in resource-constrained or time-sensitive scenarios.
<table><tr><td>Updated parameters</td><td>Time (s/epoch)</td><td>Trainable parameters</td><td></td><td>FPR95↓ AUROC↑ ID-ACC↑</td><td></td></tr><tr><td>RecOE (U)</td><td>449</td><td>2.10M</td><td>46.78</td><td>88.18</td><td>68.72</td></tr><tr><td>RecOE (h, U)</td><td>453</td><td>4.15M</td><td>44.97</td><td>88.55</td><td>71.09</td></tr><tr><td>RecOE</td><td>4351</td><td>27.66M</td><td>36.32</td><td>91.06</td><td>75.69</td></tr></table>

Table 9: Comparisons among variants of RecOE on ImageNet-1K. The trainable parameters and training time (s) per epoch are recorded, together with their detection results and ID-ACC.

## 4.6. Hyperparameter Sensitivity

Using CIFAR-10 as InD, we examine the sensitivity of RecOE to the subspace-loss weight $\gamma ,$ orthogonality regularization weight $\beta _ { ; }$ , separation margin $m ,$ and subspace dimension $k .$ We use the validation set from [13] and vary one hyperparameter at a time, keeping the others at their default values. Figure 4 shows the resulting AUROC.

• Subspace weight $\gamma .$ This coeficient governs the trade-of between the classification task on InD data $\left( \mathcal { L } _ { \mathrm { C E } } \right)$ and the subspace learning for the OoD detection task $( \mathcal { L } _ { \mathrm { s u b } } )$ . Experiments suggest that an excessively large $\gamma$ may overemphasize the subspace-based objective and interfere with learning semantic representations from InD data.

• Orthogonal regularization weight $\beta .$ The hyperparameter $\beta$ is related to penalizing correlation between the two subspaces, ensuring that the correspondingly learned two sets of bases capture complementary variations rather than redundant information. By reducing subspace overlap, it balances structural separation with the preservation of semantically meaningful representations.

![](images/ef5cf0bc9dc3f0b3996a4f19f9be166275da56b0385e9dd21dc8882ecf1c3b31.jpg)  
Figure 4: Hyperparameter sensitivity analysis. From left to right: subspace-loss weight $\gamma ,$ orthogonality regularization weight $\beta ,$ separation margin m, and subspace dimension $k .$

• Subspace separation margin m: The margin m relates to the separation tolerance by penalizing samples whose reconstruction error relative to the opponent subspace falls below the threshold. A larger m encourages a more distinct gap by pushing features apart across domains, whereas an excessively large margin can impose an overly restrictive constraint.

• Subspace dimension k: The dimension k, i.e., the rank of the projection space, determines the granularity of the captured structural information. Empirical evidence suggests that a relatively low dimensionality is more efective; it successfully distills the essential characteristics of InD data while efectively filtering out OoD noise that lacks such low-rank consistency.

## 5. Conclusion and Discussion

In this work, we move beyond the mainstream logit-output space and introduce a feature subspace learning perspective into the OE-based framework. Unlike previous feature-based OE methods building up the model for features with additionally prescribed structures or modules, RecOE learns two distribution-specific subspaces, $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } } .$ , to capture the dominant variations of InD and OoD features while explicitly separating their distributional structures. In inference, the contrast between the reconstruction errors induced by the two subspaces provides an efective criterion for OoD detection. Extensive experiments across multiple benchmarks and settings further validate the efectiveness and generalizability of our method. Overall, our method ofers new insights into feature subspace learning for OE-based OoD detection, achieving substantial performance gains with only minor computational overhead.

Several aspects of the current framework warrant further investigation. RecOE uses a shared InD subspace across classes, while its detection performance has not been examined at the class level. Future work could assess class-level diferences and explore whether class-aware reconstruction scores improve detection consistency. In addition, the fitting and separation objectives provide a geometric basis for reconstruction contrast, but generalization guarantees for unseen OoD distributions remain to be established. Further theoretical analysis could clarify how reconstruction margins, subspace over lap, and auxiliary-outlier coverage influence detection error.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant No 62503051) and the Beijing Natural Science Foundation (L2607017).

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used DeepSeek for language refinement and assistance with research code implementation. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Supplementary Material

This supplementary material provides additional implementation details, extended experimental results and further discussions complementary to the main manuscript.

## Appendix A. Implementation Details

## Appendix A.1. Training and Hyperparameter Settings

Following the setting commonly adopted in OE-based methods [13, 17], we finetune pretrained backbones rather than training the complete networks from scratch. For the CIFAR experiments, we employ a WideResNet-40- 2 [40] backbone initialized from a checkpoint obtained by training on CIFAR-10 for 100 epochs on the standard classification task. For the ImageNet-1K experiments, we utilize a ResNet-50 [41] architecture initialized with standard PyTorch pretrained weights. During finetuning, the backbone parameters are optimized using stochastic gradient descent (SGD) with a momentum of 0.9 and a weight decay of $5 \times 1 0 ^ { - 4 }$ . The learning rate of SGD is updated at every training iteration using cosine annealing [45] and the initial learning rates are set to $5 \times 1 0 ^ { - 3 }$ for CIFAR-10 and $1 \times 1 0 ^ { - 4 }$ for ImageNet-1K. The two orthonormal subspace bases are optimized using Riemannian Adam [23, 24]. Optimization details for $U _ { \mathrm { i n } }$ and $U _ { \mathrm { a u x } }$ are provided in Appendix A.2.

We use mini-batch sizes of 128 and 64 and finetune the networks for 10 and 4 epochs on CIFAR-10 and ImageNet-1K, respectively. Unless otherwise specified, the target rank of each subspace is set to 64 for CIFAR-10 and 512 for ImageNet-1K. For CIFAR-10, we set the separation margin to $m =$ 0.5, the orthogonality regularization weight to $\beta = 1 0 ^ { - 3 }$ , and the subspaceloss weight to $\gamma \ : = \ : 0 . 5$ . For ImageNet-1K, the corresponding values are set to $m = 5 , \ \beta = 1 0 ^ { - 3 }$ , and $\gamma = 0 . 5$ . Following prior work [13, 17], we select hyperparameters based on their overall performance across all three evaluation metrics on the validation set.

## Appendix A.2. Progressive-Rank Subspace Optimization

The InD and auxiliary subspaces follow the same active-rank schedule, and their bases are initialized and expanded independently. We describe the procedure using a generic basis $U \in \mathbb { R } ^ { d \times r }$ satisfying $U ^ { \top } U = I _ { r }$ , where d is the feature dimension and r is the current active rank.

Initialization. Given a target rank $k \ \leq \ d ,$ the basis starts at rank $r ^ { ( 0 ) } =$ max $\{ 1 , \lceil k / 2 \rceil \}$ . We sample a Gaussian matrix $G \in \mathbb { R } ^ { d \times r ^ { ( 0 ) } }$ with independent standard normal entries and initialize

$$
U ^ { ( 0 ) } = \operatorname { q f } ( G ) ,\tag{A.1}
$$

where $\operatorname { q f } ( \cdot )$ returns the orthonormal factor of a reduced QR decomposition.

Subspace expansion. The active rank increases every K optimization steps according to Eq. (10). For example, a target rank of $k = 6 4$ gives the sequence

$$
3 2  4 8  5 6  6 0  6 2  6 3  6 4 .
$$

When the rank increases from r to $r ^ { + } = r + \Delta r$ , where $\Delta r$ is the rank increment, we retain the current basis and sample $G \in \mathbb { R } ^ { d \times \Delta r }$ with independent standard normal entries. We project these new directions onto the orthogonal complement of the current subspace:

$$
\widetilde { G } = ( I _ { d } - U U ^ { \top } ) G .\tag{A.2}
$$

We then orthonormalize the projected directions and construct the expanded basis:

$$
{ \cal Q } _ { \mathrm { n e w } } = \operatorname { q f } ( \widetilde { \cal G } ) , \qquad { \cal U } ^ { + } = \operatorname { q f } \left( [ { \cal U } , { \cal Q } _ { \mathrm { n e w } } ] \right) .\tag{A.3}
$$

The expanded basis satisfies $( U ^ { + } ) ^ { \top } U ^ { + } = I _ { r ^ { + } }$ + and contains the previously learned subspace. All directions remain trainable after expansion, and the columns are not ordered by explained variance.

Optimization. Each rank expansion changes the dimensions of the subspace parameters. We replace each Stiefel parameter with the expanded basis and reinitialize Riemannian Adam for both bases. Optimization proceeds in stages, each with a fixed active rank r. The subspace learning rate follows a cosine schedule within each stage, with maximum value

$$
\eta _ { \mathrm { m a x } } = \eta _ { U } \left( \frac { r ^ { ( 0 ) } } { r } \right) ^ { \alpha } ,\tag{A.4}
$$

where $\eta _ { U }$ is the initial maximum learning rate and α controls its decay as the rank increases. At step τ of the current stage, the learning rate is

$$
\eta ( \tau ) = \eta _ { \mathrm { m i n } } + \frac { 1 } { 2 } \left( \eta _ { \mathrm { m a x } } - \eta _ { \mathrm { m i n } } \right) \left[ 1 + \cos \left( \frac { \pi \operatorname* { m i n } \{ \tau , K \} } { K } \right) \right] , \qquad \tau \geq 0 ,\tag{A.5}
$$

where $\tau$ counts the optimization steps since the stage began and $\eta _ { \mathrm { m i n } }$ is the minimum learning rate. The counter $\tau$ starts at zero. While $r \ < \ k$ , each stage performs K optimization steps with $\tau = 0 , \dots , K - 1$ before the next rank expansion. Each expansion starts a new stage, resets $\tau$ to zero, and updates $\eta _ { \mathrm { m a x } }$ using the new active rank. Once $r = k .$ , no further expansion occurs, and the counter continues to increase. The learning rate follows the same schedule and remains at $\eta _ { \mathrm { m i n } }$ for $\tau \geq K$ . Progressive-rank subspace optimization is summarized in Algorithm 1.

Algorithm 1 Progressive-Rank Subspace Optimization   
Require: Subspace loss $\mathcal { L } _ { \mathrm { s u b } }$ ; feature dimension $d ;$ target rank $k \leq d ;$ expansion   
interval $K ;$ total optimization steps $T ;$ learning-rate parameters $\eta _ { U } , \eta _ { \mathrm { m i n } } , \alpha$   
Ensure: Optimized bases $U _ { \mathrm { i n } } , U _ { \mathrm { a u x } } .$   
$\iota ; \ r ^ { ( 0 ) } \gets \operatorname* { m a x } \{ 1 , \lceil k / 2 \rceil \} ; \ r \gets r ^ { ( 0 ) } ; \ \tau \gets 0$   
2: for $j \in \{ \mathrm { i n , a u x } \}$ do   
3: Sample $G _ { j } \in \mathbb { R } ^ { d \times r }$ with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries   
4: $U _ { j }  \operatorname { q f } ( G _ { j } )$   
5: end for   
6: Initialize Riemannian Adam for both bases   
7: for $t = 0 , \ldots , T - 1$ do   
8: if $\tau = K$ and $r < k$ then   
9: $\Delta r \gets \operatorname* { m i n } \{ k - r , \operatorname* { m a x } \{ 1 , \lceil ( k - r ) / 2 \rceil \} \}$   
10: for j ∈ {in, aux} do   
11: Sample $G _ { j } \in \mathbb { R } ^ { d \times \Delta r }$ with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries   
12: $\widetilde { G } _ { j }  G _ { j } - U _ { j } ( U _ { j } ^ { \top } G _ { j } )$   
13: $U _ { j }  \mathrm { q f } ( [ U _ { j } , \mathrm { q f } ( \widetilde { G } _ { j } ) ] )$   
14: end for   
15: $r  r + \Delta r ; \tau  0$   
16: Replace the Stiefel parameters and reinitialize Riemannian Adam   
17: end if   
18: η<sub>max</sub> $ \eta _ { U } ( r ^ { ( 0 ) } / r ) ^ { \alpha }$   
19: $\eta ( \tau ) \gets \eta _ { \mathrm { m i n } } + \frac { 1 } { 2 } \big ( \eta _ { \mathrm { m a x } } - \eta _ { \mathrm { m i n } } \big ) \big [ 1 + \cos ( \pi \operatorname* { m i n } \{ \tau , K \} / K ) \big ]$   
20: Compute gradients of $\mathcal { L } _ { \mathrm { s u b } }$ with respect to both bases   
21: Update both bases by Riemannian Adam with learning rate $\eta ( \tau )$   
22: $\tau  \tau + 1$   
23: end for

<table><tr><td>Setting</td><td>CIFAR-10</td><td>ImageNet-1K</td></tr><tr><td>Target rank k</td><td>64</td><td>512</td></tr><tr><td>Initial rank  $r ^ { ( 0 ) }$ </td><td>32</td><td>256</td></tr><tr><td>Expansion interval K</td><td>500</td><td>1000</td></tr><tr><td>Initial maximum learning rate ηu</td><td>5</td><td>10</td></tr><tr><td>Minimum learning rate ηmin</td><td>10-3</td><td>10⁻4</td></tr><tr><td>Rank-dependent decay exponent α</td><td>0.5</td><td>0.5</td></tr></table>

Table A.1: Progressive-rank optimization settings. The expansion interval is measured in optimization steps, and the settings apply to each subspace basis.

Parameter settings. Table A.1 lists the subspace optimization settings. The backbone follows its original optimization schedule.

## Appendix A.3. Computing Environment

All experiments are conducted on a workstation running Ubuntu 24.04, equipped with an Intel Core i9-14900K CPU and two NVIDIA GeForce RTX 4090 GPUs. The CIFAR-10 experiments use a single GPU, whereas the ImageNet-1K experiments use both GPUs. The installed NVIDIA driver is version 580.142. The experiments are implemented using Python 3.7 and PyTorch 1.12 built against CUDA 11.6. Riemannian optimization of the subspace parameters is implemented using Geoopt [24].

## Appendix B. Detailed Results on Individual Datasets

In the main paper, we report the averaged OoD detection performance and standard deviations across multiple datasets for concise comparison. To provide a more detailed analysis of the efectiveness of RecOE under diverse OoD scenarios, we further report results for each OoD test dataset in this supplementary material. The detailed results on the CIFAR-10 and ImageNet-1K benchmarks are presented in Tables B.1 and B.2, respectively.

Following the experimental settings of DivOE [17], we reproduce PCA [44], KPCA (CoP and CoRP) [9], Mahalanobis++ [10], GReg [18], and PFS [19]. For CIFAR-10, we additionally reproduce VPC [20]; results for the remaining baselines are taken from DivOE. For ImageNet-1K, Table B.2 reports our reproduced baseline results and the published VPC results.

These detailed results highlight the substantial performance gains of our proposed framework over representative baselines across varied OoD datasets.

<table><tr><td rowspan="3">Method</td><td colspan="10">OoD dataset</td></tr><tr><td colspan="2">Textures</td><td colspan="2">Places365</td><td colspan="2">iSUN</td><td>LSUN</td><td>Resize</td><td>LSUN</td><td>Crop</td></tr><tr><td>FPR95↓ AUROC↑ FPR95↓ AUROC↑ FPR95↓ AUROC↑ FPR95↓ AUROC↑ FPR95↓ AUROC↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MSP</td><td>59.27</td><td>88.50</td><td>60.77</td><td>88.13</td><td>56.79</td><td>89.53</td><td>53.60</td><td>91.02</td><td>30.55</td><td>95.68</td></tr><tr><td>ODIN</td><td>54.84</td><td>80.28</td><td>48.89</td><td>85.66</td><td>28.68</td><td>93.85</td><td>23.17</td><td>95.05</td><td>9.96</td><td>97.90</td></tr><tr><td>Mahalanobis</td><td>16.99</td><td>96.69</td><td>69.96</td><td>82.54</td><td>31.62</td><td>94.44</td><td>32.41</td><td>94.70</td><td>30.50</td><td>95.06</td></tr><tr><td>Energy</td><td>53.17</td><td>85.40</td><td>40.22</td><td>89.91</td><td>34.60</td><td>92.44</td><td>28.48</td><td>93.85</td><td>8.12</td><td>98.35</td></tr><tr><td>PCA</td><td>57.62</td><td>83.93</td><td>91.54</td><td>61.23</td><td>90.21</td><td>66.36</td><td>89.89</td><td>66.33</td><td>19.95</td><td>96.27</td></tr><tr><td>KPCA (CoP)</td><td>38.30</td><td>91.86</td><td>55.98</td><td>85.83</td><td>42.83</td><td>91.23</td><td>39.10</td><td>92.37</td><td>24.89</td><td>95.14</td></tr><tr><td>KPCA (CoRP)</td><td>36.60</td><td>93.49</td><td>41.34</td><td>91.05</td><td>31.52</td><td>93.64</td><td>25.64</td><td>95.02</td><td>23.98</td><td>95.80</td></tr><tr><td>Mahalanobis++</td><td>15.73</td><td>96.99</td><td>45.33</td><td>89.19</td><td>18.63</td><td>96.67</td><td>16.61</td><td>96.99</td><td>1.52</td><td>99.69</td></tr><tr><td>OE</td><td>21.81</td><td>96.15</td><td>44.86</td><td>91.77</td><td>0.00</td><td>100.00</td><td>0.00</td><td>100.00</td><td>0.99</td><td>99.85</td></tr><tr><td>EnergyOE</td><td>33.59</td><td>85.47</td><td>58.39</td><td>68.32</td><td>0.00</td><td>100.00</td><td>0.00</td><td>100.00</td><td>0.31</td><td>99.83</td></tr><tr><td>POEM</td><td>34.73</td><td>85.28</td><td>57.82</td><td>67.93</td><td>0.00</td><td>100.00</td><td>0.00</td><td>100.00</td><td>0.43</td><td>99.74</td></tr><tr><td>MixOE</td><td>22.23</td><td>95.74</td><td>39.45</td><td>92.86</td><td>0.26</td><td>99.92</td><td>0.27</td><td>99.92</td><td>4.40</td><td>99.21</td></tr><tr><td>DivOE</td><td>12.28</td><td>97.73</td><td>44.97</td><td>91.51</td><td>0.07</td><td>99.98</td><td>0.02</td><td>99.99</td><td>3.03</td><td>99.47</td></tr><tr><td>GReg</td><td>22.28</td><td>93.96</td><td>41.18</td><td>91.21</td><td>0.00</td><td>100.00</td><td>0.00</td><td>100.00</td><td>0.00</td><td>100.00</td></tr><tr><td>PFS</td><td>14.93</td><td>91.74</td><td>21.20</td><td>89.16</td><td>3.72</td><td>98.44</td><td>4.50</td><td>98.11</td><td>0.38</td><td>99.68</td></tr><tr><td>VPC</td><td>38.30</td><td>91.22</td><td>29.17</td><td>94.16</td><td>4.10</td><td>98.50</td><td>3.53</td><td>99.00</td><td>0.52</td><td>99.70</td></tr><tr><td>RecOE (ours)</td><td>11.32</td><td>96.67</td><td>8.27</td><td>97.73</td><td>0.78</td><td>99.68</td><td>0.62</td><td>99.72</td><td>0.82</td><td>99.60</td></tr></table>

Table B.1: OoD detection performance on five OoD test datasets with CIFAR-10 as the InD data.

<table><tr><td rowspan="3">Method</td><td colspan="8">OoD dataset</td></tr><tr><td colspan="2">iNaturalist</td><td colspan="2">SUN</td><td colspan="2">Places</td><td colspan="2">Textures</td></tr><tr><td>FPR95↓ AUROC↑ FPR95↓ AUROC↑ FPR95↓ AUROC↑ FPR95↓ AUROC↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PCA</td><td>95.44</td><td>51.98</td><td>97.96</td><td>44.83</td><td>97.98</td><td>45.15</td><td>46.05</td><td>87.79</td></tr><tr><td>KPCA (CoP)</td><td>67.29</td><td>83.42</td><td>75.52</td><td>80.03</td><td>82.49</td><td>73.67</td><td>8.31</td><td>98.32</td></tr><tr><td>KPCA (CoRP)</td><td>50.23</td><td>89.33</td><td>62.45</td><td>83.77</td><td>72.61</td><td>78.86</td><td>9.10</td><td>98.17</td></tr><tr><td>Mahalanobis++</td><td>50.15</td><td>90.74</td><td>66.23</td><td>84.13</td><td>76.08</td><td>78.89</td><td>5.71</td><td>98.82</td></tr><tr><td>GReg</td><td>50.42</td><td>91.61</td><td>71.38</td><td>81.45</td><td>73.97</td><td>79.59</td><td>60.67</td><td>81.01</td></tr><tr><td>PFS</td><td>44.25</td><td>89.76</td><td>64.07</td><td>82.11</td><td>72.24</td><td>79.73</td><td>60.21</td><td>81.49</td></tr><tr><td>VPC</td><td>43.50</td><td>91.20</td><td>60.20</td><td>83.10</td><td>66.30</td><td>81.50</td><td>56.00</td><td>83.00</td></tr><tr><td>RecOE (ours)</td><td>7.85</td><td>98.39</td><td>30.67</td><td>91.29</td><td>51.69</td><td>85.74</td><td>55.08</td><td>88.80</td></tr></table>

Table B.2: OoD detection performance on four OoD test datasets with ImageNet-1K as the InD data. VPC results are taken from the original publication [20]; the remaining results are obtained from our experiments.

Notably, RecOE significantly reduces the FPR95 on complex OoD datasets where the compared methods typically struggle, such as Textures[32] and Places365[33] for CIFAR-10, and iNaturalist[38] for ImageNet-1K. While OEbased frameworks often exhibit slight performance trade-ofs across diferent datasets, our approach achieves a better overall balance. For example, although RecOE performs slightly worse than some OE baselines on iSUN and LSUN\_Resize, where performance is already near saturation, it achieves better overall performance across the full set of OoD datasets.

## References

[1] I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016.

[2] J. Gu, Z. Wang, J. Kuen, L. Ma, A. Shahroudy, B. Shuai, T. Liu, X. Wang, G. Wang, J. Cai, T. Chen, Recent advances in convolutional neural networks, Pattern Recognition 77 (2018) 354–377. doi:10.1016/j.patcog.2017.10.013.

[3] A. Nguyen, J. Yosinski, J. Clune, Deep neural networks are easily fooled: High confidence predictions for unrecognizable images, in: CVPR, 2015, pp. 427–436.

[4] T. Schlegl, P. Seeböck, S. M. Waldstein, U. Schmidt-Erfurth, G. Langs, Unsupervised anomaly detection with generative adversarial networks to guide marker discovery, in: International Conference on Information Processing in Medical Imaging, Vol. 10265, 2017, pp. 146–157.

[5] D. Hendrycks, K. Gimpel, A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: ICLR, 2017.

[6] W. Liu, X. Wang, J. Owens, Y. Li, Energy-based out-of-distribution detection, in: NeurIPS, Vol. 33, 2020, pp. 21464–21475.

[7] K. Lee, K. Lee, H. Lee, J. Shin, A simple unified framework for detecting out-of-distribution samples and adversarial attacks, in: NeurIPS, 2018, pp. 7167–7177.

[8] Y. Sun, Y. Ming, X. Zhu, Y. Li, Out-of-distribution detection with deep nearest neighbors, in: ICML, Vol. 162, PMLR, 2022, pp. 20827–20840.

[9] K. Fang, Q. Tao, K. Lv, M. He, X. Huang, J. Yang, Kernel PCA for out-of-distribution detection, in: NeurIPS, Vol. 37, 2024, pp. 134317– 134344.

[10] M. Müller, M. Hein, Mahalanobis++: Improving OOD detection via feature normalization, in: ICML, PMLR, 2025, pp. 45151–45184.

[11] K. Fang, Q. Tao, M. He, K. Lv, R. Yang, H. Hu, X. Huang, J. Yang, L. Cao, Kernel PCA for out-of-distribution detection: Non-linear kernel selection and approximation, IEEE Transactions on Pattern Analysis and Machine Intelligence (2026) 1–16doi:10.1109/TPAMI.2026.3705778.

[12] I. Sevillano-García, J. Luengo, F. Herrera, STOOD-X: Explainable out-of-distribution detection via nonparametric statistical testing on large-scale datasets, Pattern Recognition 177 (2026) 113254. doi:10.1016/j.patcog.2026.113254.

[13] D. Hendrycks, M. Mazeika, T. Dietterich, Deep anomaly detection with outlier exposure, in: ICLR, 2019.

[14] J. Chen, Y. Li, X. Wu, Y. Liang, S. Jha, ATOM: Robustifying outof-distribution detection using outlier mining, in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases, Springer, 2021, pp. 430–445.

[15] Y. Ming, Y. Fan, Y. Li, POEM: Out-of-distribution detection with posterior sampling, in: ICML, PMLR, 2022, pp. 15650–15665.

[16] J. Zhang, N. Inkawhich, R. Linderman, Y. Chen, H. Li, Mixture outlier exposure: Towards out-of-distribution detection in fine-grained environments, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 5531–5540.

[17] J. Zhu, Y. Geng, J. Yao, T. Liu, G. Niu, M. Sugiyama, B. Han, Diversified outlier exposure for out-of-distribution detection via informative extrapolation, in: NeurIPS, Vol. 36, 2023, pp. 22702–22734.

[18] S. Sharifi, T. Entesari, B. Safaei, V. M. Patel, M. Fazlyab, Gradientregularized out-of-distribution detection, in: ECCV, Springer, 2024, pp. 459–478.

[19] Y. Wu, R. Yu, X. Cheng, Z. He, X. Huang, Pursuing feature separation based on neural collapse for out-of-distribution detection, in: ICLR, 2025.

[20] K. Xia, X. Yue, Z. Wei, Y. Peng, Z. Li, J. Zhu, J. Shi, P. Xu, Let OOD feature exploring vast predefined classifiers, in: ICLR, 2026.

[21] K. Pearson, LIII. on lines and planes of closest fit to systems of points in space, The London, Edinburgh, and Dublin philosophical magazine and journal of science 2 (11) (1901) 559–572.

[22] A. Edelman, T. A. Arias, S. T. Smith, The geometry of algorithms with orthogonality constraints, SIAM Journal on Matrix Analysis and Applications 20 (2) (1998) 303–353.

[23] G. Bécigneul, O.-E. Ganea, Riemannian adaptive optimization methods, in: ICLR, 2019.

[24] M. Kochurov, R. Karimov, S. Kozlukov, Geoopt: Riemannian optimization in PyTorch (2020). arXiv:2005.02819.

[25] X. Du, Y. Sun, J. Zhu, Y. Li, Dream the impossible: Outlier imagination with difusion models, in: NeurIPS, Vol. 36, 2023, pp. 60878–60901.

[26] H. Choi, H. Jeong, J. Y. Choi, Balanced energy regularization loss for out-of-distribution detection, in: CVPR, 2023, pp. 15691–15700.

[27] H. Wang, Z. Li, L. Feng, W. Zhang, ViM: Out-of-distribution with virtual-logit matching, in: CVPR, 2022, pp. 4921–4930.

[28] Y. Chen, Y. Lin, R. Xu, P. A. Vela, WDiscOOD: Out-of-distribution detection via whitened linear discriminant analysis, in: ICCV, 2023, pp. 5298–5307.

[29] Q. Wang, J. Ye, F. Liu, Q. Dai, M. Kalander, T. Liu, J. Hao, B. Han, Out-of-distribution detection with implicit outlier transformation, in: ICLR, 2023.

[30] A. Krizhevsky, Learning multiple layers of features from tiny images, Tech. rep., University of Toronto (2009).

[31] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, ImageNet: A large-scale hierarchical image database, in: CVPR, IEEE, 2009, pp. 248–255.

[32] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi, Describing textures in the wild, in: CVPR, 2014, pp. 3606–3613.

[33] B. Zhou, A. Khosla, A. Lapedriza, A. Torralba, A. Oliva, Places: An image database for deep scene understanding (2016). arXiv:1610.02055.

[34] P. Xu, K. A. Ehinger, Y. Zhang, A. Finkelstein, S. R. Kulkarni, J. Xiao, Turkergaze: Crowdsourcing saliency with webcam based eye tracking (2015). arXiv:1504.06755.

[35] F. Yu, A. Sef, Y. Zhang, S. Song, T. Funkhouser, J. Xiao, LSUN: Construction of a large-scale image dataset using deep learning with humans in the loop (2016). arXiv:1506.03365.

[36] Y. Le, X. Yang, et al., Tiny ImageNet visual recognition challenge, CS 231N 7 (7) (2015) 3.

[37] T. Ridnik, E. Ben-Baruch, A. Noy, L. Zelnik-Manor, ImageNet-21K pretraining for the masses (2021). arXiv:2104.10972.

[38] G. Van Horn, O. Mac Aodha, Y. Song, Y. Cui, C. Sun, A. Shepard, H. Adam, P. Perona, S. Belongie, The iNaturalist species classification and detection dataset, in: CVPR, 2018, pp. 8769–8778.

[39] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, A. Torralba, SUN database: Large-scale scene recognition from abbey to zoo, in: CVPR, IEEE, 2010, pp. 3485–3492.

[40] S. Zagoruyko, N. Komodakis, Wide residual networks (2017). arXiv:1605.07146.

[41] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: CVPR, 2016, pp. 770–778.

[42] S. Liang, Y. Li, R. Srikant, Enhancing the reliability of out-ofdistribution image detection in neural networks, in: ICLR, 2018.

[43] A. Djurisic, N. Bozanic, A. Ashok, R. Liu, Extremely simple activation shaping for out-of-distribution detection, in: ICLR, 2023.

[44] X. Guan, Z. Liu, W.-S. Zheng, Y. Zhou, R. Wang, Revisit PCA-based technique for out-of-distribution detection, in: ICCV, 2023, pp. 19431– 19439.

[45] I. Loshchilov, F. Hutter, SGDR: Stochastic gradient descent with warm restarts, in: ICLR, 2017.