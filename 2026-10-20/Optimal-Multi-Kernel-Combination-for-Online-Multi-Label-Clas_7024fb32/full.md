# Optimal Multi-Kernel Combination for Online Multi-Label Classification

Tingting Zhai<sup>a,∗</sup>, Wei Liu<sup>a</sup>, Renjian Yu<sup>a</sup>

<sup>a</sup>College of Information and Artificial Intelligence, Yangzhou University, Yangzhou, China, 225127

## Abstract

Multi-label classification is particularly challenging in streaming scenarios, where data arrives sequentially. Online kernel methods ofer a promising solution. However, prevailing methods rely on a single preselected kernel, making them prone to suboptimal choices since the global data distribution is unknown in advance in online settings. Although the recent method, Online Kernel Selection for Online Multi-label Classification (OKS-OMC), enables data-driven kernel selection, it fails to fully harness the complementary information from multiple kernels and encounters the curse of kernelization. To address these limitations, this paper introduces a novel online multikernel combination method for large-scale multi-label classification. First, an eficient and robust online single-kernel multi-label method is designed. Second, learning an optimal convex combination of multiple multi-label kernel classifiers is formalized as a structural risk minimization problem. An improved alternating optimization procedure is developed, in which the combination coeficients are updated by directly minimizing the objective function, while each kernel classifier is updated by minimizing a surrogate upper bound. This procedure decouples the learning of kernel classifiers from combination coeficients, allowing independent classifier updates via our single-kernel method and accelerating convergence to high-quality solutions. Our multi-kernel method is proven to asymptotically converge to the optimal fixed convex combination of the best fixed single-kernel models chosen in hindsight. Ablation studies confirm that our method’s components outperform their counterparts in OKS-

OMC. Comprehensive evaluations across 11 datasets and seven metrics demonstrate our method’s overall superiority over existing online multi-label methods. The source code of our method is available at https://github.com/LUCKY-ting/OMKC. Keywords: online kernel methods, large-scale multi-label classification, data-driven kernel selection, multi-kernel learning, online learning

## 1. Introduction

Multi-label classification is a crucial and active subfield in machine learning, with broad applications spanning text categorization [1, 2], image annotation [3, 4], recommendation systems [5, 6], and bioinformatics [7, 8]. Unlike traditional single-label problems, where each instance is assigned to exactly one category, multi-label classification requires models to simultaneously assign multiple relevant labels to each instance [9].

Real-world applications now routinely generate massive multi-label data streams. On platforms such as Twitter and Sina Weibo, a continuous flow of posts—each associated with multiple labels —appears every second. Such streaming environments bring new challenges for classification. First, it is infeasible to retrain ofline models from scratch whenever new data arrives. Second, adapting ofline methods to streaming data is far from straightforward.

Online learning [10, 11] presents a promising solution for streaming data. This paradigm continuously updates a model with incoming data, eliminating the need for full retraining. By integrating kernel methods [12], online methods can capture complex non-linear patterns, making them well-suited for the demands of multi-label stream classification.

Existing online kernel methods for multi-label classification fall into two categories: single-kernel and multi-kernel methods. Single-kernel methods rely on a preselected kernel function to implicitly map the input data into a high-dimensional feature space, where examples become approximately linearly separable and a linear classification algorithm can be applied. Within this category, the Passive Aggressive Multi-Label (PAML) family of algorithms [11] has been introduced, extending the margin-based learning framework of SVMs to multi-label classification with the objective of maximizing classification margins. Further, the First-order Adaptive Label Thresholding (FALT) and First-order Fixed Label Thresholding (FFLT) methods [13] are proposed. These algorithms introduce distinct label thresholding strategies and reformulate the large-margin objective as an unconstrained problem, enabling joint optimization of the threshold predictor and all label scoring predictors via Online Gradient Descent (OGD). In [14], FALT is further enhanced by optimizing the distribution of multi-label classification margins. The main limitation of single-kernel methods lie in that they rely on task-specific prior knowledge or an intensive cross-validation process to preselect a kernel function. However, in online learning scenarios where the global data structure is unknown in advance, this practice is vulnerable to suboptimal kernel selection.

In contrast, multi-kernel methods adopt a data-driven approach to select appropriate kernel functions or combine diferent kernel classifiers during the online learning process. Currently, most existing online multi-kernel methods are designed for singlelabel classification tasks. These methods can be divided into the following three categories:

• Online kernel selection methods, which dynamically choose a kernel from a predefined set during the learning process. The approach in [15] uses an exponentially weighted average to assign weights to candidate kernel classifiers, samples a kernel based on these weights, and updates the corresponding classifier with OGD. The approach was later enhanced in [16], where kernel selection is formulated as a multi-armed bandit problem—with each kernel acting as an arm—and an adaptive parameter-tuning scheme is introduced to optimize the regret bound. In a diferent strategy, [17] introduced an incremental sketched kernel alignment metric to select a kernel, also employing OGD to update the chosen classifier.

• Online kernel combination methods, which also leverage a predefined kernel pool but difer by updating multiple kernel classifiers and their combination coeficients in each round. Hoi et al. [18] proposed to fuse two online algorithms:

the Perceptron algorithm that learn individual classifiers, and the Hedge algorithm that assigns their weights. To simultaneously learn classifiers and their combination coeficients, Shen et al. [19] formulated a non-convex joint optimization problem, solving it through a fusion of OGD and the Exponentiated Gradient algorithm [20]. Further, Hong [21] introduced a collaborative Hedge algorithm to improve the combination coeficient update mechanism. To handle large kernel pools, Ghari and Shen [22] proposed a graph-assisted framework that dynamically selects kernel subsets based on classifier losses or kernel similarities.

• Methods that dynamically tune the kernel parameter, which simultaneously learn the classifier and its kernel parameter during the learning process. Nguyen et al. [23] adapted the Random Fourier Features (RFF) technique [24] using the reparameterization trick. By lifting the randomness of Fourier components into an independent space, their method allows for eficient, gradient-based online learning of the kernel parameter. Zhang and Ning [25] transformed the learning problem into two optimal control problems. By solving both problems and following an alternating optimization scheme, the classifier parameters and the kernel bandwidth can be eficiently updated. Subsequently, [26] framed the entire learning task as a unified state feedback optimal control problem and employed an infinite-horizon model predictive control technique to derive online update rules for both the classifier and the kernel bandwidth.

The aforementioned multi-kernel methods fail to capture the complex relationships between instances and labels that arise in online multi-label classification. A recent approach, Online Kernel Selection for Online Multi-label Classification (OKS-OMC) [27], attempts to bridge this gap. This method extends the single-kernel multilabel FALT algorithm [13] to a multi-kernel setting by formulating a joint optimization problem for multi-label classifiers and their combination coeficients, which is then approximately addressed by minimizing a tractable surrogate that upper-bounds the original objective. OKS-OMC is proved to converge to the best fixed single-kernel multi-label classifier chosen in hindsight. However, OKS-OMC is hampered by two critical drawbacks. Firstly, its regret bound indicates that it functions primarily as an online kernel selector, failing to fully harness the complementary information from multiple kernels. Secondly, it encounters the “curse of kernelization" [28, 29], as the size of each constituent model grows linearly with the number of data points, thereby compromising the scalability essential for online learning.

In this paper, we propose a novel online multi-kernel combination method for large-scale multi-label classification. Our contributions are as follows:

1. Firstly, we design a new online single-kernel multi-label classification method by integrating advanced RFF techniques [24, 30] with the fixed label thresholding method FFLT [13]. This integration ofers a twofold advantage: 1) RFF ensures that the model size remains constant, independent of the data size; 2) The FFLT strategy enables robust learning in the high-dimensional feature spaces induced by RFF, efectively mitigating the overfitting risk in adaptively learning the label threshold.

2. Secondly, we formulate the learning of the optimal convex combination of multi-label kernel classifiers as a structural risk minimization problem. To solve it, we develop an improved alternating optimization procedure that decouples the learning of the kernel classifiers from that of the combination coeficients. This decomposition allows each kernel classifier to be updated seamlessly using our proposed single-kernel method, and facilitates fast convergence to a high-quality solution. We term the resulting method the Optimal Multi-Kernel Combination (OMKC) algorithm.

3. A theoretical analysis establishes that OMKC asymptotically converges to the optimalfixed convex combination of the optimalfixed single-kernel multi-label models chosen in hindsight. This indicates that OMKC can leverage the complementary information of multiple kernel functions.

4. Empirical results demonstrate the superiority of OMKC’s coeficient update rule and classifier update strategy over those in OKS-OMC. Furthermore, comprehensive evaluations on 11 datasets using seven standard metrics demonstrate that OMKC statistically significantly outperforms existing online multi-label classification methods in terms of overall online performance.

The paper is structured as follows. Section 2 details the notation and problem setting. Section 3 introduces the $\mathrm { R F F }$ technique, which serves as a foundation for our method. Our method is elaborated in Section 4, alongside a theoretical analysis of its regret bound. Experimental results and discussions are presented in Section 5. Finally, Section 6 summarizes the study, discusses the limitations of our method, and suggests future research directions.

## 2. Notation and problem setting

Let $( { \pmb x } _ { 1 } , Y _ { 1 } ) , \cdots , ( { \pmb x } _ { T } , Y _ { T } )$ be a multi-label data stream, where at any time step t, the instance $\pmb { x } _ { t } \in \mathbb { R } ^ { d }$ is associated with a set of relevant labels $Y _ { t } \subseteq y .$ . Here, $y =$ $\{ 1 , 2 , \cdots , L \}$ is the full set of possible labels, and ${ \bar { Y } } _ { t } ~ = ~ y ~ \backslash ~ Y _ { t }$ denotes the set of irrelevant labels for $\scriptstyle { \mathbf { } } x _ { t }$ . We use the following mathematical notation throughout this paper. The cardinality of a set S is |S |. For a positive integer N, [N] denotes the set $\{ 1 , 2 , \cdots , N \}$ . The indicator function 1[a] outputs 1 if condition a is true, and 0 otherwise. The absolute value of a scalar a is $\vert a \vert .$ . The i-th component of a vector u is $u _ { i } .$ . The $\ell _ { 1 } , \ell _ { 2 }$ and $\ell _ { \infty }$ norms of a vector u are denoted by ∥u∥<sub>1</sub>, ∥u∥, and $\| u \| _ { \infty } ,$ respectively. The transpose of a matrix A is $A ^ { \top }$ and its Frobenius norm is $\| A \| _ { F }$

The online multi-label classification problem can be framed as a sequential prediction game. In each round t of this game, the environment reveals an instance $\mathbf { \boldsymbol { x } } _ { t }$ and requires the learner to predict a set of relevant labels $\hat { Y } _ { t } \subseteq y$ for $\mathbf { \boldsymbol { x } } _ { t } .$ . After the prediction, the true relevant label set $Y _ { t }$ is disclosed and the learner incurs a loss that quantifies its prediction performance. Based on this loss, the learner then refines its model, aiming to minimize the cumulative loss over all rounds, which is equivalent to maximizing the sequential multi-label prediction performance.

## 3. Random Fourier Features (RFF)

RFF [24] is a technique to approximate shift-invariant kernels. Its core idea is to find an explicit feature mapping ϕ<sup>ˆ</sup>(·) with a user-controllable dimension, such that the

inner product $\hat { \phi } ( \pmb { x } ) ^ { \top } \hat { \phi } ( z )$ approximates the kernel function $\mathcal { K } ( \boldsymbol { x } , z )$ . A kernel is shiftinvariant if it satisfies $\mathcal { K } ( \boldsymbol { x } , z ) = K ( \delta )$ for $\delta = x - z$ and some function K. Given a shift-invariant kernel $K ( \cdot ) .$ , its Fourier transform, denoted by $p ( \pmb { u } )$ , can be expressed as

$$
p ( \pmb { u } ) = ( \frac { 1 } { 2 \pi } ) ^ { d } \int _ { \mathbb { R } ^ { d } } K ( \pmb { \delta } ) \exp ( - j \pmb { \delta } ^ { \top } \pmb { u } ) d \pmb { \delta } .\tag{1}
$$

Bochner’s theorem guarantees that $p ( \pmb { u } )$ can be viewed as a probability distribution if $K ( \delta )$ is properly scaled [24]. According to Fourier inversion theorem, $K ( \delta )$ can be recovered from its Fourier transform p(u):

$$
K ( \delta ) = \int _ { \mathbb { R } ^ { d } } p ( u ) \exp ( j u ^ { \top } \delta ) d u = \mathbb { E } _ { u \sim p ( u ) } [ \exp ( j u ^ { \top } \delta ) ] = \mathbb { E } _ { u \sim p ( u ) } [ \cos ( u ^ { \top } \delta ) ]
$$

where the last equation results from neglecting the imaginary part, $j \sin ( \pmb { u } ^ { \top } \pmb \delta )$ . Thus, we obtain $\mathcal { K } ( \pmb { x } , z ) = \mathbb { E } _ { \pmb { u } \sim p ( \pmb { u } ) } [ \cos ( \pmb { u } ^ { \top } ( \pmb { x } - z ) ) ]$ . By using the trigonometric identity

$$
\cos ( { \boldsymbol u } ^ { \top } ( { \boldsymbol x } - { \boldsymbol z } ) ) = \cos ( { \boldsymbol u } ^ { \top } { \boldsymbol x } + b ) \cos ( { \boldsymbol u } ^ { \top } { \boldsymbol z } + b ) + \sin ( { \boldsymbol u } ^ { \top } { \boldsymbol x } + b ) \sin ( { \boldsymbol u } ^ { \top } { \boldsymbol z } + b )
$$

and defining $\varphi _ { u , b } ( { \pmb x } ) = [ \cos ( { \pmb u } ^ { \top } { \pmb x } + b ) , \sin ( { \pmb u } ^ { \top } { \pmb x } + b ) ] ^ { \top }$ , we get that cos $\begin{array} { r } { \mathbf { \sigma } ( \pmb { u } ^ { \top } ( \pmb { x } - z ) ) = } \end{array}$ $\varphi _ { u , b } ( { \pmb x } ) ^ { \top } \varphi _ { { \pmb u } , b } ( z )$ . Let $\mathcal { U } ( 0 , 2 \pi )$ denote a uniform distribution in [0 2π]. Taking the expectation over $\pmb { u } \sim p ( \pmb { u } )$ and $b \sim \mathcal { U } ( 0 , 2 \pi )$ yields

$$
\mathcal { K } ( \pmb { x } , z ) = \mathbb { E } _ { \pmb { u } \sim p ( \pmb { u } ) , b \sim \mathcal { U } ( 0 , 2 \pi ) } [ \varphi _ { \pmb { u } , b } ( \pmb { x } ) ^ { \top } \varphi _ { \pmb { u } , b } ( z ) ] .\tag{2}
$$

By randomly drawing D independent samples ${ \pmb u } _ { 1 } , { \pmb u } _ { 2 } , \cdots , { \pmb u } _ { D }$ from $p ( \pmb { u } )$ , and also $b _ { 1 } , b _ { 2 } , \cdots , b _ { D }$ from $\mathcal { U } ( 0 , 2 \pi )$ , the expectation in $\operatorname { E q . } ( 2 )$ can be approximated by the sample average:

$$
\mathcal { K } ( \pmb { x } , z ) \approx \frac { 1 } { D } \sum _ { k = 1 } ^ { D } \varphi _ { u _ { k } , b _ { k } } ( \pmb { x } ) ^ { \top } \varphi _ { u _ { k } , b _ { k } } ( z ) = \hat { \phi } ( \pmb { x } ) ^ { \top } \hat { \phi } ( z )\tag{3}
$$

where $\hat { \phi } ( { \pmb x } )$ is an explicit feature map of 2D dimensionality, defined as

$$
\hat { \phi } ( { \pmb x } ) = \frac { 1 } { \sqrt { D } } [ \varphi _ { { \pmb u } _ { 1 } , b _ { 1 } } ( { \pmb x } ) ^ { \top } , \cdot \cdot \cdot , \varphi _ { { \pmb u } _ { D } , b _ { D } } ( { \pmb x } ) ^ { \top } ] ^ { \top } .\tag{4}
$$

According to $\mathrm { E q . } ( 3 ) , \hat { \phi } ( \pmb { x } ) ^ { \top } \hat { \phi } ( z )$ is unbiased estimate of $\mathcal { K } ( \boldsymbol { x } , z )$ . Further, it is proven that their diference decreases at an exponential rate in $D \ [ 2 4 ]$

To construct the feature map ${ \hat { \phi } } ( \cdot )$ , we must calculate out $p ( \pmb { u } )$ by $\operatorname { E q . } ( 1 )$ and then sample u from $p ( \pmb { u } )$ . A computational simplification arises when the kernel function is separable. If a kernel function satisfies $\begin{array} { r } { K ( \delta ) = \prod _ { i = 1 } ^ { d } k ( \delta _ { i } ) } \end{array}$ , then its Fourier transform $p ( \pmb { u } )$ factorizes accordingly: $\begin{array} { r } { p ( \pmb { u } ) \ = \ \prod _ { i = 1 } ^ { d } p ( u _ { i } ) } \end{array}$ . This factorization implies that all components of $\mathbf { \delta } _ { \mathbf { u } } ,$ that is, $u _ { 1 } , u _ { 2 } , \cdots , u _ { d } .$ , are independent and identically distributed, each following a one-dimensional marginal distribution $p ( u _ { i } )$ . Thus, to draw a $d -$ dimensional vector u from $p ( \pmb { u } )$ , one simply draws $d$ independent samples from the univariate distribution $p ( u _ { i } )$ . Table 1 lists several common shift-invariant kernels, their Fourier transforms, and the corresponding sampling distribution for $u _ { i }$

Table 1: Shift-invariant kernels, their Fourier transforms and the corresponding distribution for u .
<table><tr><td>Kernel name</td><td>K(δ)</td><td>p(u)</td><td>distribution of  $u _ { i }$ </td></tr><tr><td>Gaussian</td><td> $\exp ( - \frac { \| \delta \| ^ { 2 } } { 2 \sigma ^ { 2 } } )$ </td><td> $\begin{array} { r } { \prod _ { i = 1 } ^ { d } \frac { \sigma } { \sqrt { 2 \pi } } \exp ( - \frac { \sigma ^ { 2 } u _ { i } ^ { 2 } } { 2 } ) } \end{array}$ </td><td>Gaussian  ${ \cal N } ( 0 , 1 / \sigma ^ { 2 } )$ </td></tr><tr><td>Laplacian</td><td> $\exp ( - \frac { \Vert \delta \Vert _ { 1 } } { \sigma } )$ </td><td> $\begin{array} { r } { \prod _ { i = 1 } ^ { d } \frac { \sigma } { \pi ( 1 + \sigma ^ { 2 } u _ { i } ^ { 2 } ) } } \end{array}$ </td><td>Cauchy C(0, 1/σ)</td></tr><tr><td>Cauchy</td><td> $\begin{array} { r } { \frac { 1 } { 2 ^ { d } } \prod _ { i = 1 } ^ { d } \frac { 2 } { 1 + ( \delta _ { i } / \sigma ) ^ { 2 } } } \end{array}$ </td><td> $\begin{array} { r } { \prod _ { i = 1 } ^ { d } \frac { \sigma } { 2 } \exp ( - \sigma | u _ { i } | ) } \end{array}$ </td><td>Laplace L(0, 1/σ)</td></tr></table>

The introduction of RFF has spurred extensive research into kernel approximation techniques; we refer to [31] for a comprehensive survey. While initially developed for shift-invariant kernels, the RFF framework has since been successfully generalized to other kernel classes, including arc-cosine kernels [31, 32] and polynomial kernels [30, 33].

## 4. The proposed approach

## 4.1. RFF-based online single-kernel multi-label classification

Given a Mercer kernel $\mathcal { K } _ { p } ,$ Mercer’s theorem [12] guarantees the existence of an implicit feature map $\phi _ { p } ( \cdot )$ to a reproducing kernel Hilbert space $\mathcal { H } _ { p }$ such that this kernel can be expressed as an inner product in this space: $\mathcal { K } _ { p } ( \pmb { x } , z ) = \langle \phi _ { p } ( \pmb { x } ) , \phi _ { p } ( z ) \rangle _ { \mathcal { H } _ { p } }$ We can use this map $\phi _ { p } ( \cdot )$ to transform instances from the original input space to $\mathcal { H } _ { p }$ thereby learning our multi-label classifier in $\mathcal { H } _ { p }$

Our single-kernel multi-label classifier is constructed based on the First-order Fixed Label Thresholding (FFLT) method [13]. Specifically, the classifier maintains L predictors, each generating a score for one of the L labels, coupled with a predefined global fixed threshold γ to discriminate between relevant and irrelevant labels.

Formally, the predictors are parameterized by a matrix $\pmb { W } _ { p } = [ \pmb { w } _ { p } ^ { ( 1 ) } , \cdots , \pmb { w } _ { p } ^ { ( L ) } ]$ , where for any $i \in [ L ] , w _ { p } ^ { ( i ) } \in \mathcal { H } _ { p }$ corresponds to the scoring predictor for i-th label. For an instance $\mathbf { } _ { x _ { t } , }$ its new feature representation in $\mathcal { H } _ { p }$ is $\phi _ { p } ( { \pmb x } _ { t } )$ . The set of relevant labels for $\mathbf { \boldsymbol { x } } _ { t }$ is then predicted as $\{ i \in \mathcal { Y } : \phi _ { p } ( x _ { t } ) ^ { \top } w _ { p } ^ { ( i ) } > \gamma \}$ . The rule classifies a label as relevant if its score $\phi _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p } ^ { ( i ) }$ exceeds the threshold $\gamma .$ . The rule directly implies the following loss function for evaluating performance:

$$
\begin{array} { r } { \ell ( W _ { p } ; ( \phi _ { p } ( { \boldsymbol { \mathbf { x } } } _ { t } ) , Y _ { t } ) ) = \displaystyle \frac { 1 } { | Y _ { t } | } \sum _ { i \in Y _ { t } } \operatorname* { m a x } \left\{ 0 , 1 - \left( \phi _ { p } ( { \boldsymbol { \mathbf { x } } } _ { t } ) ^ { \top } w _ { p } ^ { ( i ) } - \gamma \right) \right\} } \\ { + \displaystyle \frac { 1 } { | \bar { Y } _ { t } | } \sum _ { i \in \bar { Y } _ { t } } \operatorname* { m a x } \left\{ 0 , 1 - \left( \gamma - \phi _ { p } ( { \boldsymbol { \mathbf { x } } } _ { t } ) ^ { \top } w _ { p } ^ { ( i ) } \right) \right\} . } \end{array}
$$

Note that $\ell ( W _ { p } ; ( \phi _ { p } ( { \pmb x } _ { t } ) , Y _ { t } ) )$ consists of a sum of losses over all labels. The loss associated with each label is weighted according to the contribution of the current instance to that label. For a relevant label, the contribution degree is quantified as $\frac { 1 } { | Y _ { t } | }$ , whereas for an irrelevant label, it is quantified as $\frac { 1 } { | \bar { Y } _ { t } | }$ . This specific weighting mechanism ensures that within each instance, relevant and irrelevant labels contribute equally to the loss, preventing either side from dominating the learning signal.

Our single-kernel multi-label model is learnt by solving the following structural risk minimization problem:

$$
\operatorname* { m i n } _ { W _ { p } } \left\{ { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } f ( W _ { p } ; ( \phi _ { p } ( x _ { t } ) , Y _ { t } ) ) \right\}\tag{5}
$$

where $f ( W _ { p } ; ( \phi _ { p } ( { \pmb x } _ { t } ) , Y _ { t } ) ) ~ = ~ \ell ( W _ { p } ; ( \phi _ { p } ( { \pmb x } _ { t } ) , Y _ { t } ) ) + \frac { \lambda } { 2 } \| W _ { p } \| _ { F } ^ { 2 }$ , and the hyperparameter $\lambda > 0$ is used to control the model complexity.

Problem (5) is convex. Thus in an online setting, it can be solved using OGD. For convenience, let $f _ { t } ( W _ { p } )$ be the shorthand for $f ( W _ { p } ; ( \phi _ { p } ( \mathbf { x } _ { t } ) , Y _ { t } ) )$ . Let $W _ { p , t } ~ =$ $[ \pmb { w } _ { p , t } ^ { ( 1 ) } , \cdots , \pmb { w } _ { p , t } ^ { ( L ) } ]$ be the single-kernel multi-label model at online round t. Following the steps of OGD, $\pmb { w } _ { p , t } ^ { ( i ) }$ is first initialized to zero vector for any i, then it can be incrementally updated as

$$
\begin{array} { r } { \pmb { w } _ { p , t + 1 } ^ { ( i ) } = \pmb { w } _ { p , t } ^ { ( i ) } - \eta \nabla _ { \pmb { w } _ { p } ^ { ( i ) } } f _ { t } ( \pmb { W } _ { p , t } ) , \ \forall i \in [ L ] , } \end{array}\tag{6}
$$

where $\nabla _ { w _ { n } ^ { ( i ) } } f _ { t } ( W _ { p , t } )$ belongs to the sub-gradient set of $f _ { t } ( W _ { p } )$ with respect to $\pmb { w } _ { p } ^ { ( i ) }$ at $W _ { p , t } .$ , and $\eta < 1 / \lambda$ is the learning rate. Given that $\nabla _ { { \pmb w } _ { p } ^ { ( i ) } } f _ { t } ( { \pmb W } _ { p , t } ) = \lambda { \pmb w } _ { p , t } ^ { ( i ) } + \gamma _ { p , t } ^ { ( i ) } \phi _ { p } ( { \pmb x } _ { t } )$

where

$$
\begin{array} { r } { \gamma _ { p , t } ^ { ( i ) } = \left\{ \begin{array} { l l } { - \frac { \mathbb { 1 } [ \phi _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } - \gamma < 1 ] } { | \pmb { Y } _ { t } | } } & { \mathrm { ~ i f ~ } i \in Y _ { t } , } \\ { \frac { \mathbb { 1 } [ \gamma - \phi _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } < 1 ] } { | \bar { Y } _ { t } | } } & { \mathrm { ~ i f ~ } i \in \bar { Y } _ { t } , } \end{array} \right. } \end{array}\tag{7}
$$

the update formula (6) of OGD can be rewritten as

$$
\pmb { w } _ { p , t + 1 } ^ { ( i ) } = ( 1 - \lambda \eta ) \pmb { w } _ { p , t } ^ { ( i ) } - \eta \gamma _ { p , t } ^ { ( i ) } \phi _ { p } ( \pmb { x } _ { t } ) , \forall i \in [ L ] .\tag{8}
$$

A recursive expansion of $\operatorname { E q . } ( 8 )$ yields $\begin{array} { r } { \pmb { w } _ { p , t + 1 } ^ { ( i ) } = \sum _ { k = 1 } ^ { t } \alpha _ { p , k } ^ { ( i ) } \phi _ { p } ( \pmb { x } _ { k } ) } \end{array}$ , where $\alpha _ { p , k } ^ { ( i ) }$ is the coeficient corresponding to $\phi _ { p } ( { \pmb x } _ { k } )$ . For many popular kernels, such as Gaussian kernels, the map $\phi _ { p } ( \cdot )$ lacks an explicit, finite-dimensional form. Consequently, the representation of these kernel-based scoring predictors grows linearly with the number of observed examples, incurring prohibitive computational and storage costs on large-scale data streams. This well-known scalability problem is termed the "curse of kernelization" [28].

RFF-based kernel approximation techniques provide a breakthrough for this problem by constructing an explicit, approximate feature map $\hat { \phi } _ { p } ( \cdot )$ that satisfies $\mathcal { K } _ { p } ( \pmb { x } , z ) \approx$ $\hat { \phi } _ { p } ( \pmb { x } ) ^ { \top } \hat { \phi } _ { p } ( z )$ . Using $\hat { \phi } _ { p } ( \cdot )$ , instances are mapped to a finite-dimensional feature space, denoted by $\hat { \mathcal { H } } _ { p }$ . Substituting $\hat { \phi } _ { p } ( \cdot )$ from Eq.(4) for the implicit map $\phi _ { p } ( \cdot )$ in $\operatorname { E q . } ( 7 )$ and (8), we obtain

$$
\pmb { w } _ { p , t + 1 } ^ { ( i ) } = ( 1 - \lambda \eta ) \pmb { w } _ { p , t } ^ { ( i ) } - \eta \gamma _ { p , t } ^ { ( i ) } \hat { \phi } _ { p } ( \pmb { x } _ { t } ) , \forall i \in [ L ] .\tag{9}
$$

Now, each predictor $\pmb { w } _ { p , t + 1 } ^ { ( i ) }$ can be represented and stored as a fixed-dimensional vector in $\mathbb { R } ^ { 2 D }$ . Consequently, the task of learning a single-kernel multi-label model is reduced to learning a linear multi-label model in $\mathbb { R } ^ { 2 D }$ , thereby perfectly combining the eficiency of the linear learning algorithm with the expressive power of nonlinear kernels.

## 4.2. Multi-kernel multi-label classification

Single-kernel methods depend heavily on a manually selected kernel, a process that is both computationally expensive and prone to suboptimal choices, especially with limited data. Online multi-kernel learning overcomes this by automating kernel selection and combination during the learning process.

Let $\mathcal { K } _ { 1 } , \cdots , \mathcal { K } _ { P }$ denote $P$ predefined Mercer kernels, with $\hat { \phi } _ { 1 } ( \cdot ) , \cdot \cdot \cdot , \hat { \phi } _ { P } ( \cdot )$ being their respective RFF approximate feature maps. Thus, an instance $\mathbf { \boldsymbol { x } } _ { t }$ is projected into $P$ feature spaces, yielding representations $\hat { \phi } _ { 1 } ( { \pmb x } _ { t } ) , \cdot \cdot \cdot , \hat { \phi } _ { P } ( { \pmb x } _ { t } )$

For each kernel $\mathcal { K } _ { p }$ , let $W _ { p }$ denote its associated single-kernel multi-label model, constructed as described in the previous subsection. Many methods exist for combining these single-kernel models. Here, we adopt the widely used convex combination strategy. Specifically, we assign a non-negative coeficient $\mu _ { p }$ to each model $W _ { p }$ , subject to $\textstyle \sum _ { p = 1 } ^ { P } \mu _ { p } = 1$ . Let $\mathcal { W } = \{ \boldsymbol { W } _ { 1 } , \cdots , \boldsymbol { W } _ { P } \}$ and $\pmb { \mu } = [ \mu _ { 1 } , \cdots , \mu _ { P } ] ^ { \top } \in \Delta$ , where $\Delta$ denotes a probability simplex in $\mathbb { R } ^ { P }$ , defined as $\Delta = \{ r \in [ 0 , 1 ] ^ { P } : \textstyle \sum _ { p = 1 } ^ { P } r _ { p } = 1 \}$ . Our multi-kernel multi-label model is parameterized by $( \mathcal W , \pmb { \mu } )$

Under this convex combination scheme, the score for the i-th label of an instance $\mathbf { \boldsymbol { x } } _ { t }$ is computed as the convex combination of predictions from all single-kernel label scoring predictors: $\begin{array} { r } { h ^ { ( i ) } ( \mathbf { x } _ { t } ) = \sum _ { p = 1 } ^ { P } \mu _ { p } \hat { \phi } _ { p } ( \mathbf { x } _ { t } ) ^ { \top } \mathbf { w } _ { p } ^ { ( i ) } } \end{array}$ . The set of relevant labels for $\mathbf { \boldsymbol { x } } _ { t }$ is then predicted as: $\hat { Y } _ { t } = \left\{ i \in \mathcal { Y } : h ^ { ( i ) } ( { \pmb x } _ { t } ) > \gamma \right\}$ , where $\gamma$ is the predefined threshold. This decision rule naturally gives rise to the following loss function for evaluating the performance of our multi-kernel multi-label model:

$$
\begin{array} { r } { \mathcal { L } ( \mathcal { W } , \mu ; ( \boldsymbol { x } _ { t } , Y _ { t } ) ) = \displaystyle \frac { 1 } { | Y _ { t } | } \sum _ { i \in Y _ { t } } \operatorname* { m a x } \left\{ 0 , 1 - \left( \sum _ { p = 1 } ^ { P } \mu _ { p } \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) ^ { \top } \boldsymbol { w } _ { p } ^ { ( i ) } - \gamma \right) \right\} } \\ { + \displaystyle \frac { 1 } { | \bar { Y } _ { t } | } \sum _ { i \in \bar { Y } _ { t } } \operatorname* { m a x } \left\{ 0 , 1 - \left( \gamma - \sum _ { p = 1 } ^ { P } \mu _ { p } \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) ^ { \top } \boldsymbol { w } _ { p } ^ { ( i ) } \right) \right\} . } \end{array}
$$

In the framework of structural risk minimization, we hope to find $( \mathcal W , \pmb { \mu } )$ such that the regularized loss is minimized:

$$
\operatorname* { m i n } _ { \mathcal { W } , \mu } \left\{ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { F } ( \mathcal { W } , \mu ; ( x _ { t } , Y _ { t } ) ) \right\}\tag{10}
$$

where

$$
\mathcal { F } ( \mathcal { W } , \mu ; ( \boldsymbol { x } _ { t } , Y _ { t } ) ) = \mathcal { L } ( \mathcal { W } , \mu ; ( \boldsymbol { x } _ { t } , Y _ { t } ) ) + \frac { \lambda } { 2 } \sum _ { p = 1 } ^ { P } \mu _ { p } \| W _ { p } \| _ { F } ^ { 2 } .
$$

The hyperparameter $\lambda > 0$ controls the strength of regularization. The combination coeficients are also embedded within the regularization term, which scales the penalty in a capacity-aware manner [27].

The optimization problem (10) is convex with respect to $\mathcal { W }$ alone and $\pmb { \mu }$ alone, but it is non-convex in the joint variable $( \mathcal W , \pmb { \mu } )$ . This lack of joint convexity presents a primary challenge. A secondary challenge arises from its scale: the problem has $2 D L P + P - 1$ parameters, which grows linearly with P. For large P, this significantly exceeds the 2DL parameters of the single-kernel problem (5).

Finding the global optimum of (10) is computationally expensive, even in an offline setting—let alone in an online scenario. Therefore, we focus on identifying a strong local optimum that guarantees robust performance on newly arriving data, rather than pursuing the globally optimal solution.

A natural approach to solve an optimization problem with two variable blocks is the Generalized Alternating Minimization (GAM) [34], which iteratively fixes one variable block and optimizes the other by minimizing the original objective function. If closed-form solutions for the marginal optimizations are unavailable, gradient descent steps can be applied instead. However, in an online setting, GAM sufers from extremely slow convergence, as confirmed in Section 5.1. To address this, we propose an improved alternating optimization strategy, updating the variables from $( \mathcal { W } _ { t } , \pmb { \mu } _ { t } )$ to $( \mathcal { W } _ { t + 1 } , \mu _ { t + 1 } )$ as follows:

Step 1: fix $\mathcal { W }$ to $\mathcal { W } _ { t }$ and obtain $\pmb { \mu } _ { t + 1 }$ by minimizing the original objective function $\mathcal { F } ( \mathcal { W } _ { t } , \mu ; ( \pmb { x } _ { t } , Y _ { t } ) ) ;$

Step 2: fix $\pmb { \mu }$ to $\pmb { \mu } _ { t + 1 }$ and obtain $\boldsymbol { \mathcal { W } } _ { t + 1 }$ by minimizing an upper bound of the original objective $\mathcal { F } ( \mathcal { W } , \mu _ { t + 1 } ; ( x _ { t } , Y _ { t } ) )$

Crucially, optimizing an upper bound in Step 2 efectively decouples the Wupdate from $\mu ,$ permitting independent optimization of W, as demonstrated later. This strategy facilitates faster convergence to a high-quality solution.

When W is fixed to $\mathcal { W } _ { t }$ in Step 1, the marginal minimization problem with respect to µ lacks a closed-form solution. We therefore perform a gradient descent on $\mu .$ Since $\pmb { \mu }$ is constrained to the probability simplex, we apply the Exponential Gradient Descent (EGD) [35] to maintain feasibility.

Specifically, define $G _ { t } ( \mu ) = \mathcal { F } ( \mathcal { W } _ { t } , \mu ; ( x _ { t } , Y _ { t } ) )$ and let $\nabla _ { \mu } G _ { t } ( \mu _ { t } )$ denote its gradient

at $\pmb { \mu } _ { t }$ . The p-th component of this gradient, denoted by $[ \nabla _ { \pmb { \mu } } G _ { t } ( \pmb { \mu } _ { t } ) ] _ { p }$ , is computed as:

$$
\begin{array} { r } { [ \nabla _ { \mu } G _ { t } ( \mu _ { t } ) ] _ { p } = \displaystyle \frac { \lambda } { 2 } \| \mathbf { \boldsymbol { W } } _ { p , t } \| _ { F } ^ { 2 } - \displaystyle \frac { 1 } { | Y _ { t } | } \sum _ { i \in Y _ { t } } a _ { t } ^ { ( i ) } \hat { \phi } _ { p } ( \mathbf { \boldsymbol { x } } _ { t } ) ^ { \top } \mathbf { \boldsymbol { w } } _ { p , t } ^ { ( i ) } } \\ { + \displaystyle \frac { 1 } { | \bar { Y } _ { t } | } \sum _ { i \in \bar { Y } _ { t } } b _ { t } ^ { ( i ) } \hat { \phi } _ { p } ( \mathbf { \boldsymbol { x } } _ { t } ) ^ { \top } \mathbf { \boldsymbol { w } } _ { p , t } ^ { ( i ) } } \end{array}\tag{11}
$$

where $\begin{array} { r } { a _ { t } ^ { ( i ) } = \mathbb { 1 } [ \sum _ { p = 1 } ^ { P } \mu _ { p , t } \hat { \phi } _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } - \gamma < 1 ] } \end{array}$ and $b _ { t } ^ { ( i ) } = \mathbb { 1 } [ \gamma - \sum _ { p = 1 } ^ { P } \mu _ { p , t } \hat { \phi } _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } < 1 ]$

Thus, each component of $\pmb { \mu } _ { t } = [ \mu _ { 1 , t } , \cdots , \mu _ { P , t } ] ^ { \top }$ can be incrementally updated as:

$$
\mu _ { p , t + 1 } = \frac { \mu _ { p , t } \exp { \left( - \beta \left[ \nabla _ { \mu } G _ { t } ( \mu _ { t } ) \right] _ { p } \right) } } { \sum _ { q = 1 } ^ { P } \mu _ { q , t } \exp { \left( - \beta \left[ \nabla _ { \mu } G _ { t } ( \mu _ { t } ) \right] _ { q } \right) } } , \forall p \in [ P ]\tag{12}
$$

where $\beta > 0$ is the learning rate for this update.

In Step 2, we consider the following upper bound of $\mathcal { F } ( \mathcal { W } , \mu _ { t + 1 } ; ( x _ { t } , Y _ { t } ) )$

$$
\begin{array} { l } { \displaystyle \mathcal { F } ( \mathcal { W } , \mu _ { t + 1 } ; ( \boldsymbol { x } _ { t } , Y _ { t } ) ) \leq \sum _ { p = 1 } ^ { P } \mu _ { p , t + 1 } \ell ( W _ { p } ; ( \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) , Y _ { t } ) ) + \frac { \lambda } { 2 } \sum _ { p = 1 } ^ { P } \mu _ { p , t + 1 } \| W _ { p } \| _ { F } ^ { 2 } } \\ { \displaystyle \quad \quad = \sum _ { p = 1 } ^ { P } \mu _ { p , t + 1 } \left( \ell ( W _ { p } ; ( \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) , Y _ { t } ) ) + \frac { \lambda } { 2 } \| W _ { p } \| _ { F } ^ { 2 } \right) } \\ { \displaystyle \quad \quad = \sum _ { p = 1 } ^ { P } \mu _ { p , t + 1 } f ( W _ { p } ; ( \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) , Y _ { t } ) ) } \end{array}\tag{13}
$$

where $\leq _ { 1 }$ follows from applying Jensen’s inequality g $\begin{array} { r } { \sum _ { p = 1 } ^ { P } \mu _ { p } z _ { p } ) \leq \sum _ { p = 1 } ^ { P } \mu _ { p } g ( z _ { p } ) } \end{array}$ [36] to the convex function $g ( z ) = \operatorname* { m a x } \{ 0 , 1 - z \}$

Eq. (13) suggests that $\mathcal { F } ( \mathcal { W } , \mu _ { t + 1 } ; ( x _ { t } , Y _ { t } ) )$ is upper bounded by a convex combination of single-kernel losses $f ( { \cal W } _ { p } ; ( \hat { \phi } _ { p } ( { \bf x } _ { t } ) , Y _ { t } ) )$ ). Minimizing this upper bound over $\mathcal { W }$ thus reduces to independently minimizing each $f ( { \cal W } _ { p } ; ( \hat { \phi } _ { p } ( { \bf x } _ { t } ) , Y _ { t } ) )$ over $W _ { p }$ . As this marginal optimization problem also lacks a closed-form solution, we apply a gradient descent on each $W _ { p }$ . Consequently, each $W _ { p , t + 1 }$ in $\mathcal { W } _ { t + 1 } = \{ \boldsymbol { W } _ { 1 , t + 1 } , \cdots , \boldsymbol { W } _ { P , t + 1 } \}$ can be obtained directly via Eq. (9). Notably, this optimization over $\mathcal { W }$ is independent of $\mu .$

To summarize, each round t involves updating the combination coeficients with Eq. (12) and updating each single-kernel multi-label model with Eq. (9).

We call this proposed approach the Optimal Multi-Kernel Combination algorithm (OMKC). This name reflects its capability to optimally combine multiple single-kernel multi-label classifiers, a property we formally demonstrate in the next section. The pseudocode for OMKC is provided in Algorithm 1.

Algorithm 1: OMKC   
Input: $\lambda > 0 , 0 < \eta < 1 / \lambda , \beta > 0 , \gamma , D , \{ \mathcal { K } _ { 1 } , \mathcal { K } _ { 2 } , \cdot \cdot \cdot , \mathcal { K } _ { P } \}$   
Output: $\mu _ { T + 1 } , W _ { p , T + 1 }$ for any $p \in [ P ]$   
1 $\pmb { \mu } _ { 1 } = [ \mu _ { 1 , 1 } , \cdots , \mu _ { P , 1 } ] ^ { \top } = [ 1 / P , \cdots , 1 / P ] ^ { \top } ;$   
$W _ { p , 1 } = [ \pmb { w } _ { p , 1 } ^ { ( 1 ) } , \cdots , \pmb { w } _ { p , 1 } ^ { ( L ) } ] = [ \pmb { 0 } , \cdots , \pmb { 0 } ] , \forall p \in [ P ]$   
2 for $p = 1 , 2 , \cdots , P$ do   
3 Obtain $p ( \pmb { u } )$ for the kernel $\mathcal { K } _ { p }$ using Eq.(1)   
4 Draw ${ \pmb u } _ { 1 } , { \pmb u } _ { 2 } , \cdots , { \pmb u } _ { D }$ from p(u), and $b _ { 1 } , b _ { 2 } , \cdots , b _ { D }$ from $\mathcal { U } ( 0 , 2 \pi )$   
5 Obtain the explicit feature map $\hat { \phi } _ { p } ( \cdot )$ associated with the kernel $\mathcal { K } _ { p }$ via   
Eq.(4)   
6 for $t = 1 , 2 , \cdots , T$ do   
7 Receive $\pmb { x } _ { t } \in \mathbb { R } ^ { d }$   
8 Calculate $\hat { \phi } _ { 1 } ( \pmb { x } _ { t } ) , \hat { \phi } _ { 2 } ( \pmb { x } _ { t } ) , \cdot \cdot \cdot , \hat { \phi } _ { P } ( \pmb { x } _ { t } )$   
9 Predict: $\hat { Y } _ { t } = \{ i \in \pmb { y } : \sum _ { p = 1 } ^ { P } \mu _ { p , t } \hat { \phi } _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } > \gamma \}$   
10 Receive $Y _ { t } \subseteq { \mathcal { Y } }$   
11 for $p = 1 , 2 , \cdots , P$ do   
12 Update $\mu _ { p , t + 1 }$ by Eq. (12)   
13 Update $W _ { p , t + 1 }$ by Eq. (9)

According to Algorithm 1, we can analyze the space and time complexity of OMKC. The total space complexity of OMKC comprises two components: the storage of RFF components and the storage of multi-label models. For each of the P kernels, OMKC maintains D RFF components, and storing each component requires O(d) space, resulting in a complexity of O(DPd). Additionally, OMKC maintains P multi-label models, each containing L label scoring predictors. Since each predictor requires O(2D) space, this part contributes a complexity of O(2DPL). Therefore, the overall space complexity is $O ( D P ( d + 2 L ) ) ,$ ).

The total time complexity for OMKC arises from two parts: computing the RFF mappings (Steps 2-5) and performing online prediction and model updates (Step 6- 13). The first part involves sampling D RFF components P times, with each component requiring O(d) time, resulting in a complexity of O(DPd). However, this computation is data-independent and can be performed in advance as a pre-processing step. Therefore, its time complexity is excluded from the total online complexity. Regarding the second part, we first analyze its per-round complexity. In each round, computing RFF feature vectors (Step 8) costs $O ( d D P )$ , online prediction (Step 9) costs O(2DPL), and updating the multi-kernel multi-label model (Steps 11-13) costs $O ( 2 D P L + P L ) = O ( 2 D P L )$ . Thus, the per-round time complexity is $O ( D P ( d + 4 L ) )$ , leading to a total complexity $O ( D P ( d + 4 L ) T )$

Crucially, the space and per-round time complexities of OMKC do not grow with the length of the data stream. This allows OMKC to circumvent the curse of kernelization.

## 4.3. Theoretical analysis

Since OMKC uses RFF feature maps satisfying $\| \hat { \phi } _ { p } ( { \pmb x } _ { t } ) \| = 1$ for all $t \in [ T ]$ and $p \in [ P ]$ , we can extend the analysis of Lemma 1 in [27] to OMKC. This yields the boundedness for each label scoring predictor $\boldsymbol { w } _ { p , t } ^ { ( i ) }$ and each single-kernel multi-label model $W _ { p , t } \colon \| w _ { p , t } ^ { ( i ) } \| \leq 1 / \lambda$ and $\| \boldsymbol { W } _ { p , t } \| _ { F } \le \sqrt { 2 } / \lambda$ for all $p , t , i .$ Based on these facts, we can prove that the classifier updating strategy of OMKC satisfies, for any $p \in [ P ]$

$$
\sum _ { t = 1 } ^ { T } f ( W _ { p , t } ; ( \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) , Y _ { t } ) ) - \operatorname* { m i n } _ { W _ { p } } \sum _ { t = 1 } ^ { T } f ( W _ { p } ; ( \hat { \phi } _ { p } ( \boldsymbol { x } _ { t } ) , Y _ { t } ) ) \leq O ( \sqrt { T } ) .\tag{14}
$$

This sublinear regret bound (14) implies that each single-kernel multi-label model maintained by OMKC asymptotically converges to the optimalfixed model chosen in hindsight from the corresponding kernel-induced RFF feature space.

Next, we prove that the gradient $\nabla _ { \mu } G _ { t } ( \mu _ { t } )$ used in the EGD step is bounded, which is essential for deriving the theoretical guarantee for OMKC’s coeficient update strategy. According to Eq.(11), we can obtain

$$
\begin{array} { l } { { | [ \nabla _ { \mu } G _ { t } ( \mu _ { t } ) ] _ { p } | \le \displaystyle \frac { \lambda } { 2 } | | W _ { p , t } | | _ { F } ^ { 2 } + \displaystyle \frac { 1 } { | Y _ { t } | } \sum _ { i \in Y _ { t } } a _ { t } ^ { ( i ) } | \hat { \phi } _ { p } ( { \pmb x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } | + \displaystyle \frac { 1 } { | \bar { Y } _ { t } | } \sum _ { i \in \bar { Y } _ { t } } b _ { t } ^ { ( i ) } | \hat { \phi } _ { p } ( { \pmb x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } | } } \\ { { \displaystyle \qquad \le \displaystyle \frac { 1 } { \lambda } + \displaystyle \frac { 1 } { \lambda } \left( \displaystyle \frac { 1 } { | Y _ { t } | } \sum _ { i \in Y _ { t } } a _ { t } ^ { ( i ) } + \displaystyle \frac { 1 } { | \bar { Y } _ { t } | } \sum _ { i \in \bar { Y } _ { t } } b _ { t } ^ { ( i ) } \right) \le \displaystyle \frac { 3 } { \lambda } } } \end{array}
$$

where Cauchy-Schwarz inequality gives

$$
| \widehat { \phi } _ { p } ( \pmb { x } _ { t } ) ^ { \top } \pmb { w } _ { p , t } ^ { ( i ) } | \leq | | \widehat { \phi } _ { p } ( \pmb { x } _ { t } ) | | \cdot | | \pmb { w } _ { p , t } ^ { ( i ) } | | \leq \frac { 1 } { \lambda } .
$$

Naturally, we obtain that $\| \nabla _ { \mu } G _ { t } ( \pmb { \mu } _ { t } ) \| _ { \infty } \leq \frac { 3 } { \lambda }$

Now, by directly applying the regret bound of EGD (Corollary 5.7 in [35]), we can prove that OMKC satisfies:

$$
\sum _ { t = 1 } ^ { T } G _ { t } ( \mu _ { t } ) - \operatorname* { m i n } _ { \mu \in \Delta } \sum _ { t = 1 } ^ { T } G _ { t } ( \mu ) \leq O ( { \sqrt { T \ln P } } ) .\tag{15}
$$

This sublinear regret bound (15) reveals that, as $T \to \infty ,$ , the average online performance of OMKC matches that of the optimalfixed convex combination of its maintained single-kernel multi-label models.

Together with (14), these results show that, as $T \to \infty , { \mathrm { O M K C } }$ performs comparably to the optimalfixed convex combination of the bestfixed single-kernel multi-label models chosen in hindsight from diferent kernel-induced RFF spaces.

## 5. Experiments

This section presents three sets of experiments. The first set empirically examines the convergence of OMKC and compares it with OKS-OMC and two methods based on GAM. The second set compares OMKC against state-of-the-art online multi-label classification algorithms and includes an ablation analysis of its label thresholding strategy. The third set presents a sensitivity analysis of the key hyperparameters in OMKC. All experiments use open benchmark datasets from the $\mathrm { { M u l a n } ^ { 1 } }$ and $\mathbf { M L D A } ^ { 2 }$ repositories, with their characteristics detailed in Table 2.

Table 2: The characteristic of the datasets used in the experiments.
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">domains</td><td rowspan="2">(T)</td><td colspan="2">#instances #features #labels</td><td rowspan="2">cardinality</td></tr><tr><td>(d)</td><td>(L)</td></tr><tr><td>Emotions</td><td>music</td><td>593</td><td>72</td><td>6</td><td>1.868</td></tr><tr><td>Medical</td><td>text</td><td>978</td><td>1449</td><td>45</td><td>1.245</td></tr><tr><td>PlantGO</td><td>biology</td><td>978</td><td>3,091</td><td>12</td><td>1.079</td></tr><tr><td>PlantPseAAC</td><td>biology</td><td>978</td><td>440</td><td>12</td><td>1.079</td></tr><tr><td>Enron</td><td>text</td><td>1,702</td><td>1,001</td><td>53</td><td>3.378</td></tr><tr><td>Image</td><td>image</td><td>2,000</td><td>294</td><td>5</td><td>1.236</td></tr><tr><td>Scene</td><td>image</td><td>2,407</td><td>294</td><td>6</td><td>1.074</td></tr><tr><td>Yeast</td><td>biology</td><td>2,417</td><td>103</td><td>14</td><td>4.237</td></tr><tr><td>HumanGO</td><td>biology</td><td>3,106</td><td>9,844</td><td>14</td><td>1.185</td></tr><tr><td>HumanPseAAC</td><td>biology</td><td>3,106</td><td>440</td><td>14</td><td>1.185</td></tr><tr><td>Tmc2007-500</td><td>text</td><td>28,596</td><td>500</td><td>22</td><td>2.220</td></tr></table>

## 5.1. Empirical convergence analysis and comparison

In this section, we verify the convergence of OMKC by examining the evolution of its average cumulative loss. We then establish its advantage through a comparison with OKS-OMC [27] and two variants of the GAM method [34]: $\operatorname { G A M } \left( { \mathcal { W } } \to \mu \right)$ and GAM $( \mu \to \mathcal W )$ . These two variants are obtained by replacing our proposed optimization strategy with the GAM method, difering only in the update order of the variable blocks in each round. Specifically, $\operatorname { G A M } \left( { \mathcal { W } } \to \mu \right)$ first updates W and then µ, whereas GAM $( \pmb { \mu }  \mathcal { W } )$ first updates µ followed by W. For a controlled ablation study focused specifically on the combination coeficient updating strategies of OMKC and OKS-OMC, we adapt the single-kernel models of OKS-OMC by replacing their label thresholding mechanism with the fixed thresholding strategy used in OMKC.

Experiments are performed on six datasets randomly chosen from Table 2. All methods use the same hyperparameter settings on each dataset. Specifically, for Enron, Yeast and Tmc2007-500 datasets, seven polynomial kernels defined on the unit sphere [30] are used. These kernels take the form $\mathcal { K } _ { p } ( \pmb { x } _ { i } , \pmb { x } _ { j } ) = ( 1 - | | \pmb { x } _ { i } - \pmb { x } _ { j } | | ^ { 2 } / 4 ) ^ { b }$ ， where the polynomial order $b \in \{ 1 , 2 , \cdots , 7 \}$ . For Image, Scene and HumanPseAAC datasets, nine Laplacian kernels of the form $\mathcal { K } _ { p } ( \pmb { x } _ { i } , \pmb { x } _ { j } ) = \exp ( - \| \pmb { x } _ { i } - \pmb { x } _ { j } \| _ { 1 } / \sigma )$ are used, with $\sigma$ taking values in $\{ 2 ^ { - 2 } , 2 ^ { - 1 } , \cdots , 2 ^ { 6 } \}$ . In approximating the above two types of kernels, two diferent values for the number of RFF components $( \mathrm { i } . \mathrm { e } . , D )$ are selected for each dataset, based on its size and dimensionality. On these datasets, other hyperparameters are set as follows: regularization coeficient $\lambda = 1 0 ^ { - 6 }$ when the number of instances $T < 1 0 ^ { 4 }$ and $\lambda = 1 0 ^ { - 8 }$ when $T > 1 0 ^ { 4 }$ , classifier learning rate $\eta = 0 . 5$ combination coeficient learning rate $\beta = 0 . 2 5$ , and global threshold $\gamma = 0$ . Given that multiple learning on an incoming example can significantly improve online performance [11, 13, 27], all methods are permitted to learn each example L times.

We record the average cumulative loss, $A \nu g ( t )$ , of all methods, evaluated in a single pass over each dataset. $A \nu g ( t )$ is calculated as

$$
A \nu g ( t ) = \frac { 1 } { t * L } \sum _ { k = 1 } ^ { t } \sum _ { l = 1 } ^ { L } \mathcal { F } ( \mathcal { W } _ { k , l } , \mu _ { k , l } ; ( \pmb { x } _ { k } , Y _ { k } ) ) .
$$

where $( \mathcal { W } _ { k , l } , \mu _ { k , l } )$ is the model maintained by each method at round number k and time number l. Fig. 1 shows the evolution of $A \nu g ( t )$ for all methods as the round number t increases, based on results averaged over 20 independent runs with randomly shufled examples.

From these figures, it is evident that the average cumulative loss Avg(t) for OMKC exhibits a decreasing trend with the round number t, across all six datasets, indicating its convergence. On all datasets, OMKC consistently achieves the lowest Avg(t), followed by OKS-OMC, while the two GAM variants yield the highest values, reflecting their slower convergence.

The diference in Avg(t) between OMKC and OKS-OMC can be attributed to their distinct updating mechanisms for combination coeficients. Specifically, OMKC directly minimizes the objective function $\mathcal { F } ( \mathcal { W } , \mu ; ( x _ { t } , Y _ { t } ) )$ , whereas OKS-OMC minimizes an upper bound of this function. In contrast, the performance gap in Avg(t) between OMKC and GAM $( \mu \to \mathcal W )$ arises from their fundamental diference in updating kernel classifiers. GAM $( \mu \to \mathcal W )$ updates classifiers by directly minimizing $\mathcal { F } ( \mathcal { W } , \mu _ { t + 1 } ; ( x _ { t } , Y _ { t } ) )$ , whereas OMKC minimizes an upper bound of this function. OMKC’s design enables it to update classifiers independently of the combination coeficients, unlike GAM $( \mu \to \mathcal W )$ , where the updates depend on them. This key design of OMKC allows it to quickly focus on promising regions and accelerate convergence to a high-quality solution.

![](images/85b2908454f308ca3299e1793c7ec5dd3211fbc93923aefca0cbda85b425f013.jpg)  
(a)

![](images/d92cb6e5bbdb3e23176f88fd9ccfd961b47976ee2beada49e0619fbb92621e0d.jpg)  
(b)

![](images/4e14a084e143ec04e7baab9d8e1cd011ee20f2c6a9f77081d41ccd38dfe5130f.jpg)  
(c)

![](images/13c0cb5e7916ac7964a558b464f3c58cf5706168eb3f984a811bbf079f57d025.jpg)  
(d)

![](images/1d5e7a2a8ed2109d56ddaca3fd89a8fbf7b0c658efc45a872329c06a48abab0a.jpg)  
(e)

![](images/909fea3145d77a2439ada6d3d7a4aaab2d5e82afae10757c367a88d9c30b5311.jpg)  
(f)

![](images/fdeb282d6e217b0b673a198d8c92b5f5d00da85e33ef2408506f3338fc6e94cf.jpg)  
(g)

![](images/d49707cf4bce95273197d5fda5d5df198d74ed0563b0d076014cf6343e3ef479.jpg)  
(h)

![](images/da24ec467636c3fd1743a4a5906f64d9a144d9dc0b15fea8d220665537a7acd1.jpg)  
(i)

![](images/d3fe10350f4e1bc1e671dbace65ae8fa8cdd69f522b03c7507ee5a28c0e5deda.jpg)  
(j)

![](images/a2115f65b1dd90d801cb29ef21b877908b4e500ba447df2c6ebad3f06fd30e5a.jpg)  
(k)

![](images/ec3db91b2f0ca53bdcd56d70eb71ecbdf65dd7eaf53cc128b116b041b6c99d1a.jpg)  
(l)  
Figure 1: The evolution of average cumulative loss Avg(t) with the round number t for both OMKC and OKS-OMC. On the HumanPseAAC dataset, the Avg(t) values for the two GAM variants are substantially higher than those of OMKC and OKS-OMC. As a result, their curves fall outside the plotted range of [0.32, 0.42].

We further observe that a larger kernel approximation error (a smaller D) correlates with a greater gap in Avg(t) between OMKC and OKS-OMC. A plausible explanation is that an increased kernel approximation error degrades the accuracy of each single-kernel multi-label model. This degradation likely amplifies the discrepancy between the gradient directions of the true objective function $\mathcal { F } ( \mathcal { W } , \mu ; ( x _ { t } , Y _ { t } ) )$ and its upper-bound surrogate used by OKS-OMC with respect to $\mu .$

In conclusion, our results show that OMKC’s combination coeficient updating strategy outperforms that of OKS-OMC, and their performance gap widens with larger kernel approximation error. Furthermore, we show that, in an online setting, the improved alternating optimization strategy of OMKC is superior to the direct alternating optimization strategy of GAM.

## 5.2. Comparative experiments

This section presents a comprehensive evaluation of the following online multilabel classification methods:

• OSML-ELM [37] and ELM-OMLL [38]: two methods based on Extreme Learning Machine (ELM). Their number of hidden layer neurons is chosen from {100 200 · · ·  1000}. For ELM-OMLL, its regularization factor is from {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1}.

• ELM-O [39]: another ELM-based method originally designed for handling noisy labels in data streams. We evaluate it directly on our datasets without introducing additional label noise. Following the authors’ recommendations, we tune the weight factor $\beta$ in $\{ 0 . 1 , 0 . 2 , \cdots , 1 \}$ , and the regularization factor γ in $\{ 0 , 2 ^ { - 1 0 } , 2 ^ { - 9 } , \cdot \cdot \cdot , 2 ^ { 0 } \}$ , while keeping all other hyperparameters as default.

• PAML-I [11] and FFLT [13]: two single-kernel methods that have been integrated with the RFF techniques. For a fair comparison, their best kernel for each dataset is selected from the predefined kernel set used by OMKC. For PAML-I, its aggressiveness hyperparameter C is searched from $\{ 2 ^ { - 1 0 } , 2 ^ { - 9 } , \cdots , 2 ^ { 0 } \}$ . For FFLT, its step-size is set equivalent to that used in OMKC, and each example is also allowed to learn L times per online round.

• OKS-OMC [27]: an online multi-kernel learning algorithm. To ensure a fair comparison, OKS-OMC is implemented with the same RFF technique and hyperparameter configurations as OMKC on each dataset.

• OMKC: our method that uses nine Laplacian kernels on the Emotions, PlantPseAAC, Image, Scene, HumanPseAAC datasets, and seven polynomial kernels defined on the unit sphere on the remaining datasets. These kernels follow the same format and hyperparameter ranges described in the previous section. The classifier learning rate η is chosen from {0 25 0 5}. The number of RFF components, D, is chosen from the set {d 2d 4d 8d 16d 32d}. To maintain a reasonable computational cost, a smaller D value in this set is used when the original data dimensionality d is larger, and a larger D value is used when d is smaller. Other hyperparameters remain unchanged from earlier descriptions.

• OMKC<sub>a</sub>: a variant of our method that replaces the fixed label thresholding mechanism in the single-kernel models with the adaptive thresholding strategy used in OKS-OMC. This variant enables an ablation study to determine which label thresholding strategy—when integrated with our combination coeficient update strategy—yields better performance in the high-dimensional RFF-induced feature space. For this ablation study, $\mathrm { O M K C _ { a } }$ is configured with the same hyperparameters as OMKC.

All compared algorithms were implemented using the source code provided by their original authors. All algorithms are evaluated under the progressive validation framework [40], which is the standard protocol for online learning [41, 20]. This framework provides a realistic assessment of the sequential prediction performance of an online algorithm, unlike the traditional train-then-test evaluation used for batch methods. In this protocol, each dataset is presented as a data stream and fed into each algorithm sequentially. At each round t, the algorithm receives an instance $\mathbf { \boldsymbol { x } } _ { t }$ , predicts its relevant label set as $\hat { Y } _ { t }$ , and then receives the true relevant label set $Y _ { t } .$ Based on the discrepancy between $\hat { Y } _ { t }$ and $Y _ { t } ,$ standard multi-label classification metrics are incrementally updated. At the end of round $t ,$ the model is updated with the example $( x _ { t } , Y _ { t } )$ . This iterative test-then-train process continues until the entire stream is processed.

For hyperparameter tuning, we reserve an initial segment of each data stream (5% for all datasets). Once optimal hyperparameters are identified, this initial segment is also used for model initialization. Each model is then evaluated sequentially on the subsequent examples in this stream.

To ensure a thorough comparison across diferent performance aspects, we employ five example-based metrics—Precision (Psn), Recall (Rcal), F1-measure (F1), Hamming loss (Hl), and Ranking loss (Rl)—along with two label-based metrics, Macroaveraged F1 (MacroF1) and Micro-averaged F1 (MicroF1). These are defined as follows:

$$
\mathrm { P s n } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { | Y _ { i } \cap \hat { Y } _ { i } | } { | \hat { Y } _ { i } | } , \quad \mathrm { R c a l } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { | Y _ { i } \cap \hat { Y } _ { i } | } { | Y _ { i } | } , \quad \mathrm { F l } = \frac { 2 * \mathrm { P s n } * \mathrm { R c a l } } { \mathrm { P s n } + \mathrm { R c a l } } ,
$$

$$
\mathrm { H l } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { L } | \hat { Y } _ { i } \Delta Y _ { i } | , \quad \mathrm { R l } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \sum _ { j \in Y _ { i } , k \in \tilde { Y } _ { i } } \mathbb { 1 } [ h ( { x } _ { i } , j ) \le h ( { x } _ { i } , k ) ] } { | Y _ { i } | * | \bar { Y } _ { i } | } ,
$$

$$
\mathbf { M a c r o F } 1 = \frac { 1 } { L } \sum _ { j = 1 } ^ { L } \frac { 2 * t p _ { j } } { 2 * t p _ { j } + f p _ { j } + f n _ { j } } , \mathbf { M i c r o F } 1 = \frac { 2 * \sum _ { j = 1 } ^ { L } t p _ { j } } { \sum _ { j = 1 } ^ { L } ( 2 * t p _ { j } + f p _ { j } + f n _ { j } ) } ,
$$

where $\Delta$ is the symmetric diference between two sets; $h ( \boldsymbol { x } _ { i } , j )$ denotes the predicted score for label j given instance x<sub>i</sub>; $t p _ { j } , f p _ { j }$ and $f n _ { j }$ denote the counts of true positives, false positives and false negatives for label $j ,$ respectively.

To ensure reliable results, each algorithm is executed 20 times, each with a difer-

ently shufled dataset. Average metric values over the 20 runs are reported in Tables 3 and 4, with the best value on each metric in bold.

## 5.2.1. An ablation study on the label thresholding strategy

Based on the results of Tables 3 and 4, we first conduct a statistical comparison between OMKC and $\mathrm { O M K C _ { a } }$ using Wilcoxon signed-rank tests [42]. This comparison aims to validate our design choice of employing a fixed label thresholding strategy in the single-kernel multi-label model of OMKC. Table 5 reports the Wilcoxon signedrank test statistics $R ^ { + }$ and $R ^ { - }$ , where $R ^ { + }$ denotes the sum of ranks for the datasets on which OMKC outperforms $\mathrm { O M K C } _ { \mathrm { a } } ,$ and $R ^ { - }$ denotes the sum of ranks for the opposite. For a confidence level of $\alpha = 0 . 0 5$ and $N = 1 1$ datasets, the null hypothesis can be rejected if min{R<sup>+</sup> $R ^ { - } \} \leq 1 0$

As shown in Table 5, on Rcal, F1, MacroF1 and MicroF1, the obtained statistics allow us to reject the null hypothesis, indicating that OMKC performs significantly better than $\mathrm { O M K C _ { a } }$ on these metrics. For all other metrics, the null hypothesis cannot be rejected, suggesting comparable performance.

To sum up, OMKC shows superior overall performance compared to $\mathrm { O M K C _ { a } }$ This justifies our choice of employing a fixed label thresholding strategy—rather than an adaptive one—in the high-dimensional RFF-induced feature space.

## 5.2.2. Results analysis

According to Tables 3 and 4, we first compute the average rank across all 11 datasets in each metric for each compared algorithm, as presented in Table 6. These metric-wise ranks are then averaged over all metrics to obtain an overall rank for each algorithm, reported in the last column of Table 6.

From these results, we observe that OMKC achieves the highest overall rank, owing to its highest average rank in each metric. It is followed by OKS-OMC and FFLT, whose metric-wise ranks are mostly second or third. Next are PAML-I and ELM-OMLL, while OSML-ELM and ELM-O have the lowest overall ranks, suggesting their inferior overall performance.

Further, we notice that the three ELM-based methods are inferior to the kernelbased methods in terms of overall performance. The inferiority probably stems from the inherent design mechanism of ELM. Specifically, ELM is a feedforward neural network with a single hidden layer. Its input weights and biases of hidden nodes are randomly assigned once and never updated; only the weights of output nodes are incrementally learnt. This frozen hidden layer results in a nonlinear feature mapping that is fixed and lacks adaptability to the data. In contrast, the kernel-based methods can discover more efective feature mappings by searching for appropriate kernels.

Table 3: Online performance metrics [%] and standard derivation (in brackets) achieved by all compared algorithms on the first six datasets.
<table><tr><td rowspan=1 colspan=13>Dataset     Algorithm   Psn (↑)   Rcal (↑)   F1 (↑)  MacroF1 (↑)MicroF1 (↑)  HI (↓)     Rl (↓)</td></tr><tr><td rowspan=1 colspan=13>OSML-ELM60.09 (1.57)56.97 (1.65)58.48 (1.50)59.38 (1.69)61.28 (1.35)22.36 (0.69)19.69 (0.78)ELM-OMLL61.77 (0.76)74.90 (1.20)67.70 (0.76)65.98 (0.71)66.59 (0.71)23.26 (0.48)18.54 (0.45)ELM-O    31.89 (0.18)81.61 (0.22)45.86 (0.15)43.82 (0.06)46.22 (0.16)61.96 (0.27)53.28 (0.06)Emotions   PAML-I   62.87 (0.86)69.17 (1.14)65.86 (0.83)64.53 (0.55)66.23 (0.50)21.88 (0.42)17.96 (0.48)(D = 16d)   FFLT      64.35 (0.77)70.97 (0.81)67.49 (0.61)65.89 (0.59)67.43 (0.58)21.15 (0.41)17.31 (0.43)OKS-OMC 64.45 (0.76)69.68 (0.82)66.96 (0.50)65.35 (0.61)67.03 (0.52)21.11 (0.38)17.09 (0.53)OMKCa   64.84 (0.93)69.13 (0.73)66.91 (0.56)65.35 (0.44)67.11 (0.41)20.91 (0.35)17.07 (0.42)OMKC    64.66 (0.65)71.36 (0.80)67.84 (0.50)66.00 (0.46)67.60 (0.42)21.12 (0.31)17.21 (0.49)</td></tr><tr><td rowspan=1 colspan=5>OSML-ELM54.45 (2.24)51.16 (2.05)52.75 (2.13)</td><td rowspan=2 colspan=6>14.24 (1.75)63.08 (1.44)33.68 (1.11)59.90 (0.87)</td><td rowspan=2 colspan=2>1.65 (0.05)4.74 (0.49)3.18 (0.11) 4.80 (0.30)</td></tr><tr><td rowspan=1 colspan=3>ELM-OMLL56.52 (1.28)</td><td rowspan=1 colspan=2>87.04 (0.68)68.53 (0.99)</td></tr><tr><td rowspan=1 colspan=3>ELM-O    27.02 (0.00)</td><td rowspan=1 colspan=1>21.21(0.00)</td><td rowspan=1 colspan=1>23.76 (0.00)</td><td rowspan=2 colspan=6>0.95 (0.00)24.06 (0.01)17.58 (1.15)62.45 (0.53)</td><td rowspan=2 colspan=1>3.79 (0.00)1.63 (0.02)</td><td rowspan=2 colspan=1>24.96 (0.17)3.87 (0.13)</td></tr><tr><td rowspan=1 colspan=3>Medical   PAML-I   53.75 (0.80)</td><td rowspan=1 colspan=1>50.02 (0.74)</td><td rowspan=1 colspan=1>51.82 (0.75)</td></tr><tr><td rowspan=1 colspan=2>(D = 4d)   FFLT</td><td rowspan=1 colspan=1>59.90 (0.76)</td><td rowspan=1 colspan=1>60.83 (0.73)</td><td rowspan=1 colspan=1>60.36 (0.69)</td><td rowspan=1 colspan=6>20.60 (1.13)66.28 (0.57)</td><td rowspan=1 colspan=1>1.69 (0.04)</td><td rowspan=1 colspan=1>4.44 (0.17)</td></tr><tr><td rowspan=1 colspan=2>OKS-OMC</td><td rowspan=1 colspan=1>64.80 (0.61)</td><td rowspan=1 colspan=1>66.76 (0.61)</td><td rowspan=1 colspan=1>65.77 (0.56)</td><td rowspan=3 colspan=6>24.56 (1.01)70.80 (0.48)24.95 (1.00)71.06 (0.52)</td><td rowspan=2 colspan=1>1.50 (0.03)1.49 (0.03)</td><td rowspan=1 colspan=1>4.18 (0.16)</td></tr><tr><td rowspan=1 colspan=2>OMKCa</td><td rowspan=1 colspan=1>65.10 (0.65)</td><td rowspan=1 colspan=1>67.18 (0.67)</td><td rowspan=1 colspan=1>66.12 (0.62)</td><td rowspan=1 colspan=1>4.16 (0.17)</td></tr><tr><td rowspan=1 colspan=2>OMKC</td><td rowspan=1 colspan=1>67.17 (0.77)</td><td rowspan=1 colspan=1>70.88 (0.62)</td><td rowspan=1 colspan=1>68.98 (0.66)</td><td rowspan=1 colspan=2>26.27 (1.1</td><td rowspan=1 colspan=4>72.09 (0.55)</td><td rowspan=1 colspan=1>1.49 (0.03)</td><td rowspan=1 colspan=1>3.69 (0.19)</td></tr><tr><td rowspan=1 colspan=3>OSML-ELM62.25 (2.03)</td><td rowspan=1 colspan=1>62.62 (2.29)</td><td rowspan=1 colspan=1>62.43 (2.15)</td><td rowspan=1 colspan=2>38.95 (4</td><td rowspan=1 colspan=3>.82)</td><td rowspan=1 colspan=1>70.47 (1.57)</td><td rowspan=1 colspan=1>4.61 (0.18)</td><td rowspan=1 colspan=1>5.23 (0.48)</td></tr><tr><td rowspan=4 colspan=3>ELM-OMLL61.57 (1.27)ELM-0    28.11 (1.40)</td><td rowspan=1 colspan=1>80.68 (1.13)</td><td rowspan=1 colspan=1>69.84 (1.12)</td><td rowspan=1 colspan=2>58.50 (1.17)</td><td rowspan=1 colspan=2>1.17)</td><td rowspan=1 colspan=2>64.58 (0.92)</td><td rowspan=1 colspan=1>7.90 (0.25)</td><td rowspan=1 colspan=1>7.99 (0.53)</td></tr><tr><td rowspan=3 colspan=1>28.71 (3.45)</td><td rowspan=3 colspan=1>28.23 (0.26)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=3>3.80 (0.51)</td><td rowspan=2 colspan=3>27.36 (0.33)</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>12.90 (1.27)</td><td rowspan=1 colspan=1>27.46 (0.60)</td></tr><tr><td rowspan=1 colspan=2>PlantGO    PAML-I</td><td rowspan=1 colspan=1>62.98 (1.05)</td><td rowspan=1 colspan=1>63.67 (1.58)</td><td rowspan=1 colspan=1>63.32 (1.28)</td><td rowspan=1 colspan=3>44.27 (1.74)</td><td rowspan=1 colspan=3>69.40 (0.85)</td><td rowspan=1 colspan=1>4.92 (0.15)</td><td rowspan=1 colspan=1>5.86 (0.28)</td></tr><tr><td rowspan=1 colspan=3>(D = d)    FFLT      67.43 (0.75)</td><td rowspan=1 colspan=1>74.93 (0.84)</td><td rowspan=1 colspan=1>70.98 (0.74)</td><td rowspan=1 colspan=3>56.80 (1.01)</td><td rowspan=1 colspan=3>72.26 (0.78)</td><td rowspan=1 colspan=1>5.08 (0.17)</td><td rowspan=1 colspan=1>5.54 (0.20)</td></tr><tr><td rowspan=1 colspan=3>OKS-OMC 69.41 (0.62)</td><td rowspan=1 colspan=1>77.34 (0.67)</td><td rowspan=1 colspan=1>73.16 (0.59)</td><td rowspan=1 colspan=3>60.13 (0.97)</td><td rowspan=1 colspan=3>73.96 (0.69)</td><td rowspan=1 colspan=1>4.83 (0.16)</td><td rowspan=1 colspan=1>5.19 (0.23)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKCa</td><td rowspan=1 colspan=1>69.59 (0.69)</td><td rowspan=1 colspan=1>77.42 (0.56)</td><td rowspan=1 colspan=1>73.29 (0.58)</td><td rowspan=1 colspan=3>59.98 (0.77)</td><td rowspan=1 colspan=3>74.09 (0.68)</td><td rowspan=1 colspan=1>4.80 (0.15)</td><td rowspan=1 colspan=1>5.11 (0.24)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKC</td><td rowspan=1 colspan=1>69.43 (0.76)</td><td rowspan=1 colspan=1>78.37 (0.51)</td><td rowspan=1 colspan=1>73.63 (0.60)</td><td rowspan=1 colspan=3>60.97 (1.31)</td><td rowspan=1 colspan=3>73.85 (0.71)</td><td rowspan=1 colspan=1>4.92 (0.16)</td><td rowspan=1 colspan=1>5.00 (0.22)</td></tr><tr><td rowspan=1 colspan=2>OSML-ELM</td><td rowspan=1 colspan=1>12.28 (1.42)</td><td rowspan=1 colspan=1>12.15 (1.44)</td><td rowspan=1 colspan=1>12.21 (1.43)</td><td rowspan=1 colspan=3>7.39 (1.19)</td><td rowspan=1 colspan=3>18.88 (1.97)</td><td rowspan=1 colspan=1>8.96 (0.11)</td><td rowspan=1 colspan=1>21.03 (0.54)</td></tr><tr><td rowspan=1 colspan=2>ELM-OMLL</td><td rowspan=1 colspan=1>26.53 (1.11)</td><td rowspan=1 colspan=1>40.83 (1.60)</td><td rowspan=1 colspan=1>32.16 (1.28)</td><td rowspan=1 colspan=3>20.16 (0.98)</td><td rowspan=1 colspan=3>31.74 (1.19)</td><td rowspan=1 colspan=1>15.50 (0.34)</td><td rowspan=1 colspan=1>29.30 (0.76)</td></tr><tr><td rowspan=1 colspan=2>ELM-O</td><td rowspan=1 colspan=1>23.86 (0.40)</td><td rowspan=1 colspan=1>44.62 (0.29)</td><td rowspan=1 colspan=1>31.09 (0.41)</td><td rowspan=1 colspan=3>6.37 (0.10)</td><td rowspan=1 colspan=3>31.02 (0.53)</td><td rowspan=1 colspan=1>17.69 (0.14)</td><td rowspan=1 colspan=1>27.07 (0.50)</td></tr><tr><td rowspan=1 colspan=1>PlantPseAAC</td><td rowspan=1 colspan=1>PAML-I</td><td rowspan=1 colspan=1>27.27 (1.29)</td><td rowspan=1 colspan=1>44.84 (3.15)</td><td rowspan=1 colspan=1>33.88 (1.59)</td><td rowspan=1 colspan=3>17.85 (1.36)</td><td rowspan=1 colspan=3>34.29 (1.08)</td><td rowspan=1 colspan=1>15.17 (1.01)</td><td rowspan=1 colspan=1>21.50 (0.57)</td></tr><tr><td rowspan=1 colspan=1>(D = 8d)</td><td rowspan=1 colspan=1>FFLT</td><td rowspan=1 colspan=1>31.66 (0.60)</td><td rowspan=1 colspan=1>47.28 (1.23)</td><td rowspan=1 colspan=1>37.92 (0.65)</td><td rowspan=1 colspan=3>14.94 (0.72)</td><td rowspan=1 colspan=3>37.11 (0.61)</td><td rowspan=1 colspan=1>14.08 (0.29)</td><td rowspan=1 colspan=1>20.00 (0.39)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OKS-OMC</td><td rowspan=1 colspan=1>31.74 (0.50)</td><td rowspan=1 colspan=1>47.08 (1.14)</td><td rowspan=1 colspan=1>37.91 (0.57)</td><td rowspan=1 colspan=3>14.42 (0.66)</td><td rowspan=1 colspan=3>36.81 (0.52)</td><td rowspan=1 colspan=1>14.19 (0.28)</td><td rowspan=1 colspan=1>20.00 (0.37)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKCa</td><td rowspan=1 colspan=1>31.50 (0.60)</td><td rowspan=1 colspan=1>46.65 (1.14)</td><td rowspan=1 colspan=1>37.60 (0.69)</td><td rowspan=1 colspan=3>15.75 (0.78)</td><td rowspan=1 colspan=3>36.53 (0.66)</td><td rowspan=1 colspan=1>14.25 (0.27)</td><td rowspan=1 colspan=1>20.02 (0.35)</td></tr><tr><td rowspan=1 colspan=2>OMKC</td><td rowspan=1 colspan=1>31.39 (0.64)</td><td rowspan=1 colspan=1>48.72 (1.13)</td><td rowspan=1 colspan=1>38.18 (0.69)</td><td rowspan=1 colspan=3>17.10 (0.80)</td><td rowspan=1 colspan=3>37.24 (0.68)</td><td rowspan=1 colspan=1>14.45 (0.29)</td><td rowspan=1 colspan=1>20.02 (0.31)</td></tr><tr><td rowspan=1 colspan=2>OSML-ELM</td><td rowspan=1 colspan=1>64.38 (0.97)</td><td rowspan=1 colspan=1>42.67 (0.99)</td><td rowspan=1 colspan=1>51.32 (0.93)</td><td rowspan=1 colspan=3>9.67 (0.58)</td><td rowspan=1 colspan=3>51.29 (0.65)</td><td rowspan=1 colspan=1>4.91 (0.04)</td><td rowspan=1 colspan=1>10.63 (0.18)</td></tr><tr><td rowspan=1 colspan=2>ELM-OMLL</td><td rowspan=1 colspan=1>54.88 (0.59)</td><td rowspan=1 colspan=1>51.65 (0.58)</td><td rowspan=1 colspan=1>53.21 (0.50)</td><td rowspan=1 colspan=3>13.52 (0.59)</td><td rowspan=1 colspan=3>50.96 (0.46)</td><td rowspan=1 colspan=1>6.05 (0.07)</td><td rowspan=1 colspan=1>15.12 (0.34)</td></tr><tr><td rowspan=1 colspan=2>ELM-O</td><td rowspan=1 colspan=1>43.64 (0.03)</td><td rowspan=1 colspan=1>55.46 (0.01)</td><td rowspan=1 colspan=1>48.84 (0.02)</td><td rowspan=1 colspan=3>4.55 (0.00)</td><td rowspan=1 colspan=3>47.35 (0.03)</td><td rowspan=1 colspan=1>7.33 (0.00)</td><td rowspan=1 colspan=1>12.55 (0.10)</td></tr><tr><td rowspan=1 colspan=2>Enron     PAML-I</td><td rowspan=1 colspan=1>62.44 (0.77)</td><td rowspan=1 colspan=1>43.65 (0.66)</td><td rowspan=1 colspan=1>51.38 (0.64)</td><td rowspan=1 colspan=3>13.32 (0.38)</td><td rowspan=1 colspan=3>50.87 (0.51)</td><td rowspan=1 colspan=1>5.06 (0.04)</td><td rowspan=1 colspan=1>11.39 (0.21)</td></tr><tr><td rowspan=1 colspan=2>(D = 2d)    FFLT</td><td rowspan=1 colspan=1>60.76 (0.60)</td><td rowspan=1 colspan=1>58.84 (0.37)</td><td rowspan=1 colspan=1>59.78 (0.41)</td><td rowspan=1 colspan=3>19.64 (0.47)</td><td rowspan=1 colspan=3>57.94 (0.28)</td><td rowspan=1 colspan=1>5.27 (0.04)</td><td rowspan=1 colspan=1>10.37 (0.20)</td></tr><tr><td rowspan=1 colspan=2>OKS-OMC</td><td rowspan=1 colspan=1>61.79 (0.52)</td><td rowspan=1 colspan=1>57.35 (0.35)</td><td rowspan=1 colspan=1>59.49 (0.36)</td><td rowspan=1 colspan=6>18.02 (0.52)57.29 (0.27)</td><td rowspan=1 colspan=1>5.22 (0.04)</td><td rowspan=1 colspan=1>9.06 (0.13)</td></tr><tr><td rowspan=1 colspan=2>OMKCa</td><td rowspan=1 colspan=1>63.23 (0.47)</td><td rowspan=1 colspan=1>58.03 (0.48)</td><td rowspan=1 colspan=1>60.52 (0.39)</td><td rowspan=1 colspan=6>18.09 (0.48)58.13 (0.23)</td><td rowspan=1 colspan=1>5.09 (0.03)</td><td rowspan=1 colspan=1>8.63 (0.11)</td></tr><tr><td rowspan=1 colspan=2>OMKC</td><td rowspan=1 colspan=1>62.09 (0.46)</td><td rowspan=1 colspan=1>59.31 (0.39)</td><td rowspan=1 colspan=1>60.67 (0.36)</td><td rowspan=1 colspan=6>19.51 (0.50)58.62 (0.29)</td><td rowspan=1 colspan=1>5.15 (0.04)</td><td rowspan=1 colspan=1>9.88 (0.19)</td></tr><tr><td rowspan=1 colspan=3>OSML-ELM51.30 (0.84)</td><td rowspan=1 colspan=1>49.83 (0.95)</td><td rowspan=1 colspan=1>50.55 (0.85)</td><td rowspan=1 colspan=6>55.27 (0.65)55.22 (0.59)</td><td rowspan=1 colspan=1>18.95 (0.21)</td><td rowspan=1 colspan=1>19.17 (0.41)</td></tr><tr><td rowspan=1 colspan=3>ELM-OMLL59.07 (0.60)</td><td rowspan=1 colspan=1>72.61 (0.66)</td><td rowspan=1 colspan=1>65.15 (0.56)</td><td rowspan=1 colspan=5>61.69 (0.49)</td><td rowspan=1 colspan=1>61.34 (0.51)</td><td rowspan=2 colspan=1>21.80 (0.34)69.33 (0.04)</td><td rowspan=2 colspan=1>18.51 (0.38)51.14 (0.46)</td></tr><tr><td rowspan=1 colspan=3>ELM-O    24.56 (0.01)</td><td rowspan=1 colspan=1>88.06 (0.26)</td><td rowspan=1 colspan=1>38.40 (0.04)</td><td rowspan=1 colspan=5>38.21 (0.02)</td><td rowspan=1 colspan=1>38.40 (0.04)</td></tr><tr><td rowspan=1 colspan=3>Image    PAML-I   60.01 (0.69)</td><td rowspan=1 colspan=1>58.42 (0.71)</td><td rowspan=1 colspan=1>59.20 (0.69)</td><td rowspan=1 colspan=3>61.84 (0.45)</td><td rowspan=1 colspan=3>61.33 (0.48)</td><td rowspan=1 colspan=1>17.36 (0.20)</td><td rowspan=1 colspan=1>16.90 (0.19)</td></tr><tr><td rowspan=1 colspan=3>(D = 16d)   FFLT      63.57 (0.60)</td><td rowspan=1 colspan=1>69.56 (0.73)</td><td rowspan=1 colspan=1>(0.62)</td><td rowspan=1 colspan=6>64.78 (0.53)64.10 (0.52)</td><td rowspan=1 colspan=1>18.42 (0.26)</td><td rowspan=1 colspan=1>15.84 (0.25)</td></tr><tr><td rowspan=1 colspan=3>OKS-OMC 63.09 (0.66)</td><td rowspan=1 colspan=2>69.12 (0.92)65.97 (0.70)</td><td rowspan=3 colspan=8>15.67 (0.21)15.46 (0.21)65.17 (0.55)64.51 (0.55)17.73 (0.28)15.46 (0.23)</td></tr><tr><td rowspan=1 colspan=3>OMKCa   63.90 (0.56)</td><td rowspan=1 colspan=2>67.80 (0.71)65.79 (0.60)</td><td rowspan=1 colspan=6>64.79 (0.57) 64.11 (0.57)</td><td rowspan=1 colspan=1>17.88 (0.27)</td></tr><tr><td rowspan=1 colspan=5>OMKC    64.32 (0.62)68.21 (0.68)66.21 (0.59)</td></tr></table>

Table 4: Online performance metrics [%] and standard derivation (in brackets) achieved by all compared algorithms on the last five datasets.
<table><tr><td rowspan=1 colspan=9>Dataset      Algorithm    Psn (↑)   Rcal (↑)   F1 (↑)  MacroF1(↑)MicroF1 (↑)  Hl (↓)    Rl (↓)</td></tr><tr><td rowspan=1 colspan=9>OSML-ELM59.08 (0.55)59.49 (0.64)59.28 (0.57)67.52 (0.52)66.85 (0.50)10.35 (0.14)9.77 (0.25)</td></tr><tr><td rowspan=1 colspan=3>ELM-OMLL63.52 (0.53)</td><td rowspan=1 colspan=1>73.81 (0.79)</td><td rowspan=1 colspan=1>68.28 (0.60)</td><td rowspan=1 colspan=1>68.78 (0.45)</td><td rowspan=2 colspan=3>67.21 (0.44)12.69 (0.18)11.67 (0.30)30.21 (0.02)76.17 (0.17)49.96 (0.08)</td></tr><tr><td rowspan=1 colspan=2>ELM-O</td><td rowspan=1 colspan=1>18.10 (0.00)</td><td rowspan=1 colspan=1>91.93 (0.25)</td><td rowspan=1 colspan=1>30.24 (0.02)</td><td rowspan=1 colspan=1>30.06 (0.03)</td></tr><tr><td rowspan=1 colspan=2>Scene      PAML-I</td><td rowspan=1 colspan=1>67.76 (0.49)</td><td rowspan=1 colspan=1>68.52 (0.55)</td><td rowspan=1 colspan=1>68.13 (0.51)</td><td rowspan=1 colspan=1>74.08 (0.41)</td><td rowspan=1 colspan=3>73.30 (0.41)8.75 (0.13) 7.36 (0.17)</td></tr><tr><td rowspan=1 colspan=2>(D = 16d)     FFLT</td><td rowspan=1 colspan=1>72.23 (0.46)</td><td rowspan=1 colspan=1>79.92 (0.40)</td><td rowspan=1 colspan=1>75.88 (0.40)</td><td rowspan=1 colspan=1>76.46 (0.37)</td><td rowspan=2 colspan=2>74.89 (0.36)9.42 (0.14)74.51 (0.36)9.59 (0.16)</td><td rowspan=2 colspan=1>7.08 (0.17)6.92 (0.18)</td></tr><tr><td rowspan=1 colspan=2>OKS-OMC</td><td rowspan=1 colspan=1>71.60 (0.49)</td><td rowspan=1 colspan=1>79.74 (0.41)</td><td rowspan=1 colspan=1>75.45(0.38)</td><td rowspan=1 colspan=1>76.08(0.35)</td></tr><tr><td rowspan=1 colspan=2>OMKC8</td><td rowspan=1 colspan=1>72.35 (0.37)</td><td rowspan=1 colspan=1>79.05 (0.36)</td><td rowspan=1 colspan=1>75.55 (0.30)</td><td rowspan=1 colspan=1>76.73 (0.28)</td><td rowspan=1 colspan=1>75.28 (0.30)</td><td rowspan=1 colspan=1>9.12 (0.14)</td><td rowspan=1 colspan=1>6.71 (0.16)</td></tr><tr><td rowspan=1 colspan=2>OMKC</td><td rowspan=1 colspan=1>72.93 (0.32)</td><td rowspan=1 colspan=1>79.69 (0.34)</td><td rowspan=1 colspan=1>76.16 (0.31)</td><td rowspan=1 colspan=1>77.06 (0.34)</td><td rowspan=1 colspan=1>75.60 (0.33)</td><td rowspan=1 colspan=1>9.04 (0.13)</td><td rowspan=1 colspan=1>6.86 (0.16)</td></tr><tr><td rowspan=1 colspan=2>OSML-ELM</td><td rowspan=1 colspan=1>66.65 (0.35)</td><td rowspan=1 colspan=1>57.58(0.36)</td><td rowspan=1 colspan=1>61.79(0.28)</td><td rowspan=1 colspan=1>36.98 (0.46)</td><td rowspan=1 colspan=1>61.32 (0.17)</td><td rowspan=1 colspan=1>21.69 (0.12)</td><td rowspan=1 colspan=1>19.62 (0.15)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-OMLL</td><td rowspan=1 colspan=1>59.68 (0.27)</td><td rowspan=1 colspan=1>74.01 (0.32)</td><td rowspan=1 colspan=1>66.07(0.23)</td><td rowspan=1 colspan=1>41.78(0.47)</td><td rowspan=1 colspan=1>64.73 (0.19)</td><td rowspan=1 colspan=1>24.04 (0.16)</td><td rowspan=1 colspan=1>18.20 (0.15)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-O</td><td rowspan=1 colspan=1>51.05 (0.06)</td><td rowspan=1 colspan=1>69.67 (0.29)</td><td rowspan=1 colspan=1>58.92 (0.07)</td><td rowspan=1 colspan=1>29.22(0.20)</td><td rowspan=1 colspan=1>58.36 (0.05)</td><td rowspan=1 colspan=1>29.53 (0.03)</td><td rowspan=1 colspan=1>21.32 (0.03)</td></tr><tr><td rowspan=1 colspan=1>Yeast</td><td rowspan=1 colspan=1>PAML-I</td><td rowspan=1 colspan=1>63.04 (0.38)</td><td rowspan=1 colspan=1>70.98 (0.81)</td><td rowspan=1 colspan=1>66.77 (0.33)</td><td rowspan=1 colspan=1>43.13 (0.41)</td><td rowspan=1 colspan=1>65.18 (0.24)</td><td rowspan=1 colspan=1>22.61 (0.18)</td><td rowspan=1 colspan=1>18.39 (0.22)</td></tr><tr><td rowspan=1 colspan=1>(D = 32d)</td><td rowspan=1 colspan=1>FFLT</td><td rowspan=1 colspan=1>64.88 (0.25)</td><td rowspan=1 colspan=1>68.17 (0.27)</td><td rowspan=1 colspan=1>66.49 (0.20)</td><td rowspan=1 colspan=1>42.58 (0.23)</td><td rowspan=1 colspan=1>65.59 (0.14)</td><td rowspan=1 colspan=1>21.40 (0.10)</td><td rowspan=1 colspan=1>18.46 (0.16)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OKS-OMC</td><td rowspan=1 colspan=1>66.72 (0.29)</td><td rowspan=1 colspan=1>67.16 (0.27)</td><td rowspan=1 colspan=1>66.94 (0.22)</td><td rowspan=1 colspan=1>41.53 (0.35)</td><td rowspan=1 colspan=1>66.08 (0.19)</td><td rowspan=1 colspan=1>20.59 (0.15)</td><td rowspan=1 colspan=1>17.29 (0.13)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKCa</td><td rowspan=1 colspan=1>67.17 (0.28)</td><td rowspan=1 colspan=1>67.29(0.26)</td><td rowspan=1 colspan=1>67.23 (0.20)</td><td rowspan=1 colspan=1>41.62 (0.38)</td><td rowspan=1 colspan=1>66.41 (0.22)</td><td rowspan=1 colspan=1>20.32 (0.14)</td><td rowspan=1 colspan=1>16.98 (0.14)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKC</td><td rowspan=1 colspan=1>67.18 (0.28)</td><td rowspan=1 colspan=1>66.93(0.26)</td><td rowspan=1 colspan=1>67.05(0.21)</td><td rowspan=1 colspan=1>41.51(0.37)</td><td rowspan=1 colspan=1>66.32 (0.20)</td><td rowspan=1 colspan=1>20.31 (0.13)</td><td rowspan=1 colspan=1>17.53(0.16)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OSML-ELM</td><td rowspan=1 colspan=1>71.19 (0.69)</td><td rowspan=1 colspan=1>73.62 (0.86)</td><td rowspan=1 colspan=1>72.39 (0.75)</td><td rowspan=1 colspan=1>47.41(1.79)</td><td rowspan=1 colspan=1>73.86 (0.52)</td><td rowspan=1 colspan=1>4.21 (0.07)</td><td rowspan=1 colspan=1>3.28 (0.12)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-OMLL</td><td rowspan=1 colspan=1>63.51 (0.57)</td><td rowspan=1 colspan=1>80.16 (0.49)</td><td rowspan=1 colspan=1>70.87 (0.52)</td><td rowspan=1 colspan=1>56.13 (1.23)</td><td rowspan=1 colspan=1>65.91 (0.50)</td><td rowspan=1 colspan=1>6.81 (0.12)</td><td rowspan=1 colspan=1>7.31 (0.33)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-O</td><td rowspan=1 colspan=1>29.49 (0.03)</td><td rowspan=1 colspan=1>48.02 (0.04)</td><td rowspan=1 colspan=1>36.54 (0.03)</td><td rowspan=1 colspan=1>6.49 (0.01)</td><td rowspan=1 colspan=1>37.05 (0.03)</td><td rowspan=1 colspan=1>14.32 (0.00)</td><td rowspan=1 colspan=1>20.46 (0.35)</td></tr><tr><td rowspan=1 colspan=1>HumanGo</td><td rowspan=1 colspan=1>PAML-I</td><td rowspan=1 colspan=1>69.70 (0.36)</td><td rowspan=1 colspan=1>70.53 (0.44)</td><td rowspan=1 colspan=1>70.11 (0.39)</td><td rowspan=1 colspan=1>48.90 </td><td rowspan=1 colspan=1>72.90 (0.32)</td><td rowspan=1 colspan=1>4.22 (0.05)</td><td rowspan=1 colspan=1>3.31 (0.10)</td></tr><tr><td rowspan=1 colspan=1>(D = d)</td><td rowspan=1 colspan=1>FFLT</td><td rowspan=1 colspan=1>74.89 (0.31)</td><td rowspan=1 colspan=1>84.53(0.27)</td><td rowspan=1 colspan=1>79.42 (0.25)</td><td rowspan=1 colspan=1>61.83(0.79)</td><td rowspan=1 colspan=1>76.87 (0.23)</td><td rowspan=1 colspan=1>4.18 (0.05)</td><td rowspan=1 colspan=1>2.98 (0.08)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OKS-OMC</td><td rowspan=1 colspan=1>74.73 (0.33)</td><td rowspan=1 colspan=1>83.23 (0.36)</td><td rowspan=1 colspan=1>78.75 (0.31)</td><td rowspan=1 colspan=1>60.51 (0.68)</td><td rowspan=1 colspan=1>76.61 (0.31)</td><td rowspan=1 colspan=1>4.17 (0.06)</td><td rowspan=1 colspan=1>2.98 (0.08)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKCa</td><td rowspan=1 colspan=1>74.82 (0.37)</td><td rowspan=1 colspan=1>83.04 (0.41)</td><td rowspan=1 colspan=1>78.72 (0.35)</td><td rowspan=1 colspan=1>60.08 (0.86)</td><td rowspan=1 colspan=1>76.75 (0.28)</td><td rowspan=1 colspan=1>4.13 (0.05)</td><td rowspan=1 colspan=1>2.98 (0.07)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKC</td><td rowspan=1 colspan=1>74.96 (0.29)</td><td rowspan=1 colspan=1>84.29 (0.32)</td><td rowspan=1 colspan=1>79.35 (0.26)</td><td rowspan=1 colspan=1>61.33 (0.72)</td><td rowspan=1 colspan=1>76.93 (0.20)</td><td rowspan=1 colspan=1>4.16 (0.04)</td><td rowspan=1 colspan=1>2.94 (0.09)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OSML-ELM</td><td rowspan=1 colspan=1>19.74 (0.90)</td><td rowspan=1 colspan=1>18.84 (0.97)</td><td rowspan=1 colspan=1>19.27 (0.92)</td><td rowspan=1 colspan=1>9.29 (0.52)</td><td rowspan=1 colspan=1>26.48 (1.06)</td><td rowspan=1 colspan=1>8.48 (0.07)</td><td rowspan=1 colspan=1>17.95 (0.33)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-OMLL</td><td rowspan=1 colspan=1>33.84 (0.72)</td><td rowspan=1 colspan=1>43.38 (0.78)</td><td rowspan=1 colspan=1>38.02 (0.71)</td><td rowspan=1 colspan=1>16.17 (0.47)</td><td rowspan=1 colspan=1>39.29 (0.64)</td><td rowspan=1 colspan=1>11.04 (0.13)</td><td rowspan=1 colspan=1>20.42 (0.38)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-O</td><td rowspan=1 colspan=1>29.83 (0.07)</td><td rowspan=1 colspan=1>47.89 (0.43)</td><td rowspan=1 colspan=1>36.76 (0.08)</td><td rowspan=1 colspan=1>6.73 (0.05)</td><td rowspan=1 colspan=1>37.06 (0.07)</td><td rowspan=1 colspan=1>14.20 (0.07)</td><td rowspan=1 colspan=1>19.68 (0.01)</td></tr><tr><td rowspan=1 colspan=1>HumanPseAAC</td><td rowspan=1 colspan=1>PAML-I</td><td rowspan=1 colspan=1>26.88 (0.82)</td><td rowspan=1 colspan=1>27.14 (1.04)</td><td rowspan=1 colspan=1>27.01(0.91)</td><td rowspan=1 colspan=1>12.44 (0.54)</td><td rowspan=1 colspan=1>33.22(0.75)</td><td rowspan=1 colspan=1>8.90 (0.08)</td><td rowspan=1 colspan=1>17.91 (0.23)</td></tr><tr><td rowspan=1 colspan=1>(D = 8d)</td><td rowspan=1 colspan=1>FFLT</td><td rowspan=1 colspan=1>36.53 (0.29)</td><td rowspan=1 colspan=1>57.94 (0.52)</td><td rowspan=1 colspan=1>44.81 (0.29)</td><td rowspan=1 colspan=1>19.13 (0.31)</td><td rowspan=1 colspan=1>44.71 (0.27)</td><td rowspan=1 colspan=1>12.03 (0.08)</td><td rowspan=1 colspan=1>15.74 (0.14)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OKS-OMC</td><td rowspan=1 colspan=1>36.82 (0.43)</td><td rowspan=1 colspan=1>57.49 (0.59)</td><td rowspan=1 colspan=1>44.89 (0.42)</td><td rowspan=1 colspan=1>19.04 (0.37)</td><td rowspan=1 colspan=1>44.38 (0.34)</td><td rowspan=1 colspan=1>12.10 (0.11)</td><td rowspan=1 colspan=1>15.14 (0.14)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>OMKCa</td><td rowspan=1 colspan=1>36.32 (0.44)</td><td rowspan=1 colspan=1>56.68 (0.64)</td><td rowspan=1 colspan=1>44.27(0.48)</td><td rowspan=1 colspan=1>18.97 (0.47)</td><td rowspan=1 colspan=1>43.37(0.34)</td><td rowspan=1 colspan=1>12.42 (0.10)</td><td rowspan=1 colspan=1>15.46 (0.17)</td></tr><tr><td rowspan=1 colspan=2>OMKC</td><td rowspan=1 colspan=1>36.71 (0.32)</td><td rowspan=1 colspan=1>58.49(0.54)</td><td rowspan=1 colspan=1>45.11(0.34)</td><td rowspan=1 colspan=1>19.46 (0.33)</td><td rowspan=1 colspan=1>44.85(0.29)</td><td rowspan=1 colspan=1>12.06 (0.09)</td><td rowspan=1 colspan=1>15.66 (0.17)</td></tr><tr><td rowspan=1 colspan=3>OSML-ELM73.07 (0.10)</td><td rowspan=1 colspan=1>63.98 (0.21)</td><td rowspan=1 colspan=1>68.22 (0.15)</td><td rowspan=1 colspan=1>48.07 (0.39)</td><td rowspan=1 colspan=1>66.22 (0.12)</td><td rowspan=1 colspan=1>6.10 (0.02)</td><td rowspan=1 colspan=1>5.32 (0.03)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-OMLL</td><td rowspan=1 colspan=1>68.47 (0.17)</td><td rowspan=1 colspan=1>75.12 (0.19)</td><td rowspan=1 colspan=1>71.64 (0.12)</td><td rowspan=1 colspan=1>47.84 (0.50)</td><td rowspan=1 colspan=1>68.80 (0.10)</td><td rowspan=1 colspan=1>6.48 (0.02)</td><td rowspan=1 colspan=1>5.30 (0.04)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ELM-O</td><td rowspan=1 colspan=1>39.95 (0.02)</td><td rowspan=1 colspan=1>57.37 (0.01)</td><td rowspan=1 colspan=1>47.10 (0.02)</td><td rowspan=1 colspan=1>7.61 (0.00)</td><td rowspan=1 colspan=1>45.93 (0.02)</td><td rowspan=1 colspan=1>12.83 (0.00)</td><td rowspan=1 colspan=1>16.70 (0.01)</td></tr><tr><td rowspan=1 colspan=1>Tmc2007-500</td><td rowspan=1 colspan=1>PAML-I</td><td rowspan=1 colspan=1>70.28 (0.15)</td><td rowspan=1 colspan=1>69.43 (0.14)</td><td rowspan=1 colspan=1>69.85 (0.13)</td><td rowspan=1 colspan=1>59.19 (0.24)</td><td rowspan=1 colspan=1>67.72 (0.12)</td><td rowspan=1 colspan=1>6.33 (0.02)</td><td rowspan=1 colspan=1>4.82 (0.03)</td></tr><tr><td rowspan=1 colspan=2>(D = 8d)   FFLT</td><td rowspan=1 colspan=1>68.85 (0.12)</td><td rowspan=1 colspan=1>81.41 (0.09)</td><td rowspan=1 colspan=1>74.60 (0.10)</td><td rowspan=1 colspan=1>63.46 (0.18)</td><td rowspan=4 colspan=3>6.28 (0.02) 4.46 (0.02)6.14 (0.01) 3.89 (0.02)72.93 (0.08)5.74 (0.02) 3.56 (0.02)65.91 (0.20)73.69 (0.07)5.68 (0.02) 3.95 (0.02)</td></tr><tr><td rowspan=1 colspan=2>OKS-OMC</td><td rowspan=1 colspan=1>70.46 (0.08)</td><td rowspan=1 colspan=1>79.18 (0.18)</td><td rowspan=1 colspan=1>74.57 (0.10)</td><td rowspan=1 colspan=1>62.41 (0.21)</td><td rowspan=1 colspan=1>71.28 (0.09)</td></tr><tr><td rowspan=1 colspan=2>OMKCa</td><td rowspan=1 colspan=1>72.25 (0.12)</td><td rowspan=1 colspan=1>80.35 (0.12)</td><td rowspan=1 colspan=1>76.08 (0.10)</td><td rowspan=1 colspan=1>63.92 (0.21)</td></tr><tr><td rowspan=1 colspan=3>OMKC    71.47 (0.09)</td><td rowspan=1 colspan=1>82.23 (0.09)</td><td rowspan=1 colspan=1>76.48 (0.07)</td><td></td></tr></table>

<sup>1</sup> ↑ indicates the higher the better, while ↓ means the lower the better.

Table 5: Rank statistics of Wilcoxon signed-rank tests on all evaluation metrics.
<table><tr><td>Statistics</td><td>Psn Rcal F1 MacroF1 MicroF1 Hl</td><td></td><td></td><td></td><td></td><td>Rl</td></tr><tr><td>R+</td><td>36</td><td>65</td><td>64</td><td>65</td><td>62</td><td>3217.5</td></tr><tr><td>R⁻</td><td>30</td><td>1</td><td>2</td><td>1</td><td>4</td><td>34 48.5</td></tr><tr><td>min{R+, R−}</td><td>30</td><td>1</td><td>2</td><td>1</td><td>4</td><td>32 17.5</td></tr></table>

<table><tr><td>Table 6: Average ranks per metric and overall ranks over all metrics.</td></tr><tr><td>Algorithm Psn Rcal F1 MacroF1 MicroF1 Hl Rl Rank</td></tr><tr><td>OSML-ELM 4.64 6.45 5.91 5.91 5.45 3.004.91</td></tr><tr><td>5.18 ELM-OMLL 5.27 3.27 3.91 3.45 4.91 5.73 5.73 4.61</td></tr><tr><td>7.00 6.73</td></tr><tr><td>ELM-O 6.734.27 6.73 7.006.73 6.45</td></tr><tr><td>PAML-I 4.36 5.27 5.00 4.09 4.91 3.32 4.18 4.45</td></tr><tr><td>FFLT 3.092.732.45 2.45 2.27 3.64 3.18 2.83</td></tr><tr><td>OKS-OMC 2.36 3.45 2.82 3.27 2.64 3.001.64 2.74</td></tr><tr><td>OMKC 1.55 2.55  1.18 1.82 1.09 2.321.64 1.73</td></tr></table>

Among the kernel-based methods, the single-kernel method PAML-I performs the worst. This is due to its update rule, which updates only two label scoring predictors per round, resulting in insuficient updates for the remaining predictors. In contrast, the single-kernel FFLT updates all label scoring predictors per round, leading to better performance. However, both single-kernel methods are outperformed by the multikernel methods OKS-OMC and OMKC.

Next, we focus on comparing OMKC with its most relevant and competitive baselines: FFLT and OKS-OMC. In order to statistically compare them, we conducted Friedman tests followed by post-hoc Bonferroni-Dunn tests [42]. Table 7 lists the average ranks of the three algorithms across 11 datasets and the corresponding Friedman

statistic $F _ { F }$ for each metric.

Table 7: The average ranks of FFLT, OKS-OMC and OMKC and the Friedman statistic $F _ { F }$ in terms of each metric.
<table><tr><td>Algorithm</td><td>Psn Rcal</td><td>F1</td><td>MacroF1</td><td>MicroF1</td><td>Hl</td><td>Rl</td></tr><tr><td>FFLT</td><td>2.636 1.818</td><td>2.182</td><td>1.909</td><td>2.273</td><td>2.455</td><td>2.818</td></tr><tr><td></td><td>OKS-OMC 2.091 2.545</td><td>2.636</td><td>2.727</td><td>2.636</td><td>2.091</td><td>1.545</td></tr><tr><td>OMKC</td><td>1.273 1.636</td><td>1.182</td><td>1.364</td><td>1.091</td><td>1.455</td><td>1.636</td></tr><tr><td> $F _ { F }$  statistic</td><td>8.906 3.011</td><td>12.407</td><td>8.906</td><td>18.810</td><td>3.444</td><td>10.167</td></tr></table>

Given three algorithms and 11 datasets, the $F _ { F }$ statistic follows an F-distribution with 2 and 20 degrees of freedom. The critical value of F(2, 20) for a significance level $\alpha = 0 . 0 5$ is 3.493. As shown in Table 7, the $F _ { F }$ values for Psn, F1, MacroF1, MicroF1 and Rl exceed this critical value. Therefore, for these metrics, we can reject the null hypothesis that these algorithms perform equivalently. In contrast, for Rcal and Hl, the $F _ { F }$ values are below the critical value. This indicates no statistically significant diference on these metrics.

Next, on the metrics showing significant Friedman test results, we applied the Bonferroni-Dunn tests [42] to compare OMKC against the other two algorithms. The critical diference (CD) for $\alpha = 0 . 0 5$ is 0.956. A diference in average ranks that exceeds this CD indicates a statistically significant diference. Figure 2 presents these comparisons visually using CD diagrams. The diagrams plot the average ranks of algorithms on the top axis, and the interval of one CD to either side of OMKC’s average rank is highlighted in red. An algorithm is deemed statistically diferent from OMKC if its rank falls outside the red interval.

According to Table 7 and Figure 2, we can draw the following observations:

• OMKC vs FFLT: OMKC demonstrates statistically significant superiority over FFLT on four key metrics: Psn, F1, MicroF1, and Rl. While OMKC also achieves higher average ranks on Rcal, MacroF1, and Hl, these diferences are not statistically significant, indicating comparable performance between the two algorithms on these metrics. In summary, the evidence confirms that OMKC delivers superior overall performance. This performance diference stems from their fundamental architectural diferences. FFLT is constrained to a single, preselected kernel, making its performance highly sensitive to that choice and vulnerable to suboptimal kernel selection. In contrast, OMKC optimally combines multiple kernel classifiers, which allows it to mitigate the risk of poor kernel choice and achieve more accurate predictions.

![](images/9eeabb54c5759270cdc6008d4c27fdd7b73f3e6ed54430207b84474049423919.jpg)  
Figure 2: Statistical comparison of OMKC with two baseline methods using the Bonferroni–Dunn test.

• OMKC vs OKS-OMC: OMKC significantly outperforms OKS-OMC on F1, MacroF1 and MicroF1, with a near-significant edge on Psn. On Rcal, Hl and Rl, both algorithms are comparable. Overall, OMKC is superior to OKS-OMC. Its advantage can be attributed to two key innovations in its design. First, unlike OKS-OMC, OMKC employs a fixed label thresholding strategy for its singlekernel models. This design is advantageous in high-dimensional feature spaces induced by RFF techniques, where adaptively learning a label thresholding predictor carries the risk of overfitting. Second, OMKC optimally combines multiple single-kernel models to leverage complementary information. In contrast, OKS-OMC is designed to converge to the optimal single-kernel model.

In summary, empirical results confirm that OMKC achieves superior overall performance compared to existing online multi-label classification methods.

## 5.3. Sensitivity analysis

In OMKC, the hyperparameters $\lambda , \eta ,$ and $\beta$ serve the same roles as in OKS-OMC. Since their sensitivity is theoretically identical to that in OKS-OMC, we omit a detailed analysis to avoid redundancy. The number of RFF components, D, governs the trade-of between the approximation accuracy of kernel functions and the computational (space and time) cost of OMKC. A larger D yields a better approximation, leading to improved performance, but at the expense of increased computation cost. We recommend selecting the largest D possible while keeping the computational cost within an acceptable limit.

The global fixed threshold γ controls both the updating and prediction of the multikernel multi-label model $( \mathcal W , \pmb { \mu } )$ in OMKC. We evaluate OMKC’s sensitivity to this parameter using three randomly chosen datasets and five key metrics. Fig. 3 illustrates how the online performance metrics of OMKC evolve as γ varies within the interval [−10 10], while holding all other hyperparameters fixed. From these figures, we observe that, across all three datasets and nearly all metrics, larger and smaller values of $\gamma$ lead to worse performance. The optimal $\gamma$ is consistently close to zero. Based on this observation, we recommend setting $\gamma = 0$ as the default value.

## 6. Conclusions and limitations

In this paper, we have introduced OMKC, an eficient online multi-kernel learning method for large-scale multi-label classification. By integrating advanced RFF techniques, OMKC efectively circumvents the curse of kernelization while maintaining high predictive performance. OMKC further improves robustness in high-dimensional RFF-induced feature spaces by employing a fixed label thresholding strategy in its single-kernel multi-label models. In OMKC, the combination coeficients of classifiers are updated by directly minimizing the joint optimization objective function, while each kernel classifier is updated by minimizing an upper bound of this objective. This strategy enables rapid convergence to a high-quality solution in online learning scenarios. We also provide a theoretical guarantee, proving that OMKC asymptotically converges to the optimal fixed convex combination of the best fixed single-kernel multi-label models in hindsight. Empirically, the superiority of OMKC was rigorously validated. Ablation studies confirm the critical role of OMKC’s coeficient update rule, especially when kernel approximation errors are large, and also show that—when coupled with this update rule—the fixed label thresholding strategy generally surpasses the adaptive one. Moreover, evaluations across 11 datasets using seven metrics show OMKC statistically significantly outperforms state-of-the-art online multi-label classification methods in overall online performance.

![](images/7a8db0c89e6fead6cfb8bebc73370167f5c85a4e49303bbaa06e3701abd496f2.jpg)

![](images/9f812b0bfc6fb60f8833366c036f12d4f5daa3317c1044da6d88bd9a91d06623.jpg)

![](images/0d3cdcde258cb7c316f8ab5c170008ed182ed39c0cc56a886029ce84e5dfa406.jpg)

![](images/c24c3fa59c981e5784d381de1f21d7e810548116bc1d73fb6364450c6d78f206.jpg)  
(a)

![](images/3f99b818cd7649801a31407ccd63a34a0dec2cce76d083951e7f089f116edb19.jpg)  
(b)

![](images/602c3e3928cdb039a91641a8730c018b459ad689bad210466a609dcd63278b50.jpg)  
(c)

![](images/c0ffaaab3761aba46fad03eb6ad3c19233ecea40e54bafa5689c6732d38a6ec7.jpg)  
(d)

![](images/f7da08b412663abd03c9d11db7dee9839b1c9c17dac711bc65c18bd07d85be05.jpg)  
(e)

![](images/2e6a95894f82e5ac495ee14aa752278c288c49baca7b5259e63f8206eb9e3425.jpg)  
(f)  
Figure 3: Evolution of online performance metrics of OMKC as the fixed threshold γ varies within the interval [−10 10].

Currently, this paper focuses primarily on stationary multi-label data streams; consequently, both the proposed OMKC and its theoretical analysis are confined to stationary settings. A promising direction for future research is to extend OMKC to handle non-stationary data streams [43], where algorithms must adapt to concept drift over time. Another limitation lies in the sensitivity of OMKC to the choice of the predefined kernel pool. An ill-suited or overly restricted set of candidate kernels can significantly degrade performance, which underscores that selecting an appropriate kernel pool remains a critical consideration in practice.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China [grant number 61906165].

## References

[1] X. Li, B. You, Q. Peng, S. Feng, Dual-view graph convolutional network for multi-label text classification, Appl. Intell. 54 (19) (2024) 9363–9380.

[2] Y. Yan, F. Liu, K. Liu, W. Xu, X. Zhuang, Label-specific multi-label text classification based on dynamic graph convolutional networks, Soft Comput. 29 (3) (2025) 1897–1909.

[3] Y. Chen, C. Zou, J. Chen, Label-aware graph representation learning for multilabel image classification, Neurocomputing 492 (2022) 50–61.

[4] S. Zhang, Z. Liao, J. Li, L. Hu, J. Zhang, A progressive attention network with transformer for multi-label image recognition, Pattern Recognit. 172 (2026) 112439.

[5] P. K. Jain, R. Pamula, E. A. Yekun, A multi-label ensemble predicting model to service recommendation from social media contents, J. Supercomput. 78 (4) (2022) 5203–5220.

[6] J. Wu, W. Gan, H. Lu, P. S. Yu, Graph contrastive learning on multi-label classification for recommendations, ACM Trans. Intell. Syst. Technol. 16 (4) (2025) 77:1–77:19.

[7] X. Du, J. Hu, Deep multi-label joint learning for RNA and dna-binding proteins prediction, IEEE ACM Trans. Comput. Biol. Bioinform. 20 (1) (2023) 307–320.

[8] R. E. Armah-Sekum, S. Szedmák, J. Rousu, Protein function prediction through multi-view multi-label latent tensor reconstruction, BMC Bioinform. 25 (1) (2024) 174.

[9] W. Liu, H. Wang, X. Shen, I. W. Tsang, The emerging trends of multi-label learning, IEEE Trans. Pattern Anal. Mach. Intell. 44 (11) (2022) 7955–7974.

[10] S. C. H. Hoi, D. Sahoo, J. Lu, P. Zhao, Online learning: A comprehensive survey, Neurocomputing 459 (2021) 249–289.

[11] T. Zhai, H. Wang, Online passive-aggressive multilabel classification algorithms, IEEE Trans. Neural Networks Learn. Syst. 34 (12) (2023) 10116–10129.

[12] B. Schölkopf, A. J. Smola, Learning with Kernels: support vector machines, regularization, optimization, and beyond, Adaptive computation and machine learning series, MIT Press, 2002.

[13] T. Zhai, H. Wang, H. Tang, Joint optimization of scoring and thresholding models for online multi-label classification, Pattern Recognit. 136 (2023) 109167. doi:10.1016/j.patcog.2022.109167.

[14] T. Zhai, K. Hu, Optimizing margin distribution for online multi-label classification, Evol. Syst. 15 (3) (2024) 1033–1042.

[15] T. Yang, M. Mahdavi, R. Jin, J. Yi, S. C. H. Hoi, Online kernel selection: Algorithms and evaluations, in: Proceedings of the 26th AAAI Conference on Artificial Intelligence (AAAI), 2012, pp. 1197–1203.

[16] J. Li, S. Liao, Improved regret bounds for online kernel selection under bandit feedback, in: Machine Learning and Knowledge Discovery in Databases - European Conference, Vol. 13716 of Lecture Notes in Computer Science, 2022, pp. 333–348.

[17] X. Zhang, S. Liao, Online kernel selection via incremental sketched kernel alignment, in: Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI), 2018, pp. 3118–3124.

[18] S. C. H. Hoi, R. Jin, P. Zhao, T. Yang, Online multiple kernel classification, Mach. Learn. 90 (2) (2013) 289–316.

[19] Y. Shen, T. Chen, G. B. Giannakis, Random feature-based online multi-kernel learning in environments with unknown dynamics, J. Mach. Learn. Res. 20 (2019) 22:1–22:36.

[20] E. Hazan, Introduction to online convex optimization, Found. Trends Optim. 2 (3-4) (2016) 157–325.

[21] S. Hong, Online multikernel learning method via online biconvex optimization, IEEE Trans. Neural Networks Learn. Syst. 35 (11) (2024) 16630–16643.

[22] P. M. Ghari, Y. Shen, Graph-aided online multi-kernel learning, J. Mach. Learn. Res. 24 (2023) 21:1–21:44.

[23] T. D. Nguyen, T. Le, H. Bui, D. Q. Phung, Large-scale online kernel learning with random feature reparameterization, in: Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI), 2017, pp. 2543–2549.

[24] A. Rahimi, B. Recht, Random features for large-scale kernel machines, in: Advances in Neural Information Processing Systems, 2007, pp. 1177–1184.

[25] J. Zhang, H. Ning, Online kernel classification with adjustable bandwidth using control-based learning approach, Pattern Recognit. 108 (2020) 107566.

[26] J. Zhang, H. Ning, X. Jing, T. Tian, Online kernel learning with adaptive bandwidth by optimal control approach, IEEE Trans. Neural Networks Learn. Syst. 32 (5) (2021) 1920–1934.

[27] T. Zhai, W. Liu, Online kernel selection for online multi-label classification, Neural Networks 192 (2025) 107958.

[28] Z. Wang, K. Crammer, S. Vucetic, Breaking the curse of kernelization: budgeted stochastic gradient descent for large-scale SVM training, J. Mach. Learn. Res. 13 (2012) 3103–3131.

[29] J. Lu, S. C. H. Hoi, J. Wang, P. Zhao, Z. Liu, Large scale online kernel learning, J. Mach. Learn. Res. 17 (2016) 47:1–47:43.

[30] J. Pennington, F. X. Yu, S. Kumar, Spherical random features for polynomial kernels, in: Advances in Neural Information Processing Systems, 2015, pp. 1846– 1854.

[31] F. Liu, X. Huang, Y. Chen, J. A. K. Suykens, Random features for kernel approximation: A survey on algorithms, theory, and beyond, IEEE Trans. Pattern Anal. Mach. Intell. 44 (10) (2022) 7128–7148.

[32] Y. Cho, L. K. Saul, Kernel methods for deep learning, in: Advances in Neural Information Processing Systems, 2009, pp. 342–350.

[33] M. Meister, T. Sarlós, D. P. Woodruf, Tight dimensionality reduction for sketching low degree polynomial kernels, in: Advances in Neural Information Processing Systems, 2019, pp. 9470–9481.

[34] P. Jain, P. Kar, Non-convex optimization for machine learning, Found. Trends

Mach. Learn. 10 (3-4) (2017) 142–336.

[35] E. Hazan, Introduction to online convex optimization, second edition (2023). URL https://arxiv.org/abs/1909.05207

[36] S. Boyd, L. Vandenberghe, Convex Optimization, Cambridge University Press, 2004.

[37] R. Venkatesan, M. J. Er, M. Dave, M. Pratama, S. Wu, A novel online multi-label classifier for high-speed streaming data applications, Evol. Syst. 8 (4) (2017) 303–315.

[38] J. Du, C. Vong, Robust online multilabel learning under dynamic changes in data distribution with labels, IEEE Trans. Cybern. 50 (1) (2020) 374–385.

[39] Y. Zou, X. Hu, P. Li, Y. Wu, J. Hu, Online multi-label classification under noisy and changing label distribution, Pattern Recognit. 173 (2026) 112892.

[40] A. Blum, A. Kalai, J. Langford, Beating the hold-out: Bounds for k-fold and progressive cross-validation, in: Proceedings of the 12th Annual Conference on Computational Learning Theory, COLT, 1999, pp. 203–208.

[41] S. Shalev-Shwartz, Online learning and online convex optimization, Found. Trends Mach. Learn. 4 (2) (2012) 107–194.

[42] J. Demsar, Statistical comparisons of classifiers over multiple data sets, J. Mach. Learn. Res. 7 (2006) 1–30.

[43] C. Fahy, S. Yang, M. Gongora, Scarcity of labels in non-stationary data streams: A survey, ACM Comput. Surv. 55 (2) (2023) 40:1–40:39.