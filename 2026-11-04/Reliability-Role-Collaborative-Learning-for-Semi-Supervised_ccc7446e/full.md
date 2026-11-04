# Reliability-Role Collaborative Learning for Semi-Supervised Cardiac Image Segmentation

Zhongrui Xie<sup>1,∗</sup>, Wenfeng Zhang<sup>1,∗</sup>, Wei Hu<sup>1</sup>, Ruitao Xie<sup>2</sup>, Wenjie Wang<sup>3</sup>, Zhouyun Xie<sup>1</sup>, Yi Zhang<sup>1,†</sup>, Tao Jia<sup>1,†</sup> <sup>1</sup>College of Computer and Information Science,

Research Center of Biomimetic Intelligence and Computational Biomedicine,

Chongqing Normal University, Chongqing, China

<sup>2</sup>Shenzhen University of Advanced Technology, Shenzhen, China

<sup>3</sup>Wuwei Tumour Hospital of Gansu Province, Wuwei, China

Abstract—Semi-supervised cardiac image segmentation aims to exploit abundant unlabeled images together with limited annotations, but the reliability of pseudo-label supervision is highly uneven across spatial regions. In a peer dual-network collaborative setting, interior regions are usually more stable, whereas boundary regions, low-confidence areas, and confidencediscrepant regions between the two peer networks are more likely to introduce noisy supervision. However, despite this spatial reliability heterogeneity, existing methods typically impose uniform supervision on all unlabeled pixels, overlooking reliability differences across regions and consequently limiting discriminative learning in challenging areas. To address this issue, we propose R2CL (Reliability-Role Collaborative Learning), a reliability-role collaborative learning framework for semi-supervised cardiac image segmentation. R2CL first performs reliability-role modeling based on confidence relationships between two networks, assigning different training roles to unlabeled regions: mutually uncertain regions are used for entropy regularization, confidencediscrepant regions for directional knowledge transfer, and mutually confident regions as reliable supervision anchors. Based on this formulation, Reliability-Role Mixed Cross Learning (RR-MCL) operationalizes these learning roles through region-mixed training, directional cross-network supervision, and entropy regularization over mutually uncertain regions. Subsequently, Reliability-Transition Prototype Contrastive Learning (RT-PCL) constructs query features from foreground-adjacent reliabilitytransition regions and learns network-specific class prototypes from mutually confident regions, thereby enhancing feature discriminability around ambiguous boundaries and difficult regions. Meanwhile, High-confidence Context Refinement (HCR) further exploits reliable interior pixels and trusted dense blocks within mutually confident regions to complement stable contextual supervision. R2CL achieves stable improvements in region-overlap and boundary-related metrics across multiple datasets, with more evident advantages on the foreground-sparse MVAA-CT dataset.

Index Terms—cardiac segmentation, semi-supervised learning, contrastive learning, mitral valve segmentation

## I. INTRODUCTION

Medical image segmentation supports computer-aided diagnosis, quantitative image analysis, and clinical planning. CNNbased models such as U-Net [1] and nnU-Net [2], together with Transformer architectures including TransUNet [3] and Swin UNETR [4], have advanced anatomical segmentation. However, their performance relies heavily on expert pixelwise annotations, which are particularly costly for volumetric images. Semi-supervised learning alleviates this burden by exploiting abundant unlabeled data alongside limited annotations.

Existing semi-supervised learning approaches include pseudo-label learning [5], Mean Teacher [6], and cross pseudo supervision [7], often combined with consistency regularization [8] and data mixing [9]. In medical image segmentation, representative methods such as UAMT [11], MC-Net [10], BCP [12], and ALHVR [13] further incorporate uncertainty estimation, perturbation consistency, copy-paste mixing, and high-value region utilization to alleviate annotation scarcity. Nevertheless, these methods to some extent implicitly assume homogeneous supervision across unlabeled pixels, thereby overlooking differences in reliability and learning roles among spatial regions.

However, the reliability of unlabeled predictions is inherently heterogeneous across spatial regions. Interior regions of anatomical structures generally exhibit stable predictions, whereas boundary regions, low-confidence areas, and disagreement regions between dual networks are more susceptible to class ambiguity and noisy pseudo-label supervision. This issue is particularly pronounced in cardiac MRI and mitral valve CT segmentation, where local false positives, structural discontinuities, and boundary deviations may compromise downstream clinical analyses, including ventricular volume estimation [14], [15], biventricular quantification [16], valve geometry assessment, and shape continuity analysis. More importantly, existing semi-supervised segmentation methods often implicitly assume that all unlabeled pixels contribute equally to model optimization and therefore impose uniform consistency regularization or pseudo-label supervision across all regions. Such a homogeneous supervision assumption overlooks the heterogeneous reliability characteristics of different spatial regions. In contrast, different unlabeled regions naturally assume distinct learning roles: some are suitable for uncertainty suppression, some facilitate directional knowledge transfer, while others provide reliable supervisory signals. Therefore, unlabeled learning should not merely be formulated as a uniform consistency regularization problem, but rather as a reliability-role learning problem.

Peer-network confidence relationships provide a basis for identifying different learning roles. Inspired by complementary supervision in co-training [17], mutually uncertain regions can be used for entropy regularization, confidence-discrepant regions for directional knowledge transfer, and mutually confident regions as candidates for reliable supervision. This role-based formulation allows different unlabeled regions to provide distinct training signals rather than relying solely on a single uniform supervisory objective.

Motivated by the above observations, we propose R2CL (Reliability-Role Collaborative Learning), a reliability-role collaborative learning framework for semi-supervised cardiac image segmentation. R2CL first performs reliabilityrole modeling based on dual-network confidence relationships, assigning distinct learning roles to unlabeled regions. RR-MCL then operationalizes these roles through region-mixed training, directional cross-network supervision, and entropy regularization. Moreover, RT-PCL focuses on foregroundadjacent reliability-transition regions, where query features are contrasted against network-specific class prototypes to enhance feature discrimination around ambiguous structures. HCR further exploits reliable interior pixels and trusted dense blocks within mutually confident areas to provide stable contextual supervision. Together, these components integrate complementary supervisory signals from different reliability roles into a unified collaborative framework, enabling rolespecific exploitation of heterogeneous unlabeled supervision.

Our main contributions are as follows:

• We propose R2CL, a reliability-role collaborative learning framework for semi-supervised cardiac image segmentation. R2CL introduces a reliability-role modeling strategy that reformulates unlabeled learning as a reliability-role learning problem based on dual-network confidence relationships. By assigning distinct learning roles to unlabeled regions, it avoids uniform supervision across all unlabeled pixels.

• We develop a role-specific collaborative learning mechanism that exploits complementary supervisory signals from different reliability roles through uncertainty suppression, reliability-transition learning, and high-confidence context refinement. This mechanism is instantiated by three complementary components, namely RR-MCL, RT-PCL, and HCR, enabling role-specific exploitation of heterogeneous unlabeled supervision.

• Experiments on the ACDC and MVAA-CT datasets demonstrate improvements in region-overlap and boundary-related metrics, with particularly evident advantages in foregroundsparse mitral valve CT segmentation.

## II. METHOD

## A. Problem Setting

We consider semi-supervised segmentation with a small labeled set $\mathcal { D } _ { l } ~ = ~ \{ ( x _ { i } ^ { l } , y _ { i } ^ { l } ) \} _ { i = 1 } ^ { N _ { l } }$ and a larger unlabeled set $\mathcal { D } _ { u } ~ = ~ \{ x _ { j } ^ { u } \} _ { j = 1 } ^ { N _ { u } }$ , where $N _ { l } \ \ll \ N _ { u }$ . Here, x denotes an input image and y denotes the corresponding segmentation annotation.

## B. Overview

As shown in Fig. 1, R2CL consists of three coupled modules: RR-MCL, RT-PCL, and HCR. RR-MCL assigns reliability roles to unlabeled regions and converts them into regionmixed training and directional cross-network supervision. RT-PCL constrains foreground-adjacent reliability-transition regions through network-specific class-prototype contrast. HCR further exploits mutually confident regions through reliable interior pixels and trusted dense blocks. The two networks share the same encoder-decoder backbone but use different decoder upsampling operations, which encourages complementary local boundary recovery and confidence responses.

## C. Reliability-Role Mixed Cross Learning

RR-MCL partitions unlabeled pixels according to peernetwork confidence. For a pixel $v ,$ let $p _ { s } ^ { c } ( v )$ denote the class-c probability predicted by network $s ,$ with confidence $q _ { s } ( v ) = \operatorname* { m a x } _ { c } p _ { s } ^ { c } ( v )$ . To avoid a fixed threshold, we use a mini-batch adaptive threshold based on the average confidence $\bar { q } ( v )$ , where $ { \boldsymbol { S } } _ { \rho }$ contains the lowest $\rho$ proportion of $\bar { q } ( v )$ in the current mini-batch:

$$
\bar { q } ( v ) = \frac { q _ { 1 } ( v ) + q _ { 2 } ( v ) } { 2 } , \quad \tau = \frac { 1 } { | S _ { \rho } | } \sum _ { v \in S _ { \rho } } \bar { q } ( v ) .\tag{1}
$$

The threshold defines four reliability regions:

$$
\begin{array} { r l r } & { } & { \Omega ^ { U U } = \{ v \mid q _ { 1 } ( v ) < \tau , \ q _ { 2 } ( v ) < \tau \} , } \\ & { } & { \Omega ^ { 1  2 } = \{ v \mid q _ { 1 } ( v ) < \tau , \ q _ { 2 } ( v ) \geq \tau \} , } \\ & { } & { \Omega ^ { 2  1 } = \{ v \mid q _ { 1 } ( v ) \geq \tau , \ q _ { 2 } ( v ) < \tau \} , } \\ & { } & { \Omega ^ { C C } = \{ v \mid q _ { 1 } ( v ) \geq \tau , \ q _ { 2 } ( v ) \geq \tau \} . } \end{array}\tag{2}
$$

Here, $\Omega ^ { U U } , \Omega ^ { 1  2 } , \Omega ^ { 2  1 }$ , and $\Omega ^ { C C }$ denote mutually uncertain, network-2-guided, network-1-guided, and mutually confident regions, respectively. The first three support region mixing, directional supervision, and transition mining, whereas $\Omega ^ { \tilde { C \tilde { C } } }$ provides reliable candidates for RT-PCL and HCR. We construct the region-mixed input using the first three regions:

$$
\begin{array} { r l } & { \Omega ^ { M i x } = \Omega ^ { U U } \cup \Omega ^ { 1  2 } \cup \Omega ^ { 2  1 } , } \\ & { { \bf x } _ { \Omega ^ { M i x } } ^ { u } = \Omega ^ { M i x } \odot { \bf x } ^ { u } , } \\ & { { \bf x } ^ { m i x } = { \bf x } _ { \Omega ^ { M i x } } ^ { u } + ( 1 - \Omega ^ { M i x } ) \odot { \bf x } ^ { l } . } \end{array}\tag{3}
$$

This mask-based composition preserves deterministic supervision in retained labeled regions and exposes selected unlabeled

![](images/9aa8cb3f13ff7e706f0294aa5e588b637ddc22a4122a8c53214df5412f8aba7d.jpg)  
Fig. 1. Overall framework of R2CL. (a) RR-MCL constructs reliability-role mixed cross learning by converting unlabeled regions into region-mixed training, directional cross-network supervision, and entropy regularization signals. (b) RT-PCL performs foreground-adjacent reliability-transition query mining and network-specific class-prototype contrast. (c) HCR exploits reliable interior pixels and trusted dense blocks in mutually confident regions to enhance structural consistency.

regions to role-specific constraints. Labeled pixels outside $\Omega ^ { \overline { { M } } i x }$ retain their ground-truth supervision:

$$
\mathcal { L } _ { l a b } ^ { m i x } = \sum _ { s = 1 } ^ { 2 } \mathbb { E } _ { v \not \in \Omega ^ { M i x } } \left[ \ell _ { s e g } \big ( \tilde { \mathbf { p } } _ { s } ( v ) , \mathbf { y } ^ { l } ( v ) \big ) \right] ,\tag{4}
$$

Here, $\tilde { { \bf p } } _ { s }$ denotes the prediction of network s on $\mathbf { x } ^ { m i x }$ mixed unlabeled regions are constrained by the following rolespecific objectives.

For the mutually uncertain region, neither network provides a sufficiently reliable supervisory signal. We impose symmetric entropy regularization on this region from two views: the original unlabeled view and the region-mixed view. The entropy constraint is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { U U } ^ { \nu } = \mathcal { L } _ { U U , 1 } ^ { \nu } + \mathcal { L } _ { U U , 2 } ^ { \nu } , \quad \nu \in \{ \mathrm { r a w } , \mathrm { m i x } \} , } \\ & { \mathcal { L } _ { U U , s } ^ { \nu } = \mathbb { E } _ { v \in \Omega ^ { U U } } \left[ H ( \mathbf { p } _ { s } ^ { \nu } ( v ) ) \right] , \quad s \in \{ 1 , 2 \} , } \\ & { \mathbf { p } _ { s } ^ { \mathrm { r a w } } ( v ) = \mathbf { p } _ { s } ( v ) , \qquad \mathbf { p } _ { s } ^ { \mathrm { m i x } } ( v ) = \tilde { \mathbf { p } } _ { s } ( v ) . } \end{array}\tag{5}
$$

In confidence-discrepant regions, the higher-confidence network’s raw prediction serves as a detached soft target for the lower-confidence network’s mixed-view prediction:

$$
\begin{array} { r l } & { \mathcal { L } _ { d i r } ^ { 1  2 } = \mathbb { E } _ { v \in \Omega ^ { 1  2 } } D _ { K L } \big ( \mathrm { s g } \big [ { \bf p } _ { 2 } ( v ) \big ] \big \| \tilde { \bf p } _ { 1 } ( v ) \big ) , } \\ & { \mathcal { L } _ { d i r } ^ { 2  1 } = \mathbb { E } _ { v \in \Omega ^ { 2  1 } } D _ { K L } \big ( \mathrm { s g } \big [ { \bf p } _ { 1 } ( v ) \big ] \big \| \tilde { \bf p } _ { 2 } ( v ) \big ) , } \\ & { \quad \mathcal { L } _ { d i r } = \mathcal { L } _ { d i r } ^ { 1  2 } + \mathcal { L } _ { d i r } ^ { 2  1 } . } \end{array}\tag{6}
$$

Here, $D _ { K L }$ denotes KL divergence and $\mathrm { s g } [ \cdot ]$ stops gradients through the soft target. This one-way gradient flow uses the high-confidence prediction as a detached soft target without updating it through the KL term. The overall RR-MCL objective is:

$$
\begin{array} { r l } & { \mathcal { L } _ { R R } = \lambda _ { r a w } ( t ) \mathcal { L } _ { U U } ^ { r a w } } \\ & { \phantom { \mathcal { L } _ { R R } = } + \lambda _ { m i x } ( t ) \left( \mathcal { L } _ { l a b } ^ { m i x } + \mathcal { L } _ { d i r } + \lambda _ { l l } \mathcal { L } _ { U U } ^ { m i x } \right) . } \end{array}\tag{7}
$$

Here, $\mathcal { L } _ { l a b } ^ { m i x }$ supervises retained labeled regions, $\mathcal { L } _ { d i r }$ performs directional soft-target transfer, and $\mathcal { L } _ { U U } ^ { r a w }$ and $\mathcal { L } _ { U U } ^ { m i x }$ are rawview and mixed-view entropy terms.

## D. Reliability-Transition Prototype Contrastive Learning

RT-PCL constrains feature representations around lowconfidence and confidence-discrepant regions. It treats transition bands between reliability roles as query-mask candidates, rather than as ground-truth anatomical boundaries. The foreground gate suppresses obvious background responses, and network-specific class prototypes provide class anchors within each feature space. Let $\bar { \bf p } ( v ) = \frac 1 2 ( { \bf p } _ { 1 } ( v ) + { \bf p } _ { 2 } ( v ) )$ denote the averaged prediction of the two networks. We first extract reliability-transition bands from $\Omega ^ { U U } , \Omega ^ { 1  2 }$ , and $\Omega ^ { 2  1 }$ . The foreground gate then produces the foreground-gated mask $\Omega ^ { F G }$ , yielding foreground-adjacent query regions:

$$
\begin{array} { r l } & { \partial \Omega = \mathrm { D i l a t e } ( \Omega , \boldsymbol { r } ) \oplus \mathrm { E r o d e } ( \Omega , \boldsymbol { r } ) , } \\ & { \Omega ^ { R T } = \partial \Omega ^ { U U } \cup \partial \Omega ^ { 1  2 } \cup \partial \Omega ^ { 2  1 } , } \\ & { \Omega ^ { F G } = \mathrm { D i l a t e } \Big ( \{ \boldsymbol { v } \mid \arg \underset { c } { \operatorname* { m a x } } \bar { p } ^ { c } ( \boldsymbol { v } ) \neq 0 \} , r _ { f } \Big ) , } \\ & { \Omega ^ { Q } = \Omega ^ { R T } \cap \Omega ^ { F G } . } \end{array}\tag{8}
$$

Here, $\Omega ^ { R T } , \ \Omega ^ { F G }$ , and $\Omega ^ { Q }$ denote the reliability-transition band, foreground-gated mask, and final query mask, respectively. The query mask only represents spatial locations.

The query label is obtained from the detached ensemble prediction:

$$
\hat { y } ( v ) = \arg \operatorname* { m a x } _ { c } \bar { p } ^ { c } ( v ) , \quad \mathcal { Q } _ { s } = \{ ( \mathbf { z } _ { s } ^ { u } ( v ) , \hat { y } ( v ) ) \ | \ v \in \Omega ^ { Q } \} .\tag{9}
$$

This hard pseudo-label is only used as the class index in the contrastive term. The foreground gate is used to suppress obvious background responses, while prototype selection reduces unreliable positive anchors.

To avoid ambiguity between unaligned feature spaces, RT-PCL constructs network-specific class prototypes. For the s-th network and class $c ,$ the reliable prototype candidate set is

$$
\begin{array} { c } { \mathcal { P } _ { s , c } = \displaystyle \left\{ \mathbf { z } _ { s } ^ { u } ( v ) ~ | ~ v \in \Omega ^ { C C } , ~ \hat { y } _ { 1 } ( v ) = \hat { y } _ { 2 } ( v ) = c , \right. } \\ { \displaystyle \left. \operatorname* { m a x } _ { k } \bar { p } ^ { k } ( v ) \geq \tau _ { p } \right\} . } \end{array}\tag{10}
$$

Here, $\tau _ { p }$ controls the confidence of prototype sources. In training, candidate pixels of each class are sampled in a classbalanced manner; if a class has insufficient candidates, it is skipped from prototype contrast in the current batch. The class prototype is defined as

$$
\begin{array} { l } { \displaystyle \mu _ { s , c } = \frac { 1 } { | \mathcal { P } _ { s , c } | } \sum _ { \mathbf { z } \in \mathcal { P } _ { s , c } } \mathbf { z } , } \\ { \displaystyle \mu _ { s , c }  \frac { \mu _ { s , c } } { \| \mu _ { s , c } \| _ { 2 } } . } \end{array}\tag{11}
$$

For query features from the s-th network, RT-PCL pulls each query toward the corresponding prototype in the same feature space and separates it from other class prototypes:

$$
\begin{array} { r l } & { \mathcal { L } _ { R T } ^ { ( s ) } = - \mathbb { E } _ { v \in \Omega ^ { Q } } \log \frac { \exp \left( \sin ( \mathbf { z } _ { s } ^ { u } ( v ) , \boldsymbol { \mu _ { s , \hat { y } ( v ) } } ) / T \right) } { \sum _ { c \in \mathcal { C } _ { s } } \exp ( \sin ( \mathbf { z } _ { s } ^ { u } ( v ) , \boldsymbol { \mu _ { s , c } } ) / T ) } , } \\ & { \mathcal { L } _ { R T } = \frac { 1 } { 2 } \left( \mathcal { L } _ { R T } ^ { ( 1 ) } + \mathcal { L } _ { R T } ^ { ( 2 ) } \right) . } \end{array}\tag{12}
$$

Here, T denotes the temperature parameter, and $\mathcal { C } _ { s }$ denotes the set of prototype classes available for the s-th network in the current batch. By constructing prototypes within each network, RT-PCL avoids directly sharing class prototypes across unaligned feature spaces and converts complementary dual-network responses into discriminative feature constraints.

## E. High-confidence Context Refinement

HCR exploits stable supervisory information in mutually confident regions to complement RT-PCL, which mainly focuses on reliability-transition regions. Let $\hat { y } _ { s } ( v ) =$ arg max<sub>c</sub> $p _ { s } ^ { c } ( v )$ denote the predicted class of the s-th network, and let yˆ(v) = arg max<sub>c</sub> $: \bar { p } ^ { c } ( v )$ denote the ensemble pseudolabel. The pseudo-foreground set and its local boundary band are

$$
\begin{array} { r l } & { \Omega ^ { P F } = \{ v \mid \hat { y } ( v ) \neq 0 \} , } \\ & { \quad \partial \hat { Y } = \mathrm { D i l a t e } ( \Omega ^ { P F } , r _ { i } ) \oplus \mathrm { E r o d e } ( \Omega ^ { P F } , r _ { i } ) . } \end{array}\tag{13}
$$

We then select reliable interior pixels that are predictionconsistent, high-confidence, foreground, and away from this boundary band:

$$
\begin{array} { c } { \Omega ^ { R I } = \{ v \in \Omega ^ { C C } \mid \hat { y } _ { 1 } ( v ) = \hat { y } _ { 2 } ( v ) = \hat { y } ( v ) , } \\ { \hat { y } ( v ) \neq 0 , \ : \operatorname* { m a x } _ { c } \bar { p } ^ { c } ( v ) \geq \tau _ { i } , \ : \ : v \notin \partial \hat { Y } \} . } \end{array}\tag{14}
$$

For this region, we impose classification and overlap con straints using the ensemble pseudo-label:

$$
\mathcal { L } _ { r i } = \sum _ { s = 1 } ^ { 2 } \left[ \mathcal { L } _ { c e } \left( f _ { s } ( x ^ { u } ) , \hat { y } ; \Omega ^ { R I } \right) + \mathcal { L } _ { d i c e } \left( \mathbf { p } _ { s } , \hat { y } ; \Omega ^ { R I } \right) \right] .\tag{15}
$$

HCR further selects trusted dense blocks according to a local reliability score:

$$
B ^ { T } = \{ B _ { k } \mid R ( B _ { k } ) \geq \tau _ { b } \} .\tag{16}
$$

Here, $R ( B _ { k } )$ is determined by the mutually confident ratio, average confidence, and foreground occupancy within the local block. For selected trusted blocks, we impose local dense classification using the ensemble pseudo-label:

$$
\mathcal { L } _ { t d b } = \sum _ { s = 1 } ^ { 2 } \mathcal { L } _ { c e } \big ( f _ { s } ( x ^ { u } ) , \hat { y } ; B ^ { T } \big ) .\tag{17}
$$

The final training objective is:

$$
\mathcal { L } = \sum _ { s = 1 } ^ { 2 } \mathcal { L } _ { s u p } ^ { ( s ) } + \mathcal { L } _ { R R } + \lambda _ { P } ( t ) \mathcal { L } _ { R T } + \lambda _ { H } \mathcal { L } _ { H C R } .\tag{18}
$$

Here, $\mathcal { L } _ { s u p } ^ { ( s ) } , \mathcal { L } _ { R T }$ , and $\mathcal { L } _ { H C R }$ denote the supervised segmentation loss, RT-PCL prototype contrastive loss, and HCR constraints, respectively.

## III. EXPERIMENTS AND RESULTS

## A. Datasets

The ACDC dataset contains short-axis cardiac cine-MRI scans from 100 patients [18]. Following the common semisupervised setting, 70 cases are used as the training pool, 10 cases are used for validation, and 20 cases are used for testing. The segmentation targets include the right ventricle, myocardium, and left ventricle.

The MVAA-CT dataset is derived from Task 1 of the MVAA 2026 challenge [19], containing 27 labeled mitral valve CT volumes and 1040 official unlabeled CT volumes. We split the 27 labeled cases into 20 training-pool cases and 7 validation cases. For low-label experiments, we fix the 7 validation cases and construct the 2-labeled and 4-labeled settings within the 20 training-pool cases. In each setting, labeled cases are randomly sampled five times, while the remaining training-pool cases are used as unlabeled data by hiding their annotations, together with the 1040 official unlabeled CT volumes. All main comparison results on MVAA-CT are reported as the mean and standard deviation over the five labeled-case samplings. Compared with relatively compact chamber structures, mitral valve targets are more foreground-sparse and thinner, making local continuity and surface boundaries more sensitive for subsequent structure quantification.

TABLE I  
COMPARISON ON THE ACDC TEST SET. BEST AND SECOND-BEST RESULTS ARE BOLD AND UNDERLINED; † DENOTES REPRODUCED RESULTS.
<table><tr><td>Method</td><td>Labeled</td><td>Unlabeled |</td><td>DSC↑</td><td>Jaccard↑</td><td>95HD↓</td><td>ASD↓</td></tr><tr><td rowspan="3">U-Net† [1]</td><td rowspan="3">7(10%) 14(20%)</td><td>0</td><td>77.07</td><td>65.14</td><td>8.47</td><td>2.65</td></tr><tr><td>0</td><td>84.12</td><td>74.21</td><td>6.46</td><td>2.03</td></tr><tr><td>0</td><td>91.47</td><td>84.68</td><td>1.40</td><td>0.34</td></tr><tr><td rowspan="9">CPS† [7] MC-Net+† [10] BCP{ [12] DiffRect† [21]</td><td rowspan="9">7(10%) 63(90%)</td><td rowspan="9"></td><td>86.74</td><td>77.43</td><td>6.64</td><td>1.80</td></tr><tr><td>86.31</td><td>76.71</td><td>8.23</td><td>2.20</td></tr><tr><td>88.23</td><td>79.71</td><td>4.54</td><td>1.41</td></tr><tr><td>89.53</td><td>81.57</td><td>4.33</td><td>1.21</td></tr><tr><td>87.99</td><td>79.28</td><td>3.28</td><td>0.90</td></tr><tr><td>89.08</td><td>80.90</td><td>1.82</td><td>0.71</td></tr><tr><td>89.35</td><td>81.35</td><td>1.54</td><td>0.61</td></tr><tr><td>88.92</td><td>79.65</td><td>4.32</td><td>1.19</td></tr><tr><td>89.93</td><td>82.16</td><td>1.43</td><td>0.47</td></tr><tr><td rowspan="9">CPS† [7] MC-Net+† [10] BCP† [12]</td><td rowspan="9"></td><td rowspan="9"></td><td></td><td></td><td>5.47</td><td>1.59</td></tr><tr><td>88.48 87.94</td><td>80.28 79.50</td><td>5.77</td><td>1.66</td></tr><tr><td>89.20</td><td>81.13</td><td>4.80</td><td>1.56</td></tr><tr><td>89.92</td><td>82.27</td><td>3.44</td><td>0.93</td></tr><tr><td>88.76</td><td>80.64</td><td>4.49</td><td>1.16</td></tr><tr><td>89.76</td><td>81.27</td><td>1.71</td><td>0.63</td></tr><tr><td>89.99</td><td>82.38</td><td>2.04</td><td>0.55</td></tr><tr><td>90.23</td><td>81.24</td><td>4.01</td><td>1.06</td></tr><tr><td>90.75</td><td>83.53</td><td>1.37</td><td>0.42</td></tr></table>

## B. Implementation Details

U-Net [1] and 3D U-Net [20] serve as the backbones for ACDC and MVAA-CT, respectively. ACDC uses 256 × 256 inputs with a batch size of 16 (8 labeled and 8 unlabeled images), whereas MVAA-CT uses $4 8 \times 1 2 8 \times 1 2 8$ patches with a batch size of 4 (2 labeled and 2 unlabeled patches). Both peer networks are optimized with AdamW at an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , for 30K iterations on ACDC and 15K on MVAA-CT. RT-PCL is activated after 3K iterations to avoid unstable early prototypes. We evaluate DSC (Dice similarity coefficient), Jaccard, 95HD (95% Hausdorff distance), and ASD (average surface distance). Reproduced methods follow their public implementations or original descriptions with necessary dataset-specific adaptations, using identical data splits, preprocessing, and evaluation metrics. All semi-supervised methods on MVAA-CT use the same unlabeled pool, including hidden-label training-pool volumes and 1040 official unlabeled CT volumes.

## C. Comparison with Existing Methods

1) Results on ACDC: Table I reports the comparison results on ACDC. Under both 10% and 20% labeled settings, R2CL achieves the best results across all four metrics among the compared semi-supervised methods. With 10% labels, R2CL achieves 89.93 DSC, outperforming DiffRect [21] by 0.40, and reduces 95HD to 1.43, compared with 1.54 for M3HL [24]. With 20% labels, R2CL achieves 90.75 DSC and reduces

95HD and ASD to 1.37 and 0.42, respectively, indicating that modeling difficult boundary-adjacent regions helps reduce boundary errors.

2) Results on MVAA-CT: Table II reports the average comparison results over five labeled-case samplings on MVAA-CT. Under both 2- and 4-labeled settings, R2CL achieves the best average results across all four metrics. Under the 2-labeled setting, R2CL achieves $7 8 . 4 8 \pm 1 . 6 9$ DSC, outperforming BCP [12] by 2.56 percentage points, while reducing 95HD and ASD to $2 . 0 8 \pm 0 . 4 5$ and $0 . 6 2 \pm 0 . 0 8 .$ , respectively. Under the 4-labeled setting, R2CL achieves $8 2 . 6 6 \pm 0 . 8 0$ DSC and 0.41 ± 0.15 ASD, indicating improved segmentation performance with limited annotations.

MVAA-CT presents a more foreground-sparse segmentation problem than ACDC, with foreground voxels occupying approximately 1.92% of labeled MVAA-CT volumes compared with 3.79% for ACDC. The thin and variable mitral valve structures are susceptible to discontinuities and false-positive predictions, particularly under the 2-labeled setting. By combining role-specific supervision with foregroundadjacent reliability-transition learning and high-confidence context refinement, R2CL may better address these challenges, as reflected in its improvements on MVAA-CT.

3) Visual comparison: Fig. 2 presents segmentation visualizations on ACDC and MVAA-CT under the corresponding low-label settings. The figure uses 7 labeled cases (10%) for ACDC and the 4-labeled setting for MVAA-CT. Visual comparisons are made only within each dataset. In the ACDC examples, several compared methods produce small falsepositive regions in the background adjacent to the true foreground, whereas R2CL better suppresses such responses. On MVAA-CT, R2CL shows more stable performance in falsepositive suppression and elongated-structure continuity.

TABLE II  
FIVE REPEATED LABELED-CASE SAMPLING RESULTS ON THE MVAA-CT VALIDATION SET. RESULTS ARE REPORTED AS MEAN ± STANDARD DEVIATION. BEST AND SECOND-BEST MEAN VALUES ARE BOLD AND UNDERLINED; † DENOTES REPRODUCED RESULTS.
<table><tr><td rowspan=1 colspan=7>Method      Labeled Unlabeled|  DSC↑   Jaccard↑  95HD↓   ASD↓</td></tr><tr><td rowspan=1 colspan=1>3D U-Net†[20]</td><td rowspan=1 colspan=1>2      04      0</td><td rowspan=1 colspan=5>74.55±3.8360.13±4.904.56±2.101.18±0.4481.06±0.9368.34±1.271.83±0.430.59±0.14</td></tr><tr><td rowspan=7 colspan=1>MC-Net+† [10]CAML† [26]BCP† [12]MLRP† [22]M3HL† [24]PICK†[27]LeFeD† [28]AdaMix† [23]R2CL(Ours)</td><td rowspan=5 colspan=1>2     1058</td><td rowspan=1 colspan=5>73.28±2.2158.92±3.363.23±1.020.74±0.2572.33±3.1757.25±4.023.79±1.47 0.80±0.4275.92±1.9861.22±2.562.58±0.670.73±0.16</td></tr><tr><td rowspan=4 colspan=5>71.46±3.5256.94±4.134.26±1.671.15±0.9471.70±3.55 56.35±4.394.22±1.35 1.19±0.2773.55±5.2159.16±5.53 3.42±1.24 0.77±0.1875.48±4.33 61.14±5.35 3.23±1.31 0.94±0.33</td></tr><tr><td rowspan=1 colspan=2>35 1.19±0.27</td></tr><tr><td rowspan=2 colspan=2>3.23±1.21</td><td rowspan=1 colspan=1>0.77±0.18</td></tr><tr><td rowspan=1 colspan=1>75.48±4.33</td><td rowspan=1 colspan=1>0.94±0.33</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2>72.59±3.02 57.26±3.67 4.11±2.48 1.52±0.4978.48±1.6964.89±2.23 2</td><td rowspan=1 colspan=1>57.26±3.67</td><td rowspan=1 colspan=2>4.11±2.48</td><td rowspan=1 colspan=1>1.52±0.49</td></tr><tr><td rowspan=1 colspan=2>.08±0.45 0</td><td rowspan=1 colspan=1>.62±0.08</td></tr><tr><td rowspan=6 colspan=1>MC-Net+† [10]]CAML† [26]BCP{†[12]MLRP†[22]M3HL† [24]PICK† [27]</td><td rowspan=3 colspan=1></td><td rowspan=1 colspan=1>80.32±1.44</td><td rowspan=1 colspan=1>66.75±1.561</td><td rowspan=1 colspan=2>.98±0.52 0</td><td rowspan=1 colspan=1>.65±0.28</td></tr><tr><td rowspan=1 colspan=1>79.87±1.22</td><td rowspan=1 colspan=1>66.43±1.37</td><td rowspan=1 colspan=2>2.02±0.58</td><td rowspan=1 colspan=1>0.72±0.26</td></tr><tr><td rowspan=1 colspan=1>79.43±1.07</td><td rowspan=1 colspan=1>66.16±1.39</td><td rowspan=1 colspan=2>2.03±0.41</td><td rowspan=1 colspan=1>0.64±0.17</td></tr><tr><td rowspan=2 colspan=1>4     1056</td><td rowspan=1 colspan=1>80.99±1.78</td><td rowspan=1 colspan=1>68.28±2.44</td><td rowspan=1 colspan=2>1.66±0.56</td><td rowspan=1 colspan=1>0.53±0.22</td></tr><tr><td rowspan=1 colspan=1>78.02±0.51</td><td rowspan=1 colspan=1>65.56±1.61</td><td rowspan=1 colspan=2>3.35±0.69</td><td rowspan=1 colspan=1>0.96±0.26</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>80.08±1.13 6</td><td rowspan=1 colspan=1>7.10±1.36</td><td rowspan=1 colspan=2>2.04±0.62 0</td><td rowspan=1 colspan=1>.65±0.20</td></tr><tr><td rowspan=1 colspan=1>LeFeD†[28]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>81.27±1.19</td><td rowspan=1 colspan=1>68.73±1.59</td><td rowspan=1 colspan=2>1.62±0.47 0</td><td rowspan=1 colspan=1>.51±0.19</td></tr><tr><td rowspan=2 colspan=1>AdaMix† [23]R2CL(Ours)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>81.04±1.60</td><td rowspan=1 colspan=1>68.71±2.261</td><td rowspan=1 colspan=2>.44±0.430</td><td rowspan=1 colspan=1>.55±0.23</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>82.66±0.80 7</td><td rowspan=1 colspan=1>0.63±1.12</td><td rowspan=1 colspan=2>1.29±0.37 0.41±0.15</td><td></td></tr></table>

Image  
GT  
MC-Net+  
BCP  
DiffRect  
MLRP  
M3HL  
AdaMix  
R2CL(Ours)  
![](images/66a9478c349e4964f0cef2f1440d14d1b7a1f7fd7cb69328d3620759302f0d06.jpg)  
(B) MVAA-CT 4 Labeled  
Fig. 2. Segmentation visualizations on ACDC test cases and MVAA-CT validation cases. Panel (A) shows ACDC with 7 labeled cases (10%), and panel (B) shows MVAA-CT under the 4-labeled setting. Comparisons are intended within each dataset.

TABLE III  
STEPWISE COMPONENT ABLATION ON ACDC AND MVAA-CT. ACDC USES THE 7-LABELED SETTING, AND MVAA-CT USES THE 2-LABELED SETTING. COMPONENT ABLATION IS CONDUCTED ON THE SAME DEFAULT SPLIT.
<table><tr><td>Dataset</td><td>Variant</td><td>DSC↑ Jaccard↑ 95HD↓ ASD↓</td><td></td><td></td><td></td></tr><tr><td rowspan="5">ACDC</td><td>Baseline</td><td>86.54</td><td>77.32</td><td>5.96</td><td>1.79</td></tr><tr><td>+RR-MCL</td><td>88.39</td><td>79.78</td><td>3.23</td><td>0.68</td></tr><tr><td>+RR-MCL+RT-PCL 89.22</td><td></td><td>81.32</td><td>1.85</td><td>0.64</td></tr><tr><td>+RR-MCL+HCR</td><td>89.56</td><td>81.61</td><td>2.41</td><td>0.66</td></tr><tr><td>R2CL(Ours)</td><td>89.93</td><td>82.16</td><td>1.43</td><td>0.47</td></tr><tr><td rowspan="5"></td><td>Baseline</td><td>71.93</td><td>56.45</td><td>4.98</td><td>1.28</td></tr><tr><td>+RR-MCL</td><td>74.84</td><td>60.25</td><td>2.59</td><td>0.82</td></tr><tr><td>MVAA-CT +RR-MCL+RT-PCL 76.06</td><td></td><td>61.71</td><td>1.93</td><td>0.62</td></tr><tr><td>+RR-MCL+HCR</td><td>77.23</td><td>63.26</td><td>2.28</td><td>0.67</td></tr><tr><td>R2CL(Ours)</td><td>78.10</td><td>64.35</td><td>1.71</td><td>0.55</td></tr></table>

## D. Ablation Analysis

## 1) Component ablation:

Table III evaluates RR-MCL, RT-PCL, and HCR against a dual-network cross-pseudo-supervision baseline on the fixed ACDC 7-labeled and MVAA-CT 2-labeled splits. Since RT-PCL and HCR rely on the reliability regions generated by RR-MCL, they are evaluated after introducing RR-MCL. On ACDC, RR-MCL improves DSC from 86.54 to 88.39 and reduces 95HD from 5.96 to 3.23. Adding RT-PCL further reduces 95HD to 1.85, whereas adding HCR to RR-MCL yields larger improvements in region-overlap metrics. The full R2CL achieves 89.93 DSC and 1.43 95HD.

On MVAA-CT, the components also provide improvements with different emphases. RR-MCL increases DSC from 71.93 to 74.84; adding RT-PCL reduces 95HD from 2.59 to 1.93, while adding HCR increases DSC to 77.23. These results indicate that low-confidence regions, reliability-transition regions, and mutually confident regions provide complementary learning signals.

![](images/0fcd700e3238421d0ffa92d1dcc1c4d7f6ff0f7d4f5f0d9737d41ee59eb60c76.jpg)

![](images/b52b28899bce630820798de2f8f8965bfeb6668b48fb7c0efabba2d1e6776538.jpg)

![](images/cdce6f84a3280f346d986ba00ebc7ae21255dcb1a1609ee81a969c77a0ef5d95.jpg)  
Fig. 3. RR-MCL training dynamics on MVAA-CT: region proportions, directional KL reduction, and raw/mixed-view uncertainty. Here, $\Omega ^ { C \bar { C } } , \Omega ^ { U U }$ $\Omega ^ { 1  2 }$ , and $\Omega ^ { 2  1 }$ denote mutually confident, mutually uncertain, network-2-guided, and network-1-guided regions, respectively.

2) RR-MCL training dynamics analysis: Fig. 3 analyzes how RR-MCL converts reliability regions into training roles on MVAA-CT. Fig. 3(a) shows that $\Omega ^ { C C }$ dominates after stabilization, while sparse but persistent difficult regions remain. Fig. 3(b) shows decreasing directional KL, and Fig. 3(c) shows that the mixed view maintains higher uncertainty after stabilization.

![](images/8ed478c7ed365e87b088e0d314e6d584c69436c0dca49d0f001aea2eb0a729ae.jpg)

![](images/3ca91dd540f43f4a2b6fa52d53834dfa4c4b7ae3f1fd097247e3464a027cea30.jpg)  
Fig. 4. Per-case DSC and ASD distributions of the full model and RT-PCL ablation variants on the ACDC test set.

3) Case-level stability: Fig. 4 compares per-sample DSC and ASD distributions across six variants on 40 test volumes/frames from 20 ACDC patients. The variants include R2CL(Ours), w/o RT-PCL, w/o Asym. Dec., w/o RT Query, w/o FG Gate, and w/o Rel. Proto. R2CL maintains a high median DSC and a lower, more compact ASD distribution, whereas removing RT-PCL or its key designs increases boundary errors in some cases.

## IV. CONCLUSION

We presented R2CL, a reliability-role collaborative learning framework for semi-supervised cardiac image segmentation. Rather than applying uniform supervision to all unlabeled pixels, R2CL models unlabeled regions according to their reliability roles and exploits them through RR-MCL, RT-PCL, and HCR. Experiments on ACDC and MVAA-CT showed consistent improvements in overlap and boundaryrelated metrics, with larger gains on the foreground-sparse MVAA-CT dataset. These results indicate that reliability-role modeling can improve segmentation robustness under limited annotations and may benefit downstream clinical applications such as anatomical quantification and morphology assessment. Future work will explore adaptive reliability estimation and broader medical segmentation scenarios.

## REFERENCES

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. MICCAI, 2015, pp. 234–241.

[2] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, pp. 203–211, 2021.

[3] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, “TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers,” Medical Image Analysis, vol. 97, p. 103280, 2024.

[4] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, “Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images,” in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries, ser. Lecture Notes in Computer Science, vol. 12962, 2022, pp. 272–284.

[5] D.-H. Lee, “Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks,” in ICML Workshop on Challenges in Representation Learning, 2013.

[6] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Proc. NeurIPS, 2017, pp. 1195–1204.

[7] X. Chen, Y. Yuan, G. Zeng, and J. Wang, “Semi-supervised semantic segmentation with cross pseudo supervision,” in Proc. CVPR, 2021, pp. 2613–2622.

[8] T. Miyato, S. ichi Maeda, M. Koyama, and S. Ishii, “Virtual adversarial training: A regularization method for supervised and semi-supervised learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 8, pp. 1979–1993, 2019.

[9] D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and C. Raffel, “MixMatch: A holistic approach to semi-supervised learning,” in Proc. NeurIPS, 2019, pp. 5049–5059.

[10] Y. Wu, Z. Ge, D. Zhang, M. Xu, L. Zhang, Y. Xia, and J. Cai, “Mutual consistency learning for semi-supervised medical image segmentation,” Med. Image Anal., vol. 81, p. 102530, 2022.

[11] L. Yu, S. Wang, X. Li, C.-W. Fu, and P.-A. Heng, “Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation,” in Proc. MICCAI, 2019, pp. 605–613.

[12] Y. Bai, D. Chen, Q. Li, W. Shen, and Y. Wang, “Bidirectional copy-paste for semi-supervised medical image segmentation,” in Proc. CVPR, 2023, pp. 11 514–11 524.

[13] T. Lei, Z. Yang, X. Wang, Y. Wang, X. Wang, F. Sun, and A. K. Nandi, “Adaptive learning of high-value regions for semi-supervised medical image segmentation,” in Proc. ICCV, 2025, pp. 21 450–21 459.

[14] G. Luo, S. Dong, K. Wang, W. Zuo, S. Cao, and H. Zhang, “Multiviews fusion CNN for left ventricular volumes estimation on cardiac MR images,” IEEE Transactions on Biomedical Engineering, vol. 65, no. 9, pp. 1924–1934, 2018.

[15] G. Luo, W. Wang, C. Tam, K. Wang, S. Cao, H. Zhang, B. Chen, and S. Li, “Dynamically constructed network with error correction for accurate ventricle volume estimation,” Medical Image Analysis, vol. 64, p. 101723, 2020.

[16] G. Luo, S. Dong, W. Wang, K. Wang, S. Cao, C. Tam, H. Zhang, J. Howey, P. Ohorodnyk, and S. Li, “Commensal correlation network between segmentation and direct area estimation for bi-ventricle quantification,” Medical Image Analysis, vol. 59, p. 101591, 2020.

[17] A. Blum and T. Mitchell, “Combining labeled and unlabeled data with co-training,” in Proc. COLT, 1998, pp. 92–100.

[18] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang et al., “Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved?” IEEE Trans. Med. Imaging, vol. 37, no. 11, pp. 2514–2525, 2018.

[19] MVAA 2026 Challenge Organizers, “MVAA 2026: Mitral valve anatomy analysis using multimodal imaging data,” Codabench online challenge benchmark, 2026, accessed: Jun. 28, 2026. [Online]. Available: https://www.codabench.org/competitions/15662/

[20] O. C¸ ic¸ek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,<sup>¨</sup> “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in Proc. MICCAI, 2016, pp. 424–432.

[21] X. Liu, W. Li, and Y. Yuan, “DiffRect: Latent diffusion label rectification for semi-supervised medical image segmentation,” in Proc. MICCAI, 2024, pp. 56–66.

[22] J. Su, Z. Luo, S. Lian, D. Lin, and S. Li, “Mutual learning with reliable pseudo label for semi-supervised medical image segmentation,” Med. Image Anal., vol. 94, p. 103111, 2024.

[23] Z. Shen, P. Cao, J. Su, J. Yang, and O. R. Zaiane, “Adaptive mix for semi-supervised medical image segmentation,” Medical Image Analysis, vol. 108, p. 103857, 2026.

[24] Y. Liu, Z. Zhang, J. Yue, W. Guo, and D. Li, “M3HL: Mutual mask mix with high-low level feature consistency for semi-supervised medical image segmentation,” in Proc. MICCAI, 2025, pp. 314–323.

[25] Y. Yang, G. Sun, T. Zhang, R. Wang, and J. Su, “Semi-supervised medical image segmentation via weak-to-strong perturbation consistency and edge-aware contrastive representation,” Med. Image Anal., vol. 101, p. 103450, 2025.

[26] S. Gao, Z. Zhang, J. Ma, Z. Li, and S. Zhang, “Correlation-aware mutual learning for semi-supervised medical image segmentation,” in Proc. MICCAI, 2023, pp. 98–108.

[27] Q. Zeng, Z. Lu, Y. Xie, and Y. Xia, “PICK: Predict and mask for semisupervised medical image segmentation,” Int. J. Comput. Vis., vol. 133, pp. 3296–3311, 2025.

[28] Q. Zeng, Y. Xie, Z. Lu, M. Lu, J. Zhang, and Y. Xia, “Consistencyguided differential decoding for enhancing semi-supervised medical image segmentation,” IEEE Trans. Med. Imaging, vol. 44, no. 1, pp. 44–56, 2025.