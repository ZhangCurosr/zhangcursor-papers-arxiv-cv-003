# UNSUPERVISED LONG-TAILED ADAPTATION OFVISION-LANGUAGE MODELS

Keliang Chen<sup>1∗</sup> Yaxin Hou<sup>1∗</sup> Hui Liu<sup>3</sup> Yuheng Jia<sup>1,2,3†</sup>

<sup>1</sup>School of Computer Science and Engineering, Southeast University, Nanjing 210096, China <sup>2</sup>Key Laboratory of New Generation Artificial Intelligence Technology and Its

Interdisciplinary Applications (Southeast University), Ministry of Education, China

<sup>3</sup>School of Computing and Information Sciences, Saint Francis University, Hong Kong, China {220255105,yaxin,yhjia}@seu.edu.cn, h2liu@sfu.edu.hk

## ABSTRACT

Adapting vision-language models to downstream tasks has achieved remarkable success by leveraging pseudo-labels generated from unlabeled data. Existing methods typically assume a uniform unlabeled data distribution, and thus the resulting pseudo-label distribution is likewise uniform. However, real-world data distributions are often long-tailed. To tackle this, we formalize a new scenario termed Unsupervised Long-Tailed Adaptation (ULTA). Under this scenario, existing methods exhibit a contrasting phenomenon: head-class performance drops sharply, which is distinct from supervised long-tailed learning where tail classes suffer the most. In particular, we uncover that the distributional mismatch not only erodes head-class boundaries, but also pushes head samples into confusable classes, reinforcing the model’s inherent bias. To address these issues, we propose a novel model called Margin-Aware Refinement with Structural alignment (MARS). Specifically, we mitigate head-class boundary erosion via Boundary-Preserving Alignment, which takes the zero-shot VLM as a fixed visual reference to suppress probability increases that lack visual support in the training targets. Building upon this, we introduce Margin-aware Self-Refinement, which employs a dynamic adjustment strategy to refine tail and confusable classes while preventing prediction bias. Extensive experiments on nine benchmark datasets demonstrate that MARS outperforms state-of-the-art methods, achieving an average accuracy improvement of 4.71 percentage points.

## 1 INTRODUCTION

Vision-language models (VLMs) (Jia et al., 2021; Li et al., 2022; Xiao et al., 2024), such as CLIP (Rad ford et al., 2021), have demonstrated remarkable zero-shot capabilities by establishing a shared feature space across visual and textual modalities via web-scale data. Despite the impressive zero-shot capa bilities, their performance is limited when transferred to specific downstream tasks due to the domain gap. While fine-tuning on labeled data can bridge this gap (Zhou et al., 2022b; Jia et al., 2022; Ouali et al., 2023; Shi et al., 2024), acquiring sufficient annotations in specialized scenarios (e.g., medical imaging Huang et al., 2021; Wang et al., 2022) is impractical. Consequently, recent studies (Huang et al., 2022; Mirza et al., 2023; Menghini et al., 2023) have shifted their focus toward enhancing performance by exploiting unlabeled data.

These studies (Zhang et al., 2024; Li et al., 2025; Wang et al., 2025) primarily derive supervisory signals from unlabeled data via pseudo-labeling. Those methods usually leverage a uniform prior explicitly or implicitly to yield class-balanced pseudo-labels. However, real-world data exhibits a long-tailed distribution, creating a mismatch with this uniform prior. As evidenced in Fig. 1(a), these methods (e.g., CPL Zhang et al., 2024) suffer from substantial performance degradation on long-tailed data. Notably, we observe an inverted degradation pattern that diverges from supervised long-tailed learning. In supervised long-tailed learning, the tail classes typically suffer the most due to the scarcity of their training samples. While in unsupervised long-tailed scenarios, existing methods lead to a severe performance drop for head classes (see Fig. 1(b)).

![](images/57d2abceb71f9810c47779a160c1c963b8c803e3f8f7496a25c14e41c339f833.jpg)  
(a)

![](images/4d34484433be6847816aef8f4a9c865c0dbb6296dce61ccd78bbe8e9371aae1f.jpg)  
(b)

![](images/e22102c1fc4bfff7bc975fc77eda0f3021c117817623ec82581861b3f099985c.jpg)  
(c)

![](images/f7b09dbc48f8f1441802c0314c40d1c8f90cf00873b264e413797ff56543954f.jpg)  
(d)  
Figure 1: (a) The overall test accuracy of the existing unsupervised method CPL (Zhang et al., 2024) under different imbalance ratios. (b) Comparison of head- and tail-class accuracies across three settings on the RESISC45 dataset: unsupervised learning with balanced data (Unsup. Bal.), supervised learning with imbalanced data (Sup. Imb.), and unsupervised learning with imbalanced data (Unsup. Imb.). (c) Comparison of average logit margin changes $( \Delta m = \mathrm { { m a r g i n } _ { F i n e - t u n e d } - \mathrm { { m a r g i n } _ { Z e r o - s h o t C L I P } ) } }$ Under the long-tailed setting, the head-class margin significantly decreases compared to the uniform setting, indicating severe boundary erosion. (d) Label counts across three cases: ground-truth (GT), zero-shot predictions (ZS Pred.), and fine-tuned predictions (FT Pred.). We group confusable pairs by color (e.g., blue: “beach” vs. “sea ice”) and observe that zero-shot CLIP inherently exhibits bia toward confusable classes, which is further exacerbated by fine-tuning.

Empirically, we find that in long-tailed scenarios, the uniform assumption in existing methods forces the model to misclassify many head samples as tail classes. Fitting these “head-as-tail” false samples leads to head-class decision boundary erosion, which causes a significant decrease in the logit margin (i.e., the gap between the ground-truth logit and the highest non-target logit) of head classes (see Fig. 1(c)). Furthermore, zero-shot CLIP’s inherent bias toward confusable classes (e.g., semantically similar pairs like “beach” vs. “sea ice”) is exacerbated during fine-tuning (see Fig. 1(d)).

To address these challenges, we propose the Margin-Aware Refinement with Structural alignment (MARS) to tackle Unsupervised Long-Tailed Adaptation (ULTA), a realistic scenario where VLMs are adapted to unlabeled data with a long-tailed distribution. MARS follows a preserve-before-refine design: it first preserves the reliable head-class boundary and then refines the tail and confusable classes. Specifically, to mitigate head-class boundary erosion caused by the erroneous assignment of pseudo-labels, we propose Boundary-Preserving Alignment (BPA). This approach exploits the class-level visual structure of zero-shot CLIP as a fixed reference and suppresses probability increases unsupported by visual evidence before they become training targets, instead of directly fitting the model’s own erroneous predictions. Building upon this, we introduce Margin-aware Self-Refinement (MSR) to further improve tail and confusable classes while preserving the well-learned decision boundary. Adopting the aligned model as a teacher, this strategy leverages the estimated class distribution and the sample-specific margin to dynamically adjust the learning focus, distinguishing valid supervision from noise to actively acquire discriminative features.

In summary, our contributions are as follows:

• To our knowledge, we are the first to study unsupervised long-tailed adaptation (ULTA), a realistic yet under-explored scenario where VLMs are adapted to long-tailed unlabeled data.

• We uncover a distinct phenomenon where existing unsupervised methods suffer severe degradation on head classes due to boundary erosion from forced uniform pseudo-labels, and observe aggravated prediction bias in confusable classes. See more analyses in Sections 3.2 and 4.4.

• We propose a novel model, MARS, to tackle these challenges. Specifically, we design Boundary-Preserving Alignment to mitigate boundary erosion by exploiting the visual structure of zero-shot CLIP, and develop Margin-aware Self-Refinement to adaptively balance teacher guidance and pseudo-label supervision, further improving discriminability for tail and confusable classes.

• Comprehensive experiments on nine benchmarks with different imbalance ratios show that MARS achieves state-of-the-art performance with an average improvement of 4.71 percentage points.

## 2 RELATED WORK

## 2.1 PROMPT LEARNING IN VLMS

Vision-language models (VLMs) (Radford et al., 2021; Jia et al., 2021; Li et al., 2022; Xiao et al., 2024) have achieved remarkable zero-shot generalization by aligning visual and textual modalities. Prompt learning is a parameter-efficient paradigm for adapting these models to downstream tasks. Specifically, CoOp (Zhou et al., 2022b;a) pioneers this direction by optimizing context vectors within the text encoder. VPT (Jia et al., 2022) extends this concept to the visual modality by injecting learnable prompts into the image encoder. To facilitate cross-modal interaction, MaPLe (Khattak et al., 2023) introduces a multi-modal prompting approach to ensure deep alignment between branches. Alternatively, some works focus on refining the encoder’s output features without fine-tuning the backbone. For example, CLIP-Adapter (Gao et al., 2024) employs lightweight residual modules to transform visual and textual features. Despite their success, these methods heavily rely on labeled data, which severely constrains their applicability in unsupervised scenarios.

## 2.2 LEARNING FROM UNLABELED DATA FOR VLMS

To mitigate the reliance on labeled data, various strategies leverage the zero-shot capabilities of VLMs to generate pseudo-labels for self-training. For instance, UPL (Huang et al., 2022) and GRIP (Menghini et al., 2023) focus on improving pseudo-label quality via confidence-based filtering or iterative top-k selection. Similarly, CPL (Zhang et al., 2024) assigns each sample a set of candidate pseudo-labels at each iteration, expecting the true label to be captured within the set. Beyond traditional pseudo-labeling, recent studies explore novel perspectives for mining supervisory signals from unlabeled data. UEO (Liang et al., 2024) employs marginal entropy as a unified metric to regularize the decision boundary, minimizing uncertainty for confident predictions while maximizing it for ambiguous samples. TMP (Li et al., 2025) reformulates the adaptation process as a binary verification task instead of the traditional classification task. CAP (Wang et al., 2025) identifies inherent biases and concept confusion in CLIP zero-shot predictions, rectifying these issues by learning a projection layer that encourages balanced predictions across classes. microCLIP (Silva et al., 2026) combines coarse-to-fine token fusion and saliency-aware pooling for unsupervised CLIP adaptation. However, their reliance on a uniform prior over unlabeled data limits their applicability to long-tailed scenarios.

## 3 METHODOLOGY

## 3.1 PROBLEM FORMULATION AND CHALLENGES

In unsupervised long-tailed adaptation (ULTA), we consider an unlabeled dataset $\mathcal { D } _ { u } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ of size N belonging to $C$ classes. While one can theoretically define $N _ { c }$ as the number of samples for class c and $\gamma = \mathrm { m a x } _ { c } N _ { c } / \mathrm { m i n } _ { c } N _ { c }$ as the imbalance ratio, neither is accessible in practice. The core challenge is that existing methods implicitly assume a uniform prior when generating pseudo-labels, which inevitably leads to severe boundary erosion and prediction bias. Our goal is to achieve robust performance across different imbalance ratios.

## 3.2 ANALYSIS OF MISALIGNMENT

Existing unsupervised methods typically rely on a uniform prior to generate pseudo-labels. In long-tailed scenarios, this assumption forces the model to align visual features with incorrect targets, causing a severe misalignment between the learned features and the true semantic information. We analyze the consequences of this misalignment from the perspective of gradient dynamics.

Consider a head-class sample x with normalized visual feature $\mathbf { v } \in \mathbb { R } ^ { d }$ and ground-truth label $c _ { h }$ which is incorrectly assigned a tail-class pseudo-label $\hat { y } = c _ { t }$ . In the standard self-training framework, this incorrect pseudo-label is treated as a hard target, forcing the visual feature to align with the textual embedding $\mathbf { t } _ { c _ { t } }$ . Formally, we minimize the cross-entropy loss $\mathcal { L } _ { c e } = - \log p ( \bar { c _ { t } } | \mathbf { v } )$ , where the probability $p$ is computed via cosine similarity with textual embeddings $\{ \mathbf { t } _ { c } \} _ { c = 1 } ^ { \top }$ scaled by a temperature parameter τ . The gradient of this loss with respect to the visual feature v is derived as:

![](images/fb5200a5efaf00095bc26f5a3ee103df7de965117d8ef5b5d8deafae9383d304.jpg)  
(a) Zero-shot CLIP

![](images/86aa9d5b5deb456755d45ff2b644066e12e5d556219d83398b140320e9e1dde0.jpg)  
(b) Baseline

![](images/39fbe260b9c0fabe45abf46c34121cf59548b073b46e9e09bc26f8be07b32384.jpg)  
(c) Ours  
Figure 2: Visualization of attention maps for misassigned samples on RESISC45. We select headclass samples where the baseline method incorrectly assigns tail-class pseudo-labels. (a) The model mainly focuses on the foreground object. (b) The model is distracted by the irrelevant background. (c) The model correctly captures the foreground object while reducing background attention.

$$
\nabla _ { \mathbf { v } } \mathcal { L } _ { c e } = \frac { 1 } { \tau } \left( \sum _ { j = 1 } ^ { C } p ( j | \mathbf { v } ) \cdot \mathbf { t } _ { j } - \mathbf { t } _ { c _ { t } } \right) .\tag{1}
$$

Before fitting this incorrect pseudo-label, the model still assigns high confidence to the correct head class $c _ { h }$ , so the probability mass is concentrated on $c _ { h }$ . Consequently, the gradient approximates:

$$
\nabla _ { \mathbf { v } } \mathcal { L } _ { c e } \approx \frac { 1 } { \tau } ( \mathbf { t } _ { c _ { h } } - \mathbf { t } _ { c _ { t } } ) .\tag{2}
$$

The feature vector v is updated by moving against the gradient direction (i.e., $\mathbf { v }  \mathbf { v } - \eta \nabla _ { \mathbf { v } } \mathcal { L } _ { c e } )$ Consequently, the resulting feature shift satisfies $\Delta \mathbf { v } \propto \mathbf { t } _ { c _ { t } } - \mathbf { t } _ { c _ { h } }$ . This derivation illustrates that the gradient drives the visual feature v to align with the distant tail embedding $\mathbf { t } _ { c _ { t } }$ , deviating from its true semantic embedding $\mathbf { t } _ { c _ { h } }$ and thereby eroding the decision boundaries of head classes. Meanwhile, to minimize the loss, the model is forced to exploit irrelevant background noise. As visualized in Fig. 2(b), for these mislabeled samples, existing methods $( \mathrm { e . g . }$ , CPL Zhang et al., 2024) cause the model to ignore the foreground object and focus on the background, thereby degrading feature discriminability.

## 3.3 BOUNDARY-PRESERVING ALIGNMENT

To mitigate this misalignment, we prevent erroneous probability changes from becoming training targets. In self-training, the targets are generated from the model’s own predictions, so an erroneous prediction is fitted again in the next round. We therefore take zero-shot CLIP, which remains fixed during adaptation, as a visual reference, and correct the targets according to whether each probability increase is supported by visual evidence.

Concretely, let $x _ { i } ^ { w }$ denote the weak view of $x _ { i } .$ . We extract the unit-normalized feature $\mathbf { u } _ { i }$ and the predicted distribution $a _ { i }$ of zero-shot CLIP on $\boldsymbol { x } _ { i } ^ { w }$ , and denote the zero-shot class as $c _ { i } = \arg \operatorname* { m a x } _ { c } a _ { i c }$ For each class c, we average the features of the samples with $c _ { i } = c$ and ℓ -normalize the result to obtain the class-level visual anchor $\pmb { \mu } _ { c } .$ , and measure the visual support of sample i for class c by $s _ { i c } = \mathbf { u } _ { i } ^ { \prime } \pmb { \mu } _ { c }$ . These anchors characterize the class-level visual structure of zero-shot CLIP. Unlike the textual embeddings, which always favor the zero-shot class, they compare each sample with the visual centroids of the zero-shot groups and thus provide complementary evidence.

At the start of each alignment epoch, let $p _ { i }$ denote the predicted distribution of the current model on $\boldsymbol { x } _ { i } ^ { w }$ and $b _ { i } = \arg \operatorname* { m a x } _ { c } p _ { i c }$ . For $\textit { j } \neq c _ { i }$ , the probability increase induced by adaptation is $g _ { i j } = \operatorname* { m a x } ( p _ { i j } - a _ { i j } , 0 )$ . We regard this increase as visually unsupported, i.e., $h _ { i j } = 1 , \mathrm { i f } s _ { i c _ { i } } > s _ { i j }$ and set $h _ { i j } = \mathrm { \bar { 0 } }$ otherwise. If $b _ { i } \neq c _ { i }$ and $s _ { i b _ { i } } > s _ { i c _ { i } }$ , the adaptation has likely corrected a zero-shot error, and we set $h _ { i j } = 0$ for all $j .$

Since the verdict of a single sample is noisy and pseudo-label errors tend to recur between the same class pairs (Fig. 1(d)), we aggregate the evidence over the unlabeled data:

$$
\rho _ { c j } = { \frac { \sum _ { i : c _ { i } = c } g _ { i j } h _ { i j } } { \sum _ { i : c _ { i } = c } g _ { i j } } } , \qquad c \neq j .\tag{3}
$$

The corrected target then returns the unsupported part of each increase to the zero-shot class:

$$
q _ { i } = p _ { i } + \sum _ { j \neq c _ { i } } \delta _ { i j } ( { \bf e } _ { c _ { i } } - { \bf e } _ { j } ) , \qquad \delta _ { i j } = \rho _ { c _ { i } j } h _ { i j } g _ { i j } ,\tag{4}
$$

where $\mathbf { e } _ { c }$ is the one-hot vector for class $c .$ Since $0 \leq \delta _ { i j } \leq g _ { i j } \leq p _ { i j } , q _ { i }$ remains a valid distribution. We fix $q _ { i }$ within each epoch and use it to supervise the strong view $x _ { i } ^ { s }$ generated by RandAugment (Cubuk et al., 2020):

$$
\mathcal { L } _ { \mathrm { B P A } } = - \frac { 1 } { B } \sum _ { i \in \mathcal { B } } \sum _ { c = 1 } ^ { C } q _ { i c } \log p ( c \mid x _ { i } ^ { s } ) ,\tag{5}
$$

where $p ( c \mid x _ { i } ^ { s } )$ is the predicted probability of class c on $\boldsymbol { x } _ { i } ^ { s }$ and B is a minibatch of size $B .$

Crucially, let $\ell ( q _ { i } )$ denote the cross-entropy between target $q _ { i }$ and the softmax of the strong-view logits $z _ { i }$ . Compared with the uncorrected target $p _ { i }$ , the corrected target changes the gradient by

$$
\nabla _ { z _ { i } } \ell ( q _ { i } ) - \nabla _ { z _ { i } } \ell ( p _ { i } ) = - \sum _ { j \neq c _ { i } } \delta _ { i j } ( { \bf e } _ { c _ { i } } - { \bf e } _ { j } ) .\tag{6}
$$

Each unsupported increase thus induces a push along $\mathbf { e } _ { c _ { i } } - \mathbf { e } _ { j }$ on the logits. Since $z _ { i c } = \mathbf { v } ^ { \top } \mathbf { t } _ { c } / \tau$ for the visual feature v of $\boldsymbol { x } _ { i } ^ { s }$ , this push moves v toward $\mathbf { t } _ { c _ { i } }$ and away from $\mathbf { t } _ { j } ,$ , counteracting the misaligned gradient in Eq. (2), while supported increases, including those that correct zero-shot errors, are preserved. As shown in Fig. 2(c), BPA restores the attention to the foreground object, and Fig. 3(a) shows that it keeps head-class accuracy stable as the imbalance ratio increases.

## 3.4 MARGIN-AWARE SELF-REFINEMENT

Although boundary-preserving alignment establishes a foundation in the joint feature space, refining the decision boundaries for the tail and confusable classes remains a critical challenge in unsupervised long-tailed scenarios. To address this, instead of directly fitting noisy pseudo-labels that can destroy the boundary, we propose a margin-aware self-refinement strategy that improves the feature space by decoupling well-learned boundary preservation from targeted feature enhancement.

We employ the aligned model from Section 3.3 as a frozen teacher and introduce a learnable adapter as the student. The refinement objective is formulated as follows:

$$
\mathcal { L } _ { \mathrm { r e f i n e } } = \alpha \cdot \mathcal { L } _ { \mathrm { K L } } ( p ^ { T } , p ^ { S } ) + ( 1 - \alpha ) \cdot \mathcal { L } _ { \mathrm { C E } } ( \hat { y } , p ^ { S } ) ,\tag{7}
$$

where $p ^ { T }$ and $p ^ { S }$ denote the predictions of the frozen teacher and the student, respectively. In this formulation, the Kullback-Leibler (KL) divergence loss ${ \mathcal { L } } _ { \mathrm { K I } }$ serves as a safety anchor to preserve the well-learned concepts from the frozen teacher, while the standard cross-entropy loss $\mathcal { L } _ { \mathrm { C E } }$ leverages pseudo-label supervision with $\begin{array} { r } { \hat { y } = \arg \operatorname* { m a x } _ { c } p _ { c } ^ { T } } \end{array}$ to acquire new knowledge. The gating factor $\alpha \in [ 0 , 1 ]$ plays a critical role in balancing knowledge preservation against the learning of new features.

To adaptively modulate $\alpha .$ , we introduce a class-level prior. Specifically, we define the gating factor as $\alpha = ( \overset { \cdot } { r } _ { \hat { y } } ) ^ { \beta }$ , where $\beta$ is a hyperparameter. We fix $\beta = 1$ across all datasets, and the sensitivity analysis in Appendix D.1 further shows that our method is robust to this hyperparameter. Here, $\begin{array} { r } { r _ { \hat { y } } = \frac { \hat { n } _ { \hat { y } } } { \operatorname* { m a x } _ { c } \left( \hat { n } _ { c } \right) } } \end{array}$ represents the normalized frequency, where $\hat { n } _ { \hat { y } }$ denotes the estimated sample count for class yˆ accumulated from the teacher’s soft predictions. High-frequency predictions primarily correspond to head classes, which require a higher α to maintain the model’s well-established boundary. In contrast, low-frequency predictions correspond to tail and confusable classes, which necessitate a lower α to facilitate the acquisition of new knowledge, thereby enhancing feature discriminability.

However, relying solely on class-level priors is insufficient, as it overlooks the quality of individual pseudo-labels within the same class. As illustrated in Fig. 3(c), our empirical observations reveal that correctly pseudo-labeled samples mostly exhibit increasing margins during training, whereas incorrectly pseudo-labeled samples maintain negative margins. Motivated by this, we propose Cumulative Margin Consistency (CMC) to quantify sample quality. Specifically, given the student’s logits $z ^ { ( t ) }$ at epoch t, the margin is defined as:

![](images/1bcbf7c2f4c0291fd7f19b56273c7afc671251159d7bd8d6762088ef003c5756.jpg)  
(a)

![](images/cedce356f0cee7182143fec33160bf03d7c989ec548a61a6ae7f5e7b63792946.jpg)  
(b)

![](images/9ccc08d683b2cf52d92a6bed432124a99c0314cecc41b2aa196ea1efa7015953.jpg)  
(c)  
Figure 3: (a) BPA preserves head-class accuracy on OxfordPets as the imbalance ratio increases. (b) The complete MARS pipeline maintains stable overall accuracy on OxfordPets across imbalance ratios. (c) Evolution of classification margins during training. Samples with correct pseudo-labels (Correct PL) consistently exhibit increasing margins, whereas mislabeled samples (Wrong PL) fluctuate around zero or remain negative.

$$
M ^ { ( t ) } ( x , \hat { y } ) = z _ { \hat { y } } ^ { ( t ) } ( x ) - \operatorname* { m a x } _ { j \neq \hat { y } } z _ { j } ^ { ( t ) } ( x ) .\tag{8}
$$

We calculate the exponential moving average of this margin with momentum λ as $\mathbf { C M C } ( x , \hat { y } )$ and refine α into a sample-specific factor αˆ to mitigate the impact of noisy pseudo-labels:

$$
\hat { \alpha } = \frac { 1 } { 2 } \left( r _ { \hat { y } } ^ { \beta } + \left( 1 - \mathcal { N } ( \mathrm { C M C } ( x , \hat { y } ) ) \right) \right) ,\tag{9}
$$

where $\mathcal { N } ( \cdot )$ is the min-max normalization function. The final refinement objective is formulated as:

$$
\mathcal { L } _ { \mathrm { r e f i n e } } = \hat { \alpha } \cdot \mathcal { L } _ { \mathrm { K L } } ( p ^ { T } , p ^ { S } ) + ( 1 - \hat { \alpha } ) \cdot \mathcal { L } _ { \mathrm { C E } } ( \hat { y } , p ^ { S } ) .\tag{10}
$$

The resulting stability across imbalance ratios is reflected in Fig. 3(b): the complete MARS pipeline retains its overall accuracy as the long-tailed distribution becomes more severe.

## 3.5 MODEL TRAINING AND PREDICTION

Our model consists of Boundary-Preserving Alignment (BPA) and Margin-aware Self-Refinement (MSR). The training process first performs boundary-preserving alignment with corrected weak-view targets, and then conducts margin-aware self-refinement with the frozen teacher to selectively refine the remaining hard classes and samples. The detailed training procedure is provided in Appendix A. Notably, both the teacher and student adapters are appended to the textual encoder. The pre-trained image and text encoders are kept frozen, and only the prompts and adapters are learned.

Upon completion of training, the teacher adapter is discarded. For an unseen test sample $x ^ { * }$ , the predicted label $y ^ { * }$ is obtained by computing the cosine similarity between the visual feature $\mathbf { v } ^ { * }$ extracted by the refined encoder and the pre-computed textual embeddings $\{ \mathbf { t } _ { c } \} _ { c = 1 } ^ { C }$

$$
y ^ { * } = \underset { c \in \{ 1 , \ldots , C \} } { \arg \operatorname* { m a x } } \left( \frac { \mathbf { v } ^ { * } \cdot \mathbf { t } _ { c } } { \tau } \right) .\tag{11}
$$

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTING

Datasets. We conduct extensive experiments on nine classification datasets from diverse domains, including CUB (CUB.; Wah et al., 2011), RESISC45 (Res.; Cheng et al., 2017), FGVCAircraft (FA.; Maji et al., 2013), Caltech101 (Cal.; Fei-Fei et al., 2004), EuroSAT (ES.; Helber et al., 2019), Flowers102 (Flw.; Nilsback & Zisserman, 2008), Food101 (Food.; Bossard et al., 2014), OxfordPets (Pets.; Parkhi et al., 2012), and StanfordCars (Cars.; Krause et al., 2013). For the long-tailed settings, the training data is generated by down-sampling the original datasets to follow an exponential decay of different ratios. Specifically, the maximum number of samples per class is set to 100 (or the original class size), while the minimum is constrained to 2. Following the definition in Section 3.1, we set the imbalance ratio $\gamma \in \{ 1 0 , 2 0 , 5 0 \}$ to simulate different long-tailed scenarios. For the fine-grained analysis, classes are divided into Head, Medium, and Tail groups using a 15/30/55 split by training-set frequency. The details of dataset statistics can be found in Appendix B.

Table 1: Comparison of the test accuracy (%) across three seeds on nine benchmark datasets with different imbalance ratios (10, 20, 50). The results are reported as mean over three random seeds, with the best results presented in bold. Zero-shot CLIP is abbreviated as ZS CLIP.
<table><tr><td colspan="10">Imbalance Ratio = 10</td></tr><tr><td>Method</td><td>CUB.</td><td>Res.</td><td>FA.</td><td>Cal.</td><td>ES.</td><td>Flw.</td><td>Food.</td><td>Pets.</td><td>Cars.</td><td>Avg.</td></tr><tr><td>ZS CLIP</td><td> $5 1 . 8 2 _ { 0 . 0 0 }$ </td><td> $5 4 . 4 8 _ { 0 . 0 0 }$ </td><td>17.580.00</td><td>90.300.00</td><td>32.880.00</td><td> $6 3 . 6 7 _ { 0 . 0 0 }$ </td><td> $7 8 . 8 0 _ { 0 . 0 0 }$ </td><td>84.360.00</td><td>58.160.00</td><td> $5 9 . 1 2 _ { 0 . 0 0 }$ </td></tr><tr><td>FPL (NeurIPS&#x27;23)</td><td>51.590.23</td><td>62.580.86</td><td>19.050.37</td><td> $9 0 . 6 7 _ { 0 . 3 9 }$ </td><td>55.491.35</td><td>61.301.72</td><td>78.100.15</td><td>86.440.33</td><td>58.100.28</td><td>62.590.63</td></tr><tr><td>GRIP (NeurIPS&#x27;23) CPL (ICML&#x27;24)</td><td>48.010.68 49.270.69</td><td>61.950.99</td><td>18.250.24</td><td>88.980.62</td><td>58.445.52</td><td>64.001.55</td><td>73.060.49</td><td>84.150.29</td><td>50.930.36</td><td>60.861.19</td></tr><tr><td></td><td>53.560.11</td><td>61.421.22</td><td>17.580.32</td><td>82.770.37</td><td>63.496.24</td><td>61.911.04</td><td>72.110.23</td><td>86.280.30</td><td>49.310.59</td><td>60.461.22</td></tr><tr><td>UEO (ICML&#x27;24)</td><td></td><td>60.070.01</td><td>18.730.15</td><td>92.740.08</td><td>43.680.14</td><td>65.710.07</td><td>80.670.04</td><td>87.940.09</td><td>58.030.18</td><td>62.350.09</td></tr><tr><td>TMP (ICML&#x27;25)</td><td>18.490.18</td><td>39.780.26</td><td>10.970.12</td><td> $8 2 . 7 6 _ { 0 . 2 6 }$ </td><td>16.800.30</td><td>13.160.29</td><td>59.640.65</td><td>33.963.59</td><td>19.460.10</td><td>32.780.64</td></tr><tr><td>CAP (ICML&#x27;25)</td><td>51.750.55</td><td> $6 8 . 8 3 _ { 2 . 6 6 }$ </td><td> $1 6 . 0 2 _ { 2 . 0 7 }$ </td><td> $8 9 . 7 2 _ { 3 . 5 1 }$ </td><td>66.372.67</td><td>60.881.73</td><td>80.680.37</td><td>86.641.41</td><td>58.260.43</td><td>64.351.71</td></tr><tr><td>microCLIP (ACL’26)</td><td>49.880.59</td><td> $6 7 . 9 8 _ { 0 . 8 7 }$ </td><td> $1 7 . 2 8 _ { 0 . 8 7 }$ </td><td> $9 3 . 0 1 _ { 0 . 1 2 }$ </td><td> $5 1 . 7 6 _ { 1 . 2 9 }$ </td><td> $6 7 . 9 6 _ { 1 . 3 4 }$ </td><td> $7 9 . 9 4 _ { 0 . 1 5 }$ </td><td>88.770.94</td><td>58.940.34</td><td> $6 3 . 9 5 _ { 0 . 2 4 }$ </td></tr><tr><td>MARS (Ours)</td><td>54.840.35</td><td> ${ \bf 6 8 . 8 5 _ { 0 . 4 1 } }$ </td><td>20.240.27</td><td>94.060.51</td><td>74.370.29</td><td>68.951.00</td><td> $\mathbf { 8 1 . 5 2 _ { 0 . 1 7 } }$ </td><td> $\mathbf { 9 0 . 9 7 _ { 0 . 4 2 } }$ </td><td> $\mathbf { 6 2 . 3 8 _ { 0 . 5 2 } }$ </td><td>一  $\mathbf { 6 8 . 4 7 _ { 0 . 2 0 } }$ </td></tr><tr><td colspan="9">Imbalance Ratio = 20</td></tr><tr><td>Method</td><td>CUB.</td><td>Res.</td><td>FA.</td><td>Cal.</td><td>ES.</td><td>Flw.</td><td>Food.</td><td>Pets.</td><td>Cars.</td><td></td></tr><tr><td>ZS CLIP</td><td> $5 1 . 8 2 _ { 0 . 0 0 }$ </td><td> $5 4 . 4 8 _ { 0 . 0 0 }$ </td><td>17.580.00</td><td> $9 0 . 3 0 _ { 0 . 0 0 }$ </td><td>32.880.00</td><td>63.670.00</td><td> $7 8 . 8 0 _ { 0 . 0 0 }$ </td><td>84.360.00</td><td>58.160.00</td><td>Avg. 59.120.00</td></tr><tr><td>FPL (NeurIPS&#x27;23) GRIP (NeurIPS&#x27;23) CPL (ICML&#x27;24) UEO (ICML&#x27;24)</td><td>51.040.26 47.630.75 48.460.39 53.200.09</td><td>60.030.75 59.271.74 58.450.32 58.690.11</td><td>18.460.21 17.200.20 16.810.23 17.281.01</td><td>90.300.09 87.750.18 79.993.02 92.750.06</td><td>51.891.32 60.012.57 64.350.34 43.150.03</td><td>60.871.52 62.260.91 60.630.94 66.130.04</td><td>77.480.15 70.860.27 70.140.30 80.480.09</td><td>86.770.33 81.850.67 85.300.31 87.610.10</td><td>57.510.41 49.650.93 47.830.61 57.820.12</td><td>61.600.56 59.610.91 59.110.72</td></tr><tr><td>TMP (ICML&#x27;25) CAP (ICML&#x27;25) microCLIP (ACL&#x27;26)</td><td>15.610.52 49.890.57 49.110.38</td><td>32.710.73 59.943.42  $6 6 . 4 7 _ { 0 . 5 8 }$ </td><td>8.100.05 13.471.82  $1 6 . 7 0 _ { 0 . 2 3 }$ </td><td> $6 9 . 8 2 _ { 0 . 6 7 }$  88.154.36  $9 2 . 8 9 _ { 0 . 2 0 }$ </td><td>12.271.69 62.070.75  $5 2 . 6 5 _ { 4 . 8 1 }$ </td><td>9.510.62 59.960.62 66.531.55</td><td>50.460.45 78.881.49  $7 9 . 4 6 _ { 0 . 1 9 }$ </td><td>22.031.43 82.053.28 87.980.75</td><td>15.110.14 56.080.49 56.890.78</td><td>61.900.18 26.180.70 61.171.87  $6 3 . 1 9 _ { 0 . 5 4 }$ </td></tr><tr><td colspan="9">MARS (Ours) 53.800.82 68.420.58 20.450.84 94.040.15 72.173.73</td></tr><tr><td></td><td>CUB.</td><td>Res.</td><td>FA.</td><td>Imbalance Ratio = 50 Cal.</td><td>ES.</td><td>Flw.</td><td></td><td>Pets.</td><td>Cars.</td><td>61.180.86 67.690.52</td></tr><tr><td colspan="11">Method Food.</td></tr><tr><td>ZS CLIP</td><td> $5 1 . 8 2 _ { 0 . 0 0 }$ </td><td> $5 4 . 4 8 _ { 0 . 0 0 }$ </td><td> $1 7 . 5 8 _ { 0 . 0 0 }$ </td><td> $9 0 . 3 0 _ { 0 . 0 0 }$ </td><td> $3 2 . 8 8 _ { 0 . 0 0 }$ </td><td> $6 3 . 6 7 _ { 0 . 0 0 }$ </td><td> $7 8 . 8 0 _ { 0 . 0 0 }$ </td><td> $8 4 . 3 6 _ { 0 . 0 0 }$ </td><td>58.160.00</td><td>Avg.  $5 9 . 1 2 _ { 0 . 0 0 }$ </td></tr><tr><td>FPL (NeurIPS&#x27;23) GRIP (NeurIPS&#x27;23)</td><td>50.600.32 47.570.28</td><td> $5 8 . 0 7 _ { 0 . 5 1 }$   $5 6 . 4 8 _ { 0 . 9 8 }$ </td><td>17.950.66 16.470.24</td><td> $8 9 . 6 7 _ { 0 . 7 5 }$ </td><td> $4 9 . 5 3 _ { 2 . 3 2 }$  53.843.69</td><td>60.092.26</td><td> $7 6 . 7 8 _ { 0 . 3 5 }$ </td><td> $8 6 . 4 1 _ { 0 . 5 4 }$ </td><td>56.760.29</td><td>60.650.89</td></tr><tr><td>CPL (ICML&#x27;24)</td><td>47.880.43</td><td> $5 6 . 6 4 _ { 0 . 2 8 }$ </td><td>16.430.23</td><td>86.520.47  $7 7 . 7 3 _ { 2 . 7 5 }$ </td><td>59.766.11</td><td>60.991.23 60.580.93</td><td>68.990.73 68.240.19</td><td>81.140.80 82.940.10</td><td>49.260.84 46.710.80</td><td>57.921.03 57.431.31</td></tr><tr><td>UEO (ICML’24)</td><td>52.460.06</td><td>56.960.17</td><td>17.490.52</td><td></td><td>42.370.05</td><td>66.010.07</td><td></td><td></td><td></td><td></td></tr><tr><td>TMP (ICML&#x27;25)</td><td>12.860.05</td><td>26.210.12</td><td></td><td> $9 2 . 7 7 _ { 0 . 0 6 }$ </td><td></td><td></td><td>80.240.05</td><td>86.990.03</td><td>57.230.17</td><td>61.390.13</td></tr><tr><td>CAP (ICML&#x27;25)</td><td>49.110.27</td><td></td><td>5.530.37</td><td>60.450.74</td><td>11.331.64</td><td>8.210.25</td><td> $3 8 . 6 3 _ { 0 . 5 2 }$ </td><td>14.311.35</td><td>12.490.29</td><td>21.110.59</td></tr><tr><td></td><td></td><td>57.291.31</td><td>12.771.70</td><td>86.665.99</td><td>52.881.60</td><td>58.430.68</td><td>77.893.03</td><td> $7 7 . 2 9 _ { 5 . 6 1 }$ </td><td>53.100.30</td><td>58.382.28</td></tr><tr><td>microCLIP (ACL&#x27;26)</td><td>48.110.60</td><td> $6 3 . 7 3 \mathrm { _ { 0 . 7 9 } }$ </td><td> $1 5 . 3 6 _ { 0 . 4 2 }$ </td><td> $9 2 . 2 5 _ { 0 . 5 1 }$ </td><td>48.163.79</td><td> $6 4 . 8 8 _ { 0 . 6 8 }$ </td><td> $7 9 . 2 2 _ { 0 . 2 9 }$ </td><td>87.930.43</td><td> $5 5 . 4 7 _ { 0 . 8 3 }$ </td><td> $6 1 . 6 8 _ { 0 . 5 3 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MARS (Ours)</td><td> $\mathbf { 5 4 . 2 9 _ { 0 . 6 8 } }$ </td><td> ${ \bf 6 8 . 1 6 _ { 0 . 4 7 } }$ </td><td> $2 0 . 0 5 _ { 0 . 2 6 }$ </td><td> $9 3 . 1 8 _ { 0 . 0 8 }$ </td><td> $\mathbf { 6 9 . 2 2 _ { 5 . 6 0 } }$ </td><td> ${ \bf 6 8 . 1 3 _ { 1 . 0 0 } }$ </td><td> $\mathbf { 8 1 . 2 4 _ { 0 . 4 7 } }$ </td><td> $\mathbf { 9 0 . 0 0 _ { 0 . 7 9 } }$ </td><td> $\mathbf { 6 0 . 5 5 _ { 0 . 4 6 } }$ </td><td> ${ \bf 6 7 . 2 0 _ { 0 . 9 0 } }$ </td></tr></table>

Baselines. We compare our method with seven existing methods, namely, Few Pseudo-Labels (FPL; Menghini et al., 2023), Grow and Refine Iteratively Pseudo-labels (GRIP; Menghini et al., 2023), Candidate Pseudo-Labeling (CPL; Zhang et al., 2024), Universal Entropy Optimization (UEO; Liang et al., 2024), True-false labels via Multimodal Prompt retrieving (TMP; Li et al., 2025), Concept-Adaptive Pseudo-labeling (CAP; Wang et al., 2025), and Coarse-Fine Token Fusion (microCLIP; Silva et al., 2026). For fair comparison, we standardize the visual backbone to OpenAI CLIP ViT-B/32 and use the same training dataset and evaluation metrics. All other settings follow the official recommendations.

Implementation details. Following the standard protocol, we set the multi-modal prompts as MaPLe (Khattak et al., 2023), with a depth of 9 and a context length of 2, initialized with the template “a photo of a ”. The weak view follows the standard CLIP preprocessing, while the strong view is generated by RandAugment (Cubuk et al., 2020). We fix β = 1 and the momentum coefficient (λ = 0.95) across all datasets. For a comprehensive evaluation of different VLMs, we also report results using OpenCLIP ViT-B/32 (LAION-2B) (Cherti et al., 2023) and SigLIP ViT-B/16 (Zhai et al., 2023). Additional reproduction details can be found in Appendix C.

Table 4: Ablation results of Boundary-Preserving Alignment (BPA) and Margin-aware Self-Refinement (MSR) with different imbalance ratios. The improvement is indicated by ↑.
<table><tr><td rowspan="2">BPA</td><td rowspan="2">MSR</td><td colspan="3">Imbalance Ratio = 10</td><td colspan="3">Imbalance Ratio = 20</td><td colspan="3">Imbalance Ratio = 50</td><td rowspan="2">Avg.</td></tr><tr><td>Res.</td><td>Flw.</td><td>ES.</td><td>Res.</td><td>Flw.</td><td>ES.</td><td>Res.</td><td>Flw.</td><td>ES.</td></tr><tr><td>x</td><td>x</td><td>54.48</td><td>63.67</td><td>32.88</td><td>54.48</td><td>63.67</td><td>32.88</td><td>54.48</td><td>63.67</td><td>32.88</td><td>50.34</td></tr><tr><td>√</td><td>x</td><td>67.05</td><td>65.23</td><td>57.26</td><td>64.69</td><td>63.98</td><td>61.92</td><td>63.34</td><td>64.20</td><td>50.68</td><td>62.04 ↑11.70</td></tr><tr><td>x</td><td>√</td><td>61.50</td><td>68.42</td><td>52.16</td><td>61.71</td><td>67.33</td><td>52.26</td><td>58.65</td><td>66.92</td><td>52.18</td><td>60.13 ↑ 9.79</td></tr><tr><td>√</td><td>√</td><td>69.43</td><td>69.41</td><td>74.40</td><td>68.52</td><td>67.73</td><td>74.22</td><td>68.66</td><td>67.98</td><td>73.68</td><td>70.45↑20.11</td></tr></table>

## 4.2 MAIN RESULTS

Our method achieves SOTA classification accuracy. Table 1 demonstrates that our method outperforms both zero-shot CLIP and existing unsupervised methods by a substantial margin across nine benchmark datasets under varying imbalance ratios (i.e., 10, 20, 50). Specifically, the performance of our proposed method exceeds the previous state-of-the-art method by an average of 4.12, 4.50, and 5.52 percentage points (pp) across three seeds under imbalance ratios of 10, 20, and 50, respectively. In the most challenging scenario with an imbalance ratio of 50, our method demonstrates remarkable robustness. While CAP suffers a 5.97 pp drop as the ratio increases from 10 to 50, our method exhibits high stability with a minimal decline of only 1.27 pp. Notably, at an imbalance ratio of 50, four of the seven baselines even fall below zero-shot CLIP on average, whereas MARS still improves over it by 8.08 pp. Moreover, under an imbalance ratio of 50, our method improves RESISC45 accuracy over the best existing method by 4.43 pp. Furthermore, pairwise t-tests at a 0.05 significance level using three corresponding seed results show that our method achieves significant improvements in 84.7% of cases (160 out of 189), further confirming its effectiveness in ULTA. Detailed results can be found in Appendix D.2.

Our method maintains performance stability and robustness under a uniform distribution. Using the same comparison protocol as Section 4.1, we further evaluate our method on three datasets (Caltech101, Flowers102, and OxfordPets) in a uniform setting. As reported in Table 2, MARS outperforms existing unsupervised methods. Specifically, MARS achieves the highest average accuracy of 86.27%, surpassing the leading baseline by 0.30 pp. This validates that our method enhances feature discriminability without compromising robustness, ensuring reliable performance across unknown distributions.

Our method is effective across different VLMs. To verify the universality of our method, we extend our evaluation to different vision-language models. As illustrated in Table 3, our method consistently improves performance on OpenCLIP across four benchmarks. Specifically, it achieves an average accuracy gain of 8.79 pp over the best-performing baseline for Open-CLIP across the three imbalance ratios. These results indicate that the proposed model serves as a generalized solution that enhances the robustness of various VLMs against long-tailed distributions. Results with imbalance ratios of 10 and 50 and additional results on SigLIP can be found in Appendix D.3.

Table 2: Comparison of test accuracy (%) under the balanced setting.
<table><tr><td>Method</td><td>Cal.</td><td>Flw.</td><td>Pets.</td><td>Avg.</td></tr><tr><td>ZS CLIP</td><td>90.30</td><td>63.67</td><td>84.36</td><td>79.44</td></tr><tr><td>FPL</td><td>92.08</td><td>67.47</td><td>87.41</td><td>82.32</td></tr><tr><td>CPL</td><td>92.37</td><td>68.20</td><td>90.32</td><td>83.63</td></tr><tr><td>UEO</td><td>92.98</td><td>66.82</td><td>89.13</td><td>82.98</td></tr><tr><td>CAP</td><td>94.36</td><td>72.35</td><td>91.22</td><td>85.98</td></tr><tr><td>microCLIP</td><td>94.12</td><td>72.11</td><td>89.92</td><td>85.38</td></tr><tr><td>MARS</td><td>94.89</td><td>72.35</td><td>91.58</td><td>86.27</td></tr></table>

Table 3: Test accuracy (%) using OpenCLIP with an imbalance ratio of 20. Zero-shot OpenCLIP is abbreviated as ZS OCLIP.
<table><tr><td>Method</td><td>Res.</td><td>Flw.</td><td>ES.</td><td>Pets.</td><td>Avg.</td></tr><tr><td>ZS OCLIP</td><td>63.90</td><td>71.43</td><td>51.74</td><td>90.62</td><td>69.42</td></tr><tr><td>FPL</td><td>60.59</td><td>66.04</td><td>50.32</td><td>87.54</td><td>66.12</td></tr><tr><td>GRIP</td><td>63.25</td><td>62.08</td><td>61.24</td><td>80.84</td><td>66.85</td></tr><tr><td>CPL</td><td>62.68</td><td>63.41</td><td>64.44</td><td>84.60</td><td>68.78</td></tr><tr><td>UEO</td><td>59.57</td><td>70.35</td><td>52.06</td><td>88.88</td><td>67.72</td></tr><tr><td>TMP</td><td>41.07</td><td>13.60</td><td>10.00</td><td>47.75</td><td>28.11</td></tr><tr><td>CAP</td><td>63.25</td><td>61.68</td><td>63.30</td><td>80.54</td><td>67.19</td></tr><tr><td>microCLIP</td><td>66.55</td><td>66.16</td><td>58.56</td><td>89.21</td><td>70.12</td></tr><tr><td>MARS</td><td>73.12</td><td>74.22</td><td>80.88</td><td>92.40</td><td>80.16</td></tr></table>

## 4.3 ABLATION STUDIES

All components contribute to the proposed method. As shown in Table 4, both Boundary-Preserving Alignment (BPA) and Margin-aware Self-Refinement (MSR) individually result in noticeable performance improvements over zero-shot CLIP across all settings. Specifically, BPA and MSR yield substantial average improvements of 11.70 pp and 9.79 pp, respectively. Compared with MSR,

BPA exhibits a larger improvement. A possible reason is that BPA mitigates feature drift induced by noisy pseudo-labels and effectively prevents severe boundary erosion, which is the primary cause of performance degradation. Moreover, the combination of the two components consistently yields the highest accuracy, with a total boost of 20.11 pp, highlighting their complementary effects on improving feature discriminability. The full model also stays stable across imbalance ratios, dropping only 0.97 pp from 10 to 50. We further examine the impact of visual encoder capacity with a ViT-B/16 backbone in Appendix D.4, where MARS outperforms the strongest baseline by 3.82 pp on average.

## 4.4 MORE ANALYSES

Does boundary-preserving alignment (BPA) effectively reduce head-class boundary erosion? To quantify the impact of BPA, we visualize the density curve of logit margin changes (∆m) for CPL

and our method in Fig. 4. The negative shift in the head-class margin reveals that CPL enforces a uniform constraint at the cost of eroding the headclass boundary. In contrast, our method rectifies this issue by ensuring that the margin distributions for all classes are shifted to the positive region $( \Delta m > 0 )$ , preventing boundary erosion. BPA also reduces head-to-tail misassignment: on RE-SISC45 with $\gamma = 5 0$ , CPL increases the number of head-class samples predicted as tail classes from 428 to 1,060, whereas BPA reduces it to 392. More datasets and a sample-level margin analysis can be found in Appendix D.6.

![](images/de124b8cd50fcc46b3df71b4d4c11b045197ad261fb1da10306b6e5699ea74e4.jpg)  
(a) Existing method

![](images/c18dd9029fc0cb890d3167ec131c522837f4a7f570c22162af0b302355638087.jpg)  
(b) Ours  
Figure 4: Density curve of logit margin changes (∆m). (a) Existing methods cause a negative shift for head classes. (b) Our method maintains positive margin changes.

Does margin-aware self-refinement (MSR) effectively improve tail and confusable classes without compromising head classes? As shown in Fig. 5, to verify whether MSR effectively addresses

this dilemma, we analyze the performance across different class groups and the frequency consistency (i.e., the alignment between predicted and ground-truth class distributions) in confusable classes. Specifically, our method achieves substantial accuracy gains in the tail and medium groups while preserving performance on head classes, confirming that discriminability is enhanced without knowledge forgetting. Furthermore, the significantly higher frequency consistency in confusable classes demonstrates that MSR effectively rectifies prediction errors caused by the model’s inherent bias. More fine-grained results and analyses can be found in Appendix D.5.

![](images/89a51b67e817980ec2474c0030129e771df7687d94e95c1747fdc910555ec5de.jpg)  
(a)

![](images/e072607acaafc2ef7ab4c8624abd1ef18a7adc938d96999d798a8b13975f37af.jpg)  
(b)  
Figure 5: Evaluation of margin-aware selfrefinement on RESISC45. (a) Accuracy across head, medium, and tail groups. (b) Frequency consistency on confusable classes.

## 4.5 MORE CHALLENGING SETTINGS

Our method remains effective on large-scale lo scalability and robustness of our method, we run additional experiments on large-scale long-tailed benchmarks (Liu et al., 2019; Yang et al., 2022; Van Horn et al., 2018; 2021) (Places-LT, ImageNet-LT, and an iNaturalist2018 subset, with imbalance ratios of 996, 256, and 500). As shown in Table 5, MARS achieves the best performance on all three benchmarks, improving the average accuracy by 2.51 pp. Notably, it improves the best baseline by 3.16 pp on iNaturalist2018. Base-to-new generalization results are provided in Appendix D.7.

Table 5: Comparison of test accuracy (%) on large-scale long-tailed datasets.
<table><tr><td>Method</td><td>Places-LT</td><td>ImageNet-LT</td><td>iNaturalist2018</td><td>Avg.</td></tr><tr><td>ZS CLIP</td><td>38.33</td><td>59.71</td><td>69.33</td><td>55.79</td></tr><tr><td>FPL</td><td>39.31</td><td>60.05</td><td>69.50</td><td>56.29</td></tr><tr><td>GRIP</td><td>36.64</td><td>59.92</td><td>66.60</td><td>54.39</td></tr><tr><td>CPL</td><td>38.38</td><td>55.77</td><td>65.30</td><td>53.15</td></tr><tr><td>UEO</td><td>39.71</td><td>60.23</td><td>70.17</td><td>56.70</td></tr><tr><td>TMP</td><td>32.18</td><td>48.32</td><td>42.17</td><td>40.89</td></tr><tr><td>CAP</td><td>37.86</td><td>61.34</td><td>70.00</td><td>56.40</td></tr><tr><td>microCLIP</td><td>40.39</td><td>58.62</td><td>69.00</td><td>56.00</td></tr><tr><td>MARS</td><td>41.35</td><td>62.96</td><td>73.33</td><td>59.21</td></tr></table>

## 5 CONCLUSION

In this paper, we investigate unsupervised long-tailed adaptation (ULTA) of vision-language models. We find that imposing uniform constraints erodes the head-class boundary while reinforcing prediction bias in confusable classes. To address this, we propose margin-aware refinement with structural alignment (MARS), which integrates boundary-preserving alignment to maintain the head-class boundary with margin-aware self-refinement to enhance the discriminability of tail and confusable classes. Extensive experiments across nine benchmarks with varying imbalance ratios validate the effectiveness of MARS, with an average improvement of 4.71 percentage points.

## REFERENCES

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In Proceedings ofthe European Conference on Computer Vision (ECCV), pp. 446–461, 2014.

Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings ofthe IEEE, 105(10):1865–1883, 2017.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2818–2829, 2023.

Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. RandAugment: Practical automated data augmentation with a reduced search space. In Advances in Neural Information Processing Systems (NeurIPS), pp. 18613–18624, 2020.

Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 178–178, 2004.

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. CLIP-Adapter: Better vision-language models with feature adapters. International Journal ofComputer Vision (IJCV), 132(2):581–595, 2024.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.

Shih-Cheng Huang, Liyue Shen, Matthew P. Lungren, and Serena Yeung. GLoRIA: A multimodal global-local representation learning framework for label-efficient medical image recognition. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3942– 3951, 2021.

Tony Huang, Jack Chu, and Fangyun Wei. Unsupervised prompt learning for vision-language models. arXiv preprint arXiv:2204.03649, 2022.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the International Conference on Machine Learning (ICML), pp. 4904–4916, 2021.

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In Proceedings ofthe European Conference on Computer Vision (ECCV), pp. 709–727, 2022.

Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. MaPLe: Multi-modal prompt learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19113–19122, 2023.

Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3D object representations for finegrained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops (ICCVW), pp. 554–561, 2013.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning (ICML), pp. 12888–12900, 2022.

Zhongnian Li, Jinghao Xu, Peng Ying, Meng Wei, and Xinzheng Xu. Learning from true-false labels via multi-modal prompt retrieving. In Proceedings ofthe International Conference on Machine Learning (ICML), pp. 36652–36666, 2025.

Jian Liang, Lijun Sheng, Zhengbo Wang, Ran He, and Tieniu Tan. Realistic unsupervised CLIP fine-tuning with universal entropy optimization. In Proceedings of the International Conference on Machine Learning (ICML), pp. 29667–29681, 2024.

Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, and Stella X. Yu. Largescale long-tailed recognition in an open world. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2537–2546, 2019.

Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

Cristina Menghini, Andrew Delworth, and Stephen Bach. Enhancing CLIP with CLIP: Exploring pseudolabeling for limited-label prompt tuning. In Advances in Neural Information Processing Systems (NeurIPS), pp. 60984–61007, 2023.

Muhammad Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Mateusz Kozinski, Rogerio Feris, and Horst Bischof. LaFTer: Label-free tuning of zero-shot classifier using language and unlabeled image collections. In Advances in Neural Information Processing Systems (NeurIPS), pp. 5765–5777, 2023.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP), pp. 722–729, 2008.

Yassine Ouali, Adrian Bulat, Brais Martinez, and Georgios Tzimiropoulos. Black box few-shot adaptation for vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15534–15546, 2023.

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3498–3505, 2012.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML), pp. 8748–8763, 2021.

Jiang-Xin Shi, Chi Zhang, Tong Wei, and Yu-Feng Li. Efficient and long-tailed generalization for pre-trained vision-language model. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pp. 2663–2673, 2024.

Sathira Silva, Eman Ali, Chetan Arora, and Muhammad Haris Khan. microCLIP: Unsupervised CLIP adaptation via coarse-fine token fusion for fine-grained image classification. In Findings of the Associationfor Computational Linguistics: ACL 2026, pp. 33277–33294, 2026.

Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The iNaturalist species classification and detection dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8769–8778, 2018.

Grant Van Horn, Elijah Cole, Sara Beery, Kimberly Wilber, Serge Belongie, and Oisin Mac Aodha. Benchmarking representation learning for natural world image collections. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12884–12893, 2021.

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 dataset. Technical report, California Institute of Technology, 2011.

Fuying Wang, Yuyin Zhou, Shujun Wang, Varut Vardhanabhuti, and Lequan Yu. Multi-granularity cross-modal alignment for generalized medical visual representation learning. In Advances in Neural Information Processing Systems (NeurIPS), pp. 33536–33549, 2022.

Yuchen Wang, Xuefeng Bai, Xiucheng Li, Weili Guan, Liqiang Nie, and Xinyang Chen. Handling imbalanced pseudolabels for vision-language models with concept alignment and confusion-aware calibrated margin. In Proceedings of the International Conference on Machine Learning (ICML), pp. 62357–62372, 2025.

Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4818–4829, 2024.

Lu Yang, He Jiang, Qing Song, and Jun Guo. A survey on long-tailed visual recognition. International Journal ofComputer Vision (IJCV), 130(7):1837–1872, 2022.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11975–11986, 2023.

Jiahan Zhang, Qi Wei, Feng Liu, and Lei Feng. Candidate pseudolabel learning: Enhancing visionlanguage models by prompt tuning with unlabeled data. In Proceedings of the International Conference on Machine Learning (ICML), pp. 60004–60020, 2024.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16816–16825, 2022a.

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. International Journal ofComputer Vision (IJCV), 130(9):2337–2348, 2022b.

## TABLE OF CONTENTS FOR THE APPENDIX

A Training Process of MARS 14   
B Dataset Details . 14   
C Implementation Details . . 14   
D More Experiment Results and Analyses 15   
D.1 Sensitivity of Hyperparameters 15   
D.2 Statistical Significance Analysis . .15   
D.3 Comparison with Different VLMs . . 15   
D.4 Comparison with Different Visual Backbones 16   
D.5 Fine-grained Results across Head, Medium, and Tail Groups . . . 16   
D.6 Head-to-Tail Misassignment Analysis . .17   
D.7 Base-to-new Generalization . .17

## A TRAINING PROCESS OF MARS

Algorithm 1 provides the complete training procedure of MARS.

Algorithm 1 Training Process of MARS   
1: Input: Unlabeled dataset $\mathcal { D } _ { u } ,$ alignment epochs $T _ { a l i g n } ,$ and maximum epochs $T _ { m a x } .$   
2: Output: Trained prompts $\theta _ { v i s u a l } , \theta _ { t e x t u a l }$ and student textual adapter $\phi _ { S } .$   
3: Initialize prompts $\theta _ { t e x t u a l } , \theta _ { v i s u a l }$ and teacher textual adapter $\phi _ { T }$   
4: Cache zero-shot features and predictions on the weak view, and construct class anchors.   
5: // Boundary-Preserving Alignment (Sec. 3.3)   
6: for epoch = 1 to $T _ { a l i g n }$ do   
7: For all $x _ { i } \in \mathcal { D } _ { u } ,$ obtain frozen $a _ { i }$ and current $p _ { i }$ on the same weak view.   
8: Compute visual conditions $h _ { i j } ,$ pairwise coefficients $\rho _ { c j }$ by Eq. (3), and targets by Eq. (4).   
9: With fixed $q _ { i } ,$ , update $\theta _ { t e x t u a l } , \theta _ { v i s u a l } , \phi _ { T }$ on strong-view minibatches via Eq. (5).   
10: end for   
11: // Margin-aware Self-Refinement (Sec. 3.4)   
12: Freeze teacher ${ \bf \dot { s } }$ predictions, $\theta _ { v i s u a l }$ and $\phi _ { T } ;$ compute class prior $r _ { \hat { y } }$ and initialize $\phi _ { S }$   
13: for epoch = $T _ { a l i g n } + 1$ to $T _ { m a x }$ do   
14: Obtain visual feature v via frozen $\theta _ { v i s u a l }$ and textual embedding t via $\theta _ { t e x t u a l }  \phi _ { S } ;$   
15: Update CMC and compute gating factor αˆ by Eq. (9);   
16: Compute $\mathcal { L } _ { \mathrm { r e f i n e } }$ and update $\theta _ { t e x t u a l } , \phi _ { S }$ via gradient descent;   
17: end for

## B DATASET DETAILS

We evaluate our method on a diverse set of benchmark datasets to demonstrate its robustness across different domains. Table 6 summarizes the detailed statistics of these datasets, including the class count and the number of samples in training and test splits. Note that we include datasets with both balanced and imbalanced test sets to conduct a comprehensive evaluation.

Table 6: Statistics for benchmark datasets used in main experiments. The rightmost column indicates whether the testing set is balanced.
<table><tr><td>Dataset</td><td>Classes</td><td>Train</td><td>Test</td><td>Balanced</td></tr><tr><td>CUB (Wah et al., 2011)</td><td>200</td><td>5,994</td><td>5,794</td><td>x</td></tr><tr><td>RESISC45 (Cheng et al., 2017)</td><td>45</td><td>6,300</td><td>25,200</td><td>√</td></tr><tr><td>FGVCAircraft (Maji et al., 2013)</td><td>100</td><td>6,667</td><td>3,333</td><td>√</td></tr><tr><td>Caltech101 (Fei-Fei et al., 2004)</td><td>100</td><td>5,777</td><td>2,465</td><td>x</td></tr><tr><td>EuroSAT (Helber et al., 2019)</td><td>10</td><td>22,000</td><td>5,000</td><td>√</td></tr><tr><td>Flowers102 (Nilsback &amp; Zisserman, 2008)</td><td>102</td><td>2,040</td><td>6,149</td><td>x</td></tr><tr><td>Food101 (Bossard et al., 2014)</td><td>101</td><td>70,700</td><td>30,300</td><td>√</td></tr><tr><td>OxfordPets (Parkhi et al., 2012)</td><td>37</td><td>3,680</td><td>3,669</td><td>x</td></tr><tr><td>StanfordCars (Krause et al., 2013)</td><td>196</td><td>8,144</td><td>8,041</td><td>x</td></tr></table>

## C IMPLEMENTATION DETAILS

We rerun all baseline methods from their official public codebases under a unified protocol. Specifically, we standardize the visual backbone to OpenAI CLIP ViT-B/32 and use the same training dataset, data split, and evaluation metrics for all methods. All other settings follow the recommendations in the corresponding official implementations, such as the training schedule and optimization hyperparameters. For MARS, in Stage 1, we use a batch size of 64 and train the model for 10 epochs with the SGD optimizer, optimizer momentum 0.9, initial learning rate 0.0035, weight decay 5e-4, and a linear-warmup cosine learning-rate schedule. In Stage 2, we train the student model for 30 epochs. We fix $\beta = \bar { 1 }$ and the momentum coefficient $\lambda = 0 . { \bar { 9 5 } }$ for all datasets. In the balanced setting of Table 2, we further set the imbalance ratio to 1 (cap the maximum number of training samples per class at 100). We conduct the experiments on a single NVIDIA RTX 4090 GPU using PyTorch v2.8.0.

## D MORE EXPERIMENT RESULTS AND ANALYSES

## D.1 SENSITIVITY OF HYPERPARAMETERS

In the previous experiments, we fix β = 1 across datasets. We evaluate the sensitivity of this exponent in Fig. 6 on Caltech101 and RESISC45.

![](images/8d5b85533a553551b469cb1899747584d8152c46e9b604c6555c298e6461edd9.jpg)

![](images/e009399b4e0daafc3ef0a643ca64ce2bb18167c51f67a79aafcca875b8dd4d88.jpg)  
Figure 6: Sensitivity to $\beta$ on Caltech101 (left) and RESISC45 (right).

## D.2 STATISTICAL SIGNIFICANCE ANALYSIS

To further validate the statistical significance of the improvements, we conduct two-sided paired t-tests at a 0.05 significance level against seven baseline methods using the three corresponding seed results. As shown in Table 7, our method achieves significant improvements in 84.7% of cases (160 out of 189), confirming its effectiveness under different imbalance ratios.

Table 7: Statistical significance of performance differences assessed with a two-sided paired t-test at a 0.05 significance level over three corresponding seeds, reported as win/tie/loss counts. Imbalance Ratio is abbreviated as IR.
<table><tr><td>Method</td><td>IR = 10</td><td>IR = 20</td><td>IR = 50</td><td>Total</td></tr><tr><td>FPL</td><td>8 /1/0</td><td>9/0/0</td><td>9/0/0</td><td>26 / 1 / 0</td></tr><tr><td>GRIP</td><td>9/0/0</td><td>8 / 1/ 0</td><td>8 / 1 / 0</td><td>25 /2 / 0</td></tr><tr><td>CPL</td><td>8 / 1/ 0</td><td>8/1/0</td><td>8 / 1 / 0</td><td>24 /3 /0</td></tr><tr><td>UEO</td><td>9/0/0</td><td>6/3/0</td><td>7/2/0</td><td>22 / 5 / 0</td></tr><tr><td>TMP</td><td>9/0/0</td><td>9/0/0</td><td>9/0 /0</td><td>27 / 0 / 0</td></tr><tr><td>CAP</td><td>4/5/0</td><td>7/2/0</td><td>7/2/0</td><td>18 / 9 / 0</td></tr><tr><td>microCLIP</td><td>6/3/0</td><td>7/2/0</td><td>5/4/0</td><td>18 /9 / 0</td></tr><tr><td>Total</td><td>53 / 10 / 0</td><td>54/9 /0</td><td>53 / 10 / 0</td><td>160 / 29 / 0</td></tr></table>

## D.3 COMPARISON WITH DIFFERENT VLMS

To verify the generalization capability of our approach, we complement the OpenCLIP results in Table 3 with imbalance ratios of 10 and 50 (Table 8), and further replace CLIP with SigLIP (Table 9). It can be observed that our method maintains a significant performance advantage over previous baselines on both VLMs, demonstrating that our strategy is effective across different VLMs. Notably, with SigLIP, our method surpasses the best-performing baseline by 5.10 pp on RESISC45 under the imbalance ratio of 50.

Table 8: Comparison of test accuracy (%) using OpenCLIP with imbalance ratios of 10 and 50. Zero-shot OpenCLIP is abbreviated as ZS OpenCLIP.
<table><tr><td rowspan="2">Method</td><td colspan="4">Imbalance Ratio = 10</td><td colspan="4">Imbalance Ratio = 50</td><td rowspan="2">Avg.</td></tr><tr><td>Res.</td><td>Flw.</td><td>ES.</td><td>Pets.</td><td>Res.</td><td>Flw.</td><td>ES.</td><td>Pets.</td></tr><tr><td>ZS OpenCLIP</td><td>63.90</td><td>71.43</td><td>51.74</td><td>90.62</td><td>63.90</td><td>71.43</td><td>51.74</td><td>90.62</td><td>69.42</td></tr><tr><td>FPL</td><td>64.05</td><td>66.43</td><td>50.10</td><td>88.03</td><td>59.27</td><td>65.18</td><td>49.98</td><td>86.92</td><td>66.25</td></tr><tr><td>GRIP</td><td>64.63</td><td>64.29</td><td>62.08</td><td>82.47</td><td>57.94</td><td>66.12</td><td>60.20</td><td>81.22</td><td>67.37</td></tr><tr><td>CPL</td><td>66.66</td><td>63.20</td><td>64.18</td><td>86.67</td><td>59.65</td><td>63.70</td><td>60.42</td><td>81.38</td><td>68.23</td></tr><tr><td>UEO</td><td>62.49</td><td>70.16</td><td>53.54</td><td>88.66</td><td>58.77</td><td>70.22</td><td>51.44</td><td>89.02</td><td>68.04</td></tr><tr><td>TMP</td><td>45.98</td><td>16.34</td><td>10.08</td><td>60.70</td><td>34.51</td><td>11.79</td><td>10.00</td><td>35.43</td><td>28.10</td></tr><tr><td>CAP</td><td>67.60</td><td>65.95</td><td>65.36</td><td>88.28</td><td>58.32</td><td>61.67</td><td>50.36</td><td>73.45</td><td>66.37</td></tr><tr><td>microCLIP</td><td>66.44</td><td>67.52</td><td>59.30</td><td>89.62</td><td>64.19</td><td>65.46</td><td>57.52</td><td>88.96</td><td>69.88</td></tr><tr><td>MARS</td><td>72.10</td><td>75.64</td><td>76.74</td><td>92.12</td><td>70.10</td><td>73.52</td><td>72.34</td><td>91.77</td><td>78.04</td></tr></table>

Table 9: Comparison of test accuracy (%) using SigLIP with varying imbalance ratios (10, 20, 50). Zero-shot SigLIP is abbreviated as ZS SigLIP.
<table><tr><td rowspan="2">Method</td><td colspan="4">Imbalance Ratio = 10</td><td colspan="4">Imbalance Ratio = 20</td><td colspan="4">Imbalance Ratio = 50</td><td rowspan="2">Avg.</td></tr><tr><td>Res.</td><td>Flw.</td><td>ES.</td><td>Pets.</td><td>Res.</td><td>Flw.</td><td>ES.</td><td>Pets.</td><td>Res.</td><td>Flw.</td><td>ES.</td><td>Pets.</td></tr><tr><td>ZS SigLIP</td><td>59.23</td><td>83.46</td><td>35.74</td><td>93.10</td><td>59.23</td><td>83.46</td><td>35.74</td><td>93.10</td><td>59.23</td><td>83.46</td><td>35.74</td><td>93.10</td><td>67.88</td></tr><tr><td>FPL</td><td>61.14</td><td>82.96</td><td>37.12</td><td>92.80</td><td>60.04</td><td>83.36</td><td>35.14</td><td>92.45</td><td>60.23</td><td>83.48</td><td>35.62</td><td>92.83</td><td>68.10</td></tr><tr><td>GRIP</td><td>62.36</td><td>83.72</td><td>45.38</td><td>91.52</td><td>61.73</td><td>83.66</td><td>42.62</td><td>91.71</td><td>60.62</td><td>83.72</td><td>34.72</td><td>91.11</td><td>69.41</td></tr><tr><td>CPL</td><td>65.56</td><td>81.33</td><td>46.74</td><td>90.92</td><td>62.76</td><td>81.22</td><td>52.14</td><td>88.91</td><td>60.50</td><td>82.50</td><td>44.24</td><td>88.72</td><td>70.46</td></tr><tr><td>UEO</td><td>59.65</td><td>83.48</td><td>40.10</td><td>93.08</td><td>59.55</td><td>83.46</td><td>39.94</td><td>93.10</td><td>59.40</td><td>83.46</td><td>39.74</td><td>93.08</td><td>69.00</td></tr><tr><td>TMP</td><td>19.25</td><td>7.82</td><td>16.98</td><td>28.75</td><td>17.05</td><td>6.05</td><td>15.36</td><td>20.58</td><td>14.27</td><td>3.85</td><td>16.84</td><td>13.44</td><td>15.02</td></tr><tr><td>CAP</td><td>72.62</td><td>84.48</td><td>50.18</td><td>86.83</td><td>69.36</td><td>85.15</td><td>46.86</td><td>84.73</td><td>63.03</td><td>83.59</td><td>47.64</td><td>82.03</td><td>71.38</td></tr><tr><td>microCLIP</td><td>69.64</td><td>85.41</td><td>56.02</td><td>93.49</td><td>68.00</td><td>84.45</td><td>48.94</td><td>92.61</td><td>65.59</td><td>82.45</td><td>46.62</td><td>92.31</td><td>73.79</td></tr><tr><td>MARS</td><td>75.15</td><td>89.98</td><td>68.36</td><td>94.99</td><td>74.03</td><td>88.96</td><td>71.92</td><td>95.28</td><td>70.69</td><td>89.15</td><td>61.46</td><td>94.88</td><td>81.24</td></tr></table>

## D.4 COMPARISON WITH DIFFERENT VISUAL BACKBONES

We further investigate the scalability of our model by upgrading the visual backbone to a highercapacity ViT-B/16 encoder. The results presented in Table 10 indicate that MARS maintains a performance advantage over existing baselines under all imbalance ratios. Specifically, our method achieves the highest average accuracy of 74.41%, outperforming the strongest baseline by 3.82 pp on average, with consistent gains across all four datasets. This validates that our approach is not only robust across different VLMs but also capable of effectively leveraging the enhanced features provided by the stronger vision backbone.

Table 10: Comparison of test accuracy (%) using ViT-B/16 as the visual backbone with imbalance ratios of 10, 20, and 50.
<table><tr><td rowspan="2">Method</td><td colspan="4">Imbalance Ratio = 10</td><td colspan="4">Imbalance Ratio = 20</td><td colspan="4">Imbalance Ratio = 50</td><td rowspan="2">Avg.</td></tr><tr><td>Res.</td><td>Cub.</td><td>Flw.</td><td>Pets.</td><td>Res.</td><td>Cub.</td><td>Flw.</td><td>Pets.</td><td>Res.</td><td>Cub.</td><td>Flw.</td><td>Pets.</td></tr><tr><td>ZS CLIP</td><td>62.25</td><td>55.32</td><td>71.15</td><td>88.20</td><td>62.25</td><td>55.32</td><td>71.15</td><td>88.20</td><td>62.25</td><td>55.32</td><td>71.15</td><td>88.20</td><td>69.23</td></tr><tr><td>FPL</td><td>65.02</td><td>56.81</td><td>66.74</td><td>89.40</td><td>63.02</td><td>55.50</td><td>65.08</td><td>89.29</td><td>61.38</td><td>55.35</td><td>65.88</td><td>88.85</td><td>68.53</td></tr><tr><td>GRIP</td><td>65.47</td><td>54.17</td><td>69.18</td><td>86.26</td><td>61.22</td><td>53.86</td><td>69.52</td><td>85.88</td><td>58.41</td><td>52.12</td><td>67.67</td><td>84.08</td><td>67.32</td></tr><tr><td>CPL</td><td>66.72</td><td>54.98</td><td>67.23</td><td>88.80</td><td>62.83</td><td>53.27</td><td>66.08</td><td>86.24</td><td>58.16</td><td>53.33</td><td>67.18</td><td>84.06</td><td>67.41</td></tr><tr><td>UEO</td><td>61.62</td><td>57.18</td><td>70.92</td><td>89.62</td><td>60.75</td><td>56.59</td><td>70.73</td><td>89.59</td><td>60.15</td><td>56.61</td><td>70.60</td><td>89.45</td><td>69.48</td></tr><tr><td>TMP</td><td>46.50</td><td>20.04</td><td>13.21</td><td>43.66</td><td>38.52</td><td>16.07</td><td>9.73</td><td>29.93</td><td>31.21</td><td>12.72</td><td>7.25</td><td>18.86</td><td>23.98</td></tr><tr><td>CAP</td><td>69.87</td><td>56.29</td><td>69.34</td><td>89.18</td><td>65.60</td><td>56.61</td><td>69.18</td><td>84.71</td><td>63.89</td><td>54.88</td><td>67.83</td><td>83.37</td><td>69.23</td></tr><tr><td>microCLIP</td><td>69.01</td><td>53.78</td><td>72.48</td><td>90.54</td><td>67.75</td><td>51.66</td><td>71.10</td><td>90.30</td><td>66.87</td><td>51.19</td><td>71.56</td><td>90.84</td><td>70.59</td></tr><tr><td>MARS</td><td>72.32</td><td>59.84</td><td>74.87</td><td>93.65</td><td>70.51</td><td>58.56</td><td>75.77</td><td>93.24</td><td>69.26</td><td>58.28</td><td>74.14</td><td>92.45</td><td>74.41</td></tr></table>

## D.5 FINE-GRAINED RESULTS ACROSS HEAD, MEDIUM, AND TAIL GROUPS

To complement the group-level analysis, we report detailed head, medium, and tail results on RESISC45 and EuroSAT in Table 11. The table shows that our method improves not only overall

accuracy but also strongly restores head-class performance while maintaining medium- and tail-class performance. For example, on RESISC45, the head accuracy improves by 18.65 pp over UEO and 32.57 pp over CAP.

Table 11: Comparison of test accuracy (%) across head, medium, and tail groups on RESISC45 and EuroSAT with an imbalance ratio of 50.
<table><tr><td rowspan="2">Method</td><td colspan="4"> $\mathrm { R e s . }$ </td><td colspan="4">ES.</td></tr><tr><td>Overall</td><td>Head</td><td>Medium</td><td>Tail</td><td>Overall</td><td>Head</td><td>Medium</td><td>Tail</td></tr><tr><td>FPL</td><td>57.61</td><td>46.59</td><td>58.52</td><td>59.72</td><td>55.51</td><td>23.40</td><td>60.20</td><td>58.52</td></tr><tr><td>CPL</td><td>56.73</td><td>38.69</td><td>57.40</td><td>60.57</td><td>58.82</td><td>13.00</td><td>65.53</td><td>63.10</td></tr><tr><td>UEO</td><td>56.97</td><td>51.90</td><td>54.43</td><td>59.83</td><td>42.42</td><td>6.80</td><td>44.67</td><td>47.23</td></tr><tr><td>CAP</td><td>58.78</td><td>37.98</td><td>59.88</td><td>63.05</td><td>50.38</td><td>26.40</td><td>60.47</td><td>49.33</td></tr><tr><td>MARS</td><td>68.70</td><td>70.55</td><td>67.01</td><td>69.11</td><td>75.68</td><td>78.60</td><td>76.60</td><td>74.73</td></tr></table>

## D.6 HEAD-TO-TAIL MISASSIGNMENT ANALYSIS

To examine the head-class degradation in Section 3.2, we track how true head-class test samples are predicted after adaptation. H-to-H and H-to-T denote the numbers of these samples misclassified as head and tail classes, respectively. As shown in Table 12, CPL reduces the head accuracy on RESISC45 from 69.20% to 38.69% and increases the H-to-T count from 428 to 1,060, while H-to-H decreases from 429 to 164. BPA restores the head accuracy to 72.89% and reduces the H-to-T count to 392 without sacrificing the tail accuracy. OxfordPets shows the same pattern.

Table 12: Head/tail accuracy (%) and prediction flow of true head-class samples on RESISC45 and OxfordPets with an imbalance ratio of 50.
<table><tr><td>Dataset</td><td>Method</td><td>Head  $\operatorname { A c c } .$ </td><td>Tail Acc.</td><td>H-to-H</td><td>H-to-T</td></tr><tr><td rowspan="3">RESISC45</td><td>ZS CLIP</td><td>69.20</td><td>56.84</td><td>429</td><td>428</td></tr><tr><td>CPL</td><td>38.69</td><td>60.57</td><td>164</td><td>1,060</td></tr><tr><td>BPA</td><td>72.89</td><td>65.37</td><td>210</td><td>392</td></tr><tr><td rowspan="3">OxfordPets</td><td>ZS CLIP</td><td>86.75</td><td>91.47</td><td>28</td><td>18</td></tr><tr><td>CPL</td><td>80.12</td><td>92.14</td><td>17</td><td>56</td></tr><tr><td>BPA</td><td>88.14</td><td>92.24</td><td>30</td><td>16</td></tr></table>

We further analyze all true head-class samples that CPL predicts as tail classes on Caltech101 and Food101 with an imbalance ratio of 50. For each sample x with ground-truth label y, we compute the logit margin change relative to zero-shot CLIP, $\Delta m = m _ { \mathrm { A d a p t e d } } ( x , y ) - m _ { \mathrm { Z e r o - s h o t C L I P } } ( x , y )$ , where $m ( x , y ) = z _ { y } ( x ) - \mathrm { m a x } _ { j \neq y } z _ { j } ( x )$ . As shown in Table 13, CPL yields strongly negative margin changes on these samples, whereas BPA turns them positive. Moreover, BPA no longer predicts most of these samples as tail classes and recovers the correct label for 94.12% of them on Caltech101 and 61.11% on Food101.

Table 13: Average logit margin changes relative to zero-shot CLIP and repairs over all true head-class samples predicted as tail classes by CPL on Caltech101 and Food101 with an imbalance ratio of 50. “No longer tail” counts the samples that BPA no longer predicts as tail classes, and “Exact repairs” counts those that BPA predicts correctly.
<table><tr><td>Dataset</td><td>Cases</td><td>CPL ∆m</td><td>BPA ∆m</td><td>No longer tail</td><td>Exact repairs</td></tr><tr><td>Caltech101</td><td>204</td><td>-3.88</td><td>+2.28</td><td>199 (97.55%)</td><td>192 (94.12%)</td></tr><tr><td>Food101</td><td>1,579</td><td>-4.41</td><td>+1.04</td><td>1,204 (76.25%)</td><td>965 (61.11%)</td></tr></table>

## D.7 BASE-TO-NEW GENERALIZATION

Beyond standard closed-set classification, we further evaluate base-to-new generalization. Following the standard 1:1 base/new split, we train only on unlabeled long-tailed base-class samples and evaluate on the base and new test sets using the harmonic mean. As shown in Table 14, our method consistently

outperforms previous methods on both datasets. Specifically, on RESISC45, MARS improves the best prior harmonic mean from 58.92% to 62.16%, yielding a gain of 3.24 pp. These results validate that our method preserves strong cross-class generalization under long-tailed adaptation.

Table 14: Base-to-new generalization on RESISC45 and OxfordPets (imbalance ratio=50).
<table><tr><td rowspan="2">Method</td><td colspan="3"> ${ \mathrm { R e s . } }$ </td><td colspan="3">Pets.</td></tr><tr><td>Base</td><td>New</td><td>H</td><td>Base</td><td>New</td><td>H</td></tr><tr><td>ZS CLIP</td><td>55.36</td><td>53.56</td><td>54.45</td><td>84.77</td><td>83.95</td><td>84.36</td></tr><tr><td>FPL GRIP</td><td>69.05 64.15</td><td>51.39 42.65</td><td>58.92 51.23</td><td>83.68 77.67</td><td>86.63 85.23</td><td>85.13 81.28</td></tr><tr><td>CPL</td><td>66.33</td><td>45.27</td><td>53.81</td><td>82.46</td><td>85.68</td><td>84.04</td></tr><tr><td>UEO</td><td>60.02</td><td>49.66</td><td>54.35</td><td>85.06</td><td>88.59</td><td>86.79</td></tr><tr><td>CAP</td><td>66.90</td><td>44.71</td><td>53.60</td><td>82.24</td><td>78.36</td><td>80.25</td></tr><tr><td>microCLIP</td><td>67.76</td><td>44.71</td><td>53.87</td><td>80.75</td><td>88.93</td><td>84.64</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MARS</td><td>69.59</td><td>56.16</td><td>62.16</td><td>89.10</td><td>89.49</td><td>89.29</td></tr></table>