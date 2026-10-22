# Rethinking Post-hoc Debiasing of Large Pretrained Models

Divyaansh Mertia<sup>∗</sup> Chirag Bhutra<sup>∗</sup> Himanshu Shekhar<sup>∗</sup> Shubham Bagwari<sup>∗</sup>

Pratik Mazumder<sup>∗</sup>

Indian Institute of Technology Jodhpur

\* {p25cs0004, b23mt1012, b23mt1021, p22cs201, pratikm}@iitj.ac.in

## Abstract

Large pretrained models such as CLIP are widely used as frozen feature extractors. However, their embeddings inherit spurious correlations from training data, introducing bias into the features. A growing body of research has sought to address this issue through post hoc debiasing, leaving the backbone untouched. We discover that despite differing mechanisms, these methods require the bias feature to be identified and expressed in text. This dependency is not innocuous. In this work, we experimentally demonstrate that this requirement incurs an unforeseen penalty in certain scenarios. For example, on the UrbanCars benchmark, having two co-occurring bias attributes, the recent methods underperform a vanilla classifier on biasedfrozenbackbone features, and even providing ground-truth bias identities does not close the gap. We show that the pattern that such methods seek is readily observable in the frozen geometry. Specifically, cross-class training pairs placed closest together by the frozen encoder almost always share a spurious attribute, regardless ofthe bias type, without requiring access to the bias labels. Based on this, we propose a novel approach to identify such pairs and learn a single linear map that transforms the features to perform post-hoc debiasing. The proposed approach achieves competitive performance across all datasets without access to bias attribute annotations or any LLMs.

## 1. Introduction

Large pretrained models such as CLIP [12] are widely used as frozen feature extractors. While classifiers built on the embeddings of such models reach high average accuracy on multiple tasks, they often fail on a predictable subset of inputs. The root cause of these failures stems from the data distributions at both the pretraining and downstream stages. Since foundational models are trained on massive, uncurated web corpora, their learned feature spaces inherently capture and encode various naturally occurring ‘spurious correlations’ [9, 15–18, 20]. Consequently, when a spurious attribute strongly co-occurs with the class label, classifiers exploit this shortcut, making the model biased toward that attribute, e.g., with a majority of white dog and black cat images, the classifier exploits this color shortcut over complex class-defining features. Such classifiers will fail at test time, if this spurious correlation breaks.

Post-hoc debiasing methods address this issue by adapting the learned representation while keeping the pretrained backbone frozen [1, 2, 10]. Despite differing implementations, they all require the spurious attribute to be identified beforehand. Earlier methods relied on practitioners to specify the spurious attribute and construct descriptive text prompts [2], while recent works use LLMs to propose candidate bias attributes based solely on class names and generate controlled scene descriptions. These are used to suppress the corresponding text-aligned direction in the embedding space [1, 10]. While this shift is typically framed as a transition toward unsupervised debiasing, it is more appropriately just a change of oracle. This fundamental dependency on external attribute specification makes these methods incapable of detecting or correcting a misidentified bias.

This reliance on text-based attribute identification introduces a critical failure mode when datasets contain complex or co-occurring shortcuts. While LLM-driven methods succeed when a single spurious attribute aligns cleanly with broad world knowledge, they struggle with multiple or non-intuitive attributes. On the UrbanCars benchmark [7], which features two simultaneous spurious attributes (background and co-occurring object), methods that obtain their attribute from a language model perform worse than an undebiased linear classifier trained on the frozen embeddings. PRISM [10] reaches only 3.23% worst-group accuracy with a trained linear head, compared to 38.71% for the undebiased baseline.

One might hypothesize that this failure stems merely from imprecise LLM-generated descriptions. To test this hypothesis, we perform an oracle experiment by supplying ground-truth description of the spurious attributes to these methods (Sec. 4.2). The accuracy improves but remains below the undebiased baseline. This demonstrates that external bias identification is necessary for such methods but insufficient: suppressing multiple text-guided subspaces can remove essential class-discriminative semantics, damaging the representation more than the shortcuts it removes. Thus, prompt refinement cannot resolve this fundamental tradeoff.

To overcome these limitations, we propose an alternative paradigm that operates directly on an intrinsic geometric signature of bias rather than on bias attribute specifications. We define a sample as bias-aligned if its spurious attribute agrees with the dominant spurious-label correlation in the training data, and bias-conflicting if it violates it. Consequently, a cross-class pair sharing the same spurious attribute necessarily consists of one bias-aligned and one bias-conflicting sample, a configuration we term a biassharing pair. We observe that in a biased frozen feature space, cross-class pairs exhibiting high cosine similarity are bias-sharing. Spurious correlation thus manifests geometrically as a localized blur along identifiable cross-class sample pairs. Detecting these pairs requires no text prompts or attribute supervision, only class labels and pairwise similarity. By pulling apart these confusable cross-class pairs while maintaining the intra-class cohesion, we learn a single metric transformation $\mathbf { W } \in \mathbb { R } ^ { d \times d }$ over the frozen embeddings. Because this geometric criterion relies strictly on feature similarity, it isolates and separates bias-sharing pairs agnostic to the number or nature of underlying shortcuts.

Our proposed framework requires no bias attribute annotations, language models, auxiliary biased models, or backpropagation through the backbone. Across extensive evaluations, our approach achieves 81.18% worst-joint accuracy on UrbanCars (compared to 61.29% for the best competing method) while simultaneously improving overall accuracy. On BAR, our method delivers state-of-the-art worst-group accuracy across all bias-conflicting ratios. Finally, on BFFHQ, our performance scales dynamically with the availability of conflicting samples, outperforming static prompt-based baselines as conflicting data increases. The contributions of this work are as follows:

• We empirically demonstrate that post-hoc debiasing methods that obtain the spurious attribute from a language model fail on multi-shortcut datasets, performing worse than an undebiased linear classifier, and that supplying ground-truth text prompts does not close the gap.

• We identify an intrinsic geometric signature of spurious correlation in frozen embedding spaces where crossclass feature pairs with high similarity are overwhelmingly bias-sharing and show it can be detected purely from pairwise cosine similarity.

• We introduce an annotation-free post-hoc debiasing approach that does not rely on external attribute descrip-

tions or language models.

• We evaluate the proposed approach across diverse multimodal and vision-only backbones (including CLIP[12], SigLIP[19], and DINOv3[14]), achieving state-of-theart worst-group performance on the multi-shortcut benchmark and competitive performance across singleshortcut benchmarks.

## 2. Related Work

There is a growing body of work that involves debiasing deep learning models. In this section, we discuss existing works that perform bias mitigation under various settings.

Robust training without group labels. When group annotations are available, distributionally robust optimization [13] and last-layer retraining on a group-balanced set [6] optimize worst-group accuracy directly. A larger body of work removes that requirement by inferring which samples are atypical: from the errors of a deliberately biased model [8, 11] then reweighting against them. These methods are effective, but they assume that the encoder is trainable, and their inference of atypical samples rests on a representation that changes as training proceeds.

Debiasing frozen vision–language embeddings. Various recent works debias pretrained vision-language models (VLMs) without updating their backbones, instead modifying the learned embedding space post-hoc. These methods are often described as annotation-free because they do not require per-sample annotations of the spurious attribute. However, this does not necessarily remove the need to identify the attribute itself. In practice, most existing methods need to be told which attribute is spurious, either by the practitioner or by a large language model.

Bias attribute named by the practitioner. FairerCLIP [2] learns functions over both encoders in a reproducing kernel Hilbert space, minimizing dependence between the representation and a specified attribute while preserving dependence on the target. Although it does not require attribute annotations per image, the bias attribute must be explicitly named, and prompts describing its values are used.

Bias attribute named by a language model. RoboShot [1] shifts the burden of specifying the bias attribute to an LLM, which proposes features that are helpful or harmful for a target class, which are then converted into embedding directions and used for projection. PRISM [10] similarly uses an LLM to propose candidate attributes, then generates controlled scene descriptions in which the class and attribute vary independently and learns a projection that is invariant to the resulting attributes. Its variant, PRISM-mini, replaces the learned projection with an orthogonal projection. These methods avoid manually specifying the bias attribute, but rely on an external model to identify candidate attributes.

<table><tr><td>Method</td><td>Venue</td><td>Bias Info</td><td>Text-Agnostic</td></tr><tr><td></td><td></td><td>Post-Hoc Debiasing (Frozen Backbone)</td><td></td></tr><tr><td>FairerCLIP [2]</td><td>ICLR’24</td><td>named</td><td>x</td></tr><tr><td>RoboShot [1]</td><td>ICLR&#x27;24</td><td>LLM</td><td>x</td></tr><tr><td>PRISM [10]</td><td>ICCV’25</td><td>LLM</td><td>x</td></tr><tr><td>PRISM-mini [10]</td><td>ICCV’25</td><td>LLM</td><td>x</td></tr><tr><td>SEMi [3]</td><td>CVPR&#x27;26</td><td>none</td><td>x</td></tr><tr><td>Ours</td><td>1</td><td>none</td><td>√</td></tr></table>

Table 1. Requirements across debiasing methods. Bias Info denotes the spurious attribute supervision required (per-sample labels, attributes named by the practitioner, LLM proposals, or none). Text-Agnostic indicates methods that do not depend on a text encoder or prompt generation.

Bias attribute is not specified. Rather than identifying a spurious attribute, SEM<sub>i</sub> [3] searches for representation components that are broadly responsive rather than selectively associated with a target class. It encodes frozen embeddings with a sparse encoder and compares neuron responses on two prompt sets: 328 neutral captions spanning diverse scenes, objects, and activities, which provide a reference distribution, and ten LLM-generated paraphrases of each class name. It suppresses neurons with low query relevance, which are treated as likely spurious.

Multiple shortcuts. Li et al. [7] observe that mitigating one shortcut can amplify reliance on another, a failure they term whac-a-mole, and introduce UrbanCars to expose it: two shortcuts are inserted at once, one of which is an artifact of dataset construction rather than a regularity of the visual world. The frozen-backbone methods above are evaluated almost exclusively on benchmarks carrying a single shortcut. We experimentally demonstrate in Sec. 4 that whac-a-mole persists on frozen embeddings, and is not resolved just by ground truth bias attribute specifications.

In this work, we present a post-hoc debiasing approach that performs post-hoc debiasing of the features of a frozen large pretrained model, without requiring any practitioner or language model or even the bias specifications. We also experimentally demonstrate that our proposed approach performs significantly better than existing post-hoc debiasing methods including on the UrbanCars benchmark.

## 3. Problem Setting

Let each example in the dataset be a tuple $( x , y , a )$ , where $x \in \mathcal { X }$ is an image, $y \in \mathcal { D }$ is its class label, and $a \in { \mathcal { A } }$ is a latent bias attribute that is not causally related to y. Under a biased training distribution $P _ { \mathrm { t r a i n } } ( x , y , a )$ , a and y are spuriously correlated. A sample x is bias-aligned when the pair $( y , a )$ follows that spurious correlation and bias-conflicting when it violates it. Let bias-ratio $\rho$ refer to the proportion of bias-conflicting samples in the dataset. A low $\rho$ denotes more biased dataset. A dataset may carry multiple bias attributes at once. We consider a setting where each class has at least one spurious attribute value that appears in the majority of its samples. The same attribute value may appear in other classes, but only in the minority of their samples. Multiple such attributes may exist for a single class. In this setting, two samples of different classes that take the same value of a are necessarily one aligned and one conflicting, since the majority value for one class can only be a minority value for the other. We call such a pair bias-sharing. Bias-sharing pairs are the ones a classifier is most likely to confuse. They agree on the bias attribute that the training correlation rewards and disagree on the class label. We seek a classifier that performs well under a target distribution $P _ { \mathrm { t e s t } }$ in which the spurious correlation is absent, and in particular perform well on the conflicting groups, where reliance on the bias attribute a is penalized rather than being rewarded. The performance is therefore measured by worst-group accuracy alongside the overall accuracy. In this work, we limit the problem setting to classifiers built on a frozen large pretrained vision encoder $f _ { I }$ producing normalized embeddings ${ \bf x } = f _ { I } ( x )$ . The encoder is not fine-tuned or trained on the biased dataset. The objective is to learn a map applied to those embeddings, together with a linear head. We operate in the bias-unaware setting, where the bias attribute a is neither observed during training nor identified in advance.

![](images/6425cd1b60ea5e93ed1610cf1a7c756c2d705d227326e5baf48cd5b83e5e6c99.jpg)  
Figure 1. On UrbanCars, most debiasing methods are worse than no debiasing. Worst-group accuracy of each method, measured against a linear classifier trained on the same frozen CLIP features with no debiasing. Bars to the left of the reference mark methods that reduce accuracy below it. The dashed bar denotes PRISM with the ground-truth attribute identities (Sec. 4.2).

## 4. Motivation

Post-hoc debiasing methods that rectify a frozen embedding space require the spurious attribute to be specified before they can act. Multiple such methods obtain this specification from a language model, which proposes candidates from class names alone. This section examines the impact of that requirement on a more realistic UrbanCars benchmark, where two spurious attributes are present at once.

<table><tr><td>Method</td><td>Bias info Ovr.</td><td>W-BG</td><td>W-Co</td><td>W-Joint</td></tr><tr><td colspan="5">Cosine similarity to prompt embeddings</td></tr><tr><td>CLIP-cosine</td><td>none 50.00</td><td>4.03</td><td>13.71</td><td>0.00</td></tr><tr><td>PRISM</td><td>LLM 53.43</td><td>6.45</td><td>16.13</td><td>1.61</td></tr><tr><td>+ oracle</td><td>oracle 59.48</td><td>33.06</td><td>47.58</td><td>30.65</td></tr><tr><td>PRISM-mini</td><td>LLM 53.63</td><td>24.19</td><td>27.42</td><td>6.45</td></tr><tr><td>+ oracle</td><td>oracle 55.65</td><td>26.61</td><td>20.97</td><td>9.68</td></tr><tr><td colspan="5">Linear head on the corrected features</td></tr><tr><td>CLIP-linear none</td><td>81.18</td><td>68.55</td><td>65.32</td><td>38.71</td></tr><tr><td>PRISM</td><td>LLM 61.49</td><td>32.30</td><td>33.10</td><td>3.23</td></tr><tr><td>+ oracle</td><td>oracle 67.94</td><td>49.20</td><td>45.20</td><td>17.74</td></tr><tr><td>PRISM-mini LLM</td><td>73.79</td><td>52.40</td><td>51.60</td><td>11.29</td></tr><tr><td>+ oracle oracle</td><td>76.61</td><td>56.45</td><td>57.26</td><td>20.97</td></tr></table>

Table 2. Correct specification is necessary but not sufficient. Replacing LLM-generated attribute descriptions with ground-truth prompts improves PRISM and PRISM-mini, but both remain below the undebiased reference.
<table><tr><td>Cross-class pair shares...</td><td>all pairs</td><td>selected</td></tr><tr><td>neither shortcut</td><td>81.9</td><td>5.2</td></tr><tr><td>background only</td><td>8.6</td><td>35.2</td></tr><tr><td>co-occurring object only</td><td>8.6</td><td>12.1</td></tr><tr><td>both shortcuts</td><td>0.9</td><td>47.6</td></tr><tr><td>at least one shortcut</td><td>18.1</td><td>94.8</td></tr><tr><td>number of pairs</td><td>16.0M</td><td>986</td></tr></table>

Table 3. What a similarity threshold selects. Composition of cross-class training pairs on UrbanCars (%) in the frozen embedding space, that share each shortcut. Selected denotes pairs whose cosine similarity exceeds the threshold used by our method (Sec. 5). Attribute labels are used only for this analysis.

## 4.1. Debiasing Performs Below the Baseline

One might expect rectifying a biased representation to always help. Yet, we observe something strange on the UrbanCars benchmark (Fig. 1). A linear classifier trained on features from a frozen pretrained model, with no debiasing, reaches 81.18% overall and 38.71% worst group accuracy. However, when we debias the features using posthoc debiasing methods that use LLMs for bias-attribute specifications and train a same-capacity linear classifier on the debiased features, these methods perform very poorly.

![](images/c6b819f26aa3e6b7175f2894c595c1bfd09948a8767c44baf3f2f8e8d2d4a3b1.jpg)  
Figure 2. Bias concentration scales with feature similarity. The percentage of cross-class training pairs sharing a spurious back ground or co-occurring object increases monotonically as a function of the cosine similarity threshold.

PRISM [10] reaches 3.23% worst-group accuracy, RoboShot [1] reaches 8.06%, and PRISM-mini [10] reaches 11.29%. Even the best method falls 27 points below the naive classifier with no debiasing. Overall accuracy also declines, ruling out a robustness–accuracy trade-off.

## 4.2. Expert Bias Specification Is Not Enough

Two competing explanations could account for the above failure. Firstly, LLM generated bias-specifications may not match the actual shortcuts. Secondly, the bias removal mechanism may be at fault. UrbanCars provides a direct test because it contains two co-occurring bias-shortcuts. We test the first explanation by replacing the LLM proposals with expert-written prompts that name the two ground-truth attributes, background and co-occurring object. Keeping the rest of the method same, we report the new results (+oracle) in Tab. 2. Using a cosine classifier, PRISM improves from 1.61% to 30.65% on worst joint accuracy with the oracle information. With a trained linear head, the improvement is smaller, from 3.23% to 17.74%. The performance never reaches the 38.71% of the undebiased linear classifier. Since the ground truth specification helps in both cases, the specifications were indeed part of the problem. But the ground truth specification is insufficient to beat the baseline. A likely reason is that the two bias shortcuts do not occupy orthogonal subspaces in the frozen embedding space. If we represent each spurious attribute as the text derived direction of its prompt, excluding the class component they share by construction, the cosine similarity between these two directions is 0.58. PRISM removes the direction associated with a specified attribute. When two directions overlap this much, removing one leaves part of the other behind, and the classifier shifts its reliance to that residual. Fixing one bias makes the other worse. This is the whac-a-mole failure described in [7], and UrbanCars was built to expose this.

![](images/f51108ddac3f7a054ff342581b99fb2ffad0f36d93eda27c27dbb3299bbff488.jpg)  
Figure 3. Method overview – Left: we select two sets of pairs in the frozen space using class labels and cosine similarity alone. P takes cross-class pairs above $\tau _ { \mathrm { p u s h } }$ , 94.8% of which are bias-sharing, and Q takes same-class pairs below $\tau _ { \mathrm { p u l l } }$ . Right: a selected pair contributes gradients until it reaches its margin and none after. Thresholds decide which pairs move, margins how far. We only train W.

UrbanCars also exposes why the bias specification failed. Since an LLM proposes bias attributes for the given classes based on real-world co-occurrence priors, natural associations such as a vehicle type appearing against a specific background, get easily identified. However, the second shortcut in UrbanCars was introduced via image composition, corresponding to no predominant natural world observation. LLMs have nothing to draw upon in this regard, leaving them blind to artifacts that are not well-explored.

## 4.3. Intrinsic Trace of Bias in the Feature Space Geometry

Prompt-based methods require bias specification because they treat the spurious attribute as an external semantic concept that must be described before it can be eliminated. However, the bias attribute also leaves an intrinsic geometric signature in the representation that can be precisely measured without requiring a semantic description

Spurious features perturb the embedding space almost as much as class boundaries. In the frozen embedding space, two images in UrbanCars of the same class on the same background have a mean cosine similarity of 0.630. We analyze the effect of changing one attribute at a time. Changing only the background reduces this similarity to 0.589. Changing only the class while keeping the background same reduces it to 0.584. Therefore, a change in spurious attribute reduces cosine similarity by almost the same amount as changing the class. Therefore, a crossclass bias-sharing pair of samples that have the same background, will be placed almost as close to each other as two same class samples, by the frozen encoder. We study this finding to discover a geometric signature of the bias.

High-similarity cross-class pairs are overwhelmingly bias-sharing. The decision boundary depends on the pairs of samples the encoder places closest together, not on the average pair. We refer to the two spurious attributes in UrbanCars, background and co-occurring object, as shortcuts. A cross-class pair drawn at random shares neither shortcut in 81.9% of cases, shares one in 17.2% cases, and both in only 0.9% cases. When we select only the pairs with high cosine similarity, this distribution reverses (Tab. 3). Pairs sharing neither shortcut fall to 5.2%, while pairs sharing both rise to 47.6%, and pairs sharing atleast one shortcut rises to 94.8%, becoming the dominant configuration. Therefore, the pairs the encoder finds most confusable due to high similarity are almost always biassharing. The reversal strengthens as the similarity threshold increases (Fig. 2). The proportion of selected pairs that share the background or the co-occurring object rises continuously with the cutoff. This is an intrinsic property of the decision boundary, not an artifact of a chosen threshold. Selection therefore requires no prior knowledge of which attribute makes a pair similar or of how many attributes are involved. It is a measurable property of the representation, and we can use this to learn a transformation that separates bias-sharing pairs to debias the features.

## 5. Proposed Approach

The analysis in Section 4.3 revealed an intrinsic geometric property: the cross-class pairs that a frozen encoder finds most confusable (high cosine similarity) are almost all biassharing. We turn this observation into an objective for posthoc debiasing. The proposed approach learns a linear map $\textbf { W } \in \mathbb { R } ^ { d \times d }$ which transforms the features of the frozen pretrained encoder to debias them (Fig. 3).

## 5.1. Bias-sharing pair selection

Let $\mathbf { x } _ { i }$ denote the frozen embedding of image i produced by the pretrained encoder, and let $y _ { i }$ denote its class label. We consider pairs of training images $( i , j )$ and measure their cosine similarity as $\left. \mathbf { x } _ { i } , \mathbf { x } _ { j } \right.$ . Motivated by our findings in Sec. 4.3, we present the following condition for selecting bias-sharing pairs from the cross-class pairs.

$$
\mathcal { P } = \left\{ ( i , j ) : y _ { i } \neq y _ { j } , ~ \langle \mathbf { x } _ { i } , \mathbf { x } _ { j } \rangle > \tau _ { \mathrm { p u s h } } \right\} .\tag{1}
$$

As characterized in Tab. 3, applying this threshold on the UrbanCars dataset yields a set $\mathcal { P }$ where 94.8% of the members are bias-sharing, compared to just 18.1% in the unconstrained cross-class distribution. Because every pair in $\mathcal { P }$ spans two distinct classes, an objective built solely on this set would leave the geometry within each individual class unconstrained. To anchor the intra-class representation, we sample a secondary set of distant same-class pairs:

$$
\begin{array} { r } { \mathcal { Q } \subseteq \big \{ ( i , j ) : y _ { i } = y _ { j } , \langle \mathbf { x } _ { i } , \mathbf { x } _ { j } \rangle < \tau _ { \mathrm { p u l l } } \big \} , } \end{array}\tag{2}
$$

taking one partner per image and retaining $| \mathcal { P } | / N$ pairs per class (N refers to the number of classes), so that the two sets contribute comparable numbers of pairs for balanced regularization. Since bias-conflicting samples constitute a severe minority of the data, the overwhelming majority of any given class is bias-aligned. Consequently, the low-similarity regime of $\mathcal { Q }$ predominantly captures natural intra-class variance among these abundant aligned samples. However, this set also captures the rare/minor instances where the spurious attribute itself differs (pairing an aligned sample with a conflicting one). Pulling these pairs together is highly beneficial, as it encourages invariance to the spurious attribute while preserving overall class cohesion. Therefore, while $\mathcal { P }$ can be processed to actively separate the confusable cross-class pairs, $\mathcal { Q }$ acts as a broad structural anchor that prevents the internal geometry of the classes from collapsing. We discuss the contribution of each set in (Sec. 6.3). Both sets are computed once, from the frozen embeddings, before W exists. Membership is fixed by the geometry measured in Sec. 4 rather than by the representation the method is producing.

## 5.2. Managing bias-sharing pairs and low similarity same class pairs

Let $\mathbf { z } _ { i } = \mathbf { W } \mathbf { x } _ { i } / \| \mathbf { W } \mathbf { x } _ { i } \|$ denote the projected embedding and $s _ { i j } = \langle \mathbf { z } _ { i } , \mathbf { z } _ { j } \rangle$ the similarity it induces. The sets $\mathcal { P }$ and $\mathcal { Q }$ are used to optimize W with a one-sided hinge against a margin:

$$
\mathcal { L } _ { \mathrm { p u s h } } = \frac { 1 } { \vert \mathcal { P } \vert } \sum _ { ( i , j ) \in \mathcal { P } } \left[ s _ { i j } - m _ { \mathrm { p u s h } } \right] _ { + } ,\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { p u l l } } = \frac { 1 } { \vert \mathcal { Q } \vert } \sum _ { ( i , j ) \in \mathcal { Q } } \left[ m _ { \mathrm { p u l l } } - s _ { i j } \right] _ { + } ,\tag{4}
$$

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p u s h } } + \mathcal { L } _ { \mathrm { p u l l } } , } \end{array}\tag{5}
$$

with $m _ { \mathrm { p u s h } } < \tau _ { \mathrm { p u s h } }$ and $m _ { \mathrm { p u l l } } > \tau _ { \mathrm { p u l l } }$

A pair contributes gradient to optimize W until its similarity reaches the margin and none thereafter. The thresholds determine which pairs are moved and the margins determine how far they are moved. Replacing both hinges with unbounded terms costs 4.84 points of Worst Joint accuracy on UrbanCars (see Tab. 6).

Note that Eqs. (1) to (5) do not use any information regarding the bias attribute, and no text embedding or auxiliary models are used for our proposed approach. The proposed objective only utilizes image-to-image similarity and class labels alone to learn a linear map to debias the features of a frozen large pretrained model.

## 5.3. Overview and Inference

In the proposed approach, we initialize the linear map $\textbf { W } \in { \mathrm { ~ \mathbb { R } ^ { d \times d } ~ } }$ as an identity matrix, ensuring optimization begins from the encoder’s native geometry. It is the sole component that receives a gradient, introducing only $d ^ { 2 } \approx 0 . 5 9 \mathrm { M }$ learnable parameters compared to the backbone’s ∼300M. After learning the projection, we fit a linear classifier on the training split while keeping W fixed. Since the CLIP+linear-head baseline in Sec. 6.2 employs an identical head and training schedule on unprojected embeddings, the two setups differ exclusively by the presence of W. The objective $\mathcal { L }$ depends on W only through cosine similarities between transformed embeddings, so ${ \bf W } _ { \bf S }$ scale is irrelevant, and the optimization effectively learns a linear map that reshapes pairwise similarities in the frozen feature space. Performance gains come from the geometric objective guiding W toward a more robust representation.

## 6. Experiments

## 6.1. Setup

Datasets. We evaluate on four benchmarks. Urban-Cars [7] is a binary vehicle classification task constructed with two simultaneous shortcuts, image background (BG), and a co-occurring object (CoObj). BAR [11] is a six-class action recognition task in which each action co-occurs with a typical place, evaluated at bias ratios ρ =1% and 5%. BFFHQ [4, 5] is an age classification task with a demographic shortcut, evaluated at ρ =0.5%, 1%, 2%, and 5%. Waterbirds [13] is a bird classification task with a background shortcut, evaluated at $\rho = 5 \%$

<table><tr><td>Method</td><td>Venue</td><td>Ovr.</td><td>W-BG</td><td>W-Co</td><td>W-Joint</td></tr><tr><td colspan="6">Cosine similarity to prompt embeddings</td></tr><tr><td>CLIP-cos</td><td></td><td>50.00</td><td>4.03</td><td>13.71</td><td>0.00</td></tr><tr><td>RoboShot ICLR&#x27;24</td><td></td><td>51.81</td><td>13.71</td><td>20.16</td><td>0.00</td></tr><tr><td>FairerCLIP ICLR’24</td><td></td><td>81.25</td><td>64.52</td><td>69.36</td><td>58.06</td></tr><tr><td>PRISM</td><td>ICCV’25</td><td>53.43</td><td>6.45</td><td>16.13</td><td>1.61</td></tr><tr><td>PRISM-m</td><td>ICCV’25</td><td>53.63</td><td>24.19</td><td>27.42</td><td>6.45</td></tr><tr><td>†SEMi</td><td>CVPR&#x27;26</td><td>78.63</td><td>69.35</td><td>71.77</td><td>61.29</td></tr></table>

<table><tr><td colspan="5">Linear head on the corrected features</td></tr><tr><td>CLIP-L</td><td>81.18</td><td>68.55</td><td>65.32</td><td>38.71</td></tr><tr><td>RoboShot ICLR&#x27;24</td><td>72.40</td><td>48.39</td><td>51.61</td><td>8.06</td></tr><tr><td>FairerCLIP ICLR’24</td><td>50.81</td><td>9.68</td><td>41.94</td><td>8.06</td></tr><tr><td>PRISM ICCV’25</td><td>61.49</td><td>32.30</td><td>33.10</td><td>3.23</td></tr><tr><td>PRISM-m ICCV’25</td><td>73.79</td><td>52.40</td><td>51.60</td><td>11.29</td></tr><tr><td>Ours 一</td><td colspan="4">90.59 ±0.71 85.22 ±1.23 83.60 ±0.93 81.18 ±0.93</td></tr></table>

Table 4. UrbanCars: two simultaneous shortcuts. Methods are grouped by the rule used to classify, so that comparison is made between methods sharing a decision procedure. Shaded grey rows are the undebiased reference for each rule. Best method: bold, second: underlined. Results are averaged over 3 seeds. For our method, we report the mean and standard deviation.<sup>†</sup>SEM transforms only the text embeddings leaving image features unchanged, so a head fit on its features reproduces CLIP-linear exactly.

Metrics and Protocol. We report overall and worst-group accuracy. On UrbanCars we separate worst-group accuracy (WGA) by shift type: Worst BG (W-BG) and Worst CoObj (W-Co) are the lowest accuracy over groups in which the respective attribute is shifted, and Worst Joint (W-Joint) over groups in which both are shifted at once. Worst Joint is the quantity that multiple shortcuts affect directly. Hyperparameters are selected per dataset on a held-out validation split, and test results are reported for the selected configuration. We run each experiment with three random seeds and report the mean and standard deviation. Further implementation details are provided in the supplementary materials.

Baselines. All methods use the same frozen CLIP ViT-L/14 backbone. We include two undebiased references: CLIP with cosine similarity to prompt embeddings (CLIPcosine), and a linear classifier trained on the frozen embeddings (CLIP-linear). CLIP-linear is the matched-capacity reference for our method, using an identical head, optimizer, and scheduler, and differing only in the absence of W. We report our method under both classification rules. The cosine variant applies W to image embeddings and performs classification by cosine similarity with the same frozen prompt embeddings used by the vanilla-cosine classifier. We compare against SEM<sub>i</sub> [3], PRISM and PRISMmini [10], RoboShot [1] and FairerCLIP [2]. PRISM and RoboShot obtain their attributes from an LLM without human specification. For every competing method, we tune all of its hyperparameters and report the best configuration found. No method receives per-sample attribute annotations.

## 6.2. Main results

Multiple shortcuts. On the UrbanCars benchmark, every method that obtains its attribute from an LLM falls below the undebiased linear classifier on all four metrics (Tab. 4). Our method achieves 81.18% Worst Joint accuracy, outperforming the best competing method by 19.89 points and CLIP-linear by 42.47 points. The gap between overall and Worst Joint accuracy shrinks from 45.16 points for the baseline to 9.41 points for our method, while overall accuracy improves from 81.18% to 90.59%. Unlike text-based baselines, our method also applies to vision-only backbones such as DINOv3. We report these results in the supplementary materials.

Single shortcuts. The advantage extends to singleshortcut benchmarks (Tab. 5). On BAR, our method achieves the highest worst-group accuracy under both classification rules at both bias ratios. Under the linear head, we reach 83.10% at 1% and 84.51% at 5%, exceeding the best competitor by 1.41 points at each ratio. Under the cosine rule, the margin widens. We reach 87.79% at 1%, exceeding PRISM by 11.73 points. On BFFHQ, our WGA scales with the availability of conflicting samples, rising from 84.00% at 0.5% to 95.87% at 5% under the linear head. PRISM improves from 83.20% to 92.80% over the same range, but our method gains 11.87 points while PRISM gains 9.60 points. Our pair-selection strategy queries the dataset’s own geometry, so it naturally improves as more conflicting samples become available. On Waterbirds, our method reaches 80.01% WGA under the linear head, surpassing PRISM by 15.22 points and FairerCLIP by 19.11 points. Under the cosine rule, we reach 81.95%, exceeding FairerCLIP by 4.54 points and PRISM by 14.04 points.

## 6.3. Ablations

Our objective combines two forces. The first separates biassharing cross-class pairs $( { \mathcal { L } } _ { \mathrm { p u s h } } ) .$ and the second maintains same-class cohesion $( \mathcal { L } _ { \mathrm { p u l l } } )$ . Tab. 6 shows that both are necessary. Removing either term drops Worst Joint accuracy by at least 22.58 points. The margins prevent unbounded optimization and add a further 4.84 points on top of the full objective. Fig. 4 Sweeps the two thresholds on UrbanCars. We use four random margin values for each threshold pair and average the results, reporting the mean and standard deviation. Corresponding grids for BAR, BFFHQ, and Waterbirds are provided in the supplementary materials. Additional analyses on backbones, class imbalance, and the separation of same-class and cross-class pairs after projection are in the supplementary materials.

<table><tr><td></td><td></td><td></td><td colspan="4">BAR</td><td colspan="6">BFFHQ</td><td colspan="2">Waterbirds‡</td></tr><tr><td></td><td></td><td></td><td>1%</td><td colspan="2"></td><td></td><td colspan="2">0.5%</td><td>1%</td><td colspan="2">2%</td><td colspan="2">5%</td><td colspan="2">5%</td></tr><tr><td>Method</td><td>Venue</td><td>Bias info</td><td>Ovr.</td><td>Worst</td><td>Ovr. Worst</td><td>Ovr.</td><td>Worst</td><td>Ovr.</td><td>Worst</td><td>Ovr.</td><td>Worst</td><td>Ovr.</td><td>Worst</td><td>Ovr.</td><td>Worst</td></tr><tr><td colspan="10">Cosine similarity to prompt embeddings</td><td colspan="7"></td></tr><tr><td>CLIP-cosine</td><td></td><td>none</td><td>96.49</td><td>74.65</td><td>96.49</td><td>74.65</td><td>70.70 37.60</td><td>70.70</td><td>37.60</td><td>70.70</td><td>37.60</td><td>70.70</td><td>37.60</td><td>86.20</td><td>39.60</td></tr><tr><td>RoboShot[1]</td><td>ICLR’24</td><td>LLM</td><td>24.08</td><td>0.00</td><td>24.08</td><td>0.00</td><td>47.90 2.80</td><td>47.90</td><td>2.80</td><td>47.90</td><td>2.80</td><td>47.90</td><td>2.80</td><td>72.10</td><td>41.90</td></tr><tr><td>FairerCLIP[2]</td><td>ICLR’24</td><td>named</td><td>94.03</td><td>59.16</td><td>82.95</td><td>21.13</td><td>73.50 38.00</td><td>83.40</td><td>56.80</td><td>94.00</td><td>83.60</td><td>94.40</td><td>83.60</td><td>89.12</td><td>77.41</td></tr><tr><td>PRISM[10]</td><td>ICCV’25</td><td>LLM</td><td>95.96</td><td>76.06</td><td>95.96</td><td>76.06</td><td>91.80 87.60</td><td>91.80</td><td>87.60</td><td>91.80</td><td>87.60</td><td>91.80</td><td>87.60</td><td>89.18</td><td>67.91</td></tr><tr><td>PRISM-mini[10]</td><td>ICCV’25</td><td>LLM</td><td>96.49</td><td>74.65</td><td>96.49</td><td>74.65</td><td>73.60 39.20</td><td>73.60</td><td>39.20</td><td>73.60</td><td>39.20</td><td>73.60</td><td>39.20</td><td>87.23</td><td>53.43</td></tr><tr><td>SEMi[3]</td><td>CVPR&#x27;26</td><td>none</td><td>95.08</td><td>66.20</td><td>95.08</td><td>66.20</td><td>92.00 88.40</td><td>92.00</td><td>88.40</td><td>92.00</td><td>88.40</td><td>92.00</td><td>88.40</td><td>83.20</td><td>52.30</td></tr><tr><td>Ours</td><td></td><td>none</td><td>97.31</td><td>87.79</td><td>97.83</td><td>87.79</td><td>95.03 88.00</td><td>95.50</td><td>90.80</td><td>96.60</td><td>92.67</td><td>97.70</td><td>95.47</td><td>89.29</td><td>81.95</td></tr><tr><td colspan="10">Linear head on the corrected features</td><td colspan="7"></td></tr><tr><td>CLIP-linear</td><td></td><td>none</td><td>96.13</td><td>70.42</td><td>97.19</td><td>78.87</td><td>90.80 78.00</td><td>92.00</td><td>82.80</td><td>94.30</td><td>87.60</td><td>96.40</td><td>92.00</td><td>88.47</td><td>73.67</td></tr><tr><td>RoboShot[1]</td><td>ICLR&#x27;24</td><td>LLM</td><td>92.10</td><td>54.90</td><td>93.70</td><td>63.40</td><td>79.10 56.00</td><td>84.50</td><td>67.60</td><td>91.60</td><td>78.80</td><td>95.70</td><td>88.40</td><td>83.50</td><td>63.60</td></tr><tr><td>FairerCLIP[2]</td><td>ICLR&#x27;24</td><td>named</td><td>71.35</td><td>16.81</td><td>74.34</td><td>59.66</td><td>48.20 42.40</td><td>83.70</td><td>53.20</td><td>93.60</td><td>88.80</td><td>96.50</td><td>92.00</td><td>89.71</td><td>60.90</td></tr><tr><td>PRISM[10]</td><td>ICCV’25</td><td>LLM</td><td>97.19</td><td>81.69</td><td>97.19</td><td>83.10</td><td>91.80 83.20</td><td>93.90</td><td>84.80</td><td>95.10</td><td>89.60</td><td>96.60</td><td>92.80</td><td>82.95</td><td>64.79</td></tr><tr><td>PRISM-mini[10]</td><td>ICCV’25</td><td>LLM</td><td>94.02</td><td>57.75</td><td>95.08</td><td>61.97</td><td>80.70 57.20</td><td>86.50</td><td>72.80</td><td>92.00</td><td>79.60</td><td>96.00</td><td>88.80</td><td>85.42</td><td>70.64</td></tr><tr><td>Ours</td><td></td><td>none</td><td>97.54</td><td>83.10</td><td>97.89</td><td>84.51</td><td>92.67 84.00</td><td>94.30</td><td>86.67</td><td>95.60</td><td>90.67</td><td>97.93</td><td>95.87</td><td>89.35</td><td>80.01</td></tr></table>

Table 5. Results on the BAR, BFFHQ, and Waterbirds datasets. For our method, we report the mean and standard deviation. Note: <sup>‡</sup> For Waterbirds, we follow the evaluation protocol of [3], including the ViT-L/14@336px backbone. Further details, including standard deviations, are provided in the supplementary materials.

<table><tr><td colspan="3">Objective</td><td colspan="2">UrbanCars</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { p u s h } }$ </td><td> $\mathcal { L } _ { \mathrm { p u l l } }$ </td><td>margins</td><td>Ovr. W-BG</td><td>W-Co W-Joint</td></tr><tr><td>√</td><td></td><td>√</td><td>85.48 73.39</td><td>70.97 48.39</td></tr><tr><td rowspan="3">√</td><td>√</td><td>√</td><td>71.77 47.58</td><td>49.19 6.45</td></tr><tr><td></td><td></td><td>87.10 78.23 75.81</td><td>58.06</td></tr><tr><td>√</td><td></td><td>87.70 78.23 75.81</td><td>61.29</td></tr><tr><td>√</td><td>√</td><td></td><td>90.73 86.29</td><td>84.68 79.03</td></tr><tr><td>一</td><td></td><td></td><td>90.93 86.29</td><td>85.48 83.87</td></tr></table>

Table 6. Ablation on loss terms

## 7. Conclusion

We have shown that debiasing a frozen encoder need not require explicit spurious-attribute identification. The bias signal is already encoded in the frozen feature geometry. Highly similar cross-class pairs are overwhelmingly biassharing, and this is detectable from pairwise similarity and class labels alone. Our method turns this observation into a training objective that separates bias-sharing pairs and maintains same-class cohesion. It consistently outperforms post-hoc debiasing approaches, including on the multi-bias benchmark, where methods dependent on LLM-generated specifications underperform an undebiased linear classifier.

![](images/2ef24fed2b2ece5e977f2f7ebc33b885b9b42d03b42485b094f34cf1648fc5e3.jpg)  
Figure 4. Threshold stability on UrbanCars. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

## References

[1] Dyah Adila, Changho Shin, Linrong Cai, and Frederic Sala. Zero-shot robustification of zero-shot models. In The Twelfth International Conference on Learning Representations, 2024. 1, 2, 3, 4, 7, 8

[2] Sepehr Dehdashtian, Lan Wang, and Vishnu Naresh Boddeti. Fairerclip: Debiasing clip’s zero-shot predictions using functions in rkhss. In International Conference on Learning Representations (ICLR), 2024. 1, 2, 3, 7, 8

[3] Quentin Guimard, Federico Bartsch, Simone Caldarella, Rahaf Aljundi, Elisa Ricci, and Massimiliano Mancini. SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models. In Findings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3, 7, 8

[4] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. 7

[5] Eungyeup Kim, Jihyeon Lee, and Jaegul Choo. Biaswap: Removing dataset bias with bias-tailored swapping augmentation. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 7

[6] Tyler LaBonte, Vidya Muthukumar, and Abhishek Kumar. Towards last-layer retraining for group robustness with fewer annotations. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2

[7] Zhiheng Li, Ivan Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Canton Ferrer, Chenliang Xu, and Mark Ibrahim. A whac-a-mole dilemma: Shortcuts come in multiples where mitigating one amplifies others. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20071–20082, 2023. 1, 3, 5, 6

[8] Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. Just train twice: Improving group robustness without training group information. In Proceedings of the 38th International Conference on Machine Learning, pages 6781–6792. PMLR, 2021. 2

[9] Nicole Meister, Dora Zhao, Angelina Wang, Vikram V. Ramaswamy, Ruth Fong, and Olga Russakovsky. Gender artifacts in visual datasets. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20267–20277, 2023. 1

[10] Mahdiyar Molahasani, Azadeh Motamedi, Michael Greenspan, Il-Min Kim, and Ali Etemad. Prism: Reducing spurious implicit biases in vision-language models with llm-guided embedding projection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 688–697, 2025. 1, 2, 3, 4, 7, 8

[11] Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin. Learning from failure: De-biasing classifier from biased classifier. In Advances in Neural Information Processing Systems, pages 20673–20684. Curran Associates, Inc., 2020. 2, 6

[12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2

[13] Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori B. Hashimoto, and Percy Liang. Distributionally robust neural networks. In International Conference on Learning Representations, 2020. 2, 7

[14] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Fed-´ erico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seung Eun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel HAZIZA, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th ´ eo Moutakanni, Leonel´ Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve Jegou, Patrick Labatut, and Piotr Bojanowski. DINOv3. Transactions on Machine Learning Research, 2026. 2

[15] Angelina Wang, Alexander Liu, Ryan Zhang, Anat Kleiman, Leslie Kim, Dora Zhao, Iroha Shirai, Arvind Narayanan, and Olga Russakovsky. Revise: A tool for measuring and mitigating bias in visual datasets. International Journal ofCom puter Vision, 130(7):1790–1810, 2022. 1

[16] Jialu Wang, Yang Liu, and Xin Wang. Are gender-neutral queries really gender-neutral? mitigating gender bias in image search. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Associa tion for Computational Linguistics, 2021.

[17] Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky. Towards fairer datasets: filtering and balancing the distribution of the people subtree in the imagenet hierarchy. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, page 547–558. Associa tion for Computing Machinery, 2020.

[18] Boya Zeng, Yida Yin, and Zhuang Liu. Understanding bias in large-scale visual datasets, 2024. 1

[19] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11975–11986, 2023. 2

[20] Kankan Zhou, Eason Lai, and Jing Jiang. VLStereoSet: A study of stereotypical bias in pre-trained vision-language models. In Proceedings of the 2nd Conference of the Asia Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Nat ural Language Processing (Volume 1: Long Papers), pages 527–538, Online only, 2022. Association for Computational Linguistics. 1

# Supplementary Material

## 1. Implementation Details

Algorithm 1 states the procedure in full. Pair selection runs once, over the cached embeddings of the frozen encoder, and is the only stage whose cost grows with the size of the training split. Learning W then operates on the selected pairs alone.

Training. $\mathbf { W } \in \mathbb { R } ^ { d \times d }$ is initialized at I and trained with Adam at a learning rate of $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , a cosine schedule, mini-batches of 256 for BAR dataset and 512 for all other datasets, drawn from $\mathcal { P } \cup \mathcal { Q }$ . The frozen encoder receives no gradient at any point. W is the only component the objective updates, at $d ^ { 2 }$ ≈ 0.59M parameters against the backbone’s ∼300M.

Classification. Under the cosine rule, we classify by similarity between the projected image embedding and the frozen class-wise prompt embeddings, exactly as CLIPcosine does, and fit nothing. Under the linear head we fit a linear classifier (logistic regression) with balanced class weights on the projected training split, with W held fixed and no further tuning. The CLIP-linear reference uses an identical head on unrectified embeddings, so the two differ only in the presence of W.

Hyperparameters. The two thresholds and margins are selected per dataset. For the push set, $m _ { \mathrm { p u s h } }$ is always chosen below $\tau _ { \mathrm { p u s h } } ,$ so the hinge pushes selected pairs past the threshold rather than merely toward it. For the pull set, $m _ { \mathrm { p u l l } }$ is always chosen above $\tau _ { \mathrm { p u l l } } .$ , so the hinge pulls selected pairs beyond the threshold. Section 3.2 gives the swept grids and the configuration marked on each grid is the one selected and used for the reported results.

## 2. Baselines

Every method is run on the same frozen features and evaluated on the same split. For SEM, we use the bias-agnostic variant $\mathrm { S E M } _ { i }$ , which names no spurious attribute, and its released implementation returns image embeddings unchanged. Therefore, for $\operatorname { S E M } _ { i } ,$ we cannot have any linearhead variant. For the methods that do transform image embeddings we obtain the head variant by fitting the logistic regression of Sec. 1 on their rectified features.

Algorithm 1 Proposed Approach   
Require: frozen embeddings $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { n }$ , labels $\{ y _ { i } \}$ , thresh  
olds $\tau _ { \mathrm { p u s h } } , \tau _ { \mathrm { p u l l } } .$ , margins m<sub>push</sub>, $m _ { \mathrm { p u l l } } .$ , steps T   
Ensure: W and a linear head   
1: $\mathcal { P }  \{ ( i , j ) : i < j , \ y _ { i } \neq y _ { j } , \ \langle \mathbf { x } _ { i } , \mathbf { x } _ { j } \rangle > \tau _ { \mathrm { p u s h } } \}$   
2: for $i = 1 , \ldots , n$ do   
3: $\mathcal { C } _ { i }  \{ j : y _ { j } = y _ { i } , \ : \langle \mathbf { x } _ { i } , \mathbf { x } _ { j } \rangle < \tau _ { \mathrm { p u l l } } \}$   
4: end for   
5: ${ \mathcal { Q } } \gets \{ ( i , j ) : { \mathcal { C } } _ { i } \neq \emptyset , \ j \sim \operatorname { U n i f } ( { \mathcal { C } } _ { i } ) \}$ , capped at   
$| \mathcal { P } | / N$ per class among the N classes   
6: $\mathbf { W }  \mathbf { I } _ { d }$   
7: for $t = 1 , \dots , T$ do   
8: draw a mini-batch $B \subseteq \mathcal { P } \cup \mathcal { Q } ; \quad \mathbf { z } _ { i } \quad $   
$\mathbf { W } \mathbf { x } _ { i } / \| \mathbf { W } \mathbf { x } _ { i } \|$   
9: $\mathbf { W }  \mathrm { A D A M } \big ( \mathbf { W } , \nabla _ { \mathbf { W } } \mathcal { L } ( \mathcal { B } ) \big )$   
10: end for   
11: fit a linear head on $\left\{ \mathbf { z } _ { i } \right\}$ over the training split

## 2.1. Oracle specifications supplied to PRISM

The oracle experiment replaces the attributes that a language model proposes with an expert-written specification of the two shortcuts UrbanCars contains, leaving every other part of PRISM and PRISM-mini approaches unchanged. The specification names the background and the co-occurring object and enumerates the values each takes. The two methods consume it differently. PRISM generates scene descriptions from the specification and learns a projection under which descriptions differing only in an attribute become indistinguishable. Therefore, what it receives is a set of sentences. PRISM-mini trains nothing and directly builds its orthogonal projection from the attribute directions. Therefore, it receives only the values of the bias attributes, not descriptions.

The descriptions given to PRISM cover the eight cells of the class × background × co-occurring object Table 1, so that the class and each attribute vary independently. Within a cell, incidental properties such as color, vehicle type, lighting, and road surface vary freely, and only the class and the two attributes are held to the values the cell specifies, which is what allows the objective to attribute the contrast to the attribute rather than to some correlated detail. One description per cell is shown below in Table 1 for demonstration purposes.

![](images/7363340cf7b83f6c1c102f34b1d1ce893348416a30be0ef3573af882c14247bd.jpg)  
Figure 1. Threshold stability on BAR at 1% conflicting ratio. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

![](images/fac6d6731b1438d47f8d11d981c1cfd32f3e01b737c35f79a57f8395e905223e.jpg)  
Figure 2. Threshold stability on BAR at 5% conflicting ratio. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

## 3. Additional Results

## 3.1. Seed sweep

Table 2 reports our results on BAR, BFFHQ, and Waterbirds datasets over three seeds under both classification rules.

## 3.2. Sensitivity to the selection thresholds

Figures 1 to 7 show how accuracy varies with the two selection thresholds. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the configuration selected and used for all reported results. For every dataset and ratio, a broad region around the selected cell performs within a few points of the best configuration, and the selected cell lies within that region. Variance concentrates where the threshold has very few pairs to work with.

![](images/e5a688c1fcc41be4df5537f57e1c46d83d733d98b271a4159aedc446b53d77ec.jpg)  
Figure 3. Threshold stability on BFFHQ at 0.5% conflicting ratio. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

![](images/46bd5ab8751a17b1117337cdb09ba8051b2186a1541325ca8fcb1410cf8c0708.jpg)  
Figure 4. Threshold stability on BFFHQ at 1% conflicting ratio. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

![](images/dc7bb4544a7609808e73a3a046bd27b4d0eb3ad5db7f66ba654b13071977e0b3.jpg)  
Figure 5. Threshold stability on BFFHQ at 2% conflicting ratio. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

## 3.3. Other backbones

Our method is not tied to a particular backbone. The objective computes cosine similarities between image embeddings. Therefore, it applies to any frozen encoder of a large pre-trained model. Table 3 repeats the UrbanCars experiment on SigLIP and DINOv3. Accuracy improves on all four metrics for every backbone. The improvement narrows as the frozen representation becomes less entangled to begin with. On CLIP, Worst Joint accuracy rises by 45.16 points, on SigLIP by 24.20 points, and on DINOv3 by 3.22 points. These rows are single runs at a fixed seed, not the three means reported in the main paper.

<table><tr><td>Class</td><td>Background</td><td>Co-obj</td><td>Prompt</td></tr><tr><td>urban car</td><td>urban</td><td>urban</td><td> $" \mathtt { A }$  sleek black SUV cruises down a neon-lit city avenue, passing a bright orange traffic cone near the curb.&quot;</td></tr><tr><td>urban car</td><td>urban</td><td>country</td><td> $" \mathtt { A }$  sleek urban car is parked on a bustling downtown street while a curious cow grazes on a patch of grass beside the curb.&quot;</td></tr><tr><td>urban car</td><td>country</td><td>urban</td><td> $" \mathtt { A }$  glossy urban car is parked on a dusty dirt road beside a bright red fire hydrant, with golden wheat fields stretching behind it.&quot;</td></tr><tr><td>urban car</td><td>country</td><td>country</td><td> $" \mathtt { A }$  sleek urban car is parked on a grassy farm lane, while a curious cow grazes beside a weathered wooden</td></tr><tr><td>country car</td><td>urban</td><td>urban</td><td>barn in the background.&quot;  $" \mathtt { A }$  country car is parked on a bustling city street lined with brick buildings, next to a bright red fire</td></tr><tr><td>country car</td><td>urban</td><td>country</td><td>hydrant.&quot;  $" \mathtt { A }$  bright red country car is stopped at a busy city intersection, with a lone cow standing on the sidewalk</td></tr><tr><td>country car</td><td>country</td><td>urban</td><td>beside it.&quot;  $" \mathtt { A }$  dusty country car sits on a dirt road beside a bright red fire hydrant.&quot;</td></tr><tr><td>country car</td><td>country</td><td>country</td><td> $" \mathtt { A }$  bright red country car rolls across a golden field as a curious cow watches from the edge of the grass.&quot;</td></tr></table>

Table 1. Oracle prompts used for UrbanCars. Each row shows one prompt per combination of class, background, and co-occurring object. These prompts replace the LLM-generated descriptions in PRISM.

<table><tr><td rowspan="2">Dataset Ratio</td><td colspan="2">Cosine</td><td colspan="2">Linear head</td></tr><tr><td>Ovr.</td><td>Worst</td><td>Ovr.</td><td>Worst</td></tr><tr><td>BAR 1%</td><td> $9 7 . 3 1 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $8 7 . 7 9 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $9 7 . 5 4 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $8 3 . 1 0 { \scriptstyle \pm 1 . 4 1 }$ </td></tr><tr><td>BAR 5%</td><td> $9 7 . 8 3 _ { \pm 0 . 2 2 }$ </td><td> $8 7 . 7 9 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $9 7 . 8 9 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 4 . 5 1 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>BFFHQ 0.5%</td><td> $9 5 . 0 3 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $8 8 . 0 0 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $9 2 . 6 7 { \scriptstyle \pm 0 . }$  25</td><td> $8 4 . 0 0 { \scriptstyle \pm 0 . 4 0 }$ </td></tr><tr><td>BFFHQ 1%</td><td> $9 5 . 5 0 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $9 0 . 8 0 { \scriptstyle \pm 0 .  }$  86</td><td> $9 4 . 3 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $8 6 . 6 7 _ { \pm 0 . 2 3 }$ </td></tr><tr><td>BFFHQ 2%</td><td> $9 6 . 6 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $9 2 . 6 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $9 5 . 6 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $9 0 . 6 7 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td>BFFHQ 5%</td><td> $9 7 . 7 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $9 5 . 4 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $9 7 . 9 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $9 5 . 8 7 _ { \pm 0 . 2 3 }$ </td></tr><tr><td>Wbirds 5%</td><td> $8 9 . 2 9 _ { \pm 0 . 0 2 }$ </td><td> $8 1 . 9 5 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $8 9 . 3 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $8 0 . 0 1 _ { \pm 0 . 1 5 }$ </td></tr></table>

Table 2. Seed sweep on BAR, BFFHQ and Waterbirds datasets. Mean and standard deviation over three seeds.

The DINOv3 rows are possible only for our method. Every frozen-backbone method we compare against derives its correction from prompts, whether written by a practitioner, proposed by a language model, and DINO being a visiononly encoder supplies none. Our objective reads imageimage similarity, so the text side is never consulted.

![](images/5daffa5d838dbbe35d5c30e55349a7c2a4d17c53cea2e23aa62747cb03c90b0d.jpg)  
Figure 6. Threshold stability on BFFHQ at 5% conflicting ratio. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

## 3.4. Class imbalance

The benchmarks of the main paper are class-balanced, which is not typical of data collected in the wild. Table 4 resamples the UrbanCars training split so that the two classes appear in a 70:30 and a 90:10 ratio, leaving the shortcut correlations unchanged. Under both classification rules, our method performs significantly better than the undebiased reference of 38.71. Therefore, the proposed method does not depend on the balanced construction of these benchmarks.

![](images/924e6570cf6d512635114f639afe37c4ab4f8a01ca6501f94b62c54eb3176984.jpg)  
Figure 7. Threshold stability on Waterbirds. Each cell reports the mean and spread across the four margin settings at that threshold pair. The red border box marks the selected configuration.

<table><tr><td colspan="2">Backbone Ovr. W-BG W-Co W-Joint</td></tr><tr><td>CLIP ViT-L/14</td><td>no debiasing 83.87 68.55 65.32 38.71</td></tr><tr><td>+ ours</td><td>90.93 86.29 85.48 83.87</td></tr><tr><td>SigLIP ViT-L/16</td><td>no debiasing 87.70 79.03 78.23 62.90</td></tr><tr><td>+ ours</td><td>92.54 87.90 88.71 87.10 </td></tr><tr><td>DINOv3 ViT-L/16† no debiasing 92.34</td><td>86.29 87.90 80.65</td></tr><tr><td>+ ours</td><td>93.15 89.52 89.52 83.87</td></tr></table>

Table 3. The method transfers across frozen encoders. UrbanCars, with and without our projection, evaluated with the linear head. Accuracy improves on all four metrics for every backbone, by a margin that narrows as the undebiased representation improves. Rows are single runs at a fixed seed. <sup>†</sup>DINOv3 is a vision-only encoder and produces no text embeddings, so none of the prompt-based methods we compare against can be applied to it and our method does not use the text side.

## 4. Waterbirds Evaluation Protocol

For Waterbirds, we follow the evaluation protocol of [1], including the ViT-L/14@336px backbone. Baselines other than CLIP-linear and our own are reproduced using the [1] implementation. We train our method using the hyperparameter search procedure described in Sec. 1. The frozen encoder is CLIP ViT-L/14@336px. We apply W to the resulting embeddings and train the linear head on the rectified features.

## 5. Additional Analysis

## 5.1. Class structure in the projected space

The main paper finds that in the frozen encoder a change of background costs almost as much similarity as a change of class. Figure 8 repeats that measurement after projection. Mean similarity rises from 0.626 to 0.792 for same-class pairs and falls from 0.551 to 0.421 for cross-class pairs, widening the gap between them from 0.075 to 0.371. The fraction of cross-class pairs more similar than the average same-class pair falls from 12.7% to 0.45%, and the probability that a randomly drawn same-class pair is more similar than a randomly drawn cross-class pair rises from 0.754 to 0.980. These are measurements on the training split and describe how the projection reshapes the representation rather than adding evidence of generalization.

<table><tr><td colspan="2">Class ratio</td><td>Ovr. W-BG</td><td>W-Co</td><td>W-Joint</td></tr><tr><td colspan="2">CLIP-cosine CLIP-linear</td><td>50.00 83.87</td><td>4.03 13.71 68.55 65.32</td><td>0.00 38.71</td></tr><tr><td>balanced</td><td>cosine head</td><td>89.11 90.93</td><td>76.61 76.61 86.29 85.48</td><td>66.13 83.87</td></tr><tr><td>70:30</td><td>cosine head</td><td>92.90 92.54</td><td>88.70 88.70 86.29 83.87</td><td>87.10 80.65</td></tr><tr><td>90:10</td><td>cosine head</td><td>88.30</td><td>80.60 75.80</td><td>66.10</td></tr><tr><td colspan="2"></td><td>90.50</td><td>87.90 80.60</td><td>80.60</td></tr></table>

Table 4. Robustness to class imbalance on UrbanCars. The training split is resampled so that the two classes appear in the stated ratio, leaving the shortcut correlations unchanged. Worst Joint accuracy never falls below twice the undebiased reference of 38.71. Both classification rules are reported, and rows are single runs at a fixed seed.

![](images/847a4fadbe16707fa9e18087730bca65db1d880667b620685932df3d8e9f21a9.jpg)

![](images/8a270462dd29525256fa01eb744e4eae16665d563579400ee8279c9927451c8a.jpg)  
Figure 8. Class separation before and after the projection. Pairwise cosine similarity over UrbanCars training pairs, split by whether the pair shares a class, with the overlap between the two shaded. AUC is the probability that a randomly drawn same-class pair is more similar than a randomly drawn cross-class pair and at 0.5 similarity would carry no information about class membership. Curves are normalized to equal height, as same-class pairs are the more numerous.

![](images/7ffe1dd27b88d63fd93ac6fd3ba550a793fe2142f981ef51494359e5b3e694c3.jpg)  
Figure 9. Where conflicting samples sit. UrbanCars training embeddings projected onto the plane spanned by the class direction and the background direction, before and after our projection. Filled points are bias-aligned samples and open points are biasconflicting. Each arrow runs from the aligned mean of a class to its conflicting mean, and the dashed line marks the midpoint between the two class means. Attribute labels are used for this measurement only and never during training.

## 5.2. Where conflicting samples move

The main paper showed that baseline finds it difficult to identify bias-conflicting samples relative to the class boundary. Projecting the embeddings onto the plane spanned by the class direction and the background direction makes that position visible (Fig. 9). Both directions are differences of group means computed after training, and the background direction is orthogonalized against the class direction so that the two axes are not redundant, and neither enters the objective. In the frozen space, a conflicting sample sits on average 47% of the way toward the opposing class along the class direction. Therefore, many of these samples shift to the wrong side of a boundary fitted to the majority. After projection, the same measurement gives 11%. The plane retains two of 768 dimensions and is shown for interpretation, and the similarity measurements above are computed in the full space.

## References

[1] Quentin Guimard, Federico Bartsch, Simone Caldarella, Rahaf Aljundi, Elisa Ricci, and Massimiliano Mancini. SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models. In Findings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 4