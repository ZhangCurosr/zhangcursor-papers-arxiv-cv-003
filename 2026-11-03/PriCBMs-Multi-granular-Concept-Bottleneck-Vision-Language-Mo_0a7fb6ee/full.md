# PriCBMs: Multi-granular Concept Bottleneck Vision-Language Models for Visual Privacy Recognition

Shenglei Pei<sup>1</sup>, Haotian Sun<sup>1</sup>, Pengfei Li<sup>1</sup>, Bingxu Lu<sup>3</sup>, Wenlong Yu<sup>2</sup>

<sup>1</sup>School of Intelligence Science and Engineering, Qinghai Minzu University, Xining 810007, China <sup>2</sup>School of Computing and Information Systems, Singapore Management University, Singapore 178902, Singapore

<sup>3</sup>Tianjin Key Laboratory of Interventional Brain-Computer Interface and Intelligent Rehabilitation and Academy for Advanced Interdisciplinary Studies, Nankai University, Tianjin 300350, China

Emails: peishenglei@126.com (Shenglei Pei); sun2293674491@gmail.com (Haotian Sun); lipengfei@qhmu.edu.cn (Pengfei Li); lubingxu@nankai.edu.cn (Bingxu Lu); wloongyu@gmail.com (Wenlong Yu).

Corresponding authors: Wenlong Yu and Bingxu Lu. Shenglei Pei ORCID: https://orcid.org/0000-0002-8070-8580

## Abstract

Widespread image sharing on social media has increased the need for visual privacy recognition to identify inadvertently exposed privacy information. However, multi-label vi sual privacy recognition presents two challenges. First, privacy categories represent high-level semantic attributes and do not have a one-to-one correspondence with visual objects. Different visual objects may involve the same privacy category, while the same type of object may contain diferent types of privacy-sensitive information. In this setting, direct prediction methods generally lack explicit guidance for identifying category-relevant privacy cues, leaving this process implicit in the learned image-to-label mapping and potentially limiting classification performance. Second, privacy cues are often distributed across image regions and may not be adequately represented by whole-image features. To address these challenges, we propose Privacy Concept Bottleneck Models (PriCBMs). By introducing privacyrelated concepts as intermediate semantic representations, PriCBMs first identify categoryrelevant privacy cues through visual–concept matching and then predict privacy categories based on concept responses. This process makes privacy predictions more interpretable. To improve the quality of these concepts, we design a concept selection strategy based on target-category relevance, cross-category discriminability, and semantic diversity. Meanwhile, we introduce a global-region modeling module to capture localized privacy evidence while retaining global context. Experiments on VISPR and FineViP show that PriCBMs outperform evaluated competitors across all four multi-label metrics. Ablation studies and concept-quality analyses support the efectiveness of the proposed design. Code is available at https://github.com/sun2293674491-dotcom/PriCBMs.

Keywords: Visual privacy recognition; Concept bottleneck models; Vision-language models; Multi-label learning

## 1 Introduction

The rapid growth of social media and image-sharing platforms has made image sharing an increasingly common form of online communication. Images shared online may inadvertently expose sensitive information related to personal identity, location, social relationships, and other private content [1, 2]. This has created a growing need for efective visual privacy recognition. For visual privacy recognition, especially for screening images before online sharing, low-latency inference and on-device deployment are desirable, as cloud-based processing may introduce transmission delays and additional privacy exposure risks.

Early studies mainly formulated visual privacy recognition as a binary private/public classification problem [3–5]. Binary prediction provides a basic indication of whether an image contains privacy-sensitive information. However, a private/public decision alone provides limited information about the specific privacy risks present in an image, making it dificult for users to understand and manage these risks. This gap between binary privacy judgments and users need for more specific privacy information has motivated the development of multi-label visual privacy recognition. Multi-label visual privacy recognition allows multiple privacy-related labels to be identified simultaneously within the same image, thereby providing more specific information about the types of privacy-sensitive content present in the image. Existing approaches to multi-label visual privacy recognition [1, 6] typically learn a direct mapping from visual features to privacy labels using multi-label classification objectives.

Despite the progress made in multi-label visual privacy recognition, two challenges remain. First, privacy categories represent high-level semantic attributes and do not have a one-to-one correspondence with visual objects. The same privacy category may involve diferent visual objects, while the same type of visual object may contain diferent types of privacy-sensitive information. This complex relationship makes it challenging to identify category-relevant privacy cues in images. Existing direct prediction methods generally lack explicit guidance for identifying category-relevant privacy cues, leaving this process implicit in the learned image to-label mapping and potentially limiting classification performance. Second, privacy cues are often distributed across diferent image regions. Although whole-image features capture overall visual information, they may not adequately represent localized privacy cues, particularly those in small or visually less prominent regions [7, 8]. Relying solely on whole-image features may therefore limit the accurate and comprehensive identification of such cues.

To address these challenges, we propose Privacy Concept Bottleneck Models (PriCBMs). Based on the concept bottleneck framework [9], PriCBMs introduce privacy-related concepts as intermediate semantic representations between images and privacy categories. By constructing a privacy concept bank, PriCBMs match visual features with predefined privacy concepts to help identify category-relevant privacy cues across diverse visual content. Subsequently, privacy categories are predicted from concept responses. This design replaces direct image-to-label prediction with an explicit concept bottleneck that guides the model to identify category-relevant privacy cues through visual–concept matching. This structural change helps improve classification performance, while the concept bottleneck enhances model interpretability. To improve the quality and discriminative power of the privacy concept bank, we design a privacy-oriented concept selection strategy based on three criteria, namely target-category relevance, cross-category discriminability, and semantic diversity. The selected concepts are then combined to form the privacy concept bank. PriCBMs further introduce a global-region modeling module that enhances the capture of localized privacy cues while preserving global contextual information. Category-specific fusion weights are used to balance global context and localized evidence for each privacy category. The contributions of this paper are summarized as follows:

• We propose Privacy Concept Bottleneck Models (PriCBMs) for multi-label visual privacy recognition. By introducing privacy-related concepts as intermediate semantic representations, PriCBMs formulate privacy prediction as an explicit image–concept–category process, thereby helping identify category-relevant privacy cues across diverse visual content.

• We develop a global-region modeling module that enhances the capture of localized privacy cues while preserving global contextual information. We also introduce a privacy-oriented concept selection strategy that considers target-category relevance, cross-category discriminability, and semantic diversity, yielding a compact and discriminative privacy concept bank.

• Experiments on two multi-label visual privacy datasets show that PriCBMs outperform the evaluated competitors across all four multi-label metrics. Ablation studies validate the proposed strategies, and concept-quality analyses confirm the quality of the selected concepts under the evaluated settings.

## 2 Related Work

Early studies on visual privacy recognition formulated the task as binary private/public clas sification. PicAlert! [4] combined visual features, user-provided tags, and metadata to predict whether an image was private or public. Privacy-aware image classification and search [3] explored privacy prediction in image classification and retrieval settings. Tonge and Caragea [5] investigated deep neural network representations for image privacy prediction, while Privacy-CNH [10] employed hierarchical visual representations for privacy classification. Yang et al. [11] further modeled object–privacy relationships using graph-based neural networks to provide explainable image privacy inference. Some studies also considered how privacy judgments may vary across users and application scenarios. Personalized privacy-aware image classification [12] modeled diferences in privacy preferences across users. VizWiz-Priv [13] investigated private visual information in images captured by blind users, while PrivacyAlert [14] studied image privacy prediction in social media sharing scenarios. Despite these advances, binary private/public prediction provides only a coarse assessment of privacy risk and cannot specify the types of sensitive information exposed in an image. PriCBMs therefore focus on multi-label visual privacy recognition, allowing multiple privacy categories to be identified within the same image.

Multi-label visual privacy recognition was developed to identify multiple types of sensitive information within the same image. VISPR [1] introduced a taxonomy of privacy attributes and established a benchmark for multi-label privacy attribute prediction. MSPP-Net [15] adopted a multi-stage semantic perception framework that progressively modeled entities, semantic attributes, and inter-object context for privacy recognition. Related visual studies have explored label-specific [16], multi-granularity [17], and knowledge-enhanced representations [18], including feature structuralization for fine-grained domain generalization [19], LLM-generated semantic descriptions for wildlife recognition [20], and parameter-free knowledge embedding with fuzzy cognitive maps for class-incremental learning [21]. These studies reflect eforts to incorporate discriminative semantic or prior knowledge into visual learning. However, in existing multi-label privacy recognition, semantic and contextual information was still mainly used to enrich learned representations, while privacy-category prediction was not explicitly constrained by categoryrelevant semantic evidence. As a result, models may rely on dominant or shared visual patterns rather than category-relevant cues needed to distinguish diferent privacy categories, potentially leading to category confusion. PriCBMs introduce category-relevant privacy concepts as intermediate representations, reformulating direct image-to-label prediction as an explicit image– concept–category process in which category prediction is mediated through privacy-concept responses.

Spatial cues have also been explored in multi-label visual privacy recognition. PrivAttNetMLC [6] employed visual attention to emphasize privacy-relevant regions, while Jiang et al. [22] employed privacy-specific spatial attention to capture attribute-dependent visual evidence across diferent image regions. These studies progressively strengthened the use of spatial cues from attention-based feature emphasis to attribute-specific spatial modeling. However, localized cues were mainly integrated through attention weighting and subsequent aggregation, which may weaken privacy evidence distributed across diferent or less prominent regions. PriCBMs instead explicitly model whole-image and region-level concept responses as complementary sources of privacy evidence and integrate them for multi-label category prediction across diferent spatial granularities. Beyond privacy recognition, Object-Centric Concept-Bottlenecks (OCB) [7] combines image- and object-level concept activations for visual classification, whereas PriCBMs use privacy-specific concepts and category-specific global-region fusion with regional features derived from pooled patch tokens.

Interpretability was also explored in multi-label visual privacy recognition. Existing approaches mainly provided spatially grounded evidence through attention or localization mechanisms, allowing users to more directly inspect where privacy-related cues occurred in an image. For example, the attention mechanism in PrivAttNet [6] highlighted image regions associated with privacy predictions, while the privacy-specific attention mechanism of Jiang et al. [22] exposed spatial evidence associated with individual privacy attributes. Such mechanisms improved the spatial inspectability of multi-label privacy predictions by revealing where relevant evidence was located. Beyond spatial localization, concept-based methods have also explored semantic explanations for visual predictions. CoE generates natural-language explanations by tracing visual concept circuits and analyzing concept polysemanticity [23]. However, dedicated studies on explaining multi-label privacy predictions remain limited, particularly in providing explicit semantic evidence that clarifies what privacy-related information supports individual category predictions. In PriCBMs, the concept bottleneck layer provides an explicit semantic interface between visual representations and privacy categories. Since category predictions are mediated through privacy concepts, the concept responses and their associations with privacy categories can be inspected to understand the semantic representations associated with individual predic tions, thereby providing inherent interpretability within the prediction process.

## 3 Method

The goal of multi-label visual privacy recognition is to identify all privacy-risk categories in an image. Let $C = C _ { p } \cup \{ c _ { \mathrm { s a f e } } \}$ , where $C _ { p }$ contains L privacy-risk categories and $c _ { \mathrm { s a f e } }$ denotes Safe/Public. For the j-th privacy-risk category $c _ { j } \in C _ { p } ,$ the ground-truth label is $y _ { j } \in \{ 0 , 1 \}$ where $y _ { j } = 1$ indicates the presence of the privacy risk. The PriCBMs framework is illustrated in Fig. 1.

![](images/b05b5f4c5ca0a04017cfbc3975dc29ca8e2bdd03d9be9a1080be5ecd524e6022.jpg)  
Figure 1: Overall framework of the proposed PriCBMs. The example image is from VISPR [1].

Unlike direct image-to-label prediction, PriCBMs introduce privacy-related natural language concepts as intermediate semantic representations, establishing an image–concept–category process. PriCBMs construct a privacy concept bank by generating category-specific candidate concepts and selecting them based on target-category relevance, cross-category discriminability, and semantic diversity.

The selected concepts and input image are encoded by Long-CLIP [24] into a shared visionlanguage embedding space. The image is represented by a global feature and multiple region-level visual features.

The global image feature is matched with the concept representations to obtain global concept responses. Meanwhile, multiple regional visual features extracted from each image are matched with the concept representations to obtain concept responses for each region. These responses are aggregated using Softplus Mean Aggregation (SMA) to capture localized privacy evidence. The global and regional concept responses are mapped to category-level scores through a shared concept-to-class association matrix and adaptively fused using category-specific weights. The fused category scores are converted into multi-label prediction probabilities.

Unlike privacy-risk categories, Safe/Public encompasses a wide range of visual content that may also appear in privacy-sensitive images, making it challenging to construct a set of discriminative visual concepts for this category. Therefore, PriCBMs treat Safe/Public as a residual category and assign it only when no privacy-risk category is activated for that image.

The following subsections detail the main components of PriCBMs and their roles in privacy recognition, beginning with privacy concept generation and selection, followed by global-region concept modeling. We then describe the inference and calibration procedure and the training objective.

## 3.1 Concept Generation and Selection

This section describes how PriCBMs construct the privacy concept bank through candidate concept generation, semantic discriminability selection, and clustering-based diversity selection, as illustrated in Fig. 2.

PriCBMs use two frozen text encoders for diferent purposes. The all-mpnet-base-v2 encoder in the SentenceTransformer framework [25], denoted as E<sub>s</sub>(·), is used for concept selection. After concept selection, the concepts are encoded by the Long-CLIP text encoder [24] for image– concept matching. The resulting concept embeddings are precomputed and kept fixed during training.

![](images/3e7c5d23d36b66dccce66b1ac4d10a43c8a87add8333bd4a143e692e719876e4.jpg)  
Figure 2: Flowchart of the concept generation and selection process.

## 3.1.1 Concept Construction

For the j-th privacy-risk category c<sub>j</sub>, we prompt GPT-4o [26] to generate candidate visual concepts specifically for that category. After duplicate and invalid outputs are removed, we

retain N valid candidate concepts as

$$
Z _ { j } = \{ z _ { j , 1 } , z _ { j , 2 } , \ldots , z _ { j , N } \} .\tag{1}
$$

## 3.1.2 Semantic Discriminability Selection

Semantic Discriminability Selection retains concepts that are relevant to the target privacyrisk category and help distinguish it from other privacy-risk categories by measuring similarity margins against other category descriptions.

Each privacy-risk category is represented by the following templated input.

$$
\phi ( c _ { j } ) = { } ^ { \cdots } \mathrm { A } \ \mathrm { p h o t o \ s h o w i n g } \ c _ { j } { } ^ { \cdots } .
$$

We encode and normalize the category descriptions and candidate concepts using $E _ { s } ( \cdot )$ as

$$
\mathbf { u } _ { j } = \frac { E _ { s } ( \phi ( c _ { j } ) ) } { \| E _ { s } ( \phi ( c _ { j } ) ) \| _ { 2 } } , \quad \mathbf { v } _ { j , i } = \frac { E _ { s } ( z _ { j , i } ) } { \| E _ { s } ( z _ { j , i } ) \| _ { 2 } } , \quad i \in \{ 1 , \ldots , N \} ,\tag{2}
$$

where $\mathbf { u } _ { j } \in \mathbb { R } ^ { d _ { t } }$ is the normalized embedding of the templated description of category $c _ { j }$ , and $\mathbf { v } _ { j , i } \in \mathbb { R } ^ { d _ { t } }$ is the normalized embedding of the i-th candidate concept $z _ { j , i }$ of category $c _ { j }$ . Since all vectors are L2-normalized, cosine similarity can be directly computed by inner products in the shared semantic embedding space. For each candidate concept $z _ { j , i }$ , we compute its similarity to the target category and its maximum similarity to the other categories as

$$
a _ { j , i } = \mathbf { v } _ { j , i } ^ { \mathrm { T } } \mathbf { u } _ { j } , \quad \boldsymbol { \eta } _ { j , i } = \operatorname* { m a x } _ { \mathbf { u } \in \mathbf { U } _ { - j } } \mathbf { v } _ { j , i } ^ { \mathrm { T } } \mathbf { u } ,\tag{3}
$$

where $a _ { j , i }$ denotes the similarity between $z _ { j , i }$ and the corresponding target category $c _ { j } , \eta _ { j , i }$ denotes its maximum similarity to other privacy-risk categories, and $\mathbf { U } _ { - j }$ contains the normalized embeddings of all privacy-risk categories except $c _ { j }$

The discriminability score is then defined as

$$
q _ { j , i } = a _ { j , i } - \eta _ { j , i } .\tag{4}
$$

A larger $q _ { j , i }$ indicates a larger semantic similarity margin between the target category and the most similar non-target category. We rank the candidates of category $c _ { j }$ by $q _ { j , i }$ in descending order. The top M concepts are then retained to form the high-quality candidate set for category $c _ { j }$ , defined as

$$
\begin{array} { r } { Z _ { j } ^ { \mathrm { t o p } } = \{ z _ { j , i } \mid i \in \mathcal { I } _ { j } ^ { M } \} , \quad \mathcal { Z } _ { j } ^ { M } = \mathrm { T o p I d x } _ { M } \left( \{ q _ { j , i } \} _ { i = 1 } ^ { N } \right) . } \end{array}\tag{5}
$$

The top M candidates are retained for diversity-based selection. We set $M = \lceil 1 . 5 K _ { c } \rceil$ , where $K _ { c }$ denotes the final number of concepts retained per category.

## 3.1.3 Clustering-based Selection

The retained candidates are discriminative but may still be semantically redundant. Therefore, we apply K-Means to the normalized embeddings of $Z _ { j } ^ { \mathrm { t o p } }$

The embeddings of the retained M candidate concepts are organized as

$$
\begin{array} { r } { \mathbf { V } _ { j } = \left[ \begin{array} { c } { \mathbf { v } _ { j , i _ { 1 } } ^ { \mathrm { T } } } \\ { \mathbf { v } _ { j , i _ { 2 } } ^ { \mathrm { T } } } \\ { \vdots } \\ { \mathbf { v } _ { j , i _ { M } } ^ { \mathrm { T } } } \end{array} \right] \in \mathbb { R } ^ { M \times d _ { t } } , \quad i _ { m } \in \mathcal { I } _ { j } ^ { M } . } \end{array}\tag{6}
$$

For each privacy-risk category, we set the number of K-Means clusters to $K _ { c }$ to partition the retained candidates into semantically distinct groups, yielding

$$
\mathcal { G } _ { 1 } , \mathcal { G } _ { 2 } , \ldots , \mathcal { G } _ { K _ { c } } = \operatorname { K M e a n s } ( \mathbf { V } _ { j } , K _ { c } ) .\tag{7}
$$

For each cluster $\mathcal { G } _ { r }$ , we select the concept nearest to its center $\pmb { \mu } _ { r }$ as

$$
z _ { j , r } ^ { * } = \underset { z _ { j , i } \in \mathcal { G } _ { r } } { \arg \operatorname* { m i n } } \left\| \mathbf { v } _ { j , i } - \pmb { \mu } _ { r } \right\| _ { 2 } , \quad r \in \{ 1 , \dots , K _ { c } \} .\tag{8}
$$

The selected cluster representatives form the final concept set for category $c _ { j }$ , with one representative retained from each semantic cluster, given by

$$
Z _ { j } ^ { * } = \{ z _ { j , 1 } ^ { * } , z _ { j , 2 } ^ { * } , . . . , z _ { j , K _ { c } } ^ { * } \} .\tag{9}
$$

By applying this selection procedure to all privacy-risk categories, we obtain the privacy concept bank $Z ^ { \ast }$ by concatenating the concept sets from all categories to construct a unified category-aware semantic concept space as

$$
Z ^ { * } = \mathrm { C o n c a t } \left( Z _ { 1 } ^ { * } , Z _ { 2 } ^ { * } , \ldots , Z _ { L } ^ { * } \right) .\tag{10}
$$

The final privacy concept bank therefore contains a total of $K = L K _ { c }$ concept entries, with $K _ { c }$ concepts retained for each of the L privacy-risk categories for subsequent image–concept matching and prediction.

## 3.2 Global-Region Concept Modeling

## 3.2.1 Global and Regional Visual Features

After the visual projection layer, the Long-CLIP visual encoder provides a class-token (CLS) representation and patch-token representations. The CLS representation captures image-level visual information, while the patch-token representations retain spatially localized features. We use this CLS representation as the global image feature and apply L2 normalization as

$$
\bar { \mathbf { f } } _ { g } = E _ { v } ^ { \mathrm { C L S } } ( I ) , \qquad \mathbf { f } _ { g } = \frac { \bar { \mathbf { f } } _ { g } } { \| \bar { \mathbf { f } } _ { g } \| _ { 2 } } ,\tag{11}
$$

where $\mathbf { f } _ { g } \in \mathbb { R } ^ { d }$ denotes the L2-normalized global image representation.

We denote the projected and L2-normalized patch-token features as

$$
\mathcal { H } = \{ \mathbf { h } _ { t } \} _ { t = 1 } ^ { G _ { h } G _ { w } } ,\tag{12}
$$

where $\mathbf { h } _ { t } \in \mathbb { R } ^ { d }$ , and $G _ { h }$ and $G _ { w }$ denote the numbers of patch tokens along the vertical and horizontal dimensions, respectively. The $G _ { h } G _ { w }$ patch tokens are arranged according to their spatial positions to form a $G _ { h } \times G _ { w }$ feature map.

$$
\mathcal { H } _ { \operatorname* { m a p } } \in \mathbb { R } ^ { G _ { h } \times G _ { w } \times d } .\tag{13}
$$

To obtain regional representations, adaptive average pooling transforms the $G _ { h } \times G _ { w }$ patchtoken feature map into an $R _ { h } \times R _ { w }$ regional grid, where $R _ { h }$ and $R _ { w }$ denote the numbers of regional bins along the vertical and horizontal dimensions, respectively, and $R = R _ { h } R _ { w }$ denotes the total number of regions. Each regional bin aggregates patch-token features from its corresponding spatial area, providing localized representations for subsequent concept matching.

For the r-th region, $r \in \{ 1 , \ldots , R \}$ , Ω denotes the set of patch-token indices assigned to the corresponding pooling bin. The regional feature is computed as

$$
\bar { \mathbf { f } } _ { \mathrm { r e g } } ^ { r } = \frac { 1 } { \left| \Omega _ { r } \right| } \sum _ { t \in \Omega _ { r } } \mathbf { h } _ { t } , \qquad \mathbf { f } _ { \mathrm { r e g } } ^ { r } = \frac { \bar { \mathbf { f } } _ { \mathrm { r e g } } ^ { r } } { \left\| \bar { \mathbf { f } } _ { \mathrm { r e g } } ^ { r } \right\| _ { 2 } } .\tag{14}
$$

## 3.2.2 Concept Responses and Softplus Mean Aggregation

Before training, the selected concept set $Z ^ { * }$ is encoded once by the frozen Long-CLIP text encoder to obtain the concept embedding matrix

$$
\mathbf { T } = E _ { t } ( Z ^ { * } ) = [ \mathbf { t } _ { 1 } , \mathbf { t } _ { 2 } , \ldots , \mathbf { t } _ { K } ] ^ { \top } \in \mathbb { R } ^ { K \times d } ,\tag{15}
$$

where $\mathbf { t } _ { n } \in \mathbb { R } ^ { d }$ denotes the text embedding of the n-th concept, $n \in \{ 1 , \ldots , K \}$ . Since the concept embeddings are precomputed before training, T remains fixed throughout training and subsequent privacy prediction.

The global response of the n-th concept is computed by cosine similarity as

$$
\begin{array} { r } { m _ { g , n } = \cos ( \mathbf { f } _ { g } , \mathbf { t } _ { n } ) , \qquad \mathbf { m } _ { g } = [ m _ { g , 1 } , m _ { g , 2 } , \dots , m _ { g , K } ] . } \end{array}\tag{16}
$$

For the region branch, we compute the cosine similarity between each regional feature and each concept to measure region-specific concept responses as

$$
m _ { \mathrm { r e g } , n } ^ { r } = \cos ( \mathbf { f } _ { \mathrm { r e g } } ^ { r } , \mathbf { t } _ { n } ) ,\tag{17}
$$

where $m _ { \mathrm { r e g } , n } ^ { r }$ denotes the cosine similarity between the r-th regional visual feature $\mathbf { f } _ { \mathrm { r e g } } ^ { r }$ and the n-th concept embedding $\mathbf { t } _ { n } .$ , thereby quantifying how strongly the n-th concept responds within the r-th region for privacy recognition.

Since privacy cues may appear only in a few image regions, negative regional responses may cancel positive responses during aggregation. To reduce such cancellation, we apply a softplus transformation followed by mean pooling as

$$
m _ { \mathrm { r e g } , n } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \tau \log \left( 1 + \exp \left( \frac { m _ { \mathrm { r e g } , n } ^ { r } } { \tau } \right) \right) ,\tag{18}
$$

where $\tau$ is the temperature parameter controlling the softplus transformation.

The aggregated regional concept responses are then collected into

$$
{ \bf m } _ { \mathrm { r e g } } = [ m _ { \mathrm { r e g } , 1 } , m _ { \mathrm { r e g } , 2 } , \dots , m _ { \mathrm { r e g } , K } ] .\tag{19}
$$

Here, $\mathbf { m } _ { g }$ and $\mathbf { m } _ { \mathrm { r e g } }$ denote global and regional concept responses.

## 3.2.3 Concept-to-Class Mapping and Fusion

Although generated category-wise, all selected privacy-related concepts are combined into a unified concept bank, whose responses are mapped to class scores using a shared learnable concept-to-class association matrix defined as

$$
\mathbf { A } \in \mathbb { R } ^ { L \times K } .\tag{20}
$$

The association weights $A _ { j , n }$ characterize the direction and strength of concept–category associations, while the signed contribution of each concept to individual predictions also depends on its response and the scaling parameter.

The shared association matrix maps the global and regional concept responses to class scores for all privacy-risk categories simultaneously as

$$
\begin{array} { r } { { \bf s } _ { g } = \lambda { \bf m } _ { g } { \bf A } ^ { \mathrm { T } } , \qquad { \bf s } _ { \mathrm { r e g } } = \lambda { \bf m } _ { \mathrm { r e g } } { \bf A } ^ { \mathrm { T } } , } \end{array}\tag{21}
$$

where λ is a learnable scaling parameter applied to the class scores, initialized to ln(50), and jointly optimized with the other trainable parameters. Sharing A ensures consistent conceptto-category semantics across both branches.

The final class logit is obtained by category-specific fusion as

$$
s _ { j } = ( 1 - \alpha _ { j } ) s _ { \mathrm { r e g } , j } + \alpha _ { j } s _ { g , j } + b _ { j } ,\tag{22}
$$

where $\alpha _ { j } \in [ 0 , 1 ]$ controls the contribution of the global branch for category $c _ { j }$ , and $b _ { j }$ is a learnable category bias initialized to the fixed constant −1.5 and subsequently optimized during training. A larger $\alpha _ { j }$ indicates stronger reliance on global context, while a smaller value emphasizes regional evidence.

Each class logit is converted into a probability for multi-label prediction as

$$
p _ { j } = \sigma ( s _ { j } ) .\tag{23}
$$

The binary prediction for each privacy-risk category $c _ { j }$ is then defined as

$$
\hat { y } _ { j } = \left\{ { \begin{array} { l l } { 1 , } & { p _ { j } > \gamma _ { j } , } \\ { 0 , } & { p _ { j } \leq \gamma _ { j } , } \end{array} } \right.\tag{24}
$$

where $\gamma _ { j }$ is the category-specific threshold for $c _ { j }$ , selected on the validation set through categorywise calibration and fixed for test evaluation.

## 3.3 Inference and Calibration

Fusion coeficients are periodically updated category-wise on the validation set during training, while the final decision thresholds are selected on the validation set before test evaluation. For each privacy-risk category $c _ { j }$ , we evaluate a set of candidate fusion coeficients. Given a candidate coeficient $\alpha ,$ , the provisional bias-free probability for the current category is computed as

$$
p _ { j } ^ { ( \alpha ) } = \sigma \left( ( 1 - \alpha ) s _ { \mathrm { r e g } , j } + \alpha s _ { g , j } \right) .\tag{25}
$$

For each candidate $\alpha ,$ an inner decision threshold on the validation set is selected by maximizing the balanced accuracy (BA)–F1 score, defined as

$$
S _ { \mathrm { B A - F 1 } } = 0 . 5 \mathrm { B A } + 0 . 5 \mathrm { F 1 } .\tag{26}
$$

Candidate fusion coeficients are evaluated using the AP–F1 score, defined as

$$
S _ { \mathrm { A P - F 1 } } = 0 . 5 \mathrm { A P } + 0 . 5 \mathrm { F } 1 ,\tag{27}
$$

where AP is computed from the continuous prediction probabilities and F1 is evaluated using the corresponding optimal inner threshold. The fusion coeficient with the highest $S _ { \mathrm { A P - F 1 } }$ is selected as the category-specific coeficient $\alpha _ { j }$

After determining $\alpha _ { j }$ , the learned category bias $b _ { j }$ is added to the fused logit as defined above. The final decision threshold $\gamma _ { j }$ is then selected on the validation set by maximizing $S _ { \mathrm { B A - F 1 } }$ . The selected $\alpha _ { j }$ and $\gamma _ { j }$ are fixed during test evaluation. The prediction for each privacy-risk category is given by

$$
\hat { y } _ { j } = \mathbf { 1 } \left[ p _ { j } > \gamma _ { j } \right] .\tag{28}
$$

Safe/Public does not correspond to an independent concept-based classifier and is treated as a residual category. It is assigned only when none of the privacy-risk categories is activated during inference for each image, as expressed by

$$
\begin{array} { r } { \hat { y } _ { \mathrm { s a f e } } = 1 \quad \mathrm { i f } \quad \forall c _ { j } \in C _ { p } , p _ { j } \leq \gamma _ { j } . } \end{array}\tag{29}
$$

Thus, Safe/Public has neither an independent decision threshold nor a separate concept bank and is predicted only when all privacy-risk predictions are zero.

## 3.4 Training Objective

During training, we optimize PriCBMs with Asymmetric Loss (ASL) [27], which is designed for imbalanced multi-label classification by reducing the dominance of easy negative examples. For sample i and privacy-risk category j, let $y _ { i , j } \in \{ 0 , 1 \}$ denote the ground-truth label and $p _ { i , j } = \sigma ( s _ { i , j } )$ denote the predicted probability. Following ASL, we define $p _ { i , j } ^ { + } = p _ { i , j } , p _ { i , j } ^ { - } = 1 { - } p _ { i , j }$ and apply probability clipping to negative labels as $\bar { p } _ { i , j } ^ { - } = \operatorname* { m i n } ( p _ { i , j } ^ { - } + m , 1 )$

The ASL objective is defined as

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { A S L } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { L } \Bigl [ - y _ { i , j } ( 1 - p _ { i , j } ^ { + } ) ^ { \gamma _ { + } } \log ( p _ { i , j } ^ { + } ) } } \\ & { } & { \qquad - ( 1 - y _ { i , j } ) ( 1 - \bar { p } _ { i , j } ^ { - } ) ^ { \gamma _ { - } } \log ( \bar { p } _ { i , j } ^ { - } ) \Bigr ] . } \end{array}\tag{30}
$$

We further regularize the trainable Long-CLIP visual encoder by preserving its L2-normalized global image representation relative to that produced by a frozen reference visual encoder during optimization to reduce representation drift:

$$
\mathcal { L } _ { \mathrm { i m g } } = 1 - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \cos \Bigl ( E _ { v } ( I _ { i } ) , E _ { v } ^ { 0 } ( I _ { i } ) \Bigr ) ,\tag{31}
$$

where B is the batch size, while $E _ { v } ( I _ { i } )$ and $E _ { v } ^ { 0 } ( I _ { i } )$ denote the L2-normalized global image representations produced by the trainable and frozen Long-CLIP visual encoders, respectively, with their similarity measured using cosine similarity.

The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { A S L } } + \beta _ { I } \mathcal { L } _ { \mathrm { i m g } } . } \end{array}\tag{32}
$$

The Long-CLIP text encoder remains frozen throughout training. Safe/Public samples are used as negative samples for all privacy-risk categories, and no independent Safe/Public classifier or Safe/Public concept loss is introduced.

## 4 Experiments

## 4.1 Datasets and Label Construction

We evaluate PriCBMs on two multi-label visual privacy recognition datasets, VISPR and FineViP [28]. VISPR contains 68 attributes, including 67 privacy-risk attributes and one Safe attribute, which is assigned when no privacy-risk attribute is present. We use a fixed 7:2:1 training/validation/test split for all VISPR experiments, including reproduced baselines and ablation studies, to ensure consistent comparisons. VISPR is released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license, while individual images retain their original licenses.

FineViP provides 22 High- and Low-risk labels corresponding to 11 privacy categories. We follow the oficial splits and retain retrieved images from provided URLs. Since we focus on identifying privacy categories rather than distinguishing risk levels, we merge the High- and Low-risk labels of each category as

$$
y _ { c } = y _ { c } ^ { \mathrm { h i g h } } \vee y _ { c } ^ { \mathrm { l o w } } ,\tag{33}
$$

where $y _ { c } ^ { \mathrm { h i g h } }$ and $y _ { c } ^ { \mathrm { l o w } }$ denote the original high- and low-risk labels, respectively. The resulting label space contains 11 privacy-risk categories and one Public category, treated as Safe/Public in our formulation. The FineViP annotations, taxonomy, and related resources are released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license, while individual images remain subject to their original licenses and terms.

## 4.2 Baselines and Comparison Groups

We organize the comparison experiments into four groups. First, we compare PriCBMs with representative multi-label visual privacy recognition methods, including PrivAttNetMLC [6] and PSSA [22]. PrivAttNetMLC employs an attention-based architecture to capture privacy-relevant visual regions and predict multiple privacy attributes within an image. PSSA introduces privacyspecific spatial attention to extract category-dependent visual features, allowing diferent privacy categories to focus on distinct regions of the same image.

Second, we compare PriCBMs with representative general multi-label classification methods, including Two-Way Multi-Label Loss (TWL) [29], MMBipolar [30], and Bipolar-based methods [31], including BipolarBCE, BipolarFocal, BipolarTWL, and BipolarSM. We also include BCE and Focal Loss [32] as standard loss-based baselines. Following the experimental settings of TWL [29] and BipolarFocal [31], we adopt the same four visual backbones, namely ResNet50 and ResNet101 [33], ResNeXt [34], and RegNet-Y-32gf [35], to evaluate BCE, Focal, TWL, MMBipolar, and the Bipolar-based methods, thereby maintaining a consistent backbone setting across these baselines.

Third, we compare PriCBMs with representative multimodal large language models (MLLMs), which are grouped according to how they are accessed during evaluation. The locally deployed models include Qwen2.5-VL [36], Qwen3.5 [37], and DeepSeek-VL2 [38], while the API-accessed models include GPT-5.5 [39], Gemini-3.6-Flash [40], and Kimi K3 [41].

Fourth, since binary private/public classification remains common in visual privacy recognition, we additionally evaluate a task-specific binary variant, termed PriCBMs-Binary, for broader comparison. PriCBMs-Binary is trained independently under the Safe/Public-versus-Privacy label space rather than obtained by directly collapsing the original multi-label predictions.

Since the multi-label baselines use diferent visual backbones and pretraining schemes, the comparison is intended to reflect representative performance under commonly used settings rather than a strictly backbone-controlled comparison. To isolate the efect of the proposed formulation, we further provide controlled comparisons under the same Long-CLIP backbone in Section 4.5.3.

## 4.3 Implementation and Training Details

We adopt Long-CLIP with ViT-L/14 as the unified vision-language encoder and initialize it from the pretrained longclip-L model. Images are resized to 224 × 224 using the default CLIP preprocessing. The text encoder remains frozen throughout training, while the visual encoder is trainable.

For the regional branch, adaptive average pooling produces a $3 \times 3$ regional grid, resulting in $R = 9$ regions. Regional concept responses are aggregated using the softplus mean with a temperature of $\tau = 0 . 0 7$ . For each privacy-risk category, we remove duplicate and invalid outputs and retain $N = 1 0 0$ valid candidate concepts. From this candidate pool, we select $K _ { c } = 3 0$ concepts through semantic discriminability and diversity selection for subsequent modeling.

We train PriCBMs using AdamW for 100 epochs. The learning rate is set to $2 \times 1 0 ^ { - 5 }$ for the visual encoder and $5 \times 1 0 ^ { - 5 }$ for the remaining trainable parameters, with an efective batch size of 32. We use Asymmetric Loss for multi-label classification and set the image feature regularization weight to $\beta _ { I } = 0 . 0 1$

All model configurations are selected using the validation set and fixed before test evaluation. Specifically, the validation set is used to determine the category-specific fusion coeficients and decision thresholds, as well as the number of retained concepts per category and the number of image regions. The test set is used only for performance reporting and is not involved in hyperparameter selection. Test-set sensitivity analyses for the number of retained concepts and the number of regions are provided in appendix A. These analyses are reported only for post-hoc robustness evaluation and are not used for model selection.

## 4.4 Evaluation Metrics

Following the standard evaluation protocol for multi-label learning [42], we report MAP-class and MAP-sample to evaluate ranking performance, together with Micro-F1 and Macro-F1 to evaluate classification performance.

Specifically, MAP-class computes the average precision for each category and then averages over categories, while MAP-sample computes the average precision for each image and then averages over samples. For all evaluated methods, MAP-class and MAP-sample are computed over the complete label space, including the residual Safe/Public category. For ranking-based evaluation, Safe/Public is assigned as the ground-truth label when no privacy-risk category is present, and its ranking score, derived from privacy-category probabilities, is defined as

$$
p _ { \mathrm { s a f e } } = 1 - \operatorname* { m a x } _ { c _ { j } \in C _ { p } } p _ { j } .\tag{34}
$$

After the decision thresholds are applied, Micro-F1 and Macro-F1 are also computed over the complete label space, including the Safe/Public category. Micro-F1 aggregates true positives, false positives, and false negatives over all labels, while Macro-F1 averages the F1 scores across labels and therefore better reflects class-balanced performance. Unless otherwise specified, all metric values reported in the tables are presented in percentage form to facilitate comparison.

For the additional binary classification experiments with PriCBMs-Binary, we report accuracy, balanced accuracy, and Macro-F1 as overall evaluation metrics. We also report class-wise precision, recall, and F1-score for the Public and Privacy classes to provide a more detailed assessment under class imbalance.

## 4.5 Results

## 4.5.1 Multi-label Classification Comparison

As shown in Tables 1 and 2, PriCBMs achieve the best performance across all four metrics on both VISPR and FineViP. The gains are particularly pronounced in MAP-class and Macro-F1. On VISPR, PriCBMs reach 67.28% MAP-class and 63.51% Macro-F1, outperforming the strongest baselines by 11.72 and 22.43 percentage points, respectively. On FineViP, the corresponding improvements are 10.46 and 12.39 percentage points. These consistent gains across both datasets indicate improved category-level prediction, as reflected in MAP-class and Macro-F1. The higher Macro-F1 indicates improved average performance across all categories, including Safe/Public, under equal category weighting.

PriCBMs also obtain the best MAP-sample and Micro-F1 results. The improvements are 1.34 and 2.02 percentage points on VISPR, compared with 5.70 and 10.33 percentage points on FineViP. This diference may be related to the larger label space of VISPR, which contains 68 categories compared with 12 in FineViP. A larger label space may increase the dificulty of sample-level ranking and joint multi-label prediction, which may explain why the performance gains on VISPR are more pronounced in category-level metrics.

PriCBMs further outperform the privacy-oriented PrivAttNetMLC and PSSA across all four metrics. Unlike these attention-based methods, PriCBMs explicitly model global and regional privacy concept responses and combine them through category-specific fusion. This allows complementary evidence from diferent spatial levels to be represented explicitly in the concept space. The results show that PriCBMs achieve better multi-label classification performance than the evaluated privacy-oriented baselines under the reported settings.

Table 1: Comparison results on the VISPR dataset.
<table><tr><td>Backbone</td><td>Method</td><td>MAP-c</td><td>MAP-s</td><td>Mic-F1</td><td>Mac-F1</td></tr><tr><td rowspan="7">ResNet50</td><td>BCE</td><td>49.54</td><td>85.36</td><td>78.20</td><td>34.80</td></tr><tr><td>Focal</td><td>48.84</td><td>84.71</td><td>77.39</td><td>35.96</td></tr><tr><td>TWL</td><td>50.02</td><td>81.08</td><td>66.34</td><td>38.24</td></tr><tr><td>MMBipolar</td><td>53.51</td><td>84.39</td><td>78.96</td><td>36.13</td></tr><tr><td>BipolarBCE</td><td>51.39</td><td>84.86</td><td>78.59</td><td>38.91</td></tr><tr><td>BipolarFocal</td><td>50.45</td><td>83.86</td><td>77.77</td><td>39.90</td></tr><tr><td>BipolarTWL</td><td>47.81</td><td>84.16</td><td>32.44</td><td>18.84</td></tr><tr><td>BipolarSM</td><td>50.32</td><td>84.14</td><td>78.34</td><td>31.78</td></tr><tr><td rowspan="8">ResNet101</td><td>BCE</td><td>50.03</td><td>85.31</td><td>77.91</td><td>35.53</td></tr><tr><td>Focal</td><td>49.32</td><td>84.49</td><td>77.92</td><td></td></tr><tr><td>TWL</td><td>49.42</td><td>80.16</td><td></td><td>36.54</td></tr><tr><td>MMBipolar</td><td>52.12</td><td>83.60</td><td>65.81 78.91</td><td>38.05</td></tr><tr><td>BipolarBCE</td><td>51.19</td><td>84.95</td><td>78.77</td><td>39.80 39.64</td></tr><tr><td>BipolarFocal</td><td>50.18</td><td>84.36</td><td>77.97</td><td>41.08</td></tr><tr><td>BipolarTWL</td><td>48.74</td><td>83.00</td><td>32.84</td><td>19.08</td></tr><tr><td>BipolarSM</td><td>49.04</td><td>83.97</td><td>78.31</td><td>32.18</td></tr><tr><td rowspan="8">ResNeXt</td><td>BCE</td><td>50.08</td><td>85.39</td><td>78.30</td><td></td></tr><tr><td>Focal</td><td>48.25</td><td>84.55</td><td>77.68</td><td>34.98</td></tr><tr><td>TWL</td><td>50.84</td><td>81.68</td><td>68.61</td><td>34.68</td></tr><tr><td>MMBipolar</td><td>53.98</td><td>85.30</td><td>78.57</td><td>39.62 37.19</td></tr><tr><td>BipolarBCE</td><td>51.88</td><td>85.25</td><td>78.37</td><td>37.56</td></tr><tr><td>BipolarFocal</td><td>50.77</td><td>85.23</td><td>77.44</td><td>38.70</td></tr><tr><td>BipolarTWL</td><td>49.02</td><td>84.49</td><td>33.20</td><td>18.87</td></tr><tr><td>BipolarSM</td><td>50.24</td><td>83.92</td><td>78.42</td><td>33.18</td></tr><tr><td rowspan="8">RegNet-Y-32gf</td><td>BCE</td><td>49.17</td><td>85.87</td><td>79.20</td><td>34.27</td></tr><tr><td>Focal</td><td>49.83</td><td>85.88</td><td>78.17</td><td>34.45</td></tr><tr><td>TWL</td><td>55.56</td><td>82.62</td><td>51.19</td><td>29.46</td></tr><tr><td>MMBipolar</td><td>54.69</td><td>85.88</td><td>80.09</td><td>39.55</td></tr><tr><td>BipolarBCE</td><td>51.30</td><td>85.96</td><td>79.50</td><td>38.24</td></tr><tr><td>BipolarFocal</td><td>50.77</td><td>85.67</td><td>78.86</td><td>39.64</td></tr><tr><td>BipolarTWL</td><td>52.58</td><td>85.04</td><td>33.24</td><td>19.61</td></tr><tr><td>BipolarSM</td><td>50.40</td><td>84.43</td><td>79.02</td><td>33.42</td></tr><tr><td colspan="2">PSSA</td><td>33.91</td><td>78.82</td><td>71.80</td><td>23.42</td></tr><tr><td colspan="2">PrivAttNetMLC</td><td>39.69</td><td>84.90</td><td>74.85</td><td>25.83</td></tr><tr><td colspan="2">PriCBMs</td><td>67.28</td><td>87.30</td><td>82.11</td><td>63.51</td></tr></table>

MAP-c/s: MAP-class/sample; Mic-F1/Mac-F1: Micro-/Macro-F1. Best results are in bold.

Table 2: Comparison results on the FineViP dataset.
<table><tr><td>Backbone</td><td>Method</td><td>MAP-c</td><td>MAP-s</td><td>Mic-F1</td><td>Mac-F1</td></tr><tr><td rowspan="8">ResNet50</td><td>BCE</td><td>67.46</td><td>80.69</td><td>68.40</td><td>62.03</td></tr><tr><td>Focal</td><td>66.90</td><td>80.26</td><td>67.66</td><td>61.71</td></tr><tr><td>TWL</td><td>66.96</td><td>81.08</td><td>60.41</td><td>57.29</td></tr><tr><td>MMBipolar</td><td>65.11</td><td>79.22</td><td>66.54</td><td>59.98</td></tr><tr><td>BipolarBCE</td><td>67.38</td><td>81.07</td><td>67.91</td><td>62.34</td></tr><tr><td>BipolarFocal</td><td>67.70</td><td>81.20</td><td>68.11</td><td>62.01</td></tr><tr><td>BipolarTWL</td><td>67.09</td><td>81.21</td><td>60.31</td><td>57.71</td></tr><tr><td>BipolarSM</td><td>66.17</td><td>80.50</td><td>68.31</td><td>62.29</td></tr><tr><td rowspan="8">ResNet101</td><td>BCE</td><td>68.50</td><td>81.23</td><td>68.80</td><td>62.86</td></tr><tr><td>Focal</td><td>68.28</td><td>80.65</td><td>68.25</td><td>62.35</td></tr><tr><td>TWL</td><td>68.46</td><td>82.15</td><td>60.20</td><td>57.73</td></tr><tr><td>MMBipolar</td><td>62.29</td><td>79.12</td><td>67.94</td><td>62.10</td></tr><tr><td>BipolarBCE</td><td>69.45</td><td>81.51</td><td>68.76</td><td>63.92</td></tr><tr><td>BipolarFocal</td><td>68.48</td><td>81.12</td><td>68.14</td><td>62.15</td></tr><tr><td>BipolarTWL</td><td>68.19</td><td>81.37</td><td>59.15</td><td>56.65</td></tr><tr><td>BipolarSM</td><td>67.14</td><td>81.01</td><td>68.75</td><td>62.75</td></tr><tr><td rowspan="8">ResNeXt</td><td>BCE</td><td>67.79</td><td>80.38</td><td>67.31</td><td></td></tr><tr><td>Focal</td><td>67.63</td><td>80.58</td><td>67.87</td><td>62.25</td></tr><tr><td>TWL</td><td>67.39</td><td>80.79</td><td>58.05</td><td>62.47 55.36</td></tr><tr><td>MMBipolar</td><td>66.69</td><td>80.09</td><td>68.16</td><td>61.88</td></tr><tr><td>BipolarBCE</td><td>68.55</td><td>80.98</td><td>68.51</td><td>63.00</td></tr><tr><td>BipolarFocal</td><td>69.30</td><td>81.35</td><td>67.85</td><td>62.53</td></tr><tr><td>BipolarTWL</td><td>67.15</td><td>81.10</td><td>57.65</td><td>55.03</td></tr><tr><td>BipolarSM</td><td>67.17</td><td>80.87</td><td>68.97</td><td>62.83</td></tr><tr><td rowspan="8">RegNet-Y-32gf</td><td>BCE</td><td>71.87</td><td>83.24</td><td>71.40</td><td></td></tr><tr><td>Focal</td><td>72.25</td><td>82.86</td><td>71.05</td><td>67.03 66.30</td></tr><tr><td>TWL</td><td>71.47</td><td>83.51</td><td>60.95</td><td>58.60</td></tr><tr><td>MMBipolar</td><td>67.64</td><td>82.23</td><td>71.69</td><td>66.96</td></tr><tr><td>BipolarBCE</td><td>72.00</td><td>83.01</td><td>71.57</td><td>66.86</td></tr><tr><td>BipolarFocal</td><td>71.70</td><td>82.72</td><td>70.64</td><td>65.62</td></tr><tr><td>BipolarTWL</td><td>71.83</td><td>83.06</td><td>61.99</td><td>59.67</td></tr><tr><td>BipolarSM</td><td>69.78</td><td>83.10</td><td>72.00</td><td>66.94</td></tr><tr><td>PSSA</td><td></td><td>41.85</td><td>65.67</td><td>37.26</td><td>28.86</td></tr><tr><td colspan="2">PrivAttNetMLC</td><td>62.89</td><td>76.48</td><td>59.12</td><td>53.84</td></tr><tr><td colspan="2">PriCBMs</td><td>82.71</td><td>89.21</td><td>82.33</td><td>79.42</td></tr></table>

MAP-c/s: MAP-class/sample; Mic-F1/Mac-F1: Micro-/Macro-F1. Best results are in bold.

## 4.5.2 Comparison with Multimodal Large Language Models

We compare PriCBMs with multimodal large language models (MLLMs) to examine whether they can directly perform multi-label visual privacy recognition. Each MLLM is prompted to identify all privacy-related categories present in the image from the predefined label space and return the corresponding multi-label predictions. If a model refuses to answer, the sample is counted as a refusal case; otherwise, its returned labels are evaluated. Because refusal reduces prediction coverage and afects the subset on which classification performance is evaluated, we report refusal rates and conditional classification performance computed on non-refused samples as separate outcomes for clearer interpretation.

Table 3: Comparison with open-source multimodal large language models.
<table><tr><td>Data</td><td>Model</td><td>Mic-F1</td><td>Mac-F1</td></tr><tr><td rowspan="5">VISPR</td><td>Q2.5-32B</td><td>43.20</td><td>40.64</td></tr><tr><td>Q2.5-72B</td><td>39.11</td><td>40.71</td></tr><tr><td>Q3.5-27B</td><td>41.75</td><td>22.07</td></tr><tr><td>DSVL2</td><td>17.51</td><td>6.71</td></tr><tr><td>PriCBMs</td><td>82.11</td><td>63.51</td></tr><tr><td rowspan="5">FineViP</td><td>Q2.5-32B</td><td>41.84</td><td>39.23</td></tr><tr><td>Q2.5-72B</td><td>42.55</td><td>42.53</td></tr><tr><td>Q3.5-27B</td><td>44.39</td><td>45.19</td></tr><tr><td>DSVL2</td><td>19.03</td><td>18.51</td></tr><tr><td>PriCBMs</td><td>82.33</td><td>79.42</td></tr></table>

Q2.5-32B, Q2.5-72B, and Q3.5-27B denote Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct, and Qwen3.5-27B, respectively. DSVL2 denotes DeepSeek-VL2.

Table 4: Comparison with API-accessed multimodal large language models.
<table><tr><td>Data</td><td>Model</td><td>Mic-F1</td><td>Mac-F1</td><td>Refusal (%)</td></tr><tr><td rowspan="4">VISPR</td><td>GPT-5.5</td><td>82.71</td><td>10.38</td><td>74.15</td></tr><tr><td>Gemini-3.6-Flash</td><td></td><td>一</td><td>92.20</td></tr><tr><td>Kimi K3</td><td>85.20</td><td>8.54</td><td>80.56</td></tr><tr><td>PriCBMs</td><td>82.11</td><td>63.51</td><td>0.00</td></tr><tr><td rowspan="4">FineViP</td><td>GPT-5.5</td><td>62.27</td><td>49.13</td><td>65.41</td></tr><tr><td>Gemini-3.6-Flash</td><td>一</td><td>一</td><td>99.21</td></tr><tr><td>Kimi K3</td><td>一</td><td>一</td><td>96.42</td></tr><tr><td>PriCBMs</td><td>82.33</td><td>79.42</td><td>0.00</td></tr></table>

“–” indicates that classification metrics are omitted when the refusal rate exceeds 90%. For APIaccessed MLLMs, Micro-F1 and Macro-F1 are computed only on the non-refused subset. For Refusal, lower is better.

For open-source MLLMs, the comparison results are shown in Table 3. These models provide predictions for the full test set and are therefore evaluated under the full-coverage setting for direct comparison with the proposed method.

As shown in Table 3, PriCBMs consistently outperform the evaluated open-source generalpurpose MLLMs on both VISPR and FineViP under the full-coverage setting. On VISPR, PriCBMs achieve 82.11% Micro-F1 and 63.51% Macro-F1, while the strongest open-source MLLMs achieve 43.20% Micro-F1 and 40.71% Macro-F1, respectively. On FineViP, PriCBMs achieve 82.33% Micro-F1 and 79.42% Macro-F1, substantially outperforming the evaluated opensource MLLMs on both metrics. These results indicate that, under the evaluated prompting setup, the tested open-source MLLMs are less efective than the proposed task-specific conceptbased framework on VISPR and FineViP.

For API-accessed MLLMs, the comparison results are shown in Table 4. Unlike the locally deployed models, which provide predictions for all images in our evaluation, API-accessed MLLMs may refuse to answer privacy-related queries, potentially due to built-in safety mechanisms.

As shown in Table 4, API-accessed MLLMs exhibit high refusal rates. GPT-5.5 refuses 74.15% of VISPR samples and 65.41% of FineViP samples, while Kimi K3 refuses 80.56% and 96.42% of samples on the two datasets, respectively. Gemini-3.6-Flash produces even fewer valid predictions, with refusal rates of 92.20% on VISPR and 99.21% on FineViP. These results show that API-accessed MLLMs have limited prediction coverage in visual privacy recognition.

The Micro-F1 and Macro-F1 scores of API-accessed MLLMs should therefore be interpreted as conditional performance on the non-refused subset rather than full-coverage performance. These scores are computed only on samples for which valid predictions are returned and are therefore not directly comparable with the full-coverage performance of PriCBMs under the same evaluation setting. The current analysis does not establish how refusal changes the class composition of the evaluated subset. Therefore, high conditional performance on valid predictions does not necessarily indicate reliable recognition across the full test set.

## 4.5.3 Efect of Concept Bottleneck and Global–Region Modeling

To examine how the models relate to the two challenges, we compare four model formulations on VISPR. L-CLIP-Linear and L-CLIP-MLP directly predict privacy labels from global Long-CLIP visual features using a linear classifier and a multi-layer perceptron, respectively, without explicit concept-based intermediate representations. In contrast, L-CLIP-CBMs introduce privacy-related concepts between visual features and privacy labels, establishing an explicit image–concept–category process while relying solely on global image features. PriCBMs further extend this formulation by incorporating region-level concept responses and category-specific global-region fusion to model complementary privacy evidence. For a fair comparison, all four formulations use the same VISPR split, Long-CLIP backbone, training protocol, and validationbased threshold calibration procedure, with Safe/Public treated as a residual category during inference. The results of this controlled comparison across all four metrics are reported in Table 5.

The comparison between direct image-to-label classifiers and L-CLIP-CBMs examines the efect of introducing privacy-related concepts as intermediate representations. As shown in Table 5, L-CLIP-CBMs achieve 67.05% MAP-class and 56.80% Macro-F1, surpassing L-CLIP-Linear by 0.55 and 12.29 percentage points, and L-CLIP-MLP by 18.05 and 25.77 percentage points, respectively. These results suggest that replacing direct image-to-label prediction with a concept-mediated process, in which the model first identifies category-relevant privacy cues through visual–concept matching and then predicts privacy categories from the resulting concept responses, can improve classification performance. This finding is consistent with our research motivation.

Table 5: Comparison of diferent model formulations on VISPR.
<table><tr><td>Model</td><td>MAP-c</td><td>MAP-s</td><td>Mic-F1</td><td>Mac-F1</td></tr><tr><td>L-CLIP-Linear</td><td>66.50</td><td>86.89</td><td>65.61</td><td>44.51</td></tr><tr><td>L-CLIP-MLP</td><td>49.00</td><td>86.07</td><td>43.00</td><td>31.03</td></tr><tr><td>L-CLIP-CBMs</td><td>67.05</td><td>87.05</td><td>77.00</td><td>56.80</td></tr><tr><td>PriCBMs</td><td>67.28</td><td>87.30</td><td>82.11</td><td>63.51</td></tr></table>

L-CLIP denotes Long-CLIP. L-CLIP-CBMs denote the global-only concept bottleneck baseline without global-region modeling, whereas PriCBMs denote the full global-region model. All four formulations use the same Safe/Public residual-category treatment during inference.

The comparison between L-CLIP-CBMs and PriCBMs further examines the benefit of incorporating region-level privacy evidence beyond global concept representations. PriCBMs increase Micro-F1 from 77.00% to 82.11% and Macro-F1 from 56.80% to 63.51%, while also improving MAP-class from 67.05% to 67.28% and MAP-sample from 87.05% to 87.30%. These improvements suggest that integrating global- and region-level concept responses provides complementary information for multi-label privacy prediction. The results are consistent with our motivation to complement global representations with localized privacy evidence.

## 4.5.4 Efect of the Vision-Language Encoder

We evaluate the efect of vision-language encoders by using Long-CLIP, CLIP, and LLM2CLIP [43] in PriCBMs while keeping other model components unchanged. This comparison focuses on diferences in overall multi-label privacy recognition performance among the three encoder configurations within the same PriCBMs framework. The results are shown in Fig. 3.

Long-CLIP achieves the best performance across all four metrics on VISPR, with 67.28% MAP-class, 87.30% MAP-sample, 82.11% Micro-F1, and 63.51% Macro-F1. Compared with the second-best result on each metric, Long-CLIP improves these metrics by 7.27, 2.28, 3.92, and 3.21 percentage points, respectively.

![](images/6cc056ec2f01555678c66b9299220300741c0ea4e34133cc75b950d6105d3cf3.jpg)  
Figure 3: Performance comparison of diferent vision-language encoders on VISPR.

## 4.5.5 Efect of Concept Selection

Concept selection builds a compact and interpretable semantic bottleneck for privacy prediction. To assess its two stages, we compare PriCBMs with three ablated variants, each starting from the same 100 candidate concepts and retaining 30 per category. The full method selects the top 45 by semantic discriminability and uses K-Means to retain 30. “w/o Semantic Discriminability Selection” applies K-Means directly to all 100 candidates; “w/o Clustering-based Selection” retains the top 30 by discriminability; and “w/o Concept Selection” randomly selects 30. The results are reported in Fig. 4 for a direct comparison.

As shown in Fig. 4, our selection method achieves the best results on three out of four metrics and remains competitive on MAP-class. A further concept-level quality comparison is provided for additional analysis in Section 5.2.

![](images/160ed9d93c1845d1d532fe6b3b21047247948da4fb9335a08043ca10dcba05f1.jpg)  
Figure 4: Ablation results of the concept selection strategy on VISPR.

## 5 Visualization and Interpretability Analysis

## 5.1 Visualization of Selected Concepts

To qualitatively assess the selected concepts, we select 12 images from VISPR [1] covering privacy-related scenarios involving textual and document information, facial and appearance cues, interpersonal interactions, contextual and group activities, and sensitive visual content. We visualize image-concept associations through the top three concepts ranked by combined global and regional responses. Due to space limitations, Fig. 5 shows the first four examples, with the full set provided in appendix C. For the sampled images, the selected concepts are generally consistent with the visible image content and privacy-related semantics, supporting their qualitative semantic relevance on these examples.

![](images/5eeb91c88c4e6514257526c32acb596d7ef1d3039636d648352939f97b113234.jpg)  
Figure 5: Visualization of selected concepts for the first four representative VISPR images [1]. The top three are ranked by combined global and regional responses.

## 5.2 Quantitative Concept Quality Evaluation

In addition to qualitative visualization, we conduct a quantitative concept-level evaluation to compare the semantic quality of the generated concepts on the sampled examples. We use the same sampled images considered in Section 5.1. For each sampled image, the top three concepts from each method are evaluated. The five compared methods are LaBo [44], w/o Semantic Discriminability Selection, $\mathrm { w / o }$ Clustering-based Selection, $\mathrm { w / o }$ Concept Selection, and PriCBMs.

Each concept is rated on a five-point scale, where higher scores indicate better semantic correctness, visual grounding, and explanatory specificity. The scoring rubric is provided in Appendix D. To reduce reliance on a single evaluator, we use three vision-language model evaluators, GPT-5.5, Gemini-3.6-Flash, and Claude-Opus-5 [45]. For a robust assessment, the Concept Quality Score (CQS) is defined as the average score across all evaluated concepts and evaluators, given by

$$
\mathrm { C Q S } = \frac { 1 } { | \mathcal C | E } \sum _ { c \in \mathcal C } \sum _ { e = 1 } ^ { E } q _ { e } ( c ) ,\tag{35}
$$

where $E = 3 ,$ C denotes evaluated image–concept pairs per method across the twelve sampled images, and $q _ { e } ( c )$ denotes the score assigned to pair c by the e-th evaluator according to the predefined five-point scoring rubric.

The results are reported in Table 6, with concept-level scores provided in Appendix D. PriCBMs achieve the highest CQS of 3.71 and obtain the highest average score under each of the three evaluators. Specifically, PriCBMs obtain average scores of 3.61, 4.03, and 3.50 from GPT-5.5, Gemini-3.6-Flash, and Claude-Opus-5, respectively. Under this model-based evaluation protocol, PriCBMs receive higher average concept-quality scores than LaBo and the evaluated ablation variants on the sampled examples for overall concept quality.

Table 6: Quantitative concept quality evaluation results.
<table><tr><td>Method</td><td>GPT-5.5</td><td>G-3.6-F</td><td>C-Opus-5</td><td>CQS (↑)</td></tr><tr><td>LaBo</td><td>2.89</td><td>3.64</td><td>2.61</td><td>3.05</td></tr><tr><td>w/o SDS</td><td>3.17</td><td>3.75</td><td>2.75</td><td>3.22</td></tr><tr><td>w/o CBS</td><td>3.19</td><td>3.64</td><td>2.94</td><td>3.26</td></tr><tr><td>w/o CS</td><td>3.11</td><td>3.64</td><td>2.81</td><td>3.19</td></tr><tr><td>PriCBMs</td><td>3.61</td><td>4.03</td><td>3.50</td><td>3.71</td></tr></table>

SDS denotes Semantic Discriminability Selection; CBS denotes Clustering-based Selection; CS denotes Concept Selection; G-3.6-F denotes Gemini-3.6-Flash; C-Opus-5 denotes Claude-Opus-5.

## 6 Conclusion

This paper introduced PriCBMs for multi-label visual privacy recognition, replacing direct image-to-label prediction with an explicit concept bottleneck that guides the model to first identify category-relevant privacy cues through visual–concept matching and then predict privacy categories from the resulting concept responses. By constructing a concept bank based on target-category relevance, cross-category discriminability, and semantic diversity, PriCBMs provide category-relevant semantic representations for privacy prediction. In addition, the globalregion modeling strategy combines whole-image and region-level concept responses to represent privacy-related information across spatial granularities.

Experiments on VISPR and FineViP demonstrate that PriCBMs outperform the evaluated baselines across all four multi-label metrics on both datasets under the reported settings. The ablation studies support the efectiveness of concept selection and global-region modeling, while the concept quality analysis and visualizations provide evidence of the semantic quality and relevance of the selected concepts on the 12 sampled images under the model-based evaluation protocol. These results suggest that concept-mediated modeling ofers a promising way to improve multi-label visual privacy recognition while providing an explicit semantic interface for concept-level interpretation.

PriCBMs still have two main limitations. First, the current regional branch relies on fixed grid-based pooling to construct region-level visual representations. Such predefined regions may not always align well with the actual boundaries of privacy-sensitive objects, particularly for small or irregularly distributed visual cues. Future work will explore object-aware regional modeling using detection or segmentation to obtain more semantically meaningful regions for concept response computation. Second, the generalization of PriCBMs across diferent datasets and visual domains has not yet been fully verified. Future work will conduct cross-dataset and domain-shift evaluations and explore domain-robust concept modeling to improve generalization across diferent visual domains.

## Data availability

Both datasets are publicly available: VISPR at https://tribhuvanesh.github.io/vpa/ and FineViP at https://github.com/aiprivacygroup-member1/FineViP.

## Funding

This work was supported by the National Natural Science Foundation of China (Grant No.   
62266035).

The funder had no role in study design, data collection and analysis, manuscript preparation, or the decision to submit the article for publication.

## Competing interests

The authors declare no competing interests.

## Use of AI tools

During the preparation of this manuscript, the authors used ChatGPT (OpenAI) to improve the language and readability of the text. After using this tool, the authors reviewed, verified, and edited the content as needed. The authors take full responsibility for the content of the publication.

## Author contributions

Shenglei Pei: Conceptualization, Methodology, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review & editing, Funding acquisition. Haotian Sun: Conceptualization, Methodology, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review & editing. Pengfei Li: Conceptualization, Methodology, Supervision, Writing – review & editing. Bingxu Lu: Conceptualization, Methodology, Supervision, Writing – review & editing. Wenlong Yu: Conceptualization, Methodology, Supervision, Writing review & editing.

## References

[1] T. Orekondy, B. Schiele, M. Fritz, Towards a visual privacy advisor: Understanding and predicting privacy risks in images, in: Proceedings of the IEEE International Conference on Computer Vision, IEEE, 2017, pp. 3686–3695.

[2] R. Zhao, Y. Zhang, T. Wang, W. Wen, Y. Xiang, X. Cao, Visual content privacy protection: A survey, ACM Comput. Surv. 57 (2025) 1–36.

[3] S. Zerr, S. Siersdorfer, J. Hare, E. Demidova, Privacy-aware image classification and search, in: Proceedings of the 35th International ACM SIGIR Conference on Research and Development in Information Retrieval, Association for Computing Machinery, New York, NY, USA, 2012, pp. 35–44. URL: https://doi.org/10.1145/2348283.2348292. doi:10.1145/2348283.2348292.

[4] S. Zerr, S. Siersdorfer, J. Hare, PicAlert!: A system for privacy-aware image classification and retrieval, in: Proceedings of the 21st ACM International Conference on Information and Knowledge Management, Association for Computing Machinery, 2012, pp. 2710–2712.

[5] A. Tonge, C. Caragea, Image privacy prediction using deep neural networks, ACM Trans. Web 14 (2020) 1–32. doi:10.1145/3383283.

[6] Z. Chen, T. Kandappu, V. Subbaraju, PrivAttNet: Predicting privacy risks in images using visual attention, in: 2020 25th International Conference on Pattern Recognition, IEEE, 2021, pp. 10327–10334.

[7] D. Steinmann, W. Stammer, A. Wüst, K. Kersting, Object-centric concept-bottlenecks, in: Advances in Neural Information Processing Systems, volume 38, Curran Associates, Inc., 2025, pp. 68899–68924. URL: https://proceedings.neurips.cc/paper\_files /paper/2025/file/639d992f819c2b40387d4d5170b8ffd7-Paper-Conference.pdf. doi:10.52202/085713-2317.

[8] H. Tang, C. Yuan, Z. Li, J. Tang, Learning attention-guided pyramidal features for few-shot fine-grained recognition, Pattern Recognit. 130 (2022) 108792. doi:10.1016/j.patcog.202 2.108792.

[9] P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, P. Liang, Concept bottleneck models, in: Proceedings of the 37th International Conference on Machine Learn ing, volume 119 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 5338–5348.

[10] L. Tran, D. Kong, H. Jin, J. Liu, Privacy-CNH: A framework to detect photo privacy with convolutional neural network using hierarchical features, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, AAAI Press, 2016.

[11] G. Yang, J. Cao, Z. Chen, J. Guo, J. Li, Graph-based neural networks for explainable image privacy inference, Pattern Recognit. 105 (2020) 107360. doi:10.1016/j.patcog.202 0.107360.

[12] E. Spyromitros-Xioufis, S. Papadopoulos, A. Popescu, Y. Kompatsiaris, Personalized privacy-aware image classification, in: Proceedings of the 2016 ACM International Conference on Multimedia Retrieval, Association for Computing Machinery, 2016, pp. 71–78.

[13] D. Gurari, Q. Li, C. Lin, Y. Zhao, A. Guo, A. Stangl, J. P. Bigham, VizWiz-Priv: A dataset for recognizing the presence and purpose of private visual information in images taken by blind people, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, 2019, pp. 939–948. doi:10.1109/CVPR.2019.00103.

[14] C. Zhao, J. Mangat, S. Koujalgi, A. Squicciarini, C. Caragea, PrivacyAlert: A dataset for image privacy prediction, in: Proceedings of the International AAAI Conference on Web and Social Media, volume 16, AAAI Press, 2022, pp. 1352–1361.

[15] Y. Li, B. Chen, Q. Jiang, T. Chen, MSPP-Net: Fine-grained image privacy identification via multi-stage semantic perception, in: International Conference on Information Security, Springer, 2025, pp. 239–257.

[16] J. Li, P. Li, X. Hu, K. Yu, Learning common and label-specific features for multi-label classification with correlation information, Pattern Recognit. 121 (2022) 108259. doi:10.1 016/j.patcog.2021.108259.

[17] Y. Gao, Z. Zhu, S. Wang, Mixture of coarse and fine-grained prompt tuning for visionlanguage model, Pattern Recognit. 170 (2026) 112074. doi:10.1016/j.patcog.2025.1120 74.

[18] S. Palazzo, F. Murabito, C. Pino, F. Rundo, D. Giordano, M. Shah, C. Spampinato, Exploiting structured high-level knowledge for domain-specific visual classification, Pattern Recognit. 112 (2021) 107806. doi:10.1016/j.patcog.2020.107806.

[19] W. Yu, D. Chen, Q. Wang, Q. Hu, Fine-grained domain generalization with feature structuralization, IEEE Trans. Multimedia 27 (2025) 8510–8524. doi:10.1109/TMM.2025.36077 16.

[20] P. Li, Y. Shao, S. Pei, et al., Wildlife recognition driven by multi-granularity prompts, Journal of Minnan Normal University (Natural Science) 38 (2025) 35–48. doi:10.12457/j .issn.2095-7122.2025.02.003.

[21] Y. Wang, J. Xie, J. Zheng, B. Lu, Y. Bi, Q. Hu, SPARK: Simple and parameter-free knowledge embedding with fuzzy cognitive maps for class incremental learning, IEEE Trans. Fuzzy Syst. 33 (2025) 2513–2525.

[22] H. Jiang, J. Zuo, Y. Lu, Connecting visual data to privacy: Predicting and measuring privacy risks in images, Electronics 14 (2025) 811.

[23] W. Yu, Q. Wang, C. Liu, D. Li, Q. Hu, CoE: Chain-of-explanation via automatic visual concept circuit description and polysemanticity quantification, in: Proceedings of the Computer Vision and Pattern Recognition Conference, IEEE, 2025, pp. 4364–4374.

[24] B. Zhang, P. Zhang, X. Dong, Y. Zang, J. Wang, Long-CLIP: Unlocking the long-text capability of CLIP, in: European Conference on Computer Vision, Springer, 2024, pp. 310–325.

[25] N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using siamese BERTnetworks, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, Association for Computational Linguistics, 2019, pp. 3982–3992. URL: https://aclanthology.org/D19-1410.

[26] OpenAI, Hello GPT-4o, OpenAI, 2024. URL: https://openai.com/index/hello-gpt-4 o/, (accessed 31 July 2026).

[27] T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, L. Zelnik-Manor, Asymmetric loss for multi-label classification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, IEEE, 2021, pp. 82–91.

[28] The FineViP Dataset, The FineViP Dataset [dataset], GitHub repository, 2025. URL: ht tps://github.com/aiprivacygroup-member1/FineViP, (accessed 31 July 2026).

[29] T. Kobayashi, Two-way multi-label loss, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, 2023, pp. 7476–7485.

[30] S. Mishra, S. Sundaram, A memory-free evolving bipolar neural network for eficient multilabel stream learning, in: ICASSP 2023 – 2023 IEEE International Conference on Acoustics, Speech and Signal Processing, IEEE, 2023, pp. 1–5.

[31] S. Mishra, K. Moorthy, S. Sardeshmukh, S. Sundaram, From ranking to exact prediction in multi-label learning with bipolar networks, in: 2025 5th International Conference on Robotics, Automation, and Artificial Intelligence, IEEE, 2025, pp. 109–113.

[32] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, Focal loss for dense object detection, in: Proceedings of the IEEE International Conference on Computer Vision, IEEE, 2017, pp. 2980–2988.

[33] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, IEEE, 2016, pp. 770–778.

[34] S. Xie, R. Girshick, P. Dollár, Z. Tu, K. He, Aggregated residual transformations for deep neural networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, IEEE, 2017, pp. 1492–1500.

[35] I. Radosavovic, R. P. Kosaraju, R. Girshick, K. He, P. Dollár, Designing network design spaces, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, 2020, pp. 10428–10436.

[36] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, J. Lin, Qwen2.5-VL technical report, arXiv preprint arXiv:2502.13923, 2025. URL: https://doi.org/10.48550/arXiv.2502.13923. doi:10.48550/arXiv.2502.13923.

[37] Qwen Team, Qwen3.5: Towards native multimodal agents, Qwen oficial blog, 2026. URL: https://qwen.ai/blog?id=qwen3.5, (accessed 15 September 2026).

[38] Z. Wu, X. Chen, Z. Pan, X. Liu, W. Liu, D. Dai, H. Gao, Y. Ma, C. Wu, B. Wang, Z. Xie, Y. Wu, K. Hu, J. Wang, Y. Sun, Y. Li, Y. Piao, K. Guan, A. Liu, X. Xie, Y. You, K. Dong, X. Yu, H. Zhang, L. Zhao, Y. Wang, C. Ruan, DeepSeek-VL2: Mixture-of-experts visionlanguage models for advanced multimodal understanding, arXiv preprint arXiv:2412.10302, 2024. URL: https://doi.org/10.48550/arXiv.2412.10302. doi:10.48550/arXiv.2412. 10302.

[39] OpenAI, OpenAI models documentation, OpenAI API documentation, 2026. URL: https: //platform.openai.com/docs/models, (accessed 31 July 2026).

[40] Google DeepMind, Gemini API model documentation, Google AI for Developers, 2026. URL: https://ai.google.dev/gemini-api/docs/models, (accessed 31 July 2026).

[41] Kimi Team, et al., Kimi K3: Open frontier intelligence, arXiv preprint arXiv:2607.24653, 2026. URL: https://doi.org/10.48550/arXiv.2607.24653. doi:10.48550/arXiv.2607. 24653.

[42] M.-L. Zhang, Z.-H. Zhou, A review on multi-label learning algorithms, IEEE Trans. Knowl. Data Eng. 26 (2014) 1819–1837. doi:10.1109/TKDE.2013.39.

[43] W. Huang, A. Wu, Y. Yang, X. Luo, Y. Yang, U. Naseem, C. Wang, Q. Dai, X. Dai, D. Chen, C. Luo, L. Qiu, L. Hu, LLM2CLIP: Powerful language model unlocks richer crossmodality representation, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, AAAI Press, 2026. doi:10.1609/aaai.v40i7.37427.

[44] Y. Yang, A. Panagopoulou, S. Zhou, D. Jin, C. Callison-Burch, M. Yatskar, Language in a bottle: Language model guided concept bottlenecks for interpretable image classification, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, 2023, pp. 19187–19197.

[45] Anthropic, Claude models documentation, Anthropic documentation, 2026. URL: https: //docs.anthropic.com/en/docs/about-claude/models, (accessed 31 July 2026).

[46] G. Ayci, A. Özgür, M. Şensoy, P. Yolum, PEAK: Explainable privacy assistant through automated knowledge extraction, arXiv preprint arXiv:2301.02079, 2023. URL: https: //doi.org/10.48550/arXiv.2301.02079. doi:10.48550/arXiv.2301.02079.

[47] A. E. Baia, A. Cavallaro, Image-guided topic modeling for interpretable privacy classification, in: Computer Vision – ECCV 2024 Workshops, Springer, 2025, pp. 200–217. doi:10.1007/978-3-031-92648-8\_13.

[48] D. Baranouskaya, A. Cavallaro, PrivLEX: Detecting legal concepts in images through vision-language models, arXiv preprint arXiv:2601.09449, 2026. URL: https://doi.or g/10.48550/arXiv.2601.09449. doi:10.48550/arXiv.2601.09449.

## A Model Configuration Analysis

## A.1 Efect of the Number of Concepts

Table A.1: Test-set sensitivity analysis with diferent $K _ { c }$ settings on VISPR. The default value $K _ { c } = 3 0$ is selected on the validation set before final test evaluation.
<table><tr><td>Dataset</td><td> $K _ { c }$ </td><td>MAP-c</td><td> $\mathbf { M A P - s }$ </td><td>Mic-F1</td><td>Mac-F1</td></tr><tr><td>VISPR</td><td>1</td><td>66.27</td><td>87.53</td><td>66.08</td><td>46.54</td></tr><tr><td>VISPR</td><td>5</td><td>68.13</td><td>87.81</td><td>75.14</td><td>54.04</td></tr><tr><td>VISPR</td><td>10</td><td>68.15</td><td>87.38</td><td>75.11</td><td>53.30</td></tr><tr><td>VISPR</td><td>30</td><td>67.28</td><td>87.30</td><td>82.11</td><td>63.51</td></tr><tr><td>VISPR</td><td>50</td><td>67.60</td><td>86.09</td><td>74.98</td><td>56.23</td></tr></table>

![](images/ae132100f6ddcef93d64c8fb82254ed0065d9f6ab67f840e1c8b2b304fd411fc.jpg)  
Figure A.1: Test-set performance trends under diferent numbers of retained concepts per category on VISPR.

We evaluate the sensitivity of PriCBMs to the number of retained concepts per class, denoted as $K _ { c } ,$ on VISPR. The default value $K _ { c } = 3 0$ is selected based on the overall validation-set performance across the four evaluation metrics before final test evaluation. After fixing this validation-selected configuration, we report the corresponding test-set results under diferent $K _ { c }$ settings in Table A.1 and Fig. A.1. These results should be interpreted only as additional test-set sensitivity analysis for robustness assessment rather than hyperparameter-selection evidence.

All model configurations used for the final evaluation are selected on the validation set and fixed before test evaluation. In this appendix, we report additional test-set sensitivity analyses to examine how PriCBMs behave under diferent configuration choices in practice. These results are used only for post-hoc robustness analysis to more comprehensively assess stability under alternative settings and are not used for hyperparameter selection.

As shown in Table A.1, diferent values of $K _ { c }$ lead to diferent trade-ofs across evaluation metrics. Smaller concept banks obtain slightly higher MAP-class or MAP-sample, whereas $K _ { c } ~ = ~ 3 0$ achieves the strongest F1-based performance on the test set. This suggests that retaining more concepts can benefit F1-based multi-label prediction, while an excessively large concept bank may introduce redundant or less discriminative concepts. Since the default value $K _ { c } = 3 0$ is selected on the validation set before test evaluation, the results in Table A.1 are reported only as post-hoc sensitivity analysis for additional robustness assessment.

## A.2 Efect of the Number of Regions

Table A.2: Test-set sensitivity analysis with diferent numbers of regions on VISPR. The default setting of 9 regions is selected on the validation set before final test evaluation.
<table><tr><td>Region</td><td>MAP-c</td><td>MAP-s</td><td>Mic-F1</td><td>Mac-F1</td></tr><tr><td>2</td><td>67.60</td><td>87.77</td><td>80.50</td><td>61.71</td></tr><tr><td>4</td><td>68.96</td><td>87.12</td><td>82.03</td><td>63.37</td></tr><tr><td>9</td><td>67.28</td><td>87.30</td><td>82.11</td><td>63.51</td></tr><tr><td>16</td><td>67.58</td><td>87.20</td><td>82.14</td><td>62.36</td></tr><tr><td>24</td><td>67.23</td><td>86.77</td><td>81.47</td><td>61.84</td></tr><tr><td>32</td><td>66.51</td><td>87.26</td><td>81.05</td><td>60.07</td></tr></table>

![](images/6f3a2050fe98eaf489825b6b9e9dde2dd7dae887f469bafaa5867999f9ca362b.jpg)  
Figure A.2: Test-set performance trends under diferent region number settings on VISPR.

We further evaluate the sensitivity of PriCBMs to the number of local regions on VISPR under diferent regional granularity settings. The evaluated settings $R \in \{ 2 , 4 , 9 , 1 6 , 2 4 , 3 2 \}$ correspond to $R _ { h } \times R _ { w }$ grids of $1 \times 2 , 2 \times 2 , 3 \times 3 , 4 \times 4 , 4 \times 6 ,$ and $4 \times 8$ , respectively. The default setting of 9 regions is selected based on the overall validation-set performance across the four evaluation metrics and fixed before final test evaluation. After this validation-based selection, we report the corresponding test-set results under diferent region numbers in Table A.2 and Fig. A.2. These results are used only to analyze the efect of regional granularity and are not used to select the final region configuration. By varying the number of regions, we examine how diferent spatial partitioning granularities afect the balance between localized privacy evidence, overall prediction stability, and the resulting classification performance under diferent region settings.

As shown in Table A.2, performance varies with the number of local regions. Diferent region settings achieve the best results on diferent metrics: 4 regions obtain the highest MAPclass, 2 regions achieve the highest MAP-sample, 16 regions obtain the highest Micro-F1, and 9 regions achieve the highest Macro-F1. This indicates that diferent regional granularities lead to diferent trade-ofs across the evaluation metrics. Using fewer regions provides coarser spatial representations, whereas increasing the number of regions allows privacy evidence to be modeled at a more localized level with finer spatial partitioning. However, increasing the number of regions does not consistently improve classification performance. Since the final setting of 9 regions is selected on the validation set before test evaluation, the test-set results in Table A.2 are interpreted as configuration sensitivity for additional post-hoc analysis rather than model-selection evidence.

## B Binary Classification Comparison

Since most existing privacy classification studies formulate the task as a privacy-sensitive/safe binary classification problem, we construct a task-specific binary variant, termed PriCBMs-Binary, and compare it with representative interpretable methods, including PEAK [46], Priv×ITM [47], and PrivLEX [48]. PriCBMs-Binary is trained independently under the Safe/Public-versus-Privacy label space rather than obtained by collapsing the multi-label predictions. Specifically, PriCBMs-Binary reuses the original privacy concept bank from the multi-label model and introduces a Safe/Public concept set of the same size as the original privacy concept bank. The multi-label concept-to-class head is replaced with a binary head, while the remaining components are unchanged. The results are reported in Table B.1.

As shown in Table B.1, PriCBMs-Binary achieves the highest performance on both VISPR and FineViP in terms of accuracy, balanced accuracy, and Macro-F1. It obtains 91.38%, 91.01%, and 90.86% on VISPR, and 98.34%, 92.22%, and 89.66% on FineViP, respectively. PriCBMs-Binary also achieves the highest F1-scores for both the Safe/Public and Privacy classes on the two datasets, indicating more balanced binary classification performance across the two classes.

Several methods show relatively low Public-class performance on the imbalanced FineViP test split. In the FineViP test split used for binary evaluation, Public accounts for 3.92% of the samples, whereas the remaining 96.08% belong to the Privacy class. Under such a skewed distribution, high overall accuracy can coexist with poor recall for the minority Public class. For example, PEAK achieves 96.11% accuracy but only 1.92% recall for the Public class. Therefore, balanced accuracy and Macro-F1 provide a more informative evaluation than overall accuracy under this imbalanced setting.

Table B.1: Binary classification comparison results on the VISPR and FineViP datasets.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">ACC</td><td rowspan="2">BA</td><td rowspan="2">F1-m</td><td colspan="3">Public/Safe</td><td colspan="3">Privacy</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="4">VISPR</td><td>PEAK</td><td>85.47</td><td>83.58</td><td>84.20</td><td>83.86</td><td>75.96</td><td>79.71</td><td>86.30</td><td>91.20</td><td>88.68</td></tr><tr><td>Priv×ITM</td><td>87.16</td><td>86.33</td><td>86.32</td><td>82.88</td><td>82.97</td><td>82.93</td><td>89.74</td><td>89.69</td><td>89.71</td></tr><tr><td>PrivLEX</td><td>88.84</td><td>88.11</td><td>88.10</td><td>85.14</td><td>85.16</td><td>85.15</td><td>91.07</td><td>91.05</td><td>91.06</td></tr><tr><td>PriCBMs-B</td><td>91.38</td><td>91.01</td><td>90.86</td><td>87.88</td><td>89.44</td><td>88.65</td><td>93.58</td><td>92.57</td><td>93.07</td></tr><tr><td rowspan="4">FineViP</td><td>PEAK</td><td>96.11</td><td>50.94</td><td>50.88</td><td>66.67</td><td>1.92</td><td>3.74</td><td>96.15</td><td>99.96</td><td>98.02</td></tr><tr><td>Priv×ITM</td><td>96.76</td><td>69.72</td><td>73.87</td><td>63.64</td><td>40.38</td><td>49.41</td><td>97.60</td><td>99.06</td><td>98.32</td></tr><tr><td>PrivLEX</td><td>96.72</td><td>64.63</td><td>69.96</td><td>68.89</td><td>29.81</td><td>41.61</td><td>97.20</td><td>99.45</td><td>98.31</td></tr><tr><td>PriCBMs-B</td><td>98.34</td><td>92.22</td><td>89.66</td><td>75.42</td><td>85.58</td><td>80.18</td><td>99.41</td><td>98.86</td><td>99.13</td></tr></table>

PriCBMs-B: PriCBMs-Binary; ACC: accuracy; BA: balanced accuracy; F1-m: Macro-F1; P/R/F1: precision/recall/F1-score.

## C Complete Visualization of Selected Concepts

To demonstrate the interpretability of PriCBMs, we provide the visualization results of selected concepts in Fig. C.1. For the sampled images, the selected concepts are generally consistent with the visible image content and privacy-related semantics, supporting their qualitative semantic relevance on these examples.

Figure C.1: Complete visualization of selected concepts for VISPR images [1]. For each image, the top three are ranked by combined global and regional responses to illustrate image-concept associations.

![](images/96d617f105eb349167d0835197120e076d9a46935a54f86ac5a904e731af1b20.jpg)

![](images/539f5af6f27260791df67565cfb86c9f1a01e6597525764fbbe2c8bafa20d9d6.jpg)

![](images/af031f090a6f8574f448777adc1fab7085143ba246137e94bbce44ce3e205f36.jpg)

![](images/1d30d3757693be22fd98f8863ca4acb1666fc06e026d7964b93ecb2130cbf937.jpg)

## Concept

An image showing a venue entry ticket with a tear-off stub.

A photo of a concert ticket with artist and venue printed prominently.

A cropped view of a ticket stub with tear line and entry barcode.

![](images/bcceddf01c45e68cb703d369179556298e18e2fe9b9abfa5b6e6c3021db6feb7.jpg)

## Concept

An image showing family members serving culturally specific food with traditional utensils.

An image showing two peers leaning together over a restaurant table.

An image showing a person arranging a shawl over a culturally specific dress.

![](images/10ecd8b8acdb11328022d045282b6ccc10ddced20bbc92c0c9f695bb993f9ca9.jpg)

## Concept

A cropped view of a headshot emphasizing facial morphology rather than clothing or setting.

A photo of a portrait-style headshot with the full face exposed.

A photo of a person with a shaved face and collared shirt.

## Concept

A photo of friends gathered in a dim bar with colored lights.

A photo of coworkers at an after-hours bar that resembles a social hangout.

An image showing two peers leaning together over a restaurant table.

![](images/d26478a26f7f6a4720ad62f41150c35db442f017750ad5e63396f8fb9d02dfd3.jpg)

## Concept

A photo of multiple complete faces in a casual outdoor group photo.

An image showing a group photo with overlapping partially visible faces.

A photo of several people whose faces share similar visible appearance cues in a group portrait.

![](images/247502317b7272ca0f8ca4ada727b7edbbb76e7e85a95603a16316ba5ea4717a.jpg)

## Concept

A cropped view of a woman in a camisole and panties.

An image showing a person in a lace bra with visible cleavage.

A photo of a woman wearing a bra and underwear set.

## Concept

A photo of a graduate posed with family immediately after receiving honors.

A photo of three young adults standing shoulder to shoulder.

An image showing partial facial features in a tight portrait crop.

![](images/b8064a867aa3c57197062e4f739b11810758e629f299b3b7337212e101f2792c.jpg)

## Concept

A cropped view of only two smiling people where they could be siblings or friends.

A cropped view of friends laughing while one wraps an arm around another.

An image showing two people gazing at each other and smiling.

![](images/924aa21c6a44ffd5ae391bd8cb738cf41d04513617ba9573e220e2e14dbe5c0c.jpg)

## Concept

An image showing a group lifting protest signs in front of a government building.

An image showing a public rally in a civic plaza with banners.

A photo of people marching while holding issue-based banners.

## Concept

A photo of a sheet of paper covered with handwritten notes.

A photo of a page with handwritten equations and symbols.

An image showing messy handwritten notes rather than clean printed typography.

![](images/2c78c74e84911073f1fc56671049a2487afde4797a9917e9bc37db365797345c.jpg)

## Concept

A photo of audience members with passive posture during a concert.

A photo of an audience watching a performer under stage lights.

A photo of an audience cluster facing a stage.

![](images/c352a33a61c554001a0fcba9a7e708fd70a15a254a7fe5209b53a3be665cafed.jpg)

## Concept

A photo of a passport at a border control desk.

An image showing the word passport in uppercase on a dark cover.

A photo of a passport booklet lying flat on a table.

## D Detailed Concept-level Scores

This appendix provides the detailed concept-level quality evaluation results corresponding to the quantitative analysis in Section 5.2. For each sampled image, the top three concepts selected by each compared method are shown together with the scores assigned by three separately queried large vision-language model evaluators, namely GPT-5.5, Gemini-3.6-Flash, and Claude-Opus-5.

The compared methods include LaBo, w/o Semantic Discriminability Selection, w/o Clustering-based Selection, $\mathrm { w / o }$ Concept Selection, and the full PriCBMs. Each concept is scored on a five-point scale according to the rubric defined in Table D.1. A higher score indicates that the concept is more visually grounded, semantically accurate, and explanatory for the corresponding image.

Table D.1: Scoring rubric for concept quality evaluation.
<table><tr><td>Score</td><td>Meaning</td></tr><tr><td>1</td><td>Clearly wrong, hallucinated, or unrelated to the image.</td></tr><tr><td>2</td><td>Partially related, but with obvious inaccuracies or hallucinated details.</td></tr><tr><td>3</td><td>Basically related, but generic or partially uncertain.</td></tr><tr><td>4</td><td>Relevant, clear, and visually verifiable.</td></tr><tr><td>5</td><td>Highly accurate, specific, and strongly explanatory.</td></tr></table>

The detailed scoring results are shown in Figs. D.1–D.4.

![](images/33d118b5eb6e717f35f97b9053d1421674d4c0ed4b84e8d901d5b4ca4ac0b2b3.jpg)

![](images/5a98ba8034d273157cbb1451b76ae4754e3ab46797fc9bab89d9eec71d6cc65c.jpg)

![](images/7a880a77806fdf8546371a27157ecf995097a7afaeb6de4646cedd55d752ac79.jpg)

<table><tr><td>Concept description</td><td>G M</td><td>C</td><td>Concept description</td><td></td><td>G M</td><td>C</td><td>Concept description</td><td>G M</td><td>C</td><td></td></tr><tr><td>PriCBMs An image showing a venue</td><td>3.67 4.33 4.00</td><td></td><td>PriCBMs</td><td></td><td>3.33 4.00 3.67</td><td></td><td>PriCBMs A photo of a graduate posed</td><td>3.33 3.67 3.00</td><td></td><td></td></tr><tr><td>entry ticket with a tear-off stub.</td><td></td><td>4 4 4</td><td></td><td>A photo of friends gathered in a dim bar with colored lights.</td><td>4 4</td><td>4</td><td>with family immediately after receiving honors.</td><td>3</td><td>3 2</td><td></td></tr><tr><td>A photo of a concert ticket with artist and venue printed prominently.</td><td>3</td><td>4</td><td>4 a social hangout.</td><td>A photo of coworkers at an after-hours bar that resembles 2</td><td></td><td>4 3</td><td>A photo of three young adults standing shoulder to shoulder.</td><td>4</td><td>4 4</td><td></td></tr><tr><td>A cropped view of a ticket stub with tear line and entry barcode.</td><td>4 5 4</td><td></td><td></td><td>An image showing two peers leaning together over a restaurant table.</td><td>4 4</td><td>4</td><td>An image showing partial facial features in a tight portrait crop.</td><td></td><td>3 4 3</td><td></td></tr><tr><td>LaBo A close-up of a venue ticket</td><td>2.67 3.00 2.67</td><td></td><td>LaBo</td><td>A photo of coworkers at an</td><td>3.33 4.00 3.33</td><td></td><td>LaBo</td><td></td><td>3.33 3.67 3.00</td><td></td></tr><tr><td>with row, section, and seat information. A cropped view of an event</td><td>4 4 4</td><td></td><td></td><td>after-hours bar that resembles 2 a social hangout.</td><td>4</td><td>3</td><td>A photo of three young adults standing shoulder to shoulder.</td><td>4</td><td>4 4</td><td></td></tr><tr><td>ticket near a stadium entry gate. An image showing someone</td><td>2 3 2</td><td></td><td>An image showing a</td><td>restaurant table scene with a partially blocked face.</td><td>4</td><td>4 3</td><td>An image showing coworkers standing shoulder to shoulder in a posed portrait.</td><td>2</td><td>3 2</td><td></td></tr><tr><td>gripping an event ticket at the venue entrance.</td><td>2 2 2</td><td></td><td>restaurant table.</td><td>An image showing two peers leaning together over a</td><td>4 4 4</td><td></td><td>A close-up of a trio posing cheek to cheek.</td><td>4</td><td>4 3</td><td></td></tr><tr><td>w/o CBS An image showing a venue</td><td>3.67 4.00 3.67</td><td></td><td>w/o CBS</td><td></td><td>3.67 4.00 3.00</td><td></td><td>w/o CBS A photo of a graduate posed</td><td>3.67 3.67 3.00</td><td></td><td></td></tr><tr><td>entry ticket with a tear-off stub. A close-up of a venue ticket</td><td>4 4 4</td><td></td><td></td><td>A photo of friends gathered in a dim bar with colored lights.</td><td>4 4</td><td>4</td><td>with family immediately after 3 3 2 receiving honors.</td><td></td><td></td><td></td></tr><tr><td>with row, section, and seat information.</td><td>4</td><td>4 4</td><td></td><td>A photo of friends under a glowing bar sign with readable letters behind them.</td><td>3 4 2</td><td></td><td>A photo of three young adults standing shoulder to shoulder.</td><td>4</td><td>4 4</td><td></td></tr><tr><td>A photo of an event ticket with a branded header strip.</td><td>3 4 3</td><td></td><td></td><td>A photo of companions raising drinks together in a casual toast.</td><td>4 3 4</td><td></td><td>A close-up of a trio posing cheek to cheek.</td><td>4</td><td>4 3</td><td></td></tr><tr><td>w/o SDS A photo of a concert ticket</td><td>3.00 4.00 3.33</td><td></td><td>w/o SDS</td><td></td><td>2.67 3.00 2.33</td><td></td><td>w/o SDS</td><td></td><td></td><td>3.67 4.00 2.33</td></tr><tr><td>with artist and venue printed prominently.</td><td>3</td><td>4 4</td><td></td><td>A photo of friends gathered in a dim bar with colored lights.</td><td>4 4</td><td>4</td><td>A photo of a graduate being embraced by family in cap and gown.</td><td></td><td>4 4 1</td><td></td></tr><tr><td>An image showing a ticket stub with fine print and barcode numbers.</td><td>4 5 4</td><td></td><td></td><td>An image showing people toasting at dinner without visible event-specific objects.</td><td>3 3 2</td><td></td><td>A photo of a family of three gathered closely for a portrait.</td><td>3</td><td></td><td>4 3</td></tr><tr><td>A cropped view of an event ticket near a stadium entry gate.</td><td></td><td></td><td>2 3 2</td><td>A photo of a professional serving coffee in a hospitality</td><td></td><td>1</td><td>A close-up of a trio posing cheek to cheek.</td><td>4</td><td></td><td>4 3</td></tr><tr><td>w/o CS</td><td>3.33 4.00 3.33</td><td></td><td></td><td>work role. w/o CS</td><td>2.33 3.33 2.67</td><td></td><td>w/o CS</td><td></td><td></td><td>3.67 3.67 2.33</td></tr><tr><td>A photo of a concert ticket printed on glossy card stock.</td><td>3 4 2</td><td></td><td></td><td>A photo of friends gathered in a dim bar with colored lights.</td><td>4 4</td><td>4</td><td>A photo of a graduate posed with family immediately after 3 3 2 receiving honors.</td><td></td><td></td><td></td></tr><tr><td>An image showing a venue entry ticket with a tear-off stub.</td><td>4</td><td>4</td><td>4</td><td>A photo of coworkers at an after-hours bar that resembles 2 a social hangout.</td><td></td><td>4 3</td><td>A photo of three young adults standing shoulder to shoulder.</td><td>4</td><td>4 4</td><td></td></tr><tr><td>A photo of a concert ticket with the event date and start time.</td><td></td><td>3 4 4</td><td></td><td>An image showing a clearly corporate interior rather than</td><td>1</td><td>2 1</td><td>A photo of a graduate being embraced by family in cap</td><td>4</td><td>4 1</td><td></td></tr></table>

Figure D.1: Detailed concept-level quality scores for sampled VISPR examples [1], part 1. For each image, the top three concepts of diferent methods and their evaluator scores are shown.

![](images/81ae0ae13410a53f98f20dd06617e63a785f6ee4773cc022ffdcb4a6421f85b6.jpg)

![](images/91568436c1cf2bd78d154d0773f1236dc5fc34dba5603392778087a7062b8064.jpg)

![](images/563b1d3b601562c31e90782050cfbb7f1d41863e171252f5e6fde7305d8c2e58.jpg)

<table><tr><td>Concept description</td><td>G M</td><td>C</td><td>Concept description</td><td></td><td>G M</td><td>C</td><td>Concept description</td><td>G M</td><td>C</td><td></td></tr><tr><td>PriCBMs A photo of a sheet of paper</td><td>4.00 4.00 3.67</td><td></td><td>PriCBMs An image showing family</td><td></td><td>3.00 3.67 3.00</td><td></td><td>PriCBMs A photo of multiple complete</td><td></td><td>3.67 4.00 3.67</td><td></td></tr><tr><td>covered with handwritten notes.</td><td>4</td><td>4 4</td><td>utensils.</td><td>members serving culturally specific food with traditional</td><td>3 4 3</td><td></td><td>faces in a casual outdoor group photo.</td><td>4</td><td>4</td><td>4</td></tr><tr><td>A photo of a page with handwritten equations and symbols.</td><td>4</td><td>4 3</td><td>restaurant table.</td><td>An image showing two peers leaning together over a</td><td>3 3 2</td><td></td><td>An image showing a group photo with overlapping partially visible faces. A photo of several people</td><td>4</td><td>4 4</td><td></td></tr><tr><td>An image showing messy handwritten notes rather than clean printed typography.</td><td>4 4</td><td>4</td><td></td><td>An image showing a person arranging a shawl over a culturally specific dress.</td><td>3 4 4</td><td></td><td>whose faces share similar visible appearance cues in a group portrait.</td><td></td><td>3 4 3</td><td></td></tr><tr><td>LaBo An image showing a manila</td><td>2.00 2.33 2.00</td><td></td><td>LaBo</td><td>An image showing customary</td><td>3.00 4.33 3.00</td><td></td><td>LaBo A photo of multiple complete</td><td>3.67 3.67 3.33</td><td></td><td></td></tr><tr><td>envelope with handwritten address fragments. A close-up of handwritten</td><td>2 2 2</td><td></td><td></td><td>serving utensils, food presentation, and celebratory attire in the same scene.</td><td>3 4 2</td><td></td><td>faces in a casual outdoor group photo. An image showing a group</td><td>4</td><td>4</td><td>4</td></tr><tr><td>recipe measurements on an index card.</td><td>2 2 2</td><td></td><td></td><td>An image showing a person adjusting a draped sari over one shoulder.</td><td>4 5</td><td>4</td><td>portrait where only one pair appears slightly closer than the rest.</td><td></td><td>3 3 2</td><td></td></tr><tr><td>A cropped view of a receipt with handwritten marks.</td><td>2 3 2</td><td></td><td></td><td>An image showing a person placing a turban onto the head 2 as part of dress.</td><td></td><td>4 3</td><td>An image showing a group photo with overlapping partially visible faces.</td><td>4</td><td></td><td>4 4</td></tr><tr><td>w/o CBS A photo of a sheet of paper</td><td>3.67 4.00 3.67</td><td></td><td>w/o CBS</td><td>An image showing family</td><td>2.67 4.00 2.33</td><td></td><td>w/o CBS A photo of multiple complete</td><td>3.00 3.33 3.00</td><td></td><td></td></tr><tr><td>covered with handwritten notes.</td><td>4</td><td>4</td><td>4</td><td>members serving culturally specific food with traditional utensils. An image showing customary</td><td>3</td><td>4 3</td><td>faces in a casual outdoor group photo.</td><td>4</td><td>4</td><td>4</td></tr><tr><td>A close-up of a paper form with handwritten entries.</td><td>3 4 3</td><td></td><td></td><td>serving utensils, food presentation, and celebratory attire in the same scene.</td><td>3 4 2</td><td></td><td>An image showing coworkers posing together in a workplace without interacting.</td><td>2 2 2</td><td></td><td></td></tr><tr><td>An image showing messy handwritten notes rather than clean printed typography.</td><td>4</td><td>4</td><td>4</td><td>An image showing a group raising candles during a cultural ritual.</td><td>2 4 2</td><td></td><td>A photo of several people whose faces share similar visible appearance cues in a group portrait.</td><td>3 4 3</td><td></td><td></td></tr><tr><td>w/o SDS A photo of a page with</td><td>3.33 3.33 2.67</td><td></td><td>w/o SDS</td><td>An image showing family</td><td>3.33 4.00 3.00</td><td></td><td>w/o SDS</td><td></td><td></td><td>3.00 3.33 3.00</td></tr><tr><td>handwritten equations and symbols.</td><td>4</td><td>4 3</td><td></td><td>members serving culturally specific food with traditional utensils.</td><td>3 4 3</td><td></td><td>An image showing a group photo with overlapping partially visible faces.</td><td>4</td><td>4</td><td>4</td></tr><tr><td>A close-up of handwritten lines of text across ruled notebook paper.</td><td>4 4 3</td><td></td><td></td><td>An image showing several people using ceremonial objects rather than just posing with colorful clothes.</td><td>3 3 2</td><td></td><td>An image showing coworkers posing together in a workplace without interacting.</td><td>2 2 2</td><td></td><td></td></tr><tr><td>An image showing a whiteboard covered in handwritten marker text.</td><td>2 2 2</td><td></td><td></td><td>A photo of a person wearing a draped sari-like wrap with a</td><td>4</td><td>5 4</td><td>A photo of several people whose faces share similar visible appearance cues in a</td><td></td><td></td><td>3 4 3</td></tr><tr><td>w/o CS</td><td>3.00 3.33 3.00</td><td></td><td></td><td>matching blouse. w/o CS</td><td>3.00 3.67 2.67</td><td></td><td>group portrait. w/o CS</td><td></td><td></td><td>3.67 3.33 3.00</td></tr><tr><td>An image showing a manila envelope with handwritten address fragments.</td><td>2 2 2</td><td></td><td></td><td>An image showing family members serving culturally specific food with traditional utensils.</td><td>3 4 3</td><td></td><td>A photo of multiple complete faces in a casual outdoor group photo.</td><td>4</td><td>4 4</td><td></td></tr><tr><td>A photo of a sheet of paper covered with handwritten notes.</td><td>4</td><td>4 4</td><td></td><td>An image showing stacked bangles and anklets worn with traditional attire.</td><td>3 4 3</td><td></td><td>An image showing a group portrait where only one pair appears slightly closer than the rest.</td><td>3 3 2</td><td></td><td></td></tr><tr><td>An image showing loose sheets with handwritten annotations.</td><td></td><td>3 4 3</td><td></td><td>An image showing several people using ceremonial objects rather than just posing</td><td>3</td><td>3 2</td><td>An image showing a group portrait where one person stands noticeably tallest in the</td><td>4</td><td>3 3</td><td></td></tr></table>

Figure D.2: Detailed concept-level quality scores for sampled VISPR examples [1], part 2. For each image, the top three concepts of diferent methods and their evaluator scores are shown.

![](images/fef30300dc4793241e509c74d9219892dbfe2db9480c7321e2e70eda68f3b3eb.jpg)

![](images/f22cb40e63cc7b4479891f0b52ed54c5aeba39fbe16ada555cb0401902d0520f.jpg)

![](images/06968b5371bc4a1c73ac1aa2e929e0727332bb45293a5d7bda83cd1286f0deb1.jpg)

<table><tr><td>Concept description</td><td>G M C</td><td></td><td>Concept description</td><td></td><td>G M C</td><td></td><td>Concept description</td><td>G M C</td><td></td><td></td></tr><tr><td>PriCBMs A cropped view of only two</td><td>3.67 4.33 3.33</td><td></td><td>PriCBMs A photo of audience members</td><td></td><td>3.67 4.00 4.00</td><td></td><td>PriCBMs A cropped view of a headshot</td><td>3.33 4.00 2.67</td><td></td><td></td></tr><tr><td>smiling people where they could be siblings or friends.</td><td>3 4 3</td><td></td><td>concert.</td><td>with passive posture during a</td><td>3 4</td><td>4</td><td>emphasizing facial morphology rather than clothing or setting.</td><td></td><td>3 4 2</td><td></td></tr><tr><td>A cropped view of friends laughing while one wraps an arm around another.</td><td>4 5</td><td>4</td><td>stage lights.</td><td>A photo of an audience watching a performer under</td><td>4 4 4</td><td></td><td>A photo of a portrait-style headshot with the full face exposed.</td><td></td><td>4 4 3</td><td></td></tr><tr><td>An image showing two people gazing at each other and smiling.</td><td>4 4 3</td><td></td><td></td><td>A photo of an audience cluster facing a stage.</td><td>4 4 4</td><td></td><td>A photo of a person with a shaved face and collared shirt.</td><td></td><td>3 4 3</td><td></td></tr><tr><td>LaBo A cropped view of friends</td><td>3.00 4.33 2.67</td><td></td><td>LaBo</td><td></td><td>2.00 4.33 2.00</td><td></td><td>LaBo A photo of a studio portrait</td><td>3.00 3.67 2.33</td><td></td><td></td></tr><tr><td>laughing while one wraps an arm around another. An image showing two adults</td><td>4 5 4</td><td></td><td></td><td>A photo of people standing in a ring around a performer.</td><td>2 4 2</td><td></td><td>with a plain background and aclearly lit face.</td><td></td><td>4 4 3</td><td></td></tr><tr><td>smiling near each other at an event without contact.</td><td>2 4 2</td><td></td><td>ahead.</td><td>An image showing viewers seated while dancers perform</td><td>2 4 2</td><td></td><td>An image showing a studio headshot with evenly illuminated irises.</td><td></td><td>2 4 2</td><td></td></tr><tr><td>A photo of two smiling adults in office clothing with no visible work materials.</td><td>3 4 2</td><td></td><td>An image showing a musician plucking a</td><td>traditional string instrument</td><td>2 5 2</td><td></td><td>A cropped view of only a smooth face where age is hard to judge.</td><td></td><td></td><td>3 3 2</td></tr><tr><td>w/o CBS</td><td>3.67 4.33 3.67</td><td></td><td>on stage. w/o CBS</td><td></td><td>3.33 3.67 3.67</td><td></td><td>w/o CBS</td><td></td><td></td><td>3.00 3.67 3.00</td></tr><tr><td>A cropped view of only two smiling people where they could be siblings or friends.</td><td>3 4 3</td><td></td><td>concert.</td><td>A photo of audience members with passive posture during a</td><td>3 4</td><td>4</td><td>A photo of a middle-aged adult with faint smile lines and mature facial features.</td><td></td><td></td><td>3 4 4</td></tr><tr><td>A photo of two friends hugging and smiling at the camera.</td><td>4 4 4</td><td></td><td>A photo of an audience stage lights.</td><td>watching a performer under</td><td>4</td><td>4</td><td>A photo of a person with a shaved face and collared shirt.</td><td></td><td></td><td>3 4 3</td></tr><tr><td>A cropped view of friends laughing while one wraps an arm around another.</td><td>4 5</td><td>4</td><td>A photo of bystanders</td><td>focused on performers instead 3 3 3</td><td></td><td></td><td>A photo of a young adult with no gray hair and minimal</td><td></td><td></td><td>3 3 2</td></tr><tr><td>w/o SDS</td><td>3.33 4.33 3.33</td><td></td><td>of interacting with them. w/o SDS</td><td></td><td>3.33 4.33 3.00</td><td></td><td>facial lines. w/o SDS</td><td></td><td></td><td>3.33 3.67 2.67</td></tr><tr><td>An image showing two adults smiling near each other at an event without contact.</td><td>2 4 2</td><td></td><td></td><td>A photo of an audience cluster facing a stage.</td><td>4 4</td><td>4</td><td>A photo of a person with a shaved face and collared shirt.</td><td></td><td></td><td>3 4 3</td></tr><tr><td>A cropped view of friends laughing while one wraps an arm around another.</td><td>4 5 4</td><td></td><td></td><td>An image showing a musician plucking a traditional string instrument</td><td>2 5 2</td><td></td><td>A cropped view of a head and shoulders portrait with exposed forehead and cheeks.</td><td></td><td></td><td>4 4 3</td></tr><tr><td>A photo of two people leaning together for the camera.</td><td>4 4 4</td><td></td><td>on stage.</td><td>A photo of an audience in a dim concert hall.</td><td>4 4 3</td><td></td><td>A cropped view of only a smooth face where age is</td><td></td><td></td><td>3 3 2</td></tr><tr><td>w/o CS</td><td>3.67 4.33 3.67</td><td></td><td>w/o CS</td><td></td><td>3.00 4.00 3.33</td><td></td><td>hard to judge. w/o CS</td><td></td><td></td><td>3.00 4.00 3.00</td></tr><tr><td>A cropped view of only two smiling people where they could be siblings or friends.</td><td>3 4 3</td><td></td><td></td><td>A photo of audience members with passive posture during a concert.</td><td>3 4</td><td>4</td><td>A photo of a middle-aged adult with faint smile lines and mature facial features.</td><td></td><td></td><td>3 4 4</td></tr><tr><td>An image showing two people smiling while one hand rests on the other&#x27;s</td><td>4 5 4</td><td></td><td></td><td>A photo of an audience cluster facing a stage.</td><td>4</td><td>4 4</td><td>A photo of a studio portrait with a plain background</td><td></td><td>4 4 3</td><td></td></tr><tr><td>shoulder. A photo of two friends hugging and smiling at the</td><td>4 4 4</td><td></td><td></td><td>A photo of people standing in</td><td></td><td></td><td>and aclearly lit face. An image showing a studio</td><td></td><td></td><td></td></tr></table>

Figure D.3: Detailed concept-level quality scores for sampled VISPR examples [1], part 3. For each image, the top three concepts of diferent methods and their evaluator scores are shown.

![](images/371837183fbec4d7249723382204d952c31e3029ceaa09e336f77fcb0933888d.jpg)  
Image P4-C1

![](images/939b161b400f330c056d0310234b2dfdeabd3129a09020a6b3024331ac725c91.jpg)  
Image P4-C2

![](images/2bad0da5c457387e270ace19cf7a49e6c9cca3eea262c9c0db74063424fd8107.jpg)  
Image P4-C3

<table><tr><td>Concept description</td><td>G</td><td>M</td><td>C</td><td>Concept description</td><td></td><td>G M</td><td>C</td><td>Concept description</td><td>G</td><td>M</td><td>C</td><td></td></tr><tr><td>PriCBMs</td><td>4.33 4.33 4.00</td><td></td><td></td><td>PriCBMs</td><td></td><td>3.33 4.00 3.33</td><td></td><td>PriCBMs</td><td></td><td>4.00 4.00 3.67</td><td></td><td></td></tr><tr><td>A cropped view of a woman in a camisole and panties.</td><td>4</td><td>4</td><td>4</td><td>An image showing a group lifting protest signs in front of a government building.</td><td>3</td><td>4</td><td>3</td><td>A photo of a passport at a border control desk.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>An image showing a person in a lace bra with visible cleavage.</td><td>4</td><td>5</td><td>4</td><td>An image showing a public rally in a civic plaza with banners.</td><td>3</td><td>4</td><td>3</td><td>An image showing the word passport in uppercase on a dark cover.</td><td>5</td><td>5</td><td>5</td><td></td></tr><tr><td>A photo of a woman wearing a bra and underwear set.</td><td>5</td><td>4</td><td>4</td><td>A photo of people marching while holding issue-based banners.</td><td>4</td><td>4</td><td>4</td><td>A photo of a passport booklet lying flat on a table.</td><td>4</td><td>4</td><td>4</td><td></td></tr><tr><td>LaBo</td><td></td><td></td><td>2.33 3.33 1.67</td><td>LaB0</td><td></td><td></td><td>3.33 4.00 2.67</td><td>LaBo</td><td></td><td>3.00 3.00 2.67</td><td></td><td></td></tr><tr><td>A close-up of a swimsuit model in a studio photo setting.</td><td>3</td><td>4</td><td>2</td><td>An image showing a street protest filling an urban intersection.</td><td>4</td><td>4</td><td>3</td><td>A close-up of a passport in a travel flat-lay with tickets and sunglasses.</td><td>2</td><td>2</td><td>2</td><td></td></tr><tr><td>An image showing a fully unclothed person in a softly lit studio backdrop.</td><td>1</td><td>2</td><td>1</td><td>An image showing a sidewalk protest outside a courthouse.</td><td>3</td><td>4</td><td>2</td><td>A photo of a passport on an airport check-in counter.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>A photo of a fashion editorial with a revealing bodysuit that looks like</td><td>3</td><td>4</td><td>2</td><td>An image showing a public rally in a civic plaza with banners.</td><td>3</td><td>4</td><td>3</td><td>A close-up of a dark blue passport booklet cover.</td><td>4</td><td>4</td><td>4</td><td></td></tr><tr><td>lingerie. w/o CBS</td><td></td><td></td><td>2.33 2.67 2.00</td><td>w/o CBS</td><td></td><td></td><td>3.00 3.67 2.67</td><td>w/o CBS</td><td></td><td></td><td>2.67 2.67 1.67</td><td></td></tr><tr><td>An image showing a full-body nude pose with no clothing present.</td><td>1</td><td>2</td><td>1</td><td>A photo of bystanders watching a public demonstration from the</td><td>3</td><td>4</td><td>2</td><td>A photo of a passport at a border control desk.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>A photo of a nude person indoors against plain walls and floor for posing.</td><td>1</td><td>2</td><td></td><td>sidewalk. A photo of a street protest filled with advocacy signs.</td><td>4</td><td>4</td><td>4</td><td>A close-up of a passport beside a boarding pass and suitcase handle.</td><td>2</td><td>2</td><td>1</td><td></td></tr><tr><td>A photo of a woman wearing a bra and underwear set.</td><td>5</td><td></td><td></td><td>An image showing a displayed statement of support or opposition</td><td>2</td><td>3</td><td>2</td><td>A photo of a passport on an airport check-in counter.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>w/o SDS</td><td>2.67 3.67 2.00</td><td></td><td></td><td>on an everyday object. w/o SDS</td><td></td><td>3.67 4.00 3.33</td><td></td><td>w/o SDS</td><td></td><td></td><td>2.67 3.33 2.00</td><td></td></tr><tr><td>A close-up of a swimsuit model in a studio photo setting.</td><td>3</td><td>4</td><td>2</td><td>An image showing a group lifting protest signs in front of a</td><td>3</td><td>4</td><td>3</td><td>An image showing a passport on a desk with travel accessories.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>An image showing a fully unclothed person in a softly lit studio backdrop.</td><td>1</td><td></td><td></td><td>government building. An image showing a public square packed with demonstrators and</td><td></td><td>4</td><td>3</td><td>A photo of a passport at a border control desk.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>An image showing a person in panties and a cropped top exposing the stomach.</td><td>4</td><td>5</td><td></td><td>banners. A photo of a street protest filled with advocacy signs.</td><td>4</td><td>4</td><td>4</td><td>An image showing a passport</td><td>2</td><td>4</td><td>2</td><td></td></tr><tr><td>w/o CS</td><td></td><td>1.67 2.67 1.33</td><td></td><td>w/o CS</td><td></td><td></td><td>3.67 4.00 2.67</td><td>w/o CS</td><td>opened to pages with visa stamps.</td><td></td><td>3.33 3.33 2.67</td><td></td></tr><tr><td>An image showing a full-body nude pose with no clothing present.</td><td>1</td><td>2</td><td>1</td><td>An image showing a sidewalk demonstration outside a public building.</td><td></td><td>4</td><td>3</td><td>A photo of a passport at a border control desk.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>A close-up of a swimsuit model in a studio photo setting.</td><td>3</td><td>4</td><td>2</td><td>An image showing a sidewalk protest outside a courthouse.</td><td>3</td><td>4</td><td>2</td><td>A photo of a passport on an airport check-in counter.</td><td>3</td><td>3</td><td>2</td><td></td></tr><tr><td>A photo of a nude person indoors against plain walls and floor for posing.</td><td>1</td><td>2</td><td>1</td><td>An image showing a public square packed with demonstrators and</td><td></td><td></td><td></td><td></td><td>A photo of a passport booklet lying</td><td></td><td></td><td></td></tr></table>

Shaded rows average the three concepts per method and evaluator. G = GPT-5.5; M = Gemini-3.6-Flash; C = Claude-Opus-5.

Figure D.4: Detailed concept-level quality scores for sampled VISPR examples [1], part 4. For each image, the top three concepts of diferent methods and their evaluator scores are shown.