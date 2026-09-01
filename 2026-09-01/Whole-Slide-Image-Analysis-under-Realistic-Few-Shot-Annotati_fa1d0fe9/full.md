# Whole-Slide Image Analysis under Realistic Few-Shot Annotation Protocols

Tifanie Godelaine , Maxime Zanella , Karim El Khoury Member, IEEE, Benoˆıt Macq Fellow, IEEE, and Christophe De Vleeschouwer Member, IEEE

## Abstract

Automating the analysis of whole-slide images has high clinical value, since characterizing cancers requires examining them in detail. Such analysis increasingly relies on vision-language models that provide patch-level zero-shot predictions. However, these predictions remain noisy and must be refined with a few annotations. A promising paradigm for this refinement is few-shot transduction. Rather than treating each patch independently, these meth ods leverage the relations between patches, together with a few annotations, to refine all predictions jointly. However, current transductive methods are evaluated under conditions that overlook key properties of whole-slide images: (i) datasets consist of independent patches extracted from multiple slides, ignoring the complex tissue organization; (ii) datasets are mostly balanced, whereas a single whole-slide image exhibits severe class imbalance, with several classes absent; and (iii) annotations are sampled at random, without reflecting how a pathologist annotates a limited number of regions. To align the transduction paradigm to realistic whole-slide settings, we introduce the following contributions. First, we propose SlideCRF, which adapts conditional random fields for whole-slide images by combining spatial and biological cues while accounting for classes that may be absent from a given slide. Second, we provide a set of realistic annotation protocols, based on spatially localized clicks and scribbles, modeling diferent pathologist interactions, such as the iterative correction of model errors. Across four datasets, we show that Slide-CRF outperforms current transductive methods in macro F1, improving over the zero-shot predictions by +24.2% and +37.5% with one and 16 clicks per present class, respectively.

Keywords: Whole-slide image analysis, conditional random fields, weak annotations, vision-language models

## 1 Introduction

Accurate analysis of whole-slide images (WSIs) is crucial for cancer diagnosis and staging, enabling pathologists to characterize complex histopathological entities. Automating this pipeline ofers substantial clinical value by reducing workload while enhancing diagnostic capabilities [1]. However, given the gigapixel scale of WSIs, they must first be divided into independent, non-overlapping patches to make processing computationally feasible.

To analyze these massive patch collections, recent frameworks have increasingly relied on histology-specific vision-language models (VLMs) [2–4]. Pre-trained on massive histological image-text pairs, these models exhibit strong generalization capabilities that enable patchlevel zero-shot (ZS) predictions without requiring prior annotations from pathologists. Yet, despite their promising performance, these initial ZS predictions often remain noisy, still necessitating a limited number of expert annotations for refinement [5].

In histology, the most commonly used approaches to refine these VLM predictions are inductive few-shot methods [6]. These methods typically rely on few annotations to train a lightweight linear classification layer or optimize a small set of adapter parameters on top of a frozen VLM backbone, and generate an independent prediction for each patch [7–9]. In contrast, a second and promising direction involves transductive few-shot methods [10, 11]. Rather than treating each patch independently, these techniques jointly refine all predictions by leveraging both expert annotations and the underlying relationships between patches [12, 13]. Conditional random fields (CRFs) fit into this paradigm: by modeling pairwise dependencies between patches, they enforce consistency across neighboring predictions while refining all patches at once [14].

Despite their conceptual advantages in modeling patch dependencies, current state-of-the-art transductive meth ods, such as Histo-TransCLIP [11] and HistoCRF [14], are evaluated under conditions that diverge sharply from realistic clinical settings. This discrepancy manifests in three main ways: (i) standard benchmarks use independent patches pooled from distinct WSIs, discarding the critical spatial context necessary to characterize complex tissue structures; (ii) these benchmarks rely on mostly balanced datasets, overlooking the severe intra-slide class imbalance and frequently absent diagnostic categories inherent to WSIs; and (iii) existing methods assume that expert annotations are sampled randomly across the entire slide. This ignores actual pathological workflows, where clinicians typically annotate a limited number of localized, contiguous regions rather than a sparse, widespread set of patches. Consequently, because existing transductive methods were not designed with these constraints in mind, their performance remains severely limited when evaluated under realistic clinical conditions.

To align transductive learning with realistic clinical settings, we introduce the following key contributions:

• We propose SlideCRF: a method to refine the patchlevel ZS predictions of a WSI using a few expert annotations by modeling the relations between patches with a CRF. To handle the realistic WSI setting, our method integrates spatial and biological relations, recovering finely structured regions while remaining robust to the absence of classes within a slide. As shown in Figure 1, by iteratively annotating a few regions, SlideCRF progressively refines the ZS predictions and recovers the underlying structures of the slide.

• We provide a set of realistic annotation protocols based on spatially localized clicks and scribbles, rather than widespread annotations, better reflecting how a pathologist interacts with a WSI. To capture realistic human-in-the-loop (HITL) dynamics, these annotations can target representative regions, focus on erroneous regions, or iteratively correct errors. Studying how each protocol afects refinement ofers a rigorous baseline to benchmark future progress in interactive digital pathology.

## 2 Related Work

## 2.1 Transductive VLM prediction

Unlike inductive few-shot methods [7–9] that classify each patch independently, few-shot transductive methods re-

fine all test predictions jointly, exploiting the relationships between unlabeled samples in addition to a few annotations.

## 2.1.1 In natural images

EM-Dirichlet [15] was the first transductive method to explicitly leverage the textual modality of VLMs alongside the image embeddings: it models the samples with a Dirichlet distribution and uses the VLM predictions to optimize its parameters without retraining the model. In the same spirit, TransCLIP [16] models the data distri bution as a Gaussian mixture and iteratively refines the pseudo-labels obtained from VLMs, both without labels and in a few-shot variant. More recently, StatA [17] introduced a transductive approach robust to test conditions, where the set of test samples may not contain classes anticipated by the model. Other methods rely on graph modeling: building upon ZLAP [18], which performs label propagation on a graph constructed from text prompts and unlabeled samples, ECALP [19] extends this framework to the few-shot setting by incorporating annotated samples into the graph and introduces a context-aware feature re-weighting that calibrates similarities prior to propagation.

## 2.1.2 In histology

Histo-TransCLIP [11] applies TransCLIP to histology, jointly refining patch-wise predictions from the distribution of patches in the latent space. Building on this transductive setting, HistoCRF [14] formulates patch classification as a CRF inference problem and combines VLM predictions with a pairwise potential derived from a small set of annotations to refine all ZS predictions at once. In contrast, PADDLE [10] operates on the patches within a local window of the WSI and, from a few labeled patches, estimates the optimal class assignments of these patches while penalizing a large number of distinct predicted classes. This constraint reflects a key characteristic of the WSI setting: only a few of the many candidate classes are present in a given slide.

Current limitations. Despite their diferences, except PADDLE which leverages spatial proximity by restricting its joint analysis on local windows, all the mentioned methods construct relationships primarily in the feature space, connecting patches according to their feature similarity, without using the raw data. In histopathology, tissue structures share biological information and exhibit strong spatial continuity. Therefore, relying solely on visual similarity may connect anatomically distant regions while overlooking local tissue organization. This motivates the integration of spatial and biological information into the refinement process, which we incorporate through a CRF combining spatial and biological cues in addition to model features.

![](images/ef91e7320e9a1dc7fd4594321c7280f4d9100fe604734d6408f25b4e2d7d16c3.jpg)  
Figure 1: Illustration of SlideCRF on two WSIs. From a few localized clicks (top) or a few localized scribbles (bottom), SlideCRF progressively refines the prediction and recovers spatially coherent regions.

## 2.2 Annotation protocols

A realistic WSI refinement method naturally raises the question of how these annotations are generated.

## 2.2.1 In medical images

To reduce the burden associated with pixel-wise segmentation mask delineation, weak annotations have been widely adopted, including points [20], scribbles [21] and bounding boxes [22], which trade annotation density for a much lower labeling cost.

To train methods that operate without dense supervision, several tools have been introduced to automatically generate sparse annotations from ground-truth masks [23]. For scribbles, ScribbleBench [24] samples control points inside a region and along its boundary, then interpolates a smooth curve between them. For clicks, points are sampled randomly within the object while satisfying minimum-distance constraints from both the boundary and previously sampled points, ensuring that they remain well distributed [25]. Alternatively, clicks are placed on the erroneous regions of the current prediction, typically on the largest erroneous region [26–28], optionally at the point furthest from its boundary [29]. Applied iteratively, this correction strategy allows the prediction to be refined over successive rounds [29].

## 2.2.2 In histology

[30] propose a method to generate scribbles from masks of tumoral regions. To simulate a pathologist drawing a continuous curve along a tumor, each generated scribble spans the whole region. Iterative correction has also been studied: [31] propose an interface for data annotation that iteratively retrains a model from clicks provided by a pathologist, and [30] study the interactive correction of erroneous regions, reaching high performance with only a few scribbles. More recently, interactive foundation models such as Vista-path [32] propagate sparse expert feedback into whole-slide segmentation, but rely on a trained model.

Current limitations. Each of these tools captures a diferent facet of the annotation process but none combines them into a realistic benchmark covering a range of annotation budgets, types, and interactions. To fill this gap, we propose realistic WSI annotation protocols that span two annotation types (clicks and scribbles) and three pathologist interactions such as the correction of errors. These protocols allow the generation of a range of annotation budgets and reflect a pathologist annotating distinctive regions. This provides a unified benchmark of pathologist interactions, from which we derive practical recommendations for annotating WSIs. The annotation protocols are further detailed in Section 4.

## 3 SlideCRF

## 3.1 CRF Objective function

We propose an adaptation of CRFs for realistic WSI analysis. To handle large-scale histopathological images, we divide them into a set of patches. The WSI is represented as a graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , in which each patch corresponds to a vertex $v \in \nu$ and relations between patches are represented by the set of undirected edges connecting the vertices E [33].

We want to obtain an assignment $\mathbf { y } = ( y _ { v } ) _ { v \in \mathcal { V } }$ , defining a class label for each patch. In CRF, the joint distribution over the hidden random variables is commonly expressed as a Gibbs distribution:

$$
P ( \mathbf { Y } = \mathbf { y } ) = \frac { 1 } { Z } \exp ( - E ( \mathbf { y } ) ) ,\tag{1}
$$

where $Z$ is a normalizing factor and $\begin{array} { r } { E ( \mathbf { y } ) = \sum _ { c \in \mathcal { C } } \Phi _ { c } ( y _ { c } ) } \end{array}$ is the energy defined on the set of maximal cliques C (a clique is a fully connected sub-graph).

If we state our problem as a Bayesian estimation $P ( \mathbf { Y } = \mathbf { y } | \mathbf { X } ) \propto P ( \mathbf { X } | \mathbf { Y } = \mathbf { y } ) P ( \mathbf { Y } = \mathbf { y } )$ and taking the negative logarithm, we obtain an objective function of the general form:

$$
f ( \mathbf { y } ) = - \sum _ { v \in \mathcal { V } } \log P ( x _ { v } | Y _ { v } = y _ { v } ) + E ( \mathbf { y } )  \\  = \underbrace { \sum _ { v \in \mathcal { V } } \phi _ { v } ( y _ { v } ) } _ { \mathrm { u n a r y ~ p o t e n t i a l s ~ \Phi _ { \Phi _ { u } } ~ } } + \underbrace { \sum _ { c \in \mathcal { C } } \Phi _ { c } ( y _ { c } ) } _ { \mathrm { p a i r w i s e ~ p o t e n t i a l s ~ \Phi _ { \Phi _ { p } } ~ } } .\tag{2}
$$

Unary and pairwise potentials are the building blocks of CRF methods. The former indicate the value that each variable is likely to take based on its own observed variables and the latter encourage neighboring variables to take compatible values.

## 3.2 Unary potential

In this work, we build on the definition of the unary potential proposed in our preliminary work [14], and add a term to be robust to the possible absence of classes within a slide.

## 3.2.1 Initial prediction term

For this term, we rely on histology-oriented VLMs to obtain our unary potential $\phi _ { v } ( y _ { v } )$ from patch-level predictions without training a task-specific model. More precisely, we encode each patch v with the visual encoder to get an embedding vector $\mathbf { f } _ { v }$ and for each class l textual embeddings $\mathbf { t } _ { l }$ with the textual encoder from the average of a set of prompts (e.g., “a histology image of {class}”) [11]. The unary potential is then computed from the cosine similarity:

$$
\phi _ { v } ( y _ { v } = l ) = - \log \left( \operatorname { s o f t m a x } _ { l } \left( { \frac { \mathbf { f } _ { v } \mathbf { t } _ { l } ^ { \top } } { \left\| \mathbf { f } _ { v } \right\| \left\| \mathbf { t } _ { l } \right\| } } \right) \right) .\tag{3}
$$

## 3.2.2 Class-presence term

The expert annotations A give an indication of which classes are actually present in the slide. We exploit this to discourage the prediction of classes that never appear in the annotation set. The prediction of a class is driven by its unary potential: a class with a low unary potential tends to be predicted. Let P be the set of annotated classes. To prevent the prediction of absent classes, we add a constant λ to the unary potential of every unannotated class l /∈ P, at every patch v:

$$
\tilde { \phi } _ { v } ( y _ { v } = l ) = \phi _ { v } ( y _ { v } = l ) + \lambda \mathcal { k } [ l \notin \mathcal { P } ] ,\tag{4}
$$

where $\nVdash [ \cdot ]$ is the indicator function.

## 3.3 Pairwise potentials

The definition of the pairwise potentials difers from that in our initial work [14] by the addition of two additional potentials to the annotation and diversity terms originally defined.

## 3.3.1 Annotation term

The first term connects each annotated patch $v \in { \mathcal { A } }$ to its set $\mathcal { M } _ { v }$ of highly similar patches according to their embedding vectors $\mathbf { f } _ { v }$ , extracted using a histology-oriented vision model. This encourages the refinement process to align their labels with expert input, promoting consistency with pathologist annotations. This leads to the following definition of the first pairwise term:

$$
\begin{array} { r } { \psi _ { v w } ( y _ { v } , y _ { w } ) = ( 1 - \delta _ { y _ { v } , y _ { w } } ) \mathrm { s i m } ( \mathbf { f } _ { v } , \mathbf { f } _ { w } ) , } \end{array}\tag{5}
$$

where similarities are measured using the cosine similarity:

$$
\mathrm { s i m } ( \mathbf { f } _ { v } , \mathbf { f } _ { w } ) = \frac { \mathbf { f } _ { v } \mathbf { f } _ { w } ^ { \top } } { \| \mathbf { f } _ { v } \| \| \mathbf { f } _ { w } \| } .\tag{6}
$$

## 3.3.2 Diversity term

For the second term, we connect each patch v to its set $\mathcal { N } _ { v }$ of most dissimilar patches. By linking patches that appear diferent but share the same label, this term encourages the refinement process to adjust such inconsistencies, promoting greater label diversity.

$$
\phi _ { v w } ( y _ { v } , y _ { w } ) = \delta _ { y _ { v } , y _ { w } } ( 1 - \mathrm { s i m } ( { \bf f } _ { v } , { \bf f } _ { w } ) ) ,\tag{7}
$$

## 3.3.3 Spatial term

For the third term, we use spatial connections between patches by connecting each patch v to the set $S _ { v }$ of its spatial neighbors. This term acts as a smoothing incentive that favors the assignment of identical labels to neighboring patches. It takes the same functional form as the annotation term (Eq. 5), but operates on a diferent set of pairs.

## 3.3.4 Biological term

The last pairwise term introduces handcrafted biological cues and encourages patches that appear diferent to the vision model but are biologically similar to receive the same label. It combines two feature groups, texture $\mathbf { f } _ { 1 }$ and color $\mathbf { f } _ { 2 } \mathbf { : }$

$$
\xi _ { v w } ( y _ { v } , y _ { w } ) = ( 1 - \delta _ { y _ { v } , y _ { w } } ) \sum _ { k = 1 } ^ { 2 } \eta _ { k } \mathrm { s i m } _ { 2 } ( \mathbf { f } _ { k , v } , \mathbf { f } _ { k , w } ) ,\tag{8}
$$

where the similarity is computed using a Gaussian kernel applied to the Euclidean distance between the features extracted from each patch:

$$
\mathrm { s i m } _ { 2 } ( \mathbf { f } _ { k , v } , \mathbf { f } _ { k , w } ) = \mathrm { e x p } \left( - \frac { | | \mathbf { f } _ { k , v } - \mathbf { f } _ { k , w } | | ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{9}
$$

Texture concatenates gray-level co-occurrence matrix [34] (contrast, homogeneity, energy, correlation) and local binary patterns [35], capturing the spatial arrangement of pixels. Color features [36] are computed on image patches converted in the hematoxylin-eosin-DAB (HED) color space, and concatenate the mean and standard deviation of the channels corresponding to the hematoxylin and eosin stains.

Combined pairwise potential. Combining the four terms gives our pairwise potential:

$$
\begin{array} { r l } { \Phi _ { p } = \underbrace { \beta \displaystyle \sum _ { v \in A } \displaystyle \sum _ { w \in \mathcal { M } _ { v } } { \psi } _ { v w } ( y _ { v } , y _ { w } ) } _ { \mathrm { ( i ) ~ a n n o t a t i o n ~ p a i r w i s e ~ t e r m } } + \underbrace { \alpha \displaystyle \sum _ { v \in \mathcal { V } } \displaystyle \sum _ { w \in \mathcal { N } _ { v } } { \phi } _ { v w } ( y _ { v } , y _ { w } ) } _ { \mathrm { ( i i ) ~ d i v e r s i t y ~ p a i r w i s e ~ t e r m } } } & { } \\ { + \underbrace { \gamma \displaystyle \sum _ { v \in \mathcal { V } } \displaystyle \sum _ { w \in \mathcal { S } _ { v } } { \psi } _ { v w } ( y _ { v } , y _ { w } ) } _ { \mathrm { ( i i i ) ~ s p a t i a l ~ p a i r w i s e ~ t e r m } } + \underbrace { \displaystyle \sum _ { v \in \mathcal { V } } \displaystyle \sum _ { w \in \mathcal { N } _ { v } } { \xi } _ { v w } ( y _ { v } , y _ { w } ) } _ { \mathrm { ( i v ) ~ b i o l o g i c a l ~ p a i r w i s e ~ t e r m } } \ , } \end{array}\tag{10}
$$

where $\alpha , \beta , \gamma$ and $\eta$ are weighting factors. To be robust to the partial presence of certain classes, α follows a step schedule: $\alpha _ { i } = \alpha$ if $i < \tau$ and $\alpha _ { i } = 0$ otherwise, where i is the message passing iteration (see next section 3.4) and τ is a fixed threshold. The sets $\mathcal { N } _ { v }$ and $\mathcal { M } _ { v }$ are not fixed but are drawn at random from within a subset of the most dissimilar or most similar patches to v each time the pairwise potential is computed to increase the number of connections without increasing memory load.

## 3.4 Objective function optimization

Combining the definitions of both potentials gives the following objective function:

$$
f ( \mathbf { y } ) = \sum _ { v \in \mathcal { V } } \tilde { \phi } _ { v } ( y _ { v } ) + \Phi _ { p }\tag{11}
$$

As the solution of Eq. (11) is intractable, a mean-field approximation is used. This minimizes the KL-divergence between $P ( \mathbf { Y } | \mathbf { X } )$ and a simpler approximation Q that can be written as the product of independent marginal distributions $\begin{array} { r } { Q ( \mathbf { y } ) = \prod _ { v } Q _ { v } ( y _ { v } ) } \end{array}$ . This results in an iterative update equation corresponding to a message passing algorithm [37] on the graph that can be written as:

$$
\begin{array} { c } { \displaystyle Q _ { v } ( l ) = \frac { 1 } { Z _ { v } } \exp \Big ( - \tilde { \phi } _ { v } ( l ) - \alpha \sum _ { w \in \mathcal { N } _ { v } } \big ( 1 - s _ { v w } \big ) Q _ { w } ( l ) } \\ { \displaystyle - \sum _ { l ^ { \prime } \in \mathcal { L } } \Big [ \beta \sum _ { w \in \mathcal { M } _ { v } } s _ { v w } Q _ { w } ( l ^ { \prime } ) + \gamma \sum _ { w \in \mathcal { S } _ { v } } s _ { v w } Q _ { w } ( l } \\ { \displaystyle l ^ { \prime } \not \in { l } } \\ { \displaystyle + \sum _ { w \in \mathcal { N } _ { v } } s _ { 2 , v w } Q _ { w } ( l ^ { \prime } ) \Big ] \Big ) , } \end{array}\tag{12}
$$

where $\begin{array} { r } { s _ { v w } = \operatorname { s i m } ( \mathbf { f } _ { v } , \mathbf { f } _ { w } ) , \ s _ { 2 , v w } = \sum _ { k } \eta _ { k } \operatorname { s i m } _ { 2 } ( \mathbf { f } _ { k , v } , \mathbf { f } _ { k , w } ) . } \end{array}$ and $Q _ { w } ( l ) \equiv Q _ { w } ( y _ { w } = l )$

Algorithm 1 SlideCRF (HITL)   
For 1 → Number of expert-annotation steps   
Get pathologist annotations A   
Initialize unary potential $\tilde { \phi } ^ { ( 0 ) }  \mathrm { E q } .$ (4)   
Iterative message passing   
For $i = 1  5 0$   
Get pairwise potential $\Phi _ { p }  \mathrm { E q } .$ (10)   
Iterative update: $Q ^ { ( i ) }  \mathrm { E q } .$ (12)   
Update unary potential: $\phi ^ { ( i ) } \stackrel { \setminus } {  } \frac { 1 } { 2 } ( \phi ^ { ( i - 1 ) } + Q ^ { ( i ) } )$   
Output: Predictions $\hat { y } \gets \arg \operatorname* { m a x } Q ^ { ( 5 0 ) }$

The key steps of the method are outlined in Algorithm 1.

## 4 Annotation protocols

An annotation protocol is defined by two components: an annotation type, which determines the shape of the annotation, and an interaction type, which determines how the pathologist selects the patches to annotate.

## 4.1 Annotation types

We consider two annotation types commonly used in medical imaging: clicks and scribbles (mainly based on Scrib bleBench [24]). An example of both annotation types is shown in Figure 2.

## 4.1.1 Clicks

A click is defined by a single control point (the center) and covers a disk of radius one patch around it, yielding five annotated patches (the center and its four neighbors).

## 4.1.2 Scribbles

A scribble is defined by several control points. A smooth curve, whose length is capped at 15 patches, is then interpolated between these points.

Generation process. In our experiments, the center of a click and the control points of a scribble are generated automatically from the dense segmentation mask by a three-step process, designed to mimic a pathologist annotating a region of a tissue class:

![](images/38a38c846c523397a75cb0247f522fd6b1cb42e53aeaa96a2634370feb30c2ad.jpg)

![](images/25ac08d4de082a498f3830f393a61f2217311b4f9a19ea89026c6f7009a01ba7.jpg)  
Figure 2: Examples of both annotation types, (a) clicks and (b) scribbles, on two WSIs.

(i) Component selection. For each class present in the slide, its connected components are extracted and one is sampled with probability proportional to its area, reflecting the tendency to annotate the largest, most visible structure.

(ii) Control point sampling. Within the selected component, a position is sampled from a Gaussian distribution. Its center is placed at the point furthest from the boundary (identified by the Euclidean distance transform), and its standard deviation is set to 0.4 times that point’s distance to the boundary. The Gaussian is further modulated by the model confidence, reflecting the tendency to annotate central and representative regions. For scribbles, the remaining control points are determined following ScribbleBench.

(iii) Repulsion. When several annotations fall in the same region, the sampling probability is weighted by the distance from already-annotated patches. This favors successive annotations to cover diferent parts of the structure.

## 4.2 Interaction types

We consider three pathologist interactions: representative, error-driven and human-in-the-loop (HITL).

## 4.2.1 Representative

Annotations target the most representative region of each class, following the three-step generation process introduced in the previous subsection.

## 4.2.2 Error-driven

Annotations correct erroneous patches predicted by the model. The control points (step (ii) of the generation process) are drawn exclusively among the mispredicted patches of the selected component. Each annotation is thus anchored on an error.

## 4.2.3 Human-in-the-loop (HITL)

Rather than spending the entire annotation budget at once, as in the representative and error-driven interactions, one annotation per class is added after each inference round. As in the error-driven interaction, each new annotation is anchored on a currently mispredicted patch, so that successive annotations progressively target the remaining errors of the model.

<table><tr><td>Dataset</td><td>Target organ</td><td>#WSI</td><td>#Cls.</td><td>Patch size</td><td>Patches/slide</td></tr><tr><td>BACH</td><td>Breast (human)</td><td>10</td><td>4</td><td>4482</td><td>1.2-4.7k</td></tr><tr><td>CATCH</td><td>Skin (canine)</td><td>70</td><td>12</td><td>4482</td><td>4.2-17.8k</td></tr><tr><td>SKINCANCER</td><td>Skin (human)</td><td>290</td><td>11</td><td>642</td><td>0.9-35.0k</td></tr><tr><td>TIGER</td><td>Breast (human)</td><td>93</td><td>2</td><td>4482</td><td>1.9-10.0k</td></tr></table>

Table 1: Dataset characteristics. #Cls. is the number of tissue classes and Patches/slide is the min–max number of extracted patches per slide. Patch sizes are in pixels.

## 5 Experimental Setup

## 5.1 Datasets

We evaluate the proposed method on four diverse histology WSI segmentation datasets: BACH [38], CATCH [39], SKINCANCER [40] and TIGER [41, 42]. These datasets span a wide range of cancer types and species, and they difer substantially in the number of classes, slide sizes and spatial complexity. This diversity allows us to assess the robustness of the proposed method across heterogeneous segmentation settings. For all datasets, dense segmentation masks are available, allowing patch-based evaluation.

## 5.2 Patch extraction

Each slide is tiled into non-overlapping patches (448×448 px, reduced to 64×64 for SKINCANCER, whose slides are sometimes considerably smaller). Background tiles are discarded with a per-dataset tissue filter: a grayscale standard deviation threshold (< 0.025) for BACH, a lowsaturation HSV criterion for CATCH, and a tissue-mask coverage ratio (> 0.5) for SKINCANCER and TIGER. Each retained patch receives a single label. For datasets annotated with polygons (BACH, TIGER) it is the first polygon intersecting the patch, defaulting to healthy when none applies. For the densely-masked datasets (CATCH, SKINCANCER) it is the pixel-majority class within the patch. This yields between ∼ 950 to ∼ 35, 000 labeled patches per slide, reflecting the wide range of slide sizes across datasets. A summary of the datasets’ characteristics is presented in Table 1.

## 5.3 Implementation details

Each slide is segmented independently: the model has access to all of its unlabeled patches and to a set of annotated patches sampled according to the protocols of Section 4. We vary the annotation budget b ∈ {1, 2, 4, 8, 16}. It is expressed in annotation actions so that protocols compared at equal budgets involve the same number of annotation actions. A single click and a single scribble each count as one action, and a budget of b corresponds to b actions per class present in the slide. As in [14], we use CONCH [2] to define the unary potential, while UNI-2h [43] provides the features used to build the pairwise potential. The weighting factors are set to α = 0.1, $\beta = \gamma = 0 . 5 , \eta _ { 1 } = 0 . 2 \ ( \mathrm { t e x t u r e } ) , \eta _ { 2 } = 0 . 1$ (color), and the Gaussian kernel defining the biological afinity uses a standard deviation of $\sigma = 0 . 1$ . The pairwise potential terms are sparsified to $| \mathcal { N } _ { v } | = 4$ and $| \mathcal { M } _ { v } | = | \mathcal { S } _ { v } | = 8$ connections per patch. Mean-field inference is run for 50 propagation steps; the diversity term is removed at $\tau = 2 5 \ \mathrm { ( i . e . }$ , halfway through inference) and the classpresence prior is set to $\lambda = 0 . 0 2$ . These hyperparameters were determined on a validation set of 20 slides randomly sampled from SKINCANCER (the largest dataset) and excluded from the test set. These hyperparameters are fixed once and kept identical across all runs on all four datasets, which is consistent with our realistic setting. An ablation study (Section 6.2) evaluates the contribution of each term in the pairwise potential.

## 5.4 Baselines

We compare SlideCRF against two few-shot inductive baselines, linear probe and Tip-Adapter-F, and four transductive ones. Among the transductive methods, we distinguish two families. Probabilistic approaches (Histo-TransCLIP, PADDLE) fit a distribution over the unlabeled patches and refine all predictions jointly through this global distribution. Graph-based approaches (ECALP, HistoCRF) instead build a graph over patches and propagate labels along its edges, so that refinement is driven by explicit pairwise relations. Among all baselines, ECALP is the closest to our setting, as it also combines a few annotations with VLM textual embeddings for label propagation; SlideCRF difers by additionally modeling spatial and biological pairwise potentials within a CRF. We additionally report the ZS CONCH prediction as a reference baseline. Whenever available, oficial implementations and recommended hyperparameters are used for each baseline.

## 5.5 Evaluation metrics

Performance is measured with the balanced accuracy and the macro F1. Balanced accuracy is the average, over the classes present in the slide, of the per-class recall, i.e., the fraction of each present class’s patches that are correctly retrieved. As it averages over the classes present in the slide, balanced accuracy does not directly take into account the prediction of classes that are absent from the slide. Macro F1 is the average F1 score of each class, which combines recall with precision, the fraction of patches predicted as a class that truly belong to it. Both metrics are computed per slide and then averaged over all evaluated patients in the dataset. We repeat each experiment over five random seeds, with each seed drawing a diferent set of annotations.

## 6 Experiments

## 6.1 SlideCRF matches or outperforms baselines

Table 2 summarizes the comparison against the baselines.

SlideCRF delivers substantial gains over the ZS prediction: with only one annotation per class present in the slide, it already improves by +29.0% in balanced accuracy and +24.2% in macro F1, and these gains keep growing with the annotation budget (up to +43.5% and +37.5%, respectively).

The two top-performing approaches are SlideCRF and ECALP. Both of these methods leverage transductive graph-based learning and deliver highly competitive results. While SlideCRF secures the highest average score across both metrics, driven by a more pronounced lead in macro F1, it performs nearly on par with ECALP in balanced accuracy.

Beyond predictive performance, these methods difer significantly in computational cost (Table 3). SlideCRF processes a slide of approximately 10<sup>5</sup> patches in about 38 seconds, remaining 13 times faster than ECALP.

Within the graph-based methods, SlideCRF improves substantially over HistoCRF [14], showing that adding spatial and biological information, together with the class-presence prior, raises performance by +10.3% in balanced accuracy and +12.4% in macro F1 at an annotation budget of one.

Both transductive probabilistic methods exploit annotations poorly. Histo-TransCLIP shows almost no sensitivity to the number of annotation actions, suggesting that it makes little use of the expert-provided annotations (its average balanced accuracy increases by only +1.2% between b = 1 and b = 16). PADDLE does benefit from additional annotations but remains far below the graphbased methods on every dataset.

Among inductive methods, Linear Probe gains little at low budgets. In contrast, Tip-Adapter-F leverages the annotations efectively: its performance increases substantially as the number of annotated patches grows, making it the strongest non-graph baseline at large budgets.

Visually (see Figure 3), ECALP and HistoCRF produce noisier predictions at region borders, as they rely solely on embedding similarities and ignore explicit spatial information, whereas SlideCRF enforces spatial smoothness through its spatial pairwise potential, yielding cleaner and more coherent regions.

## 6.2 Ablation studies

We evaluate the contribution of the novel components of our method: the spatial and biological terms, the weighting factor α, and the class-presence term λ. Unless stated

Table 2: Balanced accuracy (%) and macro F1 (%) for WSI patch classification with varying annotation budgets $b \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ clicks (in the representative interaction) across four datasets and their average. ZS is the CONCH zero-shot baseline. Bold: best per column; underline: second best.
<table><tr><td rowspan="2" colspan="2">DatasetApproach Method</td><td rowspan="2"></td><td rowspan="2">1</td><td colspan="5">Balanced accuracy (%)</td><td rowspan="2">1</td><td colspan="5">Macro F1 (%)</td></tr><tr><td>ZS</td><td>2</td><td>4</td><td>8</td><td>16</td><td>zS</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td rowspan="4">BACH</td><td>Inductive</td><td>Linear Probe Tip-Adapter-F</td><td>41.8</td><td>52.5 49.2</td><td>53.9 53.9</td><td>57.5 60.2</td><td>64.3 67.0</td><td>69.0 73.8</td><td>43.7</td><td>51.5 48.9</td><td>51.9 51.4</td><td>53.7 54.8</td><td>57.8 58.3</td><td>61.0 62.2</td></tr><tr><td>Probabilistic</td><td>Histo-TransCLIP PADDLE</td><td>41.8</td><td>54.7 40.2</td><td>55.5 43.4</td><td>55.8 45.8</td><td>55.2 47.5</td><td>55.0 49.3</td><td>43.7</td><td>55.0 34.9</td><td>55.5 38.5</td><td>55.9 41.1</td><td>55.3 42.7</td><td>55.4 44.3</td></tr><tr><td rowspan="2">Graph-based</td><td>ECALP HistoCRF</td><td rowspan="2">41.8</td><td>72.4 57.0</td><td>75.8 60.6</td><td>78.8 64.1</td><td>80.5 67.0</td><td>82.3</td><td rowspan="2"></td><td rowspan="2">59.4 43.7 44.7</td><td>63.0 46.6</td><td>65.7 49.0</td><td>66.8 50.4</td><td>69.9</td><td>51.6</td></tr><tr><td>SlideCRF</td><td>65.1</td><td>71.1</td><td>77.6</td><td>82.2</td><td>69.1 84.7</td><td>60.6</td><td>65.2</td><td>69.5</td><td>72.1</td><td>73.8</td></tr><tr><td rowspan="4">CACCH</td><td>Inductive</td><td>Linear Probe Tip-Adapter-F</td><td>39.0</td><td>50.5 44.9</td><td>51.4 51.0</td><td>56.9 60.5</td><td>64.6 70.1</td><td>71.3 76.4</td><td>36.9</td><td>45.8 43.0</td><td>46.2 49.2</td><td>50.4 58.7</td><td>56.0 66.8</td><td>61.5 71.7</td></tr><tr><td>Probabilistic</td><td>Histo-TransCLIP PADDLE</td><td>39.0</td><td>39.6 17.6</td><td>42.0 18.4</td><td>43.1 19.8</td><td>43.7</td><td>43.9</td><td>36.9</td><td>40.1 22.2</td><td>42.2 23.5</td><td>43.3 25.2</td><td>43.7 27.2</td><td>43.8 29.0</td></tr><tr><td>Graph-based</td><td>ECALP HistoCRF</td><td></td><td>76.7 50.7</td><td>81.6 55.1</td><td>84.2 59.0</td><td>21.5 86.0 62.5</td><td>23.2 87.0 64.4</td><td>36.9</td><td>63.5 46.5</td><td>67.0 49.4</td><td>69.7 52.3</td><td>71.9 54.9</td><td>74.0 56.6</td></tr><tr><td>Inductive</td><td>SlideCRF</td><td></td><td>39.0 76.0 33.5</td><td>78.3 40.8</td><td>80.5 49.7</td><td>82.6 58.8</td><td>84.1 66.8</td><td></td><td>64.3</td><td>66.1</td><td></td><td>67.9 69.4</td><td>70.6 17.0</td></tr><tr><td rowspan="4">SKIER</td><td></td><td>Linear Probe Tip-Adapter-F</td><td>14.8</td><td>32.6</td><td>43.1</td><td>55.9</td><td>66.9</td><td>74.9</td><td>9.0</td><td>12.8 13.7</td><td>13.2 19.6</td><td>13.9 27.2</td><td>15.4 33.6</td><td>37.5</td></tr><tr><td>Probabilistic</td><td>Histo-TransCLIP PADDLE</td><td>14.8</td><td>25.6 41.4</td><td>25.9 48.3</td><td>26.1 55.5</td><td>26.1 61.8</td><td>25.9 67.4</td><td>9.0</td><td>19.2 38.3</td><td>19.1 45.6</td><td>19.1 52.9</td><td>19.0 59.1</td><td>18.9 64.6</td></tr><tr><td>Graph-based</td><td>ECALP HistoCRF SlideCRF</td><td>14.8</td><td>54.4 60.6 68.8</td><td>61.1 67.7 75.4</td><td>66.0 74.3 81.4</td><td>68.5 79.3 85.4</td><td>69.9 83.3 88.4</td><td>9.0</td><td>47.5 48.0 57.6</td><td>55.2 53.2 63.9</td><td>61.2 58.5 70.2</td><td>65.1 63.3 75.2</td><td>67.5 67.6 79.6</td></tr><tr><td rowspan="4">TIER</td><td>Inductive</td><td>Linear Probe Tip-Adapter-F</td><td>74.7</td><td>77.5 76.5</td><td>77.8 77.7</td><td>78.6 79.7</td><td>80.0</td><td>81.4</td><td>69.1</td><td>71.1 70.2</td><td>71.5 71.1</td><td>71.7 72.6</td><td>72.9 74.3</td><td>74.1 76.2</td></tr><tr><td>Probabilistic</td><td>Histo-TransCLIP PADDLE</td><td>74.7</td><td>80.5 72.5</td><td>80.8 76.3</td><td>80.7</td><td>81.8 80.6</td><td>84.0 80.5</td><td>69.1</td><td>74.1 64.2</td><td>74.3 68.3</td><td>74.4 71.4</td><td>74.4 72.9</td><td>74.4 74.2</td></tr><tr><td>Graph-based</td><td>ECALP HistoCRF</td><td></td><td>81.6 76.9</td><td>83.8 77.5</td><td>78.7 85.4 78.3</td><td>80.0 86.4</td><td>81.0 87.4</td><td></td><td>73.5 67.0</td><td>75.7 67.5</td><td>77.7 68.0</td><td>79.4 68.0</td><td>81.0 67.9</td></tr><tr><td>Inductive</td><td>SlideCRF Linear Probe</td><td>74.7</td><td>76.4</td><td>78.9</td><td>82.3</td><td>78.7 85.0</td><td>79.2 87.1</td><td>69.1</td><td>73.0</td><td>75.8</td><td>79.3</td><td>82.4</td><td>84.7</td></tr><tr><td rowspan="4">AVGE</td><td></td><td>Tip-Adapter-F</td><td>42.6</td><td>53.5 50.8</td><td>56.0 56.4</td><td>60.6 64.1</td><td>66.9 71.4</td><td>72.1 77.3</td><td>39.7</td><td>45.3 43.9</td><td>45.7 47.8</td><td>47.4 53.3</td><td>50.5 58.3</td><td>53.4 61.9</td></tr><tr><td>Probabilistic</td><td>Histo-TransCLIP PADDLE</td><td>42.6</td><td>50.1 42.9</td><td>51.1 46.6</td><td>51.5 49.9</td><td>51.4 52.7</td><td>51.3 55.2</td><td>39.7</td><td>47.1 39.9</td><td>47.8 44.0</td><td>48.2 47.7</td><td>48.1 50.5</td><td>48.1 53.1</td></tr><tr><td rowspan="2">Graph-based</td><td>ECALP HistoCRF</td><td>42.6</td><td>71.3 61.3</td><td>75.6 65.2</td><td>78.6 68.9</td><td>80.4 71.9</td><td>81.7 74.0</td><td>39.7</td><td>61.0 51.5</td><td>65.2 54.1</td><td>68.6 56.9</td><td>70.8 59.1</td><td>73.1 60.9</td></tr><tr><td>SlideCRF</td><td>71.6</td><td>75.9</td><td></td><td>80.4</td><td>83.8</td><td>86.1</td><td></td><td>63.9</td><td>67.7</td><td>71.7</td><td>74.8</td><td>77.2</td></tr></table>

Table 3: Total runtime (in seconds) on an NVIDIA A10 GPU, as a function of the number of patches.
<table><tr><td>Approach</td><td>Method</td><td> $\sim 1 0 ^ { 3 }$ </td><td> $\sim 1 0 ^ { 4 }$ </td><td> $\sim 1 0 ^ { 5 }$ </td></tr><tr><td rowspan="2">Inductive</td><td>Linear Probe</td><td>6.3</td><td>12.6</td><td>32.6</td></tr><tr><td>Tip-Adapter-F</td><td>0.1</td><td>5.1</td><td>25.8</td></tr><tr><td rowspan="2">Probabilistic</td><td>Histo-TransCLIP</td><td>0.1</td><td>1.0</td><td>4.4</td></tr><tr><td>PADDLE</td><td>0.8</td><td>11.4</td><td>40.5</td></tr><tr><td rowspan="2">Graph-based</td><td>ECALP</td><td>8.0</td><td>142.6</td><td>490.9</td></tr><tr><td>SlideCRF</td><td>3.2</td><td>12.6</td><td>37.8</td></tr></table>

otherwise, all ablations use an annotation budget $b = 4$ clicks (in the representative interaction) and $\lambda = 0 . 0 2$

## 6.2.1 Spatial and biological terms

Table 4 highlights the improvements in balanced accuracy and macro F1 achieved by incorporating spatial and biological terms into SlideCRF, compared to the HistoCRF baseline.

The spatial term is by far the largest contributor, yielding gains of +5.9% in balanced accuracy and +8.1% in macro F1. This is expected, since tissue in a WSI is spatially organized into coherent regions, making a patch’s class highly predictive of its neighbors. As shown in Figure 4, the spatial term acts as a smoothing mechanism that promotes label consistency across neighboring patches.

The biological term, combining texture and color cues, further improves macro F1 by +5.4%. They allow us to refine the boundaries between the present classes. In Figure 4, we can see that the biological term, in particular the texture, helps distinguish papillary dermis (light pink) from hypodermis (dark pink) classes. Although both correspond to pink tissue with few visible nuclei from a semantic perspective, their textures difer markedly as shown in the examples of Figure 5. The color, in turn, helps distinguish papillary dermis from basal cell carcinoma: while papillary dermis is composed of pink collagen, basal cell carcinoma consists of blue basaloid cells.

![](images/f480dd7335e4dc592fcbe6102294e3a70a59a2054237d48a0ff71942f34fbae4.jpg)

![](images/e51678d5d4c54c95eca00d3624986e79823faa1fe04e7f389f35fbadeaa4c3ca.jpg)  
Figure 3: Qualitative comparison of graph-based methods on a SKINCANCER slide with b = 4 clicks (in the representative interaction). SlideCRF recovers spatially coherent regions, whereas the other methods ECALP and HistoCRF leave noisier, fragmented boundaries.

## 6.2.2 Diversity weighting factor and classpresence term

Table 5 reports the efect of the weighting factor α of the diversity term and the class-presence term λ on the balanced accuracy and the macro F1 score. The role of the diversity term is to promote label diversity, and therefore increase the number of predicted classes. Setting α uniform allows the propagation of classes missed by the ZS predictions. This explains the gains in both balanced accuracy and in macro F1, compared to the version without this term.

However, in WSIs, some classes may be absent. We therefore adopt a step schedule that sets α to 0 halfway through the propagation (α non-uniform), so as to mitigate the spread of absent labels. We report results on two datasets that illustrate two contrasting cases: in BACH, nearly all classes are present, whereas in CATCH only 4 of the 12 classes are present on average. On CATCH, the step schedule reduces the number of predicted classes while slightly increasing balanced accuracy and macro F1. On BACH, where absent labels are not a concern, performance is essentially unafected. These behaviors are also visible qualitatively in Figure 6.

Performance when the class-presence term λ is added is also reported. This term is complementary to α nonuniform, but is derived from the annotations and acts on the unary potential. This mechanism provides a substantial additional improvement, increasing balanced accuracy and macro F1 on both datasets, with +12.3% and +3.3% on CATCH, respectively, while reducing the number of predicted classes.

## 6.2.3 Limitation of class-presence term

To reflect a clinical setting in which a pathologist may overlook a small region, we evaluate the case where a class present in the slide is left unannotated. On CATCH, we omit the least, 2nd-least and 3rd-least prevalent class from the annotations and measure how the class-presence term afects the recall of the unannotated class and the macro F1. As shown in Figure 7, at the chosen strength $( \lambda = 0 . 0 2 )$ , the unannotated class is never erased and retains 17–24% of its recall, while macro F1 stays high. Suppression becomes severe only for $\lambda \ge 0 . 1$ , where the unannotated class collapses to 5% recall or below, and vanishes entirely at $\lambda = 0 . 2 5$

Table 4: Contribution of each term to the pairwise potential: the spatial and biological terms are added cumulatively on top of the annotation and diversity terms. Balanced accuracy (%) and macro F1 (%) are averaged over the four datasets.
<table><tr><td>Pairwise terms</td><td>Balanced accuracy (%)</td><td>Macro F1 (%)</td></tr><tr><td>Annotation + Diversity</td><td>74.3</td><td>58.2</td></tr><tr><td>+ Spatial</td><td>80.2</td><td>66.3</td></tr><tr><td>+ Biological (SlideCRF)</td><td>80.4</td><td>71.7</td></tr></table>

![](images/3a953e06b325a99c595240e032aae0105b4183140f608ab80e7964735565d819.jpg)  
Figure 4: Efect of the addition of the spatial and biologi cal pairwise terms on a SKINCANCER slide. The spatial term recovers more coherent tissue regions, while the biological term sharpens class boundaries.  
Figure 5: Visual diference on SKINCANCER patches (a) between papillary dermis (fine collagen) and hypodermis (empty, honeycomb-like fat vacuoles), separated by texture; (b) between papillary dermis (pink collagen) and basal cell carcinoma (blue basaloid cells), separated by color.

![](images/d16dad1e52b90113694811862ac9cc82e2605c7edeba9197cc82563166b9a5b3.jpg)

Table 5: Contribution of the weighting factor α and the class-presence term λ. $n _ { p r e d } / n _ { g t } = \mathrm { a v g } .$ . predicted vs. present classes per slide.
<table><tr><td>Dataset</td><td>Metric</td><td> $\alpha = 0$ </td><td>α uniform</td><td>α non-uniform</td><td>+λ≠0</td></tr><tr><td rowspan="3">CATCH</td><td>Bal. Acc</td><td>42.3</td><td>68.2</td><td>68.2</td><td>80.5</td></tr><tr><td>Macro F1</td><td>38.1</td><td>63.2</td><td>64.6</td><td>67.9</td></tr><tr><td> $n _ { p r e d } / n _ { g t }$ </td><td>4.1/3.7</td><td>12.9/3.7</td><td>10.1/3.7</td><td>7.6/3.7</td></tr><tr><td rowspan="3">BACH</td><td>Bal. Acc</td><td>52.2</td><td>74.3</td><td>72.8</td><td>77.6</td></tr><tr><td>Macro F1</td><td>49.4</td><td>66.4</td><td>66.9</td><td>69.5</td></tr><tr><td> $n _ { p r e d } / n _ { g t }$ </td><td>3.5/3.4</td><td>4.0/3.4</td><td>3.9/3.4</td><td>3.4/3.4</td></tr></table>

![](images/184c93b028053098625e11f0fac45b1eecc5fed48fac9b574afb355d463af5e2.jpg)  
Figure 6: Efect of the mechanisms handling absent classes on a SKINCANCER slide. Setting α to 0 halfway to the propagation (non-uniform α) and adding the classpresence term (SlideCRF) progressively suppress the prediction of absent classes, unlike a uniform α.

![](images/6dded0162ed7690d1665c1cb26afd2494745e9022dabe8b8f7a28d7d7df20275.jpg)  
Figure 7: Efect of the class-presence term strength λ on the macro F1 when a class in the slide is not annotated (least-, 2nd-least- and 3rd-least prevalent). It is important to notice that a moderate strength improves macro F1 without erasing an unannotated class.

## 6.3 HITL interaction yields the best performance

The results of the diferent annotation protocols are summarized in Figure 8. On most of the datasets, scribbles match or outperform clicks, most clearly in macro F1 at high budget: unlike a spatially compact click, a scribble spans a wider range of tissue structures and thus covers more diverse patches. For the same annotation budget, i.e., the same number of pathologist actions, this provides richer supervision and improves propagation.

We then investigate whether correcting errors iteratively through a HITL procedure is more efective than annotating the full budget at once. On every dataset, clicks-error HITL increasingly outperforms clicks-error as the budget grows, reaching +6.1% in macro F1 on average at 16 annotations per class. HITL continuously retargets the residual errors of the current, progressively improved prediction, allowing it to correct systematic confusions. This iterative benefit also extends to scribbles: scribbleerror HITL improves over single-step scribble-error by +5.3% in macro F1 on average at 16 annotation actions.

Practical recommendation. On average, iterative error correction (HITL) yields the best performance, followed by single-step representative annotation, with single-step error-driven annotation ranking last. As for the choice of modality, scribbles are the preferable default, matching or surpassing clicks on most datasets at a comparable budget.

![](images/38bb04c84514ea0c322f73ddb0f1a6f15fafea906401908738e8b813505c7a8d.jpg)

![](images/392f45764f71ffb706d358263d26160462667eb878233e0159ea0fc291b64709.jpg)

![](images/cee4c81b8e3a1b6fe471ef7004a83adcededb8fae08db1c62c57487309aa4d72.jpg)

![](images/937d3ddf01f2d11893f90e9f1c1fbd6b5cc38cce5cf8fe8b619813dba2cc9456.jpg)  
Figure 8: Macro F1 (%) of the considered annotation protocols with varying annotation budgets $b \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ across four datasets and their average. Lines share a color per annotation type and a style per interaction type.

## 7 Conclusion

We introduce SlideCRF, a CRF tailored to a realistic clinical setting of WSI analysis, where a few expert annotations and the relations between patches are exploited to refine patch-level predictions. To account for the spatial organization and the severe class imbalance specific to WSIs, our formulation combines terms integrating spatial and biological information with a term that penalizes the prediction of classes that are never annotated. Across four diverse datasets, SlideCRF outperforms strong baselines in both balanced accuracy and macro F1, highlighting its potential for clinically realistic WSI refinement. Our annotation study further shows that iterative error correction yields the largest gains, confirming the value of keeping the pathologist in the loop.

Future Work. Several directions remain open. A natural next step is to validate the proposed annotation protocols directly with pathologists, in order to confirm their clinical realism. Eficiency and accuracy could also be improved by exploiting patches at multiple resolutions, following the pyramidal structure of WSIs. Beyond histology, our framework could be extended to 3D medical images, where volumes exhibit analogous spatial connectivity. More broadly, quantifying how each protocol affects refinement provides a common ground for comparing future interactive annotation methods, which is essential before deploying them in routine clinical practice, where pathologist annotation time is scarce and directly constrains diagnostic throughput.

## References

[1] M. Moscalu, R. Moscalu, C. G. Dasc˘alu et al., “Histopathological images analysis and predictive modeling implemented in digital pathology—current afairs and perspectives,” Diagnostics, vol. 13, no. 14, p. 2379, Jul. 2023. [Online]. Available: http://dx.doi.org/10.3390/diagnostics13142379

[2] M. Y. Lu, B. Chen, D. F. K. Williamson et al., “A visual-language foundation model for computational pathology,” Nature Medicine, vol. 30, no. 3, p. 863–874, Mar. 2024. [Online]. Available: http://dx.doi.org/10.1038/s41591-024-02856-4

[3] Z. Huang, F. Bianchi, M. Yuksekgonul et al., “A visual–language foundation model for pathology image analysis using medical twitter,” Nature Medicine, vol. 29, no. 9, p. 2307–2316, Aug. 2023. [Online]. Available: http://dx.doi.org/10. 1038/s41591-023-02504-3

[4] W. Ikezogwo, S. Seyfioglu, F. Ghezloo et al., “Quilt-1m: One million image-text pairs for histopathol-

ogy,” Advances in neural information processing systems, vol. 36, pp. 37 995–38 017, 2023.

[5] M. Sikaroudi, M. Hosseini, R. Gonzalez et al., “Generalization of vision pre-trained models for histopathology,” Scientific Reports, vol. 13, no. 1, Apr. 2023. [Online]. Available: http://dx.doi.org/ 10.1038/s41598-023-33348-z

[6] J. Lee, J. Lim, K. Byeon et al., “Benchmarking pathology foundation models: Adaptation strategies and scenarios,” Computers in Biology and Medicine, vol. 190, p. 110031, May 2025. [Online]. Available: http: //dx.doi.org/10.1016/j.compbiomed.2025.110031

[7] R. Zhang, R. Fang, P. Gao et al., “Tip-adapter: Training-free clip-adapter for better vision-language modeling,” arXiv preprint arXiv:2111.03930, 2021.

[8] K. Zhou, J. Yang, C. C. Loy et al., “Learning to prompt for vision-language models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.

[9] M. Zanella and I. Ben Ayed, “Low-rank few-shot adaptation of vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1593–1603.

[10] A. Sadraoui, S. Martin, E. Barbot et al., “A transductive few-shot learning approach for classification of digital histopathological slides from liver cancer,” in 2024 IEEE International Symposium on Biomedical Imaging (ISBI), 2024, pp. 1–5.

[11] M. Zanella, F. Shakeri, Y. Huang et al., “Boosting vision-language models for histopathology classification: Predict all at once,” in International Workshop on Foundation Models for General Medical AI. Springer, 2024, pp. 153–162.

[12] T. Joachims, “Transductive inference for text classification using support vector machines,” in Proceedings of the Sixteenth International Conference on Machine Learning, ser. ICML ’99. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc., 1999, p. 200–209.

[13] D. Zhou, O. Bousquet, T. N. Lal et al., “Learning with local and global consistency,” in Proceedings of the 17th International Conference on Neural Information Processing Systems, ser. NIPS’03. Cambridge, MA, USA: MIT Press, 2003, p. 321–328.

[14] T. Godelaine, M. Zanella, K. El Khoury et al., “Conditional random fields for interactive refinement of histopathological predictions,” in ICASSP 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 7682–7686.

[15] S. Martin, Y. Huang, F. Shakeri et al., “Transductive zero-shot and few-shot clip,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 28 816–28 826.

[16] M. Zanella, B. G´erin, and I. B. Ayed, “Boosting vision-language models with transduction,” Advances in Neural Information Processing Systems, vol. 37, pp. 62 223–62 256, 2024.

[17] M. Zanella, C. Fuchs, C. De Vleeschouwer et al., “Realistic test-time adaptation of vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 25 103–25 112.

[18] Y. Kalantidis, G. Tolias et al., “Label propagation for zero-shot classification with vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 23 209–23 218.

[19] Y. Li, Y. Su, A. Goodge et al., “Eficient and contextaware label propagation for zero-/few-shot trainingfree adaptation of vision-language model,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 57 322–57 342.

[20] H. Zhang, L. Burrows, Y. Meng et al., “Weakly supervised segmentation with point annotations for histopathology images via contrast-based variational model,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15 630–15 640.

[21] Z. Wang, Y. Ye, Z. Chen et al., “From few to more: Scribble-based medical image segmentation via masked context modeling and continuous pseudo labels,” IEEE Journal of Biomedical and Health Informatics, vol. 30, no. 3, p. 2419–2431, Mar. 2026. [Online]. Available: http: //dx.doi.org/10.1109/JBHI.2025.3599066

[22] M. Gaillochet, M. Noori, S. Dastani et al., “Prompt learning with bounding box constraints for medical image segmentation,” IEEE Transactions on Biomedical Engineering, 2025.

[23] Z. Marinov, P. F. J¨ager, J. Egger et al., “Deep interactive segmentation of medical images: A systematic review and taxonomy,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024.

[24] K. Gotkowski, K. H. Maier-Hein, and F. Isensee, “Revisiting 3d medical scribble supervision: Bench marking beyond cardiac segmentation,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2025, pp. 436–446.

[25] N. Xu, B. Price, S. Cohen et al., “Deep interactive object selection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 373–381.

[26] S. Mahadevan, P. Voigtlaender, and B. Leibe, “Iteratively trained interactive segmentation,” in British Machine Vision Conference (BMVC), 2018.

[27] K. Sofiiuk, I. A. Petrov, and A. Konushin, “Reviving iterative training with mask guidance for interactive segmentation,” in 2022 IEEE International Conference on Image Processing (ICIP). IEEE, 2022, pp. 3141–3145.

[28] Q. Liu, Z. Xu, G. Bertasius et al., “Simpleclick: Interactive image segmentation with simple vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 22 290–22 300.

[29] T. Sakinis, F. Milletari, H. Roth et al., “Interactive segmentation of medical images through fully convolutional neural networks,” arXiv preprint arXiv:1903.08205, 2019.

[30] A. Habis, R. R. Nathanson, V. Meas-Yedid et al., “Scribble-based fast weak-supervision and interactive corrections for segmenting whole slide images,” 2024. [Online]. Available: https://arxiv. org/abs/2402.08333

[31] B. Lutnick, B. Ginley, D. Govind et al., “An integrated iterative annotation technique for easing neural network training in medical image analysis,” Nature Machine Intelligence, vol. 1, no. 2, pp. 112–119, 2019.

[32] P. Liang, S. Li, S. Koga, Y. Li, Z. Alipour, Y. Tang, D. Xu, and Z. Huang, “Vista-path: An interactive foundation model for pathology image segmentation and quantitative analysis in computational pathol ogy,” arXiv preprint arXiv:2601.16451, 2026.

[33] I. Ben Ayed, High-Order Models in Semantic Image Segmentation. Elsevier, 2023. [Online]. Available: http://dx.doi.org/10.1016/C2015-0-04313-1

[34] R. M. Haralick, K. Shanmugam, and I. Dinstein, “Textural features for image classification,” IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, 1973.

[35] T. Ojala, M. Pietik¨ainen, and T. M¨aenp¨a¨a, “Multiresolution gray-scale and rotation invariant texture classification with local binary patterns,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971–987, 2002.

[36] A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical staining by color deconvolution,”

Analytical and Quantitative Cytology and Histology, vol. 23, no. 4, pp. 291–299, 2001.

[37] P. Kr¨ahenb¨uhl and V. Koltun, “Eficient inference in fully connected CRFs with gaussian edge potentials,” in Advances in Neural Information Processing Systems (NeurIPS), 2011.

[38] G. Aresta, T. Ara´ujo, S. Kwok et al., “Bach: Grand challenge on breast cancer histology images,” Medical image analysis, vol. 56, pp. 122–139, 2019.

[39] F. Wilm, M. Fragoso, C. Marzahl et al., “Pan-tumor canine cutaneous cancer histology (catch) dataset,” Scientific data, vol. 9, no. 1, p. 588, 2022.

[40] S. Thomas, N. Hamilton, and S. Thomas, “Histopathology non-melanoma skin cancer segmentation dataset,” 2021. [Online]. Available: http://dx.doi.org/10.14264/8be4bd0

[41] TIGER Challenge, “TIGER: Tumor InfiltratinG lymphocytes in breast cancer challenge,” https:// tiger.grand-challenge.org, 2022, accessed: 2026-07- 24.

[42] M. Amgad, H. Elfandy, H. Hussein et al., “Structured crowdsourcing enables convolutional segmentation of histology images,” Bioinformatics, vol. 35, no. 18, pp. 3461–3467, 2019.

[43] R. J. Chen, T. Ding, M. Y. Lu et al., “Towards a general-purpose foundation model for computational pathology,” Nature Medicine, vol. 30, no. 3, p. 850–862, Mar. 2024. [Online]. Available: http: //dx.doi.org/10.1038/s41591-024-02857-3