# UniCounting: Instance-Aware Proposal Consolidation for Image-Query-Free Multi-Category Counting

Jinshi Liu<sup>1</sup>, Pan Liu<sup>2</sup>, Lei He<sup>3</sup>, Weichao Luo<sup>4,\*</sup>, Rui Qian<sup>5,\*</sup>

<sup>1</sup>Shenzhen University <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>3</sup>Hunan University of Science and Technology <sup>4</sup>Peng Cheng Laboratory <sup>5</sup>Fudan University

<sup>1</sup>jsl@szu.edu.cn, <sup>2</sup>liup28292@gmail.com, <sup>3</sup>helei\_xb@hnust.edu.cn

<sup>4</sup>luowch@pcl.ac.cn, <sup>5</sup>qiianruii@gmail.com

Co-corresponding authors.

## Abstract

Visual counting is commonly formulated as counting a single specified target, with a model receiving an image-specific exemplar, text query, or target category and returning a single count. We instead study fixed-vocabulary image-queryfree multi-category counting. A global vocabulary is fixed for each run, and, given only an RGB image, the model predicts a complete category–count vector without being told which categories appear. We present UniCounting, which casts counting as instance-aware structural inference over an over-complete proposal set. Generic segmenters produce duplicate masks, partial views, and proposals from neighboring instances; semantic scores can name them but cannot determine which denote the same object. Frozen SAM 2.1 generates masks, while frozen DINOv2 and OpenCLIP provide relation and category features. A 3,267-parameter category-shared relation head predicts same-instance afinities from instance-maskderived supervision. Sparse graph construction, representative selection, labeling, and background-margin admission then convert each admitted component into one count with replayable group evidence. Only the relation head is trained, without count or density-map targets. On COCO clean500, UniCounting obtains lower point-estimate vector ℓ error and absent-class false mass than calibrated OWLv2-All80, with comparable micro presence F1. Under a matched decoder, the learned relation reduces both errors relative to mask containment, mask IoU, CLIP, and DINO, while revealing a fragmentation–merge trade-of. We also report transfer diagnostics on OmniCount-sub, FSC-147, and CARPK.

## 1 Introduction

Visual counting is typically formulated as a conditional, single-category task. A model receives an exemplar or a category query that specifies what to count and returns a scalar count or density map for that target (Ranjan et al. 2021; Liu et al. 2022; Jiang, Liu, and Chen 2023; Amini-Naieni, Han, and Zisserman 2024). This interface is efective when the target is known in advance, but it does not directly answer a scene-level question: which object categories are present, and how many instances of each occur? We study imagequery-free multi-category counting, where a global vocabulary is fixed once for each deployment or evaluation run, but no image-dependent category selection is provided. Thus, image-query-free does not mean vocabulary-free; rather, the model receives no per-image target category, exemplar, or restricted class list. Given only an RGB image, it returns a complete category–count vector together with the proposal groups supporting its nonzero entries.

![](images/8489e12a87b8382eab91fbd8e677806e55fdbcd47815b7e522cdfaee3956f19f.jpg)  
Figure 1: Qualitative counting examples. In the upper example, colored boxes mark representative proposals for the displayed groups, and arrows connect each representative to its fixed-vocabulary category and predicted count. In the lower example, one predicted component merges proposals supporting at least five distinct ground-truth hot-dog instances. The remaining discrepancy between the ground-truth count of 11 and the single predicted component may also include upstream proposal misses.

Several lines of work provide parts of this capability. Category-labeled detectors can be converted into count vectors by summing detections, supervised multi-class counters predict fixed category channels, and reference-less counters can organize foreground objects into image-specific groups (Minderer, Gritsenko, and Houlsby 2023; Xu et al. 2021; Hobley and Prisacariu 2024; Spanakis, Oikonomidis, and Argyros 2026). What remains underexplored is the instanceconsolidation problem that appears when counting is built from generic segmentation proposals without an imagespecific target. Automatic segmenters produce over-complete candidate sets containing duplicate masks for one object, partial views of the same object, and visually similar proposals from neighboring objects. Semantic scores can assign category names to proposals, but they do not determine whether two proposals denote the same physical instance. Reliable counting therefore requires resolving proposal identity before selecting one representative and one label for each surviving group.

We introduce UniCounting, an instance-aware proposalconsolidation framework built on frozen foundation-model features. SAM 2.1 generates candidate masks, while DI-NOv2 and OpenCLIP provide relation and category features (Ravi et al. 2025; Oquab et al. 2024; Radford et al. 2021; Cherti et al. 2023). A 3,267-parameter categoryshared relation head learns pairwise same-instance afinity from instance-mask-derived supervision. UniCounting converts these afinities into a sparse graph, forms connected components, selects a maximum-category-margin representative for each component, assigns the representative’s fixedvocabulary label, and applies background-margin admission using a validation-calibrated threshold. The distinction is not the absence of a vocabulary, but whether image-dependent category selection is supplied. Each admitted component contributes one count. The resulting vector is therefore accompanied by an explicit trail from proposals and relation edges to components, representatives, labels, and admission decisions. Only the relation head is trained; the category head and the three foundation-model components remain frozen, and training uses no count or density-map targets. Figure 1 illustrates this proposal-to-group-to-count trace.

We evaluate UniCounting at two complementary levels. At the system level, complete-vector experiments measure category-wise counting, absent-category errors, and presence prediction, including a calibrated category-labeled detector as a direct reference. At the mechanism level, a matched-decoder study holds the candidate pool and downstream decoder fixed while replacing only the edge score; group-partition diagnostics and component-wise ablations then localize the efects of relation scoring, representative selection, labeling, and admission. Our primary evidence comes from the complete-vector COCO evaluation and the matched-decoder analysis, while the remaining held-out and single-coordinate datasets serve as supporting diagnostics.

Our contributions are twofold:

• We cast fixed-vocabulary image-query-free multicategory counting as an instance-aware proposalconsolidation problem and introduce UniCounting, a foundation-model pipeline with a 3,267-parameter trainable relation head to convert over-complete frozen proposals into complete count vectors with replayable group evidence.

• We provide a controlled empirical framework that separates system-level counting performance from proposalconsolidation quality through calibrated complete-vector references, matched-decoder edge comparisons, grouppartition diagnostics, and decoder ablations. Using this framework, we show that learned instance relations reduce vector $\ell _ { 1 }$ error and absent-class false mass relative to the tested edge-score alternatives, while exposing a trade-of between proposal fragmentation and instance merging.

## 2 Related Work

Queried counting. Few-shot counters identify one target from exemplar boxes (Ranjan et al. 2021; Shi et al. 2022; Liu et al. 2022; Ðukić et al. 2023). Text-conditioned methods instead use a category phrase (Jiang, Liu, and Chen 2023; Amini-Naieni et al. 2023; Kang et al. 2024; Amini-Naieni, Han, and Zisserman 2024). PseCo combines such conditioning with proposals and SAM masks (Huang et al. 2024). Recent text-guided zero-shot counters further improve crossmodal grounding and exemplar construction (Qian et al. 2025; Liu et al. 2025). In each case, target selection remains external to the counter.

Closed-set and discovered categories. Supervised multiclass counting predicts one density channel per trained class (Xu et al. 2021; Gao, Zhao, and Li 2024), while SIMCO clusters emergent foreground types (Godi et al. 2021). These establish fixed-channel and category-discovery precedents, so fixed-vocabulary output alone is not our novelty claim.

Detection, grounding, and count vectors. Fixed-category detectors provide class-labeled instances, so summing retained detections by class is a direct complete-vector baseline. Grounding DINO takes category names or referring expressions for open-set detection, and OWLv2 scales detection with language-defined categories (Liu et al. 2024; Minderer, Gritsenko, and Houlsby 2023). Because this reduction already provides vector outputs, UniCounting focuses on proposal grouping and category admission under vector-level counting errors, with a replayable group for every accepted count.

Multi-category counting. Reference-less counters typically return one anonymous count (Ranjan and Hoai 2022; Hobley and Prisacariu 2023; Xu et al. 2023). ABC123 and OCCAM discover multiple image-specific types without exemplars (Hobley and Prisacariu 2024; Spanakis, Oikonomidis, and Argyros 2026), whereas OmniCount names userspecified categories (Mondal et al. 2025). PrACo tests whether text-guided counters reject absent labels, while PrACo++ adds negative-label and distractor protocols and MUCCA provides real multi-category scenes (Ciampi et al. 2025; Pacini et al. 2026). UniCounting complements these directions by using one global vocabulary fixed for the entire run before its first image is processed, emitting every coordinate, and reporting both absent-class false mass and presence F1.

Proposal-based recognition. Foundation models support region discovery and semantic matching (Kirillov et al. 2023; Ravi et al. 2025; Radford et al. 2021; Cherti et al. 2023; Oquab et al. 2024), but their proposal sets contain duplicates and incomplete views. We isolate the conversion of these redundant proposals into a complete named count vector with group-level evidence.

## 3 Method

## 3.1 Task Definition

For each deployment or evaluation run, a global vocabulary is fixed before any image in that run is processed;

![](images/7f28b49ef3dcdab8268a4ee10e796d5926a95e35a17c8e84eae2dad82131e5c9.jpg)  
Figure 2: UniCounting modules and data flow. Frozen SAM 2.1 generates over-complete candidates, while frozen DINOv2 and OpenCLIP encode the three proposal views for relation features. The fixed category head uses only the masked- and box-view OpenCLIP embeddings to compute fixed-vocabulary category scores. The trained relation head supplies the same-instance afinity used by thresholding, top2-OR, and union-find. Deterministic maximum-category-margin representative selection, rawcosine labeling, and background-margin admission then produce the count vector and its group evidence. Snowflake, lightning, and gray modules denote frozen, trained, and parameter-free computation, respectively.

no image-dependent category selection is supplied. Let $\mathcal { V } = \{ \bar { c } _ { 1 } , \ldots , \bar { c } _ { K } \}$ denote this vocabulary and I an RGB image. We call the task image-query-free because inference receives I but no image-specific text, point, exemplar, target label, or restricted class list. Category names, templates, and text prototypes are fixed together with V for the full run. For each category $c ,$ the model returns a collection $\textstyle { \widehat { \mathcal { R } } } _ { c }$ of admitted proposal components, together with the count vector

$$
\widehat { \mathbf { n } } ( I ; \mathcal { V } ) = \big ( \widehat { n } _ { c _ { 1 } } , \dots , \widehat { n } _ { c _ { K } } \big ) , \qquad \widehat { n } _ { c } = | \widehat { \mathcal { R } } _ { c } | ,\tag{1}
$$

where every $G \in \widehat { \mathcal { R } } _ { c }$ is a connected component of proposals assigned to c and retains a designated representative $k _ { G } ^ { \star }$ Thus, $\textstyle { \widehat { \mathcal { R } } } _ { c }$ is a collection of components, not a set of representative proposals. A category is present when $\hat { n } _ { c } > 0$ . The predictor cannot read a target class, a per-image class list, or any evaluation annotation. The global vocabulary defines the ontology for the run; it is neither a per-image prompt nor an image-dependent class list.

## 3.2 Fixed Category Head

A frozen SAM 2.1 automatic mask generator (Ravi et al. 2025) produces candidates $\mathcal { M } = \{ m _ { i } \} _ { i = 1 } ^ { N }$ . Each candidate has masked, box, and context crops. Frozen DINOv2 (Oquab et al. 2024) and OpenCLIP (Radford et al. 2021; Cherti et al. 2023) encode all three views for relation prediction. For view $v ,$ let $\mathbf { e } _ { i } ^ { v }$ denote the L2-normalized OpenCLIP embedding of candidate $i ,$ and let norm(·) denote L2 normalization. The category head uses only the masked and box OpenCLIP embeddings. Fixed templates are averaged into a normalized prototype $\mathbf { t } _ { c }$ for each $c \in \mathcal V$ . The proposal feature and its raw category score are computed together as

$$
\mathbf { v } _ { i } = \operatorname { n o r m } ( 0 . 0 5 \mathbf { e } _ { i } ^ { \mathrm { m a s k } } + 0 . 9 5 \mathbf { e } _ { i } ^ { \mathrm { b o x } } ) ,\tag{2}
$$

$$
\begin{array} { r } { s _ { i c } = { \bf v } _ { i } ^ { \mathsf { T } } { \bf t } _ { c } , \qquad c \in \mathcal { V } . } \end{array}\tag{3}
$$

Template lists and taxonomy-specific name aggregation are given in the supplement. The classifier over V has no trainable parameters, no trainable background or unknown class, and no learned rejection gate. A separate fixed bank of seven background prototypes is used only by the admission rule in Section 3.5. The selected decoder uses the category cosines for representative selection and labeling.

## 3.3 Relation Head and Same-Instance Score

The relation head is the only trainable module. For each directed pair $i \  \ j ,$ , a 16-dimensional feature $\phi _ { i \to j }$ contains same-view DINO and CLIP cosine similarities, crossview CLIP similarities, directed log-area ratio, minimumto-maximum area ratio, two image-area fractions, two SAM 2.1 scores, their absolute diference, and a constant. Each dimension is centered by its relation-training mean and divided by its relation-training standard deviation after clamping that denominator below at $1 0 ^ { - 4 }$ . Feature 16 is an inactive compatibility placeholder: after standardization it is identically zero and, because the MLP already has afine biases, it does not afect any branch logit. Let $\mathbf { \bar { W } _ { 1 } } \in \mathbb { R } ^ { 6 4 }$ ×16 and $W _ { 2 } \in \mathbb { R } ^ { 3 2 \times 6 4 }$ have biases $\mathbf { b } _ { 1 } \in \mathbb { R } ^ { 6 4 }$ and $\mathbf { b } _ { 2 } \in \mathbb { R } ^ { 3 2 }$ and let $D _ { 0 . 1 }$ denote dropout with rate 0.1. For branch $r \in { \mathcal { H } } = .$ {sem, inst, comp}, let ${ \mathbf w } _ { r } \in \mathbb { R } ^ { 3 2 }$ and $\beta _ { r } \in$ R parameterize its output layer. Let $\sigma ( x ) = ( 1 + e ^ { - x } ) ^ { - 1 }$ denote the logistic sigmoid. The directed representation, branch logit, and symmetric same-instance score are

$$
\mathbf { h } _ { i \to j } = \operatorname { G E L U } \left( W _ { 2 } D _ { 0 . 1 } \big ( \operatorname { G E L U } ( W _ { 1 } \phi _ { i \to j } + \mathbf { b } _ { 1 } ) \big ) + \mathbf { b } _ { 2 } \right)\tag{4}
$$

$$
z _ { i  j } ^ { r } = \mathbf { w } _ { r } ^ { \mathsf { T } } \mathbf { h } _ { i  j } + \beta _ { r } , \qquad r \in \mathcal { H } ,\tag{5}
$$

$$
a _ { i j } ^ { \mathrm { i n s t } } = \sigma ( \frac { z _ { i  j } ^ { \mathrm { i n s t } } + z _ { j  i } ^ { \mathrm { i n s t } } } { 2 } ) .\tag{6}
$$

The two afine trunk layers contain 1,088 and 2,080 parameters, and the three output layers contain 99, giving 3,267 trainable parameters. Primary inference uses only the symmetric same-instance score; the semantic and completeness branches are used only in diagnostic decoder variants.

## 3.4 Instance Grouping, Representative Selection, and Labeling

For relation-head run s, an unordered edge is eligible when $a _ { i j } ^ { \mathrm { i n s t } } \geq \tau _ { \mathrm { i n s t } } ^ { ( s ) }$ , with $\tau _ { \mathrm { i n s t } } ^ { ( s ) }$ frozen on inner validation. For each candidate i, let $T _ { i }$ contain the neighbor indices j of up to two eligible incident edges with the largest instance scores. The top2-OR edge set is

$$
E = \big \{ \{ i , j \} : j \in T _ { i } \ \vee \ i \in T _ { j } \big \} .\tag{7}
$$

An OR rather than mutual selection preserves an edge chosen by either endpoint. Union-find on $E$ produces connected components, including singleton proposals. Each $T _ { i }$ limits direct incident selections but does not bound component size after transitive closure.

Within each component $G ,$ let $s _ { i , ( 1 ) }$ and $s _ { i , ( 2 ) }$ be the largest and second-largest raw category cosines for candidate i. UniCounting selects the maximum-category-margin representative and assigns the component the representative’s highest-scoring category:

$$
m _ { i } = s _ { i , ( 1 ) } - s _ { i , ( 2 ) } , \qquad k _ { G } ^ { \star } = \arg \operatorname* { m a x } _ { i \in G } m _ { i } ,\tag{8}
$$

$$
\hat { c } _ { G } = \arg \operatorname* { m a x } _ { c \in \mathcal { V } } s _ { k _ { G } ^ { \star } c } .\tag{9}
$$

No member averaging, semantic relation, or learned completeness score enters Eqs. 7–9.

## 3.5 Background-Margin Admission

Let $\mathcal { B } _ { \mathrm { b g } } = \{ b _ { 1 } , . . . , b _ { 7 } \}$ denote the fixed background concepts background, an unknown object, other object, texture, shadow, printed text, and empty scene. For $b \in B _ { \mathrm { b g } }$ , the normalized prototype $\mathbf { t } _ { b } ^ { \mathrm { { b g } } }$ is the mean of its normalized fixedtemplate embeddings; the exact templates are listed in the supplement. For each component $G .$ , UniCounting computes the admission margin $g _ { G } .$ , applies its binary admission indicator $A _ { G }$ , and accumulates admitted labels:

$$
g _ { G } = \underset { c \in \mathcal { V } } { \operatorname* { m a x } } s _ { k _ { G } ^ { \star } c } - \underset { b \in \mathcal { B } _ { \mathrm { b g } } } { \operatorname* { m a x } } ( \mathbf { v } _ { k _ { G } ^ { \star } } ^ { \mathsf { T } } \mathbf { t } _ { b } ^ { \mathrm { b g } } ) ,
$$

$$
A _ { G } = { \bf 1 } [ g _ { G } \geq \tau _ { \mathrm { b g } } ^ { ( s ) } ] ,\tag{10}
$$

$$
\hat { n } _ { c } = \sum _ { G } A _ { G } { \bf 1 } [ \hat { c } _ { G } = c ] .\tag{11}
$$

Algorithm 1 UniCounting inference with explicit state and   
outputs.   
Require: Pair scores $a _ { i j } ^ { \mathrm { i n s t } }$ , category scores $s _ { i c } ,$ proposal   
features $\mathbf { v } _ { i }$   
Require: Background prototypes $\{ \mathbf { t } _ { b } ^ { \mathrm { b g } } \} _ { b \in B _ { \mathrm { b g } } }$ , thresholds   
$\tau _ { \mathrm { i n s t } } ^ { ( s ) }$ and $\tau _ { \mathrm { b g } } ^ { ( s ) }$   
Ensure: Count vector nb and admitted components $\{ \widehat { \mathcal { R } } _ { c } \} _ { c \in \mathcal { V } }$   
1: for each candidate i do   
2: $T _ { i } \gets$ the top two j satisfying $a _ { i j } ^ { \mathrm { { i n s t } } } \geq \tau _ { \mathrm { { i n s t } } } ^ { ( s ) }$   
3: end for   
4: $E ~  ~ \{ \{ i , j \} ~ : ~ j ~ \in ~ T _ { i } ~ \lor ~ i ~ \in ~ T _ { j } \}$ and $U $   
UnionFind(N)   
5: for each $\{ i , j \} \in E$ do   
6: U. union(i, j)   
7: end for   
8: $\widehat { \mathbf { n } } \gets \mathbf { 0 } _ { K }$ and $\widehat { \mathcal { R } } _ { c } \gets \emptyset$ for every $c \in \mathcal V$   
9: for each $G \in U .$ components() do   
10: Compute $k _ { G } ^ { \star } , \hat { c } _ { G } ,$ and $g _ { G }$ using Eqs. 8–10   
11: $\mathbf { i f } \ g _ { G } \geq \tau _ { \mathrm { b g } } ^ { ( s ) }$ then   
12: $\hat { n } _ { \hat { c } _ { G } } \gets \hat { n } _ { \hat { c } _ { G } } + 1$ and $\widehat { \mathcal { R } } _ { \hat { c } _ { G } } \gets \widehat { \mathcal { R } } _ { \hat { c } _ { G } } \cup \{ G \}$   
13: end if   
14: end for   
15: return nb, $\{ \widehat { \mathcal { R } } _ { c } \} _ { c \in \mathcal { V } }$

The run-specific threshold $\tau _ { \mathrm { b g } } ^ { ( s ) }$ is frozen on inner validation and is a raw-cosine margin rather than a probability. We omit the superscript (s) when unambiguous. A rejected component contributes zero to every category. Its label is not changed, and its count is not redistributed.

Pair scoring costs $O ( N ^ { 2 } )$ before sparsification. The graph stores at most 2N endpoint selections, and union-find component construction is near-linear in $N + | E |$ . Exact thresholds, index orders, and deterministic tie-breaking rules are specified in the supplement.

## 3.6 Supervision and Optimization

The category head remains fixed, and the relation head is fitted on 210 COCO train2017 images (Lin et al. 2014), with a disjoint 45-image inner-validation split, denoted inner45, reserved for decoder and threshold selection. A proposal is supervised only when one non-crowd instance accounts for at least 0.8 of its mask, the second-best instance accounts for at most 0.1, and ignore overlap is at most 0.1. Reliable proposal pairs receive same-category and same-instance targets; the completeness target is directional and identifies the proposal with greater ground-truth coverage when the containment and coverage-gap conditions hold. Both directions of each pair are used for training, while Algorithm 1 reads only the same-instance branch at inference. Pair supports, the weighted training objective, and optimization settings are specified in the supplement.

Selection uses only inner45. It minimizes complete-vector $\ell _ { 1 }$ for graph construction, representative selection, and labeling, then selects admission by micro presence F1, with vector $\ell _ { 1 }$ breaking ties. No target-test set enters training or selection.

## 4 Experiments

The experiments ask how UniCounting performs under complete-vector counting, whether the relation-head score improves consolidation under a matched decoder, and which partition changes explain its gains and costs. We then analyze the remaining decoder components and their selection trace. All systems are evaluated under the protocol defined in Section 3.1. OWLv2-AllK satisfies the same completevector output requirement, although it receives the fixed Kclass list as an inference input. The Faster R-CNN reference also emits a complete COCO80 vector but uses full COCO box supervision. ABC123 and OCCAM discover anonymous image-specific types, while OmniCount predicts categories requested by a user cue. FSC-147 and CARPK provide only single-coordinate transfer diagnostics under heterogeneous cue protocols, so their results and the full protocol comparison are reported in Supplementary Table $\dot { \mathrm { S 7 } }$ rather than used as primary evidence.

## 4.1 Protocols and Metrics

Model selection and system variants. All decoder choices and thresholds are fixed on a COCO inner-validation split before target-test evaluation. The unqualified name UniCounting refers only to Algorithm 1.

Controlled edge and grouping protocol. We replace only the symmetric edge score with mask containment, mask IoU, CLIP cosine, DINO cosine, or the relation-head score, while fixing the candidate pool, pair universe, top2-OR graph construction, representative rule, labeling rule, and run-matched background-margin admission. Singleton and overlap-guard controls retain the same downstream decoder. Score-specific thresholds are selected only on inner45; exact values and tie-breaking rules are given in the supplement.

Fixed-list detector baseline. OWLv2-All80 receives the same list of 80 raw COCO category names for every image, applies classwise NMS, and counts retained detections. Its score threshold is calibrated only on inner45; checkpoint, preprocessing, search-grid, and tie-breaking details are given in the supplement. Clean500 remains untouched.

Fully supervised detector reference. We report torchvision Faster R-CNN with a ResNet-50-FPN v2 backbone and oficial COCO-trained weights (Ren et al. 2015). Classwise NMS and an inner45-selected score threshold produce a complete COCO80 count vector. Since the model uses COCO box supervision, it provides supervised context rather than a like-for-like comparison to UniCounting.

Primary complete-vector evaluation. COCO clean500 is a deterministic 500-image subset of val2017 (Lin et al. 2014). We rank eligible images by file name, exclude identities from the separate 60-image diagnostic split clean60 defined in the supplement, and take the first 500 without reading annotations or model outputs. Its image hashes are disjoint from relation-head training and inner45 validation, and all thresholds are frozen before preprocessing. Let $\{ ( I _ { i } , \mathbf { y } _ { i } ) \} _ { i = \cdot } ^ { M }$ denote the evaluation set, where $\mathbf { y } _ { i } = ( y _ { i 1 } , \dots , y _ { i K } ) \in \mathbb { N } _ { 0 } ^ { K }$ is the ground-truth count vector and $\hat { \mathbf { y } } _ { i } = \widehat { \mathbf { n } } ( I _ { i } ; \mathcal { V } )$ its prediction; here $ { \mathbb { N } } _ { 0 }$ denotes the nonnegative integers. For clean500, $M = 5 0 0$ and $K = 8 0$ . Each image contains a mean/median of 2.894/2 present categories and 7.018/4 annotated objects; 96.38% of image–class cells are zero, and the largest perclass count is 13.

We report mean vector $\ell _ { 1 }$ error $M ^ { - 1 } \textstyle \sum _ { i } \| \hat { \mathbf { y } } _ { i } - \mathbf { y } _ { i } \| _ { 1 }$ <sub>1</sub> and present-cell MAE over cells with $y _ { i c } ~ > ~ 0$ . Absent-class false mass is $\begin{array} { r } { M ^ { - 1 } \sum _ { i } \sum _ { c : y _ { i c } = 0 } \hat { y } _ { i c } , } \end{array}$ and total-count MAE is $\begin{array} { r } { M ^ { - 1 } \sum _ { i } | \sum _ { c } \hat { y } _ { i c } - \sum _ { c } \hat { y } _ { i c } | } \end{array}$ . We also report macro and micro presence $\bar { \mathrm { F 1 } }$ , where presence means a positive count; macro-F1 averages represented classes, whereas micro-F1 pools all image–category decisions. Full support and bootstrap rules are given in the supplement.

Transfer evaluations. OmniCount-sub is an internally constructed, exact-duplicate-screened 2,000-image holdout sampled from the OmniCount training/validation resources rather than the oficial test split (Mondal et al. 2025). It uses the 191-category taxonomy released publicly with OmniCount; we use PF191 as shorthand for this public, fixed output space. We therefore treat it as a project holdout, not an oficial benchmark. Its labels are not used to train the relation head, select the decoder, or tune admission, and we report vector $\ell _ { 1 }$ error and pooled micro presence F1.

FSC-147 provides all 1,190 test images but only one annotated target category per image (Ranjan et al. 2021). UniCounting predicts the complete 147-dimensional vector without target metadata, after which the evaluator extracts the target coordinate for MAE and RMSE. CARPK is evaluated analogously on its 459 test images as zero-shot transfer to the fixed car coordinate, without fine-tuning (Hsieh, Lin, and Hsu 2017). Because FSC-147 and OmniCount-sub share 26 source-image hashes, we report them separately and do not treat them as independent transfer confirmations.

Statistical reporting and reproducibility. Unless stated otherwise, relation-head rows report mean and sample standard deviation over three relation seeds. Paired 95% bootstrap intervals are computed from saved prediction vectors; resampling, seed aggregation, macro-support, and randomseed details are provided in the supplement.

## 4.2 Implementation Details

We use frozen SAM 2.1 for proposals, DINOv2 for relation features, and OpenCLIP for category features. The relation head scores all $\dot { N } ( N { - } 1 ) / 2$ proposal pairs before graph sparsification, giving $\dot { O } ( N ^ { 2 } )$ relation-scoring cost. Exact checkpoints, candidate filtering thresholds, crop construction, preprocessing, and deterministic inference settings are specified in the supplement.

## 4.3 How does UniCounting perform under complete-vector counting?

Table 1 reports complete-vector performance on COCO clean500. UniCounting is numerically lower than OWLv2- All80 in vector $\ell _ { 1 }$ (6.485 versus 6.860), present-cell MAE (2.129 versus 2.205), and absent-class false mass (0.325 versus 0.474), while their micro-F1 values are nearly identical (0.416 versus 0.415). OWLv2 has lower total-count MAE (5.800 versus 5.868) and higher macro-F1 (0.472 versus 0.407). We therefore describe the 0.375 vector- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 1 }$ gap as a point-estimate diference and make no significance claim. The matched-decoder study below separates the contribution of proposal consolidation from the remaining system components.

<table><tr><td>Method</td><td>Vector  $\ell _ { 1 } \downarrow$ </td><td>Present MAE↓</td><td>False mass↓</td><td>Total MAE↓</td><td>Macro-F1↑</td><td>Micro-F1↑</td></tr><tr><td>Zero vector</td><td>7.018</td><td>2.425</td><td>0.000</td><td>7.018</td><td>0.000</td><td>0.000</td></tr><tr><td>Fixed-list detector (OWLv2-Al180)</td><td>6.860</td><td>2.205</td><td>0.474</td><td>5.800</td><td>0.472</td><td>0.415</td></tr><tr><td>UniCounting</td><td>6.485±0.058</td><td>2.129±0.005</td><td>0.325±0.062</td><td>5.868±0.050</td><td>0.407±0.007</td><td>0.416±0.005</td></tr><tr><td>Faster R-CNN (fully supervised ref.)</td><td>3.098</td><td>0.916</td><td>0.446</td><td>2.078</td><td>0.809</td><td>0.832</td></tr></table>

Table 1: Primary complete-vector results on COCO clean500. Every row predicts the same COCO80 vector. OWLv2 receives the full raw COCO80 list identically for every image, while UniCounting uses a fixed taxonomy and no image-specific cue. Its row reports mean±sample standard deviation over three relation-head seeds; seed-specific thresholds are given in the supplement. The point estimate for UniCounting minus OWLv2 vector $\ell _ { 1 }$ is −0.375 (6.485 versus 6.860). The separated Faster R-CNN row uses full COCO box supervision and is reported only as a supervised reference, not as a like-for-like training-regime comparison.
<table><tr><td>Edge/grouping rule</td><td>T</td><td>Vector  $\ell _ { 1 } \downarrow$ </td><td>False mass↓</td><td>Total MAE↓</td><td>Micro-F1↑</td></tr><tr><td>No grouping / singletons</td><td></td><td>29.187</td><td>20.347</td><td>24.440</td><td>0.319</td></tr><tr><td>Mask containment</td><td>0.005</td><td>9.935</td><td>4.697</td><td>5.020</td><td>0.427</td></tr><tr><td>Mask IoU</td><td>0.005</td><td>13.649</td><td>8.265</td><td>7.965</td><td>0.374</td></tr><tr><td>CLIP cosine</td><td>0.500</td><td>7.335</td><td>1.452</td><td>4.987</td><td>0.431</td></tr><tr><td>DINO cosine</td><td>0.310</td><td>6.814</td><td>0.809</td><td>5.463</td><td>0.434</td></tr><tr><td>Relation-head score + overlap guard</td><td>0.465</td><td>9.597</td><td>4.361</td><td>4.635</td><td>0.414</td></tr><tr><td>Relation-head score (shared  $\tau ,$  controlled)</td><td>0.465</td><td>6.483</td><td>0.326</td><td>5.864</td><td>0.417</td></tr></table>

Table 2: Controlled edge and grouping ablation on COCO clean500. The two relation-head rows are controlled variants rather than the primary UniCounting configuration: both use a common edge threshold τ=0.465, whereas primary UniCounting use seed-specific instance thresholds selected on inner45. The table reports end-task metrics after a shared downstream decoder; full pre-admission partition diagnostics are provided in the supplement. Mask IoU is the binary-mask intersection over union; the overlap guard instead adds a positive-intersection constraint to the relation-head score. Every row is the arithmetic mean of seed-specific point metrics for seeds 17, 42, and 73. For containment, mask IoU, CLIP, DINO, relation-head, singleton, and overlap-guard rows, the candidate pool, pair universe, graph construction, representative and labeling rules, and run-matched background-margin admission are fixed; only the edge score, common edge threshold, or stated grouping control changes.

The fully supervised Faster R-CNN reference reaches 3.098 vector $\ell _ { 1 }$ and 0.832 micro-F1. This gap indicates substantial headroom when COCO box supervision is available, but it does not isolate the contribution of proposal consolidation because the detector and UniCounting use diferent training supervision.

The standalone OmniCount-sub complete-vector comparison is reported in Supplementary Table S5, while FSC-147 and CARPK single-coordinate diagnostics are reported in Supplementary Table S7. These results provide transfer context rather than evidence for the grouping mechanism.

## 4.4 Does the relation-head score improve proposal consolidation under a matched decoder?

Table 2 isolates the edge score under a shared downstream decoder. The relation-head score obtains the lowest vector $\ell _ { 1 }$ and false mass among the tested scores. Its paired vector-$\ell _ { 1 }$ and false-mass intervals against containment, mask IoU,

CLIP, and DINO exclude zero. DINO has higher micro-F1, and the overlap guard has lower total-count MAE. Thus, the relation-head score improves the two complete-vector error measures targeted by this comparison, but not every end-task metric.

## 4.5 Why does it help, and what does it sacrifice?

Figure 3 summarizes the mechanism underlying Table 2. Mask IoU and containment suppress merges more conservatively but leave substantially more instances fragmented. In contrast, the relation-head score shifts the partition toward aggressive consolidation, obtaining the lowest fragmented rate and complete-vector $\ell _ { 1 }$ error among the tested edge scores, while accepting a moderate merge-error increase. It also attains a Pair F1 of 0.694, purity of 0.925, and representative IoU of 0.215; formal definitions and the complete partition diagnostics are provided in Supplementary Table S8.

## 4.6 Decoder Component Ablation and Selection Analysis

Table 3 records the staged inner45 selection trace, with each row carrying forward the choices selected at preceding stages. Figure 4 separates the contribution of each selected decoder choice on the two frozen target sets. Maximum-category-margin representation and representative raw-cosine labeling consistently improve both vector $\ell _ { 1 }$ and micro-F1, with labeling producing the largest efect. Background-margin admission provides a smaller additional gain. In contrast, top2-OR sparsification exhibits metric- and dataset-dependent behavior, supporting selection of the complete pipeline on inner45 rather than retrospective target-set tuning. The complete audit in Supplementary Table S9 further shows that the semantic constraint consistently degrades performance.

![](images/f2b6077f9b49697d7e73d850ca8c24a9f597ae07ce3ec1682b590a08335e4e4b.jpg)

Figure 3: Fragmentation–merge trade-of under the matched decoder on COCO clean500. Each point represents an edge or grouping rule; the horizontal and vertical axes report the fragmented-instance rate and component merge error, respectively, so lower-left is preferred. Point annotations report the resulting complete-vector $\ell _ { 1 }$ error after the shared downstream decoder. The relation-head score produces the lowest fragmentation and vector error among the tested edge scores, while incurring more merges than the conservative mask-containment and mask-IoU rules.
<table><tr><td>Stage</td><td>Selected variant</td><td>Vector  $\ell _ { 1 } \ : \downarrow$  Micro-F1↑</td></tr><tr><td>Graph</td><td>same-instance edges*</td><td> $6 . 8 6 7 { \scriptstyle \pm 0 . 0 6 0 . 4 1 7 \pm 0 . 0 0 5 }$ </td></tr><tr><td></td><td>Represent. max category margin</td><td> $6 . 8 3 7 { \pm } 0 . 0 9 0 . 4 2 3 { \pm } 0 . 0 0 4$ </td></tr><tr><td></td><td>Labeling raw cosines</td><td> $6 . 8 3 7 { \pm } 0 . 0 9 0 . 4 2 3 { \pm } 0 . 0 0 4$ </td></tr><tr><td></td><td>Admission background-margin</td><td> $6 . 7 9 3 { \scriptstyle \pm 0 . 0 5 0 . 4 2 6 } { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr></table>

Table 3: Decoder choices selected on inner45. ⋆ refers to all eligible/top2-OR. Each stage carries forward the choices selected at preceding stages. The table is a staged selection trace rather than a cumulative-improvement analysis. The complete candidate matrix and frozen target-set results are reported in the supplement.

## 5 Discussion and Limitations

The complete-vector evidence is limited to the 500-image COCO evaluation and the project holdout OmniCount-sub, and we do not evaluate MUCCA or PrACo++. clean500 is disjoint from development data and selected without annotations, but remains a subset of COCO val2017, with unknown foundation-model pretraining overlap. The controlled ablation isolates five edge scores and two grouping controls under one decoder. The relation-head score reduces vector error and false mass relative to tested alternatives, but does not minimize total-count MAE; its merge error is higher than mask IoU and containment, while lower than CLIP and DINO. The candidate pool recalls only 62.4% of non-ignore instances at mask IoU 0.50. Since representative selection and admission maximize over V, changing vocabularies shifts score distributions, and reusing COCO-calibrated thresholds may impair transfer calibration.

![](images/0bb05710cc97389d00208e3bb0881d595c70465af81fcbecffc1c4ee3eec0dbc.jpg)

![](images/fdb2cfa3d8afe4d47ae8c6a70bafc7ee2a36d5cb488028843e6ff6b459244b44.jpg)  
Figure 4: Stage-wise efects of the decoder choices selected on inner45 when audited on the frozen clean500 and OmniCount-sub target sets. Panel (a) reports vector- $\cdot \ell _ { 1 }$ improvement, defined as the baseline error minus the selectedvariant error; panel (b) reports micro-F1 improvement, defined as the selected-variant score minus the baseline score. Positive values therefore denote improvement in both panels. Baselines are all-edges grouping, maximum-area representation, component-mean labeling, and no admission, respectively. Representative selection and labeling provide the largest consistent gains, whereas the efect of top2-OR sparsification is metric- and dataset-dependent.

Trainable capacity is limited to the 3,267-parameter relation head, but inference still runs frozen SAM 2.1, DI-NOv2, and OpenCLIP components with quadratic pair scoring. Deployment should further address privacy, ontology omissions, and systematic category errors.

## 6 Conclusion

UniCounting studies image-query-free multi-category counting from the perspective of proposal consolidation. Instead of directly mapping redundant segmentation proposals to category counts, it explicitly models proposal identity, forms instance-level components, and produces category-count outputs with traceable group evidence. A 3,267-parameter category-shared relation head improves proposal partition quality under a controlled decoder comparison, reducing vector-level counting error relative to several similaritybased alternatives. Our analysis also reveals an inherent trade-of in proposal consolidation: more aggressive grouping reduces fragmentation but may increase instance merging errors. These findings suggest that reliable multi-category counting requires not only semantic recognition but also explicit control over instance identity. Future work should improve proposal recall and develop more adaptive mechanisms for balancing merge and fragmentation errors.

## Acknowledgments

This work was supported by the National Natural Science Foundation (No. 62503505) and the Hunan Natural Science Foundation (No. 2025JJ60424).

## References

Amini-Naieni, N.; Amini-Naieni, K.; Han, T.; and Zisserman, A. 2023. Open-world Text-specified Object Counting. In Proceedings of the British Machine Vision Conference (BMVC).

Amini-Naieni, N.; Han, T.; and Zisserman, A. 2024. CountGD: Multi-Modal Open-World Counting. In Advances in Neural Information Processing Systems (NeurIPS), volume 37.

Cherti, M.; Beaumont, R.; Wightman, R.; Wortsman, M.; Ilharco, G.; Gordon, C.; Schuhmann, C.; Schmidt, L.; and Jitsev, J. 2023. Reproducible Scaling Laws for Contrastive Language-Image Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2818–2829.

Ciampi, L.; Messina, N.; Pierucci, M.; Amato, G.; Avvenuti, M.; and Falchi, F. 2025. Mind the Prompt: A Novel Benchmark for Prompt-Based Class-Agnostic Counting. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 7959–7968.

Gao, J.; Zhao, L.; and Li, X. 2024. NWPU-MOC: A Benchmark for Fine-Grained Multicategory Object Counting in Aerial Images. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–14.

Godi, M.; Joppi, C.; Giachetti, A.; and Cristani, M. 2021. SIMCO: SIMilarity-based object COunting. In Proceedings of the International Conference on Pattern Recognition (ICPR), 47–52.

Hobley, M.; and Prisacariu, V. 2023. Learning to Count Anything: Reference-less Class-agnostic Counting with Weak Supervision. In T4V: Transformers for Vision Workshop at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Hobley, M. A.; and Prisacariu, V. A. 2024. ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting. In Proceedings of the European Conference on Computer Vision (ECCV), 304–319.

Hsieh, M.-R.; Lin, Y.-L.; and Hsu, W. H. 2017. Drone-Based Object Counting by Spatially Regularized Regional Proposal Network. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 4145–4153.

Huang, Z.; Dai, M.; Zhang, Y.; Zhang, J.; and Shan, H. 2024. Point, Segment and Count: A Generalized Framework for Object Counting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17067–17076.

Jiang, R.; Liu, L.; and Chen, C. 2023. CLIP-Count: Towards Text-Guided Zero-Shot Object Counting. In Proceedings of the 31st ACM International Conference on Multimedia (MM), 4535–4545.

Kang, S.; Moon, W.; Kim, E.; and Heo, J.-P. 2024. VL-Counter: Text-aware Visual Representation for Zero-Shot Object Counting. Proceedings of the AAAI Conference on Artificial Intelligence, 38(3): 2714–2722.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollár, P.; and Girshick, R. 2023. Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 4015–4026.

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft COCO: Common Objects in Context. In Proceedings ofthe European Conference on Computer Vision (ECCV), 740– 755.

Liu, C.; Zhong, Y.; Zisserman, A.; and Xie, W. 2022. CounTR: Transformer-based Generalised Visual Counting. In Proceedings of the British Machine Vision Conference (BMVC).

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; Zhu, J.; and Zhang, L. 2024. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. In Proceedings of the European Conference on Computer Vision (ECCV), 38–55.

Liu, S.; Zhang, P.; Zhang, S.; and Ke, W. 2025. CountSE: Soft Exemplar Open-set Object Counting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 21536–21546.

Minderer, M.; Gritsenko, A. A.; and Houlsby, N. 2023. Scaling Open-Vocabulary Object Detection. In Advances in Neural Information Processing Systems (NeurIPS), volume 36.

Mondal, A.; Nag, S.; Zhu, X.; and Dutta, A. 2025. OmniCount: Multi-label Object Counting with Semantic-Geometric Priors. Proceedings of the AAAI Conference on Artificial Intelligence, 39(18): 19537–19545.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; Assran, M.; Ballas, N.; Galuba, W.; Howes, R.; Huang, P.-Y.; Li, S.-W.; Misra, I.; Rabbat, M.; Sharma, V.; Synnaeve, G.; Xu, H.; Jégou, H.; Mairal, J.; Labatut, P.; Joulin, A.; and Bojanowski, P. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research.

Pacini, G.; Ciampi, L.; Messina, N.; Tonellotto, N.; Amato, G.; and Falchi, F. 2026. Does it Really Count? Assessing Semantic Grounding in Text-Guided Class-Agnostic Counting. arXiv:2605.02752.

Qian, Y.; Guo, Z.; Deng, B.; Lei, C. T.; Zhao, S.; Lau, C. P.; Hong, X.; and Pound, M. P. 2025. T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 25336–25345.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models from Natural Language Supervision.

In Proceedings ofthe International Conference on Machine Learning (ICML), 8748–8763.

Ranjan, V.; and Hoai, M. 2022. Exemplar Free Class Agnostic Counting. In Proceedings of the Asian Conference on Computer Vision (ACCV), 71–87.

Ranjan, V.; Sharma, U.; Nguyen, T.; and Hoai, M. 2021. Learning to Count Everything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3394–3403.

Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.; Khedr, H.; Rädle, R.; Rolland, C.; Gustafson, L.; Mintun,

E.; Pan, J.; Alwala, K. V.; Carion, N.; Wu, C.-Y.; Girshick, R.; Dollár, P.; and Feichtenhofer, C. 2025. SAM 2: Segment Anything in Images and Videos. In Proceedings ofthe International Conference on Learning Representations (ICLR).

Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 28.

Shi, M.; Lu, H.; Feng, C.; Liu, C.; and Cao, Z. 2022. Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9529–9538.

Spanakis, M.; Oikonomidis, I.; and Argyros, A. A. 2026. OC-CAM: Class-Agnostic, Training-Free, Prior-Free and Multi-Class Object Counting. arXiv:2601.13871.

Ðukić, N.; Lukežič, A.; Zavrtanik, V.; and Kristan, M. 2023. A Low-Shot Object Counting Network with Iterative Prototype Adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 18872– 18881.

Xu, J.; Le, H.; Nguyen, V.; Ranjan, V.; and Samaras, D. 2023. Zero-Shot Object Counting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 15548–15557.

Xu, W.; Liang, D.; Zheng, Y.; Xie, J.; and Ma, Z. 2021. Dilated-Scale-Aware Category-Attention ConvNet for Multi-Class Object Counting. IEEE Signal Processing Letters, 28: 1570–1574.

# Supplementary Materials

Jinshi Liu<sup>1</sup>, Pan Liu<sup>2</sup>, Lei He<sup>3</sup>, Weichao Luo<sup>4,\*</sup>, Rui Qian<sup>5,\*</sup>

<sup>1</sup>Shenzhen University <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>3</sup>Hunan University of Science and Technology <sup>4</sup>Peng Cheng Laboratory <sup>5</sup>Fudan University

<sup>1</sup>jsl@szu.edu.cn, <sup>2</sup>liup28292@gmail.com, <sup>3</sup>helei\_xb@hnust.edu.cn

<sup>4</sup>luowch@pcl.ac.cn, <sup>5</sup>qiianruii@gmail.com

Co-corresponding authors.

This supplement specifies the implementation choices, evaluator definitions, and supporting experiments for the method in the main paper.

Reproducibility Protocol
<table><tr><td>Field</td><td>Configuration</td></tr><tr><td>Training seeds</td><td>17, 42, 73</td></tr><tr><td>COCO (Lin et al. 2014) val2017 release and split</td><td>clean500; fixed</td></tr><tr><td></td><td>SHA256-ranked identities Threshold-selection split disjoint 45-image COCO inner-</td></tr><tr><td></td><td>validation split (inner45)</td></tr><tr><td>seed</td><td>Bootstrap repetitions / 10,000 / 20260725</td></tr></table>

Table S1: Reproducibility settings for training, selection, and evaluation.

Development-split provenance. The training code consumes two fixed COCO train2017 role assignments stored in the source registry: 210 images marked relation\_train and 45 images marked inner\_validation. It does not generate either role by a random draw, a hash rule, or metricbased filtering. The sets contain 210 and 45 unique identities, respectively, with no overlap. SHA-256 of the numerically sorted, newline-delimited, 12-digit image IDs is 8f5728c98bf1 for train210 and 06744dca51b6 for inner45. Train210 contains 11,341 candidates, 3,262 reliable proposals, and 45,508 reliable unordered pairs; inner45 covers 60 of the 80 COCO categories.

COCO clean60 diagnostic set. Clean60 is the fixed 60- image COCO val2017 identity list in the released split file (identity SHA-256 f1fcbdcf). Because its construction does not specify a random-sampling formula, we treat clean60 as a diagnostic set and make no population-level representativeness claim. It is not used for relation-head training, decoder selection, or threshold selection. Its image hashes are disjoint from the 210-image relation-training set and the 45- image inner-validation set, both drawn from train2017, and clean500 is constructed after excluding all clean60 identities. The COCO80 targets contain 340 non-ignore instances from 54 categories and 155 positive image–category cells. Per image, the mean/median non-ignore instance count is 5.667/4 and the mean/median number of present categories is

<table><tr><td>Setting Value</td><td></td></tr><tr><td>SAM 2.1 (Ravi et al. 2025) checkpoint Input resolution Points per side / crop layers</td><td>sam2.1-hiera-small, SHA-256prefix0a4067b1 checkpoint-native preprocessing 24 / 0</td></tr><tr><td>Predicted-IoU thresh- 0.7 old Stability threshold</td><td></td></tr><tr><td>Box NMS threshold Minimum / maxi- 10−4 / 0.95</td><td>0.8</td></tr><tr><td>mum mask area frac- tion Near-duplicate rule</td><td>0.7</td></tr><tr><td></td><td>descending predicted IoU; dis-</td></tr><tr><td>Minimum box side / 4 pixels / none tiling</td><td>card if box IoU &gt; 0.5 and mask IoU &gt; 0.9 against an already- retained mask</td></tr></table>

Table S2: Automatic mask generation and filtering.

2.583/2; the largest single-category count is 13, and 96.771% of the 60 × 80 cells are zero.

Image-query-free protocol. UniCounting follows the runlevel vocabulary protocol of Section 3.1. No per-image category cue is supplied, and FSC-147 target coordinates are extracted only after the full fixed-vocabulary prediction is produced.

FSC-147 annotation boundary. FSC-147 specifies one benchmark target but not every countable category in a scene.

## Implementation Details

## Computing Environment

All experiments were executed in the computing environment summarized below. The CUDA version supported by the NVIDIA driver is distinguished from the CUDA version against which PyTorch was compiled.

(a) Multi-category complete vector  
![](images/a930f39173c78c98f47561319f0ae90c5313ce6a0f9a6a37c2402d8f9902ac43.jpg)  
Figure S1: Selected successful complete-vector predictions on COCO clean500. All examples use the primary UniCounting decoder with the fixed seed-17 checkpoint and thresholds selected on inner45. Colored contours indicate the representatives of admitted proposal components, and the accompanying summaries compare all nonzero ground-truth and predicted coordinates. The examples were filtered using fixed correctness and consolidation-complexity criteria and then selected for visual legibility and diversity; they are intended to illustrate successful behaviors rather than estimate average performance. Ground-truth annotations are used only for visualization and evaluation.

<table><tr><td>Feature Definition</td><td></td><td>Feature Definition</td><td></td></tr><tr><td>1</td><td> $\cos ( \mathbf { d } _ { i } ^ { m } , \mathbf { d } _ { i } ^ { m } )$ </td><td>9</td><td> $\mathrm { c l i p } ( \log ( A _ { i } / A _ { j } ) , - 5 , 5 ) / 5$ </td></tr><tr><td>2</td><td> $\cos ( \mathbf { d } _ { i } ^ { b } , \mathbf { d } _ { i } ^ { b } )$ </td><td>10</td><td> $\operatorname* { m i n } ( A _ { i } , A _ { j } ) / \operatorname* { m a x } ( A _ { i } , A _ { j } )$ </td></tr><tr><td>3</td><td> $\cos ( \mathbf { d } _ { i } ^ { x } , \mathbf { d } _ { i } ^ { x } )$ </td><td>11</td><td> $\mathrm { c l i p } ( A _ { i } / A _ { I } , 0 , 1 )$ </td></tr><tr><td>4</td><td> $\cos ( \mathbf { c } _ { i } ^ { m } , \mathbf { c } _ { i } ^ { m } )$ </td><td>12</td><td> $\mathrm { c l i p } ( A _ { j } / A _ { I } , 0 , 1 )$ </td></tr><tr><td>5</td><td> $\cos ( \mathbf { c } _ { i } ^ { b } , \mathbf { c } _ { j } ^ { b } )$ </td><td>13</td><td>qi</td></tr><tr><td>6</td><td> $\cos ( \mathbf { c } _ { i } ^ { x } , \mathbf { \bar { c } } _ { j } ^ { x } )$ </td><td>14</td><td>qj</td></tr><tr><td>7</td><td> $\cos ( \mathbf { c } _ { i } ^ { m } , \bar { \mathbf { c } } _ { j } ^ { b } )$ </td><td>15</td><td> $| q _ { i } - q _ { j } |$ </td></tr><tr><td>8</td><td> $\cos ( \mathbf { c } _ { i } ^ { b } , \mathbf { c } _ { j } ^ { \bar { m } } )$ </td><td>16</td><td>1</td></tr></table>

## Candidate Generation

The same 24-point grid, zero crop layers, and no-tiling path is applied to every image. Candidates are processed greedily in descending predicted IoU for near-duplicate removal; exact predicted-IoU ties preserve the mask generator’s output order. The formal decoder contains no density, resolution, count-bin, or ground-truth-dependent router.

## Category and Relation Heads

COCO category prototypes average seven templates: a photo of a {}., a photo of the {}., a close-up photo of a {}., a cropped photo of a {}., a photo of one {}., an image of a {}., and a {} in the scene.. FSC-147 and OmniCount use a photo of {phrase}., a close-up photo of {phrase}., multiple {phrase}., a group of {phrase}., and cropped {phrase}.. FSC-147 combines canonical, alias, and description prototype groups with weights 0.50/0.25/0.25; OmniCount uses canonical taxonomy names only. For each of the seven background concepts in $B _ { \mathrm { b g } } , \mathbf { t } _ { b } ^ { \mathrm { b g } }$ averages the normalized embeddings of a photo of {}., a cropped photo of {}., and {} in the scene..

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>GPU / memory</td><td>NVIDIA GeForce RTX 4090 / 24,564 MiB (approximately 24 GB)</td></tr><tr><td>CPU</td><td>Intel Core i9-14900KF</td></tr><tr><td>System memory</td><td>62 GiB available (approximately 64 GB physical memory)</td></tr><tr><td>Operating system Linux kernel</td><td>Ubuntu 24.04.4 LTS under WSL2 6.6.114.1-</td></tr><tr><td>Python</td><td>microsoft-standard-WSL2 3.12.3</td></tr><tr><td>NVIDIA driver / sup- 591.86 / 13.1 ported CUDA</td><td></td></tr><tr><td>PyTorch /1 CUDA</td><td>build 2.12.0+cu130/13.0</td></tr><tr><td>torchvision/cuDNN 0.27.0+cu130/9.2.0</td><td></td></tr></table>

Hardware and software environment used for the experiments.

Directed relation features. Let ${ \bf d } _ { i } ^ { v }$ and $\mathbf { c } _ { i } ^ { v }$ be the L2- normalized DINOv2 (Oquab et al. 2024) and OpenCLIP (Radford et al. 2021; Cherti et al. 2023) embeddings for view $v \in \{ m , b , x \}$ (masked, box, and context), let $A _ { i }$ be mask area in pixels, and let $A _ { I }$ be image area. We define $q _ { i }$ as the SAM 2.1 predicted-IoU score returned for candidate i. The stability score is used only for the 0.8 candidatefiltering threshold in Table S2 and is not part of $q _ { i }$ . With $\cos ( \mathbf { a } , \mathbf { \bar { b } } ) = \mathbf { a } ^ { \mathsf { T } } \mathbf { b }$ , the feature order for $i  j$ is fixed as follows. The reverse feature $\phi _ { j  i }$ is constructed by exchanging i and $j$ in every row; in particular, Features 7–8, 11–12, and 13–14 exchange positions, while Feature 9 changes sign before clipping. Each dimension is centered by its relationtraining mean and divided by its relation-training standard deviation after clamping that denominator below at $1 0 ^ { - 4 }$ Feature 16 is therefore identically zero after standardization. It is retained only as an inactive compatibility placeholder and does not afect the logits; constant ofsets are already supplied by the MLP biases.

## Pseudo-Label Construction

For candidate $m _ { i }$ and non-ignore ground-truth instance $G _ { u } .$ define intersection-over-union, candidate purity, and groundtruth coverage as

$$
\begin{array} { c } { { \mathrm { I o U } _ { i u } = \displaystyle \frac { \left| m _ { i } \cap G _ { u } \right| } { \left| m _ { i } \cup G _ { u } \right| } , ~ \displaystyle \pi _ { i u } = \displaystyle \frac { \left| m _ { i } \cap G _ { u } \right| } { \left| m _ { i } \right| } , } } \\ { { \kappa _ { i u } = \displaystyle \frac { \left| m _ { i } \cap G _ { u } \right| } { \left| G _ { u } \right| } . } } \end{array}\tag{S1}
$$

The matched instance is $u _ { i } = \arg \operatorname* { m a x } _ { u } \pi _ { i u } ,$ with the lower stored GT index breaking exact ties, and $\pi _ { i , ( 2 ) }$ denotes the second-largest non-ignore purity. Candidate i is reliable exactly when $\pi _ { i u _ { i } } \geq 0 . 8 , \pi _ { i , ( 2 ) } \leq 0 . 1$ , and its total overlap with ignore or crowd regions is at most $0 . 1 | m _ { i } |$ . Samecategory and same-instance pair targets compare the category and instance identities of reliable candidates. A directed completeness positive additionally requires both reliable proposals to match the same instance, containment $| m _ { i } \cap \mathop { m _ { j } } | / \operatorname* { m i n } ( | m _ { i } | , | m _ { j } | ) \geq 0 . 9 $ , and a GT-coverage advantage of at least 0.2 for the larger-coverage proposal. No additional proposal-status classes or heuristic proposal labels are introduced.

<table><tr><td>Edge score</td><td>threshold vector</td><td> $\ell _ { 1 } \downarrow$ </td><td>micro-F1↑</td></tr><tr><td>Mask containment</td><td>0.005</td><td>12.5333</td><td>0.4289</td></tr><tr><td>Mask IoU</td><td>0.005</td><td>15.8222</td><td>0.3920</td></tr><tr><td>CLIP cosine</td><td>0.500</td><td>7.2444</td><td>0.4632</td></tr><tr><td>DINO cosine</td><td>0.310</td><td>7.1185</td><td>0.4160</td></tr><tr><td>Relation-head score</td><td>0.465</td><td>6.7930</td><td>0.4260</td></tr></table>

Table S4: Frozen threshold selection for the controlled edge-score study in inner45. Mask containment is $| m _ { i } \cap$ $m _ { j } \mathbf { \bar { | } } / \operatorname* { m i n } ( | m _ { i } | , \mathbf { \bar { | } } m _ { j } | )$ , whereas mask IoU is $| m _ { i } \cap m _ { j } | / | m _ { i } \cup$ $m _ { j } |$ . CLIP and DINO use cosine similarity between normalized 0.05 masked-view plus 0.95 box-view features.
<table><tr><td>Method</td><td>Vector  $\ell _ { 1 } \downarrow$ </td><td>Micro-F1↑</td></tr><tr><td>Zero vector</td><td>6.447</td><td>0.000</td></tr><tr><td>UniCounting</td><td> $6 . 7 1 5 { \scriptstyle \pm 0 . 1 2 3 }$ </td><td>0.283±0.011</td></tr></table>

Table S5: OmniCount-sub complete-vector diagnostic. Both rows predict the same 191-dimensional vector. OmniCountsub is a project holdout used only as a transfer diagnostic, not as an oficial benchmark.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>DINOv2 checkpoint</td><td>facebook/dinov2-small, re- vision prefix ed25f3a</td></tr><tr><td>CLIP checkpoint</td><td>ViT-B-32 laion2b_s34b_b79k,snapshot prefix  $1 a 2 5 a 4 4$ </td></tr><tr><td>Masked background fill</td><td>black</td></tr><tr><td>Context expansion / CLIP resolution</td><td>half a box width and height / 224× 224</td></tr><tr><td>Proposal feature weights</td><td>0.05 masked + 0.95 box, then L2 nor- malization</td></tr><tr><td>Activation / normaliza- tion / dropout</td><td>GELU / training-set standardization /0.1</td></tr><tr><td>Relation feature dimen- sion</td><td>16 directed pair features</td></tr><tr><td>Relation MLP dimen- sions / parameters</td><td>16 → 64 → 32, three biased 32 → 1</td></tr><tr><td></td><td>outputs/1,088+2,080+99 = 3,267</td></tr><tr><td>Pair rows / pruning</td><td>both directions of every reliable un- ordered pair / none before scoring</td></tr></table>

Table S3: Feature extraction and trainable-head architecture.

## Training and Decoding

The category head remains fixed. Every reliable unordered pair $\{ i , \bar { j } \}$ contributes the directed rows $\phi _ { i \to j }$ and $\phi _ { j  i } ,$ giving 91,016 rows from 45,508 pairs. Same-category and same-instance targets are duplicated across directions, so their directed positive/negative supports are 74,838/16,178 and 45,628/45,388. Proposal-completeness targets remain directional, with 1,269 positives and 89,747 negatives: $y _ { i \to j } ^ { \mathrm { c o m p } } = 1$ only when j is the higher-coverage proposal satisfying the containment and coverage-gap rule. For branch $r \in$ {sem, inst, comp}, let $\mathbf { z } ^ { r }$ and $\mathbf { y } ^ { r }$ collect the logits and targets of its directed rows, and let $\dot { N } _ { r } ^ { + }$ and $N _ { r } ^ { - }$ denote its positive and negative supports. The BCEWithLogits call uses pos\_weight $= \lambda _ { r } .$ , where $\lambda _ { r } = N _ { r } ^ { - } / N _ { r } ^ { + }$ . Writing this weighted loss as BCEWithLogits , the training objective is

<table><tr><td>Method / seed</td><td>TP FP FN Micro-P Micro-R Micro-F1</td><td></td><td></td></tr><tr><td>OWLv2-All80 432 202</td><td>1015 0.681</td><td>0.299</td><td>0.415</td></tr><tr><td>UniCounting / 17 429 187 1,018</td><td>0.696</td><td>0.296</td><td>0.416</td></tr><tr><td>UniCounting / 42 4181651,029</td><td>0.717</td><td>0.289</td><td>0.412</td></tr><tr><td>UniCounting / 73 420 125 1,027</td><td>0.771</td><td>0.290</td><td>0.422</td></tr></table>

Table S6: Pooled image–category presence diagnostics on COCO clean500. TP, FP, and FN are reported for each prediction vector; precision, recall, and F1 are computed from those pooled counts. Its low micro-F1 is primarily associated with low micro recall rather than excessive false-positive presence.

$$
\begin{array} { c } { \displaystyle \ell _ { r } = \mathrm { B C E W i t h L o g i t s } _ { \boldsymbol \lambda _ { r } } ( \mathbf { z } ^ { r } , \mathbf { y } ^ { r } ) , } \\ { \displaystyle \mathcal { L } _ { \mathrm { r e l } } = \sum _ { r \in \mathrm { \scriptsize ~ \left\{ s e m , i n s t , c o m p \right\} ~ } } \ell _ { r } . } \end{array}\tag{S2}
$$

Binary cross-entropy is applied to every directional logit before any direction averaging. The relation head is trained for 80 deterministic full-batch epochs with AdamW, learning rate $3 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and dropout 0.1. At inference, the same-category and same-instance logits are averaged across directions before the sigmoid, whereas completeness retains both directions for its diagnostic representative rule. Decoder choices and thresholds are selected only on the disjoint inner45 split using the staged criteria in the main paper.

UniCounting uses the same-instance score from the jointly trained relation head, top2-OR sparsification, unionfind components, maximum-category-margin representative selection, representative raw-cosine labeling, and sevenconcept background-margin admission. For relation-head seeds 17, 42, and 73, $\tau _ { \mathrm { i n s t } } ^ { ( s ) }$ is 0.410, 0.400, and 0.400, and $\tau _ { \mathrm { b g } } ^ { ( s ) }$ is −0.045626, −0.035712, and −0.017443. An admitted component adds one to its assigned category; a rejected component adds zero everywhere. Candidate indices follow fixed cache order, pair indices follow ascending uppertriangular enumeration, and vocabulary indices follow the released class list. Exact ties use the lower pair, candidate, or vocabulary index, respectively; equality at either threshold is accepted, and all eligible edges are retained when an endpoint has fewer than two.

Diagnostic decoder definitions. Let $a _ { i j } ^ { \mathrm { s e m } }$ and $a _ { i j } ^ { \mathrm { i n s t } }$ be the direction-averaged sigmoid outputs. The “hard semantic ∧ instance” graph admits pair $\{ i , j \}$ exactly when $a _ { i j } ^ { \mathrm { s e m } } ~ \geq ~ \tau _ { \mathrm { s e m } } ^ { ( s ) }$ and $a _ { i j } ^ { \mathrm { i n s t } } ~ \geq ~ \tau _ { \mathrm { i n s t } } ^ { ( s ) }$ , ranks eligible edges by $\sqrt { a _ { i j } ^ { \mathrm { s e m } } a _ { i j } ^ { \mathrm { i n s t } } }$ , and then applies top2-OR. For seeds 17/42/73, $\tau _ { \mathrm { { s e m } } } ^ { ( s ) }$ is 0.920/0.940/0.940 and $\tau _ { \mathrm { i n s t } } ^ { ( s ) }$ is 0.410/0.400/0.400. Exact edge-score ties use the lower ascending upper-triangular pair index.

For learned-completeness representative selection, let $p _ { i \to j } ^ { \mathrm { c o m p } } = \sigma ( z _ { i \to j } ^ { \mathrm { c o m p } } )$ , where σ is the logistic sigmoid. Within a component G, each candidate receives

$$
R _ { i } = \sum _ { j \in G \backslash \{ i \} } ( p _ { j  i } ^ { \mathrm { c o m p } } - p _ { i  j } ^ { \mathrm { c o m p } } ) ,\tag{S3}
$$

and the representative maximizes $R _ { i } ;$ ties are broken by the candidate’s largest raw category cosine and then by the negative candidate index. Thus, exact remaining ties select the lower candidate index.

For component G, define the representative-evidence vector $\begin{array} { c c c } { { { \bf s } _ { G } } } & { { = } } & { { ( s _ { k _ { G } ^ { \star } c } ) _ { c \in \mathcal V } } } \end{array}$ , and let $s _ { G , ( 1 ) }$ and $s _ { G , ( 2 ) }$ be its largest and second-largest entries. The cosine– margin conjunction admits G when $\begin{array} { r l r } { s _ { G , ( 1 ) } } & { { } \ge } & { \tau _ { \mathrm { c o s } } ^ { ( s ) } } \end{array}$ and $s _ { G , ( 1 ) } ~ - ~ s _ { G , ( 2 ) } ~ \geq ~ \tau _ { \mathrm { m a r } } ^ { ( s ) }$ . For seeds 17/42/73, $( \tau _ { \cos } ^ { ( s ) } , \tau _ { \mathrm { m a r } } ^ { ( s ) } )$ is (0.243097, 0.011030), (0.235304, 0.006040), and (0.243339, 0.007480). For component-support F1, one decoded component is one binary item: its target is positive exactly when it contains at least one reliable proposal, and its prediction is positive exactly when the conjunction admits it. Component-level TP, FP, and FN counts are pooled across inner45. Each threshold pair is selected from the 41-quantile Cartesian grid by this F1, then precision, then the smaller sum of the two thresholds. Across admission rules, mean inner45 micro-F1 is the selection metric and lower vector $\ell _ { 1 }$ is the tie-break.

## Extended Results

## Evaluation Splits and Detector Protocols

Clean500 is formed by sorting the 4,940 eligible val2017 file names by SHA256(salt|file\_name), after excluding the clean60 diagnostic identities, and taking the first 500. The salt is $\mathtt { e x p \mathrm { - v 1 0 \mathrm { - } c o c o \mathrm { - } v a l \mathrm { - } c l e a n 5 0 0 \mathrm { - } v 1 \mathrm { - } 2 0 2 6 0 7 2 5 } }$

The selection procedure reads no annotation, prediction, or metric, and clean500 overlaps clean60 in zero images. All thresholds are selected on inner45 and frozen before preprocessing any clean500 or transfer-set image. Ground-truth annotations are loaded only after the prediction files have been finalized.

Unless a table states otherwise, relation-head rows report the mean and sample standard deviation of the three relationseed point estimates. Clean500 macro-F1 support contains 78 of the 80 COCO categories; toaster and hair drier are excluded because they have no positive ground-truth cells. Each paired confidence interval uses 10,000 image-bootstrap replicates with seed 20260725. Within a replicate, the same resampled image multiplicities are applied to every method and relation seed, each metric is recomputed separately for each seed, and the seed-wise metrics are then averaged. Macro-F1 support is recomputed within every replicate, and categories with zero positives in that replicate are excluded from its

<table><tr><td>Grouping rule</td><td>Pair F1↑</td><td>Purity↑</td><td>Merge error↓</td><td>Comp./instance↓</td><td>Fragmented↓</td><td>Rep. IoU↑</td></tr><tr><td>No grouping / singletons</td><td>0.0000</td><td>1.0000</td><td>0.0000</td><td>4.0770</td><td>0.6320</td><td>0.1680</td></tr><tr><td>Mask containment</td><td>0.6910</td><td>0.9090</td><td>0.1150</td><td>1.4150</td><td>0.1350</td><td>0.2020</td></tr><tr><td>Mask IoU</td><td>0.4984</td><td>0.9558</td><td>0.0515</td><td>1.8104</td><td>0.2282</td><td>0.1758</td></tr><tr><td>CLIP cosine</td><td>0.5040</td><td>0.6740</td><td>0.4680</td><td>1.6320</td><td>0.4300</td><td>0.1450</td></tr><tr><td>DINO cosine</td><td>0.6270</td><td>0.6510</td><td>0.5060</td><td>1.1950</td><td>0.1690</td><td>0.1760</td></tr><tr><td>Relation-head score + overlap guard</td><td>0.6820</td><td>0.8500</td><td>0.1660</td><td>1.4180</td><td>0.1830</td><td>0.2030</td></tr><tr><td>Relation-head score</td><td>0.6940</td><td>0.9250</td><td>0.1550</td><td>1.0370</td><td>0.0360</td><td>0.2150</td></tr></table>

Table S7: Full pre-admission partition diagnostics for the controlled edge and grouping ablation on COCO clean500. Rows follow Table 2 of the main paper. Every value is the arithmetic mean of seed-specific point metrics for seeds 17, 42, and 73. The candidate pool, pair universe, graph construction, representative rule, and labeling rule are fixed; only the edge score, common edge threshold, or stated grouping control changes. Admission is not applied for these partition diagnostics.

(a) Protocol comparison
<table><tr><td>Method</td><td>Inference-time category input</td><td>Category space</td><td colspan="2">Output form</td><td>Direct complete-vector comparison</td></tr><tr><td>ABC123 (Hobley and Prisacariu 2024)</td><td>None</td><td>Image-specific discovered types</td><td colspan="2">Anonymous multi-type counts</td><td>No</td></tr><tr><td>OCCAM (Spanakis, Oikono- midis, and Argyros</td><td>None</td><td>Image-specific discovered types</td><td colspan="2">Anonymous multi-type counts</td><td>No</td></tr><tr><td>2026) OmniCount</td><td>Image-dependent category or point cue</td><td colspan="2">Queried semantic categories</td><td colspan="2">Named queried counts</td></tr><tr><td>OWLv2 (AllK) UniCounting</td><td>Fixed K-class list None</td><td colspan="2">Run-level global named vocabulary Run-level global named</td><td colspan="2">Complete K-vector after detection</td></tr><tr><td colspan="5">Complete K-vector with vocabulary group evidence (b) Cross-protocol single-coordinate diagnostics</td></tr><tr><td colspan="2">Method Per-image cue</td><td colspan="2">FSC-147 MAE↓ RMSE↓</td><td colspan="2">CARPK (Hsieh, Lin, and Hsu 2017)</td></tr><tr><td colspan="2">CountGD (Amini-Naieni, Han, Image-dependent and Zisserman 2024) text/exemplar OmniCount (Mondal et al. 2025) Image-dependent</td><td colspan="2">5.74</td><td colspan="2">MAE↓ RMSE↓</td></tr><tr><td colspan="2" rowspan="2"></td><td colspan="2">point prompt</td><td colspan="2">24.09 3.68 5.17</td></tr><tr><td colspan="2">19.24 115.27</td><td>10.66 13.15</td></tr><tr><td colspan="2">OCCAM-S (Spanakis, none Oikonomidis, and Argyros 2026) UniCounting none</td><td>16.92</td><td>110.83 26.409±0.008 120.717±0.006 4.057</td><td>10.06 13.81</td></tr></table>

Table S8: Protocol comparison and cross-protocol transfer diagnostics. UniCounting follows the run-level vocabulary protocol defined in Section 3.1 of the main paper. Panel (a) distinguishes the inference-time category input and output space of each method; direct comparability means evaluation against a complete vector over a run-level global named vocabulary. ABC123 is omitted from panel (b) because its reported FSC-147 transfer result uses only images containing fewer than 300 objects rather than the full test split. OCCAM-S is the single-class OCCAM configuration reported by Spanakis, Oikonomidis, and Argyros (2026). UniCounting’s FSC-147 values are mean±sample standard deviation over three relation-head seeds. CARPK is reported from a single deterministic run; seed variability was not evaluated. These are single-coordinate transfer diagnostics rather than complete-vector evaluations; methods difer in their inference-time category inputs.

<table><tr><td></td><td></td><td colspan="2">Inner45</td><td colspan="2">COCO clean500</td><td colspan="2">OmniCount-sub 2K</td></tr><tr><td>Stage</td><td>Variant</td><td> $\ell _ { 1 } \downarrow$ </td><td>Micro-F1↑</td><td> $\ell _ { 1 } \downarrow$ </td><td>Micro-F1↑</td><td> $\ell _ { 1 } \downarrow$ </td><td>Micro-F1↑</td></tr><tr><td rowspan="3">Graph</td><td>hard semantic ∧ instance</td><td></td><td> $3 2 . 9 7 8 \pm 0 . 8 6 8 ~ 0 . 3 0 0 \pm 0 . 0 0 3 ~ 3 0 . 5 9 7 \pm 1 . 0 4 0 ~ 0 . 2 9 1 \pm 0 . 0 0 3 ~ 2 6 . 5 5 3 \pm 1 . 0 7 3 ~ 0 . 1 4 4 \pm 0 . 0 0 3$ </td><td></td><td></td><td></td><td></td></tr><tr><td>same-instance edges, all</td><td>6.889±0.000 0.365±0.000 6.686±0.000 0.342±0.000 6.769±0.000 0.257±0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>same-instance edges, top2*</td><td>6.867±0.0590.417±0.0056.810±0.010 0.350±0.0026.976±0.0090.250±0.002</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td>maximum area</td><td>7.296±0.078 0.332±0.0086.837±0.011 0.345±0.002 7.293±0.012 0.144±0.003</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Represent. learned completeness</td><td>7.400±0.102 0.323±0.007 6.858±0.003 0.339±0.003 7.305±0.0140.141±0.003</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>max-category-margin*</td><td>6.837±0.0900.423±0.0046.549±0.0160.413±0.0046.930±0.011 0.265±0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Labeling</td><td>component-mean category</td><td>7.667±0.059 0.262±0.003 7.227±0.029 0.246±0.005 7.562±0.012 0.053±0.004</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>representative raw cosines*</td><td>6.837±0.0900.423±0.0046.549±0.0160.413±0.0046.930±0.011 0.265±0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td>none</td><td>6.837±0.0900.423±0.0046.549±0.0160.413±0.0046.930±0.011 0.265±0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Admission background-margin* cosine + margin conjunction 6.785±0.046 0.424±0.004 6.467±0.014 0.412±0.0006.680±0.018 0.287±0.002</td><td>6.793±0.0510.426±0.0076.485±0.0580.416±0.0056.715±0.123</td><td></td><td></td><td></td><td></td><td> $0 . 2 8 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr></table>

Table S9: Complete decoder selection and frozen target-set audit. Inner45 is the only split used for selecting graph construction, representative, labeling, and admission variants. In the graph block, the representative is maximum-category-margin, labeling uses its raw cosines, and admission is disabled. The representative block carries forward the starred graph and uses representative raw-cosine labeling without admission; the labeling block additionally carries forward the starred representative; and the admission block carries forward all preceding starred choices. The clean500 and OmniCount-sub columns are reported only after all choices are frozen. Stars denote the variants selected on inner45; target-set winners are not used for model selection.

macro average. The reported 95% interval is the percentile interval over the seed-averaged replicates.

OWLv2-All80 (Minderer, Gritsenko, and Houlsby 2023) uses the fixed google/owlv2-base-patch16-ensemble checkpoint at the revision whose SHA begins cfd3195b; the full hash is provided with the accompanying artifacts. All 80 raw oficial COCO thing-category names are passed in one identical list for every image, without aliases or a prompt ensemble. Snapshot-native 960 × 960 preprocessing and processor decoding are followed by classwise batched\_nms at IoU 0.50. Every detection with score strictly above 0.350 contributes one count. The threshold is selected from 201 points on inner45 by minimum vector $\ell _ { 1 } .$ , then maximum micro-F1 and the higher threshold as ties. Its inner45 vector $\ell _ { 1 }$ is 3.6889 and micro-F1 is 0.7843. Clean500 is not used for calibration.

The fully supervised reference uses torchvision Faster R-CNN with a ResNet-50-FPN v2 backbone and the FasterRCNN\_ResNet50\_FPN\_V2 COCO\_V1 weights (Ren et al. 2015). The released checkpoint SHA-256 begins dd69338a. Oficial preprocessing is followed by explicit classwise NMS at IoU 0.50, and detections are mapped from the torchvision label space to the canonical COCO80 coordinates. A strict score threshold of 0.775 is selected from 191 points on inner45 by minimum complete-vector $\ell _ { 1 } ,$ then maximum micro-F1 and the higher threshold as ties. Its inner45 vector $\ell _ { 1 }$ is 2.3111 and micro-F1 is 0.8800. No clean500 annotation or metric is accessed before its predictions are sealed. This model uses full COCO box supervision and is therefore a supervised reference rather than a matchedtraining baseline.

## Controlled Edge and Grouping Ablation

The controlled study fixes the candidate pool and unorderedpair universe together with the top2-OR rule, connectedcomponent construction, maximum-category-margin representative rule, representative raw-cosine labeling rule, and run-matched background-margin admission rule. Five score rows apply these procedures after replacing only the pair score and its scalar threshold. The singleton control removes every edge, while the overlap guard additionally requires strictly positive mask intersection. A single threshold shared across seeds 17, 42, and 73 is selected for each score on the same 45-image inner-validation split. Selection minimizes three-seed mean vector $\ell _ { 1 } ,$ , with mean micro presence F1 and then the higher threshold used as ties. The target tests are not consulted.

Table 2 of the main paper reports end-task metrics for the five edge scores together with the singleton and overlapguard controls, while Table S8 reports their full partition diagnostics. Mask IoU uses the common threshold 0.005 selected in Table S4. The relation-head score uses the common threshold τ = 0.465 selected in the same table, whereas UniCounting uses the run-specific thresholds listed above.

Grouping diagnostic protocol. All grouping metrics are computed before category admission. Ground-truth masks are used only for post hoc evaluation, not for prediction. Let R denote candidates satisfying the reliable-match rule above, retain $u _ { i }$ for the matched non-ignore GT instance of reliable candidate i, and let C be the component partition. For $C \in { \mathcal { C } } ,$ define $\mathcal { R } _ { C } = C \cap \mathcal { R }$ and $n _ { C u } = | \{ i \in \mathcal { R } _ { C } : u _ { i } = u \}$ |. Only components in $\mathcal { C } _ { R } = \{ C : | \mathcal { R } _ { C } | > 0 \}$ enter purity and merge error.

For Pair F1, every unordered pair of reliable candidates is GT-positive when its two matched-instance indices agree and predicted-positive when both candidates lie in the same component. Counts are pooled across all images before computing

$$
\begin{array} { l c r } { \displaystyle { P _ { \mathrm { p a i r } } = \frac { T P } { T P + F P } , } } \\ { \displaystyle { R _ { \mathrm { p a i r } } = \frac { T P } { T P + F N } , } } \\ { \displaystyle { F _ { \mathrm { p a i r } } = \frac { 2 T P } { 2 T P + F P + F N } . } } \end{array}\tag{S4}
$$

Let $d _ { C } = | \{ u : n _ { C u } > 0 \}$ |. Component purity and merge error are

$$
\begin{array} { r } { \mathrm { P u r i t y } = \frac { \sum _ { C \in \mathcal { C } _ { R } } \operatorname* { m a x } _ { u } n _ { C u } } { \sum _ { C \in \mathcal { C } _ { R } } | \mathcal { R } _ { C } | } , } \\ { \mathrm { M e r g e } = \frac { \sum _ { C \in \mathcal { C } _ { R } } \mathbf { 1 } [ d _ { C } > 1 ] } { | \mathcal { C } _ { R } | } . } \end{array}\tag{S5}
$$

Thus, purity is weighted by reliable-candidate count, whereas merge error gives each evaluable component one vote. Let $\mathcal { U } = \{ u _ { i } : i \in \mathcal { R } \}$ be the supported GT instances and $k _ { u } =$ $| \{ C : \dot { n } _ { C u } > 0 \} |$ . Components per instance and Fragmented are

$$
\begin{array} { r l } & { \mathrm { C o m p / i n s t . } = \cfrac { \sum _ { u \in \mathcal { U } } k _ { u } } { | \mathcal { U } | } , } \\ & { \mathrm { F r a g m e n t e d } = \cfrac { \sum _ { u \in \mathcal { U } } \mathbf { 1 } [ k _ { u } > 1 ] } { | \mathcal { U } | } . } \end{array}\tag{S6}
$$

Their denominator excludes annotated instances with no reliable candidate. For every component, define

$$
\begin{array} { r } { u _ { C } ^ { \star } = \arg \underset { u \mathrm { \ n o n i g n o r e } } { \operatorname* { m a x } } \sum _ { i \in C } \vert m _ { i } \cap G _ { u } \vert , } \\ { \mathcal { C } _ { \mathrm { I o U } } = \left\{ C : \displaystyle \sum _ { i \in C } \vert m _ { i } \cap G _ { u _ { C } ^ { \star } } \vert > 0 \right\} , } \end{array}\tag{S7}
$$

with the lower stored GT index breaking exact ties. $\operatorname { I f } r ( C )$ is the selected representative, the reported aggregate is

$$
\mathrm { R e p I o U } = \frac { 1 } { | \mathcal { C } _ { \mathrm { I o U } } | } \sum _ { C \in \mathcal { C } _ { \mathrm { I o U } } } \mathrm { I o U } ( m _ { r ( C ) } , G _ { u _ { C } ^ { \star } } ) .\tag{S8}
$$

Thus, each evaluable component across all images receives one vote. Pair counts are pooled before F1, and each reported relation-head metric is computed per seed and then averaged over the three relation-head seeds.

Candidate recall uses all retained pre-grouping proposals, without the reliable-candidate filter. For non-ignore groundtruth instance $G _ { u }$ , let $b _ { u } = \operatorname* { m a x } _ { i }$ IoU $( m _ { i } , G _ { u } )$ over every candidate $m _ { i }$ from the same image. At threshold t, the reported micro recall is

$$
R _ { \mathrm { c a n d } } ( t ) = \frac { \sum _ { u \in \mathcal { G } } \mathbf { 1 } [ b _ { u } \geq t ] } { | \mathcal { G } | } ,\tag{S9}
$$

where G pools all non-ignore instances across clean500. Thus 62.4% at $t = 0 . 5 0$ and 48.8% at $t = 0 . 7 5$ measure the candidate-generation ceiling before edge scoring, grouping, representative selection, labeling, or admission.

## Complete-Vector Transfer Diagnostic

OmniCount-sub is a sparse transfer diagnostic in which the zero vector obtains lower vector $\ell _ { 1 }$ by predicting no positive coordinates, while UniCounting recovers nonzero category presence. This contrast shows why vector error should be interpreted jointly with presence F1.