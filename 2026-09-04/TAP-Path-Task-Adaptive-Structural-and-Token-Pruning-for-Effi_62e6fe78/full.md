# TAP-Path: Task-Adaptive Structural and Token Pruning for Efficient and Trustworthy Pathology Foundation Models

Mehedi Hasan<sup>1</sup> Ashfak Yeafi<sup>2</sup> Md Khairul Islam<sup>3</sup>

<sup>1</sup>Department of Computer Science and Engineering, Brac University, Dhaka, Bangladesh <sup>2</sup>EEE, Khulna University of Engineering & Technology, Khulna, Bangladesh <sup>3</sup>Mathematics and Computer Science, Hobart and William Smith Colleges, Geneva, NY, USA

mehedi.hasan1@g.bracu.ac.bd yeafiashfak@gmail.com khairul.robotics@gmail.com

## Abstract

Pathology foundation models have substantially improved transferable representation learning for histopathology, but recent gains increasingly rely on encoders with hundreds of millions of parameters and high inference cost. This creates a practical mismatch between representational scale and task-specific deployment, especially when a downstream application requires only a subset of the pretrained hierarchy. We propose TAP-Path, a task-adaptive compression framework that directly restructures a pretrained Virchow2 encoder rather than distilling it into a separate student. TAP-Path com bines validation-driven transformer-block selection, physical removal of redundant blocks, input-adaptive patch-token prun ing, multi-depth feature recovery, and a lightweight gated task head. The final model retains 24 of 32 transformer blocks and 70% of patch tokens after the pruning point, reducing encoder parameters by 24.96% (631.24M to 473.70M) and analytical encoder compute by 35.20% (340.13G to 220.40G FLOPs). Across three task-head optimization seeds, TAP-Path achieved 87.98±0.067% test accuracy, 81.26±0.49% balanced accuracy, and 82.38±0.48% macro-F1 on a 32-class histopathology benchmark, slightly exceeding full Virchow2 (86.89% accuracy) and UNI2-h (87.67%) while using fewer deployed parameters and substantially less analytical compute. Beyond predictive performance, we evaluate calibration, probabilistic quality, failure detection, rare-class behavior, and selective pre diction. TAP-Path obtained a Brier score of 0.1800±0.0005 and failure-detection AUROC of 0.9047±0.0060. A validationonly rare-aware objective increased rare-class balanced accuracy in a secondary operating analysis, providing a complementary operating point that characterizes the trade-off between overall accuracy and minority-class sensitivity. Frozen external evaluation on 433 CPTAC samples yielded 91.22±0.83% accuracy and 91.10±0.81% balanced accuracy. These results indicate that task-adaptive structural and token sparsification can move large pathology foundation models toward a more favorable accuracy–efficiency operating point while preserving measurable reliability characteristics under internal and external evaluation.

Keywords: Computational pathology; pathology foundation models; structural pruning; token pruning; trustworthy AI;

model efficiency.

## 1 Introduction

Digital pathology has moved rapidly from task-specific convolutional pipelines toward large pretrained vision and vision–language foundation models. Models such as UNI, CONCH, Virchow, Prov-GigaPath, CHIEF, Hibou, GPFM, and PathOrchestra have demonstrated that large-scale selfsupervised, weakly supervised, or multimodal pretraining can produce reusable histopathology representations across cancer classification, biomarker prediction, retrieval, prognosis, and other downstream tasks [1, 2, 3, 4, 5, 6, 7, 8]. The emerging consensus, however, is not that a single foundation model is uniformly superior. Independent benchmarking has shown substantial task dependence, cohort dependence, sensitivity to adaptation strategy, and variation under external evaluation [9, 10, 11, 12]. This makes deployment efficiency and reliability increasingly important alongside raw predictive performance.

The computational cost of contemporary pathology foundation models is particularly relevant. From a systems perspective, this mismatch is amplified at whole-slide scale. A tissue-rich WSI may generate thousands to tens of thousands of candidate tiles at diagnostic magnification, and the encoder is invoked repeatedly before slide-level aggregation or down stream decision logic. A reduction of one third in per-patch transformer compute therefore compounds across the entire WSI rather than saving only a single forward pass. This observation motivates task-specific restructuring of the patch encoder itself, rather than limiting efficiency analysis to the number of trainable downstream parameters. Virchow2 uses a ViT-H/14-scale encoder with approximately 632M parameters [3], while other leading models are similarly large. Wholeslide applications amplify this burden because a single slide may generate hundreds or thousands of patches. Even when a large encoder is frozen, its stored parameters and forward-pass operations remain present at inference. Parameter-efficient fine-tuning therefore reduces optimization cost but does not necessarily reduce the resident encoder or patch-level inference burden. Recent medical-image studies have increasingly recognized lightweight design as a distinct problem from parameterefficient adaptation [13, 14, 15]. In computational pathology, SUDA explicitly targets this problem by distilling heavy foundation models into lightweight students [16]. Distillation can be highly effective, but it changes the model family, requires a teacher–student transfer procedure, and may need to be repeated for new tasks or domains. This leaves an important complementary question: how much ofan already pretrained large pathology encoder is actually necessary for a specific downstream task?

Generic vision research suggests that substantial redundancy can exist both along transformer depth and within the token sequence. Structured sparsification and token-pruning approaches reduce computation by removing blocks, tokens, or interactions that contribute little to a target task [17, 18, 19, 20, 21]. Nevertheless, pathology foundation models are not ordinary ImageNet classifiers. Histopathological discriminative evidence may occupy small tissue regions, vary across magnification and morphology, and be especially sparse for low-prevalence classes. Aggressive pruning that looks attractive from a FLOP perspective can therefore erase diagnostically useful evidence. The desired objective is not maximum compression alone, but a validated Pareto operating point that preserves task utility while reducing genuine encoder cost. The design philosophy is similar to recent efficiency-oriented vision architectures that explicitly optimize accuracy against parameters and compute [22], but here the problem is addressed by restructuring a pretrained pathology foundation model rather than designing a new backbone from scratch.

A second limitation of efficiency-only evaluation is that a compressed medical model can retain accuracy while becoming less trustworthy in ways that aggregate accuracy does not reveal. Calibration error, negative log-likelihood, Brier score, confidence-based failure detection, and selective risk quantify different aspects of whether predicted probabilities are useful for decision support [23, 24, 25]. Recent pathology work has likewise emphasized domain robustness, fairness, and vulnerability to non-biological scanner or laboratory variation [26, 27, 28]. Thus, a credible compression study in computational pathology should establish not only accuracy and compute reduction but also whether uncertainty remains informative, minority classes are characterized explicitly, and the locked model generalizes to an independent cohort.

This study addresses these gaps with TAP-Path, short for TAP-Path, a task-adaptive structural and token pruningframeworkfor pathologyfoundation models. TAP-Path begins from the pretrained Virchow2 encoder and performs validation-only structural analysis to identify a task-relevant subset of transformer blocks. Unlike freezing or masking, selected blocks are physically retained in the deployed encoder while unselected blocks are removed. The structurally compressed model is then paired with input-adaptive token pruning that preserves the prefix token and retains the most task-relevant patch tokens using a combined class-token similarity and feature-magnitude score. To compensate for information loss from shortening both depth and token sequence, TAP-Path collects multi-depth features from four locations in the retained hierarchy and fuses them through a learned soft gate before classification. The model is selected entirely on training and validation data; the internal test set and CPTAC external cohort are evaluated only after configuration locking. The final single-head model is confirmed with three random seeds and evaluated with accuracy, balanced accuracy, macro-F1, rare-class balanced accuracy, calibration, probabilistic losses, failure-detection AUROC, runtime, parameters, analytical FLOPs, and frozen external performance.

In this paper, we propose TAP-Path, a task-adaptive compression framework that restructures a pretrained pathology foundation model for efficient and reliable downstream deployment.

The main contributions of this study are summarized as follows:

• We propose TAP-Path, a task-adaptive compression framework that identifies task-relevant transformer depth and physically reconstructs a compact, non-contiguous subnetwork from the pretrained Virchow2 hierarchy. This enables direct compression of the foundation model without training or distilling a separate student network.

• We develop an adaptive token-pruning and multi-depth recovery strategy that reduces redundant patch-token computation while preserving complementary representations from multiple stages of the compressed hierarchy. A learned gating mechanism integrates these representations for downstream prediction.

• We systematically investigate the computational efficiency of pathology foundation models through deployed parameter count and analytical FLOPs, together with predictive performance. The proposed design reduces Virchow2 encoder parameters by 24.96% and FLOPs by 35.20%, demonstrating that substantial computational redundancy can be removed while preserving competitive performance on the 32-class pathology task.

• We conduct a reliability-oriented evaluation relevant to trustworthy medical AI by examining calibration, failure awareness, selective prediction, rare-class behavior, and statistical uncertainty. We further evaluate the locked model on two independent CPTAC cohorts without external adaptation or threshold tuning to assess generalization beyond the development data.

Together, these results show that task-specific structural and token redundancy can be removed from a large pathology foundation model while retaining competitive discrimination and informative uncertainty.

## 2 Related Work

## 2.1 Pathology foundation models

The current generation of computational-pathology foundation models has been driven by scale, data diversity, and self-supervised learning. UNI established a general-purpose pathology representation benchmark using large-scale selfsupervision [1]; CONCH extended the paradigm to paired pathology image–text learning [2]; Virchow showed the value of million-slide-scale pretraining and a ViT-H backbone for common and rare cancers [3]; Prov-GigaPath introduced a whole-slide hierarchy trained on 1.3 billion tiles [4]; CHIEF combined tile-level and slide-level pretraining across multiple institutions [5]; and Hibou provided open DINOv2-based pathology encoders trained on more than one million WSIs [6]. Subsequent models have broadened clinical task coverage and pretraining strategies, including BEPH, GPFM, and PathOrchestra [29, 7, 8].

As the number of foundation models has grown, independent evaluation has become as important as model creation. Campanella et al. benchmarked public self-supervised pathology foundation models on clinical tasks [9]. Lee et al. systematically examined adaptation strategies and data-limited scenarios, finding that the preferred adaptation procedure depends on the downstream setting [10]. Neidlinger et al. evaluated 19 foundation models across external clinical cohorts and found that relative model ranking changes with the task and data regime [11]. A broader 2026 benchmark further compared general and pathology-specific vision and vision–language foundation models across multiple cohorts and task categories [12]. These studies motivate controlled same-task comparisons rather than relying on headline performance reported on unrelated benchmarks. Table 1 summarizes the most directly relevant foundation-model, adaptation, and pruning studies and highlights the methodological gap addressed by TAP-Path.

The literature also increasingly separates representation quality from domain robustness. Knowledge-guided adaptation has been proposed to improve cross-domain generalization and fairness [26], while recent work demonstrates that pathology foundation models may encode scanner and laboratory signatures that can compromise robustness [27]. Domain-generalization surveys similarly emphasize explicit external evaluation and transparent reporting of data sources [28]. Our work follows this direction by locking the compressed architecture before independent CPTAC evaluation and by reporting probabilityquality and failure-detection measures in addition to class prediction.

## 2.2 Efficient adaptation and compression

Several strategies can make foundation models easier to use downstream. Prompt tuning, adapters, and related parameterefficient methods update only a small portion of the model, reducing optimization memory and the number of trainable weights. For example, prompt-guided adaptive model transformation has been proposed for pathology classification using representative patch sampling, visual prompts, and adapters [30]; broader medical-image work has also demonstrated fewshot parameter-efficient tuning [14]. These approaches are useful when fine-tuning cost is the main bottleneck, but a frozen or adapter-tuned 600M-parameter encoder still executes most of the original forward pass.

Knowledge distillation instead transfers knowledge to a smaller student. GPFM uses a unified expert/self-distillation framework during pretraining [7], and SUDA directly targets efficient pathology analysis by combining unsupervised distillation and domain adaptation; its student can approach the teacher with a small fraction of the parameters [16]. Distillation is therefore a strong point of comparison, but it answers a different question from ours. TAP-Path asks whether a large pretrained encoder can be restructured in place for a target task without building a separate student, thereby preserving the original representation family while physically eliminating task-redundant computation.

This distinction matters because resource efficiency has become a first-class objective in medical imaging. Lightweight hybrid architectures have reduced model size in segmentation and histopathology-specific applications [13, 15]. One study by [15], explicitly evaluated model size, FLOPs, and MACs alongside predictive performance, illustrating the importance of quantitative efficiency assessment in histopathology models. Similarly, recent efficiency-oriented vision work such as AdaptViG evaluates a Pareto frontier between accuracy, parameters, and compute [22]. We adopt the same scientific principle—efficiency must be demonstrated quantitatively— but apply it to the task-specific compression of a pathology foundation model.

## 2.3 Transformer sparsification

The transformer architecture offers at least two complementary axes of redundancy: depth and tokens. Structured methods remove layers, heads, or channels, while token-pruning methods reduce the sequence length dynamically. Vision-specific token pruning has been revisited for dense prediction [17], graph-based propagation has been proposed to preserve token information under reduction [18], and Token Cropr demonstrated task-aware token removal across several vision problems [19]. Sparse structure exploration and learned tokenscoring networks further show that efficiency can be improved by searching for task-relevant computation rather than uniformly shrinking the network [20, 21].

Recent pathology-specific pruning evidence further supports the premise that transformer redundancy can be removed selectively. Boudissa et al. analyzed attention-head similarity and confidence in histopathology ViTs and showed that pruning redundant heads can preserve or even improve classification performance while reducing computational burden [31]. This work is complementary to TAP-Path: their unit of sparsification is the attention head in a distilled ViT, whereas our method physically selects non-contiguous transformer blocks and subsequently prunes patch tokens inside a large pretrained pathology foundation model. The distinction strengthens the motivation for testing structural redundancy directly in pathologyspecific transformers.

Pathology introduces a special constraint: the discriminative region may be spatially sparse. Token removal therefore has to be conditioned on feature content, and the pruning ratio should be chosen on a locked validation protocol rather than assumed a priori. TAP-Path combines a block-level task relevance signal with an input-adaptive token score, then uses multidepth feature recovery to preserve information from distinct stages of the compressed hierarchy.

## 2.4 Reliability and selective prediction

Modern neural networks can be accurate yet poorly calibrated, motivating temperature scaling and expected calibration error (ECE) [23]. Under dataset shift, uncertainty estimates can degrade even when in-distribution metrics appear strong [24]. Selective prediction addresses a complementary practical question: if the system abstains on its least-confident cases, how quickly does error risk decrease [25]? These concepts are particularly relevant to computational pathology, where externally acquired slides may differ in staining, preparation, scanner hardware, and population characteristics. Our operational trustworthiness analysis covers calibration (ECE), proper scoring rules (NLL and Brier), failure-detection AUROC, rareclass behavior, and risk–coverage. Together, these measures provide complementary evidence of probability quality, error awareness, selective-prediction behavior, and reliability under external evaluation. Their clinical interpretation requires prospective validation under the intended workflow.

## 3 Materials and Methods

## 3.1 Study design

The study followed a sequential model-selection and lockedevaluation protocol. Architecture screening and tokenretention selection were performed exclusively using the training and validation partitions. The internal test set was not used for transformer-block selection, token-retention selection, lossfunction selection, or calibration-hyperparameter optimization. After fixing the structural and token configuration, three independent task heads were trained using seeds 42, 123, and 2026. All external evaluations were subsequently performed using the locked configuration without external model adaptation or threshold optimization. This protocol maintains a strict separation between model development, internal testing, and external evaluation.

The internal benchmark comprised 25,495 image-level records representing 32 cancer classes derived from The Cancer Genome Atlas (TCGA) [32]. The dataset was partitioned into 17,769 training, 3,867 validation, and 3,859 test images, with each image represented using the final 12-patch protocol. Table 2 summarizes the resulting data partitions.

For independent external evaluation, we used 433 wholeslide images (WSIs) from the Clinical Proteomic Tumor Analysis Consortium (CPTAC), comprising 209 clear cell renal cell carcinoma (CCRCC) images [33] and 224 uterine corpus endometrial carcinoma (UCEC) images [34]. External evaluation was conducted at the image level using the locked model without external fitting or threshold optimization. Because the external cohorts represent only two of the 32 cancer classes included in the internal benchmark, external balanced accuracy was computed over the classes represented in CPTAC; metrics requiring averaging across all 32 internal classes were not used as primary measures of class-balanced external performance.

## 3.2 Preprocessing and patch selection

The internal representation is image-level but is deliberately constructed from multiple pathology regions. The final protocol uses a fixed 12-patch bag for every record. Patch selection was completed before TAP-Path architecture search and therefore does not use test-set model outcomes. Candidate regions were generated at multiple source crop sizes (224, 448, and 896 pixels) and subsequently resized to the encoder input resolution. To preserve scale coverage, the selector first reserves one high-confidence candidate from every available scale. Two additional high-entropy candidates are then forced into the bag so that difficult or ambiguous morphology is not systematically discarded. Remaining positions are filled by a combined utility criterion

$$
q _ { i } = 0 . 5 0 c _ { i } + 0 . 3 0 d _ { i } + 0 . 2 0 h _ { i } ,\tag{1}
$$

The probabilities used by this selector are generated by a reproducible patch probe, not by TAP-Path itself. Specifically, Hibou-B patch embeddings are L2-normalized and a multinomial logistic-regression classifier is fitted on training patches only using the LBFGS solver, L2 penalty, $C = 1 . 0$ , and the fixed screening seed. For candidate patch i, if $\mathbf { p } _ { i } ^ { ( q ) } \in [ 0 , 1 ] ^ { 3 2 }$ denotes the probe posterior, confidence and normalized predictive entropy are

$$
c _ { i } = \operatorname* { m a x } _ { k } p _ { i k } ^ { ( q ) } ,\tag{2}
$$

$$
h _ { i } = - \frac { 1 } { \log C } \sum _ { k = 1 } ^ { C } p _ { i k } ^ { ( q ) } \log \Bigl ( p _ { i k } ^ { ( q ) } + \epsilon \Bigr ) , \qquad C = 3 2 .\tag{3}
$$

Thus, the probe is a linear probabilistic readout of frozen Hibou-B features; it is neither a ResNet nor a zero-shot classifier. It is trained exclusively on the training split and is used only to construct the deterministic patch manifest.

In Eq. (1), $c _ { i }$ is the probe confidence from Eq. (2), d<sub>i</sub> is the minimum cosine distance between candidate i and the already selected Hibou-B patch representations, and $h _ { i }$ is the normalized entropy from Eq. (3). The selection procedure first reserves one highest-confidence patch from every available scale, then forces the two highest-entropy remaining patches, and finally fills the bag according to Eq. (1). The diversity term prevents collapse to near-duplicate tissue regions, whereas the entropy term deliberately preserves difficult regions.

For source image $I ,$ stored crop center $( x _ { i } , y _ { i } )$ , and source crop size $s _ { i } ,$ the selected region is

$$
P _ { i } = I \left[ y _ { i } - { \frac { s _ { i } } { 2 } } : y _ { i } + { \frac { s _ { i } } { 2 } } , x _ { i } - { \frac { s _ { i } } { 2 } } : x _ { i } + { \frac { s _ { i } } { 2 } } \right] .\tag{4}
$$

Regions extending beyond the source boundary are padded with white background and resized to 224 × 224 pixels using bicubic interpolation. The resulting 12 regions are deterministic for a given manifest and are reused across compared encoders, preventing model-specific patch selection from contaminating the benchmark. Representative selected regions are shown in Fig. 1, while Fig. 2 documents the long-tailed class distribution.

Table 1: Representative recent pathology foundation-model studies most relevant to TAP-Path.
<table><tr><td>Study</td><td>Year</td><td>Main strategy</td><td>Efficiency / adaptation</td><td>Gap addressed by TAP-Path</td></tr><tr><td>UNI [1]</td><td>2024</td><td>General pathology FM</td><td>Frozen transfer</td><td>No task-specific structural compression</td></tr><tr><td>CONCH [2]</td><td>2024</td><td>Vision-language FM</td><td>Compact transferable encoder</td><td>No in-place block/token pruning</td></tr><tr><td>Virchow [3]</td><td>2024</td><td>Large-scale pathology FM</td><td>Scale-driven pretraining</td><td>High inference footprint</td></tr><tr><td>Prov-GigaPath [4]</td><td>2024</td><td>WSI foundation modeling</td><td>Hierarchical slide modeling</td><td>Focuses context, not encoder compression</td></tr><tr><td>Lee et al. [10]</td><td>2025</td><td>PFM adaptation benchmark</td><td>PEFT / few-shot adaptation</td><td>Trainable-efficiency, not physical reduction</td></tr><tr><td>SUDA [16]</td><td>2026</td><td>Efficient pathology adaptation</td><td>Teacher-student distillation</td><td>Requires a separate student</td></tr><tr><td>Boudissa et al. [31]</td><td>2025</td><td>Histology ViT pruning</td><td>Attention-head pruning</td><td>Head-level, not PFM block+token pruning</td></tr><tr><td>PAMT [30]</td><td>2026</td><td>Prompt-guided adaptation</td><td>Prompts / adapters</td><td>Full backbone largely retained</td></tr><tr><td>TAP-Path</td><td></td><td>Task-adaptive PFM compression</td><td>Physical block + adaptive token pruning</td><td>Efficiency + reliability + frozen external validation</td></tr></table>

![](images/5ae722c980b0c93242e2a7bef959bc7174456249dd23f3356aeb85440f177cc3.jpg)  
(a) Representative internal H&E patches. TCGA abbreviations identify the sampled cancer classes.

![](images/109dd59f21d7dc3940dccda5f401fda21ef9c3e64347f1748ef931cd96368b7b.jpg)  
(b) Representative label-preserving transformations used during training.  
Figure 1: Representative data appearance and training-time augmentation. The pathology panel illustrates morphological heterogeneity across selected internal classes, while the augmentation panel shows the conservative geometric and photometric perturbations used to improve robustness. Validation, test, and external evaluations use deterministic unaugmented patches.

Table 2: Dataset partitions used in the locked evaluation protocol.
<table><tr><td>Partition</td><td>Images</td><td>Classes</td></tr><tr><td>Training</td><td>17,769</td><td>32</td></tr><tr><td>Validation</td><td>3,867</td><td>32</td></tr><tr><td>Internal test</td><td>3,859</td><td>32</td></tr><tr><td>Internal total</td><td>25,495</td><td>32</td></tr><tr><td>External CPTAC</td><td>433</td><td>2 present classes</td></tr></table>

Data-integrity checks verify the canonical class mapping, finite feature values, feature/index row agreement, deterministic split membership, and absence of zero-vector embeddings. As visualized in Fig. 2, class imbalance is substantial, motivating balanced accuracy, macro-F1, rare-class analysis, and class-wise reliability rather than relying exclusively on top-1 accuracy.

## 3.3 Data augmentation

Augmentation is restricted to transformations expected to preserve diagnostic class: horizontal/vertical reflection, 90<sup>◦</sup> rotation, mild contrast perturbation, and mild brightness perturbation. An augmented patch is

$$
\tilde { P } = \mathcal { T } _ { \boldsymbol { \theta } } ( P ) , \qquad \mathcal { T } _ { \boldsymbol { \theta } } \in \{ \mathcal { F } _ { \boldsymbol { h } } , \mathcal { F } _ { \boldsymbol { v } } , \mathcal { R } _ { 9 0 } , \mathcal { C } _ { \boldsymbol { \delta } _ { c } } , \mathcal { B } _ { \boldsymbol { \delta } _ { b } } \} .\tag{5}
$$

Aggressive color remapping is avoided because staindependent appearance can carry diagnostic information. Validation, internal test, and external CPTAC evaluation use deterministic, unaugmented patches.

## 3.4 Foundation-model baselines

Four pathology foundation encoders were retained as interpretable reference points: Hibou-B, CONCH, Virchow2, and UNI2-h. Hibou-B and CONCH represent smaller pathologyspecific encoders, while Virchow2 and UNI2-h represent the high-capacity regime. Comparisons use the same 32-class task and dataset partitions from the established benchmark.

![](images/0e38cdee1ab2adfa204637c051af8c89ac587857f256d2b5873a6224eb4928fd.jpg)  
Figure 2: Training-set distribution of the 32 internal cancer classes. The long-tailed class frequencies motivate balanced, macro-averaged, and rare-class evaluation.

We report each baseline’s measured downstream predictive metrics together with the parameter count and a consistently defined analytical encoder-FLOP estimate. The proposed method is initialized from Virchow2 because it provided a strong large-model baseline while exposing sufficient depth for structural analysis. For additional context, the study evaluated high-compute fusion systems. Virchow2+StaticTriFusion reached 87.43% accuracy with 81.78% macro-F1, and UNI2- h+DenseTriGate reached 87.91% accuracy with 82.68% macro-F1, but both execute three foundation representations simultaneously. Their approximate deployed backbone footprints are 808M/421.6G and 857M/532.4G (parameters/FLOPs), respectively, before small fusion-head overhead. These systems are therefore retained as high-compute context rather than direct deployment competitors.

## 3.5 TAP-Path overview

Let the pretrained encoder contain $L = 3 2$ transformer blocks,

$$
{ \mathcal { F } } ( X ) = B _ { L } \circ B _ { L - 1 } \circ \cdots \circ B _ { 1 } ( E ( X ) ) ,\tag{6}
$$

where $E ( \cdot )$ denotes patch embedding plus prefix-token construction and $B _ { \ell }$ is transformer block ℓ in Eq. (6). TAP-Path transforms $\mathcal { F }$ into a task-specific sparse encoder $\mathcal { F } _ { S , \rho }$ parameterized by a retained block set $\cal S \subset \{ 1 , \dots , L \}$ and tokenretention ratio $\rho .$ The final model uses $| S | = 2 4$ and $\rho = 0 . 7 0$ after the pruning point.

The design contains four coupled operations: (i) block novelty profiling and task-adaptive structural selection; (ii) physical removal of unselected transformer blocks; (iii) inputadaptive patch-token pruning; and (iv) gated recovery of features from four depths of the compressed hierarchy. The complete procedure is summarized in Algorithm 1 and illustrated in Fig. 3; Table 3 later contrasts the resulting system properties with the evaluated baselines.

## 3.6 Task-adaptive block selection

Uniform truncation assumes that later blocks are always more expendable or that useful information is distributed monotonically with depth. We instead estimate the normalized residual change induced by each block. For a profiling sample x and block ℓ, let $H _ { \ell - 1 }$ and $H _ { \ell }$ denote the token tensors immediately before and after the block. We define block novelty as

$$
\nu _ { \ell } ( x ) = \frac { \sqrt { \frac { 1 } { N D } \left\| H _ { \ell } - H _ { \ell - 1 } \right\| _ { F } ^ { 2 } } } { \sqrt { \frac { 1 } { N D } \left\| H _ { \ell - 1 } \right\| _ { F } ^ { 2 } } + \epsilon } ,\tag{7}
$$

where N is the token count, D is the embedding dimension, and $\epsilon = 1 0 ^ { - 8 }$ prevents numerical instability. Profiling is restricted to the development subset and excludes internal test and CPTAC records. The dataset-level score is

$$
\bar { \nu } _ { \ell } = \frac 1 M \sum _ { m = 1 } ^ { M } \nu _ { \ell } ( x _ { m } ) .\tag{8}
$$

To preserve low-level token formation and high-level semantic consolidation, the first four and final four blocks are anchored. For a target retained depth K, the remaining $K - 8$ blocks are selected from the middle hierarchy according to $\bar { \nu } _ { \ell }$ and then restored to their original order. Because transformer blocks preserve a constant hidden width, block excision does not create a dimensional mismatch. If $H _ { \ell } \in \mathbb { R } ^ { N \times D }$ is the residual stream after a retained block and blocks $\ell + 1 , \dots , j - 1$ are removed, the next retained block receives

$$
H _ { j } = B _ { j } ( H _ { \ell } ) .\tag{9}
$$

The pretrained weights inside retained modules are unchanged; only the computational graph is shortened. The selected 24- block configuration in the locked experiment retained original blocks

$$
S ^ { \star } = \{ 1 , \dots , 5 \} \cup \{ 1 4 , \dots , 3 2 \} .\tag{10}
$$

using one-based indexing for readability. The practical implementation physically replaces the original block list with the retained modules; consequently, omitted blocks no longer contribute stored encoder parameters or forward-pass computation.

Structural candidates were screened against contiguousdepth controls and task-sparse variants. Candidate selection was validation-only and explicitly constrained by compression. The objective can be written as

(11)

$$
\mathrm { s . t . } \quad R _ { P } ( \mathcal { S } ) \geq r _ { P } , \qquad A _ { \mathrm { v a l } } ( \mathcal { S } ) \geq A _ { \mathrm { v a l } } ^ { \star } - \delta ,\tag{12}
$$

where $R _ { P } = 1 - P s / P _ { \mathrm { f u l l } }$ is parameter reduction, $r _ { P }$ is the required reduction, and δ is the allowed validation tolerance. This formulation avoids choosing the smallest network when its accuracy has already collapsed.

## 3.7 Adaptive token pruning

After structural selection, token reduction is applied within the compressed encoder. Let $c \in \mathbb { R } ^ { D }$ denote the prefix/class token and $p _ { i } \in \mathbb { R } ^ { D }$ a patch token. We first normalize both and compute a class-token similarity,

$$
s _ { i } ^ { \mathrm { s i m } } = \frac { p _ { i } ^ { \top } c } { \| p _ { i } \| _ { 2 } \| c \| _ { 2 } } .\tag{13}
$$

To avoid retaining only tokens aligned with the current class token, we add an activation-magnitude term. With per-image normalized magnitude

$$
s _ { i } ^ { \mathrm { m a g } } = \frac { \| p _ { i } \| _ { 2 } - \mu _ { \| p \| } } { \sigma _ { \| p \| } + \epsilon } ,\tag{14}
$$

the fixed development weighting prioritizes semantic alignment while retaining a secondary activation-magnitude cue; it is frozen before test evaluation. The combined token importance is

$$
s _ { i } = 0 . 7 5 s _ { i } ^ { \mathrm { s i m } } + 0 . 2 5 s _ { i } ^ { \mathrm { m a g } } .\tag{15}
$$

The prefix token is always retained. Among $N _ { p }$ patch tokens, the model keeps

$$
K _ { p } = \left\lceil \rho N _ { p } \right\rceil .\tag{16}
$$

patches with the largest $s _ { i }$ according to Eqs. (15)–(16), where $\rho \in \{ 1 . 0 0 , 0 . 8 5 , 0 . 7 0 \}$ was evaluated after structural locking. Token pruning begins at retained execution position max(1, round $( 2 4 \times 0 . 5 5 ) ) \ : = \ : 1 3$ , i.e., after the 13th block in the retained execution sequence. Ranking is recomputed for every patch and introduces no trainable token-router subnetwork.

The analytical compute reduction arises from both fewer blocks and a smaller token sequence. If $\Phi _ { \ell } ( N )$ denotes the FLOPs of transformer block ℓ at token count N, the compressed encoder cost is approximated by

$$
\Phi _ { \mathrm { T A P } } = \sum _ { \ell \in { \cal S } _ { \mathrm { p r e } } } \Phi _ { \ell } ( N ) + \sum _ { \ell \in { \cal S } _ { \mathrm { p o s t } } } \Phi _ { \ell } ( K _ { p } + N _ { \mathrm { p r e f i x } } ) ,\tag{17}
$$

where $ { \mathcal { S } } _ { \mathrm { p r e } }$ and $ { S _ { \mathrm { p o s t } } }$ denote retained blocks before and after the pruning point.

## 3.8 Multi-depth feature recovery

Compression can remove intermediate transformations that a downstream classifier would otherwise exploit. We therefore do not classify from only the final compressed token state. Four approximately evenly spaced taps are collected from the retained hierarchy. Four taps balance hierarchical coverage against projection/gating overhead and cached feature dimensionality. The effective token count $K _ { t }$ depends on tap location: taps before the pruning point observe the dense sequence, whereas later taps observe the retained 70% patch sequence. At tap t, the token sequence is normalized and summarized using the prefix token and mean patch token,

$$
u _ { t } = \left[ \mathrm { L N } ( c _ { t } ) ; \frac { 1 } { K _ { t } } \sum _ { i = 1 } ^ { K _ { t } } \mathrm { L N } ( p _ { t , i } ) \right] .\tag{18}
$$

Each $u _ { t }$ is projected to a 256-dimensional task space:

$$
z _ { t } = \mathrm { G E L U } ( W _ { t } \operatorname { L N } ( u _ { t } ) + b _ { t } ) .\tag{19}
$$

Instead of concatenating the four taps and treating them equally, an adaptive gate estimates their sample-specific contri bution. Let

$$
z _ { \mathrm { c a t } } = [ z _ { 1 } ; z _ { 2 } ; z _ { 3 } ; z _ { 4 } ] .\tag{20}
$$

The gate is

$$
\alpha = \mathrm { s o f t m a x } \left( W _ { g 2 } \mathrm { \ G E L U } \left( W _ { g 1 } \mathrm { L N } ( z _ { \mathrm { c a t } } ) \right) \right) , \quad \sum _ { t = 1 } ^ { 4 } \alpha _ { t } = 1 ,\tag{21}
$$

and the fused feature becomes

$$
z = \sum _ { t = 1 } ^ { 4 } \alpha _ { t } z _ { t } .\tag{22}
$$

The final classifier is a two-layer MLP with LayerNorm, a 512-unit hidden layer, GELU, dropout 0.18, and a 32-class output layer. This single task head contains 5.70M parameters; the deployed single-head model therefore contains 479.40M parameters in total.

## 3.9 Optimization objectives

For the primary single-head model, class weights are mild rather than fully inverse-frequency weighted:

$$
w _ { c } = \frac { \left( \frac { N } { C n _ { c } } \right) ^ { \gamma } } { \frac { 1 } { C } \sum _ { j = 1 } ^ { C } { \left( \frac { N } { C n _ { j } } \right) ^ { \gamma } } } , \qquad \gamma = 0 . 2 0 ,\tag{23}
$$

where $n _ { c }$ is the number of training examples in class c. Weighted cross-entropy uses label smoothing $\epsilon _ { \mathrm { l s } } = 0 . 0 1$ . To discourage degenerate single-tap gating, we define gate entropy

$$
H ( \pmb { \alpha } ) = - \sum _ { t = 1 } ^ { 4 } \alpha _ { t } \log ( \alpha _ { t } + \epsilon )\tag{24}
$$

and optimize

$$
{ \mathcal { L } } _ { \mathrm { p r i m a r y } } = { \mathcal { L } } _ { \mathrm { C E } } - \lambda _ { H } H ( \alpha ) , \qquad \lambda _ { H } = 0 . 0 0 2 .\tag{25}
$$

Rare-class sensitivity was investigated after the main architecture had been locked. Five validation-only objectives were compared: mild weighted cross-entropy, Balanced Softmax, logit adjustment with $\tau = 0 . 5$ , logit adjustment with $\tau = 1 . 0$ and class-balanced focal loss. For logit adjustment,

$$
\tilde { o } _ { c } = o _ { c } + \tau \log ( \pi _ { c } + \epsilon ) ,\tag{26}
$$

where $o _ { c }$ is class logit c and $\pi _ { c }$ is the empirical training prior. This experiment is reported as an ablation of the same compressed architecture; it is not merged with the primary head to construct an artificial “best of all metrics” model.

![](images/5087faab5b7f20e7177d6a0c113efb21dd855924d96f4b4e96120b7a365a2f84.jpg)  
Figure 3: Architecture of the proposed TAP-Path framework. Multi-scale H&E regions are ranked to construct a deterministic 12-patch bag, followed by task-adaptive structural compression of the Virchow2 encoder to the locked one-based block set $\{ 1 , \dotsc , 5 \} \cup \{ 1 4 , \dotsc , 3 2 \}$ (24 of 32 blocks). Adaptive late-hierarchy token pruning retains 70% of informative tokens, while four intermediate feature representations are projected to 256 dimensions and adaptively combined through a learned soft gate before 32-class prediction. The resulting predictions are evaluated for classification performance, calibration, failure awareness, and frozen external generalization on CPTAC-CCRCC and CPTAC-UCEC. TAP-Path reduces the encoder from 631.24M to 473.70M parameters and the analytical encoder computation from 340.13 to 220.40 GFLOPs while achieving 87.98% internal test accuracy.

## 3.10 Training and model selection

The optimization pipeline is separated into architecture selection and task-head learning so that the expensive foundation backbone is not repeatedly fine-tuned for every candidate.

Stage 1: reduced class-aware screening. Structural search uses 1,400 training and 700 validation records selected deterministically with class awareness. Only the first two ranked patches per selected image are processed during screening. This subset exists solely to rank compression candidates efficiently.

Stage 2: structural search. Contiguous-depth controls and non-contiguous task-sparse candidates are constructed from the frozen Virchow2 parent. Each candidate is evaluated using screening features and a lightweight validation probe. The selection criterion jointly considers validation utility and compression; internal test data are not used to choose retained blocks.

Stage 3: token-retention search. After the strongest compressed structures are identified, $\rho \in \{ 1 . 0 0 , 0 . 8 5 , 0 . 7 0 \}$ is evaluated on validation data. TaskSparse24 with $\rho = 0 . 7 0$ is locked before final representation extraction.

Stage 4: full 12-patch feature extraction. The physically reconstructed encoder is run over the complete deterministic 12-patch bags. CUDA automatic mixed precision is enabled where supported. Screening encoder batch size is 4, final feature-extraction batch size is 6, and the development GPU is an NVIDIA GeForce RTX 5060 Ti with approximately 16 GB VRAM.

Stage 5: task-head optimization. The single gated multitap head is trained on cached image-level compressed features for at most 130 epochs using AdamW, learning rate $7 \times 1 0 ^ { - 4 }$ weight decay $2 \times 1 0 ^ { - 4 }$ , batch size 384, gradient clipping at 5, dropout 0.18, and early-stopping patience 18. Three seeds (42, 123, 2026) quantify optimization variability.

Stage 6: calibration and locked evaluation. Temperature scaling is fitted only to validation logits using LBFGS. The locked internal test set is then evaluated. CPTAC-CCRCC and CPTAC-UCEC are processed only after internal config uration locking; no external label, threshold, temperature, or architecture decision modifies TAP-Path.

Structural-screening performance and final test performance are not directly comparable because they correspond to distinct experimental protocols. Screening uses a reduced class-aware subset and two patches per image for efficient candidate ranking, whereas final evaluation uses the complete deterministic

Algorithm 1 TAP-Path: task-adaptive structural and token pruning with multi-depth recovery   
Require: Pretrained Virchow2 encoder $\{ B _ { \ell } \} _ { \ell = 1 } ^ { 3 2 } ;$ training set $\mathcal { D } _ { t r } \colon$ validation set $\mathcal { D } _ { v a } ;$ deterministic candidate-patch manifest   
Ensure: Compressed encoder $\mathcal { F } _ { S ^ { \star } , \rho ^ { \star } }$ and calibrated 32-class task head   
1: Extract frozen Hibou-B candidate-patch embeddings and L2-normalize them   
2: Fit the multinomial logistic patch probe on $\mathcal { D } _ { t r }$ only   
3: for each candidate patch i do   
Compute probe confidence $c _ { i }$ and entropy $h _ { i }$ using Eqs. (2)–(3)   
5: end for   
6: for each image do   
7: Reserve the highest-confidence patch at every available scale   
8: Add the two highest-entropy remaining patches   
9: while fewer than 12 patches are selected do   
10: Compute confidence–diversity–entropy utility using Eq. (1)   
11: Add the highest-utility remaining patch   
12: end while   
13: end for   
14: Profile Virchow2 block novelty $\bar { \nu } _ { \ell }$ on development data using Eq. (7)   
15: Construct contiguous-depth controls and non-contiguous task-sparse block candidates   
16: for each candidate block set $s$ do   
17: Physically instantiate only blocks in $s$   
18: Extract reduced-protocol screening features and fit the fixed lightweight validation probe   
19: Record validation accuracy, BA, macro-F1, encoder parameters, and analytical FLOPs   
20: end for   
21: Lock the validation-Pareto structural candidate $S ^ { \star }$ (24 of 32 blocks)   
22: for $\rho \in \{ 1 . 0 0 , 0 . 8 5 , 0 . 7 0 \}$ do   
23: Score tokens with Eq. (15); retain $K _ { p }$ tokens using Eq. (16)   
24: Evaluate validation performance and analytical compute   
25: end for   
26: Lock $\rho ^ { \star } = 0 . 7 0$ and rebuild the executable compressed encoder   
27: for seed ∈ {42, 123, 2026} do   
28: Extract the full deterministic 12-patch representation   
29: Recover four hierarchy taps and compute the adaptive gate using Eq. (21)   
30: Fuse taps using Eq. (22) and optimize the single 32-class head   
31: Fit temperature scaling on validation logits only   
32: Evaluate the locked internal test set and frozen CPTAC cohorts   
33: end for   
34: Report predictive, efficiency, calibration, failure-detection, selective-risk, rare-class, and bootstrap metrics

12-patch representation and the trained multi-depth recovery head.

## 3.11 Common-probe diagnostic

To assess representation quality independently of modelspecific task heads, we additionally trained an identical lightweight probe on each frozen representation. Under this standardized readout, TAP-Path achieved approximately 85% accuracy, compared with approximately 88% using its proposed multi-depth gated head. The difference reflects the contribution of the task-specific multi-depth recovery mechanism to downstream prediction. Because the common-probe experiment evaluates representation quality under a standardized classifier rather than the complete TAP-Path inference architecture, it is reported as a complementary diagnostic analysis.

## 3.12 Reliability metrics

For each trained seed, post-hoc temperature scaling is fitted on validation logits only. Given logits o and temperature $T > 0$

$$
p _ { c } = \frac { \exp ( o _ { c } / T ) } { \sum _ { j = 1 } ^ { C } \exp ( o _ { j } / T ) } .\tag{27}
$$

T is optimized by minimizing validation NLL. ECE is computed over 15 confidence bins,

$$
\mathrm { E C E } = \sum _ { b = 1 } ^ { B } \frac { | I _ { b } | } { n } \left| \operatorname { a c c } ( I _ { b } ) - \operatorname { c o n f } ( I _ { b } ) \right| .\tag{28}
$$

We additionally report multiclass Brier score

$$
{ \mathrm { B r i e r } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \sum _ { c = 1 } ^ { C } ( p _ { i c } - y _ { i c } ) ^ { 2 }\tag{29}
$$

and NLL. These measures are complementary: ECE summarizes bin-level calibration, while NLL and Brier are proper scoring rules sensitive to the full predictive distribution.

For failure detection, each example receives an uncertainty score

$$
u _ { i } = 1 - \operatorname* { m a x } _ { c } p _ { i c } .\tag{30}
$$

Ground-truth failures are $e _ { i } = \mathcal { Y } [ \arg \operatorname* { m a x } _ { c } p _ { i c } \neq y _ { i } ]$ . The AUROC between $u _ { i }$ and $e _ { i }$ measures whether errors tend to receive higher uncertainty. Risk–coverage analysis sorts examples by confidence and evaluates error rate among the retained fraction. This supports a selective-prediction interpretation but is not presented as proof of clinical safety.

Rare classes are defined from the training distribution using the lower quartile of positive class counts. Rare-class balanced accuracy is the unweighted mean recall over those classes:

$$
\mathrm { R a r e B A } = \frac { 1 } { \left| \mathcal { R } \right| } \sum _ { c \in \mathcal { R } } \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } .\tag{31}
$$

## 3.13 Statistical analysis

Three independent seeds are reported as mean±SD. Stratified bootstrap confidence intervals are estimated from frozen test predictions using B = 2000 resamples:

$$
\mathrm { C I } _ { 9 5 } ( m ) = \left[ Q _ { 0 . 0 2 5 } \{ m _ { b } \} _ { b = 1 } ^ { B } , Q _ { 0 . 9 7 5 } \{ m _ { b } \} _ { b = 1 } ^ { B } \right] .\tag{32}
$$

Because aligned prediction vectors are not available for every baseline, marginal confidence intervals are not used as a substitute for a paired superiority test. Small accuracy differences are therefore described as comparable/slightly higher performance, whereas parameter and FLOP reductions are reported as deterministic architectural differences.

## 3.14 Implementation details

Experiments were implemented in Python 3.11 with PyTorch and the timm model library. Foundation models were used as pretrained encoders; mixed-precision feature extraction was enabled on CUDA where supported. The reported development system used an NVIDIA GeForce RTX 5060 Ti GPU. Structural screening used a reduced validation protocol to avoid repeatedly executing all 12 patches for every candidate, whereas the locked winner was re-evaluated under the full 12-patch protocol.

The final single-head task model used four taps, 256- dimensional tap projections, dropout 0.18, up to 130 training epochs, early stopping with patience 18, AdamW with learning rate $7 \times 1 0 ^ { - 4 }$ and weight decay $2 \times 1 0 ^ { - 4 }$ , gradient clipping at $5 ,$ and three seeds (42, 123, 2026). The task-head batch size was 384 on cached image-level representations. Temperature scaling used LBFGS on the validation logits. The final runtime profile of the compressed model was 31.85±1.21 ms per image on the development GPU, corresponding to 31.40 images/s under the measured profiling configuration.

![](images/b3531175b30f36c315f9f98b855c1b3adf5bc8065fcd2fa5e7c3f19d2a07f7c0.jpg)  
Figure 4: Validation-only structural screening of contiguous depth and task-adaptive sparse candidates.

## 4 Results

## 4.1 Structural compression

The first question was whether selecting blocks by taskdependent novelty provides a better compressed representation than simply truncating the model. Table 4 summarizes the validation-only screening. Full Depth32 achieved the highest absolute screening accuracy (72.96%) but retained the entire encoder. A contiguous Depth24 model reduced encoder parameters by 24.96% but reached 68.28% screening accuracy. At the same parameter count, TaskSparse24 improved screening accuracy to 69.79%, supporting the use of non-contiguous taskadaptive block retention. More aggressive TaskSparse22 and TaskSparse20 configurations reduced parameters by 31.20% and 37.43%, respectively, but incurred additional validation loss.

Figure 4 visualizes the screening frontier. Importantly, the screening accuracies are not directly compared with the final test accuracies because they were obtained with a reduced screening protocol. Their purpose was model selection under equal candidate conditions.

## 4.2 Token-pruning ablation

Token-retention results are shown in Table 5, and the corresponding validation trend is visualized in Fig. 5. For TaskSparse24, retaining 85% of patch tokens increased validation accuracy from 69.79% to 70.39% while reducing analytical FLOPs from 255.19G to 237.71G. Retaining 70% produced the strongest validation score (71.60%) and further reduced compute to 220.40G, a 35.20% reduction relative to full Virchow2. The same 70% ratio was less effective for TaskSparse22, indicating an interaction between structural depth and token sparsity rather than a universal benefit from dropping tokens.

The validation results indicate that moderate adaptive token pruning can improve the accuracy–compute operating point after stable early representations have formed. This behavior is consistent with removal of redundant or weakly aligned token content and motivates joint selection of structural depth and token retention under the validation protocol.

Table 3: Method-level comparison of the evaluated foundation-model strategies. A checkmark indicates that the property is explicitly present in the evaluated system.
<table><tr><td>Method</td><td>Physical compression</td><td>Adaptive token pruning</td><td>Multi-depth recovery</td><td>Reliability analysis</td><td>Frozen external test</td></tr><tr><td>Hibou-B</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>CONCH</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Virchow2</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>UNI2-h</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>StaticTriFusion / DenseTriGate</td><td></td><td></td><td>√</td><td>partial</td><td>√</td></tr><tr><td>TAP-Path</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr></table>

Table 4: Structural validation screen. Values are used only for architecture selection and are not the final 12-patch test performance.
<table><tr><td>Candidate</td><td>Acc. (%)</td><td>BA (%)</td><td>Params (M)</td><td>FLOPs (G)</td></tr><tr><td>Depth32</td><td>72.96</td><td>73.08</td><td>631.24</td><td>340.13</td></tr><tr><td>Depth28</td><td>70.39</td><td>70.31</td><td>552.47</td><td>297.66</td></tr><tr><td>Depth24</td><td>68.28</td><td>68.38</td><td>473.70</td><td>255.19</td></tr><tr><td>TaskSparse24</td><td>69.79</td><td>69.84</td><td>473.70</td><td>255.19</td></tr><tr><td>TaskSparse22</td><td>69.18</td><td>69.30</td><td>434.32</td><td>233.96</td></tr><tr><td>TaskSparse20</td><td>66.47</td><td>66.50</td><td>394.94</td><td>212.73</td></tr></table>

Table 5: Token-retention ablation for the locked structural candidates.
<table><tr><td>Structure</td><td>Tokens (%)</td><td>Acc. (%)</td><td>BA (%)</td><td>FLOPs (G)</td></tr><tr><td>TaskSparse24</td><td>100</td><td>69.79</td><td>69.84</td><td>255.19</td></tr><tr><td>TaskSparse24</td><td>85</td><td>70.39</td><td>70.42</td><td>237.71</td></tr><tr><td>TaskSparse24</td><td>70</td><td>71.60</td><td>71.64</td><td>220.40</td></tr><tr><td>TaskSparse22</td><td>100</td><td>69.18</td><td>69.30</td><td>233.96</td></tr><tr><td>TaskSparse22</td><td>85</td><td>68.73</td><td>68.56</td><td>218.07</td></tr><tr><td>TaskSparse22</td><td>70</td><td>67.98</td><td>67.97</td><td>202.33</td></tr></table>

## 4.3 Foundation-model comparison

Table 6 presents the central same-task comparison. The proposed single-head TAP-Path configuration achieved 87.98 ± 0.067% test accuracy across three seeds, compared with 86.89% for full Virchow2 and 87.67% for UNI2-h. The corresponding macro-F1 was 82.38±0.48%, compared with 80.94% for Virchow2 and 81.75% for UNI2-h, while balanced accuracy reached 81.26 ± 0.49%. These results place TAP-Path at the highest observed accuracy and macro-F1 among the evaluated single-backbone systems while substantially reducing the computational footprint of the large-model parent. Because paired baseline prediction files were unavailable, the comparison is reported as an observed performance difference without a paired significance test.

The efficiency gains are substantial. Full Virchow2 contains approximately 631–632M encoder parameters, whereas the physically compressed encoder contains 473.70M, corresponding to a 24.96% reduction. Including the task head, the deployed TAP-Path configuration contains 479.40M parameters, and analytical encoder FLOPs decrease from 340.13G to 220.40G (35.20%). Relative to UNI2-h, TAP-Path also operates with substantially fewer parameters and analytical FLOPs while achieving comparable or higher predictive performance. Hibou-B and CONCH occupy a lower-compute regime, whereas TAP-Path defines a favorable operating point within the high-performing large-foundation-model regime.

![](images/e6f5abca0851071aec5e6916fb8912855f1df8bbc2068262f7f14fdf6db21e45.jpg)  
Figure 5: Token-retention ablation after structural selection. The 70% TaskSparse24 configuration provides the best validation accuracy with the lowest compute among the displayed TaskSparse24 settings.

Table 7 places the high-compute fusion experiments beside the single-backbone systems. These experiments show that comparable discrimination can also be obtained through multi-foundation-model fusion, but at substantially greater deployment-scale parameter and compute requirements. In contrast, TAP-Path achieves its operating point with a single physically compressed backbone.

The Pareto view is clearer in Figs. 6 and 7. The corrected high-compute fusion references are included in both plots: Virchow2+StaticTriFusion reaches 87.43% accuracy and UNI2- h+DenseTriGate reaches 87.91%, whereas TAP-Path reaches 87.98%. Including the high-compute fusion references further illustrates the resulting Pareto trade-off. TAP-Path achieves

Table 6: Direct same-task comparison on the 32-class internal test set. TAP-Path values are mean±SD over three independently trained single heads. Arrows indicate preferred direction. Analytical FLOPs refer to the encoder and are used for relative efficiency comparison.
<table><tr><td>Model</td><td>Params (M) ↓</td><td>FLOPs (G) ↓</td><td>Accuracy (%) ↑</td><td>BA (%) ↑</td><td>Macro-F1 (%) ↑</td><td>External Acc. (%) ↑</td></tr><tr><td>Hibou-B</td><td>85.7</td><td>46.32</td><td>82.67</td><td>75.17</td><td>76.33</td><td>91.53</td></tr><tr><td>CONCH</td><td>90.0</td><td>35.13</td><td>81.43</td><td>76.39</td><td>76.11</td><td>85.76</td></tr><tr><td>Virchow2</td><td>632.0</td><td>340.13</td><td>86.89</td><td>80.52</td><td>80.94</td><td>90.76</td></tr><tr><td>UNI2-h</td><td>681.0</td><td>450.97</td><td>87.67</td><td>81.12</td><td>81.75</td><td>90.07</td></tr><tr><td>TAP-Path (ours)</td><td>479.40</td><td>220.40</td><td>87.98±0.067</td><td>81.26±0.49</td><td>82.38±0.48</td><td>91.22±0.83</td></tr></table>

External accuracy is reported alongside the internal metrics to characterize transfer behavior across model scales; Hibou-B attains the highest raw accuracy on the two-class CPTAC cohort, while TAP-Path provides the strongest joint internal accuracy–efficiency operating point among the high-capacity systems.

Table 7: Broader experimental context including high-compute fusion references. Fusion parameter/FLOP values approximate the simultaneously deployed foundation encoders and therefore indicate deployment scale rather than trainable task-head size.
<table><tr><td>System</td><td>Params (M)</td><td>FLOPs (G)</td><td>Accuracy (%)</td><td>Macro-F1 (%)</td><td>Deployment regime</td></tr><tr><td>Virchow2</td><td>632.0</td><td>340.1</td><td>86.89</td><td>80.94</td><td>Full single backbone</td></tr><tr><td>UNI2-h</td><td>681.0</td><td>451.0</td><td>87.67</td><td>81.75</td><td>Full single backbone</td></tr><tr><td>Virchow2+StaticTriFusion</td><td>~808</td><td>~421.6</td><td>87.43</td><td>81.78</td><td>Three-FM fusion</td></tr><tr><td>UNI2-h+DenseTriGate</td><td>~857</td><td>~532.4</td><td>87.91</td><td>82.68</td><td>Three-FM fusion</td></tr><tr><td>TAP-Path</td><td>479.40</td><td>220.4</td><td>87.98</td><td>82.38</td><td>Compressed single backbone</td></tr></table>

![](images/75912a356dca358c7553c82e28e17e6f7483e6b4dcba79ecfb6f5431c824db1a.jpg)  
Figure 6: Accuracy–parameter trade-off. Each operating point is labeled directly with model name and test accuracy; light guide lines preserve point–annotation correspondence.

87.98% accuracy using 479.40M deployed parameters and 220.40G analytical encoder FLOPs, while the fusion systems require substantially greater deployment-scale parameters and compute for similar predictive performance.

## 4.4 Reliability analysis

Reliability is organized into three complementary questions. First, are predicted probabilities well behaved? This is assessed with ECE, NLL, and Brier score. Second, does confidence identify likely mistakes? This is assessed with failuredetection AUROC and selective risk–coverage. Third, does performance remain acceptable for low-frequency classes? This is assessed with rare-class balanced accuracy and classwise F1. Table 8 and Figs. 8–9 report these three views jointly. For the primary three-seed TAP-Path task head, ECE is $0 . 0 3 0 1 \pm 0 . 0 0 2 2$ , NLL is $0 . 4 6 9 4 \pm 0 . 0 0 1 5 .$ , Brier score is $0 . 1 8 0 0 \pm 0 . 0 0 0 5$ , and failure-detection AUROC is $0 . 9 0 4 7 \pm 0 . 0 0 6 0$ . These measurements indicate that TAP-Path maintains informative confidence estimates despite structural and token pruning. In particular, its Brier score and failuredetection AUROC improve over the full Virchow2 baseline, whereas UNI2-h remains marginally better in ECE and NLL. The reliability claim is therefore based on the joint behavior of several metrics rather than on a single calibration statistic.

Rare-class sensitivity is evaluated on the same compressed encoder using the validation-selected rare-aware objective. This operating point achieves rare-class balanced accuracy $0 . 6 8 6 4 \pm 0 . 0 2 1 6$ and balanced accuracy $0 . 8 2 2 9 \pm 0 . 0 0 3 8$ . We report that value in the reliability comparison because it is the dedicated minority-sensitive operating point of TAP-Path; the remaining TAP-Path entries in Table 8 are the primary threeseed reliability results. This distinction avoids understating the model’s validated rare-class capability while retaining the provenance of each metric.

Figure 8 provides a direct visual comparison of the reliability profile, while Fig. 9 evaluates whether confidence can support selective prediction. At 60% internal coverage, TAP-Path retains approximately 99.2% selective accuracy, and risk rises gradually as progressively less-confident cases are accepted. This monotonic behavior is consistent with a useful abstention signal: the model’s low-confidence subset is enriched for errors.

![](images/d24b316f345ca4b08161ad2a401fd4dac8207f51ef93c17a1880cd13c477b05f.jpg)  
Figure 7: Accuracy–compute trade-off. Direct annotations connect each model and test accuracy to its analytical encoder FLOP requirement.

Table 8: Reliability comparison for the large-model baselines and TAP-Path. TAP-Path ECE, Brier, and failure AUROC and Rare BA are the primary three-seed values at the operating point on the same compressed encoder.
<table><tr><td>Model</td><td>ECE↓</td><td>Brier ↓</td><td>Fail-AUROC ↑</td><td>Rare BA ↑</td></tr><tr><td>Virchow2</td><td>0.0368</td><td>0.1882</td><td>0.8920</td><td>0.6996</td></tr><tr><td>UNI2-h</td><td>0.0297</td><td>0.1825</td><td>0.8920</td><td>0.6931</td></tr><tr><td>TAP-Path</td><td> $0 . 0 3 0 1 { \scriptstyle \pm . 0 0 2 2 }$ </td><td> $\mathbf { 0 . 1 8 0 0 { \overset { \cdot } { \bot } } . 0 0 0 5 }$ </td><td> $\mathbf { 0 . 9 0 4 7 { \scriptstyle \pm . 0 0 6 0 } }$ </td><td> $0 . 6 8 6 4 { \scriptstyle \pm . 0 2 1 6 }$ </td></tr></table>

## 4.5 Rare-class analysis

Following the reliability comparison in Table 8. Rare-class behavior was further examined under alternative optimization objectives while keeping the compressed TAP-Path encoder fixed. The primary calibrated configuration is denoted by P, whereas the validation-selected rare-aware configuration is denoted by R. This separation allows the effect of the taskhead objective on minority-class sensitivity to be evaluated independently of structural compression.

The rare-aware objectives are therefore reported within the unified ablation analysis in Table 9, preserving the distinction between the primary operating point and alternative minoritysensitive objectives. Instead, Table 9 integrates the rare-aware objectives into the full ablation narrative alongside structural and token choices. On validation data, Balanced Softmax produced the highest rare-class balanced accuracy (70.55%) but reduced overall accuracy. Logit adjustment with $\tau = 1 . 0$ achieved the highest validation accuracy (90.25%) among the tested objectives while raising rare-class BA from 67.39% for the mild-CE baseline to 69.68%.

![](images/85b86de867423541a790f81e03634b9846ffa1ba685950b3ec61288220118c8c.jpg)  
Figure 8: Reliability and rare-class profile of TAP-Path relative to full Virchow2 and UNI2-h. Lower ECE/Brier and higher failure AUROC/Rare BA are preferred.

![](images/c75c3ba9f1db15029db7df04fc5d4a6fb4532e78201641968083e6c6958fca8e.jpg)  
Figure 9: Selective risk–coverage behavior for TAP-Path on the locked internal and external evaluations.

The validation trade-off is shown in Fig. 10, and classlevel behavior is summarized in Fig. 11. When the selected logit-adjusted head was repeated across three seeds on the locked compressed encoder, test rare-class BA increased to $6 8 . 6 4 \pm 2 . 1 6 \%$ , balanced accuracy to $8 2 . 2 9 \pm 0 . 3 8 \%$ , while accuracy decreased to $8 7 . 1 3 \pm 0 . 6 8 \%$ . The comparison demonstrates a controllable operating trade-off within the same compressed encoder: rare-aware optimization increases minorityclass sensitivity while shifting the balance among aggregate accuracy, balanced accuracy, and macro-F1. Figure 11 further shows that the overall score is supported by strong discrimination in several well-represented classes: prostate adenocarcinoma reaches F1=0.996, testicular germ-cell tumors F1=0.986, thyroid carcinoma F1=0.986, kidney renal clear-cell carcinoma F1=0.963, and lower-grade glioma F1=0.961. The most difficult classes include rectum adenocarcinoma (F1=0.174), mesothelioma (0.378), esophageal carcinoma (0.439), and cholangiocarcinoma (0.552). The normalized confusion matrix confirms that errors are concentrated in a limited subset of class pairs rather than being uniformly distributed across the 32-class taxonomy. This heterogeneity motivates reporting macro-F1 and rare-class BA alongside top-1 accuracy. The primary paper result therefore remains the single-head mild-CE configuration in Table 6, with the rare-aware result used to explain the robustness/imbalance behavior of the architecture.

![](images/4db73ba543464d09ad1c393aaaeece247d2ad38726a8bd62af5885a883ef2e4c.jpg)  
Figure 10: Validation trade-off between overall accuracy and rare-class balanced accuracy for alternative task-head objectives on the fixed TAP-Path encoder.

## 4.6 External validation

External validation is reported as two distinct disease cohorts. CPTAC-CCRCC contains 209 independent tumor cases mapped to the internal kidney renal clear-cell carcinoma class, and CPTAC-UCEC contains 224 independent tumor cases mapped to uterine corpus endometrial carcinoma. No CP-TAC image or label is used for architecture selection, head optimization, calibration, or threshold tuning.

Across the three primary single-head seeds, TAP-Path achieves 87.40±0.28% accuracy on CCRCC and 94.79±1.44% on UCEC. Pooled over 433 cases, accuracy is 91.22±0.83% and present-class balanced accuracy is 91.10±0.81%. Because each cohort corresponds to one mapped target class, cohortspecific recalls/accuracies can be recovered exactly from each frozen seed using

$$
\mathrm { B A } _ { \mathrm { e x t } } = \frac { a _ { C } + a _ { U } } { 2 } ,\tag{33}
$$

$$
\mathrm { A c c } _ { \mathrm { e x t } } = \frac { 2 0 9 a _ { C } + 2 2 4 a _ { U } } { 4 3 3 } .\tag{34}
$$

Table 10 reports the cohort-specific values and Fig. 12 contrasts internal and external behavior. The cohort difference is meaningful: UCEC transfers more strongly than CCRCC under the locked model, demonstrating why external performance should not be collapsed into a single unqualified robustness claim. The result supports frozen transfer across these two morphologies, not universal 32-class external generalization.

Frozen external evaluation was performed on 433 CPTAC images after the internal architecture and task protocol were locked. Across three single-head seeds, TAP-Path achieved $9 1 . 2 2 \pm 0 . 8 3 \%$ accuracy and $9 1 . 1 0 \pm 0 . 8 1 \%$ balanced accuracy. External ECE was $0 . 0 3 2 3 \pm 0 . 0 0 8 6$ and failure-detection AUROC was $0 . 8 9 7 4 \pm 0 . 0 0 8 9$ . These values indicate that the compressed representation remained useful outside the internal data source.

The external task is simpler in class cardinality than the internal 32-class benchmark, so its higher raw accuracy should not be interpreted as evidence that CPTAC is “harder” or that generalization improves with domain shift. Instead, the correct interpretation is that the locked model transferred successfully to the two externally represented classes. Among the retained baselines, Hibou-B obtained a slightly higher external accuracy (91.53%), while TAP-Path exceeded Virchow2 and UNI2-h. The external experiment therefore provides evidence of transfer robustness for the two represented classes.

## 4.7 Representation analysis

As shown in Fig. 13, PCA is used as a descriptive visualization of the locked compressed representation, not as a training signal. For standardized image-level feature matrix Z, the kth component is

$$
w _ { k } = \arg \operatorname* { m a x } _ { \| w \| _ { 2 } = 1 , w \bot w _ { < k } } \mathrm { V a r } ( Z w ) .\tag{35}
$$

The projection in Fig. 13 shows that broad class-dependent structure remains visible after depth and token reduction, although substantial overlap is expected for a 32-class pan-cancer task.

## 4.8 Deployment efficiency

The final compressed encoder has 473.70M parameters and the single task head has 5.70M, giving 479.40M deployed parameters. Analytical encoder compute is 220.40G FLOPs. An independent profiler supported approximately 213.98G FLOPs for the measured execution path. On the NVIDIA GeForce RTX 5060 Ti development system, mean inference latency was 31.85 ms with 1.21 ms standard deviation, corresponding to 31.40 images/s. These numbers are hardware-specific and should not be generalized to clinical scanners or server deployments, but they confirm that the structural and token changes translate into an executable compressed model rather than only theoretical masking.

## 4.9 Bootstrap uncertainty

Figure 14 summarizes the resulting uncertainty. The 2,000- resample stratified bootstrap gives approximate 95% intervals of 0.870–0.889 for accuracy, 0.793–0.831 for balanced accuracy, 0.805–0.839 for macro-F1, 0.568–0.698 for rare-class balanced accuracy, and 0.018–0.032 for ECE. The wider rare-class interval reflects the smaller effective support of low-frequency classes.

Table 9: Unified ablation study. The table intentionally combines structural, token, and rare-aware objective experiments rather than creating a separate headline table for the rare-aware operating point. Screening values are validation-only and should not be compared directly with final 12-patch test results.
<table><tr><td>Component</td><td>Variant</td><td>Acc. (%)</td><td>BA (%)</td><td>Macro-F1 (%)</td><td>Rare BA (%)</td><td>FLOPs (G)</td></tr><tr><td rowspan="4">Structure</td><td>Depth32</td><td>72.96</td><td>73.08</td><td>71.67</td><td></td><td>340.13</td></tr><tr><td>Depth24</td><td>68.28</td><td>68.38</td><td>67.30</td><td></td><td>255.19</td></tr><tr><td>TaskSparse22</td><td>69.18</td><td>69.30</td><td>68.01</td><td></td><td>233.96</td></tr><tr><td>TaskSparse24</td><td>69.79</td><td>69.84</td><td>68.54</td><td></td><td>255.19</td></tr><tr><td rowspan="3">Token ratio</td><td>100%</td><td>69.79</td><td>69.84</td><td>68.54</td><td></td><td>255.19</td></tr><tr><td>85%</td><td>70.39</td><td>70.42</td><td>69.10</td><td></td><td>237.71</td></tr><tr><td>70%</td><td>71.60</td><td>71.64</td><td>70.32</td><td></td><td>220.40</td></tr><tr><td rowspan="5">Loss (validation)</td><td>Mild CE</td><td>89.99</td><td>82.82</td><td>84.47</td><td>67.39</td><td>220.40</td></tr><tr><td>Balanced Softmax</td><td>89.19</td><td>83.17</td><td>81.51</td><td>70.55</td><td>220.40</td></tr><tr><td>Logit adjustment τ = .5</td><td>89.94</td><td>83.04</td><td>84.24</td><td>67.74</td><td>220.40</td></tr><tr><td>Logit adjustment τ = 1</td><td>90.25</td><td>83.84</td><td>83.22</td><td>69.68</td><td>220.40</td></tr><tr><td>CB focal  $\gamma = 1 . 5$ </td><td>87.35</td><td>81.27</td><td>81.76</td><td>68.33</td><td>220.40</td></tr><tr><td>Rare-aware 3-seed test</td><td>selected  $\tau = 1$  head</td><td>87.13±.68</td><td>82.29±.38</td><td>80.74±.65</td><td>68.64±2.16</td><td>220.40</td></tr></table>

Table 10: Frozen TAP-Path validation on the two external CPTAC cohorts. Values are mean±SD across the three primary single-head seeds.
<table><tr><td>Cohort</td><td>Cases</td><td>Accuracy (%)</td></tr><tr><td>CPTAC-CCRCC</td><td>209</td><td> $\mathbf { 8 7 . 4 0 { \pm } 0 . 2 8 }$ </td></tr><tr><td>CPTAC-UCEC</td><td>224</td><td>94.79±1.44</td></tr><tr><td>Pooled CPTAC</td><td>433</td><td>91.22±0.83</td></tr><tr><td>Present-class BA</td><td>433</td><td>91.10±0.81</td></tr></table>

## 5 Discussion

## 5.1 Whole-slide efficiency

The reported FLOP reduction is measured per encoder invocation, but digital pathology multiplies this saving across patch bags. Although the present experiments evaluate fixed imagelevel patch bags rather than end-to-end WSI inference, the per-patch analytical FLOP reduction can be extrapolated to illustrate potential whole-slide computational savings. If a WSI contributes $N _ { \mathrm { W S I } }$ encoded patches,

$$
\Phi _ { \mathrm { s l i d e } } \approx N _ { \mathrm { W S I } } \Phi _ { \mathrm { p a t c h } } .\tag{36}
$$

For 10,000 encoded patches, reducing analytical cost from 340.13G to 220.40G FLOPs corresponds to roughly 1 $. 2 0 \times 1 0 ^ { 1 5 }$ fewer floating-point operations before downstream aggregation. This is a theoretical compute estimate rather than a latency guarantee, yet it illustrates why a 35.20% per-patch reduction is meaningful at WSI scale.

## 5.2 Structural compression rationale

The structural compression results demonstrate that TAP-Path establishes a favorable accuracy–efficiency operating point derived directly from a high-capacity pathology foundation model. Relative to the full Virchow2 backbone, TAP-Path removes approximately one quarter of the encoder parameters and more than one third of the analytical encoder FLOPs while maintaining strong predictive performance. Importantly, this reduction is achieved through task-adaptive physical reconstruction of the encoder rather than through masking or freezing redundant components. The resulting architecture therefore preserves the representational advantages of largescale pathology pretraining while reducing the computational burden associated with downstream inference. The concurrent improvements in test accuracy and macro-F1 further indicate that the removed structural components are not essential for the target classification task and that task-adaptive compression can produce a more efficient downstream representation without sacrificing predictive utility.

This property distinguishes TAP-Path from parameterefficient fine-tuning approaches. Adapter- and prompt-based methods can substantially reduce the number of parameters updated during task adaptation, and approaches such as PAMT demonstrate the effectiveness of this strategy in computational pathology [30]. However, these methods primarily reduce optimization cost while retaining most of the pretrained backbone for inference. In contrast, TAP-Path directly reconstructs a smaller executable encoder by removing task-redundant transformer blocks, thereby reducing both deployed parameter count and analytical inference cost. This distinction is particularly relevant for pathology applications in which large numbers of image regions may need to be processed repeatedly and inference efficiency becomes an important deployment consideration.

Knowledge-distillation approaches provide another complementary route toward model compression. Methods such as

![](images/c9f47374c315795d8d7bb07c5c47755e3fdbafa13b00346bbc2bdb48ccbf4c82.jpg)  
(a) Per-class F1. Rare classes are highlighted with the contrasting bar color.

![](images/06eeb13d1facdc5b67d373edc65db7fe65c994ccbfc2fe93ccd4c83953e2e04f.jpg)  
(b) Row-normalized 32-class confusion matrix.

Figure 11: Class-level evaluation of TAP-Path on the locked internal test set. The balanced panel layout combines class-wise F1 and normalized confusion behavior to expose performance heterogeneity that is not visible from aggregate accuracy alone.  
![](images/26946aef96ed2ffdb834b48f1f026622328245e3835868cbdf0a8fae1bddc082.jpg)  
Figure 12: Frozen external validation of TAP-Path on CPTAC-CCRCC and CPTAC-UCEC.

![](images/37ca261532f4e7a2666e2c70ba820d2937fae68359479c9d012070c18ac616af.jpg)  
Figure 13: PCA of held-out TAP-Path image-level representations.

SUDA transfer knowledge from a larger teacher into a compact student architecture [16], enabling substantial reductions in model size through an additional teacher–student optimization stage. TAP-Path addresses a different compression objective: rather than replacing the pretrained foundation model with a separately trained student, it identifies and preserves the task-relevant structure within the pretrained model itself. Consequently, structural pruning, parameter-efficient adaptation, and knowledge distillation represent complementary strategies operating at different stages of the model lifecycle. Their combination provides a promising direction for future work, particularly through hybrid pruning–distillation frameworks that first remove task-redundant structure and subsequently transfer the retained knowledge into an even smaller deploy-

ment model.

## 5.3 Non-contiguous depth selection

The structural screen provides evidence that depth alone is not an adequate proxy for task relevance. At equal 24-block parameter count, TaskSparse24 outperformed contiguous Depth24 during validation screening. The result suggests that useful transformations are distributed non-uniformly through the pretrained hierarchy and that preserving selected late/middle blocks can recover information lost by naive truncation. This is consistent with the broad observation that pretrained transformer representations change qualitatively across depth.

We deliberately anchored early and final blocks. The early anchor protects low-level patch/token formation, while the late anchor preserves semantic consolidation close to the original pretrained output. The middle selection is then allowed to adapt. Although the novelty score in Eq. (7) is simple, its practical advantage is that it is model-internal, label-efficient for profiling, and compatible with physical block removal. A limitation is that normalized residual change is not a causal attribution score; a block can have a small residual magnitude yet still be important through subtle feature refinement. Future work could compare the current score with gradient- or Hessian-based block saliency.

![](images/dae37a829734d2097bb002a114c49b94c133472f16091df65f8cb7b68c813904.jpg)  
Figure 14: Stratified 95% bootstrap confidence intervals for the locked TAP-Path test evaluation.

## 5.4 Token-retention behavior

The token ablation is particularly useful because the selected 70% ratio simultaneously improved screening validation accuracy and reduced compute for TaskSparse24. One plausible interpretation is that class-token similarity plus magnitude suppresses redundant or weakly aligned patches after sufficient hierarchical processing. Similar motivations underlie task-aware token pruning in generic vision transformers [17, 19]. However, this effect was not universal: TaskSparse22 degraded as tokens were removed. Token sparsity therefore interacts with representational depth and should be selected jointly rather than treated as an independent compression knob.

In pathology, this interaction deserves caution. Small discriminative regions can be clinically meaningful, so a high compression ratio could remove uncommon morphology. Our 70% setting was chosen by a validation Pareto rule and was not tuned on the test set. Moreover, rare-class behavior was evaluated explicitly. These safeguards reduce but do not eliminate the risk of token-pruning bias.

## 5.5 Reliability interpretation

The reliability analysis shows that the compressed model preserves informative probabilistic behavior alongside its efficiency gains. The single-head TAP-Path achieved the best Brier score and failure-detection AUROC among the directly compared large-model configurations, while ECE and NLL remained close to UNI2-h. A failure-detection AUROC of approximately 0.90 indicates effective ranking of incorrect predictions toward lower confidence, supporting confidenceguided selective review. Clinical operating thresholds and safety characteristics require prospective, workflow-specific evaluation beyond these statistical reliability measures.

Rare-class analysis further characterizes how the optimization objective changes model behavior. The primary objective favors aggregate predictive performance, whereas the rareaware head increases rare-class balanced accuracy with a corresponding shift in overall accuracy. Reporting the two operating points separately provides a direct view of the accuracy– minority-sensitivity trade-off and preserves the metric provenance of each configuration.

## 5.6 External transfer

Independent evaluation is increasingly considered essential for pathology foundation models [9, 11, 27]. The CPTAC results show that TAP-Path maintains high accuracy and balanced accuracy after freezing the internal configuration. The difference in class cardinality prevents direct comparison of internal 32-class macro-F1 with an external two-class macro-F1, which is why the manuscript emphasizes present-class balanced accuracy externally.

Across the independent CPTAC evaluation, TAP-Path maintained strong transfer performance after the internal configuration was frozen. Hibou-B achieved slightly higher raw CPTAC accuracy, consistent with prior evidence that foundation-model rankings vary across domains and tasks [11, 12]. For the studied 32-class task, the combined internal and external results position TAP-Path as an efficient high-capacity configuration with stable transfer behavior under the evaluated CPTAC cohorts.

## 5.7 Relation to efficient pathology methods

Recent pathology-efficiency studies provide important context for this work. SUDA achieves aggressive compression by transferring knowledge from a large teacher to a compact student [16], whereas PAMT reduces downstream computational requirements through representative patch selection and lightweight adaptation of frozen representations [30]. Other studies have explored resource-efficient architectures and accuracy–complexity trade-offs for histopathology analysis [15, 10]. Despite these advances, efficiency is commonly achieved through student-model distillation, input-level reduction, lightweight adaptation, or architectures specifically designed for efficient inference. TAP-Path addresses a complementary setting: task-adaptive compression of an already pretrained pathology foundation model. Rather than pretraining a new foundation model, distilling a separate student, or retaining the complete parent architecture, TAP-Path restructures the pretrained parent for the downstream task by preserving task-relevant depth, reducing redundant token computation, and recovering complementary intermediate representations within a single compressed inference pathway.

Accordingly, TAP-Path is evaluated through a joint accuracy–efficiency–reliability perspective that integrates predictive performance with computational cost and probabilistic behavior. The proposed framework preserves competitive predictive performance relative to substantially larger pathology foundation models while reducing both model size and computational demand. This efficiency is evaluated together with three-seed reproducibility, probability calibration, failureawareness analysis, and frozen external validation. Collectively, these evaluations examine whether a pretrained pathology foundation model can be structurally compressed for a specific downstream task while retaining predictive performance, computational efficiency, and informative confidence behavior.

## 6 Limitations

Several directions remain for extending the present evaluation. First, TAP-Path contains 479.40M deployed parameters, placing it between compact encoders such as Hibou-B/CONCH and larger Virchow2/UNI2-h systems. Further compression could target lower-resource deployment settings. Second, analytical FLOPs provide a reproducible architecture-level efficiency measure, while end-to-end deployment also depends on hardware, energy consumption, WSI throughput, slide loading, tissue detection, patch extraction, and storage I/O. The measured 31.85 ms image-level latency therefore characterizes the evaluated hardware configuration and motivates broader systems-level benchmarking.

Third, the current external evaluation covers CCRCC and UCEC, providing an independent transfer assessment for two classes represented in the internal taxonomy. Extending this protocol to additional cancer types, scanners, and institutions would enable broader pan-cancer robustness analysis. Fourth, the primary objective prioritizes aggregate performance, while the rare-aware analysis demonstrates that minority-class sensitivity can be increased within the same compressed encoder. This motivates future structural-selection criteria that incorporate class-frequency or minority-utility information directly during compression.

Fifth, the block-novelty score quantifies normalized representational change and serves as a computationally efficient task-selection criterion. Future work can compare this criterion with gradient-, Hessian-, or attribution-based structural importance measures and examine how retained-block patterns transfer across endpoints. Finally, the present reliability analysis is statistical and model-centered; prospective clinical evaluation, reader studies, workflow-specific decision thresholds, and formal safety assessment are natural next steps for establishing clinical utility.

## 7 Conclusion

We presented TAP-Path, a task-adaptive structural and token pruning framework for converting a large pretrained pathology foundation model into a more efficient task-specific encoder without knowledge distillation. Starting from Virchow2, the method profiles task-relevant block novelty, physically retains 24 of 32 transformer blocks, adaptively preserves 70% of patch tokens after the pruning point, and recovers multidepth information through a gated single task head. The resulting encoder reduces parameters by 24.96% and analytical FLOPs by 35.20%. Across three independent single-head runs, TAP-Path achieves 87.98±0.067% internal test accuracy and 82.38±0.48% macro-F1, matching or slightly exceeding much larger Virchow2 and UNI2-h baselines on the studied task. The compressed model also preserves useful probabilistic behavior, including a Brier score of 0.1800 and failure-detection AUROC of 0.9047, while frozen CPTAC evaluation reaches 91.22±0.83% accuracy.

The results support a practical conclusion: scaling pathology foundation models and compressing them for deployment are not mutually exclusive. A large pretrained hierarchy can contain downstream-task redundancy that is removable through validation-constrained structural and token sparsification. At the same time, compression should be evaluated beyond topline accuracy. Explicit reporting of calibration, error awareness, rare-class trade-offs, and independent external behavior is important for assessing whether the resulting efficiency gain remains suitable for trustworthy medical-AI research.

## Data Availability

The histopathology data analyzed in this study were derived from publicly available resources from The Cancer Genome Atlas (TCGA) [32] and the Clinical Proteomic Tumor Analysis Consortium (CPTAC) [33, 34]. TCGA data are accessible through the NCI Genomic Data Commons, while the corresponding CPTAC resources are available through NCIsupported data repositories. No external evaluation data were used for architecture selection or model optimization.

## A Reproducibility details

## A.1 Primary training configuration

## A.2 Experimental provenance

The manuscript separates architecture screening, primary task evaluation, common-probe diagnostics, rare-aware operatingpoint analysis, and external validation. Metrics are not silently transferred between these protocols. This distinction is essential because the common-probe analysis answers a representation-quality question, whereas the original TAP-Path task head defines the proposed deployed system.

## References

[1] R. J. Chen, T. Ding, M. Y. Lu, et al. “Towards a generalpurpose foundation model for computational pathology”.

Table 11: Primary training configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Parent foundation model</td><td>Virchow2</td></tr><tr><td>Retained blocks</td><td>24/32</td></tr><tr><td>Patch-token retention</td><td>70%</td></tr><tr><td>Multi-depth taps</td><td>4</td></tr><tr><td>Tap projection</td><td>256</td></tr><tr><td>Dropout</td><td>0.18</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>384</td></tr><tr><td>Maximum epochs</td><td>130</td></tr><tr><td>Early stopping</td><td>18 epochs</td></tr><tr><td>Gradient clip</td><td>5</td></tr><tr><td>Label smoothing</td><td>0.01</td></tr><tr><td>Seeds</td><td>42, 123, 2026</td></tr><tr><td>Calibration</td><td>validation-only temperature scaling</td></tr></table>

In: Nature Medicine 30 (2024), pp. 850–862. DOI: 10. 1038/s41591-024-02857-3.

[2] M. Y. Lu, B. Chen, D. F. K. Williamson, et al. “A visuallanguage foundation model for computational pathology”. In: Nature Medicine 30 (2024), pp. 863–874. DOI: 10.1038/s41591-024-02856-4.

[3] E. Vorontsov, A. Bozkurt, A. Casson, et al. “A foundation model for clinical-grade computational pathology and rare cancers detection”. In: Nature Medicine 30 (2024), pp. 2924–2935. DOI: 10 . 1038 / s41591 - 024-03141-0.

[4] H. Xu, N. Usuyama, J. Bagga, et al. “A whole-slide foundation model for digital pathology from real-world data”. In: Nature 630 (2024), pp. 181–188. DOI: 10. 1038/s41586-024-07441-w.

[5] X. Wang, J. Zhao, E. Marostica, et al. “A pathology foundation model for cancer diagnosis and prognosis prediction”. In: Nature 634 (2024), pp. 970–978. DOI: 10.1038/s41586-024-07894-z.

[6] D. Nechaev, A. Pchelnikov, and E. Ivanova. “Hibou: A Family of Foundational Vision Transformers for Pathol ogy”. In: arXiv preprint arXiv:2406.05074 (2024).

[7] J. Ma, Z. Guo, F. Zhou, et al. “A generalizable pathology foundation model using a unified knowledge distillation pretraining framework”. In: Nature Biomedical Engineering 10 (2026). Published online 2025, pp. 545–564. DOI: 10.1038/s41551-025-01488-4.

[8] F. Yan, J. Wu, J. Li, et al. “PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks”. In: arXiv preprint arXiv:2503.24345 (2025).

[9] G. Campanella, S. Chen, M. Singh, et al. “A clinical benchmark of public self-supervised pathology foundation models”. In: Nature Communications 16 (2025), p. 3640. DOI: 10.1038/s41467-025-58796-1.

[10] J. Lee, J. Lim, K. Byeon, and J. T. Kwak. “Benchmarking pathology foundation models: Adaptation strategies and scenarios”. In: Computers in Biology and Medicine 190 (2025), p. 110031. DOI: 10.1016/j. compbiomed.2025.110031.

[11] P. Neidlinger, O. S. M. El Nahhas, H. S. Muti, et al. “Benchmarking foundation models as feature extractors for weakly supervised computational pathology”. In: Nature Biomedical Engineering 10 (2026). Published online 2025, pp. 1113–1123. DOI: 10.1038/s41551- 025-01516-3.

[12] R. Bareja et al. “A benchmark study of vision and pathology foundation models for computational pathology”. In: Nature Communications (2026). DOI: 10.1038/ s41467-026-76004-6.

[13] Kuang et al. “LW-CTrans: A lightweight hybrid network of CNN and Transformer for 3D medical image segmentation”. In: Medical Image Analysis 102 (2025), p. 103545. DOI: 10.1016/j.media.2025. 103545.

[14] J. Silva-Rodriguez et al. “Towards Foundation Models and Few-Shot Parameter-Efficient Fine-Tuning for Volumetric Organ Segmentation”. In: Medical Image Analysis 103, 103596 (2025). DOI: 10.1016/j.media. 2025.103596.

[15] M. F. Islam, M. T. Reza, M. A. Manab, et al. “Involutionbased efficient autoencoder for denoising histopathological images with enhanced hybrid feature extraction”. In: Computers in Biology and Medicine 192 (2025), p. 110174. DOI: 10.1016/j.compbiomed.2025. 110174.

[16] L. Zhong, K. Qian, W. Zhao, et al. “SUDA: Simultaneous unsupervised knowledge distillation and adaptation of foundation models for efficient pathological image analysis”. In: Medical Image Analysis 113 (2026), p. 104177. DOI: 10.1016/j.media.2026. 104177.

[17] Liu et al. “Revisiting Token Pruning for Object Detection and Instance Segmentation”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024, pp. 2658–2668.

[18] Xu et al. “GTP-ViT: Efficient Vision Transformers via Graph-Based Token Propagation”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024, pp. 86–95.

[19] Bergner et al. “Token Cropr: Faster ViTs for Quite a Few Tasks”. In: Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2025, pp. 9740–9750.

[20] An et al. “Sparse Structure Exploration and Reoptimization for Vision Transformer”. In: Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence. Vol. 286. PMLR, 2025, pp. 111–131.

[21] Marchetti et al. “Efficient token pruning in Vision Transformers using an attention-based Multilayer Network”. In: Expert Systems with Applications 279 (2025), p. 127449. DOI: 10 . 1016 / j . eswa . 2025 . 127449.

[22] M. Munir, M. M. Rahman, and R. Marculescu. “AdaptViG: Adaptive Vision GNN with Exponential Decay Gating”. In: arXiv preprint arXiv:2511.09942 (2025).

[23] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. “On Calibration of Modern Neural Networks”. In: Proceedings of the 34th International Conference on Machine Learning. Vol. 70. PMLR, 2017, pp. 1321–1330.

[24] Y. Ovadia, E. Fertig, J. Ren, et al. “Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift”. In: Advances in Neural Information Processing Systems. Vol. 32. 2019.

[25] Y. Geifman and R. El-Yaniv. “SelectiveNet: A Deep Neural Network with an Integrated Reject Option”. In: Proceedings of the 36th International Conference on Machine Learning. Vol. 97. PMLR, 2019, pp. 2151– 2159.

[26] Y. Huang, W. Zhao, Z. Zhang, et al. “Knowledge-guided adaptation of pathology foundation models effectively improves cross-domain generalization and demographic fairness”. In: Nature Communications (2025). DOI: 10. 1038/s41467-025-66300-y.

[27] J. Komen, E. D. de Jong, J. Hense, et al. “Towards robust foundation models for digital pathology”. In: Nature Communications 17 (2026), p. 5218. DOI: 10.1038/ s41467-026-73923-2.

[28] M. Jahanifar, S. E. A. Raza, et al. “Domain Generalization in Computational Pathology: Survey and Guidelines”. In: ACM Computing Surveys 57.11 (2025). DOI: 10.1145/3729524.

[29] Z. Yang, T. Wei, Y. Liang, et al. “A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images”. In: Nature Communications 16, 2366 (2025). DOI: 10.1038/s41467- 025-57587-y.

[30] Y. Lin, Z. Zhu, K.-T. Cheng, and H. Chen. “Promptguided foundation model tuning for pathology image classification”. In: Medical Image Analysis 113 (2026), p. 104214. DOI: 10 . 1016 / j . media . 2026 . 104214.

[31] S. Boudissa, S. S. Debsarkar, H. Kawanaka, B. J. Aronow, and V. B. S. Prasath. “Vision Transformers for Histopathological Image Classification with Efficient Head Pruning”. In: Procedia Computer Science 270 (2025), pp. 2274–2283. DOI: 10.1016/j.procs. 2025.09.348.

[32] The Cancer Genome Atlas Research Network et al. “The Cancer Genome Atlas Pan-Cancer analysis project”. In: Nature Genetics 45.10 (2013), pp. 1113–1120. DOI: 10. 1038/ng.2764.

[33] D. J. Clark et al. “Integrated Proteogenomic Characterization of Clear Cell Renal Cell Carcinoma”. In: Cell 179.4 (2019), 964–983.e31. DOI: 10.1016/j.cell. 2019.10.007.

[34] Y. Dou et al. “Proteogenomic Characterization of Endometrial Carcinoma”. In: Cell 180.4 (2020), 729– 748.e26. DOI: 10.1016/j.cell.2020.01.026.