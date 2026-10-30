# LORE-MIM: LOST-OR-REPLACED MASKED IMAGEMODELING FOR SELF-SUPERVISED REPRESENTATIONLEARNING IN WHOLE-BODY PET/CT

A PREPRINT

Amartya Bhattacharya<sup>\*</sup> Department of Biomedical Data Science Geisel School of Medicine at Dartmouth Hanover, NH 03755, USA Amartya.Bhattacharya.GR@dartmouth.edu

Biratal Raj Wagle   
Department of Biomedical Data Science   
Geisel School of Medicine at Dartmouth Hanover, NH 03755, USA   
Biratal.Raj.Wagle@dartmouth.edu Matthew E. Maeder Radiology   
Dartmouth Hitchcock Medical Center Lebanon, NH 03766 , USA   
Matthew.E.Maeder@dartmouth.edu

Bashirul Azam Biswas<sup>\*</sup> Department of Biomedical Data Science Geisel School of Medicine at Dartmouth Hanover, NH 03755, USA Bashirul.Azam.Biswas@dartmouth.edu

Michael S. Leapman   
Department of Urology   
Yale University   
New Haven, CT, USA   
michael.leapman@yale.edu

James B. Yu Radiation Oncology Dartmouth Hitchcock Medical Center Lebanon, NH 03766 , USA James.B.Yu@dartmouth.edu

Indrani Bhattacharya Department of Biomedical Data Science Geisel School of Medicine at Dartmouth Hanover, NH 03755, USA Indrani.Bhattacharya@dartmouth.edu

## ABSTRACT

Deep learning models for whole-body positron emission tomography/computed tomography (PET/CT) hold significant clinical potential for disease detection, treatment planning, and response assessment. However, the substantial time and cost associated with expert annotations remain major bottlenecks in developing robust and generalizable models that can assist with PET/CT interpretation. Although self-supervised learning (SSL) has emerged as a promising approach to mitigating annotation scarcity in natural and medical imaging, its application to whole-body PET/CT remains relatively underexplored due to the limited availability of large-scale datasets and the challenges associated with learning complementary cross-modal representations. To address these limitations, we curate a heterogeneous, multi-tracer, multi-institutional cohort comprising 6,331 whole-body PET/CT scans and propose LoRe-MIM (Lost-Or-Replaced Masked Image Modelling), a novel self-supervised framework designed to learn multimodal representations through masked image reconstruction. Specifically, LoRe-MIM introduces a masking strategy in which image regions are either replaced with zeros (Lost) or substituted with corresponding intensity values from the complementary imaging modality (Replaced), encouraging the model to capture both modality-specific characteristics and cross-modal relationships. We systematically evaluate the learned representations

across multiple downstream tasks, including multimodal PET/CT image reconstruction, patch-based classification into cancer vs. non-cancer, whole-body lesion segmentation, label-efficient training, and robustness assessment to noise, missing modality, and PET/CT misalignment. We assess generalizability in performance across tracers and different institutional datasets. Experimental results demonstrate that LoRe-MIM consistently outperforms the corresponding state-of-the-art baselines across these tasks, highlighting its potential as a scalable and generalizable self-supervised learning framework for whole-body PET/CT analysis.

Keywords PET/CT, self-supervised learning, multi-tracer, cross-modal learning

## 1 Introduction

Cancerous lesions in whole-body imaging exhibit substantial morphological variability in their physical appearance, shape, size, and structural characteristics across patients, cancer types, and anatomical sites [1–3]. Noninvasive imaging technologies such as positron emission tomography/computed tomography (PET/CT) play a critical role in detecting such lesions, determining disease stage, guiding treatment planning, and assessing treatment response [4]. PET captures the molecular and physiological activity of lesions, whereas CT provides complementary anatomical and structural context. However, with the rising global cancer burden [5], the volume of whole-body PET/CT examinations is increasing at a substantially faster pace than the availability of expert nuclear medicine specialists capable of interpreting them. This growing imbalance creates a critical need for automated systems that can efficiently and accurately analyze large volumes of whole-body PET/CT scans and assist clinicians. As in many other medical imaging domains, deep-learning-based methods have emerged as a promising approach for automating oncologic whole-body PET/CT interpretation [6–8].

Deep-learning-based models have been developed for whole-body lesion segmentation [9–17] and patch-level classifications tasks (cancer presence or absence) [18–21, 21–25]. The accuracy and generalizability of deep learning methods for both tasks depend heavily on the ability of the learned PET/CT imaging representations to distinguish between cancer and non-cancer voxels, capture diversity and heterogeneity in imaging features across patient populations, cancer types, tracers, scanners, and anatomical sites, and the robustness to noise, registration errors, and missing modality. Two straightforward approaches to learning such representations are supervised training with task-specific annotations and transfer learning from a model pretrained on a related domain.

Supervised deep learning approaches have been extensively investigated for lesion segmentation in whole-body PET/CT imaging [9–17]. Blanc-Durand et al. [10] employed a 3D U-Net architecture for the automated detection and segmentation of lymphoma lesions in whole-body FDG-PET/CT scans. Jemaa et al. [11] introduced a multistage segmentation framework combining 2D segmentation, connected-component analysis, and 3D segmentationbased refinement to improve lesion delineation in whole-body FDG-PET/CT images. To facilitate advancements in whole-body tumor segmentation, Gatidis et al. [13] organized the AutoPET-II challenge and made publicly available a collection of whole-body FDG-PET/CT scans representing diverse tumor types. Building upon this dataset, He et al. [12] incorporated camouflaged object detection strategies, while Liu et al. [14] developed hybrid Mamba-CNN architectures for automated lesion detection and segmentation. Although FDG remains the predominant radiotracer in deep learning-based whole-body PET/CT segmentation research, alternative tracers, including <sup>68</sup>Ga-Pentixafor [9] and prostate-specific membrane antigen (PSMA)-targeted tracers [15, 16], have also been investigated in some studies. Specifically, Xu et al. [9] designed a cascaded W-Net architecture comprising two V-Nets [17], where the first network processes CT images and the second integrates <sup>68</sup>Ga-Pentixafor PET with CT information for whole-body multiple myeloma detection. For whole-body PSMA-PET/CT lesion segmentation, Li et al. [16] proposed a hybrid 3D Transformer–CNN framework, whereas Xu et al. [15] employed U-Net-based architectures. Collectively, these su pervised PET/CT segmentation approaches leverage whole-body imaging to identify and delineate lesions associated with single or multiple tumor types.

Similar to lesion segmentation, earlier works on lesion classification relied on supervised training of convolutional neural networks (CNNs) [18–21], whereas several recent works [21–25] have explored transfer learning from pretrained models. Sibille et al. proposed a CNN to localize foci of increased FDG uptake and classify lesions suspicious for lung cancer [18]. Mu et al. developed a supervised convolutional network using FDG PET/CT scans to classify epidermal growth factor receptor (EGFR) mutation status in patients with non-small cell lung cancer (NSCLC) [19]. Shao et al. trained a dual-stream 3D CNN using <sup>18</sup>F-FDG PET/CT to distinguish benign and invasive adenocarcinoma in ground-glass lung nodules [20]. Alves et al. investigated three 3D CNN architectures inspired by the classical 2D architectures AlexNet [26], VGGNet [27], and Inception-v2 [28] for pulmonary nodule classification using <sup>18</sup>F-FDG PET/CT images [21]. They also investigated transfer learning from ResNet-50 [29] pretrained on ImageNet [30]. Tao et al. proposed an ensemble ResNet-based classification framework that leveraged transfer learning to extract feature from CT, PET, and fused PET/CT images, integrating individual classifiers through majority voting for lung tumor diagnosis [22]. Yang et al. investigated six ImageNet-pretrained CNN architectures for distinguishing metastatic and lymphomatous cervical lymph nodes using PET/CT images, employing both fine-tuned CNN classifiers and pretrained deep feature extractors combined with support vector machines [23]. Nishigaki et al. fine-tuned a Vision Transformer (ViT) pretrained on ImageNet-21k to classify benign and malignant slices from <sup>18</sup>F-FDG PET/CT scans, demonstrating superior performance compared with pretrained CNN architectures, including DenseNet and Efficient-Net [25]. Recognizing the limitations of transferring representations learned from 2D natural images to 3D medical imaging, Shao et al. explored transfer learning from Models Genesis for predicting EGFR mutation status in NSCLC patients [24].

As observed in existing whole-body PET/CT studies, lesion segmentation is commonly formulated as a fully supervised dense prediction task, whereas lesion classification often leverages representations transferred from models pretrained on large external datasets such as ImageNet or natural-image foundation models. Supervised learning, however, is constrained by the difficulty of constructing sufficiently large and diverse annotated datasets, as reliable lesion-level and voxel-level annotations are expensive to obtain and require substantial sub-speciality clinical expertise. Transfer learning from the 2D natural-image domain to 3D medical imaging introduces a different limitation: a substantial domain mismatch arising from differences in image dimensionality, visual appearance, intensity distribu tions, and underlying anatomical and functional characteristics.

Self-supervised pretraining offers a unified alternative by exploiting large collections of unlabeled PET/CT scans to learn domain-specific representations that can subsequently be adapted to segmentation through fine-tuning or used as feature extractors for lesion classification. In recent years, several studies have explored self-supervised pretraining for whole-body PET/CT lesion analysis [31–35]. Yazdani et al. adopted self-supervised SwinUNETR pretraining using 752 whole-body [<sup>68</sup>Ga]-PSMA-11 PET/CT scans for prostate cancer metastasis segmentation and demonstrated the benefit of pretraining for downstream lesion delineation [31]. Oh et al. applied contrastive-learningbased pretraining using 1,292 samples from the publicly available AutoPET III dataset and evaluated its benefit fo both lesion segmentation and staging classification on a private institutional cohort [32]. Huang et al. proposed a modality-complementary masking strategy to learn cross-modal PET/CT representations using 507 FDG cases from AutoPET III [33]. Liu et al. similarly adopted modality-complementary masking for an FDG-focused foundation model pretrained on 4,997 scans [35]. Keshavarz et al. employed multimodal SwinUNETR pretraining using 717 [<sup>18</sup>F]FDG PET/CT scans for downstream segmentation of diffuse large B-cell lymphoma (DLBCL) [34]. Collectively, these studies demonstrate the potential of self-supervised pretraining to alleviate annotation scarcity in whole-body PET/CT imaging. However, current self-supervised pretraining approaches remain limited in the scale and diversity of pretraining data, cross-modal interaction modeling, breadth and clinical comprehensiveness of downstream evaluation as detailed below.

1. Limited pretraining scale and tracer diversity: A major advantage of self-supervised learning is its ability to exploit large and heterogeneous pools of unlabeled data. However, several existing PET/CT pretraining studies have been conducted using relatively modest cohorts, including 752, 1,292, 507, and 717 scans in [31–34], respectively. In addition to the limited scale, most existing approaches are developed using a single tracer, with FDG-based cohorts used in [33–35] and PSMA-based data used in [31]. Although Oh et al. incorporated data with greater tracer diversity [32], dataset size was modest (1,292) limiting diverse feature learning, and systematic evaluation across FDG and PSMA tracers were not explored. Together, the limited cohort size and tracer diversity may constrain the ability of self-supervised pretraining to capture the full spectrum of anatomical, pathological, image acquisition related, and tracer-specific and tracer-agnostic variations encountered in clinical PET/CT imaging. As new radiotracers are being developed and used for different clinical applications, such limited training and tracer-specific models may not yield generalizable representations that are tracer-adaptable. Larger and more heterogeneous multi-institution, multi-tracer co horts are therefore needed to develop more robust and transferable PET/CT representations.

2. Limited cross-modal interaction modeling: PET and CT provide complementary information, with PET capturing molecular and physiological activity and CT providing anatomical and structural context. Features in both modalities also vary based on the anatomical region in whole-body and whether the region contains cancer or not. Clinicians use these imaging features on both modalities during interpretation, but, most existing pretraining approaches do not explicitly learn these relationships between PET and CT images. Yazdani et al. [31], for example, followed the original SwinUNETR pretraining proxy tasks, including image rotation prediction, image volume inpainting, and contrastive coding [36], while Keshavarz et al. adopted a conventional masked-reconstruction strategy without explicitly modeling modality-specific interactions. More recent approaches [32,33,35] introduce modality-complementary masking, in which one modality is masked while the complementary modality remains available. Although this strategy encourages cross-modal information learning, its robustness under modality corruption, imaging noise, and distribution shifts remains insufficiently characterized.

3. Narrow and clinically limited downstream evaluation: A robust pretrained representation should transfer across multiple clinically relevant tasks and remain reliable under heterogeneous and potentially degraded imaging conditions (e.g., poor image quality, artifacts, PET-CT misalignment due to patient movement, or missing modality). However, the majority of existing PET/CT self-supervised pretraining studies assess their representations primarily through lesion segmentation [31, 33–35]. Oh et al. additionally evaluate staging classification [32], but lesion-level classification tasks, such as distinguishing malignant from normal patches, remain comparatively underexplored. Such region-level classification is clinically important in triage settings for flagging if a patient has cancer in a specific region or in the whole body, that may warrant more detailed investigation. Moreover, segmentation performance is commonly evaluated using voxel-level metrics such as Dice score, intersection-over-union (IoU), and Hausdorff distance. While these measures quantify spatial overlap, they do not fully capture clinically meaningful performance for staging, treatment planning, and response assessment, such as whether and how many individual lesions are correctly detected or missed, and whether specific anatomic regions contain cancer. Lesion-level sensitivity, specificity, precision, falsepositive burden, and anatomical-region-specific performance therefore provide important clinically-relevant information. Last but not the least, performance robustness under likely PET/CT image quality issues, including image noise, missing modality and PET/CT misalignment has received limited attention, despite being essential for assessing model reliability under real-world acquisition variability. A comprehensive evaluation of pretrained representations should therefore span multiple downstream tasks, clinically relevant lesion-level metrics, anatomically stratified analysis, and robustness testing under realistic imaging perturbations.

In this work, we address the above limitations of existing self-supervised pretraining approaches for whole-body PET/CT by developing and validating a novel, multimodal self-supervised learning method with context-aware joint unimodal reconstruction and cross-modal translation. Our method, Lost or Replaced Masked Image Modeling (LoRe-MIM), explicitly learns PET–CT relationships, unimodal structural consistency and anatomy from a large-scale, multitracer, multi-institutional pretraining cohort. We evaluate the learned representations through comprehensive downstream evaluation across segmentation and classification tasks. In addition to conventional predictive performance, we evaluate clinically relevant lesion-level performance and robustness under clinically-relevant imaging perturbations. Our main contributions are summarized as follows:

1. Large and heterogeneous PET/CT cohort: We combine publicly available datasets, including AutoPET-III $( N \bar { = } 1 , 6 1 1 )$ , SPADE $( N = 1 , 1 2 6 )$ , ViMED $( N = 1 , \bar { 9 9 } 5 )$ , and Deep-PSMA $( N = 2 0 0 )$ , with an institutional cohort from Dartmouth Hitchcock Medical Center (DHMC; N = 1, 599), yielding a total of 6,531 whole-body PET/CT scans, of which 5,910 are used for self-supervised pretraining and 621 for evaluation. The cohort spans multiple cancer types, including melanoma, lung cancer, lymphoma, and prostate cancer, as well as multiple PET tracers, including FDG and PSMA. This large-scale collection captures substantial realworld heterogeneity in patient populations, disease distributions, scanners, institutions, acquisition settings, and tracer characteristics, providing a diverse foundation for learning transferable PET/CT representations.

2. Lost or Replaced Masked Image Modeling (LoRe-MIM): We introduce a novel multimodal masked image modeling strategy called Lost or Replaced Masked Image Modelling (LoRe-MIM) that mimics a situation of solving two jumbled-up puzzles, with some puzzle pieces that are ‘lost’ and some pieces are ‘replaced’ with a corresponding piece from the other puzzle. Rather than relying solely on conventional masking [34] or keeping one modality fully intact while masking the other [33, 35], our approach combines standard zero-mean masking with cross-modal replacement of spatially corresponding PET and CT blocks. During cross-modal replacement, a corrupted block from one modality is replaced by the co-located content from the complementary modality while the reconstruction target remains the original block. This hybrid objective encourages the model to jointly learn intra-modality contextual structure through the zero-mean masking, and bidirectional PET↔CT correspondences through the context-aware translation within a unified self-supervised framework.

3. Extensive clinically relevant downstream evaluation: We evaluate the learned representations through a variety of clinically-relevant downstream tasks including (a) whole-body multimodal image reconstruction, (b) patch-level classification into cancer vs non-cancer, (c) robustness evaluation with patch-based classifica tion, (d) whole-body lesion segmentation, and (e) label-efficient segmentation training. We evaluate whole body reconstruction performance for both PET and CT images to assess how well the model learns modalityspecific anatomic features as well as PET-CT relationships. For patch-based classification tasks, we assess performance with clean as well as degraded imaging conditions to evaluate robustness. For lesion segmentation, beyond conventional voxel-level segmentation metrics, we report clinically relevant lesion-level performance and stratified results across anatomical regions, PET tracers, different cancer types, and different multi-institutional datasets.

Our experiments demonstrate that pretraining with large-scale, multi-tracer, pan-cancer heterogeneous PET/CT datasets improves lesion segmentation, label-efficient training, classification, robustness to image quality degradation and reconstruction. Our proposed LoRE-MIM improves performance across all downstream tasks over conventional MAE-style and cross-modal masking baselines, providing generalizable, tracer-adaptable representations that can be used for different PET/CT tasks, cancers and PET tracers.

## 2 Data and Methods

## 2.1 Datasets

We utilize five retrospective, de-identified PET/CT datasets, as summarized in Table 1. The combined cohort from AutoPET-III, DHMC, SPADE, and ViMED comprises 6,331 scans, partitioned into 4,823 training, 1,087 validation, and 421 held-out test scans. An additional 200 scans from Deep-PSMA are reserved as an out-of-domain test set for whole-body lesion segmentation.

## 2.1.1 Public datasets

AutoPET-III [6, 7] contains 1,611 whole-body PET/CT studies spanning FDG and PSMA tracers, multiple cancer types, and lesion-negative controls. Its FDG cohort includes melanoma, lung cancer, and lymphoma, whereas its PSMA cohort primarily represents prostate cancer. Expert-verified lesion annotations are available for downstream segmentation.

Deep-PSMA [37] contains 100 patients with metastatic prostate cancer, each imaged using both FDG- and PSMA-PET/CT before <sup>177</sup>Lu-PSMA therapy, yielding 200 expert-annotated scans. Deep-PSMA is excluded from pretraining and used only as an external segmentation test set.

SPADE [38] contains whole-body FDG-PET/CT examinations acquired during routine staging and restaging. VI-MED [39] provides additional FDG-PET/CT data with demographic and acquisition diversity. Because VI-MED stores CT intensities in a non-standard integer range and PET data as raw emission counts rather than standardized uptake values, all datasets are harmonized through a common preprocessing pipeline.

## 2.1.2 Internal dataset.

The internal DHMC cohort comprises 1,599 unlabeled PSMA-PET/CT scans from retrospective patients with or without prostate cancer treated at Dartmouth Hitchcock Medical Center in Lebanon, New Hampshire. The study was approved by the Dartmouth College and Dartmouth Hitchcock Medical Center Institutional Review Boards, with informed consent waived because only previously acquired, de-identified data were used.

## 2.2 Data allocation

Our pretraining uses the training and validation splits of AutoPET-III, DHMC, SPADE, and VI-MED. Reconstruction is evaluated without labels on the held-out AutoPET-III and DHMC test sets. Downstream segmentation is fine-tuned using only the AutoPET-III training and validation sets and evaluated on the held-out AutoPET-III and unseen Deep-PSMA test sets. Split assignment is patient-disjoint whenever repeated scans are available, preventing scans from the same patient from appearing in multiple splits.

## 2.3 Preprocessing and patch sampling

The model operates on paired, spatially aligned PET/CT volumes that are resampled to a common voxel spacing, cropped to the body region, and independently z-score normalized during preprocessing, following a strategy similar to Liu et al. [35]. During pretraining, one spatially corresponding patch of size $9 6 \times 1 2 8 \times 1 2 8$ voxels is uniformly sampled from each PET/CT pair. A new crop is sampled whenever a volume is accessed, exposing the model to different anatomical regions across epochs. If a volume is smaller than the required patch size along any dimension, reflect padding is applied before cropping. The resulting input is $\mathbf { X } \in \mathbb { R } ^ { 2 \times 9 6 \times \dot { 1 } 2 8 \times 1 2 \dot { 8 } }$ , containing the aligned CT and PET patches.

Table 1: Summary of datasets used. The FDG/PSMA row marks tracer availability (✓ present, ✗ absent). Lu=Lung, Ly=Lymphoma, Me=Melanoma, Pr=Prostate, Neg=Negative.
<table><tr><td>Properties</td><td>AutoPET-III</td><td>DeepPSMA</td><td>DHMC</td><td>SPADE</td><td>VI-MED</td></tr><tr><td>Disease dist.</td><td>Me/Lu/Ly/Pr/Neg 188/168/145/537/573</td><td>Pr</td><td>Pr</td><td></td><td></td></tr><tr><td>FDG/PSMA</td><td>√I√</td><td> $\overline { { \checkmark / \checkmark } }$ </td><td>X/√</td><td>√Ix</td><td>√Ix</td></tr><tr><td># lesions (per case)</td><td>17.47±46.12</td><td>69.99±60.04</td><td>10.13±16.27</td><td></td><td></td></tr><tr><td># slices (per vol)</td><td>135-963</td><td>195-1261</td><td>303-380</td><td>103-307</td><td>207-551</td></tr><tr><td>Pretraining</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Segmentation</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Pretraining-train/val/test scans</td><td>1043/247/321</td><td>0/0/0</td><td>1350/149/100</td><td>1013/113/0</td><td>1417/578/0</td></tr><tr><td>Segmentation-train/val/test</td><td>1043/247/321</td><td>0/0/200</td><td>0/0/0</td><td>0/0/0</td><td>0/0/0</td></tr></table>

## 2.4 Method

## 2.4.1 LoRE-MIM: Lost or Replaced Masked Image Modeling for PET/CT self-supervised pretraining

In this work, we aim to learn latent representations from whole-body PET/CT scans through self-supervised pretraining of an encoder–decoder architecture. Specifically, we adopt the Masked Image Modeling (MIM) paradigm for 3D medical imaging, in which paired and spatially aligned 3D patches are extracted from whole-body PET/CT scans and perturbed using a masking strategy. The model is then trained to reconstruct the original 3D patches from their perturbed counterparts. This reconstruction serves as a pretext task, enabling the model to learn informative representations that can subsequently be transferred to downstream tasks, such as lesion segmentation and classification. Mathematically, given a unlabelled dataset $\mathcal { D } _ { \mathcal { U } } = \{ ( X _ { \mathrm { P E T } } ^ { ( i ) } , X _ { \mathrm { C T } } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ comprising N paired 3D PET and CT patches, where $X _ { \mathrm { P E T } } ^ { \left( i \right) } , X _ { \mathrm { C T } } ^ { \left( i \right) } \in \mathbb { R } ^ { D \times W \times H }$ , with D, W, and H denoting the depth, width, and height of each patch, respectively, we aim to pretrain a model $f _ { \theta }$ , parameterized by θ, to jointly reconstruct the original PET and CT patches from their perturbed counterparts, $\tilde { X } _ { \mathrm { P E T } } ^ { ( i ) }$ and $\tilde { X } _ { \mathrm { C T } } ^ { ( i ) }$ , as follows:

$$
( \hat { X } _ { \mathrm { P E T } } ^ { ( i ) } , \hat { X } _ { \mathrm { C T } } ^ { ( i ) } ) = f _ { \theta } \left( \tilde { X } _ { \mathrm { P E T } } ^ { ( i ) } , \tilde { X } _ { \mathrm { C T } } ^ { ( i ) } \right)\tag{1}
$$

where $\hat { X } _ { \mathrm { P E T } } ^ { ( i ) }$ and $\hat { X } _ { \mathrm { C T } } ^ { ( i ) }$ denote the reconstructed 3D PET and CT patches, respectively. The model parameters are optimized by minimizing the reconstruction loss between the original and reconstructed patches:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { \mathrm { r e c } } \left( ( \hat { X } _ { \mathrm { P E T } } ^ { ( i ) } , \hat { X } _ { \mathrm { C T } } ^ { ( i ) } ) , ( X _ { \mathrm { P E T } } ^ { ( i ) } , X _ { \mathrm { C T } } ^ { ( i ) } ) \right)\tag{2}
$$

Here, $\mathcal { L } _ { \mathrm { r e c } }$ denotes the reconstruction loss function, which quantifies the discrepancy between the original and recon structed 3D PET/CT patches.

Given a pair of spatially aligned PET and CT patches, X<sub>PET</sub>, $X _ { \mathrm { C T } } \ \in \ \mathbb { R } ^ { D \times H \times W }$ , we partition each patch into B non-overlapping spatial blocks and represent the patches as vectors of their constituent blocks:

$$
\begin{array} { r l } & { \mathbf { X } _ { \mathrm { P E T } } = \left[ \mathbf { x } _ { \mathrm { P E T } } ^ { \left( 1 \right) } , \mathbf { x } _ { \mathrm { P E T } } ^ { \left( 2 \right) } , \dotsc , \mathbf { x } _ { \mathrm { P E T } } ^ { \left( B \right) } \right] , } \\ & { } \\ & { \mathbf { X } _ { \mathrm { C T } } = \left[ \mathbf { x } _ { \mathrm { C T } } ^ { \left( 1 \right) } , \mathbf { x } _ { \mathrm { C T } } ^ { \left( 2 \right) } , \dotsc , \mathbf { x } _ { \mathrm { C T } } ^ { \left( B \right) } \right] , } \end{array}\tag{3}
$$

where $\mathbf { x } _ { \mathrm { P E T } } ^ { ( b ) }$ and $\mathbf { x } _ { \mathrm { C T } } ^ { ( b ) }$ denote the b-th spatially corresponding blocks from PET and CT, respectively, and B represents the total number of blocks within each patch. If each block has spatial dimensions $d \times h \times w$ , then

$$
B = { \frac { D H W } { d h w } } .\tag{4}
$$

Next, we define two binary masking vectors, $\mathbf { M } _ { \mathrm { P E T } }$ and $\mathbf { M } _ { \mathrm { C T } }$ , for PET and CT, respectively, as follows:

$$
\mathbf { M } _ { \mathrm { P E T } } = \left[ m _ { \mathrm { P E T } } ^ { ( 1 ) } , m _ { \mathrm { P E T } } ^ { ( 2 ) } , \dots , m _ { \mathrm { P E T } } ^ { ( B ) } \right] ,\tag{5}
$$

$$
\mathbf { M } _ { \mathrm { C T } } = \left[ m _ { \mathrm { C T } } ^ { ( 1 ) } , m _ { \mathrm { C T } } ^ { ( 2 ) } , \dots , m _ { \mathrm { C T } } ^ { ( B ) } \right] .
$$

Here, $m _ { k } ^ { ( b ) } \in \{ 0 , 1 \}$ , where $k \in \{ \mathrm { P E T } , \mathrm { C T } \}$ , indicates whether the b-th block is selected for masking $( m _ { k } ^ { ( b ) } = 1 )$ or remains unmasked $( m _ { k } ^ { ( b ) } = 0 )$ . The two masking vectors are independently sampled, with exactly K blocks selected from each modality.

We introduce two modality-specific binary random vectors, $Z _ { k } = [ z _ { k } ^ { ( 1 ) } , \dots , z _ { k } ^ { ( B ) } ]$ , where $k \in \{ \mathrm { P E T } , \mathrm { C T } \}$ denotes the imaging modality and $z _ { k } ^ { ( b ) } \sim$ Bernoulli(0.5), to independently determine whether each selected block undergoes a Lost or Replaced perturbation with equal probability. Each pair of spatially corresponding PET and CT blocks, $\mathbf { x } _ { \mathrm { P E T } } ^ { ( b ) }$ and $\mathbf { x } _ { \mathrm { C T } } ^ { ( b ) }$ , is perturbed according to its modality-specific masking indicator $m _ { k } ^ { ( b ) }$ and binary perturbation indicator $z _ { k } ^ { ( b ) }$ and we achieve the perturbed PET and CT blocks, $\widetilde { \mathbf { x } } _ { \mathrm { P E T } } ^ { ( b ) }$ and $\widetilde { \mathbf { x } } _ { \mathrm { C T } } ^ { ( b ) }$ as follows:

$$
\begin{array} { r l } & { \widetilde { \mathbf { x } } _ { \mathrm { C T } } ^ { ( b ) } = ( 1 - m _ { \mathrm { C T } } ^ { ( b ) } ) \mathbf { x } _ { \mathrm { C T } } ^ { ( b ) } + m _ { \mathrm { C T } } ^ { ( b ) } ( 1 - z _ { \mathrm { C T } } ^ { ( b ) } ) \mathbf { x } _ { \mathrm { P E T } } ^ { ( b ) } , } \\ & { \widetilde { \mathbf { x } } _ { \mathrm { P E T } } ^ { ( b ) } = ( 1 - m _ { \mathrm { P E T } } ^ { ( b ) } ) \mathbf { x } _ { \mathrm { P E T } } ^ { ( b ) } + m _ { \mathrm { P E T } } ^ { ( b ) } ( 1 - z _ { \mathrm { P E T } } ^ { ( b ) } ) \mathbf { x } _ { \mathrm { C T } } ^ { ( b ) } . } \end{array}\tag{6}
$$

Here, $b \in \{ 1 , \ldots , B \}$ . When $m _ { k } ^ { ( b ) } = 0$ , the original block remains unchanged. When $m _ { K } ^ { ( b ) } = 1$ , the block is either replaced with zeros $( z _ { k } ^ { ( b ) } = 1$ , Lost) or substituted with its spatially corresponding block from the complementary modality $( z _ { k } ^ { ( b ) } = 0$ , Replaced).

Finally, the perturbed blocks are spatially rearranged according to their original positions to reconstruct the 3D perturbed PET and CT patches, $\underset { \mathbf { X } _ { \mathrm { P E T } } } { \sim }$ and $\widetilde { \mathbf { X } } _ { \mathrm { C T } }$ , respectively, as follows:

$$
\widetilde { \mathbf { X } } _ { \mathrm { P E T } } = \mathrm { R e a s s e m b l e } \left( \widetilde { \mathbf { x } } _ { \mathrm { P E T } } ^ { ( 1 ) } , \widetilde { \mathbf { x } } _ { \mathrm { P E T } } ^ { ( 2 ) } , \dots , \widetilde { \mathbf { x } } _ { \mathrm { P E T } } ^ { ( B ) } \right) ,
$$

$$
\widetilde \mathbf { X } _ { \mathrm { C T } } = \mathrm { R e a s s e m b l e } \left( \widetilde \mathbf { x } _ { \mathrm { C T } } ^ { ( 1 ) } , \widetilde \mathbf { x } _ { \mathrm { C T } } ^ { ( 2 ) } , \dots , \widetilde \mathbf { x } _ { \mathrm { C T } } ^ { ( B ) } \right) .\tag{7}
$$

Here, Reassemble(·) denotes the spatial reconstruction operation that restores the original block arrangement, yielding perturbed patches of dimensions $\stackrel { s } { D } \times H \times W$

An overview of our proposed LoRe-MIM is provided in Figure 1.

![](images/4c2befb9b59e17935494d8146c19b550bbb00068378fe4a143ff7620950dbbb9.jpg)  
Figure 1: Overview of the proposed masking approach.

## 2.4.2 Reconstruction Objective

As outlined in Eq. 1, the perturbed 3D PET/CT patches, $\widetilde { \mathbf { X } } _ { \mathrm { P E T } } ^ { ( i ) }$ and $\widetilde { \mathbf { X } } _ { \mathrm { C T } } ^ { ( i ) }$ , are fed into the model $f _ { \theta }$ to obtain the corresponding reconstructed patches, $\widehat { \mathbf { X } } _ { \mathrm { P E T } } ^ { ( i ) }$ and $\widehat { \mathbf { X } } _ { \mathrm { C T } } ^ { ( i ) }$ . Following the reconstruction objective defined in Eq. 2, we optimize the model parameters θ by minimizing the discrepancy between the original and reconstructed 3D PET/CT patches. We adopt the standard least squares $L _ { 2 }$ loss as the reconstruction objective. However, rather than assigning equal importance to all voxels within a 3D patch, we introduce a block-wise weighting strategy that prioritizes cross-modally replaced blocks over lost blocks and unchanged blocks. Specifically, we assign a unit weight to the reconstruction errors of replaced blocks, while downweighting those of lost and unchanged blocks by a factor λ, where $0 \leq \lambda < 1$ . This strategy encourages the model to emphasize cross-modal information during reconstruction while retaining a reduced contribution from the remaining blocks. Accordingly, the block-wise weighted reconstruction objective is formulated as

$$
\begin{array}{c} \theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \frac { 1 } { N B } \sum _ { i = 1 } ^ { N } \sum _ { k \in \{ \mathrm { F E T } , \mathrm { C T } \} } \left[ \lambda \underbrace { \sum _ { b : m _ { k } ^ { ( i , b ) } = 0 } ^ { b } { C _ { k } ^ { ( i , b ) } } } _ { \underbrace { b : m _ { k } ^ { ( i , b ) } = 1 } _ { \mathrm { L a y s t r u m e n t } ; \mathrm { d e t } } } \right. \Bigg .  \\ { \left. + \underbrace { \sum _ { b : m _ { k } ^ { ( i , b ) } = 1 } ^ { b } { C _ { k } ^ { ( i , b ) } } } _ { \underbrace { b : m _ { k } ^ { ( i , b ) } = 1 } _ { \mathrm { R e p t a r e d } } } \right] . } \end{array}\tag{8}
$$

where N and B denote the number of paired 3D PET/CT patches and the number of blocks within each patch, respectively, and $k \in \{ \mathrm { P E T } , \mathrm { C T } \}$ denotes the imaging modality. Here, $m _ { k } ^ { ( i , b ) }$ represents the binary masking indicator, with $m _ { k } ^ { ( i , b ) } = 1$ indicating a selected block, while $z _ { k } ^ { ( i , b ) }$ determines whether the selected block undergoes a Lost $( z _ { k } ^ { ( i , b ) } = 0 )$ or Replaced $( z _ { k } ^ { ( i , b ) } = 1 )$ perturbation. The reconstruction error for each block is defined as

$$
\mathcal { L } _ { k } ^ { ( i , b ) } = \frac { 1 } { \vert \Omega _ { b } \vert } \left. \widehat { \mathbf { x } } _ { k } ^ { ( i , b ) } - \mathbf { x } _ { k } ^ { ( i , b ) } \right. _ { 2 } ^ { 2 } ,\tag{9}
$$

where $\mathbf { x } _ { k } ^ { ( i , b ) }$ and $\widehat { \mathbf { x } } _ { k } ^ { ( i , b ) }$ represent the original and reconstructed blocks, respectively, and $\left| \Omega _ { b } \right|$ denotes the number of voxels within each block.

## 2.5 Evaluation of LoRE-MIM Pretraining

We conduct several experiments to test if our self-supervised pretraining method enables learning robust features for improving whole-body reconstruction, distinguishing cancereous tissue from normal tissue via classification and lesion segmentation tasks, is robust to noise, missing modality, and PET/CT misalignment, and enables label-efficient training. We also test whether our method is generalizable across tracers and multi-institutional data.

## 2.5.1 Whole-body multimodal image reconstruction

We first evaluate the reconstruction capability of our self-supervised 3D patch-based pretraining model on whole-body multimodal PET/CT images. We employ a sliding-window inference strategy with a patch size of $9 6 \times 1 2 8 \times 1 2 8$ voxels to reconstruct the entire volume. Overlapping reconstructed patches are subsequently aggregated using Gaussianweighted averaging to minimize boundary artifacts and ensure spatial continuity in the reconstructed whole-body PET/CT images.

Reconstruction Evaluation Method: The reconstructed whole-body volumes are evaluated using three metrics: Mean Absolute Error (MAE), Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR). MAE quantifies the average absolute voxel-wise difference between the original and reconstructed scans. SSIM measures the structural similarity between the original and reconstructed 3D volumes, while PSNR assesses reconstruction fidelity relative to the maximum signal intensity. Overall, lower MAE and higher SSIM and PSNR values indicate better reconstruction performance.

## 2.5.2 Patch-level classification into cancer vs non-cancer

We evaluate if the learned representations can classify 3D patches into cancer vs non-cancer, and if the classification performance varies based on the amount of cancer present in each patch. We extract features from different stages of the pretrained SwinUNETR model, including the encoder output, the outputs of the five decoder stages, and feed them through a logistic regression classifier. We investigate the following configurations for this classification task

1. Feature Aggregation: We investigate two approaches for feature aggregation: flattening and global average pooling (GAP) to obtain a feature vector for each lesion.

2. Dimensionality Reduction: To reduce the dimensionality of the extracted and/or aggregated feature vectors, we apply Principal Component Analysis (PCA) and kernel principal component analysis (KPCA) with a ra dial basis function (RBF) kernel. We compare reduced feature representations with 128 and 384 components.

Classification Evaluation Method: To evaluate binary classification performance of cancer vs. non-cancer, we construct the receiver operating characteristic (ROC) curve and compute the area under the ROC curve (AUC-ROC).

## 2.5.3 Robustness evaluation with patch-based classification

We evaluate the robustness of the learned representations for patch classification using synthetically perturbed 3D PET/CT patches under three clinically relevant scenarios: (1) noisy images, (2) spatial misalignment between PET and CT modalities due to registration errors, and (3) missing modality information, where either PET or CT is unavailable.

• Robustness to Noisy Images: We independently introduce zero-mean Gaussian noise into PET and CT patches of testing cohort to evaluate the robustness of the learned representations to image corruption. These perturbations provide a controlled approximation of image quality degradation that may arise from variations in image acquisition and reconstruction. Specifically, Gaussian noise is independently added to the PET and CT voxels, with standard deviations set to 0.1 times the corresponding modality’s intensity standard deviation computed from the clean training data. The resulting corrupted intensities are clipped to the intensity range of the corresponding modality in the clean training data.

• Robustness to PET/CT misalignment: Spatial misalignment between PET and CT may arise from patient movement between acquisitions [40], differences in respiratory states [41], and artifacts introduced by CT-based attenuation correction in the presence of respiratory motion [42]. We simulate such spatial misalignment between PET and CT modalities by applying random geometric transformations to the CT channel of each test patch while keeping the PET channel unchanged. Specifically, we independently sample in-plane rotations about the z-axis uniformly from [−5<sup>◦</sup>, 5<sup>◦</sup>] and translations along the x- and y-axes uniformly from [−2, 2] voxels, while maintaining zero translation along the z-axis. The transformed CT patches are resampled using trilinear interpolation with zero padding. These perturbations simulate registration errors between PET and CT modalities while preserving both modalities.

• Robustness to Missing Modality: We simulate scenarios in which either the PET or CT modality is unavailable to evaluate the robustness of the learned representations to missing modality information. Specifically, we consider two scenarios: (1) Missing-PET, where all PET voxel intensities are set to zero while the CT channel remains unchanged, and (2) Missing-CT, where all CT voxel intensities are set to zero while the PET channel remains unchanged.

Evaluation Method: Similar to evaluation with clean dataset, we evaluate ROC-AUC for evaluating robustness of the binary classification task.

## 2.5.4 Whole-body Lesion Segmentation

We evaluate the effectiveness of the learned representations for whole-body lesion segmentation by finetuning the pretrained model using a labeled dataset $\mathcal { D } _ { \mathcal { L } } = \{ ( X _ { \mathrm { P E T } } ^ { ( i ) } , X _ { \mathrm { C T } } ^ { ( i ) } , Y ^ { ( i ) } ) \} _ { i = 1 } ^ { N _ { \mathcal { L } } }$ , comprising $N _ { \mathcal { L } }$ paired 3D PET/CT scans and their corresponding voxel-wise binary lesion segmentation masks $Y ^ { ( i ) }$ . We denote the parameters of the pretrained model by θ, which comprises three mutually disjoint subsets: the backbone parameters $\theta _ { \mathrm { b a c k b o n e } } ,$ encoder parameters $\theta _ { \mathrm { e n c } } .$ , and decoder parameters $\theta _ { \mathrm { d e c } }$ , such that $\theta = \theta _ { \mathrm { b a c k b o n e } } \cup \theta _ { \mathrm { e n c } } \cup \theta _ { \mathrm { d e c } }$ . During finetuning, we partition θ into trainable parameters $\theta _ { \mathrm { F T } }$ , which are initialized from their corresponding pretrained weights and subsequently optimized using the labeled dataset, and frozen parameters $\theta _ { \mathrm { f r o z e n } } ,$ , which remain fixed throughout finetuning. Following Rokuss et al. [8], we optimize the trainable parameters using an equally weighted combination of Dice loss and cross-entropy loss to encourage accurate lesion delineation and voxel-wise classification.

To investigate the contribution of self-supervised pretraining and the effects of adapting different model components to downstream lesion segmentation, we compare training from scratch against three finetuning strategies:

1. Training from Scratch (Scratch): We randomly initialize and optimize all model parameters θ using the labeled dataset, without leveraging pretrained weights.

2. Full Finetuning (Full FT): We initialize all model parameters from the pretrained weights and optimize the entire model, with $\theta _ { \mathrm { F T } } = \theta$ and $\theta _ { \mathrm { f r o z e n } } = \mathcal { O }$

3. Encoder and Decoder Finetuning (Enc. + Dec. FT): We optimize the pretrained encoder and decoder parameters while keeping the backbone frozen, with $\theta _ { \mathrm { F T } } = \theta _ { \mathrm { e n c } } \cup \theta _ { \mathrm { d e c } }$ and $\theta _ { \mathrm { f r o z e n } } = \theta _ { \mathrm { b a c k b o n e } }$

4. Decoder Finetuning (Dec. FT): We optimize only the pretrained decoder parameters while keeping both the backbone and encoder frozen, with $\theta _ { \mathrm { F T } } = \theta _ { \mathrm { d e c } }$ and $\theta _ { \mathrm { f r o z e n } } = \theta _ { \mathrm { b a c k b o n e } } \cup \theta _ { \mathrm { e n c } }$

Lesion Segmentation Evaluation Method: We adopt voxel-level, lesion-level, and anatomical region-level evaluation metrics as below:

1. Voxel-level Evaluation: Following Rokuss et al. [8], we use the Dice similarity coefficient, false-positive volume (FPVol), and false-negative volume (FNVol) as voxel-level evaluation metrics. Dice quantifies the spatial overlap between the predicted and ground truth segmentations. FPVol measures the total volume of tissue incorrectly predicted as lesions, whereas FNVol measures the total volume of ground truth lesion tissue missed by the model. We report these metrics only for diseased cases.

2. Lesion-level Evaluation: We identify individual lesions in the predicted and ground truth segmentation masks using morphological processing and connected component analysis. We then aggregate lesion-level counts across all test patients and compute sensitivity, defined as the proportion of ground truth lesions detected, and positive predictive value (PPV), defined as the proportion of predicted lesions that match ground truth lesions.

3. Anatomical Region-level Evaluation: Treatment decisions are often influenced by the presence of metastases in specific anatomical sites, such as the liver or bones. We therefore evaluate segmentation performance in four anatomical regions: the liver, lungs, bones, and high-risk bones. We use TotalSegmentator [43] to obtain anatomical masks and compute region-level Dice, FPVol, and FNVol using the voxel-level definitions restricted to each region. We also compute region-level sensitivity, defined as the proportion of ground truth lesion-containing regions in which at least one lesion is correctly detected, and specificity, defined as the proportion of ground truth lesion-free regions in which no lesion is predicted.

## 2.5.5 Label-Efficient Segmentation Training

Obtaining voxel-wise annotations for whole-body lesion segmentation is a time-consuming and resourceintensive process that requires substantial clinical expertise. As discussed in Sect. 1, the growing volume of whole-body PET/CT examinations further widens the gap between imaging demand and the availability of expert radiologists to generate or verify such annotations. Therefore, we evaluate the effectiveness of self-supervised pretraining under label-efficient finetuning settings by utilizing varying proportions (1%, 5%, 10%, 20%, 50%, and 100%) of the labeled training data for whole-body lesion segmentation.

Evaluation Method: We evaluate segmentation performance using Dice and report the results across all evaluated percentages of labeled training data.

## 2.6 Implementation Details

We adopt the 3D SwinUNETR [36] architecture for our pretraining model using its MONAI [44] implementation with a feature size of 48. The Swin Transformer encoder comprises four hierarchical stages with depths of (2, 2, 6, 2) and attention heads of (3, 6, 12, 24), respectively. The network operates on 3D volumetric PET/CT inputs, with the output channels configured to reconstruct both PET and CT images during pretraining and to generate binary foreground and background segmentation logits during downstream fine-tuning. implementation.

1. SwinUNETR pretraining: We set the weighting parameter λ = 0.2 in Eqn. 8 to assign a lower contribution to unperturbed or zero-masked blocks than to cross-modally perturbed blocks in the pretraining loss. We train the complete network for 200 epochs using AdamW with a batch size of 3, a maximum learning rate of $1 0 ^ { - 4 }$ , and weight decay of $1 0 ^ { - 2 }$ . Training uses automatic mixed precision and gradient checkpointing. The learning rate is linearly increased from $\mathrm { \overline { { 1 0 } } ^ { - 6 } ~ t o ~ 1 0 ^ { - 4 } }$ during the first five epochs and then decayed to

$1 0 ^ { - 6 }$ using cosine annealing over the remaining epochs. Validation is performed after every epoch, and the checkpoint with the lowest total validation reconstruction loss is retained. for downstream evaluation.

2. Classification models: We train logistic regression classifiers with our extracted features with balanced class weights, and a maximum of 2,000 optimization iterations. The inverse regularization strength is selected from $C \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 , \dot { 1 0 } ^ { 2 } , 1 0 ^ { 3 } \}$ by maximizing ROC AUC on the clean validation split. Each candidate classifier is fitted only on training features.

3. Segmentation models: We employ the nnU-Net framework adapted by Rokuss et al. [8] to finetune our LoRe-MIM-pretrained SwinUNETR. We train the model for 600 epochs using stochastic gradient descent (SGD) with Nesterov momentum and a polynomial learning rate decay schedule, with an initial learning rate of 0.001 and a weight decay of $3 \times 1 0 ^ { - 5 }$ . We optimize the model using an equally weighted combination of Dice and cross-entropy losses and select the checkpoint with the lowest validation loss. For each fine-tuning variant described in Sect. 2.5.4, we initialize the frozen network components with their corresponding pretrained weights from the pretraining phase, while the remaining trainable parameters are randomly initialized and optimized during fine-tuning.

## 3 Ablation Studies for Pretraining Approaches

## 3.1 Masking Strategies

We conduct an ablation study examining three special cases of the proposed Lost-or-Replaced (LoRe) masking framework to investigate the contributions of different masking strategies to self-supervised representation learning. A Lost perturbation sets the selected block to zero, whereas a Replaced perturbation substitutes the spatially corresponding block from the other modality. Based on this formulation, we consider three masking variants:

1. Lost-One MIM (LO-MIM): Following MUST-PET [45], we apply zero masking to one modality while leaving the other unchanged. For PET masking, we sample $m _ { \mathrm { P E T } } ^ { ( b ) } \sim$ Bernoulli(0.5) and set $z _ { \mathrm { P E T } } ^ { ( b ) } = 1$ and $m _ { \mathrm { C T } } ^ { ( b ) } = 0$ . Conversely, for CT masking, we sample $m _ { \mathrm { C T } } ^ { ( b ) } \sim$ Bernoulli(0.5) and set $z _ { \mathrm { C T } } ^ { ( b ) } = 1$ and $m _ { \mathrm { P E T } } ^ { ( b ) } = 0$ This strategy evaluates representation learning when only one modality is perturbed.

2. Lost-Both MIM (LB-MIM): We apply the same zero-masking pattern to both modalities by sampling a shared mask $m ^ { ( b ) } \sim$ Bernoulli(0.5) and setting $m _ { \mathrm { P E T } } ^ { ( b ) } = m _ { \mathrm { C T } } ^ { ( \bar { b } ) } = m ^ { ( b ) }$ and $z _ { \mathrm { P E T } } ^ { ( b ) } = z _ { \mathrm { C T } } ^ { ( b ) } = 1$ . Thus, spatially corresponding PET and CT blocks are simultaneously set to zero, removing direct input information from both modalities at the selected locations.

3. Replaced-MIM (Re-MIM): We apply only cross-modal replacement by setting $z _ { \mathrm { P E T } } ^ { ( b ) } = z _ { \mathrm { C T } } ^ { ( b ) } = 0$ and independently sampling $m _ { \mathrm { P E T } } ^ { ( b ) } , m _ { \mathrm { C T } } ^ { ( b ) } \sim$ Bernoulli(0.5). Each selected PET block is replaced with the corresponding original CT block, and each selected CT block is replaced with the corresponding original PET block. This strategy isolates the contribution of cross-modal replacement without introducing zero masking.

## 3.2 Augmentation

We investigate the impact of data augmentation during self-supervised pretraining on reconstruction and patch classification (cancer vs. non-cancer) performance. We adopt the whole-body PET/CT-specific augmentation strategies employed by Rokuss et al. [8], incorporating both spatial and intensity-based transformations. Spatial augmentations include random rotations, scaling, and mirroring, while intensity-based augmentations include Gaussian noise, Gaussian blurring, brightness scaling, contrast adjustment, and gamma correction. Additionally, we simulate reduced image resolution and spatial misalignment between PET and CT modalities to account for variations in image quality and intermodal registration.

## 3.3 Variable Masking

We investigate the effect of masking variable portions of each modality in both Re-MIM and LoRe-MIM pretraining. For each sampled training patch of $9 6 \times 1 2 8 \times 1 2 8$ voxels, the CT and PET masking ratios are drawn independently with equal probability. Randomly selected, non-overlapping blocks of $1 2 \times 1 6 \times 1 6$ voxels are masked according to the sampled ratio for each modality, with new ratios and mask locations sampled whenever a training patch was drawn. Both the ablation masking strategy (Re-MIM and LoRe-MIM) retains its original replacement strategy. We evaluated variable masking alone and in combination with augmentation. Validation used no augmentation and a fixed masking ratio of 0.5 for each modality.

## 4 Baseline Models For Comparison

We compare our pretraining approach with reconstruction performance as well as downstream tasks such as segmentation and classification.

## 4.1 Reconstruction

We compare our proposed LoRe-MIM with an off-the-shelf whole-body self-supervised model trained with FDG PET/CT, released by Liu et al. [35]. We denote this baseline as FDG-only MAE.

## 4.2 Segmentation Models

We consider two baseline models for comparison with our finetuned segmentation models: (1) SwinUNETR trained from scratch and (2) the whole-body PET/CT-based nnU-Net architecture proposed by [8], also trained from scratch. For fair comparison, we utilize the same preprocessed samples, same training framework released by [8] with similar augmenttaion and same number of training epochs.

## 4.3 Classification performance with pretrained features

We compare our pretrained representations with those extracted from DINOv3 [46] and ResNet50-3D [47]. For both baselines, PET and CT patches are independently normalized using their respective means and standard deviations (with a minimum standard deviation of $1 0 ^ { - 6 } )$ and concatenated side by side. For ResNet50-3D, global average pooling of the final feature map produces a 2048-dimensional feature vector. For DINOv3, each axial slice is resized to 224 × 224, replicated across three channels, and processed independently. The resulting slice-level features are averaged to obtain a single feature vector per patch.

## 5 Experimental Results

## 5.1 Validation Set Performance for ablation studies

We report validation set performance for the ablation studies to select the best pretraining strategy.

## 5.1.1 Different masking strategies

We assess reconstruction and patch-based classification performance on the validation set for the ablation experiments. For reconstruction performance, we consider the validation set of AutoPET-III $( N = 2 4 7 )$ and the unlabeled DHMC $( N = 1 4 9 )$ dataset. For classification performance, we consider the validation set of AutoPET-III, with patch-based labels derived from voxel-level annotations. Our proposed LoRe-MIM consistently outperforms the three alternative masking strategies, achieving the lowest MAE and the highest SSIM and PSNR for the reconstruction task, and the highest AUC for patch classification task, demonstrating the effectiveness of its learned representations across both tasks 2.

Table 2: Ablation study of masking strategies under fixed masking. Reconstruction performance on AutoPET $( n =$ 247) and DHMC $( n = 1 4 9 )$ validation samples, averaged equally across CT and PET and across all 396 cases, is evaluated using MAE (↓), SSIM (↑), and PSNR in dB $( \bar { \uparrow } )$ . Classification performance on clean test data is evaluated using AUC (↑). Bold indicates the best result for each metric.
<table><tr><td rowspan="2">Masking Approach</td><td colspan="3">Reconstruction</td><td rowspan="2">Classification</td></tr><tr><td>MAE↓</td><td>SSIM↑</td><td>PSNR ↑</td></tr><tr><td>FDG-Only MAE [35]</td><td>0.2797</td><td>0.7288</td><td>32.56</td><td>AUC↑ 0.6195</td></tr><tr><td>LO-MIM</td><td>0.4026</td><td>0.6473</td><td>30.01</td><td>0.6885</td></tr><tr><td>LB-MIM</td><td>0.3831</td><td>0.6835</td><td>30.14</td><td>0.6894</td></tr><tr><td>Re-MIM</td><td>0.2803</td><td>0.7586</td><td>32.73</td><td>0.7010</td></tr><tr><td>LoRE-MIM</td><td>0.2675</td><td>0.7762</td><td>33.07</td><td>0.7284</td></tr></table>

## 5.1.2 Augmentation and Variable Masking

Results from both reconstruction (MAE, SSIM, and PSNR) and patch-based lesion classification (AUC) indicate that neither augmentation nor variable masking consistently improves representation quality (Table 3). Therefore, in subse quent experiments, we adopt a fixed masking strategy with a Bernoulli probability of 0.5, without data augmentation.

Table 3: Ablation study of augmentation and variable masking for CM and LoRE-MIM on AutoPET-III $( n = 3 2 1 )$ and DHMC $( n ~ = ~ 1 0 0 )$ test sets. Reconstruction metrics are evaluated over masked regions after whole-volume reconstruction, averaged equally across CT and PET and across all 421 cases. Masked blocks are replaced with Gaussian samples of standard deviation 0.1. PSNR is in dB. Classification AUC values are retained from the original experiments. Bold indicates the best result for each metric.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Augmentation</td><td rowspan="2">Variable Masking</td><td colspan="3">Reconstruction</td><td rowspan="2">Classification AUC↑</td></tr><tr><td>MAE↓</td><td>SSIM↑</td><td>PSNR ↑</td></tr><tr><td rowspan="4">Re-MIM</td><td>×</td><td>X</td><td>0.3913</td><td>0.6784</td><td>29.86</td><td>0.6855</td></tr><tr><td>√</td><td>×</td><td>0.4131</td><td>0.6513</td><td>29.70</td><td>0.5783</td></tr><tr><td>×</td><td>√</td><td>0.4258</td><td>0.6568</td><td>28.82</td><td>0.5695</td></tr><tr><td>V</td><td>V</td><td>0.4989</td><td>0.5738</td><td>28.35</td><td>0.5525</td></tr><tr><td rowspan="4">LoRE-MIM</td><td>X</td><td>X</td><td>0.2718</td><td>0.7701</td><td>32.85</td><td>0.7091</td></tr><tr><td>√</td><td>×</td><td>0.2727</td><td>0.7847</td><td>32.46</td><td>0.6042</td></tr><tr><td>X</td><td>√</td><td>0.2814</td><td>0.7559</td><td>32.64</td><td>0.6142</td></tr><tr><td>V</td><td>√</td><td>0.2870</td><td>0.7721</td><td>32.42</td><td>0.5970</td></tr></table>

## 5.2 Test Set Results for Downstream Tasks

## 5.2.1 Reconstruction and classification performance

We first evaluate four different masking strategies in terms of reconstruction and classification performance on the AutoPET-III and DHMC test sets (Table 4). Test set size for reconstruction evaluation is 421 scans (AutoPet-III and DHMC), while for classification is 321 scans (AutoPet-III). Consistent with our observations on the validation set (Table 2), our proposed LoRe-MIM achieves the lowest MAE and highest SSIM, PSNR, and AUC among all four masking strategies, further demonstrating its effectiveness in learning transferable representations for both reconstruc tion and classification tasks.

Table 4: Reconstruction and classification performance. Reconstruction is evaluated on the AutoPET-III $( n = 3 2 1 )$ and DHMC (n = 100) test sets using masked-region MAE (↓), SSIM (↑), and PSNR in dB (↑), computed after wholevolume reconstruction and averaged equally across CT and PET modalities and across all 421 test cases. Classification performance is evaluated using AUC (↑), with values retained from the original classification experiments.
<table><tr><td rowspan="2">Masking Approach</td><td colspan="3">Reconstruction</td><td rowspan="2">Classification AUC↑</td></tr><tr><td>MAE↓</td><td>SSIM ↑</td><td>PSNR ↑</td></tr><tr><td>FDG-Only</td><td>0.2812</td><td>0.7256</td><td>32.45</td><td>0.6782</td></tr><tr><td>LO-MIM</td><td>0.4080</td><td>0.6432</td><td>29.74</td><td>0.6695</td></tr><tr><td>LB-MIM</td><td>0.3913</td><td>0.6784</td><td>29.86</td><td>0.6626</td></tr><tr><td>Re-MIM</td><td>0.2857</td><td>0.7510</td><td>32.51</td><td>0.6855</td></tr><tr><td>LoRE-MIM</td><td>0.2718</td><td>0.7701</td><td>32.85</td><td>0.7091</td></tr></table>

## 5.2.2 Patch-level classification into cancer vs non-cancer

We first evaluate patch-based classification performance on the clean AutoPET-III test set using features extracted from the SwinUNETR encoder. For all comparisons, we employ logistic regression (LR) for classification using the extracted features, with dimensionality reduction applied where specified. As shown in Table 5, encoder features aggregated using global average pooling (GAP) achieve a higher AUC with PCA using 384 components (0.7180) than with 128 components (0.6974). Furthermore, GAP generally outperforms feature flattening, highlighting the benefits of spatial feature aggregation. We also note that linear dimensionality reduction with PCA outperforms non-linear

KPCA for 128 and 384 feature dimensions, suggesting that learned representations have good linear separability in cancer vs non-cancer patterns. We then investigate features extracted from different decoder stages and observe that decoder representations consistently outperform encoder representations. In particular, Dec achieves the highest AUC (0.7790), surpassing both DINOv3 (0.7589) and ResNet50-3D (0.7120). These results highlight the effectiveness of intermediate decoder representations learned through LoRe-MIM pretraining for downstream patch-based classification.  
Table 5: Evaluation of patch-based classification using features extracted from different model stages and baseline architectures under clean input conditions. We compare feature aggregation and dimensionality reduction strategies using logistic regression (LR) for classification. Performance is evaluated using AUC (↑). Bold and underline indicate the best overall and best encoder performance, respectively.
<table><tr><td>Feature Extraction Stage / Model</td><td>Feature Type</td><td>Reduction</td><td>AUC↑</td></tr><tr><td rowspan="2">Enc</td><td>Flatten</td><td>KPCA (128) PCA (128)</td><td>0.6898 0.6565 0.6808</td></tr><tr><td>GAP</td><td>KPCA (128) PCA (128) KPCA (384) PCA (384)</td><td>0.6974 0.7079 0.7180</td></tr><tr><td>Dec1 Dec2 Dec3 Dec4</td><td>GAP GAP GAP GAP</td><td></td><td>0.7339 0.7370 0.7656 0.7790</td></tr><tr><td>Dec5 ResNet50-3D DINOv3</td><td>GAP GAP GAP</td><td>PCA (384)</td><td>0.7680 0.7120 0.7589</td></tr></table>

## 5.2.3 Robustness evaluation with patch-based classification

Table 6 presents the robustness evaluation of patch-based classification using features extracted from the fourth de coder stage (Dec ) of our LoRe-MIM pretrained SwinUNETR under perturbed 3D PET/CT inputs. Our pretrained decoder features consistently outperform the ResNet50-3D and DINOv3 baselines across all four perturbation settings. Under Gaussian noise, our method achieves an AUC of 0.7584, compared with 0.7555 for DINOv3 and 0.7268 for ResNet50-3D. Similarly, under simulated PET/CT misalignment, our method achieves a higher AUC (0.7648) than DINOv3 (0.7286) and ResNet50-3D (0.7332). More notably, all methods exhibit substantial performance degradation when either imaging modality is missing, with missing PET resulting in considerably lower AUC than missing CT. Nevertheless, our method maintains an AUC of 0.6975 under missing CT and 0.5995 under missing PET, outperforming DINOv3 (0.6500 and 0.5344) and ResNet50-3D (0.6217 and 0.5348), respectively. The largest performance gap between our method and DINOv3 occurs when PET is missing (0.0651 AUC), suggesting that our learned representations are less sensitive to the absence of PET information. These findings support the potential of LoRe-MIM to capture complementary cross-modal representations and improve classification robustness under incomplete or corrupted PET/CT inputs.

Table 6: Robustness evaluation of patch-based classification under four PET/CT input perturbations: Gaussian noise, PET/CT misalignment, missing CT, and missing PET. Performance is evaluated using AUC (↑). Bold indicates the best result for each perturbation.
<table><tr><td>Category</td><td>Model</td><td>Gaussian Noise</td><td>Misaligned</td><td>CT Missing</td><td>PET Missing</td></tr><tr><td rowspan="2">Baseline</td><td>ResNet50-3D</td><td>0.7268</td><td>0.7332</td><td>0.6217</td><td>0.5348</td></tr><tr><td>DINOv3</td><td>0.7555</td><td>0.7286</td><td>0.6500</td><td>0.5344</td></tr><tr><td>Ours</td><td>LoRe-MIM (Dec4)</td><td>0.7584</td><td>0.7648</td><td>0.6975</td><td>0.5995</td></tr></table>

## 5.2.4 Whole-body lesion segmentation with test-set

We use 321 cases from AutoPET-III and 200 cases from Deep-PSMA as independent test sets for whole-body lesion segmentation. These cases are strictly excluded from both the pretraining and finetuning stages, ensuring no overlap with the training or validation cohorts used in either stage.

Voxel and Lesion-level evaluation: We compare our proposed LoRe-MIM pretrained SwinUNETR against SwinUNETR and nnU-Net trained from scratch, considering three finetuning configurations: Full FT, Dec. FT, and Enc. + Dec. FT, as described in Sect. 2.5.4. Among these configurations, Enc. + Dec. FT achieves the highest Dice scores (0.581 and 0.613) and lesion sensitivities (0.776 and 0.759) on AutoPET-III and Deep-PSMA, respectively. Compared with SwinUNETR trained from scratch, these correspond to relative Dice improvements of 10.5% and 12.1%, respectively (Table 7). However, the optimal finetuning configuration varies across evaluation metrics. On AutoPET-III, Full FT achieves the lowest FPVol (15.594 cc), while Dec. FT achieves the lowest FNVol (7.405 cc). On Deep-PSMA, Dec. FT achieves the lowest FPVol (48.432 cc) and FNVol (6.588 cc). Despite these improvements, nnU-Net trained from scratch achieves the highest lesion-level PPV on both AutoPET-III (0.733) and Deep-PSMA (0.768), indicating a trade-off between lesion detection sensitivity and precision. We further present tracer-stratified Dice, FPVol, and FNVol in Figures 2 and 3. For PSMA scans in AutoPET-III, Enc. + Dec. FT achieves the lowest FPVol (17.68 cc), whereas Dec. FT achieves the lowest FNVol (7.15 cc). In contrast, for PSMA scans in Deep-PSMA, Dec. FT achieves the lowest FPVol (57.73 cc), while Full FT achieves the lowest FNVol (4.90 cc). Overall, these results demonstrate that LoRe-MIM pretraining improves segmentation overlap across both datasets, while different finetuning configurations exhibit distinct trade-offs between false-positive and false-negative predictions.

Table 7: Lesion segmentation performance on the AutoPET-III (n = 321) and Deep-PSMA (n = 200) test sets using different baseline architectures and training strategies. Three finetuning variants of LoRe-MIM (Ours) are evaluated: Full FT, Dec. FT, and Enc. + Dec. FT. FPVol and FNVol are reported in cc. ↑: higher is better; ↓: lower is better.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Baseline</td><td rowspan="2">Training Strategy</td><td rowspan="2">Variant</td><td rowspan="2">Dice↑</td><td rowspan="2">FPVol↓</td><td rowspan="2">FNVol↓</td><td rowspan="2">Lesion Sens.↑</td><td rowspan="2">PPV↑</td></tr><tr><td></td></tr><tr><td rowspan="5">AutoPET-III</td><td>nnU-Net</td><td>Scratch Scratch</td><td></td><td>0.566</td><td>17.283</td><td>9.472</td><td>0.773</td><td>0.733</td></tr><tr><td rowspan="4">SwinUNETR</td><td></td><td>Full FT</td><td>0.526</td><td>27.147</td><td>7.586</td><td>0.773</td><td>0.546</td></tr><tr><td>LoRe-MIM</td><td></td><td>0.577</td><td>15.594</td><td>9.832</td><td>0.775</td><td>0.664</td></tr><tr><td></td><td>Dec. FT Enc. + Dec. FT</td><td>0.569 0.581</td><td>19.207</td><td>7.405</td><td>0.764</td><td>0.713</td></tr><tr><td></td><td></td><td></td><td>15.662</td><td>8.362</td><td>0.776</td><td>0.675</td></tr><tr><td rowspan="5">Deep-PSMA</td><td>nnU-Net</td><td>Scratch</td><td>一</td><td>0.561</td><td>73.030</td><td>7.933</td><td>0.757</td><td>0.768</td></tr><tr><td rowspan="4">SwinUNETR</td><td>Scratch</td><td></td><td>0.547</td><td>74.315</td><td>9.785</td><td>0.711</td><td>0.732</td></tr><tr><td></td><td>Full FT</td><td>0.572</td><td>76.037</td><td>8.964</td><td>0.731</td><td>0.766</td></tr><tr><td>LoRe-MIM</td><td>Dec. FT</td><td>0.610</td><td>48.432</td><td>6.588</td><td>0.743</td><td>0.765</td></tr><tr><td></td><td>Enc. + Dec. FT</td><td>0.613</td><td>48.456</td><td>7.023</td><td>0.759</td><td>0.741</td></tr></table>

Anatomical region-level evaluation: Figures 4 and 5 present anatomical-region-level segmentation performance across four regions (liver, lung, bone, and HR-bone) on AutoPET-III and Deep-PSMA, respectively, evaluated using Dice, FPVol, FNVol, lesion sensitivity, and specificity, computed for each region independently. Across both datasets, the best-performing LoRe-MIM finetuning variants consistently improve Dice over SwinUNETR trained from scratch in all four anatomical regions, with the largest improvements observed in the liver. Specifically, liver Dice increases from 0.36 to 0.48 on AutoPET-III and from 0.25 to 0.42 on Deep-PSMA, corresponding to relative improvements of 33.3% and 68.0%, respectively. Improvements are also observed in the lung (0.44 to 0.49 and 0.42 to 0.48), bone (0.53 to 0.57 and 0.64 to 0.67), and HR-bone (0.55 to 0.60 and 0.66 to 0.68) on AutoPET-III and Deep-PSMA, respectively. In addition to improved segmentation overlap, LoRe-MIM enhances lesion-level specificity in bone and HR-bone regions. On AutoPET-III, bone specificity increases from 0.72 to 0.85, while on Deep-PSMA, HR-bone specificity increases from 0.67 to 0.87 compared with SwinUNETR trained from scratch. For liver lesions, all three LoRe-MIM finetuning variants achieve higher sensitivity (0.85–0.92) than SwinUNETR trained from scratch (0.73) on Deep-PSMA, whereas liver sensitivity remains consistently high (1.00) across all configurations on AutoPET III. Interestingly, improvements in liver Dice exhibit contrasting FPVol–FNVol trade-offs across the two datasets. On AutoPET-III, Enc. + Dec. FT reduces liver FPVol from 3.74 to 2.03 cc compared with SwinUNETR Scratch, accompanied by a slight increase in FNVol from 3.43 to 3.57 cc. Conversely, on Deep-PSMA, Dec. FT reduces liver FNVol from 4.53 to 2.74 cc while increasing FPVol from 7.77 to 21.31 cc. These findings suggest that LoRe-MIM improves anatomical-region-level segmentation across both datasets, although the relative contributions of reduced false-positive and false-negative volumes vary with the dataset and finetuning configuration.

![](images/00fe8152e962446de781f1e83171c1eafd34c7a9f34ed828bfd55bcbb0287beb.jpg)

![](images/d00235aa85d15488f9310bbb495f90a4d64ac06e447b611f97e07e3bf43c40e9.jpg)  
(a) FDG cases (n = 199) among a total of 321 cases.

![](images/2522f9903360ec28492ddd0bd12d2ede8a0b97032e2a39472a7277e870fd2c5a.jpg)

![](images/6a63396a14221ef0e27976cca8edd136ca17c3d703e978b4badf566c0a5ef59e.jpg)

![](images/1234ad6ff8bd301e9cc73f28aeec384948e7343fcacc04f60efeb967637c32e7.jpg)  
(b) PSMA cases (n = 122) among a total of 321 cases.

![](images/7ee8e3180404356cbd859ffff3d08e935461723c3172d2b1c14dd6d071dce01d.jpg)

Figure 2: Tracer-specific voxel-level evaluation on the AutoPET-III test set, comparing nnU-Net and SwinUNETR trained from scratch with SwinUNETR pretrained using LoRe-MIM under different finetuning configurations.  
![](images/b5703af66f25fe4af513c645b0080165210118ff557b860f25fe961db8fc84ec.jpg)

![](images/0f68cd09808ff1ee6fcfffa76fc84f9290d2bd69496842d7aae3bd0982ce617e.jpg)

![](images/3bd9d6571a17fe83d8332cbe466e0b1d784ad0f981c6db65779fb328341364d6.jpg)

![](images/37dc0b2ff3edec4952344b7c4182624d96362e3ce94eac83cd6f08df157cea48.jpg)  
(a) FDG cases (n = 100) among a total of 200 cases.

![](images/0a0443c1a334149cf2f949db51c89750aff558f31e01e6d6814b8098cbc08078.jpg)

![](images/d0c75e380c936c9da987f391352e6690cd25737d6df9a10421ad40ccded4cc02.jpg)

![](images/38256cd619b8c911434a8acb8705313cd23ea68dba9cd3f60b80287163c41a8c.jpg)  
(b) PSMA cases (n = 100) among a total of 200 cases.  
Figure 3: Tracer-specific voxel-level evaluation on the Deep-PSMA test set, comparing nnU-Net and SwinUNETR trained from scratch with SwinUNETR pretrained using LoRe-MIM under different finetuning configurations.

![](images/fdecf36000273dbd398752b4b393b28b1917672f050512d2bb5bfb3cba6d38d9.jpg)

![](images/3a5e9b091f1ffda3b80b57b04ae0b34a662582b79ab3f9859ece81555a3934d3.jpg)

![](images/17d3ce43ef5f17d27e89f170a23e8e1ad5efeef44c34f60bef5ceaa0fefe6305.jpg)

![](images/fcf57e82ebf38d632b3ad11671e99896d1ca01bb04e31f4fb9f970515667ceb6.jpg)

![](images/a37d74b3022606df0c47f4ea94a069b1bee310a50ff06f8bd3bb5c9cb265e357.jpg)

![](images/fb7d147974f884973aaab6fc3772afe26be7e024d0b905a85f23eed43dd2bf45.jpg)  
Figure 4: Anatomical region-level comparison of lesion segmentation performance on AutoPET-III, comparing nnU-Net and SwinUNETR trained from scratch with three finetuning variants of our LoRe-MIM pretrained SwinUNETR.

## 5.2.5 Label-efficiency curve

In Figure 6, we observe that under extremely limited labeled data (1% and 5%), our finetuning approaches achieve substantially higher Dice scores compared to their counterparts trained from scratch, with diminishing gains as the proportion of labeled data increases. These findings further support our motivation for developing a self-supervised pretraining approach that learns transferable representations for downstream tasks, especially under limited labeled data availability.

## 5.2.6 Qualitative analysis

We present two qualitative comparisons in Figures 7 and 8. Figure 7 illustrates representative examples of improved lesion recovery, false-positive elimination, and a failure case. Specifically, we select two cases from AutoPET-III (one liver lesion and one false-positive prediction) and two cases from Deep-PSMA (one liver lesion and one high-risk bone lesion). These examples demonstrate the effectiveness of our proposed LoRe-MIM finetuning variants in recovering lesions missed by models trained from scratch and suppressing false-positive predictions, while also highlighting their limitations in challenging cases. Furthermore, Figure 8 presents a whole-body coronal visualization demonstrating that our proposed LoRe-MIM eliminates a false-positive prediction observed in both scratch-trained baselines (nnUNet and SwinUNETR) while preserving the true-positive lesion.

## 6 Discussion

In this work, we aim to bridge the research gap arising from label scarcity in developing deep learning models for whole-body lesion segmentation and patch-based classification using PET/CT scans. To address this challenge, we seek to leverage well-established self-supervised pretraining schemes to learn rich and transferable representations of whole-body PET/CT scans. Existing pretraining approaches in the whole-body PET/CT domain often rely on relatively small cohorts, incorporate limited cross-modal interactions, and primarily evaluate the effectiveness of pretraining through downstream segmentation tasks only. In this work, we aim to address these limitations through a large-scale, heterogeneous pretraining dataset, a novel cross-modal masked image modeling strategy, and an extensive evaluation across multiple downstream tasks.

![](images/fe05e2739566c8fea7552e2325c56fcf2ee056adbf5d30b7932a24d262a17d91.jpg)

![](images/e488c670a3698d1355f21ede61d611d9fe62c4857068ceffe87f80525634c1f3.jpg)

![](images/e3a4c6563cdc0e29b55b01f55fdcce57ce2a795c44a9497330c493287bab58ee.jpg)

![](images/3534114c7ac9ab6105ed53e0cdf0ad0b4f3d77921c1a0a794191a71f6f129be9.jpg)

![](images/a78335d5cfd62e843c4edb9eb1734c72ca042a3fbc9334e58625bb36c12a3dcc.jpg)

![](images/b001a381eadbc6857123ca09511c632de27df8bb2bcd4d9bf5d8ceafeb124d90.jpg)  
Figure 5: Anatomical region-level comparison of lesion segmentation performance on Deep PSMA, comparing nnU Net and SwinUNETR trained from scratch with three finetuning variants of our LoRe-MIM pretrained SwinUNETR.

(a) AutoPET II  
![](images/95f34eadd39452b1ad894ec7864bceabe984b3a47f22a080054a47a19271b90c.jpg)

(b) Deep PSMA  
![](images/91007ce5a105226985f6491e5fd505ea94c58dfa73e6b317f395a7e3d60f9f46.jpg)  
· nnU-Net (Scratch) — SwinUNETR(Lore-MIM & Dec. FT)  SwinUNETR(Lore-MIM & Enc. + Dec. FT) x - SwinUNETR (Scratch) —— SwinUNETR(Lore-MIM & FulI FT)  
Figure 6: Label-efficiency curves for whole-body lesion segmentation on the AutoPET-III and Deep-PSMA datasets. Segmentation performance is evaluated using Dice scores across varying proportions of labeled training data (1%– 100%). All three finetuning strategies outperform SwinUNETR and nnU-Net trained from scratch, with more pronounced improvements under limited labeled data (1%–10%) and diminishing performance gains as the proportion of labeled data increases.

![](images/32a68a96380ac02f74437727be5f3201db62ec08830c9541971d21ff64cee79f.jpg)  
Figure 7: Qualitative comparison of lesion segmentation performance across four representative cases (Rows 1–4). Columns 1 and 2 show the CT and PET images with ground-truth lesion contours (green), respectively. Columns 3 and 4 present predictions from nnU-Net and SwinUNETR trained from scratch, respectively, while Columns 5–7 show predictions from three SwinUNETR finetuning strategies initialized with our pretrained LoRe-MIM model. Row 1 illustrates successful liver lesion recovery in an AutoPET-III case, where both scratch-trained models fail to detect the lesion, whereas all three LoRe-MIM variants successfully identify it. Row 2 demonstrates false-positive elimination in an AutoPET-III case, where both scratch-trained models incorrectly predict a lesion, while all three LoRe-MIM variants correctly produce no lesion segmentation. Row 3 presents a high-risk bone lesion from Deep-PSMA, where scratch-trained SwinUNETR misses the lesion, while all three LoRe-MIM variants successfully recover it. Finally, Row 4 illustrates a failure case involving a liver lesion from Deep-PSMA, where decoder-only finetuning fails to detect the lesion and encoder–decoder finetuning yields an incomplete segmentation, whereas full finetuning and scratch-trained nnU-Net successfully recover the lesion.

![](images/f730cd077553e9565551439092b1817413c799efd7be1c190cc85f3293aee852.jpg)  
Figure 8: Qualitative comparison of whole-body PET/CT lesion segmentation between models trained from scratch (nnUNet and SwinUNETR) and our proposed LoRe-MIM pretrained SwinUNETR. Columns 1 and 2 show the coronal CT and PET images, respectively, with ground-truth lesion contours highlighted in green. Columns 3 and 4 present the segmentation predictions from nnUNet and SwinUNETR trained from scratch, respectively. Both models correctly identify the true-positive lesion (green arrow) but introduce an additional false-positive prediction (red arrow). In contrast, our proposed LoRe-MIM pretrained SwinUNETR (Column 5) successfully eliminates the false-positive prediction while preserving the true-positive lesion, demonstrating improved lesion discrimination.

![](images/03cca1aa966e44d06682ab98562f218206d51b774a96eaa938b80758a005d978.jpg)  
Figure 9: Qualitative comparison of masked-image reconstruction for CT (top) and PET (bottom). From left to right: clean reference, masked input, CM reconstruction, and LoRE-MIM reconstruction. Red region outlines highlight corresponding regions where LoRE-MIM provides visually closer agreement with the clean reference. CM and LoRE-MIM reconstructions are outlined in red and dark green, respectively.

First, we introduce one of the largest cohorts for PET/CT pretraining, comprising 5910 unlabeled scans collected from both publicly available and institutional datasets (Sect. 2.1). In contrast to most prior pretraining cohorts that are either only FDG- or PSMA-specific, our dataset incorporates pan-cancer, multi-tracer PET/CT scans spanning diverse patient populations and scanner variations. Such heterogeneity enables the pretraining framework to learn more generalizable and robust representations across different imaging conditions and disease characteristics. We investigate the generalizability of the learned representations through whole-body lesion segmentation across independent testing cohorts, while evaluating their robustness through patch-based classification under clinically relevant perturbations.

Second, we adopt a masked image modeling (MIM) based reconstruction objective, inspired by its success in natural images and medical imaging domains. However, the application of MIM in multimodal medical imaging remains relatively underexplored, and the few existing approaches often mimic unimodal masking strategies that capture relationships between spatially aligned imaging modalities and enable accurate prediction even when one modality is missing. This leaves considerable room for exploring cross-modal representation learning. To address this limitation, we propose LoRe-MIM, a novel masking strategy that emulates solving two puzzles simulataneously, in which some puzzle pieces from each puzzle are either lost or replaces with the alternate puzzle (Sect. 2.4). Instead of adopting a fixed strategy for either removing or replacing all the masked blocks, we randomize the perturbation process using a Bernoulli random variable with a probability of 0.5. Through this stochastic masking mechanism, we encourage the pretrained model to learn anatomy and within-modal structural consistency as well as cross-modal relationships, correlations and complementarity, thereby capturing richer representations of multimodal PET/CT scans. Through ablation studies, we demonstrate that our proposed LoRe-MIM achieves the best reconstruction performance across all three metrics (MAE, SSIM, and PSNR), along with the highest classification AUC, compared to alternative masking strategies.

Third, we provide an extensive evaluation of our proposed dataset and pretraining strategy across multiple downstream tasks. While existing self-supervised pretraining approaches primarily evaluate their effectiveness through lesion segmentation, we investigate two complementary tasks: (1) patch-based classification and (2) whole-body lesion segmentation. We evaluate classification performance under both clean and perturbed imaging conditions, whereas segmentation performance is assessed at the voxel, lesion, and anatomical region levels.

In patch-based classification, we systematically evaluate different feature extraction stages, feature aggregation strate gies, and dimensionality reduction techniques. We observe that, among the evaluated configurations using encoder features, global average pooling (GAP) combined with logistic regression (LR) achieves the best classification performance. Interestingly, when adopting GAP and LR, features extracted from the decoder stages outperform those extracted from the encoder. Finally, using features extracted from the final decoder stage, we demonstrate that our pretrained representations achieve superior classification performance compared to widely adopted pretrained archi tectures, including DINOv3 and ResNet50-3D, under comparable feature dimensionality reduction settings.

We also benchmark the robustness of our pretrained representations under clinically relevant perturbations. Specifically, we introduce Gaussian noise into the 3D PET/CT patches, emulate spatial misalignment between PET and CT, and simulate missing modalities by independently removing either CT or PET. Across these perturbation settings, we observe that features extracted from the final decoder stage consistently outperform those obtained from DINOv3 and ResNet50-3D. These findings further support our hypothesis that the heterogeneous pretraining dataset and proposed LoRe-MIM strategy facilitate the learning of robust representations that maintain discriminative performance even under degraded or incomplete imaging conditions. Notably, the superior performance compared to DINOv3 highlights the potential of domain-specific multimodal pretraining over widely adopted general-purpose foundation models for PET/CT-based analysis.

Finally, we demonstrate the effectiveness of our pretraining approach for whole-body lesion segmentation. We investigate three different finetuning strategies by selectively finetuning different components of the pretrained model. Acros all three strategies, for both of our testing datasets AutoPET-III and Deep-PSMA, we observe consistent improvements in Dice scores compared to the corresponding model trained from scratch, along with reductions in FPVol and FNVol in several configurations. Specifically, on AutoPET-III, finetuning both the encoder and decoder improves Dice from 0.526 to 0.581 and reduces FPVol from 27.147 cc to 15.662 cc, whereas on Deep-PSMA, decoder finetuning improves Dice from 0.547 to 0.610 while reducing FPVol from 74.315 cc to 48.432 cc and FNVol from 9.785 cc to 6.588 cc (Table 7). We further compare our approach with an independent nnU-Net architecture optimized for AutoPET-III, trained under comparable preprocessing, augmentation, and training conditions. Although our chosen architecture, SwinUNETR, trained from scratch performs inferiorly to nnU-Net, our pretraining strategy enables SwinUNETR to achieve improved segmentation performance and outperform the nnU-Net baseline in terms of Dice, achieving 0.581 compared to 0.566 on AutoPET-III and 0.613 compared to 0.561 on Deep-PSMA. Beyond voxel-level evaluation, we conduct comprehensive lesion-level and anatomical region-level analyses and observe that our pretraining approach improves lesion detection sensitivity and segmentation accuracy in clinically critical regions, particularly the liver and high-risk bone regions. These improvements highlight the potential clinical relevance of pretraining, especially for detecting and segmenting metastases in anatomically challenging and high-risk regions. Our label-efficiency curves in Figure 6 further demonstrate that the benefits of pretraining are most pronounced when only a limited number of labeled training samples are available, with diminishing gains as the number of labeled samples increases. The improvements in voxel- and lesion-level performance on the unseen Deep-PSMA dataset, which was not introduced during either pretraining or finetuning (Table 7), together with the anatomical region-level improvements (Figure 5) and label-efficiency gains (Figure 6), demonstrate the generalizability of the learned representations and highlight the effectiveness of our heterogeneous pretraining dataset and proposed perturbation strategy.

Overall, our findings demonstrate that the combination of a large-scale, heterogeneous PET/CT dataset and the proposed LoRe-MIM masking strategy enables effective learning of generalizable and robust multimodal representations. Through extensive evaluation of reconstruction, classification, robustness, and whole-body lesion segmentation, we show that the benefits of self-supervised pretraining extend beyond improvements in individual downstream tasks. In particular, our results highlight the importance of incorporating cross-modal interactions during pretraining and leveraging heterogeneous imaging cohorts to develop more reliable and transferable deep learning models for whole-body PET/CT analysis.

## Conclusion

In this work, we introduced a novel self-supervised masked image modeling framework, LoRe-MIM, to learn generalizable and robust representations from whole-body PET/CT scans. We curated and subsequently leveraged a large-scale, heterogeneous pretraining cohort of 5,910 unlabeled PET/CT scans collected from both publicly available and institutional datasets. Our proposed block-based perturbation strategy encourages the learning of metabolic and complementary anatomical representations through a stochastic masking mechanism, in which image blocks are either removed (Lost) by replacing them with zero values or replaced (Replaced) with corresponding blocks from the alternate modality. We extensively evaluated our proposed framework across whole-body image reconstruction, patch-based classification of cancerous and non-cancerous regions, and whole-body lesion segmentation. We demonstrated that our method achieves superior reconstruction performance compared to an FDG-only pretrained model, improves classification performance under both clean and perturbed imaging conditions compared to established pretrained architectures such as DINOv3 and ResNet50-3D, and enhances lesion segmentation performance at the voxel, lesion, and anatomical region levels. Overall, these findings establish the utility of large-scale, heterogeneous pretraining datasets and the effectiveness of our proposed cross-modal self-supervised learning strategy in learning rich, generalizable, and robust representations for whole-body PET/CT analysis.

## Acknowledgements

Research reported in this publication was supported by an Institutional Development Award (IDeA) from the National Institute of General Medical Sciences of the National Institutes of Health under grant number 1P30GM149408. We also gratefully acknowledge support from the Munck-Pfefferkorn Fund, the American Cancer Society Institutiona Research Grant, the Department of Biomedical Data Science at Dartmouth College and the Department of Radiology at Dartmouth Hitchcock Medical Center. All of this support was instrumental in making this work possible. Help from Claude Opus 4.8 was taken for correcting grammatical errors.

## References

[1] Hofman, M. S. and Hicks, R. J., “How we read oncologic FDG PET/CT,” Cancer Imaging 16(1), 35 (2016).

[2] Mingels, C., Chung, K. J., Pantel, A. R., Rominger, A., Alberts, I., Spencer, B. A., Nardo, L., and Pyka, T., “Total-body PET/CT: Challenges and opportunities,” Seminars in Nuclear Medicine 55(1), 21–30 (2025).

[3] Lindgren Belal, S., Frantz, S., Minarik, D., Enqvist, O., Wikstrom, E., Edenbrandt, L., and Tr¨ ag¨ ardh, E.,˚ “Applications of artificial intelligence in PSMA PET/CT for prostate cancer imaging,” Seminars in Nuclear Medicine 54(1), 141–149 (2024).

[4] Trotter, J., Pantel, A. R., Teo, B.-K. K., Escorcia, F. E., Li, T., Pryma, D. A., and Taunk, N. K., “Positron emission tomography (PET)/computed tomography (CT) imaging in radiation therapy treatment planning: A review of PET imaging tracers and methods to incorporate PET/CT,” Advances in Radiation Oncology 8(5), 101212 (2023).

[5] Sung, H., Filho, A. M., Laversanne, M., Ferlay, J., Siegel, R. L., Soerjomataram, I., Jemal, A., and Bray, F., “Global cancer statistics 2024: Globocan estimates of incidence and mortality worldwide for 34 cancers in 186 countries,” CA: A Cancer Journalfor Clinicians 76(4), e70090 (2026).

[6] Gatidis, S., Hepp, T., Fruh, M., La Foug¨ ere, C., Nikolaou, K., Pfannenberg, C., Sch\` olkopf, B., K¨ ustner, T., Cyran,¨ C., and Rubin, D., “A whole-body FDG-PET/CT dataset with manually annotated tumor lesions,” Scientific Data 9(1), 601 (2022).

[7] Ingrisch, M., Dexl, J., Jeblick, K., Cyran, C., Gatidis, S., and Kustner, T., “Automated lesion segmentation in¨ whole-body PET/CT – multitracer multicenter generalization.” Challenge proposal, 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2024) (2024).

[8] Rokuss, M., Kovacs, B., Kirchhoff, Y., Xiao, S., Ulrich, C., Maier-Hein, K. H., and Isensee, F., “From FDG to PSMA: A hitchhiker’s guide to multitracer, multicenter lesion segmentation in PET/CT imaging,” arXiv preprint arXiv:2409.09478 (2024).

[9] Xu, L., Tetteh, G., Lipkova, J., Zhao, Y., Li, H., Christ, P., Piraud, M., Buck, A., Shi, K., and Menze, B. H., “Automated whole-body bone lesion detection for multiple myeloma on 68ga-pentixafor pet/ct imaging using deep learning methods,” Contrast media & molecular imaging 2018(1), 2391925 (2018).

[10] Blanc-Durand, P., Jegou, S., Kanoun, S., Berriolo-Riedinger, A., Bodet-Milin, C., Kraeber-Bod´ er´ e, F., Carlier,´ T., Le Gouill, S., Casasnovas, R.-O., Meignan, M., et al., “Fully automatic segmentation of diffuse large b cell lymphoma lesions on 3d fdg-pet/ct for total metabolic tumour volume prediction using a convolutional neural network.,” European Journal ofNuclear Medicine and Molecular Imaging 48(5), 1362–1370 (2021).

[11] Jemaa, S., Fredrickson, J., Carano, R. A., Nielsen, T., de Crespigny, A., and Bengtsson, T., “Tumor segmentation and feature extraction from whole-body fdg-pet/ct using cascaded 2d and 3d convolutional neural networks,” Journal ofdigital imaging 33(4), 888–894 (2020).

[12] He, J., Zhang, Y., Chung, M., Wang, M., Wang, K., Ma, Y., Ding, X., Li, Q., and Pu, Y., “Whole-body tumor segmentation from pet/ct images using a two-stage cascaded neural network with camouflaged object detection mechanisms,” Medical Physics 50(10), 6151–6162 (2023).

[13] Gatidis, S., Fruh, M., Fabritius, M. P., Gu, S., Nikolaou, K., Foug¨ ere, C. L., Ye, J., He, J., Peng, Y., Bi, L., et al.,\` “Results from the autopet challenge on fully automated lesion segmentation in oncologic pet/ct imaging,” Nature Machine Intelligence 6(11), 1396–1405 (2024).

[14] Liu, A., Jia, D., Sun, K., Meng, R., Zhao, M., Jiang, Y., Dong, Z., Gao, Y., and Shen, D., “Lm-unet: Whole-body pet-ct lesion segmentation with dual-modality-based annotations driven by latent mamba u-net,” in [International Conference on Medical Image Computing and Computer-Assisted Intervention], 405–414, Springer (2024).

[15] Xu, Y., Klyuzhin, I., Harsini, S., Ortiz, A., Zhang, S., Benard, F., Dodhia, R., Uribe, C. F., Rahmim, A., and´ Ferres, J. L., “Automatic segmentation of prostate cancer metastases in psma pet/ct images using deep neural networks with weighted batch-wise dice loss,” Computers in Biology and Medicine 158, 106882 (2023).

[16] Li, Y., Imami, M. R., Zhao, L., Amindarolzarbi, A., Mena, E., Leal, J., Chen, J., Gafita, A., Voter, A. F., Li, X., et al., “An automated deep learning-based framework for uptake segmentation and classification on psma pet/ct imaging of patients with prostate cancer,” Journal ofImaging Informatics in Medicine 37(5), 2206–2215 (2024).

[17] Milletari, F., Navab, N., and Ahmadi, S.-A., “V-net: Fully convolutional neural networks for volumetric medical image segmentation,” in [2016fourth international conference on 3D vision (3DV)], 565–571, Ieee (2016).

[18] Sibille, L., Seifert, R., Avramovic, N., Vehren, T., Spottiswoode, B., Zuehlsdorff, S., and Schafers, M., “18f-¨ fdg pet/ct uptake classification in lymphoma and lung cancer by using deep convolutional neural networks,” Radiology 294(2), 445–452 (2020).

[19] Mu, W., Jiang, L., Zhang, J., Shi, Y., Gray, J. E., Tunali, I., Gao, C., Sun, Y., Tian, J., Zhao, X., et al., “Noninvasive decision support for nsclc treatment using pet/ct radiomics,” Nature communications 11(1), 5228 (2020).

[20] Shao, X., Niu, R., Shao, X., Gao, J., Shi, Y., Jiang, Z., and Wang, Y., “Application of dual-stream 3d convolutional neural network based on 18f-fdg pet/ct in distinguishing benign and invasive adenocarcinoma in ground-glass lung nodules,” EJNMMIphysics 8(1), 74 (2021).

[21] Alves, V. M., dos Santos Cardoso, J., and Gama, J., “Classification of pulmonary nodules in 2-[18f] fdg pet/ct images with a 3d convolutional neural network,” Nuclear Medicine and Molecular Imaging 58(1), 9–24 (2024).

[22] Tao, Z., Bing-Qiang, H., Huiling, L., Hongbin, S., Pengfei, Y., and Hongsheng, D., “18f-fdg-pet/ct whole-body imaging lung tumor diagnostic model: An ensemble e-resnet-nrc with divided sample space,” BioMed Research International 2021(1), 8865237 (2021).

[23] Yang, Y., Zheng, B., Li, Y., Li, Y., and Ma, X., “Computer-aided diagnostic models to classify lymph node metastasis and lymphoma involvement in enlarged cervical lymph nodes using pet/ct,” Medical Physics 50(1), 152–162 (2023).

[24] Shao, X., Ge, X., Gao, J., Niu, R., Shi, Y., Shao, X., Jiang, Z., Li, R., and Wang, Y., “Transfer learning–based pet/ct three-dimensional convolutional neural network fusion of image and clinical information for prediction of egfr mutation in lung adenocarcinoma,” BMC Medical Imaging 24(1), 54 (2024).

[25] Nishigaki, D., Suzuki, Y., Watabe, T., Katayama, D., Kato, H., Wataya, T., Kita, K., Sato, J., Tomiyama, N., and Kido, S., “Vision transformer to differentiate between benign and malignant slices in 18f-fdg pet/ct,” Scientific Reports 14(1), 8334 (2024).

[26] Krizhevsky, A., Sutskever, I., and Hinton, G. E., “Imagenet classification with deep convolutional neural net works,” Advances in neural information processing systems 25 (2012).

[27] Simonyan, K. and Zisserman, A., “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556 (2014).

[28] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z., “Rethinking the inception architecture for computer vision,” in [Proceedings of the IEEE conference on computer vision and pattern recognition], 2818– 2826 (2016).

[29] He, K., Zhang, X., Ren, S., and Sun, J., “Deep residual learning for image recognition,” in [Proceedings of the IEEE conference on computer vision and pattern recognition], 770–778 (2016).

[30] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L., “Imagenet: A large-scale hierarchical image database,” in [2009 IEEE conference on computer vision and pattern recognition], 248–255, Ieee (2009).

[31] Yazdani, E., Karamzadeh-Ziarati, N., Cheshmi, S. S., Sadeghi, M., Geramifar, P., Vosoughi, H., Kazemi Jahromi, M., and Kheradpisheh, S. R., “Automated segmentation of lesions and organs at risk on [<sup>68</sup>Ga]Ga-PSMA-11 PET/CT images using self-supervised learning with Swin UNETR,” Cancer Imaging 24(1), 30 (2024).

[32] Oh, Y., Seifert, R., Cao, Y., Clement, C., Ferdinandus, J., Lapa, C., Liebich, A., Amon, M., Enke, J., Song, S., Meng, R., Zeng, F., Guo, N., Li, X., Heidari, P., Rominger, A., Shi, K., and Li, Q., “Developing a PET/CT foundation model for cross-modal anatomical and functional imaging,” arXiv preprint arXiv:2503.02824 (2025).

[33] Huang, J., Chen, S., Liang, X., Yang, X., Zhang, Z., Sun, Y., Wang, Y., and Tan, T., “C<sup>2</sup>MAOT: Cross-modal complementary masked autoencoder with optimal transport for cancer segmentation in PET-CT images,” in [Medical Image Computing and Computer Assisted Intervention – MICCAI 2025], Lecture Notes in Computer Science 15960, 87–97, Springer Nature Switzerland (2025).

[34] Keshavarz, S., Saeedzadeh, E., Arabi, H., Jenabi-Haghparast, E., and Sardari, D., “Self-supervised multimodal swin unetr for pet/ct segmentation of diffuse large b-cell lymphoma,” Cancer Treatment and Research Communications 49, 101369 (2026).

[35] Liu, X., Zhang, Q., Marin, T., Xia, M., Liu, C., El Fakhri, G., and Ouyang, J., “An open multi-center whole-body FDG PET/CT foundation model for tumor segmentation,” arXiv preprint arXiv:2605.21835 (2026).

[36] Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. R., and Xu, D., “Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images,” in [International MICCAI brainlesion workshop], 272– 284, Springer (2021).

[37] Jackson, P., Hofman, M., Buteau, J. P., McIntosh, L., and Sun, Y., “A repository of annotated PSMA and FDG PET/CT images for algorithm development in staging of mCRPC for treatment with <sup>177</sup>Lu-PSMA therapy,” (2025).

[38] Eyuboglu, S., Angus, G., Patel, B. N., Pareek, A., Davidzon, G., Long, J., Dunnmon, J., and Lungren, M. P., “Multi-task weak supervision enables anatomically-resolved abnormality detection in whole-body FDG-PET/CT,” Nature Communications 12(1), 1880 (2021).

[39] Nguyen, H. T., Nguyen, D. T., Nguyen, T. M. D., Nguyen, T. T., Truong, T. N., Pham, H. H., Barthelemy, J., Tran, M. Q., Nguyen, T. T., Nguyen, Q. V. H., Chau, Q. A., Mai, H. S., Nguyen, T. T., and Nguyen, P. L., “Toward a vision-language foundation model for medical data: Multimodal dataset and benchmarks for vietnamese PET/CT report generation,” in [Advances in Neural Information Processing Systems], 38, 129875–129906 (2025).

[40] Zaidi, H., “Optimisation of whole-body pet/ct scanning protocols,” Biomedical imaging and intervention journal 3(2), e36 (2007).

[41] Callahan, J., Kron, T., Schneider-Kolsky, M., and Hicks, R. J., “The clinical significance and management of lesion motion due to respiration during pet/ct scanning,” Cancer Imaging 11(1), 224 (2011).

[42] Osman, M. M., Cohade, C., Nakamoto, Y., and Wahl, R. L., “Respiratory motion artifacts on pet emission images obtained using ct attenuation correction on pet-ct,” European journal of nuclear medicine and molecular imaging 30(4), 603–606 (2003).

[43] Wasserthal, J., Breit, H.-C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., Heye, T., Boll, D. T., Cyriac, J., Yang, S., et al., “Totalsegmentator: robust segmentation of 104 anatomic structures in ct images,” Radiology: Artificial Intelligence 5(5), e230024 (2023).

[44] Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., Murrey, B., Myronenko, A., Zhao, C., Yang, D., et al., “Monai: An open-source framework for deep learning in healthcare,” arXiv preprint arXiv:2211.02701 (2022).

[45] Biswas, B. A., Bhattacharya, A., Wagle, B. R., Maeder, M. E., Yu, J. B., and Bhattacharya, I., “Must-pet: Multimodal self-supervised learning across tracers for whole-body pet/ct-based lesion segmentation,” arXiv preprint arXiv:2608.19666 (2026).

[46] Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S.,´ Ramamonjisoa, M., et al., “Dinov3,” arXiv preprint arXiv:2508.10104 (2025).

[47] Chen, S., Ma, K., and Zheng, Y., “Med3d: Transfer learning for 3d medical image analysis,” arXiv preprint arXiv:1904.00625 (2019).