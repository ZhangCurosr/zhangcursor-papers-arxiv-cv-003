# Med-RADIO: Reducing All Medical Domains Into One via Multi-Teacher Distillation

Chu Zhang<sup>1</sup>, Haoyu Jiang<sup>1</sup>, Hongyuan Zhang<sup>1</sup>, Hongbin Liu<sup>1</sup>, and Dong Yi<sup>1</sup>

<sup>1</sup>Center for Artificial Intelligence and Robotics, Hong Kong Institute of Science and Innovation, Chinese Academy of Sciences, Hong Kong, China

## Abstract

The rapid expansion of large-scale medical datasets and computational resources has driven significant progress in medical foundation models. Given the inherent heterogeneity of medical imaging modalities, current research mainly follows two paths: specialized models optimized for specific modalities, and generalist models designed to handle multiple modalities. However, medical generalist models suffer from both insufficient training data scale relative to natural image generalists and inadequate domain-specific depth relative to medical specialists. Empirically, generalist models establish a cross-modality performance baseline, while specialists define the performance ceiling within their respective domains. To elevate this baseline toward these ceilings, we propose Med-RADIO, a medical multi-teacher distillation framework that Reduces All Domains Into One by compressing complementary expertise from multiple domain-specific teachers into a unified medical vision foundation model. Our method curates both generalist and specialist teachers, allocates modality-aligned distillation streams to reorganize generalist pretraining data so it matches specialist domains, and uses a balanced loss to prevent any single teacher from dominating the distillation process. On internal and external classification benchmarks spanning five modalities, Med-RADIO improves over strong medical generalists under linear probing and remains competitive with representative specialists on most evaluated modalities. Code is available at https://github.com/CAIR-HKISI/Med-RADIO.

Keywords: Medical Vision Foundation Model, Multi-Teacher Distillation, Representation Learning

## 1 Introduction

The rapid expansion of large-scale medical datasets and computational resources has driven significant progress in medical foundation models [38]. Given the inherent heterogeneity of medical imaging modalities, current research follows two trajectories: specialized models optimized for specific modalities such as chest X-ray, pathology, and dermatology, and generalist models designed to handle multiple modalities. However, medical generalist models face a structural weakness rooted in data scale. Compared to well-established generalists in the natural image domain such as CLIP [37, 14], DINOv2 [32, 46], and SAM [21, 41, 2], they are trained on orders-of-magnitude less data (billions versus millions of images). Consequently, they fail to develop the domain-specific depth comparable to medical specialists trained on millions of modality-specific images. This data deficit restricts generalist models to a cross-modality performance baseline, while specialists define the performance ceiling within their respective domains.

This discrepancy raises a fundamental question: can we construct a unified medical vision foundation model that elevates generalist capability toward specialist performance? Several straightforward approaches prove inadequate. Ensembling [6] specialist models incurs prohibitive inference costs and yields no unified representation. Weight merging [42] is infeasible due to architectural heterogeneity across backbones. Multi-teacher knowledge distillation [52] offers a principled alternative—training a single student to integrate complementary expertise—yet applying it to medical imaging reveals three critical barriers. First, architectural heterogeneity across medical teachers produces misaligned output spaces. Second, modality heterogeneity introduces distribution sensitivity unique to healthcare: specialist teachers require modality-specific inputs to transfer domain expertise effectively. Third, optimization dynamics tend to be dominated by stronger teachers, suppressing weaker but complementary signals and causing imbalanced capability inheritance [40].

![](images/99952fcf8f7ceb6060f60e23c07bef906725b819bc277a8b3767a4bf3b775bea.jpg)  
(a)

![](images/ee649dfb37643fdd17a490cfd82da9925a4a218ff566a67e73fe6eafc22b156c.jpg)  
(b)

![](images/206e4f0ad4ddcf7c59225e16ab9fb2943d62e1e362726f9d9d6a934bf8411e16.jpg)  
(c)  
Figure 1: Motivation and overview. (a) Generalist–specialist performance gap under limited medical data scale. (b) Med-RADIO compresses heterogeneous teachers into one student encoder. (c) Linear-probing average versus medical generalist baselines (85.28% for Med-RADIO).

To address these challenges, we propose Med-RADIO, a medical multi-teacher distillation framework that Reduces All Domains Into One by compressing complementary expertise from multiple domain-specific teachers into a unified medical vision foundation model. Our key insight is data leverage: by repartitioning the generalist pretraining corpus into modality-specific streams, we indirectly access the distributed expertise of specialists—collectively exposed to tens of millions of domain-specific images—without retrieving original specialist corpora. By freezing all teachers and training only the student encoder with lightweight projectors, we decouple data scale from computational cost, achieving specialist-approaching depth at generalist-scale expense.

Our methodology rests on three innovations. (1) Teacher curation with modality-specific projectors: we systematically select complementary teachers and introduce domain-specific MLP projectors to align heterogeneous output spaces. (2) Modality-aligned distillation streams: we reorganize the pretraining corpus of UniMed-CLIP [19] into single-modality subsets, ensuring each specialist receives data drawn from its native distribution. (3) Balanced distillation loss: we employ a contribution-aware weighting scheme that prevents dominant teachers from suppressing complementary signals; UniMed-CLIP further serves as a mandatory generalist anchor, preserving cross-modal stability.

We validate Med-RADIO through extensive experiments on internal and external classification benchmarks spanning five medical imaging modalities. Our unified encoder consistently improves over strong generalist baselines and remains competitive with dedicated specialists on most modalities, demonstrating that strategic distillation enables unified breadth and depth without proportional resource investment. By elevating generalist capability toward specialist performance, Med-RADIO offers a practical path for deploying single, versatile models across heterogeneous medical imaging workflows.

## 2 Related Work

## 2.1 Medical Vision Foundation Models

Large-scale medical vision foundation models have proliferated across radiology, pathology, and multimodal imaging. Early efforts concentrated on modality-specific pretraining, yielding specialized architectures for ultrasound [17, 16, 53], MRI [5, 1], CT [33, 10, 4], chest X-ray [35, 51], fundus photography [55, 44, 45, 36], histopathology [3, 28, 13], and dermatology [49, 20]. These specialist models leverage modality-tailored architectures and inductive biases to establish strong performance ceilings within their respective domains.

Concurrently, generalist medical vision models aim to acquire cross-modality representations from largescale heterogeneous corpora, including vision-language alignment models such as BioMedCLIP [54], Pub-

MedCLIP [8], PMC-CLIP [24], BMCA-CLIP [27], and UniMed-CLIP [19], as well as unified multi-modality encoders such as Uni-Med [56], Lingshu [48], and Hulu-Med [15]. However, these models confront a fundamental structural deficit: relative to natural image foundation models trained on billions of images, medical generalists rely on corpora typically comprising only millions of images. This data gap causes generalist models to consistently underperform modality-specific experts on single-modality benchmarks.

Recent attempts to close this gap via model ensemble [29] or weight merging [50] encounter fundamental practical barriers: ensemble methods incur multiplicative inference costs incompatible with clinical workflows, while architectural heterogeneity across backbones precludes direct parameter fusion. These limitations motivate distillation-based approaches as a principled alternative.

## 2.2 Distilling Foundation Models

Knowledge distillation [52] has recently evolved from model compression toward distillation among foundation models themselves. AM-RADIO [39] and RADIOv2.5 [11] pioneered multi-teacher distillation in natural images, unifying heterogeneous encoders including SAM, CLIP, and DINOv2 into a single versatile backbone. However, these works operate on homogeneous photographic data, circumventing the modality divergence and optimization imbalance intrinsic to medical imaging.

In the medical domain, distillation remains limited in scope: existing efforts distill from single teachers [34, 9], operate within identical architectures [47], or restrict transfer to a single modality [30]. Med-RADIO instead targets multi-teacher distillation across heterogeneous medical foundation models, jointly addressing architectural misalignment, optimization imbalance, and modality distribution sensitivity.

## 2.3 Multi-Teacher Knowledge Distillation

Multi-teacher knowledge distillation (MKD) aggregates complementary supervisory signals from diverse experts [52]. Representation space alignment is critical: directly enforcing feature consistency between structurally dissimilar teachers and students imposes overly rigid constraints [26]. Early uniform weighting schemes treat all teachers equally, risking performance collapse or teacher dominance. Adaptive mechanisms have been proposed to mitigate these issues, including uncertainty-aware weighting [18], auxiliary loss-based aggregation [12], and gradient-balanced adaptive weights [25].

Medical imaging exacerbates these challenges: diverse backbones (ViT variants, Swin Transformers) produce severely misaligned output spaces; heterogeneous pretraining scales cause dominant specialists to suppress complementary generalists; and modality distribution sensitivity requires preserving specialized inductive biases under native data exposure. Med-RADIO addresses these issues with modality-specific projection heads, a contribution-aware balanced loss, and modality-aligned data streams.

## 3 Methodology

## 3.1 Framework Overview

Our objective is to develop a unified medical vision foundation model that synergistically integrates crossmodality generalization capabilities from a generalist teacher with fine-grained, modality-specific expertise from multiple specialist teachers. To achieve this, we propose a multi-teacher distillation framework comprising three interconnected components: (1) a heterogeneous ensemble of pre-trained teacher models, (2) modalityspecific translator heads for cross-architecture feature alignment, and (3) a balanced distillation objective.

As illustrated in Fig. 2, the training pipeline operates through multiple concurrent data streams: each specialist teacher supervises samples from its dedicated modality, while the generalist teacher processes the entire heterogeneous corpus to provide cross-domain regularization. The unified student encoder, shared across all streams, aggregates complementary knowledge through the balanced distillation objective (Eq. 5). This design lets the student inherit both broad transferable representations and precise domain-specific inductive biases.

![](images/5f2a82e6e8c94c33e58fc20048ccb4acd7b72349ade646f955e99f8164345a19.jpg)  
Figure 2: Med-RADIO training pipeline: modality-aligned streams, frozen generalist/specialist teachers, translator heads, and a dispersion-balanced distillation objective feeding a shared student encoder.

## 3.2 Pretraining Dataset Construction

Data Leverage. A naive approach to multi-teacher distillation would require curating separate, large-scale pretraining datasets for each specialist teacher, a prohibitively expensive endeavor given the scarcity of annotated medical data. We circumvent this bottleneck through a modality-decoupled data reconstruction strategy: rather than assembling new datasets, we repartition the existing pretraining corpus of UniMed-CLIP into exclusive single-modality subsets, each serving as a dedicated distillation stream for the corresponding specialist teacher. This design enables the generalist teacher to maintain cross-modal alignment on the complete heterogeneous mixture, while each specialist receives purified, modality-focused supervision, without any additional data collection or annotation effort.

Dataset Composition. The pretraining data of UniMed-CLIP is composed of subsets encompassing seven major medical imaging modalities: X-ray, computed tomography (CT), magnetic resonance imaging (MRI), ultrasound (US), histopathology whole-slide images (WSI), dermatology photography and Fundus. To support the multi-teacher distillation, we derive five single-modality subsets (US, MRI, CT, histopathology, and dermatology) from this corpus, alongside one multimodal mixture drawn from heterogeneous sources such as PMC-OA[24], ROCOv2[43], and LLaVA-Med[23]. Detailed dataset curation procedures and modality-wise statistics are provided in the Supplementary Material.

## 3.3 Teacher Model Composition

We construct a heterogeneous teacher ensemble $\mathcal { T } = \{ T _ { g } \} \cup \{ T _ { m } \} _ { m \in \mathcal { M } }$ comprising one generalist medical vision foundation model and multiple modality-specific specialists. Formally, let $\mathcal { M } { = } \{ \mathrm { U S , C T , M R I , P a t h , D e r m } \}$ denote the set of specialist-covered modalities. The generalist teacher $T _ { g }$ provides cross-modality representations $f _ { g } : \mathcal { X }  \mathbb { R } ^ { D _ { g } }$ , while each specialist teacher $T _ { m }$ offers modality-specific inductive biases through $f _ { m } : \mathcal { X } _ { m } \to { \mathbb { R } } ^ { D _ { m } }$ . The student encoder $f _ { s } ( \cdot ; \theta )$ is trained solely via multi-teacher distillation:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { K D } } ^ { ( g ) } ( \theta ) + \sum _ { m \in \mathcal { M } } \mathcal { L } _ { \mathrm { K D } } ^ { ( m ) } ( \theta ) ,\tag{1}
$$

where the per-teacher losses $\mathcal { L } _ { \mathrm { K D } } ^ { ( g ) }$ and $\{ \mathcal { L } _ { \mathrm { K D } } ^ { ( m ) } \}$ are detailed in Sec. 3.5.

## 3.4 Modality-Specific Translator Heads

Direct distillation from heterogeneous teachers is impeded by discrepancies in backbone architecture, feature dimensionality, and representation semantics. To bridge these gaps without modifying the frozen teachers, we introduce lightweight translator heads that project student features into each teacher’s output space.

For each teacher m, we instantiate a dedicated three-layer MLP projector $\phi _ { m } : \mathbb { R } ^ { d _ { s } }  \mathbb { R } ^ { d _ { m } }$

$$
\phi _ { m } ( \mathbf { z } ) = \mathbf { W } _ { 3 } \cdot \boldsymbol { \sigma } ( \mathbf { W } _ { 2 } \cdot \boldsymbol { \sigma } ( \mathbf { W } _ { 1 } \mathbf { z } ) ) ,\tag{2}
$$

where $d _ { s }$ is the student’s output dimension, $d _ { m }$ is teacher m’s output dimension, all hidden layers use a fixed width of 2048, and $\sigma$ denotes the SiLU activation. The student maintains two sets of projectors: generalist projectors $\left\{ \phi _ { m } ^ { g } \right\}$ aligned to the generalist teacher, and specialist projectors $\{ \phi _ { m } ^ { s } \}$ aligned to the modality-specific specialists, each instantiated with its respective target dimension $d _ { m }$

This design decouples representation alignment from feature extraction: the shared student encoder $f _ { \theta }$ learns a unified medical representation, while each $\phi _ { m }$ adapts that representation to its teacher’s geometry. All translator heads are discarded after pretraining; downstream tasks use only the bare encoder $f _ { \theta }$ , incurring no additional inference overhead.

## 3.5 Balanced Distillation Loss

A central challenge in multi-teacher distillation is teacher collapse—where a dominant teacher suppresses complementary signals. Rather than relying on dynamic weighting schemes, we address this through a statisticsbased balancing mechanism[40] built into each per-teacher loss $\mathcal { L } _ { \mathrm { K D } } ^ { ( m ) }$ in Eq. 1: each loss is normalized by its teacher’s feature dispersion, estimated offline from the corresponding modality stream prior to training.

Rather than minimizing raw feature distances, we supervise the geometric structure of the feature space via an angular loss. Given student feature x and teacher feature y, the angular distance is:

$$
\Theta ( \mathbf { x } , \mathbf { y } ) = \operatorname { a r c c o s } \left( { \frac { \mathbf { x } ^ { \top } \mathbf { y } } { \| \mathbf { x } \| \| \mathbf { y } \| } } \right) .\tag{3}
$$

To make this loss scale-invariant across teachers with different feature variances, we normalize by the intrinsic angular dispersion of each teacher’s distribution. Let $\mu _ { \mathbf { y } } = \mathbb { E } [ \mathbf { y } ] / \lVert \mathbb { E } [ \mathbf { y } ] \rVert$ denote the mean direction of teacher features; the angular dispersion is:

$$
\mathrm { D i s p } ( \Theta _ { \mathbf { y } } ) = \mathbb { E } \left[ \Theta \big ( \mathbf { y } , \mu _ { \mathbf { y } } \big ) ^ { 2 } \right] ,\tag{4}
$$

estimated offline on 10,000 samples from each teacher’s modality stream (Tab. S4 in the Supplementary Material). The normalized angular loss is:

$$
\mathcal { L } _ { \mathrm { K D } } ^ { ( m ) } ( \mathbf { x } , \mathbf { y } ) = \frac { \Theta ( \mathbf { x } , \mathbf { y } ) ^ { 2 } } { \mathrm { D i s p } ( \Theta _ { \mathbf { y } } ) } .\tag{5}
$$

This static normalization simultaneously amplifies suppressed low-dispersion signals and attenuates overdominant high-dispersion ones, achieving equitable gradient magnitudes across teachers without manual weight tuning or online adaptation.

## 4 Experiments

## 4.1 Implementation Details

Teacher Selection. We select the specialist teacher for each modality by evaluating candidate modalityspecific models under linear probing and choosing the top performer, while excluding UniMed-CLIP variants from specialist selection because they already serve as the mandatory generalist anchor. Detailed scores are provided in the Supplementary Material (Tab. S3).

Student Architecture and Training Configuration. The student encoder is ViT-Large/16 (1024-d hidden dimension, 24 transformer layers, 16 attention heads, 224 × 224 input resolution), initialized with standard ImageNet-21k pretraining [7]. Each modality-specific translator head consists of an MLP followed by LayerNorm, mapping the student’s 1024-d CLS token to the corresponding teacher’s output dimensionality; all translator heads are discarded after pretraining, and downstream evaluations use solely the bare student backbone.

Training is conducted with PyTorch and Accelerate across 6×A100 (80 GB) GPUs. The student is optimized for 10 epochs using AdamW $( \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 9 9$ , weight decay 0.05) with a cosine annealing schedule (peak LR $1 \times 1 0 ^ { - 4 } )$ and a 1-epoch linear warm-up. The global batch size is 192 (32 per teacher stream), assembled via proportional multi-stream sampling where each modality contributes samples according to its dataset fraction. Teacher-specific data augmentation is applied independently per stream.

Teacher Feature Statistics and Balancing Rationale. As formalized in Eq. 5, each teacher loss is normalized by its offline angular dispersion $\mathrm { D i s p } ( \Theta _ { \mathbf { y } } )$ (Eq. 4), amplifying low-dispersion teachers and attenuating high-dispersion ones. Estimates on 10,000 samples per stream are reported in Tab. S4 in the Supplementary Material.

Downstream Evaluation. We assess representational quality under two protocols: 1) linear probing: the pre-trained encoder is strictly frozen; only a linear head trained on top of the extracted global CLS token is optimized for 50 epochs with AdamW (peak learning rate $1 \times 1 0 ^ { - 3 }$ , cosine schedule, batch size 256), directly probing the linearly separable structure of the pre-trained representations. 2) full fine-tuning: all encoder parameters are jointly optimized end-to-end with a task-specific head for 50 epochs (peak $\mathrm { L R } 5 \times 1 0 ^ { - 5 }$ , cosine schedule, batch size 64), providing an upper-bound assessment of adaptation capacity.

Benchmarks and Metrics. Our primary (internal) evaluation spans nine classification benchmarks across five modalities: two ultrasound datasets (Thyroid-US and Breast-US), two MRI datasets (ACL Tear and Meniscus Tear), three CT datasets (MediMeTA Axial, Coronal, and Sagittal), one histopathology dataset (PCam), and one dermatology dataset (HAM). Following domain conventions, we report AUC for ultrasound/MRI and ACC for CT, histopathology, and dermatology. To assess generalization beyond this internal suite, we additionally evaluate on an external five-modality classification protocol adapted from MMKD-CLIP [47] (Tab. 2). Further details on the internal datasets are provided in the Supplementary Material.

## 4.2 Main Results

We benchmark Med-RADIO against two groups of baselines: (i) medical generalist models and (ii) modalityspecific specialist models, under both linear probing and full fine-tuning.

Comparison with Medical Generalist Baselines. Tab. 1 reports linear probing results across the nine classification benchmarks.

Under linear probing (Tab. 1), Med-RADIO achieves a mean AUC/ACC of 85.28%, surpassing the strongest generalist UniMed-CLIP (large) by +3.38 pp and all other baselines by over 7 pp. Gains are most pronounced in ultrasound and CT, the modalities where specialist priors are most complementary to generalist pretraining, with Breast-US AUC improving by +11.60 pp (the largest margin) and CT Axial/Coronal/Sagittal ACC by +4.0/+5.5/+4.7 pp, respectively. The two benchmarks where Med-RADIO falls below UniMed-CLIP (large), namely MRI Meniscus AUC (−2.0 pp) and PCam ACC (−1.9 pp), both involve fine-grained texture discrimination that is difficult to capture through global CLS supervision under the strictly frozen evaluation protocol.

Table 1: Linear probing (%) on nine classification benchmarks. Best in bold. AUC for US/MRI; ACC for CT/Histo./Derm.
<table><tr><td rowspan="2">Model</td><td colspan="2">US</td><td colspan="2">MRI</td><td colspan="3">CT</td><td rowspan="2">Histo PCam</td><td rowspan="2">Derm HAM</td><td rowspan="2">Avg</td></tr><tr><td>Thyroid</td><td>Breast</td><td>ACL</td><td>Meniscus</td><td>Axial</td><td>Coronal</td><td>Sagittal</td></tr><tr><td>Biomed-CLIP [54]</td><td>78.25</td><td>81.79</td><td>89.47</td><td>86.99</td><td>67.64</td><td>57.93</td><td>54.05</td><td>83.40</td><td>83.71</td><td>75.91</td></tr><tr><td>Pubmed-CLIP [8]</td><td>78.36</td><td>84.92</td><td>90.91</td><td>88.68</td><td>72.33</td><td>59.39</td><td>54.85</td><td>82.43</td><td>82.62</td><td>77.17</td></tr><tr><td>PMC-CLIP [24]</td><td>56.96</td><td>66.72</td><td>68.75</td><td>74.61</td><td>37.86</td><td>34.14</td><td>34.95</td><td>75.35</td><td>52.25</td><td>55.73</td></tr><tr><td>BMCA-CLIP [27]</td><td>82.57</td><td>82.10</td><td>93.42</td><td>92.06</td><td>69.90</td><td>61.65</td><td>51.78</td><td>85.56</td><td>81.88</td><td>77.88</td></tr><tr><td>MMKD-CLIP [47]</td><td>63.63</td><td>76.52</td><td>97.30</td><td>94.91</td><td>72.01</td><td>62.30</td><td>56.63</td><td>85.69</td><td>85.37</td><td>77.15</td></tr><tr><td>UniMed-CLIP (base) [19]</td><td>76.96</td><td>79.34</td><td>97.30</td><td>95.08</td><td>67.80</td><td>59.22</td><td>55.34</td><td>85.41</td><td>86.26</td><td>78.08</td></tr><tr><td>UniMed-CLIP (large) [19]</td><td>79.18</td><td>77.12</td><td>97.05</td><td>97.53</td><td>76.38</td><td>65.70</td><td>62.78</td><td>90.23</td><td>91.15</td><td>81.90</td></tr><tr><td>Med-RADIO (Ours)</td><td>84.91</td><td>88.72</td><td>98.16</td><td>95.49</td><td>80.42</td><td>71.20</td><td>67.48</td><td>88.32</td><td>92.86</td><td>85.28</td></tr></table>

![](images/397f652437572ec0b29c0979a79c040889cc125928789d55a4354ddcf78b8611.jpg)  
Figure 3: Linear probing versus the strongest dedicated specialist per modality. Bars show absolute AUC/ACC; ∆ above each group is Med-RADIO minus the best specialist.

Comparison with Modality-Specific Specialists. Fig. 3 compares Med-RADIO with dedicated specialists under linear probing. Med-RADIO matches or exceeds the best specialist on four of five modalities. The largest gains appear on ultrasound Thyroid (+10.8 pp over URFM) and CT Sagittal/Axial (+4.1/+3.7 pp over Curia); MRI and dermatology also favor Med-RADIO, with only a negligible Breast deficit (−0.6 pp). The main deficit remains histopathology (−5.8 pp vs. GPFM), consistent with the mismatch between global CLS distillation and WSI patch-level specialist pretraining.

Fig. 4 reports the corresponding fine-tuning gaps. Med-RADIO leads on five of nine tasks, especially ultrasound (Thyroid/Breast +7.0/+5.3 pp), with smaller gains on CT Sagittal/Axial and HAM. The largest deficits remain PCam and Meniscus (about −2–3 pp), mirroring the linear-probing histopathology gap.

Comparison with MTKD and Balancing Baselines. Tab. 2 compares Med-RADIO with MMKD-CLIP [47] and matched balancing variants (Uniform / Uncertainty [18] / AdaLoss [12]) under the internal/external protocol defined above; per-modality scores average the corresponding benchmarks. Med-RADIO attains the best mean (84.66%), beating MMKD-CLIP by +3.71 pp and the strongest balancing baseline (Uncertainty, 84.14%) by +0.52 pp. Gains over MMKD-CLIP concentrate on internal ultrasound/CT, where modality-aligned specialists help most; local trade-offs versus Uncertainty/AdaLoss on a few cells do not overturn the average advantage.

![](images/335e75a0c40f8cc58460cab919f20a14e289243b36bec1af90ac844730030df1.jpg)  
Figure 4: Fine-tuning ∆ (Med-RADIO − best specialist per modality; pp). Positive favors Med-RADIO.

Table 2: Internal/external linear probing (%). External tasks are held out from teacher-pool construction; balancing variants share the Med-RADIO recipe and differ only in loss weighting. Best Avg in bold.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="5">Internal</td><td colspan="5">External</td><td rowspan="2">Avg</td></tr><tr><td>US</td><td>MRI</td><td>CT</td><td>Histo.</td><td>Derm.</td><td>US</td><td>MRI</td><td>CT</td><td>Histo.</td><td>Derm.</td></tr><tr><td>MTKD</td><td>MMKD-CLIP [47]</td><td>70.08</td><td>96.11</td><td>63.65</td><td>85.72</td><td>87.87</td><td>78.17</td><td>99.96</td><td>99.69</td><td>77.00</td><td>70.02</td><td>80.95</td></tr><tr><td rowspan="4">Balancing</td><td>Uniform</td><td>83.56</td><td>95.93</td><td>73.25</td><td>86.92</td><td>90.18</td><td>79.66</td><td>99.98</td><td>98.97</td><td>71.00</td><td>75.11</td><td>82.73</td></tr><tr><td>Uncertainty [18]</td><td>84.73</td><td>96.49</td><td>74.38</td><td>88.64</td><td>93.76</td><td>81.39</td><td>100.00</td><td>99.61</td><td>73.00</td><td>77.21</td><td>84.14</td></tr><tr><td>AdaLoss [12]</td><td>82.89</td><td>95.87</td><td>73.30</td><td>89.15</td><td>94.21</td><td>82.52</td><td>99.99</td><td>99.63</td><td>75.00</td><td>77.61</td><td>83.95</td></tr><tr><td>Med-RADIO</td><td>87.11</td><td>96.69</td><td>72.98</td><td>89.64</td><td>93.44</td><td>81.04</td><td>100.00</td><td>99.69</td><td>76.00</td><td>78.25</td><td>84.66</td></tr></table>

## 4.3 Ablation Studies and Analysis

We conduct controlled ablations to quantify the individual contribution of each design component in Med-RADIO. All experiments follow the linear probing protocol on the nine benchmarks, with results shown in Fig. 5.

![](images/ea11b245949b124d46b54173501ff3a609658cdd4d42c32a1aaa89d801bf7b60.jpg)

![](images/3f8b5ea1f4dec657af13f8c59960e4cab91abbf8a97a152b65c9e5c197d85005.jpg)

![](images/37c49fb52ff5914d1dd930afa2a423ed571acc43ca885f3c0be747f9c63890ed.jpg)  
Figure 5: Linear-probing ablations on nine benchmarks. Each panel contrasts two settings (shaded regions mark where the second legend entry is higher/lower). (a) w/ vs. w/o generalist teacher; (b) ViT-Large vs. ViT-Base student; (c) w/ vs. w/o dispersion-balanced loss.

Effect of the Generalist Teacher (Fig. 5(a)). Removing the generalist while keeping all specialists lowers the mean by −1.74 pp (85.28%→83.55%), mainly on ultrasound Thyroid (−7.84 pp) and CT Axial. Histopathology is nearly unchanged, indicating that the generalist primarily supplies cross-modal regularization where specialist coverage is thinner.

Effect of Student Model Scale (Fig. 5(b)). Scaling from ViT-Base to ViT-Large yields +1.55 pp on average, with the largest lifts on CT Sagittal and MRI Meniscus, suggesting that higher capacity better absorbs

heterogeneous teacher spaces.

Effect of the Balanced Loss Strategy (Fig. 5(c)). Dispersion balancing is the most critical ingredient: removing it collapses the mean by −8.94 pp (85.28%→76.34%), with severe drops on CT and ultrasound Thyroid. Without it, high-dispersion teachers (e.g., Curia on CT) dominate and suppress complementary streams, confirming that equitable weighting is required for heterogeneous multi-teacher distillation.

## 5 Discussion and Limitation

Our results support an affirmative answer to the motivating question: a single student encoder can elevate medical generalist baselines toward specialist ceilings without ensembling models or merging incompatible weights. Under linear probing, Med-RADIO reaches 85.28% mean AUC/ACC (+3.38 pp over UniMed-CLIP large), remains competitive with dedicated specialists on four of five modalities, and outperforms both a medical MTKD baseline and matched balancing alternatives on the internal/external protocol. Taken together, these findings indicate that the breadth–depth dilemma is amenable to data leverage: by repartitioning an existing generalist corpus into modality-aligned streams and distilling frozen heterogeneous teachers through lightweight translators, specialist priors can be absorbed at generalist-scale training cost and deployed in a single forward pass.

Several practical takeaways emerge from the ablations and baseline comparisons. A mandatory generalist teacher is not redundant with specialists: removing it degrades average performance by −1.74 pp and hits ultrasound/CT hardest, consistent with a cross-modal regularizer that stabilizes domains where specialist coverage alone is insufficient. Contribution-aware balancing is likewise essential: unweighted multi-teacher optimization is vulnerable to dispersion extremes (low-dispersion teachers such as URFM being drowned out; highdispersion teachers such as Curia/GPFM dominating), and generic adaptive schemes (Uncertainty, AdaLoss) underperform our angular-dispersion weighting under an otherwise identical recipe. Finally, standard medical MTKD without modality-native teacher assignment (e.g., MMKD-CLIP) lags most clearly on ultrasound and CT, reinforcing that architectural alignment alone does not substitute for distribution-matched supervision.

Nonetheless, important limitations remain. (i) Histopathology stream quality. The largest specialist gap appears on histopathology, which we attribute primarily to a data-quality mismatch: GPFM is pretrained on curated pathology corpora, whereas our histopathology distillation stream comes from Quilt-1M [13], a noisier YouTube-collected collection. Closing this gap likely requires higher-quality, distribution-matched pathology data (and optionally multi-scale supervision), beyond global CLS distillation alone. (ii) Coverage. Our evaluation focuses on 2D classification over five modalities; chest X-ray, fundus, volumetric 3D understanding, dense prediction, and VQA are not yet systematically assessed. (iii) Teacher- and protocol-dependence. Student quality remains bounded by the frozen teacher pool and public stream corpora, and our objective emphasizes CLS angular matching under frozen linear probing, which under-emphasizes fine-grained texture cues (e.g., Meniscus, PCam). External hold-outs mitigate but do not remove this coupling; clinical claims should therefore be read as inference-efficient unification pending broader validation.

## 6 Conclusion

We presented Med-RADIO, a multi-teacher distillation framework that compresses a cross-modal generalist and modality-specific specialists into one vision encoder via translator heads, modality-aligned data streams, and contribution-aware balanced loss. Trained on a restructured 4.5M-image corpus without new specialist corpora, it achieves 85.28% mean linear-probing AUC/ACC (+3.38 pp over the strongest generalist) and matches or surpasses dedicated specialists on four of five modalities in a single forward pass. These results show that principled heterogeneous distillation offers a practical alternative to ensembles and weight merging for unifying medical vision foundations. Future work will extend Med-RADIO to additional modalities, volumetric inputs, and multi-scale pathology supervision.

## References

[1] Avci, M.Y., Borges, P., Wright, P., Yigitsoy, M., Ourselin, S., Cardoso, J.: Mr-clip: Efficient metadataguided learning of mri contrast representations. In: International Workshop on Machine Learning in Medical Imaging. pp. 85–94. Springer (2025)

[2] Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)

[3] Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Song, A.H., Chen, B., Zhang, A., Shao, D., Shaban, M., et al.: Towards a general-purpose foundation model for computational pathology. Nature medicine 30(3), 850–862 (2024)

[4] Dancette, C., Khlaut, J., Saporta, A., Philippe, H., Ferreres, E., Callard, B., Danielou, T., Alberge, L., Machado, L., Tordjman, D., et al.: Curia: A multi-modal foundation model for radiology. arXiv preprint arXiv:2509.06830 (2025)

[5] Dong, H., Chen, Y., Gu, H., Konz, N., Chen, Y., Li, Q., Mazurowski, M.A.: Mri-core: a foundation model for magnetic resonance imaging. arXiv preprint arXiv:2506.12186 (2025)

[6] Dong, X., Yu, Z., Cao, W., Shi, Y., Ma, Q.: A survey on ensemble learning. Frontiers of Computer Science 14(2), 241–258 (2020)

[7] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

[8] Eslami, S., Meinel, C., De Melo, G.: Pubmedclip: How much does clip benefit visual question answering in the medical domain? In: Findings of the Association for Computational Linguistics: EACL 2023. pp. 1181–1193 (2023)

[9] Fu, J., Li, H., Lu, T., Zhang, S., Wang, G.: Um-sam: Unsupervised medical image segmentation using knowledge distillation from segment anything model. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 616–626. Springer (2025)

[10] Hamamci, I.E., Er, S., Wang, C., Almas, F., Simsek, A.G., Esirgun, S.N., Dogan, I., Durugol, O.F., Hou, B., Shit, S., et al.: Developing generalist foundation models from a multimodal dataset for 3d computed tomography. arXiv preprint arXiv:2403.17834 (2024)

[11] Heinrich, G., Ranzinger, M., Yin, H., Lu, Y., Kautz, J., Tao, A., Catanzaro, B., Molchanov, P.: Radiov2. 5: Improved baselines for agglomerative vision foundation models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22487–22497 (2025)

[12] Hu, H., Dey, D., Hebert, M., Bagnell, J.A.: Learning anytime predictions in neural networks via adaptive loss balancing. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 3812–3821 (2019)

[13] Ikezogwo, W., Seyfioglu, S., Ghezloo, F., Geva, D., Sheikh Mohammed, F., Anand, P.K., Krishna, R., Shapiro, L.: Quilt-1m: One million image-text pairs for histopathology. Advances in neural information processing systems 36, 37995–38017 (2023)

[14] Ilharco, G., Wortsman, M., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., et al.: Openclip. Zenodo (2021)

[15] Jiang, S., Wang, Y., Song, S., Hu, T., Zhou, C., Pu, B., Zhang, Y., Yang, Z., Feng, Y., Zhou, J.T., et al.: Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668 (2025)

[16] Jiao, J., Zhou, J., Li, X., Xia, M., Huang, Y., Huang, L., Wang, N., Zhang, X., Zhou, S., Wang, Y., et al.: Usfm: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis. Medical image analysis 96, 103202 (2024)

[17] Kang, Q., Lao, Q., Gao, J., Bao, W., He, Z., Du, C., Lu, Q., Li, K.: Urfm: a general ultrasound representation foundation model for advancing ultrasound image diagnosis. IScience 28(8) (2025)

[18] Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7482–7491 (2018)

[19] Khattak, M.U., Kunhimon, S., Naseer, M., Khan, S., Khan, F.S.: Unimed-clip: Towards a unified imagetext pretraining paradigm for diverse medical imaging modalities. arXiv preprint arXiv:2412.10372 (2024)

[20] Kim, C., Gadgil, S.U., DeGrave, A.J., Omiye, J.A., Cai, Z.R., Daneshjou, R., Lee, S.I.: Transparent medical image ai via an image–text foundation model grounded in medical literature. Nature medicine 30(4), 1154–1165 (2024)

[21] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)

[22] Kurtansky, N.R., Rotemberg, V., Gillis, M., Kose, K., Reade, W., Chow, A., et al.: The slice-3d dataset: 400,000 skin lesion image crops extracted from 3d tbp for skin cancer detection. Scientific Data 11(1) (2024). https://doi.org/10.1038/s41597-024-03743-w

[23] Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H., Gao, J.: Llavamed: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems 36, 28541–28564 (2023)

[24] Lin, W., Zhao, Z., Zhang, X., Wu, C., Zhang, Y., Wang, Y., Xie, W.: Pmc-clip: Contrastive languageimage pre-training using biomedical documents. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 525–536. Springer (2023)

[25] Liu, Y., Zhang, W., Wang, J.: Adaptive multi-teacher multi-level knowledge distillation. Neurocomputing 415, 106–113 (2020)

[26] Liu, Z., Wang, Y., Chu, X., Dong, N., Qi, S., Ling, H.: A simple and generic framework for feature distillation via channel-wise transformation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1129–1138 (2023)

[27] Lozano, A., Sun, M.W., Burgess, J., Chen, L., Nirschl, J.J., Gu, J., Lopez, I., Aklilu, J., Rau, A., Katzer, A.W., et al.: Biomedica: An open biomedical image-caption archive, dataset, and vision-language models derived from scientific literature. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19724–19735 (2025)

[28] Lu, M.Y., Chen, B., Williamson, D.F., Chen, R.J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L.P., Gerber, G., et al.: A visual-language foundation model for computational pathology. Nature medicine 30(3), 863–874 (2024)

[29] Luo, X., Wang, X., Eweje, F., Zhang, X., Yang, S., Quinton, R., Xiang, J., Li, Y., Ji, Y., Li, Z., et al.: Ensemble learning of foundation models for precision oncology. arXiv preprint arXiv:2508.16085 (2025)

[30] Ma, J., Guo, Z., Zhou, F., Wang, Y., Xu, Y., Li, J., Yan, F., Cai, Y., Zhu, Z., Jin, C., et al.: A generalizable pathology foundation model using a unified knowledge distillation pretraining framework. Nature Biomedical Engineering pp. 1–20 (2025)

[31] Mei, X., Liu, Z., Robson, P.M., Marinelli, B., Huang, M., Doshi, A., Jacobi, A., Cao, C., Link, K.E., Yang, T., et al.: Radimagenet: an open radiologic deep learning research dataset for effective transfer learning. Radiology: Artificial Intelligence 4(5), e210315 (2022)

[32] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

[33] Pai, S., Hadzic, I., Bontempi, D., Bressem, K., Kann, B.H., Fedorov, A., Mak, R.H., Aerts, H.J.: Vision foundation models for computed tomography. arXiv preprint arXiv:2501.09001 (2025)

[34] Patil, K.D., Palani, G., Krishnamurthi, G.: Efficient knowledge distillation of sam for medical image segmentation. arXiv preprint arXiv:2501.16740 (2025)

[35] Perez-Garc´ ´ıa, F., Sharma, H., Bond-Taylor, S., Bouzid, K., Salvatelli, V., Ilse, M., Bannur, S., Castro, D.C., Schwaighofer, A., Lungren, M.P., et al.: Exploring scalable medical image encoders beyond text supervision. Nature Machine Intelligence 7(1), 119–130 (2025)

[36] Qiu, J., Wu, J., Wei, H., Shi, P., Zhang, M., Sun, Y., Li, L., Liu, H., Liu, H., Hou, S., et al.: Visionfm: a multi-modal multi-task vision foundation model for generalist ophthalmic artificial intelligence. arXiv preprint arXiv:2310.04992 (2023)

[37] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: Internationa conference on machine learning. pp. 8748–8763. PmLR (2021)

[38] Rajendran, P., Safari, M., He, W., Hu, M., Wang, S., Zhou, J., Yang, X.: Foundation models in medical image analysis: A systematic review and meta-analysis. arXiv preprint arXiv:2510.16973 (2025)

[39] Ranzinger, M., Heinrich, G., Kautz, J., Molchanov, P.: Am-radio: Agglomerative vision foundation model reduce all domains into one. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12490–12500 (2024)

[40] Ranzinger, M., Heinrich, G., McCarthy, C., Kautz, J., Tao, A., Catanzaro, B., Molchanov, P.: C-radiov4 (tech report). arXiv preprint arXiv:2601.17237 (2026)

[41] Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L.,¨ et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

[42] Ruan, W., Yang, T., Zhou, Y., Liu, T., Lu, J.: From task-specific models to unified systems: A review of model merging approaches. arXiv preprint arXiv:2503.08998 (2025)

[43] Ruckert, J., Bloch, L., Br¨ ungel, R., Idrissi-Yaghir, A., Sch¨ afer, H., Schmidt, C.S., Koitka, S., Pelka, O.,¨ Abacha, A.B., G. Seco de Herrera, A., et al.: Rocov2: Radiology objects in context version 2, an updated multimodal image dataset. Scientific Data 11(1), 688 (2024)

[44] Shi, D., Zhang, W., Chen, X., Liu, Y., Yang, J., Huang, S., Tham, Y.C., Zheng, Y., He, M.: Eyefound: a multimodal generalist foundation model for ophthalmic imaging. arXiv preprint arXiv:2405.11338 (2024)

[45] Shi, D., Zhang, W., Yang, J., Huang, S., Chen, X., Yusufu, M., Jin, K., Lin, S., Liu, S., Zhang, Q., et al.: Eyeclip: A visual-language foundation model for multi-modal ophthalmic image analysis. arXiv preprint arXiv:2409.06644 (2024)

[46] Simeoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi,´ S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

[47] Wang, S., Jin, Z., Hu, M., Safari, M., Zhao, F., Chang, C.W., Qiu, R.L., Roper, J., Yu, D.S., Yang, X.: Unifying biomedical vision-language expertise: Towards a generalist foundation model via multi-clip knowledge distillation. arXiv preprint arXiv:2506.22567 (2025)

[48] Xu, W., Chan, H.P., Li, L., Aljunied, M., Yuan, R., Wang, J., Xiao, C., Chen, G., Liu, C., Li, Z., et al.: Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. arXiv preprint arXiv:2506.07044 (2025)

[49] Yan, S., Yu, Z., Primiero, C., Vico-Alonso, C., Wang, Z., Yang, L., Tschandl, P., Hu, M., Ju, L., Tan, G., et al.: A multimodal vision foundation model for clinical dermatology. Nature Medicine 31(8), 2691–2702 (2025)

[50] Yang, Y., Su, G., Hu, J., Sammarco, F., Geiping, J., Wolfers, T.: Medsamix: A training-free model merging approach for medical image segmentation. arXiv preprint arXiv:2508.11032 (2025)

[51] Yang, Z., Xu, X., Zhang, J., Wang, G., Kalra, M.K., Yan, P.: Chest x-ray foundation model with global and local representations integration. IEEE transactions on medical imaging (2025)

[52] You, S., Xu, C., Xu, C., Tao, D.: Learning from multiple teacher networks. In: Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. pp. 1285–1294 (2017)

[53] Zhang, H., Wu, Y., Zhao, M., Chen, Z., Li, R., Zhu, F., Zhao, H., Yuan, X., Yang, M., Qiu, C., et al.: A fully open and generalizable foundation model for ultrasound clinical applications. arXiv preprint arXiv:2509.11752 (2025)

[54] Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., Tinn, R., Preston, S., Rao, R., Wei, M., Valluri, N., et al.: Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915 (2023)

[55] Zhou, Y., Chia, M.A., Wagner, S.K., Ayhan, M.S., Williamson, D.J., Struyven, R.R., Liu, T., Xu, M., Lozano, M.G., Woodward-Court, P., et al.: A foundation model for generalizable disease detection from retinal images. Nature 622(7981), 156–163 (2023)

[56] Zhu, X., Hu, Y., Mo, F., Li, M., Wu, J.: Uni-med: a unified medical generalist foundation model for multitask learning via connector-moe. Advances in Neural Information Processing Systems 37, 81225–81256 (2024)

## Supplementary Material

## Pretraining Corpus Reconstruction

Our pretraining corpus follows the modality-decoupled reconstruction principle: we disaggregate UniMed-CLIP’s original 5.3M multi-modal mixture into six independent data streams, enabling modality-specific teacher assignment during distillation. Tab. S1 details the composition. Single-modality streams (US, MRI, CT, Histo., Derm.) are sourced from public domain-specific datasets with expert annotations, while the multimodal mixture stream aggregates cross-domain image-text pairs from PMC-OA [24], ROCOv2 [43], and LLaVA-Med [23]. This reconstruction ensures that specialist teachers supervise only their respective modalities, while the generalist teacher provides cross-modal regularization across all streams.

Table S1: Pretraining stream composition (4.54M pairs). Specialist streams are single-modality; Mix is supervised by the generalist.
<table><tr><td>Stream</td><td>#Pairs</td><td>Sources</td><td>Teacher</td></tr><tr><td>US</td><td>390k</td><td>RadImageNet (US subset) [31]</td><td>URFM [17]</td></tr><tr><td>MRI</td><td>673k</td><td>RadImageNet (MRI subset) [31]</td><td>Curia [4]</td></tr><tr><td>CT</td><td>290k</td><td>RadImageNet (CT subset) [31]</td><td>Curia [4]</td></tr><tr><td>Histo.</td><td>196k</td><td>Quilt-1M [13]</td><td>GPFM [30]</td></tr><tr><td>Derm.</td><td>401k</td><td>ISIC2024 archive [22]</td><td>PanDerm [49]</td></tr><tr><td>Mix</td><td>2.59M</td><td>PMC-OA [24], ROCOv2 [43], LLaVA-Med [23]</td><td>UniMed-CLIP [19]</td></tr><tr><td>Total</td><td>4.54M</td><td></td><td></td></tr></table>

Preprocessing Protocol. All images are center-cropped to 224×224 resolution with ImageNet normalization $( \mu = [ 0 . 4 8 5 , 0 . 4 5 6 , 0 . 4 0 6 ] , \sigma = [ 0 . 2 2 9 , 0 . 2 2 4 , 0 . 2 2 5 ] )$ . Grayscale medical images (ultrasound, MRI, CT) are channel-replicated to RGB format for ViT compatibility. No domain-specific preprocessing (such as DICOM windowing, histogram equalization) is applied to maintain consistency with teacher model pipelines.

## Downstream Benchmarks

Table S2: Internal downstream benchmark statistics.
<table><tr><td>Modality</td><td>Dataset</td><td>#Images</td><td>#Classes</td><td>Metric</td><td>Split</td></tr><tr><td rowspan="2">US</td><td>Thyroid</td><td>349</td><td>2</td><td>AUC</td><td>75:10:15</td></tr><tr><td>Breast</td><td>780</td><td>2</td><td>AUC</td><td>75:10:15</td></tr><tr><td rowspan="2">MRI</td><td>ACL Tear</td><td>1,022</td><td>2</td><td>AUC</td><td>75:10:15</td></tr><tr><td>Meniscus Tear</td><td>4,201</td><td>2</td><td>AUC</td><td>75:10:15</td></tr><tr><td rowspan="3">CT</td><td>MediMeTA Axial</td><td>12,026</td><td>11</td><td>ACC</td><td>Official</td></tr><tr><td>MediMeTA Coronal</td><td>12,026</td><td>11</td><td>ACC</td><td>Official</td></tr><tr><td>MediMeTA Sagittal</td><td>12,026</td><td>11</td><td>ACC</td><td>Official</td></tr><tr><td>Histo.</td><td>PCam</td><td>327,680</td><td>2</td><td>ACC</td><td>Official</td></tr><tr><td>Derm.</td><td>HAM</td><td>10,015</td><td>2</td><td>ACC</td><td>Official</td></tr></table>

We evaluate Med-RADIO on nine internal classification tasks spanning five imaging modalities (Tab. S2). All datasets employ official train/test splits where provided; otherwise we adopt stratified 75:10:15 train/validation/test partitions following established medical imaging protocols [19]. Evaluation metrics follow domain conventions: AUC for ultrasound/MRI; accuracy for CT, histopathology, and dermatology.

## Teacher Model Selection Details

We provide extended details regarding the selection criteria for both generalist and specialist teachers used in our multi-teacher distillation framework. We describe the factors considered in teacher inclusion—such as domain coverage, architectural compatibility, representational diversity, and empirical performance as well as additional ablation settings evaluating alternative teacher combinations.

Table S3: Specialist-teacher candidates under linear probing. Bold: selected modality-specific specialist (UniMed-CLIP excluded from selection).
<table><tr><td rowspan="2">Modality Model</td><td rowspan="2"></td><td colspan="2">Benchmarks</td><td rowspan="2">Avg</td></tr><tr><td>Thyroid (AUC)</td><td>Breast (AUC)</td></tr><tr><td rowspan="5">US</td><td>[USFM [16]</td><td>59.30</td><td>64.73</td><td>62.01</td></tr><tr><td>EchoCare [53]</td><td>70.29</td><td>77.94</td><td>74.12</td></tr><tr><td>URFM [17]</td><td>74.15</td><td>89.32</td><td>81.74</td></tr><tr><td>UniMed-CLIP (base) [19]</td><td>76.96</td><td>79.34</td><td>78.15</td></tr><tr><td>UniMed-CLIP (large) [19]</td><td>79.18</td><td>77.12</td><td>78.15</td></tr><tr><td></td><td></td><td>ACL (AUC)</td><td>Meniscus (AUC)</td><td></td></tr><tr><td rowspan="4">MRI</td><td>[MR-CLIP [1]</td><td>80.48</td><td>87.33</td><td>83.91</td></tr><tr><td>Curia [4]</td><td>96.05</td><td>92.69</td><td>94.37</td></tr><tr><td>UniMed-CLIP (base) [19]</td><td>97.30</td><td>95.08</td><td>96.19</td></tr><tr><td>UniMed-CLIP (large) [19]</td><td>97.05</td><td>97.53</td><td>97.29</td></tr><tr><td></td><td></td><td>Ax. (ACC) Cor. (ACC) Sag. (ACC)</td><td></td><td></td></tr><tr><td rowspan="3">CT</td><td>Curia [4]</td><td>76.70</td><td>72.49 63.43</td><td>70.87</td></tr><tr><td>UniMed-CLIP (base) [19]</td><td>67.80</td><td>59.22 62.78</td><td>63.27</td></tr><tr><td>UniMed-CLIP (large) [19]</td><td>76.38</td><td>65.70 62.78</td><td>68.29</td></tr><tr><td></td><td></td><td>PCam (ACC)</td><td></td><td></td></tr><tr><td rowspan="3">Histo.</td><td>[GPFM [30]</td><td>94.09</td><td></td><td>94.09</td></tr><tr><td>UniMed-CLIP (base) [19]</td><td></td><td>85.41</td><td>85.41</td></tr><tr><td>UniMed-CLIP (large) [19]</td><td></td><td>90.23</td><td>90.23</td></tr><tr><td></td><td></td><td>HAM (ACC)</td><td></td><td></td></tr><tr><td rowspan="3">Derm.</td><td>[PanDerm [49]</td><td>89.19</td><td></td><td>89.19</td></tr><tr><td>UniMed-CLIP (base) [19]</td><td>86.26</td><td></td><td>86.26</td></tr><tr><td>UniMed-CLIP (large) [19]</td><td>91.15</td><td></td><td>91.15</td></tr></table>

Table S3 summarizes the results, with bold entries marking the selected modality-specific specialist (UniMed-CLIP rows are reported for reference but excluded from specialist selection). URFM ranks first among ultrasound specialists; Curia is the strongest dedicated specialist on MRI and CT; GPFM dominates histopathology; and PanDerm is selected for dermatology. Together with UniMed-CLIP (large) as the mandatory generalist anchor, these choices form the final teacher ensemble: UniMed-CLIP (large), URFM, Curia, GPFM, and Pan-Derm.

## Teacher Feature Statistics Computation

The statistics expose two opposing failure modes. On one end, URFM (ultrasound, Disp=0.003) and PanDerm (dermatology, 0.127) encode highly concentrated feature spaces: their raw angular losses are negligibly small relative to other streams and would be entirely suppressed in unweighted multi-teacher distillation. The dispersion denominator amplifies these signals by factors of up to 333×, ensuring the student receives meaningful ultrasound and dermatology gradients. On the other end, Curia on CT (1.264) and GPFM on histopathology (0.807) produce large raw angular losses due to high feature spread, which would otherwise dominate the joint objective and force all other teachers’ signals to zero. Dividing by these large dispersion values proportionally attenuates their contribution, preventing CT-driven optimization collapse. This dual correction is empirically validated by the balanced KD loss ablation (Fig. 5(c)): without it, CT Coronal ACC drops by $- 1 5 . 0 5 \mathrm { p p }$ , and ultrasound Thyroid AUC collapses by −14.39pp. This pattern aligns with Curia CT dominance while URFM is suppressed.

Table S4: Offline angular dispersion Disp(Θ) on 10,000 samples per native stream (Eq. 4). –: not applicable.
<table><tr><td>Modality</td><td>UniMed-CLIP</td><td>URFM</td><td>Curia</td><td>GPFM</td><td>PanDerm</td></tr><tr><td>Mixed</td><td>0.673</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Ultrasound</td><td>0.136</td><td>0.003</td><td>一</td><td>一</td><td>一</td></tr><tr><td>MRI</td><td>0.209</td><td>一</td><td></td><td>一</td><td>一</td></tr><tr><td>CT</td><td>0.218</td><td>一</td><td>1.264</td><td>一</td><td>一</td></tr><tr><td>Histopathology</td><td>1.057</td><td>一</td><td>一</td><td>0.807</td><td></td></tr><tr><td>Dermatology</td><td>0.173</td><td>一</td><td>一</td><td>一</td><td>0.127</td></tr></table>

## Hyperparameters and Training Pseudocode

We provide comprehensive implementation specifications to ensure full reproducibility of Med-RADIO. Tab. S5 consolidates all hyperparameter settings across three training stages: pretraining distillation, linear probing evaluation, and full fine-tuning adaptation. Algorithm 1 formalizes the multi-stream distillation procedure with step-by-step annotations, explicitly detailing the proportional batch assembly strategy, balanced angular loss computation, and gradient optimization workflow. Together, these specifications enable exact replication of our training pipeline and facilitate extension to alternative medical imaging domains or teacher ensemble configurations.

Table S5: Hyperparameter settings for distillation, linear probing, and full fine-tuning.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td colspan="2">Pretraining (Student Distillation)</td></tr><tr><td>Architecture</td><td>ViT-L/16 (304M params)</td></tr><tr><td>Input resolution Global batch size</td><td>224×224</td></tr><tr><td>Optimizer</td><td>192 (32 per teacher stream; 6×A100 80GB, Accelerate fp16) AdamW (β1=0.9, β2=0.999, WD= 0.05)</td></tr><tr><td>Learning rate schedule</td><td>Cosine (peak  $1 \times 1 0 ^ { - 4 }$  , warmup 1 epoch)</td></tr><tr><td>Gradient clipping</td><td>Max norm 1.0</td></tr><tr><td>Total epochs</td><td>10</td></tr><tr><td>Translator head</td><td>MLP + LayerNorm (1024-d CLS → dT)</td></tr><tr><td>Dispersion estimation</td><td>10,000 samples per teacher</td></tr><tr><td colspan="2">Downstream Linear Probing</td></tr><tr><td>Frozen encoder</td><td>ViT-L/16 (all layers frozen)</td></tr><tr><td>Classification head</td><td>Linear (CLS → #classes)</td></tr><tr><td>Batch size / LR / Epochs</td><td>256 / 1× 10-3 / 50</td></tr><tr><td>Optimizer</td><td>AdamW (cosine schedule)</td></tr><tr><td colspan="2">Downstream Full Fine-tuning</td></tr><tr><td>End-to-end training</td><td>Encoder + task head jointly optimized</td></tr><tr><td>Batch size / LR / Epochs</td><td>64 / 5×10−⁵ / 50</td></tr><tr><td>Optimizer</td><td>AdamW (cosine schedule)</td></tr></table>

Algorithm 1 Med-RADIO Multi-Teacher Distillation with Balanced Loss   
1: Input: Student encoder $f _ { \boldsymbol { \theta } } ;$ frozen teachers $\{ T _ { m } \} _ { m = 1 } ^ { 6 }$ ; translator heads $\textstyle \{ \phi _ { m } \} _ { m = 1 } ^ { 6 } ;$ angular dispersions $\{ { \mathrm { D i s p } } ( \Theta _ { m } ) \} _ { m = 1 } ^ { 6 } ;$   
data streams $\{ \mathcal { D } _ { m } \} _ { m = 1 } ^ { 6 }$   
2: Initialize: $f _ { \theta }$ from ImageNet-21k ViT-L/16; $\left\{ \phi _ { m } \right\}$ via Xavier uniform   
3: for epoch $e = 1$ to 10 do   
4: Set learning rate $\eta _ { e }$ via cosine schedule (peak $1 \times 1 0 ^ { - 4 } .$ , warmup 1 epoch)   
5: Shuffle each stream ${ \mathcal { D } } _ { m }$ with seed $e$   
6: for iteration t do   
7: // — Step 1: Assemble multi-stream batch —   
8: for each stream $m \in \{ 1 , \ldots , 6 \}$ do   
9: Compute proportional batch size: $B _ { m } \gets \lfloor ( | \mathcal { D } _ { m } | / \sum _ { m ^ { \prime } } | \mathcal { D } _ { m ^ { \prime } } | ) \cdot 1 9 2 \rfloor$   
10: Sample mini-batch $\mathcal { B } _ { m } \sim \mathcal { D } _ { m }$ of size $B _ { m }$   
11: end for   
12: Assign residual samples: $\mathcal { B } _ { \mathrm { m i x } }  \mathcal { B } _ { \mathrm { m i x } }$ ∪ Sample $\begin{array} { r l } {  { \bigl ( 1 9 2 - \sum _ { m } B _ { m } \bigr ) } \qquad } & { { } } \end{array}$   
13: Construct global batch: $\mathcal { B }  \bigcup _ { m = 1 } ^ { 6 } \mathcal { B } _ { m }$ (total 192 images)   
14: $/ / - \operatorname { S t e p } 2 \colon$ Forward pass through student and teachers   
15: $\mathbf { z } _ { s } \gets f _ { \theta } ( \mathcal { B } )$ // Student CLS tokens [192, 1024]   
16: for each stream m do   
17: Extract modality-specific tokens: $\mathbf { z } _ { s } ^ { ( m ) }  \mathbf { z } _ { s } [ \mathcal { B } _ { m } ] \quad / / [ \mathrm { B \_ m } ]$ , 1024]   
18: Compute teacher features: $\mathbf { z } _ { T _ { m } } \gets T _ { m } ( \mathcal { B } _ { m } )$ // Frozen forward pass   
19: end for   
20: $/ / - \operatorname { S t e p } 3 \colon$ Translate and compute balanced angular loss —   
21: Initialize $\mathcal { L } _ { \mathrm { t o t a l } }  0$   
22: for each stream m do   
23: Project to teacher space: $\hat { \mathbf { z } } _ { s } ^ { ( m ) } \gets \phi _ { m } ( \mathbf { z } _ { s } ^ { ( m ) } ) \quad / / \left[ \mathrm { B } _ { - } \mathrm { m } , \mathrm { d } _ { - } \mathrm { T } _ { - } \mathrm { m } \right]$   
24: Normalize: $\hat { \mathbf { z } } _ { s } ^ { ( m ) } \gets \hat { \mathbf { z } } _ { s } ^ { ( m ) } / \| \hat { \mathbf { z } } _ { s } ^ { ( m ) } \| _ { 2 } , \mathbf { z } _ { T _ { m } } \gets \mathbf { z } _ { T _ { m } } / \| \mathbf { z } _ { T _ { m } } \| _ { 2 }$   
25: Compute pairwise angular distance: $\Theta ^ { ( m ) } \gets \operatorname { a r c c o s } ( \hat { \mathbf { z } } _ { s } ^ { ( m ) } \cdot \mathbf { z } _ { T _ { m } } ^ { \top } )$ // [B m, B m]   
26: Aggregate: $\begin{array} { r } { \mathcal { L } _ { m } \gets \frac { 1 } { B _ { m } ^ { 2 } } \sum _ { i , j } \frac { ( \Theta _ { i , j } ^ { ( m ) } ) ^ { 2 } } { \mathrm { D i s p } ( \Theta _ { m } ) } } \end{array}$ $/ / \operatorname { E q . 5 }$   
27: $\mathcal { L } _ { \mathrm { t o t a l } }  \mathcal { L } _ { \mathrm { t o t a l } } + \mathcal { L } _ { m } ^ { ' \prime }$   
28: end for   
29: $/ / - \operatorname { S t e p } 4 { \mathrm { : } }$ Backward pass and optimization —   
30: Compute gradients: $\nabla _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } } , \left\{ \nabla _ { \phi _ { m } } \mathcal { L } _ { \mathrm { t o t a l } } \right\}$   
31: Clip gradients: clip $( \nabla _ { \theta }$ ,max norm $= 1 . 0 )$   
32: Update parameters: $\theta \gets \theta - \eta _ { e } \nabla _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } } ; \phi _ { m } \gets \phi _ { m } - \eta _ { e } \nabla _ { \phi _ { m } } \mathcal { L } _ { \mathrm { t o t a l } }$ for all m   
33: end for   
34: end for   
35: Output: Trained student encoder $f _ { \theta }$ (discard translator heads $\{ \phi _ { m } \} )$