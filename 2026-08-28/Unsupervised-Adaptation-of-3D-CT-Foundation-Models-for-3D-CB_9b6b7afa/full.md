# Unsupervised Adaptation of 3D CT Foundation Models for 3D CBCT Segmentation

Gauthier Miralles<sup>1,2</sup>, Loic Le Folgoc<sup>1</sup>, Vincent Jugnon<sup>2</sup>, and Pietro Gori<sup>1</sup>

<sup>1</sup> LTCI, Télécom Paris, Institut Polytechnique de Paris, Palaiseau, France <sup>2</sup> GE HealthCare, Buc, France gauthier.miralles@ip-paris.fr

Abstract. Accurate 3D segmentation of cone-beam CT (CBCT) is critical for interventional and radiation therapy applications, yet it remains limited by two compounding challenges: the scarcity of annotated CBCT data and the large domain shift from diagnostic CT. Interventional CBCT exhibits fundamental modality diferences from conventional CT, driven by acquisition and physics efects as well as contrast-specific vascular content, thereby limiting efective cross-modality model transfer. We propose a novel unsupervised domain adaptation (UDA) framework based on redundancy-reducing feature alignment, enabling 3D CBCT segmentation with no target-domain annotations or inference-time adaptation. Our framework is architecture-agnostic, seamlessly adapting both CNNbased and ViT-based foundation models (FMs). We evaluate our method on two challenging CT–CBCT liver segmentation benchmarks: one for interventional vascular procedures and one for radiation therapy, demonstrating that even large-scale pretrained segmentation networks require explicit feature-space bridging to generalize across acquisition modalities, and that our approach consistently outperforms existing pretrained FM and UDA strategies. To support reproducibility and benchmarking, we release the liver segmentations for a public CBCT dataset, along with the code, trained models, and weights at <sup>3</sup>.

Keywords: Foundation Models · Unsupervised Domain Adaptation · CBCT · Interventional Imaging · 3D · Liver.

## 1 Introduction

Deep learning has become the standard for medical image segmentation, but it relies on large annotated datasets and assumes that training and deployment data follow the same distribution. When this assumption is violated, domain shift can severely degrade performance and limit clinical applicability. This issue is particularly pronounced for Cone-Beam CT (CBCT). Although CBCT resembles conventional CT, it difers substantially due to its limited reconstructed field of view, acquisition-geometry artifacts, physics-related degradations such as scatter and beam hardening, and locally injected iodine contrast producing high-intensity regions in interventional settings (Figure 3). The domain gap between CT and CBCT restricts the transferability of pre-trained 3D CT foundation models. Preprocessing pipelines designed for full-field CT often fail on CBCT, leading to unreliable zero-shot segmentation. Moreover, generative domain adaptation methods based on, for instance, CycleGAN [1,17,18] are not directly applicable, as they typically assume similar fields of view between source and target domains.

Existing 3D foundation models can be categorized by their level of automation. Fully automatic models include CNN-based approaches such as Merlin [9] and ViT-based models such as MA-SAM [8], whereas TotalSegmentator [12], despite being automatic, comprises multiple task-specific nnU-Net [2] models rather than a unified foundation model. Prompt-based models, inspired by Segment Anything [3] and predominantly ViT-based, include SAM-Med3D [11] and MedSAM2 [10]. Hybrid models, such as Vista-3D, support both automatic inference and prompt-guided refinement [13].

Given that CT is the closest available modality to CBCT, it serves as a natural source domain for adaptation. However, acquiring CBCT annotations is extremely time-consuming for segmentation, limiting the feasibility of supervised fine-tuning. We thus focus on unsupervised domain adaptation (UDA), where only labeled CT data are available, as a natural framework for CT-to-CBCT segmentation. Restricting our comparison to methods operating natively in 3D, existing UDA approaches fall into three categories. Feature-alignment methods align source and target distributions in a shared latent space, either via adversarial discrimination, as in DA-nnUNet [4], or through margin disparity discrepancy, as in MDD-UNet [5], though the latter remains restricted to a specific segmentation task. Joint feature- and image-alignment methods, such as SIFA-3D [6], operate both in pixel and feature spaces, but require multiple GAN components and assume comparable fields of view between source and target, an unrealistic assumption for CT and CBCT. Self-training methods instead generate pseudo-labels in the target domain to supervise adaptation. As an example, MAPSeg [7] first leverages masked autoencoder pre-training to obtain a strong generalizable encoder, then applies pseudo-labelling to learn the segmentation task while preserving generalization ability.

In this work, we make minimal assumptions: no prior knowledge of field of view, no specialized initialization, and a lightweight design suitable for 3D volumes. To this end, we adopt a feature-level alignment strategy that works with any architecture, both CNN/ViT custom networks and foundation models.

Contributions. Our contributions are: (i) a 3D feature-level adaptation framework for adapting foundation model representations to CT-to-CBCT segmentation, applicable to CNN and ViT architectures; (ii) an evaluation on two complementary CT–CBCT datasets, covering both interventional and radiation therapy CBCT, demonstrating consistent gains over existing UDA methods; and (iii) the release of manually curated liver segmentations for the publicly available Pancreatic 3D CBCT dataset [14], to facilitate future research and fair comparison.

![](images/30ad9b9c662524d9eb4977fbc8622c5176fdba68b7be61a8a498fcdb6c751d70.jpg)  
Fig. 1: Overview of the proposed UDA framework. Independently of the underlying backbone architecture, which can be either CNN- or ViT-based, the original network is decomposed into a shared feature extractor ψ, a representation head $f ,$ , its adversarial counterpart $f ^ { \prime }$ (i.e., a copy of $f$ at initialization), and a taskspecific prediction head $g .$ Given source-domain inputs $x ^ { S }$ (3D CT patches) and target-domain inputs $x ^ { \check { T } }$ (3D CBCT patches), the extracted features $z = \psi ( x )$ are encouraged to become domain-invariant through a redundancy-reduction adversarial strategy acting between $f$ and $f ^ { \prime } .$ . The prediction head $g$ is kept separate to allow adaptation to the target task, while $f ^ { \prime }$ is discarded at inference time.

## 2 Method

Driven by the previous discussion, we aim to design a method that is generic, making no prior assumption on the type of domain shift, flexible, imposing no constraints on network architecture, and lightweight, allowing eficient training on 3D volumes with minimal computational resources.

We consider a source domain $\mathcal { D } _ { S } = ( \mathcal { X } ^ { S } , \mathcal { Y } , p ^ { S } ( x , y ) )$ and a target domain $\mathcal { D } _ { T } = ( \mathcal { X } ^ { T } , \mathcal { Y } , p ^ { T } ( x , y ) )$ with some domain shift captured by difering distributions $p ^ { S }$ and $p ^ { T }$ . We are given labeled samples $( \bar { x } _ { i } ^ { S } , y _ { i } ^ { S } ) _ { i = 1 } ^ { N _ { S } } \sim p ^ { S } ( x , y )$ in the source domain, and unlabeled samples $( x _ { i } ^ { T } ) _ { i = 1 } ^ { N _ { T } } \sim p ^ { T } ( x )$ in the target domain. The goal is to learn a map $h : \mathcal { X } ^ { S } \cup \mathcal { X } ^ { T }  \overset { } { \mathcal { Y } }$ that produces a consistent labeling $h ( x )$ for both source and target domain samples $x \in \mathcal { X } ^ { S } \cup \mathcal { X } ^ { T }$ . We proceed via a feature-alignment strategy, whereby (a) learned representations of source and target domain samples are driven to the same marginal distribution at convergence, and (b) the map h is made informative for the labeling task through explicit label supervision in the source domain. Next, we describe an adversarial strategy that leads to these two properties both theoretically and empirically. To derive the proposed approach, we split the predictor $h ( x ) \triangleq g ( f ( \psi ( x ) ) )$ into a feature encoder $\psi ( \cdot )$ , a representation head $f ( \cdot )$ and a task-specific prediction head $g ( \cdot )$ . The resulting diagram in Figure 1 illustrates the workflow of the framework, showing how features are extracted, aligned, and made domain-invariant, while the task head $g$ remains separate for adaptation to the target task.

Source domain label supervision. The representation head $f ( \cdot )$ and the prediction head $g ( \cdot )$ are trained to optimally solve the labeling task in the source domain via a task-specific loss $\mathcal { L } _ { \mathrm { t a s k } }$ , as shown for one sample in $\operatorname { E q . } ( 1 )$ :

$$
\underset { f , g } { \arg \operatorname* { m i n } } \ \mathcal { L } _ { \mathrm { t a s k } } ( h ( x ^ { S } ) , y ^ { S } )\tag{1}
$$

Adversarial feature alignment. The features $z \triangleq \psi ( x )$ are encouraged to become invariant to the domains, by introducing an adversarial strategy between $\psi ( \cdot )$ and an adversary $f ^ { \prime } ( \cdot )$ . The adversary $f ^ { \prime } ( \cdot )$ is trained to produce representations as dissimilar as possible from $f ( \cdot )$ when using target samples $x ^ { T }$ , and at the same time it should result in representations as similar as possible to $f ( \cdot )$ when using source samples $x ^ { S }$ . This is achieved in $\operatorname { E q . } ( 2 )$ via a separation loss $\mathcal { L } _ { \mathrm { s e p } }$ and an alignment loss $\mathcal { L } _ { \mathrm { a l i g n } } .$ , respectively, described in detail below. On the other hand, the feature extractor $\psi ( \cdot )$ is trained with Eq.(3) to output features $z \triangleq \psi ( x )$ that (1) are relevant for the labeling task; and (2) that make the representations of the adversary $f ^ { \prime } ( z )$ and of the representation head $f ( z )$ undistinguishable both in source and target domains:

$$
\underset { f ^ { \prime } } { \arg \operatorname* { m i n } } \ : \mathcal { L } _ { \mathrm { a l i g n } } \left( f ( z ^ { S } ) , f ^ { \prime } ( z ^ { S } ) \right) + \gamma \mathcal { L } _ { \mathrm { s e p } } \left( f ( z ^ { T } ) , f ^ { \prime } ( z ^ { T } ) \right)\tag{2}
$$

$$
\arg \operatorname* { m i n } _ { \psi } \mathcal { L } _ { \mathrm { t a s k } } ( h ( x ^ { S } ) , y ^ { S } ) + \alpha \mathcal { L } _ { \mathrm { a l i g n } } \left( f ( z ^ { S } ) , f ^ { \prime } ( z ^ { S } ) \right) + \gamma \mathcal { L } _ { \mathrm { a l i g n } } \left( f ( z ^ { T } ) , f ^ { \prime } ( z ^ { T } ) \right)\tag{3}
$$

where $( \alpha , \gamma ) \in \mathbb { R } ^ { + } \times \mathbb { R } ^ { + }$ are user-defined hyperparameters.

Algorithm. In practice, we alternate between single optimization steps of (1), (2) and (3) using a stochastic gradient descent on source and target batches.

Redundancy-reduction based representation alignment. The losses $\mathcal { L } _ { \mathrm { a l i g n } }$ and $\mathcal { L } _ { \mathrm { s e p } }$ should encourage alignment and separation of the latent representations $f ( \tilde { z } ) , f ^ { \prime } ( z )$ , respectively. To this end, we revisit Barlow Twins’ redundancy reduction mechanism [16] and encourage the correlation matrix of representations $f ( z ) , f ^ { \prime } ( z )$ to have specific structure conducive of these properties. Consider the representations $\overline { { f } } ( z ) \in \mathbb { R } ^ { B \times D \times D _ { s } \times H \times W }$ obtained from a batch. Let $\boldsymbol { \phi } _ { f } ( \boldsymbol { z } ) \in \mathbb { R } ^ { \bar { \boldsymbol { D } } \times \boldsymbol { N } }$ be the reshaped, flattened, centered and $L _ { 2 ^ { - } }$ normalized (along $B , D _ { s } , H , W )$ version of $f ( z )$ , and similarly $\phi _ { f ^ { \prime } } ( z )$ for $f ^ { \prime } ( z )$ . Following [16], we note $\mathcal { C } [ i , j ] \triangleq \langle \phi _ { f } ( z ) _ { i } , \phi _ { f ^ { \prime } } ( z ) _ { j } \rangle$ the cross-correlation of features $i , j \in \{ 1 , \cdots , D \}$ for representations $f ( z ) , f ^ { \prime } ( z )$ . Let $p ( z ) \triangleq \psi _ { \sharp } p ^ { S } ( z )$ and $q ( z ) \triangleq \psi _ { \sharp } p ^ { T } ( z )$ denote the pushforwards of $p ^ { S }$ and $p ^ { T }$ through ψ, respectively.

Alignment loss. We define $\mathcal { L } _ { \mathrm { a l i g n } } \left( f ( z ) , f ^ { \prime } ( z ) \right)$ in $\operatorname { E q . } ( 4 )$ as to encourage perfect correlation between the $f ( \cdot )$ and the adversary $f ^ { \prime } ( \cdot )$ along homologous feature dimensions, and to discourage redundancy across diferent feature dimensions:

$$
\mathcal { L } _ { \mathrm { a l i g n } } \left( f ( z ) , f ^ { \prime } ( z ) \right) \triangleq \sum _ { i = 1 } ^ { D } ( 1 - \mathcal { C } [ i , i ] ) ^ { 2 } + \frac { 1 } { D } \sum _ { i \neq j } \mathcal { C } [ i , j ] ^ { 2 }\tag{4}
$$

Separation loss. Diferently, we define $\mathcal { L } _ { \mathrm { s e p } } \left( f ( z ) , f ^ { \prime } ( z ) \right)$ in $\operatorname { E q . } ( 5 )$ to encourage perfect decorrelation between $f ( \cdot )$ and $f ^ { \prime } ( \cdot )$ across all feature dimensions:

$$
\mathcal { L } _ { \mathrm { s e p } } \left( f ( z ) , f ^ { \prime } ( z ) \right) \triangleq \sum _ { i = 1 } ^ { D } \mathcal { C } [ i , i ] ^ { 2 } + \frac { 1 } { D } \sum _ { i \neq j } \mathcal { C } [ i , j ] ^ { 2 }\tag{5}
$$

$\mathcal { L } _ { \mathrm { a l i g n } }$ and $\mathcal { L } _ { \mathrm { s e p } }$ are task-agnostic: they encourage representations to match or difer without being specific to a task $( \ i . e . , \ g ( \cdot ) )$ ). This contrasts with previous works, such as MDD [5], where specific losses, such as cross-entropy for classification, were used. Furthermore, by using the proposed redundancy-reduction losses, we do not need an adversary task-specific prediction head $g ^ { \prime } ( \cdot )$ , as in MDD, since we move the feature alignment from the space of $g$ to the one of $f .$ Theoretical guarantee. We rewrite the adversarial loss in terms of the correlation matrix $w ( z ) _ { i , j } = \langle \phi _ { f } ( z ) _ { i } , \phi _ { f ^ { \prime } } ( z ) _ { j } \rangle$ . Since the resulting relaxed objective is separable and quadratic in each $w ( z ) _ { i , j }$ , its minimizer is:

$$
w ( z ) _ { i , j } = { \left\{ \begin{array} { l l } { \displaystyle { \frac { p ( z ) } { p ( z ) + \gamma q ( z ) } } } & { { \mathrm { i f ~ } } i = j , } \\ { 0 } & { { \mathrm { i f ~ } } i \neq j . } \end{array} \right. }\tag{6}
$$

Since this solution may not be realizable by any $f ^ { \prime } ,$ we introduce a capacity assumption, similarly to MDD, which relies on suficiently expressive classifiers to estimate divergence and ensure theoretical feature alignment.

Assumption 1 For any $\psi ( \cdot ) , f ( \cdot )$ , there exists $f ^ { \prime } ( \cdot )$ such that its feature maps $\phi _ { f ^ { \prime } } ( z ) .$ satisfy the constraints in (6) for all $i , j \in \{ 1 , \cdots , D \}$

Lemma 1. Under Assumption 1, the adversary $f ^ { \prime } ( \cdot )$ that satisfies (6) is a global minimum of the minimization problem of (2).

Theorem 1. Minimizers ψ(·) of the minimization problem of (1), 2, 3 align the marginal distributions of features in source and target domains $i . e . , p ( z ) = q ( z )$ at optimum.

Proof sketch. Using Lemma 1, we substitute the optimal $f ^ { \prime }$ into the ψ-objective and minimize the resulting Lagrangian with respect to ψ (through the induced density $q = \psi _ { \sharp } p ^ { T } )$ ), under the normalization constraint $\textstyle \int q ( \tilde { z } ) d z = 1$ . First-order optimality conditions then imply $p ( z ) = q ( z )$ at optimum.

## 3 Experiments and Results

We evaluated our method on 3D liver segmentation using two CT–CBCT datasets. The first is an unpaired CT–CBCT dataset constructed from public sources: Pancreatic-CT-CBCT-SEG [14] (40 paired scans) and LiTS [15] (130 CT scans with liver annotations). Since Pancreatic-CT-CBCT-SEG lacks liver masks, we generated and manually curated annotations on the CT volumes and transferred them to CBCT via the registered scans. The final dataset contains 130 CT (LiTS) and 39 CBCT (DR) volumes and will be publicly released for reproducibility. The second dataset is a private clinical collection of 678 CT and 573 interventional CBCT (DI) volumes with manually curated liver segmentations. For both datasets, splits were performed at the patient level, with 2/3 of the patients used for training and validation and the remaining 1/3 reserved for testing.

Experimental details. All CT and CBCT volumes are resampled to isotropic 1.8mm voxel spacing. All UDA methods share the same 3D U-Net backbone trained with standard augmentations. CNN- and ViT-based foundation models are evaluated with their oficial implementations. Prompted variants use 1 or 10 randomly sampled liver points. For nnUNet and VISTA-3D, the representation head $f$ corresponds to the penultimate stage, and for SAM-Med3D to the final 3D attention block. In both cases, $f ^ { \prime }$ is initialized as a copy of $f ,$ ψ comprises all preceding stages, and $g$ all subsequent ones (Figure 1). Full implementation details are available in our code repository.

Baselines. We compare our method against state-of-the-art UDA approaches for 3D CBCT segmentation without target-domain annotations. These include zero-shot inference using pretrained foundation models on CBCT data, both in automatic and prompt-based mode [9,12,8,13,11,10]. We also consider 3D unsupervised domain adaptation strategies that exploit CT data to bridge the CT–CBCT domain gap through feature alignment [4,5], image translation [6], or self-training [7]. All methods are evaluated under identical training and evaluation protocols to ensure a fair comparison.

![](images/45e5a98cd16924ac41e3a67bf8d7b7e363d13e56ea05524678270dc5ca2dd119.jpg)

![](images/8307e89095f51d65a6e9ab9ef009d5c4fa41ba1e1f2b50b1de0fca0e09f87667.jpg)  
Fig. 2: Sensitivity to hyperparameters α and γ on dataset DR. Each point corresponds to an independent run (colored by F1, %). The polygon shows points with $\mathrm { F 1 \geq 8 8 . 5 \% }$ . The star (⋆) marks the best result $( \mathrm { F 1 } = 9 0 . 0 \% )$ .  
Fig. 3: Qualitative comparison of CT and CBCT. Axial slices of the same patient displayed with identical visualization windows. The liver contour (blue) is overlaid on CT (left) and CBCT (right).

Unsupervised Domain Adaptation. Table 1 benchmarks recent 3D UDA methods using F1 score across two CT–CBCT datasets. Image-alignment approaches such as SIFA-3D underperform due to the large field-of-view discrepancy between domains, while $\mathrm { M A P S e g }$ fails to surpass the source-only baseline, likely because its masked autoencoder pre-training combined with pseudolabeling is sensitive to poor initialization, causing error propagation during selftraining. Feature-alignment strategies prove more efective and lightweight overall, with DA-nnUNet consistently outperforming MDD-UNet on both datasets. Our method achieves the best results across all benchmarks while remaining lightweight, making it well-suited for integration with 3D foundation models. Furthermore, an ablation study on the sensitivity to hyperparameters α and γ (Figure 2) demonstrates that performance remains stable over a wide range of values, with a large contiguous region achieving near-optimal F1 scores. These results indicate that our approach is robust to hyperparameter selection.

Table 1: F1 performance evaluation of UDA strategies and zero-shot foundation models for CBCT liver segmentation on two datasets. Source: CT, Target: radiation therapy CBCT (DR) or interventional CBCT (DI). Best results (per category) in bold. $\Sigma _ { p }$ (M) denotes the total number of additional trainable parameters introduced by the UDA method (in millions).
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td rowspan="2">Year Mode</td><td colspan="2">F1 ↑</td><td rowspan="2"> $\Sigma _ { p } ( \mathrm { M } ) \downarrow$ </td></tr><tr><td>DR</td><td>DI</td></tr><tr><td>Source Only</td><td>nnUNet [2]</td><td></td><td>66.5</td><td>|80.1</td><td></td></tr><tr><td rowspan="5">Domain Adaptation</td><td>DA-nnUNet [4]</td><td>2024 Auto</td><td>73.3</td><td>84.6</td><td>6.62</td></tr><tr><td>Unsupervised MDD-UNet [5]</td><td>2023 Auto</td><td>66.7</td><td>80.6</td><td>0.09</td></tr><tr><td>SIFA-3D [6]</td><td>2023 Auto</td><td>55.2</td><td>64.7</td><td>37.69</td></tr><tr><td>MAPSeg [7]</td><td>2024 Auto</td><td>59.9</td><td>70.2</td><td>210.96</td></tr><tr><td>nnUNet + Ours</td><td>2026 Auto</td><td>78.0</td><td>86.0</td><td>0.08</td></tr><tr><td rowspan="8">Foundation Models</td><td>MA-SAM [8]</td><td>2024 Auto</td><td>15.4</td><td>|61.8</td><td></td></tr><tr><td>Merlin [9]</td><td>2025 Auto</td><td>18.8</td><td>33.7</td><td></td></tr><tr><td>TotalSegmentator [12] 2023</td><td>Auto</td><td>64.1</td><td>73.8</td><td></td></tr><tr><td>MedSAM2 [10]</td><td>1 pt 2025</td><td>40.4 10 pts 69.7</td><td>30.7 64.6</td><td></td></tr><tr><td>SAM-Med3D [11]</td><td>2023</td><td>1 pt 44.2</td><td>53.6 65.3</td><td></td></tr><tr><td>SAM-Med3D + Ours</td><td></td><td>10 pts 73.8 10 pts 77.6</td><td>74.0</td><td>11.37</td></tr><tr><td></td><td></td><td>Auto 42.7</td><td>48.3</td><td></td></tr><tr><td>VISTA-3D [13]</td><td>2025 1 pt 58.4</td><td></td><td>48.9</td><td></td></tr><tr><td></td><td>VISTA-3D + Ours</td><td></td><td>10 pts 80.0 10 pts 90.0</td><td>74.9 81.5</td><td>0.99</td></tr><tr><td>Target Only</td><td>nnUNet [2]</td><td></td><td>91.4|</td><td>|93.7</td><td></td></tr></table>

Foundation Models. In contrast to UDA methods, several foundation models, including VISTA-3D (interactive branch) and MedSAM2, perform worse on the DI dataset than on DR. This likely stems from iodine-induced high intensities, which fall outside the range expected by the models’ default preprocessing, such as min–max normalization or CT-specific intensity scaling. As a result, these models operate on degraded inputs, whereas nnUNet benefits from percentile normalization (0.5th–99.5th), which clips such intensity outliers. Overall, most 3D foundation models fail to achieve satisfactory zero-shot liver segmentation on CBCT data in automatic mode. When using point prompts (10 points), SAM-Med3D and VISTA-3D provide the strongest initializations. Our method enables the adaptation of both ViT-based (SAM-Med3D) and CNN-based (VISTA-3D) architectures, leading to consistent and substantial performance gains without requiring any labeled target-domain data. Qualitative results on both datasets (Figure 4) indicate that MedSAM2 tends to over-segment and exhibits reduced sensitivity to high-intensity structures compared to VISTA-3D. After adaptation with our UDA framework, the source-only model becomes more robust to highintensity artifacts (Figure 4, row 2) and field-of-view variations (Figure 4, row 3), leading to more anatomically consistent liver segmentations.

![](images/1627cf507ef2647413f940ab2493a60af70d2c5940feb77bce7c8acf899a05b5.jpg)  
Fig. 4: Qualitative comparison of liver segmentation on the DI (top two rows) and DR (bottom row) test sets. Left to right: input image, ground truth (GT), source-only model, MedSAM2 (zero-shot, 10 prompts), VISTA-3D (zero-shot, 10 prompts), and our UDA-adapted model. Predicted masks are shown in blue.

## 4 Conclusion

We investigate unsupervised adaptation of 3D CT foundation models for 3D CBCT segmentation under severe cross-modality domain shift. While 3D foundation models can provide strong structural priors, direct transfer to CBCT is insufficient without explicit feature-space adaptation. By introducing a redundancyreducing alignment strategy, pretrained models can be efectively bridged across modalities, improving performance on both interventional vascular and radiation therapy CT–CBCT benchmarks. Our framework is architecture-agnostic, supports both CNN- and ViT-based models, and requires no target-domain annotations or inference-time optimization, making it clinically practical.

Although demonstrated on liver segmentation, the approach generalizes to multi-organ tasks and may also benefit other label-scarce interventional imaging applications, such as artifact correction, denoising, or pose estimation, where domain shift limits performance. We believe this work highlights the importance of explicit representation alignment for unlocking the full potential of foundation models in CBCT.

Acknowledgements. This work was partially funded by the French Ministry for Higher Education and Research as part of CIFRE grant No. 2024/0313 and supported by GE HealthCare, where VJ and GM are employed. This work was performed using HPC resources from GENCI-IDRIS (Grant 200319327Z and 2024-A0160615058).

Disclosure of Interests. G. Miralles and V. Jugnon are employed by GE Health-Care. The remaining authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Zhu, J.-Y., Park, T., Isola, P., Efros, A. A.: Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 2242–2251 (2017)

2. Isensee, F., et al.: nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation. Nat. Methods 18, 203–211 (2021)

3. Kirillov, A., et al.: Segment Anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3992–4003 (2023)

4. Fu, J., et al.: Unsupervised Domain Adaptation for Pediatric Brain Tumor Segmentation. In: Proceedings of the ADSMI Workshop at MICCAI 2024 (2024)

5. Munk, A., Nielsen, M.: MDD-UNet: Domain Adaptation for Medical Image Segmentation with Theoretical Guarantees, a Proof of Concept. In: Lutchyn, T., Ramírez Rivera, A., Ricaud, B. (eds.) Proceedings of the 5th Northern Lights Deep Learning Conference (NLDL), Proc. Mach. Learn. Res., vol. 233, pp. 174–180 (2024)

6. Zhang, Y., et al.: Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation. In: Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), vol. 33, no. 1 (2019)

7. Zhang, X., Wu, Y., Angelini, E., Li, A., Guo, J., Rasmussen, J.M., O’Connor, T.G., Wadhwa, P.D., Jackowski, A.P., Li, H., Posner, J., Laine, A.F., Wang, Y.: MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5851–5862 (2024)

8. Chen, C., et al.: MA-SAM: Modality-agnostic SAM adaptation for 3D medical image segmentation. Med. Image Anal. 98, 103310 (2024). https://doi.org/10.1016/j. media.2024.103310

9. Blankemeier, L., et al.: Merlin: A vision-language foundation model for 3D computed tomography. Res. Sq. (2024). https://doi.org/10.21203/rs.3.rs-4546309/v1

10. Ma, J., et al.: MedSAM2: Segment Anything in 3D medical images and videos. arXiv:2504.03600 (2025)

11. Wang, H., et al.: SAM-Med3D: Towards general-purpose segmentation models for volumetric medical images. arXiv:2310.15161 (2024)

12. Wasserthal, J., et al.: TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. Radiol. Artif. Intell. 5(5), e230024 (2023) https://doi. org/10.1148/ryai.230024

13. He, Y., et al.: VISTA3D: A unified segmentation foundation model for 3D medical imaging. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20863–20873 (2025) https://doi.org/10.1109/ CVPR52734.2025.01943

14. Hong, J., et al.: Breath-hold CT and cone-beam CT images with expert manual organ-at-risk segmentations from radiation treatments of locally advanced pancreatic cancer (Pancreatic-CT-CBCT-SEG). The Cancer Imaging Archive (2021). https://doi.org/10.7937/TCIA.ESHQ-4D90

15. Bilic, P., et al.: The Liver Tumor Segmentation Benchmark (LiTS). Medical Image Analysis 84, 102680 (2023). https://doi.org/10.1016/j.media.2022.102680

16. Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S.: Barlow Twins: Self-supervised learning via redundancy reduction. In: Proceedings of the 38th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research (PMLR), vol. 139, pp. 12310–12320 (2021)

17. La Barbera, Giammarco and Boussaid, Haithem and Maso, Francesco and Sarnacki, Sabine and Rouet, Laurence and Gori, Pietro and Bloch, Isabelle: Anatomically constrained CT image translation for heterogeneous blood vessel segmentation. In: BMVC (2022)

18. Hofman, Judy and Tzeng, Eric and Park, Taesung and Zhu, Jun-Yan and Isola, Phillip and Saenko, Kate and Efros, Alexei and Darrell, Trevor: CyCADA: Cycle-Consistent Adversarial Domain Adaptation. In: ICML (2018)

## Supplementary Material

## A Reproducibility and Model Integration

Public dataset release. We release liver segmentation masks for the 39 paired CT–CBCT cases of Pancreatic-CT-CBCT-SEG. The original collection provides expert annotations for several organs at risk but does not contain liver masks. We contribute paired CT and CBCT liver masks in NIfTI format, using the naming convention LiverCT\_XXXX.nii.gz and LiverCBCT\_XXXX.nii.gz, respectively. The complete paired dataset, including the images and released masks, is publicly available through the repository referenced in the main paper.

Dataset organization and preprocessing. Source CT and target CBCT volumes are organized using separate imagesTr, imagesTs, labelsTr, and labelsTs directories. All images and label maps are resampled to isotropic 1.8 mm voxel spacing. Image volumes are resampled using continuous interpolation, whereas nearest-neighbor interpolation is used for segmentation masks to preserve their discrete labels. Training and test partitions are defined at the patient level.

Common backbone for UDA comparisons. For comparisons between featurealignment methods, we use the same 3D U-Net backbone comprising five resolution stages, 64 base channels, two $3 \times 3 \times 3$ convolutions per stage, LeakyReLU activations, and skip connections. Standard spatial and intensity augmentations include rotation, scaling, flipping, contrast modification, and additive noise. This common configuration isolates the efect of the adaptation objective from diferences in backbone architecture or preprocessing.

Integration into VISTA-3D. To integrate our method into VISTA-3D, the network is decomposed into a feature extractor $\psi ,$ , a representation head $f ,$ and the subsequent task head g. An adversarial head $f ^ { \prime }$ is initialized as a copy of $f$ and is optimized only during adaptation. Our implementation introduces an adapted VISTA-3D class containing these components and the redundancyreduction objectives. After training, $f ^ { \prime }$ is removed, and inference uses the original feature and task paths. Consequently, adaptation introduces no additional inference-time module or optimization.

Training and evaluation workflow. All paths, adaptation coeficients, and training options are specified in a single configuration file. The released implementation provides separate scripts for adaptation and target-domain evaluation, together with the source-only and adapted checkpoints. The repository also provides the data loading, preprocessing, validation, learning-rate scheduling, and metric computation procedures used for the reported experiments.