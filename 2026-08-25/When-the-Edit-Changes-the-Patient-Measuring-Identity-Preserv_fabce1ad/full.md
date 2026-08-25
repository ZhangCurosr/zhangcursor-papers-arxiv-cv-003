# When the Edit Changes the Patient: Measuring Identity Preservation in Counterfactual Retinal Images

Andrea Posada<sup>1,2⋆[0009−0005−3021−5525]</sup>, Wenke Karbole<sup>1,3⋆</sup>, Bach Ngoc   
Doan<sup>1⋆</sup>, Alexander Weers<sup>1,2</sup>, Solmaz Abdolrahimzadeh<sup>4</sup>, Maria Patsiamanidi<sup>5</sup>,   
Kahkashan Haider<sup>6</sup>, Vaishali Khare<sup>6</sup>, Daniel Rueckert<sup>1,2,7</sup>, Andrew Lotery<sup>5</sup>, Sobha Sivaprasad<sup>6</sup>, and Martin J. Menten<sup>1,2,7</sup>

<sup>1</sup> Technical University of Munich and TUM University Hospital, Germany wenke.karbole@tum.de

Munich Center of Machine Learning (MCML), Munich, Germany <sup>3</sup> Munich Data Science Institute (MDSI), Munich, Germany 4 Sapienza University of Rome, Rome, Italy

<sup>5</sup> Faculty of Medicine, University of Southampton, Southampton, United Kingdom <sup>6</sup> Moorfields Eye Hospital NHS Foundation Trust, London, United Kingdom 7 Imperial College London, UK Equal contribution.

Abstract. Counterfactual medical image generation aims to modify an existing image to reflect a hypothetical scenario in which certain characteristics of the imaged subject are altered, while keeping their identity fixed. Most existing works repurpose established image editing methods, which do not directly supervise identity preservation. Instead, they assume that identity is implicitly preserved by anchoring generation to the source image. This assumption is rarely tested and may fail in domains where biometric cues are subtle, such as retinal optical coherence tomography (OCT). In this work, we explicitly measure identity preservation for three groups of text-conditioned editing methods – source-anchored, structured-prompt, and paired-training – using referee classifiers, embedding alignment scores, and a blind reader study. We find that all methods produce high-quality OCT images with comparable editing success, yet their identity preservation difers markedly. Source-anchored editing frequently alters the depicted subject, while paired-training preserves it best. We argue that future work on medical counterfactual generation must explicitly measure and report identity preservation alongside image realism and editing success.

Keywords: Counterfactual image generation · Text-conditioned image editing · Identity preservation · Retina · OCT imaging

## 1 Introduction

Counterfactual medical image generation aims to answer the question: “What would this subject’s image look like if a given attribute were changed?” In order to be useful for applications such as visualizing disease progression and simulating treatment efects [33,9,18], the resulting synthetic images must not only be realistic and anatomically accurate, but also preserve the subject’s identity.

![](images/2a0653d4f8fb811b99e1152fe1accd73f0ae1bf21ea8a1d1db0d406606192d00.jpg)  
Fig. 1. In OCT counterfactual generation, changing global age-related macular degeneration (AMD) biomarkers requires strong edits that can disrupt the subtle cues linked to patient identity. As a result, the generated samples often fail to preserve the subject’s identity even when the edits appear faithful and realistic.

Most existing text-conditioned counterfactual methods repurpose established difusion-based image editing techniques, which do not directly supervise identity preservation. Instead, they assume it is preserved implicitly, by anchoring generation to the source image or by training on image pairs. This assumption has largely held because most medical counterfactual work has been developed and evaluated on chest X-rays [9,28,3], where the subject’s identity is strongly linked to salient biometric features: skeletal structure, body habitus, heart outlines and lung shape vary markedly between subjects and enable near-perfect patient reidentification [27]. Conversely, in retinal optical coherence tomography (OCT), identity cues are encoded by more subtle features such as retinal-layer thickness and blood vessels [17]. We hypothesize that, for OCT images, even small edits can substantially alter a subject’s biometric features and thus produce implausible counterfactuals. Consequently, the assumption of identity preservation as an automatic by-product of current text-conditioned editing methods may not hold in retinal imaging (Figure 1).

To test this hypothesis, we conduct experiments using a longitudinal OCT dataset of subjects with age-related macular degeneration (AMD). Global AMDrelated changes often superimpose subtle identity cues, making counterfactual editing challenging. We compare three text-conditioned image editing groups – source-anchored, structured-prompt, and paired-training – that have not been optimized for identity preservation. In addition to realism and editing success, we assess identity preservation using three complementary approaches: referee classifiers for the subject sex and eye identity, an alignment score between the generated edit and the subject’s real longitudinal change, and a blind reader study in which clinically trained readers judge whether image pairs depict the same eye.

Our findings show that identity preservation cannot be taken for granted: even small edits that appear realistic and achieve the intended change can alter the depicted subject’s identity. Moreover, accounting for it during model development informs hyperparameter selection at little cost to realism or editing success. We therefore argue that identity preservation in medical counterfactual generation must be explicitly measured rather than assumed, particularly for imaging modalities with subtle identity cues such as retinal OCT.

## 2 Related Work

## 2.1 Text-Conditioned Medical Counterfactual Generation

Text-conditioned counterfactual generation enables precise manipulation of specific pathological or demographic traits through language. Most existing work repurposes established text-conditioned difusion-based image editing methods [14,9,18,34,28]. These methods can be grouped into four categories according to how they modify the underlying source image: source-anchored, structuredprompt, text- and mask-conditioned, and paired-training. Across all four groups, identity preservation is usually treated as a by-product of the editing mechanism rather than an explicit objective.

Source-anchored methods closely tie the difusion process to the source image, so that identity is retained through proximity to the source alone. SDEdit [21], one of the most representative works, partially noises the source and denoises it under the new prompt. Although training-free and backbone-agnostic, these methods ofer limited editing control. Classifier-guided variants [16,1,15,14], which steer denoising with classifier gradients, provide greater control but are restricted to a classifier’s label space rather than free-form text.

Structured-prompt methods predominantly build on Prompt-to-Prompt [10] via null-text inversion [24]: an image is edited by manipulating the word-level attention maps of the source and target text prompts during the generation process. Since such edits typically do not pertain to identity, it is expected to be preserved implicitly. Although this paradigm has been applied to medical counterfactual generation [18], it relies on closely aligned prompts that difer by only a few words, whereas clinical data consists of free-text reports and multiword pathology descriptions.

Text- and mask-conditioned methods [2,7,28,8] confine changes to a userprovided or automatically estimated region, preserving structures outside the mask by design. Nonetheless, they require accurate masks and cannot model edits with global efects such as aging, sex, or difuse pathology. Because most AMD-related changes are inherently global, impacting multiple retinal layers, we do not investigate these strategies further in our study.

Paired-training methods learn counterfactual edits from same-subject image pairs. InstructPix2Pix [5] popularized this concept for natural images, while

BiomedJourney [9] and SADM [34] extend it to longitudinal medical data. Paired data, however, is not readily available in medical imaging.

## 2.2 Measuring Identity Preservation

Although identity preservation is central to a plausible counterfactual, it is rarely measured directly. Most work instead reports proximity metrics [18,20,33,25], such as pixel-level L1 distances, structural similarity [30], or perceptual distance [35]. These quantify how much an image changes as a result of an edit, thereby conflating identity with minimality. As a consequence, they are insensitive to small changes that specifically alter subject identity, while large but identity-consistent edits are overly penalized. Only a few works report attribute preservation in their evaluation [22,32,23], and methods that account for identity preservation directly during training are rarely adopted in practice [19,31]. Identity preservation thus remains an overlooked aspect of recent work on medical counterfactual image generation.

## 3 Method

We investigate to which extent image editing preserves subject identity when generating counterfactual images on a longitudinal OCT dataset of AMD patients (Section 3.1). To this end, we employ three complementary evaluation approaches – referee networks, embedding-alignment metrics, and a blind reader study (Section 3.2) – to compare three image editing strategies: source-anchored, structured-prompt, and paired-training (Section 3.3).

## 3.1 Dataset

All experiments were performed on a in-house dataset from the University Hospital Southampton containing 43 165 OCT images from 6 157 eyes of AMD patients. Most patients contributed multiple scans over a time period of up to seven years, that may span their eye’s conversion from early and intermediate AMD to late stages of the disease. For each image, we used the central B-Scan and automatically generated a text description using a vision language model for ophthalmological report generation [13].The dataset is split at the subject level into training (80%), validation (10%), and test (10%) sets. To guarantee consistency between the report and expert-derived ground truth AMD stage labels, we extracted the stage from the report using both a regex- and LLM-based approach, and disagreements were resolved by a human annotator. Reports that did not mention an AMD stage or whose staging was inconclusive were excluded from the validation and test sets.

## 3.2 Evaluation Metrics

We quantitatively measured multiple properties of a generated counterfactual $x _ { g } . ~ x _ { g }$ is generated from a real image $x _ { s }$ conditioned on a text prompt $p _ { t }$ , which describes the intended counterfactual scenario. We may also consider a real target image $x _ { t }$ as one valid counterfactual out of many. Notice that $p _ { t }$ is the textual report describing $x _ { t }$

Identity Preservation via Referee-Based Methods Since edits are restricted to AMD-related findings, all other attributes should remain unchanged. We thus assess identity preservation with two referee classifiers and a blind reader study, rating whether two images (real/edited or real/real) depict the same eye.

As the primary referee, a Siamese-based same-eye verifier $f _ { \mathrm { e y e } }$ was trained to predict whether two OCT scans originate from the same subject and laterality (F1-score of 0.97). Following biometric verification, the identity agreement between $x _ { s }$ and $x _ { g }$ is defined as:

$$
\mathrm { A g r } _ { f _ { \mathrm { e y e } } } = \frac { 1 } { | A | } \sum _ { i \in A } \mathbb { 1 } \left[ f _ { \mathrm { e y e } } \left( x _ { g } ^ { ( i ) } , x _ { s } ^ { ( i ) } \right) \geq \tau _ { 0 . 1 } \right]\tag{1}
$$

with $A = \left\{ i : f _ { \mathrm { e y e } } \left( x _ { s } ^ { ( i ) } , x _ { t } ^ { ( i ) } \right) \geq \tau _ { 0 . 1 } \right\}$ to correct for the classifier’s unreliability. The threshold $\tau _ { 0 . 1 }$ is selected on the validation split such that the false accept rate on diferent-eye pairs does not exceed 0.1%.

As an additional referee, we considered a binary sex classifier $f _ { \mathrm { s e x } }$ , which serves as a weaker but complementary identity proxy. It predicts the sex sˆ of a subject from the OCT B-scan, an attribute ideally unafected by editing (F1- score of 0.71). We measure the agreement between the sex predictions on $x _ { g }$ and $x _ { s }$ via:

$$
\operatorname { A g r } _ { f _ { \mathrm { s e x } } } = \frac { 1 } { | B | } \sum _ { i \in B } \mathbb { 1 } \left[ \hat { s } _ { g } ^ { ( i ) } = \hat { s } _ { s } ^ { ( i ) } \right]\tag{2}
$$

where $B = \left\{ i : \hat { s } _ { s } ^ { ( i ) } = \hat { s } _ { t } ^ { ( i ) } \right\}$ accounts for the classifier’s unreliability. Both referee classifiers were trained, validated, and tested on the same data splits used for the generative editing models to avoid data leakage.

We also conducted a reader study in which four clinically trained readers judged whether a pair of OCT images showed the same eye. Each reader assessed 240 randomly sampled pairs: 60 pairs per editing method plus 60 pairs of real images as a control. Within each subset, half of the pairs showed the same eye and half showed diferent eyes. Subjects were selected at random, and readers were blinded to the editing method.

Editing Success and Identity Preservation via Embedding Alignments To measure the alignment between $x _ { g }$ and $p _ { t }$ , we report their mean-corrected CLIP-score [11,36]. We further define the image change alignment (ICA) to measure whether an edit moves $x _ { s }$ in the same semantic direction as the real longitudinal transition from $x _ { s }$ to $x _ { t }$ in the latent space. Let $e _ { s } , e _ { t }$ , and $e _ { g }$ denote image embeddings using Zhang et al.’s model [36]. Then:

$$
\mathrm { I C A } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { c } \left( e _ { t } ^ { ( i ) } - e _ { s } ^ { ( i ) } , ~ e _ { g } ^ { ( i ) } - e _ { s } ^ { ( i ) } \right)\tag{3}
$$

where $S _ { c }$ is the cosine similarity and N denotes the number of samples. A score near 1 indicates that the edit is semantically aligned with the real subjectspecific transition. Because real transitions preserve identity by definition, close alignment indicates that the editing operation is not only successful but also maintains the subject’s identity.

Realism and Editing Success Metrics We measure the realism of the generated edits using the Fréchet Inception Distance (FID) [12]. Editing success is also evaluated via the F1-score and the attribution score [26] using a multiclass AMD stage classifier f (F1-score of 0.57). We use the attribution score, also known as prediction gain, to measure how the predicted logits of the target AMD class $c _ { t }$ change with the edit. While standard formulations focus only on cases where the class changes between source and target, we extend the metric to also handle same-stage edits:

$$
\phi = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { s i g n } \left( f \left( x _ { t } ^ { ( i ) } \right) _ { c _ { t } ^ { ( i ) } } - f \left( x _ { s } ^ { ( i ) } \right) _ { c _ { t } ^ { ( i ) } } \right) \cdot \left( f \left( x _ { g } ^ { ( i ) } \right) _ { c _ { t } ^ { ( i ) } } - f \left( x _ { s } ^ { ( i ) } \right) _ { c _ { t } ^ { ( i ) } } \right)\tag{4}
$$

## 3.3 Counterfactual Image Generation Strategies

We base all investigated strategies on MediSyn [6], a generalist medical text-toimage latent difusion model, and fine-tune it on text-image pairs of the OCT dataset. Since medical terminology difers markedly from natural language, we trained MediSyn with BioMedCLIP [36] and BioLORD [29] text encoders and selected the best-performing variant per method.

Source-anchored – Stochastic Diferential Editing (SDEdit) SDEdit [21] is applied at inference time. Rather than initializing the denoising process from pure noise, it starts from a noisy version of $x _ { s }$ . The strength hyperparameter γ sets the fraction of steps for the noising and denoising processes. We tuned γ over {0.2, 0.4, 0.6, 0.8, 1.0}.

Structured-prompt – Prompt-to-Prompt (P2P) P2P [10] is also applied at inference time. First, null-text inversion [24] inverts $x _ { s }$ together with its original prompt, recovering a denoising trajectory that reproduces $x _ { s }$ from noise. P2P then denoises along this trajectory conditioned on $p _ { t }$ , but it injects the attention maps of the source prompt instead of the target’s for a fraction of the steps. This fraction is determined by the self-replacement rate srs. Reusing these attention maps anchors the generation process to $x _ { s }$ . We tuned srs over {0.8, 0.6, 0.4, 0.2, 0.0}. Although $\mathrm { P 2 P }$ is designed to modify single words in otherwise identical prompts and our free-text reports difer throughout, we find that it produces meaningful counterfactual images, competitive with the other methods.

Paired training – InstructPix2Pix Unlike the inference-time strategies SDEdit and P2P, InstructPix2Pix [5] further trains the OCT-fine-tuned MediSyn model to learn semantic editing explicitly from paired data. The model is conditioned jointly on $p _ { t }$ and $x _ { s }$ to predict $x _ { t }$ . During inference, the edited image is obtained via a single conditioned denoising pass with separate classifier-free guidance scales for the image $( i g s )$ and the text $( g s )$ . We tuned igs as the main parameter for feature preservation over {1.0, 1.5, 2.0, 2.5, 3.0} and gs over {5.5, 7.5, 11.5}. Whereas the original InstructPix2Pix is trained on synthetically generated pairs, we use real source-target pairs of the same subject.

![](images/97dd0c74c25615a4e4f3a2b34cb06e00e569f525a725d750287f8429acaede30.jpg)

![](images/db536454e8a7f58da34be6da8e89edb9b81c3d53e6ab53d2e61cf7378a3b8c68.jpg)

Fig. 2. Per-method comparison on two selection criteria across realism, editing success, and identity preservation metrics. Selecting on realism and editing success alone (dashed line) can yield configurations with substantial loss of identity features. The balanced pick (solid line) recovers identity preservation while editing success and realism are largely retained.  
![](images/dddbc63be7d5631f05343133b84a9bb9b190730a99823e2e3141613960290cae.jpg)

![](images/5887d84d274e9334a50a1cef4c910f33df26912b5cc158cb3b6fea40a2986e2f.jpg)

![](images/9e0bda9d46ebea7e36f0e1fd92e7aaca263107da37031eee7a03aa82e9e0ebe1.jpg)

![](images/1a92cd51eca4fbdbc07d9fb4f8cbec3fc5535c58bf48e65608d30603c4dd5b21.jpg)  
Fig. 3. Two displayed images (real/real) or (real/edited) were graded by four clinically trained readers as the same or diferent eyes with substantial agreement (Fleiss’s $\kappa =$ 0.67). While edits generated by P2P and InstructPix2Pix are mostly recognized as belonging to the same eye, those of SDEdit reveal frequent identity loss.

Two configurations per editing strategy are chosen by the best average rank across the relevant metrics: an edit-focused selection, ranking on realism and editing success only, and a balanced selection, which additionally accounts for identity preservation (Section 4.1). We then analyze how each metric varies with the methods’ respective editing strength hyperparameters (Section 4.2).

## 4 Results and Discussion

## 4.1 Measuring Identity Preservation in Image Editing

For two out of three image editing methods, the configurations selected by the edit-focused and balanced criteria meaningfully difer (Figure 2). For P2P and SDEdit, the balanced criterion favors a lower editing strength than the editfocused criterion. This changes realism and editing success only marginally, while improving eye-identity and sex preservation, as measured by the referee classifiers. The most severe identity loss is observed for the source-anchored method SDEdit under the edit-focused criterion. Qualitative results confirm a high overall image quality and realism (Figure 1). The reader study supports our findings: edits from the edit-focused SDEdit model were more often perceived as diferent eyes than as the same (Figure 3). For P2P and InstructPix2Pix, however, reader performance was comparable to the performance on real image pairs. Overall, these results show that selecting models based on realism and editing success alone does not guarantee that subject identity is preserved.

![](images/a14ff1f62042d13040a7f5bed44b9b179fd56b0408377bc69e2c48a6da3c4787.jpg)  
Fig. 4. With increasing editing strength, all investigated text-conditioned editing strategies show a loss in identity preservation, most prominently revealed by $\mathrm { A g r } _ { \mathrm { e y e } } , \mathrm { A g r } _ { \mathrm { s e x } } .$ . Above all, we find that even minor edits can cause a drastic loss of identity features, which is not equally compensated by gains in editing success metrics.

## 4.2 Editing Strength as Decisive Hyperparameter

All methods show a clear negative correlation between identity preservation metrics and the editing strength hyperparameter (Figure 4). The F1-score remains roughly constant while the attribution score rises, which is explained by our testset composition: most transitions in our data are subclinical, with only 479 cases (26%) involving an actual AMD-stage change. Restricted to these cases, the F1- score accordingly exhibits the expected upward trend. The FID trend is consistent with prior work [1,4], in that lower editing strength keeps generated images closer to the real-data distribution. Since all editing strength hyperparameters are method-specific, the results are not directly comparable across methods. Within their respective ranges, however, InstructPix2Pix and P2P show a high robustness to variations in editing strength. This is consistent with Section 4.1, where they exhibited the strongest implicit identity signal.

## 5 Conclusion

This study has evaluated whether text-conditioned image editing methods, used for counterfactual generation, preserve subject identity when applied to retinal OCT images. While all three investigated methods produce high-quality OCT images with comparable editing success, some methods alter subtle biometric features of the eye. Therefore, identity preservation cannot be assumed as an automatic by-product of anchoring the generative process to the source image.

In response, we have demonstrated that evaluating generative models solely through the lens of image realism and editing success does not reliably capture changes in identity, yielding invalid conterfactuals. By explicitly accounting for identity preservation already during model selection and tuning, one can substantially improve this property with minimal deductions to image realism and editing success. Consequently, we argue that future work on medical counterfactual generation must explicitly measure and report identity preservation alongside image realism and editing success, particularly for imaging modalities with subtle identity cues such as retinal OCT.

Acknowledgments. This work was partially supported by the German Research Foundation (Project 532139938) and EPSRC grant EP/Y015665/1.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Augustin, M., Boreiko, V., Croce, F., et al.: Difusion visual counterfactual explanations. In: NeurIPS (2022)

2. Avrahami, O., Lischinski, D., Fried, O.: Blended difusion for text-driven editing of natural images. In: CVPR. pp. 18187–18197 (2022)

3. Bluethgen, C., Chambon, P., Delbrouck, J.B., et al.: A vision–language foundation model for the generation of realistic chest X-ray images. Nat. Biomed. Eng. 9, 494–506 (2025)

4. Boreiko, V., Augustin, M., Croce, F., et al.: Sparse visual counterfactual explanations in image space. In: DAGM GCPR (2022)

5. Brooks, T., Holynski, A., Efros, A.A.: InstructPix2Pix: Learning to follow image editing instructions. In: CVPR. pp. 18392–18402 (2023)

6. Cho, J., Zakka, C., Shad, R., et al.: A generalist model for diverse text-guided medical image synthesis. arXiv:2405.09806 (2026)

7. Couairon, G., Verbeek, J., Schwenk, H., et al.: DifEdit: Difusion-based semantic image editing with mask guidance. In: ICLR (2023)

8. Fontanella, A., Mair, G., Wardlaw, J., et al.: Difusion models for counterfactual generation and anomaly detection in brain images. IEEE Trans. Med. Imaging 44, 3574–3585 (2025)

9. Gu, Y., Yang, J., Usuyama, N., et al.: BiomedJourney: Counterfactual biomedical image generation by instruction-learning from multimodal patient journeys. arXiv:2310.10765 (2023)

10. Hertz, A., Mokady, R., Tenenbaum, J., et al.: Prompt-to-prompt image editing with cross-attention control. In: ICLR (2023)

11. Hessel, J., Holtzman, A., Forbes, M., et al.: CLIPScore: A reference-free evaluation metric for image captioning. In: EMNLP. pp. 7514–7528 (2021)

12. Heusel, M., Ramsauer, H., Unterthiner, T., et al.: GANs trained by a two timescale update rule converge to a local nash equilibrium. In: NeurIPS. pp. 6629––6640 (2017)

13. Holland, R., Taylor, T.R.P., Holmes, C., et al.: Specialized curricula for training vision-language models in retinal image analysis. npj Digit. Med. 8, 532 (2025)

14. Ilanchezian, I., Boreiko, V., Kühlewein, L., et al.: Development and validation of an ai algorithm to generate realistic and meaningful counterfactuals for retinal imaging based on difusion models. PLOS Digit Health p. e0000853 (2025)

15. Jeanneret, G., Simon, L., Jurie, F.: Adversarial counterfactual visual explanations. In: CVPR. pp. 16425–16435 (2023)

16. Jeanneret, G., Simon, L., Jurie, F.: Difusion models for counterfactual explanations. Comput. Vis. Image Underst. 249, 104207 (2024)

17. Kong, M., Hwang, S., Ko, H., et al.: Heritability of inner retinal layer and outer retinal layer thickness: the healthy twin study. Sci. Rep. 10, 3519 (2020)

18. Kumar, A., Kriz, A., Havaei, M., et al.: PRISM: High-resolution & precise counterfactual medical image generation using language-guided Stable Difusion. arXiv:2503.00196 (2025)

19. Maeng, J., Oh, K., Jung, W., et al.: IdenBAT: Disentangled representation learning for identity-preserved brain age transformation. Artif. Intell. Med. 164, 103115 (2025)

20. Melistas, T., Spyrou, N., Gkouti, N., et al.: Benchmarking counterfactual image generation. In: NeurIPS Datasets and Benchmarks Track (2024)

21. Meng, C., He, Y., Song, Y., et al.: SDEdit: Guided image synthesis and editing with stochastic diferential equations. In: ICLR (2021)

22. Menten, M.J., Holland, R., Leingang, O., et al.: Exploring healthy retinal aging with deep learning. Ophthalmol. Sci. 3 (2023)

23. Min, H., You, T., Lee, H., et al.: InstructX2X: An interpretable local editing model for counterfactual medical image generation. In: MICCAI. pp. 279–288 (2026)

24. Mokady, R., Hertz, A., Aberman, K., et al.: Null-text inversion for editing real images using guided difusion models. In: CVPR. pp. 6038–6047 (2023)

25. Monteiro, M., De Sousa Ribeiro, F., Pawlowski, N., et al.: Measuring axiomatic soundness of counterfactual image models. In: ICLR (2023)

26. Nemirovsky, D., Thiebaut, N., Xu, Y., et al.: CounteRGAN: Generating realistic counterfactuals with residual generative adversarial nets. In: UAI. vol. 180, pp. 1488–1497 (2022)

27. Packhäuser, K., Gündel, S., Münster, N., et al.: Deep learning-based patient reidentification is able to exploit the biometric nature of medical chest X-ray data. Sci. Rep. 12, 14851 (2022)

28. Pérez-García, F., Bond-Taylor, S., Sanchez, P.P., et al.: RadEdit: Stress-testing biomedical vision models via difusion image editing. In: ECCV. pp. 358–376 (2025)

29. Remy, F., Demuynck, K., Demeester, T.: BioLORD: Learning ontological representations from definitions for biomedical concepts and their textual descriptions. In: EMNLP. pp. 1454–1465 (2022)

30. Wang, Z., Bovik, A.C., Sheikh, H.R., et al.: Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process. 13, 600–612 (2004)

31. Xia, T., Chartsias, A., Wang, C., et al.: Learning to synthesise the ageing brain without longitudinal data. Med. Image Anal. 73, 102169 (2019)

32. Xia, T., Roschewitz, M., De Sousa Ribeiro, F., et al.: Mitigating attribute amplification in counterfactual image generation. In: MICCAI. pp. 546–556 (2024)

33. Yeganeh, Y., Farshad, A., Charisiadis, I., Hasny, M., Hartenberger, M., et al.: Latent drifting in difusion models for counterfactual medical image synthesis. In: CVPR. pp. 7685–7695 (2025)

34. Yoon, J.S., Zhang, C., Suk, H.I., et al.: SADM: Sequence-aware difusion model for longitudinal medical image generation. In: IPMI. pp. 388–400 (2023)

35. Zhang, R., Isola, P., Efros, A.A., et al.: The unreasonable efectiveness of deep features as a perceptual metric. In: CVPR. pp. 586–595 (2018)

36. Zhang, S., Xu, Y., Usuyama, N., et al.: A multimodal biomedical foundation model trained from fifteen million image–text pairs. NEJM AI 2, AIoa2400640 (2025)