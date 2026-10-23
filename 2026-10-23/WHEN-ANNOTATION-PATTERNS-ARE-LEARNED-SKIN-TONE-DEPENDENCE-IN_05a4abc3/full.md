# WHEN ANNOTATION PATTERNS ARE LEARNED: SKIN-TONE DEPENDENCE IN DERMATOLOGY CONCEPTS

Yijian Li, Mei Bigliardi-Qi, Saad Bedros, Paul Bigliardi, Nikolaos Papanikolopoulos<sup>⋆</sup>

University of Minnesota <sup>⋆</sup>Corresponding author: papan001@umn.edu

## ABSTRACT

Concept bottleneck models (CBMs) let clinicians inspect and correct skin disease diagnoses through clinical concepts, but only if the concept annotations are reliable. We therefore audit the reliability of SkinCon, a widely used dermatology concept dataset, across skin tones. Within the same disease and with near duplicates removed, erythema is annotated far less often on dark than on light skin, with an odds ratio of 0.17. In contrast, hyperpigmentation and hypopigmentation are annotated more often, with odds ratios of 6.2 and 4.5, respectively. We refer to this pattern as tone-dependent concept annotation. An evidential concept model trained on these annotations learns the same pattern and reproduces it on unseen images. Such a model may agree with potential erythema omissions rather than contradict them. Consistent with this, three label error detectors underperform a disease-only baseline at finding erythema that dermatologist captions mention but SkinCon labels as absent. The three detectors are based on confident learning, training dynamics, and the model’s own evidential uncertainty. Concept annotations should thus be audited within each subgroup before explanations built on them can be trusted.

Index Terms— Concept bottleneck models, label errors, skin tone bias, dermatology, explainability

## 1. INTRODUCTION

Skin tone can systematically bias clinical assessment [1]. For example, on darkly pigmented skin, the contrast between red and green that makes erythema visible is reduced. The surrounding pigment also often masks the redness. In 2021, the British Association of Dermatologists acknowledged this problem by issuing guidance on describing erythema in skin of color [2]. Because dermatology concept datasets are annotated from photographs, a sign that is difficult for a clinician to see at the bedside is likely also difficult for an annotator to identify in an image.

Concept bottleneck models (CBMs) [3, 4] are increasingly used for interpretable skin disease diagnosis [5]. They route each diagnosis through an intermediate layer of clinical concepts defined by experts. This makes the model’s reasoning readable and correctable. A clinician who disagrees with a predicted concept, such as “plaque present”, can intervene on it. Both properties assume that the concept annotations are correct. If a concept is systematically misannotated for a subgroup, both fail together. The explanation remains faithful to the model, but the model’s reasoning rests on incorrect labels. Intervention, in turn, relies on a clinician who may be equally unable to see the error in the photograph.

Label error detectors [6, 7] and evidential concept models [5] are designed to flag unreliable labels and concepts. However, both rely on a model trained on the very annotations they are meant to check. Confident learning, for instance, flags a label when the model disagrees with it. It assumes that label noise depends only on the true class. An omission caused by poor visibility violates this assumption because the sign is missed precisely in images where it is faint. A model trained on these labels may then learn to reproduce the omission rather than disagree with it, and the error may go undetected. To our knowledge, label noise that depends on group membership has been modeled for fair classification [8] but not for concept annotation.

We make three contributions. First, within the same disease, SkinCon [9] annotates erythema less often on dark skin and pigmentary changes more often. We call this tonedependent concept annotation. Second, a concept model trained on these annotations reproduces it on unseen images. Third, consistent with this, three detectors underperform a disease-only baseline at finding potential erythema omissions.

## 2. METHODS

## 2.1. Data

Fitzpatrick17k [10] is a public dataset of 16,577 clinical dermatology images covering 114 skin conditions. Of these, 3,230 carry SkinCon [9] annotations. These describe each image with 48 binary clinical concepts defined by dermatologists. Following [5], we retain the 22 concepts with more than 50 positive examples. Nearly all images also have a Fitzpatrick skin type (FST) label. FST I and II are grouped as light, III and IV as mid, and V and VI as dark. The annotated set includes 1,524 light, 1,198 mid, and 424 dark images. The other 84 lack an FST label.

Fitzpatrick17k contains many near-duplicate images and inconsistent FST labels, as shown by an independent audit [11]. We therefore remove duplicates using the audit’s keep list, retaining 11,394 of the 16,577 images. Of the 3,230 annotated images, 2,217 remain. They include 1,176 light, 743 mid, and 242 dark images. The other 56 lack an FST label. The odds ratio analysis and concept model training are run on both sets, with the deduplicated set as primary. Because the keep list does not correct FST labels, we also check the odds ratios using the second set of FST labels in Fitzpatrick17k, from Centaur Labs [10]. The first set, from Scale AI, remains primary.

## 2.2. Quantifying annotation dependence on skin tone

For each concept $C _ { k }$ , we model the annotation of image i by a logistic regression with disease fixed effects:

$$
\mathrm { l o g i t } P ( C _ { i k } = 1 ) = \alpha _ { k , d ( i ) } + \beta _ { k } D _ { i } ,\tag{1}
$$

where $C _ { i k } \in \{ 0 , 1 \}$ is the SkinCon annotation of $C _ { k }$ on image $i , d ( i )$ is the disease of image i with intercept $\alpha _ { k , d ( i ) } , D _ { i }$ equals 1 for dark skin and 0 for light skin, and $\beta _ { k }$ is the shift in log odds for dark skin, shared across diseases. Only light and dark images are used. The odds ratio of $C _ { k }$ is then

$$
\mathrm { O R } _ { k } = e ^ { \beta _ { k } } = \frac { \mathrm { o d d s } ( C _ { i k } = 1 \mid D _ { i } = 1 , \ : d ( i ) = d ) } { \mathrm { o d d s } ( C _ { i k } = 1 \mid D _ { i } = 0 , \ : d ( i ) = d ) } ,\tag{2}
$$

where $\mathrm { { \partial \ d d s } } ( A ) = P ( A ) / ( 1 - P ( A ) )$ . Because the ratio is defined within each disease $d ,$ differences in disease composition between tone groups do not affect $\mathrm { O R } _ { k }$ . The fit includes only diseases in $\mathcal { D } _ { k } = \{ d : \ n _ { d } \geq 1 0 , \ 0 < \bar { D } _ { d } < 1 , \ 0 \ <$ $\bar { C } _ { d k } ~ < ~ 1 \}$ , where $n _ { d }$ is the number of light and dark skin images of disease $d ,$ and $\bar { D } _ { d }$ and $\bar { C } _ { d k }$ are the means of $D _ { i }$ and $C _ { i k }$ within d. The last two conditions require both tone groups and both annotation values to be present; diseases that violate them carry no information about $\beta _ { k }$ . The 95% percentile interval of $\mathrm { O R } _ { k }$ comes from 2,000 bootstrap resamples of images. Each resample repeats the disease selection and the fit.

To test whether the concept model inherits these dependencies, we refit Eq. (1) with the predicted probability in place of $C _ { i k }$

$$
\mathrm { l o g i t } \mathbb { E } [ \bar { p } _ { i k } \ | \ d ( i ) , D _ { i } ] = \alpha _ { k , d ( i ) } ^ { \prime } + \beta _ { k } ^ { \prime } D _ { i } ,\tag{3}
$$

where $\bar { p } _ { i k }$ is $p _ { i k }$ from Section 2.4, averaged over the three seeds. This fractional logistic regression uses the same disease sets $\mathcal { D } _ { k }$ and bootstrap, and its odds ratio is $e ^ { \beta _ { k } ^ { \prime } }$ . As controls, both fits are repeated for papule and scale, two morphological concepts that do not depend on color.

## 2.3. Detection protocol and baseline

For concept $k ,$ we evaluate images that SkinCon labels as absent and that have an English SkinCAP caption [12]. GPT-4 refined and translated the captions, and dermatologists checked them. Each caption sentence is searched for terms that indicate concept k. Negated mentions such as “no erythema” are ignored. For erythema, the terms include “erythematous” and “red”. “Violaceous” is added because erythema often looks violaceous on dark skin [2]. If a term appears in a sentence about the image itself rather than the disease in general, the image is a positive and a potential omission. Images without a mention count as negatives, but the concept may still be present.

Because a signal could perform well by exploiting the disease mix alone, we compare every signal with a disease-only baseline that uses no image information. The baseline scores image i by $r _ { k , d ( i ) }$ , the fraction of annotated images of disease $d ( i )$ that SkinCon labels with concept k. It uses SkinCon annotations only, so it has no access to the SkinCAP reference. It is evaluated on the same images as the signals. Performance is measured by AUC, with paired 95% intervals for each signal’s difference from the baseline over 2,000 bootstrap resamples.

## 2.4. Signals under test

We evaluate one signal from each of three detector families. For confident learning [6], we use the self-confidence score $q _ { i k } = p _ { i k } ^ { C _ { i k } } ( 1 - p _ { i k } ) ^ { \bar { 1 - } C _ { i k } }$ , where $p _ { i k }$ is the predicted probability of concept k for image i from a model trained without image i. For training dynamics, the signal is the area under the margin (AUM) [7], computed on predicted probabilities rather than logits. It averages the margin $( 2 C _ { i k } - 1 ) ( 2 p _ { i k } ^ { ( t ) } - 1 )$ over the training epochs t in which image i is included, where $p _ { i k } ^ { ( t ) }$ is the prediction at epoch t. For the model’s own uncertainty, the signal is evidential uncertainty $u _ { i k } = 2 / ( a _ { i k } +$ $b _ { i k } )$ [5], where $a _ { i k }$ and $b _ { i k }$ are the parameters of the predicted Beta distribution with mean $p _ { i k }$ . A low q, a low AUM, or a high u indicates a suspected error.

All three signals require a trained concept model. We use evi-CEM [5], an evidential concept embedding model, without its concept rectification step. It uses a ResNet-34 backbone and predicts one of three coarse categories: benign, malignant, or nonneoplastic. We train it separately on the 2,217 deduplicated and 3,230 full annotated images, using five-fold cross-validation and three random seeds.

## 3. RESULTS

## 3.1. Annotation depends on skin tone in both directions

Within the same disease, SkinCon annotates erythema far less often on dark skin and both pigmentary concepts more often. In Fig. 1(a), the odds ratio is 0.17 [0.08, 0.23] for erythema, 6.19 for hyperpigmentation, and 4.50 for hypopigmentation. The two control concepts show no evidence of a tone difference. The pattern holds on the full set, with odds ratios of 0.16, 5.57, and 3.55. With the Centaur Labs labels, they are

![](images/3f850c905010ac5aff314e23e544089040ba8d70cdefe6e232291744696d1c4d.jpg)

![](images/40166425baa18f836a11b01c25519050be794274552382ccb0a1736d3c93e49b.jpg)  
Fig. 1. (a) Odds ratio of each concept on dark vs light skin for the annotations (filled) and the model’s predictions (open) in the deduplicated set, with 95% bootstrap intervals; papule and scale are controls. (b) Erythema annotation rate on dark and light skin in the full annotated set, shown for illustration only. The five diseases have the largest drop from light to dark among those with at least 10 images in each tone group.

Table 1. Detection AUC on the deduplicated set. pos: potential omissions. Totals include mid and unlabeled tones. Detector values are means over three seeds. Baseline: diseaseonly; u: evidential uncertainty; CL: self-confidence q.
<table><tr><td></td><td>pos / n</td><td>Baseline</td><td>u</td><td>AUM</td><td>CL</td></tr><tr><td>Erythema</td><td>188 / 643</td><td>0.723</td><td>0.483</td><td>0.646</td><td>0.633</td></tr><tr><td>dark skin</td><td>37 / 121</td><td>0.618</td><td>0.455</td><td>0.517</td><td>0.474</td></tr><tr><td>light skin</td><td>83 / 273</td><td>0.760</td><td>0.486</td><td>0.660</td><td>0.660</td></tr><tr><td>Hyperpig.</td><td>42 / 1604</td><td>0.749</td><td>0.664</td><td>0.751</td><td>0.672</td></tr></table>

0.14, 9.93, and 6.28 on the deduplicated set and 0.15, 8.03, and 4.23 on the full set. None of these 12 intervals includes 1. Fig. 1(b) shows the same erythema pattern within individual diseases. If the erythema difference reflects omissions, images that SkinCon labels as absent should more often have erythema on dark than on light skin. Their SkinCAP captions should then mention it more often on dark skin, giving a disease-adjusted odds ratio above 1. This could not be confirmed. On the full set, this SkinCAP odds ratio is 0.93 [0.16, 1.81], from only 109 images across seven diseases.

## 3.2. The model inherits the pattern

Fig. 1(a) shows that the model’s predictions on unseen images reproduce the pattern for all three concepts. For the two controls, the model’s odds ratios do not differ clearly from those of the annotations. The model is never given skin type, and each odds ratio compares images of the same disease. It must therefore draw the pattern from image content, possibly skin color itself. For erythema, the effect is smaller than in the annotations. The model’s odds ratio is 0.25 [0.17, 0.32], versus

Table 2. Confident learning flag rate on dark relative to light skin, deduplicated set. Ranges and ranks are over three seeds, among 22 concepts. Rows are the top three by rank and the two concepts specified in advance (<sup>†</sup>). Sig.: seeds with a ratio above 1 and $p < 0 . 0 5 / 2 2$ in a two-proportion z-test.
<table><tr><td>Concept</td><td>Dark/light</td><td>Rank</td><td>Sig.</td></tr><tr><td>Hypopigmentation</td><td>3.09-3.38</td><td>1/1/1</td><td>3/3</td></tr><tr><td>Scar</td><td>2.36–2.68</td><td>2/2/2</td><td>1/3</td></tr><tr><td>Patch</td><td>2.10–2.32</td><td>3/3/3</td><td>2/3</td></tr><tr><td>Hyperpigmentation†</td><td>1.66-1.77</td><td>4/4/5</td><td>3/3</td></tr><tr><td>Erythema†</td><td>1.33-1.53</td><td>11/6/9</td><td>0/3</td></tr></table>

0.17 for the annotations. Each seed alone gives 0.22 to 0.28, and the full set gives 0.26 versus 0.16. The smaller effect may partly reflect scale, because the model outputs probabilities while the annotations are binary. As a sensitivity check, we binarize each concept at the threshold where it is predicted as often as it is annotated. The three odds ratios become 0.17, 8.94, and 4.37. These effects are about as strong as in the annotations, or stronger.

## 3.3. Three detectors trail the baseline on erythema

If the model has learned this pattern, its signals may fail to rank potential omissions above other images. Table 1 measures this by AUC for two concepts specified in advance. For erythema, all three signals score below the disease-only baseline. With an AUC of 0.48, u is no better than chance. Paired differences average −0.24 for $u , - 0 . 0 8$ for AUM, and −0.09 for CL. Every 95% interval lies below zero in all three seeds. The highest upper bounds are −0.17 for $u , - 0 . 0 2$ for AUM, and −0.03 for CL. The result holds even when the baseline is computed with five-fold cross-validation. This lowers its AUC only slightly to 0.718. Each skin tone group shows the same direction, but the groups are small and some intervals cross zero. On the full set, all three erythema estimates also fall below the baseline. For hyperpigmentation, no signal clearly outperforms the baseline. On the full set, AUM exceeds it mainly because the baseline falls from 0.749 to 0.721.

## 3.4. Detector flags follow the annotation pattern

In an exploratory analysis, we ask whether an auditor could detect this pattern from detector flags alone. Confident learning with cleanlab’s default settings checks each SkinCon annotation and flags those that are potentially incorrect. Skin-CAP is not used here. Table 2 ranks the 22 concepts by the ratio of their flag rates on dark versus light skin. The two pigmentary concepts rank near the top and remain significant after Bonferroni correction. Erythema is never significant and would be overlooked. Each flag also has a direction, because a flag on a “present” annotation suggests removing it and a flag on an “absent” annotation suggests adding it. Separate rates for present and absent annotations show that the modest overall ratio of erythema mixes two opposite trends. Erythema marked present is flagged 2.98–4.33 times as often on dark as on light skin, while erythema marked absent is flagged only 0.41–0.55 times as often. Hyperpigmentation marked absent is flagged 2.53–3.75 times as often. In each case, the flags point in the same direction as the annotation pattern, with or without adjustment for disease.

## 4. DISCUSSION

CBMs assume that concept annotations are reliable. Our results show that tone-dependent patterns in SkinCon annotations pass into a model trained on them. Potential erythema omissions also slip past the three detectors tested here. Both findings are consistent with one explanation. A miss that the model has learned may look correct to detectors built on that model.

The erythema pattern and the pigmentary patterns point in opposite directions and may have different causes. Erythema appears less red on dark skin [2], so its pattern most plausibly reflects what annotators can see. In contrast, pigmentary changes stand out more on dark skin and are more expected there. They are also more common after inflammation in darker skin [13], so their pattern may be partly real. The SkinCAP check cannot settle this because its dermatologists saw the same photographs and may have missed the same signs. Whatever the cause, the model learns the pattern, and the detectors cannot tell whether a difference is real or comes from annotation.

One possible mechanism fits the model and detector findings. Confident learning assumes that label noise depends only on the true class. Erythema missed because it looks faint would break this assumption. Such image-dependent noise is known to be hard to handle [14, 15]. The model may learn such omissions from the image and agree with those labels. The disagreement-based signals CL and AUM would miss many of them. Evidential uncertainty only measures how much evidence the model has and ignores whether it disagrees with the label. This is consistent with all three falling behind the disease-only baseline. Faint erythema can also be missed on light skin, where the detectors fall short too.

CBMs are often trained jointly on concepts and diagnosis, so concept labels shape both what the network learns and how it explains itself. A CBM trained on SkinCon may therefore show erythema less often in its explanations on dark skin, as the annotations do. Our concept model already does this, with an odds ratio of 0.25 for erythema. A clinician who intervenes on these concepts from the photograph alone may miss the sign too. Concept accuracy measured against these labels could also overstate performance on dark skin. Other concept datasets annotated from photographs may show the same tone dependence. To detect and reduce it, we suggest three steps. First, report disease-adjusted odds ratios by skin tone for every concept. Second, compare any label error detector with a disease-only baseline. Third, describe color signs in annotation guidelines as they look on dark skin, where erythema often appears violaceous or dusky rather than red [2].

Limitations. First, our audit identifies which concepts are annotated differently on dark and light skin, but not which images are mislabeled. Second, the evidence comes from one dataset with few dark-skin images, so some intervals are wide. Images of the same disease may still differ in stage, treatment, body site, or photo conditions. FST is only a noisy proxy for skin tone [11], but the odds ratios hold under a second set of labels. Third, all model and detector results come from one concept model. Fourth, the reference itself may favor the disease-only baseline. The SkinCAP dermatologists proposed diagnoses themselves, so they may have mentioned erythema partly because it is expected for that disease rather than clearly visible in the image. We exclude general statements about the disease, but the diagnosis may still have influenced the descriptions. A natural next step is to extend the audit to Diverse Dermatology Images (DDI) [16], because SkinCon annotates its images with the same 48 concepts.

## 5. CONCLUSION

We find tone-dependent concept annotation in SkinCon. Within the same disease, erythema labels are less frequent on dark skin and pigmentary labels more frequent, with an erythema odds ratio of 0.17. A concept model inherits this pattern, with an erythema odds ratio of 0.25. Consistent with this, three detectors built on the model trail a disease-only baseline in finding potential erythema omissions. Concept annotations should be audited within each subgroup, with independent review of suspected errors.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study is a retrospective analysis of open-access data from Fitzpatrick17k [10], SkinCon [9], and SkinCAP [12]. No new data were collected. Ethical approval was not required for this use of public data.

## 7. ACKNOWLEDGMENTS

No funding was received for this study. The authors have no relevant financial or non-financial interests to disclose. The authors thank the Minnesota Supercomputing Institute at the University of Minnesota for computing resources. Anthropic’s Claude was used only to edit the language of text written by the authors. It did not contribute scientific content. All edits were reviewed by the authors.

## 8. REFERENCES

[1] M. A. Ben-Gashir, P. T. Seed, and R. J. Hay, “Reliance on erythema scores may mask severe atopic dermatitis in black children compared with their white counterparts,” Br. J. Dermatol., vol. 147, no. 5, pp. 920–925, 2002.

[2] M. Gass, O. E. Dadzie, N. Dlova, A. Petit, and N. Goad, “Skin diversity erythema description guidance,” British Association of Dermatologists, February 2021, https://cdn.bad. org.uk/uploads/2024/07/17092734/ Skin-diversity-descriptors-erythema-r pdf.

[3] P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, and P. Liang, “Concept bottleneck models,” in Proc. ICML, 2020, pp. 5338–5348.

[4] M. Espinosa Zarlenga, P. Barbiero, G. Ciravegna, G. Marra, F. Giannini, M. Diligenti, Z. Shams, F. Precioso, S. Melacci, A. Weller, P. Lio, and M. Jamnik,\` “Concept embedding models: Beyond the accuracyexplainability trade-off,” in Proc. NeurIPS, 2022, pp. 21400–21413.

[5] Y. Gao, Z. Gao, X. Gao, Y. Liu, B. Wang, and X. Zhuang, “Evidential concept embedding models: Towards reliable concept explanations for skin disease diagnosis,” in Proc. MICCAI, 2024, pp. 308–317.

[6] C. G. Northcutt, L. Jiang, and I. L. Chuang, “Confident learning: Estimating uncertainty in dataset labels,” J. Artif. Intell. Res., vol. 70, pp. 1373–1411, 2021.

[7] G. Pleiss, T. Zhang, E. Elenberg, and K. Q. Weinberger, “Identifying mislabeled data using the area under the margin ranking,” in Proc. NeurIPS, 2020, pp. 17044– 17056.

[8] J. Wang, Y. Liu, and C. Levy, “Fair classification with group-dependent label noise,” in Proc. ACM FAccT, 2021, pp. 526–536.

[9] R. Daneshjou, M. Yuksekgonul, Z. R. Cai, R. Novoa, and J. Y. Zou, “SkinCon: A skin disease dataset densely annotated by domain experts for fine-grained debugging and analysis,” in Proc. NeurIPS Datasets and Benchmarks, 2022, pp. 18157–18167.

[10] M. Groh, C. Harris, L. Soenksen, F. Lau, R. Han, A. Kim, A. Koochek, and O. Badri, “Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset,” in Proc. CVPR Workshops, 2021, pp. 1820–1828.

[11] K. Abhishek, A. Jain, and G. Hamarneh, “Investigating the quality of DermaMNIST and Fitzpatrick17k dermatological image datasets,” Sci. Data, vol. 12, pp. 196, 2025.

[12] J. Zhou, L. Sun, Y. Xu, W. Liu, S. Afvari, Z. Han, J. Song, Y. Ji, X. He, and X. Gao, “SkinCAP: A multimodal dermatology dataset annotated with rich medical captions,” arXiv:2405.18004v1, 2024, Hugging Face release v240623, field caption zh polish en.

[13] E. C. Davis and V. D. Callender, “Postinflammatory hyperpigmentation: a review of the epidemiology, clinical features, and treatment options in skin of color,” J. Clin. Aesthet. Dermatol., vol. 3, no. 7, pp. 20–31, 2010.

<sup>ness-guidance.</sup>[14] Xiaobo Xia, Tongliang Liu, Bo Han, Nannan Wang, Mingming Gong, Haifeng Liu, Gang Niu, Dacheng Tao, and Masashi Sugiyama, “Part-dependent label noise: Towards instance-dependent label noise,” in Proc. NeurIPS, 2020, pp. 7597–7610.

[15] Pengfei Chen, Junjie Ye, Guangyong Chen, Jingwei Zhao, and Pheng-Ann Heng, “Beyond class-conditional assumption: A primary attempt to combat instancedependent label noise,” in Proc. AAAI, 2021, pp. 11442– 11450.

[16] R. Daneshjou, K. Vodrahalli, R. A. Novoa, M. Jenkins, W. Liang, V. Rotemberg, J. Ko, S. M. Swetter, E. E. Bailey, O. Gevaert, et al., “Disparities in dermatology AI performance on a diverse, curated clinical image set,” Sci. Adv., vol. 8, no. 32, pp. eabq6147, 2022.