# Trick-or-Treat: Adversarial Evaluation of Multimodal Medical Reasoning with Synthetic Images

Oishi Banerjee, M.S.<sup>1,∗</sup> Eujin Lee<sup>1,2,∗</sup> Sung Eun Kim, M.D.<sup>1,∗</sup> Alexandra N. Willauer, M.D.<sup>1,3</sup> Christine Y. Zhou, D.O.<sup>1,4</sup> Zhizhou Yang, M.D.<sup>1,5</sup> Pranav Rajpurkar, Ph.D.<sup>1</sup>

<sup>1</sup>Department of Biomedical Informatics, Harvard Medical School, Boston, MA, USA <sup>2</sup>Department of Computer Science, Cornell University, Ithaca, NY, USA   
<sup>3</sup>Division of Gastroenterology, Department of Medicine, Massachusets General Hospital, Boston, MA, USA   
<sup>4</sup>Division of Pulmonary, Critical Care, and Sleep Medicine, University of Cincinnati, Cincinnati, OH, USA <sup>5</sup>Department of Surgery, Massachusets General Hospital, Boston, MA, USA

## ABSTRACT

We used medical image synthesis to create a challenging multimodal evaluation framework that exposes hallucination behaviors in frontier models. Evaluating multimodal reasoning has historically been dificult because existing medical datasets rarely pair different image modalities, such as radiographs and external clinical photographs. To address this gap, we introduce Trick-or-Treat, an adversarial multimodal medical benchmark that pairs real chest radiographs with synthetic clinical photographs to challenge models’ cross– modal reasoning. For each of a wide range of condi tions, we created two pairs of images, where the pairs share a single chest radiograph but contain diferent ex ternal photographs. In matched pairs, the chest radiograph and photograph both contain findings suggesting the same disease (e.g., varicella pneumonia suggested by lung opacities and crusted skin lesions seen in chickenpox). In mismatched pairs, the photograph depicts an alternative clinical finding that would not lead to the same unifying diagnosis (e.g. replacing crusted skin le sions with ordinary acne). Synthetic photograph generation allows clinical findings to be varied while the person’s features and photographic seting stay consistent, reducing confounding. In an investigation of three frontier models, we found that models halluci nated clinically significant photograph findings in 45– 57.5% of mismatched pairs, up from 25–37.5% for the same photographs shown alone. When faced with mismatched pairs, models tended to ignore mismatches and instead hallucinated findings that beter matched the radiograph (e.g. misreading acne as crusted skin lesions to justify diagnosing chickenpox). Trick-or-Treat both demonstrates concerning error patern in frontier models’ medical reasoning and provides a proof of con-

cept for the use of synthetic images in adversarial multimodal evaluation.

## 1 INTRODUCTION

Medical vision-language models must be able to to reason over diferent medical image modalities and recognize when they tell a unified clinical story. As an example, consider a chest radiograph suggesting viral pneumonia and an external photograph of red, fluidfilled vesicles with varying levels of crusting, as seen in chickenpox. A model should notice that this pair of images is strongly suggestive a diagnosis of varciella pneumonia. In contrast, some medical images may be less of a “match,” providing no clear support for a unifying diagnosis. Accordingly, a model should not jump to a diagnosis of varicella pneumonia if the same radiograph from above is paired with a photograph of harmless pimples. Models that reason about multiple image modalities must succeed in both scenarios, suggesting a unified diagnosis when the evidence supports it while preserving accuracy and refraining from overdiagnosis in the face of less congruent inputs.

Past work has shown that models struggle to integrate multiple modalities, with conflicting textual cues reducing the accuracy of image interpretation [1]. Recent benchmarks have tested reasoning specifically on multiple image modalities by repurposing radiology teaching cases or specifically collecting multiple ophthalmic image modalities from hospitals [2, 3, 4]. However, these approaches depend on existing case repositories or intensive clinical data acquisition, fundamentally constraining both the number of images available and the kinds ofquestions that can be asked. As a result, performing targeted evaluations across multiple image modalities remains challenging.

In this work, we constructed Trick-or-Treat, a benchmark consisting of 40 real chest radiographs and 80 synthesized clinical photographs, to test multimodal reasoning with limited confounding. We paired each real chest radiograph with two generated clinical photographs: one showing a disease sign that supports a unified diagnosis and one showing an alternative finding that is less supportive. Using synthetic data, we kept the person’s features, pose, and background consistent across both photographs, even as the findings of interest change. Image synthesis also allowed us to present particularly subtle or challenging findings, increasing the dificulty level. This setup enabled the creation of a controlled, adversarial benchmark that probes model weaknesses.

![](images/bce0aee6f97e65d888846a9d0f60a2abcdb5f84a05e858ef453f8f37f5cb6714.jpg)  
Figure 1. Hallucination induced by a mismatched photograph–radiograph pair. In this example, GPT-6 Astra describes a benign port-wine stain correctly from the photo alone (top right), but when the same photo is paired with a chest radiograph suggestive of injury, it misinterprets the mark as flank ecchymosis (botom right). Matched-pair photographs featuring true ecchymosis are interpreted correctly (left)

Our benchmark poses multiple challenges for models. First, they must accurately interpret photographs showing diverse, sometimes subtle, findings. Second, they must maintain accuracy when each photograph is paired with a radiograph, which may or may not suggest a unifying diagnosis together. We found that three frontier models struggled with both elements. When analyzing a group of photographs in isolation without radiographs, models made clinically significant hallucinations in 25–37.5% of cases. When analyzing those same photographs together with mismatched radiographs, errors became even more frequent; models hallucinated clinically significant abnormalities in 45– 57.5% of cases (Figure 1).

This work makes two distinct contributions. The first is empirical: we demonstrate that three leading vision– language models made errors when interpreting clinical photographs, especially when those errors were consistent with contextual information. The second is methodological: we show that synthetic medical data can be used to construct controlled, adversarial tests of multimodal reasoning, despite the relative scarcity of multimodal medical datasets.

## 2 RELATED WORK

## 2.1 Synthetic medical photographs

Synthetic clinical photographs have been developed to address limited training data and variation in disease appearance. DermGAN generates skin-condition images while controlling lesion size, location, and skin color, and evaluates their use in classifier training [5]. Sagers et al. use DALL·E 2 to generate dermatological photographs across skin types and study whether synthetic augmentation improves classification for underrepresented populations [6]. LesionGen extends text-conditioned synthesis with structured dermatological descriptions, enabling generation from clinically meaningful visual atributes and evaluating downstream classification and subgroup performance [7]. Together, these studies establish controllable synthesis of external disease appearances as a practical tool for medical machine learning.

Cross-modal medical image synthesis predates the current generation of multimodal foundation models. Nie et al. generate synthetic CT images from MRI of the same subject using context-aware generative adversarial networks [8]. More recent work constructs richer synthetic patient representations. Chu and Goodell combine language, visual avatars, and generated audio to simulate dificult patient conversations for medical education [9]. XGeM supports synthesis across frontal chest radiographs, lateral chest radiographs, and radiology reports, including generation of a missing view and joint generation of clinically coherent outputs [10]. Its image pairs present diferent radiographic views, rather than diferent clinical modalities. IMCBench pairs real dermatology photographs with synthetic patient profiles and EHR context to evaluate image-grounded medical conversations [11]; the photographs themselves are not generated, and the paired channels are images and records rather than two image types. To the best of our knowledge, ours is the first benchmark to use image synthesis to evaluate models’ medical reasoning on sets of images representing diferent imaging modalities.

## 2.2 Adversarial medical evaluation and hallucination

Medical hallucination benchmarks show why conventional question-answering accuracy is insuficient to establish reliable image interpretation. MedVH tests responses to mismatched images, clinically incorrect questions, absent abnormalities, misleading answer suggestions, and report-generation tasks [12]. Med-HEval organizes hallucinations around visual misinterpretation, knowledge deficiency, and context misalignment, using both closed- and open-ended medical questions to evaluate models and mitigation methods [13].

More recently, Med-R2 evaluates visually grounded reasoning across stages of a clinical workflow and uses adversarial perturbations to test robustness to misleading cues [14]. These benchmarks establish complementary ways to ask whether medical models reject unsupported premises and ground their reasoning in the available evidence.

Synthetic counterfactual images can also expose model shortcuts. RoentMod edits chest radiographs to introduce specified pathology while preserving unrelated anatomy, then uses these interventions to identify reliance on of-target findings and improve model robustness [15]. Our construction applies a related intervention across two image types: we held a real radiograph fixed and generated external photographs with narrow diferences that can reveal model shortcuts.

## 3 METHOD

## 3.1 Matched cases

The Trick-or-Treat benchmark focuses on 40 conditions, each represented by one matched image pair and a second mismatched image pair. Conditions include acute presentations, such as traumatic hemothorax, pulmonary contusion, and varicella pneumonia, as well as chronic or systemic conditions, such as systemic sclerosis, acromegaly, and Marfan syndrome. For each condition, we constructed a matched pair and a mismatched pair, combining a single chest radiograph with two diferent synthetic clinical photographs. In the matched pair, the photograph depicts findings associated with the same disease as the radiograph. In the mismatched pair, the photograph depicts an alternative finding that has litle to no connection with that disease (Figure 2). Importantly, the mismatched photograph findings and radiograph findings can coexist in a patient, particularly one with multiple conditions; they just do not point to a clear, unifying diagnosis.

## 3.2 Real chest radiographs

We selected real chest radiographs from the ReXGradient dataset [16]. All retained radiographs are anteroposterior (AP) or posteroanterior (PA) views. As models’ content safety filters are more likely to reject images ofyounger children, we used radiographs from patients over 18 years of age. 50% of our chest x-rays are labeled as coming from male patients, with the remainder being labeled as female patients. The mean age recorded for patients is 45 years, with a standard deviation of 18 years.

Candidate retrieval involved querying radiologyreport metadata with condition-specific keywords, such as “flame-shaped” for pulmonary Kaposi sarcoma. Candidates were reviewed alongside their reports to confirm a finding compatible with the target condition. Clinicians verified all chest x-rays to ensure they were compatible with the target condition. For “acromegaly” and “osteogenesis imperfecta,” we cropped radiographs to omit incompatible shoulder or arm findings while preserving a plausible height-to-width ratio.

![](images/be34f6ba28e74f0dfd5ea2aab7a5beb7a5445f7ece291a653ba70bed7ec120b8.jpg)  
Figure 2. Benchmark construction for one case (systemic sclerosis). A single real chest radiograph is paired with two synthetic photographs of the same person: a matched one showing a disease sign (microstomia and radial perioral furrowing) and a mismatched one showing an alternative finding (full lips, ordinary perioral lines). Each photograph is captioned on its own and when paired with the radiograph.

## 3.3 Synthetic clinical photographs

We used the gemini-3-pro-image text-to-image model to generate the external findings and reviewed each candidate image, using a human-in-the-loop setup. Where needed, we used diverse strategies for successful generation: 1.) improving prompts with human feedback, 2.) selecting reference images from research or educational sources licensed for noncommercial use, 3.) providing synthetic images from partly successful generation atempts with feedback on what to change, or 4.) suggesting specific coordinates where edits should be focused. To capture a disease sign and an alternative finding involving the same body region, we either generated one image based on the other or generated both images from a shared base image. As a result, both final images depict the same person with a diferent finding. We chose alternative findings with some physical resemblance to the disease signs (e.g., widespread dry skin vs. mechanic’s hands, pimples vs. crusted chickenpox vesicles, purple nail polish vs. cyanosis).

The age and sex of photograph subjects were matched to the radiograph metadata. In one case, clinicians raised concerns that the sex metadata did not match the apparent sex of the patient on a chest radiograph; we ensured that the photo appears androgynous. Ethnicity metadata is generally lacking in our chest radiograph dataset. Only 3 patients are marked as “non-Hispanic,” while the rest have no label. To avoid a previously documented failure mode where a similar model disproportionately prefers generating images of White patients when not otherwise instructed [17], ethnicity was assigned by a reproducible draw over the following ethnic groups, with probabilities approximately reflecting the United States population in 2020 [18]. The resulting distribution ofphotos is 55% White, 17.5% Black, 20.0% Hispanic/Latino, and 7.5% Asian.

Table 1. Per-photograph error rates (%) by model, photograph type, and setting. Rates are over 40 matched and 40 mismatched photographs per model and seting. False Positives (FP), the response describes a clinically significant abnormality the photograph does not contain; False Negative (FN), it fails to report one the photograph does contain; Any, at least one FP or FN. Each value is a percentage over 40 photographs with its exact (Clopper–Pearson) 95% confidence interval in parentheses. Bold marks the higher error rate ofthe two setings, within each model, photograph type, and metric. Adding a radiograph lowers error rates on matched photographs and raises them on mismatched ones.
<table><tr><td></td><td></td><td colspan="3">Matched</td><td colspan="3">Mismatched</td></tr><tr><td>Model</td><td>Setting</td><td>FP</td><td>FN</td><td>Any</td><td>FP</td><td>FN</td><td>Any</td></tr><tr><td>Claude Fable 5.1 Photo only</td><td></td><td>25.0 (12.7–41.2)</td><td>27.5 (14.6–43.9)</td><td>30.0 (16.6–46.5)</td><td>25.0 (12.7–41.2)</td><td>2.5 (0.1–13.2)</td><td>27.5 (14.6–43.9)</td></tr><tr><td></td><td>Photo + radiograph</td><td>17.5 (7.3–32.8)</td><td>12.5 (4.2–26.8)</td><td>20.0 (9.1–35.6)</td><td>45.0 (29.3–61.5)</td><td>2.5 (0.1–13.2)</td><td>47.5 (31.5–63.9)</td></tr><tr><td>GPT-6 Astra</td><td>Photo only</td><td>12.5 (4.2–26.8)</td><td>10.0 (2.8–23.7)</td><td>15.0 (5.7–29.8)</td><td>37.5 (22.7–54.2)</td><td>2.5 (0.1–13.2)</td><td>40.0 (24.9–56.7)</td></tr><tr><td></td><td>Photo + radiograph</td><td>2.5 (0.1–13.2)</td><td>2.5 (0.1–13.2)</td><td>2.5 (0.1–13.2)</td><td>57.5 (40.9–73.0)</td><td>0.0 (0.0–8.8)</td><td>57.5 (40.9–73.0)</td></tr><tr><td>Gemini 4 Argon Photo only</td><td></td><td>10.0 (2.8–23.7)</td><td>7.5 (1.6–20.4)</td><td>12.5 (4.2–26.8)</td><td>35.0 (20.6–51.7)</td><td>2.5 (0.1–13.2)</td><td>37.5 (22.7–54.2)</td></tr><tr><td></td><td>Photo + radiograph</td><td>7.5 (1.6–20.4)</td><td>2.5 (0.1–13.2)</td><td>10.0 (2.8–23.7)</td><td>45.0 (29.3–61.5)</td><td>5.0 (0.6–16.9) 47.5 (31.5–63.9)</td><td></td></tr></table>

Synthetic generation made our paired design possible: it let us construct photographs for the matched and mismatched pairs that closely resemble each other, except for the findings of interest. Iterative rounds of clinician review assessed each pair for the fidelity of the disease sign for matched-pair photographs and the suitability of the alternative finding for mismatched-pair photographs, as well as compatibility with the accompanying chest radiograph. When images were found to be unsuitable, clinician feedback was incorporated into the next round of the generation process. The final version of each question was approved by 1-3 boardcertified physicians specializing in pulmonary disease, orthopedic surgery, or internal medicine.

## 4 EVALUATION

## 4.1 Model task

We instructed models to describe the radiograph and photograph and discuss what they might be suggestive of. We also evaluated a photo-only baseline where models were instructed to repeat the task without the radiograph. Throughout, we labeled these two conditions photo + radiograph and photo only. Both setings used the same extraction and judging procedure; Appendix A provides additional evaluation details.

## 4.2 Primary outcomes

Our outcomes of interest are per-photo false-positive (FP) and false-negative (FN) rates for clinically significant abnormalities. A false positive occurs when the response hallucinates a clinically significant abnormality that is not present in the photograph. A false negative occurs when it misses a clinically significant abnormality that is present. The rubric excludes normal variants and clinically insignificant incidental features. A response may receive both labels if it hallucinates one clinically significant abnormality while missing another.

We scored photo findings with a rubric-based LLM judge, Claude Sonnet 5.5. Before judging, an extraction pass isolated the clinical-photograph descriptions and discarded the radiograph description and diagnostic synthesis. Diagnoses stated elsewhere in the response therefore were not counted as a photographic finding. We did not measure the quality of the predicted diagnoses because many of our image pairs, especially mismatched pairs, could be compatible with several diagnoses or combinations of diagnoses. Table 1 reports photo error rates across 40 matched pairs and 40 mismatched photographs per model. It also includes the photo-only baseline.

## 5 RESULTS

## 5.1 Models make errors describing the photographs alone

In the photo-only baseline, all three models made errors when describing the clinical photographs, frequently omiting findings on the matched-pair photographs and hallucinating extra findings on the mismatched-pair ones. On matched-pair photographs, which often carry more serious abnormalities, every model showed its highest false-negative rates (27.5% for Claude Fable 5.1, 10% for GPT-6 Astra, and 7.5% for Gemini 4 Argon); false negatives only occurred on 2.5% of mismatchedpair photographs. On mismatched-pair photographs, which generally depict normal findings or benign variations, GPT-6 Astra, Gemini 4 Argon and Claude Fable 5.1 showed false-positive rates of 37.5%, 35%, and 25%, respectively; we observed that models tended to invent significant abnormalities when none were present.

![](images/11286903ff7100b5c021100eaff7646a03ce1adf2623347a2eae857ac21a6abc.jpg)  
Figure 3. Per-photograph false-positive and false-negative rates by setting and photograph type. Error rates are in percent, over 40 matched and 40 mismatched photographs per model and seting. In matched pairs, adding radiographs generally decreases errors. In mismatched pairs, adding radiographs generally increases false positives, with minimal impact on false negatives.

## 5.2 A matched radiograph helps while a mismatched one hurts

Pairing the photograph with a radiograph had opposite efects depending on whether the two images point to the same diagnosis (Figure 3). With matched pairs, all three models improved compared to the photo-only baseline: false positives fell (GPT-6 Astra 12.5 to 2.5%, Gemini 4 Argon 10 to 7.5%, Claude Fable 5.1 25 to 17.5%; though pooled across models this decrease did not reach significance, exact McNemar $\mathnormal { p } = 0 . 0 8 )$ and false negatives also fell (GPT-6 Astra 10 to 2.5%, Gemini 4 Argon 7.5 to 2.5%, Claude Fable 5.1 27.5 to 12.5%; exact McNemar pooled across models, $p = 0 . 0 0 3 )$ . On mismatched pairs, false positives instead rose substantially from the photo-only baseline: GPT-6 Astra from 37.5 to 57.5%,

Gemini 4 Argon 35 to 45%, and Claude Fable 5.1 25 to 45%. This increase was significant for GPT-6 Astra and Claude Fable 5.1 individually (exact McNemar, $p = 0 . 0 2$ each) and pooled across models $( p \ < \ 0 . 0 0 1 )$ , but not for Gemini 4 Argon alone $( p \ = \ 0 . 2 2 ) ;$ ; false negatives changed litle (exact McNemar pooled, $p = 1 . 0 )$ . Qualitative inspection of mismatched pairs showed that radiographs specifically tended to elicit photographic hallucinations that supported a matching diagnosis (Figure 4).

## 6 DISCUSSION

We found error paterns in the clinical photograph comprehension of three frontier vision–language models, especially in the presence of a second image modality. First, models missed or hallucinated clinically significant findings even when considering photographs in isolation, afirming the findings of [19]. Multimodality introduces an additional challenge; our results showed that adding a radiograph as context can increase hallucinations of clinically significant photographic abnormalities, extending findings from [1] showing performance degradation from misleading textual context. In this failure mode, photographic findings can be misinterpreted and wrongly used to support a diagnosis suggested by the radiograph. Models thus flaten out complex, diverging evidence from images to invent photographic findings that

Yellow nail syndrome GPT-6 Astra

Claude Fable 5.1

Photo onlyPhoto + radiograph  
![](images/840529108b970fc0e5a53a119579194d21e968c4bdb3bbd73de0c58b304c2eb3.jpg)  
Pancoast tumor Claude Fable 5.1 RADIOGRAPH apical (superior sulcus) lung mass

•“...pupils appear roughly symmetric in this image..."

è “...the right pupil appears smaller (miosis)..."

![](images/c7163e734f77e8876e524ba47280ff3eb6f84b63111b6a312dd9bdcdb75962eb.jpg)  
Hereditary hemorrhagic GPT-6 Astra telangiectasia

RADIOGRAPH right middle lobe nodular opacity

•"No obvious papules, pustules, crusting, or ulceration."

"...small punctate red vascular spots on the tongue and lips...'

![](images/df4b0e9c2e55a42a8f781c67aeb9dd69b00c5d4fec6f194e76b52b74ff7b1184.jpg)  
Systemic sclerosis Gemini 4 Argon RADIOGRAPH bibasilar reticular opacities

“Subtle soft tissue fullness ... without overlying skin erythema or surface changes."

“...thinned lips (microcheilia), and a narrowed oral aperture (microstomia) with subtle radial perioral furrowing."

![](images/f17da400bf51ad23cf78a07971f3bfc01451a0f64eedbc81afbb604c103940ec.jpg)  
Cystic fibrosis Claude Fable 5.1 RADIOGRAPH bilateral tram-track bronchial opacities

•“No clubbing, pitting, or significant onycholysis is evident.'

● “...early digital clubbing...

![](images/d633b6b82c82b0246492f3d64fa6e1adda4c73591bf13232e7800a3d8c17b477.jpg)

## RADIOGRAPH bilateral pleural effusions

“The nails are covered with pale-yellow polish, limiting nail assessment."

"Diffuse pale-yellow, opaque discoloration of the fingernails, which appear thickened and excessively curved..."

![](images/431d91e06fd0873ea707cd17d8db0314eb9625a84e4bc0b94b62f28abc8d9b2b.jpg)  
Antisynthetase syndrome Gemini 4 Argon RADIOGRAPH mild lower-zone reticular fibrosis

•“...prominent xerosis, and superficial desquamation...

“...scaling, and fissuring on the radial aspect ... consistent with "mechanic's hands"..."

![](images/10e057ac6e7e90a33b9254b00b1892acb64a8d761eee5d920500ef35ff325046.jpg)

•"No vesicles ... Lesions appear monomorphic and at a similar stage...

“Many lesions have a central vesicle ... the lesions appear to be at different stages of evolution..."

![](images/b36efef6ec1dfa8d7c8ec0137f9ce05d4351b8813a62b0989d51a770c0bddc58.jpg)  
Löfgren syndrome  
GPT-6 Astra

RADIOGRAPH bilateral hilar lymphadenopathy

“...scattered light-brown macules..."

“...patchy reddish areas and subtle nodular elevations over the anterior shins..."

![](images/be04f76831702b9be97802a85ff638bbc5cd467472838c3e34b38c8fe733fff4.jpg)

“...smooth, dome-shaped subcutaneous nodule..."

“...firm-appearing, skin-colored subcutaneous nodule ... consistent with a rheumatoid nodule..."

Figure 4. Selected examples of photograph findings hallucinated with mismatched radiographs. Each panel shows a mismatched pair’s photograph and the key finding from its radiograph. Below are excerpts from the named model’s descriptions of the photograph, first from the photo-only baseline (green circle) and then with the radiograph added (red diamond); emphasis marks the newly hallucinated finding. In every panel, the added radiograph steers hallucinations towards a matching diagnosis.

fit the same diagnosis as the radiograph.

Our work focused on synthesizing clinical photographs while using real radiographs as context, but future work should synthesize radiographs or even extend this setup to other modalities. Additionally, all synthetic photographs passed clinician review, but further analyses could check for any subtle generation artifacts or errors that are not visible to the human eye. Additionally, we looked only at photograph caption quality and used Claude Sonnet 5.5 as a judge. Future work might also consider radiograph caption quality and overall diagnostic accuracy and incorporate clinician judgments. Abstention, hyperparameter selection, and multi-agent approaches may mitigate the issues we found and should be evaluated.

Essentially, our work suggests that strong performance on a medical image modality does not guarantee that a model can safely combine it with other modalities. Consistent with [3], we found that naively applying models to complex multimodal questions may create risks that are not apparent from their performance on individual images or conventional medical reasoning benchmarks. Our benchmark does not measure downstream outcomes, but it exposes a mechanism by which a persuasive multimodal explanation could mislead its reader. Reliable multimodal reasoning must preserve uncertainty and consider that findings from diferent modalities may be unrelated or suggestive of multiple co-occurring diagnoses.

The broader methodological contribution is a method for synthetically constructing the cases needed to expose such errors. Synthetic generation lets us depict a disease sign and an alternative finding while keeping the person and photographic seting consistent. It would be infeasible to achieve this controlled comparison with naturally collected patient data. The cases in this study revealed a recurring weakness across models, providing a concrete example of synthetic data’s usefulness in adversarial multimodal evaluation.

## 7 ACKNOWLEDGEMENTS

The Rajpurkar Lab’s work is supported by the Biswas Family Foundation’s Transformative Computational Biology Grant in collaboration with the Milken Institute.

## REFERENCES

[1] Buckley TA, Diao JA, Srivastava CN, Brodeur PG, Rajpurkar P, Rodman A, Manrai AK. Multimodal foundation models exploit text to make medical image predictions. Nature Communications 17:7475, 2026. https: //doi.org/10.1038/s41467-026-74207-5

[2] Xu R, Huang Z, Wei Y, Zhou X, Xu Z, Liu T, Jiang Z, Zhou SK. MedAtlas: Evaluating LLMs for Multi-Round, Multi-Task Medical Reasoning Across Diverse Imaging Modalities and Clinical Text. Proc. AAAI Conference on Artificial Intelligence, 40(44):38048–38056, 2026. https: //doi.org/10.1609/aaai.v40i44.41142

[3] Yao Z, Wang B, Zhang Y, Wang J, Xia I, Tang Z, Han S, Ouyang F, Yang Z, Cohan A, Yu H. Medical Thinking with Multiple Images. Proc. International Conference on Learning Representations (ICLR), 2026. https://arxiv. org/abs/2604.16506

[4] Wang G, Zhong Z, Zhou Y, Li Y, Wu E, Cheah WP, Qu R, Ren J, Shen L. X-PCR: A Benchmark for Crossmodality Progressive Clinical Reasoning in Ophthalmic Diagnosis. Proc. IEEE/CVF Conference on Computer Vision and Patern Recognition (CVPR), pp. 33110–33120, 2026. https://arxiv.org/abs/2604.20350

[5] Ghorbani A, Natarajan V, Coz D, Liu Y. DermGAN: Synthetic Generation of Clinical Skin Images with Pathology. Proc. ML for Health NeurIPS Workshop, PMLR 116:155–170, 2020. https://proceedings.mlr. press/v116/ghorbani20a.html

[6] Sagers LW, Diao JA, Groh M, Rajpurkar P, Adamson AS, Manrai AK. Improving dermatology classifiers across populations using images generated by large difusion models. SAIL Health, 2023. https://sail.health/wp-content/uploads/ 2023/11/Abstract-20 Diao.pdf

[7] Fayyad J, Bayasi N, Yu Z, Najjaran H. Lesion-Gen: A Concept-Guided Difusion Model for Dermatology Image Synthesis. Proc. ISIC Workshop at MICCAI, pp. 3–12, 2025. https://doi.org/10.1007/ 978-3-032-05825-6 1

[8] Nie D, Trullo R, Lian J, Petitjean C, Ruan S, Wang Q, Shen D. Medical Image Synthesis with Context-Aware Generative Adversarial Networks. Proc. MIC-CAI, LNCS 10435:417–425, 2017. https://doi.org/ 10.1007/978-3-319-66179-7 48

[9] Chu SN, Goodell AJ. Synthetic Patients: Simulating Difficult Conversations with Multimodal Generative AI for Medical Education. arXiv:2405.19941, 2024. https:// arxiv.org/abs/2405.19941

[10] Molino D, Di Feola F, Faiella E, Fazzini D, Santucci D, Shen L, Guarrasi V, Soda P. XGeM: A Multi-Prompt Foundation Model for Multimodal Medical Data Generation. arXiv:2501.04614, 2025. https://arxiv.org/ abs/2501.04614

[11] Xenochristou M, et al. IMCBench: A benchmark for multimodal LLMs in Image-grounded Medical Conversations. arXiv:2606.28556, 2026. https://arxiv.org/ abs/2606.28556

[12] Gu Z, Yin C, Liu F, Zhang P. MedVH: Towards Systematic Evaluation of Hallucination for Large Vision Language Models in the Medical Context. arXiv:2407.02730, 2024. https://arxiv.org/abs/2407.02730

[13] Chang A, Huang L, Bhatia P, Kass-Hout T, Ma F, Xiao C. MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models. arXiv:2503.02157, 2025. https://arxiv.org/abs/ 2503.02157

[14] Ma W, Niu F, Fan Z, Xiao Z, LiuJ, Liu Z. Med-R2: An Adversarial Benchmark for Evidence-Grounded Reasoning in Medical VLMs. arXiv:2605.24492, 2026. https: //arxiv.org/abs/2605.24492

[15] Cooke LH, Jung M, Brendel JM, Kerkovits NM, Foldyna B, Lu MT, Raghu VK. RoentMod: A Synthetic Chest X-Ray Modification Model to Identify and Correct Image Interpretation Model Shortcuts. arXiv:2509.08640, 2025. https://arxiv.org/abs/2509.08640

[16] Zhang X, Acosta JN, Miller J, Huang O, Rajpurkar P. ReXGradient-160K: A Large-Scale Publicly Available Dataset of Chest Radiographs with Free-text Reports. arXiv:2505.00228, 2025. https://arxiv.org/ abs/2505.00228

[17] Banerjee O, Willauer AN, Rajpurkar P. ReXavatars: Evaluating Generated Patient Images for Clinical Realism and Reliability. Proc. Machine Learning for Healthcare, PMLR 340:1–21, 2026.

[18] U.S. Census Bureau. QuickFacts: United States —Race and Hispanic origin. Accessed 2026. https://www.census.gov/quickfacts/fact/ table/US/PST045225

[19] Banerjee O, Kim SE, Willauer AN, Kernbach JM, Alomaish AR, Alghamdi RAS, Alomaish HR, Baharoon M, Zhang X, Acosta JN, Zhou C, Rajpurkar P. ReX-InTheWild: A Unified Benchmark for Medical Photograph Understanding. arXiv:2603.19517, 2026. https: //arxiv.org/abs/2603.19517

## A ADDITIONAL EVALUATION DETAILS

We obtained model responses with these prompts:

## Prompt: photograph + radiograph

these findings might be suggestive of.

## Prompt: photograph only

The following is a clinical photograph. Report your   
findings under these exact labels, each on its own   
line:   
PHOTO FINDINGS: describe the findings and any clinical   
abnormalities visible in the photograph.   
SUGGESTIVE OF: what these findings might be suggestive   
of.

Scoring used an extract-then-judge procedure. For photo FP and FN labels, an extraction pass retained only the “PHOTO FINDINGS” section and removed radiograph descriptions and “SUGGESTIVE OF” diagnostic hypotheses. The judge, Claude Sonnet 5.5, compared the extracted text with the ground-truth photo caption, using the same rubric for all photographs. The rubric labels hallucinated clinically significant abnormalities as FP and missed clinically significant abnormalities as FN; normal variants and clinically insignificant incidental features are excluded. The prompt is as follows:

## Judge prompt: false-positive and false-negative rubric

A clinical photograph actually shows the following   
finding(s):   
{caption}   
A model described the clinical photograph’s findings   
as:   
{photo}   
A ‘‘clinically significant abnormality that indicates   
disease’’ means a finding a clinician would treat as   
a sign of a specific disease or pathology -- NOT a   
normal variant and NOT an incidental benign feature.   
Comparing the model’s described photo findings   
against what the photograph actually shows:   
FALSE POSITIVE: YES or NO   
YES if the model asserts a clinically significant,   
disease-indicating abnormality that is NOT actually   
present (a hallucinated disease sign).   
FALSE NEGATIVE: YES or NO   
YES if the photograph actually contains   
a clinically significant, disease-indicating   
abnormality that the model FAILED to report (a missed   
disease sign). If the photograph contains no such   
disease-indicating abnormality, answer NO.   
ONE LINE: brief justification

We queried three models through their native APIs: GPT-6 Astra (Azure OpenAI deployment gpt-6-astra, chat-completions API version 2024-12-01-preview), Gemini 4 Argon (gemini-4-argon on Google Vertex AI), and Claude Fable 5.1 (claude-fable-5-1, Anthropic Messages API, anthropic-version 2023-06-01). All requests used each provider’s default sampling setings (no temperature, top-�, or seed was set). Maximum output tokens were 4,000 for GPT-6 Astra, the provider default for Gemini 4 Argon, and 12,000 for paired (two-image) requests and 6,000 for photo-only requests for Claude Fable 5.1; each request was retried on HTTP 429/5xx errors. Images were sent inline as base64 PNGs, used at native resolution when the longest side was at most 1568 pixels and downscaled to 1568 pixels otherwise. We then scored each response with an extract-then-judge procedure: an extraction pass isolated the PHOTO FINDINGS section, and the judge, claude-sonnet-5-5 (Anthropic Messages API, maximum 500 output tokens, default sampling), compared it against the ground-truth photograph caption using the rubric above to assign the false-positive and false-negative labels.