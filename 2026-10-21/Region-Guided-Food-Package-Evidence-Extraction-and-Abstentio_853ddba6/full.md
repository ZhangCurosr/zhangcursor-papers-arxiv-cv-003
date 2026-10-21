# Region-Guided Food-Package Evidence Extraction and Abstention-Aware Draft R3337 Screening with Selective VLM Review

Mayimunah Nagayi<sup>1\*[0009−0009−9241−3059]</sup>, Alice Scaria Khan<sup>2[0000−0003−1011−3693]</sup>, Tamryn Frank<sup>2[0000−0002−5180−9171]</sup>, Rina Swart<sup>3[0000−0002−7786−3117]</sup>, and Clement Nyirenda<sup>1,4[0000−0002−4181−0478]</sup>

<sup>1</sup> Department of Computer Science, University of the Western Cape, Cape Town, South Africa

\* 4163113@myuwc.ac.za

2 School of Public Health, University of the Western Cape, Cape Town, South Africa {askhan,tfrank}@uwc.ac.za

3 Department of Dietetics and Nutrition, University of the Western Cape, Cape Town, South Africa

rswart@uwc.ac.za

4 eResearch Ofice, University of the Western Cape, Cape Town, South Africa cnyirenda@uwc.ac.za

Abstract. Applying South Africa’s draft R3337 warning criteria to foodpackage photographs requires nutrition, ingredient and product-form evidence. This study presents a region-guided evidence extraction and abstention-aware screening workflow for draft R3337. A three-class detector localises nutrition information panels, ingredient lists and productidentity regions, after which PaddleOCR and structured extraction recover nutrient values, ingredient evidence and product-form information. Recovered evidence is aggregated and assessed using deterministic draft R3337 rules. When evidence is incomplete, uncertain or conflicting, the workflow abstains through Review or Insuficient evidence rather than forcing an automatic outcome. For qualifying unresolved cases, a selective vision-language model (VLM) review stage using Qwen3-VL 8B Instruct reassesses unresolved evidence. Accepted evidence is merged and the same deterministic rules are reapplied. The workflow was applied to 442 products represented by 3,929 images. Evaluation used a manual validation set of 65 products. Before selective VLM review, 31 of 60 decidable products received automatic outcomes and 30 were correct, giving 51.7% coverage and 96.8% accuracy among automatic outcomes (selective accuracy). Selective VLM review resolved eight additional products, increasing coverage to 65.0%. Of 39 automatic outcomes, 38 were correct, giving 97.4% selective accuracy. The single false warning was retained from the rules-only workflow, while no unsafe warning omissions occurred after selective review. Evidence-level evaluation nevertheless identified incorrect VLM-recovered nutrition, ingredient and product-form evidence. These findings show that selective VLM review can recover unresolved evidence, while deterministic screening and explicit abstention remain necessary for reliable image-based draft R3337 screening.

Keywords: Front-of-pack warning labels · Draft R3337 · Food-package image analysis · Structured evidence extraction · Abstention-aware screening · Vision-language models

## 1 Introduction

Front-of-pack labelling (FOPL) is intended to make important nutrition information more visible to consumers at the point of purchase. The World Health Organization [20] identifies FOPL as a policy tool that can support healthier food choices. Bopape et al. [5] found that South African consumers viewed warning labels positively and described them as concise and understandable. The Department of Health [7] published the Draft Regulations Relating to the Labelling and Advertising of Foodstufs (R3337), which propose front-of-pack warning symbols for prepackaged foods that meet specified conditions relating to total sugar, saturated fat, sodium and artificial sweeteners. Applying these criteria requires nutrient values, evidence of added nutrient sources, product form and, where applicable, nutrition information after preparation.

Previous South African draft R3337 assessments used captured or manually coded product information [2,14], whereas direct screening from food-package photographs first requires the relevant evidence to be recovered from the images. The information required for screening may appear across several photographs and package surfaces, while small text, dense layouts, glare, curved packaging and other image conditions can afect text recognition. Recognising package text alone is also insuficient. Nutrient names must be associated with the correct values, units and reference bases, while ingredient and product-form evidence must be recovered and combined at product level. Image-derived evidence may be incomplete, uncertain or conflicting, making it unsafe to force an automatic screening outcome when the required evidence cannot be established reliably.

This study develops an integrated region-guided, abstention-aware workflow for screening food-package images against draft R3337. It combines region localisation, optical character recognition (OCR), structured evidence extraction, deterministic screening and selective vision-language model (VLM) review, while retaining Review or Insuficient evidence when evidence remains unresolved. The main contributions of this study are as follows:

1. A region-guided workflow for recovering and integrating nutrition, ingredient and product-form evidence from real South African food-package photographs.

2. A deterministic implementation of the draft R3337 warning criteria with explicit abstention when image-derived evidence is incomplete, uncertain or conflicting.

3. A selective VLM evidence-review stage that reassesses qualifying unresolved evidence before deterministic rule reapplication.

4. A product-level evaluation of the proposed screening workflow against manual reference outcomes, together with an evidence-level evaluation of the reliability of VLM-recovered nutrition, ingredient and product-form evidence.

The remainder of this paper is organised as follows. Section 2 reviews related work and identifies the research gap. Section 3 describes the datasets. Section 4 presents the methodology. Section 5 presents the results and evaluation. Section 6 discusses the findings, Section 7 presents the study limitations and Section 8 concludes the paper.

## 2 Related Work

## 2.1 Front-of-Pack Warning Labels and Draft R3337 Assessment

Nutrient profiling models classify packaged foods according to their nutritional composition and nutrients of concern. Frank et al. [8] applied four nutrient profiling models to the South African packaged-food supply and found that classifications difered across the models. Draft R3337 specifies warning criteria based on total sugar, saturated fat, sodium and artificial sweeteners, together with conditions relating to added nutrient sources, product form and, where applicable, prepared-product values [7]. Abdool Karim et al. [2] used coded package and nutrition variables from 6,768 photographed products to determine warninglabel eligibility and assess on-package marketing under draft R3337. Mamabolo et al. [14] assessed 271 breakfast porridges and cereals and found that warninglabel requirements varied with preparation method and grain type. These studies show the importance of nutrient, ingredient and preparation-related evidence, but did not automate its recovery from food-package photographs.

## 2.2 Food-Package Evidence Recovery

Food-package photographs present challenges that difer from scanned documents, including curved and reflective surfaces, visually crowded layouts and uncontrolled image conditions. Guimarães et al. [11] reviewed grocery-label detection and recognition and identified product variation, limited datasets and uncontrolled image conditions. Nagayi et al. [16] evaluated four OCR systems on real South African food-package images and identified dense layouts, multilingual text, glare, curved surfaces and small text as packaging-specific challenges. Their evaluation focused on text recovery, while Nagayi et al. [15] combined region detection, OCR-derived sodium evidence, product identification and deterministic R214 screening.

Region localisation can reduce irrelevant package content before text recognition. Saputra et al. [18] used a MobileNetV1-based detector to locate nutritionlabel regions before applying preprocessing and Tesseract OCR. Their evaluation was limited to ten images and focused on extracting text from nutrition labels. Fuentes et al. [9] developed an OCR-based system that preserved spatial relationships between nutrient names and quantities when extracting nutritional information from Spanish food products. Their approach demonstrates the importance of retaining layout information when associating nutrient labels with the correct values. Pettersson et al. [17] combined visual features with OCRderived text for fine-grained grocery-product recognition and found that the two information sources complemented each other. Food-package evidence recovery therefore involves more than isolated text recognition and can depend on region localisation, spatial association and integration of visual and textual information.

## 2.3 Vision-Language Models, Reliability and Selective Prediction

Vision-language models (VLMs) can perform text recognition, visual question answering and structured information extraction from image and text inputs. Liu et al. [13] evaluated multimodal models using OCRBench across text-related image tasks and reported useful capabilities alongside continuing weaknesses in multilingual and other challenging text settings. Assiri et al. [3] evaluated multimodal language models for extracting nutritional elements and values from bilingual food-label images, demonstrating their relevance to food-package information extraction. The Qwen3-VL family described by Bai et al. [4] supports combined image and text inputs and includes an 8-billion-parameter variant. However, performance on general benchmarks or related extraction tasks does not establish reliable recovery of exact nutrient values, units, reference bases and ingredient evidence from real food-package images. Selective prediction provides an approach for handling uncertain model outputs. Geifman and El-Yaniv [10] introduced SelectiveNet, which jointly considers prediction and rejection while examining coverage and error risk. Whitehead et al. [19] applied abstention to visual question answering, where withholding an answer was preferred when confidence was insuficient. These studies show the value of allowing uncertain cases to remain unresolved rather than requiring a prediction for every input. In evidence-extraction settings, however, abstention does not establish that the evidence returned is factually correct. The reliability of recovered evidence therefore remains important alongside coverage and abstention.

## 2.4 Research Gap

Previous work has addressed draft R3337 assessment, food-package OCR and region localisation, structured nutrition extraction, multimodal food-label extraction and selective prediction. Nagayi et al. [15] integrated image-derived evidence for R214 sodium screening, but not the broader nutrition, ingredient and product-form evidence required for draft R3337. The reviewed studies do not evaluate draft R3337 screening with explicit abstention and selective VLM review. The contribution is therefore the system-level integration and evaluation of these components rather than the development of a new detector or VLM architecture.

## 3 Dataset Description

## 3.1 Food-Package Screening Dataset

The study uses real-world food-package photographs from the South African Nutrition Facts Panel project of the Department of Dietetics and Nutrition at the

University of the Western Cape (UWC). The photographs were collected in 2023 by trained personnel. The screening dataset contains 442 packaged-food products represented by 3,929 full-package photographs. Each product is represented by between three and twenty-seven photographs, averaging 8.9 photographs per product. The photographs capture diferent package surfaces, meaning that product identity, nutrition information, ingredient information and preparation details may appear in diferent images of the same product. The product is the unit of analysis. All photographs are linked to their corresponding product and are treated as sources of information rather than as independent samples. The screening dataset is separate from the annotated dataset used to develop and evaluate the region detector described in Section 3.2.

## 3.2 Detector-Development Dataset

For detector development, 664 annotated food-package images from 176 products were used, containing 885 bounding boxes across three classes, namely nutrition information panels, ingredient lists and product-identity regions. The annotations comprised 201 nutrition-panel regions, 274 ingredient-list regions and 410 product-identity regions. The dataset was divided at product level into training, validation and held-out test subsets with an approximately 80%, 10% and 10% split. The training subset contained 141 products (80.1%), 524 images and 700 bounding boxes. The validation subset contained 18 products (10.2%), 89 images and 120 bounding boxes, while the held-out test subset contained 17 products (9.7%), 51 images and 65 bounding boxes. No product appeared in more than one subset, ensuring that evaluated products were not represented during training.

## 4 Methodology

## 4.1 Overall Workflow

The proposed methodology consists of an initial region-guided screening workflow followed by selective VLM evidence review for qualifying unresolved evidence. Figure 1 presents the initial stages from the food-package images described in Section 3 through region detection, optical character recognition (OCR), evidence extraction and deterministic draft R3337 screening.

1. Input. The workflow receives the full-package photographs associated with each product.

2. Region detection. A three-class detector localises nutrition information panels, ingredient lists and product-identity regions.

3. OCR and evidence extraction. Region-guided OCR recovers nutrition, ingredient and product-form evidence, while unresolved evidence is retained.

4. Deterministic draft R3337 screening. Product-level evidence is assessed using the encoded rules without forcing outcomes when required evidence remains uncertain or unavailable.

![](images/c91f2ba0e57941543f3e9dc2668ebff07eb2bcfdce24b4fd8bb763a435c49166.jpg)  
Fig. 1. Initial region-guided evidence extraction and deterministic draft R3337 screening workflow.

5. Initial rules-only outcomes. Products are assigned Warning criteria met, No warning criteria triggered, Review or Insuficient evidence, with the latter two representing abstention.

Qualifying unresolved evidence proceeds to the selective VLM evidence-review stage described in Section 4.5 and illustrated in Figure 2. Accepted evidence is merged before the same deterministic draft R3337 rules are reapplied.

## 4.2 Region Detection and Optical Character Recognition

The annotated detector-development dataset described in Section 3.2 is used to train a three-class You Only Look Once version 26 small (YOLO26s) object detector from pretrained weights using Ultralytics version 8.4.89 [12]. Training uses automatic optimiser selection, deterministic execution with a seed of 0 and automatic mixed-precision computation. The checkpoint selected at epoch 77 based on validation performance is retained for evaluation on the held-out test subset and inference on the food-package screening dataset. During inference, images are processed at an input size of 640 × 640 pixels using a confidence threshold of 0.25. Each retained detection preserves the product and source-image identifiers, region class, bounding-box coordinates and detector confidence.

OCR is applied to the detected nutrition-panel, ingredient-list and productidentity regions using PaddleOCR version 3.5.0 with the graphics processing unit (GPU) build of PaddlePaddle version 3.3.0 [6]. The configuration uses the PP-OCRv5\_server\_det text-detection model, the en\_PP-OCRv5\_mobile\_rec English recognition model and the PP-LCNet\_x1\_0\_textline\_ori text-line-orientation model, with text-line orientation enabled. Recognised text from ingredient-list and product-identity regions is retained with its product and source-image information, while nutrition-panel outputs retain recognised tokens and their coordinates for structured extraction. Each nutrition-panel region is processed in its original form and in a second rendering produced by adding approximately 10% padding and enlarging the padded region by a factor of two. Source photographs, detected crops, alternative renderings and OCR outputs remain linked to the originating product and are not treated as independent product samples.

## 4.3 Structured Evidence Extraction and Product-Level Aggregation

Nutrition evidence. Recognised nutrition-panel tokens and their coordinates are processed using a header-aware, coordinate-based extractor that associates nutrient labels with numerical values, units and reference bases. Candidate values are identified using nutrient-label matching, row proximity and table-header interpretation. The warning-related targets are total sugar, saturated fat and sodium, while total fat is retained as supporting information rather than assessed as a draft R3337 warning component. Consistency checks across the original and padded renderings consider nutrient identity, numerical value, unit and reference basis. Evidence remains unresolved when the row association, value, unit or reference basis cannot be established reliably, or when renderings or nutrition panels produce conflicts. Salt values are retained separately and are not compared directly with sodium thresholds. When an explicit sodium value is unavailable, salt-derived sodium evidence is accepted only when the conversion, unit and applicable per-100 reference basis are clear. Preparation information is considered where available, but evidence remains unresolved when the applicable basis cannot be established reliably.

Ingredient evidence. Recognised text from ingredient-list regions is grouped by product and searched using deterministic term dictionaries and text-matching rules. The categories cover added sugar, added sodium, added saturated-fat sources and artificial sweeteners. Added saturated-fat evidence is restricted to qualifying dictionary terms and does not include every ingredient that may naturally contain saturated fat. Each category is assigned Yes, Not detected or Uncertain. Yes indicates that a configured qualifying term is identified in recognised text. Not detected indicates that no configured term is found and is not interpreted as proof that the ingredient is absent. Uncertain is assigned when recognised text or the match cannot support a clear result. Missing ingredientlist regions and empty OCR outputs remain unavailable evidence.

Product form and product-level aggregation. Product form is determined from product-identity information and package text, as the applicable draft R3337 nutrient thresholds difer for solid and liquid products. The product-form assessment returns Solid, Liquid or Review, with Review retained when the available information is insuficient or conflicting. Recovered nutrition, ingredient and product-form evidence is aggregated at product level. Where evidence remains missing, uncertain or conflicting across available sources, that uncertainty is preserved rather than resolved by forcing a value or classification. The product-level evidence is passed to deterministic draft R3337 screening in Section 4.4.

## 4.4 Deterministic Draft R3337 Screening and Abstention

The product-level evidence produced in Section 4.3 is assessed using deterministic rules that operationalise the draft R3337 warning criteria. For consistent implementation, the study applies nutrient-specific conjunctive rules to total sugar, saturated fat and sodium. A nutrient-related warning requires both qualifying added-source evidence for the corresponding nutrient and a nutrient value greater than the applicable solid or liquid threshold. The implementation uses strict greater-than comparisons consistent with the draft R3337 nutrient cutofs [7].

For solid products, the encoded thresholds are:

total sugar greater than 10.0 g per 100 g.

saturated fat greater than 4.0 g per 100 g.

– sodium greater than 400 mg per 100 g.

For liquid products, the encoded thresholds are:

– total sugar greater than 5.0 g per 100 ml.

– saturated fat greater than 3.0 g per 100 ml.

– sodium greater than 100 mg per 100 ml.

The artificial-sweetener criterion requires explicit evidence of an added artificial sweetener and does not require a corresponding nutrient-table value [7]. The accepted nutrition, ingredient and product-form evidence is then mapped to one of four product-level screening outcomes. Warning criteria met indicates that the available and accepted evidence satisfies at least one encoded draft R3337 warning criterion. It does not indicate that a warning symbol was visually detected on the package. No warning criteria triggered indicates that the available and accepted evidence is suficient for screening but does not establish an encoded warning criterion and is not interpreted as confirmation of legal compliance. Review indicates that relevant evidence is available but remains uncertain, ambiguous or conflicting, while Insuficient evidence indicates that evidence required to apply the screening rules is unavailable. Review and Insuficient evidence represent abstention outcomes rather than forced automatic decisions. Qualifying unresolved evidence from the abstention outcomes proceeds to the selective VLM evidence-review stage described in Section 4.5.

## 4.5 Selective Vision-Language Model Evidence Review

Existing automatic rules-only outcomes are not reconsidered. Figure 2 presents the six-stage selective VLM evidence-review process for qualifying rules-only abstentions.

![](images/7ebee01c306f2c7b74beb1a1c50b35249819bcfcb77205b713d82671706e8b9b.jpg)  
Fig. 2. Selective VLM evidence-review and rule-reapplication workflow.

1. Select eligible cases. Products assigned Review or Insuficient evidence proceed to VLM review only when they contain qualifying unresolved nutrition, ingredient or product-form evidence.

2. Select image evidence. Relevant nutrition-panel, ingredient-list or productidentity crops are selected for the unresolved evidence. The VLM reassesses this evidence rather than making an independent product-level screening decision.

3. Task-specific VLM review. The same Qwen3-VL 8B Instruct model is used through separate task-specific review routines for nutrition evidence, ingredient evidence and product form. The nutrition review examines unresolved nutrient values, units and reference bases. The ingredient review examines added sugar, added sodium, added saturated-fat sources and artificial sweeteners, while the product-form review assesses whether package evidence supports a solid or liquid classification.

4. Parse and merge VLM evidence. Each task-specific review returns a structured machine-readable response. Malformed, incomplete or unresolved responses are rejected, and structurally usable responses are not assumed to be factually correct. Under the evidence-merge rules, accepted VLM evidence supplements only absent or unresolved fields, while existing accepted rulesonly evidence is retained.

5. Reapply rules. The merged evidence is reassessed using the same deterministic draft R3337 rules in Section 4.4. Existing evidence requirements are retained, so a VLM-recovered nutrient value cannot establish a sugar, sodium or saturated-fat warning without the corresponding qualifying ingredient evidence.

6. Post-review rule-based outcomes. Rule reapplication produces the postreview product-level outcome. Unresolved cases continue to abstain rather than being forced into an automatic outcome. Post-review outcomes are retained separately from initial rules-only outcomes for evaluation.

During development, Phi-3.5-Vision Instruct [1] and Qwen3-VL 8B Instruct [4] were examined. Neither model was trained or fine-tuned in this study. Both pretrained models were used for inference. Phi-3.5-Vision Instruct was used only during exploratory testing and was not retained after producing confident but incorrect associations between nutrition-table rows and values. Qwen3-VL 8B Instruct was retained for the final evidence-review experiments and deployed locally through Ollama. Manual reference outcomes are withheld during VLM review, evidence merging and deterministic rule reapplication and are used only during evaluation. The factual correctness of structurally usable VLM evidence is evaluated separately using the evidence-level measures in Section 4.6.

## 4.6 Manual Validation and Evaluation Measures

A 65-product manual validation set is constructed using stratified sampling across the four initial rules-only outcome categories. The set contains 22 randomly selected products assigned Warning criteria met, nine products assigned No warning criteria triggered, 17 randomly selected products assigned Review and 17 randomly selected products assigned Insuficient evidence. These rulesonly outcomes are used only as sampling strata rather than as manual reference outcomes. The validation set is not intended to estimate the prevalence of the four outcome categories in the complete dataset. Manual reference review uses all available full-package photographs for each product and applies the same operational draft R3337 screening protocol described in Section 4.4. Reference outcomes are assigned from the complete visible package evidence.

Where a complete and readable ingredient declaration establishes that none of the qualifying ingredient conditions is present, manual review can assign No warning criteria triggered even when the corresponding nutrient values are unavailable. This difers from the automated Not detected ingredient result in Section 4.3, which indicates only that no configured qualifying term was recovered from OCR-derived text and is not treated as proof of absence.

Manual review assigns 37 products to Warning criteria met, 23 to No warning criteria triggered and five to Insuficient evidence, resulting in 60 manually decidable products. The five unresolved products lack the evidence required for a definitive reference outcome after all available package images are examined. Product-specific evidence sources, reasons for unresolved outcomes and evidencelevel reference fields for nutrition values, ingredient evidence and product form are recorded. All manual reference information is withheld from the rules-only workflow, selective VLM review and deterministic rule reapplication and is used only during evaluation.

Product-level evaluation is performed on the 60 manually decidable products using the following measures. To reflect uncertainty associated with the validation-set size, 95% Wilson confidence intervals (CIs) are reported for coverage and selective accuracy.

– Coverage: the proportion of products assigned an automatic outcome, namely Warning criteria met or No warning criteria triggered.

– Selective accuracy: the number of correct automatic outcomes divided by the total number of automatic outcomes, with abstentions excluded from the denominator.

– Abstention rate: the proportion of products assigned Review or Insuficient evidence.

– False warning: an automated Warning criteria met outcome when the manual reference is No warning criteria triggered.

– Unsafe warning omission: an automated No warning criteria triggered outcome when the manual reference is Warning criteria met.

Evidence-level VLM evaluation assesses usability and correctness for nutrition, ingredient and product-form evidence. Nutrition correctness requires agreement on the nutrient value, unit and reference basis, with strict exact matching also reported. Ingredient correctness is calculated among usable category decisions and product-form correctness among usable solid or liquid classifications.

## 4.7 Computational Environment

All computational experiments are conducted on the UWC High-Performance Computing Cluster. The GPU-enabled compute environment uses an AMD EPYC-Milan processor and an NVIDIA L4 GPU with approximately 23 GB of GPU memory. Evidence extraction, product-level aggregation and deterministic screening are implemented in Python 3.10.

## 5 Results and Evaluation

## 5.1 Region Detection and Evidence Availability

On the validation and held-out test subsets, respectively, detector precision was 0.846 and 0.889, while recall was 0.774 and 0.952. Mean average precision (mAP) at an intersection-over-union (IoU) threshold of 0.50 was 0.810 and 0.947, while mAP averaged across IoU thresholds from 0.50 to 0.95 was 0.635 and 0.757. Validation and test results are reported separately, as validation performance informed checkpoint selection while the product-disjoint test subset was reserved for final evaluation. Full-dataset inference produced 4,552 crops from 3,106 images, comprising 1,976 product-identity, 1,383 ingredient-list and 1,193 nutritionpanel regions. At least one target region was detected for every product.

OCR processing was completed for all 4,552 detected crops, although not every crop contained usable recognised text. Processing the original and padded nutrition-panel renderings produced 2,386 crop-rendering records containing 82,682 recognised tokens. Tokens were recovered from 2,363 renderings, while 23 contained no recognised tokens. Structured nutrition evidence was recovered for 334 products. Accepted total-sugar evidence was available for 170 products, total-fat for 157, saturated-fat for 156, sodium for 158 and salt for four. Usable recognised ingredient text was available for 386 products. Fifty-four products had no detected ingredient-list region, while two had a detected ingredient region with no recognised text. Product-form assessment classified 222 products as Solid and 51 as Liquid, while 169 remained Review.

## 5.2 Rules-Only Product-Level Results

Across all 442 products, the rules-only workflow produced 83 automatic outcomes, an operational decision rate of 18.8%, and abstained on 359 products. These full-dataset results describe workflow behaviour rather than accuracy, as the complete dataset was not manually validated. Among the 60 manually decidable validation products, the workflow produced 31 automatic outcomes, of which 30 agreed with the manual reference. This corresponded to 51.7% coverage (95% CI 39.3–63.8%) and 96.8% selective accuracy (95% CI 83.8–99.4%), while missing, uncertain or conflicting required evidence contributed to a 48.3% abstention rate. One false warning and no unsafe warning omissions were observed.

## 5.3 Product-Level Results After Selective VLM Review

Of the 359 products on which the rules-only workflow abstained, accepted Qwen3- VL evidence was incorporated for 232 products under the evidence-merge rules. Existing automatic rules-only outcomes remained unchanged. After evidence merging and rule reapplication, the product-level outcome changed for 144 products. Ninety-four abstentions became automatic outcomes, while 50 products moved between Review and Insuficient evidence. Across the dataset, the operational decision rate increased from 18.8% to 40.0%. These changes describe workflow behaviour rather than accuracy, as the dataset was not manually validated.

Among the 60 manually decidable validation products, selective VLM review resolved eight additional cases, increasing coverage to 65.0% (95% CI 52.4– 75.8%) and reducing the abstention rate from 48.3% to 35.0%. Overall, 38 of 39 automatic outcomes agreed with the manual reference, giving 97.4% selective accuracy (95% CI 86.8–99.5%). The single false warning remained from the rules-only workflow, with no unsafe warning omissions after selective review. Tables 1 and 2 summarise the full-dataset outcome changes and validation-set performance, respectively.

Table 1. Full-dataset outcomes before and after selective VLM evidence review.
<table><tr><td>Outcome or measure</td><td>Rules-only</td><td>After selective VLM review</td></tr><tr><td>WARNING CRITERIA MET</td><td>71</td><td>115</td></tr><tr><td>NO WARNING CRITERIA TRIGGERED</td><td>12</td><td>62</td></tr><tr><td>REVIEW</td><td>149</td><td>109</td></tr><tr><td>INSUFFICIENT EVIDENCE</td><td>210</td><td>156</td></tr><tr><td>Automatic outcomes</td><td>83 (18.8%)</td><td>177 (40.0%)</td></tr><tr><td>Abstentions</td><td>359 (81.2%)</td><td>265 (60.0%)</td></tr></table>

Table 2. Performance on the 60 manually decidable validation products before and after selective VLM review.
<table><tr><td>Measure</td><td>Rules-only</td><td>After selective VLM review</td></tr><tr><td>Coverage</td><td>31/60 (51.7%)</td><td>39/60 (65.0%)</td></tr><tr><td>Selective accuracy</td><td>30/31 (96.8%)</td><td>38/39 (97.4%)</td></tr><tr><td>Abstention rate</td><td>29/60 (48.3%)</td><td>21/60 (35.0%)</td></tr><tr><td>False warnings</td><td>1</td><td>1</td></tr><tr><td>Unsafe warning omissions</td><td>0</td><td>0</td></tr><tr><td>Newly resolved products</td><td></td><td>8</td></tr></table>

## 5.4 Evidence-Level Reliability of VLM-Recovered Evidence

Tables 3 and 4 report the usability and correctness of Qwen3-VL 8B Instruct evidence, showing that structurally usable outputs were not always factually correct.

Table 3. Nutrition-evidence reliability of Qwen3-VL 8B Instruct.
<table><tr><td>Evaluation</td><td>Usable/evaluated</td><td>Correct</td><td>Correctness</td></tr><tr><td>Nutrition overall</td><td>61/76</td><td>47/61</td><td>77.0%</td></tr><tr><td>Strict exact match</td><td>61/76</td><td>46/61</td><td>75.4%</td></tr><tr><td>Total sugar</td><td>21</td><td>14/21</td><td>66.7%</td></tr><tr><td>Saturated fat</td><td>20</td><td>15/20</td><td>75.0%</td></tr><tr><td>Sodium</td><td>20</td><td>18/20</td><td>90.0%</td></tr></table>

## 5.5 Case Analysis

Selected validation products illustrate how VLM-recovered evidence afected product-level outcomes and are not intended to represent the full validation set or complete dataset. For SaveMor Coconut Biscuits, the VLM recovered total sugar as 17.8 g per 100 g, matching the manual reference value, while qualifying added-sugar evidence supported a Warning criteria met outcome through the sugar criterion. SPAR Split Peas illustrates that a correct product-level outcome can still contain incorrect recovered evidence. The VLM recovered total sugar as 0.7 g per 100 g, while manual review recorded 2.7 g per 100 g. Both values were below the applicable solid-product threshold, so the error did not alter the final No warning criteria triggered outcome. A similar sugar-extraction error could afect the product-level outcome if the correct value exceeded the applicable threshold and qualifying added-sugar evidence was present. These cases show why product-level performance should be interpreted alongside evidence-level correctness.

Table 4. Ingredient and product-form evidence reliability of Qwen3-VL 8B Instruct.
<table><tr><td>Evidence category</td><td>Usable/decided</td><td>Correct</td><td>Correctness</td></tr><tr><td>Ingredient evidence overall</td><td>75/108</td><td>66/75</td><td>88.0%</td></tr><tr><td>Added sugar</td><td>21</td><td>19/21</td><td>90.5%</td></tr><tr><td>Added sodium</td><td>21</td><td>17/21</td><td>81.0%</td></tr><tr><td>Added saturated fat</td><td>16</td><td>13/16</td><td>81.2%</td></tr><tr><td>Artificial sweetener</td><td>17</td><td>17/17</td><td>100.0%</td></tr><tr><td>Product form</td><td>20/33</td><td>19/20</td><td>95.0%</td></tr></table>

## 6 Discussion

## 6.1 Evidence Recovery Limited Automatic Coverage

Despite strong held-out detector performance, automatic coverage was limited by incomplete nutrition, ingredient and product-form evidence. These limitations are consistent with the packaging-specific OCR challenges reported by Nagayi et al. [16] and the broader image-condition challenges identified by Guimarães et al. [11]. Manual review also encountered blurred sections and visible skew, although their efect on individual extraction errors was not isolated. Region localisation restricted OCR to detected package areas, following the detectionbefore-recognition approach used by Saputra et al. [18]. Structured screening also required nutrient labels to be associated with the correct values, units and reference bases, reflecting the importance of spatial association demonstrated by Fuentes et al. [9], while ingredient and product-form uncertainty introduced further abstentions. These findings support retaining unresolved cases rather than treating missing or uncertain evidence as a negative result.

## 6.2 Coverage, Selective Accuracy and Abstention

Selective VLM review resolved eight additional validation products without introducing a new false warning or unsafe warning omission. The small number of resolved cases, however, limits broader conclusions about this improvement. Geifman and El-Yaniv [10] examined coverage and error risk in selective prediction, while Whitehead et al. [19] showed the value of abstaining when confidence is insuficient in visual question answering. In this workflow, abstention is determined by deterministic evidence requirements rather than a probabilistic confidence score, so confidence calibration was not evaluated. The rules-only and post-review results provide two operating points for the coverage-abstention trade-of, but not a full risk-coverage curve. Coverage should therefore be interpreted together with selective accuracy and error type.

## 6.3 Benefits and Risks of Vision-Language Evidence Review

Selective VLM review recovered useful evidence for some unresolved cases, but reliability varied across nutrition, ingredient and product-form tasks. Nutrition evidence showed the lowest correctness of the three evaluated evidence types, indicating that exact nutrient recovery remained more dificult than the broader ingredient and product-form decisions in this evaluation. The case analysis also showed that an incorrect recovered nutrient value could coexist with a correct product-level outcome when the error did not change the applicable warning criterion. This distinction is important, as high selective accuracy at product level does not establish that all supporting evidence is correct. Liu et al. [13] similarly reported useful multimodal text-recognition capabilities alongside continuing weaknesses on challenging text-related image tasks, while Assiri et al. [3] demonstrated the applicability of multimodal models to nutrition extraction from food-label images. The causes of individual VLM errors were not isolated in this study, so the observed errors cannot be attributed to a single failure source. The evidence-level results therefore support retaining the VLM as a selective evidence reviewer rather than an independent product-level decision-maker.

## 6.4 Relationship to Manual Draft R3337 Assessment

Unlike the draft R3337 assessments of Abdool Karim et al. [2] and Mamabolo et al. [14], which used captured or manually coded product information, this study addresses evidence recovery directly from food-package photographs. The workflow therefore supports screening rather than automated regulatory assessment, and a No warning criteria triggered outcome does not establish legal compliance.

## 7 Limitations

The stratified 65-product validation set was not designed to estimate outcome prevalence, and five products remained unresolved after all package images had been reviewed. The same set supported product-level and evidence-level evaluation, while post-review outcomes for the 442-product dataset were not manually verified. Structurally usable VLM outputs could contain incorrect evidence. Broader evaluation using an independent dataset and systematic error-sensitivity analysis is therefore needed to assess robustness and generalisation. Evidence availability difered across nutrition, ingredient and product-form information, with limitations in nutrient-row association, ingredient recovery and unclear preparation or product-form evidence. Salt evidence was particularly scarce. The study used one South African food-package image collection, so generalisation to other products, retailers, languages and image conditions remains untested. The study did not include a full-image OCR baseline and therefore does not isolate the efect of region localisation. R3337 remained in draft form at the time of the study, and its criteria may change before the regulations are finalised. The final evidence-review evaluation used one retained VLM, so the findings do not establish generalisability across diferent VLMs. The dataset cannot be publicly released as it forms part of an ongoing project, while the code is available on request from the corresponding author.

## 8 Conclusion

This study developed and evaluated a region-guided, abstention-aware workflow for draft R3337 screening from South African food-package photographs. On the 60 manually decidable validation products, rules-only screening achieved 51.7% coverage and 96.8% selective accuracy, while selective Qwen3-VL 8B Instruct review increased coverage to 65.0% and selective accuracy to 97.4%, with one false warning and no unsafe warning omissions. Evidence-level evaluation showed that structurally usable VLM outputs could still contain incorrect supporting evidence. Selective VLM review can therefore supplement unresolved evidence, while deterministic rule reapplication and explicit abstention remain important. Further work should evaluate cross-checking between OCR-derived and VLM-recovered evidence, agreement across multiple VLMs, and improved nutrition-row association, ingredient recovery and product-form determination. The workflow provides decision support rather than final legal or regulatory determinations.

## References

1. Abdin, M., et al.: Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 (2024)

2. Abdool Karim, S., Frank, T., Khan, A.S., Tlhako, M.G., Joni, S.K., Swart, E.C.: An assessment of compliance with proposed regulations to restrict on-package marketing of packaged foods to improve nutrition in South Africa. BMC Nutrition 11, 17 (2025). https://doi.org/10.1186/s40795-025-01007-3

3. Assiri, F.Y., Alahmadi, M.D., Almuashi, M.A., Almansour, A.M.: Extract nutritional information from bilingual food labels using large language models. Journal of Imaging 11(8), 271 (2025). https://doi.org/10.3390/jimaging11080271

4. Bai, S., et al.: Qwen3-VL technical report. arXiv preprint arXiv:2511.21631 (2025)

5. Bopape, M., et al.: South African consumers’ perceptions of front-of-package warning labels on unhealthy foods and drinks. PLOS ONE 16(9), e0257626 (2021). https://doi.org/10.1371/journal.pone.0257626

6. Cui, C., et al.: PaddleOCR 3.0 technical report. arXiv preprint arXiv:2507.05595 (2025)

7. Department of Health, Republic of South Africa: Regulations relating to the labelling and advertising of foodstufs. Government Notice No. R.

3337, Government Gazette No. 48460, Regulation Gazette No. 11575, 21 April 2023 (2023), https://www.health.gov.za/wp-content/uploads/2023/05/ R3337-Draft-Labelling-Regulations-21-April-2023-sc-compressed-1.pdf

8. Frank, T., Ng, S.W., Miles, D.R., Swart, E.C.: Applying and comparing various nutrient profiling models against the packaged food supply in South Africa. Public Health Nutrition 25(8), 2296–2307 (2022). https://doi.org/10.1017/S1368980022000374

9. Fuentes, J.M., Paredes, R., Fulladosa, E., Giró, M.d.M., Claret, A.: Automatic transcription system for nutritional information charts of Spanish food products. In: Proceedings of VISAPP 2022. pp. 411–418. SciTePress (2022). https://doi.org/10.5220/0010827600003124

10. Geifman, Y., El-Yaniv, R.: SelectiveNet: A deep neural network with an integrated reject option. In: Proceedings of the 36th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 97, pp. 2151–2159. PMLR (2019)

11. Guimarães, V., Nascimento, J., Viana, P., Carvalho, P.: A review of recent advances and challenges in grocery label detection and recognition. Applied Sciences 13(5), 2871 (2023). https://doi.org/10.3390/app13052871

12. Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C., Kalfaoglu, M.E.: Ultralytics YOLO26: Unified real-time end-to-end vision models. arXiv preprint arXiv:2606.03748 (2026)

13. Liu, Y., et al.: OCRBench: On the hidden mystery of OCR in large multimodal models. Science China Information Sciences 67, 220102 (2024). https://doi.org/10.1007/s11432-024-4235-6

14. Mamabolo, M.K., et al.: Requirement for front-of-pack warning labels in South African porridges and cereals varies depending on preparation method and grain type. South African Journal of Clinical Nutrition 39(1), 9–17 (2026). https://doi.org/10.1080/16070658.2025.2603783

15. Nagayi, M., Scaria Khan, A., Frank, T., Swart, R., Nyirenda, C.: A conservative OCR-enabled workflow for R214 sodium screening of South African packaged foods. arXiv preprint arXiv:2609.15427 (2026)

16. Nagayi, M., Scaria Khan, A., Frank, T., Swart, R., Nyirenda, C.: Evaluating OCR performance on food packaging labels in South Africa. In: Artificial Intelligence Research. Communications in Computer and Information Science, vol. 2784, pp. 127–143. Springer (2026). https://doi.org/10.1007/978-3-032-11733-5\_8

17. Pettersson, T., Riveiro, M., Löfström, T.: Multimodal fine-grained grocery product recognition using image and OCR text. Machine Vision and Applications 35, 79 (2024). https://doi.org/10.1007/s00138-024-01549-9

18. Saputra, T.M.E., Susanto, A., Carmelita, B.J.: Implementation of Tesseract OCR and bounding box for text extraction on food nutrition labels. Building of Informatics, Technology and Science (BITS) 6(3), 1403–1412 (2024). https://doi.org/10.47065/bits.v6i3.6107

19. Whitehead, S., et al.: Reliable visual question answering: Abstain rather than answer incorrectly. In: Computer Vision – ECCV 2022. Lecture Notes in Computer Science, vol. 13696, pp. 148–166. Springer (2022). https://doi.org/10.1007/978-3- 031-20059-5\_9

20. World Health Organization: Guiding principles and framework manual for frontof-pack labelling for promoting healthy diets (2019), https://www.who.int/ publications/m/item/guidingprinciples-labelling-promoting-healthydiet