# Push–Pull Attentional Anchoring for Difusion Concept Erasure

Nattanat Chatthee<sup>ว</sup>, Tagon Sompong<sup>ว</sup>, Ekapol Chuangsuwanich<sup>ฬ</sup>, and Supasorn Suwajanakorn<sup>ว</sup>

<sup>ว</sup>VISTEC, Thailand <sup>ฬ</sup>Chulalongkorn University, Thailand push-pull-concept-eraser.github.io

![](images/b1a9ad66c5b42005f25d9d66e44a4a6dda60b909e90a281371509a9940f0b3b7.jpg)  
Fig. 1: Given a textual concept, our method erases the target concept while preserving non-target concepts, achieving a favorable erasure–preservation trade-of with controllable erasure strength and applicability to modern architectures such as FLUX.

Abstract. Rapid advances in difusion models have raised concerns about privacy, harmful content, and copyright violations. Concept erasure methods aim to address these risks by removing undesirable concepts from pretrained models without full retraining. However, existing approaches often degrade non-target generation quality through heuristic strength scaling (e.g., classifier-free guidance or negative prompt scaling), leading to uncontrolled and excessive semantic drift. To address this, we propose Push–Pull Attentional Anchoring (PPAA), a mechanism in the cross-attention space that displaces target representations while anchoring the scale of erasure via bounded cosine-similarity constraints. By using a relative similarity ratio instead of an absolute diference, our adaptive threshold represents a meaningful percentage of semantic retention, facilitating a single value to be applied uniformly across layers. We conduct extensive experiments across multiple concept categories, including celebrities, artistic styles, nudity, and intellectual property. Our method achieves favorable trade-ofs against prior approaches on Stable Difusion v1.4 and demonstrates its applicability to modern architectures such as FLUX.

## 1 Introduction

Rapid advances in text-to-image difusion models have achieved impressive generative capabilities, but they have also raised critical concerns regarding privacy, harmful content, and copyright infringement. Concept erasure methods aim to mitigate these risks by selectively unlearning “target” concepts from a pretrained difusion model, avoiding the high cost of full model retraining. Driven by the discovery that early “guidance-based” methods hide concepts rather than destroying them and leave models vulnerable to adversarial recovery [23], recent state-ofthe-art research has actively pursued “destruction-based” removal to erase concepts as deeply and robustly as possible [25, 36]. However, this growing focus on adversarial robustness often highlights a challenging trade-of with semantic preservation: the deeper the erasure, the higher the collateral damage to nontarget generation quality.

To mitigate this trade-of, our study revisits the equally critical requirement of preserving non-target generation quality by examining the components of modern robust erasure methods. Recent state-of-the-art frameworks, such as STEREO [36] and SUMA [25], employ an iterative two-stage design: (i) an adversarial discovery stage, which searches for prompts that can recover the erased concept, and (ii) an erasure stage, which updates the model weights to suppress the discovered targets. While Stage 1 efectively deters prompt-based attacks and improves robustness, existing Stage 2 erasure objectives tend to induce collateral damage. This largely stems from their reliance on uncontrolled parameter updates or rigid static mappings, directly inherited from earlier ap proaches [10,11,17,24], which excessively warp the local semantic neighborhood. Although existing methods can preserve broad, general-domain concepts (e.g., from MS-COCO [21]), our study reveals that the more challenging preservation of closely related, same-domain peers, such as “Will Smith” when erasing “Barack Obama,” remains unsolved.

Since adversarial prompt discovery (Stage 1) functions as a distinct, orthogonal module, we focus on the erasure objective, e.g., those used in Stage 2 [25,36] or as a standalone method [10,11,17,24], to isolate and resolve the preservation bottleneck. Following the standard setup [11,17,23], which maps an erased concept to a user-provided generic “anchor” concept, we seek to finetune the model to suppress the conditional generation of the target concept. Our work prioritizes preserving closely related peer concepts, which are unavailable during optimization, with the goal of improving the overall erasure-preservation trade-of.

To this end, we propose Push–Pull Attentional Anchoring (PPAA), a crossattention mechanism that displaces target representations while explicitly anchoring the scale of erasure via bounded cosine-similarity constraints. Unlike objectives that rely on unbounded or rigid absolute-distance mappings [11, 24, 25], our mechanism introduces a margin that operates on the relative cosine ratio between the updated representation, its original embedding, and a generic reference. Because this relative formulation takes into account each layer’s original target–anchor similarity, it yields an adaptive, concept-aware distance threshold that can be applied uniformly across layers. The resulting bounded pushpull objective enables suficient target-concept displacement to evade target prompts while limiting unintended semantic modification. Furthermore, as our formulation can still serve as a Stage 2 erasure objective in the modern frameworks [25,36], it can be combined with their adversarial discovery stage to further improve robustness.

We conduct extensive experiments across diverse concept categories, including celebrities, artistic styles, nudity, and intellectual property. Our evaluations explicitly measure both general-domain utility and strict same-domain preservation, demonstrating that Push–Pull Attentional Anchoring achieves a favorable erasure-preservation trade-of compared to existing state-of-the-art methods.

## 2 Related Work

## 2.1 Concept Erasure for Difusion Models

Early concept erasure methods [10,11,17,24] focus on fine-tuning difusion models to suppress the generation of specific target concepts by modifying the denoising process or its intermediate representations. ESD [10] explicitly steers the denoising trajectory of the target concept away from its original path by applying negative guidance [2,7,14,16]. In contrast, AC [17] maps the denoising trajectory of the target concept to that of a more generic concept. MACE [24] and UCE [11] extend this idea by performing generic concept mapping in the key-value space of cross-attention layers rather than on the denoising trajectory. However, because the target concept and its mapped or negatively guided counterpart can be distant in representation space, as also observed in [25], these methods generally require substantial parameter updates to bridge this gap, which can significantly degrade the preservation of non-target concepts.

To address this limitation, SUMA [25] leverages Textual Inversion (TI) [9] by using the final (late-step) TI embedding of the target concept and encouraging it to align with earlier-step TI embeddings obtained along the same TI optimization trajectory. These early-step embeddings are intended to capture more generic concept representations while remaining close to the target in embedding space. As a result, this alignment typically requires smaller weight modifications, leading to improved preservation of non-target concepts. However, using earlystep TI embeddings as reference concepts may not be reliable in certain cases. In particular, these embeddings can rapidly converge toward the target concept itself, which limits erasure efectiveness [25].

Unlike approaches that rely on explicitly defined reference concepts for mapping, DUO [26] applies preference-based optimization [30] within the difusion denoising process. Because this optimization operates in the denoising space, which primarily captures low-level visual structure rather than high-level semantic representations [31], it encourages less accurate denoising for the target concept without explicitly enforcing semantic change. As a result, the optimization can produce outputs that are visually degraded yet semantically unchanged, leading to inefective concept erasure.

## 2.2 Adversarial Attacks for Concept Erasure

Despite progress in concept erasure methods, recent adversarial attacks have successfully recovered supposedly erased concepts, highlighting the need for improved robustness. Recent studies [8, 23, 27, 32, 38, 40] demonstrate that supposedly erased concepts can be recovered through adversarial attacks, exposing vulnerabilities in unlearned difusion models. In particular, [23] provides a comprehensive evaluation of adversarial robustness using training-free probing techniques such as In-Painting Probing [1] and Partial Difusion Completion [34], which attempt to extract residual knowledge from erased models. These analyses reveal that many erasure approaches still retain traces of the removed concepts. Beyond probing-based attacks, several works [8, 27, 40] introduce optimizationbased attacks that search for adversarial prompts designed to recover the erased concept. These methods formulate prompt search as an optimization problem that minimizes the discrepancy between the denoising predictions of the erased model and those of the original model. Although our method does not explicitly optimize for adversarial robustness, we evaluate our approach under existing adversarial attack protocols to assess its robustness.

## 3 Proposed Method

Our objective is to erase a target concept by directly updating model weights, rather than relying on easily bypassed guidance or auxiliary modules, while minimizing collateral damage to non-target concepts. Determining the ideal replacement for an erasure target is inherently subjective; for example, erased “Barack Obama” could be mapped to “Person,” “African American,” or “U.S. president,” depending on the desired level of generality. Thus, following [11, 17, 25], we assume that the generic replacement concept is provided by the user. However, non-target peer concepts (e.g., “Will Smith”) that we intend to preserve remain unavailable during optimization and are reserved strictly for evaluation.

## 3.1 Semantic Modification Constraint Losses

We propose a semantic push–pull mechanism to drive the erased concept away from its original semantic representation while preserving shared or generic semantics. Following [11, 17, 24, 25], we apply these constraints to the outputs of the key and value projections in cross-attention, which encode semantically meaningful representations [17, 18, 24, 25]. We denote the initial cross-attention projection weights as $W _ { 0 }$ and the updated weights as W. For ease of presentation, we first present the loss formulation assuming the concept to be erased is represented by a single token embedding $p _ { e } \in \mathbb { R } ^ { d }$ , and the generic embedding by $p _ { g } \in \mathbb { R } ^ { d }$ , where d is the token embedding dimension. We next explain our push–pull objectives along with two preservation objectives adapted from prior studies [11, 26].

Semantic Push Loss $( \mathcal { L } _ { \mathrm { p u s h } } )$ ensures that the updated representation $W p _ { e }$ suficiently deviates from the original $W _ { 0 } p _ { e }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p u s h } } ( W , W _ { 0 } , p _ { e } ) = \operatorname* { m a x } \left( \sin ( W p _ { e } , W _ { 0 } p _ { e } ) - m _ { \mathrm { p u s h } } , 0 \right) , } \end{array}\tag{1}
$$

where $m _ { \mathrm { p u s h } } \in [ 0 , 1 ]$ is the margin parameter, and sim $( \cdot , \cdot )$ denotes cosine similarity. The loss penalizes the model as long as the similarity between the optimized and original representations exceeds $m _ { \mathrm { p u s h } }$ , efectively enforcing a minimum degree of semantic divergence.

Semantic Pull Loss $( \mathcal { L } _ { \mathrm { p u l l } } )$ The push objective does not explicitly regulate the semantic destination of the update. As a result, the modified weight $W$ may suppress not only erased concept-specific information but also broader shared semantics, leading to collateral damage. For example, naively erasing “Barack Obama” may suppress “Person,” “African American,” “Man,” or background context and degrade non-target concepts such as “Will Smith.” To regulate this behavior, we introduce a pull constraint that anchors the updated representation $W p _ { e }$ toward its corresponding generic representation $W _ { 0 } p _ { g } \mathrm { ; }$

$$
\mathcal { L } _ { \mathrm { p u l l } } ( W , W _ { 0 } , p _ { e } , p _ { g } ) = \operatorname* { m a x } \left( m _ { \mathrm { p u l l } } - \frac { \mathrm { s i m } ( W p _ { e } , W _ { 0 } p _ { g } ) } { \mathrm { s i m } ( W _ { 0 } p _ { e } , W _ { 0 } p _ { g } ) } , 0 \right) ,\tag{2}
$$

where $m _ { \mathrm { p u l l } }$ is the margin threshold for the pull constraint. Importantly, the objective is formulated as a ratio; similarity is normalized against its initial baseline, allowing $m _ { \mathrm { p u l l } }$ to function as a percentage-based threshold that can be uniformly applied across layers. The baseline denominator, sim $( W _ { 0 } p _ { e } , W _ { 0 } p _ { g } )$ , empirically remains positive and safely above zero. We provide these empirical statistics, along with a variant that guarantees a positive denominator in Appendix A.1.

To motivate this relative design, consider the concepts “Nurse” and “Street Performer” relative to the generic category “Professional.” If “Nurse” is initially 0.9-similar to “Professional,” while “Street Performer” is only 0.4-similar, enforcing the same absolute similarity threshold would be overly strict for the latter while too lenient for the former. By using a ratio, both concepts are evaluated relative to their own starting points. Setting $m _ { \mathrm { p u l l } } = 0 . 8$ therefore requires each concept to retain at least 80% of its original proximity to the generic concept, yielding a concept-aware formulation. Table 3 empirically shows that initial target–anchor similarities do vary widely across evaluated concepts.

Note that the Semantic Push Loss can also be viewed as operating on a percentage ratio, where its denominator sim $( W _ { 0 } p _ { e } , W _ { 0 } p _ { e } )$ is identically one, simplifying the objective to Equation 1. This provides a unified perspective on both push and pull objectives.

Semantic Preservation Loss $( \mathcal { L } _ { \mathrm { p r e s e r v e } } )$ : Inspired by [11], we minimize the deviation of generic embeddings $p _ { g }$ to prevent broader semantic drift. Unlike the $\ell _ { 2 }$ objective in [11], we use a cosine loss to be consistent with our directional view of semantic alignment in the push–pull constraints. We ensure the optimized projection $W ^ { k } p _ { g }$ remains aligned with its original representation $W _ { 0 } ^ { k } p _ { g } \mathrm { . }$

$$
\mathcal { L } _ { \mathrm { p r e s e r v e } } ( W ^ { k } , W _ { 0 } ^ { k } , p _ { g } ) = 1 - \mathrm { s i m } ( W ^ { k } p _ { g } , W _ { 0 } ^ { k } p _ { g } ) .\tag{3}
$$

Denoising Preservation Loss $\left( \mathcal { L } _ { \mathrm { d e n o i s e } } \right)$ : Following DUO [26], we ensure the model’s fundamental denoising capabilities remain intact by aligning the noise predictions at the terminal timestep T. Specifically, we enforce consistency under the null prompt $p _ { \mathrm { n u l l } }$ to maintain the overall denoising capability.

$$
\mathcal { L } _ { \mathrm { d e n o i s e } } = \mathbb { E } _ { { x _ { T } } \sim \mathcal { N } ( 0 , \mathbf { I } ) } \left[ \lVert \epsilon _ { \theta } ( { x _ { T } } , p _ { \mathrm { n u l l } } ) - \epsilon _ { 0 } ( { x _ { T } } , p _ { \mathrm { n u l l } } ) \rVert _ { 2 } ^ { 2 } \right] .\tag{4}
$$

## 3.2 Total Objective Function

The final objective minimizes the overall loss $\mathcal { L } _ { \mathrm { t o t a l } }$ , which balances the semantic modification and preservation.

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { t o t a l } } = \frac { 1 } { 2 L } \sum _ { l = 1 } ^ { L } \sum _ { j \in \{ K , V \} } \left[ \lambda _ { 1 } \mathcal { L } _ { \mathrm { p u s h } } ^ { ( l , j ) } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { p u l l } } ^ { ( l , j ) } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { p r e s e r v e } } ^ { ( l , j ) } \right] + \lambda _ { 4 } \mathcal { L } _ { \mathrm { d e n o i s e } } ,\tag{5}
$$

where $\mathcal { L } _ { * } ^ { ( l , j ) }$ denotes the specific loss term computed for layer l and the projection type $j \in \{ K , V \}$ . In each such term, the generic weight arguments $W _ { 0 }$ and $W$ are substituted with the layer-specific initial weights $\bar { W } _ { 0 } ^ { ( l , j ) }$ and trainable weights $W ^ { ( l , j ) }$ , respectively. In practice, when the target concept spans multiple tokens, we compute the loss independently for each token as well as the EOS token, and then take their average as the final loss. We jointly optimize these key and value projection matrices while keeping all other model parameters fixed.

## 4 Experiments

## 4.1 Experimental Setup

Dataset. Following [25], we evaluate our erasure method across four concept categories: Celebrity, Artistic $S t y l e .$ , Intellectual Property Entities and Nudity. We adopt a modified evaluation set to provide broader and more balanced coverage within each category. In contrast to [25], which evaluates celebrities using three white male public figures and assesses artistic style using only Van Gogh, resulting in narrow demographic and stylistic coverage, we construct more representative concept sets across all categories. Specifically, the Celebrity category includes Barack Obama, Rihanna, Margot Robbie, and David Beckham. The Artistic Style category includes Van Gogh, Picasso, Claude Monet, and Jackson Pollock. The Intellectual Property (IP) Entities category includes Mickey Mouse, R2D2, Grumpy Cat, and MacBook. For the Nudity category, we erase only the concept naked person but evaluate the efectiveness of erasure on related concepts such as naked person, naked woman, naked man, nude person, nude woman, and nude man. These categories provide a diverse and practically relevant evaluation of concept erasure.

Baselines. We focus our comparison on concept erasure approaches that directly edit model weights, ensuring that the erasure remains efective even under full access to the edited model. Accordingly, we exclude methods that rely on detachable auxiliary modules [20, 37], as they operate under diferent access assumptions and their efects can be trivially undone by removing these additional components. Following this principle, we compare our method with representative approaches across diferent optimization strategies, including negative-guidancebased methods (ESD [10]), mapping-based methods (AC [17], MACE [24], UCE [11], SUMA [25]) and preference optimization-based methods (DUO [26]). We also include STEREO [36] as it represents the state-of-the-art performance in adversarial robustness. We adopt the oficial implementations for MACE, ESD, DUO, and STEREO. AC is re-implemented based on the ESD codebase. For SUMA, Stage 1 is implemented following the STEREO repository and the authors’ guidelines, while Stage 2 uses the authors’ provided implementation. We use the recommended hyperparameter configurations for all baselines. For baselines with an explicit erasure-strength parameter, we report the erasure– preservation trade-of obtained by varying the corresponding parameter. For fair comparison, we ensure that baselines do not access the preservation concepts used for evaluation. For methods that require preservation concepts during training, we instead provide the corresponding anchor concepts.

## Evaluation Metrics. We evaluate performance across three key dimensions.

1. Erasure Efectiveness: To assess whether a target concept remains recognizable after erasure, we generate images conditioned on the concept and analyze them using pretrained classifiers. Following [24, 25], for Celebrity erasure, we utilize the GIPHY Celebrity Detector (GCD) [12] to identify the presence of specific individuals. For Nudity erasure, we use the NudeNet detector [28] as the classifier and adopt the same NudeNet label set used in [26]. For Artistic Style, we employ a Perception Encoder (PE) [5], a state-of-the-art text-image model, to perform CLIP-style zero-shot classification [17, 24, 29] across all 1,119 artist labels from the WikiArt dataset [33]. Following the verification protocol in [24], we confirm that PE achieves an accuracy exceeding 99% on our selected artistic concepts, supporting its suitability for artistic style recognition. For IP Entities, following [25], we employ LLAVA-7B [22] using the template: “Does this image contain {object} ? Answer Yes or No.” The resulting binary prediction is used to determine whether the erased entity is still present.

2. Preservation Efectiveness: We evaluate preservation by measuring the distributional discrepancy between samples generated by the erased model and those generated by the original model under the same preservation prompts. We adopt CMMD [15] as the evaluation metric, as it better aligns with human perceptual judgments and provides more stable estimates than FID [13,15]. A lower CMMD indicates stronger preservation of the preservation concepts. We consider two preservation settings. General-domain preservation evaluates overall generative quality using all 5,000 captions from the validation split of the COCO dataset [21]. Same-domain preservation evaluates whether concepts semantically related to the erased target remain preserved. For example, when erasing “Barack Obama,” we evaluate other celebrity identities such as “Will Smith.” For celebrity and artistic style erasure, we construct same-domain preservation sets using 50 celebrity concepts following [24] and 50 artistic styles sampled from WikiArt [33] and the style set of [24]. For Nudity and IP, we report only general-domain preservation, as no standardized set of semantically comparable concepts is available. Additional details are provided in Appendix B.1.

3. Erasure Robustness: Following [25, 26, 36], we evaluate adversarial robustness on the nudity-erased model using two prompt-based optimization attacks, UnlearnDif [40] and CCE [27]. This setting is particularly challenging because nudity-related content can be expressed through many semantically related prompts (e.g., sexual, pornographic, or explicit), making it more vulnerable to prompt-based attacks than other erasure categories [26]. For both UnlearnDif and CCE, we report the attack success rate (ASR) on the target concept after prompt optimization, where lower ASR indicates stronger erasure robustness. For UnlearnDif, we additionally report the Pre-Attack Success Rate (Pre-ASR), which measures the success rate before applying the attack, and the average attack time (Avg Time), where longer attack times indicate that the attack requires more optimization iterations.

Implementation details. We implement our method on Stable Difusion 1.4 (SD1.4) using the ESD codebase [10]. The model is trained for 1,000 optimization steps using the Adam optimizer, while updating only the cross-attention K and V projection layers, following prior works [11, 17]. For hyperparameter selection, we fix $\lambda _ { 1 } = \lambda _ { 2 } = \lambda _ { 4 } = 1$ and select $\lambda _ { 3 } = 3 2$ and $m _ { \mathrm { p u l l } } = 0 . 4 0$ based on a separate validation set (details in the Appendix). The only remaining hyperparameter $m _ { \mathrm { p u s h } }$ serves as a trade-of knob that adjusts the balance between erasure and preservation; we vary this parameter to produce our performance trade-of curves. The same configuration is applied across all concept categories to ensure fairness and avoid category-specific overfitting. While additional coeficients could be adjusted in practice, we intentionally keep them fixed to maintain a simple and controlled evaluation protocol. We report variations of $m _ { \mathrm { p u s h } }$ when applicable.

## 4.2 Experimental Results

Quantitative Results. As shown in Figure 2, our method achieves a competitive erasure–preservation trade-of across concept categories. In particular, our method achieves Pareto-optimal performance compared to existing baselines on Celebrity, Artistic Style, and IP Entity erasure, while achieving performance comparable to DUO on Nudity erasure, where DUO has been reported to be particularly efective. These results demonstrate the efectiveness of our approach in maintaining strong erasure while preserving non-target generation quality.

Qualitative Results. We present qualitative comparisons in Figure 3. For each paired concept, the red-labeled rows present images generated from prompts corresponding to the erased concept, while the blue-labeled rows show images generated from related concepts that should be preserved. Our method efectively removes the target concepts (Picasso style, Rihanna identity, David Beckham identity, and Van Gogh style) while preserving the visual fidelity of non-target concepts (Claude Monet style, Margot Robbie identity, Matt Damon identity, and A. Khare style). In contrast, several baselines either erase inefectively or excessively distort preserved concepts. These results highlight that our method achieves a better balance between strong concept removal and faithful concept preservation. Additional qualitative samples are provided in the Appendix C.1.

![](images/22dbaf0982599a1e7dd14087d9f73ca47fa0dde4a53ccab740f89751f9243909.jpg)

![](images/bea58cdefb39a80e3441553ac2a25ccb89a9711b0b91080ecdc40e5c79ce2099.jpg)

![](images/4a6068aea5b3468e1fb0a52fcbe5222d7265d991c8e575eb5f051c1472476777.jpg)

![](images/db1262fd1ce003939d446c8ac8d8f86206837afa94d69484d7153f9a520b3374.jpg)  
Fig. 2: Erasure–Preservation score across concept categories. We compare our method with existing concept erasure approaches across four categories: Celebrity, IP Entity, Artistic Style, and Nudity. The vertical axis measures erasure efectiveness using the Unlearning Score or classifier accuracy for detecting the target concept, where lower values indicate stronger erasure. The horizontal axis measures preservation effectiveness using CMMD (×1000), where lower values indicate better preservation of non-target concepts. Each point denotes a hyperparameter setting; baselines vary their erasure-strength parameter when available, whereas our method varies only $m _ { \mathrm { p u s h } }$ In particular, it achieves the Pareto frontier in the erasure–preservation trade-of for Celebrity, Artistic Style, and IP Entity erasure. For Nudity, our method achieves results comparable to DUO, while maintaining stronger erasure–preservation trade-ofs across the other categories.

Adversarial Robustness. Our method achieves lower ASR than methods without explicit robustness objectives (MACE, UCE, AC, and ESD) under both UnlearnDif (Table 1) and CCE (Table 2), indicating competitive adversarial robustness. Under UnlearnDif, our method also obtains lower Pre-ASR and requires a longer average attack time than these methods, suggesting that the

STEREO

SUMA

Original  
ESD  
UCE  
MACE  
AC  
DUO  
Ours  
![](images/fe1de7266b358d6cc0ec83f318dfa647e43ba461705437e655a2e1bc888e350e.jpg)  
Fig. 3: Qualitative comparison of erasure methods. Our method erases the target concepts (e.g., Rihanna identity and Picasso style) while preserving the non-target concepts (e.g., Margot Robbie identity and Claude Monet style). Compared with existing approaches, our method achieves a favorable erasure–preservation trade-of.

attack requires more optimization efort to recover the erased concept. Additionally, our method also achieves performance comparable to SUMA, which is designed to improve robustness. Nevertheless, methods specifically optimized for adversarial robustness (DUO and STEREO) generally achieve higher robustness, although they significantly degrade preservation performance, as shown in Fig. 2. We note that our method is orthogonal to robustness-focused techniques such as STE (STEREO’s Stage 1) and could potentially benefit from them. Incorporating the Stage 1 procedure proposed by STEREO, which is also adopted in both STEREO and SUMA, can further improve robustness.

Applicability to Modern Architectures. While we primarily use SD1.4 for computational eficiency, we further conduct a preliminary study on FLUX [4], a flow-based Transformer model, using representative concepts (Barack Obama, Van Gogh, and Mickey Mouse). We compare our method against the available

Table 1: Success rate of recovering the erased nudity concept via Unlearn-Dif [40]. Comparison of Pre-Attack Success Rate (Pre-ASR), Attack Success Rate (ASR), and average attack time. Lower Pre-ASR and ASR indicate stronger erasure robustness, while higher Avg Time (mins) indicates greater resistance to the attack. Although our method is not explicitly optimized for robustness, it remains competitive among methods without robustness-focused objectives (MACE, UCE, AC, and ESD).
<table><tr><td>Method</td><td>Pre-ASR ↓</td><td>ASR↓</td><td>Avg Time (mins) ↑</td></tr><tr><td>ESD</td><td>0.26</td><td>0.92</td><td>9.89</td></tr><tr><td>UCE</td><td>0.41</td><td>0.92</td><td>8.35</td></tr><tr><td>MACE</td><td>0.55</td><td>0.92</td><td>7.11</td></tr><tr><td>AC</td><td>0.20</td><td>0.88</td><td>9.98</td></tr><tr><td>DUO</td><td>0.02</td><td>0.47</td><td>13.45</td></tr><tr><td>SUMA</td><td>0.16</td><td>0.83</td><td>11.80</td></tr><tr><td>STEREO</td><td>0.00</td><td>0.08</td><td>14.74</td></tr><tr><td>Ours</td><td>0.08</td><td>0.83</td><td>11.46</td></tr></table>

Table 2: Robustness against CCE attacks [27] and preservation under nudity erasure. Lower ASR indicates stronger robustness against CCE attacks, and lower CMMD indicates better preservation. Our method achieves a favorable robustness– preservation trade-of. In contrast, DUO and STEREO suppress CCE recovery more strongly but significantly compromise preservation, whereas UCE and ESD provide weaker trade-ofs under this protocol.
<table><tr><td>Method</td><td>ASR↓</td><td>CMMD↓</td></tr><tr><td>ESD</td><td>0.69</td><td>13.94</td></tr><tr><td>UCE</td><td>0.79</td><td>4.05</td></tr><tr><td>DUO</td><td>0.01</td><td>6.44</td></tr><tr><td>STEREO</td><td>0.04</td><td>139.47</td></tr><tr><td>Ours</td><td>0.20</td><td>1.43</td></tr></table>

FLUX baselines (ESD and UCE). Fig. 4 shows that our method yields a favorable erasure–preservation trade-of. In particular, our approach efectively erases target concepts, such as “Mickey Mouse” and “Barack Obama,” while preserving non-target concepts, such as “Donald Duck” and “Will Smith.” In contrast, ESD causes a significant semantic shift in “Donald Duck,” and UCE fails to erase “Mickey Mouse.”

## 4.3 Ablation Studies

Efect of Semantic Push. Fig. 2 shows that stronger semantic push consistently lowers target-concept detection accuracy. The qualitative samples in Fig. 5 further illustrate this trend: as $m _ { \mathrm { p u s h } }$ decreases, the generated images progressively move away from the target concept while retaining coherent visual content. This demonstrates that ${ \mathcal L } _ { \mathrm { p u s h } }$ enables controllable semantic separation from the erased concept. In the main experiments, we apply ${ \mathcal { L } } _ { \mathrm { p u s h } }$ to both concept-relevant tokens and the EOS token as a simple and consistent protocol, as described in Sec. 3.1; Appendix C.5 further explores alternative token-selection designs.

Original  
ESD  
UCE  
Ours  
![](images/0b6a64b1d60c78b694d2acacce1f929e89b39f08eacac0869178894e6fc7bd1e.jpg)

<table><tr><td></td><td></td><td>Method Acc. ↓ CMMD ↓</td></tr><tr><td>UCE</td><td>0.33</td><td>268.42</td></tr><tr><td>ESD</td><td>0.00</td><td>1511.69</td></tr><tr><td>Ours</td><td>0.01</td><td>239.61</td></tr></table>

(b) FLUX quantitative results  
(a) FLUX qualitative results  
Fig. 4: Concept erasure and preservation results on FLUX. (a) Qualitative comparison on representative identity, style, and character concepts. Our method erases target concepts, such as Barack Obama, Van Gogh, and Mickey Mouse, while preserving non-target concepts, such as Will Smith, Picasso, and Donald Duck. ESD causes substantial semantic shifts in non-target concepts, whereas UCE fails to semantically erase Mickey Mouse. (b) Quantitative results. Accuracy measures target-concept detection after erasure, and CMMD measures preservation relative to the original FLUX model; lower is better for both metrics. These results demonstrate the applicability of our objective to FLUX while maintaining a favorable erasure–preservation trade-of.

Design Choice for $L _ { \mathrm { { p u l l } } } \colon$ Relative vs. Absolute. We compare our proposed $L _ { \mathrm { p u l l } }$ , which is based on a relative margin that accounts for the original similarity between the erased concept and the anchor (generic) concept, with an alternative formulation based on an absolute margin:

$$
\mathcal { L } _ { \mathrm { a b s m a r g i n - p u l l } } ( W , W _ { 0 } , p _ { e } , p _ { g } ) = \operatorname* { m a x } \big ( m _ { \mathrm { p u l l } } - \sin ( W p _ { e } , ~ W _ { 0 } p _ { g } ) , ~ 0 \big ) .\tag{6}
$$

As shown in Fig. 6, our relative margin achieves a better erasure–preservation trade-of than the absolute margin. We hypothesize that this improvement may partly stem from substantial variation in the original target–anchor similarities across layers and concepts, as reflected by the denominator statistics in Table 3. Because diferent layers capture diferent levels of abstraction [3, 19], a fixed absolute margin can impose an overly strong or overly weak constraint depending on the initial similarity scale. In contrast, our relative margin normalizes the updated similarity by its original value, yielding a proportional constraint that is more comparable across layers and concepts.

![](images/5884c10fa4e38a630f9cda3093b1a02f71ac57a53460a206d4c6ffd99fc8c9cf.jpg)  
Fig. 5: Controllable semantic deviation induced by semantic push. As m<sub>push</sub> decreases from 0.80 to 0.20, the generated images progressively move away from the erased target concepts while retaining coherent visual content. Examples include artistic style, celebrity identity, and IP entity erasure, illustrating the controllable semantic separation induced by ${ \mathcal { L } } _ { \mathrm { p u s h } }$

Efect of Preservation Losses $L _ { \mathbf { p r e s e r v e } }$ and $L _ { \mathbf { d e n o i s e } }$ . We evaluate the contribution of the preservation losses by comparing our full method with a variant that excludes $L _ { \mathrm { p r e s e r v e } }$ and $L _ { \mathrm { d e n o i s e } }$ . The results in Fig. 6 demonstrate that incorporating explicit preservation losses significantly improves preservation of non-target concepts, consistent with prior works [11, 26].

Design Choice for Distance Function: Cosine vs. L2 Distance. We compare our proposed formulation, which is based on cosine distance, with an alternative implementation based on L2 distance. Specifically, the L2-based push, pull, and preservation losses are defined as

$$
\mathcal { L } _ { \mathrm { l 2 - p u s h } } ( W , W _ { 0 } , p _ { e } ) = \operatorname* { m a x } \left( m _ { \mathrm { l 2 - p u s h } } - \lvert \lvert W p _ { e } - W _ { 0 } p _ { e } \rvert \rvert _ { 2 } ^ { 2 } , ~ 0 \right) .\tag{7}
$$

$$
\mathcal { L } _ { \mathrm { l 2 - p u l l } } ( W , W _ { 0 } , p _ { e } , p _ { g } ) = \operatorname* { m a x } \big ( \| W p _ { e } - W _ { 0 } p _ { g } \| _ { 2 } ^ { 2 } - m _ { \mathrm { l 2 - p u l l } } , 0 \big ) .\tag{8}
$$

$$
\mathcal { L } _ { \mathrm { l 2 - p r e s e r v e } } ( W , W _ { 0 } , p _ { g } ) = \lvert \lvert W p _ { g } - W _ { 0 } p _ { g } \rvert \rvert _ { 2 } ^ { 2 } .\tag{9}
$$

The $\mathcal { L } _ { \mathrm { l 2 - p u s h } }$ formulation is similar to the objective used in the SUMA [25] baseline, while $\mathcal { L } _ { \mathrm { l 2 - p r e s e r v e } }$ resembles the preservation regularization used in another baseline [11] in our main experiments. For a fair comparison, we perform a grid search for the L2-based variant with $m _ { \mathrm { l 2 - p u l l } }$ and $m _ { \mathrm { l 2 - p u s h } }$ in the range {250, 500, 750, 1000, 1250, 1500, 1750, 2000}. Based on the validation results, we select $m _ { \mathrm { l 2 - p u l l } } = 5 0 0$ and report results across a range of $m _ { \mathrm { l 2 - p u s h } } \in$ {250, 500, 750, 1000}, as larger values lead to excessive preservation degradation.

![](images/d10f19cc43cf2685ed8aa5f48c9a2f97f241020b3e872659703fa336c60af0b4.jpg)  
Fig. 6: Ablation Study on Design Choices. Ours denotes the full method, which uses cosine distance in $L _ { \mathrm { p u s h } , } L _ { \mathrm { p u l l } }$ , and $L _ { \mathrm { p r e s e r v e } } ,$ , together with the relative-margin formulation and preservation losses $\left( L _ { \mathrm { p r e s e r v e } } , L _ { \mathrm { d e n o i s e } } \right)$ . Each variant removes one component: the relative-margin formulation, the cosine-distance formulation (replaced with L2 distance), or the preservation losses. Each point corresponds to a diferent push mar-$\mathrm { g i n }$ . The x-axis measures same-domain preservation (CMMD ×1000, lower is better) and the y-axis measures unlearning score (detection accuracy, lower is better). The full method consistently achieves the best erasure–preservation trade-of, while removing any component degrades performance, indicating that each component contributes to the overall improvement.

The results in Fig. 6 suggest that our cosine-based formulation consistently achieves a significantly better erasure-preservation trade-of than the L2-based alternatives. Additionally, cosine-based margins are easier to interpret and tune in practice due to the bounded range of cosine similarity. In contrast, L2 distances are unbounded, yielding a less predictable margin search space and potentially requiring more extensive hyperparameter tuning.

## 5 Limitations

Although our method achieves a strong erasure–preservation trade-of, enables fine-grained control over semantic deviation, and demonstrates competitive adversarial robustness compared to methods without explicit defenses, it does not explicitly incorporate robustness-oriented training strategies. Consequently, the model may still remain vulnerable to adversarial attacks. Nevertheless, our framework is complementary to existing robustness techniques and could be readily integrated with approaches such as STEREO’s Stage 1 [25, 36] or other defensive strategies [39] to further enhance robustness.

## 6 Conclusion

We present Push–Pull Attentional Anchoring (PPAA), a concept erasure method based on a bounded cosine push–pull mechanism in difusion cross-attention space. This mechanism displaces target representations while maintaining their relative alignment with anchor concepts. By enforcing margins on relative cosine similarity, our approach enables controlled semantic modification and reduces collateral damage to non-target generations. Extensive experiments across multiple concept categories, including celebrities, artistic styles, nudity, and intellectual property, demonstrate that our method achieves a favorable erasure–preservation trade-of compared to existing approaches. Furthermore, our preliminary results indicate that the proposed mechanism is applicable to modern architectures, such as FLUX.

Acknowledgment. This research was supported by Research Fellowships from VISTEC, SCB Public Company Limited, and PTT Public Company Limited.

## References

1. Abitbul, K., Dar, Y.: How much training data is memorized in overparameterized autoencoders? an inverse problem perspective on memorization evaluation. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. pp. 321–339. Springer (2024)

2. AUTOMATIC1111: Stable difusion webui: Negative prompt (2022), https:// github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt, gitHub repository wiki. Accessed 1 February 2026

3. Basu, S., Zhao, N., Morariu, V.I., Feizi, S., Manjunatha, V.: Localizing and editing knowledge in text-to-image generative models. In: The Twelfth International Conference on Learning Representations (2023)

4. Black Forest Labs: Flux.1-dev. https://huggingface.co/black-forest-labs/ FLUX.1-dev (2024), hugging Face model repository. Accessed 1 February 2026

5. Bolya, D., Huang, P.Y., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., et al.: Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181 (2025)

6. Cai, H., Cao, S., Du, R., Gao, P., Hoi, S., Hou, Z., Huang, S., Jiang, D., Jin, X., Li, L., et al.: Z-image: An eficient image generation foundation model with single-stream difusion transformer. arXiv preprint arXiv:2511.22699 (2025)

7. Chang, J., Chung, H., Ye, J.C.: Contrastive cfg: Improving cfg in difusion models by contrasting positive and negative concepts. arXiv preprint arXiv:2411.17077 (2024)

8. Chin, Z.Y., Jiang, C.M., Huang, C.C., Chen, P.Y., Chiu, W.C.: Prompting4debugging: Red-teaming text-to-image difusion models by finding problematic prompts. arXiv preprint arXiv:2309.06135 (2023)

9. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022)

10. Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., Bau, D.: Erasing concepts from difusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2426–2436 (2023)

11. Gandikota, R., Orgad, H., Belinkov, Y., Materzyńska, J., Bau, D.: Unified concept editing in difusion models. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5111–5120 (2024)

12. Hasty, N., Kroosh, I., Voitek, D., Korduban, D.: Giphy celebrity detector. https: //github.com/Giphy/celeb-detection-oss (2019), gitHub repository. Accessed 1 February 2026

13. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

14. Ho, J., Salimans, T.: Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598 (2022)

15. Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., Kumar, S.: Rethinking fid: Towards a better evaluation metric for image generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9307–9315 (2024)

16. Koulischer, F., Deleu, J., Raya, G., Demeester, T., Ambrogioni, L.: Dynamic negative guidance of difusion models: Towards immediate content removal. In: Neurips Safe Generative AI Workshop 2024 (2024)

17. Kumari, N., Zhang, B., Wang, S.Y., Shechtman, E., Zhang, R., Zhu, J.Y.: Ablating concepts in text-to-image difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22691–22702 (2023)

18. Kumari, N., Zhang, B., Zhang, R., Shechtman, E., Zhu, J.Y.: Multi-concept customization of text-to-image difusion. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1931–1941 (2023)

19. Kwon, M., Jeong, J., Uh, Y.: Difusion models already have a semantic latent space. arXiv preprint arXiv:2210.10960 (2022)

20. Lee, B.H., Lim, S., Lee, S., Kang, D.U., Chun, S.Y.: Concept pinpoint eraser for text-to-image difusion models via residual attention gate. arXiv preprint arXiv:2506.22806 (2025)

21. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

22. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)

23. Lu, K., Kriplani, N., Gandikota, R., Pham, M., Bau, D., Hegde, C., Cohen, N.: When are concepts erased from difusion models? arXiv preprint arXiv:2505.17013 (2025)

24. Lu, S., Wang, Z., Li, L., Liu, Y., Kong, A.W.K.: Mace: Mass concept erasure in difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6430–6440 (2024)

25. Nguyen, K., Tran, A., Pham, C.: Suma: A subspace mapping approach for robust and efective concept erasure in text-to-image difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19587–19596 (2025)

26. Park, Y.H., Yun, S., Kim, J.H., Kim, J., Jang, G., Jeong, Y., Jo, J., Lee, G.: Direct unlearning optimization for robust and safe text-to-image models. Advances in Neural Information Processing Systems 37, 80244–80267 (2024)

27. Pham, M., Marshall, K.O., Cohen, N., Mittal, G., Hegde, C.: Circumventing concept erasure methods for text-to-image generative models. arXiv preprint arXiv:2308.01508 (2023)

28. Praneeth, B., pa4y, YonghyunPark, Urosevic, J., Ayinmehr, A., brett koonce, Golovanenko, V.: notai-tech/NudeNet. https://github.com/notAI-tech/NudeNet (jul 31 2024), https://github.com/notAI-tech/NudeNet, gitHub repository. Accessed 1 February 2026

29. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

30. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36, 53728–53741 (2023)

31. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

32. Rusanovsky, M., Malnick, S., Jevnisek, A., Fried, O., Avidan, S.: Memories of forgotten concepts. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2966–2975 (2025)

33. Saleh, B., Elgammal, A.: Large-scale classification of fine-art paintings: Learning the right metric on the right feature. arXiv preprint arXiv:1505.00855 (2015)

34. Sharma, A.S., Sarkar, N., Chundawat, V., Mali, A.A., Mandal, M.: Unlearning or concealment? a critical analysis and evaluation metrics for unlearning in difusion models. arXiv preprint arXiv:2409.05668 (2024)

35. Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020)

36. Srivatsan, K., Shamshad, F., Naseer, M., Patel, V.M., Nandakumar, K.: Stereo: A two-stage framework for adversarially robust concept erasing from text-to-image difusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 23765–23774 (2025)

37. Tian, Z., Nan, S., Xu, M., Zhai, S., Qu, W., Liu, J., Jia, R., Zhang, J.: Sparse autoencoder as a zero-shot classifier for concept erasing in text-to-image difusion models. arXiv preprint arXiv:2503.09446 (2025)

38. Tsai, Y.L., Hsu, C.Y., Xie, C., Lin, C.H., Chen, J.Y., Li, B., Chen, P.Y., Yu, C.M., Huang, C.Y.: Ring-a-bell! how reliable are concept removal methods for difusion models? arXiv preprint arXiv:2310.10012 (2023)

39. Zhang, Y., Chen, X., Jia, J., Zhang, Y., Fan, C., Liu, J., Hong, M., Ding, K., Liu, S.: Defensive unlearning with adversarial training for robust concept erasure in difusion models. Advances in neural information processing systems 37, 36748– 36776 (2024)

40. Zhang, Y., Jia, J., Chen, X., Chen, A., Zhang, Y., Liu, J., Ding, K., Liu, S.: To generate or not? safety-driven unlearned difusion models are still easy to generate unsafe images... for now. In: European Conference on Computer Vision. pp. 385– 403. Springer (2024)

## Appendix: Push–Pull Attentional Anchoring for Difusion Concept Erasure

## A Implementation Details

In the main experiments, we implement our method on Stable Difusion 1.4 (SD1.4) based on the ESD codebase [10]. We adopt the same hyperparameter configuration and use a learning rate of $5 \times 1 0 ^ { - 5 }$ . The model is trained for 1,000 optimization steps using the Adam optimizer, while updating only the cross-attention K and V projection layers, following prior works [11,17]. For the baselines, we follow their recommended configurations, including the learning rate, number of training steps, learnable parameters, and erasure strength.

## A.1 Numerical Stability and Implementation Safeguard for the Relative Pull Constraint

We analyze the denominator of the relative pull loss, sim $( W _ { 0 } p _ { e } , W _ { 0 } p _ { g } )$ , on SD1.4 to verify that the ratio is numerically well behaved in our evaluation settings. As shown in Table 3, all evaluated target–anchor pairs have positive denominators well separated from zero. We also include a random-anchor diagnostic, where random anchors are sampled from the 600 OpenImage classes instead of using the semantically related generic anchors.

Table 3: Statistics of the pull-loss denominator sim $. ( W _ { 0 } p _ { e } , W _ { 0 } p _ { g } )$ on SD1.4 across evaluated concepts. The observed denominators remain positive for both semantically related generic anchors and random anchors sampled from the 600 OpenImage classes, supporting the numerical stability of the relative pull ratio.
<table><tr><td rowspan="2">Setting</td><td colspan="4">Generic Related anchor</td><td colspan="4">Random anchor</td></tr><tr><td>Avg.</td><td>Std.</td><td>Min.</td><td>Max.</td><td>Avg.</td><td>Std.</td><td>Min.</td><td>Max.</td></tr><tr><td>Main experiments (13 concepts)</td><td>0.659</td><td>0.166</td><td>0.247</td><td>0.942</td><td>0.390</td><td>0.186</td><td>0.050</td><td>0.834</td></tr><tr><td>50 celebrities</td><td>0.482</td><td>0.161</td><td>0.126</td><td>0.881</td><td>0.367</td><td>0.184</td><td>0.055</td><td>0.817</td></tr><tr><td>50 artistic styles</td><td>0.651</td><td>0.124</td><td>0.337</td><td>0.942</td><td>0.344</td><td>0.195</td><td>0.030</td><td>0.848</td></tr><tr><td>80 COCO classes</td><td>0.606 0.142</td><td></td><td>0.182</td><td>0.927</td><td>0.511</td><td></td><td>0.174 0.130</td><td>0.926</td></tr><tr><td>600 OpenImage classes</td><td>0.574 0.150</td><td></td><td>0.133</td><td>0.927</td><td>0.489</td><td>0.181 0.047</td><td></td><td>0.991</td></tr></table>

To safeguard against negative ratios when applied to other latent spaces or architectures, a simple variant that replaces the cosine similarity in the ratio with a shifted similarity can be implemented:

$$
\widetilde s ( u , v ) = \frac { 1 + \sin ( u , v ) } { 2 } + \epsilon , \quad \epsilon > 0 .\tag{10}
$$

One can substitute s˜ for sim in both the numerator and denominator, keeping them strictly positive while preserving the monotonic interpretation of cosine similarity.

## B Additional Experimental Details

## B.1 Anchor and Preservation Concepts

Anchor Concepts. We use a generic concept as the anchor concept and assume that it is specified by the user, following [17, 25]. For Celebrity Erasure, we use person as the anchor concept, while for Artistic Style Erasure, we use a painting in the style of an artist, following [17, 25]. For Nudity Erasure, we use dressed person as the anchor concept, following [26]. For IP Entity Erasure, we use semantically related generic concepts as anchors: cartoon for Mickey Mouse, laptop for MacBook, robot for R2D2 robot, and cat for Grumpy Cat.

Same-Domain Preservation Concepts. For Celebrity and Artistic Style Erasure, we randomly sample 50 celebrities and 50 artistic styles from the concept lists provided by [24], ensuring diversity within each category. The complete list of sampled concepts is provided in Table 4.

## B.2 Validation Set

We provide details of the validation set used for hyperparameter selection in the main experiments and ablation studies (Sec. 4). The validation set includes four celebrity concepts as erasure targets: Adam Driver, Amy Adams, Nicki Minaj, and Chris Rock. Additionally, eight celebrity concepts are used as preservation targets: Andrew Garfield, Octavia Spencer, Anne Hathaway, Idris Elba, Morgan Freeman, Oprah Winfrey, Emma Stone, and Elon Musk. The validation set is demographically balanced across races (Black and White) and genders (Male and Female). None of these identities appears in the test set used in the main experiments (Sec. 4).

## B.3 Hyperparameter Selection

We provide additional details on the hyperparameter selection used in the main experiments (Sec. 4). We fix $\lambda _ { 1 } = \lambda _ { 2 } = \lambda _ { 4 } = 1$ and optimize $\lambda _ { 3 }$ and $m _ { \mathrm { p u l l } }$ using a separate validation set. We perform grid search over $\lambda _ { 3 } \in \{ 0 , 1 , 1 6 , 3 2 , 6 4 \}$ and $m _ { \mathrm { p u l l } } \in \{ 1 . 0 0 , 0 . 8 0 , 0 . 6 0 , 0 . 4 0 , 0 . 2 0 , 0 . 0 0 \}$ , and select the combination that achieves a Pareto-optimal erasure–preservation trade-of on the validation set described in Sec. B.2, resulting in $\lambda _ { 3 } = 3 2$ and $m _ { \mathrm { p u l l } } = 0 . 4 0$ . Although the validation set is constructed from celebrity identities, we use the same configuration across all concept categories.

Table 4: Same-domain preservation concepts used in evaluation. For Celebrity and Artistic Style Erasure, we randomly sample 50 celebrities and 50 artistic styles from the list provided by [24]. These concepts measure preservation performance within the same semantic domain as the erased concept.
<table><tr><td>Category Celebrity</td><td>Same-Domain Preservation Concepts</td></tr><tr><td></td><td>Maya Angelou, Gillian Anderson, Jim Morrison, Jennifer Connelly, Benicio Del Toro, Avril Lavigne, Aaron Paul, Bill Murray, Kim Jong Un, Justin Bieber, David Bowie, Barry Manilow, Judy Garland, Betty White, Denise Richards, Gal Gadot, Pierce Brosnan, Julianne Moore, David Tennant, Jackie Chan, Natalie Portman, Rachel Dratch, Liv Tyler, Gordon Ramsay, Patrick Stewart, Doris Day, Matthew Mc- Conaughey, Amy Schumer, Hayley Atwell, Niall Horan, Neil deGrasse Tyson, Heath Ledger, Kristen Stewart, Amy Poehler, Kirsten Dunst, Matt Damon, Joan Rivers, Bill Nye, Britney Spears, Lizzy Caplan, Emma Roberts, Clint Eastwood, Rachel McAdams, Harry Dean Stan- ton, Krysten Ritter, Aretha Franklin, Kate Upton, George Takei, Christina Hendricks, Andy Samberg</td></tr><tr><td>Artistic Style</td><td>Abed Abdi, Albert Marquet, Alberto Vargas, Adrian Tomine, Al- bert Edelfelt, A. J. Casson, Abbott Fuller Graves, Alan Kenny, Al- berto Sughi, Aaron Siskind, Adriaen van Outrecht, Albert Good- win, Alberto Biasi, Albert Bloch, Adrian Donoghue, Akos Major, Alberto Burri, Ai Weiwei, Alastair Magnaldo, Albert Watson, Agnes Martin, Aleksey Savrasov, Adolph Gottlieb, Adrianus Eversen, Af- fandi, Alessandro Allori, Alec Soth, Abigail Larson, Alan Parry, Akira Toriyama, Adonna Khare, Alena Aenami, Agnes Cecile, Albert Lynch, Albrecht Dürer, Alejandro Jodorowsky, Alex Alemany, Alex Colville, Albert Eckhout, Abram Efimovich Arkhipov, Adolph Men- zel, Aleksi Briclot, Agostino Tassi, Albert Bierstadt, Aaron Douglas Albert Tucker, Abraham Pether, Alberto Magnelli, Adam Hughes, Alan Schaller</td></tr></table>

## B.4 Evaluation Criteria

We evaluate our method after erasing a target concept using three criteria: Erasure Efectiveness, Preservation Efectiveness, and Erasure Robustness. We provide additional details for each criterion below.

Erasure Efectiveness: We generate 100 images of the erased concept using the edited model. We then apply the corresponding detector described in Sec. 4 to determine whether the target concept still appears in each generated image. Following [24], the efectiveness of erasure is measured as the ratio of images in which the concept is detected to the total number of images generated.

Preservation Efectiveness: We evaluate both same-domain and generaldomain preservation. For same-domain preservation, we generate 25 images per concept using both the erased model and the pretrained model. For generaldomain preservation, we use all 5,000 captions from the COCO validation split and generate one image per caption with both models. We then compute the CMMD [15] score between the images generated by the erased model and those generated by the pretrained model, which serves as the preservation metric. The final score is reported by first computing the erasure and preservation metrics for each erased concept separately and then averaging the results across all evaluated concepts. We note that for reproducibility, all images are generated using the DDIM sampler [35] with 50 difusion steps and a guidance scale of 7.5 [14].

## C Additional Results

We provide additional qualitative and quantitative results on concept erasure and preservation compared with the baselines. For adversarial robustness, we further report results for Training-Free Trajectory Probing and UnlearnDif. Additionally, we demonstrate that our method can be applied to another recent model architecture beyond SD1.4 and FLUX.

## C.1 Additional Qualitative Results

We present results for target concepts across diferent categories together with preservation concepts in Fig. 11–17. For celebrity and artistic style erasure, we show both same-domain and general-domain preservation examples. For nudity and intellectual property entities, we report only general-domain preservation results, as well-established same-domain preservation concepts are not clearly defined for these categories. The preservation concepts shown in the figures are randomly sampled. Overall, the results demonstrate that our method achieves a stronger erasure–preservation trade-of compared to the baselines.

## C.2 Additional Quantitative Results

For celebrity and artistic-style erasure, the main paper reports erasure performance against domain-specific preservation scores. To provide a more comprehensive quantitative evaluation, this appendix additionally reports erasure scores against generic-domain preservation in Fig. 7. Across both generic-domain and domain-specific preservation settings, our method achieves a strong erasure– preservation trade-of and often lies on the Pareto-optimal frontier.

## C.3 Additional Adversarial Robustness Results

Training-Free Trajectory Probing. We follow the same evaluation setting and configuration as [23] on the nudity-erased model. Base denotes an image generated from the original prompt, while Attack corresponds to the image that is most aligned with the attack concept (e.g., a naked person) according to the CLIP score. Among methods that are not explicitly designed with robustnessfocused objectives (MACE, UCE, AC, and ESD), our method achieves competitive results. As shown in Fig. 8, our method exhibits resistance to the attack, similar to methods that are explicitly optimized for adversarial robustness (STEREO and DUO).

## C.4 Additional Results on Applicability to Modern Architectures

In the main experiments, we demonstrate that our method can be efectively applied to SD1.4 and FLUX. We further evaluate its applicability to Z-Image [6], a recent variant of text-to-image models based on the Scalable Single-Stream Multimodal Difusion Transformer (S3-DiT). In this setting, we optimize the Cap Embedder, a text projection layer that controls semantic conditioning. Our qualitative results (see Fig. 10) demonstrate that our method remains efective on Z-Image, erasing the target concept while preserving the quality of non-target concepts.

## C.5 Exploring Alternative Token Selection for Semantic Push

In the main experiments, we adopt a simple and consistent token-selection protocol in which ${ \mathcal { L } } _ { \mathrm { p u s h } }$ is applied to the concept-relevant token(s) and the EOS token while excluding template tokens such as those in $^ { 6 6 } \mathrm { a }$ photo of”, as defined in Sec. 3.1. To further examine this design choice, we conduct a preliminary exploratory analysis that disentangles the contribution of these two selected token groups. Specifically, we compare applying ${ \mathcal { L } } _ { \mathrm { p u s h } }$ to both groups, only to the EOS token, and only to the concept-relevant token(s).

The results in Fig. 9 indicate that applying ${ \mathcal { L } } _ { \mathrm { p u s h } }$ only to the concept-relevant token(s) is a promising refinement. Compared with variants that include the EOS token, the concept-token-only variant produces semantic changes that remain more localized to concept-specific information, while unrelated information such as background and pose appears less directly afected. This behavior may potentially reduce unintended changes to non-target attributes and improve the erasure–preservation trade-of. Investigating token selection strategies beyond the EOS and concept-relevant token(s) provides a promising direction for further improving the erasure–preservation trade-of.

## D Negative Societal Impact

Although our concept erasure method is designed to address privacy and copyright concerns, it could potentially be misused in harmful ways, for example, by selectively erasing concepts associated with specific demographic groups or religious identities, which may enable discriminatory or exclusionary outcomes. Such misuse highlights the importance of responsible deployment and careful consideration of the societal implications of concept erasure methods.

![](images/fd8d948c50768b421271a1cd94a346f0b7d5edddd3522bf508461622001fc399.jpg)

![](images/b2e9e8d2eaf675393f31d5d952282ebf80b4cf51898f31bb10e15b3642488563.jpg)

![](images/fbfd87114c68c57ac4f953e29d063423f2bf3ee60aa94974c7bec14ea1d2fa54.jpg)

![](images/ca3d5b673cbe931c22fe359b214d950187f13b02e1dedd33be83a16d6bd76386.jpg)

![](images/4ad7331a7dee8cbd69f947c41201cda683f054a9871069ea816640d69946db79.jpg)

![](images/3c48527fff810fcdd8e7c2cbea6f962953176271352c3f824d77a74cd6c0bbf6.jpg)  
Fig. 7: Erasure–Preservation Trade-of Across Concept Categories. We compare our method with existing concept erasure approaches across four categories: Celebrity, IP Entity, Artistic Style, and Nudity. For Celebrity and Artistic, we provide both same-domain and generic-domain preservation scores. Our method consistently performs well across categories. In particular, it achieves the Pareto frontier in the erasure–preservation trade-of for Celebrity, Artistic Style, and IP Entity erasure. For Nudity, our method performs comparably to DUO. However, DUO performs poorly in the other categories.

ESD  
UCE  
MACE  
AC  
SUMA  
DUO  
STEREO  
Ours  
![](images/6db7b1de18017d99c98cd5a504316b93fb91108902048dd5adf65a9b01e78ee7.jpg)  
Fig. 8: Training-Free Trajectory Probing Results. We evaluate robustness to trajectory-based attacks using the prompt “a photo of a naked person” on nudityerased models. Base shows images generated without applying the attack procedure. Attacked (Best) denotes the image most aligned with the attack concept according to the CLIP score among all generated samples during the probing trajectory. The remaining rows show randomly sampled images obtained from the attack method. Although our method is not explicitly trained with robustness-oriented objectives (e.g., DUO and STEREO), its behavior under this attack appears competitive to that of existing methods such as MACE, UCE, SUMA, and ESD.

![](images/3a1c47aafd951801d0129938a77177a270de0f35a216422fc574b8576815ddbc.jpg)  
Fig. 9: Exploratory token selection for semantic push. We compare three tokenselection variants for $\mathcal { L } _ { \mathrm { p u s h } } \mathrm { . }$ applying it to both the concept-relevant token(s) and the EOS token, only to the EOS token, and only to the concept-relevant token(s). The concept-token-only variant localizes the semantic change more strongly to conceptspecific information, indicating a promising refinement for reducing unintended changes to non-target attributes such as pose and background.

![](images/b02faa08f594857ee1ec67fc18411336bc39ad8c02283ff3b32f7629b342ba3a.jpg)  
Fig. 10: Concept erasure and preservation results on the Z-Image model. We show that our method can be applied to Z-Image, a recent text-to-image model based on a Scalable Single-Stream Multimodal Difusion Transformer, achieving efective concept erasure while preserving the quality of non-target generations.

Original  
ESD  
UCE  
MACE  
AC  
SUMA  
DUO STEREO Ours  
![](images/f9c585655078a5aa5cc58fe27fdecd5cbb322fc731da45a0f71667d0eb728140.jpg)  
A photo of David Beckham

![](images/f5997cef62b2985ab50307c2112587a9b83c820aa8002413afa3c3b8b6124cc3.jpg)  
A photo of Matt Damon

![](images/3e1ae0dcfcfc7953df279437c818002646f74ce34db13b8fb0d120c638eadbaf.jpg)  
A photo of Britney Spears

![](images/87d5e817fdb57a062b5c1401b1c65811b517cb9514d6f82859539c5dea13c08f.jpg)  
People are holding wine glasses next to each other.

![](images/7e4458c7344070a1c56da68591914cc914973c5edb1bd6ed4871298d6ebd1483.jpg)  
Girl bending down with handbag on her shoulder petting two dogs.  
Fig. 11: Additional Qualitative Comparison on Celebrity Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue and are randomly sampled. Images in the same row are generated using the same random seed across all methods for fair visual comparison.

Original  
ESD  
UCE  
MACE  
AC  
SUMA  
DUO STEREO Ours  
![](images/926225ac27bb863223a672c4a81ec64a9c305e7d7799782560f93827e9a3f35f.jpg)  
Man sitting in snow with snowboard attached to shoes.

Fig. 12: Additional Qualitative Comparison on Celebrity Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue. Images in the same row are generated using the same random seed across all methods for fair visual comparison.

Original  
ESD  
UCE  
MACE  
AC  
SUMA  
![](images/50061560e9ac2f03266dc277fd2fac389e8bfe95f84be2d1211cc0af60ebc6e8.jpg)  
DUO STEREO Ours  
A painting in the style of Van Gogh

![](images/1bfc9d65d39a8d1430e2b990f25b75cda78962a23d9d6e4be8f51198546b1e8d.jpg)  
A painting in the style of Adonna Khare

![](images/a372eee0228bfed72ec1c58e30c047f287844092c9884af883fdea95e315c0e2.jpg)  
A painting in the style of A.J. Casson

![](images/9373cbd7b630528a4bcfb714085600f3c47ef6742507e4641d9fccfef0428d80.jpg)  
A pink clock tower rises above a tree line

![](images/c8cbc83622a27f36999c2c06478497c88ed7db0a600fc66309f2a82e965a5559.jpg)  
A cute cat laying down in a sink  
Fig. 13: Additional Qualitative Comparison on Artistic Style Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue. Images in the same row are generated using the same random seed across all methods for fair visual comparison.

Original  
ESD  
UCE  
MACE  
AC  
SUMA  
DUO STEREO Ours  
![](images/19dfd8112e1aaf0b7c78c19aa99001843482f14ca1b6ebb2f68cb6c411ce520d.jpg)  
A painting in the style of Picasso

![](images/87bf22674957ecd955f578e2947c07dcea3e3226095b00498a310c6ca9df4179.jpg)  
A painting in the style of Adam Hughes

![](images/84000047cb6e67667bcc198c85413171481d50a5fc2a50fd4b970a9c695570ef.jpg)  
A painting in the style of Akira Toriyama

![](images/b4579ee55fec32a01e01345ef90975939e793f7fa468b38d61be1a6920945897.jpg)  
A man flying through the air while riding a snowboard

![](images/e3504f17b2c2591973f286d6c28d15d40a094c5a6a11678b278cefe5c79296a1.jpg)  
A herd of animals traveling down a country road surrounded by a lush green landscape.  
Fig. 14: Additional Qualitative Comparison on Artistic Style Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue. Images in the same row are generated using the same random seed across all methods for fair visual comparison.

A dried black flower in a long, tall black & white vase.

Original  
ESD  
UCE  
MACE  
AC  
SUMA  
![](images/607c16efb9b38dac1b78c9d8b375d7a305e7e293d515a374fe7c5cb74ff14551.jpg)  
DUO STEREO Ours

A photo of Grumpy Cat  
![](images/c07b4213f8ade385782cde378b799147a2bd65b19e3068231268d72efb6de4f0.jpg)

![](images/e933b7a17d95735555409d6089ccf1c0c8a4cdb55ec8993f4d6a99dbb51ea028.jpg)  
Plates of food and glasses of wine sitting on a table

Fig. 15: Additional Qualitative Comparison on Intellectual Property Entity Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue. Images in the same row are generated using the same random seed across all methods for fair visual comparison.

Three people pose for a picture while drinking wine  
Original  
ESD  
UCE  
MACE  
AC  
SUMA  
DUO STEREO Ours  
![](images/f3b4ebe49fd75d5ec000f7d5f80245776cd568abbd5c9e29ae42052fbc1272da.jpg)

A photo of R2D2 Robot  
![](images/ff677b36abeebc6ba0b5d57dd8005df2fb28af204af752f4efcf145509886eee.jpg)

![](images/8edecb75bc36f39e5dd26f076f0b91cbe6de4d600b41d4d66b6166b6a8317c31.jpg)  
A couple of jockey's riding the horses through the beach  
Fig. 16: Additional Qualitative Comparison on Intellectual Property Entity Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue. Images in the same row are generated using the same random seed across all methods for fair visual comparison.

STEREO  
![](images/14680b6fb4a5970058124c54bd9334acc1e7ec77f46b2d10d3fd47de91ad6feb.jpg)  
Fig. 17: Additional Qualitative Comparison on Nudity Erasure. Prompts corresponding to the target concepts are shown in red, while prompts corresponding to the non-target concepts are shown in blue. Images in the same row are generated using the same random seed across all methods for fair visual comparison.