# Robust Logical Anomaly Detection under Capture Shifts via Synthetic Anomalies

Ufuk Efe<sup>a,∗</sup>, Alptekin Temizel<sup>a</sup>

<sup>a</sup>Graduate School of Informatics, Middle East Technical University, Üniversiteler Mahallesi, Dumlupınar Bulvarı No:1, Ankara, 06800, Türkiye

## Abstract

Detectors of logical anomalies, such as missing, extra or wrongly combined parts, usually score a test image by its distance from normal (good) images. They achieve near-perfect image-level AUROC on standard industrial anomaly detection benchmarks such as MVTec LOCO, but their robustness to changes in capture conditions remains largely unexamined. Under a heterogeneous perturbation protocol in which each test image is randomly kept unaltered or subjected to a brightness reduction or a zoom-out transformation, the normal-reference methods we tested lose 9.7 to 34.1 AUROC points, and two of the strongest, CSAD and SALAD, fall from 97.7% and 96.4% to 68.1% and 62.3%. We propose a framework in which a classifier, trained on normal images and on synthetic anomalies, scores a test image without comparing it with normal data. From a short description of each product, a language model derives anomaly types and editing instructions, an image editor applies them to normal images, and the classifier learns to separate the edited images from automatically discovered normal sub-modes. With its full training recipe, the framework reaches 94.9% AUROC on clean images, 1.5 to 2.8 points below SALAD and CSAD, and 94.1% under the perturbation protocol. Trained without brightness or zoom-out augmentation, so that neither test perturbation occurs in training, it still reaches 92.5% (93.2% clean). These results suggest that clean benchmark scores can hide a strong sensitivity to capture conditions, and that learning the decision from both normal images and rule violations may reduce it.

Keywords: anomaly detection, logical anomalies, robustness, distribution shift, synthetic anomalies, image editing, industrial inspection

## 1. Introduction

![](images/03b71c40f4b1bba24809ac65c6a4d5d952c276b0acbb90dede4dae29cdab9a52.jpg)  
Figure 1: Clean and heterogeneous AUROC. Five-category mean image-level logical AUROC on MVTec LOCO for clean test images and under the heterogeneous perturbation protocol, in which each test image is randomly kept unaltered or subjected to a brightness reduction or a zoom-out transformation. Our two variants are shown in blue: Ours (full recipe) and Ours (no overlap), which is trained without brightness or zoom-out augmentation and is therefore used to assess robustness. AnomalyCLIP (below the dashed line), a zero-shot method that uses no normal reference, is near chance level even on clean images.

Visual anomaly detection is essential for automated quality control in manufacturing (Lin et al., 2025). Because defective products are rare and diverse, most detectors are trained exclusively on normal images, without access to real anomalies (Lin et al., 2025; Bergmann et al., 2022). We study logical anomalies, such as missing, extra or wrongly combined parts, rather than structural defects such as scratches. On MVTec LOCO (Bergmann et al., 2022), the standard benchmark for this task, the best methods, such as CSAD (Hsieh and Lai, 2024) and SALAD (Fucka et al.ˇ , 2025), reach nearperfect image-level area under the ROC curve (AUROC). Although both also train on synthetic anomalies, their score still depends on how far a test image lies from normal data; we call such methods normal-reference methods. These benchmark results, however, come from images taken under controlled capture conditions.

On a production line, illumination and camera position do not always remain constant (Baitieva et al., 2025). Robustness to such capture shifts has been benchmarked mainly for structural defects (Cao et al., 2023; Pemula et al., 2025; Heckler-Kram et al., 2026). We evaluate logical anomaly detectors on MVTec LOCO under a heterogeneous perturbation protocol (Section 3.6). Each test image is assigned at random to one of seven conditions, unperturbed or with a brightness reduction or a zoom-out at one of three strengths, and a single AUROC is computed over the mixed test set. Under this protocol, the normal-reference methods we tested lose 9.7 to 34.1 AUROC points, and

CSAD and SALAD fall to 68.1% and 62.3% (Fig. 1). Since neither perturbation alters the logical content of an image, this drop originates in how the detectors score the perturbed images.

A likely reason for this drop is that normal-reference detectors compare a test image with normal data taken under one lighting and one camera position. A darker image or a smaller object may then be scored as anomalous even when every part is correct; indeed, under both perturbations, many normal test images score above the median clean anomaly (Fig. 5). This reading suggests learning the decision instead as a contrast between normal images and rule violations seen under the same capture conditions, so that a capture change moves both sides alike. Our framework difers from CSAD and SALAD not in using synthetic anomalies but in this test-time decision, which is fully discriminative: the classifier scores a test image without comparing it with normal data. Learning this contrast, however, needs rule violations, and real ones are rarely available.

Our framework therefore creates these rule violations as synthetic anomalies from a description of the product, so that, like the normal-reference methods, it needs no real anomaly during training. First, a person writes a short description of each category. From it, a language model proposes anomaly types, rule violations that could occur in the category, and writes an editing prompt for each; an image editor then applies one prompt to each normal training image. The edited images are then manually reviewed, assisted by a vision-language model, and only valid anomalies are retained. Normal sub-mode discovery, a simple automatic step, assigns each distinct group of normal images its own class, and the classifier is trained to distinguish these normal sub-modes from the anomaly types. Similar generate-then-train pipelines exist for structural defects (Sun et al., 2025; Fucka et al.ˇ , 2026a; Hu et al., 2026); for logical anomalies, ComGEN (Tong et al., 2025) generates anomalies with a difusion model but keeps a normal-reference decision. In summary, our contributions are threefold:

1. We show that brightness reduction and zoom-out, applied at random to the test images of MVTec LOCO, reduce the AUROC of the normal-reference methods we tested by 9.7 to 34.1 points, a loss of 38.2% to 73.5% of their performance above chance (Section 3.6).

2. We propose a framework in which a classifier, trained on normal images and on synthetic anomalies derived from a short description of each category, scores a test image without comparing it with normal data.

3. With its full training recipe, the framework reaches 94.9% AUROC on clean images, 1.5 to 2.8 points below SALAD and CSAD, and 94.1% under the same perturbations; trained without brightness or zoom-out augmentation, it still reaches 92.5% (93.2% clean).

## 2. Related Work

## 2.1. Logical Anomaly Detection

MVTec LOCO (Bergmann et al., 2022) is the standard benchmark for logical anomalies, which break rules on the presence, number, position or combination of parts, unlike the local structural defects of MVTec AD (Bergmann et al., 2019).

Because such a rule spans the whole image, the strongest methods model the image’s composition. CSAD (Hsieh and Lai, 2024) segments components without manual labels and compares their patch histograms with those of normal images, and SALAD (Fucka et al.ˇ , 2025) adds a composition branch, trained on synthetic anomalies, to its appearance branches. PSAD (Kim et al., 2024) learns part segmentation from a few labeled images, and SAM-LAD (Peng et al., 2025) matches segmented objects with those of retrieved normal images. LogSAD (Zhang et al., 2025a) needs no training: it compares a test image with a memory of normal features and checks matching rules that its authors built from GPT-4V proposals, prompted with a few normal images and the benchmark’s constraint descriptions. By contrast, general-purpose detectors such as PatchCore (Roth et al., 2022) model local normality and struggle with logical anomalies (Kim et al., 2024), and zero-shot detectors such as AnomalyCLIP (Zhou et al., 2024) use no image of the target category. Apart from AnomalyCLIP, all methods above are normal-reference methods: they score a test image, at least in part, by its distance from normal training data. We examine this normal-reference design under capture shifts (Section 4).

Another line of work employs language models for detection. LogicAD (Jin et al., 2025) describes images with text features from a vision-language model (VLM), and LogiCode (Zhang et al., 2025b) uses a large language model (LLM) to generate code that checks rules that experts extract from normal images. VLMs adapted to inspection (Xu et al., 2025; Gu et al., 2024) detect and explain defects, and SCAN (Li et al., 2026) fine-tunes a multimodal LLM to verify written inspection criteria rule by rule. In these methods, the language model participates in detection at test time or through the code it generates. In our framework, the language model contributes to generating the synthetic anomalies, which are used only in training, and the anomaly score is computed by the classifier alone.

## 2.2. Synthetic Data for Anomaly Detection

Like many detectors, our classifier is trained on synthetic anomalies, which, for logical anomaly detection, must violate a rule of the category rather than merely alter local appearance. CutPaste (Li et al., 2021) pastes image patches and DRAEM (Zavrtanik et al., 2021) blends external textures into normal images; such local changes suit structural defects but rarely break a logical rule. For structural defects, generate-then-train pipelines exist, in which an LLM or a VLM describes defects and a generative model adds them to images: AnomalyAny (Sun et al., 2025) guides Stable Difusion (Rombach et al., 2022) with a normal image, AnomalyVFM (Fucka et al. ˇ , 2026a) inpaints defects into generated images to train a zero-shot detector, and MIRAGE (Hu et al., 2026) edits normal images to train a segmentation network without real anomalies. None of these pipelines is designed for logical constraints.

For logical anomalies, synthesis without real anomalies has used hand-designed rules: CSAD pastes extra components into training images to train its segmentation, SALAD erases or inpaints components in composition maps, and LogicAL (Zhao, 2024) edits edge maps. ComGEN (Tong et al., 2025) instead learns a token for each component of an MVTec LOCO category with a difusion model and removes tokens or samples unusual combinations to create anomalies; its detector compares each test image with the nearest normal image. All of these methods compare a test image with normal data at test time, directly or, in LogicAL, through a reconstruction of the image’s normal appearance. What sets our framework apart is therefore not the use of synthetic anomalies but its test-time decision. Table 1 compares these approaches with our framework.

Table 1: Synthetic anomalies in related methods and in our framework. Rows follow a synthetic anomaly from its source to the decision at test time. A normal-reference score depends, at least in part, on a comparison with normal training data. Our framework difers from the first column in the anomalies it targets and from the other two in its test-time decision. Columns, from left: AnomalyAny, AnomalyVFM and MI RAGE (Sun et al., 2025; Fucka et al. ˇ , 2026a; Hu et al., 2026); ComGEN (Tong et al., 2025); CSAD and SALAD (Hsieh and Lai, 2024; Fucka et al.ˇ , 2025).
<table><tr><td></td><td>AnomalyAny, AnomalyVFM, MIRAGE</td><td>ComGEN</td><td>CSAD, SALAD</td><td>Ours</td></tr><tr><td>Target anomalies</td><td>Structural</td><td>Logical and structural</td><td>Mainly logical</td><td>Logical</td></tr><tr><td>Knowledge source</td><td>Defect lists from an LLM or VLM</td><td>Component tokens learned from normal images</td><td>Components segmented in normal images</td><td>Written description of the category</td></tr><tr><td>Generation</td><td>Diffusion generation, inpainting or editing</td><td>Token removal and low-density sampling in a diffusion model</td><td>Hand-designed edits of components or maps</td><td>Instruction-based editing of normal images into anomaly types</td></tr><tr><td>Test-time decision</td><td>Detector trained on the synthetic data</td><td>Segmentation from differences to the nearest normal image</td><td>Fused scores, partly normal-reference</td><td>Classifier trained on normal images and synthetic anomalies; scores a test image without comparing it</td></tr></table>

Beyond anomaly detection, generated training data usually keeps its label: DGIn-Style (Jia et al., 2024) renders existing label maps in new styles, and ALIA (Dunlap et al., 2023) edits training images along domain descriptions that an LLM summarizes from captions. Our edits instead change the label on purpose, turning a normal image into a specific rule violation, much as counterfactually augmented data (Kaushik et al., 2020) revises text to fit a counterfactual label while avoiding unnecessary changes. Our image editor follows written instructions, like InstructPix2Pix (Brooks et al., 2023). Because each unedited image is also labeled normal, the classifier can learn the difference between normal and violation on the same scene and largely under the same capture conditions. Out-of-distribution detection also creates outliers without real ones, but not by editing a given image: Dream-OOD (Du et al., 2023) decodes outliers sampled from low-likelihood regions of a latent space into images, and EOE (Cao et al., 2024) has an LLM write the names of likely outlier classes.

## 2.3. Robustness in Anomaly Detection

Logical checks rest on counts, positions and proportions of parts; brightness reduction and zoom-out change how these appear: parts lose contrast against the background, or shrink and move. Yet robustness to capture shifts has been studied mainly for structural defects. Following corruption benchmarks for image classification (Hendrycks and Dietterich, 2019), Cao et al. (2023) corrupt MVTec AD test images, while Ae-BAD (Zhang et al., 2023), RAD (Cheng et al., 2024), RobustAD (Pemula et al., 2025) and MVTec AD 2 (Heckler-Kram et al., 2026) capture real changes of illumination, viewpoint or background. Moreover, detectors that are almost perfect on MVTec AD degrade on real production data (Baitieva et al., 2025). RobustAD also cautions that robustness to synthetic shifts may not carry over to real ones, and that training with the perturbations of a benchmark can leak its test distribution into training. We therefore make no claim about real capture shifts, and we report a variant of our framework trained without test-overlapping augmentations. We are not aware of a prior evaluation of MVTec LOCO logical anomaly detectors under capture shifts.

Test-time adaptation is an orthogonal route to robustness: Adapted-MoE (Lei et al., 2025) uses the test images themselves to correct the shift between the features of training and test samples (Section 5.1).

## 3. Proposed Method

Fig. 2 gives an overview of our framework with an example from Pushpins. Section 3.1 formulates the problem. Section 3.2 describes how synthetic anomalies are generated from a written description of each category, and Section 3.3 how the normal images are divided into sub-modes. Section 3.4 presents the classifier and its training, Section 3.5 the training augmentation, and Section 3.6 the evaluation protocol under capture shifts.

![](images/8b02b389917c04acce1ad27dd1d53ee4735ef539464b96d76cbaaef92df6e6e0.jpg)  
Figure 2: Overview of our framework, with an example from Pushpins. Top: a person describes the category; the language model proposes anomaly types that break the description and writes an editing prompt for each; the image editor applies a prompt to a normal image; curation keeps the valid edits (red ellipse: two pushpins in one compartment). Bottom: the classifier, trained on K normal classes and the anomaly types, scores a test image on its own. Arrows show what passes between steps. Dashed orange boxes involve a person; gray boxes are pretrained models. The blue box is the only component used at test time. The edited image is generated by the image editor (Gemini 3 Pro Image), a generative AI model.

## 3.1. Problem Formulation

The MVTec LOCO benchmark (Bergmann et al., 2022) provides training and validation splits with only normal (good) images and a test split with normal images, logical anomalies and structural anomalies. We restrict the evaluation to the normal test images and the logical anomalies, which violate constraints on the presence, number, position or combination of parts, and report image-level AUROC. Our framework adds synthetic anomalies to the training and validation splits by editing their normal images. The classifier learns K normal classes (Section 3.3) and N<sub>c</sub> anomaly types (Section 3.2) and is selected on the validation split, so no real anomalous image is used for training or checkpoint selection. At test time, the classifier scores an image alone, without comparing it with normal data. We call this way of scoring a fully discriminative decision.

## 3.2. Anomaly Design and Generation

The classifier needs examples of rule violations, but the training split holds only normal images, so we create such violations from a description of the product. Category description. A person writes a free-text description of the normal images: which parts a correct product contains, how many, where they are and which rules relate them. For Pushpins, for example, each of the 15 compartments holds exactly one pushpin. Human knowledge of the product is a common input in logical anomaly detection: LogSAD (Zhang et al., 2025a) builds its matching rules from GPT-4V proposals prompted with the constraint descriptions of the MVTec LOCO paper, LogicQA (Kwon et al., 2025) gives the same descriptions to a VLM to write checklist questions, LogicAD (Jin et al., 2025) turns a written description of normality into logical formulas, LogiCode (Zhang et al., 2025b) turns rules that experts extract from normal images into code, and SCAN (Li et al., 2026) checks written inspection criteria rule by rule. Appendix C.1 gives our descriptions in the structured form passed to the language model.

Anomaly types and editing prompts. From the category description and a fixed definition of a logical anomaly, the language model (GPT-5; OpenAI, 2025) proposes anomaly types, each a short statement of one rule violation, and writes one editing prompt per type. We set the number of types to N = 20 per category, which is within the range of prior work, from 10 defect types per category in MIRAGE (Hu et al., 2026) to 50 logical-anomaly descriptions requested in LogSAD’s GPT-4V prompt (Zhang et al., 2025a). For Pushpins, type 6 is “One compartment contains two pushpins”, with the prompt “Add an extra pushpin to one compartment so it contains two pushpins” (Fig. 2). A person then checked the types and prompts. Appendix C gives all instructions, and Appendix C.2 all types and prompts.

Image editing. The image editor, Gemini 3 Pro Image (Google DeepMind, 2025), applies one prompt to each normal image of the training and validation splits, and each resulting synthetic anomaly takes the anomaly type of its prompt as its label. Every normal image yields one synthetic anomaly, so N sets only how the synthetic anomalies are divided among the types. Synthetic validation anomalies are made from validation images, so they share no source image with the training set.

Curation. An edit does not always produce a valid anomaly. We therefore inspected the edited images with the help of a VLM and human verification, and kept only valid logical anomalies. Anomaly types with too few valid images are not used as classes (Appendix A); N<sub>c</sub> denotes the number of types that remain. Table A.1 lists the resulting data, and Fig. 3 shows examples.

![](images/4a989300f60bed4d48fe93921df9242f11eb9601ac4115c531d97556765ceb6a.jpg)  
Figure 3: Normal images and synthetic anomalies. Each column shows a normal training image, the synthetic anomaly made from it (red ellipse: edited region) and its editing prompt. All five synthetic anomalies passed curation and are in the final training data. The synthetic anomalies are generated by the image editor (Gemini 3 Pro Image), a generative AI model.

## 3.3. Discovering Sub-Modes of Normal Images

The normal images of a category are not always homogeneous and may form several distinct sub-modes. The normal training images of Juice Bottle, for instance, show three distinct products: cherry, orange and banana juice, each with its matching fruit icon. If these products share a single normal class, the classifier may learn each icon and each juice color as normal in isolation and fail to detect an incorrect combination, which is precisely a logical anomaly. Each sub-mode is therefore assigned its own normal class.

The sub-modes are identified automatically from the normal training images alone. Features are extracted with the pretrained backbone of the classifier (ConvNeXt-L with DINOv3 weights) before any fine-tuning and clustered with a Gaussian mixture model. The number of clusters K is proposed by cross-validated likelihood, and the resulting partition is accepted only if the clusters are well separated, none of them is small and they are stable under resampling; otherwise, the next smaller K is tested, down to a single normal class. Each normal training image takes its cluster as a pseudo-label, fixed before training, and the classifier is then trained with these labels and the anomaly types by standard cross-entropy. Each validation image is assigned the pseudo-label of its most likely cluster. Appendix E lists the thresholds.

The procedure splits Juice Bottle into K = 4 classes, which follow the three fruit types with the banana images in two clusters, and Splicing Connectors into K = 3, one per connector type; the other categories keep one normal class (Table E.1). The two banana clusters are not a distinct product, but both are normal classes, so the extra split does not change which images count as normal. The idea itself is not new: splitting data into pseudo-classes or subclasses by clustering is well established (Ionescu et al., 2019;

Cohen et al., 2022; Sohoni et al., 2020; Liao et al., 2026), and LogSAD (Zhang et al., 2025a) samples the normal images that it gives to GPT-4V from the known normal sub-classes of each MVTec LOCO category. We adapt the idea to logical anomaly detection with synthetic anomalies.

## 3.4. Classification Model and Training Procedure

Classifier and anomaly score. The classifier is a ConvNeXt-L (Liu et al., 2022) with DINOv3 pretrained weights (Siméoni et al., 2025) and a linear head with one output per normal class and per anomaly type (Table B.1). It is fine-tuned end to end with crossentropy on the normal images, labeled with their pseudo-labels, and on the synthetic anomalies, labeled by anomaly type. The anomaly score, Eq. (1), is the probability mass outside the normal classes,

$$
s ( x ) = 1 - \sum _ { k \in { \mathcal { G } } } \operatorname { s o f t m a x } { \big ( } f ( x ) { \big ) } _ { k } ,\tag{1}
$$

where f(x) are the logits of the classifier and $\mathcal { G }$ holds the indices of the K normal classes.

Training. We fine-tune the classifier for 30 epochs at $4 4 8 \times 4 4 8$ pixels with AdamW (Loshchilov and Hutter, 2019); Table B.1 lists all settings. We keep the checkpoint with the lowest cross-entropy on the validation split, which holds the real validation normals with their pseudo-labels and the curated synthetic validation anomalies. This validation-loss checkpoint uses no test image or label. We do not select by validation AUROC, because it saturates at 100% early in training and then no longer distinguishes between checkpoints.

## 3.5. Training Augmentation

Because some standard augmentation operations resemble the test perturbations, we train two variants of the classifier. The first variant, the full recipe, passes normal images and synthetic anomalies through the same random augmentation pipeline (Buslaev et al., 2020): flips and $9 0 ^ { \circ }$ rotations, zoom-out with edge-replicated borders, blur, noise and compression, a photometric block (brightness and contrast, contrast-limited adaptive histogram equalization (CLAHE), tone curve, sharpening), and emboss or posterize (Table B.2). Its zoom-out and brightness change cover the same ranges as the two test perturbations (Section 3.6). The second variant, trained without testoverlapping augmentations, omits the zoom-out and photometric blocks and is otherwise identical; Cao et al. (2023) likewise exclude augmentations that overlap their test corruptions. Since neither test perturbation occurs in its training data, our robustness conclusions are based on this variant. Generic augmentations such as blur and noise remain and may contribute to the robustness (Section 5.1). The full recipe gives the higher clean AUROC, but its robustness could come from its overlap with the test perturbations.

## 3.6. Evaluation Protocol

Perturbations. Robustness is measured on test images with simulated capture shifts. We apply six perturbations from two families to the full-resolution test images, before any preprocessing of the evaluated method (Fig. 4). Brightness reduction multiplies all pixel values by 0.8, 0.6 or 0.4. Zoom-out shrinks the image by 5%, 10% or 15% and pads it back to its original size with edge-replicated borders, so every object stays in view. Together with the unperturbed (clean) condition, the six perturbations define seven test conditions, denoted Clean, B20, B40 and B60 for brightness reduction, and Z5, Z10 and Z15 for zoom-out.

We chose these two families because they are typical uncontrolled capture shifts on an inspection line, due to changes in illumination and camera placement (Baitieva et al., 2025); because they act on the whole image without changing its logical content; and because their severity can be graded. We reduce rather than increase brightness because an increase clips bright regions to uniform white, which can erase parts and boundaries and thus alter the logical content of the image; a reduction preserves all parts, at the cost of coarser quantization in dark regions. Likewise, we zoom out rather than in, because zooming in would crop parts at the border and could itself create an anomaly. Other perturbation families and real capture shifts are left for future work.

Brightness reduction  
Zoom-out  
![](images/6ade072b46063ce0474187f3fd8b716047f8a778d3d81e18d445946548e21e98.jpg)

![](images/92400b8cd4cbf4a62c4896912512519a0012e752bdd319b0954c0b015cf5d07e.jpg)

![](images/23ce187fc2f2dc1346dfcb849ad70b0622fe06e278d76b8f9457a31f60e82c07.jpg)

![](images/c06f833e6eb361568e1265f8137ce8a02159739960f8119fb3bc43f7a3f822ed.jpg)

![](images/e3ae18c7dc86b9f2d7ada7aea1b7bb07de0e94f96b81cde90f07977e243ef33b.jpg)

![](images/1cc36e6580c4b884ad01b801f4d4a20d4637d3b1b2b45e556de4e3d85bade802.jpg)

![](images/e897c70a36c71bc4d0fa8a7d6f7ffe17682403eba291e653fd484d1f27891020.jpg)

![](images/208d502e6f8312d47d0e83429c3ee526dc67cd336c7b5533ebc1b3006a922b4f.jpg)  
Figure 4: Perturbation examples (Juice Bottle). Each row starts from the clean image. Top: brightness reduced by 20%, 40% and 60%. Bottom: zoom-out by 5%, 10% and 15% with edge-replicated borders.

Heterogeneous perturbation protocol. A deployed detector usually compares every score with the same threshold, whatever the condition of the image. In the heterogeneous perturbation protocol, each test image therefore independently receives one of the seven conditions with equal probability, and one AUROC is computed over the mixed test set; Baitieva et al. (2025) applied a similar random drift, drawn per test image, to real industrial datasets, and the mixed test split of MVTec AD 2, a benchmark of mainly structural defects, likewise draws the lighting of each test image at random (Heckler-Kram et al., 2026). We repeat the random assignment 100 times, use the same assignments for every method, and report the mean as the heterogeneous AUROC.

Metrics. We use the clean AUROC $A _ { \mathrm { c l e a n } }$ , the heterogeneous AUROC $A _ { \mathrm { h e t } }$ , and the per-condition AUROC $A _ { \mathrm { { p c } } } ,$ , for which each perturbed condition is scored on its own and the six AUROCs are averaged. We report the drop and the relative drop of Eq. (2),

$$
\mathrm { d r o p } = A _ { \mathrm { c l e a n } } - A _ { \mathrm { h e t } } , \qquad \mathrm { r e l a t i v e ~ d r o p } = 1 0 0 \times { \frac { A _ { \mathrm { c l e a n } } - A _ { \mathrm { h e t } } } { A _ { \mathrm { c l e a n } } - 5 0 } } ,\tag{2}
$$

where 50 is the AUROC of chance; both are positive when AUROC falls, and the relative drop is the percentage of the clean AUROC above chance that is lost. The drop splits into two parts. The within-condition drop, $A _ { \mathrm { c l e a n } } - A _ { \mathrm { p c } }$ , occurs even when all test images share one condition: the method separates anomalies from normal images less well within a darkened or zoomed-out test set. The cross-condition drop, $A _ { \mathrm { { p c } } } \mathrm { ~ - ~ }$ $A _ { \mathrm { h e t } } .$ , arises because the conditions are mixed and the scores of diferent conditions are compared with each other. The worst condition is the perturbed condition with the lowest single-condition AUROC.

## 4. Experimental Evaluation

## 4.1. Experimental Setup

Dataset. We use all five categories (Breakfast Box, Juice Bottle, Pushpins, Screw Bag and Splicing Connectors) of MVTec LOCO AD (Bergmann et al., 2022) with the standard splits. Training and validation use real normal (good) images and synthetic anomalies. The test set contains only real images: normal images and logical anomalies; we leave the structural anomalies out. Table A.1 gives the image counts.

Implementation. The classifier follows the recipe and checkpoint selection of Section 3.4, in two versions (Section 3.5): Ours (no overlap), without test-overlapping augmentations (no zoom-out or brightness augmentation in training), and Ours (full recipe).

Baselines. We compare with four normal-reference methods, CSAD (Hsieh and Lai, 2024), SALAD (Fucka et al.ˇ , 2025), LogSAD (Zhang et al., 2025a) and Patch-Core (Roth et al., 2022), and with AnomalyCLIP (Zhou et al., 2024), a zero-shot method that uses no normal reference. All baselines use the oficial code and released weights without training any network, and each keeps the same model for all seven test conditions; for CSAD, SALAD and LogSAD, our clean measurements are within one point of the reported values. SALAD scores a composition map of each test image; because the trained model that produces these maps is not released, we use the maps that SALAD’s authors released for the clean test images, so these maps are unafected by the perturbation, which favors SALAD. Appendix D gives the settings.

Table 2: Main results. Clean and heterogeneous image-level logical AUROC (%), mean over the five categories.
<table><tr><td>Method</td><td>Reported</td><td>Clean</td><td>Hetero- geneous</td><td>Drop (points)</td><td>Relative drop (%)</td></tr><tr><td>Ours (no overlap)</td><td>一</td><td>93.2</td><td>92.5</td><td>0.7</td><td>1.6</td></tr><tr><td>Ours (full recipe)</td><td>一</td><td>94.9</td><td>94.1</td><td>0.8</td><td>1.8</td></tr><tr><td>CSAD</td><td>96.7</td><td>97.7</td><td>68.1</td><td>29.6</td><td>62.1</td></tr><tr><td>SALADª</td><td>96.5</td><td>96.4</td><td>62.3</td><td>34.1</td><td>73.5</td></tr><tr><td>LogSAD</td><td>89.3</td><td>89.2</td><td>73.9</td><td>15.3</td><td>39.0</td></tr><tr><td>PatchCoreb</td><td>1</td><td>75.4</td><td>65.7</td><td>9.7</td><td>38.2</td></tr><tr><td>AnomalyCLIPc</td><td>一</td><td>57.3</td><td>54.3</td><td>3.0</td><td>41.1</td></tr><tr><td>SALAD†</td><td>98.7</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>PSAD</td><td>98.1</td><td></td><td></td><td>一</td><td></td></tr><tr><td>SAM-LAD</td><td>95.8</td><td>一</td><td>一</td><td>一</td><td></td></tr></table>

Reported: five-category logical AUROC from the method’s own paper; the other columns are our measurements, and our clean values for CSAD, SALAD and LogSAD are within one point of the reported ones. Drop = clean − heterogeneous, in points; relative drop = 100 × drop/(clean − 50), the share of the abovechance clean AUROC that is lost; both are positive when AUROC falls. Ours (no overlap): trained without test-overlapping augmentations. Bottom block: methods with only a reported clean result, whose robustness we could not measure. Appendix D gives the settings of all baselines. <sup>a</sup> We run SALAD with its own unsupervised composition maps; SALAD<sup>†</sup> uses PSAD’s supervised part maps. The trained composition-map model is not released, so we use the maps that SALAD’s authors released for the clean images; these maps are therefore unafected by the perturbation, which favors SALAD. <sup>b</sup> PatchCore reports no MVTec LOCO result. <sup>c</sup> AnomalyCLIP uses no normal reference and is near chance on clean images.

Metrics. We report image-level AUROC on the logical anomalies, in percent and averaged over the five categories, together with the drop and the relative drop (Section 3.6).

## 4.2. Main Results

We first quantify the performance degradation of existing methods under capture shifts (Table 2, Fig. 1). Under the heterogeneous perturbation protocol, the normalreference methods we tested lose 9.7 to 34.1 points, a relative drop of 38.2% to 73.5%. Notably, among these methods, the two with the highest clean AUROC exhibit the largest degradation: CSAD drops from 97.7% to 68.1% and SALAD from 96.4% to 62.3%. AnomalyCLIP, near chance, loses only 3.0 points, which is still a relative drop of 41.1%. AnomalyCLIP’s low clean AUROC agrees with a related zero-shot evaluation on MVTec LOCO, in which AnomalyCLIP, AdaCLIP and WinCLIP reach 61.6%, 60.9% and 53.2% image-level AUROC, not reported separately for logical anomalies (Fucka et al.ˇ , 2026b). The order also changes: LogSAD moves ahead of CSAD and SALAD, so in our comparison clean AUROC does not predict the ranking under capture shifts.

In contrast, our framework is largely unafected: AUROC decreases by 0.7 points for Ours (no overlap), from 93.2% to 92.5%, and by 0.8 points for Ours (full recipe), from 94.9% to 94.1%, corresponding to relative drops of 1.6% and 1.8%. The same holds in every category (Table G.1): the heterogeneous AUROC of our framework stays within 2.4 points of its clean value, while CSAD and SALAD lose 12.6 to 41.3 points. As explained in Section 3.5, the robustness claim rests on Ours (no overlap).

On clean images, however, Ours (full recipe) is 1.5 to 2.8 points below SALAD and CSAD, and below the three methods in the bottom block of Table 2, SALAD<sup>†</sup> (Fuckaˇ et al., 2025), PSAD (Kim et al., 2024) and SAM-LAD (Peng et al., 2025), whose robustness we could not measure (Appendix D). The gap to SALAD and CSAD comes mainly from Screw Bag and Splicing Connectors (Table G.1).

## 4.3. Robustness Analysis

A performance drop shows how much a method degrades, but not why. Under reduced brightness, degradation can arise in two distinct ways, each requiring a diferent remedy. First, the method may no longer distinguish darkened anomalies from darkened normal images, even when evaluated only on darkened images (a within-condition drop). Second, it may preserve this ranking within darkened images, but shift all scores upward. In that case, a darkened normal image may receive a higher score than a clean anomaly, preventing a single threshold from separating the mixed test set despite good within-condition ranking (a cross-condition drop). Per-condition AUROC (Table 3; Section 3.6) distinguishes these efects: its diference from clean AUROC quantifies the within-condition drop, whereas its diference from heterogeneous AUROC quantifies the cross-condition drop.

Table 3: Decomposition of the drop. Image-level logical AUROC (%), five-category mean. Per-condition: the mean AUROC of the six perturbed conditions, each scored on its own (Table F.1 gives each condition). Within-condition drop = clean − per-condition: separation lost even within one condition. Cross-condition drop = per-condition − heterogeneous: lost mainly because the scores move between conditions. The two add up to the drop of Table 2 (Section 3.6); for each normal-reference method, the larger one is in bold. Worst condition: the perturbed condition with the lowest five-category mean (B40: brightness −40%; Z10: zoom-out 10%; Z15: zoom-out 15%).
<table><tr><td>Method</td><td>Clean</td><td>Per- condition</td><td>Hetero- geneous</td><td>Within-condition drop</td><td>Cross-condition drop</td><td>Worst condition</td></tr><tr><td>Ours (no overlap)</td><td>93.2</td><td>92.8</td><td>92.5</td><td>0.4</td><td>0.3</td><td>91.0 Z15</td></tr><tr><td>Ours (full recipe)</td><td>94.9</td><td>94.1</td><td>94.1</td><td>0.8</td><td>0.0</td><td>93.0 Z15</td></tr><tr><td>CSAD</td><td>97.7</td><td>75.0</td><td>68.1</td><td>22.7</td><td>6.9</td><td>61.9 Z15</td></tr><tr><td>SALAD</td><td>96.4</td><td>85.6</td><td>62.3</td><td>10.8</td><td>23.3</td><td>78.0 B40</td></tr><tr><td>LogSAD</td><td>89.2</td><td>77.5</td><td>73.9</td><td>11.7</td><td>3.6</td><td>67.6 Z10</td></tr><tr><td>PatchCore</td><td>75.4</td><td>68.3</td><td>65.7</td><td>7.1</td><td>2.6</td><td>57.3 Z15</td></tr><tr><td>AnomalyCLIP</td><td>57.3</td><td>54.8</td><td>54.3</td><td>2.5</td><td>0.5</td><td>53.5 Z15</td></tr></table>

CSAD and SALAD fail in these two ways. Most of the drop of CSAD is withincondition drop (22.7 of 29.6 points): CSAD separates normal from anomalous images worse even within one condition. Most of the drop of SALAD is cross-condition drop (23.3 of 34.1 points): SALAD’s scores move with the condition. For our framework, however, both parts stay below one point.

The diference matters: only the cross-condition drop could be repaired by percondition recalibration. Such a recalibration keeps the order of the scores within each condition, so it cannot change the per-condition AUROC, which stays at 75.0%, 85.6% and 77.5% for CSAD, SALAD and LogSAD, below our 92.8% and 94.1%. Our worst condition, the 15% zoom-out (91.0% and 93.0%), is also above the worst condition of every baseline.

![](images/98e02e1023bd6fc2d427f85959c96cf83a5ac1fcec4d46b46df8c3eb1b0d9672.jpg)  
Figure 5: Normal images scoring above the median clean anomaly. Each cell gives the share of normal (good) test images whose anomaly score is higher than the median score of the clean anomalous test images, as a five-category mean (%). The clean column is the starting point: a detector that separates clean images well starts near 0%, while PatchCore, which separates them less well, already starts at 12%. A detector whose normal scores do not move with the capture conditions keeps its clean value in every column; a rise means that a brightness reduction or a zoom-out alone moves normal images above the median clean anomaly. AnomalyCLIP is left out: near chance on clean images, it already scores 38% of its normal images above its median clean anomaly.

Fig. 5 shows how far the scores move with the condition: at the stronger settings, every normal-reference method we tested scores many normal test images above the median clean anomaly, while the classifier keeps almost all of them below it. Section 5.1 discusses possible causes of this diference.

## 4.4. Ablation Study

Table 4: Ablation. Image-level logical AUROC (%), mean over the five categories; each row below the first removes one component from the full recipe.
<table><tr><td>Variant</td><td>Clean</td><td>Heterogeneous</td></tr><tr><td>Ours (full recipe)</td><td>94.9</td><td>94.1</td></tr><tr><td>Without normal sub-mode discovery</td><td>92.9</td><td>91.7</td></tr><tr><td>Ours (no overlap): without test-overlapping augmentations</td><td>93.2</td><td>92.5</td></tr></table>

Table 4 removes one component at a time from the full recipe. Normal sub-mode discovery adds 2.0 points on clean images and 2.4 under the heterogeneous perturbation protocol, and changes only Juice Bottle and Splicing Connectors, the two categories that it splits (Section 3.3; per category in Appendix E). Because this comparison keeps the full augmentation, we interpret it as a gain in accuracy, not as evidence of robustness. The test-overlapping augmentations add 1.7 points on clean images and 1.6 under the heterogeneous perturbation protocol; without them, as in Ours (no overlap), our framework is less accurate but still robust.

## 5. Discussion

## 5.1. Source of Robustness

This section discusses possible reasons why the normal-reference methods we tested degrade substantially under the heterogeneous perturbation protocol, whereas our framework is largely unafected (Section 4.2). Normal-reference detectors compare test images with normal data captured under a single lighting condition and camera position. We hypothesize that, as a result, reduced brightness or zooming out may increase the anomaly score of a defect-free product, making it dificult to distinguish a logical anomaly from a capture-related shift. Out-of-distribution detection makes a similar distinction between semantic shifts and covariate shifts, which change appearance without altering semantic content (Yang et al., 2023). In these terms, a logical anomaly corresponds to a semantic shift and a capture shift to a covariate shift.

Evidence. Fig. 5 supports this hypothesis: at the strongest settings, CSAD and SALAD score most normal test images above the median clean anomaly. A fully discriminative decision behaves diferently: Ours (no overlap) retains its AUROC although neither perturbation occurs in its training, almost none of its normal test images score above the median clean anomaly (Fig. 5), and both parts of its drop are close to zero (Section 4.3). Candidate explanations. These observations show that our decision is robust, but do not explain why. One possibility is that the score is relative: it weighs evidence for normal classes against evidence for anomaly types learned from images captured under the same conditions. In contrast, a distance-to-normal score may increase with any appearance change. Remaining training augmentations, such as blur and noise, may also contribute.

Two failure modes. The two failure modes described in Section 4.3 have diferent practical implications. When performance loss is primarily a cross-condition drop, as for SALAD, the method largely preserves its ranking within each condition. The main issue is that a single decision threshold no longer works across conditions. This can be addressed outside the detector by monitoring lighting and camera placement and applying condition-specific thresholds.

When the loss is primarily a within-condition drop, as for CSAD, condition-specific thresholds cannot restore performance (Section 4.3). In this case, the practical safeguard is to maintain capture conditions as closely as possible to those used in training. Test-time adaptation. Both safeguards operate outside a trained detector; test-time adaptation (Section 2.3) instead adapts to the test images and could be applied to our framework as well. Our results address a prior question: whether the decision itself is robust without adaptation.

## 5.2. Limitations and Open Questions

Clean AUROC. On clean images, the full recipe is 1.5 to 2.8 points below SALAD and CSAD (Section 4.2). On a line where capture is held as tightly as in training, a normal-reference method may therefore be the better choice.

Human input. A person writes the short description of each category, which defines what counts as normal, and checks what is generated from it. Automating both steps is future work.

Synthetic anomalies. The description also limits the anomaly types available for training: types derived from it cannot guarantee that every real anomaly is covered, and we did not evaluate performance separately on real anomalies of types outside the generated list. Generation artifacts, such as blending seams, could act as shortcuts; because the test set contains only real images, any harm from them is already reflected in the reported AUROC.

Scope of the evaluation. Our evaluation is narrower than the setting that motivates it. Our test perturbations are synthetic and come from two families, brightness reduction and zoom-out (Section 3.6 gives the reasons). Real low-light capture also adds sensor noise and uneven illumination, and the object may shift relative to the camera; our perturbations simulate neither. Robustness to wider and real capture shifts remains open.

Closed models. The language model and the image editor are closed models, so the generation stage cannot be repeated exactly; the material we will release is listed in the data availability statement.

Role of the anomaly design. Our results evaluate the framework as a whole and do not isolate the contribution of designing anomalies with a language model. Whether this design outperforms simpler synthesis, such as cut-and-paste anomalies (Li et al., 2021), remains open and would require a controlled comparison with the rest of the framework unchanged.

## 6. Conclusion

We showed that the normal-reference detectors are fragile under two simple capture shifts. Under the heterogeneous perturbation protocol, they lose 9.7 to 34.1 AUROC points, 38.2% to 73.5% of their performance above chance. They degrade in two ways: some lose discrimination within a condition, which no per-condition recalibration can repair, and others shift their scores between conditions.

We proposed a framework in which a classifier is trained on normal images and synthetic anomalies generated from brief category descriptions, then scores test images without reference to normal data. Anomaly types derived from a short description of each category are applied to normal images with an instruction-based image editor, and the classifier learns to separate the edited images from automatically discovered normal sub-modes. With its full training recipe, the framework reaches 94.9% AUROC on clean images, 1.5 to 2.8 points below SALAD and CSAD, and 94.1% under the perturbation protocol; trained without brightness or zoom-out augmentation, it loses only 0.7 points (93.2% to 92.5%).

These results suggest that clean AUROC alone may not predict how a detector behaves on a production line, and that a change in lighting or camera position alone can cause normal-reference detectors to score defect-free products as anomalous. Our evaluation, however, covers one benchmark and two synthetic shift families, and a person still writes the category description and checks what is generated from it. Future work includes automating these two steps, testing wider and real capture shifts, localizing the detected anomalies and extending the framework to structural anomalies.

## Declaration of competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This work was supported by the Scientific and Technological Research Council of Türkiye (TÜB˙ITAK) under the B˙IDEB 2211 National PhD Scholarship Program.

## CRediT authorship contribution statement

Ufuk Efe: Conceptualization, Methodology, Software, Investigation, Data curation, Visualization, Writing (original draft). Alptekin Temizel: Methodology, Supervision, Writing (review and editing).

## Data availability

The evaluation is conducted on the publicly available MVTec LOCO AD dataset (Bergmann et al., 2022). The category descriptions, anomaly types and editing prompts, the curated synthetic training images and the code used in this study will be made publicly available upon acceptance of the manuscript; the synthetic images, which are derived from MVTec LOCO AD, will be released under its CC BY-NC-SA 4.0 license.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic) to assist with drafting, editing and checking the manuscript text, and Claude (Anthropic) and Codex (OpenAI) to assist with writing and checking the analysis code. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The language model and the image editor used to generate synthetic training data are part of the proposed method and are described in Section 3; the synthetic anomalies shown in Fig. 2, Fig. 3 and Figs. C.1– C.5 were generated by this image editor.

Table A.1: Data per category. Real MVTec LOCO normal (good) images of the train, validation and test splits and the real logical test anomalies; synthetic anomalies used for training and validation; the number of normal sub-modes K; and the number of the 20 designed anomaly types used in training
<table><tr><td></td><td colspan="3">Good</td><td>Test</td><td colspan="2">Synthetic</td><td></td><td></td></tr><tr><td>Category</td><td>Train</td><td>Val</td><td>Test</td><td>logical</td><td>Train</td><td>Val</td><td>K</td><td>Types</td></tr><tr><td>Breakfast Box</td><td>351</td><td>62</td><td>102</td><td>83</td><td>311</td><td>55</td><td>1</td><td>19</td></tr><tr><td>Juice Bottle</td><td>335</td><td>54</td><td>94</td><td>142</td><td>329</td><td>53</td><td>4</td><td>20</td></tr><tr><td>Pushpins</td><td>372</td><td>69</td><td>138</td><td>91</td><td>273</td><td>51</td><td>1</td><td>17</td></tr><tr><td>Screw Bag</td><td>360</td><td>60</td><td>122</td><td>137</td><td>258</td><td>45</td><td>1</td><td>17</td></tr><tr><td>Splicing Connectors</td><td>360</td><td>60</td><td>119</td><td>108</td><td>292</td><td>48</td><td>3</td><td>18</td></tr></table>

Table A.1 lists the number of images per category and split. The synthetic columns count the synthetic anomalies kept after curation (Section 3.2), and K is the number of normal sub-modes (Appendix E). An anomaly type becomes a class only if curation keeps at least eight of its training images and at least one of its validation images; “Types” counts these classes (N ), and the other types are marked “(not used)” in Tables C.1–C.5.

## Appendix B. Training details

This appendix provides the details needed to reproduce the training of the classifier. Table B.1 lists the training settings, which are the same for all categories, and Table B.2 the augmentation pipeline. Training runs with PyTorch’s deterministic algorithms enabled, so a run is reproducible on the same hardware and software. One run takes about 40 minutes on one NVIDIA GeForce RTX 3080 (12 GB), with PyTorch 2.9.1, timm 1.0.22 (Wightman, 2019) and Albumentations 2.0.8 (Buslaev et al., 2020).

Table B.1: Training settings of the classifier (all categories).
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>ConvNeXt-L (Liu et al., 2022), DINOv3 LVD-1689M weights (timm: convnext_1arge.dinov3_1vd1689m; Siméoni et al., 2025)</td></tr><tr><td>Fine-tuning</td><td>full network, no frozen layers</td></tr><tr><td>Head</td><td>global average pooling, LayerNorm, linear layer</td></tr><tr><td>Outputs</td><td>K normal sub-modes + 20 anomaly types (21 outputs; Juice Bottle 24, Splicing Connectors 23); a type not used as a class keeps its output but has no training</td></tr><tr><td>Input</td><td>images resized to 448 × 448, ImageNet mean and standard deviation</td></tr><tr><td>Optimizer</td><td>AdamW (Loshchilov and Hutter, 2019),  $\beta = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - 8 } .$  weight decay 0.05</td></tr><tr><td>Learning rate</td><td>10-⁵ for backbone and head, constant (no schedule)</td></tr><tr><td>Batch</td><td>2 images, gradient accumulation over 2 steps (effective batch 4)</td></tr><tr><td>Epochs</td><td>30</td></tr><tr><td>Regularization</td><td>no label smoothing, no dropout, no stochastic depth lowest cross-entropy on the validation split (real normal images and synthetic</td></tr><tr><td>Checkpoint</td><td>anomalies)</td></tr><tr><td>Seed</td><td>42</td></tr></table>

Table B.2: Augmentation pipeline (Albumentations), applied in this order to normal images and synthetic anomalies alike. A “one of” block applies one of its members with the given probability. The blocks marked \* contain the test perturbations and are removed in the variant without test-overlapping augmentations. Validation and test images are only resized and normalized.
<table><tr><td>Block</td><td>p</td><td>Operators and ranges</td></tr><tr><td>Flips and rotation</td><td>0.5 each</td><td>horizontal flip, vertical flip, rotation by a multiple of 90°</td></tr><tr><td>Zoom-out*</td><td>0.3</td><td>downscale by 0–15%, edge-replicate padding back to 448 × 448</td></tr><tr><td>Blur (one of)</td><td>0.5</td><td>Gaussian (σ 1–5), defocus (radius 2–3), motion (kernel 3–7), zoom blur (factor up to 1.05)</td></tr><tr><td>Noise (one of)</td><td>0.5</td><td>per-pixel multiplicative noise (0.7–1.3), Gaussian noise (std 0.01–0.05), JPEG compression, downscaling by 0.5</td></tr><tr><td>Photometric* (one of)</td><td>0.5</td><td>brightness (−60% to +40% of the image mean) and contrast (±0.2), CLAHE (clip limit 1–4), random tone curve, sharpening</td></tr><tr><td>Color (one of)</td><td>0.25</td><td>emboss, posterize (4 bits)</td></tr></table>

## Appendix C. Prompts and generation settings

This appendix traces the synthetic anomalies from the category description to the edited image (Fig. 2). A person writes a free-text description of the category, and the language model converts it into a structured description (Step 1a), proposes anomaly types based on that description (Step 1b) and writes one editing prompt per type (Step 1c). The image editor (gemini-3-pro-image-preview, accessed through Vertex AI) then applies the prompts (Step 2). The instructions difer slightly between categories; the diferences are marked below.

Step 1a: category description. For each category, a person wrote a short free-text description of the normal images. The language model converted this text into a structured description with the following fixed instruction. Only this structured description is passed on to Steps 1b and 1c; all such descriptions are given in Appendix C.1.

Using the given dataset description, create a structured dataset description. It should be written so that it generalizes. It should be in English. Keep it plain and clear but complete. I only want the dataset description; do not add unnecessary detail. [Screw Bag and Splicing Connectors only:] Below is an example of a good description writtenfor another dataset. [the structured description of another category]

Logical anomaly definition. The following definition, based on Bergmann et al. (2022), is part of the Step 1b and Step 1c prompts:

Logical anomaly: Violations of the logical or geometrical constraints that govern component presence and relations, e.g., permissible objects in invalid locations, missing/extra parts, or incorrect arrangements. These depend on long-range dependencies and global composition rather than local textures, requiring reasoning over counts, positions, and relationships.

Step 1b: anomaly types. The structured description and the definition are passed with the following prompt (spelling corrected):

Considering the provided structured dataset description and the definition of a logical anomaly, generate a clear, concise, and specific list of 20 logical anomaly cases that could appear in this dataset. Each anomaly should be stated as a short, explicit sentence. [Not in the Juice Bottle version:] Each anomaly should be clear, should not include “or” statements. Each anomaly should be independent from each other, i.e. one anomaly type cannot be produced by another anomaly type.

[Splicing Connectors only:] IMPORTANT (splicing\_connectors specific): Write anomalies in a type-agnostic way where possible. For example: Instead of “T2 connector replaced with T3” → “One connector is a diferent type than the other”. Instead of“Yellow cable used with T3 connectors” → “Cable color does not match the connector type”. Use relative terms: “fewer ports”, “more ports”, “diferent color”, “wrong type”. This is because the input image type (T2/T3/T5) is unknown at generation time.

DATASET DESCRIPTION: [structured category description]

LOGICAL ANOMALY DEFINITION: [definition above]

Step 1c: editing prompts. The 20 types are converted into editing instructions with the following prompt:

ROLE: You turn anomaly statements into short image edit prompts.

YOU WILL RECEIVE: 1) DATASET DESCRIPTION 2) LOGICAL ANOMALY DEFINITION 3) POSSIBLE ANOMALIES LIST (numbered)

TASK: Convert EACH anomaly item into ONE straightforward, minimal edit prompt for an AI image editing tool that modifies the given image to create exactly that anomaly. [Not in the Juice Bottle version:] These prompts will be given to an AI image editor along with a normal (good) reference imagefrom the dataset, with the goal oftransforming that image into an anomalous one using the prompt.

RULES: Keep the same numbering as the input list. One sentence per item. State only the change needed. Do not give specific examples.

[Splicing Connectors only:] ADDITIONAL RULES (splicing\_connectors specific): Since the input image’s connector type (T2/T3/T5) is unknown at generation time, prompts must be type-agnostic. Use relative terms: “a diferent type”, “fewer ports”, “more ports”, “the wrong color”. Do NOT use specific values: “T2”, “T3”, “T5”, “yellow”, “blue”, “red”. Example: Instead of “Change cable to yellow” → “Change the cable to a color that does not match the connector type”. Example: Instead of “Replace with T5 connector” → “Replace one connector with a connector that has more ports”.

DATASET DESCRIPTION: [structured category description]

LOGICAL ANOMALY DEFINITION: [definition above]

POSSIBLE ANOMALIES LIST: [the 20 types from Step 1b]

Step 2: image editing. The image editor receives the editing prompt and the image and returns one candidate with its default sampling settings (temperature 1.0, top-p 0.95, top-k 64); the output is resized to the input resolution. This step is not exactly reproducible, because the image editor is a closed preview model, so we will release the category descriptions, prompts and synthetic anomalies.

## Appendix C.1. Category descriptions

These are the structured descriptions produced in Step 1a, as passed to the language model in Steps 1b and 1c (only the formatting is adapted). They define what a normal image is, and the anomaly types of Appendix C.2 are derived from them as violations.

## Breakfast Box.

Dataset Content: A collection of images showing a two-compartment breakfast tray containing food items arranged in a consistent layout.

## Tray Characteristics

• The tray is always a white foam tray with two compartments.

• The left compartment is consistently narrower, and the right compartment is consistently wider.

## Left Compartment Content (Fixed Set, Variable Order)

• The left compartment contains exactly three whole fruits in every image: 2 oranges, 1 peach.

• The fruits are placed in a single vertical column (top-to-bottom, one fruit per position).

• The order is variable across images (any permutation of two oranges and one peach is allowed).

• No other items appear in the left compartment.

Right Compartment Content (Fixed Regions). The right compartment is consistently divided into two content regions:

1. Upper Region (Cornflakes): Contains only cornflakes. Cornflakes occupy more than half of the right compartment’s area. The cornflakes form a single contiguous, unmixed region (no other items present within this upper area).

2. Lower Region (Mixed Banana and Almonds): Contains a mixture of sliced banana and almonds. Banana slices and almonds are randomly interspersed (no fixed pattern or separation). No other items appear in the lower region.

## Juice Bottle.

Dataset Content: The dataset contains images of transparent bottles filled with juice.

## Bottle Characteristics

• All bottles are transparent and filled with juice.

• Bottles are nearly full, leaving a small empty space at the top.

• Bottles have no caps and their openings are uncovered.

## Labels on Bottles. Each bottle has exactly two labels:

1. First Label (Fruit Icon Label): Positioned at the center of the bottle. Contains an icon representing the type of fruit juice inside. The color of the juice inside the bottle corresponds with the fruit icon on this label. The fruit icon is centered on this label.

2. Second Label (“100% Juice”): Located at the bottom part of the bottle. Always contains the text “100% Juice”.

## Pushpins.

Dataset Content: A collection of images showing a transparent plastic organizer box containing pushpins arranged in a fully regular compartment layout.

## Container Characteristics

• The container is always the same type of transparent plastic box.

• The box is partitioned into a fixed 3 × 5 grid of compartments: 3 rows, 5 columns, 15 compartments total.

• Each compartment is a distinct separate cell (i.e., a divider-based section).

## Compartment Content (Fixed Count, Variable Placement)

• Every image contains exactly one pushpin in each compartment.

• All pushpins are: Yellow; The same size.

• The pose and placement of each pushpin within its compartment are variable: Orientation is random. Position within the compartment is random.

• No compartment contains zero pushpins or more than one pushpin.

• No additional objects appear in the compartments or the box.

## Screw Bag.

Dataset Content: A collection of images showing a transparent plastic bag con taining a fixed set of metallic hardware components.

## Container Characteristics

• The container is always a transparent plastic bag.

• All components appear inside the bag.

Bag Content (Fixed Set, Fixed Counts). Each image contains exactly the following items inside the bag:

• 2 nuts (consistent standard size across all samples)

• 2 washers (consistent standard size across all samples)

• 2 screws: 1 long screw (consistent standard length across all samples); 1 short screw (consistent standard length across all samples)

## Variability Constraints

• The positions and orientations of all items may vary arbitrarily across images.

• The item counts and the long vs. short screw length distinction are invari ant for all samples.

## Splicing Connectors.

Dataset Content: A collection of images showing electrical cable splicing connectors placed on a metal grid background.

Scene Structure (Fixed Counts). Each image contains exactly (no more, no less):

• 2 splicing connectors (no more, no less; one on the left, one on the right)

• 1 cable connecting the two connectors (no more, no less)

Connector Types and Cable Color Mapping. Connectors can be one of the following types (defined by the number of ports/levers):

• T2: 2-port connector (small, 2 levers) → Yellow cable

• T3: 3-port connector (medium, 3 levers) → Blue cable

• T5: 5-port connector (large, 5 levers) → Red cable

## Consistency Rules (Invariant Across All Normal Samples)

• The two connectors in the same image must be the same type (both T2, both T3, or both T5).

• The cable color is strictly determined by connector type (T2=yellow, T3=blue, T5=red), with no exceptions. Only these 3 colors exist in normal samples; cables of other colors never appear.

• The cable must be inserted into both connectors (i.e., physically connected on both ends).

• The cable must be connected to the same lever index/row on both connectors (the specific lever used may vary between images, but must match left-to-right).

• All connector levers must be closed.

• The connectors must be upright and aligned, facing each other in a straight configuration.

• The cable length must remain within a standard expected range (not ex cessively short or long).

• The cable must be intact and continuous (not broken or cut).

## Allowed Variations

• The positions, spacing, orientation, and viewpoint of the connectors and cable may vary between images, as long as all rules above remain satisfied.

## Appendix C.2. Anomaly types, editing prompts and examples

Tables C.1–C.5 list the 20 anomaly types of each category from Step 1b and their editing prompts from Step 1c, exactly as used for generation. Types marked “(not used)” fail the class rule of Appendix A. Figs. C.1–C.5 show, for every used type, a normal training image and a synthetic anomaly generated from that image with the type’s prompt.

Table C.1: Breakfast Box: anomaly types and editing prompts.
<table><tr><td></td><td>ID Anomaly type</td><td>Editing prompt</td></tr><tr><td></td><td>1 The tray is black.</td><td>Change the tray color to black.</td></tr><tr><td></td><td>2 The tray material is metal.</td><td>Change the tray material to metal.</td></tr><tr><td></td><td>3 The tray has three compartments.</td><td>Add a third compartment to the tray.</td></tr><tr><td></td><td></td><td>4 The left compartment is wider than the right com- Resize the compartments so the left compartment is wider than the right compartment.</td></tr><tr><td></td><td>partment. 5 The left compartment contains two fruits.</td><td>Remove one fruit from the left compartment so it contains only two whole fruits.</td></tr><tr><td></td><td>6 The left compartment contains four fruits.</td><td>Add one extra whole fruit to the left compartment so it contains four whole fruits.</td></tr><tr><td></td><td></td><td>7 The left compartment contains one orange and two Replace one orange in the left compartment with a peach so it contains one orange and two peaches.</td></tr><tr><td></td><td>peaches. 8 The left compartment contains three oranges.</td><td>Replace the peach in the left compartment with an orange so it</td></tr><tr><td></td><td>9 The left compartment contains an apple.</td><td>contains three oranges. Add an apple to the left compartment.</td></tr><tr><td></td><td>segment.</td><td>10 The left compartment contains a peeled orange Replace one whole fruit in the left compartment with a peeled orange segment.</td></tr><tr><td></td><td>ment.</td><td>11 The left compartment contains a peeled peach seg- Replace one whole fruit in the left compartment with a peeled peach segment.</td></tr><tr><td></td><td>the same vertical position. (not used)</td><td>12 Two fruits in the left compartment overlap within Move one fruit in the left compartment to overlap another fruit at the same vertical position.</td></tr><tr><td></td><td></td><td>13 A whole orange is placed in the right compartment. Add a whole orange into the right compartment.</td></tr><tr><td></td><td></td><td>14 Cornflakes appear in the lower region of the right Add cornflakes into the lower region of the right compartment.</td></tr><tr><td></td><td>compartment.</td><td>15 Banana slices appear in the upper cornflakes re- Add banana slices into the upper cornflakes region of the right</td></tr><tr><td></td><td>gion. 16 Almonds are embedded inside the upper cornflakes Add almonds embedded within the upper cornflakes region of</td><td>compartment.</td></tr><tr><td></td><td>region.</td><td>the right compartment. 17 The cornflakes region occupies less than half of the Reduce the cornflakes region so it covers less than half of the</td></tr><tr><td></td><td>right compartment area.</td><td>right compartment area. 18 The cornflakes region is split into two disconnected Split the cornflakes region into two disconnected patches.</td></tr><tr><td></td><td>patches.</td><td>19 The lower region contains banana slices without al- Remove all almonds from the lower region so it contains only</td></tr><tr><td></td><td>monds.</td><td>banana slices. 20 The lower region contains almonds without banana Remove all banana slices from the lower region so it contains</td></tr><tr><td></td><td>slices.</td><td>only almonds.</td></tr></table>

![](images/440d19fe9809a308bcd113aabd89d363a838e7f076a44971cbccdbfb996ba2f0.jpg)  
Figure C.1: Breakfast Box: example pairs. Normal training images (left) and synthetic anomalies made from them (right). Numbers are the type IDs of Table C.1. The synthetic anomalies are generated by the image editor (Gemini 3 Pro Image), a generative AI model (Section 3.2).

Table C.2: Juice Bottle: anomaly types and editing prompts.
<table><tr><td>ID</td><td>Anomaly type</td><td>Editing prompt</td></tr><tr><td>1</td><td>Bottle is opaque.</td><td>Make the bottle opaque.</td></tr><tr><td>2</td><td>Bottle contains no juice (empty or water).</td><td>Remove the juice, leaving the bottle empty.</td></tr><tr><td>3</td><td>Liquid level is far below nearly full&#x27;.</td><td>Lower the liquid level to well below nearly full.</td></tr><tr><td>4</td><td>Bottle is completely full with no headspace.</td><td>Fill the bottle to the brim, removing all headspace.</td></tr><tr><td>5</td><td>Headspace appears at the bottom or side.</td><td>Relocate the headspace to the bottom of the bottle.</td></tr><tr><td>6</td><td>Bottle has a cap attached.</td><td>Add a cap to the bottle.</td></tr><tr><td>7</td><td>Bottle opening is covered by a seal/film/cork.</td><td>Cover the bottle opening with a seal.</td></tr><tr><td>8</td><td>Bottle has more than two labels.</td><td>Add an extra label to the bottle.</td></tr><tr><td>9</td><td>Bottle has only one label.</td><td>Remove one label so only a single label remains.</td></tr><tr><td>10</td><td>Fruit icon label is not at the center of the bottle.</td><td>Move the fruit icon label away from the center of the bottle.</td></tr><tr><td>11</td><td>Fruit icon label lacks any fruit icon.</td><td>Remove the fruit icon from the fruit icon label.</td></tr><tr><td>12</td><td>Fruit icon is off-center within its label.</td><td>Shift the fruit icon within its label off-center.</td></tr><tr><td>13</td><td>Fruit icon depicts a different fruit than the juice.</td><td>Change the fruit icon to depict a different fruit than the juice.</td></tr><tr><td></td><td>14 Juice color does not match the fruit icon.</td><td>Change the juice color so it does not match the fruit icon.</td></tr><tr><td>15</td><td>100% Juice&#x27; label is not at the bottom.</td><td>Move the “100% Juice&quot; label away from the bottom.</td></tr><tr><td>16</td><td>Bottom label text differs from exactly ‘100%</td><td>Replace the bottom label text with wording other than “100%</td></tr><tr><td></td><td>Juice&#x27;. 100% Juice&#x27; label is missing.</td><td>Juice.&quot; Remove the “100% Juice&quot; label.</td></tr><tr><td>17 18a</td><td>Both labels say &#x27;100% Juice&#x27; (no fruit icon label Remove two labels so no labels remain.</td><td></td></tr><tr><td>19</td><td>present). Label roles are swapped (fruit icon at bottom, Swap the positions of the fruit icon label and the “100% Juice&quot;</td><td></td></tr><tr><td>20</td><td>&#x27;100% Juice&#x27; at center). Fruit icon label shows a non-fruit graphic (e.g.,</td><td>label. Replace the fruit icon with a non-fruit graphic on its label.</td></tr></table>

Type 18: the prompt removes both labels instead of following the statement; the class follows the prompt.

![](images/a315e3dbc9fc1544109ddf96c52e979ae3b7769cb8b344b5d1f88522f6e39979.jpg)  
Figure C.2: Juice Bottle: example pairs. Normal training images (left) and synthetic anomalies made from them (right). Numbers are the type IDs of Table C.2. The synthetic anomalies are generated by the image editor (Gemini 3 Pro Image), a generative AI model (Section 3.2).

Table C.3: Pushpins: anomaly types and editing prompts.
<table><tr><td></td><td>ID Anomaly type</td><td>Editing prompt</td></tr><tr><td></td><td>1 All compartments are empty; no pushpins are Remove all pushpins from every compartment so the entire box present in the box.</td><td>is empty.</td></tr><tr><td></td><td>ments is missing, merging them into one larger so they merge into one larger compartment.</td><td>2 The divider wall between two adjacent compart- Remove the divider wall between two adjacent compartments</td></tr><tr><td></td><td>compartment.</td><td>3 The box has 15 compartments but they are not Misalign the compartments so they are not in straight rows and</td></tr><tr><td></td><td>aligned into straight rows and columns. 4 One compartment contains no pushpin.</td><td>columns. Remove the pushpin from one compartment leaving it empty.</td></tr><tr><td></td><td>5 Two compartments contain no pushpin.</td><td>Remove the pushpins from two compartments leaving them empty.</td></tr><tr><td></td><td>6 One compartment contains two pushpins.</td><td>Add an extra pushpin to one compartment so it contains two pushpins.</td></tr><tr><td></td><td>7 Two compartments contain two pushpins</td><td>Add an extra pushpin to two compartments so each contains two pushpins.</td></tr><tr><td></td><td>8 One pushpin is located outside all compartments Move one pushpin outside the box. but still inside the box. (not used)</td><td></td></tr><tr><td></td><td>stead of inside a compartment. (not used)</td><td>9 One pushpin is resting on top of a divider wall in- Place one pushpin on top of a divider wall instead of inside a compartment.</td></tr><tr><td></td><td></td><td>10 One compartment contains a paperclip instead of a Replace the pushpin in one compartment with a paperclip.</td></tr><tr><td></td><td>pushpin.</td><td>11 One compartment contains a pushpin together with Add a paperclip next to the pushpin in one compartment.</td></tr><tr><td></td><td>a paperclip. 12 One pushpin is red instead of yellow.</td><td>Change the color of one pushpin from yellow to red.</td></tr><tr><td></td><td>13 One pushpin is blue instead of yellow.</td><td>Change the color of one pushpin from yellow to blue.</td></tr><tr><td></td><td>14 One pushpin is green instead of yellow</td><td>Change the color of one pushpin from yellow to green.</td></tr><tr><td></td><td></td><td>15 One compartment contains three pushpins instead Add two extra pushpins to one compartment so it contains three</td></tr><tr><td></td><td>of one.</td><td>pushpins. 16 One pushpin is noticeably larger than the others. Make one pushpin noticeably larger than the others.</td></tr><tr><td></td><td></td><td>17 One pushpin is noticeably smaller than the others. Make one pushpin noticeably smaller than the others.</td></tr><tr><td></td><td>(not used)</td><td>18 One pushpin is missing its metal pin and only the Remove the metal pin from one pushpin leaving only the head.</td></tr><tr><td></td><td>head is present. 19 One pushpin is visibly bent into a curved shape. Bend one pushpin into a curved shape.</td><td></td></tr><tr><td></td><td>spanning across compartments.</td><td>20 Two pushpins are physically connected by a thread Connect two pushpins in adjacent compartments with a thread.</td></tr></table>

![](images/eca1ea2c7326ef158f3b13f7d9572c1ca1e1f8a68f184e2d6b95317c915db04a.jpg)

![](images/cad18025e8622aabe3a18ad514950d16a5e4db9b36341a696089a5febbb0164c.jpg)

![](images/a630275608f0605f8cbba8221b270f6bc74fd2374887b68ba1439b9dcdbaccd7.jpg)

![](images/306e379efd347ad3bdfebc1ee0d6048497f6496fac108e37c93cfd805c15a7e6.jpg)

![](images/d5c0fe6a8c9d36875bb523c5fa1091253d89278c7b1e1013c760263e27fdb6a6.jpg)  
#9: not used

#8: not used  
![](images/ec86ba68cba7283f113ff54714e0697658da2ef347e755a3e6970b54026d4692.jpg)

![](images/722e6f1ba216a89a9260afa35f96d23e02bca1ad3f409415c5bf91d7c12df4c3.jpg)

![](images/8f6c215dc86cc8e52885db0d3eebc7c02eb841a94cf7ef9fa0986c3cdaee7b09.jpg)  
#15

![](images/9ea3a204ffdf9784c2afb81fd7848d2dbae233ebfab128d75055972d2bf0a12b.jpg)

#19  
![](images/d9b655b5ea3da61407949e2b7f0380fe189b0f8a6584cc82143311dc6212d7ae.jpg)  
#20  
Figure C.3: Pushpins: example pairs. Normal training images (left) and synthetic anomalies made from them (right). Numbers are the type IDs of Table C.3. The synthetic anomalies are generated by the image editor (Gemini 3 Pro Image), a generative AI model (Section 3.2).

Table C.4: Screw Bag: anomaly types and editing prompts.
<table><tr><td>ID</td><td>Anomaly type</td><td>Editing prompt</td></tr><tr><td></td><td>dard length.</td><td>1 The long screw is significantly longer than its stan- Make the long screw significantly longer than its standard length.</td></tr><tr><td></td><td>dard length.</td><td>2 The long screw is significantly shorter than its stan- Make the long screw significantly shorter than its standard length.</td></tr><tr><td></td><td>standard length.</td><td>3 The short screw is significantly shorter than its Make the short screw significantly shorter than its standard length.</td></tr><tr><td>4</td><td>components remain inside it.</td><td>The plastic bag is torn open while all six required Tear the plastic bag open while keeping all six components in- side the bag.</td></tr><tr><td></td><td>standard lengths.</td><td>5 Both screws are significantly shorter than their Make both screws significantly shorter than their standard lengths.</td></tr><tr><td></td><td>6 Exactly one nut is inside the bag.</td><td>Remove one nut so that exactly one nut remains inside the bag.</td></tr><tr><td></td><td>7 Exactly three nuts are inside the bag.</td><td>Add one additional nut so that there are exactly three nuts inside the bag.</td></tr><tr><td></td><td>8 No nuts are inside the bag.</td><td>Remove both nuts so that no nuts are inside the bag.</td></tr><tr><td></td><td>9 Exactly four nuts are inside the bag.</td><td>Add two additional nuts so that there are exactly four nuts in- side the bag.</td></tr><tr><td></td><td>10 Exactly one washer is inside the bag.</td><td>Remove one washer so that exactly one washer remains inside the bag.</td></tr><tr><td></td><td>11 Exactly three washers are inside the bag. (not used)</td><td>Add one additional washer so that there are exactly three wash- ers inside the bag.</td></tr><tr><td></td><td>12 No washers are inside the bag.</td><td>Remove both washers so that no washers are inside the bag. Add two additional washers so that there are exactly four wash-</td></tr><tr><td></td><td>13 Exactly four washers are inside the bag.</td><td>ers inside the bag.</td></tr><tr><td></td><td>14 Only the long screw is inside the bag.</td><td>Remove the short screw, both nuts, and both washers so only the long screw remains inside the bag.</td></tr><tr><td></td><td>15 Only the short screw is inside the bag.</td><td>Remove the long screw, both nuts, and both washers so only the short screw remains inside the bag.</td></tr><tr><td></td><td>16 No screws are inside the bag.</td><td>Remove both screws so that no screws are inside the bag. 17 Three screws are inside the bag, including two long Add one additional long screw so there are three screws inside</td></tr><tr><td></td><td>screws and one short screw.</td><td>the bag: two long and one short.</td></tr><tr><td></td><td>screw and two short screws.</td><td>18 Three screws are inside the bag, including one long Add one additional short screw so there are three screws inside the bag: one long and two short.</td></tr><tr><td></td><td>size. (not used)</td><td>19 One nut is significantly smaller than its standard Make one nut significantly smaller than its standard size.</td></tr><tr><td></td><td>dard size. (not used)</td><td>20 One washer is significantly smaller than its stan- Make one washer significantly smaller than its standard size.</td></tr></table>

![](images/9a0d50b00ab656261ade0840aa0add9c7b2a57e48ad281d317f456c0b37522b9.jpg)

![](images/b17d6f89668002e94490c230f2e78083260f855db941cfecc914363b4039b75f.jpg)

![](images/7287b141d4522d4151fad144a6a51164858b4bfc7422d598a928c7b9d3b08587.jpg)  
Figure C.4: Screw Bag: example pairs. Normal training images (left) and synthetic anomalies made from them (right). Numbers are the type IDs of Table C.4. The synthetic anomalies are generated by the image editor (Gemini 3 Pro Image), a generative AI model (Section 3.2).

Table C.5: Splicing Connectors: anomaly types and editing prompts.
<table><tr><td>ID</td><td>Anomaly type</td><td>Editing prompt</td></tr><tr><td></td><td>1 Only one splicing connector is present.</td><td>Remove the right splicing connector so only one connector re- mains.</td></tr><tr><td></td><td>2 Two cables are present between the connectors.</td><td>Add a second cable connecting the two connectors.</td></tr><tr><td></td><td>3 No cable is present between the connectors.</td><td>Remove the cable so no cable is present between the connec- tors.</td></tr><tr><td></td><td>ples.</td><td>4 The cable is a color that is not used in normal sam- Change the cable to a color that is not used in normal samples.</td></tr><tr><td></td><td>number of ports/levers).</td><td>5 The two connectors are different types (different Replace one connector with a connector that has a different number of ports/levers.</td></tr><tr><td></td><td></td><td>6 The cable color does not match the connector type. Change the cable to a color that does not match the connector type.</td></tr><tr><td></td><td>7 The cable is inserted into the left connector but not inserted into the right connector.</td><td>Remove the cable from the right connector so it is only inserted into the left connector.</td></tr><tr><td></td><td>used)</td><td>8 The cable is not inserted into either connector. (not Remove the cable from both connectors so it is not inserted into either connector.</td></tr><tr><td></td><td>the two connectors.</td><td>9 The cable is inserted into different lever indices on Move one end of the cable to a different lever index so the two ends are inserted into different lever indices.</td></tr><tr><td></td><td>10 At least one lever on a connector is open.</td><td>Open at least one lever on a connector.</td></tr><tr><td></td><td></td><td>11 A lever is missing or visibly broken on a connector. Remove or visibly break a lever on a connector.</td></tr><tr><td></td><td>12 One connector is rotated so it is not upright.</td><td>Rotate one connector so it is not upright.</td></tr><tr><td></td><td>13 One connector is rotated so its ports face away from the other connector. (not used)</td><td>Rotate one connector so its ports face away from the other con- nector.</td></tr><tr><td></td><td>other in a straight configuration.</td><td>14 The two connectors are not aligned to face each Reposition the connectors so they do not face each other in a straight configuration.</td></tr><tr><td></td><td>both ends, leaving the other connector uncon- the other connector unconnected.</td><td>15 The cable is inserted into the same connector at Insert both ends of the cable into the same connector and leave</td></tr><tr><td></td><td>nected. 16 The two connectors are touching or overlapping.</td><td>Move the connectors so they touch or overlap.</td></tr><tr><td></td><td>dard expected range.</td><td>17 The cable is excessively short relative to the stan- Shorten the cable so it is excessively short relative to the ex- pected range.</td></tr><tr><td></td><td>dard expected range.</td><td>18 The cable is excessively long relative to the stan- Lengthen the cable so it is excessively long relative to the ex- pected range.</td></tr><tr><td></td><td>length.</td><td>19 The cable is visibly cut, creating a break along its Add a visible cut in the cable so it is broken along its length.</td></tr><tr><td></td><td>not a valid port/lever entry (i.e., attached to the valid port/lever entry.</td><td>20 The cable connects to a connector position that is Attach one end of the cable to the connector body instead of a</td></tr></table>

![](images/6b0681274b1a5f0a71dcedf31d60dfe14d111c3382775173ba48a25c95aec0c9.jpg)  
Figure C.5: Splicing Connectors: example pairs. Normal training images (left) and synthetic anomalies made from them (right). Numbers are the type IDs of Table C.5. The synthetic anomalies are generated by the image editor (Gemini 3 Pro Image), a generative AI model (Section 3.2).

## Appendix D. Baseline setups

This appendix lists the settings of each baseline, so that the robustness comparison can be repeated. All baselines use the oficial code and released weights without training any network (Section 4).

CSAD and SALAD. Oficial code of CSAD (https://github.com/Tokichan/ CSAD; Hsieh and Lai, 2024) and SALAD (https://github.com/MaticFuc/SALAD; Fucka et al.ˇ , 2025) with the weights released by the authors, in their default settings.

LogSAD. Oficial code of LogSAD (https://github.com/zhang0jhon/LogSAD; Zhang et al., 2025a) in its full-data setting, with its frozen foundation models at 448 × 448. We score each image by the larger of LogSAD’s two scores before its final sigmoid, keeping rule-flagged images on top, because the float64 sigmoid saturates at exactly 1.0 for large values and creates ties; where it does not saturate, the ranking and the AUROC are unchanged.

PatchCore. Oficial code of PatchCore (https://github.com/amazon-science/ patchcore-inspection; Roth et al., 2022) with an ImageNet WideResNet-50-2, layer2 and layer3 features aggregated over 3 × 3 neighborhoods and pooled to 1024 dimensions, a 10% greedy coreset of the clean training patches, and the largest nearestneighbor patch distance as the image score. We resize the whole image to $2 5 6 \times 2 5 6$ instead of using the default center crop, which would cut of the ends of the non-square LOCO images.

AnomalyCLIP. Oficial code of AnomalyCLIP (https://github.com/zqhang/ AnomalyCLIP; Zhou et al., 2024), run zero-shot with OpenAI CLIP ViT-L/14@336 and the prompt checkpoint the authors trained on VisA, at 518 × 518. The image score is the probability of the anomaly prompt; no LOCO image is used as a normal reference.

Methods with only a reported clean result. $\mathrm { S A L A D ^ { \dagger } }$ (Fucka et al.ˇ , 2025), the setting of SALAD trained with the supervised part maps of PSAD, would need training; PSAD (Kim et al., 2024) has no released MVTec LOCO weights and needs training with hand-labeled part masks; SAM-LAD (Peng et al., 2025) has no oficial implementation. We therefore report their published clean AUROC and could not measure their robustness.

## Appendix E. Discovering sub-modes of normal images

Table E.1: Normal sub-mode discovery per category. Only normal training images are used. The candidate K is chosen by 5-fold cross-validated log-likelihood with the one-standard-error rule; a candidate must pass three gates (silhouette ≥ 0 25, smallest cluster ≥ 10%, median bootstrap $\mathbf { A R I } \geq 0 . 8 ) .$ , otherwise K − 1 is tried.
<table><tr><td>Category</td><td>Candidate K</td><td>Gates: silhouette / smallest cluster / ARI</td><td>Chosen K</td></tr><tr><td>Juice Bottle</td><td>4</td><td>K=4: 0.53 / 11% / 0.94 pass</td><td>4</td></tr><tr><td>Splicing Connectors</td><td>4</td><td>K=4: 0.20 / 14% / 0.94 fail (silhouette); K=3: 0.28 / 33% / 1.00 pass</td><td>3</td></tr><tr><td>Screw Bag</td><td>2</td><td> $K { = } 2 \colon 0 . 3 1 \textup { / } 1 1 \% / 0 . 6 6 \textup { f a i l } ( \mathrm { A R I } )$ </td><td>1</td></tr><tr><td>Breakfast Box</td><td>1</td><td>single candidate, no gate</td><td>1</td></tr><tr><td>Pushpins</td><td>1</td><td>single candidate, no gate</td><td>1</td></tr></table>

This appendix gives the exact rule of normal sub-mode discovery (Section 3.3). It runs once per category before training and uses only the normal training images. The features are the pooled output of the frozen, pretrained ConvNeXt-L DINOv3 backbone at 448 × 448, normalized to unit length and reduced to eight dimensions by principal component analysis (PCA). Gaussian mixtures with a tied covariance are fitted for $K = 1 , \ldots , 5$ , and the candidate is the smallest K within one standard error of the best 5-fold cross-validated log-likelihood (Hastie et al., 2009). A candidate $K > 1$ must pass the three gates of Table E.1, where the silhouette (Rousseeuw, 1987) is computed on the fitted clusters and the adjusted Rand index (ARI; Hubert and Arabie, 1985) against 20 refits on bootstrap samples; otherwise $K - 1 , K - 2 , . .$ are tried, down to $K = 1$

Finally, validation normal images are assigned to sub-modes by the mixture fitted on the training images.

Under this rule, in Juice Bottle, the banana juice images form two of the four clusters (75 and 37 images); in Splicing Connectors, each of the three clusters holds 120 images. Over 45 combinations of the gate thresholds (silhouette 0.15–0.35, smallest cluster 5–15%, ARI 0.7–0.9), the same two categories are split; only the number of sub-modes varies.

With a single normal class instead of these sub-modes, the clean AUROC falls from 97.0% to 92.2% in Juice Bottle and from 90.3% to 84.9% in Splicing Connectors; the other categories are unchanged by construction.

## Appendix F. Results per condition

Table F.1: Results per condition. Image-level logical AUROC (%), each condition scored alone, mean over the five categories. B20/B40/B60: brightness reduced by 20/40/60%; Z5/Z10/Z15: zoom-out by 5/10/15% with edge replication.
<table><tr><td>Method</td><td>Clean</td><td>B20</td><td>B40</td><td>B60</td><td>Z5</td><td>Z10</td><td>Z15</td></tr><tr><td>Ours (no overlap)</td><td>93.2</td><td>93.2</td><td>93.4</td><td>93.8</td><td>92.8</td><td>92.4</td><td>91.0</td></tr><tr><td>Ours (full recipe)</td><td>94.9</td><td>94.8</td><td>94.7</td><td>94.4</td><td>94.1</td><td>93.6</td><td>93.0</td></tr><tr><td>CSAD</td><td>97.7</td><td>92.2</td><td>82.0</td><td>74.4</td><td>74.7</td><td>64.8</td><td>61.9</td></tr><tr><td>SALAD</td><td>96.4</td><td>89.4</td><td>78.0</td><td>80.8</td><td>94.7</td><td>87.3</td><td>83.4</td></tr><tr><td>LogSAD</td><td>89.2</td><td>87.5</td><td>84.0</td><td>80.0</td><td>75.7</td><td>67.6</td><td>70.5</td></tr><tr><td>PatchCore</td><td>75.4</td><td>75.4</td><td>74.3</td><td>72.7</td><td>67.2</td><td>62.8</td><td>57.3</td></tr><tr><td>AnomalyCLIP</td><td>57.3</td><td>55.3</td><td>55.1</td><td>55.4</td><td>54.3</td><td>55.5</td><td>53.5</td></tr></table>

Table F.1 gives the AUROC of every method in each test condition scored on its own, as the five-category mean; the per-condition AUROC and the worst condition (Section 3.6) are in Table 3. The single-condition AUROC need not decrease monotonically with severity, because normal and anomalous scores shift by diferent amounts at each level.

Table G.1: Results per category. Image-level logical AUROC (%): clean (Cl.) and heterogeneous (Het.). Ours (no overlap): trained without test-overlapping augmentations.
<table><tr><td></td><td colspan="2">Breakfast Box</td><td colspan="2">Juice Bottle</td><td colspan="2">Pushpins</td><td colspan="2">Screw Bag</td><td colspan="2">Splicing Connectors</td></tr><tr><td>Method</td><td>Cl.</td><td>Het.</td><td>Cl.</td><td>Het.</td><td>Cl.</td><td>Het.</td><td>Cl.</td><td>Het.</td><td>Cl.</td><td>Het.</td></tr><tr><td>Ours (no overlap)</td><td>92.8</td><td>93.3</td><td>92.7</td><td>92.3</td><td>99.5</td><td>99.0</td><td>91.3</td><td>90.1</td><td>89.5</td><td>87.8</td></tr><tr><td>Ours (full recipe)</td><td>97.1</td><td>96.4</td><td>97.0</td><td>96.4</td><td>100.0</td><td>99.8</td><td>90.1</td><td>87.7</td><td>90.3</td><td>90.2</td></tr><tr><td>CSAD</td><td>95.8</td><td>60.0</td><td>95.5</td><td>71.2</td><td>99.7</td><td>66.2</td><td>99.9</td><td>87.3</td><td>97.4</td><td>56.1</td></tr><tr><td>SALAD</td><td>92.3</td><td>61.0</td><td>99.8</td><td>66.0</td><td>100.0</td><td>68.0</td><td>94.0</td><td>60.4</td><td>96.1</td><td>55.8</td></tr><tr><td>LogSAD</td><td>97.1</td><td>70.4</td><td>94.6</td><td>81.7</td><td>84.0</td><td>82.7</td><td>79.9</td><td>62.5</td><td>90.2</td><td>72.4</td></tr><tr><td>PatchCore</td><td>81.4</td><td>70.9</td><td>92.8</td><td>86.5</td><td>70.4</td><td>57.5</td><td>56.4</td><td>55.6</td><td>76.1</td><td>58.1</td></tr><tr><td>AnomalyCLIP</td><td>68.5</td><td>61.9</td><td>61.7</td><td>60.2</td><td>54.6</td><td>50.6</td><td>47.8</td><td>48.7</td><td>53.7</td><td>50.1</td></tr></table>

Table G.1 gives the clean and heterogeneous AUROC of every method per category. Our framework shows its largest drop in Splicing Connectors without test-overlapping augmentations and in Screw Bag with the full recipe; CSAD and SALAD show their largest drops in Splicing Connectors.

## References

Baitieva, A., Bouaouni, Y., Briot, A., Ameln, D., Khalfaoui, S., Akcay, S., 2025. Beyond academic benchmarks: Critical analysis and best practices for visual industrial anomaly detection, in: CVPRW, pp. 4015–4025. doi:10.1109/CVPRW67362. 2025.00386. VAND 3.0 Workshop.

Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C., 2022. Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization. Int. J. Comput. Vis. 130, 947–969. doi:10.1007/s11263-022-01578-9.

Bergmann, P., Fauser, M., Sattlegger, D., Steger, C., 2019. MVTec AD – a comprehensive real-world dataset for unsupervised anomaly detection, in: CVPR, pp. 9584–9592. doi:10.1109/CVPR.2019.00982.

Brooks, T., Holynski, A., Efros, A.A., 2023. InstructPix2Pix: Learning to follow image editing instructions, in: CVPR, pp. 18392–18402. doi:10.1109/CVPR52729. 2023.01764.

Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M., Kalinin, A.A., 2020. Albumentations: Fast and flexible image augmentations. Information 11, 125. doi:10.3390/info11020125.

Cao, C., Zhong, Z., Zhou, Z., Liu, Y., Liu, T., Han, B., 2024. Envisioning outlier exposure by large language models for out-of-distribution detection, in: ICML, pp. 5629–5659. URL: https://proceedings.mlr.press/v235/cao24d.html.

Cao, T., Zhu, J., Pang, G., 2023. Anomaly detection under distribution shift, in: ICCV, pp. 6488–6500. doi:10.1109/ICCV51070.2023.00599.

Cheng, Y., Cao, Y., Chen, R., Shen, W., 2024. RAD: A comprehensive dataset for benchmarking the robustness of image anomaly detection, in: CASE, pp. 2123– 2128. doi:10.1109/CASE59546.2024.10711763.

Cohen, N., Abutbul, R., Hoshen, Y., 2022. Out-of-distribution detection without class labels, in: ECCVW, pp. 101–117. doi:10.1007/978-3-031-25063-7\_7.

Du, X., Sun, Y., Zhu, X., Li, Y., 2023. Dream the impossible: Outlier imagination with difusion models, in: NeurIPS, pp. 60878–60901. doi:10.52202/075280-2660.

Dunlap, L., Umino, A., Zhang, H., Yang, J., Gonzalez, J.E., Darrell, T., 2023. Diversify your vision datasets with automatic difusion-based augmentation, in: NeurIPS, pp. 79024–79034. doi:10.52202/075280-3456.

Fucka, M., Zavrtanik, V., Sko ˇ caj, D., 2025. SALAD – semantics-aware logicalˇ anomaly detection, in: ICCV, pp. 21843–21852. doi:10.1109/ICCV51701.2025. 02028.

Fucka, M., Zavrtanik, V., Skoˇ caj, D., 2026a. AnomalyVFM – transforming visionˇ foundation models into zero-shot anomaly detectors, in: CVPR, pp. 35555– 35566. URL: https://openaccess.thecvf.com/content/CVPR2026/html/ Fucka\_AnomalyVFM\_--\_Transforming\_Vision\_Foundation\_Models\_into\_ Zero-Shot\_Anomaly\_Detectors\_CVPR\_2026\_paper.html.

Fucka, M., Zavrtanik, V., Skoˇ caj, D., 2026b. ObjectCore – eˇ ficient few-shot logical anomaly detection using object representations, in: WACV, pp. 3857–3867. doi:10. 1109/WACV61042.2026.00376.

Google DeepMind, 2025. Gemini 3 Pro Image (Nano Banana Pro). https:// deepmind.google/models/gemini-image/pro/. Accessed 16 February 2026.

Gu, Z., Zhu, B., Zhu, G., Chen, Y., Tang, M., Wang, J., 2024. AnomalyGPT: Detecting industrial anomalies using large vision-language models, in: AAAI, pp. 1932–1940. doi:10.1609/aaai.v38i3.27963.

Hastie, T., Tibshirani, R., Friedman, J., 2009. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Series in Statistics. 2nd ed., Springer, New York. doi:10.1007/978-0-387-84858-7.

Heckler-Kram, L., Neudeck, J.H., Scheler, U., König, R., Steger, C., 2026. The MVTec AD 2 dataset: Advanced scenarios for unsupervised anomaly detection. Int. J. Comput. Vis. 134, 175. doi:10.1007/s11263-026-02743-0.

Hendrycks, D., Dietterich, T., 2019. Benchmarking neural network robustness to common corruptions and perturbations, in: ICLR. URL: https://openreview.net/ forum?id=HJz6tiCqYm.

Hsieh, Y.H., Lai, S.H., 2024. CSAD: Unsupervised component segmentation for logical anomaly detection, in: BMVC. URL: https://bmvc2024.org/ proceedings/854/.

Hu, J., Borsatti, F., Stropeni, A., Dalle Pezze, D., Barusco, M., Susto, G.A., 2026. MIRAGE: Model-agnostic industrial realistic anomaly generation and evaluation for visual anomaly detection, in: CVPRW, pp. 7722–7731. URL: https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Hu\_ MIRAGE\_Model-agnostic\_Industrial\_Realistic\_Anomaly\_Generation\_ and\_Evaluation\_for\_Visual\_CVPRW\_2026\_paper.html. VAND 4.0 Workshop.

Hubert, L., Arabie, P., 1985. Comparing partitions. J. Classif. 2, 193–218. doi:10. 1007/BF01908075.

Ionescu, R.T., Khan, F.S., Georgescu, M.I., Shao, L., 2019. Object-centric autoencoders and dummy anomalies for abnormal event detection in video, in: CVPR, pp. 7834–7843. doi:10.1109/CVPR.2019.00803.

Jia, Y., Hoyer, L., Huang, S., Wang, T., Van Gool, L., Schindler, K., Obukhov, A., 2024. DGInStyle: Domain-generalizable semantic segmentation with image difusion models and stylized semantic control, in: ECCV, pp. 91–109. doi:10.1007/ 978-3-031-72933-1\_6.

Jin, E., Feng, Q., Mou, Y., Lakemeyer, G., Decker, S., Simons, O., Stegmaier, J., 2025. LogicAD: Explainable anomaly detection via VLM-based text feature extraction, in: AAAI, pp. 4129–4137. doi:10.1609/aaai.v39i4.32433.

Kaushik, D., Hovy, E., Lipton, Z.C., 2020. Learning the diference that makes a diference with counterfactually-augmented data, in: ICLR. URL: https:// openreview.net/forum?id=Sklgs0NFvr.

Kim, S., An, S., Chikontwe, P., Kang, M., Adeli, E., Pohl, K.M., Park, S.H., 2024. Few shot part segmentation reveals compositional logic for industrial anomaly detection, in: AAAI, pp. 8591–8599. doi:10.1609/aaai.v38i8.28703.

Kwon, Y., Moon, D., Oh, Y., Yoon, H., 2025. LogicQA: Logical anomaly detection with vision language model generated questions, in: ACL (Industry Track), pp. 411– 432. doi:10.18653/v1/2025.acl-industry.29.

Lei, T., Chen, S., Wang, B., Jiang, Z., Zou, N., 2025. Adapted-MoE: Mixture of experts with test-time adaption for anomaly detection, in: ICIC, pp. 427–441. doi:10.1007/ 978-981-96-9891-2\_36.

Li, C.L., Sohn, K., Yoon, J., Pfister, T., 2021. CutPaste: Self-supervised learning for anomaly detection and localization, in: CVPR, pp. 9659–9669. doi:10.1109/ CVPR46437.2021.00954.

Li, Y., Yuan, S., Zhu, Z., Liu, M., Zuo, W., Shi, G., 2026. Beyond common sense: Grounding logical anomaly detection in inspection criteria, in: ECCV, pp. 475–493. doi:10.1007/978-3-032-37032-7\_26.

Liao, K., Yang, J., Tao, T., Wu, W., Jiang, J., Xiao, J., 2026. Multi-prototype compactness and boundary-aware synthesis for unsupervised anomaly detection, in: CVPR, pp. 28524–28533. URL: https://openaccess.thecvf. com/content/CVPR2026/html/Liao\_Multi-Prototype\_Compactness\_and\_ Boundary-Aware\_Synthesis\_for\_Unsupervised\_Anomaly\_Detection\_ CVPR\_2026\_paper.html.

Lin, Y., Chang, Y., Tong, X., Yu, J., Liotta, A., Huang, G., Song, W., Zeng, D., Wu, Z., Wang, Y., Zhang, W., 2025. A survey on RGB, 3D, and multimodal approaches for unsupervised industrial image anomaly detection. Inf. Fusion 121, 103139. doi:10. 1016/j.inffus.2025.103139.

Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022. A ConvNet for the 2020s, in: CVPR, pp. 11966–11976. doi:10.1109/CVPR52688.2022.01167.

Loshchilov, I., Hutter, F., 2019. Decoupled weight decay regularization, in: ICLR. URL: https://openreview.net/forum?id=Bkg6RiCqY7.

OpenAI, 2025. GPT-5 system card. https://cdn.openai.com/ gpt-5-system-card.pdf. Accessed 8 October 2026.

Pemula, L., Zhang, D., Dabeer, O., 2025. Robust AD: A real world benchmark dataset for robustness in industrial anomaly detection, in: CVPRW, pp. 4047–4057. doi:10. 1109/CVPRW67362.2025.00389. VAND 3.0 Workshop.

Peng, Y., Lin, X., Ma, N., Du, J., Liu, C., Liu, C., Chen, Q., 2025. SAM-LAD: Segment Anything Model meets zero-shot logic anomaly detection. Knowl.-Based Syst. 314, 113176. doi:10.1016/j.knosys.2025.113176.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: CVPR, pp. 10674–10685. doi:10. 1109/CVPR52688.2022.01042.

Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P., 2022. Towards total recall in industrial anomaly detection, in: CVPR, pp. 14298–14308. doi:10. 1109/CVPR52688.2022.01392.

Rousseeuw, P.J., 1987. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. J. Comput. Appl. Math. 20, 53–65. doi:10.1016/ 0377-0427(87)90125-7.

Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P., 2025. DINOv3. arXiv preprint arXiv:2508.10104. doi:10.48550/arXiv.2508.10104.

Sohoni, N.S., Dunnmon, J.A., Angus, G., Gu, A., Ré, C., 2020. No subclass left behind: Fine-grained robustness in coarse-grained classification problems, in: NeurIPS, pp. 19339–19352. URL: https://proceedings.neurips.cc/paper\_files/ paper/2020/hash/e0688d13958a19e087e123148555e4b4-Abstract.html.

Sun, H., Cao, Y., Dong, H., Fink, O., 2025. Unseen visual anomaly generation, in: CVPR, pp. 25508–25517. doi:10.1109/CVPR52734.2025.02375.

Tong, X., Chang, Y., Zhao, Q., Yu, J., Wang, B., Lin, J., Lin, Y., Mai, X., Wang, H., Tao, Z., Wang, Y., Zhang, W., 2025. Component-aware unsupervised logical anomaly generation for industrial anomaly detection, in: ICRA, pp. 16722–16729. doi:10.1109/ICRA55743.2025.11127296.

Wightman, R., 2019. PyTorch image models. https://github.com/huggingface/ pytorch-image-models. GitHub repository, accessed 3 October 2026.

Xu, X., Cao, Y., Zhang, H., Sang, N., Huang, X., 2025. Customizing visual-language foundation models for multi-modal anomaly detection and reasoning, in: CSCWD, pp. 1443–1448. doi:10.1109/CSCWD64889.2025.11033177.

Yang, J., Zhou, K., Liu, Z., 2023. Full-spectrum out-of-distribution detection. Int. J. Comput. Vis. 131, 2607–2622. doi:10.1007/s11263-023-01811-z.

Zavrtanik, V., Kristan, M., Skocaj, D., 2021. DRAEM – a discriminatively trainedˇ reconstruction embedding for surface anomaly detection, in: ICCV, pp. 8310–8319. doi:10.1109/ICCV48922.2021.00822.

Zhang, J., Wang, G., Jin, Y., Huang, D., 2025a. Towards training-free anomaly detection with vision and language foundation models, in: CVPR, pp. 15204–15213. doi:10.1109/CVPR52734.2025.01416.

Zhang, Y., Cao, Y., Xu, X., Shen, W., 2025b. LogiCode: An LLM-driven framework for logical anomaly detection. IEEE Trans. Autom. Sci. Eng. 22, 7712–7723. doi:10.1109/TASE.2024.3468464.

Zhang, Z., Zhao, Z., Zhang, X., Sun, C., Chen, X., 2023. Industrial anomaly detection with domain shift: A real-world dataset and masked multi-scale reconstruction. Comput. Ind. 151, 103990. doi:10.1016/j.compind.2023.103990.

Zhao, Y., 2024. LogicAL: Towards logical anomaly synthesis for unsupervised anomaly localization, in: CVPRW, pp. 4022–4031. doi:10.1109/CVPRW63382. 2024.00406. VAND 2.0 Workshop.

Zhou, Q., Pang, G., Tian, Y., He, S., Chen, J., 2024. AnomalyCLIP: Objectagnostic prompt learning for zero-shot anomaly detection, in: ICLR. URL: https: //openreview.net/forum?id=buC4E91xZE.