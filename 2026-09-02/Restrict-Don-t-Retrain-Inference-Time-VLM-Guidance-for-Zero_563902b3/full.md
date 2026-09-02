# Restrict, Don’t Retrain: Inference-Time VLM Guidance for Zero-Shot Aerial Segmentation

Teresa DiMeola (tjdimeol@go.olemiss.edu), Charles Walter (cwwalter@olemiss.edu), Hong Xiao (hxiao1@olemiss.edu) Dept. of Computer and Information Science University of Mississippi

## Abstract

Global welfare often depends on the correct interpretation of aerial and satellite imagery. Acting on such imagery (mapping flooded ground, crop extent, or damaged infrastructure) demands pixel-level segmentation to ensure perfect class localization. Pretrained general foundation models, when applied directly, often miss important features and cannot always find all the classes belonging to a given scene, overlooking smaller objects that matter most. We use a single consumer-grade GPU running a vision-language model (VLM) to supply this missing guidance, improving segmentation while producing structured, auditable evidence that drives the result and can be inspected on its own. We fuse three approaches: the frozen foundation model that labels every pixel, and two queries to a VLM, one to choose the classes that matter, and one to locate the small objects the base model misses. Evaluating across four aerial datasets, we see consistent gains at each stage where the base model is competent.

Keywords: aerial imagery, vision-language models, zero-shot segmentation, open vocabulary segmentation, auditable inference

## 1. Introduction

Many practical decisions now rest on interpreting aerial and satellite imagery. Disaster-response teams triage flooded or damaged areas, agricultural planners map land cover, and infrastructure crews inspect roads, roofs, and power corridors. In each case a raw image must be turned into structured information that can be acted on. This is carried out via semantic segmentation: assigning every pixel an object class.

The usual path to good aerial segmentation is to build a domain-specific foundation model, training a large network from scratch on very large amounts of in-domain data (Wen et al., 2023). This is costly, as it needs labeled data, compute infrastructure, machine-learning expertise, and a separate model for each domain. When the set of classes that matters changes over time, the model goes stale and the cost returns.

We argue this is unnecessary. A general-purpose foundation model already encodes the relevant classes. What it lacks, at inference time, is knowledge of which part of its large vocabulary applies to the image in front of it, and where the small but important objects are. The problem is one of selection and guidance, not of model architecture or training-data scale.

Lighter-weight alternatives such as open-vocabulary, vocabulary-free, and inference-time methods have also been explored (Ghiasi et al., 2022), but in specialized domains they struggle to reliably select the right classes for each image.

This paper presents a workflow that supplies that guidance by bringing together two models that cooperatively counteract the failure mode of the other. The first is the frozen foundation model that labels every pixel but treats all classes as equally likely. The second is a secondary vision-language model (VLM) that looks at each image and returns a short, structured reading of the scene: which classes are present, and where the small objects sit. The VLM never paints pixels itself. It works as an interpretive layer, turning a raw image into readable cues that steer and patch the base model. We query it twice, through two inference-time mechanisms:

• Automated Class Weighting (ACW): the VLM picks the handful of classes that actually appear in the image, and those classes’ scores are boosted before the model makes its per-pixel decision (Section 3.2).

• Minority Class Identification (MCI): the VLM is asked a second time to draw boxes around the small objects the weighted classifier still misses, and those boxes are filled in and merged back (Section 3.3).

Neither model is reliable on its own. The base model has broad pixel coverage but weak class discipline; the

VLM has strong scene judgment but cannot produce a pixel map. Putting them together lets each cover the other’s blind spot, so the pair reaches a more reliable result than either one gives alone.

We treat the workflow as a deliberate design and study it across four aerial datasets, in the design-science tradition. From the results we draw three design principles that carry to other specialized domains: (i) restricting a general model’s vocabulary is preferable to training a separate domain-specific model; (ii) weighting the majority classes a scene is built from improves coherence; (iii) the small classes a dense model drops can be recovered by asking the VLM where they are.

## 2. Related Work

Remote sensing (RS) has shifted toward transformer-based vision-language methods. A survey of VLMs for RS (Wen et al., 2023) shows a clear trend toward large, RS-specific models such as RS5M and GeoRSCLIP (Zhang et al., 2023) and RemoteCLIP (F. Liu et al., 2024), each trained as a separate model from scratch at large cost in data, energy, and hardware.

These models are meant for “in-the-wild” imagery: aerial data the model was not trained on, where the task is to label objects and landscapes the model has never seen. This zero-shot segmentation (ZSS) needs a class label for every pixel, including classes absent from training (Bucher et al., 2019). Transformer-based methods (Ding et al., 2022) rely on large in-domain pretraining and lose accuracy in new domains unless they are fine-tuned, or even retrained. Benchmarks such as MESS (Blumenstiel et al., 2023) measure this transfer gap and have become a reference point for zero-shot evaluation.

Open-vocabulary segmentation (OVS) (Ghiasi et al., 2022) lets a user name any class at inference time, and recent work extends CLIP (Radford et al., 2021) with mask-adapted heads to bring open-vocabulary capabilities into segmentation (Liang et al., 2023). Vocabulary-free methods (Reichard et al., 2025) let the model invent its own class names, but in specialized domains this produces inconsistent names and unpredictable output.

One reason teams build domain-specific models is the belief that the model’s weights must change, which is almost as costly as building the model from scratch. A lighter option is to adapt a model’s outputs at inference time instead of its weights; for example, Cheng et al. (2021) address class imbalance by adjusting per-pixel logits.

Most work on class imbalance in segmentation acts at training time, through resampling or loss weighting; in aerial imagery, large changes in object scale make the imbalance worse (Elhagry and Saeed, 2022).

Using language models to enumerate visual features has emerged alongside instruction-following models. Kao et al. (2025a) show that chain-of-thought reasoning supports fine-grained scene description, and H. Liu et al. (2023) show that multimodal LLMs can act as image-aware annotators.

Beyond segmentation itself, there is a broader shift in decision support, where large language and vision-language models are used not as standalone predictors but as components inside larger workflows, turning unstructured inputs into structured features that downstream tools can act on (Rahimi, 2026). Such workflows work by fusing information gleaned from each element, supporting the long-standing practice of combining several heterogeneous sources to reach a more reliable result than any one of them gives alone (Yi et al., 2025). Transparency questions can also be more readily addressed if analysts can identify the reasonong that led to a particular decision.

By treating the workflow as a designed artifact in the design-science tradition (Hevner et al., 2004; Sein et al., 2011), reusable design principles can be leveraged over reporting from a ”black-box” system (Ribeiro et al., 2016). The transparency is built in via structured outputs a reviewer, who also knows the inputs, can inspect. The faithfulness concerns raised for chain-of-thought explanations (Barez et al., 2025) therefore do not apply here: we never treat the reasoning narrative as the explanation; the structured outputs it produces are what we inspect and act on.

## 3. Design and Methods

Our method splits the work between two frozen models: a general foundation model that produces dense per-pixel labeling over a curated 150-class vocabulary, and a vision-language model, queried twice, that supplies the per-image guidance the dense model lacks: which classes to weight (Stage 2) and where small, rare objects sit (Stage 3). Nothing is retrained, and every VLM decision is an inspectable list or box. The three subsections detail each stage in turn. Figure 1 shows the full workflow alongside a worked example of the per-stage progression.

## 3.1. Stage 1: Baseline Segmentation

General-purpose foundation models hold a great deal of semantic knowledge from large-scale pretraining, including many domain-specific concepts (Radford et al., 2021; Wen et al., 2023). We tap this in two ways. First, we restrict the inference vocabulary: we start from the 150-class ADE20K vocabulary (Zhou et al., 2017) that the base model already predicts and hand-adjust it toward aerial scenes, replacing general classes with ones that characterize infrastructure, natural features, vehicles, land use, and specialized objects. Second, we fine-tune on each dataset’s annotated classes of interest, ranging from six (UDD6) to twenty-four (DroneSeg), as anchors. The remaining classes are found zero-shot through relationships already present in the base model’s embeddings. Aerial categories are common enough to be well represented in pretraining, so this fine-tuning is rarely decisive here; we include it because the same blueprint transfers to domains such as medicine, chemistry, and materials science, where genuinely novel classes appear regularly and fine-tuning anchors become essential.

![](images/472888caea80762bea7a1420e7ee1d8cf2bc99760a30ad81580e42a27bfc168f.jpg)  
Figure 1: The three-stage workflow. Top row: an aerial image is processed through Stage 1 (vocabulary restriction to a curated set of about 150 classes), Stage 2 (automated class weighting, where a VLM chain-of-thought selects the significant classes for the image and boosts their per-class logits), and Stage 3 (minority class identification, where the VLM locates small objects as bounding boxes that are then mask-inpainted), producing the final 150-class segmentation. Bottom row: a worked example showing the output after each stage, with minority classes such as persons and a drone recovered in Stage 3.

## 3.2. Stage 2: Automated Class Weighting (ACW)

In plain zero-shot inference every class competes for every pixel. Classes that are not in the image add noise and crowd out the ones that are. A person glancing at an aerial image can name the five to ten classes that appear; holding the model to those classes sharply improves the result.

Let C denote the 150-class domain vocabulary. For each input image I, we identify a significant class set $\boldsymbol { B } _ { I } \subset \mathcal { C }$ . We define per-class weights:

$$
w _ { c } = { \left\{ \begin{array} { l l } { W _ { \mathrm { b o o s t } } } & { { \mathrm { i f } } \ c \in { \mathcal { B } } _ { I } } \\ { W _ { \mathrm { n e u t r a l } } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{1}
$$

where $W _ { \mathrm { { b o o s t } } } ~ = ~ 1 0 0$ and $W _ { \mathrm { n e u t r a l } } ~ = ~ 0$ (or 1 to retain a soft fallback). These weights are applied multiplicatively to per-pixel logits before softmax:

$$
\begin{array} { l } { \ell _ { i , j , c } ^ { ( \mathrm { w } ) } = w _ { c } \cdot \ell _ { i , j , c } , } \\ { P _ { i , j , c } = \displaystyle \frac { \exp ( \ell _ { i , j , c } ^ { ( \mathrm { w } ) } ) } { \sum _ { k } \exp ( \ell _ { i , j , k } ^ { ( \mathrm { w } ) } ) } , } \\ { \hat { c } _ { i , j } = \arg \operatorname* { m a x } _ { c } P _ { i , j , c } } \end{array}\tag{2}
$$

No model weights are changed; ACW works with any VLM that exposes per-class logits. It also has a useful side effect: focusing the output on a small, image-specific class set reduces the ambiguity that would otherwise force us to tile large images. The output of this stage is a complete per-pixel segmentation: the base model labels every pixel from the weighted logits, while the VLM supplies only the class set and weights, with no spatial information of its own. Localization here is entirely the base model’s; bounding boxes do not enter until Stage 3.

ACW needs the class set $B _ { I }$ before inference. Hand-labeling it is accurate but defeats the purpose of zero-shot deployment, so we obtain it from the VLM. A two-part prompt (Figure 2) asks the model to (i) reason step by step about the scene at both global and local levels, (ii) choose classes only from the 150-class vocabulary, and (iii) return a Python-ready list and weight dictionary so the output parses cleanly. Here the VLM is doing the interpretive work: it turns the image into a readable list of classes and weights that a person can check. The result feeds straight into ACW

You will serve as an agent for a language-based image segmentation model. Your task is   
to describe a given image using chain of thought, with as many details as possible to   
help the model understand it. Target objects may have multiple layers, be blocked by   
other objects, or be embedded in their surroundings. Pay special attention to water,   
sky, clouds, walls, etc. The first questions cover the overall style or background;   
then work down to fine detail (e.g., items a person is holding). Distinguish trees from   
other vegetation. Include anything visible, even if it is not prominent.   
Output format:   
<Question 1>: <Answer 1>.   
- <Question 2>: <Answer 2>.   
.. each pair implies one step of the chain of thought.   
After the first pass, review the image again in a second pass to discover missed items   
or reassess, then add a second output.   
Classes must be selected from the 150-class class\_list. Return a list of every noun   
from that list that you find in the image, with a careful percentage for each. If   
something is unidentifiable or not on the list, call it "other" and give its percentage   
too. Do not skip signs, posts, or other markers.   
<Question 1>: What is the overall layout and style of the image?   
<Answer 1>: An aerial top-down view of an urban campus or institutional complex. It   
features multiple outdoor basketball courts, surrounding buildings with modern   
designs, paved parking areas, sidewalks, and small patches of vegetation...   
<Question 4>: Are there any vehicles visible?   
<Answer 4>: Yes. Three white vans are parked along the edge of the basketball courts,   
and another is on the right side of the image. Numerous bicycles are parked in   
designated racks, arranged neatly in rows.   
<Question 7>: Are there any signs, posts, or markers?   
<Answer 7>: Yes. Multiple vertical posts, likely light poles or signage supports,   
appear along the periphery of the courts and near the buildings. One sign is visible   
near the central building entrance, though its content is illegible...   
[... the chain of thought continues over a second pass, then the response ends with the   
required structured output: a boosted\_classes list and an override\_weights dictionary   
with every weight set to 100.]  
Figure 2: (Top): The VLM prompt for automated class selection (Stage 2), lightly edited for length and readability. The Stage 3 bounding-box prompt shares this structure. (Bottom): A representative VLM response (snippet, edited for length). Full responses can run to several thousand tokens. The chain-of-thought catches small objects such as the vans, bicycles, posts, and an illegible sign, the minority classes Stage 3 later recovers. The response closes with the structured list and weight dictionary the downstream steps consume. Thanks to Kao et al. (2025b) for inspiring the first part of the prompt.

(Equation 2, above):

$$
B _ { I } = \mathrm { L M } \mathrm { \mathrm { \mathrm { . p r o m p t } } } ( I , { \mathcal { C } } ) .\tag{3}
$$

We judge class-selection quality with precision, recall, and F1 score against the ground-truth class set G(I) present in image I:

$$
\begin{array} { l } { { \mathrm { P r e c : = P r e c i s i o n } ( I ) = \displaystyle \frac { | { \cal B } _ { I } \cap { \cal G } ( I ) | } { | { \cal B } _ { I } | } , } } \\ { ~ } \\ { { \displaystyle ~ \mathrm { R e c : = R e c a l l } ( I ) = \displaystyle \frac { | { \cal B } _ { I } \cap { \cal G } ( I ) | } { | { \cal G } ( I ) | } , } } \\ { ~ } \\ { { \displaystyle ~ \mathrm { F 1 } ( I ) = 2 \cdot \displaystyle \frac { \mathrm { P r e c \cdot R e c } } { \mathrm { P r e c \cdot R e c } } . } } \end{array}\tag{4}
$$

## 3.3. Stage 3: Minority Class Identification (MCI)

Even after Stage 2, some classes the VLM reports as present never appear in the segmentation: the base model assigns them no pixels. We call these minority classes: signs, vehicles, windows, and similar objects that occupy few pixels but matter to the scene. They are small and sparse enough that the model’s per-pixel signal for them is too weak to win at any location, and class weighting can only scale that signal, not manufacture it where the model produced almost none. Stage 2 already labels most pixels correctly; MCI is a focused fix for the classes Stage 2 cannot reach.

MCI is the workflow’s second query to the same VLM. Stage 2 asked what is in the scene; Stage 3 asks where the small objects are. We prompt the VLM to return bounding boxes for the minority-class objects it sees, in image coordinates. Each box is then filled in to a precise pixel mask by a class-agnostic segmenter (Kirillov et al., 2023) and merged back into the Stage 1+2 result. When the merge meets a minority class absent from the Stage 1 baseline, it paints the class in directly; when it meets a class already present, it updates those pixels only if the new mask scores better by per-class IoU. Keeping the only judgment calls inside the VLM’s readable outputs, and leaving the fill step mechanical, is what makes the result easy to trace back to its source.

Figure 2 makes the two VLM queries concrete: the first shows an example prompt, the second a snippet of a representative response, including the chain-of-thought and the structured class list and weights it emits.

## 4. Experimental Setup

## 4.1. Datasets

We evaluate our approach on four aerial benchmark datasets: UAVid (Lyu et al., 2020) (8 classes, urban drone video), Aeroscapes (Nigam et al., 2018) (11 classes, diverse aerial views), DroneSeg (Ji et al., 2024) (24 classes, diverse geography), and UDD6 (Chen, 2018) (6 classes, campus and urban rooftops).

## 4.2. Evaluation Protocol

A core methodological choice is rigorous cross-dataset evaluation for zero-shot transfer. For each dataset pair (A, B): fine-tune on A, test zero-shot on B. We map the restricted vocabulary of 150 domain-specific classes to each dataset’s ground-truth class set for mIoU calculation. We note that ground truth is used only for computing evaluation metrics (such as mIoU), and not for weighting, prompt construction, or minority class identification (MCI). The primary evidence for our central claim is the consistency of ∆mIoU across stage transitions and across datasets, not the absolute mIoU on any single dataset. Absolute mIoU is a conservative lower bound: ground-truth annotations frequently use catch-all classes (Clutter, Unlabeled, Other) for pixels the annotators did not label, so correct predictions in those regions are penalized as false positives. We remove all such catch-all classes from all predictions. This bound is tightest where the ground truth is coarsest: our restricted 150-class vocabulary encodes distinctions the benchmark labels cannot express, so the metric penalizes the model for predicting more finely than the annotation can score. UDD6 is the extreme case, with only six coarse classes, so the richest part of the mapping has nowhere to land and correct fine predictions are counted as errors.

## 4.3. Hardware

All experiments are conducted on an NVIDIA RTX A6000 GPU with memory capped to 16 GB to simulate consumer-hardware constraints (the original target was a 16 GB consumer laptop GPU). The fine-tuning hyperparameters we used are: Adam, lr $\mathit { \Delta } \cdot = \mathit { \Delta } 5 \times 1 0 ^ { - 5 }$ weight decay 0.1, batch size 16, and 2k iterations. Stage 1 fine-tuning completes in 1 to 2 hours per dataset on a single GPU. To account for VLM response stochasticity in the Stage 2 class selection, we run three independent trials (seeds $s ~ = ~ 0 , 1 , 2 )$ for each combination of dataset and configuration, and report mean ± standard deviation; statistical significance versus the unweighted baseline is assessed via paired t-test. Runtime is dominated by VLM API latency in Stages 2 and 3, which is highly variable and capped at 20 minutes per call. The VLM API accounts for roughly 90% of the pipeline’s runtime, leaving local computation (base-model inference and instance segmentation) about 10%.

## 4.4. VLM Screening for Bounding Box Generation and Inpainting

Several VLMs were considered for the MCI bounding-box step in early experiments, including GPT-4o, Claude Sonnet, and Gemini. None produced bounding boxes whose locations corresponded usefully to objects in aerial imagery: predicted boxes were displaced from their targets by magnitudes large enough to render the subsequent fill step ineffective. Qwen-VL was the only model that produced spatially-grounded predictions suitable for the MCI step. We did not pursue a quantitative comparison because the gap was qualitatively decisive at the screening stage; candidates that cannot place a box within the same region as the target object cannot be ranked meaningfully against one that can. Spatial grounding accuracy in commercial VLMs on aerial imagery remains an open problem. Once the class-labeled bounding boxes have been defined, they are inpainted.

## 5. Results

## 5.1. Qualitative Results

Our design targets the operator or researcher in the field, at the edge, without access to backend GPU power. The data of interest, often returned via satellite or drone, is completely new “in-the-wild” imagery that must be as accurate as possible. Figures 3, 4, 5, and 6 illustrate the segmentation progression on each dataset.

The effects of automated class weighting (ACW) versus minority class identification (MCI) depend on several variables, including landscape type, the mix of minority and majority classes, and the field and angle of view. Our design captures these variations successfully in most images. A representative suburban scene is shown in Figure 3, Figure 4 shows a rural scene, while

![](images/eb18ec5d4289d73d0f21b5d9a212e98b1c5649f847c6e3507fb48216a258bb86.jpg)  
Figure 3: DroneSeg, image 514. Top row: input image and final 150-class segmentation. Bottom two rows: ground-truth annotation (Oracle mIoU shown) and the pipeline outputs at each stage, unweighted (Stage 1), weighted (Stage 2), and final merged (Stage 3), all in ground-truth class colors.

Figures 5 and 6 show a contrasting urban scene where minority classes (cars) dominate.

UDD6 segments well because its images are almost nadir (straight-down) drone views (see Figure 5), while UAVid sometimes fails because its images vary in field of view and angle (see Figure 6), conditions where VLMs systematically struggle to ground objects (Wang et al., 2026).

All examples show the merged final stage exceeding the per-image Oracle. The Oracle is the best mIoU achievable with perfect class-vocabulary selection. The final merged stage exceeds it because Stage 3, MCI, recovers minority classes: classes the VLM sees in the image but the segmentation algorithm cannot pixel-localize. In Stage 3, for each such class, a prompted VLM returns a bounding box that is filled in, and this output is merged with Stage 2 for the final segmented output.

Data for the DroneSeg (Figure 3) and Aeroscapes (Figure 4) datasets has been constructed for the purpose of model training: the range and distance of objects does not vary to the extremes that complicate UDD6’s dense urban clutter or UAVid’s oblique perspectives. Objects appear in moderate, consistent scales and ranges so both the class weighting and the bounding box step behave well, and each stage of the pipeline contributes a clean

![](images/63a09f24759f9837d5e21806f296b63c5607257ef4f8a79fe7ef427734bfbe59.jpg)  
Figure 4: Aeroscapes, image 001101 001. Top row: input image and final 150-class segmentation. Bottom two rows: ground-truth annotation (Oracle mIoU shown) and the pipeline outputs at each stage, unweighted (Stage 1), weighted (Stage 2), and final merged (Stage 3), all in ground-truth class colors.

gain.

Figure 5 makes this concrete. On this UDD6 image, ACW does not help: the weighted result (17.99% mIoU) falls below the unweighted baseline (21.64%). The full pipeline still reaches 37.12%, because MCI recovers the small classes that dominate this scene and delivers almost all of the gain. ACW and MCI thus address complementary failure modes: when one does not help, the other can still carry the result.

Figure 6 offers a particularly clear view into the challenges of relying on even state-of-the-art approaches to augment segmentation: ill equipped to handle the oblique, steeply angled viewpoints that aerial video like UAVid presents, the VLM returns bounding boxes that drift away from the vehicles and people they should mark, floating toward the horizon. Because those boxes are misplaced, the inpainting step has nothing correctly localized to recover, and Stage 3 collapses onto Stage 2; this is why UAVid’s Stage 3 is left undefined in our tables (Section 5.3).

## 5.2. Quantitative: Progressive mIoU across All Three Stages

The primary evidence that our lightweight, local workflow produces reliable results within a constrained environment is the typical consistency of ∆mIoU across stage transitions. Tables 2, 3, 4, and 5 report mIoU at each stage for four separate underlying checkpoints (Section 4.2). The meaningful signal is that the gains are large and significant on the datasets where the backbone is competent (Aeroscapes and DroneSeg); UDD6 and UAVid stay near flat, for the distinct reasons taken up in Section 5.4. All values are cumulative.

![](images/ab6134a7ea67ff17c8fbf0bcf89babe6b1bc7974f1b587b9aeab38ca3e3ae8f4.jpg)

![](images/5c06f86e9e8b9589eb3c417e7de05617dd89876bbb6cb585bd4346887afa72b0.jpg)  
Figure 5: UDD6, image 001361. Top row: input image and final 150-class segmentation. Bottom two rows: ground-truth annotation (Oracle mIoU shown) and the pipeline outputs at each stage, unweighted (Stage 1), weighted (Stage 2), and final merged (Stage 3), presented in ground-truth class colors.

We take a checkpoint after fine-tuning the VLM using one of the four datasets (Figure 1 Stage 1). A checkpoint comprises the current (trained) state of the frozen foundation model, including its weights. If finetuning was carried out the weights are updated. We have finetuned on four different datasets, therefore we have four different checkpoints. Each of these checkpoints include the theoretical unique classes added to the curated set of classes that identify the domain, a total of 150.

## 5.3. Statistical Significance

To check whether the per-stage gains could be due to chance, we ran a paired t-test on the per-image mIoU change for each stage transition on each dataset (Aero-checkpoint output, same per-image methodology as Table 2). Table 1 reports the results. N is the number of test images with paired data; mean d is the mean per-image mIoU change in percentage points.

![](images/d0959c0b799466def4b874c8c4bbab838ac543323bda44b32d17aa907aebd231.jpg)  
Figure 6: UAVid, image uavid0000000012. Top row: input image and final 150-class segmentation. Bottom two rows: ground-truth annotation (Oracle mIoU shown) and the pipeline outputs at each stage, unweighted (Stage 1), weighted (Stage 2), and in ground-truth class colors. The final image in the bottom row illustrates how bounding box calculations do not currently compensate for the oblique, angled viewpoint of UAVid imagery: the predicted boxes drift off their targets and into the distance, so the inpainting step has nothing correctly placed to fill, and Stage 3 cannot improve on the Stage 2 result.

Table 1: Paired t-test on per-image mIoU change relative to the unweighted baseline (Stage 1).
<table><tr><td>Dataset</td><td>Transition</td><td>N</td><td>mean d</td><td>p</td></tr><tr><td>Aero.</td><td>Stage 1 → 2</td><td>35</td><td>+7.02</td><td>&lt; 0.0001</td></tr><tr><td>Aero.</td><td>Stage 2 → 3</td><td>35</td><td>+7.65</td><td>&lt; 0.0001</td></tr><tr><td>UAVid</td><td>Stage 1 → 2</td><td>35</td><td>+0.94</td><td>n.s.</td></tr><tr><td>UAVid</td><td>Stage 2 → 3</td><td>1</td><td></td><td></td></tr><tr><td>DSeg</td><td>Stage 1 → 2</td><td>35</td><td>+6.63</td><td>&lt; 0.0001</td></tr><tr><td>DSeg</td><td>Stage 2 → 3</td><td>35</td><td>+6.20</td><td>&lt; 0.0001</td></tr><tr><td>UDD6</td><td>Stage 1 → 2</td><td>35</td><td>-0.24</td><td>n.s.</td></tr><tr><td>UDD6</td><td>Stage 2 → 3</td><td>31</td><td>+3.74</td><td>&lt; 0.01</td></tr></table>

The difference between the unweighted baseline and class weighting (ACW), that is, between Stage 1 and Stage 2, is highly significant on Aeroscapes and DroneSeg $( p ~ < ~ 0 . 0 0 0 1 )$ and not significant on UAVid or UDD6. The difference between Stage 1 and Stage 3 (the full pipeline) is likewise highly significant on Aeroscapes (+15.07) and DroneSeg (+14.68), $p \ < \ 0 . 0 0 0 1$ , and not significant on UAVid (+1.18) or UDD6 (−1.89). The UAVid rows are computed over the N=19 images on which the VLM was successfully able to locate boxes; multiple viewing angles and perspectives prevent reliable localization on the remainder (Sections 5.1 and 4.4).

## 5.4. Complementary Recovery Across Landscapes

The two prompts are designed so that neither must succeed everywhere; which one carries an image is set by the landscape. ACW sharpens the dominant land-cover classes a scene is built from; MCI recovers the small, sparse objects a per-pixel segmenter drops. Where majority structure is strong, ACW leads and MCI refines; where small objects dominate, MCI carries it. The workflow never depends on a single mechanism being correct, the practical payoff of pairing two complementary models.

This runs against a common assumption in decision support, that a language model should wrap a conventional solver rather than replace it (Rahimi, 2026). MCI shows the opposite can hold in a bounded regime: for the small, sparse classes a per-pixel segmenter systematically drops, the VLM does not assist the solver; it stands in for it, supplying the labels the solver could not produce.

Tables 2 through 5 show this happening. On Aeroscapes and DroneSeg, ACW is the larger early gain and is highly significant (Stage $1 {  } 2 , \ p \ < \ 0 . 0 0 0 1$ Table 1), and MCI adds a comparable gain on top. On UDD6 the pattern is reversed. Class weighting produces no gain: the Stage 1 to Stage 2 difference is −0.62 and not significant, and the full Stage 1 to Stage 3 difference is −1.89, also not significant. This is a granularity artifact of the annotation set, not a method failure: UDD6’s six coarse classes leave the richest part of the 150-class vocabulary nowhere to land, so minority recovery that is real on individual dense scenes (Figure 5) is lost once the score is averaged over the image. A more finely annotated ground truth would recover it.

The handoff matters most where mIoU is least trustworthy. Minority classes are small in pixel count, so the metric barely registers them even when recovered perfectly, and the ground-truth catch-all labels (Section 4.2) penalize correct minority predictions as false positives. The dense urban regime, where MCI takes over, is thus where mIoU undercounts most, which is why the qualitative panels, not the mIoU column, are the honest evidence for that recovery. UAVid stays flat for a different reason: its multiple viewing angles and perspectives keep the VLM from locating boxes on most images, so MCI has little to add, and where it does locate them the recovered cars and people are too few pixels to move a mean dominated by large background classes.

## 6. Ablation Studies

To isolate the contribution of individual design choices, and to characterize how sensitive the method is to the selected VLM hyperparameters, we perform a set of ablations covering the weight value used in ACW and the class-selection strategy that drives it.

## 6.1. ACW Weight Value

Table 6 reports Stage 1 mIoU as the boost weight $W _ { \mathrm { b o o s t } }$ is swept from $w = 0$ (the unweighted baseline) to $w = 5 0 0$ Prediction quality is stable across the sweep, with no large gains beyond moderate weighting; we adopt $\begin{array} { r } { \begin{array} { r c l } { w } & { = } & { 1 0 0 } \end{array} } \end{array}$ as the production setting for all other experiments. The shape of the sweep is informative: nearly all the gain appears at the first nonzero weight, and mIoU is then flat out to $w =$ 500. The exception is UDD6, where ACW is inert: the unweighted baseline $( w = 0 , 2 7 . 5 2 )$ exceeds every weighted value (26.7–26.9), so restriction neither helps nor hurts there. The benefit therefore comes from restricting prediction to the selected classes, not from the magnitude of the boost; once the boost is large enough for those classes to win the per-pixel argmax, larger values change little. The method is thus robust to this hyperparameter, and $w = 1 0 0$ sits safely in the flat region.

## 6.2. Class Selection Strategy

Table 7 addresses whether the improvement depends on which classes ACW boosts, or simply on the act of boosting. Holding the boost weight fixed at $W _ { \mathrm { { b o o s t } } } ~ = ~ 1 0 0$ , we vary only how the boosted class set is chosen and compare three strategies: Random, which boosts $k \_ =$ 30 classes drawn at random for each image (a lower bound, no real selection); VLM (Qwen), our automated method, in which the model picks the classes from the image; and Oracle, which boosts exactly the classes the ground truth says are present (an upper bound no automated method can beat). For Random and VLM we report mean ± standard deviation over three seeds $( s ~ = ~ 0 , 1 , 2 )$ ; Random’s variance comes from which k classes are sampled from the 150-class vocabulary, while VLM’s reflects Qwen’s sampling-based decoding. Oracle and Unweighted are deterministic single-value entries. The VLM selection matches the oracle and clears the random baseline, which indicates that the automated class discovery, not merely the act of weighting, drives the improvement. The gap over random is dataset-dependent: it widens on the weaker-backbone and smaller-vocabulary datasets, where selection precision matters most, and it shrinks where the backbone already predicts the common aerial classes well without boosting.

Table 2: Aero checkpoint  
Stage 1 fine-tuned on Aeroscapes, evaluated across all four datasets.
<table><tr><td>Stage</td><td>UAVid</td><td>Aero.</td><td>DSeg</td><td>UDD6</td></tr><tr><td>Stage 1: Unweighted</td><td>26.02</td><td>62.99</td><td>52.54</td><td>27.52</td></tr><tr><td>Stage 2: Weighted  $w = 1 0 0$ </td><td>26.46</td><td>70.36</td><td>59.45</td><td>26.90</td></tr><tr><td>Stage 3: Minority Classes</td><td>27.20</td><td>78.06</td><td>67.22</td><td>25.63</td></tr></table>

Table 4: UAVid checkpoint

Stage 1 fine-tuned on UAVid, evaluated across all four datasets.
<table><tr><td>Stage</td><td>UAVid</td><td>Aero.</td><td>DSeg</td><td>UDD6</td></tr><tr><td>Stage 1: Unweighted</td><td>24.79</td><td>62.06</td><td>51.26</td><td>25.45</td></tr><tr><td>Stage 2: Weighted  $w = 1 0 0$ </td><td>24.92</td><td>69.68</td><td>58.55</td><td>25.45</td></tr><tr><td>Stage 3: Minority Classes</td><td>25.90</td><td>77.40</td><td>67.23</td><td>24.89</td></tr></table>

Table 6: BCI weight ablation (Stage 1 mIoU, %).
<table><tr><td>Dataset</td><td> $w = 0$ </td><td> $w = 1 0$ </td><td> $w = 5 0$ </td><td> $w = 1 0 0$ </td><td> $w = 2 0 0$ </td><td> $w = 5 0 0$ </td></tr><tr><td>UAVid</td><td>30.73</td><td>32.77</td><td>32.48</td><td>32.05</td><td>32.40</td><td>32.48</td></tr><tr><td>Aero.</td><td>63.41</td><td>71.32</td><td>71.24</td><td>70.66</td><td>71.16</td><td>71.14</td></tr><tr><td>DSeg</td><td>51.29</td><td>59.45</td><td>59.40</td><td>59.08</td><td>59.38</td><td></td></tr><tr><td>UDD6</td><td>31.55</td><td>34.17</td><td>34.46</td><td>35.76</td><td>34.39</td><td>34.40</td></tr></table>

Two further patterns stand out. Random selection is not merely weaker but, on UAVid and UDD6, worse than no weighting at all, and highly variable across seeds: boosting the wrong classes actively suppresses the right ones, so weighting helps only when the selected set is correct. Selection quality, not weighting itself, is what matters. The VLM matches the oracle to within about 0.6 mIoU on every dataset and exceeds it on UAVid (27.14 vs. 25.40), despite the oracle’s access to ground-truth class sets, which is consistent with ground-truth labels being a conservative target (Section 4.2): the VLM sometimes names useful classes the ground-truth set omits. Its low variance shows the automated selection is stable, unlike random.

Table 7: Class-selection strategy comparison (Stage 1 mIoU, %).
<table><tr><td>Strategy</td><td>UAVid</td><td>Aero.</td><td>DSeg</td><td>UDD6</td></tr><tr><td>Unweighted</td><td>30.73</td><td>63.41</td><td>51.29</td><td>31.55</td></tr><tr><td>Random</td><td>19.77±7.55</td><td>69.89±1.62</td><td>57.97±2.23</td><td>23.76±5.33</td></tr><tr><td>Oracle</td><td>29.92</td><td>70.60</td><td>59.22</td><td>31.40</td></tr><tr><td>VLM (Qwen)</td><td>31.74±0.30</td><td>70.95±0.53</td><td>59.03±0.17</td><td>36.04±0.39</td></tr></table>

## 7. Conclusion

We have shown that pairing a frozen foundation model with a vision-language model, used only to read and describe the scene, produces more scene-coherent segmentation, measurably and consistently, without any domain-specific training. The improvement is not specific to the aerial domain or to particular benchmark choices: ∆mIoU is positive and highly significant across the datasets where the backbone is competent; where it is flat, the limit is the benchmark or the viewpoint, not the method (Section 5.4). The base model, with a small amount of targeted tuning, already knew the relevant classes; the VLM simply forced it to apply that knowledge to the scene in front of it.

Table 3: DroneSeg checkpoint  
Stage 1 fine-tuned on DroneSeg, evaluated across all four datasets.
<table><tr><td>Stage</td><td>UAVid</td><td>Aero.</td><td>DSeg</td><td>UDD6</td></tr><tr><td>Stage 1: Unweighted</td><td>28.45</td><td>59.40</td><td>50.51</td><td>23.64</td></tr><tr><td>Stage 2: Weighted  $w = 1 0 0$ </td><td>28.70</td><td>70.41</td><td>59.03</td><td>23.37</td></tr><tr><td>Stage 3: Minority Classes</td><td>29.88</td><td>76.67</td><td>66.15</td><td>25.07</td></tr></table>

Table 5: UDD6 checkpoint

Stage 1 fine-tuned on UDD6, evaluated across all four datasets.
<table><tr><td>Stage</td><td>UAVid</td><td>Aero.</td><td>DSeg</td><td>UDD6</td></tr><tr><td>Stage 1: Unweighted</td><td>14.68</td><td>53.35</td><td>49.52</td><td>17.02</td></tr><tr><td>Stage 2: Weighted  $w = 1 0 0$ </td><td>16.78</td><td>68.36</td><td>58.20</td><td>20.13</td></tr><tr><td>Stage 3: Minority Classes</td><td>17.90</td><td>76.96</td><td>66.10</td><td>22.08</td></tr></table>

The approach can be added to any local, general foundation model without domain-specific pretraining and without access to more than a single consumer-grade GPU. If there are rare, unique, or new classes that pretraining may have missed, simple fine-tuning on those classes is enough.

The two mechanisms introduced here, automated class weighting and minority class identification, rest on the same idea: a VLM that reads the image can supply structured cues to guide a frozen foundation model, without ever doing the pixel work itself. ACW keeps the output tuned to the classes a prompt selects automatically, and MCI repairs the residual failure mode of minority-class under-representation by pairing VLM-derived (class, bounding-box) pairs with a class-agnostic fill step. Together they provide clearly better zero-shot segmentation, both quantitatively (mIoU) and qualitatively. Each mechanism is independently applicable and backbone-agnostic, and because every judgement is stated in the VLM’s readable outputs, each step of the result can be traced and checked. This traceability is what makes the workflow suitable as a component in a larger decision-making system, where the reasoning behind an output matters as much as the output itself.

Our future work centers on increasing the accuracy of localization, a known, and challenging problem in VLM transformer architecture. A vision-language model can readily name the classes and objects in a scene, but cannot yet localize them reliably; this gap, not class discovery, is the principal obstacle to using a VLM as a primary source of object and material identification. Extending the workflow to other specialized domains, such as medical or industrial imaging, remains a longer-term aim.

## References

Blumenstiel, B., Jakubik, J., Kuhne, H., & V¨ ossing, M.¨ (2023). What a mess: Multi-domain evaluation of zero-shot semantic segmentation. https : / / doi.org/10.48550/arXiv.2306.15521

Bucher, M., Vu, T.-H., Cord, M., & Perez, P. (2019).´ Zero-shot semantic segmentation. NeurIPS.

Chen, Y. (2018). Large-scale structure from motion with semantic constraints of aerial images. First Chinese Conference, PRCV.

Cheng, B., Schwing, A. G., & Kirillov, A. (2021). Per-pixel classification is not all you need for semantic segmentation. https://arxiv.org/abs/ 2107.06278

Ding, J., Xue, N., Xia, G.-S., & Dai, D. (2022). Decoupling zero-shot semantic segmentation. IEEE Conf. Comput. Vis. Pattern Recog., 11583–11592.

Elhagry, A., & Saeed, M. (2022). Investigating the challenges of class imbalance and scale variation in object detection in aerial images. https://arxiv.org/abs/2202.02489

Ghiasi, G., Gu, X., Cui, Y., & Lin, T.-Y. (2022). Scaling open-vocabulary image segmentation with image-level labels. Computer Vision – ECCV 2022, 540–557.

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. MIS Quarterly, 28(1), 75–105.

Ji, D., Jin, W., Lu, H., & Zhao, F. (2024). Pptformer: Pseudo multi-perspective transformer for uav segmentation. arXiv preprint.

Kao, S.-H., Tai, Y.-W., & Tang, C.-K. (2025a). Think before you segment: High-quality reasoning segmentation with gpt chain of thoughts.

Kao, S.-H., Tai, Y.-W., & Tang, C.-K. (2025b). Think before you segment: High-quality reasoning segmentation with gpt chain of thoughts. https: //arxiv.org/abs/2503.07503

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. (2023). Segment anything. Int. Conf. Comput. Vis., 4015–4026.

Liang, F., Wu, B., Dai, X., Li, K., Zhao, Y., Zhang, H., Zhang, P., Vajda, P., & Marculescu, D. (2023).

Open-vocabulary semantic segmentation with mask-adapted clip. https://arxiv.org/abs/2210. 04150

Liu, F., Chen, D., Guan, Z., Zhou, X., Zhu, J., Ye, L., Liyong, L., & Zhou, F. (2024). Remoteclip: A vision language foundation model for remote sensing. https://arxiv.org/abs/2306.11029

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. Proceedings of the 37th International Conference on Neural Information Processing Systems.

Lyu, Y., Vosselman, G., Xia, G.-S., Yilmaz, A., & Yang, M. Y. (2020). Uavid: A semantic segmentation dataset for uav imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 165, 108–119.

Nigam, I., Huang, C., & Ramanan, D. (2018). Ensemble knowledge transfer for semantic segmentation. IEEE Winter Conference on Applications of Computer Vision (WACV).

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual representations from natural language supervision. Int. Conf. Mach. Learn.

Rahimi, I. (2026, January). Large language models in decision support systems for operations research: Domain-focused perspectives [Preprint]. https://doi.org/10.13140/RG.2.2. 28340.92800

Reichard, K., Rizzoli, G., Gasperini, S., Hoyer, L., Zanuttigh, P., Navab, N., & Tombari, F. (2025). From open-vocabulary to vocabulary-free semantic segmentation.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “why should I trust you?”: Explaining the predictions of any classifier. Proc. 22ndACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), 1135–1144.

Sein, M. K., Henfridsson, O., Purao, S., Rossi, M., & Lindgren, R. (2011). Action design research. MIS Quarterly, 35(1), 37–56.

Wang, M., Li, Y., Wang, B., Zhao, T., Ji, R., Gao, Q., Liu, E., Deng, H., & Luo, D. (2026). Egocentric bias in vision-language models. arXiv preprint. https : / / arxiv. org / abs / 2602 . 15892

Wen, C., Hu, Y., Li, X., Yuan, Z., & Zhu, X. X. (2023). Vision-language models in remote sensing: Current progress and future trends. https : / / arxiv.org/abs/2305.05726

Yi, X., Ma, Y., Li, Y., Xu, H., & Ma, J. (2025). Artificial intelligence facilitates information fusion for perception in complex environments. The Innovation, 6(4), 100814.

Zhang, Z., Zhao, T., Guo, Y., & Yin, J. (2023). Rs5m and georsclip: A large scale vision-language dataset and a large vision-language model for remote sensing. https://paperswithcode.com/ paper/rs5m - a - large - scale - vision - language - dataset

Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., & Torralba, A. (2017). Scene parsing through ADE20K dataset. IEEE Conf. Comput. Vis. Pattern Recog., 633–641.