# SePArate: Segmenting Paterns from Defects in Wafer Manufacturing Using Weak Supervision

Dain Kwon   
Seoul National University   
Seoul, South Korea   
dain.kwon@snu.ac.kr   
Kanghyun Choi   
Seoul National University   
Seoul, South Korea   
kanghyun.choi@snu.ac.kr   
Minseok Choi   
SK Hynix   
Seoul, South Korea   
choineral@gmail.com   
Changmin Shin   
Seoul National University   
Seoul, South Korea   
scm8432@snu.ac.kr   
Hyeyoon Lee   
Seoul National University   
Seoul, South Korea   
hylee817@snu.ac.kr   
Sunjong Park   
Seoul National University   
Seoul, South Korea   
ryan0507@snu.ac.kr   
Jaewon Jang   
SK Hynix   
Seoul, South Korea   
jaewon2.jang@sk.com   
Jinho Lee   
Seoul National University   
Seoul, South Korea   
leejinho@snu.ac.kr

## Abstract

In semiconductor manufacturing, defect analysis is essential, but manual inspection cannot scale. However, existing automated inspection methods remain insuficient for root-cause analysis and process optimization. To this end, we present SePArate, a weakly supervised wafer defect segmentation method. SePArate enables pixel-level separation of patterns by leveraging only image-level annotations. It consists of a three-phase training: encoder pretraining, knowledge transfer to learn spatial cues, and training on synthetic mixed-defect data for accurate segmentation. Experiments demonstrate that SePArate outperforms the baselines. The code is available at https://github.com/meowrowan/SePArate.

## Keywords

Semiconductor Defects, Mixed-type Defect Patterns, Image Segmentation, Weakly Supervised Semantic Segmentation

## ACM Reference Format:

Dain Kwon, Changmin Shin, Sunjong Park, Kanghyun Choi, Hyeyoon Lee, Jaewon Jang, Minseok Choi, and Jinho Lee. 2026. SePArate: Segmenting Patterns from Defects in Wafer Manufacturing Using Weak Supervision. In 63rd ACM/IEEE Design Automation Conference (DAC ’26), July 26–29, 2026, Long Beach, CA, USA. ACM, New York, NY, USA, 7 pages. https://doi.org/10. 1145/3770743.3804046

## 1 Introduction

Defect analysis is an essential step in semiconductor manufacturing to minimize faults during production. By inspecting patterns of failures, engineers can trace underlying causes of defects, leading to process refinement based on the found issues [2, 4, 23]. Figure 1 illustrates an inspection scenario. During the production, defects are detected, either at the wafer-level, or at the specific componentlevel (e.g., mat-level for a DRAM [8]). These defect maps become classified images organized into datasets. Then, experts analyze these to identify issues, trace their causes, and provide feedback.

![](images/07f1a90975586eb304544bb551d837a244836418ef50ecd91917ba4d8fc3e685.jpg)

![](images/3d6194f589b226e621e29d2ae1cc5bae0c1853e8ed8d7d67c43fe0f63059ada2.jpg)  
Figure 1: Automation of the overall visual inspection of defects and continuous improvement by integrating SePArate.

However, the sheer volume of inspection data makes manual inspection infeasible. Modern manufacturing processes generate terabytes of defect data each day [12, 17], containing diverse and intricate pattern variations that could guide further manufacturing process optimization. In such conditions, human inspection cannot scale to the full dataset. Therefore, automated defect pattern analysis is a natural progression for wafer defect detection.

Despite recent advances, current AI-based defect analysis methods [5, 16] remain limited in practical utility. Many of them are classification methods, i.e., they detect whether predefined patterns exist on wafers. This leads to a significant limitation: They discard geometric attributes such as shape, area, and orientation. These attributes provide essential information about the root cause of the defects and clues on how to fix them. Consequently, automated pixel-level defect pattern annotation is necessary, but AI-based segmentation methods do not support wafer defect analysis and are hard to train due to the absence of a large, annotated dataset.

To address this, we propose SePArate: a weakly supervised defect pattern segmentation approach requiring only classification labels without costly pixel-level annotations. SePArate generates pixel-level masks that isolate individual patterns from defect images, enabling finer categorization, new pattern discovery, and automated database construction that previously required manual curation. An end-to-end automation of the defect analysis pipeline can be initiated, as illustrated in Figure 1. Using a combination of self-generated supervision and a task-specific fine-tuning approach, SePArate efectively isolates diverse defect patterns. Results on a production-line DRAM mat-level dataset and a public waferlevel dataset demonstrate that SePArate significantly outperforms baselines. The key contributions of our work are as follows.

• We are the first to tackle segmentation for binary mixed-type defect pattern images with weak supervision.

• We devise SePArate, a framework that trains a defect pattern segmentation network under image-level supervision, without requiring additional costly instance annotations.

• SePArate greatly surpasses existing baselines, highlighting the value of its defect-aware design.

## 2 Background

## 2.1 Defect Data in Wafer Manufacturing

Defects occur at various granularities in wafer manufacturing, as shown in the leftmost side of Figure 1. For generalization, we use two datasets of diferent granularity: a private, DRAM mat-level defect dataset MATDefects, obtained from an actual DRAM manufacturing process, and a public wafer-level defect dataset MixedWM38 [25]. The pixels are binary: 0 for normal, and 1 for defects.

MATDefects contains data from DRAM products, where each pixel represents a DRAM cell (i.e., a bit). A DRAM die [8] often exhibits a regular structure composed of multiple banks, each comprising multiple mats. Each mat is a square cell array, typically 512-2,048 in size, depending on the product. MATDefects includes single and mixed-type defect patterns. The single defect patterns are classified as: row-wise (wordline or wordline driver issues), column-wise (bitline or sense amplifier issues), or group (individual cell issues). As in Figure 2, multiple single-type defects may appear in an image, forming a mixed-type defect pattern.

MixedWM38 contains images generated from full wafers, where each pixel corresponds to a die. It contains 38 single and mixed-type patterns, with 8 single-type patterns, also shown in Figure 2.

## 2.2 Segmentation without Ground-Truth Data

Defect pattern datasets usually provide only image-level ground truths, but we need pixel-level annotation for each pattern to do segmentation. Two approaches are feasible: zero-shot foundation models for segmentation [11, 19], or weakly supervised semantic segmentation (WSSS) methods [1, 9, 26].

Zero-shot models such as CLIPSeg [11] and GroundedSAM [19] use vision-language pretraining to align visual features with text semantics. As they enable text-guided segmentation without labeled examples, defect-pattern text prompts may be used to guide them.

There are also WSSS methods [1, 9, 26], which train segmentation models with image-level labels. Most methods train a classifier, generate pixel-level pseudo-labels from intermediate activations (e.g., Class Activation Maps (CAMs) [1, 9] or attention maps [26]), then train a segmentation model with these synthetic labels.

![](images/7703d2c251b591253e79eefb7bc554e467f0d2aaf9800dcc3ef10220b4324cb5.jpg)  
Figure 2: Single and mixed-type defect pattern examples for MATDefects and MixedWM38.

The main challenge with these alternatives is that they rely on visual cues–such as color, texture, or shading–that are largely absent in our binary data. With no diversity in pixel values to infer about shape, size, or depth, we are only left with geometric structure to diferentiate between patterns.

Therefore, a new defect-aware approach is required: a weakly supervised framework that explicitly leverages structural and distributional cues without depending on visual semantics. SePArate is explicitly designed for this setting, providing a practical solution for defect pattern segmentation.

## 2.3 Existing Works for Binary Defect Data

For binary defect images, existing works have focused on multilabel classification, detecting the presence ofeach single-type defect pattern in mixed-type defects. The classification models evolve from CNNs [6, 10, 21, 24, 25, 28] to transformers and state space models [5, 16]. There exist a few segmentation works, [14, 15] but they assume only a single defect instance per image and thus cannot separate multiple single-type defect patterns.

One work, SSB-Rec [27], employs defect segmentation-like training. However, its primary objective remains to enhance classification quality. We include it as a baseline to test its applicability to defect pattern segmentation. To our knowledge, no prior work has attempted to isolate single-defect patterns from mixed-type defects.

## 3 Challenges in Defect Pattern Segmentation

The goal is to separate defect pixels into pattern-specific subsets within a defective region. However, defect pattern segmentation poses unique challenges that are not present in natural image segmentation. This is because, although both are image representations, they have basically diferent characteristics. We list them as below.

(1) Dificulty of obtaining high-quality labels. Obtaining pixellevel annotations is particularly challenging and expensive. The pixel-level labels require substantial eforts from experts, compared to labels for image classification. Moreover, common crowdsourcing for annotation is often infeasible in the semiconductor domain due to strict confidentiality requirements.

![](images/0146f04cfd8ad09d2e9f9cb71aa695ded3a7665c337b30c43c3ef0fa850bab3c.jpg)  
Figure 3: A full overview of our proposed method, SePArate. i) Phase 1 trains the encoder for multi-label classification, enabling the model to learn about defect patterns using image-level labels. ii) Phase 2 transfers classification knowledge to the U-Net by training with soft supervision maps and pattern-weighted Dice loss. iii) Phase 3 further tunes the model using synthetic mixed-defect data, improving precision with pattern-weighted Dice loss and absence-penalty scheduling.

(2) Binary-valued pixels. Defect image pixels only have binary values: normal (0) or defective (1). This implies that there are limited features to learn from, as the image only comprises a white foreground region with a lack ofvisual cues. Without such cues, models must rely on structural patterns and distributional statistics, making the task uniquely challenging.

(3) Sparsity of defect patterns. Defect patterns may exhibit pixellevel sparsity (small defective regions within large backgrounds) and image-level sparsity (rare occurrence of certain patterns). As shown in Figure 2, some patterns have extremely sparse appearances, where background (non-defect regions) dominate the image. This indicates pixel-level sparsity. Some patterns may be scarce due to specific causes in manufacturing. This leads to image-level sparsity where data imbalance may occur.

Unlike existing baselines that neglect these factors, SePArate explicitly addresses such domain-specific challenges to achieve efective defect pattern segmentation quality.

## 4 Methodology

In Figure 3, we illustrate the overall training with SePArate. The entire process operates without expensive pixel-level annotations. SePArate uses the image-level labels to first learn basic pattern-level features, then leverages this knowledge for identifying key spatial features. Specifically, SePArate trains the model through three phases: 1) Pretraining the encoder for classification (section 4.1), 2) Adapting it to learn spatial cues of defect patterns (section 4.2), and 3) Final training on synthetic mixed-defect data for precise pattern segmentation (section 4.3). We use the U-Net [20] architecture, as it is widely used for segmentation on real-world images.

## 4.1 Phase 1: Encoder Training

Phase 1 initializes the U-Net encoder with basic knowledge required to identify each single-type defect pattern. This ability is later exploited to generate cues that guide the model to spatially localize each defect pattern. In Phase 1, we use the available image-level labels to train the encoder for classification task. For this, an MLPbased classification head is attached after the encoder to predict the presence of each single-type defect in the image. The model is trained with the mean of Binary Cross-Entropy Loss for each label, as defect pattern classification is a multi-label task.

## 4.2 Phase 2: Spatial Knowledge Transfer

In Phase 2, we train the full U-Net architecture (encoder and decoder). The goal at this stage is to learn spatial indicators of defect patterns. This is done by training the model with soft supervision maps with a pattern-weighted Dice loss. Given the absence of semantic information in the images, we generate soft supervision maps to supply pattern-aware hints, stabilizing the model’s early learning process to diferentiate defect patterns. The pattern-weighted Dice Loss helps this process by assigning loss weights, ensuring balanced optimization across defect patterns.

4.2.1 Soft Supervision Maps. Soft supervision maps act as intermediate pseudo-labels that guide the model toward defect regions. They incorporate global and local spatial cues, each extracted from the image-level classifier, and the defective regions of the image. To construct these maps, we combine local density maps and class activation maps (CAM, [3, 22]) from the pretrained classifier.

![](images/3f752ff307ccb53a7672c7ec24cb603a2e9559720d6add075461478fd06bbacd.jpg)  
Figure 4: Generation of soft supervision maps. For each image, local density map (bottom left) and CAM (top center) are computed, combined to generate soft supervision maps.

On the one hand, we generate a local density map of the defect images. Local density maps provide global cues of the image by highlighting regions where defect pixels are concentrated. Defect patterns typically form clusters, so we compute these maps using a simple mean filter that amplifies dense defect regions and smooths out sparse areas. High-value regions in the density map represent probable defect zones, as illustrated in Figure 4.

On the other hand, we also utilize CAM, which complements the density maps by identifying regions that influence the classifier’s decision for each defect type. While density alone ofers spatial importance, it does not diferentiate between patterns. Therefore, CAM introduces type-specific relevance, enabling the supervision to carry pattern-specific information.

Finally, we generate the final soft supervision maps by multiplying each type-specific CAM with the density map, followed by normalization on pixel values. These score-like pixel values rep resent the joint contribution of defect concentration and patternspecific signals from the classifier. The U-Net is trained to align its output with these maps, using the pattern-weighted Dice loss which balances training across defect patterns.

4.2.2 Patern-Weighted Dice Loss. Due to their sparse and monotonic regions, defect patterns are dificult for the model to separate in the early stages of training. This often leads the model to default to predicting mostly absent masks for some patterns. To mitigate this issue and encourage the model to first detect coarse regions of patterns, thereby improving early-stage recall, we introduce the pattern-weighted Dice Loss. We use the loss to ensure the model to primarily capture the existing defects and further enable balanced training for all defect patterns, regardless of the available samples of each defect pattern.

The original Dice loss [13] is widely used for segmentation tasks:

$$
\mathcal { L } _ { \mathrm { d i c e } , i , p } = 1 - \frac { 2 ( \sum y _ { i , p } \cdot \hat { y } _ { i , p } ) + \epsilon } { \sum y _ { i , p } + \sum \hat { y } _ { i , p } + \epsilon }\tag{1}
$$

where $\hat { y } _ { i , p }$ denotes the predicted mask and $y _ { i , p }$ denotes the ground truth mask for image � and single-type pattern $\mathcal { P } \cdot$ We observed that directly using the Dice loss led to a drastic degradation of less frequent and scarce defect patterns. This behavior stems from the loss, pushing the model toward absent-mask predictions for rarely occurring patterns. Therefore, we assign loss weights to every image-pattern pair to help the model focus more on the presence of each pattern, rather than being biased by the absence of patterns. For each image � and pattern ${ \boldsymbol { p } } ,$ we define the indicator:

$$
\begin{array} { r } { \mathrm { L e t } \delta _ { i , p } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \mathrm { p a t t e r n } \ p \mathrm { e x i s t s } \mathrm { i n t h e } \mathrm { i m a g e } i , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{2}
$$

Let the dataset-level occurrence count ofpattern � be $\begin{array} { r } { K _ { p } = \sum _ { i = 1 } ^ { N } \delta _ { i , p } } \end{array}$ where � is the total number of images in the train set.

The Pattern-Weighted Dice Loss is then defined as

$$
\mathcal { L } _ { \mathrm { P W D L } } = \sum _ { p = 1 } ^ { P } \frac { 1 } { K _ { p } } \sum _ { i = 1 } ^ { N } \delta _ { i , p } \left( 1 - \frac { 2 ( \sum y _ { i , p } \cdot \hat { y } _ { i , p } ) + \epsilon } { \sum y _ { i , p } + \sum \hat { y } _ { i , p } + \epsilon } \right) ,\tag{3}
$$

Equivalently,

$$
\mathcal { L } _ { \mathrm { { P W D L } } } = \sum _ { p = 1 } ^ { P } \frac { 1 } { K _ { \mathstrut p } + \epsilon } \sum _ { i : \delta _ { i , p } = 1 } \mathcal { L } _ { \mathrm { { d i c e } } , i , p }\tag{4}
$$

The Pattern-Weighted Dice Loss stabilizes learning across patterns of diferent prevalence and improves segmentation on scarce defect patterns.

## 4.3 Phase 3: Synthetic Defect Training

Phase 3 further refines the performance of SePArate with finetuning using a synthetic dataset with accurate segmentation masks.

4.3.1 Synthetic Dataset Generation. As discussed in the previous section, because the datasets are binary-valued images, any 1- valued pixels correspond to one ofthe target patterns. Therefore, for single-type defects, all 1-valued pixels can directly serve as the segmentation masks for those patterns. The remaining problem is that this only applies to the single-type defects, and not to the mixedtype defects. To tackle this problem, we synthesize multi-type defect masks by merging single-type defect masks. For example, in the MATDefects dataset, a row-defect map and a column-defect map can simply be added together with slight shifts for augmentation. This creates a synthetic, but realistic mixed-type defect maps, which are shown in Figure 5. Thus, for the training, we use the synthetic dataset’s segmentation masks as the ground truth.

4.3.2 Absence Penalty Loss & Scheduling. When training the model in Phase 3, we train the model with the aforementioned Pattern-Weighted Dice Loss and the Absence Penalty (AP) Loss. Here, the Absence Penalty Loss is introduced to reduce the false-positive pixels. Since the absent sample pairs are ignored in Phase 2 by design Equation (3), the trained model has a tendency to include irrelevant pixels, or sometimes show defect patterns that are not present. To address this, we add a loss term that penalizes such a prediction with Binary Cross-Entropy (BCE) Loss.

$$
\begin{array} { r } { \mathrm { L e t } \hat { y ^ { \prime } } _ { i , p } ( x ) = \left\{ \begin{array} { l l } { \hat { y } _ { i , p } ( x ) , } & { \mathrm { i f } \hat { y } _ { i , p } ( x ) \geq \tau } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{5}
$$

where � indexes the location of pixels in $\hat { y } _ { i , p } ,$ and � is the confidence threshold. If the confidence is greater than or equal to �, it is included in the loss. We gradually reduce the value of � from 0.5 to 0.0, depending on loss convergence. Define the pixel-wise BCE against an all-zero target as

$$
\mathcal { L } _ { \mathrm { B C E } } \Big ( \hat { y ^ { \prime } } _ { i , p } , 0 \Big ) = - \frac { 1 } { | \Omega | } \sum _ { x \in \Omega } \log \bigr ( 1 - \hat { y ^ { \prime } } _ { i , p } ( x ) + \epsilon \bigr ) ,\tag{6}
$$

![](images/f8e0a5f141bffe42f504acc4f951b6de72b05e8c078a4ecb4452e7b6d0e392e3.jpg)  
Figure 5: Generation of synthetic mixed-type defect images.

Table 1: Comparison of Mean IoU (mIoU) for Defect Pattern Segmentation (%)
<table><tr><td rowspan="2">Method</td><td colspan="4">MATDefects</td><td colspan="8">MixedWM38</td></tr><tr><td>Column</td><td>Row</td><td>Group</td><td>Mean</td><td>Center</td><td>Donut</td><td>EdgeLoc</td><td>EdgeRing</td><td>Loc</td><td>NearFull Scratch</td><td>Random</td><td>Mean</td></tr><tr><td>CLIPSeg [11]</td><td>4.20</td><td>4.32</td><td>4.12</td><td>4.21</td><td>5.16</td><td>5.06</td><td>4.94</td><td>4.82</td><td>4.87</td><td>4.87 5.17</td><td>5.28</td><td>5.02</td></tr><tr><td>MCT [26]</td><td>42.53</td><td>50.60</td><td>31.66</td><td>41.60</td><td>5.60</td><td>16.89</td><td>15.34 33.73</td><td>5.74</td><td>21.43</td><td>4.36</td><td>82.86</td><td>23.24</td></tr><tr><td>ReCAM [1]</td><td>37.23</td><td>0.00</td><td>23.71</td><td>20.31</td><td>10.52</td><td>37.51</td><td>9.63 11.56</td><td>8.69</td><td>63.92</td><td>8.33</td><td>93.55</td><td>30.46</td></tr><tr><td>S2C [9]</td><td>60.57</td><td>40.13</td><td>32.49</td><td>44.40</td><td>1.43</td><td>0.20</td><td>13.20 10.75</td><td>18.18</td><td>0.00</td><td>8.90</td><td>0.00</td><td>6.58</td></tr><tr><td>SSB-Rec [27]</td><td>55.59</td><td>50.89</td><td>44.81</td><td>50.43</td><td>5.44</td><td>18.71</td><td>0.84</td><td>26.57 56.32</td><td>32.94</td><td>16.43</td><td>67.53</td><td>28.10</td></tr><tr><td>Ours, SePArate</td><td>70.91</td><td>66.66</td><td>64.68</td><td>67.42</td><td>94.10</td><td>60.56</td><td>32.20</td><td>69.23 75.35</td><td>83.33</td><td>39.21</td><td>98.31</td><td>69.04</td></tr></table>

MCT: MCTFormer, SSB-Rec: Semantic Segmentation-Based Wafer Map Mixed-Type Defect Pattern Recognition

where Ω denotes the set of all pixels above the threshold in the image for $\hat { y ^ { \prime } } _ { i , p } .$ . Then the Absence Penalty loss is

$$
\mathcal { L } _ { \mathrm { { A P } } } = \sum _ { p = 1 } ^ { P } \frac { 1 } { K _ { p } ^ { \prime } + \epsilon } \sum _ { i = 1 } ^ { N } a _ { i , p } \mathcal { L } _ { \mathrm { { B C E } } } \left( \hat { y ^ { \prime } } _ { i , p } , 0 \right) ,\tag{7}
$$

where $a _ { i , p } = 1 - \delta _ { i , p }$ and $\begin{array} { r } { K _ { \mathcal { p } } ^ { \prime } = \sum _ { i = 1 } ^ { N } 1 - \delta _ { i , j } } \end{array}$

Consequently, the total loss is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { P W D L } } + \lambda _ { \mathrm { a p } } \mathcal { L } _ { \mathrm { A P } } ,\tag{8}
$$

where $\lambda _ { \mathrm { d i c e } } , \lambda _ { \mathrm { a p } } \geq 0$ are weighting hyperparameters.

## 5 Experimental Results

## 5.1 Experimental Settings

We use two datasets for evaluation: MATDefects and MixedWM38. Each dataset was randomly split into 80% training, 10% validation, and from the remaining 10%, we manually annotated 1,000 samples and used them as test sets. Model prediction was performed for defective regions in each image. In MixedWM38, all non-defective pixels are unified to 0, resulting in binary images. While class labels were provided, test-set masks were manually annotated at the pixel level. Our framework was implemented with PyTorch [18]. Experiments were run on NVIDIA RTX 4090 GPUs.

We chose baselines compatible with our scenario: Segmentation methods that do not require pixel-level annotations (See 2.2): CLIPSeg [11], MCT [26], ReCAM [1], S2C [9] and SSB-Rec [27]. CLIPSeg [11] is a widely used zero-shot segmentation model. S2C [9] generates pixel-level masks refined by SAM [7]. MCT [26] uses class-specific tokens in the transformer to obtain discriminative attention maps. ReCAM [1] introduces a loss function to improve object boundaries in CAM-based pseudo-labels. Lastly, SSB-Rec [27] also leverages the idea of generating a pseudo dataset with single-type defect images, but uses it to train classification models. We modify the method to do a segmentation task, which is trivially done as it already uses U-Net with a segmentation-based loss.

## 5.2 Comparison of Defect Segmentation Results

5.2.1 Performance Comparison. Table 1 shows the performance comparison by evaluating mean Intersection-over-Union (mIoU) results. As shown in table 1, SePArate outperforms baselines in both datasets by a great margin. For the MATDefects dataset, the average mIoU is 67.42%, which shows a superior value to be used in prac tice. For the MixedWM38 dataset, the average mIoU is 69.04%, also showing a reliable performance, considering the larger number of classes. On the other hand, the baselines do not show practical performance, or even fail to work on some patterns. This degradation occurs because baselines rely on rich visual features that are not present in binary defect images. Therefore, its general knowledge does not help the segmentation of the defect patterns when using only weak visual cues from the binary pixels. Unlike the baselines, only SePArate successfully performs defect pattern segmentation, validating the design and its suitability for practical use.

Table 2: Ablation Study of SePArate with MATDefects
<table><tr><td rowspan="3">Phase 1</td><td rowspan="3">Phase 2</td><td rowspan="3">Phase 3</td><td colspan="4">mIoU (%)</td></tr><tr><td>Column</td><td>Row</td><td>Group</td><td>Mean</td></tr><tr><td></td><td>X</td><td>X</td><td>0.00</td><td>0.00</td><td>0.61</td><td>0.20</td></tr><tr><td></td><td>√</td><td>X</td><td>42.18</td><td>42.32</td><td>33.98</td><td>39.49</td></tr><tr><td></td><td>X</td><td>√</td><td>62.30</td><td>56.54</td><td>0.14</td><td>39.66</td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td>70.91</td><td>66.66</td><td>64.68</td><td>67.42</td></tr></table>

5.2.2 Visualizations. We visualize defect segmentation results for the test images of MATDefects and MixedWM38 in Figure 7. Each example shows the original defect image where multiple pattern types overlap. SePArate successfully disentangles these mixed patterns. Pixels of the same pattern share identical colors. The segmented regions exhibit clear morphological distinctions that accurately represent each defect pattern. This shows the practical value of leveraging SePArate for automated defect analysis, preserving critical pattern and shape information.

![](images/35c763ade831e8ee59c06b39c36bcce5586fec69fe8e1075ee89438129305618.jpg)

![](images/4d1cf1c78c7efd740bf3e6c9c3166b3257072955ad66acbc9d363329129ff6d6.jpg)  
Figure 6: Precision and Recall comparison with and without Phase 2 in SePArate, with MATDefects. It shows improved metrics when trained using all 3 phases.

![](images/361a995a6512d88cb4bf3cccf428146abae75c0c85e3c2994d4e4af1f62cb370.jpg)  
Figure 7: Examples of segmented defect pattern in test images, using our proposed method, SePArate. For proprietary reasons, the examples of MATDefects were slightly modified from the original data. Key patterns were preserved regardless of modification.

## 5.3 Further Analyses of SePArate

5.3.1 Ablation Study ofPhases. Table 2 shows the ablation study of SePArate where we evaluate models trained with selective phases.

The first row shows that Phase 1 yields almost no segmentation capability, with a mIoU of 0.20%. When adding Phase 2, the mean score increases to 39.49%, indicating that soft supervision helps the model learn spatial cues but remains insuficient. Similarly, using only Phase 3 with Phase 1 increases the mean mIoU further to 39.66%, showing that refinement is efective but still incomplete, as the model fails on "group" patterns. The final setting with all phases corresponds to SePArate, achieving the highest performance with a mean mIoU of 67.42%, validating the overall design.

A notable observation is the substantial mIoU gap in "group" pattern segmentation when comparing the third and final rows. Without soft supervision maps (third row), the model fails to isolate group defects. This confirms the crucial role of Phase 2. As group defects typically co-occur with line-defects (Figure 7), Phase 2 proves essential for disentangling such overlapping patterns.

We further analyze the synergistic efects of phases in SePArate, shown in Figure 6 with MATDefects. The results for the model trained with all phases of SePArate are shown in green. The results in red are obtained only with Phase 1 and Phase 3. Notably, the 0.1% recall for the "group" pattern when Phase 2 is excluded suggests that the model only captures a very small portion of true defect pixels. The model collapses without an early-stage understanding ofspatial context about defect patterns, highlighting Phase 2’s primary role in supplying knowledge of defect regions. By comparing the two, we can see that the overall segmentation is better with Phase 2.

![](images/86e391f5d40c6419749778226159c547921afbf2b849d08f4ff61fbb534d42de.jpg)  
Figure 8: Loss term ablation for Phase 3 with MixedWM38. The left figure compares mIoU across single-type defect patterns, and the right depicts the diference in Precision.

5.3.2 Loss Term Analysis. Figure 8 illustrates the improvement in segmentation as additional loss terms and scheduling strategies are added to Phase 3. From left to right, each setting introduces a new component: Pattern-Weighted Dice Loss (L<sub>����</sub>) only, L<sub>����</sub> + Absence Penalty $( { \mathcal { L } } _ { A P } ) ,$ and finally L<sub>����</sub> + L<sub>��</sub> + �-Scheduling. This results in a gradual increase in IoU across all defect patterns.

Pattern-Weighted Dice Loss Only serves as the baseline. While it handles class imbalance, it shows limited performance on sparse patterns (e.g., Scratch, EdgeRing), which remain underrepresented. Adding the Absence Penalty term provides negative supervision for pattern-absent regions, reducing false positives. This yields noticeable IoU gains on previously over-predicted patterns (e.g., EdgeRing). Finally, �-Scheduling further improves the model by suppressing all false-positive pixels during training. As training progresses, decreasing � makes the model more conservative, reducing false positives and improving IoU for nearly all patterns. Overall, the figure demonstrates the cumulative benefit of the three phases, showing the best performance when used altogether.

## 6 Conclusion

In this work, we present SePArate, a pioneering weakly supervised framework that advances automatic semiconductor defect inspection from image-level classification to pattern-level segmentation. Unlike conventional approaches requiring costly pixel-level annotations, our method uses only image-level labels to separate defect instances, while remaining scalable and cost-efective. Experiments on defect datasets show that SePArate consistently outperforms baselines. By providing a single type of pattern-level feedback for defects, our framework can enhance optical inspection, long-term monitoring, and continuous manufacturing improvement.

## Acknowledgments

This paper was the result of the research project supported by SK Hynix Inc (MDUS001T). Jinho Lee is the corresponding author.

## References

[1] Zhaozheng Chen, Tan Wang, Xiongwei Wu, Xian-Sheng Hua, Hanwang Zhang, and Qianru Sun. 2022. Class re-activation maps for weakly-supervised semantic segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 969–978.

[2] Mehret Getachew, Birhanu Beshah, and Ameha Mulugeta. 2024. Data analytics in zero defect manufacturing: a systematic literature review and proposed framework. International Journal ofProduction Research (2024), 1–33.

[3] Jacob Gildenblat and contributors. 2021. PyTorch library for CAM methods. https://github.com/jacobgil/pytorch-grad-cam.

[4] Shaun S Gleason, Kenneth W Tobin Jr, Thomas P Karnowski, and Fred Lakhani. 1998. Rapid yield learning through optical defect and electrical test analysis. In Metrology, Inspection, and Process Controlfor Microlithography XII, Vol. 3332. SPIE, 232–242.

[5] Hongquan He, Guowen Kuang, Qi Sun, and Hao Geng. 2024. PaLM: Point Cloud and Large Pre-trained Model Catch Mixed-type Wafer Defect Pattern Recognition. In 2024 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 1–6.

[6] Tongwha Kim and Kamran Behdinan. 2023. Advances in machine learning and deep learning applications towards wafer map defect recognition and classifica tion: a review. Journal of Intelligent Manufacturing 34, 8 (2023), 3215–3247.

[7] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings ofthe IEEE/CVF international conference on computer vision. 4015–4026.

[8] Kibong Koo, Sunghwa Ok, Yonggu Kang, Seungbong Kim, Choungki Song, Hyey oung Lee, Hyungsoo Kim, Yongmi Kim, Jeonghun Lee, Seunghan Oak, Yosep Lee, Jungyu Lee, Joongho Lee, Hyungyu Lee, Jaemin Jang, Jongho Jung, Byeongchan Choi, Yongju Kim, Youngdo Hur, Yunsaing Kim, Byongtae Chung, and Yongtak Kim. 2012. A 1.2V 38nm 2.4Gb/s/pin 2Gb DDR4 SDRAM with bank group and ×4 half-page architecture. In IEEE International Solid-State Circuits Conference.

[9] Hyeokjun Kweon and Kuk-Jin Yoon. 2024. From sam to cams: Exploring segment anything model for weakly supervised semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19499– 19509.

[10] Kiryong Kyeong and Heeyoung Kim. 2018. Classification of mixed-type defect patterns in wafer bin maps using convolutional neural networks. IEEE Transactions on Semiconductor Manufacturing 31, 3 (2018), 395–402.

[11] Timo Lüddecke and Alexander Ecker. 2022. Image segmentation using text and image prompts. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 7086–7096.

[12] Anne Meixner. [n. d.]. Too Much Fab And Test Data, Low Utilization. https: //semiengineering.com/too-much-fab-and-test-data-low-utilization/.

[13] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. 2016. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV). Ieee, 565–571.

[14] Subhrajit Nag, Dhruv Makwana, Sparsh Mittal, C Krishna Mohan, et al. 2022. WaferSegClassNet-A light-weight network for classification and segmentation of semiconductor wafer defects. Computers in Industry 142 (2022), 103720.

[15] Takeshi Nakazawa and Deepak V Kulkarni. 2019. Anomaly detection and segmentation for wafer defect patterns using deep convolutional encoder–decoder neural network architectures in semiconductor manufacturing. IEEE Transactions on Semiconductor Manufacturing 32, 2 (2019), 250–256.

[16] Mu Nie, Shidong Zhu, Aibin Yan, Cheng Zhuo, Xiaoqing Wen, and Tianming Ni. 2025. Eficient Modulated State Space Model for Mixed-Type Wafer Defect Pattern Recognition. In 2025 Design, Automation & Test in Europe Conference (DATE). IEEE, 1–6.

[17] Upasana Pandya. [n. d.]. Modernizing Semiconductor Manufacturing Analytics Platform using AWS Data Analytics Services. https://designthesolution.org/wpcontent/uploads/2024/10/process-track-upasana-modernizing-semiconductormanufacturing-analytics-aws-data-analytics-services.pdf.

[18] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

[19] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. 2024. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159 (2024).

[20] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention. Springer, 234–241.

[21] Muhammad Saqlain, Qasim Abbas, and Jong Yun Lee. 2020. A deep convolu tional neural network for wafer defect identification on an imbalanced dataset in semiconductor manufacturing processes. IEEE Transactions on Semiconductor Manufacturing 33, 3 (2020), 436–444.

[22] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedan tam, Devi Parikh, and Dhruv Batra. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings ofthe IEEE international conference on computer vision. 618–626.

[23] NG Shankar and ZW Zhong. 2005. Defect detection on semiconductor wafer surfaces. Microelectronic engineering 77, 3-4 (2005), 337–346.

[24] Ho Sun Shon, Erdenebileg Batbaatar, Wan-Sup Cho, and Seong Gon Choi. 2021. Unsupervised pre-training of imbalanced data for identification of wafer map defect patterns. IEEE Access 9 (2021), 52352–52363.

[25] Junliang Wang, Chuqiao Xu, Zhengliang Yang, Jie Zhang, and Xiaoou Li. 2020. Deformable convolutional networks for eficient mixed-type wafer defect pattern recognition. IEEE Transactions on Semiconductor Manufacturing 33, 4 (2020), 587–596.

[26] Lian Xu, Wanli Ouyang, Mohammed Bennamoun, Farid Boussaid, and Dan Xu. 2022. Multi-class token transformer for weakly supervised semantic segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 4310–4319.

[27] Jinda Yan, Yi Sheng, and Minghao Piao. 2023. Semantic segmentation-based wafer map mixed-type defect pattern recognition. IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems 42, 11 (2023), 4065–4074.

[28] Jianbo Yu. 2019. Enhanced stacked denoising autoencoder-based feature learning for recognition of wafer map defects. IEEE Transactions on Semiconductor Manufacturing 32, 4 (2019), 613–624.