# InfantNet: A Large Scale Dataset for Infant Body Pose and Shape Estimation

Zhipeng Hui<sup>1,\*</sup> Jiahao Wang<sup>1,\*</sup> Xun Dong<sup>1,\*</sup> Guofeng Zhang<sup>1</sup> Xingrui Wang<sup>1</sup> Jiawei Peng<sup>1</sup> Qihao Liu<sup>1</sup> Xiaoding Yuan<sup>1</sup> Yi Zhang<sup>2</sup> Junjie Oscar Yin<sup>3</sup> Wufei Ma<sup>1</sup> Qinfan Lyu<sup>1</sup> Sara N. Lupolt<sup>1</sup> Keeve Edward Nachman<sup>1</sup> Adam Kortylewski<sup>4</sup> Alan Yuille<sup>1</sup>

<sup>1</sup>Johns Hopkins University <sup>2</sup>Sony Corporation of America <sup>3</sup>University of Washington <sup>4</sup>CISPA Helmholtz Center for Information Security <sup>\*</sup>Equal contribution

![](images/2091eac31a391f9f9a90e6105b8c6599c2f8129130dfe31b799943f0fe137144.jpg)

Figure 1. Examples of SMIL meshes from InfantNet. The dataset captures diverse infant morphology, articulation, and real-world condi tions, providing reliable large-scale 3D supervision for training and testing infant-specific models.

## Abstract

Infant pose and shape estimation is essential for applications in childcare, developmental monitoring, and medical diagnosis. However, existing methods and datasets are largely designed for adults, and direct transfer to infants fails due to substantial differences in body proportions, articulation limits, and frequent self-occlusion. To address this gap, we introduce InfantNet, the largest real-image infant dataset to date, comprising 108,902 RGB images of infants aged 6-18 months. Each image is annotated with 2D keypoints, and a curated subset of 11,642 images additionally includes 3D pose and shape annotations with full SMIL parameters, verified through multi-view rendering and reprojection-errorfiltering. We use an iterative annotation pipeline to ensure high fidelity across both 2D and 3D labels. InfantNet establishes a large-scale, comprehensive

benchmark for infant 2D keypoint detection and 3D poseand-shape recovery. Baseline experiments demonstrate that state-of-the-art adult pose estimators do not generalize well to infants, whereasfine-tuning on InfantNet yields a consistent improvement. The gains are even more pronouncedfor 3D pose and shape estimation, where SMIL-based estimation demonstrates robust performance with accuracy comparable to results reported for SMPL-based models. By releasing the InfantNet dataset and benchmark, we provide a vital resource for advancing infant pose analysis and related healthcare applications.

## 1. Introduction

Estimating the pose and shape of the human body is a foundational task in computer vision, with broad applications in action recognition, human-computer interaction, and healthcare [22, 26, 32, 33, 37]. Over the past decades, human pose and shape estimation algorithms have achieved remarkable progress. However, existing methods and datasets predominantly focus on adults, whose body morphology and poses differ significantly from those of infants. Directly applying adult-trained models to infants results in poor performance due to substantial domain gaps, including differences in body proportions, limited articulation, and frequent self-occlusion typical of infant behavior. These limitations significantly hinder the deployment of such models in infant-related applications, despite their critical importance in areas such as developmental monitoring, childcare, and early medical diagnosis. For instance, quantifying infant micro-activities [32] requires fine-grained 3D pose analysis. However, in the absence of accurate infantspecific models, researchers [33] currently rely on keypoint detectors trained primarily on adult data combined with multi-view triangulation as a noisy proxy, which are complicated to deploy in real-world home environments.

A major obstacle to advancing infant-specific pose estimation is the lack of large-scale, high-quality annotated datasets. Existing infant datasets typically contain only a few thousand images [12, 17, 19], in contrast to adult datasets such as 3DPW [42] and Human3.6M [20], which comprise over 51,000 and 3.6 million images, respectively. Moreover, current infant datasets are often synthetic or focus mainly on sparse 2D annotations [12, 17, 19], lacking the 3D supervision needed to train and evaluate modern human modeling methods [15, 44].

To address this gap, we introduce InfantNet, a largescale dataset specifically designed for infant pose and shape estimation. Comprising more than 100K high-resolution RGB images of infants mainly aged 6 to 18 months, Infant-Net includes accurate annotations of 17 anatomically consistent 2D keypoints. Frames are sampled from videos at up to 2 FPS to reduce temporal redundancy while preserving pose diversity (Fig. 4). A carefully curated subset of 11,642 images further includes 3D pose and shape annotations by SMIL [18] parameters, enabling future research for 3D infant shape and pose estimation. We adopt a robust iterative annotation pipeline that combines human supervision with optimization-based refinement to ensure high-quality labels in 2D. We further validate 3D SMIL [18] parameters by predicting the resulting meshes from multiple viewpoints and computing reprojection errors. Examples of images and 3D pose and shape annotations are shown in Figure 1.

InfantNet is designed with a “privacy-first” data collection framework. The HomeVisit subset was collected under institutional review board (IRB) approval with written informed caregiver consent and digital face-masking. For the YouTube subset, we strictly adhere to a nonredistribution policy, releasing only original URLs and anonymized derivatives (keypoints, SMIL parameters). To prevent misuse, the dataset is restricted to research via controlled academic licensing, a signed Data Use Agreement (DUA) prohibiting biometric identification, and a robust takedown mechanism. (See Sec. 3.1 for details.)

Our benchmark shows that state-of-the-art adult pose estimators generalize poorly to infants. Fine-tuning on InfantNet substantially reduces this gap, improving 2D performance by 6.2% AP and reducing 3D MPJPE by 78%. Despite training on only 10,071 frames, our infant-specific 4D-Humans model achieves 46.7 mm PA-MPJPE on Infant-Net, approaching the adult result of 32.4 mm [15] trained and tested on the Human3.6M dataset, and surpasses HMR (56.8 mm) and PARE (50.6 mm). These results highlight the strong data efficiency of InfantNet, enabling highfidelity 3D modeling with 350× fewer training samples than typical adult datasets. Our model also enables significantly more accurate detection of infant hand-to-mouth behavior [33], supporting reliable assessment of environmental exposure risks in real-world settings.

In summary, the main contributions of our paper are:

• We introduce InfantNet, a large-scale dataset for infant pose and shape estimation, containing 108,902 RGB images with high-quality 2D keypoint annotations. A curated subset of 11,642 images further includes 3D pose and shape labels based on SMIL. The dataset enables comprehensive evaluation of both 2D and 3D pose and shape estimation models.

• We develop an end-to-end SMIL-based framework for infant 3D pose and shape estimation. To our knowledge, this is the first attempt to directly regress SMIL parameters from images with neural networks. By generating high-quality pseudo ground-truth through optimization and verification, we train end-to-end models to directly regress SMIL parameters from images, enabling accurate infant-specific 3D modeling.

• We systematically analyze the domain gap between adult and infant pose estimation. Fine-tuning on InfantNet substantially improves both 2D and 3D performance, demonstrating the importance of domain-specific data. Our models achieve competitive accuracy with far fewer training samples and support downstream applications such as infant microactivity analysis.

• We set up a benchmark on InfantNet using state-of-the-art human pose and shape estimation algorithms.

By releasing data, code, and evaluation protocols, we hope to provide a foundation for future research in infant pose and shape estimation, with implications for early developmental assessment, clinical monitoring, and beyond.

## 2. Related Work

In this section, we discuss existing infant datasets and related pose and shape estimation algorithms.

<table><tr><td>Dataset</td><td>Syn/Real</td><td># Images</td><td>Keypoints</td><td>Segmentation</td><td>3D Annotations</td></tr><tr><td>BabyPose [35]</td><td>Real</td><td>16,000</td><td>Yes</td><td>No</td><td>Depth only</td></tr><tr><td>BabyNet [12]</td><td>Real</td><td>2,984</td><td>No</td><td>BBox only</td><td>No</td></tr><tr><td>MINI-RGBD [17]</td><td>Synthetic</td><td>12,000</td><td>Yes</td><td>No</td><td>Depth only</td></tr><tr><td>SyRIP [19]</td><td>Syn+Real</td><td>1,700</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>InfantNet (ours)</td><td>Real</td><td>108,902</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

Table 1. Comparison of InfantNet with existing open-source infant pose datasets. InfantNet is the largest real-world infant dataset to date, providing 108,902 images with keypoint annotations, full segmentation masks, and a curated subset with 3D SMIL parameters. In contrast, prior datasets are limited in scale, modality, or 3D supervision (e.g., MINI-RGBD provides depth only).

## 2.1. Infant Datasets

Human pose and shape estimation has benefited enormously from large-scale adult datasets such as Human3.6M [20], MPII [1], and AMASS [34]. In contrast, infant datasets remain scarce due to ethical, logistical, and technical challenges in capturing infant motion in naturalistic environments. Early efforts include BabyPose [35], which provides 16 depth videos from NICU settings with 12-joint annotations, and the dataset from [16], focusing on supine infants with limited poses. BabyNet [12] targets activity classification without fine-grained pose labels. MINI-RGBD [17] offers synthetic depth sequences of 12 infant models, while SyRIP [19] combines 1,700 synthetic and real images with sparse 2D keypoints. These datasets are small, often synthetic or modality-specific, and lack segmentation masks or 3D ground truth pose and shape parameters. Our proposed InfantNet addresses these limitations by providing real infant RGB images with high-quality 2D keypoints, segmentation masks, bounding boxes, and 3D pose and shape annotations. Table 1 compares InfantNet with prior datasets, showing its unique scale, diversity, and multimodal supervision, which together enable robust evaluation and training of infant-specific models.

## 2.2. Infant Pose and Shape Estimation

2D infant pose estimation initially applied general-purpose models such as OpenPose [7], HRNet [41], and Alpha-Pose [14], but these perform poorly due to infants’ distinct proportions, articulation limits, and frequent selfocclusion [39]. Later work fine-tuned these CNN-based estimators on small infant datasets [16], reducing error and approaching human inter-rater variability. SHIFT [4] adapts adult data to unlabeled infant data, reflecting the scarcity of labeled infant-specific datasets. More recent transformerbased methods, such as ViTPose [44], AggPose [6], and Sapiens [23], improve spatial reasoning in cluttered settings [21], but are susceptible to overfitting without large and diverse infant-specific datasets.

In the 3D domain, early infant methods relied on depth sensors for direct joint prediction [17] or stereo systems for triangulation [40], achieving good accuracy in clinical settings but requiring specialized hardware. The SMIL model [18] represents a key step toward parametric modeling of infants: adapting the adult SMPL template [30] with an infant-specific mesh and priors learned from noisy, incomplete RGB-D recordings. While SMIL enables fullbody mesh registration, its reliance on limited and lowquality depth sequences constrains generalization. ZeDOi [48] adapts adult priors with limited infant data, focusing on keypoint estimation. Other approaches leverage synthetic renderings [19] or alternative sensors such as pressure mats [9, 13], but these are less practical in daily settings.

Despite this progress, the field lacks a large-scale, realworld RGB benchmark with consistent 2D and 3D annotations. Our InfantNet fills this gap, enabling systematic development and evaluation of infant pose and shape estimation models that generalize across naturalistic scenarios.

## 2.3. Adult pose and shape estimation

3D human pose and shape estimation has been extensively studied through parametric models such as SMPL [30] and its extensions (e.g. SMPL-X [37]). Learning-based approaches including HMR [22], VIBE [24], PARE [25], and 4D-Humans [15] use feedforward regression to directly predict body parameters from images or videos, often with strong temporal consistency. However, these methods are trained on adult datasets and exhibit poor generalization to infants, whose body proportions, articulation limits, and frequent occlusions differ significantly. In this work, we adapt such adult regression networks to the infant domain using InfantNet and SMIL-based annotations.

## 3. InfantNet Dataset

In this section, we present the InfantNet dataset and detail the data collection and annotation procedures.

## 3.1. Ethical Design and Data Governance

The collection, processing, annotation, and release of InfantNet follow the IEEE Code of Ethics, with a multi-layer ethical protocol to ensure privacy.

IRB and Consent. The HomeVisit subset was collected under an approved Institutional Review Board (IRB) protocol. We obtained written informed consent from all caregivers, explicitly covering anonymized data sharing for research. Privacy and Anonymization. To protect identities, we apply digital face-masking to all YouTube and HomeVisit frames and strip all identifying metadata. For the YouTube subset, we strictly follow a non-redistribution policy: we release only anonymized, derived data (keypoints, SMIL parameters) and original URLs. Figure 1 shows our facemasking strategy, and more examples can be found in Figure S1 of Appendix A.3. We also provide experimental results that demonstrate the robustness of our model and viability of our dataset under this strategy in Appendix A.7.

Controlled Access. InfantNet will be released on Hugging Face with gated and manually approved access under CC BY-NC-ND 4.0. Access requires institutional verification and agreement to a Data Use Agreement prohibiting commercial use, biometric identification, and redistribution of modified derivatives. A dedicated takedown mechanism is maintained for ongoing compliance.

Societal Impact. InfantNet addresses the lack of largescale resources for infant healthcare research. Our goal is to enable better care for infants while maintaining the highest standards of participant privacy. Specific notes on data and code publications can be found in Appendix A.8.

## 3.2. Data Collection

YouTube data collection. We curated publicly available infant videos from YouTube, capturing infants in a wide range of poses and body shapes. These raw videos serve as one source of our dataset and are later processed through annotation and quality filtering. The YouTube data is collected strictly following the platform policy and privacy safeguards, as detailed in Sec. 3.1.

HomeVisit Data Collection. Following written informed consent from all caregivers, we recorded multi-view videos of infants in their domestic play areas. The setup used four statically mounted GoPro Hero9 cameras positioned orthogonally for overlap coverage. To support accurate SMIL fitting, we performed rigorous per-session camera calibration. For HomeVisit, the female-to-male video ratio is 47%:53%. We do not infer race, ethnicity, geographic origin, or precise age when such metadata is unavailable.

## 3.3. Data Annotation

We annotated frames from both sources, filtering to ensure exactly one infant per frame and sampling at a maximum rate of 2 FPS to preserve temporal diversity. In total, 206,436 frames were extracted prior to quality filtering. The YouTube subset contains 51,039 frames. These videos feature around 25 distinct infants per 4 minutes, yielding an estimated total of around 3,000 unique infants. The HomeVisit subset consists of 15 multi-camera recording sessions (each with four synchronized cameras), from which 155,397 frames were extracted. After frame extraction, we performed 2D body keypoint annotation followed by multi-stage quality control, removing frames with severe occlusion, motion blur, or other degradations. The resulting high-quality annotations serve as ground-truth supervision for SMIL fitting and subsequent model training.

To annotate such a large and heterogeneous dataset, we adopted a semi-automatic, inspector-guided annotation pipeline, inspired by human-in-the-loop strategies in 3D model fitting works such as Animal3D [43]. As shown in Figure 2, our annotation workflow consisted of six stages to ensure efficiency and accuracy.

Initial keypoint estimation. We employed RTMO-L [31] from MMPose [11] to produce 2D pseudo-labels for all frames. These predictions were loaded into CVAT [10], where human annotators conducted multi-step refinement.

Initial filtering and manual correction. Annotators reviewed the RTMO-L [31] predictions in CVAT and removed frames with severe motion blur, camera shake, extreme lighting artifacts (e.g., overexposure or harsh shadows), strong occlusion (e.g., when the infant is mostly outside the field of view or heavily covered by objects), missing infants, or subjects older than 18 months. Adult skeletons were discarded, and one infant per frame was retained. For remaining frames, if the RTMO-L prediction provided accurate and anatomically plausible keypoints, the annotation was accepted without modification to maximize efficiency. Otherwise, annotators manually corrected the predictions to ensure anatomical consistency. All annotations followed the COCO format with 17 joints: nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, and right ankle. Each keypoint was tagged as visible, partially occluded, orfully occluded.

Inspector verification. An inspector reviewed annotations for consistency and quality. Frames with annotation errors were returned to annotators with revision feedback.

Iterative refinement. This correction–verification cycle was repeated until the inspector approved all annotations per recording session, ensuring high-quality labels.

Bounding box generation. Once the keypoints were finalized, bounding boxes were generated using SAM2 [38]. Visible keypoints were used to prompt mask generation, and the final bounding boxes were computed as the minimal rectangles covering both the masks and any occluded joints, thereby mitigating segmentation errors due to occlusion (e.g., infants partially covered by furniture).

Final bounding box validation. Inspectors performed a second review to verify bounding box accuracy and tightness, ensuring annotation completeness and quality.

This staged workflow allowed us to balance scalability and precision, producing high-quality 2D annotations with explicit human oversight at each critical step. Examples of the 2D annotations in different stages are shown in Figure 3.

## 3.4. Fitting SMIL on Infant Images

We use an optimization-based method [3, 45] to fit SMIL [18] on infant images from our dataset following the FiDIP [19] workflow. We optimize the shape parameters $\beta$ and pose parameters θ in SMIL by minimizing the loss:

$$
\mathcal { L } _ { \mathrm { S M I L } } = \mathcal { L } _ { \mathrm { d a t a } } + \mathcal { L } _ { \mathrm { p o s e } } + \mathcal { L } _ { \mathrm { s h a p e } } + \mathcal { L } _ { \mathrm { o t h e r s } } .\tag{1}
$$

The data term, ${ \mathcal { L } } _ { \mathrm { d a t a } } .$ , includes mesh-to-scan ${ \mathcal L } _ { \mathrm { m 2 s } }$ and scan-to-mesh $\mathcal { L } _ { \mathrm { s } 2 \mathrm { m } }$ distances. $\mathcal { L } _ { \mathrm { p o s e } }$ and $\mathcal { L } _ { \mathrm { s h a p e } }$ enforce priors on θ and $\beta ,$ , respectively. $\mathcal { L } _ { \mathrm { o t h e r s } }$ contains losses for landmark keypoints alignment $( { \mathcal { L } } _ { \mathrm { l m } } )$ , table contact $( \mathcal { L } _ { \mathrm { t a b l e } } )$ , temporal smoothness $( \mathcal { L } _ { \mathrm { s m } } )$ , and self-intersections $( \mathcal { L } _ { \mathrm { s c } } ) .$

![](images/6aecb6fc6a7528a6f318cce3dc5437bac2500e290990d3a18256bbcec04c759f.jpg)  
Figure 2. Annotation pipeline of our InfantNet Dataset. Frames from YouTube and HomeVisit recordings were annotated using a semi automatic, inspector-guided workflow. For 2D labels, RTMO-L provided initial keypoints that were refined by annotators with iterative quality control, following the COCO format. For 3D labels, we applied an optimization-based method to fit SMIL to infant images, and annotators further filtered the results to ensure accurate pose and shape annotations.

![](images/20d010e3c70355e64d4155aff5b0c05d2a6cf2118c02e125f1782f05469f6bc3.jpg)  
Figure 3. Visualization of different annotation stages in our Infant-Net Dataset: Original image, Segmentation mask, Bounding box, and Annotated Body Keypoints.

The registration proceeds in three steps: estimating initial shape, solving per-frame poses with smoothness and data terms, and registering meshes via constrained deformation. We validate SMIL parameters with multi-view rendering and reprojection checks, removing results inconsistent with the image, distorted in any view, or showing implausible poses or shapes. We also project 3D keypoints to 2D and discard frames with reprojection error above 16 pixels. The filtered parameters provide high-quality 3D groundtruth for end-to-end training. Further 3D annotation quality analysis is provided in Sec. 3.5.

## 3.5. Data Summary

We partition the dataset into training and testing splits while maintaining identity consistency across videos and sessions. Table 2 reports the statistics of raw frame counts and the final numbers after annotation-stage filtering.

<table><tr><td>Subset</td><td>Viewpoint Info</td><td>Raw Frames</td><td>Filtered Frames</td></tr><tr><td>YouTube (Train)</td><td>Varied</td><td>44,257</td><td>36,185</td></tr><tr><td>YouTube (Test)</td><td>Varied</td><td>6,782</td><td>3,959</td></tr><tr><td>HomeVisit (Train)</td><td>4 fixed cameras</td><td>123,876</td><td>56,968</td></tr><tr><td>HomeVisit (Test)</td><td>4 fixed cameras</td><td>31,521</td><td>11,790</td></tr><tr><td>Total</td><td>一</td><td>206,436</td><td>108,902</td></tr></table>

Table 2. Dataset Statistics. “Filtered Frames” denotes the final samples retained after removing low-quality, ambiguous, or noncompliant frames during annotation.

YouTube Subset. The YouTube subset consists of 40,144 frames with diverse environments and camera conditions. 3,959 frames were designated for testing, while 36,185 frames were assigned to training, ensuring an identitydisjoint split with no overlap of infants between sets.

HomeVisit Subset. The HomeVisit subset contains 15 in-home recording sessions, producing a total of 155,397 frames. For evaluation, we selected 3 sessions as the testing set, while the remaining 12 sessions were assigned to training. After annotation-stage filtering, the testing set contains 11,790 frames and the training set contains 56,968 frames.

Format. All annotations follow the standard COCO [28] keypoints format and include 17 body keypoints with visibility flags (visible, partially occluded, fully occluded), as well as bounding boxes. The dataset’s multi-source and multi-view composition enables both robust training and principled benchmarking in this challenging domain.

Pose Diversity. We measure pose diversity as the average standard deviation of pelvis-centered 3D joint coordinates across the dataset, capturing spatial variability of articulated body parts, including joint distribution, pose spread, and limb movement range [27]. The 3D annotations achieve a pose diversity score of 115.0, exceeding several widely used large-scale human datasets (Fig. 4) and demonstrating substantial structural variability despite focusing on infants. Prior to multi-view rendering and reprojection-based filtering (Sec. 3.4), the pose diversity score is 138.2. The postfiltering diversity remains at a high magnitude, confirming that the filtering process preserves pose variability and does not bias the dataset toward specific poses or shapes.

Occlusion. 75.58% of frames have fully occluded keypoints, and 92.24% have fully/partially occluded keypoints. We further categorize occluded keypoints as outof-frame, self-occluded by the infant body, or externally occluded by other objects. Among occluded keypoints, 5.5% are out of frame, 64.5% are self-occluded, and 30.0% are externally occluded. Additional evaluation and analysis of occlusion and edge cases are provided in Appendix A.4. 3D Annotation Quality. We

![](images/753b473f79744c73ba09969cb01f7f7b1676aff04c9f8961818aa3fd07f7ae86.jpg)  
Figure 4. Pose diversity statistics measured by per-joint standard deviation between Infant-Net and large-scale human datasets [5, 8, 27, 29, 34]. InfantNet exhibits broad variation in 3D joint configurations.

validate the 3D annotations from consistency and reliability. We assess consistency using multi-view rendering, reprojection error, and bbox-normalized PCK. The 3D annotations achieve a mean reprojection error of 9.10 px and 83.04%/98.40%/99.99% PCK@0.05/0.1/0.2. We evaluate reliability using multi-view joint optimization on calibrated HomeVisit frames. We jointly optimize each body annotation across views using multi-view reprojection error and SMIL priors. The optimized annotations differ from initial annotations by 80.6 mm MPJPE, while independently optimized candidates agree closely with only 3.1 mm mean pairwise MPJPE, supporting 3D annotation reliability.

## 4. Experiments

We benchmark state-of-the-art 2D and 3D pose estimation models on InfantNet to quantify the domain gap between adult-centric models and infant morphology. Secs. 4.1 and 4.2 evaluate 2D transferability and 3D recovery, while Sec. 4.3 shows InfantNet’s value in real-world applications.

## 4.1. 2D Keypoint Estimation

We benchmark a series of widely used 2D human pose estimation models on our proposed infant dataset to evaluate their ability to transfer from adult-centric data to the unique morphology and movement patterns of infants.

<table><tr><td rowspan="2">Model</td><td colspan="4">Direct Inference (%)</td><td colspan="4">Fine-tuned (%)</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td></tr><tr><td>Hourglass 256x256</td><td>86.27</td><td>95.58</td><td>92.37</td><td>88.43</td><td>92.63</td><td>98.99</td><td>97.90</td><td>93.95</td></tr><tr><td>Hourglass 384x384</td><td>87.39</td><td>96.59</td><td>93.43</td><td>89.27</td><td>93.40</td><td>98.98</td><td>97.89</td><td>94.52</td></tr><tr><td>HRFormer-small 256x192</td><td>87.65</td><td>96.53</td><td>93.36</td><td>89.82</td><td>93.14 98.97</td><td></td><td>97.87</td><td>94.36</td></tr><tr><td>HRFormer-small 384x288</td><td>88.00</td><td>96.54</td><td>94.30</td><td>90.07</td><td>93.51</td><td>98.97</td><td>97.91</td><td>94.60</td></tr><tr><td>HRFormer-base 256x192</td><td>88.24</td><td>95.50</td><td>93.33</td><td>90.26</td><td>93.85</td><td>98.98</td><td>97.92</td><td>95.09</td></tr><tr><td>HRFormer-base 384x288</td><td>88.71 96.38</td><td></td><td>94.16</td><td>90.82</td><td>94.01</td><td>98.98</td><td>97.93</td><td>95.27</td></tr><tr><td>Res50 256x192</td><td>86.48</td><td>96.62</td><td>93.37</td><td>88.59</td><td>92.73</td><td>99.00</td><td>97.91</td><td>94.04</td></tr><tr><td>Res50 384x288</td><td>86.57</td><td>96.58</td><td>93.31</td><td>88.63</td><td>93.03</td><td>99.00</td><td>97.93</td><td>94.16</td></tr><tr><td>Res101 256x192</td><td>87.28</td><td>96.56</td><td>93.44</td><td>89.29</td><td>93.18</td><td>98.99</td><td>97.90</td><td>94.37</td></tr><tr><td>Res101 384x288</td><td>87.59</td><td>96.50</td><td>93.38</td><td>89.57</td><td>93.51</td><td>98.99</td><td>97.92</td><td>94.78</td></tr><tr><td>Res152 256x192</td><td>87.31</td><td>95.56</td><td>93.41</td><td>89.38</td><td>93.31</td><td>98.98</td><td>97.92</td><td>94.57</td></tr><tr><td>Res152 384x288</td><td>87.89</td><td>96.46</td><td>93.33</td><td>89.99</td><td>93.87</td><td>98.95</td><td>97.89</td><td>94.90</td></tr><tr><td>HRNet-w32-dark 256x192</td><td>88.60</td><td>96.61</td><td>93.48</td><td>90.37</td><td>93.82</td><td>98.98</td><td>97.92</td><td>94.89</td></tr><tr><td>HRNet-w32-dark 384x288</td><td>88.74</td><td>96.50</td><td>94.29</td><td>90.64</td><td>93.98</td><td>98.97</td><td>97.92</td><td>95.19</td></tr><tr><td>HRNet-w48-dark 256x192</td><td>88.92 96.56</td><td></td><td>94.40</td><td>90.69</td><td>94.16</td><td>98.96</td><td>97.91</td><td>95.28</td></tr><tr><td>HRNet-w48-dark 384x288</td><td>88.74</td><td>96.51</td><td>94.32</td><td>90.60</td><td>94.21</td><td>98.98</td><td>97.91</td><td>95.34</td></tr><tr><td>HRNet-w32 256x192</td><td>88.04 96.60</td><td></td><td>93.45</td><td>89.95</td><td>93.36</td><td>98.98</td><td>97.88</td><td>94.50</td></tr><tr><td>HRNet-w32 384x288</td><td>88.19 96.53</td><td></td><td>93.39</td><td>90.24</td><td></td><td>93.64 98.96</td><td>97.88</td><td>94.82</td></tr><tr><td>HRNet-w48 256x192</td><td>88.38</td><td>96.57</td><td>93.41</td><td>90.24</td><td></td><td>93.74 98.98</td><td>97.92</td><td>94.85</td></tr><tr><td>HRNet-w48 384x288</td><td>88.37</td><td>96.50</td><td>93.37</td><td>90.39</td><td>93.97</td><td>98.97</td><td>97.90</td><td>95.12</td></tr><tr><td>ViTPose-small 256x192</td><td>87.56 96.56</td><td></td><td>93.47</td><td>89.46</td><td>93.36</td><td>98.99</td><td>97.93</td><td>94.43</td></tr><tr><td>ViTPose-base 256x192</td><td>89.02</td><td>96.54</td><td>94.48</td><td>90.89</td><td>93.91</td><td>98.97</td><td>97.91</td><td>95.11</td></tr><tr><td>ViTPose-large 256x192</td><td>90.02</td><td>96.44</td><td>94.34</td><td>91.99</td><td>94.75</td><td>98.96</td><td>97.91</td><td>95.81</td></tr><tr><td>ViTPose-huge 256x192</td><td>89.95</td><td>96.39</td><td>94.30</td><td>91.99</td><td>94.77</td><td>98.95</td><td>97.90</td><td>95.82</td></tr><tr><td>Sapiens-0.3B</td><td>90.08</td><td>97.60</td><td>95.43</td><td>91.96</td><td>93.35</td><td>98.97</td><td>97.90</td><td>94.48</td></tr></table>

Table 3. Comparison of 2D pose estimation models on the InfantNet benchmark under two settings: direct inference using pretrained models and fine-tuning on InfantNet. We report Average Precision (AP), AP at IoU thresholds 0.5 (AP.5) and 0.75 (AP.75), and Average Recall (AR). Direct transfer of pretrained models leads to a clear performance drop, while fine-tuning consistently improves performance, with an average relative gain of 6.2% in AP and 5.2% in AR. ViTPose-huge achieves the best results after adaptation (AP: 94.77%, AR: 95.82%), highlighting the importance of domain-specific data for accurate infant pose estimation.

Setup. We benchmark Transformer-based models (ViT-Pose [44], HRFormer [46], Sapiens [23]), CNN-based models (HRNet [41], Hourglass [36]), and CNN variants with DARK post-processing (HRNet-Dark [47]). All models are initialized from COCO-pretrained weights. We compare models with and without fine-tuning on our InfantNet train split $( \mathrm { Y o u T u b e } _ { \mathrm { T r a i n } }$ ∪ Home $\mathrm { V i s i t _ { T r a i n } ) }$ for 40 epochs following the MMPose setting, with a learning rate of $1 \times 1 0 ^ { - 4 }$ on 4 NVIDIA A100-SXM4-40GB GPUs.

Evaluation. We select the best-performing checkpoint on the validation set and report its corresponding results. The split preserves identity separation across sessions. Metrics follow COCO’s 17-keypoint protocol: mean Average Precision (AP), AP.5, AP.75, and Average Recall (AR).

Results. We benchmark a wide range of 2D pose estimation models on InfantNet under two settings: (1) direct inference using COCO-pretrained weights, and (2) fine-tuning the models on our InfantNet training set. Table 3 summarizes the Average Precision (AP), Average Recall (AR), and their variants across 25 different configurations.

(a) Direct Inference Results. Sapiens achieves the highest AP (0.9008), while ViTPose-large and ViTPose-huge achieve the best AR (0.9199). HRNet-W48-Dark 256x192 is also competitive (AP: 0.8892, AR: 0.9069). Overall, transformer-based models, particularly Sapiens and ViT-Pose, show the best direct transfer to InfantNet.

<table><tr><td rowspan="2">Model</td><td colspan="2">Adult Body Model</td><td colspan="2">Infant Body Model</td></tr><tr><td>MPJPE↓</td><td>PA-MPJPE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>HMR [22]</td><td>393.30</td><td>124.15</td><td>103.36 (↓74%)</td><td>58.79 (↓53%)</td></tr><tr><td>PARE [25]</td><td>364.49</td><td>120.06</td><td>83.73 (↓77%)</td><td>63.16 (↓47%)</td></tr><tr><td>4D-Humans [15]</td><td>526.62</td><td>125.02</td><td>84.73 (↓84%)</td><td> $4 6 . 7 4 \ : ( \downarrow 6 3 \% )$ </td></tr></table>

Table 4. Comparison of 3D pose and shape estimation on Infant-Net using adult body models (left) vs. infant body models (right). Metrics: MPJPE and PA-MPJPE (mm). Infant models consistently improve accuracy across all architectures, highlighting the importance of infant data and models.

![](images/cc535f187c5b5f2e3c3c22b4ae41800c114c1e9a8a1427998c416ef8b7e7d284.jpg)  
Figure 5. Failure modes and effects of fine-tuning: Without Infant-Net FT, keypoint estimators frequently fail, e.g., by mislocalizing the left arm under severe self-occlusion (left) or predicting leg incorrectly in atypical infant poses (right). After FT, models can accurately localize joints in these challenging scenarios.

(b) Fine-tuning on InfantNet. Fine-tuning consistently boosts performance across models. On average, it yields a relative gain of 6.2% in AP and 5.2% in AR compared to direct inference. The ViTPose-huge model again achieves the best performance (AP: 0.9477, AR: 0.9582) after finetuning. Overall, models with higher input resolution outperform lower-resolution counterparts, demonstrating the importance of spatial detail for infant pose estimation.

(c) Source-specific and Cross-domain Evaluation. We evaluate model performance separately on the YouTube and HomeVisit subsets. For each source, we measure direct inference using COCO-pretrained weights, fine-tuning only on the InfantNet training portion of that source, and crossdomain generalization, where models are fine-tuned on one source (e.g., HomeVisit) and evaluated on the other (e.g., YouTube). Fine-tuning yields a relative gain of 6.1% in AP and 5.3% in AR on HomeVisit and 5.0% in AP and 3.7% in AR on YouTube. The source-specific and cross-domain comparison table is provided in Appendix Table S1 and S2. Failure Mode Analysis. Models without InfantNet finetuning fail on infant-specific poses (e.g., high flexibility), limb proportions (shorter limbs relative to torso), and selfocclusions (Fig. 5). After FT, models locate joints correctly. Hardest Joints. Joint PCK statistics across body parts (Table 5) show that leg joints are the most challenging. Before fine-tuning, ViTPose-H achieves the lowest PCK on legs (0.793), compared to head (0.962), torso (0.842), and arms (0.865). Fine-tuning on InfantNet improves performance across all regions, including legs (0.793→0.828), yet lower limbs remain challenging. This gap highlights challenges in infant pose estimation, including large articulation, different body proportions from adults, and frequent occlusions.

<table><tr><td>Method</td><td>All</td><td>Head</td><td>Torso</td><td>Arms</td><td>Legs</td></tr><tr><td>ViTPose-H (Before FT)</td><td>0.871</td><td>0.962</td><td>0.842</td><td>0.865</td><td>0.793</td></tr><tr><td>ViTPose-H (FT on InfantNet)</td><td>0.906</td><td>0.986</td><td>0.890</td><td>0.901</td><td>0.828</td></tr></table>

Table 5. PCK (%) breakdown of ViTPose-H before and after finetuning on InfantNet. Fine-tuning improves all body parts, but leg joints remain the most challenging, reflecting the unique articulation and occlusion patterns in infant poses.

## 4.2. 3D Pose and Shape Estimation

We further benchmark representative 3D pose and shape estimation models on our proposed InfantNet dataset to assess their generalizability to infant body models, establishing baselines for 3D infant pose and shape estimation. Notably, a direct comparison with previous infant datasets is not possible, as InfantNet is the first to provide the high-quality 3D ground truth required for such evaluation (see Table 1).

Setup. While no algorithms are explicitly designed for 3D infant pose and shape estimation, adult-based models can be adapted to infants. We benchmark CNN-based HMR [22] and PARE [25], and transformer-based 4D-Humans [15] as baselines. We provide a detailed analysis comparing endto-end regression with SMIL-based optimization in Appendix A.6. We train each model for 100 epochs using a learning rate of $2 . 5 \times 1 0 ^ { - 5 }$ for HMR and PARE on 3 NVIDIA TITAN RTX GPUs, and a learning rate of $2 . 0 \times 1 0 ^ { - 5 }$ for 4D-Humans on 4 NVIDIA A5000 GPUs.

Evaluation. We evaluate models with standard 3D human pose and shape estimation metrics. We report mean per joint position error (MPJPE) in mm, which measures the absolute error between predicted and ground-truth 3D joint positions, and Procrustes-aligned mean per joint position error (PA-MPJPE) in mm, which evaluates relative joint configuration error after rigid alignment.

Model & Data Preparation. We train the models on 10,071 frames and evaluate on 1,571 frames. HMR and 4D-Humans are trained with both 2D and 3D supervision. For PARE, we segment the mean SMIL mesh into seven parts with Blender [2] for part-attention and mask supervision.

Results. We benchmark 3D human pose and shape estimation models on InfantNet under two settings: (1) training and inference with SMIL-adapted weights, and (2) direct inference with pretrained adult-model weights. Fig. 6 presents visualizations of both settings. Table 4 reports MPJPE and PA-MPJPE across six configurations.

(a) Adult Body Model. Direct inference using human pose and shape estimation models yields lower accuracy compared to results obtained with infant-specific body models, affirming the domain gap between adult and infant poses and shapes. Among adult-centric models, PARE achieves the strongest performance (MPJPE: 364.49 mm,

PARE Adult

PARE Infant  
HMR Infant  
HMR Adult  
4D-Humans Infant  
![](images/7c1204c62534592b83042d73040368dc5777c9599446faf79ba3e725dbf18e7d.jpg)  
Figure 6. Visualization of 3D pose and shape estimation. From left to right, the columns show the input image, direct inference with adul body model and data, and training with infant body model and data for HMR [22], PARE [25], and 4D-Humans [15], respectively.

PA-MPJPE: 120.06 mm), while its accuracy remains considerably lower than when trained with infant body models.

(b) Infant Body Model. Adapting adult-oriented models to SMIL substantially improves performance across all architectures. On average, there is a 78% reduction in MPJPE and 54% reduction in PA-MPJPE compared to direct adultmodel inference. We provide cross-subset results in Appendix A.5, demonstrating strong cross-domain generalization. Our infant-specific estimations are almost as good as adult-based estimations (e.g., HMR reports 87.97 mm MPJPE on Human3.6M). Among the models, 4D-Humans achieves the strongest results (MPJPE: 84.73 mm, PA-MPJPE: 46.74 mm), likely due to transformer-based global reasoning. In comparison, keypoint-only ZeDO-i [48] achieves 97.1/77.3 mm. These results show our adapted models provide a better geometric prior for the infant body than adult-oriented models. To ensure privacy, digital facial masking is added. Additional experiments show negligible performance changes (approximately 1-5 mm) with masking, and details are provided in Appendix A.7. These adapted end-to-end models establish a new state-of-the-art for monocular 3D infant pose and shape estimation, providing a solid foundation for applications in developmental monitoring, childcare, and early medical diagnosis.

## 4.3. Real-world Application

To demonstrate InfantNet’s practical impact, we evaluate its performance on detecting hand-to-mouth (HTM) behavior, a critical indicator of environmental exposure (e.g., soil/dust ingestion) [32, 33]. Using 3D estimates from our framework, we identify HTM events via normalized hand-tomouth Euclidean distances. Our infant-specific 4D-Humans model achieves 83% detection accuracy, compared to 30% of the adult-oriented baseline. This performance gap confirms that anatomical fidelity is not merely a refinement but a prerequisite for reliable infant behavior analysis and risk assessment in real-world settings.

## 5. Conclusion

We introduce InfantNet, the first large-scale dataset dedicated to infant pose and shape estimation, addressing the limitations of existing adult-centric models in this domain. Our results show that adult-trained models generalize poorly to infant morphology, while training on InfantNet effectively bridges this gap, enabling accurate 2D and 3D estimation. Beyond benchmarking, our framework provides a reliable foundation for downstream healthcare applications, such as quantifying infant hand-to-mouth behavior for exposure risk assessment. While our current 3D subset provides a rigorous foundation, we aim to scale the dataset significantly by implementing a human-inthe-loop active learning framework. This will allow for the efficient expansion of high-quality 3D labels across more diverse domestic environments and ages. Developed under strict ethical oversight with digital face-masking and controlled access, InfantNet is designed for responsible research. By releasing our data, models, and evaluation tools, we provide a standardized platform to advance infant-focused vision models, and ultimately foster positive impact in both academic research and pediatric care.

## References

[1] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In CVPR, pages 3686–3693, 2014. 3

[2] Blender Foundation. Blender. https://www.blender. org/, 2002. 7

[3] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it smpl: Automatic estimation of 3d human pose and shape from a single image. In ECCV, pages 561–578. Springer, 2016. 4, 5

[4] Sarosij Bose, Hannah Dela Cruz, Arindam Dutta, Elena Kokkoni, Konstantinos Karydis, and Amit Kumar Roy Chowdhury. Leveraging synthetic adult datasets for unsupervised infant pose estimation. In CVPRW, pages 5562–5571, 2025. 3

[5] Zhongang Cai, Daxuan Ren, Ailing Zeng, Zhengyu Lin, Tao Yu, Wenjia Wang, Xiangyu Fan, Yang Gao, Yifan Yu, Liang Pan, Fangzhou Hong, Mingyuan Zhang, Chen Change Loy, Lei Yang, and Ziwei Liu. HuMMan: Multi-modal 4d human dataset for versatile sensing and modeling. In ECCV, pages 557–577. Springer, 2022. 6

[6] Xu Cao, Xiaoye Li, Liya Ma, Yi Huang, Xuan Feng, Zening Chen, Hongwu Zeng, and Jianguo Cao. Aggpose: Deep aggregation vision transformer for infant pose estimation. arXiv preprint arXiv:2205.05277, 2022. 3

[7] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE TPAMI, 43(1): 172–186, 2019. 3

[8] Jihoon Chung, Cheng hsin Wuu, Hsuan ru Yang, Yu-Wing Tai, and Chi-Keung Tang. Haa500: Human-centric atomic action dataset with curated videos. In ICCV, 2021. 6

[9] Henry M Clever, Zackory Erickson, Ariel Kapusta, Greg Turk, Karen Liu, and Charles C Kemp. Bodies at rest: 3d human pose and shape estimation from a pressure image using synthetic data. In CVPR, pages 6215–6224, 2020. 3

[10] CVAT Contributors. CVAT: Computer vision annotation tool. https://github.com/opencv/cvat, 2018. 4

[11] MMPose Contributors. Openmmlab pose estimation toolbox and benchmark. https://github.com/openmmlab/mmpose, 2020. 4

[12] Amel Dechemi, Vikarn Bhakri, Ipsita Sahin, Arjun Modi, Julya Mestas, Pamodya Peiris, Dannya Enriquez Barrundia, Elena Kokkoni, and Konstantinos Karydis. Babynet: A lightweight network for infant reaching action recognition in unconstrained environments to support future pediatric rehabilitation applications. In 2021 30th IEEE International Conference on Robot & Human Interactive Communication (RO-MAN), pages 461–467. IEEE, 2021. 2, 3

[13] Marco Donati, Francesca Cecchi, Filippo Bonaccorso, Marco Branciforte, Paolo Dario, and Nicola Vitiello. A modular sensorized mat for monitoring infant posture. Sensors, 14(1):510–531, 2013. 3

[14] Hao-Shu Fang, Jiefeng Li, Hongyang Tang, Chao Xu, Haoyi Zhu, Yuliang Xiu, Yong-Lu Li, and Cewu Lu. Alphapose: Whole-body regional multi-person pose estimation

and tracking in real-time. IEEE TPAMI, 45(6):7157–7173, 2022. 3

[15] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4d: Reconstructing and tracking humans with transformers. In ICCV, pages 14783–14794, 2023. 2, 3, 7, 8, 5, 6

[16] Daniel Groos, Lars Adde, Ragnhild Støen, Heri Ramampiaro, and Espen AF Ihlen. Towards human-level performance on automatic pose estimation of infant spontaneous movements. Computerized Medical Imaging and Graphics, 95:102012, 2022. 3

[17] Nikolas Hesse, Christoph Bodensteiner, Michael Arens, Ulrich G Hofmann, Raphael Weinberger, and A Sebastian Schroeder. Computer vision for medical infant motion analysis: State of the art and RGB-D data set. In Proceed ings of the European conference on computer vision (ECCV) workshops, pages 0–0, 2018. 2, 3

[18] Nikolas Hesse, Sergi Pujades, Javier Romero, Michael J. Black, Christoph Bodensteiner, Michael Arens, Ulrich G. Hofmann, Uta Tacke, Mijna Hadders-Algra, Raphael Wein berger, Wolfgang Muller-Felber, and A. Sebastian Schroeder.¨ Learning an infant body model from RGB-D data for accurate full body motion analysis. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2018. 2, 3, 4

[19] Xiaofei Huang, Nihang Fu, Shuangjun Liu, and Sarah Ostad abbas. Invariant representation learning for infant pose estimation with small data. In IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021, 2021. 2, 3, 4

[20] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE TPAMI, 36(7):1325–1339, 2014. 2, 3

[21] Lennart Jahn, Sarah Flugge, Dajie Zhang, Luise Poustka,¨ Sven Bolte, Florentin W¨ org¨ otter, Peter B Marschik, and¨ Tomas Kulvicius. Comparison of marker-less 2d imagebased methods for infant pose estimation. Scientific Reports, 15(1):12148, 2025. 3

[22] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In CVPR, pages 7122–7131, 2018. 2, 3, 7, 8, 5

[23] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. In ECCV, 2024. 3, 6

[24] Muhammed Kocabas, Nikos Athanasiou, and Michael J Black. Vibe: Video inference for human body pose and shape estimation. In CVPR, pages 5253–5263, 2020. 3

[25] Muhammed Kocabas, Chun-Hao P. Huang, Otmar Hilliges, and Michael J. Black. Pare: Part attention regressor for 3d human body estimation. In ICCV, pages 11127–11137, 2021. 3, 7, 8, 5

[26] Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. Learning to reconstruct 3d human pose and shape via model-fitting in the loop. In ICCV, pages 2252–2261, 2019. 2

[27] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A largescale 3d expressive whole-body human motion dataset. In NeurIPS, 2023. 6

[28] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755. Springer, 2014. 5

[29] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. Ntu rgb+d 120: A largescale benchmark for 3d human activity understanding. IEEE TPAMI, 42(10):2684–2701, 2020. 6

[30] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model. ACM TOG, 34(6):248:1–248:16, 2015. 3

[31] Peng Lu, Tao Jiang, Yining Li, Xiangtai Li, Kai Chen, and Wenming Yang. Rtmo: Towards high-performance onestage real-time multi-person pose estimation. In CVPR, pages 1491–1500, 2024. 4

[32] Sara N Lupolt, Qinfan Lyu, Guofeng Zhang, Jiahao Wang, Stacey Tang, Jamie Cho, Christina Huynh, Alan Yuille, Kristin Voegtline, and Keeve E Nachman. Application of a computer vision algorithm to quantify the frequency and duration of children’s microactivities in different play scenarios. Journal of Exposure Science & Environmental Epidemiology, pages 1–12, 2025. 2, 8

[33] Sara N Lupolt, Guofeng Zhang, Jiahao Wang, Stacey Tang, Qinfan Lyu, Jamie Cho, Christina Huynh, Qihao Liu, Jiawei Peng, Xingrui Wang, et al. Development and evaluation of a computer vision algorithm for quantification of children’s microactivities. Journal ofExposure Science & Environmental Epidemiology, pages 1–9, 2025. 2, 8

[34] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In ICCV, pages 5442–5451, 2019. 3, 6

[35] Lucia Migliorelli, Sara Moccia, Rocco Pietrini, Virgilio Paolo Carnielli, and Emanuele Frontoni. The babypose dataset. Data in brief, 33:106329, 2020. 3

[36] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In ECCV, pages 483–499. Springer, 2016. 6

[37] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In CVPR, pages 10975– 10985, 2019. 2, 3

[38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2:¨ Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 4

[39] Giuseppa Sciortino, Giovanni Maria Farinella, Sebastiano Battiato, Marco Leo, and Cosimo Distante. On the estimation of children’s poses. In International conference on image analysis and processing, pages 410–421. Springer, 2017. 3

[40] Ameur Soualmi, Christophe Ducottet, Hugues Patural, Antoine Giraud, and Olivier Alata. A 3d pose estimation framework for preterm infants hospitalized in the neonatal unit. Multimedia Tools and Applications, 83(8):24383– 24400, 2024. 3

[41] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose esti mation. In CVPR, pages 5693–5703, 2019. 3, 6

[42] Timo von Marcard, Roberto Henschel, Michael J. Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3d human pose in the wild using imus and a moving camera. In ECCV, pages 601–617, 2018. 2

[43] Jiacong Xu, Yi Zhang, Jiawei Peng, Wufei Ma, Artur Jesslen, Pengliang Ji, Qixin Hu, Jiehua Zhang, Qihao Liu, Jiahao Wang, et al. Animal3d: A comprehensive dataset of 3d ani mal pose and shape. In ICCV, pages 9099–9109, 2023. 4

[44] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vit pose: Simple vision transformer baselines for human pose estimation. NeurIPS, 35:38571–38584, 2022. 2, 3, 6

[45] Junjie Oscar Yin, Ting Li, Jiahao Wang, Yi Zhang, and Alan Yuille. Easyret3d: Uncalibrated multi-view multi-human 3d reconstruction and tracking. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3128–3137. IEEE, 2025. 4, 5

[46] Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, and Jingdong Wang. Hrformer: High resolution vision transformer for dense predict. NeurIPS, 34: 7281–7293, 2021. 6

[47] Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, and Ce Zhu. Distribution-aware coordinate representation for human pose estimation. In CVPR, pages 7093–7102, 2020. 6

[48] Zhuoran Zhou, Zhongyu Jiang, Wenhao Chai, Cheng-Yen Yang, Lei Li, and Jenq-Neng Hwang. Efficient domain adaptation via generative prior for 3d infant pose estimation. In Proceedings of the IEEE/CVF Winter Conference on Appli cations ofComputer Vision (WACV) Workshops, 2024. 3, 8

# InfantNet: A Large Scale Dataset for Infant Body Pose and Shape Estimation

Supplementary Material

## A. Appendix

## A.1. 2D Source-Specific Evaluations

We analyze model performance by separately training and evaluating on the YouTube and HomeVisit subsets to understand how well existing human pose estimators adapt to each subset in isolation.

Setup. We follow the evaluation procedure outlined in Sec. 4.1, using the same model settings and metrics.

Evaluation. For each source, models are finetuned on the corresponding training split $( \mathrm { Y o u T u b e } _ { \mathrm { T r a i n } }$ or ${ \mathrm { H o m e V i s i t } } _ { \mathrm { T r a i n } } )$ and evaluated on the matching test split $( \mathrm { Y o u T u b e _ { T e s t } }$ or $\mathrm { H o m e V i s i t _ { T e s t } ) }$ . No cross-source data are used during training in this setting. All identity separations within each source are preserved. Metrics follow the standard COCO keypoint protocol over 17 keypoints: mean Average Precision (AP), AP.5, AP.75, and Average Recall (AR). We report results using the checkpoint that achieves the best validation AP on each source.

Result. In addition to the mixed-subset results presented in the main paper, we further benchmark the models on each source independently. Specifically, we evaluate their performance on the YouTube and HomeVisit subsets under two settings: (1) direct inference using COCO-pretrained weights, and (2) fine-tuning on the source-specific training split. Table S1 and S2’s left two blocks report the corresponding Average Precision (AP), Average Recall (AR), AP.5, and AP.75 for all evaluated configurations.

(a) Direct Inference on YouTube Test Set. On the YouTube Test Set, ViTPose-large achieves the highest performance among all models, obtaining an AP of 0.9233 and an AR of 0.9443. ViTPose-huge also performs strongly (AP: 0.9184, AR: 0.9437). Among CNN-based models, HRNet-W48-Dark 256x192 is the strongest performer (AP: 0.8977, AR: 0.9234), while the transformer-based HRFormer-base 384x288 achieves AP and AR of 0.8993 and 0.9257, respectively.

(b) Direct Inference on HomeVisit Test Set. On the HomeVisit Test Set, ViTPose-large again achieves the highest AP (0.8901) and ViTPose-huge achieves the highest AR (0.9120). CNN models perform competitively, with HRNet-W48-Dark 256x192 reaching AP(0.8849) and AR(0.9013). Across both subsets, transformer-based ViT-Pose models consistently achieve the strongest directinference results, and larger model variants within each family outperform their smaller counterparts.

(c) Fine-tuning on YouTube Training Set. Fine-tuning on the YouTube training subset consistently improves performance across all architectures. On average, fine-tuning yields a relative gain of 5.0% in AP and 3.7% in AR compared to direct inference. Among transformer-based models, ViTPose-huge achieves the highest AP (0.9446) after adaptation, while ViTPose-large obtains the highest AR (0.9600). CNN-based models also benefit substantially from fine-tuning; HRNet-W48-Dark 384x288 improves from an AP of 0.8961 to 0.9366 and from an AR of 0.9225 to 0.9535.

(d) Fine-tuning on HomeVisit Training Set. Fine-tuning on the HomeVisit training subset yields consistent improvements across all architectures. On average, fine-tuning provides a relative gain of 6.1% in AP and 5.3% in AR compared to direct inference. ViTPose-large achieves the highest AP and AR after adaptation (AP: 0.9433, AR: 0.9567). CNN-based models also show substantial gains; Res152 256x192 improves from an AP of 0.8700 to 0.9308 and from an AR of 0.8893 to 0.9428.

## A.2. 2D Cross-Subset Evaluations

We next evaluate cross-subset generalization by fine-tuning on one source and testing on the other. This setting allows us to assess the robustness of each model under domain shift. It also reveals how well knowledge learned from one environment transfers to a visually distinct target domain.

Setup. The experiment uses the same protocol detailed in Sec. 4.1, maintaining identical model configurations and evaluation metrics.

Evaluation. For cross-subset evaluation, models are fine-tuned on the training split of one source and evaluated on the test split of the other. Specifically, we consider two cross-subset directions: models fine-tuned on $\mathrm { H o m e V i s i t _ { T r a i n } }$ are evaluated on ${ \mathrm { Y o u T u b e } } _ { \mathrm { T e s t } } .$ and models fine-tuned on $\mathtt { Y o u T u b e } _ { \mathrm { T r a i n } }$ are evaluated on $\mathrm { H o m e V i s i t } _ { \mathrm { T e s t } }$ No data from the target source are used during training. Identity separations within each source are preserved. Evaluation in this section adopts the same COCO keypoint metrics and checkpoint selection protocol introduced in our source-specific experiments, ensuring methodological consistency across all experiments.

Result. In addition to the source-specific results reported in Sec. A.1, we further evaluate cross-subset generalization between the YouTube and HomeVisit subsets. Table S1 and S2’s rightmost blocks report the corresponding Average Precision (AP), Average Recall (AR), AP.5, and AP.75 for all configurations.

<table><tr><td rowspan="2">Model</td><td colspan="4">Direct Inference</td><td colspan="4">Source-Specific Fine-tuned</td><td colspan="4">Cross-Subset Fine-tuned</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td></tr><tr><td>Hourglass 256x256</td><td>0.8588</td><td>0.9790</td><td>0.9440</td><td>0.8868</td><td>0.9193</td><td>0.9898</td><td>0.9678</td><td>0.9374</td><td>0.8667</td><td>0.9896</td><td>0.9566</td><td>0.8962</td></tr><tr><td>Hourglass 384x384</td><td>0.8707</td><td>0.9794</td><td>0.9452</td><td>0.8987</td><td>0.9284</td><td>0.9897</td><td>0.9781</td><td>0.9450</td><td>0.8808</td><td>0.9894</td><td>0.9566</td><td>0.9091</td></tr><tr><td>HRFormer-small_256x192</td><td>0.8761</td><td>0.9779</td><td>0.9537</td><td>0.9062</td><td>0.9247</td><td>0.9896</td><td>0.9778</td><td>0.9422</td><td>0.8734</td><td>0.9896</td><td>0.9549</td><td>0.9023</td></tr><tr><td>HRFormer-small_384x288</td><td>0.8852</td><td>0.9887</td><td>0.9557</td><td>0.9145</td><td>0.9307</td><td>0.9895</td><td>0.9788</td><td>0.9475</td><td>0.8816</td><td>0.9896</td><td>0.9658</td><td>0.9115</td></tr><tr><td>HRFormer-base_256x192</td><td>0.8900</td><td>0.9883</td><td>0.9559</td><td>0.9172</td><td>0.9325</td><td>0.9893</td><td>0.9782</td><td>0.9489</td><td>0.8964</td><td>0.9898</td><td>0.9673</td><td>0.9235</td></tr><tr><td>HRFormer-base_384x288</td><td>0.8993</td><td>0.9875</td><td>0.9640</td><td>0.9257</td><td>0.9371</td><td>0.9893</td><td>0.9789</td><td>0.9535</td><td>0.9063</td><td>0.9897</td><td>0.9681</td><td>0.9301</td></tr><tr><td>Res50-256x192</td><td>0.8629</td><td>0.9888</td><td>0.9444</td><td>0.8936</td><td>0.9162</td><td>0.9898</td><td>0.9682</td><td>0.9354</td><td>0.8601</td><td>0.9793</td><td>0.9558</td><td>0.8914</td></tr><tr><td>Res50-384x288</td><td>0.8583</td><td>0.9781</td><td>0.9424</td><td>0.8906</td><td>0.9228</td><td>0.9900</td><td>0.9790</td><td>0.9399</td><td>0.8658</td><td>0.9894</td><td>0.9458</td><td>0.8958</td></tr><tr><td>Res101-256x192</td><td>0.8736</td><td>0.9887</td><td>0.9546</td><td>0.9024</td><td>0.9205</td><td>0.9898</td><td>0.9785</td><td>0.9400</td><td>0.8638</td><td>0.9794</td><td>0.9553</td><td>0.8939</td></tr><tr><td>Res101-384x288</td><td>0.8720</td><td>0.9782</td><td>0.9543</td><td>0.9007</td><td>0.9301</td><td>0.9898</td><td>0.9788</td><td>0.9453</td><td>0.8832</td><td>0.9798</td><td>0.9556</td><td>0.9096</td></tr><tr><td>Res152-256x192</td><td>0.8790</td><td>0.9891</td><td>0.9551</td><td>0.9070</td><td>0.9263</td><td>0.9898</td><td>0.9786</td><td>0.9428</td><td>0.8784</td><td>0.9898</td><td>0.9573</td><td>0.9078</td></tr><tr><td>Res152-384x288</td><td>0.8864</td><td>0.9889</td><td>0.9556</td><td>0.9132</td><td>0.9319</td><td>0.9897</td><td>0.9784</td><td>0.9480</td><td>0.8913</td><td>0.9896</td><td>0.9564</td><td>0.9175</td></tr><tr><td>HRNet-w32_dark_256x192</td><td>0.8888</td><td>0.9788</td><td>0.9567</td><td>0.9151</td><td>0.9312</td><td>0.9894</td><td>0.9784</td><td>0.9470</td><td>0.8938</td><td>0.9896</td><td>0.9665</td><td>0.9165</td></tr><tr><td>HRNet-w32_dark_384x288</td><td>0.8954</td><td>0.9885</td><td>0.9655</td><td>0.9204</td><td>0.9343</td><td>0.9894</td><td>0.9778</td><td>0.9517</td><td>0.8992</td><td>0.9896</td><td>0.9685</td><td>0.9220</td></tr><tr><td>HRNet-w48_dark_256x192</td><td>0.8977</td><td>0.9888</td><td>0.9663</td><td>0.9234</td><td>0.9333</td><td>0.9887</td><td>0.9770</td><td>0.9501</td><td>0.8977</td><td>0.9898</td><td>0.9677</td><td>0.9210</td></tr><tr><td>HRNet-w48_dark_384x288</td><td>0.8961</td><td>0.9886</td><td>0.9662</td><td>0.9225</td><td>0.9366</td><td>0.9880</td><td>0.9768</td><td>0.9535</td><td>0.9056</td><td>0.9898</td><td>0.9671</td><td>0.9276</td></tr><tr><td>HRNet-w32_256x192</td><td>0.8828</td><td>0.9888</td><td>0.9561</td><td>0.9084</td><td>0.9241</td><td>0.9898</td><td>0.9778</td><td>0.9427</td><td>0.8861</td><td>0.9895</td><td>0.9670</td><td>0.9116</td></tr><tr><td>HRNet-w32_384x288</td><td>0.8889</td><td>0.9882</td><td>0.9653</td><td>0.9160</td><td>0.9333</td><td>0.9895</td><td>0.9782</td><td>0.9496</td><td>0.8940</td><td>0.9894</td><td>0.9673</td><td>0.9185</td></tr><tr><td>HRNet-w48_256x192</td><td>0.8890</td><td>0.9886</td><td>0.9559</td><td>0.9161</td><td>0.9292</td><td>0.9889</td><td>0.9774</td><td>0.9476</td><td>0.8924</td><td>0.9899</td><td>0.9675</td><td>0.9176</td></tr><tr><td>HRNet-w48_384x288</td><td>0.8935 0.8826</td><td>0.9789 0.9796</td><td>0.9560 0.9567</td><td>0.9179</td><td>0.9355</td><td>0.9886</td><td>0.9779</td><td>0.9520</td><td>0.8986</td><td>0.9896</td><td>0.9677</td><td>0.9229</td></tr><tr><td>ViTPose-small_256x192</td><td>0.9050</td><td>0.9895</td><td>0.9679</td><td>0.9087 0.9271</td><td>0.9273 0.9350</td><td>0.9900 0.9897</td><td>0.9788 0.9791</td><td>0.9434 0.9504</td><td>0.8864 0.9112</td><td>0.9798</td><td>0.9580</td><td>0.9118</td></tr><tr><td>ViTPose-base_256x192</td><td>0.9233</td><td>0.9886</td><td>0.9760</td><td>0.9443</td><td>0.9443</td><td>0.9895</td><td>0.9785</td><td>0.9600</td><td>0.9332</td><td>0.9896 0.9895</td><td>0.9779 0.9785</td><td>0.9317</td></tr><tr><td>ViTPose-large_256x192</td><td>0.9184</td><td>0.9876</td><td>0.9660</td><td>0.9437</td><td>0.9446</td><td>0.9887</td><td>0.9780</td><td>0.9597</td><td>0.9347</td><td>0.9888</td><td>0.9779</td><td>0.9501</td></tr><tr><td>ViTPose-huge_256x192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.9521</td></tr></table>

Table S1. Comparison of 2D pose estimation models on the InfantNet Subsets $( \mathrm { Y o u T u b e _ { T e s t } } )$ benchmark under three evaluation settings: direct inference on $\mathrm { Y o u T u b e _ { T e s t } }$ using COCO-pretrained models (left block), source-specific fine-tuning on $\mathtt { Y o u T u b e } _ { \mathrm { T r a i n } }$ (middle block), and cross-subset fine-tuning on $\mathrm { { I o m e V i s i t } _ { T r a i n } }$ (right block). We report Average Precision (AP), AP at IoU thresholds 0.5 (AP.5) and 0.75 (AP.75), and Average Recall (AR). Across all architectures, fine-tuning substantially improves performance. On average, source-specific adaptation yields the relative gains of 5.0% in AP and 3.7% in AR, whereas cross-subset fine-tuning provides margin relative gains of 0.5% in AP and 0.3% in AR over direct inference.

Fine-tuning on HomeVisit and Evaluating on YouTube. Fine-tuning on the HomeVisit subset produces only marginal gains on the YouTube test set, with an average relative gain of 0.5% in AP and 0.3% in AR. This limited increase is expected and reflects a clear domain mismatch: HomeVisit videos feature fixed viewpoints, stable lighting, and clean indoor scenes, whereas YouTube videos contain far greater variability in camera motion, compression artifacts, occlusions, and infant appearance. Consequently, models fine-tuned on HomeVisit adapt to a simpler and narrower distribution, offering little transfer to the more complex YouTube subset. Higher-capacity models still show modest improvements. ViTPose-huge reaches the highest AP (0.9347) and the highest AR (0.9521) after fine-tuning.

Fine-tuning on YouTube and Evaluating on HomeVisit. Fine-tuning on the YouTube train set leads to substantial improvements on the HomeVisit test set, with an average relative gain of 5.2% in AP and 4.7% in AR. It reflects the fact that YouTube provides a larger and more diverse training distribution that transfers effectively to the more controlled HomeVisit subset. All model families benefit consistently from this direction of adaptation. Transformer-based architectures exhibit the largest absolute performance, with ViTPose-large and ViTPose-huge reaching AP values of 0.9347 and 0.9362, respectively, after fine-tuning. Overall, these results demonstrate a clear asymmetry in cross-subset transfer: training on the more complex and diverse YouTube domain significantly boosts performance on the HomeVisit subset, whereas the reverse direction yields only marginal gains. This highlights the value of high-variability data for robust infant pose estimation.

## A.3. 3D Infant Pose and Shape Annotations

Figure S1 presents additional examples of the 3D infant meshes obtained through parameters generated from our SMIL optimization process. These visualizations illustrate the diversity of infant poses and body shapes captured in the fitted meshes used as 3D annotations in InfantNet to train the end-to-end models, complementing the samples shown in the main paper.

<table><tr><td rowspan="2">Model</td><td colspan="4">Direct Inference</td><td colspan="4">Source-Specific Fine-tuned</td><td colspan="4">Cross-Subset Fine-tuned</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td></tr><tr><td>Hourglass 256x256</td><td>0.8643</td><td>0.9570</td><td>0.9257</td><td>0.8834</td><td>0.9222</td><td>0.9898</td><td>0.9689</td><td>0.9357</td><td>0.9116</td><td>0.9799</td><td>0.9687</td><td>0.9273</td></tr><tr><td>Hourglass 384x384</td><td>0.8724</td><td>0.9572</td><td>0.9358</td><td>0.8907</td><td>0.9288</td><td>0.9897</td><td>0.9790</td><td>0.9415</td><td>0.9177</td><td>0.9797</td><td>0.9689</td><td>0.9329</td></tr><tr><td>HRFormer-small_256x192</td><td>0.8766</td><td>0.9573</td><td>0.9359</td><td>0.8954</td><td>0.9235</td><td>0.9896</td><td>0.9687</td><td>0.9369</td><td>0.9194</td><td>0.9796</td><td>0.9685</td><td>0.9350</td></tr><tr><td>HRFormer-small_384x288</td><td>0.8774</td><td>0.9568</td><td>0.9352</td><td>0.8960</td><td>0.9261</td><td>0.9893</td><td>0.9688</td><td>0.9398</td><td>0.9219</td><td>0.9896</td><td>0.9688</td><td>0.9369</td></tr><tr><td>HRFormer-base_256x192</td><td>0.8769</td><td>0.9560</td><td>0.9346</td><td>0.8977</td><td>0.9326</td><td>0.9897</td><td>0.9791</td><td>0.9452</td><td>0.9260</td><td>0.9893</td><td>0.9688</td><td>0.9423</td></tr><tr><td>HRFormer-base_384x288</td><td>0.8832</td><td>0.9559</td><td>0.9343</td><td>0.9023</td><td>0.9369</td><td>0.9895</td><td>0.9788</td><td>0.9487</td><td>0.9294</td><td>0.9893</td><td>0.9688</td><td>0.9447</td></tr><tr><td>Res50-256x192</td><td>0.8621</td><td>0.9575</td><td>0.9258</td><td>0.8833</td><td>0.9189</td><td>0.9898</td><td>0.9691</td><td>0.9330</td><td>0.9102</td><td>0.9795</td><td>0.9686</td><td>0.9248</td></tr><tr><td>Res50-384x288</td><td>0.8654</td><td>0.9573</td><td>0.9256</td><td>0.8848</td><td>0.9212</td><td>0.9899</td><td>0.9691</td><td>0.9346</td><td>0.9150</td><td>0.9797</td><td>0.9686</td><td>0.9294</td></tr><tr><td>Res101-256x192</td><td>0.8718</td><td>0.9569</td><td>0.9357</td><td>0.8897</td><td>0.9263</td><td>0.9899</td><td>0.9791</td><td>0.9400</td><td>0.9186</td><td>0.9797</td><td>0.9690</td><td>0.9331</td></tr><tr><td>Res101-384x288</td><td>0.8763</td><td>0.9569</td><td>0.9356</td><td>0.8941</td><td>0.9291</td><td>0.9899</td><td>0.9693</td><td>0.9425</td><td>0.9246</td><td>0.9896</td><td>0.9691</td><td>0.9398</td></tr><tr><td>Res152-256x192</td><td>0.8700</td><td>0.9566</td><td>0.9263</td><td>0.8893</td><td>0.9308</td><td>0.9898</td><td>0.9791</td><td>0.9428</td><td>0.9211</td><td>0.9796</td><td>0.9689</td><td>0.9374</td></tr><tr><td>Res152-384x288</td><td>0.8778</td><td>0.9569</td><td>0.9354</td><td>0.8955</td><td>0.9338</td><td>0.9899</td><td>0.9791</td><td>0.9449</td><td>0.9246</td><td>0.9799</td><td>0.9691</td><td>0.9389</td></tr><tr><td>HRNet-w32_dark_256x192</td><td>0.8844</td><td>0.9575</td><td>0.9363</td><td>0.8999</td><td>0.9329</td><td>0.9898</td><td>0.9791</td><td>0.9449</td><td>0.9256</td><td>0.9797</td><td>0.9689</td><td>0.9398</td></tr><tr><td>HRNet-w32_dark_384x288</td><td>0.8831</td><td>0.9568</td><td>0.9355</td><td>0.9017</td><td>0.9365</td><td>0.9899</td><td>0.9794</td><td>0.9475</td><td>0.9267</td><td>0.9895</td><td>0.9690</td><td>0.9426</td></tr><tr><td>HRNet-w48_dark_256x192</td><td>0.8849</td><td>0.9571</td><td>0.9362</td><td>0.9013</td><td>0.9369</td><td>0.9898</td><td>0.9794</td><td>0.9485</td><td>0.9278</td><td>0.9797</td><td>0.9691</td><td>0.9415</td></tr><tr><td>HRNet-w48_dark_384x288</td><td>0.8815</td><td>0.9570</td><td>0.9356</td><td>0.9005</td><td>0.9383</td><td>0.9897</td><td>0.9794</td><td>0.9500</td><td>0.9317</td><td>0.9896</td><td>0.9789</td><td>0.9450</td></tr><tr><td>HRNet-w32_256x192</td><td>0.8788</td><td>0.9576</td><td>0.9361</td><td>0.8965</td><td>0.9294</td><td>0.9899</td><td>0.9792</td><td>0.9426</td><td>0.9191</td><td>0.9797</td><td>0.9686</td><td>0.9340</td></tr><tr><td>HRNet-w32_384x288</td><td>0.8797</td><td>0.9571</td><td>0.9355</td><td>0.8979</td><td>0.9327</td><td>0.9898</td><td>0.9790</td><td>0.9436</td><td>0.9222</td><td>0.9796</td><td>0.9687</td><td>0.9380</td></tr><tr><td>HRNet-w48_256x192</td><td>0.8809</td><td>0.9572 0.9568</td><td>0.9357 0.9356</td><td>0.8978</td><td>0.9336</td><td>0.9897</td><td>0.9790 0.9792</td><td>0.9454</td><td>0.9211</td><td>0.9797</td><td>0.9690</td><td>0.9374</td></tr><tr><td>HRNet-w48_384x288</td><td>0.8801 0.8714</td><td>0.9568</td><td>0.9357</td><td>0.8992 0.8898</td><td>0.9369 0.9244</td><td>0.9897 0.9898</td><td>0.9690</td><td>0.9478 0.9378</td><td>0.9274</td><td>0.9894</td><td>0.9691</td><td>0.9428</td></tr><tr><td>ViTPose-small_256x192</td><td>0.8849</td><td>0.9566</td><td>0.9361</td><td>0.9029</td><td>0.9348</td><td>0.9898</td><td>0.9790</td><td>0.9473</td><td>0.9190 0.9252</td><td>0.9895 0.9892</td><td>0.9690</td><td>0.9344</td></tr><tr><td>ViTPose-base_256x192</td><td>0.8901</td><td>0.9560</td><td>0.9350</td><td>0.9117</td><td>0.9433</td><td>0.9896</td><td>0.9791</td><td>0.9567</td><td>0.9347</td><td>0.9893</td><td>0.9688 0.9784</td><td>0.9407</td></tr><tr><td>ViTPose-large_256x192</td><td>0.8890</td><td>0.9561</td><td>0.9439</td><td>0.9120</td><td>0.9413</td><td>0.9897</td><td>0.9791</td><td>0.9561</td><td>0.9362</td><td>0.9893</td><td>0.9783</td><td>0.9498</td></tr><tr><td>ViTPose-huge_256x192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.9501</td></tr></table>

Table S2. Comparison of 2D pose estimation models on the InfantNet Subsets(HomeVisit<sub>Test</sub>) benchmark under three evaluation settings: direct inference using COCO-pretrained models (left block), fine-tuning on HomeVisit (middle block), and fine-tuning on YouTube (right block). We report Average Precision (AP), AP at IoU thresholds 0.5 (AP.5) and 0.75 (AP.75), and Average Recall (AR). Across all architectures, fine-tuning substantially improves performance. In average, source-specific adaptation yields the relative gains of 6.1% in AP and 5.3% in AR, whereas cross-subset fine-tuning offers comparably strong relative gains of 5.2% in AP and 4.7% in AR over direct inference.

<table><tr><td>Model</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>HMR [22]</td><td>118.53</td><td>62.30</td></tr><tr><td>PARE [25]</td><td>107.52</td><td>63.83</td></tr><tr><td>4D-Humans [15]</td><td>84.50</td><td>42.40</td></tr></table>

Table S3. Cross-subset 3D pose and shape estimation. The models are trained on the HomeVisit subset and tested on the YouTube subset. Metrics: MPJPE and PA-MPJPE (mm). Infant models consistently improve accuracy across all architectures, even with cross-subset training and evaluation, validating the generalizability of the infant models.

## A.4. Occlusion and Edge Case Analysis

On frames with ≥ 5 occluded keypoints, the infant-specific 4D-Humans achieves 91.28 mm MPJPE and 48.84 mm PA-MPJPE, comparable to Table 4. We further define rare poses and viewpoints by 5th-NN distance in pelviscentered SMIL-joint space and normalized camera-pose space, respectively. On the top 10% rare-pose samples, the model achieves 103.53/57.25 mm MPJPE/PA-MPJPE, and 85.47/46.28 mm on the top 10% rare-viewpoint samples, indicating robustness to challenging occlusions, poses, and viewpoints. Figure S2 presents examples of our infantspecific 4D-Humans model on edge cases, including occlusion, rare poses, and rare viewpoints.

## A.5. 3D Cross-Subset Evaluations

We evaluate the generalizability of the 3D infant pose and shape estimation models under cross-subset conditions, where the HomeVisit subset serves as training data and the YouTube subset is used for evaluation. This experiment mimics a more realistic scenario and reflects the sensitivity of the infant models to variations in environmental settings.

Setup. We follow the same training and evaluation protocol described in Sec. 4.2, with identical model configurations and evaluation metrics.

Data Preparation. We train the models exclusively on the HomeVisit subset and evaluate on the YouTube subset to assess cross-subset generalization. The models are trained with 9,362 frames and evaluated with 2,280 frames.

![](images/7d5185b5d6d55fba2d0ce6c9dbaef166b31faf202f18923d731db3ae66f7ee34.jpg)  
Figure S1. Additional examples of meshes generated via optimization-based SMIL fitting for the InfantNet dataset. The refined SMIL parameters become the 3D annotations used to train end-to-end infant models. InfantNet captures a diverse set of infant shapes and poses with high-quality 2D and 3D annotations.

Results. As shown in Table S3, training and evaluating on different data sources leads to moderate accuracy drops compared to the results in Sec. 4.2. Nevertheless, all infant models still outperform their adult baselines across architectures, indicating strong cross-domain generalization. Among them, 4D-Humans achieves the best performance (MPJPE: 84.50 mm, PA-MPJPE: 42.40 mm). The results suggest that the 3D infant pose and shape annotations of InfantNet retain cross-subset robustness despite differences in capture environmental settings.

![](images/5fa7066209ac42653f8c12e6fe57209cbff3e0ef50066b548257672968ce496b.jpg)  
Figure S2. Examples of occlusion, pose, viewpoint edge cases, respectively.

## A.6. End-to-End vs Optimization

To evaluate the benefits of learning-based 3D infant pose and shape estimation, we compare our end-to-end models from Sec. 4.2 against the optimization-based SMIL fitting process used to generate ground-truth 3D annotations in Sec. 3.4. This comparison highlights two key aspects for end-to-end models: inference efficiency and robustness to cases where optimization fails or produces severely deformed meshes. By evaluating inference speed and estimation quality, we demonstrate that end-to-end regression not only achieves faster inference, but also yields more reliable estimations, reflecting the practical values of end-toend models for 3D infant modeling.

Inference Speed. From Table S4, end-to-end models are significantly faster than optimization-based SMIL fitting in terms of inference. While optimization requires an average of 38.7020 seconds per image for inference, end-to-end models operate in the millisecond range, with HMR achieving the fastest speed (≈ 3.25 ms). The results clearly demonstrate the efficiency of end-to-end regression, enabling real-time inference that optimization-based methods cannot achieve.

Estimation Quality. We conduct this experiment to evaluate the reliability of end-to-end regression in cases where optimization-based SMIL fitting fails. As shown in Figure S3, optimization outputs exhibit mesh distortions, such as implausible limb articulations and inconsistent body proportions. In contrast, the end-to-end 4D-Humans model produces anatomically coherent meshes that align better with the underlying infant silhouette. The results highlight the practical significance of learning-based regression, yielding more reliable 3D infant pose and shape estimations.

<table><tr><td>Model</td><td>Avg Time (s / img)</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>SMIL-Opt [3, 45]</td><td>38.7020</td></tr><tr><td>End-to-End Models</td><td></td></tr><tr><td>HMR [22] PARE [25]</td><td>0.0033 0.0091</td></tr><tr><td>4D-Humans [15]</td><td>0.0212</td></tr><tr><td></td><td></td></tr></table>

Table S4. Comparison of average inference speed between optimization-based SMIL fitting and end-to-end infant modeling models. End-to-end models achieve millisecond-level inference, demonstrating the substantial computational advantage of learning-based 3D infant pose and shape estimation models.
<table><tr><td>Model</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>HMR [22]</td><td>100.00</td><td>55.60</td></tr><tr><td>PARE [25]</td><td>78.48</td><td>59.41</td></tr><tr><td>4D-Humans [15]</td><td>86.57</td><td>50.15</td></tr></table>

Table S5. 3D pose and shape estimation on masked infant inputs. Metrics: MPJPE and PA-MPJPE (mm). Applying masking results in negligible degradation, demonstrating model robustness and dataset viability.

## A.7. 3D Evaluation with Face-Masked Data

We evaluate the impact of privacy-preserving face masking on 3D infant pose and shape estimation. This experiment assesses the robustness of the proposed end-to-end infant models to masked inputs and the research viability of our anonymized dataset for accurate 3D infant pose and shape estimation.

![](images/3865f30829bf1cdccbc3f71c49b1fa40dc70ad65eb0f9bd903e3cf428d966fde.jpg)  
Figure S3. Visualization of 3D pose and shape estimation model performances. The columns from left to right: the input image, optimization-based SMIL fitting, and inference with end-toend model (4D-Humans [15]) trained with infant body model and data, respectively. The end-to-end 4D-Humans model produces anatomically plausible meshes that better align with the infant silhouette, while optimization-based SMIL fitting produces deformed meshes, illustrating the robustness of end-to-end models in 3D infant estimation compared to optimization-based fitting.

Setup. We again follow the same train and evaluation protocol described in Sec. 4.2, using identical model configurations and evaluation metrics.

Data Preparation. We train and evaluate the models with the same set of data described in Sec. 4.2, replacing the input images with versions in which the face regions are masked using our privacy-preserving pipeline. The masking removes identifiable facial features such as eyes and noses while preserving the overall appearance of the head and body contours, thereby retaining the visual cues for accurate 3D estimation.

Results. As shown in Table S5, masking-based anonymization does not introduce systematic degradation in 3D infant pose and shape estimation, with performance variations limited to approximately 1–5 mm compared to the original results in Sec. 4.2. HMR and PARE exhibit slight improvements under masked inputs (e.g., PARE improves from 83.73 mm to 78.48 mm in MPJPE), while 4D-Humans shows a marginal decrease (84.73 mm to 86.57 mm in MPJPE). The improvements observed for the CNNbased models (HMR, PARE) may stem from their reliance on structural cues, and suppressing high-frequency facial details removes visual noise without affecting overall geometry. In contrast, the small performance drop for 4D-

Humans suggests the sensitivity to fine-grained local texture cues of attention-based architecture. The minimal impact of anonymization indicates that the masking procedure preserves essential structural and geometric cues required for accurate 3D reconstruction, while effectively removing identifiable facial information. These findings validate both the robustness of our models and the viability of our privacy-preserving dataset design.

## A.8. Public Release

Data. To support future research in 3D infant modeling, we plan to release all the annotations of the training and evaluation data, with strict privacy safeguards upon acceptance of this paper, and checkpoints of the models finetuned on our InfantNet.

Code. We will release the implementation of our data processing and end-to-end 3D infant pose and shape estimation models for reproducibility.