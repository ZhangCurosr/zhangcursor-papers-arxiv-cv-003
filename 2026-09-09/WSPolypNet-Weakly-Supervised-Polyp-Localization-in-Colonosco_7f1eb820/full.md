# WSPolypNet: Weakly Supervised Polyp Localization in Colonoscopy Videos

Giseong Hwang \* Soonchunhyang University hwanggiseong@sch.ac.kr

Seoyeon Han Seoul National University ss stella@snu.ac.kr

Insung Hwang Seoul National University insung0608@snu.ac.kr

Minjae Jo<sup>\*</sup> Seoul National University whalswo0503@snu.ac.kr

Donghoon Han Seoul National University leukinii@snu.ac.kr

Pa Hong Samsung Changwon Hospital papa.hong@samsung.com

Yeonghyeon Park Kyungpook National University yeonghyeon05@knu.ac.kr

Haneul Kim Chung-Ang University kimhn126@cau.ac.kr

Ken Ying-Kai Liao NVIDIA AI Technology Center kenyingkail@nvidia.com

Kyeonghun Kim OUTTA kyeonghun.kim@outta.ai

Yului Jeong Seoul National University yule27@snu.ac.kr

Nam-Joon Kim<sup>†</sup> Seoul National University knj01@snu.ac.kr

Abstract—Because dense frame-level annotation of colonoscopy videos is costly, we propose WSPolypNet, a weakly supervised framework for polyp localization using only video-level labels. WSPolypNet employs a 3D convolutional neural network trained with video-level labels to generate class activation maps (CAMs) that identify candidate polyp regions without frame-level spatial annotations. The CAM-derived localization cues are enhanced using a multi-view strategy and provided to MedSAM2 as point prompts. These prompts propagate segmentation masks across the video, refining the coarse localization according to polyp boundaries. WSPolypNet achieved CorLoc scores of 47.80%, 43.68%, and 35.01% at IoU thresholds of 0.3, 0.5, and 0.7, respectively, compared with 36.87%, 33.72%, and 27.94% for the single-view setting. For small polyps, multi-view localization improved CorLoc@0.5 from 16.01% to 30.97%. The framework also achieved a recall of 94.51%. These results demonstrate the[ potential of weakly supervised spatiotemporal learning to reduce spatial annotation requirements in colonoscopy videos.

Index Terms—polyp localization, weakly supervised learning, video-level supervision, class activation map, MedSAM2

## I. INTRODUCTION

Colorectal cancer (CRC) is one of the most common and deadly cancers worldwide, accounting for approximately 2 million new cases and nearly 1 million deaths annually. The global burden of CRC is expected to continue increasing, with the number of new cases projected to exceed 3.2 million per year by 2050 [1]. Because a substantial proportion of CRC cases arise from adenomatous polyps, detecting and removing these precancerous lesions before malignant progression is critically important. Winawer et al. [2] reported that early detection and removal of adenomatous polyps could reduce the expected incidence of CRC by up to 90%. Furthermore, the removal of adenomas has been associated with a 53% reduction in CRC-related mortality compared with that expected in the general population [3].

Colonoscopy is widely used as a primary screening and diagnostic procedure for detecting and removing precancerous polyps [1], [4]. Nevertheless, conventional colonoscopy still has several limitations. Previous studies have reported that as many as 20%–26% of polyps may be missed during routine colonoscopic examinations [5]. In addition, the complex structure of the intestinal wall, the limited field of view, and visual disturbances in endoscopic images can make reliable polyp detection challenging. Consequently, computer-aided detection (CADe) systems have been extensively investigated to assist clinicians and improve the consistency of polyp detection.

Despite these advances, many existing polyp detection methods process frames extracted from endoscopic videos as independent static images. These image-based approaches do not explicitly model temporal correlations between consecutive frames and may therefore produce temporally inconsistent predictions in real-world colonoscopy videos [6]. In particular, complex intestinal structures and visual patterns resembling polyps can make it difficult to distinguish polyps from the surrounding tissue using a single frame. Therefore, reliable and consistent polyp localization in colonoscopy videos requires both spatial information and temporal modeling [7].

Meanwhile, conventional supervised polyp detection and segmentation methods that aim to achieve accurate spatial localization generally rely on detailed spatial annotations, such as bounding boxes or pixel-level masks. Providing such annotations for every frame of an endoscopic video requires substantial time and effort, making it difficult to scale to large video datasets. Moreover, pixel-level annotation can be affected by inter-observer variability because of the complex structure of the intestinal wall and ambiguous polyp boundaries. To reduce this annotation burden, weakly supervised learning approaches have received increasing attention. In particular, Weakly Supervised Video Object Localization (WSVOL) aims to localize objects in videos using coarse supervision, such as video-level labels, rather than dense frame-level spatial annotations [7], [8].

Previous studies have explored the use of video-level supervision to classify the presence of target objects or estimate their temporal locations within videos [7], [9]. In parallel, weakly supervised object localization methods have investigated spatial object localization using only imagelevel supervision [10]. However, spatially localizing polyps in colonoscopy videos using only video-level supervision remains relatively underexplored. Moreover, polyp appearance and viewpoint continuously change throughout endoscopic videos, while complex intestinal structures and various visual disturbances may cause weakly supervised localization models to activate irrelevant regions [11]. Therefore, a method is needed that can exploit spatiotemporal information to localize polyps reliably without dense spatial annotations.

To address these challenges, we propose a weakly supervised framework that simultaneously determines the presence of polyps at the video level and estimates their spatial locations in individual frames using only video-level labels. The proposed framework employs a 3D convolutional neural network (3D CNN) to learn spatiotemporal representations from consecutive endoscopic frames and generates class activation maps (CAMs) [10], [12] to identify candidate polyp regions without additional spatial annotations. The CAM-derived localization cues are subsequently provided to MedSAM2 [13] as prompts to refine the coarse regions to better align with polyp boundaries. Thus, the proposed framework is designed to estimate both polyp presence and spatial location using only video-level supervision, without requiring frame-level bounding-box or pixel-level annotations.

The main contributions of this study are summarized as follows:

• We propose a weakly supervised polyp localization framework that simultaneously determines polyp presence at the video level and performs spatial localization in individual frames using only video-level labels.

• We integrate a 3D CNN with CAMs to exploit spatiotemporal information from consecutive endoscopic frames and estimate candidate polyp regions without frame-level spatial annotations.

• We construct a localization refinement pipeline that uses CAM-derived localization information as prompts for MedSAM2 to refine coarse localization results according to polyp boundaries.

## II. RELATED WORK

## A. AI-based Polyp Detection and Localization

Deep learning has been widely applied to polyp detection and segmentation in endoscopic images. Representative approaches include real-time CNN-based detection, multi-scale feature fusion and attention mechanisms, and Transformerbased architectures [14]–[16].

However, most existing methods process individual frames independently and often rely on strong spatial supervision, such as bounding boxes or pixel-level masks. Such frame-wise approaches do not fully exploit temporal information across consecutive frames and require substantial annotation effort for long endoscopic videos.

B. Weakly Supervised Object and Video Localization

Weakly Supervised Object Localization (WSOL) aims to estimate object locations using image-level labels without dense spatial annotations. Representative methods include CAM [10], Grad-CAM [12], approaches that discover complementary object regions [11], [17], and Transformer-based localization methods [18], [19].

Weakly supervised localization has also been extended to video using video-level supervision and temporal information, including temporal CAM-based methods, weakly supervised surgical video localization, and Transformer-based video object localization [7], [9], [20].

However, CAM-based localization often focuses on only the most discriminative object regions, resulting in incomplete or spatially coarse localization [11], [17]. This limitation is particularly problematic for small polyps, while spatial localization of polyps in colonoscopy videos using only videolevel supervision remains relatively underexplored.

## C. Promptable Medical Image and Video Segmentation

Promptable segmentation models such as SAM [21] generate dense masks from sparse spatial prompts. SAM 2 [22] extends this capability to video, while MedSAM2 [13] adapts it to medical image and video analysis.

Because these models require an initial spatial prompt, they are complementary to weakly supervised localization. In our framework, CAM-derived candidate locations are used as point prompts for MedSAM2, which refines the coarse localization into segmentation masks without requiring frame-level spatial annotations for the target training videos.

## III. PROPOSED METHOD

## A. Overall Framework

The proposed framework performs weakly supervised polyp localization in colonoscopy videos using video-level labels. Given a preprocessed video clip, a 3D convolutional neural network (3D CNN) first extracts spatiotemporal representations and predicts the presence of a polyp at the video level. Class activation maps (CAMs) are then derived from the trained classifier to obtain coarse spatial localization cues without requiring frame-level bounding-box or pixel-level mask annotations.

The CAM-derived localization cues are further enhanced using a multi-view strategy and converted into candidate point prompts for MedSAM2. MedSAM2 propagates the resulting segmentation masks bidirectionally across the video sequence, after which the candidate tracks are evaluated to determine the final localization result. The overall framework is illustrated in Fig. 1.

## B. ROI Preprocessing

The positive and negative video sources exhibit substantial differences in image resolution and peripheral visual appearance, including black borders and interface overlays. Such source-specific cues may introduce visual bias unrelated to polyp appearance. To mitigate this issue, we apply field-ofview (FOV)-based region-of-interest (ROI) preprocessing to all input frames.

![](images/574c25636e9fe13704971dabf83d01d9d5a1d8697aab139ed5e5cd6688f10632.jpg)  
Fig. 1: Overall framework of the proposed method. Given an input video clip, a 3D CNN predicts video-level polyp presence and generates class activation maps (CAMs), from which candidate points are derived. Each point is independently provided to MedSAM2 as a prompt, which propagates the resulting mask across the clip to produce the final localization.

For each video, the effective endoscopic FOV is estimated using temporal brightness voting over frames uniformly sampled across the temporal dimension. The detected FOV is represented by its convex hull, after which the bounding region enclosing the hull is cropped and pixels outside the hull are masked to zero. When substantial variations in FOV shape are observed across temporally separated sampled frames, the union of the detected regions is used to avoid excluding valid endoscopic areas.

Because fixed background patterns and ROI boundaries may themselves serve as source-specific cues, additional spatial randomization is applied during training. The extracted ROI is randomly resized to 90–100% of the training canvas while preserving its aspect ratio and is placed at a randomly selected valid position. This augmentation reduces the likelihood that the classifier relies on consistent padding patterns or ROI locations rather than polyp-related visual features. During evaluation, random scaling is disabled and the ROI is deterministically placed at the center of the canvas.

This preprocessing removes irrelevant peripheral regions while preserving the effective endoscopic field of view and visible mucosal area as much as possible. Fig. 2 shows a representative clip before and after ROI preprocessing.

## C. Spatiotemporal Feature Learning and CAM Generation

Let an input video clip be denoted by $\mathbf { X } \in \mathbb { R } ^ { C \times T \times H \times W }$ The clip is fed into a 3D CNN to extract spatiotemporal feature representations from consecutive endoscopic frames. The final convolutional feature tensor is denoted by ${ \textbf { F } } \in$ $\mathbb { R } ^ { K \times T ^ { \prime } \times H ^ { \prime } \times W ^ { \prime } }$ , where K is the number of feature channels and $T ^ { \prime } , H ^ { \prime } ,$ , and $W ^ { \prime }$ denote the temporal and spatial dimensions of the feature tensor, respectively.

For video-level polyp classification, global average pooling (GAP) is applied to F, and the resulting feature vector is passed through a binary classification layer that produces a single logit. The classifier is trained using a video-level binary label $y \in \{ 0 , 1 \}$ and binary cross-entropy with logits loss (BCEWithLogitsLoss). During inference, the output logit is converted to a polyp-presence probability using a sigmoid function.

![](images/b462f4339386a112994754e65a15a37be9bfc624e2ebb5f6753b096b094e0e5a.jpg)  
Fig. 2: Effect of FOV-based ROI preprocessing on a representative clip. Top: original frames. Bottom: ROI-preprocessed frames.

To obtain spatial localization cues from the trained classifier, a class activation map (CAM) is computed from the final convolutional feature tensor. Let $w _ { k }$ denote the classification weight associated with the k-th feature channel. The CAM is defined as

$$
M ( t , u , v ) = \sum _ { k = 1 } ^ { K } w _ { k } F _ { k } ( t , u , v ) ,\tag{1}
$$

where $F _ { k } ( t , u , v )$ denotes the activation of the k-th feature channel at temporal index t and spatial location $( u , v )$ . The resulting class-specific response map indicates the contribution of each spatiotemporal location to the polyp-present prediction and serves as the initial localization cue for the subsequent refinement stage.

## D. MedSAM2-based Localization Refinement

To improve localization, particularly for small polyps, the CAM evidence of each frame is aggregated over five views: the full 224×224 model input and four overlapping corner crops. Each corner crop is a 144×144 window anchored at one corner of the input (covering 64% of each side, so that adjacent crops overlap in a 64-px-wide central band). Every crop is bilinearly resized back to 224×224 — a ≈1.56× magnification — and passed through the same frozen classifier, so that polyps near the field-of-view boundary, which the full view under-resolves on the coarse feature grid, are enlarged before the CAM is computed. A centered crop was also examined but gave no improvement and is not used. For each view $q ,$ let $A _ { t } ^ { ( q ) } ( u , v )$ denote the raw CAM response at frame t. The corresponding localization score map is computed as

$$
L _ { t } ^ { ( q ) } ( u , v ) = S _ { t } ^ { ( q ) } ( u , v ) \cdot \sigma \left( A _ { t } ^ { ( q ) } ( u , v ) \right) ,\tag{2}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and $S _ { t } ^ { ( q ) } ( u , v )$ denotes the spatial weight obtained by applying softmax over the spatial locations of the raw CAM response. This formulation combines the relative spatial importance of each location with its activation strength.

The localization maps from the five views are mapped back to the coordinate system of the original frame and fused using a pixel-wise maximum operation:

$$
L _ { t } ^ { \mathrm { f u s e d } } ( u , v ) = \operatorname* { m a x } _ { q \in \{ 1 , \ldots , 5 \} } L _ { t } ^ { ( q ) } ( u , v ) .\tag{3}
$$

For each frame, the peak response of the fused localization map is defined as

$$
s _ { t } = \operatorname* { m a x } _ { u , v } L _ { t } ^ { \mathrm { f u s e d } } ( u , v ) ,\tag{4}
$$

and the seed frame is selected as

$$
t ^ { * } = \arg \operatorname* { m a x } _ { t } s _ { t } .\tag{5}
$$

The fused localization map of the seed frame $t ^ { * }$ is min– max normalized to [0, 1] and then Gaussian smoothed with $\sigma ~ = ~ 4 ~ \mathrm { p x }$ in the 224 × 224 input space. Non-maximum suppression is applied greedily: the current maximum is taken as a candidate point, all locations within a radius of 16 px are suppressed, and the step repeats until five candidate points are obtained, ordered by decreasing response.

Each candidate point is independently provided to Med-SAM2 as a positive point prompt. MedSAM2 generates an initial segmentation mask at the seed frame and propagates the mask bidirectionally to the preceding and subsequent frames, producing one candidate segmentation track for each point.

The candidate tracks are evaluated using the CAM-derived fused localization responses. For each frame, the mean fused localization response is computed within the bounding region enclosing the predicted mask, and the track score is obtained by averaging these frame-level responses over the video sequence. The first-ranked candidate is selected by default, while another candidate replaces it only when its track score exceeds that of the first-ranked candidate by more than 20%.

For a given frame, the propagated mask is replaced by an independent frame-wise prediction when either (i) the propagated mask is empty, or (ii) its confidence falls below a threshold $\tau _ { \mathrm { c o n f } } ~ = ~ 0 . 7 5 5$ , where the per-frame confidence is the mean sigmoid probability over the foreground pixels of the mask. Condition (ii) covers tracking drift, in which the seed frame is localized correctly but the target is lost on other frames of the clip. The frame-wise fallback prediction is obtained by prompting MedSAM2 with a single positive point at that frame’s own fused-CAM peak, without using temporal information. Through this refinement process, the coarse CAM-based localization cues are converted into segmentation masks with more precise spatial boundaries.

## IV. EXPERIMENTS

## A. Dataset

For video-level binary classification, we constructed a training dataset consisting of positive and negative video clips. Positive samples were generated from 100 frame sequences extracted from the LDPolypVideo dataset. The frames in each sequence were divided into non-overlapping clips of up to 30 consecutive frames while preserving their original temporal order. This procedure resulted in a total of 871 positive video clips.

Negative samples were generated from 60 original endoscopic videos without polyps. Each negative clip was constructed from 30 temporally consecutive, non-overlapping frames extracted from a single source video. Frames originating from different source videos were not combined within the same clip. This procedure yielded a total of 615 negative video clips.

The final training dataset consisted of 871 positive clips and 615 negative clips, yielding a total of 1,486 video clips. The output frame rate of all generated clips was set to 1 FPS. In addition, to reduce source-specific visual bias arising from differences in image resolution and peripheral appearance between the positive and negative data sources, the ROI preprocessing described in Section III-B was applied to all input frames.

## B. Implementation Details

All 3D CNN backbones were initialized with weights pretrained on Kinetics-400 and trained under the same optimization settings. The models were optimized using AdamW with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of 0.01. The backbone was frozen for the first five epochs and subsequently unfrozen for end-to-end fine-tuning. Training was performed for a total of 50 epochs using a cosine learningrate schedule.

![](images/82176ccdd5a64984c90cc233a469bcfd6160776e4e04c280884c1fc36c34a78b.jpg)  
Fig. 3: Qualitative localization comparison on representative test frames. Green and yellow boxes denote ground truth and predictions, respectively. Columns compare five 3D CNN backbones, the single-view setting, and the full WSPolypNet framework.

## C. Evaluation Metrics

Localization performance was evaluated using Correct Localization (CorLoc). A localization result was considered correct when the intersection over union (IoU) between the predicted and ground-truth bounding boxes exceeded a predefined threshold. CorLoc was evaluated at IoU thresholds of 0.3, 0.5, and 0.7. Recall was additionally reported to evaluate the ability of each method to identify positive samples.

For the analysis of the multi-view strategy, polyps were divided into small and large groups based on the ratio of the ground-truth bounding-box area to the total frame area. Polyps with an area ratio of 0.05 or less were categorized as small, whereas those with an area ratio greater than 0.05 were categorized as large.

## D. Experimental Results

To compare the localization capability of different 3D CNN backbones, five architectures were evaluated under the same training and evaluation settings. Table I summarizes their Recall and CorLoc performance. Representative qualitative results are shown in Fig. 3.

Among the evaluated backbones, X3D achieved the highest localization performance, with CorLoc scores of 21.03%, 9.32%, and 1.63% at IoU thresholds of 0.3, 0.5, and 0.7, respectively. Although R(2+1)D-18 achieved the highest Recall,

TABLE I: Performance comparison of different 3D CNN backbones.
<table><tr><td>Backbone</td><td>Recall</td><td>CorLoc@0.3</td><td>CorLoc@0.5</td><td>CorLoc@0.7</td></tr><tr><td>Slow R50</td><td>80.22</td><td>18.94</td><td>7.87</td><td>1.05</td></tr><tr><td>X3D</td><td>74.18</td><td>21.03</td><td>9.32</td><td>1.63</td></tr><tr><td>SlowFast R50</td><td>69.23</td><td>13.57</td><td>3.63</td><td>0.59</td></tr><tr><td>R3D-18</td><td>91.21</td><td>16.75</td><td>6.53</td><td>0.82</td></tr><tr><td>R(2+1)D-18</td><td>94.51</td><td>12.72</td><td>2.78</td><td>0.11</td></tr></table>

TABLE II: Localization performance of the proposed framework.
<table><tr><td>Method</td><td>Recall</td><td>CorLoc@0.3</td><td>CorLoc@0.5</td><td>CorLoc@0.7</td></tr><tr><td>Single-view</td><td>74.18</td><td>36.87</td><td>33.72</td><td>27.94</td></tr><tr><td>WSPolypNet</td><td>94.51</td><td>47.80</td><td>43.68</td><td>35.01</td></tr></table>

X3D was selected as the backbone of WSPolypNet because the primary objective of this study is spatial polyp localization.

After selecting X3D as the backbone, CAM-based localization was combined with MedSAM2 refinement to construct the complete WSPolypNet framework. Table II compares the single-view configuration with the proposed framework.

WSPolypNet achieved CorLoc scores of 47.80%, 43.68%, and 35.01% at IoU thresholds of 0.3, 0.5, and 0.7, respectively, compared with 36.87%, 33.72%, and 27.94% for the singleview configuration. WSPolypNet also achieved a Recall of 94.51%, indicating improved detection sensitivity compared with the single-view configuration.

TABLE III: Comparison of single-view and multi-view localization performance for large and small polyps.
<table><tr><td>Polyp Size</td><td>Setting</td><td>CorLoc@0.3</td><td>CorLoc@0.5</td><td>CorLoc@0.7</td></tr><tr><td rowspan="2">Large</td><td>Single-view</td><td>74.67%</td><td>71.84%</td><td>65.60%</td></tr><tr><td>Multi-view</td><td>67.20%</td><td>65.75%</td><td>62.34%</td></tr><tr><td rowspan="2">Small</td><td>Single-view</td><td>19.31%</td><td>16.01%</td><td>10.45%</td></tr><tr><td>Multi-view</td><td>35.39%</td><td>30.97%</td><td>20.29%</td></tr></table>

To further analyze the effect of multi-view localization with respect to polyp size, we compared the single-view and multiview configurations separately for large and small polyps. Table III summarizes the results.

The multi-view strategy substantially improved localization performance for small polyps. In particular, CorLoc at an IoU threshold of 0.5 increased from 16.01% to 30.97%. In contrast, for large polyps, CorLoc at the same threshold decreased from 71.84% to 65.75%. These results indicate that the proposed multi-view strategy is particularly effective for improving small-polyp localization, although it introduces a trade-off in the localization performance of large polyps.

## V. CONCLUSION

We proposed WSPolypNet, a weakly supervised framework that combines X3D-based CAM localization with MedSAM2 refinement for polyp localization in colonoscopy videos. WSPolypNet achieved a CorLoc of 43.68% at an IoU thresh old of 0.5, while multi-view localization improved small-polyp CorLoc from 16.01% to 30.97%. A current limitation is that the framework selects and propagates a single high-confidence point prompt through MedSAM2, which restricts localization to one polyp even when multiple polyps are present in the same video. Although the current framework is more suitable for annotation assistance than complete annotation replacement, future work will focus on improving localization accuracy, supporting multiple simultaneous polyps, and extending the method to longer colonoscopy videos in which polyps appear only briefly.

## REFERENCES

[1] F. Bray, M. Laversanne, H. Sung, J. Ferlay, R. L. Siegel, I. Soerjomataram, and A. Jemal, “Global cancer statistics 2022: Globocan estimates of incidence and mortality worldwide for 36 cancers in 185 countries,” CA: a cancer journal for clinicians, vol. 74, no. 3, pp. 229– 263, 2024.

[2] S. J. Winawer, A. G. Zauber, M. N. Ho, M. J. O’Brien, L. S. Gottlieb, S. S. Sternberg, J. D. Waye, M. Schapiro, J. H. Bond, J. F. Panish et al., “Prevention of colorectal cancer by colonoscopic polypectomy,” New England Journal of Medicine, vol. 329, no. 27, pp. 1977–1981, 1993.

[3] A. G. Zauber, S. J. Winawer, M. J. O’Brien, I. Lansdorp-Vogelaar, M. van Ballegooijen, B. F. Hankey, W. Shi, J. H. Bond, M. Schapiro, J. F. Panish et al., “Colonoscopic polypectomy and long-term prevention of colorectal-cancer deaths,” New England Journal of Medicine, vol. 366, no. 8, pp. 687–696, 2012.

[4] D. A. Corley, C. D. Jensen, A. R. Marks, W. K. Zhao, J. K. Lee, C. A. Doubeni, A. G. Zauber, J. De Boer, B. H. Fireman, J. E. Schottinger et al., “Adenoma detection rate and risk of colorectal cancer and death,” New england journal of medicine, vol. 370, no. 14, pp. 1298–1306, 2014.

[5] A. M. Leufkens, M. G. Van Oijen, F. P. Vleggaar, and P. D. Siersema, “Factors influencing the miss rate of polyps in a back-to-back colonoscopy study,” Endoscopy, vol. 44, no. 05, pp. 470–475, 2012.

[6] G.-P. Ji, Y.-C. Chou, D.-P. Fan, G. Chen, H. Fu, D. Jha, and L. Shao, “Progressively normalized self-attention network for video polyp segmentation,” in International conference on medical image computing and computer-assisted intervention. Springer, 2021, pp. 142–152.

[7] S. Belharbi, I. B. Ayed, L. McCaffrey, and E. Granger, “Tcam: Temporal class activation maps for object localization in weakly-labeled unconstrained videos,” in 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2023, pp. 137–146.

[8] M. P. Kumar and B. S. Murugan, “Explainable weakly supervised deep learning model for anomaly classification and localization in wireless capsule endoscopy images,” International Journal of Imaging Systems and Technology, vol. 35, no. 3, p. e70121, 2025.

[9] G. Liao, M. Jogan, S. Koushik, E. Eaton, and D. A. Hashimoto, “Disentangling spatio-temporal knowledge for weakly supervised object detection and segmentation in surgical video,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 8013–8023.

[10] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning deep features for discriminative localization,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2921–2929.

[11] J. Choe and H. Shim, “Attention-based dropout layer for weakly supervised object localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2219– 2228.

[12] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.

[13] J. Ma, Z. Yang, S. Kim, B. Chen, M. Baharoon, A. Fallahpour, R. Asakereh, H. Lyu, and B. Wang, “Medsam2: Segment anything in 3d medical images and videos,” arXiv preprint arXiv:2504.03600, 2025.

[14] G. Urban, P. Tripathi, T. Alkayali, M. Mittal, F. Jalali, W. Karnes, and P. Baldi, “Deep learning localizes and identifies polyps in real time with 96% accuracy in screening colonoscopy,” Gastroenterology, vol. 155, no. 4, pp. 1069–1078, 2018.

[15] Y. Xie, Y. Yu, M. Liao, and C. Sun, “Gastric polyp detection module based on improved attentional feature fusion,” BioMedical Engineering OnLine, vol. 22, p. 68, 2023.

[16] W. Wang, X. Yang, and J. Tang, “Vision transformer with hybrid shifted windows for gastrointestinal endoscopy image classification,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 9, pp. 4452–4461, 2023.

[17] X. Zhang, Y. Wei, J. Feng, Y. Yang, and T. S. Huang, “Adversarial complementary learning for weakly supervised object localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 1325–1334.

[18] W. Gao, F. Wan, X. Pan, Z. Peng, Q. Tian, Z. Han, B. Zhou, and Q. Ye, “Ts-cam: Token semantic coupled attention map for weakly supervised object localization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 2886–2895.

[19] S. Gupta, S. Lakhotia, A. Rawat, and R. Tallamraju, “Vitol: Vision transformer for weakly supervised object localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022, pp. 4101–4110.

[20] S. Murtaza, M. Pedersoli, A. Sarraf, and E. Granger, “Leveraging transformers for weakly supervised object localization in unconstrained videos,” arXiv preprint arXiv:2407.06018, 2024.

[21] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4015–4026.

[22] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala,¨ N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer,´ “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.