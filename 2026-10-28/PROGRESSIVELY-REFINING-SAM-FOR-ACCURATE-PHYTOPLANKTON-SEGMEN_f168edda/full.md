# PROGRESSIVELY REFINING SAM FOR ACCURATE PHYTOPLANKTON SEGMENTATION

Jialin Li Jinkun Jiang Yuezun Li<sup>∗</sup> Junyu Dong

School of Computer Science and Technology, Ocean University of China, Qingdao, China

## ABSTRACT

Accurate phytoplankton segmentation is essential for ecological monitoring, yet it remains largely underexplored due to the subtle visual characteristics of phytoplankton and the scarcity of pixel-level annotations. To address these challenges, we propose PhySeg, a SAM-based segmentation framework equipped with a lightweight refinement module that progressively restores fine-grained details through multiscale feature fusion. In addition, we construct a phytoplankton segmentation dataset by manually annotating instancelevel masks for an existing dataset, providing a valuable resource for future research. Experimental results demonstrate that PhySeg improves segmentation accuracy, validating the effectiveness of the proposed module. Code is available at https://github.com/lijialin404/PhySeg.

Index Terms— Phytoplankton Image Segmentation, Segment Anything Model, Refinement

## 1. INTRODUCTION

Phytoplankton underpin marine primary production and food webs and play an important role in the global carbon cycle [1]. Characterizing their morphology and behavior is therefore essential for understanding marine ecosystems. Several studies have focused on phytoplankton classification [2, 3], supported by public benchmarks such as WHOI-Plankton [4]. Beyond classification, detection and tracking methods [5, 6, 7] enable the spatial localization and temporal association of individual organisms, supported by datasets such as PMID2019 [8] and PMOT2023 [9].

Despite this progress, category labels, bounding boxes, and trajectories provide limited information about the precise morphology of individual organisms. Therefore, pixel-level segmentation for phytoplankton is required to support more detailed ecological analysis [10]. However, phytoplankton segmentation remains comparatively underexplored, with two major challenges. First, general-purpose segmentation models, including the Segment Anything Model (SAM) [11], are not specifically designed for phytoplankton. Substantial variations in morphology and scale, together with slender structures, low contrast, and ambiguous boundaries (see examples in Fig. 3), make it difficult to recover fine details reliably [2, 8, 10]. Second, adapting these models requires highquality segmentation annotations, which remain scarce in existing phytoplankton datasets. Recent surveys likewise identify limited annotated data and the need for segmentationoriented methods as persistent challenges in phytoplankton image analysis [2, 3].

In this paper, we address these challenges through two contributions. 1) Methodology. We introduce a dedicated phytoplankton segmentation framework built on SAM2 [12], named PhySeg. Drawing on its strong general-purpose segmentation capabilities, we adapt SAM2 to phytoplankton microscopy and develop an add-on refiner that integrates multiscale features to progressively recover fine structural details. 2) Dataset. We extend PMID2019 with 11990 manually annotated instance masks across 9824 images. Unlike general objects, annotating phytoplankton requires substantial manual effort due to their slender structures and intricate boundaries. Despite its modest scale, this dataset provides a valuable resource for developing and evaluating phytoplankton segmentation methods. Experiments on this dataset demonstrate the effectiveness of the proposed framework, establishing a baseline for phytoplankton segmentation.

## 2. METHOD

PhySeg is built upon SAM2 and introduces a lightweight refiner to progressively recover fine-grained structures. Specifically, given an input image and a box prompt, SAM2 first produces coarse logits, which are then progressively refined by fusing them with multi-scale features. To adapt SAM2 to phytoplankton segmentation, we apply LoRA [13] to the image encoder. The overall architecture is illustrated in Fig. 1.

Progressive Refinement. In SAM2, coarse logits are directly upsampled and thresholded to obtain the final mask. However, upsampling alone does not explicitly correct errors in the coarse predictions. This naturally raises a question: can we refine the coarse predictions during upsampling?

Motivated by this idea, a Progressive Refinement strategy is introduced, which refines coarse logits through multiple steps, rather than relying on a single step to address various errors. Each step consists of two small convolutional modules, named Feature Fusion and Residual Head. The Feature Fusion module aggregates the input features at the current step, while the Residual Head module takes the fused features as input and predicts the residual logits. The residual logits are added to the coarse logits step by step to progressively refine the results.

![](images/a01170ad9ed595a4120bbf00fb238c1628aa14ad960cb79cc47d69a52a9b6c7f.jpg)  
Fig. 1. Detailed structure of the proposed PhySeg.

![](images/96d18b640ba78f435e40ac60d82608df477b7a5e10c15a113b1efc78100bb022.jpg)  
Fig. 2. Overview of the Feature Fusion module. $\mathrm { C G G } _ { k \times k } ^ { d }$ denotes a k × k convolution with dilation rate d, followed by GroupNorm and GELU. The dilation rate is 1 when omitted.

Structure Design. SAM2 inherently encodes semantic and spatial information at multiple scales. We therefore reorganize these intermediate features as complementary cues for mask refinement. Specifically, we use three feature maps from the decoding process, denoted as $F _ { 1 6 } , F _ { 8 }$ , and $F _ { 4 }$ , with spatial resolutions of 1/16, 1/8, and $1 / 4$ of the input image, respectively.

While $F _ { 1 6 }$ provides rich semantic information, its limited spatial detail compromises the initial representation on which subsequent corrections rely. We therefore introduce $F _ { 8 }$ at Step 0 to incorporate finer spatial cues from the outset, providing a stronger foundation for subsequent refinement. At Step 1, we introduce $F _ { 4 }$ for finer spatial details while retaining $F _ { 8 }$ to provide complementary structural context. After this feature transition in the first two steps, Step 2 directly uses $F _ { 4 }$ to further handle finer local errors. In this way, the refinement process gradually shifts from correcting the overall structure to improving local details.

In addition, the updated logits at each step, together with the fused features, serve as the hidden state $H _ { i }$ , carrying forward the refined features and current mask prediction and guiding the next step to achieve a balance between area and boundaries. At Step 0, an additional Feature Fusion module aggregates context from the input features before they are combined with the coarse logits to construct the initial hidden state $H _ { 0 }$ . Besides, the shallow encoder feature $S _ { 0 }$ is introduced at every step to provide local spatial details.

As shown in Fig. 2, inspired by the ASPP module [14], we design the Feature Fusion module to address the need for both local and global information during refinement. Input features are first aligned via Region of Interest (ROI) and then processed by three parallel convolutional branches with different dilation rates to capture local information, while Global Average Pooling is added to provide a shared global reference for refinement at different spatial locations. The fused features are further fed into the Residual Head R, which consists of a 3 × 3 convolution followed by a 1 × 1 convolution, to predict the residual logits for correction at each step.

Training Objective. The refined prediction is supervised using a combination of focal loss [15] and Dice loss [16]: ${ \mathcal { L } } =$ $\mathcal { L } _ { \mathrm { F o c a l } } + \mathcal { L } _ { \mathrm { D i c e } }$

## 3. PHYTOPLANKTON SEGMENTATION DATASET

Few public datasets provide pixel-level annotations for phytoplankton segmentation. To address this gap, we construct the Phytoplankton Segmentation Dataset (PhySegD) by manually annotating ground-truth masks for the PMID2019 dataset.

Creating PhySegD is particularly labor-intensive despite its moderate scale. In bright-field microscopy images, halos surrounding phytoplankton are often difficult to distinguish from the boundaries of semi-transparent cells, necessitating multiple rounds of expert review. Moreover, impurities with textures similar to those of phytoplankton may interfere with or obscure cell boundaries, further complicating accurate mask annotation. Representative samples from different genus-level categories and their annotations are shown in Fig. 3, while Table 1 reports the numbers of images and instances across the 15 categories. Overall, PhySegD contains 9824 images and 11990 instances, divided into 6880 training images with 8378 instances, 1968 validation images with 2395 instances, and 976 test images with 1217 instances.

![](images/533d04071f2cbabf739a6e4371ac346e5a5888aa170fc270dfbe1606a5a1200a.jpg)  
Fig. 3. Visual examples of the phytoplankton segmentation dataset.

Table 1. Category statistics of the dataset.
<table><tr><td>ID</td><td>Genus</td><td>Images</td><td>Instances</td></tr><tr><td>1</td><td>bacteriastrum</td><td>677</td><td>703</td></tr><tr><td>2</td><td>biddulphia</td><td>356</td><td>360</td></tr><tr><td>3</td><td>ceratium</td><td>2267</td><td>2332</td></tr><tr><td>4</td><td>chaetoceros</td><td>1095</td><td>1184</td></tr><tr><td>5</td><td>coscinodiscus</td><td>1123</td><td>1140</td></tr><tr><td>6</td><td>dinophysis</td><td>413</td><td>417</td></tr><tr><td>7</td><td>ditylum</td><td>99</td><td>100</td></tr><tr><td>8</td><td>eucampia</td><td>1446</td><td>1514</td></tr><tr><td>9</td><td>guinardia</td><td>857</td><td>869</td></tr><tr><td>10</td><td>helicotheca</td><td>369</td><td>374</td></tr><tr><td>11</td><td>pleurosigma</td><td>228</td><td>237</td></tr><tr><td>12</td><td>protoperidinium</td><td>105</td><td>105</td></tr><tr><td>13</td><td>rhizosolenia</td><td>77</td><td>77</td></tr><tr><td>14</td><td>skeletonema</td><td>1053</td><td>1141</td></tr><tr><td>15</td><td>thalassionema</td><td>1389</td><td>1437</td></tr></table>

## 4. EXPERIMENTS

Experimental Setup. We evaluate PhySeg on the proposed PhySegD dataset and compare it with representative maskrefinement methods, including CascadePSP [17], BPR [18], CRM [19], and HQ-SAM2 [20]. We report mean Intersection over Union (mIoU) and Dice scores to evaluate region accuracy, together with boundary F-scores at thresholds of 1, 2, and 4 pixels (BF1, BF2, and BF4) to assess boundary quality [21]. Higher values indicate better performance for all metrics. Among these metrics, mIoU serves as the primary measure of overall segmentation accuracy.

Implementation Details. During training, we apply random horizontal and vertical flips, rotations by multiples of 90<sup>◦</sup>, and color jittering. The same geometric transformations are applied consistently to the images, masks, and bounding boxes. Training is conducted in two stages. First, we adapt SAM2 to phytoplankton images using LoRA. The model is optimized for 15 epochs using AdamW with a learning rate of $2 \times 1 0 ^ { - 5 }$ a weight decay of 0.01, and a batch size of 1. Second, we freeze SAM2 and its LoRA parameters and train only the refiner for 15 epochs. We use AdamW with an initial learning rate of $5 \times 1 0 ^ { - 5 }$ , a weight decay of 0.01, and a batch size of 1. The learning rate follows a cosine annealing schedule with a minimum value of 10<sup>−6</sup>.

Comparison Results. Table 2 compares PhySeg with existing segmentation and refinement methods. The upper part with the superscript <sup>∗</sup> denotes models evaluated using their official pretrained weights without fine-tuning. The lower part reports the results of these models after fine-tuning on our dataset. For CRM, BPR, and CascadePSP, we fine-tune the official pretrained models using their recommended training configurations. For HQ-SAM2, we reimplement the HQ module ourselves, strictly following its instructions. PhySeg achieves the best mIoU, BF1, BF2, and Dice scores, reaching 91.87%, 68.32%, 84.02%, and 95.68%, respectively. It also outperforms the mask-based refinement methods despite requiring only a box prompt. Fig. 4 further shows that Phy-Seg better preserves slender structures and corrects inaccurate regions in the initial masks.

Table 2. Quantitative performance comparison of different methods on the test set. “Box” indicates that a bounding box is used as the model prompt, whereas “Mask” indicates that the SAM2+LoRA prediction is provided as the initial mask.
<table><tr><td>Method</td><td>Venue</td><td>Prompt</td><td>mIoU↑</td><td>Dice↑</td><td>BF1↑</td><td>BF2↑</td><td>BF4↑</td></tr><tr><td rowspan="5">CascadePSP* [17] BPR* [18] CRM* [19] SAM2* [12] HQ-SAM2* [20]</td><td>CVPR&#x27;20</td><td>Mask</td><td>87.08</td><td>92.77</td><td>53.45</td><td>72.41</td><td>90.57</td></tr><tr><td>CVPR&#x27;21</td><td>Mask</td><td>88.51</td><td>93.53</td><td>59.61</td><td>77.99</td><td>92.88</td></tr><tr><td>CVPR&#x27;22</td><td>Mask</td><td>68.20</td><td>76.08</td><td>34.94</td><td>49.04</td><td>65.39</td></tr><tr><td>ICLR’25</td><td>Box</td><td>81.03</td><td>87.82</td><td>42.86</td><td>60.29</td><td>80.46</td></tr><tr><td>NeurIPS&#x27;23</td><td>Box</td><td>82.99</td><td>89.35</td><td>47.57</td><td>65.83</td><td>84.64</td></tr><tr><td rowspan="4">CascadePSP [17] BPR [18] CRM [19]</td><td>CVPR&#x27;20</td><td>Mask</td><td>89.72</td><td>94.37</td><td>62.69</td><td>79.85</td><td>93.36</td></tr><tr><td>CVPR&#x27;21</td><td>Mask</td><td>91.73</td><td>95.60</td><td>67.79</td><td>83.51</td><td>94.93</td></tr><tr><td>CVPR&#x27;22</td><td>Mask</td><td>67.19</td><td>75.03</td><td>36.00</td><td>51.55</td><td>68.36</td></tr><tr><td>ICLR’22</td><td>Box</td><td>91.20</td><td>95.30</td><td>65.50</td><td>82.52</td><td>94.86</td></tr><tr><td>SAM2+LoRA [13] HQ-SAM2 [20]</td><td>NeurIPS&#x27;23</td><td>Box</td><td>91.73</td><td>95.60</td><td>67.52</td><td>83.71</td><td>95.31</td></tr><tr><td>PhySeg (Ours)</td><td></td><td>Box</td><td>91.87</td><td>95.68</td><td>68.32</td><td>84.02</td><td>95.19</td></tr></table>

![](images/cf15e1df98da243a8ccebfff124fefd1949654c068d8b811f862f5157c0796d9.jpg)  
(a)  
(b)  
(c)  
(d)  
(e)  
(f)  
Fig. 4. Visual examples of our method and others: (a) original image, (b) ground truth, (c) SAM2, (d) SAM2+LoRA, (e) HQ-SAM2, and (f) PhySeg (Ours).

BF1 and BF2 require the predicted boundary to be closer to the real boundary, while BF4 uses a wider tolerance. Compared with HQ-SAM2, our method achieves improvements in mIoU, Dice, BF1, and BF2, with a slight decrease of 0.12 percentage points in BF4. Overall, these results show that Phy-Seg achieves a better balance between boundary refinement and region correction, which does not contradict the fact that there is still room for improvement in BF4.

Ablation Study. Table 3 evaluates the contributions of the feature inputs and progressive refinement steps. H denotes the hidden state transferred across steps; $S _ { 0 }$ denotes the shallow feature extracted from Hiera Stage 0. As shown in Fig. 1, Steps 0, 1, and 2 denote the progressive procedure of refinement. We transfer H across steps to find a balance between area refinement and boundary refinement. As shown by the ablation study, adding H leads to an increase in mIoU and Dice while causing a slight decrease in BF, which is consistent with the design of H.

Table 3. Ablation study on the val-set.
<table><tr><td>Method</td><td colspan="4">mIoU↑ Dice↑ BF1↑ BF2↑ BF4↑</td></tr><tr><td>PhySeg</td><td>92.07</td><td>95.80 67.75</td><td>84.11</td><td>95.47</td></tr><tr><td>w/o  $H$ </td><td>91.97</td><td>95.74 67.92</td><td>84.35</td><td>95.61</td></tr><tr><td>w/o  $S _ { 0 }$ </td><td>92.03</td><td>95.77 67.65</td><td>83.95</td><td>95.49</td></tr><tr><td>w/o Step 0</td><td>92.00</td><td>95.76 67.45</td><td>84.04</td><td>95.55</td></tr><tr><td>w/o Step 0 &amp; Step 1</td><td>91.96</td><td>95.74 67.36</td><td>83.88</td><td>95.45</td></tr></table>

In addition, removing $S _ { 0 }$ , Step 0, or Step 0 and Step 1 consistently leads to performance degradation across multiple metrics. Since the original SAM2 already provides a relatively strong segmentation baseline, the resulting changes are not necessarily large in magnitude. Moreover, these components contain partially overlapping feature information, which may further limit the apparent gain contributed by each individual component. However, the overall trend observed in the ablation results is consistent with our design motivation: these features are introduced as refinement prompts to progressively guide the segmentation results.

## 5. CONCLUSION

In this work, we propose PhySeg, a phytoplankton segmentation method built upon SAM2 and equipped with a newly designed refiner module. By progressively fusing multi-scale features, the refiner recovers missing details and corrects inaccurate regions in the initial segmentation, thereby improving the delineation of fine phytoplankton structures. In addition, we construct a phytoplankton segmentation dataset by manually annotating ground-truth masks for the existing PMID2019 dataset. Experimental results demonstrate the effectiveness of the proposed method.

## 6. REFERENCES

[1] P. G. Falkowski, R. T. Barber, and V. Smetacek, “Biogeochemical controls and feedbacks on ocean primary production,” Science, 1998.

[2] T. Eerola et al., “Survey of automatic plankton image recognition: Challenges, existing solutions and future perspectives,” Artificial Intelligence Review, 2024.

[3] N. Taheriashtiani et al., “Artificial intelligence and microscopy: A systematic review of deep learning in microalgae and cyanobacteria species identification,” Algal Research, 2026.

[4] E. C. Orenstein, O. Beijbom, E. E. Peacock, and H. M. Sosik, “WHOI-Plankton: A large scale fine grained visual recognition benchmark dataset for plankton classification,” CoRR, 2015.

[5] Y. Huo, Q. Lv, and J. Dong, “Enhancing phytoplankton recognition through a hybrid dataset and morphological description-driven prompt learning,” Journal of Marine Science and Engineering, 2025.

[6] T. Zhang, J. Dong, H. Zhao, Q. Li, and X. Sun, “Lightweight phytoplankton detection network based on knowledge distillation,” Journal of Applied Sciences, 2020.

[7] Y. Yu, Q. Lv, Y. Li, Z. Wei, and J. Dong, “PhyTracker: An online tracker for phytoplankton,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[8] Q. Li, X. Sun, J. Dong, and S. Han, “Developing a microscopic image dataset in support of intelligent phytoplankton detection using deep learning,” ICES Journal ofMarine Science, 2020.

[9] J. Yu, Q. Lv, Y. Li, J. Dong, H. Zhao, and Q. Li, “PMOT2023: A large-scale multi-object tracking (MOT) dataset with application to phytoplankton observation,” Journal of Marine Science and Engineering, 2023.

[10] V. Sonnet, L. Guidi, C. B. Mouw, G. Puggioni, and S.- D. Ayata, “Length, width, shape regularity, and chain structure: Time series analysis of phytoplankton morphology from imagery,” Limnology and Oceanography, 2022.

[11] A. Kirillov et al., “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[12] N. Ravi et al., “SAM 2: Segment anything in images and videos,” in International Conference on Learning Representations, 2025.

[13] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022.

[14] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.

[15] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar,´ “Focal loss for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.

[16] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in 2016 Fourth International Conference on 3D Vision (3DV), 2016.

[17] H. K. Cheng, J. Chung, Y.-W. Tai, and C.-K. Tang, “CascadePSP: Toward class-agnostic and very highresolution segmentation via global and local refinement,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[18] C. Tang, H. Chen, X. Li, J. Li, Z. Zhang, and X. Hu, “Look closer to segment better: Boundary patch refinement for instance segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[19] T. Shen et al., “High quality segmentation for ultra highresolution images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[20] L. Ke et al., “Segment anything in high quality,” in Advances in Neural Information Processing Systems, 2023.

[21] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.