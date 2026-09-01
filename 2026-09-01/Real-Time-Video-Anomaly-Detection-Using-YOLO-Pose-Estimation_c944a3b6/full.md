# Real-Time Video Anomaly Detection Using YOLO Pose Estimation and CLIP-Based Semantic Scoring

Vanodhya G. Warnasooriya<sup>1</sup>, Amir Hajian<sup>1</sup>, Watchara Ruangsang<sup>2</sup>, and Supavadee Aramvith<sup>1</sup>

<sup>1</sup>Dept. of Electrical Engineering, Faculty of Engineering, Chulalongkorn University, Bangkok 10330, Thailand

<sup>2</sup>Media Technology Program, King Mongkut’s Univ. of Technology Thonburi, Bangkok 10150, Thailand

{vanowarna, amirhajian85}@gmail.com, watchara.ruan@kmutt.ac.th, supavadee.a@chula.ac.th

Abstract—We propose a lightweight two-stage framework for real-time video anomaly detection. The first stage employs YOLO v11n-pose to detect persons and extract seventeen skeletal keypoints in a single forward pass. The second stage encodes each cropped person region through CLIP ViT-B/32 and computes cosine similarity against predefined textual descriptions of anomalous behaviors. This architecture eliminates the need for optical flow, standalone pose estimators, and density-based scoring modules. Experiments on CUHK Avenue, ShanghaiTech Campus, and a custom indoor dataset collected at Chulalongkorn University demonstrate an end-to-end throughput of approximately 51 FPS on an NVIDIA Titan XP GPU, a 3.36× speedup over the multifeature baseline, while maintaining frame-level AUROC values of 89.26%, 70.26%, and 84.13%, respectively.

Index Terms—anomaly detection, video surveillance, CLIP, YOLO, pose estimation, real-time processing, text-prompt classification

## I. INTRODUCTION

The widespread deployment of closed-circuit television (CCTV) across urban and institutional environments has driven growing demand for intelligent video analytics [1]. Human operators monitoring multiple feeds are susceptible to attention fatigue, with effectiveness declining sharply within minutes [2], motivating automated systems for detecting falls, altercations, and abnormal postures.

Prior work on video anomaly detection spans reconstruction-based [3], [4], prediction-based [5], and attributebased [6] paradigms. Reiss and Hoshen [6] achieved strong accuracy by combining AlphaPose, FlowNet2, and CLIP features with GMM/kNN scoring, but the multi-component pipeline limits throughput to approximately 15 FPS. Two recent developments enable a simpler design: CLIP [7] provides zero-shot anomaly classification via natural-language prompts without task-specific anomaly labels, and YOLO v11n-pose [8] performs person detection with seventeen-keypoint regression in a single forward pass. Although VadCLIP [9] also adapts CLIP for video anomaly detection, it relies on learnable prompt tokens trained with weak supervision and is designed for offline analysis rather than real-time deployment.

This work targets person-level anomalies (specifically falling, lying on the floor, fighting, and sitting on the floor) in fixed indoor CCTV environments. The system is designed for deployment on a single desktop-class GPU (NVIDIA Titan XP) to sustain real-time throughput above 30 FPS on standard surveillance video feeds.

Building on these foundations, we present a two-stage framework in which YOLO v11n-pose handles detection and CLIP ViT-B/32 handles semantic anomaly scoring. Our contributions are as follows:

• A streamlined pipeline that replaces multi-feature density estimation with direct CLIP semantic scoring, yielding a 3.36× throughput gain over the prior baseline while maintaining comparable accuracy.

• A text-prompt-based classification mechanism that leverages CLIP’s zero-shot capability to identify and categorize multiple anomaly types without requiring labeled anomaly training data.

• Evaluation on three surveillance benchmarks and deployment on live CCTV feeds at Chulalongkorn University, confirming stable real-time operation.

## II. RELATED WORK

Video anomaly detection methods broadly follow reconstruction-based or prediction-based paradigms. Hasan et al. [3] trained convolutional autoencoders on normal video sequences and flagged high reconstruction error as anomalous. Ganokratanaa et al. [10] proposed a deep spatiotemporal translation network (DSTN) that fuses background-removed frames with optical flow for pixel-level localization. Morais et al. [11] introduced MPED-RNN, which learns normal motion regularity from skeleton trajectories using a multi-timescale recurrent architecture and detects anomalies as deviations from predicted joint positions. Reiss and Hoshen [6] combined pose (AlphaPose), velocity (FlowNet2), and deep features (CLIP) with GMM/kNN density estimation, achieving strong accuracy but at a throughput of only 15 FPS due to the multi-component pipeline.

Recent vision-language models offer a promising alternative. CLIP [7] maps images and text into a shared embedding space through contrastive pre-training, enabling zeroshot classification via natural-language prompts. Wu et al. [9] developed VadCLIP, which adapts CLIP for weakly supervised video anomaly detection using learnable prompts. Moriyama et al. [12] proposed a language-guided decision override that adapts CLIP for retraining-free anomaly detection. However, these methods typically operate offline and are not designed for real-time deployment. Our work shares the use of vision-language models with these approaches [6], [9], [12]; the contribution lies in system integration and architectural simplification, demonstrating that direct CLIP cosine similarity can replace multi-component density estimation for real-time person-level anomaly detection rather than introducing a fundamentally new scoring mechanism.

On the detection side, YOLO v11n-pose [8] performs person detection and seventeen-keypoint regression in a single forward pass using Cross-Stage Partial Networks with compact C3K2 blocks, providing a favorable accuracy-speed balance for real-time surveillance.

## III. PROPOSED METHOD

The proposed framework processes each video frame through two sequential stages: person detection with pose estimation, followed by CLIP-based semantic anomaly scoring. Fig. 1 illustrates the complete pipeline.

## A. Stage 1: Person Detection and Pose Estimation

Given an input frame $I _ { t } ~ \in ~ \mathbb { R } ^ { H \times W \times 3 }$ at time step t, the first stage applies YOLO v11n-pose to simultaneously detect all persons and regress their skeletal poses. For each detected person i, the model outputs a bounding box $b _ { i }$ $( x _ { i } , y _ { i } , w _ { i } , h _ { i } )$ with confidence $c _ { i }$ , alongside seventeen keypoints $\mathbf { k } _ { i } = \{ ( x _ { j } ^ { k } , y _ { j } ^ { k } , v _ { j } ^ { k } ) \} _ { j = 1 } ^ { 1 7 }$ following the COCO topology (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles). The keypoint visibility flags, $v _ { j } ^ { k }$ , indicate whether each joint is visible, partially occluded, or not detected. Fig. 2(a) shows an example with an overlaid keypoint skeleton and a bounding box.

Each detected person is cropped from the original frame using the bounding box coordinates with a padding margin $\delta = 1 0$ pixels to preserve contextual information:

$$
p _ { i } = I _ { t } [ y _ { i } - \delta : y _ { i } + h _ { i } + \delta , ~ x _ { i } - \delta : x _ { i } + w _ { i } + \delta ]\tag{1}
$$

producing a set of person crops $\{ p _ { i } \} _ { i = 1 } ^ { N _ { t } }$ , where $N _ { t }$ denotes the number of detections at time t. The padding value $\delta =$ 10 pixels was selected empirically on CU Indoor Anomaly frames (native resolution 640 × 480), where it corresponds to approximately 1.6% of the frame width and provides sufficient surrounding context for the downstream CLIP encoder without introducing excessive background clutter.

## B. Stage 2: CLIP-Based Semantic Anomaly Scoring

Each person crop $p _ { i }$ is resized to $2 2 4 \times 2 2 4$ pixels and encoded by the CLIP ViT-B/32 visual encoder into a normalized embedding $\mathbf { f } _ { i } ^ { \mathrm { i m g } } \in \mathbb { R } ^ { 5 1 2 }$ . A fixed set of $M = 4$ textual prompts describing anomalous behaviors (“A person lying on the floor”, “A person falling down”, “A person fighting with another person”, “A person sitting on the floor”) is similarly encoded into textual embeddings $\mathbf { \bar { f } } _ { i } ^ { \mathrm { t x t } } \in \mathbb { R } ^ { 5 1 2 }$ , computed once at initialization and cached.

The per-person anomaly score is the maximum cosine similarity across all prompts:

$$
s _ { i } ^ { t } = \operatorname* { m a x } _ { j \in \{ 1 , \dots , M \} } \mathbf { f } _ { i } ^ { \mathrm { i m g } } \cdot \mathbf { f } _ { j } ^ { \mathrm { t x t } }\tag{2}
$$

and the frame-level score aggregates across all detected persons:

$$
S _ { t } = \operatorname* { m a x } _ { i \in \{ 1 , . . . , N _ { t } \} } \ s _ { i } ^ { t }\tag{3}
$$

## C. Temporal Smoothing and Classification

Raw scores are smoothed with a one-dimensional Gaussian kernel $( \sigma = 5$ , half-window $w = 1 5 )$ to suppress transient fluctuations. The smoothed score $\hat { S } _ { t }$ is compared against a decision threshold $\theta ;$ a frame is classified as anomalous when $\hat { S } _ { t } ~ \geq ~ \theta$ . The optimal threshold $\theta ^ { * } ~ = ~ 0 . 7$ is selected by sweeping [0.1, 0.9] on a validation partition to maximize the AUROC.

## IV. EXPERIMENTAL SETUP

## A. Datasets

We evaluate on three datasets: CUHK Avenue [13] (37 videos, 30,652 frames from a fixed university pathway camera), ShanghaiTech Campus [14] (437 videos, 317,398 frames from 13 diverse campus scenes), and CU Indoor Anomaly, a custom dataset of 40 videos (17,798 frames; 11,841 normal, 5,957 anomalous) collected from CCTV cameras in four buildings at Chulalongkorn University, with framelevel annotations for four anomaly categories (falling, lying, sitting on floor, fighting). The dataset is partitioned into 33 training and 7 test videos.

## B. Implementation Details

YOLO v11n-pose is initialized from COCO-pretrained weights and fine-tuned on 2,864 training images (augmented from 1,154 manually labeled frames) from CU Indoor Anomaly for 500 epochs (640 × 640 input, batch 16, SGD with lr = 0.01) on an NVIDIA Titan XP GPU. For CUHK Avenue and ShanghaiTech, the same fine-tuned model is used; COCO pretraining provides sufficient generalization for person detection across diverse scenes, and the CU Indoor fine-tuning primarily improves bounding-box precision for indoor poses. CLIP ViT-B/32 is used with pre-trained weights without fine-tuning; input crops are resized to $2 2 4 \times 2 2 4$ following the standard CLIP preprocessing. We note that the term “zero-shot” in this work refers specifically to the CLIP-based anomaly scoring component, which requires no labeled anomaly examples; the YOLO detector is supervised for person localization.

## C. Evaluation Protocol

We adopt the frame-level area under the receiver operating characteristic curve (AUROC) as the primary evaluation metric, following established practice in the anomaly detection literature [6], [13], [14]. AUROC quantifies the probability that a randomly sampled anomalous frame receives a higher anomaly score than a randomly sampled normal frame. For YOLO v11n-pose, we additionally report precision, recall, and mean average precision at IoU thresholds of 0.5 (mAP@.5) and 0.5:0.95 (mAP@.5:.95), evaluated on the CU Indoor Anomaly test partition. Throughput is measured as end-to-end frames per second on a single NVIDIA Titan XP GPU, inclusive of all preprocessing, inference, and postprocessing stages.

![](images/8d3a05cea5d91f0d0ef7a104c2502069580264d114c128dbac59591f6b459ea9.jpg)  
Fig. 1: Overview of the proposed anomaly detection pipeline. Stage 1 employs YOLO v11n-pose for person detection and skeletal keypoint extraction. Stage 2 encodes each cropped person region via the CLIP image encoder and computes cosine similarity against textual anomaly prompts from the CLIP text encoder. The frame-level anomaly score is then temporally smoothed and compared against a threshold for final classification.

TABLE I: Frame-Level AUROC (%) Comparison with Existing Methods
<table><tr><td>Method</td><td>Avenue</td><td>SHTech</td><td>CU Indoor</td></tr><tr><td>Conv-AE [3]</td><td>80.00</td><td>60.85</td><td>一</td></tr><tr><td>MPED-RNN [11]</td><td>83.50</td><td>73.40</td><td></td></tr><tr><td>DSTN [10]</td><td>86.40</td><td>73.10</td><td></td></tr><tr><td>Attr-Based [6]</td><td>86.00</td><td>71.10</td><td></td></tr><tr><td>Multi-feat. baseline†</td><td>89.26</td><td>70.26</td><td>82.12</td></tr><tr><td>Proposed</td><td>89.26‡</td><td>70.26</td><td>84.13</td></tr></table>

<sup>†</sup>Our reimplementation of [6] with YOLO-based detection. <sup>‡</sup>Shares the same YOLO detector and evaluation as <sup>†</sup>; only the scoring stage differs (CLIP vs. density estimation).

## V. RESULTS AND ANALYSIS

## A. Anomaly Detection Performance

Table I reports frame-level AUROC results. On CUHK Avenue, the proposed framework achieves 89.26%, matching the multi-feature baseline employing FlowNet2, AlphaPose, and GMM/kNN scoring. On the ShanghaiTech Campus, the score of 70.26% reflects the challenge posed by highly diverse scenes and anomaly types (cycling, skateboarding, vehicle intrusion) that fall outside the coverage of the four fixed text prompts. On CU Indoor Anomaly, the method reaches 84.13%, surpassing the baseline by roughly two percentage points. We attribute this gain to the strong semantic alignment between CLIP’s text-image similarity and the specific anomaly categories (falling, lying, fighting, sitting) present in this dataset. Notably, the proposed method outperforms DSTN [10] on Avenue by 2.86 percentage points while operating at substantially higher throughput. To clarify the Table I footnotes: the multifeature baseline is our reimplementation of [6] with a YOLObased detector and scores anomalies via GMM/kNN density estimation over concatenated pose, velocity, and deep features, whereas the proposed method scores via direct CLIP textimage cosine similarity. Both pipelines share the same YOLO detector, so Avenue and ShanghaiTech values are identical; the AUROC difference on CU Indoor isolates the effect of the scoring mechanism.

TABLE II: YOLO v11n-pose Detection Performance on CU Indoor Anomaly
<table><tr><td>Category</td><td>Prec.</td><td>Rec.</td><td>mAP@.5</td><td>mAP@.5:.95</td></tr><tr><td>All</td><td>0.91</td><td>0.97</td><td>0.92</td><td>0.67</td></tr><tr><td>Normal</td><td>0.95</td><td>0.98</td><td>0.96</td><td>0.71</td></tr><tr><td>Abnormal</td><td>0.86</td><td>0.96</td><td>0.88</td><td>0.63</td></tr></table>

## B. Qualitative Detection Results

Fig. 2 shows representative outputs. Normal persons are enclosed in green bounding boxes (a, c), while detected anomalies are enclosed in red boxes with YOLO confidence and CLIP scores (b, d).

## C. YOLO Detection Performance

Table II reports the fine-tuned YOLO v11n-pose detection metrics on the CU Indoor Anomaly test set, evaluated across 237 images containing 360 annotated person instances. The model achieves 91% precision and 97% recall overall, with mAP@.5 of 92%. Normal poses are detected with higher precision (95%) than abnormal ones (86%), which is expected given the greater variability and less stereotyped configurations associated with anomalous body positions such as lying and sitting on the floor. The high recall across both categories (98% normal, 96% abnormal) confirms that the detector rarely misses persons in the scene, which is critical for the downstream CLIP scoring stage. The mAP@.5:.95 gap between the normal (71%) and abnormal (63%) classes suggests that bounding-box localization is less precise for abnormal poses, though this has minimal impact on CLIP scoring because the crops include contextual padding.

## D. Processing Throughput

Table III compares per-component and end-to-end throughput between the baseline and proposed pipelines. The baseline employs Mask R-CNN for detection, FlowNet2 for optical flow, AlphaPose for skeleton estimation, and GMM/kNN for density-based scoring. The proposed method removes all intermediate modules and relies exclusively on YOLO v11n-pose and CLIP, achieving an end-to-end throughput of 51.39 FPS, a 3.36× speedup over the baseline. The bottleneck shifts from object detection (23.41 FPS in the baseline) to YOLO v11npose (87.76 FPS), with CLIP scoring running comfortably at 124.04 FPS. Note that the baseline density estimation speed (2578.80 FPS) reflects the lightweight nature of GMM/kNN scoring on pre-extracted feature vectors. The gap between per-component rates and the measured end-to-end throughput arises from overhead not captured by isolated benchmarks:

![](images/446b5f1311c79346b32cbb13b7063d94072a9c01269725a6575ff72b9cb72878.jpg)

(a) GUI output during normal behavior (walking)  
![](images/0c8cf7d92e6c22e91d8ba66c6c35383a365089eafcfa3c853afea6294f00a5c2.jpg)

(b) GUI output during detected anomaly (lying on floor)  
![](images/5f12093c9a5df9e8ee6e957c0196f4916f06b5c149e340bd43264a20060f8e43.jpg)

![](images/cdfe47ed1e9495b6471d271fed78fb845ac3834d4063a99a608c76f01308e4a8.jpg)  
(c) Normal: bounding box  
(d) Abnormal: falling  
Fig. 2: Real-time detection results on the CU Indoor Anomaly dataset. (a)–(b) Full system GUI showing live video feed, realtime anomaly score graph, and frame history panel for normal and abnormal scenarios. (c)–(d) Close-up views showing green bounding boxes for normal behavior and red bounding boxes with anomaly scores for detected anomalies.

TABLE III: Processing Throughput Comparison (Frames Per Second)
<table><tr><td>Component</td><td>Baseline</td><td>Proposed</td></tr><tr><td>Object detection</td><td>23.41</td><td>87.76</td></tr><tr><td>Deep features (CLIP)</td><td>45.11</td><td></td></tr><tr><td>Density estimation</td><td>2578.80</td><td></td></tr><tr><td>CLIP scoring</td><td></td><td>124.04</td></tr><tr><td>End-to-end</td><td>15.32</td><td>51.39</td></tr></table>

video frame decoding from the CCTV stream, CPU-to-GPU data transfer for each person crop, variable CLIP batch sizes depending on the number of persons detected per frame, and Python/OpenCV rendering of bounding-box overlays. These overheads are inherent to a live deployment pipeline.

## VI. DISCUSSION

The principal strength of the proposed framework is its architectural simplicity. By routing person crops directly through CLIP for text-image similarity, the pipeline eliminates optical flow extraction (FlowNet2), standalone skeleton estimation (AlphaPose), and iterative density model fitting (GMM/kNN), reducing the system to two neural network models. This accelerates throughput from 15.32 to 51.39 FPS while requiring only a single GPU and two model checkpoints for deployment.

The text-prompt mechanism provides practical flexibility: adapting the system to a new surveillance environment requires only updating the prompt set, with no retraining. CLIP’s zero-shot capability further enables generalization to anomaly categories absent from the training data, provided they can be described in natural language. The system was deployed on five live CCTV feeds in the elevator area on the 12th floor of the Charoen Wisawakam Building at Chulalongkorn University, achieving stable operation at approximately 10 FPS per feed, including overhead video capture, real-time boundingbox overlays, and concurrent updates to anomaly score graphs.

However, the current system is designed specifically for person-level anomalies in controlled indoor settings. Anomalies that do not involve people, such as vehicle intrusion, or open-set anomaly types not covered by the four fixed prompts, would require complementary detection modules in a production deployment. The fixed-prompt design was chosen deliberately for zero-training deployability and interpretability: operators can inspect and modify the prompt set without machine-learning expertise. Learnable prompt expansion, including LLM-generated candidate prompts, is deferred to future work.

## VII. CONCLUSION

This paper presented a two-stage anomaly detection framework that couples YOLO v11n-pose with CLIP ViT-B/32, replacing multi-component pipelines that require optical flow and density estimation. The key finding is that direct textimage similarity via CLIP can serve as an effective scoring mechanism, achieving a 3.36× throughput gain (51.39 FPS)

while maintaining comparable AUROC on established benchmarks and improving performance on the targeted indoor dataset (84.13%). Deployment on live CCTV feeds at Chulalongkorn University confirmed stable operation at over 10 FPS per feed.

The current design has limitations. The fixed set of four text prompts constrains the detectable anomaly categories, as reflected in the lower ShanghaiTech Campus AUROC (70.26%) where anomalies such as cycling and vehicle intrusion fall outside prompt coverage. Additionally, the YOLO detector is fine-tuned on indoor scenes, which may reduce generalization to markedly different environments.

Future work will address these limitations by using learnable prompt expansion, inspired by VadCLIP [9], to broaden anomaly coverage; illumination-robust preprocessing for lowlight environments; and developing a unified end-to-end model suitable for deployment on resource-constrained edge devices.

## ACKNOWLEDGMENT

The authors gratefully acknowledge Tatsawal Chantharachana, Siwanont Tantayapirom, and Tanakrit Sutthipattanakit from the Department of Electrical Engineering, Faculty of Engineering, Chulalongkorn University, for their contributions to the initial development of the YOLO-based detection and CLIP-based scoring components that form the experimental foundation of this work. This work was supported by the Thailand Science Research and Innovation Fund, Chulalongkorn University (IND FF 69 105 2100 018), and by the Video Technology Research Group (VTRG), Department of Electrical Engineering, Faculty of Engineering, Chulalongkorn University. The first author acknowledges the Graduate Scholarship Program for ASEAN or Non-ASEAN Countries.

## REFERENCES

[1] B. Ramachandra, M. Jones, and R. Vatsavai, “A survey of single-scene video anomaly detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 5, pp. 2293–2312, May 2022.

[2] D. C. Gill and T. O’Brien, “CCTV operator performance and fatigue: A systematic review,” Security Journal, vol. 33, no. 4, pp. 558–574, 2020.

[3] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, “Learning temporal regularity in video sequences,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 733–742.

[4] M. Ravanbakhsh, M. Nabi, E. Sangineto, L. Marcenaro, C. Regazzoni, and N. Sebe, “Abnormal event detection in videos using generative adversarial nets,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2017, pp. 1577–1581.

[5] W. Liu, W. Luo, D. Lian, and S. Gao, “Future frame prediction for anomaly detection – A new baseline,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 6536–6545.

[6] T. Reiss and Y. Hoshen, “An attribute-based method for video anomaly detection,” arXiv preprint arXiv:2212.00789, 2022.

[7] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 8748–8763.

[8] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLO (v11),” 2024. [Online]. Available: https://docs.ultralytics.com/models/yolo11/

[9] P. Wu, X. Zhou, G. Pang, L. Zhou, Q. Yan, P. Wang, and Y. Zhang, “VadCLIP: Adapting vision-language models for weakly supervised video anomaly detection,” in Proc. AAAI Conf. Artif. Intell., 2024, pp. 6074–6082.

[10] T. Ganokratanaa, S. Aramvith, and N. Sebe, “Unsupervised anomaly detection and localization based on deep spatiotemporal translation network,” IEEE Access, vol. 8, pp. 50312–50327, Mar. 2020.

[11] R. Morais, V. Le, T. Tran, B. Saha, M. Mansour, and S. Venkatesh, “Learning regularity in skeleton trajectories for in-the-wild action detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 11810–11819.

[12] R. Moriyama, S. Suzuki, N. Kaneko, and K. Sumi, “Language-guided decision override for adaptive and retraining-free video anomaly detection,” in Proc. 36th British Mach. Vis. Conf. (BMVC), 2025.

[13] C. Lu, J. Shi, and J. Jia, “Abnormal event detection at 150 FPS in MATLAB,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2013, pp. 2720–2727.

[14] W. Luo, W. Liu, and S. Gao, “A revisit of sparse coding based anomaly detection in stacked RNN framework,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 341–349.