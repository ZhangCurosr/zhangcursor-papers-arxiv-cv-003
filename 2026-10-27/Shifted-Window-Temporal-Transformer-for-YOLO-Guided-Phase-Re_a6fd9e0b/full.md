# Shifted Window Temporal Transformer for YOLO-Guided Phase Recognition in Laparoscopic Cholecystectomy Surgery

Sheekar Banerjee<sup>∗1†</sup>, Md. Shahabuddin<sup>1†</sup>, Md. Mahbub Hasan Akash<sup>1</sup>, Nabilah .<sup>1</sup>, Mohammad Arafat Hussain<sup>2</sup>

<sup>1</sup>Department of Computer Science and Engineering, IUBAT—International University of Business Agriculture and   
Technology, 4 Embankment Drive Road, Sector-10, Uttara Model Town, Dhaka, 1230, Bangladesh.   
<sup>2</sup>Boston Children’s Hospital, Harvard Medical School, 401 Park Dr, Boston, USA.

<sup>†</sup>Sheekar Banerjee and Md. Shahabuddin contributed equally to this work.

## Abstract

Intelligent operating rooms require accurate awareness of surgical workflow for better intra-operative decisions, teamwork, and surgical documentation. Therefore, accurately recognizing surgical phase during surgery is crucial. However, previous artificial intelligence-based methods provide fluctuating predictions for the transitions, restricting their clinical use. To address this challenge, we propose a clinically meaningful framework that integrates a Swin Transformer and Temporal Transformer Encoder (TTE) for efective surgical phase recognition and a parallel YOLOv11s network for real-time surgical instrument localization. To avoid phase flickering and to produce stable workflow predictions, a causal temporal smoothing is applied. The proposed framework is tested on the Cholec80 dataset for the simultaneous recognition of the surgical phases and instruments. The overall phase recognition accuracy of this model was 92%, and the macro average F1-score was 0.86. This performance surpasses CNN-based methods such as EndoNet (81.7%) and SV-RCNet (85.3%), as well as temporal convolutional and transformer-based approaches including TeCNO (88.6%), Trans-SVNet (90.9%), and LoViT (91.5%). Besides, the module of instrument detection achieved mAP@0.5 = 0.863. The reliability of the results is confirmed, obtaining

a high recognition rate of the workflows, and a high accuracy of the localization of the instruments. The proposed framework can serve as a foundation for intelligent surgical assistance, enriching situation awareness, and potential robotic surgery applications. Code will be made publicly available upon acceptance at: https://github.com/ac005sheekar/LapChol-Surgery-Swin-Cholec80.

Keywords: Surgical Phase Recognition, Surgical Tool Detection, Laparoscopic Cholecystectomy, Cholec80, Swin Transformer, Temporal Transformer Encoder, YOLOv11s, Multi-Task Learning, Sliding Window Prediction, Phase Smoothing.

## 1 Introduction

Advances in artificial intelligence (AI) have enabled intelligent clinical decisionsupport systems for improving surgical workflow analysis and inter-operative awareness [11, 12]. In laparoscopic cholecystectomy, large volumes of surgical videos are generated, but manual analysis is time-consuming and impractical, motivating automated video analysis systems [9, 10]. Automatic surgical workflow analysis requires understanding both visual information and procedural context. An efective system must recognize surgical phases while identifying the instruments used during the procedure. Since consecutive frames often have similar visual appearances, spatial information alone may lead to ambiguous predictions and unstable phase transitions. Therefore, robust workflow recognition requires efective spatial and temporal feature modeling [2–6].

Deep learning has substantially advanced surgical workflow analysis. Early CNNbased approaches, including EndoNet [1] and SV-RCNet [46], demonstrated efective feature learning from laparoscopic videos [1, 2]. More recently, Vision Transformers (ViTs) have improved long-range contextual modeling through self-attention [14, 15]. The Swin Transformer further provides hierarchical spatial representations using shifted-window attention [13], while temporal transformers and multi-task learning enable efective modeling of procedural dynamics and complementary phase– instrument information [4–6, 42, 43]. Even though these advances have been made, there are still a number of challenges. Phase recognition and instrument detection are frequently considered conflicting tasks, thus depriving them from exploiting a relationship between the usage of instruments and the advancement of a workflow. Furthermore, two similar visual frames can result in ‘phase flicker”, and class imbalance can lead to less good recognition results for less common surgical phases [3, 6, 40].

Recent advances in artificial intelligence have enabled automated surgical procedure tracking, skill assessment, and intra-operative decision support. Surgical workflow analysis focuses primarily on phase recognition and instrument detection. Early methods relied on handcrafted features and Hidden Markov Models (HMMs), whereas deep learning enabled end-to-end feature learning directly from surgical videos.

CNN-based approaches have been shown to be efective in capturing spatial information in surgical phase recognition [1, 16]. But frame level CNNs cannot capture long-term temporal relations between adjoining frames, which makes the predictions unstable. CNN-LSTM and CNN-GRU architectures help improve temporal modeling but are limited by sequential processing [2, 5, 22, 37]. Vision Transformers (ViTs) and temporal transformer architectures can ofer much better long-range spatial and temporal modeling [4–6, 14, 15, 42, 43] respectively. The Swin Transformer further proposes hierarchical representations with shifted-window self-attention for eficient multi-scale contextual modeling, which is beneficial for surgical video analysis [6, 13].

The localization of surgical instruments can complement the information needed for workflow recognition. Traditional two-stage detectors are sensitive to occlusion, illumination variation, and instrument deformation. While YOLO demonstrations are classified by single stage detectors, they can ofer a balance between the detection accuracy and the speed of the machine motion inference in the real-time applications [17–21, 45]. Multi-task learning also allows a connection between surgical phases and instruments to be exploited by using shared feature representations. Joint models of these tasks have been seen to yield better performance than independent models in previous studies; performance is a challenge and problems with task balancing and negative transfer still exist[1–5, 30, 31].

Surgical datasets are intrinsically unbalanced, the most common aspects of surgery being available in large numbers in the training set, but clinically important aspects occurring with less frequency. Class-balanced sampling, focal loss, and cost-sensitive learning are commonly used to improve minority-class recognition [30–32, 40]. In addition, visually similar consecutive frames can cause phase flickering during realtime inference. Temporal post-processing methods, including causal smoothing and rolling-queue strategies, improve prediction consistency while maintaining low latency [2, 6, 42].

Despite these advances, phase recognition and instrument detection are often treated as separate tasks, and the integration of hierarchical spatial transformers with temporal transformers remains relatively unexplored [4–6, 43]. Real-time deployment is further afected by class imbalance and prediction instability [30]. To address these gaps, this work proposes a unified framework combining a Swin Transformer, Temporal Transformer, YOLO-based instrument localization, and causal temporal smoothing for reliable surgical workflow analysis.

In this paper, to tackle the above problems, we aim at developing a comprehensive spatial–temporal framework that works for surgical phase recognition and real-time instrument localization at the same time. The network combines a Swin Transformer base and Temporal Transformer network for encoding spatial and long-range temporal information and a bespoke YOLOv11s detector in order to locate surgical instruments [17, 20, 21, 41]. Additionally, a causal rolling-queue temporal smoothing strategy is used to smoothen out the prediction and to achieve temporal consistency

at real-time run time.

The major contributions of this work are summarized as follows:

(a) A single framework to simultaneously recognize surgical phases and locate instruments in real-time during Laparoscopic Cholecystectomy.

(b) A novel Swin Transformer and Temporal Transformer based architecture to incorporate efective spatial–temporal modeling and mitigate phase flickering.

(c) A trained YOLOv11s detector for real-time surgical instrument localization and additional procedural context.

(d) Proposing a causal rolling-queue temporal smoothing strategy and extensive experiments using the Cholec80 dataset towards stable and reliable workflow real-time analysis.

## 2 Results

The proposed dual-stream multi-task framework was evaluated on the Cholec80 test split (Videos 61–80) for surgical phase recognition, instrument presence prediction, and YOLOv11s instrument localization. Evaluation included phase classification metrics and object detection performance.

## 2.1 Qualitative Instrument Localization

Figure 1 illustrates representative qualitative results of the YOLOv11s surgical instrument localization on frames from the Cholec80 dataset during laparoscopic cholecystectomy. The proposed detector is able to correctly detect and spatially locate multiple surgical instruments under various surgical scenes, as illustrated in the figure, such as Bipolar, Specimen Bag, Grasper, Clipper, Hook, and Irrigator. The bounding boxes predicted are similar to the visible instruments, confirming the detector’s capacity to accurately capture the spatial location of instruments from diferent viewpoints, under various illumination conditions, occlusions and surgical contexts. The qualitative results corroborate the quantitative detection results seen in the previous analysis.

![](images/a13146036a4361b35ecc626603101068b0490ef8212092f30dc9a47105b59501.jpg)  
Fig. 1: Qualitative examples of YOLOv11s-based surgical instrument localization on laparoscopic cholecystectomy frames from the Cholec80 dataset.

## 2.2 Surgical Phase Recognition

Table 1 summarizes the seven-class phase recognition results. The highest F1-scores are obtained for CalotTriangleDissection (0.95) and GallbladderDissection (0.94), while CleaningCoagulation achieves the lowest F1-score (0.73). The lower performance of some phases is associated with class imbalance and visual similarity between neighboring procedural stages.

Table 2 shows an overall accuracy of 92%, macro F1-score of 0.86, and weighted F1- score of 0.92 across 47,625 test samples. The weighted F1 reflects strong performance on dominant classes, while the macro F1 indicates comparatively lower performance on less frequent phases.

Table 1: Classification report for the seven surgical phases on the Cholec80 dataset.
<table><tr><td>Phases</td><td>Precision</td><td>Recall</td><td>F1-score</td><td>Support</td></tr><tr><td>Preparation</td><td>0.94</td><td>0.79</td><td>0.86</td><td>2162</td></tr><tr><td>CalotTriangleDissection</td><td>0.93</td><td>0.97</td><td>0.95</td><td>17456</td></tr><tr><td>ClippingCutting</td><td>0.84</td><td>0.93</td><td>0.88</td><td>2951</td></tr><tr><td>GallbladderDissection</td><td>0.98</td><td>0.90</td><td>0.94</td><td>19144</td></tr><tr><td>GallbladderPackaging</td><td>0.82</td><td>0.94</td><td>0.88</td><td>2089</td></tr><tr><td>CleaningCoagulation</td><td>0.65</td><td>0.83</td><td>0.73</td><td>2293</td></tr><tr><td>GallbladderRetraction</td><td>0.87</td><td>0.71</td><td>0.78</td><td>1530</td></tr></table>

Table 2: Overall performance of the proposed multi-task model on the Cholec80 dataset.
<table><tr><td>Metric</td><td>Precision</td><td>Recall</td><td>F1-score</td><td>Support</td></tr><tr><td>Accuracy</td><td></td><td></td><td>0.92</td><td>47625</td></tr><tr><td>Macro Average</td><td>0.86</td><td>0.87</td><td>0.86</td><td>47625</td></tr><tr><td>Weighted Average</td><td>0.92</td><td>0.92</td><td>0.92</td><td>47625</td></tr></table>

Table 3 compares our framework with state-of-the-art methods on the Cholec80 dataset. Our approach achieves 92.0% accuracy, outperforming CNN-based methods (EndoNet, SV-RCNet) and temporal convolutional networks (TeCNO), and remaining competitive with recent transformer-based approaches (Trans-SVNet, LoViT). In terms of Jaccard index, our method achieves a macro-averaged score of 0.76, which is competitive with Trans-SVNet (0.797) and TeCNO (0.751), despite being evaluated on a smaller subset (Videos 61–80) of the standard test split. We note that prior methods are evaluated on Videos 41–80, whereas our evaluation uses Videos 61–80. This comparison is therefore conservative, as our model is tested on a potentially more challenging portion of the dataset. Direct re-implementation of prior methods was not performed due to computational constraints; we report their published metrics for reference.

## 2.3 Confusion Matrix Analysis

Figure 2 summarizes the per-class F1-scores, showing strongest performance for the major dissection phases and lower performance for more dificult or less frequent phases.

Figure 3 shows strong diagonal dominance in both training and test confusion matrices. Most misclassifications occur between neighboring phases with similar anatomical views and instrument usage, indicating that the temporal model provides reasonable generalization.

![](images/69d0e03e52b510cb7ae3ab726bc909f954a144316c100cee893fac8171c4ca3a.jpg)  
Fig. 2: Per-class F1-score for surgical phase recognition.

![](images/d077abe822680cb5a78db3e0027280ffa80aac730aeb9591fd0cd9e0f9c3ce55.jpg)  
(a) Training set confusion matrix.

![](images/800e2e62ce14f64adaa41b8c98af86838a7891cbdc50a6cbd4b8de302c758aff.jpg)  
(b) Test set confusion matrix.  
Fig. 3: Normalized confusion matrices of the proposed Swin-Temporal phase recognition model.

## 2.4 YOLOv11s Instrument Detection

Figures 4(a)–(d) summarize the YOLOv11s instrument detection performance. The detector achieves an overall F1-score of 0.84 at a confidence threshold of 0.266 and an mAP@0.5 of 0.863. The Clipper achieves the highest AP (0.959), whereas the Irrigator shows the lowest AP (0.787), mainly due to occlusion and limited samples.

The detector reaches a precision of 1.00 at a confidence threshold of 0.977 and recall of 0.92 at a threshold of 0.000. These threshold-dependent values illustrate the trade-of between precision and recall during instrument localization.

![](images/202597b3afb6bf71b85544ceca1d3f98d57723b651007fed263c950afe846397.jpg)  
(a) F1 vs. confidence threshold.

![](images/508d2af29cfb4a4fb072759632e9f4befcf780857773dff2b81b708781adda6d.jpg)  
(b) Precision vs. confidence threshold.

![](images/dd80bc6bcaf875ea90c8268be13853dba50614c18aea0e34f6482c85b94542dc.jpg)  
(c) Precision–Recall curve.

![](images/1fe2343aa2ec1d5ebdf0aff383880a56b3870dd4864f4be892f5d6bc76516d49.jpg)  
(d) Recall vs. confidence threshold.  
Fig. 4: Performance curves of the YOLOv11s surgical instrument detector.

Figure 5 shows the YOLOv11s training and validation curves over 50 epochs. The losses decrease and stabilize after approximately 30 epochs, while the final validation performance reaches precision 0.85, recall 0.82, mAP@0.5 of 0.863, and mAP@0.5:0.95 of approximately 0.50.

## 3 Discussion

The experimental results demonstrate that the proposed dual-stream multi-task framework can simultaneously perform surgical phase recognition and real-time instrument localization on the Cholec80 dataset. The Swin–Temporal Transformer achieves strong phase recognition, with the highest F1-scores for CalotTriangleDissection and GallbladderDissection, while most errors occur between adjacent phases. The YOLOv11s detector achieves an overall mAP@0.5 of 0.863, demonstrating efective instrument localization despite class imbalance and challenging laparoscopic conditions.

The results indicate that combining spatial instrument localization with temporal workflow modeling provides complementary information for surgical video analysis. The detector performs particularly well for the Clipper, whereas the Irrigator is more afected by occlusion, smoke, and fluid interference. The optimal detection threshold

![](images/e902d398f9540155476f9a0794b847bc402a8f237c4091fbeb4e5f30ad25e4ea.jpg)  
Fig. 5: Training and validation performance of the shifted window transformer based YOLOv11s instrument detection model.

Table 3: Comparison with state-of-the-art methods on the Cholec80 dataset.   
Results for prior methods are as reported in their original publications. Our method   
is evaluated on Videos 61–80 (a subset of the standard 41–80 test split); therefore,   
this comparison is conservative. The Jaccard index for our method is computed from the reported precision, recall, and support values in Table 1.

<table><tr><td>Method</td><td>Year</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>Jacc.</td><td>Architecture</td></tr><tr><td>EndoNet [1]</td><td>2017</td><td>81.7%</td><td></td><td></td><td></td><td></td><td>CNN + LSTM</td></tr><tr><td>SV-RCNet [46]</td><td>2018</td><td>85.3%</td><td></td><td></td><td></td><td></td><td>CNN + LSTM</td></tr><tr><td>TeCNO [3]</td><td>2020</td><td>88.6%</td><td></td><td></td><td></td><td>75.1%</td><td>Multi-stage TCN</td></tr><tr><td>Trans-SVÑet [4]</td><td>2021</td><td>90.9%</td><td></td><td></td><td></td><td>79.7%</td><td>Hybrid Transformer</td></tr><tr><td>LoViT [6]</td><td>2023</td><td>91.5%</td><td></td><td></td><td></td><td></td><td>Long Video Transformer</td></tr><tr><td>Ours</td><td>2026</td><td>92.0%</td><td>0.92</td><td>0.92</td><td>0.86</td><td>76.2%</td><td>Swin + TTE + YOLOv11s</td></tr></table>

of 0.266 yields an F1-score of 0.84. In addition, the causal rolling-window smoothing strategy improves temporal consistency by reducing isolated frame-level phase fluctuations.

The experimental results address the main objectives of the proposed framework. First, YOLOv11s provides real-time instrument localization with an mAP@0.5 of 0.863 and an F1-score of 0.84. Second, the instrument detection branch provides reliable tool localization, although performance decreases for classes afected by occlusion and limited samples. Third, the Swin–Temporal Transformer achieves 92% phase recognition accuracy and a macro F1-score of 0.86, with most errors occurring between visually similar neighboring phases. These results demonstrate the efectiveness of the proposed framework for integrated surgical workflow analysis.

The proposed framework can support practical surgical video analysis by providing continuous phase information together with instrument localization. Workflow recognition and object detection might help to support the indexation of surgical videos and the review of the workflow in the post-operative phase and also the implementation of intelligent assistance systems.

Table 3 compares our framework with state-of-the-art methods on the Cholec80 dataset. Our approach achieves 92.0% accuracy, outperforming CNN-based methods (EndoNet, SV-RCNet) and temporal convolutional networks (TeCNO), and remaining competitive with recent transformer-based approaches (Trans-SVNet, LoViT). In terms of Jaccard index, our method achieves a macro-averaged score of 0.76, which is competitive with Trans-SVNet (0.797) and TeCNO (0.751), despite being evaluated on a smaller subset (Videos 61–80) of the standard test split. We note that prior methods are evaluated on Videos 41–80, whereas our evaluation uses Videos 61–80. This comparison is therefore conservative, as our model is tested on a potentially more challenging portion of the dataset. Direct re-implementation of prior methods was not performed due to computational constraints; we report their published metrics for reference.

To cater to online visualization of one frame to the next, a causal FIFO smoothing approach, which does not rely on future frames, is developed with W = 10 to minimize the isolated fluctuation of the phases. Moreover, the results of these infrequent and frequent phases indicate the possibility for automated organization and retrieval of surgical video segments. But the technology would need to be further tested under various surgical clinics, surgeons, and surgical room conditions before it’s clinical deployment.

## 4 Methods

The proposed framework is an integrated multi-task computer vision system for simultaneous surgical phase recognition and instrument detection. Laparoscopic video frames are first preprocessed and organized into synchronous sliding windows of T = 12 consecutive frames. A shared hierarchical Swin Transformer [13] extracts spatial features, followed by a Temporal Transformer Encoder [15] for long-range temporal modeling. The resulting features are jointly used for multi-class surgical phase classification and multi-label instrument recognition. Loss function based on focusing and inverse-frequency sampling is used to address class imbalance [30, 31]. Concurrently, a detector based on YOLO is working in processing the full-resolution frames for real time localization of the instrument with a confidence threshold of 0.25.

![](images/8c1b0460779554b2de80c708d140457ef0ac0ee8dea774ed386279b6c17c0d8e.jpg)  
Fig. 6: Overall Architecture of the Proposed Dual-Stream Multi-Task Surgical Workflow Analysis Framework

## 4.1 Dataset

This proposed framework is examined on the public laparoscopic cholecystectomy Cholec80 dataset [40] consisting of 80 high-definition videos obtained from 13 surgeons. The videos were captured at 25 frames per second (fps) in 1920×1080 and/or 854×480 pixels. There is one annotation per frame with surgical phase and instrument presence.

The operations are broken down into seven surgical steps: (a) preparation, (b) calot Triangle Dissection, (c) clipping Cutting, (d) gallbladder dissection, (e) gallbladder packaging, (f) cleaning coagulation and (g) gallbladder retraction. Instrument presence is represented by a 7-dimensional binary vector, allowing multiple instruments to be simultaneously present in a frame; therefore, instrument recognition is formulated as a multi-label classification task.

Following the standard evaluation protocol, Videos 1–40 are used for training, Videos 41–60 for validation, and Videos 61–80 for testing. The video-level split prevents temporal data leakage. To reduce redundancy, videos are uniformly downsampled from 25 fps to 1 fps using frame\_step=25.

## 4.2 Video Preprocessing and Temporal Clip Construction

The extracted frames are resized to 224×224 pixels and converted from BGR to RGB format. Pixel values are normalized to [0, 1] by dividing by 255, followed by ImageNet

channel-wise normalization:

$$
\mu = [ 0 . 4 8 5 , 0 . 4 5 6 , 0 . 4 0 6 ] , \qquad \sigma = [ 0 . 2 2 9 , 0 . 2 2 4 , 0 . 2 2 5 ] .
$$

To preserve temporal context, consecutive frames are organized into sliding-window clips of $T = 1 2$ frames. A temporal window stride of 8 frames and a clip stride of 5 frames are used to generate overlapping sequences while maintaining chronological continuity. Each sample is represented as a tensor of shape $( B , T , C , H , W )$ .

## 4.3 Model Architecture

The proposed architecture consists of a spatio-temporal multi-task classification stream and a parallel YOLO-based localization stream. The classification stream extracts spatial representations from individual frames using a shared Swin Transformer and subsequently models their temporal relationships using a Temporal Transformer Encoder. The resulting context-aware features are used for simultaneous phase and instrument prediction.

A hierarchical Swin Transformer (swin\_t) is used as the shared spatial backbone. It independently processes each frame in the input tensor $( B , T , C , H , W )$ using shifted-window self-attention and produces spatial feature representations with $d _ { \mathrm { m o d e l } } = 7 6 8$ . These features preserve relevant visual patterns and contextual information from the laparoscopic frames.

A parallel YOLOv11s-based localization branch processes individual full-resolution frames independently from the spatio-temporal classification stream. The detector predicts bounding boxes for seven instrument classes using a confidence threshold of $\tau = 0 . 2 5$ . The resulting spatial detections are combined with the predicted surgical phase to provide real-time workflow visualization.

## 4.4 Loss Function and Class Imbalance

The proposed multi-task network jointly optimizes surgical phase recognition and instrument presence prediction. Phase recognition is formulated as a multi-class classification task, while instrument recognition is treated as a multi-label task. To formalize this optimization, the following notation is used throughout this section. Let $\mathcal { L } _ { \mathrm { t o t a l } }$ denote the combined training loss, $\mathcal { L } _ { \mathrm { p h a s e } }$ and ${ \mathcal { L } } _ { \mathrm { t o o l } }$ denote the phase classification loss and instrument presence loss respectively, and $\lambda _ { \mathrm { p h a s e } }$ and $\lambda _ { \mathrm { t o o l } }$ denote the correspond ing task-balancing weights. For the focal loss, $p _ { t }$ represents the predicted probability of the target class, $\alpha _ { t }$ is the class-balancing factor, and $\gamma$ is the focusing parameter that controls the rate at which easy examples are down-weighted. For the temporal post-processing stage, $z _ { t }$ denotes the raw logits produced by the phase classification head at time step $t , y _ { \mathrm { r a w } } ( t )$ is the raw predicted phase before smoothing, $W$ is the rolling queue length, and $\hat { y } _ { \mathrm { s m o o t h e d } } ( t )$ is the final smoothed phase prediction. For the multi-task formulation, C is the number of surgical phase classes, $z \in \mathbb { R } ^ { C }$ is the logit vector for phase prediction, $c$ is the phase class index, $v _ { k }$ is the logit for instrument $k ,$ and $\sigma ( \cdot )$ denotes the sigmoid activation function.

The total training loss is defined as

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { p h a s e } } \mathcal { L } _ { \mathrm { p h a s e } } + \lambda _ { \mathrm { t o o l } } \mathcal { L } _ { \mathrm { t o o l } } ,\tag{1}
$$

where $\lambda _ { \mathrm { p h a s e } }$ and $\lambda _ { \mathrm { t o o l } }$ balance the contributions of the two tasks.

Cholec80 contains substantial class imbalance, with common phases having considerably more samples than shorter transition phases. To improve minority-class learning, focal loss and inverse-frequency sampling are employed [30, 31]. The focal loss is defined as

$$
\mathrm { F L } ( p _ { t } ) = - \alpha _ { t } ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) ,\tag{2}
$$

where $p _ { t }$ is the predicted probability of the target class, $\alpha _ { t }$ is the class-balancing factor, and $\gamma$ is the focusing parameter. Inverse-frequency sampling further increases the representation of minority phases during training.

To reduce frame-level phase fluctuations caused by visual ambiguity, occlusion, or detection errors, a causal FIFO rolling queue of length $W = 1 0$ is applied during inference. The raw prediction is

$$
y _ { \mathrm { r a w } } ( t ) = \arg \operatorname* { m a x } ( \mathrm { S o f t m a x } ( z _ { t } ) ) ,\tag{3}
$$

where $z _ { t }$ denotes the raw logits generated by the phase classification head at time step $t .$ The final phase prediction is obtained by majority voting over the most recent $W$ predictions:

$$
{ \hat { y } } _ { \mathrm { s m o o t h e d } } ( t ) = \operatorname { m o d e } \left( \left\{ y _ { \mathrm { r a w } } ( i ) ~ | ~ i = \operatorname * { m a x } ( 1 , t - W + 1 ) , \ldots , t \right\} \right) .\tag{4}
$$

This causal strategy improves temporal consistency without using future frames.

Surgical phase recognition is formulated as a seven-class problem. Given logits $z \in \mathbb { R } ^ { C }$ , where $C = 7$ , the phase probability is

$$
P ( y _ { \mathrm { p h a s e } } = c \mid x ) = \frac { \exp ( z _ { c } ) } { \sum _ { j = 1 } ^ { C } \exp ( z _ { j } ) } ,\tag{5}
$$

with prediction

$$
\hat { y } _ { \mathrm { p h a s e } } = \arg \operatorname* { m a x } _ { c } P ( y _ { \mathrm { p h a s e } } = c \mid x ) .\tag{6}
$$

Instrument recognition is formulated as a seven-label binary classification task because multiple instruments may occur simultaneously. For instrument $k .$

$$
P ( v _ { k } = 1 \mid x ) = \sigma ( v _ { k } ) = \frac { 1 } { 1 + \exp ( - v _ { k } ) } ,\tag{7}
$$

where $v _ { k }$ is the logit for instrument k and $\sigma ( \cdot )$ is the sigmoid activation function. Thus, both tasks share the learned spatio-temporal representation while maintaining independent output spaces.

The proposed framework achieved a good performance, but several limitations have been spotted. Limited Generalization evaluation was performed using the Cholec80 dataset. Larger multi-center data should be used for validation to determine the robustness in other clinical settings. Challenging Surgical Conditions: Surgical smoke, bleeding, tissue occlusion, lens fogging and fast camera movements may result in a decreased performance.

Even after temporal modeling, we observe that there is the possibility of wrong classification at times, when the phases are close by each other visually. Even in the presence of focal loss and class-balanced sampling, rare surgical phases & instruments are harder to recognize which is referred to as Class Imbalance. In sum, the proposed framework is an eficient combination of spatial feature extraction and temporal modeling and shows good promise regarding its application in future intelligent operating rooms.

The work proposed in this paper was a clinically interpretable multi-task deep learning method for simultaneously showing surgical phase recognition and real-time instrument localization for a laparoscopic cholecystectomy. The proposed framework adopts a hierarchical Swin Transformer for spatial feature extraction and a Temporal Transformer Encoder (TEc) to model intricate temporal relationships, while introducing a parallel YOLOv11s detector that precisely localizes surgical instruments in frames. In addition, a rolling-queue temporal smoothing method was implemented to enhance the stability in prediction, so as to prevent the Phase-Flicker phenomenon during real-time prediction. The proposed framework was evaluated on the Cholec80 benchmark dataset, yielding experimental results of a high precision of 81.64%, macro average F1-score of 66.85% and Quadratic Weighted Kappa score of 89.89%. The mAP @ 0.5 of the YOLOv11s detector for finding instrument vars in the real-time environment was also reliable, reaching 0.863 in the overall consideration. These results have shown that such workflow recognition obtained using this type of spatiotemporal instrument-aware information can provide better accuracy and nice clinical interpretation, and that it is well embedded in intelligent computer-assisted surgery.

Despite the successful outcomes provided by the proposed framework, it was tested on only one benchmark data set, and lack of time interval flexibility could impede long-distance procedural modeling operations. In the future, multi-center clinical validation, adaptive temporal modeling, domain adaptation, and self-supervised learning will be explored to further enhance the system’s robustness and generalization. More integration of surgical work flow recognition and localization will also be written about so as to enhance reliable decision assistance in future robotic surgery system and the intelligent operating rooms.

## Statements and Declarations

## Funding

The authors declare that no funds, grants, or other support were received during the preparation of this manuscript.

## Competing Interests

The authors have no competing interest.

## Author Contributions

All authors contributed to the study conception and design. S.B. and M.S. performed coding implementation, material preparation, data collection, and analysis. The first draft of the manuscript was written by S.B. and M.S., while all authors commented on previous versions of the manuscript. All authors read and approved the final manuscript while S.B. supervised the entire research process.

## Data Availability

The Cholec80 dataset analyzed during the current study is publicly available. Requests for access can be directed to the dataset creators.

## Code Availability

The code will be made publicly available upon acceptance at https://github. com/ac005sheekar/LapChol-Surgery-Swin-Cholec80. The surgical phase recognition demonstration will be released at https://youtu.be/ZyNXr5PbIHA

## Ethics Approval

Not applicable. This study used a publicly available dataset (Cholec80) and did not involve human participants or animals.

## References

[1] Twinanda, A.P., Shehata, S., Mutter, D., Marescaux, J., de Mathelin, M. & Padoy, N. (2017). “EndoNet: A deep architecture for recognition tasks on laparoscopic videos.” IEEE Transactions on Medical Imaging, 36(1), pp. 86–97.

[2] Jin, Y., Dou, Q., Chen, H., Yu, L., Qin, J., Fu, C.-W. & Heng, P.-A. (2018). “SV-RCNet: Workflow recognition from surgical videos using recurrent convolutional network.” IEEE Transactions on Medical Imaging, 37(5), pp. 1114–1126.

[3] Czempiel, T., Paschali, M., Keicher, M., Simson, W., Feussner, H., Kim, S.T. & Navab, N. (2020). “TeCNO: Surgical phase recognition with multi-stage temporal convolutional networks.” In: Medical Image Computing and Computer-Assisted Intervention (MICCAI 2020), LNCS 12263, pp. 343–352.

[4] Gao, X., Jin, Y., Long, Y., Dou, Q. & Heng, P.-A. (2021). “Trans-SVNet: Hybrid embedding aggregation for surgical workflow analysis.” International Journal of Computer Assisted Radiology and Surgery, 16, pp. 1793–1803.

[5] Jin, Y., Long, Y., Chen, C., Zhao, Z., Dou, Q. & Heng, P.-A. (2021). “Temporal Memory Relation Network for Workflow Recognition From Surgical Video.” IEEE Transactions on Medical Imaging, 40(7), pp. 1911–1923.

[6] Liu, Y., Boels, M., Garcia-Peraza-Herrera, L.C., Vercauteren, T., Dasgupta, P., Granados, A. & Ourselin, S. (2023). “LoViT: Long Video Transformer for surgical phase recognition.” In: Medical Image Computing and Computer-Assisted Intervention (MICCAI 2023), pp. 356–366.

[7] Funke, I., Bodenstedt, S., Oehme, F., von Bechtolsheim, F., Weitz, J. & Speidel, S. (2019). “Using 3D convolutional neural networks to learn spatiotemporal features for automatic surgical gesture recognition in video.” In: Medical Image Computing and Computer-Assisted Intervention (MICCAI 2019), pp. 467–475.

[8] Zia, A., Hung, A., Essa, I. & Jarc, A. (2018). “Surgical activity recognition in robot-assisted radical prostatectomy using deep learning.” In: Medical Image Computing and Computer-Assisted Intervention (MICCAI 2018), pp. 419–427.

[9] Padoy, N. (2019). “Machine and deep learning for workflow recognition during surgery.” Minimally Invasive Therapy & Allied Technologies, 28(2), pp. 82–90.

[10] Lalys, F. & Jannin, P. (2014). “Surgical process modelling: A review.” International Journal of Computer Assisted Radiology and Surgery, 9, pp. 495–511.

[11] Maier-Hein, L., Vedula, S.S., Speidel, S., Navab, N., Kikinis, R., Park, A., Eisenmann, M., Feussner, H., Forestier, G., Giannarou, S., Hashizume, M., Katic, D., Kenngott, H., Kranzfelder, M., Malpani, A., M¨arz, K., Neumuth, T., Padoy, N., Pugh, C., Schoch, N., Stoyanov, D., Taylor, R., Wagner, M., Hager, G.D. & Jannin, P. (2017). “Surgical data science for next-generation interventions.” Nature Biomedical Engineering, 1, pp. 691–696.

[12] Ward, T.M., Hashimoto, D.A., Ban, Y., Rattner, D.W., Inoue, H. & Lillemoe, K.D. (2021). “Artificial intelligence in surgery: The next frontier.” Annals of Surgery, 273(2), pp. 233–235.

[13] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. & Guo, B. (2021). “Swin Transformer: Hierarchical vision transformer using shifted windows.” In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10012–10022.

[14] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. & Houlsby, N. (2021). “An image is worth 16×16 words: Transformers for image recognition

[15] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. & Polosukhin, I. (2017). “Attention is all you need.” In: Advances in Neural Information Processing Systems (NeurIPS), pp. 5998–6008.

[16] He, K., Zhang, X., Ren, S. & Sun, J. (2016). “Deep residual learning for image recognition.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.

[17] Redmon, J., Divvala, S., Girshick, R. & Farhadi, A. (2016). “You only look once: Unified, real-time object detection.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779–788.

[18] Ren, S., He, K., Girshick, R. & Sun, J. (2017). “Faster R-CNN: Towards real-time object detection with region proposal networks.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(6), pp. 1137–1149.

[19] Lin, T.-Y., Doll´ar, P., Girshick, R., He, K., Hariharan, B. & Belongie, S. (2017). “Feature pyramid networks for object detection.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2117– 2125.

[20] Sarikaya, D., Corso, J.J. & Guru, K.A. (2017). “Detection and localization of robotic tools in robot-assisted surgery videos using deep neural networks for region proposal and detection.” IEEE Transactions on Medical Imaging, 36(7), pp. 1542–1549.

[21] Jin, A., Yeung, S., Jopling, J., Krause, J., Azagury, D., Milstein, A. & Fei-Fei, L. (2018). “Tool detection and operative skill assessment in surgical videos using region-based convolutional neural networks.” In: IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 691–699.

[22] Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. & Bengio, Y. (2014). “Learning phrase representations using RNN encoder-decoder for statistical machine translation.” In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724–1734.

[23] Kingma, D.P. & Ba, J. (2015). “Adam: A method for stochastic optimization.” In: International Conference on Learning Representations (ICLR).

[24] Loshchilov, I. & Hutter, F. (2019). “Decoupled weight decay regularization.” In: International Conference on Learning Representations (ICLR).

[25] Smith, L.N. (2017). “Cyclical learning rates for training neural networks.” In: IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 464– 472.

[26] Smith, L.N. & Topin, N. (2019). “Super-convergence: Very fast training of neural networks using large learning rates.” Proceedings of SPIE, 11006, p. 1100612.

[27] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. & Chintala, S. (2019). “PyTorch: An imperative style, high-performance deep learning library.” In: Advances in Neural Information Processing Systems (NeurIPS), pp. 8024–8035.

[28] Iofe, S. & Szegedy, C. (2015). “Batch normalization: Accelerating deep network training by reducing internal covariate shift.” In: Proceedings of the 32nd International Conference on Machine Learning (ICML), pp. 448–456.

[29] Szegedy, C., Vanhoucke, V., Iofe, S., Shlens, J. & Wojna, Z. (2016). “Rethinking the inception architecture for computer vision.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2818–2826.

[30] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. & Fei-Fei, L. (2009). “ImageNet: A large-scale hierarchical image database.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248–255.

[31] Sudre, C.H., Li, W., Vercauteren, T., Ourselin, S. & Cardoso, M.J. (2017). “Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.” In: Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (DLMIA), pp. 240–248.

[32] Milletari, F., Navab, N. & Ahmadi, S.-A. (2016). “V-Net: Fully convolutional neural networks for volumetric medical image segmentation.” In: Fourth International Conference on 3D Vision (3DV), pp. 565–571.

[33] Ronneberger, O., Fischer, P. & Brox, T. (2015). “U-Net: Convolutional networks for biomedical image segmentation.” In: Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015), pp. 234–241.

[34] Tan, M. & Le, Q.V. (2019). “EficientNet: Rethinking model scaling for convolutional neural networks.” In: Proceedings of the 36th International Conference on Machine Learning (ICML), pp. 6105–6114.

[35] Wang, X., Girshick, R., Gupta, A. & He, K. (2018). “Non-local neural networks.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7794–7803.

[36] Lea, C., Vidal, R. & Hager, G.D. (2016). “Temporal convolutional networks: A unified approach to action segmentation.” In: European Conference on Computer Vision (ECCV) Workshops, pp. 47–54.

[37] Carreira, J. & Zisserman, A. (2017). “Quo Vadis, Action Recognition? A new model and the Kinetics dataset.” In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6299–6308.

[38] Simonyan, K. and Zisserman, A. (2014). “Two-stream convolutional networks for action recognition in videos.” Advances in Neural Information Processing Systems (NeurIPS), pp. 568–576.

[39] Islam, M., Seenivasan, L., Ming, L.C. and Ren, H. (2020). “Learning and reasoning with the graph structure representation in robotic surgery.” Medical Image Computing and Computer-Assisted Intervention (MICCAI 2020), pp. 627–636.

[40] Stauder, R., Ostler, D., Kranzfelder, M., Koller, S., Feussner, H. and Navab, N. (2016). “The TUM LapChole dataset for surgical workflow analysis.” Proceedings of the 17th Annual Meeting of the German Society for Computer- and Robot-Assisted Surgery (CURAC), pp. 115–120.

[41] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. and Zagoruyko, S. (2020). “End-to-end object detection with transformers.” European Conference on Computer Vision (ECCV), pp. 213–229.

[42] Bertasius, G., Wang, H. and Torresani, L. (2021). “Is space-time attention all you need for video understanding?” Proceedings of the 38th International Conference on Machine Learning (ICML).

[43] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Luˇci´c, M. and Schmid, C. (2021). “ViViT: A video vision transformer.” Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6836–6846.

[44] Yan, S., Xiong, X., Arnab, A., Lu, Z., Zhang, M., Sun, C. and Schmid, C. (2022). “Multiview transformers for video recognition.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3323– 3333.

[45] Radosavovic, I., Kosaraju, R.P., Girshick, R., He, K. and Dollar, P. (2020). “Designing network design spaces.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10428–10436.

[46] Jin, Y., Dou, Q., Chen, H., Yu, L., Qin, J., Fu, C.-W. & Heng, P.-A. (2018). “SV-RCNet: Workflow recognition from surgical videos using recurrent convolutional network.” IEEE Transactions on Medical Imaging, 37(5), pp. 1114–1126.