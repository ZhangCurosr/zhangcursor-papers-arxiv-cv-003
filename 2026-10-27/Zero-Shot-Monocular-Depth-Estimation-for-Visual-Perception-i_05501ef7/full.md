# Zero-Shot Monocular Depth Estimation for Visual Perception in Lower-Limb Prostheses

Ishi Vijayvergiya, Marc Holle, Youssef Dawoud, Amir El-Ghoussani, Michele De Vita, Rohan Asthana,¨

Philipp Beckerle, and Vasileios Belagiannis

Friedrich-Alexander-Universitat Erlangen-N ¨ urnberg¨

Erlangen, Germany

{ishi.vijayvergiya, marc.hoelle, youssef.dawoud, amir.el-ghoussani,

michele.de.vita, rohan.asthana, philipp.beckerle, vasileios.belagiannis}@fau.de

Abstract—Monocular depth estimation has advanced rapidly with the development of foundation models that demonstrate strong zero-shot generalisation across diverse visual domains. However, the applicability of these models to visual perception for lower-limb prosthetics remains largely unexplored. Reliable depth perception is particularly important for prosthesis users, who often experience significant challenges when navigating stairs, uneven terrain, and other complex environments. To address this gap, we introduce the Lower-Limb Prosthesis (LP) dataset, a benchmark designed to evaluate depth perception from an emulated knee-level prosthetic viewpoint. Using this dataset, we evaluate four state-of-the-art monocular depth estimation foundation models. Our results provide the first systematic benchmark of monocular depth estimation performance from a prosthetic viewpoint, while highlighting both the challenges and the potential of deploying foundation models in prosthetic perception systems. The code and dataset are available at https://github.com/Ishivijay/lp dataset.

Index Terms—Monocular depth estimation, foundation models, zero-shot, medical prosthesis

## I. INTRODUCTION

Visual perception is fundamental to autonomous navigation and environmental understanding. In the context of medical prosthetics, the ability to accurately perceive and interpret the surrounding environment is critical for safe and effective device operation. However, current lower-limb prostheses lack robust perceptual capabilities, leaving users at significantly increased risk of falls, particularly when navigating stairs, uneven terrain, and cluttered environments [1]. Integrating reliable depth perception into prosthetic systems could enable real-time hazard detection, adaptive gait planning, and enhanced environmental awareness. By improving the ability to navigate complex environments safely, such capabilities could reduce fall risk, enhance mobility, and increase user confidence in daily activities.

Computer vision approaches are increasingly being integrated into lower-limb wearable robotics to provide active environmental awareness. In lower-limb prosthetics, visionin-the-loop control using gaze-tracking cues can be used to identify user intention and anticipate terrain transitions, providing volitional control of active knee-ankle prostheses [2].

![](images/c4ca97dbd550eee45b4175b03021b0f46d1ad47aa874c36d285161db3e041610.jpg)  
Fig. 1. LP dataset acquisition setup. The dataset was collected with a wearable camera mounted on the lower limb in four configurations: above or below the knee, on each leg. These configurations yield egocentric recordings from an emulated knee-level prosthetic viewpoint during locomotion across a diverse set of mounting positions.

Visual-inertial odometry can be combined with 3D terrain reconstruction to classify locomotion [3]. Similarly, depthaware pose estimation has also been integrated into lowerlimb exoskeleton systems to enable real-time gait analysis and prevent asymmetric locomotion and falls [4], as well as to facilitate gesture-driven control interfaces leveraging monocular depth features [5]. While these approaches demonstrate the value of vision, they generally rely on task-specific training, custom multicamera setups, or cooperative rehabilitation environments.

Monocular depth estimation has become an important capability in the field of computer vision [6]. Recent foundation models have achieved impressive zero-shot generalisation across a variety of scenes. Models such as Marigold [7], Depth Anything V2 [8], Depth Pro [9], and VGGT [10] employ a range of training strategies to predict dense depth maps from single RGB images. However, these models are typically developed and benchmarked on conventional datasets featuring a standard eye-level perspective. These advances raise a natural question: Can state-of-the-art monocular depth estimation foundation models generalize zero-shot to the unique, highly dynamic, and knee-level perspective of a prosthesis-mounted

camera?

While prior datasets have captured egocentric or wearable camera images of locomotion environments, they focus mainly on RGB-based scene classification or activity recognition and do not provide paired ground-truth depth annotations. Existing depth estimation benchmarks, on the other hand, target indoor scenes NYUv2 [11], ScanNet [12], or autonomous driving KITTI [13], none of which reflect the specific viewing angle, motion patterns, and close-range stair navigation experienced by a prosthesis user. To our knowledge, no existing dataset provides paired RGB-depth data recorded from a lower-limbmounted camera during locomotion, capturing the perspective of a lower-limb prosthesis.

To address this gap, we introduce the Lower-limb Prosthesis (LP) dataset, the first dataset designed specifically for visual perception in medical prosthesis applications. The LP dataset consists of synchronized RGB-depth image pairs captured from a camera mounted on a lower-limb during stair ascent and descent across six recording sessions. The dataset covers a variety of scene elements, including staircases, corridors, landings, and doorways, all captured from an emulated kneelevel prosthetic viewpoint during locomotion.

Using this dataset, we carry out a comprehensive zero-shot evaluation of four state-of-the-art monocular depth estimation foundation models. This study provides the first empirical evidence on the robustness and limitations of modern depth estimation models when applied to data from a lower-limbmounted camera emulating a prosthetic viewpoint, enabling novel research for visual perception in medical prostheses.

The main contributions of this paper are:

1) We introduce the LP dataset, the first dataset for visual depth perception for lower-limb prosthesis, comprising six stair-navigation sequences with ground-truth depth from a depth camera.

2) We present a systematic zero-shot evaluation of four state-of-the-art monocular depth estimation foundation models (Marigold, Depth Anything V2, Depth Pro, and VGGT) on this novel domain.

3) We provide a detailed analysis of model performance across different scene types and discuss the implications for integrating depth estimation into prosthetic perception systems.

## II. RELATED WORK

## A. Monocular Depth Estimation

Monocular depth estimation aims to infer scene depth from a single RGB image and has been extensively studied in computer vision. Early supervised approaches relied on paired RGB–depth datasets, such as NYU Depth V2 [11] and KITTI [13], and employed convolutional neural networks for depth prediction [14]. To reduce dependence on costly ground-truth depth annotations, self-supervised methods were subsequently introduced, leveraging photometric consistency across video frames or stereo image pairs as a supervisory signal [15]. More recently, research has shifted toward foundation models designed to generalize across diverse domains and visual environments. Depth Anything [16] and its successor, Depth Anything V2 [8], are trained at scale using a combination of labelled and unlabelled data. Marigold [7] repurposes the Stable Diffusion architecture for monocular depth estimation through a diffusion-based denoising framework. Depth Pro [9] combines a DINOv2 backbone with a dedicated focal-length estimation module, while VGGT [10] leverages vision transformers to achieve strong depth estimation accuracy and cross-domain generalisation. Despite their impressive performance on standard benchmark datasets, the robustness of these models under the unconventional camera viewpoints encountered in lower-limb prosthetic applications remains largely unexplored.

## B. Egocentric and Wearable Vision Datasets

Several datasets have been developed using wearable or body-mounted cameras to support mobility and visual perception research. For example, the ExoNet database [17] provides wearable-camera images of walkable environments for scene classification, while other datasets have been proposed for depth estimation using head-mounted or chest-mounted cameras. However, these datasets are primarily designed for general egocentric perception and do not capture the unique visual characteristics of a lower-limb prosthesis-mounted camera, including its low viewpoint, reduced camera height, and distinctive motion dynamics during locomotion and stair negotiation. As a result, they provide limited insight into the performance of depth estimation models in prosthetic navigation scenarios, highlighting the need for datasets specifically designed to evaluate perception from a prosthetic perspective.

## C. Visual Perception for Assistive Devices

Research on intelligent lower-limb prostheses has traditionally focused on mechanical design, control strategies, and electromyography (EMG)-based intent recognition [18]. Recent work started incorporating environmental sensing to improve prosthetic functionality. Examples include the use of wearable-camera imagery for locomotion environment classification [17] and the identification of walkable surfaces from egocentric images [19]. While these approaches provide valuable semantic information about the environment, they do not explicitly capture its geometric structure. Incorporating depth-aware perception into prosthetic systems could enable advanced capabilities such as obstacle detection, step-height estimation, terrain assessment, and real-time path planning. Despite these potential benefits, the application of monocular depth estimation to lower-limb prosthetic navigation remains largely unexplored.

## III. DATASET

## A. Dataset Acquisition and Setup

The LP dataset comprises indoor scenes collected in a multi-storey building using an RGB camera and a co-located depth camera mounted on a participant’s knee to emulate the viewpoint of a knee-level prosthesis-mounted camera. Representative examples are shown in Fig. 2. Data were collected under controlled conditions in a relatively uncluttered indoor setting to establish a clean baseline benchmark and avoid recording unconsented individuals; door thresholds, densely furnished spaces, other people, and moving animals were excluded. This initial collection focused on constant-speed walking in the selected indoor setting. In total three healthy male participants in their mid-twenties took part in the study. One participant was from Europe, and the other two were from Asia. The height of each individual varied from 1.7 m to 1.87 m. All measurements were collected while participants walked at a constant speed. Data were acquired using four mounting configurations: left-above (LA), left-below (LB), right-above (RA), and right-below (RB), where the camera was positioned either above or below the left or right knee joint, as shown in Fig. 1. The sensor was secured to the participant’s leg using a mounting strap, with power and data transmitted via a USB Type-C 3.2 connection to a laptop carried by an accompanying person. The dataset contains 3,144 synchronized RGB–depth image pairs distributed across six recording sequences encompassing stair ascent, stair descent, and corridor walking, as summarized in Table I. Data acquisition was performed using an Orbbec Femto Bolt depth camera, which operates on Microsoft’s Azure Kinect Timeof-Flight (ToF) technology. The device features a 1-megapixel indirect ToF (iToF) depth sensor configured in wide field-ofview (WFOV) mode, yielding a $1 2 0 ^ { \circ } \times 1 2 0 ^ { \circ }$ field of view and producing depth maps at a resolution of 1024 × 1024 pixels. Co-registered RGB images were captured via the integrated 4K RGB sensor and synchronized with the depth maps at the hardware level.

TABLE I  
OVERVIEW OF THE SIX RECORDING SEQUENCES IN THE LP DATASET.
<table><tr><td>Sequence</td><td>Direction</td><td>Images</td></tr><tr><td>LA downstairs</td><td>Down</td><td>428</td></tr><tr><td>LA upstairs</td><td> $\mathrm { U p }$ </td><td>629</td></tr><tr><td>LB upstairs</td><td>Up</td><td>591</td></tr><tr><td>RA downstairs</td><td>Down</td><td>375</td></tr><tr><td>RA upstairs</td><td>Up</td><td>622</td></tr><tr><td>RB downstairs</td><td>Down</td><td>499</td></tr><tr><td>Total</td><td></td><td>3,144</td></tr></table>

## B. Dataset Structure and Formats

For each sequence, the dataset provides sharpened RGB images sharpen\_rgb, raw depth maps raw\_depth, colorized depth visualizations colored\_depth, temporally smoothed depth smooth\_depth and transformed depth transformed\_depth.

## IV. EXPERIMENTS

We perform zero-shot inference using four state-of-theart foundation models for monocular depth estimation: Marigold [7], Depth Anything V2 [8], Depth Pro [9], and VGGT [10]. To ensure a consistent comparison, all models were evaluated using the same three-stage pipeline: (1) zeroshot inference on raw RGB images, (2) post-processing and alignment of predicted depth maps with the spatial dimensions of the ground-truth depth data, and (3) quantitative evaluation using standard depth estimation metrics.

1) Model Configurations: Marigold was evaluated using the pre-trained $\mathtt { m a r i g o l d - d e p t h - v 1 - 1 }$ diffusion checkpoint [7], with half-precision (FP16) inference enabled. Depth Anything V2 was deployed with the large Vision Transformer encoder (ViT-L) configuration [8], and inference was configured to produce raw depth maps. Depth Pro was loaded using its official model factory, generating metric depth predictions in meters together with estimated camera focal lengths [9]. VGGT, originally developed for multi-view 3D reconstruction, was adapted for single-image depth estimation using the VGGT-1B checkpoint (1 billion parameters) [10] obtained from Hugging Face. Input images were preprocessed using the official VGGT preprocessing pipeline in crop mode [10], and depth maps were extracted from the model outputs for evaluation.

2) Evaluation Protocol: All models were evaluated on the complete set of sequences in the LP dataset under identical experimental conditions. When the spatial resolution of a predicted depth map differed from that of the ground-truth depth map, bilinear interpolation was used to resize the prediction to the ground-truth dimensions.

To enable fair comparison across models with differing scale and shift characteristics, a least-squares alignment procedure was applied per image. Specifically, a global scale factor s and shift term t were estimated by solving

$$
\mathbf { G } \mathbf { T } = { \boldsymbol { s } } \cdot \mathbf { p r e d } + t ,\tag{1}
$$

where GT denotes the ground-truth depth map and pred denotes the predicted depth map [7]. The evaluation metrics included absolute relative error (Abs Rel), squared relative error (Sq Rel), root mean square error (RMSE), root mean square error in logarithmic space $\mathrm { ( R M S E _ { l o g } ) }$ , and threshold accuracies $\delta _ { 1 } , \delta _ { 2 } .$ , and $\delta _ { 3 }$ . Both the least-squares alignment in Eq. 1 and all evaluation metrics were computed exclusively over valid ground-truth depth pixels $( G T ~ > ~ 0 )$ , excluding unmeasured sensor dropout regions.

## V. DISCUSSION

## A. Quantitative Results

Tables II and III summarize the zero-shot depth estimation performance of the four foundation models on the LP dataset following least-squares alignment. On the staircase scene (Table II), VGGT achieved the strongest performance, obtaining the best Abs Rel (0.1532), Sq Rel (0.0818), RMSE (0.3449), and $\delta _ { 1 } ~ ( 0 . 8 0 9 9 )$ . On the corridor scene (Table III), Marigold outperformed all other models across every evaluation metric, achieving the lowest Abs Rel (0.1963), Sq Rel (0.0813), RMSE (0.3357), and $\mathrm { R M S E _ { l o g } } ~ ( 0 . 2 2 0 5 )$ , together with the highest threshold accuracies $( \delta _ { 1 } = 0 . 7 5 9 2 , \delta _ { 2 } = 0 . 9 3 7 1$ , and $\delta _ { 3 } ~ = ~ 0 . 9 6 6 9 )$ . Depth Pro and VGGT remained competitive across both scenes, whereas Depth Anything V2 consistently produced the largest errors and lowest threshold accuracies. Despite differences in ranking, all models demonstrated effective zero-shot generalisation to the emulated knee-level prosthetic viewpoint, with Marigold and VGGT exhibiting the strongest overall performance.

![](images/ffedea7ef1846046c13da9ef6386afd7096caf802c7659b7bb69efa4084e0e7a.jpg)  
Fig. 2. Qualitative comparison of zero-shot monocular depth estimation on the LP dataset. Nine representative scenes (rows), spanning corridors, staircases, landings, and doorways, are shown with, from left to right, the RGB input, the Ground truth and the depth predicted by Marigold, Depth Anything V2, Depth Pro, and VGGT. Depth is rendered with a colormap, where warm colors denote near and cool colors denote far surfaces. Black regions in the ground truth denote invalid sensor returns that are caused by surfaces beyond the operating range, grazing-angle specular reflections on floor tiles, or areas outside the depth field of view.

## B. Qualitative Results

Figure 2 presents qualitative comparisons of the depth maps generated by the four foundation models. By visual inspection of Figure 2, all models successfully capture the global scene geometry and depth gradients, clearly distinguishing nearby structures from more distant regions. Depth Anything V2 and VGGT generally produce smoother and more spatially coherent depth maps, preserving large-scale scene layouts across a variety of environments. Marigold also generates plausible depth estimates but appears more sensitive to local structures and depth discontinuities, resulting in finer-grained depth variations. Depth Pro produces metric depth predictions with clear foreground–background separation, although its outputs are occasionally more over-smoothed than those of the other models. Despite the challenging knee-level camera viewpoint and motion characteristics of the lower-limbmounted setup, all methods generate depth predictions that remain broadly consistent with the underlying scene structure. These qualitative results demonstrate the strong zeroshot generalisation capabilities of contemporary monocular depth estimation foundation models in prosthetic perception scenarios.

ZERO-SHOT DEPTH ESTIMATION RESULTS ON THE LP DATASET (STAIRCASE SCENE) WITH LEAST-SQUARES ALIGNMENT. LOWER IS BETTER FOR ↓ METRICS; HIGHER IS BETTER FOR ↑ METRICS. BEST IN BOLD.  
TABLE II
<table><tr><td>Metric</td><td>Marigold</td><td>DA-V2</td><td>Depth Pro</td><td>VGGT</td></tr><tr><td>Abs Rel↓</td><td>0.1568</td><td>0.2066</td><td>0.1599</td><td>0.1532</td></tr><tr><td>Sq Rel ↓</td><td>0.0858</td><td>0.1158</td><td>0.0829</td><td>0.0818</td></tr><tr><td>RMSE↓</td><td>0.3549</td><td>0.3947</td><td>0.3489</td><td>0.3449</td></tr><tr><td> $\mathrm { R M S E } _ { \mathrm { l o g } } \downarrow$ </td><td>0.1972</td><td>0.2806</td><td>0.2006</td><td>0.2011</td></tr><tr><td> $\delta _ { 1 } \uparrow$ </td><td>0.8075</td><td>0.7179</td><td>0.8008</td><td>0.8099</td></tr><tr><td> $\delta _ { 2 } \uparrow$ </td><td>0.9434</td><td>0.8955</td><td>0.9408</td><td>0.9434</td></tr><tr><td> $\delta _ { 3 }$  ↑</td><td>0.9756</td><td>0.9489</td><td>0.9754</td><td>0.9755</td></tr></table>

TABLE III

ZERO-SHOT DEPTH ESTIMATION RESULTS ON THE LP DATASET (CORRIDOR SCENE) WITH LEAST-SQUARES ALIGNMENT. LOWER IS BETTER FOR ↓ METRICS; HIGHER IS BETTER FOR ↑ METRICS. BEST IN BOLD.
<table><tr><td>Metric</td><td>Marigold</td><td>DA-V2</td><td>Depth Pro</td><td>VGGT</td></tr><tr><td> $\mathbf { A b s \ R e l \downarrow }$  Sq Rel ↓</td><td>0.1963 0.0813</td><td>0.2739</td><td>0.2219</td><td>0.2477</td></tr><tr><td>RMSE↓</td><td>0.3357</td><td>0.1513 0.5067</td><td>0.0936 0.3595</td><td>0.1241 0.4084</td></tr><tr><td> $\mathrm { R M S E } _ { \log \mathrm { ~ \downarrow ~ } }$ </td><td>0.2205</td><td>0.3980</td><td>0.2483</td><td>0.2898</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\delta _ { 1 } \uparrow$ </td><td>0.7592</td><td>0.5442</td><td>0.6632</td><td>0.6100</td></tr><tr><td> $\delta _ { 2 } \uparrow$ </td><td>0.9371</td><td>0.8119</td><td>0.9184</td><td>0.8824</td></tr><tr><td> $\delta _ { 3 } \uparrow$ </td><td>0.9669</td><td>0.9277</td><td>0.9635</td><td>0.9581</td></tr></table>

## C. Future Work

Although the examined zero-shot foundation models demonstrate strong generalisation, their large parameter sizes and high computational demands currently prevent them from being deployed directly on active prosthetic hardware. To address this limitation, future work should investigate two complementary directions. First, model compression, pruning, and quantization techniques should be explored to enable these large foundation models to run efficiently on resourceconstrained hardware. Second, deploying lightweight monocular depth architectures [20], [21] is necessary for real-time edge processing. Real-time gait adaptation requires visual feedback to be processed within the swing phase (typically under 400 ms). Future work should therefore focus on deploying both optimised foundation models and lightweight adapted networks on wearable edge accelerators. Furthermore, because prosthetic locomotion is safety-critical, integrating monocular depth uncertainty estimation [22] and pixel-wise out-of-distribution detection [23] is essential to identify unreliable depth predictions and unexpected terrain hazards. In this context, the LP dataset provides a relevant benchmark for evaluating the impact of such optimizations on depth estimation performance under prosthesis-specific operating conditions.

## VI. CONCLUSION

In this study, we present the Lower-Limb Prosthesis (LP) dataset, a benchmark comprising synchronised RGB and ground-truth depth data collected from a knee-level camera mounted on healthy participants to emulate a lower-limb prosthesis. The dataset addresses a significant gap in prosthetic perception research by providing depth-aware visual data captured during stair navigation and indoor locomotion from this emulated prosthetic perspective. Using the LP dataset, we evaluated four state-of-the-art monocular depth estimation foundation models and found that, despite being trained on conventional benchmarks, they generalise effectively to this previously unexplored domain. These baseline results demonstrate the feasibility of using zero-shot foundation models to support downstream prosthetic applications such as obstacle detection, terrain assessment and adaptive gait planning. This opens up the possibility of advanced visual perception in intelligent lower-limb prostheses. By introducing a dedicated benchmark and establishing strong baseline performance, this work provides a foundation for future research into visual perception in lower-limb prostheses.

## REFERENCES

[1] W. C. Miller, M. Speechley, and B. Deathe, “The prevalence and risk factors of falling and fear of falling among lower extremity amputees,” Archives of physical medicine and rehabilitation, vol. 82, no. 8, pp. 1031–1037, 2001.

[2] X. Chen, Z. Chen, Y. Wang, S. Yin, C. Chen, J. Huang, and C. Fu, “A gaze-guided human locomotion intention recognition and volitional control method for knee-ankle prostheses,” IEEE Transactions on Cognitive and Developmental Systems, 2025.

[3] S. Zhao, Z. Yu, Z. Wang, H. Liu, Z. Zhou, L. Ruan, and Q. Wang, “A learning-free method for locomotion mode prediction by terrain reconstruction and visual-inertial odometry,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 3895–3905, 2023.

[4] Y. Wang, Z. Pei, C. Wang, and Z. Tang, “Depth-aware pose estimation using deep learning for exoskeleton gait analysis,” Scientific Reports, vol. 13, no. 1, p. 22681, 2023.

[5] S. Qiu, Y. Liu, C. Wang, J. Zhang, W. Xian, and C. Chen, “Visionbased gesture-driven control method for rehabilitation lower extremity exoskeleton robots,” IEEE Sensors Journal, 2025.

[6] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab, “Deeper depth prediction with fully convolutional residual networks,” in 2016 Fourth international conference on 3D vision (3DV). IEEE, 2016, pp. 239–248.

[7] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9492–9502.

[8] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, “Depth anything v2,” Advances in Neural Information Processing Systems, vol. 37, pp. 21 875–21 911, 2024.

[9] A. Bochkovskiy, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. Richter, and V. Koltun, “Depth pro: Sharp monocular metric depth in less than a second,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 75 602–75 637.

[10] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “Vggt: Visual geometry grounded transformer,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 5294– 5306.

[11] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, “Indoor segmentation and support inference from rgbd images,” in European conference on computer vision. Springer, 2012, pp. 746–760.

[12] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839.

[13] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354–3361.

[14] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014.

[15] C. Godard, O. Mac Aodha, M. Firman, and G. J. Brostow, “Digging into self-supervised monocular depth estimation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 3828– 3838.

[16] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth anything: Unleashing the power of large-scale unlabeled data,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 10 371–10 381.

[17] B. Laschowski, W. McNally, A. Wong, and J. McPhee, “Exonet database: Wearable camera images of human locomotion environments,” Frontiers in Robotics and AI, vol. 7, p. 562061, 2020.

[18] M. R. Tucker, J. Olivier, A. Pagel, H. Bleuler, M. Bouri, O. Lambercy, J. d. R. Millan, R. Riener, H. Vallery, and R. Gassert, “Control strategies for active lower extremity prosthetics and orthotics: a review,” Journal of neuroengineering and rehabilitation, vol. 12, no. 1, p. 1, 2015.

[19] B. Laschowski, W. McNally, A. Wong, and J. McPhee, “Environment classification for robotic leg prostheses and exoskeletons using deep convolutional neural networks,” Frontiers in Neurorobotics, vol. 15, p. 730965, 2022.

[20] J. Hornauer, L. Nalpantidis, and V. Belagiannis, “Visual domain adaptation for monocular depth estimation on resource-constrained hardware,” in IEEE/CVF International Conference on Computer Vision Workshops, ICCVW 2021, Montreal, QC, Canada, October 11-17, 2021. IEEE, 2021, pp. 954–962.

[21] M. Rudolph, Y. Dawoud, R. Guldenring, L. Nalpantidis, and V. Belagian-¨ nis, “Lightweight monocular depth estimation through guided decoding,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 2344–2350.

[22] J. Hornauer, A. El-Ghoussani, and V. Belagiannis, “Revisiting gradientbased uncertainty for monocular depth estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 6, pp. 4395– 4408, 2025.

[23] M. Holle, W. Kellermann, and V. Belagiannis, “Uncertainty-aware¨ likelihood ratio estimation for pixel-wise out-of-distribution detection,” in 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). IEEE, 2025, pp. 783–793.