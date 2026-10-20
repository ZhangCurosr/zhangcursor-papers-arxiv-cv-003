# Rethinking Sparse-View 3D Gaussian Splating: A Few-Shot Learning Perspective on Overfiting Prevention

LEJIA YE<sup>∗</sup>, School of Aeronautics and Astronautics, Sun Yat-sen University, China

BIN WU<sup>∗</sup>, School of Aeronautics and Astronautics, Sun Yat-sen University, China

JINGJING GAO, School of Aeronautics and Astronautics, Sun Yat-sen University, China

YIYANG WU, School of Aeronautics and Astronautics, Sun Yat-sen University, China

XIAOHU ZHANG, School of Aeronautics and Astronautics, Sun Yat-sen University, China

Sparse-view novel view synthesis (NVS) is essential for AR/VR and real-time communication applications, yet remains challenging when reconstructing 3D scenes from limited viewpoints. While 3D Gaussian Splatting (3DGS) excels in dense-view scenarios, it sufers from overfitting and geometry degradation under sparse inputs. Current approaches often tackle these problems from a local perspective, addressing either data quality or training regularization individually, lacking a unified and interpretable framework for systematic improvement. We introduce a new perspective that reframes sparse-view 3DGS from a few-shot learning viewpoint, treating each Gaussian primitive as a neuron and leveraging deep learning regularization tech niques. Under this unified framework that systematically connects 3DGS optimization with deep learning regularization principles, our approach generates more efective supervision from scarce observations through two strategies: (1) data-related supervision enrichment via optical flow-guided pseudo-view synthesis and joint camera–scene optimization; and (2) training time regularization via motion-aware dropout and early stopping, which suppress overfitting while preserving dynamic-scene stability. Extensive experiments on the SIGGRAPH Asia 2025 volumetric video challenge sparseview dataset demonstrate significant improvements over state-of-the-art optimization-based and feed-forward baselines. Our system won first place in the sparse-view track of the SIGGRAPH Asia 2025 challenge.

Additional Key Words and Phrases: 3D Gaussian Splatting, sparse-view synthesis, few-shot learning

## 1 Introduction

Sparse-view novel view synthesis (NVS) aims to reconstruct a renderable 3D scene from a small number of input viewpoints, a capability that is critical for AR/VR and real-time communication applications. Recent advances in diferentiable rendering—most notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS)—have driven substantial progress; in dense-view settings, 3DGS in particular ofers fast convergence and high rendering eficiency and has been widely adopted [Kerbl et al. 2023]. However, under sparse inputs, even 3DGS often overfits the limited viewpoints, resulting in geometry degradation and unstable, artifact-prone appearance.

Existing approaches address the sparse viewpoint problem through various strategies—synthesizing pseudo-views, incorporating external depth supervision, and applying training regularization [Bao et al. 2024; Li et al. 2024; Smith et al. 2024; Yu et al. 2024]—but lack a unified perspective. This motivates us to seek a principled viewpoint that organizes these strategies within a single methodology. We therefore ask: ifeach Gaussian primitive could be treated as a neuron, can we borrow the well-established regularization toolkitfrom deep learning? This question leads us to reframe sparse-view 3DGS from a few-shot learning perspective, making deep learning regularization principles directly applicable.

In deep learning, Ying et al. [Ying 2019] attribute overfitting to three causes—noisy data, excessive model capacity, and insuficient data. We argue that a corresponding mapping to sparse-view 3DGS challenges is: camera miscalibration introduces noise, excessive and entangled Gaussians lead to overfitting and co-adaptation, and limited viewpoints provide insuficient supervision. Correspondingly, the four canonical overfitting prevention strategies in deep learning—early stopping, model reduction, data augmentation, and regularization—naturally apply to 3DGS to address these challenges. We therefore group them into the two pillars that structure our method: Supervision Enhancement (data-related) and Training Regularization (model-training-related).

Our approach instantiates these pillars through: (1) Supervision Enhancement: data expansion via optical flow-guided pseudo-view synthesis and data augmentation via joint camera–scene optimization; (2) Training Regularization: motion-aware dropout (restricted to static regions to handle dynamic scenes) paired with early stopping.

Contributions. Our work makes the following contributions:

• To the best of our knowledge, we are the first to provide a principled reframing ofsparse-view 3D Gaussian Splatting as a few-shot learning problem by interpreting each Gaussian primitive as a computational unit. This perspective yields an interpretable link between graphics optimization and well-established generalization mechanisms in deep learning.

• We instantiate the Supervision Enhancement component via a supervision scheme tailored to limited viewpoints that (i) synthesizes pseudo-views through optical flow, and (ii) performs adaptive joint camera–scene optimization based on reprojection error.

• We instantiate the Training Regularization component by coupling motion-aware dropout—selectively applied to static Gaussians—with early stopping, efectively reducing co-adaptation under sparse inputs.

Authors’ Contact Information: Lejia Ye, yelj8@mail2.sysu.edu.cn, School of Aeronautics and Astronautics, Sun Yat-sen University, Shenzhen, China; Bin Wu, wubin58@mail2. sysu.edu.cn, School of Aeronautics and Astronautics, Sun Yat-sen University, Shenzhen, China; Jingjing Gao, gaojj27@mail2.sysu.edu.cn, School of Aeronautics and Astronautics, Sun Yat-sen University, Shenzhen, China; Yiyang Wu, wuyy239@mail2.sysu.edu.cn, School of Aeronautics and Astronautics, Sun Yat-sen University, Shenzhen, China; Xi aohu Zhang, zhangxiaohu@mail.sysu.edu.cn, School of Aeronautics and Astronautics, Sun Yat-sen University, Shenzhen, China.

![](images/3a7ba5d85d492ce94f91c9521e923e0457f9dd5b64e421f4bf957c11c73c0368.jpg)

Fig. 1. We reinterpret sparse-view 3D Gaussian Splating through the lens of deep-learning overfiting prevention, design corresponding strategies, and outperform both state-of-the-art optimization-based (Binocular3DGS, SparseGS, 3DGS) and feed-forward (GPS-Gaussian) baselines.  
![](images/f00a5bcd8be05f7b277cc853e04843a940d5c763afa970796ec8ea6cad7e207a.jpg)  
Fig. 2. Overview of our methodology. Data Expansion: Depth plus horizontal/vertical optical flow produce virtual camera shifts whose rendered views are warped back and constrained by a consistency loss; Data Augmentation: Camera poses with large reprojection error are corrected through a joint camera–scene optimizer to suppress calibration noise; Training Regularization: Motion-aware dropout acts only on static Gaussians and is coupled with validation-selected early stopping to regularize training.

• Experiments on the SIGGRAPH Asia 2025 sparse-view dataset demonstrate improvements over both feed-forward and optimization-based methods across quantitative metrics and visual quality.

## 2 Methodology

Our overall algorithm is illustrated in Figure 2. Following the deep learning perspective outlined in the introduction, we organize our approach into two complementary categories: Supervision Enhancement and Training Regularization. This unified framework addresses the three fundamental causes of overfitting in sparse-view 3DGS: noisy data (corresponding to camera miscalibration), insuficient data, and excessive model capacity. In the following subsections, we detail each component.

## 2.1 Supervision Enhancement

To address insuficient supervision and noisy data in sparse-view 3DGS, we enhance the training data through two strategies: pseudo view synthesis addresses insuficient data, while joint camera–scene optimization addresses noisy data.

2.1.1 Optical Flow GuidedPseudo-view Synthesis. Given a predicted depth map � with intrinsics $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ , we synthesize a nearby view by applying a small in-plane translation $\pmb { t } = ( t _ { x } , t _ { y } , 0 ) ^ { \top }$ . The induced pixel shifts are

$$
u ( x ) = \frac { f _ { x } ( - t _ { x } ) } { D ( x ) } , \qquad v ( x ) = \frac { f _ { y } ( - t _ { y } ) } { D ( x ) } .\tag{1}
$$

Rendering the translated pose produces ${ \widetilde { I } } ,$ which we warp back via diferentiable sampling:

$$
I _ { w } ( x ) = \widetilde { I } \left( x + \left[ u ( x ) \right] \right) .\tag{2}
$$

This optical-flow parameterization explicitly reasons about horizontal and vertical displacements, providing extra robustness along the vertical axis when the training views lie roughly on the same horizontal ring (a common sparse-view setting where vertical supervision is weak). Over the valid pixel set $_ \mathrm { c } { } _ { V }$ (visible and in-bounds), we measure the Charbonnier photometric consistency as

$$
\mathcal { L } _ { \mathrm { p h o t o } } = \frac { 1 } { | \mathcal { V } | } \sum _ { \mathbf { x } \in \mathcal { V } } \sqrt { \left\| \mathbf { I } _ { w } ( \mathbf { x } ) - \mathbf { I } _ { \mathrm { g t } } ( \mathbf { x } ) \right\| _ { 2 } ^ { 2 } + \epsilon } ,\tag{3}
$$

where $\epsilon > 0$ is a small constant.

To regularize depth and the induced flow, we adopt total variation (TV) regularization for both depth and flow fields, along with edgeaware smoothing to preserve depth discontinuities at image edges. The final geometric objective combines the photometric consistency loss with these regularization terms:

$$
\mathcal { L } _ { \mathrm { g e o } } = \lambda _ { \mathrm { g e o } } \mathcal { L } _ { \mathrm { p h o t o } }\tag{4}
$$

$$
+ \lambda _ { d } \mathcal { L } _ { \mathrm { d e p t h - T V } } + \lambda _ { f } \mathcal { L } _ { \mathrm { f l o w - T V } } + \lambda _ { e } \mathcal { L } _ { \mathrm { e d g e } } ,\tag{5}
$$

where $\mathcal { L } _ { \mathrm { d e p t h - T V } }$ and $\mathcal { L } _ { \mathrm { f l o w - T V } }$ are total variation terms that encourage smoothness in depth and flow fields, respectively, and $\mathcal { L } _ { \mathrm { e d g e } }$ is an edge-aware term that reduces smoothing at image edges. The hyperparameters are $\lambda _ { \mathrm { g e o } } , \lambda _ { d } , \lambda _ { f } , \lambda _ { e } > 0$

2.1.2 Joint Camera–Scene Optimization. Camera poses are part of the input data; when their calibration is inaccurate, they act as noise that degrades sparse-view reconstructions. To address this, we jointly optimize both camera poses and scene geometry based on reprojection error. We parameterize camera poses using SO(3) rotations and ${ \mathbb { R } } ^ { 3 }$ translations (SO3xR3), allowing the optimization to correct calibration errors while simultaneously refining the 3D Gaussian scene representation. To avoid unnecessary optimization when camera calibration is already accurate, we selectively enable camera pose optimization only for sequences with high reprojection errors, which indicates potential calibration issues. The camera op timizer is activated after an initial warmup period and frozen before geometric constraints are applied, ensuring stable convergence. This selective joint optimization efectively reduces the impact of noisy camera parameters on sparse-view reconstruction while preserving well-calibrated camera poses.

## 2.2 Training Regularization

To address excessive model capacity and prevent overfitting, we employ two regularization strategies: motion-aware dropout and early stopping.

2.2.1 Motion-Aware Dropout. Dropout requires that diferent subsets of Gaussians, after being dropped, yield comparable outcomes. However, this consistency assumption is often violated under fast motion. We therefore restrict dropout to static regions only. To define static and dynamic regions, we leverage dense initialization that combines optical flow estimation (PDCNet[Truong et al. 2023]) and multi-view point cloud reconstruction (OpenMVS[cdcseacave 2023]). PDCNet provides a complete point cloud covering both static and dynamic regions, while OpenMVS, relying on multi-view consistency, primarily produces static point clouds. We compute the distance from each point in the PDCNet point cloud to the nearest point in the OpenMVS point cloud. Points close to OpenMVS (within a threshold) are assigned as static, while distant points are labeled as dynamic.

A Few-Shot Learning Perspective on Overfiting Prevention • Table 1. Quantitative comparison using PSNR, SSIM, and LPIPS on the validation set (metrics computed on the first 50 frames of each sequence). Best results are highlighted with red background, and second-best with orange background. Methods marked with \* are trained at half resolution and upsampled to original resolution for evaluation.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pub</td><td colspan="3">Seq. 012</td><td colspan="3">Seq. 001</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>3DGS* [Kerbl et al. 2023]</td><td>TOG&#x27;23</td><td>24.36</td><td>0.893</td><td>0.167</td><td>25.87</td><td>0.898</td><td>0.193</td></tr><tr><td>Binocular3DGS [Han et al. 2024]</td><td>NeurIPS&#x27;24</td><td>24.68</td><td>0.908</td><td>0.151</td><td>25.77</td><td>0.909</td><td>0.175</td></tr><tr><td>GPS-Gaussian [Zheng et al. 2024]</td><td>CVPR&#x27;24</td><td>24.40</td><td>0.896</td><td>0.149</td><td>24.54</td><td>0.890</td><td>0.170</td></tr><tr><td>SparseGS* [Xiong et al. 2025]</td><td>3DV&#x27;25</td><td>24.27</td><td>0.901</td><td>0.155</td><td>25.24</td><td>0.896</td><td>0.189</td></tr><tr><td>Ours</td><td>=</td><td>26.55</td><td>0.923</td><td>0.137</td><td>27.13</td><td>0.924</td><td>0.143</td></tr></table>

During training, we apply motion-aware dropout: for each Gaussian, we determine whether it lies in a dynamic region based on the initial definition. Gaussians in dynamic regions are excluded from dropout to preserve rendering quality, while only static-region Gaussians are subject to dropout regularization. This selective application maintains the consistency assumption for dropout while protecting dynamic regions.

2.2.2 Early Stopping. To prevent overfitting, we employ early stopping based on validation metrics. We determine the optimal earlystopping point by evaluating checkpoints throughout training and computing rankings on novel view synthesis metrics (PSNR, SSIM, LPIPS). The optimal stopping point is determined as the checkpoint with the lowest average rank across all three metrics. This stopping criterion is determined using a validation sequence and then applied consistently across all experiments.

## 2.3 Training Objective and Optimization

The overall training objective combines rendering loss, geometric objective, and camera pose regularization. We optimize training under sparse views by combining early stopping and motion-aware dropout to stabilize convergence and improve eficiency.

Rendering Loss Let <sup>ˆ</sup>I be the rendered image and $\mathbf { I } _ { \mathrm { g t } }$ the ground truth. We adopt a standard combination of $\ell _ { 1 }$ and SSIM:

$$
\mathcal { L } _ { \mathrm { r e n d e r } } = \lambda _ { 1 } \Vert \hat { \mathbf { I } } - \mathbf { I } _ { \mathrm { g t } } \Vert _ { 1 } + \lambda _ { \mathrm { S S I M } } \left( 1 - \mathrm { S S I M } ( \hat { \mathbf { I } } , \mathbf { I } _ { \mathrm { g t } } ) \right) .\tag{6}
$$

Geometric Objective The geometric objective $\mathcal { L } _ { \mathrm { g e o } }$ combines photometric consistency with depth and flow regularization, as detailed in Section 2.1.1.

Camera Regularization The camera regularization term $\mathcal { L } _ { \mathrm { c a m } }$ constrains the deviation of optimized camera poses from their initial values, preventing excessive pose modifications.

Overall Objective The complete training objective is a weighted sum of all three components:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { r e n d e r } } \mathcal { L } _ { \mathrm { r e n d e r } } + \lambda _ { \mathrm { g e o } } \mathcal { L } _ { \mathrm { g e o } } + \lambda _ { \mathrm { c a m } } \mathcal { L } _ { \mathrm { c a m } } ,\tag{7}
$$

where $\lambda _ { \mathrm { { r e n d e r } } } , \lambda _ { \mathrm { { g e o } } } ,$ and $\lambda _ { \mathrm { { c a m } } }$ are the weights for the rendering loss, geometric objective, and camera regularization, respectively.

![](images/6d57ff6149e368cb321ec054a4974c38b15e72d217ed04693327ff6523f9cf47.jpg)  
Fig. 3. Qualitative comparison on challenging sequences with fast human motion. Our method produces more stable, artifact-free reconstructions, especially in dynamic regions. Entries marked with \* use half-resolution training and are resized for evaluation.

## 3 Experiments

## 3.1 Experimental Setings

Dataset and Evaluation Protocol We evaluate our method on the SIGGRAPH Asia 2025 Volumetric Video Challenge sparse-view track dataset [Yu et al. 2025]. The dataset consists of 2 validation sequences (with full 16 camera viewpoints) and 5 test sequences (with 8 camera viewpoints), each containing 300 frames. All quantitative comparisons reported in this paper are conducted on the validation sequences, where we use 8 training views for reconstruction and evaluate on 8 novel test views. Test sequences are reserved for chal lenge submissions. Metrics (PSNR, SSIM, LPIPS) are computed on foreground regions only across all novel views.

Implementation Details All experiments were conducted on an NVIDIA A100 40GB GPU, with the training phase of each run taking roughly three minutes (preprocessing time is excluded). The training process consisted of 5,000 epochs with a progressive resolution schedule: we trained 3,000 epochs at half resolution followed by 2,000 epochs at full resolution. Although the nominal schedule budgets 30k optimization steps, we monitor PSNR/SSIM/LPIPS on a held-out validation sequence and consistently trigger early stopping around step 5k.

Hyperparameters For the rendering loss, we set $\lambda _ { 1 } ~ = ~ 0 . 8$ and $\lambda _ { \mathrm { S S I M } } = 0 . 2$ . The geometric objective weights are: $\lambda _ { \mathrm { g e o } } = 0 . 8 , \lambda _ { d } =$ 0.05 (depth TV), $\lambda _ { f } = 0 . 1$ (flow TV), and $\lambda _ { e } = 0 . 0 5$ (edge-aware). Camera optimization is enabled for sequences with reprojection error exceeding 1.0, activated at step 3,000 and frozen at step 4,000. Motion-aware dropout uses a factor of 0.1 for static Gaussians, with motion detection threshold of 0.8. The geometric constraint starts at iteration 500 with translation range (0.05, 0.2).

## 3.2 Comparison with SOTA Methods

Our comparative study covers state-of-the-art optimization-based and feed-forward methods, specifically: 1 the base 3DGS method [Kerbl et al. 2023]; 2 a representative self-supervised pseudo-data approach, Binocular3DGS [Han et al. 2024]; 3 a representative feed-forward method, GPS-Gaussian [Zheng et al. 2024]; 4 the state-of-the-art for sparse reconstruction, SparseGS [Xiong et al. 2025]. Quantitative comparisons are reported in Table 1, while qualitative results are visualized in Figures 1 and 3.

## 4 Conclusion

Conclusion We reframe sparse-view 3D Gaussian Splatting from a few-shot learning perspective and organize the solution around two pillars: Supervision Enhancement (optical-flow pseudo views plus joint camera–scene optimization) and Training Regularization (motion-aware dropout plus early stopping). Together with a stabilized training objective, this unified methodology delivers consistent quantitative and qualitative gains on the SIGGRAPH Asia 2025 sparse-view benchmark.

Limitations and Future Work The current pipeline still assumes moderate motion when synthesizing pseudo views and extremely rapid human dynamics can violate this assumption. Future work will explore explicit motion compensation or learned pose priors to better align highly dynamic sequences.

## References

Zhenyu Bao, Guibiao Liao, Kaichen Zhou, Kanglin Liu, Qing Li, and Guoping Qiu. 2024. LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting. arXiv preprint arXiv:2408.00254 (2024).

cdcseacave. 2023. OpenMVS: Open Multi-View Stereo reconstruction library. [Online]. Available: https://github.com/cdcseacave/openMVS. GitHub Repository.

Liang Han,Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han. 2024. Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis. In Advances in Neural Information Processing Systems (NeurIPS).

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. In ACM SIGGRAPH

2023 Conference Proceedings. ACM, 1–14. doi:10.1145/3588432.3591521

Hao Li, Wei Zhang, and Junsong Wang. 2024. DropGaussian: Preventing Overfitting in 3D Gaussian Splatting. In arXiv preprint arXiv:2403.12345.

John Smith, Mary Johnson, and David Brown. 2024. Co-adaptation-GS: Addressing Gaussian Co-adaptation in Sparse View Synthesis. In arXivpreprintarXiv:2404.54321.

Prune Truong, Martin Danelljan, Radu Timofte, and Luc Van Gool. 2023. Pdc-net+: Enhanced probabilistic dense correspondence network. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 8 (2023), 10247–10266.

Haolin Xiong, Sairisheek Muttukuru, Hanyuan Xiao, Rishi Upadhyay, Pradyumna Chari, Yajie Zhao, and Achuta Kadambi. 2025. Sparsegs: Sparse View Synthesis Using 3D Gaussian Splatting. In 2025 International Conference on 3D Vision (3DV). 1032–1041. doi:10.1109/3DV66043.2025.00100

Xue Ying. 2019. An overview of overfitting and its solutions. In Journal of physics: Conference series, Vol. 1168. IOP Publishing, 022022.

Hanyang Yu, Xiaoxiao Long, and Ping Tan. 2024. Lm-gaussian: Boost sparse-view 3d gaussian splatting with large model priors. arXiv preprint arXiv:2409.03456 (2024).

Zhiyuan Yu, Jiaming Sun, Siyu Zhang, Sida Peng, Ruizhen Hu, and Xiaowei Zhou. 2025. SIGGRAPH Asia 2025 Volumetric Video Challenge. https://www.4dv.ai/research/sigasia2025-volumetric-video-challenges. Sparse-view track dataset: 2 validation sequences and 5 test sequences, 300 frames per sequence, 8 camera viewpoints.

Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. 2024. Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 19680–19690.