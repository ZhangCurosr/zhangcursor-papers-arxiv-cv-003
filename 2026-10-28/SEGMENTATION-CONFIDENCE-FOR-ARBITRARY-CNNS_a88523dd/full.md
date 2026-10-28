# SEGMENTATION CONFIDENCE FOR ARBITRARY CNNS

Baris Oguz<sup>⋆</sup>, Xing Yao<sup>†</sup>, Claudia Tawil<sup>⋆</sup>, Alison Pouch<sup>⋆</sup>, Brett Byram<sup>†</sup> Gabriel Arenas<sup>⋆</sup>, Nadav Schwartz<sup>⋆</sup>, Ipek Oguz<sup>†</sup>

<sup>⋆</sup> University of Pennsylvania <sup>†</sup> Vanderbilt University

## ABSTRACT

Convolutional neural networks (CNN) are widely used for medical image segmentation. However, they lack an inherent measure of confidence. Existing methods for estimating prediction uncertainty often require a custom network architecture and/or dedicated training setups and need to be trained from scratch. In addition, the deduction of a single confidence score per input from the uncertainty maps is not always straightforward. In this study, we explore voxel-based uncertainty estimation methods that can be added to existing segmentation pipelines without training from scratch. We also propose a novel method to estimate a single confidence score from the uncertainty maps. Our evaluations on different ROIs and modalities suggest that our confidence score can predict segmentation performance in unseen data.

Index Terms— Confidence, uncertainty, segmentation

## 1. INTRODUCTION

Deep learning models, and in particular CNNs, are widely used in medical image segmentation [1, 2, 3, 4, 5, 6, 7]. However, these models often provide little insight into their decision process [8, 9, 10], and they lack an inherent measure of confidence. The produced probability maps provide an indirect signal of confidence per voxel. However, there is no obvious way to produce a single measure of confidence for the entire image. Since CNNs are highly successful at segmentation, the probabilities are usually higher inside and around the ROI, and mostly 0 further from the ROI. This results in a sharp transition between the foreground and background, rather than a gradual increase, limiting the ability to use volume-wide statistics as a measure of confidence. While classification models output probabilities per class, which can estimate confidence, this is not available for segmentation.

There has been research to overcome this shortcoming [11], often with Bayesian frameworks. Gal and Ghahramani [12] have shown that CNNs trained with dropout can approximate Bayesian inference by keeping dropout active during test-time and performing multiple forward passes for classification. Ghoshal et al. [13] propose Monte Carlo drop weights, and decompose predictive probabilities into aleatoric and epistemic uncertainties. Teye et al. [14] use batch normalization layers instead of dropout. Wang et al. [15] use test-time augmentation and Monte Carlo simulation. There have also been non-Bayesian approaches. Some methods [16, 17] train an ensemble of models to estimate classification uncertainty. Kohl et al. [18] propose a probabilistic U-Net. Yao et al. [19] combine an aleatoric uncertainty-based correction strategy with semi-automated prompt augmentation. Most of these approaches require dedicated architectures [20] or training an ensemble of models [21] to accomplish segmentation and confidence/uncertainty estimation simultaneously. Many of them produce an uncertainty map, but do not deduce a single score for the entire image. This limits their usability in typical medical image analysis workflows. Notably, Wang et al. [15] use the volume variation coefficient (VVC) of multiple forward passes to estimate structure-wise uncertainty.

While CNNs perform well overall, a poor prediction for a particular image may lead to mistakes. Flagging uncertain predictions could provide alternatives, such as obtaining a repeat scan if there is an image quality issue. Another use case is flagging uncertain outputs in large-scale studies for visual quality control, manual correction, or exclusion from downstream analysis. Confidence scores can also be useful in active learning [22, 23] for prioritizing low-confidence images for annotation, as these are likely not well represented in the training dataset. Finally, a model’s ability to report its own confidence is a significant step towards explainable AI.

In this study, we use two alternatives (Sec. 2.2) to produce multiple predictions from a single input image, which can be readily implemented as an add-on to existing segmentation pipelines. We then explore two approaches to estimate a voxel-based uncertainty map (Sec. 2.3), and propose a novel method to estimate a single, image-wide confidence score (Sec. 2.4). We evaluate our method on placenta (ultrasound) and hippocampus (MRI) datasets to test its generalizability.

## 2. METHODS

## 2.1. Data and Segmentation Models

Placenta 3D ultrasound (3DUS) dataset: This dataset contains N = 210 images from 121 unique subjects in the first trimester, and N = 49 images from 41 unique subjects taken during the second trimester. Placentas were manually segmented using ITK-SNAP (www.itksnap.org). Images were down-sampled to $1 2 8 \times 1 2 8 \times 1 2 8$ voxels before training.

Automated quantification of the placenta from 3DUS data is challenging [24], and quick identification of sub-optimal segmentations during clinical screening can allow the option of a repeat scan while the patient is still in the clinic.

Hippocampus MRI dataset: We used the hippocampus task from the Medical Segmentation Decathlon [25]. We used the $N = 2 6 0$ training set with anterior and posterior hippocampus labels. Images were up-sampled to ${ 6 \bar { 4 } ^ { 3 } }$ voxels.

Segmentation: We employed typical 2D and 3D U-Net models with 2 levels of down / up-sampling, and skip-connections. All convolutional layers have $3 ^ { 2 } \ : ( 3 ^ { 3 } )$ pixel (voxel) kernels. The placenta dataset has a single ROI. The hippocampus dataset has 2 ROIs (anterior/posterior), and we trained a separate model per ROI. We used binary cross-entropy loss.

For our 2D models for both the placenta and the hippocampus, we observed that a single model can be successfully trained on sagittal and coronal slices, since the general shape and position of the ROI are roughly similar along these axes. Slices along the axial plane differ considerably from the other 2 planes, and inclusion in training significantly lowered performance. Therefore, our final 2D models were trained on sagittal and coronal slices. The placenta model has 364K trainable parameters, while the two 2D hippocampus models each have 162K. We also trained 3D models for each hippocampus ROI (1,938K trainable parameters each). We used 4-fold cross-validation. Since our focus is not on segmentation accuracy, but rather on the performance of confidence scoring on the test set, we used 1 partition for training and 3 for testing within each fold to provide more test cases, rather than the usual 3:1 train/test split. The random partitioning for placentas was based on the unique subjects to avoid leakage.

## 2.2. Test-time augmentation strategies

Test-time flip augmentations: We used test-time augmentation with horizontal/vertical flips for the 2D models. We thus applied a combination of 4 augmentations per 2D slice. As our models were trained on both the sagittal and coronal planes, we produced 8 full-volume 3D outputs per input.

This approach was viable for our 2D models, since the random online augmentations during training included the horizontal/vertical flips. Specifically, for training the placenta model, we used horizontal/vertical flips, in-plane rotations, cropping, blurring, and gamma adjustment. For the hippocampus models, we only applied horizontal/vertical flips. It is common practice to utilize random augmentations like these during training, and modifying the test-time loop to include the flip augmentations and produce multiple outputs per input is a straightforward approach to apply our confidence scoring method to existing segmentation pipelines.

Test-time noise augmentations: Some models may perform worse with test-time flips, e.g., if augmentations were not used during training, or if the ROI is highly asymmetric. To test our method’s applicability in such cases, we trained two 3D models for the two hippocampus ROIs without any augmentations. At test-time, we added random Gaussian noise $( t o r c h . r a n d n \times 0 . 5 )$ to a random 20% subset of the voxels. We blurred $( \sigma = 1 )$ the noise before adding to the original image for a more realistic look. For each input, we produced 15 noisy versions, resulting in 16 outputs. While other noise distributions may be more suitable for various modalities, this general approach is applicable to any existing 2D or 3D models, whether augmentations were used during training or not.

## 2.3. Uncertainty estimation strategies

Standard deviation (stdev): First, we simply calculate the voxel-wise stdev of the multiple probability maps. We normalize to [0, 1] via min-max scaling. This gives us an approximate uncertainty map u(x), in which values close to 0.0 denote voxels where the multiple predictions mostly agree, and values close to 1.0 denote the most disagreement. As expected, we usually observed highest uncertainty along the boundary of the ROI (Fig. 1c), as well as in regions with artifacts or other noise. We used this approach in our 2D models. Entropy: When the multiple predictions all have a probability close to 0.5 for a given voxel, since there is agreement, the low stdev registers as a low uncertainty voxel, even though the 0.5 probabilities suggest high uncertainty. To remedy this, we also calculated the binary entropy $H ( p ) = - p \ln p - ( 1 -$ $p ) \ln ( 1 - p )$ for all the outputs and took their voxel-wise mean. In our 3D models, we min-max scaled this mean entropy map, and averaged it with the stdev map u(x) (Fig. 2).

## 2.4. Confidence score estimation

We use the voxel-wise mean of the multiple output probability maps to create a final probability map $p ( x )$ and threshold it to create a binary mask $M _ { 0 } ( x ) = \{ p ( x ) \geq 0 . 5 0 \}$ (Fig. 1d).

Next, we calculated a base mask volume $V _ { 0 } = | M _ { 0 } ( x ) |$ This volume is independent of the uncertainty $u ( x )$ , since it includes all voxels that have $p ( x ) ~ \geq ~ 0 . 5 0$ regardless of uncertainty. If we introduce an uncertainty constraint such that only voxels under a given uncertainty are allowed, $\mathrm { e . g . }$ $u ( x ) \ \leq \ 0 . 9$ , then the segmentation would be more conservative and thus the resultant mask volume would be lower than $V _ { 0 }$ . We observe that if the overall model prediction for a given image $I ( x )$ was highly confident, there would be very few voxels with a high uncertainty $u ( x )$ . Conversely, in a lowconfidence prediction, there would be high uncertainty in substantially more voxels. This leads to the intuition that if the mask volume decreases rapidly as we tighten the uncertainty constraint, then the confidence for the prediction as a whole is likely low. In contrast, a very gradual decrease in volume suggests low uncertainty and high confidence (Fig. 3a, 3b).

![](images/cd1a7fa53952dd5be23d5e7ab4c876850471033930ce9142c5e136d138039e24.jpg)  
Fig. 1. a) 3DUS placenta input, b) ground truth, c) uncertainty map $u ( x )$ , d) thresholded segmentation mask $M _ { 0 } ( x )$

![](images/47688fa81c7d18a416b6d59cb7b4ed0eee1c716d0b6d1ead9935bc215d7c32d6.jpg)  
Fig. 2. A hippocampus sample. a) 3D MRI input, b) ground truth: anterior (red) and posterior (green) ROIs, c) stdev uncertainty map, d) entropy uncertainty map, e) stdev+entropy uncertainty map. (c-e) show $u ( x )$ for anterior ROI only.

To quantify this intuitive concept, we introduced $N _ { u }$ uncertainty constraints decreased by uniform steps $\Delta _ { u } = 1 / N _ { u } ,$ and calculated $N _ { u } - 1$ more masks similarly to $M _ { 0 }$ . In our experiments, we empirically used $N _ { u } ~ = ~ 1 0$ and therefore $\Delta _ { u } \ = \ 0 . 1$ . Thus, for example, $M _ { 1 }$ is the set of voxels in the segmentation that have a corresponding uncertainty of less than 0.90 $( p ( x ) \geq 0 . 5 0 \land u ( x ) \leq 0 . 9 0 )$ , and $M _ { 2 }$ only includes voxels in the mask that have an uncertainty less than 0.80. In general, for $i \in \left[ 0 , N _ { u } \mathrm { ~ - ~ } 1 \right]$ , we have $M _ { i } ( x ) = \{ p ( x ) \geq $ $0 . 5 \wedge u ( x ) \leq ( 1 - i * \Delta _ { u } ) \big \}$ , with volume $V _ { i } = \vert M _ { i } ( x ) \vert$

As mentioned above, $M _ { 0 }$ is the largest mask and the volumes decrease through $V _ { 0 }$ to $V _ { N _ { u } - 1 }$ . How these volumes change is informative (Fig. 3a vs. 3b). To capture this, we calculated the $N _ { u } - 1$ ratios of each volume to the base volume, $V _ { i } / V _ { 0 }$ , for $i \in [ 1 , N _ { u } - 1 ]$ . We also calculated the $N _ { u } - 2$ ratios of consecutive steps, $V _ { i + 1 } / V _ { i }$ , for $i \in [ 1 , N _ { u } - 2 ]$ . Since the volumes are monotonically decreasing from $V _ { 0 }$ through $V _ { N _ { u } - 1 }$ , all of these ratios are in the [0, 1] range.

V<sub>i</sub> is a measure of overall certainty (Fig. 3c). Dividing by $V _ { 0 }$ normalizes for object size. We note that simply averaging $u ( x )$ across $V _ { 0 }$ would be sensitive to object size, without a straightforward way to compare different subjects. Our formulation transforms the uncertainty map $u ( x )$ to a series of volumetric measures $V _ { i } ,$ allowing us to compare different subjects by simply expressing the volumes as the ratio $V _ { i } / V _ { 0 }$ . The $\Delta _ { u }$ determines the granularity of this transformation, and we found 0.1 to be adequate for our purposes. In contrast, $V _ { i + 1 } / V _ { i }$ is a measure of speed of uncertainty change (Fig. 3d): the blue curve locally has slower change than the red curve, thus a smaller $V _ { i + 1 } / V _ { i }$ . To obtain a single confidence score C from these ratios, we simply took their mean, as a balanced and robust indicator:

![](images/b304640251a81e59514951b81efc427c9ce63717086f618d78aaf4618b11d71b.jpg)  
Fig. 3. a) Segmentations with low uncertainty/high confidence. b) Segmentations with high uncertainty/low confidence. c) 1D illustration of $V _ { i }$ . The blue curve is overall more uncertain than the red curve, and thus has a lower $V _ { i } .$ . d) Illustration of $V _ { i + 1 } / V _ { i }$ . The blue curve has a (locally) slower change in uncertainty, and thus a smaller $V _ { i + 1 } / V _ { i }$

$$
C = \frac { \sum _ { i = 1 } ^ { N _ { u } - 1 } ( \frac { V _ { i } } { V _ { 0 } } ) + \sum _ { i = 1 } ^ { N _ { u } - 2 } ( \frac { V _ { i + 1 } } { V _ { i } } ) } { ( N _ { u } - 1 ) + ( N _ { u } - 2 ) }\tag{1}
$$

## 2.5. Correlation with Dice and outlier detection

We examined the correlation of our confidence C with the segmentation Dice. We also used these scores to identify poor segmentations. To do this in unsupervised manner, we applied an elliptic envelope model to the confidence scores (without taking Dice as input), fitting an ellipsoid around the central cluster of the data points, to separate inliers from outliers. We used the scikit-learn implementation of the model, with a contamination value of 0.1. Since we were interested in only the abnormally low confidence cases, we filtered out cases with above mean confidence (i.e., the highest confidence scores) from the outlier set. We then performed a two-tailed t-test to assess Dice differences between inliers and outliers.

## 2.6. Using confidence to select annotation candidates

We also explored how confidence scores can be used to select images to annotate and add to the training set from a pool of unlabeled candidates. We trained a base 2D model (Sec. 2.1) on 21 random images from the N = 121 placenta dataset. We split the remaining images into $N = 6 0$ test set, and $N = 4 0$ annotation candidates. Using the base model, we produced segmentations and confidence scores for the $N \ = \ 4 0$ candidates. We compared 5 strategies for selecting candidates: best Dice, worst Dice, best confidence, worst confidence, and random. We note that the best/worst Dice strategies were included as oracle baselines, and would not be available in practice, while confidence can be computed with the base model. We added the images selected by each strategy to the $N = 2 1$ base set in increments of 5 images, and trained new models. Since the number of images in the training set were limited, we trained 3 models for each scenario with different random seeds. We compared performance on the $N = 6 0$ test set.

## 3. RESULTS AND DISCUSSION

Compared methods: We compare our confidence score to the volume variation coefficient (VVC) method [15] for estimating a volume-wide confidence. Other methods noted in the Introduction do not produce a volume-wide score.

2D models: For the 2D models, we used flip augmentations during test-time, and estimated uncertainty with standard deviation (Fig. 4). For the placenta, the mean Dice was between 0.81-0.82 across the 4 folds. Using confidence $C ,$ the mean

![](images/72ca851a3ebdfba07e53811e5e8febc78a0695b4e4e845b49b057a9680c82576.jpg)  
Fig. 4. Dice-confidence and Dice-VVC correlations. Top, placenta; middle, anterior hippocampus; bottom, posterior hippocampus. 2D models with flip aug., stdev uncertainty. Blue/red dots denote inliers/outliers (Sec. 2.5), with shades corresponding to individual folds. Confidence has stronger correlation with Dice than VVC across each ROI/fold.

Dice difference between inliers-outliers were in the range 0.14-0.23. We note that the outlier detection only uses C, and the t-test is between the associated Dice scores rather than C (which would be lower in the outlier group, by construction). The inlier-outlier Dice difference was 0.07-0.10 with VVC. For anterior hippocampus, mean Dice was between 0.84-0.85. Inlier-outlier Dice difference was 0.09-0.11 for C and 0.04-0.08 for VVC. For posterior hippocampus, the mean Dice was between 0.82-0.84. Inlier-outlier Dice difference was 0.08-0.21 for C and 0.05-0.21 for VVC.

3D models: Here, we used Gaussian noise augmentation, and estimated uncertainty with stdev and entropy. For anterior hippocampus (not shown for brevity), the mean Dice was between 0.83-0.84. Inlier-outlier Dice difference was 0.05-0.07 with confidence, and 0.05-0.07 with VVC. For posterior hippocampus (Fig. 5), the mean Dice was between 0.79-0.81. Inlier-outlier Dice difference was 0.08-0.14 with confidence, and 0.07-0.14 with VVC. The inlier-outlier differences were significant for each fold $( p < 0 . 0 5 )$ for both C and VVC, in both 2D and 3D models.

Selection by confidence: Fig. 6 shows the mean test performance for each selection strategy, where N = 0 is the base model. Adding images with the worst Dice/confidence scores improves test performance faster than other strategies. The Dice scores of the “worst confidence” strategy were significantly better than the “random” strategy for the smaller N $( p < 0 . 0 1$ for $N \in \{ 5 , . . . , 2 0 \}$ , and $p < 0 . 0 3$ for $N = 2 5$ in paired t-tests). The performances converge, as expected, as we approach the entire candidate pool size, with the random strategy performing better at $N = 3 0 ( p < 0 . 0 3 )$ ).

![](images/146724c0d23f1e82f253a6723b9b33fbe177d99f1628f1a42b9639a21b7f31ec.jpg)  
Fig. 5. Posterior hippocampus Dice, confidence, VVC. 3D models with Gaussian noise aug., stdev+entropy uncertainty. Blue/red dots denote inliers/outliers (Sec. 2.5).

![](images/ed25524964591573c6b2912c9f899881bdd5e9d940285516a320824c9de75d55.jpg)  
Fig. 6. Candidate selection strategies for manual segmentation. Selecting images with the worst confidence significantly outperforms random selection, and approaches the oracle strategy (selection by worst Dice).

Discussion: We proposed a novel method to estimate a single confidence score for CNNs used in medical image segmentation, by examining how rapidly the segmentation volume decreases as we gradually allow less uncertainty.

Our confidence score C is highly correlated with the segmentation Dice. We observe that the upper left quadrant of the plots in Figs. 4 and 5 is typically empty, i.e., C is rarely high for low-Dice predictions in our experiments. This suggests high sensitivity, which is useful for practical workflows. C values are usually in the range of 0.60-0.90, and most cases with low Dice receive a C below 0.75. The unsupervised outlier detection experiment confirms this finding. For each ROI in Fig. 4, VVC fails to flag several low-Dice (< 0.6) cases that our method successfully captures (red arrows). Compared to VVC, our method has more consistent performance for a given task: e.g., for the anterior hippocampus (Fig. 4, middle row), the VVC correlations fluctuate from -0.247 to -0.668, whereas our correlations are stable in the [0.699, 0.711] range. Our correlations and inlier-outlier differences substantially surpass VVC counterparts in most experiments.

In our experiments we used test-time flips and Gaussian noise. These or other suitable augmentations may be used for new tasks. More extensive evaluation remains as future work.

Our proposed confidence score C shows a strong correlation with Dice in our extensive experiments, reliably indicating segmentation quality. Our method’s compatibility with existing segmentation pipelines and its applicability without requiring specialized architectures make it a versatile tool. This technique can help in many scenarios such as identifying poor-quality inputs or for selecting images for manual segmentation. Overall, our approach contributes a valuable layer of uncertainty estimation and quality control, enhancing the potential for increased clinical adoption of CNNs. Application in clinic remains as future work. Our code is available at github.com/MedICL-VU/ segmentation-confidence.

Acknowledgments: This work was supported, in part, by

NIH R01HD109739, U01-HD087180, R03-HD-069742, and Penn Presbyterian George L. and Emily McMichael Harrison Fund for Research in Obstetrics and Gynecology.

## 4. REFERENCES

[1] J Long, E Shelhamer, and T Darrell, “Fully convolutional networks for semantic segmentation,” in IEEE CVPR, 2015, pp. 3431–3440.

[2] O Ronneberger, P Fischer, and T Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI. Springer, 2015, pp. 234–241.

[3] K He, X Zhang, S Ren, and J Sun, “Deep residual learning for image recognition,” CVPR, pp. 770–778, 2016.

[4] G Huang, Z Liu, L Van Der Maaten, and KQ Weinberger, “Densely connected convolutional networks,” in CVPR, 2017, pp. 4700–4708.

[5] F Chollet, “Xception: Deep learning with depthwise separable convolutions,” CVPR, pp. 1251–1258, 2017.

[6] O Oktay, J Schlemper, LL Folgoc, M Lee, M Heinrich, K Misawa, K Mori, S McDonagh, NY Hammerla, B Kainz, et al., “Attention u-net: Learning where to look for the pancreas,” arXiv:1804.03999, 2018.

[7] F Isensee, PF Jaeger, SAA Kohl, J Petersen, and KH Maier-Hein, “nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, pp. 203–211, 2021.

[8] N Hasani, MA Morris, A Rhamim, RM Summers, E Jones, E Siegel, and B Saboury, “Trustworthy AI in medical imaging,” PET clinics, vol. 17, no. 1, 2022.

[9] H Chen, C Gomez, CM Huang, and M Unberath, “Explainable medical imaging ai needs human-centered design: guidelines and evidence from a systematic review,” NPJ digital medicine, vol. 5, no. 1, pp. 156, 2022.

[10] W Jin, X Li, M Fatehi, and G Hamarneh, “Guidelines and evaluation of clinical explainable ai in medical image analysis,” Medical image analysis, 2023.

[11] S Czolbe, K Arnavaz, O Krause, and A Feragen, “Is segmentation uncertainty useful?,” IPMI, 2021.

[12] Y Gal and Z Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in ICML. PMLR, 2016, pp. 1050–1059.

[13] B Ghoshal, A Tucker, B Sanghera, and W Lup Wong, “Estimating uncertainty in deep learning for reporting confidence to clinicians in medical image segmentation and diseases detection,” Computational Intelligence, vol. 37, no. 2, pp. 701–734, 2021.

[14] M Teye, H Azizpour, and K Smith, “Bayesian uncertainty estimation for batch normalized deep networks,” in ICML. PMLR, 2018, pp. 4907–4916.

[15] G Wang, W Li, M Aertsen, J Deprest, S Ourselin, and T Vercauteren, “Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with CNNs,” Neurocomputing, 2019.

[16] B Lakshminarayanan, A Pritzel, and C Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” NeurIPS, vol. 30, 2017.

[17] D Karimi, Q Zeng, P Mathur, A Avinash, S Mahdavi, I Spadinger, P Abolmaesumi, and SE Salcudean, “Accurate and robust deep learning-based segmentation of the prostate clinical target volume in ultrasound images,” Medical image analysis, vol. 57, pp. 186–196, 2019.

[18] S Kohl, B Romera-Paredes, C Meyer, J De Fauw, JR Ledsam, K Maier-Hein, SM Eslami, DJ Rezende, and O Ronneberger, “A probabilistic u-net for segmentation of ambiguous images,” NeurIPS, vol. 31, 2018.

[19] X Yao, H Liu, D Hu, D Lu, A Lou, H Li, R Deng, G Arenas, B Oguz, N Schwartz, et al., “False negative/positive control for sam on noisy medical images,” arXiv:2308.10382, 2023.

[20] H Akrami, A Joshi, S Aydore, and R Leahy, “Quantile regression for uncertainty estimation in VAEs with applications to brain lesion detection,” IPMI, 2021.

[21] A Mehrtash, WM Wells, CM Tempany, P Abolmaesumi, and T Kapur, “Confidence calibration and predictive uncertainty estimation for deep medical image segmentation,” IEEE TMI, vol. 39, no. 12, pp. 3868–3878, 2020.

[22] H Liu, H Li, X Yao, Y Fan, D Hu, BM Dawant, V Nath, Z Xu, and I Oguz, “Colossal: A benchmark for coldstart active learning for 3d medical image segmentation,” in MICCAI. Springer, 2023, pp. 25–34.

[23] V Nath, D Yang, BA Landman, D Xu, and HR Roth, “Diminishing uncertainty within the training pool: Active learning for medical image segmentation,” IEEE TMI, vol. 40, no. 10, pp. 2534–2547, 2020.

[24] N Schwartz, I Oguz, J Wang, A Pouch, N Yushkevich, S Parameshwaran, J Gee, P Yushkevich, and B Oguz, “Fully automated placental volume quantification from 3d ultrasound for prediction of small-for-gestational-age infants,” JUM, vol. 41, no. 6, pp. 1509–1524, 2022.

[25] A Simpson et al., “A large annotated medical image dataset for the development and evaluation of segmentation algorithms,” arXiv:1902.09063, 2019.