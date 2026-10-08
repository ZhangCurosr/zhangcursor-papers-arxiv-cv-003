# STORK: Spatio-Temporal Observation of uterine contRactions via neural networKs

Melissa Schween<sup>1,2,3</sup>, Tristan Gottwald<sup>2,3</sup>, Jordina Aviles Verdera<sup>1,2,3</sup>, Lisa Story<sup>4</sup>, Mary Rutherford<sup>4</sup>, and Jana Hutter<sup>1,2,3</sup>

1 CAIMED, Leibniz University Hannover, Hannover, Germany <sup>2</sup> L3S, Leibniz University Hannover, Hannover, Germany

<sup>3</sup> Institute of Information Processing, Leibniz University Hannover, Germany

4 Department for Biomedical Engineering, King’s College London, London, UK schween@tnt.uni-hannover.de

Abstract. Uterine contractions in fetal MRI are typically identified manually and discarded, limiting insights into contraction dynamics. We formalize Uterine Contractile Activity Detection (UCAD) as a weaklysupervised learning problem and introduce STORK, a multi-instance learning model trained on dynamic MRI series using only coarse, serieslevel labels. STORK factorizes 3D spatio-temporal convolutions into parallel branches across temporal hyperplanes to capture coherent tissue motion without the cost of full 4D convolutions. Per-frame embeddings, combining intensity and Demons-estimated displacement fields, are aggregated by a linear mean-pooling head. This ensures that frame-level contraction scores can be recovered post-hoc without frame-level training supervision. Evaluated on around 700 multi-vendor dynamic fetal MRI series, STORK achieves a series-level AUROC of 95.0% and AUPRC of 94.6%, substantially outperforming 3D ResNet and ConvNeXt baselines. Grad-CAM analysis suggests that the model draws on predictive features extending beyond the placenta into the uterine tissue, ofering an automated tool for richer phenotyping of uterine behavior.

Keywords: Fetal MRI · Relaxometry · Multi Instance Learning.

## 1 Introduction

Frequent antenatal health assessments are conducted typically by Ultrasound (US) to assess growth and Cardiotocography (CTG) to quantify uterine contractile activity, fetal state and reaction to contractions. Both, however, struggle to provide a comprehensive assessment. US is operator-dependent and sufers from a limited field-of-view particularly at late gestational age. CTG lacks spatial resolution and functional insights into the placental core. Dynamic Magnetic Resonance Imaging (MRI) addresses some of these challenges, providing wholeuterus spatio-temporal coverage and diverse functional contrasts. However, while used diagnostically in CTG, uterine contractions hamper MR imaging and contribute to artifacts. Currently, afected images are manually identified and discarded. Automated contraction detection would allow for explicit quantification of contractile activity, data-driven rejection, and adaptive imaging protocols.

![](images/1ac20b9221784b7818ab6ea84f2c8aef6e702fc8e7fd85bf3e7c543c14563699.jpg)  
Fig. 1: Illustration of contractile activity captured during a dynamic Multi-Echo Gradient Echo MRI scan, showing quantitative mean $\mathrm { T 2 ^ { * } }$ and volume (left), acquired data at the first echo time (center) and quantitative $\mathrm { T 2 ^ { * } }$ maps (right).

Figure 1 illustrates the quantitative $\mathrm { T 2 ^ { * } }$ and volumetric changes associated with contractions, motivating their detection from dynamic MRI data. We introduce a weakly-supervised solution trained on series-level labels, called STORK. Our contributions include (1) a formal problem definition of Uterine Contractile Activity Detection, (2) a hyperplane-factorized MIL architecture for dynamic MRI under weak series-level supervision, (3) interpretability analysis by leveraging a linear mean-pooling backbone, aggregating per-frame embeddings into series-level predictions while ensuring that frame-level scores can be recovered post-hoc, (4) discovery of discriminative features beyond the placenta, and (5) comprehensive comparison against convolutional and temporal baselines.

## 2 Related Work

## 2.1 Clinical Background and Uterine MRI

Fetal dynamic MRI, particularly T2\* relaxometry, is increasingly used for contractile assessment, capturing reductions in placental T2\*, volume changes and qualitative patterns such as darkened bands through the placental core [1, 3, 7, 15, 27]. Prior work has used image-based features to diferentiate contraction types [7], manually identified contractions to exclude corrupted quantitative data [2, 28], and demonstrated downstream hypoxic efects in the fetal brain and liver. Most recently, a retrospective multi-vendor analysis introduced a step towards automatic classification using thresholding on placental $\mathrm { T 2 ^ { * } }$ curves [3].

## 2.2 Machine Learning Background

Prior machine learning applications to fetal MRI focused on anatomical segmentation and quality enhancement [10,24], typically treating motion as an artifact.

In contrast, we utilize motion, via displacement fields, as the primary discriminative signal for contractile activity. For sequence-level annotations, Multiple Instance Learning (MIL) treats a sequence (e.g., frames) as a bag of instances, labeled positive if at least one instance is positive [16,23]. Permutation-invariant pooling aggregates per-instance features into a bag representation [16]. Learned attention also yields instance-level scores, enabling weakly-supervised localization without frame annotations. This paradigm was successfully adopted from computational pathology [4, 21], to lung cancer diagnosis [6]. For per-instance spatio-temporal encoding, 3D CNN backbones like ResNet [13] or ConvNeXt [19] remain a high-performing choice in medical imaging [25]. Recent architectures leverage tensor factorization for high-dimensional data like MRI volumes [31,34]. For instance, FlowVN [31] replaces full 4D kernels with orthogonal 3D convolutions across dimension subsets to capture directional spatio-temporal correlations within its regularization term while reducing complexity. While FlowVN uses this factorization for rapid 4D flow reconstruction, we repurpose it as a feedforward feature extractor. Instead of using all four 3D combinations, STORK only uses the three combinations including the temporal components as one of the dimensions, since the purely spatial (xyz) plane contributes no motion information relevant to contractile activity. Similar factorization techniques have also been adopted outside medical imaging to represent dynamic 3D scenes by accelerating radiance fields [5, 9].

## 3 Uterine Contractile Activity Detection (UCAD)

Uterine contraction is essentially muscular activity of the myometrium manifesting as coordinated tissue deformation over space and time with the efects varying depending on the tissue and proximity of the placenta. Detection is complicated by other moving structures (maternal respiration, bowel peristalsis, fetal motion) producing local change of similar magnitude, and by placental-related pathologies manifesting with similar structural changes. We therefore approach detection through observable induced changes rather than by explicitly defining contractions.

Let a dynamic MRI acquisition be a temporal sequence of 3D volumes

$$
\begin{array} { r } { \hat { \mathbf { X } } = \big ( \hat { \mathbf { x } } _ { 1 } , \hat { \mathbf { x } } _ { 2 } , \ldots , \hat { \mathbf { x } } _ { T } \big ) , \quad \hat { \mathbf { x } } _ { t } \in \mathbb { C } ^ { E \times \hat { H } \times \hat { W } \times \hat { D } } , } \end{array}\tag{1}
$$

where $T$ is the number of acquired time frames, $\hat { H } \times \hat { W } \times \hat { D }$ is the spatial extent, and E the number of acquired echo times with each image consisting of magnitude and phase. We work in image space; raw k-space is outside the scope. Let $\mathbf { r } \in \varOmega \subset \mathbb { R } ^ { 3 }$ denote spatial position, and let $\delta _ { t } ( \mathbf { r } ) \in \mathbb { R } _ { \geq 0 }$ denote a local measure of change at location r between time t − 1 and t, derived from $\mathbf { x } _ { t - 1 }$ and $\mathbf { x } _ { t }$ Conceptually, the observed change superimposes contractile motion $\delta _ { t , \mathrm { c o n } }$ and non-contractile motion $\delta _ { t , \mathrm { n o i s e } } .$ , such that $\delta _ { t } ( \mathbf { r } ) = \delta _ { t , \mathrm { c o n } } ( \mathbf { r } ) \oplus \delta _ { t , \mathrm { n o i s e } } ( \mathbf { r } )$ , where ⊕ denotes superposition of motion sources rather than a separable additive decomposition. The two components are not individually observable, and isolating $\delta _ { t , \mathrm { c o n } }$ is itself part of the learning problem. A contraction corresponds to the existence of at least one sub-region $\mathcal { R } _ { i , t } \subset \Omega$ in which the contractile component is sustained above a threshold τ subject to spatial and temporal coherence conditions:

$$
\delta _ { t , \mathrm { c o n } } ( \mathbf { r } ) > \tau \quad \forall \mathbf { r } \in \mathcal { R } _ { i , t } .\tag{2}
$$

Spatially, $\mathcal { R } _ { i , t }$ must be a connected region of volume at least $a _ { \mathrm { m i n } }$ , ruling out isolated voxels and scattered responses caused by noise. Temporally, the activity must persist across a minimum duration, holding over all frames in a window $[ t , t + \Delta ]$ with $\varDelta \geq \varDelta _ { \mathrm { m i n } }$ . Equation (2) is a schema rather than an operational definition as no closed form is known for $\delta _ { t , \mathrm { c o n } } , \tau , a _ { \mathrm { m i n } }$ , or $\varDelta _ { \mathrm { m i n } }$ These quantities, together with the rejection of non-contractile movement, are what the model must learn. The learning objective is a mapping $f$ from the sequence of MRI volumes $\hat { \bf X }$ to a series-level contraction probability, indicating whether a contraction occurs at any time and anywhere within the series:

$$
\begin{array} { r l r } { f : \mathbb { C } ^ { T \times E \times \hat { H } \times \hat { W } \times \hat { D } }  [ 0 , 1 ] , } & { { } } & { \hat { y } = f ( \hat { \mathbf { X } } ) \approx p ( y = 1 \mid \hat { \mathbf { X } } ) , } \end{array}\tag{3}
$$

where $y \in \{ 0 , 1 \}$ is the series-level label. Beyond this series-level decision, the quantity of ultimate interest is where and when a contraction occurs (the subregions $\mathcal { R } _ { i , t }$ of eq. (2)), which we treat as a secondary objective, to be recovered from the series-level model rather than dedicated supervision.

## 4 Methodology

We instantiate the mapping $f$ of $\mathrm { e q . }$ . (3) as a three-stage pipeline, consisting of a preprocessing step, a feature extractor and a backbone (see fig. 2).

## 4.1 Preprocessing

Each series is reduced from the acquisition $\hat { \mathbf X }$ to the model input X through cropping, per-voxel displacement, normalization, and resampling, resulting in

$$
{ \bf X } = ( { \bf x } _ { 1 } , { \bf x } _ { 2 } , \ldots , { \bf x } _ { T } ) , \quad { \bf x } _ { t } \in \mathbb { R } ^ { C \times H \times W \times D } ,\tag{4}
$$

where $H \times W \times D$ is the fixed spatial target resolution and $C = 4$ the channel dimension. A dilated binary segmentation mask of the placenta [11] defines the cropping and normalization regions but is not applied to the network input. Frames are cropped to the axis-aligned bounding box of the placenta mask, retaining broader uterine context. For each pair of consecutive frames, we estimate a per-voxel displacement ${ \mathbf { u } } _ { t } ( { \mathbf { r } } ) \in \mathbb { R } ^ { 3 }$ by Demons registration [29] and clip it to a physiologically plausible range. This provides an explicit representation of inter-frame tissue motion. Stacking the single-echo intensity with the displacement components yields the input $\mathbf { x } _ { t } = [ \mathrm { i n t e n s i t y } , \mathbf { u } _ { t } ( \mathbf { r } ) ]$ of eq. (4). The single-echo choice is justified in Section 5.5. Channels are normalized per series to preserve temporal intensity changes that signal a contraction. Using the dilated placenta mask as the region interest, the intensity channel is normalized using a single mean and standard deviation computed across all T frames. Displacement components are standardized similarly. Finally, frames are resized via trilinear interpolation to a fixed spatial resolution. Resampling is performed last to ensure displacement field calculation and normalization operate at native resolution, giving a series tensor of shape (T, C, H, W, D).

![](images/cc473b432cbaf92c645ea53e4b20432a4edf269583af193141537c16616ff78f.jpg)  
Fig. 2: STORK network architecture composed of a preprocessing stage (green), a feature extractor (blue) and backbone (pink). Preprocessing stage transforms each series into a 4-channel input combining magnitude images with Demons displacement fields. The feature extractor processes the 4D volume by leveraging 3D convolutions on hyperplanes always involving the temporal dimension. The backbone aggregates information over the time dimension by average pooling combined with a linear projection.

## 4.2 Network Architecture

The network architecture comprises a feature extractor and a backbone, mapping the spatio-temporal input of shape (T, C, H, W, D) to a series-level classification.

Feature Extractor. A contraction is defined jointly over space and time (eq. (2)), so a feature extractor that encodes each frame in isolation cannot represent it. Rather than applying a full 4D convolution, we factorize it into three parallel 3D convolutional branches with kernel size k = 9, each operating on a diferent spatio-temporal hyperplane (T HW, T HD, TW D), similar to approaches in [5, 9, 31]. This preserves the spatio-temporal structure, while avoiding the cost of a full 4D convolution. We remove the spatial hyperplane (HW D), since the spatio-temporal structure captured by the other hyperplanes is more relevant for classification than the volume’s spatial structure alone. Coherent tissue motion over time is encoded directly rather than reconstructed from independent frame features. Each branch outputs d feature maps which are concatenated and fused back to d channels via a depth-wise convolution, mixing the space-time evidence at each location. Finally, global 3D average pooling over the spatial dimensions reduces the tensor to a temporal sequence of vectors, and a linear layer projects each frame to an embedding $\mathbf { z } _ { t } \in \mathbb { R } ^ { d _ { \mathrm { e m b } } }$ that encodes its contextual neighbors.

Backbone. To aggregate the sequence $\mathbf { z } = ( \mathbf { z } _ { 1 } , \dots , \mathbf { z } _ { T } )$ under weak series-level labels, we use a parameter-free temporal mean and a linear layer:

$$
\hat { y } = \sigma \left( \mathbf { w } ^ { \top } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { z } _ { t } + b \right) ,\tag{5}
$$

where $\sigma$ is the sigmoid function. Because the mean and the linear mapping commute, the series logit decomposes exactly into independent per-frame contributions $( \mathbf { w } ^ { \top } \mathbf { z } _ { t } + b )$ . This linearity enables frame-level localization post-hoc without explicit frame annotations. The model is trained end-to-end minimizing the binary cross-entropy between $\hat { y }$ and the series label $y .$ To counter class imbalance, positive instances are weighted by the ratio of negative-to-positive series in the training set.

## 5 Experiments

We evaluate performance using Area Under the Receiver Operating Characteristic (AUROC) and Area Under the Precision-Recall curve (AUPRC), with the latter providing a more stringent evaluation under the class imbalance present in our dataset (see Section 5.1). Series-level metrics assess if a series is correctly classified as containing contractile activity via the aggregated series logit. Framelevel metrics instead evaluate standalone frame predictions against frame-level label, assessing localized contractile activity within a series, serving as a secondary, finer-grained view of model behavior. All results report the mean and standard deviation of ten random runs.

## 5.1 Dataset

Dynamic MRI data was acquired in pregnant participants as part of ethically approved studies (21/LO/0742, 22/YH/0210, 23/LO/0685, 21/SS/0082, 19/LO/ 0852) on clinical MRI scanners (0.55T, 1.5T, 3T). A multi-echo gradient echo planar imaging sequence with imaging parameters varying by field strength was employed [3]. A total of 757 data sets were available, acquired in 549 pregnant participants (mean gestational age=32.2 weeks). Placental masks were obtained with a previously trained network [11], $\mathrm { T 2 ^ { * } }$ fitting was performed via monoexponential least squares fit. Only cases defined contractile by both proposed thresholding mean T2\* [3] and by gamma-fitting the placental mean $\mathrm { T 2 ^ { * } }$ value were maintained (57 cases dropped). Only around $4 0 \%$ of the cases got labeled as showing contractions, leading to an imbalanced dataset. Note that this labeling detects contractile activity through placental $\mathrm { T 2 ^ { * } }$ signal only, whereas our model operates on the image crop including surrounding anatomy (see Section 4.1). Frame- and series-level predictions may therefore respond to contractile activity outside the placenta, not captured by the weak labels. Data is partitioned at the subject level to prevent leakage. Splits are stratified by (cohort, label) to balance the positive rate per cohort. Subjects with manual frame-level labels are reserved for the test set. The remaining split is 70%, 10%, 20% (train, val, test).

## 5.2 Implementation details

Frames are resized by trilinear interpolation to $H \times W \times D = 6 4 \times 6 4 \times 3 2$ Segmentation masks are dilated by 10 mm before cropping, and the crop bounding box is padded by 15 voxels to retain surrounding anatomical context. The intensity channel uses echo 1 of the multi-echo acquisition, following the single echo approach ablated in Section 5.5. Frame-to-frame displacement fields are computed with SimpleITK Demons registration (20 iterations, standard deviation 1.0), clipped to ±20 mm, and z-scored independently per channel using mask-region statistics; intensity is z-scored analogously. No data augmentation is applied. Parameters are optimized with AdamW [20], learning rate $5 \times 1 0 ^ { - 4 }$ weight decay $1 0 ^ { - 2 }$ , under a schedule that warms up linearly for 3 epochs before decaying by cosine annealing. Gradients are clipped to a norm of 10 for stability. Models are trained with batch size 1 for up to 40 epochs, selecting the checkpoint with the best series-level validation AUROC and stopping early after 10 epochs without improvement. Each configuration is trained with 10 random seeds, which jointly control the subject-level data split and model initialization; we report mean ± std across seeds. All reported experiments were conducted on a workstation with a Nvidia RTX 4090 with 24 GB VRAM and a Intel(R) Core(TM) i9-11900K.

## 5.3 Main Results

We introduce a naive baseline from components commonly used for image [12, 17, 33] and time-series [8, 14] processing. A ResNet-10 [13] extracts spatial features from each volume independently, and a Temporal Convolutional Network (TCN) [18] aggregates the resulting embeddings across time. We further include a second baseline replacing the ResNet backbone with ConvNeXt [19,32], incorporating design choices from Vision Transformers to achieve comparable performance while maintaining a convolutional nature. Such architectures had recently strong performance in medical imaging tasks [25]. As a third point of comparison, we replace the mean-pooling aggregation of STORK with attentionbased MIL pooling [16], the standard learned aggregation in weakly-supervised settings. Since attention pooling is permutation-invariant and therefore blind to acquisition order, we add a fixed sinusoidal positional encoding [30] to the frame embeddings before the attention module, making temporal position available to the attention scores.

Comparing STORK to the baselines in Table 1 shows that the ResNet baseline achieves a performance just above random, while the ConvNeXt baseline performs slightly better. As both TCN-based baselines lack a per-frame head, we cannot provide frame-wise metrics. The variant of STORK using attention retains a per-frame linear head and is therefore evaluated at both levels. It recovers most of the performance but remains below STORK at the series level and exhibits higher seed variance at the frame level. Learned attention thus offers no advantage over the parameter-free temporal mean here, while being less stable under weak supervision. Our model achieves strong series-level performance, with an AUROC of 95.0% and AUPRC of 94.6%. Notably, STORK also performs well on frame-level metrics despite receiving no frame-level supervision during training. Furthermore, STORK’s architecture allows for the processing of 70 temporal slices per second with a a peak VRAM usage of 3.4 GB making our approach suitable for near real-time processing at moderate hardware requirements. Overall, STORK clearly outperforms the TCN-based baselines by a wide margin and exceeds the attention variant at the series level, while, unlike the baselines, providing frame-level localization despite training only on series-level labels.

Table 1: Test-set results over seeds. Best values in bold.
<table><tr><td rowspan="2">Approach</td><td colspan="4">Metrics (mean ± std)</td></tr><tr><td></td><td></td><td>|Series AUROC ↑ Series AUPRC ↑ Frame AUROC ↑ Frame AUPRC ↑</td><td></td></tr><tr><td>STORK (ours)</td><td> $\mathbf { 0 . 9 5 0 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 9 4 6 \pm 0 . 0 1 6 }$ </td><td> ${ \bf 0 . 7 9 7 \pm 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 6 7 4 \pm 0 . 0 2 6 }$ </td></tr><tr><td>STORK + Attention</td><td> $0 . 9 1 5 \pm 0 . 0 2 2$ </td><td> $0 . 8 9 9 \pm 0 . 0 3 6$ </td><td> $0 . 7 6 3 \pm 0 . 0 5 7$ </td><td> $0 . 6 5 7 \pm 0 . 0 9 1$ </td></tr><tr><td>ResNet Baseline</td><td> $0 . 5 5 6 \pm 0 . 0 9 6$ </td><td> $0 . 5 2 0 \pm 0 . 0 9 6$ </td><td> $\mathrm { n / a }$ </td><td> $\mathrm { n / a }$ </td></tr><tr><td>ConvNeXt Baseline</td><td> $0 . 6 1 0 \pm 0 . 0 6 4$ </td><td> $0 . 5 5 8 \pm 0 . 0 7 3$ </td><td> $\mathrm { n / a }$ </td><td> $\mathrm { n / a }$ </td></tr></table>

## 5.4 Latent Space Interpretability

The feature extractor maps high-dimensional MRI volumes into a low-dimensional latent space. Figure 3 visualizes these representations for the test set using t-SNE [22] to project sequence embeddings into two dimensions, comparing the ResNet baseline (fig. 3a) against our STORK architecture (fig. 3b). While the ResNet baseline fails to separate contraction (red) and non-contraction (blue) sequences, STORK achieves a distinct separation. This highlights the strength of our model, learning a meaningful representation of the data, while the baseline struggles to learn distinguishing features.

## 5.5 Ablations

We ablate the choice of echo to isolate its contribution, and use Grad-CAM to inspect where the model’s predictions originate. Table 2 evaluates the influence of the acquired echoes. Echo 1 (indexing from 0) attains the best value on two metrics. With marginal diferences between single and multi-echo, we adopt Echo 1 for STORK to reduce computational cost. Frame-level scores are obtained post-hoc from the per-frame decomposition of the series logit (eq. (5)) and evaluated against the manually annotated test frames. Frame-wise analysis with Grad-CAM [26] showed (fig. 4) that the model often attends to areas outside of the placenta, here the uterine wall. This is consistent with our hypothesis that other parts of the uterus carry relevant signal. These attributions ofer temporally resolved localization that may support broader clinical application, but

![](images/a181698ca64d327dc3673c793e6abba8a1353714c388137b5d8ef5b4b59d3e13.jpg)

Fig. 3: t-SNE plot of the latent spaces for ResNet baseline (a) and STORK(b), showing separation of non-contractile (blue) and contractile (red) cases.  
Table 2: Echo ablation comparing single and the best combined echoes at the series and frame level. Best values in bold. Echoes are indexed from 0.
<table><tr><td rowspan="2"></td><td colspan="4">Metrics  $( \mathrm { m e a n } \pm \mathrm { s t d } )$ </td></tr><tr><td>Preprocessing|Series AUROC ↑ Series AUPRC ↑ Frame AUROC ↑ Frame AUPRC ↑</td><td></td><td></td><td></td></tr><tr><td>Echo 0</td><td> $0 . 9 3 7 \pm 0 . 0 1 4$ </td><td> $0 . 9 3 4 \pm 0 . 0 1 7$ </td><td> $0 . 7 8 4 \pm 0 . 0 2 5$ </td><td> $0 . 6 5 4 \pm 0 . 0 4 1$ </td></tr><tr><td>Echo 1</td><td> $\mathbf { 0 . 9 5 0 \pm 0 . 0 1 1 }$ </td><td> $0 . 9 4 6 \pm 0 . 0 1 6$ </td><td> $0 . 7 9 7 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 6 7 4 \pm 0 . 0 2 6 }$ </td></tr><tr><td>Echo 2</td><td> $\mathbf { 0 . 9 5 0 \pm 0 . 0 1 1 }$ </td><td> $0 . 9 4 3 \pm 0 . 0 2 0$ </td><td> $0 . 7 9 5 \pm 0 . 0 1 6$ </td><td> $0 . 6 5 8 \pm 0 . 0 2 4$ </td></tr><tr><td>Echo 0, 1, 2</td><td> $0 . 9 4 8 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 9 4 8 \pm 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 7 9 9 \pm 0 . 0 1 9 }$ </td><td> $0 . 6 6 4 \pm 0 . 0 4 3$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

remain qualitative and cannot be validated against the placenta-derived labels.   
A quantitative attribution analysis is left to future work.

## 6 Conclusion and Discussion

STORK introduces an end-to-end, learning-based method for detecting uterine contractile activity in dynamic MRI. Unlike prior work constrained by handcrafted pipelines on mean placental $\mathrm { T 2 ^ { * } }$ curves [3], our approach learns discriminative features directly from the image, extending across the uterus rather than the placenta alone. Leveraging a large dataset of over 700 dynamic series, STORK attains strong series-level performance under weak supervision, and recovers frame-level localization without frame-level training. A central limitation is the training label quality, which relies on automated $\mathrm { T 2 ^ { * } }$ curve proxies. The introduced label noise may result in the model learning signal from regions the proxy labels never verify. Despite this, automated, non-invasive contraction monitoring provides a richer phenotype of uterine behavior. Clinically, it ofers insights into complications like fetal growth restriction and may help predict how a fetus tolerates the physiological stress of labor. Methodologically, our framework for recovering frame-level localization from weak labels extends naturally to other transient biomedical events. Finally, integrating this model for real-time detection during acquisition could allow scanning protocols to dynamically adapt temporal resolution or extend acquisitions during live events.

![](images/865259ed1ad9d2921a75e5f17d32158fda6267a33700435c0d737610efa2a054.jpg)  
Fig. 4: Frame-wise Grad-CAM. Top: reference intensity with placenta contour (cyan); bottom: same frames with the Grad-CAM overlay for the series logit.

This work was supported by the Bavarian State Ministry of Health, Care and Prevention [project EndoKI], DFG Heisenberg [502024488], ERC StG EARTHWORM [101165242], ERC Proof-of-concept grant SYNCWORM [101293293] and CAIMed - Lower Saxony Center for Artificial Intelligence and Causal Methods in Medicine [ZN4257] funding. The authors have no competing interests to declare.

## References

1. Abaci Turk, E., Stout, J.N., Feldman, H.A., Gagoski, B., Zhou, C., Tamen, R., Manhard, M.K., Adalsteinsson, E., Roberts, D.J., Golland, P., Grant, P.E., Barth, W.H.: Change in T2\* measurements of placenta and fetal organs during Braxton Hicks contractions. Placenta 128, 69–71 (2022)

2. Aviles Verdera, J., Neves Silva, S., Payette, K.M., Tomi-Tricot, R., Hall, M., Story, L., Malik, S.J., Hajnal, J.V., Rutherford, M.A., Hutter, J.: Real-time fetal brain and placental t2\* mapping at 0.55t mri. Magnetic Resonance in Medicine 94(2), 615–624 (2025)

3. Aviles Verdera, J., Payette, K., Hall, M., Bradshaw, C., Story, L., Bansal, S., Waheed, H., Tomi-Tricot, R., Hajnal, J.V., Rutherford, M., Hutter, J.: Comprehensive assessment of uterine contractility using a large database of dynamic t2\* studies. Placenta 169, 39–48 (2025)

4. Campanella, G., Hanna, M.G., Geneslaw, L., Miraflor, A., Werneck Krauss Silva, V., Busam, K.J., Brogi, E., Reuter, V.E., Klimstra, D.S., Fuchs, T.J.: Clinicalgrade computational pathology using weakly supervised deep learning on whole slide images. Nature Medicine 25(8), 1301–1309 (2019)

5. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. In: CVPR 2023. pp. 130–141 (June 2023)

6. Chen, J., Zeng, H., Zhang, C., Shi, Z., Dekker, A., Wee, L., Bermejo, I.: Lung cancer diagnosis using deep attention-based multiple instance learning and radiomics. Medical Physics 49(5), 3134–3143 (2022)

7. Dellschaft, N.S., Hutchinson, G., Shah, S., Jones, N.W., Bradley, C., Leach, L., Platt, C., Bowtell, R., Gowland, P.A.: The haemodynamics of the human placenta in utero. PLOS Biology 18(5) (May 2020)

8. Farha, Y.A., Gall, J.: Ms-tcn: Multi-stage temporal convolutional network for action segmentation. In: CVPR 2019. pp. 3575–3584 (2019)

9. Fridovich-Keil, S., Meanti, G., Warburg, F.R., Recht, B., Kanazawa, A.: K-planes: Explicit radiance fields in space, time, and appearance. In: CVPR 2023. pp. 12479– 12488. IEEE (2023)

10. Hai, Y.V.T., Phuong, D.L.C., Duy, Q.V.: Applications of artificial intelligence in fetal mri: a systematic review. Egyptian Journal of Radiology and Nuclear Medicine 56, 224 (2025)

11. Hall, M., Verdera Aviles, J., Cromb, D., Silva Neves, S., Rutherford, M., Counsell, S.J., Hajnal, J.V., Story, L., Hutter, J.: Placental t2\* as a measure of placental function across field strength from 0.55t to 3T. Scientific Reports 14(1), 28594 (Nov 2024)

12. Hasanah, S.A., Pravitasari, A.A., Abdullah, A.S., Yulita, I.N., Asnawi, M.H.: A deep learning review of resnet architecture for lung disease identification in cxr image. Applied Sciences 13(24), 13111 (2023)

13. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR 2016. pp. 770–778 (2016)

14. He, Y., Zhao, J.: Temporal convolutional networks for anomaly detection in time series. In: Journal of Physics: Conference Series. vol. 1213. IOP Publishing (2019)

15. Hutter, J., Kohli, V., Dellschaft, N., Uus, A., Story, L., Steinweg, J.K., Gowland, P., Hajnal, J.V., Rutherford, M.A.: Dynamics of t2\* and deformation in the placenta and myometrium during pre-labour contractions. Scientific Reports 12(1), 18542 (Nov 2022)

16. Ilse, M., Tomczak, J.M., Welling, M.: Attention-based deep multiple instance learning. In: Dy, J.G., Krause, A. (eds.) ICML 2018. Proceedings of Machine Learning Research, vol. 80, pp. 2127–2136 (2018)

17. Khan, A., Sohail, A., Zahoora, U., Qureshi, A.S.: A survey of the recent architectures of deep convolutional neural networks. Artificial Intelligence Review 53(8), 5455–5516 (2020)

18. Lea, C., Vidal, R., Reiter, A., Hager, G.D.: Temporal convolutional networks: A unified approach to action segmentation. In: ECCV 2016. pp. 47–54. Springer (2016)

19. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: CVPR 2022. pp. 11976–11986 (2022)

20. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR 2019. OpenReview.net (2019)

21. Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F.: Data-eficient and weakly supervised computational pathology on whole-slide images. Nature Biomedical Engineering 5(6), 555–570 (2021)

22. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9(11), 2579–2605 (2008)

23. Maron, O., Lozano-Pérez, T.: A framework for multiple-instance learning. In: Jordan, M.I., Kearns, M.J., Solla, S.A. (eds.) NeurIPS 1997. pp. 570–576 (1997)

24. Meshaka, R., Gaunt, T., Shelmerdine, S.C.: Artificial intelligence applied to fetal mri: A scoping review of current research. The British Journal of Radiology 96(1147), 20211205 (2023)

25. Roy, S., Kirchhof, Y., Ulrich, C., Rokuss, M., Wald, T., Isensee, F., Maier-Hein, K.: Mednext-v2: Scaling 3d convnexts for large-scale supervised representation learning in medical image segmentation. arXiv:2512.17774 (2025)

26. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: ICCV 2017. pp. 618–626 (2017)

27. Sinding, M., Peters, D.A., Frøkjær, J.B., Christiansen, O.B., Uldbjerg, N., Sørensen, A.: Reduced placental oxygenation during subclinical uterine contractions as assessed by BOLD MRI. Placenta 39, 16–20 (Mar 2016)

28. Slator, P.J., Hutter, J., Palombo, M., Jackson, L.H., Ho, A., Panagiotaki, E., Chappell, L.C., Rutherford, M.A., Hajnal, J.V., Alexander, D.C.: Combined difusionrelaxometry MRI to identify dysfunction in the human placenta. Magnetic Resonance in Medicine 82(1), 95–106 (Jul 2019)

29. Thirion, J.: Image matching as a difusion process: an analogy with maxwell’s demons. Medical Image Analysis 2(3), 243–260 (1998)

30. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Guyon, I., von Luxburg, U., Bengio, S., Wallach, H.M., Fergus, R., Vishwanathan, S.V.N., Garnett, R. (eds.) NeurIPS 2017. pp. 5998–6008 (2017)

31. Vishnevskiy, V., Walheim, J., Kozerke, S.: Deep variational network for rapid 4d flow mri reconstruction. Nature Machine Intelligence 2(4), 228–235 (2020)

32. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S.: Convnext v2: Co-designing and scaling convnets with masked autoencoders. In: CVPR 2023. pp. 16133–16142 (2023)

33. Xu, W., Fu, Y.L., Zhu, D.: Resnet and its application to medical image processing: Research progress and challenges. Computer Methods and Programs in Biomedicine 240, 107660 (2023)

34. Zhou, Q., Ye, Y., Cai, Z.: Spatiotemporal-aware neural fields for dynamic ct reconstruction. In: AAAI Conference on Artificial Intelligence. vol. 39 (2025)