# ReTailor: A Residual Feature Tailoring Framework for Real-Time Cross-Sensor Pansharpening Adaptation

Jing-Ze Liang<sup>a,1</sup>, Wei-Jie Xiao<sup>a,1</sup>, Zi-Han Cao<sup>a,1</sup>, Yang-Dong Li<sup>a</sup>, Tian-Yi Hu<sup>a</sup> and Liang-Jian Deng<sup>a,b,∗</sup>

<sup>a</sup>University of Electronic Science and Technology of China, Chengdu, 611731, Sichuan, China

<sup>b</sup>Multi-Hazard Early Warning Key Laboratory of Sichuan Province, Chengdu, 611731, Sichuan, China

## A R T I C L E I N F O

Keywords:   
Cross-sensor pansharpening   
Instantaneous adaptation   
Continual test-time adaptation   
Adaptive selection

## A BS T R AC T

Recent pansharpening models achieve strong fusion performance on trained sensors, yet often generalize poorly to unseen sensors. Existing cross-sensor adaptation is limited by target-sensor retraining, which requires target-sensor training pairs and a dedicated training stage, and by per-image zero-shot adaptation, which is computationally expensive due to whole-network optimization for each test image. To address these problems, we propose ReTailor, a real-time sequential cross-sensor adaptation framework. Only a lightweight residual Feature Tailor is rapidly optimized under self-supervised, physics-aware constraints to correct cross-sensor shifts without modifying the well-learned fusion prior. For eficient and efective deployment, ReTailor combines content-aware patch selection, which identifies a few informative patches per image for optimization at negligible cost to improve both quality and eficiency, with rolling residual bootstrap, which inherits sensor-related knowledge across the image stream to further improve quality. Extensive experiments on cross-sensor adaptation settings show that ReTailor mitigates cross-sensor degradation both efectively and eficiently. Moreover, we compare ReTailor with Low-Rank Adaptation (LoRA), a widely used method for fine-tuning, to demonstrate the advantages of our approach. We further evaluate the core principle of ReTailor on two additional remote sensing tasks: super-resolution pansharpening and image segmentation, where it consistently improves performance. In general, ReTailor provides a highly efective and eficient strategy for deploying pansharpening models under real-world cross-sensor degradation, while also ofering useful insights into managing cross-sensor degradation in other remote sensing tasks. Code will be released at https://github.com/JingZe-Liang/ReTailor.

## 1. Introduction

Pansharpening reconstructs a high-resolution multispectral (HRMS) image from the fine spatial details of a highresolution panchromatic (PAN) image and the spectral information of a low-resolution multispectral (LRMS) image. Due to its importance in remote-sensing interpretation and downstream analysis, pansharpening has been well studied [1, 2, 3]. Traditional methods mainly include component substitution, multiresolution analysis, and variational optimization [4]. Component-substitution and multiresolution-analysis methods transfer PAN details through explicit injection rules [5, 6, 7, 8], whereas variational methods formulate fusion as a regularized reconstruction problem [9, 10, 11, 12]. Deeplearning methods instead learn nonlinear mappings from PAN and LRMS observations to HRMS images. Representative architectures include convolutional neural networks [13, 14, 15, 16], Transformers [17], difusion models [18], and statespace models [19]. Despite this progress, the performance of existing pansharpening methods often degrades heavily in cross-sensor applications. Sensor-dependent discrepancies in

![](images/99c351d3d509c93cf85ad352f213402391c0656a235969eb5891629e4b191e4d.jpg)  
Figure 1: Comparison of cross-sensor adaptation paradigms. Existing methods (top) trade fusion quality against eficiency and ground-truth (GT) requirements, whereas ReTailor (bottom) achieves high-quality, real-time adaptation without GT while preserving the pretrained fusion prior.

spatial and spectral responses, noise patterns, and radiometric statistics can cause the learned fusion priors to break down when transferred across sensors [14]. Existing solutions mainly follow two routes: target-sensor retraining and zeroshot adaptation. Retraining recalibrates a model to the target sensor distribution, but it requires target-sensor training pairs, either ground truth (GT) data or reduced-resolution pairs simulated from target observations, together with lengthy optimization. GT data are also dificult to acquire [14]. Zeroshot methods avoid additional training by optimizing the model at inference time. However, most approaches still require tens of seconds to several minutes of processing for each input pair [20, 22]. Traditional methods avoid both retraining and test-time optimization, but their limited nonlinear representation capacity generally results in lower fusion quality [1, 23]. A systematic comparison of these solution paradigms with our method is presented in Fig. 1. In summary, an efective adaptation method should befast, require little or no GT data, and maintain highfusion quality.

![](images/1d84d4f5f82f6cb865fa9dafa160ad583192f67f8d62b6e4e205702fe2bfe244.jpg)  
Figure 2: Empirical overview of ReTailor. (a) Pairwise cosine similarity between residual embeddings of independently adapted FusionNet Feature Tailors on 20 target images per transfer (mean ± std over 190 pairs). (b) Fusion quality (HQNR, $1 - D _ { \lambda } ,$ $1 - D _ { s } ;$ higher is better) on QB→GF2 (upper axes) and WV3→WV2 (lower axes); the ReTailor-family curve takes the best mean of FT-only and ReTailor over both backbones (the HQNR and $1 - D _ { \lambda }$ values are those of ReTailo $\mathrm { F u s i o n N e t }$ alone, the best method in Table 1). (c) End-to-end per-image duration of ZS-Pan [20], LoRA [21], and ReTailor (FusionNet); smaller is faster.

To meet these requirements, we propose ReTailor, a crosssensor adaptation framework as illustrated in Fig. 3. ReTailor efectively mitigates cross-sensor pansharpening degradation, outperforming existing baselines in fusion quality (Fig. 2(b)) while reducing per-image adaptation latency to a fraction of a second (Fig. 2(c)). Given a stream of target PAN/LRMS pairs, each pair is first partitioned into spatially aligned patches. Adaptation of the seed image (first image) uses all patches, whereas content-aware patch selection determines the adaptation subset for each subsequent image with negligible computational overhead. The corresponding patch set is then used to adapt the pretrained fusion model. Between the frozen Feature Extractor and Channel Mapper, both defined in Sec. 3.2, we insert a lightweight residual Feature Tailor. Using the selected informative patch set, the Feature Tailor is updated to correct the cross-sensor degradation under self-supervised, physics-aware constraints, while the Feature Extractor and Channel Mapper remain frozen. After adaptation, all candidate patches are encoded by the frozen Feature Extractor, corrected by the updated Feature Tailor, and mapped by the frozen Channel Mapper into HRMS patches, which are then stitched into the full-resolution output.

To identify fewer yet more informative patches as quickly as possible, while retaining suficient information for rapid and generalizable adaptation to cross-sensor degradation rather than biased overfitting, ReTailor leverages two complementary and mutually constraining cues. Within-patch PAN variation provides spatial-detail cues, whereas inter-band LRMS variation provides complementary spectral cues; both are available directly from the PAN/LRMS pair [24]. As target images are processed sequentially, rolling residual bootstrap propagates the learned Tailor correction across images rather than restarting adaptation from the default initialization for each image. Supported by the ablation and the analysis in Sec. 5.2.1, this cross-image knowledge propagation preserves previously learned sensor-related corrections and improves per-image adaptation quality. We further verify that it remains stable over long image sequences (also in Sec. 5.2.4). Fig. 2(a) provides the visual evidence for this design.

Following this workflow, these components work together with distinct roles to enable eficient and efective crosssensor adaptation. Content-aware patch selection prioritizes patches that are most informative for correcting crosssensor degradation while avoiding uninformative samples that may lead to biased adaptation, with negligible additional computational overhead. The residual Feature Tailor then extracts corrective information from the selected patches into lightweight residual adjustments, enabling rapid crosssensor correction. Rolling residual bootstrap further carries forward efective corrections learned from preceding images, exploiting their shared cross-sensor degradation characteristics to improve subsequent adaptation. Together, ReTailor delivers strong cross-sensor fusion performance, achieving up to 0.9245 HQNR while keeping end-to-end latency below 0.57 s (selection + adaptation + pansharpening) per image across all evaluated settings.

We evaluate ReTailor against representative traditional and learning-based pansharpening methods on QB→GF2 and WV3→WV2. ReTailo $\mathrm { F u s i o n N e t }$ achieves the highest HQNR and the lowest spectral distortion $D _ { \lambda }$ on both transfers, reaching HQNR values of 0.8976 and 0.9245, respectively. These gains are achieved with only about six selected patches and 0.12–0.20 s of adaptation per image, demonstrating that high-quality cross-sensor correction can be obtained with a very limited adaptation budget. This eficiency is further confirmed by fair comparisons with LoRA, where ReTailor reduces mean end-to-end latency by almost 50% while consistently lowering $D _ { \lambda }$ across all evaluated settings. Detailed ablation studies validate the contribution of each proposed component. Beyond standard pansharpening, we further examine the ReTailor principle on two additional cross-sensor remote-sensing tasks, namely arbitrary-scale super-resolution pansharpening and semantic segmentation, where it consistently remains efective.

![](images/0006b6f4b547c9e0034a79243b253c8a7f7417d3fadbd3e6547bb23ac6b2410b.jpg)  
Figure 3: Deployment pipeline of ReTailor. Content-aware selection picks informative PAN/LRMS patches. The frozen Feature Extractor produces latent features �, the trainable Feature Tailor adds a residual correction to give �<sup>∗</sup>, and the frozen Channel Mapper outputs the source and tailored predictions. Together with the observations and target-sensor modulation transfer function (MTF) priors, these form the spectral, spatial, and frozen-source losses that update only the Tailor. The lower panel shows the three-layer Tailor.

The main contributions of this work are summarized as follows:

1. We develop ReTailor, an eficient cross-sensor adaptation framework that preserves the pretrained fusion prior, rapidly corrects cross-sensor degradation, and maintains high fusion quality.

2. We carefully design a fully tensorized and highly parallel content-aware patch selection method, a lightweight residual Feature Tailor, and a rolling residual bootstrap strategy, which play complementary roles and jointly benefit the ReTailor framework.

3. We conduct extensive experiments on cross-sensor pansharpening datasets covering both 4-band and 8- band multispectral imagery, with comparisons against representative traditional and learning-based methods as well as a parameter-matched LoRA baseline. We further extend the core principle of ReTailor to arbitrary-scale super-resolution pansharpening and cross-sensor semantic segmentation, highlighting its broader potential for cross-sensor adaptation in remote sensing.

## 2. Related Work

## 2.1. Learning-Based Pansharpening

Deep learning methods have achieved remarkable performance in pansharpening by learning nonlinear mappings from PAN and LRMS observations to HRMS images [25, 26, 27, 28]. Early approaches were mainly based on convolutional neural networks (CNNs). PNN pioneered direct end-to-end learning for pansharpening from PAN and upsampled LRMS inputs [13], while PanNet incorporated high-pass domain priors to explicitly enhance spatial-detail injection [29]. Subsequent CNN-based methods further improved feature interaction and detail reconstruction, as exemplified by FusionNet [15, 30] and GPPNN [16, 31].

More recent studies have explored architectures beyond conventional CNNs. PanFormer employs Transformer-based modeling to capture long-range spatial–spectral dependencies [17]. PanDif formulates pansharpening as a difusionbased reconstruction process [18], while Pan-Mamba introduces state-space modeling for eficient long-range and cross-modal interaction [19]. Despite these architectural advances, most learning-based methods are trained under sensor-specific data distributions, and their performance often degrades when the spectral responses, spatial characteristics, or imaging statistics of the deployment sensor difer from those encountered during training.

## 2.2. Cross-Sensor Pansharpening Strategies

Applying a pansharpening model trained on one sensor to another remains challenging because sensor-dependent spatial-spectral responses, noise characteristics, and radiometric distributions may violate the learned fusion prior, resulting in substantial cross-sensor degradation [14, 32, 33, 34, 35]. Existing solutions can be broadly grouped into three categories:

(1) Model Retraining. Retraining and fine-tuning methods update all or part of a pretrained pansharpening model using data acquired from the target sensor [36]. Targetadaptive CNN-based pansharpening, for example, constructs reduced-resolution training pairs from target observations and recalibrates the learned fusion mapping according to the target acquisition characteristics [14]. More recent sensortransfer approaches reduce this cost by selecting a compact subset of target-domain samples and updating only parameters that are sensitive to the sensor shift [37]. Although efective, these methods require target-sensor training pairs, either GT samples or reduced-resolution pairs simulated under Wald’s protocol [38], and an additional, often computationally demanding and complex training stage, with the pretrained model being partially or fully updated.

(2) Zero-Shot Methods. Zero-shot methods optimize a pretrained pansharpening model during inference without requiring target GT supervision [20, 22]. In remote sensing, these methods typically construct self-supervised objectives from the observed LRMS–PAN pair and exploit the sensor observation process to adapt the model to the current input. They avoid the need for additional target-sensor training data while maintaining relatively high fusion quality. Parametereficient techniques can also be incorporated into this zeroshot setting. For example, LoRA freezes the pretrained model weights and represents the update using trainable low-rank factors [21], thereby restricting inference-time optimization to a small number of parameters. When optimized separately for each target image under a self-supervised objective, LoRA serves as a parameter-eficient zero-shot adaptation mechanism. Despite their efectiveness, conventional zeroshot methods often require tens of seconds to several minutes to process a single LRMS–PAN input.

(3) Traditional Methods. Traditional methods based on component substitution, multiresolution analysis, or variational optimization require neither target-sensor training nor inference-time network adaptation. Component substitution and multiresolution analysis methods inject PAN details through predetermined transformations, whereas variational methods reconstruct the fused image under explicit observation models and regularization constraints [5, 6, 39, 40]. These methods are generally computationally eficient and can operate directly on a single LRMS–PAN pair. However, their predefined fusion rules and limited nonlinear representation capacity restrict their fusion performance.

In summary, existing strategies face a trade-of among fusion quality, training-data requirements, and computational eficiency. Model retraining can achieve strong target-sensor performance but requires target-sensor training pairs and a dedicated training stage, whereas zero-shot methods remove the need for such training pairs at the cost of substantial per-image inference-time computation. Traditional methods are eficient and training-free but their performance is limited by predefined fusion mechanisms. By contrast, ReTailor addresses this trade-of by providing an eficient, efective, and practical solution to adapt pretrained pansharpening models to unseen sensors.

## 3. Method

This section presents ReTailor. Section 3.1 gives the overall pipeline and notation. Section 3.2 describes the residual Feature Tailor inserted into a frozen backbone, and Section 3.3 defines the physics-aware objective that optimizes it without target HRMS. Section 3.4 introduces content-aware patch selection, which builds a compact and high-value adaptation set (complexity in Appendix D), and Section 3.5 introduces rolling residual bootstrap, which lets each image inherit the sensor-related correction learned by its predecessors.

## 3.1. Framework Overview

As shown in Fig. 3, ReTailor combines three components: a residual Feature Tailor that corrects target-sensor representation shifts, content-aware patch selection that concentrates adaptation on informative observations, and rolling residual bootstrap that passes the previously learned sensor-related correction along the target-image stream.

For the �-th target image, the aligned LRMS and PAN observations are partitioned into $N _ { c }$ spatially corresponding candidate patch pairs. We collect these candidates in the ordered collection

$$
\begin{array} { r } { \mathbf { \mathcal { V } } ^ { t } = \left[ ( \mathbf { Y } _ { i } ^ { t } , \mathbf { P } _ { i } ^ { t } ) \right] _ { i = 1 } ^ { N _ { c } } , } \end{array}
$$

where � indexes the image-processing order and � indexes candidate patches within image �. Adaptation and inference use diferent observations. Let $S ^ { t } \subseteq \{ 1 , \dots , N _ { c } \}$ denote the adaptation index set, $\Omega ^ { t } = \mathcal { V } ^ { t } [ S ^ { t } ]$ the corresponding selected observations, and $M ^ { t } = | S ^ { t } |$ the number of selected patches. The seed image uses all candidates to establish an initial target-sensor correction, whereas each subsequent image uses content-aware patch selection to construct a compact adaptation set. Selection thus shrinks only the adaptation set; inference still covers the full image.

ReTailor decomposes a compatible pretrained pansharpening backbone into a frozen Feature Extractor $F _ { \mathrm { e x t } }$ and a frozen Channel Mapper $F _ { \mathrm { m a p } } .$ , and inserts a residual Feature Tailor $G _ { \phi }$ at their fused-feature interface. Only the Tailor parameters $\phi$ are optimized on $\Omega ^ { t }$ under the physics-aware adaptation objective. For the seed image $( t = 0 )$ , adaptation starts from the zero-output initialization $\phi _ { \mathrm { i n i t } }$ and uses all $N _ { c }$ candidates as a one-time bootstrap. For each later image $t \geq 1$ adaptation starts from the preceding state $\phi ^ { t - 1 }$ and refines it using the selected observations, producing the current Tailor state $\phi ^ { t }$

After adaptation, $\phi ^ { t }$ is fixed and applied to all $N _ { c }$ candidate patches of the current image. The frozen Feature

Extractor first produces their fused representations, the Feature Tailor applies the learned residual correction, and the frozen Channel Mapper reconstructs the corresponding HRMS patches. These predictions are finally restored to their spatial order and stitched into the full-resolution output $\widehat { \mathbf { X } } ^ { t }$

## 3.2. Residual Feature Tailoring

Residual Feature Tailoring preserves the source fusion pathway while adapting only its fused representation. Suppressing image and patch indices, the frozen Feature Extractor produces the fused latent tensor $\mathbf { Z } = F _ { \mathrm { e x t } } ( \mathbf { Y } , \mathbf { P } ; \theta _ { \mathrm { e x t } } )$ , and the frozen Channel Mapper produces the source prediction $\widehat { \mathbf { X } } _ { \mathrm { s r c } } = F _ { \mathrm { m a p } } ( \mathbf { Z } ; \theta _ { \mathrm { m a p } } ) + A ( \mathbf { Y } )$ . Here, $\theta _ { \mathrm { e x t } }$ and $\theta _ { \mathrm { { m a p } } }$ are the frozen parameters of the Feature Extractor and Channel Mapper, respectively, and (�) is the fixed, non-trainable LRMS shortcut added to the Channel Mapper output to align it with the HRMS prediction space; it is � for a backbone without such a shortcut.

The Feature Tailor predicts an additive correction of this fused tensor; its implementation is detailed in the lower panel of Fig. 3:

$$
\begin{array} { r l } & { \mathbf { Z } ^ { * } = \mathbf { Z } + G _ { \phi } ( \mathbf { Z } ) , } \\ & { \widehat { \mathbf { X } } _ { * } = F _ { \mathrm { m a p } } \left( \mathbf { Z } ^ { * } ( \phi ) ; \theta _ { \mathrm { m a p } } \right) + \mathcal { A } ( \mathbf { Y } ) , } \end{array}\tag{1}
$$

where $G _ { \phi }$ is the trainable Feature Tailor, $\mathbf { Z } ^ { \ast }$ is the corrected fused tensor, and $\widehat { \mathbf { X } } _ { \ast }$ is the tailored prediction. Task-specific insertion interfaces and channel widths are summarized in Table 14. The modular decomposition underlying this insertion choice and the corresponding placement experiment are detailed in Appendix B.2.

## 3.3. Physics-Aware Self-Supervised Adaptation Objective

At full-resolution deployment, target HRMS supervision is unavailable. ReTailor therefore adapts the Feature Tailor using only the selected observations $\Omega ^ { t }$ , their frozen-source predictions, and the target-sensor MTF kernels ${ \bf { B } } _ { \mathrm { { M S } } }$ and $\mathbf { B } _ { \mathrm { P } } [ 7 ]$ . With the adaptation index set $S ^ { t }$ defined in Section 3.1, the objective is

$$
\mathcal { L } _ { \mathrm { a d a p t } } ^ { t } \big ( \phi ; S ^ { t } \big ) = \frac { 1 } { | S ^ { t } | } \sum _ { i \in S ^ { t } } \biggl [ \eta _ { \mathrm { s p e } } \mathcal { L } _ { \mathrm { s p e } } ^ { t , i } ( \phi ) + \eta _ { \mathrm { s p a } } \mathcal { L } _ { \mathrm { s p a } } ^ { t , i } ( \phi ) + \eta _ { \mathrm { o r i } } \mathcal { L } _ { \mathrm { o r i } } ^ { t , i } ( \phi ) \biggr ] ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { s p e } } ^ { t , i } , \mathcal { L } _ { \mathrm { s p a } } ^ { t , i }$ , and $\mathcal { L } _ { \mathrm { o r i } } ^ { t , i }$ are the spectral, spatial, and frozensource consistency losses on patch � of image $t , \eta _ { \mathrm { s p e } } , \eta _ { \mathrm { s p a } } ,$ , and $\eta _ { \mathrm { o r i } }$ balance their contributions, and only $\phi$ is updated. The image, patch, and parameter indices are omitted below for readability.

For each selected patch, two observation-consistency terms are combined with a frozen-source regularizer:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s p e } } = \left\| D _ { \mathrm { M S } } ^ { ( b ) } \Big ( \widehat { \mathbf { X } } _ { * } ; \mathbf { B } _ { \mathrm { M S } } \Big ) - { \mathbf { Y } } \right\| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { s p a } } = \left\| \widehat { \mathbf { X } } _ { * } - \Big ( \widehat { \mathbf { X } } _ { * } * \mathbf { B } _ { \mathrm { M S } } \Big ) \odot \Big [ \widehat { \mathbf { P } } \mathcal { O } \left( \widehat { \mathbf { P } } * { \mathbf { B } } _ { \mathrm { P } } \right) \Big ] \right\| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { o r i } } = \left\| \widehat { \mathbf { X } } _ { * } - \widehat { \mathbf { X } } _ { \mathrm { s r c } } \right\| _ { 1 } . } \end{array}
$$

Spectral consistency. The spectral term requires the tailored prediction to reproduce the observed LRMS patch after targetsensor degradation, which follows the consistency property of Wald’s protocol [38] and the spectral-preservation objectives of unsupervised pansharpening [20, 41]. Here, � indexes the compatible frozen backbone, and $D _ { \mathrm { M S } } ^ { ( b ) } ( \cdot ; \mathbf { B } _ { \mathrm { M S } } )$ applies the target-sensor MS MTF blur and backbone-compatible resampling to a predicted HRMS patch.

Spatial consistency. The spatial term constrains the tailored prediction using the relative high-frequency modulation observed in the PAN patch, in the spirit of contrast-based (ratio) detail injection [7, 42] and the spatial-preservation objectives of unsupervised pansharpening [20, 41]. The broadcast PAN patch $\widehat { \mathbf { P } }$ is represented in the HRMS channel space, and $\mathbf { B } _ { \mathrm { P } }$ is the target-sensor PAN MTF kernel.

Frozen-source consistency. The frozen-source term regularizes the tailored prediction toward $\hat { \mathbf { X } } _ { \mathrm { s r c } } ,$ which is independent of $\phi ,$ discouraging the observation-driven update from drifting away from the pretrained fusion prior.

The notation $\lVert \cdot \rVert _ { 1 }$ denotes the mean absolute value, while $* , \odot ;$ , and $\oslash$ denote convolution, element-wise multiplication, and element-wise division, respectively.

Because adaptation runs for only a few gradient steps on $S ^ { t }$ , fusion quality hinges on whether the selected patches provide suficiently informative and complementary spatial– spectral constraints—the question taken up by the contentaware patch selection in Section 3.4.

## 3.4. Content-Aware Patch Selection

Adapting on a subset of patches saves time, but the subset must be chosen well: what matters is not only how many patches are kept but which ones constrain the Tailor update. A fixed random subset ignores patch content and may include patches that carry little information about the cross-sensor shift. Content-aware scores such as the Tailorparameter gradient norm [43] or the loss reduction after a trial update [44] would require at least one extra forward and backward pass per candidate, which defeats the purpose of fast adaptation.

ReTailor instead scores each candidate from PAN/LRMS cues that are available before adaptation and need no forward or gradient computation, so selection takes only a few milliseconds. To keep high-scoring but spatially clustered candidates from dominating the budget, regional discounting lowers the priority of candidates whose coarse region is already represented. The resulting set carries strong adaptation evidence with broad spatial coverage.

As summarized in Fig. 4, for each later image $t \geq 1$ , the selector maps $\mathcal { V } ^ { t }$ to a nonempty set $S ^ { t }$ , with $M ^ { t }$ determined by �. The seed image $( t = 0 )$ instead uses all $N _ { c }$ candidates, as specified in Section 3.5.

## 3.4.1. PAN/LRMS variation measuresfor adaptation patch selection

For each aligned candidate patch in a target image, the selector computes two scalar variation measures that provide complementary adaptation evidence: $g _ { i }$ quantifies withinpatch PAN spatial variation, whereas $s _ { i }$ quantifies inter-band

![](images/b3c87eb0fe81731a73dbec4390c987a28b6f886df19deeff269fe5a666c38e53.jpg)  
Figure 4: Content-aware patch selection. PAN spatial variation and LRMS inter-band variation of the $N _ { c }$ candidates are combined into a PAN-anchored score, discounted by regional occupancy, and greedily admitted until relative stopping (�) yields $M ^ { t }$ patches for adaptation; inference still covers all $N _ { c }$ candidates.

LRMS spectral variation. Let $L = h w$ denote the number of PAN pixels, $L _ { s } = h _ { s } w$ the number of LRMS spatial samples per band, and $C _ { \mathrm { M S } }$ the number of LRMS spectral bands. We represent the PAN patch as $\mathbf { p } _ { i } = \mathrm { v e c } ( \mathbf { P } _ { i } ) \in \mathbb { R } ^ { L }$ and the LRMS patch as $\mathbf { Y } _ { i } \in \mathbb { R } ^ { C _ { \mathrm { M S } } \times \dot { L _ { s } } }$ . To remove a global PAN brightness ofset and the LRMS component shared across spectral bands, ReTailor applies centering projections before computing these measures:

$$
\begin{array} { l l } { { \displaystyle { \bf H } _ { L } = { \bf I } _ { L } - \frac { 1 } { L } { \bf 1 } _ { L } { \bf 1 } _ { L } ^ { \top } } , } & { { \displaystyle g _ { i } = \frac { 1 } { \sqrt { L } } \left\| { \bf H } _ { L } { \bf p } _ { i } \right\| _ { 2 } , } } \\ { { \displaystyle { \bf H } _ { \mathrm { M S } } = { \bf I } _ { C _ { \mathrm { M S } } } - \frac { 1 } { C _ { \mathrm { M S } } } { \bf 1 } _ { C _ { \mathrm { M S } } } { \bf 1 } _ { C _ { \mathrm { M S } } } ^ { \top } } , } & { { \displaystyle s _ { i } = \frac { 1 } { C _ { \mathrm { M S } } L _ { s } } \left\| { \bf H } _ { \mathrm { M S } } { \bf Y } _ { i } \right\| _ { 1 , 1 } , } } \end{array}
$$

where ${ \mathbf { I } } _ { n }$ and ${ \mathbf { 1 } } _ { n }$ are the $n \times n$ identity matrix and length-� all-ones vector, $\| \cdot \| _ { 2 }$ is the Euclidean norm, and $\| \cdot \| _ { 1 , 1 }$ is the sum of absolute matrix entries. The projection $\mathbf { H } _ { L }$ centers PAN values over spatial positions, whereas ${ \bf H } _ { \mathrm { M S } }$ centers each LRMS spatial sample over spectral bands. Consequently, $g _ { i }$ and $s _ { i }$ are invariant to the corresponding additive common modes and can be computed in parallel for all candidates.

## 3.4.2. PAN-anchored evidence composition and coarse-region mapping

Because the PAN and LRMS cues $g _ { i }$ and $s _ { i }$ have diferent numerical ranges, ReTailor independently applies withinimage min–max normalization to each cue. For $x \in \{ g , s \}$ $\widetilde { x } _ { i }$ denotes the normalized value of $x _ { i }$ , with the minimum and maximum computed over $j \in \{ 1 , \dots , N _ { c } \}$

The normalized PAN spatial-variation measure $\widetilde { g } _ { i }$ is used as the score anchor (supported by ablation in Sec. 5.1.2) because it quantifies within-patch variation at the target spatial scale, providing a direct proxy for the high-frequency detail constrained by the PAN observation. This is consistent with pansharpening practice, where detail injection is gated by local statistics of the PAN/MS pair [45] or modulated by local PAN contrast [42], and learning-based methods operate on the high-pass PAN content [29]. The normalized LRMS inter-band spectral-variation measure $\widetilde { s } _ { i }$ captures complementary spectral evidence from the lower-resolution LRMS observation. Spectral fidelity is commonly assessed through inter-band relationships [46], and band-dependent injection models treat each band individually [47], so patches with strong inter-band variation constrain the spectral loss most. We therefore use a one-sided composition in which LRMS evidence supplements the PAN-based score only when $\widetilde { s } _ { i }$ exceeds $\widetilde { g } _ { i }$ :

$$
b _ { i } = \widetilde { g } _ { i } + \lambda _ { c } [ \widetilde { s } _ { i } - \widetilde { g } _ { i } ] _ { + } , \qquad 0 \leq \lambda _ { c } \leq 1 ,\tag{3}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ and $\lambda _ { c }$ controls the LRMS supplement. The positive-part term treats the LRMS cue as a bounded supplement rather than a competing score: it leaves the PAN baseline unchanged when $\widetilde { s } _ { i } \ \le \ \widetilde { g } _ { i }$ and increases the score when the LRMS cue indicates additional spectral distinctiveness.

The base score in Eq. (3) ranks candidates solely by PAN/LRMS evidence and does not account for whether high-scoring candidates are spatially clustered. ReTailor therefore divides the image into $N _ { r }$ fixed coarse regions. Each candidate is assigned to the coarse region in which its patch center lies, and $r ( i ) \in \{ 1 , \ldots , N _ { r } \}$ denotes that region index. During selection, candidates in regions already represented in the selected set receive lower efective scores. The selector configuration is $\Xi _ { \mathrm { s e l } } = ( \lambda _ { c } , \lambda _ { r } , N _ { r } , \tau )$ , where $\lambda _ { r }$ controls regional discounting and $\tau \in [ 0 , 1 ]$ controls relative stopping. The settings used in ReTailor are specified in Appendix $\mathrm { A } .$

## 3.4.3. Tensorized region-aware selection and relative stopping

To exploit informative patches for cross-sensor adaptation while avoiding regional overfitting, ReTailor combines parallelized region-aware greedy selection with �-controlled relative stopping. Candidates are first stably sorted by decreasing $b _ { i } \colon$

$$
\begin{array} { r l } & { \pi = \arg \operatorname { s o r t } _ { \downarrow } ^ { \mathrm { s t a b l e } } \left( ( b _ { i } ) _ { i = 1 } ^ { N _ { c } } \right) , } \\ & { \bar { b } _ { \ell } = b _ { \pi _ { \ell } } , \qquad \bar { r } _ { \ell } = r ( \pi _ { \ell } ) , \quad \ell \in \{ 1 , \ldots , N _ { c } \} , } \end{array}
$$

where ${ \pmb { \pi } } = ( \pi _ { 1 } , \ldots , \pi _ { N _ { c } } ) ^ { \top }$ is the permutation induced by the stable descending sort, whose �-th entry $\pi _ { \ell }$ records the original index of the candidate ranked $\ell \mathrm { - t h }$ , i.e., $\bar { b } _ { \ell } = b _ { \pi _ { \ell } }$ and $\bar { r } _ { \ell } = r _ { \pi _ { \ell } }$ ; equal scores retain their original index order. The search starts from:

$$
S ^ { ( 0 ) } = \varnothing , \qquad { \bf a } ^ { ( 0 ) } = { \bf 1 } _ { N _ { c } } , \qquad { \bf n } ^ { ( 0 ) } = { \bf 0 } _ { N _ { r } } ,
$$

where the superscript denotes the admission round of the greedy search. After � admissions, $S ^ { ( m ) }$ is the selected set, $\mathbf { a } ^ { ( m ) }$ is the availability mask over sorted positions, and $\mathbf { n } ^ { ( m ) }$ records the number of selected candidates in each coarse region. The displayed initialization is the $m = 0$ state, before any candidate is admitted.

At round $m \ < \ N _ { c } ,$ , ReTailor gathers each candidate’s current regional occupancy and computes its masked efective score. The sorted base scores and region indices are collected as $\bar { \mathbf { b } } = ( \bar { b } _ { 1 } , \dots , \bar { b } _ { N _ { c } } ) ^ { \top }$ and $\bar { \mathbf { r } } = ( \bar { r } _ { 1 } , \ldots , \bar { r } _ { N _ { c } } ) ^ { \top }$ , respectively:

$$
\begin{array} { r l } & { \mathbf { d } ^ { ( m ) } = \mathbf { 1 } _ { N _ { c } } + \lambda _ { r } \mathbf { n } ^ { ( m ) } [ \bar { \mathbf { r } } ] , } \\ & { \mathbf { e } ^ { ( m ) } = \mathrm { w h e r e } \left( \mathbf { a } ^ { ( m ) } = 1 , \bar { \mathbf { b } } \oslash \mathbf { d } ^ { ( m ) } , - \infty \right) , \qquad \lambda _ { r } \ge 0 , } \end{array}
$$

where $\mathbf { d } ^ { ( m ) }$ is the region-discount vector and $\mathbf { e } ^ { ( m ) }$ is the masked efective-score vector used for admission. The indexed vector $\mathbf { n } ^ { ( m ) } [ \bar { \mathbf { r } } ]$ gathers the current regional count for each sorted candidate, so $\lambda _ { r } \mathbf { n } ^ { ( m ) } [ \bar { \mathbf { r } } ]$ increases the discount in proportion to previous admissions from the same region. The operator where(�, �, �) acts element-wise, returning � when � is true and � otherwise; hence, an available candidate receives $e _ { \ell } ^ { ( m ) } = \bar { b } _ { \ell } / \mathop { d ^ { ( m ) } } _ { \ell }$ , whereas an admitted candidate is assigned −∞ and excluded from later rounds. Thus, $\mathbf { e } ^ { ( m ) }$ combines the base scores in <sup>̄</sup>� with the regional discount in $\mathbf { d } ^ { ( m ) }$ . Ranking the available candidates by $\mathbf { e } ^ { ( m ) }$ favors candidates with strong adaptation evidence and complementary regional coverage, thereby limiting repeated selection from the same region and reducing regional overfitting, as verified in Section 5.1.3.

At round $m ,$ the selector chooses the available position with the largest efective score, $\ell _ { m + 1 } = \operatorname* { m i n } \big ( \arg \operatorname* { m a x } _ { \ell } e _ { \ell } ^ { ( m ) } \big )$ and maps it to the original candidate index $i _ { m + 1 } = \pi _ { \ell _ { m + 1 } }$ The first candidate is admitted unconditionally and sets the reference score $e _ { \ell _ { 1 } } ^ { ( 0 ) }$ . Each subsequent candidate is admitted only when $e _ { \ell _ { m + 1 } } ^ { ( m ) } \ \geq \ \tau e _ { \ell _ { 1 } } ^ { ( 0 ) }$ ; otherwise, selection terminates before admission. Therefore, � directly specifies the minimum fraction of the first selected candidate’s efective score that a later candidate must retain, thereby preventing the admission of candidates with weak adaptation evidence. Once admitted, $i _ { m + 1 }$ is added to the selected set, the availability mask and regional counts are updated, and $\mathbf { d } ^ { ( m ) }$ and $\mathbf { e } ^ { ( m ) }$ are recomputed for the next round while the base-score vector <sup>̄</sup>� remains fixed. In our experiments, the average number of selected patches is approximately six. The returned set is $S ^ { t }$ with $M ^ { t } = m _ { \star }$ where $m _ { \star }$ is the number of admitted candidates.

The tensorized design greatly reduces the time cost of evidence gathering and patch selection (Table 16, Appendix D.2), enabling ReTailor to select informative patches without compromising deployment eficiency.

## 3.5. Rolling Residual Bootstrap

Consider a deployment stream $\{ T ^ { t } \} _ { t = 0 } ^ { T }$ of target images requiring pansharpening, all acquired by the same target sensor. Because the mismatch is dominated by sensordependent discrepancies, the corrections needed by diferent images in the stream are related: each combines a shared sensor-related component with image-specific variation. Section 5.2.1 formulates and verifies this in residual space (see also Fig. 2(a)). Reusing the previous Tailor parameters to initialize adaptation retains the previously learned sensorrelated component, so the objective of Section 3.3 only needs to focus on refining the image-specific part.

To establish this shared sensor-related prior at $t = 0 ,$ we perform full-patch adaptation across all $N _ { c }$ candidates of the seed image from the zero-output initialization $\phi _ { \mathrm { i n i t } }$ , yielding $\phi ^ { 0 }$ . For each $t \geq 1$ , adaptation starts from the inherited $\phi ^ { t - \overline { { 1 } } }$ and is optimized on the selected set $\Omega ^ { t }$ , giving $\phi ^ { t }$ , which is then used for full-patch inference on image � and passed to image � + 1 (Fig. 5). Although only the seed image undergoes full-patch adaptation while subsequent images use selected subsets, overall performance is insensitive to the choice of the seed image, as verified in Section 5.2.3; this observation is consistent with the shared correction component principle behind the bootstrap.

## 4. Experiments and Results

Section 4.1 describes the setup. Section 4.2 reports the main cross-sensor results, Section 4.3 compares ReTailor with parameter-matched LoRA, and Section 4.4 transfers the Feature Tailor to two other remote-sensing tasks.

## 4.1. Experimental Setup

Datasets and backbones. We evaluate QB→GF2 and WV3→WV2 on the PanCollection benchmark [1], with FusionNet [15] and U2Net [48] as frozen source-pretrained backbones (Appendix B.1).

Compared methods. The baselines comprise five traditional methods (BT-H [49], C-BDSD [50], BDSD-PC [47], MTF-GLP [7], MF [51]), the source-pretrained FusionNet and U2Net applied directly, two learning-based methods (WFANet [52], FusionMamba [53]), and two zero-shot methods (PsDip [22], ZS-Pan [20]). FT-only is our reference variant: the frozen backbone with the Feature Tailor adapted on eight random patches and cold-started for each image (budget chosen in Appendix C). It lacks only content-aware selection and rolling residual bootstrap, and thus isolates the residual design. Section 4.3 further compares with LoRA [21], a representative parameter-eficient fine-tuning method.

![](images/1d7913209108fb8409e9258b66609f683fd699cd9f31ee768eb715fa108309e7.jpg)  
Figure 5: Rolling residual bootstrap. The seed image adapts $\phi _ { \mathrm { i n i t } }$ on all candidates to give $\phi ^ { 0 }$ . Each later image inherits $\phi ^ { t - 1 }$ refines it on its selected patches, and passes $\phi ^ { t }$ on. Only Tailor parameters are transferred, and Adam is reinitialized for each image.

Metrics and timing. We use the reference-free HQNR, $D _ { \lambda }$ (spectral distortion), and $D _ { s }$ (spatial distortion) [46, 54], where higher HQNR and lower $D _ { \lambda }$ and $D _ { s }$ are better, and report image-level mean ± standard deviation. The seed image, i.e., the first image in the stream, is excluded from all reported metrics. It alone undergoes full-patch adaptation (Section 3.5), which may give it higher scores and thereby inflate the reported mean. In deployment, its one-time cost is amortized over the stream. For segmentation we report aAcc, mIoU, mAcc, and building IoU and accuracy, emphasizing mIoU and building IoU since the background dominates.

## 4.2. Main Cross-Sensor Pansharpening Results

Table 1(a) shows that $\mathrm { R e T a i l o r } _ { \mathrm { F u s i o n N e t } }$ achieves the best HQNR and $D _ { \lambda }$ on both transfers (0.8976 and 0.9245), exceeding the best traditional method in HQNR by 0.0821 and 0.0441. The next-best HQNR is ReTailor on WV3→WV2 and the zero-shot ZS-Pan on QB→GF2. The Tailor also lifts weaker source models: for U2Net, HQNR rises from 0.7058 to 0.8172 on QB→GF2 and from 0.8686 to 0.9215 on WV3→WV2. Compared with FT-only, ReTailor improves HQNR in all four backbone–transfer settings with fewer patches (6.05–6.47 on average) and 13.0%–20.1% shorter adaptation time (Table 1(b)). Fig. 6 gives visual examples.

## 4.3. Comparison with Parameter-Matched LoRA

To test whether ReTailor’s gains go beyond a small trainable-parameter count, we compare it with standard LoRA at closely matched parameter counts, using the same image stream, content-aware patches, and adaptation epochs per image. ReTailor adapts one residual module at a frozen interface and inherits it across images, whereas LoRA attaches low-rank branches to several layers and is cold-started for each image, following its standard per-image paradigm.

Fig. 7 shows that ReTailor cuts mean end-to-end duration (including patch selection, adaptation, and full-patch inference; LoRA’s timing excludes patch selection) by 45.6%– 52.9% in all four settings, and Table 2 shows that it lowers $D _ { \lambda }$ by 22.2%–47.5%, achieves higher HQNR in three settings, and lower $D _ { s }$ on both U2Net transfers. At matched capacity, a compact residual correction inherited across images is thus both faster and more faithful than layer-wise low-rank updates learned from scratch for each image.

## 4.4. Generalization to Other Cross-Sensor Tasks

We test whether the Feature Tailor, ReTailor’s core update module, transfers to arbitrary-scale super-resolution pansharpening (G-ZAP) and semantic segmentation (CrossEarth). In both, the original network is frozen and only the inserted Tailor is trained (Appendix A.3).

Super-resolution pansharpening. G-ZAP performs arbitrary-scale pansharpening with an implicit neural representation [55]. The Tailor is adapted from unlabeled target PAN/LRMS observations. Table 3 shows gains on all metrics for both transfers. On QB→GF2, HQNR rises from 0.8344 to 0.9508, with $D _ { \lambda }$ and $D _ { s }$ falling from 0.0793 and 0.0940 to 0.0203 and 0.0295. On WV3→WV2, HQNR rises from 0.9376 to 0.9468.

Semantic segmentation. On the CrossEarth Aerial-to-Satellite protocol, labeled target images are used to predict building masks. We compare Source-only with two targetsupervised variants under the same schedule: Head-only, which updates the original output head (14.47M parameters), and Feature Tailor (14.16M parameters, 2.18% fewer). In Table 4, Feature Tailor improves every metric, raising mIoU from 74.74 to 85.09 and building IoU from 50.26 to 70.61 over Head-only. The larger gain on building metrics than on aAcc shows that the improvement is not driven by the dominant background class, and Fig. 8 shows more complete building regions, including small and fragmented ones.

Main cross-sensor results on QB→GF2 and WV3→WV2. (a) Fusion quality; best in bold, second-best underlined. (b) Selected patches and adaptation time (patch selection plus Tailor optimization, excluding the identical full-patch inference); better in bold. (a) Fusion quality.
<table><tr><td></td><td colspan="3">QB→GF2</td><td colspan="3">WV3→WV2</td></tr><tr><td>Method</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td>BT-H</td><td> $0 . 8 1 5 5 \pm 0 . 0 2 8 8$ </td><td> $0 . 0 6 0 3 \pm 0 . 0 2 3 3$ </td><td> $0 . 1 3 2 2 \pm 0 . 0 1 8 8$ </td><td> $0 . 8 4 1 4 \pm 0 . 0 4 1 3$ </td><td> $0 . 0 7 9 0 \pm 0 . 0 3 2 9$ </td><td> $0 . 0 8 6 8 \pm 0 . 0 1 6 0$ </td></tr><tr><td>C-BDSD</td><td> $0 . 6 9 9 6 \pm 0 . 0 3 2 3$ </td><td> $0 . 2 1 4 9 \pm 0 . 0 3 0 2$ </td><td> $0 . 1 0 9 1 \pm 0 . 0 1 5 4$ </td><td> $0 . 6 9 5 6 \pm 0 . 0 4 6 1$ </td><td> $0 . 2 2 5 3 \pm 0 . 0 4 8 8$ </td><td> $0 . 1 0 1 9 \pm 0 . 0 2 9 6$ </td></tr><tr><td>BDSD-PC</td><td> $0 . 7 7 9 7 \pm 0 . 0 3 9 6$ </td><td> $0 . 0 7 6 9 \pm 0 . 0 2 8 5$ </td><td> $0 . 1 5 5 7 \pm 0 . 0 2 7 7$ </td><td> $0 . 8 2 2 8 \pm 0 . 0 3 4 4$ </td><td> $0 . 1 4 3 9 \pm 0 . 0 2 7 2$ </td><td> $0 . 0 3 9 1 \pm 0 . 0 1 7 2$ </td></tr><tr><td>MTF-GLP</td><td> $0 . 8 0 6 4 \pm 0 . 0 4 4 3$ </td><td> $0 . 0 4 2 1 \pm 0 . 0 1 7 2$ </td><td> $0 . 1 5 8 6 \pm 0 . 0 3 4 1$ </td><td> $0 . 8 8 0 4 \pm 0 . 0 4 2 7$ </td><td> $0 . 0 3 6 0 \pm 0 . 0 1 6 6$ </td><td> $0 . 0 8 7 2 \pm 0 . 0 3 1 0$ </td></tr><tr><td>MF</td><td> $0 . 8 0 0 0 \pm 0 . 0 4 3 8$ </td><td> $0 . 0 5 2 6 \pm 0 . 0 1 9 6$ </td><td> $0 . 1 5 6 0 \pm 0 . 0 3 2 6$ </td><td> $0 . 8 8 0 2 \pm 0 . 0 4 1 9$ </td><td> $0 . 0 4 4 0 \pm 0 . 0 1 9 1$ </td><td> $0 . 0 7 9 7 \pm 0 . 0 2 6 5$ </td></tr><tr><td>PsDip</td><td> $0 . 7 8 2 5 \pm 0 . 0 3 2 5$ </td><td> $0 . 1 3 2 3 \pm 0 . 0 2 8 6$ </td><td> $0 . 0 9 8 1 \pm 0 . 0 2 5 6$ </td><td> $0 . 8 9 8 0 \pm 0 . 0 2 2 6$ </td><td> $0 . 0 3 8 5 \pm 0 . 0 2 3 9$ </td><td> $0 . 0 6 5 9 \pm 0 . 0 1 5 8$ </td></tr><tr><td>ZS-Pan</td><td> $0 . 8 9 2 5 \pm 0 . 0 2 4 0$ </td><td> $0 . 0 7 7 8 \pm 0 . 0 1 8 2$ </td><td> $\mathbf { 0 . 0 3 2 3 \pm 0 . 0 1 5 9 }$ </td><td> $0 . 9 1 1 2 \pm 0 . 0 3 3 6$ </td><td> $0 . 0 4 7 6 \pm 0 . 0 2 7 0$ </td><td> $0 . 0 4 3 5 \pm 0 . 0 1 3 0$ </td></tr><tr><td>FusionMamba</td><td> $0 . 7 6 2 7 \pm 0 . 0 7 2 4$ </td><td> $0 . 1 6 2 7 \pm 0 . 0 6 9 5$ </td><td> $0 . 0 8 9 3 \pm 0 . 0 3 5 2$ </td><td> $0 . 9 0 2 2 \pm 0 . 0 2 4 2$ </td><td> $0 . 0 5 7 2 \pm 0 . 0 2 7 2$ </td><td> $0 . 0 4 2 9 \pm 0 . 0 1 1 2$ </td></tr><tr><td>FusionNet</td><td> $0 . 8 4 0 3 \pm 0 . 0 5 0 5$ </td><td> $0 . 1 1 2 7 \pm 0 . 0 4 9 9$ </td><td> $0 . 0 5 3 2 \pm 0 . 0 1 0 6$ </td><td> $0 . 8 8 8 6 \pm 0 . 0 2 1 2$ </td><td> $0 . 0 5 5 3 \pm 0 . 0 2 6 8$ </td><td> $0 . 0 5 9 1 \pm 0 . 0 1 5 4$ </td></tr><tr><td>U2Net</td><td> $0 . 7 0 5 8 \pm 0 . 0 7 1 3$ </td><td> $0 . 2 2 2 8 \pm 0 . 0 6 3 7$ </td><td> $0 . 0 9 2 6 \pm 0 . 0 4 3 7$ </td><td> $0 . 8 6 8 6 \pm 0 . 0 8 0 3$ </td><td> $0 . 0 9 5 4 \pm 0 . 0 7 8 6$ </td><td> $0 . 0 4 0 4 \pm 0 . 0 0 9 7$ </td></tr><tr><td>WFANet</td><td> $0 . 7 0 8 8 \pm 0 . 0 4 4 9$ </td><td> $0 . 1 9 7 1 \pm 0 . 0 4 7 9$ </td><td> $0 . 1 1 6 5 \pm 0 . 0 3 7 9$ </td><td> $0 . 9 1 4 2 \pm 0 . 0 2 9 6$ </td><td> $0 . 0 5 1 0 \pm 0 . 0 2 9 7$ </td><td> $0 . 0 3 6 7 \pm 0 . 0 0 5 3$ </td></tr><tr><td> $\mathsf { F T - o n l y } _ { \mathrm { F u s i o n N e t } }$ </td><td> $0 . 8 8 7 1 \pm 0 . 0 5 7 4$ </td><td> $0 . 0 7 6 3 \pm 0 . 0 4 5 8$ </td><td> $\underline { { 0 . 0 4 0 4 } } \pm 0 . 0 1 7 9$ </td><td>0.9075 ± 0.0259</td><td> $0 . 0 4 6 3 \pm 0 . 0 2 1 7$ </td><td> $0 . 0 4 8 5 \pm 0 . 0 1 5 0$ </td></tr><tr><td> $\mathsf { F T - o n l y } _ { \mathrm { U 2 N e t } }$ </td><td> $0 . 8 0 5 8 \pm 0 . 0 4 7 8$ </td><td> $0 . 1 0 8 7 \pm 0 . 0 4 0 2$ </td><td> $0 . 0 9 5 9 \pm 0 . 0 3 6 7$ </td><td> $0 . 9 2 0 0 \pm 0 . 0 4 6 0$ </td><td> $0 . 0 5 8 3 \pm 0 . 0 4 4 3$ </td><td> $\mathbf { 0 . 0 2 3 2 \pm 0 . 0 0 5 1 }$ </td></tr><tr><td> $\mathsf { R e T a i l o r } _ { \mathrm { F u s i o n N e t } }$ </td><td> $\mathbf { 0 . 8 9 7 6 \pm 0 . 0 2 6 4 }$ </td><td> $\mathbf { 0 . 0 3 3 9 \pm 0 . 0 2 9 2 }$ </td><td> $0 . 0 7 0 6 \pm 0 . 0 2 1 8$ </td><td> $\mathbf { 0 . 9 2 4 5 \pm 0 . 0 3 9 4 }$ </td><td> $\mathbf { 0 . 0 3 3 5 \pm 0 . 0 1 9 8 }$ </td><td> $0 . 0 4 3 8 \pm 0 . 0 2 3 3$ </td></tr><tr><td> $\mathsf { R e T a i l o r } _ { \mathrm { U 2 N e t } }$ </td><td> $0 . 8 1 7 2 \pm 0 . 0 4 8 5$ </td><td> $0 . 0 9 9 0 \pm 0 . 0 4 1 0$ </td><td> $0 . 0 9 3 1 \pm 0 . 0 3 3 8$ </td><td> $0 . 9 2 1 5 \pm 0 . 0 4 1 9$ </td><td> $0 . 0 5 0 8 \pm 0 . 0 4 1 6$ </td><td> $0 . 0 2 9 2 \pm 0 . 0 0 9 1$ </td></tr></table>

(b) Adaptation cost.
<table><tr><td colspan="2"></td><td colspan="2">QB→GF2</td><td colspan="2">WV3→WV2</td></tr><tr><td>Backbone</td><td>Method</td><td>Selected ↓</td><td>Time (s) ↓</td><td>Selected ↓</td><td>Time (s) ↓</td></tr><tr><td rowspan="2">FusionNet</td><td>FT-only</td><td> $8 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 1 4 8 8 \pm 0 . 0 3 6 3$ </td><td> $8 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 1 3 1 \pm 0 . 0 8 2 4$ </td></tr><tr><td>ReTailor</td><td> ${ \bf 6 . 4 7 \pm 1 . 6 0 }$ </td><td> $\mathbf { 0 . 1 2 4 0 \pm 0 . 0 3 8 4 }$ </td><td> ${ \bf 6 . 0 5 \pm 1 . 6 1 }$ </td><td> $\mathbf { 0 . 1 7 2 9 \pm 0 . 0 5 6 8 }$ </td></tr><tr><td rowspan="2">U2Net</td><td>FT-only</td><td> $8 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 3 2 6 \pm 0 . 0 5 7 4$ </td><td> $8 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 4 0 8 \pm 0 . 0 3 8 1$ </td></tr><tr><td>ReTailor</td><td> ${ \bf 6 . 4 7 \pm 1 . 6 0 }$ </td><td> $\mathbf { 0 . 2 0 2 4 \pm 0 . 0 7 5 8 }$ </td><td> ${ \bf 6 . 0 5 \pm 1 . 6 1 }$ </td><td> $\mathbf { 0 . 1 9 2 4 \pm 0 . 0 5 6 3 }$ </td></tr></table>

![](images/77bddfdcee0e0209f11a3d85bb7bb9cc9e1b3a223e8b4d5f5750c2f6d4e41801.jpg)  
Figure 6: Qualitative comparison. For each transfer: fused image with two regions of interest (ROIs), magnified ROIs, and HQNR map (0–1, warmer is higher). Columns: the seven baselines with the highest HQNR (order of Table 1) and the FT-only and ReTailor variants.

Table 2  
ReTailor versus parameter-matched LoRA [21]. Both use content-aware patches and 10 adaptation epochs per image; better in bold.
<table><tr><td>Backbone Transfer</td><td></td><td>Method</td><td>Params</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td rowspan="3">FusionNet</td><td rowspan="2"> ${ \mathsf { W V } } 3 {  } { \mathsf { W V } } 2$ </td><td>ReTailor</td><td>221,472</td><td> $\mathbf { 0 . 9 2 4 5 \pm 0 . 0 3 9 4 }$ </td><td> $\mathbf { 0 . 0 3 3 5 \pm 0 . 0 1 9 8 }$ </td><td> $0 . 0 4 3 8 \pm 0 . 0 2 3 3$ </td></tr><tr><td>LoRA</td><td>222,000</td><td> $0 . 9 2 4 1 \pm 0 . 0 2 9 7$ </td><td> $0 . 0 4 4 9 \pm 0 . 0 2 3 2$ </td><td> $\mathbf { 0 . 0 3 2 6 \pm 0 . 0 1 2 6 }$ </td></tr><tr><td rowspan="2"> $\mathsf { Q B } \to \mathsf { G F } 2$ </td><td>ReTailor</td><td>221,472</td><td> $0 . 8 9 7 6 \pm 0 . 0 2 6 4$ </td><td> $\mathbf { 0 . 0 3 3 9 \pm 0 . 0 2 9 2 }$ </td><td> $0 . 0 7 0 6 \pm 0 . 0 2 1 8$ </td></tr><tr><td rowspan="2"></td><td>LoRA</td><td>221,920</td><td> $\mathbf { 0 . 9 1 } 2 2 \pm 0 . 0 4 6 6$ </td><td> $0 . 0 6 4 6 \pm 0 . 0 4 2 8$ </td><td> $\mathbf { 0 . 0 2 5 0 \pm 0 . 0 0 7 9 }$ </td></tr><tr><td rowspan="2"> ${ \mathsf { W V } } 3 {  } { \mathsf { W V } } 2$  U2Net</td><td>ReTailor</td><td>221,472</td><td> $\mathbf { 0 . 9 2 1 5 \pm 0 . 0 4 1 9 }$ </td><td> $\mathbf { 0 . 0 5 0 8 \pm 0 . 0 4 1 6 }$ </td><td> $\mathbf { 0 . 0 2 9 2 \sigma \pm 0 . 0 0 9 1 }$ </td></tr><tr><td rowspan="2"></td><td>LoRA ReTailor</td><td>224,288 221,472</td><td> $0 . 9 0 6 2 \pm 0 . 0 4 6 3$   $\mathbf { 0 . 8 1 7 2 \pm 0 . 0 4 8 5 }$ </td><td> $0 . 0 6 5 3 \pm 0 . 0 4 3 7$ </td><td> $0 . 0 3 0 7 \pm 0 . 0 0 8 7$ </td></tr><tr><td rowspan="2">QB→GF2</td><td>LoRA</td><td>226,677</td><td> $0 . 7 3 2 6 \pm 0 . 0 6 2 5$ </td><td> $\mathbf { 0 . 0 9 9 0 \pm 0 . 0 4 1 0 }$   $0 . 1 7 7 3 \pm 0 . 0 5 2 5$ </td><td> $\mathbf { 0 . 0 9 3 1 \pm 0 . 0 3 3 8 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $0 . 1 1 0 2 \pm 0 . 0 4 1 6$ </td></tr></table>

![](images/38624dba881970520c6368d119928b1a02ad845668be5b913f050db71fd73e5e.jpg)  
Table 4  
Figure 7: End-to-end duration of ReTailor and LoRA. Mean perimage latency in the four backbone–transfer settings (smaller radius is faster); ReTailor’s latency includes patch selection, adaptation, and full-patch inference.

Table 3  
Feature Tailor on G-ZAP [55]. Mean ± std over 20 test images per transfer; better in bold.
<table><tr><td>Method</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td> $Q B {  } G F 2$ </td><td></td><td></td><td></td></tr><tr><td>G-ZAP</td><td> $0 . 8 3 4 4 \pm 0 . 0 3 1 9$ </td><td> $0 . 0 7 9 3 \pm 0 . 0 2 3 2$ </td><td> $0 . 0 9 4 0 \pm 0 . 0 1 8 9$ </td></tr><tr><td> ${ \mathsf { G } } { - } { \mathsf { Z } } { \mathsf { A } } { \mathsf { P } } + { \mathsf { F } } { \mathsf { T } }$ </td><td> $\mathbf { 0 . 9 5 0 8 \pm 0 . 0 1 4 3 }$ </td><td> $\mathbf { 0 . 0 2 0 3 \pm 0 . 0 0 7 6 }$ </td><td> $\mathbf { 0 . 0 2 9 5 \pm 0 . 0 1 3 2 }$ </td></tr><tr><td> $W V 3  W V 2$ </td><td></td><td></td><td></td></tr><tr><td>G-ZAP</td><td> $0 . 9 3 7 6 \pm 0 . 0 2 2 9$ </td><td> $0 . 0 3 4 0 \pm 0 . 0 2 1 0$ </td><td> $0 . 0 2 9 4 \pm 0 . 0 0 6 5$ </td></tr><tr><td> ${ \mathsf { G } } { - } { \mathsf { Z } } { \mathsf { A } } { \mathsf { P } } + { \mathsf { F } } { \mathsf { T } }$ </td><td> $\mathbf { 0 . 9 4 6 8 \pm 0 . 0 1 2 9 }$ </td><td> $\mathbf { 0 . 0 2 5 4 \pm 0 . 0 0 9 6 }$ </td><td> $\mathbf { 0 . 0 2 8 5 \pm 0 . 0 0 9 1 }$ </td></tr></table>

## 5. Ablation Study

We ablate ReTailor’s two complementary components one at a time: content-aware patch selection retains the most informative patches to improve fusion quality, and rolling residual bootstrap preserves the learned sensor-related correction across images. Section 5.1 compares selection with random sampling and examines its budget, score composition, regional discounting, and scalability. Section 5.2 compares rolling with cold start and examines whether corrections are shared across images and whether the stream is robust to its seed image and length. All variants share the same backbone, 10-epoch schedule, and optimizer, with FT-only (Random-8, cold start) as the baseline.

CrossEarth Aerial-to-Satellite segmentation [56, 57] (%). Best in bold, second-best underlined.
<table><tr><td></td><td></td><td></td><td> $\mathsf { m A c c } \uparrow$ </td><td>Bldg. loU ↑</td><td>Bldg. Acc. ↑</td></tr><tr><td>Method</td><td> $\mathsf { a A c c } \uparrow$ </td><td>mloU ↑</td><td></td><td></td><td></td></tr><tr><td>Source-only</td><td>99.10 99.23</td><td>71.67 74.74</td><td>78.20 80.64</td><td>44.25 50.26</td><td>56.76 61.56</td></tr><tr><td>Head-only</td><td></td><td>85.09</td><td>90.97</td><td>70.61</td><td>82.14</td></tr><tr><td>Feature Tailor</td><td>99.57</td><td></td><td></td><td></td><td></td></tr></table>

## 5.1. Content-Aware Patch Selection

Selection acts within each image. Section 5.1.1 tests whether content-aware scoring picks better patches than random sampling and whether adaptive stopping reduces cost, Section 5.1.2 justifies anchoring the score on PAN variation, Section 5.1.3 verifies the contribution of regional discounting, and Section 5.1.4 tests whether the advantage persists on large images.

## 5.1.1. Selection quality and budget

Random-8 and Content-aware-8 share one budget, so their diference isolates which patches are chosen. Content-aware-Adaptive adds relative stopping to the same scoring, isolating the efect of a smaller, image-dependent budget.

In Table 5, Content-aware-8 gives higher HQNR and lower $D _ { \lambda }$ than Random-8 in all four settings, so observationbased scoring supplies better adaptation evidence at the same budget. Content-aware-Adaptive retains these advantages over Random-8 in all four settings while cutting adaptation time by 9.9%–17.1% (Fig. 9).

## 5.1.2. Evidence composition

The selector anchors its score on PAN variation and adds LRMS inter-band variation as a supplement (Eq. (3)). To justify this asymmetry, we compare PAN-only-8, LRMSonly-8, LRMS-anchored-8 (roles reversed, same $\lambda _ { c } = 0 . 2 )$ , and Content-aware-8 using FusionNet on QB→GF2, with per-image cold start as in Table 5.

ReTailor  
![](images/4d2b5c211d29850953c1aa24b9fc261bd0a214a378f136070fed60cde4374613.jpg)  
Figure 8: Qualitative CrossEarth segmentation [56, 57]. First eight Satellite test samples; columns: input, ground truth, Source-only, Head-only, Feature Tailor (white: building).

Ablation of patch selection. All variants use per-image cold start; best in bold, second-best underlined within each backbone–transfer block.
<table><tr><td>Backbone</td><td>Transfer</td><td>Selector</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td rowspan="4">FusionNet</td><td rowspan="4">WV3→WV2</td><td>Random-8</td><td> $0 . 9 0 7 5 \pm 0 . 0 2 5 9$ </td><td> $0 . 0 4 6 3 \pm 0 . 0 2 1 7$ </td><td> $0 . 0 4 8 5 \pm 0 . 0 1 5 0$ </td></tr><tr><td>Content-aware-8</td><td> $\mathbf { 0 . 9 1 8 6 \pm 0 . 0 2 6 1 }$ </td><td> $\mathbf { 0 . 0 3 8 5 \pm 0 . 0 2 4 1 }$ </td><td> $\mathbf { 0 . 0 4 4 6 \pm 0 . 0 1 2 8 }$ </td></tr><tr><td>Content-aware-Adaptive</td><td> $\underline { { 0 . 9 0 8 8 } } \pm 0 . 0 2 2 4$ </td><td> $\underline { { 0 . 0 4 5 3 \pm 0 . 0 2 1 1 } }$ </td><td> $\underline { { 0 . 0 4 8 1 \pm 0 . 0 1 4 6 } }$ </td></tr><tr><td>Random-8</td><td> $0 . 8 8 7 1 \pm 0 . 0 5 7 4$ </td><td> $0 . 0 7 6 3 \pm 0 . 0 4 5 8$ </td><td> $\mathbf { 0 . 0 4 0 4 \pm 0 . 0 1 7 9 }$ </td></tr><tr><td rowspan="4"></td><td rowspan="4">QB→GF2</td><td>Content-aware-8</td><td> $\mathbf { 0 . 8 9 7 7 \pm 0 . 0 4 1 1 }$ </td><td> $\mathbf { 0 . 0 6 1 2 \pm 0 . 0 4 0 8 }$ </td><td> $0 . 0 4 3 8 \pm 0 . 0 1 3 8$ </td></tr><tr><td>Content-aware-Adaptive</td><td> $\underline { { 0 . 8 9 0 1 \pm 0 . 0 4 6 7 } }$ </td><td></td><td> $\underline { { 0 . 0 4 1 1 \pm 0 . 0 1 4 2 } }$ </td></tr><tr><td></td><td></td><td> $\underline { { 0 . 0 7 1 8 \pm 0 . 0 4 7 8 } }$ </td><td></td></tr><tr><td>Random-8</td><td> $0 . 9 2 0 0 \pm 0 . 0 4 6 0$ </td><td> $0 . 0 5 8 3 \pm 0 . 0 4 4 3$ </td><td> $\mathbf { 0 . 0 2 3 2 \pm 0 . 0 0 5 1 }$ </td></tr><tr><td rowspan="4">U2Net</td><td rowspan="4">WV3→WV2</td><td>Content-aware-8</td><td> $\mathbf { 0 . 9 2 6 5 \pm 0 . 0 4 4 2 }$ </td><td> $0 . 0 5 1 2 \pm 0 . 0 4 5 0$ </td><td> $0 . 0 2 3 5 \pm 0 . 0 0 5 2$ </td></tr><tr><td>Content-aware-Adaptive</td><td> $\underline { { 0 . 9 2 1 7 \pm 0 . 0 1 9 3 } }$ </td><td> $\overline { { { \bf 0 . 0 4 1 8 \pm 0 . 0 1 5 6 } } }$ </td><td> $\overline { { 0 . 0 3 8 1 \pm 0 . 0 2 4 8 } }$ </td></tr><tr><td>Random-8</td><td> $0 . 8 0 5 8 \pm 0 . 0 4 7 8$ </td><td> $0 . 1 0 8 7 \pm 0 . 0 4 0 2$ </td><td> $0 . 0 9 5 9 \pm 0 . 0 3 6 7$ </td></tr><tr><td>Content-aware-8</td><td> $\mathbf { 0 . 8 1 8 0 \pm 0 . 0 5 3 9 }$ </td><td> $\mathbf { 0 . 0 9 8 0 \pm 0 . 0 4 8 5 }$ </td><td> $0 . 0 9 3 1 \pm 0 . 0 3 4 6$   $\overline { { { \bf 0 . 0 8 8 6 \pm 0 . 0 2 3 1 } } }$ </td></tr></table>

Ablation of evidence composition. QB→GF2, FusionNet, eight patches per variant, per-image cold start; best in bold, secondbest underlined.
<table><tr><td>Variant</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td>PAN-only-8</td><td> $0 . 8 7 9 4 \pm 0 . 0 5 0 2$ </td><td> $0 . 0 8 2 4 \pm 0 . 0 4 0 9$ </td><td> $\mathbf { 0 . 0 4 2 2 \pm 0 . 0 1 5 8 }$ </td></tr><tr><td>LRMS-only-8</td><td> $0 . 8 6 7 5 \pm 0 . 0 6 2 9$ </td><td> $\overline { { 0 . 0 9 1 0 \pm 0 . 0 5 0 4 } }$ </td><td> $0 . 0 4 6 6 \pm 0 . 0 2 0 1$ </td></tr><tr><td>LRMS-anchored-8</td><td> $0 . 8 6 8 8 \pm 0 . 0 5 9 0$ </td><td> $0 . 0 8 9 6 \pm 0 . 0 4 6 9$ </td><td> $0 . 0 4 6 6 \pm 0 . 0 1 9 1$ </td></tr><tr><td>Content-aware-8</td><td> $\mathbf { 0 . 8 9 7 7 \pm 0 . 0 4 1 1 }$ </td><td> $\mathbf { 0 . 0 6 1 2 \pm 0 . 0 4 0 8 }$ </td><td> $\underline { { 0 . 0 4 3 8 \pm 0 . 0 1 3 8 } }$ </td></tr></table>

PAN-only-8 outperforms LRMS-only-8 (HQNR +0.0119), so PAN variation is the stronger cue. Reversing the roles barely helps (HQNR 0.8675 to 0.8688), whereas the PANanchored Content-aware-8 reaches the best HQNR (0.8977) and ${ D _ { \lambda } } ~ ( 0 . 0 6 1 2 )$ . PAN variation should thus anchor the score, with LRMS variation adding complementary spectral evidence.

## 5.1.3. Regional discount

Regional discounting lowers the priority of candidates whose coarse region is already represented (Sec. 3.4). To verify this design, we compare Random-8, No-discount-8 $( \lambda _ { r } = 0$ , which admits the eight highest-scoring candidates regardless of their regions), and Content-aware-8 using FusionNet on both transfers, with per-image cold start as in Table 5.

In Table 7, No-discount-8 outperforms Random-8 on both transfers (HQNR +0.0008 and +0.0045; $D _ { \lambda }$ down to 0.0428 and 0.0684), so the base score itself identifies informative patches. Adding regional discounting further raises HQNR by 0.0103 and 0.0061 (to 0.9186 and 0.8977) and lowers $D _ { \lambda }$ to 0.0385 and 0.0612. The score and the discount are therefore complementary: the former locates informative patches, and the latter keeps the fixed budget from clustering in a few high-variation regions.

![](images/a37e03972239a42ebb2bac6be32650ef0fbe73b445260d492d4d67b175f74db8.jpg)  
Figure 9: Adaptation time of the three selectors. Bars: mean per-image time (patch selection plus Tailor optimization); numbers inside: mean selected patches (Table 5). The vertical axis starts at 0.12 s.

Ablation of regional discounting. FusionNet, eight patches per variant, per-image cold start; best in bold, second-best underlined within each transfer block. Random-8 and Contentaware-8 replicate Table 5.
<table><tr><td>Variant</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td>WV3→WV2</td><td></td><td></td><td></td></tr><tr><td>Random-8</td><td> $0 . 9 0 7 5 \pm 0 . 0 2 5 9$ </td><td> $0 . 0 4 6 3 \pm 0 . 0 2 1 7$ </td><td> $0 . 0 4 8 5 \pm 0 . 0 1 5 0$ </td></tr><tr><td>No-discount-8</td><td> $0 . 9 0 8 3 \pm 0 . 0 2 3 6$ </td><td> $0 . 0 4 2 8 \pm 0 . 0 2 3 5$ </td><td> $\overline { { 0 . 0 5 1 2 \pm 0 . 0 1 4 0 } }$ </td></tr><tr><td>Content-aware-8</td><td> $\overline { { { \bf 0 . 9 1 8 6 \pm 0 . 0 2 6 1 } } }$ </td><td> $\overline { { { \bf 0 . 0 3 8 5 \pm 0 . 0 2 4 1 } } }$ </td><td> $\mathbf { 0 . 0 4 4 6 \pm 0 . 0 1 2 8 }$ </td></tr><tr><td>QB→GF2</td><td></td><td></td><td></td></tr><tr><td>Random-8</td><td> $0 . 8 8 7 1 \pm 0 . 0 5 7 4$ </td><td> $0 . 0 7 6 3 \pm 0 . 0 4 5 8$ </td><td> $\mathbf { 0 . 0 4 0 4 \pm 0 . 0 1 7 9 }$ </td></tr><tr><td>No-discount-8</td><td> $0 . 8 9 1 6 \pm 0 . 0 4 9 9$ </td><td> $0 . 0 6 8 4 \pm 0 . 0 4 3 8$ </td><td> $0 . 0 4 3 1 \pm 0 . 0 1 2 9$ </td></tr><tr><td>Content-aware-8</td><td> $\overline { { { \bf 0 . 8 9 7 7 \pm 0 . 0 4 1 1 } } }$ </td><td> $\overline { { { \bf 0 . 0 6 1 2 \pm 0 . 0 4 0 8 } } }$ </td><td> $\overline { { 0 . 0 4 3 8 \pm 0 . 0 1 3 8 } }$ </td></tr></table>

## 5.1.4. Large images

Intuitively, enlarging the image hurts random selection but helps content-aware selection: as the candidate pool grows, random sampling is more likely to admit weak patches for Tailor adaptation, while the larger image also produces more good and more diverse patches for the scorer to pick from. Content-aware selection should therefore widen its lead over random as the image grows. To verify this, we compare Random-8 with Content-aware-8 on two $4 , 0 0 0 \times 4 , 0 0 0$ images, one per transfer, with per-image cold start as in Table 5.

In Table 8, content-aware selection remains better than random selection on all three metrics in all four cases, and its HQNR gains widen to 0.0247–0.0540 from 0.0065–0.0122 on standard images (Table 5), consistent with the intuition above: random sampling admits more uninformative patches from the larger pool, whereas the scorer benefits from the richer and more diverse informative regions that the larger image supplies.

## 5.2. Rolling Residual Bootstrap

Rolling residual bootstrap acts along the image stream. Section 5.2.1 probes whether the reused corrections are shared across images, Section 5.2.2 compares rolling with perimage cold start under matched budgets, and Sections 5.2.3 and 5.2.4 further test the design by varying the seed image and lengthening the stream.

## 5.2.1. Shared correction across images

Rolling relies on the corrections required by images of one transfer being related. We formalize this as a decomposition. Let $\mathbf { \Delta } \Delta _ { t , i } = \mathbf { Z } _ { t , i } ^ { * } - \mathbf { Z } _ { t , i }$ be the residual correction the Tailor applies to candidate patch � of image �, and let

$$
\mathbf { r } _ { t } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } \mathrm { G A P } ( \Delta _ { t , i } )
$$

be its image-level embedding, where GAP(⋅) is spatial global average pooling. We model

$$
\mathbf { r } _ { t } = \mathbf { c } + \mathbf { e } _ { t } ,\tag{4}
$$

where � is a sensor-related component determined by the transfer (source model, target sensor, and interface) and common to all images, and $\mathbf { e } _ { t }$ is image-specific variation caused by scene content and noise. If $\mathbf { e } _ { t }$ has zero mean and is uncorrelated across images, then $\mathbb { E } [ { \bf r } _ { a } ^ { \top } { \bf r } _ { b } ] = \| { \bf c } \| _ { 2 } ^ { 2 }$ and $\mathbb { E } \| \mathbf { r } _ { t } \| _ { 2 } ^ { 2 } = \| \mathbf { c } \| _ { 2 } ^ { 2 } + \mathbb { E } \| \mathbf { e } _ { t } \| _ { 2 } ^ { 2 }$ for � ≠ $^ { b , }$ so the mean pairwise cosine similarity of $\mathbf { r } _ { t }$ approximates

$$
\rho = \frac { \Vert \mathbf { c } \Vert _ { 2 } ^ { 2 } } { \Vert \mathbf { c } \Vert _ { 2 } ^ { 2 } + \mathbb { E } \Vert \mathbf { e } _ { t } \Vert _ { 2 } ^ { 2 } } \in [ 0 , 1 ] ,
$$

the share of the correction that is sensor-related. Eq. (4) thus has two testable implications: $\rho$ should be high, and the residual similarity should exceed the similarity of the frozen features $\mathbf { z } _ { t }$ (defined like $\mathbf { r } _ { t }$ from $\mathbf { Z } _ { t , i } )$ , which serves as a control for similarity unrelated to the correction. A high � also explains why rolling helps: the inherited Tailor already contains �, so adaptation only needs to learn $\mathbf { e } _ { t } ,$ , which Section 5.2.2 verifies.

To test this, we use FusionNet on both transfers. For each of the 20 target images, a Tailor is cold-started and adapted with Random-8 for 10 epochs, carrying nothing across images, so no common component is injected by inheritance. We then compute the cosine similarity over all image pairs of a transfer for $\mathbf { r } _ { t }$ and for $\mathbf { z } _ { t }$ . The probe is analysisonly and not part of deployment.

Table 10 and Fig. 2(a) support both implications. The residual similarity, which estimates $\rho ,$ is 0.8537 on WV3→WV2 and 0.9383 on QB→GF2, so most of the correction is sensorrelated, and Fig. 2(a) shows uniformly high similarity across all image pairs. It exceeds the latent similarity by 0.1755 and 0.0699 with lower variability, so the shared structure comes from the correction rather than from the features themselves. The lower $\rho$ on WV3→WV2 indicates a larger image-specific part $\mathbf { e } _ { t } ,$ which the per-image refinement of rolling residual bootstrap handles. Independently adapted Tailors thus converge to a common correction �, consistent with the gain of inherited initialization in Section 5.2.2.

Table 8  
Patch selection on large images. One $4 { , } 0 0 0 \times 4 { , } 0 0 0$ image each from WV2 and GF2, eight patches per method, per-image cold start; better in bold.
<table><tr><td>Backbone</td><td>Transfer</td><td>Selection</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td rowspan="5">FusionNet</td><td rowspan="2">WV3→WV2</td><td>Random-8</td><td>0.9266</td><td>0.0579</td><td>0.0165</td></tr><tr><td>Content-aware-8</td><td>0.9513</td><td>0.0392</td><td>0.0098</td></tr><tr><td rowspan="2"> $\mathsf { Q B } \to \mathsf { G F } 2$ </td><td>Random-8</td><td>0.8211</td><td>0.1378</td><td>0.0476</td></tr><tr><td>Content-aware-8</td><td>0.8751</td><td>0.1012</td><td>0.0263</td></tr><tr><td rowspan="2">WV3→WV2</td><td>Random-8</td><td>0.8358</td><td>0.1049</td><td>0.0663</td></tr><tr><td></td><td>Content-aware-8</td><td>0.8808</td><td>0.0887</td><td>0.0334</td></tr><tr><td rowspan="3">U2Net</td><td>QB→GF2</td><td>Random-8</td><td>0.7199</td><td>0.1982</td><td>0.1021</td></tr><tr><td></td><td>Content-aware-8</td><td>0.7559</td><td>0.1684</td><td>0.0910</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9  
Ablation of rolling residual bootstrap. Both arms use Random-8; better in bold. Time covers patch selection and Tailor optimization.
<table><tr><td>Backbone</td><td>Transfer</td><td>Initialization</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td><td>Adaptation time (s) ↓</td></tr><tr><td rowspan="2">FusionNet</td><td> ${ \mathsf { W V } } 3 {  } { \mathsf { W V } } 2$ </td><td>Cold-start Rolling</td><td> $0 . 9 0 7 5 \pm 0 . 0 2 5 9$   $\mathbf { 0 . 9 2 4 9 \pm 0 . 0 4 1 7 }$ </td><td> $0 . 0 4 6 3 \pm 0 . 0 2 1 7$   $\mathbf { 0 . 0 3 1 7 \pm 0 . 0 1 9 2 }$ </td><td> $0 . 0 4 8 5 \pm 0 . 0 1 5 0$   $\mathbf { 0 . 0 4 5 2 \pm 0 . 0 2 6 6 }$ </td><td> $0 . 2 1 3 1 \pm 0 . 0 8 2 4$   $\mathbf { 0 . 1 9 7 4 \ : \pm { \ : 0 . 0 5 8 1 } }$ </td></tr><tr><td>QB→GF2</td><td>Cold-start Rolling</td><td> $0 . 8 8 7 1 \pm 0 . 0 5 7 4$   $\mathbf { 0 . 9 0 7 4 \pm 0 . 0 2 1 2 }$ </td><td> $0 . 0 7 6 3 \pm 0 . 0 4 5 8$   $\mathbf { 0 . 0 2 6 0 \pm 0 . 0 2 1 2 }$ </td><td> $\mathbf { 0 . 0 4 0 4 \ : \pm 0 . 0 1 7 9 }$   $0 . 0 6 8 3 \pm 0 . 0 1 8 6$ </td><td> $0 . 1 4 8 8 \pm 0 . 0 3 6 3$   $\mathbf { 0 . 1 3 1 5 \ : \pm { \ : 0 . 0 4 4 7 } }$ </td></tr><tr><td rowspan="2">U2Net</td><td>WV3→WV2</td><td>Cold-start Rolling</td><td> $0 . 9 2 0 0 \pm 0 . 0 4 6 0$   $\mathbf { 0 . 9 2 4 9 \pm 0 . 0 4 0 8 }$ </td><td> $0 . 0 5 8 3 \pm 0 . 0 4 4 3$   $\mathbf { 0 . 0 5 4 2 \pm 0 . 0 4 1 5 }$ </td><td> $0 . 0 2 3 2 \pm 0 . 0 0 5 1$   $\mathbf { 0 . 0 2 2 1 \pm 0 . 0 0 4 5 }$ </td><td> $0 . 2 4 0 8 \pm 0 . 0 3 8 1$   $\mathbf { 0 . 2 3 2 7 \ : \pm 0 . 0 1 9 5 }$ </td></tr><tr><td>QB→GF2</td><td>Cold-start Rolling</td><td> $0 . 8 0 5 8 \pm 0 . 0 4 7 8$   $\mathbf { 0 . 8 1 7 4 \ : \pm 0 . 0 3 4 9 }$ </td><td> $0 . 1 0 8 7 \pm 0 . 0 4 0 2$   $\mathbf { 0 . 0 9 7 3 \pm 0 . 0 3 0 5 }$ </td><td> $0 . 0 9 5 9 \pm 0 . 0 3 6 7$   $\mathbf { 0 . 0 9 4 5 \pm 0 . 0 2 3 7 }$ </td><td> $0 . 2 3 2 6 \pm 0 . 0 5 7 4$   $\mathbf { 0 . 2 1 9 4 \ : \pm 0 . 0 2 8 3 }$ </td></tr></table>

Table 10  
Cosine similarity between target images. $\mathsf { M e a n } \pm \mathsf { s t d }$ over 190 image pairs per transfer (380 pooled) with independently cold-started Tailors; higher in bold.
<table><tr><td>Transfer</td><td></td><td>Latent similarity Residual similarity</td></tr><tr><td>WV3→WV2</td><td> $0 . 6 7 8 2 \pm 0 . 2 9 9 5$ </td><td> $\mathbf { 0 . 8 5 3 7 \pm 0 . 1 1 0 7 }$ </td></tr><tr><td>QB→GF2</td><td> $0 . 8 6 8 4 \pm 0 . 1 0 8 9$ </td><td> $\mathbf { 0 . 9 3 8 3 \pm 0 . 0 4 2 7 }$ </td></tr><tr><td>Aggregated</td><td> $0 . 7 7 3 3 \pm 0 . 2 4 4 6$ </td><td> $\mathbf { 0 . 8 9 6 0 \pm 0 . 0 9 4 0 }$ </td></tr></table>

Seed-image resampling. Twenty sequences per transfer, each with a diferent test image as seed and the others randomly permuted; the backbone is reloaded for each sequence.  
Table 11
<table><tr><td>Transfer</td><td>Mean HQNR↑</td><td>Var.  $( 1 0 ^ { - 6 } )$ </td><td>Std</td><td>Min-Max</td></tr><tr><td> $\mathsf { Q B } \to \mathsf { G F } 2$ </td><td>0.8999</td><td>7.50</td><td>0.0027</td><td>0.8959-0.9072</td></tr><tr><td> ${ \mathsf { W V } } 3 {  } { \mathsf { W V } } 2$ </td><td>0.9278</td><td>2.60</td><td>0.0016</td><td>0.9255-0.9312</td></tr></table>

## 5.2.2. Cold start versus rolling start

Both arms use Random-8, the same 10-epoch budget, and freshly created Adam optimizers; only the initial Tailor parameters difer (zero-output for cold start, the previous image’s for rolling).

In Table 9, rolling improves HQNR and $D _ { \lambda }$ in all four settings and $D _ { s }$ in three, with no increase in adaptation time. The gain comes purely from the inherited sensor-related knowledge.

## 5.2.3. Robustness to the seed image

Since $\phi ^ { 0 }$ is trained on the seed image alone, we test whether its choice matters: for each transfer, twenty sequences let every test image serve once as the seed image, with the remaining images randomly permuted and the pretrained backbone reloaded per sequence. Table 11 shows negligible variation across sequences, so results do not hinge on the seed image, consistent with the shared correction measured above.

## 5.2.4. Stability over long sequences

Deployment streams can be much longer than the twentyimage test stream, so we extend the sequence to 100 images by resampling the target images and run ReTailor end to end. Table 12 shows that all metrics are essentially unchanged from the 20-image stream, indicating that the inherited correction does not drift over long sequences.

## 6. Discussion

## 6.1. Why ReTailor Works

ReTailor inserts a lightweight learnable module into a frozen pretrained fusion network, preserves the knowledge learned for the source sensor, and repairs the cross-sensor shift using only the most valuable patches; passing the learned correction from one image to the next further refines the result. Together, these components yield eficient, fast, and high-quality cross-sensor correction. Its efectiveness rests on three reasons.

Evidence within an image. The information needed to diagnose the sensor shift is already contained in the target PAN and LRMS observations, and a scoring rule that combines patch-level measures with the regional distribution can identify the most valuable patches quickly. These highvalue patches provide suficient evidence to support fast optimization for the new sensor, which improves ReTailor’s adaptation quality at negligible cost.

Table 12  
Long-sequence stability (FusionNet). The standard 20-image stream versus 100-image sequences resampled from the same images.
<table><tr><td>Transfer</td><td>Length</td><td>HQNR ↑</td><td> $D _ { \lambda } \downarrow$ </td><td> $D _ { s } \downarrow$ </td></tr><tr><td rowspan="2">QB→GF2</td><td>20</td><td> $0 . 8 9 7 6 \pm 0 . 0 2 6 4$ </td><td> $0 . 0 3 3 9 \pm 0 . 0 2 9 2$ </td><td> $0 . 0 7 0 6 \pm 0 . 0 2 1 8$ </td></tr><tr><td>100</td><td> $0 . 8 9 7 8 \pm 0 . 0 2 5 3$ </td><td> $0 . 0 3 0 5 \pm 0 . 0 2 7 9$ </td><td> $0 . 0 7 3 6 \pm 0 . 0 2 2 6$ </td></tr><tr><td rowspan="2"> ${ \mathsf { W V } } 3 {  } { \mathsf { W V } } 2$ </td><td>20</td><td> $0 . 9 2 4 5 \pm 0 . 0 3 9 4$ </td><td> $0 . 0 3 3 5 \pm 0 . 0 1 9 8$ </td><td> $0 . 0 4 3 8 \pm 0 . 0 2 3 3$ </td></tr><tr><td>100</td><td> $0 . 9 2 3 8 \pm 0 . 0 3 9 1$ </td><td> $0 . 0 3 2 8 \pm 0 . 0 2 1 5$ </td><td> $0 . 0 4 5 2 \pm 0 . 0 2 2 2$ </td></tr></table>

Correction at the fusion interface. The Feature Tailor performs a concentrated, rapid correction at the interface between the frozen Feature Extractor and the frozen Channel Mapper, where a small additive residual sufices to absorb the sensor-specific component of the fused representation. This preserves the basic fusion information established during prior training while repairing the cross-sensor shift, and it corrects more eficiently than a parameter-matched LoRA, enabling fast transfer.

Inheritance across images. For the same cross-sensor transfer task, a considerable part of the required correction is shared across images, since the sensor response remains stable within a stream. By inheriting what the preceding images have already learned and then rapidly learning only the image-specific remainder on each new image, every image becomes a refinement of a common correction rather than an independent restart, which makes cross-sensor transfer work better along the stream.

## 6.2. Beyond Pansharpening

Inserting a concentrated correction at a single interface of a frozen network is not specific to pansharpening: the same Feature Tailor improves an implicit-representation pansharpening network under observation-driven losses and a supervised building-segmentation network under labeled losses, with the original models kept frozen. Cross-sensor and cross-domain shifts are pervasive in remote sensing, so this single-interface correction is also worth considering for tasks beyond pansharpening where a pretrained model must be adapted cheaply.

## 6.3. Applicability and Future Directions

ReTailor targets streams from a stable target sensor with fixed MTF priors, a common case in operational pipelines, where the inherited Tailor gives each image a reusable starting point to refine. The current design still learns one Tailor per sensor pair, so a new transfer direction must restart the bootstrap. A natural next step is one Tailor that corrects the degradations of multiple sensors, sharing the one-time adaptation cost across sensors.

## 7. Conclusion

We present ReTailor, a real-time sequential framework for cross-sensor pansharpening that adapts frozen backbones without target ground truth. A lightweight residual Feature Tailor corrects cross-sensor shifts while preserving the pretrained fusion prior, content-aware patch selection supplies informative adaptation patches at negligible cost, and rolling residual bootstrap carries the sensor-related knowledge along the stream to improve quality. On QB→GF2 and ${ \mathrm { W V } } 3 {  } \mathrm { W V } 2$ , ReTailor reaches HQNR of 0.8976 and 0.9245 with a fraction of a second of adaptation per image, and at matched parameter counts it roughly halves the latency of LoRA while lowering spectral distortion. Systematic ablations confirm the efectiveness of each component. The core principle of ReTailor also transfers to super-resolution pansharpening and semantic segmentation. These results make ReTailor a practical choice for cross-sensor pansharpening, while providing useful insights for other cross-sensor remote-sensing applications.

## Acknowledgment

This work was supported by the Project of the Department of Science and Technology of Sichuan Province under Grant 2025YFNH0001.

CRediT authorship contribution statement. Jing-Ze Liang: Conceptualization, Methodology, Software, Validation, Investigation, Writing – original draft, Writing – review & editing. Wei-Jie Xiao: Methodology, Validation, Investigation, Writing – review & editing. Zi-Han Cao: Validation, Investigation, Visualization, Writing – review & editing. Yang-Dong Li: Investigation, Data curation, Writing – review & editing. Tian-Yi Hu: Investigation, Data curation, Writing – review & editing. Liang-Jian Deng: Supervision, Project administration, Writing – review & editing.

Declaration of competing interest. The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## A. Implementation Details

## A.1. Pansharpening Protocol

All Feature Tailors use the three-layer residual topology of Section 3.2; Table 14 lists their interfaces and sizes, and Table 13 lists the optimization settings. The LeakyReLU slope is 0.2 for G-ZAP and 0.01 otherwise. Within each comparison only the studied factor changes, and timing is compared only between arms run on the same GPU (RTX 4090).

Each target image is a 512 × 512 PAN image with its 128 × 128 LRMS counterpart, partitioned into an 8 × 8 grid of $N _ { c } = 6 4$ non-overlapping 64 × 64 candidate patches; a four-pixel context rim is used at inference. Batch sizes are 8 for adaptation and 32 for inference. Random selection and test-image sampling use a fixed seed. Random selection uses

Optimization settings. Epochs are per target image; each epoch takes ⌈�<sup>�</sup>∕8⌉ steps (later images), 8 steps (seed image), or 1 step (LoRA). LR: learning rate; WD: weight decay.
<table><tr><td>Study</td><td>Schedule</td><td>Optimizer</td><td>LR</td><td>WD</td></tr><tr><td>FusionNet/U2Net</td><td>30 epochs (seed image)</td><td>Adam</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td></td><td>10 epochs (later images)</td><td>Adam</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>LoRA</td><td>10 epochs</td><td>Adam</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>G-ZAP</td><td>500 epochs</td><td>Adam</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>10-5</td></tr><tr><td>CrossEarth A2S</td><td>5,000 iterations</td><td>AdamW</td><td> $1 0 ^ { - 4 }$ </td><td>0.05</td></tr></table>

$M = 8 ,$ and content-aware selection uses a $2 \times 2$ region grid with $\lambda _ { c } = 0 . 2 , \lambda _ { r } = 0 . 1$ , and $\tau = 0 . 5 5$

For both backbones, ${ \cal D } _ { \mathrm { M S } } ^ { ( b ) }$ applies the MS MTF kernel followed by fourfold nearest-neighbor downsampling. The MTF kernels use the per-sensor GNyq values of the standard pansharpening toolbox [58]. The evaluation protocol uses the same MTF convention [59, 60]. Each PAN patch is radiometrically aligned to its LRMS range and broadcast over the spectral channels before the spatial loss, with weights $\eta _ { \mathrm { s p a } } = \eta _ { \mathrm { s p e } } = 1 \ \mathrm { a n d } \ \eta _ { \mathrm { o r i } } = 0 . 1$

## A.2. LoRA Baseline

Low-rank branches are attached to the eligible convolutional layers of each backbone (10 for FusionNet, 19 for U2Net); depthwise and transposed convolutions and nonconvolutional modules are left unchanged. The rank is $r =$ $\lceil P _ { T } / p _ { 1 } \rceil$ , where $P _ { T }$ is the Feature Tailor parameter count and $p _ { 1 }$ the parameter count of a rank-1 adapter over the injected layers, with scaling factor � = �. This gives � = 75 and 76 for FusionNet and � = 32 and 33 for U2Net on WV3→WV2 and QB→GF2, and trainable parameters within 0.2%–2.4% of the Tailor size (Table 2).

## A.3. Cross-Task Settings

Both tasks apply the residual correction of Eq. (1) at an intermediate interface of a frozen network (Table 14) and train only the Tailor, from unlabeled PAN/LRMS observations for G-ZAP and from segmentation labels for CrossEarth.

For G-ZAP, each transfer starts from the source-sensor checkpoint of the original protocol [55]. The Tailor is adapted from target PAN/LRMS observations and their multiscale degradations under the task’s semi-supervised losses, and is inherited along the image sequence. The 500-epoch budget follows the original protocol.

CrossEarth follows the Aerial-to-Satellite (A2S) protocol [56, 57]. Both trained variants start from the same Aerial checkpoint, are trained on the labeled Satellite training set with the same schedule and nearly equal trainable parameters, and are evaluated on the Satellite test set. Head-only updates the original segmentation head, contrasting feature-space residual correction with conventional head adaptation.

Table 14  
Feature Tailor configurations. Insertion point, feature width �, and trainable parameters for each model.
<table><tr><td>Model/task</td><td>Insertion point</td><td>Width d</td><td>Params</td></tr><tr><td>FusionNet</td><td>res4 → conv3</td><td>32</td><td>221,472</td></tr><tr><td>U2Net</td><td>s2block4 → to_hrms</td><td>32</td><td>221,472</td></tr><tr><td>G-ZAP</td><td>INR query → decoder</td><td>48</td><td>498,096</td></tr><tr><td>CrossEarth A2S pixel decoder →</td><td>Mask2Former head [61]</td><td>256</td><td>14,158,080</td></tr></table>

## B. Design Choices

## B.1. Backbone Selection

We use FusionNet and U2Net as frozen backbones because they are established yet structurally diferent, a residual detail-injection CNN [15] and a spatial–spectral double-U-Net [48], so consistent gains on both reflect the adaptation strategy rather than a single architecture. Both also split into a Feature Extractor and a Channel Mapper, which provides the interface for the Tailor (Table 14).

## B.2. Feature Tailor Insertion Position

As defined in Section 3.2, the Tailor sits between $F _ { \mathrm { e x t } }$ and $F _ { \mathrm { m a p } } .$ , where the fused feature � combines spectral and spatial information from both modalities but is not yet tied to the HRMS space. This agrees with findings that intermediate features transfer better than shallow or outputlevel ones [62, 63]. In Fig. 10, inserting the Tailor before the Channel Mapper (pre-CM) gives the highest HQNR on WV3 for both backbones, above both the no-Tailor baseline and insertion after the Channel Mapper (post-CM). Correcting the fused feature is therefore more efective than correcting the output.

![](images/55328e54cde0b19da2261a2afa179c365c4abcbc5063a93a0f56b26db70200c5.jpg)  
Figure 10: HQNR for Tailor placements on WV3. Baseline: no Tailor; pre-CM/post-CM: before/after the Channel Mapper.

## C. Sensitivity to Patch Budget and Patch Size

We varied the number of selected patches from 1 to 32 on WV3→WV2. In Fig. 11, HQNR rises markedly up to eight patches for both backbones, while 16 patches add only a modest gain at twice the workload and 32 add almost nothing. We therefore use eight patches as the reference budget.

We also varied the PAN patch size from $8 \times 8$ up to the largest size that fits in GPU memory. In Fig. 12, HQNR rises sharply from $8 \times 8$ to $3 2 \times 3 2 ;$ at 64 × 64, FusionNet is near its best value and U2Net stays close to its $3 2 \times 3 2$ value. We therefore use $6 4 \times 6 4$ , which gives suficient context at moderate cost.

## D. Complexity and Eficiency of Selection and Tailor Updates

## D.1. Asymptotic Complexity

For each later image, selection operates directly on the aligned observations: the two variation cues are computed in parallel for all $N _ { c }$ candidates, followed by at most $M ^ { t }$ greedy rounds. Its complexity is

$$
\mathcal { O } \left( N _ { c } h w + N _ { c } C _ { \mathrm { M S } } h _ { s } w _ { s } + M ^ { t } N _ { c } \right) ,
$$

where $h \times w$ and $h _ { s } \times w _ { s }$ are the PAN and LRMS patch sizes and $C _ { \mathrm { M S } }$ is the number of LRMS bands. No backbone forward pass, Tailor update, or pairwise candidate-relation matrix as in [43, 44] is required.

The number of selected patches determines the Tailorupdate cost. For a PAN input of size $H \times W$ with $N _ { c }$ patches of size $h \times w \left( H W = N _ { c } h w \right)$ and � adaptation epochs,

![](images/29a610d3426924d892f8ac5e30cbe70b4a0e65416709e895ed93a1a5fe898c4a.jpg)  
Figure 11: HQNR versus number of selected patches (WV3→WV2).

![](images/4621a0193c3168710fb515980822c7da48f00e60595e1159d6e4351a06fcdbfe.jpg)  
Figure 12: HQNR versus PAN patch size (WV3→WV2). U2Net is limited to 64 × 64 by GPU memory.

Complexity of Tailor updates. (a) Update complexity; (b) compression ratios of ReTailor (attention assumes global selfattention).
<table><tr><td>(a)</td><td>Whole image</td><td>Fixed (M)</td><td>ReTailor (Mt)</td></tr><tr><td>CNN</td><td> $\mathcal { O } ( E H W )$ </td><td> $\mathcal { O } ( E M h w )$ </td><td> $\mathcal { O } ( E M ^ { t } h w )$ </td></tr><tr><td>Attention</td><td> $\mathcal { O } \big ( E ( H W ) ^ { 2 } \big )$ </td><td> $\mathcal { O } \left( E M ( h w ) ^ { 2 } \right)$ </td><td> $\mathcal { O } \big ( E M ^ { t } ( h w ) ^ { 2 } \big )$ </td></tr><tr><td>(b)</td><td></td><td>Whole/ReTailor</td><td>Fixed/ReTailor</td></tr><tr><td rowspan="2">CNN</td><td></td><td> $N _ { c }$ </td><td>M</td></tr><tr><td></td><td> $\overline { { { M _ { \small { c } } ^ { t } } } }$ </td><td>Mt</td></tr><tr><td rowspan="2">Attention</td><td></td><td></td><td>M</td></tr><tr><td></td><td> $\overline { { M ^ { t } } }$ </td><td>Mt</td></tr></table>

Selector latency on QB→GF2. Time of a single selection, mean ± std over images.
<table><tr><td>Candidate evaluation</td><td>Latency (ms) ↓</td><td>Speedup ↑</td></tr><tr><td>Sequential</td><td> $5 . 2 8 \pm 0 . 8 4$ </td><td>1.00×</td></tr><tr><td>Parallel</td><td> ${ \bf 1 . 9 9 \pm 0 . 2 8 }$ </td><td>2.65×</td></tr></table>

Table 15 compares whole-image adaptation, a fixed budget of � patches, and ReTailor with the image-dependent count $M ^ { t }$

Selecting on average 6.05 and 6.47 of 64 candidates reduces the Tailor-update workload by 90.5% and 89.9% relative to whole-image adaptation on WV2 and GF2, and the measured adaptation time by 9.9%–17.1% relative to fixed-8 adaptation, including selection overhead.

## D.2. Eficiency of the Tensorized Greedy Scan

We compared the tensorized selector with a functionally equivalent Python-loop variant that uses identical scores, regions, and stopping rule but evaluates candidates sequentially within each round. On the post-seed QB→GF2 images (64 candidates each, RTX 4090), each selector was run 20 times after warm-up per image. Table 16 shows that tensorization reduces the latency of a single selection from $5 . 2 8 \pm 0 . 8 4$ ms to $1 . 9 9 { \pm } 0 . 2 8 $ ms (62.3% lower, 2.65× faster) while producing identical selections.

## References

[1] L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, A. Plaza, Machine learning in pansharpening: A benchmark, from shallow to deep networks, IEEE Geoscience and Remote Sensing Magazine 10 (3) (2022) 279–315.

[2] F. Dadrass Javan, F. Samadzadegan, S. Mehravar, A. Toosi, R. Khatami, A. Stein, A review of image fusion techniques for pan-sharpening of high-resolution satellite imagery, ISPRS Journal of Photogrammetry and Remote Sensing 171 (2021) 101–117.

[3] J. Duran, A. Buades, B. Coll, C. Sbert, G. Blanchet, A survey of pansharpening methods with a new band-decoupled variational model, ISPRS Journal of Photogrammetry and Remote Sensing 125 (2017) 78–105.

[4] K. Amolins, Y. Zhang, P. Dare, Wavelet based image fusion techniques — an introduction, review and comparison, ISPRS Journal of Photogrammetry and Remote Sensing 62 (2007) 249–263.

[5] T.-M. Tu, S.-C. Su, H.-C. Shyu, P. S. Huang, A new look at IHS-like image fusion methods, Information Fusion 2 (3) (2001) 177–186.

[6] T.-M. Tu, P. S. Huang, C.-L. Hung, C.-P. Chang, A fast intensity-hue-saturation fusion technique with spectral adjustment for IKONOS imagery, IEEE Geoscience and Remote Sensing Letters 1 (4) (2004) 309–312.

[7] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, M. Selva, MTF-tailored multiscale fusion of high-resolution MS and Pan imagery, Photogrammetric Engineering & Remote Sensing 72 (5) (2006) 591–596.

[8] J. Cheng, H. Liu, T. Liu, F. Wang, H. Li, Remote sensing image fusion via wavelet transform and sparse representation, ISPRS Journal of Photogrammetry and Remote Sensing 104 (2015) 158–173.

[9] C. Ballester, V. Caselles, L. Igual, J. Verdera, B. Rougé, A variational model for P+XS image fusion, International Journal of Computer Vision 69 (2006) 43–58.

[10] F. Fang, F. Li, C. Shen, G. Zhang, A variational approach for pan-sharpening, IEEE Transactions on Image Processing 22 (7) (2013) 2822–2834.

[11] F. Palsson, J. R. Sveinsson, M. O. Ulfarsson, A new pansharpening algorithm based on total variation, IEEE Geoscience and Remote Sensing Letters 11 (1) (2014) 318–322.

[12] R. Gogineni, A. Chaturvedi, Sparsity inspired pan-sharpening technique using multi-scale learned dictionary, ISPRS Journal of Photogrammetry and Remote Sensing 146 (2018) 360–372.

[13] G. Masi, D. Cozzolino, L. Verdoliva, G. Scarpa, Pansharpening by convolutional neural networks, Remote Sensing 8 (7) (2016) 594.

[14] G. Scarpa, S. Vitale, D. Cozzolino, Target-adaptive CNNbased pansharpening, IEEE Transactions on Geoscience and Remote Sensing 56 (9) (2018) 5443–5457.

[15] L.-J. Deng, G. Vivone, C. Jin, J. Chanussot, Detail injectionbased deep convolutional neural networks for pansharpening, IEEE Transactions on Geoscience and Remote Sensing 59 (8) (2021) 6995–7010.

[16] S. Xu, J. Zhang, Z. Zhao, K. Sun, J. Liu, C. Zhang, Deep gradient projection networks for pan-sharpening, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 1366–1375.

[17] H. Zhou, Q. Liu, Y. Wang, PanFormer: A transformer based model for pan-sharpening, in: IEEE International Conference on Multimedia and Expo (ICME), 2022, pp. 1–6.

[18] Q. Meng, W. Shi, S. Li, L. Zhang, PanDif: A novel pansharpening method based on denoising difusion probabilistic model, IEEE Transactions on Geoscience and Remote Sensing 61 (2023) 1–17.

[19] X. He, K. Cao, J. Zhang, K. Yan, Y. Wang, R. Li, C. Xie, D. Hong, M. Zhou, Pan-Mamba: Efective pan-sharpening with state space model, Information Fusion 115 (2025) 102779.

[20] Q. Cao, L.-J. Deng, W. Wang, J. Hou, G. Vivone, Zeroshot semi-supervised learning for pansharpening, Information Fusion 101 (2024) 102001.

[21] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models, in: International Conference on Learning Representations (ICLR), 2022.

[22] X. Rui, X. Cao, Y. Li, D. Meng, Variational zero-shot multispectral pansharpening, IEEE Transactions on Geoscience and Remote Sensing 62 (2024) 1–16.

[23] I. Amro, J. Mateos, M. Vega, R. Molina, A. K. Katsaggelos, A survey of classical methods and new trends in pansharpening of multispectral images, EURASIP Journal on Advances in Signal Processing 2011 (2011) 79.

[24] C. S. Yilmaz, V. Yilmaz, O. Gungor, A theoretical and practical survey of image fusion methods for multispectral pansharpening, Information Fusion 79 (2022) 1–43.

[25] Y. Xing, M. Wang, S. Yang, L. Jiao, Pan-sharpening via deep metric learning, ISPRS Journal of Photogrammetry and Remote Sensing 145 (2018) 165–183.

[26] M. Jiang, H. Shen, J. Li, Q. Yuan, L. Zhang, A diferential information residual convolutional neural network for pansharpening, ISPRS Journal of Photogrammetry and Remote Sensing 163 (2020) 257–271.

[27] H. Zhang, J. Ma, GTP-PNet: A residual learning network based on gradient transformation prior for pansharpening, ISPRS Journal of Photogrammetry and Remote Sensing 172 (2021) 223–239.

[28] J. He, Q. Yuan, J. Li, Y. Xiao, L. Zhang, A self-supervised remote sensing image fusion framework with dual-stage selflearning and spectral super-resolution injection, ISPRS Journal of Photogrammetry and Remote Sensing 204 (2023) 131–144.

[29] J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley, PanNet: A deep network architecture for pan-sharpening, in: IEEE International Conference on Computer Vision (ICCV), 2017, pp. 1753–1761.

[30] Y. Cui, P. Liu, Y. Ma, L. Chen, M. Xu, X. Guo, Pansharpening via predictive filtering with element-wise feature mixing, ISPRS Journal of Photogrammetry and Remote Sensing 219 (2025) 22–37.

[31] Y. Han, H. Chi, J. Huang, X. Gao, Z. Zhang, F. Ling, TemPanSharpening: A multi-temporal pansharpening solution based on deep learning and edge extraction, ISPRS Journal of Photogrammetry and Remote Sensing 211 (2024) 406–424.

[32] M. Volpi, G. Camps-Valls, D. Tuia, Spectral alignment of multi-temporal cross-sensor images with automated kernel canonical correlation analysis, ISPRS Journal of Photogrammetry and Remote Sensing 107 (2015) 50–63.

[33] M. Luo, S. Ji, Cross-spatiotemporal land-cover classification from VHR remote sensing images with deep learning based domain adaptation, ISPRS Journal of Photogrammetry and Remote Sensing 191 (2022) 105–128.

[34] Z. Qiu, H. Shen, L. Yue, G. Zheng, Cross-sensor remote sensing imagery super-resolution via an edge-guided attentionbased network, ISPRS Journal of Photogrammetry and Remote Sensing 199 (2023) 226–241.

[35] T. Xin, J.-L. Xiao, Z. Xia, S. Yin, L.-J. Deng, Training and inference within 1 second – tackle cross-sensor degradation of real-world pansharpening with eficient residual feature tailoring, in: AAAI Conference on Artificial Intelligence (AAAI), Vol. 40, 2026, pp. 11141–11149.

[36] K. Shen, X. Yang, S. Lolli, G. Vivone, A continual learningguided training framework for pansharpening, ISPRS Journal of Photogrammetry and Remote Sensing 196 (2023) 45–57.

[37] Z. Xia, C. Sun, T. Xin, Y. Zeng, H. Chen, L.-J. Deng, SWIFT: A general sensitive weight identification framework for fast sensor-transfer pansharpening, in: AAAI Conference on Artificial Intelligence (AAAI), Vol. 40, 2026, pp. 10960– 10968.

[38] L. Wald, T. Ranchin, M. Mangolini, Fusion of satellite images of diferent spatial resolutions: Assessing the quality of

resulting images, Photogrammetric Engineering & Remote Sensing 63 (6) (1997) 691–699.

[39] B. Aiazzi, S. Baronti, M. Selva, Improving component substitution pansharpening through multivariate regression of MS+Pan data, IEEE Transactions on Geoscience and Remote Sensing 45 (10) (2007) 3230–3239.

[40] X. Otazu, M. González-Audícana, O. Fors, J. Núñez, Introduction of sensor spectral response into image fusion methods. application to wavelet-based methods, IEEE Transactions on Geoscience and Remote Sensing 43 (10) (2005) 2376–2385.

[41] J. Ma, W. Yu, C. Chen, P. Liang, X. Guo, J. Jiang, Pan-GAN: An unsupervised pan-sharpening method for remote sensing image fusion, Information Fusion 62 (2020) 110–120.

[42] G. Vivone, R. Restaino, M. Dalla Mura, G. Licciardi, J. Chanussot, Contrast and error-based fusion schemes for multispectral image pansharpening, IEEE Geoscience and Remote Sensing Letters 11 (5) (2014) 930–934.

[43] M. Paul, S. Ganguli, G. K. Dziugaite, Deep learning on a data diet: Finding important examples early in training, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 34, 2021, pp. 20596–20607.

[44] K. Killamsetty, D. Sivasubramanian, G. Ramakrishnan, R. Iyer, GLISTER: Generalization based data subset selection for eficient and robust learning, in: AAAI Conference on Artificial Intelligence (AAAI), Vol. 35, 2021, pp. 8110–8118.

[45] B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis, IEEE Transactions on Geoscience and Remote Sensing 40 (10) (2002) 2300–2312.

[46] L. Alparone, B. Aiazzi, S. Baronti, A. Garzelli, F. Nencini, M. Selva, Multispectral and panchromatic data fusion assessment without reference, Photogrammetric Engineering & Remote Sensing 74 (2) (2008) 193–200.

[47] G. Vivone, Robust band-dependent spatial-detail approaches for panchromatic sharpening, IEEE Transactions on Geoscience and Remote Sensing 57 (9) (2019) 6421–6433.

[48] S. Peng, C. Guo, X. Wu, L.-J. Deng, U2Net: A general framework with spatial-spectral-integrated double U-Net for image fusion, in: 31st ACM International Conference on Multimedia (ACM MM), 2023, pp. 3219–3227.

[49] S. Lolli, L. Alparone, A. Garzelli, G. Vivone, Haze correction for contrast-based multispectral pansharpening, IEEE Geoscience and Remote Sensing Letters 14 (12) (2017) 2255– 2259.

[50] A. Garzelli, Eficient MMSE pansharpening based on nonlocal optimization, in: IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2014, pp. 195–198.

[51] R. Restaino, G. Vivone, M. Dalla Mura, J. Chanussot, Fusion of multispectral and panchromatic images based on morphological operators, IEEE Transactions on Image Processing 25 (6) (2016) 2882–2895.

[52] J. Huang, R. Huang, J. Xu, S. Peng, Y. Duan, L.-J. Deng, Wavelet-assisted multi-frequency attention network for pansharpening, in: AAAI Conference on Artificial Intelligence (AAAI), Vol. 39, 2025, pp. 3662–3670.

[53] S. Peng, X. Zhu, H. Deng, L.-J. Deng, Z. Lei, FusionMamba: Eficient remote sensing image fusion with state space model, IEEE Transactions on Geoscience and Remote Sensing 62 (2024) 1–16.

[54] B. Aiazzi, L. Alparone, S. Baronti, R. Carlà, A. Garzelli, L. Santurri, Full-scale assessment of pansharpening methods and data products, in: Image and Signal Processing for Remote Sensing XX, Vol. 9244, 2014, p. 924402.

[55] Z. Yang, S. Yin, J. Liang, L.-J. Deng, G-ZAP: A generalizable zero-shot framework for arbitrary-scale pansharpening, in: European Conference on Computer Vision (ECCV), 2026, pp. 57–73.

[56] Z. Gong, Z. Wei, D. Wang, X. Hu, X. Ma, H. Chen, Y. Jia, Y. Deng, Z. Ji, X. Zhu, X. Yang, N. Yokoya, J. Zhang, B. Du, J. Yan, L. Zhang, CrossEarth: Geospatial vision foundation model for domain generalizable remote sensing semantic segmentation, IEEE Transactions on Pattern Analysis and Machine Intelligence 48 (5) (2026) 5147–5164.

[57] S. Ji, S. Wei, M. Lu, Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set, IEEE Transactions on Geoscience and Remote Sensing 57 (1) (2019) 574–586.

[58] G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, L. Wald, A critical comparison among pansharpening algorithms, IEEE Transactions on Geoscience and Remote Sensing 53 (5) (2015) 2565–2586.

[59] G. Vivone, M. Dalla Mura, A. Garzelli, F. Pacifici, A benchmarking protocol for pansharpening: Dataset, preprocessing, and quality assessment, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 14 (2021) 6102–6118.

[60] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, J. Chanussot, A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods, IEEE Geoscience and Remote Sensing Magazine 9 (1) (2021) 53–81.

[61] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, R. Girdhar, Masked-attention mask transformer for universal image segmentation, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 1280–1289.

[62] J. Yosinski, J. Clune, Y. Bengio, H. Lipson, How transferable are features in deep neural networks?, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 27, 2014, pp. 3320–3328.

[63] Q. Dou, D. Coelho de Castro, K. Kamnitsas, B. Glocker, Domain generalization via model-agnostic learning of semantic features, in: Advances in Neural Information Processing Systems (NeurIPS), Vol. 32, 2019, pp. 6447–6458.