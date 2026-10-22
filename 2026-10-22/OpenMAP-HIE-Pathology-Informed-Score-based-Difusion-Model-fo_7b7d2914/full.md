# OpenMAP-HIE: Pathology Informed Score-based Difusion Model for Unsupervised Neonatal Hypoxic-Ischemic Encephalopathy Lesion Segmentation

Pengyu Kan<sup>a</sup>, Craig Jones<sup>a,b</sup> and Kenichi Oishi<sup>c,d,∗</sup>

<sup>a</sup>Department ofComputer Science, Johns Hopkins University, Baltimore , MD, United States

<sup>b</sup>Department of Biomedical Engineering, University of Alberta, Edmonton , AB, Canada

<sup>c</sup>Department of Radiology, Johns Hopkins University School of Medicine, Baltimore , MD, United States

<sup>d</sup>Department of Neurology, Johns Hopkins University School of Medicine, Baltimore , MD, United States

## A R T I C L E I N F O

Keywords:   
Hypoxic–ischemic encephalopathy   
Neonatal brain MRI   
Unsupervised anomaly detection   
Difusion models   
Pathology informed modeling   
Score-based modeling

## A BS T RA C T

Accurate localization of neonatal hypoxic–ischemic encephalopathy (HIE) lesions on apparent difusion coeficient (ADC) images is clinically important but is limited by scarce expert annotations and substantial variation in lesion appearance. We propose OpenMAP-HIE, a pathology-informed scorebased difusion model for unsupervised HIE lesion localization without voxel-wise lesion supervision. A denoising difusion model learns the healthy neonatal ADC distribution, while a ControlNet-based conditional branch preserves patient-specific anatomy. At inference, anomaly evidence is derived directly from diferences between conditional and unconditioned noise predictions at independently evaluated difusion noise levels, avoiding dependence on a final pseudo-healthy reconstruction or sequential reverse-difusion chain. Based on the characteristic ADC reduction associated with acute HIE, only discrepancies consistent with the expected pathological direction are retained and aggregated across noise levels using a geometric mean. Spatial mean filtering further suppresses isolated responses. The framework was evaluated on the BONBID-HIE dataset. The ADC-only method achieved a Dice score of 0.283 ± 0.244, average precision of 0.505, Hausdorf distance of 25.45±15.63 mm, mean average surface distance of 9.40±8.49 mm, and normalized surface distance of 0.355 ± 0.236, providing the best overlap and surface-distance performance among the evaluated unsupervised methods and Dice performance comparable to supervised ADC-only models. Optional population-derived ADC �-map refinement further increased Dice to 0.602 ± 0.269. These findings support pathology-informed score discrepancies as a useful signal for neonatal HIE lesion localization, while highlighting remaining challenges in detecting small and low-contrast lesions.

## 1. Introduction

Hypoxic ischemic encephalopathy (HIE) is a clinical syndrome of neurological disorder caused by impaired cerebral oxygen delivery and blood flow around the time of birth. It afects approximately 1–5 per 1,000 term-born neonates (Long and Brandon, 2007). Therapeutic hypothermia reduces the risk of death and disability in infants with moderate-to-severe HIE (Silveira and Procianoy, 2015). However, mortality and adverse long-term outcomes, including neurodevelopmental delay and cognitive or motor impairment, still afect approximately 30%–50% of afected infants (Bao et al., 2025). Accurate identification and localization of HIE-related brain injury are therefore important for quantitatively characterizing lesion severity and for understanding disease progression (Vermeulen et al., 2008; Martinez-Biarge et al., 2011; Chang et al., 2020; Bobba et al., 2023).

Difusion-weighted imaging (DWI) has been applied for detecting acute HIE-related injury, because abnormalities may appear earlier and more observable than signal changes on conventional T1- or T2-weighted MRI (Triulzi et al.,

2006; Bano et al., 2017). Cytotoxic edema restricts water difusion, typically producing increased signal intensity on DWI and decreased apparent difusion coeficient (ADC) values (Onda et al., 2024). ADC maps thus provide quantitative information for identifying HIE-related injury and assessing its severity and prognosis (Bobba et al., 2023; Bao et al., 2025).

Automated localization remains challenging because HIE lesions are often small, multifocal, difuse, and heterogeneous in appearance. In the BONBID-HIE cohort, more than half of afected neonates had lesions occupying less than 1% of the brain volume (Bao et al., 2025), resulting in severe foreground–background imbalance and poorly defined lesion boundaries. Scanner field strength, acquisition protocol, developmental anatomy, and postnatal imaging time further complicate the distinction between pathology and normal neonatal variability. These characteristics also make manual delineation dificult. Quantitative lesion analysis typically requires experienced pediatric neuroradiologists, often using multi-reader consensus (Bao et al., 2025), but lesion boundaries may remain ambiguous and dependent on reader judgment. Consequently, expert voxellevel annotation is labor intensive and dificult to scale.

Supervised deep-learning approaches have achieved encouraging performance for neonatal HIE lesion segmentation (Bao et al., 2025; Rao et al., 2025; Almansour et al.,

![](images/6edabfadbff5bc5adf864b26bf3f251e0589d406b1db72abf6fe10e5a7260ed3.jpg)  
Figure 1: Architecture of the anatomy-conditioned difusion model. The healthy-prior U-Net is pretrained in stage 1 and frozen during stage 2. A trainable ControlNet encoder, initialized from the U-Net encoder, incorporates the spatial condition � alongside the noisy input � and timestep �. Multi-scale ControlNet features are passed through zero-initialized 1 × 1 convolutions and added to the U-Net skip features, enabling subject-specific conditioning of the noise prediction. Dashed gray lines denote U-Net skip connections, and orange arrows indicate ControlNet residual connections.

2025). However, the scarcity of publicly available neonatal HIE data (Bao et al., 2025), along with the substantial expertise and efort required for annotation, restricts the scalability and external generalizability of these methods. Unsupervised anomaly detection provides an alternative by learning a normative representation from healthy images and identifying deviations in a potentially pathological subject without directly optimizing on lesion labels (Schlegl et al., 2019; Fernando et al., 2021). A common strategy is to reconstruct a pseudo-healthy counterpart of the observed image and interpret the reconstruction residual as abnormal ity (Schlegl et al., 2019; Wolleb et al., 2022; Wyatt et al., 2022).

However, reconstruction-based anomaly detection must balance pathology removal against preservation of healthy subject-specific anatomy. Excessive corruption or insuficient conditioning may alter normal structures and generate false-positive residuals, whereas strong subject conditioning may reproduce pathological signals and reduce lesion contrast. This trade-of is particularly problematic for HIE because difusion abnormalities may be subtle, spatially diffuse, and poorly separated from normal neonatal variability (Forbes et al., 2000).

Difusion models have recently emerged as expressive generative priors for unsupervised medical anomaly detection (Wolleb et al., 2022; Wyatt et al., 2022; Frotscher et al., 2026). Their multi noise level denoising process provides an opportunity to identify deviations from learned normal anatomy other than the residual of a single final reconstruction. However, reliably distinguishing subtle pathological signals from subject-specific anatomy, acquisition-related variability, and stochastic denoising errors remains challenging.

In this work, we propose OpenMAP-HIE, an anatomyconditioned, pathology-informed score-based difusion framework for unsupervised neonatal HIE lesion localization (Fig. 1). A difusion model learns a healthy neonatal ADC prior, and a ControlNet branch incorporates the observed image as a spatial condition to capture patient-specific anatomy. We derive anomaly evidence from discrepancies between the healthy-prior and subject-conditioned score estimates, implemented through their noise-prediction differences (Fig. 2). These discrepancies characterize how subject-specific conditioning modifies the local denoising response of the learned healthy prior. Both branches evaluate the same noisy input at each noise level, with inputs generated directly from the observed image rather than propagated through a sequential reverse-difusion chain. This formulation avoids dependence on a final pseudohealthy reconstruction and the propagation of errors between reverse-difusion steps.

To improve pathological specificity, we incorporate a directional prior based on the characteristic ADC reduction associated with acute HIE (Forbes et al., 2000; Bobba et al., 2023; Bao et al., 2025). Specifically, we retain discrepancies whose corresponding denoising estimates indicate a higher

Normal slice: Forward – noise sample Patient-conditioned noise reverse Healthy-prior noise reverseIDirectional discrepancy  
![](images/eb11ac766c940b3ff129c435c98287514e634a429a676709396daa476271fead.jpg)  
(a)

HIE lesion slice:  Forward – noise sample  Patient-conditioned noise reverse Healthy-prior noise reverseIDirectional discrepancy  
![](images/4a703d1a013080d44935893cbcacf1634da83a546911cbffebf0e254e013e052.jpg)  
(b)  
Figure 2: Illustration of inference using pathology-informed noise-prediction discrepancy aggregation for (a) a normal slice and (b) a slice containing HIE lesions. The observed ADC image $\mathbf { x } _ { 0 }$ is independently perturbed at multiple noise levels. At each level, the patient-conditioned (green dashed line) and healthy-prior (red dashed line) branches evaluate the same noisy input, and their predicted noise diference is retained only in the direction consistent with the reduced-ADC prior. Normal tissue is expected to exhibit relatively small discrepancies, whereas lesion regions may produce larger, directionally consistent responses across noise levels. The resulting discrepancy maps are aggregated using the geometric mean, $A _ { \mathrm { G M } }$ , to obtain the anomaly map. Blue points denote noisy inputs, green and orange markers denote patient-conditioned and healthyprior predictions, respectively, and red brackets indicate directional discrepancies. The curves provide a sequential forwarddifusion trajectory. The lesion ground truth annotation in (b) is shown for reference only.

ADC under the healthy prior than under subject conditioning, and suppress discrepancies in the opposite direction. The retained responses are aggregated across independently evaluated noise levels using a geometric mean to emphasize directionally consistent evidence. Population-derived ADC �-maps (Bao et al., 2025; Rao et al., 2025) are additionally investigated as an optional refinement, while the core framework requires only the subject’s ADC image at inference.

The principal contributions of this work are:

• We introduce an anatomy-conditioned difusion framework that contrasts healthy-prior and subject-conditioned responses without using lesion masks as optimization targets.

• We formulate anomaly evidence through conditional– unconditional score discrepancies, implemented as noise-prediction diferences and aggregated across independently evaluated noise levels, without requiring a sequential reverse-difusion chain or a final pseudohealthy reconstruction.

• We incorporate an HIE-specific directional prior that retains score discrepancies consistent with the expected ADC reduction of acute injury, adding pathology informed directionality to the score-based anomaly representation.

• We evaluate the framework on the public BONBID-HIE benchmark using voxel-ranking, lesion-overlap, and surface-distance metrics, and further assess populationderived ADC �-map refinement as an optional extension.

The proposed ADC-only method achieved improved lesion overlap and surface localization relative to the evaluated unsupervised baselines, while the pathology-informed directional constraint provided substantial gains in lesion localization. Population-derived �-map refinement further improved overlap and boundary accuracy when populationlevel normative information was available. The code of this work is available at github.com/OishiLab/OpenMAP-HIE.

## 2. Related Works

## 2.1. Supervised Anomaly Detection

Supervised medical image segmentation has advanced through encoder–decoder architectures such as U-Net (Ronneberger et al., 2015) and UNet++ (Zhou et al., 2019; Rao et al., 2025). Recent HIE segmentation systems have extended to three-dimensional U-Net variants, SegResNet, Swin-UNETR (Hatamizadeh et al., 2021; Almansour et al., 2025), ensembles, and loss functions designed for highly imbalanced lesions (Bao et al., 2025). These approaches learn lesion appearance directly from voxel-wise annotations. In contrast, our framework learns a healthy-image prior and derives lesion evidence without using lesion masks as optimization targets.

## 2.2. Unsupervised Anomaly Detection

Reconstruction-based anomaly detection learns normal image appearance and identifies abnormalities through discrepancies between an input and its reconstructed counterpart. Autoencoders and variational autoencoders learn constrained representations of healthy images (Baur et al., 2021), while f-AnoGAN combines a GAN-based normalimage model with an encoder for eficient latent inference (Schlegl et al., 2019). Denoising autoencoders instead learn to recover healthy images from artificially corrupted inputs, where appropriately designed coarse noise can improve anomaly detection while preserving normal anatomical detail (Kascenas et al., 2022). IterMask<sup>2</sup> combines iterative spatial mask refinement with high-frequency image information to improve reconstruction of normal tissue and reduce false-positive responses (Liang et al., 2024).

These methods depend on reconstruction errors being larger in pathological regions than in healthy regions. However, normal anatomy may be reconstructed imperfectly, while suficiently expressive models may partially reproduce pathological structures. Consequently, reconstruction fidelity alone does not guarantee pathological specificity.

## 2.3. Difusion Model and Score Method

Denoising difusion probabilistic models (DDPMs) learn a data distribution by progressively perturbing images with noise and training a neural network to predict the noise of removal for the reverse process (Ho et al., 2020). Under the standard Gaussian perturbation formulation, noise prediction provides a parameterization of the score, the gradient of the perturbed data log density with respect to the noisy input (Song et al., 2021). This connection enables anomaly assessment through both reconstructed images and noiselevel dependent noise predictions. Further, compared with GANs, difusion models generally ofer stable training and high-fidelity outputs (Dhariwal and Nichol, 2021), making them suitable for learning healthy anatomical priors.

Early difusion-based methods primarily used pseudohealthy reconstruction. AnoDDPM uses structured Simplex noise to obscure abnormalities at lower difusion noise levels while retaining more image context (Wyatt et al., 2022). Wolleb et al. (2022) combined deterministic Denoising Diffusion Implicit Model (DDIM) inversion (Song et al., 2020) with classifier guidance (Dhariwal and Nichol, 2021) to generate counterfactual healthy images. Patched difusion models restrict corruption to local image regions and use surrounding anatomy as contextual information for denoising process, with pseudo-healthy reconstruction based on ensembles of predictions through sliding-window procedure (Behrendt et al., 2024).

Subsequent methods have attempted to restrict restoration to suspected pathological regions to avoid unnecessary modification of healthy anatomy. AutoDDPM first generates an anomaly-likelihood map and then uses masking, stitching, and joint-noise resampling to inpaint candidate abnormalities while harmonizing the reconstructed content with the original image (Bercea et al., 2023). Temporal Harmonization for Optimal Restoration (THOR) instead constructs intermediate anomaly maps during reverse diffusion and uses them as implicit guidance to repeatedly restore presumed healthy regions from the original input (Bercea et al., 2024). These methods are conceptually related to difusion-based inpainting approaches such as RePaint (Lugmayr et al., 2022), in which known image regions are repeatedly reintroduced during sampling to constrain the generated result.

Masking and local restoration have also been explored in latent difusion frameworks, with introducing localized latent-space corruptions during training as proxies for anomalous regions. MAD-AD applies forward difusion only to randomly selected patches of healthy latent representations during training and jointly learns to reconstruct the clean latent representation and predict the region of corruption. At inference, its mask-prediction head identifies candidate anomalous patches, which are exclusively corrected during reverse difusion while the remaining regions are preserved (Beizaee et al., 2025a). DeCo-Dif reformulates latent diffusion to learn a direction of deviation from normality. It progressively corrects anomalous latent patches while leaving normal patches unchanged and combines reconstruction discrepancies measured in the latent and image spaces to produce the final anomaly map (Beizaee et al., 2025b).

Rather than producing a final pseudo-healthy reconstruction or iteratively refining a candidate inpainting mask, Aggregated Normative Difusion (ANDi) derives anomaly evidence directly from deviations within the difusion process (Frotscher et al., 2026). At each noise level, ANDi compares the analytical reverse-posterior mean conditioned on the observed image with the reverse mean predicted by the difusion model, both evaluated at the same noisy input. The resulting voxel-wise squared diferences are aggregated across noise levels using a geometric mean. The voxel-wise squared deviations are aggregated across multiple sampled noise levels using a geometric mean. Consequently, ANDi does not require an additional inpainting stage or calculate its anomaly map only from a final reconstruction residual. Extending this reconstruction-free direction, InvAD instead uses DDIM inversion (Song et al., 2020) to map image representations toward the noise prior and evaluates deviations through Gaussian likelihood scoring in the resulting final latent space (Sakai et al., 2026).

Our approach is most closely related to ANDi in its aggregation of discrepancies across noise levels. The key distinction is the comparison itself: we contrast subjectconditioned and learned healthy-prior score estimates, implemented through their noise-prediction diferences, rather than comparing a model prediction with an analytical posterior target. We further filter the discrepancy signals using directional information from an HIE-specific pathological prior.

## 2.4. HIE Pathology Directional Prior

Most reconstruction-based medical anomaly detection methods define abnormity from the diference between an observed image and a pseudo-healthy reconstruction, commonly using absolute or squared intensity residuals (Schlegl et al., 2019; Wolleb et al., 2022; Wyatt et al., 2022). More recent difusion-based approaches characterize abnormality through deviations from learned normal denoising dynamics (Bercea et al., 2024; Frotscher et al., 2026; Beizaee et al., 2025a) or gaussian prior likelihood (Sakai et al., 2026). These disease-agnostic formulations are applicable to diverse anomaly types, but generally treat positive and negative intensity deviations symmetrically. Consequently, a large anomaly score does not have strong pathological interpretability and cannot properly distinguish the target pathology from the anatomical variance and imaging noise.

This limitation is particularly relevant to neonatal HIE. During the acute difusion-restriction phase, cytotoxic edema typically produces lower ADC values in injured tissue than in healthy tissue (Forbes et al., 2000; Bobba et al., 2023; Onda et al., 2024; Bao et al., 2025). ADC thresholds and population-derived statistical maps have therefore been investigated for identifying and quantifying HIErelated injury (Bao et al., 2025). However, the difusionbased anomaly detection methods considered above do not explicitly incorporate the expected direction of the ADC abnormality into their multi noise-level discrepancy estimation. Our method incorporates this pathological information by retaining only positive diferences between subjectconditioned and healthy-prior unconditional noise predictions, corresponding to higher healthy-reference posterior than subject-conditioned reverse-mean estimates. Unlike a conventional unsigned reconstruction residual, this formulation suppresses denoising discrepancies that are large in magnitude but inconsistent with the characteristic ADC reduction of acute HIE.

## 3. Methods

## 3.1. Difusion Model

Let $\mathbf { x } _ { 0 } \ \sim \ q _ { \mathrm { h } } ( \mathbf { x } )$ denote a healthy neonatal ADC slice sampled from the healthy distribution. A denoising difusion probabilistic model (DDPM) (Ho et al., 2020) progressively perturbs $\mathbf { x } _ { 0 }$ through the forward Markov process:

$$
q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) = \mathcal { N } \left( \mathbf { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } \right) ,\tag{1}
$$

where $t = 1 , \dots , T$ and $\beta _ { t } \in ( 0 , 1 )$ specifies the variance schedule and $\alpha _ { t } = 1 - \beta _ { t }$ . Defining $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ , a perturbed image at an arbitrary noise level � can be sampled directly as

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \qquad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) .\tag{2}
$$

The reverse transition is parameterized as

$$
p _ { \theta } ( x _ { t - 1 } | x _ { t } ) = \mathcal { N } \left( \mathbf { x } _ { t - 1 } ; \pmb { \mu } _ { \theta } ( \mathbf { x } _ { t } , t ) , \sigma _ { t } ^ { 2 } \mathbf { I } \right) ,\tag{3}
$$

where � denotes the parameters of the healthy-prior denoiser and $\sigma _ { t } ^ { 2 }$ is determined by the adopted variance schedule. Under the noise-prediction parameterization, the reverse mean is

$$
{ \mu _ { \theta } } ( \mathbf { x } _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( { \bf { x } } _ { t } - \frac { { \beta _ { t } } } { \sqrt { 1 - { { \bar { \alpha } } _ { t } } } } \boldsymbol { \epsilon } _ { \theta } ( \mathbf { x } _ { t } , t ) \right) ,\tag{4}
$$

Following DDPM (Ho et al., 2020) and using the Evidence Lower Bound (ELBO) loss, we train with minimizing the noise prediction loss:

$$
\mathcal { L } ( \epsilon _ { \theta } ) : = \mathbb { E } _ { x _ { 0 } \sim q ( x _ { 0 } ) , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \| \epsilon _ { \theta } ( x _ { t } , t ) - \epsilon \| ^ { 2 } ]\tag{5}
$$

, where $x _ { t }$ can be expressed with $x _ { 0 }$ based on Eq. (2).

Based on the Stochastic Diferential Equation (SDE) formulation for the difusion process (Song et al., 2021), the noise prediction model $\epsilon _ { \theta }$ is equivalent to the score function:

$$
\epsilon _ { \theta } ( x _ { t } , t ) \propto - \nabla _ { x _ { t } } \log { q ( x _ { t } | t ) }\tag{6}
$$

, which points toward the higher density for the observed noised sample $x _ { t }$ at time �.

## 3.2. Spatial Conditioning through ControlNet

To preserve the subject level anatomical features and reduce hallucination from the reverse process, we learn the conditional difusion process with $\epsilon _ { \theta , \phi } ( x _ { t } , t | c = x _ { 0 } )$ using ControlNet (Zhang et al., 2023). For training, we learn with:

$$
\mathcal { L } ( \epsilon _ { \theta , \phi } ) : = \mathbb { E } _ { x _ { 0 } \sim q ( x _ { 0 } ) , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) } [ \| \epsilon _ { \theta , \phi } ( x _ { t } , t | c ) - \epsilon \| ^ { 2 } ]\tag{7}
$$

To prevent the model from ignoring its pre-trained generative priors, having the injected ControlNet signals to completely dominate the core difusion branch and learning the shortcut of directly using condition $ { \boldsymbol { { c } } } \  { \boldsymbol { { } = } } \ x _ { 0 }$ to explicitly calculate $\epsilon _ { t } .$ we adopt the conditioning dropout strategy outlined in Ho and Salimans (2021) and Zhang et al. (2023). Specifically, during training, we randomly replace the conditioning inputs with a null condition $\emptyset$ into the model using a dropout probability of $p _ { \mathrm { d r o p } } = 0 . 1$ . Thus, we have an alternative condition pipeline during training:

$$
c = \tilde { c } = \left\{ \begin{array} { r l } { \emptyset } & { { } , \mathrm { w i t h } \mathrm { p r o b a b i l i t y } p _ { \mathrm { d r o p } } } \\ { x _ { 0 } } & { { } , \mathrm { o t h e r w i s e } } \end{array} \right.\tag{8}
$$

The base U-Net is pretrained on healthy images and remains frozen during ControlNet training, providing a fixed healthyprior backbone. We use the null-conditioned prediction as the unconditional reference response:

$$
\epsilon _ { \theta , \phi } ( x _ { t } , t \mid c = \emptyset ) \simeq \epsilon _ { \theta } ( x _ { t } , t ) .\tag{9}
$$

, which retains the pretrained healthy-prior backbone.

## 3.3. Classifier-Free Guidance and Interpretation of the Conditional–Unconditional Discrepancy

At inference, classifier-free guidance (CFG) (Ho and Salimans, 2021) provides a convenient formulation for combining the healthy-prior reference and subject-conditioned denoising responses. The guided noise prediction is defined as

$$
\epsilon ^ { \mathrm { C F G } } ( \mathbf { x } _ { t } , t \mid \mathbf { c } ) = \epsilon ^ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) + w \left[ \epsilon ^ { \mathrm { c o n d } } ( \mathbf { x } _ { t } , t \mid \mathbf { c } ) - \epsilon ^ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) \right] ,\tag{10}
$$

where $w \ge 0$ denotes the guidance scale, controlling the contribution of the subject-specific condition.

The unconditional prediction is obtained from the ControlNetaugmented model using a null condition,

$$
\epsilon ^ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) = \epsilon _ { \theta , \phi } ( \mathbf { x } _ { t } , t \mid \mathbf { c } = \emptyset ) .\tag{11}
$$

The conditional prediction is obtained by providing the observed ADC image as the spatial condition,

$$
\epsilon ^ { \mathrm { c o n d } } ( \mathbf { x } _ { t } , t \mid \mathbf { c } ) = \epsilon _ { \theta , \phi } ( \mathbf { x } _ { t } , t \mid \mathbf { c } = \mathbf { x } _ { 0 } ) .\tag{12}
$$

We therefore use $\epsilon ^ { \mathrm { u n c o n d } }$ as the healthy-reference response, whereas $\epsilon ^ { \mathrm { c o n d } }$ represents the denoising response after incorporating subject-specific information from the observed image.

The efect of the CFG scale can be understood directly from Eq. (10):

• When $w = 0 ;$ , the guided prediction reduces to the null-conditioned healthy-reference response.

• For $0 < w < 1$

$$
\epsilon ^ { \mathrm { C F G } } = ( 1 - w ) \epsilon ^ { \mathrm { u n c o n d } } + w \epsilon ^ { \mathrm { c o n d } } ,\tag{13}
$$

which interpolates between the healthy-reference and subject-conditioned responses. Smaller values of $w$ reduce the influence of the observed subject and therefore allow stronger restoration toward the learned healthy distribution. However, this process may also reduce subject-specific anatomical fidelity and introduce reconstruction changes unrelated to pathology.

• When $w = 1$ , the guided prediction becomes the standard conditional prediction. Importantly, the healthy prior is still retained through the frozen pretrained U-Net, while the ControlNet branch contributes its learned subject-specific residual.

• For $\ w \mathrm { ~  ~ { ~ > ~ } ~ } 1$ , CFG extrapolates beyond the nullconditioned healthy-reference response, along the conditional–unconditional direction. This formulation highlights the conditional–unconditional discrepancy

$$
\delta _ { t } = \epsilon ^ { \mathrm { c o n d } } ( \mathbf { x } _ { t } , t \mid \mathbf { c } = \mathbf { x } _ { 0 } ) - \epsilon ^ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) ,\tag{14}
$$

which quantifies how the predicted denoising response changes when the observed subject image is introduced as spatial conditioning. Since the noise prediction is proportional to the negative score of the perturbed image distribution, as described in Eq. (6), $\delta _ { t }$ can equivalently be interpreted as a local diference between the subject-conditioned and healthyreference denoising fields.

In image regions whose appearance is compatible with the learned healthy distribution, the subject condition is expected to require only a limited modification of the healthy-reference denoising response, resulting in a relatively small discrepancy. In contrast, image characteristics that are poorly represented by the healthy prior may induce a larger conditional correction. Such deviations may arise from pathology, but may also reflect normal anatomical variation, registration diferences, acquisition-related intensity variation, or imaging artifacts. The conditional–unconditional discrepancy therefore characterizes deviation from the learned healthy denoising behavior, as illustrated in Fig. 2.

Repeated extrapolation with $w \ > \ 1$ progressively bias the reconstruction toward features supported by the subject-specific condition. In a pathological input, this may result in stronger preservation or emphasis of lesion-associated appearance in the final reconstruction, which motivates the “lesion-emphasizing” CFG setting and conversely “healthy-restoration” for $w < 1$ evaluated in the reconstruction-based ablation in Sec. 4.4.

However, at a fixed noisy input $\mathbf { x } _ { t } ,$ CFG does not introduce additional discriminative information into this discrepancy, since

$$
\epsilon ^ { \mathrm { C F G } } - \epsilon ^ { \mathrm { u n c o n d } } = w \delta _ { t } .\tag{15}
$$

Thus, changing � only scales the conditional–unconditional direction at a given noise level. Although the guidance scale can substantially alter a complete reverse-difusion trajectory and therefore afect reconstruction-based anomaly maps, the underlying discrepancy $\delta _ { t }$ itself is independent of the selected CFG scale. In the following section, we therefore use the conditional–unconditional discrepancy directly and introduce HIE pathology informed directional information to determine which components of the deviation are biologically consistent with acute difusion restriction.

## 3.4. Pathology-Informed Directional Discrepancy and Multi Noise-Level Accumulation

Acute HIE injury is typically associated with restricted water difusion and reduced ADC values relative to healthy brain tissue (Onda et al., 2024). We incorporate this characteristic as a directional pathological prior when comparing the subject-conditioned denoising response with the healthyreference response.

For a given noisy input $\mathbf { x } _ { t } .$ , we consider two reversedifusion responses: a healthy-prior branch and a subjectconditioned branch. The healthy-prior branch corresponds to the null-conditioned prediction,

$$
\epsilon ^ { \mathrm { p r i o r } } ( \mathbf { x } _ { t } , t ) \equiv \epsilon ^ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) ,\tag{16}
$$

as defined in Eq. (11), whereas the subject-conditioned branch uses $\epsilon ^ { \mathrm { c o n d } } ( \mathbf { x } _ { t } , t \mid \mathbf { c } = \mathbf { x } _ { 0 } )$ from Eq. (12).

For each branch � ∈ {prior, cond}, the corresponding reverse transition can be written as

$$
\hat { \mathbf { x } } _ { t - 1 } ^ { b } = \pmb { \mu } _ { t } ^ { b } + \sigma _ { t } \mathbf { z } _ { t } ^ { b } , \qquad \mathbf { z } _ { t } ^ { b } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{17}
$$

where

$$
\pmb { \mu } _ { t } ^ { b } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathbf { x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon ^ { b } ( \mathbf { x } _ { t } , t ) \right) .\tag{18}
$$

To isolate the efect of subject-specific conditioning, the two branches are evaluated from the same noisy input $\mathbf { x } _ { t }$ using the same stochastic realization $\mathbf { z } _ { t }$

The stochastic terms therefore cancel when the two reverse estimates are compared:

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { t - 1 } ^ { \mathrm { p r i o r } } - \hat { \mathbf { x } } _ { t - 1 } ^ { \mathrm { c o n d } } = \pmb { \mu } _ { t } ^ { \mathrm { p r i o r } } - \pmb { \mu } _ { t } ^ { \mathrm { c o n d } } . } \end{array}\tag{19}
$$

Substituting Eq. (18), the reverse-mean diference can be expressed directly in terms of the corresponding noise predictions:

$$
\mu _ { t } ^ { \mathrm { p r i o r } } - \mu _ { t } ^ { \mathrm { c o n d } } = \frac { \beta _ { t } } { \sqrt { \alpha _ { t } ( 1 - \bar { \alpha } _ { t } ) } } \left( \epsilon _ { t } ^ { \mathrm { c o n d } } - \epsilon _ { t } ^ { \mathrm { p r i o r } } \right) .\tag{20}
$$

Since $\epsilon ^ { \mathrm { p r i o r } } \equiv \epsilon ^ { \mathrm { u n c o n d } }$ , this can equivalently be written as

$$
\begin{array} { r l } & { \pmb { \mu } _ { t } ^ { \mathrm { p r i o r } } - \pmb { \mu } _ { t } ^ { \mathrm { c o n d } } = \kappa _ { t } \left( \pmb { \epsilon } _ { t } ^ { \mathrm { c o n d } } - \pmb { \epsilon } _ { t } ^ { \mathrm { u n c o n d } } \right) , } \\ & { \mathrm { w h e r e } \kappa _ { t } = \cfrac { \beta _ { t } } { \sqrt { \alpha _ { t } ( 1 - \bar { \alpha } _ { t } ) } } . } \end{array}\tag{21}
$$

For all valid difusion timesteps, $\kappa _ { t } > 0$

In a healthy region, both branches are expected to generate similar reverse estimates because the observed appearance is compatible with the learned healthy distribution. In an HIE lesion, however, the subject-conditioned branch is expected to preserve more of the abnormally low ADC signal present in the observed image, while the unconditional branch favors restoration toward a healthy-compatible appearance with a relatively higher ADC value. Therefore, for a voxel � within a low-ADC lesion, we expect

$$
\mu _ { t , i } ^ { \mathrm { p r i o r } } - \mu _ { t , i } ^ { \mathrm { c o n d } } > 0 .\tag{22}
$$

Accordingly, we define the directional discrepancy score at reverse step � as

$$
d _ { t , i } = \left[ \mu _ { t , i } ^ { \mathrm { p r i o r } } - \mu _ { t , i } ^ { \mathrm { c o n d } } \right] _ { + } , \quad \left[ u \right] _ { + } = \operatorname* { m a x } ( u , 0 ) ,\tag{23}
$$

where the positive-part operator retains only discrepancies consistent with restoration from a lower subject-conditioned ADC value toward a higher healthy-prior ADC value. Using Eq. (21), this discrepancy can be further expressed in terms of the predicted noise:

$$
\begin{array} { r l } & { d _ { t , i } = \left[ \kappa _ { t } \left( \epsilon _ { t , i } ^ { \mathrm { c o n d } } - \epsilon _ { t , i } ^ { \mathrm { u n c o n d } } \right) \right] _ { + } } \\ & { ~ \propto \left[ \epsilon _ { t , i } ^ { \mathrm { c o n d } } - \epsilon _ { t , i } ^ { \mathrm { u n c o n d } } \right] _ { + } } \end{array}\tag{24}
$$

, since $\kappa _ { t } > 0 .$

This formulation is directly related to the CFG representation introduced in Sec. 3.3. From Eq. (10), for $w > 0 ,$ , CFG therefore preserves the sign of the underlying conditional– unconditional discrepancy while scaling its magnitude. Because our anomaly score is defined directly from $\epsilon ^ { \mathrm { c o n d } } -$ �<sup>uncond</sup>, it does not require selecting a CFG scale. Similarly, although the reverse-mean discrepancy in Eq. (21) contains the timestep-dependent coeficient $\kappa _ { t } .$ this coeficient does not alter the sign used by the pathological directional constraint. We therefore use the noise-prediction discrepancy itself as the operational anomaly representation rather than explicitly weighting it by $\kappa _ { t }$

A subtle lesion may produce only a weak discrepancy at any individual noise level. We therefore aggregate the directional evidence across a selected subset of independently evaluated noise levels $\mathcal { T } \subseteq \{ 1 , \dots , T \}$ . For each $t \in \tau$ , a noisy input is constructed directly from the same observed image $\mathbf { x } _ { 0 }$ according to Eq. (2), with no state propagated from one evaluated noise level to another. The accumulated anomaly score at voxel � is defined using the geometric mean (Frotscher et al., 2026):

$$
A _ { i } = \exp \left( \frac { 1 } { \vert \mathcal { T } \vert } \sum _ { t \in \mathcal { T } } \log \left( d _ { t , i } + \zeta _ { \mathrm { s t a b } } \right) \right) ,\tag{25}
$$

where $\zeta _ { \mathrm { s t a b } } ~ > ~ 0$ is a small numerical stability constant introduced to avoid taking the logarithm of zero.

The positive-part operation is applied before accumulation so that discrepancies opposite to the expected low-ADC lesion direction do not cancel directionally consistent responses. Compared with an arithmetic mean, geometric aggregation reduces the influence of isolated large discrepancies and favors voxels that exhibit consistently elevated directional responses across multiple corruption levels (Frotscher et al., 2026). Because each noise level is evaluated independently from the original subject image, the accumulated score reflects persistent disagreement between the subject-conditioned and healthy-reference denoising fields without requiring a sequential reverse-difusion trajectory or a final pseudo-healthy reconstruction.

The directional constraint improves pathological relevance but does not by itself establish lesion specificity. Normal anatomical variation, partial-volume efects, registration diferences, acquisition-related intensity variation, and imaging artifacts may also produce positive discrepancies. The resulting score is therefore interpreted as evidence of a persistent deviation from the learned healthy denoising behavior that is directionally consistent with acute HIErelated ADC reduction, rather than as a calibrated lesion probability.

## 3.5. Spatial Smoothing of the Anomaly Map

Although accumulation across multiple noise levels enhances persistent conditional–unconditional discrepancies, the resulting anomaly map may contain spatially isolated high-frequency responses. As following Frotscher et al. (2026), we apply a mean filtering to improve the local spatial coherence of the accumulated score.

Let $A _ { i }$ denote the accumulated directional discrepancy at voxel �. The spatially regularized anomaly score is defined as

$$
\widetilde { A } _ { i } = \frac { 1 } { | \mathcal { N } _ { k } ( i ) | } \sum _ { j \in \mathcal { N } _ { k } ( i ) } A _ { j } ,\tag{26}
$$

where $\mathcal { N } _ { k } ( i )$ denotes the local $k \times k$ neighborhood centered at voxel �, and � is the mean-filter kernel size. The same filter is applied to all subjects before thresholding the continuous anomaly map.

Mean filtering reduces isolated high-valued responses by averaging them with low-valued neighboring voxels, whereas spatially coherent regions remain relatively elevated because their neighboring voxels exhibit consistently high discrepancy scores. However, an excessively large kernel may dilute focal lesion responses and spread elevated anomaly scores into adjacent healthy tissue, with increasing false positive detections. We use � = 5 in the final framework, as selected using the validation set. The sensitivity of the method to the kernel size is evaluated in Table 5.

## 3.6. Population Z-Map Guided Post-processing Lesion Refinement

The difusion-derived anomaly map characterizes subjectspecific disagreement between the conditional and unconditional noise predictions. Although this discrepancy provides complementary evidence of abnormal tissue, it may also respond to normal anatomical variation, tissue boundaries, registration errors, and acquisition-related artifacts. We therefore combine the difusion anomaly score with a population-normalized ADC �-map (Bao et al., 2025), which provides an independent spatial measure of deviation from the healthy reference population. Because acute HIE lesions are generally associated with restricted difusion and reduced ADC values (Onda et al., 2024), strongly negative �-scores provide biologically consistent evidence for lesion localization (Bao et al., 2025; Rao et al., 2025).

Let $A _ { i }$ denote the accumulated directional noise-prediction discrepancy at voxel �, as defined in Eq. (25), and let $Z _ { i }$ denote the corresponding population ADC �-score. All operations are performed independently for each subject:

$$
\Omega = \left\{ i : A _ { i } { \mathrm { ~ a n d ~ } } Z _ { i } { \mathrm { ~ a r e ~ f i n i t e } } \right\} .\tag{27}
$$

High-confidence low-ADC core. A high-confidence lesion core is first identified solely from the population $Z -$ map:

$$
C _ { \mathrm { r a w } } = \left\{ i \in \Omega : Z _ { i } \leq - 2 . 0 \right\} ,\tag{28}
$$

as following Rao et al. (2025) where −2.0 is an consistent threshold for lesion on �-map.

Three-dimensional connected-component analysis was performed over the entire volume binary mask. Voxels sharing a face, edge, or corner were considered connected (26- connectivity), and connected regions containing fewer than 150 voxels were removed:

$$
C = \bigcup _ { k : | C _ { k } | \geq 1 5 0 } C _ { k } , \qquad C _ { k } = \mathrm { C C } _ { 2 6 } ( C _ { \mathrm { r a w } } ) ,\tag{29}
$$

where $\mathrm { C C } _ { 2 6 } ( \cdot )$ denotes 26-connected component decompo sition. The size constraint of each lesion component suppresses isolated negative �-score responses that are unlikely to represent spatially coherent HIE lesions.

Difusion-supported component recovery. The difusion anomaly map is subsequently used to recover complementary lesion components that may not be included in the high-confidence �-map core. A subject-specific highanomaly threshold is defined as $\tau _ { A } = Q _ { 9 9 } ( A ; \Omega )$ , corresponding to the 99th percentile of the normalized anomaly scores. Candidate voxels must simultaneously exhibit a high difusion anomaly score and a negative population deviation:

$$
R = \left\{ i \in \Omega : A _ { i } \geq \tau _ { A } \ \wedge Z _ { i } \leq - 1 . 9 5 \right\} .\tag{30}
$$

Thus, the difusion score alone cannot introduce a candidate voxel without supporting evidence of reduced ADC relative to the healthy population.

The candidate mask � is decomposed into 26-connected components,

$$
\{ R _ { j } \} _ { j = 1 } ^ { J } = \mathrm { C C } _ { 2 6 } ( R ) .\tag{31}
$$

A candidate component $R _ { j }$ is retained only if its size is larger than 10 voxels, i.e. $1 0 \leq \check { | } R _ { j } |$ and it is spatially adjacent to the cleaned �-map core:

$$
R _ { j } \cap \mathrm { D i l a t e } _ { 2 6 } ( C , 1 ) \neq \emptyset ,\tag{32}
$$

where Dilat ${ \bf e } _ { 2 6 } ( C , 1 )$ denotes a one-voxel three-dimensional dilation of � using 26-connectivity. When a candidate component satisfies both constraints, the entire connected component is included in the prediction. This component-level operation allows the method to recover spatially coherent lesion extensions while preventing distant high-anomaly regions from being detected independently. The set of accepted components is therefore

$$
\mathcal { I } _ { \mathrm { a c c } } = \left\{ j : 1 0 \leq | R _ { j } | \ \wedge R _ { j } \cap \mathrm { D i l a t e } _ { 2 6 } ( C , 1 ) \neq \emptyset \right\}\tag{33}
$$

The final binary lesion map for this current subject is defined as

$$
M = C \cup \bigcup _ { j \in \mathcal { I } _ { \mathrm { a c c } } } R _ { j } .\tag{34}
$$

This conservative fusion combines two complementary representations of abnormality. The population �-map provides the primary localization of spatially coherent low-ADC tissue, whereas the difusion discrepancy supplies subject-specific evidence for recovering nearby lesion components. No manual lesion annotation is required to generate the final prediction at inference.

## 3.7. Dataset and Preprocessing

We used the BONBID-HIE dataset (Bao et al., 2025) for model development and evaluation. The dataset was divided into 85 training subjects, 4 validation subjects, and 44 test subjects, as following its predefined partitions, without performing an additional subject-level split and with no subject overlap among the three partitions. All 133 subjects in the selected release were included, with no subject-level exclusions. Each subject included an three-dimensional ADC map and its corresponding population-normalized �-map. To account for variations in voxel spacing and spatial misalignment across scans, all volumes were afine registered to the JHU-neonate-SS atlas template (Oishi et al., 2011) using ANTsPy (Tustison et al., 2021). At the end, all scans were resampled to a common matrix size $2 5 6 \times 2 5 6 \times 1 2 8$ and voxel size $0 . 5 8 \times 0 . 5 8 \times 0 . 7 4 ~ m m ^ { 3 }$ , with nearest neighbor interpolation, as following Rao et al. (2025). The intensity of each volume went through its independent 99th percentile normalization.

The difusion model was trained using two-dimensional axial ADC slices. To construct a training set representing non-lesion brain tissue, we examined the corresponding lesion mask for each slice from the training subjects and retained only slices containing no annotated lesion voxels. The lesion masks were used exclusively for training-slice selection and were not provided to the model as inputs or optimization targets. The �-maps were not used to train the difusion model and were introduced only during the postprocessing stage described in Section 3.6. We have 6042 slices of non-lesion axial brain slices for training and 4838 slices of lesion axial brain slices from the training subjects for the hyperparameter selection for the post-processing stage.

The predefined validation partition was used to monitor loss and model performance during training to reduce overfitting and to select the mean-filter kernel size. Evaluation was performed over complete 3D volumes from the 44 test subjects.

## 3.8. Network Architecture and Implementation Details

The two-stage training procedure is summarized in Algorithm 1, and the discrepancy-based inference procedure is described in Algorithm 2. The difusion denoiser was implemented using a U-Net architecture (Ronneberger et al., 2015). The ControlNet conditioning branch was constructed using the encoder portion of the U-Net architecture, allowing the observed subject image to be encoded at multiple spatial resolutions (Zhang et al., 2023) (Fig. 1). As following Zhang et al. (2023), at each resolution level, the features produced by the ControlNet encoder were projected through a zeroinitialized convolution and injected into the corresponding feature stream of the primary U-Net through a residual connection. Specifically, the fused feature representation at resolution level $\ell$ was defined as

$$
\widetilde { \mathbf { h } } _ { \ell } = \mathbf { h } _ { \ell } ^ { \mathrm { U N e t } } + \mathcal { Z } _ { \ell } \left( \mathbf { h } _ { \ell } ^ { \mathrm { C o n t r o l } } \right) ,\tag{35}
$$

where $\mathbf { h } _ { \ell } ^ { \mathrm { U N e t } }$ and $\mathbf { h } _ { \ell } ^ { \mathrm { C o n t r o l } }$ denote the primary U-Net and ControlNet feature maps, respectively, and $\mathcal { Z } _ { \ell }$ denotes the zero-initialized convolutional projection. Both the weights and biases of $\mathcal { Z } _ { \ell }$ were initialized to zero. Consequently, the ControlNet residual was initially zero and the pretrained U-Net was frozen throughout the second stage of training, allowing the pretrained U-Net to preserve its learned healthy denoising behavior at the beginning of the second training stage. As training progressed, the residual projections learned to introduce subject-specific spatial information and to adjust the pre-learned healthy prior.

Both training stages were optimized using AdamW (Loshchilov and Hutter, 2017) with a batch size of 8. We employed a warm-up cosine-decay learning-rate schedule. During the first 5% of the total optimization steps, the learning rate was increased from $2 \times 1 0 ^ { - 5 }$ to the target learning rate of $1 \times 1 0 ^ { - 4 }$ . It was subsequently decayed according to a cosine schedule over the remaining optimization steps. An exponential moving average of the model parameters was maintained with a decay rate of 0.995.

In the first stage, the U-Net difusion model was trained for $K _ { 1 } ~ = ~ 3 0 0$ epochs to learn the healthy neonatal image distribution. In the second stage, the pretrained U-Net served as the healthy generative prior, and the ControlNet branch was trained for an additional $K _ { 2 } = 2 0 0$ epochs to incorporate subject-specific spatial conditioning. To enable both nullconditioned and subject-conditioned noise predictions, the conditioning image was randomly dropped with probability $p _ { \mathrm { d r o p } } = 0 . 1$ during ControlNet training.

The difusion process used $T = 1 0 0 0$ noise steps with a linear variance schedule, in which $\beta _ { t }$ increased from $\beta _ { 1 } =$ $1 \times 1 0 ^ { - 4 } \mathrm { t o } \beta _ { T } = 0 . 0 2$ . During training, the input images were normalized, and random horizontal flipping was applied with a probability of 0.5.

During inference, following the intermediate-noise strategy described in (Frotscher et al., 2026), we accumulated the conditional-unconditional discrepancy over difusion timesteps $\textit { \textbf { \textit { T } } } = \{ 7 5 , 7 6 , \dots , 2 0 0 \}$ . This interval was selected to balance corruption of the pathological signal with preservation of subject-specific anatomy. At very low noise levels, the lesion appearance remains largely preserved in $\mathbf { x } _ { t } ,$ providing insuficient perturbation for the healthy prior to reveal a clear discrepancy. Conversely, at substantially higher noise levels, excessive corruption removes subjectspecific anatomical information and may introduce reconstruction diferences unrelated to pathology. The selected intermediate timesteps weaken the lesion-related signal while retaining suficient anatomical information for spatially meaningful comparison between the conditional and unconditional noise predictions.

## 3.9. Evaluation Metrics

We evaluated the predicted anomaly maps using complementary voxel-ranking, volumetric-overlap, and surfacedistance metrics. Let $A _ { s , i }$ denote the continuous anomaly score at voxel � of subject �, and let $G _ { s }$ denote the corresponding ground truth lesion mask. Given a threshold $\lambda ,$ the binary prediction is $P _ { s } ( \lambda ) = \{ i \ : \ A _ { s , i } > \ \lambda \}$ . Each metric was first computed over the complete 3D volume of each subject, rather than independently on individual 2D slices, and was reported as the mean ± standard deviation across the test subjects.

Dice similarity coeficient and threshold selection. The Dice similarity coeficient measures volumetric overlap between the predicted mask � and the ground truth mask $G \mathrm { : }$

$$
\operatorname { D i c e } ( P , G ) = { \frac { 2 | P \cap G | } { | P | + | G | } } ,\tag{36}
$$

where $| \cdot |$ denotes the number of foreground voxels. Dice ranges from 0 to 1, with higher values indicating greater overlap.

For each evaluated method, we determined the ceiling Dice by selecting a single threshold $\lambda ^ { * }$ that maximized the mean subject-wise Dice score on the test set:

$$
\lambda ^ { * } = \underset { \lambda \in \Lambda } { \arg \operatorname* { m a x } } \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { s = 1 } ^ { N _ { \mathrm { t e s t } } } \mathrm { D i c e } \big ( P _ { s } ( \lambda ) , G _ { s } \big ) ,\tag{37}
$$

where Λ denotes the candidate threshold set and $N _ { \mathrm { t e s t } }$ is the number of test subjects. We reported the resulting ceiling Dice as the Dice score throughout the manuscript. The same threshold $\lambda ^ { * }$ was then applied to all test subjects to compute the following described surface-distance metrics without further optimization. These binary-segmentation metrics therefore describe performance at a Dice-optimal threshold selected using the test-set annotations.

Averageprecision. Average precision (AP) evaluates voxellevel ranking by summarizing the precision–recall curve obtained from the continuous anomaly scores:

$$
\mathsf { A P } = \sum _ { k } \left( \operatorname { R e c } _ { k } - \operatorname { R e c } _ { k - 1 } \right) \operatorname { P r e c } _ { k } ,\tag{38}
$$

where $\mathrm { P r e c } _ { k }$ and $\operatorname { R e c } _ { k }$ denote precision and recall at the �-th sorted threshold. AP does not depend on the selected threshold $\lambda ^ { * }$ and is particularly informative because lesion voxels constitute a small proportion of the evaluated brain volume. Higher AP indicates better ranking of lesion voxels above non-lesion voxels.

Surface-distance metrics. For clarity, we omit the subject index below and write $P ~ = ~ P _ { s } ( \lambda ^ { * } )$ and $\textit { G } = \textit { G } _ { s }$ . Let $S _ { P } = \partial P$ and $S _ { G } = \partial G$ denote the predicted and ground truth lesion surfaces. The distance from a surface voxel � to a surface � is

$$
d ( x , S ) = \operatorname* { m i n } _ { y \in S } \| x - y \| _ { 2 } ,\tag{39}
$$

where coordinates are expressed in physical space, accounting for voxel spacing. All distances are reported in millimeters.

The symmetric Hausdorf distance (HD) measures the maximum surface disagreement:

$$
\mathrm { H D } ( P , G ) = \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { x \in S _ { P } } d ( x , S _ { G } ) , \operatorname* { m a x } _ { y \in S _ { G } } d ( y , S _ { P } ) \right\} .\tag{40}
$$

Lower HD indicates a smaller maximum boundary error.

The mean average surface distance (MASD) measures the mean bidirectional distance between the two lesion surfaces:

$$
\begin{array} { c } { { \displaystyle { \mathrm { \bf M A S D } } ( P , G ) = \frac { 1 } { 2 } \left[ \bar { d } ( S _ { P } , S _ { G } ) + \bar { d } ( S _ { G } , S _ { P } ) \right] , } } \\ { { \displaystyle \bar { d } ( S _ { 1 } , S _ { 2 } ) = \frac { 1 } { | S _ { 1 } | } \sum _ { x \in S _ { 1 } } d ( x , S _ { 2 } ) . } } \end{array}\tag{41}
$$

Algorithm 1 Two-stage training of the healthy-prior DDPM   
and ControlNet   
Input: Healthy ADC dataset $D _ { \mathrm { h } } ;$ difusion schedule $\{ \beta _ { t } \} _ { t = 1 } ^ { T } ;$ training iter  
ations $K _ { 1 }$ and $K _ { 2 } ;$ condition-dropout probability $p _ { \mathrm { d r o p } }$   
Output: Healthy-prior parameters � and ControlNet parameters $\phi$   
1: Define $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$   
Stage 1: Healthy-prior DDPM   
2: Initialize the unconditional denoiser $\epsilon _ { \theta }$   
3: for $k = 1 , \ldots , K _ { 1 }$ do   
4: Sample $\mathbf { x } _ { 0 } \sim \mathcal { D } _ { \mathrm { h } } , t \sim \mathcal { V } \{ 1 , \dots , T \} .$ , and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
5: Construct $\mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$   
6: Compute $\mathcal { L } _ { \mathrm { p r i o r } } = \| \epsilon - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) \| _ { 2 } ^ { 2 }$   
7: Update � using $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p r i o r } }$   
8: end for   
Stage 2: ControlNet   
9: Freeze � and initialize the ControlNet parameters $\phi$   
10: for $k = 1 , \dots , K _ { 2 }$ do   
11: Sample $\mathbf { x } _ { 0 } \sim \mathsf { \bar { D } _ { h } } , t \sim \mathcal { V } \{ 1 , \dots , T \}$ , and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
12: Construct $\mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$   
13: Set $\widetilde { \mathbf { c } } = \varnothing$ with probability $p _ { \mathrm { d r o p } } ;$ otherwise se $\widetilde { \mathbf { c } } = \mathbf { x } _ { 0 }$   
14: Compute $\mathcal { L } _ { \mathrm { c o n d } } = \| \epsilon - \epsilon _ { \theta , \phi } ( \mathbf { x } _ { t } , t \mid \widetilde { \mathbf { c } } ) \| _ { 2 } ^ { 2 }$   
15: Update only � using $\nabla _ { \phi } \mathcal { L } _ { \mathrm { c o n d } }$   
16: end for   
17: return $\theta , \phi$

Algorithm 2 Multi noise-level conditional-unconditional   
score discrepancy inference   
Input: Patient ADC slice ${ \bf x } _ { 0 } ;$ selected noise levels $\tau ;$ trained parameters   
�, �; small numerical stability constant $\zeta _ { \mathrm { s t a b } } ;$ brain mask �   
Output: Difusion-derived anomaly map �   
1: Set patient condition $\mathbf { c } \gets \mathbf { x } _ { 0 }$   
2: Initialize $\mathbf A \gets \mathbf 0$   
3: for all $t \in { \mathcal { T } }$ do   
4: Sample $\epsilon _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
5: $\mathbf { x } _ { t } \gets \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t }$ ⊳ Direct perturbation   
6: $\hat { \epsilon } _ { t } ^ { \mathrm { p r 1 0 r } } \gets \epsilon _ { \theta , \phi } ( \mathbf { x } _ { t } , t | \boldsymbol { \mathcal { D } } )$   
7: $\widehat { \epsilon } _ { t } ^ { \mathrm { c o n d } } \gets \epsilon _ { \theta , \phi } ( \mathbf { x } _ { t } , t \mid \mathbf { c } )$   
8: $\mathbf { d } _ { t } \gets \widehat { \epsilon } _ { t } ^ { \mathrm { c o n d } } - \widehat { \epsilon } _ { t } ^ { \mathrm { p r i o } }$   
9: � ← � + log $\left( [ \mathbf { d } _ { t } ] _ { + } + \zeta _ { \mathrm { s t a b } } \right)$ ⊳ Pathological directional prior   
10: end for   
11: $\mathbf { A } _ { \mathrm { d i f f } }  \mathrm { e x p } \big ( \mathbf { A } / | \mathcal { T } | \big ) \odot \mathbf { B }$ ⊳ Geometric mean   
12: return $\mathbf { A } _ { \mathrm { d i f f } }$ ⊳ No sequential reverse chain

Lower MASD indicates closer average agreement between the predicted and reference boundaries.

Normalized surface distance (NSD) measures the proportion of surface voxels within a predefined dilated tolerance � of the corresponding surface:

$$
\mathrm { N S D } _ { \tau } ( P , G ) = \frac { \displaystyle \sum _ { x \in S _ { P } } \mathbf { 1 } \big [ d ( x , S _ { G } ) \leq \tau \big ] + \displaystyle \sum _ { y \in S _ { G } } \mathbf { 1 } \big [ d ( y , S _ { P } ) \leq \tau \big ] } { | S _ { P } | + | S _ { G } | } ,\tag{42}
$$

where �[⋅] is the indicator function. We used a tolerance of $\tau = 2 . 0$ mm in the resampled image space, following Rao et al. (2025). NSD ranges from 0 to 1, with higher values indicating better surface agreement within the specified tolerance.

## 3.10. Statistical Analysis

Statistical comparisons were performed using two-sided paired Wilcoxon signed-rank tests, with each test subject serving as the paired unit. Dice was the primary outcome for comparisons between the proposed method and competing baselines, as well as for selected ablation analyses. Statistical significance was assessed at $\alpha \ : = \ : 0 . 0 5$ . Mean paired Dice diferences were reported alongside $p \mathrm { - }$ -values to describe the magnitude of the observed efects.

For methods evaluated using ceiling Dice, subject-level scores were computed at the same method-specific threshold selected to maximize mean test-set Dice. These statistical comparisons therefore characterize performance under the oracle threshold evaluation protocol rather than at independently selected operating thresholds.

## 3.11. Alternative Discrepancy Formulations for Ablation

To assess the design choices underlying the proposed discrepancy score, we consider two alternative formulations. These formulations are evaluated only as ablation variants and are not used in the final framework. The evaluation of the ablation variants is in Table 6.

Reverse-chain accumulation. Our final method evaluates each selected noise level independently by directly sampling $x _ { t }$ using Eq. (2). As an alternative, we construct a dependent reverse trajectory in which each state is generated from the preceding reverse step:

$$
\mathbf { x } _ { t - 1 } ^ { \mathrm { c h a i n } } = \mu _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } ^ { \mathrm { c h a i n } } , t ) + \sigma _ { t } \mathbf { z } _ { t } , \quad \mathbf { z } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) .\tag{43}
$$

The discrepancy at each timestep is then evaluated on $\mathbf { x } _ { t } ^ { \mathrm { c h a i n } }$ This variant process tests whether preserving a continuous reverse trajectory is preferable to independently examining matched corruption levels of the observed image.

Forward-noise reference. We additionally replace the unconditional model prediction with the Gaussian noise used in the forward process:

$$
\begin{array} { r } { \widetilde { d } _ { t } = \epsilon ^ { \mathrm { c o n d } } ( \mathbf { x } _ { t } , t \mid c = \mathbf { x } _ { 0 } ) - \epsilon _ { t } , } \\ { \mathrm { w h e r e } \ x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t } \ } \end{array}\tag{44}
$$

This variant formulation tests whether anomaly evidence can be obtained from conditional noise prediction error alone, without using the learned unconditional response as a healthy reference.

## 4. Results

## 4.1. Unsupervised Benchmark Results

Table 1 compares the proposed method with existing state-of-the-arts unsupervised anomaly detection approaches. The compared approaches were trained over the same training subjects and used their original proposed postprocessing smooth methods. Among the evaluated baselines, f-AnoGAN (Schlegl et al., 2019) failed to converge reliably and AnoDDPM (Wyatt et al., 2022) remained unstable when trained with Simplex noise. Therefore, quantitative results are not reported for these two methods.

Without spatial filtering, our method achieved a Dice score of 0.232, significantly exceeding the best-performing converged baselines, MAD-AD (Beizaee et al., 2025a) with Dice 0.166 (two-sided paired Wilcoxon signed-rank test $p <$ $0 . 0 0 1 , n = 4 4 )$ . It also obtained an NSD of 0.284, compared with the best baseline MAD-AD’s NSD of 0.213. The corresponding HD and MASD were 29.29 mm and 12.62 mm, respectively, representing the second-best performance for both distance-based metrics after MAD-AD. The unfiltered anomaly map achieved an AP of0.374, which was lower than those of DAE (0.665) and MAD-AD (0.492), but higher than the remaining difusion-based and feature-based baselines.

Applying mean filtering with a kernel size of $k = 5$ consistently improved the proposed method across all reported metrics. Dice significantly increased from 0.232 to 0.283 (two-sided paired Wilcoxon signed-rank test $p ~ = ~ 0 . 0 0 1$ � = 44). AP increased from 0.374 to 0.505, while HD decreased from 29.29 to 25.45 mm, and MASD decreased from 12.62 to 9.40 mm. NSD also increased from 0.284 to 0.355. The mean-filtered model therefore achieved the best Dice, HD, MASD, and NSD among all successfully trained methods.

Relative to MAD-AD, the strongest competing baseline for spatial localization, the mean-filtered model significantly improved Dice by 0.117 (two-sided paired Wilcoxon signedrank test $p ~ < ~ 0 . 0 0 1 , ~ n ~ = ~ 4 4 )$ , reduced HD by 1.99 mm, reduced MASD by 1.93 mm, and increased NSD by 0.142. Although DAE (Kascenas et al., 2022) achieved the highest AP of 0.665, its Dice 0.159 and NSD 0.176 were substantially lower than those of our proposed mean-filtered method. Overall, these findings indicate that the proposed approach provides more accurate lesion overlap and boundary localization, while DAE retains an advantage in global voxel-ranking performance.

To assess the efect of lesion burden on detection performance, we further stratified the test subjects according to their ground-truth lesion percentage into small (< 1%), medium (1%–5%), and large (≥ 5%) lesion-burden groups, as following Rao et al. (2025) in Table 2. Before mean filtering, performance improved consistently with increasing lesion burden: Dice increased from 0.080 for the smalllesion group to 0.287 and 0.603 for the medium- and largelesion groups, respectively. Correspondingly, HD decreased from 40.27 mm to 26.39 mm and 10.23 mm, while NSD increased from 0.155 to 0.388 and 0.640.

Mean filtering provided the greatest benefit for subjects with small and medium lesion burdens (Table 2, gray-shaded rows). For the small-lesion group, mean filtering with kernel size of 5 (MF5) increased Dice from 0.080 to 0.141, AP from 0.030 to 0.135, and NSD from 0.155 to 0.262. For the medium-lesion group, Dice increased from 0.287 to 0.377, AP from 0.186 to 0.356, and NSD from 0.388 to 0.487. In contrast, performance for the large-lesion group was already comparatively high before filtering, and MF5 produced only a small increase in AP (0.541 to 0.562) and HD improvement (10.23 to 10.13 mm), while Dice remained nearly unchanged (0.603 versus 0.600).

![](images/9496375ec082b1bacc05e704b628d00820d328fec022f5eefebe10d51fde068a.jpg)  
Figure 3: Qualitative comparison of unsupervised anomaly detection methods in subjects with (a) small (< 1%), (b) medium (1% − 5%), and (c) large (≥ 5%) lesion burdens, defined by the subject-level lesion percentage. Columns show the input ADC image, expert lesion annotation, and anomaly maps from selected top-performing baselines, including DAE (Kascenas et al., 2022), ANDi (Frotscher et al., 2026), MAD-AD (Beizaee et al., 2025a), where the originally proposed smoothing methods from each paper have been applied. Our proposed method before and after mean filtering (MF; � = 5) is highlighted in green box. Red overlays in the ground truth (GT) column indicate annotated lesions, and cyan contours delineate their boundaries on the anomaly maps. Brighter colors indicate higher anomaly scores.

Table 1  
Quantitative comparison of unsupervised anomaly detection methods for neonatal HIE lesion localization. Values are reported as mean ± standard deviation where available. The best and second-best results are shown in bold and underlined, respectively. For each model’s result, the originally proposed smoothing methods have been applied. For Dice, <sup>†</sup> and <sup>‡</sup> indicate significantly lower scores than Ours and Ours with mean filtering (� = 5), respectively (under two-sided paired Wilcoxon signed-rank tests, all $p < 0 . 0 0 1 , n = 4 4 )$
<table><tr><td>Method</td><td>Dice ↑</td><td>AP ↑</td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td>MASD ↓(mm)</td><td>NSD ↑</td></tr><tr><td>f-AnoGAN (Schlegl et al., 2019)</td><td colspan="5">Unstable training and unable to achieve convergence.</td></tr><tr><td>DAE (Kascenas et al., 2022)</td><td> $0 . 1 5 9 \pm 0 . 2 6 8 ^ { \dagger \ddagger }$ </td><td>0.665</td><td> $3 0 . 3 8 \pm 1 5 . 4 1$ </td><td> $1 3 . 6 5 \pm 9 . 8 7$ </td><td> $0 . 1 7 6 \pm 0 . 2 1 0$ </td></tr><tr><td>ÀnoDDPM (Wyatt et al., 2022)</td><td colspan="5">Unstable training with simplex noise.</td></tr><tr><td>AutoDDPM (Bercea et al., 2023)</td><td> $0 . 0 8 5 \pm 0 . 1 2 6 ^ { \dagger \ddagger }$ </td><td>0.083</td><td> $3 1 . 9 6 \pm 1 5 . 5 9$ </td><td> $1 4 . 5 3 \pm 9 . 7 3$ </td><td> $0 . 1 6 9 \pm 0 . 1 9 4$ </td></tr><tr><td>pDDPM (Behrendt et al., 2024)</td><td> $0 . 1 2 1 \pm 0 . 2 3 5 ^ { \dagger \ddagger }$ </td><td>0.084</td><td> $3 9 . 0 8 \pm 1 5 . 0 5$ </td><td> $2 1 . 1 6 \pm 1 1 . 0 8$ </td><td> $0 . 0 5 0 \pm 0 . 1 1 1$ </td></tr><tr><td>THOR (Bercea et al., 2024)</td><td> $0 . 1 6 2 \pm 0 . 2 3 1 ^ { \dagger \ddagger }$ </td><td>0.281</td><td> $3 1 . 8 9 \pm 1 5 . 5 6$ </td><td> $1 4 . 3 5 \pm 9 . 6 8$ </td><td> $0 . 1 9 6 \pm 0 . 2 1 8$ </td></tr><tr><td>IterMask² (Liang et al., 2024)</td><td> $0 . 1 2 1 \pm 0 . 2 3 5 ^ { \dagger \ddagger }$ </td><td>0.050</td><td> $3 8 . 8 7 \pm 1 5 . 3 3$ </td><td> $2 1 . 0 9 \pm 1 1 . 1 1$ </td><td> $0 . 0 5 0 \pm 0 . 1 1 1$ </td></tr><tr><td>MAD-AD (Beizaee et al., 2025a)</td><td> $0 . 1 6 6 \pm 0 . 2 0 5 ^ { \dagger \ddagger }$ </td><td>0.492</td><td> $2 7 . 4 4 \pm 1 3 . 2 5$ </td><td> ${ underline { { 1 1 . 3 3 \pm 7 . 8 4 } } } $ </td><td> $0 . 2 1 3 \pm 0 . 1 8 7$ </td></tr><tr><td>DeCo-Diff (Beizaee et al., 2025b)</td><td> $0 . 1 3 1 \pm 0 . 2 1 4 ^ { \dagger \ddagger }$ </td><td>0.261</td><td> $3 0 . 7 9 \pm 1 4 . 8 4$ </td><td> $1 4 . 8 6 \pm 9 . 5 1 $ </td><td> $0 . 1 5 4 \pm 0 . 1 8 9$ </td></tr><tr><td>ANDi (Frotscher et al., 2026)</td><td> $0 . 1 0 8 \pm 0 . 1 9 9 ^ { \dagger \ddagger }$ </td><td>0.267</td><td> $3 3 . 0 9 \pm 1 4 . 7 9$ </td><td> $1 2 . 8 5 \pm 8 . 3 7$ </td><td> $0 . 1 3 9 \pm 0 . 1 7 9$ </td></tr><tr><td>InvAD + U-Net (Sakai et al., 2026)</td><td> $0 . 1 3 1 \pm 0 . 2 3 5 ^ { \dagger \ddagger }$ </td><td>0.174</td><td> $3 2 . 4 7 \pm 1 5 . 9 5$ </td><td> $1 6 . 2 6 \pm 1 0 . 0 2$ </td><td> $0 . 1 3 6 \pm 0 . 2 0 3$ </td></tr><tr><td>InvAD + original setup (Sakai et al., 2026)</td><td> $0 . 1 2 1 \pm 0 . 2 3 5 ^ { \dagger \ddagger }$ </td><td>0.107</td><td> $3 9 . 0 9 \pm 1 5 . 0 5$ </td><td> $2 1 . 1 7 \pm 1 1 . 0 8$ </td><td> $0 . 0 5 0 \pm 0 . 1 1 1 2$ </td></tr><tr><td>Ours</td><td> $0 . 2 3 2 \pm 0 . 2 3 9$ </td><td>0.374</td><td> $2 9 . 2 9 \pm 1 6 . 4 3$ </td><td> $1 2 . 6 2 \pm 1 0 . 1 1$ </td><td> $0 . 2 8 4 \pm 0 . 2 5 6$ </td></tr><tr><td>Ours + mean filtering (k = 5)</td><td> $\mathbf { 0 . 2 8 3 \pm 0 . 2 4 4 }$ </td><td>0.505</td><td> $\mathbf { 2 5 . 4 5 \pm 1 5 . 6 3 }$ </td><td> ${ \bf 9 . 4 0 \pm 8 . 4 9 }$ </td><td> $\mathbf { 0 . 3 5 5 \pm 0 . 2 3 6 }$ </td></tr></table>

Performance of our proposed model stratified by subject-level lesion burden. Subjects were divided into small $( < 1 \% )$ , medium (1%–5%), and large (≥ 5%) lesion-burden groups according to the percentage of voxels labeled as lesions in each subject. MF5 denotes mean filtering with a kernel size of 5 (rows shaded in gray). Results are reported as mean ± standard deviation where available, and the best result within each lesion-burden group is highlighted in bold.
<table><tr><td>Lesion burden</td><td>Post-processing</td><td>Dice ↑</td><td>AP ↑</td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td> $\mathbf { \boldsymbol { \mathsf { M } } } \mathbf { \boldsymbol { A } } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { D } } \downarrow \left( \mathbf { \boldsymbol { m } } \mathbf { \boldsymbol { m } } \right)$ </td><td>NSD ↑</td></tr><tr><td rowspan="2">Small  $\big ( < 1 \% \big ) , n = 2 2$ </td><td></td><td> $0 . 0 8 0 \pm 0 . 0 8 6$ </td><td>0.030</td><td> $4 0 . 2 7 \pm 1 3 . 9 0$ </td><td> $1 7 . 2 5 \pm 9 . 5 8$ </td><td> $0 . 1 5 5 \pm 0 . 1 1 1$ </td></tr><tr><td>MF5</td><td> $\mathbf { 0 . 1 4 1 \pm 0 . 1 5 4 }$ </td><td>0.135</td><td> $\mathbf { 3 3 . 6 0 \pm 1 5 . 1 9 }$ </td><td> ${ \bf 1 0 . 4 9 \pm 5 . 6 4 }$ </td><td> $\mathbf { 0 . 2 6 2 \pm 0 . 2 0 8 }$ </td></tr><tr><td rowspan="2">Medium (1%–5%), n = 12</td><td></td><td> $0 . 2 8 7 \pm 0 . 1 4 8$ </td><td>0.186</td><td> $2 6 . 3 9 \pm 8 . 7 7$ </td><td> $7 . 8 0 \pm 3 . 5 3 $ </td><td> $0 . 3 8 8 \pm 0 . 1 5 0$ </td></tr><tr><td>MF5</td><td> $\mathbf { 0 . 3 7 7 \pm 0 . 2 0 8 }$ </td><td>0.356</td><td> ${ \bf 1 8 . 4 3 \pm 6 . 9 2 }$ </td><td> ${ \pm } \mathord { \Delta } . 4 7 \pm 3 . 5 4$ </td><td> $\mathbf { 0 . 4 8 7 \pm 0 . 2 3 2 }$ </td></tr><tr><td rowspan="2">Large (≥ 5%), n = 10</td><td></td><td> $\mathbf { 0 . 6 0 3 \pm 0 . 1 4 7 }$ </td><td>0.541</td><td> $1 0 . 2 3 \pm 5 . 0 3$ </td><td> $\mathbf { 2 . 6 8 \pm 1 . 5 9 }$ </td><td> $\mathbf { 0 . 6 4 0 \pm 0 . 1 6 0 }$ </td></tr><tr><td>MF5</td><td> $0 . 6 0 0 \pm 0 . 1 6 7$ </td><td>0.562</td><td> ${ \bf 1 0 . 1 3 \pm 4 . 7 5 }$ </td><td> $2 . 7 6 \pm 0 . 9 3$ </td><td> $0 . 5 8 2 \pm 0 . 1 3 7$ </td></tr></table>

Fig. 3 compares anomaly maps from selected top-performin baselines with those produced by our method in subjects with diferent lesion burdens. In the illustrated cases, DAE (Kascenas et al., 2022) produces spatially extensive responses beyond the annotated lesions, whereas ANDi (Frotsche et al., 2026) exhibits scattered responses with limited separation between lesion and non-lesion tissue. MAD-AD (Beizaee et al., 2025a) produces more localized responses but also highlights regions outside the lesion annotations, particularly in the small-lesion example. Our method shows closer spatial correspondence with the annotated lesions in the medium- and large-burden examples, although scattered responses remain outside the lesion boundaries. Mean filtering reduces isolated responses and produces smoother anomaly maps while preserving the lesion associated responses. The small-lesion example remains challenging, where the lesion is detected but is less noticeable with surrounding false positive responses. Some apparent false positive responses occur in regions with relatively low ADC values or interhemispheric asymmetry in ADC intensity, indicating that the directional constraint does not fully suppress responses outside the annotated lesions.

## 4.2. Comparison with Supervised ADC-Only Models

Table 3 compares the proposed unsupervised anomaly detection method with supervised Swin-UNETR (Almansour et al., 2025) and UNet++ (Rao et al., 2025) segmentation models under matched ADC-only and population �- map enhanced settings, which are the top performed models from the BONBID-HIE challenge (Bao et al., 2025). In the ADC-only setting (the gray-shaded rows in Table 3), our method achieved a Dice score of 0.232, which was comparable to the supervised UNet++ model of 0.237, and an NSD of 0.284, approaching the UNet++ NSD of 0.298. Swin-UNETR achieved highest ADC-only Dice and NSD values of 0.311 and 0.406, respectively.

Despite the comparable overlap with UNet++, the ADConly anomaly detection model remained less competitive in voxel-ranking and boundary-distance performance. It achieved an AP of 0.374, compared with 0.645 for Swin-UNETR and 0.803 for UNet++. Similarly, its HD of 29.29 mm and MASD of 12.62 mm were higher than those of both supervised models. These findings show that the proposed method can approach a supervised segmentation network in terms of lesion overlap and tolerance-based surface agreement without being optimized using a voxelwise segmentation loss, although a gap remains in anomaly score ranking and precise boundary localization.

## 4.3. Efect of Combining Population-Derived ADC �-Map Information

We additionally evaluated an optional post-processing procedure incorporating a population-derived ADC �-map (Table 3). For the supervised models, the continuous �- map and, a thresholded $Z \le - 2 . 0$ mask (Rao et al., 2025) were supplied as additional input channels during training and inference. In contrast, our difusion model remained ADC-only, and the �-map was used exclusively during postprocessing to identify a high-confidence low-ADC core and to retain nearby difusion-supported components. We also evaluated a �-map-only baseline using $Z \le - 2 . 0$ for binary segmentation.

The �-map-only baseline achieved a Dice score of 0.583, indicating that population-derived ADC information alone provides strong lesion localization performance. Incorporating our difusion-derived anomaly information increased Dice to 0.602 and NSD from 0.641 to 0.667, corresponding to absolute gains of 0.019 and 0.026, respectively. HD decreased slightly from 19.40 to 19.11 mm, whereas MASD remained essentially unchanged (4.72 versus 4.73 mm). These results demonstrate a modest but statistically significant improvement in Dice (two-sided paired Wilcoxon signed-rank test $p \ = \ 0 . 0 0 2 0 , n \ = \ 4 4 )$ accompanied by a numerical increase in surface agreement over the �-map-only baseline.

Relative to the ADC-only difusion prediction, postprocessing increased Dice from 0.232 to 0.602 and NSD from 0.284 to 0.667. The post-processing’s Dice exceeded those of Swin-UNETR and UNet++ using ADC and the continuous �-map (0.553 and 0.578, respectively), but remained below their configurations additionally incorporating the thresholded binary �-mask (0.657 and 0.609). Its NSD of 0.667 was slightly below the range achieved by the �-map assisted supervised models (0.674–0.724). AP was not reported for our post-processing procedure because it directly produces a binary lesion mask.

Efect of incorporating population-derived ADC �-map information. ADC-only models serve as the reference configurations (rows shaded in gray). For the supervised models, the �-map and a thresholded binary �-mask $( Z \le - 2 . 0 )$ are provided as additional input channels (Bao et al., 2025; Rao et al., 2025; Almansour et al., 2025). For the population-derived statistical baseline, −� is used as the continuous anomaly score for AP, whereas $Z \le - 2 . 0$ is used to obtain the binary mask for Dice and surface-distance metrics. Our unsupervised method uses the �-map only during post processing. AP is not reported for this post-processing result because it directly produces a binary lesion mask. The best result within each model is shown in bold.
<table><tr><td>Model</td><td>Input or refinement</td><td>Dice ↑</td><td> $\mathsf { \pmb { A } } \mathsf { \pmb { P } } \mathrm { \dag }$ </td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td> $\mathsf { M A S D \perp ( m m ) }$ </td><td>NSD ↑</td></tr><tr><td colspan="7">Supervised segmentation methods</td></tr><tr><td>Swin-UNETR</td><td>ADC only</td><td> $0 . 3 1 1 \pm 0 . 2 5 1$ </td><td>0.645</td><td> $2 2 . 5 6 \pm 1 3 . 3 9$ </td><td> $8 . 3 1 \pm 8 . 3 8$ </td><td> $0 . 4 0 6 \pm 0 . 2 4 7$ </td></tr><tr><td>(Almansour et al., 2025)</td><td> $\mathsf { A D C } + Z \mathsf { - m a p }$ </td><td> $0 . 5 5 3 \pm 0 . 2 6 0$ </td><td>0.918</td><td> ${ \bf 1 6 . 0 7 \pm 1 2 . 0 0 }$ </td><td> $4 . 1 0 \pm 6 . 3 1$ </td><td> $0 . 6 7 4 \pm 0 . 2 5 6$ </td></tr><tr><td></td><td> $\mathsf { A D C } + Z \mathsf { - m a p } + Z \leq - 2 . 0$ </td><td> $\mathbf { 0 . 6 5 7 \pm 0 . 2 5 8 }$ </td><td>0.927</td><td> $1 8 . 3 2 \pm 1 5 . 5 7$ </td><td> ${ \bf 3 . 8 9 \pm 5 . 4 6 }$ </td><td> $\mathbf { 0 . 7 2 4 \pm 0 . 2 4 2 }$ </td></tr><tr><td>UNet++</td><td> $\mathsf { A D C \circ n l y }$ </td><td> $0 . 2 3 7 \pm 0 . 2 6 0$ </td><td>0.803</td><td> $2 3 . 5 8 \pm 1 2 . 8 0$ </td><td> $9 . 6 8 \pm 8 . 3 7$ </td><td> $0 . 2 9 8 \pm 0 . 2 3 5$ </td></tr><tr><td>(Rao et al., 2025)</td><td> $\mathsf { A D C } + Z \mathsf { - m a p }$ </td><td> $0 . 5 7 8 \pm 0 . 2 4 6$ </td><td>0.915</td><td> ${ \bf 1 5 . 8 8 \pm 1 2 . 7 7 }$ </td><td> $3 . 8 4 \pm 5 . 5 5$ </td><td> $0 . 6 9 4 \pm 0 . 2 4 9$ </td></tr><tr><td></td><td> $\mathsf { A D C } + Z \mathsf { - m a p } + Z \leq - 2 . 0$ </td><td> $\mathbf { 0 . 6 0 9 \pm 0 . 2 5 0 }$ </td><td>0.913</td><td> $1 7 . 5 1 \pm 1 4 . 4 7$ </td><td> ${ \bf 3 . 7 1 \pm 4 . 7 1 }$ </td><td> $\mathbf { 0 . 7 0 0 \pm 0 . 2 4 1 }$ </td></tr><tr><td colspan="7">Population-derived statistical reference</td></tr><tr><td>Population ADC Z-map</td><td>Z-map only  $( Z \leq - 2 . 0 \ \mathrm { f o r \ s e g m e n t a t i o n } )$ </td><td> $0 . 5 8 3 \pm 0 . 2 6 8$ </td><td>0.922</td><td> $1 9 . 4 0 \pm 1 3 . 4 5$ </td><td> $4 . 7 2 \pm 6 . 1 7$ </td><td> $0 . 6 4 1 \pm 0 . 2 5 6$ </td></tr><tr><td colspan="7">Unsupervised anomaly detection method</td></tr><tr><td></td><td> $\mathsf { A D C \circ n l y }$ </td><td> $0 . 2 3 2 \pm 0 . 2 3 9$ </td><td>0.374</td><td> $2 9 . 2 9 \pm 1 6 . 4 3$ </td><td> $1 2 . 6 2 \pm 1 0 . 1 1$ </td><td> $0 . 2 8 4 \pm 0 . 2 5 6$ </td></tr><tr><td>Ours</td><td> $\mathsf { A D C } + Z \mathsf { - m a p }$  post-processing</td><td> $\mathbf { 0 . 6 0 2 \pm 0 . 2 6 9 }$ </td><td> $--$ </td><td> $\mathbf { 1 9 . 1 1 \pm 1 9 . 0 0 }$ </td><td> $\mathbf { 4 . 7 3 \pm 6 . 4 0 }$ </td><td> $\mathbf { 0 . 6 6 7 \pm 0 . 2 6 2 }$ </td></tr></table>

Ablation study of the proposed framework. All compared models applied the HIE pathology informed directional information. Results are reported as mean ± standard deviation where applicable. The best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Configuration</td><td>Dice ↑</td><td> $\mathsf { \pmb { A } } \mathsf { \pmb { P } } \mathrm { \dag }$ </td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td> $\mathbf { \boldsymbol { \mathsf { M } } } \mathbf { \boldsymbol { A } } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { D } } \downarrow \left( \mathbf { \boldsymbol { m } } \mathbf { \boldsymbol { m } } \right)$ </td><td>NSD ↑</td></tr><tr><td>DDPM</td><td> $0 . 1 6 7 \pm 0 . 2 3 6$ </td><td>0.235</td><td> $3 2 . 0 3 \pm 1 6 . 6 2$ </td><td> $1 4 . 5 8 \pm 1 0 . 0 3$ </td><td> $0 . 2 1 2 \pm 0 . 2 4 2$ </td></tr><tr><td>+ ControlNet (healthy-restoration,  ${ \mathsf { C F G } } { = } 0 . 8 )$ </td><td> $0 . 1 2 1 \pm 0 . 1 9 0$ </td><td>0.414</td><td> $3 9 . 4 4 \pm 1 6 . 0 3$ </td><td> $1 7 . 4 7 \pm 1 0 . 1 9$ </td><td> $0 . 1 6 9 \pm 0 . 1 9 3$ </td></tr><tr><td>+ ControlNet (lesion-emphasizing,  ${ \mathsf { C F G } } { = } 7 . 5 )$ </td><td> $0 . 1 3 8 \pm 0 . 2 4 2$ </td><td>0.332</td><td> $3 1 . 3 0 \pm 1 6 . 2 9$ </td><td> $1 5 . 3 1 \pm 1 0 . 3 7$ </td><td> $0 . 1 6 0 \pm 0 . 2 2 2$ </td></tr><tr><td>+ Multi noise-level discrepancy accumulation</td><td> $\underline { { 0 . 2 3 2 \pm 0 . 2 3 9 } }$ </td><td>0.374</td><td> $\underline { { 2 9 . 2 9 \pm 1 6 . 4 3 } }$ </td><td> $1 2 . 6 2 \pm 1 0 . 1 1$ </td><td> $\underline { { 0 . 2 8 4 \pm 0 . 2 5 6 } }$ </td></tr><tr><td>+ Mean filtering  $\left( k = 5 \right)$ </td><td> $\mathbf { 0 . 2 8 3 \pm 0 . 2 4 4 }$ </td><td>0.505</td><td> $\mathbf { 2 5 . 4 5 \pm 1 5 . 6 3 }$ </td><td> ${ \bf 9 . 4 0 \pm 8 . 4 9 }$ </td><td> $\mathbf { 0 . 3 5 5 \pm 0 . 2 3 6 }$ </td></tr></table>

Fig. 4 illustrates the efect of post-processing on the ADC-only difusion prediction in a representative case. Many spatially isolated responses are removed, while the main lesion-associated regions are retained. The resulting segmentation broadly follows regions with low ADC $Z -$ values and overlaps with the expert annotations outlined in cyan.

## 4.4. Ablation Analysis of Framework Components

Table 4 evaluates the contributions of the principal components of the proposed framework. The conventional DDPM baseline achieved a Dice score of 0.167, an AP of 0.235, and an NSD of 0.212, with an HD of 32.03 mm and an MASD of 14.58 mm.

Introducing subject-specific conditioning through ControlNet and deriving the anomaly map from a single CFGbased reconstruction (described in Sec. 3.3) did not result in consistent improvements across the evaluation metrics. With a lesion-recovery setting $w = 0 . 8 ,$ AP increased from 0.235 to 0.414. However, Dice, HD, MASD, and NSD deteriorated. This result indicates that reducing the contribution of the subject condition can increase voxel-wise sensitivity to abnormal regions, while simultaneously introducing reconstruction diferences that are spatially inconsistent with the lesion boundaries. Increasing the guidance scale to $w = 7 . 5 ,$ corresponding to the lesion-strengthening setting, partially improved Dice and the surface-distance metrics relative to $w = 0 . 8 .$ , but reduced AP to 0.332 and did not provide a consistent advantage over the DDPM baseline. These findings demonstrate the limited and inconsistent lesion sensitivity of anomaly maps derived from a single reconstructed image.

![](images/3a40fa21da539670424b53d3eb19d2ecf0d23d8b12511ff8bcef12c641078d10.jpg)  
Figure 4: Qualitative comparison of reconstruction-based anomaly detection and the proposed discrepancy-based approach. (a) Input ADC image and expert lesion annotation. (b–d) Reconstructed images and corresponding anomaly maps obtained using DDPM and DDPM with ControlNet at CFG scales of 0.8 and 7.5. (e) Anomaly maps produced by the proposed method before and after mean filtering (MF, � = 5). (f) The corresponding population-derived Z-map for the input and the Z-map guided post-processing lesion refinement. Cyan contours delineate the expert-annotated lesion boundaries. All methods’ results have been applied with HIE pathological informed sign direction.

The proposed multi noise-level discrepancy accumulation directly compares the conditional and unconditional noise predictions and aggregates their directionally consistent diferences across the selected reverse difusion steps. This component significantly improved Dice from 0.167 to 0.232 (two-sided paired Wilcoxon signed-rank test $p \ <$ 0.001, $\ n \ = \ 4 4 )$ , reduced HD from 32.03 to 29.29 mm, and reduced MASD from 14.58 to 12.62 mm. NSD also increased from 0.212 to 0.284. Although its AP of 0.374 was lower than that obtained with the $w = 0 . 8$ reconstruction, the consistently improved overlap and surface-distance metrics indicate more accurate spatial localization of the detected abnormalities. Importantly, because the discrepancy is computed directly from $\epsilon ^ { \mathrm { c o n d } } - \epsilon ^ { \mathrm { u n c o n d } }$ , its magnitude is independent of the CFG scale �.

Finally, applying mean filtering with a kernel size of � = 5 produced the best performance across all metrics. This improvement suggests that spatial smoothing suppresses isolated high-frequency responses while preserving spatially coherent lesion evidence.

Fig. 4 illustrates the diferences between reconstructionbased anomaly maps and our discrepancy-based approach in a representative case. DDPM alone produces substantial anatomical deviations from the input image, whereas ControlNet conditioning better preserves subject-specific anatomy. However, both DDPM and ControlNet-guided reconstruction with $\mathrm { C F G } = \ 0 . 8$ produce weak anomaly responses within the annotated lesion region. Increasing the guidance scale to $\mathrm { C F G } = \ 7 . 5$ produces higher anomaly scores overall, making the lesion region more noticeable. However, elevated scores also occur in surrounding nonlesion tissue, limiting spatial specificity. In contrast, our method produces more spatially confined responses that follow the annotated lesion distribution, although some lesion regions remain weakly highlighted and scattered responses persist outside the annotation. Mean filtering reduces isolated responses and produces a smoother map while retaining the main lesion-associated pattern. These observations complement the quantitative ablation results and illustrate the benefit of aggregating conditional–unconditional discrepancies across noise levels rather than relying on a final reconstruction residual.

## 4.5. Ablation Analysis of Mean Filtering Kernel Size

Table 5 evaluates the efect of the mean filtering kernel size on the direction-aware anomaly map. Here, $k \ = \ 0$ denotes the unfiltered anomaly map. Applying a moderate degree of spatial smoothing consistently improved performance over the unfiltered result. Increasing the kernel size from $k = 0$ to $k = 3$ improved Dice from 0.232 to 0.262, AP from 0.374 to 0.481 and NSD from 0.284 to 0.324, while reducing HD from 29.29 to 27.24 mm and MASD from 12.62 to 11.01 mm. Further increasing the kernel size to $k \ = \ 5$ yielded the highest AP (0.505), the lowest HD (25.45 mm), the lowest MASD (9.40 mm), and the highest

Ablation study of the mean-filter kernel size applied to the direction-aware anomaly map. Kernel size $k = 0$ indicates that no mean filtering was applied. The gray row denotes the kernel size used in the main experiments. Results are reported as mean ± standard deviation where applicable, and the best value for each metric is highlighted in bold.
<table><tr><td>Kernel size k</td><td>Dice ↑</td><td> $\mathsf { \pmb { A } } \mathsf { \pmb { P } } \mathrm { \dag }$ </td><td> $\mathsf { H D } \downarrow ( \mathsf { m m } )$ </td><td> $\mathbf { \boldsymbol { \mathsf { M } } } \mathbf { \boldsymbol { A } } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { D } } \downarrow \left( \mathbf { \boldsymbol { m } } \mathbf { \boldsymbol { m } } \right)$ </td><td>NSD ↑</td></tr><tr><td>0</td><td> $0 . 2 3 2 \pm 0 . 2 3 9$ </td><td>0.374</td><td> $2 9 . 2 9 \pm 1 6 . 4 3$ </td><td> $1 2 . 6 2 \pm 1 0 . 1 1$ </td><td> $0 . 2 8 4 \pm 0 . 2 5 6$ </td></tr><tr><td>3</td><td> $0 . 2 6 2 \pm 0 . 2 4 5$ </td><td>0.481</td><td> $2 7 . 2 4 \pm 1 6 . 3 7$ </td><td> $1 1 . 0 1 \pm 9 . 7 4$ </td><td> $0 . 3 2 4 \pm 0 . 2 5 0$ </td></tr><tr><td>5</td><td> $0 . 2 8 3 \pm 0 . 2 4 4$ </td><td>0.505</td><td> $\mathbf { 2 5 . 4 5 \pm 1 5 . 6 3 }$ </td><td> ${ \bf 9 . 4 0 \pm 8 . 4 9 }$ </td><td> $\mathbf { 0 . 3 5 5 \pm 0 . 2 3 6 }$ </td></tr><tr><td>7</td><td> $\mathbf { 0 . 2 8 7 \pm 0 . 2 5 2 }$ </td><td>0.503</td><td> $2 6 . 2 0 \pm 1 5 . 4 7$ </td><td> $9 . 5 8 \pm 8 . 4 3$ </td><td> $0 . 3 4 8 \pm 0 . 2 2 9$ </td></tr><tr><td>8</td><td> $0 . 2 8 2 \pm 0 . 2 5 5$ </td><td>0.497</td><td> $2 6 . 5 1 \pm 1 5 . 1 4$ </td><td> $9 . 7 4 \pm 8 . 3 8$ </td><td> $0 . 3 3 1 \pm 0 . 2 2 4$ </td></tr><tr><td>10</td><td> $0 . 2 7 0 \pm 0 . 2 5 9$ </td><td>0.485</td><td> $2 8 . 1 6 \pm 1 6 . 7 6$ </td><td> $1 0 . 9 7 \pm 9 . 5 2$ </td><td> $0 . 3 0 4 \pm 0 . 2 2 2$ </td></tr></table>

Ablation of the anomaly-score representation compared with sequential reverse propagation and discrepancy to forward process noise. “Independent” indicates that each noise level was sampled directly from the original patient ADC image without propagating a reverse-difusion chain, and “Reverse chain” evaluates successive states along a sequential reverse-difusion trajectory (Eq. (43)). The final row replaces the learned unconditional noise prediction with the Gaussian noise $\epsilon _ { t }$ sampled during forward corruption (Eq. (44)). The biological directional constraint was enabled for all configurations. Rows shaded in gray use a mean filtering with kernel size of 5 (MF5). Values are reported as mean ± standard deviation where available, and the best result for each metric is highlighted in bold.
<table><tr><td>Score representation</td><td>Evaluation</td><td>Dice ↑</td><td> $\mathsf { \pmb { A } } \mathsf { \pmb { P } } \mathrm { \dag }$ </td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td> $\mathsf { M A S D \perp ( m m ) }$ </td><td>NSD ↑</td></tr><tr><td> $\epsilon _ { t } ^ { \mathsf { c o n d } } - \epsilon _ { t } ^ { \mathsf { u n c o n d } }$ </td><td>Independent</td><td> $0 . 2 3 2 \pm 0 . 2 3 9$ </td><td>0.374</td><td> $2 9 . 2 9 \pm 1 6 . 4 3$ </td><td> $1 2 . 6 2 \pm 1 0 . 1 1$ </td><td> $0 . 2 8 4 \pm 0 . 2 5 6$ </td></tr><tr><td> $\epsilon _ { t } ^ { \tt c o n d } - \epsilon _ { t } ^ { { \tt u n c o n d } } + \tt M F 5$ </td><td>Independent</td><td> $\mathbf { 0 . 2 8 3 \pm 0 . 2 4 4 }$ </td><td>0.505</td><td> $\mathbf { 2 5 . 4 5 \pm 1 5 . 6 3 }$ </td><td> ${ \bf 9 . 4 0 \pm 8 . 4 9 }$ </td><td> $\mathbf { 0 . 3 5 5 \pm 0 . 2 3 6 }$ </td></tr><tr><td> $\epsilon _ { t } ^ { \mathsf { c o n d } } - \epsilon _ { t } ^ { \mathsf { u n c o n d } }$ </td><td>Reverse chain</td><td> $0 . 0 7 0 \pm 0 . 1 1 9$ </td><td>0.031</td><td> $3 2 . 0 5 \pm 1 6 . 0 1$ </td><td> $1 5 . 8 5 \pm 1 0 . 3 6$ </td><td> $0 . 1 7 0 \pm 0 . 2 3 8$ </td></tr><tr><td> $\epsilon _ { t } ^ { \mathsf { c o n d } } - \epsilon _ { t } ^ { \mathsf { u n c o n d } } + \mathsf { M F 5 }$ </td><td>Reverse chain</td><td> $0 . 0 7 0 \pm 0 . 1 0 8$ </td><td>0.036</td><td> $4 9 . 5 9 \pm 2 5 . 2 6$ </td><td> $2 3 . 3 6 \pm 1 5 . 4 4$ </td><td> $0 . 1 1 4 \pm 0 . 1 5 4$ </td></tr><tr><td> $\epsilon _ { t } ^ { \mathsf { c o n d } } - \epsilon _ { t }$  using forward noise</td><td>Independent</td><td> $0 . 0 4 5 \pm 0 . 2 1 1$ </td><td>0.017</td><td> $7 5 . 3 0 \pm 9 . 2 9$ </td><td> $5 2 . 6 9 \pm 7 . 7 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

NSD (0.355). These improvements indicate that mean filtering efectively suppresses spatially isolated responses while preserving coherent lesion evidence.

A kernel size of � = 7 produced the numerically highest Dice score of 0.287, compared with 0.283 for $k = 5 ,$ but resulted in slightly inferior AP and surface-distance based metrics. Performance subsequently declined with larger kernels. ${ \mathrm { A t ~ } } k { \mathrm { ~ = ~ } } 1 0 .$ , Dice decreased to 0.270, HD increased to 28.16 mm, MASD increased to 10.97 mm and NSD decreased to 0.304. This degradation suggests that excessive smoothing increasingly mixes lesion and healthy-tissue responses, reducing boundary accuracy and the localization of smaller lesions. Thus, the validation-selected kernel size of � = 5 also provided a favorable balance of overlap, ranking, and surface agreement on the test set.

## 4.6. Ablation Analysis of Discrepancy Formulations

Table 6 compares independent noise-level evaluation with reverse-chain accumulation and a formulation that replaces the unconditional prediction with the sampled forward noise, as described in Sec. 3.11. The pathologyinformed directional constraint was applied to all configurations.

Without mean filtering, independent evaluation outperformed reverse-chain accumulation (Eq. (43)) in Dice (0.232 versus 0.070), AP (0.374 versus 0.031), and all surface metrics. This advantage persisted after mean filtering: reversechain formulation’s Dice remained at 0.070 and AP increased only to 0.036, while HD and MASD worsened to 49.59 and 23.36 mm, respectively, and NSD decreased to 0.114. The reduced performance of chain-dependent accumulation suggests that maintaining a continuous reverse trajectory is not advantageous for discrepancy-based anomaly localization. Because each state is generated from the preceding reverse estimate, stochastic perturbations and reconstruction errors introduced at earlier steps propagate through the remainder of the trajectory. Moreover, as the reverse process progressively modifies the image, subtle pathological information present in the original subject image may be attenuated, causing later discrepancy estimates to become less directly associated with the observed abnormality. In contrast, independently constructing the input at each noise level connects every comparison to the same observed image and prevents errors at one timestep from afecting discrepancy measurements at subsequent timesteps.

Replacing the unconditional prediction with the sampled forward noise (Eq. (44)) produced the lowest performance across all metrics, with a Dice score of 0.045, AP of 0.017, HD of 75.30 mm, MASD of 52.69 mm, and NSD of 0.000. The sampled forward noise is independent of the subject anatomy and does not encode the learned healthy-data distribution. Consequently, its diference from the conditional prediction primarily reflects noise-prediction error rather than deviation from a healthy reference. By comparison, the unconditional branch provides the denoising response favored by the learned healthy prior for the same noisy input.

Directional characteristics of the discrepancy across the selected noise levels. The discrepancy is defined as $d _ { t } ( \mathbf { v } ) = \epsilon _ { t } ^ { \mathrm { c o n d } } ( \mathbf { v } ) -$ $\epsilon _ { t } ^ { \mathrm { u n c o n d } } ( \mathbf { v } )$ . A positive value indicates that the healthy-prior response predicts a higher ADC value than the patient-conditioned response, consistent with the expected direction of reduced ADC in acute HIE injury. Values are reported as the timestep-wise mean and 95% confidence interval across the selected noise levels (� = 125).
<table><tr><td>Statistic</td><td>Lesion Mean (95% CI)</td><td>Background Mean (95% CI)</td></tr><tr><td>Positive discrepancy probability,  $\operatorname* { P r } ( d _ { t } > 0 )$ </td><td>0.886 (0.884, 0.888)</td><td>0.530 (0.526, 0.533)</td></tr><tr><td>Mean positive discrepancy,  $\mathbb { E } [ d _ { t } \mid d _ { t } > 0 ]$ </td><td>0.266 (0.260, 0.272)</td><td>0.155 (0.151, 0.159)</td></tr><tr><td>Mean negative discrepancy,  $\mathbb { E } [ d _ { t } \mid d _ { t } < 0 ]$ </td><td>-0.107  $( - 0 . 1 1 0 , - 0 . 1 0 4 )$ </td><td>-0.176 (-0.178, -0.174)</td></tr><tr><td>Mean discrepancy magnitude,  $\mathbb { E } [ | d _ { t } | ]$ </td><td>0.248 (0.242, 0.254)</td><td>0.165 (0.162, 0.168)</td></tr></table>

Ablation of the pathology-informed directional sign constraint and mean-filter post-processing. MF5 denotes mean filtering with a kernel size of 5. Rows using MF5 are shaded, and the best result for each metric is shown in bold.
<table><tr><td>HIE Pathology prior directional sign ±</td><td>Post-processing</td><td>Dice ↑</td><td> $\mathsf { \pmb { A } } \mathsf { \pmb { P } } \mathrm { \dag }$ </td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td> $\mathsf { M A S D \perp ( m m ) }$ </td><td>NSD ↑</td></tr><tr><td>No</td><td></td><td> $0 . 1 1 5 \pm 0 . 1 4 8$ </td><td>0.222</td><td> $3 3 . 9 9 \pm 1 6 . 2 4$ </td><td> $1 5 . 1 8 \pm 1 0 . 0 4$ </td><td> $0 . 1 8 9 \pm 0 . 1 8 4$ </td></tr><tr><td>No</td><td>MF5</td><td> $0 . 2 0 9 \pm 0 . 2 2 1$ </td><td>0.490</td><td> $3 0 . 1 6 \pm 1 4 . 8 3$ </td><td> $1 1 . 4 8 \pm 8 . 8 2$ </td><td> $0 . 2 6 6 \pm 0 . 2 0 7$ </td></tr><tr><td>Yes</td><td></td><td> $0 . 2 3 2 \pm 0 . 2 3 9$ </td><td>0.374</td><td> $2 9 . 2 9 \pm 1 6 . 4 3$ </td><td> $1 2 . 6 2 \pm 1 0 . 1 1$ </td><td> $0 . 2 8 4 \pm 0 . 2 5 6$ </td></tr><tr><td>Yes</td><td>MF5</td><td> $\mathbf { 0 . 2 8 3 \pm 0 . 2 4 4 }$ </td><td>0.505</td><td> $\mathbf { 2 5 . 4 5 \pm 1 5 . 6 3 }$ </td><td> ${ \bf 9 . 4 0 \pm 8 . 4 9 }$ </td><td> $\mathbf { 0 . 3 5 5 \pm 0 . 2 3 6 }$ </td></tr></table>

The results therefore support both central components of the proposed formulation: independent evaluation across noise levels prevents trajectory-dependent error propagation, while the learned unconditional prediction provides a meaningful healthy prior reference for detecting subject-specific pathological deviations.

## 4.7. Verification of the HIE Pathology Directional Prior

We first examined whether the direction of the conditional– unconditional discrepancy was consistent with the expected reduction in ADC associated with acute HIE lesions. Table 7 summarizes the discrepancy $d _ { t } ( i ) = \epsilon _ { t , i } ^ { \mathrm { c o n d } } - \epsilon _ { t , i } ^ { \mathrm { u n c o n d } }$ within lesion and background voxels across the selected noise levels. Through the positive scaling relationship established in the Methods Sec. 3.4, positive value is equivalent to the healthy-prior response predicting a higher ADC value than the subject-conditioned response. Lesion voxels exhibited a positive discrepancy at 88.6% of the evaluated voxel– timestep pairs, compared with 53.0% in the background. The mean positive discrepancy was 0.266 in lesion versus 0.155 in background and the mean negative discrepancy was -0.107 and −0.176 respectively. These observations indicate that lesion voxels exhibit not only larger conditional– unconditional diferences but also a systematic shift in the direction expected for reduced ADC. In contrast, background discrepancies nearly symmetrically occur in both directions.

Table 8 further evaluates whether incorporating this directional prior improves anomaly localization with and without mean filtering. Without mean filtering, the directional constraint significantly increased Dice from 0.115 to 0.232 (absolute gain of 0.117; two-sided paired Wilcoxon signedrank test $p < 0 . 0 0 1 , n = 4 4 )$ . AP increased from 0.222 to 0.374, while HD decreased from 33.99 to 29.29 mm, MASD decreased from 15.18 to 12.62 mm, and NSD increased from 0.189 to 0.284. The improvement persisted when both formulations used mean filtering with a kernel size of 5. Adding the directional constraint increased Dice from 0.209 to 0.283 (absolute gain of 0.074; two-sided paired Wilcoxon signed-rank test $p ~ < ~ 0 . 0 0 1 , ~ n ~ = ~ 4 4 )$ . HD decreased from 30.16 to 25.45 mm, MASD decreased from 11.48 to 9.40 mm, and NSD increased from 0.266 to 0.355, alongside a smaller increase in AP from 0.490 to 0.505. Thus, the directional constraint improved all evaluated metrics both with and without mean filtering.

To evaluate whether the low-ADC directional prior also benefits existing methods, we applied it to the final reconstruction residuals of representative unsupervised baselines, retaining only deviations consistent with lower ADC in the observed image than in the reconstruction. This outputlevel modification left the original models and reconstruction procedures unchanged (Table 9). Compared with their original results in Table 1, directional filtering increased Dice from 0.159 to 0.172 for DAE, from 0.085 to 0.104 for AutoDDPM, and from 0.166 to 0.173 for MAD-AD, while pDDPM remained unchanged at the reported precision (0.121). Among these modified baselines, MAD-AD achieved the highest Dice (0.173) and NSD (0.226) and the lowest MASD (11.16 mm), whereas DAE achieved the highest AP (0.680) and the lowest HD (25.13 mm).

Quantitative comparison of unsupervised anomaly-detection baselines with introducing HIE pathology directional prior information at final reconstruction. Values are reported as mean ± standard deviation where available. The best result for each metric is highlighted in bold. For each model’s result, the originally proposed smoothing methods have been applied. For Dice, <sup>†</sup> and <sup>‡</sup> indicate significantly lower scores than Ours and Ours with mean filtering $\left( k = 5 \right)$ , respectively (under two-sided paired Wilcoxon signed-rank tests, all $p < 0 . 0 0 1 , n = 4 4 )$
<table><tr><td>Model</td><td>Dice ↑</td><td>AP↑</td><td> $\mathbf { H } \mathbf { D } \downarrow ( \mathbf { m } \mathbf { m } )$ </td><td> $\mathbf { \boldsymbol { \mathsf { M } } } \mathbf { \boldsymbol { A } } \mathbf { \boldsymbol { S } } \mathbf { \boldsymbol { D } } \downarrow \ \left( \mathbf { \boldsymbol { m } } \mathbf { \boldsymbol { m } } \right)$ </td><td>NSD ↑</td></tr><tr><td>DAE (Kascenas et al., 2022)</td><td> $0 . 1 7 2 \pm 0 . 2 7 2 ^ { \dagger \ddagger }$ </td><td>0.680</td><td> $\mathbf { 2 5 . 1 3 \pm 1 3 . 1 1 }$ </td><td> $1 3 . 4 7 \pm 9 . 5 6$ </td><td> $0 . 1 8 8 \pm 0 . 2 1 9$ </td></tr><tr><td>AutoDDPM (Bercea et al., 2023)</td><td> $0 . 1 0 4 \pm 0 . 1 3 5 ^ { \dagger \ddagger }$ </td><td>0.119</td><td> $3 0 . 7 8 \pm 1 5 . 9 9$ </td><td> $1 3 . 1 2 \pm 9 . 5 9$ </td><td> $0 . 2 0 9 \pm 0 . 2 0 6$ </td></tr><tr><td>pDDPM (Behrendt et al., 2024)</td><td> $0 . 1 2 1 \pm 0 . 2 3 5 ^ { \dagger \ddagger }$ </td><td>0.084</td><td> $3 9 . 0 7 \pm 1 5 . 0 7$ </td><td> $2 1 . 1 7 \pm 1 1 . 0 8$ </td><td> $0 . 0 5 0 \pm 0 . 1 1 1$ </td></tr><tr><td>MAD-AD (Beizaee et al., 2025a)</td><td> $\mathbf { 0 . 1 7 3 \pm 0 . 1 9 8 } ^ { \div \ddagger }$ </td><td>0.491</td><td> $2 7 . 4 4 \pm 1 3 . 6 7$ </td><td> ${ \bf 1 1 . 1 6 \pm 7 . 9 9 }$ </td><td> $\mathbf { 0 . 2 2 6 \pm 0 . 1 9 2 }$ </td></tr></table>

Our method with mean filtering achieved a Dice score of 0.283, significantly exceeding each low-ADC direction filtered baseline (two-sided paired Wilcoxon signed-rank tests, all $p ~ < ~ 0 . 0 0 1 , ~ n ~ = ~ 4 4 )$ . It also achieved a lower MASD (9.40 mm) and higher NSD (0.355) than all modified baselines. Its HD (25.45 mm) was slightly higher than that of DAE, while its AP (0.505) exceeded those of AutoD-DPM, pDDPM, and MAD-AD but remained below DAE. Thus, our model performance advantage persisted when the same pathological direction was incorporated into baseline outputs.

## 5. Discussion

## 5.1. Overall Detection Performance

The proposed ADC-only framework demonstrated strong overall performance for unsupervised HIE lesion localization. However, the improvement was not uniform across all evaluation criteria and this behavior reflects the diferent properties measured by these metrics. AP evaluates the ranking of continuous anomaly scores across lesion and background voxels without selecting a single operating threshold, whereas Dice and the surface metrics evaluate the geometry of the resulting binary segmentation. The comparatively strong Dice and surface performance of our method therefore suggests improved spatial localization, while its lower AP indicates that the continuous anomaly scores remain imperfectly calibrated and that some background voxels still receive scores overlapping those of lesion voxels. Improving cross-subject score calibration and suppressing non-pathological background responses remain important directions for further investigation.

The comparison with supervised ADC-only models also demonstrates the potential of learning from healthy-image structure rather than directly optimizing voxel-wise lesionmask prediction. Competitive performance in selected spatial metrics suggests that a learned healthy prior, combined with subject conditioning and pathological directionality, can provide informative localization cues without voxelwise lesion supervision in the training objective. This is particularly relevant when available annotations do not adequately represent the diversity of lesion appearances. The findings support healthy-prior modeling as a complementary route to quantitative HIE lesion assessment.

The lesion-burden analysis indicates that the primary limitation of the proposed method is the detection of subjects with small lesion burdens. Larger HIE abnormalities afect more voxels and are more likely to produce spatially coherent, persistent discrepancies across multiple difusion noise levels. In contrast, small or low-contrast lesions generate weaker and more localized responses that can be dificult to distinguish from normal anatomical variability, registration diferences, and acquisition noise. Mean filtering partially mitigated this limitation by suppressing isolated highresponse voxels and reinforcing discrepancies supported by neighboring voxels, resulting in the largest improvements for the small- and medium- burden groups. For subjects with large lesions, however, the anomaly signal was already spatially coherent, leaving less noise for the filter to remove. Additional smoothing could also blur lesion boundaries or extend responses into adjacent healthy tissue, explaining the minimal Dice improvement and the reduction in NSD observed in this group.

The relatively large inter-subject standard deviations further indicate heterogeneous performance across the test cohort. Such variability may reflect diferences in lesion burden, anatomical distribution, scan timing, and acquisition characteristics. Future studies should therefore evaluate performance after stratification by lesion burden, injury pattern, scanner field strength, and postnatal age at MRI, in addition to validating the method in independent multi-center cohorts.

## 5.2. Intermediate Denoising Discrepancy versus Reconstruction Diference

Most difusion-based anomaly-detection methods identify abnormalities by comparing an observed image with its final pseudo-healthy reconstruction. Although intuitive, this formulation compresses the entire reverse difusion trajectory into a single terminal image. Subtle pathological information expressed during intermediate denoising steps may therefore be attenuated or lost before the final reconstruction is obtained. Moreover, the reconstruction diference reflects not only lesion restoration but also stochastic sampling variation, incomplete lesion removal, and unintended modification of subject-specific healthy anatomy. The latter can generate false-positive responses even when the reconstructed image appears globally plausible.

Our CFG reconstruction experiments illustrate these limitations. Reducing the guidance scale to $w = 0 . 8$ encouraged stronger restoration toward the learned healthy distribution and increased $\mathbf { A P }$ relative to the DDPM baseline, but substantially degraded Dice and the surface-distance metrics. This suggests that weaker subject conditioning increased sensitivity to image diferences while also introducing spatial changes unrelated to the true lesion. Conversely, increasing the guidance scale to $w = 7 . 5$ preserved and strengthened subject-specific information but did not produce consistently accurate anomaly localization.

Our proposed method instead measures the conditional– unconditional diference directly in the intermediate noiseprediction space. Since both predictions are evaluated from the same noisy image $\mathbf { x } _ { t } ,$ the comparison controls for the common forward corruption and isolates the change in the denoising response induced by the subject-specific condition. The unconditional branch represents the response favored by the learned healthy prior, and the conditional branch retains information from the observed subject. Therefore, their diference provides a more direct measure of how the observed anatomy modifies the healthy denoising field than a comparison between two final reconstructed images. The substantially higher Dice obtained using multi noiselevel discrepancy accumulation than using either CFG-based reconstruction supports the utility of extracting anomaly evidence before it is compressed into the final reconstruction.

Evaluating each noise level independently provides an additional advantage over following a single reverse chain. In the chain-dependent formulation, each state inherits stochastic variation and reconstruction error from all preceding transitions, and progressive restoration may attenuate subtle pathological signals before they contribute to later discrepancy estimates. In contrast, independently sampling $\mathbf { x } _ { t }$ from the observed image connects every comparison to the same subject and prevents errors at one noise level from propagating to another. Accumulation across the selected intermediate noise levels emphasizes discrepancies that persist under diferent corruption strengths rather than relying on a single potentially unstable observation. Intermediate noise levels are particularly useful because they partially suppress the observed lesion while retaining suficient subject-specific anatomical information. At substantially higher noise levels, the anatomical condition may become weak, whereas near-clean inputs may provide insuficient separation between the two branch responses.

## 5.3. Pathology-Informed Directionality and Interpretability

A central feature of the proposed framework is that it does not treat all conditional–unconditional discrepancies as equally suggestive of pathology. Acute HIE is commonly associated with cytotoxic edema, restricted difusion, and reduced ADC values (Forbes et al., 2000; Bobba et al., 2023; Onda et al., 2024). We therefore retain discrepancy components for which the healthy-prior response predicts a higher ADC value than the patient-conditioned response. This constraint provides a clinically motivated direction to the anomaly score and suppresses deviations that are inconsistent with the expected low-ADC appearance of acute HIE. In contrast to an unsigned reconstruction residual, the resulting score can consequently be interpreted as evidence that the observed tissue deviates from the learned healthy distribution in a direction consistent with water difusion restriction.

The quantitative analysis on the distribution of discrepancy supports this assumption. At the selected noise levels, the expected positive discrepancy occurred more frequently in annotated lesion voxels than in background voxels. Lesion voxels also exhibited a substantially larger positive discrepancy. Consistent with these observations, incorporating the pathology-informed sign constraint improved lesion localization relative to the corresponding unsigned formulation. These findings demonstrate that the selected direction is statistically associated with annotated HIE abnormalities and is more specific than using unsigned discrepancy magnitude alone.

An additional advantage of the proposed formulation is that the pathological prior is incorporated directly into the intermediate difusion discrepancy, rather than being applied only to a completed reconstruction-derived anomaly map. Most conventional reconstruction-based methods define direction-agnostic anomaly evidence using an absolute or squared residual between the observed image and its pseudo-healthy reconstruction. Although a low-ADC directional constraint can be applied to the signed reconstruction residual, such output-level filtering is performed only after the reconstruction process has mixed pathological deviations with reconstruction blur, anatomical mismatch, and stochastic generation errors. It can remove some directionally inconsistent responses, but it cannot recover pathological evidence that was attenuated during reconstruction or determine whether the final residual was supported consistently across the difusion trajectory.

The comparative experiments support this distinction. Applying the same low-ADC directional rule as post-processing to conventional anomaly-detection outputs improved their localization performance, indicating that the pathological prior is useful even at the output level. However, these gains were smaller than those obtained when the directional constraint was integrated into the multi noise-level discrepancy formulation. This diference suggests that biological knowledge is more efective when intermediate denoising

responses contribute to the anomaly score, rather than being used only to reject candidates after reconstruction.

False-positive responses in regions with low ADC or interhemispheric intensity asymmetry are consistent with the intended behavior of the directional constraint, but also highlight its limited pathological specificity. The constraint selects voxels with deviations in the expected direction of ADC reduction, but directional agreement alone is insuficient to attribute these deviations to HIE-related injury. Incorporating regional anatomical context and spatially adaptive filtering may help distinguish lesion-associated responses from non-pathological variability.

## 5.4. Utility and Limitations of Population �-Score Refinement

The population �-map analysis demonstrates that voxelwise normative information can substantially improve lesion localization when an appropriate reference population and accurate spatial correspondence are available. However, this improvement introduces requirements that may limit generalizability in routine clinical settings. Constructing a meaningful ADC �-map requires spatially resolved estimates of the healthy population mean and standard deviation. These reference statistics may vary with postmenstrual age, scanner vendor, magnetic field strength, difusion acquisition protocol, reconstruction procedure, and preprocessing pipeline. This issue is particularly relevant in neonatal imaging because brain structure and difusion properties change rapidly during early development.

Voxel-wise �-maps also depend on accurate alignment between the observed scan and the normative template. Errors in nonlinear registration may generate artificial deviations near tissue boundaries, ventricles, and rapidly developing anatomical structures. Consequently, a �-map derived from one cohort may not transfer reliably to a population with diferent demographics, acquisition characteristics, or spatial normalization quality. Site-specific recalibration or construction of a new normative reference may therefore be required before deployment.

For these reasons, we consider the �-map refinement an optional population-reference assisted extension rather than the principal contribution of the proposed framework. The primary ADC-only model does not require population-level voxel-wise means and standard deviations or the precise nonlinear normalization required to construct a normative �-map. Although its localization performance is lower than that of the �-map assisted configuration, it reduces dependence on cohort-specific reference data and may therefore ofer greater portability across clinical populations and acquisition settings. Future work should evaluate whether normative harmonization, age-conditioned reference models, or registration-robust population representations can retain the localization benefit of �-scores while improving crosscohort generalizability.

## 5.5. Limitations and Future Work

Several limitations should be considered when interpreting the present findings. First, although lesion annotations were not used as voxel-wise training targets, they were used to exclude lesion-containing slices when constructing the healthy training subset. The proposed training procedure is therefore annotation-free at the voxel-wise training stage but is not entirely independent of annotations. Moreover, unannotated slices from an HIE-afected subject may still contain subtle or difuse abnormalities that were not included in the reference mask, potentially contaminating the learned healthy distribution. Future studies should train the framework using an independently acquired cohort of clinically healthy neonates with ADC imaging. This would eliminate the need for lesion mask-based slice selection and permit a more direct evaluation of fully annotation-independent training.

Second, the present evaluation was conducted using a single cohort and a fixed pre-processing and registration pipeline. Diferences in scanner manufacturer, field strength, acquisition protocol, reconstruction method, patient age, and clinical population may alter both the ADC distribution and the model response. The population-derived �-map is particularly sensitive to the population, scanning protocols and the accuracy of nonlinear registration. Multicenter external validation is therefore required to determine whether the learned healthy prior and operating threshold transfer across institutions.

The current implementation is based on independently processed 2D axial slices. Although this design reduces computational requirements and increases the number of available training samples, it does not explicitly model through-plane anatomical continuity. This limitation may particularly afect small lesions, lesions with weak contrast in an individual slice, and abnormalities extending obliquely or across multiple axial levels. The slice-wise predictions are subsequently assembled into a 3D volume, but this does not recover contextual information that was unavailable to the model during inference. Extending the framework to 2.5D or fully 3D difusion architectures may improve spatial consistency and the detection of small or fragmented lesions, although these approaches will require substantially greater memory and training data.

The pathology-informed directional prior also assumes that acute HIE injury is associated with reduced ADC relative to healthy tissue. This assumption is clinically motivated but is not universally valid across all imaging times and injury patterns. ADC values evolve after the hypoxic– ischemic event and may pseudo-normalize during the subacute (Vermeulen et al., 2008). Some injured regions may therefore exhibit weak, absent, or temporally altered ADC reduction despite clinically meaningful tissue damage. Such lesions may not produce a consistently positive conditional– unconditional discrepancy and could be suppressed by the proposed one-sided scoring rule. Future work could incorporate time from injury, gestational age, and treatment timing into the model.

The hard positive-part operation constitutes another limitation. Setting all opposite-direction discrepancies to zero provides a direct implementation of the reduced ADC prior, but it imposes an abrupt boundary at zero. Some lesion voxels may exhibit weakly negative discrepancies or fluctuate around zero across the selected noise levels because of subtle lesion contrast, model uncertainty, or temporal variation in ADC abnormalities. Hard clipping completely removes these responses, potentially reducing sensitivity to small or low-contrast lesions.

A future extension could replace hard clipping with a soft directional gate, controlled by a parameter �, that assigns reduced rather than zero weight to near-zero or slightly opposite-direction responses. This would preserve the preference for the expected HIE direction while accounting for uncertainty in the estimated discrepancy. Population-derived ADC �-maps could also be introduced as a continuous spatial weight at this stage, allowing strongly negative �- scores to provide supporting evidence for weak difusion discrepancies.

Finally, the continuous anomaly score is not a calibrated lesion probability. Its magnitude may vary with the sampled noise realization, the selected noise levels, the number of accumulated responses, and the difusion model’s uncertainty. Although geometric aggregation reduces the influence of isolated large responses, it does not eliminate stochastic variability or guarantee a common score scale across subjects and cohorts. Future work should quantify repeatability across random seeds and noise realizations, investigate ensembling or deterministic sampling, and estimate voxel-wise uncertainty.

## 6. Conclusion

We proposed OpenMAP-HIE, a pathology-informed score-based difusion model for unsupervised localization of neonatal HIE abnormalities on ADC images without using lesion annotations as voxel-wise training targets. Rather than relying on a final pseudo-healthy reconstruction, the framework compares and aggregates patient-conditioned and healthy-prior noise predictions across independently evaluated multi noise levels, retaining discrepancies consistent with the expected ADC reduction in acute HIE. The proposed ADC-only method improved lesion overlap and surface-distance performance over the evaluated unsupervised baselines and achieved Dice performance comparable to supervised ADC-only segmentation models, while population-derived ADC �-map refinement provided further improvements.

## CRediT authorship contribution statement

Pengyu Kan: Conceptualization, Formal analysis, Investigation, Methodology, Software, Visualization, Writing - Original Draft, Writing – review and editing. Craig Jones: Conceptualization, Formal analysis, Methodology, Writing – review and editing. Kenichi Oishi: Conceptualization, Formal analysis, Methodology, Funding acquisition, Project administration, Supervision, Writing – review and editing.

## Acknowledgements

This publication was made possible by grants from the National Institutes of Health R01NS126549. The contents of this paper are solely the responsibility of the authors and do not necessarily represent the oficial view of NIH.

## Declaration of competing interest

Kenichi Oishi is a consultant for Corporate-M and AnatomyWorks. These arrangements are being managed by Johns Hopkins University in accordance with its conflict-ofinterest policies.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used ChatGPT (OpenAI) to improve the language and readability of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Data availability

The BONBID-HIE dataset used in this study is publicly available through the BONBID-HIE challenge website (https://bonbid-hie2023.grand-challenge.org/), subject to the dataset’s access conditions and terms of use.

## References

Almansour, M., Ho, M.L., Palaniappan, K., 2025. Enhancing lesion segmentation in the bonbid-hie challenge: An ensemble strategy. AI for Brain Lesion Detection and Trauma Video Action Recognition: First BONBID-HIE Lesion Segmentation Challenge and First Trauma Thompson Challenge, Held in Conjunction with MICCAI 2023, Vancouver, BC, Canada, October 16 and 12, 2023, Proceedings 14567, 14.

Bano, S., Chaudhary, V., Garga, U.C., 2017. Neonatal hypoxic-ischemic encephalopathy: a radiological review. Journal of pediatric neurosciences 12, 1.

Bao, R., Foster, A.N., Song, Y., Vyas, R., Kesri, A., Toubal, I.E., Kazemi, E.S., Rahmon, G., Kucukpinar, T., Almansour, M., et al., 2025. Bonbidhie 2023: lesion segmentation challenge in boston neonatal brain injury data for hypoxic ischemic encephalopathy. IEEE transactions on medical imaging .

Baur, C., Denner, S., Wiestler, B., Navab, N., Albarqouni, S., 2021. Autoencoders for unsupervised anomaly segmentation in brain mr images: a comparative study. Medical image analysis 69, 101952.

Behrendt, F., Bhattacharya, D., Krüger, J., Opfer, R., Schlaefer, A., 2024. Patched difusion models for unsupervised anomaly detection in brain mri, in: Medical imaging with deep learning, PMLR. pp. 1019–1032.

Beizaee, F., Lodygensky, G., Desrosiers, C., Dolz, J., 2025a. Mad-ad: Masked difusion for unsupervised brain anomaly detection, in: International Conference on Information Processing in Medical Imaging, Springer. pp. 139–153.

Beizaee, F., Lodygensky, G.A., Desrosiers, C., Dolz, J., 2025b. Correcting deviations from normality: A reformulated difusion model for multiclass unsupervised anomaly detection, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 19088– 19097.

Bercea, C.I., Neumayr, M., Rueckert, D., Schnabel, J.A., 2023. Mask, stitch, and re-sample: Enhancing robustness and generalizability in anomaly detection through automatic difusion models, in: ICML 3rd Workshop on Interpretable Machine Learning in Healthcare (IMLH). URL: https: //openreview.net/forum?id=kTpafpXrqa.

Bercea, C.I., Wiestler, B., Rueckert, D., Schnabel, J.A., 2024. Difusion models with implicit guidance for medical anomaly detection, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 211–220.

Bobba, P.S., Malhotra, A., Sheth, K.N., Taylor, S.N., Ment, L.R., Payabvash, S., 2023. Brain injury patterns in hypoxic ischemic encephalopathy of term neonates. Journal of Neuroimaging 33, 79–84.

Chang, P.D., Chow, D.S., Alber, A., Lin, Y.K., Youn, Y.A., 2020. Predictive values of location and volumetric mri injury patterns for neurodevelopmental outcomes in hypoxic-ischemic encephalopathy neonates. Brain sciences 10, 991.

Dhariwal, P., Nichol, A., 2021. Difusion models beat gans on image synthesis. Advances in neural information processing systems 34, 8780– 8794.

Fernando, T., Gammulle, H., Denman, S., Sridharan, S., Fookes, C., 2021. Deep learning for medical anomaly detection–a survey. ACM computing surveys (CSUR) 54, 1–37.

Forbes, K.P., Pipe, J.G., Bird, R., 2000. Neonatal hypoxic-ischemic encephalopathy: detection with difusion-weighted mr imaging. American journal of neuroradiology 21, 1490–1496.

Frotscher, A., Kapoor, J., Wolfers, T., Baumgartner, C.F., 2026. Unsupervised anomaly detection in medical imaging using aggregated normative difusion. Medical Image Analysis 109, 103895. URL: https: //www.sciencedirect.com/science/article/pii/S1361841525004414, doi:https://doi.org/10.1016/j.media.2025.103895.

Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R., Xu, D., 2021. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images, in: International MICCAI brainlesion workshop, Springer. pp. 272–284.

Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851.

Ho, J., Salimans, T., 2021. Classifier-free difusion guidance, in: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. URL: https://openreview.net/forum?id=qw8AKxfYbI.

Kascenas, A., Pugeault, N., O’Neil, A.Q., 2022. Denoising autoencoders for unsupervised anomaly detection in brain mri, in: International Conference on Medical Imaging with Deep Learning, PMLR. pp. 653–664.

Liang, Z., Guo, X., Noble, J.A., Kamnitsas, K., 2024. Itermask 2: Iterative unsupervised anomaly segmentation via spatial and frequency masking for brain lesions in mri, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 339–348.

Long, M., Brandon, D.H., 2007. Induced hypothermia for neonates with hypoxic-ischemic encephalopathy. Journal of Obstetric, Gynecologic, & Neonatal Nursing 36, 293–298.

Loshchilov, I., Hutter, F., 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 .

Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L., 2022. Repaint: Inpainting using denoising difusion probabilistic models, in: 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), IEEE. pp. 11451–11461.

Martinez-Biarge, M., Diez-Sebastian, J., Kapellou, O., Gindner, D., Allsop, J., Rutherford, M., Cowan, F., 2011. Predicting motor outcome and death in term hypoxic-ischemic encephalopathy. Neurology 76, 2055–2061.

Oishi, K., Mori, S., Donohue, P.K., Ernst, T., Anderson, L., Buchthal, S., Faria, A., Jiang, H., Li, X., Miller, M.I., et al., 2011. Multi-contrast human neonatal brain atlas: application to normal neonate development analysis. Neuroimage 56, 8–20.

Onda, K., Chavez-Valdez, R., Graham, E.M., Everett, A.D., Northington, F.J., Oishi, K., 2024. Quantification of difusion magnetic resonance imaging for prognostic prediction of neonatal hypoxic-ischemic encephalopathy. Developmental neuroscience 46, 55–68.

Rao, A., Shukla, A., Bhargava, J., Ou, Y., Bao, R., 2025. Spatial priorguided boundary and region-aware 2d lesion segmentation in neonatal

hypoxic ischemic encephalopathy, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 520–530.

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer. pp. 234–241.

Sakai, S., He, X., Gu, C., Sigal, L., Hasegawa, T., 2026. Invad: Inversionbased reconstruction-free anomaly detection with difusion models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21389–21398.

Schlegl, T., Seeböck, P., Waldstein, S.M., Langs, G., Schmidt-Erfurth, U., 2019. f-anogan: Fast unsupervised anomaly detection with generative adversarial networks. Medical image analysis 54, 30–44.

Silveira, R.C., Procianoy, R.S., 2015. Hypothermia therapy for newborns with hypoxic ischemic encephalopathy. Jornal de pediatria 91, S78–S83.

Song, J., Meng, C., Ermon, S., 2020. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 .

Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B., 2021. Score-based generative modeling through stochastic diferential equations, in: International Conference on Learning Representations. URL: https://openreview.net/forum?id=PxTIG12RRHS.

Triulzi, F., Parazzini, C., Righini, A., 2006. Patterns of damage in the mature neonatal brain. Pediatric radiology 36, 608–620.

Tustison, N.J., Cook, P.A., Holbrook, A.J., Johnson, H.J., Muschelli, J., Devenyi, G.A., Duda, J.T., Das, S.R., Cullen, N.C., Gillen, D.L., Yassa, M.A., Stone, J.R., Avants, B.B., 2021. The antsx ecosystem for quantitative biological and medical imaging. Scientific Reports 11, 1–10. doi:10.1038/s41598-021-87564-6.

Vermeulen, R.J., van Schie, P.E., Hendrikx, L., Barkhof, F., van Weissenbruch, M., Knol, D.L., Pouwels, P.J., 2008. Difusion-weighted and conventional mr imaging in neonatal hypoxic ischemia: two-year followup study. Radiology 249, 631–639.

Wolleb, J., Bieder, F., Sandkühler, R., Cattin, P.C., 2022. Difusion models for medical anomaly detection, in: International Conference on Medical image computing and computer-assisted intervention, Springer. pp. 35– 45.

Wyatt, J., Leach, A., Schmon, S.M., Willcocks, C.G., 2022. Anoddpm: Anomaly detection with denoising difusion probabilistic models using simplex noise, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), IEEE. pp. 649–655.

Zhang, L., Rao, A., Agrawala, M., 2023. Adding conditional control to text-to-image difusion models, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 3813–3824.

Zhou, Z., Siddiquee, M.M.R., Tajbakhsh, N., Liang, J., 2019. Unet++: Redesigning skip connections to exploit multiscale features in image segmentation. IEEE transactions on medical imaging 39, 1856–1867.