# Stochastic Nonlinearities Improve Uncertainty Estimation

Mohammad Sarhangzadeh m.sarhangzadeh@bilkent.edu.tr

NeuraVision Lab, Bilkent University Ankara, Turkey

Amirreza M. Shebly

<sup>6</sup>amirreza@bilkent.edu.tr

<sub>0</sub><sup>2</sup>Doruk Oner

2doruk.oner@bilkent.edu.tr

## Abstract

Reliable uncertainty estimation is important across visual recognition tasks, providing information about the confidence and reliability of model predictions. Many uncertainty estimation methods obtain predictive diversity through mechanisms that can suppress intermediate features, perturb the input, or require substantial additional computation. Stochastic activation functions, where network units randomly select between multiple nonlinearities during a forward pass, have previously been used to improve optimization behavior, mitigate limitations of fixed ReLU activations, and increase output diversity. In this work, we reframe stochastic activation switching as a lightweight mechanism for uncertainty estimation. The proposed method samples between nonlinear activation functions, inducing predictive diversity while keeping intermediate feature pathways active. This enables uncertainty estimation from a single trained model without the training cost of ensembles, task-specific uncertainty heads, or architectural redesign. We evaluate the proposed approach across classification, semantic segmentation, and out-of-distribution detection using convolutional and transformer-based architectures on diverse datasets. Across these settings, stochastic activation switching improves uncertainty quality while preserving competitive primary-task performance, showing that stochastic nonlinearities provide an effective and information-preserving source of predictive variation for visual recognition.

## Introduction

Deep neural networks (DNNs) [22] have driven major advances in computer vision, achieving state-of-the-art performance across image classification [33], object detection [51, 52], semantic segmentation, and dense prediction tasks [42, 53]. Despite this progress, vision models frequently produce overconfident and poorly calibrated predictions [24], even when confronted with ambiguous inputs, corrupted samples, or out-of-distribution data [27]. This inability to reliably “know what it does not know” remains an obstacle to deploying deep learning systems in real-world visual recognition pipelines, where overconfident errors can lead to unreliable downstream decisions. Uncertainty estimation addresses this limitation by augmenting predictions with confidence measures that help identify unreliable outputs [31].

![](images/6448c34f4d4f9d0ef3cfe2b497120b59c1ccb1e242d204fe9c6d9379f2da7b3c.jpg)  
Figure 1: Uncertainty from stochastic nonlinearities. Stochastic forward passes are shown for a network trained to predict road maps. Clear roads produce similar outputs and low variance, while ambiguous roads and road-like buildings produce higher variance.

However, obtaining uncertainty estimates that are accurate, well-calibrated, computationally efficient, and compatible with modern architectures remains a challenging problem [1].

A common strategy for uncertainty estimation is to generate multiple predictions for the same input and measure their disagreement. Deep ensembles [34] achieve this by training multiple independent models and are often considered a strong baseline for predictive uncertainty. However, training and storing multiple models is computationally expensive. Single-model alternatives such as Monte Carlo dropout [19] reduce this cost by injecting stochasticity into the network during inference. Nevertheless, dropout-based methods obtain diversity by randomly suppressing features, which can disrupt information flow and degrade the intermediate representations required for accurate classification or dense prediction. This trade-off motivates the need for uncertainty estimation mechanisms that can induce predictive diversity without disrupting learned feature representations.

Stochastic activation functions have previously been explored as a means to improve gradient flow, introduce regularization, and increase predictive diversity in deep networks. Instead of using a fixed nonlinearity for each neuron or layer, stochastic activation methods allow activation units to randomly select from a predefined set of nonlinear functions during each forward pass. Prior work has used this randomness to improve robustness, alleviate the dead-unit problem, and encourage more diverse network behavior [16, 23, 64]. However, these methods have primarily been studied from the perspective of optimization and regularization, rather than a mechanism for uncertainty estimation.

In this work, we reframe stochastic activation functions as a mechanism for uncertainty quantification. Our main observation is that stochastic switching of nonlinearities can introduce useful predictive variation while preserving the flow of information through the network. Rather than removing features, the method keeps intermediate representations available and perturbs the nonlinear transformation applied to them. As a result, a single trained model can produce multiple plausible outputs without discarding intermediate information (Fig. 1). This property is particularly relevant for image classification and semantic segmentation, where preserving semantic and spatial structure is important for reliable predictions.

We propose a simple framework for uncertainty estimation based on stochastic activation switching. The method does not require training multiple models or substantially redesigning the underlying architecture. In our experiments, we apply it to both convolutional networks and transformer-based [58] vision models to evaluate its applicability across different model classes. At inference time, repeated stochastic forward passes produce a distribution of predictions from a single set of learned weights, enabling uncertainty to be estimated from predictive disagreement. Because the stochasticity is introduced through activation selection rather than feature removal, the proposed approach provides an efficient alternative to feature-masking methods while avoiding the computational burden of explicit ensembles.

We evaluate the framework on classification and segmentation across diverse datasets, modalities, scales, and in- and out-of-distribution settings. We compare against common uncertainty baselines and analyze task performance and uncertainty quality, with ablations on stochastic activation and inference settings. Results show competitive predictive performance and strong uncertainty estimates from a single trained model.

The main contributions of this work are as follows: (i) We propose stochastic nonlinearities as a computationally efficient and information-preserving approach for uncertainty estimation in image classification and semantic segmentation. (ii) We show that the proposed single-model approach achieves competitive uncertainty calibration and predictive performance compared with commonly used uncertainty estimation baselines, while avoiding the cost of training multiple models. (iii) We conduct extensive experiments and ablation across small, medium, and large-scale datasets, different visual modalities, convolutional and transformer-family architectures, and both in-distribution and out-of-distribution settings.

## 2 Related Work

## 2.1 Uncertainty Estimation

Bayesian Neural Networks (BNNs) [4] model epistemic uncertainty through posterior distributions over network parameters, but approximate Bayesian inference is costly at scale. Laplace approximations [10] reduce this cost by approximating the posterior around a trained solution, though performance depends on the quality of the approximation and curvature estimate. Lightweight probabilistic networks [21] instead propagate activation uncertainty using assumed density filtering, offering a more efficient alternative but relying on approximate distributional assumptions.

Monte Carlo Dropout (MC Dropout) [19] estimates uncertainty by keeping dropout active during inference and aggregating stochastic predictions. Although simple, lightweight, and model-agnostic [39], it obtains diversity by randomly suppressing features, so each stochastic pass discards part of the learned representation rather than transforming the same information differently. Variants such as MC-DropConnect [45], which stochastically masks network weights, Concrete Dropout [20], which learns dropout rates, and R-drop [63], which uses auxiliary regularization, have also been proposed, but their uncertainty still comes from removing information rather than preserving and transforming it.

Deep Ensembles [34] estimate uncertainty from disagreement among multiple independently trained networks. While effective, their diversity comes at the cost of repeated training, storage, and inference, with the computational burden typically scaling with the number of ensemble members. Efficiency-driven alternatives reduce this overhead by using fixed partially overlapping masks as in Masksembles [13], training checkpoints [30], low-rank parameter sharing [61], or multi-branch architectures [56], but fully independent ensembles often remain stronger in calibration and reliability [50].

Deterministic single-forward-pass methods provide an alternative that avoids repeated stochastic inference, including DUQ [57] and SNGP [36]. More recently, Deep Deterministic Uncertainty (DDU) [47] combines a regularized feature representation with post-hoc density estimation to obtain uncertainty from a single deterministic forward pass. Although these methods offer low inference cost, their uncertainty depends on the learned feature geometry and the assumptions of the distance based estimator.

Other single-model methods estimate uncertainty through input or model perturbations at inference time. Test-time augmentation (TTA) [60] uses transformed inputs, but requires manually chosen augmentations and repeated inference. Monte Carlo Batch Normalization [46] samples stochastic batch-normalization statistics, while iterative methods [14] use repeated refinement or convergence behavior. HCM [7] provides a sampling-free alternative based on hyperspherical confidence, but depends on its geometric formulation and optimization. These methods reduce the need for independently trained ensembles, but rely on input transformations, normalization noise, iterative refinement, or geometric constraints.

## 2.2 Stochastic activations

Stochastic activations have traditionally been used for regularization, optimization, and, in recent work, generative diversity. Early forms, such as dropout [28], can be interpreted as stochastic activation mechanisms that prevent co-adaptation of features. For optimization, methods such as those proposed by [23] address specific pathologies in ReLU networks, including the “dying ReLU” phenomenon [64], by injecting controlled randomness to preserve gradient flow in saturated regions and improve convergence stability.

Recent studies, including [38] and others exploring noise injection and adaptive activations [12, 25], explored stochastic activations primarily to enhance training dynamics, sparsity, and runtime adaptability. These methods switch between activation functions to improve optimization efficiency and occasionally preserve stochasticity during inference for improved generative behavior. In language and generative modeling, Lomeli et al. [38] demonstrated that such test-time stochasticity can serve as an alternative to temperature sampling [29] for increasing output diversity.

A related activation-based approach, Random Activation Functions (RAF) [55], uses activation-function diversity for uncertainty estimation through an ensemble of independently trained models, each assigned a randomly selected activation function that remains fixed within that model. Predictive uncertainty is estimated from disagreement across the ensemble members. In contrast, our method dynamically samples activation choices within a single model during both training and inference. Our work therefore leverages stochastic activations explicitly for uncertainty estimation. By maintaining activation stochasticity during inference, a single model can generate a functionally diverse set of predictions, providing a measure of combined predictive uncertainty without feature masking or architectural duplication. Similar to MC-Dropout, this uncertainty is derived from disagreement across repeated stochastic forward passes and is not explicitly decomposed into separate epistemic and aleatoric components. This reframes stochastic activations not only as a training mechanism, but also as an efficient, information-preserving path to reliable uncertainty estimation.

## 3 Method

## 3.1 Motivation

As discussed in the previous sections, existing uncertainty estimation methods often obtain predictive diversity by either suppressing features, training multiple models, perturbing the input, or introducing additional inference procedures. These choices can make uncertainty estimation less information-preserving, computationally expensive, or dependent on design choices that must be tuned for each architecture and dataset. Our goal is to obtain diverse predictions from a single trained model without removing intermediate features, duplicating the network, or introducing additional trainable parameters.

We achieve this by replacing deterministic activation functions with stochastic nonlinearities. Instead of assigning a fixed activation function to each unit, the network randomly switches between the backbone’s original activation and an alternative nonlinearity during each forward pass. The model is trained with the same stochastic mechanism used at inference time, so the learned weights become robust to activation-level variation rather than treating it as a test-time perturbation only. At test time, the weights are kept fixed and multiple stochastic forward passes are performed through the same model. The resulting prediction variability is then used to estimate uncertainty.

This formulation introduces diversity through transformations rather than feature removal. Every activation pathway remains active, and stochasticity only changes how the intermediate signal is transformed. As illustrated in Fig. 2, this leads to more structured uncertainty than feature masking when stochasticity is increased at inference time. Therefore, the network can sample different functional behaviors while preserving the information carried by its features. The method does not require model-specific uncertainty heads, additional learnable parameters, or repeated model training, making it simple to apply across different classification and segmentation architectures.

## 3.2 Stochastic Activation Switching

Let $f _ { \theta }$ denote a neural network with parameters θ. In a standard deterministic network, each pre-activation $z _ { j } ^ { ( l ) }$ at unit j in layer l is passed through a fixed activation function chosen by the backbone architecture. We replace this fixed choice with a stochastic activation switch. During each stochastic forward pass t, the unit applies either the original activation of the backbone, denoted by $\sigma _ { \mathrm { o r i g } } ^ { ( l ) } .$ , or an alternative nonlinearity $\sigma _ { \mathrm { a l t } }$

$$
\begin{array} { r } { h _ { j } ^ { ( l , t ) } = \left\{ \begin{array} { l l } { \sigma _ { \mathrm { o r i g } } ^ { ( l ) } \left( z _ { j } ^ { ( l , t ) } \right) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - p , } \\ { \sigma _ { \mathrm { a l t } } \left( z _ { j } ^ { ( l , t ) } \right) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } p . } \end{array} \right. } \end{array}\tag{1}
$$

Equivalently, let $m _ { j } ^ { ( l , t ) }$ ∼ Bernoulli(p) denote the binary activation-switching variable. Then Eq. 1 can be written as

$$
h _ { j } ^ { ( l , t ) } = \left( 1 - m _ { j } ^ { ( l , t ) } \right) \sigma _ { \mathrm { o r i g } } ^ { ( l ) } \left( z _ { j } ^ { ( l , t ) } \right) + m _ { j } ^ { ( l , t ) } \sigma _ { \mathrm { a l t } } \left( z _ { j } ^ { ( l , t ) } \right) .\tag{2}
$$

Here, $p$ controls the probability of switching away from the backbone activation. In our experiments, $\sigma _ { \mathrm { o r i g } } ^ { ( l ) }$ is the native activation used by the architecture, such as ReLU in convolutional networks or GELU in transformer-family models, while $\sigma _ { \mathrm { a l t } }$ is chosen as an alternative differentiable nonlinearity. Unless otherwise stated, we use $p = 0 . 5$ , corresponding to uniform switching between the two nonlinearities. This choice maximizes the entropy of the Bernoulli activation selection, giving both activation pathways equal probability and avoiding a systematic preference for either nonlinearity.

Uncertainty Maps  
![](images/7cbd4774be07a354cafcfba1c8afb79c5124f3965a14fe0a8e56060ceb0f72b9.jpg)  
Figure 2: Uncertainty quality under test-time stress. The left panel shows the input, prediction, ground truth, and errors; the right panels compare uncertainty maps from stochastic activation switching and MC Dropout, with brighter values indicating higher uncertainty. We increase stochasticity only at inference time to examine its effect on uncertainty. MC Dropout removes intermediate features, causing uncertainty to spread broadly, whereas stochastic activation switching keeps feature pathways active and concentrates uncertainty near boundaries and prediction errors. This supports our motivation that activation-level stochasticity can induce predictive diversity without disrupting intermediate representations.

The sampling operation can be applied at different levels of granularity, leading to different forms of functional diversity:

• Node-level: The most granular approach, where a unique activation is sampled for each individual activation unit j in layer l.

• Channel-level: A strategy primarily for convolutional networks, where a single activation is sampled for each channel c in layer l and shared across all spatial locations within that channel.

• Layer-level: The coarsest strategy, where one activation is sampled for the entire layer l and applied to all units in that layer.

Regardless of the sampling granularity, each stochastic forward pass defines a different activation pathway through the same trained weights. For an input x, the t-th pass produces:

$$
\bar { y } ^ { ( t ) } = f _ { \theta } \Big ( x ; \sigma ^ { ( t ) } \Big )\tag{3}
$$

where $\boldsymbol { \sigma } ^ { ( t ) }$ denotes the sampled activation configuration used during that pass. Over T stochastic passes, the model generates a sequence of predictions

$$
\begin{array} { r } { Y ( x ) = \Big \{ \bar { y } ^ { ( 1 ) } , \bar { y } ^ { ( 2 ) } , \dots , \bar { y } ^ { ( T ) } \Big \} } \end{array}\tag{4}
$$

![](images/88be58bd3a555015f86cfb69a36a886a4c01be17801369d2205e4a30ea351c33.jpg)  
Figure 3: Toy regression under heteroscedastic noise. A 3-layer MLP is trained on noisy sinusoidal samples, and stochastic activation switching is compared with MC-Dropout under the same setup. Both methods use repeated stochastic passes. Our method shows clearer combined predictive uncertainty, remaining more certain in well-supported regions while uncertainty increases with stronger noise and outside the observed range.

all produced by the same parameters $\theta$ but different activation pathways. We refer to this sequence as a stochastic activation trajectory.

Figure 3 provides an intuitive example of this behavior in a toy regression setting with heteroscedastic noise. Compared with MC-Dropout, our stochastic activation trajectories remain more stable where the data are well supported, while their variation increases in noisier regions and outside the training range.

## 3.3 Training and Uncertainty Estimation

Training follows the standard supervised learning procedure for the target task. Given an input–label pair $( x , y )$ , the model performs a stochastic forward pass using activation choices sampled according to Eq. 1. The network parameters $\theta$ are optimized using the task loss:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } \Big ( f _ { \theta } ( x ; \sigma ^ { ( t ) } ) , y \Big ) ,\tag{5}
$$

where $\boldsymbol { \sigma } ^ { ( t ) }$ denotes the sampled activation configuration for forward pass t. No additional uncertainty objective, auxiliary head, or architecture-specific parameter is introduced. The same stochastic activation mechanism is used during training and inference, but the weights are updated only during training.

At inference time, the learned weights θ are fixed. For an input x, we perform T stochastic forward passes by repeatedly sampling activation configurations. Each pass produces a probability prediction $p ^ { ( t ) } ( x ) = f _ { \theta } \Big ( x ; \sigma ^ { ( t ) } \Big )$ , where $p ^ { ( t ) } ( x )$ denotes class probabilities for classification or pixel-wise class probabilities for segmentation. At inference time, the final prediction and uncertainty are computed as

$$
\hat { \mu } ( x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } p ^ { ( t ) } ( x ) , \qquad \hat { U } ( x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Big ( p ^ { ( t ) } ( x ) - \hat { \mu } ( x ) \Big ) ^ { 2 } .\tag{6}
$$

Here, ${ \hat { \mu } } ( x )$ is used as the final prediction, and ${ \hat { U } } ( x )$ denotes the empirical variance across stochastic predictions. For classification, this variance is computed over class probabilities. For segmentation, it is computed at each pixel, producing a spatial uncertainty map. High uncertainty indicates that different stochastic activation configurations lead to inconsistent predictions, while low uncertainty indicates stable predictions across activation choices.

To analyze whether the stochastic mechanism preserves information more effectively than feature-masking approaches, we introduce Information Destruction Rate (IDR) as a diagnostic measure. IDR is not used as a training objective; it is used only to compare the effect of inference-time stochasticity on feature representations. Let $I _ { \mathrm { f u l l } } ^ { ( l ) } = I ( h _ { \mathrm { i n } } ^ { ( l ) } ; \bar { h } _ { \mathrm { o u t } } ^ { ( l ) } )$ denote the estimated mutual information [67] between the input and output of layer l under the deterministic forward pass. For stochastic pass t, let $I _ { t } ^ { ( l ) }$ be the corresponding mutualinformation estimate under stochastic inference. We define

$$
\mathrm { I D R } ^ { ( l ) } = 1 - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { I _ { t } ^ { ( l ) } } { I _ { \mathrm { f u l l } } ^ { ( l ) } } .\tag{7}
$$

Lower IDR indicates that stochastic inference preserves more of the deterministic layer-wise input–output information. We use this analysis to compare stochastic activation switching with feature-masking baselines such as MC Dropout [19], where uncertainty is induced by removing activations rather than transforming them.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate the proposed stochastic activation switching framework across image segmentation and image classification tasks. The experiments are designed to test whether the method provides reliable uncertainty estimates while maintaining competitive task performance across different domains, architectures, and distribution shifts. For fair comparison, all methods are trained and evaluated using the same dataset splits, input resolution, batch size, preprocessing, augmentation pipeline, backbone architecture, and optimization schedule within each task, unless a baseline requires its standard uncertainty-specific modification.

Baselines. We compare against MC Dropout [19], DDU [47], TTA [60], Laplace Redux [10], Masksembles [13], Deep Ensembles [34], and, for segmentation, an Iterative Network baseline. MC Dropout, TTA, Masksembles, and our method are single-model multi-pass methods, while Laplace Redux applies a post-hoc posterior approximation to a trained model and DDU applies post-hoc density estimation to learned features. Deep Ensembles require five independently trained models, Masksembles uses five fixed masks, and the Iterative baseline uses three refinement iterations [14]. Unless otherwise stated, our method and MC Dropout use five stochastic forward passes, TTA uses five augmented inference samples. For Laplace Redux, we follow the sampling configuration appropriate to each dataset and backbone. We report relative training and inference cost where appropriate.

Datasets. For segmentation, we use Massachusetts Roads [44], DRIVE [54], CREMI [18], and LiTS [3], with RoadTracer [2], CHASE-DB1 [17], and ISBI 2012 [6] used for cross-dataset OOD evaluation. For classification, we use CIFAR-10 [32], MiniImageNet [59], BloodMNIST, and PathMNIST [65, 66], with SVHN [49] and CIFAR-100 [32] used as OOD datasets for CIFAR-10 and MiniImageNet, respectively. , and BloodMNIST and PathMNIST used as cross-domain OOD sets for each other.

![](images/fddd787341738d6559d15a5869c426e9ba0e4719dbfc682edbcc2f06a492d482.jpg)  
Figure 4: Visualization of our stochastic activation based uncertainty estimation. From left to right: input image, ground-truth mask, model prediction, and the uncertainty map computed from the variance across 5 stochastic activation trajectories.

Backbones. For segmentation, we use U-Net-based architectures [53] for Massachusetts Roads, DRIVE, and LiTS, and SwinTransformer V2 [9, 37] for CREMI. For classification, we use ResNet-style convolutional backbones for CIFAR-10, and MiniImageNet. For highresolution MedMNIST classification, we use SwinTransformer V2 [9, 37]. In all cases, our method keeps the same backbone as the corresponding baseline and only replaces deterministic activations with stochastic activation switching. Unless otherwise stated, stochasticity is applied at the node level, as this provides the best overall trade-off in our granularity ablation, as shown in Section 5. For segmentation experiments, we switch between the backbone’s native activation and Tanh, while for classification experiments we use the backbone’s native activation and SiLU [15], following the activation-choice ablation.

Metrics. For segmentation, we report F1 and structural overlap metrics: Correctness, Completeness, and Quality (CCQ) [62], together with Pearson correlation between uncertainty and prediction error, expected calibration error (ECE) [48], and Brier score [5]. Segmentation OOD detection uses ROC-AUC and PR-AUC. For classification, we report accuracy, ECE, Brier score, and OOD metrics AUC, AUPR, and FPR95 [41]. Classification directly yields one image-level uncertainty score, while for segmentation we spatially average the pixel-wise uncertainty map to obtain one score per image. OOD metrics are computed from these image-level scores. For segmentation and CIFAR-10 classification, we report mean ± standard deviation over three random seeds.

## 4.2 Results

Segmentation Tables 1 and 2 show that stochastic activation switching achieves best or second-best results in most task and uncertainty metrics while retaining the original architecture and single-model training setup. Table 1 shows strong segmentation performance on both thin structures such as roads and vessels and more complex biomedical settings.

This indicates that activation-level stochasticity can support uncertainty estimation while retaining strong spatial representations for dense prediction. Table 2 shows that the resulting uncertainty aligns well with prediction errors, with strong calibration and Brier scores.

<table><tr><td>Dataset</td><td>Method</td><td>Correct. (↑)</td><td>Complete. (↑)</td><td>Qual. (↑)</td><td>F1 (↑)</td></tr><tr><td rowspan="7">Massachusetts</td><td>MC Dropout</td><td>91.00</td><td>87.20</td><td>81.90</td><td>78.62</td></tr><tr><td>DDU</td><td>91.44</td><td>86.10</td><td>80.52</td><td>79.23</td></tr><tr><td>TTA</td><td>91.46</td><td>86.73</td><td>81.19</td><td>79.43</td></tr><tr><td>Masksembles</td><td>81.63</td><td>87.67</td><td>81.94</td><td>80.02</td></tr><tr><td>Deep Ensemble</td><td>92.37</td><td>88.11</td><td>82.67</td><td>80.54</td></tr><tr><td>Laplace Redux</td><td>91.45</td><td>86.11</td><td>80.52</td><td>79.23</td></tr><tr><td>Iterative</td><td>90.07</td><td>84.97</td><td>78.39</td><td>78.58</td></tr><tr><td>Ours</td><td>91.88</td><td>88.24</td><td>82.11</td><td>79.71</td></tr><tr><td rowspan="10">DRIVE</td><td></td><td>±0.43</td><td>±0.38</td><td>±0.65</td><td>±0.16</td></tr><tr><td>MC Dropout</td><td>96.18</td><td>85.13</td><td>82.53</td><td>79.41</td></tr><tr><td>DDU</td><td>96.13</td><td>85.83</td><td>82.86</td><td>80.81</td></tr><tr><td>TTA</td><td>95.45</td><td>84.41</td><td>82.29</td><td>79.98</td></tr><tr><td>Masksembles</td><td>96.82</td><td>85.11</td><td>82.69</td><td>81.02</td></tr><tr><td>Deep Ensemble</td><td>97.28</td><td>85.84</td><td>83.72</td><td>81.84</td></tr><tr><td>Laplace Redux</td><td>96.21</td><td>84.54</td><td>82.81</td><td>77.56</td></tr><tr><td>Iterative</td><td>96.86</td><td>82.61</td><td>81.87</td><td>80.30</td></tr><tr><td>Ours</td><td>98.38</td><td>85.61</td><td>82.91</td><td>81.00</td></tr><tr><td></td><td>±0.84</td><td>±0.48</td><td>±0.53</td><td>±0.28</td></tr><tr><td rowspan="10">LiTS</td><td>MC Dropout</td><td>87.09</td><td>71.06</td><td>65.82</td><td>96.41</td></tr><tr><td>DDU</td><td>87.45</td><td>78.82</td><td>69.73</td><td>97.02</td></tr><tr><td>TTA</td><td>72.04</td><td>64.01</td><td>58.73</td><td>93.87</td></tr><tr><td>Masksembles</td><td>88.27</td><td>77.83</td><td>70.75</td><td>98.01</td></tr><tr><td>Deep Ensemble</td><td>87.59</td><td>81.38</td><td>72.84</td><td>98.25</td></tr><tr><td>Laplace Redux</td><td>87.27</td><td>80.40</td><td>71.43</td><td>97.51</td></tr><tr><td>Iterative</td><td>87.98</td><td>74.55</td><td>70.26</td><td>97.39</td></tr><tr><td>Ours</td><td>89.39</td><td>79.57</td><td>72.93</td><td>98.11</td></tr><tr><td></td><td>±0.44</td><td>±0.35</td><td>±0.59</td><td>±0.10</td></tr><tr><td>MC Dropout</td><td>97.40</td><td>94.45</td><td>92.68</td><td>78.58</td></tr><tr><td rowspan="8">CREMI</td><td>DDU</td><td>96.41</td><td></td><td></td><td></td></tr><tr><td>TTA</td><td></td><td>94.76</td><td>93.16</td><td>79.48</td></tr><tr><td>Masksembles</td><td>97.04 96.90</td><td>94.65</td><td>92.63</td><td>76.23</td></tr><tr><td>Deep Ensemble</td><td>97.24</td><td>95.01 95.83</td><td>93.56 93.93</td><td>81.76 79.92</td></tr><tr><td>Laplace Redux</td><td>97.26</td><td>95.62</td><td>93.76</td><td>79.58</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Iterative</td><td>96.02</td><td>87.90</td><td>87.19</td><td>65.73</td></tr><tr><td>Ours</td><td>98.47</td><td>96.61</td><td>94.76</td><td>81.94</td></tr><tr><td></td><td></td><td>±0.55</td><td>±0.43</td><td>±0.71</td><td>±0.11</td></tr></table>

Table 1: Segmentation performance across aerial and medical benchmarks.

Together with Table 1, these results show improved uncertainty quality without sacrificing segmentation performance or introducing task-specific uncertainty modules.

For OOD evaluation, models trained on Massachusetts, DRIVE, and CREMI are tested on RoadTracer, CHASE-DB1, and ISBI 2012, respectively. Table 3 shows that disagreement between stochastic activation trajectories remains informative under distribution shift.

Classification. We evaluate classification uncertainty on natural and medical benchmarks. Table 4 reports classification performance and OOD detection. The results show that stochastic activation switching improves uncertainty behavior while maintaining strong classification performance. On the natural-image benchmarks, the method achieves strong calibration and OOD detection; on MiniImageNet, however, its classification accuracy is lower than Deep Ensembles (66.90 vs. 70.50), despite better calibration and OOD detection metrics. On high-resolution medical benchmarks, method transfers to Swin Transformer V2, maintaining high in-distribution accuracy while improving cross-domain OOD separation.

<table><tr><td>Dataset</td><td>Method</td><td>Corr. (↑)</td><td>ECE (↓)</td><td>Brier (↓)</td><td>Train</td><td>Infer</td></tr><tr><td rowspan="9">Massachusetts</td><td>MC Dropout</td><td>79.73</td><td>0.026</td><td>0.033</td><td>×1</td><td>×5</td></tr><tr><td>DDU</td><td>75.64</td><td>0.027</td><td>0.031</td><td>x1</td><td></td></tr><tr><td>TTA</td><td>79.90</td><td>0.025</td><td>0.029</td><td>x1</td><td>×5</td></tr><tr><td>Masksembles</td><td>79.62</td><td>0.024</td><td>0.031</td><td>×1</td><td>×5</td></tr><tr><td>Deep Ensemble</td><td>80.03</td><td>0.024</td><td>0.027</td><td>×5</td><td>×5</td></tr><tr><td>Laplace Redux</td><td>77.51</td><td>0.034</td><td>0.034</td><td>×1</td><td>一</td></tr><tr><td>Iterative</td><td>77.28</td><td>0.028</td><td>0.030</td><td>×3</td><td>×3</td></tr><tr><td>Ours</td><td>80.38</td><td>0.020</td><td>0.025</td><td>×1</td><td>×5</td></tr><tr><td></td><td>±1.08</td><td>±0.002</td><td>±0.006</td><td></td><td></td></tr><tr><td rowspan="8">DRIVE</td><td>MC Dropout</td><td>80.58</td><td>0.032</td><td>0.040</td><td>×1</td><td>×5</td></tr><tr><td>DDU</td><td>80.45</td><td>0.031</td><td>0.034</td><td>x1</td><td>一</td></tr><tr><td>TTA</td><td>83.00</td><td>0.058</td><td>0.031</td><td>×1</td><td>×5</td></tr><tr><td>Masksembles</td><td>81.06</td><td>0.029</td><td>0.035</td><td>×1</td><td>×5</td></tr><tr><td>Deep Ensemble</td><td>81.72</td><td>0.017</td><td>0.031</td><td>×5</td><td>×5</td></tr><tr><td>Laplace Redux</td><td>79.56</td><td>0.086</td><td>0.043</td><td>x1</td><td>一</td></tr><tr><td>Iterative</td><td>79.91</td><td>0.024</td><td>0.035</td><td>×3</td><td>×3</td></tr><tr><td>Ours</td><td>81.80</td><td>0.015</td><td>0.029</td><td>×1</td><td>×5</td></tr><tr><td rowspan="8">LiTS</td><td></td><td>±1.43</td><td>±0.001</td><td>±0.007</td><td></td><td></td></tr><tr><td>MC Dropout</td><td>52.28</td><td>0.004</td><td>0.003</td><td>×1</td><td>×5</td></tr><tr><td>DDU</td><td>52.55</td><td>0.004</td><td>0.003</td><td>×1</td><td>一</td></tr><tr><td>TTA</td><td>52.41</td><td>0.004</td><td>0.003</td><td>x1</td><td>×5</td></tr><tr><td>Masksembles</td><td>52.80</td><td>0.001</td><td>0.003</td><td>x1</td><td>×5</td></tr><tr><td>Deep Ensemble</td><td>53.30</td><td>0.001</td><td>0.001</td><td>×5</td><td>×5</td></tr><tr><td>Laplace Redux</td><td>52.35</td><td>0.003</td><td>0.003</td><td>x1</td><td>一</td></tr><tr><td>Iterative Ours</td><td>53.34 54.22</td><td>0.002</td><td>0.002</td><td>×3</td><td>×3</td></tr><tr><td rowspan="9">CREMI</td><td></td><td>±0.64</td><td>0.001</td><td>0.002</td><td>×1</td><td>×5</td></tr><tr><td>MC Dropout</td><td></td><td>±0.000</td><td>±0.000</td><td></td><td></td></tr><tr><td>DDU</td><td>37.23 34.12</td><td>0.035 0.016</td><td>0.017 0.018</td><td>x1 x1</td><td>×5</td></tr><tr><td>TTA</td><td>38.70</td><td>0.011</td><td>0.016</td><td>x1</td><td>一 ×5</td></tr><tr><td>Masksembles</td><td>36.97</td><td>0.014</td><td>0.017</td><td>x1</td><td>×5</td></tr><tr><td>Deep Ensemble</td><td>38.49</td><td>0.009</td><td>0.014</td><td>×5</td><td>×5</td></tr><tr><td>Laplace Redux</td><td>35.47</td><td>0.011</td><td>0.015</td><td>x1</td><td>一</td></tr><tr><td>Iterative</td><td>39.48</td><td>0.050</td><td>0.023</td><td>×3</td><td>×3</td></tr><tr><td></td><td>42.58</td><td>0.004</td><td>0.009</td><td></td><td></td></tr><tr><td>Ours</td><td></td><td>±1.80</td><td>±0.002</td><td>±0.004</td><td>×1</td><td>×5</td></tr></table>

Table 2: Segmentation uncertainty, calibration, and relative computational cost.

<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=2>Mass. → RoadTracer</td><td rowspan=1 colspan=2>DRIVE → CHASE</td><td rowspan=1 colspan=2>CREMI → ISBI</td></tr><tr><td rowspan=1 colspan=1>AUROC (↑)</td><td rowspan=1 colspan=1>AUPR (↑)</td><td rowspan=1 colspan=1>AUROC (↑)</td><td rowspan=1 colspan=1>AUPR (↑)</td><td rowspan=1 colspan=1>AUROC (↑)</td><td rowspan=1 colspan=1>AUPR (↑)</td></tr><tr><td rowspan=1 colspan=1>MC Dropout</td><td rowspan=1 colspan=1>53.07</td><td rowspan=1 colspan=1>48.51</td><td rowspan=1 colspan=1>82.21</td><td rowspan=1 colspan=1>53.07</td><td rowspan=1 colspan=1>87.34</td><td rowspan=1 colspan=1>94.64</td></tr><tr><td rowspan=1 colspan=1>DDU</td><td rowspan=1 colspan=1>54.63</td><td rowspan=1 colspan=1>52.54</td><td rowspan=1 colspan=1>85.99</td><td rowspan=1 colspan=1>61.77</td><td rowspan=1 colspan=1>88.79</td><td rowspan=1 colspan=1>94.82</td></tr><tr><td rowspan=1 colspan=1>TTA</td><td rowspan=1 colspan=1>55.05</td><td rowspan=1 colspan=1>56.62</td><td rowspan=1 colspan=1>89.94</td><td rowspan=1 colspan=1>67.31</td><td rowspan=1 colspan=1>93.10</td><td rowspan=2 colspan=1>94.8894.14</td></tr><tr><td rowspan=2 colspan=1>MasksemblesDeep Ensemble</td><td rowspan=1 colspan=1>54.00</td><td rowspan=1 colspan=1>51.91</td><td rowspan=1 colspan=1>86.35</td><td rowspan=1 colspan=1>64.51</td><td rowspan=1 colspan=1>87.72</td></tr><tr><td rowspan=1 colspan=1>57.32</td><td rowspan=1 colspan=1>57.85</td><td rowspan=1 colspan=1>89.56</td><td rowspan=1 colspan=1>66.33</td><td rowspan=1 colspan=1>85.73</td><td rowspan=1 colspan=1>94.69</td></tr><tr><td rowspan=1 colspan=1>Laplace Redux</td><td rowspan=1 colspan=1>52.82</td><td rowspan=1 colspan=1>52.04</td><td rowspan=1 colspan=1>86.46</td><td rowspan=1 colspan=1>57.90</td><td rowspan=1 colspan=1>88.68</td><td rowspan=1 colspan=1>94.77</td></tr><tr><td rowspan=1 colspan=1>Iterative</td><td rowspan=1 colspan=1>57.04</td><td rowspan=1 colspan=1>49.68</td><td rowspan=1 colspan=1>83.23</td><td rowspan=1 colspan=1>57.32</td><td rowspan=1 colspan=1>88.46</td><td rowspan=1 colspan=1>94.53</td></tr><tr><td rowspan=2 colspan=1>Ours</td><td rowspan=1 colspan=1>56.74</td><td rowspan=1 colspan=1>56.00</td><td rowspan=1 colspan=1>90.52</td><td rowspan=1 colspan=1>66.31</td><td rowspan=1 colspan=1>95.43</td><td rowspan=1 colspan=1>95.68</td></tr><tr><td rowspan=1 colspan=1>±0.71</td><td rowspan=1 colspan=1>±1.36</td><td rowspan=1 colspan=1>±0.47</td><td rowspan=1 colspan=1>±0.19</td><td rowspan=1 colspan=1>±0.24</td><td rowspan=1 colspan=1>±0.30</td></tr></table>

Table 3: Cross-dataset segmentation OOD detection.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=2>Acc (↑)</td><td rowspan=1 colspan=1>ECE (↓)</td><td rowspan=1 colspan=1>Brier (↓)</td><td rowspan=1 colspan=1>AUC (↑)</td><td rowspan=1 colspan=1>AUPR (↑)</td><td rowspan=1 colspan=1>FPR95 (↓)</td></tr><tr><td rowspan=8 colspan=1>CIFAR-10</td><td rowspan=2 colspan=1>MC DropoutDDU</td><td rowspan=1 colspan=2>93.54</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>84.22</td><td rowspan=1 colspan=1>87.36</td><td rowspan=2 colspan=1>27.1024.11</td></tr><tr><td rowspan=1 colspan=2>94.12</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.090</td><td rowspan=1 colspan=1>85.20</td><td rowspan=1 colspan=1>89.26</td></tr><tr><td rowspan=4 colspan=1>TTAMasksemblesDeep EnsembleLaplace Redux</td><td rowspan=1 colspan=2>94.23</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>93.64</td><td rowspan=1 colspan=1>97.29</td><td rowspan=1 colspan=1>17.97</td></tr><tr><td rowspan=2 colspan=2>94.6194.94</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.080</td><td rowspan=1 colspan=1>91.69</td><td rowspan=1 colspan=1>92.88</td><td rowspan=1 colspan=1>18.47</td></tr><tr><td rowspan=1 colspan=1>4.94</td><td rowspan=1 colspan=1>0.009</td><td rowspan=1 colspan=1>0.074</td><td rowspan=1 colspan=1>91.92</td><td rowspan=1 colspan=1>93.15</td><td rowspan=1 colspan=1>14.70</td></tr><tr><td rowspan=1 colspan=2>93.83</td><td rowspan=1 colspan=1>0.090</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>92.60</td><td rowspan=1 colspan=1>94.98</td><td rowspan=1 colspan=1>18.83</td></tr><tr><td rowspan=2 colspan=1>Ours</td><td rowspan=2 colspan=2>94.59±0.31</td><td rowspan=2 colspan=1>0.009±0.001</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>95.14</td><td rowspan=1 colspan=1>97.54</td><td rowspan=1 colspan=1>16.36</td></tr><tr><td rowspan=1 colspan=1>±0.005</td><td rowspan=1 colspan=1>±0.22</td><td rowspan=1 colspan=1>±0.35</td><td rowspan=1 colspan=1>±0.29</td></tr><tr><td rowspan=7 colspan=1>MiniImageNet</td><td rowspan=6 colspan=1>MC DropoutDDUTTAMasksemblesDeep EnsembleLaplace Redux</td><td rowspan=1 colspan=2>64.70</td><td rowspan=1 colspan=1>0.067</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=1>81.57</td><td rowspan=1 colspan=1>86.96</td><td rowspan=1 colspan=1>53.94</td></tr><tr><td rowspan=1 colspan=2>65.83</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.191</td><td rowspan=1 colspan=1>84.21</td><td rowspan=1 colspan=1>88.49</td><td rowspan=1 colspan=1>49.85</td></tr><tr><td rowspan=1 colspan=2>68.52</td><td rowspan=1 colspan=1>0.044</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>79.10</td><td rowspan=1 colspan=1>84.41</td><td rowspan=1 colspan=1>54.28</td></tr><tr><td rowspan=1 colspan=2>66.24</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.194</td><td rowspan=1 colspan=1>83.52</td><td rowspan=1 colspan=1>88.06</td><td rowspan=1 colspan=1>48.35</td></tr><tr><td rowspan=2 colspan=2>70.5064.48</td><td rowspan=1 colspan=1>0.039</td><td rowspan=1 colspan=1>0.187</td><td rowspan=1 colspan=1>84.13</td><td rowspan=1 colspan=1>89.20</td><td rowspan=1 colspan=1>47.98</td></tr><tr><td rowspan=1 colspan=1>0.052</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>80.39</td><td rowspan=1 colspan=1>86.14</td><td rowspan=1 colspan=1>54.55</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=2>66.90</td><td rowspan=1 colspan=1>0.031</td><td rowspan=1 colspan=1>0.178</td><td rowspan=1 colspan=1>86.42</td><td rowspan=1 colspan=1>90.66</td><td rowspan=1 colspan=1>45.24</td></tr><tr><td rowspan=7 colspan=1>BloodMNIST</td><td rowspan=6 colspan=1>MC DropoutDDUTTAMasksemblesDeep EnsembleLaplace Redux</td><td rowspan=1 colspan=2>98.42</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1>0.027</td><td rowspan=1 colspan=1>93.81</td><td rowspan=1 colspan=1>96.12</td><td rowspan=1 colspan=1>22.36</td></tr><tr><td rowspan=2 colspan=2>98.5298.60</td><td rowspan=1 colspan=1>0.009</td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1>98.42</td><td rowspan=1 colspan=1>98.86</td><td rowspan=1 colspan=1>9.38</td></tr><tr><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>98.43</td><td rowspan=1 colspan=1>99.09</td><td rowspan=1 colspan=1>6.40</td></tr><tr><td rowspan=1 colspan=2>98.40</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>98.74</td><td rowspan=1 colspan=1>99.11</td><td rowspan=3 colspan=1>7.615.117.61</td></tr><tr><td rowspan=2 colspan=2>98.4580.35</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>99.06</td><td rowspan=1 colspan=1>99.50</td></tr><tr><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>0.029</td><td rowspan=1 colspan=1>94.23</td><td rowspan=1 colspan=1>97.16</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=2>98.72</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1>98.98</td><td rowspan=1 colspan=1>99.53</td><td rowspan=1 colspan=1>4.50</td></tr><tr><td rowspan=7 colspan=1>PathMNIST</td><td rowspan=6 colspan=1>MC DropoutDDUTTAMasksemblesDeep EnsembleLaplace Redux</td><td rowspan=1 colspan=2>94.14</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>0.096</td><td rowspan=1 colspan=1>89.43</td><td rowspan=1 colspan=1>68.70</td><td rowspan=2 colspan=1>25.1524.52</td></tr><tr><td rowspan=1 colspan=2>94.53</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>88.39</td><td rowspan=1 colspan=1>55.97</td></tr><tr><td rowspan=1 colspan=2>94.00</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.098</td><td rowspan=1 colspan=1>85.72</td><td rowspan=1 colspan=1>56.60</td><td rowspan=2 colspan=1>20.9823.38</td></tr><tr><td rowspan=1 colspan=2>94.74</td><td rowspan=1 colspan=1>0.037</td><td rowspan=1 colspan=1>0.098</td><td rowspan=1 colspan=1>85.36</td><td rowspan=1 colspan=1>57.25</td></tr><tr><td rowspan=2 colspan=2>95.0193.77</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.076</td><td rowspan=1 colspan=1>86.04</td><td rowspan=1 colspan=1>58.20</td><td rowspan=1 colspan=1>22.09</td></tr><tr><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>44.68</td><td rowspan=1 colspan=1>27.51</td><td rowspan=1 colspan=1>92.09</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=2>95.40</td><td rowspan=1 colspan=1>0.037</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>94.87</td><td rowspan=1 colspan=1>80.58</td><td rowspan=1 colspan=1>11.60</td></tr></table>

Table 4: Classification accuracy and OOD detection on natural and medical benchmarks.

Overall. Stochastic activations improves uncertainty quality while maintaining competitive task performance across most settings, with MiniImageNet the main exception in. The method performs well across visual domains and architectures, including U-Net, ResNetstyle classifiers, and Swin Transformer V2. These results support the main claim that useful predictive diversity can be obtained from a single trained model without changing the input, adding an uncertainty head, redesigning the architecture, or training multiple models.

## 5 Ablation Study

We analyze how the main design choices of stochastic activation switching affect model behavior. All ablations are conducted on CIFAR-10 with the same ResNet-style backbone and training protocol. We study four aspects: stochasticity granularity, the activation pair, information preservation, and inference-pass count. These analyses explain not only whether the method works, but also why local activation-level stochasticity provides predictive diversity without substantially disrupting the learned representation.

Granularity of Stochasticity. We compare layer-level, channel-level, and node-level stochastic activation sampling. Layer-level sampling shares one activation choice across an entire layer, channel-level sampling shares one choice within each feature channel, and node-level sampling independently samples the activation at each unit. As shown in Table 5, although layer-level stochasticity gives the lowest ECE, node-level sampling provides the strongest overall trade-off across predictive performance and uncertainty quality, improving accuracy, NLL [24], and Brier score. This indicates that injecting stochasticity at the most local scale creates richer functional diversity inside the network. This micro-level variation improves predictive uncertainty estimation while reducing local activation co-adaptation.

<table><tr><td rowspan=1 colspan=1>Granularity</td><td rowspan=1 colspan=1>Acc. (↑)</td><td rowspan=1 colspan=1>NLL (↓)</td><td rowspan=1 colspan=1>ECE (↓)</td><td rowspan=1 colspan=1>Brier (↓)</td></tr><tr><td rowspan=3 colspan=1>Layer-levelChannel-levelNode-level</td><td rowspan=1 colspan=1>93.04</td><td rowspan=1 colspan=1>0.204</td><td rowspan=1 colspan=1>0.005</td><td rowspan=3 colspan=1>0.1010.0940.088</td></tr><tr><td rowspan=2 colspan=1>93.6694.68</td><td rowspan=2 colspan=1>0.2000.162</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>0.009</td></tr></table>

Table 5: Ablation on stochastic activation granularity.

Choice of Stochastic Activation. We pair the backbone’s native activation with one alternative and ablate this choice in Table 6. ReLU+SiLU gives the best accuracy and NLL, whereas ReLU+Tanh gives the best ECE and Brier, indicating that smoother alternatives favor task performance while stronger nonlinear changes improve calibration. Using more than two candidate activations was less stable, likely due to excessive feature-distribution variation, so we use two-activation switching. We do not tune activations per dataset: SiLU is fixed for classification and Tanh for segmentation across datasets, modalities, and architectures. Other pairs remain competitive, suggesting that per-dataset tuning is unnecessary.

<table><tr><td>Activation Pair</td><td>Acc. (↑)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Brier (↓)</td></tr><tr><td>ReLU + Tanh [B5] ReLU + SiLU [] ReLU + GELU [] ReLU + Mish [3] ReLU + ELU [8]</td><td>94.43 94.68 94.12 93.98 93.67</td><td>0.176 0.162 0.184 0.192 0.206</td><td>0.008 0.009 0.012 0.014 0.018</td><td>0.085 0.088 0.095 0.101 0.109</td></tr></table>

Table 6: Effect of the activation pair used for stochastic switching.

Information Destruction Analysis. Unlike masking-based uncertainty methods, stochastic activation switching perturbs the nonlinear transformation while keeping feature pathways active. As shown in Fig. 5, our method gives lower IDR than MC Dropout across datasets, indicating that stochastic activation switching preserves intermediate representations under stochastic inference. This supports our motivation that predictive diversity can be induced through activation transformations without removing feature information.

![](images/b45ca754609f17c69c37593e7470b009915e7ca6e401fb3bf85e7c7554f28eaa.jpg)  
Figure 5: IDR comparison across datasets. Lower is better.

Number of inference passes. Finally, we study the effect of the number of stochastic forward passes used at inference time. Increasing T makes the predictive mean and uncertainty estimate more stable, but also increases inference time approximately linearly. Table 7 shows that uncertainty metrics improve as more passes are used, while accuracy saturates after a small number of samples. We therefore use T = 5 in the main experiments, as it provides a practical balance between uncertainty quality and computational cost.

<table><tr><td>Passes T</td><td>Accuracy (↑)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Brier (↓)</td><td>Time (s)</td></tr><tr><td>1</td><td>93.81</td><td>0.223</td><td>0.026</td><td>0.096</td><td>3.11</td></tr><tr><td>3</td><td>94.26</td><td>0.186</td><td>0.013</td><td>0.085</td><td>7.06</td></tr><tr><td>5</td><td>94.68</td><td>0.162</td><td>0.009</td><td>0.088</td><td>11.06</td></tr><tr><td>10</td><td>94.37</td><td>0.176</td><td>0.007</td><td>0.082</td><td>20.99</td></tr></table>

Table 7: Effect of the number of inference passes on acc, uncertainty quality, and runtime.

## 6 Conclusion

We reframe stochastic activation switching as an uncertainty estimation mechanism. Instead of relying on feature masking, input perturbations, or explicit ensembles, the method induces predictive diversity by sampling between nonlinear activation functions while keeping intermediate feature pathways active. Across classification, segmentation, and out-of-distribution detection experiments, the proposed approach improves uncertainty quality without degrading the main task performance, and remains effective across different visual domains and architectures. The results show that stochastic nonlinearities can provide a lightweight and information-preserving source of predictive variation from a single trained model, without requiring separate uncertainty heads or task-specific architectural redesign. A limitation is that the activation pair influences the trade-off between predictive performance and calibration, while repeated stochastic passes introduce additional inference cost. Future work can explore adaptive activation selection and more efficient sampling strategies to reduce inference overhead while preserving uncertainty estimation quality.

## References

[1] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra Acharya, et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion, 76:243–297, 2021.

[2] Favyen Bastani, Songtao He, Sofiane Abbar, Mohammad Alizadeh, Hari Balakrishnan, Sanjay Chawla, Sam Madden, and David DeWitt. RoadTracer: Automatic Extraction of Road Networks from Aerial Images . In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

[3] Patrick Bilic, Patrick Christ, Hongwei Bran Li, Eugene Vorontsov, Avi Ben-Cohen, Georgios Kaissis, Adi Szeskin, Colin Jacobs, Gabriel Efrain Humpire Mamani, Gabriel Chartrand, et al. The liver tumor segmentation benchmark (lits). Medical Image Analysis, 84:102680, 2023.

[4] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International Conference on Machine Learning, 2015.

[5] Glenn W. Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3, 1950.

[6] Albert Cardona, Stephan Saalfeld, Stephan Preibisch, Benjamin Schmid, Albert Cheng, John Pulokas, Pavel Tomancak, and Volker Hartenstein. An integrated micro- and macroarchitectural analysis of the drosophila brain by computer-assisted serial section electron microscopy. PLoS Biology, 8(10):e1000502, 2010.

[7] Eunseo Choi, Ho-Yeon Kim, Jaewon Lee, Taeyong jo, Myungjun lee, and Heejin Ahn. Uncertainty estimation via hyperspherical confidence mapping. In International Conference on Learning Representations, 2026.

[8] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). International Conference on Learning Representations, 2016.

[9] Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, and Radu Timofte. Swin2sr: Swinv2 transformer for compressed image super-resolution and restoration. In European Conference on Computer Vision, 2022.

[10] Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux-effortless bayesian deep learning. Advances in Neural Information Processing Systems, 2021.

[11] Kingma Diederik. Adam: A method for stochastic optimization. (No Title), 2014.

[12] Fabing Duan, François Chapeau-Blondeau, and Derek Abbott. Optimized injection of noise in activation functions to improve generalization of neural networks. Chaos, Solitons & Fractals, 178:114363, 2024.

[13] Nikita Durasov, Timur Bagautdinov, Pierre Baque, and Pascal Fua. Masksembles for uncertainty estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[14] Nikita Durasov, Doruk Oner, Jonathan Donier, Hieu Le, and Pascal Fua. Enabling uncertainty estimation in iterative neural networks. In International Conference on Machine Learning, 2024.

[15] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3– 11, 2018.

[16] Durmu¸s Ersoy and Burcu Erkmen. A stochastic computing method for generating activation functions in multilayer feedforward neural networks. Electrica, 21(3):376– 387, 2021.

[17] Muhammad Moazam Fraz, Paolo Remagnino, Andreas Hoppe, Bunyarit Uyyanonvara, Alicja R Rudnicka, Christopher G Owen, and Sarah A Barman. An ensemble classification-based approach applied to retinal blood vessel segmentation. IEEE Transactions on Biomedical Engineering, 59(9):2538–2548, 2012.

[18] Jan Funke, Fabian Tschopp, William Grisaitis, Arlo Sheridan, Chandan Singh, Stephan Saalfeld, and Srinivas C Turaga. Large scale image segmentation with structured loss based deep learning for connectome reconstruction. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(7):1669–1680, 2018.

[19] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, 2016.

[20] Yarin Gal, Jiri Hron, and Alex Kendall. Concrete dropout. Advances in Neural Infor mation Processing Systems, 2017.

[21] Jochen Gast and Stefan Roth. Lightweight probabilistic deep networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

[22] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. The MIT Press, 2016.

[23] Caglar Gulcehre, Marcin Moczulski, Misha Denil, and Yoshua Bengio. Noisy activation functions. In International Conference on Machine Learning, 2016.

[24] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, 2017.

[25] MM Hammad. Deep learning activation functions: fixed-shape, parametric, adaptive, stochastic, miscellaneous, non-standard, ensemble. arXiv preprint arXiv:2407.11090, 2024.

[26] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[27] D Hendrycks. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

[28] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors, 2012.

[29] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020.

[30] Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, and Kilian Q. Weinberger. Snapshot ensembles: Train 1, get m for free. In International Conference on Learning Representations, 2017.

[31] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In Advances in Neural Information Processing Systems, 2017.

[32] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[33] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 2012.

[34] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, 2017.

[35] Yann LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural Networks: Tricks ofthe Trade, pages 9–50. Springer, 2002.

[36] Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, and Balaj Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. In Advances in Neural Information Processing Systems, 2020.

[37] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[38] Maria Lomeli, Matthijs Douze, Gergely Szilvasy, Loic Cabannes, Jade Copet, Sainbayar Sukhbaatar, Jason Weston, Gabriel Synnaeve, Pierre-Emmanuel Mazaré, and Hervé Jégou. Stochastic activations. arXiv preprint arXiv:2509.22358, 2025.

[39] Antonio Loquercio, Mattia Segu, and Davide Scaramuzza. A general framework for uncertainty estimation in deep learning. IEEE Robotics and Automation Letters, 5(2): 3153–3160, 2020.

[40] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[41] Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. In Advances in Neural Information Processing Systems, 2018.

[42] Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7):3523–3542, 2021.

[43] Diganta Misra. Mish: A self regularized non-monotonic activation function. In British Machine Vision Conference, 2020.

[44] Volodymyr Mnih. Machine Learning for Aerial Image Labeling. PhD thesis, University of Toronto, 2013.

[45] Aryan Mobiny, Pengyu Yuan, Supratik K Moulik, Naveen Garg, Carol C Wu, and Hien Van Nguyen. Dropconnect is effective in modeling uncertainty of bayesian deep networks. Scientific Reports, 11(1):5458, 2021.

[46] Jishnu Mukhoti, Puneet K. Dokania, Philip Torr, and Yarin Gal. On batch normalisation for approximate bayesian inference. In Third Symposium on Advances in Approximate Bayesian Inference, 2021.

[47] Jishnu Mukhoti, Andreas Kirsch, Joost Van Amersfoort, Philip HS Torr, and Yarin Gal. Deep deterministic uncertainty: A new simple baseline. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[48] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In AAAI conference on artificial intelligence, 2015.

[49] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, volume 5, page 7. Granada, 2011.

[50] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. Advances in Neural Information Processing Systems, 2019.

[51] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016.

[52] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards realtime object detection with region proposal networks. In Advances in Neural Information Processing Systems, 2015.

[53] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention, 2015.

[54] Joes Staal, Michael D Abràmoff, Meindert Niemeijer, Max A Viergever, and Bram Van Ginneken. Ridge-based vessel segmentation in color images of the retina. IEEE Transactions on Medical Imaging, 23(4):501–509, 2004.

[55] Yana Stoyanova, Soroush Ghandi, and Maryam Tavakol. Toward robust uncertainty estimation with random activation functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 15152–15160, 2023.

[56] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. Branchynet: Fast inference via early exiting from deep neural networks. In International Conference on Pattern Recognition, 2016.

[57] Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a single deep deterministic neural network. In International Conference on Machine Learning, 2020.

[58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

[59] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in Neural Information Processing Systems, 2016.

[60] Guotai Wang, Wenqi Li, Michael Aertsen, Jan Deprest, Sébastien Ourselin, and Tom Vercauteren. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing, 338:34–45, 2019.

[61] Yeming Wen, Dustin Tran, and Jimmy Ba. Batchensemble: an alternative approach to efficient ensemble and lifelong learning. In International Conference on Learning Representations, 2020.

[62] Christian Wiedemann, Christian Heipke, Helmut Mayer, and Olivier Jamet. Empirical evaluation of automatically extracted road axes. Empirical evaluation techniques in computer vision, 12:172–187, 1998.

[63] Lijun Wu, Juntao Li, Yue Wang, Qi Meng, Tao Qin, Wei Chen, Min Zhang, Tie-Yan Liu, et al. R-drop: Regularized dropout for neural networks. Advances in Neural Information Processing Systems, 2021.

[64] Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li. Empirical evaluation of rectified activations in convolutional network. arXiv preprint arXiv:1505.00853, 2015.

[65] Jiancheng Yang, Rui Shi, and Bingbing Ni. Medmnist classification decathlon: A lightweight automl benchmark for medical image analysis. In IEEE 18th International Symposium on Biomedical Imaging, 2021.

[66] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification. Scientific Data, 10(1):41, 2023.

[67] Tal Zeevi, Ravid Shwartz-Ziv, Yann LeCun, Lawrence H Staib, and John A Onofrey. Rate-in: Information-driven adaptive dropout rates for improved inference-time uncertainty estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20757–20766, 2025.

# Supplementary Material

## A Approach Overview

Stochastic Activation Switching replaces each deterministic activation in the backbone during training and inference with a per-unit Bernoulli switch between the backbone’s native activation $\sigma _ { \mathrm { o r i g } }$ and a fixed alternative $\sigma _ { \mathrm { a l t } }$ . During training, a fresh activation configuration is sampled at every forward pass, and the network is optimized with the standard task loss, so that the learned weights become robust to activation-level variation rather than to a single fixed nonlinearity. At inference time, the trained weights are kept fixed and T stochastic forward passes are drawn from the same input; the empirical mean of the resulting predictions is used as the point estimate, and the empirical variance across passes is used as the uncertainty estimate. Because every activation pathway remains active in all passes, stochasticity changes only the shape of the nonlinearity applied at each unit, without removing features from the intermediate representation. This perturbation-without-removal property is the qualitative distinction from dropout-style mechanisms [19] and the basis for the uncertainty estimator analyzed in Section C.

## B Implementation Details

Activation pair. For classification, we retain the backbone’s native activation as $\sigma _ { \mathrm { o r i g } }$ and use $\sigma _ { \mathrm { a l t } } = \mathrm { S i L U }$ [15]. For segmentation, we likewise retain the native activation and use $\sigma _ { \mathrm { a l t } } =$ tanh. These fixed task-level choices are used across datasets and architectures, following the activation-choice ablation in Table 6 of the main paper.

Granularity and sampling rate. We use node-level granularity with an independent mask per activation unit and $p = 0 . 5 ;$ both choices follow the granularity ablation in Table 5 of the main paper. The choice $p = 0 . 5$ maximizes the entropy of the Bernoulli switch, giving both activations equal probability and avoiding a systematic preference for either nonlinearity. Channel- and layer-level variants share a single mask across a channel or an entire layer and are not used in the main results.

Training. Training follows the standard supervised pipeline of the corresponding deterministic baseline for each dataset. The optimizer, learning rate schedule, batch size, augmentation, and number of epochs are kept identical to those of the deterministic backbone. The only modification is that each activation block performs Bernoulli switching at every forward pass during both training and inference.

Inference. At inference time, the trained weights are kept fixed and T = 5 stochastic forward passes are drawn per input, following the inference-pass ablation in Table 7 of the main paper. The empirical mean is used as the prediction, and the empirical variance across the T passes is used as the uncertainty estimate, as defined in Equations (7)–(8) of the main paper.

Backbones. For segmentation, we use a U-Net architecture [53] on Massachusetts [44], DRIVE [54], and LiTS [3], and Swin Transformer V2 [37] on CREMI [6]. For natural-image classification, we use a ResNet-style [26] convolutional backbone on CIFAR-10 [32] and MiniImageNet [59]. For high-resolution medical classification, we use Swin Transformer V2 on BloodMNIST and PathMNIST [66]. In all cases, the backbone matches the corresponding deterministic baseline, and only the activation block is replaced by the stochastic switch.

Datasets and splits. For segmentation, Massachusetts Roads is used with the official trainvalidation-test split, and RoadTracer [2] serves as the held-out cross-dataset OOD evaluation set. DRIVE is used with the standard train/test split, and CHASE-DB1 [17] is used for crossdataset OOD evaluation. LiTS axial slices are used as 2D inputs. CREMI is used with its published volume split, and ISBI 2012 [18] is used for cross-dataset OOD evaluation. For classification, CIFAR-10 uses the canonical 50,000/10,000 train/test split, with SVHN [49] serving as the OOD set. MiniImageNet is split following the standard 600 images per class across 100 classes, with CIFAR-100 used as the OOD set. BloodMNIST and PathMNIST follow the MedMNIST v2 official splits at $2 2 4 \times 2 2 4$ resolution and are treated as a crossdomain OOD pair for each other.

Augmentation. For segmentation training, inputs are augmented with random crops, random horizontal and vertical flips, and $9 0 ^ { \circ }$ rotations; no augmentation is applied at test time. For natural-image classification, we apply random crops with 4-pixel padding and random horizontal flips. For high-resolution medical classification, we apply random resized crops and horizontal flips. All inputs are channel-normalized using dataset-specific means and standard deviations.

Training hyperparameters. Table 8 summarizes the optimizer, learning rate, batch size, and number of epochs used for each backbone. All models are trained from scratch with the standard supervised cross-entropy loss for classification and a combined soft-Dice plus binary cross-entropy loss for segmentation. The optimizer and schedule are identical for the deterministic baseline and for Stochastic Activation Switching; only the activation block differs.
<table><tr><td>Backbone</td><td>Dataset</td><td>Optimizer</td><td>Batch</td><td>LR / schedule</td><td>Epochs</td></tr><tr><td>U-Net</td><td>Massachusetts</td><td>Adam []</td><td>32</td><td> $\overline { { 1 \times 1 0 ^ { - 3 } } }$  , cosine</td><td>200</td></tr><tr><td>U-Net</td><td>DRIVE</td><td>Adam</td><td>8</td><td> $1 \times 1 0 ^ { - 3 }$  , cosine</td><td>200</td></tr><tr><td>U-Net</td><td>LiTS</td><td>Adam</td><td>64</td><td> $1 \times 1 0 ^ { - 4 }$  , cosine</td><td>200</td></tr><tr><td>Swin-V2</td><td>CREMI</td><td>AdamW []</td><td>32</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>200</td></tr><tr><td>ResNet-18</td><td>CIFAR-10</td><td>SGD (m=0.9)</td><td>256</td><td>0.1, cosine</td><td>200</td></tr><tr><td>ResNet-50</td><td>MiniImageNet</td><td>SGD (m=0.9)</td><td>128</td><td>0.1, cosine</td><td>200</td></tr><tr><td>Swin-V2</td><td>BloodMNIST</td><td>AdamW</td><td>128</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>200</td></tr><tr><td>Swin-V2</td><td>PathMNIST</td><td>AdamW</td><td>256</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>200</td></tr></table>

Table 8: Per-dataset training hyperparameters used for all methods. Weight decay is $\overline { { 5 \times } } 1 0 ^ { - 4 }$ for SGD and $1 \times 1 0 ^ { - 4 }$ for AdamW.

Baseline configurations. MC Dropout [19] uses a dropout rate of $p _ { d } = 0 . 2$ inserted after each activation block and kept active at inference with T = 5 stochastic forward passes. DDU [47] applies post-hoc density estimation to features from the trained deterministic backbone. Masksembles [13] uses five fixed partially overlapping masks, producing five predictions at inference. Deep Deep Ensembles [34] are formed from five independently trained models with different random seeds; the prediction is the per-class probability average over members and the uncertainty is the empirical variance across members. Laplace Redux [10] applies a last-layer Laplace approximation to the trained backbone with a diagonal covariance and prior precision tuned on the validation split. Test-Time Augmentation [60] samples five augmented versions of each test input (random crops and horizontal flips for classification; the segmentation augmentation pipeline above for dense prediction) and averages the predictions. The iterative segmentation baseline [14] uses three refinement iterations following the configuration of its released reference implementation.

Compute. All experiments were run with PyTorch on a single NVIDIA H100 GPU. Inference with $T = 5$ stochastic forward passes increases the per-input inference cost by approximately a factor of five over the deterministic backbone, as reported in Table 7 of the main paper. All results in this work are reported from a single random seed per method, except for the five-member Deep Ensemble baseline.

## C Mathematical Analysis of Stochastic Activation Switching

We provide a mathematical characterization of stochastic activation switching. We show that the method defines a well-posed stochastic predictor, derive closed-form conditional moments for a single stochastic activation unit, bound the induced per-unit variance under a bounded activation gap, and analyze the convergence of the empirical Monte Carlo estimators used at inference time.

## C.1 Notation and Setup

Fix a neural network $f _ { \theta } : \mathcal { X }  \mathcal { Y }$ with parameters θ, whose nonlinear blocks contain activation units indexed by layer l and unit $\dot { j . } \operatorname { L e t } z _ { j } ^ { ( l , t ) }$ denote the pre-activation of unit $j$ in layer l during stochastic forward pass t. Let $\sigma _ { \mathrm { o r i g } } , \dot { \sigma } _ { \mathrm { a l t } } : \mathbb { R } $ R be the original and alternative scalar activation functions. At pass $t \in \{ 1 , . . . , T \}$ and unit $( l , j )$ , draw an independent Bernoulli mask

$$
m _ { j } ^ { ( l , t ) } \sim \mathrm { B e r n } ( p ) , \qquad p \in ( 0 , 1 ) ,
$$

and define the stochastic activation output as

$$
h _ { j } ^ { ( l , t ) } = \big ( 1 - m _ { j } ^ { ( l , t ) } \big ) \sigma _ { \mathrm { o r i g } } \big ( z _ { j } ^ { ( l , t ) } \big ) + m _ { j } ^ { ( l , t ) } \sigma _ { \mathrm { a l t } } \big ( z _ { j } ^ { ( l , t ) } \big ) .\tag{8}
$$

This is the same Bernoulli activation-switching rule used in the main paper. Unless stated otherwise, we analyze the node-level case, where each unit samples its own mask. For channel-level or layer-level sampling, the same mask is shared across a channel or across an entire layer.

In the following derivations, expectations are taken with the current pre-activation $z _ { j } ^ { ( l , t ) }$ fixed. This is important because, in deeper layers, $z _ { j } ^ { ( l , t ) }$ is itself affected by random activation choices made in previous layers. Thus, we first analyze the local randomness caused by the current mask, and full-network expectations can then be obtained by averaging over the upstream randomness as well. For readability, we omit layer and pass indices when the meaning is clear.

## C.2 The Stochastic Predictor and Empirical Estimators

Let

$$
\sigma ^ { ( t ) } = \big ( m _ { j } ^ { ( l , t ) } \big ) _ { l , j }
$$

denote the collection of all activation-switching masks sampled during stochastic forward pass t. With the trained weights θ fixed, the output of the network for input x is

$$
\eta ^ { ( t ) } = f _ { \theta } \Big ( x ; \sigma ^ { ( t ) } \Big ) .\tag{9}
$$

Thus, each stochastic pass produces a random prediction only through the sampled activation configuration. If the original and alternative activations differ for at least one unit that affects the output, then different mask samples can produce different predictions. If the two activations are identical, the stochastic predictor reduces to the deterministic network.

Over $T$ stochastic forward passes, the prediction trajectory is

$$
\left\{ \eta ^ { ( 1 ) } , \eta ^ { ( 2 ) } , \dots , \eta ^ { ( T ) } \right\} .
$$

Following the main paper, we estimate the predictive mean and uncertainty by

$$
\hat { \mu } _ { T } ( x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \eta ^ { ( t ) } , \qquad \hat { U } _ { T } ( x ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Big ( \eta ^ { ( t ) } - \hat { \mu } _ { T } ( x ) \Big ) ^ { 2 } .\tag{10}
$$

Here, ${ \hat { \mu } } _ { T } ( x )$ is the Monte Carlo estimate of the expected prediction, and $\hat { U } _ { T } ( x )$ is the empirical variance across stochastic predictions. Their population counterparts are

$$
\mu _ { \infty } ( x ) = \mathbb { E } \Big [ \eta ^ { ( t ) } \Big ] , \qquad V _ { \infty } ( x ) = \mathrm { V a r } \Big [ \eta ^ { ( t ) } \Big ] ,\tag{11}
$$

which correspond to the ideal mean and variance under infinitely many stochastic activation samples.

## C.3 Per-Unit Moments and Bounded Activation Gap

We first analyze the stochasticity introduced at a single activation unit. Conditioned on a fixed pre-activation value $z ,$ the only source of randomness is the Bernoulli switching variable m. Thus, the unit output is a two-point random variable: it takes the original activation value with probability $1 - p ,$ , and the alternative activation value with probability $p .$

Lemma C.1 (Unit-wise moments). For any fixed $z \in \mathbb { R }$ , the stochastic activation output h in Equation (8) satisfies

$$
\mathbb { E } [ h \mid z ] = ( 1 - p ) \sigma _ { \mathrm { o r i g } } ( z ) + p \sigma _ { \mathrm { a l t } } ( z ) ,\tag{12}
$$

and

$$
\operatorname { V a r } [ h \mid z ] = p ( 1 - p ) \left( \sigma _ { \mathrm { a l t } } ( z ) - \sigma _ { \mathrm { o r i g } } ( z ) \right) ^ { 2 } .\tag{13}
$$

Therefore, the per-unit variance is zero whenever the two activation functions produce the same value at z. In other words, stochastic activation switching creates local uncertainty only where the activation pair disagrees.

Proof. Since $m \sim \mathbf { B e r n } ( p )$ , we have $\mathbb { E } [ m ] = p$ . Taking expectation in Equation (8) directly gives Equation (12). For the second moment, because $m \in \{ 0 , 1 \}$ , we have $m ^ { 2 } = m$ . Thus,

$$
\mathbb { E } [ h ^ { 2 } \mid z ] = ( 1 - p ) \sigma _ { \mathrm { o r i g } } ^ { 2 } ( z ) + p \sigma _ { \mathrm { a l t } } ^ { 2 } ( z ) .
$$

Subtracting $\left( \mathbb { E } [ h \mid z ] \right) ^ { 2 }$ gives Equation (13).

Lemma C.1 shows that the amount of stochasticity introduced by one unit is controlled by two quantities: the switching probability $p ,$ and the gap between the two activation functions at the current pre-activation value. If this activation gap is bounded, then the induced per-unit variance is also bounded.

Theorem C.1 (Bounded noise envelope). Let

$$
\Delta = \operatorname* { s u p } _ { z \in \mathbb { R } } \left. \sigma _ { \mathrm { a l t } } ( z ) - \sigma _ { \mathrm { o r i g } } ( z ) \right. .\tag{14}
$$

If $\Delta < \infty ,$ then for every $z \in \mathbb { R }$

$$
\operatorname { V a r } [ h \mid z ] \leq p ( 1 - p ) \Delta ^ { 2 } \leq { \frac { 1 } { 4 } } \Delta ^ { 2 } .\tag{15}
$$

The upper bound is tight at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$

Proof. From Lemma C.1,

$$
\operatorname { V a r } [ h \mid z ] = p ( 1 - p ) \left( \sigma _ { \mathrm { a l t } } ( z ) - \sigma _ { \mathrm { o r i g } } ( z ) \right) ^ { 2 } .
$$

By definition of $\Delta ,$

$$
\bigl ( \sigma _ { \mathrm { a l t } } ( z ) - \sigma _ { \mathrm { o r i g } } ( z ) \bigr ) ^ { 2 } \leq \Delta ^ { 2 } .
$$

Also, for $p \in [ 0 , 1 ]$ , we have $\begin{array} { r } { p ( 1 - p ) \leq \frac { 1 } { 4 } } \end{array}$ , with equality at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ . Combining these two facts proves the result. □

Remark C.1 (Per-unit bound only). The bound in Theorem C.1 applies to a single activation unit. It does not directly imply a bound on the full network prediction variance, because perunit perturbations can be amplified, damped, or combined as they propagate through later layers.

For activation pairs with unbounded global gap, such as ReLU–Tanh, the same argument gives a bound on any bounded pre-activation range. In particular, if $z \in [ - R , R ]$ , one may replace $\Delta$ in Equation (14) by

$$
\Delta _ { R } = \operatorname* { s u p } _ { z \in [ - R , R ] } \left. \sigma _ { \mathrm { a l t } } ( z ) - \sigma _ { \mathrm { o r i g } } ( z ) \right. ,
$$

which is finite whenever the activation functions are finite on the interval.

For some activation pairs, the global gap ∆ can be computed explicitly. The following corollary gives this value for the ReLU–SiLU pair, which is used in our classification experiments.

Corollary C.1 (ReLU–SiLU activation-gap constant). Let

$$
\sigma _ { \mathrm { o r i g } } ( z ) = \operatorname { R e L U } ( z ) = \operatorname* { m a x } ( 0 , z ) , \qquad \sigma _ { \mathrm { a l t } } ( z ) = \operatorname { S i L U } ( z ) = z \operatorname { s i g m o i d } ( z ) .
$$

Define the gap function

$$
\phi ( z ) = \sigma _ { \mathrm { a l t } } ( z ) - \sigma _ { \mathrm { o r i g } } ( z ) .\tag{16}
$$

Then

$$
\phi \left( z \right) = \left\{ \begin{array} { l l } { - z \mathrm { s i g m o i d } ( - z ) , } & { z \geq 0 , } \\ { z \mathrm { s i g m o i d } ( z ) , } & { z < 0 . } \end{array} \right.\tag{17}
$$

Thefunction $\phi$ is non-positive and even, meaning $\phi ( - a ) = \phi ( a )$ . Moreover,

$$
\Delta _ { \mathrm { S i L U } } = \operatorname* { s u p } _ { z \in \mathbb { R } } | \phi ( z ) | = z ^ { \star } \mathrm { s i g m o i d } ( - z ^ { \star } ) \approx 0 . 2 7 8 4 6 ,\tag{18}
$$

where $z ^ { \star } \approx 1 . 2 7 8 4 6$ is the unique positive solution of

$$
e ^ { z ^ { \star } } ( z ^ { \star } - 1 ) = 1 ,\tag{19}
$$

equivalently,

$$
z ^ { \star } { \mathrm { s i g m o i d } } ( z ^ { \star } ) = 1 .
$$

Proof. For $z \geq 0$ , we have

$$
\phi ( z ) = z \mathrm { s i g m o i d } ( z ) - z = - z \big ( 1 - \mathrm { s i g m o i d } ( z ) \big ) = - z \mathrm { s i g m o i d } ( - z ) .
$$

For $z < 0 , { \mathrm { R e L U } } ( z ) = 0 ,$ so

$$
\phi ( z ) = z \mathrm { s i g m o i d } ( z ) .
$$

For any $a > 0$

$$
\phi ( a ) = - a \mathrm { s i g m o i d } ( - a ) , \qquad \phi ( - a ) = - a \mathrm { s i g m o i d } ( - a ) ,
$$

so $\phi ( - a ) = \phi ( a )$ . Thus, the gap is symmetric around zero.

It remains to find the maximum magnitude. For $z > 0$

$$
\phi ( z ) = - z \mathrm { s i g m o i d } ( - z ) .
$$

Differentiating gives

$$
\phi ^ { \prime } ( z ) = \mathrm { s i g m o i d } ( - z ) \left( z \mathrm { s i g m o i d } ( z ) - 1 \right) .
$$

Since sigmoid $\left( - z \right) > 0$ , the stationary point satisfies

$$
z { \mathrm { s i g m o i d } } ( z ) = 1 .
$$

This is equivalent to

$$
z = 1 + e ^ { - z } \quad \Longleftrightarrow \quad e ^ { z } ( z - 1 ) = 1 .
$$

Solving this equation gives $z ^ { \star } \approx 1 . 2 7 8 4 6$ , and therefore

$$
| \phi ( z ^ { \star } ) | = z ^ { \star } \mathrm { s i g m o i d } ( - z ^ { \star } ) \approx 0 . 2 7 8 4 6 .
$$

By symmetry, the negative side has the same maximum magnitude.

Corollary C.2 (Numerical bound at $\begin{array} { r } { p = \frac { 1 } { 2 } ) } \end{array}$ . For ReLU–SiLU switching with $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$

$$
\operatorname* { s u p } _ { z \in \mathbb { R } } \mathrm { V a r } [ h \mid z ] = \frac { 1 } { 4 } \Delta _ { \mathrm { S i L U } } ^ { 2 } \approx 0 . 0 1 9 3 9 .\tag{20}
$$

## C.4 Convergence of the Empirical Estimators

We now analyze the Monte Carlo estimators used at inference time. For a fixed input x, each stochastic forward pass produces a random prediction $\eta ^ { ( t ) }$ . Throughout this subsection, we assume that masks are resampled independently across stochastic forward passes, so that $\eta ^ { ( 1 ) } , \ldots , \eta ^ { ( T ) }$ are i.i.d. draws from the stochastic predictor for the fixed input x. The empirical mean ${ \hat { \mu } } _ { T } ( x )$ estimates the expected prediction, while the empirical variance ${ \hat { U } } _ { T } ( x )$ estimates the predictive variance induced by stochastic activation switching.

Theorem C.2 (Unbiasedness and concentration of $\hat { \mu } _ { T } )$ . The empirical mean is an unbiased estimator ofthe population mean:

$$
\mathbb { E } [ \hat { \mu } _ { T } ( x ) ] = \mu _ { \infty } ( x ) .\tag{21}
$$

Assume the scalar output ofinterest is bounded almost surely, so that

$$
\eta ^ { ( t ) } \in [ a , b ] \qquad f o r \ : a l l \ : t .
$$

This assumption holds for post-softmax class probabilities, pixel-wise probabilities in semantic segmentation, and sigmoid-bounded scalar outputs, but not for unbounded logits without additional assumptions. Then,for any $\delta > 0$

$$
\operatorname* { P r } ( | \hat { \mu } _ { T } ( x ) - \mu _ { \infty } ( x ) | \ge \delta ) \le 2 \exp \left( - \frac { 2 T \delta ^ { 2 } } { ( b - a ) ^ { 2 } } \right) .\tag{22}
$$

For a K-dimensional bounded output, applying Equation (22) coordinate-wise and using a union bound replaces the leading constant 2 by 2K. For pixel-wise outputs, the same argument applies coordinate-wise to each pixel and class probability.

Proof. Unbiasedness follows from linearity of expectation:

$$
\mathbb { E } [ \hat { \mu } _ { T } ( x ) ] = \mathbb { E } \Bigg [ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \eta ^ { ( t ) } \Bigg ] = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \eta ^ { ( t ) } ] = \mu _ { \infty } ( x ) .
$$

The concentration bound in Equation (22) is Hoeffding’s inequality applied to the i.i.d. bounded sequence $\{ \eta ^ { ( t ) } \} _ { t = 1 } ^ { T }$ . The K-dimensional version follows by applying the same bound to each coordinate and taking a union bound. □

The empirical variance ${ \hat { U } } _ { T } ( x )$ is computed with normalization $1 / T$ , as in the main paper. As with the usual sample variance, this estimator is slightly biased for finite T, but the bias vanishes as T increases.

Lemma C.2 (Bias of $\hat { U } _ { T } )$ . The empirical uncertainty estimator satisfies

$$
\mathbb { E } [ \hat { U } _ { T } ( x ) ] = \frac { T - 1 } { T } V _ { \infty } ( x ) .\tag{23}
$$

Therefore, the Bessel-corrected estimator

$$
\hat { U } _ { T } ^ { \mathrm { u } } = \frac { T } { T - 1 } \hat { U } _ { T }
$$

is unbiased.

Proof. Let η denote a generic draw from the same distribution as each $\eta ^ { ( t ) }$ . Using the sample-variance identity,

$$
\sum _ { t = 1 } ^ { T } \left( \eta ^ { ( t ) } - \hat { \mu } _ { T } \right) ^ { 2 } = \sum _ { t = 1 } ^ { T } \eta ^ { ( t ) 2 } - T \hat { \mu } _ { T } ^ { 2 } .
$$

Taking expectations gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \eta ^ { ( t ) 2 } \right] = T \mathbb { E } [ \eta ^ { 2 } ] ,
$$

and

$$
\mathbb { E } \left[ T \hat { \mu } _ { T } ^ { 2 } \right] = T \left( \mu _ { \infty } ^ { 2 } + \frac { V _ { \infty } } { T } \right) .
$$

Dividing by T yields

$$
\mathbb { E } [ \hat { U } _ { T } ] = V _ { \infty } \left( 1 - \frac { 1 } { T } \right) = \frac { T - 1 } { T } V _ { \infty } .
$$

Theorem C.3 (Convergence of $\hat { U } _ { T } )$ . Under the bounded-output assumption of Theorem C.2, the empirical variance converges to the population predictive variance:

$$
\hat { U } _ { T } ( x ) \to V _ { \infty } ( x ) \quad i n L ^ { 2 } \quad a s T \to \infty .\tag{24}
$$

Moreover, its root-mean-squared error is oforder ${ \cal O } ( T ^ { - 1 / 2 } )$

Proof. We decompose the mean-squared error into squared bias and variance:

$$
\mathbb { E } \Big [ \big ( \hat { U } _ { T } - V _ { \infty } \big ) ^ { 2 } \Big ] = \big ( \mathbb { E } [ \hat { U } _ { T } ] - V _ { \infty } \big ) ^ { 2 } + \mathbf { V } \mathbf { a r } ( \hat { U } _ { T } ) .\tag{25}
$$

By Lemma C.2, the bias is

$$
\mathbb { E } [ \hat { U } _ { T } ] - V _ { \infty } = - \frac { V _ { \infty } } { T } .
$$

Since the output is bounded in $[ a , b ]$ , Popoviciu’s inequality gives

$$
V _ { \infty } \leq \frac { ( b - a ) ^ { 2 } } { 4 } .
$$

Therefore, the squared bias is bounded by

$$
\frac { ( b - a ) ^ { 4 } } { 1 6 T ^ { 2 } } = O ( T ^ { - 2 } ) .
$$

For the variance term, the Bessel-corrected estimator can be written as the degree-2 Ustatistic

$$
\hat { U } _ { T } ^ { \mathrm { u } } = \frac { 1 } { T ( T - 1 ) } \sum _ { i \neq j } \frac { 1 } { 2 } \left( \eta ^ { ( i ) } - \eta ^ { ( j ) } \right) ^ { 2 } .\tag{26}
$$

Its kernel is

$$
h ( u , \nu ) = \frac { 1 } { 2 } ( u - \nu ) ^ { 2 } .
$$

Because $u , \nu \in [ a , b ]$ , the kernel is bounded:

$$
h ( u , \nu ) \in \left[ 0 , { \frac { ( b - a ) ^ { 2 } } { 2 } } \right] .
$$

The standard variance bound for bounded U-statistics gives

$$
\mathrm { V a r } \big ( \hat { U } _ { T } ^ { \mathrm { u } } \big ) = O \bigg ( \frac { ( b - a ) ^ { 4 } } { T } \bigg ) .
$$

Since

$$
\mathrm { V a r } ( \hat { U } _ { T } ) = \left( \frac { T - 1 } { T } \right) ^ { 2 } \mathrm { V a r } \big ( \hat { U } _ { T } ^ { \mathrm { u } } \big ) ,
$$

the same ${ \cal O } ( T ^ { - 1 } )$ rate holds for $\mathrm { V a r } ( \hat { U } _ { T } )$

Combining the ${ \cal O } ( T ^ { - 2 } )$ squared-bias term with the ${ \cal O } ( T ^ { - 1 } )$ variance term gives

$$
\mathbb { E } \Big [ \big ( \hat { U } _ { T } - V _ { \infty } \big ) ^ { 2 } \Big ] = O ( T ^ { - 1 } ) .
$$

Taking the square root gives root-mean-squared error ${ \cal O } ( T ^ { - 1 / 2 } )$ , proving $L ^ { 2 }$ convergence.

The constants in the above concentration and U-statistic bounds are conservative. In practice, the number of inference passes should be chosen by balancing uncertainty stability and computational cost, as studied in the inference-pass ablation of the main paper.

## D Computational Cost and Timing Analysis

We analyze the computational cost of stochastic activation switching in terms of both training and inference time. The analysis separates the overhead introduced by the stochastic activation mechanism itself from the cost of performing multiple stochastic forward passes.

Measurement protocol. All timings are measured on a single NVIDIA H100 GPU. For CIFAR-10, training time corresponds to one training epoch and inference time corresponds to evaluation over the complete test set. All methods use the same backbone and input setting. Our method and MC Dropout use $T = 5$ stochastic forward passes, while Deep Ensembles use five independently trained models. The reported times therefore capture the complete end-to-end cost of each uncertainty-estimation procedure rather than only the deterministic backbone execution.

Sampling overhead. To isolate the sampling overhead under the same architecture and input setting, we compare the deterministic model, MC Dropout, Deep Ensembles, and our method on CIFAR-10. The deterministic model requires 40.05s for one training epoch and 2.83s for full test-set inference. MC Dropout requires 40.97 s and 13.66 s, respectively, while our method requires $4 2 . 7 3 \pm 0 . 3 6 \mathrm { s }$ for training and $1 4 . 2 1 \pm 0 . 1 8 \mathrm { s }$ for inference. Thus, our method has training and inference costs close to MC Dropout under the same $T = 5$ sampling setting, with only a small additional overhead from stochastic activation switching. A five-member Deep Ensemble requires 164.52 s for training and 13.27 s for inference. Therefore, our method has a similar multi-pass inference cost to Deep Ensembles while requiring substantially less training and model storage, since only a single trained network is retained instead of five independently trained models.

Tables 9 and 10 compare the wall-clock computational cost across classification and segmentation. Training time corresponds to one complete epoch over the training set; for Deep Ensembles, the reported value includes one epoch for each of the five independently trained members. Inference time corresponds to processing the complete test set using the full inference procedure of each method.

Across both tasks, stochastic activation switching retains a single-model training profile and has training cost close to other single-model approaches, while avoiding the repeated optimization and model storage required by Deep Ensembles. Its main computational overhead occurs at inference because uncertainty estimation requires multiple stochastic forward passes. Accordingly, its inference cost is generally comparable to other multi-evaluation approaches, including MC Dropout and Deep Ensembles, while TTA similarly incurs additional cost from repeated augmented evaluations. In contrast, deterministic inference, DDU, Laplace Redux, and Masksembles generally provide lower wall-clock inference latency.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Training Time (s) ↓</td><td rowspan=1 colspan=1>Inference Time (s) ↓</td></tr><tr><td rowspan=1 colspan=1>Deterministic</td><td rowspan=1 colspan=1>40.05</td><td rowspan=2 colspan=1>2.8313.662.8618.433.1613.273.07</td></tr><tr><td rowspan=1 colspan=1>MC DropoutDDUTTAMasksemblesDeep EnsembleLaplace Redux</td><td rowspan=1 colspan=1>40.9749.9340.0543.30194.5257.05</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>42.73±0.36</td><td rowspan=1 colspan=1>14.21±0.18</td></tr></table>

Table 9: Training and inference time on CIFAR-10.
<table><tr><td rowspan="2">Method</td><td colspan="2">Massachusetts</td><td colspan="2">DRIVE</td><td colspan="2">LiTS</td><td colspan="2">CREMI</td></tr><tr><td>Train</td><td>Infer.</td><td>Train</td><td>Infer.</td><td>Train</td><td>Infer.</td><td>Train</td><td>Infer.</td></tr><tr><td>Deterministic</td><td>54.41</td><td>1.52</td><td>37.30</td><td>24.30</td><td>28.27</td><td>10.98</td><td>36.98</td><td>9.22</td></tr><tr><td>MC Dropout</td><td>54.54</td><td>4.46</td><td>38.28</td><td>140.43</td><td>32.92</td><td>55.78</td><td>39.27</td><td>52.90</td></tr><tr><td>DDU</td><td>74.41</td><td>3.01</td><td>51.70</td><td>26.40</td><td>48.38</td><td>11.09</td><td>52.45</td><td>9.36</td></tr><tr><td>Masksembles</td><td>57.91</td><td>1.90</td><td>39.40</td><td>29.60</td><td>33.73</td><td>12.51</td><td>37.64</td><td>12.51</td></tr><tr><td>Deep Ensemble</td><td>261.79</td><td>4.25</td><td>214.74</td><td>135.11</td><td>103.84</td><td>54.47</td><td>189.48</td><td>49.42</td></tr><tr><td>Ours</td><td>58.46±3.44</td><td>4.89±0.62</td><td>38.80±2.70</td><td>145.50±2.10</td><td>32.10±0.77</td><td>59.69±0.58</td><td>38.03±2.36</td><td>62.57±1.33</td></tr></table>

Table 10: Training and inference time on the segmentation benchmarks. All times are reported in seconds.

These results clarify that the main computational advantage of stochastic activation switching lies in single-model training and storage rather than minimum inference latency. The additional inference computation provides the predictive diversity used for uncertainty estimation and accompanies the calibration and OOD-detection improvements reported in the main paper.

## E Corruption Robustness

We evaluate classification uncertainty under input corruption using CIFAR-10-C with Gaussian noise at severities 1, 3, and 5. This setting tests whether uncertainty estimates remain meaningful as the input distribution becomes degraded. Table 11 reports the full uncertainty and calibration results, while Figure 6 visualizes the accuracy degradation.

Under mild corruption, our method preserves strong predictive accuracy and obtains the best Brier score, indicating that stochastic activation trajectories provide useful uncertainty without strongly affecting the main prediction. At moderate and severe corruption, MC Dropout achieves the highest accuracy, suggesting that stronger feature removal can act as a noise-robust regularizer under heavy corruption. However, this robustness comes from explicitly masking activations and therefore removing part of the intermediate representation. In contrast, our method introduces micro-level stochasticity through activation switching, which keeps the information flow mostly intact while still producing diverse predictions for uncertainty estimation. As a result, our method remains close to MC Dropout in accuracy while obtaining competitive uncertainty metrics. At the highest corruption severity, our method achieves the best NLL [24] and ECE [48], while Laplace Redux obtains the best Brier score but with substantially lower accuracy.

![](images/d93158d1d230563230fe543ffb015e4b1ad40be2a121317348df35479b329995.jpg)

Figure 6: Accuracy under CIFAR-10-C Gaussian noise corruption at severities 1, 3, and 5. MC Dropout gives slightly higher accuracy under heavier corruption, likely because feature masking acts as a stronger noise-robust regularizer. Our method remains close in accuracy while preserving information flow through activation switching instead of feature removal.
<table><tr><td rowspan=1 colspan=1>Severity</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Acc. (↑)</td><td rowspan=1 colspan=1>NLL (↓)</td><td rowspan=1 colspan=1>ECE ()</td><td rowspan=1 colspan=1>Brier (↓)</td></tr><tr><td rowspan=2 colspan=1>1</td><td rowspan=2 colspan=1>MC DropoutDeep EnsembleTTÁLaplace ReduxOurs</td><td rowspan=1 colspan=1>82.9584.9629.2028.29</td><td rowspan=1 colspan=1>0.5170.4863.0302.315</td><td rowspan=1 colspan=1>0.0150.0420.4010.213</td><td rowspan=1 colspan=1>0.2430.2201.0190.872</td></tr><tr><td rowspan=1 colspan=1>83.96</td><td rowspan=1 colspan=1>0.521</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>0.217</td></tr><tr><td rowspan=2 colspan=1>3</td><td rowspan=2 colspan=1>MC DropoutDeep EnsembleTTALaplace ReduxOurs</td><td rowspan=2 colspan=1>43.8943.0312.0812.3943.11</td><td rowspan=1 colspan=1>1.8892.2124.1183.043</td><td rowspan=1 colspan=1>0.1840.2760.4290.222</td><td rowspan=1 colspan=1>0.7560.8081.1520.993</td></tr><tr><td rowspan=1 colspan=1>2.165</td><td rowspan=1 colspan=1>0.202</td><td rowspan=1 colspan=1>0.767</td></tr><tr><td rowspan=2 colspan=1>5</td><td rowspan=2 colspan=1>MC DropoutDeep EnsembleTTALaplace ReduxOurs</td><td rowspan=1 colspan=1>17.2215.8711.5111.44</td><td rowspan=1 colspan=1>3.5484.0124.0113.863</td><td rowspan=1 colspan=1>0.2630.2660.3530.288</td><td rowspan=1 colspan=1>1.0121.0451.0890.970</td></tr><tr><td rowspan=1 colspan=1>16.98</td><td rowspan=1 colspan=1>2.980</td><td rowspan=1 colspan=1>0.162</td><td rowspan=1 colspan=1>1.085</td></tr></table>

Table 11: CIFAR-10-C Gaussian noise corruption results at severities 1, 3, and 5.

## F OOD Uncertainty Distributions

To further inspect the OOD behavior of stochastic activation switching, we visualize the distribution of uncertainty scores for ID and OOD samples. Figure 7 shows that ID samples usually form compact low-uncertainty distributions, while OOD samples shift toward larger and more dispersed uncertainty values. This indicates that the model remains confident on familiar inputs and becomes less certain when the input distribution changes. The separation is especially clear for DRIVE→CHASE-DB1, CIFAR-10→SVHN, and CREMI→ISBI, where OOD samples occupy a broader high-uncertainty range. For Massachusetts→RoadTracer, the two distributions overlap more strongly because both datasets contain aerial road imagery with similar visual structure; nevertheless, the OOD distribution still shows a wider uncertainty spread. This supports the quantitative OOD results and shows that stochastic activation trajectories provide an interpretable signal for detecting distribution shift.

![](images/642055ee0c8a8c9ecf3b2e5174ca6616442de14a4860583a59e589e9bd915592.jpg)

![](images/1e27c088d2e3f72ce6bfe610e1f22294e6bceff94cc55dac0a98507f4a5a16c0.jpg)

![](images/5a390276244ab6462c326d4d3b26ceddcef6c4d84106c854efed595f8e5e7b75.jpg)

![](images/6671bd28e3a352d241335fd30531cd9f1a690826a26b28afe22a254085ee0437.jpg)  
Figure 7: Uncertainty comparison of ID and OOD samples. ID scores are concentrated at lower uncertainty, while OOD scores are dispersed and shifted toward higher uncertainty.

Qualitative OOD segmentation examples are shown in Fig. 8, where a model trained on CREMI is evaluated on ISBI 2012. Each row shows the original image, ground-truth mask, prediction, and variance-based uncertainty map. The uncertainty maps are concentrated around ambiguous membranes, boundary regions, and structures where the stochastic forward passes disagree. This provides spatial evidence that the model does not only assign a global OOD score, but also localizes where the cross-dataset shift creates uncertainty in the segmentation output.

![](images/7208d727d1917672f2e56537056f948eaa35a7390965dbe0a8b12c615e0c4d6f.jpg)  
Figure 8: Qualitative OOD segmentation examples on ISBI 2012 using a model trained on CREMI. Each row shows the original image, ground-truth mask, prediction, and variancebased uncertainty map.