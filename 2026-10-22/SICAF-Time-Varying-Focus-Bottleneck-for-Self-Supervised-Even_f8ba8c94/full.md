# SICAF: Time-Varying Focus Bottleneck for Self-Supervised Event-Based Optical Flow with Spiking Neural Network

Shuangming Yang<sup>∗</sup> Tianjin University Tianjin, China yangshuangming@tju.edu.cn

Shangqi Guo Tsinghua University Beijing, China shangqi\_guo@mail.tsinghua.edu.cn

Qing He Tianjin University Tianjin, China 3020234423@tju.edu.cn

Badong Chen Xi’an Jiaotong University Xi’an, China chenbd@mail.xjtu.edu.cn

## Abstract

Event-based optical flow estimation achieves high temporal resolution and dynamic range by exploiting asynchronous events, while self-supervised learning provides an effective solution to the absence of ground-truth labels. Spiking neural networks (SNNs), as brain-inspired models, process information through asynchronous spikes, naturally aligning with optical flow events. However, the temporal irregularity across event streams hinders the compact extraction of spatiotemporal features, while the absence of explicit labels leads to spurious correlations, thereby increasing redundancy within latent representations. In this paper, we propose a novel “Compactly Compress, Time-varying Focus” training strategy. We present SICAF (Spikes Information Compression And Focus) framework that formulates the learning process as a constrained optimization with temporal modulation. It encourages self-supervised SNNs to extract compact latent representations while preventing over-compression that would discard temporal saliency. To better enable feature selectivity across time steps, we design an estimator with learnable temporal attention, achieving context awareness with cross-time dependencies. Experimental results on several datasets demonstrate that SICAF achieves state-of-the-art performance in self-supervised SNNs, with a 8.14% improvement in AEE on the MVSEC dataset and a 13.49% increase in robustness under white-box adversarial attacks.

## 1 Introduction

Event-based optical flow estimation is an advanced technique that utilizes event cameras to measure motion in visual scenes [1–3]. With high temporal resolution and low power consumption, event cameras outperform frame-based sensors by delivering precise, rapid motion detection, enabling energy-efficient optical flow estimation in high-speed and low-light environments [4, 5]. Since ground-truth optical flow labels are difficult to obtain, recent work has adopted self-supervised learning for event-based optical flow without annotation [6–9].

Emerging as a promising candidate for neuromorphic computing, spiking neural networks (SNNs) employ temporal coding mechanisms to efficiently encode time-dependent dynamic information [10, 11]. Besides, SNNs operate in an event-driven spiking manner, allowing flexible processing of asynchronous and sparse data [12, 13]. These properties make SNNs highly promising for handling event-based optical flow [1, 14]. Recent work on SNNs has successfully developed self-supervised learning strategies for event-based optical flow estimation [7, 6, 15], and further incorporated learnable membrane-potential parameters to improve their representational capacity [7]. Yang et al. [16] propose SeLHIB, a self-supervised high-order information bottleneck training scheme, which improves the robustness of event-based optical flow estimation.

Despite these significant advances, a glaring limitation persists: variations in motion patterns give rise to event streams that are highly irregular along the temporal dimension, while existing optical flow learning methods largely overlook such temporal heterogeneity. This omission hinders the model’s ability to capture informative motion cues in the event stream, especially in complex scenarios involving fast motion, adverse illumination conditions, or highly dynamic backgrounds. This brings us to an intriguing inquiry: Is it feasible to design an SNN learning strategy that intrinsically accounts for the temporal evolution of spike-based latent representations when extracting compact yet informative motionfeatures?

![](images/8b1b04671dfcf8269cac1c669f58c14ad10ae2f0e5de960a778eacbb553acaa5.jpg)  
Figure 1: Visualization of performance evaluation with previous self-supervised SNN (top) and the proposed SICAF (bottom). Our SICAF enhances the estimation accuracy by emphasizing temporal saliency. Concurrently, our SICAF effectively captures structured yet informative motion features, improving the issues of incorrect flow direction estimation and the loss of fine-grained motion details.

In response to the aforementioned challenge, we propose a pioneering training strategy termed Compactly Compress, Time-varying Focus. Central to our approach is a novel adaptive focus mechanism equipped with a flexible information compression strategy across time steps to achieve temporal feature selectivity. While learning compact latent representations, the proposed approach explicitly takes into account the temporal heterogeneity of the input event streams. As illustrated in Fig. 1, previous self-supervised SNNs often suffer from incorrect optical flow direction estimation and the loss of fine-grained motion details, whereas our approach strengthens the capability of SNNs to represent motion patterns with more accurate optical flow predictions.

In event-based optical flow estimation, general self-supervised SNNs typically adopt contrast maximization, as illustrated in Fig. 2(a). First, such self-supervised SNNs lack a mechanism for compressing redundant and noisy information, and therefore fail to extract compact, task-relevant spatiotemporal representations from event data. Second, contrast maximization drives learning solely based on the spatial alignment of events, without exploiting their timestamps or temporal distribution. SeLHIB [16], as shown in Fig. 2(b), captures more compact representations by formulating information bottleneck (IB) constraint. Nevertheless, this form of IB fails to consider the irregular temporal distribution of event streams. A uniform information bottleneck applied across time is inadequate for capturing temporally salient features. As a result, jointly handling temporal non-uniformity and redundancy compression remains somewhat uncharted.

Targeting the above challenge, we propose Spikes Information Compression And Focus (SICAF), which formulates the learning process as a constrained optimization with temporal modulation. As depicted in Fig. 2(c), SICAF learns a latent and compact space in a time-varying compression manner while capturing temporal salient feature across time steps. To this end, we first design a novel estimator with learnable temporal attention to modulate the compression process. Specifically, to capture temporal dynamics in the representation learning process, we map the spike firing rates of latent representations into queries and keys to compute temporal attention scores. This enables the model to capture feature variations along the temporal dimension and provides a time-varying focusing capability for the input event stream. Building on Rényi’s α-entropy, the estimator robustly measures the entropy of high-dimensional latent representations at different time steps. These entropy values, weighted by the temporal attention scores, form a time-varying focusing constraint that modulates the strength of information compression. Extensive experiments demonstrate that SICAF achieves the top performance in self-supervised SNNs. Overall, the main contributions of this paper are:

![](images/19523e8ea44114188b4382d28e163d654e74d37e8a99b1dc86eb7715493f124f.jpg)  
Figure 2: Comparison between the general self-supervised SNNs [7, 6, 15], SeLHIB [16], and ours for handling optical flow events with self-supervised learning strategy. (a) General selfsupervised SNNs typically rely on contrast maximization. This objective enhances the spatial contrast of motion-compensated events to drive learning, but it neglects the temporal non-uniformity of event data. (b) SeLHIB presents an information bottleneck constraint to encourage compact latent representations. However, because it enforces a uniform compression constraint across time, it struggles to preserve temporally salient features. (c) SICAF addresses these issues by formulating a constrained optimization process with temporally modulated focus. Within this framework, a learnable temporal-attention–based estimator allows SICAF to explicitly model temporal heterogeneity, encouraging the formation of compact yet more temporally informative representations.

• We present a novel "compactly compress, time-varying focus" paradigm, which learns to enhance the temporal saliency of representations while suppressing redundant event information.

• We propose SICAF, a constrained optimization framework with temporal modulation by neural activities. It enjoys a concise and tractable objective and enables adaptive, timevarying compression of irregular optical-flow events.

• We design a learnable temporal-attention estimator that directly extracts fine-grained temporal cues from spiking neural activities and dynamically guides the compression process.

• Experiments demonstrate that SICAF achieves superior performance compared with existing state-of-the-art SNN-based optical flow models, while significantly enhancing the robustness. SICAF can serve as a strong baseline and inspire further advancements in the field of neuromorphic computing and brain-inspired intelligence.

## 2 Related Work

## 2.1 Event-based Optical Flow Estimation with SNNs

Event-based optical flow estimation with SNNs has attracted growing interest. Early work exploited adaptive neurons and spike timing-dependent plasticity (STDP) to develop unsupervised SNN training for efficient high-bandwidth flow estimation [14]. Hybrid architectures such as Spike-FlowNet further showed that SNN encoders can match artificial neural network (ANN) performance when paired with ANN decoders [8]. Subsequent efforts introduced self-supervised frameworks leveraging temporal cues from short event slices [6], multi-sensor fusion with frame-based inputs [17], and adaptive neural mechanisms for improved spatiotemporal extraction [7]. S-TLLR proposed a three-factor temporal local learning rule inspired by STDP, achieving state-of-the-art performance under self-supervision [18]. ANN-to-SNN conversion approaches have also been explored to enhance estimation accuracy [9]. Directly trained SNNs, however, offer unique advantages by fully exploiting the intrinsic temporal dynamics of spiking neurons, enabling low-latency and efficient spatiotemporal feature learning [19, 20]. In this work, we follow this direction and focus on direct training of SNNs to unlock their full potential for event-based optical flow estimation.

## 2.2 Direct Training of SNNs with the IB Principle

Recent advances have applied the IB principle to directly train SNNs. Methods such as HOSIB and HHO-IB introduce higher-order IB formulations for surrogate-gradient learning, yielding notable gains in generalization and robustness [21, 22]. SeLHIB further presents an IB-based self-supervised framework for event-based optical flow with SNNs, substantially improving robustness under noisy conditions [23]. However, this uniform temporal compression suppresses critical temporal cues. We address this limitation by proposing a strategy that preserves compactness while selectively retaining temporally salient features.

## 3 Preliminary

Spiking Neuron Model. Spiking neurons are the core computational units of SNNs. They integrate incoming postsynaptic potentials, emit a spike once the membrane potential crosses a firing threshold, and thereby enable efficient event-driven processing of asynchronous data. The mathematical modeling is presented below:

$$
u _ { p r e } ^ { ( t ) } = \tau u ^ { ( t - 1 ) } + i ^ { ( t ) } ,\tag{1}
$$

$$
u ^ { ( t ) } = u _ { p r e } ^ { ( t ) } ( 1 - o ^ { ( t ) } ) ,\tag{2}
$$

where $u _ { p r e } ^ { ( t ) }$ and $u ^ { ( t ) }$ denote the pre-update and post-update membrane potentials in the leaky integrateand-fire (LIF) model, respectively. $i ^ { ( t ) }$ represents the synaptic input current at timestep t, while $o ^ { ( t ) } \in \{ 0 , 1 \}$ indicates the binary output spike generated by the spiking neuron. The detailed neuron parameter settings are provided in Appendix A.4.

Contrast Maximization. Contrast maximization is a self-supervised approach for optical flow estimation that exploits the spatiotemporal misalignment of events in a continuous stream [24]. By applying motion compensation, events are warped to a common reference frame, reducing motioninduced blur. Assuming $u ( x )$ denotes the optical flow at each pixel location and $\bar { t } ( x , y ; u \mid t _ { \mathrm { r e f } } )$ the average timestamp at each pixel. The objective is to minimize the normalized sum of squared average timestamps, yielding the following convex contrastive loss:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t } } ( t _ { \mathrm { r e f } } ) = \frac { \sum _ { x } \bar { t } _ { p = + } ( x ; u \mid t _ { \mathrm { r e f } } ) ^ { 2 } + \bar { t } _ { p = - } ( x ; u \mid t _ { \mathrm { r e f } } ) ^ { 2 } } { \sum _ { x } \left[ n ( x ^ { \prime } ) > 0 \right] + \epsilon } .\tag{3}
$$

## 4 Method

Fig. 2(c) visually elucidates the overall procedure of SICAF. After being encoded through $E ( Z | X )$ the input X is mapped to the latent representation $Z ,$ and the upper bound of the mutual information between X and $Z ,$ denoted $I ( X ; Z ) \leq I _ { c } .$ , enforces compact compression during the feature extraction process. We compute time-focused scores from the spike-rate matrices of Z along the temporal dimension, enabling contextual awareness across time dependencies. Moreover, we employ the Rényi’s α-entropy estimator to robustly quantify the information content of the latent variables. The entropy is multiplied by the time-focused scores, forming an estimator with learnable temporal attention that maximizes the entropy estimation to preserve the temporal heterogeneity of event streams. The pseudocode of the method is provided in Appendix A.5, and the generalization bound as well as the sample complexit bound theorems and their proofs are presented in AppendixA.7.

## 4.1 Temporal Attention Scores

Unlike conventional ANNs, SNNs communicate through discrete spike events, allowing temporal information to be encoded directly in spike timing as activity propagates across layers. Motivated by this inherent temporal structure, we compute attention weights over latent representations across time, forming a temporal attention matrix $T _ { A }$ . Specifically, for a latent representation $Z \in \mathbb { R } ^ { T \times C \times D }$ , we first compute its spike rate matrix $T _ { Z } \in \mathbb { R } ^ { \hat { T } \times C }$ , and then project it into two latent spaces as follows:

$$
Z _ { Q } = W _ { Q } \cdot T _ { Z } , \quad Z _ { K } = W _ { K } \cdot T _ { Z }\tag{4}
$$

where $Z _ { Q } , Z _ { K } \in \mathbb { R } ^ { T \times T }$ represent temporal queries and keys, respectively. The temporal attention matrix is computed as:

$$
A = \mathrm { s o f t m a x _ { d i m = 2 } } \left( { \frac { Z _ { Q } \cdot Z _ { K } ^ { T } } { \sqrt { d _ { k } } } } \right) ,\tag{5}
$$

where $d _ { k }$ is the key dimension. The softmax is applied along the key temporal dimension (the second dimension of the attention matrix), such that $\begin{array} { r } { \sum _ { j = 1 } ^ { T } A _ { i j } = 1 } \end{array}$ for each query timestep i.We then average the attention weights over all query timesteps to obtain the normalized temporal attention distribution:

$$
T _ { A , j } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } A _ { i j } , \qquad j = 1 , \ldots , T .\tag{6}
$$

where $T _ { A } ~ \in ~ \mathbb { R } ^ { T \times 1 }$ assigns an attention weight to each timestep and satisfies $T _ { A , j } ~ \geq ~ 0$ and $\textstyle \sum _ { j = 1 } ^ { T } T _ { A , j } \ = \ 1$ . Since event-based streams exhibit sparse and irregular temporal distributions, $T _ { A }$ explicitly captures this variability and provides a principled basis for the temporally selective accumulation mechanism used in SICAF.

![](images/d9437f842caf7842ab4b6a94e5fd7ef8b5bb3d0c6114339763b5cc1f2f26d344.jpg)  
Figure 3: Overview of the time-sensitive Rényi’s α-entropy estimator. Temporal attention scores are derived from spike firing rates of latent representations to capture temporal heterogeneity. Rényi’s α-entropy provides a robust and flexible measure of information content, enabling selective feature compression when combined with temporal attention.

## 4.2 Time-Sensitive Rényi’s α-Entropy Estimator

We construct a time-sensitive Rényi’s α-entropy estimator $H T _ { \alpha } ( Z )$ , as illustrated in Fig 3, and propose a spike-rate-based temporal selection information bottleneck. In information theory, Rényi’s α-entropy can measure the amount of information contained in a variable while exhibiting better Loss of fine-numerical stability and scalability compared to Shannon entropy. It thus provides a robust and flexible direction estimation <sup>grained</sup> <sup>details</sup>theoretical tool for estimating high-dimensional temporal information features.

Given a kernel function $k : \chi \times \chi \to \mathbb { R }$ , the Gram matrix $K$ in the latent variable $Z$ can be calculated as: $K _ { i j } = k ( Z _ { i } , Z _ { j } )$ , and the normalized positive semi-definite matrix Φ is computed as: $\begin{array} { r } { \Phi _ { i j } = \frac { K _ { i j } } { \sqrt { K _ { i i } K _ { j j } } } } \end{array}$

For the latent variable $Z ,$ its Rényi’s α-entropy is given by:

$$
\begin{array} { l } { \displaystyle { H _ { \alpha } ( Z ) = \frac { 1 } { 1 - \alpha } \log _ { 2 } \left( \mathrm { t r } ( \Phi ^ { \alpha } ) \right) } } \\ { \displaystyle { \ = \frac { 1 } { 1 - \alpha } \log _ { 2 } \left( \sum _ { i = 1 } ^ { N } \lambda _ { i } ( \Phi ) ^ { \alpha } \right) , } } \end{array}\tag{7}
$$

where $\lambda _ { i }$ denotes the $i _ { t h }$ eigenvalue of the latent representation $Z .$ In the subsection 4.1, we compute the temporal attention scores based on spike-rate temporal information, thereby constructing a time-sensitive Rényi’s α entropy estimator

$$
H T _ { \alpha } ( Z ) = \sum _ { j = 1 } ^ { T } T _ { A j } H _ { \alpha } ( Z _ { j } ) ,\tag{8}
$$

This estimator flexibly evaluates the information content of the latent variable across the temporal dimension, adaptively assigning different attention weights. Thus, in the entropy estimation process, it preserves the temporal heterogeneity of the latent variable.

## 4.3 SICAF Learning Strategy

The IB framework achieves redundant information compression by minimizing the mutual information $I ( Z ; X )$ between the input X and the latent representation $Z .$ However, most existing SNN-based IB approaches fail to consider the intrinsic temporal characteristics of SNNs, leading to limited advantages in temporal sequence tasks such as optical flow estimation. To address this limitation, we construct a time-sensitive Rényi’s α-entropy estimator and propose a spike-based temporal attention information bottleneck learning strategy.

We introduce $Y$ as an abstract task-output variable for the theoretical formulation; it denotes neither available ground-truth optical flow labels nor explicitly generated pixel-wise pseudo-labels.For input X and output $Y$ , we use an encoder $E ( Z | \dot { X } )$ that projects the input X into the latent space to obtain a representation $Z .$ . The objective of SICAF is to realize Compactly Compress and Time-varying Focus during this mapping process. Specifically, Compactly Compress imposes an upper-bound constraint on the mutual information between $X$ and $Z \colon { \mathsf { \bar { I } } } ( X ; Z ) \leq I _ { c }$ , while Time-varying Focus enforces a temporal modulation constraint during compression, expressed as: $\begin{array} { r } { H T _ { \alpha } ( Z ) = \sum _ { i = 1 } ^ { T } A _ { i } H _ { \alpha } ( Z _ { i } ) \geq H _ { c } } \end{array}$ , where the learning process is described as an optimization problem with temporal modulation constraints.

Assuming that the mapping between the latent representation $Z$ and the output $Y$ is defined by $q ( y | z )$ the optimization objective can be expressed as:

$$
\begin{array} { r l r } {  { \mathcal { L } ( q , E ) = \underset { E , q } { \mathrm { m i n } } \ \mathbb { E } _ { x , y \sim p ( x , y ) } \big [ \mathbb { E } _ { z \sim E ( z \mid x ) } [ - \log q ( y | z ) ] \big ] } } \\ & { } & { \quad \quad s . t . \quad I ( X ; Z ) \le I _ { c } , \quad H T _ { \alpha } ( Z ) \ge H _ { c } , } \end{array}\tag{9}
$$

where the computation of $H T _ { \alpha } ( Z )$ is detailed in subsection 4.2. The mutual information $I ( X ; Z )$ is defined as:

$$
\begin{array} { l } { { \displaystyle I ( X ; Z ) = \int p ( x , z ) \log \frac { p ( x , z ) } { p ( x ) p ( z ) } d x d z \ ~ } } \\ { { \displaystyle ~ = \int p ( x ) \mathbb { E } _ { E ( z \mid x ) } \log \frac { E ( z \mid x ) } { p ( z ) } d x d z , } } \end{array}\tag{10}
$$

where $p ( x )$ and $p ( z )$ denote the marginal distributions of the input variable and the latent variable, respectively. Since the distribution $p ( z )$ of the latent variable is often intractable, we approximate it with a variational distribution $r ( z )$ that satisfies the KL divergence constraint $\mathrm { K L } [ p ( \dot { z } ) ] \lVert \boldsymbol { r } ( z ) ] \ge 0$ Hence,

$$
\int p ( z ) \log p ( z ) d z \geq \int p ( z ) \log r ( z ) d z .\tag{11}
$$

Using this inequality, we obtain an upper bound for the mutual information $I ( X ; Z )$

$$
\begin{array} { l } { { \displaystyle I ( X ; Z ) \leq \int p ( x ) \mathbb { E } _ { E ( z \mid x ) } \log \frac { E ( z \mid x ) } { r ( z ) } d x d z } } \\ { { \displaystyle \qquad = \mathbb { E } _ { x \sim p ( x ) } [ \mathrm { K L } [ E ( z \mid x ) \| r ( z ) ] ] . } } \end{array}\tag{12}
$$

Therefore, the constrained optimization objective can be approximated by the following upper bound:

$$
\begin{array} { r l r } {  { \widetilde { \mathcal { L } } ( q , E ) = \underset { E , q } { \mathrm { m i n } } \ \mathbb { E } _ { x , y \sim p ( x , y ) } \big [ \mathbb { E } _ { z \sim E ( z \mid x ) } [ - \log q ( y \vert z ) ] \big ] } } \\ & { } & { s . t . \quad \mathbb { E } _ { x \sim p ( x ) } [ \mathrm { K L } [ E ( z \vert x ) \| r ( z ) ] ] \le I _ { c } , } \\ & { } & { H T _ { \alpha } ( Z ) \ge H _ { c } . \quad \quad } \end{array}\tag{13}
$$

For the model training process, we introduce regularization coefficients $\beta$ and $\gamma$ to transform the constrained optimization into an unconstrained training objective:

$$
\begin{array} { r l } & { J _ { \mathrm { S I C A F } } = \underset { E , q } { \mathrm { m i n } } ~ \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { y } \sim p ( \boldsymbol { x } , \boldsymbol { y } ) } \left[ \mathbb { E } _ { \boldsymbol { z } \sim E ( \boldsymbol { z } \mid \boldsymbol { x } ) } [ - \log q ( \boldsymbol { y } \vert \boldsymbol { z } ) ] \right] } \\ & { ~ + \beta \cdot \left( \mathbb { E } _ { \boldsymbol { x } \sim p ( \boldsymbol { x } ) } [ \mathrm { K L } [ E ( \boldsymbol { z } \vert \boldsymbol { x } ) \Vert r ( \boldsymbol { z } ) ] ] - I _ { c } \right) } \\ & { ~ - \gamma \cdot \left( H T _ { \alpha } ( \boldsymbol { Z } ) - H _ { c } \right) . } \end{array}\tag{14}
$$

![](images/6b94acbf34531b64b4c4072d942474cb30bec3f015133cae5a020cc01380877b.jpg)  
(a)

![](images/ebace9360db34f9aaab26b508e7feac1d3c1ccac6e2d7ed8423afa9c33136c53.jpg)  
(b)  
Figure 4: (a) Comparison of feature maps in latent space. Both SeLHIB and our SICAF demonstrates sparser feature maps compared to baseline self-supervised SNN, indicating the acquisition of more structured spatiotemporal features. (b) The radar charts of SSIM for self-supervised SNN, SeLHIB, and SICAF, respectively.

## 4.4 Mechanistic Analysis of Temporal Heterogeneity in SICAF

To evaluate the effectiveness of SICAF in achieving compact compression and time-varying focusing for irregular event streams, we visualize the latent feature maps of trained models and measure the structural similarity of their neuronal firing patterns across consecutive time steps. As shown in Fig. 4(a), the SNN baseline exhibits the densest activations, whereas both SeLHIB and SICAF produce noticeably sparser patterns. This suggests that both of these two strategies encourage more compact and better-structured feature representations.

To more intuitively examine how the firing patterns of the three methods vary across time steps, we plot radar charts of structural similarity (SSIM) to measure the similarity between adjacent time steps. The SSIM is defined as:

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } ) ( 2 \sigma _ { x y } + C _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) } ,\tag{15}
$$

where $\mu _ { x } , \sigma _ { x } ^ { 2 } , \mu _ { y }$ , and $\sigma _ { y } ^ { 2 }$ denote the mean and variance of images x and $y , \sigma _ { x y }$ is their covariance, and $C _ { 1 }$ and $C _ { 2 }$ are stabilization constants. As shown in Fig. 4(b), SICAF achieves consistently lower SSIM scores than both the conventional self-supervised SNN and the SeLHIB baseline. Because SSIM measures structural similarity between temporally adjacent firing maps, lower SSIM indicates stronger temporal decorrelation, implying that the network produces activation patterns with higher temporal resolution. In essence, the lower SSIM values observed in SICAF reflect a theoretically grounded effect: SICAF enforces temporal diversity during minimizing redundant dependencies, thereby promoting a temporally selectivity regime that aligns with the irregular statistics of eventbased optical flow.

## 5 Experiments

## 5.1 Experimental Setup

We demonstrate the effectiveness of SICAF on event-based optical flow estimation task with different datasets, including multi-vehicle stereo event-camera (MVSEC), high quality frames (HQF) [25], and event camera dataset (ECD) [26]. All experiments were conducted using the PyTorch framework on RTX 4090 and A6000 GPUs. Adam optimizer is used with learning rate of 0.0002 and batch size of 8 for 50 epochs. Nearly 15 minutes of event data are contained in the training sequences, and they are split into 140 sequences with 128×128 size with 500k events in each sequence. Optical flow predictions are produced at each frame timestep with dt=2, i.e., 25% of the window events at a time.

## 5.2 Performance Evaluation on Event-based Optical Flow Estimation

We evaluated the generalization ability of SICAF in the self-supervised optical flow estimation task across the MVSEC, ECD, and HQF datasets, and compared it with the SOTA methods. The results on the MVSEC dataset are shown in Table 1, while the results on the ECD and HQF datasets are shown in Appendix A.8. The results demonstrate that SICAF significantly improves the accuracy of SNN in self-supervised optical flow estimation tasks. Specifically, compared to the work that applies IB to the SNN self-supervised optical flow task [23], SICAF achieves a 8.16% improvement, which suggests that the integration of compact compression and time-varying focusing enables SICAF to achieve compact yet selective feature extraction, improving the model’s generalization ability. Furthermore, we compared the qualitative optical flow estimation results of SICAF and the method without SICAF, as shown in Figure 5. From the comparison of the flow estimation results, it is noticeable that SICAF provides more accurate and consistent motion direction predictions, while preserving finer motion details and producing clearer and more complete motion contours.

Table 1: Experimental results of SICAF on MVSEC compared to previous work. (lower is better, best in red, runner-up in blue).
<table><tr><td rowspan="2">Methods</td><td colspan="5">Average Endpoint Error (AEE) ↓</td></tr><tr><td>Indoor (flying1)</td><td>Indoor (flying2)</td><td>Indoor (flying3)</td><td>Outdoor (day1)</td><td>Avg.</td></tr><tr><td>LIF-EV-FlowNet [6]</td><td>0.71</td><td>1.44</td><td>1.16</td><td>0.53</td><td>0.96</td></tr><tr><td>LIF-FireNet [6]</td><td>0.98</td><td>1.77</td><td>1.50</td><td>0.57</td><td>1.21</td></tr><tr><td>ALIF-EV-FlowNet [6]</td><td>1.00</td><td>1.78</td><td>1.55</td><td>0.57</td><td>1.23</td></tr><tr><td>PLIF-EV-FlowNet [6]</td><td>0.75</td><td>1.52</td><td>1.23</td><td>0.60</td><td>1.03</td></tr><tr><td>XLIF-EV-FlowNet [6]</td><td>0.73</td><td>1.45</td><td>1.17</td><td>0.45</td><td>0.95</td></tr><tr><td>ALIF-FireNet [6]</td><td>1.04</td><td>1.85</td><td>1.58</td><td>0.62</td><td>1.27</td></tr><tr><td>PLIF-FireNet [6]</td><td>0.90</td><td>1.67</td><td>1.41</td><td>0.56</td><td>1.14</td></tr><tr><td>XLIF-FireNet [6]</td><td>0.98</td><td>1.82</td><td>1.54</td><td>0.54</td><td>1.22</td></tr><tr><td>Adaptive-SpikeNet [7]</td><td>0.84</td><td>1.59</td><td>1.36</td><td>0.48</td><td>1.07</td></tr><tr><td>Adaptive-SpikeNet [7]</td><td>0.79</td><td>1.37</td><td>1.11</td><td>0.44</td><td>0.93</td></tr><tr><td>SeLHIB [23]</td><td>0.74</td><td>1.48</td><td>1.21</td><td>0.50</td><td>0.98</td></tr><tr><td>FSFN [15]</td><td>0.76</td><td>1.19</td><td>1.00</td><td>0.50</td><td>0.86</td></tr><tr><td>EV-FlowNet + SICAF</td><td>0.71</td><td>1.37</td><td>1.09</td><td>0.43</td><td>0.90</td></tr><tr><td>FSFN + SICAF</td><td>0.73</td><td>1.09</td><td>0.93</td><td>0.42</td><td>0.79</td></tr></table>

## 5.3 Robustness of Event-based Optical Flow Estimation in Noisy Situations

Due to environmental interference, event cameras often generate spurious events caused by nonphysical brightness changes. These noisy events can interfere with the model’s ability to extract accurate motion features. To further investigate the impact of SICAF on the robustness of SNNs in self-supervised optical flow estimation, we artificially injected various types of noise into the input event streams and compared the results with a baseline model, as shown in Table 2. The results demonstrate that SICAF significantly enhances robustness under both random and adversarial noise, achieving a 13.49% improvement in accuracy under white-box adversarial conditions. Additional experimental results on the ECD and HQF datasets are provided in Appendix A.9. SICAF enhances robustness under white-box adversarial attacks through its time-sensitive adaptive compression mechanism. The time-varying compression framework enables SICAF to dynamically focus on critical events while suppressing irrelevant or noisy data, leading to more robust and structured feature extraction and thus making it less susceptible to adversarial perturbations. It also adjusts the weighting of temporal features, further aids in capturing temporal heterogeneity and reducing the impact of adversarial noise.

![](images/5061c38101e7881175c16ad7f3a94565a1d8c96b10e4e3f4fa0d0bddf6708952.jpg)  
Figure 5: Qualitative evaluation of optical flow estimation results on sequences from the MVSEC dataset.

Table 2: Experimental results of SICAF on MVSEC under different noise (lower is better, best in red, runner-up in blue).
<table><tr><td rowspan="2">Noise type</td><td rowspan="2">Methods</td><td colspan="5">AEE</td></tr><tr><td>Indoor (flying1)</td><td>Indoor (flying2)</td><td>Indoor (flying3)</td><td>Outdoor (day1)</td><td>Average</td></tr><tr><td rowspan="4">SP</td><td>LIF-EV-FlowNet [6]</td><td>0.91</td><td>1.68</td><td>1.39</td><td>0.56</td><td>1.14</td></tr><tr><td>XLIF-EV-FlowNet [6]</td><td>0.86</td><td>1.76</td><td>1.21</td><td>0.62</td><td>1.11</td></tr><tr><td>SICAF-LIF</td><td>0.82</td><td>1.44</td><td>1.20</td><td>0.60</td><td>1.01</td></tr><tr><td>SICAF-XLIF</td><td>0.75</td><td>1.43</td><td>1.18</td><td>0.44</td><td>0.95</td></tr><tr><td rowspan="4">Gaussian</td><td>LIF-EV-FlowNet [6]</td><td>0.95</td><td>1.72</td><td>1.44</td><td>0.62</td><td>1.18</td></tr><tr><td>XLIF-EV-FlowNet [6]</td><td>0.93</td><td>1.70</td><td>1.39</td><td>0.62</td><td>1.16</td></tr><tr><td>SICAF-LIF</td><td>0.83</td><td>1.49</td><td>1.25</td><td>0.59</td><td>1.04</td></tr><tr><td>SICAF-XLIF</td><td>0.94</td><td>1.68</td><td>1.42</td><td>0.53</td><td>1.14</td></tr><tr><td rowspan="4">Black-box attack</td><td>LIF-EV-FlowNet [6]</td><td>1.15</td><td>1.96</td><td>1.72</td><td>0.79</td><td>1.41</td></tr><tr><td>XLIF-EV-FlowNet [6]</td><td>0.93</td><td>1.79</td><td>1.78</td><td>0.73</td><td>1.31</td></tr><tr><td>SICAF-LIF</td><td>1.03</td><td>1.80</td><td>1.54</td><td>0.69</td><td>1.27</td></tr><tr><td>SICAF-XLIF</td><td>1.06</td><td>1.82</td><td>1.58</td><td>0.70</td><td>1.29</td></tr><tr><td rowspan="4">White-box attack</td><td>LIF-EV-FlowNet [6]</td><td>1.02</td><td>1.77</td><td>1.52</td><td>0.68</td><td>1.26</td></tr><tr><td>XLIF-EV-FlowNet [6]</td><td>0.91</td><td>1.82</td><td>1.49</td><td>0.71</td><td>1.23</td></tr><tr><td>SICAF-LIF</td><td>0.86</td><td>1.53</td><td>1.36</td><td>0.59</td><td>1.09</td></tr><tr><td>SICAF-XLIF</td><td>0.90</td><td>1.63</td><td>1.39</td><td>0.58</td><td>1.13</td></tr></table>

## 5.4 Ablation on Loss Function Components

To investigate the roles of various components of loss functions, we compare the performance of $L _ { c o n t r a s t }$ and $\operatorname { S I C A F w . O . } L _ { s m o o t h }$ , with w.o.SICAF and SICAF on MVSEC. The results in Table 3 show that adding $L _ { s m o o t h } ~  t 0 ~ L _ { c o n t r a s t }$ improves the estimation accuracy, indicating that it helps denoise after contrast enhancement, reducing the impact of irrelevant information, and thus extracting features more accurately. When SICAF is used without smoothing, the model’s accuracy decreases, possibly because high-contrast noise is misinterpreted as valid motion information, leading to feature extraction errors. To mitigate this effect, adding $L _ { s m o o t h }$ is crucial for SICAF.While the above study confirms the necessity of all three loss terms, it does not isolate the contributions of the internal mechanisms of SICAF itself. To address this, we further conduct a component-level ablation in Section 5.5.

## 5.5 Ablation on Internal Components of SICAF

To investigate the roles of the key components of SICAF, we compare the performance of SICAF, SICAF under uniform compression, SICAF w.o. compact compression, and SICAF w.o. time-varying focus on MVSEC. The results in Table 4 show that SICAF achieves the lowest AEE across all four sequences through its temporally sensitive information-compression strategy, demonstrating the effectiveness of the proposed method.

Table 3: Ablation study on the various components of loss functions (lower is better, best in red, runner-up in blue).
<table><tr><td rowspan="2">Methods</td><td rowspan="2"> $L _ { c o n t r a s t }$ </td><td rowspan="2"> $L _ { s m o o t h }$ </td><td rowspan="2">SICAF</td><td colspan="5">AEE</td></tr><tr><td>Indoor (flying1)</td><td>Indoor (flying2)</td><td>Indoor (flying3)</td><td>outdoor (day1)</td><td>Avg.</td></tr><tr><td>(a)</td><td>√</td><td>X</td><td>X</td><td>0.77</td><td>1.50</td><td>1.25</td><td>0.56</td><td>1.02</td></tr><tr><td>(b)</td><td>√</td><td>√</td><td>X</td><td>0.81</td><td>1.45</td><td>1.19</td><td>0.57</td><td>1.01</td></tr><tr><td>(c)</td><td>√</td><td>X</td><td>√</td><td>0.83</td><td>1.60</td><td>1.31</td><td>0.54</td><td>1.07</td></tr><tr><td>(d)</td><td>√</td><td>√</td><td>√</td><td>0.71</td><td>1.37</td><td>1.09</td><td>0.43</td><td>0.90</td></tr></table>

Table 4: Ablation study on the key components of SICAF on the MVSEC dataset (lower is better, best in red, runner-up in blue).
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Compact Compression</td><td rowspan="2">Time-Varying Focus</td><td rowspan="2">Uniform Compression</td><td colspan="5">AEE</td></tr><tr><td>Indoor (flying1)</td><td>Indoor (flying2)</td><td>Indoor (flying3)</td><td>Outdoor (day1)</td><td>Avg.</td></tr><tr><td>(a)</td><td>√</td><td>√</td><td>x</td><td>0.72±0.031</td><td>1.38±0.026</td><td>1.09±0.03</td><td>0.47±0.053</td><td>0.91±0.012</td></tr><tr><td>(b)</td><td>√</td><td>x</td><td>√</td><td>0.81±0.047</td><td>1.53±0.044</td><td>1.28±0.064</td><td>0.55±0.041</td><td>1.04±0.019</td></tr><tr><td>(c)</td><td>x</td><td>√</td><td>x</td><td>0.81±0.075</td><td>1.51±0.029</td><td>1.24±0.010</td><td>0.56±0.061</td><td>1.03±0.043</td></tr><tr><td>(d)</td><td>√</td><td>x</td><td>x</td><td>0.81±0.021</td><td>1.52±0.059</td><td>1.28±0.025</td><td>0.61±0.055</td><td>1.06±0.036</td></tr></table>

When the temporally varying compression mechanism is replaced with uniform compression, the model’s accuracy decreases, possibly because a uniform information-compression strategy is unable to extract compact and critical features effectively from event streams with temporal heterogeneity, where excessive compression of temporal segments containing important motion information leads to an increase in prediction error. To mitigate this effect, the temporally varying compression is crucial for SICAF.

Removing either of the two key components of SICAF also degrades the prediction accuracy, indicating that the compact compression mechanism suppresses redundant information, while the temporally sensitive entropy constraint $H T _ { \alpha } ( Z )$ preserves critical temporal information. These two components work collaboratively in event-based optical flow estimation and are both indispensable.

## 6 Conclusion

In this paper, we propose SICAF to address the challenges faced by SNNs in self-supervised optical flow estimation. SICAF introduces a novel “Compactly Compress, Time-varying Focus” training strategy and designs an estimator equipped with learnable temporal attention. Through a temporally modulated constrained optimization process, SICAF enables SNNs to extract compact latent representations while preserving the temporal heterogeneity of input events, thereby enhancing their ability to capture irregular spatiotemporal structures in event streams. Overall, SICAF provides a new pathway toward more efficient feature extraction and enhanced robustness in SNNs.

## Acknowledgment

This work was supported by the National Natural Science Foundation of China under Grant Nos.   
62622610, 62376185 and U25A20540.

## References

[1] J. Cuadrado, U. Rançon, B. R. Cottereau, and et al., “Optical flow estimation from event-based cameras and spiking neural networks,” Frontiers in Neuroscience, vol. 17, p. 1160034, 2023.

[2] Z. Ding, R. Zhao, J. Zhang, and et al., “Spatio-temporal recurrent networks for event-based optical flow estimation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 1, 2022, pp. 525–533.

[3] F. Paredes-Vallés, K. Y. W. Scheper, C. De Wagter, and et al., “Taming contrast maximization for learning sequential, low-latency, event-based optical flow,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9695–9705.

[4] G. Gallego, H. Rebecq, and D. Scaramuzza, “A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 3867– 3876.

[5] Y. Schnider, S. Wo´zniak, M. Gehrig, and et al., “Neuromorphic optical flow and real-time implementation with event cameras,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4129–4138.

[6] J. Hagenaars, F. Paredes-Vallés, and G. De Croon, “Self-supervised learning of event-based optical flow with spiking neural networks,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 7167–7179.

[7] A. K. Kosta and K. Roy, “Adaptive-spikenet: Event-based optical flow estimation using spiking neural networks with learnable neuronal dynamics,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 6021–6027.

[8] C. Lee, A. K. Kosta, A. Z. Zhu, and et al., “Spike-flownet: Event-based optical flow estimation with energy-efficient hybrid neural networks,” in European Conference on Computer Vision, 2020, pp. 366–382.

[9] H. Sun, J. Wang, W. Cai, and et al., “St-flownet: An efficient spiking neural network for event-based optical flow estimation,” arXiv, 2025, dOI: 10.48550/arXiv.2503.10195. [Online]. Available: https://doi.org/10.48550/arXiv.2503.10195

[10] M. Dampfhoffer, T. Mesquida, A. Valentian, and et al., “Backpropagation-based learning techniques for deep spiking neural networks: A survey,” IEEE Transactions on Neural Networks and Learning Systems, 2023.

[11] G. Gallego, T. Delbrück, G. Orchard, and et al., “Event-based vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 154–180, 2020.

[12] S. Barchid, J. Mennesson, J. Eshraghian, and et al., “Spiking neural networks for frame-based and event-based single object localization,” Neurocomputing, vol. 559, p. 126805, 2023.

[13] G. Qiao, N. Ning, Y. Zuo, and et al., “Spatio-temporal fusion spiking neural network for frame-based and event-based camera sensor fusion,” IEEE Transactions on Emerging Topics in Computational Intelligence, 2024.

[14] F. Paredes-Vallés, K. Y. W. Scheper, and G. C. H. E. D. Croon, “Unsupervised learning of a hierarchical spiking neural network for optical flow estimation: From events to global motion perception,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 8, pp. 2051–2064, 2019.

[15] M. P. E. Apolinario and K. Roy, “S-tllr: Stdp-inspired temporal local learning rule for spiking neural networks,” arXiv preprint arXiv:2306.15220, 2023.

[16] S. Yang, B. Linares-Barranco, Y. Wu et al., “Self-supervised high-order information bottleneck learning of spiking neural network for robust event-based optical flow estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[17] C. Lee, A. K. Kosta, and K. Roy, “Fusion-flownet: Energy-efficient optical flow estimation using sensor fusion and deep fused spiking-analog network architectures,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 6504–6510.

[18] M. P. E. Apolinario and K. Roy, “S-tllr: Stdp-inspired temporal local learning rule for spiking neural networks,” arXiv preprint arXiv:2306.15220, 2023.

[19] Y. Wu, L. Deng, G. Li, and et al., “Direct training for spiking neural networks: Faster, larger, better,” in Proceedings of the AAAI conference on artificial intelligence, vol. 33, no. 01, 2019, pp. 1311–1318.

[20] H. Zheng, Y. Wu, L. Deng, and et al., “Going deeper with directly-trained larger spiking neural networks,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 12, 2021, pp. 11 062–11 070.

[21] S. Yang and B. Chen, “Effective surrogate gradient learning with high-order information bottleneck for spike-based machine intelligence,” IEEE Transactions on Neural Networks and Learning Systems, 2023.

[22] K. Wu, E. Shunzhuo, N. Yang, and et al., “A novel approach to enhancing biomedical signal recognition via hybrid high-order information bottleneck driven spiking neural networks,” Neural Networks, vol. 183, p. 106976, 2025.

[23] S. Yang, B. Linares-Barranco, Y. Wu, and B. Chen, “Self-supervised high-order information bottleneck learning of spiking neural network for robust event-based optical flow estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 2280–2297, 2025, dOI: 10.1109/TPAMI.2024.3510627. [Online]. Available: https://doi.org/10.1109/TPAMI.2024.3510627

[24] G. Gallego, M. Gehrig, and D. Scaramuzza, “Focus is all you need: Loss functions for eventbased vision,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 12 280–12 289.

[25] T. Stoffregen, C. Scheerlinck, D. Scaramuzza et al., “Reducing the sim-to-real gap for event cameras,” in Computer Vision–ECCV 2020. Springer, 2020, pp. 534–549.

[26] E. Mueggler, H. Rebecq, G. Gallego et al., “The event-camera dataset and simulator: Eventbased data for pose estimation, visual odometry, and slam,” The International Journal of Robotics Research, vol. 36, no. 2, pp. 142–149, 2017.

[27] A. Z. Zhu, D. Thakur, T. Özaslan et al., “The multivehicle stereo event camera dataset: An event camera dataset for 3d perception,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2032–2039, 2018.

[28] J. Hagenaars, F. Paredes-Vallés, and G. De Croon, “Self-supervised learning of event-based optical flow with spiking neural networks,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 7167–7179.

## A Technical appendices and supplementary material

## A.1 Broader Impacts

This paper focuses on improving the compression process of traditional IB to achieve both compactness and diversity of features, thereby enhancing the generalization and robustness of SNN models in optical flow estimation tasks. In general, this work has no negative societal impacts.

## A.2 Limitation

The proposed SICAF framework assists SNNs in achieving both compactness and diversity in feature extraction for optical flow estimation tasks. It addresses the issue of information loss that arises from traditional IB methods. However, in practical applications, SICAF may require parameter tuning to achieve better accuracy, which could be time-consuming.

## A.3 Experiment Settings

All experiments were conducted using the PyTorch framework on RTX 4090 and A6000 GPUs. Adam optimizer is used with learning rate of 0.0002 and batch size of 8 for 50 epochs. Gradients are clipped based on a global norm of 100, and the number of events for each input partition is set to 1000. 10,000 events are used for each training event partition. The scaling weight for $L _ { s m o o t h }$ is set as 0.001. Nearly 15 minutes of event data are contained in the training sequences, and they are split into 140 sequences with 128×128 size with 500k events in each sequence. Random horizontal, vertical, and polarity flips are used to augment data. The SNN employs a direct training approach using surrogate gradients. MVSEC dataset [27] is first used for evaluation with the ground-truth data. Optical flow predictions are produced at each frame timestep with dt=2, i.e., 25% of the window events at a time. To compare with ground-truth data, we convert predicted optical flow from units of pixels/partitions to pixel displacement by a multiplication factor $d t _ { g t } / d t _ { i n p u t }$ . ECD and HQF datasets are also used for evaluation of SICAF. Since there is no ground-truth data for these two datasets, the quality of the estimated optical flow is evaluated according to metrics based on the contrast maximization framework [4, 24]. The average test accuracy over two runs under different random seeds and is reported as final test accuracy.

For robustness evaluation, we consider both black-box and white-box adversarial attacks. For the black-box attack, following “Adversarially Robust Spiking Neural Networks Through Conversion”, we adopt a random-direction zeroth-order gradient estimation strategy. Candidate adversarial inputs are constructed using independent Gaussian random directions $\delta \stackrel { \sim } { \sim } \mathcal { N } ( 0 , I )$ as $\boldsymbol { x } ^ { \prime } = \boldsymbol { x } + \epsilon \boldsymbol { \delta }$ . The number of queries is set to 5, the perturbation magnitude is set to $\epsilon = 0 . 1$ , and the perturbed input is finally clipped to the range [0, 1]. For the white-box attack, following $\ddot { \mathbf { \Omega } } ^ { 6 6 } \mathbf { A }$ Comprehensive Analysis on Adversarial Robustness of Spiking Neural Networks”, we employ the FGSM. The perturbation is generated according to the sign of the gradient of the loss with respect to the input, with the perturbation magnitude set to $\epsilon = 0 . 1$

## A.4 Additional Details on SNN Neurons

In LIF neurons, when the membrane potential V reaches the threshold $m ,$ the neuron generates an output spike. The membrane potential threshold can be either a fixed value or can vary with the input. To better handle the complex motion changes in optical flow estimation, we set the membrane potential threshold m and the leakage parameter λ of the LIF neurons as learnable parameters to achieve improved performance. In addition, we adjust the neuron model to reduce adaptation delay and adapt more quickly to the rapidly changing motion characteristics in optical flow estimation. The output current I and membrane potential threshold m of the adjusted neuron are as follows:

$$
I _ { i } ^ { k } = \sum _ { i } W _ { i j } ^ { f } \delta _ { j } ^ { k } + \sum _ { l } W _ { i l } ^ { r } u _ { j } \delta _ { l } ^ { k - 1 } - \kappa P _ { i } ^ { k } ,\tag{16}
$$

$$
P _ { i } ^ { k } = \eta P _ { i } ^ { k - 1 } + \frac { 1 - \eta } { \vert { R } _ { i } \vert } \sum _ { j \in { R } _ { i } } \delta _ { j } ^ { k - 1 } ,\tag{17}
$$

$$
m _ { i } ^ { k } = \alpha _ { 0 } + \alpha _ { 1 } P _ { i } ^ { k } ,\tag{18}
$$

where $R _ { i }$ is the set of receptive fields of neuron i across all channels. For convenience, we list the hyperparameters of the neurons and provide their corresponding initialization value ranges, as shown in Table 5.

Table 5: SNN hyperparameter initialization table
<table><tr><td>Hyperparameter</td><td colspan="2">Initialization range</td></tr><tr><td> $\alpha _ { 0 }$  (learnable)</td><td rowspan="4"></td><td>0.01</td></tr><tr><td> $\alpha _ { 1 }$  (learnable)</td><td>1</td></tr><tr><td>ν (learnable)</td><td> $\begin{array} { r } { \nu = \frac { 1 } { 1 + \exp ( - a ) } , a \in \mathcal { N } ( - 4 , 0 . 1 ) } \end{array}$ </td></tr><tr><td> $\delta$ </td><td>δ ∈ {0, 1}</td></tr><tr><td rowspan="2">η (learnable)</td><td rowspan="2"></td><td> $\begin{array} { r } { \eta = \frac { 1 } { 1 + \exp ( - c ) } , c \in \mathcal { N } ( - 4 , 0 . 1 ) } \end{array}$ </td></tr><tr><td></td></tr><tr><td>m</td><td rowspan="4"></td><td>0.8</td></tr><tr><td>κ (learnable)</td><td>0.01</td></tr><tr><td> $P$ </td><td>0</td></tr></table>

## A.5 Pseudocode

The overall network in this paper processes event streams through four strided convolutional encoders, two residual blocks, and four decoder layers. SICAF trains the network parameters by maximizing the conditional entropy of intermediate variables, balancing the feature compactness and diversity of latent space.

Algorithm 1: The training process of the SICAF structure learning framework   
Input: Event data ${ \overline { { ( X ) } } } ;$ label $\overline { { ( Y ) } }$   
Output: Updated network weights (W); updated microscopic parameters $( \lambda ^ { \prime } , \gamma ^ { \prime } )$   
Parameters: SNN neuron parameters $( \lambda ^ { \prime } , \dot { \gamma } ^ { \prime } ) ; { \mathrm { S I C A F } }$ hyperparameters $( \lambda , \gamma ) ;$ batch size $( B ) ;$   
training epochs $( T E ) ;$ learning rate (r); network depth ${ \bar { ( H ) ; } }$ neuronal membrane potential $V$   
and spike output $O ;$   
Forward:;   
Initialize the model and optimizer;   
Preprocess the input data to obtain batch data;   
for $\bar { i } \gets 1$ to $T E$ do   
Fetch a mini-batch of size $B ;$   
for $h \gets 1$ to $H - 1$ do   
Calculate $V _ { h }$ and $O _ { h } ;$   
Calculate $V _ { H }$ and ${ \hat { Y } } ;$   
$/ /$ Calculate the loss   
$L _ { \mathrm { l o s s } } \gets \mathrm { S I C A F } \big ( L _ { \mathrm { f l o w } } ( Y , \hat { Y } ) , \lambda , I ( X ; Z ) , \gamma , H T _ { \alpha } ( Z ) \big )$   
Backward:;   
for $h  H$ to 1 do   
if $h \neq H$ then   
$\dot { / } / $ Update the network weights   
$\begin{array} { r } { \Delta W _ { h } = \sum _ { k } \frac { \partial L _ { \mathrm { l o s s } } } { \partial O _ { h } ^ { k } } \frac { \partial O _ { h } ^ { k } } { \partial U _ { h } ^ { k } } \frac { \partial U _ { h } ^ { k } } { \partial W _ { h } ^ { k } } } \end{array}$ ;   
$W _ { h } \gets W _ { h } - r \dot { \Delta } W _ { h } ;$   
$/ /$ Update the microscopic parameters   
$\begin{array} { r } { \Delta \lambda _ { h } ^ { \prime } = \sum _ { k } \frac { \partial L _ { \mathrm { l o s s } } } { \partial O _ { h } ^ { k } } \frac { \partial O _ { h } ^ { k } } { \partial U _ { h } ^ { k } } \frac { \partial U _ { h } ^ { k } } { \partial \lambda _ { h } ^ { \prime } } } \end{array}$ ;   
$\begin{array} { r } { \Delta \beta _ { h } ^ { \prime } = \sum _ { k } \frac { \partial L _ { \mathrm { l o s s } } } { \partial O _ { h } ^ { k } } \frac { \partial O _ { h } ^ { k } } { \partial U _ { h } ^ { k } } \frac { \partial U _ { h } ^ { k } } { \partial \beta _ { h } ^ { \prime } } } \end{array}$   
$\lambda _ { h } ^ { \prime }  \lambda _ { h } ^ { \prime } - r \Delta \bar { \lambda } _ { h } ^ { \prime } ;$   
$\beta _ { h } ^ { \prime } \gets \beta _ { h } ^ { \prime } - r \Delta \beta _ { h } ^ { \prime } ;$   
else   
$\begin{array} { r } { \Delta W _ { h } = \sum _ { k } \frac { \partial L _ { \mathrm { l o s s } } } { \partial O _ { h } ^ { k } } \frac { \partial O _ { h } ^ { k } } { \partial U _ { h } ^ { k } } \frac { \partial U _ { h } ^ { k } } { \partial W _ { h } ^ { k } } ; } \end{array}$   
$W _ { h } \gets W _ { h } - r \dot { \Delta } W _ { h } ;$

Meanwhile, the microscopic parameters in adaptive neurons adjust throughout the training process, further optimizing the network’s weight parameters. The general training process is outlined in Algorithm 1.

## A.6 Implementation Details of the SICAF Objective

The encoder output is denoted by $\mu _ { \phi } ( x )$ , and the variational encoding distribution is defined as

$$
E _ { \phi } ( z \mid x ) = \mathcal { N } \left( z ; \mu _ { \phi } ( x ) , \sigma ^ { 2 } I \right) , \qquad \sigma ^ { 2 } = \exp ( \mathrm { l o g v a r } _ { t } ) + 1 0 ^ { - 1 0 } ,\tag{19}
$$

where logvar is learnable and initialized to −1. During training, we directly use $\mu _ { \phi } ( x )$ as the bottleneck representation to reduce estimation variance. For a mini-batch of size $B ,$ , we approximate the marginal distribution using kernel density estimation (KDE):

$$
r ( z ) = \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \mathcal { N } \left( z ; \mu _ { j } , \sigma ^ { 2 } I \right) ,\tag{20}
$$

where $\mu _ { j } = \mu _ { \phi } ( x _ { j } )$ . With $D _ { i j } = \| \mu _ { i } - \mu _ { j } \| _ { 2 } ^ { 2 }$ , we use the following mean-based surrogate for the information-bottleneck term:

$$
\widehat { I } ( X ; Z ) = \log B - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \left[ \sum _ { j = 1 } ^ { B } \exp \left( - \frac { D _ { i j } } { 2 \sigma ^ { 2 } } \right) \right] .\tag{21}
$$

This expression is obtained by evaluating log $E _ { \phi } ( z \mid x _ { i } ) - \log r ( z ) { \mathrm { ~ a t ~ } } z = \mu _ { i }$ , rather than computing the full expectation over $E _ { \phi } ( \boldsymbol { z } \mid \boldsymbol { x } _ { i } )$ . The connection to the information-bottleneck objective is given by the identity:

$$
\begin{array} { r } { \mathbb { E } _ { x } \left[ \mathrm { K L } \left( E _ { \phi } ( \boldsymbol { z } \mid \boldsymbol { x } ) \parallel \boldsymbol { r } ( \boldsymbol { z } ) \right) \right] = I ( \boldsymbol { X } ; \boldsymbol { Z } ) + \mathrm { K L } \left( p ( \boldsymbol { z } ) \parallel r ( \boldsymbol { z } ) \right) . } \end{array}\tag{22}
$$

Using KDE to approximate $p ( z )$ , we neglect the marginal-distribution mismatch term and use the mean-based surrogate above in training. This introduces an approximation rather than an exact KL evaluation. For entropy regularization, we use the RBF kernel:

$$
K _ { i j } = \mathrm { e x p } \left( - \frac { \| Z _ { i } - Z _ { j } \| _ { 2 } ^ { 2 } } { 2 \sigma _ { \mathrm { k e r } } ^ { 2 } } \right) ,\tag{23}
$$

with a fixed bandwidth $\sigma _ { \mathrm { k e r } } = 1 . 0$ , which is distinct from the learnable encoding standard deviation σ. We set $\beta = 4 . 8 \times 1 0 ^ { - 3 }$ and $\gamma = 4 \times 1 0 ^ { - 4 }$ . Moreover, $I _ { c } = H _ { c } = 0$ are fixed constants and therefore do not affect model gradients in the penalized objective with fixed $\beta$ and $\gamma$

## A.7 Theoretical Guarantees of SICAF

In this section, we provide a generalization bound and a corresponding sample-complexity result for SICAF. Let

$$
S = \{ X _ { i } \} _ { i = 1 } ^ { n } \sim \mathcal { D } ^ { n }
$$

be an i.i.d. training sample drawn from the data distribution D. The encoder $f _ { \theta }$ maps the input $X$ to a latent representation

$$
Z = f _ { \theta } ( X ) ,
$$

which is subsequently processed by a predictor h. We denote the population and empirical risks by

$$
\mathcal { T } _ { \mathrm { S I C A F } } ( h ) = \mathbb { E } _ { X \sim \mathcal { D } } \left[ \ell \big ( h ( Z ) \big ) \right] ,\tag{24}
$$

and

$$
\widehat { \mathcal { T } } _ { \mathrm { S I C A F } , S } ( h ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell \big ( h ( Z _ { i } ) \big ) , \qquad Z _ { i } = f _ { \theta } ( X _ { i } ) ,\tag{25}
$$

respectively.

For each input $X ,$ , let

$$
\mathbf { a } ( X ) = \left( a _ { 1 } ( X ) , \ldots , a _ { T } ( X ) \right) ^ { \top } \in \mathbb { R } ^ { T }
$$

denote the temporal-attention vector produced by the temporal-attention estimator. We assume that

$$
a _ { t } ( X ) \geq 0 , \qquad \sum _ { t = 1 } ^ { T } a _ { t } ( X ) = 1 .\tag{26}
$$

The corresponding time-sensitive Rényi entropy is

$$
H _ { T } ^ { \alpha } ( Z ) = \sum _ { t = 1 } ^ { T } a _ { t } ( X ) H _ { \alpha } ( Z _ { t } ) .\tag{27}
$$

We define the mean temporal-attention vector as

$$
\bar { \mathbf { a } } = \mathbb { E } _ { X \sim { \mathcal { D } } } \left[ \mathbf { a } ( X ) \right] ,\tag{28}
$$

and define the effective temporal-attention fluctuation by

$$
\Phi _ { \mathrm { e f f } } = \mathbb { E } _ { X \sim \mathcal { D } } \left[ \left\| \mathbf { a } ( X ) - \bar { \mathbf { a } } \right\| _ { 2 } ^ { 2 } \right] .\tag{29}
$$

Moreover, let

$$
\Delta _ { H } = \left| H _ { T } ^ { \alpha } ( Z ) - \widehat { H } _ { T } ^ { \alpha } ( Z ) \right|\tag{30}
$$

denote the discrepancy between the population time-sensitive Rényi entropy and its finite-sample estimate.

Theorem 1 (Generalization bound of SICAF). Let $\mathcal { F }$ be the hypothesis class induced by the encoder f<sub>θ</sub> and predictor h. Assume that:

1. the latent representation is norm-bounded, $i . e .$

$$
\| Z \| _ { 2 } \leq B \quad a l m o s t s u r e l y ;\tag{31}
$$

2. the loss function ℓ is L-Lipschitz with respect to the latent representation;

3. the temporal-attention vector satisfies (26);

4. the time-sensitive Rényi entropy is bounded as

$$
0 \leq H _ { T } ^ { \alpha } ( Z ) \leq H _ { \operatorname* { m a x } } ;\tag{32}
$$

5. the loss fluctuation associated with temporal modulation is sub-Gaussian with variance proxy $\sigma ^ { \mathrm { 2 } } .$

Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$ over the draw of S, the following bound holds:

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { S I C A F } } ( h ) - \widehat { \mathcal { I } } _ { \mathrm { S I C A F } , S } ( h ) \leq \sqrt { \frac { 2 L ^ { 2 } B ^ { 2 } } { n } \left( I ( X ; Z ) + \ln \frac { 1 } { \delta } \right) } } \\ & { \qquad + \sqrt { 2 \sigma ^ { 2 } \Phi _ { \mathrm { e f f } } } + \gamma \Delta _ { H } . } \end{array}\tag{33}
$$

Here, $I ( X ; Z )$ denotes the mutual information between the input and the latent representation, $\Phi _ { \mathrm { e f f } }$ quantifies the variability of temporal attention across input samples, and $\Delta _ { H }$ denotes the finite-sample estimation error ofthe time-sensitive Rényi entropy.

Proof. We decompose the generalization gap into three components:

$$
\mathcal { T } _ { \mathrm { S I C A F } } ( h ) - \widehat { \mathcal { I } } _ { \mathrm { S I C A F } , S } ( h ) = \Delta _ { \mathrm { I B } } + \Delta _ { \mathrm { t e m p } } + \Delta _ { \mathrm { e n t } } ,\tag{34}
$$

where $\Delta _ { \mathrm { I B } }$ is the representation-complexity term, $\Delta _ { \mathrm { t e m p } }$ is the additional deviation induced by temporal attention, and $\Delta _ { \mathrm { e n t } }$ is the error caused by estimating the entropy regularizer from finitely many samples.

First, since ℓ is L-Lipschitz and $\| Z \| _ { 2 } \le B$ , the information-theoretic generalization inequality gives, with probability at least $1 - \delta ,$

$$
\Delta _ { \mathrm { I B } } \leq \sqrt { \frac { 2 L ^ { 2 } B ^ { 2 } } { n } \left( I ( X ; Z ) + \ln \frac { 1 } { \delta } \right) } .\tag{35}
$$

This term is controlled by the information-bottleneck constraint used in SICAF. In particular, enforcing a smaller information budget for $I ( X ; Z )$ reduces the complexity contribution associated with the latent representation.

Second, SICAF employs an input-dependent temporal-attention vector $\mathbf { a } ( X )$ . Its deviation from the population mean a¯ introduces an additional fluctuation in the temporally weighted objective. By the assumed sub-Gaussianity of this fluctuation and the definition in (29), a concentration bound yields

$$
\Delta _ { \mathrm { t e m p } } \leq \sqrt { 2 \sigma ^ { 2 } \mathbb { E } _ { X \sim \mathcal { D } } \left[ \left\| \mathbf { a } ( X ) - \bar { \mathbf { a } } \right\| _ { 2 } ^ { 2 } \right] } = \sqrt { 2 \sigma ^ { 2 } \Phi _ { \mathrm { e f f } } } .\tag{36}
$$

Therefore, a more stable temporal-attention distribution leads to a smaller value of $\Phi _ { \mathrm { e f f } }$ and hence a tighter bound.

Third, the SICAF objective contains the entropy regularization term $- \gamma H _ { T } ^ { \alpha } ( Z )$ . In practice, this population quantity is replaced by the finite-sample estimate $\widehat { H } _ { T } ^ { \alpha } ( Z )$ . Therefore, the absolute deviation of the entropy contribution is bounded by

$$
\begin{array} { c } { { \Delta _ { \mathrm { e n t } } = \gamma \left| H _ { T } ^ { \alpha } ( Z ) - \widehat H _ { T } ^ { \alpha } ( Z ) \right| } } \\ { { = \gamma \Delta _ { H } . } } \end{array}\tag{37}
$$

Under (32), this term is finite; in particular, $\Delta _ { H } \leq H _ { \operatorname* { m a x } }$ whenever both entropy quantities lie in $[ 0 , H _ { \mathrm { m a x } } ]$

Combining (35), (36), and (37) in (34) gives (33).

Theorem 2 (Sample complexity of SICAF). Suppose that the assumptions of Theorem 1 hold and that

$$
I ( X ; Z ) \leq I _ { 0 } .\tag{38}
$$

Assumefurther that the total temporal-modulation contribution is bounded by

$$
\sqrt { 2 \sigma ^ { 2 } \Phi _ { \mathrm { e f f } } } + \gamma \Delta _ { H } \leq \varepsilon _ { \mathrm { t e m p } } .\tag{39}
$$

For any target tolerance ε satisfying

$$
\varepsilon > \varepsilon _ { \mathrm { t e m p } } ,\tag{40}
$$

it is sufficient to choose

$$
n \geq { \frac { 2 L ^ { 2 } B ^ { 2 } } { \left( \varepsilon - \varepsilon _ { \mathrm { t e m p } } \right) ^ { 2 } } } \left( I _ { 0 } + \ln { \frac { 1 } { \delta } } \right)\tag{41}
$$

to guarantee, with probability at least $1 - \delta ,$ , that

$$
\mathcal { T } _ { \mathrm { S I C A F } } ( h ) - \widehat { \mathcal { T } } _ { \mathrm { S I C A F } , S } ( h ) \leq \varepsilon .\tag{42}
$$

Proof. By Theorem 1,

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { S I C A F } } ( h ) - \widehat { \mathcal { I } } _ { \mathrm { S I C A F } , S } ( h ) \leq \sqrt { \frac { 2 L ^ { 2 } B ^ { 2 } } { n } \left( I ( X ; Z ) + \ln \frac { 1 } { \delta } \right) } } \\ & { \quad \quad \quad \quad + \sqrt { 2 \sigma ^ { 2 } \Phi _ { \mathrm { e f f } } } + \gamma \Delta _ { H } . } \end{array}\tag{43}
$$

Using (38) and (39), we obtain

$$
\mathcal { T } _ { \mathrm { S I C A F } } ( h ) - \widehat { \mathcal { I } } _ { \mathrm { S I C A F } , S } ( h ) \leq \sqrt { \frac { 2 L ^ { 2 } B ^ { 2 } } { n } \left( I _ { 0 } + \ln \frac { 1 } { \delta } \right) } + \varepsilon _ { \mathrm { t e m p } } .\tag{44}
$$

To ensure that the right-hand side of (44) is at most $\varepsilon ,$ it suffices to require

$$
\sqrt { \frac { 2 L ^ { 2 } B ^ { 2 } } { n } \left( I _ { 0 } + \ln \frac { 1 } { \delta } \right) } \leq \varepsilon - \varepsilon _ { \mathrm { t e m p } } .\tag{45}
$$

The right-hand side is positive by (40). Squaring both sides gives

$$
{ \frac { 2 L ^ { 2 } B ^ { 2 } } { n } } \left( I _ { 0 } + \ln { \frac { 1 } { \delta } } \right) \leq \left( \varepsilon - \varepsilon _ { \mathrm { t e m p } } \right) ^ { 2 } .\tag{46}
$$

Rearranging (46) with respect to n yields (41). Therefore, (42) holds with probability at least $1 - \delta$ □

Remark. Theorem 2 does not discard the temporal terms. Instead, it explicitly requires the temporal attention fluctuation and entropy-estimation error to satisfy the error-budget condition in (39). The original mutual-information-only sample-complexity expression is recovered as a special case when $\varepsilon _ { \mathrm { t e m p } } = 0$ , or approximately when the temporal contribution is negligible relative to the target generalization tolerance.

## A.8 Additional Self-Supervised Optical Flow Estimation Results

In addition to validating the performance of SICAF on the MVSEC dataset, we also evaluated SICAF on the ECD and HQF datasets. For datasets lacking ground-truth labels, we adopted the RSAT metric to measure the quality of optical flow estimation by the model on these datasets. Its mathematical definition is as follows:

$$
\mathrm { R S A T } : = \frac { \mathcal { L } _ { \mathrm { c o n t r a s t } } ( t _ { \mathrm { r e f } } ^ { \mathrm { f w } } | \mathbf { u } ) } { \mathcal { L } _ { \mathrm { c o n t r a s t } } ( t _ { \mathrm { r e f } } ^ { \mathrm { f w } } | \mathbf { 0 } ) } .\tag{47}
$$

The contrast loss $\mathcal { L } _ { \mathrm { c o n t r a s t } } ( t _ { \mathrm { r e f } } )$ is defined as:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t } } ( t _ { \mathrm { r e f } } ) = \frac { \sum _ { x } \bar { t } _ { p ^ { \prime } = + } ( \mathbf { x } ; \mathbf { u } | t _ { \mathrm { r e f } } ) ^ { 2 } + \bar { t } _ { p ^ { \prime } = - } ( \mathbf { x } ; \mathbf { u } | t _ { \mathrm { r e f } } ) ^ { 2 } } { \sum _ { x } \mathbb { I } [ n ( \mathbf { x } ^ { \prime } ) > 0 ] + \epsilon } ,\tag{48}
$$

where ${ \bar { t } } _ { p ^ { \prime } = + }$ and $\bar { t } _ { p ^ { \prime } = - }$ represent the positive and negative timestamps respectively, $n ( \mathbf { x } ^ { \prime } )$ indicates the number of events at position $\mathbf { x } ^ { \prime } .$ , and ϵ is a small constant added for numerical stability. When $\mathrm { R S A T } < 1$ , it indicates that the predicted optical flow outperforms the baseline scenario composed of null vectors.

Table 6 and Figure 6 present the results for the ECD and HQF datasets, respectively. From the experimental results, it can be observed that SICAF also effectively enhances the model’s generalization ability in self-supervised optical flow estimation on the ECD and HQF datasets. These results are attributed to the maximum entropy information bottleneck increasing the randomness of the intermediate variable $T ,$ allowing the model to extract useful information while giving deterministic prediction results a greater variance, thus improving the model’s generalization.

Table 6: Experimental results on ECD and HQF datasets. (Higher FWL and lower RSAT indicate better performance. Best in red, runner-up in blue.)
<table><tr><td rowspan="2">Methods</td><td colspan="2">ECD</td><td colspan="2">HQF</td></tr><tr><td>FWL↑</td><td>RSAT↓</td><td>FWL↑</td><td>RSAT↓</td></tr><tr><td>LIF-EV-FlowNet [6]</td><td>1.21</td><td>0.95</td><td>1.24</td><td>0.94</td></tr><tr><td>ALIF-EV-FlowNet [6]</td><td>1.17</td><td>0.98</td><td>1.21</td><td>0.98</td></tr><tr><td>PLIF-EV-FlowNet [6]</td><td>1.24</td><td>0.95</td><td>1.28</td><td>0.93</td></tr><tr><td>XLIF-EV-FlowNet [6]</td><td>1.23</td><td>0.95</td><td>1.25</td><td>0.93</td></tr><tr><td>SICAF</td><td>1.26</td><td>0.94</td><td>1.30</td><td>0.92</td></tr></table>

![](images/516a125e882c718d85d6120fe9bb348795f268c43f0fdb6b3b3b2825e6f7b43b.jpg)  
Figure 6: Qualitative evaluation of optical flow estimation results on sequences from the ECD and HQF datasets.

## A.9 Additional Details on Robustness Experiments

To assess the robustness of the model, we introduced random noise attacks and adversarial noise attacks during the model testing process. Random noise attacks involve using clean datasets without additional noise during model training and adding noise to the datasets during testing to simulate disturbances in real-world scenarios. In this study, random noise attacks mainly consisted of adding two common types of noise, Gaussian noise and salt-and-pepper (SP) noise. Gaussian noise with a mean of $\mu = 0$ and variance of σ can be expressed mathematically as follows:

$$
f _ { G a u s s i a n } ( x ) = \frac { 1 } { \sqrt { 2 \pi } \sigma } e ^ { - \frac { ( x - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } ,\tag{49}
$$

where x is a random variable, $\mu$ and σ represent its mean and variance, respectively. SP noise is composed of white and black pixels generated by severe disturbances. In addition, we explore adversarial noise attacks in two forms, i.e. white-box and black-box attacks. In white-box attack, the attacker possesses detailed information about the model, including data preprocessing, structure, and parameters, using this internal model information to generate adversarial samples. In our experiments, white-box attacks employed the fast gradient sign method (FGSM), which primarily utilizes the gradient of the loss function with respect to the input data x to generate adversarial noise. The expression for the generated noise is as follows:

$$
\eta _ { w h i t e - b o x } = \varepsilon \cdot s i g n ( \nabla _ { x } J ( \theta , x , y ) ) ,\tag{50}
$$

where the parameter ε represents the intensity of the added noise, and $J ( \theta , x , y )$ is the loss function. $\nabla _ { x } J$ denotes the gradient of the loss function with respect to the input x. Using the FGSM algorithm, we obtain white-box adversarial noise and add it to the input dataset during the testing phase. Black-box attacks refer to scenarios where the attacker is unaware of the internal details and can only access the input and output information. Here, we use zero-order optimization methods to estimate gradients using experimental function values without directly computing or using gradient information. Zero-order optimization methods are a way to generate adversarial examples without explicit gradient information, suitable for black-box adversarial attacks. These methods optimize the input by querying the target function’s output to make the model produce incorrect predictions, thus generating black-box adversarial samples. Suppose our model is $f ,$ and the input sample is x. The goal of generating adversarial attack samples is to maximize the classification error or loss of f(x).

First, we initialize the input sample $x _ { a d \nu } = i n p u t s$ and generate a random direction vector $\pmb { \delta } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ where $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is drawn from a multivariate normal distribution with a mean of 0 and a covariance matrix of the identity matrix. The purpose of generating the vector $\pmb { \delta }$ is to explore the solution space around the input sample through random search, thereby indirectly estimating the gradient direction and finding perturbation directions that can effectively generate adversarial samples.

We then compute the loss change for the sample when it is moved in the direction of $\delta .$ . Let L(inputs, target) denote the model’s loss function, where inputs is the input sample and target is the true label of the sample. The change in loss when the sample is moved in the direction of $\bar { \boldsymbol \delta }$ can be expressed as:

$$
\Delta L = L ( f ( x _ { a d \nu } + \alpha \delta ) , t a r g e t ) - L ( f ( x _ { a d \nu } ) , t a r g e t ) ,\tag{51}
$$

where α denotes the perturbation strength. After obtaining the loss change in the direction of $\delta ,$ we estimate the gradient and update the input sample as follows:

$$
x _ { a d \nu } = x _ { a d \nu } + \alpha \delta \cdot s i g n ( \Delta L ) ,\tag{52}
$$

Table 7: Experimental results of SICAF on ECD under different noise (lower is better, best in red).
<table><tr><td rowspan="2">Noise type Methods</td><td colspan="2">SP</td><td colspan="2">Gaussian</td><td colspan="2">Black-box attack</td><td colspan="2">White-box attack</td></tr><tr><td>[28]</td><td>SICAF</td><td>[28]</td><td>SICAF</td><td>[28]</td><td>SICAF</td><td>[28]</td><td>SICAF</td></tr><tr><td>boxes</td><td>0.97</td><td>0.94</td><td>0.97</td><td>0.94</td><td>1.01</td><td>1.00</td><td>0.99</td><td>0.96</td></tr><tr><td>Ocalibration</td><td>0.97</td><td>0.96</td><td>0.98</td><td>0.96</td><td>1.00</td><td>1.00</td><td>0.99</td><td>0.98</td></tr><tr><td>dynamic_6dof</td><td>0.94</td><td>0.92</td><td>0.95</td><td>0.92</td><td>1.00</td><td>0.98</td><td>0.97</td><td>0.94</td></tr><tr><td>office_zigzag</td><td>0.97</td><td>0.96</td><td>0.98</td><td>0.97</td><td>1.00</td><td>1.00</td><td>0.99</td><td>0.98</td></tr><tr><td>poster_6dof</td><td>0.97</td><td>0.94</td><td>0.97</td><td>0.94</td><td>1.01</td><td>0.99</td><td>0.99</td><td>0.90</td></tr><tr><td>shapes_6dof</td><td>1.00</td><td>0.99</td><td>1.01</td><td>0.99</td><td>1.00</td><td>0.99</td><td>1.00</td><td>0.98</td></tr><tr><td>slider_depth</td><td>0.91</td><td>0.93</td><td>0.91</td><td>0.93</td><td>0.99</td><td>0.95</td><td>0.92</td><td>0.91</td></tr><tr><td>Mean</td><td>0.96</td><td>0.95</td><td>0.97</td><td>0.95</td><td>1.01</td><td>0.99</td><td>0.98</td><td>0.96</td></tr></table>

To ensure that the generated adversarial samples are perceptually similar to the original samples, we need to keep the perturbation within a certain range to maintain the stealthiness and effectiveness of the attack. Therefore, we apply pruning operations to the generated samples:

$$
x _ { a d \nu } = c l i p ( x _ { a d \nu } , i n p u t s - \epsilon , i n p u t s + \epsilon ) ,\tag{53}
$$

where ϵ denotes the perturbation range set for the pruning operation. Table 7 and Table 8 show the robustness test results of SICAF on the ECD and HQF datasets.

Table 8: Experimental results of SICAF on HQF under different noise (lower is better, best in red).
<table><tr><td rowspan="2">Noise type Methods</td><td colspan="2">SP</td><td colspan="2">Gaussian</td><td colspan="2">Black-box attack</td><td colspan="2">White-box attack</td></tr><tr><td>[28]</td><td>SICAF</td><td>[28]</td><td>SICAF</td><td>[28]</td><td>SICAF</td><td>[28]</td><td>SICAF</td></tr><tr><td>bike_bar_hdr</td><td>0.97</td><td>0.95</td><td>0.98</td><td>0.95</td><td>1.00</td><td>0.98</td><td>0.98</td><td>0.95</td></tr><tr><td>boxes</td><td>0.94</td><td>0.92</td><td>0.95</td><td>0.93</td><td>1.00</td><td>0.97</td><td>0.97</td><td>0.93</td></tr><tr><td>desk_6k</td><td>0.92</td><td>0.90</td><td>0.93</td><td>0.91</td><td>0.99</td><td>0.95</td><td>0.95</td><td>0.92</td></tr><tr><td>desk_fast</td><td>0.92</td><td>0.89</td><td>0.93</td><td>0.90</td><td>0.99</td><td>0.95</td><td>0.95</td><td>0.91</td></tr><tr><td>desk_hand_only</td><td>1.00</td><td>0.98</td><td>1.00</td><td>0.98</td><td>0.99</td><td>0.99</td><td>0.98</td><td>0.97</td></tr><tr><td>desk_slow</td><td>0.89</td><td>0.88</td><td>0.91</td><td>0.91</td><td>0.98</td><td>0.95</td><td>0.90</td><td>0.89</td></tr><tr><td>engineering_posters</td><td>0.96</td><td>0.94</td><td>0.97</td><td>0.95</td><td>0.99</td><td>0.95</td><td>0.99</td><td>0.95</td></tr><tr><td>high_texture_plants</td><td>0.94</td><td>0.92</td><td>0.98</td><td>0.96</td><td>1.00</td><td>0.98</td><td>0.96</td><td>0.92</td></tr><tr><td>poster_pillar_1</td><td>0.98</td><td>0.96</td><td>0.98</td><td>0.96</td><td>0.99</td><td>0.97</td><td>0.98</td><td>0.95</td></tr><tr><td>poster_pillar_2</td><td>0.97</td><td>0.95</td><td>0.98</td><td>0.95</td><td>0.99</td><td>0.97</td><td>0.98</td><td>0.95</td></tr><tr><td>reflective_materials</td><td>0.97</td><td>0.94</td><td>0.97</td><td>0.95</td><td>0.99</td><td>0.95</td><td>0.98</td><td>0.95</td></tr><tr><td>slow_and_fast_desk</td><td>0.96</td><td>0.94</td><td>0.97</td><td>0.95</td><td>0.98</td><td>0.95</td><td>0.98</td><td>0.95</td></tr><tr><td>slow_hand</td><td>0.97</td><td>0.95</td><td>0.97</td><td>0.95</td><td>0.99</td><td>0.95</td><td>0.99</td><td>0.94</td></tr><tr><td>still_life</td><td>0.94</td><td>0.93</td><td>0.95</td><td>0.93</td><td>0.99</td><td>0.98</td><td>0.93</td><td>1.00</td></tr><tr><td>Mean</td><td>0.95</td><td>0.93</td><td>0.96</td><td>0.94</td><td>0.99</td><td>0.97</td><td>0.96</td><td>0.93</td></tr></table>

## A.10 Computational Efficiency Analysis

We evaluate the computational efficiency of SICAF against LF-EV-FlowNet in terms of inference latency, inference energy per batch, memory usage, and the number of spikes per inference batch. Table 9 summarizes the results.

Table 9: Computational efficiency comparison between LF-EV-FlowNet and SICAF. Lower values are preferable for all reported metrics.
<table><tr><td>Model</td><td>Latency (ms/batch)</td><td>Inference energy per batch (J)</td><td>Memory (MiB)</td><td>Spikes per inference batch</td></tr><tr><td>LF-EV-FlowNet</td><td> $2 9 . 7 6 4 \pm 7 . 8 8 1$ </td><td> $4 . 3 5 4 \pm 1 . 1 5 9$ </td><td> $2 4 1 . 5 4 8 \pm 6 . 8 8 5$ </td><td> $1 9 7 , 7 2 4 . 6 7 \pm 7 6 9 . 8 6$ </td></tr><tr><td>SICAF</td><td> $2 7 . 1 6 4 \pm 6 . 6 4 5$ </td><td> $3 . 8 7 3 \pm 0 . 4 3 6$ </td><td> $2 4 7 . 5 0 3 \pm 6 . 5 8 2$ </td><td> $8 3 , 9 7 8 . 0 0 \pm 2 7 4 . 9 9$ </td></tr></table>

As shown in Table 9, SICAF reduces the mean inference latency from 29.764 to 27.164 ms/batch and the mean inference energy from 4.354 to 3.873 J/batch, corresponding to reductions of approximately 8.7% and 11.0%, respectively. The mean number of spikes per inference batch decreases by approximately 57.5%, indicating sparser spiking activity. Meanwhile, memory usage increases from 241.548 to 247.503 MiB, an increase of approximately 2.5%. These results indicate lower average inference latency and energy consumption with a modest increase in memory usage.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: We have described the proposed method (SICAF) and the scope of experiments in the abstract and introduction.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have outlined the potential limitations in the Appendix A.2.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The theorem and its proof are provided in the Appendix.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have provided the corresponding experimental details and some parameter settings in the Appendix, along with the pseudocode of the method.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: Our experiments use publicly available benchmark datasets, and the code will be released to ensure that others can reproduce the results.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: We have provided a detailed description of our experimental setup in the Appendix.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: It is worth noting that in the experimental section, we compare our results with the best-reported results from other works.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We have specified the GPUs used in the experiments in the Appendix: GeForce RTX 4090 and A6000.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: This work follows the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: This paper focuses on improving the compression process of traditional IB to achieve both compactness and diversity of features, thereby enhancing the generalization and robustness of SNN models in optical flow estimation tasks. In general, this work has no negative societal impacts.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper proposes a framework for optimizing the SNN model in optical flow estimation tasks. Neither the model nor the standard benchmark data used poses such risk.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have properly cited the original papers corresponding to the code and datasets used in our work, and have ensured that their licenses and terms of use are respected.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not release new assets.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The research does not involve human subjects, therefore IRB approval is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The development of the core methods in this paper does not involve large language models.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.