# MULTI-DEPTH TEMPORAL FUSION FOR FEEDFORWARD, LOCALLY TRAINED SPIKING NEURAL NETWORKS

Aidin Attar Department of Information Engineering, University of Padua Padua, Italy

PREPRINT

Eleonora Cicciarella Department of Information Engineering, University of Padua Padua, Italy

Michele Rossi   
Department of Information   
Engineering, University of Padua   
Department of Mathematics   
“Tullio Levi-Civita”, University of Padua   
Padua, Italy

Corresponding author: aidin.attar@phd.unipd.it

## ABSTRACT

We propose a new spiking neural network (SNN) design to process static images and event streams using time-to-first-spike (TTFS) latencies. Our key research question is which architectural choices best accommodate local and online learning in multi-layer convolutional SNNs. This question is addressed via an original framework combining residual-like connections with multi-depthfeature aggregation and consensus. The full SNN pipeline features an early-vision front end, to convert raw visual data into sparse spike latencies, a four-layer convolutional backbone trained layerwise with unsupervised spike-timing dependent plasticity (STDP), a deterministic Multi-Depth Temporal Fusion (MDTF) and a final classifier trained with reward-modulated spike-timing-dependent plasticity (R-STDP). Rather than replacing early features in deeper layers, the proposed MDTF preserves early temporal evidence, adding sparse residual events from intermediate layers, and incorporating deeper features only when they agree in time with earlier representations. The resulting architecture is experimentally validated across MNIST, Fashion-MNIST, CIFAR-10, and N-MNIST, delivering strong classification performance under a fully local learning regime. Selective multi-depth fusion significantly outperforms traditional STDP/R-STDP baselines on highervariability visual tasks (achieving +18.2 pp on Fashion-MNIST and +29.2 pp on CIFAR-10). Furthermore, activity-budget analyses show that the network retains high accuracy even when removing a large fraction of late or weak spike events, confirming its high data efficiency and reduced event-processing requirements. The codebase is publicly available at https://github.com/aidinattar/multi-depth-temporal-fusion-snn.

Keywords Spiking neural networks · Reward-modulated STDP · Time-to-first-spike coding · Local learning · Neuromorphic vision

## 1 Introduction

Artificial intelligence systems have become increasingly effective, but often at the cost of larger (deeper) models, higher training complexity, and growing energy demands at both training and inference time. For this reason, there is sustained interest in neural models that can retain useful representational power while operating under tighter efficiency constraints. SNNs are natural candidates in this sense because they process information through discrete events and exploit time as an explicit computational dimension. Moreover, their sparse and event-driven nature makes them ideal for deployment on neuromorphic hardware [1–4].

Despite these advantages, training SNNs remains nontrivial. In contrast to conventional artificial neural networks (ANNs), spiking neurons combine internal state dynamics with discontinuous spike emission, making the learning problem inherently time-dependent and overall more challenging to solve. A large body of work has recently addressed this through surrogate-gradient methods, which make end-to-end training possible by replacing the non-differentiable spike function with a differentiable approximation during optimization [5, 6]. Another major line of work relies on direct ANN-to-SNN conversion, where a high-performing ANN model is first trained and then transferred into the spiking domain [7–10]. These approaches have produced good results. However, they leave open a complementary and yet relevant question: how much can be achieved in deep SNNs when learning remains local, online, and aligned with the constraints that originally motivated spike-based computation? With “local” learning rules, we mean learning dynamics where synaptic updates solely depend on pre- and post-synaptic neural activity, possibly combined with a modulatory signal, without entailing the propagation of global error signals (e.g., backpropagation). This type of adaptation is here referred to as local plasticity and is the main focus of the present work.

Local plasticity mirrors the mechanisms that biological systems use to learn and is also attractive as a means to design online training algorithms for neuromorphic hardware, where propagating global error signals can be costly. Among local learning rules, STDP is one of the most established. It captures the dependence of synaptic change on the relative timing of pre- and post-synaptic spikes and has long been regarded as a plausible mechanism for unsupervised feature representation [11]. However, this same locality also defines its main limitation: standard STDP strengthens responses to recurring input patterns, but provides no direct mechanism for favouring features that are useful for a specific final task.

In this respect, reward-modulated STDP (R-STDP) introduces an important extension: by coupling local synaptic eligibility with a delayed scalar reinforcement signal, it allows weight updates to remain local while still depending on the task outcome [12]. R-STDP provides a biologically inspired form of three-factor learning that stands between purely unsupervised plasticity and fully supervised optimization. Previous work has shown that it can be effective as it promotes the learning of more discriminative features than pure STDP. Moreover, it enables classification to be performed directly in the spiking domain, eliminating the need for an external classifier [13, 14]. These results provide evidence that local reward-based plasticity can support effective task-driven learning in practical architectures, suggesting its applicability beyond simplified experimental settings.

Most prior work on R-STDP was obtained on relatively compact architectures. However, as model depth increases, its effectiveness becomes considerably less clear. In standard deep learning (DL), the benefits of increased depth are largely enabled by architectural strategies that improve optimization and representation learning, such as residual connections and multi-scale feature aggregation [15, 16]. Related ideas have also been shown to play an important role in modern deep SNNs, especially in gradient-based optimization where residual connections can significantly improve the final model accuracy [17]. However, it remains unclear whether residual paths and multi-branch feature aggregation retain the same value when the network is trained only through local STDP/R-STDP updates. In a deep spiking model trained with STDP or R-STDP, greater architectural complexity can improve representation learning but also exacerbates spike sparsity, unstable dynamics, and makes temporal credit assignment (TCA) more difficult. Currently, the balance between these effects is still not well understood.

In this work, we shed new light on the role of online learning rules in multi-layer convolutional SNNs, where features encode the time to first spike (TTFS) and training is achieved via local STDP and R-STDP. Our architectural design follows the objective of preserving informative spike-time evidence as it propagates through increasingly deep processing stages. Thanks to an original Multi-Depth Temporal Fusion (MDTF), such evidence is progressively refined by architecturally combining residual-inspired connections, multi-scale feature aggregation and consensus. Since learning for the internal network layers is based on STDP, without using a global error signal, any information that is suppressed by an intermediate stage cannot be directly recovered by later plasticity. This makes the representation reaching the final readout layer particularly important. Our SNN architecture is trained and tested on visual recognition tasks, considering both static images and event-based visual inputs, where increasing input variability makes the preservation and selective refinement of early temporal evidence progressively more important.

In the proposed SNN design, network topology and learning dynamics are tightly coupled: architectural choices (namely, our MDTF) determine not only which representations reach the final classifier, but also which spike-time events remain available to subsequent synaptic adaptation. We exploit this coupling explicitly, treating depth as a problem of temporal evidence routing under local learning rather than simply as the addition of further processing stages.

The proposed model is designed around this interaction between information flow and local plasticity, ensuring that useful spike-time evidence remains available as processing depth increases. The input layer, referred to as the “front end”, is based on a functional model of early vision. It performs deterministic transformations to move visual streaming data into feature vectors encoded in terms of spike latency, and consists of four stages: local decorrelation to reduce redundancy, polarity processing to separate positive and negative contrasts, gain control to stabilize response ranges, and latency coding to convert the resulting information into sparse first-spike times [18–22]. The following convolutional layers constitute the “SNN backbone”, trained using STDP. The deterministic MDTF adapts residual and inception-like ideas: early spike-time evidence is preserved, intermediate representations contribute through sparse residual events, and deeper evidence is retained only when it is temporally consistent with the early and intermediate representations. The final decision stage consists of a single fully connected spiking R-STDP readout with multiple prototypes per class that operates directly on the event representation produced by the backbone. The experimental evaluation then separates the roles of the visual front end, feature propagation, deeper branching, and reward-modulated decision learning. The main contributions of this work are summarized as follows:

1. We propose a label-free latency encoder that combines local decorrelation, polarity separation, contrast-context gating, and response calibration to convert local visual streaming input data into calibrated first-spike latency maps;

2. We design the MDTF, to selectively refine the temporal features produced by the backbone. It exploits residual and agreement-based temporal routing to preserve early spike-time evidence while allowing deeper branches to add sparse corrections to the initial code;

3. We propose a multi-prototype R-STDP readout that performs classification entirely in the spiking domain, reinforcing target-class prototypes while selectively punishing the most competitive non-target ones;

4. The proposed pipeline is thoroughly validated via ablation studies in four representative visual datasets (MNIST, Fashion-MNIST, CIFAR-10, and N-MNIST), highlighting the role of each architectural component. Strong classification performance is achieved, proving that selective multi-depth fusion significantly outperforms traditional STDP/R-STDP baselines on higher-variability visual tasks, achieving +18.2 pp on Fashion-MNIST and +29.2 pp on CIFAR-10. Moreover, activity-budget analyses show that the network retains high accuracy even when removing a large fraction of late or weak spike events, confirming its high data efficiency and reduced event-processing requirements. These results position our solution as a practical and effective framework for defining and training multi-layer SNNs in online learning scenarios.

The remainder of this paper is organized as follows. Section 2 reviews the main literature on SNN learning rules and deep spiking architectures. The proposed processing pipeline is outlined in Section 3, while Section 4 describes the spike-time signals and feature representations used by the proposed model. Section 5 introduces the local learning framework, and Section 6 presents and discusses the experimental results. Finally, Section 7 summarizes our findings and outlines future research directions.

## 2 Related Work

Learning in SNNs is usually approached either through local plasticity rules or through optimization methods directly adapted from deep learning. The former emphasizes biological plausibility by relying on local synaptic plasticity, whereas the latter prioritizes task performance and trainability through gradient-based optimization. The present work bridges these complementary directions by investigating increasingly deep convolutional architectures while utilizing local STDP/R-STDP learning mechanisms. In computational vision models, STDP has been used to learn selective features from unlabeled input, especially when paired with temporal coding and competition [23, 24]. Convolutional STDP systems have also been studied as unsupervised visual feature learners with latency-coded input, while classification is performed by a downstream classifier [25–27]. Related work has connected STDP to stable Hebbian learning, temporal prediction, and probabilistic inference in competitive spiking circuits [28–30]. These studies show that STDP can organize event-based visual representations into practical vision pipelines. However, they also expose its central limitation: because synaptic updates only rely on local temporal correlations, the learned features are not necessarily the most discriminative ones for the downstream task. Three-factor and reward-modulated rules address this limitation by combining local eligibility with a modulatory signal that carries information about outcome, error, or reward [12, 31–33]. With this approach, reward shapes synaptic changes that are still expressed through spike timing. R-STDP is therefore well matched to task-driven spiking systems with local updates and event-based computation. Beyond static image recognition, local plasticity and event-driven SNNs have also been studied in settings where temporal processing and online adaptation are sought, including neuromorphic sensing, event-based datasets, reservoir computing, and robotic control [3, 34–36]. These application domains differ substantially in their input statistics, temporal structure, and task objectives, making comparisons across them difficult. We therefore restrict our experimental study to visual classification, considering both static and event-based inputs under a common recognition setting, so that the effects of temporal encoding, local plasticity, and architectural routing can be examined without conflating them with differences in task formulation.

In vision models based on first-spike timing, reward modulation makes it possible to keep the computation fully spiking while biasing plasticity toward task-relevant features. R-STDP has been shown to improve over unsupervised STDP in object categorization [13] and was later incorporated into a convolutional architecture for digit recognition, where early layers were trained with STDP and deeper layers with reward-modulated plasticity [14]. These works are particularly relevant because classification remains entirely within the spiking network, without requiring a separate analog classifier. However, the reward signal can modulate only the spike-time features that reach the decision layer; information suppressed by earlier locally trained stages cannot be recovered through global error propagation. As inputs and architectures become richer, preserving the structure and timing of this evidence therefore becomes increasingly important.

![](images/0ff04a1dcf6d47cdcf28e389e7ff8388a02ea7c04a4665c315307bf009fb24c8.jpg)  
Figure 1: Schematic overview of the proposed locally trained TTFS architecture. The model uses a four-layer spiking convolutional backbone $( S _ { 1 } { - } S _ { 4 } )$ . The MDTF merges representations at different depths. It preserves the early code $P = H ^ { ( 1 ) }$ , incorporates the sparse residual contribution $\Delta _ { \mathrm { r e s } }$ derived from the intermediate representation $I = H ^ { ( 2 ) }$ and adds the agreement contribution $\Delta _ { \mathrm { a g r e e } }$ obtained from temporally consistent events in I and the deep representation $D = H ^ { ( 4 ) }$ . These components are concatenated into the final latency representation $H = [ P , \Delta _ { \mathrm { r e s } } , \Delta _ { \mathrm { a g r e e } } ] ,$ , which is classified by a reward-modulated multi-prototype R-STDP readout through global winner-take-all spike timing. The earliest output spike determines the class score, silent neurons are assigned the maximum time, and membrane potential is used only for tie breaking. Labels are used only to define the reward and punishment signals at the decision stage.

Under these settings, little is known about how intermediate spike-time representations are to be routed in deeper SNNs. In fact, under local plasticity, a deeper layer cannot rely on a global error signal to recover information that has already been suppressed. This makes the preservation and selective enrichment of early temporal evidence a central architectural problem. In conventional deep learning, depth becomes effective when information flow is controlled through mechanisms such as residual connections and multi-branch feature extraction [15, 16]. Similar ideas have been successful in gradient-trained SNNs, where residual paths help stabilize the training of deeper networks [17]. Under STDP and R-STDP, however, the role of these mechanisms is different. A skip path or a branch changes which spikes arrive, when they arrive, and which events remain available to local plasticity. This is the setting considered in the present work, where we investigate how biologically inspired processing, residual connections, inception-like branching, and reward-modulated decision learning can be combined to preserve useful temporal evidence in locally trained convolutional SNNs.

## 3 Processing Pipeline Overview

The proposed model is shown in Fig. 1. It is a fully SNN-based architecture organized around four core components: (i) a label-free TTFS front end to convert images into sparse latency maps; (ii) a stage-wise convolutional backbone to learn local temporal features via unsupervised STDP; (iii) an original MDTF based on residual and agreement-based temporal connections preserving the stable base code, while selectively incorporating early events from deeper branches; (iv) a multi-prototype R-STDP readout to perform the final classification. The entire pipeline is trained with local learning rules via STDP (convolutional layers S1–S4) and R-STDP (readout).

The front end defines the sensory latency code used by the stages that follow, producing a latency-based and temporally ordered representation. This module is inspired by functional early-vision processing.

The following convolutional stages (the backbone) learn local feature maps using unsupervised STDP, organizing recurring visual structure into a stable spiking code for subsequent classification. The MDTF merges deeper representations utilizing residual connections and inception-like encoding with the objective of preserving the initial code, while providing additional refinements. Under local learning, the MDTF serves a different purpose than in backpropagation-trained convolutional neural networks (CNNs): rather than facilitating gradient flow, it determine which spike-time events remain available to subsequent stages.

The final classifier is a spiking R-STDP readout trained via local spike-timing-dependent synaptic updates modulated by reward and punishment signals. Population coding is also exploited by using multiple neurons for each class.

## 4 Signals and Features

## 4.1 Spike-Time Representation

An activation in the considered model is represented as the time of its first spike. For a feature index i, we denote this latency by $L _ { i } .$ . Earlier spikes correspond to stronger temporal evidence, while missing spikes are treated as silent features. The same convention is used for input maps, intermediate convolutional responses, residual paths, and readout inputs. This representation determines how architectural operations are defined. Concatenating feature tensors preserves multiple sources of spike-time evidence as separate channels. Selecting top-k events retains only the earliest or most informative latency changes within a sample. Agreement between branches is also temporal: two branches are considered consistent when they produce early events at corresponding locations or feature groups whose spike times fall within a predefined temporal tolerance. These are the general principles by which the backbone manipulates latency fields directly throughout the network.

## 4.2 Visual Front End and Latency Encoding

The visual front end has been designed taking inspiration from a functional early-vision module [37]. It reshapes the input into a form that can be used by spike-timing plasticity, using operations that parallel key computational roles of early sensory pathways, namely redundancy reduction, contrast-polarity separation, local gain control, and sparse latency coding. Local decorrelation reduces input patch-level correlations in line with efficient-coding principles of early vision [18, 19, 37]. Polarity splitting treats positive and negative signals as distinct non-negative event channels, analogously to ON/OFF visual pathways [20]. Signed-context gating and polarity balancing act as deterministic contrast-normalization steps, serving the functional role of gain control without specifying a circuit-level inhibitory mechanism [21, 38]. The final response-to-latency map then returns a sparse time-to-first-spike code, where stronger responses fire earlier and weak ones remain silent [22, 23, 39].

This initial stage establishes the spike-time representation that serves as the basis for all subsequent local plasticity mechanisms and corresponds to the Preprocessing block in the pipeline of Fig. 1. We denote the transformation carried out by this front end as

$$
H ^ { ( 0 ) } = \Phi ( { \boldsymbol { x } } ) ,\tag{1}
$$

where x denotes the input sample and $H ^ { ( 0 ) }$ is the output expressed as a latency tensor, which is presented to the next convolutional spiking layer (block S1 in Fig. 1). The mapping Φ is label-free. For static inputs, its data-dependent statistics are estimated once from the training split and then frozen; for event-based inputs, the transformation uses fixed sample-wise operations and no dataset-level fitted statistics. Its purpose is to transform local sensory evidence into a sparse temporal code before STDP-based learning starts.

In what follows, we detail the processing steps constituting Φ. We first describe the transformations specific to static images, followed by the preprocessing adopted for event-based inputs. The resulting representations are then converted into spike times through a latency-encoding stage.

We denote the static input image tensor by $p \in \mathbb { R } ^ { C \times H \times W }$ , where $C$ is the number of image channels (one for grayscale images and three for RGB images), while H and W denote the spatial dimensions.

Local decorrelation. Local patches are decorrelated through a local-decorrelation transform, implemented as a regularized zero-phase covariance normalization whose statistics are estimated from local image patches sampled from the training set. Let $q _ { u } \in \mathbb { R } ^ { C _ { \Pi } P ^ { 2 } }$ be the vectorized $P \times P$ patch $\mathcal { P }$ of $\dot { \mathbf { \rho } } _ { p }$ centered at location u. The empirical mean $\mu$ and covariance $\Sigma$ are computed once from patches sampled from the training data set. The local covariance-normalization matrix is

$$
W = ( \Sigma + \epsilon I ) ^ { - 1 / 2 } ,\tag{2}
$$

where $\epsilon > 0$ is a small regularization constant that improves the conditioning of $W$ and ensures a well-defined and numerically stable inverse square root. For each patch, the decorrelated patch vector is computed as

$$
\tilde { q } _ { u } = { W } ( q _ { u } - \mu ) .\tag{3}
$$

The vector $\tilde { q } _ { u }$ is then reshaped onto the original $C _ { \Pi } \times P \times P$ patch layout, and the value at the central spatial position of each channel c is retained as the signed decorrelated response:

$$
z _ { c } ( u ) , \qquad c = 1 , \dots , C _ { \Pi } .\tag{4}
$$

Note that this operation can be equivalently implemented using a convolutional filter bank whose filters are determined by the local covariance-normalization matrix. The resulting maps have the same number of channels as the original sensory tensor, but their local second-order correlations have been reduced while preserving the spatial layout. The response $z _ { c } ( u )$ is real-valued, with magnitude measuring the strength of the local decorrelated contrast, and sign encoding the corresponding polarity.

Signed-context gating. Since the following spiking layers operate on non-negative events, both signs of $z _ { c } ( u )$ must be handled explicitly prior to performing latency encoding. Before splitting the two signs into separate event maps, we apply a local signed-context gate. The gate remains weak in locally one-sided regions, while it activates when both polarities have non-negligible support, progressively biasing the response toward the locally dominant sign. Specifically, for channel c and location u, we define the local positive and negative supports as

$$
\begin{array} { l } { { S _ { c } ^ { + } ( u ) = \displaystyle \sum _ { v \in \mathcal { N } ( u ) } \kappa ( v - u ) [ z _ { c } ( v ) ] _ { + } , } } \\ { { S _ { c } ^ { - } ( u ) = \displaystyle \sum _ { v \in \mathcal { N } ( u ) } \kappa ( v - u ) [ - z _ { c } ( v ) ] _ { + } , } } \end{array}\tag{5}
$$

where $[ \cdot ] _ { + }$ denotes the positive part function, $\mathsf { i . e . , } [ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ for $a \in \mathbb { R } , \mathcal { N } ( u )$ is a neighborhood of $u ,$ and $\kappa ( \cdot )$ is a normalized kernel function. The degree of mixed polarity is then measured by

$$
\rho _ { c } ( u ) = \frac { \operatorname* { m i n } ( S _ { c } ^ { + } ( u ) , S _ { c } ^ { - } ( u ) ) } { \operatorname* { m a x } ( S _ { c } ^ { + } ( u ) , S _ { c } ^ { - } ( u ) ) } .\tag{6}
$$

When both local supports are zero, we set $\rho _ { c } ( u ) = 0$ by convention. Values of $\rho _ { c } ( u )$ close to zero indicate a locally one-sided response, while larger values indicate that non-negligible contributions from both signs are present. We further define the gate strength as

$$
g _ { c } ( u ) = \alpha _ { \mathrm { c t x } } \ \mathrm { c l i p } \left( \frac { \rho _ { c } ( u ) - \rho _ { 0 } } { \rho _ { 1 } - \rho _ { 0 } } , 0 , 1 \right) ,\tag{7}
$$

where $\alpha _ { \mathrm { { c t x } } } \geq 0$ controls the maximum strength, $[ \rho _ { 0 } , \rho _ { 1 } ]$ is the interval in which mixed-polarity evidence activates the gate, and ${ \mathrm { c l i p } } ( x , 0 , 1 )$ returns x if $x \in [ 0 , 1 ]$ , 0 if $x < 0 ,$ , and $1 \mathrm { i f } x > 1$

Let $d _ { c } ( u ) = \mathrm { s i g n } ( S _ { c } ^ { + } ( u ) - S _ { c } ^ { - } ( u ) )$ denote the locally dominant polarity around location u. We first define a competitive target response by preserving responses whose sign agrees with the locally dominant polarity and attenuating responses with the opposite sign by a factor $\eta \in [ 0 , 1 ] $

$$
\bar { z } _ { c } ( u ) = \left\{ \begin{array} { l l } { z _ { c } ( u ) , } & { \mathrm { s i g n } ( z _ { c } ( u ) ) = d _ { c } ( u ) , } \\ { \eta z _ { c } ( u ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

Here, $\eta = 1$ leaves both polarities unchanged, whereas smaller values increasingly suppress responses whose sign disagrees with the locally dominant one. The gated response is then obtained by interpolating between the original response and the competitive target:

$$
\hat { z } _ { c } ( u ) = \left( 1 - g _ { c } ( u ) \right) z _ { c } ( u ) + g _ { c } ( u ) \zeta _ { c } \bar { z } _ { c } ( u ) ,\tag{9}
$$

where $\zeta _ { c }$ is a channel-wise normalization factor chosen to preserve the total response energy of channel c in the competitive target. The gate $g _ { c } ( u )$ determines how strongly the local competition is applied. In one-sided regions, where one polarity clearly dominates and $g _ { c } ( u )$ is small, the original response is left nearly unchanged. When both polarities have non-negligible support within $\mathcal { N } ( u ) , g _ { c } ( u )$ increases and the response is progressively biased toward the locally dominant polarity. The competition is therefore local and label-free, and its purpose is to reduce sign ambiguity before the responses are converted into spike latencies.

Polarity split. After gating, the two polarities are explicitly represented as two separate non-negative maps

$$
r _ { c , \sigma } ( u ) = [ \sigma \hat { z } _ { c } ( u ) ] _ { + } , \quad \sigma \in \{ + , - \} .\tag{10}
$$

Thus, for static inputs, the polarity split maps each signed channel into two non-negative event channels. If the sensory tensor has $C _ { \Pi }$ channels, the split produces $2 C _ { \Pi }$ response maps. This representation makes both positive and negative contrast patterns available to STDP as non-negative spike events, ensuring that information carried by either polarity can contribute to local plasticity.

Calibration. For each polarity channel c and spatial location u, we estimate the extrema from the training split as

$$
m _ { c , \sigma } ( u ) = \operatorname* { m i n } _ { x \in \mathcal { D } _ { \mathrm { t r a i n } } } r _ { c , \sigma } ^ { ( x ) } ( u ) , \quad M _ { c , \sigma } ( u ) = \operatorname* { m a x } _ { x \in \mathcal { D } _ { \mathrm { t r a i n } } } r _ { c , \sigma } ^ { ( x ) } ( u ) .\tag{11}
$$

These values are then used to calibrate the response as:

$$
a _ { c , \sigma } ( u ) = \frac { r _ { c , \sigma } ( u ) - m _ { c , \sigma } ( u ) } { M _ { c , \sigma } ( u ) - m _ { c , \sigma } ( u ) } ,\tag{12}
$$

whenever $M _ { c , \sigma } ( u ) > m _ { c , \sigma } ( u )$ . If the two extrema coincide, we set $a _ { c , \sigma } ( u ) = 0$ . The extrema remain fixed after their initial estimation.

The calibrated positive and negative responses are then locally rebalanced within local patches. Let $\mathcal { P }$ denote the patch containing location u, over which polarity statistics and normalization are computed jointly. The rebalanced response is

$$
a _ { c , \sigma } ^ { \star } ( u ) = \lambda _ { \mathcal { P } } \omega _ { c , \sigma } ( u ) a _ { c , \sigma } ( u ) ,\tag{13}
$$

where $\omega _ { c , \sigma } ( u )$ is a deterministic local reweighting factor derived from the relative polarity activity, and $\lambda _ { \mathcal { P } }$ is chosen such that the summed positive and negative response mass within the local patch $\scriptstyle \dot { \mathcal { P } }$ is preserved. The reweighting reduces excessive positive-polarity dominance and suppresses weak conflicting responses in sufficiently active regions. Depending on the dataset configuration, polarity balancing may also include a local competition between oppositepolarity responses, attenuating the weaker response while preserving the overall local response magnitude. The resulting responses $a _ { c , \sigma } ^ { \star }$ are used for latency encoding.

Event-based data. For event-based inputs, the preceding static-image transformations are replaced by a dedicated preprocessing procedure, called “EventMap”, that operates directly on the asynchronous event stream, with each sensory sample represented as

$$
\mathcal { E } = \{ ( u _ { i } , t _ { i } , \sigma _ { i } ) \} _ { i = 1 } ^ { N } ,
$$

where $u _ { i } ~ = ~ ( x _ { i } , y _ { i } )$ denotes the spatial location, $t _ { i }$ the event timestamp, and $\sigma _ { i } ~ \in ~ \{ + , - \}$ its polarity. Before constructing the latency representation, isolated events are removed using the following filter, acting as a spatiotemporal denoiser. Specifically, an event at time $t _ { i }$ is discarded if no other event occurs within its one-pixel spatial neighborhood in $\lvert t _ { i } - \delta _ { t } , t _ { i } + \delta _ { t } \rvert$ , where $\delta _ { t }$ is a parameter. The temporal extent of each sample is then normalized between its first and last retained events. Specifically,

$$
\tau _ { i } = \frac { t _ { i } - t _ { \operatorname* { m i n } } } { t _ { \operatorname* { m a x } } - t _ { \operatorname* { m i n } } } ,\tag{14}
$$

and the normalized interval is partitioned into $B$ uniform temporal bins. Each event i is assigned to as specific bin $b _ { i }$ as follows,

$$
b _ { i } = \operatorname* { m i n } \left\{ B - 1 , \lfloor B \tau _ { i } \rfloor \right\} .\tag{15}
$$

Separate event-count maps are formed for each position $u _ { i }$ , temporal bin $b _ { i }$ and polarity σ:

$$
N _ { b , \sigma } ( u ) = | \{ i : u _ { i } = u , \sigma _ { i } = \sigma , b _ { i } = b \} | .\tag{16}
$$

Thus, temporal order is retained at a coarse resolution through the bin index $b ,$ while the number of events within each bin and polarity determines the local response strength.

Next, the count maps are logarithmically compressed and normalized with respect to their local spatial mean:

$$
q _ { b , \sigma } ( u ) = \frac { \log ( 1 + N _ { b , \sigma } ( u ) ) } { \operatorname { A v g } _ { v \in \mathcal { N } _ { r } ( u ) } \log ( 1 + N _ { b , \sigma } ( v ) ) } .\tag{17}
$$

A sample-wise normalization is subsequently applied across temporal bins, polarities, and spatial locations:

$$
a _ { b , \sigma } ^ { \star } ( u ) = \left[ \frac { q _ { b , \sigma } ( u ) } { \displaystyle \operatorname* { m a x } _ { b ^ { \prime } , \sigma ^ { \prime } , v } q _ { b ^ { \prime } , \sigma ^ { \prime } } ( v ) } \right] ^ { \gamma } .\tag{18}
$$

Zero-valued locations remain inactive. The resulting 2B non-negative maps (one per polarity) are passed to the common response-to-latency conversion described below.

Unlike the static-image front end, this event-based transformation does not rely on dataset-level statistics estimated from the training split. In fact, binning, denoising, compression, and normalization are deterministic operations, with normalization performed independently for each sample.

Latency encoding. Following the input-specific processing described above, both static and event-based representations are converted into spike times through a common latency-encoding procedure.

To simplify the notation, the non-negative response maps produced by the front end are collected along a common channel index $j .$ . For static inputs, j indexes the polarity-resolved channels $( c , \sigma )$ , with $j = 2 c$ for positive polarity and $j = 2 c + 1$ otherwise. Instead, for event-based inputs j indexes the polarity-time channels $( b , \sigma )$ . In both cases, we denote the resulting response at spatial location u by $a _ { i } ^ { \star } ( u )$ . The final front-end stage converts these response strengths into first-spike latencies. Responses exceeding the unit range are saturated before the latency mapping:

$$
\tilde { a } _ { j } ^ { \star } ( u ) = \operatorname* { m i n } \{ a _ { j } ^ { \star } ( u ) , 1 \} .\tag{19}
$$

The normalized spike latency at channel $j$ and spatial location u is then obtained as:

$$
L _ { j } ( u ) = \left\{ \begin{array} { l l } { \ell _ { \mathrm { m a x } } - ( \ell _ { \mathrm { m a x } } - \ell _ { \mathrm { m i n } } ) \tilde { a } _ { j } ^ { \star } ( u ) , } & { a _ { j } ^ { \star } ( u ) > \ell _ { \mathrm { s i l e n t } } , } \\ { \infty , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{20}
$$

Here, $\theta _ { \mathrm { s i l e n t } }$ denotes the response threshold below which no spike is emitted. In the experiments reported in this paper, we use the normalized latency interval $[ \ell _ { \mathrm { m i n } } , \ell _ { \mathrm { m a x } } ] = [ 0 , 1 ]$ . Thus, stronger active responses produce earlier spikes, responses exceeding the unit range are assigned the minimum latency $\ell _ { \mathrm { m i n } } .$ and responses below threshold $\theta _ { \mathrm { s i l e n t } }$ remain inactive.

The front end therefore produces continuous latency maps. For static inputs, continuous latencies are passed directly to $S _ { 1 }$ , whereas event-based latencies are discretized on a finite temporal grid before entering the backbone. After front-end processing, static grayscale and RGB inputs respectively yield two and six latency maps, whereas event-based inputs yield one latency map for each polarity-time channel. As shown in Fig. 1, these maps constitute the output of the Preprocessing block and the subsequent input to S1, the first stage of the four-layer spiking convolutional backbone S1-S4. The complete processing sequence is summarized in Algorithm 1.

## 4.3 Design Principles for Multi-Depth Temporal Fusion

In this section, we discuss the general principles that underpin the design of the Multi-Depth Temporal Fusion mechanisms within the backbone of the proposed SNN. Their specific implementation is described in Section 4.4.

We distinguish three latency representations involved in the fusion process: P denotes the preserved shallow representation; I is an intermediate representation obtained by further processing $P ;$ and $D$ represents a deeper temporal representation derived from I, see Fig. 1.

The residual principle adopted here is to preserve P directly, while allowing the intermediate representation to contribute only through a sparse set of early events. Specifically, we define the residual contribution $\Delta _ { \mathrm { r e s } }$ as

$$
\Delta _ { \mathrm { r e s } } = \mathrm { T o p K } _ { k _ { \mathrm { r e s } } } ( I ) ,\tag{21}
$$

where $\mathrm { { T o p K } } _ { k } ( X )$ denotes the temporal top-k selection operator, which retains the k earliest finite events of the latency representation X and suppresses the remaining ones. In this way, deeper processing can enrich the representation without replacing the preserved shallow code.

Referring to the residual/agreement fusion component in Fig. 1, the inception-like aspect of the architecture arises from combining temporal evidence available at different processing depths. In addition to the sparse intermediate contribution $\Delta _ { \mathrm { r e s } } ,$ , the fusion mechanism retains deep events only when they are temporally consistent with the corresponding intermediate events. For events at the same feature location i, this occurs when both representations contain a finite event and their latency difference does not exceed the agreement tolerance $m _ { \mathrm { a g r e e } }$ . We therefore define the agreement candidate as

$$
\begin{array} { r } { \widetilde { \Delta } _ { \mathrm { a g r e e } } ( i ) = \left\{ \begin{array} { l l } { \mathrm { m i n } \{ I ( i ) , D ( i ) \} , } & { I ( i ) < \infty , \quad D ( i ) < \infty , } \\ { \quad | I ( i ) - D ( i ) | \le m _ { \mathrm { a g r e e } } , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{22}
$$

Algorithm 1 Label-free early-vision latency encoder   
Require: Training samples $\mathcal { D } _ { \mathrm { t r a i n } } .$ , sample x   
Ensure: Latency tensor $H ^ { ( 0 ) } = \Phi ( x )$   
Fit front-end statistics   
1: if static-image input then   
2: Extract patches from $\Pi ( { \mathcal { D } } _ { \mathrm { t r a i n } } ) .$   
3: Estimate $\mu , \Sigma$ and compute the local normalization matrix $W = ( \Sigma + \epsilon I ) ^ { - 1 / 2 } .$   
4: Estimate calibration statistics $\{ m _ { j } ( u ) , M _ { j } ( u ) \}$   
5: Estimate polarity-balancing normalizers.   
6: else   
7: Fix event-map binning and normalization parameters.   
8: end if   
9: Freeze all estimated quantities.   
Encode sample   
10: p ← Π(x)   
11: if static-image input then   
12: z ← LocalDecorrelate $( p ; \mu , W )$   
13: zˆ ← SignedContextGate(z)   
14: r ← PolaritySplit(ˆz)   
15: $a  \mathrm { C a l i b r a t e } ( r ; \{ \dot { m _ { j } } , M _ { j } \} )$   
16: a<sup>⋆</sup> ← BalancePolarities(a)   
17: else   
18: a<sup>⋆</sup> ← EventMap(p)   
19: end if   
20: $H ^ { ( 0 ) } \gets$ LatencyEncode(a<sup>⋆</sup>)   
21: return $H ^ { ( 0 ) }$

where ∅ denotes silence. When the agreement condition is satisfied, the earliest of the two events I and D is retained;   
otherwise, the corresponding location remains silent.

The agreement contribution is then further sparsified as

$$
\begin{array} { r } { \Delta _ { \mathrm { a g r e e } } = \mathrm { T o p K } _ { k _ { \mathrm { a g r e e } } } \left( \widetilde { \Delta } _ { \mathrm { a g r e e } } \right) . } \end{array}\tag{23}
$$

In this way, isolated or temporally inconsistent deep events are suppressed, while agreement-gated events indicate that deeper processing has produced temporal evidence consistent with the intermediate representation.

The final fusion combines the representation P with the sparse intermediate $\Delta _ { \mathrm { r e s } }$ and agreement-based $\Delta _ { \mathrm { a g r e e } }$ contributions, adding sparse and temporally validated evidence without replacing (but preserving) the early latency code $P .$

## 4.4 Instantiation of the Multi-Depth Temporal Fusion Mechanism

Throughout this section, $H ^ { ( \ell ) }$ denotes the latency representation exported after the corresponding processing stage, with $H ^ { ( 0 ) }$ denoting the encoded input to the backbone. The four-stage hierarchy is defined as

$$
\begin{array} { r l } & { H ^ { ( 1 ) } = C _ { 1 } \Big ( S _ { 1 } ( H ^ { ( 0 ) } ) \Big ) , } \\ & { H ^ { ( 2 ) } = C _ { 2 } \Big ( S _ { 2 } ( H ^ { ( 1 ) } ) \Big ) , } \\ & { H ^ { ( 3 ) } = S _ { 3 } ( H ^ { ( 2 ) } ) , } \\ & { H ^ { ( 4 ) } = S _ { 4 } ( H ^ { ( 3 ) } ) . } \end{array}\tag{24}
$$

We reserve H without a stage superscript for the final fused representation.

Referring to Fig. 1, the encoded input $H ^ { ( 0 ) }$ is first processed by the two early stages $S _ { 1 } / C _ { 1 }$ and $S _ { 2 } / C _ { 2 }$ , where the spiking layers learn local temporal features and the corresponding $C _ { 1 }$ and $C _ { 2 }$ operations perform deterministic min-latency pooling and export the resulting latency representations. The deeper path then processes $H ^ { ( 2 ) }$ through $S _ { 3 }$ and $S _ { 4 }$ stages, producing $\bar { H ^ { ( 4 ) } }$ , which provides the deep representation used for temporal agreement. Note that $H ^ { ( 3 ) }$ is an intermediate representation of this deep path and is not directly included in the final code.

Referring to the general construction introduced in the previous section, $H ^ { ( 1 ) } , H ^ { ( 2 ) }$ , and $H ^ { ( 4 ) }$ correspond respectively to the preserved $( P )$ , intermediate (I), and deep (D) representations. Hence, $\Delta _ { \mathrm { r e s } }$ is obtained from $\bar { H ^ { ( 2 ) } }$ , while $\Delta _ { \mathrm { a g r e e } }$ is computed from the temporal agreement between $H ^ { ( 2 ) }$ and $H ^ { ( 4 ) }$ . The fused representation passed to the classifier is therefore

$$
\begin{array} { r } { H = \left[ H ^ { ( 1 ) } , ~ \Delta _ { \mathrm { r e s } } , ~ \Delta _ { \mathrm { a g r e e } } \right] . } \end{array}\tag{25}
$$

This is the residual/inception-like fusion used in the proposed model. It is residual because the early $H ^ { ( 1 ) }$ code remains directly available to the classifier, while subsequent stages contribute only sparse additional evidence. Its inception-like character arises from combining representations obtained at different processing depths, with the deepest events retained only when they are temporally consistent with the intermediate code.

## 5 Local Learning Framework

This section defines the local plasticity rules and the training schedule for the backbone and the readout layers.

## 5.1 Learning Rules

Synaptic adaptation is local at the level of individual connections. The convolutional backbone and the readout share the same causal dependence on the relative first-spike times of pre- and postsynaptic units, but use different weight-dependent update functions. We write the local update in the general form

$$
\begin{array} { r } { \Delta w _ { i j } = \lambda _ { j } \left\{ \begin{array} { l l } { A ^ { + } F _ { + } ( w _ { i j } ) , } & { t _ { i } \leq t _ { j } , } \\ { - A ^ { - } F _ { - } ( w _ { i j } ) , } & { t _ { i } > t _ { j } , } \end{array} \right. } \end{array}\tag{26}
$$

where $A ^ { + } , A ^ { - } > 0$ denote the potentiation and depression amplitudes, respectively, and $\lambda _ { j }$ is a postsynaptic modulation factor.

For the convolutional STDP layers, the update magnitude depends exponentially on the current synaptic weight:

$$
F _ { + } ^ { \mathrm { c o n v } } ( w ) = e ^ { - \beta w } , \qquad F _ { - } ^ { \mathrm { c o n v } } ( w ) = e ^ { \beta ( w - 1 ) } .\tag{27}
$$

Potentiation therefore decreases toward the upper weight bound, whereas depression decreases toward the lower bound.   
Updated weights are clipped to the admissible interval.

The R-STDP readout instead uses an additive update,

$$
F _ { + } ^ { \mathrm { r e a d o u t } } ( w ) = F _ { - } ^ { \mathrm { r e a d o u t } } ( w ) = 1 ,\tag{28}
$$

so that the update magnitude does not directly depend on the current synaptic weight. Readout stability is instead enforced through weight clipping and fixed per-neuron $\ell _ { 1 }$ normalization. For unsupervised STDP layers, $\lambda _ { j }$ is only determined by local winner selection and layer-specific learning-rate scheduling. These layers therefore learn recurring temporal patterns without using labels.

Before the readout, the fused latency tensor H is flattened into a vector of presynaptic spike-time features. This operation changes only the tensor shape: each entry remains a first-spike latency or a silent feature. The readout is therefore implemented as a fully connected spiking decision layer: each output neuron receives the flattened temporal representation and emits one class-prototype latency. The readout follows the same pre/post spike-timing criterion introduced in (26), with the additive choice $\bar { F } _ { + } ^ { \mathrm { r e a d o u i } } ( w ) = F _ { - } ^ { \mathrm { r e a d o u t } } ( w ) = 1$ , while $\lambda _ { j }$ is determined by a class-level reward signal. Each class is represented by a small population of readout neurons; a technique known as “population coding”. Given an input sample with label $y ,$ the readout produces one latency per output neuron, and the representative latency $\tau _ { c }$ of class c is the earliest finite latency among its prototypes. Therefore, the predicted class is

$$
\hat { y } = \underset { c } { \mathrm { a r g m i n } } \tau _ { c } .\tag{29}
$$

Reward-modulated learning. Firstly, we define a sample-dependent temporal corridor around the current class responses:

$$
\bar { \tau } = \frac { 1 } { | \mathcal { F } | } \sum _ { c \in \mathcal { F } } \tau _ { c } , \quad \quad \tau _ { y } ^ { \star } = \bar { \tau } - \frac { m } { 2 } , \quad \quad \tau _ {  y } ^ { \star } = \bar { \tau } + \frac { m } { 2 } ,\tag{30}
$$

where $\mathcal { F }$ is the set of classes with finite readout activity and m is a temporal margin. The target class $c = y$ is encouraged to fire before the corridor center, while non-target classes are encouraged to fire after it. The modulation

that is applied to the target class is proportional to the violation of this temporal objective. Hence, for each output neuron $\cdot j$ belonging to the target class (set $\mathcal { P } _ { y } )$ , it holds

$$
\lambda _ { j } ^ { + } = \mathrm { c l i p } \left( \beta _ { + } \frac { [ \tau _ { j } - \tau _ { y } ^ { \star } ] _ { + } } { \tau _ { \operatorname* { m a x } } } , 0 , \lambda _ { \operatorname* { m a x } } \right) , \qquad j \in \mathscr { P } _ { y } ,\tag{31}
$$

where $\tau _ { \mathrm { m a x } }$ is the maximum normalized latency scale, $\beta _ { + }$ is a target-side gain, and $\lambda _ { \mathrm { m a x } }$ limits the magnitude of the update. According to (31), a target neuron that already fires early receives little or no update; a target neuron that fires too late receives a stronger rewarded STDP update.

For each non-target class $c \neq y$ , we compute the distance of the corresponding output from the upper margin $\boldsymbol { \tau } _ { \lnot y } ^ { \star }$

$$
v _ { c } = [ \tau _ { \lnot y } ^ { \star } - \tau _ { c } ] _ { + } , \qquad c \neq y .\tag{32}
$$

Punishment is sparse. Rather than depressing every non-target class, the proposed rule selects the $K$ classes that violate the temporal margin the most. This sparse update ensures stability, by avoiding unnecessary depression of non-competing classes and concentrating plasticity on the outputs that currently interfere the most with the target decision. Accordingly, only the K largest violations are treated as hard negatives (set $\mathcal { P } _ { \mathrm { h n } } )$ . For each selected non-target class, readout neurons $j \in \mathcal { P } _ { \mathrm { h n } }$ are updated with anti-STDP:

$$
\lambda _ { j } ^ { - } = - \operatorname * { c l i p } \left( \beta _ { - } \frac { v _ { c ( j ) } } { \tau _ { \operatorname* { m a x } } } , 0 , \lambda _ { \operatorname* { m a x } } \right) , \qquad j \in \mathscr { P } _ { \mathrm { h n } } ,\tag{33}
$$

where $c ( j )$ is the class of neuron $j$ and $\mathcal { P } _ { \mathrm { h n } }$ is the hard-negative prototype set. When multiple prototypes within the same class are updated, they are ranked by increasing output latency, with decreasing membrane potential used to break equal-latency ties. Using multiple prototypes allows each class to be represented by different spike-time response patterns rather than by a single readout neuron, while updating only the most relevant prototypes limits the number of synapses modified for each sample. The sign of $\lambda _ { j }$ selects reward or punishment, while the synaptic update itself remains timing-based. Positive modulation (31) applies the rewarded STDP rule to target prototypes; negative modulation (33) applies the anti-STDP rule to confusing non-target prototypes. For correctly classified samples, the same margin-based updates are retained, but their modulation is reduced by a dataset-specific scaling factor. If no output neuron fires, output latencies are assigned the maximum latency and membrane potential is used to resolve the resulting tie; in this low-activity condition, anti-STDP (33) is suppressed, and only target-side update (31) is applied. This readout training approach remains within the R-STDP family, replacing the purely binary reward of pure-STDP with a local temporal objective focused on the classes that actually compete for the decision.

The mechanism amounts to a temporal competition between class populations: target-class prototypes are reinforced when they fire too late, whereas the most competitive non-target prototypes are punished when they fire too early. The remaining details, including clipping of modulation magnitudes, rank-based decay across prototypes, and silent-state guards, are stabilization mechanisms used to keep the readout in an active but sparse firing regime.

## 5.2 Training Schedule

Training follows the model structure. The network is trained layer by layer, with no end-to-end backpropagation. Each trainable layer is exposed to the spike-time representation produced by the preceding layers, updates only the synapses assigned to that layer, and is then frozen before the next layer is trained. This schedule restricts each backbone layer to local spike-timing information, while task-dependent modulation is introduced only at the readout.

The visual front end is configured before training the backbone. For static inputs, the patch mean $\mu ,$ covariance $\Sigma ,$ local covariance-normalization matrix $W ,$ , calibration ranges, and polarity-balancing normalizers are estimated once from the training split and then kept fixed for training, validation, and test phases. For event-based inputs, the front end instead uses fixed denoising, temporal binning, and normalization operations applied independently to each sample, without using dataset-level statistics. No label information is used in either case and the front end acts as a deterministic encoder, which makes it possible to reuse the same latency maps across all backbone and readout variants.

The convolutional backbone is trained layerwise. The spiking layers $S _ { 1 }$ and $S _ { 2 }$ learn recurring local temporal patterns through unsupervised STDP, while $C _ { 1 }$ and $C _ { 2 }$ deterministically export the corresponding latency representations. Once a spiking layer has been trained, it is frozen and its output representation is cached for the following layers. The deeper section of the backbone processes the intermediate representation $I = H ^ { ( 2 ) }$ through $S _ { 3 }$ and $S _ { 4 } ,$ producing the deep representation $D = H ^ { ( 4 ) }$ used for temporal agreement. The final representation is then constructed deterministically by preserving $P = H ^ { ( 1 ) }$ , extracting the sparse residual contribution $\Delta _ { \mathrm { r e s } }$ from $I ,$ and adding the agreement contribution $\Delta _ { \mathrm { a g r e e } }$ obtained from temporally consistent events in I and D. These three contributions are concatenated to form the fused latency representation $H \doteq [ P , \Delta _ { \mathrm { r e s } } , \Delta _ { \mathrm { a g r e e } } ]$ used by the readout classifier, see (25).

Table 1: Training schedule of the proposed locally trained pipeline.
<table><tr><td>Stage</td><td>Plasticity</td><td>Role in the pipeline</td></tr><tr><td>Visual front end</td><td>None; deterministic preprocessing</td><td>Converts static images or event streams into label-free latency repre- sentations. Static-input statistics are estimated once from the training split, whereas event-based normalization is applied independently to</td></tr><tr><td>Early convolutional stages</td><td>Unsupervised STDP in S1-S2; deter- ministic  $C _ { 1 } { - } C _ { 2 }$  export</td><td>each sample. To learn early temporal features and produce the preserved representa- tion  $P = \dot { H } ^ { ( 1 ) }$  and the intermediate representation  $I = H ^ { ( 2 ) }$  ; each trained spiking stage is frozen before subsequent stages are fitted.</td></tr><tr><td>Deep convolutional path</td><td>Unsupervised STDP in  $S _ { 3 } – S _ { 4 }$ </td><td>Further processes I to produce the deep representation  $D = H ^ { ( 4 ) }$  used together with I by the temporal agreement mechanism.</td></tr><tr><td>Sparse Multi-Depth Temporal Fusion</td><td>None; deterministic selection and agreement</td><td>Preserves P, extracts the sparse residual contribution  $\Delta _ { \mathrm { r e s } }$  from  $I ,$  and adds the agreement contribution  $\Delta _ { \mathrm { a g r e e } }$  obtained from temporally consistent events in I and D.</td></tr><tr><td>Multi-prototype readout</td><td>Reward-modulated STDP</td><td>Learns class decisions from the frozen fused representation  $H =$   $[ P , \Delta _ { \mathrm { r e s } } , \Delta _ { \mathrm { a g r e e } } ]$  using target-side reinforcement, hard-negative anti- STDP, and multiple prototypes per class.</td></tr></table>

The readout layer is trained after the backbone has been fixed. During readout training, each input sample is forwarded through the frozen backbone, output latencies are computed, and the R-STDP update described in Section 5.1 is applied immediately (samplewise). This preserves the online character of the learning rule and keeps the supervision signal local to the final decision stage.

Table 1 summarizes the resulting schedule. Exact numerical hyperparameters, such as epoch counts, thresholds, top-k values, temporal margins, and modulation caps, are treated as experimental settings and specified in Appendix A.

This layer-wise organization also determines the ablation study presented in Section 6. Its core components are the latency front end, the locally trained STDP backbone, the sparse MDTF module, and the multi-prototype R-STDP readout layer. Front-end components are evaluated separately through dedicated controls, while readout stabilization mechanisms are kept fixed within each reported operating point. This separation makes it possible to distinguish improvements due to a better input latency code, the residual and inception-based temporal routing in the backbone, or reward-modulated decision learning.

## 6 Experiments and Results

The experimental evaluation is designed to test the central hypothesis of this work: deep architectural structure can be made useful in an SNN trained by local synaptic plasticity. All reported models use the layer-wise training protocol described in Section 5. The preprocessing transform is estimated from the training data split without using labels, the convolutional spiking backbone is trained by local STDP, and the final readout layer is trained on frozen spike-time features using reward-modulated STDP. Class labels are therefore only used by the reward signal of the readout; they are not used to train the backbone.

For reproducibility, the settings for all training parameters are reported in Appendix A for all the considered datasets.

## 6.1 Datasets and Performance Metrics

Benchmarks. The proposed design is evaluated on static and event-based tasks. The considered datasets have an increasing complexity. MNIST (static, grayscale digits) is the simplest, providing a low-variability setting. Fashion-MNIST (static, grayscale apparel) preserves the same spatial format but introduces a substantially more heterogeneous shape distribution across input samples. CIFAR-10 (static, RGB images) features more complex input images with colors. N-MNIST (dynamic, event-based) features event-based signals which serve to evaluate the event-based version of the front end.

Performance metrics. The primary metric is test-set accuracy. For the R-STDP readout, however, accuracy is interpreted jointly with firing stability. The epoch used for reporting test accuracy is selected according to validation performance. Test-set trajectories are reported only to assess the stability of the readout dynamics over training and are not used for model selection.

Table 2: Main classification results and frozen-code statistics. Accuracy is reported in percent on the test split for validation-selected operating points; multi-seed rows report mean ± standard deviation over readout seeds. Code dimension, mean active events, and active density are measured on the sparse latency representation delivered to the readout. All rows use the same four-layer locally trained residual/agreement spiking backbone and a reward-modulated temporal R-STDP readout.
<table><tr><td>Dataset</td><td>Input structure</td><td>Code dim.</td><td>Events/sample</td><td>Density</td><td>Accuracy (%)</td></tr><tr><td>MNIST</td><td>28×28 grayscale digits</td><td>17408</td><td>994.5</td><td>5.71%</td><td> $9 6 . 6 \pm 0 . 6$ </td></tr><tr><td>Fashion-MNIST</td><td>28×28 grayscale apparel</td><td>17408</td><td>1182.3</td><td>6.79%</td><td> $8 6 . 3 \pm 0 . 7$ </td></tr><tr><td>CIFAR-10</td><td>32×32 RGB natural images</td><td>24704</td><td>3100.6</td><td>12.6%</td><td> $6 2 . 5 \pm 0 . 4$ </td></tr><tr><td>N-MNIST</td><td>34×34 polarity event streams</td><td>24704</td><td>1992.7</td><td>8.07%</td><td> $9 5 . 1 \pm 0 . 6 $ </td></tr></table>

Table 3: Front-end component configurations. Table entries report test accuracy in percent on the full train/test protocol, using the same four-layer backbone architecture and R-STDP readout framework.
<table><tr><td>Dataset</td><td>Front-end configuration</td><td>Main configuration change</td><td>Accuracy</td></tr><tr><td>MNIST</td><td>simple latency</td><td>local decorrelation and polarity processing</td><td>9.80</td></tr><tr><td>MNIST</td><td>full without signed context</td><td>signed-context gate</td><td>95.41</td></tr><tr><td>MNIST</td><td>full without polarity balancing</td><td>local polarity-balance gates</td><td>95.02</td></tr><tr><td>MNIST</td><td>full static front end</td><td></td><td>96.68</td></tr><tr><td>Fashion-MNIST</td><td>simple latency</td><td>local decorrelation and polarity processing</td><td>10.00</td></tr><tr><td>Fashion-MNIST</td><td>full without signed context</td><td>signed-context gate</td><td>86.23</td></tr><tr><td>Fashion-MNIST</td><td>full without polarity balancing</td><td>local polarity-balance gates</td><td>86.24</td></tr><tr><td>Fashion-MNIST</td><td>full static front end</td><td></td><td>86.31</td></tr><tr><td>CIFAR-10</td><td>RGB latency</td><td>local decorrelation, polarity split, calibration</td><td>9.8</td></tr><tr><td>CIFAR-10</td><td>decorrelation + polarity + calibration</td><td>local polarity balancing</td><td>60.48</td></tr><tr><td>CIFAR-10</td><td>channel-peak calibration</td><td>dataset-level response calibration</td><td>58.12</td></tr><tr><td>CIFAR-10</td><td>full static front end</td><td></td><td>62.50</td></tr><tr><td>N-MNIST</td><td>raw event frames</td><td>event-map construction</td><td>93.16</td></tr><tr><td>N-MNIST</td><td>full map without denoising</td><td>event denoising filter</td><td>93.49</td></tr><tr><td>N-MNIST</td><td>full map without log compression</td><td>log compression</td><td>93.51</td></tr><tr><td>N-MNIST</td><td>full event-map</td><td></td><td>95.01</td></tr></table>

## 6.2 Main Results

The numerical results show a consistent pattern across the four benchmarks, see Table 2. On MNIST, the locally trained representation/readout pipeline reaches the high-accuracy regime expected for a low-variability digit task, indicating that the proposed latency code does not sacrifice performance in this simple setting. Fashion-MNIST introduces substantially more shape variability across input samples; the corresponding results show that the proposed SNN is as well successful on a more heterogeneous recognition problem. CIFAR-10 is more demanding because the front end in this case must preserve weak color and texture evidence, impacting the accuracy. The event-based N-MNIST is a dynamic input: also in this case, once asynchronous polarity events are converted into a locally normalized event-map latency representation, the proposed four-layer backbone and R-STDP readout remain effective.

The code statistics in Table 2 show that these accuracies are obtained from sparse frozen representations. As the input structure becomes richer, the exported code uses more active latency events, but remains sparse relative to its full dimensionality. Fig. 2 provides a stability check on the readout dynamics: test accuracy remains close to its best value over the final training phase, with no isolated transient peak.

## 6.3 Front-End Configuration Ablations

The input encoder implements a sequence of deterministic operations. For each, in this section we compare the full front end with alternative configurations in which selected processing mechanisms are removed or replaced. The backbone and readout retain the same four-layer architecture and learning framework within each dataset.

![](images/2e84ce29761db44948637a06667c8ca417e69e6477d8680e77bde00bd06d2b96.jpg)

![](images/fd67237c5fc7f789a76dfbbadbf3e83cba6dd597f806d96d32eabe2fd5c9f98f.jpg)

![](images/a4527578d1b6120f264d2509b1352135aeb57f414f9c9785510923e302f8e807.jpg)

![](images/497df6657b91034db6c320b122f3c4cd28ef4b0267bc21f2d0db94a954b4b780.jpg)

![](images/88a441d98c782fdc21b2580244f2c22c13bde2854bf24ff991e12bc56d5d8d55.jpg)  
Figure 2: Training dynamics of the reward-modulated readout on the main datasets. Solid black curves report test accuracy, dashed gray curves report training accuracy, and the vertical dotted line marks the best validation epoch. The panels show stable readout regimes, with test accuracy remaining close to its best value over the final training phase.

The results of this ablation study are reported in Table 3. Direct latency coding from raw intensity (termed “simple latency” or “RGB latency”, depending on the dataset) is close to chance on all static benchmarks, showing that a usable temporal code does not emerge from raw intensity ordering alone. On MNIST, removing signed-context or local polarity-balance processing produces measurable losses, whereas on Fashion-MNIST all configurations perform close to the “full front end”. On CIFAR-10, local decorrelation combined with polarity split and response calibration achieves most of the full static-front-end performance, whereas the alternative channel-peak configuration performs less effectively. On the N-MNIST event-based dataset, raw event frames already provide a strong event-based code, achieving accuracies higher than 93%, but the full event-map representation returns the best results among the tested configurations. In this case, removing denoising or log compression gives a modest but consistent loss.

## 6.4 Comparison against State-of-the-Art STDP/R-STDP Baselines

Next, we compare the proposed architecture with a local STDP/R-STDP baseline that shares the same broad processing logic: temporal coding, unsupervised STDP feature learning, and a final spiking decision stage trained through reward-modulated plasticity. The selected baseline follows the convolutional STDP/R-STDP scheme of [14], which is particularly close to our setting because classification is performed directly in the spiking domain without an external classifier. Other local spike-timing approaches follow related but different learning schemes, for example by combining unsupervised STDP features with a separate classifier [26, 27] or by using directly supervised synaptic updates [40, 41] instead of reward-modulated plasticity. The comparison is therefore intended to contextualize the effect of the proposed representation and routing mechanisms within a closely matched local STDP/R-STDP setting.

Table 4 presents the experimental results of this comparison. On MNIST, the baseline already provides high-accuracy, and the proposed model does not present noticeable improvements. The advantage offered by the proposed architecture appears on more complex datasets where the early latency code alone is no longer sufficient. On Fashion-MNIST and CIFAR-10, our method gives major gains over the baseline. The baseline underperforms on N-MNIST; this is because it does not account for a front end layer to express dynamic inputs in a latency format that is suitable for further

Table 4: Comparison with a local STDP/R-STDP baseline from [14]. Accuracies are reported in percent with 95% Wilson confidence intervals over the test split. The p-values are Holm-corrected two-proportion tests at the aggregate accuracy level. Very small corrected values are clipped at $1 0 ^ { - 1 6 }$ for readability.
<table><tr><td>Dataset</td><td>Local STDP/R-STDP baseline</td><td>Proposed full model</td><td>∆pp</td><td>PHolm</td></tr><tr><td>MNIST</td><td>97.0 [96.6,97.3]</td><td>96.7 [96.3,97.1]</td><td>-0.3</td><td>0.18</td></tr><tr><td>Fashion-MNIST</td><td>68.2 [67.5,69.1]</td><td>86.3 [85.6,87.0]</td><td> $+ 1 8 . 2$ </td><td> $< 1 0 ^ { - 1 6 }$ </td></tr><tr><td>CIFAR-10</td><td>33.3 [32.4,34.2]</td><td>62.5 [61.6,63.5]</td><td> $+ 2 9 . 2$ </td><td> $< 1 0 ^ { - 1 6 }$ </td></tr><tr><td>N-MNIST</td><td>22.0† [21.2,22.8]</td><td>95.0 [94.6,95.5]</td><td> $+ 7 3 . 0$ </td><td> $< 1 0 ^ { - 1 6 }$ </td></tr></table>

<sup>†</sup>Direct-transfer diagnostic: the local STDP/R-STDP baseline is applied to N-MNIST event streams without the event-map preprocessing used by the proposed model.

Table 5: Data-efficiency for the proposed pipeline. Entries are test accuracies in percent vs number of training examples.
<table><tr><td>Dataset</td><td>100</td><td>500</td><td>1k</td><td>5k</td><td>10k</td><td>20k</td><td>30k</td></tr><tr><td>MNIST</td><td>41.45</td><td>78.41</td><td>84.08</td><td>91.90</td><td>93.17</td><td>94.28</td><td>94.93</td></tr><tr><td>Fashion-MNIST</td><td>49.59</td><td>71.20</td><td>74.42</td><td>80.73</td><td>82.45</td><td>84.37</td><td>85.02</td></tr><tr><td>CIFAR-10</td><td>12.57</td><td>34.98</td><td>41.66</td><td>53.54</td><td>56.40</td><td>59.31</td><td>61.16</td></tr><tr><td>N-MNIST</td><td>27.67</td><td>51.43</td><td>68.11</td><td>88.50</td><td>90.78</td><td>92.44</td><td>93.29</td></tr></table>

SNN processing. This result highlights the importance of transforming asynchronous event streams into a spike-time representation that is compatible with the backbone.

## 6.5 Data Efficiency

The data-efficiency experiments retrain the layer-wise pipeline at progressively larger training budgets, while keeping the readout training protocol fixed. Reducing the number of examples affects both the unsupervised STDP layers and the reward-modulated readout, so the experiment probes the full local-learning pipeline, not only the final classifier. Table 5 reports the resulting accuracy results.

Fig. 3 shows the same experiment as a scaling curve. Accuracy increases monotonically on all four datasets, indicating that both the STDP backbone and the R-STDP readout benefit from additional training samples. MNIST and Fashion-MNIST show the expected early-saturation pattern for grayscale recognition, with Fashion-MNIST reaching a lower ceiling because of its larger intra-class variability. CIFAR-10 remains the most data-demanding static benchmark, while N-MNIST improves sharply once the event-map representation leaves the few-sample regime. The monotonic scaling indicates that the locally trained pipeline continues to benefit from additional training data over the tested range.

## 6.6 Spike-Budget Pareto

The sparse latency representation at the output of the backbone allows the final readout layer to be evaluated under an explicit activity budget. To do so, for each input sample we retain the top-scoring sparse spike features at the output of the backbone and retrain the readout on the truncated code. The score is induced by latency, so earlier spikes correspond to stronger retained evidence. This experiment does not change the trained backbone. Rather, it probes how much of its final representation is required by the readout layer to provide a certain performance level. The truncation amounts to removing weak or late events from the backbone output. Because the datasets have different native activity levels, the retained budget is expressed as a percentage of the mean full-code event count on the test split.

Fig. 4 shows a gradual loss of accuracy as activity is reduced. The simpler static benchmarks retain most of their full-code performance at relatively small budgets, whereas CIFAR-10 and N-MNIST require more retained events to reach the high-accuracy regime. This indicates that the fused sparse code contains a concentrated subset of high-value early evidence, and that removing later or weaker events is a suitable means to control the tradeoff between maximum accuracy and readout activity.

## 6.7 Residual and Agreement Ablations

The routing ablation clarifies the role of depth in the proposed architecture. We expect representations obtained at increasing processing depth to be most effective when they provide sparse corrections to the preserved representation

![](images/391808b462a9faff7af186d8ee0b95ff2912f5df42db489e5f7efd046a12515f.jpg)  
(a) Accuracy scaling.

![](images/df9b25fbf05d507b916a8e7e4a3a9a6628edfdac6b30d08aadde2e7139001a66.jpg)  
(b) Train-test gap.

Figure 3: Data-efficiency behaviour of the locally trained pipeline. The figure uses line pattern and marker shape rather than color: circles denote MNIST, squares denote Fashion-MNIST, triangles denote CIFAR-10, and diamonds denote N-MNIST.  
![](images/bb37bbffc508aeaa9da2050760ddd51354368402a288b067cf743f468d28af99.jpg)  
(a) Spike-budget Pareto.

![](images/00c88394f373faf868c7111f5388ee7d8eb11f3bbe11f5c4c5ba8c6e4453973e.jpg)  
(b) Operating-point budget.  
Figure 4: Spike-budget Pareto and event-operation proxy for the final sparse latency code. (a) Test accuracy relative to the matched full-code readout after retaining only the strongest latency events in each sample and retraining the R-STDP readout with the backbone fixed. The horizontal dotted line marks the untruncated code normalized to 100% (b) Zoom on the high-accuracy operating regime. Filled markers indicate the interpolated retained activity required to preserve 95% of full-code accuracy, yielding a compact proxy for the event-operation budget of the readout code.

$P = H ^ { ( 1 ) }$ , not as standalone replacements. To check this, we compare residual readouts that preserve $P$ and progressively incorporate contributions derived from the intermediate representation $I = H ^ { ( 2 ) }$ and the deep representation $D = \overline { { H ^ { ( 4 ) } } }$ . For the diagnostic route without temporal agreement, we denote the directly selected deep contribution by

$$
\Delta _ { D } = \mathrm { T o p K } _ { k _ { D } } ( D ) .\tag{34}
$$

This quantity is used only in the ablation: the proposed model instead retains the agreement-filtered contribution $\Delta _ { \mathrm { a g r e e } } ,$ see (23). All variants use the same trained frozen backbone and the same temporal R-STDP readout protocol; only the combination of backbone representation presented to the readout is changed.

Fig. 5a shows that the preserved representation P remains the dominant evidence stream, as expected. However, additional contributions from later representations I and D provide selective refinements, especially on the highervariability datasets. On MNIST, where P already provides a highly discriminative code, incorporating the additional sparse contributions produces little or no change in accuracy. The scaling analysis in Fig. 5b further characterizes the role of depth: serial replacement evaluates P (preserved), I (intermediate) and D (deep) in isolation, losing accuracy when P is discarded, and showing that I and D alone are insufficient. Residual routing keeps P available by incorporating sparse contributions derived from I and then from D, confirming that their integration leads to an increase in the accuracy.

![](images/8aff2b636ec4909bb6fce8e855881ed7c7dbd806e57c6a573d7d7ef2ebc1b0fd.jpg)

![](images/3896d2e3528e0fad5ed84a872d86a3b9e1923d0bc71f5831e33e9aa72ec00872.jpg)  
(a) Residual routing ablation.  
(b) Temporal evidence preservation.  
Figure 5: Residual routing and temporal-evidence preservation. (a) Percentage-point changes relative to the preserved representation $P = H ^ { ( 1 ) }$ for the residual route $[ P , \Delta _ { \mathrm { r e s } } ]$ , its direct deep extension $[ P , \Delta _ { \mathrm { r e s } } , \Delta _ { D } ]$ , and the proposed agreement-filtered code $[ P , \Delta _ { \mathrm { r e s } } , \Delta _ { \mathrm { a g r e e } } ]$ . (b) Mean retained accuracy across the four datasets, normalized by the $P \mathrm { - o n l y }$ route; shaded envelopes indicate across-dataset dispersion. At the three processing depths, the serial curve evaluates $P , I$ , and $D _ { : }$ , whereas the residual curve evaluates $\mathrm { \bar { \it P } } , [ { \it P } , \Delta _ { \mathrm { r e s } } ]$ , and $[ P , \bar { \Delta } _ { \mathrm { { r e s } } } , \Delta _ { \mathrm { { a g r e e } } } ]$ , respectively.

## 6.8 Decision-Level Confusion Repair

To analyze how the fused representation changes individual decisions, we compare the P-only readout with the full fused representation $H = [ \bar { P } , \Delta _ { \mathrm { { r e s } } } , \Delta _ { \mathrm { { a g r e e } } } ]$ on the same test samples. Each input sample is assigned to one of three transition types: a repaired error, when the prediction from P is wrong and the prediction from H is correct; a new error, when the prediction from $P$ is correct and that from H is wrong; or a changed but unresolved error, when both predictions are wrong but the predicted class changes. This analysis separates genuine corrections from newly introduced errors and changes among already incorrect predictions.

Fig. 6 shows that the full representation only modifies a limited fraction of the test decisions. On Fashion-MNIST, CIFAR-10, and N-MNIST, repaired errors outnumber newly introduced errors, so the additional contributions provide a positive net effect on classification accuracy. CIFAR-10 shows the largest amount of decision change: H repairs a visible subset of the errors made from P, while another subset changes predicted class but remains incorrect. MNIST behaves differently because the P-only representation is already close to saturation, and the additional contributions produce little net change. Fig. 6b shows the split between errors types: unchanged, changed plus unresolved and repaired. Most errors made from P remain unchanged, whereas $\Delta _ { \mathrm { r e s } }$ and $\Delta _ { \mathrm { a g r e e } }$ affect a dataset-dependent subset of the remaining decisions.

## 6.9 Interpretation of Representation and Routing

The front-end ablations show that the representation outputted by the front end is a major determinant of performance. On the static datasets, direct latency coding from raw intensities is insufficient, whereas the structured front end produces a spike-time representation that supports substantially higher accuracy. N-MNIST presents a different input regime: raw event frames are already informative, but the binned event-count latency representation provides the best performance among the tested configurations. Thus, the static and event-based front ends perform different input-specific transformations, while providing the backbone with a common TTFS representation.

The routing experiments then clarify how this representation should be propagated to the readout. The preserved representation $\hat { P }$ remains the main evidence source, whereas the intermediate and deep representations I and D are most useful through the sparse contributions defined by the residual and agreement mechanisms. In particular, preserving P is more effective than replacing it as processing depth increases, while $\Delta _ { \mathrm { r e s } }$ and $\Delta _ { \mathrm { a g r e e } }$ provide additional evidence without discarding the early code. The decision-level analysis further shows that these contributions modify only a subset of the decisions made from P, repairing part of the remaining errors rather than broadly changing the classifier output.

![](images/c299b41be3cf8482780cc5bb56d1556dc8b2ce0e094efe9408619c76631006ed.jpg)  
(a) Decision transitions $P \to H .$

![](images/edf89d4743dd578163703d1d0af24093282f6aa6d872525014d2e08f565c5023.jpg)  
(b) Fate of P-only errors.  
Figure 6: Decision transitions induced by the full fused representation $H = [ P , \Delta _ { \mathrm { { r e s } } } , \Delta _ { \mathrm { { a g r e e } } } ]$ relative to the P-only readout. (a) Fraction of test samples corresponding to repaired errors (P wrong, H correct), new errors (P correct, H wrong), and changed but unresolved errors (both predictions wrong, but with different predicted classes). (b) Fate of the errors produced by the $P \mathrm { - o n l y }$ readout, partitioned into repaired errors, changed but unresolved errors, and errors whose predicted class remains unchanged.

## 7 Concluding Remarks

This work proposes an original SNN multi-layer decision model amenable to online and localized STDP/R-STDP based training. It encodes image and visual streaming data into latency representations that are processed via multiple convolutional SNN layers, trained in an online fashion. A final readout layer, trained with a reward-modulated STDP mechanism, performs the final classification task.

Our primary architectural innovation combines concepts from residual connections, inception and consensus into a multi-layer SNN backbone. This design preserves early latency codes while progressively refining them, ensuring critical spike-time evidence remains available across deeper layers. Experiments show that preserving and refining the early temporal code is preferable to progressively replacing it with deeper representations. Across the considered datasets, the early code remains the main source of discriminative evidence, while the additional refinements become increasingly useful on more complex data. A decision-level analysis shows that these contributions affect only a subset of predictions, repairing part of the residual errors rather than broadly altering the decisions produced from the early code. Moreover, an activity-budget analysis shows that much of the classification performance is retained after removing a substantial fraction of the later or weaker events presented to the readout. On the higher-variability visual tasks, the performance of the so obtained SNN consistently surpass existing STDP/R-STDP baselines, positioning our framework as a viable and effective approach for online learning scenarios.

Future work can investigate whether the same temporal-routing principle extends to larger event-based tasks, recurrent architectures, and neuromorphic hardware, performing direct activity and energy measurements.

## CRediT authorship contribution statement

Aidin Attar: Conceptualization, Methodology, Software, Investigation, Writing – original draft, Writing – review & editing.

Eleonora Cicciarella: Methodology, Writing – review & editing.

Michele Rossi: Supervision, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI usage

During the preparation of this work, the authors used OpenAI’s ChatGPT to assist with language refinement and readability. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## References

[1] K. Roy, A. Jaiswal, P. Panda, Towards spike-based machine intelligence with neuromorphic computing, Nature 575 (2019) 607–617.

[2] J. D. Nunes, M. Carvalho, D. Carneiro, J. S. Cardoso, Spiking Neural Networks: A Survey, IEEE Access 10 (2022) 60738–60764.

[3] M. Davies, N. Srinivasa, T.-H. Lin, G. Chinya, Y. Cao, S. H. Choday, G. Dimou, P. Joshi, N. Imam, S. Jain, Y. Liao, C.-K. Lin, A. Lines, R. Liu, D. Mathaikutty, S. McCoy, A. Paul, J. Tse, G. Venkataramanan, Y.-H. Weng, A. Wild, Y. Yang, H. Wang, Loihi: A Neuromorphic Manycore Processor with On-Chip Learning, IEEE Micro 38 (2018) 82–99.

[4] O. Richter, C. Wu, A. M. Whatley, G. Köstinger, C. Nielsen, N. Qiao, G. Indiveri, DYNAP-SE2: a scalable multi-core dynamic neuromorphic asynchronous spiking neural network processor, Neuromorphic Computing and Engineering 4 (2024) 014003.

[5] C. Lee, P. Panda, G. Srinivasan, K. Roy, Training Deep Spiking Convolutional Neural Networks With STDP-Based Unsupervised Pre-training Followed by Supervised Fine-Tuning, Frontiers in Neuroscience 12 (2018).

[6] E. O. Neftci, H. Mostafa, F. Zenke, Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks, IEEE Signal Processing Magazine 36 (2019) 51–63.

[7] J. Ding, Z. Yu, Y. Tian, T. Huang, Optimal ANN-SNN Conversion for Fast and Accurate Inference in Deep Spiking Neural Networks, volume 3, 2021, pp. 2328–2336.

[8] S. Deng, S. Gu, Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks, 2021.

[9] T. Bu, W. Fang, J. Ding, P. Dai, Z. Yu, T. Huang, Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks, 2023.

[10] Z. Hao, T. Bu, J. Ding, T. Huang, Z. yu, Reducing ANN-SNN Conversion Error through Residual Membrane Potential, Proceedings of the AAAI Conference on Artificial Intelligence 37 (2023) 11–21.

[11] G.-q. Bi, M.-m. Poo, Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type, Journal of Neuroscience 18 (1998) 10464–10472.

[12] N. Frémaux, W. Gerstner, Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules, Frontiers in Neural Circuits 9 (2016).

[13] M. Mozafari, S. R. Kheradpisheh, T. Masquelier, A. Nowzari-Dalini, M. Ganjtabesh, First-Spike-Based Visual Categorization Using Reward-Modulated STDP, IEEE Transactions on Neural Networks and Learning Systems 29 (2018) 6178–6190.

[14] M. Mozafari, M. Ganjtabesh, A. Nowzari-Dalini, S. J. Thorpe, T. Masquelier, Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks, Pattern Recognition 94 (2019) 87–95.

[15] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, A. Rabinovich, Going deeper with convolutions, in: 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1–9.

[16] K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[17] W. Fang, Z. Yu, Y. Chen, T. Huang, T. Masquelier, Y. Tian, Deep Residual Learning in Spiking Neural Networks, Advances in Neural Information Processing Systems 34 (2021) 21056–21069.

[18] J. J. Atick, A. N. Redlich, What Does the Retina Know about Natural Scenes?, Neural Computation 4 (1992) 196–210.

[19] X. Pitkow, M. Meister, Decorrelation and efficient coding by retinal ganglion cells, Nature Neuroscience 15 (2012) 628–635.

[20] T. Ichinose, S. Habib, On and off signaling pathways in the retina and the visual system, Frontiers in Ophthalmology 2 (2022).

[21] M. Carandini, D. J. Heeger, Normalization as a canonical neural computation, Nature reviews. Neuroscience 13 (2011) 51–62.

[22] R. V. Rullen, S. J. Thorpe, Rate Coding Versus Temporal Order Coding: What the Retinal Ganglion Cells Tell the Visual Cortex, Neural Computation 13 (2001) 1255–1283.

[23] T. Masquelier, S. J. Thorpe, Unsupervised Learning of Visual Features through Spike Timing Dependent Plasticity, PLOS Computational Biology 3 (2007) e31.

[24] P. U. Diehl, M. Cook, Unsupervised learning of digit recognition using spike-timing-dependent plasticity, Frontiers in Computational Neuroscience 9 (2015).

[25] A. Safa, I. Ocket, A. Bourdoux, H. Sahli, F. Catthoor, G. G. Gielen, Event Camera Data Classification Using Spiking Networks with Spike-Timing-Dependent Plasticity, in: 2022 International Joint Conference on Neural Networks (IJCNN), 2022, pp. 1–8.

[26] P. Falez, P. Tirilly, I. M. Bilasco, P. Devienne, P. Boulet, Unsupervised visual feature learning with spike-timingdependent plasticity: How far are we from traditional feature learning approaches?, Pattern Recognition 93 (2019) 418–429.

[27] S. R. Kheradpisheh, M. Ganjtabesh, S. J. Thorpe, T. Masquelier, STDP-based spiking deep convolutional neura networks for object recognition, Neural Networks 99 (2018) 56–67.

[28] M. C. W. v. Rossum, G. Q. Bi, G. G. Turrigiano, Stable Hebbian Learning from Spike Timing-Dependent Plasticity, Journal of Neuroscience 20 (2000) 8812–8821.

[29] R. P. N. Rao, T. J. Sejnowski, Spike-Timing-Dependent Hebbian Plasticity as Temporal Difference Learning, Neural Computation 13 (2001) 2221–2237.

[30] D. Kappel, B. Nessler, W. Maass, STDP Installs in Winner-Take-All Circuits an Online Approximation to Hidden Markov Model Learning, PLOS Computational Biology 10 (2014) e1003511.

[31] E. M. Izhikevich, Solving the distal reward problem through linkage of STDP and dopamine signaling, BMC Neuroscience 8 (2007) S15.

[32] R. Legenstein, D. Pecevski, W. Maass, A Learning Theory for Reward-Modulated Spike-Timing-Dependent Plasticity with Application to Biofeedback, PLOS Computational Biology 4 (2008) e1000180.

[33] Ł. Kusmierz, T. Isomura, T. Toyoizumi, Learning with three factors: modulating Hebbian plasticity with errors,´ Current Opinion in Neurobiology 46 (2017) 170–177.

[34] G. Orchard, A. Jayawant, G. K. Cohen, N. Thakor, Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades, Frontiers in Neuroscience 9 (2015) 437.

[35] A. Patiño-Saucedo, H. Rostro-González, T. Serrano-Gotarredona, B. Linares-Barranco, Liquid State Machine on SpiNNaker for Spatio-Temporal Classification Tasks, Frontiers in Neuroscience 16 (2022).

[36] I. J. Tsang, F. Corradi, M. Sifalakis, W. Van Leekwijck, S. Latré, Radar-Based Hand Gesture Recognition Using Spiking Neural Networks, Electronics 10 (2021) 1405.

[37] H. B. Barlow, Possible Principles Underlying the Transformations of Sensory Messages, in: W. A. Rosenblith (Ed.), Sensory Communication, The MIT Press, 2012, p. 0.

[38] D. J. Heeger, Normalization of cell responses in cat striate cortex, Visual Neuroscience 9 (1992) 181–197.

[39] S. Thorpe, D. Fize, C. Marlot, Speed of processing in the human visual system, Nature 381 (1996) 520–522.

[40] Y. Hao, X. Huang, M. Dong, B. Xu, A biologically plausible supervised learning method for spiking neural networks using the symmetric STDP rule, Neural Networks 121 (2020) 387–395.

[41] G. Goupy, P. Tirilly, I. M. Bilasco, Paired competing neurons improving STDP supervised local learning in spiking neural networks, Frontiers in Neuroscience 18 (2024).

## A Reproducibility Details

Tables 6-8 report the numerical settings of the main four-layer models. Static images and event streams use different deterministic encoders, after which the same layer-wise backbone, sparse MDTF module, and temporal R-STDP readout are applied. In the following tables, $k / s / p$ denotes kernel size, stride, and padding; $\mathrm { T o p } { - } k _ { \mathrm { m a p } }$ and ${ \mathrm { T o p } } { \cdot } k _ { \mathrm { f u s } }$ denote the map-wise and fusion sparsity caps, respectively; Ep. denotes the number of training epochs.

Table 6: Front-end and shared optimization settings. The whitening stride is used only to sample fitting patches; the learned whitening operator is applied densely with unit stride. Static calibration is fitted on the training split and held fixed thereafter.
<table><tr><td rowspan=1 colspan=10>Component       Parameter                MNIST/Fashion-MNIST        CIFAR-10                    N-MNIST</td></tr><tr><td rowspan=16 colspan=10>Input source       Raw data                 grayscale images               RGB images                   polarity event streamRepresentation             calibrated local-decorrelation polar- calibrated local-decorrelation polar-event-map TTFSity TTFS                     ity TTFSOutput channels            2                           6                           10 $7 , 2 , 1 0 ^ { - 2 }$                     $9 , 2 , 1 0 ^ { - 2 }$  $1 . 0 , 1 0 ^ { 6 } , 6 4$                    $1 . 0 , 1 0 ^ { 6 } , 6 4$ positive/negative; training-set min- positive/negative; training-set min- native ON/OFF channelsmax per channel and locationpatch, α, loser factor, mixed- 5, .15, .25, [.2, .7]             disabledratio intervalpatch, dominant compres- 4, .10/.15, .10                disabledpatch, α, loser factor, negative 6, .10, .25, .15, .10; energy gate6, .10, .25, .15, .10; energy gate –disabled</td></tr><tr><td rowspan=1 colspan=1>Latency encoder</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Latency encoder</td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>Local decorrelation</td><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Local decorrelation</td><td rowspan=1 colspan=1>PCA component fraction, max.</td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>patches, fit batch</td></tr><tr><td rowspan=1 colspan=1>Polarity calibration</td><td rowspan=1 colspan=1>split and scaling</td></tr><tr><td rowspan=2 colspan=1>Signed-context gate</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>max per channel and location</td></tr><tr><td rowspan=1 colspan=1>patch, α, loser factor, mixed-</td><td rowspan=1 colspan=1>5, .15, .25, [.2, .7]</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Polarity reweighting</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>sion/weak boost, margin</td></tr><tr><td rowspan=1 colspan=1>Pair compensation</td><td rowspan=1 colspan=1>patch, α, loser factor, negative</td><td rowspan=1 colspan=1>6, .10, .25, .15, .10; energy gate</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>bias, margin</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Event map</td><td rowspan=1 colspan=1>temporal bins and denoising</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1>5 × 2 polarities; 10 000 µs</td></tr><tr><td rowspan=1 colspan=1>Event map</td><td rowspan=1 colspan=1>response normalization</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1>log1p; local radius 2, € =</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ sample-maximum nor-</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=1>malization; γ = 1</td></tr><tr><td rowspan=1 colspan=1>Convolutional</td><td rowspan=1 colspan=1>weight bounds and target tim</td><td rowspan=1 colspan=1>e [0, 1], .95</td><td rowspan=1 colspan=6>[0, 1], .95</td><td rowspan=1 colspan=1>[0, 1], .95</td></tr><tr><td rowspan=2 colspan=1> $S _ { 1 }$ threshold sched-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>initial rate, minimum, anneaing</td><td rowspan=1 colspan=1>l- 1, 2, .95</td><td rowspan=1 colspan=6>1, 2, .95</td><td rowspan=1 colspan=1>0, .2, .95</td></tr><tr><td rowspan=1 colspan=1>S2 threshold sched-</td><td rowspan=1 colspan=1>initial rate, minimum, annea</td><td rowspan=1 colspan=1>l- 1, 4, .95</td><td rowspan=2 colspan=6>1,4, .95</td><td rowspan=2 colspan=1>1,4, .95</td></tr><tr><td rowspan=1 colspan=1>ule</td><td></td><td></td></tr><tr><td rowspan=2 colspan=1> $S _ { 3 : 4 }$  threshold</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>inginitial rate, minimum, anneal</td><td rowspan=1 colspan=1>- 0, 0, 1</td><td rowspan=1 colspan=6>0,0,1</td><td rowspan=1 colspan=1>0,0,1</td></tr><tr><td rowspan=1 colspan=1>schedule</td><td rowspan=1 colspan=1>ing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Weight initialization</td><td rowspan=2 colspan=1> $S _ { 1 } ; S _ { 2 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>normal .5 ± .01; normal .3 ± .01</td><td rowspan=1 colspan=6>normal .5 ± .01; normal .3 ± .01</td><td rowspan=1 colspan=1>normal .5 ± .01; normal.3 ±</td></tr><tr><td rowspan=1 colspan=1>Weight initialization</td><td rowspan=1 colspan=1> $S _ { 3 } ; S _ { 4 }$ </td><td rowspan=1 colspan=1>identity-centered, $g = . 4 5 , \sigma =$ </td><td rowspan=1 colspan=6>identity-centered, $g = . 4 5 , \sigma =$ </td><td rowspan=3 colspan=1>.01identity-centered, g = .45,σ = .01.3, .01, [0, 1]; fixed l1 norm</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>.01</td><td rowspan=1 colspan=6>.01</td></tr><tr><td rowspan=1 colspan=1>Readout initializa-</td><td rowspan=1 colspan=1>mean, std., bounds; normaliza</td><td rowspan=1 colspan=1>- .3, .01, [0, 1]; fixed l1 norm</td><td rowspan=1 colspan=6>.3, .01, [0, 1]; fixed $\ell _ { 1 }$ norm</td></tr><tr><td rowspan=1 colspan=1>tion</td><td rowspan=1 colspan=1>tion</td><td rowspan=3 colspan=1>5, .5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Readout regularizer</td><td rowspan=2 colspan=1>threshold rate and annealing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=6>5, .5</td><td rowspan=1 colspan=1>5,.5</td></tr><tr><td rowspan=1 colspan=1>Readout stability</td><td rowspan=1 colspan=1>minimum active output neu- 36</td><td rowspan=2 colspan=8>36</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>rons</td></tr></table>

Table 7: Backbone and fusion parameters for the four-layer locally trained spiking pipeline. The output column reports the latency representation consumed by the following stage. The $C _ { 2 }$ representations were exported with pooling kernel/stride $\bar { 2 / 1 }$ . The deep rows describe the adaptive path used by the final model.
<table><tr><td>Block</td><td>Setting</td><td>Input</td><td>Output</td><td>k/s/p or op</td><td> $\theta _ { 0 } { \mathbf o r } m _ { \mathrm { a g r e e } }$ </td><td> $\mathrm { { \bf T o p } } { \cdot } k _ { \mathrm { m a p / f u s } }$ </td><td>g, σ</td><td> $A ^ { + } / A ^ { - } / \beta$ </td><td> $\mathbf { E p . }$ </td></tr><tr><td> $S _ { 1 } { + } C _ { 1 }$ </td><td>MNIST/Fashion-MNIST 2× 28 × 28</td><td></td><td> $1 2 8 \times 6 \times 6$ </td><td> $5 / 1 / 0 ; \mathrm { p o o l } 4 / 4$ </td><td>5.0</td><td>0</td><td></td><td>.100/.100/1.008</td><td></td></tr><tr><td> $S _ { 1 } { + } C _ { 1 }$ </td><td>CIFAR-10</td><td>6×32×32</td><td> $1 2 8 \times 7 \times 7$ </td><td> $5 / 1 \dot { / } 0 ; \bar { \mathrm { p o o l } } 4 \dot { / } 4$ </td><td>10.0</td><td>0</td><td></td><td>.100/.100/1.00 18</td><td></td></tr><tr><td> $S _ { 1 } { + } C _ { 1 }$ </td><td>N-MNIST</td><td>10×34×34</td><td> $1 2 8 \times 7 \times 7$ </td><td> $5 / 1 \dot { / } 0 ; \bar { \mathrm { p o o l } } 4 \dot { / } 4$ </td><td>4.0</td><td>0</td><td></td><td> $\cdot 1 0 0 ^ { ' } / . 0 8 0 ^ { ' } / . 8 5 \quad 2$ </td><td></td></tr><tr><td> $S _ { 2 } { + } C _ { 2 }$ </td><td>MNIST/Fashion-MNIST 128× 6×6</td><td></td><td>256×5×5</td><td> $1 / 1 / 0 ; \mathrm { p o o l } 2 / 1$ </td><td>24.0</td><td>32</td><td></td><td>.100/.100/1.003</td><td></td></tr><tr><td> $S _ { 2 } { + } C _ { 2 }$ </td><td>CIFAR-10</td><td>128×7×7</td><td>256×6×6</td><td> $1 { \dot { / } } 1 { \dot { / } } 0 ; { \bar { \mathsf { p o o l } } } 2 { \dot { / } } 1$ </td><td>24.0</td><td>32</td><td></td><td>.100/.100/1.00 3</td><td></td></tr><tr><td> $S _ { 2 } { + } C _ { 2 }$ </td><td>N-MNIST</td><td>128×7×7</td><td>256×6×6</td><td> $1 \dot { / } 1 \dot { / } 0 ; \bar { \mathrm { p o o l } } 2 \dot { / } 1$ </td><td>8.0</td><td>32</td><td></td><td>.100/.100/1.00 2</td><td></td></tr><tr><td>S3</td><td>adaptive deep path</td><td>256×H×W</td><td>256×H×W</td><td>1/1/0</td><td>0.5</td><td>64</td><td></td><td>.45,.010.003/.003/.85</td><td>2</td></tr><tr><td>S4</td><td>adaptive deep path</td><td>256×H×W</td><td>256×H×W</td><td>1/1/0</td><td>0.5</td><td>64</td><td></td><td>.45, .010 .003/.003/.85</td><td>2</td></tr><tr><td>Fusion</td><td></td><td></td><td> $d _ { \mathrm { f e a t } } = 1 7 4 0 8$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fusion</td><td>CIFAR-10</td><td>MNIST/Fashion-MNIST P :128×6×6 + 2×256×5×5  $P : 1 2 8 \times 7 \times 7 + 2 \times 2 5 6 \times 6 \times 6$ </td><td> $d _ { \mathrm { f e a t } } = 2 4 7 0 4$ </td><td>[P, ∆res, ∆agree] [P, ∆res, ∆agree]</td><td>magree magree</td><td>= .001 128/16 = .001 128/16</td><td></td><td></td><td></td></tr><tr><td>Fusion</td><td>N-MNIST</td><td> $P : 1 2 8 \times 7 \times 7 + 2 \times 2 5 6 \times 6 \times 6$ </td><td> $d _ { \mathrm { f e a t } } = 2 4 7 0 4$ </td><td>[P, ∆res, ∆agree]</td><td>magree 二</td><td>.001 128/16</td><td></td><td></td><td></td></tr></table>

Table 8: Temporal R-STDP readout operating points for the main reported models. All rows use additive STDP, fixed perneuron $\ell _ { 1 }$ weight normalization, global winner-take-all decoding by output spike time, $\tau _ { \mathrm { m a x } } = 1 , m _ { \mathrm { r o } } = . 0 0 5$ , prototype decay .5, target scale 2, floor 0, target guard 1.25, anti-guard 0, and learning-rate annealing .75. “Updated prototypes” gives the number selected from the target population and from each selected hard-negative class, respectively.
<table><tr><td>Dataset</td><td> $d _ { \mathrm { f e a t } }$ </td><td> $N _ { \mathrm { o u t } }$ </td><td>Proto./class</td><td> $\theta _ { 0 }$ </td><td></td><td>K Updated proto. Non-target scale</td><td> $\lambda _ { \mathrm { m a x } }$ </td><td>Correct scale</td></tr><tr><td>MNIST</td><td>17408</td><td>80</td><td></td><td>8 120</td><td>7</td><td>1/3</td><td>0.25 0.0035</td><td>0.0075</td></tr><tr><td>Fashion-MNIST 17 408</td><td></td><td>40</td><td></td><td>4 223</td><td>5</td><td>1/2</td><td>0.32 0.0050</td><td>0.1000</td></tr><tr><td>CIFAR-10</td><td>24 704</td><td>40</td><td></td><td>4400</td><td>5</td><td>1/2</td><td>0.32 0.0050</td><td>0.1000</td></tr><tr><td>N-MNIST</td><td>24 704</td><td>60</td><td></td><td>6 243</td><td>9</td><td>1/3</td><td>0.35 0.0030</td><td>0.0050</td></tr></table>