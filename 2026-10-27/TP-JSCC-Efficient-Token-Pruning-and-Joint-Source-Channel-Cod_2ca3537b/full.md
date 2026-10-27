# TP-JSCC: Efficient Token Pruning and Joint Source-Channel Coding for LVLMs-based 3D-VQA in Cloud-Edge Collaboration

Zhengxin Li, Chongzhen Tian, Hui Yuan, Senior Member, IEEE, Hao Liu, and Sam Kwong, Fellow, IEEE

Abstract—Deploying large vision-language models (LVLMs) on resource-constrained embodied agents for 3D visual question answering necessitates cloud-edge collaboration. However, offloading massive visual data to the cloud over bandwidth-limited and noisy channels introduces severe latency and transmission distortion. To address this, we propose TP-JSCC, an efficient token pruning and joint source-channel coding method for LVLMs-based 3D-VQA in Cloud-Edge Collaboration. First, we develop a Local and Global Semantics-based Hybrid Token Pruning (LGS-HTP) module to significantly reduce the visual token count. By decoupling foreground-background tokens and jointly selecting local and global semantics, LGS-HTP effectively reduces redundancy while preserving reasoning-critical tokens. Second, we introduce a lightweight Token Joint Source-Channel Coding (T-JSCC) module to further compress the pruned tokens and ensure their robust transmission over error-prone channels. By employing lightweight cascaded linear layers and Signal-to-Noise Ratio (SNR)-adaptive mechanism, this module adapts to fluctuating channel conditions while maintaining minimal computational overhead. Extensive experiments on ScanQA and SQA3D datasets indicate the exceptional efficiency and robustness of our approach. Specifically, on ScanQA validation set, the LGS-HTP module could achieve 90% pruning ratio while incurring less than a 1% CIDEr loss. Building on this, the T-JSCC further compresses the tokens by 18×, resulting in an overall 180× data volume reduction for robust transmission, while saving 90% in computational complexity for LVLM deployment. Our code is available at https://github.com/yuanhui0325/TP-JSCC.

Index Terms—Token Compression, Joint Source-Channel Coding, Large Vision-Language Models, 3D Visual Question Answering, Embodied Agents.

## I. INTRODUCTION

L <sup>ARGE</sup> <sup>Vision-Language</sup> <sup>Models</sup> <sup>(LVLMs)</sup> <sup>have</sup> <sup>achieved</sup>unprecedented advancements, driven by an exponential unprecedented advancements, driven by an exponential increase in computational power and the surge of largescale datasets. These models exhibit excellent capabilities in multimodal understanding and reasoning, significantly enhancing the cognitive competence of embodied agents (EAs) [1]. Particularly in unstructured environments requiring complex reasoning, such as 3D Visual Question Answering (3D-VQA), conventional lightweight models are limited by shallow semantic representations and often fail to capture robust cross-modal correspondences. In contrast, LVLMs leverage extensive prior knowledge to achieve superior cross-modal alignment [2]. Consequently, integrating LVLMs into EAs is an inevitable trend toward general-purpose embodied intelligence.

However, the remarkable capabilities of LVLMs come at the cost of substantial computational overhead. Most EAs, such as intelligent drones and domestic service robots, are subject to strict physical constraints, including size, thermal limits, and battery power, rendering on-device LVLM inference computationally prohibitive. Therefore, it is necessary to develop a Cloud-Edge Collaborative Paradigm (CE-CP) for EAs [3]. In this paradigm, EAs function as edge terminals focusing on data acquisition, compression, and transmission, while resourceabundant cloud servers execute the LVLM inference.

Although CE-CP avoids the on-device computational bottlenecks, it shifts the heavy burden to the communication channels. In error-prone communication environments, such as post-disaster rescue and deep-well exploration, communication channels are typically plagued by severe channel noise and strictly limited bandwidth. Under such conditions, directly transmitting raw high-resolution visual data becomes intractable. Specifically, the massive volume of raw pixels incurs unacceptable latency, while their inherent vulnerability to noise leads to severe transmission distortion, degrading the inference performance of LVLMs. Therefore, robust Joint Source-Channel Coding (JSCC) paradigms are necessary to ensure reliable transmission.

However, before designing the coding scheme, a fundamental question must be addressed: whether to compress raw visual pixels or their higher-level semantic representations (i.e., visual tokens). Compressing visual tokens offers three distinct advantages. First, visual encoders intrinsically filter out pixel-level noise and spatial redundancy, yielding a highly compact representation that optimizes bandwidth utilization at the source. Second, despite being distilled features, tokens retain structured correlations within the semantic space, unlocking the potential for deep compression far beyond the theoretical limits of pixel-level coding. Third, as latent representations, visual tokens are highly non-invertible, which substantially mitigates the risk of privacy leakage. Therefore, from the perspectives of both communication efficiency and privacy preservation, transmitting compressed visual tokens emerges as a highly effective paradigm [4]–[7].

To fully exploit the transmission potential of visual tokens over bandwidth-limited and error-prone communication channels, we propose an efficient token pruning and JSCC method for LVLM-based 3D-VQA in cloud-edge collaboration, named

TP-JSCC. It effectively tackles the core bottlenecks of redundancy and distortion through two collaborative modules. First, to fundamentally eliminate token-level redundancy, we propose a Local and Global Semantics-based Hybrid Token Pruning (LGS-HTP) module. By explicitly decoupling foreground from background tokens and jointly preserving local and global semantics, this module effectively mitigates the severe performance degradation of LVLM typically induced by massive pruning. Second, to further mitigate feature-level redundancy within the pruned tokens and enhance transmission robustness, we propose a Token-based Joint Source-Channel Coding (T-JSCC) module. Constrained by the limited computational capacity of EAs, T-JSCC adopts an all-linear cascaded architecture to ensure minimal on-device overhead. Furthermore, it incorporates a Signal-to-Noise Ratio (SNR) adaptive mechanism to dynamically mitigate the impact of fluctuating channel noise, ensuring robust token reconstruction at the cloud server. The main contributions of this paper are summarized as follows.

1) TP-JSCC method: To address LVLM inference demands under on-device computational constraints and severely noisy, low-bandwidth communication channels, we propose an efficient token pruning and JSCC method. The edge device handles token compression, while the cloud server performs decoding and complex LVLM inference, effectively offloading the computational burden away from the edge.

2) LGS-HTP module: To tackle the issue of token redundancy, we propose the LGS-HTP module which first extracts foreground-background priors to decouple tokens, and then jointly selects local semantic tokens and global semantic tokens to perform pruning. It removes redundant tokens while preserving the reasoning-critical semantics.

3) T-JSCC module: To further reduce token redundancy and mitigate severe transmission distortion of pruned tokens over error-prone communication channels, we design a lightweight T-JSCC module based on an all-linear cascaded projection. Furthermore, we incorporate an SNR-adaptive mechanism to enable channel-aware transmission, ensuring high robustness against varying channel conditions.

The remainder of this paper is organized as follows. In Section II, we introduce related work on 3D VQA, visual token pruning, and deep joint source-channel coding briefly. In Section III, we systematically introduce the framework of TP-JSCC and mathematically formulate the token pruning and coding problem, based on which we present the proposed TP-JSCC in detail in Section IV. Experimental results and conclusions are given in Sections V and VI, respectively.

## II. RELATED WORK

## A. 3D-VQA

Existing 3D-VQA methods can be broadly categorized into three paradigms according to the representation modality of the input data: point-cloud-based methods [8]–[11], hybrid methods combining point clouds and multi-view images [12]– [15], and multi-view-image-based methods [16]–[19]. Specifically, point-cloud and hybrid methods typically employ 3D encoders to extract high-dimensional geometric features for cross-modal fusion. However, their overall performance is severely bottlenecked by the scale and annotation quality of 3D training datasets. For instance, direct 3D point-cloud training is data-starved, as the point-cloud modality in existing highquality datasets such as ScanQA [8] contain only about 41,000 point-cloud-text pairs, which is orders of magnitude smaller than 2D datasets (e.g., LAION [20] includes approximately 5.8 billion image-text pairs). Consequently, these models often struggle to establish robust cross-modal alignment, thereby constraining their generalization capabilities.

In contrast, multi-view-image-based methods directly process 2D image sequences and leverage the powerful zeroshot capabilities of pretrained LVLMs to perform 3D-VQA reasoning, effectively bypassing the point-cloud-text alignment bottleneck. A representative pipeline, LLaVA-OV [18], utilizes the frozen SigLIP [21] image encoder to extract visual tokens and feeds them alongside text tokens into a pretrained Qwen [22] model. Motivated by this architecture, cdViews [19] introduces a lightweight attention-based view selector to replace uniform multi-view sampling, filtering for semantically relevant views to improve reasoning.

Given that frequently transmitting all multi-view visual tokens in a cloud-edge collaborative architecture inevitably introduces substantial question-irrelevant redundancy and severely degrades communication efficiency, we adopt the novel multiview paradigm (cdViews + LLaVA-OV) as our base architecture. Utilizing view selection as a front-end filter, we construct a robust token compression framework upon this foundation to resolve the inherent transmission bottleneck.

## B. Visual Token Pruning

Although LVLMs have significantly advanced 3D-VQA, their superior capability intrinsically relies on processing a massive number of visual tokens. The direct transmission of these tokens which are semantically redundant [23], degrades communication efficiency while simultaneously imposing an unnecessary computational burden on cloud servers [24]. Consequently, visual token pruning has emerged as an indispensable strategy to jointly minimize both communication payload and LVLM computation overhead.

Existing token pruning approaches can be classified into text-dependent [25]–[28] and text-independent [29]–[35] categories. Text-dependent pruning utilizes cross-modal attention mechanisms to measure semantic alignment between visual tokens and textual queries. Although such methods theoretically preserve more task-relevant features, they face a severe operational trade-off: lightweight alignment networks often fail to establish accurate cross-modal connections, increasing the risk of irreversible removal of critical tokens [36]; conversely, employing heavy neural networks to enhance alignment precision incurs prohibitive computational overhead [26], fundamentally contradicting the objective of pruning.

In contrast, text-independent methods aim to preserve maximum semantic information by assessing token importance via intrinsic statistical properties [29]. Being predominantly training-free [29]–[35], these methods avoid the ambiguity of supervision labels in weakly defined pruning tasks [28] and provide strong generalization across diverse applications. Motivated by these advantages, we adopt a text-independent strategy to optimize the balance between compression efficiency and deployment generality.

![](images/7433c51bd682a952c5b5cb9cc12ff9b9553a28179c8c1262ea30606e9e11b3ce.jpg)

![](images/4a1d519e85b4e057c1da775da9c208f3b3386734d5d5be994a913426104a1a8b.jpg)  
Fig. 1. Framework of the proposed Token Pruning Joint Source-Channel Coding (TP-JSCC). The LGS-HTP module effectively selects global semantic tokens to reduce visual redundancy and achieve a scalable compression ratio. Simultaneously, the T-JSCC module encodes and transmits the selected sparse token to maintain transmission robust to complex wireless channel conditions.

## C. Deep Joint Source-Channel Coding

Conventional communication systems founded upon Shannon’s separation principle often suffer from the ‘cliff effect’ in low-latency and finite-blocklength regimes, wherein a slight degradation in channel quality causes an abrupt performance collapse [37]–[40]. In contrast, deep-learning-based JSCC performs end-to-end joint optimization. This paradigm demonstrates superior robustness to variations in communication channel conditions and exhibits graceful rather than catastrophic performance degradation, perfectly aligning with highly dynamic practical error-prone communication channels [38], [41].

Existing deep JSCC methods are broadly categorized into human-perception oriented [38], [42], [43] and machineperception oriented [44], [45] approaches. The former maximizes high-fidelity pixel reconstruction (evaluated by metrics like Peak Signal-to-Noise Ratio, i.e., PSNR), which inherently preserves a massive amount of task-irrelevant information [44]. Therefore, machine-perception JSCC aligns better with vision tasks by directly transmitting essential semantic features optimized for tasks.

Nevertheless, applying existing machine-perception JSCC to our cloud-edge LVLM framework faces two obstacles. First, there is a mismatch in representational modality: most existing methods focus on compressing CNN feature maps [44]–[47], and are not intrinsically designed for the discrete token sequences required by LVLMs. Second, there is a critical mismatch in computational constraints: although a few tokenbased JSCC schemes have recently emerged [4]–[7], they heavily rely on computationally intensive Transformer architectures and do not take the intra and inter token redundancy into account. Such Transformer-based designs impose unacceptable memory, communication, and computation burdens on compute-sensitive EAs in CE-CP. This precisely motivates our design of a highly streamlined, Multi-Layer Perceptron (MLP)-based Lightweight Token-JSCC method tailored for robust token transmission under strict edge constraints.

## III. PROBLEM FORMULATION

The overall architecture of our proposed TP-JSCC is illustrated in Fig. 1. Formally, we define the 3D-VQA task as follows. Let q denote the question and S denote the 3D scene. We define $\mathcal { V } _ { \mathrm { a } } = \{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { a } } }$ as the set of $N _ { \mathrm { a } }$ view images captured from $s ,$ , where $\mathbf { I } _ { i }$ represents the image from the i-th view. Given ${ \mathbf { } } q ,$ a view selector $\mathcal { F } _ { \mathrm { c d v i e w s } } ( \cdot )$ is used to choose the most informative subset of views:

$$
\mathcal { V } _ { \mathrm { s } } = \mathcal { F } _ { \mathrm { c d v i e w s } } ( \mathcal { V } _ { \mathrm { a } } , \pmb { q } ) = \{ \mathbf { I } _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { s } } } ,\tag{1}
$$

where $N _ { \mathrm { s } }$ is the number of selected views. We then employ the SigLIP [21] image encoder $\mathcal { F } _ { \mathrm { e n c } } ^ { \mathrm { v i s } } ( \cdot )$ to extract multi-view native visual tokens:

$$
\mathcal { T } _ { \mathrm { N } } = \mathcal { F } _ { \mathrm { e n c } } ^ { \mathrm { v i s } } ( \mathcal { V } _ { \mathrm { s } } ) = \{ \mathbf { T } _ { \mathrm { N } , i } \} _ { i = 1 } ^ { N _ { \mathrm { s } } } ,\tag{2}
$$

where $\mathbf { T } _ { \mathrm { N } , i } \in \mathbb { R } ^ { N _ { \mathrm { p } } \times D _ { \mathrm { N } } } , N _ { \mathrm { p } }$ denotes the number of tokens per view, and $D _ { \mathrm { N } }$ is the output feature dimension of $\mathcal { F } _ { \mathrm { e n c } } ^ { \mathrm { v i s } } ( \cdot )$ . Next, a projector composed of linear layers, denoted by $\mathcal { F } _ { \mathrm { p r o j } } ( \cdot )$ , is applied to obtain the projected visual tokens for the LVLM:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { P } } = \mathcal { F } _ { \mathrm { p r o j } } ( \mathcal { T } _ { \mathrm { N } } ) = \{ \mathbf { T } _ { \mathrm { P } , i } \} _ { i = 1 } ^ { N _ { \mathrm { s } } } , } \end{array}\tag{3}
$$

where $\mathbf { T } _ { \mathrm { P } , i } \in \mathbb { R } ^ { N _ { \mathrm { P } } \times D _ { \mathrm { P } } }$ and $D _ { \mathrm { P } }$ is the hidden dimension of the LVLM. Meanwhile, the text encoder $\mathcal { F } _ { \mathrm { e n c } } ^ { \mathrm { t x t } } ( \cdot )$ encodes q into text tokens $\mathbf { T } _ { \mathrm { t x t } } \in \mathbb { R } ^ { N _ { \mathrm { t x t } } \times D _ { \mathrm { P } } }$ , where $N _ { \mathrm { t x t } }$ denotes the number of text tokens. Finally, the integrated multimodal inputs are fed into the LVLM $\mathcal { L } _ { \mathrm { V L M } } ( \cdot )$ , followed by the text decoder $\mathcal { F } _ { \mathrm { d e c } } ^ { \mathrm { t x t } } ( \cdot )$ to generate the answer r:

$$
r = \mathcal { F } _ { \mathrm { d e c } } ^ { \mathrm { t x t } } ( \mathcal { L } _ { \mathrm { V L M } } ( \mathbf { T } _ { \mathrm { t x t } } , \mathcal { T } _ { \mathrm { P } } ) ) .\tag{4}
$$

However, due to the extremely large parameter size of $\mathcal { L } _ { \mathrm { V L M } } ( \cdot )$ , on-device deployment by EAs is unaffordable. This motivates us to adopt CE-CP [3] to address the problem: the edge device (i.e., EAs) transmits $\mathbf { T } _ { \mathrm { t x t } }$ and $\mathcal { T } _ { \mathrm { N } }$ (a transmissionfriendly, more compact representation, which can be easily projected to $\mathcal { T } _ { \mathrm { P } } )$ to the cloud server over wireless channels, and the cloud server executes $\mathcal { L } _ { \mathrm { V L M } } ( \cdot )$ to generate the answer.

We assume $\mathbf { T } _ { \mathrm { t x t } }$ is delivered through a reliable error-free channel $\mathcal { C } _ { \mathrm { L o s s l e s s } } ( \cdot )$ due to its compact size and critical role in reasoning. This assumption is motivated by the widely used Unequal Error Protection (UEP) concept in communications [48], where key information is protected while large data allows for lossy transmission. In contrast, the much larger $\mathcal { T } _ { \mathrm { N } }$ must be transmitted through a lossy channel $\mathcal { C } _ { \mathrm { L o s s y } } ( \cdot )$ . Since $\mathcal { C } _ { \mathrm { L o s s y } } ( \cdot )$ is bandwidth-limited and noisy, direct transmission not only incurs excessive latency but also leads to severe 3D-VQA performance degradation.

To address these challenges, we propose TP-JSCC, a twostep approach consisting of pruning and JSCC. Specifically, we first utilize $\mathcal { T } _ { \mathrm { P } }$ and the semantic priors in $\nu _ { \mathrm { s } }$ to guide the pruning of $\mathcal { T } _ { \mathrm { N } }$ , obtaining the pruned tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ :

$$
\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } = \mathcal { F } _ { \mathrm { T P } } ( \mathcal { T } _ { \mathrm { N } } , \mathcal { T } _ { \mathrm { P } } , \mathcal { V } _ { \mathrm { s } } , R _ { \mathrm { p r } } ) ,\tag{5}
$$

where $R _ { \mathrm { p r } }$ denotes the pruning ratio, and $\mathcal { F } _ { \mathrm { T P } } ( \cdot )$ denotes the pruning algorithm. As $\overset { \cdot } { \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } }$ is vulnerable to channel noise, we employ a JSCC encoder $\mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { e n c } }$ and decoder $\mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { d e c } } ,$ to further compress and transmit the pruned tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ in the edge device, reconstructing them into $\widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } }$ at the cloud server, which are then projected to $\widehat { T } _ { \mathrm { P } } ^ { \mathrm { T P } }$ (the input tokens of the ${ \mathcal { L } } _ { \mathrm { V L M } } ) $

$$
\begin{array} { r } { \widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } } = \mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { d e c } } \left( \mathcal { C } _ { \mathrm { L o s s y } } \left( \mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { e n c } } ( \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } | \boldsymbol { \Theta } _ { \mathrm { e n c } } ) \right) | \boldsymbol { \Theta } _ { \mathrm { d e c } } \right) , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { \widehat { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } } = \mathcal { F } _ { \mathrm { p r o j } } ( \widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } } ) . } \end{array}\tag{7}
$$

where $\Theta _ { \mathrm { e n c } }$ and $\Theta _ { \mathrm { d e c } }$ denote the learnable parameters of the JSCC encoder and decoder.

The goal of pruning is to find the optimal $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P * } }$ that could be projected to $\bar { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } }$ by minimizing the Kullback-Leibler (KL) divergence $\mathcal { D } _ { \mathrm { K I } }$ between the output distributions of ${ \mathcal { L } } _ { \mathrm { V L M } }$ under the original input $( \mathcal { T } _ { \mathrm { P } } )$ and the pruned input $( \mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } } )$

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P * } } = \underset { | \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } | \leq K } { \arg \operatorname* { m i n } } \mathcal { D } _ { \mathrm { K L } } \big ( \mathcal { L } _ { \mathrm { V L M } } ( \mathbf { X } _ { 1 } ) \ \| \ \mathcal { L } _ { \mathrm { V L M } } ( \mathbf { X } _ { 2 } ) \big ) , } \end{array}\tag{8}
$$

where $\mathbf { X } _ { 1 } = ( \mathbf { T } _ { \mathrm { t x t } } , \mathcal { T } _ { \mathrm { P } } )$ and $\mathbf { X } _ { 2 } = ( \mathbf { T } _ { \mathrm { t x t } } , \mathcal { F } _ { \mathrm { p r o j } } ( \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } ) ) , \mid \cdot \mid$ denotes the token counting, and $K = N _ { \mathrm { s } } \times N _ { \mathrm { p } } \times ( 1 - R _ { \mathrm { p r } } )$ is the total token budget. Although trainable pruning methods can directly optimize the objective in Eq. (8), the massive parameter scale of LVLM makes gradient backpropagation exceedingly difficult and incurs prohibitively high training costs. Therefore, we opt for a training-free token pruning approach, which approximates the objective in Eq. (8) by maximizing the semantic information retained in the pruned tokens. Specifically, cognitive science research reveals that when humans observe complex visual scenes, their perception inherently relies on integrating both local features and global context [49]. Inspired by this cognitive mechanism, we propose that the selected tokens should capture both the local semantics $( \mathrm { i . e . , }$ fine-grained details of objects) and the global semantics (i.e., the overall scene context). Therefore, we formulate the pruning process as finding the optimal tradeoff $( N _ { \mathrm { l o c } } ^ { * } , N _ { \mathrm { g l o } } ^ { * } )$ between the number of local semantic tokens $N _ { \mathrm { l o c } }$ and the number of global semantic tokens $N _ { \mathrm { g l o } }$ in $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ under a fixed pruning budget K:

$$
\begin{array} { r l r } & { ( N _ { \mathrm { l o c } } ^ { * } , N _ { \mathrm { g l o } } ^ { * } ) { = } \mathrm { a r g m a x ~ } \mathcal { U } \Big ( \mathcal { F } _ { \mathrm { p r o j } } \big ( \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P , l o c } } \cup \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P , g l o } } \big ) \Big ) , } & \\ & { \qquad \mathrm { s . t . ~ } | \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P , l o c } } | + | \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P , g l o } } | { = } N _ { \mathrm { l o c } } { + } N _ { \mathrm { g l o } } { = } K } & \end{array}\tag{9}
$$

where $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P , l o c } }$ and $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P , g l o } }$ denote the pruned local and global semantic tokens. Their union yields the pruned tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ Moreover, $\mathcal { U } ( \cdot )$ represents the task performance of ${ \mathcal { L } } _ { \mathrm { V L M } }$

The goal of JSCC is to further compress and robustly transmit the pruned tokens over noisy wireless channels, minimizing the distortion introduced by compression and transmission on the input features $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ . Specifically, the objective is to ensure that the reconstructed tokens $( \widehat { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ and $\widehat { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } } )$ remain as consistent as possible with the original tokens $( \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ and ${ \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } } )$ . To achieve this, we jointly optimize the learnable parameters of the JSCC encoder and decoder:

$$
\operatorname * { a r g m i n } _ { \Theta _ { \mathrm { e n c } } , \Theta _ { \mathrm { d e c } } } \Big ( \lambda _ { \mathrm { N } } \mathcal { D } ( \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } , \widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } } ) + \lambda _ { \mathrm { P } } \mathcal { D } ( \mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } } , \widehat { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } } ) \Big ) ,\tag{10}
$$

where $\mathcal { D } ( \cdot , \cdot )$ denotes a distance metric (e.g., Mean Squared Error (MSE)) to measure the semantic consistency, and $\lambda _ { \mathrm { N } } , \lambda _ { \mathrm { P } } > 0$ are balance parameters weighting the two losses.

## IV. PROPOSED METHOD

As illustrated in Fig. 1, building upon the CE-CP defined in Section III, our proposed method directly tackles the transmission bottleneck of massive visual tokens via a highly efficient two-stage strategy.

First, to fundamentally reduce the communication payload and LVLM computational overhead, we design the LGS-HTP module to achieve token-level compression. Guided by projected tokens $\mathcal { T } _ { \mathrm { P } }$ and priors from a lightweight segmentation network, it performs token pruning on the native tokens $\mathcal { T } _ { \mathrm { N } }$ to obtain a subset of tokens $\dot { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } }$ , while preserving both local and global semantics.

Second, to further compress the tokens and transmit them over a noisy communication channel, we introduce the T-JSCC module. At the edge device, it maps the pruned tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ into robust communication channel symbols. At the cloud server, it reconstructs these symbols into reconstructed visual tokens $\widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } }$ , which are then projected to ${ \widehat { T } } _ { \mathrm { P } } ^ { \mathrm { T P } }$ and aligned with the reliably transmitted textual tokens $\mathbf { T } _ { \mathrm { t x t } }$ to execute LVLM reasoning and generate the final answer r. The detailed designs of the LGS-HTP and T-JSCC modules are elaborated in Section IV-A and IV-B, respectively.

## A. Local and Global Semantics-based Hybrid Token Pruning

The overall pipeline of the LGS-HTP module is illustrated in Fig. 2. We first extract foreground-background priors (denoted as Pixel Mask in Fig. 2) using a lightweight segmentation network, thereby decoupling the original tokens into foreground and background parts. Accordingly, we then introduce a foreground enhancement factor $\lambda _ { \mathrm { b i a s } }$ to adjust the pruning budgets for foreground and background tokens, resulting in the per-view pruning budgets $( N _ { i } ^ { \mathrm { { F g } } } , N _ { i } ^ { \mathrm { { b g } } } )$ . Since the channel dimensions of input tokens vary across LVLMs with different parameter scales, performing pruning directly on the native visual tokens $\mathcal { T } _ { \mathrm { N } }$ (output by the image encoder) enhances the generalization capability of the pruning model. However, the actual inputs to the LVLM are the projected tokens $\mathcal { T } _ { \mathrm { P } }$ . Relying exclusively on $\mathcal { T } _ { \mathrm { N } }$ for pruning would inevitably neglect the inherent semantic requirements of the LVLM, thereby leading to performance degradation (as empirically verified in the ablation study in Section V-D2). Therefore, we propose to first determine the pruning selection on $\mathcal { T } _ { \mathrm { P } }$ , and subsequently use these results to guide the pruning of $\mathcal { T } _ { \mathrm { N } }$ . Specifically, utilizing the per-view pruning budgets as constraints, we independently cluster the foreground and background of the projected tokens $\mathcal { T } _ { \mathrm { P } }$ to construct the semantic structure. Based on this, we then jointly select the local and global semantic tokens in $\mathcal { T } _ { \mathrm { P } }$ yielding $\dot { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } }$ , which are then used to guide the pruning of $\mathcal { T } _ { \mathrm { N } }$ yielding the final pruned token set $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { \breve { T P } } }$ . The detailed designs of these steps are elaborated in Sections IV-A1 and IV-A2.

![](images/070cf55b6362568a4f20f09601020d91c5ff2fcbea2c8d01fcf9edf3698da8b1.jpg)  
Fig. 2. Illustration of the proposed LGS-HTP module for the i-th view. The module comprises four steps: (1) inputs preprocessing for initial mask and token preparation, (2) enhancement for dynamic budget allocation, (3) clustering for semantic grouping, and (4) selection of the final retained tokens.

1) Foreground-Background Token Decoupling and Budget Allocation: In multi-view 2D images, the spatial density of visual information typically exhibits obvious non-uniformity. Foreground regions concentrate intricate spatial structures and high-level semantic attributes of the target objects, thus carrying higher information density. In contrast, background regions predominantly comprise large homogeneous areas with lower information density. Applying uniform pruning overlooks this discrepancy, inevitably discarding critical foreground cues while retaining massive background redundancy.

To address this, we first employ a lightweight U-Net [50] based segmentation network to segment the input multiview images. As illustrated in Fig. 3, this network adopts a classical encoder-decoder paradigm, where skip connections are used to fuse deep semantic features with shallow spatial details. This design efficiently produces a high-quality pixelwise segmentation mask $\mathbf { M } _ { \mathrm { p i x } } \mathbf { \bar { \Pi } } \in \{ 0 , 1 \} ^ { 3 8 4 \times 3 \mathbf { \bar { 8 } } 4 }$ with minimal computational overhead. Here, ‘1’ denotes the foreground region and $\cdot _ { 0 } \cdot \mathrm { \ }$ denotes the background region.

![](images/7979c3842d7d14a29e60507482a1f7e0fcadae867f2db414d8f7af4f0e22f387.jpg)  
Fig. 3. Segmentation network for obtaining foreground-background priors

To align this pixel-level spatial prior $\mathbf { M } _ { \mathrm { p i x } }$ with the token space, we downsample it to the same resolution as the visual tokens and flatten it into a 1D vector, obtaining a token-level binary mask $m _ { \mathrm { t o k } } \in \{ 0 , 1 \} ^ { N _ { \mathrm { p } } \times 1 }$ , where $N _ { \mathrm { p } }$ is the original token count. Based on $_ { \mathbf { \mathit { \phi } } ^ { m _ { \mathrm { t o k } } } }$ , we decouple the projected visual tokens $( \mathcal { T } _ { \mathrm { P } } )$ into foreground tokens $( T _ { \mathrm { P } } ^ { \mathrm { f g } } )$ and background tokens $( T _ { \mathrm { P } } ^ { \mathrm { b g } } )$ , thereby providing vital semantic guidance for the subsequent pruning process.

Next, we adopt a non-uniform budget allocation strategy to maximize effective information. Given a target pruning ratio $R _ { \mathrm { p r } } ,$ the total budget for the i-th view is $\begin{array} { r l } { K _ { i } } & { { } = } \end{array}$ $\lfloor N _ { \mathrm { p } } \times ( 1 - R _ { \mathrm { p r } } ) \rfloor$ , where ⌊·⌋ denotes the floor operation. Recognizing that foreground regions usually encapsulate highfrequency textures and complex geometric structures essential for reasoning, we introduce a foreground enhancement factor $\lambda _ { \mathrm { b i a s } }$ to adaptively divide this budget. Taking the i-th view’s tokens $\mathbf { T } _ { \mathrm { P } , i }$ as an example:

$$
N _ { i } ^ { \mathrm { f g } } = \left\lfloor K _ { i } \times \frac { \left. \mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { f g } } \right. \times \lambda _ { \mathrm { b i a s } } } { \left. \mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { f g } } \right. \times \lambda _ { \mathrm { b i a s } } + \left. \mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { b g } } \right. } \right\rfloor ,\tag{11}
$$

$$
N _ { i } ^ { \mathrm { b g } } = K _ { i } - N _ { i } ^ { \mathrm { f g } } ,\tag{12}
$$

where $| \cdot |$ denotes the cardinality of the token set; $\mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { f g } }$ and $\mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { b g } }$ represent the decoupled foreground and background tokens of $\mathbf { T } _ { \mathrm { P } , i }$ , respectively; $N _ { i } ^ { \mathrm { f g } }$ and $N _ { i } ^ { \mathrm { { b g } } }$ denote the final allocated budgets for the foreground and background tokens of the i-th view, respectively.

![](images/b75cab0370587eb5fa48ecbded0aa769a8d0573a0d29f04a5f7aa08cf1ca1954.jpg)  
Fig. 4. Pipeline of the hybrid token selection for the foreground tokens of the i-th view (the background tokens are selected following the same process).

Overall, $\lambda _ { \mathrm { b i a s } }$ adaptively skews the proportion of retained tokens toward the foreground within the total budget $K _ { i } ,$ thereby guaranteeing the preservation of critical foreground semantics. The optimal value of $\lambda _ { \mathrm { b i a s } }$ is determined empirically through a grid search, which will be detailed in Section V-B1.

2) Local and Global Semantics-based Hybrid Token Selection: Specifically, the proposed LGS-HTP module comprises two stages: (a) Token clustering to explicitly model the semantic hierarchies, and (b) Hybrid token selection to jointly extract local and global semantics.

(a) Token clustering: Semantic redundancy fundamentally lies in the strong correlations among tokens. To systematically mitigate this, we first independently cluster the foreground and background tokens (as depicted in the ‘Clustering’ stage of Fig. 2) based on their feature embeddings to characterize their underlying semantic structures, thereby providing explicit guidance for the subsequent pruning.

Specifically, taking the foreground tokens of the i-th view $\mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { f g } }$ as an example, we set the target number of clusters to $\begin{array} { r l r } { \widehat { N } _ { i } ^ { \mathrm { f g } } } & { { } = } & { \left| \frac { N _ { i } ^ { \mathrm { f g } } } { N _ { \mathrm { s s } } } \right| } \end{array}$ , where $N _ { \mathrm { s s } }$ denotes the token budget allocated per cluster. We then adopt hierarchical agglomerative clustering [51] to model the semantic cluster structure. Unlike centroid-based algorithms [52] that rely on stochastic initialization, hierarchical clustering is deterministic and exhibits superior stability. Finally, this process partitions the foreground tokens into $\widehat { N } _ { i } ^ { \mathrm { f g } }$ semantic clusters, denoted as $\mathcal { C } _ { i } ^ { \mathrm { f g } } = \{ { \bf c } _ { i , c } ^ { \mathrm { f g } } \} _ { c = 1 } ^ { \widehat { N } _ { i } ^ { \mathrm { f g } } }$ The background clusters $\mathcal { C } _ { i } ^ { \mathrm { { b g } } } = \{ \mathbf { c } _ { i , c } ^ { \mathrm { { b g } } } \} _ { c = 1 } ^ { \widehat { N } _ { i } ^ { \mathrm { { b g } } } }$ are obtained similarly, where ${ \bf c } _ { i , c } ^ { \mathrm { f g } }$ and $\mathbf { c } _ { i , c } ^ { \mathrm { { b g } } }$ denote the c-th cluster of the i-th view’s foreground and background tokens, respectively.

(b) Hybrid token selection: As illustrated in Fig. 4, given the decoupled foreground and background semantic clusters, we execute two selection branches within each cluster: a local semantic branch to extract a single local semantic token and a global semantic branch to select global semantic tokens.

In the local semantic branch, we aim to mitigate the redundancy between similar tokens. Specifically, taking ${ \bf c } _ { i , c } ^ { \mathrm { f g } }$ as an example, we define its local semantic centroid as the feature mean of all constituent tokens: $\begin{array} { r } { \varphi _ { i , c } ^ { \mathrm { f g } } = \frac { 1 } { | { \bf c } _ { i , c } ^ { \mathrm { f g } } | } \sum _ { t \in { \bf c } _ { i , c } ^ { \mathrm { f g } } } t } \end{array}$ where |·| denotes the cardinality of the cluster and t represents an arbitrary token vector within ${ \bf c } _ { i , c } ^ { \mathrm { f g } }$ . We then select the token closest to ${ \varphi } _ { i , c } ^ { \mathrm { f g } }$ as the local semantic token:

$$
\pmb { t } _ { i , c } ^ { \mathrm { f g , l o c } } = \arg \operatorname* { m i n } _ { \pmb { t } \in { \bf c } _ { i , c } ^ { \mathrm { f g } } } \left\| \pmb { t } - \varphi _ { i , c } ^ { \mathrm { f g } } \right\| _ { 2 } ,\tag{13}
$$

where $t _ { i , c } ^ { \mathrm { f g , l o c } }$ denotes the local semantic token of cluster ${ \bf c } _ { i , c } ^ { \mathrm { f g } } .$ This operation precisely extracts the token that best generalize the cluster semantics, thereby removing redundancy of similar tokens. The local semantic tokens for the background clusters are obtained in the same manner.

In the global semantic branch, we aim to avoid the global semantics distortion caused by aggressive token dropping. To this end, we first select $K _ { i }$ uniformly sampled tokens from the original tokens via 1D uniform sampling, denoted as $\mathcal { G } _ { i } = \{ t _ { m } ^ { \mathrm { g l o } } \} _ { m = 1 } ^ { K _ { i } }$ where $\pmb { t } _ { m } ^ { \mathrm { g l o } }$ denotes the m-th sampled token and $K _ { i }$ is the total pruning budget for the i-th view. Then, conditioned on the local semantic branch, we further select $K _ { \mathrm { g l o } }$ global semantic tokens within each cluster whose 1D physical locations best align with the predefined uniformly sampled tokens in $\mathcal { G } _ { i }$ , where $K _ { \mathrm { g l o } } = N _ { \mathrm { s s } } - 1$ , with the −1 term reserving a single token budget for the local semantic token within each cluster. Notably, the optimal $K _ { \mathrm { g l o } }$ is experimentally tuned in Section V-B2. Specifically, for ${ \bf c } _ { i , c } ^ { \mathrm { f g } }$ , except for the local semantic token $t _ { i , c } ^ { \mathrm { f g , l o c } }$ , we define a spatial alignment score $S _ { t }$ for each remaining token t in the cluster as

$$
S _ { t } = \operatorname* { m i n } _ { t ^ { \mathrm { g l o } } \in \mathcal { G } _ { i } } \left| p ( t ) - p ( t ^ { \mathrm { g l o } } ) \right| ,\tag{14}
$$

where $p ( \cdot )$ denotes the 1D coordinate of the token. Subsequently, the $K _ { \mathrm { g l c } }$ tokens yielding the minimum alignment scores are selected to form the global semantic token set of the cluster:

$$
\begin{array} { r } { { \bf T } _ { i , c } ^ { \mathrm { f g , g l o } } = \mathrm { T o p K } ( s , K _ { \mathrm { g l o } } ) , } \end{array}\tag{15}
$$

where $\begin{array} { r c l } { s } & { = } & { \{ S _ { t } \} _ { t \in \mathbf { c } _ { i , c } ^ { \mathrm { f g } } } } \end{array}$ denotes the set of spatial alignment scores for all remaining tokens within the cluster, and $\mathrm { T o p K } ( \cdot , K _ { \mathrm { g l o } } )$ identifies the subset of tokens corresponding to the smallest alignment scores. The global semantic tokens for the background clusters are extracted analogously.

Finally, we aggregate the local and global semantic tokens for all clusters. The final pruned token sets for the foreground and background of this view are mathematically written as

$$
\mathbf { T } _ { i } ^ { \mathrm { f g , k e e p } } = \bigcup _ { c = 1 } ^ { \widehat { N } _ { i } ^ { \mathrm { f g } } } \left( \{ t _ { i , c } ^ { \mathrm { f g , l o c } } \} \cup \mathbf { T } _ { i , c } ^ { \mathrm { f g , g l o } } \right) ,\tag{16}
$$

$$
\mathbf { T } _ { i } ^ { \mathrm { b g , k e e p } } = \bigcup _ { c = 1 } ^ { \widehat { N } _ { i } ^ { \mathrm { b g } } } \left( \{ t _ { i , c } ^ { \mathrm { b g , l o c } } \} \cup \mathbf { T } _ { i , c } ^ { \mathrm { b g , g l o } } \right) .\tag{17}
$$

![](images/e84b26d8194cca32296f57e3a37373d31af7601d1694c4e51b3a0cde334a1844.jpg)  
Fig. 5. Overall architecture of the proposed T-JSCC network. A lightweight MLP network first reduces the token channel dimension to 64. The compressed tokens are then power-normalized and transmitted over error-prone channels. A symmetric MLP decoder is subsequently utilized for token reconstruction. Furthermore, the channel SNR $\gamma$ is injected into both the encoding and decoding processes to achieve channel-adaptive transmission.

The complete set of pruned tokens for the i-th view is thus $\mathbf { T } _ { \mathrm { P } , i } ^ { \mathrm { T P } } = \mathbf { \dot { T } } _ { i } ^ { \mathrm { f g , k e e p } } \cup \mathbf { T } _ { i } ^ { \mathrm { b g , k e e p } }$ . By iterating this process across all views, we obtain the multi-view pruned token set $\mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } }$

Notably, while the pruning decisions are formulated in the projected token space $( D _ { \mathrm { P } } )$ to leverage its superior representational capacity, directly transmitting these tokens $( \mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } } )$ incurs prohibitive communication overhead due to their significantly larger channel dimensions. Given the strict spatial correspondence between the projected and the native tokens, we directly map the corresponding indices of $\mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } }$ within $\mathcal { T } _ { \mathrm { P } }$ onto $\mathcal { T } _ { \mathrm { N } }$ to obtain pruned native tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ , which is then fed into the subsequent T-JSCC module.

## B. Token Joint Source Channel Coding

While the LGS-HTP module effectively eliminates tokenlevel redundancy, significant redundancy exists along the channel dimension of the pruned tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { \bar { T P } } }$ . To further mitigate this channel-level redundancy and ensure robust transmission under the limited computational constraints of EAs, we propose a T-JSCC module which achieves end-to-end token compression and noise robustness with strictly bounded ondevice overhead.

1) Lightweight Token-JSCC Architecture: As illustrated in Fig. 5, the proposed T-JSCC adopts a symmetric MLP-based encoder-decoder architecture. Crucially, it dynamically incorporates the channel state (SNR) γ as a conditioning variable, which can be accurately estimated in real time using dedicated equipment and algorithms at the receiver [53], thereby providing channel-aware adaptivity across fluctuating channel conditions.

Specifically, for an arbitrary token in the i-th view, denoted as $\bar { \mathbf { t } _ { i , n } } \in \mathbb { R } ^ { \bar { D } _ { \mathrm { N } } }$ , we concatenate it with γ to form the network input $[ \pmb { t } _ { i , n } ; \gamma ] \in \mathbb { R } ^ { D _ { \mathrm { N } } + 1 }$ . The encoder $\mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { e n c } }$ employs a progressive bottleneck design to reduce dimensionality, mapping the conditional input to the channel symbols:

$$
\begin{array} { r } { \pmb { y } _ { i , n } = \mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { e n c } } ( [ \pmb { t } _ { i , n } ; \gamma ] ) , } \end{array}\tag{18}
$$

where the dimensional transitions across the hidden layers strictly follow the configurations in Fig. 5, i.e., $D _ { \mathrm { N } } + 1 $ $5 1 2  2 5 6  1 2 8  D _ { \mathrm { c } }$ . This bottlenecked MLP minimizes computational overhead and subsequently reduces inference latency while preserving its nonlinear representation ability. Furthermore, the encoder and decoder process each token independently and in parallel with shared parameters, ensuring the overall complexity scales linearly with the number of input tokens to facilitate on-device deployment.

Subject to average power constraints [39], the encoded symbols are transmitted over an error-prone communication channel. The decoder $\mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { d e c } }$ mirrors the bottleneck architecture of $\mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { e n c } }$ and symmetrically injects the channel state γ, reconstructing each token as

$$
\widehat { \pmb { t } } _ { i , n } = \mathcal { F } _ { \mathrm { J S C C } } ^ { \mathrm { d e c } } ( [ \widehat { \pmb { y } } _ { i , n } ; \gamma ] ) .\tag{19}
$$

After reconstructing the total native token set $\widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } }$ , we employ the identical pre-trained projector $\mathcal { F } _ { \mathrm { p r o j } } ( \cdot )$ on the cloud server to linearly project it to $\tilde { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } }$ for the subsequent LVLM reasoning.

2) Loss Design for T-JSCC module: Imposing a reconstruction constraint only on the native tokens $( \widehat { T } _ { \mathrm { N } } ^ { \mathrm { T P } } )$ is insufficient to guarantee the reconstruction quality of the projected tokens $( \mathrm { \widehat { T } _ { P } ^ { T P } } )$ obtained via the projection $\mathcal { F } _ { \mathrm { p r o j } }$ . This is because neural network projections inherently amplify minor input perturbations [54], even slight reconstruction errors in the reconstructed native tokens can be significantly magnified by the projector.

To address this, we formulate a coarse-to-fine joint constraint. The native reconstruction loss acts as a coarse-grained constraint, stabilizing the decoded $\widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } }$ to provide a proper convergence direction. Meanwhile, the projected reconstruction loss serves as a fine-grained constraint, ensuring the reconstruction quality of the projected tokens.

Specifically, we define the coarse-grained reconstruction loss between the reconstructed native tokens $\widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } }$ and the target native tokens $\mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } }$ as

$$
\mathcal { L } _ { \mathrm { N } } = \left. \widehat { \mathcal { T } } _ { \mathrm { N } } ^ { \mathrm { T P } } - \mathcal { T } _ { \mathrm { N } } ^ { \mathrm { T P } } \right. _ { 2 } ^ { 2 } ,\tag{20}
$$

where $\lVert \cdot \rVert _ { 2 } ^ { 2 }$ signifies the MSE. Concurrently, the fine-grained loss between the projected tokens $\widehat { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } }$ and the targets pro-

jected tokens $\mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } }$ is defined as

$$
\mathcal { L } _ { \mathrm { P } } = \left. \widehat { \mathcal { T } } _ { \mathrm { P } } ^ { \mathrm { T P } } - \mathcal { T } _ { \mathrm { P } } ^ { \mathrm { T P } } \right. _ { 2 } ^ { 2 } .\tag{21}
$$

Finally, the overall training objective is formulated as

$$
\mathcal { L } _ { \mathrm { J S C C } } = \lambda _ { \mathrm { N } } \mathcal { L } _ { \mathrm { N } } + \lambda _ { \mathrm { P } } \mathcal { L } _ { \mathrm { P } } ,\tag{22}
$$

where $\lambda _ { \mathrm { N } }$ and $\lambda _ { \mathrm { P } }$ are the balancing coefficients for the coarseand fine-grained losses, respectively.

## V. EXPERIMENTAL RESULTS

## A. Experimental Setup

We evaluated our method on the ScanQA [8] and SQA3D [9] benchmarks. ScanQA provides ∼41K QA pairs covering diverse tasks (e.g., recognition and spatial reasoning), while SQA3D offers ∼33K questions emphasizing situated understanding. Given its comprehensive nature, ScanQA serves as our primary testbed for all detailed analyses and ablation studies. Meanwhile, we further evaluate our model on SQA3D to validate its generalizability.

For quantitative evaluation, we employed two complementary metrics: Top-1 Exact Match (EM) and Consensus-based Image Description Evaluation (CIDEr). Top-1 EM strictly measures exact string-matching accuracy, serving as a direct indicator of absolute correctness. Meanwhile, CIDEr evaluates n-gram similarities, providing robustness against synonymous expressions to comprehensively assess semantic consistency and linguistic quality. Throughout our experiments and hyperparameter selection, we jointly considered both metrics to optimally balance strict reasoning accuracy with textual expressiveness.

For all experiments, we employed LLaVA-OV [18] as the foundational LVLM. As a state-of-the-art (SOTA) open-source LVLM, LLaVA-OV performs excellent in high-resolution visual representation and multi-view reasoning, making it an ideal baseline architecture to validate the effectiveness of our proposed token compression framework.

## B. Hyper-parameter Settings

Our proposed method involves six user-defined hyperparameters: the foreground-enhancement bias $\lambda _ { \mathrm { b i a s } }$ , the global semantic token number per cluster $K _ { \mathrm { g l o } } .$ , the compressed channel dimension $D _ { c }$ in T-JSCC, the loss weights of T-JSCC $\lambda _ { \mathrm { N } }$ and $\lambda _ { \mathrm { P } } .$ , and the SNR training range for T-JSCC. In the following Section V-B1 to V-B5, we conduct comprehensive experiments to determine their optimal configurations.

1) Selection of the foreground-enhancement bias $\lambda _ { \mathrm { b i a s } } .$ This parameter dictates the token budget allocation between the foreground and background. While a higher $\lambda _ { \mathrm { b i a s } }$ prioritizes critical foreground semantics, an excessively high value disproportionately compresses the background, sacrificing the indispensable global context required for spatialrelation reasoning. To optimally balance foreground salience with contextual preservation, we conduct a grid search for $\lambda _ { \mathrm { b i a s } } \in \{ 1 . 0 , 1 . 1 , . . . , 2 . 0 \}$ under 90% pruning (Table I).

As observed, performance follows an inverted-U trajectory. Lower values $( \lambda _ { \mathrm { b i a s } } ~ \leq ~ 1 . 2 )$ , retain excessive background redundancy, while higher values $( \lambda _ { \mathrm { b i a s } } ~ \ge ~ 1 . 6 )$ compromise global spatial context. Both extremes lead to consistent drops in both metrics. Ultimately, performance peaks simultaneously at $\lambda _ { \mathrm { b i a s } } = 1 . 5$ , which we establish as the default configuration for all subsequent experiments.

TABLE I  
EFFECT OF THE FOREGROUND-ENHANCEMENT BIAS $\lambda _ { \mathrm { b i a s } }$ UNDER 90%TOKEN PRUNING ON THE SCANQA VALIDATION SET. THE BEST,SECOND-BEST, AND THIRD-BEST RESULTS ARE HIGHLIGHTED IN BOLD,UNDERLINE, AND italic, RESPECTIVELY.
<table><tr><td> $\lambda _ { \mathrm { b i a s } }$ </td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>1.0</td><td>28.5989</td><td>98.4296</td></tr><tr><td>1.1</td><td>28.5989</td><td>98.6438</td></tr><tr><td>1.2</td><td>28.4278</td><td>98.3734</td></tr><tr><td>1.3</td><td>28.5775</td><td>98.5151</td></tr><tr><td>1.4</td><td>28.5561</td><td>98.2152</td></tr><tr><td>1.5</td><td>28.6203</td><td>98.5228</td></tr><tr><td>1.6</td><td>28.4492</td><td>98.0354</td></tr><tr><td>1.7</td><td>28.4064</td><td>97.8054</td></tr><tr><td>1.8</td><td>28.3209</td><td>97.7389</td></tr><tr><td>1.9</td><td>28.4278</td><td>97.7351</td></tr><tr><td>2.0</td><td>28.2995</td><td>97.3203</td></tr></table>

2) Selection of the global semantic token number per cluster $K _ { \mathrm { g l o } } .$ As formulated in Section III, precisely calibrating the number of local and global semantic tokens is critical. Since our method selects only a single local semantic token per cluster, we tune the global semantic token number per cluster, $K _ { \mathrm { g l o } }$ , under two distinct pruning constraints (80% and 90% ratios) to strike the optimal balance (Table II and III).

TABLE II  
EFFECT OF $K _ { \mathrm { g l o } }$ IN LGS-HTP UNDER 80% TOKEN PRUNING ON THESCANQA VALIDATION SET. THE BEST, SECOND-BEST, AND THIRD-BESTRESULTS ARE HIGHLIGHTED IN BOLD, UNDERLINE, AND italic,RESPECTIVELY.
<table><tr><td> $K _ { \mathrm { g l o } }$ </td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>w/o Pruning</td><td>29.6898</td><td>99.4314</td></tr><tr><td>0</td><td>29.3262</td><td>98.1047</td></tr><tr><td>1</td><td>29.7540</td><td>99.5614</td></tr><tr><td>2</td><td>29.3048</td><td>99.5475</td></tr><tr><td>3</td><td>28.6845</td><td>97.4190</td></tr><tr><td>4</td><td>28.2353</td><td>96.9864</td></tr></table>

TABLE III

EFFECT OF $K _ { \mathrm { g l o } }$ IN LGS-HTP UNDER 90% TOKEN PRUNING ON THESCANQA VALIDATION SET. THE BEST, SECOND-BEST, AND THIRD-BESTRESULTS ARE HIGHLIGHTED IN BOLD, UNDERLINE, AND italic,RESPECTIVELY.
<table><tr><td> $K _ { \mathrm { g l o } }$ </td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>w/o Pruning</td><td>29.6898</td><td>99.4314</td></tr><tr><td>0</td><td>28.2567</td><td>95.8569</td></tr><tr><td>1</td><td>28.6203</td><td>98.5228</td></tr><tr><td>2</td><td>27.8717</td><td>96.9990</td></tr><tr><td>3</td><td>28.0214</td><td>97.5871</td></tr><tr><td>4</td><td>28.0214</td><td>97.6235</td></tr></table>

As observed, retaining only local semantic tokens $( K _ { \mathrm { g l o } } =$ 0) results in insufficient global semantic coverage, which noticeably degrades performance across both metrics and causes a substantial plunge under the extreme 90% ratio. Conversely, allocating an excessive budget to global semantic tokens (e.g., $K _ { \mathrm { g l o } } ~ = ~ 3 ~ \mathrm { o r } ~ 4 )$ inevitably weakens key local semantics, similarly leading to performance degradation. In contrast, a moderate allocation $( K _ { \mathrm { g l o } } = 1 )$ exhibits exceptional resilience. Remarkably, at an 80% pruning ratio, LGS-HTP even surpasses the performance of the unpruned baseline. Considering its robust stability across varying constraints, we set the default $K _ { \mathrm { g l o } }$ to 1, which optimally balances local and global semantics to yield peak performance.

3) Selection of the compressed channel dimension $D _ { \mathrm { c } }$ in T-JSCC: The compressed channel dimension $D _ { c }$ directly dictates the transmission payload. While expanding $D _ { c }$ enhances reconstruction fidelity, it inevitably incurs a higher communication overhead. Conversely, over-compression severely degrades semantics. To strike a rigorous balance, we evaluate varying $D _ { c }$ configurations under 90% pruning (Table IV).

TABLE IV  
IMPACT OF THE COMPRESSED CHANNEL DIMENSION $D _ { c }$ IN T-JSCC UNDER 90% TOKEN PRUNING ON THE SCANQA VALIDATION SET.
<table><tr><td> $D _ { c }$ </td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>16</td><td>26.9733</td><td>92.6928</td></tr><tr><td>32</td><td>27.5722</td><td>94.5314</td></tr><tr><td>64</td><td>27.9144</td><td>96.3107</td></tr><tr><td>128</td><td>28.1711</td><td>96.5524</td></tr><tr><td>256</td><td>28.2995</td><td>97.5974</td></tr></table>

## Performance vs. Compressed Channel Dimension (D<sub>c</sub>)

![](images/cb92b4497315f1ab2e6e7c7a1bcd8c3462047c54a63a3b29cf2144f98bf086a8.jpg)  
Fig. 6. Impact of the compressed channel dimension $D _ { c }$ on 3D-VQA performance under 90% pruning.

As observed, increasing $D _ { c }$ consistently improves both evaluation metrics, with the most significant performance gains occurring when scaling from 16 to 64. Further expanding $D _ { c }$ to 256 yields only marginal improvements but drastically increases the data volume of the compressed tokens. Visualizing these trajectories (Fig. 6) reveals a distinct inflection point (i.e., the ‘Elbow Point’) precisely at $D _ { c } = 6 4 .$ . Consequently, to maximize reasoning accuracy while strictly control the communication payload, we establish $D _ { c } = 6 4$ as the default compressed channel dimension.

4) Selection of the loss weight of T-JSCC: To determine the optimal balance between the coarse-grained reconstruction error $\mathcal { L } _ { \mathrm { N } }$ and the fine-grained reconstruction error $\mathcal { L } _ { \mathrm { P } }$ , we evaluate weight allocations $\lambda _ { \mathrm { { N } } } : \lambda _ { \mathrm { { P } } }$ from 10:0 to 0:10 at $D _ { c } = 6 4$ (Table V).

As observed, relying exclusively on the coarse-grained constraint (10:0) maintains a reasonable Top-1 EM but severely degrades CIDEr. Conversely, entirely discarding the coarsegrained constraint (0:10) causes a precipitous drop in both metrics. These findings indicate that both fine-grained and coarsegrained losses are indispensable. Furthermore, as the weight of ${ \mathcal { L } } _ { \mathrm { P } }$ is appropriately increased, the CIDEr score exhibits a steady upward trajectory, achieving the peak performance at a ratio of 1:9. Consequently, to optimally balance training stability and semantic fidelity, we designate $\lambda _ { \mathrm { N } } : \lambda _ { \mathrm { P } } = 1 : 9$ as the default loss-weight configuration.

TABLE V  
EFFECT OF THE T-JSCC LOSS-WEIGHT SETTING λ<sub>N</sub> : λ<sub>P</sub> ON THESCANQA VALIDATION SET. THE BEST, SECOND-BEST, AND THIRD-BESTRESULTS ARE HIGHLIGHTED IN BOLD, UNDERLINE, AND italic,RESPECTIVELY.
<table><tr><td> $\lambda _ { \mathrm { N } } : \lambda _ { \mathrm { P } }$ </td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>10:0</td><td>28.2567</td><td>95.0720</td></tr><tr><td>9:1</td><td>27.8289</td><td>95.2286</td></tr><tr><td>8:2</td><td>27.9358</td><td>95.4264</td></tr><tr><td>7:3</td><td>27.9144</td><td>95.2138</td></tr><tr><td>6:4</td><td>27.9572</td><td>96.0160</td></tr><tr><td>5:5</td><td>27.9144</td><td>96.3107</td></tr><tr><td>4:6</td><td>28.2567</td><td>96.6053</td></tr><tr><td>3:7</td><td>27.9786</td><td>97.0020</td></tr><tr><td>2:8</td><td>28.1497</td><td>96.8125</td></tr><tr><td>1:9</td><td>28.0428</td><td>97.3429</td></tr><tr><td>0:10</td><td>26.3102</td><td>95.0770</td></tr></table>

5) Selection of the SNR training range for T-JSCC: To determine the optimal SNR training range, we initially profile the natural 3D-VQA performance degradation over an AWGN channel without joint T-JSCC training (Table VI).

TABLE VI  
3D-VQA PERFORMANCE UNDER DIFFERENT AWGN CHANNEL SNRS ON THE SCANQA VALIDATION SET.
<table><tr><td>SNR</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>w/o AWGN Channel</td><td>28.0428</td><td>97.3429</td></tr><tr><td>0</td><td>7.5508</td><td>25.3298</td></tr><tr><td>5</td><td>12.1497</td><td>38.1479</td></tr><tr><td>10</td><td>18.5882</td><td>65.4938</td></tr><tr><td>15</td><td>23.7647</td><td>87.1400</td></tr><tr><td>20</td><td>26.7380</td><td>94.0205</td></tr><tr><td>25</td><td>27.5936</td><td>96.3079</td></tr><tr><td>30</td><td>27.9144</td><td>96.6293</td></tr></table>

As observed, without channel coding, the performance suffers a catastrophic degradation under severe noise conditions (e.g., SNR = 0 dB). As the SNR increases, the performance gradually recovers and eventually saturates beyond 25 dB, indicating that the primary transmission bottleneck is concentrated within the 0∼25 dB SNR regime. Consequently, we adopted 0∼25 dB as the definitive SNR sampling range for training.

## C. Performance Comparison

1) LGS-HTP Performance: We first compare the proposed LGS-HTP with three representative baseline pruning methods: Farthest Point Sampling (FPS)-based pruning, random pruning, and uniform pruning. The experimental results are summarized in Table VII.

As shown, under 80% pruning, LGS-HTP consistently achieves the best Top-1 EM while maintaining competitive CIDEr scores. Although FPS achieves relatively strong Top-1 EM by retaining richer local semantics, its lack of global semantics leads to a notable CIDEr drop. In contrast, uniform pruning yields the highest CIDEr, because it preserves more comprehensive global semantics, which provides the broad scene context essential for generating the descriptive responses that the CIDEr metric favors. However, its indiscriminate selection weakens local semantic cues, leading to the lowest Top-1 EM.

TABLE VII  
COMPARISON OF LGS-HTP WITH BASELINE TOKEN PRUNING METHODSAT 80% AND 90% PRUNING RATIOS ON THE SCANQA VALIDATION SET.THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS FOR EACH PRUNINGRATIO ARE HIGHLIGHTED IN BOLD, UNDERLINE, AND italic,RESPECTIVELY.
<table><tr><td>Pruning Method</td><td>Pruning Ratio</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>w/o Pruning</td><td>0%</td><td>29.6898</td><td>99.4314</td></tr><tr><td>FPS</td><td>80%</td><td>29.5187</td><td>94.0496</td></tr><tr><td>Random</td><td>80%</td><td>29.2406</td><td>99.3210</td></tr><tr><td>Uniform</td><td>80%</td><td>28.8770</td><td>101.1091</td></tr><tr><td>Proposed LGS-HTP</td><td>80%</td><td>29.7540</td><td>99.5614</td></tr><tr><td>FPS</td><td>90%</td><td>28.3850</td><td>89.2389</td></tr><tr><td>Random</td><td>90%</td><td>28.0428</td><td>94.7790</td></tr><tr><td>Uniform</td><td>90%</td><td>28.0428</td><td>98.4563</td></tr><tr><td>Proposed LGS-HTP</td><td>90%</td><td>28.6203</td><td>98.5228</td></tr></table>

Under the more aggressive pruning rate (i.e., 90%), the performance gap becomes more obvious, with LGS-HTP consistently exhibiting superiority over all baselines. It is worth noting that LGS-HTP overtakes uniform pruning in CIDEr scores. This is primarily because under such extreme pruning constraints, the severe loss of local semantics in uniform sampling inevitably pulls down its CIDEr metric. Conversely, our LGS-HTP inherently preserves both local and global semantics, thereby yielding superior overall results.

To comprehensively evaluate the effectiveness of our approach, we then compare LGS-HTP against several recent SOTA token pruning methods on the ScanQA validation set and the SQA3D test set. The performance under 80% and 90% pruning ratios is summarized in Table VIII.

As shown, LGS-HTP consistently establishes new stateof-the-art results on the ScanQA dataset across all metrics and pruning constraints. On the SQA3D dataset, we observe an interesting phenomenon at the 80% pruning ratio: our method ranks third, slightly trailing behind PruMerge and DivPrune. This can be attributed to the inherent characteristics of SQA3D, where the ground-truth answers are mainly short phrases. At an 80% pruning rate, global semantics is not yet the primary performance bottleneck, as a relatively small number of tokens can already cover the semantics required for these concise answers. Consequently, other methods that concentrate their token budget heavily on local semantics gain a marginal advantage in this specific scenario.

However, as the pruning rate increases to the extreme 90%, where only a minimal number of tokens is retained, the lack of global semantics becomes the main bottleneck. Under this strict condition, methods that over-emphasize local semantics suffer from severe observation blind spots, leading to noticeable performance drops. In contrast, our LGS-HTP inherently preserves a balanced representation of both local and global semantics. As a result, it maintains stable performance and significantly outperforms all other SOTA methods on SQA3D at the 90% pruning ratio, thoroughly validating the superiority and robustness of our local-global semantic theory in extreme pruning scenarios.

2) Overall Performance: After validating the efficacy of the LGS-HTP module, we extend our evaluation to the end-to-end performance of the integrated ‘LGS-HTP + T-JSCC’ pipeline. Given the scarcity of token-JSCC methods specifically designed for LVLM-driven 3D-VQA, establishing direct comparisons on the ScanQA validation set is currently unfeasible. Therefore, in this section, we primarily investigate the overall effectiveness of our proposed method, as well as the impact of different SNR training strategies on the model’s noise resilience.

We first evaluate a baseline strategy that uniformly samples the SNR from 0∼25 dB during training, named UFM. The corresponding experimental results are presented in Table IX.

First, a direct comparison between Table VI and Table IX highlights the critical necessity of our proposed T-JSCC method. Under identical SNR conditions, particularly in extreme low-SNR regimes (e.g., at 0 dB), the integration of T-JSCC yields a dramatic performance improvement over the baseline (boosting the Top-1 EM from a mere 7.55 to 26.97). This indicates that our method successfully mitigates the severe impact of channel noise on 3D-VQA performance.

Nevertheless, further analysis of Table IX reveals that the UFM strategy leaves substantial room for improvement. Notably, as the SNR improves to favorable conditions (e.g., 20 and 25 dB), the performance exhibits a counter-intuitive decline rather than a continuous increase. We attribute this anomaly to the training instability caused by directly mixing high and low SNR samples. Because the reconstruction loss under extreme noise is inherently much larger than that under clean channels, jointly optimizing them with equal probability leads to severe gradient fluctuations. This prevents the model from stably converging across varying channel conditions.

To mitigate this optimization bias, we introduce a progressive (PRG) sampling strategy, adopting an anti-curriculum learning [55] schedule. During early training epochs, we strictly confine SNR sampling to the high-noise regime (e.g., 0∼5 dB), compelling the encoder to prioritize noise-robust semantic feature extraction. As training advances, we incrementally expand the sampling range epoch by epoch to encompass the full 0∼25 dB spectrum. The corresponding experimental results are presented in Table X.

A comparative analysis between Table IX and Table X reveals that the PRG strategy significantly enhances the model’s low-SNR resilience. Notably, under severe channel conditions (e.g., SNR = 0 dB and 5 dB), the Top-1 EM achieves substantial improvements. Concurrently, the peak performance is firmly sustained under favorable channel conditions (e.g., SNR = 25 dB). More importantly, instead of the counterintuitive decline at higher SNRs, the performance now exhibits a stable and consistent upward trajectory as channel conditions improve, successfully resolving the training instability observed in the UFM baseline.

## D. Ablation Studies

In this section, we conduct comprehensive ablation studies to validate our proposed method. First, in Section V-D1, we ablate the core components within the LGS-HTP module to verify their individual contributions. Next, to demonstrate the effectiveness of our strategy (using the projected tokens to guide the pruning of the native visual tokens), we compare the performance of different pruning schemes in Section V-D2. Finally, to highlight the practical efficiency of our approach, we report the computational overhead of the LVLM before and after integrating our method in Section V-D3.

TABLE VIII  
COMPARISON WITH SOTA PRUNING METHODS ON THE SCANQA VALIDATION SET AND THE SQA3D TEST SET. THE BEST, SECOND-BEST, ANDTHIRD-BEST RESULTS FOR EACH PRUNING RATIO ARE HIGHLIGHTED IN BOLD, UNDERLINE, AND italic, RESPECTIVELY.
<table><tr><td rowspan="2">Pruning Method</td><td rowspan="2">Pruning Ratio</td><td colspan="2">ScanQA</td><td colspan="2">SQA3D</td></tr><tr><td>Top-1 EM</td><td>CIDEr</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>w/o Pruning</td><td>0%</td><td>29.6898</td><td>99.4314</td><td>50.6394</td><td>132.6469</td></tr><tr><td>ToMe [39]</td><td>80%</td><td>27.8717</td><td>99.4191</td><td>48.8491</td><td>127.9719</td></tr><tr><td>GPrune [84]</td><td>80%</td><td>24.6203</td><td>77.9843</td><td>46.4052</td><td>120.9888</td></tr><tr><td>CDPrune [85]</td><td>80%</td><td>28.3209</td><td>91.8665</td><td>49.0480</td><td>128.19</td></tr><tr><td>DivPrune [86]</td><td>80%</td><td>29.1978</td><td>93.3695</td><td>50.0995</td><td>131.0206</td></tr><tr><td>PruMerge [87]</td><td>80%</td><td>29.5401</td><td>92.1868</td><td>50.1563</td><td>131.1077</td></tr><tr><td>VisPrune [88]</td><td>80%</td><td>28.7487</td><td>90.3498</td><td>49.6164</td><td>129.8927</td></tr><tr><td>LGS-HTP</td><td>80%</td><td>29.7540</td><td>99.5614</td><td>49.7869</td><td>130.4288</td></tr><tr><td>ToMe</td><td>90%</td><td>26.6524</td><td>94.9143</td><td>48.7070</td><td>128.0065</td></tr><tr><td>GPrune</td><td>90%</td><td>21.8182</td><td>67.7759</td><td>43.9898</td><td>114.3564</td></tr><tr><td>CDPrune</td><td>90%</td><td>26.3743</td><td>87.4624</td><td>48.4513</td><td>126.2958</td></tr><tr><td>DivPrune</td><td>90%</td><td>27.3583</td><td>87.3509</td><td>49.1901</td><td>128.5078</td></tr><tr><td>PruMerge</td><td>90%</td><td>26.6096</td><td>82.4897</td><td>47.826</td><td>125.2658</td></tr><tr><td>VisPrune</td><td>90%</td><td>26.3529</td><td>82.2962 98.5228</td><td>48.3376 49.5311</td><td>126.4189</td></tr><tr><td>LGS-HTP</td><td>90%</td><td>28.6203</td><td></td><td></td><td>130.0889</td></tr></table>

TABLE IX

END-TO-END 3D-VQA PERFORMANCE OF THE PROPOSED METHOD UNDER DIFFERENT SNRS WITH UNIFORM SNR SAMPLING (UFM) DURING TRAINING ON THE SCANQA VALIDATION SET.
<table><tr><td>SNR</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>0</td><td>26.9733</td><td>94.0668</td></tr><tr><td>5</td><td>27.6578</td><td>96.0911</td></tr><tr><td>10</td><td>28.0428</td><td>96.4681</td></tr><tr><td>15</td><td>27.9786</td><td>97.1514</td></tr><tr><td>20</td><td>27.9572</td><td>97.0533</td></tr><tr><td>25</td><td>27.9572</td><td>96.5340</td></tr></table>

TABLE X

END-TO-END 3D-VQA PERFORMANCE OF THE PROPOSED METHOD UNDER DIFFERENT SNRS WITH PROGRESSIVE SNR RANGE EXPANSION (PRG) DURING TRAINING ON THE SCANQA VALIDATION SET.
<table><tr><td>SNR</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>0</td><td>27.2941</td><td>94.8344</td></tr><tr><td>5</td><td>27.8503</td><td>96.6230</td></tr><tr><td>10</td><td>27.8717</td><td>96.2943</td></tr><tr><td>15</td><td>27.8717</td><td>96.5267</td></tr><tr><td>20</td><td>27.9786</td><td>96.8218</td></tr><tr><td>25</td><td>28.1070</td><td>96.9992</td></tr></table>

1) Ablation Studies on LGS-HTP module: To rigidly validate the efficacy of the core components within the proposed LGS-HTP module, we conduct comprehensive ablation studies under the extreme 90% pruning regime (Table XII). Since local semantic extraction is indispensable for basic reasoning, we separate the ablation variables into: the foreground and background token decoupling mechanism and the initial sampling strategy for global semantics. The corresponding experimental results are presented in Table XI.

As shown in Table XI, explicitly decoupling tokens consistently increase performance across all selection strategies.

TABLE XI  
ABLATION STUDY OF LGS-HTP COMPONENTS: TOKEN DECOUPLING ANDGLOBAL SEMANTIC TOKENS SELECTION STRATEGIES UNDER 90% TOKENPRUNING ON THE SCANQA VALIDATION SET. THE BEST, SECOND-BEST,AND THIRD-BEST RESULTS ARE HIGHLIGHTED IN BOLD, UNDERLINE, ANDitalic, RESPECTIVELY.
<table><tr><td>Token Decoupling</td><td>Selection Strategy</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td> $\times$ </td><td>FPS</td><td>28.5134</td><td>95.1825</td></tr><tr><td> $\times$ </td><td>Random</td><td>28.0856</td><td>95.8969</td></tr><tr><td> $\times$ </td><td>Uniform</td><td>28.1497</td><td>98.0843</td></tr><tr><td> $\checkmark$ </td><td>FPS</td><td>29.1551</td><td>95.0144</td></tr><tr><td> $\checkmark$ </td><td>Random</td><td>28.2995</td><td>96.0001</td></tr><tr><td> $\checkmark$ </td><td>Uniform</td><td>28.6203</td><td>98.5228</td></tr></table>

Specifically, when paired with our uniform selection strategy, integrating the decoupling mechanism boosts Top-1 EM by 0.47 and CIDEr by 0.46 compared to the variant without decoupling (line 3). Parallel improvements are also observed under both random and FPS paradigms.

We further compare our uniform sampling against two established baselines: random sampling and FPS. Compared to random sampling, the uniform sampling yields consistent improvements, indicating that explicitly maintaining global semantics outperforms indiscriminate retention. Notably, while FPS yields a marginally higher Top-1 EM (29.16), it results in a severe drop in CIDEr (95.01) compared to the uniform strategy (98.52). We attribute this to the inherent mechanics of FPS: by greedily sampling maximally distant tokens to enforce strict diversity, it tends to over-emphasize local semantics while severely disrupting the global semantics. This omission undermines the comprehensive scene context, which is essential for CIDEr evaluation.

In contrast, the proposed uniform selection strikes the optimal balance. By strategically preserving broad global semantics without sacrificing local semantics, it ensures highly competitive Top-1 accuracy while delivering robust CIDEr performance.

2) Effectiveness of the Guidance Strategy of LGS-HTP module: To validate the effectiveness of the guidance strategy employed during the pruning process, we evaluate an unguided pruning variant that derives pruning decisions directly from the native tokens $\mathcal { T } _ { \mathrm { N } }$ . The corresponding performance results are reported in Table XII:

TABLE XII  
PERFORMANCE COMPARISON BETWEEN THE UNGUIDED PRUNING VARIANT AND OUR PROPOSED GUIDANCE STRATEGY ON THE SCANQA VALIDATION SET. THE BEST RESULTS AMONG THE PRUNED VARIANTS FOR EACH PRUNING RATIO ARE HIGHLIGHTED IN BOLD.
<table><tr><td>Guidance</td><td>Pruning Ratio</td><td>Top-1 EM</td><td>CIDEr</td></tr><tr><td>X</td><td>80%</td><td>29.1551</td><td>98.4722</td></tr><tr><td>√</td><td>80%</td><td>29.7540</td><td>99.5614</td></tr><tr><td>×</td><td>90%</td><td>27.7219</td><td>97.3155</td></tr><tr><td>√</td><td>90%</td><td>28.6203</td><td>98.5228</td></tr></table>

As shown in Table XII, the proposed guidance strategy consistently outperforms the unguided variant across different pruning ratios. These results indicate that deriving pruning decisions exclusively from the native visual tokens (T ) neglects the inherent semantic requirements of the LVLM. By leveraging the projected tokens (T ) for guidance, our method effectively aligns the retained visual features with the model’s reasoning demands, thereby mitigating performance degradation even at high compression rates.

3) Computational Efficiency Analysis: To validate the feasibility of the proposed TP-JSCC in a cloud-edge collaborative system, this subsection reports the computational complexity of each module within TP-JSCC and the complexity of the LVLM after applying TP-JSCC. The detailed experimental results are presented in Table XIII:

TABLE XIII  
COMPUTATIONAL COMPLEXITY (IN GFLOPS) OF DIFFERENT MODULES AND THE LVLM UNDER VARIOUS COMPRESSION RATIOS.
<table><tr><td>TP-JSCC</td><td>Compression Ratio</td><td>LVLM</td><td></td><td></td><td>LGS-HTP T-JSCC Total Saving</td></tr><tr><td>X</td><td></td><td>93255.12</td><td></td><td></td><td></td></tr><tr><td>√</td><td>90×</td><td>19049.47</td><td>626.49</td><td>3.94</td><td>78.90%</td></tr><tr><td>√</td><td>180×</td><td>9659.90</td><td>626.49</td><td>1.92</td><td>88.97%</td></tr></table>

As shown in Table XIII, our proposed method effectively offloads the computationally intensive LVLM to the cloud server, incurring minimal overhead on the edge device (i.e., the combined complexity of LGS-HTP and T-JSCC is strictly bounded below 633 GFLOPs, which can be easily managed by an edge GPU like the NVIDIA Jetson TX2). Furthermore, the pruning mechanism drastically reduces the inference cost of the LVLM, which significantly lightens the computational burden on the cloud server. Notably, the 90× and 180× total compression ratios are achieved by cascading token pruning (10× for 90% pruning, 5× for 80%) with T-JSCC channel reduction (18×, from 1152 to 64 channels), which simultaneously eases the wireless transmission pressure. Overall, our method achieves nearly 80% and 90% total computational savings under these two compression settings, respectively.

## VI. CONCLUSION

This paper focuses on a novel token-oriented CE-CP to address the severe computational constraints of EAs and the communication bottlenecks in 3D-VQA tasks over errorprone communication environments. Specifically, we proposed a highly efficient two-stage compression strategy comprising an LGS-HTP module and a T-JSCC module. By leveraging foreground-background priors and a joint selection mechanism for local and global semantic tokens, the LGS-HTP module effectively mitigates semantic redundancy. Subsequently, the T JSCC module achieves robust, channel-aware token transmission by using an all-linear cascaded architecture and an SNRadaptive mechanism. Extensive evaluations on the ScanQA dataset indicate the exceptional efficiency and robustness of the proposed method. Notably, our method reduces the edgeto-cloud transmission payload by approximately 180×. The LGS-HTP module incurs less than a 1% CIDEr loss at a 90% pruning ratio, and remarkably, under extreme channel noise (SNR = 0 dB), the end-to-end CIDEr score drops by a mere 4.9% with the help of the proposed T-JSCC module, while saving 90% computational complexity for the LVLM. Our work provides a highly efficient and resilient tokenlevel transmission solution, paving the way for the robust deployment of LVLM in resource-constrained EAs.

## REFERENCES

[1] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning, 2023, pp. 2165–2183.

[2] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23 716–23 736, 2022.

[3] W. Shi, J. Cao, Q. Zhang, Y. Li, and L. Xu, “Edge computing: Vision and challenges,” IEEE internet of things journal, vol. 3, no. 5, pp. 637–646, 2016.

[4] J. Ying, Z. Qin, Y. Feng, L. Wang, and X. Tao, “Joint semantic-channel coding and modulation for token communications,” IEEE Transactions on Wireless Communications, vol. 25, pp. 8179–8193, 2025.

[5] A. Devoto, J. Pomponi, M. Merluzzi, P. Di Lorenzo, and S. Scardapane, “Adaptive semantic token communication for transformer-based edge inference,” IEEE Transactions on Machine Learning in Communications and Networking, 2026.

[6] A. Devoto, J. Pomponi, S. Petruzzi, P. Di Lorenzo, and S. Scardapane, “Adaptive semantic token selection for ai-native goal-oriented communications,” in 2024 IEEE Globecom Workshops (GC Wkshps), 2024, pp. 1–6.

[7] L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, “Token communications: A large model-driven framework for cross-modal context-aware semantic communications,” IEEE Wireless Communications, vol. 32, no. 5, pp. 80–88, 2025.

[8] D. Azuma, T. Miyanishi, S. Kurita, and M. Kawanabe, “Scanqa: 3d question answering for spatial scene understanding,” in proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 19 129–19 139.

[9] X. Ma, S. Yong, Z. Zheng, Q. Li, Y. Liang, S.-C. Zhu, and S. Huang, “SQA3d: Situated question answering in 3d scenes,” in The Eleventh International Conference on Learning Representations, 2023.

[10] Z. Zhu, X. Ma, Y. Chen, Z. Deng, S. Huang, and Q. Li, “3d-vista: Pretrained transformer for 3d vision and text alignment,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2911–2921.

[11] Y. Man, L.-Y. Gui, and Y.-X. Wang, “Situational awareness matters in 3d vision language reasoning,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 678–13 688.

[12] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3d-llm: Injecting the 3d world into large language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 20 482–20 494, 2023.

[13] Z. Zhu, Z. Zhang, X. Ma, X. Niu, Y. Chen, B. Jia, Z. Deng, S. Huang, and Q. Li, “Unifying 3d vision-language understanding via promptable queries,” in European Conference on Computer Vision, 2024, pp. 188– 206.

[14] W. Mo and Y. Liu, “Bridging the gap between 2d and 3d visual question answering: A fusion approach for 3d vqa,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 5, 2024, pp. 4261– 4268.

[15] S. Chen, X. Chen, C. Zhang, M. Li, G. Yu, H. Fei, H. Zhu, J. Fan, and T. Chen, “Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 428–26 438.

[16] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud et al., “Openeqa: Embodied question answering in the era of foundation models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 16 488–16 498.

[17] B. Liu, Y. Dong, Y. Wang, Z. Ma, Y. Tang, L. Tang, Y. Rao, W.-C. Ma, and R. Krishna, “Coarse correspondences boost spatial-temporal reasoning in multimodal language model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 3783–3792.

[18] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, and C. Li, “LLaVA-onevision: Easy visual task transfer,” Transactions on Machine Learning Research, 2025.

[19] F. WANG, S. Yu, J. Wu, J. Tang, H. Zhang, and Q. Sun, “3d question answering via only 2d vision-language models,” in Forty-second International Conference on Machine Learning, 2025.

[20] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25 278–25 294, 2022.

[21] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[22] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.

[23] Y. Liang, C. GE, Z. Tong, Y. Song, J. Wang, and P. Xie, “EVit: Expediting vision transformers via token reorganizations,” in International Conference on Learning Representations, 2022.

[24] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[25] J. Cao, P. Ye, S. Li, C. Yu, Y. Tang, J. Lu, and T. Chen, “Madtp: Multimodal alignment-guided dynamic token pruning for accelerating visionlanguage transformer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 15 710–15 719.

[26] K. Huang, H. Zou, Y. Xi, B. Wang, Z. Xie, and L. Yu, “Ivtp: Instructionguided visual token pruning for large vision-language models,” in European conference on computer vision, 2024, pp. 214–230.

[27] X. Ye, Y. Gan, Y. Ge, X.-P. Zhang, and Y. Tang, “Atp-llava: Adaptive token pruning for large vision language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 24 972–24 982.

[28] W. Ye, Q. Wu, W. Lin, and Y. Zhou, “Fit and prune: Fast and trainingfree visual token pruning for multi-modal large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 21, 2025, pp. 22 128–22 136.

[29] D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman, “Token merging: Your vit but faster,” in The Eleventh International Conference on Learning Representations, 2023.

[30] H.-W. Huang, F.-C. Chen, W. Chai, C.-C. Su, L. Xia, S. Jung, C.-Y. Yang, J.-N. Hwang, M. Sun, and C.-H. Kuo, “Zero-shot 3d question answering via voxel-based dynamic token compression,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 19 424–19 434.

[31] Y. Jiang, Q. Wu, W. Lin, W. Yu, and Y. Zhou, “What kind of visual tokens do we need? training-free visual token pruning for multi-modal large language models from the perspective of graph,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 4, 2025, pp. 4075–4083.

[32] Q. Zhang, M. Liu, L. Li, M. Lu, Y. Zhang, J. Pan, Q. She, and S. Zhang, “Beyond attention or similarity: Maximizing conditional diversity for token pruning in MLLMs,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[33] S. R. Alvar, G. Singh, M. Akbari, and Y. Zhang, “Divprune: Diversitybased visual token pruning for large multimodal models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9392–9401.

[34] Y. Shang, M. Cai, B. Xu, Y. J. Lee, and Y. Yan, “Llava-prumerge: Adaptive token reduction for efficient large multimodal models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 22 857–22 867.

[35] Q. Zhang, A. Cheng, M. Lu, R. Zhang, Z. Zhuo, J. Cao, S. Guo, Q. She, and S. Zhang, “Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 20 857–20 867.

[36] Z. Wen, Y. Gao, W. Li, C. He, and L. Zhang, “Token pruning in multimodal large language models: Are we solving the right problem?” in Findings ofthe Associationfor Computational Linguistics: ACL 2025, 2025, pp. 15 537–15 549.

[37] C. E. Shannon, “A mathematical theory of communication,” The Bell system technical journal, vol. 27, no. 3, pp. 379–423, 1948.

[38] E. Bourtsoulatze, D. Burth Kurka, and D. Gund¨ uz, “Deep joint source-¨ channel coding for wireless image transmission,” IEEE Transactions on Cognitive Communications and Networking, vol. 5, no. 3, pp. 567–579, 2019.

[39] Z. Chen, H. Yuan, H. A. S. Mekki, M. M. Hassan, and G. Zhang, “Deepjscc-pcg: Deep joint source channel coding for point cloud geometry,” IEEE Signal Processing Letters, 2025.

[40] H. A. Sirag Mekki, H. Yuan, M. M. Hassan, and Z. Chen, “Deepjsccbased latent space power control for robust and efficient 3d point cloud transmission,” ETRI Journal, 2026.

[41] D. Gund¨ uz, M. A. Wigger, T.-Y. Tung, P. Zhang, and Y. Xiao, “Joint¨ source–channel coding: Fundamentals and recent progress in practical designs,” Proceedings of the IEEE, 2024.

[42] D. B. Kurka and D. Gund ¨ uz, “Deepjscc-f: Deep joint source-channel¨ coding of images with feedback,” IEEE journal on selected areas in information theory, vol. 1, no. 1, pp. 178–193, 2020.

[43] T.-Y. Tung and D. Gund ¨ uz, “Deepwive: Deep-learning-aided wireless¨ video transmission,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2570–2583, 2022.

[44] J. Shao, Y. Mao, and J. Zhang, “Learning task-oriented communication for edge inference: An information bottleneck approach,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 197–211, 2021.

[45] M. Jankowski, D. Gund¨ uz, and K. Mikolajczyk, “Deep joint source-¨ channel coding for wireless image retrieval,” in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 5070–5074.

[46] C. Tian, Z. Li, H. Yuan, R. Hamzaoui, L. Shen, and S. Kwong, “Feature compression for cloud-edge multimodal 3d object detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[47] Z. Li, C. Tian, H. Yuan, X. Lu, and H. Malekmohamadi, “3d-msfc: A 3d multi-scale features compression method for object detection,” Displays, vol. 85, p. 102880, 2024.

[48] A. E. Mohr, E. A. Riskin, and R. E. Ladner, “Unequal loss protection: Graceful degradation of image quality over packet erasure channels through forward error correction,” IEEE journal on selected areas in communications, vol. 18, no. 6, pp. 819–828, 2002.

[49] M. Bar, “Visual objects in context,” Nature Reviews Neuroscience, vol. 5, no. 8, pp. 617–629, 2004.

[50] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention, 2015, pp. 234–241.

[51] F. Murtagh and P. Legendre, “Ward’s hierarchical agglomerative clustering method: which algorithms implement ward’s criterion?” Journal of classification, vol. 31, no. 3, pp. 274–295, 2014.

[52] S. Lloyd, “Least squares quantization in pcm,” IEEE transactions on information theory, vol. 28, no. 2, pp. 129–137, 1982.

[53] D. R. Pauluzzi and N. C. Beaulieu, “A comparison of snr estimation techniques for the awgn channel,” IEEE Transactions on communications, vol. 48, no. 10, pp. 1681–1691, 2002.

[54] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” arXiv preprint arXiv:1412.6572, 2014.

[55] F. Liu, Y. Tian, Y. Chen, Y. Liu, V. Belagiannis, and G. Carneiro, “Acpl: Anti-curriculum pseudo-labelling for semi-supervised medical image classification,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 20 697–20 706.