# PDNet: Exploring Pruning and Distillation for Real-Time Worker Activity Recognition in Industrial Assembly

Malek Baba<sup>1</sup> <sup>a</sup>, Thinesh Thiyakesan Ponbagavathi<sup>2</sup> <sup>b</sup> and Alina Roitberg <sup>3</sup> <sup>c</sup>

<sup>1</sup>Stuttgart University, Stuttgart, Germany

<sup>2</sup>Institute for Artificial Intelligence (KI), Stuttgart University, Stuttgart, Germany

<sup>3</sup>Intelligent Assistive Systems (IAS) group, University of Hildesheim, Hildesheim, Germany Malek.baba@outlook.com, thinesh.thiyakesan-ponbagavathi@ki.uni-stuttgart.de, roitberg@uni-hildesheim.de

Keywords: Resource-Efficient Deep Learning, Network Pruning, Knowledge Distillation, 3D Convolutional Neural Networks, Human Activity Recognition

Abstract: In this work, we address real-time fine-grained human activity recognition (HAR) in industrial assembly and introduce PDNet, a new framework that jointly exploits structured pruning and knowledge distillation for 3D convolutional networks. We start by training a 3D-CNN backbone in a supervised manner. Then, PDNet applies channel-wise structured pruning to obtain a more compact network. In the third step, our method is optimized using knowledge distillation from the large unpruned teacher to recover accuracy lost due to pruning. On the challenging MECCANO dataset with an X3D-M backbone, the pruned model without distillation reaches a Top-1 accuracy of 25.54 %, while PDNet raises this to 39.46 %, with only 6.90 MB model size and 3.78 GFLOPs. These results show that distillation within PDNet is highly effective in restoring pruned model performance and highlight its high potential for resource-constrained HAR in industrial assembly.

## 1 INTRODUCTION

In the context of modern manufacturing, often framed under the paradigm of Industry 4.0, artificial intelligence and, in particular, deep learning have become a key enabler. An important element of such systems is the real-time analysis of worker activities, which enables the quantification and evaluation of human performance during complex assembly operations (Tao, Lai, Leu, & Yin, 2018; Chen, Zendehdel, Leu, & et al., 2022). By accurately recognizing and predicting worker actions, these intelligent systems can provide immediate assistance and guidance, thereby optimizing assembly processes and ensuring adherence to standard operating procedures. This capability is also essential for fostering effective human-robot collaboration (Al-Amin et al., 2020) and can significantly reduce operational errors, directly advancing the core objectives of smart manufacturing (Chen, Leu, Tao, & Yin, 2020).

Deep learning-based human activity recognition models, for example architectures based on 3D convolutional neural networks (3D CNNs) (Ji, Xu, Yang, & Yu, 2013), play a key role in this context. Such models can jointly capture spatial appearance and temporal dynamics from video streams, enabling accurate identification of complex actions (Ji et al., 2013). Their ability to model complex motion patterns makes them especially suitable for industrial settings, where accurately recognizing subtle worker activities is critical (Tran, Bourdev, Fergus, Torresani, & Paluri, 2015).

X3D Model Optimization via Pruning and Distillation  
![](images/d5eeaab6e1949645279294ebbc8a01fd572f032e71df196cd3491515d952258a.jpg)  
Figure 1: Accuracy-latency-model size trade-off for finegrained industrial activity recognition on the MECCANO dataset. The X-axis shows CPU inference time per clip (ms) and the Y-axis Top-1 accuracy, the circle area encodes model size. Our pruned X3D with knowledge distillation (PDNet, red) recovers most of the accuracy lost by the pruned baseline (purple).

However, by extending convolutions into the temporal dimension, 3D architectures typically require substantially more parameters and computations than their 2D counterparts, which complicates deployment under tight resource constraints. Standard architectures such as I3D-ResNet50 exceed 25 GFLOPs, which makes them unsuitable for real-time or edge applications (Hedegaard & Iosifidis, 2022). Efficient architectural designs, such as lightweight 3D backbones, aim to balance computational cost and accuracy (Feichtenhofer, 2020). Additional strategies, including model quantization and pruning, are often applied to reduce model size and inference latency, which supports deployment on resource-constrained embedded systems and FPGA accelerators (Toupas, Bouganis, & Tzovaras, 2023). Together, these techniques are crucial for achieving real-time operation in industrial environments (Toupas et al., 2023). In addition to architecture-level and compression strategies, input-level efficiency has also been explored. For example, Noor et al. (Noor & Park, 2023) propose a lightweight skeleton-based 3D-CNN that operates directly on sequences of human joint keypoints rather than RGB frames. By using pose sequences as input, the network drastically reduces input dimensionality and removes irrelevant background information, enabling a compact 3D-CNN to achieve real-time inference with low latency while maintaining strong accuracy for fall detection and general action recognition. At the same time, another compression technique, knowledge distillation, has gained traction across many domains and has been explored to improve accuracy of heavily compressed models for driver activity recognition (Tanama, Peng, Marinov, Stiefelhagen, & Roitberg, 2023). Yet, there is little work that explores pruning and knowledge distillation for 3D CNNs in general activity recognition as well as in the industrial assembly domain: most existing efforts either target 2D architectures or apply compression techniques in isolation.

To close this gap, we introduce a new framework that jointly leverages structured pruning and knowledge distillation on top of a 3D-CNN for fine-grained recognition of industrial assembly activities, enabling model compression while effectively recovering the accuracy typically lost during pruning. Overall, the key contributions of this work are as follows:

• We first establish a strong baseline by benchmarking several compact 3D CNN architectures—including 3D-MobileNet (Kop¨ ukl¨ u, Kose,¨ Gunduz, & Rigoll, 2021) and X3D variants (X3D-M, X3D-L) (Feichtenhofer, 2020)—on the MEC-CANO dataset for fine-grained industrial assembly recognition. This analysis compares accuracy, inference time, and model size under realistic deployment constraints and identifies X3D-M as a promising starting point for further optimization.

• Introducing Structured Pruning for 3D-CNN Efficiency: To the best of our knowledge, we are the first to explore structured pruning for efficiency optimization of 3D-CNNs in industrial assembly scenarios. Using L1-norm-based channel pruning at multiple sparsity levels, we significantly reduce computational cost and memory usage while quantifying the corresponding accuracy degradation. This step reveals both the potential and the limitations of pruning-only compression for fine-grained 3D video understanding.

• PDNet: Pruning-Distillation Method for Fine-Grained Industrial HAR: Building on the pruning analysis, we introduce PDNet, a framework that couples L1-norm–based structured pruning of X3D-M with knowledge distillation from a highcapacity SlowFast teacher (Feichtenhofer, Fan, Malik, & He, 2018). Within PDNet, pruning reduces parameters and FLOPs, while distillation is explicitly used to recover the accuracy lost by compression, resulting in a compact 3D-CNN that maintains competitive performance for finegrained recognition of industrial assembly activities under resource constraints.

## 2 Related Work

Recent studies have employed a combination of lightweight model architectures, model compression techniques, and hardware optimization strategies to achieve improved performance. Lightweight 2D CNN-based networks such as MobileNet (Prathima et al., 2025) and EfficientNet (Narayan et al., 2023) are frequently adopted for their ability to maintain reasonable accuracy while significantly reducing computational load, making them suitable for deployment on edge devices. Model compression techniques are also widely applied; many existing approaches focus on reducing the memory footprint and latency of deep neural networks while preserving high classification accuracy. One key strategy involves model compression techniques, such as pruning and quantization. Structured pruning methods eliminate entire filters or channels, which align better with hardware execution patterns compared to unstructured sparsity (He & Xiao, 2024). Techniques like Dep-Graph (Fang, Ma, Song, Mi, & Wang, 2023) further extend structured pruning by incorporating graph representations of neural architectures, enabling finegrained control over network topology. Quantization reduces model precision (e.g., from 32-bit floatingpoint to 8-bit integers), and post-training quantization (PTQ) and quantization-aware training (QAT) approaches have shown promising results for maintaining accuracy on industrial workloads (Novac et al., 2021). Knowledge distillation (KD) compresses large teacher models into lightweight student models without significant loss in accuracy (Hinton, Vinyals, & Dean, 2015), with recent work exploring its combination with quantization for additional efficiency gains (Tanama et al., 2023).

## 3 METHODOLOGY

We introduce PDNet, a unified optimization framework designed to obtain resource-efficient 3D-CNNs for fine-grained industrial activity recognition. PDNet consists of three components that operate sequentially.

First, we train a compact 3D convolutional neural network (3D-CNN) as our baseline model, described in Section 3.1.1. This backbone provides the initial spatiotemporal feature extractor from which all subsequent optimizations are derived. Second, we apply L1-norm-based structured pruning (Section 3.1.2) to reduce the parameter count and computational complexity of the baseline model. This step produces a compact student network with substantially lower resource demands but also introduces a measurable drop in recognition accuracy. Finally, we incorporate knowledge distillation (Section 3.1.3), using a highcapacity teacher model to guide the training of the pruned student. This distillation phase recovers much of the performance lost during pruning, resulting in an efficient 3D-CNN that maintains competitive accuracy under strict deployment constraints.

## 3.1 PDNet: Pruning and Distillation for Recognition of Assembly Activities

## 3.1.1 Backbone 3D-CNN

First, we select an efficient 3D-CNN backbone. To this end, we benchmark several lightweight 3D architectures on the MECCANO dataset, focusing on the trade-off between accuracy, parameter count, and inference cost. 3D-MobileNet (Kop¨ ukl¨ u et al., 2021)¨ adapts the MobileNetV2 design by utilizing depthwise separable convolutions, which factorize standard 3D kernels to drastically reduce floating-point operations. Alternatively, the X3D family (Feichtenhofer, 2020) optimizes efficiency through a multi-axis expansion strategy, scaling dimensions such as frame rate, duration, and network width independently rather than uniformly. X3D offers a spectrum of variants ranging from the ultra-compact X3D-XS (Extra Small) and X3D-S, designed for edge devices, to larger models like X3D-M, providing a superior trade-off between accuracy and computational complexity compared to traditional heavy models. Based on this empirical comparison describt in the experiments section, we adopt X3D-M as our baseline 3D-CNN. This model is first optimized in a supervised fashion using the standard cross-entropy loss, and then serves as the starting point for the structured pruning and knowledge distillation stages of PDNet.

## 3.1.2 Structured Pruning

The second stage of PDNet applies structured pruning to reduce model complexity while preserving a hardware-friendly network structure (Fang et al., 2023). Structured pruning accelerates neural networks by removing groups of parameters such as channels or filters, while maintaining structural integrity for efficient inference (Fang et al., 2023). Often, the importance of groups is measured by the L1-norm of their weights, which promotes sparsity by encouraging the sum of absolute values of weights to be small.

A central concept is the pruning ratio p, representing the fraction of channels or dimensions removed from each layer. When applied symmetrically to both input and output channels of a convolutional layer, the parameter reduction is approximately

$$
\mathrm { P a r a m e t e r } \mathrm { R e d u c t i o n } = 1 - ( 1 - p ) ^ { 2 } .
$$

For example, a pruning ratio of $p = 0 . 3 \ ( 3 0 \% )$ corresponds to around a 51% reduction in parameters.

Algorithm 1: Baseline Selection, Pruning, and   
Knowledge Distillation Fine-Tuning   
Given: candidate baseline models   
$\mathcal { F } = \{ f _ { 1 } , f _ { 2 } , \ldots , f _ { k } \}$ , teacher model   
$f _ { \mathrm { t e a c h e r } } ,$ training set   
$X = \{ ( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { N } , y _ { N } ) \}$   
foreach $f _ { j } \in \mathcal { F }$ do   
Train $f _ { j }$ on X   
Select model with best accuracy as $f _ { \mathrm { b a s e } }$   
initialize $f _ { \mathrm { p r u n e d } }  f _ { \mathrm { b a s e } }$   
initialize pruning ratio $r  r _ { \mathrm { s t a r t } }$   
while performance targets (size, latency,   
$G F L O P S )$ not met and $r \leq r _ { m a x }$ do   
f<sub>pruned</sub> $ \mathrm { p r u n e } ( f _ { \mathrm { b a s e } } , r )$   
increase pruning ratio $r  r + \Delta r$   
Update $f _ { \mathrm { p r u n e d } }$ by minimizing   
$L _ { \mathrm { K D } } ( f _ { \mathrm { t e a c h e r } } ( x ) , f _ { \mathrm { p r u n e d } } ( x ) , y )$ (aka   
knowledge distillation)   
return f<sub>pruned</sub> as final optimized model

## 3.1.3 Knowledge Distillation

In the third stage of PDNet, we employ knowledge distillation to recover the accuracy lost through structured pruning of the X3D-M backbone. Knowledge distillation is a model compression technique in deep learning where a large, pre-trained teacher network transfers its knowledge to a smaller student network, enabling the student to achieve competitive accuracy with fewer parameters and lower computational requirements (Gou, Yu, Maybank, & Tao, 2021). The teacher generates softened probability outputs through a temperature-controlled softmax function:

$$
p _ { i } = \frac { \exp ( z _ { i } / T ) } { \sum _ { j } \exp ( z _ { j } / T ) } ,
$$

where $z _ { i }$ denotes the teacher logits and $T > 1$ is the temperature that smooths the distribution to reveal dark knowledge about inter-class relationships (Gou et al., 2021). The student is trained to minimize a combination of the standard supervised loss and a distillation loss, which encourages it to mimic the teacher’s softened output distribution. Formally, the loss function is expressed as

$$
\begin{array} { r } { \mathcal { L } = \boldsymbol { \alpha } \cdot \mathrm { C E } ( \mathbf { y } , \hat { \mathbf { y } } _ { s } ) + ( 1 - \boldsymbol { \alpha } ) \cdot T ^ { 2 } \cdot \mathrm { K L } ( p _ { t } ( T ) \| p _ { s } ( T ) ) , } \end{array}
$$

where α balances the contribution of the ground-truth labels and teacher outputs, and $p _ { t } , p _ { s }$ represents the teacher’s and student’s softened predictions respectively (Gou et al., 2021).

Our method is summarized in Algorithm 1.

## 3.2 Implementation Details

The experiments were conducted using the PyTorch framework with the MECCANO dataset (Ragusa, Furnari, & Farinella, 2023) as the primary benchmark. We trained 3D convolutional models such as 3D-MobileNet and X3D variants using cross-entropy loss and the SGD optimizer, with a base learning rate of 0.1, batch sizes of 12, 32, and 15 respectively, and a cosine annealing schedule over 120 epochs. Based on a trade-off analysis involving recognition accuracy, model size, and inference speed, the X3D-M model was selected as the optimal baseline. Model compression was performed using a structured pruning approach implemented via the Torch-Pruning library (Fang et al., 2023), applying an L1-norm channel importance criterion at varying sparsity ratios. Each pruned model was then fine-tuned to recover part of the accuracy loss. To further mitigate the performance degradation introduced by pruning, we applied knowledge distillation using the pre-trained SlowFast model released by the MECCANO dataset authors as the teacher network and the pruned X3D-M model as the student. The student model was trained with a distillation temperature $T = 4$ and balancing coefficient ${ \bf \mathrm { Q } } = 0 . 7 .$ , using a batch size of 16, a base learning rate of 0.1, and the SGD optimizer for 180 epochs. The final optimized model achieved substantial reductions in computational cost while maintaining competitive accuracy suitable for real-time inference.

## 4 EXPERIMENTS

## 4.1 Benchmark & Metrics Details

The models were benchmarked on the MECCANO Dataset. MECCANO (Ragusa et al., 2023) is a egocentric dataset designed to model human-object interactions during mechanical assembly. It captures more than 20 hours of video from a first-person perspective, where users manipulate tools and parts to build a small mechanical system. The dataset features fine-grained activity labels and multiple annotators to ensure reliability. This dataset is unique for capturing 61 fine-grained action classes involved in complex assembly procedures and is characterized by a challenging long-tailed class distribution and high intra-class variability.

Accuracy relied on standard classification metrics: Top-1 Test Accuracy (%) and Top-5 Test Accuracy (%). To accurately assess robustness across all activity types, particularly the rare ones, Mean Per-Class Accuracy was also calculated, addressing the inherent bias posed by the imbalanced class distribution.

Computational efficiency was evaluated using three metrics: model size (MB), inference time (ms), and computational cost (GFLOPs). The model size was determined by measuring the disk size of the saved .pth file. The inference time for a single video sample consisting of 16 frames was measured using the PyTorch profiler by averaging the forward-pass latency over 1000 iterations. The theoretical computational complexity, expressed in GFLOPs (billions of floating-point operations), represents the total number of operations required for a single forward pass of the model. All measurements were conducted on an AMD EPYC 7502P 32-core CPU.

## 4.2 Results

Baseline Model Selection Several compact 3D convolutional architectures, including 3D-MobileNet, X3D-L, and X3D-M, were evaluated using the PySlowFast framework against the high-capacity reference model SlowFast (Ragusa et al., 2023), which achieved the highest accuracy (45.16% Top-1). Among the lightweight candidates, 3D-MobileNet achieved 27.31% Top-1 and X3D-L achieved 34.50% Top-1. Consequently, the X3D-M model was selected as the optimal baseline for subsequent optimization because it achieved the highest Top-1 accuracy (36.49%) among compact models.

Table 1: Comparison of baseline models on the Meccano dataset against SlowFast Model
<table><tr><td>Model</td><td>Top-1 test (%)</td><td>Top-5 test (%)</td></tr><tr><td>SlowFast</td><td>45.16</td><td>73.75</td></tr><tr><td>3D MobileNet</td><td>27.31</td><td>50.51</td></tr><tr><td>X3D-L</td><td>34.50</td><td>63.09</td></tr><tr><td>X3D-M</td><td>36.49</td><td>65.64</td></tr></table>

Pruning Experiments To evaluate the effect of structured pruning on the baseline X3D-M model, we gradually increased the pruning ratio from no pruning up to 25%. At each pruning ratio, the least important filters were removed to reduce model complexity while retaining as much representational capacity as possible. The resulting models were analyzed in terms of model size, latency, and computational cost (GFLOPs). The experiments showed that as the pruning ratio increased, model size and computational cost decreased, but this was accompanied by some degradation in accuracy. For example, pruning up to 25% reduced the model size by approximately 41% while maintaining a reasonable level of performance, highlighting the trade-off between efficiency and predictive capability.

Table 2: Pruning Results Summary
<table><tr><td>Ratio (%)</td><td>Size (MB)</td><td>Time (ms) GFLOPs</td></tr><tr><td>0.0</td><td>21.70</td><td>107.72 6.20</td></tr><tr><td>5.0</td><td>10.75</td><td>98.32 5.65</td></tr><tr><td>10.0</td><td>9.71</td><td>83.75 5.14</td></tr><tr><td>15.0</td><td>8.73</td><td>83.33 4.62</td></tr><tr><td>20.0</td><td>7.78</td><td>81.17 4.18</td></tr><tr><td>25.0</td><td>6.90</td><td>80.77 3.78</td></tr></table>

Main Results: Post-Pruning Accuracy Recovery through Knowledge Distillation The subsequent phase, focused on accuracy recovery, utilized Knowledge Distillation (KD) to fine-tune the highly compressed X3D-M student model using the highcapacity SlowFast teacher model. This combined Pruned + KD strategy was remarkably successful, yielding a final model that achieved a Top-1 accuracy of 39.46% and a Top-5 accuracy of 68.72%. Crucially, this optimization not only recovered the significant performance drop caused by pruning (which reduced Top-1 accuracy from 35.40% to 25.54%) but also enabled the student model to surpass the performance of the original uncompressed X3D-M baseline model by 4.06 percentage points (39.46% vs. 35.40%). However, the class-based accuracy analysis presented a more nuanced view: the overall mean class accuracy slightly declined from 12.01% (original X3D-M) to 10.95% (Pruned + KD model), a decrease of 1.06%. This divergence between high Top-1 accuracy and low mean per-class accuracy is largely due to the highly imbalanced, long-tailed class distribution of the 61 fine-grained action classes present in the MECCANO dataset, where many rare actions are difficult to classify. Despite the slight drop in mean performance, KD significantly boosted accuracy for several key frequent classes, such as check booklet (improving by +11.68%) and tighten bolt with hands (+4.54%). Conversely, some visually or temporally complex fine-grained actions, such as fit rim tire (−17.65%) and put red 4 perforated junction bar (−20.00%), saw decreases, highlighting the limitations of the reduced model capacity for certain tail classes. Overall, the resulting Pruned + KD model maintained the efficiency of the pruned version (6.90 MB, 3.78 GFLOPs) while providing a substantial performance recovery, confirming the efficacy of this joint optimization approach for practical, real-time HAR deployment as shown in figure 1.

Table 3: Comparison of the different variants of the X3D-M model. The pruning and fine-tuning approach achieves large efficiency gains while maintaining competitive accuracy.
<table><tr><td colspan="3">Accuracy</td><td colspan="3">Computational Efficiency</td></tr><tr><td>Model Variant</td><td>Top-1 (%)</td><td>Top-5 (%)</td><td>Size (MB)</td><td>Time (ms)</td><td>GFLOPs</td></tr><tr><td>Original X3D-M</td><td>36.49</td><td>65.64</td><td>21.70</td><td>107.72</td><td>6.20</td></tr><tr><td>Pruned Only</td><td>25.54</td><td>58.64</td><td>6.90</td><td>80.77</td><td>3.78</td></tr><tr><td>PDNet (Ours)</td><td>39.46</td><td>68.72</td><td>6.90</td><td>80.77</td><td>3.78</td></tr></table>

## 5 CONCLUSIONS

In this work, we proposed a resource-efficient optimization pipeline for 3D-CNNs tailored to real-time human activity recognition in industrial assembly settings. By selecting X3D-M as a compact baseline and applying structured pruning followed by knowledge distillation from a SlowFast teacher model, we achieved substantial reductions in model size and computational cost while improving Top-1 accuracy beyond the original model. The final compressed model maintains strong performance at only 3.78 GFLOPs and 6.90 MB, demonstrating the effectiveness of combining pruning and distillation for deployment in resource-constrained Industry 4.0 environments. While the current results are promising, further experiments are needed to fully explore the potential of this optimization pipeline. Future work could investigate alternative student-teacher combinations to identify more effective distillation pairings, evaluate different pruning strategies or sparsity levels, and assess the generalizability of the approach across other datasets or activity domains. Additionally, benchmarking the compressed models on a wider range of hardware platforms, including edge devices and industrial-grade embedded systems, will provide deeper insights into real-world performance and latency characteristics. These investigations will help refine the pipeline and ensure robust, efficient deployment in diverse industrial scenarios.

## Acknowledgements

The authors would like to acknowledge the use of AIassisted tools, including Perplexity AI and ChatGPT, during the preparation of this manuscript. These tools were employed to support literature exploration, writing refinement, and editing.

## References

Al-Amin, A., Qin, R., Tao, W., Doell, D., Lingard, R., Yin, Z., & Leu, M. (2020). Fusing and refining convolutional neural network models for assembly action recognition in smart manufacturing. Proceedings of the Institution of Mechanical Engineers, Part C: Journal ofMechanical Engineering Science, 236, 2046-2059. doi: 10.1177/0954406220931547

Chen, H., Leu, M. C., Tao, W., & Yin, Z. (2020, 11). Design of a real-time human-robot collaboration system using dynamic gestures (Vol. Volume 2B: Advanced Manufacturing). doi: 10.1115/IMECE2020-23650

Chen, H., Zendehdel, N., Leu, M. C., & et al. (2022, December 20). Multi-modal fine-grained activity recognition and prediction in assembly. Research Square Preprint. (Preprint available at Research Square) doi: 10.21203/rs.3.rs -2387286/v1

Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). Depgraph: Towards any structural pruning. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition (pp. 16091–16101).

Feichtenhofer, C. (2020). X3D: expanding architectures for efficient video recognition. CoRR, abs/2004.04730.

Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2018). Slowfast networks for video recognition. CoRR, abs/1812.03982.

Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021, March). Knowledge distillation: A survey. International Journal ofComputer Vision, 129(6), 1789–1819.

He, Y., & Xiao, L. (2024, May). Structured pruning for deep convolutional neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5), 2900–2919.

Hedegaard, L., & Iosifidis, A. (2022). Continual 3d convolutional neural networks for real-time processing of videos. In Computer vision – eccv 2022 (p. 369–385). Springer Nature Switzerland.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.

Ji, S., Xu, W., Yang, M., & Yu, K. (2013). 3d convolutional neural networks for human action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(1), 221-231.

Kop¨ ukl¨ u, O., Kose, N., Gunduz, A., & Rigoll, G.¨ (2021). Resource efficient 3d convolutional neural networks.

Narayan, V., Mall, P. K., Alkhayyat, A. H., Abhishek, K., Kumar, S., & Pandey, P. (2023). Enhancenet: An approach to boost the performance of deep learning model based on real-time medical images. Journal ofSensors.

Noor, N., & Park, I. K. (2023). A lightweight skeleton-based 3d-cnn for real-time fall detection and action recognition. 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2171-2180. Retrieved from https://api.semanticscholar.org/ CorpusID:266370283

Novac, P., et al. (2021). Quantization and deployment of deep neural networks on microcontrollers. Sensors, 21(9), 2984. doi: 10.3390/s21092984

Prathima, C., Mahanandi, Y., Bhargavi, M. M., Teja, G. S., Gowripriya, M., & Sai, M. L. (2025). Real-time identification of medicinal plants using deep learning techniques. 2025 4th International Conference on Sentiment Analysis and Deep Learning (ICSADL), 1061-1067.

Ragusa, F., Furnari, A., & Farinella, G. M. (2023). Meccano: A multimodal egocentric dataset for humans behavior understanding in the industrial-like domain. Computer Vision and Image Understanding (CVIU). doi: https:// doi.org/10.1016/j.cviu.2023.103764

Tanama, C., Peng, K., Marinov, Z., Stiefelhagen, R., & Roitberg, A. (2023). Quantized distillation: Optimizing driver activity recognition models

for resource-constrained environments. 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 5479-5486.

Tao, W., Lai, Z., Leu, M. C., & Yin, Z. (2018). Worker activity recognition in smart manufacturing using imu and semg signals with convolutional neural networks. EasyChair Preprints. doi: 10.29007/bld3

Toupas, P., Bouganis, C.-S., & Tzovaras, D. (2023). Fmm-x3d: Fpga-based modeling and mapping of x3d for human action recognition. In 2023 ieee 34th international conference on application-specific systems, architectures and processors (asap) (p. 119-126). doi: 10.1109/ ASAP57973.2023.00030

Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning spatiotemporal features with 3d convolutional networks.