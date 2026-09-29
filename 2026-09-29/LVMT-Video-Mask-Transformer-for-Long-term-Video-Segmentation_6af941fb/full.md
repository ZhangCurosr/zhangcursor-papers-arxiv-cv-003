# LVMT: Video Mask Transformer for Long-term Video Segmentation

Narges Norouzi<sup>1</sup>

Niccolo Cavagnero\` <sup>1</sup> Gijs Dubbelman<sup>1</sup>

<sup>1</sup>Eindhoven University of Technology

Idil Esen Zulfikar<sup>2</sup> Bastian Leibe<sup>2</sup> Daan de Geus<sup>1</sup>

<sup>2</sup>RWTH Aachen University

## Abstract

Existing online video segmentation methods struggle to track objects in long, complex videos with long-term occlusions. We hypothesize that this limitation is caused by (i) the inability of their temporal propagation mechanism to adaptively select the object information that is propagated across time, and (ii) their inability to be trained on long videos due to memory requirements and vanishing gradients. To address the first limitation, we propose to use a lightweight GRU-based temporal propagation module that can learn to select which information it keeps in memory and propagates across time. Second, to allow training on long videos, we leverage Truncated Query Propagation (TQP), a training strategy in which the model processes a video in chunks offrames, where information about tracked objects is propagated between chunks but backpropagation is only conducted in individual chunks, enabling longer temporal supervision without out-of-memory issues, inference overhead, or vanishing gradients. The resulting model is called the Long-term Video Mask Transformer (LVMT). Extensive experiments on six benchmarks show that LVMT sets a new state ofthe art across a range ofvideo segmentation tasks, while retaining the speed of the highly efficient model it is based on, making it 10×faster than the prior state ofthe art. Code: https://www.tue-mps.org/lvmt.

## 1. Introduction

Video segmentation refers to the task of segmenting, classifying, and tracking object instances consistently across all frames of a video sequence. Recent video segmentation approaches VidEoMT [28] and PMT [4] show that large, extensively pre-trained Vision Transformers (ViT) encoders can replace the many complex components that are commonly used in earlier models [20, 44, 45], resulting in much simpler and faster architectures that obtain competitive, stateof-the-art accuracy. VidEoMT employs an encoder-only segmentation model [17] for each video frame, and achieves tracking by propagating query embeddings with information about the previous frame’s objects to the model for the next time step. To allow the pre-trained encoder to be reused and support multiple tasks in parallel, PMT instead leverages a lightweight decoder that is applied on top of a frozen ViT, with temporal modeling working the same as for VidEoMT. Despite their high accuracy and efficiency, these models still struggle with long-term object tracking, just like prior methods. Especially in long, complex videos and under long-term occlusion, their predictions exhibit erroneous identity switches, i.e., inconsistent object identity assignment across frames. The objective of this paper is to improve the accuracy of these models by better modeling long-range temporal information while preserving the simplicity and speed of current efficient architectures.

![](images/4f321483cedddf3b795e78d89af035b6826ad2c15095cdd8b957b2a813a1e802.jpg)  
Figure 1. PMT vs. LVMT. Mean AP ± std. dev. over five runs. Across ViT-L/B/S, LVMT improves AP by at least +4.6 over the efficient PMT [4] baseline at similar FPS. Evaluated on OVIS val.

A key limitation of these existing efficient models is that their temporal query propagation mechanism cannot adaptively select which information about objects it propagates across time. The propagated queries are always the sum of temporally-agnostic learnable queries and per-object query embeddings from the previous frame. In case of long occlusions, information about occluded objects is eventually diluted by the iterative addition of the temporally-agnostic queries, causing the model to struggle to re-identify these occluded objects. A straightforward solution would be to use an explicit external memory bank with the queries for all tracked objects [13, 41], including the occluded ones. However, such a growing query history increases both memory usage and computation with video length, resulting in poor model efficiency when applied to long videos.

Instead of a growing query history, we propose to use a lightweight learnable memory implemented as a GRU cell [9]. In this recurrent module, the model can adaptively select which information from the previous time step it keeps in memory and propagates to the next time step. With this, we hypothesize that the model can learn to keep information about occluded objects in memory when it needs to, allowing for re-identification when these objects reappear.

Although the recurrent unit improves overall performance, we empirically observe that identity switches remain a key failure mode in challenging videos. From our analysis, reported in Sec. 6.1, we find that the model suffers from identity switches on videos that contain many objects, frequent disappearances, and long-term occlusions. Since the model is trained only on short clips, which is the default for PMT [4], such gaps exceed its training horizon, complicating identity recovery at test time. A natural solution would be to train on longer video clips. However, this introduces two practical challenges. First, processing more frames substantially increases memory consumption and can lead to out-of-memory errors. Second, optimizing recurrent updates over long horizons causes vanishing gradients through time, yielding worse performance.

To address these limitations, we adopt Truncated Query Propagation (TQP), a training strategy inspired by TBPTT [39]. With TQP, each video is divided into chunks, and queries are propagated forward across chunk boundaries to preserve temporal context. During backpropagation, gradients are truncated within each chunk, so gradients from temporally distant predictions are not propagated through the full sequence. This allows the model to learn longer-term temporal dependencies during training without suffering from vanishing gradients or memory issues.

Applying the GRU-based memory and TQP to PMT, we present the Long-term Video Mask Transformer (LVMT) model, which better incorporates and preserves long-term temporal information to improve accuracy without compromising efficiency. LVMT offers several advantages: (i) the fixed-size GRU state enables long-range temporal modeling without growing memory usage or computation; (ii) chunkbased training with TQP allows training on arbitrarily long videos without memory issues; (iii) TQP stabilizes gradient propagation through the recurrent GRU updates by truncating the backward pass over time; (iv) as a training-time strategy, TQP improves accuracy without adding inferencetime overhead and without altering the model architecture.

Our comprehensive experimental analysis demonstrates that LVMT consistently outperforms PMT while maintaining nearly identical efficiency across all benchmarks. Notably, on OVIS [31], a benchmark specifically designed to stresstest models under heavy, long-term occlusion, LVMT improves AP by at least +4.6 over PMT across ViT-L/B/S backbones at similar prediction speeds (see Fig. 1). Moreover, LVMT surpasses the prior state-of-the-art method DVIS-DAQ [45] by +2.4 AP while retaining PMT’s efficiency, enabling it to be over 10× faster than DVIS-DAQ. These gains are further validated on other benchmarks for video instance, panoptic, and semantic segmentation in Sec. 5.

In summary, we make the following contributions:

• We introduce a lightweight GRU-based temporal propagation mechanism for video segmentation that captures long-range context while avoiding both growing memory usage over time and excessive computational overhead.

• We leverage Truncated Query Propagation (TQP), which adapts TBPTT with across-chunk gradient accumulation to query propagation, enabling long-video training with bounded peak memory and improved gradient stability.

• The resulting model, LVMT, achieves a stronger performance vs. latency trade-off than prior methods.

## 2. Related Work

Video Segmentation. Current state-of-the-art video segmentation methods [15, 16, 19, 20, 43–45] are universal models, which means that they use a single framework for video instance segmentation (VIS) [42], video panoptic segmentation (VPS) [18], and video semantic segmentation (VSS) [27]. These methods typically follow a decoupled paradigm, using a segmenter for per-frame segmentation and a tracker for temporal association. While effective, these specialized components increase the complexity of the architecture and severely reduce their efficiency.

To improve efficiency without harming accuracy, recent work introduces VidEoMT [28], an encoder-only video segmentation method that replaces complex components with a large, extensively pre-trained ViT encoder and simple temporal query propagation. The follow-up method PMT [4] extends VidEoMT to the setting where the ViT encoder remains frozen and can be reused for various downstream tasks, by introducing a lightweight decoder. While these models are effective, they still struggle with long-term tracking due to the non-adaptive query propagation mechanism and their inability to train effectively on long videos. This work addresses both limitations.

Long-term Temporal Modeling. Several works explore long-term temporal modeling for object tracking and video segmentation. In multiple object tracking, TrackFormer [24] keeps a short-term history of tracked objects and aims to re-identify those in future frames. However, Norouzi et al. [28] show that such propagation is inefficient for video segmentation, as it requires applying non-maximum suppression over masks to remove duplicate queries at each frame.

GenVIS [13] introduces an explicit query memory for VIS. Unlike TrackFormer, it retains object queries indefinitely, enabling long-term association. However, this comes at the cost of increasing memory and computation as video length grows, due to repeated cross-attention over stored queries. Recently, several methods, such as LiVOS [21], XMem [7], and Cutie [8], explored long-term temporal modeling for video object segmentation (VOS). Designed for VOS, they rely on multiple explicit memory banks and iterative memory retrieval, resulting in computational costs that scale poorly with video length and object count and making them unsuitable for efficient application to VIS, VPS, and VSS.

More recently, SAM3 [2] achieves strong performance in promptable video segmentation and tracking. However, it relies on a complex architecture comprising vision and text encoders together with multiple task-specific modules for detection, segmentation, and tracking, many of which are similar to the ones shown to be highly inefficient by Norouzi et al. [28]. Moreover, as the number of tracked entities increases, the model speed consistently decreases.

In contrast, we aim for efficient long-term modeling and propose a lightweight GRU-based propagation mechanism with truncated chunk-based training, enabling long-horizon consistency at the high efficiency of VidEoMT and PMT.

## 3. Preliminaries

Task Definition. This work focuses on online video segmentation: at each time step, the model produces a segmentation mask and category label for each object in the frame, and it re-identifies objects that were detected in previous frames. In this work, we use “object” as a general term that may refer to object instances in VIS, semantic classes in VSS, or both in VPS. Formally, given a video $\mathcal { V } = \{ \mathbf { I } _ { 1 } , \mathbf { I } _ { 2 } , \ldots , \mathbf { I } _ { T } \}$ of T frames, for each frame $\mathbf { I } _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ a model must predict a set of $K _ { t }$ mask-label pairs $\mathcal { V } _ { t } = \{ ( \mathbf { m } _ { t , i } , c _ { t , i } ) \} _ { i = 1 } ^ { K }$ , where $\mathbf { m } _ { t , i } \in \{ 0 , 1 \} ^ { H \times W }$ is a binary mask and $c _ { t , i } \in \{ 1 , \ldots , C \}$ is a class label. Importantly, for video segmentation, the model must not only produce accurate mask-label pairs for each frame, it should also maintain stable correspondences across frames. In particular, a pair $( \mathbf { m } _ { t , i } , c _ { t , i } )$ for object i at timestep t should correspond to the same object as $\left( \mathbf { m } _ { t - 1 , i } , c _ { t - 1 , i } \right)$ for object i at timestep $t - 1$ , effectively tracking an object across time. This must be achieved in an online manner: at timestep $t ,$ the predictions $\mathcal { V } _ { t }$ may only depend on the current frame $\mathbf { I } _ { t }$ and the previously observed frames $\big \{ \mathbf { I } _ { 1 } , \dotsc , \mathbf { I } _ { t - 1 } \big \}$

EoMT, VidEoMT, and PMT. EoMT [17] revisits image segmentation in the era of vision foundation models and shows that the complex components of prior models [5, 6] become largely redundant at increased model and pre-training scale. Therefore, it removes these components, and instead uses only a ViT encoder. Like prior work [3, 6, 37], EoMT operates on a set of K learnable queries $\mathbf { Q } ^ { \mathrm { l r n } } = \{ \mathbf { q } _ { i } ^ { \mathrm { l r n } } \in \mathbb { R } ^ { \bar { D } } \} _ { i = 1 } ^ { K }$ , where each query learns to represent a single object. In EoMT, instead of using complex decoders, these queries are inserted directly into the ViT encoder after the first $L _ { 1 }$ encoder layers, and the remaining $L _ { 2 }$ layers process them together with the patch tokens as a single sequence. Finally, the model produces a segmentation mask and class label for each processed query with a lightweight head. The simple design of EoMT leads to consistent efficiency gains over previous models.

VidEoMT [28] extends this idea to online video segmentation by propagating queries across adjacent frames using a lightweight queryfusion layer. Specifically, the previousframe output queries $\hat { \mathbf { Q } } _ { t - 1 }$ are linearly projected and added to the learnable queries $\mathbf { Q } ^ { \mathrm { { l r n } } }$ before being fed into the last $L _ { 2 }$ layers of the encoder that processes current frame $\mathbf { I } _ { t } \colon$

$$
\mathbf { Q } _ { t } ^ { \mathcal { F } } = \mathtt { L i n e a r } \Big ( \hat { \mathbf { Q } } _ { t - 1 } \Big ) + \mathbf { Q } ^ { \mathrm { l r n } } .\tag{1}
$$

The fused queries $\mathbf { Q } _ { t } ^ { \mathcal { F } }$ replace $\mathbf { Q } ^ { \mathrm { { l r n } } }$ in the last $L _ { 2 }$ layers, so temporal information is propagated while the learnable queries support the detection of newly appearing objects.

Despite this streamlined design, both models require fine-tuning the full encoder, since the pre-trained attention layers must adapt to the injected queries. PMT [4] retains the same philosophy as EoMT and VidEoMT but moves query processing into a separate Segmenter-like [36] Transformer decoder. This decoupling allows the ViT encoder to remain frozen, making PMT compatible with multi-task deployment while preserving the simple design and efficiency of encoder-only architectures.

## 4. Long-term Video Mask Transformer

In this work, we take the state-of-the-art model PMT as our baseline and improve long-term modeling with two crucial improvements: (i) We allow the model to adaptively retrieve and store object information in memory using a GRU (Sec. 4.1). (ii) We allow for training on long videos while limiting memory usage and preventing vanishing gradients using Truncated Query Propagation (Sec. 4.2). The resulting model is called the Long-term Video Mask Transformer (LVMT), and it is visualized in Fig. 2.

## 4.1. GRU-based Query Propagation

Motivation. VidEoMT and PMT adopt the same query fusion strategy, defined in Eq. (1). At frame t, the input queries are obtained by linearly projecting the previousframe output $\hat { \mathbf { Q } } _ { t - 1 }$ and adding it to the learnable queries $\mathbf { Q } ^ { \mathrm { { l r n } } }$ . Although this strategy offers a simple and efficient mechanism for temporal propagation, it has a fundamental limitation: it does not allow the model to adaptively select the information that it wishes to keep, as the propagated queries are always just a simple sum of the learnable queries and the projected previous output.

![](images/235b4442dec399c11b74b23ddc46c3f556552be3247add0969fcb3eb494bd61f.jpg)  
Figure 2. LVMT architecture. The Plain Mask Decoder (PMD) takes input queries and projected patch features from a frozen ViT encoder to produce segmentation queries for mask and class prediction. $\mathbf { A } \mathbf { t } t = 0$ , learnable queries $\mathbf { Q } ^ { \mathrm { { l r } \bar { \mathbf { n } } } }$ are used to initialize both the decoder input and the GRU hidden state. Thereafter, a GRU cell adaptively updates the hidden state from the current segmentation queries to yield propagation queries for the next frame. Truncated Query Propagation (TQP), visualized in Fig. 3, is applied at training time.

As a result, in case of long-term occlusions where an object disappears from the scene for a long time window, the query corresponding to this object is continuously updated with the learnable queries, while the per-frame encoder does not add any meaningful information to the propagated query because the object is not present in the frame. We expect that this causes the query to become diluted and lose information about the previous object, preventing it from being re-identified if it reappears after a long occlusion, making the model unsuitable for handling long-term occlusions.

Method. To address this limitation, we introduce a lightweight memory mechanism for temporal propagation that can adaptively select which information it propagates across time. Specifically, we replace the query fusion mechanism with a Gated Recurrent Unit (GRU) [9], a wellestablished recurrent architecture for modeling temporal dynamics. In this GRU, the learnable queries act as the initial hidden state, and this hidden state is adaptively updated using the previous-frame output queries. The updated hidden state is then fed into the current frame’s decoder.

With this operation, the model has the freedom to be selective in which information is stored in the hidden state and thereby propagated across time. As a result, when an object is no longer present, it can learn to keep information about this object to ensure that it can be re-identified when it reappears, allowing it to handle long-term occlusions. We empirically analyze this memory retention in Sec. B.7.

Fig. 2 shows how the GRU-based query propagation replaces the query propagation in the overall architecture. Each individual frame is first fed into the frozen ViT and learned projection layers to obtain patch features $\mathbf { X } _ { t } ^ { l } .$ . Then, these patch features are fed into the Plain Mask Decoder (PMD) from PMT [4] together with the propagated queries from the previous frame. For the first frame, at $t = 0$ , PMD is fed the learnable queries $\mathbf { Q } ^ { \mathrm { l r n } }$ instead of the propagated queries, as propagated queries are not available yet:

$$
\begin{array} { r } { \mathbf { Q } _ { 0 } ^ { \mathcal { S } } = \operatorname { P M D } \left( \mathbf { Q } ^ { \mathrm { l r n } } , \mathbf { X } _ { 0 } ^ { l } \right) . } \end{array}\tag{2}
$$

This yields segmentation queries $\mathbf { Q } _ { 0 } ^ { S }$ that are used to produce classification and mask predictions for frame $t = 0$

For the next frames, the propagated queries are produced using the GRU. Concretely, we use one GRU cell with shared parameters across all query slots. Since no previous hidden state is available at $t = 0$ , we initialize the hidden state with the learnable queries:

$$
\mathbf { h } _ { 0 } = \mathbf { Q } ^ { \mathrm { l r n } } , \qquad \mathbf { Q } _ { 1 } ^ { \mathcal { P } } = \mathsf { G R U C e 1 1 } \bigl ( \mathbf { Q } _ { 0 } ^ { \mathcal { S } } , \mathbf { \Delta h } _ { 0 } \bigr ) .\tag{3}
$$

Here, $\mathbf { Q } _ { 1 } ^ { \mathcal { P } }$ denotes the propagated query representation passed to the next frame at $t = 1$ , which is equal to the updated hidden state $\mathbf { h } _ { 1 }$ . For frames $t > 0 ,$ , PMD takes as input the propagated query representation for the current frame, together with the corresponding lateral patch features.

$$
\begin{array} { r } { \mathbf { Q } _ { t } ^ { \mathcal { S } } = \operatorname { P M D } \big ( \mathbf { Q } _ { t } ^ { \mathcal { P } } , \mathbf { X } _ { t } ^ { l } \big ) , \qquad t > 0 . } \end{array}\tag{4}
$$

After decoding frame t, the resulting segmentation queries $\mathbf { Q } _ { t } ^ { S }$ are used for current-frame classification and mask prediction, and also for updating the recurrent state:

$$
\mathbf { Q } _ { t + 1 } ^ { \mathcal { P } } = \operatorname { G R U C e 1 1 } \left( \mathbf { Q } _ { t } ^ { S } , \mathbf { \mu } _ { \mathbf { h } _ { t } } \right) , \qquad t > 0 .\tag{5}
$$

Here, $\mathbf { Q } _ { t + 1 } ^ { \mathcal { P } }$ denotes the propagated queries passed to PMD at timestep $t + 1$ , which are equal to the hidden state $\mathbf { h } _ { t + 1 }$

Training Video Clip with T Frames  
![](images/1e14189043b569ff7bf9784dc02bd80b59e174c61ded7a73aa4ea7527cb85149.jpg)  
Figure 3. Truncated Query Propagation (TQP). A video of T frames is partitioned into M chunks of F frames, and processed in order in a single iteration. For each chunk, we compute a loss and backpropagate it to calculate the gradients. The optimizer is updated with the average, accumulated gradients after all chunks are consumed. Queries are detached and propagated across chunk boundaries, bounding peak memory to a single chunk and preventing the model from suffering from vanishing gradients caused by long-horizon recurrence.

## 4.2. Truncated Query Propagation (TQP)

Motivation. After implementing the GRU-based propagation explained in Sec. 4.1, we empirically observe that the segmentation accuracy and temporal consistency improves. However, we also find that identity preservation remains challenging and that identity switches still occur frequently.

When analyzing the cases for which the highest number of identity switches occur, we observe that errors typically occur for videos that contain many object instances, frequent object disappearances, and long disappearance spans (see Sec. 6.1). Notably, the average disappearance span in these videos is about 2.6× longer than the clip length used to train the model. Since the model is trained on short clips, it receives only limited temporal supervision during training, and the model is not exposed to enough disappearing and reappearing entities. Hence, in scenes where objects disappear for longer than the training horizon, the hidden state is insufficiently trained to preserve object identity across such interruptions. This can lead to identity switches at test time.

An intuitive next step is therefore to increase the training clip length, exposing the model to longer temporal dependencies. However, we find that naively training on longer clips is ineffective and introduces two problems. First, it causes a drop in accuracy, which we attribute to the harder optimization of recurrent propagation over long sequences, where gradients must pass through many GRU updates and gradually ‘vanish’. Second, it consistently increases memory consumption, often leading to out-of-memory errors.

Method. To address this, we apply the principle of Truncated Backpropagation Through Time (TBPTT) [39] to our GRU-based query propagation setting, which we refer to as Truncated Query Propagation (TQP). In TQP, the hidden state is propagated over long videos, but the gradient flow is truncated at the boundaries of short chunks. This enables long-horizon supervision for the model without backpropagating through the full sequence, keeping training memoryefficient and improving optimization stability.

TQP, visualized in Fig. 3, partitions a training clip of T frames into a sequence of $M = \lceil T / F \rceil$ chunks, denoted as $\mathcal { C H } = \{ \mathrm { c h } _ { 1 } , \mathrm { c h } _ { 2 } , \mathrm { . ~ . ~ . ~ } , \mathrm { c h } _ { M } \}$ , where each chunk contains at most F frames. The chunks are processed in order within a single training iteration, and the optimizer updates the weights only once after all M chunks of the video have been consumed. To optimize the model, TQP leverages the average of the gradients computed for the individual chunks.

For each chunk i, we apply the forward and backward pass as usual, yielding a loss $\bar { \mathcal { L } } ^ { ( i ) }$ and gradients to update the weights based on this loss. Importantly, to enable the model to learn long-term temporal behavior, we allow information flow across chunks during the forward. Since the propagated queries are the GRU hidden state, i.e., $\mathbf { Q } _ { t } ^ { \mathcal { P } } = \mathbf { h } _ { t }$ , we pass the final hidden state of chunk i to the next chunk as the initial propagated query state, after detaching it from the computational graph:

$$
{ \bf Q } _ { \mathrm { i n i t } } ^ { { \cal P } , i + 1 } = { \bf h } _ { \mathrm { i n i t } } ^ { i + 1 } = \operatorname* { d e t a c h } \left( { \bf h } _ { \mathrm { e n d } } ^ { i } \right) .\tag{6}
$$

As a result, temporal information is preserved across chunks, while the computational graph remains bounded and full backpropagation through the entire sequence is avoided. This enables long-horizon supervision for the GRU with bounded memory and more stable optimization.

## 5. Experiments

Datasets. We evaluate LVMT on six standard video segmentation benchmarks. For Video Instance Segmentation (VIS), we mainly focus on OVIS [31], with challenging scenarios with heavy occlusions and crowded scenes, and YouTube-VIS 2022 [42], with long videos. We additionally evaluate on YouTube-VIS 2019 and 2021. For Video Panoptic Segmentation (VPS) we adopt VIPSeg [26] and for Video Semantic Segmentation (VSS) we use VSPW [25].

Implementation Details. Unless stated otherwise, we use a frozen ViT-L [12] encoder initialized with DINOv3 [34]. Models are trained with the AdamW optimizer [22] using mixed precision. By default, videos are processed in temporal chunks of $F = 5$ frames, with an overall temporal window of $T = 1 5$ frames, i.e., M = 3 chunks.

For ground-truth matching, we follow standard practice: each object is matched to a query in the frame it first appears, and the assignment is kept across subsequent frames. For a fair comparison, we adopt the same batch size, learning rate, and learning rate scheduler as PMT [4], and the same video resolutions and number of training iterations as previous work [4, 20, 28]; see Sec. A for details.

Performance Metrics. We evaluate our models using standard performance metrics for video segmentation. In particular, for VIS, we report Average Precision (AP) and Average Recall (AR) [42]. For VPS, we use Video Panoptic Quality (VPQ) [18] and Segmentation and Tracking Quality (STQ) [38]. For VSS, we adopt mean Intersection over Union (mIoU) and Video Consistency (mVC) [25].

We further evaluate the temporal consistency and identity preservation of our method with specialized tracking metrics. We report IDF1 [33], Association Accuracy (AssA) [23], mostly tracked (MT), mostly lost (ML) [11], and identity switches (IDS) [11]. Sec. A.4 provides more details.

Efficiency Metrics. For computational efficiency, we report both frames per second (FPS) and GFLOPs. FPS are measured as the average number of frames processed per second in the validation set at batch size 1, on a single NVIDIA H100 GPU with FlashAttention v2 [10] and torch.compile [1] enabled in default settings. GFLOPs are computed withfvcore [32] as the average over all validation images, where $\mathrm { G F L O P s } = \mathrm { F L O P s } \times \bar { 1 0 ^ { 9 } }$

## 6. Results

## 6.1. Main Results

In Tab. 1, we present a stepwise analysis of modifications from PMT [4] toward our final LVMT model, reporting mean AP over five runs. We choose PMT as our baseline because it achieves state-of-the-art accuracy at high FPS.

GRU-based Query Propagation. In step (1), we replace PMT’s query fusion with our GRU-based query propagation. We find that this improves AP by +2.4 on the challenging OVIS dataset, with only a negligible impact on the number of parameters, GFLOPs and prediction speed. This suggests that allowing the model to adaptively select the information to store in memory allows it to propagate more useful information to the decoder that processes future frames, improving video segmentation performance.

Despite improved performance, we still empirically find that the model struggles to re-identify objects after long occlusions, causing erroneous identity switches.

<table><tr><td>Method</td><td>Step GRU TQP</td><td></td><td></td><td> $T _ { \mathrm { t r a i n } }$ </td><td>Mean AP ↑</td><td></td><td>Params ↓ GFLOPs ↓ FPS ↑</td><td></td></tr><tr><td>PMT [4]</td><td>(0)</td><td>×</td><td>×</td><td>5</td><td> $5 1 . 8 \pm 0 . 3$ </td><td>358M</td><td>1014</td><td>97</td></tr><tr><td></td><td>(1)</td><td>√</td><td>X</td><td>5</td><td> $5 4 . 2 \pm 0 . 4$ </td><td>363M</td><td>1015</td><td>95</td></tr><tr><td></td><td>(2)</td><td>√</td><td>×</td><td>10</td><td> $5 2 . 3 \pm 0 . 3$ </td><td>363M</td><td>1015</td><td>95</td></tr><tr><td></td><td>(3)</td><td>√</td><td>√</td><td>10</td><td> $5 6 . 1 \pm 0 . 3$ </td><td>363M</td><td>1015</td><td>95</td></tr><tr><td>LVMT (Ours)</td><td>(4)</td><td>√</td><td>√</td><td>15</td><td> ${ \bf 5 6 . 4 \pm 0 . 4 }$ </td><td>363M</td><td>1015</td><td>95</td></tr></table>

Table 1. Stepwise modifications from PMT to LVMT on OVIS val [31]. Mean AP ± standard deviation are reported over five independent runs. $T _ { \mathrm { t r a i n } }$ is the training-clip length in frames.

To further explore this phenomenon, we compute some statistics for the 20 videos on which the model predictions from step (1) contain the most and the least identity switches. The exact statistics are reported in Sec. B.2.

We find that videos with the most identity switches contain about 3× more objects per video, have 5× more cases where objects are occluded $( i . e .$ , they disappear and reappear), and that the mean length of an occlusion is 2× longer. In fact, we find that the mean occlusion length is 13 frames, which is 2.6× longer than the training clips on which the model trains. This suggests that the model should be trained on longer clips, to allow it to learn long-term modeling.

Training with Longer Clips. Therefore, in step (2), we increase the training horizon from $T _ { \mathrm { t r a i n } } = 5 \mathrm { t o } T _ { \mathrm { t r a i n } } = 1 0$ frames. We use $T _ { \mathrm { t r a i n } } = 1 0$ as the longest feasible setting, since longer clips lead to out-of-memory errors. Interestingly, training with this longer horizon reduces the AP by 1.9 points compared to step (1). This indicates that naively increasing the training clip does not automatically improve performance. We expect that this happens because gradients ‘vanish’ when optimizing the recurrent model on long sequences, harming the optimization process. We confirm this through a gradient-flow analysis in Sec. B.6.

TQP Training Strategy. To address this issue, we apply our TQP training strategy in step (3). TQP yields a significant +3.8 AP boost compared to naive longer-clip training, while maintaining inference efficiency. When using even longer clips in the final step (4), performance is boosted slightly further. This demonstrates that TQP’s chunked training allows the model to fully benefit from long-clip training without being impacted by vanishing gradients or increased memory usage. Overall, the results show that LVMT’s use of GRUbased propagation and long-clip training with TQP allow it to outperform the state-of-the-art PMT baseline by a significant +4.6 AP while preserving simplicity and efficiency, showcasing its effectiveness.

## 6.2. Comparison with State-of-the-Art Models

Video Instance Segmentation (VIS). In Tab. 2, we compare LVMT to existing state-of-the-art models on the challenging OVIS [31] and YouTube-VIS 2022 [42] benchmarks. We compare to two categories of models: those that fine-tune the encoder and those that keep the encoder frozen. In LVMT, we decide to keep the encoder frozen like in PMT [4] because (i) this allows the encoder to be reused for other tasks, and (ii) it simply performs better, as we show in more detail in Sec. B.5.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Pre-training</td><td rowspan="2">Encoder</td><td colspan="5">OVIS val [31]</td><td colspan="5">YouTube-VIS 2022 val [42]</td></tr><tr><td>AP</td><td> $\mathrm { A P } _ { 7 5 }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td><td>GFLOPs</td><td>FPS</td><td> $\mathbf { A P } ^ { \mathrm { L } }$ </td><td> $\mathrm { A P } _ { 7 5 } ^ { \mathrm { L } }$ </td><td> $\mathbf { A R } _ { 1 0 } ^ { \mathrm { L } }$ </td><td>GFLOPs</td><td>FPS</td></tr><tr><td>DVIS++ [44]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td>E</td><td>49.6</td><td>55.0</td><td>54.6</td><td>868</td><td>17</td><td>37.5</td><td>39.4</td><td>43.5</td><td>820</td><td>18</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td>也</td><td>53.2</td><td>59.1</td><td>58.2</td><td>863</td><td>15</td><td>39.5</td><td>40.5</td><td>44.9</td><td>815</td><td>15</td></tr><tr><td>DVIS-DAQ [45]†</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td></td><td>54.3</td><td>60.2</td><td>59.8</td><td>1173</td><td>8</td><td>42.0</td><td>43.0</td><td>48.4</td><td>826</td><td>10</td></tr><tr><td>LOMM [19]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td>主主</td><td>51.7</td><td>57.5</td><td>56.2</td><td>899</td><td>12</td><td>48.2</td><td>53.2</td><td>52.6</td><td>842</td><td>12</td></tr><tr><td>VidEoMT [28]†</td><td>ViT-L [12]</td><td>DINOv2</td><td></td><td>52.5</td><td>57.2</td><td>57.5</td><td>934</td><td>115</td><td>42.6</td><td>46.1</td><td>48.1</td><td>557</td><td>161</td></tr><tr><td>VidEoMT [28]†</td><td>ViT-L [12]</td><td>DINOv3</td><td>主</td><td>51.9</td><td>57.3</td><td>57.4</td><td>934</td><td>104</td><td>42.8</td><td>44.5</td><td>50.3</td><td>557</td><td>137</td></tr><tr><td>CAVIS [20]†</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td></td><td>53.1</td><td>58.8</td><td>58.5</td><td>1168</td><td>10</td><td>41.4</td><td>41.0</td><td>47.4</td><td>815</td><td>15</td></tr><tr><td>PMT [4]†</td><td>ViT-L [12]</td><td>DINOv2</td><td></td><td>51.8</td><td>57.7</td><td>56.0</td><td>1014</td><td>99</td><td>41.6</td><td>45.6</td><td>45.3</td><td>617</td><td>129</td></tr><tr><td>LVMT (Ours)†</td><td>ViT-L [12]</td><td>DINOv2</td><td></td><td>55.6</td><td>61.6</td><td>60.5</td><td>1015</td><td>97</td><td>48.4</td><td>51.1</td><td>52.1</td><td>618</td><td>127</td></tr><tr><td>CAVIS [20]†</td><td>ViT-Adapter-L [5]</td><td>DINOv3</td><td></td><td>53.4</td><td>59.3</td><td>58.3</td><td>1168</td><td>9</td><td>42.2</td><td>37.6</td><td>49.6</td><td>815</td><td>13</td></tr><tr><td>PMT [4]†</td><td>ViT-L [12]</td><td>DINOv3</td><td></td><td>52.0</td><td>56.0</td><td>57.7</td><td>1014</td><td>97</td><td>45.8</td><td>46.6</td><td>53.2</td><td>617</td><td>123</td></tr><tr><td>LVMT (Ours)†</td><td>ViT-L [12]</td><td>DINOv3</td><td></td><td>56.7</td><td>61.8</td><td>61.6</td><td>1015</td><td>95</td><td>51.5</td><td>58.3</td><td>55.4</td><td>618</td><td>121</td></tr></table>

Table 2. LVMT for VIS on OVIS and YouTube-VIS 2022 [31, 42]. <sup>†</sup>Input resolution of 544 shortest image side for OVIS.
<table><tr><td>Method</td><td>IDF1 (%) ↑</td><td>AssA (%) ↑</td><td>MT (%) ↑</td><td>ML (%) ↓</td><td>Total IDS ↓</td></tr><tr><td>CAVIS [20]</td><td>79.2</td><td>74.7</td><td>76.3</td><td>6.1</td><td>2683</td></tr><tr><td>VidEoMT [28]</td><td>78.7</td><td>73.3</td><td>75.5</td><td>6.2</td><td>2801</td></tr><tr><td>PMT [4]</td><td>78.9</td><td>73.3</td><td>75.7</td><td>6.3</td><td>2755</td></tr><tr><td>LVMT (Ours)</td><td>81.8</td><td>78.4</td><td>78.8</td><td>5.5</td><td>2001</td></tr></table>

Table 3. Tracking quality on OVIS val [31]. We compare LVMT with existing methods using specialized tracking metrics.

Comparing to existing models that also keep the encoder frozen, LVMT obtains a much higher accuracy, with +3.3 AP on OVIS and +9.3 AP on YouTube-VIS 2022 compared to CAVIS [20] with DINOv3, while being much faster. As already demonstrated in Sec. 6.1, LVMT also performs considerably better than PMT [4], while preserving its efficiency.

LVMT also significantly outperforms methods that do finetune the encoder. Specifically, it beats current state-ofthe-art method DVIS-DAQ [45] on OVIS by +2.4 AP and LOMM [19] on YouTube-VIS 2022 by +3.3 AP, while being over 10× faster than both of them. When using the same DINOv2 encoder, LVMT performs comparably to LOMM at an AP of 48.4 and 48.2, respectively, while still being much faster. Compared to the highly efficient method VidEoMT [28], LVMT obtains a considerably higher accuracy, at +4.2 AP for OVIS and +8.7 from YouTube-VIS 2022, while being only slightly less fast.

In Sec. B.1, we demonstrate that similar trends hold on the YouTube-VIS 2019 and 2021 datasets, albeit with smaller absolute differences due to the lower complexity of these datasets. Overall, these results demonstrate the strength and effectiveness of LVMT and its components for video segmentation on complex and long videos.

Tracking Quality. Tab. 3 further analyzes the tracking performance of LVMT on OVIS using specialized tracking metrics. LVMT achieves the best performance across all reported metrics. Most notably, compared to PMT, it reduces the number of identity switches by about 27%. This shows that LVMT reduces identity errors, as intended. We also provide qualitative results in Sec. E to further illustrate the tracking quality of LVMT.

Video Panoptic Segmentation (VPS). LVMT also obtains new state-of-the-art results for VPS on the VIPSeg [26] benchmark, as demonstrated in Tab. 4. It significantly outperforms both PMT and CAVIS that also use a frozen encoder, while being much faster than CAVIS. Compared to existing state-of-the-art model DVIS-DAQ, which finetunes the encoder, LVMT improves performance by +2.9 VPQ when using DINOv3 pretraining. When using DINOv2, the models perform more similarly at a 0.8 VPQ difference, but LVMT is over 10× faster and allows the encoder to be reused, making it more practically useful. These results confirm LVMT’s effectiveness in obtaining state-of-the-art accuracy for video segmentation while maintaining high FPS.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">PT Encoder</td><td rowspan="2"></td><td colspan="4">VIPSeg val [26]</td></tr><tr><td>VPQ</td><td>STQ</td><td>GFLOPs FPS</td><td></td></tr><tr><td>DVIS++ [44]</td><td>ViT-Adapter-L [5] D2</td><td></td><td>也</td><td>56.0</td><td>49.8</td><td>2290</td><td>13</td></tr><tr><td></td><td>DVIS-DAQ [45] ViT-Adapter-L [5] D2</td><td></td><td>也</td><td>57.4</td><td>52.0</td><td>2315</td><td>4</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5] D2</td><td></td><td>也</td><td>56.9</td><td>51.0</td><td>2612</td><td>10</td></tr><tr><td>VidEoMT [28]</td><td>ViT-L [12]</td><td>D2</td><td>也</td><td>55.2</td><td>48.9</td><td>1897</td><td>75</td></tr><tr><td>VidEoMT [28]</td><td>ViT-L [12]</td><td>D3</td><td>此</td><td>55.1</td><td>48.1</td><td>1897</td><td>71</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5] D2</td><td></td><td></td><td>56.4</td><td>49.0</td><td>2612</td><td>10</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>D2</td><td></td><td>55.3</td><td>48.2</td><td>2037</td><td>60</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>D2</td><td></td><td>56.6</td><td>51.3</td><td>2038</td><td>59</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5] D3</td><td></td><td></td><td>56.8</td><td>51.2</td><td>2612</td><td>9</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>D3</td><td></td><td>55.5</td><td>49.2</td><td>2037</td><td>58</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>D3</td><td></td><td>60.3</td><td>53.4</td><td>2038</td><td>57</td></tr></table>

Table 4. LVMT for VPS on VIPSeg [26]. PT: Pre-training. D2: DINOv2 [29]. D3: DINOv3 [34].
<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2"></td><td rowspan="2">PT Encoder</td><td colspan="4">VSPW val [25]</td></tr><tr><td> $\overline { { \mathrm { { m V C } } _ { 1 6 } } }$ </td><td>mIoU</td><td>GFLOPs FPS</td><td></td></tr><tr><td>DVIS++ [44]</td><td>ViT-Adapter-L [5] D2</td><td></td><td>也</td><td>94.2</td><td>62.8</td><td>2290</td><td>13</td></tr><tr><td>VidEoMT [28]</td><td>ViT-L [12]</td><td>D2</td><td>也</td><td>95.0</td><td>64.9</td><td>1909</td><td>73</td></tr><tr><td>VidEoMT [28] ViT-L [12]</td><td></td><td>D3</td><td>此</td><td>94.4</td><td>64.0</td><td>1909</td><td>71</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>D2</td><td></td><td>94.6</td><td>64.3</td><td>2049</td><td>60</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>D2</td><td></td><td>95.2</td><td>65.4</td><td>2050</td><td>59</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>D3</td><td></td><td>94.9</td><td>65.7</td><td>2049</td><td>58</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>D3</td><td></td><td>95.3</td><td>66.4</td><td>2050</td><td>57</td></tr></table>

Table 5. LVMT for VSS on VSPW [25]. PT: Pre-training. D2: DINOv2 [29]. D3: DINOv3 [34].

Video Semantic Segmentation (VSS). The strength of LVMT is further demonstrated for the VSS task on the VSPW [25] benchmark in Tab. 5. Even with a frozen encoder, LVMT surpasses VidEoMT [28] by +0.9 mVC and +2.4 mIoU with DINOv3, while having only slightly lower FPS. Compared to PMT [4], which operates under the same frozen-encoder setting and virtually identical FPS, LVMT gains +0.4 mVC and +0.7 mIoU in combination with DI-NOv3. In short, LVMT also sets a new state of the art on VSPW while preserving high efficiency of current models.

<table><tr><td>Memory</td><td>AP</td><td>AP75</td><td>AR10</td><td>GFLOPs</td><td>FPS</td></tr><tr><td>LSTM [14]</td><td>56.0</td><td>61.3</td><td>60.9</td><td>1016</td><td>93</td></tr><tr><td>S5 [35]</td><td>55.8</td><td>61.8</td><td>60.5</td><td>1014</td><td>96</td></tr><tr><td>DeepGRU [46]</td><td>56.5</td><td>61.8</td><td>61.5</td><td>1019</td><td>91</td></tr><tr><td>Memory Bank [13]</td><td>54.5</td><td>59.8</td><td>59.9</td><td>1017</td><td>86</td></tr><tr><td>GRU [9]</td><td>56.7</td><td>61.8</td><td>61.6</td><td>1015</td><td>95</td></tr></table>

Table 6. Effect of memory mechanism on OVIS val [31]. We compare various memory mechanisms for temporal modeling.

## 6.3. Ablations

We report the main ablation studies in this section, with additional results provided in Sec. C.

Choice of Memory Mechanism. LVMT uses a GRU cell [9] as the memory mechanism that keeps track of the queries representing the tracked objects. To assess the effectiveness of this design choice, we compare it to alternative memory mechanisms in Tab. 6. We consider recurrent units, including LSTM [14], GRU [9], and RVM’s DeepGRU [46]; a structured state-space model, S5 [35], designed for efficient long-range sequence modeling; and an explicit memory-bank design with cross-attention, following GenVIS [13].

Among these choices, the GRU provides the best overall trade-off, achieving the highest AP while remaining among the fastest variants. In particular, it improves over LSTM by +0.7 AP at slightly higher FPS, and over the memorybank baseline by a larger margin of +2.2 AP. We expect that the the simple GRU works better than the alternatives because they add unnecessary complexity that complicates the model’s optimization process. For the task of query propagation, it appears to be sufficient to equip the model with a simple mechanism that it can use to adaptively select which information it stores in memory and propagates, without requiring complex operations.

Effect of GRU and TQP. Tab. 7 studies the individual and joint effects of GRU-based query propagation and TQP for increasing training clip length. As already shown in Tab. 1, GRU-based propagation boosts performance when training on short clips but causes a performance drop when naively training on long clips, which is solved by adopting TQP’s chunked training mechanism.

The results in Tab. 7 additionally show that naive training on long clips without GRU-based propagation does not work well either, with 51.1 vs. 52.6 AP on 10 frames. This suggests that the vanishing gradient problem is not unique to the GRU mechanism, but also occurs for the default recurrent query fusion mechanism used by the PMT baseline.

Similarly, the results also show that a model with TQP but without GRU propagation does not perform as a model that uses both, with 54.5 vs. 56.5 AP on 10 frames. This indicates that both TQP and GRU are critical in achieving a good performance for long and complex video segmentation, and that both these contributions are complementary.

<table><tr><td> $\overline { { T _ { \mathrm { t r a i n } } } }$ </td><td>GRU</td><td>TQP</td><td>AP</td><td>AP75</td><td>AR10</td><td>GFLOPs</td><td>FPS</td></tr><tr><td>5</td><td>×</td><td>X</td><td>52.0</td><td>56.0</td><td>57.7</td><td>1014</td><td>97</td></tr><tr><td>5</td><td>√</td><td>X</td><td>54.7</td><td>60.6</td><td>59.7</td><td>1015</td><td>95</td></tr><tr><td>10</td><td>×</td><td>×</td><td>51.1</td><td>56.0</td><td>56.7</td><td>1014</td><td>97</td></tr><tr><td>10</td><td>√</td><td>X</td><td>52.6</td><td>56.6</td><td>58.4</td><td>1015</td><td>95</td></tr><tr><td>10</td><td>×</td><td>√</td><td>54.5</td><td>59.6</td><td>59.9</td><td>1014</td><td>97</td></tr><tr><td>10</td><td>√</td><td>√</td><td>56.5</td><td>61.5</td><td>61.5</td><td>1015</td><td>95</td></tr><tr><td>15</td><td>×</td><td>X</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td></tr><tr><td>15</td><td>√</td><td>×</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td></tr><tr><td>15</td><td>×</td><td>√</td><td>54.6</td><td>60.1</td><td>59.5</td><td>1014</td><td>97</td></tr><tr><td>15</td><td>√</td><td>√</td><td>56.7</td><td>61.8</td><td>61.6</td><td>1015</td><td>95</td></tr><tr><td>20</td><td>× &gt;</td><td>X</td><td>00M</td><td>OOM</td><td>00M</td><td>00M</td><td>00M</td></tr><tr><td>20</td><td></td><td>X</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td></tr><tr><td>20</td><td>×</td><td>√</td><td>54.6</td><td>60.2</td><td>59.3</td><td>1014</td><td>97</td></tr><tr><td>20</td><td>√</td><td>√</td><td>56.8</td><td>62.2</td><td>61.9</td><td>1015</td><td>95</td></tr></table>

Table 7. Effect of GRU and TQP on OVIS val [31]. We evaluate the effects of GRU-based query propagation and TQP across different training clip lengths $T _ { \mathrm { t r a i n } } .$ . All results are measured using 8 NVIDIA H100 (94GB) with batch size 8 (one clip per GPU) and multi-scale resolution (320–640px shortest side, capped at 768).

Finally, we find that training without TQP results in outof-memory errors for clips of 15 frames or longer, whereas TQP enables training at these clip lengths. However, increasing $T _ { \mathrm { t r a i n } }$ from 15 to 20 frames yields only a modest gain of +0.1 AP. This is consistent with our analysis in Sec. 6.1, which shows that the average occlusion length in OVIS is 13 frames. Since $T _ { \mathrm { t r a i n } } = 1 5$ already covers the average occlusion. We therefore expect our long-term training strategy to provide greater benefits on datasets with longer occlusions, where longer training clips would expose the model to longer occlusion events. In this work, we use $T _ { \mathrm { t r a i n } } = 1 5$ frames by default, as it provides a good trade-off between performance and training time.

## 7. Conclusion

In this paper, we have addressed the challenge of long-term object tracking for video segmentation tasks. Existing video segmentation methods struggle with long-term tracking because (i) they do not have a mechanism that can adaptively store relevant information about tracked objects, and (ii) they cannot be trained effectively on long videos. We addressed these limitations by (i) employing a lightweight GRU-memory that can adaptively propagate relevant information across time, and (ii) introducing Truncated Query Propagation (TQP), in which the model is trained on chunks of videos to enable stable and memory-efficient optimization on long sequences. The resulting model, the Long-term Video Mask Transformer (LVMT), outperforms state-of-theart methods across all video segmentation tasks while being over 10× faster. At the same time, it is consistently more accurate than the efficient PMT baseline at comparable speed.

## Acknowledgments

This work was partly funded by the Cynergy4MIE project, supported by the Chips Joint Undertaking and its members, including top-up funding from National Authorities under Grant Agreement No. 101140226, the BMFTR project WestAI (grant no. 16IS22094D), and the EU project JUPITER AI Factory (grant no. 101250682). The experiments utilized the Dutch national infrastructure, supported by the SURF Cooperative under grant nos. EINF-14337 and EINF-17956 and funded by the Dutch Research Council (NWO), computing resources granted by RWTH Aachen under project seg4video, and resources provided by the Gauss Centre for Supercomputing e.V. through the John von Neumann Institute for Computing (NIC) on the GCS supercomputers JUWELS / JUPITER at the Julich Supercomputing Centre.¨

## References

[1] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In ASPLOS, 2024. 6, 11

[2] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle,¨ Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar,´ Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment Anything with Concepts. In ICLR, 2026. 3

[3] Niccolo Cavagnero, Gabriele Rosi, Claudia Cuttano,\` Francesca Pistilli, Marco Ciccone, Giuseppe Averta, and Fabio Cermelli. PEM: Prototype-based Efficient MaskFormer for Image Segmentation. In CVPR, 2024. 3

[4] Niccolo Cavagnero, Narges Norouzi, Gijs Dubbelman, and \` Daan de Geus. PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders. In CVPRW, 2026. 1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16

[5] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision Transformer Adapter for Dense Predictions. In ICLR, 2023. 3, 7, 12

[6] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. In CVPR, 2022. 3, 11

[7] Ho Kei Cheng and Alexander G. Schwing. XMem: Longterm video object segmentation with an atkinson-shiffrin memory model. In ECCV, 2022. 3

[8] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the Object Back into Video Object Segmentation. In CVPR, 2024. 3

[9] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre,¨ Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. In Empirical Methods in Natural Language Processing, 2014. 2, 4, 8, 11, 14

[10] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR, 2024. 6, 11

[11] Patrick Dendorfer, Aljosa Osep, Anton Milan, Konrad Schindler, Daniel Cremers, Ian Reid, Stefan Roth, and Laura Leal-Taixe. MOTChallenge: A Benchmark for Single-´ Camera Multiple Target Tracking. IJCV, 2021. 6, 11

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR, 2021. 6, 7, 12, 13

[13] Miran Heo, Sukjun Hwang, Jeongseok Hyun, Hanjung Kim, Seoung Wug Oh, Joon-Young Lee, and Seon Joo Kim. A Generalized Framework for Video Instance Segmentation. In CVPR, 2023. 2, 3, 8

[14] Sepp Hochreiter and Jurgen Schmidhuber. Long Short-Term¨ Memory. Neural Computation, 1997. 8

[15] De-An Huang, Zhiding Yu, and Anima Anandkumar. MinVIS: A Minimal Video Instance Segmentation Framework without Video-based Training. In NeurIPS, 2022. 2

[16] Lei Ke, Henghui Ding, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, and Fisher Yu. Video mask transfiner for highquality video instance segmentation. In ECCV, pages 731– 747, 2022. 2

[17] Tommie Kerssies, Niccolo Cavagnero, Alexander Hermans,\` Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus. Your ViT is Secretly an Image Segmentation Model. In CVPR, 2025. 1, 3

[18] Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Video Panoptic Segmentation. In CVPR, 2020. 2, 6

[19] Seunghun Lee, Jiwan Seo, Minwoo Choi, Kiljoon Han, Jahoon Jeong, Zane Durante, Ehsan Adeli, Sang Hyun Park, and Sunghoon Im. LOMM: Latest Object Memory Management for Temporally Consistent Video Instance Segmentation. In ICCV, 2025. 2, 7, 12

[20] Seunghun Lee, Jiwan Seo, Kiljoon Han, Minwoo Choi, and Sunghoon Im. Context-Aware Video Instance Segmentation. In ICCV, 2025. 1, 2, 6, 7, 11, 12

[21] Qin Liu, Jianfeng Wang, Zhengyuan Yang, Linjie Li, Kevin Lin, Marc Niethammer, and Lijuan Wang. Livos: Light video object segmentation with gated linear matching. In CVPR, pages 8668–8678, 2025. 3

[22] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In ICLR, 2019. 6, 11

[23] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip H. S. Torr, Andreas Geiger, Laura Leal-Taixe, and Bastian Leibe.´ HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. IJCV, 2021. 6, 11

[24] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. TrackFormer: Multi-Object Tracking with Transformers. In CVPR, 2022. 2

[25] Jiaxu Miao, Yunchao Wei, Yu Wu, Chen Liang, Guangrui Li, and Yi Yang. VSPW: A Large-scale Dataset for Video Scene Parsing in the Wild. In CVPR, 2021. 5, 6, 7, 11

[26] Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, and Yi Yang. Large-Scale Video Panoptic Segmentation in the Wild: A Benchmark. In CVPR, 2022. 5, 7, 11

[27] David Nilsson and Cristian Sminchisescu. Semantic Video Segmentation by Gated Recurrent Flow Propagation. In CVPR, 2018. 2

[28] Narges Norouzi, Idil Zulfikar, Niccolo Cavagnero, Tommie\` Kerssies, Bastian Leibe, Gijs Dubbelman, and Daan de Geus. VidEoMT: Your ViT is Secretly Also a Video Segmentation Model. In CVPR, 2026. 1, 2, 3, 6, 7, 11, 12, 13

[29] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo,´ Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DI-NOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research, 2024. 7, 11, 13

[30] PyTorch Contributors. GRUCell — PyTorch 2.12 Documentation. https://docs.pytorch.org/docs/2.9/ generated/torch.nn.GRUCell.html. Accessed: 2026-06-24. 11

[31] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded Video Instance Segmentation: A Benchmark. IJCV, 2022. 2, 5, 6, 7, 8, 11, 12, 13, 15, 16

[32] Meta Research. fvcore, 2023. 6, 11

[33] Ergys Ristani, Francesco Solera, Roger S. Zou, Rita Cucchiara, and Carlo Tomasi. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. In ECCV, 2016. 6, 11

[34] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco¨ Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana, Claire´ Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J ´ egou, Patrick Labatut, and Pi-´ otr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025. 6, 7, 11, 13

[35] Jimmy T. H. Smith, Andrew Warrington, and Scott Linderman. Simplified State Space Layers for Sequence Modeling. In ICLR, 2023. 8

[36] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for Semantic Segmentation. In ICCV, 2021. 3

[37] Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers. In CVPR, 2021. 3

[38] Mark Weber, Jun Xie, Maxwell Collins, Yukun Zhu, Paul Voigtlaender, Hartwig Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, et al. STEP: Segmenting and Tracking Every Pixel. In NeurIPS, 2021. 6

[39] Ronald J. Williams and Jing Peng. An Efficient Gradient-Based Algorithm for Online Training of Recurrent Network Trajectories. Neural Computation, 1990. 2, 5

[40] Bo Wu and Ram Nevatia. Tracking of Multiple, Partially Occluded Humans based on Static Body Part Detection. In CVPR, 2006. 11

[41] Junfeng Wu, Qihao Liu, Yi Jiang, Song Bai, Alan Yuille, and Xiang Bai. In Defense of Online Models for Video Instance Segmentation. In ECCV, 2022. 2

[42] Linjie Yang, Yuchen Fan, Ning Xu, Ding Yang, Dong Yue, Jianchao Liang, Thomas Huang, and Humphrey Huang. Video Instance Segmentation. In ICCV, 2019. 2, 5, 6, 7, 11, 12

[43] Tao Zhang, Xingye Tian, Yu Wu, Shunping Ji, Xuebo Wang, Yuan Zhang, and Pengfei Wan. DVIS: Decoupled Video Instance Segmentation Framework. In ICCV, 2023. 2

[44] Tao Zhang, Xingye Tian, Yikang Zhou, Shunping Ji, Xuebo Wang, Xin Tao, Yuan Zhang, Pengfei Wan, Zhongyuan Wang, and Yu Wu. DVIS++: Improved Decoupled Framework for Universal Video Segmentation. IEEE TPAMI, 2025. 1, 7, 11, 12

[45] Yikang Zhou, Tao Zhang, Shunping Ji, Shuicheng Yan, and Xiangtai Li. Improving Video Segmentation via Dynamic Anchor Queries. In ECCV, 2024. 1, 2, 7, 12

[46] Daniel Zoran, Nikhil Parthasarathy, Yi Yang, Drew A. Hudson, Joao Carreira, and Andrew Zisserman. Recurrent Video Masked Autoencoders. arXiv preprint arXiv:2512.13684, 2025. 8

# LVMT: Video Mask Transformer for Long-term Video Segmentation

Supplementary Material

## Appendix

## Table of contents

• §A: Implementation Details

• §B: Additional Experiments

• §C: Additional Ablations

• §E: Qualitative Results

## A. Implementation Details

## A.1. Training

Following PMT [4], we leverage frozen DINOv3 [34] pretrained encoders as the default backbone of LVMT, while also reporting results with DINOv2 [29] for comparison. The decoder is pretrained for instance segmentation at image level on the COCO dataset, without temporal supervision, following common practice. In particular, we initialize the decoder from PMT publicly available checkpoints. The GRU-based query propagation module is trained from scratch on the target video dataset. If not stated otherwise, we adopt the common protocol for video segmentation training, as implemented in CAVIS, VidEoMT and PMT [4, 20, 28].

## A.2. Hyperparameters

For all experiments, we follow prior works [4, 20, 28] with respect to precision, input resolution, and the number of training iterations. For optimization, we use automatic mixed precision and the AdamW optimizer [22] with a learning rate of $1 0 ^ { - 4 }$ , a linear warmup over the first 6,000 iterations, and polynomial learning rate decay with a power of 0.9. Specifically, we train with a batch size of 8 on 8 NVIDIA H100 GPUs. Unless stated otherwise, during training, we process each video using temporal chunks of $F = 5$ frames within an overall temporal window of T = 15 frames, i.e. M = 3 chunks. We train for 160k iterations on YouTube-VIS [42] (all versions) and OVIS [31], 40k iterations on VIPSeg [26], and 20k iterations on VSPW [25]. All models use 200 learnable queries with the same dimension as the decoder.

For temporal propagation, we use a GRU [9] cell with shared parameters across all object queries. We implement it using PyTorch’s GRUCell [30], with both input and hidden dimensions set to the decoder feature dimension.

Loss. To supervise the model, we adopt the same objective as Mask2Former [6]. Specifically, we use the classification cross-entropy loss $\mathcal { L } _ { \mathrm { c e } }$ for class predictions, and the binary cross-entropy loss $\mathcal { L } _ { \mathrm { b c e } }$ and Dice loss ${ \mathcal { L } } _ { \mathrm { d i c e } }$ for mask predictions. The total loss is defined as:

$$
\mathcal { L } _ { \mathrm { t o t } } = \lambda _ { \mathrm { b c e } } \mathcal { L } _ { \mathrm { b c e } } + \lambda _ { \mathrm { d i c e } } \mathcal { L } _ { \mathrm { d i c e } } + \lambda _ { \mathrm { c e } } \mathcal { L } _ { \mathrm { c e } } ,\tag{7}
$$

where $\lambda _ { \mathrm { b c e } } = 5 . 0 , \lambda _ { \mathrm { d i c e } } = 5 . 0$ , and $\lambda _ { \mathrm { c e } } ~ = ~ 2 . 0$ , strictly following Mask2Former [6]. Deep supervision is enabled by default across all decoder layers.

To ensure temporally consistent supervision, we follow the ground-truth matching strategy introduced in DVIS++ [44] and adopted by all competitor approaches. Specifically, each ground-truth object is matched to a query when it first appears, and this assignment is kept fixed in subsequent frames. This encourages the same query to represent the same object throughout the video.

## A.3. Evaluation

During evaluation, we follow the online video segmentation setting and process each video sequentially, one frame at a time. We report efficiency using two metrics: computational cost, measured in GFLOPs, where one GFLOP corresponds to 10<sup>9</sup> floating-point operations, and runtime speed, measured in frames per second (FPS).

For FPS measurement, we run 100 warm-up iterations and enable FlashAttention v2 [10], automatic mixed precision, and torch.compile with its default settings [1]. QKV projections in attention layers are fused to improve latency. GFLOPs are computed using fvcore [32]. All measurements are conducted on a single NVIDIA H100 GPU with PyTorch 2.9.0 and CUDA 12.8, using a batch size of 1. For each benchmark dataset, FPS and GFLOPs metrics are averaged over all frames in the validation set.

## A.4. Tracking metrics

In addition to standard video instance segmentation metrics, we report specialized tracking metrics in Tab. 3 of the main manuscript to evaluate the temporal consistency and identity preservation of LVMT across frames. Specifically, we report the ID F1 score (IDF1) [33], which measures identity consistency by computing the F1 score between predicted and ground-truth identities over the entire video. We also report Association Accuracy (AssA) [23], which evaluates the quality of temporal associations between matched object instances. MT and ML [11, 40] measure track completeness, where MT denotes the number of objects tracked for at least 80% of their lifespan, and ML denotes the number tracked for less than 20%. Finally, we report the total number of identity switches [11], where lower values indicate more stable identity preservation over time.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Pre-training</td><td rowspan="2">Encoder</td><td colspan="5">YouTube-VIS 2019 val [42]</td><td colspan="5">YouTube-VIS 2021 val [42]</td></tr><tr><td>AP</td><td>AP75</td><td>AR10</td><td>GFLOPs</td><td>FPS</td><td>AP</td><td>AP75</td><td>AR10</td><td>GFLOPs</td><td>FPS</td></tr><tr><td>DVIS++ [44]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td>也</td><td>67.7</td><td>75.3</td><td>73.7</td><td>846</td><td>18</td><td>62.3</td><td>70.2</td><td>68.0</td><td>830</td><td>17</td></tr><tr><td>DVIS-DAQ [45]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td></td><td>68.3</td><td>76.1</td><td>73.5</td><td>851</td><td>10</td><td>62.4</td><td>70.8</td><td>68.0</td><td>836</td><td>10</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td></td><td>68.9</td><td>76.2</td><td>73.6</td><td>838</td><td>15</td><td>64.6</td><td>72.5</td><td>69.3</td><td>824</td><td>15</td></tr><tr><td>LOMM [19]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td>主主主</td><td>69.1</td><td>76.5</td><td>73.5</td><td>842</td><td>12</td><td>65.0</td><td>72.7</td><td>69.1</td><td>842</td><td>12</td></tr><tr><td>VidEoMT [28]</td><td>ViT-L [12]</td><td>DINOv2</td><td>也</td><td>68.6</td><td>75.6</td><td>73.9</td><td>566</td><td>160</td><td>63.1</td><td>69.3</td><td>68.1</td><td>560</td><td>160</td></tr><tr><td>VidEoMT [28]</td><td>ViT-L [12]</td><td>DINOv3</td><td>也</td><td>68.9</td><td>77.4</td><td>74.8</td><td>566</td><td>133</td><td>63.2</td><td>71.6</td><td>69.1</td><td>560</td><td>133</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5]</td><td>DINOv2</td><td></td><td>68.5</td><td>75.8</td><td>73.5</td><td>838</td><td>15</td><td>64.3</td><td>72.0</td><td>68.9</td><td>824</td><td>15</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>DINOv2</td><td></td><td>68.8</td><td>75.2</td><td>73.9</td><td>617</td><td>129</td><td>63.8</td><td>69.4</td><td>68.1</td><td>616</td><td>129</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>DINOv2</td><td></td><td>70.1</td><td>78.3</td><td>74.4</td><td>618</td><td>127</td><td>66.4</td><td>75.3</td><td>70.4</td><td>617</td><td>127</td></tr><tr><td>CAVIS [20]</td><td>ViT-Adapter-L [5]</td><td>DINOv3</td><td></td><td>68.8</td><td>75.6</td><td>73.3</td><td>838</td><td>13</td><td>63.9</td><td>71.6</td><td>68.2</td><td>824</td><td>13</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>DINOv3</td><td></td><td>69.2</td><td>76.5</td><td>74.6</td><td>617</td><td>124</td><td>64.3</td><td>71.2</td><td>69.0</td><td>616</td><td>124</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>DINOv3</td><td></td><td>69.6</td><td>76.0</td><td>74.7</td><td>618</td><td>122</td><td>65.7</td><td>73.0</td><td>69.9</td><td>617</td><td>122</td></tr></table>

Table A. LVMT for VIS on YouTube-VIS 2019 and 2021 [42].

## B. Additional Experiments

## B.1. Comparison with State-of-the-Art Models

In the main paper, we compare LVMT to state-of-the-art methods on the OVIS [31] and YouTube-VIS 2022 [42] benchmarks (Tab. 2). In Tab. A, we further evaluate our method on the YouTube-VIS 2019 and 2021 validation sets. As in the main paper, we compare against methods that either fine-tune the encoder or keep it frozen. LVMT achieves high accuracy also on YouTube-VIS 2019 and 2021, although the absolute gains on these datasets are smaller than the ones on more challenging benchmarks. This is expected, as YouTube-VIS 2019 and 2021 are less challenging and already more saturated, leaving less room for improvement.

Compared with finetuned encoder methods that achieve state-of-the-art performance, LVMT not only achieves superior accuracy but is also substantially more efficient. Equipped with DINOv2 pre-training, it outperforms the most accurate finetuned method, LOMM [19], by +1.0 AP on YouTube-VIS 2019 and +1.4 AP on YouTube-VIS 2021, while being more than 10× faster.

Compared with the most efficient competitor, VidEoMT [28], LVMT consistently improves accuracy across both DINOv2 and DINOv3 pre-trainings, while maintaining comparable computational cost. A direct comparison with PMT [4] further isolates the effect of our temporal propagation mechanism. Under the same frozen-encoder setting, LVMT largely improves over PMT accuracy on both YouTube-VIS 2019 and YouTube-VIS 2021, with almost identical inference speed.

Overall, these results align with the findings in the main paper and show that LVMT improves the accuracy of the strongest efficient baselines while preserving their efficiency across all YouTube-VIS benchmarks.

## B.2. Identity Consistency Analysis

As discussed in the main paper, adding the GRU to PMT (step (1) in Tab. 1) substantially improves AP. However, we empirically find that the model still struggles to recover object identities after long occlusions. To better understand this failure mode, we use the predictions of the GRU-based model from step (1) to divide the OVIS validation set into two groups: the 20 videos with the highest number of identity switches and the 20 videos with the lowest number of identity switches. We compute several statistics of these two sets of videos leveraging the corresponding ground-truth annotations, and we report them in Tab. B.

<table><tr><td>GT Metric</td><td>High-IDS</td><td>Low-IDS</td></tr><tr><td>Mean objects per video</td><td>10</td><td>3</td></tr><tr><td>Disappear rate†(%)</td><td>63</td><td>13</td></tr><tr><td>Mean disappearance length (frames)</td><td>13</td><td>6</td></tr></table>

Table B. Analysis of high- and low-IDS videos on OVIS val. Highand low-IDS groups are defined as the 20 videos with the highest and lowest number of identity switches, respectively, measured using the GRU-based model from step (1) in Tab. 1 in the main paper. IDS denotes identity switches. <sup>†</sup>Fraction of frames where ≥1 object is absent due to occlusion or leaving the scene. All statistics are computed from ground-truth (GT) annotations.

The analysis reveals that videos with crowded scenes and long-term occlusions are considerably more challenging. Compared to the low-IDS group, they contain approximately 3× more objects per video, exhibit nearly 5× higher disappearance rates, and have more than twice the average disappearance length. Notably, the mean disappearance length reaches 13 frames, which is 2.6× longer than the training horizon of the model at step (1) $( T _ { \mathrm { t r a i n } } = 5 )$ . This suggests that the model struggles to re-identify objects after long occlusions because it has only been trained on much shorter temporal contexts. This motivates the adoption of a longer training strategy based on TQP.

This indicates that the recurrent memory is often required to bridge temporal gaps that are substantially longer than those encountered during training, motivating the longer training strategy introduced by TQP.

## B.3. Effect of GRU and TQP on VidEoMT

In Tab. C, we study the effect of GRU-based propagation and TQP training on VidEoMT [28].

For VidEoMT, adding the GRU improves AP by +1.7 points, showing that recurrent memory can strengthen the original query-propagation mechanism. Applying TQP without the GRU improves AP by +2.0 points, indicating that longer-horizon supervision is also beneficial for VidEoMT. When both components are combined, VidEoMT improves over its baseline by +2.3 AP.

<table><tr><td>Model</td><td>GRU</td><td>TQP</td><td>AP</td><td> $\mathrm { A P } _ { 7 5 }$ </td><td>AR10</td><td>GFLOPs FPS</td><td></td></tr><tr><td>VidEoMT [28]</td><td>X</td><td>×</td><td>50.1</td><td>53.9</td><td>55.8</td><td>934</td><td>104</td></tr><tr><td>LVMT (Ours) VidEoMT [28]</td><td>X</td><td>X</td><td>51.1</td><td>56.0</td><td>56.7</td><td>1014 935</td><td>97</td></tr><tr><td>LVMT (Ours)</td><td>√ √</td><td>× X</td><td>51.8 52.6</td><td>56.5 56.6</td><td>57.3 58.4</td><td>1015</td><td>102 95</td></tr><tr><td>VidEoMT [28]</td><td>X</td><td>√</td><td>52.1</td><td>56.7</td><td>57.5</td><td>934</td><td>104</td></tr><tr><td>LVMT (Ours)</td><td>X</td><td>√</td><td>54.5</td><td>59.6</td><td>59.9</td><td>1014</td><td>97</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VidEoMT [28]</td><td>√</td><td>√</td><td>52.4</td><td>56.8</td><td>57.8</td><td>935</td><td>102</td></tr><tr><td>LVMT (Ours)</td><td>√</td><td>√</td><td>56.5</td><td>61.5</td><td>61.5</td><td>1015</td><td>95</td></tr></table>

Table C. Effect of GRU and TQP. Impact of GRU-based propagation and TQP training on VidEoMT [28] and LVMT on OVIS val with 10-frame training clips.
<table><tr><td>Method</td><td>Size</td><td>AP</td><td>Params</td><td>GFLOPs</td><td>FPS</td></tr><tr><td>VidEoMT [28]</td><td rowspan="3">L</td><td>51.9</td><td>316M</td><td>934</td><td>104</td></tr><tr><td>PMT [4]</td><td>52.0</td><td>358M</td><td>1014</td><td>97</td></tr><tr><td>LVMT (Ours)</td><td>56.7</td><td>363M</td><td>1015</td><td>95</td></tr><tr><td rowspan="3">VidEoMT [28] PMT [4]</td><td rowspan="3">B</td><td>42.7</td><td>93M</td><td>304</td><td>178</td></tr><tr><td>42.9</td><td>116M</td><td>350</td><td>160</td></tr><tr><td>48.1</td><td>120M</td><td>351</td><td>155</td></tr><tr><td rowspan="3">VidEoMT [28] PMT [4]</td><td rowspan="3">S</td><td>31.4</td><td>24M</td><td>100</td><td>227</td></tr><tr><td>31.5</td><td>29M</td><td>110</td><td>188</td></tr><tr><td>39.1</td><td>30M</td><td>111</td><td>182</td></tr></table>

Table D. Impact of model size on OVIS val [31]. We compare VidEoMT [28], PMT [4], and LVMT across different model sizes.

These results follow a similar trend to LVMT, but the gains are substantially smaller. This suggests that GRUbased propagation and TQP can improve VidEoMT, while their benefits are better unlocked by the PMT-style architecture used in LVMT. We hypothesize that fine-tuning the encoder in VidEoMT on relatively limited video segmentation datasets may increase overfitting and limit the effect of long-horizon temporal supervision.

## B.4. Effect of Model Size

To evaluate how LVMT scales with backbone size, Tab. D reports results for ViT-S/B/L backbones and compares against VidEoMT [28] and PMT [4]. Across all three model sizes, LVMT consistently achieves higher AP than both baselines while maintaining a similar FPS to PMT.

More notably, the advantage over PMT grows as the backbone becomes smaller. This suggests that the proposed memory mechanism is especially helpful when the visual backbone has less capacity. Understanding why smaller backbones benefit more from the proposed temporal memory is an interesting direction for future work.

## B.5. Effect of Encoder Fine-Tuning

In all experiments reported in the main manuscript, we keep the ViT encoder frozen. In this section, we study whether end-to-end encoder fine-tuning can further improve performance over the frozen-encoder setting. Tab. E reports this comparison for PMT [4] and LVMT equipped with both

<table><tr><td>Method</td><td>Backbone</td><td>Encoder</td><td>AP</td><td> $\mathrm { A P } _ { 7 5 }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td></tr><tr><td colspan="6">DINOv2 [29]</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td></td><td>51.8</td><td>57.7</td><td>56.0</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td></td><td>55.6</td><td>61.6</td><td>60.5</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>也</td><td>53.8</td><td>56.0</td><td>58.8</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>也</td><td>56.5</td><td>61.2</td><td>61.7</td></tr><tr><td colspan="6">DINOv3 [34]</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>XX</td><td>52.0</td><td>56.0</td><td>57.7</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td></td><td>56.7</td><td>61.8</td><td>61.6</td></tr><tr><td>PMT [4]</td><td>ViT-L [12]</td><td>也</td><td>52.1</td><td>56.1</td><td>57.6</td></tr><tr><td>LVMT (Ours)</td><td>ViT-L [12]</td><td>也</td><td>56.1</td><td>60.7</td><td>61.2</td></tr></table>

Table E. Effect of encoder fine-tuning on OVIS val [31]. We compare frozen and fine-tuned encoders for PMT [4]and LVMT using DINOv2 and DINOv3 pre-training.

DINOv2 [29] and DINOv3 [34] pre-training.

With DINOv2, fine-tuning the encoder improves AP for both PMT and LVMT. With DINOv3, however, the gain becomes negligible or even negative. This suggests that DINOv3 already provides strong frozen representations, and naive end-to-end fine-tuning may perturb these features rather than improve them. This observation aligns with the current design principle of vision foundation models, where a strong frozen encoder can serve as a reusable representation across different downstream tasks without requiring task-specific fine-tuning [4, 34].

Importantly, LVMT with a frozen DINOv3 encoder remains the strongest setting overall. It outperforms all PMT variants, as well as its fine-tuned counterpart, while maintaining essentially the same computational cost.

## B.6. Analysis of Temporal Gradient Flow

In Sec. 6.1 of the main manuscript, we observed that increasing the training-clip length, $T _ { \mathrm { t r a i n } } .$ , from 5 to 10 frames decreases AP from 54.2 to 52.3. We hypothesize that this drop results from vanishing gradients. To verify our hypothesis, we directly analyze the gradient reaching the propagated query states at different frames.

Let $\mathbf { s } _ { t } \in \mathbb { R } ^ { Q \times C }$ denote the propagated query state entering frame t, where Q and C are the number of object queries and the embedding dimensionality of the object queries, respectively. We compute the total training loss, and then differentiate it with respect to the query state at each frame:

$$
\mathbf { g } _ { t } = { \frac { \partial { \mathcal { L } } _ { \mathrm { t o t a l } } } { \partial \mathbf { s } _ { t } } } .\tag{8}
$$

Since the gradient $\mathbf { g } _ { t }$ is a $Q \times C$ tensor, we compute its Frobenius norm to obtain a single measure of the overall learning signal reaching frame t:

$$
\Vert \mathbf { g } _ { t } \Vert _ { \mathrm { F } } = \sqrt { \sum _ { q = 1 } ^ { Q } \sum _ { c = 1 } ^ { C } \left( g _ { t } ^ { q , c } \right) ^ { 2 } } .\tag{9}
$$

This scalar value allows a direct comparison of gradient magnitudes across the temporal chain.

To compare gradient propagation over chains of different

![](images/dec9b0aa9c71006ccdbcd002803bd79f40310b329ee821c2f3a0af19240f8856.jpg)  
Figure A. Temporal gradient flow with and without TQP. TQP shortens the backward path and increases within-chunk gradient retention from approximately 5% to 37%.

lengths, we calculate the relative gradient retention:

$$
R = \frac { G _ { \mathrm { f i r s t } } } { G _ { \mathrm { l a s t } } } \times 1 0 0 \% ,\tag{10}
$$

where $G _ { \mathrm { f i r s t } }$ and $G _ { \mathrm { l a s t } }$ denote the gradient norms at the earliest and latest states of each backward chain.

The results are presented in Figure A. This figure shows that, without TQP, only approximately 5% of the gradient signal at the end of the 10-frame chain reaches the earliest propagated state. This sharp decay means that the impact of the supervision signal on the early query states is very weak, leading to only small weight updates and preventing the model from learning effectively on long videos. These results confirm our hypothesis that the model suffers from a vanishing-gradient problem.

With TQP, the relative gradient retention increases to at least 34% within each five-frame chunk. In other words, the shorter backward paths preserve a substantially stronger learning signal, allowing for more meaningful weight updates. Query values are still carried forward across chunk boundaries to preserve information about tracked objects, while detachment prevents gradients from propagating into preceding chunks. Thus, TQP maintains temporal continuity while limiting the length of each backward path, thereby mitigating the long-horizon vanishing-gradient problem.

## B.7. GRU Memory Retention Under Occlusion

In this analysis, we examine how the GRU’s behavior changes when handling occluded objects. Our hypothesis, as mentioned in Sec. 4.1, is that the GRU learns to depend more on the queries propagated from the previous frames $( i . e . ,$ , the memory), rather than the queries from the current frame, in case an object is occluded in the current frame. We refer to this hypothesized behavior as memory retention.

Following the default GRU formulation [9], the hidden state of the GRU is updated by

$$
\mathbf { h } _ { t } = \mathbf { z } _ { t } \odot \mathbf { h } _ { t - 1 } + \left( 1 - \mathbf { z } _ { t } \right) \odot \mathbf { n } _ { t } ,\tag{11}
$$

![](images/e8f224390b846798c914876e270a0ede6fcf46c1cfcb6863ea7f564d59d883b5.jpg)  
Figure B. GRU memory retention around object occlusion. Occluded-object queries rely more strongly on memory during the hidden interval.

where:

• $\mathbf { h } _ { t }$ is the GRU’s updated hidden state, representing the new object queries that will be propagated to the next frame;

$\mathbf { h } _ { t - 1 }$ is the GRU’s hidden state from the previous time step, which contains the queries propagated from the previous frames;

$\mathbf { n } _ { t }$ is the candidate state, which is weighted combination of both the previous hidden state and the queries generated in the current frame, controlled by an internal reset gate;

• $\mathbf { z } _ { t }$ is the update gate, which determines how much of the previous hidden state is retained and how much is replaced by the candidate state;

• ⊙ denotes the Hadamard product.

The update gate $\mathbf { z } _ { t }$ takes values between 0 and 1 and controls the balance between the previous hidden state and the candidate state. A value close to 1 means that the GRU mainly preserves the previous hidden state, whereas a value close to 0 means that it updates the hidden state using the candidate state. In LVMT, high values for $\mathbf { z } _ { t }$ mean that the new object queries maintain most of the information from the propagated queries, and low values mean that the new object queries are updated with more information from the current frame’s queries. This means that we can use $\mathbf { z } _ { t }$ as an indicator of memory retention. If the value for $\mathbf { z } _ { t }$ becomes higher in case of occlusions, this means the model relies more on the memory.

To assess whether this happens, we obtain $\mathbf { z } _ { t }$ for each query and frame, and report its value across different frames in cases with and without occlusions. For this experiment, using the OVIS validation set, we identify occlusion episodes as contiguous spans in which a matched object is fully absent for at least 4 frames. This procedure yields 166 occlusion episodes involving 132 distinct objects across 52 videos.

We report the results in Figure B. We find that queries associated with occluded objects retain more of their hidden state when the objects are occluded. Their update-gate value increases from an average of ∼ 0.66 before occlusion to ∼ 0.75 at occlusion onset, corresponding to a relative increase of approximately 10%. In comparison, the updategate values of unmatched queries or queries belonging to visible objects remain virtually unchanged. After the onset response, the update-gate value of the occluded-object queries gradually decreases toward its pre-occlusion level as the objects reappear.

<table><tr><td>Chunk size</td><td>AP</td><td> $\mathrm { A P } _ { 7 5 }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td><td>GFLOPs</td><td>FPS</td></tr><tr><td>1</td><td>50.9</td><td>54.2</td><td>55.4</td><td>1015</td><td>95</td></tr><tr><td>3</td><td>55.8</td><td>60.9</td><td>61.0</td><td>1015</td><td>95</td></tr><tr><td>5</td><td>56.7</td><td>61.8</td><td>61.6</td><td>1015</td><td>95</td></tr><tr><td>10</td><td>54.3</td><td>59.7</td><td>60.0</td><td>1015</td><td>95</td></tr></table>

Table F. Effect of chunk size on OVIS val [31]. We explore different chunk sizes used by TQP during training.
<table><tr><td>Hidden state initialization</td><td>AP</td><td> $\mathrm { A P } _ { 7 5 }$ </td><td> $\mathrm { A R } _ { 1 0 }$ </td></tr><tr><td>Zero Init</td><td>55.2</td><td>59.6</td><td>60.6</td></tr><tr><td>Random Gaussian Init</td><td>55.8</td><td>61.5</td><td>60.8</td></tr><tr><td>Learnable State</td><td>55.7</td><td>61.8</td><td>60.6</td></tr><tr><td>Learnable Object Queries</td><td>56.7</td><td>61.8</td><td>61.6</td></tr></table>

Table G. Effect of hidden-state initialization on OVIS val [31]. We evaluate different GRU hidden-state initialization strategies.

These results show that, when an object becomes occluded, the GRU preserves approximately 10% more information accumulated from previous frames where the object was visible, instead of updating the query from the newly provided frame where the object is occluded. This increased retention allows more stored information to be preserved and propagated to subsequent frames. When the object reappears, the GRU again incorporates more information from the current frame. These results support our hypothesis of memory retention under occlusion.

## C. Additional Ablations

## C.1. Effect of Chunk Size on TQP

Tab. F studies the chunk size F used by TQP during training. A chunk size of $F = 1$ frame is too small for temporal propagation and gives the weakest result. Larger chunks provide more temporal context and lead to clear gains, with the best performance achieved at F = 5 frames.

When the chunk size is increased to F = 10 frames, AP drops to 54.3. This behavior is consistent with what we observe in Tab. 7 of the main manuscript, where overly long clips make optimization harder due to vanishing gradients. Therefore, we use $F = 5$ frames in all main experiments.

## C.2. Effect of Hidden-State Initialization.

Tab. G compares different strategies to initialize the hidden GRU state. Initializing the hidden state with the shared learnable object queries achieves the best performance. This result indicates that the hidden state benefits from task-aligned initialization. Since shared object queries are optimized for both object classification and mask prediction, they provide a strong starting point for temporal propagation. In contrast, zero and random initialization lack task-specific information.

Having dedicated learnable weights for the hidden state also performs worse, likely because they are less directly coupled to the final prediction objectives.

## D. Limitation

Training cost. TQP bounds peak training memory without affecting inference cost, but its sequential chunk processing increases training time. Specifically, training requires $M = \left\lceil T _ { \mathrm { t r a i n } } / F \right\rceil$ forward and backward passes per iteration, one for each chunk. Consequently, wall-clock training time and computation increase with the number of chunks. Therefore, developing a method that retains TQP’s benefits without additional training time, while preserving inference efficiency, is a valuable direction for future work.

## E. Qualitative Results

Fig. C shows a challenging OVIS [31] video with multiple visually similar motorcycles and riders undergoing heavy occlusion. PMT [4] starts producing identity switches at frame 4, when one motorcycle becomes occluded, and the errors become more severe at frame 8 under stronger occlusion. In contrast, LVMT preserves the correct identities throughout the sequence.

![](images/db85b9961dd32287dd519ea41e4827f8d02ba97bae85ca0a3b69ce84507399a2.jpg)  
PMT [4]  
LVMT (Ours)  
Ground Truth  
Figure C. Qualitative results on OVIS [31]. Comparison between PMT [4], LVMT, and the ground-truth annotations on selected frames $t = \{ 0 , 4 , 8 , 1 0 , 1 1 \}$ . PMT suffers from identity switches at t = 4 and t = 8 under heavy occlusion among similar motorcycles and riders, while LVMT preserves more consistent identities across the sequence.