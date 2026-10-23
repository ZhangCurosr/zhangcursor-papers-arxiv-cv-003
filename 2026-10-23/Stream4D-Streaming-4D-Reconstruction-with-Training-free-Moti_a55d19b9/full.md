# Stream4D: Streaming 4D Reconstruction with Training-free Motion Decoupling via Causal KV-Cache Atention

Wei Zhang   
zhangwei707@mail.nwpu.edu.cn   
School of Computer Science   
Northwestern Polytechnical   
University   
Xi’an, China   
Shiqiang Gong   
gongshiqiang@mail.nwpu.edu.cn   
Northwestern Polytechnical   
University   
Xi’an, China

Ying Feng yfeng.von@gmail.com vivo BlueImage Lab vivo Mobile Communication Co., Ltd. Hangzhou, China

Huaqi Zhang zhanghuaqi@vivo.com vivo BlueImage Lab vivo Mobile Communication Co., Ltd. Hangzhou, China

Mengtong Wu   
wumengtong@mail.nwpu.edu.cn   
School of Computer Science   
Northwestern Polytechnical   
University   
Xi’an, China

Qi Zhang<sup>∗</sup> nwpuqzhang@gmail.com vivo BlueImage Lab vivo Mobile Communication Co., Ltd. Hangzhou, China

![](images/76f7c99da974276697e4a1e83af43c3e6d689f53f6a7e51a55c9b899e3289395.jpg)

![](images/09f728e9263e49438dadb87dd89808b4e289fe9e907c9e0876c23a1edb5ac354.jpg)

Figure 1: We present Stream4D, a training-free framework that infers dynamic masks from causal-attention cues to decouple motion in streaming reconstruction. By masking dynamic regions during historical aggregation, Stream4D stabilizes online pose estimation and reduces long-sequence ghosting, yielding cleaner dynamic reconstructions.

<sup>∗</sup>Corresponding author.

## Abstract

4D scene reconstruction requires robust handling of dynamic objects, yet existing streaming methods commonly assume static scenes, leading to pose drift and ghosting artifacts over long sequences. We introduce Stream4D, a training-free, motion-decoupled streaming reconstruction framework that can be directly integrated into causal Transformer backbones (e.g., StreamVGGT, Stream3R). Our key observation is that causal attention implicitly encodes rich temporal motion cues within the KV cache. We adaptively aggregate these cues from multiple attention layers and across temporal windows, and fuse them with depth priors to handle challenging cases such as occlusions and textureless regions, producing precise dynamic masks that decouple static and dynamic regions. We then apply a soft key masking scheme that leverages these dynamic masks to suppress interference from dynamic regions in causal attention, thereby improving camera pose estimation accuracy. Stream4D inherits the eficiency of causal attention and remains compatible with existing streaming architectures. On diverse dynamic-scene benchmarks, Stream4D significantly improves camera pose estimation and static-scene reconstruction quality, yields reliable dynamic masks, and efectively mitigates long-sequence ghosting.

## CCS Concepts

• Computing methodologies → Reconstruction.

## Keywords

4D Reconstruction, Causal Attention, Dynamic Segmentation, KV Cache, Streaming Inference

## ACM Reference Format:

Wei Zhang, Ying Feng, Mengtong Wu, Shiqiang Gong, Huaqi Zhang, and Qi Zhang. 2026. Stream4D: Streaming 4D Reconstruction with Training-free Motion Decoupling via Causal KV-Cache Attention. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10– 14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https: //doi.org/10.1145/3767308.3836547

## 1 Introduction

4D scene reconstruction is a core capability for embodied intelligence systems [16, 17, 20, 49], with broad applications in autonomous [6, 8, 57], AR/VR [54, 56, 58], and robotic navigation [26, 27, 29, 30]. Traditional SfM/MVS pipelines rely on explicit geometric constraints and iterative optimization [2, 4], making it dificult to balance speed and scalability [47, 48, 51, 52]. Recent end-to-end learning methods have advanced multi-view 3D prediction, yet pairwise approaches typically require costly global alignment and post-processing, while memory-augmented schemes are prone to error accumulation [50]. Fast3R [40] and VGGT [23] enable dense global interaction via Transformer feed-forward structures, but depend on global self-attention, forcing reprocessing of the entire sequence for each incoming frame. To achieve online and eficient inference, causal architectures such as StreamVGGT [59] and Stream3R [9] leverage temporally causal attention and KV-Cache historical memory, substantially reducing computation and storage on long sequences—marking a key step toward scalable 4D vision.

However, existing streaming reconstruction methods commonly assume static scenes and overlook interference from dynamic ob jects in the real world. Moving objects break photometric consistency and epipolar geometry, mislead camera pose estimation, and accumulate into ghost artifacts over long sequences. Although 3D foundation models like VGGT [23, 25] can quickly provide highquality initial estimates, their training and inference typically default to static assumptions and lack motion-decoupling mechanisms. Prior attempts such as VGGT4D [5] and Easi3R [1] perform training free dynamic segmentation via attention statistics, but respectively rely on global bidirectional or pairwise local interactions: the former is incompatible with causal streaming, and shallow-layer attention becomes inefective under causality constraints; the latter only captures near-frame relations, yielding limited temporal consistency and boundary quality, and fails to prevent dynamic information from accumulating over time. Consequently, stabilizing dynamic scene reconstruction under causal streaming remains a bottleneck for practical long-sequence 4D reconstruction.

We present Stream4D: a training-free, pluggable, motion decoupled streaming 4D reconstruction framework compatible with causal backbones (e.g., StreamVGGT [59], Stream3R [9]) for dynamic scenes. Our main contributions are as follows:

• We reveal that causal attention implicitly encodes rich temporal motion cues within the KV cache, and propose an atention-based saliency aggregation method that adaptively aggregates these cues from multiple attention layers and across temporal windows.

• We introduce a robust dynamic mask generation pipeline that fuses attention-derived saliency with depth priors, effectively handling challenging cases such as occlusions and textureless regions.

• We design a soft key masking scheme that directly operates on the KV cache to suppress dynamic regions during historical aggregation, improving camera pose estimation accuracy without retraining or architectural modifications.

• Extensive experiments on dynamic-scene benchmarks demonstrate that Stream4D, without any additional training or finetuning, consistently improves pose accuracy, produces reliable dynamic masks, and significantly mitigates long-term ghosting artifacts.

## 2 Related Work

Streaming Reconstruction. Streaming approaches ofer a more scalable alternative for 3D reconstruction problems, such as monocular SLAM pipelines [28]. Inspired by learned online reconstruction, Spann3R [22] extends DUSt3R [25] with memory-based updates to achieve incremental reconstruction via recursive memory writes. Fast3R [40] and VGGT [23] replace asymmetric decoders with Transformer-based attention stacks, enabling direct multi-view fusion and delivering strong reconstruction quality. However, these methods rely on global full attention, requiring reprocessing of the entire sequence as length grows, which limits real-time scalability. CUT3R [24] adopts an RNN-style architecture to incrementally process unstructured inputs, but sufers from limited memory capacity and poor compatibility with modern hardware acceleration. Recently, StreamVGGT [59] and Stream3R [9] re-conceptualize point-graph prediction as a decoder-only Transformer [21, 34] task, achieving eficient causal inference via techniques like KV-Cache and windowed attention, scaling well to long sequences while remaining compatible with modern LLM-style training infrastructure. Nonetheless, these streaming methods generally assume static scenes and lack mechanisms to handle dynamic objects, constraining applicability in real-world scenarios [12, 13, 33, 35, 36].

4D Scene Reconstruction. Early 4D reconstruction methods typically rely on RGB-D sensors [11, 15], limiting deployment in realworld settings [7, 31, 32, 38, 55]. Many approaches jointly estimate depth, pose, and residual motion, often using motion masks on static regions, including self-supervised variants and test-time optimization schemes such as CasualSAM [53] and Robust-CVD. More recent pipelines, e.g., MegaSaM [14], leverage strong monocular depth priors [41] to achieve robust pose estimation and reconstruction on dynamic videos, while Uni4D [43] combines multiple vision foundation models with multi-stage bundle adjustment for high-quality 4D results. Although efective, these pipelines depend on powerful priors and extensive test-time optimization, which hinders scalability. Recent work reduces or replaces test-time optimization using learned priors or attention mechanisms: MonST3R [45] fine-tunes on dynamic data and exploits optical flow; DAS3R [39] augments the DPT head to produce feed-forward motion masks; CUT3R [24] fine-tunes MASt3R [10] on mixed static/dynamic data for fast reconstruction; Easi3R [1] adaptively modulates attention at inference to perform training-free 4D reconstruction on DUSt3R [25]. Despite these advances, most pipelines still require fine-tuning or secondstage post-processing, and training-free variants are often tightly coupled to specific backbones. More critically, these methods are designed around global bidirectional Transformer architectures, making them incompatible with causal streaming and leaving the long-sequence accumulation ofdynamic information unresolved. In contrast, our Stream4D is the first training-free 4D reconstruction framework tailored for causal streaming architectures. It achieves motion decoupling via selective KV-Cache updates, supports online reconstruction over long sequences, and avoids ghost accumulation.

## 3 Method

Given a casually captured video sequence $\{ \mathbf { I } _ { t } \in \mathbb { R } ^ { H \times W \times 3 } \} _ { t = 1 } ^ { T }$ , our goal is to estimate dynamic-object masks, camera motion, and a static scene point cloud. The dynamic-object masks are represented as a segmentation sequence $\{ \mathbf { M } _ { t } \} _ { t = 1 } ^ { T }$ . The camera motion is represented as the sequence of extrinsic and intrinsic parameters $\{ ( \bar { \mathbf { P } } _ { t } , \mathbf { K } _ { t } ) \} _ { t = 1 } ^ { T } .$ . The static scene point cloud is denoted by $\chi ^ { * }$ . We first introduce how a causal streaming architecture enables eficient incremental processing via KV Cache (Sec. 3.1). Next, we show how to aggregate motion cues from attention layers and generate dynamic masks (Sec. 3.2). Finally, we present motion-decoupled 4D reconstruction using the dynamic masks (Sec. 3.3).

## 3.1 Streaming 4D Geometry Reconstruction

Given an image sequence $\mathbf { I } _ { t } ,$ a generic 4D reconstruction pipeline can be written as

$$
\mathbf { F } _ { t } = \operatorname { E n c o d e r } ( \mathbf { I } _ { t } ) , \quad \mathbf { G } _ { t } = \operatorname { D e c o d e r } ( \mathbf { F } _ { t } ) , \quad ( \mathbf { P } _ { t } , \mathbf { C } _ { t } ) = \operatorname { H e a d } ( \mathbf { G } _ { t } ) ,\tag{1}
$$

where the encoder maps the input frame to image tokens ${ \bf F } _ { t } \in  { }$ $\mathbb { R } ^ { N \times C }$ , the decoder fuses multi-view information to produce geometry tokens $\mathbf { G } _ { t } \in \mathbb { R } ^ { N \times C }$ , and the prediction head outputs a point map $\mathbf { P } _ { t } ^ { \mathsf { ^ { * } } } \in \mathbb { R } ^ { 3 \times H \times W }$ and a confidence map $\mathbf { C } _ { t } \in \mathbb { R } ^ { H \times W }$ . Existing methods difer in how they aggregate multi-view information in the decoder: pairwise methods (e.g., DUSt3R [25], MASt3R [10]) process image pairs via cross-attention; memory-augmented methods $( \mathrm { e . g . }$ Spann3R [22], CUT3R [24]) maintain an external memory for incremental updates; and global interaction methods $( \mathrm { e . g . }$ , Fast3R [40], VGGT [23]) adopt all-to-all self-attention to achieve the highest accuracy but require reprocessing the entire sequence for each incoming frame, thus cannot support online streaming inference.

![](images/37be5dfbba3c1095049a6f06835ebf51b1ecc18fdb8e359285a48fc4b39265fa.jpg)  
Figure 2: Streaming reconstruction overview. Each incoming frame is encoded into tokens and decoded with temporal causal attention that attends only to past frames via a KV Cache. The cache is incrementally updated with $\left( \mathbf { K } _ { t } , \mathbf { V } _ { t } \right)$ , enabling eficient online inference.

Causal streaming architecture. To enable eficient streaming reconstruction, recent works (e.g., StreamVGGT [59], Stream3R [9], InfiniteVGGT [44]) introduce causal Transformer architectures, where temporal causal attention restricts each frame to attend only to past frames:

$$
\left\{ { \bf G } _ { t } \right\} _ { t = 1 } ^ { T } = \mathrm { D e c o d e r } \left( \mathrm { C a u s a l - A t t n } \left( \left\{ { \bf F } _ { t } \right\} _ { t = 1 } ^ { T } \right) \right) ,\tag{2}
$$

and utilize a KV Cache to store historical key-value pairs $C _ { t } \ =$ $\{ ( \mathbf { K } _ { i } , \mathbf { V } _ { i } ) \} _ { i = 1 } ^ { t - 1 }$ , avoiding redundant computation. At inference time, the model accesses cached history via cross-attention:

$$
\mathbf { G } _ { t } = \operatorname { D e c o d e r } ( \operatorname { C r o s s A t t n } ( \mathbf { F } _ { t } , C _ { t - 1 } ) ) , \quad C _ { t } = C _ { t - 1 } \cup \{ ( \mathbf { K } _ { t } , \mathbf { V } _ { t } ) \} .\tag{3}
$$

This design enables online incremental reconstruction (Fig. 2). However, existing causal architectures generally assume static scenes and lack mechanisms to handle dynamic objects. Dynamic objects can mislead pose estimation and geometric modeling and, critically, accumulate in the KV Cache over long sequences, producing ghosting artifacts.

## 3.2 Streaming Dynamic Segmentation

We propose a training-free dynamic segmentation method tailored to causal streaming architectures, which aggregates temporal motion cues directly from the KV Cache. Unlike global bidirectional methods (e.g., VGGT4D) or pairwise local methods (e.g., Easi3R) limited to adjacent frames, our approach achieves global temporal consistency via adaptive historical-frame selection and multi-modal fusion with depth priors (Fig. 4).

Attention-based saliency aggregation. In streaming inference, current-frame queries $\mathbf { Q } _ { t }$ attend to cached historical keys via:

$\mathbf { A } _ { t } ^ { ( l ) } = \ \mathrm { S o f t m a x } ( \mathbf { Q } _ { t } ^ { ( l ) } [ \mathbf { K } _ { 1 } ^ { ( l ) } , \dots , \mathbf { K } _ { t - 1 } ^ { ( l ) } ] ^ { \top } / \sqrt { d } )$ , where � indexes the layer. Dynamic objects exhibit high-variance, low-mean attention patterns due to spatial motion, while static regions show low variance and high mean (Fig. 3). We exploit this by computing per-token statistics $\mu _ { t , i } ^ { ( l ) } , \sigma _ { t , i } ^ { ( l ) }$ over an adaptive historical set $\mathcal { H } _ { t } \subseteq \left\{ 1 , \dots , t - 1 \right\}$ aggregating across frames and layers to obtain dynamic saliency:

$$
\mathbf { S } _ { t } ^ { ( l ) } = \big ( 1 - \bar { \mu } _ { t } ^ { ( l ) } \big ) \odot \bar { \sigma } _ { t } ^ { ( l ) } , \quad \mathbf { S } _ { t } ^ { \mathrm { a t t n } } = \sum _ { l \in \mathcal { L } } w _ { l } \cdot \mathbf { S } _ { t } ^ { ( l ) } ,\tag{4}
$$

where $\mathcal { L }$ spans shallow, mid-level, and deep layers with weights �<sub>�</sub> favoring mid-level semantics.

![](images/c07eb289bd065a73dd63bc48b3b9d202de4db1e37ac3afcb1dbe84b628c96d64.jpg)  
Figure 3: Temporal saliency aggregation and fusion visualization. We compare attention statistics across frames and layers: per-token mean and variance heatmaps revealing low-mean/high-variance patterns on dynamic objects; cross-frame attention maps highlighting temporal inconsistency; multi-scale attention fusion; and final fused saliency and binary masks.

![](images/46df0d05dcc6e5705c5dd48121804f05d30b1db1eb0af5fdb89164752c6bdcf5.jpg)  
Figure 4: Streaming dynamic segmentation architecture. Left: standard causal streaming reconstruction with a KV Cache, where each frame attends only to past tokens. Right: our Stream4D augments the causal pipeline with motion-decoupled masking. Queries load cached keys/values to compute attention scores, aggregate temporal saliency across frames, and produce dynamic masks. These masks are then applied to modify the cached keys, efectively suppressing interference from dynamic regions during historical aggregation.

Depth-based complementary cues. To handle challenging cases such as occlusions and textureless regions, we fuse attention saliency with depth priors from predicted depth maps $\mathbf { D } _ { t } . \mathrm { A }$ foreground prior assumes moving objects are closer:

$$
\mathbf { S } _ { t } ^ { \mathrm { d e p t h - p r i o r } } = \operatorname { S i g m o i d } ( \boldsymbol { \beta } \cdot ( \operatorname { Q u a n t i l e } ( \mathbf { D } _ { t } , \boldsymbol { q } ) - \mathbf { D } _ { t } ) ) ,\tag{5}
$$

and a temporal change measure captures inconsistency over a window �:

$$
\mathrm { S } _ { t } ^ { \mathrm { d e p t h - c h a n g e } } = \mathrm { N o r m a l i z e } \left( \frac { 1 } { \tau } \sum _ { i = t - \tau } ^ { t - 1 } \left| \mathbf { D } _ { t } - \mathbf { D } _ { i } \right| \right) .\tag{6}
$$

The final saliency fuses attention and depth scores:

$$
\mathsf { S } _ { t } = \alpha _ { \mathrm { a t t n } } \cdot \mathsf { S } _ { t } ^ { \mathrm { a t t n } } + \alpha _ { \mathrm { d e p t h } } \cdot ( \alpha _ { \mathrm { p r i o r } } \cdot \mathsf { S } _ { t } ^ { \mathrm { d e p t h } \cdot \mathrm { p r i o r } } + \alpha _ { \mathrm { c h a n g e } } \cdot \mathsf { S } _ { t } ^ { \mathrm { d e p t h } \cdot \mathrm { c h a n g e } } ) ,\tag{7}
$$

where fusion weights �<sub>atn</sub>, �<sub>depth</sub>, �<sub>prior</sub>, �<sub>change</sub> are empirically determined. The fused saliency is then processed by temporal smoothing $\begin{array} { r } { \tilde { \mathbf { S } } _ { t } = \frac { 1 } { \tau + 1 } \sum _ { k = t - \tau } ^ { t } \mathbf { S } _ { k } } \end{array}$ , Otsu thresholding [42], and morphological refinement to produce the binary mask $\mathbf { M } _ { t }$

## 3.3 Motion-Decoupled Streaming Reconstruction

With dynamic masks {M<sub>�</sub>} from Sec. 3.2, we decouple static reconstruction from dynamic interference. The core mechanism is soft key masking, which directly operates on the KV Cache to suppress dynamic regions during historical aggregation.

Soft key masking. We directly modify the cached keys in the KV Cache by applying a soft suppression based on the dynamic masks:

$$
C _ { t } ^ { \mathrm { m a s k e d } } = \{ ( \tilde { \mathbf { K } } _ { i } , \mathbf { V } _ { i } ) \} _ { i = 1 } ^ { t - 1 } , \quad \mathrm { w h e r e } \quad \tilde { \mathbf { K } } _ { i } = \mathbf { K } _ { i } \odot ( \mathbf { 1 } - \boldsymbol { \gamma } \cdot \mathbf { M } _ { i } ) ,\tag{8}
$$

where $\mathbf { M } _ { i } \in \{ 0 , 1 \} ^ { N }$ marks dynamic tokens and $\gamma \in ( 0 , 1 )$ controls the suppression strength. The decoder then performs temporal attention using the modified cache:

$$
\mathbf { G } _ { t } = \mathrm { D e c o d e r } \Big ( \mathrm { C r o s s A t t n } ( \mathbf { F } _ { t } , C _ { t - 1 } ^ { \mathrm { m a s k e d } } ) \Big ) .\tag{9}
$$

By directly suppressing dynamic keys in the cache, this scheme prevents dynamic features from interfering with historical aggregation, thereby improving camera pose estimation accuracy. We apply soft key masking to the first few decoder layers where low-level geometric features dominate, and maintain a mask cache to ensure consistent suppression across the sequence.

Static-region geometric constraints. Following standard practice, we restrict photometric and geometric consistency losses to static regions identified by the mask:

$$
\mathcal { L } ( t ) = \sum _ { i \in \mathcal { H } _ { t } } \mathbf { C } _ { t } \odot ( \mathbf { 1 } - \mathbf { M } _ { t } ) \odot \left( \lambda _ { \mathrm { p h o t o } } \cdot \boldsymbol { \rho } ( \mathbf { I } _ { t } - \mathbf { I } _ { i } ^ { \pi } ) + \lambda _ { \mathrm { g e o m } } \cdot \lVert \mathbf { P } _ { t } - \mathbf { P } _ { i } ^ { \pi } \rVert _ { 1 } \right) ,\tag{10}
$$

where $\mathbf { I } _ { i } ^ { \pi } , \mathbf { P } _ { i } ^ { \pi }$ denote warped photometric and geometric predictions, and $\mathbf { C } _ { t }$ is the confidence map. The static point cloud $\chi ^ { * }$ is updated incrementally by inserting high-confidence static points after outlier filtering.

Table 1: Dynamic object segmentation on DAVIS benchmarks. We report mean IoU $\left( \mathcal { T } _ { M } \right)$ and IoU recall $( { \mathcal { T } } _ { R } )$ . “Type” indicates Ofline vs. Streaming (single-pass, causal). <sup>∗</sup> indicates our adapted implementation with segmentation capability, as the original methods do not support segmentation.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Type</td><td colspan="2">DAVIS-2016</td><td colspan="2">DAVIS-2017</td><td colspan="2">DAVIS-all</td></tr><tr><td> $\mathcal { T } _ { M }$ </td><td> $\mathcal { T } _ { R }$ </td><td> $\mathcal { T } _ { M }$ </td><td> $\mathcal { T } _ { R }$ </td><td> $\mathcal { T } _ { M }$ </td><td> $\mathcal { T } _ { R }$ </td></tr><tr><td>DUSt3R [25]</td><td>Offline</td><td>42.10</td><td>45.70</td><td>35.20</td><td>35.30</td><td>35.90</td><td>34.00</td></tr><tr><td>Easi3R_dust3r [1]</td><td>Offline</td><td>50.10</td><td>55.77</td><td>46.86</td><td>50.54</td><td>44.10</td><td>50.85</td></tr><tr><td>Easi3R_monst3r [1]</td><td>Offline</td><td>54.93</td><td>68.00</td><td>54.75</td><td>66.16</td><td>51.64</td><td>63.06</td></tr><tr><td>MonST3R [45]</td><td>Offline</td><td>40.42</td><td>40.39</td><td>38.07</td><td>36.05</td><td>36.98</td><td>34.52</td></tr><tr><td>DAS3R [39]</td><td>Offline</td><td>41.13</td><td>38.67</td><td>44.51</td><td>43.95</td><td>43.33</td><td>38.93</td></tr><tr><td>VGGT4D [5]</td><td>Offline</td><td>62.12</td><td>76.80</td><td>56.45</td><td>65.62</td><td>50.75</td><td>55.59</td></tr><tr><td> $\mathrm { C U T 3 R ^ { * } \left[ 2 4 \right] }$ </td><td>Streaming</td><td>28.45</td><td>18.76</td><td>31.28</td><td>27.92</td><td>27.65</td><td>23.18</td></tr><tr><td> $\mathrm { S t r e a m 3 R } ^ { \ast } \ [ 9 ]$ </td><td>Streaming</td><td>32.18</td><td>21.34</td><td>35.07</td><td>31.45</td><td>30.82</td><td>27.06</td></tr><tr><td>StreamVGGT* [59]</td><td>Streaming</td><td>34.91</td><td>23.52</td><td>37.33</td><td>33.86</td><td>33.20</td><td>29.24</td></tr><tr><td>Stream4D (ours)</td><td>Streaming</td><td>43.39</td><td>39.34</td><td>46.45</td><td>42.90</td><td>43.06</td><td>41.32</td></tr></table>

![](images/60c08fcbe3bf125dfd8db581e71680d9eebe61c9bd62cb196e010822bd031bb0.jpg)  
Figure 5: Qualitative results on DAVIS dataset. Each panel shows the input frame, the Easi3R mask, and our Stream4D mask. Easi3R exhibits temporal jitter and boundary breaks under occlusions and non-rigid motions. Our streaming multi-level temporal aggregation with consistency constraints produces temporally smooth and spatially complete masks across slow motions (camel, goat, etc.) and fast motions (skateboarder, motorcyclist), demonstrating robust segmentation under causal KV-Cache attention.

## 4 Experiments

We evaluate Stream4D across three core tasks: dynamic object segmentation (Sec. 4.1), 4D reconstruction (Sec. 4.2), and long-sequence streaming eficiency (Sec. 4.3), followed by ablations. Our protocol emphasizes real-time settings, causal inference, and training-free integration atop pretrained streaming backbones.

Baselines. We compare against state-of-the-art streaming reconstruction methods and training-free 4D approaches. StreamVGGT [59] and Stream3R [9] are representative causal streaming architectures but assume static scenes; VGGT4D [5] and Easi3R [1] perform training-free dynamic segmentation under global bidirectional designs. Stream4D applies training-free motion decoupling directly on pretrained causal streaming models, requiring no fine-tuning or additional data.

Datasets. We assess dynamic object segmentation on the DAVIS-2016 [18], DAVIS-2017 [19], and DAVIS-all benchmarks, following the oficial protocols. For 4D reconstruction and camera pose estimation, we evaluate on the DyCheck dataset [3], reporting the pose error metrics (ATE, RTE, RRE). Additionally, to validate scalability and eficiency, we conduct performance benchmarks on sequences with varying lengths ranging from 10 to 200 frames.

Table 2: 4D reconstruction on DyCheck. We report pose metrics (ATE, RTE, RRE); lower is better. “Type” indicates Ofline vs. Streaming (single-pass, causal).
<table><tr><td>Method</td><td>Type</td><td>ATE↓</td><td>RTE↓</td><td>RRE↓</td></tr><tr><td>Easi3R_monst3r [1]</td><td>Offline</td><td>0.032</td><td>0.008</td><td>1.075</td></tr><tr><td>Easi3R_dust3r [1]</td><td>Offline</td><td>0.022</td><td>0.009</td><td>0.806</td></tr><tr><td>MonST3R [45]</td><td>Offline</td><td>0.038</td><td>0.010</td><td>1.172</td></tr><tr><td>POMATO [46]</td><td>Offline</td><td>0.128</td><td>0.027</td><td>3.648</td></tr><tr><td>SpatialTrackerV2 [37]</td><td>Offline</td><td>0.011</td><td>0.006</td><td>0.347</td></tr><tr><td>DAS3R [39]</td><td>Offline</td><td>0.052</td><td>0.012</td><td>1.560</td></tr><tr><td>VGGT [23]</td><td>Offline</td><td>0.013</td><td>0.008</td><td>0.418</td></tr><tr><td>VGGT4D [5]</td><td>Offline</td><td>0.010</td><td>0.007</td><td>0.374</td></tr><tr><td>CUT3R [24]</td><td>Streaming</td><td>0.036</td><td>0.013</td><td>0.860</td></tr><tr><td>StreamVGGT [59]</td><td>Streaming</td><td>0.049</td><td>0.032</td><td>4.127</td></tr><tr><td>Stream3R [9]</td><td>Streaming</td><td>0.015</td><td>0.018</td><td>0.545</td></tr><tr><td>Stream4D (ours)</td><td>Streaming</td><td>0.014 (↓6.7%)</td><td>0.017 (↓5.6%)</td><td>0.539 (↓1.1%)</td></tr></table>

![](images/1188385fa5e037b66883b4f1490e6fd16c6dd67b10faae4d3be167e07748c7b0.jpg)  
Figure 6: Ghosting suppression in streaming 4D reconstruction. Naïve causal streaming methods integrate geometry from all past frames, causing moving objects to be repeatedly fused into the scene and leaving stacked replicas along their trajectories (ghosting artifacts), which grow with sequence length and can eventually corrupt nearby static surfaces. In contrast, Stream4D derives training-free dynamic masks from causal attention cues and applies soft key masking to suppress dynamic regions in the KV cache, so that dynamic objects appear only in the current frame while the background remains clean and sharp even in long sequences.

## 4.1 Dynamic Object Segmentation

We evaluate dynamic object segmentation on DAVIS-2016 [18], DAVIS-2017 [19], and DAVIS-all following the oficial DAVIS protocol and splits. Metrics are the mean IoU $\mathcal { T } _ { M }$ and the IoU recall J<sub>�</sub>, which jointly measure segmentation accuracy and robustness. All online methods are strictly single-pass and forbid futureframe access to comply with causal streaming constraints. Our method and the causal baselines use identical resolution and preprocessing to ensure a controlled comparison. Since the original StreamVGGT [59] lacks a segmentation head, we integrate the temporal attention aggregation to produce saliency maps, then apply temporal consistency filtering and thresholding to generate binary masks. Easi3R\_online keeps the original implementation and parameters, with the only change of disabling historical-frame access to match causal streaming. Ofline methods (DUSt3R [25], MonST3R [45], DAS3R [39], VGGT4D [5], and Easi3R [1] variants) use author-recommended configurations and serve as performance upper bounds.

Quantitative results. Table 1 presents the results on three DAVIS benchmarks. In the ofline setting, methods like VGGT4D and Easi3R\_monst3r establish strong upper bounds by leveraging bidirectional global attention over the entire video sequence. However, performance typically degrades significantly in the streaming setting due to the absence of future context and the inherent dificulty of incremental temporal reasoning. Existing streaming methods (CUT3R, Stream3R, StreamVGGT), originally designed for geometry estimation without native segmentation support, struggle to produce accurate dynamic object masks even with our carefully adapted implementation, highlighting a fundamental limitation of purely geometric streaming architectures.

![](images/2715f5ac7d83c443d21c7e56279f4ad64186497c19c8906e98909f23c6049e78.jpg)  
Figure 7: Qualitative comparison of estimated camera trajectories on three DyCheck sequences (creeper, handwavy, mochi-highfive). We plot the predicted trajectory (solid blue) against the ground truth (dashed gray) in the �-� plane. Left: Stream3R exhibits noticeable drift and erratic deviations, particularly in sequences with large dynamic motion (handwavy, mochi-high-five), where moving objects corrupt the pose estimation. Right: our Stream4D produces trajectories that more closely follow the ground truth, demonstrating that motion-decoupled masking efectively mitigates the interference of dynamic regions on camera pose estimation.

Our Stream4D substantially outperforms all streaming baselines across all three benchmarks, narrowing the gap with ofline methods. Compared to the strongest streaming competitor StreamVGGT, Stream4D achieves absolute gains of +8.48 (43.39 vs. 34.91) on DAVIS-2016, +9.12 (46.45 vs. 37.33) on DAVIS-2017, and +9.86 (43.06 vs. 33.20) on DAVIS-all in terms of mean IoU (J<sub>�</sub>). Similar improvements are consistently observed in recall (J<sub>�</sub>), demonstrating the efectiveness of our unified architecture for joint geometry and segmentation under strict streaming constraints.

Qualitative results. Fig. 5 presents representative sequences comparing the input frame, Easi3R masks, and our masks. Due to pairwise adjacent statistics only, Easi3R is prone to temporal jitter and boundary breaks under short-term occlusions and non-rigid deformations. In contrast, our streaming multi-level temporal aggregation with consistency constraints maintains stable segmentation and complete boundaries across both slow motions (e.g., camel, goat, pig, dog) and fast motions (e.g., skateboarder, motorcyclist);

masks evolve smoothly over time and align with true motion patterns, demonstrating the robustness and adaptability of our causal attention-based approach in online settings. More visualized results can be found in the appendix.

## 4.2 4D Reconstruction

We evaluate 4D reconstruction performance on the DyCheck dataset, a challenging benchmark comprising real-world monocular videos with complex object motion, primarily focusing on camera pose estimation accuracy. Following prior work, we report standard pose metrics including Absolute Trajectory Error (ATE), Relative Translation Error (RTE), and Relative Rotation Error (RRE). All streaming methods operate in single-pass causal mode, strictly forbidding access to future frames to ensure a fair and realistic evaluation protocol.

Quantitative results. Table 2 summarizes results on DyCheck. Ofline methods such as VGGT4D achieve strong performance by leveraging full-sequence optimization, improving over their respective bases (VGGT, MonST3R, DUSt3R). Among streaming methods, CUT3R—despite being extensively trained across diverse datasets—serves as a competitive concurrent baseline. Our method (Stream4D), operating training-free and within causal streaming constraints, delivers accurate and complete reconstructions comparable to or better than most baselines, and on pose metrics achieves consistent improvements over Stream3R (ATE 0.014, ↓6.7%; RTE 0.017, ↓5.6%; RRE 0.539, ↓1.1%), while maintaining identical reconstruction metrics to isolate the efect of motion decoupling.

![](images/f05bf829fcc6fc1285f58804954720037a67be72a5117f2711d0752e3126fa03.jpg)  
Figure 8: Inference time comparison across diferent sequence lengths. While Easi3R exhibits exponential time growth and sufers from Out-of-Memory (OOM) errors between 100 and 200 frames, Stream4D maintains linear scalability, performing comparably to the eficient StreamVGGT baseline.

Qualitative results. Fig. 7 compares the reconstruction quality of Stream3R [9] and Stream4D on representative dynamic sequences. As shown, the baseline Stream3R exhibits severe entanglement between dynamic and static elements, leading to geometric corruption, appearance distortion, and ghosting artifacts in regions of significant motion. In contrast, our Stream4D significantly improves reconstruction fidelity in this challenging streaming setting. The improvement stems from two key designs: (1) attention-based dynamic segmentation that robustly separates motion from the static background without requiring external supervision, and (2) soft key masking that suppresses dynamic interference during historical aggregation. More visualized results are provided in the appendix.

Ghosting suppression. A core failure mode of causal streaming reconstruction in dynamic scenes is temporal over-integration: as each new frame attends to and fuses information from the past, moving objects are repeatedly written into the scene, creating stacked geometric replicas along their motion paths (ghosting artifacts). This artifact intensifies with longer sequences and may eventually bleed into nearby static surfaces, degrading ground, building facades, and road textures. As illustrated in Fig. 6, baseline streaming methods exhibit clear motion trails—pedestrians and vehicles appear multiple times at diferent locations, forming overlapping silhouettes and blurred structures.

Stream4D suppresses this accumulation by decoupling motion from the causal fusion process. Specifically, we aggregate temporal saliency from causal attention to derive dynamic masks on-the-fly, and apply soft key masking to suppress dynamic regions in the cached keys. This prevents dynamic features from dominating historical aggregation, so that the static background remains stable and sharp, and dynamic objects are retained only in the current frame without overlapping replicas, even over long streaming sequences.

## 4.3 Eficiency and Scalability

We evaluate the inference time and scalability of our proposed method across varying image sequence lengths (� ∈ {10, 20, 50, 100, 200}) with an input resolution of 512 × 384 on a single NVIDIA A100 GPU. Figure 8 illustrates the comparison between Easi3R [1], StreamVGGT [59], and our Stream4D. As shown, Easi3R [1] sufers from severe computational bottlenecks due to its reliance on global pairwise attention. Its inference time grows quadratically, reaching 260.0s at � = 100. Crucially, due to its heavy memory footprint, Easi3R [1] fails to process longer sequences entirely, triggering an Out-of-Memory (OOM) error before reaching � = 200.

In contrast, Stream4D demonstrates robust scalability with nearlinear growth in computation. By leveraging a causal architecture similar to StreamVGGT, Stream4D maintains eficient inference even for long sequences. At � = 200, Stream4D achieves an inference time of 93.1s, which is comparable to the highly eficient StreamVGGT baseline (84.1s). The marginal increase in latency (≈10%) is attributed to the lightweight overhead introduced by our dynamic segmentation and soft key masking mechanisms. This result confirms that Stream4D successfully avoids the OOM issues of Easi3R while maintaining high computational eficiency for longterm 4D reconstruction, making it highly practical for continuous real-world video applications.

## 5 Conclusion

We present the Stream4D, a training-free framework for motiondecoupled 4D scene reconstruction tailored to causal streaming architectures. Our key insight is that causal attention implicitly encodes rich temporal motion cues within the KV cache. Leveraging multi-level attention aggregation with depth priors, we derive reliable dynamic masks that decouple static and dynamic regions. We then apply soft key masking to directly suppress dynamic regions in the cached keys, preventing dynamic features from interfering with historical aggregation. This design improves camera pose estimation accuracy and helps mitigate ghosting artifacts over long sequences, all under strict single-pass, causal inference constraints. Experiments on DAVIS and DyCheck demonstrate that Stream4D achieves competitive or superior performance compared to existing streaming baselines in both dynamic segmentation and pose estimation, while maintaining eficient inference for long-sequence 4D reconstruction.

Limitations and future work. While Stream4D demonstrates the potential of mining motion cues from causal attention, current aggregation relies primarily on attention statistics and depth priors. Future work will explore richer cue extraction from the KV cache, such as incorporating cross-scale temporal patterns and semantic-aware saliency. Beyond static-scene reconstruction, we aim to extend the framework toward dynamic object reconstruction by leveraging the decoupled motion representations.

## References

[1] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. 2025. Easi3r: Estimating disentangled motion from dust3r without training. arXiv preprint arXiv:2503.24391 (2025).

[2] Yasutaka Furukawa and Jean Ponce. 2009. Accurate, dense, and robust multiview stereopsis. IEEE transactions on pattern analysis and machine intelligence 32, 8 (2009), 1362–1376.

[3] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. 2022. Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems 35 (2022), 33768–33780.

[4] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. 2020. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2495–2504.

[5] Yu Hu, Chong Cheng, Sicheng Yu, Xiaoyang Guo, and Hao Wang. 2025. VGGT4D: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruc tion. arXiv preprint arXiv:2511.19971 (2025)

[6] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. 2024. Sgaussian: Self-supervised street gaussians for autonomous driving. CoRR (2024).

[7] Tao Huang, Rui Wang, Xiaofei Liu, Yi Qin, Li Duan, and Liping Jing. 2026. Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification. arXiv preprint arXiv:2602.05535 (2026).

[8] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. 2023. Tri-perspective view for vision-based 3d semantic occupancy prediction. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 9223–9232.

[9] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. 2025. Stream3r: Scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893 (2025).

[10] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. 2024. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision. Springer, 71–91.

[11] Hao Li, Yuhao Wang, Wenning Hao, Pingping Zhang, Dong Wang, and Huchuan Lu. 2026. RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 28179–28189.

[12] Zixu Li, Yupeng Hu, Zhiwei Chen, Qinlei Huang, Guozhi Qiu, Zhiheng Fu, and Meng Liu. 2026. ReTrack: Evidence-Driven Dual-Stream Directional Anchor Calibration Network for Composed Video Retrieval. In AAAI, Vol. 40. 23373– 23381.

[13] Zixu Li, Yupeng Hu, Zhiwei Chen, Haokun Wen, Xuemeng Song, and Liqiang Nie. 2026. COMBINER: Composed Image Retrieval Guided by Attribute-based Neighbor Relations. IEEE TIP (2026).

[14] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. 2025. MegaSaM: Accurate, fast and robust structure and motion from casual dynamic videos. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 10486– 10496.

[15] Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon. 2011. Kinectfusion: Real-time dense surface mapping and tracking. In 2011 10th IEEE international symposium on mixed and augmented reality. Ieee, 127–136.

[16] John Oliensis. 2000. A critique of structure-from-motion algorithms. Computer Vision and Image Understanding 80, 2 (2000), 172–214.

[17] Onur Özyeşil, Vladislav Voroninski, Ronen Basri, and Amit Singer. 2017. A survey of structure from motion\*. Acta Numerica 26 (2017), 305–364.

[18] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. 2016. A benchmark dataset and eval uation methodology for video object segmentation. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 724–732.

[19] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. 2017. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675 (2017).

[20] Zunhai Su, Weihao Ye, Hansen Feng, Keyu Fan, Jing Zhang, Dahai Yu, Zhengwu Liu, and Ngai Wong. 2026. XStreamVGGT: Extremely Memory-Eficient Stream ing Vision Geometry Grounded Transformer with KV Cache Compression. arXiv preprint arXiv:2601.01204 (2026).

[21] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[22] Hengyi Wang and Lourdes Agapito. 2024. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 (2024).

[23] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. 2025. Vggt: Visual geometry grounded transformer.

In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 5294– 5306.

[24] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. 2025. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference. 10510– 10522.

[25] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. 2024. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20697–20709.

[26] Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, et al. 2024. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19757– 19767.

[27] Weilin Wu, Shifan Yang, Qizhao Lin, XingHong Chen, Kunping Yang, Jing Wang, and Guannan Chen. 2025. A Novel Perspective on Low-Light Image Enhancement: Leveraging Artifact Regularization and Walsh-Hadamard Transform. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 12160–12169. doi:10.1145/3746027.3758142

[28] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. 2025. Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory. arXiv preprint arXiv:2507.02863 (2025).

[29] Yuqi Wu, Wenzhao Zheng, Sicheng Zuo, Yuanhui Huang, Jie Zhou, and Jiwen Lu. 2025. Embodiedocc: Embodied 3d occupancy prediction for vision-based online scene understanding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 26360–26370

[30] ZhengXian Wu, Hangrui Xu, Kai Shi, Zhuohong Chen, Yunyao Yu, Chuanrui Zhang, Zirui Liao, Jun Yang, Zhenyu Yang, Haonan Lu, et al. 2026. ProMSA: Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering. arXiv preprint arXiv:2606.27974 (2026).

[31] Zhengxian Wu, Chuanrui Zhang, Shen’Ao Jiang, Hangrui Xu, Zirui Liao, Luyuan Zhang, Li Huaqiu, Peng Jiao, and Haoqian Wang. 2026. Language-guided and motion-aware gait representation for generalizable recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 10871–10878.

[32] Zhengxian Wu, Chuanrui Zhang, Hangrui Xu, Peng Jiao, and Haoqian Wang. 2025. DAGait: Generalized skeleton-guided data alignment for gait recognition. In 2025 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 1–6.

[33] Xi Xiao, Xingjian Li, Yunbei Zhang, Cheng Han, Tianming Liu, Tianyang Wang, RunminJiang,Jihun Hamm, Xiao Wang, and Min Xu. 2026. Layer-Specific Prompt Fusion Discovery via Diferentiable Search in Vision Foundation Models. arXiv preprint arXiv:2606.26379 (2026).

[34] Xi Xiao, Chen Liu, Chih-Ting Liao, Yunbei Zhang, Qizhen Lan, Yuxiang Wei, Lin Zhao, Janet Wang, Jianyang Gu, Muchao Ye, et al. 2026. Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs. arXiv preprint arXiv:2606.26387 (2026).

[35] Xi Xiao, Chenrui Ma, Yunbei Zhang, Chen Liu, Zhuxuanzi Wang, Yanshu Li, Lin Zhao, Guosheng Hu, Tianyang Wang, and Hao Xu. 2026. Not all directions matter: Towards structured and task-aware low-rank model adaptation. In Proceedings of the 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 2132–2154.

[36] Xi Xiao, Yunbei Zhang, Xingjian Li, Tianyang Wang, Xiao Wang, Yuxiang Wei, Jihun Hamm, and Min Xu. 2025. Visual instance-aware prompt tuning. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 2880–2889.

[37] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. 2025. Spatialtrackerv2: Advancing 3d point tracking with explicit camera motion. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 6726–6737.

[38] Hangrui Xu, Zhengxian Wu, Chuanrui Zhang, Zhuohong Chen, Zhifang Liu, Peng Jiao, and Haoqian Wang. 2026. Psgait: Gait recognition using parsing skeleton. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 10427–10431.

[39] Kai Xu, Tze Ho Elden Tse, Jizong Peng, and Angela Yao. 2024. Das3r: Dynamics-aware gaussian splatting for static scene reconstruction. arXiv preprint arXiv:2412.19584 (2024).

[40] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaf, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. 2025. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference. 21924–21935.

[41] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10371–10381.

[42] Pei Yang, Wei Song, Xiaobing Zhao, Rui Zheng, and Letu Qingge. 2020. An improved Otsu threshold segmentation algorithm. International Journal of Computational Science and Engineering 22, 1 (2020), 146–153.

[43] David Yifan Yao, Albert J Zhai, and Shenlong Wang. 2025. Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 1116–1126.

[44] Shuai Yuan, Yantai Yang, Xiaotian Yang, Xupeng Zhang, Zhonghao Zhao, Lingming Zhang, and Zhipeng Zhang. 2026. InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams. arXiv preprint arXiv:2601.02281 (2026).

[45] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. 2024. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825 (2024).

[46] Songyan Zhang, Yongtao Ge, Jinyuan Tian, Guangkai Xu, Hao Chen, Chen Lv, and Chunhua Shen. 2025. POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction. arXiv preprint arXiv:2504.05692 (2025).

[47] Wei Zhang, Qiang Li, and Qi Wang. 2025. Refined Cascade Cost Volume for Multi-View Remote Sensing Image Reconstruction. IEEE Transactions on Geoscience and Remote Sensing (2025).

[48] Wei Zhang, Qiang Li, Yuan Yuan, and Qi Wang. 2024. Visual Consistency Enhancement for Multiview Stereo Reconstruction in Remote Sensing. IEEE Transactions on Geoscience and Remote Sensing 62 (2024), 1–11. doi:10.1109/TGRS.2024.3482697

[49] Wei Zhang, Songhua Li, Yihang Wu, Qiang Li, and Qi Wang. 2026. VGGT-CD: Training-Free Robust Registration for 3D Change Detection. arXiv:2605.16859 [cs.CV] https://arxiv.org/abs/2605.16859

[50] Wei Zhang, Yihang Wu, Songhua Li, Wenjie Ma, Xin Ma, Qiang Li, and Qi Wang. 2025. Review of feed-forward 3d reconstruction: From dust3r to vggt. arXiv preprint arXiv:2507.08448 (2025).

[51] Wei Zhang, Yihang Wu, Shengkai Yu, Songhua Li, Qiang Li, and Qi Wang. 2026. GPR-MVS: Global Propagation Regularization for Large Scale Multi-view Stereo. IEEE Transactions on Geoscience and Remote Sensing (2026), 1–1. doi:10.1109/ TGRS.2026.3710063

[52] Wei Zhang, Zhigang Yang, Qiang Li, and Qi Wang. 2025. Semantic-guided Multiview Stereo Reconstruction for Aerial Image. IEEE Transactions on Geoscience and Remote Sensing (2025).

[53] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. 2022. Structure and motion from casual videos. In European Conference on Computer Vision. Springer, 20–37.

[54] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. 2024. Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 19680–19690.

[55] Yan Zhong, Xingyu Wu, Xinping Zhao, Li Zhang, Xinyuan Song, Lei Shi, and Bingbing Jiang. 2026. Semi-supervised multi-label feature selection with consistent sparse graph learning. Neural Networks (2026), 109265.

[56] Yan Zhong, Chenxi Yang, Suyuan Zhao, and Tingting Jiang. 2025. Semi-supervised blind quality assessment with confidence-quantifiable pseudo-label learning for authentic images. In Forty-second International Conference on Machine Learning.

[57] Yan Zhong, Ruoyu Zhao, Chao Wang, Jiaqi He, Qinghai Guo, Jianguo Zhang, Zhichao Lu, and Luziwei Leng. 2026. Dyn-SSM: Towards the Eficient Long Sequence Learning via Bio-interpretable Dynamics in Spiking State Space Models. IEEE Transactions on Cognitive and Developmental Systems (2026).

[58] Yan Zhong, Xinping Zhao, Li Zhang, Xinyuan Song, and Tingting Jiang. 2025. Adaptive Prompt Learning for Blind Image Quality Assessment with Multi-modal Mixed-datasets Training. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 7453–7462

[59] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. 2025. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539 (2025).