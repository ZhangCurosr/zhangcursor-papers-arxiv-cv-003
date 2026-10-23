# UniVOE: A Removal-Driven Framework for Interaction-Aware Universal Video Object Editing

Shuzhou Yang<sup>1†</sup> Lanqing Guo<sup>2∗</sup> Lingen Li<sup>3</sup> Weiqi Li<sup>1</sup>

Zhao Zhong<sup>2</sup> Liefeng Bo<sup>2</sup> Jian Zhang<sup>1∗</sup>

<sup>1</sup>SECE, Peking University <sup>2</sup>Tencent Hunyuan <sup>3</sup>The Chinese University of Hong Kong

https://univoe.github.io/

![](images/1e65f771f0c3d3f431fdc6bf2807587f2e8688c6c387285079e930ed5d13e632.jpg)  
Figure 1: We propose UniVOE, a video object editing pipeline that enables object-centered removal and injection, together with predictions of changes in object-related efects. Such as footprint, shadow, reflection, and lighting. Users can easily remove or insert target objects using prompts, as shown above each editing result.

## Abstract

Video object removal and injection are reciprocal editing operations between object-present and object-absent video states, yet they difer substantially in training dificulty. Removal can be efectively bootstrapped from video inpainting and further adapted to erase object-related interaction efects. Injection, however, requires large-scale paired supervision to model object appearance, dynamics, and scene-dependent effects, while perfectly aligned real-world pairs are dificult to collect and synthetic data lacks suficient realism and diversity. To address this challenge, we present UniVOE, a removaldriven framework for training both video object removal and

injection models. We first train an efect-aware remover by transferring video inpainting priors and finetuning on objectpresent/object-absent pairs. The trained remover converts real object-present videos into object-absent counterparts, creating large-scale paired supervision for injection. An injection model is trained on these pairs in the reverse direction to recover objects together with their dynamics and interaction effects. Moreover, since pixel-level reconstruction alone cannot adequately capture perceptual realism, we introduce a rewardbased post-training method that aligns injection results with human preferences using a proprietary reward model. Experiments against strong open-source and commercial systems demonstrate that UniVOE achieves state-of-the-art performance on both object removal and injection, particularly in challenging interaction-efect cases.

## Introduction

Video object removal and injection are critical in visual content creation, with broad applications in film post-production, advertising, and virtual production. Unlike image editing, video editing must account for not only the target object itself, but also its dynamic imprint on surrounding scene. A moving object may cast shadows, produce reflections, and contact surfaces. Editing only the object but leaving these interaction efects untreated during removal or synthesizing them incorrectly in injection will produce perceptible artifacts. A practical tool should jointly edit the object and its efects.

Recent difusion-based video editing methods made significant progress in fidelity and temporal consistency (Yang et al. 2026c; Li et al. 2025a; Zhou et al. 2023; Kushwaha et al. 2026; Yang et al. 2026a). Some works further recognize the close relationship between object removal and insertion (Fu et al. 2026): efect-aware removers erase shadows and reflections together with the target object, while reciprocal formulations use insertion as an auxiliary inverse task to improve removal. General video inpainting systems also support adding or removing objects (Bian et al. 2025). Nevertheless, realistic object injection remains dificult. It must hallucinate absent foreground, infer plausible object motion, and predict interactive efects such as shadows and reflections. The main bottleneck is supervision. Ideally, both directions should be trained with paired videos of the same scene in object present and absent states under identical lighting, and background dynamics. Such pairs are hard to capture in the wild, while synthetic rendering, despite perfect alignment, lacks diversity, photorealism, and dynamics of real footage.

Our key insight is that object removal and injection share the same paired data structure, but are asymmetric in learning dificulty. Let $V _ { \mathrm { o b j } }$ and $V _ { \mathrm { a b s } }$ denote the object present and absent states of the same video. The two tasks can be viewed as conditional generation in opposite directions: $\pi _ { \mathrm { r e m } } ( V _ { \mathrm { a b s } } \mid V _ { \mathrm { o b j } } , c ) \longleftrightarrow \pi _ { \mathrm { i n j } } ( V _ { \mathrm { o b j } } \mid V _ { \mathrm { a b s } } , c ) .$ , where c denotes the editing condition, such as text prompt in Fig. 1. The same pair can supervise both directions: removal learns to eliminate the object and its efects, while injection learns to predict them. However, removal can be scaled from video inpainting, which provides abundant supervision for background completion, temporal structure, and scene consistency. With limited paired $\bar { V _ { \mathrm { o b j } } } – \bar { V _ { \mathrm { a b s } } }$ data, this prior can be adapted to complete object removal and efect erasing. Injection, in contrast, requires cleaner supervision for object appearance, motion, and interaction. This reciprocal-but-asymmetric structure motivates removal-driven strategy: strengthen easier removal direction, and use it to derive supervision for injection.

We propose UniVOE, a removal-driven framework for realistic video object removal and injection. As shown in Fig. 2, UniVOE first trains an efect-aware remover by transferring scalable inpainting priors to object removal and refines with limited object-present/absent pairs (Miao et al. 2026; Fu et al. 2026). The trained remover is then applied to real objectpresent videos to derive object-absent counterparts, while the original videos serve as injection targets containing real object motion and interaction efects. These removal-derived pairs, together with limited open-source data, supervise an injection model with flow-matching (Lipman et al. 2023). However, pixel-wise supervision alone cannot fully close the perceptual gap of injection: for an object-absent video, multiple appearances, trajectories, and contact patterns may be plausible, while failures such as over-saturation, unnatural motion, missing contact shadows, or unrealistic reflections are better captured by perceptual judgment than by reconstruction error. We therefore introduce an online post-training stage tailored to object injection. It employs a proprietary reward model, finetuned from Qwen3-VL (Bai et al. 2025), to rank the motion quality and naturalness of edited videos, steering optimization toward model’s performance ceiling. More importantly, as an editing task, the complete frame sequence should be fully covered during ranking and training. To make online post-training feasible for large difusion models at realistic video resolutions and durations, we design a VRAM-eficient Group-Relative Advantage Regression (GRAR). It avoids the memory overhead of maintaining a reference policy while preserving reward-guided training. Experiments demonstrate that UniVOE achieves SOTA empirical results on both video object removal and injection. As shown in Fig. 1, for removal, it robustly eliminates target objects together with their shadows, reflections, and contact cues. For injection, it synthesizes objects with plausible motion and interaction efects. Compared with strong opensource and commercial systems, UniVOE is competitive and clearly stronger in challenging interaction-efect cases that prior methods often mishandle. Our contributions are summarized as follows:

![](images/8c53d6f32d86585bf87bc5fc352731fcef119b89515de36cded498e5e1b32a50.jpg)  
Figure 2: UniVOE uses removal to scale paired video data up, drives the training of injection.

• We propose an efective training framework to optimize video object removal and injection models with limited real-world paired data, named UniVOE. We exploit their reciprocal-but-asymmetric training structure: removal is first scaled with easier inpainting task and then reused as a supervision engine for the harder injection direction.

• To enhance the realism of injection, we develop a novel post-training algorithm, called GRAR, to overcome the limitation of pixel-level reconstruction and align the injected results with human preference. Experiments verify its efectiveness in improving injection quality.

• We apply various techniques to removal model for largescale training data curation. First, object deletion is bootstrapped with video inpainting. Next, we adapt the model with paired data to enable interactive efects removal.

Besides, we accelerate inference to facilitate large-scale generation. Finally, we add an efective video filtering mechanism to retain only high-quality data for training.

• Extensive comparisons with open-source and commercial systems show that UniVOE achieves SOTA performance on both object removal and injection, especially in challenging interaction-efect erasing and prediction.

## Related Work

## Video Object Removal

Video object removal is a special case of video inpainting: the masked region must be filled with plausible and temporally coherent background. Early methods rely on temporal adversarial losses, optical-flow propagation, or spatio-temporal attention to improve completion quality (Chang et al. 2019; Zeng, Fu, and Chao 2020; Liu et al. 2021; Li et al. 2022; Zhou et al. 2023; Zhang, Rao, and Agrawala 2023; Gu et al. 2025; Gao et al. 2025), recent difusion-based pipelines extend the generative range to large holes (Li et al. 2025a; Bian et al. 2025). These generic inpainters fill the masked region but are not designed to clean up object-induced efects outside the mask. A second line of work, namely efect-aware removal, targets the shadows, reflections, mirror images, and deformations that an object leaves on the surrounding scene (Miao et al. 2026; Kushwaha et al. 2026). EfectErase (Fu et al. 2026) treats object insertion as an auxiliary inverse task to regularize removal. In contrast, we treat removal as a supervision engine for injection: a trained efect-aware remover converts object-present videos into object-absent counterparts for paired supervision of object injection.

## Video Object Injection

Object injection places a target object into a host scene while keeping it visually compatible with background. At image level, prior work has explored exemplar-guided editing, object compositing, and zero-shot object-level customization (Yang et al. 2023; Song et al. 2023; Lu, Liu, and Kong 2023; Chen et al. 2024; Li et al. 2025b). Lifting these to video is harder: the inserted object must remain temporally consistent and move coherently. Recent video editing methods add controllability through first-frame editing, reference-object preservation, or compositing (Ku et al. 2024; Mou et al. 2024; Wei et al. 2024; Tu et al. 2025; Yang et al. 2026b; Ye et al. 2025), but they typically expect the user to specify object identity, motion track, and other injection-related controls up front. Unlike these methods, UniVOE ofers lighter interface (text by default, with an optional reference image). This enables flexible use and large-scale paired data curation.

## Post-training for Preference Alignment

Aligning generation results with human perception generally takes one of two routes. Ofline preference optimisation (Rafailov et al. 2023; Xu et al. 2023; Prabhudesai et al. 2024; Wallace et al. 2024) fine-tunes the model on a fixed set of pairwise preferences. It is simple and stable, but bound to expensive human annotation, and the model never trains on its own samples. Online post-training instead scores samples drawn from the current policy on the fly, which saves manual annotation and allows the model to explore its higher performance boundaries. Within online for difusion and flow-matching, recent work explores PPO-style trust regions (Liu et al. 2026) and negative-aware fine-tuning, exemplified by DifusionNFT (Zheng et al. 2025). Although these are efective on image models, applying them of-theshelf to large video backbones for editing tasks runs into a memory wall: keeping a frozen reference policy alongside a trainable one, plus storing rollouts of long video latents, needs larger VRAM. Therefore, we propose a VRAMeficient post-training strategy tailored to video editing. It follows a single-pass reward-weighted regression but discards the reference policy, named GRAR.

## Method

UniVOE contains two models, an efect-aware remover $\pi _ { \mathrm { r e m } }$ and an injector $\pi _ { \mathrm { i n j } }$ , which share the same structure but are trained for opposite directions of object present/absent transformation. Both are built on Wan2.1-14B (Wan et al. 2025).

## Preliminaries: Latent Flow Matching

Let $x _ { 0 } \in \mathbb { R } ^ { C \times T \times H \times W }$ denote the latent of a clean video, and let $\epsilon \sim \mathcal { N } ( 0 , I )$ be Gaussian noise of the same shape. For timestep $t \sim \dot { \mathcal { U } } [ 0 , \dot { 1 } ]$ ], we use the rectified-flow interpolant:

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon , \qquad v ^ { \star } ( x _ { 0 } , \epsilon , t ) = \partial _ { t } x _ { t } = \epsilon - x _ { 0 } .\tag{1}
$$

A network $v _ { \theta } ( x _ { t } , t \mid c )$ predicts velocity conditioned on c. In our case, c includes an input video, video mask, prompt, and an optional reference image. Flow-matching objective is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } _ { 0 } , \boldsymbol { \epsilon } , \boldsymbol { t } , \boldsymbol { c } } \left[ \Vert \boldsymbol { v } _ { \theta } ( \boldsymbol { x } _ { t } , \boldsymbol { t } \mid \boldsymbol { c } ) - \boldsymbol { v } ^ { \star } ( \boldsymbol { x } _ { 0 } , \boldsymbol { \epsilon } , \boldsymbol { t } ) \Vert _ { 2 } ^ { 2 } \right] . } \end{array}\tag{2}
$$

At inference, samples are generated by using the learned velocity field from noise to data. Both $\pi _ { \mathrm { r e m } }$ and $\pi _ { \mathrm { i n j } }$ ofUniVOE optimize with this objective, but difer in condition c.

## Stage 1: Efect-Aware Video Object Removal

As Fig. 2 shows, video inpainting and object removal share a common goal: recovering background occluded by target region. This similarity allows us to bootstrap object removal from inpainting data. Further, removal requires erasing object-induced interaction efects, $e . g .$ , shadows and reflections that extend beyond object. We train remover in two steps: inpainting pretraining and efect-aware fine-tuning.

Video inpainting. We employ SAM2 (Ravi et al. 2025) to segment foreground objects in real-world videos, obtain their masks $M$ . We collect over 63K high-quality videos containing salient objects and apply Gaussian dilation to M to ensure suficient coverage of object boundaries. In training, we randomly sample two videos, $V i d _ { A }$ and $V i d _ { B } .$ , and use M from $V i \bar { d } _ { B }$ to mask corresponding regions in $V i d _ { A }$ producing realistic object-shaped missing regions with temporal variations. The masked $V i d _ { A }$ is used as input, while the original $V i d _ { A }$ serves as the ground truth. The removal model is optimized with the reconstruction loss in Eq. (2), enabling it to learn a strong scene completion prior for plausibly recovering background content occluded by objects.

![](images/8a50ae72e2f3bf5d30f07faad3a99efad358b9449fd93633cdb977dd46424c73.jpg)  
Figure 3: Workflow of UniVOE. (a) The remover and injector share the same network, but use diferent instructions and VOE units. (b) For removal, VOE unit contains an input video and an inflated binary mask video to indicate object to be removed. For injection, an all-one mask is used to enable global interaction-aware injection. The reference image is optional in removal and injection. It is generated with external image editing model, providing more specific editing guidance. (c) To improve physical realism of injected results, we propose GRAR to update the pre-trained injection model toward human preferences.

Efect-aware fine-tuning. An inpainting model fills content inside the masked region, while object removal requires eliminating object-induced efects, such as shadows, reflections, or other interactions that may extend beyond the object mask. To make the remover efect-aware, we further fine-tune it on synthesized and captured object-present/object-absent video pairs collected from prior datasets (Miao et al. 2026; Fu et al. 2026), yielding 68K filtered training pairs. During training, we use inflated object mask M that fully covers the target object but leaves interaction efects outside. This encourages the model to recover the complete object-absent scene rather than merely inpainting the object region, while improving robustness to inaccurate masks at inference time. Process is formulated as: $\pi _ { \mathrm { r e m } } \left( V _ { \mathrm { a b s } } \mid V _ { \mathrm { o b j } } , M , c _ { \mathrm { r e m } } \right)$ , where $V _ { \mathrm { o b j } }$ and $V _ { \mathrm { a b s } }$ denote the object-present and object-absent videos, respectively. Removal condition $c _ { \mathrm { r e m } }$ supports both text-only and reference-guided modes:

$$
c _ { \mathrm { r e m } } = { \left\{ \begin{array} { l l } { p , } & { { \mathrm { w i t h o u t a r e f e r e n c e ~ i m a g e } } , } \\ { ( p , I _ { \mathrm { r e f } } ) , } & { { \mathrm { w i t h a r e f e r e n c e ~ i m a g e } } , } \end{array} \right. }\tag{3}
$$

where p denotes the instruction as shown in Fig. $3 ( \mathrm { a } ) , I _ { \mathrm { r e f } }$ is a reference image specifying the desired object-absent state.

As shown in Fig. 3(b), the reference-guided mode provides an explicit cue in addition to the input video and mask. Specifically, we edit the first frame of $V _ { \mathrm { o b j } }$ using the same removal instruction p with HunyuanImage3 (Cao et al. 2025), producing $I _ { \mathrm { r e f } }$ in which the target object has been removed. When $I _ { \mathrm { r e f } }$ is provided, UniVOE prepends it to $V _ { \mathrm { o b j } }$ as an additional conditioning frame. Correspondingly, one noise frame is prepended to the latent noise sequence and one allzero mask frame is prepended to the mask sequence, so that the video, noise, and mask remain temporally aligned. The reference frame provides an explicit but static object-absent prior, guiding the model to remove the dynamic object consistently with the edited first frame throughout the video.

Inference Acceleration. When applying remover to largescale data curation, its inference cost directly limits the scale of paired data we can generate. To accelerate it, we distill remover from the default 50-steps sampling into 12-steps. Concretely, we shorten the flow-matching sampler to 12-steps with a fixed sigma shift of 5.0, and finetune with Eq. (2). To synchronize timesteps of the backbone and branch, we add a LoRA (Hu et al. 2022) of rank 32 to attention and feedforward layers of backbone, jointly update the full branch together with LoRA. In inference, we directly remove objects in 12-steps, making large-scale object-absent video generation practical. Note that our removal results in this paper are all generated from distilled 12-steps model, which yields a substantial speed-up with imperceptible quality degradation.

## Stage 2: Removal-Driven Injection Pre-training

Given a collection of real object-present videos $\{ V _ { \mathrm { o b j } } ^ { ( i ) } \}$ , we apply the trained remover $\pi _ { \mathrm { r e m } }$ to derive paired supervision:

$$
V _ { \mathrm { a b s } } ^ { ( i ) } = \pi _ { \mathrm { r e m } } ( V _ { \mathrm { o b j } } ^ { ( i ) } , \tilde { M } ^ { ( i ) } ) .\tag{4}
$$

The derived video $V _ { \mathrm { a b s } } ^ { ( i ) }$ serves as the injection input, while the original video $V _ { \mathrm { o b j } } ^ { ( i ) }$ naturally becomes a photorealistic target containing real object motion and interaction effects. Each removal-derived training sample is represented as $\left( V _ { \mathrm { a b s } } ^ { ( i ) } , V _ { \mathrm { o b j } } ^ { ( i ) } , p ^ { ( i ) } , I _ { \mathrm { r e f } } ^ { ( i ) } \right)$ , where $\boldsymbol { p } ^ { ( i ) }$ is the injection prompt and $I _ { \mathrm { r e f } } ^ { ( i ) }$ is an optional reference image. We generate object prompts from the target video $V _ { \mathrm { o b j } } ^ { ( i ) }$ using Qwen3-VL. As shown in Fig. 3(b), $I _ { \mathrm { r e f } } ^ { ( i ) }$ provides specific appearance and interaction constraints. During training, it can be taken from the first frame of $V _ { \mathrm { o b j } } ^ { ( i ) }$ . In inference, it is obtained by applying image editing tool (Cao et al. 2025) to the first frame of $V _ { \mathrm { a b s } } ^ { ( i ) } .$ enabling more explicit injection control.

Totally, we employ removal model to produce 69K usable paired data, combined with the previously filtered 68K opensource data, all 137K video pairs are used to train injection model. We define injection condition as $c _ { \mathrm { i n j } } = ( V _ { \mathrm { a b s } } , \bar { p } , I _ { \mathrm { r e f } } )$ The injector can be trained with:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { ( V _ { \mathrm { o b j } } , c _ { \mathrm { i n j } } ) \sim \mathcal { D } _ { \mathrm { i n j } } , \epsilon , t } \left[ \| \big ( v _ { \theta } ( x _ { t } , t \mid c _ { \mathrm { i n j } } ) - v ^ { \star } \big ) \| _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

where $\mathcal { D } _ { \mathrm { i n j } }$ denotes the union of training data. The resulting injection model is denoted as $\pi _ { \mathrm { i n j } }$ , which is trained to inject not only the object itself, but also its related realistic interaction efects such as shadows, reflections, contact, etc.

## Stage 3: Post-Training with GRAR

Although Stage 2 provides basic object injection ability, pixel-wise supervision cannot fully capture perceptual quality. For an object-absent video, multiple appearances, dynamics, and interaction patterns can be plausible, while artifacts such as over-saturation, floating objects, implausible reflections, and incoherent motion are perceptually critical. Hence, we introduce online post-training to improve realism and interaction plausibility.

Given the injection model $\pi _ { \mathrm { i n j } }$ , we generate $V _ { \mathrm { g e n } } \sim \pi _ { \theta } ^ { \mathrm { i n j } } ( \cdot |$ $V _ { \mathrm { a b s } } , c _ { \mathrm { i n j } } )$ and compute a reward $r = Q _ { \theta } ( V _ { \mathrm { g e n } } ) .$ , where $Q _ { \theta }$ evaluates object motion and appearance quality. Since native Qwen3-VL cannot reliably assess such qualities, we collect professionally annotated rankings on motion quality and appearance realism and use them to finetune Qwen3-VL into a reward model aligned with human preferences.

In each iteration, we rollout K samples for each condition:

$$
\{ V _ { \mathrm { g e n } } ^ { ( j ) } \} _ { j = 1 } ^ { K } \sim \pi _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { i n j } } ( \cdot \mid V _ { \mathrm { a b s } } , c _ { \mathrm { i n j } } ) ,\tag{6}
$$

and score them through $Q _ { \theta }$ . Based on the relative advantages within the group, assign positive or negative signs to $\dot { \kappa }$ scores. We first calculate the group median $\tilde { r } _ { c } ,$ then assign:

$$
A ^ { ( j ) } = \left\{ { + 1 , } \atop { { \cal Q } _ { \theta } ( V _ { \mathrm { g e n } } ^ { ( j ) } ) < \tilde { r } _ { c } } , \right.\tag{7}
$$

This positive or negative sign depends on the in-group ranking, making the update bounded and robust to reward scores.

Advantage regression. Post-training methods mainly require double models, a reference policy, a trainable policy, with multiple forward passes. This is because, limited by reward hacking, updates driven solely by rewards cause the model to overfit or collapse into a spurious optimum, far removed from the high-quality distribution learned in pretraining. Hence, reference policy is necessary to constrain post-training scope, but inevitably increases VRAM overhead, limiting its potential for use in larger-scale models. Existing methods are mainly applied to image models or small video backbones such as 1.3B (Pan et al. 2025). But as to stronger video model such as our injector with 23B parameters, using of-the-shelf solutions is infeasible. One workaround is to reduce VRAM usage by sampling partial frames for training. However, for video editing tasks, abnormal editing efects in any frame are intolerable, i.e., training with full frame sequence is essential for editing. In our case, using 97 frames of 480p leads to unacceptably VRAM load.

To this end, we propose Group-Relative Advantage Regression (GRAR), a single-policy surrogate that preserves preference-aligned optimization without a separate reference model. We rethink the role ofreference policy from the nature of video editing. Reference policy constrains the magnitude of post-training for generation. However, for editing, input video itself constrains the output content, $i . e . .$ , the model does not generate a new video from scratch but merely modifies parts, with the core content directly dictated by input video. Therefore, GRAR only regress single trainable policy with reward values without redundant reference policy.

For each rollout $j ,$ we treat the generated video as the online sample $x _ { 0 } ^ { ( j ) }$ and recompute its flow-matching target following Eq. (1). GRAR objective is then formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G R A R } } ( \theta ) = \mathbb { E } _ { c _ { i n j } , j , t } \left[ A ^ { ( j ) } \left. v _ { \theta } ( x _ { t } ^ { ( j ) } , t \mid c _ { i n j } ) - v ^ { \star , ( j ) } \right. _ { 2 } ^ { 2 } \right] . } \end{array}\tag{8}
$$

When $A ^ { ( j ) } = + 1$ , minimizing Eq. (8) pulls the model toward the velocity field of high-reward rollouts. When $A ^ { ( j ) } = - 1$ the sign reverses the update and pushes the model away from low-reward rollouts. In this way, GRAR provides a negative-aware online update while requiring only a single trainable policy during optimization, making online posttraining feasible for large-scale video editing models.

Online training loop. At each iteration, we sample a batch of 8 contexts and generate $K = 1 6$ rollouts per context using $\pi _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { i n j } }$ . We score them with $Q _ { \theta }$ and compute signed advantages using Eq. (7). In optimization, only $\pi _ { \theta } ^ { \mathrm { i n j } }$ is forwarded to minimize Eq. (8), avoiding reference policy storage and forward passes. After each update, we synchronize the rollout policy as $\theta _ { \mathrm { o l d } }  \theta$ . Final policy serves as UniVOE injector.

## Experiments

## Implementation Details

Both remover and injector share the same model structure, each contains 23B parameters, including a 14B frozen Transformer backbone, a 3B trainable control branch, and other modules such as VAE, etc. To support practical video editing lasting over nine seconds (i.e., more than 200 frames), we train with 97 frames of video clip with resolution of 832×480. Learning rate for remover and injector pre-training is 1e-5, and injector post-training is 1e-6. During inference, both remover and injector support 2 conditioning paradigms: text-only or text-and-reference image. As some compared methods require reference image to inject objects. For fair comparison, here we compare removal with text-only input and compare injection with text-and-reference input.

## Comparison on Object Removal

We compare removal model with 5 SOTA open-source methods, including VideoPainter (SIGGRAPH 2025) (Bian et al.

UniVOE (Ours)  
Input frames  
VideoPainter  
Seedance2.0  
VACE  
EffectErase  
![](images/a2c18e90ca9b7126ddd6bc4214b1e47ffd812597fe68f7587e6c6f600782b0d3.jpg)  
Figure 4: Visual comparison on video object removal. Other methods either remain object-related efects traces (e.g., lighting of the car), or predict unnatural background contents. In contrast, our method removes object itself and recovers physically realistic scene. It outperforms both open-source and commercial models. More visual comparison can be found in supplementary.

Place a man wearing a black t-shirt, brown shorts, and white sneakers, carrying a green backpack, walking up the stairsfrom the lower landing.  
![](images/0e41b1ee0a1bd4ef4547126c36614543507f1e4e696c67631ce4621305e3afd9.jpg)  
Figure 5: Visual comparison on video object injection. Corresponding prompt is given above. Other methods either inject background-inconsistent object, or fail to inject harmonious object. Our model predicts matching target and credible dynamics.

<table><tr><td>Data</td><td>Metric</td><td>VidPaint</td><td>VACE</td><td>Minimax</td><td>EffErase</td><td>Bernini</td><td>Kling3</td><td>Seedance2</td><td>Ours</td></tr><tr><td rowspan="4">VOR</td><td>PSNR ↑</td><td>17.425</td><td>11.567</td><td>17.243</td><td>23.210</td><td>16.380</td><td>21.581</td><td>20.002</td><td>23.752</td></tr><tr><td>SSIM↑</td><td>0.705</td><td>0.601</td><td>0.692</td><td>0.734</td><td>0.674</td><td>0.712</td><td>0.715</td><td>0.786</td></tr><tr><td>LPIPS ↓</td><td>0.253</td><td>0.441</td><td>0.229</td><td>0.142</td><td>0.247</td><td>0.182</td><td>0.204</td><td>0.130</td></tr><tr><td>FVD↓</td><td>563.08</td><td>825.69</td><td>171.66</td><td>150.82</td><td>238.47</td><td>210.58</td><td>174.36</td><td>146.37</td></tr><tr><td rowspan="4">ROSE</td><td>PSNR ↑</td><td>25.693</td><td>15.377</td><td>28.839</td><td>24.666</td><td>23.814</td><td>22.433</td><td>27.286</td><td>29.596</td></tr><tr><td>SSIM↑</td><td>0.896</td><td>0.388</td><td>0.745</td><td>0.918</td><td>0.876</td><td>0.903</td><td>0.912</td><td>0.919</td></tr><tr><td>LPIPS ↓</td><td>0.1205</td><td>0.3528</td><td>0.1178</td><td>0.100</td><td>0.136</td><td>0.115</td><td>0.106</td><td>0.098</td></tr><tr><td>FVD↓</td><td>291.81</td><td>755.80</td><td>146.69</td><td>111.55</td><td>184.73</td><td>201.16</td><td>128.42</td><td>85.25</td></tr></table>

Table 1: Comparison on removal. The best results are highlighted in bold. Video materials are given in supplementary.

2025), VACE (ICCV 2025) (Jiang et al. 2025), Minimax-Remover (NeurIPS 2026) (Zi et al. 2026), EfectErase (CVPR 2026) (Fu et al. 2026), and Bernini (Team et al. 2026), and commercial models Kling3-Omni (Team et al. 2025) and Seedance2.0 (Seedance et al. 2026). Note that we only input text prompt to the removal model. As shown in Fig. 4, we compare removal performance on various interaction effects, e.g., heavy occlusion and obvious lighting. One can see that among all State-Of-The-Art (SOTA) methods, our method realizes the best object removal and efects recovery. Especially for the bottom cases, where the glass and oven are removed, respectively. For glass case, our model recovers naturally flowing milk but others fail. For oven case, other methods all remain the shadow of removed oven, resulting sub-optimal recovered background efected by it. In contrast, UniVOE precisely recovers a pizza in direct sunlight and its shadow. We believe this is because UniVOE is firstly trained from massive inpainting data, which brings strong object deletion and background recovery priors.

<table><tr><td>Data</td><td>Metric</td><td>VidPaint</td><td>VACE</td><td>UniVideo</td><td>Bernini</td><td>Kling3</td><td>Seedance2</td><td>Ours</td></tr><tr><td rowspan="4">VOR</td><td>DINO-I↑</td><td>0.310</td><td>0.232</td><td>0.387</td><td>0.327</td><td>0.303</td><td>0.381</td><td>0.390</td></tr><tr><td>CLIP-P↑</td><td>0.224</td><td>0.200</td><td>0.228</td><td>0.208</td><td>0.227</td><td>0.217</td><td>0.232</td></tr><tr><td>LPIPS ↓</td><td>0.257</td><td>0.366</td><td>0.314</td><td>0.329</td><td>0.295</td><td>0.249</td><td>0.237</td></tr><tr><td>FVD↓</td><td>391.63</td><td>505.02</td><td>300.69</td><td>337.48</td><td>296.67</td><td>259.31</td><td>234.56</td></tr><tr><td rowspan="4">ROSE</td><td>DINO-I↑</td><td>0.241</td><td>0.192</td><td>0.346</td><td>0.319</td><td>0.312</td><td>0.334</td><td>0.347</td></tr><tr><td>CLIP-P↑</td><td>0.220</td><td>0.217</td><td>0.236</td><td>0.242</td><td>0.221</td><td>0.243</td><td>0.248</td></tr><tr><td>LPIPS↓</td><td>0.130</td><td>0.224</td><td>0.279</td><td>0.245</td><td>0.153</td><td>0.161</td><td>0.114</td></tr><tr><td>FVD↓</td><td>351.42</td><td>352.24</td><td>282.01</td><td>238.72</td><td>177.63</td><td>143.56</td><td>85.12</td></tr></table>

Table 2: Comparison on injection. The best results are highlighted in bold. Video materials are given in supplementary.

![](images/82467c70db851a90978a84569c37bae7a9bcab17aa3a3916a93cf4d270626ea0.jpg)  
Figure 6: Visual ablation of training data used for the remover. “w/o inpainting” is trained by noisy removal pairs and carries the risk of overall color shifts. “UniVOE-” introduces inpainting data to avoid excessive background editing.

<table><tr><td>Settings</td><td>Inpaint</td><td>Removal</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FVD↓</td></tr><tr><td>w/o inpainting</td><td>0K</td><td>68K</td><td>20.578</td><td>0.711</td><td>0.158</td><td>163.130</td></tr><tr><td>UniVÓE-</td><td>34K</td><td>34K</td><td>22.891</td><td>0.740</td><td>0.141</td><td>149.777</td></tr></table>

Table 3: Ablation on inpainting data for training remover.

For quantitative evaluation, we compare on two benchmarks: ROSE containing 56 pairs, and VOR containing 100 pairs. Note that test sets are excluded from the training data. We employ fidelity metrics to compare objectively, including PSNR (Hore and Ziou 2010), SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018), and FVD (Unterthiner et al. 2019). As shown in Tab. 1, our method achieves the best objective metrics across benchmarks, which means it removes objects and efects, and recovers the most authentic background. This is consistent with subjective perception, further proves the efectiveness of the proposed new training method, i.e., starting removal training from video inpainting.

## Comparison on Object Injection

We compare UniVOE-injector with 4 open-source methods: VideoPainter, VACE, UniVideo (ICLR 2026) (Wei et al. 2026), and Bernini (Team et al. 2026), and commercial models, Kling3-Omni and Seedance2.0. Similar to UniVideo, our injector is also a mask-free approach, capable of injecting based on prompts without explicit mask inputs. This setting is more challenging and practical for large-scale batch video injection. As shown in Fig. 5, our method generates the most realistic injected results. VideoPainter, VACE, and UniVideo fail to place realistic objects. Although Kling3 and Seedance2 can produce competitive results, our outputs show more harmonious appearances and natural dynamics. Source videos can be found in supplementary for intuitive illustration.

We conduct quantitative comparison on the same test sets used before. Similar to previous injection method (Wei et al. 2026), we employ LPIPS, FVD, DINO-I (Caron et al. 2021), and CLIP-P, where DINO-I aims to measure identity consistency, CLIP-P measures prompt following. As shown in Tab. 2, our method outperforms all algorithms compared across benchmarks, which proves our superior injection ability and is consistent with visual results.

<table><tr><td>Settings</td><td>DINO-I↑</td><td>CLIP-P↑</td><td>LPIPS ↓</td><td>FVD↓</td></tr><tr><td>w/o GRAR</td><td>0.353</td><td>0.221</td><td>0.307</td><td>361.286</td></tr><tr><td>UniVOE</td><td>0.390</td><td>0.232</td><td>0.237</td><td>234.562</td></tr></table>

Table 4: Ablation on post-training stage for object injection.

## Ablation Study

Scaling up removal data through inpainting. We realize SOTA removal ability by bootstrapping with inpainting. We compare two ablation settings to support this claim: (1) “UniVOE-”, a degraded model trained on 68K hybrid data, including 34K inpainting pairs and 34K removal pairs. (2) “w/o inpainting” that is totally trained on 68K removal pairs. Note that UniVOE- only uses part of inpainting and removal pairs to train, which aims to maintain the same data scale as “w/o inpainting” for fair comparison. Quantitative comparison is given in Tab. 3. Although “UniVOE-” is degraded by retraining on smaller data scale, it still outperforms “w/o inpainting” across all metrics. Visual results in Fig. 6 show that “w/o inpainting” may shift overall tone. This is because the paired object-present/absent pairs capture in the wild cannot ensure pixel-level background consistency, using such noisy data to train misleads the model to over-edit, changing overall pixels even they are actually not related to the removed object. While inpainting data guaranties strict pixel-level correspondence, improving background preservation ability.

Ablation on post-training stage. To validate the positive impact of post-training for video object injection, we compare performance between “UniVOE” and the ablation setting “w/o GRAR”, in which the injector that is only pretrained. As shown in Tab. 4, “UniVOE” stably outperforms “w/o GRAR” across all metrics. Due to limited space of main paper, related visual comparison is given in supplementary.

## Conclusion

We presented UniVOE, a removal-driven framework for realistic video object removal and injection. Our key observation is that the two tasks share the same object-present/objectabsent training structure, but difer in learning dificulty: removal can be easily bootstrapped from scalable video inpainting, whereas injection requires strict data pairs for object motion, contact, and interaction efects. UniVOE turns this reciprocal-but-asymmetric structure into a new training strategy. It first learns a strong efect-aware remover, then uses the remover to derive object-absent counterparts from real object-present videos, making the original videos serve as supervision for injection. To further address the perceptual ambiguity left by reconstruction loss, UniVOE introduces an online reward-based post-training stage and Group-Relative Advantage-Regression (GRAR), enabling VRAMeficient preference-aligned optimization for large video diffusion backbones. Extensive experiments show that UniVOE achieves SOTA performance on object editing, with clear advantages on challenging interaction-efect cases such as shadows, reflections, and motion-dependent visual changes.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; Ge, W.; Guo, Z.; Huang, Q.; Huang, J.; et al. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Bian, Y.; Zhang, Z.; Ju, X.; Cao, M.; Xie, L.; Shan, Y.; and Xu, Q. 2025. VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers.

Cao, S.; Chen, H.; Chen, P.; Cheng, Y.; Cui, Y.; Deng, X.; Dong, Y.; Gong, K.; Gu, T.; Gu, X.; et al. 2025. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951.

Caron, M.; Touvron, H.; Misra, I.; Jégou, H.; Mairal, J.; Bojanowski, P.; and Joulin, A. 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Chang, Y.-L.; Liu, Z. Y.; Lee, K.-Y.; and Hsu, W. 2019. Freeform video inpainting with 3d gated convolution and temporal patchgan. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Chen, X.; Huang, L.; Liu, Y.; Shen, Y.; Zhao, D.; and Zhao, H. 2024. AnyDoor: Zero-Shot Object-Level Image Customization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Fu, Y.; Zheng, Y.; Dai, Z.; and Ding, H. 2026. EfectErase: Joint Video Object Removal and Insertion for High-Quality Efect Erasing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Gao, J.; Chen, Z.; Liu, X.; Feng, J.; Si, C.; Fu, Y.; Qiao, Y.; and Liu, Z. 2025. Longvie: Multimodal-guided controllable ultra-long video generation. arXiv preprint arXiv:2508.03694.

Gu, Z.; Yan, R.; Lu, J.; Li, P.; Dou, Z.; Si, C.; Dong, Z.; Liu, Q.; Lin, C.; Liu, Z.; et al. 2025. Difusion as shader: 3daware video difusion for versatile video generation control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers.

Hore, A.; and Ziou, D. 2010. Image quality metrics: PSNR vs. SSIM. In Proceedings of the International Conference on Pattern Recognition.

Hu, E. J.; yelong shen; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Jiang, Z.; Han, Z.; Mao, C.; Zhang, J.; Pan, Y.; and Liu, Y. 2025. VACE: All-in-One Video Creation and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Ku, M.; Wei, C.; Ren, W.; Yang, H.; and Chen, W. 2024. AnyV2V: A Tuning-Free Framework for Any Video-to-Video Editing Tasks. Transactions on Machine Learning Research.

Kushwaha, S. S.; Nag, S.; Tian, Y.; and Kulkarni, K. 2026. Object-WIPER: Training-Free Object and Associated Efect Removal in Videos. arXiv preprint arXiv:2601.06391.

Li, X.; Xue, H.; Ren, P.; and Bo, L. 2025a. DifuEraser: A Difusion Model for Video Inpainting. arXiv preprint arXiv:2501.10018.

Li, Y.; Li, X.; Zhang, Z.; Bian, Y.; Liu, G.; Li, X.; Xu, J.; Hu, W.; Liu, Y.; Li, L.; Cai, J.; Zou, Y.; He, Y.; and Shan, Y. 2025b. IC-Custom: Diverse Image Customization via In-Context Learning. arXiv preprint arXiv:2507.01926.

Li, Z.; Lu, C.-Z.; Qin, J.; Guo, C.-L.; and Cheng, M.-M. 2022. Towards An End-to-End Framework for Flow-Guided Video Inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In Proceedings of the International Conference on Learning Representations.

Liu, J.; Liu, G.; Liang, J.; Li, Y.; Liu, J.; Wang, X.; Wan, P.; Zhang, D.; and Ouyang, W. 2026. Flow-grpo: Training flow matching models via online rl. In Proceedings of the Advances in neural information processing systems.

Liu, R.; Deng, H.; Huang, Y.; Shi, X.; Lu, L.; Sun, W.; Wang, X.; Dai, J.; and Li, H. 2021. FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Lu, S.; Liu, Y.; and Kong, A. W.-K. 2023. TF-ICON: Difusion-Based Training-Free Cross-Domain Image Composition. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Miao, C.; Feng, Y.; Zeng, J.; Gao, Z.; Liu, H.; Yan, Y.; Qi, D.; Chen, X.; Wang, B.; and Zhao, H. 2026. ROSE: Remove Objects with Side Efects in Videos. In Proceedings ofthe Thirty-ninthAnnual Conference on Neural Information Processing Systems.

Mou, C.; Cao, M.; Wang, X.; Zhang, Z.; Shan, Y.; and Zhang, J. 2024. Revideo: Remake a video with motion and content control. In Proceedings ofthe Advances in Neural Information Processing Systems.

Pan, P.; Zhao, J.; Lin, Y.; Lin, C.; Li, C.; Liu, H.; Shen, T.; and Mu, Y. 2025. ID-Crafter: VLM-Grounded Online RL for Compositional Multi-Subject Video Generation. arXiv preprint arXiv:2511.00511.

Prabhudesai, M.; Goyal, A.; Pathak, D.; and Fragkiadaki, K. 2024. Aligning Text-to-Image Difusion Models with Reward Backpropagation. In Proceedings of the International Conference on Learning Representations.

Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; and Finn, C. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Proceedings of the Advances in Neural Information Processing Systems.

Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.; Khedr, H.; Rädle, R.; Rolland, C.; Gustafson, L.; et al. 2025. SAM 2: Segment anything in images and videos. In International Conference on Learning Representations.

Seedance, T.; Chen, D.; Chen, L.; Chen, X.; Chen, Y.; Chen,Z.; Chen, Z.; Cheng, F.; Cheng, T.; Cheng, Y.; Chi, M.; Chi,

X.; Cong, J.; Cui, Q.; Ding, F.; Dong, Q.; Du, Y.; et al. 2026. Seedance 2.0: Advancing Video Generation for World Complexity. arXiv preprint arXiv:2604.14148.

Song, Y.; Zhang, Z.; Lin, Z.; Cohen, S.; Price, B.; Zhang, J.; Kim, S. Y.; and Aliaga, D. 2023. ObjectStitch: Object compositing with difusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18310–18319.

Team, B.; Liu, C.; Chen, J.; Li, L.; Chi, L.; Sun, M.; Li, Z.; Fu, Y.; Guo, R.; Wu, Y.; Bai, G.; and Yuan, Z. 2026. Bernini: Latent Semantic Planning for Video Difusion. arXiv preprint arXiv:2605.22344.

Team, K.; Chen, J.; Ci, Y.; Du, X.; Feng, Z.; Gai, K.; Guo, S.; Han, F.; He, J.; He, K.; et al. 2025. Kling-Omni Technical Report. arXiv preprint arXiv:2512.16776.

Tu, Y.; Luo, H.; Chen, X.; Ji, S.; Bai, X.; and Zhao, H. 2025. VideoAnydoor: High-Fidelity Video Object Insertion with Precise Motion Control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers.

Unterthiner, T.; Van Steenkiste, S.; Kurach, K.; Marinier, R.; Michalski, M.; and Gelly, S. 2019. FVD: A new metric for video generation. In Proceedings of ICLR 2019 Workshop.

Wallace, B.; Dang, M.; Rafailov, R.; Zhou, L.; Lou, A.; Purushwalkam, S.; Ermon, S.; Xiong, C.; Joty, S.; and Naik, N. 2024. Difusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.- W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; Zeng, J.; Wang, J.; Zhang, J.; Zhou, J.; et al. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing.

Wei, C.; Liu, Q.; Ye, Z.; Wang, Q.; Wang, X.; Wan, P.; Gai, K.; and Chen, W. 2026. UniVideo: Unified Understanding, Generation, and Editing for Videos. In Proceedings of the International Conference on Learning Representations.

Wei, Y.; Zhang, S.; Qing, Z.; Yuan, H.; Liu, Z.; Liu, Y.; Zhang, Y.; Zhou, J.; and Shan, H. 2024. DreamVideo: Composing your dream videos with customized subject and motion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Xu, J.; Liu, X.; Wu, Y.; Tong, Y.; Li, Q.; Ding, M.; Tang, J.; and Dong, Y. 2023. ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. In Proceedings of the Advances in Neural Information Processing Systems.

Yang, B.; Gu, S.; Zhang, B.; Zhang, T.; Chen, X.; Sun, X.; Chen, D.; and Wen, F. 2023. Paint by Example: Exemplar-Based Image Editing With Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yang, S.; Cun, X.; Li, X.; Li, Y.; and Zhang, J. 2026a. 4DVD: cascaded dense-view video difusion model for high-quality 4D content generation. International Journal of Computer Vision.

Yang, S.; Li, X.; Cun, X.; Wang, G.; Li, L.; Shan, Y.; and Zhang, J. 2026b. GenCompositor: Generative Video Compositing with Difusion Transformer. In Proceedings of the International Conference on Learning Representations.

Yang, S.; Mou, C.; Yu, J.; Wang, Y.; Meng, X.; and Zhang, J. 2026c. Neural video field editing. Computational Visual Media.

Ye, Z.; He, X.; Liu, Q.; Wang, Q.; Wang, X.; Wan, P.; Zhang, D.; Gai, K.; Chen, Q.; and Luo, W. 2025. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216.

Zeng, Y.; Fu, J.; and Chao, H. 2020. Learning Joint Spatial-Temporal Transformations for Video Inpainting. In Proceedings ofthe European Conference on Computer Vision.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image difusion models. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Zheng, K.; et al. 2025. DifusionNFT: Online Difusion Reinforcement with the Forward Process. arXiv preprint arXiv:2509.16117.

Zhou, S.; Li, C.; Chan, K. C. K.; and Loy, C. C. 2023. ProPainter: Improving Propagation and Transformer for Video Inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Zi, B.; Peng, W.; Qi, X.; Wang, J.; Zhao, S.; Xiao, R.; and Wong, K.-F. 2026. Minimax-remover: Taming bad noise helps video object removal. In Proceedings of the Advances in Neural Information Processing Systems.

## A Video Filtering Gate for Removal Counterparts

As described in the main paper, our injector is pre-trained on removal-derived pairs: we run the efect-aware remover $\pi _ { \mathrm { r e m } }$ on real object-present videos $V _ { \mathrm { o b j } }$ to obtain object-absent counterparts $V _ { \mathrm { a b s } } ,$ and reverse the direction so that $V _ { \mathrm { o b j } }$ becomes a photorealistic injection target. This bootstrap is only as clean as the remover output. A pair in which the remover left a residual object, a hole, or a ghosting artifact, or in which it corrupted the background outside the object region, is worse than useless as injection supervision: reversing such a pair teaches the injector to hallucinate incomplete objects or to repaint background that should stay fixed. We therefore pass every removal-derived pair through an automatic filtering gate before it enters the injection training set $\mathcal { D } _ { \mathrm { i n j } }$ and apply the same gate to open-source object-present/absent pairs (Fu et al. 2026). This section details that gate.

Object region. The gate is region-aware: it distinguishes the object region (where the object must disappear) from the background (which must be preserved). For each clip we localize the target with GroundingDINO on the first frame, propagate the box through SAM2 to obtain a per-frame instance mask, and take the union over all target objects into a single binary mask. The mask is softly dilated (Gaussian blur followed by a low threshold) so that the “inside” region covers the object together with a margin for its immediate interaction efects, and the complement defines the background region used for the fidelity check below.

Two acceptance criteria. A pair is accepted only if it passes both of the following, mirroring the two failure modes of removal:

• Background fidelity (outside the dilated mask). The remover must not over-edit the scene. We measure the residual change between $V _ { \mathrm { a b s } }$ and $V _ { \mathrm { o b j } }$ restricted to the background region; a pair is rejected when this residual is large, indicating that the remover altered content unrelated to the target (e.g., a global color shift or repainted structure).

• Removal completeness (inside the mask). The target must be genuinely gone, not merely faded, partially erased, displaced, or replaced by a look-alike. We assess the object region by visual-semantic comparison between $V _ { \mathrm { a b s } }$ and $V _ { \mathrm { o b j } } ;$ any surviving trace of the target causes rejection.

VLM-judge implementation. Because “a surviving trace” and “an unnatural background” are perceptual rather than pixel-exact, both criteria are adjudicated by a visionlanguage judge (Qwen3-VL) run at temperature 0 for determinism. The judge is prompted to (i) re-caption both videos and enumerate every object present in $V _ { \mathrm { o b j } }$ but absent in $V _ { \mathrm { a b s } } ,$ , not only the one named in the removal instruction, since the remover sometimes cleanly deletes co-occurring objects, (ii) list the actual changes, (iii) run an explicit residual check for any leftover trace of the target, and (iv) emit a binary decision with a short reason. The instruction that produced the clip is provided only as the intended edit; the judge is told to compare the two videos independently rather than trust the instruction, so that imperfect removals are not waved through. The decision rule is strict-by-default: when the judge is uncertain whether the object is fully gone, it rejects.

Focused residual gate on the object region. Full-frame video judgment tends to overlook small residuals that appear only in the middle or late frames. To harden the completeness check, we add a second, focused pass that operates only on the object region: we crop the dilated mask bounding box from $V _ { \mathrm { o b j } }$ and $V _ { \mathrm { a b s } } .$ tile eight uniformly sampled frames of each into a static grid, and ask the judge to inspect the result grid cell by cell. If any cell still shows the target — complete, partial, a faded outline, or a similar substitute — the pair is rejected. Static grids expose per-frame detail that a played-back video dilutes, making this pass stricter than the video-level judgment; a pair must clear both to be retained.

Deployment and yield. The gate is embedded in the multinode removal pipeline: each node runs the remover and the judge over a shard of clips, with many judge queries issued concurrently, so filtering keeps pace with generation across the cluster. After applying the gate, we retain 69K high-quality removal-derived pairs, which are combined with 68K gate-filtered open-source pairs to form the injection pretraining set ${ \mathcal { D } } _ { \mathrm { i n j } }$

## B Pseudo Algorithm for GRAR

Algorithm 1 details the GRAR online post-training loop. Each iteration first rolls out K candidates per context from the frozen rollout policy $\pi _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { i n j } }$ and scores them with the human-aligned reward model $\mathsf { \bar { Q } } _ { \theta } . \mathrm { ~ A ~ } \{ + 1 , - 1 \}$ signed advantage is then assigned by comparing each score against the group median $\tilde { r } _ { c } ,$ which keeps the update bounded and insensitive to the absolute reward scale. Finally, each rollout is re-used as an online flow-matching sample and the single trainable policy $\pi _ { \theta } ^ { \mathrm { i n j } }$ is regressed toward $( A { = } { + } 1 )$ or away from $( A { = } { - } 1 )$ its own velocity target. No reference policy is stored or forwarded, so the only network instantiated during optimization is $\pi _ { \theta } ^ { \mathrm { i n j } }$ itself, making full-sequence (97-frame, 480p) post-training of the 23B injector feasible within an 80 GB VRAM budget.

## C Compare to DifusionNFT

GRAR shares the negative-aware spirit of Difusion-NFT (Zheng et al. 2025), an online post-training method for flow-matching models, but removes its frozen reference policy. In this section we make the relation precise. We first restate the GRAR objective in a self-contained form, then show that (i) the GRAR gradient is first-order consistent with the reward policy gradient (Proposition 2), and (ii) the GRAR update coincides with the negative-anchored DifusionNFT update up to a first-order residual in the step size (Lemma 1), while instantiating only a single trainable network. We close with the memory argument and the 1.3B empirical comparison.

Algorithm 1: Group-Relative Advantage Regression   
(GRAR) — Stage 3 online post-training   
Require: Stage-2 injector $\pi _ { \theta } ^ { \mathrm { i n j } }$ ; reward model $Q _ { \theta } ;$ context   
pool $\mathcal { C } = \mathrm { \bar { \{ } } ( \it { V _ { \mathrm { a b s } } } , \it { c _ { \mathrm { i n j } } } ) \it { \Sigma \} } ;$ ; group size $K { = } 1 6 ;$ contexts per   
iter $B { = } 8 ;$ VAE encoder E; learning rate η; iterations $N$   
1: $\theta _ { \mathrm { o l d } }  \theta$ {rollout policy initialized from current policy}   
2: for $n = 1$ to N do   
3: sample B contexts $\{ ( V _ { \mathrm { a b s } } ^ { ( i ) } , c _ { \mathrm { i n j } } ^ { ( i ) } ) \} _ { i = 1 } ^ { B } \sim \mathcal { C } ; ~ \mathcal { D }  \emptyset$   
4: // (a) Rollout $\pmb { \& }$ reward scoring (no gradient)   
5: for each context $i = 1$ to B do   
6: for $j = 1$ to $K$ do   
7: $\bar { V _ { \mathrm { g e n } } ^ { ( i , j ) } } \sim \pi _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { i n j } } ( \cdot \mid V _ { \mathrm { a b s } } ^ { ( i ) } , c _ { \mathrm { i n j } } ^ { ( i ) } )$ {multi-step sam  
pler, frozen $ { \bar { \theta } } _ { \mathrm { o l d } } \}$   
8: $r ^ { ( i , j ) }  Q _ { \theta } ( V _ { \mathrm { g e n } } ^ { ( i , j ) } )$   
9: end for   
10: // (b) Group-relative signed advantage Eq. (10)   
11: $\tilde { r } _ { c } ^ { ( i ) } \gets \mathrm { m e d i a n } \{ r ^ { ( i , 1 ) } , \dots , r ^ { ( i , K ) } \}$   
12: for $j = 1$ to K do   
13: $\Breve { A } ^ { ( i , j ) }  + 1 \mathbf { i f } \ r ^ { ( i , j ) } \geq \tilde { r } _ { c } ^ { ( i ) }$ else $- 1$   
14: $x _ { 0 } ^ { ( i , j ) }  \mathcal { E } ( V _ { \mathrm { g e n } } ^ { ( i , j ) } )$ {treat the rollout as the online   
sample}   
15: $\mathcal { D }  \mathcal { D } \cup \{ ( x _ { 0 } ^ { ( i , j ) } , c _ { \mathrm { i n j } } ^ { ( i ) } , A ^ { ( i , j ) } ) \}$   
16: end for   
17: end for   
18: // (c) Single-policy advantage regression Eq. (11)   
19: for minibatch $( x _ { 0 } , \dot { c } _ { \mathrm { i n j } } , A ) \subset \overline { { \mathcal { D } } }$ do   
20: $t \sim p ( t ) , \quad \dot { \epsilon } \sim \mathcal { N } ( \dot { 0 } , \mathbf { I } )$   
21: $x _ { t }  ( 1 - t ) x _ { 0 } + t \epsilon , \quad v ^ { \star }  \epsilon - x _ { 0 }$ {flow-matching   
target, Eq. (9)}   
22: $\mathcal { L } _ { \mathrm { G R A R } } \dot { \gets } A \lVert v _ { \theta } ( x _ { t } , t \mid c _ { \mathrm { i n j } } ) - v ^ { \star } \rVert _ { 2 } ^ { 2 }$   
23: $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { G R A R } }$ {only $\pi _ { \theta } ^ { \mathrm { i n j } }$ is forwarded; no   
reference policy}   
24: end for   
25: $\theta _ { \mathrm { o l d } }  \theta$ {synchronize rollout policy}   
26: end for   
27: return $\pi _ { \theta } ^ { \mathrm { i n j } }$ {final UniVOE injector}

Setup and notation. Let $\pi _ { \theta } ^ { \mathrm { i n j } } ( \cdot ~ | ~ V _ { \mathrm { a b s } } , c _ { \mathrm { i n j } } )$ be the injector and $Q _ { \theta }$ the reward model, and write $\begin{array} { r l } { J ( \theta ) } & { { } = } \end{array}$ $\mathbf { \tilde { E } } _ { V _ { \mathrm { g e n } } \sim \pi _ { \theta } ^ { \mathrm { i n j } } } [ Q _ { \theta } ( V _ { \mathrm { g e n } } ) ]$ for the expected reward. For a rollout latent $x _ { 0 } ,$ noise $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ , and $t \sim \mathcal { U } [ 0 , 1 ]$ , the rectifiedflow interpolant and target are

$$
x _ { t } = \left( 1 - t \right) x _ { 0 } + t \epsilon , \qquad v ^ { \star } = \epsilon - x _ { 0 } .\tag{9}
$$

For each context we draw K rollouts $\{ x _ { 0 } ^ { ( j ) } \}$ from the frozen policy $\pi _ { \theta _ { \mathrm { o l d } } } ^ { \mathrm { i n j } }$ , score them $r ^ { ( j ) } = Q _ { \theta } ( x _ { 0 } ^ { ( j ) } )$ , and assign the median-split signed advantage

$$
A ^ { ( j ) } = \left\{ \begin{array} { l l } { + 1 , } & { r ^ { ( j ) } \geq \tilde { r } _ { c } , } \\ { - 1 , } & { r ^ { ( j ) } < \tilde { r } _ { c } , } \end{array} \right. \quad \tilde { r } _ { c } = \mathrm { m e d i a n } \{ r ^ { ( j ) } \} .\tag{10}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G R A R } } ( \theta ) = \mathbb { E } _ { c _ { \mathrm { i n j } } , j , t } \big [ A ^ { ( j ) } \| v _ { \theta } ( x _ { t } ^ { ( j ) } , t \mid c _ { \mathrm { i n j } } ) - v ^ { \star , ( j ) } \| _ { 2 } ^ { 2 } \big ] . } \end{array}\tag{11}
$$

We write $\eta$ for the SGD step size, and $\theta _ { \mathrm { o l d } }$ for the rollout parameters, which are synchronized to θ after every update, so that $\lVert \theta - \theta _ { \mathrm { o l d } } \rVert = O \dot { ( } \eta )$ within an iteration.

Bounded, zero-mean advantage. Proposition 1 (bounded, zero-mean advantage). The signed advantage in Eq. (10) satisfies $| A ^ { ( j ) } | = 1 ;$ and if the group rewards are i.i.d. from an absolutely continuous distribution, then $\mathbb { E } [ A ^ { ( j ) } ] = 0 .$   
$P r o o f . | A ^ { ( j ) } | = 1$ is immediate from $\operatorname { E q . }$ . (10). For continuous rewards, ties occur with probability zero, so exactly half of the group lies above the median; the numbers of +1 and −1 advantages are therefore equal and $\begin{array} { r } { \mathbb { E } [ \sum _ { j } A ^ { ( j ) } ] = 0 . } \end{array}$ , whence $\mathbb { E } [ A ^ { ( j ) } ] = 0$ □ Boundedness keeps the update insensitive to the numerical scale of $Q _ { \theta } ,$ , and zero mean makes $A ^ { ( j ) }$ a valid (baselinesubtracted) advantage weight. GRAR as a policy-gradient surrogate. The flowmatching evidence lower bound (Lipman et al. 2023) gives log $\pi _ { \boldsymbol { \theta } } ( x _ { 0 } \mid c ) \geq - { \mathcal { L } } _ { \mathrm { F M } } ( x _ { 0 } , { \boldsymbol { \theta } } ) { \hat { \big ) } } + C ( x _ { 0 } , c )$ , where $\bar { \mathcal { L } } _ { \mathrm { F M } } ( x _ { 0 } , \theta ) \dot { = } \mathbb { E } _ { t , \epsilon } \| v _ { \theta } - v ^ { \star } \| _ { 2 } ^ { 2 }$ and $C$ is independent of $\theta ;$ hence $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } \approx - \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { F M } } .$   
Proposition 2 (surrogate policy gradient). Assume $\theta \mapsto v _ { \theta }$ is continuously diferentiable and the reward is bounded. Then

$\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { G R A R } } ( \theta ) \big | _ { \theta = \theta _ { \mathrm { o l d } } } = - \nabla _ { \theta } J ( \theta ) \big | _ { \theta = \theta _ { \mathrm { o l d } } } + \mathcal { R } , \quad \| \mathcal { R } \| = O ( \| \theta - \theta _ { \mathrm { o l d } } \| ) . } \end{array}$ (12) Consequently, a descent step on $\mathcal { L } _ { \mathrm { G R A R } }$ is, to first order, an ascent step on the expected reward $J .$ Proof. The advantage policy gradient is $\begin{array} { r l } { \nabla _ { \theta } J } & { { } = } \end{array}$ $\mathbb { E } _ { { x _ { 0 } } \sim \pi _ { \theta } } [ A \nabla _ { \theta } \log \pi _ { \theta } ]$ , valid because the latent support (fulldimensional Gaussian noise) does not depend on θ and any c-only baseline may be subtracted from the zero-mean score. Substituting $\nabla _ { \theta }$ log $\pi _ { \theta } = - \nabla _ { \theta } \mathcal { L } _ { \mathrm { F M } } + \nabla _ { \theta } g ,$ with $g \geq 0$ the ELBO gap, yields $\nabla _ { \boldsymbol { \theta } } J = - \mathbb { E } _ { \boldsymbol { x } _ { 0 } \sim \pi _ { \boldsymbol { \theta } } } [ A \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { F M } } ] + \mathbb { E } [ A \nabla _ { \boldsymbol { \theta } } g ]$ Fixing the sampling distribution to $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ,$ the first term equals $- \nabla _ { \theta } \mathcal { L } _ { \mathrm { G R A R } } ;$ replacing $\pi _ { \theta }$ by $\pi _ { \theta _ { \mathrm { o l d } } }$ incurs an of-policy bias of order $\lVert \boldsymbol { \theta } - \boldsymbol { \theta } _ { \mathrm { o l d } } \rVert$ through the importance ratio, and $\mathbb { E } [ A \nabla _ { \theta } g ]$ is a variational residual that is θ-independent to leading order. Collecting both into R gives Eq. (12). □

First-order relation to DifusionNFT. DifusionNFT anchors the update to a frozen reference $v _ { \theta _ { \mathrm { o l d } } } \colon$ its positive branch regresses $v _ { \theta }$ toward $v ^ { \star }$ (as GRAR does for $A ^ { ( j ) } =$ +1), whereas its negative branch regresses toward the anchored target (Zheng et al. 2025)

$$
v _ { \mathrm { n e g } } ^ { \star } : = v _ { \theta _ { \mathrm { o l d } } } + \lambda ( v _ { \theta _ { \mathrm { o l d } } } - v ^ { \star } ) , \qquad \lambda > 0 ,\tag{13}
$$

which requires an extra forward pass through $v _ { \theta _ { \mathrm { o l d } } }$ at every gradient step (Zheng et al. (2025) use $\lambda = 1 )$ Lemma 1 (first-order equivalence). Let $v _ { \theta }$ be $L _ { v ^ { - 1 } }$ Lipschitz in θ with $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } _ { \mathrm { o l d } } \rVert \leq \bar { \eta L } _ { v } .$ Then

$$
\lVert \boldsymbol { v } _ { \boldsymbol { \theta } } - \boldsymbol { v } _ { \mathrm { n e g } } ^ { \star } \rVert _ { 2 } ^ { 2 } = - \lambda \lVert \boldsymbol { v } _ { \boldsymbol { \theta } } - \boldsymbol { v } ^ { \star } \rVert _ { 2 } ^ { 2 } + B ( \theta _ { \mathrm { o l d } } ) + O ( \eta ) ,\tag{14}
$$

Table 5: Post-training ablation for video object injection. Evaluate on VOR.
<table><tr><td>Settings</td><td>DINO-I↑</td><td>CLIP-P ↑</td><td>LPIPS↓</td><td>FVD↓</td></tr><tr><td>w/ DiffusionNFT</td><td>0.388</td><td>0.229</td><td>0.289</td><td>314.101</td></tr><tr><td>w/ GRAR</td><td>0.387</td><td>0.231</td><td>0.287</td><td>283.812</td></tr></table>

where $B ( \theta _ { \mathrm { o l d } } )$ does not depend on θ. Hence $\nabla _ { \theta } \lVert v _ { \theta } -$ $v _ { \mathrm { n e g } } ^ { \star } \| _ { 2 } ^ { 2 } = - \lambda \nabla _ { \theta } \| v _ { \theta } - v ^ { \star } \| _ { 2 } ^ { 2 } + O ( \eta )$   
Proof. Let $\Delta : = v _ { \theta } - v _ { \theta _ { \mathrm { o l d } } } , \mathrm { s o } \left\| \Delta \right\| \leq L _ { v } \left\| \theta - \theta _ { \mathrm { o l d } } \right\| = O ( \eta )$ and u $\iota : = v \theta _ { \mathrm { o l d } } - v ^ { \star }$ , which is independent of θ. From Eq. (13), $v _ { \theta } - v _ { \mathrm { n e g } } ^ { \star } = \Delta - \lambda u$ , so

$$
\lVert \boldsymbol { v } _ { \theta } - \boldsymbol { v } _ { \mathrm { n e g } } ^ { \star } \rVert _ { 2 } ^ { 2 } = \lVert \Delta \rVert ^ { 2 } - 2 \lambda \langle \Delta , u \rangle + \lambda ^ { 2 } \lVert u \rVert ^ { 2 } .\tag{15}
$$

Expanding $\| v _ { \theta } - v ^ { \star } \| _ { 2 } ^ { 2 } = \| u + \Delta \| ^ { 2 } = \| u \| ^ { 2 } + 2 \langle \Delta , u \rangle + \| \Delta \| ^ { 2 }$ gives $2 \langle \bar { \Delta , u } \rangle = \| \bar { v _ { \theta } } - \bar { v ^ { \star } } \| ^ { 2 } - \| u \| ^ { 2 } - \| \Delta \| ^ { 2 }$ . Substituting into Eq. (15),

$$
| | v _ { \theta } - v _ { \mathrm { n e g } } ^ { \star } | | _ { 2 } ^ { 2 } = - \lambda | | v _ { \theta } - v ^ { \star } | | _ { 2 } ^ { 2 } + ( \lambda + \lambda ^ { 2 } ) | | u | | ^ { 2 } + ( 1 + \lambda ) | | \Delta | | ^ { 2 } .
$$

The middle term depends only on $\theta _ { \mathrm { o l d } }$ (set it to $B ( \theta _ { \mathrm { o l d } } ) )$ ; the last is $O ( \eta ^ { 2 } )$ . Diferentiating in θ removes B and leaves an ${ \cal { O } } ( \eta )$ gradient residual. □

Because the positive branch is identical in the two losses and the negative branch agrees up to Eq. (14), the Difusion-NFT gradient and the GRAR gradient—with $A ^ { ( j ) } = \pm 1$ selecting the branch and λ absorbed into the learning rate— coincide to first order in η. Under the per-iteration synchronization $\theta _ { \mathrm { o l d } }  \theta$ and the small step size used here, the reference lags the policy by $O ( \eta )$ , so this first-order agreement holds throughout training and the accumulated trajectory deviation stays within the mini-batch SGD noise floor. GRAR thus realizes the same negative-anchored update while dropping the frozen reference forward pass.

Memory and feasibility. The practical consequence of Lemma 1 is that GRAR removes the second network. DifusionNFT must keep v (trainable, with optimizer states and activations) co-resident with a frozen $v _ { \theta _ { \mathrm { o l d } } }$ evaluated by an additional forward pass, roughly doubling the resident model memory. At our 23B injector scale with full-sequence 97- frame 480p inputs, this exceeds the 80 GB budget, whereas GRAR instantiates only $\pi _ { \theta } ^ { \mathrm { i n j } }$ and remains feasible. This is why prior online post-training is largely restricted to image models or small (∼1.3B) video backbones (Pan et al. 2025).

Empirical comparison at 1.3B. Since strict DifusionNFT is infeasible at 23B, we verify the first-order equivalence at a scale where both fit in memory: a 1.3B injector pre-trained on the same paired data and then post-trained with GRAR and with strict DifusionNFT under matched rollouts, reward model, and schedule. The two obtain closely matching quantitative scores (Tab. 5, lower block) and closely overlapping reward curves (Fig. 8), with qualitatively indistinguishable injections (Fig. 7), confirming that the single-policy GRAR update tracks the DifusionNFT trajectory in practice. At the full 23B scale, GRAR is the only feasible option under the 80 GB budget, and we therefore adopt it as the default posttraining objective for the UniVOE injector.

## D Ablation on Post-training Stage

To validate the positive impact of post-training for video object injection, we compare performance between “UniVOE” and the ablation setting “w/o RL”, the injector that is only pre-trained. In the main paper, we have provided quantitative results and mentioned that visual comparison is given in supplementary, here we provide qualitative results. We showcase visual comparison in Fig. 9, “w/o RL” shows common issues exhibiting in video generation models, such as color mutation and unrealistic distortion, etc. In most common scenes, these perceptually implausible cases can be well resolved by post-training as shown in Fig. 9. This paper proves the crucial efectiveness of reward-based post-training for physical-realism of video object injection.

## E More Results

In Fig. 10, we additionally provide some results of Uni-VOE, including object removal and injection. This further showcase superior performance and impressive generalization ability of UniVOE. The corresponding video demos can be found in our media supplement.

## F Limitations and Future Work

Dependence on the removal-derived bootstrap. The injector inherits whatever biases survive the filtering gate. The gate is strict-by-default and removes pairs with residual objects or corrupted background, but it cannot manufacture supervision that the remover never produces. Consequently, injection quality is upper-bounded by the coverage of the remover and the retained pairs. Designing more eficient posttraining algorithms for injectors is an efective way to break through the performance ceiling imposed by the training set. This has been proven in our paper. In addition, jointly improving the remover and injector, or recycling high-reward injection rollouts back into the training pool, is also a natural way to break this dependence. We leave this cycle-like coordinated training strategy as a future work.

Reward-bounded post-training. GRAR aligns the injector with the preferences encoded in $Q _ { \theta } ,$ , so its ceiling is the fidelity of that reward model. Because the signed advantage in Eq. (10) depends only on the within-group ranking rather than absolute scores, GRAR is robust to the reward scale and to occasional mis-scored rollouts. However, developing more diverse and preference-aligned reward model will further improve the quality of reward-guided post-training. Our team will continue to develop reward models for videos and apply and open-source them in future work.

Scope. Finally, our study targets object removal and injection on a single 23B video backbone at 97-frame 480p. The removal-driven recipe and the GRAR post-training loss are not specific to injection, and applying them to these broader settings is left to future work.

On the lasting value of specialist editing models. One might argue that, as video foundation models keep scaling, future backbones will natively support universal human and object removal and injection, which would seem to diminish the value of dedicated specialist models such as ours.

![](images/8bab80b688f800e0588ca8189ff742db4d7cb023732312f97c6d0328a491eadc.jpg)

![](images/5cf0494b548c94359d930381b60063360b623d82ddec29e3619112cb620c3d41.jpg)  
Figure 7: Visual comparison of models trained by GRAR and DifusionNFT. One can see that their outputs are visually consistent, which is consistent with the quantitative results in Tab. 5. This further demonstrates that the performance of the proposed GRAR approximates that of DifusionNFT.

![](images/0bd6a36ae47c1e1a14a0ed66240ac265efd29ae76390991a2d15c09fdb1795d4.jpg)  
Figure 8: Reward curves of GRAR and DifusionNFT on the smaller injectors with Wan2.1-1.3B backbone.

We take the opposite view, for two reasons that our results directly support. First, scale alone does not close the gap on vertical editing tasks: despite orders-of-magnitude fewer parameters, our 23B specialist outperforms much larger general-purpose systems in our comparisons (e.g., the 200Bparameter Seedance 2.0), indicating that task-specific data curation and post-training remain decisive for editing quality that generic base models do not yet deliver. Second, and more fundamentally, training a foundation model to edit natively requires millions of paired before/after editing supervision clips, which is infeasible to obtain by manual annotation alone. Industrial pipelines therefore rely on dedicated data operators that synthesize such pairs automatically. UniVOE is precisely such an operator: coupled with the strict quality gate, it can emit large volumes of high-quality objectpresent/object-absent and injection pairs, which are exactly the supervision a foundation model needs during its SFT stage. In this sense, specialist editing models are not superseded by stronger foundation models; they become the engines that supply the paired data on which those foundation models are trained. This reframes the contribution of UniVOE from a standalone editor to a reusable component of the broader video-model data ecosystem.

![](images/8893b8cdb0fc543ce02f0fa78d5bfc2bec3252bb16ee8aedd91039d955d0874c.jpg)  
Figure 9: Ablation on the post-training stage of injector. We showcase 4 cases from input video and the edited results of “w/o RL” and “UniVOE”, respectively. Related prompts are given below. Results of “w/o RL” exhibit color mutation (top-left), subject disappear (top-right), subject distortion (bottom-left), and semantic inconsistency (bottom-right). These issues can be well addressed in common scenarios by post-training.

![](images/7fcb94680e9ea355595963af0432be46ff81c7f5b89daa02c8a600aa736b6eec.jpg)  
Input Video

![](images/0aa05e5c4cb33f442d3863ba1dd2b99e96373200dc09fb9bd97555e52074b1e9.jpg)  
Edited Video

Figure 10: More results of video object removal and injection of UniVOE. For removal, UniVOE can fill realistic background content for the removed region. For injection, UniVOE not only inserts desired objects, but also predicts its natural interaction efects with background. Source videos can be found in supplementary materials.