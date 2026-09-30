# Multi-Granularity Language-Guided Imitation Learning via Instruction Decomposition

Yi-Pei Chiu and Wei-Ta Chu

Abstract— Using language instructions as conditions to guide robot policy learning has recently become an important research domain. However, existing language-guided policy learning methods typically use an overall task description to guide the entire demonstration trajectory. For manipulation tasks involving multiple execution stages, these methods assign the same language description to different subtasks, making it difficult to distinguish the behaviors required at different stages. In this work, we propose a multi-granularity language guidance method based on instruction decomposition. The proposed method decomposes an overall task description into more finegrained, concrete subtask-level language instructions, thereby enhancing learning efficiency and improving performance. We evaluate the proposed method in the setting of multi-task imitation learning and validate its effectiveness.

## I. INTRODUCTION

A task described by natural language is often not a single atomic action, but a composition of multiple intermediate steps that must be executed in a coherent order. Humans can naturally infer such procedural structure from high-level instructions, decomposing an abstract goal into a sequence of motion primitives. However, this remains challenging for robots. They must not only ground the language instruction in the physical environment, but also determine how the specified goal can be achieved through temporally extended behaviors. Recent studies in language-guided robotic policy learning have adopted natural language as a task condition [1][2][3]. Nevertheless, most existing approaches condition the policy on a single language goal and directly learn the corresponding behavior, without explicitly modeling the intermediate steps.

As shown in Figure 1, prior language-guided imitation learning methods typically rely on a single overall instruction to condition the entire trajectory. However, such coarse language supervision often fails to distinguish the different sub-goals and action requirements involved in a multi-step task. For example, a high-level instruction such as “Pick up the object and place it into the container” does not provide explicit guidance for each intermediate stage, such as moving the gripper above the object, grasping the object, lifting it, moving it above the container, and opening the gripper to release it. As a result, the policy must learn substantially different behaviors under the same overall language condition. This makes it difficult to associate each stage of the trajectory with the appropriate action, and may lead to ambiguous or misaligned behaviors, especially in long-horizon or multistage manipulation tasks.

![](images/597e2df5d1e174c585beb128896623b18ec48c38d2b16feb8961d854353f38d1.jpg)  
Fig. 1. Comparison of language guidance strategies in imitation learning. Prior methods use a single overall instruction for the entire trajectory, while our method introduces multi-granularity language guidance by incorporating task-level instructions, fine-grained subtask-level instructions, and their combination during policy learning.

Motivated by these observations, we propose MuGIL, a Multi-Granularity language guidance framework for Imitation Learning. Instead of conditioning the policy solely on a single task-level instruction, MuGIL leverages language guidance at multiple granularities during training. Specifically, the policy is trained to capture the global task objective from the overall instruction while also learning fine-grained action guidance from subtask-level descriptions.

The rest of this paper is organized as follows. Sec. II presents related works on language-guided policy learning in several aspects. Sec. III provides details of the proposed MuGIL framework. Sec. IV describes the evaluation results and ablation studies, followed by the concluding remarks in Sec. V.

## II. RELATED WORK

In recent years, diffusion models have attracted growing attention in robotic learning. Diffusion-based methods learn to generate action sequences by iteratively denoising Gaussian-corrupted actions conditioned on observations [4][5]. Meanwhile, language has become an increasingly important representation of task goals in robotic manipulation. In imitation learning, many approaches encode natural language task descriptions into embeddings using pretrained language models, which are then provided as conditional inputs to policies trained on multi-task datasets.

Distill-Down [2] extends a single-task diffusion policy [6] to multi-task policy learning. PlayFusion [7] applies diffusion-based policy learning to unstructured and suboptimal play data, where language annotations are used to specify goal-directed skills. MDT [3] further explores diffusion policies with multimodal goals by incorporating both language and target-image goals during training. Together, these works demonstrate the effectiveness of language-guided diffusion-based policies in robotic manipulation. However, all these methods take a single overall instruction to guide the entire trajectory.

Some studies [8][9][10][11] show that decomposing complex, high-level instructions into low-level subgoals helps robots complete sophisticated tasks more reliably. PALO [8] employs vision-language models to translate high-level task descriptions into reusable subtasks, enabling rapid adaptation with minimal supervision. CLAP [9] decomposes high-level task instructions into step-wise instructions and uses them to guide language-aligned 3D keypoint prediction. These works show that decomposing high-level instructions into more low-level subgoals can improve policy adaptation and generalization.

RACER [10] augments expert demonstrations with failure recovery trajectories and fine-grained language annotations, allowing the policy to learn how to recover from execution failures. STEER [11] relabels existing robot demonstrations with dense natural language commands that describe modular manipulation skills, enabling the learned policy to be controlled not only by what task to perform but also by how the behavior should be executed. These works show that richer language can provide sufficient details beyond just simple language instructions.

Like these works, our method also provides richer language guidance to policy learning. Previous works often use decomposed or richer language for policy adaptation, 3D keypoint prediction, failure recovery, or reasoning-based generalization. On the other hand, we focus on the effect of multi-granularity language guidance in imitation learning. Specifically, we jointly incorporate task-level instructions, fine-grained subtask-level instructions, and their combination during training. Rather than treating fine-grained language as a replacement for task-level instructions, we study how these different levels of language guidance can complement each other during policy learning.

## III. METHOD

## A. Problem Formulation

In language-guided imitation learning, the objective is to train a policy that predicts robot actions from visual observations, conditioned on natural language instructions. Each individual trajectory is represented as ${ \boldsymbol { \tau } } ~ = ~ \{ ( o _ { t } , a _ { t } ) \} _ { t = 1 } ^ { T } ,$ where $o _ { t }$ denotes the observation image at timestep t, and $a _ { t }$ denotes the corresponding robot action.

Unlike conventional language-guided imitation learning settings, where a single language instruction ℓ is assigned to an entire trajectory, we extend a single language instruction into temporally aligned language instructions $\ell _ { i }$ Each timestep is associated with a corresponding language condition. Specifically, starting from the overall language annotation for each trajectory, we decompose it into multiple finer-grained language instructions. Based on this process, we construct a multi-granularity language-guided dataset $\boldsymbol { \mathcal { D } } ~ = ~ \{ ( o _ { i } , \bar { a } _ { i } , \ell _ { i } ) \} _ { i = 1 } ^ { N }$ , where N is the total number of samples collected from all trajectories, and the language condition $\ell _ { i }$ may represent descriptions at different levels, including an original task-level instruction, a finer subtasklevel instruction, or their combination. The goal is to learn a language-guided policy $\pi _ { \boldsymbol { \theta } } \left( \bar { a } _ { i } \mid o _ { i } , \ell _ { i } \right)$ that predicts a sequence of actions ${ \bar { a } } _ { i } \ = \ ( a _ { i } , a _ { i + 1 } , \ldots , a _ { i + h - 1 } )$ of length $h ,$ conditioned on the current observation $o _ { i }$ and the corresponding language instructions $\ell _ { i } .$ . Our policy is trained to maximize the log-likelihood of the action sequence given the observation and language goal:

![](images/3a72ee24d25a7ed18ad9c1b08edb8c480683edacbe3b66be3635aed9d418fec4.jpg)  
Fig. 2. Illustration of the fine-grained language annotation process. Step 1: Detect keyframes based on robot states. Step 2: Generate fine language instructions by a VLM. Step 3: Human verify to reduce annotation mess. Step 4: Assigning fine language instructions to corresponding subtask segments.

$$
\mathbb { E } \left[ \sum _ { \left( o _ { i } , \bar { a } _ { i } , \ell _ { i } \right) \in \mathcal { D } } \log \pi _ { \theta } \left( \bar { a } _ { i } \mid o _ { i } , \ell _ { i } \right) \right] .\tag{1}
$$

## B. Fine-Grained Language Annotation

Most existing language-guided datasets [12][13][14] provide a single task-level language instruction for an entire demonstration. The intermediate execution stages are not explicitly provided to guide detailed sub-goals. To provide more detailed language supervision, we decompose the original language annotation for each demonstration into finergrained subtask-level annotations. Figure 2 illustrates the process.

The LIBERO dataset [12] used in this work is collected with a 7-DoF Franka Emika Panda robot. In addition to visual observations and robot actions, the dataset also contains proprioceptive robot states, including the 7-dimensional joint state $\mathbf { q } _ { t } \in \mathbb { R } ^ { 7 }$ and the 2-dimensional gripper state ${ \bf g } _ { t } ~ \in ~ \mathbb { R } ^ { 2 }$ . To identify boundaries between subtasks, we first detect keyframes based on the robot’s state. Following prior works [15][16], a timestep is selected as a keyframe candidate if (1) the joint velocity is close to $0 ;$ and (2) the gripper open/close state remains unchanged. The intuition is that when the robot temporarily slows down and maintains a stable gripper state, it often reaches an intermediate state that may correspond to the beginning or the end of a subtask. After keyframe detection, let $\mathcal { W } = \{ w _ { 1 } , w _ { 2 } , \dots , w _ { K } \}$ denote the set of detected keyframes in a demonstration, where $w _ { j }$ denotes the timestep index of the j-th keyframe, and $K$ denotes the total number of detected keyframes in the demonstration.

![](images/8a0f2e743acb271e1525d10b2d49a1f17907aa8f98eaa8ded0fb1dc7db1770f1.jpg)  
Fig. 3. Illustration of the mixed language instruction strategy. For each timestep t in a trajectory τ, a language mode m is sampled from coarse $\ell ^ { c } ,$ fine $\ell ^ { f } ,$ and both $\ell ^ { b }$ language instructions.

Given the detected keyframes W and the original language instruction $\ell ^ { c } ,$ , we use GPT-4o [17] to generate fine language instructions based on the original language instruction, the keyframe, and a text prompt. The generated fine instruction is expected to describe a local objective of the current subtask, such as “Move the gripper towards the object” and “Close the gripper to pick the object”. The fine language instruction for the j-th keyframe is denoted as $\ell _ { j } ^ { f }$ in the following. Since automatically generated annotations may not perfect, we manually inspect and correct them. This verification step is important for reducing annotation mess and ensuring that the fine language instructions are semantically consistent with the corresponding robot behaviors.

Finally, the fine language instructions are assigned back to the timesteps of the original demonstration. For each detected keyframe $w _ { j } .$ , we define a subtask segment as the temporal interval from the previous keyframe to the current keyframe, i.e., $[ w _ { j - 1 } , w _ { j } ]$ ]. The fine instruction generated for this segment is denoted as $\ell _ { j } .$ . Then, for every timestep t within this segment, the same fine language instruction is assigned as

$$
\ell ( t ) = \ell _ { j } , \quad t \in [ w _ { j - 1 } , w _ { j } ] .\tag{2}
$$

In this way, all movements within the same subtask segment are associated with the same fine language instruction.

Through this annotation process, we obtain a multigranularity language annotated dataset, where each demonstration contains both the original task-level instruction and the finer subtask-level instructions.

## C. Mixed Language Instruction Strategy

Now each demonstration is associated with language instructions at two different levels. To train the policy, we consider three language-guided modes: coarse, fine, and both language instructions. The coarse language instruction describes the overall task objective. The fine language instruction describes the current subtask stage and provides more detailed local information. The both setting combines the coarse and fine language instructions, thereby containing both the global task objective and the local subtask context.

We introduce a mixed language instruction strategy that allows the policy to learn from multiple forms of language guidance, as shown in Figure 3. For each timestep t in a trajectory τ, we select a language instruction $\ell _ { t }$ according to one of the three language modes. The language instruction used for training is defined as

$$
\ell _ { t } = \left\{ \begin{array} { l l } { \ell _ { t } ^ { c } , } & { m _ { t } = \mathrm { c o a r s e } , } \\ { \ell _ { t } ^ { f } , } & { m _ { t } = \mathrm { f n e } , } \\ { \ell _ { t } ^ { b } , } & { m _ { t } = \mathrm { b o t h } , } \end{array} \right. \quad m _ { t } \in \{ \mathrm { c o a r s e } , \mathrm { f n e } , \mathrm { b o t h } \} .\tag{3}
$$

Since the coarse instruction describes the overall task objective, we set $\ell _ { t } ^ { c } \ = \ \ell ^ { c }$ for all timesteps in the same trajectory. The both language instruction is constructed by combining the coarse and fine language instructions: $\ell _ { t } ^ { b } =$ Concat $( \bar { \ell ^ { c } } , \ell _ { t } ^ { f } )$

During training, the language mode is sampled for each

![](images/9344300782ec77da3f3d91f6873265ef13ae6eb45e925aba867d603705b9e206.jpg)  
Fig. 4. Illustration of the proposed Multi-Granularity Language Guidance for Imitation Learning (MuGIL) method.

timestep according to a categorical distribution:

$$
m _ { t } \sim \mathrm { C a t e g o r i c a l } ( p _ { c } , p _ { f } , p _ { b } ) ,\tag{4}
$$

where $p _ { c } , p _ { f } .$ , and $p _ { b }$ denote the sampling probabilities for the coarse, fine, and both language modes, respectively. In our experiments, we empirically set the sampling ratio as $p _ { c } : p _ { f } : p _ { b } = 0 . 6 : 0 . 1 : 0 . 3$ . We further investigate the influence of different sampling ratios through an ablation study in Section IV-C.3.

Given a trajectory with T timesteps, this strategy samples one of the three language modes for each timestep t. Different timesteps within the same trajectory may be conditioned on language instructions at different levels. This design allows the policy to learn from complementary information provided by different language modes. Therefore, the proposed mixed language instruction strategy enables the policy to benefit from multi-granularity language guidance during policy learning.

## D. Language-Guided Policy Learning

1) Diffusion-Based Policy Architecture: Figure 4 illustrates the language-guided policy learning process. We adopt MDT [3] as the base architecture for language-guided imitation learning. To enable MDT to leverage partially annotated datasets [18], [19], it is designed to learn from multimodal goals, including language goals and visual goals. In our work, we focus on evaluating the effect of language guidance in policy learning. Therefore, we use language as the primary goal condition, while retaining the visual goal for auxiliary objectives introduced in MDT.

MDT uses an encoder-decoder transformer [20] architecture for diffusion-based action prediction. Given the current visual observation and the language goal, the encoder converts the conditioning inputs into a set of latent representation tokens. The decoder acts as a diffusion denoiser, which predicts the action sequence by iteratively denoising noisy actions from the encoder outputs.

For visual observations, MDT uses a ResNet-18 encoder [6] to extract observation features, which are represented as an observation token. For language conditioning, the language instruction is encoded by a frozen CLIP text encoder [21] and represented as a language token. Different from conventional language-guided policies that use a single task-level language instruction for the entire trajectory, our method adopts the mixed language instruction strategy mentioned above. After obtaining the observation and language tokens, the MDT encoder processes these tokens through several self-attention transformer layers and produces latent representations.

The MDT decoder generates the action sequence through a diffusion denoising process. During training, Gaussian noise is added to an action sequence, and the decoder is trained to denoise the noisy action sequence. In each decoder layer, cross-attention is used to incorporate the conditioning information from the encoder outputs into the denoising process. The current noise level $\sigma _ { k }$ is encoded by a sinusoidal embedding followed by an Multi-Layer Perceptron (MLP) that produce a latent noise token, which is further injected into the transformer decoder blocks.

Following diffusion-based policy learning, MDT generates actions by denoising noisy action sequences. Given the visual observation $o _ { i }$ and the language instruction $\ell _ { i } ,$ a neural network $D _ { \theta }$ is trained to approximate the score function of the diffusion process via Score Matching (SM) [22]:

$$
\mathcal { L } _ { \mathrm { S M } } = \mathbb { E } _ { \sigma , \bar { a } _ { i } , \epsilon } \left[ \alpha ( \sigma _ { k } ) \| D _ { \theta } \left( \bar { a } _ { i } + \epsilon , o _ { i } , \ell _ { i } , \sigma _ { k } \right) - \bar { a } _ { i } \| _ { 2 } ^ { 2 } \right]\tag{5}
$$

where ϵ denotes the sampled Gaussian noise [4], $\sigma _ { k }$ denotes the noise level at diffusion step $k ,$ and $\alpha ( \sigma _ { k } )$ is a weighting function that depends on the noise level. During training, noise is sampled randomly from a noise distribution and added to the ground-truth action sequence. The denoising network $D _ { \theta }$ then predicts the denoised actions and is optimized by minimizing the score matching loss.

In addition to the diffusion objective, MDT introduces two auxiliary objectives: Masked Generative Foresight (MGF) and Contrastive Latent Alignment (CLA). MGF is used to encourage the latent embedding to predict future observation information. Given the current observation $o _ { i }$ , MGF reconstructs the image patches of a future observation $o _ { i + v } ,$ where v denotes the foresight distance. Let $( { \bf u } _ { 1 } , \dots , { \bf u } _ { U } ) =$ $\mathrm { p a t c h } ( o _ { i + v } )$ denote the $U$ image patches extracted from $o _ { i + v }$ . The MGF loss is defined as

$$
\mathcal { L } _ { \mathrm { M G F } } ( o _ { i } ) = \frac { 1 } { U } \sum _ { { \bf u } \in \mathrm { p a t c h } ( o _ { i + v } ) } { { \bf 1 } } _ { \mathrm { m } } ( { \bf u } ) \left( { \bf u } - \hat { \bf u } \right) ^ { 2 } ,\tag{6}
$$

where uˆ is the reconstructed patch corresponding to u, and the indicator function ${ \bf 1 } _ { \mathrm { m } } ( { \bf u } )$ is 1 if u is masked and 0 otherwise.

CLA is an auxiliary objective to align visual and language goal representations using contrastive learning [23]. Given a training batch of size B with multimodal goals, MDT obtains two normalized embeddings, $\mathbf { z } _ { i } ^ { \mathrm { o } }$ and $\mathbf { z } _ { i } ^ { 1 } .$ , which correspond to the observation-goal conditioned state embedding and the language-goal conditioned state embedding, respectively. These embeddings are obtained by pooling the MDT latent tokens into a single vector for each goal modality. The CLA loss is computed using a symmetric InfoNCE [24] objective:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { C L A } } = - \displaystyle \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { B } \Bigg [ \log \left( \frac { \exp \left( \frac { C ( { \bf z } _ { i } ^ { 0 } , { \bf z } _ { i } ^ { 1 } ) } { v } \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \frac { C ( { \bf z } _ { i } ^ { 0 } , { \bf z } _ { j } ^ { 1 } ) } { v } \right) } \right) } & { } \\ { + \log \left( \frac { \exp \left( \frac { C ( { \bf z } _ { i } ^ { 0 } , { \bf z } _ { i } ^ { 1 } ) } { v } \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \frac { C ( { \bf z } _ { j } ^ { 0 } , { \bf z } _ { i } ^ { 1 } ) } { v } \right) } \right) \Bigg ] . } \end{array}\tag{7}
$$

with temperature parameter v. The MDT training objective is a weighted sum of the score matching loss and the auxiliary losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M D T } } = \mathcal { L } _ { \mathrm { S M } } + \alpha \mathcal { L } _ { \mathrm { M G F } } + \beta \mathcal { L } _ { \mathrm { C L A } } , } \end{array}\tag{8}
$$

where $\alpha = 0 . 1$ and $\beta = 0 . 1$ in our experiments.

2) Subtask-Aware Loss: In addition to the training objectives described above, we introduce an auxiliary objective called Subtask-Aware Loss (SAL). The purpose of SAL is to encourage the policy to learn representations that are aware of the current subtask stage. As a result, the policy should not only understand the overall task goal, but also recognize the current subtask stage from the observation and the language condition.

Let $\boldsymbol { \mathcal { B } } = \{ b ^ { 0 } , b ^ { 1 } , \ldots , b ^ { S } \}$ denote the subtask boundaries, where $b ^ { 0 }$ and $b ^ { S }$ correspond to the start and end of a trajectory, respectively. The timestep t is assigned to the s-th subtask stage if

$$
b ^ { s } \leq t < b ^ { s + 1 } , \quad s \in \{ 0 , 1 , . . . , S - 1 \} .\tag{9}
$$

The ground-truth subtask label is defined as

$$
y _ { t } = s , \quad { \mathrm { i f ~ } } b ^ { s } \leq t < b ^ { s + 1 } ,\tag{10}
$$

In our experiments, we set $S = 4$ for all demonstrations.

We formulate this objective as a subtask classification task. Given the latent representation tokens produced by the MDT encoder, we first aggregate them into a single representation vector $\mathbf { z } _ { i }$ . This representation is then passed into a subtask classification head $h _ { \mathrm { S A I } }$ to predict the current subtask stage. The predicted subtask probability is computed as

$$
{ \bf p } _ { i } = \mathrm { s o f t m a x } \left( h _ { \mathrm { S A L } } ( { \bf z } _ { i } ) \right) ,\tag{11}
$$

where $\mathbf { p } _ { i } \in \mathbb { R } ^ { S }$ denotes the predicted probability distribution over $S$ subtask classes.

We use cross-entropy loss to train the subtask classification objective:

$$
\mathcal { L } _ { \mathrm { S A L } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \mathbf { p } _ { i } ( y _ { i } ) , \quad y _ { i } \in \{ 0 , 1 , \ldots , S - 1 \} ,\tag{12}
$$

where B is the batch size, S is the number of subtask classes, and $y _ { i }$ is the ground-truth subtask label of the i-th training sample. By predicting the label of the current subtask, SAL encourages the encoder to encode stage-level information for policy learning.

The overall training objective of MuGIL is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M u G I L } } = \mathcal { L } _ { \mathrm { M D T } } + \lambda \mathcal { L } _ { \mathrm { S A L } } , } \end{array}\tag{13}
$$

where $\lambda = 0 . 2$ in our experiments.

## IV. EXPERIMENTS

## A. Experimental Settings

We evaluate our method on four task suites from the LIBERO benchmark [12]. In our experiments, we use 20 demonstrations per task for training. During evaluation, we evaluate each method with 20 rollouts per task. We report the average success rate over all tasks in each suite.

We use a single NVIDIA GeForce RTX 4080 GPU (CUDA 13.0) with 32 logical CPU threads. In our experiments, all methods are trained with the observation images from the static camera. All input images are normalized using the CLIP image normalization statistics. We apply random shift augmentation to the observation images during training and resize them to 224 × 224 pixels. For action prediction, we use the default end-effector action space in all our experiments. All models are trained for 50 epochs, and we used the checkpoint from the final training epoch. The detailed hyperparameters are provided in Table I.

## B. Performance on the LIBERO benchmark

Table II reports the success rates of different methods on the LIBERO benchmark. We compare the baseline MDT policy with our proposed MuGIL framework under two settings: using the mixed language instruction strategy only, and further incorporating the subtask-aware loss ${ \mathcal { L } } _ { \mathrm { S A L } }$

Overall, MuGIL achieves better performance than the baseline MDT. Compared with MDT, applying the mixed language instruction strategy improves the average success rate from 68.13% to 72.5%. This shows that training the policy with the language guidance at different levels can provide richer information for policy learning. In particular, the mixed language strategy improves the performance on LIBERO-Goal, LIBERO-Object, and LIBERO-Spatial by 7.0%, 8.0%, and 4.0%, respectively. The results suggest that fine-grained language guidance can help the policy better understand the current manipulation context.

TABLE I  
HYPERPARAMETERS OF POLICY TRAINING IN THIS WORK.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>epochs</td><td>50</td><td>batch size</td><td>64</td></tr><tr><td># encoder layers</td><td>4</td><td># decoder layers</td><td>6</td></tr><tr><td>attention heads</td><td>8</td><td>action chunk size</td><td>10</td></tr><tr><td>history length</td><td>1</td><td>goal window sampling size</td><td>49</td></tr><tr><td>hidden dimension</td><td>512</td><td>image encoder</td><td>ResNet18</td></tr><tr><td>attention dropout</td><td>0.3</td><td>residual dropout</td><td>0.1</td></tr><tr><td>MLP dropout</td><td>0.05</td><td>input dropout</td><td>0.0</td></tr><tr><td>optimizer</td><td>AdamW</td><td>betas</td><td>[0.9, 0.9]</td></tr><tr><td>learning rate</td><td>1e-4</td><td>weight decay</td><td>0.05</td></tr><tr><td>other weight decay</td><td>0.05</td><td>trainable parameters</td><td>78.3 M</td></tr><tr><td>σmax</td><td>80</td><td>σmin</td><td>0.001</td></tr><tr><td> $\sigma _ { t }$ </td><td>0.5</td><td>time steps</td><td>Exponential</td></tr><tr><td>sampler</td><td>DDIM</td><td>language goal encoder</td><td>CLIP ViT-B/32</td></tr></table>

However, on LIBERO-10, the success rate slightly decreases from 48.0% to 46.5% when only the mixed language instruction strategy is applied. One reason is that LIBERO-10 contains longer-horizon tasks that require completing multiple subtask stages sequentially. For such tasks, maintaining a stable and consistent task objective throughout the trajectory becomes more important. Although mixed language instructions provide more diverse language information, they may also introduce ambiguity if the policy does not explicitly recognize the current subtask stage. After incorporating ${ \mathcal { L } } _ { \mathrm { S A L } }$ the improvement is especially significant on LIBERO-10, where the success rate increases from 46.5% to 63.5%. This result suggests that ${ \mathcal { L } } _ { \mathrm { S A L } }$ helps the policy capture subtask stage information and improves its subtask awareness. By explicitly encouraging the learned representation to encode the current execution stage, the policy can better handle long horizon tasks.

Although adding ${ \mathcal { L } } _ { \mathrm { S A L } }$ slightly reduces the performance on LIBERO-Object and LIBERO-Spatial compared with using mixed language alone, it still achieves the highest average success rate across all suites. These results indicate that combining multi-granularity language guidance with subtask-aware supervision provides a more effective learning signal.

## C. Ablation Study

1) Stage-Wise Success Counts: To further analyze whether MuGIL improves task execution across different subtask stages, we evaluate the stage-wise success counts for each task suite, as shown in Figure 5. Inspired by the stagewise evaluation in RoboEval [25], we define the stage-wise success count as the number of successful rollouts at each subtask stage, rather than only measuring whether the entire task is completed. This metric provides a more detailed view of how well a policy progresses through different subtask stages.

In Figure 5, the yellow curve represents our proposed MuGIL method, while the blue curve represents the baseline

![](images/25637c451cc85024688742d87eae5e853f96ec7419bcb1c95d64873128cf6d97.jpg)  
Fig. 5. Stage-wise success counts on different LIBERO task suites.

MDT trained only with coarse language instructions. Across the four LIBERO task suites, MuGIL generally achieves higher stage-wise success counts than MDT. This indicates that the proposed multi-granularity language guidance and subtask-aware learning help the policy complete more intermediate subtask stages during execution.

The improvement is especially clear on LIBERO-Goal and LIBERO-10, which suggests that MuGIL is more effective at maintaining task progress when the policy is required to complete multiple subtask stages sequentially. For LIBERO-Object and LIBERO-Spatial, the early-stage success counts remain relatively high for both MDT and MuGIL. Nevertheless, MuGIL still maintains comparable or better performance across later stages.

These results suggest that MuGIL not only improves final task success but also enhances the policy’s ability to complete intermediate subtask stages, demonstrating better subtask-level execution capability.

2) Semantic Fine Language Guidance: To further investigate the importance of semantic information in fine-grained language guidance, we compare our proposed method under two settings: semantic fine language and non-semantic fine language. Semantic fine language refers to fine language instructions that contain detailed semantic information about the current subtask, while non-semantic fine language refers to fine language instructions that provide only simple step identifiers without detailed semantic descriptions.

In the original design, the fine language instruction contains meaningful subtask descriptions, such as “Move the gripper towards the milk” or “Close the gripper to pick the milk.” These instructions provide detailed semantic information about the current manipulation stage. For comparison, we construct a non-semantic fine language setting. In this setting, the original coarse language instruction remains unchanged, but the fine language instruction is replaced with non-semantic step identifiers, such as “Task 1. Step 1” and “Task 1. Step 2.” Similarly, for the both language mode, the subtask description is replaced with the corresponding nonsemantic step identifier. This setting preserves the temporal subtask information but removes the semantic meaning of each fine-grained instruction. Therefore, this ablation allows us to examine whether the performance improvement comes from the semantic content of fine language or merely from the additional subtask stage indicator.

TABLE II  
SUCCESS RATES ON DIFFERENT LIBERO TASK SUITES. THE BEST RESULTS ARE SHOWN IN BOLD, AND THE SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td>Method</td><td>Mixed Language</td><td> ${ \mathcal { L } } _ { \mathrm { S A L } }$ </td><td>LIBERO- Goal</td><td>LIBERO- Object</td><td>LIBERO- Spatial</td><td>LIBERO- 10</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>MDT</td><td></td><td></td><td>63.0</td><td>88.0</td><td>73.5</td><td>48.0</td><td>68.13</td></tr><tr><td rowspan="2">MuGIL</td><td>√</td><td></td><td>70.0</td><td>96.0</td><td>77.5</td><td>46.5</td><td>72.50</td></tr><tr><td>√</td><td>√</td><td>77.5</td><td>94.5</td><td>73.5</td><td>63.5</td><td>77.25</td></tr></table>

TABLE III

COMPARISON BETWEEN SEMANTIC AND NON-SEMANTIC FINE-GRAINED LANGUAGE.
<table><tr><td>Method</td><td>LIBERO- Goal</td><td>LIBERO- Object</td><td>LIBERO- Spatial</td><td>LIBERO- 10</td><td> $\mathbf { A v } \mathbf { g }$ </td></tr><tr><td>MuGIL w/ Semantic Fine</td><td>77.5</td><td>94.5</td><td>73.5</td><td>63.5</td><td>77.25</td></tr><tr><td>MuGIL w/ Non-Semantic Fine</td><td>72.5</td><td>91.0</td><td>65.0</td><td>53.5</td><td>70.50</td></tr></table>

TABLE IV

DIFFERENT LANGUAGE MODE SAMPLING RATIOS ON DIFFERENT LIBERO TASK SUITES.
<table><tr><td rowspan="2">Method</td><td colspan="3">Sampling Ratio</td><td rowspan="2">LIBERO-Goal</td><td rowspan="2">LIBERO-Object</td><td rowspan="2">LIBERO-Spatial</td><td rowspan="2">LIBERO-10</td><td rowspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>Coarse</td><td>Fine</td><td>Both</td></tr><tr><td rowspan="3">MuGIL</td><td>0.5</td><td>0.2</td><td>0.3</td><td>77.0</td><td>94.0</td><td>68.0</td><td>53.5</td><td>73.13</td></tr><tr><td>0.6</td><td>0.1</td><td>0.3</td><td>77.5</td><td>94.5</td><td>73.5</td><td>63.5</td><td>77.25</td></tr><tr><td>0.7</td><td>0.0</td><td>0.3</td><td>74.0</td><td>86.0</td><td>69.0</td><td>56.5</td><td>71.38</td></tr></table>

Table III shows the comparison between semantic and nonsemantic fine language guidance. MuGIL with semantic fine language consistently outperforms the non-semantic variant across all LIBERO task suites. The average success rate decreases from 77.25% to 70.50% when the semantic fine language is replaced with non-semantic step identifiers. This result indicates that the semantic content of fine-grained language plays an important role in policy learning.

This suggests that meaningful subtask descriptions provide useful local guidance for distinguishing different manipulation stages. Although the non-semantic setting still provides information about the index of the current subtask, it does not describe what manipulation behavior should be performed at that stage. As a result, the policy receives less informative guidance and achieves lower performance.

3) Language Mode Sampling Ratios: As described in Section III-C, one of the three language modes is sampled during training, and the sampling ratio controls how frequently each mode of language guidance is used. To analyze the influence of different language mode sampling ratios, we vary the sampling ratios of the coarse, fine, and both language modes.

Table IV shows the results under different sampling ratios. The setting with the ratio $p _ { c } : p _ { f } : p _ { b } = 0 . 6 : 0 . 1 : 0 . 3$ achieves the best overall performance. This suggests that a balanced combination of coarse, fine, and both language instructions provides the most effective supervision. When the sampling ratio of fine language instruction is removed, as in the ratio $0 . 7 : 0 . 0 : 0 . 3 .$ , the performance decreases noticeably. This indicates that explicitly training the policy with fine-only language conditions is important, as it helps the policy learn how to follow fine language instructions directly. On the other hand, reducing the sampling ratio of coarse language instruction, as in the ratio 0.5 : 0.2 : 0.3, also leads to lower performance across all task suites. This result implies that coarse language instruction remains important for providing task-level guidance, as reducing its sampling ratio weakens the supervision of the overall task objective. This makes it harder for the policy to maintain a consistent task objective during execution. Overall, the proper sampling ratio 0.6 : 0.1 : 0.3 provides a better balance between global task guidance and local subtask guidance.

## V. CONCLUSION

We propose MuGIL, a multi-granularity language guidance framework for language-guided imitation learning. Instead of relying only on a single language instruction, MuGIL decomposes it into multiple finer language instructions that describe different subtasks. By providing language guidance at different levels, the policy can receive both global task-level guidance and local subtask-level guidance during training. To effectively incorporate different forms of language guidance, we introduce a mixed language instruction strategy, which allows the policy to learn from coarse, fine, and both language instructions. In addition, we design a subtask-aware loss to encourage the policy to distinguish different subtasks, thereby enhancing its awareness of task progress.

We evaluate MuGIL on the LIBERO benchmark across multiple task suites. The experimental results show that

MuGIL improves policy performance compared with the baseline MDT. In particular, the mixed language instruction strategy provides more informative language supervision, while the proposed SAL further improves performance on more challenging long-horizon tasks. The ablation studies further show that MuGIL can maintain more successful rollouts across different subtasks, and that the semantic content of fine language instructions plays an important role in policy learning.

In the future, we can investigate how to use fine language instructions not only as training supervision but also as an explicit mechanism for subtask switching during execution. Instead of relying on predefined temporal segments, the policy can automatically determine whether the current subtask has been completed and switch to the next subtask accordingly. Such an online subtask transition mechanism may further improve the flexibility and robustness of longhorizon manipulation policies.

Acknowledgement. This work was funded in part by the National Science and Technology Council, Taiwan, under grants 115-2622-8-006-015, 114-2622-E-006-028, 114- 2221-E-006-047-MY3, 115-2425-H-006-005, 115-2218-E-006-024, and 114-2634-F-006-002.

## REFERENCES

[1] C. Lynch and P. Sermanet, “Language conditioned imitation learning over unstructured data,” in Proceedings of Robotics: Science and Systems, 2021.

[2] H. Ha, P. Florence, and S. Song, “Scaling up and distilling down: Language-guided robot skill acquisition,” in Proceedings of Conference on Robot Learning, 2023, pp. 3766–3777.

[3] M. Reuss, O. E. Ya<sup>¨</sup> gmurlu, F. Wenzel, and R. Lioutikov, “Multimodal˘ diffusion transformer: Learning versatile behavior from multimodal goals,” in Proceedings of Robotics: Science and Systems, 2024.

[4] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Proceedings of Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851.

[5] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in Proceedings of International Conference on Learning Representations, 2021.

[6] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[7] L. Chen, S. Bahl, and D. Pathak, “Playfusion: Skill acquisition via diffusion from language-annotated play,” in Proceedings of Conference on Robot Learning, 2023, pp. 2012–2029.

[8] V. Myers, B. C. Zheng, O. Mees, S. Levine, and K. Fang, “Policy adaptation via language optimization: Decomposing tasks for few-shot imitation,” in Proceedings ofConference on Robot Learning, 2024, pp. 1402–1426.

[9] J. Hu, L. Wang, S. Li, Y. Jiang, X. Li, P. Weng, and Y. Ban, “Generalizable coarse-to-fine robot manipulation via language-aligned 3d keypoints,” arXiv preprint arXiv:2509.23575, 2025.

[10] Y. Dai, J. Lee, N. Fazeli, and J. Chai, “Racer: Rich language-guided failure recovery policies for imitation learning,” in Proceedings of International Conference on Robotics and Automation, 2025, pp. 15 657–15 664.

[11] L. Smith, A. Irpan, M. G. Arenas, S. Kirmani, D. Kalashnikov, D. Shah, and T. Xiao, “Steer: Flexible robotic manipulation via dense language grounding,” in Proceedings of International Conference on Robotics and Automation, 2025, pp. 16 517–16 524.

[12] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” in Proceedings of Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 44 776–44 791.

[13] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “Rlbench: The robot learning benchmark & learning environment,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3019–3026, 2020.

[14] M. Shridhar, L. Manuelli, and D. Fox, “Cliport: What and where pathways for robotic manipulation,” in Proceedings of Conference on Robot Learning, 2022, pp. 894–906.

[15] S. James and A. J. Davison, “Q-attention: Enabling efficient learning for vision-based robotic manipulation,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1612–1619, 2022.

[16] X. Ma, S. Patidar, I. Haughton, and S. James, “Hierarchical diffusion policy for kinematics-aware multi-task robotic manipulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 081–18 090.

[17] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[18] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7327–7334, 2022.

[19] C. Lynch, M. Khansari, T. Xiao, V. Kumar, J. Tompson, S. Levine, and P. Sermanet, “Learning latent plans from play,” in Proceedings of Conference on Robot Learning, 2020, pp. 1113–1132.

[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proceedings of Advances in Neural Information Processing Systems, vol. 30, 2017.

[21] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in Proceedings of International Conference on Machine Learning, 2021, pp. 8748–8763.

[22] P. Vincent, “A connection between score matching and denoising autoencoders,” Neural Computation, vol. 23, no. 7, pp. 1661–1674, 2011.

[23] R. Hadsell, S. Chopra, and Y. LeCun, “Dimensionality reduction by learning an invariant mapping,” in Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, vol. 2, 2006, pp. 1735–1742.

[24] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[25] Y. R. Wang, C. Ung, C. Tan, G. Tannert, J. Duan, J. Li, A. Le, R. Oswal, M. Grotz, W. Pumacay et al., “Roboeval: Where robotic manipulation meets structured and scalable evaluation,” in Proceedings of International Conference on Robotics and Automation, 2026.