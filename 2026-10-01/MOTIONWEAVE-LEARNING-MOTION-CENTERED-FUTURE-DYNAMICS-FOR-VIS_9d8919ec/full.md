# MOTIONWEAVE: LEARNING MOTION-CENTERED FUTURE DYNAMICS FOR VISION-LANGUAGE-ACTION POLICIES

Jingqiu Wang, Yan Wang

School of Data Science and Engineering East China Normal University, Shanghai, China

## ABSTRACT

Vision-Language-Action (VLA) models have recently incorporated world models to provide richer dynamic supervision beyond sparse action labels. However, explicitly predicting future images or videos may include control-irrelevant appearance, while guidance derived from holistic future visual representations and shared global action features may fail to establish timestep-specific correspondence between actions and local visual changes. To address this issue, we propose MotionWeave, a motion-centric futuredynamics framework for action-chunk prediction with two modules: the Action-Induced Motion Grounder (AIMG) and the Horizon Residual Composer (HRC). Specifically, AIMG conditions on action and proprioceptive representations to construct horizon-specific queries that localize interaction regions associated with each future action timestep from current visual tokens. HRC extracts differences between interaction representations at adjacent horizons, encodes them as temporal motion cues, and injects them into action tokens through a gated residual. During training, robot-arm masks rendered from future frames are used to construct KL-based motiongrounding supervision, while inference uses only the current observation. On six MetaWorld tasks, MotionWeave achieves a 75.3% average success rate, an absolute gain of 8.6% over π<sub>0</sub> (66.7%), especially on sustained-interaction tasks. Our code is available at https://github.com/autu-mn/MotionWeave.

Index Terms— Embodied Intelligence, vision-language-action models, world models, robot manipulation, multimodal learning

## 1. INTRODUCTION

Vision-Language-Action (VLA) models unify visual perception, language understanding, and action generation, providing an effective paradigm for learning generalizable robot policies [1, 2, 3, 4, 5, 6]. Existing approaches typically build on pretrained visionlanguage models [7, 8, 9] and use robot action data to predict continuous control signals directly from current observations. However, low-dimensional and sparse action supervision mainly constrains the final control output, making it difficult to explicitly characterize which visual regions change during manipulation and how these changes evolve along an action sequence. Consequently, the visual representations may lack the local interaction information and temporal cues required for precise manipulation.

To obtain richer dynamic supervision, recent studies have introduced world models into VLA by predicting future images, videos, or latent representations [11, 12, 13, 14, 15, 16, 17]. WorldVLA [13] and DreamVLA [15] explicitly reconstruct or densely imagine complete future scenes, inevitably entangling control-irrelevant static appearance and background; WoG [16] and Fast-WAM [17] relax pixel-level reconstruction into implicit or test-time-free guidance, yet still supervise the holistic evolution of future visual observations rather than horizon-specific local changes. Meanwhile, action chunking typically decodes a sequence from shared global features [18, 19], leaving different timesteps with homogeneous visual evidence and no explicit timing-to-consequence correspondence [20, 21]. No unified path yet exists from localizing actionrelevant regions to step-wise control via their temporal evolution.

Based on these observations, we argue that VLA should not redundantly reconstruct complete future visual observations, but should instead explicitly localize action-relevant dynamic regions and exploit their temporal evolution to guide chunk generation. To this end, we propose MotionWeave, a motion-centered future dynamics learning framework composed of two tightly coupled modules: the Action-Induced Motion Grounder (AIMG) for horizon-specific region localization, and the Horizon Residual Composer (HRC) for fine-grained motion extraction and injection.

AIMG conditions on action representations and proprioceptive states to construct time-indexed queries, attending to current visual tokens to localize regions likely to change at each future horizon and producing interaction representations aligned with the action timeline. HRC then differences adjacent-horizon interaction representations to encode direction and magnitude of change, injecting them into the corresponding action representation via a gated residual. Together they close the loop from region localization, through motion cues, to step-wise control. Future observations provide dynamic supervision only during training and are not used for inference. The policy therefore retains direct action prediction at inference time without reconstructing complete future visual observations or incurring additional inference overhead.

The main contributions of this paper are as follows:

• We propose the Action-Induced Motion Grounder (AIMG), which constructs action- and state-conditioned horizon queries and uses future robot-region supervision to shape their attention over the current VLM visual tokens. This provides different action horizons with spatially differentiated visual evidence.

• We propose the Horizon Residual Composer (HRC), which represents the temporal evolution of the grounded interaction features through adjacent-horizon differences and injects the resulting cues into action representations through a gated residual update.

• We achieve state-of-the-art performance on six MetaWorld tasks, achieving an average success rate of 75.3%, an absolute gain of 8.6% over π<sub>0</sub>.

![](images/653d4079df583e6c687bae725e2a0585949493451be84194d0245c343c8520ce.jpg)  
Fig. 1. Overview of MotionWeave. The current observation $I _ { t }$ and instruction l are tokenized into visual tokens and text tokens, concatenated with a learnable action query placeholder $A _ { 0 } ,$ , and jointly processed by LLM to produce action tokens $A ;$ a separate proprioceptive encoder maps $S _ { t }$ to $P .$ . The Action-Induced Motion Grounder (AIMG) combines A and $\bar { P }$ into horizon-specific queries that attend over visual tokens V to produce interaction tokens M with spatial attention α. The Horizon Residual Composer (HRC) injects adjacent-horizon differences of M into A via gate $g$ to obtain $A ^ { \prime }$ , decoded by Action DiT into a 4-step chunk. Robot-arm masks [10] supervise α via KL loss in training only.

## 2. METHODOLOGY

## 2.1. Overall Framework

We consider vision-language-action control with action chunking. At time step t, the policy is conditioned on the current RGB observation $I _ { t } ,$ language instruction l, and proprioceptive state $S _ { t } .$ , and predicts a future action chunk $\hat { a } _ { t : t + H - 1 }$ of length $H = 4 ,$ , i.e., the conditional distribution $p ( \hat { a } _ { t : t + H - 1 } \mid I _ { t } , l , S _ { t } )$ . Future RGB observations $I _ { t + 1 : t + H }$ are used only during training to construct horizonspecific motion supervision and are unavailable during inference. For simplicity, we omit the time index t below.

As shown in Fig. 1, image, text, and proprioception are processed through three branches. InternVL3-2B encodes $I _ { t }$ and l into visual and text tokens, which are concatenated with a learnable action-query placeholder $A _ { 0 }$ and jointly processed by its Qwen2.5-series language decoder. The model outputs action tokens $A ~ \in ~ \mathbb { R } ^ { N _ { a } \times D ^ { \mathbf { \lambda } } }$ at the $A _ { 0 }$ positions and spatial visual tokens $V = \{ v _ { n } \} _ { n = 1 } ^ { N _ { v } }$ . Here $N _ { v }$ is the number of visual tokens actually output by the InternVL3-2B forward pass (in our implementation $N _ { v } = 2 5 6$ , corresponding to a 16 × 16 spatial grid); both the AIMG attention $\boldsymbol { \alpha } \in \mathbb { R } ^ { \hat { H } \times N _ { v } }$ and the motion-supervision masks are built on this same grid to ensure strict alignment among the three. Separately, a Proprio encoder maps $S _ { t }$ to the proprioceptive embedding P; $P$ does not enter the language model and is supplied directly to the subsequent modules. The overall process is

$$
\begin{array} { r } { ( A , V , P ) \xrightarrow { \mathrm { A I M G } } ( M , \alpha ) \xrightarrow { \mathrm { H R C } } A ^ { \prime } \xrightarrow { \mathrm { A c t i o n D i T } } \hat { a } _ { t : t + H - 1 } . } \end{array}\tag{1}
$$

Here, AIMG retrieves horizon-specific evidence from $V$ to produce M and $\alpha$ , HRC writes the temporal evolution of M back into $A$ to obtain $A ^ { \prime } .$ , and Action DiT decodes aˆ. Both modules are jointly trained with the action and motion-grounding objectives.

## 2.2. Action-Induced Motion Grounder (AIMG)

AIMG does not reconstruct future RGB frames. Instead, it identifies which of the current $N _ { v }$ visual-token positions will change within the next H steps. Let $e _ { h }$ be the learnable horizon embedding for the h-th future timestep, where $h = 1 , \dots , H$ . After LayerNorm and linear projection, the action tokens A and proprioceptive embedding P are mapped into a common motion space of dimension $d _ { m } ,$ and horizon-specific queries are formed as

$$
q _ { h } = e _ { h } + \mathrm { M e a n } ( \mathrm { P r o j } _ { a } ( A ) ) + \mathrm { P r o j } _ { p } ( P ) ,\tag{2}
$$

where Mean(·) is taken over the projected action queries so that each horizon query carries global action guidance. A lightweight residual transformation then refines $q _ { h }$ . Let Proj denote the shared visualvalue projection. The attention and interaction representations [22] are:

$$
\alpha _ { h , n } = \mathrm { s o f t m a x } _ { n } \bigg ( \frac { q _ { h } ^ { \top } \mathrm { P r o j } _ { v } ( v _ { n } ) } { \sqrt { d _ { m } } } \bigg ) ,\tag{3}
$$

$$
m _ { h } = \mathrm { L N } \Bigg ( q _ { h } + \sum _ { n = 1 } ^ { N _ { v } } \alpha _ { h , n } \mathrm { P r o j } _ { v } ( v _ { n } ) \Bigg ) .\tag{4}
$$

The spatial attentions form $\mathrm { A t t n ~ } \in \ \mathbb { R } ^ { H \times N _ { v } }$ and receive KL motion-grounding supervision. The aggregated interaction tokens $M = [ m _ { 1 } , \dots , m _ { H } ]$ are passed to HRC. All $m _ { h }$ come from the current observation and are differentiated by $e _ { h }$ , aligning them with future action timesteps.

During training, each sample provides $I _ { t } , I _ { t + 1 } , \ldots , I _ { t + H }$ . For each $h ,$ we render the corresponding robot-arm mask with Robot Engine [10] and adopt the mask of $I _ { t + h }$ as the horizon-specific motion target: it marks the arm region executing the h-th future action while suppressing static background, texture, and other taskirrelevant appearance, so the mask itself serves as the motion of interest. Each binary mask is downsampled to the $1 6 \times 1 6$ visualtoken grid, smoothed with a small constant for numerical stability, and normalized to yield the target distribution $\mu _ { h }$ . This construction requires no manual annotation and reuses existing expert trajectories with off-the-shelf mask generation.

## 2.3. Horizon Residual Composer (HRC)

The interaction representations produced by AIMG are independent and do not explicitly encode scene evolution. After projecting M into the HRC space, we define

$$
\Delta m _ { h } = m _ { h } - m _ { h - 1 } , \qquad m _ { 0 } : = m _ { 1 } ,\tag{5}
$$

so that $\Delta m _ { 1 } = 0$ and later differences encode the direction and magnitude of change relative to the preceding timestep. HRC uses motion differences as queries and action queries as keys and values in cross-attention, allowing each local-evolution cue to probe the action query that needs correction:

$$
U = \mathrm { M H A } \left( \mathrm { P r o j } _ { m } ( \Delta M ) , \mathrm { P r o j } _ { a } ( A ) , \mathrm { P r o j } _ { a } ( A ) \right) ,\tag{6}
$$

where $\Delta M = [ \Delta m _ { 1 } , . . . , \Delta m _ { H } ]$ and ${ \mathrm { P r o j } } _ { a }$ is the HRC-side action projection, independent of the AIMG-side projection [22]. The update and proprioception-conditioned residual gate are

$$
\Delta A = \mathrm { P r o j } _ { o } ( U ) , \qquad g = \sigma \big ( \mathrm { P r o j } _ { g } ( P ) \big ) ,\tag{7}
$$

and the motion-grounded action query is obtained by subtractive residual update:

$$
A ^ { \prime } = A - g \odot \Delta A .\tag{8}
$$

Unlike direct overwriting, this form lets each action query absorb only the motion increment associated with its timestep. The correction direction comes from cross-attention over temporal differences, while the gate adaptively controls injection strength and preserves the original control prior in A.

## 2.4. Training Objective

Conditioned on A<sup>′</sup>, P, noisy action chunk $a ^ { \tau }$ , and diffusion timestep $\tau ,$ Action DiT [23] predicts $\hat { a } _ { t : t + H - 1 }$ with the flow-matching [24] loss $\mathcal { L } _ { \mathrm { a c t } }$ . AIMG aligns each $\alpha _ { h }$ with its mask distribution $\mu _ { h } \colon$

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \frac { 1 } { H \log N _ { v } } \sum _ { h = 1 } ^ { H } \mathrm { K L } ( \mu _ { h } \parallel \alpha _ { h } ) ,\tag{9}
$$

where $\mathrm { K L } ( \mu _ { h } \parallel \alpha _ { h } )$ penalizes attention mass placed outside the arm region at horizon $h ,$ and division by log $N _ { \imath }$ normalizes the loss by the entropy of the uniform distribution over visual-token positions. The complete objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { m o t i o n } } \mathcal { L } _ { \mathrm { m o t i o n } } ,\tag{10}
$$

with $\lambda _ { \mathrm { m o t i o n } } = 0 . 0 5$ in the implemented configuration. At inference time, the policy receives only $I _ { t } , l ,$ and $S _ { t } \colon$ the mask-rendering and KL branches are removed, and DiT denoising outputs four actions without reconstructing future visual observations.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Tasks and data. We select six challenging MetaWorld [25] tasks: Pick Place, Disassemble, Stick Pull, Assembly, Shelf Place, and Hand Insert, covering grasping, assembly, pulling, insertion, and long-horizon placement. Each task contains 25 expert trajectories of 175 timesteps, for 150 trajectories and 26,250 frames in total. The input consists of a single-view 224×224 RGB image, a language instruction, and the proprioceptive state, with supervision on the next 4-step actions (H = 4).

Model and training. Our method MotionWeave builds on InternVL3-2B, whose language side uses a Qwen2.5-series decoder [26, 9], with full fine-tuning. Training runs for 20k steps on two 48GB NVIDIA A40 GPUs with FSDP full sharding and gradient checkpointing; the per-GPU batch size is 8 and the global batch size is 16. The optimizer uses learning rate $1 \times 1 0 ^ { - 5 }$ , weight decay 0, and gradient clipping 1.0. The action head is a 12-layer, 768-dimensional, 12-head DiT [23] with a 4-step action horizon, a flow-matching objective [24], 100 training diffusion steps, and 10 inference steps. AIMG uses 4 horizon queries with hidden dimension 512, supervised by future motion maps from t + 1 to t + 4; HRC uses 768-dimensional, 8-head attention to update action tokens through a gated residual. The total loss is $\mathcal { L } _ { \mathrm { a c t } } + 0 . 0 5 \mathcal { L } _ { \mathrm { m o t i o n } } ;$ future frames are used only during training.

Evaluation protocol. We evaluate each task over 25 episodes using initialization seeds shared across methods, with a 175-step limit and replanning every 2 executed steps. Success follows the official MetaWorld binary criterion; we report the macro-average success rate across six tasks.

## 3.2. Main Results

Table 1 compares MotionWeave with π<sub>0</sub> [5], DreamVLA [15], WoG [16], and Fast-WAM [17] under the shared protocol. The baselines cover the two alternatives discussed in Sec. 1: direct action mapping from global features (π ), and action guidance based on modeling complete future visual observations (DreamVLA, WoG, Fast-WAM). Predicting complete future visual observations introduces abundant static background, texture, and appearance content that is redundant for action decisions, whereas Motion-Weave directly models action-relevant motion regions, directions, and magnitudes, concentrating limited capacity on how the robot and objects will change. MotionWeave reaches an average success rate of 75.3%, yielding an absolute gain of 8.6% over the strongest baseline π , and leads on five of six tasks. The advantage is largest on sustained-interaction tasks, including Disassemble (92.0% vs. 68.0%) and Shelf Place (76.0% vs. 64.0%). On Pick Place, π performs better (72.0% vs. 60.0%), where the target displacement is short and global features already suffice. Fig. 2 shows rollouts on all six tasks from observation to successful completion.

## 3.3. Ablation Studies

Table 2 separates the effects of horizon-specific grounding, explicit attention supervision, and temporal composition. Adding the AIMG architecture without the grounding objective improves the average success rate from 58.0% to 62.0%. This modest gain suggests that horizon embeddings and action-conditioned queries already introduce useful temporal differentiation, but the action loss alone provides only indirect spatial guidance. Supervising the horizon-wise attention distributions further raises the result to 70.0%, supporting the central role of AIMG: future robot-region targets shape how each horizon query retrieves control-relevant evidence from the current VLM visual tokens. Finally, HRC improves the result from 70.0% to 75.3%. Thus, horizon-specific localization is useful on its own, while explicitly composing differences between adjacent grounded representations provides additional temporal information for action generation. Together, the ablation follows the intended design path from spatial grounding to temporal differencing and action conditioning.

![](images/41ca4013621f83d7737c115c43802b830a2168d3ad87ffb76038ad9c60a499db.jpg)  
Fig. 2. Qualitative rollouts on six tasks. Each row shows key frames from observation to successful completion.

Table 1. Success rates (%) on six MetaWorld tasks. All methods share the evaluation random seeds, and bold denotes the best result for each task. The “Source” column reports the publication status of each baseline.
<table><tr><td>Method</td><td>Source</td><td>Pick Place</td><td>Disassemble</td><td>Stick Pull</td><td>Assembly</td><td>Shelf Place</td><td>Hand Insert</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>π0 [5]</td><td>RSS&#x27;25</td><td>72.0</td><td>52.0</td><td>68.0</td><td>76.0</td><td>64.0</td><td>68.0</td><td>66.7</td></tr><tr><td>DreamVLA [15]</td><td>NeurIPS’25</td><td>60.0</td><td>68.0</td><td>72.0</td><td>70.0</td><td>56.0</td><td>64.0</td><td>65.0</td></tr><tr><td>WoG [16]</td><td>ICML&#x27;26</td><td>28.0</td><td>60.0</td><td>68.0</td><td>24.0</td><td>64.0</td><td>60.0</td><td>50.7</td></tr><tr><td>Fast-WAM [17]</td><td>arXiv&#x27;26</td><td>44.0</td><td>56.0</td><td>16.0</td><td>64.0</td><td>56.0</td><td>60.0</td><td>49.3</td></tr><tr><td>MotionWeave (ours)</td><td></td><td>60.0</td><td>92.0</td><td>76.0</td><td>80.0</td><td>76.0</td><td>68.0</td><td>75.3</td></tr></table>

Table 2. Component ablation results for MotionWeave. We report macro-average success (%) over the six tasks. Each row adds one component relative to the preceding configuration.
<table><tr><td>Variant</td><td>Avg.</td></tr><tr><td>Baseline</td><td>58.0</td></tr><tr><td> $+ \mathrm { A I M G } \mathrm { w } / 0 \mathcal { L } _ { \mathrm { m o t } }$ </td><td>62.0</td></tr><tr><td>+ AIMG</td><td>70.0</td></tr><tr><td> $\mathbf { + A I M G + H R C }$ </td><td>75.3</td></tr></table>

Fig. 3 reports sensitivity to two hyperparameters: the horizon query count peaks at 4 with 75.3%, aligned with the action chunk length $H \ = \ 4 ,$ , so one query per action step provides a clearer horizon-action correspondence; redundant queries introduce tokens with competing attention. The motion loss weight peaks at $\lambda _ { \mathrm { m o t i o n } } = 0 . 0 5$ , after which success decreases.

## 4. CONCLUSION

MotionWeave enables action-chunk policies to represent the visual changes relevant to each action step without reconstructing complete future visual observations. AIMG uses action- and proprioceptionconditioned horizon queries to localize interaction regions in current visual tokens, while HRC encodes changes between adjacent horizons and injects these motion cues into the corresponding action tokens through a gated residual. Future-frame masks supervise horizon-specific grounding during training, allowing the policy to predict actions directly from the current observation at inference. Ablation results confirm that both horizon-wise grounding and temporal differencing contribute to the overall performance gain. On six MetaWorld tasks, MotionWeave achieves an average success rate of 75.3%, yielding an absolute gain of 8.6% over the strongest baseline π , and improves performance on five tasks. These results show that aligning localized motion evidence with action horizons improves visuomotor control, especially in tasks that require sustained interaction. Future work will investigate whether the motion-grounded representations learned in simulation can generalize to real-robot settings with different viewpoints, object appearances, contact dynamics, and execution noise. Extending the evaluation to unseen tasks and hardware platforms will further clarify the transferability of horizon-specific motion guidance.

![](images/95dd2be52a95768a51f74d28fc8ae348e896373412834df073e49cb2f1f763de.jpg)

![](images/62d96c3ad12e0a26bb124cc7afa925a246fc3f622b1529d3c102c8cb5b5eef9a.jpg)  
Fig. 3. Hyperparameter curves in average success (%). (a) Horizon query count peaks at 4 with 75.3%, aligned with the action chunk length $H = 4 ,$ , falling back to 73.3%/70.0% at 6/8 queries. (b) Motion loss weight λ<sub>motion</sub> peaks at 0.05 with 75.3%, falling back to 72.7%/68.7% at 0.10/0.20.

## 5. REFERENCES

[1] Anthony Brohan, Noah Brown, Justice Carbajal, et al., “RT-1: Robotics transformer for real-world control at scale,” in Robotics: Science and Systems XIX, 2023.

[2] Brianna Zitkovich, Tianhe Yu, Sichun Xu, et al., “RT-2: Vision-Language-Action models transfer web knowledge to robotic control,” in Proceedings of the 6th Conference on Robot Learning, 2023, vol. 229 of Proceedings of Machine Learning Research, pp. 2165–2183.

[3] Octo Model Team, Dibya Ghosh, Homer Walke, et al., “Octo: An open-source generalist robot policy,” in Robotics: Science and Systems XX, 2024.

[4] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al., “OpenVLA: An open-source Vision-Language-Action model,” in Conference on Robot Learning, 2024, vol. 270 of Proceedings ofMachine Learning Research, pp. 2679–2713.

[5] Kevin Black, Noah Brown, Danny Driess, et al., “π<sub>0</sub>: A Vision-Language-Action flow model for general robot control,” in Robotics: Science and Systems XXI, 2025.

[6] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, et al., “Open X-Embodiment: Robotic learning datasets and RT-X models,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 6892–6903.

[7] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, 2021, vol. 139 of Proceedings ofMachine Learning Research, pp. 8748–8763.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[9] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, et al., “Qwen2.5 technical report,” CoRR, vol. abs/2412.15115, 2024.

[10] Xinyu Hou, Mingxi Cheng, Tianyi Zhang, et al., “Robot Engine: A unified engine for robot data generation,” CoRR, vol. abs/2503.18738, 2025.

[11] Yilun Du, Sherry Yang, Bo Dai, et al., “Learning universal policies via text-guided video generation,” CoRR, vol. abs/2307.06405, 2023.

[12] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap, “Mastering diverse domains through world models,” CoRR, vol. abs/2301.04104, 2023.

[13] Jun Cen, Chaohui Yu, Hangjie Yuan, et al., “WorldVLA: Towards autoregressive action world model,” CoRR, vol. abs/2506.21539, 2025.

[14] Yucheng Hu, Yanjiang Guo, Pengchao Wang, et al., “Video Prediction Policy: A generalist robot policy with predictive visual representations,” CoRR, vol. abs/2412.14803, 2025.

[15] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin, “DreamVLA: A Vision-Language-Action model dreamed with comprehensive world knowledge,” in Advances in Neural Information Processing Systems, 2025, vol. 38, pp. 27463– 27496.

[16] Yue Su, Sijin Chen, Haixin Shi, Mingyu Liu, Zhengshen Zhang, Ningyuan Huang, Weiheng Zhong, Zhengbang Zhu, Yuxiao Liu, and Xihui Liu, “World Guidance: World modeling in condition space for action generation,” in Proceedings ofthe 43rd International Conference on Machine Learning (ICML), Seoul, South Korea, July 2026.

[17] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao, “Fast-WAM: Do world action models need test-time future imagination?,” CoRR, vol. abs/2603.16666, 2026.

[18] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn, “Learning fine-grained bimanual manipulation with lowcost hardware,” in Robotics: Science and Systems XIX, 2023.

[19] Cheng Chi, Zhenjia Xu, Siyuan Feng, et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems XIX, 2023.

[20] Ruijie Zheng, Yongyuan Liang, et al., “TraceVLA: Visual trace prompting enhances manipulation,” CoRR, vol. abs/2412.06045, 2024.

[21] Chuan Wen, Xingyu Lin, John So, et al., “Any-Point trajectory modeling for policy learning,” in Robotics: Science and Systems XX, 2024.

[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, 2017, vol. 30, pp. 5998–6008.

[23] William Peebles and Saining Xie, “Scalable diffusion models with transformers,” in IEEE/CVF International Conference on Computer Vision, 2023, pp. 22470–22480.

[24] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le, “Flow matching for generative modeling,” in The Eleventh International Conference on Learning Representations, 2023.

[25] Tianhe Yu, Deirdre Quillen, Zhanpeng He, et al., “Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Conference on Robot Learning, 2020, vol. 100 of Proceedings ofMachine Learning Research, pp. 1094– 1100.

[26] Jinguo Zhu, Weiyun Wang, Zhe Chen, et al., “InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models,” CoRR, vol. abs/2504.10479, 2025.