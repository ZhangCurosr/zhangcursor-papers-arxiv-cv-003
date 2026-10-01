# IS BETTER TEACHER SUPERVISION ENOUGH? UNLOCKING STUDENT-SIDE LEARNING IN MULTI-MODAL ON-POLICY DISTILLATION

Siyuan Liu<sup>1</sup> Kanghui Tian<sup>2</sup> Yue Duan<sup>1</sup> Yutao He<sup>1</sup>

Shangdong Yang<sup>3</sup> Jian Zhang<sup>1†</sup> Yinghuan Shi<sup>1†</sup>

<sup>1</sup>Nanjing University <sup>2</sup>Fudan University <sup>3</sup>Nanjing University of Posts and Telecommunications syliu@smail.nju.edu.cn, {zhang.jian,syh}@nju.edu.cn

![](images/2d080723410889c68e7f05ef6e9e674f840c1b64e0f6bb940d00fa96b96f9efa.jpg)  
Figure 1: Beyond enhancing teacher-side supervision: Previous methods improve the teacherside signal, while S-OPD adds two student-side objectives to enhance perception. This improves performance across model scales (2B and 4B) and distillation paradigms (OPD and OPSD).

## ABSTRACT

On-policy distillation (OPD) improves reasoning by providing token-level supervision from a teacher on a student’s own trajectories. Existing methods primarily focus on enhancing this teacher-side guidance (e.g., by enriching teacher inputs and refining teacher feedback), yet we find that limited student perception is another critical bottleneck in multimodal OPD. By providing oracle visual facts, the performance of OPD-trained students can still be substantially improved for both weak and strong teachers. To address this bottleneck, we propose S-OPD, a simple multimodal on-policy distillation framework that explicitly strengthens student perceptual learning through two objectives. Specifically, Teacher-calibrated Policy Contrast separates student policies under original and masked images with teacher-based token-level gating, strengthening the student’s reliance on visual evidence during reasoning. Policy Agreement aligns student policies under original and noise-perturbed images, further improving perceptual robustness to visual noise. Notably, our method can be seamlessly plugged into existing OPD frameworks, requiring no additional data annotations, model parameters or inference operations. Extensive experiments on eight benchmarks across student scales and

distillation paradigms demonstrate consistent performance improvements, with gains of up to 4.25 points on LogicVista. When combined with existing teacherside supervision methods, our method can yield further gains. Code is available at https://github.com/Sirilaw/S-OPD.

## 1 INTRODUCTION

Multimodal large language models (MLLMs) (Bai et al., 2025; Zhu et al., 2025) have demonstrated strong capabilities in solving tasks that require models to integrate visual information with multistep reasoning, like visual question answering (Liu et al., 2024; Ma et al., 2026), geometry problem solving (Lu et al., 2021; 2024) and chart-oriented reasoning (Masry et al., 2022; Chen et al., 2025). This motivates efforts to equip compact models for these capabilities with lower deployment costs. On-policy distillation (OPD) (Gu et al., 2024; Agarwal et al., 2024; Lu & Lab, 2025) is appealing as it trains a student on its own trajectories using feedback from a strong teacher, helping the student learn better behavior at its own rollout distribution during training. Compared to outcome-based reinforcement learning (Shao et al., 2024; Guo et al., 2025), OPD provides dense token-level supervision by scoring student-generated prefixes with the teacher model, allowing intermediate decisions to receive direct supervision.

Recent efforts to improve OPD primarily focus on two aspects: enriching teacher inputs and refining teacher feedback. The first provides teachers with additional privileged information, such as reference rationales (Zhao et al., 2026) and in multimodal settings, image crops (Yuan et al., 2026) or recoverable visual cues (Tian et al., 2026b). The second utilizes differences of teacher predictions conditioned on various visual inputs to reweight the token-level supervision (Liu et al., 2026; Liang et al., 2026) or reconstruct teacher targets (Zhang et al., 2026). Despite these advances, genuine multimodal reasoning requires models to integrate visual perception, a critical component that provides visual evidence, with multi-step textual reasoning (Zhang et al., 2024; Tong et al., 2024; Wang et al., 2026). Prior approaches seek to improve the teacher-side guidance that the student receives. Yet this guidance is conveyed through text-token supervision. Students with limited visual perception may struggle to ground the guidance in the visual inputs, limiting the benefits of distillation. This motivates us to ask: Is student-side visual perception capability another critical bottleneckfor multimodal OPD?

![](images/612f05acc73349f39ef57227c62f56ebae9fd25dc742899213c5c41e3aac01b1.jpg)  
(a) Evaluation pipeline.

![](images/654a4f9a8a174497fc80c03dc6b82894527fd99ee5d8732c2e992c92e24ad294.jpg)  
(b) Accuracy and accuracy gaps for 2B and 4B students.  
Figure 2: Oracle visual facts evaluation on Geometry3K.

To investigate this question, we evaluate base and OPD-trained students, including Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct (Bai et al., 2025), on the same Geometry3K problems with and without oracle visual facts (Figure 2a). These facts describe detailed diagram content without providing answers or reasoning steps, reducing the demand on visual perception (see Section 3.1 for more details). The resulting accuracy gap indicates how much students benefit when the required visual information is explicitly available in textual form. A larger gap suggests that students still struggle to extract task-relevant visual evidence on their own despite training. As shown in Figure 2b, we find that substantial gaps persist after OPD, even with stronger teachers (Li et al., 2026a) that are trained with GRPO (Shao et al., 2024), suggesting that teacher supervision alone does not sufficiently address students’ perceptual limitations.

The persistent gap raises a further question: How does student visual perception influence reasoning performance after OPD on different benchmarks? To examine this, we systematically analyze four benchmarks covering mathematical, logical and general domains (see Section 3.1 for more details). Specifically, we estimate the KL divergence between two student policies under original and masked images to measure the visual sensitivity of the student. We then divide samples into four quartiles according to this measure and compare model performance across groups. As shown in Figure 3, samples with higher KL divergence–i.e. those exhibiting greater sensitivity to visual input–generally achieve better performance across both student scales. This observation demonstrates a strong positive correlation between the student visual perception and reasoning performance, highlighting the necessity of explicitly strengthening student perceptual learning within OPD.

Building upon our earlier observations, how can we exploit OPD’s token-level learning structure to explicitly strengthen student perception? We propose S-OPD, a simple multimodal OPD framework that explicitly strengthens student perceptual learning with two KL objectives. Our key idea is to shape student perceptual sensitivity according to whether an input change alters visual evidence: visual evidence removal should induce a corresponding policy change, while perturbations preserving task-relevant content should maintain consistent predictions. To enhance perceptual sensitivity to evidence removal, Teacher-calibrated Policy Contrast (TPC) maximizes the KL divergence between student policies under original and masked images. To focus this objective on visually dependent tokens (Huang et al., 2026),

![](images/4de572c15ca2d48caed72795c1e0fad627688a900c9c27e1c1b5cb9dcba44c57.jpg)  
Figure 3: Performance across quartiles of estimated KL divergence between student policies under original and masked images (Q1–Q4, lowest to highest). A clear trend is observed: Higher-KL groups generally achieve higher accuracy.

we use differences in the teacher’s predictions under the same two images to gate which tokens KL maximization is applied to. Ablations of different gating strategies in Section 4.2 demonstrate the importance of this teacher-based selection. However, encouraging policy divergence under evidence removal alone does not constrain the student’s response to irrelevant visual variations. We therefore introduce Policy Agreement (PA) as a complementary regularizer that minimizes the KL divergence between student policies under original and noise-perturbed images.

As shown in Figure 1, despite its simplicity, S-OPD achieves consistent performance gains on various reasoning tasks across student scales and distillation paradigms. In particular, methods that enhance teacher-side supervision achieve further performance gains when equipped with S-OPD, highlighting the additional benefits of our student perceptual learning.

Our contributions are:

• We investigate an overlooked student-side visual perception bottleneck in multimodal onpolicy distillation, and reveal a positive association between students’ perceptual sensitivity and reasoning performance across diverse tasks and models.

• We propose S-OPD, a simple multimodal OPD framework that enhances student perceptual learning with two KL objectives targeting perceptual sensitivity and stability.

• Extensive experiments demonstrate consistent performance improvements across multiple student scales, distillation paradigms and benchmarks, with gains reaching 4.25 points on LogicVista. Methods that enhance teacher-side supervision achieve further gains when augmented with S-OPD.

## 2 RELATED WORK

On-Policy Distillation. Knowledge distillation (Hinton et al., 2015) trains a student to match a larger teacher’s predictive distribution, while sequence-level distillation (Kim & Rush, 2016) uses teacher-generated sequences as training targets. However, training on fixed or teacher-generated sequences creates a mismatch with inference (Agarwal et al., 2024), where students condition on their own generated prefixes. On-policy distillation (Gu et al., 2024; Lu & Lab, 2025) addresses this mismatch by having the teacher provide dense token-level supervision on trajectories sampled by the student. Recent work improves this supervision along two dimensions: enriching teacher inputs (Zhao et al., 2026; Tian et al., 2026a) to improve teacher advantage and refining teacher feedback (Jin et al., 2026; Jang et al., 2026; Xu et al., 2026) to improve the reliability of tokenlevel guidance. These efforts primarily enhance teacher-side supervision, while S-OPD focuses on student-side learning to help students better exploit the teacher guidance they receive.

Multimodal On-Policy Distillation. Recent work improves multimodal supervision from three perspectives: enriching its sources through text-to-vision reasoning transfer (Bousselham et al., 2026) or heterogeneous teacher capabilities (Yin et al., 2026); constructing informative teacher– student asymmetry through privileged teacher crops (Yuan et al., 2026), recoverable visual cues (Tian et al., 2026b), or augmented student views (Li et al., 2026b); and refining distillation signals through visual interventions (Sun et al., 2026) and modality-aware constraints (Aniri et al., 2026). Within signal refinement, VA-OPD (Liu et al., 2026) uses teacher visual advantage to allocate supervision across rollouts and token groups and VAD (Zhang et al., 2026) reconstructs visually attributable distillation targets. Beyond these advances, S-OPD directly shapes student visual perceptual sensitivity through two cross-view KL objectives, explicitly promoting sensitivity to evidence removal and stability under content-preserving perturbations.

Perceptual Learning for Multimodal Reasoning. Prior work has explored multiple training and inference strategies for improving visual perception. Visual contrast uses prediction differences to strengthen visual grounding in decoding (Leng et al., 2024) and preference optimization (Xie et al., 2024), while consistency learning promotes the robustness of neural networks in stability training (Zheng et al., 2016) and semi-supervised learning (Tarvainen & Valpola, 2017). In multimodal reinforcement learning, Video-R1 (Feng et al., 2025) contrasts temporally ordered and shuffled video inputs to encourage temporal reasoning, and PAPO (Wang et al., 2026) encourages separation between original- and masked-image policies to strengthen visual perception. Token-selective method (Huang et al., 2026) focus optimization on tokens with higher visual dependency.

Unlike prior applications of visual contrast and consistency, S-OPD addresses the student perceptual bottleneck that limits the effectiveness of teacher supervision in OPD. Notably, S-OPD uses the changes in teacher-assigned token probabilities to select tokens for policy contrast. Teacher feedback thus provides both distillation targets and token-level calibration for student perceptual learning.

## 3 METHOD

We propose S-OPD, a simple framework that explicitly shapes student visual perception sensitivity through two KL objectives. Teacher-Calibrated Policy Contrast (TPC) maximizes the KL divergence between student policies under original and masked images at tokens selected using the teacher’s cross-image responses, strengthening perceptual sensitivity to evidence removal. Policy Agreement (PA) minimizes the KL divergence between student policies under original and mildly perturbed images, improving perceptual stability under content-preserving perturbations. Figure 1 illustrates the overall framework and Figure 4 shows the details and training dynamics of our method.

## 3.1 PROBLEM FORMULATION

Let D denote the training distribution of multimodal inputs $x = ( q , I )$ , where $q$ is a question and I is an image. Let $\pi _ { \theta }$ denote the student policy and $\pi _ { T }$ the frozen teacher policy. For each input, the student samples a response $y = ( y _ { 1 } , \dots , y _ { T } ) \sim \pi _ { \theta } ( \cdot \mid x )$ , where $T$ is the response length. We define $c _ { t } = ( q , I , y _ { < t } )$ as the context at step t, under which both policies are evaluated using the same student-generated prefix.

Following prior OPD studies (Gu et al., 2024; Lu & Lab, 2025), we train the student to minimize the reverse KL divergence to the teacher:

$$
\mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot \mid c _ { t } ) \parallel \pi _ { T } ( \cdot \mid c _ { t } ) \right) \right] .\tag{1}
$$

In practice, we estimate the KL divergence using the sampled-token $k _ { 1 }$ estimator, log $\pi _ { \theta } ( y _ { t } \mid c _ { t } ) -$ $\log \pi _ { T } ( y _ { t } \mid c _ { t } )$ , evaluated on student-sampled tokens.

![](images/e5ddb9e684d25b97600db2c6c1524851cd1c4f4ce0c8f0ceeb207b3eb3dcd0b9.jpg)  
(a) Details of two learning objectives in S-OPD.

![](images/e4ef56284f698a1e0fcd7077cc8e0fd4ee547e57a2faf39dfb23ac0c9c20f9d1.jpg)  
(b) Training dynamics of 2B and 4B student models.  
Figure 4: (a) Details of S-OPD with original, masked, and noisy images. The teacher compares original and masked images to gate student policy contrast, while policy agreement aligns the student’s original and noisy policies. (b) S-OPD achieves higher performance and reduces the teacher-student log-probability gap during training on Geometry3K.

Bottleneck analysis of multimodal on-policy distillation. We first investigate whether student visual perception remains a bottleneck for multimodal OPD. We train Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct (Bai et al., 2025) on ViRL39K for 1 epoch with standard OPD (Lu & Lab, 2025). We use Qwen3-VL-8B-Instruct (Bai et al., 2025) and its variant trained with GRPO on ViRL39K (Wang et al., 2025) as teachers, following the use of post-trained teachers to strengthen supervision (Li et al., 2026a). Base and OPD-trained students are evaluated on the Geometry3K test split with and without oracle visual facts. These facts are constructed by verbalizing official diagram logic form annotations with fixed templates, without accessing answers or ra tionales. We insert the facts after the answer choices as a Diagram observations block, keeping all other inputs and decoding settings fixed. Construction and audit details can be found in Appendix A.3. As shown in Figure 2b, oracle facts substantially improve accuracy even after OPD with the stronger GRPO teacher, indicating persistent student limitations in visual perception, i.e. the ability to recover visual evidence.

Visual perception sensitivity and reasoning performance. We next examine how student visual perception sensitivity relates to reasoning performance on Geometry3K (Lu et al., 2021), Math-Vista (Lu et al., 2024), LogicVista (Xiao et al., 2024), and MMMU (Yue et al., 2024), covering mathematical, logical and general reasoning. For each student response generated from the original image, we measure the average token-level KL divergence between student predictive distributions under original and randomly masked images, keeping the question and response prefixes fixed:

$$
s ( x , y ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \bigl ( \pi _ { \theta } ( \cdot \mid c _ { t } ) \parallel \pi _ { \theta } ( \cdot \mid c _ { t } ^ { m } ) \bigr ) ,\tag{2}
$$

where $c _ { t } ^ { m }$ replaces the original image in $c _ { t }$ with its masked version and higher s indicates greater sensitivity to evidence removal. Within each benchmark and student scale, we divide examples into quartiles by their estimated scores. Figure 3 shows that higher-scoring groups generally achieve higher accuracy across both student scales.

Together, these findings motivate explicitly shaping how students respond to visual evidence during OPD. S-OPD augments teacher–student supervision with student-side objectives that encourage sensitivity to evidence removal and consistency under mild perturbations. To define these objectives, we introduce masked and noisy image views:

$$
{ \boldsymbol { I } } ^ { m } = { \mathcal { M } } ( { \boldsymbol { I } } ; { \boldsymbol { \rho } } ) , \qquad { \boldsymbol { I } } ^ { g } = \mathrm { c l i p } ( { \boldsymbol { I } } + { \boldsymbol { \epsilon } } ) , \quad { \boldsymbol { \epsilon } } \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } { \bf I } ) ,\tag{3}
$$

where M randomly masks a proportion $\rho$ of image patches, σ is the Gaussian noise standard deviation, I is the identity matrix, and clip restricts pixels to their valid range. Masking reduces available visual evidence, while mild noise is intended to preserve task content. The corresponding contexts, $c _ { t } ^ { m } = ( q , I ^ { m } , y _ { < t } )$ and $c _ { t } ^ { g } = ( q , I ^ { g } , y _ { < t } )$ , share the original response, enabling comparisons at the same position. We provide the visualization of masked and noisy images in Appendix A.1.

## 3.2 TEACHER-CALIBRATED POLICY CONTRAST

To enhance student visual perception, TPC encourages the student to use task-relevant visual evidence. Specifically, it increases the discrepancy between the student’s policies under original and masked images at tokens where the teacher indicates that the removed evidence is relevant.

To focus this contrast on tokens affected by masking, we select positions where the teacher assigns greater probability under the original image:

$$
\Delta _ { t } ^ { T } = \log \frac { \pi _ { T } ( y _ { t } \mid c _ { t } ) } { \pi _ { T } ( y _ { t } \mid c _ { t } ^ { m } ) } , \qquad g _ { t } = \mathbb { I } \left[ \Delta _ { t } ^ { T } > \tau \right] ,\tag{4}
$$

where τ specifies the required log-probability drop and is set to 0 by default. Despite using a zero margin, the teacher gate remains selective in practice, activating on only 46.9% of response tokens; detailed per-rollout statistics are provided in Appendix B.3. This gate uses the teacher’s response to masking as a token-level signal of visual dependence (Huang et al., 2026). At the selected positions, TPC maximizes the KL divergence between the student’s original and masked policies, weighted by the teacher’s log-probability drop:

$$
\mathcal { L } _ { \mathrm { T P C } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } , \boldsymbol { y } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \vert \boldsymbol { q } , \boldsymbol { I } ) } \left[ \frac { 1 } { Z } \sum _ { t = 1 } ^ { T } g _ { t } \Delta _ { t } ^ { T } D _ { \mathrm { K L } } \left( \pi _ { \boldsymbol { \theta } } ( \cdot \vert \boldsymbol { c } _ { t } ) \Vert \pi _ { \boldsymbol { \theta } } ( \cdot \vert \boldsymbol { c } _ { t } ^ { m } ) \right) \right] ,\tag{5}
$$

where $Z = \textstyle \operatorname* { m a x } ( 1 , \sum _ { t = 1 } ^ { T } g _ { t } )$ normalizes by the number of selected tokens. In practice, we use the sampled-token $k _ { 3 }$ estimator, $r _ { t } ^ { m } - \log r _ { t } ^ { \dot { m } } - 1$ , where $r _ { t } ^ { m } = \pi _ { \theta } ( y _ { t } \mid c _ { t } ^ { m } ) / \pi _ { \theta } ( y _ { t } \mid c _ { t } )$ , and stop gradients through the masked-image policy. Thus, the teacher determines where contrast is applied, while the student’s cross-image discrepancy provides the contrastive learning signal.

## 3.3 POLICY AGREEMENT

To make the student’s use of visual evidence robust to local image variations, PA encourages consistent predictions under mild Gaussian perturbations. These perturbations are intended to preserve task-relevant content, so the student’s predictions should remain stable despite changes in individ ual pixel values. Accordingly, PA minimizes the KL divergence between the student’s original and noisy policies at the same response prefixes:

$$
\mathcal { L } _ { \mathrm { P A } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot \vert q , I ) } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot \vert c _ { t } ) \Vert \pi _ { \theta } ( \cdot \vert c _ { t } ^ { g } ) \right) \right] .\tag{6}
$$

We implement the KL divergence using the sampled-token $k _ { 3 }$ estimator, $r _ { t } ^ { g } - \log r _ { t } ^ { g } - 1$ , where $r _ { t } ^ { g } = \hat { \pi _ { \theta } } ( y _ { t } \mid c _ { t } ^ { g } ) / \pi _ { \theta } ( y _ { t } \mid c _ { t } )$ , and stop gradients through the noisy-image policy. The noisy-image policy thus provides a fixed target for each update, encouraging the student to maintain consistent predictions under perturbations that preserve task-relevant visual evidence.

## 3.4 OVERALL OBJECTIVE

S-OPD combines teacher–student distillation with explicit objectives for visual perception:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S - O P D } } ( \theta ) = \mathcal { L } _ { \mathrm { O P D } } ( \theta ) + \lambda _ { \mathrm { T P C } } \mathcal { L } _ { \mathrm { T P C } } ( \theta ) + \lambda _ { \mathrm { P A } } \mathcal { L } _ { \mathrm { P A } } ( \theta ) , } \end{array}\tag{7}
$$

where $\lambda _ { \mathrm { T P C } }$ and $\lambda _ { \mathrm { P A } }$ control the strengths of policy contrast and agreement. Alongside token-level teacher guidance, these objectives train the student to respond to relevant evidence removal and remain stable under mild visual perturbations.

For each input, S-OPD samples only a single rollout under the original image. Notably, S-OPD can be seamlessly plugged into existing OPD frameworks, requiring no additional model parameters or inference operations. Its additional computation is confined to training, with a modest and practical overhead (see Appendix B.6).

## 4 EXPERIMENTS

Our experiments test whether strengthening student visual perception enables more effective use of teacher supervision in multimodal OPD. We address four questions. Q1: Does S-OPD improve multimodal OPD across student scales and teacher strengths? Q2: Are these gains sustained across various tasks beyond the training domain? Q3: How do TPC and PA contribute to these improvements, and does teacher calibration enhance the effectiveness of student policy contrast? Q4: What additional computational cost does S-OPD introduce?

Data. For the main experiments, we train on ViRL39K (Wang et al., 2025). We evaluate the models across eight benchmarks covering three capabilities: mathematical reasoning on MathVerse (Zhang et al., 2024), MathVista (Lu et al., 2024), MathVision (Wang et al., 2024), and WeMath (Qiao et al., 2024); logical reasoning on LogicVista (Xiao et al., 2024) and VisualPuzzles (Song et al., 2025); and general visual reasoning on ZeroBench-sub (Roberts et al., 2025) and MMMU-val (Yue et al., 2024). Geometry3K (Lu et al., 2021) is reserved for ablation studies due to its compact size and focused geometry domain.

Models and baselines. We use Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct as student models, with Qwen3-VL-8B-Instruct serving as the teacher (Bai et al., 2025). For each student size, we conduct on-policy distillation with either the original 8B teacher or an enhanced teacher obtained by training the 8B model with GRPO (Shao et al., 2024) on ViRL39K for one epoch. We compare our method against four baselines in the main experiments: the corresponding base model, GRPO, vanilla OPD (Lu & Lab, 2025), and VA-OPD (Liu et al., 2026). We also include OPSD (Zhao et al., 2026) as a baseline in Section 4.2. Details are in Appendix A.

## 4.1 MAIN RESULTS

Table 1 reports results across all benchmarks for both 2B and 4B students. We compare S-OPD against the corresponding distillation baseline under the same teacher configuration.

S-OPD benefits different student scales and teacher strengths. S-OPD improves average performance across all four student–teacher configurations. With the 8B teacher, the gains are larger for the 2B student than for the 4B student (+1.37 vs. +0.61 points), suggesting greater room for improvement through student-side learning at the smaller scale. Strengthening the teacher with GRPO raises the OPD baseline, yet S-OPD provides further gains of 0.92 and 0.82 points respectively. The gain decreases for the 2B student but increases for the 4B student, indicating that the benefit of student-side optimization depends on both student capacity and teacher strength.

Gains span reasoning domains but vary across tasks. S-OPD improves the average score within mathematical, logical, and general reasoning in every student–teacher configuration. The distribution of gains nevertheless differs across scales and teachers. With the 8B teacher, the 2B student improves on all eight benchmarks, including MathVista (+1.80) and ZeroBench (+2.70). With the GRPO-trained teacher, the most pronounced gain shifts to logical reasoning for the 4B student, which improves by 4.25 points on LogicVista, while its general reasoning performance changes little overall. These results suggest that student-side perceptual learning benefits diverse reasoning tasks, with the largest gains depending on the student–teacher configuration.

Student-side learning complements enhanced teacher supervision. Adding our objectives to VA-OPD improves average performance by 1.03 points for the 2B student and 0.72 points for the 4B student, with notable improvements including MathVision (+2.70) for the 2B student and MMMU val (+1.67) for the 4B student. These gains show that improving teacher supervision leaves additional opportunities on the student side: S-OPD remains effective when combined with a method that already enhances the supervision students receive.

## 4.2 ABLATION STUDIES

Component analysis. Table 2 isolates the contributions of TPC and PA across both 2B and 4B student scales, and shows that TPC and PA contribute differently to in-domain learning and out-ofdomain generalization. TPC provides stronger in-domain gains, improving Geometry3K accuracy by 1.63 and 2.26 points for the 2B and 4B students, respectively, compared with 1.00 and 1.27 points from PA. PA is particularly effective for transfer to general reasoning, achieving the highest scores at both scales, and also improves logical reasoning for the 2B student. These gains suggest that encouraging consistency under visual noise helps students generalize beyond the training domain. Combining both modules improves over OPD on Geometry3K and all three other categories at both scales, yielding average gains of 1.60 and 1.04 points. The results support their complementary roles: TPC strengthens learning from visual evidence within the training domain, while PA promotes perceptual stability that supports cross-domain transfer.

Table 1: Main results on eight benchmarks. We compare the distillation performance against the corresponding baseline. For OPD and S-OPD, we use either Qwen3-VL-8B-Instruct or its GRPOtrained variant as the teacher; the subscript G denotes the latter teacher setting. Avg. is computed over the eight benchmarks.
<table><tr><td></td><td colspan="4">Mathematical Reasoning</td><td colspan="2">Logical Reasoning</td><td colspan="2">General Reasoning</td><td>Overall</td></tr><tr><td>Method</td><td>MathVerse MathVista MathVision</td><td></td><td></td><td>WeMath</td><td>LogicVista VisualPuzzles ZeroBench</td><td></td><td></td><td>MMMU</td><td>Avg.</td></tr><tr><td colspan="10">Qwen3-VL-2B-Instruct</td></tr><tr><td>Base model</td><td>45.51</td><td>61.2</td><td>30.29</td><td>32.29</td><td>34.52</td><td>16.18</td><td>13.17</td><td>45.78</td><td>34.87</td></tr><tr><td>GRPO</td><td>48.43</td><td>62.3</td><td>31.62</td><td>36.57</td><td>42.73</td><td>23.16</td><td>14.67</td><td>49.67</td><td>38.64</td></tr><tr><td>OPD</td><td>46.41</td><td>65.2</td><td>34.35</td><td>36.67</td><td>36.24</td><td>12.16</td><td>9.28</td><td>48.44</td><td>36.09</td></tr><tr><td>S-OPD</td><td>47.81(+1.40)</td><td>67.0(+1.8)</td><td></td><td>35.93(+1.58) 37.43(+0.76) 37.65(+1.41)</td><td></td><td>12.67 (+0.51)</td><td></td><td></td><td>11.98(+2.70) 49.22(+0.78) 37.46(+1.37)</td></tr><tr><td>OPDG</td><td>50.68</td><td>67.6</td><td>35.95</td><td>39.24</td><td>39.49</td><td>14.89</td><td>12.57</td><td>47.33</td><td>38.47</td></tr><tr><td>S-OPDG</td><td>52.76(+2.08)</td><td>68.3(+0.7)</td><td>36.03(+0.08) 40.29(+1.05) 39.25(-0.24)</td><td></td><td></td><td>15.62(+0.73)</td><td></td><td>13.78(+1.21) 49.11(+1.78) 39.39(+0.92)</td><td></td></tr><tr><td>VA-OPD</td><td>45.48</td><td>63.7</td><td>31.77</td><td>33.14</td><td>36.91</td><td>10.79</td><td>11.38</td><td>47.56</td><td>35.09</td></tr><tr><td>w/ S-OPD 47.51 (+2.03)</td><td></td><td>64.3(+0.6)</td><td></td><td>34.47(+2.70) 34.19(+1.05) 38.03(+1.12)</td><td></td><td>11.56(+0.77)</td><td></td><td></td><td>10.78(-0.60) 48.11(+0.55) 36.12(+1.03)</td></tr><tr><td colspan="10">Qwen3-VL-4B-Instruct</td></tr><tr><td>Base model</td><td>42.28</td><td>73.8</td><td>49.31</td><td>55.43</td><td>54.81</td><td>25.43</td><td>15.57</td><td>61.56</td><td>47.27</td></tr><tr><td>GRPO</td><td>39.11</td><td>73.5</td><td>48.39</td><td>52.57</td><td>59.73</td><td>41.35</td><td>20.36</td><td>62.78</td><td>49.72</td></tr><tr><td>OPD</td><td>46.14</td><td>74.2</td><td>47.33</td><td>50.95</td><td>54.81</td><td>29.71</td><td>19.46</td><td>61.44</td><td>48.01</td></tr><tr><td>S-OPD</td><td>45.84(-0.30)</td><td>75.2(+1.0)</td><td></td><td>47.57(+0.24) 52.38(+1.43) 55.93(+1.12)</td><td></td><td>30.22(+0.51)</td><td></td><td></td><td>20.05(+0.59) 61.78(+0.34) 48.62(+0.61)</td></tr><tr><td>OPDG</td><td>46.60</td><td>76.6</td><td>48.55</td><td>55.71</td><td>54.14</td><td>33.99</td><td>20.36</td><td>64.33</td><td>50.04</td></tr><tr><td>S-OPDG</td><td>46.81(+0.21) 77.0(+0.4)</td><td></td><td></td><td></td><td></td><td>48.22(-0.33) 57.43(+1.72) 58.39(+4.25) 34.25(+0.26)</td><td></td><td></td><td>20.05(-0.31) 64.76(+0.43) 50.86(+0.82)</td></tr><tr><td>VA-OPD</td><td>44.01</td><td>72.6</td><td>46.45</td><td>51.05</td><td>52.80</td><td>30.39</td><td>14.67</td><td>60.11</td><td>46.51</td></tr><tr><td>w/S-OPD 44.49(+0.48) 73.5(+0.9) 46.81(+0.36) 51.52(+0.47) 53.47(+0.67) 30.99(+0.60)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>15.27(+0.60) 61.78(+1.67) 47.23(+0.72)</td><td></td><td></td></tr></table>

Table 2: Component analysis with 2B and 4B students on Geometry3K. Avg. denotes the arithmetic average over all benchmarks including the test split of Geometry3K.
<table><tr><td rowspan="2">Method</td><td colspan="5">2B Student</td><td colspan="5">4B Student</td></tr><tr><td></td><td>Geometry3K Mathematical</td><td>Logical</td><td>General</td><td>Avg.</td><td></td><td>Geometry3K Mathematical</td><td>Logical</td><td>General</td><td>Avg.</td></tr><tr><td>OPD</td><td>40.93</td><td>45.98</td><td>25.83</td><td>30.40</td><td>37.48</td><td>46.76</td><td>54.38</td><td>42.89</td><td>36.76</td><td>47.06</td></tr><tr><td>w/ TPC</td><td>42.56</td><td>47.10</td><td>26.05</td><td>29.77</td><td>38.07</td><td>49.02</td><td>54.94</td><td>42.88</td><td>37.99</td><td>47.83</td></tr><tr><td>w/PA</td><td>41.93</td><td>46.78</td><td>27.91</td><td>31.27</td><td>38.60</td><td>48.03</td><td>54.42</td><td>41.46</td><td>39.06</td><td>47.41</td></tr><tr><td>S-OPD</td><td>44.26</td><td>47.39</td><td>28.02</td><td>30.92</td><td>39.08</td><td>49.02</td><td>55.03</td><td>43.33</td><td>38.57</td><td>48.10</td></tr></table>

Effectiveness of teacher-calibrated gating. We examine whether the teacher provides useful guidance on where to apply policy contrast (PC). As shown in Table 3, teacher-calibrated gating outperforms studentbased gating and no gating by 1.15 and 1.02 average points, respectively. More importantly, comparing these results with the PA-only variant in Table 2 reveals that TPC is not inherently beneficial: applying PC to all tokens reduces the average score from 38.60 to 38.06. This comparison shows that teacher calibration is critical for preventing indiscriminate contrastive supervision from harming the student policy. Student-based gating offers no improvement over applying TPC to all tokens, suggesting that the student’s

![](images/b27c3f8e954c28d34e881ff5315ad853f74b621bfeb2af25abe01c3e9423b1c3.jpg)  
Figure 5: S-OPD gains across student scales and distillation paradigms.

own visual sensitivity is insufficient to identify where policy contrast is beneficial. These results support using the teacher already available in OPD to guide the student’s perceptual learning beyond providing distillation targets. To test whether the gains simply arise from selecting fewer tokens, we randomly select the same number of tokens as the teacher gate for each response and reassign the original teacher-derived weights to these randomly selected tokens. This matched random baseline falls short of teacher-calibrated gating by 0.69 average points, indicating that the teacher identifies informative tokens for TPC.

Table 3: Comparison of gating strategies for TPC with the remaining S-OPD configuration fixed. OPD is included as a reference. Random gating selects the same number of tokens as teachercalibrated gating for each response. Bold values indicate the best results.
<table><tr><td>Gating strategy</td><td>Token selection</td><td>Geometry3K</td><td>Mathematical</td><td>Logical</td><td>General</td><td>Avg.</td></tr><tr><td>OPD (reference)</td><td>一</td><td>40.93</td><td>45.98</td><td>25.83</td><td>30.40</td><td>37.48</td></tr><tr><td>No gating</td><td>All</td><td>42.56</td><td>46.37</td><td>26.55</td><td>30.72</td><td>38.06</td></tr><tr><td>Student-based</td><td>Adaptive</td><td>41.43</td><td>46.80</td><td>25.44</td><td>30.92</td><td>37.93</td></tr><tr><td>Random</td><td>Count-matched</td><td>40.61</td><td>47.47</td><td>26.57</td><td>30.95</td><td>38.39</td></tr><tr><td>Teacher-calibrated (Ours)</td><td>Adaptive</td><td>44.26</td><td>47.39</td><td>28.02</td><td>30.92</td><td>39.08</td></tr></table>

Compatibility across distillation paradigms and student scales. Figure 5 evaluates S-OPD across different student scales and teacher advantage sources trained on Geometry3K. Standard OPD obtains supervision from a larger teacher, whereas OPSD (Zhao et al., 2026) uses privileged information to strengthen supervision in a self-teacher setting. S-OPD improves both paradigms: under standard OPD, it increases the average performance of the 2B and 4B students by 1.60 and 1.04 points, respectively; under OPSD, the corresponding gains are 0.71 and 0.66 points. Larger gains under standard OPD suggest that student-side optimization offers more room for improvement in this setting, while the positive gains under OPSD show that it remains effective when teacher advantage comes from additional information rather than greater model capacity. Together, these results demonstrate the applicability of S-OPD across different student sizes and forms of teacher supervision, with the extent of its benefit depending on the distillation setting.

Impact of the coefficients of TPC and PA. We examine whether S-OPD requires precise coefficient tuning by varying λ and $\lambda _ { \mathrm { P A } }$ for students trained on Geometry3K. As shown in Table 4, average performance varies by only 0.74 points for TPC and 0.50 points for PA. Although individual domains favor different coefficients, the overall performance remains stable, allowing us to obtain comparable results across a range of objective weights. We obtain the highest average score with $\bar { \lambda _ { \mathrm { T P C } } } ~ = ~ 0 . 0 0 5$ and $\lambda _ { \mathrm { P A } } = 0 . 0 2$ on Geometry3K. We set both coefficients to 0.02 for the main experiments on ViR

Table 4: Ablation of the TPC and PA coefficients.
<table><tr><td>Value</td><td>Geometry3K</td><td>Mathematical</td><td>Logical</td><td>General</td><td>Avg.</td></tr><tr><td colspan="6"> $\lambda _ { \mathrm { T P C } }$ </td></tr><tr><td>0.005</td><td>44.26</td><td>47.39</td><td>28.02</td><td>30.92</td><td>39.08</td></tr><tr><td>0.01</td><td>44.92</td><td>47.73</td><td>25.83</td><td>30.93</td><td>38.82</td></tr><tr><td>0.02</td><td>43.05</td><td>47.00</td><td>26.64</td><td>30.36</td><td>38.34</td></tr><tr><td colspan="6"> $\lambda _ { \mathrm { P A } }$ </td></tr><tr><td>0.01</td><td>44.92</td><td>46.85</td><td>26.55</td><td>30.91</td><td>38.58</td></tr><tr><td>0.02</td><td>44.26</td><td>47.39</td><td>28.02</td><td>30.92</td><td>39.08</td></tr><tr><td>0.04</td><td>44.09</td><td>47.12</td><td>26.13</td><td>31.38</td><td>38.62</td></tr></table>

efficients to 0.02 for the main experiments on ViRL39K and the corresponding ablations are in Appendix B.4.

Impact of the masking ratio and Gaussian noise level. We vary the masking ratio and Gaussian noise level to identify effective perturbation strengths for TPC and PA. In Table 5, increasing ρ from 0.2 to 0.6 raises the average score from 38.49 to 39.08, but further increasing it to 0.8 lowers the score to 38.74. Similarly, Gaussian noise performs best at $\sigma \ : = \ : 0 . 2$ , with weaker and stronger noise yielding lower average scores. We thus obtain the best overall results with substantial masking for TPC and moderate noise for PA. The preferred strength varies by domain, suggesting that different tasks

Table 5: Ablation of the masking ratio and Gaussian noise level.
<table><tr><td rowspan=1 colspan=6>ValueGeometry3K Mathematical LogicalGeneral  $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td rowspan=1 colspan=6>Masking ratio ρ</td></tr><tr><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>43.43</td><td rowspan=1 colspan=1>47.11</td><td rowspan=1 colspan=1>24.94</td><td rowspan=1 colspan=1>32.34</td><td rowspan=1 colspan=1>38.49</td></tr><tr><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>42.43</td><td rowspan=1 colspan=1>47.18</td><td rowspan=1 colspan=1>26.65</td><td rowspan=1 colspan=1>31.20</td><td rowspan=1 colspan=1>38.54</td></tr><tr><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>44.26</td><td rowspan=1 colspan=1>47.39</td><td rowspan=1 colspan=1>28.02</td><td rowspan=1 colspan=1>30.92</td><td rowspan=1 colspan=1>39.08</td></tr><tr><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>44.43</td><td rowspan=1 colspan=1>47.34</td><td rowspan=1 colspan=1>26.02</td><td rowspan=1 colspan=1>31.42</td><td rowspan=1 colspan=1>38.74</td></tr><tr><td rowspan=1 colspan=6>Gaussian noise level σ</td></tr><tr><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>42.93</td><td rowspan=1 colspan=1>46.85</td><td rowspan=1 colspan=1>26.55</td><td rowspan=1 colspan=1>30.91</td><td rowspan=1 colspan=1>38.36</td></tr><tr><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>44.26</td><td rowspan=1 colspan=1>47.39</td><td rowspan=1 colspan=1>28.02</td><td rowspan=1 colspan=1>30.92</td><td rowspan=1 colspan=1>39.08</td></tr><tr><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>43.05</td><td rowspan=1 colspan=1>47.12</td><td rowspan=1 colspan=1>26.13</td><td rowspan=1 colspan=1>31.38</td><td rowspan=1 colspan=1>38.51</td></tr></table>

may place distinct demands on visual perception. We choose $\rho = 0 . 6$ and $\sigma = 0 . 2$ for both Geometry3K ablations and ViRL39K main experiments.

Computational overhead analysis. S-OPD improves reasoning performance without adding trainable parameters, extra autoregressive rollouts, or inference overhead. Its additional computation is confined to training, where teacher gating and the auxiliary student objectives evaluate perturbed image views using the same generated responses. Under identical hardware configura tions, these operations increase training time per step by 28.6% for the 2B student on ViRL39K and 16.0% for the 4B student on Geometry3K, at a moderate cost while preserving deployment efficiency. Appendix B.6 provides detailed measurements.

## 5 CONCLUSION

In this paper, we identify student visual perception as a persistent bottleneck in multimodal onpolicy distillation, even when teacher supervision is strengthened. Motivated by this, we propose S-OPD to explicitly improve student perceptual sensitivity and robustness within the token-level distillation process. Teacher-calibrated Policy Contrast encourages sensitivity to visual evidence removal at teacher-selected tokens, while Policy Agreement promotes consistent predictions under visual noise. Experiments on eight benchmarks demonstrate consistent average improvements across student scales and teacher strengths, with further gains when integrated into existing methods that enhance teacher supervision. These benefits require no additional annotations, model parameters, or inference-time operations, with only moderate training overhead. Our findings establish student-side learning as a complementary direction for advancing multimodal OPD.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024.

Aniri, Jinhe Bi, Peng Liao, Zengjie Jin, Volker Tresp, Fei Shen, Yunpu Ma, Tat-Seng Chua, et al. OPD-V: Visual on-policy self-distillation with modality balance. arXiv preprint arXiv:2608.05131, 2026.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Walid Bousselham, Hilde Kuehne, and Cordelia Schmid. VOLD: Reasoning transfer from llms to vision-language models via on-policy distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26209–26218, 2026.

Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Yufeng Zhong, and Lin Ma. Chart-r1: Chain-of-thought supervision and reinforcement for advanced chart reasoner. arXiv preprint arXiv:2507.15509, 2025.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, 2024.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. Advances in Neural Information Processing Systems, 2025.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Siyuan Huang, Xiaoye Qu, Yafu Li, Yun Luo, Zefeng He, Daizong Liu, and Yu Cheng. Spotlight on token perception for multimodal reinforcement learning. In International Conference on Learning Representations, 2026.

Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggyu Lim, and Taesup Kim. Stable on-policy distillation through adaptive target reformulation. In Findings of the Association for Computational Linguistics: ACL 2026, 2026.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. In International Conference on Machine Learning, 2026.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, 2016.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huanang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026a.

Yijiang Li, Yijun Liang, Yunjie Tian, Bingyang Wang, Ke Zhang, Zhenfei Yin, Di Fu, Philip Torr, and Nuno Vasconcelos. Self-supervised visual on-policy distillation. arXiv preprint arXiv:2608.14144, 2026b.

Yijun Liang, Yunjie Tian, Yijiang Li, Yuqi Jia, Furong Huang, Tianyi Zhou, and Di Fu. Visual contrastive self-distillation. arXiv preprint arXiv:2607.21556, 2026.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Ruiqi Liu, Xiaolei Lv, Gengsheng Li, Ximo Zhu, Zhiheng Wang, Zhengbo Zhang, Junkai Chen, Zhiheng Li, Bo Li, Jun Gao, et al. Visual-advantage on-policy distillation for vision-language models. arXiv preprint arXiv:2605.21924, 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/on-policy-distillation.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024.

Qinghe Ma, Zhen Zhao, Yiming Wu, Jian Zhang, Lei Bai, and Yinghuan Shi. Are tools always beneficial? learning to invoke tools adaptively for dual-mode multimodal llm reasoning. arXiv preprint arXiv:2605.19852, 2026.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the associationfor computational linguistics, 2022.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, et al. We-Math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

Jonathan Roberts, Mohammad Reza Taesiri, Ansh Sharma, Akash Gupta, Samuel Roberts, Ioana Croitoru, Simion-Vlad Bogolin, Jialu Tang, Florian Langer, Vyas Raina, et al. ZeroBench: An impossible visual benchmark for contemporary large multimodal models. arXiv preprint arXiv:2502.09696, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, 2025.

Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. VisualPuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. arXiv preprint arXiv:2504.10342, 2025.

Haoxiang Sun, Zhihang Yi, Langxuan Deng, Yuhao Zhou, Peiqi Jia, Jian Zhao, Li Yuan, Jiancheng Lv, and Tao Wang. V-Zero: Answer-label-free on-policy distillation with contrastive evidence gating for fine-grained visual reasoning. arXiv preprint arXiv:2606.25319, 2026.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems, volume 30, 2017.

Kanghui Tian, Siyuan Liu, Tianxiang Jiang, Shuai Dong, Yizhuo Li, Tian Ding, Yuan Guo, Songze Li, Haowen Hou, Congcong Wang, and Yi Wang. What should a self-teacher see? privileged context design for on-policy self-distillation. arXiv preprint arXiv:2609.25623, 2026a.

Kanghui Tian, Siyuan Liu, Ziang Yan, Sheng Xia, Shuai Dong, and Yi Wang. Vicur: Visual cues as recoverable privilege for multimodal on-policy distillation. arXiv preprint arXiv:2606.05718, 2026b.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In IEEE/CVF Conference on Com puter Vision and Pattern Recognition, 2024.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. VL-Rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In Advances in Neural Information Processing Systems, 2024.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, and Heng Ji. Perception-aware policy optimization for multimodal reasoning. In International Conference on Learning Representations, 2026.

Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, et al. Zooming without zooming: Region-to-image distillation for fine-grained multimodal perception. arXiv preprint arXiv:2602.11858, 2026.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. LogicVista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Yuxi Xie, Guanzhen Li, Xiao Xu, and Min-Yen Kan. V-dpo: Mitigating hallucination in large vision language models via vision-guided direct preference optimization. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pp. 13258–13273, 2024.

Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. TIP: Token importance in on-policy distillation. arXiv preprint arXiv:2604.14084, 2026.

Qixiang Yin, Huanjin Yao, Yuchen Cai, Jianghao Chen, Ziyi Wang, Min Yang, Fei Su, and Zhicheng Zhao. H-OPD: Confidence aware heterogeneous multi-teacher multimodal on-policy distillation. arXiv preprint arXiv:2607.02592, 2026.

Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Vision-opd: Learning to see fine details for multimodal llms via on-policy self-distillation. arXiv preprint arXiv:2605.18740, 2026.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Kangning Zhang, Yixing Li, Shuai Shao, Qingyao Li, Zhengxi Lu, Zhiyuan Yao, Jianghao Lin, Wenxiang Jiao, Yuan Lu, Weiwen Liu, et al. VAD: Attributing visual evidence for target reconstruction in multimodal on-policy distillation. arXiv preprint arXiv:2607.28590, 2026.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. MathVerse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, 2024.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Stephan Zheng, Yang Song, Thomas Leung, and Ian Goodfellow. Improving the robustness of deep neural networks via stability training. In Proceedings of the ieee conference on computer vision and pattern recognition, pp. 4480–4488, 2016.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## A IMPLEMENTATION DETAILS

## A.1 TRAINING DETAILS

All the experiments are implemented on top of the verl framework (Sheng et al., 2025) and run on NVIDIA A100 80GB GPUs. We train the student model with PyTorch Fully Sharded Data Parallel (FSDP) (Zhao et al., 2023) and use vLLM (Kwon et al., 2023) for on-policy rollout generation and frozen-teacher forward passes. Table 6 summarizes the key training configurations for the main ViRL39K experiments and the Geometry3K ablations.

Masking ratio ρ = 0.8  
Table 6: Key training configurations for the ViRL39K experiments and the Geometry3K ablations.
<table><tr><td>Setting</td><td>ViRL39K Distillation</td><td>ViRL39K GRPO</td><td>Geometry3K Distillation</td></tr><tr><td>Training data</td><td>ViRL39K</td><td>ViRL39K</td><td>Geometry3K train</td></tr><tr><td>Max. prompt length</td><td>4,096</td><td>4,096</td><td>1,024</td></tr><tr><td>Max. response length</td><td>4,096</td><td>4096</td><td>2,048</td></tr><tr><td>Prompt batch size</td><td>192</td><td>192</td><td>128</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Rollouts per prompt (n)</td><td>1</td><td>5</td><td>1</td></tr><tr><td>Training epochs</td><td>1</td><td>1</td><td>20</td></tr><tr><td>λTPC</td><td>0.02</td><td></td><td>0.005</td></tr><tr><td> $\lambda _ { \mathrm { P A } }$ </td><td>0.02</td><td></td><td>0.02</td></tr></table>

Details of masking. We construct the masked image in RGB pixel space before applying the model-specific image processor. Specifically, we partition the original-resolution image into nonoverlapping $1 6 \times 1 6$ pixel blocks, including partial blocks along the image boundaries, and independently sample a mask indicator $m _ { i j } \sim$ Bernoulli(0.6) for each block. If $m _ { i j } = 1$ , all pixels in that block are replaced with black pixels, i.e., RGB (0, 0, 0). Thus, 0.6 denotes the expected masking ratio rather than an exact per-image proportion. We modify only the pixel values and preserve the image dimensions; no image patches or visual tokens are removed. Consequently, the original and masked images have the same visual-token sequence length and two-dimensional positional structure after preprocessing. Figure 6 visualizes images under different masking ratios.

![](images/59d6cc8317f3d417bd3ca8e5168a1b8622179aadf83ae780aee74bd51a3f1954.jpg)  
Masking ratio ρ = 0.2

![](images/db89364dba540f2445fa10401693681f102fc7b56316c60e7881be29f8510cff.jpg)  
Masking ratio ρ = 0.4

![](images/183d7aa1d59a56efee0c72372a144333d92b57a8bf3e62fc98b68c5362ff6ec4.jpg)  
Masking ratio ρ = 0.6

![](images/ae21bd04cdc1e61ef3ee65f7b1223b2db396c4ca3ae79415f98243278f89a32c.jpg)  
Figure 6: Masked images used for TPC at different masking ratios $\rho .$ Randomly selected image patches are replaced with black pixels, removing increasing amounts of visual evidence as $\rho$ increases.

Details of Gaussian perturbation. We construct the noisy image by adding Gaussian noise before applying the model-specific image processor. Given an unsigned 8-bit RGB image $v ,$ we first rescale its pixel values from [0, 255] to [0, 1] and independently sample $\epsilon _ { h w c } \sim \mathcal { N } ( 0 , \breve { \sigma } ^ { 2 } )$ for every spatial location and color channel, with $\sigma = 0 . 2$ . The perturbed image is computed as

$$
v _ { \mathrm { n } } = { \mathrm { r o u n d } } \left[ 2 5 5 \cdot \mathrm { c l i p } \left( { \frac { v } { 2 5 5 } } + \epsilon , 0 , 1 \right) \right] ,\tag{8}
$$

and is converted back to an unsigned 8-bit RGB image before standard visual preprocessing. Therefore, the reported noise scale is defined with respect to the [0, 1] pixel range; equivalently, $\sigma = 0 . 2$ corresponds to approximately 51 intensity levels on the [0, 255] scale. Figure 7 visualizes images under different Gaussian perturbation noise.

## A.2 EVALUATION DETAILS

All the checkpoints are evaluated on MathVerse\_MINI, MathVista\_MINI, MathVision, WeMath, LogicVista, VisualPuzzles, ZeroBench\_sub, MMMU\_DEV\_VAL with greedy decoding using the default VLMEvalKit (Duan et al., 2024) dataset definitions and a common Qwen3-VL wrapper in the main experiments. We use the shared system prompt You are a helpful assistant. and dataset-specific user prompts listed in Table 7.

![](images/4db77f845a582c0e4d974f3b0e6da02a91d20b0bdcd073c8436fe66d69d943f7.jpg)  
Figure 7: Noisy images used for PA at different Gaussian noise levels σ. Independent zero-mean Gaussian noise is added to each RGB channel in the [0, 1] pixel range, followed by clipping. Larger σ produces stronger perturbations while retaining the overall image structure.

Table 7: Generation user prompts used for each evaluation dataset. Braced terms denote fields from each VLMEvalKit example, and [Options] denotes the formatted list of answer choices.  
Dataset User prompt template   
MathVerse MINI {question}   
MathVista MINI {question}   
MathVision {question}   
WeMath Question: {question}   
Options: [Options]   
LogicVista {question}   
VisualPuzzles {question}   
Options: [Options]   
Solve the multiple-choice question and then answer with   
the option letter from the given choices. The last   
line of your response should be of the following   
format: ’Answer: \$LETTER’ (without quotes), where   
LETTER is one of the options. Think step by step   
before answering.   
ZEROBench sub {question}   
Let’s think step by step and give the final answer in   
curly braces, like this: {final answer}.   
MMMU DEV VAL Question: {question}   
Options: [Options]   
Please select the correct answer from the options   
above.

Rule-based benchmarks use their official exact-match or multiple-choice scorers. Benchmarks requiring answer extraction or semantic grading use a locally served Qwen3-30B-A3B. MathVerse is averaged over its Vision Intensive, Vision Only, Text Dominant, Text Lite, and Vision Dominant subsets; the text-only subset is excluded. We report MathVista on its 1,000-example mini split, WeMath using Core (Strict), and MMMU on the validation split.

## A.3 ORACLE VISUAL-FACT INTERVENTION DETAILS

Oracle-fact construction. We use oracle visual facts as a test-time intervention to make the geometric information encoded in each diagram explicit. For every Geometry3K example, we deterministically convert diagram logic form from the official logic form.json annotations into natural-language statements describing geometric entities, incidence relations, lengths, angles, parallelism, perpendicularity, equality, congruence, and other geometric properties. The renderer has no access to the answer, solution, program, theorem sequence, or text-based logic forms. It also rejects Find and UseTheorem expressions, including nested occurrences. We compare two paired inputs that share the same image, question, answer choices, system prompt, and output instruction. The oracle condition differs only by the insertion of the rendered statements under Diagram observations: in the same user message. Thus, the intervention augments the diagram infor mation available to the model without changing its parameters or removing the original image.

![](images/ab06613a4ce875fd1509b1e948b44bbd818e106984b4b2afc6f9cca22d641a7f.jpg)  
Figure 8: Oracle visual-fact intervention on Geometry3K. Top panels report accuracy with the original image alone (w/o visualfacts) and with additional oracle visual facts derived from diagram annotations (w/ visual facts); bottom panels show the corresponding accuracy gap in percentage points. Compared with standard OPD, S-OPD improves accuracy without visual facts and reduces the accuracy gap from 7.8 to 2.3 points for the 2B student and from 15.1 to 12.1 points for the 4B student.

Evaluation protocol. We evaluate every checkpoint on the same official 601-example test split of Geometry3K. Inference uses greedy decoding (temperature 0) and a maximum generation length of 4096 tokens. We report answer-content accuracy after mapping predicted option letters to their corresponding choice values and normalizing common LAT X and numeric variants. The Accuracy Gap is the paired difference between accuracy with oracle visual facts and image-only accuracy.

Results. We revisit the oracle visual-fact intervention from our motivational study to examine whether S-OPD alleviates the student-side perceptual bottleneck that motivated our method. As shown in Figure 8, compared with standard OPD, S-OPD improves image-only accuracy and reduces the Accuracy Gap at both model scales. For the 2B student, image-only accuracy increases from 34.4% to 44.1%, while accuracy with oracle visual facts improves from 42.3% to 46.4%, narrowing the gap from 7.8 to 2.3 percentage points. S-OPD thus achieves the highest accuracy in both conditions and the smallest gap among the compared methods at this scale. This result directly complements our motivational finding: whereas stronger teacher-side supervision alone leaves a substantial gap, explicitly strengthening student perception helps the student recover more task-relevant evidence from the diagram itself. For the 4B student, S-OPD improves image-only accuracy from 50.7% to 52.1% and reduces the gap from 15.1 to 12.1 points. Here, the evidence is more qualified, as oracle-fact accuracy also decreases from 65.9% to 64.2%, and OPD with the GRPO teacher retains the smallest gap. Overall, these results, particularly the strong improvement for the 2B student, support our central motivation that student-side perceptual learning is an important complement to teacher-side supervision in multimodal OPD.

Audit. Our data audit finds no unsupported expressions, parse failures, missing images, missing answers, or use of forbidden annotation sources in the rendered facts.

## B ABLATION DETAILS

## B.1 VISUAL PERCEPTION BEYOND REASONING BENCHMARKS

To further examine whether S-OPD improves student visual perception, we evaluate the distilled students on three additional benchmarks, V<sup>∗</sup>Bench (Wu & Xie, 2024), ZoomBench (Wei et al., 2026) and HallusionBench (Guan et al., 2024). Specifically, V<sup>∗</sup>Bench and ZoomBench assess finegrained visual understanding, while HallusionBench explicitly diagnoses failures arising from visual illusions and language hallucinations. As shown in Table 8, S-OPD improves or matches OPD in 10 out of the 12 model–data configurations.

Table 8: Comparison of OPD and S-OPD across different student model sizes and training datasets. All models are evaluated on three out-of-domain visual benchmarks.
<table><tr><td rowspan="2">Benchmark</td><td colspan="4">Trained on Geometry3K</td><td colspan="4">Trained on ViRL39K</td></tr><tr><td colspan="2">2B</td><td colspan="2">4B</td><td colspan="2">2B</td><td colspan="2">4B</td></tr><tr><td></td><td>OPD</td><td>S-OPD</td><td>OPD</td><td>S-OPD</td><td>OPD</td><td>S-OPD</td><td>OPD</td><td>S-OPD</td></tr><tr><td>V*Bench</td><td>73.30</td><td>74.35</td><td>74.87</td><td>74.35</td><td>74.87</td><td>75.92</td><td>62.67</td><td>63.30</td></tr><tr><td>ZoomBench</td><td>41.42</td><td>42.37</td><td>43.55</td><td>44.26</td><td>41.30</td><td>40.95</td><td>43.67</td><td>44.14</td></tr><tr><td>HallusionBench</td><td>55.73</td><td>56.26</td><td>64.77</td><td>64.98</td><td>52.58</td><td>53.00</td><td>74.35</td><td>74.35</td></tr></table>

Table 9: Detailed component analysis results on Geometry3K. The Geometry3K test split is used for in-domain performance evaluation. Bold and underlined numbers indicate the best and second-best results within each student size, respectively.
<table><tr><td></td><td>In-domain</td><td colspan="4">Mathematical Reasoning</td><td colspan="2">Logical Reasoning</td><td colspan="2">General Reasoning</td><td>Overall</td></tr><tr><td>Method</td><td>Geo3K-Test</td><td>MVerse</td><td></td><td>MVista MVision</td><td>WeMath</td><td>LogicVista</td><td>VisPuzzles</td><td>ZeroBench MMMU-val</td><td></td><td>Avg.</td></tr><tr><td colspan="9">Qwen3-VL-2B-Instruct</td><td></td></tr><tr><td>Vanilla OPD</td><td>40.93</td><td>49.28</td><td>63.3</td><td>34.96</td><td>36.38</td><td>39.83</td><td>11.82</td><td>11.68</td><td>49.11</td><td>37.48</td></tr><tr><td>w/ TPC</td><td>42.56</td><td>49.84</td><td>65.2</td><td>35.07</td><td>38.29</td><td>40.71</td><td>11.39</td><td>11.98</td><td>47.56</td><td>38.07</td></tr><tr><td>w/PA</td><td>41.93</td><td>50.69</td><td>65.3</td><td>34.36</td><td>36.76</td><td>43.14</td><td>12.67</td><td>14.97</td><td>47.56</td><td>38.60</td></tr><tr><td>S-OPD</td><td>44.26</td><td>50.76</td><td>65.3</td><td>35.43</td><td>38.05</td><td>42.51</td><td>13.53</td><td>13.17</td><td>48.67</td><td>39.08</td></tr><tr><td colspan="10">Qwen3-VL-4B-Instruct</td></tr><tr><td>Vanilla OPD</td><td>46.76</td><td>44.24</td><td>74.1</td><td>46.68</td><td>52.48</td><td>55.81</td><td>29.97</td><td>14.07</td><td>59.44</td><td>47.06</td></tr><tr><td>w/ TPC</td><td>49.02</td><td>45.14</td><td>74.4</td><td>47.07</td><td>53.14</td><td>55.70</td><td>30.05</td><td>15.87</td><td>60.11</td><td>47.83</td></tr><tr><td>w/PA</td><td>48.03</td><td>43.83</td><td>74.1</td><td>47.63</td><td>52.10</td><td>55.26</td><td>27.65</td><td>15.57</td><td>62.56</td><td>47.41</td></tr><tr><td>S-OPD</td><td>49.02</td><td>45.38</td><td>74.4</td><td>47.20</td><td>53.14</td><td>56.60</td><td>30.05</td><td>16.47</td><td>60.67</td><td>48.10</td></tr></table>

The improvements are consistent across different student capacities and training distributions. On V<sup>∗</sup>Bench, S-OPD improves the 2B student by 1.05 points when trained on either Geometry3K or ViRL39K, and improves the 4B student trained on ViRL39K by 0.63 points. On ZoomBench, it yields gains of 0.95 and 0.71 points for the 2B and 4B students trained on Geometry3K, respectively, while improving the ViRL39K-trained 4B student by 0.47 points. Importantly, S-OPD also consistently improves or preserves performance on HallusionBench across all four settings, with gains of up to 0.53 points. Although a few configurations exhibit small fluctuations, the overall trend across model sizes, training datasets, and diagnostic tasks indicates that the benefits of S-OPD extend beyond the in-domain reasoning benchmarks. In particular, the gains on fine-grained perception and hallucination-oriented evaluations provide additional evidence that S-OPD encourages students to ground their predictions more effectively in visual evidence.

## B.2 MAIN COMPONENT ANALYSIS

Table 9 examines TPC and PA across both student scales. TPC improves in-domain accuracy and all four mathematical reasoning benchmarks for both students, consistent with its role in linking predictions to supporting visual evidence through teacher-calibrated policy contrast. PA yields more varied gains, with notable improvements on LogicVista and ZeroBench for the 2B student and MMMU for the 4B student. This pattern suggests that agreement under mild visual perturbations can support generalization beyond the geometric training domain, although its benefits depend on the task and student scale.

Combining both objectives achieves the highest overall average at both scales, improving over vanilla OPD by 1.60 and 1.04 points for the 2B and 4B students, respectively. Gains extend to both in-domain accuracy and out-of-domain average performance. While the combination does not preserve every component’s best individual score, its stronger aggregate results support the complementary roles of sensitivity to evidence removal and stability under mild visual variation in student learning.

Table 10: Ablation of the TPC and PA coefficients on ViRL39K.
<table><tr><td>Value</td><td>Mathematical</td><td>Logical</td><td>General</td><td>Avg.</td></tr><tr><td colspan="5"> $\lambda _ { \mathrm { T P C } }$ </td></tr><tr><td>0.005</td><td>47.07</td><td>25.09</td><td>29.28</td><td>37.13</td></tr><tr><td>0.01</td><td>46.20</td><td>25.05</td><td>28.71</td><td>36.54</td></tr><tr><td>0.02</td><td>47.04</td><td>25.16</td><td>30.60</td><td>37.46</td></tr><tr><td colspan="5">λPA</td></tr><tr><td>0.01</td><td>46.41</td><td>24.37</td><td>28.57</td><td>36.44</td></tr><tr><td>0.02</td><td>47.04</td><td>25.16</td><td>30.60</td><td>37.46</td></tr><tr><td>0.04</td><td>47.15</td><td>24.84</td><td>30.44</td><td>37.39</td></tr></table>

## B.3 EMPIRICAL SPARSITY OF THE TEACHER GATE

We examine whether the zero-margin criterion (τ = 0) makes the teacher gate overly dense. In a representative S2B–T8B training run with 60% random patch masking, we analyze 3,840 logged rollouts containing 8,649,621 valid response tokens. The gate retains 4,054,346 tokens, corresponding to a token-weighted activation rate of 46.87%. When each rollout is weighted equally, the mean and median activation rates are 48.29% and 48.39%, respectively, with an interquartile range of 43.50–53.48% and a 5th–95th percentile range of 32.36–65.64%. These results show that masking does not induce a uniformly positive teacher likelihood drop: even with $\tau = 0 ,$ , more than half of the response tokens are filtered out on average.

## B.4 IMPACT OF THE COEFFICIENTS OF TPC AND PA

We extend the coefficient ablations on Geometry3K in the main text to ViRL39K. As shown in Table 10, average performance varies by 0.92 points across the tested TPC coefficients and 1.02 points across the PA coefficients, with all settings outperforming the OPD baseline of 36.09. Both coefficients achieve the highest average score at 0.02, while increasing $\lambda _ { \mathrm { P A } }$ to 0.04 yields nearly identical performance (37.39 vs. 37.46). Together with the Geometry3K results, these findings show that S-OPD remains effective across a range of coefficient choices on both training datasets. We set $\lambda _ { \mathrm { T P C } } = \lambda _ { \mathrm { P A } } = 0 . 0 2$ by default in the main experiments on ViRL39K, as this configuration achieves the highest average score.

## B.5 TOKEN-LEVEL LOG-PROBABILITY ANALYSIS

Full-rollout log-probability analysis. Figure 9 compares the teacher–student token-level logprobability gap over complete rollouts on ViRL39K and Geometry3K. S-OPD exhibits similar training dynamics to OPD and reaches comparable final gaps at both student scales, with smaller gaps during parts of training for the 4B student, particularly on Geometry3K. Since this metric averages over initial planning, intermediate reasoning, and final-answer generation, it can obscure change within individual stages. Together with the accuracy gains and the reductions in middle-segment gaps analyzed below, these results suggest that S-OPD preserves overall teacher alignment while improving learning within the reasoning process, consistent with its goal of strengthening the stu dent’s use of visual evidence.

Segment-level log-probability analysis. We divide each rollout into four equal-length segments, Q1–Q4, to examine alignment throughout the response. A typical solution roughly progresses from establishing an approach, through inspecting visual evidence and deriving intermediate results, to producing the final answer, although these stages need not align exactly with segment boundaries. Under this interpretation, improvements in the middle segments are particularly relevant to learning how to connect visual evidence with reasoning. S-OPD maintains a smaller Q2 gap through much of training for the 4B student on Geometry3K and reduces the Q3 gap during later training for the 2B student on ViRL39K. Both 4B settings also show smaller Q3 gaps during portions of training. This pattern suggests that S-OPD benefits intermediate solution construction rather than concentrating its effect on final-answer prediction. Together with the comparable full-rollout gaps and improved accuracy, these observations are consistent with the intended mechanism: visual contrast and alignment help students use teacher supervision more effectively within the reasoning process.

![](images/007ca97534d6ea88b884e831176136e4bfa2d3110a20b11a55259fe790742024.jpg)  
Figure 9: Token-level log-probability analysis of 2B and 4B students on ViRL39K and Geometry3K during training. We report the teacher–student token-level log-probability gap under different teacher–student configurations.

Table 11: Computational overhead of S-OPD. Both methods use four NVIDIA A100 80GB GPUs, two for the student and two for the teacher. Training time excludes initialization, checkpoint saving, and validation. Overhead is relative to OPD in time per step.
<table><tr><td></td><td colspan="2">ViRL39K, 2B</td><td colspan="2">Geometry3K, 4B</td></tr><tr><td>Metric</td><td>OPD</td><td>S-OPD</td><td>OPD</td><td>S-OPD</td></tr><tr><td>Tokens/response</td><td>2235.9</td><td>2233.2</td><td>802.4</td><td>800.3</td></tr><tr><td>Time/step (s)</td><td>189.44</td><td>243.56</td><td>92.26</td><td>107.06</td></tr><tr><td>Gen. (ms/token)</td><td>0.233</td><td>0.252</td><td>0.315</td><td>0.333</td></tr><tr><td>Overhead</td><td>一</td><td>+28.6%</td><td>一</td><td>+16.0%</td></tr></table>

## B.6 COMPUTATIONAL OVERHEAD ANALYSIS

S-OPD introduces additional training computation through teacher evaluation of masked images for token gating and student evaluation of masked and noisy images for TPC and PA. All branches reuse the same student-generated response, requiring no additional autoregressive rollouts, trainable parameters, or annotations. Table 11 quantifies the resulting overhead: average time per step increases from 189.44 to 243.56 seconds for the 2B student on ViRL39K and from 92.26 to 107.06 seconds for the 4B student on Geometry3K, corresponding to moderate increases of 28.6% and 16.0%, respectively. Average response lengths remain nearly unchanged in both settings. These auxiliary computations are used only during training; inference retains the original student architecture and decoding procedure, with no additional computational overhead.

## C QUALITATIVE CASE STUDIES

## C.1 SAMPLE-LEVEL ROLLOUT ANALYSIS

To complement the aggregate benchmark results, we examine three sample-level cases from Math-Verse, LogicVista, and MMMU-val. These examples are selected using two criteria: the OPD baseline produces an incorrect answer, whereas S-OPD produces the correct answer; and solving the problem requires information that is available only from the image. The cases cover complementary forms of visual reasoning, including reading geometric dimensions, propagating motion through spatial relations, and recognizing fine-grained visual patterns. Rather than serving as additional quantitative evidence, these examples illustrate how the two models differ in their use of visual evidence along the reasoning process.

![](images/51d69612363ed339fd3c3f8a809245c06875ce704ab4abdc363ef502418abd60.jpg)  
Figure 10: Qualitative examples from MathVerse, LogicVista, and MMMU-val where 2B S-OPD correctly grounds its reasoning in visual evidence, while 2B OPD produces incorrect answers.

![](images/73cce1c9bcebfdd9076210fd80a7054c2c2789609fabb4cb74766a573f5c7f40.jpg)  
In which direction does the orange gear rotate? (A) Clockwise (B) Counterclockwise (C) No rotation

![](images/fa2da9de5a794896266fd8cd851496f62e84a49dccc8ae1a3618a707129dda0c.jpg)  
What kind of Assue does this image depict? A. Cardiac muscle. B. Bone. C. Skeletal muscle

## C.1.1 MATHVERSE: TRIANGULAR PRISM VOLUME CALCULATION

OPD baseline (incorrect, answer is 8 cm<sup>3</sup>)

Official extracted answer: 8.

Although the response identified 2 cm, 4 cm, and 8 cm, it repeatedly confused the sides of the triangle with the lengths of the prism, and the final output was truncated. The original response was 14k characters long.

S-OPD (Ours) (correct, answer is 32 cm<sup>3</sup>)

Looking at the image:

• The triangle has a base of 4 cm and a height of 2 cm.

• The side of the prism that is perpendicular to the base is 8 cm.

$$
{ \mathrm { B a s e ~ A r e a } } = { \frac { 1 } { 2 } } \times 4 \times 2 = 4 { \mathrm { ~ c m } } ^ { 2 }
$$

$$
V = 4 \times 8 = 3 2 { \mathrm { c m } } ^ { 3 }
$$

So, the volume of the triangular prism is ${ \mathrm { 3 2 ~ c m ^ { 3 } } } .$

Analysis. This case tests whether the model can assign the three visible measurements to the correct geometric quantities. The OPD baseline successfully reads the values 2, 4, and 8 from the image, indicating that its failure is not caused by completely missing the relevant visual content. Instead, it repeatedly confuses the dimensions of the triangular base with the length of the prism. This role-assignment error leads to an incorrect extracted answer of $\mathrm { 8 c m ^ { 3 } }$ and a long, self-correcting response that ultimately terminates without a consistent derivation.

S-OPD uses the same visual measurements more coherently. It identifies 4 cm and 2 cm as the base and height of the right-triangular face, respectively, and interprets 8 cm as the length of the prism. It therefore computes the triangular area as ${ \frac { 1 } { 2 } } \times 4 \times { \dot { 2 } } = 4 { \mathrm { c m } } ^ { 2 }$ and the volume as $4 \times 8 = 3 2 \mathrm { { c m } ^ { 3 } }$ . The contrast between the responses suggests that, for this example, S-OPD improves the integration and semantic assignment of perceived quantities rather than merely extracting more numbers from the image.

## C.1.2 LOGICVISTA: GEAR ROTATION DIRECTION REASONING

![](images/5493d19c102946c875b65dbec7efe36c3474c7a226c7ffaa17ead08e8d210e2a.jpg)

Analysis. As shown in Figure 10, this example requires the direction of motion to be propagated through two successive gear contacts. The OPD baseline correctly recognizes the green arrow and infers that the rightmost gear rotates counterclockwise. It also correctly determines that the middle gear must rotate clockwise. However, it fails to apply the direction reversal at the second contact and consequently predicts that the orange gear also rotates clockwise. The error therefore occurs during relational composition rather than initial visual perception.

S-OPD explicitly follows the two visible gear contacts. Starting from the counterclockwise rotation of the rightmost gear, it reverses the direction once for the middle gear and a second time for the orange gear, yielding the correct answer, counterclockwise. This case illustrates that effective visual grounding must be maintained across multiple reasoning steps: correctly reading the arrow alone is insufficient unless the model consistently applies the spatial interaction represented by each gear contact.

## C.1.3 MMMU-VAL: MUSCLE TISSUE IMAGE IDENTIFICATION

![](images/42a27659cb8e5e9756620b762db16be56537121c8469cef2d6548cdfc57fd9af.jpg)

Analysis. Both models detect the prominent striated appearance of the tissue, but they interpret the arrangement of the fibers differently. The OPD baseline associates striation with cardiac muscle and selects option A, without sufficiently accounting for the long, unbranched, and highly parallel organization visible in the image. Its prediction is therefore based on a relevant but non-discriminative visual attribute.

S-OPD considers both the shared attribute and the features that distinguish the candidate tissue types. In particular, it contrasts the branching organization expected of cardiac muscle with the long, cylindrical, parallel fibers characteristic of skeletal muscle. This fine-grained comparison leads to the correct selection of option C. Unlike the preceding examples, which emphasize quantitative and spatial reasoning, this case demonstrates the use of multiple visual attributes for category discrimination.

## C.2 TOKEN-LEVEL PROBABILITY VISUALIZATION AND ANALYSIS

To examine how teacher gating modulates visually induced policy differences at the token level, we visualize three representative examples covering mathematical reasoning, abstract visual logic, and scientific diagram understanding. For each example, the student first generates a fixed response y from the original image I and the question $q .$ We then score the same response and prefix under the original image I and a patch-masked image $\widetilde { I }$ using both the student and teacher models. This controlled comparison isolates how removing visual evidence changes the probability assigned to each generated token.

Figure 11 visualizes the three cases. The signed student policy contrast is defined as

$$
\Delta _ { t } ^ { S } = \log p _ { S } ( y _ { t } \mid q , I , y _ { < t } ) - \log p _ { S } ( y _ { t } \mid q , \widetilde { I } , y _ { < t } ) ,\tag{9}
$$

where orange indicates that the original image increases the probability of token $y _ { t } .$ , blue indicates the opposite direction, and color intensity represents the absolute magnitude. The teacher gate strength is

$$
s _ { t } = \Bigl [ \log p _ { T } ( y _ { t } \mid q , I , y _ { < t } ) - \log p _ { T } ( y _ { t } \mid q , \widetilde { I } , y _ { < t } ) \Bigr ] _ { + } ,\tag{10}
$$

with darker purple denoting stronger teacher support from the original visual evidence. Finally, the gated policy contrast combines the teacher gate with the non-negative student contrast:

$$
C _ { t } ^ { \mathrm { g a t e d } } = s _ { t } \left( \exp ( d _ { t } ) - d _ { t } - 1 \right) , \qquad d _ { t } = \log p _ { S } ( y _ { t } \mid q , \tilde { I } , y _ { < t } ) - \log p _ { S } ( y _ { t } \mid q , I , y _ { < t } ) .\tag{11}
$$

Darker yellow therefore marks tokens for which the student policy changes substantially and the teacher simultaneously provides positive visual support.

The three cases reveal consistent but task-dependent behavior. In the MathVerse example, the student contrast responds broadly to visually grounded terms such as “unit,” “sec,” and the coordinates of $P ( x , y )$ , whereas the teacher gate places stronger emphasis on mathematical entities such as $\theta ,$ sec $\theta , P ,$ and the predicted answer. Their combination concentrates the gated contrast on the quantities required to infer sec θ from the unit-circle diagram. In LogicVista, the student is sensitive to several local shape descriptions, while the teacher more selectively emphasizes tokens describing the structural pattern, including intersections and shape categories such as “cross”, “triangle”, and “square”. The resulting gated contrast highlights the tokens used to distinguish the two sets. In the MMMU-val example, the contrast is concentrated around branch identities, node relations, and the answer token corresponding to the most recent common ancestor, reflecting the relational structure of the phylogenetic tree.

Across all three task types, the teacher gate and student policy contrast exhibit clearly different token-level emphasis. The teacher signal is therefore not a simple copy or uniform rescaling of the student’s visual sensitivity. Instead, the gated policy contrast preserves student discrepancies only when they are supported by the teacher’s response to the visual evidence, concentrating the learning signal on visually grounded and decision-relevant reasoning steps. The consistent behavior across distinct visual-reasoning domains supports the intended role of our teacher-gating mechanism as a selective token-level routing strategy.

![](images/e0486bcb365ad310e34cac4756ba1030c0f2eafa955f9caf37d8307046dd112e.jpg)  
Figure 11: Qualitative visualization of token-level policy contrast and teacher gating across three types of visual reasoning: mathematical geometry (MathVerse), abstract visual logic (LogicVista), and scientific diagram understanding (MMMU-val). In the student policy contrast, orange and blue indicate positive and negative changes in token log-probability between the original and masked visual inputs, respectively, while darker colors denote larger absolute changes. For the teacher gate strength, darker purple indicates a larger positive teacher log-probability gap between the original and masked images. For the gated policy contrast, darker yellow indicates a larger teacher-modulated policy discrepancy. The teacher and student exhibit distinct token-level emphasis across all three tasks, and the gated policy contrast selectively concentrates on key visual elements and decision-relevant reasoning steps, demonstrating the intended effect of our teacher-gating mechanism.