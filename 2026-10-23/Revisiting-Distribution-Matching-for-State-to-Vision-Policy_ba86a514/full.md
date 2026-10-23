# Revisiting Distribution Matching for State-to-Vision Policy Distillation in Laparoscopic Robotic Manipulation

Mingyu Sheng<sup>1</sup>, Jianan Fan<sup>1</sup>, Dongnan Liu<sup>1</sup>, Guoyan Zheng<sup>2</sup>, Ron Kikinis<sup>3</sup>, Weidong Cai<sup>1</sup>

Abstract— End-to-end control policies trained via Reinforcement Learning (RL) have emerged as an alternative to conventional surgical robotic control approaches, with recent studies exploring state-based policies trained on privileged state observations. However, such approaches rely on precise physical states that are difficult to obtain reliably in surgical environments. Vision-based RL provides an alternative by learning manipulation policies directly from RGB observations, avoiding explicit state estimation and manually engineered perception pipelines while enabling policies to acquire taskrelevant visual representations. Nevertheless, obtaining visionbased policies via pure RL methods for surgical manipulation remains challenging. Thus, imitation learning (IL) becomes another solution, yet it requires a large number of expert demonstrations, which are difficult to obtain in minimally invasive surgery. In this work, we propose a framework for state-tovision policy distillation in laparoscopic robotic manipulation. Specifically, a state-based policy with privileged observations is employed as the teacher, and a vision-based policy learns from RGB observations and robot proprioceptive information, which requires less expert demonstrations than IL and introduces superior performance than RL methods. We conduct extensive experiments on various surgical manipulation tasks in the simulated environment. Experimental results show that, in the context of laparoscopic robotic manipulation, training visionbased policies with general-purpose RL methods is challenging, whereas our proposed state-to-vision policy distillation leads to more consistent and stable performance across tasks.

## I. INTRODUCTION

Dexterous manipulation skills are critical components of surgical robotics tasks such as needle insertion, knot tying, and tissue retraction [1]. In recent years, significant progress has been made in applying deep reinforcement learning (RL) to accomplish such complex autonomous or semiautonomous subtasks [2], [3]. This progress has been largely enabled by the development of high-fidelity surgical robot simulators [4], [5], [6], [7], [8], [9]. However, a key limitation of RL-based surgical robotics lies in its reliance on privileged state information, which typically assumes accurate and noise-free access to the physical states of surgical objects, with limited integration of visual perception [10]. This issue is particularly pronounced in fine manipulation tasks such as needle handling, where even minor deviations in object pose can lead to task failure [2].

Recently, benchmarks have been proposed to provide a diverse set of learning-based tasks that capture varying levels of dexterity and precision required in the surgical context [8]. Benchmarks provide standardized platforms for evaluating autonomous policies. Nevertheless, surgical manipulation tasks still remain highly challenging due to their stringent requirements for precision, robustness, and safety in constrained operating environments.

To reduce the reliance on privileged state information, vision-based policy learning provides an alternative by directly mapping visual observations to control actions [11], [12], [13]. Such policies can avoid state estimation and manually engineered perception pipelines by leveraging visual information to learn task-relevant representations. However, learning effective control policies directly from visual observations via RL remains a major hurdle in surgical robotics. In the context of surgical manipulation tasks, the difficulty is further amplified by the severe visual sparsity of taskrelevant objects. For instance, the needle and instruments of surgical robots often occupy only a few pixels in highresolution RGB observations, making reliable perception and localization hard to achieve. Furthermore, the instability of RL exacerbates the training difficulty in such sparse and precision-critical settings [14]. Imitation learning (IL) defines a stable training objective, which transforms RL into supervised learning given a large number of expert demonstrations as data labels [15], [16]. However, compared with general robotic manipulation tasks, such as opening bottles or folding clothes, expert demonstrations for laparoscopic manipulation tasks are much more expensive and scarce.

To facilitate vision-based policy learning under these limitations, policy distillation provides an effective approach for transferring knowledge from a state-based policy to an RGB-based policy [17], [18], [19]. The state-based policy has access to privileged robot and object state information and serves as the teacher, and the RGB-based policy then learns to reproduce the teacher’s behavior from visual observations and robot proprioceptive states. Such knowledge transfer provides guidance for learning visuomotor policies from high-dimensional visual inputs, enabling the vision-based policy to acquire effective manipulation behaviors without relying on privileged state information during execution. However, most studies on policy distillation consider discrete action settings, such as those in Atari games, and thus the potential of policy distillation for continuous-control surgical robotic manipulation remains unexplored.

In this work, we propose a policy distillation framework for continuous-control laparoscopic robotic manipulation. Specifically, we proposed a distribution-matching-based policy learning approach that overcomes the limitation of previous methods by Gaussian alignment. We demonstrate the effectiveness of the approach through both theoretical anal ysis and empirical validation. We employ SurRoL [8] as the simulation environment, where a Patient Side Manipulator (PSM) executes the manipulation task while an Endoscopic Camera Manipulator (ECM) provides RGB observations. During training, the state-based policy has access to all privileged state information, including the coordinates and orientations of all objects, and serves as the teacher for the vision-based policy. In contrast, the vision-based policy only receives RGB observations and robot proprioceptive states, and is trained to distill the behavior of the state-based teacher without accessing privileged object states during execution. Through theoretical analysis and empirical evaluation, we provide insights into various distribution matching objectives for policy distillation, which may serve as a useful reference for future research on vision-based policy learning in surgical robotics manipulation. The main contributions of this work are summarized as follows:

• We formulate state-to-vision policy distillation as a stochastic distribution matching problem, where a vision-based student policy learns to match both the action and the uncertainty of a privileged state-based teacher policy, reducing the demand for expert demonstrations and demonstrating stronger stability.

• We provide a unified first- and second-order gradient analysis of commonly used distribution matching objectives, revealing their distinct gradient scaling, parameter coupling, and curvature properties for Gaussian policies.

• We evaluate the proposed method across various surgical robotic manipulation tasks, and the experimental results demonstrate its stronger stability and higher success rate on complex tasks.

## II. RELATED WORK

## A. Vision-Based Reinforcement Learning

Vision-based policies show greater potential for realworld applications because they can prevent the policy from becoming overly reliant on privileged states, but they are more difficult to train due to partial observability and the high dimensionality of image inputs. To date, there have been numerous studies on how to train vision-based policies. Asymmetric actor critic [20] leverages state-based critics to provide target values for vision-based actors, avoiding difficult vision-based critic learning. Nevertheless, asymmetric training may lead to an unbalanced model learning trajectory. To ensure that both the actor and the critic receive image inputs, Yarats et al. [12], [13] developed DrQ and DrQv2 for image-based RL training by introducing an augmentation technique. Laskin et al. [11] proposed a method based on contrastive representation learning. With sufficient expert demonstrations, imitation learning can obtain policies via supervised learning [15], [16], whereas the high demand for expert demonstrations is a limitation in the context of surgical robotic manipulation, where expert demonstrations are more expensive and scarce. These methods all achieve great performance in humanoid robotic control, but are challenged when applied to surgical robotic manipulation.

## B. Learning-Based Surgical Robotic Manipulation

To date, most methods focus on state-based policies, where the robotic agent is assumed to have access to the privileged states of objects [2], [21], [22], [23], [24], [25]. Based on state-based policies, surgical robotic manipulation is inevitably solved in two steps: perception and then manipulation. For example, the agent acquires object states via a hand-crafted image processing pipeline and then feeds the states into the policy network to obtain the robotic actions [26]. Such working pipelines are intuitive and simple to deploy in the real world. Nevertheless, a pure vision-based policy is still an attractive solution because of its one-step processing and stronger robustness against noise and varying scenes. To this end, some studies start to involve pixel-based observations [3], [27], [28], [29]. Haiderbhai et al. [3] and Gondokaryono et al. [28] trained RGB-based policies with conventional CNNs and LSTMs for basic manipulation tasks. To deal with more complex tasks, most vision-based studies depend on additional input observations, such as depth and segmentation masks, which may require extra processing pipelines, or rely on pretrained large models, resulting in high computational costs and potential error propagation. In this case, how to efficiently obtain a simple and effective vision-based policy remains a problem.

## C. Policy Distillation

The concept of policy distillation was first introduced by Rusu et al. [18], aiming to transfer the knowledge of multiple task-specific deep Q-networks to a single network. Czarnecki et al. [19] further pointed out that the learning trajectory can oscillate once reward learning is involved in the distillation process. Based on policy distillation, various visionbased policies for real-world manipulation tasks have been developed, by using Kullback-Leibler (KL) divergence or Mean Square Error (MSE) to transfer policy knowledge [30], [17], [31]. In most settings, policy distillation employs one policy as the teacher and another as the student, which can largely avoid the high demand for expert demonstrations, and trains the student policy through relatively stable supervised learning. We identify that the above properties of policy distillation are more appropriate for surgical vision-based policy training (i.e., low demand for expert demonstrations and a comparatively smooth training trajectory). Furthermore, most studies pertain to discrete action spaces, which cannot be applied to continuous-control laparoscopic robotic manipulation. These considerations motivate our study.

## III. METHODS

The overall workflow of this study is displayed in Fig. 1. Specifically, when training a vision-based policy (see

![](images/4b0899dc3474983d03b14d6eb6eeecbdd1823ce61fc6de2bb1dfe48bcb7c7d8e.jpg)  
Fig. 1. Semantic decomposition of this work. (a) The first step is to train a state-based policy. (b) Then, the state-based policy is employed as the teacher to transfer policy knowledge to the student network by aligning their action distributions. A new replay buffer $\mathfrak { B } _ { 2 }$ is used in (b) because $\dot { \mathfrak { B } } _ { 1 }$ in (a) contains transitions collected by earlier versions of the teacher policy, which may interfere with the student policy’s learning of the final teacher policy.

Fig. 1(b)), we employ a frozen teacher policy to interact with the environment (i.e., teacher-driven distillation) for its relatively stronger stability and reproducibility [32].

## A. Background

Soft Actor-Critic (SAC) [33] is an off-policy RL algorithm that learns a stochastic policy to maximize the expected cumulative return. It learns a policy $\pi _ { \psi }$ and critics $Q _ { \phi } .$ and $Q _ { \phi _ { 2 } }$ , where ψ and $\phi _ { i }$ are trainable network parameters. The parameter $\phi _ { i }$ is optimized by minimizing the Bellman error:

$$
\mathcal { L } ( \phi _ { i } , \mathfrak { B } ) = \mathbb { E } _ { b \sim \mathfrak { B } } \left[ \left( Q _ { \phi _ { i } } ( o , a ) - ( r + \gamma ( 1 - d ) \mathcal { T } ) \right) ^ { 2 } \right] \mathrm { , }\tag{1}
$$

where $\boldsymbol { b } = \left( o , a , o ^ { \prime } , r , d \right)$ is a transition with observation $^ { O , }$ action $^ { a , }$ next observation $\boldsymbol { o ^ { \prime } } ,$ reward r and done signal d, B represents the replay buffer, $\gamma$ is the discount factor, and $\tau$ is the target value, defined as:

$$
\mathcal T = \operatorname* { m i n } _ { i = 1 , 2 } Q _ { \phi _ { i } } ^ { * } ( o ^ { \prime } , \tilde { a } ^ { \prime } ) - \alpha \log \pi _ { \psi } ( \tilde { a } ^ { \prime } | o ^ { \prime } ) ,\tag{2}
$$

where $\tilde { a } ^ { \prime }$ indicates the predicted next action sampled from the policy $\pi _ { \psi } ( \tilde { a } ^ { \prime } | o ^ { \prime } ) , \ Q _ { \phi _ { i } } ^ { * }$ denotes the exponential moving average (EMA) of the network parameters $( \mathrm { i . e . , } \phi _ { i } )$ of $Q _ { \phi _ { i } }$ EMA is empirically employed to improve training stability in off-policy RL algorithms, and the entropy coefficient $\alpha > 0$ determines the relative weighting between entropy maximization and value function optimization.

In continuous-control tasks, the predicted policy $\pi _ { \psi }$ is typically a Gaussian distribution:

$$
\pi _ { \psi } ( \tilde { a } | o ) = \mathcal { N } ( \mu _ { \psi } ( o ) , \mathrm { d i a g } ( \sigma _ { \psi } ^ { 2 } ( o ) ) ) ,\tag{3}
$$

where $\tilde { a }$ is the predicted action given the observation $^ { O , }$ $\tilde { a } , \mu _ { \psi }$ and $\sigma _ { \psi }$ are treated as vectors whose dimensionality corresponds to the number of joints, as the joints are assumed

to be conditionally independent of each other. With the critic given by $Q _ { \phi _ { i } } ,$ , the actor network is trained by maximizing the expected return of its actions as follows:

$$
\mathcal { L } ( \psi ) = \mathbb { E } _ { a \sim \pi } \left[ \operatorname* { m i n } _ { i = 1 , 2 } Q _ { \phi _ { i } } ( o , \tilde { a } ) - \alpha \log \pi ( \tilde { a } | o ) \right] ,\tag{4}
$$

where action a˜ is sampled stochastically from the policy via reparameterization for gradient propagation:

$$
\widetilde { a } ( o , \xi ) = \operatorname { t a n h } \left( \mu _ { \psi } ( o ) + \sigma _ { \psi } ( o ) \odot \xi \right) ,\tag{5}
$$

where $\xi \sim \mathcal { N } ( 0 , I )$ is a standard Gaussian noise vector. In continuous-control reinforcement learning, stochastic policies are commonly parameterized as factorized Gaussian distributions, where the action dimensions are assumed to be conditionally independent. This parameterization provides tractable computation of policy likelihoods and facilitates efficient reparameterized sampling.

## B. State-to-Vision Policy Distillation

We revisit the SAC algorithm and observe that its learned Gaussian policy is particularly well-suited to distillation. Unlike conventional distillation methods that compute various divergence measures via Monte-Carlo estimators, the SAC policy explicitly models both the mean $\mu$ and the standard deviation $\sigma ,$ enabling closed-form distillation between teacher and student policies. Therefore, the discrepancy between the teacher and student policies can be accurately measured in the distribution space, providing a more deterministic and interpretable optimization. Moreover, the predicted standard deviation explicitly characterizes the uncertainty of the policy, allowing the student to learn both the action tendency and the confidence of the teacher behavior. Such uncertaintyaware policy transfer may lead to learning behaviors that differ from those of the conventional mean-only transfer.

For convenience and readability, the subsequent derivations are presented in terms of one joint, because different joints are assumed to be conditionally independent. First, we define the teacher policy and the student policy as:

$$
\begin{array} { r l } & { \pi _ { t } ( \tilde { a } | o _ { t } ) = \mathcal { N } ( \mu _ { t } ( o _ { t } ) , \sigma _ { t } ^ { 2 } ( o _ { t } ) ) , } \\ & { \pi _ { s } ( \tilde { a } | o _ { s } ) = \mathcal { N } ( \mu _ { s } ( o _ { s } ) , \sigma _ { s } ^ { 2 } ( o _ { s } ) ) , } \end{array}\tag{6}
$$

where the network parameter $\psi$ is replaced with t and s to indicate the state-based network (i.e., teacher) and the RGB-based network (i.e., student), respectively, $o _ { t }$ indicates the privileged state-based observation, and $o _ { s }$ indicates the visual observation (i.e., RGB frames). Under challenging laparoscopic robotic manipulation tasks, directly training $\pi _ { s }$ using RL is difficult. Therefore, identifying an effective distillation objective to align the two policies introduces an alternative solution for obtaining an RGB-based policy. The distillation loss is defined as:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = D ( \pi _ { t } , \pi _ { s } ) ,\tag{7}
$$

where $D$ is a differentiable function that measures the difference between $\pi _ { t }$ and $\pi _ { s } .$ . Given a pre-trained and frozen state-based policy $\pi _ { t }$ , optimizing the visual policy $\pi _ { s }$ by minimizing Eq. (7) can transfer policy knowledge from $\pi _ { t }$ to $\pi _ { s }$

## C. Distribution Matching Objectives

Given the stochastic Gaussian policies of the teacher and student, state-to-vision policy transfer can be formulated as a distribution matching problem. Specifically, we seek to minimize a divergence or distance between the two policies, such as the Wasserstein Distance (WD) and the Kullback-Leibler (KL) Divergence. Forward KL (FKL) divergence is a prevalent technique in various deep learning fields, such as model distillation, knowledge transfer, and representation alignment. Its Gaussian closed form is defined as:

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { F K L } } ( \pi _ { t } , \pi _ { s } ) = { \cal D } _ { \mathrm { K L } } ( \pi _ { t } \vert \vert \pi _ { s } ) } \ ~ } \\ { { \displaystyle ~ = \log ( \frac { \sigma _ { s } } { \sigma _ { t } } ) + \frac { \sigma _ { t } ^ { 2 } + ( \mu _ { t } - \mu _ { s } ) ^ { 2 } } { 2 \sigma _ { s } ^ { 2 } } - 0 . 5 } . } \end{array}\tag{8}
$$

Similarly, the Reverse KL (RKL) divergence between the state-based policy and the vision-based policy is written as:

$$
D _ { \mathrm { R K L } } ( \pi _ { t } , \pi _ { s } ) = D _ { \mathrm { K L } } ( \pi _ { s } | | \pi _ { t } ) .\tag{9}
$$

Although RKL is not a prevalently used method for distillation, we investigate it to gain more insights into how the uncertainty influences the learning behavior. FKL and RKL measure the discrepancy between the teacher and student policies, but their asymmetric formulations lead to distinct optimization behaviors. FKL exhibits a modecovering tendency, which may encourage the student policy to increase its variance to cover the probability mass of the teacher distribution. In contrast, RKL exhibits a modeseeking tendency, favoring concentrated distributions and potentially underestimating the uncertainty of the policy action [19], [34]. To balance these complementary behaviors, we further consider the Jeffreys divergence (JD), also known as the symmetric KL divergence (SKL), which incorporates both forward and reverse KL divergences:

FIRST- AND SECOND-ORDER DERIVATIVES OF DIFFERENT DISTRIBUTION MATCHING OBJECTIVES WITH RESPECT TO THE VISUAL POLICY  
PARAMETERS µ<sub>s</sub> AND σ<sub>s</sub>. WE DEFINE $\Delta \mu = \mu _ { s } - \mu _ { t }$ . THE ROW FOR $\frac { \partial D } { \partial \sigma _ { s } \mu _ { s } }$ IS OMITTED BECAUSE $\begin{array} { r } { \frac { \partial ^ { 2 } D } { \partial \sigma _ { s } \mu _ { s } } = \frac { \partial ^ { 2 } D } { \partial \mu _ { s } \sigma _ { s } } } \end{array}$
<table><tr><td>Derivative</td><td>FKL</td><td></td><td>RKL</td><td>JD</td><td>WD</td><td>MSE</td></tr><tr><td>∂D</td><td>∆μ</td><td></td><td>∆µ</td><td> $\Delta \mu \left( \frac { 1 } { \sigma _ { s _ { - } } ^ { 2 } } + \frac { 1 } { \sigma _ { t } ^ { 2 } } \right)$ </td><td>2∆µ</td><td>2∆µ</td></tr><tr><td>∂µs ∂D</td><td></td><td>σ2</td><td>σ2</td><td>σ8 σt + ∆µ2</td><td></td><td></td></tr><tr><td>∂σs</td><td></td><td> $\frac { 1 } { \sigma _ { s } } - \frac { \sigma _ { t } ^ { 2 } + \Delta \mu ^ { 2 } } { \sigma _ { s } ^ { 3 } }$  1</td><td> $\cdot \frac { 1 } { \sigma _ { s } } + \frac { \sigma _ { s } } { \sigma _ { t } ^ { 2 } }$ </td><td>σ σ3</td><td>2(σs − σt)</td><td>0</td></tr><tr><td>∂2D ∂µ2</td><td></td><td>σ2</td><td>-16</td><td>1 σ 1 σ2 十</td><td>2</td><td>2</td></tr><tr><td>∂2D</td><td></td><td> $\frac { 1 } { \sigma _ { s } ^ { 2 } } + \frac { 3 ( \sigma _ { t } ^ { 2 } + \Delta \mu ^ { 2 } ) } { \sigma _ { s } ^ { 4 } }$ </td><td>+</td><td>3(σ2 + ∆µ2) 十</td><td>2</td><td>0</td></tr><tr><td>∂σ2 ∂2D ∂µs∂σs</td><td></td><td>2∆µ σ3</td><td>σ2 σi 0</td><td>σ4 2∆µ σ3</td><td>0</td><td>0</td></tr></table>

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { J D } } ( \pi _ { t } , \pi _ { s } ) = { \cal D } _ { K L } \big ( \pi _ { t } \big | | \pi _ { s } \big ) + { \cal D } _ { K L } \big ( \pi _ { s } \big | \big | \pi _ { t } \big ) } } \\ { { \displaystyle ~ = \frac { 1 } { 2 } \cdot \Big ( \frac { \sigma _ { s } ^ { 2 } } { \sigma _ { t } ^ { 2 } } + \frac { \sigma _ { t } ^ { 2 } } { \sigma _ { s } ^ { 2 } } + \frac { \big ( \mu _ { t } - \mu _ { s } \big ) ^ { 2 } } { \sigma _ { t } ^ { 2 } } } } \\ { { \displaystyle ~ + \frac { \big ( \mu _ { t } - \mu _ { s } \big ) ^ { 2 } } { \sigma _ { s } ^ { 2 } } \Big ) - 1 } . } \end{array}\tag{10}
$$

In addition to the aforementioned information-theoretic divergences, we also examine the Wasserstein Distance (WD), a geometrically grounded discrepancy based on optimal transport. For two univariate Gaussian distributions, the squared 2-Wasserstein distance admits the following closedform expression [35]:

$$
D _ { \mathrm { W D } } ( \pi _ { t } , \pi _ { s } ) = W _ { 2 } ^ { 2 } ( \pi _ { t } , \pi _ { s } ) = ( \mu _ { t } - \mu _ { s } ) ^ { 2 } + ( \sigma _ { t } - \sigma _ { s } ) ^ { 2 } .\tag{11}
$$

This formulation explicitly accounts for discrepancies in both the mean and the standard deviation of the two policy distributions. Interestingly, the mean-matching term in Eq. 11 is identical to that of the conventional Mean-Square Error (MSE) distillation loss:

$$
D _ { \mathrm { M S E } } ( \pi _ { t } , \pi _ { s } ) = ( \mu _ { t } - \mu _ { s } ) ^ { 2 } ,\tag{12}
$$

which considers only the discrepancy between the means.

Different distillation losses exhibit distinct optimization behaviors and advantages. For instance, WD, MSE, and RKL exhibit strictly convex optimization landscapes with respect to $\mu _ { s }$ and $\sigma _ { s } ,$ while FKL and JD introduce uncertaintyadaptive optimization through standard-deviation scaling of the gradients, which is analyzed in Section III-D.

## D. Gradient Analysis

The first- and second-order derivatives of the different distribution matching objectives are summarized in Table I. As shown in the first row for $\frac { \partial D } { \partial \mu _ { s } }$ , all objectives drive the visual policy mean $\mu _ { s }$ toward the state policy mean $\mu _ { t }$ Furthermore, for FKL, RKL, and JD, the magnitude of the resulting gradient is additionally modulated by the standard deviations of the two policies, leading to different degrees of adaptive scaling.

In particular, FKL, RKL, and JD assign larger gradient magnitudes to policy dimensions with lower uncertainty, i.e., smaller standard deviations. This property enables the optimization of $\mu _ { s }$ to adapt to the confidence of the corresponding policy distributions. In contrast, the gradients of $\mu _ { s }$ for both WD and MSE are independent of $\sigma _ { t }$ and $\sigma _ { s } .$ Consequently, their optimization of the policy mean follows the same objective, as both minimize the squared difference between $\mu _ { s }$ and $\mu _ { t }$ . Although WD additionally minimizes the discrepancy between $\sigma _ { s }$ and $\sigma _ { t } .$ , under teacher-driven distillation, this variance-matching term has minimal effect on the gradient of $\mu _ { s }$ or the interaction with the environment. Therefore, for teacher-driven distillation, with respect to the optimization of $\mu _ { s }$ , WD is similar to MSE.

![](images/a10875e8cb05ae2721478ce09f333a6d226c0b7ea11685d81249d7b1f4aefc05.jpg)  
Fig. 2. Toy illustration of the PD regions of the FKL and JD Hessians under different teacher policy parameters. The PD region of FKL is always a proper subset of that of JD.

## E. Positive-Definiteness of Different Hessian Matrices

The Hessian matrix provides a useful perspective for understanding the optimization behavior of different distribution matching objectives. A positive-definite (PD) Hessian indicates that the objective exhibits fully positive curvature in all directions, corresponding to a locally strictly convex optimization landscape around the considered point. The Hessian matrix with respect to $\mu _ { s }$ and $\sigma _ { s }$ is defined as:

$$
H _ { D } = \nabla ^ { 2 } D = \left[ \begin{array} { c c } { \frac { \partial ^ { 2 } D } { \partial u _ { s } ^ { 2 } } } & { \frac { \partial ^ { 2 } D } { \partial u _ { s } \partial \sigma _ { s } } } \\ { \frac { \partial ^ { 2 } D } { \partial \sigma _ { s } \partial u _ { s } } } & { \frac { \partial ^ { 2 } D } { \partial v _ { s } ^ { 2 } } } \end{array} \right] .\tag{13}
$$

The RKL, WD and MSE Hessian matrices are clearly positive definite, whereas parameters $\mu _ { s }$ and $\sigma _ { s }$ are decoupled, i.e., $\partial ^ { 2 } D _ { \mathrm { R K L } } / ( \partial \mu _ { s } \partial \sigma _ { s } ) = 0$ , which means that updating $\sigma _ { s }$ does not affect the optimization of $\mu _ { s }$ . Consequently, when training the student policy, RKL is similar to a prior-scaled WD or MSE objective, where the gradient with respect to $\mu _ { s }$ is scaled by a fixed factor $\sigma _ { t } ^ { - 2 }$

For FKL and JD, the Hessian matrices are not always positive definite because their determinants are not greater than zero for all parameters:

$$
\begin{array} { r } { \operatorname* { d e t } ( H _ { \mathrm { F K L } } ) = \sigma _ { s } ^ { - 6 } \cdot \left( 3 \sigma _ { t } ^ { 2 } - ( \Delta \mu ) ^ { 2 } - \sigma _ { s } ^ { 2 } \right) , } \end{array}\tag{14}
$$

$$
\begin{array} { r l } & { \operatorname* { d e t } ( H _ { \mathrm { J D } } ) = \left( \displaystyle \frac { 1 } { \sigma _ { t } ^ { 2 } } + \displaystyle \frac { 1 } { \sigma _ { s } ^ { 2 } } \right) \cdot \left( \displaystyle \frac { 1 } { \sigma _ { t } ^ { 2 } } + \displaystyle \frac { 3 \left( \sigma _ { t } ^ { 2 } + ( \Delta \mu ) ^ { 2 } \right) } { \sigma _ { s } ^ { 4 } } \right) } \\ & { \quad \quad \quad - \displaystyle \frac { 4 ( \Delta \mu ) ^ { 2 } } { \sigma _ { s } ^ { 6 } } . } \end{array}\tag{15}
$$

This indicates that their Hessian matrices can be indefinite and that the optimization landscape is non-convex in certain directions. On the other hand, FKL and JD introduce adaptively scaled gradients with respect to $\mu _ { s }$ , where the scaling factor is determined by the student uncertainty in FKL and jointly by the teacher and student uncertainties in JD, resulting in uncertainty-aware optimization behavior. In contrast, RKL scales the gradient with respect to $\mu _ { s }$ solely by the teacher uncertainty, which remains fixed during student optimization.

Although neither the FKL nor the JD Hessian is positive definite over its entire domain, the JD Hessian exhibits a broader domain of positive definiteness than the FKL Hessian. We analyze this in Section III-F.

## F. Relaxed Positive-Definiteness Condition

Let $\mathbf { \pmb { x } } = [ \mu _ { t } , \sigma _ { t } , \mu _ { s } , \sigma _ { s } ] ^ { T }$ denote parameters of two policies, and let:

$$
H _ { \mathrm { F K L } } ( \pmb { x } ) = \nabla ^ { 2 } D _ { \mathrm { F K L } } ( \pmb { x } ) , H _ { \mathrm { R K L } } ( \pmb { x } ) = \nabla ^ { 2 } D _ { \mathrm { R K L } } ( \pmb { x } )\tag{16}
$$

denote the corresponding Hessian matrices. The JD is given by Eq. (10) and therefore its Hessian satisfies:

$$
{ \cal H } _ { \mathrm { J D } } = { \cal H } _ { \mathrm { F K L } } + { \cal H } _ { \mathrm { R K L } } .\tag{17}
$$

We have previously established that the RKL Hessian is strictly positive definite, i.e.,

$$
H _ { \mathrm { R K L } } ( { \pmb x } ) \succ 0 ,\tag{18}
$$

throughout the considered parameter domain. Consequently, for any nonzero vector z:

$$
z ^ { \top } H _ { \mathrm { R K L } } ( { \pmb x } ) z > 0 .\tag{19}
$$

This property leads directly to the following proposition.

Proposition 1. For any x satisfying $H _ { \mathrm { F K L } } ( { \pmb x } ) \ \succ \ 0$ , we have $H _ { \mathrm { J D } } ( { \pmb x } ) \succ 0$ . Moreover, there exists an $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ satisfying $H _ { \mathrm { J D } } ( \pmb { x } _ { 0 } )  0$ while ${ \cal H } _ { \mathrm { F K L } } ( { \pmb x } _ { 0 } ) \not \approx 0$ . Therefore, the PD region of FKL is a proper subset of that of JD:

$$
\boxed { \mathcal { P } _ { \mathrm { F K L } } \subsetneq \mathcal { P } _ { \mathrm { J D } } , }\tag{20}
$$

$$
\mathcal { P } _ { \mathrm { F K L } } = \pmb { x } : H _ { \mathrm { F K L } } ( \pmb { x } ) \succ 0 , \mathcal { P } _ { \mathrm { J D } } = \pmb { x } : H _ { \mathrm { J D } } ( \pmb { x } ) \succ 0 .\tag{21}
$$

Proof: Consider an arbitrary ${ \pmb x } \in \mathcal { P } _ { \mathrm { F K L } }$ . By definition, we have $H _ { \mathrm { F K L } } ( { \pmb x } ) ~ \succ ~ 0$ . For any nonzero vector z, we therefore obtain:

$$
z ^ { \top } H _ { \mathrm { F K L } } ( { \pmb x } ) z > 0 .\tag{22}
$$

Using Eqs. (17), (19), and (22), we have:

$$
\begin{array} { r l } & { z ^ { \top } H _ { \mathrm { J D } } ( \pmb { x } ) z = z ^ { \top } \left( H _ { \mathrm { F K L } } ( \pmb { x } ) + H _ { \mathrm { R K L } } ( \pmb { x } ) \right) z } \\ & { \qquad = z ^ { \top } H _ { \mathrm { F K L } } ( \pmb { x } ) z + z ^ { \top } H _ { \mathrm { R K L } } ( \pmb { x } ) z > 0 . } \end{array}\tag{23}
$$

TABLE II  
SUCCESS RATE (%) OF VARIOUS METHODS ACROSS DIFFERENT TASKS. THE ”LEARNING” COLUMN INDICATES REINFORCEMENT LEARNING (RL), IMITATION LEARNING (IL), OR POLICY DISTILLATION (PDL). BOLD MARKS THE BEST RESULT FOR EACH TASK ACROSS DIFFERENT METHODS. UNDERLINE DENOTES THE SECOND-BEST ONE. RESULTS MARKED WITH ”\*” ARE DIRECTLY COPIED FROM ORIGINAL PUBLICATIONS.
<table><tr><td>Method</td><td>Learning</td><td>NeedleReach</td><td>GauzeRetrieve</td><td>NeedlePick</td><td>PegTransfer</td><td>NeedleRegrasp</td></tr><tr><td>CURL [11]</td><td>RL</td><td>98.8</td><td>1.8</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DrQv2 [13]</td><td>RL</td><td>99.8</td><td>2.8</td><td>1.2</td><td>0.0</td><td>0.0</td></tr><tr><td>Diffusion [16]</td><td>RL</td><td>80.2</td><td>13.4</td><td>22.4</td><td>2.2</td><td>1.4</td></tr><tr><td>ACT [27]</td><td>IL</td><td>N/A</td><td>N/A</td><td>76.0*</td><td>N/A</td><td>N/A</td></tr><tr><td>ALOHA [15]</td><td>IL</td><td>99.8</td><td>76.8</td><td>93.6</td><td>16.4</td><td>10.0</td></tr><tr><td>Ours (FKL)</td><td>PDL</td><td>98.6</td><td>66.0</td><td>70.8</td><td>75.4</td><td>71.6</td></tr><tr><td>Ours (RKL)</td><td>PDL</td><td>99.8</td><td>76.8</td><td>39.8</td><td>74.8</td><td>70.2</td></tr><tr><td>Ours (JD)</td><td>PDL</td><td>98.8</td><td>71.8</td><td>80.4</td><td>77.4</td><td>74.8</td></tr><tr><td>Ours (MSE)</td><td>PDL</td><td>96.8</td><td>69.8</td><td>74.2</td><td>73.8</td><td>69.8</td></tr></table>

Hence, $H _ { \mathrm { J D } } ( { \pmb x } ) \ \succ \ 0 .$ . Since x is chosen arbitrarily from $\mathcal { P } _ { \mathrm { F K L } }$ , we obtain $\mathcal { P } _ { \mathrm { F K L } } \subseteq \mathcal { P } _ { \mathrm { J D } }$ . Specifically, by setting $\mu _ { t } = 0$ and $\sigma _ { t } = 0 . 5 ,$ , we consider the student parameters $\mu _ { s } ~ = ~ 0 . 7 5$ and $\sigma _ { s } ~ = ~ 1 . 0 $ , i.e., existing a point $\begin{array} { r l } { \pmb { x } _ { 0 } } & { { } = } \end{array}$ $[ 0 , 0 . 5 , 0 . 7 5 , 1 . 0 ] ^ { T }$ . At this point, using Eqs. (14) and (15), we have det $( H _ { \mathrm { F K L } } ( { \pmb x } _ { 0 } ) ) = - 0 . 8 1 2 5 < 0 \implies H _ { \mathrm { F K L } } ( { \pmb x } _ { 0 } ) \ \forall \ 0 .$ and det $\big ( H _ { \mathrm { J D } } ( \pmb { x } _ { 0 } ) \big ) \ = \ 2 9 . 9 3 7 5 > 0 \implies H _ { \mathrm { J D } } ( \pmb { x } _ { 0 } ) \ \succ \ 0$ Therefore, we have:

$$
\mathcal { P } _ { \mathrm { F K L } } \subsetneq \mathcal { P } _ { \mathrm { J D } } .\tag{24}
$$

This proves the proposition.

The above result establishes that the PD region of FKL is a proper subset of that of JD, indicating that JD admits a larger region of local convexity. We provide a toy illustration in Fig. 2. Therefore, JD does not merely symmetrize the divergence and balance mode covering and mode seeking. It also introduces a more relaxed positive-definiteness condition than FKL and can exhibit more favorable local optimization.

## IV. EXPERIMENTS AND RESULTS

## A. Implementation Details

All experiments are conducted in the SurRoL simulator [8]. The teacher policy receives privileged state information, including the coordinates and orientations of objects such as the PSM, needles, and gauze, whereas the student policy takes RGB frames from the ECM along with only the PSM gripper coordinates, without access to the states of other objects. All state policies are implemented as three-layer MLPs, and all RGB policies are implemented with three convolutional layers and three fully connected layers similar to those of CURL [11] and DrQ [12]. The teacher policy is first trained using Soft Actor-Critic (SAC) with the default hyperparameter settings [33], including the learning rate, the discount factor, etc. After training, the teacher policy is frozen and used to provide supervision for training the vision-based student policy. In particular, the JD loss is scaled by 0.5 to align its gradient magnitude with those of the other distribution matching objectives for fairness. To reduce experimental time costs, WD is not implemented separately because it shares the similar learning objective as MSE. The success rate for each evaluation is reported over 100 independent trials. Each experiment is independently repeated five times using different random seeds. All results are reported as averages across the five runs. Student policies trained by different loss distillation methods utilize the same teacher policy. All methods are trained via the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ , a batch size of 64, and $5 \times 1 0 ^ { 5 }$ iterations. During training, we periodically evaluate the policy every 1000 iterations and save the checkpoint with the highest reward for subsequent evaluation. For more implementation details, please visit https://github.com/MingyuShengSMY/SurgiPD.git.

## B. Results Comparison Across Different Tasks

Success rates of various methods under different tasks are shown in Table II. For pure RL-based vision methods (e.g., CURL and DrQv2), their performances are good on the simplest task (i.e., NeedleReach), whereas their effectiveness is significantly challenged when facing more complex tasks, such as NeedlePick and NeedleRegrasp where more dexterous manipulations are involved. ALOHA demonstrates superior performance on NeedleReach, GauzeRetrieve, and NeedlePick, but also exhibits performance degradation on more challenging tasks (i.e., PegTransfer and NeedleRegrasp). In terms of policy distillation, although its performance is similar to that of other methods on the simple manipulation task (i.e., NeedleReach), and slightly worse than ALOHA on GauzeRetrieve and NeedlePick, it can maintain a success rate of about 70% on complex manipulation tasks, showing relatively stronger stability and robustness.

## C. Results Comparison Across Different Distillations

To investigate the contributions of different distillation settings, we evaluate and compare the performances of FKL, RKL, JD, and MSE across various tasks (see Table II). Specifically, noticeably inferior performance is observed for RKL on NeedlePick, at 39.8%, showing its potential instability. Recalling the gradient equation of RKL in Table I, we hypothesize that the instability may result from the combined effects of the prior-scaled gradient (discussed in Section III-E) and the high certainty of the teacher policy, whose standard deviation is empirically around 0.3 in our experiments.

![](images/2b8d1afc81284d0c5dea45a86e7226e70ec2cce5e9dc51fd4fb9699b9d392005.jpg)  
Fig. 3. Feature maps extracted from the last convolutional layer. (a), (b), (c), and (d) represent GauzeRetrieve, NeedlePick, PegTransfer, and NeedleRegrasp, respectively. The small red orb marks the position to which the target objects (e.g., needle and gauze) should be moved. For a human-friendly view, the raw feature maps (64 × 64 × 32) are averaged across the channel dimension and refined via Gaussian blurring and interpolation.

In this case, the gradient is amplified threefold, and thus may cause gradient oscillation and lower performance.

Uncertainty-adaptive distillations (i.e., FKL and JD) usually outperform non-adaptive ones (i.e., RKL and MSE) on complex tasks (see PegTransfer and NeedleRegrasp in Table II). On the other hand, for simpler tasks, such as NeedleReach and GauzeRetrieve, there is no significant enhancement gained from uncertainty adaptation. This observation suggests the potential effectiveness of uncertainty adaptation for complex laparoscopic manipulation tasks.

Comparing FKL and JD, the performance of JD is slightly better than that of FKL across different tasks to varying degrees, corresponding to the Hessian matrix analysis in Section III-F, where JD features a broader convex optimization landscape that benefits the training.

## D. Feature Map Analysis

To understand what visual cues the policy relies on, we visualize the CNN feature maps, i.e., the output of the last convolutional layer. As shown in Fig. 3, high activations consistently concentrate on the gripper and the target object, while background regions remain weakly activated. Moreover, these high-activation regions follow the gripper and object as their positions change, rather than staying fixed to image coordinates. This indicates that the CNN learns task-relevant semantic representations encoding the relative spatial configuration between the gripper and the object, explaining for the robustness to object position changes.

## E. Conclusion

In this study, we propose a policy distillation framework for visuomotor laparoscopic robotic manipulation tasks, which can transfer policy knowledge from a state-based policy to a vision-based policy through distribution matching objectives. Our experimental results in the SurRoL simulator demonstrate that our method shows superior performance and robustness on relatively complex manipulation tasks. We also observe the potential effectiveness of the uncertainty adaptation mechanism by comparing different distillation objectives, and investigate the intermediate feature maps to show their capability of task-relevant semantic understanding. Although our experiments are conducted in the SurRoL simulator, existing works indicate that transferring such policies to real hardware is viable via sim-to-real methods, such as domain randomization [8], [36], [37]. Furthermore, in this work, we only utilize a simple network architecture consisting of three convolutional layers and three fully connected layers, without relying on advanced mechanisms such as attention, multi-scales, and mixture-ofexperts, and only leverage RGB frames and robot posture states as input, indicating the potential scalability to further enhance performance.

## REFERENCES

[1] P. Fiorini, K. Y. Goldberg, Y. Liu, and R. H. Taylor, “Concepts and trends in autonomy for robot-assisted surgery,” Proceedings of the IEEE, vol. 110, no. 7, pp. 993–1011, 2022.

[2] R. Bendikas, V. Modugno, D. Kanoulas, F. Vasconcelos, and D. Stoyanov, “Learning needle pick-and-place without expert demonstrations,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3326–3333, 2023.

[3] M. Haiderbhai, R. Gondokaryono, A. Wu, and L. A. Kahrs, “Sim2real rope cutting with a surgical robot using vision-based reinforcement learning,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 4354–4365, 2024.

[4] E. Tagliabue, A. Pore, D. Dall’Alba, E. Magnabosco, M. Piccinelli, and P. Fiorini, “Soft tissue simulation environment to learn manipulation tasks in autonomous robotic surgery,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020, pp. 3261– 3266.

[5] P. M. Scheikl, B. Gyenes, R. Younis, C. Haas, G. Neumann, M. Wagner, and F. Mathis-Ullrich, “Lapgym-an open source framework for reinforcement learning in robot-assisted laparoscopic surgery,” Journal of Machine Learning Research, vol. 24, no. 368, pp. 1–42, 2023.

[6] S. Schmidgall, A. Krieger, and J. Eshraghian, “Surgical gym: a high-performance gpu-based platform for reinforcement learning with surgical robots,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 13 354–13 361.

[7] V. M. Varier, D. K. Rajamani, F. Tavakkolmoghaddam, A. Munawar, and G. S. Fischer, “Ambf-rl: a real-time simulation based reinforcement learning toolkit for medical robotics,” in 2022 International Symposium on Medical Robotics (ISMR), 2022, pp. 1–8.

[8] J. Xu, B. Li, B. Lu, Y.-H. Liu, Q. Dou, and P.-A. Heng, “Surrol: an open-source reinforcement learning centered and dvrk compatible platform for surgical robot learning,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021, pp. 1821– 1828.

[9] Q. Yu, M. Moghani, K. Dharmarajan, V. Schorp, W. C.-H. Panitch, J. Liu, K. Hari, H. Huang, M. Mittal, and K. Goldberg, “Orbitsurgical: an open-simulation framework for learning surgical augmented dexterity,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 15 509–15 516.

[10] Y. Long, A. Lin, D. H. C. Kwok, L. Zhang, Z. Yang, K. Shi, L. Song, J. Fu, H. Lin, and W. Wei, “Surgical embodied intelligence for generalized task autonomy in laparoscopic robot-assisted surgery,” Science Robotics, vol. 10, no. 104, p. eadt3093, 2025.

[11] M. Laskin, A. Srinivas, and P. Abbeel, “Curl: Contrastive unsupervised representations for reinforcement learning,” in International conference on machine learning (ICML), 2020, pp. 5639–5650.

[12] I. Kostrikov, D. Yarats, and R. Fergus, “Image augmentation is all you need: Regularizing deep reinforcement learning from pixels,” arXiv preprint arXiv:2004.13649, 2020.

[13] D. Yarats, R. Fergus, A. Lazaric, and L. Pinto, “Mastering visual continuous control: Improved data-augmented reinforcement learning,” arXiv preprint arXiv:2107.09645, 2021.

[14] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, 2018.

[15] Z. Fu, T. Z. Zhao, and C. Finn, “Mobile aloha: Learning bimanual mobile manipulation using low-cost whole-body teleoperation,” in 8th Annual Conference on Robot Learning, 2024.

[16] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[17] I.-C. A. Liu, S. Uppal, G. S. Sukhatme, J. J. Lim, P. Englert, and Y. Lee, “Distilling motion planner augmented policies into visual control policies for robot manipulation,” in Proceedings of the 5th Conference on Robot Learning, vol. 164, 2022, pp. 641–650.

[18] A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy distillation,” Computer Science, 2015.

[19] W. M. Czarnecki, R. Pascanu, S. Osindero, S. Jayakumar, G. Swirszcz, and M. Jaderberg, “Distilling policy distillation,” in Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, vol. 89, 2019, pp. 1331–1340.

[20] L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel, “Asymmetric actor critic for image-based robot learning,” arXiv preprint arXiv:1710.06542, 2017.

[21] B. Azad, S. Zargarzadeh, and F. Rudzicz, “Surgidiff: a contextaware diffusion recommender for safe surgical autonomy,” in 2026 International Symposium on Medical Robotics (ISMR), 2026, pp. 143– 149.

[22] Y. Long, W. Wei, T. Huang, Y. Wang, and Q. Dou, “Human-in-theloop embodied intelligence with interactive simulation environment for surgical robot learning,” IEEE Robotics and Automation Letters, vol. 8, no. 8, pp. 4441–4448, 2023.

[23] G. Wang, X. Sun, Y. Xing, R. Cao, Z. Ma, and X. Zhang, “Dira: Diffusion-based imitation-to-reinforcement adaptation for task automation of surgical robots,” IEEE Robotics and Automation Letters, 2026.

[24] T. Huang, K. Chen, B. Li, Y.-H. Liu, and Q. Dou, “Guided reinforcement learning with efficient exploration for task automation of surgical robot,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, p. 459.

[25] T. Huang, K. Chen, W. Wei, J. Li, Y. Long, and Q. Dou, “Valueinformed skill chaining for policy learning of long-horizon tasks with surgical robot,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 8495–8501.

[26] A. Wilcox, J. Kerr, B. Thananjeyan, J. Ichnowski, M. Hwang, S. Paradis, D. Fer, and K. Goldberg, “Learning to localize, grasp, and hand over unmodified surgical needles,” in 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 9637–9643.

[27] W. Duan, B. Li, and Y.-H. Liu, “Real-time mask-conditioned surgical grasping via action chunking transformers,” in 2026 IEEE 20th International Conference on Control and Automation (ICCA), 2026, pp. 982–987.

[28] R. Gondokaryono, M. Haiderbhai, S. A. Suryadevara, and L. A. Kahrs, “Learning nonprehensile dynamic manipulation: Sim2real visionbased policy with a surgical robot,” IEEE Robotics and Automation Letters, vol. 8, no. 10, pp. 6763–6770, 2023.

[29] Y. He, P. Guo, M. Xu, Z. Li, A. Myronenko, D. Imans, B. Liu, D. Yang, M. Gu, and Y. Ji, “Cosmos-h-surgical: Learning surgical robot policies from videos via world modeling,” arXiv preprint arXiv:2512.23162, 2025.

[30] S. Levine, C. Finn, T. Darrell, and P. Abbeel, “End-to-end training of deep visuomotor policies,” Journal of Machine Learning Research, vol. 17, no. 39, pp. 1–40, 2016.

[31] W. Chen and N. Rojas, “Trakdis: a transformer-based knowledge distillation approach for visual reinforcement learning with application to cloth manipulation,” IEEE Robotics and Automation Letters, vol. 9, no. 3, pp. 2455–2462, 2024.

[32] T. Ave, T. De Schepper, and K. Mets, “Policy compression for intelligent continuous control on low-power edge devices,” Sensors, vol. 24, no. 15, p. 4876, 2024.

[33] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in International conference on machine learning (ICML), 2018, pp. 1861–1870.

[34] A. GX-Chen, J. Prakash, J. Guo, R. Fergus, and R. Ranganath, “Kl-regularized reinforcement learning for generative modelling is designed to mode collapse,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 50 665–50 693.

[35] M. Gelbrich, “On a formula for the l2 wasserstein metric between measures on euclidean and hilbert spaces,” Mathematische Nachrichten, vol. 147, no. 1, pp. 185–203, 1990.

[36] Z. Y. Chiu, F. Richter, E. K. Funk, R. K. Orosco, and M. C. Yip, “Bimanual regrasping for suture needles using reinforcement learning for rapid motion planning,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 7737–7743.

[37] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-toreal transfer of robotic control with dynamics randomization,” in 2018 IEEE international conference on robotics and automation (ICRA), 2018, pp. 3803–3810.