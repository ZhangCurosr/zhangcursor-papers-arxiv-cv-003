# Social Humanoid: Socially Responsive Human–Humanoid Interaction

Vongani H. Maluleke<sup>†∗</sup> James Ni<sup>†∗</sup> Kie Horiuchi<sup>†‡</sup> Lea Wilken<sup>§</sup> Jiahui Lei<sup>†</sup> Evonne Ng<sup>§</sup> Roei Herzig<sup>†</sup> Jitendra Malik<sup>†</sup> Trevor Darrell<sup>†</sup> Angjoo Kanazawa<sup>†</sup>

UC Berkeley<sup>†</sup>, Sony Group Corporation<sup>‡</sup>, Meta<sup>§</sup>

![](images/e61cb1ec914f15f40fc35b468b386c004bedbcbc9cc600a7559db27d8dceab21.jpg)  
Figure 1: Social Humanoid enables humanoids to perceive, anticipate, and respond to people with coordinated whole-body behavior. Examples include a hug (top) and a greeting followed by posing together for a photo (bottom). Given RGB-D video, our method reconstructs the observed human motion, then conditions on it to predict future human motion and a humanoid response. The generated motions are executed by a low-level tracking controller. Video results are available on our webpage: https://von31.github.io/social-humanoid/

## Abstract:

Humanoid robots are typically trained as solitary agents, leaving them unable to coordinate physically with human partners who naturally anticipate and adjust to one another. To bridge this gap, we present Social Humanoid, a hierarchical system that generates and executes socially responsive whole-body motion on a 29-DoF Unitree G1 conditioned on a human partner’s motion. Its high-level policy performs flexible sequence modeling of human and humanoid motion through a shared tokenized representation, generating reactive humanoid motion and anticipating the partner’s motion when they are occluded, while a low-level wholebody tracking controller executes the generated motion on the robot. To train the system, we introduce a two-phase adaptive retargeting procedure that retargets one interaction partner to the robot while morphing the other partner to preserve contact dynamics, reachability, and relative body scale. We apply this procedure across five motion-capture datasets to produce a new Human-Humanoid Interaction (HHI) benchmark of 260 hours of human-humanoid interactions. Our results demonstrate that Social Humanoid produces coordinated, human-aligned behavior across diverse interaction types in simulation and the real world.

Keywords: Human-Humanoid Interaction, Motion Retargeting

## 1 Introduction

Human interaction is the art of attunement, a continuous interplay of perception, anticipation, and response that enables seamless coordination. Whether synchronizing a high-five or collaborating on a shared task, people adjust effortlessly through gaze, posture, and timing, without words. This ability is a hallmark of social intelligence [1, 2, 3], allowing humans to simultaneously anticipate and align with others in real time. In contrast, existing works train humanoids as solitary agents, leaving them unable to perceive social cues or respond to the people around them. As humanoids move out of labs and into homes, workplaces, and shared public spaces, they will need the same capacity to operate alongside and with people.

To address this, we formulate the problem of human-humanoid interaction, in which a humanoid must continuously generate and execute coordinated, socially responsive whole-body motion given observations of a human partner. We present Social Humanoid, a hierarchical system with a highlevel and a low-level policy. The high-level policy generates reactive humanoid motion from the observed partner’s motion and anticipates the partner’s motion, even when they are occluded or human input is missing, enabling continuous interaction. The low-level policy then executes the generated motion on a real Unitree G1 humanoid.

While recent advances in human motion generation [4, 5, 6, 7] can synthesize realistic dyadic interactions offline, deploying this capability on physical humanoids introduces three challenges: First, a data gap: human-human interaction is captured at scale, but no dataset captures natural humanhumanoid interaction and naively retargeting each partner independently destroys the contact, reach, and spatial structure that defines the interaction. Second, a representation gap: human and humanoid motion live in different spaces, which forces the generative model to learn the differences between the two embodiments instead of focusing on the interaction itself. This gap has been largely overlooked in prior work. Third, a generation gap: a natural approach is to generate the interaction with a human motion model and retarget to humanoid through optimization, but optimization-based retargeting is designed for clean, offline data and is brittle to noisy, discontinuous partner observations. A socially responsive humanoid needs a feed-forward model to generate humanoid motion directly.

Social Humanoid addresses these challenges with three contributions that build on one another. (i) For the data gap, we introduce a two-phase Adaptive-HHI Retargeting procedure that retargets one partner to the 29-DoF Unitree G1 and morphs the other’s body pose to preserve contact, reach, and relative scale. Applied to Chi3D [8], Hi4D [9], Harmony4D [10], Inter-X [11], and Embody3D [12], it yields the HHI Benchmark, comprising 260 hours of dyadic and polyadic interactions paired with the G1. (ii) For the representation gap, our key insight is that a shared motion representation lets a single generative model reason jointly about both embodiments, which we instantiate with a shared-codebook VQ-VAE that tokenizes human and humanoid motion into a single discrete vocabulary. This vocabulary also supports real-time shadow teleoperation of the G1 without optimization-based retargeting (Sec. 5.2).. (iii) For the generation gap, our high-level policy is a Diffusion Forcing Transformer that autoregressively generates humanoid responses over this shared vocabulary; its diffusion-forcing objective is well-suited to imperfect, real-world input and supports fluid, long-horizon interaction. Paired with NVIDIA SONIC [13] as the low-level whole-body tracking controller, Social Humanoid demonstrates coordinated, interactive, and human-aligned behaviors in both simulation and real-world hardware experiments. Generation and execution run asynchronously, with a median end-to-end latency of 308 ms from camera frame to humanoid command.

## 2 Related Work

Human Motion Generation. Large-scale motion datasets such as AMASS [14] have enabled major advances in generative human motion modeling. Early approaches relied on recurrent and convolutional architectures for motion prediction and synthesis [15, 16, 17, 18, 19, 20], followed by structured and probabilistic formulations for multimodal motion generation [21, 22, 23]. More recently, transformer-, VAE-, autoregressive-, and diffusion-based approaches have significantly improved long-horizon coherence, controllability, and diversity [24, 25, 26, 27, 28, 29].

![](images/d872a98f6e60ad85f0c7090c524577bc435a593ea03c04cdea1ea70d689b802a.jpg)  
Figure 2: Social Humanoid Architecture. From RGB-D observations of a human partner (red), we reconstruct the human pose and localize the body in 3D using depth and estimated gravity. Relative human and humanoid states are encoded through the VQ-VAE encoders $\mathrm { E _ { h } }$ and E<sub>r</sub>. The resulting latent motion tokens $z _ { * }$ , relative pose transforms $T ^ { * }$ , and canonical motion transforms $\Delta T ^ { * }$ are concatenated and processed by the DFoT policy, then decoded via $\mathrm { D } _ { h }$ and $\mathrm { D } _ { \ i }$ <sub>r</sub> to future human behavior (blue) and socially coordinated humanoid motion (purple). SONIC executes the motion in sim or the real-world, and the robot state is updated for closed-loop interaction.

Humanoid Motion Generation. Parallel progress in humanoid motion generation has combined reinforcement learning [30, 31], imitation learning and retargeting [32, 33, 34], and generative mo tion priors [35, 36, 37, 38, 39] to produce increasingly natural humanoid behaviors. Recent systems further leverage monocular video reconstruction and diffusion-based policies for contextual and goal-aware control [40, 41], while optimization-based methods such as OmniRetarget [42] improve geometric and kinematic consistency during motion transfer. However, these approaches primarily focus on single-agent motion generation or independently retargeted trajectories, often failing to preserve the contact, reachability, and relative geometry required for socially coordinated human humanoid interaction. In contrast, our interaction-aware adaptive retargeting sequentially optimizes humanoid retargeting and partner co-morphing to preserve socially meaningful interaction structure.

Dyadic Interaction and Human Motion Anticipation. Humans coordinate interactions by continuously anticipating others’ intentions through posture, timing, gaze, and motion cues [3, 43, 44]. Motivated by this, recent work has explored dyadic human interaction [45, 46, 47, 48, 49, 50, 6, 51, 52], enabling realistic, coordinated behaviors between human partners. Complementary efforts study socially expressive and interaction-aware motion generation, including dance, social affordances, and language-conditioned interaction [53, 54, 4, 55, 56]. While these methods produce compelling human-human interactions, they do not address the embodiment mismatch, streaming perception, and control constraints required for deployment on physical humanoids.

Human-Humanoid Interaction and Embodied Control. Prior work on human-humanoid interaction has explored physical assistance and contact-rich behaviors such as standing support and hugging [57, 58], as well as compliant and safe humanoid control for close-proximity interaction [59, 60]. Anticipating human intentions or trajectories has also been studied for cooperative robotics [61], typically as auxiliary modules for timing or safety rather than as part of motion generation itself. Other recent approaches generate humanoid motion from high-level textual descriptions, including social interaction prompts [39]. Hierarchical control frameworks further demonstrate the benefits of decomposing high-level motion planning and low-level motor execution for robust robot behavior [62]. In contrast to prior work, Social Humanoid unifies predictive human motion un derstanding, cross-embodiment generative modeling, and embodied humanoid control into a single framework for socially coordinated human-humanoid interaction.

![](images/7c2745ce85b480032817fecee790ab6716375a042b5d4876dcf61a61f5e2b2db.jpg)  
Figure 3: Shared-Codebook VQ-VAE. Human and humanoid motion sequences are independently encoded into latent features and mapped to a shared vector-quantized codebook that learns a common motion vocabulary across embodiments. Embodiment-specific decoders reconstruct the respective motions, enabling aligned, transferable motion representations between humans and humanoids.

## 3 Human-Humanoid Interaction Generative Modeling

Enabling socially coordinated human-humanoid interaction requires not only realistic motion synthesis, but also the ability to react continuously to a human partner. We therefore train a generative model on human-humanoid interaction trajectories, enabling fast, reactive motion generation at inference time. The core of our Social Humanoid system is a high-level policy that predicts humanoid kinematic motion from observed human motion, which are subsequently executed by a low-level tracking policy; see Fig. 2 for an overview of the system. The construction of the interaction dataset via interaction-aware retargeting is described in Section 4.

Formally, we adopt a canonical-frame motion representation rather than absolute world coordinates, inspired by MAGNet [7]. Human articulated motion is given by SMPL joint rotations in 6D form, $\Theta ^ { \bar { h } } \in \mathbb { R } ^ { 2 \bar { 1 } \times 6 }$ , and humanoid motion by the 29 actuated joint angles $\bar { \Theta ^ { r } } \in \mathbb { R } ^ { 2 9 }$ . Root motion is encoded as a relative canonical transform between consecutive frames, $\Delta { \bf T } ^ { \mathrm { c a n } } = [ \Delta \Phi ; \Delta { \bf \Gamma } ]$ , where $\Delta \Phi \in \mathbb { R } ^ { 6 }$ is the relative root orientation (6D) and $\Delta \mathbf { r } \in \mathbb { R } ^ { 3 }$ the relative root translation in the canonical frame. We additionally retain the absolute canonical-to-root transform $\mathbf { T } ^ { \mathrm { c a n } \to \mathrm { r o o t } }$ and the cross-agent transform $\mathbf { T } ^ { r  h } ( \mathbf { T } ^ { \bar { h }  r } )$ that places each agent in its partner’s canonical frame, both in $\mathbb { R } ^ { 9 }$ (6D rotation, 3D translation). The complete per-agent motion representations are given by

$$
\begin{array} { r } { \mathbf { m } ^ { r } = [ \Delta \mathbf { T } ^ { \mathrm { c a n } , r } ; \mathbf { T } ^ { \mathrm { c a n }  \mathrm { r o o t } , r } ; \mathbf { \Theta } ^ { r } ; \mathbf { T } ^ { r  h } ] , \mathbf { m } ^ { h } = [ \Delta \mathbf { T } ^ { \mathrm { c a n } , h } ; \mathbf { T } ^ { \mathrm { c a n }  \mathrm { r o o t } , h } ; \mathbf { \Theta } ^ { h } ; \mathbf { T } ^ { h  r } ] . } \end{array}\tag{1}
$$

## 3.1 Shared-Codebook VQ-VAE

Generating motion for Social Humanoids requires efficient, streaming-based generation, which motivates us to compactly tokenize high-dimensional human and humanoid motion trajectories. Recognizing that human and humanoid motions may share a similar latent structure, we encode both trajectories into a unified discrete vocabulary using a shared-codebook VQ-VAE (Fig. 3). Crucially, this shared representation simplifies the training of the subsequent diffusion model (see Sec. 5), as the primary motion spaces of the human partner and the humanoid are inherently heterogeneous.

Embodiment-specific Encoders and Decoders. Human and robot motions lie in different feature spaces due to their differing dimensionalities and kinematic structures. We therefore use embodiment-specific encoders and decoders. Let ${ \bf x } _ { t } ^ { \mathrm { h } } \ = \ [ { \bf T } _ { t } ^ { \mathrm { c a n } \to \mathrm { r o o t } , h } ; \Theta _ { t } ^ { h } ] \ \in \ \mathbb { R } ^ { D _ { \mathrm { h } } }$ and ${ \bf x } _ { t } ^ { \mathrm { r } } \ =$ $[ \mathbf { T } _ { t } ^ { \mathrm { c a n }  \mathrm { r o o t } , r } ; \mathbf { \bar { \Theta } } _ { t } ^ { r } ] \in \mathbb { R } ^ { D _ { \mathrm { r } } }$ denote the human and robot motion features at frame t. The human and robot encoders, $\mathrm { E _ { h } }$ and $\mathrm { E _ { r } , }$ are 1D-convolutional networks with temporal downsampling by a factor of 4 and dilated ResNet-1D blocks, producing latent sequences $\mathbf { v } ^ { \mathrm { h } } , \mathbf { \bar { v } } ^ { \mathrm { r } } \in \mathbb { R } ^ { T _ { z } \times d _ { z } }$

These features are quantized into discrete sequences ${ \bf z } ^ { \mathrm { h } } , { \bf z } ^ { \mathrm { r } }$ using a shared codebook. Finally, symmetric decoders, $\mathrm { D _ { h } }$ and $\mathrm { D } _ { \mathrm { r } }$ , upsample these quantized latents back to the original frame rate.

Shared Codebook. Both embodiments share a single EMA codebook, $\mathcal { C } = \{ \mathbf { c } _ { k } \} _ { k = 1 } ^ { K } \subset \mathbb { R } ^ { d _ { z } }$ defining a common discrete motion vocabulary. Given human and humanoid motions, the temporally downsampled continuous representations $\mathbf { v } _ { } ^ { a }$ produced by the encoders for agent $a \ \in \ \{ \mathrm { h , r } \}$ are quantized as:

$$
\mathbf { z } ^ { a } = \mathrm { Q u a n t i z e } ( \mathbf { v } ^ { a } ; \mathcal { C } ) , \qquad \mathrm { w h e r e } \quad \mathbf { z } _ { t } ^ { a } = \arg \operatorname* { m i n } _ { \mathbf { c } _ { k } \in \mathcal { C } } \| \mathbf { v } _ { t } ^ { a } - \mathbf { c } _ { k } \| _ { 2 } .\tag{2}
$$

The resulting discrete latents $\mathbf { z } ^ { a }$ are subsequently passed to their respective decoders.

However, a shared codebook alone cannot guarantee identical code assignments for paired frames. We thus introduce two alignment terms: (i) Continuous alignment directly minimizes the distance between encoder outputs, $\mathcal { L } _ { \mathrm { a l i g n } } = \| \mathbf { v } ^ { \mathrm { h } } - \mathbf { v } ^ { \mathrm { r } } \| _ { 2 } ^ { 2 }$ , shaping a shared latent geometry. (ii) Discrete alignment applies a symmetric cross-entropy loss over the distance logits $\ell ^ { \mathrm { h } } , \ell ^ { \mathrm { r } } \in \mathbb { R } ^ { K }$ with stopgradient targets $c ^ { \mathrm { h } } , c ^ { \mathrm { r } }$ , computed as $\begin{array} { r } { \mathcal { L } _ { \mathrm { a g r e e } } = \frac { 1 } { 2 } [ \mathrm { C E } ( \ell ^ { \mathrm { h } } , c ^ { \mathrm { r } } ) + \mathrm { C E } ( \ell ^ { \mathrm { r } } , c ^ { \mathrm { h } } ) ] } \end{array}$ ]. Together, these terms encourage cross-embodiment code prediction and tie the codebook to corresponding human–robot poses rather than letting each encoder claim separate regions of $\mathcal { C } .$

Training Objective. The complete objective combines per-embodiment reconstruction, the VQ commitment loss $\mathcal { L } _ { \mathrm { c o m m i t } } = \lVert \mathbf { v } - \mathrm { s g } ( \mathbf { z } ) \rVert _ { 2 } ^ { 2 }$ , and the two proposed alignment terms:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e c } } + \beta \mathcal { L } _ { \mathrm { c o m m i t } } + \lambda _ { a l i g n } \mathcal { L } _ { \mathrm { a l i g n } } + \lambda _ { a g r e e } \mathcal { L } _ { \mathrm { a g r e e } } .\tag{3}
$$

The codebook entries are the only shared parameters across the architecture; all other model weights remain strictly embodiment-specific. The shared codebook also enables human-to-humanoid motion translation in a single feed-forward pass $( E _ { h } \to \mathcal { C } \to D _ { r } )$ , supporting real-time shadow teleoperation of the G1 without optimization-based retargeting (Appendix 5.2).

## 3.2 Sequence Modeling of Human–Humanoid Interaction

To model human–humanoid interaction in the shared token space introduced above, we use a flexible sequence model based on Diffusion Forcing Transformer (DFoT) [63, 64]. DFoT is a conditional generative model that integrates the denoising process of DDPMs [65] with the temporal strengths of transformers [66]. A core advantage of this design is its ability to learn non-Markovian temporal dependencies across entire sequences using adaptive noise levels.

Given the observed latent history $\{ { \mathbf z } ^ { \mathrm { h } } , { \mathbf z } ^ { \mathrm { r } } \} _ { 1 : T _ { c } }$ , DFoT predicts the future latents $\{ \hat { \mathbf { z } } ^ { \mathrm { h } } , \hat { \mathbf { z } } ^ { \mathrm { r } } \} _ { T _ { c } + 1 : T _ { c } + T _ { f } }$ During training, each future latent frame $\mathbf { z } _ { i } ^ { a }$ for agent $a \in \{ \mathrm { h } , \mathrm { r } \}$ at timestep i is noised independently according to its own frame- and agent-specific diffusion timestep $t _ { i } ^ { a } .$ , yielding ${ \bf z } _ { i } ^ { a } ( t _ { i } ^ { a } ) = \sqrt { \bar { \alpha } _ { t _ { i } ^ { a } } } { \bf z } _ { i } ^ { a } +$ $\sqrt { 1 - \bar { \alpha } _ { t _ { i } ^ { a } } } \epsilon _ { i } ^ { a }$ , where $\mathbf { z } _ { i } ^ { a }$ is the clean latent, α¯ is the noise schedule, and $\epsilon _ { i } ^ { a } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Crucially, human and robot frames are noised independently $( t _ { i } ^ { \mathrm { h } } \neq t _ { i } ^ { \mathrm { r } } )$ to prevent shortcutting. Context frames $1 { : } T _ { c }$ remain clean $( t _ { i } ^ { a } = 0 )$ to preserve the history conditioning signal.

The network consists of transformer layers processing the noisy latents $\mathbf { z } _ { i } ^ { a } ( t _ { i } ^ { a } )$ combined with a sinusoidal embedding for noise-level, RoPE for temporal order, and a learned agent embedding. We train with a Smooth- $. L _ { 1 }$ objective to recover clean latents, and during inference, the predicted future latents are passed to embodiment-specific decoders for motion reconstruction.

Sampling and Generation. Inference with DFoT uses autoregressive DDIM sampling [67] with $\eta = 0$ to generate coordinated human and humanoid motion. The denoising process is controlled by a sampling strategy matrix $\mathbf { K } \in \mathbb { Z } ^ { S \times ( N \times P ) }$ , which specifies per-token noise levels across denoising steps. Independent token-wise noise scheduling enables flexible generation over the agent–time grid, allowing the same model to perform tasks such as joint future generation, partner prediction, and partner inpainting through different sampling schedules without retraining.

Training Losses. We optimize Social Humanoid across three complementary spaces—latent, motion, and reconstruction—to balance diffusion accuracy with kinematic fidelity:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = w _ { \mathrm { l a t e n t } } { \mathcal { L } } _ { \mathrm { l a t e n t } } + w _ { \mathrm { m o t i o n } } { \mathcal { L } } _ { \mathrm { m o t i o n } } + w _ { \mathrm { r e c o n } } { \mathcal { L } } _ { \mathrm { r e c o n } } ,\tag{4}
$$

![](images/2ee56cabd04fa4342e14ef90084768cc87ee9f199fde281a58b738f0f77c620f.jpg)  
Figure 4: Adaptive-HHI Retargeting. Our retargeting pipeline starts from two ground truth humans. In the first pass, we retarget the ego human (orange) to the humanoid using a contact-adaptive weight matrix to encourage body pose in regions close to the physical interaction to be preserved. In a second pass, we co-morph the partner’s pose towards more plausible interaction via a contactgated morphing strategy driven by the amount of physical contact present in the interaction.

where $w _ { \mathrm { l a t e n t } } , w _ { \mathrm { m o t i o n } } ,$ and $w _ { \mathrm { r e c o n } }$ are weighting coefficients. The latent denoising loss recovers clean latents via Smooth- $. L _ { 1 }$ for human (h) and robot (r); the motion loss decodes predicted latents and applies Smooth- $L _ { 1 }$ in raw motion space against $\mathbf { x } ^ { a }$ from Eq. 1; the reconstruction loss enforces a robust bottleneck by passing ground-truth motion through a non-diffused autoencoding loop.

## 3.3 End-to-End Deployment

We deploy Social Humanoid on a Unitree G1 using the perception, generation, and control pipeline in Fig. 2, with most computation offboard. A tripod-mounted ZED 2i stereo camera observes the human partner. We estimate the partner’s SMPL pose using Fast SAM 3D Body [68] and recover metric root position from stereo depth. AprilTags and onboard odometry localize the humanoid in the same gravity-aligned frame. We encode both agents’ states into the shared interaction representation for DFoT, which autoregressively generates humanoid motion conditioned on observed partner motion and recent humanoid history using 30-step DDIM sampling. NVIDIA SONIC [13] tracks the decoded motion at 50 Hz. Generation and execution run asynchronously, with 308 ms median latency from camera frame to executed humanoid command. See Appendix A.6 for details.

## 4 Humanoid-Human Interaction Data Generation

Social humanoid development is bottlenecked by a lack of human-humanoid interaction datasets. While human-human data is abundant, direct kinematic retargeting fails due to morphology gaps. To this end, we propose an interaction-preserving retargeting method maintaining contact locations, relative poses, and inter-person geometry.

Building on OMNIRETARGET [42]’s Sequential Quadratic Programming (SQP) and interactionmesh formulation [69], we address a central challenge of dyadic retargeting: bridging the morphology gap without distorting the interaction. Retargeting the humanoid independently can place interaction points out of reach, while uniformly rescaling the pair would make the human partner unrealistically small, creating a mismatch between the people the humanoid interacts with at deployment. We use a two-pass pipeline that retargets the ego, then adapts the partner’s pose—not its scale—to the retargeted humanoid (Fig. 4).

Pass I: Ego-to-humanoid Retargeting. Given an interaction between a source human (the ego) and a partner, this pass retargets the ego to the humanoid, leaving the partner unchanged. Following OMNIRETARGET [69, 42], we solve a per-frame SQP over the humanoid configuration that preserves the local shape of the ego’s interaction mesh, with Laplacian coordinates scaled to humanoid height, and keeps the humanoid’s wrist and elbow distances to the partner equal to the ego’s. Temporal smoothness and posture regularization complete the objective, subject to joint limits and foot-sticking constraints. Both the shape and distance terms emphasize the wrists, where most contact occurs. The full objective is given in Appendix A.3.1 and our contact-aware grounding in Appendix A.3.3.

Table 1: Baseline comparison on HHI data.
<table><tr><td></td><td>Qual.</td><td colspan="3">Phys. Realism</td><td> $\mathrm { D y n . }$ </td><td>Pos.</td></tr><tr><td>Method</td><td>FD↓</td><td>FS↓</td><td>IP↓</td><td>MI↓</td><td>MPJVE↓</td><td>W-MPJPE↓</td></tr><tr><td>NN</td><td>0.053</td><td>0.138</td><td>0.057</td><td>0.056</td><td>0.014</td><td>1.099</td></tr><tr><td>RS</td><td>0.180</td><td>0.134</td><td>0.058</td><td>0.056</td><td>0.015</td><td>1.100</td></tr><tr><td>R2R [6]</td><td>0.504</td><td>0.921</td><td>0.088</td><td>0.088</td><td>0.019</td><td>0.545</td></tr><tr><td>Ours</td><td>0.118</td><td>0.876</td><td>0.051</td><td>0.002</td><td>0.007</td><td>0.128</td></tr></table>

Table 2: Ablations on HHI-Chi3D.
<table><tr><td></td><td>Qual.</td><td colspan="3">Phys. Realism</td><td> $\mathrm { D y n . }$ </td><td>Pos.</td></tr><tr><td>Variant</td><td>FD↓</td><td>FS↓</td><td>IP↓</td><td>MI↓</td><td>MPJVE↓</td><td>W-MPJPE↓</td></tr><tr><td>No VQ-VAE</td><td>0.253</td><td>0.931</td><td>0.027</td><td>0.018</td><td>0.007</td><td>0.075</td></tr><tr><td>Sep. VQ-VAE 0.181</td><td></td><td>0.833</td><td>0.088</td><td>0.212</td><td>0.007</td><td>0.085</td></tr><tr><td>Ours</td><td>0.072</td><td>0.845</td><td>0.026</td><td>0.019</td><td>0.005</td><td>0.057</td></tr></table>

Pass II: Partner Co-Morphing. Pass I leaves the human partner unchanged, causing contact misalignment because the G1 is substantially shorter than the source human. For example, the partner’s hands may reach above the humanoid’s shoulders during hugs. Pass II adapts the partner’s pose to the retargeted humanoid. For all contacts, we apply arm-chain IK over the partner’s shoulder, elbow, and wrist during frames with humanoid arm contact, preserving the source ego–partner wrist and elbow offsets relative to the retargeted humanoid while respecting anatomical joint limits. For highcontact segments, we additionally solve a body-pose QP over key vertex clusters (wrists, chest, and neck), which additionally adjusts the partner’s hips, spine, and knees. This restores contact fidelity across large morphology gaps while preserving realistic human proportions.

We retarget each clip twice, swapping the ego and partner, so both interaction roles are represented in the dataset.

## 5 Experiments

Metrics. We evaluate three tasks. For generative modeling, given the past and current human motion and the past humanoid motion, the high-level policy predicts the humanoid’s reactive motion, which we compare against the retargeted HHI data in terms of positional accuracy, motion dynamics, physical realism, and interaction coordination. For retargeting, we evaluate how faithfully a human– human interaction is transferred to the humanoid. Finally, we evaluate whether the low-level policy can execute the generated motions in simulation and on the real humanoid.

W-MPJPE is the mean per-joint 3D position error in the world frame. MPJVE is the mean per-joint velocity error, which captures temporal smoothness. Foot Skating (FS) is the skating contact ratio, measuring unrealistic foot sliding. Interpenetration (IP) is the penetration depth between human and humanoid body parts. Motion Interaction (MI) is the difference between the ground-truth and predicted correlations of human and humanoid joint positions. Frechet Distance (FD)´ compares the distributions of real and generated motion to measure overall quality with respect to the training data. For retargeting, Pose distortion is the root-relative MPJPE (cm) between the humanoid link and the corresponding joints of the human, scaled to humanoid height. $E _ { \mathrm { w r i s t } }$ (cm) is the mean absolute error between the humanoid–partner wrist distance and the demonstrated ego–partner wrist distance, where partner joints are taken at the homologous SMPL-X joint and the demonstration is scaled to humanoid height. Lower is better for all metrics except contact preservation and success rate.

Baselines. Evaluating socially responsive humanoid interaction is challenging because existing methods either synthesize human-human interactions offline or generate isolated humanoid behaviors without addressing executable cross-embodiment interaction generation. We therefore compare against retrieval-based Nearest Neighbor (NN) and Random Sample (RS), which measure dataset memorization and distributional coverage, and against Ready-to-React (R2R) [6], a reaction policy for two-character interaction generation. NN retrieves the training window with the lowest L2 distance between normalized partner joint positions in the canonical frame, and RS samples a window uniformly at random.

Table 3: Retargeting comparison across three interaction datasets. Best per column in bold; ↑/↓ indicate higher/lower is better.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td>Pose Distortion</td><td>Contact Preservation</td><td>Foot Skating</td><td rowspan="2">Interaction</td><td>Downstream RL Policy</td></tr><tr><td>MPJPE [cm] ↓</td><td>Dur. ↑</td><td>Dur. ↓  $E _ { \mathrm { w r i s t } }$ </td><td>Success Rate(%) ↑</td></tr><tr><td rowspan="3">CHI3D</td><td>PHC</td><td>6.179</td><td>0.478</td><td>0.107</td><td>[cm] ↓ 22.382</td><td>68</td></tr><tr><td>OmniRetarget</td><td>4.184</td><td>0.915</td><td>0.009</td><td>1.379</td><td>100</td></tr><tr><td>Ours</td><td>3.893</td><td>0.936</td><td>0.003</td><td>0.501</td><td>100</td></tr><tr><td rowspan="3">Hi4D</td><td>PHC</td><td>6.892</td><td>0.089</td><td>0.087</td><td>21.088</td><td>79</td></tr><tr><td>OmniRetarget</td><td>4.704</td><td>0.827</td><td>0.006</td><td>1.729</td><td>100</td></tr><tr><td>Ours</td><td>4.371</td><td>0.881</td><td>0.005</td><td>0.804</td><td>100</td></tr><tr><td rowspan="3">Harmony4D</td><td>PHC</td><td>7.656</td><td>0.215</td><td>0.291</td><td>23.365</td><td>62</td></tr><tr><td>OmniRetarget</td><td>5.220</td><td>0.676</td><td>0.033</td><td>2.981</td><td>81</td></tr><tr><td>Ours</td><td>4.638</td><td>0.775</td><td>0.020</td><td>1.909</td><td>97</td></tr></table>

## 5.1 Results

In Table 1, we present a comprehensive comparison across various evaluation metrics assessing motion and interaction quality, plausibility, and fidelity of the high-level policy on our HHI dataset.

Retargeting Motion Quality. Table 3 compares our retargeting method, Adaptive-HHI Retargeting, with PHC and OmniRetarget on CHI3D, Hi4D, and Harmony4D. As no ground-truth humanoid motion exists, we evaluate pose distortion of each method against the source human–human demonstration, scaled to the humanoid’s height. Ours achieves the lowest pose distortion, highest contact preservation, and lowest foot skating on all three datasets. The largest gain is in interaction fidelity: wrist error $E _ { \mathrm { w r i s t } }$ falls from 1.38 to 0.50 cm on CHI3D and from 2.98 to 1.91 cm on Harmony4D relative to OmniRetarget, while PHC exceeds 20 cm. For the downstream RL policy, OmniRetarget and ours both reach 100% success on CHI3D and Hi4D, but on Harmony4D ours reaches 97% versus 81% for OmniRetarget and 62% for PHC.

![](images/e13b98045af134d0381c8ac0f7fb543d691dbeb2a151d50697532c68cd899db2.jpg)  
Figure 5: Retargeting Qualitative Results. Each panel shows a ground-truth human–human interaction (ego in orange, partner in dark blue) next to its retargeted counterpart. The ego is retargeted to the humanoid (tan), and the partner’s pose is morphed to preserve interaction (light blue). Across diverse interactions our Adaptive-HHI Retargeting method preserves the contacts and relative positioning between the two agents.

High-level Policy Motion Quality. Table 1 evaluates the generative modeling capability of the high-level policy. Our method achieves the lowest positional error with a W-MPJPE of 0.128, substantially outperforming R2R (0.545), NN (1.099), and RS (1.100), and the best motion dynamics with an MPJVE of 0.007, indicating smoother, more temporally coherent motion. It also achieves the best interpenetration (0.051) and motion interaction (0.002), indicating physically consistent interactions with the human partner. NN achieves the best FD (0.053 vs. our 0.118), and NN and RS have lower foot skating (FS), since retrieval replays recorded training motions with physically valid foot contacts. Despite this limitation, the proposed approach still produces coherent full-body motions with substantially improved interaction.

![](images/5cf5ff90dcab1aaa7e7688472d6ae33cd5520b2535ca52809f13bd53c1290c3d.jpg)  
Figure 6: Real-world deployment. Frames run left to right in time. Top two rows: separate high five and hand-wave trials. Bottom: a single continuous take in which the humanoid transitions directly from a handshake into a bend

Ablations. In Table 2, we ablate two design choices of the high-level generative model on HHI-Chi3D to assess the importance of discrete motion tokenization and codebook sharing: removing VQ-VAE tokenization (No VQ-VAE) and using per-agent rather than shared codebooks with no alignment (Sep. VQ-VAE). Our shared-codebook model achieves the best FD (0.072), IP (0.026), MPJVE (0.005), and W-MPJPE (0.057), confirming that discrete tokenization regularizes the highlevel policy and a shared codebook is key to coherent interaction.

Sim and Real Humanoid Evaluation. Our retargeting pipeline and high-level policy do not model physics explicitly, so we test whether the generated humanoid motions are physically feasible by executing them with the low-level policy, both in simulation and on the real G1. In simulation, a rollout succeeds if the low-level policy tracks the generated motion without falling or other unsafe behavior; we evaluate 30 generated motions per scenario. On the real humanoid, a trial succeeds if the G1 executes the appropriate reaction and completes the interaction without falling or operator intervention; we run 30 trials per scenario across 4 participants. As Table 4 shows, the low-level policy reproduces nearly all generated motions in simulation (90–100% per scenario), and on the real G1 (Fig. 6) success matches simulation in every scenario.

Table 4: Success rate (%) in sim and on the real G1.
<table><tr><td>Scenario</td><td>Sim Real</td></tr><tr><td>Handshake</td><td>100 100</td></tr><tr><td>Hand wave</td><td>100 100</td></tr><tr><td>High-five</td><td>90 90</td></tr><tr><td>Hug</td><td>100 100</td></tr><tr><td>Bend</td><td>100 100</td></tr><tr><td>Linked arms</td><td>100 100</td></tr></table>

## 5.2 Shadow Teleoperation

Beyond interaction generation, the shared-codebook VQ-VAE supports shadow teleoperation, in which the humanoid reproduces the whole-body motion of a human operator. As in crossembodiment translation (Appendix A.4.2), we encode the operator’s motion with the human encoder

![](images/aed0f36fea098f614b09bccd01c2da736a2922521d7739d364e5805b03c843a1.jpg)  
Figure 7: Shadow teleoperation. The G1 shadows a human operator performing arm movements (top) and sitting down on a couch (bottom). The operator’s motion is encoded with $E _ { h } ,$ , quantized against the shared codebook C, and decoded with D<sub>r</sub> into humanoid motion. Frames run left to right in time.

$E _ { h }$ , quantize it against the shared codebook C, and decode the resulting tokens with the humanoid decoder $D _ { r }$ . No humanoid motion is used as input, and no optimization-based retargeting is run.

Role of the shared codebook. Shadow teleoperation relies on human and humanoid poses mapping to the same codes, which is enforced by the alignment terms $\mathcal { L } _ { \mathrm { a l i g n } }$ and $\mathcal { L } _ { \mathrm { a g r e e } }$ in Eq. 3. Without them, the shared codebook reconstructs each embodiment well but translates poorly, with a humanto-humanoid MPJPE of 12.52 cm; with both terms, this error falls to 2.79 cm and code agreement between paired frames rises from 0.00 to 0.78 (Table A.5).

Qualitative results. Fig 7 shows the G1 shadowing an operator who sits down on a couch. As the operator moves from standing to sitting, the humanoid follows into a seated posture through its own kinematics rather than copying joint angles one-to-one.

## 6 Conclusion

We present Social Humanoid, a system for generating and executing socially responsive wholebody motion on a Unitree G1 conditioned on a human partner. It combines a two-phase Adaptive-HHI Retargeting procedure that produces paired human–humanoid data while preserving contact and scale, a shared-codebook VQ-VAE that maps both embodiments into a single discrete latent space, which also enables real-time shadow teleoperation, and a Diffusion Forcing Transformer that generates humanoid responses in this space, executed on the robot via SONIC. Together with the 260-hour HHI Benchmark, we show in simulation and on the real G1 that the system produces coordinated, human-aligned responses across a range of interaction types. We will release our data and code to support further work toward humanoids that coordinate physically with people.

Limitations and Future Work. First, we do not model hand articulation, facial expressions, or object pose, limiting dexterous, expressive, and object-centric interactions. Extending the shared codebook to fine-grained hand motion is a natural next step, while object interaction will likely require additional representation design. Second, we rely on SONIC and off-the-shelf human pose estimation, and inherit their failure modes in real-world deployment. Our system also tracks kinematic trajectories without modeling interaction forces, human comfort, or safety constraints during close contact, so safe deployment in unconstrained environments will require additional safety poli cies. Third, trackability and trial success measure execution, not social interaction quality; an HRI study is needed to assess the timeliness, naturalness, safety, and appropriateness of the humanoid’s responses. Fourth, we focus on dyadic human–humanoid interaction; extending to multiple humans, robots, or heterogeneous agents is an important direction for future work. Finally, when interacting with the robot, the lack of facial expressions or eye contact makes its behavior difficult to interpret, leading to awkward social interactions. An interesting direction for future work would be to enable more legible social interactions through facial expressions and eye contact.

## Acknowledgments

We thank Zeldah Maluleke, Kaito Suzuki, Marsalis Gibson and Sharmita Dey for serving as the human partner/operator in our real-world demonstrations and assisting with recording the demonstrations, and Zekai Wang and our BAIR colleagues for helpful discussions. This project was funded in part by Sony Corporation, Bakar Fellows, NSF CAREER award no 2442491, the Humanoid Intelligence Center program, ONR MURI N00014-21-1-280, and MURI N000142612024.

## References

[1] E. L. Thorndike. Intelligence and its Uses. Harper’s Magazine, 140:227–235, 1920.

[2] H. Gardner. Frames of Mind: The Theory of Multiple Intelligences. Basic Books, New York, 1983.

[3] D. Goleman. Social intelligence: The new science of human relationships. Bantam, 2006.

[4] V. Maluleke, L. Muller, J. Rajasegaran, G. Pavlakos, S. Ginosar, A. Kanazawa, and J. Malik.¨ Synergy and synchrony in couple dances. arXiv preprint arXiv:2409.04440, 2024.

[5] A. Ghosh, R. Dabral, V. Golyanik, C. Theobalt, and P. Slusallek. Remos: 3d motionconditioned reaction synthesis for two-person interactions. In European Conference on Computer Vision, pages 418–437. Springer, 2024.

[6] Z. Cen, H. Pi, S. Peng, Q. Shuai, Y. Shen, H. Bao, X. Zhou, and R. Hu. Ready-to-react: Online reaction policy for two-character interaction generation. arXiv preprint arXiv:2502.20370, 2025.

[7] V. H. Maluleke, K. Horiuchi, L. Wilken, E. Ng, J. Malik, and A. Kanazawa. Diffusion forcing for multi-agent interaction sequence modeling, 2025. URL https://arxiv.org/abs/2512. 17900.

[8] M. Fieraru, M. Zanfir, E. Oneata, A.-I. Popa, V. Olaru, and C. Sminchisescu. Threedimensional reconstruction of human interactions. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[9] Y. Yin, C. Guo, M. Kaufmann, J. J. Zarate, J. Song, and O. Hilliges. Hi4d: 4d instance segmentation of close human interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[10] R. Khirodkar, J.-T. Song, J. Cao, Z. Luo, and K. Kitani. Harmony4d: A video dataset for in-the-wild close human interactions, 2024. URL https://arxiv.org/abs/2410.20294.

[11] L. Xu, X. Lv, Y. Yan, X. Jin, S. Wu, C. Xu, Y. Liu, Y. Zhou, F. Rao, X. Sheng, et al. Interx: Towards versatile human-human interaction analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22260–22271, 2024.

[12] C. McLean, M. Meendering, T. Swartz, O. Gabbay, A. Olsen, R. Jacobs, N. Rosen, P. de Bree, T. Garcia, G. Merrill, J. Sandakly, J. Buffalini, N. Jain, S. Krenn, M. Kumar, D. Markovic, E. Ng, F. Prada, A. Saba, S. Zhang, V. Agrawal, T. Godisart, A. Richard, and M. Zollhoefer. Embody 3d: A large-scale multimodal motion and behavior dataset. Technical report, arXiv, 2025. URL https://arxiv.org/pdf/2510.16258. arXiv preprint.

[13] Z. Luo, Y. Yuan, T. Wang, C. Li, S. Chen, F. Castaneda, Z.-A. Cao, J. Li, D. Minor, Q. Ben,˜ X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, Z. Wang, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. Fan, and Y. Zhu. Sonic: Supersizing motion tracking for natural humanoid whole-body control. arXiv preprint arXiv:2511.07820, 2025.

[14] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019.

[15] K. Fragkiadaki, S. Levine, P. Felsen, and J. Malik. Recurrent network models for human dynamics. In Proceedings of the IEEE international conference on computer vision, pages 4346–4354, 2015.

[16] F. G. Harvey and C. Pal. Recurrent transition networks for character locomotion. In SIG-GRAPH Asia 2018 Technical Briefs, pages 1–4. 2018.

[17] M. Kaufmann, E. Aksan, J. Song, F. Pece, R. Ziegler, and O. Hilliges. Convolutional autoencoders for human motion infilling. In 2020 International Conference on 3D Vision (3DV), pages 918–927. IEEE, 2020.

[18] D. Pavllo, D. Grangier, and M. Auli. QuaterNet: A quaternion-based recurrent model for human motion. In British Machine Vision Conference (BMVC), 2018.

[19] I. Habibie, D. Holden, J. Schwarz, J. Yearsley, and T. Komura. A recurrent variational autoencoder for human motion synthesis. In British Machine Vision Conference (BMVC), 2017.

[20] E. Barsoum, J. Kender, and Z. Liu. HP-GAN: Probabilistic 3D human motion prediction via GAN. In Computer Vision and Pattern Recognition Workshops (CVPRW), 2018.

[21] E. Aksan, M. Kaufmann, and O. Hilliges. Structured prediction helps 3D human motion modelling. In International Conference on Computer Vision (ICCV), 2019.

[22] Y. Yuan and K. Kitani. Dlow: Diversifying latent flows for diverse human motion prediction. In European Conference on Computer Vision (ECCV), 2020.

[23] Y. Zhang, M. J. Black, and S. Tang. We are more than our joints: Predicting how 3D bodies move. In Computer Vision and Pattern Recognition (CVPR), 2021.

[24] Y. Duan, T. Shi, Z. Zou, Y. Lin, Z. Qian, B. Zhang, and Y. Yuan. Single-shot motion completion with transformer. arXiv preprint arXiv:2103.00776, 2021.

[25] C. Guo, X. Zuo, S. Wang, S. Zou, Q. Sun, A. Deng, M. Gong, and L. Cheng. Action2motion: Conditioned generation of 3d human motions. In Proceedings of the 28th ACM international conference on multimedia, pages 2021–2029, 2020.

[26] C. Ahuja and L.-P. Morency. Language2pose: Natural language grounded pose forecasting. In 2019 International conference on 3D vision (3DV), pages 719–728. IEEE, 2019.

[27] B. Jiang, X. Chen, W. Liu, J. Yu, G. Yu, and T. Chen. Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems, 36:20067–20079, 2023.

[28] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. arXiv preprint arXiv:2208.15001, 2022.

[29] M. Zhang, X. Guo, L. Pan, Z. Cai, F. Hong, H. Li, L. Yang, and Z. Liu. Remodiffuse: Retrievalaugmented motion diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 364–373, 2023.

[30] W. Yu, G. Turk, and C. K. Liu. Learning symmetric and low-energy locomotion. ACM Transactions on Graphics (TOG), 37(4):1–12, 2018.

[31] W. Yu, V. C. Kumar, G. Turk, and C. K. Liu. Sim-to-real transfer for biped locomotion. In 2019 ieee/rsj international conference on intelligent robots and systems (iros), pages 3503– 3510. IEEE, 2019.

[32] S. Choi, M. J. Song, H. Ahn, and J. Kim. Self-supervised motion retargeting with safety guarantee. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 8097–8103. IEEE, 2021.

[33] Z. Luo, J. Cao, A. W. Winkler, K. Kitani, and W. Xu. Perpetual humanoid control for real-time simulated avatars. In International Conference on Computer Vision (ICCV), 2023.

[34] Y. Ze, Z. Chen, J. P. Araujo, Z. ang Cao, X. B. Peng, J. Wu, and C. K. Liu. Twist: Teleoperated´ whole-body imitation system, 2025. URL https://arxiv.org/abs/2505.02833.

[35] Y. Du, R. Kips, A. Pumarola, S. Starke, A. Thabet, and A. Sanakoyeu. Avatars grow legs: Generating smooth human motion from sparse tracking inputs with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 481–490, 2023.

[36] S. Zhang, B. L. Bhatnagar, Y. Xu, A. Winkler, P. Kadlecek, S. Tang, and F. Bogo. Rohm: Robust human motion reconstruction via diffusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14606–14617, 2024.

[37] R. Boige, Y. Flet-Berliac, A. Flajolet, G. Richard, and T. Pierrot. Pasta: Pretrained action-state transformer agents. arXiv preprint arXiv:2307.10936, 2023.

[38] I. Radosavovic, B. Zhang, B. Shi, J. Rajasegaran, S. Kamat, T. Darrell, K. Sreenath, and J. Malik. Humanoid locomotion as next token prediction. Advances in neural information processing systems, 37:79307–79324, 2024.

[39] Z. Jiang, Y. Xie, J. Li, Y. Yuan, Y. Zhu, and Y. Zhu. Harmon: Whole-body motion generation of humanoid robots from language descriptions. arXiv preprint arXiv:2410.12773, 2024.

[40] A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, T. Darrell, P. Abbeel, J. Malik, and A. Kanazawa. Visual imitation enables contextual humanoid control, 2025. URL https://arxiv.org/abs/2505.03729.

[41] Q. Liao, T. E. Truong, X. Huang, G. Tevet, K. Sreenath, and C. K. Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion. arXiv preprint arXiv:2508.08241, 2025.

[42] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, and G. Shi. Omniretarget: Interaction-preserving data generation for humanoid whole-body loco manipulation and scene interaction, 2025. URL https://arxiv.org/abs/2509.26633.

[43] J. B. Freeman and K. L. Johnson. More than meets the eye: Split-second social perception. Trends in cognitive sciences, 20(5):362–374, 2016.

[44] N. Sebanz, H. Bekkering, and G. Knoblich. Joint action: bodies and minds moving together. Trends in cognitive sciences, 10(2):70–76, 2006.

[45] L. Muller, V. Ye, G. Pavlakos, M. Black, and A. Kanazawa. Generative proxemics: A prior for¨ 3d social interaction from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9687–9697, 2024.

[46] Q. Fang, Y. Fan, Y. Li, J. Dong, D. Wu, W. Zhang, and K. Chen. Capturing closely interacted two-person motions with reaction priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 655–665, 2024.

[47] Z. Wang, J. Wang, Y. Li, D. Lin, and B. Dai. Intercontrol: Zero-shot human interaction generation by controlling every joint. Advances in Neural Information Processing Systems, 37: 105397–105424, 2024.

[48] P. Ruiz-Ponce, G. Barquero, C. Palmero, S. Escalera, and J. Garc´ıa-Rodr´ıguez. in2in: Leveraging individual information to generate human interactions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1941–1951, 2024.

[49] K. Ji, Y. Shi, Z. Jin, K. Chen, L. Xu, Y. Ma, J. Yu, and J. Wang. Towards immersive human-x interaction: A real-time framework for physically plausible motion synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10173–10183, 2025.

[50] W. Jiang, J. Wang, K. Ji, B. Jia, S. Huang, and Y. Shi. Arflow: Human action-reaction flow matching with physical guidance. arXiv preprint arXiv:2503.16973, 2025.

[51] J. Tanke, T. Shibuya, K. Uchida, K. Saito, and Y. Mitsufuji. Dyadic mamba: Long-term dyadic human motion synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2868–2877, 2025.

[52] A. Ghosh, B. Zhou, R. Dabral, J. Wang, V. Golyanik, C. Theobalt, P. Slusallek, and C. Guo. Duetgen: Music driven two-person dance generation via hierarchical masked modeling. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025.

[53] Y. Liu, C. Chen, and L. Yi. Interactive humanoid: Online full-body motion reaction synthesis with social affordance canonicalization and forecasting. arXiv preprint arXiv:2312.08983, 2023.

[54] L. Siyao, T. Gu, Z. Yang, Z. Lin, Z. Liu, H. Ding, L. Yang, and C. C. Loy. Duolando: Follower gpt with off-policy reinforcement learning for dance accompaniment. arXiv preprint arXiv:2403.18811, 2024.

[55] M. Shan, L. Dong, Y. Han, Y. Yao, T. Liu, I. Nwogu, G.-J. Qi, and M. Hill. Towards open domain text-driven synthesis of multi-person motions. In European Conference on Computer Vision, pages 67–86. Springer, 2024.

[56] K. Fan, J. Tang, W. Cao, R. Yi, M. Li, J. Gong, J. Zhang, Y. Wang, C. Wang, and L. Ma. Freemotion: A unified framework for number-free text-to-motion synthesis. In European Conference on Computer Vision, pages 93–109. Springer, 2024.

[57] H. Lefevre, T. Chaki, T. Kawakami, A. Tanguy, T. Yoshiike, and A. Kheddar. Humanoid-\` human sit-to-stand-to-sit assistance. IEEE Robotics and Automation Letters, 2024.

[58] A. E. Block, H. Seifi, O. Hilliges, R. Gassert, and K. J. Kuchenbecker. In the arms of a robot: Designing autonomous hugging robots with intra-hug gestures. ACM Transactions on Human-Robot Interaction, 12(2):1–49, 2023.

[59] G. B. Margolis, M. Wang, N. Fey, and P. Agrawal. Softmimic: Learning compliant whole-body control from examples. arXiv preprint arXiv:2510.17792, 2025.

[60] Q. Lu, Y. Feng, B. Shi, M. Piseno, Z. Bao, and C. K. Liu. Gentlehumanoid: Learning upper-body compliance for contact-rich human and object interaction. arXiv preprint arXiv:2511.04679, 2025.

[61] C.-M. Huang and B. Mutlu. Anticipatory robot control for efficient human-robot collaboration. In 2016 11th ACM/IEEE international conference on human-robot interaction (HRI), pages 83–90. IEEE, 2016.

[62] A. Bajcsy, A. Loquercio, A. Kumar, and J. Malik. Learning vision-based pursuit-evasion robot policies. arXiv preprint arXiv:2308.16185, 2023.

[63] B. Chen, D. Mart´ı Monso, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann. Diffusion´ forcing: Next-token prediction meets full-sequence diffusion. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 24081–24125. Curran Associates, Inc., 2024. doi:10.52202/079017-0759. URL https://proceedings.neurips.cc/paper\_files/ paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf.

[64] K. Song, B. Chen, M. Simchowitz, Y. Du, R. Tedrake, and V. Sitzmann. History-guided video diffusion. In ICML, 2025.

[65] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[66] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[67] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[68] T. Yang, S. He, H. Jing, J. Yang, Z. Liu, C. Zou, and Y. Wang. Fast sam 3d body: Accelerating sam 3d body for real-time full-body human mesh recovery. arXiv preprint arXiv:2603.15603, 2026.

[69] E. S. L. Ho, T. Komura, and C.-L. Tai. Spatial relationship preserving character motion adaptation. ACM Transactions on Graphics, 29(4):33:1–33:8, 2010. doi:10.1145/1778765.1778770. SIGGRAPH 2010.

[70] S. Nakaoka and T. Komura. Interaction mesh based motion adaptation for biped humanoid robots. In Humanoid Robots (Humanoids), 2012 12th IEEE-RAS International Conference on, pages 625–631, Nov. 2012. doi:10.1109/HUMANOIDS.2012.6651585.

## Supplementary Material

## Table of Contents

A.1Introduction 16   
A.2Human-Humanoid Interaction (HHI) Dataset 16   
A.3Retargeting Details 17   
A.3.1Pass I Optimization 17   
A.3.2Adaptive-HHI Retargeting Ablation . 18   
A.3.3Contact-Aware Grounding 18   
A.3.4Hyperparameters . 18   
A.4Shared-Codebook VQVAE 19   
A.4.1Loss Ablation 19   
A.4.2Cross-Embodiment Translation: Human ⇒ Robot 19   
A.5High-level Policy Details 21   
A.5.1DFoT Training Loss 21   
A.5.2Implementation Details 22   
A.5.3Network Configurations 22   
A.6End-to-end System Details 22   
A.6.1Camera Setup and Calibration 23   
A.6.2Global Human Estimation 23   
A.6.3Global Robot Estimation 24   
A.6.4System Architecture 24   
A.6.5Latency Experiments 24   
A.6.6System Safety 25

## A.1 Introduction

We provide additional details on the dataset (Section A.2), retargeting (Section A.3), high-level policy (Section A.5), and end-to-end system (Section A.6) for Social Humanoid.

## A.2 Human-Humanoid Interaction (HHI) Dataset

Table A.1: Summary statistics of our retargeted benchmark. Clip counts, frames, and durations are reported after retargeting each source dataset to the Unitree G1 with Adaptive-HHI.
<table><tr><td>Dataset</td><td>Clips</td><td>Total Frames</td><td>Total Time (hrs)</td></tr><tr><td>CHI3D [8]</td><td>245</td><td>25,588</td><td>0.24</td></tr><tr><td>Hi4D [9]</td><td>200</td><td>23,890</td><td>0.22</td></tr><tr><td>Harmony4D [10]</td><td>416</td><td>153,160</td><td>1.42</td></tr><tr><td>Inter-X [11]</td><td>22,776</td><td>4,049,990</td><td>37.50</td></tr><tr><td>Embody 3D [12]</td><td>14,692</td><td>23,864,668</td><td>220.97</td></tr></table>

Table A.1 summarizes our unified human-humanoid Interaction (HHI) benchmark, constructed by retargeting five popularly used human-human interaction mocap datasets to the Unitree G1 humanoid with our Adaptive-HHI pipeline (Sec. A.3). The source datasets span a diverse spectrum of close-range interactive behaviors — from contact-rich dyadic interactions (Chi3D [8], Hi4D [9]) and in-the-wild multi-person activities (Harmony4D [10]) to large-scale captures of multi-person interactions (Inter-X [11], Embody 3D [12]). Each clip is processed according to its number of participants, and all sequences are resampled to 30 FPS. We adopt an 80/10/10% train/validation/test split.

## A.3 Retargeting Details

In this section, we compare Adaptive-HHI to existing retargeting methods and provide additional implementation details and hyperparameter configurations for our retargeting pipeline.

Table A.2 compares Adaptive-HHI to prior retargeting methods along four axes: hard kinematic feasibility, preservation of inter-agent interaction, partner co-morphing, and adaptive contact handling. Interaction meshes were introduced for close-interaction character animation [69] and later adapted to biped humanoids [70], where Laplacian deformation preserves spatial relationships under hard joint constraints, but only for the robot’s own body parts; the demonstrated interactions are self-contacts, and no human partner enters the optimization. PHC [33] imitates single-agent motion through a simulated physics policy, so feasibility is implicit and retargeted references can still exhibit motion artifacts. GMR [34] solves per-frame IK with joint and velocity limits, but its soft keypoint-matching objectives admit foot skating and penetration, and it has no notion of a partner. VideoMimic [40] adds scene context but handles collision and contact through soft penalties that compete with its matching costs. OmniRetarget [42] formulates retargeting as an interactionmesh-regularized SOCP with hard constraints and preserves robot–object–terrain relationships, yet the surrounding scene is a fixed input to the solve, not an optimization variable. This is the core limitation for human–humanoid interaction: when the partner cannot adapt, the robot must absorb the entire embodiment gap, and the interaction breaks. Adaptive-HHI keeps OmniRetarget’s SOCP base and adds (i) partner co-morphing, adapting the partner’s arm configuration to the retargeted humanoid on contact frames so the dyad deforms together, and (ii) wrist- and elbow-distance preservation with wrist-boosted Laplacian weights that concentrate on interaction-critical regions while preserving human-scale structure elsewhere. Adaptive-HHI is the only method that satisfies all four criteria for dyadic human–humanoid interaction.

Table A.2: Comparison of retargeting methods. Adaptive-HHI extends OmniRetarget’s interaction-mesh SOCP base with partner co-morphing (joint optimization over the partner SMPL-X) and distance-weighted, wrist-boosted adaptive Laplacian weighting for interaction and humanscale preservation.
<table><tr><td>Method</td><td>Hard Kin.</td><td>Preserve Interact.</td><td>Partner Co-Morph</td><td>Adaptive Contact</td></tr><tr><td>IMMA [70]</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>PHC [33]</td><td>x</td><td>X</td><td>x</td><td>X</td></tr><tr><td>GMR [34]</td><td>X</td><td>X</td><td>x</td><td>X</td></tr><tr><td>VideoMimic [40]</td><td>Soft</td><td>X</td><td>X</td><td>X</td></tr><tr><td>OmniRetarget [42]</td><td>√</td><td>X</td><td>X</td><td>x</td></tr><tr><td>Adaptive-HHI Retargeting (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## A.3.1 Pass I Optimization

Following the Laplacian-mesh formulation of OMNIRETARGET [69, 42], we build a per-frame Delaunay tetrahedralization over the ego’s keypoints, matched semantically to humanoid keypoints, and couple the humanoid to the partner through contact-driving joints P = {L/R Wrist, L/R Elbow}. Let $\mathbf { q } _ { t } \in \mathbb { R } ^ { 3 6 }$ be the humanoid configuration (29 DoF plus a 7D base pose), $\mathbf { x } _ { t } ( \mathbf { q } _ { t } )$ and $\mathbf { x } _ { t } ^ { \mathrm { { s r c } } }$ the humanoid and ego mesh vertices, and $r _ { j , t } ( \mathbf { q } _ { t } )$ and $d _ { j , t }$ the distances from the humanoid’s and the ego’s joint $j \in \mathcal { P }$ to the partner’s joint j. We solve, per frame:

$$
\begin{array} { l } { \displaystyle \underset { \mathbf { q } _ { t } } { \mathop { \operatorname* { m i n } } } \sum _ { v } w _ { v } \big \| \mathcal L \big ( \mathbf { x } _ { t } ( \mathbf { q } _ { t } ) \big ) _ { v } - s \mathcal L \big ( \mathbf { x } _ { t } ^ { \mathrm { s r c } } \big ) _ { v } \big \| ^ { 2 } + \displaystyle \sum _ { j \in \mathcal P } \lambda _ { j } \big ( r _ { j , t } ( \mathbf { q } _ { t } ) - d _ { j , t } \big ) ^ { 2 } } \\ { \displaystyle \qquad + \lambda _ { s } \| \mathbf { q } _ { t } - \mathbf { q } _ { t - 1 } \| ^ { 2 } + \mathbf { q } _ { t } ^ { \top } \mathbf { Q } \mathbf { q } _ { t } , } \end{array}\tag{5}
$$

subject to joint limits, a trust region on $\mathbf { q } _ { t } ,$ , and OMNIRETARGET’s velocity-based foot-sticking constraints. We optimize via SQP. The terms enforce (i) local shape preservation, matching the humanoid’s Laplacian coordinates $\mathcal { L } ( \cdot ) _ { v }$ at each keypoint v to the ego’s, scaled by the humanoid-to-ego height ratio $s ; ( \mathbf { i i } )$ contact and reach preservation, matching ${ \boldsymbol { r } } _ { j , t }$ to $d _ { j , t } ;$ (iii) temporal smoothness with respect to the previous solution, weighted by $\lambda _ { s } ;$ and (iv) posture regularization, with diagonal ${ \bf Q } \succeq 0$ (e.g. favoring neutral ankle pitch). The per-keypoint weights $w _ { v }$ and per-joint weights $\lambda _ { j }$ emphasize the wrists. Grounding is described in Sec. A.3.3.

## A.3.2 Adaptive-HHI Retargeting Ablation

Table A.3 ablates our retargeting on CHI3D, starting from OmniRetarget. Adding contact-adaptive weights reduces wrist error from 1.38 to 0.98 cm and slightly raises contact preservation (0.915 to 0.933), but increases foot skating (0.009 to 0.014). Adding contact-and-reach preservation gives the best result on all three metrics: wrist error falls to 0.50 cm, contact preservation reaches 0.936, and foot skating drops to 0.003. Overall, wrist error is reduced by about 64% relative to the baseline, with most of the gain from the second component.

Table A.3: Adaptive retargeting ablation on CHI3D. Mean ± std over clips. Best per column in bold.
<table><tr><td rowspan="2">Method</td><td></td><td>Interaction Contact Preservation Foot Skating</td><td></td></tr><tr><td> $E _ { \mathrm { w r i s t } } \ [ \mathrm { c m } ] \downarrow$ </td><td>Dur. ↑</td><td>Dur. ↓</td></tr><tr><td>Baseline (OmniRetarget)</td><td> $1 . 3 7 9 \pm 0 . 5 5 0$ </td><td> $0 . 9 1 5 \pm 0 . 1 6 0$ </td><td> $0 . 0 0 9 \pm 0 . 0 1 5$ </td></tr><tr><td>+ contact-adaptive weights  $( w _ { v , t } )$ </td><td> $0 . 9 7 6 \pm 0 . 5 3 4$ </td><td> $0 . 9 3 3 \pm 0 . 1 3 1$ </td><td> $0 . 0 1 4 \pm 0 . 0 1 8$ </td></tr><tr><td>+ contact-and-reach preservation (Ours)</td><td> $\mathbf { 0 . 5 0 1 } \pm \mathbf { 0 . 2 9 0 }$ </td><td> ${ \bf 0 . 9 3 6 \pm 0 . 1 1 9 }$ </td><td> $\mathbf { 0 . 0 0 3 4 \pm 0 . 0 1 0 }$ </td></tr></table>

## A.3.3 Contact-Aware Grounding

We observed that pinning the feet to the floor on every frame fails during dynamic maneuvers such as high kicks and vertical jumps, and that the Laplacian term does not constrain global vertical positioning. We resolve this by introducing contact-aware grounding via three complementary terms: (1) velocity-based foot sticking, inherited from OMNIRETARGET, holds a foot’s horizontal position fixed only during stance intervals estimated from toe velocity; (2) per-frame ground shifting applies a smoothed correction to the robot base height, aligning with the floor during stance and tracking the source sole height during airborne phases, which we detect as runs of 3–30 frames with the sole more than 4 cm above a rolling-minimum floor estimate; and (3) ankle posture priors combine footkeypoint downweighting with a diagonal ankle-pitch prior to suppress toe-down artifacts arising from human–humanoid foot mismatches.

## A.3.4 Hyperparameters

Table A.4 lists the hyperparameters used throughout the retargeting pipeline. All values were kept constant across all five source datasets. The distance weights reflect the contact semantics of dyadic interaction: wrist terms receive the largest weights since wrist placement dominates contact in our data (high-fives, handshakes, pushes), and elbow terms anchor the upper-arm orientation. The grounding parameters operate at the temporal scale of flight phases (3–30 frames at 30 Hz), separating genuine jumps from foot roll and capture drift, and the small smoothing width $( \sigma = 1$ frame) removes solver jitter without attenuating fast maneuvers such as kicks.

Table A.4: Retargeting hyperparameters. Values are shared across all datasets without per-clip tuning.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Pass I: Laplacian term Base keypoint weight Wrist / foot keypoint multiplier Laplacian target scale s</td><td>10  $\times 4 / \times 0 . 1$   $h _ { \mathrm { r o b o t } } / h _ { \mathrm { e g o } }$ </td></tr><tr><td>Pass I: distance and regularization Wrist / elbow distance weight  $\lambda _ { j }$  Smoothness weight  $\lambda _ { s }$  Ankle-pitch prior Shoulder-roll limit</td><td>50 / 25 0.5 2.0 100°</td></tr><tr><td>Pass I: SQP Iterations per frame (first / subsequent) Trust-region radius Foot-sticking velocity threshold Grounding</td><td>50 / 10 0.2 0.01 m/frame</td></tr><tr><td>Rolling-minimum floor window Airborne height threshold Airborne run length Ground-shift smoothing Pass II: arm IK</td><td>61 frames 4 cm 3–30 frames σ = 1 frame</td></tr><tr><td>Contact threshold (source meshes) Gauss-Newton iterations per frame Target / source-pose / damping weight Trust-region radius Frame rate Target rate</td><td>5 cm 10 200 / 0.1 / 0.05 0.6 rad 30 Hz</td></tr></table>

## A.4 Shared-Codebook VQVAE

## A.4.1 Loss Ablation

Table A.5 ablates the two alignment losses used to build the shared human (H) and robot (R) codebook. Without alignment, the codebook reconstructs each embodiment well but translates poorly across them: H→R and R→H FD are 0.206 and 0.157, and no paired frames share a code. Adding both alignment losses reduces translation FD by more than an order of magnitude (to 0.0028 and 0.0044), lowers cross-embodiment reconstruction error (H→R from 12.52 to 2.79 cm and R→H from 12.95 to 3.71 cm), and raises code agreement from 0 to 0.78, while self-reconstruction slightly improves (H 3.67→3.56 cm, R 2.81→2.61 cm). Each loss alone helps (code agreement 0.51 continuous, 0.63 discrete), but the combination is best. Discrete alignment alone slightly hurts self reconstruction (H 3.86 cm, R 2.88 cm) and H FD (0.0054), which the continuous loss recovers.

## A.4.2 Cross-Embodiment Translation: Human ⇒ Robot

To translate a human motion into humanoid motion, we encode it with the human encoder $\mathrm { E } _ { h } .$ , quantize it against the shared codebook C, and decode the resulting tokens with the humanoid decoder $\mathrm { D } _ { r } .$ No humanoid motion is used at inference, and no optimization-based retargeting is run. Translation is therefore a single feed-forward pass through the tokenizer. We evaluate it on held-out paired clips from the HHI benchmark, where the retargeted humanoid motion serves as ground truth for the translated output.

Qualitative results. Figure A.1 shows two translated sequences. The top rows of each example show the human input (red) and the translated humanoid motion (purple) in simulation. The bottom rows show the same motion tracked on the physical G1 by SONIC. In the first example (a standing, reaching gesture), the translated humanoid reproduces the arm extension and the timing of the reach. In the second example (a crouch), the humanoid matches the overall posture and descent to the floor. In both cases the output stays within the G1’s joint limits and proportions, so the motion is directly trackable on hardware. Because the human and humanoid have very different limb lengths and degrees of freedom, the translation does not copy joint angles one-to-one. It reproduces the pose through the humanoid’s own kinematics.

![](images/e4ff83becb0373bae18526611c8650fa9be0dcb26774d30448202b8bc11eafb2.jpg)  
Figure A.1: Cross-embodiment motion translation with a shared-codebook VQ-VAE. Human motion (red) is encoded into a discrete latent space shared between the human and humanoid embodiments, then decoded into humanoid motion (purple). Each example shows three frames of the input human motion alongside the translated humanoid motion in simulation (top row), and the same frames executed on the physical humanoid robot (bottom row).

Table A.5: Shared-codebook design ablation on HHI. H and R denote human and robot motion; H→R and R→H denote encoding in one embodiment and decoding in the other. FD is Frechet´ distance. MPJPE is the root-relative mean per-joint position error of the decoded pose against ground truth (cm; 21 SMPL-X body joints for H, 29 actuated joints of the G1 for R, hands excluded). Code Agree. is the fraction of paired H/R frames assigned the same code. Best per column in bold.
<table><tr><td></td><td colspan="4">FD↓</td><td colspan="4">MPJPE [cm] ↓</td><td>Code Agree. ↑</td></tr><tr><td>Method</td><td>H</td><td>R</td><td>H→R</td><td>R→H</td><td>H</td><td>R</td><td>H→R</td><td>R→H</td><td>H=R</td></tr><tr><td>No alignment</td><td>0.0047</td><td>0.0026</td><td>0.2062</td><td>0.1570</td><td>3.665</td><td>2.809</td><td>12.520</td><td>12.951</td><td>0.000</td></tr><tr><td>Continuous only</td><td>0.0045</td><td>0.0026</td><td>0.0034</td><td>0.0052</td><td>3.665</td><td>2.656</td><td>3.240</td><td>4.239</td><td>0.508</td></tr><tr><td>Discrete only</td><td>0.0054</td><td>0.0031</td><td>0.0046</td><td>0.0054</td><td>3.855</td><td>2.876</td><td>3.063</td><td>4.036</td><td>0.634</td></tr><tr><td>Both (Ours)</td><td>0.0047</td><td>0.0020</td><td>0.0028</td><td>0.0044</td><td>3.556</td><td>2.613</td><td>2.793</td><td>3.706</td><td>0.780</td></tr></table>

Quantitative results. Table A.5 reports translation quality alongside self-reconstruction. With both alignment losses, H→R translation reaches an FD of 0.0028 and a root-relative MPJPE of 2.79 cm. This is close to the humanoid’s own self-reconstruction error (2.61 cm) and is lower than the human’s self-reconstruction error (3.56 cm). Without alignment, the same shared codebook reconstructs each embodiment well (H: 3.67 cm, R: 2.81 cm) but translates poorly, with an H→R MPJPE of 12.52 cm and an FD of 0.206. This shows that sharing a codebook alone does not produce a transferable vocabulary, and that the alignment terms are what make the codes consistent across embodiments. Code agreement between paired human and humanoid frames rises from 0.00 to 0.78 with both losses.

## A.5 High-level Policy Details

This section details the training procedures, hyperparameters, and network architectures for both the shared-codebook VQ-VAE and the DFoT components of our high-level policy.

## A.5.1 DFoT Training Loss

We optimize Social Humanoid across three complementary spaces—latent, motion, and reconstruction—to balance diffusion accuracy with kinematic fidelity:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = w _ { \mathrm { l a t e n t } } { \mathcal { L } } _ { \mathrm { l a t e n t } } + w _ { \mathrm { m o t i o n } } { \mathcal { L } } _ { \mathrm { m o t i o n } } + w _ { \mathrm { r e c o n } } { \mathcal { L } } _ { \mathrm { r e c o n } } ,\tag{6}
$$

with weighting coefficients $w _ { \mathrm { l a t e n t } } = 1 , w _ { \mathrm { m o t i o n } } = 0 . 5$ , and $w _ { \mathrm { r e c o n } } = 0 . 5$ . The individual terms are

$$
\mathcal { L } _ { \mathrm { l a t e n t } } = \| \mathbf { z } ^ { h } - \hat { \mathbf { z } } ^ { h } \| _ { 1 } + \| \mathbf { z } ^ { r } - \hat { \mathbf { z } } ^ { r } \| _ { 1 } ,\tag{7a}
$$

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \sum _ { a \in \{ h , r \} } \Vert \mathbf { x } ^ { a } - \mathrm { D } _ { a } ( \hat { \mathbf { z } } ^ { a } ) \Vert _ { 1 } ,\tag{7b}
$$

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \sum _ { a \in \{ h , r \} } \| \mathbf { x } ^ { a } - \mathrm { D } _ { a } \big ( \mathrm { E } _ { a } \big ( \mathbf { x } ^ { a } \big ) \big ) \| _ { 1 } .\tag{7c}
$$

Denoising Loss. The primary supervision recovers the clean latents via a Smooth- $L _ { 1 }$ loss between the target and predicted latents for both the human (h) and robot (r):

$$
\mathcal { L } _ { \mathrm { l a t e n t } } = \| \mathbf { z } ^ { h } - \hat { \mathbf { z } } ^ { h } \| _ { 1 } + \| \mathbf { z } ^ { r } - \hat { \mathbf { z } } ^ { r } \| _ { 1 } .\tag{8}
$$

Motion Loss. To ensure physical plausibility, an auxiliary term decodes the predicted latents and applies a Smooth- $. L _ { 1 }$ loss in the raw motion space against the ground-truth motion features $\mathbf { x } ^ { a }$ :

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \sum _ { a \in \{ h , r \} } \| \mathbf { x } ^ { a } - \mathrm { D } _ { a } \big ( \hat { \mathbf { z } } ^ { a } \big ) \| _ { 1 } .\tag{9}
$$

Reconstruction Loss. To enforce a robust latent bottleneck, we pass the ground-truth motion through a non-diffused autoencoding loop and penalize the round-trip reconstruction error:

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \sum _ { a \in \{ h , r \} } \| \mathbf { x } ^ { a } - \mathrm { D } _ { a } \big ( \mathrm { E } _ { a } \big ( \mathbf { x } ^ { a } \big ) \big ) \| _ { 1 } .\tag{10}
$$

## A.5.2 Implementation Details

We train the shared-codebook VQ-VAE and DFoT models using the AdamW optimizer with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , a weight decay of $1 \times 1 0 ^ { - 4 }$ , and a mini-batch size of 256. The learning rate is cosine-decayed to zero over the course of training. Both the Shared-Codebook VQ-VAE and DFoT architectures utilize GELU activations and LayerNorm layers in their feed-forward blocks, and the DFoT model relies on a standard post-norm Transformer encoder architecture.

For the DFoT, we adopt a discrete diffusion formulation with an $x _ { 0 } \mathrm { { \cdot } }$ -prediction parameterization. During training, we sample a diffusion timestep $t \in \{ 1 , \ldots , 1 0 0 0 \}$ independently for each agent token to prevent architectural shortcutting, training the network to reconstruct the corresponding clean motion token $x _ { 0 } .$ . At inference time, we employ deterministic DDIM sampling over 30 steps. Unless otherwise stated, each VQ-VAE model is trained for 100,000 steps, and each DFoT model is trained for 300,000 steps. Final checkpoints are selected based on the lowest validation loss.

All training experiments are conducted on a single NVIDIA RTX A6000 GPU with 48 GB of VRAM. Under the configuration detailed in Table A.6, a full DFoT model convergence requires approximately 1 day of training time.

## A.5.3 Network Configurations

Table A.6 summarizes the hyperparameters of both sub-modules, shared-codebook VQ-VAE and DFot.

Shared-codebook VQ-VAE. Each agent’s motion is encoded by an embodiment-specific 1D convolutional encoder operating along the temporal axis with a compression factor of $\omega { = } 4$ , mapping per-agent pose features of dimension $D _ { \mathrm { i n } }$ (with $D _ { h }$ for the human and $D _ { r }$ for the humanoid) into a shared latent space of dimension $d _ { \mathrm { v q } } = 5 1 2$ . Both embodiments quantize against a single codebook of $K { = } 5 1 2$ entries updated via exponential moving average, which encourages a common motion vocabulary across the human–humanoid embodiment gap; embodiment-specific decoders mirror their encoders to reconstruct each agent’s pose representation.

DFoT. The high-level policy operates on unified motion tokens of dimension $D _ { \mathrm { m } } .$ which augment the VQ latents with localized cross-agent transformations $( 4 \times 9 \times P$ pairwise relative features). Per-token noise levels and agent identity are injected through sinusoidal and learned embeddings, respectively, concatenated with the motion tokens and projected to the model dimension d=512. Temporal position is encoded with RoPE, and the core consists of 6 Transformer blocks with 8 attention heads attending jointly over time and agents (sequence length $T ^ { \prime } \times P )$ . A linear head maps back to $D _ { \mathrm { m } }$

## A.6 End-to-end System Details

We implement a real-time system for Social Humanoid interaction. The system consists of a unified robot-human perception stack, followed by the high-level and low-level policies. This section describes the perception stack, system architecture, and the end-to-end latency.

Table A.6: Network configuration details. T is the raw sequence length and $T ^ { \prime } = T / \omega$ the number of tokens after temporal compression $( \omega { = } 4 ) ; P$ is the number of interacting agents; $D _ { \mathrm { i n } }$ is the peragent pose dimension $( D _ { h }$ or $D _ { r } ) ;$ D is the local conditioning dimension; $d _ { \mathrm { e } }$ is the embedding dimension; and $D _ { \mathrm { m } } = d _ { \mathrm { v q } } + 4 \times 9 \times P$ is the unified motion token dimension tracking localized cross-agent transformations.
<table><tr><td>Module</td><td>Component</td><td>Input shape</td><td>Operation &amp; hyperparameters</td></tr><tr><td rowspan="2">Shared- Codebook VQ-VAE</td><td>Human &amp; humanoid encoder Codebook</td><td> $( T , P , D _ { \mathrm { i n } } { + } D _ { \mathrm { c } } )$ </td><td> $1 \mathrm { D } \mathrm { C o n v } + \mathrm { R e s N e t } , 2 \mathrm { l a y e r s } , d _ { \mathrm { v q } } = 5 1 2$ </td></tr><tr><td>Human &amp; humanoid decoder</td><td> $( T ^ { \prime } , P , d _ { \mathrm { v q } } )$   $( T ^ { \prime } , P , d _ { \mathrm { v q } } { + } D _ { \mathrm { c } } )$ </td><td>VQ, K=512 codes, EMA updates Mirror of encoder, 2 layers</td></tr><tr><td rowspan="4">DFoT</td><td>Noise emb.</td><td> $( T ^ { \prime } , P , d _ { \mathrm { e } } )$ </td><td>Sinusoidal,  $d _ { \mathrm { e } } { = } 2 5 6$ </td></tr><tr><td>Agent emb.</td><td> $( T ^ { \prime } , P , d _ { \mathrm { e } } )$ </td><td>Learned,  $d _ { \mathrm { e } } { = } 2 5 6$ </td></tr><tr><td>Input proj. Time emb.</td><td> $( T ^ { \prime } , P , D _ { \mathrm { m } } { + } 2 d _ { \mathrm { e } } )$ </td><td>Linear + LayerNorm, 3 layers, → d=512</td></tr><tr><td>Core blocks Output head</td><td> $( T ^ { \prime } , P , d )$   $( T ^ { \prime } \times P , d )$ </td><td>RoPE on hidden dim d Transformer, 6 layers, 8 heads, MLP 4d</td></tr></table>

## A.6.1 Camera Setup and Calibration

We use an exoview ZED 2i camera to capture RGB+D at 1280x720 and 15 Hz (the capture rate is bandwidth-limited, see Sec. A.6.4). To estimate the gravity direction n and camera-to-floordistance $d ,$ we capture 30 depth frames, apply per-pixel temporal median, and backproject to a point cloud in the camera optical frame. We fit a plane $\mathbf { n } ^ { \top } \mathbf { p } = d$ with RANSAC under the constraints $\| \mathbf { n } \| = 1$ and $d > 0$ , and refine the winning hypothesis via SVD on the inliers. To separate the ground plane from other flat surfaces, we: (i) enforce gravity direction $\mathbf { n } ^ { \top } \bar { \mathbf { p } } > 0$ for the inlier centroid p¯, (ii) use directionality heuristic requiring a normal that is sufficiently vertical, and (iii) require a minimum inlier count.

## A.6.2 Global Human Estimation

Given per-frame RGB+D, we want to estimate the human’s SMPL pose and full SE(3) pelvis pose in a gravity-aligned, floor-origin camera frame $C _ { \mathrm { g c a m } }$ . We detect humans with YOLOv11 and pass each detection to Fast Sam 3D Body (FS3DB) [68], which consists of an optimized MHR head and an MHR-to-SMPL regressor. Because the regressor is not trained to preserve global orientation, we recover orientation by using the Kabsch algorithm between corresponding regressed SMPL vertices and MHR vertices in the camera frame. To recover translation, we query a depth patch at selected vertices on the torso to get estimates for the pelvis depth. We perform outlier rejection using z-buffer occlusion and MAD, aggregate the predictions by the detector’s confidence on the vertices, and then backproject the 2D pelvis using the camera intrinsics. A constantvelocity Kalman filter smooths the resulting trajectory against noisy depth. This yields $T _ { h  C _ { \mathrm { c a m } } }$ for every detected human, which we convert to $T _ { h  C _ { \mathrm { g c a m } } }$ using (n, d).

Table A.7: End-to-End Latency.
<table><tr><td>Stage</td><td>p50 (ms)</td><td>p95 (ms)</td></tr><tr><td>perception</td><td>104.4</td><td>120.4</td></tr><tr><td>fetch</td><td>7.9</td><td>8.4</td></tr><tr><td>sync</td><td>20.3</td><td>21.2</td></tr><tr><td>wire</td><td>52.7</td><td>66.4</td></tr><tr><td>JPEG+zstd</td><td>23.5</td><td>24.4</td></tr><tr><td>human</td><td>79.4</td><td>115.6</td></tr><tr><td>high-level</td><td>103.2</td><td>118.2</td></tr><tr><td>interpolate</td><td>38.6</td><td>57.8</td></tr><tr><td>VQ-VAE</td><td>8.1</td><td>10.4</td></tr><tr><td>DFoT</td><td>50.4</td><td>58.5</td></tr><tr><td>postproc</td><td>+133</td><td></td></tr><tr><td>low-level</td><td>21.1</td><td>22.0</td></tr><tr><td>SONIC</td><td>0.9</td><td>1.0</td></tr><tr><td>wire</td><td>20.2</td><td>21.0</td></tr><tr><td>wire buffer</td><td>+100</td><td></td></tr><tr><td>total (min)</td><td>308.1</td><td>376.2</td></tr><tr><td>total (max)</td><td>541.1</td><td>609.2</td></tr></table>

To maintain identities across frames, we construct and run a Kalman filter per tracked human. For each incoming frame, we KF-predict each active track and use the Hungarian algorithm on 3D Euclidian distance to assign each detected human to a track. Unassigned humans are allocated new tracks, and unassigned tracks retire after several consecutive misses. The robot maintains interaction with a single track determined on startup by the closest human. If no humans are visible, the high level policy can switch from partner-conditioned prediction to joint prediction.

## A.6.3 Global Robot Estimation

We place 8x8 inch AprilTags from the 36h11 family on the front and rear of the robot. During an initialization phase, we use the AprilTags to derive an estimate of the robot’s SE(3) pelvis pose $T _ { r _ { 0 }  C _ { \mathrm { c a m } } }$ and similarly use (n, d) to convert to $T _ { r _ { 0 }  C _ { \mathrm { g c a m } } }$ . The robot is positioned so that at least one tag is unoccluded during calibration. At inference time, we use an onboard, proprioceptionbased state estimation available in the Unitree SDK. Because the estimator is with respect to some canonical frame determined on robot startup, we use the estimator to compute $T _ { r  r _ { 0 } ; }$ , and compose it with the calibration estimate to get $T _ { r  C _ { \mathrm { g c a m } } }$ . Empirically, the Unitree estimator is sufficiently stable for our interaction scenarios.

## A.6.4 System Architecture

A laptop connected to the ZED 2i and the Unitree G1 performs time synchronization and publishes the RGB+D images and the robot estimation plus proprioception as a bundle at 15 Hz. Due to FS3DB’s compute requirements, we connect the laptop to a remote server using WiFi. We JPEGcompress the RGB stream and apply Zstd compression on the depth stream to maximize streaming rate. We run two instances of FS3DB with a global estimation coordinator at 15 Hz to tolerate variations in GPU load. The coordinator bundles the human and robot estimates and sends them to the high-level policy, which interpolates the incoming data to 30 Hz, performs inference, and emits chunks of 30 Hz robot motions at 7.5 Hz. Finally, SONIC interpolates the motions to 50 Hz and tracks them by providing desired DoF angles for a PD controller. We run inference on 2x NVIDIA 4090 GPUs for global human estimation, 1x NVIDIA 4090 GPU for the high-level policy, and 1x NVIDIA 4060 Mobile GPU for the low-level policy.

We support two optional improvements upon the existing system which tradeoff latency to improve motion generation quality and stability. After DFoT inference, we delay the emission of robot motion chunks by 1 chunk to improve decoded motion continuity. For robustness against deviations in network latency and other sources of variation, SONIC maintains a buffer of previous motion and delays execution by 100 ms.

## A.6.5 Latency Experiments

We evaluate the end-to-end latency of our pipeline. In Table A.7, we report the 50 and 95-percentile latencies (n = 200), splitting the evaluation by stage. Stage 1: perception, which includes (i)fetch: retrieving the RGB+D image from the camera, (ii) sync: synchronization with robot estimation and construction of bundle, (iii) wire: network and blocking on input on the server-side, (iv) JPEG+zstd: compression encoding and decoding. Stage 2: human estimation, which bundles distributed coordination, FS3DB inference, and global estimation. Stage 3: high-level policy, which includes (i) interpolate: interpolation of SMPL from 15→30 Hz, (ii) VQ-VAE: encoding and decoding, (iii) DFoT: DFoT inference, (iv) postproc (optional): buffered decoding to improve motion continuity. Stage 4: low-level policy, which includes (i) SONIC: construction of observations, policy inference, sending commands to G1, (ii) wire: network transport back to laptop, (iii) wire buffer (optional): execution delay to improve tolerance to variation. We highlight optional components which tradeoff motion generation quality for latency. In reactive regimes, we achieve a median latency of 308ms.

## A.6.6 System Safety

We validate safety progressively, from simulation to a gantry-supported robot to gantryless deployment. SONIC holds a standing pose until a dedicated operator, separate from the person interacting with the robot, engages it, and then smoothly executes high-level policy (HLP) frames. We clamp the desired yaw change to 45<sup>◦</sup>/s. SONIC smoothly returns to standing if consecutive HLP predictions differ by more than 0.7 rad, or if the operator judges the behavior unsafe. Across five 60 s rollouts, we observed no unsafe generated behavior. Note that the low-level policy is neither force-adaptive nor force-compliant, so physical contact is not actively regulated.