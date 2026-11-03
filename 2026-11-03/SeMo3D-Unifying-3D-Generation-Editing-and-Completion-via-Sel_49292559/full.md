# SeMo3D: Unifying 3D Generation, Editing, and Completion via Selective Modeling

Xiaokun Sun, Jian Yang, Zhenyu Zhang<sup>∗</sup> Nanjing University <sup>∗</sup>Corresponding Author

xiaokun sun@smail.nju.edu.cn, csjyang@nju.edu.cn, zhangjesse@foxmail.com Project Page: https://xiaokunsun.github.io/SeMo3D.github.io/

![](images/bebaccc0f69887f34a85049558598c1b60071416dad27b15240a07ca93aea093.jpg)  
Figure 1. SeMo3D unifies 3D generation, editing, and completion via Selective Modeling with image guidance and optional text and points, learning from standard 3D assets without paired editing or completion data. It further supports agentic interactive 3D creation.

## Abstract

3D generation, editing, and completion are typically treated as regenerating the complete 3D asset under task-specific constraints, even when only a local region needs modification. This leads to task isolation, inconsistency in content that should remain unchanged, and dependence on taskspecific paired data. We address these issues with a key observation: the three tasks all synthesize a designated region while keeping the rest intact—the full region for generation, the target region for editing, and the missing region for completion. We term this paradigm Selective Modeling and build SeMo3D upon it; it naturally unifies the three tasks, preserves unchanged content by construction, and enables training without paired editing or completion data, since any 3D asset can be partitioned into a selected region and

its context. For training, we propose region-selective denoising, which randomly selects regions ofvarying sizes on standard 3D assets and applies noise and loss only to them, jointly learning all three tasks. For architecture, we propose Omni3DiT, a multi-condition 3D diffusion transformer that distinguishes regions to createfrom those to retain and fuses image, text, and point guidance for fine-grained control. Experiments across three benchmarks show SeMo3D surpasses state-of-the-art methods on all tasks, delivering multimodal controllability in generation and strong consistency and accuracy in editing and completion. SeMo3Dfurther serves as a unified modeling tool for agents, enabling efficient 3D creation through natural-language interaction. Project page: https://xiaokunsun.github.io/ SeMo3D.github.io/.

![](images/3c57fe05ffe5615bfcfcdf3019841e51b281c85921c69ca60da365ceb2b96fc5.jpg)  
Figure 2. (a) Existing holistic modeling regenerates the complete asset even for local modification, requiring per-task models and paired editing/completion data, while regions that should remain unchanged are often altered (black boxes). (b) Our selective modeling synthesizes only the designated region, unifying the three tasks, preserving the rest by construction, and needing no task-specific paired data.

## 1. Introduction

3D content creation—spanning generation [36, 40], editing [21], completion [84], and their combinations—is central to games [92], AR/VR [1, 44], robotics [12, 55], and digital twins [47]. Although typically studied in isolation, these tasks are conventionally viewed through the same lens: regenerating a complete 3D asset under task-specific constraints—matching the condition for generation, and additionally the source asset for editing or the partial input for completion. A unified framework covering all three within one model is thus appealing yet remains an open problem.

Driven by generative models [24, 41] and large-scale 3D datasets [13, 14], all three tasks have advanced rapidly. As illustrated in Fig. 2(a), mainstream feed-forward approaches [11, 56, 72, 76–78] share a common paradigm we term holistic modeling: creating the complete 3D asset under image or text guidance. To preserve consistency with the source, editing and completion methods [11, 56, 72, 76] additionally condition on the source asset or partial input, yet they still regenerate the entire object even when only a local region needs to be modified.<sup>1</sup> This creates a fundamental mismatch between what needs to change and what is actually modeled, which leads to three key limitations. (1) Task isolation. As each task imposes a different constraint, generation [77, 78, 94], editing [46, 72, 76], and completion [11, 25, 45] rely on separate models with distinct architectures and objectives, duplicating design and training effort across tasks and multiplying deployment costs. (2) Region inconsistency. Because the full asset is regenerated, source fidelity can only be encouraged rather than guaranteed, and regions that should remain intact are frequently altered (black boxes in Fig. 2(a)). (3) Editing/completion data dependency. Regenerating a whole asset that respects the source demands source–target supervision, so many editing [19, 72, 76, 93] and completion [11, 25, 45] methods require costly large-scale paired datasets (e.g., PartFlow [72] curates 100K editing pairs via a seven-stage pipeline), limiting scalability. Moreover, most methods accept only a single guidance modality (image or text), offering limited control: text lacks geometric precision, while images leave occluded regions ambiguous.

Our key observation is that, despite their different constraints, the three tasks share a common structure: synthesizing a designated region while keeping the rest intact— the full, target, and missing regions for generation, editing, and completion. This motivates selective modeling: given an asset represented by spatially aligned tokens (e.g., structured latents [77, 78]), the model synthesizes only the selected tokens and reuses the rest (Fig. 2(b)), unlike holistic modeling. Selective modeling thus (1) unifies the three tasks, which differ only in the selected region; (2) preserves unchanged content by construction, as context tokens are never regenerated;<sup>2</sup> and (3) requires no task-specific paired data, since any ordinary asset can be partitioned into a selected region and its context. This resolves the three limitations above in a single paradigm.

Based on this insight, we present SeMo3D, a multicondition model unifying 3D generation, editing, and completion via Selective Modeling, as illustrated in Fig. 1. We instantiate selective modeling through two key components. On the training side, we propose region-selective denoising: randomly selecting regions of varying sizes on ordinary 3D assets, adding noise only to the selected tokens, and denoising them while keeping the rest clean. As the selected region ranges from a small portion to the full region, this single procedure spans the entire task spectrum, from local editing through partial completion to full generation, without task-specific paired supervision. On the architecture side, we design Omni3DiT, a unified multicondition 3D diffusion transformer that jointly processes clean context and noisy selected tokens, and further fuses complementary image, text, and point conditions for finegrained control. Experiments on three benchmarks show that SeMo3D outperforms specialized state-of-the-art methods on all tasks, with flexible multi-condition control in generation and high-fidelity preservation in editing and completion. SeMo3D further supports agentic and interactive 3D modeling through natural-language interaction.

Our contributions are summarized as follows:

• We formulate selective modeling, a unified paradigm that casts 3D generation, editing, and completion as synthesizing the designated region while preserving the rest, and present SeMo3D to instantiate this paradigm.

• We introduce region-selective denoising, which learns all three tasks from standard 3D datasets alone, without costly large-scale paired editing or completion datasets.

• We develop Omni3DiT, a multi-condition 3D diffusion transformer integrating clean 3D context with image, text, and point guidance for unified controllable 3D creation.

## 2. Related Work

3D Generation. Driven by 2D generative models [24, 41, 50, 54, 61, 62], 3D generation has undergone rapid paradigm shifts. Early methods [8, 51, 52, 59, 69] use score distillation sampling (SDS) to lift 2D priors to 3D, but suffer from long optimization times. Multi-view 2D methods [42, 43, 67, 73, 79] accelerate generation by first synthesizing multi-view images and then producing 3D shapes in a feed-forward manner. More recently, native 3D generative models [20, 28, 32, 74, 77, 78, 86, 94, 95, 97] operating directly on 3D latents have pushed quality and scalability further. Despite this progress, most methods condition on a single modality (image or text), limiting fine-grained controllability. Recent efforts incorporate pairwise combinations such as image-text [7, 9] or image-point [29, 75], but none of the above jointly leverages all three modalities to combine the visual detail of images, the semantic expressiveness of text, and the geometric precision of points.

3D Editing and Completion. 3D editing and completion have evolved analogously: from SDS-based optimization [10, 27, 30, 37, 57, 100, 101] through multi-view 2D methods [3, 4, 16, 18, 35, 70] to native 3D approaches [11, 25, 45, 56, 72, 76]. Native 3D methods are training-free or training-based. Training-free methods [26, 31, 33, 82, 89, 99] repurpose pretrained generators via inversion or optimization; they largely maintain source consistency but require slow, multi-step pipelines. Training-based methods [6, 11, 15, 19, 25, 45, 46, 56, 71, 72, 76, 80, 93] enable fast feed-forward inference but rely on large paired editing or completion datasets (e.g., 100K pairs in PartFlow [72], 116K in 3DEditFormer [76]) and often alter regions that should remain unchanged. Moreover, both lines mostly use a single modality, limiting controllability.

Unified 3D Multimodal Models. Recent works [85, 87, 88, 90, 91] consolidate multimodal output capabilities into one model, typically unifying 3D understanding with generation and editing [85, 88, 90, 91] or bridging 2D and 3D generation [87]. However, these models are largely built on VLM-based architectures, limiting their native 3D modeling capacity, and their notion of “unity” centers on merging different output modalities (e.g., text, image, and 3D) rather than different 3D creation sub-tasks (generation, editing, and completion). They also typically condition on only one or two modalities, and those that support editing stil rely on large-scale paired editing supervision.

Compared with all three lines of work above, SeMo3D offers a distinct solution: it recasts 3D generation, editing, and completion as a single selective-modeling problem formulated directly on native 3D representations, unifying the three tasks without task-specific architectures or per-task paired supervision while preserving the remaining content by construction. Additionally, SeMo3D supports joint image, text, and point conditioning, enhancing controllability.

## 3. Methodology

## 3.1. Preliminaries

Structured Latents. Following Trellis2 [78], we represent 3D assets as spatially structured latent tokens through a three-stage encoding. The first stage, sparse structure (SS), encodes a 3D asset into a dense latent grid $\mathbf Z ^ { \mathrm { s s } } \in \mathbb R ^ { N ^ { 3 } \times d _ { s } }$ covering all $N ^ { 3 }$ voxels, from which an SS decoder recovers the occupied voxel set O. The second and third stages, collectively termed structured latent (SLat) stages, produce sparse tokens only at positions in O: shape tokens $\mathbf { Z } ^ { \mathrm { s h a p e } } \in \mathbb { R } ^ { | \mathcal { O } | \times d _ { g } }$ encoding fine geometry, and texture tokens ${ \bf Z } ^ { \mathrm { t e x } } \in \mathbb { R } ^ { | \mathcal { O } | \times d _ { t } }$ encoding PBR materials conditioned on $\mathbf { Z } ^ { \mathrm { s h a p e } }$ . Each token at every stage corresponds to a localized spatial region, making the representation amenable to spatially selective operations.

![](images/0f2fa9cad075087012cda53ab76c2e18e4d7557c87b94fe7c65302760e6654fe.jpg)  
Figure 3. Overview of SeMo3D. (a) Region-selective denoising. Training applies noise and loss only to a randomly selected set S while keeping the context set C clean; as S ranges from a local box to the full region, one procedure covers editing, completion, and generation. Inference denoises S from pure noise with a task-defined partition while reusing clean C at every step. (b) Omni3DiT jointly processes clean context and noisy selected tokens, fusing image guidance with optional text and point conditions for fine-grained controllable synthesis.

Rectified Flow Matching. Each stage employs a rectified flow model [39, 41] to generate its respective latent tokens. Given data $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , the forward process linearly interpolates between them as $\pmb { x } _ { t } = \left( 1 - t \right) \pmb { x } _ { 0 } + t \pmb { \epsilon }$ for $t \in [ 0 , 1 ]$ . A velocity network ${ \pmb v } _ { \theta }$ is trained by minimizing the conditional flow matching (CFM) objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } = \mathbb { E } _ { t , { \boldsymbol { x } } _ { 0 } , { \epsilon } } \| { \boldsymbol { v } } _ { \boldsymbol { \theta } } ( { \boldsymbol { x } } _ { t } , t , { \boldsymbol { c } } ) - ( { \epsilon } - { \boldsymbol { x } } _ { 0 } ) \| _ { 2 } ^ { 2 } , } \end{array}\tag{1}
$$

where c denotes the input conditions, and samples are drawn by integrating the learned ODE from t=1 (pure noise) to t=0 (data). All three stages share this formulation; they differ only in $\pmb { x } _ { 0 } ( \mathbf { Z } ^ { \mathrm { s s } } , \mathbf { Z } ^ { \mathrm { s h a p e } }$ , or $\mathbf { Z } ^ { \mathrm { t e x } }$ , respectively) and in the model that operates on each.

## 3.2. Overview

Fig. 3 illustrates the overall framework. For a given stage, let $\mathbf { Z } = \{ z _ { i } \} _ { i \in \mathcal { I } }$ denote its structured latent tokens, where I indexes all token positions (all $N ^ { 3 }$ voxels for SS; the occupied set O for SLat). We partition I into a selected set S (to be synthesized) and a context set $\mathcal { C }$ (to be preserved), where $S \cup { \mathcal { C } } = { \mathcal { I } }$ and ${ \mathcal { S } } \cap { \mathcal { C } } = \emptyset$ . The model denoises only the tokens in S while keeping tokens in $\mathcal { C }$ clean. The three tasks differ solely in how this partition is defined: generation selects the full region $\scriptstyle ( S = T , C = 0 )$ ; editing, the target region; and completion, the missing region.

SeMo3D instantiates this paradigm at each of the three stages, with a dedicated flow model per stage. For training, region-selective denoising (Sec. 3.3, Fig. 3(a)) randomly partitions tokens of ordinary 3D assets and applies noise and loss only to S, so one procedure spans all three tasks without per-task paired supervision. For the architecture, Omni3DiT (Sec. 3.4, Fig. 3(b)) jointly processes clean context tokens and noisy selected tokens, fusing image, text, and point guidance for controllable synthesis. At inference, image guidance is required; text and sparse points are independently optional.

## 3.3. Region-Selective Denoising

As shown in Fig. 3(a), unlike the standard objective (Eq. (1)) that treats all tokens uniformly, region-selective denoising applies noise and loss only to the selected set S while keeping context tokens in C intact. We describe the training and inference procedures below.

Training. Given the tokens Z of an ordinary 3D asset and a partition $( \mathcal { S } , \mathcal { C } )$ , we construct a mixed input $\tilde { \boldsymbol { z } } _ { t }$ by adding noise only to tokens in S while keeping tokens in C clean:

$$
\tilde { z } _ { t , i } = \left\{ \begin{array} { l l } { ( 1 - t ) z _ { 0 , i } + t \epsilon _ { i } , } & { i \in \mathcal { S } , } \\ { z _ { 0 , i } , } & { i \in \mathcal { C } , } \end{array} \right.\tag{2}
$$

where $z _ { 0 , i }$ is the clean token at position i. The model takes $\tilde { \boldsymbol { z } } _ { t }$ together with a binary mask distinguishing S from ${ \mathcal { C } } ,$ and predicts the velocity field over all positions. The loss is

![](images/8af92b33701930ffd8c3c4a4fd8e1cb412ad07bd51f0763a8f4226c337081c19.jpg)  
Figure 4. Agentic 3D modeling with SeMo3D. Agent prepares input images, determines edit regions, and invokes the appropriate SeMo3D mode in response to requests. Left: generation then editing. Right: completion then editing. Refer to Supp. Mat. for video results.

computed only over S; predictions on C are discarded:

$$
\mathcal { L } _ { \mathrm { R S D } } = \mathbb { E } _ { t , z _ { 0 } , \epsilon , \mathcal { S } } \frac { 1 } { | \mathcal { S } | } \sum _ { i \in \mathcal { S } } \lVert \pmb { v } _ { \boldsymbol { \theta } } ( \tilde { z } _ { t } , t , \pmb { c } ) _ { i } - ( \epsilon _ { i } - z _ { 0 , i } ) \rVert _ { 2 } ^ { 2 } .\tag{3}
$$

Here, $1 / | S |$ normalizes the loss across selected-set sizes. Each step samples one of two partition modes with equal probability: (1)full-region mode (generation) selects all positions $( S { = } T , \mathcal { C } { = } \emptyset )$ , reducing Eq. (3) to the CFM objective (Eq. (1)) scaled by $1 / | \mathcal { T } | ; ( 2 )$ local-region mode (editing/completion) selects tokens within a random 3D box.

Inference. Tokens in S start from pure noise and are denoised via ODE integration; tokens in C stay clean. The partition is task-dependent: for generation, the full region is selected (S=I); for editing, the user provides a mask of the target region; for completion, the missing region is either user-specified or derived as the complement of the partial input’s occupancy; as the SS stage decides occupancy within S, only the actually missing geometry is synthesized. The three stages run sequentially: the SS stage denoises S to produce O; the SLat stages inherit the same partition, restricted to O, for shape and texture synthesis.

Benefiting from training regions ranging from a small portion to the full region, the model covers local editing, partial completion, and full generation in one framework, and, though trained only with boxes, handles free-form target and missing regions (Sec. 4.2). Tokens in C are never regenerated, guaranteeing source consistency by construction. Moreover, all three tasks are learned from standard 3D assets without costly per-task paired supervision.

## 3.4. Omni3DiT

Omni3DiT extends Trellis2’s three-stage imageconditioned 3D DiT [50] with four key designs to enable selective modeling with optional text and point guidance (Fig. 3(b)): role embedding, text conditioning, point conditioning, and compositional condition dropout.

Role Embedding. In selective modeling, z˜ (Eq. (2)) mixes tokens at different noise levels: tokens in S carry noise at timestep t, while tokens in C remain clean. To help the model distinguish noisy tokens from clean ones, we encode the binary mask of Sec. 3.3 as a learned role embedding added after the input projection: tokens in S receive $\boldsymbol { e } _ { \mathcal { S } } ~ \in ~ \mathbb { R } ^ { d }$ and tokens in C receive $\boldsymbol { e } _ { \mathcal { C } } ~ \in ~ \mathbb { R } ^ { d }$ , where d is the transformer hidden width. Both embeddings are zeroinitialized to preserve pretrained behavior at initialization.

Text Conditioning. Trellis2 conditions only on images via DINOv3 [60] cross-attention, yet images cannot describe occluded content. We add a parallel cross-attention text branch [7, 9] injecting CLIP [53] features. Both branches share one query and sum their outputs into a residual stream. The text branch output projection is zero-initialized, leaving the pretrained pathway initially undisturbed.

Point Conditioning. Image and text provide rich visual and semantic guidance but cannot specify exact 3D locations. We therefore add points as a third condition, using sparse rather than dense points since they make user interaction far easier. In the SS stage, each point is represented by a shared learnable embedding, positioned via RoPE [64] at its voxel coordinate, and concatenated into the token sequence [34]. Since points share the 3D space of latent tokens, self-attention enables direct spatial reasoning between points and latent tokens, proving substantially more effective than point cross-attention (Sec. 4.3). The SLat stages omit point conditioning, as image and text suffice for geometry and texture once the sparse structure is determined.

Compositional Condition Dropout. Naively finetuning with all conditions active risks letting the pretrained image branch dominate, suppressing the newly added text and point pathways. At each training step, a random subset of modalities is kept active while the rest are replaced with null embeddings. The probabilities are set so that the image condition is frequently withheld, compelling the text and point branches to independently carry the full guidance load.

![](images/376a76c590b58bda7e509b21be104920b52ff20d699f87ac009b15199be34b00.jpg)  
Figure 5. Qualitative comparison of 3D generation on Toys4K [63]. Only top-performing baselines are shown for brevity. HY3D-Omni [29] <sup>“Stylized</sup> <sup>dinosaur</sup> <sub>with</sub> <sub>uneven</sub> <sub>eyes,</sub> textures are from HY3D-2.1 [28]. With three complementary modalities, SeMo3D achieves more controllable and faithful generation.

All new modules are zero-initialized or additive, largely <sup>“Polished</sup> <sup>dark</sup> preserving pretrained behavior at initialization for stable finetuning. Together with region-selective denoising, these components enable Omni3DiT to unify generation, editing, and completion under flexible image, text, and point guidance, as validated by ablation studies in Sec. 4.3.

## 4. Experiments

Built on these components, SeMo3D supports agentic 3D creation through successive natural-language instructions when integrated with an agent (Fig. 4). We report key settings and results here; further details are in the Supp. Mat.

## 4.1. Experimental Settings

Implementation Details. We reuse Trellis2’s pretrained SC-VAE [78] and finetune its three-stage flow models at 512 resolution on Trellis-500K [77] with learning rate $5 \times 1 0 ^ { - 5 }$ and batch size 32. The SS and SLat stages are finetuned for 200K and 100K steps, respectively. Each training sample pairs one asset with a randomly chosen rendered view and dataset caption as its image and text conditions, with points sampled from the asset’s geometry. Generating a 512-resolution asset takes ∼5 s, on par with Trellis2. For agentic modeling, GPT-6 Astra [48] with Codex uses GPT-Image to prepare reference images, determines edit regions, and invokes the appropriate SeMo3D mode.

Baselines. We compare with leading methods per task: (1) 3D generation: Trellis [77], ShapeLLM-Omni [88], HY3D-2.1 [28], Trellis2 [78], Pixal3D [32], TIGON [7], and HY3D-Omni [29], spanning image, text, image-text, and image-point inputs; (2) 3D editing: training-free Vox-Hammer [33] and Nano3D [89], and training-based 3DEditFormer [76] and PartFlow [72]; (3) 3D completion: LaS-Comp [82] and point-guided HY3D-Omni [29] fed partial inputs as points. Baselines receive all supported conditions. Benchmarks. We evaluate on the public benchmarks Toys4K [63], Edit3D-Bench [33], and Omni-Comp [82] for the three tasks, respectively. For generation, we report LPIPS [96], FID [22], KID [5], DINO similarity [49], and Chamfer Distance (CD). For editing, we assess region consistency with CD, PSNR, SSIM [68], and LPIPS; perceptual quality with FID and FVD; and accuracy with DINO and CLIP similarity. For completion, we report CD, Earth Mover’s Distance (EMD), and F-Score (at a 0.01 threshold). Evaluation Details. For variable-resolution methods, outputs are standardized to 512. For editing, mask-based methods receive the same ground-truth masks for fair comparison. For completion, SeMo3D derives its selected region by inverting the partial input’s occupancy mask (Sec. 3.3). Conditions are sampled from the ground truth as in training.

Table 1. Quantitative results on 3D generation. Best and secondbest in bold and underlined. Cond.: I = Image, T = Text, P = Point.
<table><tr><td>Method</td><td>Cond.</td><td>LPIPS</td><td>FID↓</td><td>KID (× 100) ↓</td><td>DINO ↑</td><td>CD↓</td></tr><tr><td>Trellis [77]</td><td>I</td><td>0.133</td><td>6.65</td><td>0.121</td><td>0.888</td><td>0.0413</td></tr><tr><td>Trellis [77]</td><td>T</td><td>0.183</td><td>15.48</td><td>0.202</td><td>0.795</td><td>0.0736</td></tr><tr><td>ShapeLLM-Omni [88]</td><td>I</td><td>0.167</td><td>27.15</td><td>1.167</td><td>0.751</td><td>0.0820</td></tr><tr><td>ShapeLLM-Omni [88]</td><td>T</td><td>0.185</td><td>27.30</td><td>0.836</td><td>0.760</td><td>0.0666</td></tr><tr><td>HY3D-2.1 [28]</td><td></td><td>0.143</td><td>5.68</td><td>0.062</td><td>0.869</td><td>0.0638</td></tr><tr><td>Trellis2 [78]</td><td>1ll</td><td>0.111</td><td>5.05</td><td>0.057</td><td>0.908</td><td>0.0330</td></tr><tr><td>Pixal3D [32]</td><td></td><td>0.143</td><td>7.56</td><td>0.174</td><td>0.876</td><td>0.0453</td></tr><tr><td>TIGON [7]</td><td>I, T</td><td>0.127</td><td>5.90</td><td>0.092</td><td>0.897</td><td>0.0429</td></tr><tr><td>HY3D-Omni [29]</td><td>I, P</td><td>0.121</td><td>7.18</td><td>0.107</td><td>0.879</td><td>0.0317</td></tr><tr><td>SeMo3D (Ours)</td><td>I, T, P</td><td>0.079</td><td>4.99</td><td>0.059</td><td>0.926</td><td>0.0118</td></tr></table>

## 4.2. Evaluation

3D Generation. Fig. 5 and Tab. 1 present comparisons on Toys4K. SeMo3D achieves the best LPIPS, FID, DINO, and CD, with the second-best KID, surpassing image-only, textonly, and dual-condition baselines. Fig. 5 shows why: for the cake (row 1), text-only Trellis produces the tiers and candles but misses the reference appearance; image-only methods match the appearance but miss the tiers hidden from view, while SeMo3D recovers the hidden tiers with points fixing the layout. Images thus specify appearance, text semantics, and points spatial layout, together producing more faithful results than any single modality.

3D Editing. Fig. 6 and Tab. 2 show editing comparisons on Edit3D-Bench. SeMo3D leads across all three metric groups. In region consistency (CD, PSNR, SSIM, LPIPS),

Image

![](images/81d7267fbd57f304631439ccd042d9e5ddf3b021c1f8e20ac2fc129e521912b0.jpg)

Text

Source

“A bedroom set with a bookshelf.”

SeMo3D

Nano3D

VoxHammer-I VoxHammer-T

3DEditFormer

![](images/390ec0cb2d65783e5d65c5da7f20e0a9b0d786d6a025531b587d541dd8290eb2.jpg)

PartFlow

“A little girl riding a motorcycle with a pink umbrella.”

![](images/1a9422c2133e5a19a0da1514c9e69abc5c1ab41b7dbed3ea8a4df356d806cfcb.jpg)

![](images/950bace59ae3f53e755bdc5ebe4a4c29bff5eaf65102ba999da20c5d2403178e.jpg)  
Figure 6. Qualitative comparison of 3D editing on Edit3D-Bench [33]. Translucent volumes mark edit regions; points are omitted as Edit3D-Bench lacks them. Without paired editing data, SeMo3D precisely modifies the target region while preserving the rest.

“A man with a fox on his shoulder.”

![](images/44990e258ae3d551761e333ac939949b7431d3e9631b58ad52ea475c36101166.jpg)  
Figure 7. Qualitative comparison of 3D completion on Omni-Comp [82]. Without any paired completion data, SeMo3D completes missing regions with coherent geometry that blends seamlessly with the preserved structure. <sup>†</sup>With additional sparse point guidance.

Table 2. Quantitative results on 3D editing. Cond. as in Tab. 1. Edit3D-Bench [33] does not provide point conditions.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Cond.</td><td colspan="4">Consistency</td><td colspan="2">Quality</td><td colspan="2">Accuracy</td></tr><tr><td>CD↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td><td>DINO↑</td><td>CLIP↑</td></tr><tr><td>VoxHammer [33]</td><td></td><td>0.0111</td><td>31.21</td><td>0.973</td><td>0.038</td><td>26.35</td><td>586.1</td><td>0.869</td><td>0.281</td></tr><tr><td>VoxHammer [33]</td><td></td><td>0.0092</td><td>31.55</td><td>0.976</td><td>0.033</td><td>23.69</td><td>339.2</td><td>0.833</td><td>0.271</td></tr><tr><td>Nano3D [89]</td><td>ITIII</td><td>0.0101</td><td>31.42</td><td>0.975</td><td>0.035</td><td>24.12</td><td>388.6</td><td>0.861</td><td>0.280</td></tr><tr><td>3DEditFormer [76]</td><td></td><td>0.0213</td><td>27.44</td><td>0.948</td><td>0.061</td><td>28.03</td><td>572.4</td><td>0.858</td><td>0.278</td></tr><tr><td>PartFlow [72]</td><td></td><td>0.0153</td><td>30.13</td><td>0.967</td><td>0.044</td><td>25.33</td><td>437.7</td><td>0.858</td><td>0.277</td></tr><tr><td>SeMo3D (Ours)</td><td>I, T</td><td>0.0088</td><td>37.39</td><td>0.986</td><td>0.028</td><td>22.46</td><td>113.6</td><td>0.875</td><td>0.286</td></tr></table>

the advantage stems from keeping tokens in C clean by construction, guaranteeing rather than approximating source fidelity: in Fig. 6, the girl’s umbrella (row 2) and the man’s head (row 3) remain faithful to the source in SeMo3D, whereas baselines noticeably deviate in these unedited regions. Multi-condition guidance further improves semantic accuracy (DINO, CLIP). SeMo3D achieves these results without paired editing data, learning editing entirely from ordinary 3D assets—a practical advantage over data-hungry methods like 3DEditFormer [76] and PartFlow [72].

3D Completion. Fig. 7 and Tab. 3 report completion results on Omni-Comp. SeMo3D achieves the best CD, EMD, and F-Score. Even without additional point guidance, SeMo3D outperforms all compared methods across all three completion metrics. Unlike point-guided HY3D-Omni, which regenerates whole shapes, SeMo3D preserves existing geometry, synthesizing only the missing part, despite its irregular, non-box shape. In Fig. 7, completed regions blend seamlessly with the existing structure. No paired completion data is used; SeMo3D derives this capability purely from region-selective denoising on standard 3D assets, demonstrating the generality of selective modeling.

Table 3. Quantitative results on 3D completion. Cond. as in Tab. 1. <sup>†</sup>With additional sparse point guidance.
<table><tr><td>Method</td><td>Cond.</td><td>CD↓</td><td>EMD↓</td><td>F-Score ↑</td></tr><tr><td>LaS-Comp [82]</td><td>I</td><td>0.0226</td><td>0.0375</td><td>0.359</td></tr><tr><td>LaS-Comp [82]</td><td>T</td><td>0.0313</td><td>0.0489</td><td>0.335</td></tr><tr><td>HY3D-Omni [29]</td><td>I, P</td><td>0.0360</td><td>0.0559</td><td>0.347</td></tr><tr><td>HY3D-Omni† [29]</td><td>I, P</td><td>0.0270</td><td>0.0464</td><td>0.412</td></tr><tr><td>SeMo3D (Ours)</td><td>I, T</td><td>0.0208</td><td>0.0346</td><td>0.468</td></tr><tr><td>SeMo3D† (Ours)</td><td>I, T, P</td><td>0.0133</td><td>0.0223</td><td>0.541</td></tr></table>

## 4.3. Ablation Study

Multi-Conditioning. We ablate the optional text and point conditions in Tab. 4 (i)–(iv) and Fig. 8(a). Starting from image-only input (i), adding text (ii) improves perceptual quality and semantic fidelity (FID, DINO) in generation and editing but barely changes geometry: the creature gains eyes and a tail, yet its body stays a flat box. Adding points instead (iii) yields larger gains in geometry (CD, EMD, F-

Table 4. Quantitative ablation. (i)–(iv): condition combinations; (v)–(vii): full-model component ablations. Editing consistency metrics are omitted given identical context tokens across variants. Edit3D-Bench lacks points, so (iii)–(iv) omit editing and (v)–(vii) edit with I, T.
<table><tr><td rowspan="2"></td><td rowspan="2">Configuration</td><td rowspan="2">Cond.</td><td colspan="5">Generation</td><td colspan="4">Editing</td><td colspan="3">Completion</td></tr><tr><td>LPIPS ↓ FID↓</td><td></td><td>KID (×100) ↓</td><td>DINO ↑</td><td>CD↓</td><td>FID ↓</td><td>FVD↓</td><td>DINO↑</td><td>CLIP↑</td><td>CD↓</td><td>EMD↓</td><td>F-Score ↑</td></tr><tr><td>(i)</td><td>SeMo3D (Ours)</td><td>I</td><td>0.113</td><td>5.49</td><td>0.074</td><td>0.904</td><td>0.0334</td><td>22.53</td><td>118.2</td><td>0.873</td><td>0.284</td><td>0.0207</td><td>0.0346</td><td>0.471</td></tr><tr><td>(ii)</td><td>SeMo3D (Ours)</td><td>I, T</td><td>0.112</td><td>5.34</td><td>0.069</td><td>0.911</td><td>0.0326</td><td>22.46</td><td>113.6</td><td>0.875</td><td>0.286</td><td>0.0208</td><td>0.0346</td><td>0.468</td></tr><tr><td>(iii)</td><td>SeMo3D (Ours)</td><td>I, P</td><td>0.082</td><td>5.06</td><td>0.065</td><td>0.926</td><td>0.0116</td><td>1</td><td></td><td>1</td><td>一</td><td>0.0134</td><td>0.0222</td><td>0.542</td></tr><tr><td>(iv)</td><td>SeMo3D (Ours)</td><td>I, T, P</td><td>0.079</td><td>4.99</td><td>0.059</td><td>0.926</td><td>0.0118</td><td></td><td></td><td>一</td><td>=</td><td>0.0133</td><td>0.0223</td><td>0.541</td></tr><tr><td>(v)</td><td>w/o Role Emb.</td><td>I, T, P</td><td>0.078</td><td>5.01</td><td>0.057</td><td>0.925</td><td>0.0118</td><td>22.52</td><td>114.2</td><td>0.866</td><td>0.278</td><td>0.0156</td><td>0.0249</td><td>0.528</td></tr><tr><td>(vi)</td><td>Point via Cross-Attn.</td><td>I, T, P</td><td>0.102</td><td>5.32</td><td>0.066</td><td>0.917</td><td>0.0244</td><td>22.47</td><td>112.9</td><td>0.875</td><td>0.285</td><td>0.0178</td><td>0.0276</td><td>0.499</td></tr><tr><td>(vii)</td><td>Standard CFG Dropout</td><td>I, T, P</td><td>0.110</td><td>5.44</td><td>0.072</td><td>0.902</td><td>0.0331</td><td>22.51</td><td>116.9</td><td>0.872</td><td>0.283</td><td>0.0201</td><td>0.0350</td><td>0.467</td></tr></table>

![](images/81c7cb7dbf5ef4d5ef6285ffed611ffec5fde69a0e913b8530b294df887e51ac.jpg)  
Figure 8. (a) Qualitative ablation. (b) Multimodal control: with a fixed image, varying text and points yields diverse yet faithful results.

![](images/59bdb7eccdcf3cfc8dead0bc996d133cf2c71e621dc1648e387b41e771bd4a27.jpg)  
“Grey seat, gold Figure 9. Agentic 3D modeling with GPT-6 Astra [48] and Codex: <sup>frame.”</sup>(a) direct Blender coding [17]; (b) SeMo3D integration, yielding higher modeling fidelity, faster execution, and lower agent cost.

Score): the creature’s body better matches the point layout. The full model (iv) achieves the best perceptual quality and semantic fidelity while matching (iii) in geometry, combining visual, semantic, and spatial cues no single modality provides. Fig. 8(b) shows that, with a fixed image, varying text and points yields a sofa, suitcase, basket, or ottoman in one case and a tractor, rover, truck, or ATV in the other, demonstrating fine-grained multimodal control. Role Embedding. Removing the role embedding (v) barely affects generation, where all tokens are selected, but degrades editing and completion (Tab. 4), as distinguishing clean context from noisy tokens becomes harder. Point Embedding vs. Cross-Attention. Injecting points via crossattention (vi) doubles the generation CD and degrades completion (Tab. 4); the creature in Fig. 8(a) elongates, ignoring the point layout, which confirms that self-attention integrates points more effectively. Compositional Condition Dropout. Replacing compositional condition dropout with standard Classifier-Free Guidance [23] (CFG) dropout (vii), which retains either all available conditions or none, weakens text and point guidance: metrics nearly revert to imageonly (i), and the output in Fig. 8(a) has an elongated body, disregarding text and points. SeMo3D Integration. Compared with direct Blender coding [17], equipping the same agent with SeMo3D improves modeling fidelity while reducing execution time and agent cost (Fig. 9).

![](images/b725159e4ec36404683798c640c8e540bd42143d72d29c32d275e1c342b13815.jpg)  
Figure 10. Failure cases: coarse fronds and missing fruits.

## 5. Conclusion

We present SeMo3D, a unified framework that casts 3D generation, editing, and completion as selective modeling— synthesizing only the designated region while preserving the rest. Around this core idea, we design region-selective denoising and Omni3DiT, enabling one model to handle all three tasks without task-specific paired data while accepting complementary image, text, and point guidance. Experiments show that SeMo3D surpasses specialized stateof-the-art methods across all three tasks. Agent integration further enables an intuitive, efficient 3D creation workflow. Limitations and Future Work. SeMo3D inherits Trellis2’s structured latents [78], whose limited resolution cannot capture very fine structures such as thin palm fronds (Fig. 10, top); higher-fidelity 3D representations [38, 58, 65] could alleviate this. Moreover, the CLIP [53] text encoder struggles with complex compositional prompts: the orange fruits are reduced to orange texture rather than generated as objects (Fig. 10, bottom); stronger visionlanguage encoders [2, 83] are a promising remedy.

## 6. LLM Usage Statement

Codex was used throughout the manuscript for language polishing, concision, and grammar checking. It was also used to obtain reviewer-style feedback, assist with code development, organize experimental results, and check references. GPT-Image was used to generate and edit reference images for the agentic 3D modeling experiments. In these experiments, GPT-6 Astra with Codex served as the agent, preparing inputs, determining edit regions, invoking SeMo3D, and generating code for the direct Blender baseline. The authors reviewed all AI-assisted content and take full responsibility for the scientific content, code, experimental results, references, and conclusions.

## References

[1] Sarker M Asish, Bhoj B Karki, Bharat KC, Niloofar Kolahchi, and Shaon Sutradhar. Synthesizing six years of ar/vr research: A systematic review of machine and deep learning applications. In Proc. IEEE Conf. Virtual Reality, 2025. 2

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 9

[3] Roi Bar-On, Dana Cohen-Bar, and Daniel Cohen-Or. Editp23: 3d editing via propagation of image prompts to multi-view. arXiv preprint arXiv:2506.20652, 2025. 3

[4] Amir Barda, Matheus Gadelha, Vladimir G Kim, Noam Aigerman, Amit H Bermano, and Thibault Groueix. Instant3dit: Multiview inpainting for fast editing of 3d objects. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2025. 3

[5] Mikołaj Binkowski, Danica J Sutherland, Michael Arbel,´ and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 6, 13

[6] Weiwei Cai, Shuangkang Fang, Weicai Ye, Xin Dong, Yunhan Yang, Xuanyang Zhang, Wei Cheng, Yanpei Cao, Gang Yu, and Tao Chen. Native 3d editing with full attention. arXiv preprint arXiv:2511.17501, 2025. 3

[7] Jiazhong Cen, Jiemin Fang, Sikuang Li, Guanjun Wu, Chen Yang, Taoran Yi, Zanwei Zhou, Zhikuan Bao, Lingxi Xie, Wei Shen, et al. Text-image conditioned 3d generation. arXiv preprint arXiv:2603.21295, 2026. 3, 5, 6

[8] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proc. IEEE Int. Conf. Comput. Vis., 2023. 3

[9] Wenyue Chen, Wenjue Chen, Peng Li, Qinghe Wang, Xu Jia, Heliang Zheng, Rongfei Jia, Yuan Liu, and Ronggang Wang. Know3d: Prompting 3d generation with knowledge from vision-language models. arXiv preprint arXiv:2603.22782, 2026. 3, 5

[10] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. Gaussianeditor: Swift and con trollable 3d editing with gaussian splatting. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2024. 3

[11] Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Nießner, Chi-Wing Fu, and Jiaya Jia. Diffcomplete: Diffusion-based generative 3d shape completion. In Adv. Neural Inf. Process. Syst., 2023. 2, 3

[12] Yuval Cohen, Amir Biton, and Shraga Shoval. Fusion of computer vision and ai in collaborative robotics: A review and future prospects. Applied Sciences, 2025. 2

[13] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. In Adv. Neu ral Inf. Process. Syst., 2023. 2

[14] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2023. 2

[15] Zhenjiang Du, Yan Zhang, Zhitao Liu, Guan Wang, Zeyu Ma, Ning Xie, and Yang Yang. Sgcdiff: Sketch-guided cross-modal diffusion model for 3d shape completion. Neu rocomputing, 2025. 3

[16] Ziya Erkoc¸, Can Gumeli, Chaoyang Wang, Matthias¨ Nießner, Angela Dai, Peter Wonka, Hsin-Ying Lee, and Peiye Zhuang. Preditor3d: Fast and precise 3d shape editing. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2025. 3

[17] Blender Foundation. Blender - a 3d modelling and rendering software. https://www.blender.org/, 2026. 8, 15

[18] Will Gao, Dilin Wang, Yuchen Fan, Aljaz Bozic, Tuur Stuyck, Zhengqin Li, Zhao Dong, Rakesh Ranjan, and Nikolaos Sarafianos. 3d mesh editing using masked lrms. In Proc. IEEE Int. Conf. Comput. Vis., 2025. 3

[19] Inbar Gat, Dana Cohen Bar, Guy Levy, Elad Richardson, and Daniel Cohen Or. Shapeup: Scalable imageconditioned 3d editing. In ACM SIGGRAPH, 2026. 2, 3

[20] Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution

and arbitrary-topology 3d shape modeling. arXiv preprint arXiv:2503.21732, 2025. 3

[21] Yingqing He, Zhaoyang Liu, Jingye Chen, Zeyue Tian, Hongyu Liu, Xiaowei Chi, Runtao Liu, Ruibin Yuan, Yazhou Xing, Wenhai Wang, et al. Llms meet multimodal generation and editing: A survey. arXiv preprint arXiv:2405.19334, 2024. 2

[22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Adv. Neural Inf. Process. Syst., 2017. 6, 13

[23] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 8, 13

[24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Adv. Neural Inf. Process. Syst., 2020. 2, 3

[25] Anita Hu and Maria Shugrina. Axolotl3d: a unified framework for faithful 3d shape completion. arXiv preprint arXiv:2607.20660, 2026. 2, 3

[26] Shimin Hu, Yuanyi Wei, Fei Zha, Yudong Guo, and Juyong Zhang. Easy3e: Feed-forward 3d asset editing via rectified voxel flow. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2026. 3

[27] Tianxin Huang, Zhiwen Yan, Yuyang Zhao, and Gim H Lee. Compc: completing a 3d point cloud with 2d diffusion priors. In Proc. Int. Conf. Learn. Represent., 2025. 3

[28] Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, et al. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025. 3, 6

[29] Team Hunyuan3D, Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Linus, Penghao Wang, et al. Hunyuan3domni: A unified framework for controllable generation of 3d assets. arXiv preprint arXiv:2509.21245, 2025. 3, 6, 7, 14

[30] Yoni Kasten, Ohad Rahamim, and Gal Chechik. Point cloud completion with pretrained text-to-image diffusion models. In Adv. Neural Inf. Process. Syst., 2023. 3

[31] An Li, Zhe Zhu, and Mingqiang Wei. Genpc: zero-shot point cloud completion via 3d generative priors. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2025. 3

[32] Dong-Yang Li, Wang Zhao, Yuxin Chen, Wenbo Hu, Meng-Hao Guo, Fang-Lue Zhang, Ying Shan, and Shi-Min Hu. Pixal3d: Pixel-aligned 3d generation from images. arXiv preprint arXiv:2605.10922, 2026. 3, 6

[33] Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, and Lu Sheng. Voxhammer: Training-free precise and coherent 3d editing in native 3d space. arXiv preprint arXiv:2508.19247, 2025. 3, 6, 7, 13, 14, 20

[34] Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, et al. Segvigen: Repurposing 3d generative model for part segmentation. ACM Trans. Graph., 2026. 5

[35] Peng Li, Suizhi Ma, Jialiang Chen, Yuan Liu, Congyi Zhang, Wei Xue, Wenhan Luo, Alla Sheffer, Wenping Wang, and Yike Guo. Cmd: Controllable multiview diffusion for 3d editing and progressive generation. In ACM SIGGRAPH, 2025. 3

[36] Xiaoyu Li, Qi Zhang, Di Kang, Weihao Cheng, Yiming Gao, Jingbo Zhang, Zhihao Liang, Jing Liao, Yan-Pei Cao, and Ying Shan. Advances in 3d generation: A survey. arXiv preprint arXiv:2401.17807, 2024. 2

[37] Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Text-driven 3d editing via focal-fusion assembly. In Proc. AAAI Conf. Artif. Intell., 2024. 3

[38] Yuan Li, Congyi Zhang, Xifeng Gao, and Xiaohu Guo. Supervoxelgpt: Adaptive and ordered 3d tokenization for autoregressive shape generation. In Proc. Eur. Conf. Comput. Vis., 2026. 9

[39] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4

[40] Jian Liu, Xiaoshui Huang, Tianyu Huang, Lu Chen, Yuenan Hou, Shixiang Tang, Ziwei Liu, Wanli Ouyang, Wangmeng Zuo, Junjun Jiang, et al. A comprehensive survey on 3d content generation. arXiv preprint arXiv:2402.01166, 2024. 2

[41] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2, 3, 4

[42] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 3

[43] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2024. 3

[44] Julio Castro Lopes and Rui Pedro Lopes. Computer vision´ in augmented, virtual, mixed and extended reality environments—a bibliometric review. Visual Informatics, 2024. 2

[45] Zhaoyang Lyu, Zhifeng Kong, Xudong Xu, Liang Pan, and Dahua Lin. A conditional point diffusion-refinement paradigm for 3d point cloud completion. In Proc. Int. Conf. Learn. Represent., 2022. 2, 3

[46] Ziqi Ma, Hongqiao Chen, Yisong Yue, and Georgia Gkioxari. Feedforward 3d editing via text-steerable imageto-3d. arXiv preprint arXiv:2512.13678, 2025. 2, 3

[47] Andrew Melnik, Benjamin Alt, Giang Nguyen, Artur Wilkowski, Maciej Stefanczyk, Qirui Wu, Sinan Harms,´ Helge Rhodin, Manolis Savva, and Michael Beetz. Digita twin generation from visual data: A survey. arXiv preprint arXiv:2504.13159, 2025. 2

[48] OpenAI. GPT-6. https://developers.openai. com/api/docs/models, 2026. 6, 8, 14

[49] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6, 13

[50] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proc. IEEE Int. Conf. Comput. Vis., 2023. 3, 5

[51] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3

[52] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2024. 3

[53] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proc. Int. Conf. Mach. Learn., 2021. 5, 9, 13

[54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image ¨ synthesis with latent diffusion models. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2022. 3

[55] Javier Ruiz-del Solar, Patricio Loncomilla, and Naiomi Soto. A survey on deep learning methods for robot vision. arXiv preprint arXiv:1803.10862, 2018. 2

[56] Simon Schaefer, Juan D Galvis, Xingxing Zuo, and Stefan Leutenegger. Sc-diff: 3d shape completion with latent diffusion models. arXiv preprint arXiv:2403.12470, 2024. 2, 3

[57] Etai Sella, Gal Fiebelman, Peter Hedman, and Hadar Averbuch-Elor. Vox-e: Text-guided voxel editing of 3d objects. In Proc. IEEE Int. Conf. Comput. Vis., 2023. 3

[58] Wenda Shi, Chenyuan Pan, Dengming Zhang, Yiren Song, Biao Zhang, and Xingxing Zou. Anysurf: Any surface generation with directed edge. arXiv preprint arXiv:2605.26149, 2026. 9

[59] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In Proc. Int. Conf. Learn. Represent., 2024. 3

[60] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 5

[61] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3

[62] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3

[63] Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2021. 6, 13, 20

[64] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024. 5

[65] Zhenhong Sun, Haozhe Liu, Yifu Wang, Xibin Song, Senbo Wang, Huadong Mo, Daoyi Dong, Hongdong Li, and Pan Ji. P2voxel: Pyramid pivot voxelization for 3d mesh tok enization. arXiv preprint arXiv:2608.07549, 2026. 9

[66] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2016. 13

[67] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In Proc. Eur. Conf. Comput. Vis., 2024. 3

[68] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 2004. 6, 13

[69] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongx uan Li, Hang Su, and Jun Zhu. Prolificdreamer: Highfidelity and diverse text-to-3d generation with variational score distillation. In Adv. Neural Inf. Process. Syst., 2023. 3

[70] Guangshun Wei, Yuan Feng, Long Ma, Chen Wang, Yuanfeng Zhou, and Changjian Li. Pcdreamer: Point cloud completion through multi-view diffusion priors. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2025. 3

[71] Hao Wen, Weibin Yun, Hongxing Fan, Haotian Lu, Rui Chen, Zehuan Huang, and Lu Sheng. Learning 3d editing without paired supervision via generative prior distillation. arXiv preprint arXiv:2609.04942, 2026. 3

[72] Jiawei Weng, Saining Zhang, Zhenxin Diao, Peishuo Li, Henghaofan Zhang, Junhao Chen, and Hao Zhao. Feedforward 3d editing learns from semantic-part transformation. arXiv preprint arXiv:2605.27351, 2026. 2, 3, 6, 7, 14

[73] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. In Adv. Neural Inf. Process. Syst., 2024. 3

[74] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, et al. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025. 3

[75] Jiatong Xia, Zicheng Duan, Anton van den Hengel, and Lingqiao Liu. Points-to-3d: Structure-aware 3d generation with point cloud priors. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2026. 3

[76] Ruihao Xia, Yang Tang, and Pan Zhou. Towards scalable and consistent 3d editing. arXiv preprint arXiv:2510.02994, 2025. 2, 3, 6, 7, 14

[77] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2025. 2, 3, 6, 13

[78] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2026. 2, 3, 6, 9, 13

[79] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024. 3

[80] Yizhao Xu, Hongyuan Zhu, Caiyun Liu, Tianfu Wang, Keyu Chen, Sicheng Xu, Jiaolong Yang, Nicholas Jing Yuan, and Qi Zhang. Beyond voxel 3d editing: Learning from 3d masks and self-constructed data. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn. Findings, 2026. 3

[81] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, et al. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2024. 14

[82] Weilong Yan, Haipeng Li, Hao Xu, Nianjin Ye, Yihao Ai, Shuaicheng Liu, and Jingyu Hu. Las-comp: Zero-shot 3d completion with latent-spatial consistency. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2026. 3, 6, 7, 13, 21

[83] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 9

[84] Liu Yang and Xiao-qun Wu. 3d shape completion via deep learning: A method survey. Journal ofGraphics, 2023. 2

[85] Zongyuan Yang, Mingjing Yi, Wanli Ma, Chenzhuo Fan, Bocheng Li, Baolin Liu, Yuke Lou, Yingde Song, Yongping Xiong, Zhengdong Guo, et al. Eva01: Unified native 3d understanding and generation via mixture-of-transformers. arXiv preprint arXiv:2605.16745, 2026. 3

[86] Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging. In Proc. IEEE Int. Conf. Comput. Vis., 2025. 3

[87] Chongjie Ye, Cheng Cao, Chuanyu Pan, Yiming Hao, Yihao Zhi, Yuanming Hu, and Xiaoguang Han. Omni123: Exploring 3d native foundation models with limited 3d data by unifying text to 2d and 3d generation. arXiv preprint arXiv:2604.02289, 2026. 3

[88] Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. Shapellm-omni: A native multimodal llm for 3d generation and understanding. arXiv preprint arXiv:2506.01853, 2025. 3, 6

[89] Junliang Ye, Shenghao Xie, Ruowen Zhao, Zhengyi Wang, Hongyu Yan, Wenqiang Zu, Lei Ma, and Jun Zhu. Nano3d: A training-free approach for efficient 3d editing without masks. arXiv preprint arXiv:2510.15019, 2025. 3, 6, 7, 14

[90] Junliang Ye, Zehuan Huang, Yansong Qu, Chunshi Wang, Yunhan Yang, Yang Li, Yawei Luo, Zhuo Chen, Sheng Lu, Jun Zhu, and Chunchao Guo. Universe3d: Emerging properties of unified multimodal models in 3d understanding

and generation. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn. Findings, 2026. 3

[91] Junliang Ye, Kenkun Liu, Guocun Wang, Yang Li, Yansong Qu, Chunshi Wang, Jingwei Xu, Yunhan Yang, Zibo Zhao, Jiachen Xu, et al. Hunyuan3d-buffalo 1.0: A unified multimodal model for scalable 3d generation, understanding, and editing. arXiv preprint arXiv:2608.02711, 2026. 3

[92] Qi-Yue Yin, Jun Yang, Kai-Qi Huang, Mei-Jing Zhao, Wan-Cheng Ni, Bin Liang, Yan Huang, Shu Wu, and Liang Wang. Ai in human-computer gaming: Techniques, challenges and opportunities. Machine intelligence research, 2023. 2

[93] Youtan Yin, Yanning Zhou, Jiacheng Wei, Xiaofeng Yang, Jun Zhang, Jiayang Bai, Jingwen Ye, Weidong Zhang, and Guosheng Lin. Editverse3d: High-quality 3d ob ject editing with region-aware learning. arXiv preprint arXiv:2607.07187, 2026. 2, 3

[94] Biao Zhang, Jiapeng Tang, Matthias Nießner, and Peter Wonka. 3dshape2vecset: A 3d shape representation fo neural fields and generative diffusion models. ACM Trans. Graph., 2023. 2, 3

[95] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Trans. Graph., 2024. 3

[96] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2018. 6, 13

[97] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 3

[98] Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. In Proc. Int. Conf. Learn. Represent., 2024. 14

[99] Zhenglin Zhou, Fan Ma, Chengzhuo Gui, Xiaobo Xia, Hehe Fan, Yi Yang, and Tat-Seng Chua. Anchorflow: Training-free 3d editing via latent anchor-aligned flows. In Proc. IEEE Conf. Comput. Vis. Pattern Recogn., 2026. 3

[100] Jingyu Zhuang, Chen Wang, Liang Lin, Lingjie Liu, and Guanbin Li. Dreameditor: Text-driven 3d scene editing with neural fields. In ACM SIGGRAPH Asia, 2023. 3

[101] Jingyu Zhuang, Di Kang, Yan-Pei Cao, Guanbin Li, Liang Lin, and Ying Shan. Tip-editor: An accurate 3d editor following both text-prompts and image-prompts. ACM Trans. Graph., 2024. 3

# SeMo3D: Unifying 3D Generation, Editing, and Completion via Selective Modeling

Supplementary Material

This supplementary material provides training details (Sec. 7), experimental details (Sec. 8), robustness analysis (Sec. 9), and additional results (Sec. 10).

## 7. Training Details

Training Data. We train on the training split of Trellis-500K [77], filtering out assets with an aesthetic score below 4.5. Each sample pairs one asset with a randomly chosen rendered view as the image condition and a randomly chosen dataset caption as the text condition. No paired editing or completion data is used.

Point Conditions. Points lie on the $1 6 ^ { 3 }$ SS grid, with at most 32 per sample. During training, we first use farthest point sampling (FPS) to obtain 32 spatially distributed points from occupied voxels inside S, then draw k uniformly from 1 to 32 and randomly select k of these points; unused slots are filled with masked padding tokens. This random subsampling improves robustness to variations in the number and spatial distribution of user-provided point conditions (Sec. 9).

Hyperparameters. All three stages are initialized from Trellis2 [78], with settings listed in Tab. 5. We use AdamW with a learning rate of $5 \times 1 0 ^ { - 5 }$ , weight decay of 0.01, an EMA rate of 0.9999, and bfloat16 mixed precision.

Region-Selective Denoising. Full-region and local-region modes are sampled with equal probability. In the localregion mode, we sample a 3D box with random side lengths (Tab. 5) and position, restricted to the half of the asset facing the condition camera so that the selected region is observed by the image. Sparse boxes are rejected, and accepted ones are shrunk to fit their occupied voxels. The SS grid has resolution N=16; at inference, its partition is transferred to the 32<sup>3</sup> SLat grid by nearest-neighbor upsampling.

Compositional Condition Dropout. Each step keeps one condition combination and nulls the rest. For the SS stage, {I, T, P}, {I, T}, and {I, P} are each sampled with probability 0.2, and {I}, {T}, {P}, and ∅ with 0.1 each. For the SLat stages, {I, T}, {I}, {T}, and ∅ are sampled with probabilities 0.5, 0.2, 0.2, and 0.1, respectively. The image is thus withheld in 30% of steps, and ∅ trains the unconditional branch for classifier-free guidance [23]. In the standard CFG dropout ablation, each stage retains all available conditions with probability 0.9 and drops them all with probability 0.1: the SS stage uses only {I, T, P} or ∅, and the SLat stages use only {I, T} or ∅.

Table 5. Per-stage training settings.
<table><tr><td></td><td>SS</td><td>SLat (shape)</td><td>SLat (texture)</td></tr><tr><td>Grid resolution</td><td> $1 6 ^ { 3 }$ </td><td>323</td><td> $3 2 ^ { 3 }$ </td></tr><tr><td>Width / blocks / heads</td><td></td><td>1536 / 30 / 12</td><td></td></tr><tr><td>Conditions</td><td>I, T, P</td><td>I, T</td><td>I, T</td></tr><tr><td>Training steps</td><td>200K</td><td>100K</td><td>100K</td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Box side length (voxels)</td><td>4-16</td><td>8-32</td><td>8-32</td></tr></table>

## 8. Experimental Details

## 8.1. Benchmarks and Metrics

Generation. We use the Toys4K [63] test split of Trellis-500K, which is disjoint from the training split, keeping the 3,180 objects with text annotations (105 categories). Each object comes with one condition image and one caption, and its point condition consists of 32 points sampled by FPS from the asset’s occupancy. Generated and ground-truth meshes are rendered from 8 views at $1 5 ^ { \circ }$ elevation with 45<sup>◦</sup> azimuth spacing. LPIPS [96] and DINOv2 [49] similarity are averaged over view-aligned render pairs, while FID [22] and KID [5] compare the two render sets using Inception-v3 features [66]. CD uses 10K surface points per mesh, with each mesh normalized to a unit box.

Editing. Edit3D-Bench [33] contains 300 editing cases, each with a source asset, a 3D edit region, an edited image, and a text prompt. Edited and source assets are rendered from 16 views at $1 5 ^ { \circ }$ and $3 0 ^ { \circ }$ elevations. Consistency is measured outside the edit region: PSNR, SSIM [68], and LPIPS on unmasked pixels of view-aligned renders, and CD on 20K surface points with those inside the edit region removed. Quality is measured by FID between source and edited renders and by FVD between their 120-frame orbit videos. Accuracy is measured on 6 frontal views by DI-NOv2 similarity to the edited image and CLIP [53] similarity to the prompt.

Completion. Omni-Comp [82] contains 180 partial inputs created by single-view scanning, part removal, and random cropping. For SeMo3D, the point condition consists of 32 points sampled by FPS from ground-truth surface points farther than 0.02 from the partial input. The no-point variant uses the same image and text conditions without these additional points. We sample 16K points from both the completed mesh and the ground truth, and apply the same normalization transform to both so that they remain aligned in a shared coordinate system. CD, EMD, and F-Score at a 0.01 threshold are computed on the whole shape, reflecting

![](images/645523b00bf6597eda3423a851903257df3f7fe957b60df010f2c16342ba2bae.jpg)

Figure 11. ULIP and Uni3D scores disagree with visual quality: Trellis2 is closer to the ground truth than Trellis-I but scores lower.  
![](images/a6372e262b8f97711e31d42cf902aa38811d8d467000b169d96232c3759fa166.jpg)  
Figure 12. Robustness to mask size. SeMo3D yields consistent edits as the edit region (translucent) grows by up to 40%.

S<sup>o</sup>both the completed region and the preserved input.

Why Not 3D Embedding Metrics? ULIP [81] and Uni3D [98] scores do not reliably reflect generation quality. As shown in Fig. 11, Trellis2 is visibly closer to the ground truth than Trellis-I and outperforms it in Tab. 1 of the main paper, yet receives lower scores. We therefore rely on metrics computed against the ground truth.

## 8.2. Baselines and Inference

Baselines. All baselines use their official settings and receive every condition they support, with variable-resolution outputs standardized to 512. For editing, the training-free VoxHammer [33] and Nano3D [89] take the same groundtruth edit region as SeMo3D, whereas the feed-forward 3DEditFormer [76] and PartFlow [72] do not accept masks. For completion, HY3D-Omni [29] takes the partial input as its point condition, and HY3D-Omni<sup>†</sup> additionally receives

![](images/23f6a2093c152e977adff8bfb98295a1807b4c5acd2da8d625ef9ef3bf252e5c.jpg)

![](images/85eb8d32718d357be84ac2f9bcb68f443bd6e6c0664a49ac72c4286a73d044fc.jpg)  
Figure 13. Robustness to the number of condition points. With the image and text fixed, SeMo3D preserves the overall geometry across 16, 32, and 48 points, despite being trained with at most 32.

![](images/f1a654c52030fd2a48495b1fe039dc2b8d76e2c66e124332c955b5d15b63366d.jpg)  
Figure 14. Robustness to the spatial distribution of condition points. With the image and text fixed, SeMo3D produces consistent overall structure across three point distributions, with variations in local details.

sparse point guidance.

SeMo3D Inference. We use an Euler flow sampler with classifier-free guidance. For generation, all stages use 12 steps with guidance scales of 7.5, 7.5, and 1.0 for the SS, shape, and texture stages, respectively. For editing and completion, the SS stage uses 50 steps and the SLat stages use 12, all with a guidance scale of 3.0.

## 8.3. Agentic Interactive 3D Creation

We use GPT-6 Astra [48] with Codex as the agent. Each sequence starts with an image or a partial point cloud, followed by editing requests.

Generation. The agent directly invokes SeMo3D’s genera-

tion mode with the input image to obtain a 3D asset.

Completion. The agent first renders a depth map of the partial point cloud and uses GPT-Image to infer an RGB image of the complete asset. It then invokes SeMo3D’s completion mode with this image and the partial point cloud to recover the full asset.

Editing. Given an editing instruction, the agent uses GPT-Image to produce an edited image and autonomously determines the target region. It then invokes SeMo3D’s editing mode with the edited image, the target-region mask, and the source asset’s cached latent representation to obtain the edited asset. Each output’s latent representation is cached for subsequent edits.

Direct-Coding Baseline. The same agent creates and modifies assets entirely by writing Blender [17] code, without using SeMo3D.

## 9. Robustness

Mask Size. User-specified masks are rarely tight, so we enlarge each Edit3D-Bench edit region by up to 40% about its center. As shown in Fig. 12, SeMo3D yields nearly identical edits across all sizes, without adding spurious geometry. Since the SS stage decides occupancy within S, a looser mask does not force content into it.

Number of Condition Points. We vary the number of condition points across 16, 32, and 48, while keeping the image and text conditions fixed. As shown in Fig. 13, the generated assets retain consistent overall geometry across all three counts. The 48-point results further show generalization beyond the training maximum of 32 points.

Spatial Distribution of Condition Points. We also compare three spatial distributions of condition points for each asset, keeping the image and text conditions fixed. As shown in Fig. 14, SeMo3D preserves the overall structure of the mecha and tricycle across these distributions, while local geometry and appearance may vary. Together, these results indicate robustness to variations in both the number and spatial distribution of point conditions.

## 10. Additional Results

Agentic Interactive 3D Modeling. Using the setup in Sec. 8.3, Figs. 15 and 16 show generation followed by two edits, while Figs. 17 and 18 start with completion. The examples span human characters, animals, fantasy creatures, furniture, and everyday objects, covering accessory addition, part removal and replacement, pose adjustment, and local geometric refinement. Tasks range from replacing the contents of a compass case and adding armor to a creature to modifying the backrest of a completed chair. These results illustrate SeMo3D’s versatility across asset categories and its ability to support successive user-directed changes while maintaining consistency in unedited regions.

3D Generation, Editing, and Completion. Figs. 19 to 21 show additional comparisons. In generation, SeMo3D recovers parts hidden from the input view, such as the stein handle, which most image-only baselines miss. In editing, it modifies only the target region while preserving the rest, as with the rose-holding turtle. In completion, it retains the partial input and fills the missing parts coherently, whereas HY3D-Omni may duplicate structures, such as the extra bicycle frame.

Multimodal Control. Fig. 22 further shows that, with a fixed image, varying text and points produces diverse assets faithful to the image appearance.

Image

Generated 3D

Edit 1

Edit 2

![](images/133c8873cf627812f8cc077df05b78670ff6e1a7f490548fc4cb2ac55f7766c2.jpg)  
Figure 15. Additional agentic interactive 3D modeling: generation followed by two edits.

Generated 3D

Edit 1

Edit 2

![](images/1c67846e7b27c88d8c772b8fbba1fa961e663dae4bdb8f55820c2711778d4c00.jpg)  
Figure 16. Additional agentic interactive 3D modeling: generation followed by two edits.

![](images/0475b5f26a7f6f730d6d0f727d84bf0a5dbb473c0e54d7acc04703e0b40acd7e.jpg)  
Figure 17. Additional agentic interactive 3D modeling: completion followed by two edits.

Partial Input

Completed 3D

Edit 1

Edit 2

![](images/25e9b56f59d32d7d513a251803c79ed5cbabcb8e8f4a8f3f9d0ec978f3536533.jpg)  
Figure 18. Additional agentic interactive 3D modeling: completion followed by two edits.

Image

Point

GT

SeMo3D

Trellis-I

Trellis-T

HY3D-2.1

Trellis2

Pixal3D

TIGON HY3D-Omni

![](images/56adae9268ebf84237ed9dce8f16291e5456f20aad1fc575b10f47f8c26cc442.jpg)

## Figure 19. Additional qualitative comparison of 3D generation on Toys4K [63].

![](images/a3203ec83111f652de5c3c79f8aa3679d296114927959752e70e6213d3048411.jpg)  
Figure 20. Additional qualitative comparison of 3D editing on Edit3D-Bench [33]. Translucent volumes mark the edit regions.

![](images/198cc2213ac53dd81a943a9cd349b10d587a623748bacf0b48d2fccc3f5bada4.jpg)  
Figure 21. Additional qualitative comparison of 3D completion on Omni-Comp [82].

![](images/ef66e8f3b93c253c5860204634e1a8c4b685b10f4fe82a1b481e1aa981495259.jpg)  
Figure 22. More multimodal control results.