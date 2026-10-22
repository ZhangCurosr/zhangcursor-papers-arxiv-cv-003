# T-DIG: TEXT-CONDITIONED DIVERSITY FOR IMAGE GENERATION

Simran Khanuja<sup>1</sup> Rohit Gandikota<sup>2</sup> Gaurav Parmar<sup>1</sup> Graham Neubig<sup>1</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Northeastern University

Correspondence: skhanuja@andrew.cmu.edu <sup></sup> image-study.org/t-dig

## ABSTRACT

Text-to-image (T2I) platforms typically return several images per prompt so users can explore different visual interpretations and choose one that matches their intent. However, T2I models are usually optimized for single-image quality which comes at a cost of multi-image diversity, limiting this exploration. Diversity is also contextual: one user wants variations in background, while other in artistic style, and the same user may later want variations in clothing. The generated images should differ along whatever axis the user cares about. In this paper, we introduce T-DIG, a text-conditioned diversity framework for image generation and editing. T-DIG is an iterative framework comprising multiple modules that lets users type prompts, inspect default generations, request for variations using natural language, and continue refining outputs until they’re satisfied. At its core, T-DIG uses an LLM policy that produces K axis-conditioned prompts or edit instructions for a frozen generator or editor. We train this policy with GRPO using a text-conditioned diversity reward and alignment penalties. Because the image generator remain frozen, T-DIG applies to open-weight, distilled, and proprietary models. We also release the first dataset for text-conditioned prompt diversity in image generation, with the training and validation splits built from human-written axes and a test set drawn from real user interactions with T-DIG. T-DIG far exceeds baselines in improving text-conditioned diversity, while preserving baseprompt alignment across a variety of metrics and human judgements. We show that user-directed diversity exploration can be added to a wide range of image generation models, open and closed, without the need to access their weights.<sup>1</sup>

## 1 INTRODUCTION

Text-to-image platforms (Midjourney, Inc., 2024; Adobe Inc., 2025) typically show four to eight images per prompt, so that the user has multiple options to choose from. However, different users may often want to see different kinds of variation, which T2I models don’t typically support. We conducted a human study where users were shown images for “people sitting at a table with laptops and books”; one user asked for variation in “laptops and camera angle”, another wanted to see “people doing different things”, and a third wanted to vary “people’s ethnicity”. Requests also ranged from being highly specific, to fairly abstract and open-ended. For a dog playing in a park, a user asked for different “dog breeds”, whereas for a comic-book panel..., they wanted “crazy endings that would surprise them”.

Prior works have proposed many methods to increase diversity of image generation models, but none can accommodate for these varying scenarios. One category of diversity methods (Parmar et al., 2026; Corso et al., 2024) maximizes a pairwise distance in latent embedding spaces (Fu et al., 2023; Oquab et al., 2023). These approaches make the images generically diverse but do not take the user’s request into consideration. Some methods do target a specific proxy of diversity, for example, varying modes of an underspecified prompt (Morshed & Boddeti, 2025), the identities of people in a scene (Borse et al., 2026), or demographic and geographic coverage (Askari Hemmat et al., 2024;

![](images/f83b1041f9f7d3cdd2c0e531a61c61423fb1c66baab8131ce946c7281955bd12.jpg)  
Figure 1: Text-conditioned Diversity: For one prompt, i.i.d. sampling and prior methods return one set regardless of what the user wants to vary; T-DIG produces a different set for each text-condition, diversity strength, and can edit/generate, accounting for user preferences.

Miao et al., 2024); but they do not generalize beyond that pre-specified target. All prior methods return one output set per prompt with a predetermined type of variation, as shown in Figure 1.

In this paper, we treat diversity as a conditional generation problem: given a prompt and an axis of variation written in free-form text, produce K images that differ along that axis while following the prompt. We build T-DIG, an interactive framework around this idea (Section 2). At its core, an LLM policy expands the base prompt into K axis-conditioned prompts or edit instructions for a frozen T2I model. We train the policy with GRPO, rewarding text-conditioned diversity measured with TPIPS (Wang et al., 2026) and penalizing loss of alignment with the base prompt. TPIPS is a VLM fine-tuned on human perceptual judgments to score image similarity along a free-form textspecified aspect, so it lets us measure diversity along user-specified axes. Because the T2I model stays frozen and need not be differentiable, T-DIG applies to open-weight, distilled, and API-only generators and editors.

To the best of our knowledge, no existing dataset pairs prompts with user-written axes of variation. We therefore build training and validation data from HPDv2 prompts (Wu et al., 2023) with axes elicited from a VLM using human-written examples in context, and a test set of 405 (prompt, axis) pairs written by participants using T-DIG. Across four open-weight T2I models, T-DIG more than doubles diversity along the requested axis over i.i.d. sampling on average (2.3×, and up to 3.1×), while lowering alignment with the base prompt by less than 2%. Existing diversity methods (that do not condition on any axis), fall short on one side or the other: the most diverse baseline, CADS (Sadat et al., 2024), reaches 1.8× at a 6% alignment drop, while Scaling Group Inference (Parmar et al., 2026) preserves alignment best (under 1% drop) but gains only 1.3×, at up to 22× the sampling cost. Policies trained with open-weight models also transfer to proprietary ones – on GPT Image 2, our Qwen-Image policy raises diversity about 40% more than asking GPT-5.5 to produce varied prompts along the given axis. In pairwise human evaluation, raters find T-DIG’s sets more varied along the requested axis than every baseline’s on 64–96% of instances across all T2I models in our study. Our overall contributions are:

1. Task formulation: We formulate T2I diversity as conditional generation, where users state in free-form text what should vary, from broad axes like style to prompt-specific requests no fixed taxonomy covers.

2. Interactive framework: T-DIG lets users request, control, and iteratively refine variations in natural language, by generating new sets or editing existing ones.

![](images/926d033ffb4feac90c3173a975746598efd07211909df5f39fd86ab306c69939.jpg)  
Figure 2: T-DIG framework. After inspecting default samples, the user writes an axis or picks a suggested one, sets the diversity strength, and chooses to generate a new set or edit the current one. A trained policy writes K prompts or edit instructions for a frozen T2I model or editor, and the user accepts the result or refines it for another round. Only the policies are trained.

3. Model-agnostic training: We train generation and editing policies with GRPO on a textconditioned diversity reward and a baseline-relative alignment penalty; the T2I model stays frozen, so T-DIG works with open-weight, distilled, and proprietary models.

4. User-derived dataset for text-conditioned diversity: We release human-grounded training and validation data, along with a test set collected from real user interactions and a human rating study against three baselines.

## 2 FRAMEWORK DESIGN

Figure 2 shows how a user interacts with T-DIG. We describe each step of interaction and the diversity module we train below:

1. I.I.D Sampling: The user enters a prompt p and sees K samples drawn with independent seeds. These show what the T2I model does by default; if the set is already diverse according to what the user wants, they stop here. Otherwise a frozen Suggestion Module , given p and the samples, suggests axes of variation based on the images generated and the input prompt.

2. Axis: The user writes the axis of variation a in free-form text, or picks or edits a suggestion that we give. An axis can be broad (“artistic style”, “composition”), specific (“the arrangement of people around the table”), or simply “overall”.

3. Strength and set size: The user sets a set size $K \in \{ 2 , 4 , 8 , 1 6 \}$ and a diversity strength s, which ranges from minimal substitutions along the axis to richer scenes that vary more.

4. Rendering Mode: The user can choose to either generate a new set of images that vary along the axis, or edit the one already generated if they want to add more variation to the current set.

5. Inspect and refine: The user accepts the new set $\mathcal { X } ^ { \prime }$ , or starts another round with $\mathcal { X } ^ { \prime }$ as input, changing any of $a , s , K$ , or the mode; and our suggestions are recomputed for the new set.

Diversity Module is a pair of post-trained LLM policies, $\pi _ { \mathrm { g e n } }$ and $\pi _ { \mathrm { e d i t } }$ . Each is a LoRA adapter $\Delta W = { \bar { B } } A$ on a frozen language model with weights $W _ { 0 } ,$ and strength s sets its scale $\alpha _ { s } \in [ 0 , 1 ]$ giving the policy $\pi _ { \alpha _ { s } }$ with weights $W _ { 0 } + \alpha _ { s } \Delta W$ . Given p, a, and K (for editing additionally the current set X), it writes K prompts or edit instructions:

$$
\mathbf { q } = \pi _ { \alpha _ { s } } ( p , a , K ; \mathcal { X } ) = \{ q _ { 1 } , . . . , q _ { K } \} ,\tag{1}
$$

which a frozen T2I model turns into the new set, $\chi ^ { \prime } = R _ { \mathrm { g e n } } ( \mathbf { q } )$ in generation mode and $\mathcal { X } ^ { \prime } =$ $R _ { \mathrm { e d i t } } ( \mathcal { X } , \mathbf { q } )$ in editing mode. Only the prompt-set adapters are trained; the generator, the editor, and the suggestion VLM stay frozen.

## 3 TRAINING

In this section, we describe how we train the two LLM policies of the Diversity Module with GRPO, for generation and editing.

## 3.1 TRAINING DATA

Base prompts: We start from the HPDv2 training split (Wu et al., 2023), which contains userwritten T2I prompts. After removing empty entries and exact duplicates, 103,672 unique prompts remain. We filter this pool with GPT-5.6 Luna (OpenAI, 2026a) using medium reasoning effort. The filter removes prompts that depict named real individuals, contain unsafe content, lack a depictable subject, or are dominated by style and quality descriptors. A prompt is retained only if the model can identify at least three meaningful axes of variation. This step retains 79.9% of the unique pool. Even after exact-duplicate removal and filtering, we find that many retained prompts are semantically similar. To reduce redundancy in training, we cluster the remaining prompts and choose a sample from each, leaving roughly 20,000 prompts (filtering and clustering details in Appendix A.1).

Axes: In a pilot study, participants saw i.i.d. outputs for 20 prompts and wrote what they would like to vary. This is used as seed data to elicit axes of variation from a VLM at scale. For each base prompt p we collect above, we render K i.i.d. samples with the T2I model and ask Gemini 3.8 Flash (Google, 2026) for up to four free-form axes, using the pilot instruction and responses as in-context examples. For a random 10% of prompts we also add the axis “overall”, which supports axis-free requests in case people would like to see overall diversity. After removing malformed outputs and holding out 200 prompts for validation, the training set contains 19,765 prompts and 76,089 (prompt, axis) pairs.

## 3.2 POLICY TRAINING

Both policies are trained with a set-level reward that favors diversity along the text condition while penalizing loss of alignment with the base prompt. We describe the different components that make up the final reward below:

Text-conditioned diversity: For an image set $\mathcal { X } = \{ x _ { 1 } , . . . , x _ { K } \}$ and an axis in the form of text prompt a, TPIPS (Wang et al., 2026) embeds each image conditioned on the axis, $e _ { i } =$ $f _ { \mathrm { T P I P S } } ( x _ { i } ; a )$ , which means TPIPS embeddings of images are similar when they look alike along a, even if they differ in other aspects. We measure the spread of these embeddings with the Vendi Score (VS) (Friedman & Dieng, 2022); given unit-normalized TPIPS embeddings and their cosinesimilarity matrix $\begin{array} { r } { S _ { i j } = e _ { i } ^ { \top } e _ { j } , \bar { \mathrm { V S } } = \exp \big ( - \sum _ { k } \lambda _ { k } \log \lambda _ { k } \big ) } \end{array}$ , where $\lambda _ { k }$ are the eigenvalues of $S / K$ It essentially gives us the effective number of distinct images in the set, from 1 (all identical along a) to K (all dissimilar). We normalize it by set size:

$$
D _ { a } ( \mathcal { X } ) = \frac { \mathrm { V S } ( \{ e _ { i } \} _ { i = 1 } ^ { K } ) - 1 } { K - 1 } \in [ 0 , 1 ] .\tag{2}
$$

Base-prompt alignment: We measure alignment with VQAScore (Lin et al., 2024). Alignment under i.i.d. sampling varies widely across prompts, so a single threshold would penalize many prompts before the policy changes anything. We therefore set a per-prompt threshold $\mathbf { \nabla } ^ { - } ( p , R )$ to the mean VQAScore of K i.i.d. samples from the renderer R being trained for, and penalize only images that fall below their own i.i.d score threshold:

$$
\mathcal { L } _ { \mathrm { a l i g n } } ( \mathcal { X } ; p , R ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \operatorname* { m a x } \big ( 0 , \tau ( p , R ) - \mathrm { V Q A S c o r e } ( x _ { i } , p ) \big ) .\tag{3}
$$

Prompt length: Without a length term, the policy learns to write overly detailed prompts that the alignment hinge does not penalize. This makes diversity leak off-axis, and text past the encoder’s context limit is also truncated, hence a length penalty is needed. As with alignment, we set the budget relative to the base prompt where each expanded prompt may add at most w words to $p ,$ $c ( p ) = | p | + w$ , where | · | is the word count and w is a hyperparameter:

$$
\mathcal { L } _ { \mathrm { l e n } } ( \mathbf { q } ; p ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { \operatorname* { m a x } \left( 0 , | q _ { i } | - c ( p ) \right) } { c ( p ) } .\tag{4}
$$

Preference preservation: Even after alignment and length penalties, a policy can raise VQAScore while making the images worse, for example by asking the model to print the base prompt as a caption inside the image, which only human preference scores tend to penalize. We therefore add a hinge on HPSv3 (Ma et al., 2025), calibrated per prompt like the alignment threshold: $\kappa ( p , R )$ is the mean HPSv3 of the K i.i.d. samples $R ( p , z _ { j } )$ , and

$$
\mathcal { L } _ { \mathrm { p r e f } } ( \mathcal { X } ; p , R ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \operatorname* { m a x } \big ( 0 , \kappa ( p , R ) - \mathrm { H P S v 3 } ( x _ { i } , p ) \big ) .\tag{5}
$$

Final Reward: The final reward combines the four terms above; $\lambda , \mu$ and $\eta$ trade diversity along text-condition (a), against alignment with base prompt $( p )$ , prompt length, and human preference:

$$
r ( \mathbf { q } , \boldsymbol { \mathcal { X } } ; p , a , R ) = D _ { a } ( \boldsymbol { \mathcal { X } } ) - \lambda \mathcal { L } _ { \mathrm { a l i g n } } ( \boldsymbol { \mathcal { X } } ; p , R ) - \mu \mathcal { L } _ { \mathrm { l e n } } ( \mathbf { q } ; p ) - \eta \mathcal { L } _ { \mathrm { p r e f } } ( \boldsymbol { \mathcal { X } } ; p , R ) ,\tag{6}
$$

Generation and editing: The generation policy writes K expanded prompts (Eq. 1) that vary along a and otherwise preserve the content and wording of $p$ (instructions in appendix, figures 7, 8). The editing policy is trained on the same data, with $K \mathrm { i . i . d }$ . samples of $p$ as the source set $\mathcal { X } _ { \mathrm { s r c } } = \{ \tilde { x } _ { 1 } , \dots , \tilde { x } _ { K } \}$ , and writes one instruction per image, $x _ { i } = R _ { \mathrm { e d i t } } ( \bar { x _ { i } } , q _ { i } )$ . Since an instruction can raise $D _ { a }$ by replacing the whole image, we subtract $\nu \mathcal { L } _ { \mathrm { d r i f t } }$ from Eq. 6 for the editing policy (below) where d is the DreamSim distance (Fu et al., 2023) and δ allows for on-axis edits:

$$
\mathcal { L } _ { \mathrm { d r i f t } } ( \mathcal { X } , \mathcal { X } _ { \mathrm { s r c } } ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \operatorname* { m a x } \big ( 0 , d ( \tilde { x } _ { i } , x _ { i } ) - \delta \big ) ,\tag{7}
$$

Optimization: We optimize both policies with GRPO (Shao et al., 2024). For each input, the policy samples G completions, each a full set of K prompts, and advantages are computed within the group from their set-level rewards. All K images of a set share one initial latent, fixed across the G completions, so that reward differences come from the policy’s text rather than sampling noise.

## 4 EXPERIMENTS

## 4.1 SETUP

T2I Models: We evaluate across open-weight, distilled, and proprietary models: Stable Diffusion 3.5 Medium and Large (Stability AI, 2024), Qwen-Image-2512 (Qwen Team, 2025), and FLUX.2 [klein] 9B (distilled) (Black Forest Labs, 2026); and API models – GPT Image 2, and Gemini 3.1 Flash Image (OpenAI, 2026b; Google DeepMind, 2026). For editing we use the corresponding editor of each family. We train at 512px with reduced step counts, similar to FlowGRPO (Liu et al., 2026) and evaluate at 1024px with each model’s recommended sampler.

Prompt-set policies: We use models from the Qwen3.5 family (Team, 2026) since they have a variety of model sizes available. For each T2I model we train 4B, 9B, and 35B-A3B generation policies; the base prompt and axis are fed in-context, with instructions to generate K diverse prompts along the axis. Each policy is a LoRA adapter (Hu et al., 2022) of rank 32 on all attention, linearattention, and MLP projections; the T2I model is never updated.

Training: We use set size $K = 4$ , reward weights $\lambda = 2 , \mu = 0 . 1$ and $\eta = 0 . 0 5$ , and length allowance $w = 2 0$ . We train LoRA adapters (rank 32) with GRPO at learning rate $2 \times 1 0 ^ { - 5 }$ without a KL term, and train for 800 steps. Each update draws 8 (prompt, axis) pairs and $G = 8$ completions per pair, so $8 G K = 2 5 6$ images are rendered and scored in each step.

Evaluation data: We validate on the 200 held-out prompts of Section 3.1. We also build a test set from real use (Section 4.4), yielding 405 interactions, roughly 100 from each open-weight model above. For comparison with axis-agnostic methods, we evaluate T-DIG conditioned on overall on the COCO val2017 split (Lin et al., 2014) (COCO-250), where no axis is specified.

## 4.2 BASELINES

We compare T-DIG, untrained (α = 0) and trained (α = 1), against four baselines. Untrained T-DIG runs the same Qwen3.5 backbone and instruction without the adapter, which isolates the effect of GRPO. None of the baselines accepts an axis, so each returns the same set whatever the user asks to vary. Baselines draw K independent initial latents, whereas T-DIG renders all K prompts from one shared latent, so that differences within a set come from the prompts alone.

I.I.D. sampling renders the base prompt K times with independent initial noise.

PAG (Yun et al., 2025) fine-tunes GPT-2 with GFlowNets to sample diverse rewrites of a prompt; it was trained against Stable Diffusion v1.4 and receives no axis. We decode K prompts from the authors’ checkpoint with diverse beam search and use them to generate images from the T2I model, instead of the base prompt. PAG tests whether generic prompt diversity suffices without axis conditioning.

CADS (Sadat et al., 2024) adds annealed noise to the conditioning embedding at inference time. We use it on open-weight T2I models where those embeddings are available, with the official textto-image settings τ = 0.6, τ = 0.9, s = 0.25, ψ = 1.

Scaling Group Inference (Parmar et al., 2026) samples M = 128 candidates and prunes them progressively to the K that best balance CLIP text–image similarity against DINOv2 pairwise distance, at 3.3–21.9× the denoiser cost of i.i.d. sampling.

## 4.3 METRICS

Alignment: We report VQAScore with CLIP-FlanT5-XXL (Lin et al., 2024), the probability that a VQA model answers “Yes” to “Does this figure show ‘{prompt}’?”, and CLIPScore (Hessel et al., 2021) the cosine similarity between the CLIP embeddings of the image and the prompt; both are computed against the base prompt.

Diversity: Our primary metric is normalized TPIPS-Vendi, TPIPS (Wang et al., 2026; Friedman & Dieng, 2022), conditioned on the requested axis; it is the only metric here that measures variation along a text-condition. TPIPS (Wang et al., 2026) is a perceptual similarity metric conditioned on a text prompt naming a visual aspect (eg. “lighting”, “pose”), trained on human similarity judgments. Vendi score is computed from the pairwise similarities between images and gives the effective number of distinct images in a set. DINO (Oquab et al., 2023) and DreamSim (Fu et al., 2023) measure axis-agnostic spread using self-supervised DINOv2 features and DreamSim (perceptual similarity metric trained on human judgments), respectively.

Preference and aesthetics: We additionally report ImageReward (Xu et al., 2023), HPSv3 (Ma et al., 2025), and LAION-Aesthetics V2 (Schuhmann, 2022). ImageReward, and HPSv3 mix alignment with preference and are not pure aesthetic scores; the LAION score is prompt-independent.

## 4.4 HUMAN EVALUATION

Stage 1, test-set collection: We hire participants on Prolific<sup>2</sup> to interact with the framework we build (Section 2), where each participant makes five sets of images using one T2I model, iterating until satisfied. Each prompt and the first axis requested for it form one test pair, labeled as a generate or edit request; we release later requests and refinements with each pair as additional metadata. After excluding participants who fail quality checks (eg. empty or copied comments), the test set contains 405 (prompt, axis) pairs from 58 participants across the four open-weight T2I models.

Stage 2, ratings: For each test pair, we render sets of $K = 4$ images with T-DIG, both trained $( \alpha \ : = \ : 0 . 5 ,$ , default diversity slider position on interface) and untrained $( \alpha = 0 )$ , and with i.i.d. sampling, CADS $( s = 0 . 2 5 )$ , and Scaling Group Inference $( \lambda = 1 , M = 1 2 8 )$ (we remove underperforming baselines like PAG). We collect ratings pairwise: each screen shows T-DIG next to one baseline, in random order, with the baseline rotating across pairs. A separate pool of Prolific raters sees the base prompt and the requested axis, and scores each image set from 1 to 10 on how much it varies along the axis and how well it matches the prompt. Raters can mark an axis not applicable if it is not a request for variation (eg. a complaint about the images). Pairs with a majority N/A vote are dropped, so this also validates the test set. Each pair gets three ratings (about 2,400 in total). We report two win rates by majority vote: how often T-DIG is rated more varied (diversity win rate), and how often it is rated more varied and at least as well aligned (diversity + alignment win rate).

## 5 RESULTS AND ANALYSIS

We evaluate T-DIG on – a) the interactive test set and report automatic metrics and human preference ratings (Table 1); b) COCO-250 to compare with methods designed for axis-free diversity (Figure 5), and ablate the reward (Table 2). We analyze results through the research questions below.

RQ1: How does T-DIG compare with baselines on text-conditioned diversity? We describe each method by its change from i.i.d. sampling on the same T2I model, and report averages/ranges across models (Table 1). None of the baselines takes an axis as input, so they can only make the set vary more in general: Scaling Group Inference raises TPIPS by +0.03 to +0.05 at $3 . 3 \substack { - 2 1 . 9 \times }$ the denoiser cost, PAG by $+ 0 . 0 3 \mathrm { t o } + 0 . 0 9$ , and CADS by +0.08 to +0.17. Training-free T-DIG (α=0), which prompts the untrained policy with the axis, already matches CADS $( + 0 . 0 8 ~ { \mathrm { t o } } \ + 0 . 1 4 )$ , and training $( \alpha { = } 1 )$ roughly doubles that gain $\left( + 0 . 1 4 \ \mathrm { t o \ + 0 . 2 6 } \right)$ . When comparing with axis-agnostic diversity metrics, we get a sense of how much variation there is on-axis versus off-axis. CADS comes within 0.05 of T-DIG on $\mathrm { D I N O } _ { v }$ and exceeds it on $\mathrm { D r e a m S i m } _ { v }$ for some models, so its sets vary about as much overall, but very less of that variation lies along the requested axis, as measured by $\mathrm { T P I P S } _ { v }$ . An increase in diversity often comes at the cost of alignment with the base prompt. T-DIG changes VQAScore by $- 0 . 0 5 \ \mathrm { t o \  t 0 . 0 1 }$ , against −0.03 to −0.08 for CADS and −0.08 to −0.16 for PAG. Per +0.1 TPIPS gained (slope in Table 1), T-DIG loses at most 0.02 VQAScore, while CADS loses up to 0.08 and PAG up to 0.41; Scaling Group Inference loses up to 0.03, but for a much smaller diversity gain at a much higher inference cost. With respect to preference and aesthetics, T-DIG keeps ImageReward and HPSv3 near the i.i.d. level (HPSv3 −6 to +3%) except on FLUX.2 [klein] (RQ2), whereas CADS loses 14 to 24% HPSv3. LAION-Aesthetics, which ignores the prompt, rises slightly with T-DIG (+0.13 to +0.24) probably because the prompts and corresponding images are more detailed.

RQ2: How does the performance of T-DIG vary across T2I models? The diversity gain is largest where i.i.d. sampling varies least. Qwen-Image and FLUX.2 [klein] give nearly identical images when sampled i.i.d. $( \mathrm { T P I P S } _ { v }$ of 0.12 and 0.14), and T-DIG raises $\mathrm { T P I P S } _ { v }$ by +0.26 and +0.25; SD3.5 L and M already vary more (0.20 and 0.21) and gain +0.14 and $+ 0 . 1 5$ . All four models land at 0.35–0.40 at $\alpha { = } 1$ , so the gain is bounded by where the metric saturates, not by the model. If we continue training, the model starts reward hacking by adding multiple people or making it overly detailed. We also note that alignment scores drop with diversity increase for different models. On SD3.5 L and M, whose i.i.d. alignment is lowest (VQAScore 0.80 to 0.83), T-DIG’s difference with i.i.d is not statistically significant; but on Qwen-Image and FLUX.2 [klein] (0.89 and 0.91) it loses 0.03 and 0.05. FLUX.2 [klein] also loses the most preference (0.40 ImageReward, 10% HPSv3), since its near-identical i.i.d. images score very high to begin with.

RQ3: How does T-DIG compare with baselines on overall diversity? Even with no axis, T-DIG gives the widest quality–diversity trade-off. On COCO-250 with Qwen-Image, sweeping α from 0 to 1 raises $\mathrm { D I N O } _ { v }$ over i.i.d. sampling by +0.21 to +0.39 (about $3 \times \mathrm { i . i . d }$ . at $\alpha { = } 1 )$ while losing only 0.01 to 0.02 VQAScore. Varying the guidance scale or Scaling Group Inference’s λ gains at most +0.13, and CADS reaches $+ 0 . 3 2 \ \mathrm { t o \ + 0 . 3 7 }$ but drops VQAScore by 0.08 to 0.11.

![](images/e8d04f83b2c50425a105e5fcfa1f4e4ebf412bca8a13e6b33d8b3f48b4378750.jpg)

<table><tr><td></td><td colspan="3">Diversity</td><td colspan="2">Alignment</td><td colspan="3">Preference</td><td>Human</td><td colspan="3">Trade-off vs. I.I.D.</td></tr><tr><td>Method</td><td>TPIPSv</td><td>DINOv</td><td>Dream- Simv</td><td>VQA</td><td>CLIP</td><td>IR</td><td></td><td>HPSv3 Aes.</td><td>Win (%)</td><td>TPIPSv (×)</td><td>VQA (%)</td><td>Slope</td></tr><tr><td>Qwen-Image</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>I.I.D.</td><td>0.124</td><td>0.209</td><td>0.194</td><td>0.889</td><td>0.809 1.27</td><td></td><td>11.55</td><td>5.95</td><td>68.0</td><td>1.00</td><td>0.0</td><td></td></tr><tr><td>Group Inf.</td><td>0.168</td><td>0.281</td><td>0.263</td><td>0.885</td><td>0.812</td><td>1.22</td><td>11.43</td><td>5.91</td><td>76.0</td><td>1.36</td><td>-0.4</td><td>-0.008</td></tr><tr><td>CADS</td><td>0.293</td><td>0.502</td><td>0.540</td><td>0.808</td><td>0.760 0.46</td><td></td><td>8.76</td><td>5.77</td><td>80.8</td><td>2.37</td><td>-9.1</td><td>-0.048</td></tr><tr><td>PAG</td><td>0.218</td><td>0.381</td><td>0.326</td><td>0.725</td><td>0.736 0.77</td><td></td><td>10.35</td><td>6.35</td><td></td><td>1.76</td><td>-18.4</td><td>-0.174</td></tr><tr><td>T-DIG α=0</td><td>0.268</td><td>0.364</td><td>0.300</td><td>0.859</td><td>0.7961.16</td><td></td><td>10.92</td><td>5.95</td><td>58.3</td><td>2.17</td><td>-3.3</td><td>-0.020</td></tr><tr><td>T-DIG α=0.5</td><td>0.367</td><td>0.524</td><td>0.443</td><td>0.862</td><td>0.782 1.16</td><td></td><td>10.79</td><td>6.00</td><td></td><td>2.96</td><td>-3.1</td><td>-0.011</td></tr><tr><td>T-DIG α=1</td><td>0.379</td><td>0.542</td><td>0.452</td><td>0.859</td><td>0.7751.21</td><td></td><td>10.84</td><td>6.08</td><td></td><td>3.06</td><td>-3.3</td><td>-0.012</td></tr><tr><td>SD3.5 L</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>I.I.D.</td><td>0.202</td><td>0.343</td><td>0.315</td><td>0.831</td><td>0.834 1.23</td><td></td><td>8.90</td><td>5.80</td><td>82.6</td><td>1.00</td><td>0.0</td><td></td></tr><tr><td>Group Inf.</td><td>0.253</td><td>0.446</td><td>0.422</td><td>0.819</td><td>0.831 1.16</td><td></td><td>8.62</td><td>5.77</td><td>56.5</td><td>1.25</td><td>-1.5</td><td>-0.025</td></tr><tr><td>CADS</td><td>0.284</td><td>0.516</td><td>0.520</td><td>0.769</td><td>0.811 0.81</td><td></td><td>7.66</td><td>5.68</td><td>60.9</td><td>1.41</td><td>-7.5</td><td>-0.076</td></tr><tr><td>PAG</td><td>0.242</td><td>0.434</td><td>0.387</td><td>0.749</td><td>0.782 0.89</td><td></td><td>8.35</td><td>6.06</td><td></td><td>1.20</td><td>-10.0</td><td>-0.205</td></tr><tr><td>T-DIG i α=0</td><td>0.283</td><td>0.419</td><td>0.379</td><td>0.802</td><td>0.8081.01</td><td></td><td>8.37</td><td>5.85</td><td>52.6</td><td>1.40</td><td>-3.5</td><td>-0.036</td></tr><tr><td>T-DIG α=0.5</td><td>0.357</td><td>0.537</td><td>0.473</td><td>0.843</td><td>0.7971.15</td><td></td><td>8.82</td><td>5.97</td><td></td><td>1.77</td><td>+1.5</td><td>+0.008</td></tr><tr><td>T-DIG α=1</td><td>0.346</td><td>0.561</td><td>0.469</td><td>0.829</td><td>0.784 1.22</td><td></td><td>8.86</td><td>6.04</td><td></td><td>1.71</td><td>-0.3</td><td>-0.002</td></tr><tr><td>SD3.5 M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>I.I.D.</td><td>0.211</td><td>0.338</td><td>0.364</td><td>0.801</td><td>0.8371.10</td><td></td><td>7.57</td><td>5.56</td><td>57.1</td><td>1.00</td><td>0.0</td><td></td></tr><tr><td>Group Inf.</td><td>0.256</td><td>0.454</td><td>0.468</td><td>0.790</td><td>0.829 0.96</td><td></td><td>7.28</td><td>5.56</td><td>42.9</td><td>1.21</td><td>-1.4</td><td>-0.024</td></tr><tr><td>CADS</td><td>0.298</td><td>0.524</td><td>0.560</td><td>0.759</td><td>0.796 0.45</td><td></td><td>5.94</td><td>5.48</td><td>64.3</td><td>1.42</td><td>-5.2</td><td>-0.048</td></tr><tr><td>PAG</td><td>0.238</td><td>0.394</td><td>0.403</td><td>0.689</td><td>0.772 0.79</td><td></td><td>7.32</td><td>5.99</td><td></td><td>1.13</td><td>-14.0</td><td>-0.410</td></tr><tr><td>T-DIG α=0</td><td>0.290</td><td>0.386</td><td>0.377</td><td>0.818</td><td>0.8161.07</td><td></td><td>7.82</td><td>5.71</td><td>50.0</td><td>1.38</td><td>+2.1</td><td>+0.022</td></tr><tr><td>T-DIG α=0.5</td><td>0.362</td><td>0.470</td><td>0.470</td><td>0.787</td><td>0.801 1.03</td><td></td><td>7.45</td><td>5.79</td><td></td><td>1.72</td><td>-1.8</td><td>-0.009</td></tr><tr><td>T-DIG α=1</td><td>0.359</td><td>0.505</td><td>0.488</td><td>0.814</td><td>0.804 1.12</td><td></td><td>7.80</td><td>5.79</td><td>一</td><td>1.70</td><td>+1.6</td><td>+0.008</td></tr><tr><td>FLUX.2 [klein]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>I.I.D.</td><td>0.143</td><td>0.228</td><td>0.228</td><td>0.912</td><td>0.792 1.25</td><td></td><td>10.18</td><td>5.75</td><td>61.9</td><td>1.00</td><td>0.0</td><td></td></tr><tr><td>Group Inf.</td><td>0.177</td><td>0.317</td><td>0.277</td><td>0.913</td><td>0.800</td><td>1.24</td><td>10.12</td><td>5.76</td><td>69.6</td><td>1.24</td><td>+0.1</td><td>+0.004</td></tr><tr><td>CADS</td><td>0.276</td><td>0.506</td><td>0.478</td><td>0.880</td><td>0.769 0.77</td><td></td><td>8.01</td><td>5.63</td><td>69.6</td><td>1.92</td><td>-3.4</td><td>-0.024</td></tr><tr><td>PAG</td><td>0.225</td><td>0.369</td><td>0.341</td><td>0.793</td><td>0.734 0.93</td><td></td><td>9.42</td><td>6.14</td><td></td><td>1.57</td><td>-13.1</td><td>-0.147</td></tr><tr><td>T-DIG iα=0</td><td>0.281</td><td>0.350</td><td>0.326</td><td>0.883</td><td>0.772 1.09</td><td></td><td>9.49</td><td>5.78</td><td>47.6</td><td>1.96</td><td>-3.2</td><td>-0.021</td></tr><tr><td>T-DIG α=0.5</td><td>0.371</td><td>0.524</td><td>0.474</td><td>0.869</td><td>0.753 0.88</td><td></td><td>8.97</td><td>5.86</td><td></td><td>2.59</td><td>-4.7</td></table>

Table 1: Results on the interactive test set: K=4, 9B policy; higher is better for every column. Diversity: TPIPS , DINO and DreamSim are normalized Vendi scores (range [0, 1]) of TPIPS embeddings conditioned on the requested axis, and of DINOv2 and DreamSim embeddings, which ignore the axis. Alignment: VQAScore and CLIPScore against the base prompt. Preference & aesthetics: ImageReward, HPSv3 and LAION-Aesthetics V2. Human: % of instances where raters scored T-DIG <sub>α=0.5</sub>’s set as more varied along the axis than row method and at least as well aligned with the prompt. Trade-off(vs. I.I.D.): TPIPS<sub>v</sub> as a multiple of i.i.d. sampling’s, the relative change in VQAScore, and the slope of the line from i.i.d. sampling, i.e. the change in VQAScore per +0.1 TPIPS<sub>v</sub>. Bold and underline mark best and second best respectively.

Measured as the slope of the line from i.i.d. sampling, T-DIG loses 0.006 VQAScore per +0.1 DINO<sub>v</sub>, against 0.029 for CADS and 0.019 for lowering the guidance scale, and within a budget of 0.02 VQAScore it gains +0.39 DINO , against at most +0.11 for any baseline. The slopes are similar on the other T2I models: across all four, T-DIG loses 0.006 to 0.010 VQAScore per +0.1 DINO<sub>v</sub>, against 0.013 to 0.034 for CADS and 0.019 to 0.038 for the guidance scale (FLUX.2 [klein] is guidance-distilled, so it has no guidance scale to sweep).

RQ4: What is the effect of the policy size on performance? We compare 4B, 9B, and 35B-A3B policies on the held-out pairs. All sizes reach nearly the same diversity on every T2I model. Larger policies help alignment slightly; 9B matches or beats 4B on VQAScore for every model (−0.002 to +0.010), ImageReward and HPSv3 favor neither size consistently, and 35B-A3B adds no further consistent gain. Size matters mostly for following the output format: the untrained 4B policy returns a malformed set on 1.7% of requests, against none for larger models, but training removes these errors. Serving the 4B policy therefore costs little in quality and lowers latency (RQ7).

<table><tr><td rowspan="2">Reward</td><td colspan="3">Added to TPIPSv</td><td colspan="4">Held-out pairs</td><td rowspan="2"></td><td rowspan="2">Words Malf.↓</td></tr><tr><td> $L _ { \mathrm { a l i g n } }$ </td><td> $L _ { \mathrm { l e n } }$ </td><td> $L _ { \mathrm { p r e f } }$ </td><td>TPIPSv ↑</td><td>VQA↑</td><td>HPS↑</td><td>IR↑</td></tr><tr><td>Untrained</td><td></td><td></td><td></td><td>0.333</td><td>0.865</td><td>12.34</td><td>1.19</td><td>18</td><td>1</td></tr><tr><td>TPIPSv only</td><td></td><td></td><td></td><td>0.617</td><td>0.228</td><td>1.44</td><td>-1.74</td><td>90</td><td>7</td></tr><tr><td>+ alignment</td><td>√</td><td></td><td></td><td>0.458</td><td>0.857</td><td>11.64</td><td>1.04</td><td>123</td><td>1</td></tr><tr><td>+ length</td><td>√</td><td>√</td><td></td><td>0.475</td><td>0.845</td><td>11.38</td><td>0.81</td><td>34</td><td>1</td></tr><tr><td>+ preference (T-DIG)</td><td>√</td><td>√</td><td>√</td><td>0.454</td><td>0.849</td><td>12.35</td><td>1.03</td><td>29</td><td>0</td></tr></table>

Table 2: Reward ablation on FLUX.2 [klein], 9B policy on the held-out pairs. Words: mean words per expanded prompt. Malf.: malformed completions.

RQ5: How do people rate T-DIG’s outputs? Interactive study (Stage 1): Across 405 (prompt, axis) pairs, users rated the diversity of T-DIG’s outputs 3.95/5 and their alignment with the prompt 4.09/5 on average, 89% found asking for variations more useful than accepting i.i.d. outputs, and 96% enjoyed using the tool. In their comments, participants noted that it “handled the requested variation axes well” and did “a better job than [they were] expecting” on prompts that image generators often get wrong. On the drawbacks, roughly half the participants noted that the strength slider is not perceptually calibrated (we linearly scale lora weights); but we leave this for future work. The feedback was mostly overwhelmingly positive (details in Appendix Table 3). Pairwise ratings (Stage 2): By majority vote, raters found T-DIG’s set (α=0.5) more varied than the other method’s on 64 to 96% of screens in every comparison. In Table 1 we show the % of instances which were rated more varied and at least as well aligned: T-DIG wins 57 to 83% against i.i.d. sampling, 61 to 81% against CADS, 43 to 76% against Scaling Group Inference, and 48 to 58% against training-free T-DIG. The remaining pairs are ties/split votes/loss by majority voting. Note that absolute ratings on a scale of 1-10 were collected in a pairwise relative evaluation setup. Mean diversity ratings – T-DIG scores 7.4 to 8.3 out of 10, against 2.4 to 3.9 for i.i.d. sampling, 3.2 to 5.1 for Scaling Group Inference, 3.9 to 5.1 for CADS, and 5.6 to 6.9 for training-free T-DIG.

RQ6: What is the contribution of each reward term? Table 2 ablates terms of Equation 6. Diversity alone reward-hacks: TPIPS nearly doubles (0.617) but the images stop following the prompt (VQAScore 0.228). The alignment penalty restores alignment (0.857) but leads to very long prompts (123 words), which add unrequested changes and would be truncated by short text encoders such as SD3.5’s 77-token CLIP. The length penalty shortens prompts to 34 words, but even with these two penalties the policy finds other ways to reward-hack; for example, it starts writing the base prompt as a caption inside the image, which inflates alignment scores but lowers preference scores (HPSv3 11.38 vs. 12.34 untrained). The full reward, which adds the preference penalty, is robust and keeps nearly the same diversity as the alignment penalty alone (0.454 vs. 0.458).

RQ7: How much latency do the policy models add to inference? Very little relative to rendering. On one H200 with vLLM, expanding a prompt into four takes a median of 0.40 s with the 4B policy, 0.59 s with 9B, and 0.51 s with 35B-A3B. Since rendering four images takes 5 to 11 s with SD3.5 and Qwen-Image, the 9B policy adds only 6–12%. Only the 4-step FLUX.2 [klein] renders almost as fast (1.5 s), and there the 4B policy keeps the overhead to about 27%.

RQ8: How well do T-DIG’s policies trained on open-weight models transfer to proprietary ones? Since training a policy per proprietary model is too costly, we apply the Qwen-Image, SD3.5 L, and FLUX.2 [klein] policies (α=0.5) as they are to GPT Image 2 and Gemini 3.1 Flash Image on 50 test prompts, and compare them with i.i.d. sampling and with prompt expansion by proprietary LLMs (GPT-5.5, Gemini 3.8 Flash) given the same instruction. All three policies transfer and beat the proprietary LLMs’ expansion. The Qwen-Image policy transfers best, raising TPIPS over i.i.d. by +0.27 on GPT Image 2 and +0.20 on Gemini, against +0.19 and +0.15 for GPT-5.5 and Gemini 3.8 Flash; FLUX.2 [klein] follows closely (+0.26, +0.19), and SD3.5 L gains least (+0.23, +0.17). On alignment, VQAScore drops by 0.02 and 0.05 (0.01 and 0.03 more than GPT-5.5 and Gemini 3.8 Flash), and HPSv3 by 9% and 7%.

## 6 RELATED WORK

Diversity in diffusion models: Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020) are a powerful class of generative models that generate high quality samples for challenging modalities such as images Rombach et al. (2022); Esser et al. (2024); Labs (2024), 3D (Xiang et al., 2024; Poole et al., 2022), and video. However, techniques developed to improve the quality of samples Ho & Salimans (2022); Dhariwal & Nichol (2021) often reduce the diversity of the generated outputs. Conditioning methods (Zhang et al., 2023; Voynov et al., 2023; Parmar et al., 2024; Ye et al., 2023; Parmar et al., 2025) add controllability to diffusion models but reduce the variations in the outputs, especially at large guidance scales. Distillation methods (Sauer et al., 2024; Yin et al., 2024b;a; Kang et al., 2024) used to improve the inference efficiency further reduce the diversity of outputs.

T2I diversity methods fall into a few categories. Sampler interventions alter the generative trajectory: CADS anneals noise into the text conditioning (Sadat et al., 2024), PG and DiverseFlow make a batch’s trajectories repel (Corso et al., 2024; Morshed & Boddeti, 2025), and CNO optimizes the initial noise contrastively (Kim et al., 2026). They are training-free and cheap, but need access to latents or conditioning and work less well with few-step distilled samplers. Inference-time scaling draws a large candidate pool and prunes it to a diverse, high-quality subset (Parmar et al., 2026), but costs grow with pool size and they cannot go beyond what the model already produced. Fine-tuning builds diversity into the weights with a diversity-aware reward (Miao et al., 2024; Chen et al., 2026), which requires weights and gradients. Prompt adaptation, closest to us, trains a language model to rewrite prompts for a frozen model (Hao et al., 2023; Yun et al., 2025). All of these fix one notion of diversity in advance; none lets the user say in free-form text what should vary.

Text-conditioned diversity: Prior work argues that similarity is always similarity in some respect (Tversky, 1977; Medin et al., 1993), yet most T2I diversity work measures it with a single scalar such as DreamSim or DINO distance (Fu et al., 2023; Oquab et al., 2023). Conditional metrics have recently emerged: some measure one aspect, such as style or relational structure (Somepalli et al., 2024; Nguyen et al., 2026); some score a fixed list of attributes (Rassin et al., 2025; Chen et al., 2026); and the concurrent TPIPS (Wang et al., 2026) takes the condition as free-form text. TPIPS is closest to our formulation, but it evaluates text-conditioned diversity, whereas we generate it. Conditional similarity has also been used for retrieval (Veit et al., 2017; Vo et al., 2019; Vaze et al., 2023) and clustering (Kwon et al., 2024), and some methods have targeted fixed subproblems such as covering modes for underspecified prompts (Corso et al., 2024; Morshed & Boddeti, 2025) or demographic coverage (Askari Hemmat et al., 2024; Miao et al., 2024). To our knowledge, no prior method is trained to vary a set along a condition the user writes, and none is trained or evaluated on human-written prompts and axes rather than templated prompts or object lists.

## 7 CONCLUSION

In this paper, we introduce the task of text-conditioned diversity for image generation, where users specify in free-form text what should vary across a set of images. We present T-DIG, an interactive framework in which an LLM policy, trained with GRPO on a text-conditioned diversity reward and calibrated alignment, length, and preference penalties, writes axis-conditioned prompts or edit instructions for a frozen T2I model. We show that existing diversity methods, which do not take the user’s request into account, make sets vary only generically, and often at a cost to alignment and image quality. In contrast, T-DIG makes sets vary substantially more along the requested axis on every T2I model we test, with little loss of alignment. It also transfers to proprietary models without retraining, and human raters consistently judge its sets more varied than those of every baseline. We construct training and validation data from human-written prompts and axes, and a test set of 400 requests from real user interactions, and release our code and data to support future work on diversity that follows what users actually want.

## 8 LIMITATIONS

While T-DIG lets users control what varies in free-form text, it has a few limitations. T-DIG’s ability to generate variations along a axis still depends on the capabilities of the base image generative model. For example, in our human studies (Appendix A.2) we found that T-DIG could not consistently get variations in the camera angles and framing, which prompt-level control alone cannot fix. Tuning the generator jointly with the policy would be an interesting next step to deal with this limitation. Participants also found the diversity strength poorly calibrated – since it linearly scales the LoRA weights $( \alpha _ { s }$ , Section 2), low settings felt too subtle and high settings too sensitive (RQ5), suggesting that a perceptually calibrated mapping from strength to α is needed. Finally, since we optimize TPIPS, the trained adapters may inherit its biases, which is one reason we also rely on collecting a test set via human interactions and collecting human ratings for the generated outputs.

## ETHICS STATEMENT

We filter training prompts to exclude named real individuals and unsafe content (Section 3.1). Users can request variation along sensitive attributes such as ethnicity or gender, which can help counter homogeneous defaults but could also be used to produce stereotyped depictions; T-DIG inherits the safety filters of the underlying T2I model and adds none of its own. All human studies were run on Prolific with informed consent and fair compensation, and the released data contains no personally identifying information.

## REPRODUCIBILITY STATEMENT

We describe the training data construction in Section 3.1 and Appendix A.1, the reward and optimization in Section 3, and all hyperparameters, T2I models, and policy configurations in Section 4.1. Instructions given to the policies and VLMs are in the appendix. Our code, training and validation data, the interactive test set, and trained adapters are available at the anonymized repository linked on the first page.

## AI USAGE STATEMENT

We used LLMs as part of our method and data pipeline: GPT-5.6 Luna filtered the HPDv2 prompts, and Gemini 3.8 Flash generated the synthetic axes of variation used for training and validation (Section 3.1); the policies themselves are fine-tuned Qwen3.5 models. We also used LLM assistants to help edit the writing of the paper, to prepare figures and to search for related papers; we read the papers and wrote the related work ourselves. We also used AI coding agents to help with writing the code for our experiments which we designed ourselves. We take responsibility for the final content of this work, including text, code, claims or artifacts produced with the aid of generative AI.

## ACKNOWLEDGEMENTS

We were fortunate to obtain feedback on early drafts of this work. In particular, we would like to thank Deva Ramanan, Fernando Diaz, and members of Neulab at CMU, LTI. This work was supported by grants from the Defense Science and Technology Agency Singapore.

## REFERENCES

Adobe Inc. Adobe firefly: Generative ai for creatives. https://www.adobe.com/ products/firefly.html, 2025.

Reyhane Askari Hemmat, Melissa Hall, Alicia Sun, Candace Ross, Michal Drozdzal, and Adriana Romero-Soriano. Improving geo-diversity of generated images with contextualized vendi score guidance. In European Conference on Computer Vision, pp. 213–229. Springer, 2024.

Black Forest Labs. FLUX.2 [klein]: Towards Interactive Visual Intelligence, 2026. URL https://bfl.ai/blog/ flux2-klein-towards-interactive-visual-intelligence.

Shubhankar Borse, Farzad Farhadzadeh, Munawar Hayat, and Fatih Porikli. Resolving the identity crisis in text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 36703–36712, 2026.

Chubin Chen, Sujie Hu, Jiashu Zhu, Meiqi Wu, Jintao Chen, Yanxun Li, Nisha Huang, Chengyu Fang, Jiahong Wu, Xiangxiang Chu, et al. Taming preference mode collapse via directional decoupling alignment in diffusion reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12775–12786, 2026.

Gabriele Corso, Yilun Xu, Valentin De Bortoli, Regina Barzilay, and Tommi Jaakkola. Particle guid ance: non-iid diverse sampling with diffusion models. In International Conference on Learning Representations, volume 2024, pp. 22480–22507, 2024.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning, pp. 12606–12633, 2024.

Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410, 2022.

Stephanie Fu, Netanel Yakir Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=DEiNSfh1k7.

Google. Gemini 3.8 Flash api documentation. https://ai.google.dev/gemini-api/ docs/models/gemini-3.8-flash, 2026. API model: gemini-3.8-flash. Accessed September 19, 2026.

Google DeepMind. Gemini 3.1 Flash Image model card, 2026. URL https://deepmind. google/models/model-cards/gemini-3-1-flash-image/.

Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36:66923–66939, 2023.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. In European Conference on Computer Vision, pp. 428–447. Springer, 2024.

Byungjun Kim, Soobin Um, and Jong Chul Ye. Diverse text-to-image generation via contrastive noise optimization. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=EVRMnAREc3.

Sehyun Kwon, Jaden Park, Minkyu Kim, Jaewoong Cho, Ernest K Ryu, and Kangwook Lee. Image clustering conditioned on text criteria. In International Conference on Learning Representations, volume 2024, pp. 1393–1432, 2024.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ European conference on computer vision, pp. 740–755. Springer, 2014.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pp. 366–384. Springer, 2024.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026.

Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15086–15095. IEEE, 2025.

Douglas L Medin, Robert L Goldstone, and Dedre Gentner. Respects for similarity. Psychological review, 100(2):254, 1993.

Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lijuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10844–10853. IEEE, 2024.

Midjourney, Inc. Midjourney website. https://www.midjourney.com/home, 2024.

Mashrur M Morshed and Vishnu Boddeti. Diverseflow: Sample-efficient diverse mode coverage in flows. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23303–23312. IEEE, 2025.

Thao Nguyen, Sicheng Mo, Krishna Kumar Singh, Yilin Wang, Jing Shi, Nicholas Kolkin, Eli Shechtman, Yong Jae Lee, and Yuheng Li. Relational visual similarity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24141–24150, 2026.

OpenAI. GPT-5.6 Luna model documentation. https://developers.openai.com/api/ docs/models/gpt-5.6-luna, 2026a. API model: gpt-5.6-luna. Accessed September 19, 2026.

OpenAI. GPT Image 2 model documentation, 2026b. URL https://developers.openai. com/api/docs/models/gpt-image-2. Model snapshot: gpt-image-2-2026-04-21.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036, 2024.

Gaurav Parmar, Or Patashnik, Kuan-Chieh Wang, Daniil Ostashev, Srinivasa Narasimhan, Jun-Yan Zhu, Daniel Cohen-Or, and Kfir Aberman. Object-level visual prompts for compositional image generation. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers, pp. 1–12, 2025.

Gaurav Parmar, Or Patashnik, Daniil Ostashev, Kuan-Chieh Wang, Kfir Aberman, Srinivasa Narasimhan, and Jun-Yan Zhu. Scaling group inference for diverse and high-quality generation. In International Conference on Learning Representations, volume 2026, pp. 89763–89790, 2026.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv, 2022.

Qwen Team. Qwen-Image-2512 model card, 2025. URL https://huggingface.co/Qwen/ Qwen-Image-2512.

Royi Rassin, Aviv Slobodkin, Shauli Ravfogel, Yanai Elazar, and Yoav Goldberg. GRADE: Quantifying sample diversity in text-to-image models, 2025. URL https://openreview.net/ forum?id=JddNOaw66n.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on com puter vision and pattern recognition (CVPR), pp. 10674–10685. ieee, 2022.

Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann Weber. Cads: Unleashing the diversity of diffusion models through condition-annealed sampling. In International Conference on Learning Representations, volume 2024, pp. 23723–23755, 2024.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIG-GRAPH Asia 2024 Conference Papers, pp. 1–11, 2024.

Christoph Schuhmann. LAION-Aesthetics Predictor V2, 2022. URL https://github.com/ christophschuhmann/improved-aesthetic-predictor.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015.

Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Investigating style similarity in diffusion models. In European Conference on Computer Vision, pp. 143–160. Springer, 2024.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Stability AI. Introducing Stable Diffusion 3.5, 2024. URL https://stability.ai/ news-updates/introducing-stable-diffusion-3-5. Official release announcement. Accessed September 15, 2026.

Qwen Team. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

Amos Tversky. Features of similarity. Psychological review, 84(4):327, 1977.

Sagar Vaze, Nicolas Carion, and Ishan Misra. Genecis: A benchmark for general conditional image similarity. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6862–6872. IEEE, 2023.

Andreas Veit, Serge Belongie, and Theofanis Karaletsos. Conditional similarity networks. In 2017 IEEE conference on computer vision and pattern recognition (CVPR), pp. 1781–1789. IEEE, 2017.

Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text and image for image retrieval-an empirical odyssey. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6432–6441. IEEE, 2019.

Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. In ACM SIGGRAPH 2023 conference proceedings, pp. 1–11, 2023.

Sheng-Yu Wang, Yotam Nitzan, Aaron Hertzmann, Jun-Yan Zhu, Eli Shechtman, Alexei A Efros, and Richard Zhang. The many senses of visual similarity: A text-prompted image perceptual metric. arXiv preprint arXiv:2607.18237, 2026.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and ¨ William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman,¨ and Taesung Park. One-step diffusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6613–6623. IEEE, 2024b.

Taeyoung Yun, Dinghuai Zhang, Jinkyoo Park, and Ling Pan. Learning to sample effective and diverse prompts for text-to-image generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23625–23635. IEEE, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3813–3824. IEEE, 2023.

## A APPENDIX

## A.1 TRAINING DATA CONSTRUCTION DETAILS

As described in Section 3.1, we filter the HPDv2 dataset to construct our base-prompt pool. The filtering prompts can be found in Figures 4 and 5. Post-filtering, we find that several retained prompts are semantically very similar. Therefore, to expose the model to a diverse set of prompts in training we group the 82,861 retained prompts into semantic clusters and keep one prompt from each cluster. We embed every prompt with all-MiniLM-L6-v2, normalize the embeddings, and apply agglomerative clustering under cosine distance with a threshold of $\tau = 0 . 3 5$ , which yields 23,020 clusters. We swept τ from 0.15 to 0.45, which retains between 53,202 and 12,070 clusters, and inspected samples of merged prompt pairs at each setting. Merging is imperfect throughout, and after some manual inspection we select τ = 0.35, as we observed that this value gave us broad topic coverage. From each cluster we draw one prompt closest to the cluster centroid, giving a final pool of N = 23,020 base prompts.

## i.i.d. sampling

## T-DIG (α=0.5)

Qwen-Image Prompt: "An astronaut floating outside a spaceship, trying to catch a floating slice of pizza." Axis: “show different astronaut reactions like reaching out desperately, saluting the pizza, floating upside down in panic..."

![](images/2723bcd2766e3f0219abe584a718c60180adc9a9ce1966ca4408ad02b76045f1.jpg)

![](images/11ebf92552577f7354f0dbd2bdaf81813cb6a87d578d8bbf65557304e2e825e0.jpg)

![](images/31033825b2fdbf998bf840531e31a12c16016e456c8388b8fd54ffbaa7d1d946.jpg)  
sD3.5 L Prompt: “a family standing in the rain" Axis: "art style"

![](images/ac5f7aae98b4d0edb56d87de410edc94342a547f1a9cca0396fbe8341909628a.jpg)

![](images/8489c1ae6b25c68526883f99ceb4bc6b42487092acaa8b668336c2afe66059b9.jpg)

![](images/2169cc2523a06327d8c76dffa34a7ec06af49b3db468950e0e93a9eab4e9c8d1.jpg)

![](images/115b6c13dcf2d09b142d3cec326adf4a607e42bc8575607a597ce37e33dde583.jpg)

![](images/3849e578097e4673edb5a13fcb844f410a4554ca912733b93e0508d0fad018a6.jpg)

![](images/c649db20f396abe82412b15d01ef5faf70f174d3aa3007ccccdf157550bc1993.jpg)

![](images/caf08eba021470a101e45c2e6a167ce35a73c84498dc2a2278552670f75f919a.jpg)

![](images/5c987ba066efa7abf5056686d64e0c5a862b9386c4e60e57daef684cb9e6be76.jpg)

![](images/973dc9507f06699468f097075693d990374f2bb5a6717e7ab418b153783c62cb.jpg)

![](images/f7eed368d5afe86a7d17a1320df02be9e60281dfdee6d0a7b27f857d4c907d18.jpg)

![](images/2b612362fcc11a5f1017bcda3a3f5fa3a97dc5c103e0854ea90f8dbb5ebd1fb0.jpg)  
sD3.5 M Prompt: “helicopter flying alongside a flying squirrel with rabies" Axis: “squirrel size, squirrel color, helicopter color, terrain"

![](images/5fa15fe1a7a0dcd80008230d558459f727489042b065f5a0cbb2d1eebcc4a454.jpg)

![](images/98d451f37d04f3d3ad069513f19f90e95a09883d1539c6b5f7a591ade8d29b78.jpg)

![](images/90e53db813f46b480e0b9f48fdf7f580d8ccbcb12584dc864fa9ed991ef387fa.jpg)  
FLuX.2 [klein] Prompt: “A sparrows sitting with a mockingbird on a mango tree" Axis: "Magnification, mango type, ripeness'

![](images/e08ca5af43f4e788e98fa5f8c0825867739311e87456e6397e89a5f31067221e.jpg)

![](images/40f5b95327844bf3529a1208bce49f8494b8fb8c76bb8faa18c9c1d673b3cd2e.jpg)

![](images/96fd972ceb20743cc4175909707a12ecd8661c8c30c0033b1cf4a5c68cf77eba.jpg)

![](images/42594f5617be21a9f3b1cadd768515c295d64b92cd42ab5fff83b4c4d11a9a84.jpg)

![](images/52e460f00f2912182b0aee8719d5483e26eedf1f263247939ae3a22b263cd082.jpg)

![](images/9ffc82e04aad4595d83e12b7f24b04523089839fc2d6a64f469939f4b74dd253.jpg)

![](images/dd27e1c534dba1b2d392974b5911e949b71a75e848dd6374205c7c687980a439.jpg)

![](images/b4b9f3200c8c4cbff95f51566338ff31d88a379a5f721ea915bc123d194d3715.jpg)

![](images/f739ed613b4f93be1cc294f6534069e733b4d9f9528efdacfe853a1c9b989484.jpg)

![](images/2cdbfbb8baf6108b1da50cee83ac8c1a6858d6413254e3be5b1db780e29ab989.jpg)

![](images/edf63522c7c31710873a721b2905b87ba6221cfa6e20f1be8617494663402e6c.jpg)

![](images/861930749511de1ea31dc1d4dde12d0ebd4701b4bda426c9e0f2e430f71c3026.jpg)

![](images/94aa1acb8993515e30eff75936877f3c78339e0d2e670a464b9de9a971f2f953.jpg)  
Figure 3: i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

![](images/72792a489046fed1b5589025bef4f2b87f67105f3d32d5b4d3510f3d80bced6b.jpg)  
Figure 4: Prompt used to filter the HPDv2 base-prompt pool (part 1 of 2).

![](images/429f41b262acfaf4a1bc7de95ed9de7a9b132941101f45063c829ec4bc1e3539.jpg)  
Figure 5: Prompt used to filter the HPDv2 base-prompt pool (part 2 of 2).

![](images/55ad53a477924b829e618d1090816e5ebd12368f40602863176591d74cd73d47.jpg)  
Figure 6: Prompt used for image-conditioned axis elicitation.

Prompt Template: Generation Policy   
You are given an image-generation prompt and a diversity axis.   
Base prompt: {base prompt}   
Axis: {axis}   
Write exactly K image-generation prompts which vary the main prompt along the given axis.   
RULES, in order of importance:   
1. THE BASE PROMPT’S CHOICES ARE CONSTRAINTS, NOT VARIABLES. Anything it states   
explicitly must survive in every prompt: every subject, its count, its kind, and every attribute it   
names. If the base prompt says “two men”, every prompt says two men — not a woman, not three   
men, not a crowd.   
2. WHERE THE BASE PROMPT ALREADY FIXES SOMETHING, VARY WITHIN IT, NEVER   
AWAY FROM IT. The axis can only open up what the prompt left open. Contradicting a stated   
detail is the single worst thing you can do here.   
3. Vary ONLY the given axis, and together cover clearly distinct values of it.   
4. KEEP EACH PROMPT UNDER c WORDS. A short prompt keeps the renderer on the base   
prompt; every extra description competes with it and pulls the image away from what was asked.   
Add the axis change and stop.   
5. Hold everything else as close to the base prompt as you can — ideally reuse its exact wording and   
only substitute the part the axis changes.   
Output exactly K prompts, each wrapped in <prompt></prompt> tags, and nothing else.   
Row variants.   
• Word cap (Rule 4). c = |p| + 20, where |p| is the number of words in the base prompt, for every   
T2I model.   
• overall rows. The opening line becomes “You are given an image-generation prompt and a   
request for variety,” the axis line reads Axis: overall, and the task asks for K prompts “as   
different from one another as the base prompt allows,” noting that overall names no single   
dimension to sweep. Rule 2’s second sentence becomes “A prompt free to differ everywhere else   
is still not free to contradict a stated detail, and doing so is the single worst thing you can do   
here.” Rule 3 is replaced by: “THE K PROMPTS MUST NOT COLLAPSE, AND MUST NOTALL   
DIFFER IN THE SAME ONE WAY. Writing the same prompt K times is the obvious failure; K   
prompts that differ only in, say, the time of day is the one to actually watch for, because it looks   
like an answer. Spend more than one degree offreedom.” Rule 4 ends “Add what you are changing   
and stop,” and Rule 5 becomes: “Keep every detail the base prompt states, in its own wording.   
Everything it does not state is yours to change.”  
Figure 7: Prompt used to train the generation policy with GRPO, where K is the number of prompts the policy writes per completion.

![](images/a18fae527290e853a5d2bca18205234120ae4f8870909d2fe8682b766f4deb2c.jpg)  
Figure 8: Prompt used to train the editing policy with GRPO where K is the number of source images and edit instructions the policy writes. Unlike the generation task, the word cap is a fixed 30 words per instruction and does not depend on the renderer.

## A.2 PARTICIPANT FEEDBACK

Table 3 collects verbatim feedback from participants in the interactive study (Section 4.4), grouped into overall impressions, sets they liked, and some limitations.

Table 3: Participant feedback from the interactive studies. Quotes are verbatim, trimmed where marked. FLUX: FLUX.2 [klein]; SD3.5-M/L: SD3.5 Medium/Large; Qwen: Qwen-Image.
<table><tr><td>Participant comment</td><td>What it shows / what to improve</td></tr><tr><td>Overall experience (end-of-study questionnaire)</td><td></td></tr><tr><td>“As someone who enjoys using image generators, I did pur- posely try to use a couple of difficult prompts that assistants of- ten mess up, and this model did a better job than I was expecting, to be honest. Impressive." [FLUX]</td><td>Experienced users rate it above their ex- pectations on hard prompts.</td></tr><tr><td>“I would say the tool overall is quite good compared to other tools I have used before for image generation." [SD3.5-L]</td><td>Favourable comparison with existing tools.</td></tr><tr><td>“This was really fun. I'd definitely use such a tool at a reasonable asking price." [Qwen]</td><td>Willingness to use, and pay for, axis- conditioned variety.</td></tr><tr><td>“[...] if I put more time in to tweaking this I could see making something like this part of my creative workflow." [SD3.5-L]</td><td>Fits a creative workflow.</td></tr><tr><td>“The tool covered most of the variations I needed and was easy to use." [Qwen]</td><td>Covers users’ axes; easy to use.</td></tr><tr><td>“The task was interesting and the image variations were easy to understand."[Qwen]</td><td>The varied sets are legible to users.</td></tr><tr><td>"It handled everything I asked of it." [Qwen]</td><td>Requests are fulfilled.</td></tr><tr><td>"No it was able to realise everything i wanted." [SD3.5-L]</td><td>Requests are fulfilled.</td></tr><tr><td>“Really enjoyed this! I tried to mix it up with different images for different items." [SD3.5-L]</td><td>Engagement.</td></tr><tr><td>“I wish I could do it all over again. It was fun!" [SD3.5-L]</td><td>Engagement; every participant asked to be invited to further studies.</td></tr><tr><td>“Thanks for the opportunity to participate in this study. I enjoyed Engagement. it!" [Qwen]</td><td></td></tr><tr><td colspan="2">Sets participants liked (comment on one set, with the participant's prompt and axis)</td></tr><tr><td>provided an option I wouldn't have thought of (the stained glass material. Variety as a source of new ideas. one) which is easily my favorite and the one I would choose to</td><td>“[...] the assistant did a great job, adhered to the prompt and A mechanical butterfly made of gold; axis</td></tr><tr><td>keep." [FLUX] "The clothing changed clearly while the musician, guitar and A street musician performing in a busy city setting stayed consistent." [Qwen]</td><td>city square; axis musician's clothing style. The axis moves, the rest is held fixed.</td></tr><tr><td>“The variety in this is the best I've seen so far. There is life and color changes, lighting changes, and the backgrounds fit well background: city vs. playground vs. down- with the people." [SD3.5-L]</td><td>1 A child running to her mother; axis the town vs. countryside. Variety that stays coherent with the scene.</td></tr><tr><td>“Extremely impressed that it handled both the prompt and the edit requests PERFECTLY. No problems." [SD3.5-M]</td><td>Animated picture of a banana DJing at a rave located in a castle; axis animation style, banana's facial expression. Prompt and a two-part axis satisfied together.</td></tr><tr><td>“Really impressed for how well the image generator handled all of the changes I requested." [Qwen]</td><td>A preacher with face tattoos playing video games inside of a cave; axis cave, angle, controller, face tattoos, ethnicity, facial</td></tr><tr><td>Limitation: viewpoint, framing and distance</td><td>expression. Handles a six-part request.</td></tr><tr><td>“It would not change the angle of the pictures to any significant extent. I tried editing these, generating new, changing the prompt too, and it would not seem to work." [SD3.5-M] “The cookies varied in type and style, the kitchens varied [...]. The only issue was all of the images kept the same basic angle/- camera position." [SD3.5-L]</td><td>Camera axes (angle, framing, zoom) are the most frequent failure. Train on viewpoint-specific rows and reward view change in image space; check how far each renderer honours view terms.</td></tr><tr><td>Limitation: preserving the prompt at high variety “[...] the variability is much better, but the pictures lost my core</td><td>Stronger variety drops stated constraints</td></tr><tr><td>instruction that the alien be ON a cloud and the people NOT know she is there." [SD3.5-L] “It changed woman, expressions and colors of the clothing, but it also changed hair color which I didn't ask and was a crucial part of the prompt." [Qwen]</td><td>or adds content. Check each constraint of the base prompt, not only set-level alignment, and penalise unrequested changes.</td></tr><tr><td>Limitation: spread along the requested axis “They varied the right thing, but all the styles of Yukatas were only varied with colors but still very similar in style [...]."</td><td>The axis moves, but narrowly or with repeats. Reward coverage of distinct</td></tr><tr><td>[Qwen] “They correctly varied the races, but it's interesting that two women are Asian instead of providing a greater variety." [SD3.5- L]</td><td>values along the axis (e.g. four different ethnicities), not any change.</td></tr><tr><td>Limitation: stvle and medium "Instead of changing the illustration style, the ai has changed the</td><td>Medium changes (watercolour, sketch, animation style) are often realised as</td></tr><tr><td>architectural features of the original drawings which is not the variation I was seeking." [SD3.5-M] “I wanted to see different artistic styles, not just different outfits." [SD3.5-L]</td><td>content changes. Place style terms prominently and measure style separately from content.</td></tr><tr><td>Limitation: pose and action “I wish the images were less static and more dynamically ori- ented. In all four images, the pair is simply facing the camera</td><td>Pose and action are under-varied. Add pose and action axes to the training mix.</td></tr><tr><td>with no large emphasis on action." [Qwen] “[...] the poses didn't change, I expected there to be maybe some wings flapping or even a bird in flight." [SD3.5-L]</td><td></td></tr><tr><td>Limitation: implicit intent and context “[...] it appears to not pick up the implicit intent as well as it does with the explicit intent of the user." [Qwen]</td><td>Infer what a request implies (e.g. varying animals implies varying their size and pose), and read context in the prompt.</td></tr><tr><td>“I would have expected the poses of varied animals to be more different to each other, and the sizes of the animals to be more varied overall. I know this isn't directly specified in the axis, but feel like its implied [...]." [Qwen]</td><td></td></tr><tr><td>Limitation: what the renderer can express “I wanted it to show numbers 1-24 instead of 1-12, but it only</td><td>Counts, rare concepts and specific products fail already in i.i.d. samples</td></tr><tr><td>showed clocks with 12 numbers." [SD3.5-L] “I could not get the model to only place 4 female dancers and 1 male dancer on the stage. There was always at least one photo</td><td>Detect when the renderer cannot express a request and flag it rather than vary</td></tr><tr><td></td><td>around it.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Artifacts accumulate over edit rounds of</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>with 3 or 5 female dancers." [Qwen]</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Limitation: image quality under repeated editing</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>“The pictures changed according to the axes, but the artifacts that</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>were already there increased, making them unusable in practice."</td><td></td></tr><tr><td></td><td>an image-to-image editor. Limit edit</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>[SD3.5-M]</td><td>depth, or re-render from the rewritten</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td></tr><tr><td>"I tried "Middle," "More," and "Most," but the formation shape didn't change the way I wanted, and artifacts gradually in- creased." [SD3.5-M]</td><td></td></tr><tr><td>Limitation: varying from one chosen image “The main thing that I would want the vary tool to do here, is</td><td rowspan="3">The most requested feature. Condition the policy and editor on one selected image and return four variations of it along the axis.</td></tr><tr><td>take one image as the base and vary from that." [FLUX] “I would have liked to have been able to vary one picture at a time rather than having the tool decide which photos to vary [...]." [SD3.5-L]</td></tr><tr><td>Limitation: keeping parts of an image “There were certain aspects of generated images I would like to Region-level control: keep or change keep, like the specific position of a person or object but I could parts of an image, not only whole images.</td></tr><tr><td>only keep the entire image or nothing at all." [Qwen] “One of the images produced cartoon animals and I wanted to keep this particular image but delete the cartoon animals but was unable to." [Qwen]</td><td></td></tr><tr><td>Limitation: controlling the strength of variety “The slider function seems to be either very subtle and fine or outrageous." [FLUX]</td><td rowspan="2">Make strength predictable: calibrate the scale per renderer and allow a separate strength per attribute.</td></tr><tr><td>“I did want to vary multiple factors in an image to different de- grees (i.e. high variance in the chef's race/gender, but low vari- ance in the outfit), and I couldn't control the variance of the dif- ferent factors." [SD3.5-L]</td></tr></table>

## i.i.d. sampling

Prompt: "A photorealistic city skyline, showing a modern and futuristic city from the future located on a distant planet." Axis: “Styles of buildings, based on different genres of sci-fi"

T-DIG (α=0.5)

![](images/e62af14ea2f0e7eace3687addc24c04dc204ce688c2489bcb455c6e81bef914b.jpg)

![](images/772b3cb52edc3a648c16c3fbb34a51fe9a1cdd24f84b90b32c6342ff1cb38534.jpg)

![](images/726396902c0bfff40ab8723b288aabc55a2da33b6c734bf0bd3012ed06b5f712.jpg)

![](images/68374dc54c44585ac3a3e82cf7d73eb18bacbde26673b53ba27ecf901b0c8fd7.jpg)  
Prompt: "A small mountain cabin beside a lake, surrounded by pine trees and mountains, realistic landscape photography" Axis: "season"

![](images/ce1e02e9a8ef03aeda8509d0b2ce6a04826c21e49141de9e8748e2f5b7478893.jpg)

![](images/ff48d4297eecafbd476a5bbf5fa2929d24b2e94632d00f7a60bdeb00a411a70d.jpg)

![](images/42bf066028352833ce4694014fb5daebdfdfc3610f9de10d1e01889fdf589764.jpg)

![](images/42bdb3bbe0e37ee94ae033ef46193a5b55096ed14a4602b6f97b84a760355d61.jpg)

![](images/2f67cd3f124c468159c0259e91b79314d30a003e4d86338f063e1d72c608f321.jpg)

![](images/0c2f24928ccb989c1d3955634a6274b5ac823770b1eb1d9e1843b00bca1948ab.jpg)

![](images/e5f78c2629961340617922449f3ae6008fcdc8d394cd77f87400a70a5f119aad.jpg)

![](images/86bb8e5d2cc53ba09dd9e5cdd81a2e400e2b9b8d592b0dbf624cd587fcaccc45.jpg)

![](images/7787ac65d3b7b9fe57713f7d5460de5cf681e8822b818a5d05706315a4177495.jpg)

![](images/46ac5a8a023d8765552e13db5777be4577050f8a7814d6f34a0099ae0e0bc132.jpg)

![](images/9d0b220c5af7bda1c1e7ce61098b9e72f9ea15e635f12cfd6dcec3229f482d7e.jpg)  
Prompt: "Two steampunk robots with exposed clockwork internals are attacking the viewer of this image, directly swiping and punching towards the foreground."  
Axis: "type of metal each robot is made out of

![](images/53b062d77f527964c3284c54b5efda22751b0980146f2cc25f125c87b7a87041.jpg)

![](images/9c68365ebd491b92f7d47d7f0c1c086a5b51ca580376f8a74932e2e6f7aae534.jpg)

![](images/36443f74e12c9340179dd83e22ea76c5fbd3bcae20f3a884a7727bac5e91c3ac.jpg)

![](images/e87889966961b6a60f425abb270133f9c2a9226b2537c014ccbcaa7640a4abc5.jpg)  
Prompt: "A cute animal playing in a shady forest amongst the trees" Axis: "Different types of animal'

![](images/d68be978df8c926f89986c156adf50715462b0ca20895e98767cd5871ad0e762.jpg)

![](images/8ca0159d1566052083c7f0cf4223f11eb7c5ecccf13e442c5232c3141e18733e.jpg)

![](images/d6645cf498e4b6ff4404ecccc6f9720137ed203b1fae00422316abc24c0cdd85.jpg)

![](images/6dfb7ddd2331734c4bf289fd39d0371f4c57a05fa9e2fb93e373e29a3c6cd310.jpg)

![](images/4e1ae9ac996e3f6c0476968fe0d07c3a9c68a8b45686a6cd3587a6bd8d5035a6.jpg)

![](images/fe582c8211fa2bf8f85c5136d3ae5ddbccefd39dbdfce43b4febf8641ad7151a.jpg)

![](images/39b7dc34c349c5e15fda894919d13961b6af50a394c3943be5db5c6424bb1b97.jpg)  
Prompt: "A busy market scene" Axis: “different ages in time"

![](images/01103387900e872acba96145306d1e8d249fb9ee2fb17ce3b246d598bf58881a.jpg)

![](images/148f4b0283d908551635dbca120581117e7aedb1800772aa8ad26543a18d697d.jpg)

![](images/9d0e0ed3bea3c0723735cb6cfddbb57c46ae490cfa63453ee066a028b4aa6216.jpg)

![](images/a54edd3548d901df6b98a42cb4179b1afe1d74719233c127905ca8aa8643298a.jpg)

![](images/fb5211155395852b6d58c618248a0629f4a7d9badc54018e93adc1afb7dbff03.jpg)

![](images/40eaf6bea0d201c1de1ddede3ffa122f6df0530470996561aab98719dc2dfced.jpg)

![](images/0ce7f2255a88f2c349b2ae70b504f3ba6aaeae960ee29068f717bcfd9c7bea3f.jpg)

![](images/c3aaf6f35737a2fabb7cf30e475403243e7988112b6e4f7b1de45407d101a7de.jpg)  
Prompt: "show me a snail wearing shoes" Axis: "Different shoe types and number of shoes"

![](images/2c556dd216323e08bc211b9a9551f6c502f8eefd9c99e45d2070a50984a8b0eb.jpg)

![](images/efcd981e24c39d32180aaa6415d63047a5b77e334ae05e94652763a4b398aab8.jpg)

![](images/3e97c6c482262ff639484c82c1c4632ac951af95d3e96ae7063f51b8c83a2353.jpg)

![](images/f90fbc9443a9da2dc73571df477ee5e1dff344caec850f992b2ddf2930bd1f8c.jpg)

![](images/01f1e6ab7aef35a196b012ab76dc6d19b4e973a6a91beaba8c3d36e8b1114fda.jpg)

![](images/e59ce8c92489d319966aebf96a43d748499999ebfee5e764540e26ca3d456dc1.jpg)

![](images/97cb71881aaa7c625ec9b8d9e90d2a545eaa48b2d8513dca44ed43ee0bc9ac13.jpg)

![](images/1ef2d39d7b2ed9b9f449958a3ca27a8f3f791c3f77bb28c2e8b9531e6f675026.jpg)

![](images/7e8748198a8d31712b7aa471bf7d6e6689f02e58386a68683f0841aebae4487d.jpg)

![](images/1fcbbbe8fae364c9af64c9b43ceade062a5f30657a2844a64f5376f41445fe95.jpg)

![](images/116fd7dad15fa871107980a68a01ac0f26685072698161485663e5162b804286.jpg)

![](images/c1768398e10f0b75c40eab03f26297ec2665785240078167d79da9201e24c4d6.jpg)

![](images/ced956525ef2b38a6b038603b57ebb59be209879fa7dab75ab645a6b64716923.jpg)

![](images/898ad4831f026baba6637e9b572f83774c09c9c917cfdc9e751c6bcd609dc8f9.jpg)  
Figure 9: Qwen-Image (1/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

Prompt: "A group of three people seated at a table sharing a plate of nachos. The table should be clearly seen." Axis: "the background scenery"

![](images/649a22a2e968ad7059d53151fd6400f85e7fc8430c5e0f47b2d3d7f4d8c0a786.jpg)

![](images/c722c0f9ad13aa1e5d6b8f0ffcdd56f58faad9d1972c57f8cdb435ff221c0c93.jpg)  
T-DIG (α=0.5)  
Prompt: "a tropical bird sat on a branch" Axis: “"the background of the photo'

![](images/3e2c6589aff7f84818f9413924aea18e34a942b79d203e0de550a9c24a661b70.jpg)

![](images/448f3ece2cb2dd0d30641b7c17a5389862ba146b20a3248db630214e49459d42.jpg)

![](images/172d3a1cc8757d00598635aa985e70f984960bb4f0626e2d214e9ddf98ba080a.jpg)

![](images/064d4fb53828d49881ec51091720812a65b63cf0754d26afcd2ff771e3135f8f.jpg)

![](images/2b1efca19db307f98c1241960643f8acc36045ffe8db8dbff41da792c6c75c7e.jpg)

![](images/bf9249e94ae5ecaa3dcb51241bdf28b3a8136443223d349c2eb758d395c3c396.jpg)

![](images/a60929d1be200f59f04908a9c723a0f2beef9778a1c789e4e7de57cf089138a9.jpg)

![](images/f750eb9694e23a298dfcb4c99978b0a9277b390caf5d283b110a41743bfe717e.jpg)

![](images/7f63e5ff0e8a7a41747ff829872054b186cc1b6cd7bf871cc253177d9ca208ae.jpg)

![](images/5b2cab5d37dc5de4f45a02a5d543a28829eeb511bacbfb2eae68fcf6265380be.jpg)  
Prompt: "a backyard garden filled with lush foliage and a variety of flowers." Axis: "the colour and amount of flowers"

![](images/0fa759d539904a783b31c8cf73735f4b203b0800e572baef5a859ccbf639ded8.jpg)

![](images/0b16f776f4a37ba738d4ddab0508c93fc668b2cdc64804dd861ea8db078de6e8.jpg)

![](images/e16cb1d423782bf269ab17030a894970bf25881daa9a14aa77a802130d856d28.jpg)

![](images/0bc9ee455816b8de83fad02a89aaf71522bde93f9e9754860d9a24c2d3201dd5.jpg)

![](images/a0323be96f8d8d0510351b3b20cf424197de882f924914f3ad274736c4df4f46.jpg)

![](images/a11e5fa851e14be3dd7c09d29e248e2beaf5407d949b199d1e65e1c965490e9c.jpg)

![](images/26ae39d2b13814922cdd019a4863ba10a80e9762c25609f83eb4a236e25c2a29.jpg)

![](images/9eb785859bf1ddfff22ae8c6710251114cefac531ac9e0740edd62966a39a4fa.jpg)

![](images/7dbbb0d6b4f82f6e79305c9d1f706a4a4e1b3565a3ce0e778804333988ae8934.jpg)

![](images/9d070a09945cbe7f4f12cff30f10cf19101373abca9fcb3a1b6ab9b8f7852ff1.jpg)

![](images/dadbccd522b46bbba70ac871a1b128b6cc55d1aec246d110cf7c24cb26b66843.jpg)

![](images/cc734f48606921aabe46bac46c446730261e115492bf551fea7c138a1425e0d8.jpg)  
Prompt: "A charming neighborhood bakery on a quiet street, with fresh pastries in the window, a few customers outside, realistic photography" Axis: "bakery architecture style'

![](images/8a8351a3fd4c58c8b7b02b7156c59c58c211f995e89e43a64c03c80da514db99.jpg)

![](images/5aefe96bfbc82ec2edb04549ca1722818c2b22e86489f3b06c24599c35b3122d.jpg)

![](images/de21edcecf7a62a152c5dcf3f429cb93ab780a548f0fd0110332cc7a590d262d.jpg)

![](images/01d47fc196467b680148a9f00f29749914673135ea0487a9b0309b42dfb2bc84.jpg)

![](images/3694e34016f7b1cb23fcd1528ddf2e03609e3a23db672184499425c0c005d859.jpg)

![](images/4487e37e37507c024e92bfb272412c9c4c213b3cf50de4d24a7d22c22937726d.jpg)

![](images/a8520359d49ab113fcdbae41ef176e4fe5e1f956490874b1f435d72c9275f07b.jpg)

![](images/8928acf52da455c4266183b7d4905197d25f68d1076294dcccb47b5ec3a31285.jpg)  
Prompt: "I want two players from opposite teams locked in concentration during a sporting event. Their faces are wrought with concentration." Axis: "I want a more dynamic image of the sport or competition being played. Also, could the actual sport in question be varied, but distinguishable to what it actually is. For example, rugby, hockey, and American football players."

![](images/008a3a81c312838b4e047717bbbb6c170438e014149def232adfff7af8adaec4.jpg)

![](images/bc10b8ac42b20b48f8f14f96c50cfa624bcc5e4bd51282945b419c66b2a1c197.jpg)

![](images/2b3250fcea24458b5b36958941297099c06c04dd571f16e0d08d04c550657ecd.jpg)  
Prompt: "A family celebrating Diwali outside their home" Axis: "Make it night and use different lighting style."

![](images/21e57be0f3c5514566f346811c2a78e26fb85754dd3db31e83b024b9cf889da4.jpg)

![](images/42f97532a8c6e140cf332b519dbfb023a50b50ecc6609f2f8bac9b87f6cf2ba7.jpg)

![](images/1181bc6fe418492676dabc3aaeea2655d0e74c440cfc67d5486a59e9ad057622.jpg)

![](images/540b1f206509f7598dc3460d69f4c879a6b8d74617e2316b87e86a21468945d5.jpg)

![](images/8313e385e455517f49a4a7dc9995f72c13a78dc79fa874b7234f5a50522f517e.jpg)

![](images/32b73f6003556b319df40e67a8b1de92090e8fb166c644f5f6e959339c48752b.jpg)

![](images/5512c14b18e3307b83eb71a8b3cc25afda7956590e677bbce52e1a925e986745.jpg)

![](images/5bc936623ab52e94b4b87897d0b073304b08d02f746d6ae2b95bcac2800eae4a.jpg)

![](images/0a0d5d8c9511ac11326f99c1997992baaf335a8cee3d6850b00108b1851f52e2.jpg)

![](images/7ad89121abe68396a84574678a811182d8178045ecfc493bb5efb7db233f124e.jpg)

![](images/6dd63418a341aed6b62502a9285b57b86ebaf1adfc3583a96737b2e838061b8e.jpg)

![](images/8fc5afcdbd0c74d123f98461052d5de16f1a11466307bf34fe9601567b8b325e.jpg)

![](images/9adea1703eb0422bec93007459ee8fab0ee495525c99c185dbc336880fe4c6fb.jpg)  
Figure 10: Qwen-Image (2/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

Prompt: "Workers replacing a roof on a building" Axis: "Each image should have a differnt type of building"

![](images/8b39c9793241f62e20f5dff62ec3c5449bb9a2933492c1c93d1eea8288c64d7f.jpg)  
T-DIG (α=0.5)

![](images/ec654af1e7ed3d934591ce81e432fd204a82c2f266307b9df9ca1c2d0dda7bcc.jpg)

![](images/dc1183443034142e4c32b3787d154e6297b6c64aafca26bd77983ffdabb8e8b0.jpg)

![](images/cfe34e96bca0966dd8aaedaeba59b8066a210b50044bf22df32f21baf9b5bf34.jpg)

![](images/8dad11f8f0819c993e8c109ae92a30468700f407ee599528ac78ba64859ac8c1.jpg)

![](images/62bfd8861a496e36f5fbc355537b98f892022270a9c241e7205f270e1410b95d.jpg)

![](images/93056d6130099467fb462e75e5c1c041eab503b21633caa9a4fb25325202f256.jpg)  
Prompt: "A scholar 2D stickman reading a massive, ancient book by candlelight in a dusty library." Axis: "color scheme'

![](images/610d8b6ad656c78ef3f9449607877d539096b422a872d9ba3a727d6dd7053763.jpg)

![](images/52f12f93c2567c67d067e4b74a164cf42da37aade902e7170ade25448b3511ec.jpg)

![](images/36666b5d5c7631ed8850d26748da047639abb368e530c5694c02afdbf90bd539.jpg)

![](images/1057136d149d22e65cd0ade629b8433ae4a690930eaaf79b42d0fd09acbab05b.jpg)

![](images/b580d1ae226a6c999ee6fff6bc565b56afb4705bbe72221931ba143db922a69b.jpg)

![](images/3bf28ba4077b360ff22283927b279d642a3915570df1c47d2d1a0eedfae42f0f.jpg)

![](images/292685a8673ad6b1731eafce03fee4eb11babd178deb8771e87671a24fd82754.jpg)  
Prompt: "A colour swatch of chalk paints against a dark cherry wood table" Axis: “Colour swatches, not pots"

![](images/aac49f556abaa7c76a91f348de1907d4da58e3e1c846cabd05fbd39f5af03bcb.jpg)

![](images/dd6af7acd1be44c9842d8292f833ef832f46407edb60e6b3ca89b43c79f36364.jpg)

![](images/f1a3690806142bf030b6e1d688c61d2f502fd8204f286464b406a61c4352d540.jpg)

![](images/175e5f2e775242dd516198101eea88b9284eeedc82c74556e16a66757b18a5c4.jpg)

![](images/f868fcdf5ee2daf039fe0233fa9f10cf0041ac1994cf21dae81419f015c4eac2.jpg)

![](images/ba88feb9b726c554edceac8e71219d098b3dcca576f08a22fbc7b711263902db.jpg)

![](images/157c88ddbc3383b9ab9ae166ec48e2e767d4e94f0681f684471931f8e7ffafa8.jpg)

![](images/d4d4544ec35337256c77d8d8e15b88fe3f12cdc05e5a3860b59d79e2aedfaf03.jpg)

![](images/0776fdf3e2af4165661221ab07794a43fbff3cb245957f25a7b71d95e72a30b6.jpg)  
Prompt: "A pizza being served to a hungry family at a restaurant, in the style of a colourful anime" Axis: "The style of anime"

![](images/4995e00ecdfe047b1ac84b67116c40df4569e155f4d717ab5f473bcbccc6c013.jpg)

![](images/a8bf0db1def66115885b2d85b15c9012f9de34f878e01942e797ef221bb33c01.jpg)

![](images/cad6ed3d1aadac84244432a15df12caae0a1585adcaeaf9ebeaf922f9b95b090.jpg)

![](images/25a9342f21069fd484608924f1d576ff61feaca786abf51bffd52a23707fe6c2.jpg)

![](images/771e1a5517fe8c1e1749ccb153f07bb0e90fb16b80775b3b5bf8ce688076231d.jpg)

![](images/8ff81810acc33d27f378ae45ca473f3013f27a6c5b3e7f266b96f75c8daa5d67.jpg)  
Prompt: "A rockstar performing a solo on an electric guitar on a stage in front of a massive crowd." Axis: "add these effects in the pictures - laser beams, heavy strobe lights, columns of fire, thick smoke"

![](images/08ccb24c5d350641b79e0d5502295b7a209926b22a497a1d922a2fcd4e202026.jpg)

![](images/c0c16cac916d43bfc69e1ce18b0ca069efd037e654dd6aeecc193fc0b8d2ad71.jpg)

![](images/6fe9ecd65bb6b8216b155f0432c3bc6565dc38a74c82d120b2d57700daa4a6d9.jpg)

![](images/7fad9900425e02a9aead7d74fb698c0cff44aaae066f0a3555f8859e588f05bf.jpg)

![](images/460a29d94539a27edffdf7d0026e7009bbfa86cba43ea7aafdeefd401a6fdff2.jpg)

![](images/29c341b1da67ec5cbc4675963ade05df95eb8446a7abce5617682b5a2eadb03f.jpg)  
Prompt: "A server is serving food to the customer." Axis: "Food and server'

![](images/d60ec4380f8027aa4176d974fb7228e2ef866bbdbf22514ef4d63486dc379c15.jpg)

![](images/749d38ddc1cd474c41f99db548360e7679ee08e6593d9589df12277fba6a44ec.jpg)

![](images/fa65f18ba2bd0e39b9dc9d1acc872d8c2d5fe398e227e4e6014ed6618e26a44d.jpg)

![](images/2690fd0c20b11361b788a7f5d7a8b8041b05fb9f67f0d0f3b58e5a8564716fa1.jpg)

![](images/2c50d8de9c8d5056507306112d2194eb20529c6ccafedca2e5c703aa3f545429.jpg)

![](images/af32069c1ce2eec85e37d80b41b3db4829bf7a9e78147d3f50ab8891fe6d0ca7.jpg)

![](images/a49ea95cf844ac215f112a337640518b1918764bb741569c6b8d7cc16f3dabc4.jpg)

![](images/7587867f7195f2e1ddd62c96110d8f96a6cfd416117ffcb050f6a0dd3b29fb08.jpg)

![](images/b182557f4c42d7bd0238944db27f1b5ff943197e5326613281128088a99c2b5a.jpg)

![](images/2d6e9d3578c1c049a8475b7a1e4c62188107fb3fdb2774c8da914203a6feaffd.jpg)

![](images/7f3c40ef7ab6dd64880bdd649e3bf95d3251b9a48334f204317ee02c80a09296.jpg)

![](images/4e75684221fb075350ce3acc20fec99ac3e73c7cd4a24db29bd4a6bb34ac2065.jpg)

![](images/b341a37ba3df30d7e406e229eb1c11301707451330a88e563d35ecbd4ff7065d.jpg)  
Figure 11: Qwen-Image (3/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

Prompt: "Two cats dancing together" Axis: “Clothing"

![](images/835c8c62a020fb63a6aacb7b2089ca84e5442072137a2f4006f2a4e1fa8f1965.jpg)

![](images/ee848fb1b318e1a3fdc8a6037267ce20e1e7c097dbf56e0ffa84ccd01d87e55a.jpg)

![](images/2669e21ee0bbfae314755390a3028da05e51d16d8a58b6da77746e3d96f1533f.jpg)  
T-DIG (α=0.5)

![](images/f4206b3887cb198bfa43b9025e80ff2707976bc34106503b0435f09f711d7cc6.jpg)

![](images/d566d9dcabf6316bcedd336e3ff1d5097de8af184f9360308936cda0dddf3df6.jpg)

![](images/b991bf253c44051f29309e32a43cfd42a2a0ac31e101b5878602cbf3e0bd38aa.jpg)  
Prompt: "a room with no people decorated for a chiid's birthday party" Axis: "color"

![](images/91a6a3d79f2e75ab4fa36fa6a610a7a9df74f5342f5634ae1ef744676c469a32.jpg)

![](images/847f38af8b9e5cd121a8c0c572c893935206ad757abd413b7c244c8a490b0bc3.jpg)

![](images/1cce4dc135c71ba0f9a32d5defd3137ed0742ed26076251bf881e6375a3be9fa.jpg)

![](images/5b0e2d213c793d74afe414e8c0ce91ae88f304e2477db33c55ce80208535f4c0.jpg)

![](images/a04b44050566ee6d5e2e1fad3f6bf2f10333ac609e7c17d195b6dc140cdd06ca.jpg)

![](images/00bdff26b94bccc0bacbe39474a6bd4643d291246fb437c3a227c0220fd40789.jpg)

![](images/1a8fbdaf42945db7c057f1857462e63739af4355e009b1cd85eb0a0d64329e4e.jpg)

![](images/12cf1d1ada79049cf02641d15bec304ee48572a09eecc9e94d80c448b35dff08.jpg)

![](images/38cb47f238d08280bba9040ca481cfc0921a211bd1f0aeb1984db8d56b669210.jpg)  
Prompt: "a painter is standing at a scenic view with an easel, painting supplies, and half-finished artwork" Axis: "vary genders and appearances, vary seasons, vary scenes so they are not all mountains + water"

![](images/0e679754c245974bf097b990f8837da4e1a6dedc7d9c364e682a0880fbb343bc.jpg)

![](images/d0584426d61dfa95fcd370dee085bbefc5ff388a7d84818fa85a78fe3ba14a62.jpg)

![](images/4775e48f849b662fc3c33d4580779924cb27c915c4ba1d8950797799e9bc754e.jpg)

![](images/72c9aa84f84fd0a7191b075150af60d6c299be09c46b73ef0ac16d2a38c93067.jpg)  
Prompt: "an astronaut looking up at the stars" Axis: "constellations"

![](images/f420679af1ec3f902a999a8bb0cd4b5c06b1dcb8edaa3442607d119e25d4e101.jpg)

![](images/353ac0793f674759f32d3f711793c1ec6db71ea30fe0e039c95cd55a58c91c43.jpg)

![](images/d4f61cbf039ccde9f3b92f4a3bd5d7446243d0e7dff0644288298caf329aa826.jpg)

![](images/4be30d27070580d7f7962d9d95f763aff880fac824821fe7edfac6fecfab46b4.jpg)

![](images/60ee75e79e3baa5c3f3f192504e5c6a8fd637192862010e394719a30623e2c39.jpg)

![](images/723beca5b59fa994021c848b73c698b915af2b5594244e5d9cedea69d9cf4cf2.jpg)

![](images/668324ab177e98f6e3da3fd622bc8d3201ba888a3208b4a2cc7ce8919dc1e249.jpg)  
Prompt: "A bakery displaying multiple delicious looking cakes" Axis: "size, color, height, slices"

![](images/d061cc6dcb37db0b8795d1584683d47a802d6782227bc33fc1dfb9cda6860867.jpg)

![](images/42f5b2b9e77fd19d5ac3fd2b01b4dfca4bed9b4c3cc41d0fa4312116d7ce7117.jpg)

![](images/21917df2e4042b946734e70ac04d1fdc204f89e1e52595459062e7f231b1abab.jpg)

![](images/1fb8d572446b5f02ccddc3ff4cfb1f9a8dbee750ad04cc32723dbb5eb33b56bb.jpg)

![](images/c79fc3ae56f5efa9a2807bca0f798dcd22b7373203d25a5278631da54e093d63.jpg)

![](images/4a2b62d1fb38d923b9fc96d69d151255bd8f1c838adb3185a230e61da7eeb14e.jpg)

![](images/51be57464a4fffeec1fd768a67e8aee496e7f34164288eeb21d5967308c6caaa.jpg)

![](images/bec5ff2843f185119c7af99f13165096c79ec0f348d528b1061ff030b7f4abfd.jpg)

![](images/f8d38ae8395f2ba94b951813c61624dabe6ec5fe38914923a9b794cc3a3f3e68.jpg)  
Prompt: "a cat playing with a toy held by its owner" Axis: "the type of toy the cat is playing with"

![](images/bb582a00091d5652ed4ab1fd50a214cf51feb63c11139856e83c1adca2aa366d.jpg)

![](images/a97f7fa56b65c25232170c3524bc61f491ceeae1f98c0589f083c98650f448d7.jpg)

![](images/8aacdd07bb4181c76435320de3cbebb703f73f0959066905b1555034a082f65b.jpg)

![](images/a4e97175ad79ebc5a51fbfed6d687be026f14ecc94bc4c6f21d7819f6625cfcb.jpg)

![](images/1890628627ef4e9b62ac6bbb0f0739654223c7f5e50ee585fe7308b70cd4aa86.jpg)

![](images/bc6635daee32c9b3be940aa5a30d2e2cc9e689f854f7a0cda2ee50f6672d1d8c.jpg)

![](images/b0e3b57b7068d3a606020a0cff9b566b9a14f33eae1ae18d193a30a9263713b5.jpg)

![](images/b64b7e412ff508700214c74fb7cab274e8926e08988df577d8f9781d923d811f.jpg)

![](images/273e1f990bfc94c20f61bcf56c77eb773c3de618b335b6bba0bdaaa9da90f57c.jpg)

![](images/d00eec96501a4161f42f7437a8c4218530cd6a9374f8a6c2a5b0b18f1f058ded.jpg)

![](images/64c54d7654871c9b9feddb51cc9c840f2d2df11574eb57e548f7ded63f686597.jpg)

![](images/52acf9a625815b7b7d5f1a92b47f0d4efb5518f36ed86dfe6236614de54343a8.jpg)

![](images/908a3a0fa4bc84ebdb6153e4629c8ca2bf49c60ae3306c0565b344a9ba9c4232.jpg)  
Figure 12: SD3.5 L (1/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

Prompt: "a delicious plate of food in a restaurant' Axis: "different country cuisines"

T-DIG (α=0.5)

![](images/b751a2d7ca92b77a8344536643114093edc5b7693f0648a7807adce0eaf00e09.jpg)

![](images/f2f58315eb18448f5f49f0cfee36ab6992741ba9bf53d46f002bd1c851e257cc.jpg)

![](images/aab96518e3e93153073e1f8002a5d254a91c2bbfcc8a941ace9007ae72fc5a5f.jpg)

![](images/f532634ce971766dd88b17d7dd2a88e59311661bd99862a8b30b25cf94f46e5a.jpg)

![](images/b7c7bdd76dceffe0680731a030e8e67f5786f6d31ab2e978308f3d4ef4e12f90.jpg)

![](images/9833d94db2a999c70af626030f8024612ac19ae221981e3206dd45d9e8304402.jpg)

![](images/c08fc5034742bc437cc681ea729ac1bb42723059b000d4d0e87f7c19a136c7d6.jpg)  
Prompt: "a crocodil making balloon animals in front of the Taj Mahal" Axis: "shape of balloons, size of balloons, time of day"

![](images/0c616d124332dec27fccb56cfc1fff32de028693f697162a7d6e19acaabab33e.jpg)

![](images/2116e8e580ee46ca450f5d311734d32d11334b349ea8b51afa35b9391aa869f0.jpg)

![](images/d8a05bde76e8b92dbb7980d6f5edf3f1a4e5730261988521103113d470a59850.jpg)

![](images/cd94cbdff037d14ae4275ba7073ce48fad503c974a866c5c96abd9cfb0856599.jpg)

![](images/ba252d0e17f42b5a090ace92c1f4394581837969af115f276b009022034d06af.jpg)

![](images/6497295e5056e24d270cae2c66a24ee2ff4232238dba84081b865d1f2b14c2db.jpg)

![](images/cc8b8764ef727cac07ce464e7e8117f542a4fd51183fdf328f5f7c0261ef71e1.jpg)

![](images/817744ab83f190297c35832f1b0d7e23bda139c9be557d4b65a304a8ce16391d.jpg)  
Prompt: "A crab having a human-like job" Axis: "Different type of jobs"

![](images/14882a1eea969c6d0e74b31781b4e587c81a1c6a5e05b7026ba5bd3fa81755ce.jpg)

![](images/fc27e89f975145c28a7ef9414913bff3992b6dc6a0be0e09de5e1235c91eb2d8.jpg)

![](images/95976d95eb21bc4f33b95bf127cbd291e863fae06caa73fc73400a740195f81b.jpg)

![](images/cee38f45dabc54c62b3f2c85b78b8d02ff92a78cfc9184429c8c0eec35b412d0.jpg)

![](images/3625eeb219ef536cc42434b291a7ae294093e05160fc9a85509a957db2caf716.jpg)

![](images/0645f3fab80c8393ffa1b6b16e03b1959651a5bdac9d67d2b3c5344bcf2ef127.jpg)

![](images/4785e1b86d78cecd49a80ed2091565665268bf0b7be22ed35434a595d417405a.jpg)

![](images/45cf43b79e6656dec5ad405b444c644e58d1c8e366cf0ee865203a000a25b0c9.jpg)  
Prompt: "car built with Lego but not yet finished" Axis: "car model, colors, completion level"

![](images/f3884a13512189088b486e69d1522b5a697dde266e7906e19e64ad5e91e70559.jpg)

![](images/d8ffcb5dfc0046a1fa8ca514baabc80d68883ce11efd9df9c95e6d0e00ac8b6a.jpg)

![](images/d8f30f15fffee6b0cc8cc5311fc62cde21e9f40735153167b917712091daa6a5.jpg)

![](images/479b4d1d3ade7ad3ab08b0d305fc1b5b79ec95b5a0aff431826b11df385ba997.jpg)  
Prompt: "A human sized fish wandering the streets of medieval London" Axis: "puffer fish, autumn, stormy morning"

![](images/1f78363bdaf04be3a77aad304950a9b457e196cb49b6a015c74d68335a6a429a.jpg)

![](images/0defa592dd4d5efa5a5e7e4bb0aad94c2776ae6324268220369d42b41ca6d4d4.jpg)

![](images/660ea42f35114563a92781aecc31d19fbaadb1672788ce4925fe092304792c62.jpg)

![](images/4b912000d61ce76fd4035f1ecc729e41e3f9b84457a6dbf6f365ec4a2408ba8d.jpg)

![](images/e83895dcf111bd0d26af85228ef7f0c61701c9b1d9322c31ad96c3ad32dd40a4.jpg)

![](images/b85882115c370e9455a2fe06a27eefe5439e39f1ed2ca501f54629cb7431ccba.jpg)

![](images/304698a398ebd7ed080599d48d93773313e41e7fe2bf7d6eeeb78d09aac4b0db.jpg)

![](images/fd2960039ed06c5e69c5ed2b69355e1816cc4e5a96c968338769cfd97801dd57.jpg)

![](images/885371290815cc884f77875c66c232ced9ce69563cda74e1f6a8b5244b0110df.jpg)  
Prompt: "A f1 car driving at high speed." Axis: “Type of street, colors and weather"

![](images/bb9e11563510de9cfb42d08975a90d8187b27e254dad22fa34eaa5d102e7239e.jpg)

![](images/602f9b92b4dd517c177b1ed769986221fe87adcab78c615329ebe59f164a3850.jpg)

![](images/665434e37f72c825b5d3a992864db602e257696f8a7f056ff49418618128b556.jpg)

![](images/26a2b124df77831e28770763c020de16442f3447eb5dd435c6adfee4d7827527.jpg)

![](images/a83ab3a030175b60c62fc34b698b9e7f3f8152cb9a29a27a40e8810e2fb34106.jpg)

![](images/33cb06cfe7275a7b57b6031b5f67e0a092b0a43e5aac0cb13a70e84fa24fee57.jpg)

![](images/b677a9269e9873ac9e1247c1307d0ad8d42855acc215c0ff1b0681c75e2eb988.jpg)

![](images/adf94b219659e3bfbd07e5996ffa4d49b5fd8c5f53f6744d723ede2ab22b1215.jpg)

![](images/a978aee715bb1cd9a023a367071e14680f681ca1fd163188de9f610683b5be4e.jpg)

![](images/1e191bc99d6ee4f90234fc310d6f64878518a491d6269691c33269c5572c6061.jpg)

![](images/f5743bf0493e9b4b2bf5ea3c53dd23cabfa70aa48940d84c6c6a6ed810dd3b79.jpg)

![](images/072857bca242e40916ebd3445efdef0823f32809af1ec6bbd68813f395eebf88.jpg)  
Figure 13: SD3.5 L (2/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

Prompt: "cows in the desert' Axis: "cows"

## i.i.d. sampling

![](images/961fc30edaf86ce4e786fb98fb5fb8fc9b1f99ab551b897d3a9156b5dd7a7fc8.jpg)

![](images/f7333ee2a45037fdc1dbad178bb53aab5eb5c1b3cec9dfbb4c1f3f2af5253c9a.jpg)

![](images/5815f40f2a6f650c37d95848e78372862b1ff57e6bb3675872fdc0800e25dd20.jpg)

![](images/b57f6f17ab75478231337f61b751d7ea97c7bc354802e1d03f38ce3efd9cf45c.jpg)  
T-DIG (α=0.5)

![](images/44e595ad44167ddf799eae0c1ace89e158e02c0676e4bedfaa7df9fbec3c6add.jpg)

![](images/b992b6a9fa3e74d4c828cff1dbe1669d9526df0a4e7c7c9a0df787195e1273fd.jpg)

![](images/080fc89bf73e76d1275ab97931fc0d6b460107ff7b19183936da4a854c23d317.jpg)

![](images/70ff8c6bf51081b68b3cc9d9e2dda643a85ccea5b73de27f04c94844604618e8.jpg)  
Prompt: "a dog holding onto a group of balloons that is trying to lift the dog off the ground while the dog pulls backwards trying to stay on the ground' Axis: “image style - for example: steampunk, anime, realism, watercolor"

![](images/5b81f8b3588f96ef54b818468243eb6c00122cf5b555880f882fb0789f7d203e.jpg)

![](images/8f05b891536c3d04108f08e634ec51c277dae59925fdd8967dc95e46161fe313.jpg)

![](images/121488bd70eda33d8bc6a0e59f8c08fc2505edb3b87d603a153077c5b0f9808f.jpg)

![](images/e309a3e8f8814c2b0116bafe52fcf35b5ca2e3df4ef706cfec4a4319bb3895a3.jpg)  
Prompt: "A grand piano made of glass" Axis: “design of the grand piano"

![](images/909b586a11697beb3edf8cbe1840400ad18a31c1ba7eeaf0762ac1aa60ccbcdd.jpg)

![](images/448071d377de6283b588908b72bfef0d51aec73cdefc07297471e5044b01d242.jpg)

![](images/17ef5f3b1173372b1b337a3101f27a7393696afdccd4cd132845b22f38140788.jpg)

![](images/ac5ddecd90276bd786325a96857b6888a47f7e5ab082d41dbb49a64b41b3fa42.jpg)

![](images/809d9197b8d19e0402aa66894d53099e23c9afa6c1a935e884ded96ea83a95be.jpg)

![](images/06f11e63308b699a4c8e7c5508bef8945c3a0c9c43d19f2d43b29c601c2d00b2.jpg)  
Prompt: "A storm unleashing rain and thunders" Axis: "Time of the day and background'

![](images/a6056e6c2f1ea84fe924598d01ceeba3c84926ef6cf832314577d3c3e12b219e.jpg)

![](images/f72f40b8974dee9a5508556a5630acabbef95ac2c46afd90845536be3f212a67.jpg)

![](images/95f10f04fb6434ff1a6fff07c1acd93deac1161fa5fd757ab5330dde818500ef.jpg)

![](images/c53568dab342a7df6de02582027fd6c9c569db96ff59146f783d5e98ff71d521.jpg)

![](images/8590e96a4c545e5b3e8a4b44aefee34de0843970dbefefbfa7211e6cf23342eb.jpg)

![](images/26480d488ecd5c64082ebaf524fc26b295cb4fafb40e04491ec5657a867353fa.jpg)

![](images/7f35efd3252c43786fa763490704ee2f7a00debdfd538015b564455870c855b6.jpg)

![](images/1fc9b8dd27459ad96e3bd395b352926aa71617bdb6097e4750ad114b6b7853a4.jpg)

![](images/70d3ef9c54d2af02449a987b3df6027c9f423c4d1a61f74474210803c762c426.jpg)  
Prompt: "a fantasy mouse walking down a trail to its burrow' Axis: “the size of the trail and the distance of the camera from the scene"

![](images/2e5ef05b11b837a6b76ed868cc439d75e1e79931cb96ad0150efbf39cb871eb7.jpg)

![](images/5cc10bcb5d6f684497f1b054dad4319fbc42659090ea8e4e8e3d255a28e5f793.jpg)

![](images/10abb01c46a1a2d5a43a3a359c9f90b50933505cdf670ff70316c1cafac84037.jpg)

![](images/108464ad42390e9a29c7c757ddd4a34522ef92bec5fbf046fecae8618cebcb0d.jpg)

![](images/54becc1263bd4f45c9a5dd4d186f62d1748e0714c030f9cce34dc17290e6d779.jpg)

![](images/ae3ad79bccfacb9abdff5e3c6b968bfd186079dce8299331068863aec2d73b09.jpg)

![](images/abf4af3986f84a9bad615232d12c654d152ee452326466942e61ff90be039083.jpg)

![](images/5c6523da2e970ff70dba9a16bd8cecd9f5e105dad120d94c81d512f7843ae629.jpg)

![](images/fad142526ac53c69010d1b485302b998c19c3641446a577b8888220d405acc3e.jpg)  
Prompt: "a dog plays with two cat siblings while a caged bird watches the playful activities, indoor setting" Axis: "only indoor settings but have different furniture and times of day"

![](images/8af20bc057776a55be64a6c1838a2ee6e0cf8e1c05b1d4d64089960cca6cf6d7.jpg)

![](images/7099cd1c362235e800ba0f61d5cff2cb0dac93241668fa45bb15365b907d4076.jpg)

![](images/efc035963c102d9c9fd8554678de788d5ec81032c7be30ff4cb1b4c0828c281e.jpg)

![](images/0f81b06b91e982016141295c828f5a7795a245542b167a7a20a1cfe510f0f76c.jpg)

![](images/50e6bb0fbae3f7128879ae20a1e247a18c7bbeec25d68d02cbb62cd853e1c053.jpg)

![](images/5798dc12c8696c4147ff28691c55f3016fdcefe67ac91496b44eea8df2000f6b.jpg)

![](images/3dcf7ae6031393f2a3a4b24c2f991cc7f52fcca46aca38ba5c7d4b4a00e06c9c.jpg)

![](images/09053646c6eb4e826a86f30abb08711688abbc8be0ecfb45df318501deac26ce.jpg)

![](images/500ff47a03ce4ff1a6e602314967ae6700779668f70d3fac00e94465f4426f59.jpg)

![](images/86ef109a368828ea215525a713173d1f93c670d882c9236feb3be68fabc309f8.jpg)

![](images/d41ee6cbd07645535fa9830ac758539d0095c26e1282f00ac7335dba6fb6ffac.jpg)

![](images/69dd8004dd662cdf00d2424590e12f1ce97cc5a694418ab5bec1a9bd47a14b5d.jpg)  
Figure 14: SD3.5 L (3/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

Prompt: "a spider with its babies living in a humans house" Axis: “different image styles"

T-DIG (α=0.5)

![](images/5b60822bd623ba395649b73795bb57d85c81ac791017bcfd4ee8f9d587050e0c.jpg)

![](images/a533bdd02eeff8001f805b45c03477bcfdd4c795b7c7d8a16407fadd67588aff.jpg)

![](images/4d92a1b833cce4a1c73fdf38b591b1be2b508982a1c51eab8e83fd03164d2853.jpg)  
Prompt: "a cute beetle car that has polka dots" Axis: “different settings"

![](images/10f8b55e57bd2e142be9072e797e9bb95cea260210ecd69be1272eed06dbafc0.jpg)

![](images/ba6d5c70b87efbfb15fd594acedc8a1de066182ae6805dfe209902b0f50eebea.jpg)

![](images/c70ca9fd5403512a230f352df0ea33fb3e43400faa11973335f1fa2297891680.jpg)

![](images/804c0e640644ef7450690dd85b2ab8959a1a7d5afc4ace12798c4654ef569bc7.jpg)

![](images/0ebb704cfc396366eb3ad52f95d3d543fe4f1d245d8dae188d8a085adb34a8fb.jpg)

![](images/b631c77cf4088217dbc0cb0cccec6f7d81b109548f08f99d6c31bc64d7bd67dc.jpg)

![](images/cc62c110f66da943468febc1f57a5b7c4aa82f9c0f6a02fe5fd947641f4cebe4.jpg)

![](images/f1633721de4470b02c1d44747a81dd7d27a291797e8709da2a32306012496da2.jpg)

![](images/1b010da02a786bb5d45feddf547e759f5ae580617d92dd8c02bdc0a8772662c8.jpg)  
Prompt: "Bottle of water with a tiny marmaid swimming inside it" Axis: "Face of a marmaid'

![](images/2d8eea5db0ed68586ca8f5c633c8cfabcdbc0b4b3a8bd24b27ac6640798df4ad.jpg)

![](images/44dcf8d7f9196d4e27e1d52a6bd1cce385a65753224681b2ef73758d12ee47c4.jpg)

![](images/e352e7f371b258fd4eb5dad27d6702b5152b848fd6e2f92b1f4eb05bbf2b803a.jpg)

![](images/e82fb171a4aab183876f6d97f42b45d3ed393c947e2eefdd7ffda70a47b626bd.jpg)

![](images/21dec60d0320d4d487f33dad3513fbea7ae9d74d8db8281c2eba341c0543dec3.jpg)

![](images/8d2019cff920d8bf9152c7f7d2f57a2f4949a1f09c4a4e176ec0a0d668a547a0.jpg)

![](images/b3fa2d0dadd8f5520bb9a5471166afef156da7d92780ba5e72fa3a2946c1ab75.jpg)

![](images/036e623650c0a6059f0215655644c40d85ca34f0eeaddf9f6ab9c1d5dca628f3.jpg)

![](images/4b4b4f0c9fd566b89ea7ed401ceb8e3744ce789408f0b5481b04ddccc8bddece.jpg)

![](images/97c75e4fcc682cd43f8b10a60681bfda5f8beaa981bf65cddb086c212126638f.jpg)

![](images/c3d174e1410d09e9c22cc433afbbbb03f57546e2f9bce7d377f440cf77ff5bb0.jpg)  
Prompt: "A sloth driving a tesla through a carwash" Axis: "vehicle color, sloth, carwash"

![](images/f05f86d2a63d40118ed7a2d8d7f57cc8a346e8f231cbbdd6133442d37a132c4f.jpg)

![](images/11bca25604d95e8dc870adfb70cf441dd189c5d8d1838fb289fec063c8729f18.jpg)

![](images/7c6d1e7b1a8a7ae3c8dd1167860495f27843068e061fdd1b9aee6c898b3ad800.jpg)

![](images/c3488b8ebe0777cab6e26f4674a48cb175551dcb2cd106b05a82c166c14904c9.jpg)

![](images/80f9f870949ea764466574bbeb010dcd7e2809a9ff53d98dac001872bd54e85f.jpg)

![](images/72177ad46c78f431f0b526b8ea546d1b0d514c7d26bed464e5305ce3669f4183.jpg)

![](images/b39a3b46f74059d84789105d0161ea800db38bd2113a762e850c021a30194459.jpg)

![](images/575919e74de3b41af5422580b93ea69d49ac08fb97b57e5415df838f281ea606.jpg)

![](images/a9388b2c8758643d7159facd9809e4372ce9ef521eec3a181bf0f6ea45bf1c0a.jpg)  
Prompt: "a couple flying through space holding hands. The female is Caucasian with long dark hair and a short fringe and the male is Caucasian with short ginger hair and a stubble beard and glasses" Axis: “"Different settings, some with planets and some with stars"

![](images/2238bd0a080259720a0bfc454421b13ef2d881f2c0d75ca5fa6b65e98f6bd5bd.jpg)

![](images/e67bb4d6141b31a0b8b4bfc4607c6a14f2f200b7f0dcdaffbab3ddb02fed06cf.jpg)

![](images/b36d5bd0276310f8460bc032d0503ae535f60fa64ee98f9575c773bce42e108e.jpg)  
Prompt: "a photorealistic black bear tending to his garden at a suburban home' Axis: "vary the type of garden in each image"

![](images/b5cf9612ed39b9479dcc40d32608384d509fc03cf8f21824a3a2c9cbd86b5d26.jpg)

![](images/33b3bb174d1886eb751909beb235a2bda5c79f7898a8c67fd37f1f61162aed7d.jpg)

![](images/424f012306512558d53b874ff3e842d0f8659a16862a418381949ea55b1aa094.jpg)

![](images/da58a8cf0525accd64371775ffe699c4dea94561621e1ad80d64b5794b544b8e.jpg)

![](images/22709f28d569358291ab8f267c3451b61e61aa390a3777c8bbadf0ef948b7907.jpg)

![](images/a38c171ee6e797e0af2c0093c995ac8ac833f2259eb5d8a7d5c76eeb46b08099.jpg)

![](images/721414465ddebcae1425f9a53497816f99f9bd1f6889dc1942ad1b33115301ea.jpg)

![](images/bd5acea4f5ebe5eebb4ec5649d4029849cfc5ef1cce756e0884f51a3d1156925.jpg)

![](images/e64772b9d88e6e932c838e914950034a7753388fdf921666f6b21e192568a697.jpg)

![](images/2baa25f16130c0d4992cb544bd5ec0f450c2f56834e74bba78553787f843a8de.jpg)

![](images/da6cfb270eb57878815b4888c94409fd97a88b7002e2f03bdb09df9223fc68df.jpg)

![](images/3dd0da321b0876499bffd18869db7ad773446eff619003f62fa02a32398703c9.jpg)

![](images/c58be830b59343c8a51ae127dcd6a72092de787905f4b88e955929cbd367f145.jpg)  
Figure 15: SD3.5 M (1/2): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

Prompt: "an animated shot of a monkey climbing a tree" Axis: "monkey characteristics, time of day, positioning, mood"

i.i.d. sampling  
![](images/16f07c5a80e6437ca4fd111cafb9345398a938e8ac3d7bebe9a10fe64d34be8e.jpg)

![](images/69decc3c35831f8438cc13f93a6d6871a15ed9ef94c7b9aff9d5e6569e8b0d60.jpg)

![](images/f101b0b8f2dfbd129406abe95427e482ffd258f9f2a8a945cd94c24c9638f44b.jpg)

![](images/98ce5fc36ddffecba7cb699b70d5ad9f6c5e3332b2c742f82ccad4d4359f0a2e.jpg)

![](images/1777f46444105aea07a632cf071409ddf636b1039b64bc8e1c3dd1cc07e0add7.jpg)  
T-DIG (α=0.5)

![](images/87e5dd4beb5ceb55bf0f3b95212980ca2640d8121e1342ae92dc3814ad2427ea.jpg)

![](images/d8fc91d0bfc566398e24f7753cf2b7b9aab519284f6bf0fda9e1bded8c1a773e.jpg)

![](images/7b164f6e4b161db3761c278a145e0f0bbf46c2b1686e19e2106bbdbc4ee1cf94.jpg)  
Prompt: "two Bernese mountain dogs sitting in a coffee shop enjoying a drink and a slice of cake" Axis: "Change the image types, illustrated, cartoon, photo realistic and water colour"

![](images/ab6ca8837b6c28694b109c42e3edb2a28877452880051406fde8269aaf431334.jpg)

![](images/59ec45ff15af0807f9c9255877b7778f58b4b57ede13224fcca2721de64e63c4.jpg)

![](images/dcf2fd7809cbeec5ab6f400b20a8425f95baa3df05f5fb8cfe9a6eca233ba1f4.jpg)

![](images/d4c4963e9ac7bef544ab15c4c6e80094c22c6ed0a8d58440961284fa988ebf44.jpg)

![](images/3a9cdb813fd150020e4310cb79d1458bdc4b8ff6d33a069515a5bc4d4ee1b247.jpg)  
Prompt: "A man in his twenties sitting by the window of a diner, lost in thought" Axis: “Time of dav, view outside"

![](images/3181bcde48919fcfe38a55271ad27094ad4f01b36e740037928abddc52d22c69.jpg)

![](images/23bc2471c4ff802e72cdde5b54482d0974ce34047a1fb5606800f9abf6a0656d.jpg)

![](images/0c0b3556dd4ced3ffe7b7f0ae5660b1a3855fb5a2f9d0bedfed8537071e2c58f.jpg)

![](images/32271f2879cd50e2f3d6caced35d05e9321f6cdebd2a2b162639807312a04d4e.jpg)

![](images/7e8a93a4333f52e6546e1b6fe49fd22170ec8c4d959cee901aa38f9cb79f1fd8.jpg)

![](images/bfbb56f216634f45db0a04e7ec1839bd888d328eba828d08afdd88798e5d56bd.jpg)

![](images/43c09d1d4fb5e6b97c66cfd4b94fe1d06788318f5b39b153561f140595d735b0.jpg)

![](images/d53d10a559d5bce294e9cbb9e4e5b7ef095639d6e7199b33d4571abcbd6e07a8.jpg)

![](images/da14131ab2640a93ac5b06c881b0274c8f775733b41b6ea9585d924c9e3b1f6d.jpg)

![](images/b690e633feff9219ae055f4071a43ecbd969c6d00bc699b27aee58e1ba0263d0.jpg)  
Prompt: "A person sitting on a porch swing with a cat on their lap" Axis: "Dress the person for different weather from very hot to very cold"

![](images/d040da646d6cbf5f92a0e91316907beeef0d3db17b15dcaf2aaa0f8b863ab570.jpg)

![](images/dec790139a17715a3c6a4878728bc063860f391d716877e0e200888cd69fd04b.jpg)

![](images/0d70d1d48df4d05edcbf3800018badb4249d3c85434dcb55e26802bcf3873c3b.jpg)

![](images/020b1c1273744d421d1a71e5aadfea507fd9ac4d6cb0844302ad75df39d1cd7c.jpg)

![](images/b1902d6930eb6a8b6a5c25901544bbac1b06ef3b4a23debe02bbbc6a1de3c4cb.jpg)

![](images/0ccba55bd58228ee0b2dcd988a96971f57ae838c5ad0b8fbfb2dd9158be0c547.jpg)

![](images/58745f5861a2420c804ce5627d5aa9d2ea5486067a2d815c13db2f44ce649619.jpg)

![](images/86fa1a28bb75d0c7119003f1621a156b3d1dc4e4cb3517a81c49e1419fe0d705.jpg)

![](images/1557c6a218ad57ee7d5b3bf9f7313bc0052aeec6035fa17518275e3ab1965507.jpg)  
Prompt: "Show between 4-6 people riding a rollercoaster, with facial expressions visible" Axis: "All the people in each image should experience the same feeling, on a scale from joy to terror, and they should be different between the images"

![](images/f3ada35a4402d6613a383b4441057a36bf95c978f6a58c50c396a9a95914f5e7.jpg)

![](images/46d67d04db4c7159acec5b6a7e26f7376e13266b41e53ceed9b0ba400cff86fd.jpg)

![](images/224283e9d790efcbdf0fdf5cb86a4e543ea9f6cf36efcd715cb2c93a9a02a0ee.jpg)

![](images/f292b9da30c2214b3a745e63a8d826bd2914b77d60b8a0af320fd83c6081ec0a.jpg)  
Prompt: "an empty park with a playground on a stormy day" Axis: "playground attributes, park environment, storm severity"

![](images/64ba9cfa3089f31e3e2617a40b6fc36dcf912b3e26dee90b600af9ee2fcd7a7c.jpg)

![](images/772bdc2b0525fc8ba3b2c5e5d5392d5a4d47bb0969d41b746afab56d7f029754.jpg)

![](images/480f02079fcade441d2a2e57bbc9f98b33cf9615500332273b267ce9d8344d63.jpg)

![](images/27ca1c76011af986917f72a7683e530d0c23b850383a97dc38d1a4f176753a10.jpg)

![](images/70e219e92cebbe75f150de7dba4176701a85ef3e90e3b7011ab0bc5c5c3b5234.jpg)

![](images/77a0a7c17ca013c58e8f5e24e2721138370d7acbc7d2381f997cc3a67c717c39.jpg)

![](images/d8ae03d606bc1036a6be5e334632c9b1b881b674c1cc4e59fdaaebcb75c98092.jpg)

![](images/22a2454c52c718cfcabc26073b199d26b157e0a2ec80bd9fd62be0b29df2d662.jpg)

![](images/da715d2398938406b6cad1e7bcb1b6c7c1e617848237ad8ecd6039ef794a5609.jpg)

![](images/2f8283ada35b15388f8359c9f44d1a30a0d5e6199de9304356be6e16dd1cbdf2.jpg)

![](images/4c1c490964ab66a33edd816759afc343a981c743379d7f65ae70a13ed1a136d5.jpg)

![](images/637d3df559f833518eaaaea5e10bba2ee3a56087a4bcd76efa142daf689310d6.jpg)  
Figure 16: SD3.5 M (2/2): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

Prompt: "a red bicycle leaning against a sunlit brick wall' Axis: "the surrounding environment'"

T-DIG (α=0.5)  
![](images/8eea8e4ce9f023a8976af1142254021429b506bf015b4bca5ee20494427c7fc4.jpg)

![](images/6b67bd7ada83f078f662e075e37b2436d5f41163cb852f7a227a1f6d51b6ea3f.jpg)

![](images/38ec07e8f3c7bcde9df62fffb57a44dcb03b1f6e4b1eaa59a6f03acf81ae13b8.jpg)

![](images/db7908c2c2f713dcebcd34b88207f18be167edcfe470451a3800c6b96f21e0c7.jpg)

![](images/c43521d078bff27c01cee3ce9227b969f66d5a489b21194db4dd8ed6a72d563c.jpg)  
Prompt: "a ruggedly handsome middle-aged man with a cigarette in his mouth, his head tilted slightly downward' Axis: "Hair-style'

![](images/8a1b9d8bf58d0521bf5f3227720caf43b988ad8143090d779e49a68bebb09be9.jpg)

![](images/0d3f492cf5c97c94931cabbd4fc563b6c01444245b0485b83ea57af1e62fa77d.jpg)

![](images/d2e0a0615b88339de483be9640b4f1f5fe5077b2cff16fab04f8913766bd0771.jpg)

![](images/6d9ced00c3d5e89b61c09ba2d7dc1f14098f23e573b10b884bec62de8c0ff3be.jpg)

![](images/33a726ff687be67870cab6140ced3e1817f17d28dee53d3048ec951ee8fcde68.jpg)

![](images/f12dd46e1474c976db16f89b54ff193602c89508a50e1abe2243453fe165c5ca.jpg)

![](images/1c7aa729e5da561713d49d6621371cc5c4283aeb8b0e3e7623f16b7de3854df7.jpg)

![](images/da90fada8b868e430b77d974cdce5a1faf3a7ae6c468b18b6d5f45b0bee7a484.jpg)  
Prompt: "A Labrador playing on a riverbank" Axis: “Colors of Labrador"

![](images/cc54f437244f829f129efff34a2c23e282a2a6cbda50e21f99b14fb77de4127b.jpg)

![](images/8cf3a3431595821ba3fc2bd0d508e2d215a98a1816d006f12458b2a15f5f8aeb.jpg)

![](images/aa14b0596998c2ea2c28e8316381891251b22fc4f1175418e7d99c9f6906a864.jpg)

![](images/95c510ecd6abc6279e74a89556401e26066152eac1237ad0d01fe39d17ed457d.jpg)

![](images/5dfc77707fd9f39e6c7edd65709bf773b2a5be12eb347cccb6cf3950558a3188.jpg)

![](images/02863d4aece8210fbefb1168844a0e0105d7822bcafc3c7ee71a4306745220c7.jpg)

![](images/ce330f70e5eecfe12e0083b5a3af69d4b86d6bda63cf68968e88566e1664e904.jpg)

![](images/b2eda1c0340e533d4b4481c3bb8dd3450c1ca1f1ab2a58bb2a812054c49089c5.jpg)

![](images/303d6c8b8c46cd406a09f7cfabb1d293387433a9187b61c8189d23bf32a5c3a8.jpg)  
Prompt: "a yoga class in progress." Axis: "setting of the class'

![](images/056154333c594bf19c205e5fa68a214d19c8678b9fd7ca676ee5dc10a9ea1d57.jpg)

![](images/88cfcb08e43c99b8fdc411ad24c1f9b34862c7a0c3aa73d57abe942d044cc095.jpg)  
Prompt: "A pug sleeping in a bed' Axis: "Bed type"

![](images/e345368ccc777ecc9b37d89e7d266079bdd92d613768baba63b5b900c777a7f6.jpg)

![](images/db60fee89e35976b1d38a683c307da15abbf119c1f4ba31d0e0b1313a2a18bca.jpg)

![](images/4aeb60c00bcb9e7bc4547b2256a01a22ebcb83d2b2c885cf10573444311051d6.jpg)

![](images/e8aabea37d0494a1386207a58613c0e9d78f301118b94586cc1bfddda19c99a0.jpg)

![](images/4a676a94c25ff3d8cdc778891d299c9f0fd5f1bd218bc1cd181d76aba3fd8223.jpg)

![](images/cd5d5d3887049ca6062f28ab6bf79b0a2ca024d71576541d3322e380068178ca.jpg)

![](images/b56d622fd538d4edd70a976b86064f980c0bd571510e79457c903d28841faebc.jpg)  
Prompt: "A flower with a bee on it" Axis: “Change the flower to a different color in each one"

![](images/e78f90e3ecb7e6cc8588c3b0423b37e361f661ad60b75be4e1b5dcce110a8802.jpg)

![](images/d8707a532c96a9dab0a095c8c0d7e126a34a6b7e176c715047eaa0af699ae21f.jpg)

![](images/eaac246b2b56d2b260a0874aebde9b858b4421e20b54c8886a652255214ed3e8.jpg)

![](images/f4e0a50e2ebb1a4858ff3092dcbd875c024a4b95c70450b8714101bba4b309a1.jpg)

![](images/c90770da851635609bb764a5575cf602f5ff2961269dc88cffe9985e93621b5a.jpg)

![](images/a04e45779fba1562afde43f298a84488f456e513134766eff92b3ff9d6cf3a7b.jpg)

![](images/73ac01c8d0caa2546edbe92bc47d841b390e2e9304d975eec0afb58dd61abb48.jpg)

![](images/4925cc77013a899e8d2c79ec11bcbda52eadf3ebfddc16ed982476f31731999a.jpg)

![](images/34f5b90a08f82af9e7b82aad8d2aebb28a9523549620f481f71d11aad3c2a857.jpg)

![](images/42fe9879eaa23864567f4fea5a87f3a82eaddbba737c8d3ec0ebd872020a2976.jpg)

![](images/5608e13f6c000b86ff7b275d6da9100813e4300921ac25493efff48c14adc516.jpg)  
Figure 17: FLUX.2 [klein] (1/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

## i.i.d. sampling

## T-DIG (α=0.5)

Prompt: "I want you to give an image of a girl and a robot in the middle of a flower field, seated on a picnic blanket, robot with a friendly look" Axis: "Give me a different time of day for each picture, from morning to midnight."

![](images/d9a81878d4bc97919473fe0bfbcc5ffdca9444824f54ab956630431e1704e29b.jpg)

![](images/16f4a2ac8ae77e11c1cb85a232855ee45f0948637c1674ac0a596fb166d8bb1d.jpg)  
Prompt: "A girl speaking on the phone" Axis: “Context, phone type"

![](images/2b3966ef88df5dba61d19f8c302f1110008838f8d58525741914d10ee21562fb.jpg)

![](images/9d25a3ba2947f61506a476310eaed1a96ccaee866e84802b4f9135e9e9c227e4.jpg)

![](images/942913324dfff838cec2f8e799744acb3b303cdbc909a26255519dcd69784660.jpg)

![](images/128139f733cde8a3a05df2406827880eb2e41b458e3d815205219190a6b83896.jpg)

![](images/e0206ba45fd7f5679323ea9970dad22cbaeb68de01d776b62f5aaa92a0597484.jpg)

![](images/a96b4a11a04d120eb4c843afb68e0cf0ac752a03b89a39e97a3e9b5575c23a69.jpg)

![](images/08becf7b3306d32c326bd106424c588184ea07c781bf6d95fb1065922a5ee8cb.jpg)

![](images/9508eca46a96ee4da541d20cc0c6698dda8f1326799b26cf13c6e28c2a958100.jpg)

![](images/47ec49b44d27980ee766167e5bcd79a934ba6156fab5f08358e4c02e26d82d2d.jpg)

![](images/8c0ce2669b86677b015fbbb075332b4cdb8ad9a13e596759aafeb2d97a1040a3.jpg)  
Prompt: "a baker decorating a square cake" Axis: “if vou could create different themed cakes that would be great"

![](images/5f3f3e3df44dd3561e284077a61499e56746627390b45d5b568ab101ee5e1e6d.jpg)

![](images/89964715e6bdec069f099035a9f39c5c09529012d792804632ef5a35e1002f6d.jpg)

![](images/f7d62f33433a14c186b3037311140b3e9e5b064b614663635daebb73e8d8f589.jpg)

![](images/875b38c2800dc3d6d9a088c3ac2433567e75c642ea4b155548ee9394c771c711.jpg)

![](images/4600e9d2ac8f0700d0d6b0782781abb5b4fda7a695724309634bc160a2b99daa.jpg)

![](images/4744c0981c327bb0ce54081846baede79373ad2557b8797647a5d0f0985b5dac.jpg)

![](images/13f96fa1443a5ee41abb83a0e52da1b344f7ca13f8bcde65800720568eb2ada4.jpg)

![](images/a6e05fffeec67ce4bac0a3185e2ac9af9d6dd9dc958bbdc6523b29364dbe13cd.jpg)

![](images/da364ba41df86fe44c06258a8017129961d6f196be37ab1d97803df2873cfc6e.jpg)

![](images/73d12db82bbd45d918a5794a5898304de221f2f11f11b94132cafee6df2b4e5f.jpg)

![](images/bb95ce3017258af94e747bf507b5c4c6e9bf2d9fec931f4cdac0013865557479.jpg)

![](images/6bd848f9d1bfb6ad98dde2b96c90c2cafc535433e61d8590e1e708cdadd89f45.jpg)  
Prompt: "Create a low angle rear view of a black porsche 911 speeding along a race track at dusk." Axis: "Create some alternate extreme weather options - rain, snow, lightning storm, etc. whilst keeping to the same time of day."

![](images/043f56b9a4b7a9619ce78767bda63c0e75d4c4fcfae1cab27a4e5aa964a91f11.jpg)

![](images/f9de14a93458801fb448625180dd30313016ae4c6be4ea88c9dc128862b80bdf.jpg)

![](images/b87193e83638be3c52788b01b22892450999ad6a62df008904e0abcc75aa1b21.jpg)

![](images/e6dd95e70a248e348f135f086d2e9eebfc009f58150dccf8792c5dd3323587f0.jpg)

![](images/44338511689026c4d785f6fdaf202f19a0bc3dba012dfa639ba1b94b0f2715a5.jpg)

![](images/0dc9c0de6adfde3f9748bbb1db3510631747e3ffe09d1b8afa39045f17313c81.jpg)

![](images/612822aa1bdce64fbfe99ef50134eff96c1e18383fc12b36969e9ee98f77062c.jpg)

![](images/e0a1dd5044fe0aba1cb4aa5e4d7a8ff97d202c7ec723ecd29bd18d84f8f0c52a.jpg)

Prompt: "a block of cheese with a face and hands singing to a blank pizza base that only has tomato sauce on it and nothing else that it is about to be put on tomato sauce"

Axis: "I want it in different setting such a kitchen, restaurant, a home"

![](images/2eb4d1d8a559c792c70615c5e2eb55a6962e91d002ec4d164f45d68f2b270d67.jpg)

![](images/6f42b41488398294537f7251a410eb56370d94aef9caaa6f91d24c3e3fe9b707.jpg)

![](images/17f02b8244094dfc14fefb275716ca1a1368982c6e3047122446117a50d27a12.jpg)

![](images/8808e6b40893694fcb118d4a2fb48b352fc5cfb12fb1de42e64f9c65ef0c393b.jpg)

![](images/89996e8cbd6552e05251ddab80e4b6673479cbad402ac548d8db1a8f4282f11a.jpg)

![](images/423a936d8539a41288aebe3eaf50f11771e6d7541e56ed25328c1366abb34b7f.jpg)

![](images/628eda824b1490a8cb347ed1d9abd675fc1195b4cabd8661bfa62582f434f40f.jpg)

![](images/6cc876aaa60258ade82767e52c675abcf9990c885606874e51f3870f6e474c68.jpg)  
Prompt: "Show me the inside of a computer, but make it pink. Little mouses are tinkering away, trying to repair something that's broken, pink interior, little mouses tinkering, broken component" Axis: “cartoon style"

![](images/b3f2593090d9079c9f9987375ec1a12027aed4f4f9ec8a64bcbe2bdf64c679da.jpg)

![](images/c21e1d63f7238f42124f46a1e613c6dd13a2b2d99cee44f47773926dd3cd9d4c.jpg)

![](images/a53d05587f404eb658c69f5cf1e3938d7a965f15d99cb3d0a40157fc43b6fd93.jpg)

![](images/2367084d4c2e7cb38e91038f719dd17070ce88eec7515aabcd22c7312bcf1510.jpg)

![](images/7799d1739cfd8782413af4b9051cae9210a74fe582b7a4b4ac0547caf20c0263.jpg)

![](images/3a94227e54837b63905006ea52d72a4dd576f80573d159954e06f2dd5861f7d9.jpg)

![](images/625332d2c5924602f85f1c79dd0605b05ce7336b7169e1a55ebdc64eaaf800fc.jpg)

![](images/a59ae3da354c74573fdab447ced743bdafece72f0660ff6de71feb6316e94545.jpg)

Figure 18: FLUX.2 [klein] (2/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.

![](images/38790e99fb7fee00ba3b84031de019f8d3a3eea8c463f1daebaba6b3af7ec61c.jpg)

## i.i.d. sampling

Prompt: "baseball game from the point of view of someone sitting behind home plate, drinking a soda" Axis: "Change the sky to different times of days"

![](images/349695ac0c7a5cc48aa3e51957d9b4eea203d689a2818045b57230d83a428e2a.jpg)

![](images/01d656247f65a91492a3f03923bd67968bdd83f4eb8bbd2143da76b0cd2705e9.jpg)

![](images/7feb431021684a93ae17f39fd69b052edc427d6763621e1ce2e67a3be013f973.jpg)

![](images/3378da9895f96e42ad6507155b5b0da1ec003492d115e6614a39e6e1592ecdcf.jpg)

![](images/cb5439e2fb6f0aa60741894ece24b1eb199660191dd58f0c5f5a8342a25c0e54.jpg)

![](images/634f08fb4cc195fde6c52f82e98e38c6a1af0f592105f93ecf1c1ac4e79a7305.jpg)  
Prompt: "a steaming bowl of ramen on a small cafe counter" Axis: "who is around the cafe counter"

![](images/4e66c2a2e270935cd75709e08e02b5187048bcdb85cbfe6972f1c9afd8950524.jpg)

![](images/1ab94408c7e9e1f469de0245fd3a6fe18e274f1ff51b9b0fd1f73f6b53a8c0b2.jpg)

![](images/119fb6bb3d28a16dbe594f5d111ab0b482d131fa60445f2e7afeb6dd33a60228.jpg)  
Prompt: "a musician playing a telecaster.' Axis: “gender of the person"

![](images/c06e037866e357d05bab8b9c706785690775afce3d2077c6786c96df9ccec894.jpg)

![](images/daecd6bf5d4019f7f686d5ae4b4fa42847d2077836e08356e0709461055ad5a5.jpg)

![](images/2ffd35593f54cfaf787c387160297305eec8be968abb53fda52ace729ca38e25.jpg)  
Prompt: "A of lioness in zoo sleeping' Axis: "change lighting to evening"

![](images/b393e0464524b870bc78fa7294f4eb195801b5b3913bce9f52c495b048badb61.jpg)

![](images/bbc4809d91ae81d195f3c28745a6ddd3dfecdb9b391b59e29153dca86ecc625a.jpg)  
Prompt: "A pizza on a single bed."

## T-DIG (α=0.5)

![](images/d51afef114ab58b7e99a7d3995b553b30ef82f5143944814f675b250b27861ec.jpg)

![](images/9b201cd9cb9beff6a9af4c1995dc72413ed9b4ddc11420d46a9ac90b52f974e2.jpg)

![](images/6463c4c78a7d405951186911b6fb6c2febe2c7bdbe06003c9a6d06e47ec11457.jpg)  
Axis: "Make each pizza have varying toppings, from meat to vegetarian to vegan."

![](images/5360ef05da0faac3ce8f0213ca1400639c7712ebb453b3f3aa7b3f4688c2e316.jpg)

![](images/69d9b6fc5404a2ac729c0345de101e9ab47723bb16966e6440b94e5f3bc240f3.jpg)

![](images/c627bddbe3c31cb40a538ae50396beb09ef748f4a7fb0a43c806013b8e3ff4a9.jpg)

![](images/65e35c89a4f12154f4a19e6c27da0796571ef0c9ef1f3b986fe1d69ce27bafee.jpg)  
Prompt: "Two Cavalier King Charles spaniels, one blenheim and one all light brown/ginger sitting next to each other" Axis: "Body posture variation"

![](images/5433ffeb660ad701cba309a144ce08a93e40be380c463aad445a65adeb38c0fc.jpg)

![](images/80c5a5d20ceed50f101d07fe4d08fe05b7dac7c7412474d7056c2b06f2bf24ec.jpg)

![](images/0e191e98e0434f50d47975b14ff3fbab31128c1a585fc0e9c68e29e6bfb84960.jpg)

![](images/32f77b984a71eaa22a4c89b402a8ebdf261a8c9bde2c284f677e2c45aead1f18.jpg)

![](images/ad31049c4c17bb35fa1df9c2d08371c14d0b6f98c3c82bfdeeb6aec71daed898.jpg)

![](images/d71e00325a8f88b3afdc4d0386b526246f08de9854b6d58d60c2e622fb88b66c.jpg)

![](images/7f875dd56ea21867a59abd0b692b346559d0befe14e641e28e8a88e0da33a0fe.jpg)

![](images/d942f0379da296c035f35c1e8efc528f4c86698c2c822886d8feaf9841a00b0a.jpg)

![](images/038d0bab52631d84e859ec52eb3284d4ae2533e68db8e0979c898d8b9d370586.jpg)

![](images/cd65171d5b69f3b5062ce23ef1b2b0c7931eb76f72cab6fee5dc8812271974a5.jpg)

![](images/ebedaa2fd539b43d6e91291e73f534229b1fc621f14ec90984528565cfe5d593.jpg)

![](images/6f646fce51a563f42bc0dfac05e99eb11133e37c1d19370bfa56f19180fb90d4.jpg)

![](images/e4def8468e43371e833c7ccb50d41892d37c2d25d0a7d3d6eff4245946b5b1ac.jpg)

![](images/78a1ffd84c7f9898d34ed4c7eb0bad607fb1f0d8ce9b16ee3aafbc88adc6918d.jpg)

![](images/9c56fc76c2d6c1786d6642a97b62cb0af929b542c44e63de2012ec1730054177.jpg)

![](images/154ae83d58003293bb06f997cc9ea6d18f6b9e44f6852aed758c06edeb939174.jpg)

![](images/74da273d4403f7314462c121dae486f4abbfcebc4bce8a2fdd7cae3e53bcfc85.jpg)

![](images/63b1cbdb886bfb8dfa12e33dab5ff92e83bfde80edfe673994975cda22e470ce.jpg)

![](images/2da8f4ed162cde7162561b4cec576a4f8258a053fd7eb9695fdd0ea2abb5fe36.jpg)  
Figure 19: FLUX.2 [klein] (3/3): i.i.d. sampling (left) vs. T-DIG (α=0.5, right) on requests from the interactive test set. Prompts and axes are the study participants’ own text, verbatim.