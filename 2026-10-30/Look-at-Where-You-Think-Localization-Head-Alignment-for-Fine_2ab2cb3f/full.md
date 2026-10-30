# Look at Where You Think: Localization-Head Alignment for Fine-Grained Perception in Thinking VLMs

Yuan Wang<sup>1</sup> Wei Wu<sup>2</sup> Hongning Wang<sup>1,\*</sup>

<sup>1</sup>Department of Computer Science and Technology, Tsinghua University

<sup>2</sup>Ant Group

wang-yua22@mails.tsinghua.edu.cn wuwei19850318@gmail.com hw-ai@tsinghua.edu.cn

\*Corresponding author.

## Abstract

Thinking vision-language models reason for thousands of tokens, yet still miss small details in high-resolution images. Zooming methods pay for extra inference passes, and outcome rewards never tell the model where to look at. In Qwen3.5-4B, we find that the bottleneck is selection, not seeing: a few attention heads, which we call localization heads, pick the evidence out of thousands of visual tokens; ablating four of them costs 13.2 points more than ablating four random heads, and their focus fades as the reasoning grows. We therefore propose Localization-Head Alignment (LHA), which makes attention a training signal: whenever the reasoning states a visual fact, these heads learn to look at the focal region rather than the background, as an auxiliary loss in both SFT and RL training. On benchmarks of V\*, HR-Bench 4K and TreeBench, LHA lifts Qwen3.5-4B from 74.1% to 76.6% accu racy, the best single-pass 4B result we evaluate, ahead of multi-pass zooming and released 7B models. Alignment alone adds 1.8 points over otherwise identical training while shortening reasoning by 20%, and the gains carry over to Qwen2.5-VL-3B.

## 1 Introduction

Thinking vision-language models (VLMs) reason for thousands of tokens before they answer (Qwen Team, 2026; Huang et al., 2026b), yet thinking longer does not make them see better: they still fail on fine-grained perception in high-resolution images (Wu and Xie, 2024; Wang et al., 2025b, 2026a). For instance, ask such a model what color a kite is in a 4K photograph. The image arrives as about 8,100 visual tokens, of which the kite covers only a handful, and by the time the reasoning gets to the kite, language priors (Liu et al., 2025a), unrelated image regions and the model’s own growing text all compete for its attention. A model that looked at the right place early can drift away just when it needs to read the evidence.

![](images/6c636dc9edfae76fb76bdb6ec6dd894396659cb0bcee76d75aeae81153df1dde.jpg)  
Figure 1: Supervising where a few heads look gives the most accurate single-pass Qwen3.5-4B model, with shorter reasoning. Overall accuracy on V\*, HR-Bench 4K and TreeBench against mean generated tokens. Hollow markers use extra passes or crops. The arrow adds the alignment loss to otherwise identical SFT and RL; the inset breaks down LHA’s gain over the base model by benchmark.

Following the evidence from the pixels to the answer, we find that the bottleneck is selection, not seeing (§4). The vision encoder preserves small evidence; it is the thousands of surrounding tokens that bury it. A few localization heads handle the selection: ablating just four of them in Qwen3.5-4B costs 13.2 points more than ablating four random heads. They perceive the image at the tokens where the reasoning states a visual fact, not at the answer, and their preference for the evidence fades as the reasoning chain grows. On V\* for Qwen3.5-4B and in all 12 model–dataset settings we test across six direct-answer VLMs (Appendix C.5), correct answers are from the reasoning where more attention is paid to the evidence than incorrect ones.

Yet no training signal speaks to these heads directly. Cross-entropy imitates every output token alike, and outcome-reward RL such as GRPO (Shao et al., 2024) credits a whole trajectory by its final answer; neither tells the heads where to look at while the model reasons. Prior attention supervision targets other units: EASE fits a head-averaged map of one layer (Hu et al., 2026), Faithful-MR1 an inserted focus token (Tian et al., 2026) and AVAR the total visual attention (Luo et al., 2026); zooming methods add inference passes instead (Zhang et al., 2025a; Shen et al., 2025; Zheng et al., 2026).

Our idea is to make attention itself the training target, placed exactly where the diagnosis says perception happens. In particular, Localization-Head Alignment (LHA; Figure 3) supervises the localization heads, found from held-out grounding boxes and shown by ablation to matter, and it supervises them at the tokens where the reasoning states a fact about an annotated object. For example, when the model writes “the kite is blue”, we train the localization heads at the token that predicts “kite” to attend to the kite’s box. The loss contrasts the heads’ attention on that evidence with an equally large background; the softmax normalizer cancels in the ratio, so it teaches where the heads look, not how much they look at the image. One loss serves both stages, as an auxiliary objective on the model’s own correct trajectories in SFT and on correct rollouts during GRPO, and inference is unchanged.

LHA is the most accurate single-pass Qwen3.5- 4B model we evaluate, at 76.6% (Figure 1): it beats every inference-time intervention, including ViCrop with two extra passes per question, the attention-supervised RL method EASE by 3.0 points, and released perception models up to 7B. Both stages contribute. On the same self-generated trajectories, alignment SFT adds 1.4 points over cross-entropy SFT, and the gap grows to 1.8 after identical outcome RL (95% CI [0.5, 2.9]), most on questions that relate several objects. In RL, the attention term adds 0.9 points over the same RL without it, because, as a gradient decomposition shows, outcome rewards decide which trajectory to reinforce while the attention term consistently moves the heads toward the evidence. LHA is also cheap: SFT updates 2.6% of the language model, aligned policies write 19% shorter rollouts so that each RL step runs 15% faster, and the final model reasons with 24% fewer tokens than the base model. The principle carries over to direct-answer Qwen2.5-VL-3B (Bai et al., 2025) (2.5 points).

Our contributions are:

• Diagnosis. In a thinking VLM the evidence survives encoding and is selected by a few causally important localization heads, which read the image where visual facts are stated and lose it as the chain grows (§4).

• Method. LHA turns attention into an auxiliary objective for SFT and RL that, unlike prior attention supervision, acts on causally important heads at the fact tokens inside the reasoning, at no inference cost (§5).

• Evidence. The most accurate single-pass Qwen3.5-4B model in our comparison, matched gains in accuracy on two backbones with shorter reasoning and faster RL training, and gradient and routing analyses showing that the attention term supplies the routing signal outcome rewards lack (§6, §7).

## 2 Related Work

Thinking VLMs drift toward language priors as their chains grow (Liu et al., 2025a; Shi et al., 2026). Inference-time methods recover small details with guided search, attention-selected crops or learned crop tools, at the cost of extra passes (Wu and Xie, 2024; Shen et al., 2025; Zhang et al., 2025a; Zheng et al., 2026). Perception RL rewards correct answers, grounded boxes or stated perception (Huang et al., 2026b; Wang et al., 2026a,b), and these signals act on the policy’s outputs. The closest methods supervise attention during RL. EASE matches a head-averaged map of one layer to a smoothed evidence prior (Hu et al., 2026). Faithful-MR1 supervises an inserted focus token on a prescribed set of layer–head pairs before a counterfactual attention reward (Tian et al., 2026). AVAR rewards visual attention after an attentionguided cold start (Luo et al., 2026).

Functional heads that retrieve context (Wu et al., 2025) or ground image regions (Kang et al., 2025b) have mainly been used for analysis and for trainingfree grounding or cropping (Zhang et al., 2025a).

Relative to EASE’s layer-averaged maps, Faithful-MR1’s inserted focus token and AVAR’s visual-attention reward, LHA supervises independently selected localization heads at the native queries that predict visual facts. Its contrast between the evidence and an equally large background cancels the attention softmax normalizer and is neutral under uniform attention, giving a direct regional-preference objective that alignment SFT and attention RL share without changing inference. Appendix A discusses each line of work in detail.

## 3 Setup

Model. We study Qwen3.5-4B (Qwen Team, 2026) in thinking mode. Its language model interleaves Gated DeltaNet linear-attention layers with eight full-attention layers $( 3 , 7 , \ldots , 3 1 )$ of 16 query heads each. Because linear-attention layers compress the context into a fixed-size state, we study the 128 full-attention heads, which attend to every visual token directly.

Benchmarks and protocol. We evaluate on 1,396 multiple-choice questions about small evidence in high-resolution images: V\* (191; Wu and Xie, 2024), HR-Bench 4K (800, 200 questions in four option orders; Wang et al., 2025b) and TreeBench (405; Wang et al., 2026a). Qwen3.5-4B models are decoded in thinking mode at temperature 1 with a 32K-token budget. Released models from other backbones use their own generation format. Rules extract the answers, and an LLM judge (Qwen3.6-35B-A3B) reads only the responses whose committed option the rules cannot parse. A response is budget-truncated if it reaches the 32K budget without finishing. The protocol and the base-model reference were frozen before the experiments, and every model in Table 1 is scored by the same rules and judge. We checked the scoring of all reported evaluations by hand, including every judge decision, and it agreed with the manual labels throughout (Appendix B). RLtrained Qwen3.5-4B models are reported as the mean of two independent decodes where both are available (single-decode rows are marked); paired bootstrap intervals for all controlled comparisons are collected in Appendix B.

## 4 Where Thinking VLMs Select the Evidence

Where does perception break in a thinking VLM? We follow the evidence from the pixels to the answer. Evidence masking, post-encoder token controls and head interventions point to a concrete supervision target, namely a few localization heads at the positions that predict visual facts.

## 4.1 The answer hinges on small evidence

Masking the ground-truth region of a question costs Qwen3.5-4B 22.1 points more than masking an equal-area region elsewhere (Figure 2a), and

the smaller the evidence, the larger the cost (Appendix C.1). Fine-grained perception is therefore a search for a few relevant tokens among thousands.

## 4.2 The evidence survives encoding but is buried

Is small evidence lost in the vision encoder or among the visual tokens that follow it? We edit the visual tokens after encoding, so the encoder still sees the whole image. Given only the tokens inside the evidence box, a small subset of what it already receives, Qwen3.5-4B answers $\mathbf { V } ^ { * }$ and TreeBench questions as accurately as from the full image (60.7% vs. 60.1%). Given an equal-area set of other tokens instead, it falls to 38.0%. Appending the box tokens to the full image adds almost nothing, and on synthetic small-object scenes the box tokens alone even raise Qwen2.5-VL-3B accuracy by 16.8 points (Appendix C.2). The encoder preserves the evidence, and more of it does not help; what the model must do is pick it out from the tokens around it.

## 4.3 A few localization heads select the evidence

Following retrieval-head scoring (Wu et al., 2025), we rank the 128 full-attention heads of Qwen3.5- 4B by how strongly they attend to the evidence box. The ranking uses held-out grounding questions, never correctness; computed independently on $\mathbf { V } ^ { * }$ , it has Spearman correlation 0.80 with this ranking and shares six of the top eight heads (Appendix C.3). Zeroing only the top four heads costs 13.2 points more than zeroing four random heads of the same layers, with large drops on every benchmark (Figure 2b). Such heads are known from short-answer grounding (Kang et al., 2025b; Gandikota and Bau, 2026; Zhang et al., 2025a). In a thinking model, they matter for the answer itself, not only for the reasoning that precedes it (Appendix C.4): in a forced-choice probe without any reasoning, ablating eight probe-selected heads lowers accuracy from 52.6% to 36.8%, while layermatched random heads leave it at 52.6%.

## 4.4 Evidence routing predicts correctness and fades during reasoning

On $\mathbf { V } ^ { * }$ , where a single small object decides the answer, the routing of these heads at the end of the question already separates correct from incorrect answers (Figure 2d). TreeBench questions relate several objects that the reasoning visits one by one, so their evidence is read inside the chain.

(a)  
(b)  
![](images/b5e76f573e0c8ae6a9f44f2b51c37316d2b6c4b948494ce6aa91070b671c8733.jpg)

![](images/b88adc26ee4c19e064034de703820d3a2d2a1f69bd2c22b1b748b89890c5944c.jpg)  
(c)

![](images/62872356b5be5ec5b0153a4704f7e0019ff26dde34e9a48968aed4bba7fefc81.jpg)

(d)  
![](images/5db4bc038f4d3223f0cf5cf2cfbcf2a685fa6c34af66675ab2bb9df40d3f2653.jpg)  
Figure 2: Thinking VLMs need small evidence and rely on a few localization heads to select it (Qwen3.5-4B). (a) Masking the evidence hurts far more than masking an equal-area region elsewhere. (b) Ablating four localization heads hurts far more than ablating four random heads of the same layers. (c) The heads’ share of visual attention inside the evidence box declines along the chain of thought. (d) On $\mathrm { V } ^ { \ast }$ , the heads concentrate on the evidence at the end of the question far more often when the answer is correct. Error bars: 95% bootstrap CIs.

Inside the reasoning, the heads read the image at the tokens that state visual facts, where they clearly prefer the stated object over an equal-area background (Appendix C.5). This reading weakens as the chain grows: the heads’ share of visual attention on the evidence falls by a quarter from the beginning of the chain to its end (Figure 2c). The same decline appears on $\mathbf { V } ^ { * }$ and TreeBench rollouts, where the heads’ preference for the evidence over an equal-size background drops steadily from the first tenth of the reasoning to the last (Appendix C.5). Restoring only the amount of attention that reaches the image does not help $( \mathsf { A p - }$ pendix C.6). The heads need to look at the right place, not more at the image.

Takeaway. The evidence is in the input, and the model has heads that can find it. They attend to it at fact tokens rather than at the answer, and less sharply as the reasoning chain grows. An outcome reward sees only the final answer, so LHA places regional supervision where visual facts are generated.

## 5 Localization-Head Alignment

The diagnosis tells us to supervise a few localization heads at the positions where visual facts are read. LHA turns this into one routing loss used in two phases (Figure 3): alignment SFT on the model’s own correct trajectories, then attention RL, i.e., outcome-reward RL that keeps the same loss on correct rollouts. We describe it for the thinking Qwen3.5-4B; an adapted version for the directanswer Qwen2.5-VL-3B supervises the answer position (Appendix E.4).

## 5.1 Routing loss on localization heads

For an image x and question q, let $y = ( y _ { 1 } , \dots , y _ { T } )$ be a trajectory of the model and V the positions of the visual tokens in the input. For head h with attention $a _ { h , i }$ at query position i and a set of visual tokens $R \subseteq \nu$ , let

$$
p _ { h , i } ( R ) = \frac { \sum _ { v \in R } a _ { h , i } ( v ) } { \sum _ { v \in \mathcal { V } } a _ { h , i } ( v ) }\tag{1}
$$

be the share of the head’s visual attention that falls in R. Normalizing within the image discards attention on text and sink tokens (Xiao et al., 2024; Kang et al., 2025a), which grows with the chain (§4.4).

Heads. H contains the top-scoring localization heads of §4.3. For Qwen3.5-4B these are the ten top heads, which lie in layers 15, 19 and 23 and include the four ablated heads.

Positions. A thinking model reads the image at fact tokens, such as a stated count or a word of an annotated object phrase. There the supervised heads prefer the evidence over an equal-size background, $p ( G ) / ( p ( G ) + p ( B ) ) \approx 0 . 7 5$ , whereas at the final answer token the preference is at chance (0.50), because the answer follows from facts already stated (Appendix C.5). We therefore supervise the facts, not the answer; a direct-answer model reads the image at the answer and is supervised there. For each supervised token $y _ { t }$ we take the query at position t−1, whose output predicts y<sub>t</sub>, and collect these positions in $\mathcal { F } ( y )$ . Each $i \in \mathcal { F } ( y )$

![](images/650dd31fbf33bf8935f2c7f61f778a3ff8e5051a977ac1ab49b5c1bae812fbff.jpg)  
Figure 3: LHA on a TallyQA training question. Top: in the model’s own correct reasoning, each fact phrase names an annotated object (same color in the image). At the query t−1 that predicts a fact token, the routing loss raises the attention of the localization heads (red cells in the LLM) on the object’s visual tokens G against an equally large background $B ;$ the answer token, where the preference is already at chance, is skipped (.75 vs .50 on average). Bottom: the heads are the top ten of 128, ranked on grounding questions; ablating the top four costs 13.2 points more than ablating four random heads. The same loss joins cross-entropy in alignment SFT (flame: trained; snowflake: frozen) and GRPO on correct rollouts in attention RL, where it lifts the heads’ preference σ(s) from .76 to .86 in 50 steps (Table 3b). Inference is unchanged.

receives an evidence set $G _ { i } \subseteq \mathcal { V }$ , the visual tokens covered by the annotated boxes the token refers to, and a background $B _ { i }$ of the same size drawn from the visual tokens outside every box.

Loss. We score how strongly head h prefers the evidence at position i by the log-odds

$$
s _ { h , i } = \log \frac { p _ { h , i } ( G _ { i } ) } { p _ { h , i } ( B _ { i } ) } ,\tag{2}
$$

and penalize it with a logistic loss, averaged over a set of positions $\mathcal { P }$

$$
\mathcal { L } _ { \mathrm { a l i g n } } ( \boldsymbol y ; \mathcal { P } ) = - \frac { 1 } { | \mathcal { P } | | \mathcal { H } | } \sum _ { i \in \mathcal { P } , h \in \mathcal { H } } \log \sigma ( \boldsymbol s _ { h , i } ) .\tag{3}
$$

Since $\sigma ( s _ { h , i } ) = p _ { h , i } ( G _ { i } ) / ( p _ { h , i } ( G _ { i } ) + p _ { h , i } ( B _ { i } ) ) ,$ minimizing the loss moves attention from the background to the evidence. Two properties make the loss target where the head looks. The softmax normalizer, and with it the denominator of Eq. (1), cancels in $s _ { h , i } .$ , so the loss depends only on the attention logits of evidence and background tokens; attention to text, sinks and the rest of the image does not enter it. Moreover, because $B _ { i }$ is as large as $G _ { i }$ , uniform attention scores a neutral ${ \frac { 1 } { 2 } } ,$ , so the loss measures per-token preference, not box size.

From diagnosis to design. Each choice comes from a diagnostic finding. The heads are found from grounding boxes alone, without looking at correctness, and the model causally relies on them (§4.3). The positions are where these heads read the image; the answer token, where their preference is at chance, is left alone. Box-matched fact tokens also concentrate the signal: only 13.7% of 64 positions sampled uniformly from a trajectory fall on fact content (Appendix E). The contrast stays inside the image because answers use context around the box and because restoring visual mass alone does not help (§4.4): the loss should move attention within the image, not add more.

## 5.2 Phase 1: alignment SFT

The first phase trains on correct trajectories y with

$$
\mathcal { L } _ { \mathrm { S F T } } = \mathcal { L } _ { \mathrm { C E } } ( y ) + \alpha \mathcal { L } _ { \mathrm { a l i g n } } \big ( y ; \mathcal { F } ( y ) \big ) ,\tag{4}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is the token-averaged cross-entropy of the whole trajectory. The trajectories are the base model’s own correct best-of-five samples on boxannotated counting and spatial questions from TallyQA (Acharya et al., 2019), PixMo-Count (Deitke et al., 2024), Spatial-MQA (Liu et al., 2025b) and VSR (Liu et al., 2023), so no external teacher is involved. We train only the attention projections of the supervised layers (2.6% of the language-model parameters), and the alignment gradient reaches only the query projections of the supervised heads (Appendix E). Because the raw gradients differ by orders of magnitude, α is set at every update from

the batch gradients:

$$
\alpha = \rho f \frac { \| g _ { \mathrm { C E } } ^ { Q K } \| _ { 2 } } { \| g _ { \mathrm { a l i g n } } \| _ { 2 } } ,\tag{5}
$$

where $g _ { \mathrm { C E } } ^ { Q K }$ is the cross-entropy gradient on the query and key projections, f the fraction of trajectories in the batch with $\mathcal { F } ( y ) \neq \emptyset$ , and $\rho$ the dose. The update uses $g _ { \mathrm { C E } } + \alpha g _ { \mathrm { a l i g n } } ,$ clipped to norm 1.

## 5.3 Phase 2: attention-supervised RL

The second phase starts from the aligned model and adds the same loss to outcome-reward RL. For each question it samples a group of K rollouts $\{ y ^ { k } \}$ with rewards $r ^ { k } \in \{ 0 , 1 \}$ (1 for the correct committed option) and minimizes

$$
\mathcal { L } _ { \mathrm { R L } } = \mathcal { L } _ { \mathrm { G R P O } } + \frac { \beta } { | \mathcal { D } | } \sum _ { y \in \mathcal { Y } ^ { + } } \mathcal { L } _ { \mathrm { a l i g n } } \big ( y ; \mathcal { F } ( y ) \big ) ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { G R P O } }$ is the clipped group-relative policy loss with a KL penalty to the starting model (Shao et al., 2024) and $y ^ { + }$ the correct rollouts among the rollouts $\mathcal { V }$ of an optimizer mini-batch. The reward decides which trajectories to reinforce; the attention term decides where the localization heads look inside them. Only correct rollouts are supervised, because their reasoning reached the annotated answer. Their fact tokens are the object nouns of the annotated phrases and the count tokens that enumerate instances (“person $2 ^ { \circ } )$ in the thinking span. Unlike in SFT, where each fact token has its own $G _ { i } .$ , all fact queries of a rollout share one evidence set G, the union of the annotated boxes the rollout refers $^ { \mathrm { t o , } }$ and the gradient flows through all parameters (Appendix E).

Configuration. LHA uses $\rho { = } 0 . 1$ in alignment SFT and $\beta { = } 0 . 0 1$ in attention RL, which trains on questions that the base model answers correctly in some but not all samples (Appendix E.2).

## 6 Experiments

## 6.1 Baselines and controls

Released perception models, evaluated under our protocol on their own backbones: ZwZ-4B (Wei et al., 2026) on Qwen3-VL-4B, and DeepEyes-7B (Zheng et al., 2026) and TreeVGR-7B (Wang et al., 2026a) on Qwen2.5-VL-7B (Bai et al., 2025) (NoisyRollout-7B (Liu et al., 2025c) in Appendix G). Inference-time interventions on Qwen3.5-4B, which buy accuracy with extra inference passes (Appendix F.1): ViCrop (Zhang et al., 2025a) (relative-attention crop added as a second image), VCD (Leng et al., 2024) (contrastive decoding with a distorted image), Zoom-Eye (Shen et al., 2025) (confidence-guided tree search over sub-images) and a fixed-grid crop. Attention-supervised RL: EASE (Hu et al., 2026), which adds a KL term toward a Gaussian evidence prior on a head-averaged attention map during DAPO (Yu et al., 2025b). Outcome RL from the base model with GRPO (Shao et al., 2024) or DAPO stays within decoding noise of the base model (Appendix G).

Our training rows. CE SFT and alignment SFT share trajectories, data order, updates and trainable parameters and differ only in the alignment loss (α); each is then followed by the same outcome RL. LHA continues alignment SFT with attention RL, and its $\beta { = } 0$ control runs the same RL from the same checkpoint without the attention term (Appendix E.2; paired intervals in Table 4).

## 6.2 Attention supervision makes perception more accurate

LHA is the most accurate single-pass Qwen3.5-4B model in Table 1, at 76.6% (2.5 points above the base model with 24% fewer tokens), and it outperforms every inference-time intervention, including ViCrop, which runs two extra attention passes and a crop on every question.

Alignment makes SFT more accurate, and the gain survives RL. Alignment SFT alone reaches 76.1%, 1.4 points above CE SFT on the same trajectories and on par with the best inference-time intervention at a single pass. After identical outcome RL the gap grows to 1.8 points (95% CI [0.5, 2.9]), and the aligned model uses 20% fewer completion tokens. The gain holds under greedy decoding (2.1 points, 95% CI [0.4, 4.0]; Appendix F.2), and it is not a by-product of fewer truncated responses: on the 2,713 question–decode pairs where both models finish, the aligned pipeline answers 43 more questions correctly, and these pairs account for 86.8% of its token saving (Appendix F.4). It is largest on questions that combine several pieces of evidence: 4.0 points on the spatial questions of $\mathrm { V } ^ { \ast }$ 3.5 on the cross-instance questions of HR-Bench and 3.4 on the reasoning questions of TreeBench. Aligning just 44 fact tokens thus leaves a mark that outlasts a full RL stage without the loss.

Attention RL adds a direct routing signal. With the same RL, the outcome reward alone leaves the aligned model where SFT put it (75.7% for the β=0 control), whereas keeping the loss lifts it to 76.6%: 0.9 points above the control in each of the two decodes, with 1.8 on $\mathrm { V } ^ { \ast }$ and 1.6 on TreeBench. Meanwhile, the attention term sharpens the heads evidence preference, while the control’s stays flat (§7). LHA’s largest gain over the base model is 5.7 points on TreeBench reasoning.

<table><tr><td rowspan="2">Method</td><td colspan="3"> $\mathbf { V } ^ { * }$ </td><td colspan="3">HR-Bench 4K</td><td colspan="3">TreeBench</td><td rowspan="2">Overall</td><td rowspan="2">Tokens</td></tr><tr><td>Attr.</td><td>Spatial</td><td>All</td><td>FSP</td><td>FCP</td><td>All</td><td>Perc.</td><td>Reas.</td><td>All</td></tr><tr><td colspan="10">Released perception models (own backbones)</td><td></td><td></td><td></td></tr><tr><td> $\boldsymbol { Z } \mathrm { w } \boldsymbol { Z } \mathrm { - } 4 \mathbf { B } ^ { \dagger }$ </td><td>90.4</td><td>88.2</td><td>89.5</td><td>90.0</td><td>72.2</td><td>81.1</td><td>63.8</td><td>39.5</td><td>48.4</td><td>72.8</td><td>185</td></tr><tr><td>DeepEyes-7B (no tools)†</td><td>73.0</td><td>68.4</td><td>71.2</td><td>85.8</td><td>61.0</td><td>73.4</td><td>57.0</td><td>28.9</td><td>39.3</td><td>63.2</td><td>45</td></tr><tr><td>TreeVGR-7B†</td><td>82.6</td><td>84.2</td><td>83.2</td><td>87.8</td><td>58.8</td><td>73.2</td><td>61.1</td><td>37.5</td><td>46.2</td><td>66.8</td><td>60</td></tr><tr><td>Qwen3.5-4B (base)†</td><td>84.3</td><td>85.5</td><td>84.8</td><td>91.0</td><td>77.0</td><td>84.0</td><td>73.8</td><td>35.5</td><td>49.6</td><td>74.1</td><td>4,678</td></tr><tr><td colspan="10">Inference-time interventions on Qwen3.5-4B 3 (extra passes)</td><td></td><td></td></tr><tr><td> $\operatorname { V i C r o p }$ </td><td>83.5</td><td>86.8</td><td>84.8</td><td>91.9</td><td>79.0</td><td>85.4</td><td>72.1</td><td>41.4</td><td>52.7</td><td>75.9</td><td>4,230</td></tr><tr><td> $\mathrm { V C D } ^ { \dag }$ </td><td>82.6</td><td>93.4</td><td>86.9</td><td>90.5</td><td>80.0</td><td>85.2</td><td>71.1</td><td>37.9</td><td>50.1</td><td>75.3</td><td>4,612</td></tr><tr><td> $\mathrm { Z o o m E y e ^ { \dagger } }$ </td><td>87.0</td><td>97.4</td><td>91.1</td><td>89.5</td><td>69.0</td><td>79.2</td><td>69.8</td><td>37.1</td><td>49.1</td><td>72.1</td><td>5,371</td></tr><tr><td>Fixed-grid zoom†</td><td>82.6</td><td>90.8</td><td>85.9</td><td>90.8</td><td>76.2</td><td>83.5</td><td>69.1</td><td>39.1</td><td>50.1</td><td>74.1</td><td>4,725</td></tr><tr><td colspan="10">Training on Qwen3.5-4B (single pass)</td><td></td><td></td><td></td></tr><tr><td>EASE</td><td>82.6</td><td>86.2</td><td>84.0</td><td>90.1</td><td>77.2</td><td>83.7</td><td>70.8</td><td>36.1</td><td>48.9</td><td>73.6</td><td>5,316</td></tr><tr><td> $\mathrm { C E } \mathrm { S F T ^ { \dag } }$ </td><td>82.6</td><td>80.3</td><td>81.7</td><td>91.0</td><td>78.5</td><td>84.8</td><td>73.2</td><td>39.1</td><td>51.6</td><td>74.7</td><td>3,730</td></tr><tr><td> $\mathrm { \ A l i g n m e n t { S F T } ^ { \dag } }$ </td><td>86.1</td><td>85.5</td><td>85.9</td><td>92.8</td><td>78.0</td><td>85.4</td><td>73.2</td><td>41.4</td><td>53.1</td><td>76.1</td><td>3,940</td></tr><tr><td> $\mathrm { C E } \ S \mathrm { F T }  \mathrm { R L }$ </td><td>84.3</td><td>84.2</td><td>84.3</td><td>90.9</td><td>77.5</td><td>84.2</td><td>75.2</td><td>35.9</td><td>50.4</td><td>74.4</td><td>3,876</td></tr><tr><td> $\mathrm { A l i g n m e n t S F T }  \mathrm { R L }$ </td><td>87.0</td><td>88.2</td><td>87.4</td><td>90.8</td><td>81.0</td><td>85.9</td><td>72.8</td><td>39.3</td><td>51.6</td><td>76.1</td><td>3,112</td></tr><tr><td>LHA (attention RL)</td><td>85.7</td><td>86.8</td><td>86.1</td><td>92.1</td><td>81.0</td><td>86.6</td><td>71.8</td><td>41.2</td><td>52.5</td><td>76.6</td><td>3,541</td></tr></table>

Table 1: Supervising where a few heads look gives the most accurate single-pass Qwen3.5-4B model. Accuracy (%) in thinking mode (temperature 1, 32K-token budget) on $\mathbf { V } ^ { * }$ (attribute and spatial questions), HR-Bench 4K (single- and cross-instance) and TreeBench (perception and reasoning); Overall pools all 1,396 questions and Tokens is the mean completion length. Below EASE, SFT on the same self-generated trajectories uses cross-entropy (CE) alone or adds the alignment loss, and each is followed by the same outcome RL; LHA continues alignment SFT with attention RL. Rows average two decodes (†: one). Bold: best single-pass Qwen3.5-4B model.

Comparison with attention-supervised RL. EASE, the closest attention-supervised RL method, reaches 73.6% on our backbone, 3.0 points below LHA. Both supervise attention toward the evidence, but at different places: EASE matches a layer-averaged map to a smoothed prior, whereas LHA supervises the localization heads at the fact queries, with a normalizer-free contrast.

Comparison with released perception models. Under the same protocol, the 4B LHA model is more accurate overall than every released perception model we evaluate, including those built on 7B backbones. It leads ZwZ-4B by 3.8 points and TreeVGR-7B by 9.8, and it leads both on HR-Bench 4K and TreeBench, where evidence is tiny or spread over several objects.

<table><tr><td>Method</td><td>V*</td><td>Tree</td><td>V*+Tree</td></tr><tr><td>Base</td><td>74.3</td><td>39.8</td><td>50.8</td></tr><tr><td>CE SFT</td><td>77.0</td><td>39.8</td><td>51.7</td></tr><tr><td>Alignment SFT</td><td>78.0</td><td>40.5</td><td>52.5</td></tr><tr><td>CE SFT → RL</td><td>75.4</td><td>38.8</td><td>50.5</td></tr><tr><td>Alignment SFT → RL</td><td>78.5</td><td>41.0</td><td>53.0</td></tr></table>

Table 2: Qwen2.5-VL-3B (direct answers, greedy decoding). Accuracy (%) on $\mathbf { V } ^ { * }$ and TreeBench; both RL rows use the same outcome-reward RL.

Second backbone. An adapted direct-answer implementation carries the regional-supervision principle to Qwen2.5-VL-3B (Table 2; Appendix E.4). Supervised at the query that predicts the answer, alignment SFT already beats CE SFT on V\* and TreeBench, and after the same RL stage the aligned pipeline leads by 2.5 points (Table 4).

## 7 Analysis

Attention RL supplies a direct routing gradient. At the alignment-SFT checkpoint, we decompose the RL gradient on the query and key projections of layers 15, 19 and 23 and measure how each objective moves the routing score s of the supervised heads at fact tokens, on 12 rollout groups and eight held-out correct trajectories (Table 3a; Appendix E.3). In mixed groups the outcome gradient is more than an order of magnitude larger than the attention term, yet nearly orthogonal to the routing direction, and its effect on s flips sign from group to group. By contrast, the attention term is small but consistent: it raises s in all eight groups where it is active (cosine 0.10–0.32), including all-correct groups, where every outcome advantage is zero and outcome RL provides no signal at all. Outcome credit decides which trajectory to reinforce, while the attention term decides where to look; because it always pushes the same way, its effect accumulates.

<table><tr><td colspan="6">(a) Gradients at the start of RL ||g||</td></tr><tr><td colspan="2">mixed</td><td> $\cos ( - g , \nabla s )$  mixed</td><td colspan="2">mean  $\Delta s$  correct mixed</td><td> $\Delta s { > } 0$  groups</td></tr><tr><td colspan="6">Outcome PG 1.06</td></tr><tr><td colspan="2">KL 0.11</td><td>-0.04-0.04 -0.01-0.05</td><td>0 -0.03</td><td>-0.09 0.01</td><td>2/8 3/8</td></tr><tr><td colspan="6">Attention 0.06</td></tr><tr><td colspan="2"></td><td>0.10–0.32</td><td>0.18</td><td>0.13</td><td>8/8</td></tr><tr><td colspan="6">(b) Evidence preference during RL</td></tr><tr><td colspan="6"></td></tr><tr><td>Step</td><td>0</td><td>25</td><td></td><td>held-out</td><td>V*</td></tr><tr><td colspan="6"></td></tr><tr><td>β=0</td><td>.760</td><td>.760</td><td>.761</td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td>LHA</td><td></td><td>.820</td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td>.770</td><td>.842</td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td>.877</td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td>.860</td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>.895</td></tr><tr><td colspan="6"></td></tr><tr><td></td><td>.760</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td></td></table>

Table 3: The attention term supplies the routing signal that outcome RL lacks (Qwen3.5-4B, same alignment-SFT start). (a) Gradients on rollout groups of five: median norm and cosine with the routing direction ∇s in mixed groups, mean first-order change of s in all-correct and mixed groups, and how many of these eight groups it raises. (b) $p ( G ) / ( p ( G ) + p ( B ) )$ of the supervised heads on 82 fixed training-domain trajectories; held-out: the 17 not used in SFT; $\mathbf { V } ^ { \ast } \colon$ : at the object words of the base model’s $\mathbf { V } ^ { * }$ reasoning, against the benchmark’s own boxes (.844 at the start). Last two columns at step 50.

Routing improves over training. We teacherforce the same 82 correct training-domain trajectories through every checkpoint and track the supervised heads (Table 3b). With the attention term, their evidence preference rises from 0.760 to 0.860 in 50 steps, including on the 17 trajectories held out from SFT; without it, the $\beta { = } 0$ run stays at 0.761 (Appendix E.2). The sharper routing carries over to a benchmark that no stage trains on: at the 1,343 object words in the base model’s $\mathrm { V } ^ { \ast }$ reasoning, the supervised heads prefer $\mathbf { V } ^ { \ast } \mathbf { \bar { s } }$ own evidence boxes with .895 after attention RL, against .842 for the control (.844 at the start), and their in-box attention share rises from 0.076 to 0.126.

The redirection is visible on a second backbone. On Qwen2.5-VL-3B, alignment SFT nearly triples the visual attention inside the V\* evidence box at the answer, whereas CE SFT barely moves it, and the change survives RL (Appendix E.4).

Looking at the right place makes training and inference cheaper. Alignment SFT is light: it updates 2.6% of the language model, and the alignment gradient reaches only the queries of ten heads. In RL, the aligned policy writes 19% shorter rollouts than the CE-initialized one, so each step runs 15% faster; though it recomputes attention, LHA runs 8% faster per step than its $\beta { = } 0$ control, as its rollouts are 15% shorter (Table 9). At test time, alignment SFT→RL generates 20% fewer tokens than CE SFT→RL and LHA 24% fewer than the base model, not only in the truncated tail (Appendix F.4).

## 8 Conclusion

A thinking VLM usually has the evidence it needs and heads that can find it, but it lacks a training signal that tells those heads where to look. LHA provides this signal by supervising their attention directly. Grounding boxes and ablations identify the heads, box annotations pick the fact tokens, and one normalizer-free contrast supervises them in SFT and again in RL. Alignment makes SFT more accurate and the gain survives outcome RL; in RL the attention term keeps sharpening where the heads look. In a single pass, the resulting 4B model beats inference-time zooming and larger released models with shorter reasoning and faster RL training, and the principle carries over to a directanswer model. Localization heads are a practical handle for training perception: boxes are needed only in training, and at test time the model finds the evidence itself.

## Limitations

We train two Qwen backbones at the 3–4B scale and evaluate on three high-resolution benchmarks of multiple-choice questions; larger models, other model families and open-ended tasks such as charts, documents and video remain to be explored. LHA relies on box annotations for its training questions and on a small set of grounding questions to locate the heads of each new model, so extending it to domains without region annotations would call for automatic grounding. Fact tokens are identified by matching annotated phrases and counts, which leaves facts stated in other words unsupervised. Because RL training is expensive, each RL configuration is trained once, and further training seeds would better characterize run-to-run variance. Finally, the loss reads the attention of the supervised heads during training, which adds training cost, although inference is unchanged.

## References

Manoj Acharya, Kushal Kafle, and Christopher Kanan. 2019. TallyQA: Answering complex counting questions. In Proceedings of the AAAI Conference on Artificial Intelligence.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, et al. 2025. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Jianmin Chen, Jiaqi Tang, Wei Wei, Xiaogang Xu, Jiafei Wu, Zhe Liu, Qianzhou Wang, Yingying Yan, Botong Geng, Yuyang Xia, Lei Zhang, and Qifeng Chen. 2026. Remember-R1: Mitigating long-context visual forgetting through reinforcement learning. In Proceedings ofthe 34th ACM International Conference on Multimedia. ArXiv:2608.01314.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision. ArXiv:2403.06764.

Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James R. Glass. 2024. Lookback lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1419–1436, Miami, Florida, USA. Association for Computational Linguistics.

Matt Deitke, Christopher Clark, Sangho Lee, et al. 2024. Molmo and PixMo: Open weights and open data for state-of-the-art vision-language models. arXiv preprint arXiv:2409.17146.

Parsa Esmaeilkhani and Longin Jan Latecki. 2025. Direct visual grounding by directing attention of visual tokens. Preprint, arXiv:2511.12738.

Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. 2025. GRIT: Teaching MLLMs to think with images. In Advances in Neural Information Processing Systems. ArXiv:2505.15879.

Rohit Gandikota and David Bau. 2026. Gaze heads: How VLMs look at what they describe. Preprint, arXiv:2606.14703.

Ruina Hu, Chen Wang, Lai Wei, Jionghao Bai, Bin Yu, Weiran Huang, Kai Wang, and Yue Wang. 2026. Attend to evidence: Evidence-anchored spatial attention supervision for multimodal RLVR. Preprint, arXiv:2605.30912. Accepted to EMNLP 2026.

Siyuan Huang, Xiaoye Qu, Yafu Li, Yun Luo, Zefeng He, Daizong Liu, and Yu Cheng. 2026a. Spotlight on token perception for multimodal reinforcement learning. In International Conference on Learning Representations. ArXiv:2510.09285.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. 2026b. Vision-R1: Incentivizing reasoning capability in multimodal large language models. In International Conference on Learning Representations. ArXiv:2503.06749.

Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. 2025a. See what you are told: Visual attention sink in large multimodal models. In The Thirteenth International Conference on Learning Representations. ArXiv:2503.03321.

Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. 2025b. Your large vision-language model only needs a few attention heads for visual grounding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng Liu. 2025a. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. In Advances in Neural Information Processing Systems. ArXiv:2505.21523.

Fangyu Liu, Guy Emerson, and Nigel Collier. 2023. Visual spatial reasoning. Transactions ofthe Associationfor Computational Linguistics, 11:635–651.

Jingping Liu, Ziyan Liu, Zhedong Cen, Yan Zhou, Yinan Zou, Weiyan Zhang, Haiyun Jiang, and Tong Ruan. 2025b. Can multimodal large language models understand spatial relations? In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 620–632. Association for Computational Linguistics.

Lemao Liu, Masao Utiyama, Andrew Finch, and Eiichiro Sumita. 2016. Neural machine translation with supervised attention. In Proceedings of COLING

2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 3093– 3102.

Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. 2025c. NoisyRollout: Reinforcing visual reasoning with data augmentation. In Advances in Neural Information Processing Systems. ArXiv:2504.13055.

Ruilin Luo, Chufan Shi, Yizhen Zhang, Cheng Yang, Songtao Jiang, Tongkun Guan, Ruizhe Chen, Ruihang Chu, Peng Wang, Mingkun Yang, Yujiu Yang, Junyang Lin, and Zhibo Yang. 2026. From narrow to panoramic vision: Attention-guided coldstart reshapes multimodal reasoning. In International Conference on Learning Representations. ArXiv:2603.03825.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, and 7 others. 2022. Incontext learning and induction heads. Transformer Circuits Thread.

Tingting Qiao, Jianfeng Dong, and Duanqing Xu. 2018. Exploring human-like attention supervision in visual question answering. In Proceedings of the AAAI Conference on Artificial Intelligence.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3. 5.

Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, and Katerina Fragkiadaki. 2025. Grounded reinforcement learning for visual reasoning. In Advances in Neural Informa tion Processing Systems.

Ramprasaath R. Selvaraju, Stefan Lee, Yilin Shen, Hongxia Jin, Shalini Ghosh, Larry Heck, Dhruv Batra, and Devi Parikh. 2019. Taking a HINT: Leveraging explanations to make vision and language models more grounded. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV).

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. 2025. ZoomEye: Enhancing multimodal LLMs with human-like zooming capabilities through tree-based image exploration. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 6602–6618, Suzhou, China. Association for Computational Linguistics.

Chufan Shi, Cheng Yang, Yaokang Wu, Linghao Jin, Bo Shui, Taylor Berg-Kirkpatrick, and Xuezhe Ma. 2026. Are VLMs seeing or just saying? uncovering the illusion of visual re-examination. In Proceedings of the 43rd International Conference on Machine Learning. ArXiv:2605.15864.

Emma Strubell, Patrick Verga, Daniel Andor, David Weiss, and Andrew McCallum. 2018. Linguisticallyinformed self-attention for semantic role labeling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 5027–5038. Association for Computational Linguistics.

Hai-Long Sun, Zhun Sun, Houwen Peng, and Han-Jia Ye. 2025. Mitigating visual forgetting via take-along visual conditioning for multi-modal long CoT reasoning. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5158–5171. Association for Computational Linguistics.

Changyuan Tian, Zhicong Lu, Huaxing Liu, Xiang Wang, Shuai Li, Yu Chen, Wenqian Lv, Zichuan Lin, Juncheng Diao, and Deheng Ye. 2026. Faithful-MR1: Faithful multimodal reasoning via anchoring and reinforcing visual attention. Preprint, arXiv:2605.22072.

Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, Zhuochen Wang, and Zhaoxiang Zhang. 2026a. Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology. In International Conference on Learning Representations. ArXiv:2507.07999.

Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. 2025a. Pixel reasoner: Incentivizing pixel-space reasoning with curiositydriven reinforcement learning. arXiv preprint arXiv:2505.15966.

Haozhe Wang, Qixin Xu, Changpeng Wang, Taofeng Xue, Chong Peng, Wenhu Chen, and Fangzhen Lin. 2026b. Bad seeing or bad thinking? rewarding perception for multimodal reasoning. In Proceedings of the 43rd International Conference on Machine Learning. ArXiv:2605.14054.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. 2025b. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7907–7915.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, and Heng Ji. 2025c. Perception-aware policy optimization for multimodal reasoning. Preprint, arXiv:2507.06448.

Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang,

Linghe Kong, Yue Wang, Zhuosheng Zhang, and Weiran Huang. 2026. Zooming without zooming: Region-to-image distillation for fine-grained multimodal perception. In Proceedings of the 43rd International Conference on Machine Learning. ArXiv:2602.11858.

Penghao Wu and Saining Xie. 2024. V\*: Guided visual search as a core mechanism in multimodal LLMs. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2025. Retrieval head mechanistically explains long-context factuality. In The Thirteenth International Conference on Learning Representations.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In International Conference on Learning Representations.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, and Wenbing Tao. 2025a. Perception-R1: Pioneering perception policy with reinforcement learning. In Advances in Neural Information Processing Systems. ArXiv:2504.07954.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, et al. 2025b. DAPO: An open-source LLM reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. 2025a. MLLMs know where to look: Training-free perception of small visual details with multimodal LLMs. In The Thirteenth International Conference on Learning Representations.

Wuwei Zhang, Fangcong Yin, Howard Yen, Danqi Chen, and Xi Ye. 2025b. Query-focused retrieval heads improve long-context reasoning and re-ranking. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 23791–23805, Suzhou, China. Association for Computational Linguistics.

Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, and Qing Li. 2025c. Adaptive chain-of-focus reasoning via dynamic visual search and zooming for efficient VLMs. Preprint, arXiv:2505.15436.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. 2025d. SparseVLM: Visual token sparsification for efficient vision-language model inference. In Proceedings of the 42nd International Conference on Machine Learning. ArXiv:2410.04417.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. 2026. DeepEyes: Incentivizing “thinking with images” via reinforcement learning. In International Conference on Learning Representations. ArXiv:2505.14362.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, et al. 2025. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

Xin Zou, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang, Jia Liu, Chang Tang, and Xuming Hu. 2025. Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 80873–80899. PMLR.

## Appendix

A Extended related work.

B Evaluation protocol, scoring audit and paired statistics.

C Diagnostics behind §4, including head selection.

D Localization heads in Qwen2.5-VL-3B.

E Training details, routing during RL, the RL gradient analysis and the Qwen2.5-VL-3B implementation.

F Greedy decoding, gains by evidence size, response length and inference cost.

G All evaluated endpoints, at temperature 1 and with greedy decoding.

H Full-width figures.

## A Extended Related Work

Perception in thinking VLMs. Longer chains drift toward language priors (Liu et al., 2025a), and the visual re-examination that thinking models verbalize rarely happens (Shi et al., 2026). Fading attention to the image motivates re-injecting visual information (Sun et al., 2025; Zou et al., 2025) or rewarding sustained attention (Chen et al., 2026). We locate the problem in a few localization heads. We supervise where they look inside the image, evidence against background, rather than how much attention the image receives.

Inference-time zooming and tools. Guided search (Wu and Xie, 2024; Shen et al., 2025), attention-selected crops (Zhang et al., 2025a), contrastive decoding (Leng et al., 2024) and crop tools learned with RL (Zheng et al., 2026; Zhang et al., 2025c) recover small details at extra inference cost. LHA keeps a single pass and combines with these methods.

RL for perception and grounded reasoning. Rule-based RL has been adapted to perception (Yu et al., 2025a; Huang et al., 2026b; Liu et al., 2025c). Grounded-reasoning methods reward boxes written in the chain (Fan et al., 2025; Sarch et al., 2025; Wang et al., 2026a) or synthesize harder training data (Wei et al., 2026). Perception-aware objectives act on the policy’s outputs. PAPO adds a KL term between predictions with intact and masked images (Wang et al., 2025c), VPPO reweights advantages by visual dependency (Huang et al., 2026a), and MoCA rewards stated perception through a text-only verifier (Wang et al., 2026b). LHA supervises the internal routing of a few heads directly, on the fact tokens of correct rollouts.

Supervising attention. Attention has long been supervised with word alignments (Liu et al., 2016), with syntactic parents for one head (Strubell et al., 2018) and with human attention in VQA (Qiao et al., 2018; Selvaraju et al., 2019). For VLMs, KLAL aligns answer-token attention with boxes (Esmaeilkhani and Latecki, 2025). EASE adds a KL term toward a smoothed evidence prior on the head-averaged attention of one layer, during RL on correct rollouts (Hu et al., 2026). Faithful-MR1 supervises the attention of an inserted focus token on a prescribed set of layer–head pairs, followed by RL with a counterfactual attention reward (Tian et al., 2026). AVAR combines an attention-guided cold start with visual-attention rewards (Luo et al., 2026). LHA differs from all three in its supervision interface. It targets functional heads selected independently of the training objective, at the native positions that predict visual facts in the model’s own reasoning. Its regional auxiliary loss contrasts the evidence with an equally large background. This ratio is unaffected by total visual mass and attention sinks (Xiao et al., 2024; Kang et al., 2025a), and the same loss serves SFT and RL.

Attention heads as mechanisms. Sparse functional heads are documented in language models (Olsson et al., 2022; Wu et al., 2025; Zhang et al., 2025b; Chuang et al., 2024) and in VLMs (Kang et al., 2025b; Gandikota and Bau, 2026). We make them the target of a training loss.

## B Evaluation Protocol

Suite and decoding. The suite contains 1,396 questions at native resolution: 191 from V\*, 800 from HR-Bench 4K and 405 from TreeBench; a 4K image yields about 8,100 visual tokens. Qwen3.5- 4B models run in thinking mode at temperature 1 with a 32K response budget, served with SGLang, and released models use their own generation format. Qwen2.5-VL-3B answers directly with greedy decoding (Appendix E.4); its RL comparison covers the 596 V\* and TreeBench questions. Two decodes of the same checkpoint differ by about 15 correct answers (1.1 points), so trained Qwen3.5- 4B models are reported as two-decode means wherever both decodes are available; single-decode rows

<table><tr><td>Decoding</td><td>∆</td><td>95% CI +1-</td></tr><tr><td>Alignment SFT→RL vs. CE SFT→RL T=1, decode 1 +1.3 T=1, decode 2 +2.2 T=1, both decodes +1.8 greedy +2.1</td><td>[-0.4, 3.0] [0.5, 3.9] [0.5, 2.9]</td><td>91/73 92/61</td></tr><tr><td colspan="3">LHA vs. its β=0 control T=1, decode 1 +0.9 [−0.9, 2.8] +0.9</td></tr><tr><td>T=1, decode 2 T=1, both decodes +0.9</td><td>[−1.1,2.7] [−0.5, 2.2]</td><td>87/75</td></tr><tr><td colspan="3">Qwen2.5-VL-3B: alignment vs. CE SFT→RL greedy, V*+Tree +2.5 [0.3, 4.7] 30/15</td></tr></table>

Table 4: Paired comparisons between runs that differ in one component. Accuracy difference (points), 95% bootstrap interval clustered by source question, and questions gained (+) and lost (−).

## are marked.

Scoring and audit. Rules extract the committed option, and an independent judge model (Qwen3.6- 35B-A3B) reads only the responses the rules cannot parse; all Qwen3.5-4B and released models share the rules and the judge. We checked the scoring of all reported evaluations by hand, including every judge decision, and the automatic scores agreed with the manual labels in every case. The base-model reference was evaluated once under the frozen protocol and never re-scored.

Separation of training and evaluation data. All training questions come from TallyQA, PixMo-Count, Spatial-MQA (Liu et al., 2025b) and VSR. Before training we removed every question whose image matches an evaluation image, exactly or as a near-duplicate under a perceptual hash (53 questions). The localization heads are ranked on 96 grounding questions from the RL training pool whose images appear in no SFT trajectory or evaluation set (Appendix C.3).

Paired comparisons. Table 4 gives paired bootstrap intervals for the controlled comparisons of §6. Each pair shares questions and decoding seeds; resampling is stratified by benchmark, and HR-Bench’s four option rotations are resampled together by source question.

## C Diagnostics

All numbers in this section are for the frozen Qwen3.5-4B base model under the protocol of Appendix B. The exceptions are Appendix C.2 (Qwen2.5-VL-3B) and the direct-answer models of Appendix C.5.

<table><tr><td>Subset</td><td>n</td><td>Evid.</td><td>Ctrl.</td></tr><tr><td>V*</td><td>191</td><td>43.5 82.2</td><td>-38.7 [-46.6, -30.4]</td></tr><tr><td>TreeBench</td><td>333</td><td>33.3</td><td>45.9 -12.6 [-18.6, -6.6]</td></tr><tr><td>Area &lt;0.21%</td><td>174</td><td>42.5 70.1</td><td>-27.6 [-36.2, −19.0]</td></tr><tr><td>0.21–2.4%</td><td>175</td><td>37.1</td><td>62.3 -25.1 [-33.7, -16.6]</td></tr><tr><td>Area &gt;2.4%</td><td>175</td><td>31.4</td><td>45.1 -13.7 [−22.9, -4.6]</td></tr><tr><td>All</td><td>524</td><td>37.0</td><td>59.2 -22.1 [-27.1, −17.2]</td></tr></table>

Table 5: Evidence masking by subset. Accuracy (%) with the evidence or an equal-area control region masked; area is the box’s fraction of the image.

## C.1 Evidence masking

Table 5 splits the masking result of §4.1 by benchmark and by the area of the evidence box. The masked and control images share one PNG pipeline, so the evidence-versus-control contrast isolates the evidence. Masking the control region changes accuracy by −1.1 points relative to the original images (CI [−4.4, 2.3]). The drop is largest on V\*, whose questions concern a single small object, and it grows as the box shrinks. Without the evidence, the model also searches longer (9,251 vs. 5,784 mean tokens).

## C.2 Encoder-side token controls

A matrix of input interventions on Qwen2.5-VL-3B separates what the vision encoder preserves from what the language model selects (Figure 10). The runs use native resolution (up to 12.8M pixels), greedy decoding, a 4,096-token budget and one LLM judge. We report paired changes against the full image. The key condition gives the language model only the post-encoder visual tokens inside the ground-truth box, taken from the encoding of the full image. These tokens are a subset of those the model already receives. We remove the others to locate where the evidence is lost, not for efficiency as in token pruning (Chen et al., 2024; Zhang et al., 2025d). Two conditions control for input size and position: the full image given twice, and the full image plus a random crop.

On 499 rendered small-object scenes, keeping only the box tokens raises accuracy by 16.8 points (CI [12.6, 21.0]), and a re-encoded crop of the box raises it by 27.7 (CI [23.2, 32.1]). Appending the same box tokens to the full image adds 1.8 points (CI [−1.0, 4.8]), and the size and position controls add 1.8 and 0.2. On these synthetic scenes the evidence therefore survives encoding, and removing the other tokens helps more than repeating the evidence.

<table><tr><td>Visual tokens given</td><td> $\mathrm { V } ^ { \ast }$ </td><td>Tree</td><td>All</td></tr><tr><td>Full image</td><td>83.8</td><td>46.5</td><td>60.1</td></tr><tr><td>Evidence box only</td><td>77.5</td><td>51.1</td><td>60.7</td></tr><tr><td>Equal-area control only</td><td>44.5</td><td>34.2</td><td>38.0</td></tr><tr><td>Full image + box tokens</td><td>83.8</td><td>47.7</td><td>60.9</td></tr><tr><td>Full image + control tokens</td><td>80.6</td><td>45.6</td><td>58.4</td></tr></table>

Table 6: Post-encoder token controls on Qwen3.5-4B (thinking mode, one decode; 191 $\mathrm { V } ^ { \ast }$ and 333 TreeBench questions). Accuracy (%).

Qwen3.5-4B on natural images. We repeat the token edits on Qwen3.5-4B in thinking mode for the 524 V\* and TreeBench questions whose boxes admit an equal-area control (Table 6). The edits remove visual tokens after the vision encoder has processed the full image and keep the remaining tokens at their original positions. Keeping only the tokens inside the evidence box preserves accuracy (60.7% against 60.1% with the full image), whereas keeping an equal-area control set instead drops it to 38.0% (box minus control: +22.7 points, CI [17.7, 27.9]). Appending the box tokens to the full image changes accuracy by only +0.8 points. The box tokens alone thus carry the answer, and giving the model more evidence does not help; selecting it does. On ${ \mathrm { V } } ^ { * }$ , keeping only the box tokens costs some context (−6.3 points); on TreeBench, whose questions relate several boxed objects, accuracy is even slightly higher (+4.5). For Qwen2.5-VL-3B on V\*, box tokens alone cost 9.9 points (CI [−16.8, −3.7]). Because answers draw on some context around the box, LHA contrasts the evidence with an equally large background inside the image instead of removing the rest of it.

## C.3 Localization head selection

We score each of the 128 heads of the eight fullattention layers on 96 held-out grounding questions. They are drawn from the RL training pool and share no images with the SFT trajectories or any evaluation set. For a head, the score is log $\left( p _ { G } + \epsilon \right) - \log ( \bar { p } _ { C } + \epsilon )$ . Here $p _ { G }$ is the share of its attention over visual tokens that falls in the evidence box, and $\bar { p } _ { C }$ is the mean share over two equal-area control boxes. Accuracy is never used for selection. The four highest-scoring heads are L19H14, L15H11, L19H2 and L23H9. The ranking computed independently on $\mathrm { V } ^ { \ast }$ correlates with this ranking (Spearman 0.80) and shares six of the top eight heads. LHA supervises the top ten (three in layer 15, six in layer 19, one in layer 23).

<table><tr><td></td><td>Base</td><td>R4</td><td>L4</td><td>L4–R4 [95% CI]</td></tr><tr><td> $\mathrm { V } ^ { \ast }$ </td><td>84.8</td><td>83.2</td><td>69.1</td><td>-14.1 [−19.9, -8.4]</td></tr><tr><td>HR-4K</td><td>84.0</td><td>83.3</td><td>68.3</td><td>-15.0 [-18.0, -12.0]</td></tr><tr><td>Tree</td><td>49.6</td><td>49.6</td><td>40.5</td><td>-9.1[-13.8, -4.4]</td></tr><tr><td>All</td><td>74.1</td><td>73.5</td><td>60.3</td><td>-13.2 [-15.5, -10.9]</td></tr></table>

Table 7: Localization-head ablation by benchmark. Accuracy (%) of the base model and with four random (R4) or localization (L4) heads ablated.

## C.4 Localization-head ablation

Table 7 gives the generation ablation of §4.3 per benchmark. The drop appears on all three benchmarks.

A forced-choice probe isolates the answer decision from generation. We select the top eight heads on 96 $\mathbf { V } ^ { * }$ questions and test on the other 95, reading the option distribution directly after an empty thinking block. On the held-out questions, the selected heads place more attention on the answer box than layer-matched random heads (+0.95 in log ratio, CI [0.81, 1.10]). Zeroing them lowers forced-choice accuracy from 52.6% to 36.8%. Layer-matched random or bottom-ranked heads leave it at 52.6% and 51.6%.

## C.5 Where the image is read: direct answers and thinking

Direct answers. Correct direct answers have a higher evidence share than incorrect ones across six models. The models are Qwen2.5-VL-3B and 7B (Bai et al., 2025), LLaVA-OneVision-7B (Li et al., 2024), InternVL3-8B (Zhu et al., 2025), Pixel-Reasoner (Wang et al., 2025a) and a region-supervised Qwen2.5-VL-3B. For the five top-ranked localization heads of each model, we measure the share of their visual attention inside the evidence box at the first answer token. Across 12 model–dataset settings on $\mathbf { V } ^ { * }$ , TreeBench and a held-out set, correct answers have the higher share in all 12, by 0.040 to 0.113, with CIs above zero in nine.

Qwen3.5-4B before reasoning. Figure 4 plots the evidence enrichment of the four localization heads at the last question token, for questions the base model answers correctly and incorrectly. Enrichment is the share of their visual attention inside the ground-truth box divided by the box’s share of visual tokens. For boxes smaller than one visual token we use the token that contains the box center. On ${ \mathrm { V } } ^ { * }$ , enrichment exceeds 1 for 80% of correct and 55% of incorrect answers (medians 2.56 and 1.44). It predicts correctness with AUC 0.66 (CI [0.54, 0.77]; 162 correct, 29 incorrect). TreeBench questions relate several objects that the reasoning visits one by one. Their evidence is read at the fact tokens of the chain rather than at the question end, which is where LHA places its supervision.

![](images/2cd669e809220dbe26ee13ae7c19d6c528afa1d292a85d6569508e4dd549ea6c.jpg)  
Figure 4: Question-time evidence enrichment of the four localization heads for correct and incorrect base answers (log scale; bars: medians; dotted: uniform attention).

Qwen3.5-4B along the reasoning. We take 200 training questions outside the head-selection images, 50 from each of the four source datasets, that the frozen base model answers correctly in one to four of five samples. For each question we keep two finished rollouts, one correct and one incorrect, each under 8,000 tokens. We teacher-force these rollouts and record the ten supervised heads at fact tokens in the first and second half of the chain. The share is $p ( G ) / ( p ( G ) + p ( B ) )$ ), averaged over queries, heads and rollouts, with G the box of the stated object and B a disjoint equal-area background. It is 0.77 in the first half of the chain and 0.72 in the second (median enrichment 3–3.6). At the token that states the final answer it is 0.50, the value of uniform attention. The image is thus read where facts are stated, and this reading weakens as the chain grows.

Qwen3.5-4B on benchmark rollouts. We repeat the measurement on 220 base rollouts of 110 V\* and TreeBench questions, one correct and one incorrect per question, at eight positions sampled from each tenth of the reasoning. The evidence set is the union of the question’s annotated boxes. The share falls steadily from 0.71 in the first tenth of the chain to 0.61 in the last, and to 0.59 at the answer, in correct and incorrect rollouts alike.

## C.6 Decomposing the drift of visual attention

The drift of visual attention splits into dilution by the growing text and a weakening preference for the image. For a head and query position, let $m _ { \mathrm { v i s } }$ be its attention mass on image keys, and let $n _ { \mathrm { i m g } }$ and $n _ { \mathrm { t e x t } }$ be the numbers of image and text keys (excluding the first four sink tokens). Let $\Delta =$ $\mathrm { L M E } ( z _ { \mathrm { i m g } } ) - \mathrm { L M E } ( z _ { \mathrm { t e x t } } )$ be the gap between the log-mean-exp logits of the two key sets. Then, ignoring sinks,

$$
\log m _ { \mathrm { v i s } } = \log \sigma \big ( \Delta + \log n _ { \mathrm { i m g } } - \log n _ { \mathrm { t e x t } } \big ) ,
$$

so the decay splits into a dilution term, log $n _ { \mathrm { t e x t } } ,$ and an affinity term, $\Delta .$ . We teacher-force 1,072 same-question pairs of best-of-five base rollouts on training questions (2,144 rollouts, median thinking length about 3,300 tokens). We record the four localization heads along the chain (Figure 11). The log visual share falls with slope −1.09 against log $n _ { \mathrm { t e x t } }$ , close to pure dilution by the growing text. The affinity $\Delta$ falls as well (slope −0.43, CI [−0.46, −0.40]). It still falls at −0.24 (CI [−0.26, −0.21]) when computed from the non-rotary query and key dimensions alone. The heads’ content preference for the image therefore weakens, in addition to their positional reach. Within the image, on the 1,040 pairs (2,080 rollouts) with valid evidence boxes, the share on the evidence box drops from 0.53 to 0.40 between the first and last quarter of the chain (slope −0.06 against log $n _ { \mathrm { t e x t } } .$ , Figure 11c). Over all 128 full-attention heads the affinity slope is −0.15, so the localization heads drift faster than the average head.

Table 8 adds $\beta \log ( n _ { \mathrm { t e x t } } / n _ { 0 } )$ to the image-key logits of the four heads at every decoding step, where $n _ { 0 }$ is the number of text keys at the end of the prompt. At β=1 this bias offsets the dilution term. Accuracy stays at the base level. Restoring how much attention reaches the image does not tell the heads where in it to look, and that is what the contrastive loss of LHA supervises.

## D Localization Heads in Qwen2.5-VL-3B

This section analyzes Qwen2.5-VL-3B-Instruct answering directly, without thinking. Attention is read at the query that produces the answer. The evidence share of a head is the fraction of its visual attention that falls on the visual tokens inside the ground-truth box. Layers and heads are indexed from 0.

<table><tr><td>β</td><td>Acc.</td><td> $\Delta$  vs. base [95% CI]</td><td>Trunc.</td><td>Tokens</td></tr><tr><td>0 (base)</td><td>74.1</td><td></td><td>37</td><td>4,678</td></tr><tr><td>0.25</td><td>74.1</td><td>−0.1 [−1.9, 1.7]</td><td>48</td><td>4,751</td></tr><tr><td>0.5</td><td>73.9</td><td>−0.3 [−2.0, 1.4]</td><td>42</td><td>5,013</td></tr><tr><td>1</td><td>74.2</td><td>+0.1 [−1.8, 2.0]</td><td>54</td><td>5,100</td></tr></table>

Table 8: Length-aware visual logit bias at the four localization heads (1,396 questions, one decode). Trunc.: budget-truncated responses; Tokens: mean completion tokens.

![](images/799e331c4557fdd0e571f7f3b8e841d6fb692e9d14378c7c9fab769ca73f44ee.jpg)

![](images/64b6f08ed7025611d28d8f2e80fed2a14e8e9ca28cf2ee932f8c0d106d5dd4f7.jpg)  
Figure 5: Evidence routing predicts answer correctness (Qwen2.5-VL-3B, 166 $\mathbf { V } ^ { \ast }$ questions, five fixed localization heads at the answer position). (a) Evidence share for correct and incorrect greedy answers; diamonds: means. (b) Accuracy by quartile of the share.

Localization heads are sparse and deep. Over the 191 $\mathbf { V } ^ { * }$ questions, 17 of the 576 heads place more than 5% of their visual attention on the evidence box, which covers 0.83% of the visual tokens on average (Figure 12a). The strongest head, L27H1, places 17.9% there, and the five strongest lie in layers 24 to 27. On a single question the same pattern appears layer by layer, as the evidence share stays at or below 0.03 through layer 22 and peaks at 0.45 in layer 27 (Figure 12c).

Routing predicts correctness. On 166 V\* questions, five fixed localization heads place 22.1% of their visual attention on the evidence when the answer is correct and 11.0% when it is wrong (Mann– Whitney $\scriptstyle { p = 0 . 0 0 6 } ;$ ; Figure 5). Accuracy rises from 63.4% in the lowest quartile of this share to 90.7% in the highest (point-biserial r=0.24).

Adding a crop moves attention onto the evidence. Figures 13 and 14 show two questions that the model answers wrongly from the full image. It answers both correctly once a crop around the evidence box is appended as a second image (ZoomIn). The evidence share of eight localization heads at the answer tokens rises from 7.1% to

59.7% on TreeBench and from 2.8% to 40.3% on RefChartQA. Most of the added share falls on the crop.

## E Training Details

## E.1 Alignment SFT on Qwen3.5-4B

We train the query, key, value and output projections of the full-attention layers 15, 19 and 23 (12 matrices, 110M parameters). Training uses FP32 with AdamW, a constant learning rate of $1 0 ^ { - 5 }$ batch size 16 and gradient clipping at 1. Alignment is data-efficient. A compact set of 65 correct best-of-five responses of the base model suffices, drawn from the box-annotated training questions of §5.2 (34 PixMo-Count, 29 TallyQA and 2 spatial questions) and disjoint from the head-selection set; a fixed hash of each question splits a pool of 82 into these 65 and 17 held-out trajectories used for routing measurements (Appendix E.2). The alignment loss acts on the predictors of 44 fact tokens, in the 15 trajectories whose reasoning names an annotated object or states its count exactly as annotated; a phrase is eligible only if it does not occur in the question and its context passes a manual review, and coordinates generated by the model are never used as evidence. This compact set of supervised positions already yields the gains of Table 1. Factbased selection concentrates the supervision: of 64 query positions sampled uniformly from each trajectory, 13.7% fall on fact content, against 98.6% for fact-first selection. Training runs for 32 updates. The alignment term is computed with hidden states and keys detached. Its background set $B _ { i }$ is fixed before training: the visual tokens outside every annotated box that lie nearest to the centroid of $G _ { i }$ as many as in $G _ { i }$ . CE SFT uses exactly the same data order, updates and parameters with $\alpha { = } 0$ . Table 4 reports the paired evaluation of the matched SFT→RL pipelines under sampled and greedy decoding.

## E.2 Attention RL on Qwen3.5-4B

We use verl with GRPO, 16 questions $\times \ 8$ rollouts per step, temperature 1, a 32,768-token response budget and a low-variance KL loss of weight 0.01. All parameters are trained, and attention uses FlashAttention-2. CE SFT→RL and alignment SFT→RL in Table 1 use learning rate $5 \times 1 0 ^ { - 7 }$ with one update per step and train for 100 steps on the first 1,600 training questions, in the same fixed order. LHA and its $\beta { = } 0$ control (Table 4) start from the same alignment-SFT checkpoint and differ only in $\beta .$ They use a constant learning rate of $1 0 ^ { - 6 }$ and mini-batches of 8 questions (two optimizer updates per step) on the 3.3K training questions that the base model answers correctly in one to four of five samples, in the same fixed order, and both train for 50 steps. The attention branch of the shared training code is enabled only for $\beta > 0$ . Each run uses four GPUs of the same node. Aligned policies write shorter rollouts and therefore train faster (Table 9); this holds for LHA although it recomputes the attention of the supervised heads.

<table><tr><td>Run</td><td>s / step</td><td>Rollout tokens</td></tr><tr><td>Outcome RL, 100 steps</td><td></td><td></td></tr><tr><td> $\mathrm { C E } \ S \mathrm { F T } \{  \mathrm { R L }$ </td><td>448</td><td>3,170</td></tr><tr><td>Alignment  $\mathrm { S F T } {  } \mathrm { R L }$ </td><td> $3 7 9 \ : ( - 1 5 \% )$ </td><td> $2 { \sqrt { 5 7 0 } } \left( - 1 9 \% \right)$ </td></tr><tr><td>Attention RL, 50 steps</td><td></td><td></td></tr><tr><td>β=0 control</td><td>667</td><td>6,179</td></tr><tr><td>LHA</td><td> $6 1 3 ( - 8 \% )$ </td><td> ${ \bar { \mathbf { 5 } } } , 2 3 \mathbf { 0 } \left( - 1 5 \% \right)$ </td></tr></table>

Table 9: Training cost of RL (Qwen3.5-4B, four GPUs of one node). Wall-clock time per step and mean rollout length, averaged over training; the two runs of each pair use the same RL data order and hyperparameters.

Compute. All Qwen3.5-4B training and evaluation runs on one node with eight NVIDIA H20 GPUs (96 GB each); Qwen2.5-VL-3B runs on a server with eight NVIDIA A100-80GB GPUs. On the H20 node, alignment SFT and CE SFT each take about 0.8 GPU-hours on one GPU. CE $\mathrm { S F T } {  } \mathrm { R I }$ and alignment SFT→RL take about 50 and 42 GPU-hours, LHA and its $\beta { = } 0$ control about 34 and 37, and the EASE baseline about 46, each on four GPUs. One decode of the 1,396 evaluation questions takes about two GPU-hours. Including exploratory runs, the project used the H20 node for about eight weeks, at most about 11K GPUhours, and the A100 server for about a month, at most about 6K GPU-hours, for Qwen2.5-VL-3B and external baselines.

Attention term. The term supervises at most 64 fact positions per correct rollout: object nouns of the annotated phrases and enumerated counts inside the reasoning, excluding restated answers, coordinates and option letters. It needs the attention weights of the ten supervised heads at these queries, which we recompute explicitly. For each rollout the evidence set is the union of the annotated boxes it refers to, and the background is an equally large random set of visual tokens disjoint from every box. Like the policy loss, the term is averaged over the rollouts of each mini-batch; hidden states and keys are not detached, and advantages are left unchanged.

![](images/f2ae4409f8862875d94778de76996e81ba8b0210923c24a2f269e4094edbef37.jpg)  
Figure 6: Evidence preference along RL (Qwen3.5- 4B, same alignment-SFT start). The ten supervised heads at the fact tokens of 82 fixed training-domain trajectories, teacher-forced through each checkpoint; dashed: the 17 held out from SFT; bands: 95% bootstrap intervals.

![](images/84ef789439765211d381850318be7581725c2c6120dbf1ce9b149890b189fc66.jpg)

![](images/07f6534e86a7464c1570df2693405bf6a1323d45eb5f4c8083584da8318f70a1.jpg)  
Figure 7: Gradient decomposition at the alignment-SFT policy (Qwen3.5-4B; points: groups of five rollouts). (a) Norm of each objective’s gradient against its cosine with the routing direction $\nabla s .$ . (b) First-order change of s under one gradient step (bars: means); allwrong groups receive neither term.

Routing on fixed trajectories. We teacherforce 82 fixed trajectories, the 65 alignment trajectories and 17 held-out ones (135 fact positions), through each RL checkpoint and measure $p ( G ) / ( p ( G ) + p ( B ) )$ of the ten supervised heads at the fact positions (Figure 6). It is 0.760 at the alignment-SFT start; after 50 steps it is 0.860 with the attention term and 0.761 without it, a paired difference of +0.10 [0.08, 0.12] over trajectories, with 95% of trajectories higher under the attention term.

## E.3 Gradient decomposition of the RL objective

Figure 7 measures which part of ${ \mathcal { L } } _ { \mathrm { R I } }$ moves the routing of the supervised heads. We take gradients at the alignment-SFT policy of Qwen3.5-4B with respect to the query and key projections of layers 15, 19 and 23. The routing score $s = \log ( p _ { G } / p _ { B } )$ is the attention of the ten supervised heads on the evidence G relative to an equally large background B. It is measured at the queries that predict fact tokens and averaged over eight held-out correct trajectories. Each objective’s gradient g is computed on 12 groups of five rollouts sampled from the base model: four all-correct, four mixed and four all-wrong. We use the RL weights, $\beta { = } 0 . 0 1$ and a KL coefficient of 0.01 toward the base model as reference. For each objective, $- \nabla s ^ { \intercal } g$ is the first-order change of s under a gradient step, and the cosine between −g and ∇s measures its alignment with the routing direction. The attention term gives $- \nabla s ^ { \intercal } g$ from 0.02 to 0.37 and a cosine from 0.10 to 0.32 in the eight groups where it is active. In all-correct groups every advantage is zero, so the outcome policy gradient vanishes while the attention term contributes a mean gain of 0.18. In mixed groups the outcome policy gradient is about 18 times larger than the attention term, with a cosine to ∇s between −0.04 and 0.04. The KL term stays small in every group, so using the base model rather than the aligned start as its reference has little effect on the decomposition.

## E.4 Qwen2.5-VL-3B implementation

The Qwen2.5-VL-3B validation applies the same regional-supervision principle in the no-think setting. It trains an evidence-versus-background loss on localization heads in SFT, follows it with outcome-reward RL, and compares against CE SFT→RL, which drops only the alignment term. Its implementation differs from the Qwen3.5-4B algorithm as follows (Qwen3.5-4B first, Qwen2.5- VL-3B second).

• Supervised query: fact tokens inside the reasoning; the query that predicts the first answer token.

• Heads: ten fixed heads ranked on held-out grounding questions (Appendix C.3); the five heads with the highest localization score on each training example.

• Background: an equal-size set outside every box; all non-evidence visual tokens.

• Trainable parameters: full attention projections of layers 15, 19 and 23 (2.6% of the language model), alignment gradient to queries only; LoRA (rank 16) on the attention projections of

![](images/046603dc1065eaf3db0c7f0b48e20f5d0680bd747cd6187ec77071d1c15c8afe.jpg)  
Figure 8: Alignment SFT raises the evidence share at the answer position, and RL preserves it (Qwen2.5- VL-3B, V\*). Grey: CE; red: alignment.

all 36 layers.

• Weighting: adaptive α of Eq. (5); a fixed weight of 0.5.

• SFT data: 65 self-generated correct trajectories (32 updates); 30K annotated short-answer examples (1,000 updates), with multi-box examples trained by cross-entropy only.

• RL: GRPO with outcome reward, 16×8 rollouts (Appendix E.2); GRPO with an LLM-judged outcome reward plus a 0.2 format score, 32×8 rollouts, 300 steps; all parameters trained in both.

• Evaluation: thinking mode at temperature 1 (Appendix B); direct answers with greedy decoding, scored by rule matching on ${ \mathrm { V } } ^ { * }$ , an LLM judge on TreeBench and the main-text scorer on HR-Bench 4K. The RL comparison of Qwen2.5-VL-3B is evaluated on V\* and TreeBench.

Routing after each stage. Figure 8 measures the share of visual attention inside the V\* evidence box at the query that predicts the first answer token. The share is Eq. (1) with R the evidence box, averaged over the five heads with the highest share on each question, on 164–166 paired V\* questions per model. These heads are chosen per question with the ground-truth box, as in training. The measure therefore shows that the training objective is optimized, and it complements the fixed-head diagnostics of §4. The mean share is 0.19 for the base model, 0.21 after CE SFT and after CE SFT→RL, 0.56 after alignment SFT and 0.54 after alignment SFT→RL. A question counts as focused when the share is at least 0.3. The focused fraction is 28.9% for the base model, 31.1% after CE SFT and 63.4% after alignment SFT. It is 33.7% after CE SFT→RL and 64.5% after alignment SFT→RL.

## F Additional Results

## F.1 Inference cost

<table><tr><td>Method</td><td>Passes</td><td>+Img.</td><td>Tokens</td><td>Acc.</td></tr><tr><td>Base†</td><td>1</td><td>0</td><td>4,678</td><td>74.1</td></tr><tr><td>ViCrop</td><td>3</td><td>3</td><td>4,230</td><td>75.9</td></tr><tr><td>VCD†</td><td>2×/token</td><td>1</td><td>4,612</td><td>75.3</td></tr><tr><td>ZoomEye†</td><td>124.2</td><td>123.2</td><td>5,371</td><td>72.1</td></tr><tr><td>Fixed-grid zoom†</td><td>1</td><td>4</td><td>4,725</td><td>74.1</td></tr><tr><td>GRPO</td><td>1</td><td>0</td><td>4,828</td><td>74.0</td></tr><tr><td>DAPO</td><td>1</td><td>0</td><td>4,011</td><td>73.6</td></tr><tr><td>EASE</td><td>1</td><td>0</td><td>5,316</td><td>73.6</td></tr><tr><td>CE SFT → RL</td><td>1</td><td>0</td><td>3,876</td><td>74.4</td></tr><tr><td>Alignment SFT → RL</td><td>1</td><td>0</td><td>3,112</td><td>76.1</td></tr><tr><td>+ head-guided zoom</td><td>3</td><td>4</td><td>2,972</td><td>76.6</td></tr><tr><td>LHA</td><td>1</td><td>0</td><td>3,541</td><td>76.6</td></tr></table>

Table 10: Inference cost per question. Passes: full forward passes over the prompt (VCD: two per decoding step); +Img.: image encodings beyond the original image; Acc.: overall accuracy (%).

Table 10 counts passes and extra image encodings.

## F.2 Greedy decoding

Alignment SFT→RL is also the most accurate model under greedy decoding (Table 12). Greedy decoding makes the base model loop: 266 of 1,396 responses (19.1%) are budget-truncated, compared with 37 at temperature 1, so all accuracies are lower. Its gain over CE SFT→RL (+2.1 points, CI [0.4, 4.0]) is close to that at temperature 1, and it generates 24.1% fewer completion tokens.

## F.3 Gains by evidence size

On the 524 V\* and TreeBench questions with an equal-area control (Appendix C.1), we split the questions into tertiles by the fraction of the image covered by the evidence boxes and compare alignment SFT→RL with CE SFT→RL, averaging both decodes. Alignment SFT→RL gains over CE SFT→RL in every tertile: +2.6 points for the smallest boxes (below 0.25% of the image), +1.1 for the middle tertile and +2.0 for the largest boxes (above 2.6%). On the 800 HR-Bench 4K questions it gains 1.7 points (CI [0.2, 3.2]).

## F.4 Response length

Over both decodes, alignment SFT→RL is shorter than CE SFT→RL throughout the length distribution, not only in the truncated tail (Figure 9). It generates 8.69M tokens against 10.82M (19.7% fewer) and is budget-truncated on 35 instead of

63 responses; in 44 of the 60 pairs where only one model is truncated, the aligned model finishes. Pairs where both finish carry 86.8% of the saving. On V\*, HR-Bench 4K and TreeBench it answers 12, 27 and 10 more questions with 23.4%, 18.0% and 21.3% fewer tokens.

## G All Endpoints

Table 11 lists every thinking-mode evaluation in the paper, with both decodes of each trained model, and Table 12 gives the greedy-decoding evaluation of Appendix F.2.

Table 11: All thinking-mode endpoints (temperature 1, 32K budget) used in the paper. V<sup>∗</sup>, HR-4K and Tree give correct answers out of 191, 800 and 405; Overall is accuracy in % with the count out of 1,396; Tokens is the mean completion length and Tr. the number of budget-truncated responses. Dec. indexes the two independent decodes (1 and 2) of the same checkpoint; HR-Bench rotations of one image are scored as separate questions.
<table><tr><td>Endpoint</td><td>Dec.</td><td>V*</td><td>HR-4K</td><td>Tree</td><td>Overall</td><td>Tokens</td><td>Tr.</td></tr><tr><td>Qwen3.5-4B (base)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-4B (báse)</td><td>1</td><td>162</td><td>672</td><td>201</td><td>74.1 (1035)</td><td>4,678</td><td>37</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Head ablation at inference (base model) Ablate the four localization heads</td><td>1</td><td>132</td><td>546</td><td>164</td><td>60.3 (842)</td><td>11,471</td><td>327</td></tr><tr><td>Ablate four random heads (same layers)</td><td>1</td><td>159</td><td>666</td><td>201</td><td>73.5 (1026)</td><td>3,384</td><td>24</td></tr><tr><td>RL from the base model (1,810-question pool, 100 updates)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>1</td><td>163</td><td>664</td><td>201</td><td>73.6 (1028)</td><td>4,875</td><td>40</td></tr><tr><td>GRPO</td><td>21</td><td>159</td><td>672</td><td>208</td><td>74.4 (1039)</td><td>4,780</td><td>41</td></tr><tr><td>DAPO</td><td></td><td>153</td><td>678</td><td>200</td><td>73.9 (1031)</td><td>3,999</td><td>27</td></tr><tr><td>DAPO</td><td>212</td><td>160</td><td>654</td><td>210</td><td>73.4 (1024)</td><td>4,023</td><td>28</td></tr><tr><td>EASE</td><td></td><td>158</td><td>672</td><td>195</td><td>73.4 (1025)</td><td>5,156</td><td>61</td></tr><tr><td>EASE</td><td></td><td>163</td><td>667</td><td>201</td><td>73.9 (1031)</td><td>5,476</td><td>66</td></tr><tr><td>Training on self-generated trajectories</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CE SFT</td><td>1</td><td>156</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Alignment SFT</td><td></td><td>164</td><td>678</td><td>209</td><td>74.7 (1043)</td><td>3,730</td><td>28</td></tr><tr><td>CE SFT → RL</td><td>1</td><td>159</td><td>683</td><td>215</td><td>76.1 (1062)</td><td>3,940</td><td>32</td></tr><tr><td></td><td>1 2</td><td>163</td><td>680 667</td><td>208 200</td><td>75.0 (1047)</td><td>3,843</td><td>24</td></tr><tr><td>CE SFT → RL</td><td>1</td><td>167</td><td>683</td><td>215</td><td>73.8 (1030) 76.3 (1065)</td><td>3,909</td><td>39 21</td></tr><tr><td>Alignment SFT → RL Alignment SFT → RL</td><td>2</td><td>167</td><td>691</td><td>203</td><td>76.0 (1061)</td><td>3,185 3,039</td><td>14</td></tr><tr><td>LHA with β=0 (control)</td><td>1</td><td>162</td><td>689</td><td>206</td><td>75.7 (1057)</td><td>3,742</td><td>23</td></tr><tr><td>LHA with β=0 (control)</td><td>2</td><td>160</td><td>691</td><td>206</td><td>75.7 (1057)</td><td>3,676</td><td>15</td></tr><tr><td>LHA: alignment SFT → attention RL</td><td></td><td>166</td><td>688</td><td>216</td><td>76.6 (1070)</td><td>3,472</td><td>32</td></tr><tr><td>LHA: alignment SFT → attention RL</td><td>12</td><td>163</td><td>697</td><td>209</td><td>76.6 (1069)</td><td>3,610</td><td>31</td></tr><tr><td>Inference-time methods on the base model</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ViCrop (rel-att, layer 23)</td><td>1</td><td>161</td><td>685</td><td>211</td><td>75.7 (1057)</td><td>4,122</td><td></td></tr><tr><td>ViCrop (rel-att, layer 23)</td><td>2</td><td>163</td><td>682</td><td>216</td><td>76.0 (1061)</td><td></td><td>20</td></tr><tr><td>VCD</td><td>1</td><td>166</td><td>682</td><td>203</td><td>75.3 (1051)</td><td>4,337 4,612</td><td>18</td></tr><tr><td>ZoomEye</td><td></td><td>174</td><td>634</td><td>199</td><td>72.1 (1007)</td><td></td><td>101</td></tr><tr><td>Fixed-grid zoom (2×2 tiles)</td><td></td><td>164</td><td>668</td><td>203</td><td>74.1 (1035)</td><td>5,371 4,725</td><td>92 40</td></tr><tr><td>Zoom-Refine (two passes)</td><td></td><td>165</td><td>658</td><td>201</td><td>73.4 (1024)</td><td>8,232</td><td>13</td></tr><tr><td>Self-Refine</td><td></td><td>160</td><td>645</td><td>203</td><td>72.2 (1008)</td><td>7,994</td><td>6</td></tr><tr><td>Length-aware visual logit bias, β=0.25</td><td></td><td>160</td><td>666</td><td>208</td><td>74.1 (1034)</td><td>4,751</td><td>48</td></tr><tr><td>Length-aware visual logit bias, β=0.5</td><td>1</td><td>155</td><td>670</td><td>206</td><td>73.9 (1031)</td><td>5,013</td><td>42</td></tr><tr><td>Length-aware visual logit bias, β=1</td><td>1</td><td>161</td><td>668</td><td>207</td><td>74.2 (1036)</td><td>5,100</td><td>54</td></tr><tr><td>Head-guided zoom (inference-time crops from attention maps)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base + localization-head crop</td><td></td><td>171</td><td>675</td><td>195</td><td>74.6 (1041)</td><td>4,276</td><td>22</td></tr><tr><td>Alignment SFT → RL + locâlization-head crop</td><td>1</td><td>170</td><td>695</td><td>195</td><td>75.9 (1060)</td><td>2,857</td><td>7</td></tr><tr><td>Alignment SFT → RL + layer-23 head-average crop</td><td></td><td>163</td><td>695</td><td>204</td><td>76.1 (1062)</td><td>2,894</td><td></td></tr><tr><td>Alignment SFT → RL + both crops</td><td>1</td><td>168</td><td>694</td><td>212</td><td>76.9 (1074)</td><td>2,959</td><td>99</td></tr><tr><td>Alignment SFT → RL + both crops</td><td>12</td><td>169</td><td>691</td><td>204</td><td>76.2 (1064)</td><td>2,984</td><td>8</td></tr><tr><td>Released checkpoints (own backbones; raw prompt, single pass, no tools)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ZwZ-4B</td><td></td><td>171</td><td>649</td><td>196</td><td>72.8 (1016)</td><td>185</td><td>0</td></tr><tr><td>ZwZ-7B TreeVGR-7B</td><td>1</td><td>159</td><td>594</td><td>181</td><td>66.9 (934)</td><td></td><td>31 0</td></tr><tr><td></td><td></td><td>159</td><td>586</td><td>187</td><td>66.8 (932)</td><td>60</td><td>0</td></tr><tr><td>DeepEyes-7B</td><td></td><td>136</td><td>587</td><td>159</td><td>63.2 (882)</td><td></td><td>45 0</td></tr><tr><td>VGPO-7B</td><td></td><td>143</td><td>576</td><td>163</td><td>63.2 (882)</td><td>41</td><td>0</td></tr><tr><td>VPPO-7B</td><td></td><td>144</td><td>574</td><td>162</td><td>63.0 (880)</td><td></td><td>56 0</td></tr><tr><td>Reflection-V-7B</td><td></td><td>141</td><td>568</td><td>165</td><td>62.6 (874)</td><td>78</td><td>0</td></tr><tr><td>NoisyRollout-7B Qwen2.5-VL-7B</td><td></td><td>149 149</td><td>565 545</td><td>154 159</td><td>62.2 (868) 61.1 (853)</td><td></td><td>45 0 44 0</td></tr></table>

continued on next page

<table><tr><td>Endpoint</td><td>Dec.</td><td>V*</td><td>HR-4K</td><td>Tree</td><td>Overall</td><td>Tokens</td><td> $\mathrm { T r }$ </td></tr><tr><td>Pixel-Reasoner-7B</td><td>1</td><td>112</td><td>526</td><td>114</td><td>53.9 (752)</td><td>130</td><td>0</td></tr></table>

<table><tr><td>Method</td><td> $\mathbf { V } ^ { * }$ </td><td>HR-4K</td><td>Tree</td><td>Overall</td><td>Tokens</td><td>Trunc. (%)</td><td>∆ vs. base [95% CI]</td></tr><tr><td>Qwen3.5-4B (base)</td><td>78.0</td><td>75.8</td><td>42.7</td><td>66.5</td><td>7,144</td><td>19.1</td><td></td></tr><tr><td>ViCrop (rel-att)</td><td>79.1</td><td>79.1</td><td>41.2</td><td>68.1</td><td>6,498</td><td>16.8</td><td>+1.6 [−0.6, 4.0]</td></tr><tr><td>GRPO</td><td>73.3</td><td>76.5</td><td>42.0</td><td>66.0</td><td>7,737</td><td>20.9</td><td>-0.4 [-2.3, 1.5]</td></tr><tr><td>DAPO</td><td>75.4</td><td>79.0</td><td>44.2</td><td>68.4</td><td>5,830</td><td>14.8</td><td>+1.9[-0.1, 4.0]</td></tr><tr><td>EASE</td><td>73.8</td><td>74.8</td><td>44.0</td><td>65.7</td><td>8,002</td><td>21.8</td><td>-0.8[-2.7, 1.2]</td></tr><tr><td>CE SFT → RL</td><td>80.6</td><td>82.1</td><td>45.7</td><td>71.3</td><td>4,864</td><td>10.4</td><td>+4.9 [2.7, 7.1]</td></tr><tr><td>Alignment SFT → RL</td><td>82.7</td><td>83.5</td><td>49.4</td><td>73.5</td><td>3,691</td><td>6.4</td><td>+7.0 [4.8, 9.3]</td></tr></table>

Table 12: Greedy decoding (temperature 0, same budget and scoring; one decode per model). ∆: paired difference to the base model with 95% CIs. Alignment SFT→RL leads CE SFT→RL by +2.1 points (95% CI [0.4, 4.0]).

## H Full-Width Figures

![](images/f37d880bd02cef85559cc816ec7fc5a2abf3b57c5f8578fec53394a638622bae.jpg)  
Figure 9: The alignment loss shortens the whole length distribution. Survival curves of response length over two decodes; inset: token saving split by which model finishes; dashed line: 32K budget.

![](images/b7d7bb59d8071f2ed1973ff56f44c6ffe94428375ea1d658e7e451600845cc3e.jpg)

![](images/24058bb435dc738b69ace10170ea31897bef556136d5add4e3e41d475f6a093b.jpg)  
Δ accuracy vs. full image (pp)

![](images/f7a4b1f08c0ee004b45049a92e5673b7fc1c69977641d6b65e95e40d8d6c23da.jpg)  
Δ accuracy vs. full image (pp)  
Figure 10: Encoder-side token controls on Qwen2.5-VL-3B (Appendix C.2). Accuracy change relative to the full image, with 95% CIs. Red: evidence given as pixels or post-encoder tokens; grey: input-size and position controls; blue: location cues.

Chain-position quartile  
(a)  
![](images/4ca99de1b1b51c43705663e32a005966caf27d837d9bf8daae9d243a8721402e.jpg)

(b)  
![](images/c1c1a498186651323f1a0aac77a3b9767064007d90c5a0e0a2281cb82e412c3b.jpg)

![](images/ed23ed65635555d4d92dba0dc50ac9c29b9326ac6552d1abc1dc13bc838a1bc1.jpg)  
Figure 11: Visual attention drifts away from the evidence as the chain grows (Appendix C.6; four localization heads, 1,072 same-question rollout pairs). (a) Log share of attention on visual tokens. (b) Image–text logit gap $\Delta ,$ from all and from non-rotary dimensions. (c) Share of visual attention on the evidence box.

![](images/64616f934fe67d769c04aa390b12dd5273999d4bba200b829a56394065b4a7b7.jpg)

![](images/e5b3d4c020209276f7be6373a56bb7f3f12cfac69f7f2152b1577c4adbd500e0.jpg)

![](images/ab785f31d204bd911a784050e9885f48749757a5624efa9460ab054518614232.jpg)  
Figure 12: A few deep heads localize the evidence (Appendix D; Qwen2.5-VL-3B, direct answers). (a) Evidence share of each of the 576 heads on ${ \mathrm { V } } ^ { * } ;$ white boxes mark the five strongest. (b) Strongest head and head mean per layer; dotted: mean box area. (c) Attention of each layer on one ${ \mathrm { V } } ^ { * }$ question (cyan box), relative to uniform; red layers have evidence share ${ \geq } 0 . 1 0$

a  
![](images/f474350b9ac576040f0c72809d74110565255148e1f89d329a7e663ba1658bd8.jpg)

b  
![](images/87d4955b2a7cca5e0c2fed074899365ec9feff13b88e390c322dc774c6bf72d5.jpg)

![](images/e14c1786b7487a00e0746e0ff8035399d40550b6abf46eea1133274477236123.jpg)

![](images/344738ddca792c6b561ad8ef013303d704712b0e1399f99bcd1b63163eecc840.jpg)

![](images/672427b9c3d318c0a8f621d9fe54de4e2c37770cffd2ef1a323c60c8096c9c85.jpg)

Figure 13: TreeBench #181: the crop pulls attention onto the occluding signboard (Qwen2.5-VL-3B). (a) From the full image the model answers D, the yellow dragon flag, and its localization heads put 7.1% of their visual attention on the evidence (green box). (b) With the crop appended it answers A, the signboard, and the share rises to 59.7%. (c, d) Attention mass of the same heads over the input sequence.  
![](images/e024b71002a55555b59217a9540db580035f53ca2f44399e52817ccc71a7c24e.jpg)

b  
![](images/f0034e0548dfe216ef77e056b7793aebc172e54209a154bbd7362d868e98e394.jpg)

![](images/7cc52851dcad4c63c39d136c3e43cc5c3a1f70a7477ee0602e02cb6e1dc1a87a.jpg)

d  
![](images/ecfece37190debb4167237df5574f7f72163cb1b79feb93758016f6453a642d1.jpg)  
Figure 14: RefChartQA #207: the crop isolates the maximal bar. (a) Full image: wrong answer 2019/2020, evidence share 2.8%. (b) With the crop: correct answer 2018/2019, 40.3%. (c, d) As in Figure 13.