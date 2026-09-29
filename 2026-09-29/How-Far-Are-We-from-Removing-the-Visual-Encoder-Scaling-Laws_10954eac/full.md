# How Far Are We from Removing the Visual Encoder? Scaling Laws for Encoder-Free Multimodal Pretraining

Lin Chen<sup>1,2,3∗</sup> Bolin Ni<sup>3†</sup> Qi Yang<sup>3</sup> Lan Jiang<sup>3</sup> Kun Ding<sup>1</sup> Xiaoran Fan<sup>3</sup> Hower Yang<sup>3</sup> Ying Wang<sup>1</sup> Shiming Xiang<sup>1,2</sup>

<sup>1</sup>CASIA <sup>2</sup>UCAS <sup>3</sup>Foundation Model Department, Tencent

## Abstract

Most modern multimodal large language models (MLLMs) build on a pretrained visual encoder that provides a strong visual prior. Encoder-free MLLMs instead learn visual representations directly from raw pixels, offering a simple and unified architecture, but their scaling behavior has not been systematically characterized. To fill this gap, we compare scaling laws for encoder-free and encoder-based MLLMs and report three main findings: (1) Removing the visual encoder shifts the compute-optimal allocation for the multimodal objective toward larger models, while leaving that for text nearly unchanged. (2) The two architectures exhibit nearly overlapping loss–compute frontiers on the text objective, but diverge on the multimodal objective: encoder-free models underperform at small scales yet are predicted to catch up at around 10<sup>22</sup> FLOPs, well within practical pretraining budgets. (3) Without a visual encoder, the language model learns to take over its role via vision-specific adaptation: bidirectional interactions among visual tokens become increasingly beneficial as training compute grows, visual processing shifts toward earlier layers, and expert routing for visual tokens becomes more concentrated. Overall, our results indicate that the advantage of the visual prior provided by a pretrained encoder diminishes with scale, positioning encoder-free architectures as a promising direction for multimodal pretraining.

## 1 Introduction

Most modern multimodal large language models (MLLMs) [2, 22, 37] adopt an encoder-based architecture: a pretrained visual encoder [47, 58] supplies the language model with semantically rich visual representations, providing a strong visual prior learned from large-scale image–text data. To achieve a simple and unified architecture, encoder-free MLLMs remove the visual encoder and feed projected image patches directly into the decoder, which must then learn visual representations from raw pixels [3, 9, 13, 15, 23, 30, 39, 54]. Although these studies show initial feasibility, the scaling behavior of encoder-free MLLMs has not been systematically characterized.

To this end, we conduct a controlled scaling study of encoder-free and encoder-based MLLMs, in which the two model families share the same sparse decoder ladder, data mixture, optimization setup, and visual-token granularity. We fit scaling laws separately for the text and multimodal objectives to quantify how the efficiency gap between the two architectures evolves with scale. To understand the mechanisms behind these trends, we further probe the decoder’s internals and examine how it compensates for the missing visual encoder. Our main findings are as follows.

(1) Compute-optimal encoder-free training favors larger models (§3.1). For the text objective, the two architectures exhibit nearly identical compute-optimal allocation trends. In contrast, the multimodal objective shows a different pattern: removing the visual encoder increases the model allocation exponent from a = 0.464 to a = 0.570, shifting the optimum toward larger model scale. This shift suggests that encoder-free models require greater decoder capacity to jointly support visual representation learning and language modeling.

(2) Encoder-free models are predicted to catch up within practical pretraining budgets (§3.2). As shown in Fig. 1 (Left), the two architectures exhibit nearly overlapping loss–compute frontiers for the text objective. In contrast, Fig. 1 (Right) shows that encoder-free models require more training compute than encoder-based models to reach the same validation loss on the multimodal objective. Nevertheless, their loss decreases more rapidly with compute, narrowing the gap at scale. Extrapolating the fitted scaling laws beyond our measured range predicts that the multimodal crossover occurs on the order of $1 \bar { 0 ^ { 2 2 } }$ FLOPs under compute-optimal allocation, and at a higher compute budget under 5× overtraining. For reference, the pretraining compute of recent flagship models, such as Kimi K2.5 [29], is approximately $1 0 ^ { 2 5 }$ FLOPs.<sup>1</sup> Furthermore, fits on individual multimodal topics show that the crossover varies by topic, arriving earlier on topics that rely mainly on language and much later on perception-intensive ones.

![](images/99bff11f510da48e1cdd86b806653d95fab51e4784830253b96532662eeb6fc8.jpg)

![](images/9ce0706c2ccf9feb7a0848f64083df7bf277c7d0c4e8ba6f115141978a9304b4.jpg)  
Figure 1: Compute-optimal loss frontiers for encoder-free and encoder-based models. Left: The text loss frontiers of the two architectures nearly overlap. Right: Encoder-free models have higher multimodal loss over the measured range, but their loss decreases faster with compute.

(3) The decoder takes over visual encoding via vision-specific adaptation (§3.3). The decoder increasingly relies on bidirectional attention among visual tokens, recovering the patch-level contextualization that a visual encoder would otherwise provide. Moreover, visual token representations diverge from their inputs much earlier than in encoder-based models, so the shallow decoder layers effectively serve as an implicit visual encoding stage, whereas text tokens are processed almost identically in both architectures. This vision-specific adaptation further extends to the MoE experts, where the routing of visual tokens becomes more concentrated, consistent with some experts taking over the vision-specific role of the visual encoder. These adaptations suggest that encoder-free models may benefit from decoder architectures designed explicitly for native visual representation learning, rather than directly inheriting designs built for language.

Overall, encoder-free models require more training compute within the fitted range, but their more rapidly improving multimodal frontier predicts an efficiency crossover within practical pretraining budgets. These findings position encoder-free architectures as a promising direction, and we expect this work to encourage broader exploration of encoder-free multimodal pretraining.

## 2 Preliminaries

## 2.1 Estimating Scaling Laws

Problem Definition. Let M denote FLOPs per token [5] and D the number of objective tokens, so the training budget is $C = M D$ . At a target budget C, the compute-optimal allocation [25, 27] for objective $\breve { \mathcal { L } }$ is the feasible point on $C = \bar { M } D$ that minimizes loss:

$$
M _ { \mathrm { o p t } } ( C ) , D _ { \mathrm { o p t } } ( C ) = \underset { M , D } { \arg \operatorname* { m i n } } \mathcal { L } ( M , D ) \quad \mathrm { s . t . } \quad M D = C .\tag{1}
$$

Across budgets, these optima follow the compute-optimal allocation law:

$$
M _ { \mathrm { o p t } } ( C ) \propto C ^ { a } , \qquad D _ { \mathrm { o p t } } ( C ) \propto C ^ { b } , \qquad a + b = 1 .\tag{2}
$$

The corresponding compute-optimal frontiers follow:

$$
\begin{array} { r } { \mathcal { L } ^ { * } ( C ) = E + K C ^ { - \gamma } , } \end{array}\tag{3}
$$

where $\gamma > 0$ is the loss–compute exponent, $K > 0$ is a fitted prefactor, and E is the entropy floor induced by the data distribution [25].

IsoFLOP Profiles. Following Chinchilla [25], an IsoFLOP profile at budget C is obtained by varying M, setting $D = C / M$ , and fitting validation loss as a quadratic function of log M. The fitted minimum defines $\dot { M } _ { \mathrm { o p t } } ( C )$ . The corresponding $D _ { \mathrm { o p t } } ( C ) \dot { = } C / M _ { \mathrm { o p t } } ( C )$ and $\mathcal { L } ^ { * } \bar { ( } C )$ then follow directly. Repeating this procedure across budgets provides the optima used to estimate the allocation law and the compute-optimal frontiers.

## 2.2 Efficiency Gain

Following MAI-Thinking-1 [45], let λ be a loss reachable by both systems and let $C _ { s } ( \lambda )$ denote the actual training compute required by system $s \in \{ \mathrm { t a r } , \mathrm { r e f } \}$ to reach it under the training regime being compared. The compute efficiency gain is

$$
\mathrm { E G } _ { \mathrm { t a r }  \mathrm { r e f } } ^ { C } ( \lambda ) = C _ { \mathrm { r e f } } ( \lambda ) / C _ { \mathrm { t a r } } ( \lambda ) .\tag{4}
$$

Further, let $M _ { s } ( \lambda )$ denote the FLOPs per token of the model actually used by system s at this point of equal loss. The model efficiency gain is

$$
\mathrm { E G } _ { \mathrm { t a r }  \mathrm { r e f } } ^ { M } ( \lambda ) = M _ { \mathrm { r e f } } ( \lambda ) / M _ { \mathrm { t a r } } ( \lambda ) .\tag{5}
$$

For both metrics, values above 1.0 indicate that the target is more efficient than the reference: the target requires less training compute for $\mathrm { E G } ^ { C }$ or fewer FLOPs per token for $\mathrm { E G } ^ { M }$ to reach λ.

## 2.3 Model Ladder

We compare encoder-free and encoder-based MLLMs on a matched ladder of 11 sparse MoE language models with 1.1B–44B total and 71M– 2.4B active non-embedding parameters, sharing the same data mixture, optimization setup, and visual-token granularity. As shown in Fig. 2, the encoder-based model encodes images with a pretrained SigLIP 2 ViT [58], followed by a ConvPool adapter and a projector, and applies causal attention to all tokens. Following common practice [2, 22], the ViT keeps the same

![](images/a182c2896bb2ea9265a39e9dba64780757fc4459f5d4231eb784c4e5ef190d87.jpg)  
Figure 2: Architectural comparison of encoderbased and encoder-free MLLMs.

size across decoder scales and is trained jointly with the decoder. The encoder-free model instead maps raw image patches into the decoder through a patch projection [23], where visual tokens attend bidirectionally within each image [15, 19, 23, 39] and all other attention remains causal. We also study a fully causal variant. More implementation details are provided in Appendix A.

## 2.4 Compute Accounting

The decoder FLOPs per token can be decomposed into a term from matrix multiplications applied to each token, which is independent of sequence length, and a term from self-attention, which grows with sequence length:

$$
M _ { o } ^ { ( s ) } = M _ { \mathrm { b a s e } } + M _ { \mathrm { a t t n } } ^ { ( s ) } \ell _ { o } , \qquad C _ { o } ^ { ( s ) } = M _ { o } ^ { ( s ) } D _ { o } ,\tag{6}
$$

where $s \in \{ \mathrm { f r e e } , \mathrm { b a s e d } \}$ indexes the model family and $o \in \{ \mathrm { t e x t } , \mathrm { m m } \}$ indexes the objective. Here, $D _ { o }$ counts all tokens processed in batches for objective $o \left( D _ { \mathrm { m m } } \right.$ includes both visual and text tokens). The mean packed length $\ell _ { o }$ depends on the objective, while $M _ { \mathrm { a t t n } } ^ { ( s ) }$ differs between the two families because they use different attention patterns over visual tokens. The fixed expert activation ratio of 8/256 keeps M approximately proportional to the number of active parameters.

Since we vary decoder scale with a fixed visual encoder, we use decoder FLOPs as the primary compute measure, focusing the analysis on the tradeoff between decoder capacity and training tokens. Appendix C shows that including visual encoder FLOPs leaves the main conclusions unchanged and further strengthens the relative efficiency of encoder-free models.

![](images/514f55ba927a02887afa4765fabb794f679c50ee23592c10be6e556dda1380e7.jpg)  
Figure 3: Estimating compute-optimal allocation with IsoFLOP profiles for the text objective (left) and the multimodal objective (right).

## 3 Scaling Laws for Encoder-Free Multimodal Pretraining

In this section, we explore how removing the pretrained visual encoder changes the scaling behavior of the language model. §3.1 first examines how encoder removal changes compute-optimal allocation. §3.2 then uses the resulting allocation laws to test whether encoder-free models can catch up at scale, under both compute-optimal allocation and overtraining. Finally, §3.3 examines how the decoder takes over visual encoding through vision-specific adaptation.

## 3.1 How Does Encoder Removal Change Compute-Optimal Allocation?

We estimate compute-optimal allocation for both encoder-free and encoder-based models using the IsoFLOP profiles described in §2.1, with the results shown in Fig. 3. At each budget for each model, the best point in the profile gives $M _ { \mathrm { o p t } }$ and $D _ { \mathrm { o p t } }$ , whose scaling with C yields the allocation exponents a and b.

On the text objective, the two architectures have nearly identical model allocation exponents $( a ~ = ~ 0 . 4 2 7$ for encoder-free and $a \ = \ 0 . 4 2 2$ for encoder-based models), which is expected. On the multimodal objective, however, removing the encoder increases a from 0.464 to 0.570, indicating that compute-optimal training allocates more compute to model scale. For these multimodal estimates, a bootstrap gives central 80% intervals

Table 1: Model allocation exponent a of encoder-free models under bidirectional and causal attention over visual tokens.
<table><tr><td>Attention</td><td>Text</td><td>Multimodal</td></tr><tr><td>Bidirectional</td><td>0.427</td><td>0.570</td></tr><tr><td>Causal</td><td>0.436</td><td>0.557</td></tr></table>

of [0.546, 0.595] and [0.458, 0.472] for the encoder-free and encoder-based models, respectively (Appendix B.4). The shift persists under causal attention over visual tokens, which gives $a = 0 . 5 5 7$ (Tab. 1; Appendix E.2), suggesting that it is not an artifact of the attention mask. Together, these results suggest that removing the visual encoder increases the decoder’s representational burden and favors larger models.

Takeaway 1. Removing the visual encoder leaves the compute-optimal allocation on the text objective unchanged, but shifts multimodal training toward larger models (Fig. 3).

## 3.2 Can Encoder-Free Models Catch Up at Scale?

After estimating compute-optimal allocation, we explore how much compute and decoder scale encoder-free models need to match encoder-based models at equal validation loss. For all metrics in this subsection, encoder-free is the target and encoder-based is the reference. We report two ratios. $\mathrm { E G } ^ { C }$ compares the training compute required at equal loss. Because the ratio is reference over target, values below 1.0 mean encoder-free models need more compute. $\mathrm { E G } ^ { M }$ compares the FLOPs per token selected at the point of equal loss. Values below 1.0 mean the encoder-free model at equal loss uses a larger decoder. We first evaluate both ratios on the compute-optimal frontier, then examine how they shift under overtraining.

![](images/b600f91e5e6f442131c1e0a320cfe1cf89178856b2c62e745fb9d375b1002299.jpg)  
Figure 4: Efficiency gain analysis on the compute-optimal frontier for the text (top) and multimodal (bottom) objectives. (A) fitted loss–compute scaling laws, (B) compute efficiency gain $( \mathrm { E G } ^ { C } )$ , (C) model efficiency gain $( \mathrm { E G } ^ { M } )$ .

## 3.2.1 Efficiency Gain Under Compute-Optimal Allocation

We first compare the two systems on their respective compute-optimal frontiers, where each training budget is allocated between model scale and tokens to minimize loss. Fixing a target loss then determines both the required compute and the corresponding optimal FLOPs per token. Fig. 4 summarizes the fitted loss–compute scaling laws and the resulting efficiency gains.

Text Objective. On $\mathcal { L } _ { \mathrm { t e x t } }$ , the two fitted loss curves are almost indistinguishable. Their loss–compute exponents are also nearly identical (0.0973 versus 0.0979). The fitted and measured $\mathrm { E G } ^ { C }$ values stay around 0.98, and the fitted $\mathrm { E G } ^ { C }$ curve is nearly flat. $\mathrm { E G } ^ { M }$ at equal loss also stays close to 1.0, around 0.99, with only a slight downward drift across the fitted range. Both deviations remain smal and nearly constant across the fitted range, so text acts as a nearly matched control rather than a regime with a meaningful encoder-free penalty.

Multimodal Objective. On ${ \mathcal { L } } _ { \mathrm { m m } } .$ , encoder-free models require more training compute to attain the same loss throughout the measured range. However, this gap narrows with scale: the fitted $\mathrm { E G } ^ { C }$ increases because the encoder-free loss decreases more rapidly with training compute. At equal loss, encoder-free models also favor a larger decoder, with $\mathrm { E G } ^ { \bar { M } } \approx 0 . 8 0$ , corresponding to approximately 1.25× the FLOPs per token. Extrapolating the fitted scaling laws places the efficiency crossover on the order of $1 0 ^ { \dot { 2 } 2 }$ FLOPs (Fig. 4A, bottom). The point estimate is $6 . 1 \times 1 0 ^ { 2 1 }$ FLOPs, with a conditional bootstrap 80% interval of $[ 4 . 2 \times \mathrm { { 1 0 ^ { 2 1 } } , 1 . \dot { 0 } \times 1 0 ^ { 2 2 } } ]$ (Appendix B.4). This projection assumes that the fitted laws persist beyond the measured range, with the visual encoder held at a fixed size and the irreducible loss determined only by the data distribution. Even accounting for this uncertainty, the crossover remains roughly three orders of magnitude below the pretraining compute of recent flagship models, e.g., approximately $1 0 ^ { 2 5 }$ FLOPs for Kimi K2.5 [29].

## 3.2.2 Efficiency Gain Under Overtraining

The compute-optimal frontier is not the only practical regime, since deployment models are often overtrained by spending extra tokens at a fixed model scale to reduce inference cost at a target quality [48]. We therefore test whether overtraining changes the efficiency gain. Starting from the compute-optimal point at base budget $C _ { \mathrm { b a s e } } ,$ , overtraining keeps the model scale fixed and trains on $k$ times as many tokens, where k is the overtraining factor, so the actual training compute is $C _ { \mathrm { a c t u a l } } = k C _ { \mathrm { b a s e } } .$

![](images/b79fc09593ac67c28124e80165405d1791c333076279d4df58c60164359901cf.jpg)  
Figure 5: Efficiency gain analysis under overtraining for the text (top) and multimodal (bottom) objectives, with shades denoting overtraining factors $k = 1 { - } 5$ from darkest to lightest. (A) loss– compute scaling laws, (B) compute efficiency gain $( \mathrm { E G } ^ { C } )$ , (C) model efficiency gain $( \mathrm { E G } ^ { M } )$

We model overtraining as a shift in the prefactor of the loss–compute law, following prior scaling analyses [21]. Under the separable form $\mathcal { L } ( M , D ) = E + A M ^ { \dot { - } \alpha } + B D ^ { - \beta }$ with $\bar { C _ { \mathrm { b a s e } } } = M \bar { D }$ the compute-optimal frontier is $\mathcal { L } ^ { * } ( C _ { \mathrm { b a s e } } ) = E + K C _ { \mathrm { b a s e } } ^ { - \gamma }$ with $\gamma = \alpha \beta / ( \alpha + \beta )$ . Fixing $M =$ $M _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } )$ and setting $D = k D _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } )$ gives

$$
\begin{array} { r } { \mathcal { L } ( C _ { \mathrm { b a s e } } , k ) = E + g ( k ) K C _ { \mathrm { b a s e } } ^ { - \gamma } , } \end{array}\tag{7}
$$

where the multiplier $g ( k )$ depends on k but not on $C _ { \mathrm { b a s e } } .$ . Overtraining therefore leaves E and $\gamma$ unchanged and rescales only the reducible term (derivation in Appendix D). In practice, we reuse $\dot { E } ,$ $K ,$ , and $\gamma$ from the compute-optimal fit and estimate only an empirical $g ( k )$ for each k (Appendix $\mathrm { A } . 3 )$ which requires far fewer runs and lets us run overtraining experiments at smaller model scales.

Fig. 5 shows how overtraining changes the comparison at equal loss. On the multimodal objective, the shift is asymmetric: the same overtraining factor k changes the two systems’ prefactors by different amounts because their data exponents and the optimal split between loss terms differ. $\mathrm { A t } k = 5 ,$ , the extrapolated crossover remains on the order of $1 0 ^ { 2 2 }$ FLOPs but arrives later than under computeoptimal allocation. The point estimate is $1 . 2 \times 1 0 ^ { 2 2 }$ FLOPs, with a conditional bootstrap 80% interval of $[ 8 . 4 \times 1 0 ^ { 2 1 } , 2 . 0 \times \dot { 1 0 } ^ { 2 2 } ]$ (Appendix B.4). At the largest fitted budget, overtraining also lowers $\mathrm { E G } ^ { \bar { C } }$ from 0.62 to 0.52 and $\mathrm { E G } ^ { \bar { M } }$ from 0.80 to 0.74. This is consistent with the allocation shift in $\ S 3 . 1 \colon$ since encoder-free models favor larger decoders on multimodal data, spending extra compute on tokens at a fixed model scale benefits them less. By contrast, encoder-free and encoder-based models remain nearly matched on the text objective under overtraining, with both ratios changing by less than 1% from their $k = 1$ values.

Takeaway 2. Both architectures nearly overlap on text loss frontiers, while encoder-free models initially lag on multimodal loss but are predicted to catch up within practical budgets (Figs. 4–5).

## 3.2.3 Analysis by Topic

The aggregate multimodal loss summarizes the overall trend but obscures variation across topics. We therefore repeat the analysis at equal loss on each major multimodal topic (Fig. 6). Appendix F reports the underlying IsoFLOP profiles.

![](images/f1b02c5c8fe7cfd46073a45603ed681201ca7f1e82de1625113b61df621f0b18.jpg)

![](images/c9cfd49b735cbd57af21af536ad1cb1007b5f19ddea8c9ae221335f50a6993ff.jpg)

![](images/1e53e3807684516ea76b240fdb44c4b3fc3c9063ec30cc88e600d5c45e15dbc5.jpg)  
Training Compute, C (FLOPs)

![](images/b5404b94f80530cfc422d003a6b056daef3bddfd08df43f8c9a5af9e352e2132.jpg)

![](images/8ad91e0f8a61f779c50c05084365364b7b398379670ac24cc1550add0592707c.jpg)  
Figure 6: Compute efficiency gain on each multimodal topic. Curves from dark to light correspond to overtraining factors $k = 1 { - } 5$ Solid segments span the fitted budgets, and dashed segments extrapolate the fitted laws.

(A) Encoder-free  
![](images/cfc7a1513893a7b528d65427de8fa5909a637ab2b4c0b55cad8f7fcb26e96aae.jpg)  
Total training tokens (B)

(B) Encoder-based  
![](images/0dd704b62e5832c8de07a9cb347b3ae6f60e68a28c7457497338dc13cce91486.jpg)  
Total training tokens (B)

(C) Visual attention  
![](images/4c9516798a35921b0b0a6fc2f6374ca8d1b16ba416c4623e21bbf00b08df38e4.jpg)  
Decoder layer  
Figure 7: Visual learning across training tokens. (A, B) Multimodal validation loss during training. (C) Attention mass on visual tokens per decoder layer for the 8B models, before and after the encoder-free loss drop (shaded in A, B).

Within our compute range, encoder-free models remain less compute-efficient than encoder-based models on every topic, yet topics differ markedly in the size of the remaining gap and how quickly it narrows with scale. Extrapolating the fitted laws, encoder-free models catch up first on STEM, which is already close to parity at the largest fitted budget, then on Charts, whose gap narrows quickly, and considerably later on GUI, OCR, and Caption. This ordering is consistent with how strongly each topic relies on pretrained visual representations. The STEM subset consists mainly of text, symbols, and simple diagrams, and chart inputs can often be reduced to symbolic content. Once this content is extracted, prediction depends mainly on language and reasoning. By contrast, captioning requires rich representations of natural images, while GUI and OCR demand detailed spatial and textual perception, for which a pretrained encoder provides a strong prior that encoder-free models must learn from scratch.

Takeaway 3. The crossover varies by topic, arriving earlier on topics that rely mainly on language (e.g., STEM) and much later on perception-intensive ones (e.g., Caption) (Fig. 6).

## 3.3 How Does the Decoder Take Over Visual Encoding?

Removing the visual encoder shifts visual representation learning into the decoder. To trace this shift, we first compare how the multimodal loss of encoder-free and encoder-based models evolves over training tokens, and then examine three probes inside the decoder: attention over visual tokens, layerwise evolution of visual representations, and expert routing.

Emergence of Visual Encoding During Training. The multimodal loss of encoder-based models decreases smoothly, whereas that of encoder-free models decreases slowly at first, then drops sharply within a short span (Fig. 7A). To understand this drop, we compare attention to visual tokens in the 8B models before and after it: at layer 12, the encoder-free model’s attention to visual tokens rises from 0.217 to 0.645, approaching that of the encoder-based model (Fig. 7C). We hypothesize that the slow early phase reflects the decoder bootstrapping its own visual representations. Since the loss is applied only to text tokens, visual tokens receive learning signal only when text attends to them. Initially uninformative and thus ignored, they learn slowly until they become useful enough to attract attention, after which learning accelerates. A pretrained encoder supplies useful visual representations from the start and thus avoids this stage, consistent with the high attention to visual tokens in the encoder-based model at both checkpoints. At every multimodal IsoFLOP budget, the compute-optimal models have already passed this drop, so the fitted frontier reflects the smooth regime after it.

Encoder-Like Contextualization: Bidirectional Attention. We compare bidirectional and causal attention over visual tokens by rerunning the encoder-free ladder with the causal variant (Fig. 8). Causal attention is slightly better on the text objective at the measured budgets, reaching the same loss for about 1% less compute $( \mathrm { E G } ^ { C } = 1 . 0 1 1 )$ , but this gain shrinks with compute. On the multimodal objective, causal attention is mildly worse on average $( \mathrm { E G } ^ { C } = 0 . 9 9 0 )$ and the multimodal gain of bidirectional attention becomes larger with compute. This pattern is consistent with a transfer of function from encoder to decoder. In encoder-based models, the

![](images/53252321391f0f9e0467198ebb4342c09a8857f53dd8a1b867fef712231cece8.jpg)

![](images/b91d5468ba23c6ae797f7db2c0a26ff1d2cc6c0a2ce531fadb3b5b021a064b4e.jpg)  
Figure 8: Comparison between causal and bidirectional attention on visual tokens. Values above 1.0 favor causal.  
Training Compute, C (FLOPs)

ViT bidirectionally contextualizes image patches before they reach the decoder. Once the ViT is removed, bidirectional attention among visual tokens allows the decoder to assume part of this role. The increasing multimodal benefit of bidirectional attention with scale suggests that larger decoders exploit these interactions better, while its diminishing cost on text indicates limited interference with language modeling.

Encoder-Like Early Processing: Layerwise Representation Evolution. In encoder-based MLLMs, the ViT has already transformed visual tokens into semantic representations, so they undergo little additional processing in shallow decoder layers [18]. If the decoder takes over this transformation, its shallow layers should instead rewrite visual tokens substantially. Fig. 9 tests this using cosine similarity between each layer’s token representations and their layer-0 inputs. Without the visual encoder, visual tokens move away from their inputs much earlier (left), while text token trajectories remain close across

![](images/a2b743173049977232aaa0c197027c4463f7f0a19d350895d013fcd1c69ab1bb.jpg)

![](images/b3b80027a3ef45ce8249678ed7a4e4fdc0a7ac17f45fe84d09f3433c0d04f7f0.jpg)  
Figure 9: Layerwise representation evolution. Cosine similarity to the input representation at layer 0 across decoder layers for visual tokens (left) and text tokens (right).

the two systems (right). The same pattern holds across model scales (Appendix E.1). The shallow decoder layers thus act as an implicit visual encoding stage, performing the transformation that the ViT performs in encoder-based models, and this change is specific to visual tokens.

Encoder-Like Dedicated Capacity: Expert Routing. Both systems use the same sparse decoder, so routing differences show how the decoder absorbs the changed visual representations. We quantify expert load imbalance using MaxVio [60], the relative excess of the most-loaded expert’s load over the perfectly balanced load. As shown in Fig. 10, both systems have similarly low overall MaxVio when visual and text tokens are aggregated, although the encoder-free values are slightly higher. Separating tokens by modality reveals substantially greater expert load imbalance for both visual and text tokens than the aggregate suggests. For text tokens, the two architectures remain closely matched. In contrast, across the four largest model sizes, encoder-free models exhibit consistently higher average MaxVio and a wider band for visual tokens throughout training. The close match on text argues against a routing shift across the whole model and localizes the effect to visual processing. This concentration is consistent with the decoder allocating a subset of its experts to play the role of the vision-specific parameters that the ViT previously provided.

Takeaway 4. The decoder adapts to take over visual encoding: bidirectional attention among visual tokens, shallow-layer visual processing, and concentrated expert routing (Figs. 8–10).

![](images/2d7ab825c86103d9519edbf88cc72bc7f0a00d2e499510a9c219f2b4d5dcbdb7.jpg)

![](images/afa144f34e98c820f7db419340ebd5c95cb27cfc273b79d7d48a04bcc970c94a.jpg)

![](images/8d303e44efc025a8ea594415dd19bc38df7e5b39ae43a3af9da95dbbc28de987.jpg)  
Figure 10: Expert load imbalance over training. MaxVio is computed jointly over visual and text tokens (left), over visual tokens only (middle), and over text tokens only (right). Higher values indicate greater imbalance.

## 4 Related Work

Encoder-Free MLLMs. Encoder-free MLLMs remove the visual encoder and pass projected image patches directly into the decoder. Early systems such as Fuyu, EVE, and SOLO established the feasibility of this design [3, 9, 13]. Moving beyond feasibility, SAIL systematically studies model and data scalability, cross-modal information flow, and visual representation learning within a single Transformer [30]. Later work improves visual competence through alignment or distillation [35, 53, 59, 65], additional capacity dedicated to each modality [14, 41, 42], native vision-language primitives [15], and extensions to video [16, 32, 66], 3D [52], segmentation [68], and unified understanding and generation [7, 17, 31, 33, 39, 62]. Recent large systems such as the Gemma 4 12B Unified model [23] and Inkling [54] also explore native multimodal input without a visual encoder. Our work complements these efforts by comparing behavior between encoder-free and encoder-based MLLMs.

Scaling Laws. Predicting training behavior at large scale from small runs is now standard practice, spanning loss trends, data scaling, compute-optimal allocation, and training hyperparameters [1, 5, 6, 11, 25, 27, 34, 61]. Recent work extends this methodology to native multimodal pretraining. In particular, the compute-optimal data requirement grows faster for vision than for language in unified multimodal pretraining [57], and native MLLMs trained under data constraints exhibit coupled scaling between the visual encoder and the language model [55]. Others compare early and late fusion trained from scratch [49] or study how the data mixture affects the scaling of encoder-free MoE models [63]. In contrast, we compare encoder-free models against an encoder-based baseline on a shared MoE decoder ladder, predict the compute budget at which encoder-free models catch up under both compute-optimal allocation and overtraining, and analyze how the decoder takes over visual encoding via vision-specific adaptation.

## 5 Conclusion

Encoder-free MLLMs are less compute-efficient on the multimodal objective at the scales we evaluate. However, their compute-optimal loss decreases faster as compute increases. Our fitted scaling laws predict a crossover on the order of $1 0 ^ { 2 2 }$ FLOPs under compute-optimal allocation and at a higher budget under $5 \times$ overtraining. Encoder-free scaling also favors larger decoders, delays the decoder’s reliance on visual input, strengthens interactions among visual tokens, shifts visual representation transformation earlier, and concentrates expert routing. In contrast, text scaling remains largely unchanged. These findings point to future work on decoder architectures and training strategies that improve the compute efficiency of native visual input.

## Acknowledgments

We would like to thank Yan Fang for many fruitful and insightful discussions throughout the course of this work.

## References

[1] Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws in language and vision. Advances in Neural Information Processing Systems, 35:22300– 22312, 2022.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sagnak Ta¸sırlar. Introducing our multimodal models, 2023. URL˘ https://www.adept. ai/blog/fuyu-8b.

[4] Tamay Besiroglu, Ege Erdil, Matthew Barnett, and Josh You. Chinchilla scaling: A replication attempt. arXiv preprint arXiv:2404.10102, 2024.

[5] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. DeepSeek LLM: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

[6] Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, and Xia Song. Scaling optimal LR across token horizons. In International Conference on Learning Representations, volume 2025, pages 83640–83657, 2025.

[7] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

[8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024.

[9] Yangyi Chen, Xingyao Wang, Hao Peng, and Heng Ji. Solo: A single transformer for scalable vision-language modeling. arXiv preprint arXiv:2407.06438, 2024.

[10] Leshem Choshen, Yang Zhang, and Jacob Andreas. A hitchhiker’s guide to scaling law estimation. arXiv preprint arXiv:2410.11840, 2024.

[11] Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International Conference on Machine Learning, pages 4057–4086. PMLR, 2022.

[12] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1280–1297, 2024.

[13] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. Advances in Neural Information Processing Systems, 37:52545–52567, 2024.

[14] Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, and Xinlong Wang. EVEv2: Improved baselines for encoder-free vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21014–21025. IEEE, 2025.

[15] Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, and Ziwei Liu. From pixels to words–towards native vision-language primitives at scale. In International Conference on Learning Representations, volume 2026, pages 109909– 109929, 2026.

[16] Haiwen Diao, Jiahao Wang, Penghao Wu, Yuhao Dong, Yuwei Niu, Yue Zhu, Zhongang Cai, Weichen Fan, Linjun Dai, Silei Wu, et al. From pixels to words–towards native one-vision models at scale. arXiv preprint arXiv:2605.28820, 2026.

[17] Haiwen Diao, Penghao Wu, Hanming Deng, Jiahao Wang, Shihao Bai, Silei Wu, Weichen Fan, Wenjie Ye, Wenwen Tong, Xiangyu Fan, et al. SenseNova-U1: Unifying multimodal understanding and generation with NEO-unify architecture. arXiv preprint arXiv:2605.12500, 2026.

[18] Yingqi Fan, Junlong Tong, Anhao Zhao, and Xiaoyu Shen. What do visual tokens really encode? Uncovering sparsity and redundancy in multimodal large language models. arXiv preprint arXiv:2603.00510, 2026.

[19] Yan Fang, Mengcheng Lan, Zilong Huang, Weixian Lei, Yunqing Zhao, Yujie Zhong, Yingchen Yu, Qi She, Yao Zhao, and Yunchao Wei. Let ViT speak: Generative language-image pretraining. arXiv preprint arXiv:2605.00809, 2026.

[20] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. MME: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2026.

[21] Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, et al. Language models scale reliably with over-training and on downstream tasks. In International Conference on Learning Representations, volume 2025, pages 67661–67682, 2025.

[22] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

[23] Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4˘ technical report. arXiv preprint arXiv:2607.02770, 2026.

[24] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.

[25] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

[26] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon/.

[27] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[28] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision, pages 235–251. Springer, 2016.

[29] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. Kimi K2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[30] Weixian Lei, Jiacong Wang, Haochen Wang, Xiangtai Li, Jun Hao Liew, Jiashi Feng, and Zilong Huang. The scalability of simplicity: Empirical analysis of vision-language learning with a single transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20758–20769. IEEE, 2025.

[31] Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. OneCAT: Decoder-only auto-regressive model for unified understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 30235–30245, 2026.

[32] Handong Li, Yiyuan Zhang, Longteng Guo, Xiangyu Yue, and Jing Liu. Breaking the encoder barrier for seamless video-language understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23167–23176. IEEE, 2025.

[33] Hao Li, Changyao Tian, Jie Shao, Xizhou Zhu, Zhaokai Wang, Jinguo Zhu, Wenhan Dou, Xiaogang Wang, Hongsheng Li, Lewei Lu, et al. SynerGen-VL: Towards synergistic image understanding and generation with vision experts and token folding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29767–29779. IEEE, 2025.

[34] Houyi Li, Wenzhen Zheng, Qiufeng Wang, Hanshan Zhang, Zili Wang, Shijie Xuyang, Yuantao Fan, Zhenyu Ding, Haoying Wang, Ning Ding, et al. Predictable scale: Part I, Step Law– optimal hyperparameter scaling law in large language model pretraining. arXiv preprint arXiv:2503.04715, 2025.

[35] Tianle Li, Yongming Rao, Winston Hu, and Yu Cheng. BREEN: Bridge data-efficient encoderfree multimodal learning with learnable queries. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5384–5395. IEEE, 2026.

[36] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, 2023.

[37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

[38] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2024.

[39] Zhiheng Liu, Weiming Ren, Xiaoke Huang, Shoufa Chen, Tianhong Li, Mengzhao Chen, Yatai Ji, Sen He, Jonas Schult, Tao Xiang, Wenhu Chen, Ping Luo, Luke Zettlemoyer, and Yuren Cong. TUNA-2: Pixel embeddings beat vision encoders for unified understanding and generation. arXiv preprint arXiv:2604.24763, 2026.

[40] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521, 2022.

[41] Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, et al. Mono-InternVL-1.5: Towards cheaper and faster monolithic multimodal large language models. arXiv preprint arXiv:2507.12566, 2025.

[42] Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jiawen Liu, Jifeng Dai, Yu Qiao, and Xizhou Zhu. Mono-InternVL: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24960–24971. IEEE, 2025.

[43] Ahmed Masry, Jia Qing Tan, Shafiq Joty, Enamul Hoque, et al. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofthe Association for Computational Linguistics: ACL 2022, pages 2263–2279, 2022.

[44] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A dataset for VQA on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2199–2208. IEEE, 2021.

[45] Microsoft AI. MAI-Thinking-1: Building a hill-climbing machine. https://microsoft.ai/ pdf/mai-thinking-1.pdf, 2026.

[46] Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, and Yair Carmon. Resolving discrepancies in compute-optimal scaling of language models. Advances in Neural Information Processing Systems, 37:100535–100570, 2024.

[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[48] Nikhil Sardana, Jacob Portes, Sasha Doubov, and Jonathan Frankle. Beyond Chinchilla-optimal: Accounting for inference in language model scaling laws. arXiv preprint arXiv:2401.00448, 2023.

[49] Mustafa Shukor, Enrico Fini, Victor Guilherme Turrisi da Costa, Matthieu Cord, Joshua Susskind, and Alaaeldin El-Nouby. Scaling laws for native multimodal models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 12–23. IEEE, 2025.

[50] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8309–8318. IEEE, 2019.

[51] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[52] Yiwen Tang, Ziyu Guo, Zhuhao Wang, Renrui Zhang, Qizhi Chen, Junli Liu, Delin Qu, Dong Wang, Bin Zhao, and Xuelong Li. Exploring the potential of encoder-free architectures in 3D LMMs. In International Conference on Learning Representations, volume 2026, pages 150063–150084, 2026.

[53] Chenxin Tao, Shiqian Su, Xizhou Zhu, Chenyu Zhang, Zhe Chen, Jiawen Liu, Wenhai Wang, Lewei Lu, Gao Huang, Yu Qiao, et al. HoVLE: Unleashing the power of monolithic visionlanguage models with holistic vision-language embedding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14559–14569. IEEE, 2025.

[54] Thinking Machines Lab. Inkling: Our open-weights model. https://thinkingmachines. ai/news/introducing-inkling/, 2026.

[55] Changyao Tian, Hao Li, Gen Luo, Xizhou Zhu, Weijie Su, Hanming Deng, Jinguo Zhu, Jie Shao, Ziran Zhu, Yunpeng Liu, et al. NaViL: Rethinking scaling properties of native multimodal large language models under data constraints. Advances in Neural Information Processing Systems, 38:85618–85646, 2026.

[56] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.

[57] Shengbang Tong, David Fan, John Nguyen, Ellis Brown, Gaoyue Zhou, Shengyi Qian, Boyang Zheng, Théophane Vallaeys, Junlin Han, Rob Fergus, et al. Beyond language modeling: An exploration of multimodal pretraining. arXiv preprint arXiv:2603.03276, 2026.

[58] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[59] Han Wang, Yongjie Ye, Bingru Li, Yuxiang Nie, Jinghui Lu, Jingqun Tang, Yanjie Wang, and Can Huang. Vision as LoRA. arXiv preprint arXiv:2503.20680, 2025.

[60] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664, 2024.

[61] Pingjie Wang, Zechen Hu, Peiru Yang, Fu Guo, and Debing Zhang. Smooth scaling laws hide stepwise token learning. arXiv preprint arXiv:2606.29858, 2026.

[62] Xinlong Wang, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Zhen Li, Yuqi Wang, et al. Multimodal learning with next-token prediction for large multimodal models. Nature, 650(8101):327–333, 2026.

[63] Haoyuan Wu, Aoqi Wu, Hai Wang, Jiajia Wu, Jinxiang Ou, and Bei Yu. Scaling native multimodal pre-training from scratch. arXiv preprint arXiv:2607.22043, 2026.

[64] xAI. RealWorldQA, 2024. URL https://huggingface.co/datasets/xai-org/ RealworldQA.

[65] Rui Yang, Lin Song, Yicheng Xiao, Runhui Huang, Yixiao Ge, Ying Shan, and Hengshuang Zhao. HaploVL: A single-transformer baseline for multi-modal understanding. arXiv preprint arXiv:2503.14694, 2025.

[66] Jinhui Yi, Syed Talal Wasim, Yanan Luo, Muzammal Naseer, and Juergen Gall. Video-Panda: Parameter-efficient alignment for encoder-free video-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24119–24128. IEEE, 2025.

[67] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

[68] Tao Zhang, Xiangtai Li, Zilong Huang, Yanwei Li, Weixian Lei, Xueqing Deng, Shihao Chen, Shunping Ji, and Jiashi Feng. Pixel-SAIL: Single transformer for pixel-grounded understanding. URL https://arxiv. org/abs/2504.10465, 2025.

## Appendix

A Implementation Details 16   
A.1 Visual Front End Architectures 16   
A.2 Decoder Architecture and Model Ladder 16   
A.3 Training Setup . 16   
A.4 Training and Validation Data 16   
A.5 Scaling Law Fitting Setup 17   
B Robustness of the Scaling Law Estimates 17   
B.1 Comparison with the Validation Curve Envelope . 17   
B.2 Extrapolation Error Analysis . 18   
B.3 Sensitivity to the Irreducible Loss 18   
B.4 Conditional Bootstrap Uncertainty 20   
C Compute Accounting for the Visual Front End 21   
C.1 Compute-Optimal Allocation . 21   
C.2 Loss–Compute Exponent 22   
C.3 Summary 22   
D Derivation of Eq. 7 22   
E Probes Inside the Decoder 23   
E.1 Layerwise Visual Representation Evolution Across Scales 23   
E.2 Compute-Optimal Allocation Under Causal Attention 23   
F IsoFLOP Profiles by Topic 25   
F.1 Pure Text Topics 25   
F.2 Multimodal Topics 26   
G Downstream Evaluation 26   
G.1 Benchmarks . 26   
G.2 Evaluation Protocol 26   
G.3 Results . 26

## A Implementation Details

## A.1 Visual Front End Architectures

Encoder-Free Front End. Both front ends produce one visual token per $3 2 \times 3 2$ pixel region. Our encoder-free front end adapts the patch projection pipeline used in the Gemma 4 12B Unified model [23]. In the original pipeline, $1 6 \times 1 6$ patches are merged in $3 \times 3$ spatial groups, yielding visual tokens that each cover a $4 8 \times 4 8$ pixel region. To ensure a controlled comparison, we instead use $2 \times 2$ merging, matching the encoder-based front end in both visual-token granularity and token count. Each merged patch of raw pixels passes through a LayerNorm, a linear layer, and a second LayerNorm. Learned factorized 2D position embeddings, obtained by summing separate embeddings for the two spatial axes, are then added, followed by a third LayerNorm, an RMSNorm, and a linear projection to the decoder width.

Encoder-Based Front End. The encoder-based front end uses a pretrained SigLIP 2 ViT [58] with 27 layers, width 1152, patch size 16, and AnyRes processing. $\textup { A 2 } \times 2$ ConvPool adapter and a projector match the encoder-free visual-token granularity. We choose this roughly 400M encoder scale as a representative practical setting used by recent advanced MLLMs [2, 22, 29]. The ViT is trained jointly with the decoder in every run, while its architecture and parameter count remain fixed across decoder scales.

Matched Comparison Protocol. Both front ends use the same preprocessing pipeline, consume the same pixels, and pass the same number of visual tokens to the decoder. The main comparison therefore holds visual content and token count fixed while varying the representation attached to each token and the attention pattern over visual tokens described in §3.3. Fixing the ViT across the ladder isolates decoder scaling and avoids introducing joint scaling of the encoder and decoder as an additional variable. Consequently, the ViT accounts for a smaller fraction of total model capacity as the decoder grows, and the reported scaling trends are conditional on this regime with a fixed encoder size. Jointly scaling the visual encoder would define a different allocation problem, requiring a separate sweep that balances potential representation gains against additional compute in the front end.

## A.2 Decoder Architecture and Model Ladder

The two systems share a complete ladder of 11 rungs. Model depth and width vary across rungs, while the MoE topology remains fixed. Every rung uses 256 routed experts, activates the top 8 experts for each token, and includes one shared expert [12]. The activation ratio of routed experts is therefore fixed at $8 / 2 5 6 = 1 / 3 2$ across model scales, keeping all rungs within a consistent architectural family. The first layer is dense, followed by MoE layers. All models use RMSNorm [67] and RoPE [51].

## A.3 Training Setup

Optimization and Hyperparameters. All runs use a sequence length of 4,096, the Muon optimizer [26], and 2,000 warmup steps. Training hyperparameters are selected as a function of model scale using an internal scaling law. In particular, the batch size and learning rate vary across the model ladder. At each rung, identical hyperparameters are used for the encoder-based and encoder-free MLLMs.

Overtraining Runs and Multiplier Fitting. For both model families, we train compute-optimal model sizes at $k \in \{ 2 , 3 , 4 , 5 \}$ . To allow for deviations from the idealized separable loss model underlying Eq. 7, we keep $E , K _ { s }$ , and $\gamma _ { s }$ fixed to their compute-optimal estimates and fit an empirical multiplier $g _ { s } ^ { \mathrm { e m p } } ( k )$ using the available overtraining runs. $\mathrm { A t } k = 5$ , the fitted multipliers are 0.645 for encoder-free and 0.692 for encoder-based models. These empirical estimates are used for the overtraining analysis in Fig. 5.

## A.4 Training and Validation Data

Training Mixture. The corpus is a 1:1 mixture of text and multimodal data. Since we focus on architectural scaling rather than effects of the data mixture, we keep this ratio fixed rather than treating it as a study variable. The balanced mixture provides sufficient multimodal training tokens for robust estimation of multimodal scaling. Multimodal sources cover tasks such as captioning, charts, grounding, GUI, OCR, STEM, and knowledge. Text sources cover domains such as STEM, code, books, and wikis.

Table 2: Scaling exponents estimated by IsoFLOP and validation curve envelope fitting.
<table><tr><td>Objective</td><td>System</td><td>Estimator</td><td>Loss γ</td><td>Model a</td><td>Data b</td></tr><tr><td>Text</td><td>Encoder-free</td><td>IsoFLOP</td><td>0.0973</td><td>0.427</td><td>0.573</td></tr><tr><td>Text</td><td>Encoder-free</td><td>Envelope</td><td>0.0905</td><td>0.430</td><td>0.570</td></tr><tr><td>Text</td><td>Encoder-based</td><td>IsoFLÓP</td><td>0.0979</td><td>0.422</td><td>0.578</td></tr><tr><td>Text</td><td>Encoder-based</td><td>Envelope</td><td>0.0915</td><td>0.435</td><td>0.565</td></tr><tr><td>Multimodal</td><td>Encoder-free</td><td>IsoFLOP</td><td>0.3778</td><td>0.570</td><td>0.430</td></tr><tr><td>Multimodal</td><td>Encoder-free</td><td>Envelope</td><td>0.3668</td><td>0.583</td><td>0.417</td></tr><tr><td>Multimodal</td><td>Encoder-based</td><td>IsoFLÓP</td><td>0.2998</td><td>0.464</td><td>0.536</td></tr><tr><td>Multimodal</td><td>Encoder-based</td><td>Envelope</td><td>0.3050</td><td>0.475</td><td>0.525</td></tr></table>

Validation Losses. Validation data are disjoint from training and identical across systems. We separately track losses on pure text and on multimodal topics: text losses are computed on sequences containing no visual tokens, whereas multimodal losses are computed over text prediction tokens in sequences conditioned on images, with visual tokens masked out from the loss, and are grouped by topic. These masked visual positions remain included in $D _ { \mathrm { m m } }$ and in decoder FLOP accounting. The primary validation view weights major topics uniformly while preserving the training mixture weights of sources within each topic.

## A.5 Scaling Law Fitting Setup

Following prior scaling law work [25, 27], we use held-out validation loss as the primary metric and fit separate laws for the text and multimodal objectives, $\mathcal { L } _ { \mathrm { t e x t } }$ and ${ \mathcal { L } } _ { \mathrm { m m } } .$ . IsoFLOP fits use six logarithmically spaced budgets for each objective, covering $\mathrm { 2 \times 1 0 ^ { 1 9 } t o 2 \times 1 0 ^ { 2 0 } }$ FLOPs for $\mathcal { L } _ { \mathrm { t e x t } }$ and $1 \times 1 0 ^ { 2 0 }$ to $\mathrm { i \times 1 0 ^ { 2 1 } }$ FLOPs for ${ \mathcal { L } } _ { \mathrm { m m } }$

## B Robustness of the Scaling Law Estimates

In this section, we assess the robustness of the scaling law estimates. We begin by checking the IsoFLOP estimator against a validation curve envelope estimator in Appendix B.1. Next, we evaluate held-out extrapolation accuracy in Appendix B.2. Appendix B.3 tests the sensitivity of the loss– compute fits to the shared irreducible loss. Appendix B.4 quantifies conditional fitting uncertainty with a residual bootstrap that preserves pairs matched by budget

## B.1 Comparison with the Validation Curve Envelope

This subsection compares the IsoFLOP estimates with a validation curve envelope. We first describe the estimator, then report the text and multimodal fits in Tab. 2.

Validation Curve Envelope. We adapt the envelope approach of Chinchilla [25], which uses training loss curves, to validation loss trajectories: the held-out validation loss of each run, measured at its intermediate checkpoints. At each of 200 logarithmically spaced compute budgets, we linearly interpolate each trajectory in log compute and fit a local quadratic in log M to the five model scales around the lowest observed loss on the ladder. The quadratic vertex gives a continuous estimate of the optimum derived from the trajectories. Because adjacent budgets are highly correlated, we aggregate these estimates by their median within 24 equal bins in log compute before fitting the allocation and loss laws. We use Huber regression in log space for the allocation laws to limit the influence of isolated errors from local interpolation. Unlike the six IsoFLOP profiles, this estimator constructs a dense frontier from points along the validation trajectories.

Text Objective. Fig. 11(a) shows the text fits. Over its broader trajectory range, the envelope estimator gives $a = 0 . 4 3 0$ for encoder-free and $a = 0 . 4 3 5$ for encoder-based models, close to the IsoFLOP values of 0.427 and 0.422. Its loss–compute exponents of 0.0905 and 0.0915 are also close to the IsoFLOP estimates.

(b) Multimodal objective  
![](images/7a2642137f68eb37e5a4b1fd37f6b1c8f897650d4d55826df624cb702e3c46c4.jpg)  
Figure 11: Validation curve envelopes grouped by objective: (a) text and (b) multimodal. Within each group, the top row shows unsmoothed encoder-free and encoder-based validation trajectories. The bottom row shows representative estimates of compute-optimal $M _ { \mathrm { o p t } }$ and $D _ { \mathrm { o p t } }$ derived from the trajectories and binned in log compute, with fitted allocation laws.

Multimodal Objective. Fig. 11(b) shows the corresponding multimodal fits. The envelope estimator gives a = 0.583 for encoder-free and $a = 0 . 4 7 5$ for encoder-based models, close to the IsoFLOP values of 0.570 and 0.464. Its loss–compute exponents of 0.3668 and 0.3050 are also close to the IsoFLOP values of 0.3778 and 0.2998, preserving the same ordering: the encoder-free frontier is steeper than the encoder-based frontier.

## B.2 Extrapolation Error Analysis

We test whether the compute-optimal fits forecast held-out IsoFLOP optima when the target budget is a modest multiple of the largest fitted budget. For each objective, we fit separate loss–compute scaling laws for each architecture to the first four IsoFLOP optima, then evaluate the forecast at a held-out budget beyond the fitting window. Held-out optima are estimated independently with the quadratic IsoFLOP procedure from §2.1. All parameters of each architecture are estimated using only the fitting subset at lower compute.

Text Objective. For this check, we additionally run a separate IsoFLOP profile at $4 \times 1 0 ^ { 2 0 } \mathrm { F L O P s }$ outside the main fitting range of the scaling laws, and use it only as a held-out extrapolation target. The forecasting fit at lower compute spans $\overline { { 2 } } \times 1 0 ^ { 1 9 } \mathrm { t o } 8 \times 1 0 ^ { 1 9 }$ FLOPs, making the held-out target a 5× extrapolation beyond the largest fitted budget. The held-out profiles contain seven encoder-free and six encoder-based model scales, with both estimated minima lying inside the sampled model range. Fig. 12 shows signed relative errors of −1.03% and −0.87%.

Multimodal Objective. The fitting window spans IsoFLOP optima through $4 \times 1 0 ^ { 2 0 }$ FLOPs. We forecast the optimum at $1 \times 1 0 ^ { 2 1 }$ FLOPs, a 2.5× extrapolation beyond the largest fitted budget. Fig. 13 shows signed relative errors of −0.01% for encoder-free models and +2.19% for encoder-based models. Across both objectives, the same fitting procedure forecasts held-out compute-optimal losses over these extrapolation factors with relative errors within about 2%.

## B.3 Sensitivity to the Irreducible Loss

This subsection specifies the fit with a common $E$ used for the loss–compute laws, then tests how the estimates move when E is perturbed (Fig. 14 and Tab. 3).

Common Irreducible Loss. Following prior scaling law work [24, 25], we interpret the irreducible term E as the conditional entropy floor induced by the data distribution and prediction objective. Because the encoder-free and encoder-based systems are trained and evaluated on the same examples with the same next-token objective, their Bayes-optimal loss floor is shared. We therefore impose one $E _ { o }$ per objective across architectures, while allowing the finite compute terms $K _ { s }$ and $\gamma _ { s }$ to differ.

![](images/1af2689cb38da1f5d23a44f6918b50f7cff35bbd934b631863fcde1d652a7ed5.jpg)

Figure 12: Extrapolation error analysis on text loss. Laws fitted through $8 \times 1 0 ^ { 1 9 }$ FL $\mathcal { O } \mathrm { P s }$ are evaluated against held-out IsoFLOP optima at 4 $\times ~ 1 0 ^ { 2 0 }$ FLOPs, a $5 \times$ extrapolation. Dashed segments indicate extrapolation.  
![](images/307575513ede4b5273719314d305e0583db40c3c2d176270c6eadbf6ba679b21.jpg)  
Figure 13: Extrapolation error analysis on multimodal loss. Laws fitted through $4 \times 1 0 ^ { 2 0 }$ FLOPs are evaluated against held-out IsoFLOP optima at $1 \times 1 0 ^ { 2 1 }$ FLOPs, a $2 . 5 \times$ extrapolation. Dashed segments indicate extrapolation.

This constraint of a common E is a structural assumption, not a claim that our observations at finite scale identify the asymptote. Its interpretation further assumes that both model families can eliminate approximation error that depends on architecture as scale grows. In our setting, with six multimodal frontier points per system over approximately one decade of compute, E trades off strongly with the fitted slope. Separate floors for each architecture can therefore absorb differences within the finite range without reliably identifying distinct entropy limits. More broadly, asymptotic parameters of scaling laws are known to be sensitive to the fitting sample and specification [4, 10, 46]. We consequently treat the constraint as theoretically motivated and test below which conclusions depend on it.

For objective o and system $s \in$ {free, based}, we jointly solve

$$
\operatorname* { m i n } _ { E _ { o } , \{ K _ { s } , \gamma _ { s } \} } \sum _ { s } \sum _ { i } \left[ \mathcal { L } _ { s , i } - E _ { o } - K _ { s } C _ { s , i } ^ { - \gamma _ { s } } \right] ^ { 2 } ,\tag{8}
$$

where $K _ { s } > 0$ and $\gamma _ { s } > 0$ . The observations $( C _ { s , i } , \mathcal { L } _ { s , i } )$ are the compute-optimal IsoFLOP vertices. The fitted floors are $\hat { E } _ { \mathrm { t e x t } } = 0 . 6 0 7$ and $\hat { E } _ { \mathrm { m m } } = 0 . 4 0 3$

Sensitivity. We then fix $E _ { o }$ on a grid from $0 . 0 5 \hat { E } _ { o } \mathrm { t o } 1 . 4 5 \hat { E } _ { o }$ and, at every grid point, refit $( K _ { s } , \gamma _ { s } )$ by constrained least squares. The two objectives are swept independently. Fig. 14 reports the resulting loss–compute exponents, crossover points, and joint fitting error. The exponent ordering does not depend on the assumed floor: on the multimodal objective, $\gamma _ { \mathrm { f r e e } }$ exceeds $\gamma _ { \mathrm { b a s e d } }$ at every grid point, whereas the two text exponents remain nearly identical throughout. Within ±10% of the fitted floor, $E _ { \mathrm { m m } } / \hat { E } _ { \mathrm { m m } } \in [ 0 . 9 0 , 1 . 1 0 ]$ , the multimodal exponent gap stays between 0.077 and 0.078 while the joint fitting error rises by at most 27%. The gap between the two objectives is likewise not an artifact of an overestimated floor. Lowering E reduces every fitted exponent, yet even at $E _ { o } = 0 . 0 5 \hat { E } _ { o } ,$ close to a pure power law, the encoder-based multimodal exponent (0.144) remains twice the text exponent (0.073). The fitted exponent values themselves, and any extrapolated crossover, are more sensitive to E (Tab. 3). These are sensitivity ranges rather than statistical confidence intervals.

![](images/63c6644a80445fc6913cdd7340796ad059e34eaf862859a48d6daedcb8a8c0ea.jpg)

![](images/2912f7416c465ab4b991be3efecfea4122397a23e58c75c8a47444dc32e93a8f.jpg)

![](images/87ac8951328f3c66a43f0547905faad6e636e0c842a872a24c267b2ef8ed09bf.jpg)  
Figure 14: Sensitivity to the irreducible loss of each objective. Each row fixes $E _ { o } / \hat { E } _ { o }$ and jointly refits the encoder-free and encoder-based loss–compute scaling laws with a common $E _ { o }$ . (A) Fitted loss–compute exponent γ. (B) Implied crossover compute under compute-optimal allocation $( k = 1 )$ and 5× overtraining $( k = 5 )$ . The dashed line marks the largest fitted budget. (C) Joint SSE on raw loss, normalized by its minimum. The red band marks $\pm 1 0 \%$ around the fitted floor.

Table 3: Multimodal sensitivity to perturbations of the fitted shared irreducible loss.
<table><tr><td> $E _ { \mathrm { m m } } / \hat { E } _ { \mathrm { m m } }$ </td><td> $E _ { \mathrm { m m } }$ </td><td></td><td> $\gamma _ { \mathrm { f r e e } } - \gamma _ { \mathrm { b a s e d } }$ </td><td>Compute-optimal 5× overtraining</td><td></td></tr><tr><td> $[ 0 . 9 0 , 1 . 1 0 ]$ </td><td>[0.363, 0.444]</td><td> $[ 0 . 0 7 7 , 0 . 0 7 8 ]$ </td><td></td><td> $[ 4 . 8 , 9 . 0 ] \times 1 0 ^ { 2 1 }$ </td><td> $[ 0 . 9 , 1 . 9 ] \times 1 0 ^ { 2 2 }$ </td></tr></table>

## B.4 Conditional Bootstrap Uncertainty

In this section, we use a conditional residual bootstrap with paired resampling across architectures to characterize fitting uncertainty in the multimodal loss–compute and allocation exponents, their differences between architectures, and the extrapolated crossover budgets under compute-optimal allocation and 5× overtraining.

Bootstrap Protocol. For each architecture, we compute the residuals between the six estimated optimal losses from the IsoFLOP profiles and the corresponding values of the fitted compute law, then center them by subtracting the mean for each architecture. At each bootstrap replicate, we sample six matched residual pairs with replacement and add them to the fitted losses at the six fixed compute budgets. We then jointly refit the two loss–compute scaling laws by least squares on the untransformed loss scale, estimating a common $E _ { \mathrm { m m } }$ anew in each replicate. This paired resampling preserves the association at each budget between the encoder-free and encoder-based residuals.

We apply the same sampled budget indices to the centered residual pairs from the fits of the allocation laws. Each bootstrap replicate then yields estimates of the loss–compute and allocation exponents, their differences between architectures, and the crossover budgets under compute-optimal allocation and 5× overtraining. For the 5× crossover, each replicate reuses the original estimates of the multipliers $g _ { s } ( 5 )$ rather than estimating them again.

Following the reporting convention of Chinchilla [25], we report central 80% bootstrap percentile intervals, bounded by the 10th and 90th percentiles of the bootstrap distribution.

Loss–Compute Exponents. On the multimodal objective, the encoder-free loss–compute exponent has a bootstrap median of 0.3781, with a central 80% interval of [0.3686, 0.3873], whereas the encoder-based exponent has a median of 0.3002, with an interval of [0.2882, 0.3112]. Their difference, $\Delta \gamma = \gamma _ { \mathrm { f r e e } } - \gamma _ { \mathrm { b a s e d } } .$ , has a median of 0.0781 and an interval of [0.0687, 0.0876]. The difference is positive across all 4,000 conditional bootstrap replicates.

![](images/25d122689dad6e60adbfeb86521ce4993fd9ca4819ef83ee780d6e0a0720e71e.jpg)

![](images/40b565782798514517127fc0e4afbb2c27c6f6851c572f4de8bd3c344bbfe639.jpg)

![](images/c5329a8de6efbd9b7871a174873818f752c49932b176c82f98d51982f6e13ccd.jpg)  
Figure 15: Conditional bootstrap uncertainty of the multimodal scaling fits and extrapolated crossover. Left: joint distribution of the encoder-free and encoder-based loss–compute exponents γ. Middle: joint distribution of their model allocation exponents a. Right: distributions of the extrapolated crossover compute under compute-optimal allocation and $5 \times$ overtraining. Horizontal bars span the 10th to 90th percentiles, and dashed lines mark point estimates.

Table 4: Conditional bootstrap uncertainty of the multimodal scaling exponents. Estimates are the fitted IsoFLOP exponents. Intervals are central 80% bootstrap percentile intervals (10th to 90th percentiles).
<table><tr><td rowspan="2"></td><td colspan="2">Encoder-free</td><td colspan="2">Encoder-based</td></tr><tr><td>Exponent Estimate</td><td>80% interval</td><td>Estimate</td><td>80% interval</td></tr><tr><td>Loss γ</td><td>0.3778</td><td>[0.3686, 0.3873]</td><td>0.2998</td><td>[0.2882, 0.3112]</td></tr><tr><td>Model a</td><td>0.570</td><td>[0.546, 0.595]</td><td>0.464</td><td>[0.458, 0.472]</td></tr><tr><td>Data b</td><td>0.430</td><td>[0.405, 0.454]</td><td>0.536</td><td>[0.528, 0.543]</td></tr></table>

Compute-Optimal Allocation Exponents. The model allocation exponent $a _ { \mathrm { f r e e } }$ has a bootstrap median of 0.570, with a central 80% interval of [0.546, 0.595], whereas $a _ { \mathrm { b a s e d } }$ has a median of 0.465, with an interval of [0.458, 0.472]. Their difference, $\Delta a = a _ { \mathrm { f r e e } } - a _ { \mathrm { b a s e d } }$ , has a median of 0.105 and an interval of [0.085, 0.127], and is positive across all conditional bootstrap replicates. The corresponding data exponents have medians of 0.430 and 0.536 for the encoder-free and encoderbased systems, with central 80% intervals of [0.405, 0.454] and [0.528, 0.543], respectively. The left and middle panels of Fig. 15 show the joint bootstrap distributions, while Tab. 4 summarizes the marginal estimates and differences between architectures.

Crossover. The compute-optimal crossover has a median of $6 . 1 \times 1 0 ^ { 2 1 }$ FLOPs and an interval of $[ 4 . 2 \times 1 0 ^ { 2 1 } , 1 . 0 \times \mathrm { \hat { 1 0 ^ { 2 2 } } } ]$ , and the crossover under 5× overtraining has a median of $1 . 2 \times 1 0 ^ { 2 2 }$ FLOPs and an interval of $\left[ 8 . 4 \times 1 0 ^ { 2 1 } , 2 . 0 \times 1 0 ^ { 2 2 } \right]$ ]. All replicates produce finite crossovers within $\left[ 1 0 ^ { 2 1 } , 1 0 ^ { 2 3 } \right] \mathrm { F L O P s }$ , and refits that each leave out one budget place the compute-optimal crossover between $4 . { \overset { - } { 3 } } \times 1 0 ^ { 2 1 }$ and $9 . 8 \times 1 0 ^ { 2 1 }$ FLOPs. The right panel of Fig. 15 shows the two distributions.

## C Compute Accounting for the Visual Front End

This appendix adds the training cost of each visual front end. Let ϕ denote the resulting ViT share of encoder-based training compute. Because the loss of every trained configuration is unchanged, we estimate the encoder-based frontier under full accounting from a parametric fit ${ \mathcal { L } } ( M , D ) =$ $E + A M ^ { - \alpha } + B D ^ { - \beta }$ to its IsoFLOP points, minimizing it over M at each total budget.

## C.1 Compute-Optimal Allocation

Under full accounting, the resulting compute-optimal allocation follows $M _ { \mathrm { o p t } } \propto C ^ { 0 . 2 7 3 }$ and $D _ { \mathrm { o p t } } \propto$ $C ^ { 0 . 7 2 7 }$ for encoder-based models, compared with $a = 0 . 4 6 4$ and $b = 0 . 5 3 6$ when only decoder FLOPs are counted, while the encoder-free exponents remain $a = 0 . 5 7 0$ and $b = 0 . 4 3 0$ (Tab. 5). Because $M _ { \mathrm { f u l l } } = M _ { \mathrm { d e c } } +$ const, we have d ln $M _ { \mathrm { f u l l } } / d \ln M _ { \mathrm { d e c } } = 1 - \phi .$ , so the encoder-based exponent measured in total FLOPs per token is compressed while ϕ is large and approaches the decoder-accounting value as $\phi  0$ . Under both conventions, encoder-free training still favors larger models.

## C.2 Loss–Compute Exponent

Refitting $\begin{array} { r } { \mathcal { L } ^ { * } = E + K C ^ { - \gamma } } \end{array}$ to the six encoder-based optima under full accounting, with the shared $\hat { E } _ { \mathrm { m m } }$ fixed, gives $\gamma _ { \mathrm { b a s e d } } = 0 . 3 6 2$ with a root mean square error of 0.003, while the encoder-free exponent remains $0 . 3 7 8 \left( \mathrm { T a b } . 5 \right)$ . The exponent gap narrows from 0.078 to 0.016 but keeps its sign. The narrowing is a finite-scale effect: charging the ViT raises the cost of small encoder-based runs relatively more than that of large ones, which steepens the frontier in total compute. As ϕ → 0 at larger scales, the encoder-based exponent approaches the decoder-accounting value of 0.300, and the gap approaches 0.078.

## C.3 Summary

Neither conclusion of the main text depends on the accounting convention. For allocation, encoderfree training favors larger models than encoder-based training under both conventions, and encoderfree models retain the larger loss–compute exponent. For the crossover, charging the fixed ∼400M ViT only adds cost to encoder-based runs and leaves the encoder-free frontier unchanged, so it shifts every comparison at equal loss toward encoder-free models. At the crossover under decoder accounting in §3.2, encoder-free models therefore already require less total compute, and the crossover under full accounting occurs no later than $6 . 1 \times 1 0 ^ { 2 1 } \dot { \mathrm { F L O P s } }$ . We do not refit a numerical crossover under this convention.

Table 5: Multimodal scaling exponents under decoder and full compute accounting.
<table><tr><td>Accounting</td><td>System</td><td>Model a</td><td>Data b</td><td>Loss γ</td></tr><tr><td>Decoder only</td><td>Encoder-free</td><td>0.570</td><td>0.430</td><td>0.378</td></tr><tr><td>Decoder only</td><td>Encoder-based</td><td>0.464</td><td>0.536</td><td>0.300</td></tr><tr><td>Full</td><td>Encoder-free</td><td>0.570</td><td>0.430</td><td>0.378</td></tr><tr><td>Full</td><td>Encoder-based</td><td>0.273</td><td>0.727</td><td>0.362</td></tr></table>

## D Derivation of Eq. 7

We derive Eq. 7 from §3.2.2. Here k multiplies the compute-optimal token count at fixed model scale: $M = M _ { \mathrm { o p t } } ( \dot { C } _ { \mathrm { b a s e } } ) , D = k D _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } ) , \mathrm { \dot { a n d } } C _ { \mathrm { a c t u a l } } \dot { = } k C _ { \mathrm { b a s e } } .$

Eliminating D via $D = C _ { \mathrm { b a s e } } / M$ , the excess loss at fixed $C _ { \mathrm { b a s e } }$ becomes

$$
\mathcal { L } ( M , C _ { \mathrm { b a s e } } / M ) - E = A M ^ { - \alpha } + B C _ { \mathrm { b a s e } } ^ { - \beta } M ^ { \beta } .
$$

This expression is strictly convex in log M, so its unique stationary point is the global optimum. Differentiating with respect to M and setting the derivative to zero yields

$$
- \alpha A M ^ { - \alpha - 1 } + \beta B C _ { \mathrm { b a s e } } ^ { - \beta } M ^ { \beta - 1 } = 0 ,
$$

hence

$$
M ^ { \alpha + \beta } = { \frac { \alpha A } { \beta B } } C _ { \mathrm { b a s e } } ^ { \beta } .
$$

Writing $\rho = \alpha + \beta$ and $m = ( \alpha A / \beta B ) ^ { 1 / \rho } ,$ , we obtain

$$
M _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } ) = m C _ { \mathrm { b a s e } } ^ { \beta / \rho } .
$$

The corresponding token count is $D _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } ) = C _ { \mathrm { b a s e } } / M _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } )$ , which simplifies to

$$
D _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } ) = m ^ { - 1 } C _ { \mathrm { b a s e } } ^ { \alpha / \rho } .
$$

Keeping the model scale fixed at $M _ { \mathrm { o p t } } ( C _ { \mathrm { b a s e } } )$ and multiplying the optimal token count by $k ,$ direct substitution yields

$$
\begin{array} { r l } & { \mathcal { L } ( C _ { \mathrm { b a s e } } , k ) - E = A \Big ( m C _ { \mathrm { b a s e } } ^ { \beta / \rho } \Big ) ^ { - \alpha } + B \Big ( k m ^ { - 1 } C _ { \mathrm { b a s e } } ^ { \alpha / \rho } \Big ) ^ { - \beta } } \\ & { \qquad = \left( A m ^ { - \alpha } + B m ^ { \beta } k ^ { - \beta } \right) C _ { \mathrm { b a s e } } ^ { - \gamma } , } \end{array}
$$

where $\gamma = \alpha \beta / \rho .$ . Let

$$
H ( k ) = A m ^ { - \alpha } + B m ^ { \beta } k ^ { - \beta } .
$$

The loss–compute scaling law on the compute-optimal frontier at $k = 1$ implies $H ( 1 ) = K$ . Thus, defining the dimensionless multiplier $g ( k ) \dot { = } H ( \dot { k } ) / H ( 1 )$ gives

$$
\begin{array} { r } { \mathcal { L } ( C _ { \mathrm { b a s e } } , k ) = E + g ( k ) K C _ { \mathrm { b a s e } } ^ { - \gamma } . } \end{array}
$$

Since H(k) depends on k but not on $C _ { \mathrm { b a s e } } .$ , overtraining changes only the prefactor and leaves the compute exponent unchanged.

## E Probes Inside the Decoder

## E.1 Layerwise Visual Representation Evolution Across Scales

§3.3 shows that, without a visual encoder, visual tokens move away from their layer-0 inputs at shallower decoder layers, while text token trajectories remain close across the two systems. Fig. 16 repeats this comparison at 20, 24, and 28 layers. The same pattern holds at every scale: encoder-free visual states diverge earlier, whereas the text curves stay closely matched. The earlier visual rewriting is therefore not an artifact of a single model size.

![](images/5a9afbfe76a99134406c0bde43e5e66844c19217fb72f6c4fa85a414159d3a8e.jpg)  
Figure 16: Layerwise similarity to the decoder input across three scales. Encoder-free visual states diverge earlier at every scale, while text trajectories remain closely matched.

## E.2 Compute-Optimal Allocation Under Causal Attention

To test whether bidirectional attention among visual tokens causes the encoder-free allocation shift in §3.1, we repeat the IsoFLOP procedure of §2.1 on the causal encoder-free runs from Fig. 8. The two settings differ only in how visual tokens attend within each image. The data mixture, optimizer, and training schedule are otherwise matched, and we do not retune hyperparameters for causal attention.

Fig. 17 compares the two attention settings. Under causal attention, the text objective gives $M _ { \mathrm { o p t } } \propto$ $C ^ { 0 . 4 3 6 }$ and $D _ { \mathrm { o p t } } \propto C ^ { 0 . 5 6 4 }$ , while the multimodal objective gives $M _ { \mathrm { o p t } } \propto C ^ { 0 . 5 5 7 }$ and $D _ { \mathrm { o p t } } \propto C ^ { \hat { 0 } . 4 4 3 }$ These M exponents differ by only 0.009 and 0.013 from the bidirectional results (0.427 for text and 0.570 for multimodal). The largest multimodal vertex is extrapolated and should be read with caution. Even so, the multimodal fit still favors much larger models, so the encoder-free allocation trend is not explained by bidirectional attention among visual tokens.

![](images/645f5e63b078a2292b01df1d76957048b03da29fc66f09af53e1b2410001d4e5.jpg)  
Text objective

![](images/ed8bff0e01e7eaaffd81f6e479c98dbe056742fb004104ea953660da5cd598d6.jpg)

![](images/cd2087351684d0ce43ba9eec253fc068ef605bd1689189da74d55bab8412c85e.jpg)

![](images/06f5138f05aa68b135b65d3d1a079daeef2803d7b2e3e1d943f29491afb61f3a.jpg)

![](images/cfb116a304a6f8be5ceca776569697667ebcc17ff5d25ebdc14e3c444aadcd3e.jpg)

![](images/db39da76ac7a8603ca4966fd90dc070ece85fda2d1c2a9f9d0942d7becc4257b.jpg)

![](images/66f10f568d67751cbcb0fb41718ac0e91de728ed07ee5575c389dec0517ba1fa.jpg)

![](images/6869ccffdbc06ff0adb4b92b0142ca7f08bab928737896109d49eeb7e0cb5c0b.jpg)  
Figure 17: Compute-optimal allocation of encoder-free models under bidirectional and causal attention over visual tokens, for the text objective (left) and the multimodal objective (right). The top row shows IsoFL ${ \cal O } \mathrm { P }$ profiles, and the bottom row shows the fitted $M _ { \mathrm { o p t } }$ and $D _ { \mathrm { o p t } }$ laws. Diamonds mark fitted vertices and dashed segments indicate extrapolation.

![](images/42e4b2122b1d348535473cb7ec95f604c6cad834ed14c26356e9fcf208bfa2a8.jpg)  
Figure 18: IsoFLOP profiles and fitted compute-optimal allocation laws for three pure text sets. The encoder-free and encoder-based exponents remain close on every topic.

![](images/6f7685a582b78fdeb8d62e9789eda85eceb52ba2d961ade88948f0fae53a7a40.jpg)  
Figure 19: IsoFLOP profiles by topic and fitted compute-optimal allocation laws for five multimodal topics. On most topics, encoder-free models favor larger model scale.

## F IsoFLOP Profiles by Topic

The main text fits allocation laws to the aggregate validation loss. Here we repeat the analysis on each validation topic separately. No new models are trained. Instead, we evaluate the existing checkpoints on each topic and fit IsoFLOP profiles as in §2.1. Because the model ladder was chosen for the aggregate loss, the optimum for some topics falls near or beyond its edge. We keep mild extrapolations, shown with open markers, and drop vertices that lie far outside the sampled range.

## F.1 Pure Text Topics

We start with pure text sets, where the visual encoder plays no role and the two architectures should behave alike. Fig. 18 confirms this on ASR, STEM, and code. The allocation laws of the two architectures nearly coincide, with M exponents differing by at most 0.004. This control indicates that the differences in the multimodal topics below come from how visual inputs are processed, not from the two training runs themselves.

## F.2 Multimodal Topics

We now turn to the five multimodal topics from the main text (Fig. 19). Unlike the text sets, these topics show a clear gap between the two architectures. On most of them, encoder-free models again favor larger M and smaller D, matching the aggregate result, although the size of the shift varies.

## G Downstream Evaluation

Our scaling analyses are based on validation loss. To check whether the same trends hold on downstream tasks, we evaluate the pretrained checkpoints on a set of multimodal benchmarks, without any further training.

## G.1 Benchmarks

Perception. CV-Bench [56] probes basic visual abilities, covering two-dimensional spatial relationships and counting as well as three-dimensional depth and distance. POPE [36] measures object hallucination by asking whether a queried object appears in the image, and we pool its random, popular, and adversarial subsets. MME [20] poses concise yes/no questions spanning perception and cognition. We report accuracy on all three benchmarks.

Document Understanding. ChartQA [43] requires extracting values from charts and reasoning over them visually or logically. We report relaxed accuracy, which applies the standard relative tolerance to numerical answers. DocVQA [44] evaluates question answering over document images, where both textual content and layout matter. We report average normalized Levenshtein similarity, which gives partial credit to near-miss answers. AI2D [28] tests reasoning about the components and relationships in scientific diagrams, and we report multiple-choice accuracy. TextVQA [50] requires reading text in natural images and relating it to the surrounding scene, and we report the VQA consensus score against human answers.

General VQA. RealWorldQA [64] tests understanding of real-world scenes, including images captured from vehicles. MMStar [8] consists of questions curated so that answering them requires visual evidence, spanning a range of perception and reasoning abilities. MMBench [38] offers broad coverage of multimodal capabilities, and we use its English development set. ScienceQA-IMG [40] is the subset of ScienceQA whose school-level science questions come with images. We evaluate answer selection on it without providing the reference explanations. We report accuracy on all four benchmarks.

## G.2 Evaluation Protocol

We evaluate every pretrained checkpoint in the same 3-shot setting: each query is preceded by three solved examples, and all models share the same examples, prompt, and image preprocessing. The examples are drawn from the training split when one exists. Otherwise, we draw them from three images in the evaluation set and exclude all questions about those images from scoring. Images are resized with their aspect ratio preserved, to at most 192 visual tokens per example and 768 for the query, and the full prompt is kept within 4,096 tokens. We use no external OCR, reference explanations, or test-time fine-tuning. For multiple-choice and yes/no questions, the model selects the candidate answer to which it assigns the highest likelihood. For open-ended questions, it decodes greedily for up to 32 tokens.

## G.3 Results

Encoder-free models still score below encoder-based models at the scales we evaluate, but the gap narrows as training compute increases (Fig. 20). At the largest token budget, the gap also tends to narrow as the model grows (Tab. 6). Both trends are consistent with the loss-based findings.

![](images/413b22a88906e6f4d85d9aece49dd0beae28b7d4365d29e1c2c282f7f40f9a61.jpg)  
Multimodal training compute, $C _ { \mathrm { m m } }$ (FLOPs)  
Figure 20: Downstream benchmark performance against multimodal training compute. The y-axis is the unweighted average 3-shot score over the 11 benchmarks.

Table 6: 3-shot benchmark scores at about 100B training tokens. “Based” and “Free” denote encoderbased and encoder-free models, respectively.
<table><tr><td rowspan="2">Benchmark</td><td colspan="2">6.1B-A425M</td><td colspan="2">8B-A577M</td><td colspan="2">13B-A922M</td><td colspan="2">16B-A1B</td><td colspan="2">22B-A1.6B</td><td colspan="2">33B-A2.2B</td></tr><tr><td>Based</td><td>Free</td><td>Based</td><td>Free</td><td>Based</td><td>Free</td><td>Based</td><td>Free</td><td>Based</td><td>Free</td><td>Based</td><td>Free</td></tr><tr><td<tr><td></td><td colspan="10"></td></tr><tr><td>Perception</td><td>CV-Bench</td><td>42.5</td><td>46.9</td><td>48.2</td><td>43.5</td><td>46.1</td><td>50.4</td><td>49.5</td><td>44.1</td><td>51.5</td><td>48.7</td><td>56.2</td></tr><tr><td>53.8</td><td>POPE</td><td>66.8</td><td>60.3</td><td>64.9</td><td>61.5</td><td>75.2</td><td>57.0</td><td>69.9</td><td>53.0</td><td>67.2</td><td>66.7</td><td>71.6</td></tr><tr><td>61.8</td><td>MME</td><td>62.1</td><td>58.0</td><td>64.3</td><td>53.1</td><td>60.9</td><td>57.1</td><td>67.7</td><td>57.9</td><td>67.9</td><td>58.7</td><td>60.7</td></tr><tr><td colspan="9">58.8</td><td colspan="2"></td></tr><tr><td></td><td>ChartQA</td><td>34.8</td><td>10.1</td><td>48.5</td><td>29.7</td><td>Document 51.9</td><td>29.4</td><td>47.4</td><td>35.6</td><td>53.1</td><td>37.8</td><td>54.3</td></tr><tr><td>42.7</td><td>DocVQA</td><td>39.6</td><td>15.4</td><td>59.8</td><td>36.4</td><td>67.3</td><td>46.5</td><td>63.9</td><td>47.4</td><td>70.6</td><td>51.0</td><td>68.9</td></tr><tr><td>57.0</td><td>AI2D</td><td>44.8</td><td>40.8</td><td>46.5</td><td>41.7</td><td>50.2</td><td>45.7</td><td>50.7</td><td>47.2</td><td>55.4</td><td>49.5</td><td>57.3</td></tr><tr><td>53.5</td><td>TextVQA</td><td>21.2</td><td>21.5</td><td>52.9</td><td>22.2</td><td>54.7</td><td>33.6</td><td>44.4</td><td>23.7</td><td>58.7</td><td>39.5</td><td>49.9</td></tr><tr><td colspan="9">43.8</td><td colspan="2"></td></tr><tr><td></td><td>RealWorldQA</td><td>28.5</td><td>23.2</td><td>47.1</td><td>35.4</td><td>General VQA 39.4</td><td>36.1</td><td>41.6</td><td>43.7</td><td>43.4</td><td>47.1</td><td>42.7</td></tr><tr><td>39.9</td><td>MMStar</td><td>32.2</td><td>27.3</td><td>34.4</td><td>28.3</td><td>35.8</td><td>28.1</td><td>36.9</td><td>29.5</td><td>35.8</td><td>29.2</td><td>37.6</td></tr><tr><td>31.7</td><td>MMBench-EN</td><td>55.6</td><td>40.1</td><td>55.5</td><td>41.6</td><td>64.7</td><td>46.3</td><td>60.8</td><td>47.2</td><td>67.1</td><td>48.3</td><td>69.1</td></tr><tr><td>57.6</td><td>ScienceQA-IMG</td><td>50.5</td><td>47.1</td><td>49.1</td><td>52.0</td><td>60.5</td><td>55.3</td><td>57.0</td><td>52.3</td><td>63.9</td><td>58.8</td><td>66.1</td></tr><tr><td>62.9</td><td>Average</td><td>43.5</td><td>35.5</td><td>51.9</td><td>40.5</td><td>55.2</td><td>44.1</td><td>53.6</td><td>43.8</td><td>57.7</td><td>48.7</td><td>57.7</td></tr></table>