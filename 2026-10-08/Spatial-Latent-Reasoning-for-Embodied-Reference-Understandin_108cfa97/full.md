# Spatial Latent Reasoning for Embodied Reference Understanding

Ling Li<sup>1</sup>, Jianhui Zhong<sup>2</sup>, Wei Liu<sup>2</sup>, Zheng Jiang<sup>1</sup>, Yuxuan Liu<sup>1</sup>, Jingyu Li<sup>3</sup>, Zhidong Deng<sup>1</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Dalian University of Technology <sup>3</sup>University of Science and Technology of China

## ABSTRACT

Pointing-gesture visual grounding requires connecting hand geometry with the visual identity and extent of a referred object. A central challenge for continuous latent reasoning is how to organize these complementary cues into useful intermediate supervision. We propose Spatial Latent Reasoning (SLR), a framework that structures this supervision around an ordered sequence of geometric and visual states. A spatial ray state is supervised by fingertip position and pointing direction, followed by four states aligned with target-region features. To construct the visual targets, we introduce parity pooling, which applies polyphase grouping to average region tokens on four interleaved spatial supports. All states are generated recurrently during training and inference; auxiliary annotations are required only during training. On EgoPoint-Ground, the framework improves mIoU over same-backbone supervised fine-tuning by 2.8, 17.5, and 21.1 percentage points on Qwen3.5-4B, Qwen2.5-VL-7B, and Qwen3-VL-8B, respectively, with improvements on both hard subsets. On YouRefIt, it achieves 77.6% precision at IoU 0.5, a numerical margin of 5.2 percentage points over the reported state of the art under differing evaluation protocols. Ablations support joint geometric and visual supervision on the standard and similar-object sets, and favor parity over three alternative pooling operators on the standard set. These results support task-structured supervision for continuous pointing grounding. We will release the code and supporting materials.

## 1 INTRODUCTION

Embodied reference understanding identifies the object a person refers to through language and gesture in a shared environment (Chen et al., 2021). We focus on its image-based pointing setting: given an image and a language query, the model predicts the referred object’s bounding box (Li et al., 2026c). Consider several similar cups on a table and the query “this cup.” Recognizing the cups does not resolve which instance is intended; interpreting the hand’s direction does not determine the object’s full extent. Resolving the reference requires connecting gesture geometry with visual evidence of the target’s identity and spatial extent.

Intermediate reasoning offers a way to organize these cues before predicting the intended referent. Explicit approaches generate textual steps or invoke visual tools (Hu et al., 2024; Zheng et al., 2025); continuous approaches feed generated hidden states back as input embeddings (Hao et al., 2025). LVR (Li et al., 2026a) supervises visual-feature reconstruction with teacher-forced region tokens, while RIS (Cui et al., 2026) combines box and semantic supervision with staged internalization. These methods establish region-grounded supervision as a foundation for latent reasoning. Embodied reference understanding with pointing gestures requires modeling a specific relation: how the observed hand indicates the intended referent. Region reconstruction and box prediction do not directly supervise that hand geometry. A fixed latent budget also requires a choice of how to dis tribute target-region evidence across states. This motivates our central question: how should gesture geometry and target appearance supervise latent reasoningfor embodied reference understanding? In this paper, we propose Spatial Latent Reasoning (SLR), organizing gesture geometry and target appearance into ordered, model-generated continuous states. A spatial ray state is trained through a lightweight auxiliary head to predict the fingertip as the ray origin and the finger-base-to-fingertip direction as its orientation. Direction is defined on the visual-token grid to account for nonsquare layouts. These labels depend only on the observed hand, providing supervision distinct from targetregion reconstruction.

![](images/9b8fb881c53fdb538d2bdaf247855c78d63f06925e94bebd4abe5948e63bc752.jpg)  
Figure 1: Comparison of reasoning approaches on an EgoPoint-Ground example (Li et al., 2026c). (a) Tool-assisted reasoning crops and re-encodes image regions during text generation (Zheng et al., 2025). (b) LVR (Li et al., 2026a) supervises continuous states by reconstructing annotated region features. (c) SLR combines one spatial ray state with four states supervised by parity-pooled target features. Image crops and ray overlays illustrate the reasoning operations and supervision targets; generic latent boundary markers are omitted for clarity.

The recurrence carries the full-dimensional spatial state; the head’s predictions are not fed back. Geometric supervision, subsequent visual alignment, and answer generation jointly train this state to retain task-relevant context alongside pointing information. Four subsequent target visual states align with referred-region features, conditioning their generation on the geometry-supervised state. Training and inference share this model-generated recurrence; auxiliary labels enter only the losses. We introduce parity pooling to distribute target-region evidence across four visual states. Drawing on region aggregation (He et al., 2014) and polyphase decomposition (Smith, 2011), it groups tokens by global row and column parity and averages each group. The resulting targets summarize interleaved positions across a variable-size ROI, providing distributed regional evidence without crop re-encoding or textual descriptions. Comparisons with quadrant grouping, random grouping, and max aggregation evaluate this construction. Figure 1 contrasts the reasoning paths; Appendix A provides further comparisons.

Our contributions are threefold: (1) We propose SLR, an ordered supervision scheme that couples a full-dimensional, geometry-supervised ray state with subsequent target visual states for embodied reference understanding. (2) We introduce parity pooling for latent supervision, converting variable-size target regions into four phase-specific feature targets with interleaved spatial support. (3) SLR improves EgoPoint-Ground mIoU over same-backbone SFT by 2.8, 17.5, and 21.1 percentage points on Qwen3.5-4B, Qwen2.5-VL-7B, and Qwen3-VL-8B, respectively, with gains on both hard subsets. On YouRefIt, it reaches 77.6% precision at IoU 0.5.

## 2 METHOD

Building on Latent Visual Reasoning (LVR) (Li et al., 2026a), SLR generates one spatial ray state and four target visual states before decoding the bounding box:

$$
( I , q ) \longrightarrow h _ { \mathrm { s p } } \longrightarrow h _ { 0 0 } \longrightarrow h _ { 0 1 } \longrightarrow h _ { 1 0 } \longrightarrow h _ { 1 1 } \longrightarrow y _ { \mathrm { b b o x } } .\tag{1}
$$

Figure 2 summarizes the geometric and visual supervision of this sequence.

## 2.1 PROBLEM FORMULATION

Let $I \in \mathbb { R } ^ { H _ { 0 } \times W _ { 0 } \times 3 }$ be an image with a pointing gesture and q a grounding query. We learn

$$
f _ { \theta } : ( I , q ) \mapsto \widehat { b } , \qquad b = ( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ) \in \mathcal { B } ,\tag{2}
$$

where $\mathcal { B } = \{ b \in [ 0 , 1 ] ^ { 4 } : x _ { 1 } < x _ { 2 } , y _ { 1 } < y _ { 2 } \}$ is the space of normalized, axis-aligned boxes. Training examples form $\boldsymbol { \mathcal { D } } = \{ ( I _ { n } , q _ { n } , b _ { n } ^ { \star } ) \} _ { n = 1 } ^ { N }$ , where $b _ { n } ^ { \star }$ is the annotated target box. We serialize each box as an answer sequence $y ^ { \star } = \tau ( b ^ { \star } )$ using integer coordinates on a 1000 1000 grid. Training examples may also provide finger-base and fingertip annotations $( r _ { n } , p _ { n } )$ . Geometry and target-region features supply auxiliary losses only and are never inputs to the generated states.

(a) Image-conditioned continuous reasoning  
![](images/bcbc5b2d0ec215d6f246e7223919d15dc36134486067857f286cce03fc488210.jpg)

![](images/12056070cd2957d5f4a0dcec27879fe060d89ea7caaaa7ab079afe5f2953d0fd.jpg)  
Figure 2: Overview of SLR. (a) A frozen visual encoder and merger provide image features. The causal MLLM generates five recurrent states in its hidden space, then decodes the target box; s and e denote latent boundary markers. (b) A spatial readout predicts fingertip position and pointing direction, while visual states align with parity-pooled region features with stopped gradients. Auxiliary readouts and annotations are used only for training. The ray overlays and grid are schematic; readout dimensions correspond to Qwen3.5-4B.

Model representations. A frozen visual encoder and projector $E _ { \phi }$ map the image to a post-merge grid $V = E _ { \phi } ( I ) \in \mathbb { R } ^ { H _ { g } \times W _ { g } \times d } ,$ . The multimodal prompt $P ( I , q )$ packs these visual tokens with query embeddings and modality/position markers. A causal language backbone $F _ { \theta }$ generates continuous states $Z _ { \theta } ( I , q ) = ( z _ { 1 } , \dots , z _ { K } )$ , where $z _ { k } \in \mathbb { R } ^ { d }$ , before the box answer. Appendix A.6 gives the autoregressive formulation and the original LVR supervision objective.

## 2.2 RECURRENT CONTINUOUS LATENT STATES

Given the multimodal prompt $P ( I , q )$ , a latent-start marker s, and a latent-end marker $e ,$ each continuous slot receives the final hidden vector from the preceding position:

$$
\begin{array} { r l } & { z _ { 1 } = F _ { \theta } ( [ P ( I , q ) , s ] ) _ { \mathrm { l a s t } } , } \\ & { z _ { k } = F _ { \theta } ( [ P ( I , q ) , s , z _ { 1 } , \ldots , z _ { k - 1 } ] ) _ { \mathrm { l a s t } } , \quad 2 \leq k \leq K . } \end{array}\tag{3}
$$

Here $F _ { \theta } ( \cdot ) _ { \mathrm { l a s t } }$ is the final-layer hidden vector before vocabulary projection. We use $K = 5$ , with

$$
( z _ { 1 } , z _ { 2 } , z _ { 3 } , z _ { 4 } , z _ { 5 } ) = ( h _ { \mathrm { s p } } , h _ { 0 0 } , h _ { 0 1 } , h _ { 1 0 } , h _ { 1 1 } ) ,\tag{4}
$$

and decode the answer after $[ P , s , z _ { 1 } , z _ { 2 } , z _ { 3 } , z _ { 4 } , z _ { 5 } , e ]$ . All states retain dimension d (2560 for Qwen3.5-4B). Slot roles follow their order and supervision; we retain Coconut’s generic boundary markers without role-specific markers or embeddings.

Auxiliary losses supervise the producer vectors $z _ { k }$ injected into the slots, rather than the outputs at those slots after recomputation. Gradients propagate through the recurrence; causal masking prevents later placeholders and teacher-forced answer tokens from influencing earlier states.

## 2.3 SPATIAL RAY-STATE LATENT

An auxiliary head supervises the origin and orientation of a pointing ray from $h _ { \mathrm { s p } }$ . Position anchors the gesture in the image, while direction specifies its orientation; together they describe a pointing cue without selecting a candidate object. The full-dimensional state also receives gradients from the subsequent visual and answer objectives, allowing the recurrence to combine this geometric cue with evidence needed for referent selection.

Geometric supervision in two coordinate systems. Let $\boldsymbol { r } = \left( x _ { r } , y _ { r } \right)$ and $p = ( x _ { p } , y _ { p } )$ denote the annotated finger base and fingertip in normalized-1000 image coordinates, with the horizontal axis pointing right and the vertical axis pointing down. The fingertip regression target lies in $[ - 1 , 1 ] ^ { 2 } \colon$

![](images/60d9bc1a533e109cff54387695d2b13e77761702d5ba82563996068f1a80f5ec.jpg)  
Figure 3: Global-grid parity pooling. The target ROI spans rows and columns 1–6 of an illustrative $8 \times 8 ~ \mathrm { g r i d }$ . Each branch averages nine feature vectors selected by global row and column parity and aligns the detached mean with its generated state $h _ { a b } = z _ { 2 + 2 a + b }$ . Dashed arrows connect both inputs to the cosine loss; pooled features serve only as training targets. The inset compares contiguous quadrants with interleaved parity groups.

$$
p ^ { \star } = \left( 2 \frac { x _ { p } } { 1 0 0 0 } - 1 , 2 \frac { y _ { p } } { 1 0 0 0 } - 1 \right) .\tag{5}
$$

We scale the finger-base-to-fingertip displacement to the post-merge grid before normalizing:

$$
\Delta _ { g } = \left( W _ { g } \frac { x _ { p } - x _ { r } } { 1 0 0 0 } , \ H _ { g } \frac { y _ { p } - y _ { r } } { 1 0 0 0 } \right) , \qquad d ^ { \star } = \frac { \Delta _ { g } } { \operatorname* { m a x } ( \lVert \Delta _ { g } \rVert _ { 2 } , \epsilon ) } , \qquad \epsilon = 1 0 ^ { - 1 2 } .\tag{6}
$$

This scaling accounts for nonsquare grids; normalizing afterward removes displacement magnitude while preserving orientation in the visual grid. The position and direction targets use different scales; expressing both in grid coordinates defines the pointing ray

$$
p _ { g } = \left( W _ { g } \frac { x _ { p } } { 1 0 0 0 } , \ H _ { g } \frac { y _ { p } } { 1 0 0 0 } \right) , \qquad \mathcal { R } ( t ) = p _ { g } + t d ^ { \star } , \quad t \geq 0 .\tag{7}
$$

The finger base determines direction only; the spatial labels do not use target boxes or candidate objects. Answer and target-alignment gradients can nevertheless make the state target-aware.

Auxiliary readout. A lightweight training head reads $h _ { \mathrm { s p } }$ using LayerNorm and two linear maps:

$$
\begin{array} { r l } & { o = W _ { 2 } \mathrm { ~ G E L U } ( W _ { 1 } \mathrm { ~ L N } ( h _ { \mathrm { s p } } ) + b _ { 1 } ) + b _ { 2 } , } \\ & { \widehat { p } = o _ { 1 : 2 } , \qquad \widehat { d } = \frac { o _ { 3 : 4 } } { \operatorname* { m a x } ( \| o _ { 3 : 4 } \| _ { 2 } , \epsilon ) } . } \end{array}\tag{8}
$$

The head maps $d  2 5 6  4$ , with unconstrained fingertip outputs and a normalized direction. Its predictions are not injected into the recurrent path.

Spatial loss and validity masking. For each valid example, we average SmoothL1 over the two fingertip coordinates and combine it with directional cosine distance:

$$
\begin{array} { r } { \ell _ { \mathrm { s p a t i a l } } = \frac { 1 } { 2 } \operatorname { S m o o t h L 1 } _ { \beta = 0 . 1 } ( \widehat { p } , p ^ { \star } ) + \frac { 1 } { 2 } ( 1 - \widehat { d } ^ { \top } d ^ { \star } ) . } \end{array}\tag{9}
$$

Let $\boldsymbol { B } _ { \mathrm { s p } }$ b bcontain minibatch examples with finite finger-base and fingertip annotations and $\left\| \boldsymbol { p } - \boldsymbol { r } \right\| _ { 2 } >$ $1 0 ^ { - 6 }$ in normalized-1000 coordinates. Averaging $\ell _ { \mathrm { s p a t i a l } }$ over this set gives $\mathcal { L } _ { \mathrm { s p a t i a l } } ;$ an empty set gives zero loss. Examples with invalid finger geometry skip only the spatial auxiliary term; their answer loss and any valid target-region supervision remain active.

## 2.4 PARITY-POOLED VISUAL TARGETS

Target support. On the post-merge visual grid $V .$ , token $( r , c )$ has normalized center $u _ { r , c } = ( ( c +$ $\begin{array} { r } { \frac { 1 } { 2 } ) / W _ { g } , ( r + \frac { 1 } { 2 } ) / H _ { g } ) } \end{array}$ . We select tokens whose centers lie inside the target box:

$$
\Omega _ { b } = \{ ( r , c ) : u _ { r , c } \in [ x _ { 1 } , x _ { 2 } ] \times [ y _ { 1 } , y _ { 2 } ] \} , \qquad M _ { b } ( r , c ) = { \bf 1 } [ ( r , c ) \in \Omega _ { b } ] .\tag{10}
$$

The support is defined directly by the annotated bounding box.

Polyphase decomposition. In multirate signal processing, a discrete signal is decomposed into subsequences indexed by sampling phase (Smith, 2011). For a two-dimensional signal X, stridetwo sampling along each axis gives four polyphase components:

$$
( { \mathcal { P } } _ { \alpha \beta } X ) _ { i , j } = X _ { 2 i + \alpha , 2 j + \beta } , \qquad ( \alpha , \beta ) \in \{ 0 , 1 \} ^ { 2 } .\tag{11}
$$

The components partition the lattice, as in polyphase representations for image sampling (Chaman & Dokmanic, 2021); we partition the ROI mask and visual features to define supervision targets.´

Region-conditioned phase averages. For phase $( \alpha , \beta )$ , define

$$
\Omega _ { b } ^ { \alpha \beta } = \{ ( r , c ) \in \Omega _ { b } : r \mathrm { m o d } 2 = \alpha , \mathrm { ~ } c \mathrm { m o d } 2 = \beta \} , \qquad n _ { \alpha \beta } = | \Omega _ { b } ^ { \alpha \beta } | .\tag{12}
$$

For each nonempty support $( n _ { \alpha \beta } > 0 )$ , parity pooling computes a mean feature

$$
\begin{array} { l } { \displaystyle \mu _ { \alpha \beta } ( V , b ) = \frac { 1 } { n _ { \alpha \beta } } \sum _ { ( r , c ) \in \Omega _ { b } ^ { \alpha \beta } } V _ { r , c } } \\ { = \frac { \sum _ { i , j } ( \mathcal { P } _ { \alpha \beta } M _ { b } ) _ { i , j } ( \mathcal { P } _ { \alpha \beta } V ) _ { i , j } } { \sum _ { i , j } ( \mathcal { P } _ { \alpha \beta } M _ { b } ) _ { i , j } } . } \end{array}\tag{13}
$$

The descriptors supervise $\left( z _ { 2 } , z _ { 3 } , z _ { 4 } , z _ { 5 } \right)$ in order (00, 01, 10, 11). The four phase indices are anchored to the full image grid and remain fixed when the target ROI changes.

Figure 3 contrasts these interleaved supports with contiguous quadrants. For example, phase (0, 1) pools tokens at even rows and odd columns throughout the ROI, rather than selecting one side of the object. The fixed phase-to-state correspondence assigns each recurrent state a consistent sampling rule across differently sized regions.

Each ROI token contributes to exactly one phase. Empty phases are omitted from the auxiliary loss without duplicating tokens, while all four recurrent slots remain. The phase means recover the overall ROI mean when weighted by their counts, but discard within-group arrangement. Appendix C gives the occupancy and mean-recovery identities.

## 2.5 VISUAL ALIGNMENT AND JOINT TRAINING

Let $\mu _ { n , k }$ denote the phase mean assigned to state $z _ { n , k + 1 }$ , and let  collect minibatch example–slot pairs with nonempty phase supports. Using stop-gradient targets, we minimize

$$
\mathcal { L } _ { \mathrm { t a r g e t } } = \frac { 1 } { | A | } \sum _ { ( n , k ) \in A } \left( 1 - \frac { z _ { n , k + 1 } ^ { \top } \mathrm { s g } ( \mu _ { n , k } ) } { \| z _ { n , k + 1 } \| _ { 2 } \| \mathrm { s g } ( \mu _ { n , k } ) \| _ { 2 } } \right) ,\tag{14}
$$

with stabilized norms and zero loss if  is empty. The average is over active pairs, and cosine alignment constrains direction rather than magnitude.

<sup>Let</sup> T <sup>be</sup> <sup>the</sup> <sup>set</sup> <sup>of</sup> <sup>supervised</sup> <sup>answer-token</sup> <sup>positions</sup> <sup>in</sup> <sup>the</sup> <sup>minibatch.</sup> <sup>The</sup> <sup>complete</sup> <sup>objective</sup> <sup>is</sup>

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { a n s } } = - \frac { 1 } { | { \mathcal T } | } \sum _ { ( n , t ) \in { \mathcal T } } \log p _ { \theta } ( y _ { n , t } ^ { \star } \mid I _ { n } , q _ { n } , Z _ { \theta } ( I _ { n } , q _ { n } ) , y _ { n , < t } ^ { \star } ) , } \ ~ } \\ { { \displaystyle { \mathcal { L } } = { \mathcal L } _ { \mathrm { a n s } } + \lambda _ { \mathrm { t a r g e t } } { \mathcal L } _ { \mathrm { t a r g e t } } + \lambda _ { \mathrm { s p a t i a l } } { \mathcal L } _ { \mathrm { s p a t i a l } } , \qquad \lambda _ { \mathrm { t a r g e t } } = \lambda _ { \mathrm { s p a t i a l } } = 0 . 0 0 5 . } } \end{array}\tag{15}
$$

Only the serialized box and end-of-sequence token contribute teacher-forced cross-entropy; all prompt, marker, latent, and padding positions are masked. We freeze the visual encoder and merger/projector, and train the language-side parameters, including text embeddings, and spatial head. All five recurrent states are model-generated. Answer and target-alignment gradients propagate through earlier states, coupling intermediate supervision to the final grounding objective.

## 2.6 INFERENCE

Given $( I , q )$ , the model generates five states, appends the latent-end marker, and greedily decodes the box until the end-of-sequence token or output limit. The spatial readout is unused, and no ray– object intersection is performed. Compute depends on prompt and answer length, caching, and the fixed latent budget. Appendices B.2 and B.3 detail parsing, constrained decoding, and timing.

## 3 EXPERIMENTS

## 3.1 DATASETS AND EVALUATION

We evaluate on EgoPoint-Ground (Li et al., 2026c), which provides egocentric hand–target annotations, and YouRefIt (Chen et al., 2021), an embodied reference benchmark with language and ges-

Table 1: Results on EgoPoint-Ground. SLR uses one spatial and four parity-supervised visual states. Shading marks SLR; bold denotes the best grounding scores per backbone, including ties. Human scores provide a reference; token counts and times appear in Table 9 in Appendix E.
<table><tr><td>Model</td><td>Method</td><td>P@0.3↑</td><td>P@0.5↑</td><td>P@0.7↑</td><td>mIoU ↑</td></tr><tr><td>Human Study</td><td>Human Participants</td><td>0.922</td><td>0.909</td><td>0.830</td><td>0.805</td></tr><tr><td rowspan="5">Qwen2.5-VL-7B</td><td>Zero-shot</td><td>0.650</td><td>0.589</td><td>0.500</td><td>0.546</td></tr><tr><td>Standard SFT</td><td>0.644</td><td>0.589</td><td>0.489</td><td>0.538</td></tr><tr><td>Text CoT (Wei et al., 2022) (adapted)</td><td>0.744</td><td>0.661</td><td>0.494</td><td>0.575</td></tr><tr><td>PointVG-R (Li et al., 2026d)</td><td>0.833</td><td>0.761</td><td>0.639</td><td>0.670</td></tr><tr><td>SLR</td><td>0.839</td><td>0.806</td><td>0.706</td><td>0.713</td></tr><tr><td rowspan="4">Qwen3-VL-8B</td><td>Zero-shot</td><td>0.556</td><td>0.528</td><td>0.506</td><td>0.516</td></tr><tr><td>Standard SFT</td><td>0.594</td><td>0.550</td><td>0.517</td><td>0.533</td></tr><tr><td>Text CoT (Wei et al., 2022) (adapted)</td><td>0.794</td><td>0.778</td><td>0.717</td><td>0.701</td></tr><tr><td>SLR</td><td>0.828</td><td>0.800</td><td>0.761</td><td>0.744</td></tr><tr><td rowspan="5">Qwen3.5-4B</td><td>Zero-shot (Non-thinking)</td><td>0.356</td><td>0.322</td><td>0.289</td><td>0.332</td></tr><tr><td>Zero-shot (Thinking)</td><td>0.578</td><td>0.544</td><td>0.456</td><td>0.484</td></tr><tr><td>Standard SFT</td><td>0.861</td><td>0.839</td><td>0.761</td><td>0.750</td></tr><tr><td>LVR (Li et al., 2026a)</td><td>0.600</td><td>0.561</td><td>0.511</td><td>0.525</td></tr><tr><td>SLR</td><td>0.906</td><td>0.878</td><td>0.811</td><td>0.778</td></tr></table>

ture. Within EgoPoint-Ground, Hard-Similar focuses on visually similar objects and Hard-Complex on complex scenes, allowing evaluation under two challenging grounding conditions.  
We report mIoU and localization accuracy P@τ on EgoPoint-Ground at $\tau \in \{ 0 . 3 , 0 . 5 , 0 . 7 \}$ . YouRefIt uses thresholds 0.25, 0.5, 0.75 ; each row identifies its evaluation metric. We organize the main comparisons by backbone and use mIoU to summarize box overlap across EgoPoint-Ground examples. Thresholded accuracy additionally measures localization at increasing precision requirements. Table 9 reports token counts and times; Appendix B.3 details their measurement.

## 3.2 COMPARISON METHODS

The comparison covers direct prediction (zero-shot and standard SFT), explicit reasoning (Text CoT (Wei et al., 2022) and PointVG-R (Li et al., 2026d)), and continuous visual reasoning (LVR (Li et al., 2026a)). Tables 1 and 2 show the available comparisons on Qwen2.5-VL-7B, Qwen3-VL-8B, and Qwen3.5-4B. Rows marked “adapted” denote backbone-specific adaptations. Tables group methods by backbone to distinguish improvements from the proposed framework from differences in the underlying model; human performance is reported separately as a reference.

## 3.3 IMPLEMENTATION AND CHECKPOINT SELECTION

Our Qwen3.5-4B implementation freezes the visual encoder and merger and trains the language model and spatial readout. It uses AdamW with learning rate 10−<sup>5</sup>, weight decay 0.01, and an effective batch size of 12 across four devices. Table 6 summarizes the reference settings; Appendix B.1 details the architecture, training, and preprocessing.

We select checkpoints by validation mIoU using free greedy generation with a 96-token output limit and caching disabled. The highest-scoring checkpoint is selected, with the earlier epoch breaking an exact tie, and is fixed across the parent test set and both hard subsets. This protocol separates checkpoint selection from final evaluation and avoids subset-specific checkpoint selection.

## 3.4 MAIN RESULTS

Improvements across models and localization criteria. Tables 1 and 2 show higher mIoU than same-backbone SFT in all nine backbone–evaluation combinations. Standard-set gains range from 2.8 to 21.1 percentage points, supporting the framework across different backbones. The benefit also extends beyond the direct-prediction baseline: on Qwen2.5-VL-7B and Qwen3-VL-8B, SLR exceeds the strongest reported reasoning baseline in standard-set mIoU by 4.3 points in each case. SLR leads all three localization thresholds within each backbone on the standard set. For Qwen3.5- 4B, P@0.7 rises from 0.761 to 0.811, showing that the gain includes boxes meeting a stricter overlap criterion. This supports improved localization quality, although aggregate IoU metrics do not separate referent-selection errors from box-placement errors.

Table 2: Key comparisons on the two EgoPoint-Ground hard subsets. Shading identifies SLR; bold scores mark the best reported result within each backbone and subset, including ties. Table 8 in Appendix D reports all methods, localization metrics, token counts, and times.
<table><tr><td rowspan="2">Model Method</td><td rowspan="2"></td><td colspan="2">Hard-Similar</td><td colspan="2">Hard-Complex</td></tr><tr><td>P@0.5↑</td><td>mIoU↑</td><td>P@0.5↑</td><td>mIoU ↑</td></tr><tr><td rowspan="4">Qwen2.5-VL-7B</td><td>Standard SFT</td><td>0.229</td><td>0.202</td><td>0.458</td><td>0.389</td></tr><tr><td>Text CoT (Wei et al., 2022) (adapted)</td><td>0.188</td><td>0.198</td><td>0.448</td><td>0.411</td></tr><tr><td>PointVG-R (Li et al., 2026d)</td><td>0.313</td><td>0.276</td><td>0.573</td><td>0.473</td></tr><tr><td>SLR</td><td>0.323</td><td>0.327</td><td>0.583</td><td>0.511</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>Standard SFT</td><td>0.281</td><td>0.290</td><td>0.385</td><td>0.349</td></tr><tr><td>Text CoT (Wei et al., 2022) (adapted)</td><td>0.333</td><td>0.353</td><td>0.500</td><td>0.452</td></tr><tr><td>SLR</td><td>0.302</td><td>0.315</td><td>0.542</td><td>0.495</td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>Standard SFT</td><td>0.396</td><td>0.361</td><td>0.594</td><td>0.536</td></tr><tr><td>LVR (Li et al., 2026a)</td><td>0.167</td><td>0.158</td><td>0.396</td><td>0.396</td></tr><tr><td>SLR</td><td>0.458</td><td>0.421</td><td>0.656</td><td>0.551</td></tr></table>

Performance depends on the grounding condition. Both hard subsets retain mIoU gains over SFT across all three backbones. Qwen3.5-4B gains more on Hard-Similar than Hard-Complex (6.0 versus 1.5 points), making visually ambiguous referents a particularly informative setting for the component analysis below. The two subsets also expose differences between metrics. On Qwen2.5- VL-7B Hard-Complex, PointVG-R leads at P@0.3 (0.656 versus 0.615), while SLR leads at P@0.7 (0.469 versus 0.365) and mIoU (0.511 versus 0.473). Thus, the ranking changes as the overlap requirement becomes stricter. On Qwen3-VL-8B Hard-Similar, Text CoT has higher mIoU (0.353 versus 0.315), indicating that the advantage over explicit reasoning is setting-dependent. Table 8 in Appendix D gives the full results. Figures 4 and 5 illustrate predictions and attention.

Evidence on a second benchmark. On YouRefIt, the gains over the same-backbone zero-shot model are similar across IoU thresholds 0.25/0.5/0.75: 8.4/8.0/8.1 points (Table 3). The persistence of the gain at the strictest threshold supports an improvement in precise grounding on this benchmark. This comparison measures the trained framework against the pretrained model; it does not isolate the contribution beyond task-specific SFT. Literature scores retain their original protocols.

Table 3: Results on YouRefIt. Published baselines use their original metrics and evaluation protocols. Shading identifies SLR; the zero-shot model uses the same backbone.
<table><tr><td>Method</td><td>Reported Metric</td><td>@0.25↑</td><td>@0.50↑</td><td>@0.75↑</td></tr><tr><td>Human Participants (Chen et al., 2021)</td><td>Accuracy</td><td>0.942</td><td>0.858</td><td>0.533</td></tr><tr><td>YouRefIt Full (Chen et al., 2021)</td><td>Accuracy</td><td>0.547</td><td>0.405</td><td>0.140</td></tr><tr><td>REP (Shi &amp; Yang, 2022)</td><td>Precision</td><td>0.588</td><td>0.457</td><td>0.188</td></tr><tr><td>Touch-Line (VTL) (Li et al., 2023)</td><td>Precision</td><td>0.711</td><td>0.635</td><td>0.390</td></tr><tr><td>AD-DINO (ADTL) (Guo et al., 2024)</td><td>Accuracy</td><td>0.763</td><td>0.724</td><td>0.554</td></tr><tr><td>CAPE (Eyiokur et al., 2026a)</td><td>mAP</td><td>0.750</td><td>0.654</td><td>0.357</td></tr><tr><td>DA-ERU (Eyiokur et al., 2026b)</td><td>mAP</td><td>0.787</td><td>0.676</td><td>0.381</td></tr><tr><td>Zero-shot (Qwen3.5-4B)</td><td>Precision</td><td>0.766</td><td>0.696</td><td>0.460</td></tr><tr><td>SLR (Qwen3.5-4B)</td><td>Precision</td><td>0.850</td><td>0.776</td><td>0.541</td></tr></table>

Qualitative grounding. Figure 4 illustrates referent selection among similar products and localization in a cluttered scene. Our predictions overlap the annotated targets in both examples, complementing the aggregate results with concrete instances of spatial grounding.

## 3.5 ABLATION STUDIES

Using Qwen3.5-4B, we evaluate components across all three EgoPoint-Ground sets and pooling on the standard set. Figure 6 and Table 7 in Appendix B.5 report the target-state count sweep.

Geometry is most effective with visual supervision on Hard-Similar. Table 4 shows that the full model achieves the best mIoU on the standard and Hard-Similar sets. Adding the spatial state to Target-only yields a larger gain on Hard-Similar than on the standard set (5.2 versus 1.4 points).

Zero-shot  
Standard SFT  
Text CoT  
PointVG-R  
Ours  
![](images/d10043e0a2b920d61cbc2ea27c957e64dc28be809be63a13167d1640971dd6a3.jpg)  
Figure 4: Grounding comparisons for two pointing examples. Columns show Zero-shot, Standard SFT, Text CoT, PointVG-R, and SLR (labeled “Ours” in the supplied visualization). Green and red boxes indicate the annotated targets and predicted regions, respectively, for each of the five compared methods.

Table 4: Component ablations on Qwen3.5-4B. Shading marks the full model (one spatial and four target states); bold marks the best grounding scores per subset, including ties. Spatial-only retains the ray state; Target-only retains the four visual states; SFT uses no latent segment.
<table><tr><td>Backbone</td><td>Variant</td><td>P@0.3↑</td><td>P@0.5↑</td><td>P@0.7↑</td><td>mIoU↑</td></tr><tr><td colspan="6">EgoPoint-Ground</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>Spatial + Parity (SLR)</td><td>0.906</td><td>0.878</td><td>0.811</td><td>0.778</td></tr><tr><td>Spatial-only</td><td>0.867</td><td>0.844</td><td>0.767</td><td>0.747</td></tr><tr><td>Target-only</td><td>0.883</td><td>0.856</td><td>0.783</td><td>0.764</td></tr><tr><td>SFT (no latent states)</td><td>0.861</td><td>0.839</td><td>0.761</td><td>0.750</td></tr><tr><td colspan="6">Hard-Similar: multiple visually similar objects</td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>Spatial + Parity (SLR)</td><td>0.500</td><td>0.458</td><td>0.396</td><td>0.421</td></tr><tr><td>Spatial-only</td><td>0.323</td><td>0.302</td><td>0.250</td><td>0.286</td></tr><tr><td>Target-only</td><td>0.427 0.417</td><td>0.396 0.396</td><td>0.344 0.344</td><td>0.369 0.361</td></tr><tr><td colspan="6">SFT (no latent states) Hard-Complex: complex scenes</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>Spatial + Parity (SLR)</td><td>0.719</td><td>0.656</td><td>0.479</td><td>0.551</td></tr><tr><td>Spatial-only</td><td>0.688</td><td>0.625</td><td>0.479</td><td>0.558</td></tr><tr><td>Target-only</td><td>0.760</td><td>0.677</td><td>0.521</td><td>0.597</td></tr><tr><td>SFT (no latent states)</td><td>0.667</td><td>0.594</td><td>0.458</td><td>0.536</td></tr></table>

Crucially, Spatial-only falls below SFT on Hard-Similar (0.286 versus 0.361), whereas the joint model reaches 0.421. The strongest result therefore comes from supervising both cues together, consistent with the motivation that pointing direction and target appearance provide complementary information when similar objects compete as referents. The Hard-Similar gain also holds at the strictest overlap threshold: joint supervision raises P@0.7 from 0.344 for Target-only to 0.396. Thus, the improvement extends to more precisely localized predictions. Spatial-only does not show this gain, suggesting that a directional cue is most useful when paired with supervision of the target’s visual extent. On Hard-Complex, Target-only leads the full model (0.597 versus 0.551), showing that this benefit depends on scene conditions. These configuration comparisons do not isolate the causal contribution of a specific geometric mechanism.

Target construction matters at a fixed state budget. Table 5 compares four target states without a spatial state. Parity exceeds Quadrant, Random, and Max in mIoU by 3.6, 2.9, and 1.8 points and leads at all three thresholds. Since all variants use four states, the advantage is not explained by a larger latent-step budget. The controls probe different aspects of the target construction: Quadrant changes interleaved support to contiguous regions, Random changes group membership, and Max changes aggregation on the parity supports (Appendix B.4). Their ordering favors the combination of spatial interleaving and mean aggregation on the standard set. Each parity target aggregates interleaved locations throughout the ROI, preserving phase-specific differences while providing distributed regional evidence. Mean aggregation incorporates all sampled features, whereas Max retains their componentwise extrema. The budget sweep (Appendix B.5) gives a complementary result: four states lead in mIoU, but one state is close (0.764 versus 0.761), and larger budgets bring no improvement. Together, these findings support a compact supervision budget with deliberately constructed visual targets and identify target organization as an important design choice for recurrent visual grounding.

![](images/c98597051f52091bcf0a777e93df32fd2693983eb18025d2a21bf2e6f78029f7.jpg)

Table 5: Pooling-operator ablations on EgoPoint-Ground with Qwen3.5-4B, four target latents, and no spatial state. Shading identifies parity pooling; bold scores mark the best grounding result, including ties. Appendix B.4 defines the operators.
<table><tr><td>Backbone</td><td>Pooling</td><td>P@0.3↑</td><td>P@0.5↑</td><td>P@0.7↑</td><td>mIoU↑</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>Parity</td><td>0.883</td><td>0.856</td><td>0.783</td><td>0.764</td></tr><tr><td>Quadrant</td><td>0.839</td><td>0.817</td><td>0.744</td><td>0.728</td></tr><tr><td>Random</td><td>0.844</td><td>0.822</td><td>0.756</td><td>0.735</td></tr><tr><td>Max</td><td>0.850</td><td>0.833</td><td>0.756</td><td>0.746</td></tr></table>

![](images/dd134fe5da8f8252b2d5ceae915c271331a74b13d565ad5f511bbb1f590fc726.jpg)  
Figure 5: Attention across five recurrent latent steps. Left: image-token attention for one pointing example for the spatial state $h _ { \mathrm { s p } }$ and target states $h _ { 0 0 } , h _ { 0 1 } , h _ { 1 0 } , h _ { 1 1 }$ , averaged over all 16 heads in layers 20, 24, 28, and 32 and normalized over image tokens using a shared color scale. Green and red boxes mark the ground-truth and predicted target, respectively; the dashed yellow box marks the ground-truth hand region. Right: mean attention mass assigned to context, image tokens, and previous latent states over all evaluation samples.

Attention across latent states. Figure 5 shows a change in attention allocation across the five latent states. The first spatial state attends predominantly to context. The four subsequent visual states combine attention to context, image tokens, and previous latent states. Image attention peaks at the first visual state, while attention to earlier latent states increases through the sequence. This pattern is consistent with a progression from contextual processing toward integrating visual scene evidence with recurrent context. The heatmaps show one example normalized over image tokens, whereas the curves average source allocation over all evaluation samples; concentrated image attention therefore need not dominate total attention. These views illustrate complementary roles of spatial and visual states, although attention alone does not establish a causal reasoning mechanism.

## 4 CONCLUSION

SLR organizes pointing geometry and target appearance into recurrent continuous states. Improvements over same-backbone SFT across three backbones and both EgoPoint-Ground hard subsets, alongside positive YouRefIt results, support task-structured intermediate supervision for embodied reference understanding.

## 5 SCOPE AND FUTURE WORK

Beyond the current single-image, two-dimensional setting, future work includes depth and temporal supervision, weaker annotation requirements, and scene-adaptive states. Controlled state interventions could clarify the reasoning mechanism, while repeated runs and broader evaluations would assess robustness.

## AI USE STATEMENT

We used large language models to assist with manuscript preparation, including drafting and revising passages, improving grammar and clarity, organizing the presentation of technical material, and refining figure and table captions. The authors are responsible for reviewing and verifying all AIassisted content and take full responsibility for the accuracy and integrity of the final manuscript.

## ETHICS STATEMENT

Egocentric imagery can expose people and private environments beyond the intended target. Image use and redistribution must respect dataset access conditions and participant privacy. The photographs and overlays in Figures 1–3 illustrate the method; Figures 4 and 5 present prediction and attention examples. A predicted region should not be interpreted as a definitive account of a person’s intention. Human evaluation requires documented participant and consent procedures.

## REPRODUCIBILITY STATEMENT

Section 2 specifies the recurrence, supervision targets, and objectives. Appendix B.1 documents the reference training configuration; Appendices B.2 and B.3 specify decoding, metric computation, and timing scope. The available source package contains configuration templates, an evaluation-record merger, source hashes, and contract tests for masking, target construction, and gradient boundaries. It does not include model weights, dataset records, or final per-example results. Reproducing these comparisons also requires split manifests, checkpoints, and run configurations.

## REFERENCES

Anadi Chaman and Ivan Dokmanic. Truly shift-invariant convolutional neural networks. In´ Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3773–3783, 2021. URL https://openaccess.thecvf.com/content/CVPR2021/ html/Chaman\_Truly\_Shift-Invariant\_Convolutional\_Neural\_Networks\_ CVPR\_2021\_paper.html.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, 2024. URL https://openaccess.thecvf.com/content/ CVPR2024/html/Chen\_SpatialVLM\_Endowing\_Vision-Language\_Models\_ with\_Spatial\_Reasoning\_Capabilities\_CVPR\_2024\_paper.html.

Yixin Chen, Qing Li, Deqian Kong, Yik Lun Kei, Song-Chun Zhu, Tao Gao, Yixin Zhu, and Siyuan Huang. YouRefIt: Embodied reference understanding with language and gesture. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1385–1395, 2021. URL https://openaccess.thecvf.com/content/ICCV2021/html/ Chen\_YouRefIt\_Embodied\_Reference\_Understanding\_With\_Language\_ and\_Gesture\_ICCV\_2021\_paper.html.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. SpatialRGPT: Grounded spatial reasoning in visionlanguage models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/f38cb4cf9a5eaa92b3cfa481832719c6-Abstract-Conference.html.

Jin Cui, Xinyue Long, Xunyong Zhang, Yadong Zhang, Chuanchang Su, Jingye Gan, Boran Zhao, and Pengju Ren. Retrieve, integrate, and synthesize: Spatial-semantic grounded latent visual reasoning. arXiv preprint arXiv:2605.07106, 2026. URL https://arxiv.org/abs/2605. 07106.

Jiajun Deng, Zhengyuan Yang, Tianlang Chen, Wengang Zhou, and Houqiang Li. TransVG: End-to-end visual grounding with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1769–1779, 2021. URL https://openaccess.

thecvf.com/content/ICCV2021/html/Deng\_TransVG\_End-to-End\_Visual\_ Grounding\_With\_Transformers\_ICCV\_2021\_paper.html.

Shuai Dong, Siyuan Wang, Xingyu Liu, Chenglin Li, Haowen Hou, and Zhongyu Wei. Interleaved latent visual reasoning with selective perceptual modeling. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 29316– 29335, 2026. URL https://aclanthology.org/2026.acl-long.1351/.

Yifan Du, Kun Zhou, Yingqian Min, Yue Ling, Wayne Xin Zhao, Youbin Wu, and Ji-Rong Wen. Revisiting the necessity of lengthy chain-of-thought in vision-centric reasoning generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12019–12029, 2026. URL https://openaccess.thecvf.com/ content/CVPR2026/html/Du\_Revisiting\_the\_Necessity\_of\_Lengthy\_ Chain-of-Thought\_in\_Vision-centric\_Reasoning\_Generalization\_ CVPR\_2026\_paper.html.

Fevziye Irem Eyiokur, Dogucan Yaman, Hazım Kemal Ekenel, and Alexander Waibel. CAPE: A CLIP-aware pointing ensemble of complementary heatmap cues for embodied reference understanding. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3939–3950, 2026a. URL https://openaccess.thecvf.com/content/ WACV2026/html/Eyiokur\_CAPE\_A\_CLIP-Aware\_Pointing\_Ensemble\_of\_ Complementary\_Heatmap\_Cues\_for\_WACV\_2026\_paper.html.

Fevziye Irem Eyiokur, Dogucan Yaman, Hazım Kemal Ekenel, and Alexander Waibel. A multimodal depth-aware method for embodied reference understanding. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2026b. URL https://arxiv.org/ abs/2510.08278.

Hao Guo, Wei Fan, Baichun Wei, Jianfei Zhu, Jin Tian, Chunzhi Yi, and Feng Jiang. AD-DINO: Attention-dynamic DINO for distance-aware embodied reference understanding. arXiv preprint arXiv:2411.08451, 2024. URL https://arxiv.org/abs/2411.08451.

Ziyu Guo, Rain Liu, Xinyan Chen, and Pheng-Ann Heng. ATLAS: Agentic or latent visual reasoning? one word is enough for both. arXiv preprint arXiv:2605.15198, 2026. URL https://arxiv.org/abs/2605.15198.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. In Conference on Language Modeling, 2025. URL https://arxiv.org/abs/2412.06769.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In European Conference on Computer Vision, pp. 346–361, 2014. URL https://arxiv.org/abs/1406.4729.

Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask R-CNN. In´ Proceedings of the IEEE International Conference on Computer Vision, pp. 2961–2969, 2017. URL https://openaccess.thecvf.com/content\_iccv\_2017/html/He\_ Mask\_R-CNN\_ICCV\_2017\_paper.html.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/fb82011040977c7712409fbdb5456647-Abstract-Conference.html.

Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, and Zicheng Liu. Latent visual reasoning. In International Conference on Learning Representations, 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ efa0939caa8a906fd0a61a9c60e46e8b-Paper-Conference.pdf.

Kelvin Li, Chuyi Shang, Leonid Karlinsky, Rogerio Feris, Trevor Darrell, and Roei Herzig. Latent implicit visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 33457–33466, 2026b. URL https://openaccess.thecvf.com/content/CVPR2026/html/Li\_Latent\_ Implicit\_Visual\_Reasoning\_CVPR\_2026\_paper.html.

Ling Li, Bowen Liu, Zinuo Zhan, Peng Jie, Jianhui Zhong, Kenglun Chang, and Zhidong Deng. Beyond language: Grounding referring expressions with hand pointing in egocentric vision. In European Conference on Computer Vision, 2026c.

Ling Li, Bowen Liu, Zinuo Zhan, Jianhui Zhong, Ziyu Zhu, Bingcai Wei, Kenglun Chang, and Zhidong Deng. PointVG-R: Internalizing geometric reasoning in MLLMs for precise pointing localization via visual chain of thought. arXiv preprint arXiv:2606.24539, 2026d. URL https: //arxiv.org/abs/2606.24539.

Yang Li, Xiaoxue Chen, Hao Zhao, Jiangtao Gong, Guyue Zhou, Federico Rossano, and Yixin Zhu. Understanding embodied reference with touch-line transformer. In International Conference on Learning Representations, 2023. URL https://yzhu.io/publication/ pointing2023iclr/.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= lLmqxkfSIw.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual CoT: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/0ff38d72a2e0aa6dbe42de83a17b2223-Abstract-Datasets\_and\_ Benchmarks\_Track.html.

Cheng Shi and Sibei Yang. Spatial and visual perspective-taking via view rotation and relation reasoning for embodied reference understanding. In European Conference on Computer Vision, 2022. URL https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/ papers/136960199.pdf.

Julius O. Smith. Spectral Audio Signal Processing. W3K Publishing, 2011. URL https://www. dsprelated.com/freebooks/sasp/Polyphase\_Decomposition.html.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper/2022/hash/ 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024. URL https://openaccess.thecvf.com/content/ CVPR2024/html/Wu\_V\_Guided\_Visual\_Search\_as\_a\_Core\_Mechanism\_in\_ Multimodal\_CVPR\_2024\_paper.html.

Qiong Wu, Xiangcong Yang, Yiyi Zhou, Chenxin Fang, Baiyang Song, Xiaoshuai Sun, and Rongrong Ji. Grounded chain-of-thought for multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 33577–33587, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Wu\_ Grounded\_Chain-of-Thought\_for\_Multimodal\_Large\_Language\_Models\_ CVPR\_2026\_paper.html.

Jiaer Xia, Bingkui Tong, Yuhang Zang, Rui Shao, and Kaiyang Zhou. Bootstrapping grounded chain-of-thought in multimodal LLMs for data-efficient model adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 208–217, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Xia\_ Bootstrapping\_Grounded\_Chain-of-Thought\_in\_Multimodal\_LLMs\_for\_ Data-Efficient\_Model\_Adaptation\_ICCV\_2025\_paper.html.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. SoftCoT: Soft chain-of-thought for efficient reasoning with LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23336–23351, 2025. URL https://aclanthology.org/2025.acl-long.1137/.

Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In International Conference on Learning Representations, 2024. URL https://arxiv.org/ abs/2310.07704.

Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L. Berg. MAttNet: Modular attention network for referring expression comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1307–1315, 2018. URL https://openaccess.thecvf.com/content\_cvpr\_2018/html/Yu\_ MAttNet\_Modular\_Attention\_CVPR\_2018\_paper.html.

Richard Zhang. Making convolutional networks shift-invariant again. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 7324–7334, 2019. URL https://proceedings.mlr.press/v97/ zhang19a.html.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. DeepEyes: Incentivizing “thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. URL https://arxiv.org/abs/2505.14362.

## APPENDIX

## A RELATED WORK

## A.1 REFERRING EXPRESSIONS AND EMBODIED POINTING

Referring expression comprehension localizes the object specified by a language expression. MAttNet (Yu et al., 2018) separates subject appearance, location, and inter-object relationships into modules whose contributions depend on the expression. TransVG (Deng et al., 2021) instead uses Transformer-based multimodal fusion and directly regresses bounding-box coordinates. These approaches establish that localization depends on both object content and its relation to the surrounding scene. Pointing introduces a further source of reference: the gesture can disambiguate objects whose appearance and linguistic descriptions are similar.

YouRefIt (Chen et al., 2021) explicitly studies embodied reference understanding with language and gesture, including the perspective differences between the person making a reference and the observer. EgoPoint (Li et al., 2026c) develops egocentric hand-pointing data and a structured visual chain-of-thought baseline. PointVG-R (Li et al., 2026d) pursues precise pointing localization through geometric reasoning traces, supervised fine-tuning, and reinforcement learning. Together, these works motivate treating pointing as a relation between a gesture and a candidate referent, rather than identifying the hand or recognizing the target category alone. Our setting follows this grounding task, but studies how to supervise the model’s recurrent continuous states. The fingertip and root-to-tip direction supervise a two-dimensional ray representation. The box is predicted from the image, query, and latent sequence, without a prescribed ray–object intersection rule.

## A.2 GROUNDED MULTIMODAL MODELS AND SPATIAL KNOWLEDGE

Grounded multimodal language models connect linguistic representations to image regions. Kosmos-2 (Peng et al., 2024) associates text spans with bounding boxes and represents location information in its output sequence. Ferret (You et al., 2024) combines discrete spatial coordinates with continuous visual features to represent regions at different granularities. These models demonstrate the value of spatially indexed features. Prompted regions and output coordinates serve different roles from recurrent hidden states generated before the answer.

SpatialVLM (Chen et al., 2024) improves qualitative and quantitative spatial reasoning through automatically generated spatial question–answer data. SpatialRGPT (Cheng et al., 2024) develops grounded spatial reasoning using region information, a spatial data pipeline, and optional depth features. Their emphasis is learning spatial knowledge and answering spatial questions, including relations in three dimensions. Our geometry supervision is narrower: a fingertip position and an image-grid-scaled direction specify a two-dimensional pointing state. This auxiliary objective does not reconstruct the three-dimensional scene or uniquely determine the referred object. It i combined with target-region visual supervision because gesture geometry and candidate appearance supply different information for the grounding decision.

## A.3 EXPLICIT VISUAL REASONING AND THINKING WITH IMAGES

Chain-of-thought prompting makes intermediate reasoning explicit in text (Wei et al., 2022). Multimodal extensions can also expose visual evidence or manipulate images during reasoning. V\* (Wu & Xie, 2024) uses guided visual search and visual working memory to acquire fine-grained evidence. Visual Sketchpad (Hu et al., 2024) enables models to draw and use visual tools, while DeepEyes (Zheng et al., 2025) trains image-tool use with reinforcement learning. We use thinking with images as a family-level description for such explicit visual interactions; Figure 1 depicts a representative crop-and-inspect path, not a shared implementation of all these methods.

A complementary line of work grounds intermediate rationales in image regions. Visual CoT (Shao et al., 2024) provides bounding-box annotations for intermediate visual reasoning and supports iterative attention to relevant regions. Grounded Chain-of-Thought (Wu et al., 2026) connects reasoning steps to localized visual cues and evaluates answer–grounding consistency. Xia et al. (2025) bootstrap grounded reasoning traces for data-efficient model adaptation. PointVG-R (Li et al., 2026d)

specializes explicit geometric reasoning to pointing localization. These approaches expose intermediate evidence and incorporate geometry into explicit reasoning.

The relevant design question is which intermediate representation is useful for the task. In controlled vision-centric generalization experiments, Du et al. (2026) find that concise grounded reasoning can be more effective than lengthy textual chains. This supports examining reasoning format and budget, but does not establish that text is universally inadequate or that continuous states are necessary. Our model uses recurrent hidden-state feedback without intermediate textual rationales or visual-tool calls. Any accuracy or latency advantage over explicit reasoning must nevertheless be measured under matched evaluation conditions; shorter output omits recurrent computation costs.

## A.4 CONTINUOUS AND LATENT VISUAL REASONING

Coconut (Hao et al., 2025) provides the central continuous transition used here: a hidden state is fed back as the next input embedding without first selecting a vocabulary token. SoftCoT (Xu et al., 2025) offers a different route to soft reasoning representations, using an assistant model and a learned projection to supply soft thoughts to a language model. These methods motivate embedding-space computation without specifying the gesture geometry or target appearance to encode.

LVR (Li et al., 2026a) directly addresses visual grounding of latent computation by aligning predicted continuous states with visual embeddings from annotated regions. It is the closest algorith mic foundation for our target supervision, and already supports reasoning with a bounded number of visual latents. ILVR (Dong et al., 2026) interleaves textual and latent reasoning and selectively mod els perceptual features from intermediate images. LIVR (Li et al., 2026b) uses learned latent tokens and an attention-bottleneck training stage to encourage implicit visual processing without intermediate visual annotations. Its learned prompt tokens differ from the freely generated hidden vector recursively fed back in Coconut and our model. ATLAS (Guo et al., 2026) also internalizes visual operations, but uses functional vocabulary tokens rather than the same full-dimensional continuous recurrence. Thus, “latent visual reasoning” encompasses distinct mechanisms and supervision.

Especially relevant is RIS (Cui et al., 2026), which grounds latent reasoning in both spatial and semantic information. RIS uses auxiliary box and semantic prediction heads, a curriculum from explicit grounded steps to latent reasoning, and a later attention-masking stage. Its reported setup also uses five latent tokens. RIS therefore provides a direct precedent for spatial-semantic latent supervision and auxiliary readouts under a compact reasoning budget. Our design assigns task-specific supervision to an ordered recurrent sequence: one state predicts fingertip position and pointing direction, followed by four states aligned with parity-pooled target features. It does not require intermediate textual reasoning traces or region-description targets. Our focus is this task-specific factorization of geometry and visual evidence within freely generated recurrent states.

Across these comparisons, our auxiliary labels constrain what information is recoverable from the recurrent states. They are not substituted for those states, and are not available at inference. Successful decoding of geometry is evidence of representation content; demonstrating that the box prediction depends on that geometry requires additional interventions or controlled ablations.

## A.5 REGION POOLING AND POLYPHASE REPRESENTATIONS

Region pooling converts a variable spatial support into a fixed-size representation. Spatial pyramid pooling (He et al., 2014) aggregates features in spatial bins at multiple resolutions. RoIAlign, introduced in Mask R-CNN (He et al., 2017), avoids quantizing region boundaries and uses bilinear interpolation to improve feature alignment. These methods motivate a general distinction between the region being represented and the operator used to summarize its features. Parity pooling use post-merge image tokens within the annotated box, without cropping or re-encoding.

The four groups arise from a classical signal-processing construction. Polyphase decomposition partitions a discrete signal into interleaved subsequences indexed by sampling phase (Smith, 2011). In two dimensions, the row and column residues modulo two define four interleaved lattices. Work on convolutional downsampling further studies how sampling phase affects shift sensitivity: Zhang (2019) applies low-pass filtering before subsampling, and Chaman & Dokmanic (2021) develops´ adaptive polyphase sampling for shift-invariant networks. These are precedents for reasoning about sampling phase, without implying that all uses of parity groups inherit these invariance properties.

We transfer the four-phase lattice partition to the construction of visual supervision targets. A mean is computed over each parity class within the target support, with phase indices anchored to the full image grid. Unlike four quadrants, the classes are spatially interleaved instead of representing contiguous parts of the box. Unlike one region mean, they retain four separate phase-conditioned summaries. The polyphase partition itself is established, and averaging each class is lossy: our construction does not imply invertibility, anti-aliasing, or translation invariance. Our contribution concerns the use of these phase-conditioned visual targets alongside a supervised ray state in recurrent pointing-grounding computation. Comparisons with quadrant means, count-matched random-group means, and maxima on parity supports test the grouping and aggregation choices. Their relative performance is an empirical question, rather than a consequence of the signal-processing analogy.

## A.6 AUTOREGRESSIVE FORMULATION AND LVR SUPERVISION

For a multimodal prompt and model-generated states $Z _ { \theta } ( I , q )$ , answer generation factorizes as

$$
p _ { \theta } ( y \mid I , q ) = \prod _ { t = 1 } ^ { | y | } p _ { \theta } ( y _ { t } \mid I , q , Z _ { \theta } ( I , q ) , y _ { < t } ) .\tag{16}
$$

Visual supervision in LVR. LVR (Li et al., 2026a) gives continuous states a visual training target. An annotated ROI selects projected image tokens $( v _ { 1 } , \ldots , v _ { M } )$ . Its supervised stage places these tokens in the latent segment as teacher-forced inputs and trains the preceding hidden states $\widetilde { h } _ { j }$ to reconstruct them, jointly with next-token prediction:

$$
\mathcal { L } _ { \mathrm { L V R } } ^ { \mathrm { S F T } } = \mathcal { L } _ { \mathrm { N T P } } + \frac { \lambda _ { \mathrm { L V R } } } { M } \sum _ { j = 1 } ^ { M } \Vert \widetilde { h } _ { j } - \mathrm { s g } ( v _ { j } ) \Vert _ { 2 } ^ { 2 } .\tag{17}
$$

At inference, LVR feeds predicted hidden vectors back between latent boundary markers.

## B ADDITIONAL EXPERIMENTAL DETAILS

## B.1 REFERENCE IMPLEMENTATION AND TRAINING

Table 6 summarizes the exported Exp3.12 configuration for Qwen3.5-4B. It initializes from the base model, freezes the visual encoder and merger, and trains language-side parameters, including token embeddings, together with the spatial head. The existing generic latent markers are retained without role-specific additions. No intermediate textual reasoning steps are used as training targets. The reference configuration disables selector injection and auxiliary bounding-box heads.

Table 6: Reference training configuration for Qwen3.5-4B with a four-epoch training budget. These settings describe the reference implementation rather than all baseline runs.

<table><tr><td>Setting</td><td>Reference value</td></tr><tr><td>Backbone / hidden dimension</td><td>Qwen3.5-4B / 2560</td></tr><tr><td>Spatial / target states</td><td>1/4</td></tr><tr><td>Image pixel limits</td><td>50,176 to 262,144</td></tr><tr><td>Optimizer / learning rate</td><td> $\mathrm { A d a m W } / 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay / seed</td><td>0.01/0</td></tr><tr><td>AdamW  $( \beta _ { 1 } , \beta _ { 2 } ) / \epsilon$ </td><td> $\left( 0 . 9 , 0 . 9 9 9 \right) / 1 0 ^ { - 8 }$ </td></tr><tr><td>Devices / batch per device / accumulation 4 / 1 / 3</td><td></td></tr><tr><td>Training budget</td><td>4 epochs</td></tr><tr><td>Spatial / target loss weight</td><td>0.005 / 0.005</td></tr><tr><td>Fingertip / direction coefficient</td><td>0.5 / 0.5</td></tr><tr><td>SmoothL1 parameter</td><td> $\beta = 0 . 1$ </td></tr><tr><td>Generation limit / cache</td><td>96 tokens / disabled</td></tr></table>

The archived environment is documented as Linux with Python 3.10, PyTorch 2.10.0 (CUDA 12.8), Transformers 5.10.2, and four H100 GPUs. Training uses fully sharded data parallelism with activation checkpointing. Model parameters are converted directly to BF16; the export preserves its BF16

Adam-state strategy without FP32 master weights. AdamW is reinitialized at each epoch, and the Exp3.12 path uses a constant learning rate without a scheduler. Precision, optimizer-state continuity, and scheduling are integral to this recipe; changing any of them alters the training configuration.

The dataset and distributed sampler are shuffled, with the sampler epoch updated each epoch. The nominal effective batch is $4 \times \bar { 1 } \times 3 = 1 2$ Checkpoints are saved each epoch as model weights, without a complete optimizer and random-state resume snapshot. Selection uses the validationgeneration protocol in Section 3.3. The training configuration’s separate 48-token field is inactive for online generation evaluation; the exported evaluation worker explicitly uses 96 tokens. Actual post-merge grid dimensions are read from the image processor rather than a fixed input resolution.

The source package includes configuration and source-identity checks and CPU contract tests for the spatial/parity objectives, valid masks, detached targets, and direct gradient paths. These checks establish implementation contracts, not reproduced GPU training or benchmark performance.

## B.2 BOUNDING-BOX DECODING PROTOCOLS

Free greedy decoding evaluates localization together with structured answer generation. The reference parser accepts an explicit bbox 2d array or either supported box-tag form, and uses the last structured box in output order. Four unmarked numbers are not accepted as a box. The scorer sorts each pair of horizontal and vertical endpoints, computes continuous-area IoU, and does not clip coordinates to the image range. A missing parse or a nonpositive union yields zero IoU. Parse success identifies a coordinate structure but does not guarantee a valid in-image box.

The tokenizer-aware constrained protocol operates during autoregressive answer generation. Fixed syntax tokens are supplied by the decoder; coordinate values are selected from the model logits using a prefix trie over tokenizations of admissible integers. At each coordinate step, the decoder selects from the allowed next-token set $\mathcal { C } _ { t } ( \widehat { y } _ { < t } ) \subseteq \mathcal { V }$ according to

$$
\widehat { y } _ { t } = \arg \operatorname* { m a x } _ { v \in \mathcal { C } _ { t } ( \widehat { y } _ { < t } ) } p _ { \theta } ( v \mid I , q , Z _ { \theta } ( I , q ) , \widehat { y } _ { < t } ) .\tag{18}
$$

The trie handles multi-token integers using the tokenizer, grammar, and decoded coordinates, without ground-truth boxes. It is a separate control available in the shared implementation; the exported Exp3.12 worker uses free greedy decoding. A matched free/constrained comparison must keep the checkpoint and latent recurrence fixed and report each protocol and its caching settings separately.

## B.3 METRICS, RECORD INTEGRITY, AND TIMING

For N evaluated examples, let $s _ { i } = \mathrm { I o U } ( \widehat { b } _ { i } , b _ { i } ^ { \star } )$ , with unparseable predictions assigned zero. We breport the following mean overlap and thresholded accuracy

$$
\mathrm { m I o U } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } , \qquad \mathrm { P @ } \tau = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ s _ { i } \geq \tau ] , \quad \tau \in \{ 0 . 3 , 0 . 5 , 0 . 7 \} .\tag{19}
$$

All evaluated examples remain in the denominator. The worker requires an annotated target box for every example. Parse failures, end-of-sequence termination, and output-limit hits are logged separately so that changes in answer formatting can be distinguished from localization quality.

The launcher assigns four disjoint shards without padding. Its merger checks complete example coverage, common checkpoint identity, data and record hashes, and unchanged model parameters. Metrics are aggregated over examples rather than obtained by an unweighted average of shard means. The package retains generated token IDs and per-example elapsed time in JSONL records; its summary CSV does not provide mean token count or mean latency.

Generated-token counts can be computed from the stored output IDs, including the terminal token when emitted. They exclude the image/query prefix and its latent carrier and boundary markers. The five recurrent states must therefore be reported separately from output length. The recorded elapsed time starts before sample collation and includes device transfer, provenance hashing, instrumented generation, text decoding, and scoring; it excludes model loading and the initial full-model hash. These instrumented times are not pure decoding latency. A dedicated latency comparison requires a common hardware, batch size, caching, warm-up, synchronization, and timing boundary, with diagnostic overhead removed or reported consistently. Fewer latents need not imply faster inference.

## B.4 REFERENCE POOLING CONTROLS

All four reference operators use the same ROI token set and produce four detached targets, with empty groups masked out of the alignment loss. Parity averages each global row/column parity group. Quadrant partitions the ROI tokens at the box midpoint, assigning tokens on the vertical or horizontal dividing line to the left or top half, respectively, and averages each group in top-left, top-right, bottom-left, bottom-right order. Max retains the parity groups and takes an elementwise maximum within each group, isolating the aggregation rule from the spatial partition.

The reference Random operator deterministically permutes ROI token indices by a SHA-256 ordering keyed by the partition version, partition seed, image-relative path, grid dimensions, and ROIindex fingerprint. It splits this ordering into four disjoint groups whose sizes match the corresponding parity groups, and averages each group. For a fixed key and ROI, the partition is repeatable and does not resample during training. This controls the number of tokens per target while changing group membership. These definitions specify the reference controls. The configuration templates do not establish whether the reported Random run used this implementation.

## B.5 TARGET-LATENT BUDGET ANALYSIS

Figure 6 and Table 7 compare four target-state budgets without a spatial state. Four states attain the highest reported mIoU (0.764), closely followed by one state (0.761); one state has higher P@0.7 (0.794 versus 0.783). Increasing the budget to nine or sixteen does not improve mIoU. This sweep supports a compact supervision budget and motivates the four-state reference used in the pooling comparison, without establishing a unique optimum. Changes in target grouping accompany changes in state count, so the comparison concerns the complete supervision configurations.

Table 7: Ablation of target-state count on EgoPoint-Ground with Qwen3.5-4B and no spatial state. Shading marks the four-state pooling reference; the full model uses five states. Bold values denote the best grounding scores, including ties. See Appendix B.3 for token-count ( ) conventions.
<table><tr><td>Backbone</td><td>Pooling latents</td><td>P@0.3↑</td><td>P@0.5↑</td><td>P@0.7 ↑</td><td>mIoU↑</td><td>Tokens†</td><td>Time (s) ↓</td></tr><tr><td colspan="8">EgoPoint-Ground</td></tr><tr><td></td><td>4 (reference)</td><td>0.883</td><td>0.856</td><td>0.783</td><td>0.764</td><td>6.000</td><td>1.766</td></tr><tr><td>Qwen3.5-4B</td><td>1</td><td>0.867</td><td>0.844</td><td>0.794</td><td>0.761</td><td>3.000</td><td>1.258</td></tr><tr><td></td><td>9</td><td>0.817</td><td>0.789</td><td>0.744</td><td>0.712</td><td>11.000</td><td>2.517</td></tr><tr><td></td><td>16</td><td>0.867</td><td>0.844</td><td>0.789</td><td>0.757</td><td>18.000</td><td>3.578</td></tr></table>

![](images/486288aeb482bd06b6661fdcd2b69641d050250e3d9bf54875288df848b82659.jpg)  
Figure 6: Effect of target-state count on EgoPoint-Ground mIoU with Qwen3.5-4B and no spatial state. Points correspond to Table 7; connecting lines aid visual comparison. Shading marks the four-state reference, and the bold value denotes the highest mIoU. Results are point estimates for the evaluated configurations, with no uncertainty intervals reported.

## C PARITY-POOLING PROPERTIES

Structural properties. The supports satisfy $\begin{array} { r } { \Omega _ { b } = \bigsqcup _ { \alpha , \beta \in \{ 0 , 1 \} } \Omega _ { b } ^ { \alpha \beta } } \end{array}$ . In particular, each selected visual token contributes to exactly one descriptor. For a nonempty rectangular support spanning h consecutive rows and w consecutive columns, let $h _ { \alpha }$ and $w _ { \beta }$ count rows and columns of each parity:

$$
n _ { \alpha \beta } = h _ { \alpha } w _ { \beta } , \qquad h _ { \alpha } \in \{ \lfloor h / 2 \rfloor , \lceil h / 2 \rceil \} , \qquad w _ { \beta } \in \{ \lfloor w / 2 \rfloor , \lceil w / 2 \rceil \} .\tag{20}
$$

Hence all four phases are populated when $h , w \geq 2$ . Thin or unresolved ROIs can leave phases empty; we omit those supervision terms without duplicating tokens, while retaining all four states.

Moreover, the ordinary ROI mean is recovered from the phase means by

$$
\overline { { V } } _ { b } = \frac { 1 } { | \Omega _ { b } | } \sum _ { ( r , c ) \in \Omega _ { b } } V _ { r , c } = \sum _ { \alpha , \beta : n _ { \alpha \beta } > 0 } \frac { n _ { \alpha \beta } } { | \Omega _ { b } | } \mu _ { \alpha \beta } ( V , b ) .\tag{21}
$$

Thus, phase means retain the information needed for the single pooled mean, given their counts, while exposing differences between interleaved subsets. Phase-wise averaging is nevertheless lossy: it does not reconstruct the feature grid or guarantee distinct descriptors. Our use of fixed phase order also does not imply translation invariance of the encoder or of the recurrent predictor.

## D FULL RESULTS ON THE HARD SUBSETS

Table 8 expands Table 2 with all reported methods, the human reference, all localization thresholds, token counts, and times. Hard-Similar (valmulti96) and Hard-Complex (valhard96) are reported separately. The Qwen2.5-VL-7B and Qwen3-VL-8B Standard SFT rows use run 02. denotes the reported token counts. Appendix B.3 discusses timing scope and token-count conventions for interpreting the reported token counts and elapsed times.

Table 8: Full results on the EgoPoint-Ground Hard-Similar and Hard-Complex subsets. Shading identifies SLR; bold values denote the best grounding scores within each backbone and subset, including ties. Dashes indicate unreported values; N/A indicates inapplicable entries.
<table><tr><td colspan="8" rowspan="1">Model           Method                  P@0.3↑P@0.5↑P@0.7↑mIoU↑ Tokens† Time (s) ↓</td></tr><tr><td colspan="8" rowspan="2">Hard-Similar: multiple visually similar objectsHuman Study    Human Participants                                                    N/A</td></tr><tr><td colspan="1" rowspan="1">0.965</td><td colspan="1" rowspan="1">0.948</td><td colspan="1" rowspan="1">0.917</td><td colspan="1" rowspan="1">0.856</td><td colspan="1" rowspan="1">N/A</td><td colspan="1" rowspan="1">N/A</td></tr><tr><td colspan="2" rowspan="1">Zero-shot</td><td colspan="1" rowspan="1">0.281</td><td colspan="1" rowspan="1">0.240</td><td colspan="1" rowspan="1">0.167</td><td colspan="1" rowspan="1">0.214</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">0.864</td></tr><tr><td colspan="2" rowspan="3">Standard SFTQwen2.5-VL-7BText CoT (adapted)PointVG-R</td><td colspan="1" rowspan="1">0.281</td><td colspan="1" rowspan="1">0.229</td><td colspan="1" rowspan="1">0.146</td><td colspan="1" rowspan="1">0.202</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">0.576</td></tr><tr><td colspan="1" rowspan="1">0.292</td><td colspan="1" rowspan="1">0.188</td><td colspan="1" rowspan="1">0.063</td><td colspan="1" rowspan="1">0.198</td><td colspan="1" rowspan="1">166.854</td><td colspan="1" rowspan="1">3.312</td></tr><tr><td colspan="1" rowspan="1">0.354</td><td colspan="1" rowspan="1">0.313</td><td colspan="1" rowspan="1">0.177</td><td colspan="1" rowspan="1">0.276</td><td colspan="1" rowspan="1">175.521</td><td colspan="1" rowspan="1">3.511</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">SLR</td><td colspan="1" rowspan="1">0.354</td><td colspan="1" rowspan="1">0.323</td><td colspan="1" rowspan="1">0.313</td><td colspan="1" rowspan="1">0.327</td><td colspan="1" rowspan="1">7.000</td><td colspan="1" rowspan="1">0.755</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Zero-shot</td><td colspan="1" rowspan="1">0.271</td><td colspan="1" rowspan="1">0.271</td><td colspan="1" rowspan="1">0.271</td><td colspan="1" rowspan="1">0.271</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">1.111</td></tr><tr><td colspan="2" rowspan="2">Standard SFTQwen3-VL-8BText CoT (adapted)</td><td colspan="1" rowspan="1">0.292</td><td colspan="1" rowspan="1">0.281</td><td colspan="1" rowspan="1">0.281</td><td colspan="1" rowspan="1">0.290</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">1.074</td></tr><tr><td colspan="1" rowspan="1">0.375</td><td colspan="1" rowspan="1">0.333</td><td colspan="1" rowspan="1">0.323</td><td colspan="1" rowspan="1">0.353</td><td colspan="1" rowspan="1">179.302</td><td colspan="1" rowspan="1">4.473</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">SLR</td><td colspan="1" rowspan="1">0.313</td><td colspan="1" rowspan="1">0.302</td><td colspan="1" rowspan="1">0.292</td><td colspan="1" rowspan="1">0.315</td><td colspan="1" rowspan="1">7.000</td><td colspan="1" rowspan="1">0.884</td></tr><tr><td colspan="1" rowspan="2"></td><td colspan="1" rowspan="1">Zero-shot (Non-thinking)</td><td colspan="1" rowspan="1">0.198</td><td colspan="1" rowspan="1">0.167</td><td colspan="1" rowspan="1">0.156</td><td colspan="1" rowspan="1">0.161</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">1.238</td></tr><tr><td colspan="1" rowspan="1">Zero-shot (Thinking)</td><td colspan="1" rowspan="1">0.271</td><td colspan="1" rowspan="1">0.240</td><td colspan="1" rowspan="1">0.188</td><td colspan="1" rowspan="1">0.228</td><td colspan="1" rowspan="1">130.615</td><td colspan="1" rowspan="1">5.149</td></tr><tr><td colspan="1" rowspan="1">Qwen3.5-4B</td><td colspan="1" rowspan="1">Standard SFT</td><td colspan="1" rowspan="1">0.417</td><td colspan="1" rowspan="1">0.396</td><td colspan="1" rowspan="1">0.344</td><td colspan="1" rowspan="1">0.361</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">1.101</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">LVR</td><td colspan="1" rowspan="1">0.177</td><td colspan="1" rowspan="1">0.167</td><td colspan="1" rowspan="1">0.135</td><td colspan="1" rowspan="1">0.158</td><td colspan="1" rowspan="1">6.000</td><td colspan="1" rowspan="1">1.329</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">SLR</td><td colspan="1" rowspan="1">0.500</td><td colspan="1" rowspan="1">0.458</td><td colspan="1" rowspan="1">0.396</td><td colspan="1" rowspan="1">0.421</td><td colspan="1" rowspan="1">7.000</td><td colspan="1" rowspan="1">1.899</td></tr><tr><td colspan="2" rowspan="2">Hard-Complex: complex scenesHuman Study</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Human Participants</td><td colspan="1" rowspan="1">0.958</td><td colspan="1" rowspan="1">0.920</td><td colspan="1" rowspan="1">0.809</td><td colspan="1" rowspan="1">0.796</td><td colspan="1" rowspan="1">N/A</td><td colspan="1" rowspan="1">N/A</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Zero-shot</td><td colspan="1" rowspan="1">0.531</td><td colspan="1" rowspan="1">0.490</td><td colspan="1" rowspan="1">0.396</td><td colspan="1" rowspan="1">0.416</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">0.869</td></tr><tr><td colspan="1" rowspan="3">Qwen2.5-VL-7B</td><td colspan="1" rowspan="1">Standard SFT</td><td colspan="1" rowspan="1">0.510</td><td colspan="1" rowspan="1">0.458</td><td colspan="1" rowspan="1">0.323</td><td colspan="1" rowspan="1">0.389</td><td colspan="1" rowspan="1">0.000</td><td colspan="1" rowspan="1">0.564</td></tr><tr><td colspan="1" rowspan="1">Text CoT (adapted)</td><td colspan="1" rowspan="1">0.583</td><td colspan="1" rowspan="1">0.448</td><td colspan="1" rowspan="1">0.208</td><td colspan="1" rowspan="1">0.411</td><td colspan="1" rowspan="1">168.760</td><td colspan="1" rowspan="1">3.368</td></tr><tr><td colspan="1" rowspan="1">PointVG-R</td><td colspan="1" rowspan="1">0.656</td><td colspan="1" rowspan="1">0.573</td><td colspan="1" rowspan="1">0.365</td><td colspan="1" rowspan="1">0.473</td><td colspan="1" rowspan="1">178.938</td><td colspan="1" rowspan="1">3.535</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">SLR</td><td colspan="1" rowspan="1">0.615</td><td colspan="1" rowspan="1">0.583</td><td colspan="1" rowspan="1">0.469</td><td colspan="1" rowspan="1">0.511</td><td colspan="1" rowspan="1">7.000</td><td colspan="1" rowspan="1">0.748</td></tr><tr><td>Model</td><td>Method</td><td>P@0.3↑</td><td>P@0.5↑</td><td>P@0.7↑</td><td>mIoU↑</td><td>Tokens†</td><td>Time (s) ↓</td></tr><tr><td rowspan="5">Qwen3-VL-8B</td><td>Zero-shot</td><td>0.427</td><td>0.396</td><td>0.302</td><td>0.354</td><td>0.000</td><td>1.116</td></tr><tr><td>Standard SFT</td><td>0.427</td><td>0.385</td><td>0.313</td><td>0.349</td><td>0.000</td><td colspan="7">1.095</td></tr><tr><td>Text CoT (adapted)</td><td>0.531</td><td>0.500</td><td>0.406</td><td>0.452</td><td>180.333</td><td colspan="7">4.676</td></tr><tr><td>SLR</td><td>0.583</td><td>0.542</td><td>0.448</td><td>0.495</td><td>7.000</td><td colspan="7">0.930</td></tr><tr><td>Zero-shot (Non-thinking)</td><td>0.323</td><td>0.302</td><td>0.260</td><td>0.274</td><td>0.000</td><td colspan="7">1.352</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>Zero-shot (Thinking)</td><td>0.375</td><td>0.333</td><td>0.240</td><td>0.293</td><td>150.188</td><td>5.629</td></tr><tr><td>Standard SFT</td><td>0.667</td><td>0.594</td><td>0.458</td><td>0.536</td><td>0.000</td><td colspan="7">1.145</td></tr><tr><td>LVR</td><td>0.500</td><td>0.396</td><td>0.333</td><td>0.396</td><td>6.000</td><td colspan="7">1.308</td></tr><tr><td>SLR</td><td>0.719</td><td>0.656</td><td>0.479</td><td>0.551</td><td>7.000</td><td colspan="7">1.889</td></tr></table>

## E SUPPLEMENTARY COMPUTATION RECORDS

Table 9 retains the token counts and times accompanying the main comparisons and ablations. The values are reproduced without converting their units. In particular, the reported zero counts for direct prediction and the fixed counts for latent methods are not total answer lengths. Continuous states, boundary markers, and generated text require separate accounting; these aggregate records do not establish a common counting convention across method families. Timing scope is discussed in Appendix B.3. Localization results in the main text do not rely on an efficiency ranking.

Table 9: Reported token counts and elapsed times for Tables 1, 4, and 5. Appendix B.3 details token-count ( ) and timing conventions. Standard denotes EgoPoint-Ground.
<table><tr><td>Backbone</td><td>Method / variant</td><td>Evaluation set</td><td>Tokens†</td><td>Time (s)</td></tr><tr><td colspan="5">Main comparisons</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Zero-shot</td><td>Standard</td><td>0.000</td><td>1.181</td></tr><tr><td></td><td>Standard SFT</td><td>Standard</td><td>0.000</td><td>0.885</td></tr><tr><td></td><td>Text CoT (adapted)</td><td>Standard</td><td>173.533</td><td>4.402</td></tr><tr><td></td><td>PointVG-R</td><td>Standard</td><td>183.622</td><td>4.708</td></tr><tr><td>Qwen3-VL-8B</td><td>SLR</td><td>Standard</td><td>7.000</td><td>1.362</td></tr><tr><td rowspan="4"></td><td>Zero-shot</td><td>Standard</td><td>0.000</td><td>1.444</td></tr><tr><td>Standard SFT</td><td>Standard</td><td>0.000</td><td>1.421</td></tr><tr><td>Text CoT (adapted)</td><td>Standard</td><td>176.006</td><td>5.602</td></tr><tr><td>SLR</td><td>Standard</td><td>7.000</td><td>1.448</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>Zero-shot (Non-thinking)</td><td>Standard</td><td>0.000</td><td>1.424</td></tr><tr><td>Zero-shot (Thinking)</td><td>Standard</td><td>150.267</td><td>5.685</td></tr><tr><td>Standard SFT LVR</td><td>Standard</td><td>0.000</td><td>1.118</td></tr><tr><td></td><td>Standard</td><td>6.000</td><td>1.341</td></tr><tr><td></td><td>SLR</td><td>Standard</td><td>7.000</td><td>1.859</td></tr><tr><td colspan="2">Component configurations Qwen3.5-4B</td><td></td><td></td><td></td></tr><tr><td></td><td>Spatial + Parity (SLR) Spatial-only</td><td>Standard</td><td>7.000</td><td>1.859</td></tr><tr><td></td><td></td><td>Standard</td><td>3.000</td><td>1.268</td></tr><tr><td></td><td>Target-only</td><td>Standard</td><td>6.000</td><td>1.766</td></tr><tr><td></td><td>SFT</td><td>Standard</td><td>0.000</td><td>1.118</td></tr><tr><td></td><td>Spatial + Parity (SLR)</td><td>Hard-Similar</td><td>7.000</td><td>1.899</td></tr><tr><td></td><td>Spatial-only</td><td>Hard-Similar</td><td>3.000</td><td>1.276</td></tr><tr><td></td><td>Target-only</td><td>Hard-Similar</td><td>6.000</td><td>1.753</td></tr><tr><td></td><td>SFT</td><td>Hard-Similar</td><td>0.000</td><td>1.101</td></tr><tr><td></td><td>Spatial + Parity (SLR)</td><td>Hard-Complex</td><td>7.000</td><td>1.889</td></tr><tr><td></td><td>Spatial-only</td><td>Hard-Complex</td><td>3.000</td><td>1.243</td></tr><tr><td></td><td>Target-only</td><td>Hard-Complex</td><td>6.000</td><td>1.743</td></tr><tr><td></td><td>SFT</td><td>Hard-Complex</td><td>0.000</td><td>1.145</td></tr><tr><td colspan="5">Pooling operators</td></tr><tr><td>Qwen3.5-4B</td><td>Parity</td><td>Standard</td><td>6.000</td><td>1.766</td></tr><tr><td></td><td>Quadrant</td><td>Standard</td><td>6.000</td><td>1.754</td></tr><tr><td></td><td>Random</td><td>Standard</td><td>6.000</td><td>1.741</td></tr><tr><td>Max</td><td></td><td>Standard</td><td>6.000</td><td>1.740</td></tr></table>