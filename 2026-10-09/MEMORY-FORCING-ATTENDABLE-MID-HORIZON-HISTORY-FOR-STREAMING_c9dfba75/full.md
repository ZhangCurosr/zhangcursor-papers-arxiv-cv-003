# MEMORY FORCING: ATTENDABLE MID-HORIZON HISTORY FOR STREAMING VIDEO GENERATION

Jiaming Zhang<sup>1,</sup>∗<sup>,</sup>†, Xinyu Wang<sup>2,</sup>∗<sup>,</sup>†, Huafeng Shi<sup>3</sup>, Gangshan Wu<sup>1</sup>, Limin Wang<sup>1,4,</sup>‡

<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University

<sup>2</sup>Tsinghua University

<sup>3</sup>Kling Team, Kuaishou Technology

<sup>4</sup>Shanghai Artificial Intelligence Laboratory

## ABSTRACT

Autoregressive video diffusion enables causal video streaming without a bidirectional pass over the full clip, but existing few-step systems usually retain only the opening and most recent frames in a fixed-size KV cache. Once an event leaves this window, later frames can no longer attend to it, a failure we term midhorizon forgetting. We present Memory Forcing, a few-step streaming method that preserves this missing history without increasing the cache size. Its Archive & Working Banks partition the cache into sink, archive, and working regions, retaining diverse intermediate events alongside recent motion under fixed memory. Because absolute temporal indices drift outside the training range, Bank-aware RoPE reassigns indices at attention time so each bank remains distinguishable. At 1.3B, Memory Forcing leads on longer clips, shows the smallest drop from 5s to 60s among methods reporting all four lengths, and preserves subjects and scenes through leave-and-return. The same design scales to Wan2.2 5B, producing more physically plausible, realistic, and dynamic videos and, to our knowledge, the first public 5B model on this forcing line.

## 1 INTRODUCTION

Modern video diffusion models (Ma et al., 2025; Zhang et al., 2026b; Wang et al., 2026a; Ma et al., 2026; Zhang et al., 2026c; 2025; 2026a; He et al., 2024; Chen et al., 2026; He et al., 2025; Huang et al., 2026; Fang et al., 2026) generate short clips with rich detail and coherent motion by denoising the full sequence with bidirectional attention. These offline generators set a high quality bar but cannot emit a frame until the clip is finished. Streaming applications instead need a causal generator with low latency and temporal consistency over long horizons. CausVid (Yin et al., 2025) and Self-Forcing (Huang et al., 2025) distill a bidirectional teacher into a few-step causal student and can produce frames on the fly. Their strictly causal prediction lets each frame inherit errors from its predecessors, so defects compound into drift. Rolling Forcing (Liu et al., 2025b) mitigates that in-window drift with rolling-window joint denoising, so frames that share a window stay aligned.

Streaming video generation (Wang et al., 2026b) conditions each new chunk on frames already produced. Those frames are stored as keys and values in a KV cache, which later denoising steps attend without re-encoding the past. This cache keeps attention cost from growing with clip length. Most current systems manage this cache as a recent FIFO of fixed length. New chunks are appended, and the oldest keys are dropped when the window is full. Local motion is preserved, but events that have left the window are gone. A subject that leaves and later returns, or a scene that cuts away and comes back, is no longer addressable. We call this mid-horizonforgetting, where lost events sit between the opening of the clip and the recent FIFO window, so later queries have no slot to retrieve them.

To keep those mid-horizon events attendable, we present Memory Forcing, a few-step streaming method that keeps rolling and DMD. The opening of a clip, the recent motion, and the events in between are different kinds of history. A single FIFO keeps only the recent keys, so intermediate events have nowhere to go. We introduce Archive & Working Banks and partition the KV cache into sink, archive, and working banks. New chunks enter working. When working is full, Similarity writes a chunk into archive, and a full archive drops the oldest chunk by FIFO. Mid-horizon events remain attendable, local motion stays in the recent window, and the cache does not grow.

![](images/9ab6f6d08819f3209f2f5a3d5452a9a743b2b63d2a002db0cd1ffae7abbf9627.jpg)  
Figure 1: Qualitative samples from Memory Forcing. Each row is one generated clip, frames in time order. Subjects, motion, and scene layout stay consistent across the clip.

Those banks must be placed on a time axis so later queries can tell the opening, the recent window, and mid-horizon events apart. Previous methods write keys with Absolute time τ<sup>abs</sup>, the frame index at write time. On a long video those indices grow outside the trained range, so later queries read stored events in the wrong place. The same rule also fails a three-bank cache. Archive is older than the recent window, and the three roles need different places on the axis, which one growing index cannot give them. We introduce Bank-aware RoPE, store raw keys for the whole window, and re-assign a temporal index τ before every attention. The assigned indices stay in the trained range, the three banks remain distinguishable, and later queries read stored events in the right place.

We evaluate the same method at Wan2.1-T2V-1.3B and Wan2.2 5B (Wang et al., 2025a). 1.3B matches prior few-step models and covers the main table and ablations that turn off the partition or τ. To our knowledge, the 5B checkpoint is the first public 5B on this forcing line and holds up better on crowded multi-subject scenes. Experiments show Memory Forcing tops longer clips, drops leas from 5s to 60s among methods reporting all four lengths, and still retrieves leave-and-return.

The contributions of this work are three. First, we introduce Archive & Working Banks, a sink / archive / working cache that keeps mid-horizon events attendable, with Similarity as the default write into archive. Second, we introduce Bank-aware RoPE, which stores raw keys for the whole window and re-assigns a temporal index before every attention. The three banks can then be told apart, and later queries read stored events in the right place. Third, experiments show Memory Forcing tops longer clips, drops least from 5s to 60s among methods reporting all four lengths, and still retrieves leave-and-return. To our knowledge, we also introduce the first public 5B.

## 2 RELATED WORK

Bidirectional video diffusion. Video diffusion models denoise the full clip with bidirectional attention. Latent U-Nets such as SVD (Blattmann et al., 2023a) and VideoLDM (Blattmann et al., 2023b) set that recipe. DiT-scale generators set the quality bar: Sora (Brooks et al., 2024), MovieGen (Polyak et al., 2024), CogVideoX (Yang et al., 2025b), HunyuanVideo (Kong et al., 2024), and Wan (Wang et al., 2025a), but cannot emit a causal stream. Autoregressive models such as NOVA (Deng et al., 2025), SANA-Video (Chen et al., 2025), and InfinityStar (Liu et al., 2025a) emit frames in time order and can grow a clip without a bidirectional pass over the full sequence.

Forcing, from causal distillation to a two-bank cache. CausVid (Yin et al., 2025) distills a bidirectional teacher with DMD (Yin et al., 2024). Self-Forcing (Huang et al., 2025) trains on self-generated history. CausalForcing (Zhu et al., 2026) uses a causal teacher ODE. SkyReels-V2 (SkyReels Team, 2025) follows Diffusion Forcing (Chen et al., 2024). Rolling Forcing (Liu et al., 2025b) adds rolling-window repair, an attention sink, and a recent FIFO $L _ { \mathrm { t e m } }$ . LongLive (Yang et al., 2025a) is scored in the same table. Rolling Forcing holds in-window drift down, and the KV cache is still a sink plus $L _ { \mathrm { t e m } } ,$ so mid-horizon events fall out of the FIFO.

Long-range memory and position. RewardForcing (Lu et al., 2025), DeepForcing (Yi et al., 2025), and RollingSink (Li et al., 2026) strengthen only the sink, so intermediate history still leaves the window. Infinity-RoPE (Yesiltepe et al., 2025) and MemRoPE (Kim et al., 2026) rewrite time or fold the past into fixed tokens, with no archive. LongLive (Yang et al., 2025a) and AnchorForcing (Yang et al., 2026) recache on prompt changes, not leave-and-return. DummyForcing (Guo et al., 2026) and Forcing-KV (Ji et al., 2026) compress the window, not mid-horizon slots. These still leave mid-horizon events without a slot, so we use Archive & Working Banks to keep them attendable and Bank-aware RoPE so later queries read them in the right place.

## 3 METHOD

Memory Forcing keeps the Rolling Forcing (Liu et al., 2025b) training recipe, its rolling schedule, and its DMD objective. What changes is the KV cache later steps attend and the temporal index used once that cache is split by role. Sections 3.1–3.2 state the inherited generator, the DMD gradient, and rolling-window joint denoising. Sections 3.3–3.6 specify the sink, archive, and working banks, Bank-aware RoPE, Aligned cache insertion, and the same recipe on Wan2.2 5B.

## 3.1 PRELIMINARIES

Autoregressive video diffusion. Autoregressive video diffusion combines temporal causality with iterative denoising, so each new frame may look only at history that has already been produced. For a clip $\boldsymbol { x } _ { 1 : N } = ( x _ { 1 } , \dots , x _ { N } )$ the joint therefore factorizes as

$$
p ( \boldsymbol x _ { 1 : N } ) = \prod _ { i = 1 } ^ { N } p ( x _ { i } \mid \boldsymbol x _ { < i } ) ,\tag{1}
$$

and each factor is realized by denoising Gaussian noise conditioned on previously generated frames. Streaming systems generate these frames in small chunks, reuse a temporal key–value cache of recent latents and a small sink of the opening frames, and condition on the current text embedding (Yin et al., 2025; Yang et al., 2026; Huang et al., 2025; Lu et al., 2025). One step may emit a chunk rather than a singleton, still called a frame unless size matters.

Those conditionals depend on how history is constructed. Two choices are Teacher Forcing (TF) and Diffusion Forcing (DF) (Chen et al., 2024). Under TF the conditional for the ith frame at noise level $t _ { j }$ is $p ( x _ { i } ^ { t _ { j } } \mid x _ { < i } ^ { 0 } )$ , so every history frame is clean. Under DF it is $p ( x _ { i } ^ { t _ { j } } \mid x _ { < i } ^ { t \geq 0 } )$ , so the histories are ground-truth frames corrupted with independent noise. Training sees ground-truth prefixes while inference feeds the model’s own predictions, known as exposure bias (Schmidt, 2019). Self-Forcing (Huang et al., 2025) lets $G _ { \theta }$ denoise each noisy frame given rolled-out $\hat { x } _ { < i } ^ { 0 }$

$$
p ( \hat { x } _ { i } ^ { t _ { j } } \mid \hat { x } _ { < i } ^ { 0 } ) , \qquad \hat { x } _ { i } ^ { t _ { j - 1 } } = \Psi \big ( G _ { \theta } ( \hat { x } _ { i } ^ { t _ { j } } , t _ { j } , \hat { x } _ { < i } ^ { 0 } ) , t _ { j - 1 } \big ) ,\tag{2}
$$

with $\hat { x } _ { i } ^ { t _ { T } } \sim \mathcal { N } ( 0 , I )$ . Here $\Psi$ injects Gaussian noise at the next lower noise level. Training and inference then see the same exposure, so the student denoises on top of its own errors rather than a ground-truth prefix. Relative to TF and DF, the train-time conditionals stay closer to a real rollout.

Distribution matching distillation. A self-rollout has no paired ground-truth frame, so a pixel denoising loss cannot be formed. CausVid (Yin et al., 2025) distills a bidirectional multi-step teacher into a few-step causal student with Distribution Matching Distillation (DMD) (Yin et al., 2024). DMD minimizes a reverse KL between the student’s output distribution $p _ { \mathrm { g e n } , t }$ and a smoothed data distribution $p _ { \mathrm { d a t a } , t }$ at sampled times $t ,$ as the difference of two scores.

![](images/03077bbeb20fab0f02f69e200d187be4d9c7c2ad21b8abbf6acdda88acfb4882.jpg)  
Figure 2: Archive & Working Banks, same cache each row. (a) Only the current window is filled. (b) The first chunk goes into sink and stays. (c) Later chunks enter working while slots remain. (d) When working is full, Similarity writes into archive. (e) FIFO then drops the oldest archive chunk.

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } _ { t } \left[ \nabla _ { \theta } \mathrm { K L } ( p _ { \mathrm { g e n } , t } \parallel p _ { \mathrm { d a t a } , t } ) \right] } \\ & { \qquad \approx - \mathbb { E } _ { t } \int \left( s _ { \mathrm { d a t a } } ( \Psi ( G _ { \theta } ( \epsilon ) , t ) , t ) - s _ { \mathrm { g e n } } ( \Psi ( G _ { \theta } ( \epsilon ) , t ) , t ) \right) \frac { \mathrm { d } G _ { \theta } ( \epsilon ) } { \mathrm { d } \theta } \mathrm { d } \epsilon , } \end{array}\tag{3}
$$

where Ψ is the forward process, ϵ is Gaussian noise, $G _ { \theta }$ is the generator, and $s _ { \mathrm { d a t a } }$ and $s _ { \mathrm { g e n } }$ score the data and the generator’s outputs. The data score comes from a frozen teacher, the student score from a separate network. Then equation 2 trains in few steps with this DMD and flow critic.

## 3.2 ROLLING-WINDOW JOINT DENOISING

Self-Forcing with DMD aligns train and inference history through equation 2. Each step still denoises one frame against a clean prefix, and that frame never attends back into its history. A local error is written into the next prefix and cannot be revised, so later frames inherit and compound it.

Rolling Forcing (Liu et al., 2025b) replaces that singleton with a window of T frames that share one forward pass and rise in noise from old to new. Neighbors revise a local error before commit, while each step emits one clean chunk. Starting at frame i, one roll updates that window by

$$
p _ { \theta } \big ( \hat { x } _ { i : i + T - 1 } ^ { t _ { 0 : T - 1 } } \mid \hat { x } _ { i : i + T - 1 } ^ { t _ { 1 : T } } , \hat { x } _ { < i } ^ { 0 } \big ) = \Psi \big ( G _ { \theta } \big ( \hat { x } _ { i : i + T - 1 } ^ { t _ { 1 : T } } , t _ { 1 : T } , \hat { x } _ { < i } ^ { 0 } \big ) , t _ { 0 : T - 1 } \big ) .\tag{4}
$$

The front is almost clean after several steps, and the back is still nearly Gaussian as it has just entered. The window then rolls forward and a new Gaussian chunk is appended. Causality is relaxed inside the window, so frames stay in order. Errors can be revised while noisy, so less enters it.

![](images/7efb5ba3a015dbbd86fbac7c9f592d93c88949151c3dc0eaf5e5df9ed75a2c19.jpg)  
Figure 3: Bank-aware RoPE assigns τ on one window. Absolute $\tau ^ { \mathrm { a b s } }$ lets later indices grow past the trained range. Contiguous $\tau ^ { \mathrm { c t g } }$ numbers from 0. Collapsed $\tau ^ { \mathrm { c o l } }$ puts sink and archive at 0. Stepped $\tau ^ { \mathrm { s t p } }$ puts sink at 0, archive at 1, working from 2. Interpolated $\tau ^ { \mathrm { i t { \bar { p } } } }$ places archive in (0, 1).

## 3.3 ARCHIVE & WORKING BANKS

Rolling Forcing keeps a global sink and a recent FIFO. The sink anchors palette and exposure, while the FIFO carries local motion for recent chunks. After a chunk leaves the FIFO it is in neither, so a person who leaves and later returns, or a scene that cuts away and comes back, is no longer addressable. The rolling window can still repair neighbors in one pass, but it does not store what happened in between, the mid-horizon gap we change. We close that gap inside the KV cache the rolling model already writes. A fewstep generator already commits clean keys as it rolls, so the question is not whether to cache but how those keys are kept after they have left the recent window. We do not add a side store, and we do not cut the first frames into a longer prefix that pretends mid-horizon content is still nearby.

Algorithm 1 Archive and Working Step   
Require: C = [sink | archive | working]   
Require: clean chunk x and query window Q   
1: if sink is not full then   
2: write $\mathrm { K V } ( x )$ to sink   
3: else   
4: if working is full then   
5: y ← arg min<sub>z∈working</sub> sim(z, x)   
▷ equation 5   
6: move y from working to archive   
7: if archive is full then   
8: FIFO-evict the oldest archive chunk   
9: end if   
10: end if   
11: append $\mathrm { K V } ( x )$ to the working tail   
12: end if   
13: $K _ { \mathrm { c t x } } \gets$ [sink; archive; working; current]   
14: apply RoPE ▷ §3.4   
15: return Attend $( Q , K _ { \mathrm { c t x } } )$

As shown in Figure 2, the cache is kept as $C \doteq [ \mathrm { s i n k }$ | archive | working] and split into three banks by role, not by a time cut of the first frames. Sink still holds the opening chunk and stays fixed, so the origin does not drift. Working holds recent history so the tail stays continuous. Archive holds intermediate events that working later had to give up, so mid-horizon stays diverse.

A new clean chunk is written into working, except while the sink is still being filled, and a full sink does not move. The working tail keeps the newest keys next to the next frames, which local motion needs. If working is full, dropping an arbitrary recent chunk would break that continuity, and archive should not collect near-duplicates of x. We therefore move

$$
y = \arg \operatorname* { m i n } _ { z \in \mathrm { w o r k i n g } } \sin ( z , x )\tag{5}
$$

from working into archive, where sim compares working keys to x. That promotion makes archive a diverse mid-horizon store. If archive is full, FIFO drops the oldest chunk to bound the cache, and the new chunk sits at the working tail. Algorithm 1 does this write and returns Attend $( Q , K _ { \mathrm { c t x } } )$

## 3.4 BANK-AWARE ROPE

Video transformers encode order with three-dimensional rotary embeddings (Su et al., 2024). We retarget the time axis. A query at index τ and a key at index $\tau ^ { \prime }$ are rotated by $\tilde { q } = R ( \tau ) q , \tilde { k } =$ $R ( \tau ^ { \prime } ) k$ , where R is the rotary block, and $\tilde { q } ^ { \top } \tilde { k } = q ^ { \top } R ( \tau - \tau ^ { \prime } )$ k depends only on the relative offset, so attention can learn shift-invariant patterns. In the usual cache rotation is applied at write time, so the stored key is already $\tilde { k } = R ( \tau ) k$ . Later steps reuse that key and do not apply a new τ . Once written, the phase stays fixed until eviction and the key stays bound to the write-time index. Values stay unrotated, so moving that key later means storing it raw and rotating it at attention time.

Rolling Forcing makes one exception for the sink: those keys are stored raw and rotated only when replayed just before $L _ { \mathrm { t e m } }$ , and the rest still follows write-time rotation. Every other key is written with Absolute time $\tau ^ { \mathrm { a b s } }$ . Past the training horizon these $\tau ^ { \mathrm { a b s } }$ indices grow without bound and leave the trained range, so later keys attend under untrained phases and cannot be moved, and only the sink can be placed again. That binding is worse after three banks: archive keys are older than $L _ { \mathrm { t e m } } ,$ so their $\tau ^ { \mathrm { a b s } }$ indices drift farther, and re-encoding only the sink leaves archive on stale indices. They also need a mid-horizon band that $\tau ^ { \mathrm { a b s } }$ cannot provide, so that RoPE cannot be inherited as is.

Therefore, we store raw keys for the whole window and re-assign τ at every attention. Relative time encodings are prior tools (Yesiltepe et al., 2025), and attaching them to the three-bank cache is the contribution. We still have to decide how sink, archive, working, and current sit on the axis. As shown in Figure 3, we design the four assignments $( N _ { a }$ and $N _ { w }$ count archive and working frames).

• Contiguous $\tau ^ { \mathrm { c t g } }$ . The aim is to keep every index inside the trained range as one relative clip. We concatenate sink, archive, working, and current in time order, number them from the left, and fix the sink at 0 with no gap, as if the three banks were one FIFO.

$$
\tau ^ { \mathrm { c t g } } = ( 0 , 1 , \dots , N - 1 ) .\tag{6}
$$

The indices never leave the trained range and the rule is cheap to apply. The three banks then share one scale, so archive sits next to the sink and looks like a moment ago.

• Collapsed $\tau ^ { \mathrm { c o l } }$ . The aim is to pin the opening shot and park mid-horizon keys on that origin. Every sink and archive token is given index $0 ,$ working starts at 1 and increases by one frame, and current continues after working. Archive has no internal order on the axis.

$$
\left\{ \begin{array} { r l } { \tau ^ { \mathrm { c o l } } ( \mathrm { S i n k } ) = 0 , } & { { } } \\ { \tau ^ { \mathrm { c o l } } ( \mathrm { A r c h i v e } _ { a } ) = 0 } & { { } ( a = 1 , \dots , N _ { a } ) , } \\ { \tau ^ { \mathrm { c o l } } ( \mathrm { W o r k i n g } _ { w } ) = w } & { { } ( w = 1 , \dots , N _ { w } ) . } \end{array} \right.\tag{7}
$$

The origin stays fixed, working still has a local order, and all indices remain small. A mid-horizon event cannot be told from the opening shot because both sit at 0.

• Stepped $\tau ^ { \mathrm { s t p } }$ . The aim is to mark three timescales without growing indices. The sink sits at 0, archive at 1, and working at $2 , 3 , . . . ,$ , with current after working. Neighboring banks differ by one and do not share a slot, but archive is still a single index.

$$
\left\{ \begin{array} { c } { \tau ^ { \mathrm { s t p } } ( \mathrm { S i n k } ) = 0 , } \\ { \tau ^ { \mathrm { s t p } } ( \mathrm { A r c h i v e } _ { a } ) = 1 \quad ( a = 1 , \dots , N _ { a } ) , } \\ { \tau ^ { \mathrm { s t p } } ( \mathrm { W o r k i n g } _ { w } ) = w + 1 \quad ( w = 1 , \dots , N _ { w } ) . } \end{array} \right.\tag{8}
$$

The banks no longer collide and the indices stay small. Archive is still a single point, so frames inside archive cannot be ordered on the axis.

• Interpolated $\tau ^ { \mathrm { i t p } }$ . The aim is to put the sink at 0, working near the present, and archive in between with its frames distinguishable. The $N _ { a }$ archive frames are spread evenly on $[ 0 , 1 ]$ , so neighboring archive keys differ by $1 / ( N _ { a } - 1 )$ ) and collapse to 0 if $N _ { a } { = } 1$ . Working increases from 1 as a near-present chain, and current follows working.

$$
\left\{ \begin{array} { c } { { \tau ^ { \mathrm { i t p } } ( \mathrm { S i n k } ) = 0 , } } \\ { { \tau ^ { \mathrm { i t p } } ( \mathrm { A r c h i v e } _ { a } ) = \displaystyle \frac { a - 1 } { N _ { a } - 1 } \quad ( a = 1 , \dots , N _ { a } , 0 \mathrm { i f } N _ { a } { = } 1 ) , } } \\ { { \tau ^ { \mathrm { i t p } } ( \mathrm { W o r k i n g } _ { w } ) = w \quad ( w = 1 , \dots , N _ { w } ) . } } \end{array} \right.\tag{9}
$$

The sink stays at the origin, working stays near the present, and archive occupies the midhorizon band with its frames distinguishable.

![](images/68f3649eab6e6fb81e5e61e4bf15dcbe47e4089c88560ebbd56778d69a751722.jpg)  
Figure 4: Further qualitative samples from Memory Forcing. Each row is one generated clip, frame in time order. Fine action and wide scenes keep identity and layout across the clip.

## 3.5 TRAINING

We keep Rolling Forcing’s distillation loop, with DMD and a flow critic on non-overlapping windows (§3.1, §3.2). A rolling-only rollout assembles the clean video from frames that left different noise levels, so the DMD clip is uneven in clarity. Training mixes rolling and self-forcing windows with equal probability. Self-forcing windows regularize camera motion, and inference uses the rolling window alone. Algorithm 1 now selects RoPE and appends a key, so each step practices the three-bank write and Bank-aware RoPE, and a later clean chunk uses that same write.

If that insertion attends only the tokens already in the cache, the stored keys are formed under a different context than denoising. We therefore use Aligned cache insertion. The insertion forward looks at the sink, the extracted archive and working banks, and the current window, the same set that a denoising step attends. Keys that enter the cache then match the attention used for the loss.

## 3.6 SCALING TO 5B

Public few-step autoregressive long-video models stop at Wan2.1-T2V-1.3B (Wang et al., 2025a), so we attach the same recipe to Wan2.2 5B (Wang et al., 2025a) to test whether the three-bank cache and Bank-aware RoPE scale on a stronger backbone. The banks, Interpolated τ<sup>itp</sup>, mixed training, Aligned cache insertion, and DMD stay as in §3.3–3.5. The 5B cut, cache, and attention budget use that backbone’s tokens per frame, and both scales are primary results in §4.

Wan2.2 5B begins as a bidirectional backbone. A forcing student is causal, so we first convert that backbone with the first two stages of CausalForcing (Zhu et al., 2026), autoregressive diffusion training and causal ODE initialization. The first stage trains a multi-step causal teacher under teacher forcing, so frame i at noise t sees a clean prefix $x _ { < i } ^ { 0 }$ as in §3.1. The resulting PF-ODE is injective at the frame level. The second stage samples trajectories from that teacher and trains the few-step student to regress the clean frame from the noisy state on the same prefix,

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } \big \| { G } _ { \theta } ( x _ { i } ^ { t } , x _ { < i } ^ { 0 } , t ) - x _ { i } ^ { 0 } \big \| _ { 2 } ^ { 2 } ,\tag{10}
$$

where $( x _ { i } ^ { t } , x _ { i } ^ { 0 } )$ lie on the teacher’s PF-ODE. After these two stages the 5B student is causal. We then train it with the Memory Forcing recipe of §3.3–3.5.

Both stages supervise a next-frame map given clean history, so clip motion sets the target size. Lowmotion clips make that map nearly the identity, so AR conversion is easy but the student copies the last frame. High-motion clips enlarge the residual and ODE-target variance, so the same regression is hard to fit. We therefore filter Koala-36M (Wang et al., 2025b) on quality and aesthetics, and mix low, mid, and high motion at 4:3:3, yielding 280,000 clips for AR diffusion training and 36,000 fo causal ODE initialization, and the 5B result in §4 supports this mix and that the recipe scales.

Table 1: Comparison on official VBench Total and Quality on VBench-Complex, Gen-Complex, and MovieGen at 5s, 15s, 30s, and 60s. CF ODE and SF ODE are the CausalForcing and Self Forcing pretraining. Red is best and blue is second-best in each column.
<table><tr><td rowspan="2">Method</td><td colspan="5">VBench (Total)</td><td colspan="5">VBench-Complex (Quality)</td><td colspan="5">Gen-Complex (Quality)</td><td colspan="5">MovieGen (Quality)</td></tr><tr><td>5s</td><td>15s</td><td>30s</td><td>60s</td><td>∆5s→60s</td><td>5s</td><td>15s</td><td>30s</td><td>60s</td><td>∆5s→60s</td><td>5s</td><td>15s</td><td>30s</td><td>60s</td><td>∆5s→60s</td><td>5s</td><td>15s</td><td>30s</td><td>60s</td><td>∆5s→60s</td></tr><tr><td>CausVid</td><td>82.99</td><td></td><td></td><td></td><td></td><td>80.83</td><td></td><td></td><td></td><td></td><td>80.35</td><td></td><td></td><td></td><td></td><td>79.62</td><td></td><td></td><td></td><td></td></tr><tr><td>NOVA</td><td>78.54</td><td>74.78</td><td></td><td></td><td></td><td>73.32</td><td>69.18</td><td></td><td></td><td></td><td>74.86</td><td>69.41</td><td></td><td></td><td></td><td>74.08</td><td>68.86</td><td></td><td></td><td></td></tr><tr><td>SkyReels-V2</td><td>82.65</td><td>80.11</td><td></td><td></td><td></td><td>81.08</td><td>75.07</td><td></td><td></td><td></td><td>78.83</td><td>71.93</td><td></td><td></td><td></td><td>79.88</td><td>75.35</td><td></td><td></td><td></td></tr><tr><td>Self-Forcing</td><td>84.09</td><td></td><td></td><td></td><td></td><td>81.37</td><td></td><td></td><td></td><td></td><td>79.99</td><td></td><td></td><td></td><td></td><td>81.09</td><td></td><td></td><td></td><td></td></tr><tr><td>LongLive</td><td>83.23</td><td>82.95</td><td>82.10</td><td>82.21</td><td>-1.02</td><td>79.73</td><td>80.01</td><td>79.07</td><td>78.81</td><td>-0.92</td><td>81.51</td><td>80.59</td><td>79.98</td><td>80.36</td><td>-1.15</td><td>80.15</td><td>79.79</td><td>79.20</td><td>79.13</td><td>-1.02</td></tr><tr><td>LongLive inf</td><td>82.97</td><td>82.84</td><td>82.43</td><td>81.77</td><td>-1.20</td><td>79.45</td><td>80.70</td><td>80.45</td><td>80.37</td><td>0.92</td><td>81.57</td><td>80.81</td><td>80.52</td><td>79.61</td><td>-1.96</td><td>80.14</td><td>79.93</td><td>79.52</td><td>78.82</td><td>-1.32</td></tr><tr><td>SANA-Video</td><td>83.18 83.63</td><td>83.17 82.82</td><td>82.80</td><td>82.22</td><td>-0.96</td><td>80.18</td><td>78.63</td><td>78.44</td><td>77.29</td><td>-2.89</td><td>83.27</td><td>80.37</td><td>80.46</td><td>78.59</td><td>-4.68</td><td>80.91</td><td>80.31</td><td>79.22</td><td>78.00</td><td>-2.91 -2.36</td></tr><tr><td>RollingForcing SF ODE</td><td>83.43</td><td></td><td>82.04</td><td>81.64</td><td>-1.99</td><td>80.23</td><td>80.83</td><td>79.00</td><td>78.59</td><td>-1.64</td><td>81.48</td><td>78.83</td><td>77.77</td><td>78.36</td><td>-3.12</td><td>80.21</td><td>79.07</td><td>78.61</td><td>77.85</td><td>-4.19</td></tr><tr><td>RollingForcing CF ODE InfinityStar</td><td>84.03</td><td>80.33 75.94</td><td>78.61 66.76</td><td>77.24 61.47</td><td>-6.19 -22.56</td><td>82.38</td><td>81.16 71.45</td><td>80.51 62.31</td><td>79.12 58.86</td><td>-3.26 -21.48</td><td>80.18 79.71</td><td>77.95 72.59</td><td>77.88 63.36</td><td>76.75 59.10</td><td>-3.43 -20.61</td><td>82.87 80.03</td><td>80.91 70.75</td><td>79.64 62.19</td><td>78.68 60.19</td><td>-19.84</td></tr><tr><td>Infinity-RoPE SF ODE</td><td>83.78</td><td>82.78</td><td>81.58</td><td>79.99</td><td>-3.79</td><td>80.34 81.04</td><td>80.59</td><td>79.74</td><td>79.18</td><td>-1.86</td><td>81.15</td><td>79.44</td><td>78.55</td><td>77.67</td><td>-3.48</td><td>81.09</td><td>79.70</td><td>78.65</td><td>77.07</td><td>-4.02</td></tr><tr><td>Infinity-RoPE CF ODE</td><td>84.08</td><td>82.29</td><td>80.71</td><td>79.51</td><td>-4.57</td><td>81.65</td><td>79.29</td><td>77.63</td><td>77.40</td><td>-4.25</td><td>80.13</td><td>77.95</td><td>77.22</td><td>76.88</td><td>-3.25</td><td>82.42</td><td>80.30</td><td>79.21</td><td>78.11</td><td>-4.31</td></tr><tr><td>RewardForcing</td><td>84.21</td><td>83.63</td><td>83.03</td><td>82.06</td><td>-2.15</td><td>82.06</td><td>81.55</td><td>81.20</td><td>80.73</td><td>-1.33</td><td>81.26</td><td>79.77</td><td>79.47</td><td>79.36</td><td>-1.90</td><td>81.56</td><td>80.96</td><td>80.63</td><td>80.19</td><td>-1.37</td></tr><tr><td></td><td>84.08</td><td>83.18</td><td>82.37</td><td>82.37</td><td>-1.71</td><td>81.05</td><td>80.54</td><td>80.29</td><td>79.89</td><td>-1.16</td><td>80.00</td><td>78.81</td><td>78.50</td><td>77.88</td><td>-2.12</td><td>81.05</td><td>79.93</td><td>78.57</td><td>78.17</td><td>-2.88</td></tr><tr><td>DeepForcing sF ODE</td><td>84.05</td><td>82.42</td><td>80.55</td><td>80.24</td><td>-3.81</td><td>81.75</td><td>79.90</td><td>77.28</td><td>77.22</td><td>-4.53</td><td>80.39</td><td>78.02</td><td>75.70</td><td>75.44</td><td>-4.95</td><td>82.36</td><td>80.05</td><td>77.87</td><td>76.71</td><td>-5.65</td></tr><tr><td>DeepForcing CF ODE CausalForcing</td><td>83.90</td><td></td><td></td><td></td><td></td><td>81.84</td><td></td><td></td><td></td><td></td><td>80.56</td><td></td><td></td><td></td><td></td><td>82.36</td><td></td><td></td><td></td><td></td></tr><tr><td>AnchorForcing</td><td>84.16</td><td>83.26</td><td>82.76</td><td>81.73</td><td>-2.43</td><td>80.92</td><td>81.19</td><td>80.74</td><td>80.41</td><td>-0.51</td><td>80.39</td><td>78.53</td><td>78.45</td><td>78.15</td><td>-2.24</td><td>81.30</td><td>80.97</td><td>80.47</td><td>79.81</td><td>-1.49</td></tr><tr><td>Memory Forcing</td><td>83.88</td><td>83.51</td><td>83.58</td><td>83.20</td><td>-0.68</td><td>82.41</td><td>82.13</td><td>81.89</td><td>81.99</td><td>-0.42</td><td>81.07</td><td>81.54</td><td>80.48</td><td>79.92</td><td>-1.15</td><td>82.28</td><td>81.55</td><td>80.98</td><td>80.66</td><td>-1.62</td></tr><tr><td>Memory Forcing (5B)</td><td>84.24</td><td>83.83</td><td>83.11</td><td>81.93</td><td>-2.31</td><td>81.34</td><td>80.69</td><td>80.43</td><td>79.77</td><td>-1.57</td><td>80.16</td><td>79.87</td><td>80.15</td><td>79.19</td><td>-0.97</td><td>81.34</td><td>80.63</td><td>79.91</td><td>79.16</td><td>-2.18</td></tr></table>

Table 2: Ablations on Wan2.1-T2V-1.3B (Wang et al., 2025a) scored on official VBench (Huang et al., 2024) at 30s and 60s. Every panel reports Total/Quality/Semantic at 16 FPS and $8 3 2 \times 4 8 0$ (a) Memory banks. (b) Bank-aware RoPE. (c) Similarity eviction.
<table><tr><td>Memory</td><td>Total</td><td>Quality</td><td>Semantic</td></tr><tr><td>Official</td><td>82.04</td><td>82.64</td><td>79.61</td></tr><tr><td>Causal-Forcing checkpoint</td><td>78.61</td><td>81.30</td><td>67.85</td></tr><tr><td>Working</td><td>82.41</td><td>83.46</td><td>78.22</td></tr><tr><td>Archive + working</td><td>83.58</td><td>84.57</td><td>79.63</td></tr></table>

## 4 EXPERIMENTS

<table><tr><td>RoPE</td><td>Total</td><td>Quality</td><td>Semantic</td></tr><tr><td>Contiguous τctg</td><td>81.31</td><td>82.37</td><td>77.10</td></tr><tr><td> $\mathrm { S t e p p e d } \tau ^ { \mathrm { s t p } }$ </td><td>82.56</td><td>83.37</td><td>79.29</td></tr><tr><td>Collapsed τcol</td><td>81.50</td><td>82.41</td><td>77.87</td></tr><tr><td>Interpolated τitp</td><td>83.58</td><td>84.57</td><td>79.63</td></tr></table>

<table><tr><td>Write</td><td>Total</td><td>Quality</td><td>Semantic</td></tr><tr><td>FIFO (30s)</td><td>83.39</td><td>84.59</td><td>78.59</td></tr><tr><td>Similarity (30s)</td><td>83.58</td><td>84.57</td><td>79.63</td></tr><tr><td>FIFO (60s)</td><td>82.96</td><td>83.94</td><td>79.05</td></tr><tr><td>Similarity (60s)</td><td>83.20</td><td>84.11</td><td>79.57</td></tr></table>

We report Memory Forcing at Wan2.1-T2V-1.3B and at Wan2.2 5B as two primary results. Ablations that isolate the partition, τ, or eviction stay on 1.3B.

## 4.1 IMPLEMENTATION DETAILS

Model. We implement Memory Forcing with Wan2.1-T2V-1.3B (Wang et al., 2025a) as the default 5s, 16 FPS, $8 3 2 \times 4 8 0$ base, and with Wan2.2-TI2V-5B (Wang et al., 2025a) as a second primary result. We adopt Causal Forcing (Zhu et al., 2026) weights from two stages, autoregressive diffusion training and causal ODE initialization. Text prompts are drawn from the filtered and LLMaugmented VidProM (Wang & Yang, 2024) dataset. We set T=4, 3 latent frames per chunk, and denoising steps [1000, 750, 500, 250]. Training uses 1,000 steps, batch size 32 on 32 A800 GPUs, and a window of 33 latent frames. The KV cache is partitioned into sink, archive, and working banks of 3, 6, and 12 frames. We use Bank-aware RoPE with default Interpolated $\tau ^ { \mathrm { i t p } }$ , and AdamW for both the generator $G _ { \theta }$ (learning rate $1 . 5 \times 1 0 ^ { - 6 } )$ and the fake score $s _ { \mathrm { g e n } }$ (learning rate $4 . 0 \times 1 0 ^ { - 7 } )$ . The generator is updated every 5 steps of fake score updates.

Evaluation. We evaluate on four prompt suites: official VBench (Huang et al., 2024) and MovieGen (Polyak et al., 2024), plus two harder sets, 20 VBench prompts as VBench-Complex and 30 LLM prompts as Gen-Complex. These fifty prompts emphasize multi-subject scenes, person-toperson interaction, and other mid-horizon difficulties. We generate each prompt at four durations, 5s (21 latent frames), 15s (63), 30s (120), and 60s (240), all at 16 FPS and $8 3 2 \times 4 8 0$ . We score videos with the VBench toolkit, reporting Total, Quality, and Semantic on official VBench and Quality only on MovieGen, VBench-Complex, and Gen-Complex.

## 4.2 COMPARISONS

Table 1 compares Memory Forcing with recent open streaming and long-video models on the four prompt suites and four durations in §4.1. The grid is meant to be read across settings, not as a single 5s contest. Where a method is publicly released on more than one initialization, we report both the Self-Forcing and CausalForcing weights so that a gap is not an artifact of the starting checkpoint. Although Memory Forcing is not the strongest row at 5s, it stays at the top of the table on the longer clips, and the drop from 5s to 60s is the smallest among methods that report all four lengths. That pattern is meant to show that mid-horizon history remains usable after the short window.

![](images/113bf62b70f3e25f3e59ed7aac59abb4cddae75ad4309a4a7c7c7f0b2b3998bc.jpg)  
Figure 5: Qualitative comparison at 1.3B against four recent open-source models. Memory Forcing keeps the subject’s identity and the scene layout, while the baselines drift or collapse.

![](images/84348a351c8b1be10cac1074949e767b09ec7dc351238b7b50006f4033e64e0f.jpg)  
Figure 6: Memory Forcing at Wan2.1-T2V-1.3B and Wan2.2 5B on the same prompts. The 5B run is more physically plausible, more realistic, and more dynamic.

## 4.3 ABLATIONS

Memory banks. Table 2a changes only the cache: official Rolling Forcing, a Causal-Forcing checkpoint, a working bank alone, and archive plus working. Archive plus working is the strongest row. Working alone is lower on Total, Quality, and Semantic, so the extra bank is a gain. Offi cial Rolling Forcing sits below it, and the Causal-Forcing checkpoint is lower still, especially on Semantic. A dedicated mid-horizon bank therefore keeps events that a flat window drops.

Bank-aware RoPE. Table 2b freezes the three-bank cache and swaps only the index map: Contiguous τ<sup>ctg</sup>, Stepped τ<sup>stp</sup>, Collapsed τ<sup>col</sup>, and default Interpolated τ<sup>itp</sup>. Interpolated τ<sup>itp</sup> is the strongest row. Stepped is second. Contiguous and Collapsed fall further, especially on Semantic. Archive history has to sit on (0, 1) with frames that remain distinguishable; packing it with the present, collapsing it onto the sink, or stepping the whole bank onto one integer all weaken retrieval.

Similarity eviction. Table 2c freezes the three-bank cache and τ<sup>itp</sup> and changes only which working chunk is written into archive. Similarity is the training default. FIFO is used only at inference, where it moves the oldest working chunk into archive. Similarity is the stronger of the two rows on Total, Quality, and Semantic. FIFO turns archive into a second recency queue, while Similarity moves the working chunk least like the incoming event so mid-horizon slots stay diverse.

## 4.4 QUALITATIVE RESULTS

We use qualitative frames to show Memory Forcing on its own samples, then in a 1.3B comparison, and then at Wan2.2 5B on harder scenes. Figure 4 shows Memory Forcing clips on diverse prompts, with identity and layout held across each clip. Figure 5 compares Memory Forcing at 1.3B with four recent open-source models. Frames are aligned across time on the same prompt. The baselines drift or collapse the subject’s identity and the scene layout. Memory Forcing keeps both, which is the point of keeping an archive. Figure 6 then shows the same method at Wan2.2 5B. On crowded and multi-subject scenes the 5B run keeps objects and interactions that the 1.3B rollout starts to lose.

## 5 CONCLUSION

We present Memory Forcing, a few-step streaming method for retrieving events after they leave the recent attention window. Its Archive & Working Banks retain a diverse mid-horizon history within a fixed cache, while Bank-aware RoPE gives the three banks distinct temporal positions and keeps that history accessible as generation proceeds. Together, these changes make long rollouts markedly more stable without growing attention cost with clip length. At 1.3B, Memory Forcing leads on longer clips, shows the smallest drop from 5s to 60s among methods with complete results, and preserves subjects and scenes through leave-and-return. We further show that the same design scales to Wan2.2 5B, producing more physically plausible, realistic, and dynamic videos and, to our knowledge, the first public 5B model on this forcing line.

## AI USE STATEMENT

In this work, we used generative AI tools to generate a synthetic evaluation prompt set and to implement parts of the method. We have not used generative AI tools to develop theoretical models, formulate mathematical claims, propose or refine hypotheses, design experiments, assist with translation, clean or reformat datasets, support qualitative analysis, or interpret results, and the writing of proofs is not applicable to this work. Additionally, we used generative AI tools to edit the manuscript for readability and to edit training and inference code. We have reviewed all AI-assisted work. The thirty Gen-Complex prompts were checked by the authors before scoring, and LLM-edited code was run and checked against the intended cache, write, and RoPE behavior. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work studies few-step streaming video generation. We do not recruit human subjects or collect a new personal-data corpus. Training text comes from public VidProM, and evaluation uses public VBench and MovieGen prompts together with a small author-checked set. Like other text-to-video models, Memory Forcing can be used to fabricate or extend footage of people and events, and we treat that misuse risk as the main ethical concern.

## REPRODUCIBILITY STATEMENT

We have already described the training and inference details in the main text. Code and model weights will be released after the paper is accepted.

## REFERENCES

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023a.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023b.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI technical report, 2024.

Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitz-´ mann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems, volume 37, pp. 24081–24125, 2024.

Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, and Enze Xie. SANA-Video: Efficient video generation with block linear diffusion transformer. CoRR, abs/2509.24695, 2025.

Liyang Chen, Tianxiang Ma, Jiawei Liu, Bingchuan Li, Zhuowei Chen, Lijie Liu, Xu He, Gen Li, Qian He, and Zhiyong Wu. Human-centric video generation via collaborative multi-modal conditioning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 2939–2947, 2026.

Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In International Conference on Learning Representations, 2025.

Zhixue Fang, Xu He, Songlin Tang, Haoxian Zhang, Qingfeng Li, Xiaoqiang Liu, Pengfei Wan, and Kun Gai. 3d-aware implicit motion control for view-adaptive human video generation. arXiv preprint arXiv:2602.03796, 2026.

Hang Guo, Zhaoyang Jia, Jiahao Li, Bin Li, Yuanhao Cai, Jiangshan Wang, Yawei Li, and Yan Lu. Efficient autoregressive video diffusion with dummy head. CoRR, abs/2601.20499, 2026.

Xu He, Qiaochu Huang, Zhensong Zhang, Zhiwei Lin, Zhiyong Wu, Sicheng Yang, Minglei Li, Zhiyi Chen, Songcen Xu, and Xiaofei Wu. Co-speech gesture video generation via motiondecoupled diffusion model. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2263–2273. IEEE, 2024.

Xu He, Zhiyong Wu, Xiaoyu Li, Di Kang, Chaopeng Zhang, Jiangnan Ye, Liyang Chen, Xiangjun Gao, Han Zhang, and Haolin Zhuang. Magicman: Generative novel view synthesis of humans with 3d-aware diffusion and iterative refinement. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 3437–3445, 2025.

Jiehui Huang, Yuechen Zhang, Xu He, Yuan Gao, Zhi Cen, Bin Xia, Yan Zhou, Xin Tao, Pengfei Wan, and Jiaya Jia. Unityvideo: Unified multi-modal multi-task learning for enhancing worldaware video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4471–4481, 2026.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In Advances in Neural Information Processing Systems, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Yicheng Ji, Zhizhou Zhong, Jun Zhang, Qin Yang, XiTai Jin, Ying Qin, Wenhan Luo, Shuiyang Mao, Wei Liu, and Huan Li. Forcing-KV: Hybrid KV cache compression for efficient autoregressive video diffusion models. CoRR, abs/2605.09681, 2026.

Youngrae Kim, Qixin Hu, C.-C. Jay Kuo, and Peter A. Beerel. MemRoPE: Training-free infinite video generation via evolving memory tokens. CoRR, abs/2603.12513, 2026.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. CoRR, abs/2412.03603, 2024.

Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limitedhorizon training and open-ended testing in autoregressive video diffusion. CoRR, abs/2602.07775, 2026.

Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. InfinityStar: Unified spacetime autoregressive modeling for visual generation. In Advances in Neural Information Processing Systems, 2025a.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. CoRR, abs/2509.25161, 2025b.

Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, Yujun Shen, and Min Zhang. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. CoRR, abs/2512.04678, 2025.

Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Bingyuan Wang, Qinghe Wang, Xuanhua He, Hongfa Wang, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025.

Yue Ma, Xinyu Wang, Qianli Ma, Qinghe Wang, Mingzhe Zheng, Xiangpeng Yang, Hao Li, Chongbo Zhao, Jixuan Ying, Harry Yang, et al. Group editing: Edit multiple images in one go. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 43418–43428, 2026.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media founda tion models. CoRR, abs/2410.13720, 2024.

Florian Schmidt. Generalization in generation: A closer look at exposure bias. CoRR, abs/1910.00292, 2019.

SkyReels Team. SkyReels-V2: Infinite-length film generative model. CoRR, abs/2504.13074, 2025.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. CoRR, abs/2503.20314, 2025a.

Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, and Di Zhang. Koala-36M: A largescale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025b.

Wenhao Wang and Yi Yang. VidProM: A million-scale real prompt-gallery dataset for text-to-video diffusion models. In Advances in Neural Information Processing Systems, volume 37, pp. 65618– 65642, 2024.

Xinyu Wang, Huafeng Shi, Zian Li, Yan Zhou, Xiaoqiang Liu, Yue Ma, and Pengfei Wan. Subjectanchor: Subject-aware memory-to-video for multi-shot storytelling. arXiv preprint arXiv:2609.34502, 2026a.

Xinyu Wang, Chongbo Zhao, Fangneng Zhan, and Yue Ma. Liveedit: Towards real-time diffusionbased streaming video editing. In European Conference on Computer Vision, pp. 631–649. Springer, 2026b.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. LongLive: Real-time interactive long video generation. CoRR, abs/2509.22622, 2025a.

Yang Yang, Tianyi Zhang, Wei Huang, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, and Peng-Tao Jiang. Anchor forcing: Anchor memory and tri-region RoPE for interactive streaming video diffusion. CoRR, abs/2603.13405, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025b.

Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-RoPE: Action-controllable infinite video generation emerges from autoregressive self-rollout. CoRR, abs/2511.20649, 2025.

Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. CoRR, abs/2512.05081, 2025.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and ¨ Bill Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, volume 37, pp. 47455–47487, 2024.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Ruicheng Zhang, Mingyang Zhang, Jun Zhou, Zhangrui Guo, Xiaofan Liu, Zunnan Xu, Zhizhou Zhong, Puxin Yan, Haocheng Luo, and Xiu Li. Mind-v: Hierarchical video generation for longhorizon robotic manipulation with rl-based physical alignment. arXiv preprint arXiv:2512.06628, 2025.

Ruicheng Zhang, Guangyu Chen, Zunnan Xu, Zihao Liu, Zhizhou Zhong, Mingyang Zhang, Jun Zhou, and Xiu Li. Robostereo: Dual-tower 4d embodied world models for unified policy optimization. arXiv preprint arXiv:2603.12639, 2026a.

Ruicheng Zhang, Kaixi Cong, Jun Zhou, Zhizhou Zhong, Zunnan Xu, Shuiyang Mao, Wei Liu, and Xiu Li. Kvpo: Ode-native grpo for autoregressive video alignment via kv semantic exploration. arXiv preprint arXiv:2605.14278, 2026b.

Ruicheng Zhang, Jun Zhou, Zunnan Xu, Zihao Liu, Jiehui Huang, Mingyang Zhang, Yu Sun, and Xiu Li. Zo3t: Zero-shot 3d-aware trajectory-guided image-to-video generation via test-time training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 12708– 12716, 2026c.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. CoRR, abs/2602.02214, 2026.