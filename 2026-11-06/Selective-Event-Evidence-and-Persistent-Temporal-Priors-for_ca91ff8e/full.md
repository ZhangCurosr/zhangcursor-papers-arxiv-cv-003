# Selective Event Evidence and Persistent Temporal Priors for Event-Based Point Tracking

Xiaokai Song, Zhang Li, Leqi Liu, Yibin Ye, Kun Wang, and Xichao Teng

Abstract—Event cameras enable tracking under fast motion and challenging illumination. However, stable long-term point tracking remains challenging: event activity is sparse and varies with scene content and motion, complicating correspondence, while trajectory estimates can drift over long sequences. Together, these challenges motivate a tracker that selectively uses informative event evidence and retains trajectory constraints across temporal windows. We propose Selective Event Evidence and Persistent Temporal Priors (SEPT), an event-only framework for tracking any point (TAP) over long sequences. SEPT introduces two complementary modules: Task-Adaptive Event Evidence Modulation (TEEM) and Persistent Trajectory Prior (PTP). Using only tracking supervision, TEEM learns a bounded reliability field that weights the event tensor across space, time, and channels before feature extraction. PTP carries geometric information across overlapping windows by marginalizing a fixed local quadratic trajectory model with a Schur complement. Its visibility-aware correction refines overlap coordinates, while a clipped terminal velocity initializes coordinates at newly exposed timestamps. SEPT builds on the ETAP architecture. SEPT is trained exclusively on synthetic EventKubric data and evaluated on five datasets across TAP and feature tracking. Results show that SEPT outperforms the retrained ETAP baseline in average Jaccard for TAP and in feature age and expected feature age for feature tracking; it also surpasses several methods combining event and frame inputs on EC and EDS. The source code is publicly available at https://github.com/sssxxxkkkk/SEPT.

Index Terms—Event cameras, tracking any point, point tracking, event representation, temporal prior, Schur complement.

## I. INTRODUCTION

Tracking any point (TAP) estimates the trajectory and visibility of an arbitrary query throughout a sequence [1]. Such tracks provide a geometric substrate for tasks including augmented reality, robot manipulation, visual odometry, and dynamic-scene analysis [1], [2], [3], [4]. The field has progressed from local feature correspondence to joint reasoning over many queries and time steps: TAP-Vid [1] formalized a broad evaluation protocol, while PIPs [5], TAPIR [6], and CoTracker [7], [8] refine trajectories using temporal context, matching, or collective motion cues. Nevertheless, framebased observations can become unreliable when exposure causes blur, illumination saturates the sensor, or inter-frame motion is large [5], [6], [9].

Event cameras provide a complementary sensing modality. Instead of capturing complete images at a fixed rate, these sensors asynchronously emit brightness-change events, providing high temporal resolution, high dynamic range, and minimal motion blur [10], [11]. These properties have enabled eventbased feature tracking and continuous trajectory estimation in regimes that are difficult for standard cameras [12], [13], [14], [15]. They do not make correspondence automatic, however. Event activity is sparse and depends jointly on scene gradients, relative motion, contrast sensitivity, and the temporal interval used to form a dense representation. Thus, the event pattern produced by one surface can change substantially across motions and time scales.

This observation motivates event-based TAP, which seeks the broad query coverage of modern TAP while retaining the sensing advantages of events. ETAP [16] demonstrates that event-only tracking can be built on motion-robust correlations and feature alignment, while MATE [17] adds explicit motion guidance for variable dynamics. A remaining issue is that an accumulated event tensor mixes persistent correspondence cues with responses that may be weak, transient, or poorly localized. We therefore investigate whether tracking supervision alone can guide a tracker to emphasize the event responses most useful for correspondence. This motivates our first component, Task-Adaptive Event Evidence Modulation (TEEM).

Long-term tracking introduces a second challenge. Events arrive continuously, whereas modern joint trackers must use bounded temporal windows to control the cost of feature encoding, correlation construction, and transformer refinement [5], [7]. Sliding windows with iterative updates are a practical solution, and overlapping windows allow a tracker to carry forward current coordinate estimates [7], [18]. Yet a window transition still truncates the optimization history. States that leave the active window are discarded, even though they constrain the trajectory in the overlap and may remain useful for subsequent refinement. Simply copying an overlap estimate provides an initialization, but it does not preserve a structured account of the information contributed by the discarded states. Recurrently copying high-dimensional latent features is also not necessarily reliable, because those features are continually updated in a window-specific correlation context. This motivates our second component, the Persistent

![](images/a69863e41bd8070daef337b61ddbc34e1e606abb0cad7bd72ea41563588c33e2.jpg)  
Fig. 1. Bottlenecks and corresponding solutions in event-based point tracking. (a) Sparse, nonuniform event activity yields competing local matches; TEEM learns task-adaptive event modulation before feature extraction. (b) As the tracking window advances, historical constraints are discarded and trajectories ma drift; PTP carries a Schur-style trajectory prior into the next window.

Trajectory Prior (PTP), which retains trajectory information without requiring unbounded back-propagation or a large learned memory.

As illustrated in Fig. 1, Selective Event Evidence and Persistent Temporal Priors (SEPT) addresses these two issues with an event-only extension of the ETAP [16] tracking architecture. TEEM predicts a bounded reliability field over the input event tensor. It combines spatiotemporal summary statistics—signed response, response magnitude, and temporal variation—with channel-wise statistics. The resulting weight is applied before event feature extraction. TEEM is trained using only point-tracking supervision: it receives no contour labels, segmentation masks, or event-importance annotations. Its zero-initialized final projections make TEEM an identity mapping at the start of training, allowing it to learn selective modulation without perturbing the initial event representation.

PTP preserves trajectory-level information across overlapping windows. For a completed window, we construct a small quadratic model over coordinate perturbations with observation and second-order temporal-smoothness terms. We partition its states into those that leave the window and those that remain in the overlap, and analytically eliminate the former using a Schur complement. The resulting precision matrix provides a compact prior on the retained coordinates. In the next window, stored overlap coordinates and visibility logits initialize the recurrent state, while a clipped terminal velocity initializes coordinates at newly exposed timestamps. The prior then applies a visibility-aware correction at every coordinaterefinement iteration. This is a Schur-style prior rather than a full bundle-adjustment marginalization: it is derived from a fixed trajectory model, not from a data-dependent linearization of visual residuals.

The two components operate at complementary information bottlenecks. TEEM improves the quality of observation evidence before it reaches the encoder and correlation module. PTP improves the persistence of solved geometric information after a temporal window shifts. This separation is useful in practice. Better event evidence can produce a more reliable overlap trajectory, while a structured temporal prior can reduce boundary discontinuities in regions with sparse event evidence. Neither mechanism replaces the other.

Our contributions are threefold:

• We introduce TEEM, a lightweight, identity-initialized module that learns coupled spatiotemporal and channelwise weights for event evidence from downstream tracking supervision.

• We introduce PTP, a parameter-free fixed-lag trajectory prior that preserves the influence of expired window states through a Schur complement and applies a visibilityaware, preconditioned coordinate correction during iterative tracking.

• We evaluate SEPT under an event-only training protocol on five datasets for TAP and feature tracking. We analyze the contributions of TEEM and PTP and the sensitivity to training-data allocation.

The remainder of the paper is organized as follows. Section II reviews related work, Section III presents SEPT, Section IV reports experimental results, Section V discusses limitations and future directions, and Section VI concludes the paper.

## II. RELATED WORK

## A. Event-Based Feature Point Tracking

Early event-based tracking focuses on a sparse set of salient features. One family of methods groups nearby events in the $( x , y , t )$ domain and estimates a feature state from each cluster. For example, eCDT [19] clusters adjacent events with the same polarity and matches cluster descriptors to extend feature tracks directly from the event stream. By contrast, HOTS [20] encodes the recency of activity at each pixel, yielding local spatiotemporal patterns for comparison or clustering. A related point-set formulation builds an edge or event template around a detected feature and aligns incoming events through ICP registration [21], [22]. EKLT [12] combines event-driven updates with frame-based feature detection, using the stable photometric information in frames and the high temporal resolution of events. RATE [23] combines asynchronous feature initialization with track management for realtime operation. Learned methods instead use local features and association functions, as in DDFT [14], EventPoint [24], and 3D Feature Tracking via Event Camera [15]. At the object level, DANet [25] uses event motion and contour cues, while CrossEI [26] fuses event and frame cues. These target-tracking formulations differ from arbitrary-query point tracking.

Dense optical-flow estimation provides a complementary representation by predicting a displacement field over the image or active pixels [27], [28]. It is useful for short-term motion, but it does not by itself maintain the identity and visibility of an arbitrary query. Moreover, flow remains susceptible to the aperture problem and occlusion, and its estimate depends on the temporal interval used to aggregate events. Errors can therefore accumulate when local flow estimates are chained into long trajectories. These limitations motivate a tracker that explicitly maintains query-conditioned trajectories rather than only dense instantaneous motion.

## B. Tracking Any Point

TAP extends sparse feature tracking to arbitrary query points, including those in weakly textured or non-corner regions and those that become temporarily occluded, by estimating their trajectories and visibility. TAP-Vid [1] established a unified benchmark and evaluation protocol for this setting. Frame-based methods have subsequently evolved from local matching to iterative trajectory refinement and joint reasoning across time and tracks. PIPs [5] refines point trajectories over a sequence rather than treating frame pairs independently, PIPs++ [29] extends the temporal receptive field and uses multiple appearance templates for long-term tracking, and TAPIR [6] adds per-frame initialization and temporal refinement. CoTracker [7], [8] further exploits interactions among many trajectories and uses causal, overlapping windows to support efficient online inference.

Applying TAP to event cameras is nontrivial. Unlike frame features, events provide only brightness changes; their spatial density varies with texture and motion, and the same surface can generate different event patterns under different velocities or directions. ETAP [16] established event-only TAP with motion-robust correlation features and feature alignment. MATE [17] adds motion guidance and variable-motion-aware processing to reduce ambiguities in sparse event observations. These works establish the feasibility of event-based TAP, but responses within an accumulated event tensor differ in how reliably they support point correspondence. TEEM addresses this issue before feature extraction by learning a bounded, task-driven weighting over the event representation, without requiring labels for event types or manually designed importance rules.

## C. Long-Term Feature and Trajectory Tracking

Long-term tracking must preserve identity when a point moves far from its query location, becomes briefly unobservable, or crosses successive processing windows. Classical feature trackers typically propagate a local estimate and can lose a track once the appearance model or search region is no longer valid. Modern point trackers alleviate this failure in different ways. PIPs and TAPIR repeatedly refine trajectories using temporal context; PIPs++ uses multiple appearance templates and temporally flexible refinement; and CoTracker couples tracks through attention and trains its causal model using windows unrolled over longer sequences [5], [29], [6], [7], [8]. In windowed trackers, the state passed to the next window is often limited to a coordinate initialization or a learned latent representation with implicit geometric meaning.

Fixed-lag estimators provide a complementary principle: when old states leave a window, their information can be marginalized into a prior on the retained states rather than discarded [30], [31], [32], [33]. PTP adopts this idea at the trajectory level. It retains overlap coordinates, visibility logits, and a clipped terminal velocity. It uses a small Schurstyle precision matrix obtained by marginalizing a fixed local quadratic trajectory model. The resulting prior corrects the retained coordinates throughout the next window’s refinement. Compared with high-dimensional feature memory, this design gives the transferred state an explicit geometric meaning, keeps its size bounded per track, and avoids an unbounded gradient path. It is intentionally not a replacement for learned correspondence: new event evidence remains responsible for data association, while PTP supplies continuity when the finite window would otherwise forget solved geometric constraints.

## III. METHOD

We first formulate event-based TAP over overlapping windows, then describe the SEPT architecture and its two components.

## A. Problem Formulation

An event camera produces an asynchronous stream $\varepsilon =$ $\left\{ \boldsymbol { e } _ { k } \right\}$ , where $\begin{array} { r c l } { e _ { k } } & { = } & { \left( x _ { k } , y _ { k } , \tau _ { k } , p _ { k } \right) } \end{array}$ records a brightness change at pixel $( x _ { k } , y _ { k } )$ with timestamp $\tau _ { k }$ and polarity $p _ { k } \in \{ - 1 , + 1 \}$ . We evaluate trajectories at discrete timestamps $\{ \tau _ { t } \} _ { t = 1 } ^ { T }$ . Following ETAP [16], events preceding each timestamp are converted into a mixed-density stack with C channels. Each channel aggregates a nested subset of recent events, providing complementary temporal support. After channel-wise normalization over nonzero entries, the input is denoted by $\mathbf { V } \in \mathbb { R } ^ { T \times C \times H \times W }$ ; the batch dimension is omitted throughout this section.

Given N queries ${ \mathcal Q } = \{ ( t _ { i } , \mathbf q _ { i } ) \} _ { i = 1 } ^ { N }$ , with $\mathbf { q } _ { i } ~ \in \mathbb { R } ^ { 2 }$ , the tracker predicts coordinates $\hat { \mathbf { x } } _ { t , i }$ and visibility probabilities $\hat { v } _ { t , i }$ for $t \geq t _ { i } . \mathrm { ~ A ~ }$ query identifies the same physical surface point throughout the sequence, including times when that point is occluded. Processing uses windows ${ \mathcal W } _ { w } = \{ a _ { w } , \dots , a _ { w } + S -$ 1 , where $a _ { w } = 1 + w s , S$ is the window length, and $s < S$ is the window stride. The overlap between consecutive windows therefore has length $K = S - s .$ . In our experiments, $C = 1 0$ $S = 8$ , and $s = 4$ , giving $K = 4 ;$ the remaining numerical settings are reported in Sec. IV-A.

![](images/27f5d4f8c29b6574a092cbb3db7658aa6245a78dc3b3f53e88904d089f7b6f2f.jpg)  
Fig. 2. Overview of SEPT. TEEM learns spatiotemporal and channel-wise weights for the event tensor before feature extraction. The tracker iteratively refines query trajectories using local correlations and joint attention. At a window transition, PTP transfers overlap coordinates, visibility logits, and a clipped terminal displacement; its fixed Schur precision couples overlap-coordinate deviations during refinement in the next window. Refined latent descriptors are not reused across windows.

## B. Overview of SEPT

Given the event stack, queries, and overlapping windows defined above, Fig. 2 shows how SEPT processes each window and transfers information to the next. TEEM modulates the event stack before the ETAP encoder constructs a feature pyramid. Query-conditioned correlations and a joint space–time transformer then iteratively refine coordinates and descriptors within each window. At a window transition, PTP initializes the next window from the completed trajectory and corrects the overlapping coordinates at each refinement iteration.

1) Query-conditioned correspondence: Let $\widetilde { \mathbf { V } }$ denote the modulated input. An encoder ϕ and successive average pooling produce features $\mathbf { F } _ { t } ^ { l }$ at pyramid levels $l = 0 , \ldots , L - 1$ . If the encoder stride is $d ,$ feature-space coordinates are $\begin{array} { r } { \mathbf { z } _ { t , i } = \mathbf { x } _ { t , i } / d . } \end{array}$ The initial descriptor for query i is bilinearly sampled from $\mathbf { F } _ { t _ { i } } ^ { 0 }$ at $\mathbf { q } _ { i } / d$ and replicated across the active window. At refinement iteration m, its current descriptor $\mathbf { f } _ { t , i } ^ { ( m ) } \in \mathbb { R } ^ { D }$ is correlated with features around the current coordinate:

$$
c _ { t , i } ^ { l , ( m ) } ( \delta ) = \frac { \left. \mathbf { f } _ { t , i } ^ { ( m ) } , \mathbf { F } _ { t } ^ { l } \Big ( \mathbf { z } _ { t , i } ^ { ( m ) } / 2 ^ { l } + \delta \Big ) \right. } { \sqrt { D } } ,\tag{1}
$$

where $\delta \in \{ - r , \ldots , r \} ^ { 2 }$ and noninteger locations are bilinearly sampled. Concatenating these values over offsets and levels gives the correlation descriptor $\mathbf { c } _ { t , i } ^ { ( m ) }$

2) Iterative trajectory refinement: The update transformer receives correlation descriptors, displacement embeddings relative to the first time in the window, current point descriptors, query-support indicators, and the window’s initial visibilitylogit state. Spatial and temporal positional encodings are added to these tokens. Joint attention across trajectories and timestamps produces coordinate and descriptor increments:

$$
\begin{array} { r l } & { ( \Delta \mathbf { z } ^ { ( m ) } , \Delta \mathbf { f } ^ { ( m ) } ) = \Phi _ { \boldsymbol { \theta } } ( \mathbf { h } ^ { ( m ) } ) , } \\ & { \qquad \mathbf { z } ^ { ( m + 1 / 2 ) } = \mathbf { z } ^ { ( m ) } + \Delta \mathbf { z } ^ { ( m ) } , } \\ & { \qquad \mathbf { f } ^ { ( m + 1 ) } = \mathbf { f } ^ { ( m ) } + \mathcal { U } ( \Delta \mathbf { f } ^ { ( m ) } ) , } \end{array}\tag{2}
$$

where $\mathbf { h } ^ { ( m ) }$ denotes the assembled tokens and is the descriptor-update normalization and multilayer perceptron (MLP). PTP converts the intermediate coordinates $\mathbf { z } ^ { ( m \bar { + } 1 / 2 ) }$ into $\mathbf { z } ^ { ( m + 1 ) }$ as derived in Sec. III-D. After M iterations, coordinates are rescaled to image pixels, and a linear prediction head maps the final descriptors to visibility logits $\ell _ { t , i } ,$ with $\hat { v } _ { t , i } = \sigma ( \ell _ { t , i } )$ . Queries that have not been introduced by the end of the active window are excluded from joint attention.

## C. Task-Adaptive Event Evidence Modulation

Event activity depends on local contrast, motion, and accumulation time. Consequently, a strong event response need not provide a reliable correspondence, and the relative utility of representation channels can change over time. TEEM learns a task-dependent modulation before feature encoding, using complementary local event statistics and global channel statistics.

1) Spatiotemporal and channel statistics: For each input time, we construct three channel-reduced maps:

$$
\begin{array} { l } { { \displaystyle S _ { t } ^ { \mathrm { s g n } } ( x , y ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } V _ { t c x y } , } } \\ { { \displaystyle S _ { t } ^ { \mathrm { m a g } } ( x , y ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } | V _ { t c x y } | , } } \\ { { \displaystyle S _ { t } ^ { \mathrm { d i f } } ( x , y ) = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } | V _ { t c x y } - V _ { t - 1 , c x y } | . } } \end{array}\tag{3}
$$

The difference map is zero at the first input time. The signed and magnitude maps describe complementary aspects of event activity, while the difference map captures changes between adjacent stacks. Their concatenation $\textbf { S } \in ~ \mathbb { R } ^ { T \times 3 \times H \times W }$ is spatially average-pooled before entering a small 3-D convolutional network:

$$
\mathbf { a } = \pmb { \mathcal { I } } ( \psi _ { \mathrm { s t } } ( \mathcal { P } ( \mathbf { S } ) ) ) .\tag{4}
$$

Here, $\mathcal { P }$ pools spatial dimensions; $\psi _ { \mathrm { s t } }$ comprises a $3 \times 3 \times 3$ convolution, a Gaussian error linear unit (GELU) activation, and ${ \mathrm { ~ a ~ 1 ~ } } \times { \mathrm { ~ 1 ~ } } \times { \mathrm { ~ 1 ~ } }$ projection; and interpolates to the input resolution. The output $\boldsymbol { a } _ { t x y }$ is shared by all channels.

In parallel, we compute per-channel spatial means,

$$
g _ { t c } ^ { \mathrm { s g n } } = \frac { 1 } { H W } \sum _ { x , y } V _ { t c x y } , \qquad g _ { t c } ^ { \mathrm { m a g } } = \frac { 1 } { H W } \sum _ { x , y } | V _ { t c x y } | .\tag{5}
$$

A two-layer MLP maps the concatenated 2C statistics at each time to channel logits $\mathbf { b } _ { t } ~ \in ~ \mathbb { R } ^ { C }$ . Thus, the spatiotemporal branch models local activity and its temporal variation, while the channel branch adjusts the balance among the stack’s temporal supports.

2) Residual modulation and scale normalization: The two branches are combined through a bounded residual gate,

$$
\widehat { W } _ { t c x y } = 1 + \alpha \operatorname { t a n h } ( a _ { t x y } + b _ { t c } ) , \qquad 0 \leq \alpha \leq 1 .\tag{6}
$$

The unnormalized gate satisfies $1 - \alpha \leq \widehat { W } \leq 1 + \alpha$ . To stabilize the aggregate input scale, we normalize it over active entries. Let $\Omega _ { t } = \{ ( c , x , y ) : V _ { t c x y } \neq 0 \}$ and define

$$
\nu _ { t } = \left\{ \begin{array} { l l } { | \Omega _ { t } | ^ { - 1 } \displaystyle \sum _ { ( c , x , y ) \in \Omega _ { t } } \widehat W _ { t c x y } , } & { | \Omega _ { t } | > 0 , } \\ { ( C H W ) ^ { - 1 } \displaystyle \sum _ { c , x , y } \widehat W _ { t c x y } , } & { | \Omega _ { t } | = 0 . } \end{array} \right.\tag{7}
$$

The modulated representation is

$$
W _ { t c x y } = \frac { \widehat { W } _ { t c x y } } { \operatorname* { m a x } ( \nu _ { t } , \varepsilon _ { \mathrm { w } } ) } , \qquad \widetilde { V } _ { t c x y } = W _ { t c x y } V _ { t c x y } .\tag{8}
$$

The bound in Eq. (6) applies before normalization; Eq. (8) retains relative modulation while stabilizing the mean gain on active responses. Multiplication also preserves zero entries in the input. Both output projections are initialized to zero, so that ${ \widehat { W } } = { \bf \bar { W } } = 1$ initially. TEEM therefore begins as an identity mapping and learns to redistribute event evidence through downstream tracking supervision, without event-importance annotations.

## D. Persistent Trajectory Prior

Overlapping windows allow coordinates to be copied into the next window, but copying alone does not define how subsequent updates should respect the previous trajectory. PTP supplies this missing constraint with a compact quadratic prior on overlap deviations. It combines a fixed temporal precision with the completed window’s coordinates and predicted visibility.

1) Marginalizing a local trajectory model: Consider one point and one coordinate component in a completed window. Let $\mathbf { r } \in \mathbb { R } ^ { S }$ denote a perturbation about its refined trajectory. We define

$$
\begin{array} { l } { { \displaystyle { \cal E } ( { \bf r } ) = \frac { \lambda _ { o } } { 2 } \| { \bf r } \| _ { 2 } ^ { 2 } + \frac { \lambda _ { s } } { 2 } \| { \bf D } _ { 2 } { \bf r } \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle ~ = \frac { 1 } { 2 } { \bf r } ^ { \top } { \bf H } { \bf r } } , \quad { \bf H } = \lambda _ { o } { \bf I } + \lambda _ { s } { \bf D } _ { 2 } ^ { \top } { \bf D } _ { 2 } , } \end{array}\tag{9}
$$

where $( \mathbf { D } _ { 2 } \mathbf { r } ) _ { j } = r _ { j } - 2 r _ { j + 1 } + r _ { j + 2 } , \lambda _ { o } > 0$ , and $\lambda _ { s } \geq 0$ . The first term anchors the completed solution; the second couples changes to that solution across time. Because the model acts on perturbations, it does not require the reference trajectory itself to have constant velocity.

Partition $\mathbf { r } = [ \mathbf { r } _ { o } ^ { \top } , \mathbf { r } _ { r } ^ { \top } ] ^ { \top }$ into the first s states, which leave the window, and the last K states, which remain. Minimization over the expired states gives

$$
\mathbf { r } _ { o } ^ { * } = - \mathbf { H } _ { o o } ^ { - 1 } \mathbf { H } _ { o r } \mathbf { r } _ { r } .\tag{10}
$$

Substituting this solution into Eq. (9) yields the reduced energy

$$
\begin{array} { r l } & { E _ { \mathrm { r e d } } ( \mathbf { r } _ { r } ) = \frac { 1 } { 2 } \mathbf { r } _ { r } ^ { \top } \mathbf { P r } _ { r } , } \\ & { \qquad \mathbf { P } = \mathbf { H } _ { r r } - \mathbf { H } _ { r o } \mathbf { H } _ { o o } ^ { - 1 } \mathbf { H } _ { o r } . } \end{array}\tag{11}
$$

Equivalently, P is the precision obtained by marginalizing the zero-mean Gaussian associated with Eq. (9). This reduction preserves the temporal coupling specified by the fixed quadratic model. The resulting precision is shared across points and is independent of visual residuals; the scenedependent reference comes from the completed trajectory. Since the model and window geometry are fixed, P and its diagonal preconditioner are computed once. Fig. 3 summarizes the state partition, Schur elimination, and overlap-only correction applied during iterative refinement.

2) State transfer and window initialization: Let $\hat { \mathbf { z } } _ { j , i }$ and $\hat { \ell } _ { j , i }$ denote the completed window’s feature-space coordinates and visibility logits at local time j. PTP retains

$$
\bar { \bf z } _ { j , i } = \hat { \bf z } _ { s + j , i } , \qquad \bar { \boldsymbol { \ell } } _ { j , i } = \hat { \ell } _ { s + j , i } , \quad j = 1 , \ldots , K .\tag{12}
$$

It also computes a clipped terminal displacement

$$
\begin{array} { r } { \mathbf { u } _ { i } = \mathrm { c l i p } _ { u _ { \mathrm { m a x } } } \big ( \hat { \mathbf { z } } _ { S , i } - \hat { \mathbf { z } } _ { S - 1 , i } \big ) , } \end{array}\tag{13}
$$

where $\mathrm { c l i p } _ { u _ { \mathrm { m a x } } }$ limits the vector’s Euclidean norm to $u _ { \mathrm { m a x } } .$ For a query carried from the preceding window, the next coordinate initialization is

$$
\mathbf { z } _ { j , i } ^ { ( 0 ) } = \left\{ \bar { \mathbf { z } } _ { j , i } , \begin{array} { l l } { 1 \leq j \leq K , } \\ { \bar { \mathbf { z } } _ { K , i } + ( j - K ) \mathbf { u } _ { i } , } & { K < j \leq S . } \end{array} \right.\tag{14}
$$

Visibility logits are copied to the overlap, and the terminal logit is repeated at newly exposed timestamps. New queries retain the query-based initialization. Stored coordinates, logits, and terminal displacements are detached at the boundary, and refined latent descriptors are not propagated into the next window. This keeps the transferred state bounded per track while retaining a geometric reference for the next refinement.

![](images/9fafea49ac9550f58c0ba202590f1b7072ed8561bd2aecb0d07f86a5cca8a065.jpg)  
Fig. 3. Schur-style persistent trajectory prior. Expired states are eliminated from a fixed quadratic perturbation model, leaving precision P over the retained overlap. The next window copies overlap coordinates and visibility logits and initializes coordinates at newly exposed timestamps using a clipped terminal displacement. After each transformer update, a validity-masked and visibility-weighted preconditioned gradient softly corrects only overlap coordinates; coordinates at newly exposed timestamps receive no prior correction. State counts are illustrative.

3) Visibility-aware overlap correction: PTP also constrains the overlap after every transformer update. For point i, collect the current intermediate overlap coordinates into $\mathbf { Z } _ { i } ^ { ( m + 1 / 2 ) } \in$ $\mathbb { R } ^ { K \times 2 }$ and the stored reference into $\bar { \bf Z } _ { i }$ . Let M<sub>i</sub> be a diagonal validity mask: an entry is one only when its timestamp lies in the sequence and is no earlier than the query time. The residual and its masked prior energy are

$$
\begin{array} { l } { { \displaystyle { \bf R } _ { i } ^ { ( m ) } = { \bf Z } _ { i } ^ { ( m + 1 / 2 ) } - \bar { \bf Z } _ { i } } , } \\ { { \displaystyle E _ { i } ^ { ( m ) } = \frac { 1 } { 2 } \mathrm { t r } \Big ( ( { \bf R } _ { i } ^ { ( m ) } ) ^ { \top } { \bf M } _ { i } { \bf P } { \bf M } _ { i } { \bf R } _ { i } ^ { ( m ) } \Big ) } . } \end{array}\tag{15}
$$

The gradient is $\mathbf { M } _ { i } \mathbf { P M } _ { i } \mathbf { R } _ { i } ^ { ( m ) }$ . We precondition this gradient by the diagonal of P and scale it using the previous window’s visibility:

$$
\begin{array} { r l } & { \mathbf { D } _ { P } = \mathrm { d i a g } \big ( \{ \mathrm { m a x } ( P _ { j j } , \varepsilon _ { \mathrm { p } } ) \} _ { j = 1 } ^ { K } \big ) , } \\ & { \mathbf { C } _ { i } = \mathrm { d i a g } \big ( \{ c _ { \mathrm { m i n } } + ( 1 - c _ { \mathrm { m i n } } ) \sigma ( \bar { \ell } _ { j , i } ) \} _ { j = 1 } ^ { K } \big ) . } \end{array}\tag{16}
$$

The resulting coordinate correction is

$$
\Delta \mathbf { Z } _ { i , \mathrm { p r i o r } } ^ { ( m ) } = - \eta \mathbf { C } _ { i } \mathbf { D } _ { P } ^ { - 1 } \mathbf { M } _ { i } \mathbf { P } \mathbf { M } _ { i } \mathbf { R } _ { i } ^ { ( m ) } ,\tag{17}
$$

where $\eta$ controls the update magnitude. The overlap update is

$$
\begin{array} { r } { \mathbf { Z } _ { i } ^ { ( m + 1 ) } = \mathbf { Z } _ { i } ^ { ( m + 1 / 2 ) } + \Delta \mathbf { Z } _ { i , \mathrm { p r i o r } } ^ { ( m ) } . } \end{array}\tag{18}
$$

Coordinates outside the overlap receive only the transformer update. When no preceding window is available, the prior correction is zero.

The visibility factor weakens the constraint on previously uncertain tracks, while $c _ { \operatorname* { m i n } } > 0$ retains a minimum influence. Retaining logits for initialization and applying the sigmoid only for confidence weighting keeps the transferred state in consistent units. The prior is soft: new event evidence can move coordinates away from the stored trajectory. PTP introduces no trainable parameters, and its per-window correction costs $O ( N K ^ { 2 } )$ with a shared $K \times K$ precision matrix.

## E. Training Objective

SEPT is supervised by ground-truth coordinates $\mathbf { x } _ { t , i } ^ { * }$ and visibility labels $y _ { t , i } ~ \in ~ \{ 0 , 1 \}$ . Let $A _ { w }$ contain the valid point–time pairs in window w that are no earlier than the corresponding query, and let $n _ { w } = \operatorname* { m a x } ( | A _ { w } | , 1 )$ . Occluded coordinates receive weight $\omega _ { t , i } ~ = ~ y _ { t , i } + \beta ( 1 - y _ { t , i } )$ , with $\beta = 0 . 2$ . For $N _ { w }$ training windows and M refinements, the coordinate loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t r a c k } } = \displaystyle \frac { 1 } { N _ { w } M } \sum _ { w = 1 } ^ { N _ { w } } \sum _ { m = 1 } ^ { M } \frac { \gamma ^ { M - m } } { n _ { w } } } \\ & { \qquad \cdot \sum _ { ( t , i ) \in \mathcal { A } _ { w } } \omega _ { t , i } \frac { \| \hat { { \mathbf { x } } } _ { t , i } ^ { w , ( m ) } - { \mathbf { x } } _ { t , i } ^ { * } \| _ { 1 } } { 2 } , } \end{array}\tag{19}
$$

where $\gamma \ = \ 0 . 8$ gives greater weight to later refinements. The factor $1 / 2$ averages the two coordinate components. Each window’s final visibility prediction is supervised by binary cross-entropy (BCE):

$$
\mathcal { L } _ { \mathrm { v i s } } = \frac { 1 } { N _ { w } } \sum _ { w = 1 } ^ { N _ { w } } \frac { 1 } { n _ { w } } \sum _ { ( t , i ) \in \mathcal { A } _ { w } } \mathrm { B C E } ( \hat { v } _ { t , i } ^ { w } , y _ { t , i } ) .\tag{20}
$$

Probabilities are clipped away from zero and one for numerical stability. The total objective is

$$
\begin{array} { r } { \mathcal { L } = 0 . 1 \mathcal { L } _ { \mathrm { t r a c k } } + \mathcal { L } _ { \mathrm { v i s } } . } \end{array}\tag{21}
$$

Supervision is applied to each window, including its overlap. TEEM receives gradients through event encoding and tracking, while PTP modifies the forward coordinate refinement without an additional prior loss. Coordinates are detached between refinement iterations, and carried states are detached between windows. Training uses the same coordinate and visibility supervision as our retrained baseline, without time-inverted event pairs or contrastive feature-alignment supervision [16].

## IV. EXPERIMENTS

We evaluate SEPT on two complementary tasks: TAP and feature tracking. After describing the implementation and

synthetic training setup, we report results for both tasks and analyze the contributions of TEEM and PTP.

## A. Implementation Details

We train on three NVIDIA GeForce RTX 4090 GPUs, each with 24 GB of memory, using distributed data parallelism. The batch size is one sample per GPU, giving a global batch size of three. Each training sample contains $T = 2 4$ event tensors with $C = 1 0$ channels at $H = W = 5 1 2$ resolution and $N = 2 5 6$ randomly sampled trajectories. The encoder stride is $d = 4$ the feature dimension is $D = 1 2 8$ , and query correlation uses $L = 4$ pyramid levels with radius $r = 3$ . We use windows of length $S = 8$ with stride $s = 4$ (overlap length $K = 4 )$ and perform $M = 4$ refinement iterations.

We use bfloat16 mixed precision and AdamW with a learning rate of $5 \times 1 0 ^ { - 4 }$ , a minimum learning rate of $5 \times 1 0 ^ { - 5 }$ weight decay of $1 0 ^ { - 4 }$ , and 170 000 optimization steps. A cosine schedule decays the learning rate to its minimum over training.

For the two proposed modules, TEEM uses a hidden dimension of eight, a spatial downsampling factor of four, modulation strength $\alpha = 1$ , and normalization floor $\varepsilon _ { \mathrm { w } } = 1 0 ^ { - 4 }$ PTP uses $\lambda _ { o } = \lambda _ { s } = 1$ , update weight $\eta = 0 . 2 5 ,$ confidence floor $c _ { \operatorname* { m i n } } = 0 . 1 0$ , precision-diagonal floor $\varepsilon _ { \mathrm { p } } = 1 0 ^ { - 6 }$ , and terminal-displacement clip $u _ { \mathrm { m a x } } = 4$ feature-map pixels. No auxiliary TEEM or PTP regularization is applied; all variants share the same coordinate and visibility losses.

## B. Synthetic Training Data and Baseline

We use 7021 synthetic EventKubric samples generated following the ETAP pipeline [16]. Dynamic scenes are constructed with Kubric [34] and rendered as red-green-blue (RGB) video using Blender. Following the VID2E videoto-events workflow [35], the rendered frames are temporally interpolated and converted into synthetic event streams. Point trajectories and visibility labels are derived from Kubric’s depth, segmentation, and surface-normal annotations. Fig. 4 shows six examples with varied motion; the RGB frames visualize the scenes, while the event representations are the tracker inputs. The default training/validation/test allocation is 80:15:5; dagger-marked variants use a 90:5:5 allocation of the same corpus. Thus, the dagger indicates a larger training subset rather than newly generated scenes.

All variants are trained exclusively on synthetic event data. ETAP denotes the baseline retrained from its released source code under our training protocol. We omit time-inverted event pairs and the associated contrastive feature-alignment loss for both the baseline and SEPT. This baseline comparison isolates the architectural modifications under a common event-only training protocol.

## C. Task 1: Tracking Any Point

1) Datasets and metrics: We evaluate TAP on the synthetic EventKubric test set [16] and EVIMO2 [36], an event-camera benchmark for motion segmentation, optical flow, structure from motion, and visual-inertial odometry in indoor scenes.

![](images/b37fc4f4d0c1753f4812c499673ed48f5ee61f0b1c38ed8c87b01dbd6df18d7f.jpg)

![](images/93c1c8638270aca4f45b09b058b6466c11a9f0f050536d2703905846df1ecd87.jpg)

![](images/0a411ef5c703b56fcbd683fc7b6356e36ddafda018d1f1d8ed690781138bfcc2.jpg)

![](images/c7656b97f87a62bdeff638c6b5d3722fe1473fc98ad0d342da373ba8528e3a25.jpg)

![](images/126a25a10f59aa7e21615d57c738375cbc6550a4aefabd7da80f96f3114a7b5f.jpg)

![](images/377d5700fb81cace7f6507f38b9dc99aceaae55b5ff90f9c4aee2ee0aab7f696.jpg)  
Fig. 4. Synthetic EventKubric examples with varied motion. In each pair (a)–(f), the left panel shows the rendered RGB frame and the right panel its event representation. Colored markers and curves indicate query locations and ground-truth trajectories; an × marks a query point that is occluded at the displayed time.

TABLE I  
TAP RESULTS ON EVENTKUBRIC AND EVIMO2. ETAP [16] IS OUR RETRAINED BASELINE; SEPT COMBINES TEEM AND PTP.
<table><tr><td>Dataset</td><td>Method AJ ↑</td><td></td><td> $\delta _ { \mathrm { a v g } } ^ { x }$ </td><td>↑OA↑</td></tr><tr><td>EventKubric ETAP†</td><td></td><td>0.512</td><td>0.633</td><td>0.951</td></tr><tr><td></td><td>SEPT†</td><td>0.578</td><td>0.693</td><td>0.955</td></tr><tr><td>EVIMO2</td><td>ETAP†</td><td>0.635</td><td>0.751</td><td>0.943</td></tr><tr><td></td><td>SEPT†</td><td>0.673</td><td>0.786</td><td>0.936</td></tr></table>

EventKubric assesses correspondence and visibility estimation in simulated scenes, while EVIMO2 tests transfer to real event observations. We report three metrics introduced by TAP-Vid [1]: average Jaccard (AJ), average points within threshold $( \delta _ { \mathrm { a v g } } ^ { x } ) _ { \cdot }$ , and occlusion accuracy (OA). AJ evaluates localization jointly with visibility, ${ \delta _ { \mathrm { a v g } } ^ { x } }$ measures visible-point localization, and OA measures visibility classification. All scores are on [0, 1], with higher values indicating better performance.

2) Results on EventKubric: On the synthetic EventKubric test set, SEPT raises AJ from 0.512 to 0.578 and the visiblepoint localization score $\delta _ { \mathrm { a v g } } ^ { x }$ from 0.633 to 0.693 (Table I). OA is already high for ETAP and changes only from 0.951 to 0.955. The larger localization gain indicates that SEPT primarily improves point correspondence over time while preserving strong visibility prediction.

3) Results on EVIMO2: On the real EVIMO2 sequences, SEPT raises AJ from 0.635 to 0.673 and $\delta _ { \mathrm { a v g } } ^ { x }$ from 0.751 to 0.786, although OA decreases from 0.943 to 0.936 (Table I). The AJ gain despite this small visibility penalty shows that the localization benefit transfers from synthetic training to real events without fine-tuning. Across both benchmarks, improved localization is the consistent source of SEPT’s TAP gain; the small OA decline on EVIMO2 identifies real-data visibility estimation as the remaining weakness.

TABLE II  
FEATURE TRACKING ON EC AND EDS. E DENOTES EVENTS; E+F DENOTES EVENTS AND FRAMES. MATE RESULTS ARE QUOTED FROM ITS ORIGINAL PAPER. M3ED RESULTS ARE REPORTED BY SEQUENCE IN TABLE III.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Input</td><td colspan="2">EC</td><td colspan="2">EDS</td></tr><tr><td>FA</td><td>EFA</td><td>FA</td><td>EFA</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>ICP [21]</td><td>E</td><td>0.256</td><td>0.245</td><td>0.060</td><td>0.040</td></tr><tr><td>EKLT [12]</td><td>E+F</td><td>0.811</td><td>0.775</td><td>0.325</td><td>0.205</td></tr><tr><td>DDFT [14]</td><td>E+F</td><td>0.825 0.844</td><td>0.818</td><td>0.576</td><td>0.472</td></tr><tr><td>FE-TAP [40]</td><td>E+F</td><td>0.337</td><td>0.838</td><td>0.676</td><td>0.589</td></tr><tr><td>EM-ICP [22]</td><td>E</td><td></td><td>0.334</td><td>0.161</td><td>0.120</td></tr><tr><td>HASTE [41]</td><td>E</td><td>0.442</td><td>0.427</td><td>0.096</td><td>0.063</td></tr><tr><td>DDFT E2VID [14], [42] E MATE [17]</td><td>E</td><td>0.794 0.885</td><td>0.786 0.875</td><td>0.589 0.713</td><td>0.495</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.626</td></tr><tr><td>ETAP† [16]</td><td>E</td><td>0.855</td><td>0.850</td><td>0.711</td><td>0.605</td></tr><tr><td>SEPT† (ours)</td><td>E</td><td>0.866</td><td>0.861</td><td>0.732</td><td>0.619</td></tr></table>

## D. Task 2: Feature Tracking

1) Datasets and metrics: We evaluate feature tracking on the Event Camera (EC) dataset [37] and the Event-aided Direct Sparse Odometry (EDS) dataset [38] using the data and protocol of ETAP [16]; we additionally evaluate on the Multi-Robot, Multi-Sensor, Multi-Environment Event Dataset (M3ED) [39]. Feature age (FA) measures normalized tracking lifetime among tracks with positive lifetime; expected feature age (EFA) also accounts for immediately failed tracks [14]. Following the same evaluation protocol, we average both metrics over localization thresholds from 1 to 31 pixels. Higher FA and EFA are better.

For M3ED, we construct feature-tracking ground truth from the provided depth maps and camera poses. Candidate image points with valid depth are back-projected into 3-D and reprojected through the camera trajectory to obtain their 2- D positions over time. Because this geometric construction assumes that a point is stationary in the world, we manually review the proposed trajectories and exclude points on moving objects or with unreliable projections before accepting them as ground truth. Points that leave the field of view or become occluded retain their identities but are marked invisible and excluded from position-error evaluation. The selected sequences span vehicle-mounted and unmanned aerial vehicle recordings, daytime and nighttime conditions, and urban, indoor, and outdoor scenes, providing a diverse test of transfer beyond the EC and EDS benchmarks. Each evaluated segment lasts 2 s and is sampled at 25 Hz, yielding 51 timestamps. For FA and EFA, each point’s evaluable span is its initial uninterrupted ground-truth-visible interval; the first localization error beyond the selected threshold ends its predicted lifetime.

Because the RGB and event sensors in our M3ED setup are not spatially co-registered, we exclude RGB–event fusion methods that require aligned inputs.

2) Results on EC and EDS: Table II compares SEPT with event-only methods and methods combining events and frames. Published baseline scores are reproduced from ETAP [16], except for MATE, whose scores are quoted from its original paper [17]. MATE’s official implementation and pretrained weights were unavailable to us, so we could not re-evaluate it under our pipeline. Since its training conditions differ from ours, its scores provide a reference comparison rather than a controlled same-protocol comparison. The ETAP<sup>†</sup> and SEPT<sup>†</sup> results are from our own runs. SEPT achieves 0.866 FA and 0.861 EFA on EC, exceeding our retrained ETAP scores of 0.855 and 0.850. On EDS, SEPT improves FA from 0.711 to 0.732 and EFA from 0.605 to 0.619. Compared with MATE’s published results, SEPT achieves higher FA on EDS, while MATE reports higher FA and EFA on EC and higher EFA on EDS.

3) Results on M3ED: Table III reports FA and EFA for each of the nine evaluated M3ED sequences. Averaged over sequences, SEPT reaches 0.797 FA and 0.795 EFA, compared with 0.795 and 0.792 for the retrained ETAP baseline. SEPT therefore improves both feature-age measures on this real benchmark despite training exclusively on synthetic EventKubric data. We attribute the smaller margin to differences in event content and motion between the synthetic training scenes and M3ED, which make the learned TEEM weighting harder to transfer. Fig. 5 qualitatively compares trajectories from SEPT, DDFT, and HASTE on four M3ED sequences.

## E. Ablation and Training-Data Sensitivity

We first analyze TEEM and PTP individually, then their combined performance and inference cost, and finally their sensitivity to the synthetic training-data allocation. Table IV reports ETAP, the two single-module variants, and SEPT on all five datasets under both training allocations.

1) Contribution ofTEEM: ETAP+TEEM increases AJ over ETAP on both EventKubric and EVIMO2 under each training allocation (Table IV). Its feature-tracking effect varies by dataset, indicating that learned event weighting is more sensitive to the event distribution when used alone.

Fig. 6 presents event tensors, TEEM weight maps, and modulated evidence from the in-house calibration sequence and real EC, EDS, EVIMO2, and M3ED recordings. Without edge masks or event-importance labels, TEEM learns spatially structured weights that follow checkerboard boundaries, scene contours, and moving-object outlines even where accumulated events leave motion trails. The modulated evidence retains recognizable geometric structure rather than uniformly amplifying event activity. This task-driven selectivity provides a concrete explanation for the ETAP+TEEM gains over ETAP on several datasets.

To test whether the visual differences reflect better alignment with object contours, we compare the original and modulated event evidence against visible object boundaries from aligned EventKubric RGB and segmentation data (Fig. 7). Across 10 test sequences (240 frames), TEEM increases the boundary-hit rate from 55.3% to 66.6% and reduces the mean boundary-response width from 4.013 to 3.359 pixels (16.3%). These contour-based measurements show that TEEM produces more compact and better-aligned event responses at visible object boundaries.

2) Contribution of PTP: ETAP+PTP improves AJ on both TAP datasets and FA on EC, EDS, and M3ED under both training allocations (Table IV). Unlike TEEM, PTP has no trainable weights; these consistent gains support retaining geometric constraints across window boundaries rather than relying on coordinate initialization alone.

TABLE III  
PER-SEQUENCE FEATURE TRACKING ON THE NINE EVALUATED M3ED LEFT-CAMERA SEGMENTS. ENTRIES ARE FA/EFA, AVERAGED OVER LOCALIZATION THRESHOLDS FROM 1 TO 31 PIXELS. DDFT [14] USES ITS EVENT-ONLY CONFIGURATION. THE FINAL ROW IS THE UNWEIGHTED MEAN ACROSS SEQUENCES.
<table><tr><td>Sequence</td><td>HASTE [41]</td><td>AEB-Tracker [43]</td><td>DDFT [14]</td><td>ETAP†</td><td>SEPT† (ours)</td></tr><tr><td>car_urban_day_ penno_small_loop</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>car_urban_night_</td><td>0.733/0.669</td><td>0.530/0.455</td><td>0.813/0.807</td><td>0.831/0.829</td><td>0.824/0.822</td></tr><tr><td>penno_big_loop car_urban_night_</td><td>0.693/0.632</td><td>0.553/0.504</td><td>0.683/0.657</td><td>0.836/0.828</td><td>0.854/0.845</td></tr><tr><td>penno_small_loop falcon_indoor_</td><td>0.577/0.541</td><td>0.457/0.429</td><td>0.557/0.540</td><td>0.736/0.735</td><td>0.731/0.729</td></tr><tr><td>flight_1 falcon_indoor_</td><td>0.576/0.502</td><td>0.426/0.387</td><td>0.652/0.644</td><td>0.740/0.739</td><td>0.742/0.741</td></tr><tr><td>flight_2 falcon_outdoor_day_</td><td>0.611/0.558</td><td>0.539/0.482</td><td>0.750/0.741</td><td>0.827/0.825</td><td>0.802/0.801</td></tr><tr><td>fast_flight_2 falcon_outdoor_night_</td><td>0.614/0.606</td><td>0.434/0.396</td><td>0.799/0.704</td><td>0.878/0.877</td><td>0.881/0.881</td></tr><tr><td>high_beams falcon_outdoor_night_</td><td>0.673/0.664</td><td>0.353/0.349</td><td>0.380/0.376</td><td>0.860/0.859</td><td>0.854/0.853</td></tr><tr><td>penno_parking_1 falcon_outdoor_night_</td><td>0.428/0.383</td><td>0.361/0.329</td><td>0.452/0.441</td><td>0.747/0.742</td><td>0.777/0.773</td></tr><tr><td>penno_parking_2</td><td>0.501/0.452</td><td>0.322/0.316</td><td>0.493/0.489</td><td>0.697/0.695</td><td>0.710/0.709</td></tr><tr><td>Mean</td><td>0.601/0.556</td><td>0.441/0.405</td><td>0.620/0.600</td><td>0.795/0.792</td><td>0.797/0.795</td></tr></table>

![](images/4bd3ddb7f0385e39db63825e224b23797fc9df9f4ddca9eded82793658fb7d9d.jpg)  
Fig. 5. Qualitative feature-tracking results on M3ED. Rows show four sequences; columns compare SEPT, DDFT [14], and HASTE [41] from left to right. Colored curves show predicted tracks, and × marks the ground-truth point position at each timestamp.

3) Combined effects and inference cost: With the expanded training split, SEPT exceeds either single-module variant in AJ on EventKubric and EVIMO2 and in FA and EFA on EDS. It also improves the primary tracking metric over the retrained ETAP baseline on all five datasets. TEEM and PTP act at complementary stages: TEEM selects event evidence before encoding, whereas PTP carries and refines the resulting trajectory across windows. PTP alone retains the highest EC FA and EFA, showing that the benefit of their combination depends on the evaluation domain.

TABLE IV  
MODULE ABLATION AND TRAINING-DATA SENSITIVITY ACROSS TAP (EVENTKUBRIC AND EVIMO2) AND FEATURE TRACKING (EC, EDS, AND M3ED). BOLD MARKS THE BEST SCORE FOR EACH DATASET AND METRIC WITHIN A TRAINING SPLIT.
<table><tr><td></td><td colspan="5">TAP</td><td></td><td colspan="5">Feature tracking</td></tr><tr><td></td><td colspan="3">EventKubric</td><td colspan="3">EVIMO2</td><td colspan="2">EC</td><td colspan="2">EDS</td><td colspan="2">M3ED</td></tr><tr><td>Model</td><td>AJ</td><td> $\delta _ { \mathrm { a v g } } ^ { x }$ </td><td>OA</td><td>AJ</td><td> $\delta _ { \mathrm { a v g } } ^ { x }$ </td><td>OA</td><td>FA</td><td>EFA</td><td>FA</td><td>EFA</td><td>FA</td><td>EFA</td></tr><tr><td colspan="9">Standard training split (80:15:5)</td><td></td><td></td><td></td><td></td></tr><tr><td>ETAP</td><td>0.524</td><td>0.643</td><td>0.949</td><td>0.622</td><td>0.737</td><td>0.940</td><td>0.839</td><td>0.833</td><td>0.699</td><td>0.599</td><td>0.789</td><td>0.787</td></tr><tr><td>ETAP+TEEM</td><td>0.531</td><td>0.649</td><td>0.944</td><td>0.655</td><td>0.771</td><td>0.937</td><td>0.797</td><td>0.792</td><td>0.719</td><td>0.605</td><td>0.795</td><td>0.795</td></tr><tr><td>ETAP+PTP</td><td>0.566</td><td>0.681</td><td>0.946</td><td>0.663</td><td>0.777</td><td>0.943</td><td>0.866</td><td>0.861</td><td>0.726</td><td>0.614</td><td>0.797</td><td>0.794</td></tr><tr><td>SEPT</td><td>0.554</td><td>0.676</td><td>0.948</td><td>0.643</td><td>0.761</td><td>0.930</td><td>0.854</td><td>0.849</td><td>0.722</td><td>0.615</td><td>0.797</td><td>0.794</td></tr><tr><td colspan="9">Expanded training split (90:5:5)†</td><td></td><td></td><td></td><td></td></tr><tr><td>ETAP</td><td>0.512</td><td>0.633</td><td>0.951</td><td>0.635</td><td>0.751</td><td>0.943</td><td>0.855</td><td>0.850</td><td>0.711</td><td>0.605</td><td>0.795</td><td>0.792</td></tr><tr><td>ETAP+TEEM</td><td>0.524</td><td>0.643</td><td>0.943</td><td>0.652</td><td>0.762</td><td>0.937</td><td>0.869</td><td>0.864</td><td>0.686</td><td>0.587</td><td>0.792</td><td>0.789</td></tr><tr><td>ETAP+PTP</td><td>0.534</td><td>0.652</td><td>0.939</td><td>0.645</td><td>0.760</td><td>0.935</td><td>0.874</td><td>0.869</td><td>0.725</td><td>0.609</td><td>0.797</td><td>0.794</td></tr><tr><td>SEPT</td><td>0.578</td><td>0.693</td><td>0.955</td><td>0.673</td><td>0.786</td><td>0.936</td><td>0.866</td><td>0.861</td><td>0.732</td><td>0.619</td><td>0.797</td><td>0.795</td></tr></table>

![](images/adc3b26bbb005b2ce349928204a6da802cafba85cc1ff71050058a99583ebacf.jpg)  
Fig. 6. TEEM visualizations on real event recordings across datasets and timestamps. Each example shows the input event tensor, the learned event-weight map, and the modulated event evidence. Color bars indicate the weight values.

![](images/120d684deaabb290f568159f9e53da04aeafa47a8eac8f9051ce2fce45854882.jpg)  
Fig. 7. TEEM evidence at visible object boundaries on EventKubric. Segmentation-derived ground-truth contours (green) are overlaid on the aligned RGB frame, original event tensor, and TEEM-modulated event evidence.

Fig. 8 shows the accuracy–resource trade-off under the expanded training split. SEPT achieves the highest EventKubric AJ but also has the longest inference time and ties ETAP+TEEM for the highest peak reserved GPU memory. The memory increase occurs only when TEEM is enabled; ETAP+PTP matches ETAP’s peak reserved GPU memory while improving AJ with little added time. PTP therefore offers the most favorable accuracy–efficiency balance among the single-module variants and points to compact trajectory priors as a promising design direction.

4) Sensitivity to training-data allocation: On the shared 80:15:5 validation set, the validation-selected checkpoints yield mean visible-point endpoint error (EPE) values of 6.3902 for SEPT and 6.6403 for ETAP+PTP; lower is better. On the EventKubric and EVIMO2 tests under the same 80:15:5 allocation, ETAP+PTP nevertheless has higher AJ than SEPT.

![](images/b44d073367db0cabcffd19abb5c5651354b12ef35668ca21f8d0f3363f81e1ab.jpg)

![](images/93720b6448b3e83de04e2a7100be680c90f6fec4bcb7f580c12b93ee0a09eca6.jpg)  
Fig. 8. EventKubric AJ versus (a) mean inference time per forward pass and (b) peak reserved GPU memory for ETAP and the ablated variants. Peak allocated memory changes little across variants and is not plotted. Results use the expanded training split.

We attribute SEPT’s weaker test AJ despite its lower validation EPE to overfitting of the learnable TEEM front end under the smaller synthetic training allocation. PTP provides a strong coordinate-space prior without additional learned weights, whereas the full model needs broader training coverage for TEEM to improve test-time correspondence.

With the 90:5:5 allocation, SEPT overtakes ETAP+PTP in AJ on both TAP test sets and delivers gains over ETAP across all five benchmarks. The additional synthetic examples allow the learned event weighting and the persistent trajectory prior to work together more effectively.

## V. DISCUSSION

Event cameras respond to brightness changes rather than static appearance. In a static scene with stable illumination, few scene events are triggered, and the remaining event stream may be dominated by sensor noise. A TAP method then receives sparse observations with a low signal-to-noise ratio and few stable features to match. This limitation follows from the sensing principle itself and cannot always be addressed by improving the event-only tracker. A promising direction is multimodal fusion: visible-light images can provide appearance cues when the scene is static, while inertial measurements can supply complementary motion information.

A second challenge is balancing accuracy against processing cost. SEPT attains the highest AJ on both TAP benchmarks but has higher peak reserved GPU memory and longer inference time than ETAP. By contrast, PTP alone improves accuracy with little overhead and without increasing peak reserved GPU memory. Future trackers should retain such compact geometric constraints while reducing the cost of learned event modulation, especially for low-latency applications.

## VI. CONCLUSION

Stable long-term event-based point tracking requires reliable correspondences from sparse, motion-dependent event activity and trajectory continuity across finite windows. SEPT addresses these complementary challenges with TEEM, which learns task-driven event weights before feature extraction, and PTP, which carries a parameter-free, Schur-marginalized geometric prior through overlapping windows. Trained exclusively on synthetic EventKubric data, SEPT improves TAP AJ on EventKubric and EVIMO2 and feature-tracking FA and EFA on EC, EDS, and M3ED over our retrained ETAP baseline; on EC and EDS, it also surpasses several trackers using both events and frames. Ablations show that PTP provides consistent gains at modest cost and that broader synthetic training improves the full model, which incurs a higher inference cost. These findings support combining selective event evidence with explicit geometric memory. Future work should reduce event-modulation cost and supply complementary cues when event activity is sparse.

## REFERENCES

[1] C. Doersch et al., “TAP-Vid: A benchmark for tracking any point in a video,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, 2022, pp. 13 610–13 626, doi: 10.52202/068431-0989.

[2] H. Li et al., “TAPTRv2: Attention-based position update improves tracking any point,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024, pp. 101 074–101 095, doi: 10.52202/079017-3205.

[3] M. Vecerik et al., “RoboTAP: Tracking arbitrary points for few-shot visual imitation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 5397–5403, doi: 10.1109/icra57147.2024.10611409.

[4] W. Chen, L. Chen, R. Wang, and M. Pollefeys, “LEAP-VO: Long-term effective any point tracking for visual odometry,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 19 844–19 853, doi: 10.1109/cvpr52733.2024.01876.

[5] A. W. Harley, Z. Fang, and K. Fragkiadaki, “Particle video revisited: Tracking through occlusions using point trajectories,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 59–75, doi: 10.1007/ 978-3-031-20047-2 4.

[6] C. Doersch et al., “TAPIR: Tracking any point with per-frame initialization and temporal refinement,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 10 061–10 072, doi: 10.1109/iccv51070.2023. 00923.

[7] N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht, “CoTracker: It is better to track together,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 18–35, doi: 10.1007/ 978-3-031-73033-7 2.

[8] N. Karaev, Y. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht, “CoTracker3: Simpler and better point tracking by pseudolabelling real videos,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 6013–6022, doi: 10.1109/iccv51701.2025.00568.

[9] G. Le Moing, J. Ponce, and C. Schmid, “Dense optical tracking: Connecting the dots,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 19 187–19 197, doi: 10.1109/cvpr52733. 2024.01815.

[10] P. Lichtsteiner, C. Posch, and T. Delbruck, “A 128×128 120 dB 15 µs latency asynchronous temporal contrast vision sensor,” IEEE J. Solid-State Circuits, vol. 43, no. 2, pp. 566–576, 2008, doi: 10.1109/jssc.2007. 914337.

[11] G. Gallego et al., “Event-based vision: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 1, pp. 154–180, 2022, doi: 10.1109/ tpami.2020.3008413.

[12] D. Gehrig, H. Rebecq, G. Gallego, and D. Scaramuzza, “EKLT: Asynchronous photometric feature tracking using events and frames,” Int. J. Comput. Vis., vol. 128, no. 3, pp. 601–618, 2020, doi: 10.1007/ s11263-019-01209-w.

[13] E. Mueggler, G. Gallego, and D. Scaramuzza, “Continuous-time trajectory estimation for event-based vision sensors,” in Proc. Robotics: Science and Systems (RSS), 2015, doi: 10.15607/rss.2015.xi.036.

[14] N. Messikommer, C. Fang, M. Gehrig, and D. Scaramuzza, “Datadriven feature tracking for event cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 5642–5651, doi: 10.1109/cvpr52729.2023.00546.

[15] S. Li, Z. Zhou, Z. Xue, Y. Li, S. Du, and Y. Gao, “3D feature tracking via event camera,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 18 974–18 983, doi: 10.1109/cvpr52733.2024.01795.

[16] F. Hamann, D. Gehrig, F. Febryanto, K. Daniilidis, and G. Gallego, “ETAP: Event-based tracking of any point,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 27 186–27 196, doi: 10.1109/cvpr52734.2025.02532.

[17] H. Han, W. Zhai, Y. Cao, B. Li, and Z.-j. Zha, “MATE: Motionaugmented temporal consistency for event-based point tracking,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 8340–8349, doi: 10.1109/iccv51701.2025.00781.

[18] X. Wang, H. Yu, L. Yu, W. Yang, and G.-S. Xia, “Toward robust keypoint detection and tracking: A fusion approach with event-aligned image features,” IEEE Robot. Autom. Lett., vol. 9, no. 9, pp. 8059–8066, 2024, doi: 10.1109/lra.2024.3440099.

[19] S. Hu, Y. Kim, H. Lim, A. J. Lee, and H. Myung, “eCDT: Event clustering for simultaneous feature detection and tracking,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2022, pp. 3808–3815, doi: 10.1109/iros47612.2022.9981451.

[20] X. Lagorce, G. Orchard, F. Galluppi, B. E. Shi, and R. B. Benosman, “HOTS: A hierarchy of event-based time-surfaces for pattern recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 7, pp. 1346– 1359, 2017, doi: 10.1109/tpami.2016.2574707.

[21] B. Kueng, E. Mueggler, G. Gallego, and D. Scaramuzza, “Low-latency visual odometry using event-based feature tracks,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2016, pp. 16–23, doi: 10.1109/ iros.2016.7758089.

[22] A. Z. Zhu, N. Atanasov, and K. Daniilidis, “Event-based feature tracking with probabilistic data association,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2017, pp. 4465–4470, doi: 10.1109/icra.2017.7989517.

[23] M. Ikura, C. Le Gentil, M. G. Muller, F. Schuler, A. Yamashita, and¨ W. Sturzl, “RATE: Real-time asynchronous feature tracking with event¨ cameras,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2024, pp. 11 662–11 669, doi: 10.1109/iros58592.2024.10802050.

[24] Z. Huang, L. Sun, C. Zhao, S. Li, and S. Su, “EventPoint: Selfsupervised interest point detection and description for event-based camera,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2023, pp. 5396–5405, doi: 10.1109/wacv56688.2023.00536.

[25] Y. Fu et al., “Distractor-aware event-based tracking,” IEEE Trans. Image Process., vol. 32, pp. 6129–6141, 2023, doi: 10.1109/tip.2023.3326683.

[26] Z. Chen, J. Wu, W. Dong, L. Li, and G. Shi, “CrossEI: Boosting motionoriented object tracking with an event camera,” IEEE Trans. Image Process., vol. 34, pp. 73–84, 2025, doi: 10.1109/tip.2024.3505672.

[27] A. Z. Zhu, L. Yuan, K. Chaney, and K. Daniilidis, “EV-FlowNet: Selfsupervised optical flow estimation for event-based cameras,” in Proc. Robotics: Science and Systems (RSS), 2018, doi: 10.15607/rss.2018.xiv. 062.

[28] M. Gehrig, M. Millh”ausler, D. Gehrig, and D. Scaramuzza, “E-RAFT: Dense optical flow from event cameras,” in Proc. Int. Conf. 3D Vis. (3DV), 2021, pp. 197–206, doi: 10.1109/3dv53792.2021.00030.

[29] Y. Zheng, A. W. Harley, B. Shen, G. Wetzstein, and L. J. Guibas, “PointOdyssey: A large-scale synthetic dataset for long-term point tracking,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 19 855–19 865, doi: 10.1109/iccv51070.2023.01818.

[30] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon, “Bundle adjustment—a modern synthesis,” in Vision Algorithms: Theory and Practice, 2000, pp. 298–372, doi: 10.1007/3-540-44480-7 21.

[31] A. Mourikis and S. Roumeliotis, “A multi-state constraint Kalman filter for vision-aided inertial navigation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2007, pp. 3565–3572, doi: 10.1109/robot.2007.364024.

[32] S. Leutenegger, S. Lynen, M. Bosse, R. Siegwart, and P. Furgale, “Keyframe-based visual-inertial odometry using nonlinear optimiza-

tion,” Int. J. Robot. Res., vol. 34, no. 3, pp. 314–334, 2015, doi: 10.1177/0278364914554813.

[33] T. Qin, P. Li, and S. Shen, “Vins-mono: A robust and versatile monocular visual-inertial state estimator,” IEEE Trans. Robot., vol. 34, no. 4, pp. 1004–1020, 2018, doi: 10.1109/tro.2018.2853729.

[34] K. Greff et al., “Kubric: A scalable dataset generator,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 3749–3761, doi: 10.1109/cvpr52688.2022.00373.

[35] D. Gehrig, M. Gehrig, J. Hidalgo-Carrio, and D. Scaramuzza, “Video to´ events: Recycling video datasets for event cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 3586–3595, doi: 10.1109/cvpr42600.2020.00364.

[36] L. Burner, A. Mitrokhin, C. Fermuller, and Y. Aloimonos, “EVIMO2:¨ An event camera dataset for motion segmentation, optical flow, structure from motion, and visual inertial odometry in indoor scenes with monocular or stereo algorithms,” arXiv preprint arXiv:2205.03467, 2022, doi: 10.48550/arxiv.2205.03467.

[37] E. Mueggler, H. Rebecq, G. Gallego, T. Delbruck, and D. Scaramuzza, “The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM,” Int. J. Robot. Res., vol. 36, no. 2, pp. 142–149, 2017, doi: 10.1177/0278364917691115.

[38] J. Hidalgo-Carrio, G. Gallego, and D. Scaramuzza, “Event-aided direct´ sparse odometry,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5781–5790, doi: 10.1109/cvpr52688.2022.00569.

[39] K. Chaney et al., “M3ED: Multi-robot, multi-sensor, multi-environment event dataset,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2023, pp. 4016–4023, doi: 10.1109/cvprw59228. 2023.00419.

[40] J. Liu, B. Wang, Z. Tan, J. Zhang, H. Shen, and D. Hu, “Tracking any point with frame-event fusion network at high frame rate,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2025, pp. 18 834– 18 840, doi: 10.1109/iros60139.2025.11247196.

[41] I. Alzugaray and M. Chli, “HASTE: Multi-hypothesis asynchronous speeded-up tracking of events,” in Proc. Brit. Mach. Vis. Conf. (BMVC), 2020, doi: 10.5244/c.34.164.

[42] H. Rebecq, R. Ranftl, V. Koltun, and D. Scaramuzza, “High speed and high dynamic range video with an event camera,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 6, pp. 1964–1980, 2021, doi: 10.1109/ tpami.2019.2963386.

[43] Z. Wang, T. Molloy, P. van Goor, and R. Mahony, “Asynchronous blob tracker for event cameras,” IEEE Trans. Robot., vol. 40, pp. 4750–4767, 2024, doi: 10.1109/tro.2024.3454410.