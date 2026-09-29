# λ-JEPA: SPECTRAL ANTI-COLLAPSEREGULARIZATION FOR SELF-SUPERVISED LEARNING

Berker Demirel<sup>\*</sup> Clementine Domin´ e´<sup>\*</sup> Valentino Maiorca Marco Fumero Marco Mondelli Francesco Locatello Institute of Science and Technology Austria (ISTA) Am Campus 1, 3400 Klosterneuburg, Austria

## ABSTRACT

Joint-embedding self-supervised learning typically combines an invariance objective across augmented views with additional mechanisms to prevent representational collapse. These objectives are often applied after a projection head, while downstream tasks use the backbone representation before the projector. We find that this mismatch does not necessarily prevent dimensional collapse in the backbone, which can retain low effective rank and potentially limit downstream transfer. To address this, we introduce SACReg, a spectral anti-collapse regularizer motivated by an analysis of λ-balance, which captures the relative scale of weight matrices across layers. In a two-layer linear network, we show that (i) λ-balance prevents collapse, and (ii) our regularizer applied to the backbone induces λ-balance. In the nonlinear case, this regularizer leads to anti-collapse as well and, in realistic architectures on ImageNet100, it empirically increases the representations’ ranks. We apply SACReg to JEPA and propose λ-JEPA, which improves over LeJEPA and VISReg on ImageNet-1k classification and in average linear-probe transfer performance across eight downstream image datasets. On video self-supervised learning, λ-JEPA improves over LeVJEPA and V-JEPA 2 on the Something-Something-v2 and Kinetics-400 benchmarks. Code is available at https://github.com/berkerdemirel/lambda-jepa.

## 1 INTRODUCTION

Self-supervised learning aims to learn transferable representations from unlabeled data without requiring manual annotations (Jing & Tian, 2021). In computer vision, joint-embedding selfsupervised learning (JE-SSL) has become one of the most promising approaches, encouraging representations of related views or regions of an image to agree while additional mechanisms prevent representational collapse (Jing et al., 2022; Bardes et al., 2022). These mechanisms include using contrastive negatives (Chen et al., 2020; He et al., 2020), asymmetric teacher-student training (Caron et al., 2021; Zhou et al., 2022; Grill et al., 2020) and explicit anti-collapse regularization (Balestriero & LeCun, 2025; Bardes et al., 2022; Wu et al., 2026).

JE-SSL methods typically optimize a projected representation during pretraining, while downstream tasks discard the non-linear projection head and use the backbone representation instead. These two representation spaces can differ substantially in both geometry and downstream performance (Bordes et al., 2023). In contrastive self-supervised learning, this has also been studied through dimensional collapse, where learned representations span only a lower dimensional part of the available feature space (Jing et al., 2022). We observe an analogous phenomenon in explicitly regularized JE-SSL methods. Although their anti-collapse objectives successfully maintain high rank projected representations, the corresponding backbone representations can remain low rank. This matters for task-agnostic representation learning, where different downstream tasks can benefit from different invariances (Ericsson et al., 2021; Tian et al., 2020b). Dimensional collapse in the backbone can therefore limit transfer by reducing the set of feature directions available for downstream tasks. Consistent with this view, effective rank has been shown to predict downstream performance in JE-SSL (Garrido et al., 2023).

The rank of learned representations has been studied extensively in the feature-learning literature, with prior work linking representation rank to distinct learning regimes in linear and nonlinear networks (Saxe et al., 2014; Atanasov et al., 2022). Building on this perspective, we analyze a two-layer linear model with λ-balance initialization, which captures the relative scale of adjacent layers, and connects it to representation rank. The resulting spectral anti-collapse mechanism motivates a covariance regularizer for nonlinear networks, applied directly to the backbone features to restore their representation rank. In other words, our λ-balance theory prescribes both the form of the regularizer and where it should be applied. We then apply it accordingly to JE-SSL in a method we call λ-JEPA, using view-averaged backbone representations to prevent collapse and promote variation across images (see Figure 6). In controlled experiments, we show that λ-JEPA improves class-relevant separation while preserving augmentation dependent variation. At larger scale, it outperforms both LeJEPA (Balestriero & LeCun, 2025) and VISReg (Wu et al., 2026) on ImageNet-1K (Russakovsky et al., 2015) trained from scratch, with performance approaching DINO (Caron et al., 2021) in both linear probing and transfer learning. The same approach applies off the shelf to video JEPA models, where λ-JEPA trained from scratch outperforms V-JEPA 2 (Assran et al., 2025) and LeVJEPA (Kuhn et al., 2026) on Something-Something-v2 (Goyal et al., 2017) and Kinetics-400 (Kay et al., 2017).

We summarize our main contributions as:

• We show that standard JE-SSL methods designed to prevent collapse in the projected representation do not ensure a high-rank backbone, potentially limiting its representational capacity for downstream tasks.

• We derive SACReg, a spectral anti-collapse regularizer that guarantees encoder non-collapse in a two-layer linear network, motivating encoder-side spectral regularization. We extend this to nonlinear encoders by directly regularizing representation covariance and apply it to JE-SSL using view-averaged backbone representations to prevent collapse and promote variation across images.

• We propose λ-JEPA, a standalone SSL method that applies SACReg to both the backbone and projected representations. Across image JE-SSL benchmarks, λ-JEPA improves frozen transfer while remaining competitive on ImageNet-1k classification, and extends to video SSL with large gains on temporal recognition.

## 2 PROBLEM AND RELATED WORKS

Joint-embedding SSL Modern joint-embedding self-supervised learning (JE-SSL) methods differ primarily in how they avoid collapsed solutions. We focus on a family of explicitly regularized methods (VICReg (Bardes et al., 2022), LeJEPA (Balestriero & LeCun, 2025), VISReg (Wu et al., 2026)). In the projected space, VI-CReg controls feature variance and covariance, while LeJEPA regularizes the embedding distribution toward an isotropic Gaussian. VISReg similarly combines variance regularization with

![](images/7212c2551c6b762c539605aecc1227984702f1ecb871a8ce42da3174874f9ef5.jpg)  
Figure 1: Backbone and loss space geometry for explicitly regularized JE-SSL methods. The projection head changes both pairwise similarities and RankMe.

an additional distributional constraint. Following the use of projection heads in SimCLR (Chen et al., 2020), these objectives act after a non-linear projection head, which is discarded for downstream tasks.

We focus on a specific consequence of this separation: in these explicitly regularized methods, anticollapse constraints are imposed on z, yet with nonlinear projectors, preventing dimensional collapse in z does not in general prevent it in the backbone representation h.<sup>1</sup>. Figure 1 shows how two key aspects of explicitly regularized JE-SSL methods change across the projection head: (i) agreement between augmented views and (ii) prevention of representational collapse. Positive pairs remain highly aligned in both the backbone and projected spaces. In contrast, different images have very similar representations in the backbone, indicating substantial concentration. The same discrepancy appears in representation rank. RankMe (Garrido et al., 2023) measures how evenly variance is distributed across feature directions and is typically higher in the projected space, whereas backbone remains low rank, especially for VISReg and LeJEPA. For task-agnostic representation learning, low rank backbones can be restrictive because different downstream tasks may rely on different feature directions (Ericsson et al., 2021; Tian et al., 2020b). Consistently, Garrido et al. (2023) have shown that it correlates with downstream performance in JE-SSL, motivating backbone-level anti-collapse regularization.

Feature learning and anti-collapse of the backbone representations The question of what determines the rank of learned representations has been studied extensively in the feature learning literature which has so far had little contact with SSL but offers directly relevant tools. In particular, deep linear networks provide a tractable setting for studying these effects. Despite their linear endto-end mapping, their factorized parameterization induces nonlinear optimization dynamics (Baldi & Hornik, 1989; Fukumizu, 1998; Saxe et al., 2014; Du et al., 2019; Jacot et al., 2018; Chizat et al., 2019; Braun et al., 2022) and captures phenomena observed in nonlinear networks (Saxe et al., 2019; Kunin et al., 2024; Nam et al., 2025; Anguita et al., 2026). Initialization plays a central role in these dynamics: small initializations favor low-rank or sparse solutions, whereas appropriate large-scale limits yield kernel-like behavior (Saxe et al., 2014; Gunasekar et al., 2018; Chizat & Bach, 2020; Woodworth et al., 2020; Li et al., 2021). In nonlinear networks, initialization scale likewise influences the learning regime (Luo et al., 2021; Atanasov et al., 2022). Beyond overall initialization scale, the relative scales of adjacent layers, as studied through λ-balanced initializations, also in fluence learning dynamics and implicit bias (Azulay et al., 2021; Kunin et al., 2024; Domine et al.,´ 2025; Jarvis et al., 2025). Their consequences for representation rank, however, remain less characterized, particularly in the nonlinear regime and in larger scale networks. Our work addresses this gap by investigating how layer balance shapes representation rank and using these insights to derive a regularizer that prevents collapse in the backbone representations of SSL models.

## 3 METHOD

Our goal is to prevent collapse in the backbone representations retained for downstream transfer. We first study a tractable two-layer linear network, where we show that negative layer balance yields a non-collapse guarantee and motivates an encoder-side spectral regularizer. We extend this principle to nonlinear encoders by directly regularizing representation covariance, then apply it to JE-SSL on view-averaged backbone representations to promote variation across images. Figure 6 illustrates the resulting λ-JEPA objective and the placement of SACReg in the backbone and projected spaces.

## 3.1 DERIVING A SPECTRAL ANTI-COLLAPSE MECHANISM IN LINEAR NETWORKS

Linear-network setting. Consider a supervised dataset $\mathcal { D } = \{ ( \mathbf { x } _ { n } , \mathbf { y } _ { n } ) \} _ { n = 1 } ^ { P } , \mathbf { x } _ { n } \in \mathbb { R } ^ { N _ { i } } , \mathbf { y } _ { n } \in$ $\mathbb { R } ^ { N _ { o } }$ . We model the input–output mapping using a two-layer linear network, $\widehat { \mathbf { y } } _ { n } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } \mathbf { x } _ { n } .$ where $\mathbf { W } _ { 1 } ~ \in ~ \mathbb { R } ^ { N _ { h } \times N _ { i } }$ and $\dot { { \bf W } } _ { 2 } \in \mathbb { R } ^ { \breve { N } _ { o } \times N _ { h } }$ denote the encoder and decoder weight matrices, respectively. The network is trained on the mean-squared-error loss $\mathcal { L } _ { \mathrm { t a s k } } ( \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } ) \ =$ $\begin{array} { r } { \frac { 1 } { 2 } \left. \left\| \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } \mathbf { x } - \mathbf { y } \right\| _ { 2 } ^ { 2 } \right. } \end{array}$ , where ⟨·⟩ denotes the empirical average over the training set. Although the task loss depends only on the end-to-end mapping $\mathbf { M } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 }$ , its factorization across the two layers affects the parameter and representation dynamics. We characterize this factorization through the layer-imbalance matrix $\begin{array} { r } { \pmb { \Delta } \doteq \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { \dot { W } } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } } \end{array}$ . A network is said to be λ-balanced when $\mathbf { \Delta } \Delta \mathbf { \Psi } = \lambda \mathbf { I } _ { N _ { h } }$ . Previous work has shown that λ-balanced initializations provide an analytically tractable framework for characterizing the transition between rich and lazy learning dynamics (Domine et al., 2025; Nam et al., 2025). Under unregularized gradient flow, ´ $\pmb { \Delta }$ is conserved, so the layer balance remains fixed at its initial value. Prior studies have also linked these learning regimes to the rank of the representations Saxe et al. (2014); Atanasov et al. (2022). Motivated by this connection, we investigate whether a suitable layer balance can prevent representation collapse. We show that an appropriately chosen negative balance provides such a guarantee, and use this result to derive a regularizer that dynamically promotes the same anti-collapse property during training.

Balancedness anti-collapse property. A negative layer balance provides an explicit anti-collapse guarantee. If $\begin{array} { r } { \Delta = \lambda _ { \mathrm { b a l } } \bar { \bf I } _ { N _ { h } } ^ { - } } \end{array}$ with $\lambda _ { \mathrm { b a l } } < 0$ , then $\mathbf { W } _ { 1 } \mathbf { \bar { W } } _ { 1 } ^ { \top } \succeq - \lambda _ { \mathrm { b a l } } \mathbf { \bar { I } } _ { N _ { h } }$ , and $\sigma _ { \mathrm { m i n } } \mathrm { \bar { ( } W _ { 1 } ) } \geq \sqrt { - \lambda _ { \mathrm { b a l } } }$ so the encoder has full row rank. This also yields a guarantee for the hidden representations under the assumption that, for some $\kappa _ { x } > 0 .$ , the input covariance $\begin{array} { r } { \pmb { \Sigma } _ { x } = \frac { 1 } { P } \sum _ { n = 1 } ^ { P } ( \pmb { x } _ { n } - \bar { \mathbf { x } } ) ( \mathbf { x } _ { n } - \bar { \mathbf { x } } ) ^ { \top } \succeq } \end{array}$ $\kappa _ { x } \mathbf { I } _ { N _ { i } }$ , with $\begin{array} { r } { \bar { \mathbf { x } } = \frac { 1 } { P } \sum _ { n = 1 } ^ { P } \mathbf { x } _ { n } } \end{array}$ . In fact, the centered covariance of the hidden representations $\mathbf { h } _ { n } =$ $\mathbf { W } _ { 1 } \mathbf { x } _ { n }$ satisfies $\mathbf { C } _ { h } = \mathbf { W } _ { 1 } \boldsymbol { \Sigma } _ { x } \mathbf { W } _ { 1 } ^ { \top } \succeq - \kappa _ { x } \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } }$ . Therefore, negative layer balance prevents representation collapse, see Lemma A.4 in Appendix A.1.3 for a formal statement and proof.

Although a negative balanced initialization prevents collapse, extending the same initialization strategy to deep nonlinear networks is non-trivial. We therefore seek a regularized training objective that dynamically selects a negative layer balance, independently of its initial value. Motivated by this objective, we derive a regularizer that follows from a constrained optimization problem.

Layer balance through regularization. Among all factorizations of a fixed end-to-end mapping $\begin{array} { r } { \mathbf { M } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } } \end{array}$ , we look for a constrained minimizer that satisfies $\begin{array} { r } { \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { \tilde { I } } _ { N _ { h } } . } \end{array}$

Theorem 3.1. Fix an end-to-end mapping $\mathbf { M } \in \mathbb { R } ^ { N _ { o } \times N _ { i } } , \gamma > 0$ and $\lambda _ { \mathrm { { r e g } } } > 0 .$ . Let $( \mathbf { W } _ { 1 } ^ { \star } , \mathbf { W } _ { 2 } ^ { \star } )$ be a solution of

$$
\operatorname* { m i n } _ { \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } } \quad \frac { \lambda _ { \mathrm { r e g } } } { 2 } \left( \| \mathbf { W } _ { 1 } \| _ { F } ^ { 2 } + \| \mathbf { W } _ { 2 } \| _ { F } ^ { 2 } \right) - \frac { \gamma } { 2 } \log \operatorname* { d e t } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) \quad s u b j e c t \ t o \quad \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } = \mathbf { M } ,\tag{1}
$$

$$
\begin{array} { r } { w i t h \ \mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } \succ 0 . \ T h e n , \ ( \mathbf { W } _ { 2 } ^ { \star } ) ^ { \top } \mathbf { W } _ { 2 } ^ { \star } - \mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } } \end{array}
$$

The proof of Theorem 3.1 is provided in Appendix A.1.4. This motivates the following definition.

$$
\begin{array} { l } { { \displaystyle { \bf D e f i n i t i o n ~ 3 . 2 ~ ( L i n e a r ~ S A C R e g ) . ~ } A s s u m e ~ t h a t ~ } { \bf W } _ { 1 } { \bf W } _ { 1 } ^ { \top } ~ \succ ~ 0 , ~ \gamma > ~ 0 ~ a n d ~ \lambda _ { \mathrm { r e g } } ~ > ~ 0 . ~ W e }  \\ { { d e f i n e } } \\ { { \displaystyle ~ \mathcal { R } _ { \mathrm { L S A C } } ( { \bf W } _ { 1 } , { \bf W } _ { 2 } ) = \frac { \lambda _ { \mathrm { r e g } } } { 2 } ~ \big ( \| { \bf W } _ { 1 } \| _ { F } ^ { 2 } + \| { \bf W } _ { 2 } \| _ { F } ^ { 2 } \big ) - \frac { \gamma } { 2 } \log \operatorname* { d e t } \big ( { \bf W } _ { 1 } { \bf W } _ { 1 } ^ { \top } \big ) } , }  \\ { { \displaystyle ~ a n d ~ t h e ~ t r a i n i n g ~ o b j e c t i v e ~ \mathcal { L } _ { \mathrm { L S A C } } ( { \bf W } _ { 1 } , { \bf W } _ { 2 } ) = \mathcal { L } _ { \mathrm { t a s k } } \big ( { \bf W } _ { 1 } , { \bf W } _ { 2 } ) + \mathcal { R } _ { \mathrm { L S A C } } ( { \bf W } _ { 1 } , { \bf W } _ { 2 } ) . } } \end{array}
$$

The regularizer combines symmetric $L _ { 2 }$ weight decay with a log-determinant term acting specifically on the encoder Gram matrix. This placement is motivated by the non-collapse condition: a negative layer balance bounds the encoder Gram matrix away from being singular and under the input-covariance assumption, guarantees non-collapse of the hidden representations. The log-determinant term penalizes vanishing encoder singular values, while weight decay controls the overall weight scale.

We next note that the balance identified by the constrained optimization problem also emerges dynamically during training. If $N _ { h } \ \leq \ N _ { i } ^ { \cdot }$ and $\mathbf { W } _ { 1 } ( 0 ) \mathbf { W } _ { 1 } ( 0 ) ^ { \top } \ \succ \ 0 .$ , the regularized gradientflow solution for the mean-squared-error objective exists for all $t \ \geq \ 0 .$ , and the encoder retains full row rank throughout training. The task-loss terms cancel in the imbalance dynamics, giving $\tau \dot { \Delta } = - 2 \lambda _ { \mathrm { r e g } } \Delta - 2 \gamma { \bf I } _ { N _ { h } }$ . Consequently, $\begin{array} { r } { \pmb { \Delta } ( t ) + \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } = e ^ { - 2 \lambda _ { \mathrm { r e g } } t / \tau } \left( \pmb { \Delta } ( 0 ) + \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } \right) } \end{array}$ . Thus, independently of the dataset and initial imbalance, the network converges exponentially with rate $2 \lambda _ { \mathrm { r e g } } / \tau$ to $\lambda _ { \mathrm { b a l } } ^ { \mathrm { e n c } }$ -balanced solution, with $\lambda _ { \mathrm { b a l } } ^ { \mathrm { e n c } } = - \gamma / \lambda _ { \mathrm { r e g } }$ (see Theorem A.7 in Appendix A.1.5). The network preserves this balance throughout training (Corollary A.8). Unlike λ-balanced initialization, which fixes the balance only through the starting condition, the proposed objective continuously attracts the network towards the selected balance. The ratio $\gamma / \lambda _ { \mathrm { r e g } }$ determines the fixed point, while $\lambda _ { \mathrm { r e g } }$ controls the rate of convergence towards it. The ratio $\gamma / \lambda _ { \mathrm { r e g } }$ determines the selected balance and the corresponding lower bound on the encoder spectrum and yields a non-collapse guarantee. This motivates applying the penalty directly to the representation covariance in nonlinear encoders, as developed next.

## 3.2 SPECTRAL REGULARIZATION OF NONLINEAR REPRESENTATIONS

Although full matrix imbalance is not conserved in nonlinear networks, prior work has shown that it influences their learning dynamics (Kunin et al., 2024; Anguita et al., 2026; Jarvis et al., 2025). Motivated by this connection and our linear analysis, we apply the spectral anti-collapse principle directly to the encoder’s representation covariance, preserving the encoder-side placement without relying on exact balance dynamics. Let $\mathbf { h } = f _ { \theta } ( \mathbf { x } ) \in \mathbb { R } ^ { N _ { h } }$ denote the representation produced by a differentiable encoder $f _ { \theta }$ . For a minibatch of size B, collect the representations row-wise in $\dot { \mathbf { H } } \in \mathbb { R } ^ { B \times N _ { h } }$ and define $\begin{array} { r } { { \boldsymbol \mu } = \frac { 1 } { B } { \bf H } ^ { \top } { \bf 1 } _ { B } , { \bf P } _ { B } = { \bf I } _ { B } - \frac { 1 } { B } { \bf 1 } _ { B } { \bf 1 } _ { B } ^ { \top } , \widetilde { \bf H } = { \bf P } _ { B } { \bf H } , { \bf C } _ { h } = \frac { 1 } { B } \widetilde { \bf H } ^ { \top } \widetilde { \bf H } } \end{array}$ . Here, µ and $\mathbf { C } _ { h }$ are the empirical representation mean and centered covariance, respectively.

Definition 3.3 (Nonlinear SACReg). Let $\lambda _ { \mathrm { m e a n } } > 0 , \lambda _ { \mathrm { c o v } } > 0 , \gamma > 0 ,$ , and $\varepsilon \geq 0 .$ . Assume that $\mathbf { C } _ { h } + \varepsilon \mathbf { I } _ { N _ { h } } \succ 0$ . We define

$$
\begin{array} { c } { \displaystyle \mathcal { R } _ { \mathrm { S A C } } ( \mu , { \bf C } _ { h } ) = \frac { \lambda _ { \mathrm { m e a n } } } { 2 } \| \mu \| _ { 2 } ^ { 2 } + \frac { \lambda _ { \mathrm { c o v } } } { 2 } \operatorname { t r } ( { \bf C } _ { h } ) - \frac { \gamma } { 2 } \log \operatorname* { d e t } \left( { \bf C } _ { h } + \varepsilon { \bf I } _ { N _ { h } } \right) , } \\ { a n d t h e c o r r e s p o n d i n g t r a i n i n g o b j e c t i v e \mathcal { L } _ { \mathrm { S A C } } = \mathcal { L } _ { \mathrm { t a s k } } + \mathcal { R } _ { \mathrm { S A C } } ( \mu , { \bf C } _ { h } ) . } \end{array}\tag{2}
$$

The covariance terms play the same roles as in the linear setting: $\lambda _ { \mathrm { c o v } }$ controls the total representation variance, while $\gamma$ controls the strength of the log-determinant penalty on small covariance eigenvalues. The parameter ε provides numerical stabilization. The nonlinear regularizer also includes a mean penalty $\| \pmb { \mu } \| _ { 2 } ^ { 2 }$ , weighted by $\lambda _ { \mathrm { m e a n } }$ , which pulls the representation mean towards the origin. For a bias-free linear encoder, centered inputs produce centered representations, so no separate mean penalty is needed. For a linear encoder with whitened inputs, the covariance penalty reduces exactly to the encoder portion of the linear regularizer, with $\lambda _ { \mathrm { c o v } } = \lambda _ { \mathrm { r e g } }$ . This connection is derived in Appendix A.2.1.

Anti-collapse property. We next characterize the regularizer’s inductive bias and anti-collapse properties. The mean and trace penalties control representation location and scale, while the negative log-determinant penalizes small covariance eigenvalues.

Theorem 3.4 (Optimal mean and covariance). The unique minimizer of $\mathcal { R } _ { \mathrm { S A C } }$ over $\mu \in$ $\mathbb { R } ^ { N _ { h } } , ~ { \bf C } _ { h } ~ \succeq ~ 0$ such that ${ \bf C } _ { h } + \varepsilon { \bf I } _ { N _ { h } } \ \succ \ 0$ is $\begin{array} { r c l } { \pmb { \mu } ^ { \star } ~ = ~ \mathbf { 0 } , ~ \mathbf { C } _ { h } ^ { \star } ~ = ~ \left( \frac { \gamma } { \lambda _ { \mathrm { c o v } } } - \varepsilon \right) _ { + } \mathbf { I } _ { N _ { h } } } \end{array}$ , where $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$

Thus, the regularizer favors zero-mean representations with isotropic covariance, which is full rank when $\gamma > \lambda _ { \mathrm { c o v } } \varepsilon$ . For $\varepsilon = 0$ , the preferred variance is $\gamma / \lambda _ { \mathrm { c o v } }$ . These are the preferred moments of the regularizer alone; the task loss, network parameterization, and minibatch rank constraints may prevent their attainment.

When $\varepsilon \quad = \quad 0 .$ , the negative log-determinant is a strict barrier against covariance collapse: $\lambda _ { \operatorname* { m i n } } ( \mathbf { C } _ { h } ) \to 0$ implies $\bar { \mathcal { R } } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { \bar { C } } _ { h } )  + \infty$ . At the other extreme, the trace term prevents unbounded growth: $\bar { \lambda _ { \operatorname* { m a x } } } ( \mathbf { C } _ { h } ) \ \stackrel { \cdot } { \to } \ + \infty$ implies ${ \mathcal R } _ { \mathrm { S A C } } ( { \pmb \mu } , { \bf C } _ { h } )  + \infty$ . The mean term supplies a third, independent barrier: $\| \mu \| \to \infty$ implies ${ \mathcal { R } } _ { \mathrm { S A C } } ( { \boldsymbol { \mu } } , \mathbf { C } _ { h } ) \to + \infty$ , so the representation cannot escape the regularizer by drifting off in mean instead of collapsing or exploding in covariance. Consequently, every finite sublevel set has bounded mean and covariance eigenvalues bounded above and away from zero; see Theorem A.15 in Appendix A.2.2. We evaluate this property experimentally in nonlinear ReLU networks, following Kunin et al. (2024). Consistent with our predictions, the regularizer increases representation rank across initialization scales (Fig. 4 in Appendix A.2.3).

Altogether, these results suggest that SACReg promotes higher-rank representations in nonlinear networks, supporting its role as an anti-collapse regularizer while also preserving feature learning.

## 3.3 SPECTRAL ANTI-COLLAPSE FOR JE-SSL

As shown in Fig. 1, standard JE-SSL objectives can leave backbone representations dimensionally collapsed. To address this, we apply the spectral anti-collapse regularizer (SACReg), introduced in Section 3.2, to joint-embedding self-supervised learning (JE-SSL), targeting variation between images to prevent representation collapse. Standard JE-SSL objectives enforce invariance and prevent collapse in the projected space z. Motivated by our analysis of λ-balance and the resulting anti-collapse guarantees, our standalone SSL method, λ-JEPA, applies SACReg to both backbone and projected representations. For a batch of B images, each with V augmented views, let $\mathbf { h } _ { i } ^ { ( j ) } ~ = ~ \bar { f } _ { \theta } ( \mathbf { \bar { v } } _ { i } ^ { ( j ) } )$ , and $\mathbf { z } _ { i } ^ { ( j ) } = p _ { \phi } ( \mathbf { h } _ { i } ^ { ( j ) } )$ denote the backbone and projected representations, respectively. Define their view centers by $\begin{array} { r } { \bar { \mathbf { h } } _ { i } = \frac { 1 } { V } \sum _ { j = 1 } ^ { V } \mathbf { h } _ { i } ^ { ( j ) } , \bar { \mathbf { z } } _ { i } = \frac { 1 } { V } \sum _ { j = 1 } ^ { V } \mathbf { z } _ { i } ^ { ( j ) } } \end{array}$ , and write $\bar { \bf H } = \{ \bar { \bf h } _ { i } \} _ { i = 1 } ^ { B } , \bar { \bf Z } = \{ \bar { \bf z } _ { i } \} _ { i = 1 } ^ { B }$ , and $\mathbf { Z } = \{ \mathbf { z } _ { i } ^ { ( j ) } \} _ { i = 1 , \dots , B ; j = 1 , \dots , V }$ . For a batch of view centers $\mathbf { R } = \{ \mathbf { r } _ { i } \} _ { i = 1 } ^ { B } \subset \mathbb { R } ^ { d }$ , define its empirical mean and centered covariance as $\begin{array} { r } { { \pmb \mu } _ { R } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } { \bf r } _ { i } } \end{array}$ , and $\begin{array} { r } { \Sigma _ { R } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } ( { \bf r } _ { i } - { \pmb \mu } _ { R } ) ( { \bf r } _ { i } - { \pmb \mu } _ { R } ) ^ { \top } } \end{array}$ . We average over $V$ views so that SACReg mainly encourages variation across images rather than augmentations.

$$
\begin{array} { r l } & { \mathrm { D e f i n i t i o n } 3 . 5 \ : ( \mathrm { S A C R e g ~ f o r ~ J E - S S L } ) , \ : \ : F o r \varepsilon \ge 0 \ : s u c h \ : t h a t \ : \Sigma _ { R } + \varepsilon \mathbf { I } _ { d } \succ 0 , \ : w e d e f i n e } \\ & { \qquad \mathrm { S A C R e g } ( \mathbf { R } ) = \frac { 1 } { 2 } \left[ \| \mu _ { R } \| _ { 2 } ^ { 2 } + \mathrm { t r } ( \Sigma _ { R } ) - \log \operatorname* { d e t } ( \Sigma _ { R } + \varepsilon \mathbf { I } _ { d } ) - d \right] . } \\ & { G i v e n \ : a n i n v a r i a n c e o b j e c t i v e \ : \mathcal { L } _ { \mathrm { S S L } } ( \mathbf { Z } ) \ : \ : a n d \ : w e i g h t s \ : \beta _ { h } , \beta _ { z } > 0 , \ : t h e \ : \lambda \cdot J E P A \ : o b j e c t i v e \ : i s } \\ & { \qquad \mathcal { L } _ { \lambda \cdot J E P A } = \mathcal { L } _ { \mathrm { S S L } } ( \mathbf { Z } ) + \beta _ { h } \mathrm { ~ S A C R e g } ( \mathbf { \bar { H } } ) + \beta _ { z } \mathrm { ~ S A C R e g } ( \mathbf { \bar { Z } } ) . } \end{array}\tag{3}
$$

The mean term centers the representation, the trace controls its overall scale, and the negative log-determinant penalizes vanishing covariance directions. The projected-space term provides anticollapse where the SSL objective is optimized, while the backbone term directly protects the representation retained for downstream transfer. When applying our regularizer to an existing SSL method that already contains its own projected-space anti-collapse mechanism, we leave its original objective unchanged and add only the backbone term, $\beta _ { h } \mathrm { S \bar { A } C R e g } ( \bar { \bf H } )$ . In practice, when the feature dimension exceeds the number of image centers in a minibatch, we evaluate the regularizer over orthogonal lower-dimensional slices of the representation, details are given in Appendix B.

## 4 EXPERIMENTS

We evaluate whether SACReg improves the performance and transferability of representations. We first report the performance of the standalone λ-JEPA objective on image and video benchmarks, then use controlled ImageNet-100 (Tian et al., 2020a) experiments to isolate the effect of backbone regularization across diverse SSL objectives and analyze how it changes representation geometry.

## 4.1 SETUP

We consider two experimental settings. Our main evaluation covers large-scale image and video SSL. For images, we train ViT-S and ViT-B (Dosovitskiy et al., 2021) on ImageNet-1k (Russakovsky et al., 2015) for 100 and 400 epochs from scratch. For video, we follow LeVJEPA (Kuhn et al., 2026) and train ViT-S and ViT-B from scratch on a class-balanced 20% subset of Kinetics-710 (Li et al., 2023), for up to 1,085 epochs. Our controlled experiments use ViT-S on ImageNet-100 (Tian et al., 2020a) to study the effect of SACReg across different SSL objectives.

Due to compute limitations, we do not retune hyperparameters for the 400-epoch ImageNet-1k or 1,085-epoch video runs; we use these primarily to study scaling with extended training. For evaluation, we follow standard frozen feature protocols: the Lightly (Susmelj et al., 2020) linear probe benchmark on ImageNet-1k, the VISReg (Wu et al., 2026) pipeline for transfer and the LeVJEPA pipeline for video. Details are given in Appendix C. Whenever public checkpoints trained under a common codebase are available (OpenKnowledge AI<sup>2</sup>), we evaluate them under the same pipeline, otherwise we report published numbers. We set the loss weights using the gradient-based calibration procedure described in Appendix C.3.

## 4.2 PERFORMANCE ON IMAGE SELF-SUPERVISED LEARNING

Table 1a reports our main 100-epoch results on ImageNet-1k linear probing and average transfer across eight classification datasets. We compare most directly with LeJEPA (Balestriero & LeCun, 2025) and VISReg (Wu et al., 2026), the related explicitly regularized JE-SSL methods. We include DINO (Caron et al., 2021) and iBOT (Zhou et al., 2022) as established SSL baselines for broader comparison which, until now, have significantly outperformed JEPA-based models. At 100 epochs, where training length is matched, λ-JEPA substantially improves over prior explicitly regularized methods on both ImageNet-1k linear probing and transfer. The gains are particularly strong on transfer, improving over LeJEPA by 8.9 points with ViT-S and over VISReg by 4.5 points with ViT-B. λ-JEPA is also competitive with DINO and iBOT: its ImageNet-1k performance approaches these methods, while both model sizes achieve the highest average transfer among the 100-epoch results.

(a) Main: 100 epochs
<table><tr><td>Method</td><td>Backbone</td><td>Linear</td><td>Transfer</td></tr><tr><td>LeJEPA</td><td>ViT-S/16</td><td>62.5</td><td>68.7</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>70.0</td><td>75.7</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>70.9</td><td>75.9</td></tr><tr><td>λ-JEPA</td><td>ViT-S/16</td><td>69.7</td><td>77.6</td></tr><tr><td>LeJEPA</td><td>ViT-B/16</td><td>69.7</td><td>74.2</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>73.9</td><td>78.3</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>77.0</td><td>80.7</td></tr><tr><td>VISReg</td><td>ViT-B/16</td><td>70.3</td><td>76.4</td></tr><tr><td>λ-JEPA</td><td>ViT-B/16</td><td>74.2</td><td>80.9</td></tr></table>

(b) Longer training
<table><tr><td>Method</td><td>Backbone</td><td>Ep.</td><td>Linear</td><td>Transfer</td></tr><tr><td>LeJEPA</td><td>ViT-S/16</td><td>300</td><td>66.4</td><td>71.9</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>300</td><td>73.8</td><td>79.4</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>300</td><td>75.2</td><td>79.5</td></tr><tr><td>λ-JEPA</td><td>ViT-S/16</td><td>400</td><td>72.3</td><td>79.3</td></tr><tr><td>LeJEPA</td><td>ViT-B/16</td><td>300</td><td>72.4</td><td>75.2</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>300</td><td>75.2</td><td>79.3</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>300</td><td>78.6</td><td>82.3</td></tr><tr><td> $\mathrm { M o C o } \mathrm { v } 3 ^ { \ddagger }$ </td><td>ViT-B/16</td><td>300</td><td>75.9</td><td>80.5</td></tr><tr><td> $\mathrm { D I N O ^ { \ddagger } }$ </td><td>ViT-B/16</td><td>400</td><td>77.2</td><td>83.1</td></tr><tr><td> $\mathrm { i B O T ^ { \ddagger } }$ </td><td>ViT-B/16</td><td>400</td><td>78.5</td><td>83.2</td></tr><tr><td> $\mathrm { V I S R e g ^ { \ddag } }$ </td><td>ViT-B/16</td><td>400</td><td>74.6</td><td>79.1</td></tr><tr><td> $\lambda { \mathrm { - J E P A } }$ </td><td>ViT-B/16</td><td>400</td><td>76.0</td><td>82.0</td></tr></table>

Table 1: ImageNet-1k linear probing and mean linear-probe transfer over eight datasets. (a) Main 100-epoch comparison. (b) Longer-training results, including public SSL checkpoints for context. <sup>‡</sup> Transfer results are reported by VISReg; all others are evaluated by us under the matched VISReg protocol. Detailed results are in Appendix C.6.

Table 1b reports results with extended pretraining: we use the same recipe and hyperparameters as the 100-epoch experiments, without any additional tuning for the longer runs. Extending λ-JEPA to 400 epochs improves both ImageNet-1k linear probing and transfer performance. Since the available checkpoints use different training lengths, we include them as longer training comparisons rather than strictly matched baselines. With ViT-B, λ-JEPA reaches 82.0 average transfer, outperforming the 400-epoch VISReg result by 2.9 points and coming within 1.2 points of DINO and iBOT. With ViT-S, it reaches 79.3 average transfer, essentially matching the 300-epoch DINO and iBOT results.

Takeaway. λ-JEPA substantially improves over prior explicitly regularized JE-SSL methods on both ImageNet-1k classification and transfer. To the best of our knowledge, it is the first JEPAbased method to approach DINO and iBOT on both evaluations, while achieving the highest average transfer performance at 100 epochs.

## 4.3 PERFORMANCE ON VIDEO SELF-SUPERVISED LEARNING

We apply the same objective to video by replacing the loss inside the LeVJEPA (Kuhn et al., 2026) training pipeline while keeping the model, optimizer and data pipeline unchanged. We train these models from scratch and freeze the encoders after pretraining. The frozen encoders are evaluated on three benchmarks with different characteristics: ImageNet-1k for recognition from a single image, Something-Something-v2 (SSv2) (Goyal et al., 2017) for temporal reasoning and Kinetics-400 (Kay et al., 2017) for action recognition.

Table 2 compares λ-JEPA with LeVJEPA, V-JEPA 2 (Assran et al., 2025) and VideoMAEv2 (Wang et al., 2023). At 240 epochs, λ-JEPA improves over LeVJEPA and V-JEPA 2 on ImageNet-1k for both ViT-S and ViT-B. The largest gains are on SSv2: our ViT-B improves over LeVJEPA by 13.5 points at 240 epochs and by 7.9 points at 1,085 epochs. For ImageNet-1k and SSv2, LeVJEPA evaluates frozen features with an attentive probe followed by a linear classifier, whereas on K400 it reports a linear probe on mean-pooled features. In our setting, SACReg is applied to the CLS representation, so we report a CLS probe in Table 2, evaluating the representation directly regularized during pretraining. It reaches 45.7 at 1,085 epochs compared with 44.6 for LeVJEPA’s probe over the mean-pooled features. For completeness, we also evaluate the same frozen encoder using mean-pooled features and an attentive probe, obtaining 43.4 and 62.0, respectively. The lower mean-pooled result is consistent with these features not being directly regularized in our setup.

<table><tr><td>Method IN-1k</td></tr><tr><td>SSv2 K400 ViT-S/16, 240 epochs</td></tr><tr><td>LeVJEPA 39.4 V-JEPA 2 38.7</td></tr><tr><td>λ-JEPA 46.8 37.3 40.0</td></tr><tr><td>ViT-B/16, 240 epochs</td></tr><tr><td>VideoMAEv2 47.1 V-JEPA 2 51.6</td></tr><tr><td>LeVJEPA 50.7 30.4 λ-JEPA 52.9 43.9 44.2</td></tr><tr><td>ViT-B/16, longer training</td></tr><tr><td>VideoMAEv2 53.4 43.6 37.4 40.7</td></tr><tr><td>V-JEPA 2 51.6 42.5</td></tr><tr><td>LeVJEPA 61.0 40.4 44.6 λ-JEPA 57.1 48.3 45.7</td></tr></table>

Table 2: Video SSL on IN-1k, SSv2, and K400. Longer training LeVJEPA and λ-JEPA use 1085 epochs.

![](images/648971d68515723722764a64170523358eb31afcaf8239705b4fd7540605882d.jpg)

![](images/d0c4bd61740f17cc7a6b71f0c84488a4ed0ab2c42e2a8cbb3784ab7da5824ae8.jpg)  
Figure 2: Effect of backbone SACReg across SSL objectives on ImageNet-100. Top: backbone statistics and downstream evaluation. Bottom: class-center separability versus view sensitivity along class-prediction directions. Open and green markers denote baseline and +SACReg, respectively; arrows show their change. Detailed results are in Appendix C.6.

Takeaway. λ-JEPA extends effectively to video SSL, with its clearest gains on temporal evaluations, showing that the benefits of backbone spectral regularization transfer beyond image pretraining.

## 4.4 CONTROLLED EXPERIMENTS ON IMAGENET-100

We next use controlled ImageNet-100 experiments to better understand the effects of SACReg across SSL objectives. Our analysis focuses on two questions: how SACReg changes representation geometry and downstream performance, and what variation is recovered in the additional representation directions. To this end, we apply SACReg to the backbone representations of six existing augmentation-based SSL methods on ImageNet-100 (Tian et al., 2020a): LeJEPA (Balestriero & Le-Cun, 2025), VICReg (Bardes et al., 2022), VISReg (Wu et al., 2026), SimCLR (Chen et al., 2020), DINO (Caron et al., 2021), and BYOL (Grill et al., 2020). For each method, we keep its original objective and training recipe unchanged and add only $\beta _ { \mathrm { h } }$ SACReg to the backbone representation. The regularizer weight is set using the gradient based calibration procedure (see Appendix C.3 for details) and is not tuned individually to the methods except DINO. For DINO we tune the weight separately due to its sensitivity to the gradient ratios.

Figure 2 (top) shows that applying SACReg to the backbones produces a consistent change in the representations across objectives. At the backbone, RankMe increases substantially for every method, while the cosine geometry becomes less concentrated. Negative-pair cosine similarities approach zero while positive-pairs generally decrease as well. In contrast, corresponding statistics in the projection/loss space change much less. This suggests that SACReg primarily reshapes the backbone representation without substantially impeding the learning after the projection. The same intervention also improves downstream performance. kNN accuracy improves across all methods convincingly, while linear probe accuracy also improves, with smaller gains for VICReg. Notably, the gains therefore do not come from making positive pairs more invariant.

Augmentation thickness We next distinguish variation between images from variation across augmented views of the same image. Let $Q$ denote a source image and h its backbone representation under a random augmentation. We define $\mathbf { B } _ { h } = \operatorname { C o v } \left( \mathbb { E } [ \mathbf { h } \mid \boldsymbol { \breve { Q } } ] \right)$ and ${ \bf A } _ { h } = \mathbb { E } \left[ \mathrm { C o v } ( { \bf h } \mid Q ) \right]$ , where $\mathbf { B } _ { h }$ captures variation between image centers, with $r _ { h } = \mathrm { r a n k } ( { \bf B } _ { h } )$ , and ${ \bf A } _ { h }$ captures variation across views of the same image. We then define the augmentation thickness

$$
\begin{array} { r } { \Theta _ { h } = \mathbf { B } _ { h } ^ { \dagger / 2 } \mathbf { A } _ { h } \mathbf { B } _ { h } ^ { \dagger / 2 } , \qquad \theta _ { h } ( \mathbf { a } ) = \frac { \mathbf { a } ^ { \top } \mathbf { A } _ { h } \mathbf { a } } { \mathbf { a } ^ { \top } \mathbf { B } _ { h } \mathbf { a } } , } \end{array}\tag{4}
$$

so thickness measures augmentation variation relative to the separation of image centers along that direction. Appendix E shows that thickness has a two-sided role: too little can discard meaningful variation across augmentations, while too much allows within-image variation to dominate the separation between images.

To study thickness, we use the class indicators. For each class $c ,$ let $\mathbf { a } _ { c }$ be its linear prediction from whitened image centers and $d _ { h } ( c ) ^ { 2 }$ the corresponding error. We measure class-center separability by $1 - d _ { h } ( c ) ^ { 2 }$ and view sensitivity by $\mathbf { a } _ { c } ^ { \top } \Theta _ { h } ( \dot { \mathbf { I } } _ { r _ { h } } + \check { \Theta } _ { h } ) ^ { - 1 } \mathbf { a } _ { c }$ . Details are in Appendix C.

Figure 2 (bottom) shows that, averaged across classes, all six SSL methods have greater class-center separability and greater view sensitivity under SACReg. Therefore, SACReg’s downstream gains are not obtained by suppressing augmentation-dependent variation. Specifically, the more consistent effect is an improvement in how image centers are organized along downstream-relevant directions. For λ-JEPA, class-center separability improves for 96 of 100 classes, whereas view sensitivity moves in both directions. SACReg therefore consistently improves class-center separation without uniformly moving the representation toward either invariance or augmentation sensitivity.

Takeaway. Across diverse SSL objectives, SACReg consistently increases representation rank and improves downstream performance. The augmentation thickness analysis shows that these gains are not explained by increased augmentation invariance. SACReg improves class-center separation while preserving augmentation variation.

## 5 CONCLUSION

In this paper, we targeted the asymmetry between anti-collapse regularization in joint embedding self-supervised models and the representation used for downstream tasks. First, we showed that preventing collapse in the projected representation does not guarantee a high rank backbone representation. This matters because a low rank backbone reduces the representational capacity, which can harm downstream performance. Our solution is directly motivated by a linear analysis of neg ative layer balance, which is known as $\lambda$ balance in the feature learning literature. From this, we introduced SACReg, which directly regularizes the representation covariance, and used it to build λ-JEPA. In controlled experiments, SACReg consistently increases backbone rank and improves downstream performance while preserving augmentation-dependent variation. At scale, λ-JEPA narrows the gap between explicitly regularized JEPA methods and DINO, while improving frozen transfer across image and video SSL benchmarks.

SACReg regularizes CLS tokens and uses global views. Extending it to patch-level features and local crops is non-trivial: local views introduce substantially larger augmentation-induced variation, making the per-image view centers noisier and the covariance across centers harder to regularize reliably. In our experiments, this led to unstable training. Understanding how different augmenta tions shape this variation and which augmentation-sensitive directions are useful for downstream tasks remains an open direction for future work. Developing a formulation that more robustly regularizes spatially local representations while separating augmentation variation from betweenimage variation could further improve performance on dense prediction tasks. Additionally, JEPA-style models have already been connected with world modeling and identifiabilty theory for learning causal representations (Klindt et al., 2026). This is conceptually natural, as JEPA models optimize the same principles of causal representations: invariance to relevant transformations and anti-collapse (Yao et al., 2025). However, methods in causal representation learning leverage mostly negative samples (and sometimes reconstruction) to implement the anti-collapse properties. We encourage future work to explore the role of our regularizer in the identifiability of world models.

## AI USE STATEMENT

We used generative AI tools as coding assistants to support the implementation and modification of the codebase and for language editing. These tools were not used to autonomously generate research results or replace the authors’ scientific judgment. The authors determined the research methodology and experimental design, reviewed and validated all AI-assisted implementations.

## ETHICS STATEMENT

This work studies methods for self-supervised representation learning using established image and video datasets and does not involve human-subject experiments or the collection of new personal or sensitive data. We are not aware of specific ethical concerns introduced by the proposed method beyond those generally associated with large-scale representation learning and computer vision models. The work is intended as a general-purpose methodological contribution and we encourage appropriate consideration of dataset provenance, bias and privacy risks when deploying learned representations.

## REPRODUCIBILITY STATEMENT

We provide detailed information required to reproduce our theoretical and empirical results throughout the paper and appendices. The assumptions, statements and proofs supporting the theoretical results are provided in Appendix A. For the empirical results, implementation details for SACReg are given in Appendix B, Appendix C includes information about datasets, architectures, training procedures, loss-weight calibration, evaluation pipelines and baselines used in our experiments. Additional analyses and their definitions are provided in Appendices D and E. We reuse public baseline implementations and checkpoints whenever available, and explicitly document differences in training and evaluation protocols when exact matching is not possible. In particular, for video SSL, some baselines use different frozen-feature evaluation protocols (e.g., CLS, mean-pooled, or attentive probes); we report the corresponding protocols and evaluate our representations under multiple probe choices to facilitate comparison. Furthermore, we provide supplementary material containing the code used to run every experiment reported in the paper. These materials are intended to enable reproduction of both the proposed method and the empirical evaluations.

## ACKNOWLEDGMENTS

This research was funded in whole or in part by the Austrian Science Fund (FWF) [10.55776/COE12]. CD thank Samuel Liplle and Nicolas Aguita for their constructive discussions. This research was supported by the ISTA Responsible AI Program, made possible through the generous support of Garrett Camp and the Camp Foundation.

## REFERENCES

Nicolas Anguita, Francesco Locatello, Andrew M Saxe, Marco Mondelli, Flavia Mancini, Samuel Lippl, and Clementine Domine. A theory of how pretraining shapes inductive bias in fine-tuning. arXiv preprint arXiv:2602.20062, 2026.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khali dov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners: The silent alignment effect. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=1NvflqAdoom.

Shahar Azulay, Edward Moroshko, Mor Shpigel Nacson, Blake E Woodworth, Nathan Srebro, Amir Globerson, and Daniel Soudry. On the implicit bias of initialization shape: Beyond infinitesimal mirror descent. In International Conference on Machine Learning, pp. 468–477. PMLR, 2021.

Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53–58, 1989.

Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.

A. Bardes, J. Ponce, and Y. LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.

Florian Bordes, Randall Balestriero, Quentin Garrido, Adrien Bardes, and Pascal Vincent. Guillotine regularization: Why removing layers is needed to improve generalization in self-supervised learning. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https: //openreview.net/forum?id=ZgXfXSz51n.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In European conference on computer vision, pp. 446–461. Springer, 2014.

Lukas Braun, Clementine Carla Juliette Domin´ e, James E Fitzgerald, and Andrew M Saxe. Exact´ learning dynamics of deep linear networks with prior knowledge. In Advances in Neural Information Processing Systems, 2022.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9650–9660, October 2021.

T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the International Conference on Machine Learning, 2020.

Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In Conference on learning theory, pp. 1305–1338. PMLR, 2020.

Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. Advances in Neural Information Processing Systems, 32, 2019.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613, 2014.

Clementine Carla Juliette Domin´ e, Nicolas Anguita, Alexandra Maria Proca, Lukas Braun, Daniel´ Kunin, Pedro A. M. Mediano, and Andrew M Saxe. From lazy to rich: Exact learning dynamics in deep linear networks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=ZXaocmXc6d.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=YicbFdNTTy.

Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International conference on machine learning, pp. 1675– 1685. PMLR, 2019.

Linus Ericsson, Henry Gouk, and Timothy M. Hospedales. How well do self-supervised models transfer? In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5414–5423, June 2021.

Kenji Fukumizu. Effect of batch learning in multilayer neural networks. In International Conference on Neural Information Processing, 1998. URL https://api.semanticscholar.org/ CorpusID:605683.

Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann LeCun. RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank. In Proceedings ofthe International Conference on Machine Learning, 2023.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The ”something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.

Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre Richemond, Elena´ Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias in terms of optimization geometry. In International Conference on Machine Learning, pp. 1832– 1841. PMLR, 2018.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and gen-´ eralization in neural networks. Advances in neural information processing systems, 31, 2018.

Devon Jarvis, Sebastian Lee, Clementine Carla Juliette Domin´ e, Andrew M Saxe, and Stefano´ Sarao Mannelli. A theory of initialisation’s impact on specialisation. Journal of Statistical Mechanics: Theory and Experiment, 2025(11):114001, 2025.

Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=YevsQ05DEN7.

Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. IEEE transactions on pattern analysis and machine intelligence, 43(11):4037–4058, 2021.

Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

David Klindt, Yann LeCun, and Randall Balestriero. When does lejepa learn a world model? arXiv preprint arXiv:2605.26379, 2026.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pp. 3519– 3529. PMLR, 2019.

Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pp. 554–561, 2013.

Alex Krizhevsky. Learning multiple layers of features from tiny images. https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf, 2009.

Lukas Kuhn, Lucas Maes, Giuseppe Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, and Florian Buettner. Levjepa: Efficient & scalable video pretraining without the heuristics. arXiv preprint arXiv:2608.27395, 2026.

Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, and Cengiz Pehlevan. Grokking as the transition from lazy to rich training dynamics. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=vt5mnLVIVo.

Daniel Kunin, Allan Raventos, Cl´ ementine Domin´ e, Feng Chen, David Klindt, Andrew Saxe,´ and Surya Ganguli. Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 81157–81203. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-2580. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Unlocking the potential of image vits for video understanding. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1632–1643. IEEE, 2023.

Zhiyuan Li, Yuping Luo, and Kaifeng Lyu. Towards resolving the implicit bias of gradient descent for matrix factorization: Greedy low-rank learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=AHOs7Sm5H7R.

Tao Luo, Zhi-Qin John Xu, Zheng Ma, and Yaoyu Zhang. Phase diagram for two-layer relu neural networks at infinite-width limit. Journal ofMachine Learning Research, 22(71):1–47, 2021.

Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

Yoonsoo Nam, Seok Hyeong Lee, Clementine Carla Juliette Domin´ e, Yeachan Park, Charles´ London, Wonyl Choi, Niclas Alexander Goring, and Seungjai Lee. Position: Solve layer-¨ wise linear models first to understand neural dynamical phenomena (Neural collapse, emergence, Lazy/Rich regime, and grokking). In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 81897–81929. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/nam25a.html.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pp. 722–729. IEEE, 2008.

Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal ofcomputer vision, 115(3):211–252, 2015.

Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In International Conference on Learning Representations (ICLR), 2014. URL https://openreview.net/forum?id=\_wzZwKpTDF\_9C.

Andrew M Saxe, James L McClelland, and Surya Ganguli. A mathematical theory of semantic development in deep neural networks. Proceedings of the National Academy of Sciences, 116 (23):11537–11546, 2019.

Igor Susmelj, Matthias Heller, Philipp Wirth, Jeremy Prescott, and Malte Ebner. Lightly, 2020. URL https://github.com/lightly-ai/lightly.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In European conference on computer vision, pp. 776–794. Springer, 2020a.

Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? Advances in neural information processing systems, 33:6827–6839, 2020b.

Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae $\mathbf { v } 2 \colon$ Scaling video masked autoencoders with dual masking. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14549–14560. IEEE, 2023.

Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In Conference on Learning Theory, pp. 3635–3673. PMLR, 2020.

Haiyu Wu, Randall Balestriero, and Morgan Levine. VISReg: Variance-invariance-sketching regularization for JEPA training. arXiv preprint arXiv:2606.02572, 2026.

Dingling Yao, Dario Rancati, Riccardo Cadei, Marco Fumero, and Francesco Locatello. Unifying causal representation learning with the invariance principle. In International Conference on Learning Representations, volume 2025, pp. 53847–53890, 2025.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=ydopy-e6Dg.

## A A NON-COLLAPSE REGULARIZER INSPIRED BY THE FEATURE-LEARNING LITERATURE

## A.1 LINEAR NETWORK

## A.1.1 PROBLEM SETUP

Consider a supervised learning problem with training data $\left\{ \left( \mathbf { x } _ { n } , \mathbf { y } _ { n } \right) \right\} _ { n = 1 } ^ { P } , \mathbf { x } _ { n } \in \mathbb { R } ^ { N _ { i } } , \mathbf { y } _ { n } \in \mathbb { R } ^ { N _ { o } }$ Let ${ \bf X } = [ { \bf x } _ { 1 } \mathrm { ~  ~ \cdot ~ } \cdot \mathrm { ~ \bf ~ x } _ { P } ] \in \mathbb { R } ^ { N _ { i } \times P }$ , and ${ \bf Y } = [ { \bf y } _ { 1 } \mathrm { ~  ~ \cdot ~ } \cdot \mathrm { ~ \bf ~ y } _ { P } ] \in \mathbb { R } ^ { N _ { o } \times P }$ . We consider a twolayer linear network $\widehat { \mathbf { y } } _ { n } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } \mathbf { x } _ { n }$ , with encoder and decoder matrices $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { N _ { h } \times N _ { i } }$ $\mathbf { W } _ { 2 } \in$ $\mathbb { R } ^ { \mathbf { \breve { N } } _ { o } \times N _ { h } }$ . The corresponding end-to-end mapping is $\begin{array} { r } { \mathbf { M } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } } \end{array}$

The empirical mean-squared error is

$$
\mathcal { L } _ { \mathrm { t a s k } } ( \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } ) = \frac { 1 } { 2 P } \left\| \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } \mathbf { X } - \mathbf { Y } \right\| _ { F } ^ { 2 } .\tag{5}
$$

Throughout this section, we consider continuous-time gradient flow, with time constant $\tau > 0 \mathrm { : }$

$$
\begin{array} { r } { \tau \dot { \mathbf { W } } _ { 1 } = - \nabla \mathbf { w } _ { 1 } \mathcal { L } _ { \mathrm { t a s k } } , \qquad \tau \dot { \mathbf { W } } _ { 2 } = - \nabla \mathbf { w } _ { 2 } \mathcal { L } _ { \mathrm { t a s k } } . } \end{array}\tag{6}
$$

Definition A.1 (λ-balancedness). Define the layer imbalance matrix by

$$
\begin{array} { r } { \pmb { \Delta } ( \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } ) = \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \in \mathbb { R } ^ { N _ { h } \times N _ { h } } . } \end{array}\tag{7}
$$

The network is said to be $\lambda _ { \mathrm { b a l } }$ -balanced $i f \Delta ( \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } ) = \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } }$ . The case $\lambda _ { \mathrm { b a l } } = 0$ corresponds to standard balancedness.

## A.1.2 BALANCE CONSERVATION WITHOUT EXPLICIT REGULARIZATION

For completeness we first recall the balance-conservation property of two-layer linear networks trained without explicit regularization, also derived in Domine et al. (2025).´

Lemma A.2 (Conservation of layer imbalance). Suppose that the network is trained by gradient flow on a differentiable loss of the form $\mathcal { L } _ { \mathrm { t a s k } } ( \mathbf { W } _ { 1 } , \hat { \mathbf { W } } _ { 2 } ) = \ell ( \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } )$ , without weight decay or a log-determinant regularizer. Then

$$
\frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) = \mathbf { 0 } .\tag{8}
$$

Consequently, if the network is $\lambda _ { \mathrm { b a l } }$ -balanced at initialization, then it remains $\lambda _ { \mathrm { b a l } }$ -balanced throughout gradient-flow training.

Proof. Let $\mathbf { G } = \nabla _ { \mathbf { M } } \ell ( \mathbf { M } ) \big | _ { \mathbf { M } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } }$ . The gradient-flow equations for the unregularized task loss are

$$
\tau \dot { \mathbf { W } } _ { 1 } = - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } , \qquad \tau \dot { \mathbf { W } } _ { 2 } = - \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } .
$$

Therefore,

$$
\tau \frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } \right) = - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } ,\tag{9}
$$

$$
\tau \frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) = - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } .\tag{10}
$$

The two expressions are identical. Subtracting them proves equation 8.

Remark ${ \mathrm { A } } . 3$ (Gradient flow versus gradient descent). The conservation law in Lemma $\mathrm { A } . 2$ is exact under continuous-time gradient flow. For finite-step gradient descent, additional terms of order $\eta ^ { 2 }$ generally appear, so exact conservation need not hold for a nonzero learning rate η.

## A.1.3 NON-COLLAPSE PROPERTY OF BALANCED INITIALIZATION

We first show that a negative layer balance is sufficient to prevent encoder collapse. Under a positivedefiniteness assumption on the input covariance, it also prevents collapse of the hidden representations.

Lemma A.4 (Non-collapse under negative layer balance). Let $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { N _ { h } \times N _ { i } }$ and $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { N _ { o } \times N _ { h } }$ satisfy

$$
\mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } = \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } } , \qquad \lambda _ { \mathrm { b a l } } < 0 .\tag{11}
$$

Then $\mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \succeq - \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } }$ . Consequently, $\mathbf { W } _ { 1 }$ has full row rank, with $\sigma _ { \mathrm { m i n } } ( \mathbf { W } _ { 1 } ) \geq \sqrt { - \lambda _ { \mathrm { b a l } } }$

Furthermore, let $\begin{array} { r } { \bar { \mathbf { x } } = P ^ { - 1 } \sum _ { n = 1 } ^ { P } \mathbf { x } _ { n } } \end{array}$ and suppose that the centered empirical input covariance satisfies $\begin{array} { r } { \pmb { \Sigma } _ { x } = \frac { 1 } { P } \sum _ { n = 1 } ^ { P } ( \pmb { x } _ { n } - \bar { \mathbf { x } } ) ( \mathbf { x } _ { n } - \bar { \mathbf { x } } ) ^ { \top } \succeq \kappa _ { x } \mathbf { I } _ { N _ { i } } } \end{array}$ for some $\kappa _ { x } > 0$ . For the hidden representations $\begin{array} { r } { \mathbf h _ { n } = \mathbf W _ { 1 } \mathbf x _ { n } , l e t \bar { \mathbf h } = P ^ { - 1 } \sum _ { n = 1 } ^ { P } \mathbf h _ { n } } \end{array}$ . Their centered empirical covariance then satisfies

$$
\boxed { \mathbf { C } _ { h } = \frac { 1 } { P } \sum _ { n = 1 } ^ { P } ( \mathbf { h } _ { n } - \bar { \mathbf { h } } ) ( \mathbf { h } _ { n } - \bar { \mathbf { h } } ) ^ { \top } \succeq - \kappa _ { x } \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } } } .\tag{12}
$$

Proof. Rearranging equation 11 gives

$$
\mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } = \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } } \succeq - \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } } \succ 0 ,
$$

since $\mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } \succeq 0$ and $\lambda _ { \mathrm { b a l } } < 0$ . This establishes the encoder rank and singular-value bounds.

By linearity, $\bar { \mathbf { h } } = \mathbf { W } _ { 1 } \bar { \mathbf { x } } .$ so $\mathbf { h } _ { n } - \bar { \mathbf { h } } = \mathbf { W } _ { 1 } ( \mathbf { x } _ { n } - \bar { \mathbf { x } } )$ . Therefore,

$$
\mathbf { C } _ { h } = \mathbf { W } _ { 1 } \pmb { \Sigma } _ { x } \mathbf { W } _ { 1 } ^ { \top } \succeq \kappa _ { x } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \succeq - \kappa _ { x } \lambda _ { \mathrm { b a l } } \mathbf { I } _ { N _ { h } } \succ 0 ,
$$

which proves the representation non-collapse bound.

Thus, the representation covariance is positive definite, with variance at least $- \kappa _ { x } \lambda _ { \mathrm { b a l } } ~ > ~ 0$ in every unit direction. This rules out complete collapse, in which all hidden representations coincide $( { \bf C } _ { h } = { \bf 0 } )$ , and, more generally, dimensional collapse, in which the hidden representations lie in a proper affine subspace of $\mathbb R ^ { N _ { h } } \mathbf { \bar { \Phi } } ( \mathrm { r a n k } ( \mathbf { C } _ { h } ) < N _ { h } )$

Combining Lemma A.2 with Lemma $_ { \mathrm { A . 4 , } }$ we conclude that a network initialized with a negative layer balance retains the corresponding non-collapse bounds throughout unregularized gradient-flow training. This guarantee depends on the balance imposed at initialization.

## A.1.4 CONSTRAINED OPTIMIZATION CHARACTERIZATION

Lemma A.4 identifies negative layer balance as a sufficient condition for non-collapse. We next show that the proposed regularizer selects precisely such a balance among factorizations of a fixed end-to-end mapping. The regularizer arises from a constrained minimum-regularization problem.

Theorem A.5 (Balancedness via Constrained Optimization ). Fix an end-to-end mapping $\textbf { M } \in$ $\mathbb { R } ^ { N _ { o } \times N _ { i } } , \gamma > 0 , \lambda _ { \mathrm { r e g } } > 0$ and consider

$$
\begin{array} { r l } { \underset { { \bf W } _ { 1 } , { \bf W } _ { 2 } } { \mathrm { m i n } } } & { \frac { \lambda _ { \mathrm { r e g } } } { 2 } \left( \| { \bf W } _ { 1 } \| _ { F } ^ { 2 } + \| { \bf W } _ { 2 } \| _ { F } ^ { 2 } \right) - \frac { \gamma } { 2 } \log \operatorname* { d e t } \left( { \bf W } _ { 1 } { \bf W } _ { 1 } ^ { \top } \right) } \\ { s u b j e c t \ t o } & { { \bf W } _ { 2 } { \bf W } _ { 1 } = { \bf M } . } \end{array}\tag{13}
$$

Let $( \mathbf { W } _ { 1 } ^ { \star } , \mathbf { W } _ { 2 } ^ { \star } )$ be a constrained minimizer with $\mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } \succ 0$ . Then

$$
\boxed { ( \mathbf { W } _ { 2 } ^ { \star } ) ^ { \top } \mathbf { W } _ { 2 } ^ { \star } - \mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } } .\tag{14}
$$

Proof. Introduce a Lagrange multiplier $\pmb { \Lambda } \in \mathbb { R } ^ { N _ { o } \times N _ { i } }$ for the constraint $\mathbf { W } _ { 2 } \mathbf { W } _ { 1 } = \mathbf { M }$ . The Lagrangian is

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } , \mathbf { A } ) = \frac { \lambda _ { \mathrm { r e g } } } { 2 } \left( \| \mathbf { W } _ { 1 } \| _ { F } ^ { 2 } + \| \mathbf { W } _ { 2 } \| _ { F } ^ { 2 } \right) - \frac { \gamma } { 2 } \log \operatorname* { d e t } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) } \\ { \displaystyle \quad + \left. \mathbf { A } , \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } - \mathbf { M } \right. _ { F } . } \end{array}\tag{15}
$$

The first-order stationarity conditions are

$$
{ \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } } - \gamma \left( { \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } } \right) ^ { - 1 } \mathbf { W } _ { 1 } + \mathbf { W } _ { 2 } ^ { \top } \mathbf { A } = \mathbf { 0 } ,\tag{16}
$$

$$
\lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } + \pmb { \Lambda } \mathbf { W } _ { 1 } ^ { \top } = \mathbf { 0 } .\tag{17}
$$

Multiplying equation 16 on the right by $\mathbf { W } _ { 1 } ^ { \top }$ gives

$$
\lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } - \gamma \mathbf { I } _ { N _ { h } } + \mathbf { W } _ { 2 } ^ { \top } \mathbf { A } \mathbf { W } _ { 1 } ^ { \top } = \mathbf { 0 } .\tag{18}
$$

Multiplying equation 17 on the left by $\mathbf { W } _ { 2 } ^ { \top }$ gives

$$
\lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } + \mathbf { W } _ { 2 } ^ { \top } \mathbf { A } \mathbf { W } _ { 1 } ^ { \top } = \mathbf { 0 } .\tag{19}
$$

Eliminating the common multiplier term $\mathbf { W } _ { 2 } ^ { \top } \mathbf { A } \mathbf { W } _ { 1 } ^ { \top }$ yields $\lambda _ { \mathrm { r e g } } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } - \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } \right) = \gamma \mathbf { I } _ { N _ { h } } ,$ which is equivalent to equation 14. □

Corollary A.6 (Non-collapse at constrained regularizer minima). Under the assumptions of Theorem $A . 5 ,$

$$
\boxed { \mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } \succeq \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } } .\tag{20}
$$

Consequently, the encoder hasfull row rank, with $\begin{array} { r } { \sigma _ { \operatorname* { m i n } } ( \mathbf { W } _ { 1 } ^ { \star } ) \geq \sqrt { \frac { \gamma } { \lambda _ { \mathrm { r e g } } } } . } \end{array}$

If, in addition, the centered empirical input covariance $\Sigma _ { x }$ defined in Lemma A.4 satisfies $\Sigma _ { x } \succeq$ $\kappa _ { x } \mathbf { I } _ { N _ { i } }$ for some $\kappa _ { x } ~ > ~ 0$ , then the centered empirical covariance of the hidden representations $\mathbf h _ { n } ^ { \star } = \mathbf W _ { 1 } ^ { \star } \mathbf x _ { n }$ satisfies

$$
\boxed { \mathbf { C } _ { h } ^ { \star } = \mathbf { W } _ { 1 } ^ { \star } \pmb { \Sigma } _ { x } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } \succeq \kappa _ { x } \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } } .\tag{21}
$$

Proof. Theorem A.5 gives

$$
( \mathbf { W } _ { 2 } ^ { \star } ) ^ { \top } \mathbf { W } _ { 2 } ^ { \star } - \mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } .
$$

Since $\gamma > 0$ and $\lambda _ { \mathrm { r e g } } > 0$ , this is a negative layer balance. Both bounds therefore follow directly from Lemma A.4 with $\lambda _ { \mathrm { b a l } } = - \gamma / \lambda _ { \mathrm { r e g } } < 0$ □

Thus, the representation covariance is positive definite, preventing both complete and dimensional collapse.

## A.1.5 BALANCE SELECTED BY THE ENCODER-SIDE REGULARIZER

For arbitrary ∆, the regularizer exponentially drives the network toward the selected balance.

Theorem A.7 (Global existence and exponential convergence of layer balance). Consider gradient flow with time scale $\tau > 0$ on the regularized mean-squared-error objective in equation 3.2, with $\lambda _ { \mathrm { r e g } } > 0$ and $\gamma > 0 .$ $I f \mathbf { W } _ { 1 } ( 0 ) \mathbf { W } _ { 1 } ( 0 ) ^ { \top } \succ 0$ , then the solution exists for all $t \geq 0$ and the encoder remainsfull row rank.

Define

$$
\begin{array} { r } { \pmb { \Delta } ( t ) = \mathbf { W } _ { 2 } ( t ) ^ { \top } \mathbf { W } _ { 2 } ( t ) - \mathbf { W } _ { 1 } ( t ) \mathbf { W } _ { 1 } ( t ) ^ { \top } , \qquad \pmb { \Delta } _ { \star } = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } . } \end{array}
$$

The imbalance satisfies

$$
\tau \dot { \Delta } ( t ) = - 2 \lambda _ { \mathrm { r e g } } \big ( \Delta ( t ) - \Delta _ { \star } \big ) ,
$$

and hence

$$
\boxed { \Delta ( t ) - \Delta _ { \star } = e ^ { - 2 \lambda _ { \mathrm { r e g } } t / \tau } \big ( \Delta ( 0 ) - \Delta _ { \star } \big ) } .\tag{22}
$$

Thus, for every admissible initialization, the layer imbalance converges exponentially to $\Delta$ <sub>⋆</sub> at rate $2 \lambda _ { \mathrm { r e g } } / \tau ,$ , selecting the asymptotic balance $\lambda _ { \mathrm { b a l } } ^ { \mathrm { e n c } } = - \gamma / \lambda _ { \mathrm { r e g } }$

Proof. We first establish global existence and preservation of the encoder full row rank. Along gradient flow,

$$
\frac { \mathrm { d } } { \mathrm { d } t } \mathcal { L } _ { \mathrm { L S A C } } = - \frac { 1 } { \tau } \left( \| \nabla _ { \mathbf { W } _ { 1 } } \mathcal { L } _ { \mathrm { L S A C } } \| _ { F } ^ { 2 } + \| \nabla _ { \mathbf { W } _ { 2 } } \mathcal { L } _ { \mathrm { L S A C } } \| _ { F } ^ { 2 } \right) \leq 0 .
$$

Writing $\sigma _ { i }$ for the encoder singular values, the regularizer is

$$
\mathcal { R } _ { \mathrm { L S A C } } = \frac { \lambda _ { \mathrm { r e g } } } { 2 } \| \mathbf { W } _ { 2 } \| _ { F } ^ { 2 } + \sum _ { i = 1 } ^ { N _ { h } } \left( \frac { \lambda _ { \mathrm { r e g } } } { 2 } \sigma _ { i } ^ { 2 } - \gamma \log { \sigma _ { i } } \right) .
$$

Each scalar summand is bounded below and diverges at both zero and infinity. Since the meansquared-error task loss is nonnegative, the initial objective sublevel set bounds the weights and keeps every encoder singular value away from zero. The trajectory thus remains in a compact subset of the full-row-rank domain, where the gradient field is smooth. Standard ODE continuation gives global existence and $\mathbf { W } _ { 1 } ( t ) \mathbf { W } _ { 1 } ( t ) ^ { \top } \succsim \overline { { 0 } }$ for all $t \geq 0$

Now define

$$
\mathbf { G } = \nabla _ { \mathbf { M } } \ell ( \mathbf { M } ) { \big | } _ { \mathbf { M } = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } } , \qquad \mathbf { H } = \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } .
$$

Since $\mathbf { H } \succ 0 ,$ , it is symmetric and invertible. The gradient of the encoder-side log-determinant term is

$$
\nabla _ { \mathbf { W } _ { 1 } } \left[ - \frac { \gamma } { 2 } \log \operatorname* { d e t } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) \right] = - \gamma \mathbf { H } ^ { - 1 } \mathbf { W } _ { 1 } .\tag{23}
$$

The gradient-flow equations are therefore

$$
\begin{array} { r } { \tau \dot { \mathbf { W } } _ { 1 } = - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } - \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } + \gamma \mathbf { H } ^ { - 1 } \mathbf { W } _ { 1 } , } \end{array}\tag{24}
$$

$$
\begin{array} { r } { \tau \dot { \mathbf { W } } _ { 2 } = - \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } . } \end{array}\tag{25}
$$

We first compute the dynamics of the decoder Gram matrix. By the product rule,

$$
\tau \frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } \right) = \left( \tau \dot { \mathbf { W } } _ { 2 } \right) ^ { \top } \mathbf { W } _ { 2 } + \mathbf { W } _ { 2 } ^ { \top } \left( \tau \dot { \mathbf { W } } _ { 2 } \right) .
$$

Substituting equation 25 gives

$$
\begin{array} { r l } & { \tau \frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } \right) = \left( - \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } \right) ^ { \top } \mathbf { W } _ { 2 } + \mathbf { W } _ { 2 } ^ { \top } \left( - \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } \right) } \\ & { \qquad = - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - 2 \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } . } \end{array}\tag{26}
$$

Similarly, the encoder Gram matrix satisfies

$$
\tau \frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) = \left( \tau \dot { \mathbf { W } } _ { 1 } \right) \mathbf { W } _ { 1 } ^ { \top } + \mathbf { W } _ { 1 } \left( \tau \dot { \mathbf { W } } _ { 1 } \right) ^ { \top } .
$$

Substituting equation 24, we obtain

$$
\begin{array} { r l r } {  { \tau \frac { \mathrm { d } } { \mathrm { d } t } ( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } ) = ( - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } - \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } + \gamma \mathbf { H } ^ { - 1 } \mathbf { W } _ { 1 } ) \mathbf { W } _ { 1 } ^ { \top } } } \\ & { } & { ~ + \mathbf { W } _ { 1 } ( - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } - \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } + \gamma \mathbf { H } ^ { - 1 } \mathbf { W } _ { 1 } ) ^ { \top } } \\ & { } & { = - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } - 2 \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } } \\ & { } & { ~ + \gamma \mathbf { H } ^ { - 1 } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } + \gamma \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \mathbf { H } ^ { - 1 } . } \end{array}\tag{27}
$$

Because $\mathbf { H } = \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top }$ , we have

$$
\mathbf { H } ^ { - 1 } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } = \mathbf { I } _ { N _ { h } } , \qquad \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \mathbf { H } ^ { - 1 } = \mathbf { I } _ { N _ { h } } .
$$

Therefore,

$$
\tau \frac { \mathrm { d } } { \mathrm { d } t } \left( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) = - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } - 2 \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } + 2 \gamma \mathbf { I } _ { N _ { h } } .\tag{28}
$$

Recall that

$$
\begin{array} { r } { \pmb { \Delta } = \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } . } \end{array}
$$

Subtracting equation 28 from equation 26 yields

$$
\begin{array} { r l } & { \tau \dot { \mathbf { A } } = \left( - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - 2 \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } \right) } \\ & { \quad \quad - \left( - \mathbf { W } _ { 2 } ^ { \top } \mathbf { G } \mathbf { W } _ { 1 } ^ { \top } - \mathbf { W } _ { 1 } \mathbf { G } ^ { \top } \mathbf { W } _ { 2 } - 2 \lambda _ { \mathrm { r e g } } \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } + 2 \gamma \mathbf { I } _ { N _ { h } } \right) } \\ & { = - 2 \lambda _ { \mathrm { r e g } } \left( \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } - \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \right) - 2 \gamma \mathbf { I } _ { N _ { h } } . } \end{array}\tag{29}
$$

The task-loss terms cancel exactly, leaving

$$
\boxed { \tau \dot { \Delta } = - 2 \lambda _ { \mathrm { r e g } } \Delta - 2 \gamma { \bf I } _ { N _ { h } } } .
$$

It remains to solve this linear matrix differential equation. Dividing by $\tau$ gives

$$
\frac { \partial } { \partial \ L } + \frac { 2 \lambda _ { \mathrm { r e g } } } { \tau } \Delta \ L = - \frac { 2 \gamma } { \tau } \mathbf { I } _ { N _ { h } } .
$$

Multiplying both sides by the integrating factor $e ^ { 2 \lambda _ { \mathrm { r e g } } t / \tau }$ yields

$$
\frac { \mathrm { d } } { \mathrm { d } t } \left[ e ^ { 2 \lambda _ { \mathrm { r e g } } t / \tau } \Delta ( t ) \right] = - \frac { 2 \gamma } { \tau } e ^ { 2 \lambda _ { \mathrm { r e g } } t / \tau } \mathbf { I } _ { N _ { h } } .
$$

Integrating from 0 to t, we obtain

$$
\begin{array} { r l r } {  { e ^ { 2 \lambda _ { \mathrm { r e g } } t / \tau } \pmb { \Delta } ( t ) - \pmb { \Delta } ( 0 ) = - \frac { 2 \gamma } { \tau } \int _ { 0 } ^ { t } e ^ { 2 \lambda _ { \mathrm { r e g } } s / \tau } \mathrm { d } s \mathbf { I } _ { N _ { h } } } } \\ & { } & { = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } ( e ^ { 2 \lambda _ { \mathrm { r e g } } t / \tau } - 1 ) \mathbf { I } _ { N _ { h } } . } \end{array}
$$

Multiplying by $e ^ { - 2 \lambda _ { \mathrm { r e g } } t / \tau }$ gives

$$
\Delta ( t ) = e ^ { - 2 \lambda _ { \mathrm { r e g } } t / \tau } \Delta ( 0 ) - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \left( 1 - e ^ { - 2 \lambda _ { \mathrm { r e g } } t / \tau } \right) \mathbf { I } _ { N _ { h } } .
$$

Equivalently, defining ${ \pmb { \Delta } } _ { \star } = - ( \gamma / \lambda _ { \mathrm { r e g } } ) { \bf I } _ { N _ { h } }$

$$
\boxed { \Delta ( t ) - \Delta _ { \star } = e ^ { - 2 \lambda _ { \mathrm { r e g } } t / \tau } \big ( \Delta ( 0 ) - \Delta _ { \star } \big ) } .
$$

Since $\lambda _ { \mathrm { r e g } } > 0$ and $\tau > 0$ , the imbalance converges exponentially to $\Delta$ <sub>⋆</sub> at rate $2 \lambda _ { \mathrm { r e g } } / \tau$ □

Corollary A.8 (Persistence of the selected balance). If the network is initialized such that $\Delta ( 0 ) =$ $- \frac { \gamma } { \lambda _ { \mathrm { r e g } } } { \bf I } _ { N _ { h } }$ , then $\begin{array} { r } { \pmb { \Delta } ( t ) = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } } \end{array}$ for all $t \geq 0 .$ . Thus, the balance condition selected by the regularizer is invariant under the regularized gradient-flow dynamics.

Proof. Substituting the assumed initial condition into equation 22 gives the result immediately.

## A.1.6 SINGULAR-MODE CHARACTERIZATION

The previous results establish the selected balance and the anti-collapse property without requiring an explicit factorization. For completeness, we now give the corresponding allocation of each endto-end singular mode across the two layers, also derived in Domine et al. (2025). We later show that´ the regularization leads to the balanced factorization (see Appendix A.1.7).

Theorem A.9 (Singular-mode allocation). Assume $N _ { h } \ \leq$ min $\{ N _ { i } , N _ { o } \}$ , and let $\mathbf { M } = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top }$ where ${ \bf S } = \mathrm { d i a g } ( s _ { 1 } , \dots , s _ { N _ { h } } )$ ,with $s _ { i } \geq 0 ,$ , defines an $N _ { h }$ -mode singular value decomposition, padded with zero singular values when ran $\mathbf { \bar { \rho } } _ { \mathrm { { i } } } ( \mathbf { M } ) < N _ { h }$ . Define $\begin{array} { r } { \lambda _ { b a l } \stackrel { \mathrm { ~ - ~ } \gamma } { = } - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } } \end{array}$ . Up to an orthogonal transformation $\mathbf { R } \in \mathbb { R } ^ { N _ { h } \times N _ { h } }$ of the hidden layer and rotations within degenerate singular subspaces, a minimizer of Theorem $A . 5$ can be written as $\mathbf { W } _ { 1 } ^ { \star } = \mathbf { R } \mathbf { A } \mathbf { V } ^ { \top } , \mathbf { W } _ { 2 } ^ { \star } = \mathbf { U } \mathbf { B } \mathbf { R } ^ { \top }$ , where $\mathbf { A } = \mathrm { d i a g } ( a _ { 1 } , \ldots , a _ { N _ { h } } ) , \mathbf { B } = \mathrm { d i a g } ( b _ { 1 } , \ldots , b _ { N _ { h } } )$ , and

$$
a _ { i } ^ { 2 } = \frac { - \lambda _ { b a l } + \sqrt { \lambda _ { b a l } ^ { 2 } + 4 s _ { i } ^ { 2 } } } { 2 } , \qquad b _ { i } ^ { 2 } = \frac { \lambda _ { b a l } + \sqrt { \lambda _ { b a l } ^ { 2 } + 4 s _ { i } ^ { 2 } } } { 2 } .\tag{30}
$$

In particular, $a _ { i } b _ { i } = s _ { i }$ , and $b _ { i } ^ { 2 } - a _ { i } ^ { 2 } = \lambda _ { b a l }$

Proof. According to Eq. 14 $( \mathbf { W } _ { 2 } ^ { \star } ) ^ { \top } \mathbf { W } _ { 2 } ^ { \star } - \mathbf { W } _ { 1 } ^ { \star } ( \mathbf { W } _ { 1 } ^ { \star } ) ^ { \top } = \lambda _ { b a l } \mathbf { I } _ { N _ { h } }$ . The two Gram matrices therefore commute and share an orthogonal eigenbasis R. Consequently, the factors admit the aligned representation with $\mathbf { B } ^ { 2 } - \mathbf { A } ^ { 2 } = \bar { \lambda _ { b a l } } \mathbf { I } _ { N _ { h } }$

The product constraint gives

$$
\mathbf { W } _ { 2 } ^ { \star } \mathbf { W } _ { 1 } ^ { \star } = \mathbf { U } \mathbf { B } \mathbf { A } \mathbf { V } ^ { \top } = \mathbf { U } \mathbf { S } \mathbf { V } ^ { \top } ,
$$

and hence BA = S. For each singular direction i, we therefore have

$$
a _ { i } b _ { i } = s _ { i } , \qquad b _ { i } ^ { 2 } - a _ { i } ^ { 2 } = \lambda _ { b a l } .
$$

Setting $x _ { i } = b _ { i } ^ { 2 }$ gives $x _ { i } ( x _ { i } - \lambda _ { b a l } ) = s _ { i } ^ { 2 }$ . The solution is

$$
b _ { i } ^ { 2 } = \frac { \lambda _ { b a l } + \sqrt { \lambda _ { b a l } ^ { 2 } + 4 s _ { i } ^ { 2 } } } { 2 } ,
$$

and subtracting $\lambda _ { b a l }$ gives

$$
a _ { i } ^ { 2 } = \frac { - \lambda _ { b a l } + \sqrt { \lambda _ { b a l } ^ { 2 } + 4 s _ { i } ^ { 2 } } } { 2 } .
$$

## A.1.7 RICH AND LAZY REGIME

The ratio $- \gamma / \lambda _ { \mathrm { r e g } }$ determines the selected layer balance and thereby influences the learning regime. For a fixed end-to-end mapping, it governs how each singular mode is distributed between the encoder and decoder. Zero balance $( \lambda _ { \mathrm { b a l } } = 0 )$ corresponds to equal layer singular values, whereas nonzero balance introduces an asymmetry. The effect of this balance on learning depends on the architecture, initialization scale, and sign of $\lambda _ { \mathrm { b a l } }$ (Domine et al., 2025). In the linear setting con-´ sidered here, sufficiently strong negative imbalance can favor lazy dynamics while preserving the anti-collapse guarantee. By contrast, zero-balanced initialization can support rich dynamics at sufficiently small initialization scales. The relationship between imbalance and learning regime is more complex in nonlinear networks, where full matrix imbalance is not conserved and does not necessarily induce lazy learning. Appropriate asymmetric parameter scalings can instead accelerate the onset of rich feature-learning dynamics while preserving the non-collapse property (Anguita et al., 2026; Kumar et al., 2024; Kunin et al., 2024). We discuss this distinction further in Appendix A.2.3.

## A.1.8 EXPERIMENTAL VERIFICATION

We empirically verify Theorem 3.1 (and its dynamical counterpart, Theorem A.7) in a minimal twolayer linear network with no architectural bottleneck, $N _ { i } = N _ { h } \dot { \overline { { { = } } } } N _ { o } = 8$ , so that $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } \in \mathbb { R } ^ { 8 \times 8 }$ are both square and (generically) invertible.

![](images/c7c7cd556890a67e2ee8befdeefd40cbd01f4511cf46bc318161fd0340d8e56d.jpg)

![](images/f1324c8bcb411b10b88d60333c528490dd9455afe76c5c6d53ee7855b539ec1d.jpg)

![](images/09d0523ed440db75637009efe0646bb2f51cc97ad7a1e23bb13b6476629442d5.jpg)  
Figure 3: Block Gram matrix $\mathbf { Q } = \left( \begin{array} { c c } { \mathbf { W } _ { 1 } ^ { \top } \mathbf { W } _ { 1 } } & { \mathbf { \Phi } \mathbf { \Phi } \mathbf { { M } } ^ { \top } } \\ { \mathbf { \Phi } \mathbf { M } } & { \mathbf { W } _ { 2 } \mathbf { W } _ { 2 } ^ { \top } } \end{array} \right)$ $M = \mathbf { W } _ { 2 } \mathbf { W } _ { 1 }$ , for a two-layer linear network. (A) Training with symmetric $L _ { 2 }$ weight decay $( \dot { \lambda } _ { \mathrm { r e g } } = 0 . 1 )$ and an encoder-side negative log-determinant penalty $( \gamma = 0 . 2 )$ . The regularizer drives the layer imbalance $\pmb { \Delta } = \mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } -$ $\mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top }$ towards ${ \bf \Delta } \Delta _ { \star } = - ( \gamma / \lambda _ { \mathrm { r e g } } ) { \bf I } _ { N _ { h } } = - 2 { \bf I } _ { N _ { h } }$ , corresponding to $\lambda _ { \mathrm { b a l } } ~ = ~ - 2$ . (B) Closedform prediction from Appendix A.1.6. (C) Unregularized control network initialized at $\lambda _ { \mathrm { b a l } } = - 2$ This balance is conserved under gradient flow, providing a comparison with the balance selected dynamically in (A).

Task. We use an orthogonal input design ${ \bf X } = \sqrt { N _ { i } } { \bf I } _ { N _ { i } }$ and a fixed $\{ 0 , \pm 1 / \sqrt { N _ { i } } \}$ -valued target matrix Y with hierarchical block structure. The resulting least-squares mapping has three distinct singular values (multiplicities 2, 2, 4, reflecting the hierarchy), letting us check the predicted balance across multiple directions at once.

Training protocol. In Fig. 3 (A), both layers are initialized with unbalanced i.i.d. Gaussian weights, $\mathbf { W } _ { 1 } ( 0 ) , \mathbf { \bar { W } } _ { 2 } ( 0 ) \sim \mathcal { N } ( \bar { 0 , } \sigma ^ { 2 } ) , \sigma = 0 . 5 .$ . The network is trained by full-batch gradient descent on $\mathcal { L } _ { \mathrm { r e g } } = \mathcal { L } _ { \mathrm { t a s k } } + \mathcal { R } _ { \mathrm { L S A C } } ( \eta = 1 0 ^ { - 3 } , T = 3 \times 1 0 ^ { 4 }$ steps), with the log-determinant term on the encoder Gram matrix, −γ log det $( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { T } )$ , and weight decay $\lambda _ { r e g }$ on both layers $( ( \lambda _ { r e g } , \gamma ) = ( 0 . 1 , 0 . 2 ) )$ .

Verification protocol. $\begin{array} { r l r } { \mathrm { A t } \textit { \ t } } & { { } = } & { T _ { \cdot } } \end{array}$ , we compare three block Gram matrices $\textrm { \textbf { Q } } =$ $\left( \begin{array} { c c } { { \dot { \bf W } _ { 1 } ^ { \top } { \bf W } _ { 1 } } } & { { \bar { \bf M } ^ { \top } } } \\ { { \bar { \cal M } } } & { { \bf W } _ { 2 } { \bf W } _ { 2 } ^ { \top } } \end{array} \right)$ , where $M \ = \ \mathbf { W } _ { 2 } \mathbf { W } _ { 1 } , \mathbf { \Gamma } ( \mathbf { A } ) \ \mathbf { Q } _ { \mathrm { e m p } }$ from the regularized network; (B) $\mathbf { Q } _ { \mathrm { t h e o r y } } ,$ , obtained by Theorem A.9 Appendix A.1.6, with $\lambda _ { \mathrm { b a l } } ~ = ~ - \gamma / \lambda _ { r e q } ~ = ~ - 2 ;$ and (C) $\mathbf { Q } _ { \mathrm { b a l } }$ , from a network initialized exactly at this $\lambda _ { \mathrm { b a l } }$ -balanced solution and trained unregularized $( \lambda _ { r e g } = \gamma = 0 )$ , whose balance is conserved by construction under unregularized gradient flow.

Fig. 3 shows near-perfect agreement across all three panels. The empirical imbalance eigenvalues concentrate tightly around the predicted, negative fixed point, $\lambda _ { i } ( \bar { \Delta ( T ) } ) \in [ - 2 . 0 0 5 , - \bar { 1 . 9 8 4 } ]$ ≈ $\lambda _ { \mathrm { b a l } }$ , and the small relative error,

$$
e _ { r e l } = \frac { \lVert \mathbf { Q } _ { \mathrm { e m p } } - \mathbf { Q } _ { \mathrm { t h e o r y } } \rVert _ { F } } { \lVert \mathbf { Q } _ { \mathrm { t h e o r y } } \rVert _ { F } } = 0 . 0 0 9 5 ,
$$

confirms that the encoder-side log-determinant regularizer drives the network towards the predicted, collapse-preventing balanced fixed point, without impeding task fit (final $\mathrm { M S E } \approx 5 . 4 7 \times \mathrm { \bar { 1 0 ^ { - 4 } } } ,$ .

## A.2 FROM LINEAR TO NONLINEAR COVARIANCE REGULARIZATION

## A.2.1 FROM LINEAR BALANCE TO NONLINEAR COVARIANCE REGULARIZATION

For a linear encoder with centered, whitened inputs and $\varepsilon ~ = ~ 0$ , the covariance penalty reduces exactly to the encoder portion of the linear regularizer, with $\lambda _ { \mathrm { c o v } } = \lambda _ { \mathrm { r e g } }$

Lemma A.10 (Linear encoder mean and covariance). Let $\mathbf { x } \in \mathbb { R } ^ { N _ { i } }$ have finite second moments, mean $\pmb { \mu } _ { x } = \mathbb { E } [ \mathbf { x } ]$ ], and covariance $\Sigma _ { x } = \mathbb { E } [ ( { \bf x } - { \pmb \mu } _ { x } ) ( { \bf x } - { \pmb \mu } _ { x } ) ^ { \top } ]$ . For the bias-free linear representation $\mathbf { h } = \mathbf { W } _ { 1 } \mathbf { x }$ , its mean and covariance satisfy $\pmb { \mu } = \mathbf { W } _ { 1 } \pmb { \mu } _ { x } , \mathbf { \dot { C } } _ { h } = \mathbf { W } _ { 1 } \pmb { \Sigma } _ { x } \mathbf { W } _ { 1 } ^ { \top }$

In particular, for centered, whitened inputs, $\pmb { \mu _ { x } } = \mathbf { 0 } a n d \pmb { \Sigma _ { x } } = \mathbf { I } _ { N _ { i } } , s o \pmb { \mu } = \mathbf { 0 } , \mathbf { C } _ { h } = \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top }$ , and $\mathrm { t r } ( \mathbf { C } _ { h } ) = \| \mathbf { W } _ { 1 } \| _ { F } ^ { 2 }$

Proof. By linearity of expectation, $\pmb { \mu } = \mathbb { E } [ \mathbf { W } _ { 1 } \mathbf { x } ] = \mathbf { W } _ { 1 } \pmb { \mu } _ { x }$ . Consequently, $\mathbf { h } - \pmb { \mu } = \mathbf { W } _ { 1 } ( \mathbf { x } - \pmb { \mu } _ { x } )$ giving $\mathbf { C } _ { h } = \mathbb { E } [ ( \mathbf { h } - \pmb { \mu } ) ( \mathbf { h } - \pmb { \mu } ) ^ { \top } ] = \mathbf { W } _ { 1 } \pmb { \Sigma } _ { x } \dot { \mathbf { W } } _ { 1 } ^ { \top }$ . For centered, whitened inputs, these identities reduce to $\pmb { \mu } = 0$ and $\dot { \mathbf { C } } _ { h } = \mathbf { W } _ { 1 } \dot { \mathbf { W } } _ { 1 } ^ { \top }$ . Taking the trace yields t $\mathrm { r } ( \mathbf { C } _ { h } ) = \| \mathbf { W } _ { 1 } \| _ { F } ^ { 2 }$ □

Recovery of the linear encoder regularizer. Recall that the nonlinear regularizer is

$$
\mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) = \frac { \lambda _ { \mathrm { m e a n } } } { 2 } \| \pmb { \mu } \| _ { 2 } ^ { 2 } + \frac { \lambda _ { \mathrm { c o v } } } { 2 } \operatorname { t r } ( \pmb { \mathbf { C } } _ { h } ) - \frac { \gamma } { 2 } \log \operatorname* { d e t } ( \mathbf { C } _ { h } + \varepsilon \mathbf { I } _ { N _ { h } } ) .
$$

Assume $\mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } \succ 0$ , centered and whitened inputs, $\varepsilon = 0 .$ , and $\lambda _ { \mathrm { c o v } } = \lambda _ { \mathrm { r e g } }$ . Lemma A.10 then gives

$$
\mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) = \frac { \lambda _ { \mathrm { r e g } } } { 2 } \| \mathbf { W } _ { 1 } \| _ { F } ^ { 2 } - \frac { \gamma } { 2 } \log \operatorname* { d e t } ( \mathbf { W } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } ) .
$$

The mean penalty vanishes because ${ \pmb \mu } = { \bf 0 }$ , regardless of the positive value of $\lambda _ { \mathrm { m e a n } }$ . Thus, the nonlinear regularizer reduces exactly to the encoder portion of the linear regularizer:

$$
\mathcal { R } _ { \mathrm { L S A C } } ( \mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } ) = \mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) + \frac { \lambda _ { \mathrm { r e g } } } { 2 } \| \mathbf { W } _ { 2 } \| _ { F } ^ { 2 } .
$$

Role of the mean penalty. For nonlinear encoders, centered inputs need not produce centered representations. Moreover, covariance is invariant under a common translation of all representations, so neither the trace nor the log-determinant term controls their mean. The additional $\mathbf { \bar { \boldsymbol { \mu } } } \| \boldsymbol { \mu } \| _ { 2 } ^ { 2 }$ penalty controls this translation by favoring zero-mean representations. It therefore extends the regularizer without changing its reduction to the bias-free linear setting with centered, whitened inputs. The same identities hold for empirical means and covariances.

## A.2.2 LOG-DETERMINANT COVARIANCE REGULARIZER WITH THE MEAN

Let $\mathbf { h } = f _ { \theta } ( \mathbf { x } ) \in \mathbb { R } ^ { N _ { h } }$ denote the representation produced by a possibly nonlinear encoder. For a minibatch of size $B ,$ collect the representations row-wise in $\mathbf { H } \in \mathbb { R } ^ { \tilde { B } \times N _ { h } }$ . Define the empirical mean, centering matrix, centered representations, and empirical covariance as $\begin{array} { r } { { \pmb \mu } = \frac { 1 } { B } { \bf H } ^ { \top } { \bf 1 } } \end{array}$ $\begin{array} { r } { \mathbf { P } _ { B } \ = \ \mathbf { I } _ { B } - \frac { 1 } { R } \mathbf { 1 } \mathbf { 1 } ^ { \top } , \ \widetilde { \mathbf { H } } \ = \ \mathbf { P } _ { B } \mathbf { H } , \ \mathbf { C } _ { h } \ = \ \frac { 1 } { R } \widetilde { \mathbf { H } } ^ { \top } \widetilde { \mathbf { H } } \ \succeq \ 0 } \end{array}$ . The linear analysis never needs to control the representation’s mean: for a bias-free linear encoder $\mathbf { h } = \mathbf { W } _ { 1 } \mathbf { x }$ with centered input, $\mathbb { E } [ \mathbf { h } ] = \mathbf { W } _ { 1 } \bar { \mathbb { E } } [ \mathbf { x } ] = \mathbf { 0 }$ identically, so the mean carries no independent degree of freedom for $\mathcal { R } _ { \mathrm { S A C } }$ to regularize. This guarantee does not survive the transfer to a general nonlinear (or biased) encoder: a bias term, or the nonlinearity itself (e.g. a ReLU clipping one side of an otherwise symmetric preactivation), can give h a nonzero mean that moves independently of $\mathbf { C } _ { h } - \mathrm { i n v i s i b l e }$ to both $\mathrm { t r } ( { \bf C } _ { h } )$ and log det $\left( \mathbf { C } _ { h } \right)$ , since $\mathbf { C } _ { h }$ is computed from the centered representation by construction. We therefore regularize $\pmb { \mu }$ directly alongside $\mathbf { C } _ { h }$ . Motivated by the linear encoder regularizer, we give the following definition.

Definition A.11 (Nonlinear SACReg). Let $\lambda _ { \mathrm { m e a n } } > 0 , \lambda _ { \mathrm { c o v } } > 0 , \gamma > 0 ,$ , and $\varepsilon \geq 0 .$ . Assume that $\mathbf { C } _ { h } + \varepsilon \mathbf { I } _ { N _ { h } } \succ 0$ . We define

$$
\mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) = \frac { \lambda _ { \mathrm { m e a n } } } { 2 } \| \pmb { \mu } \| _ { 2 } ^ { 2 } + \frac { \lambda _ { \mathrm { c o v } } } { 2 } \operatorname { t r } ( \mathbf { C } _ { h } ) - \frac { \gamma } { 2 } \log \operatorname* { d e t } \left( \mathbf { C } _ { h } + \varepsilon \mathbf { I } _ { N _ { h } } \right) .\tag{31}
$$

The mean term controls the representation’s location, the trace term controls its overall scale, and the negative log-determinant penalizes small covariance eigenvalues. The regularizer constrains only the representation mean and covariance; it does not directly constrain higher-order moments or enforce Gaussianity.

To understand the inductive bias of the regularizer, we consider its minimizer in isolation. This does not imply that the mean and covariance of the complete learning objective must attain this value, since the task loss may introduce competing constraints. First, we write the regularizer in spectral form.

Proposition A.12 (Spectral form). Let $\lambda _ { 1 } ( \mathbf { C } _ { h } ) , \ldots , \lambda _ { N _ { h } } ( \mathbf { C } _ { h } )$ be the eigenvalues of $\mathbf { C } _ { h }$ . Then

$$
\mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) = \frac { \lambda _ { \mathrm { m e a n } } } { 2 } \| \pmb { \mu } \| ^ { 2 } + \frac { 1 } { 2 } \sum _ { i = 1 } ^ { N _ { h } } \left[ \lambda _ { \mathrm { c o v } } \lambda _ { i } ( \mathbf { C } _ { h } ) - \gamma \log \left( \lambda _ { i } ( \mathbf { C } _ { h } ) + \varepsilon \right) \right] .\tag{32}
$$

Proof. Because $\mathbf { C } _ { h }$ is symmetric and positive semidefinite, it admits an eigendecomposition $\mathbf { C } _ { h } = \mathbf { Q } \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { N _ { h } } ) \mathbf { Q } ^ { \top }$ . Consequently, $\begin{array} { r } { \mathrm { t r } ( { \bf C } _ { h } ) = \sum _ { i = 1 } ^ { N _ { h } } \lambda _ { i } } \end{array}$ and log det $\left( \mathbf { C } _ { h } + \varepsilon \mathbf { I } _ { N _ { h } } \right) =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N _ { h } } \log ( \lambda _ { i } + \varepsilon ) } \end{array}$ . The mean term does not depend on $\mathbf { C } _ { h }$ . Substitution into equation 31 proves the result. □

We can now write the optimal mean and covariance.

Theorem A.13 (Optimal mean and covariance). The unique minimizer of $\mathcal { R } _ { \mathrm { S A C } }$ over $\mu \in$ $\mathbb { R } ^ { N _ { h } } , \mathbf { C } _ { h } \succeq 0$ such that $\mathbf { C } _ { h } + \varepsilon \mathbf { I } _ { N _ { h } } \succ 0$ is

$$
\boxed { \pmb { \mu } ^ { \star } = \mathbf { 0 } , \qquad \mathbf { C } _ { h } ^ { \star } = \left( \frac { \gamma } { \lambda _ { \mathrm { c o v } } } - \varepsilon \right) _ { + } \mathbf { I } _ { N _ { h } } } ,\tag{33}
$$

where $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$ . Therefore, the covariance-level optimum is full-rank if and only $i f \gamma >$ $\lambda _ { \mathrm { c o v } } \varepsilon ,$ , independently of $\scriptstyle \mu ^ { \star }$

Proof. By Proposition A.12, $\mathcal { R } _ { \mathrm { S A C } }$ is a sum of a term depending only on µ and a term depending only on $\mathbf { C } _ { h }$ , so the two may be minimized separately. The mean term $\frac { \lambda _ { \mathrm { m e a n } } } { 2 } \lVert \boldsymbol { \mu } \rVert ^ { 2 }$ is uniquely minimized at $\mathbf { \nabla } \mu ^ { \star } = \textbf { 0 }$ . For the covariance term, each eigenvalue independently minimizes the strictly convex function

$$
\phi ( s ) = \frac { \lambda _ { \mathrm { c o v } } } { 2 } s - \frac { \gamma } { 2 } \log ( s + \varepsilon ) , \qquad s \geq 0 \quad , s + \varepsilon \geq 0
$$

Its derivatives are $\begin{array} { r } { \phi ^ { \prime } ( s ) = \frac { \lambda _ { \mathrm { c o v } } } { 2 } - \frac { \gamma } { 2 ( s + \varepsilon ) } } \end{array}$ and $\phi ^ { \prime \prime } ( s ) = \textstyle { \frac { \gamma } { 2 ( s + \varepsilon ) ^ { 2 } } } > 0$ . The unconstrained stationary point satisfies $\begin{array} { r } { \phi ^ { \prime } ( s ) = 0 \Leftrightarrow s = \frac { \gamma } { \lambda _ { \mathrm { c o v } } } - \varepsilon . \mathrm { I f } \gamma / \lambda _ { \mathrm { c o v } } - \varepsilon > 0 , } \end{array}$ , this stationary point is the unique minimizer. Otherwise, $\varepsilon > 0$ , so $s = 0$ is admissible and is the unique minimizer. □

Corollary A.14 (Isotropic non-collapsed target, centered). In the idealized case $\varepsilon = 0$

$$
\boxed { \pmb { \mu } ^ { \star } = \mathbf { 0 } , \qquad \mathbf { C } _ { h } ^ { \star } = \frac { \gamma } { \lambda _ { \mathrm { c o v } } } \mathbf { I } _ { N _ { h } } } .\tag{34}
$$

Thus, the regularizer simultaneously:

1. centers the representation at the origin;

2. prevents the covariance eigenvalues from vanishing;

3. equalizes the covariance spectrum;

4. controls the overall representation scale; and

5. selects a condition number equal to one.

Next we derive the non-collapse property of the regularizer.

Theorem A.15 (Mean boundedness and strict covariance non-collapse). Suppose that $\varepsilon = 0$ . Then

$$
\lambda _ { \mathrm { m i n } } ( \mathbf { C } _ { h } ) \to 0 \quad \Longrightarrow \quad \mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) \to + \infty ,\tag{35}
$$

$$
\lambda _ { \operatorname* { m a x } } ( \mathbf { C } _ { h } ) \to + \infty \quad \Longrightarrow \quad \mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) \to + \infty ,\tag{36}
$$

$$
\| \pmb { \mu } \|  \infty \quad \Longrightarrow \quad \mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } )  + \infty .\tag{37}
$$

Consequently, any fixed finite upper bound on the regularizer bounds the mean and keeps all covariance eigenvalues bounded above and away from zero. Specifically, for every finite K, there exist constants $0 < m _ { K } \le M _ { K } < \infty$ and $0 \leq \dot { M } _ { K } ^ { \prime } < \infty$ such that

$$
R _ { \mathrm { S A C } } ( \mu , \mathbf { C } _ { h } ) \leq K \quad \Longrightarrow \quad m _ { K } \mathbf { I } _ { N _ { h } } \preceq \mathbf { C } _ { h } \preceq M _ { K } \mathbf { I } _ { N _ { h } } , \qquad \| \mu \| _ { 2 } \leq M _ { K } ^ { \prime } .\tag{38}
$$

Proof. Suppose that $\varepsilon = 0$ and $\mathbf { C } _ { h } \succ 0$ . Let $\lambda _ { 1 } ( \mathbf { C } _ { h } ) , \ldots , \lambda _ { N _ { h } } ( \mathbf { C } _ { h } ) > 0$ denote the eigenvalues of $\mathbf { C } _ { h }$ . Since $\begin{array} { r } { \mathrm { t r } ( \mathbf { C } _ { h } ) = \sum _ { i = 1 } ^ { N _ { h } } \lambda _ { i } ( \mathbf { C } _ { h } ) } \end{array}$ and log det $\begin{array} { r } { \mathbf { \mu } ( \mathbf { C } _ { h } ) = \sum _ { i = 1 } ^ { N _ { h } } \log \lambda _ { i } ( \mathbf { C } _ { h } ) } \end{array}$ , write

$$
\mathcal { R } _ { \mathrm { S A C } } ( \mu , \mathbf { C } _ { h } ) = \frac { \lambda _ { \mathrm { m e a n } } } { 2 } \| \mu \| ^ { 2 } + \sum _ { i = 1 } ^ { N _ { h } } \phi \left( \lambda _ { i } ( \mathbf { C } _ { h } ) \right) , \qquad \phi ( s ) = \frac { \lambda _ { \mathrm { c o v } } } { 2 } s - \frac { \gamma } { 2 } \log s , \quad s > 0 .
$$

We first show that $\phi$ is bounded below. Its derivatives are $\begin{array} { r } { \phi ^ { \prime } ( s ) = \frac { \lambda _ { \mathrm { c o v } } } { 2 } - \frac { \gamma } { 2 s } } \end{array}$ and $\phi ^ { \prime \prime } ( s ) = \textstyle { \frac { \gamma } { 2 s ^ { 2 } } } > 0$ so ϕ is strictly convex, with unique stationary point $s = \gamma / \lambda _ { \mathrm { c o v } }$ and global minimum

$$
b : = \operatorname* { m i n } _ { s > 0 } \phi ( s ) = \phi \biggl ( \frac { \gamma } { \lambda _ { \mathrm { c o v } } } \biggr ) = \frac { \gamma } { 2 } - \frac { \gamma } { 2 } \log \biggl ( \frac { \gamma } { \lambda _ { \mathrm { c o v } } } \biggr ) .
$$

In particular, $\phi ( s ) \geq b$ for every $s > 0$ , so $\begin{array} { r } { \sum _ { i = 1 } ^ { N _ { h } } \phi ( \lambda _ { i } ( { \bf C } _ { h } ) ) \ge N _ { h } b } \end{array}$ for every $\mathbf { C } _ { h } \succ 0$

Covariance collapse. Without loss of generality order the eigenvalues so that $\lambda _ { 1 } ( \mathbf { C } _ { h } ) = \lambda _ { \operatorname* { m i n } } ( \mathbf { C } _ { h } )$ Then

$$
\mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) \geq \phi \left( \lambda _ { \operatorname* { m i n } } ( \mathbf { C } _ { h } ) \right) + ( N _ { h } - 1 ) b .
$$

As $\begin{array} { r } { s \downarrow 0 , \frac { \lambda _ { \mathrm { c o v } } } { 2 } s  0 } \end{array}$ and $\begin{array} { r } { - \frac { \gamma } { 2 } \log s  + \infty , } \end{array}$ so $\begin{array} { r } { \operatorname* { l i m } _ { s \downarrow 0 } \phi ( s ) = + \infty } \end{array}$ . Since $( N _ { h } - 1 ) b$ is a fixed finite constant, $\bar { \lambda _ { \operatorname* { m i n } } } ( \mathbf { C } _ { h } ) \to 0 \bar { \Rightarrow } \mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) \to + \infty$ , proving equation 35.

Covariance blowup. As $\begin{array} { r } { s  + \infty , \frac { \lambda _ { \mathrm { c o v } } } { \gamma } s  + \infty } \end{array}$ dominates $\begin{array} { r } { - \frac { \gamma } { 2 } \log s , \ : \mathrm { s o } \ \phi ( s )  + \infty ; } \end{array}$ the same argument with $\lambda _ { \operatorname* { m a x } } ( \mathbf { C } _ { h } )$ in place of $\tilde { \lambda } _ { \mathrm { m i n } } ( \mathbf { C } _ { h } )$ proves equation 36.

Mean blowup. Since $\phi ( s ) \geq b$ for every $s > 0$ , we have

$$
\mathcal { R } _ { \mathrm { S A C } } ( { \pmb \mu } , { \bf C } _ { h } ) \ge \frac { \lambda _ { \mathrm { m e a n } } } { 2 } \| { \pmb \mu } \| _ { 2 } ^ { 2 } + N _ { h } b \qquad \mathrm { f o r e v e r y } { \bf C } _ { h } \succ 0 .
$$

Because $\lambda _ { \mathrm { { m e a n } } } > 0$ and b is finite, $\| \mu \| _ { 2 } \to \infty$ implies ${ \mathcal { R } } _ { \mathrm { S A C } } ( { \pmb { \mu } } , { \bf C } _ { h } )  + \infty$ , even when $\mathbf { C } _ { h }$ varies. This proves equation 37.

The preceding limits imply that, for any fixed finite $K ,$ the condition $\mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) \leq K$ keeps the covariance eigenvalues bounded above and away from zero, and bounds $\| \pmb { \mu } \|$ . Otherwise, a sequence satisfying this condition would approach covariance collapse, unbounded covariance, or unbounded mean, forcing the regularizer to diverge and contradicting its upper bound $K$ . This proves equation 38. □

Remark A.16 (Effect of the numerical stabilizer). The regularizer prevents covariance collapse when $\varepsilon = 0$ , but only discourages it when $\varepsilon > 0$ . In the latter case, its unique covariance-level minimizer is still full-rank provided $\gamma > \lambda _ { \mathrm { c o v } } \varepsilon$ (Theorem A.13). Mean boundedness remains guaranteed in both cases.

Remark A.17 (Scope of the nonlinear extension). The nonlinear covariance regularizer is motivated by the exact linear balance analysis, but it does not imply that the linear balance identity $\mathbf { W } _ { 2 } ^ { \top } \mathbf { W } _ { 2 } -$ $\begin{array} { r } { \dot { \mathbf { W } } _ { 1 } \mathbf { W } _ { 1 } ^ { \top } = - \frac { \gamma } { \lambda _ { \mathrm { r e g } } } \mathbf { I } _ { N _ { h } } } \end{array}$ continues to hold for a general nonlinear encoder. The nonlinear parameter dynamics are modified by the encoder Jacobian, and there is generally no conserved difference between the decoder Gram matrix and the representation covariance.

## A.2.3 FEATURE LEARNING: TWO-LAYER RELU EXPERIMENT

Task and data. We study nonlinear teacher–student regression with a fixed, untrained two-layer ReLU teacher of hidden width $N _ { h _ { 0 } } = 3$ . The teacher is

$$
f _ { \star } ( \mathbf { x } ) = \sum _ { j = 1 } ^ { N _ { h _ { 0 } } } a _ { j } ^ { \star } \sigma \big ( ( \mathbf { w } _ { j } ^ { \star } ) ^ { \top } \mathbf { x } \big ) , \qquad \sigma ( z ) = \mathrm { m a x } \{ 0 , z \} ,
$$

where $\mathbf { w } _ { i } ^ { \star }$ are sampled independently and uniformly from the unit sphere $\mathbb { S } ^ { d - 1 }$ , and $a _ { j } ^ { \star }$ are independent Rademacher random variables. For each run, we sample $n = 1 { , } 0 0 0$ training inputs independently from $\mathbb { S } ^ { N _ { i } - 1 }$ and assign noiseless labels $y _ { i } = f _ { \star } ( \mathbf { x } _ { i } )$ . A separate test set of $n _ { \mathrm { t e s t } } = 1 0 { , } 0 0 0$ inputs, generated with seed 201, is reused at every checkpoint for both test-loss and representationrank measurements.

![](images/cc421cac964b4e0b4eb63a6793bc8fd612856bba70d8bb62147f4286f43420d2.jpg)  
Figure 4: Effective rank and kernel distance of the NTK from initialization. Top row: Both metrics as functions of the scale τ and relative scale λ at step $t = 2 0 { , } 0 0 0$ , without regularization $( w _ { \mathrm { r e g } } = 0 )$ Bottom row: The same metrics at the final checkpoint as functions of the scale τ and regularization weight $w _ { \mathrm { r e g } } .$ , with $\lambda = 0$ fixed.

Student architecture. The student is a two-layer, bias-free ReLU network with hidden width $N _ { h } = 5 0 \colon$

$$
\mathbf { h } _ { \theta } ( \mathbf { x } ) = \sigma ( \mathbf { W } \mathbf { x } ) \in \mathbb { R } ^ { N _ { h } } , \qquad f _ { \theta } ( \mathbf { x } ) = \mathbf { a } ^ { \top } \mathbf { h } _ { \theta } ( \mathbf { x } ) .
$$

We measure the rank of the post-ReLU representation $\mathbf { h } _ { \theta } ( \mathbf { x } )$ , the network’s only hidden layer. The student uses a symmetric initialization: hidden units form $\dot { N } _ { h } / 2$ pairs with identical input weights and opposite readout weights,

$$
\begin{array} { r } { \mathbf { w } _ { j + N _ { h } / 2 } = \mathbf { w } _ { j } , \qquad a _ { j + N _ { h } / 2 } = - a _ { j } , \qquad j = 1 , \ldots , N _ { h } / 2 . } \end{array}
$$

This ensures $f _ { \theta _ { 0 } } ( \mathbf { x } ) = \mathbf { 0 }$ for every input while retaining a nontrivial initial neural tangent kernel (NTK).

Initialization. Two parameters control initialization: an overall scale $\tau > 0$ and a per-neuron imbalance λ. For each independently sampled pair, we initialize

$$
\mathbf { w } _ { j } = \frac { \tau } { \alpha } \mathbf { u } _ { j } , \qquad a _ { j } = \tau \alpha s _ { j } ,
$$

where $\mathbf { u } _ { j }$ is uniform on $\mathbb { S } ^ { N _ { i } - 1 } , s _ { j } \ \in \ \{ - 1 , + 1 \}$ is a Rademacher random variable, and $\alpha > 0$ satisfies

$$
\lambda = a _ { j } ^ { 2 } - \| \mathbf { w } _ { j } \| _ { 2 } ^ { 2 } = \tau ^ { 2 } \left( \alpha ^ { 2 } - \alpha ^ { - 2 } \right) .
$$

Equivalently,

$$
\alpha = \left( \frac { \lambda / \tau ^ { 2 } + \sqrt { ( \lambda / \tau ^ { 2 } ) ^ { 2 } + 4 } } { 2 } \right) ^ { 1 / 2 } .
$$

Thus, negative λ corresponds to larger encoder weights, whereas positive λ corresponds to larger readout weights.

Objective and optimization. We train the student using full-batch gradient descent on the mean squared error,

$$
\mathcal { L } _ { \mathrm { M S E } } ( \boldsymbol { \theta } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { i } ) - y _ { i } \right) ^ { 2 } ,
$$

with an additional regularization term in the regularized experiments. The learning rate is scaled as $\eta = 5 \times 1 0 ^ { - 3 } / \tau ^ { 2 }$ . We apply the log-determinant covariance regularizer directly to the hidden representation of the toy two-layer ReLU student network,

$$
\mathbf { h } = \mathrm { R e L U } ( \mathbf { W } _ { 1 } \mathbf { x } ) \in \mathbb { R } ^ { N _ { h } }
$$

(here $N _ { h } = 5 0 )$ , immediately after the nonlinearity and before the readout layer. We use the implementation of the regularizer described in Appendix B.1.

We use $\lambda _ { \mathrm { c o v } } = \gamma$ . Therefore, the regularized training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + w _ { \mathrm { r e g } } \cdot \mathcal { R } _ { \mathrm { S A C } } ( \pmb { \mu } , \mathbf { C } _ { h } ) ,
$$

with $\mathcal { L } _ { \mathrm { t a s k } }$ the mean-squared error between the student’s output and the teacher’s labels, and $w _ { \mathrm { r e g } } \in$ $\{ 0 , 0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 2 , \hat { 0 } . 1 \}$ swept as the regularizer strength.

Feature learning. Using the two-layer ReLU network setup of Kunin et al. (2024), we investigate how the rank effects and the learning regime of the initial imbalance λ relate to those of the SAC regularizer. Linear theory suggests that a negative relative scale should favor the same anti-collapse behavior promoted by SAC regularization.

• Rank. There is no exact correspondence between the initial imbalance λ and the regularization weight $w _ { \mathrm { r e g } }$ . The regularizer appears to exert a substantially stronger effect on rank (see Fig. 4). Nevertheless, both negative λ (relative to $\lambda \geq 0 )$ and positive $w _ { \mathrm { r e g } }$ favor higher rank in the Rich regime, supporting a qualitative connection between initial imbalance and anti-collapse regularization (see Fig. 4). Interestingly, the rank effect of negative λ emerges during training rather than reflecting a decaying initial offset. All negative-λ runs with $w _ { \mathrm { r e g } } = 0$ begin at approximately the same effective rank fraction, effrank frac = 0.467, consistent with their shared initialization (see Table 3). This quantity then increases monotonically throughout training, indicating that the higher final rank develops through the training dynamics.

• Learning regime. We characterize the learning regime by measuring the departure of the neural tangent kernel (NTK) from its value at initialization, quantified by kernel distance. Increasing $w _ { \mathrm { r e g } }$ leads to an increase in kernel distance across the initialization scales considered, indicating that stronger regularization promotes greater kernel evolution (see Fig. 4). It also extends the range of initialization scales over which feature learning occurs, although kernel evolution remains limited at sufficiently large scales (see Fig. 4). At small initialization scales, varying λ alone yields kernel distance values between 0.16 and 0.58, comparable to the range obtained by varying $w _ { \mathrm { r e g } }$ . At large initialization scales, however, both parameters have little effect on kernel evolution. These results suggest that λ primarily modulates the degree of feature learning within an already rich regime, whereas $w _ { \mathrm { r e g } }$ can also promote departures from otherwise lazy dynamics. Altogether, we find that regularization promotes feature learning across a range of initialization scales, with stronger regularization associated with more pronounced feature learning (see Fig. 4 in Appendix A.2.3). A similar trend emerges in the self-supervised setting. We show that both the backbone representations and the empirical NTK move substantially from initialization during training, consistent with rich learning dynamics (Fig. 5 and Appendix A.2.4).

## A.2.4 FEATURE LEARNING IN THE VIT

We additionally measure feature and kernel drift during self-supervised training on ImageNet-100. We train a ViT-S/16 for 100 epochs with SACReg applied to the backbone representation and evaluate all checkpoints on fixed held-out images.

Table 3: Effective rank fraction across training steps for different values of layer imbalance λ.
<table><tr><td rowspan="2"> $\lambda$ </td><td colspan="6">Training step</td></tr><tr><td>0</td><td>100</td><td>1000</td><td>5000</td><td>10000</td><td>20000</td></tr><tr><td>-1.00</td><td>0.467</td><td>0.467</td><td>0.484</td><td>0.554</td><td>0.574</td><td>0.581</td></tr><tr><td>-0.75</td><td>0.467</td><td>0.467</td><td>0.487</td><td>0.563</td><td>0.587</td><td>0.595</td></tr><tr><td>-0.50</td><td>0.467</td><td>0.467</td><td>0.492</td><td>0.577</td><td>0.606</td><td>0.616</td></tr><tr><td>-0.25</td><td>0.467</td><td>0.469</td><td>0.500</td><td>0.596</td><td>0.627</td><td>0.639</td></tr></table>

![](images/d95ceefd678c7aa8d94e0b3e16a6ef7dc3b61c9f9fa99173bf1d5aafa99057e7.jpg)

![](images/1d78c95a5a4df50b675919f3e2e7504a06c95e33187bc19f97674f085db499f0.jpg)  
Figure 5: Feature and kernel drift during self-supervised training. ImageNet-100, ViT-S/16 trained for 100 epochs with SACReg applied at the backbone. Left: feature drift from initialization, measured by 1 − CKA at the CLS token after transformer blocks 3, 6, 9, and 12. Right: empirical NTK distance from initialization (solid) and from the previous checkpoint (dashed). Representations drift increasingly with depth, while the NTK moves substantially early in training and subsequently stabilizes, consistent with rich rather than lazy learning dynamics.

Feature drift. For 4,096 validation images, let $\mathbf { X } _ { 0 } , \mathbf { X } _ { t } \in \mathbb { R } ^ { 4 0 9 6 \times 3 8 4 }$ denote the CLS representations at initialization and epoch $t ,$ respectively, at a given transformer block. We measure their dissimilarity using one minus linear CKA (Kornblith et al., 2019),

$$
1 - \operatorname { C K A } ( \mathbf { X } _ { 0 } , \mathbf { X } _ { t } ) , \qquad \operatorname { C K A } ( \mathbf { X } , \mathbf { Y } ) = \frac { \| \mathbf { X } _ { c } ^ { \top } \mathbf { Y } _ { c } \| _ { F } ^ { 2 } } { \| \mathbf { X } _ { c } ^ { \top } \mathbf { X } _ { c } \| _ { F } \left\| \mathbf { Y } _ { c } ^ { \top } \mathbf { Y } _ { c } \right\| _ { F } } ,
$$

where $\mathbf { X } _ { c }$ and ${ \bf Y } _ { c }$ are column-centered. We report this quantity after blocks 3, 6, 9, and 12. Linear CKA is invariant to orthogonal transformations and isotropic rescaling, so the measure captures changes in representation geometry rather than changes in feature coordinates or overall scale.

Kernel drift. We estimate the empirical NTK of the backbone output with respect to the model parameters on a fixed subset of 256 validation images. To avoid constructing the full vector-valued NTK, we use eight fixed unit-norm random directions $\mathbf { v } _ { k } \in \mathbb { R } ^ { 3 8 4 }$ in output space and compute

$$
{ \bf g } _ { i k } ^ { ( t ) } = \nabla _ { \theta } \left( { \bf v } _ { k } ^ { \top } { \bf h } _ { \theta } ( { \bf x } _ { i } ) \right) , \qquad { \bf K } _ { t } [ i , j ] = \frac { 1 } { 8 } \sum _ { k = 1 } ^ { 8 } \left. { \bf g } _ { i k } ^ { ( t ) } , { \bf g } _ { j k } ^ { ( t ) } \right. .
$$

The same images and random directions are used at every checkpoint. Up to a constant factor, $\mathbf { K } _ { t }$ is an unbiased estimate of the trace NTK; this factor cancels in the normalized comparison. We measure kernel drift as

$$
d ( \mathbf { K } _ { t } , \mathbf { K } _ { s } ) = 1 - \frac { \langle \mathbf { K } _ { t } , \mathbf { K } _ { s } \rangle _ { F } } { \| \mathbf { K } _ { t } \| _ { F } \| \mathbf { K } _ { s } \| _ { F } } ,
$$

and report distance both from initialization $( s ~ = ~ 0 )$ and from the preceding checkpoint. In the lazy-training limit, the NTK remains fixed, so its distance from initialization stays at zero.

Figure 5 shows substantial drift in both representations and the empirical NTK. Feature drift increases with depth, reaching its largest value at the final transformer block. The NTK moves away

![](images/f00c5d66bed953d9309a4b37f0d72af22b8fdd873fe2d77e115797fa8f46ffa0.jpg)  
Figure 6: Schematic overview of λ-JEPA. Augmented views are mapped by the encoder $( f _ { \theta } )$ to hidden representations (h), then by $( p _ { \varphi } )$ to embeddings (z). SACReg applies regularization directly in the hidden space to prevent per-view collapse, encouraging a more dispersed representation distribution (green) compared with training without SACReg (orange). Filled circles denote image centers; open circles denote individual views.

from its initialization during the first 25 epochs and then changes only slightly between subsequent checkpoints. Together, these measurements are consistent with rich feature learning.

## B IMPLEMENTATION OF THE REGULARIZER

Invariance objective. For image i with V augmented views, we use the mean squared distance between all pairs of projected representations,

$$
\mathcal { L } _ { \mathrm { S S L } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { 2 } { V ( V - 1 ) } \sum _ { j < k } \frac { 1 } { d _ { z } } \left\| \mathbf { z } _ { i } ^ { ( j ) } - \mathbf { z } _ { i } ^ { ( k ) } \right\| _ { 2 } ^ { 2 } ,
$$

where B is the number of images in the batch and $d _ { z }$ is the projector dimension. This objective contains neither negatives nor an anti-collapse term. Using $\begin{array} { r } { \bar { \bf z } _ { i } = \frac { 1 } { V } \sum _ { j = 1 } ^ { V } { \bf z } _ { i } ^ { ( j ) } } \end{array}$ , it is equivalently $\frac { 2 V } { V - 1 }$ times the average squared distance from each view to its image center. All runs use global views of a single size: $V \ = \ 4$ on ImageNet-100 and $V ~ = ~ 6$ on ImageNet-1k and video. The invariance and regularization weights are set jointly as described in Appendix C.

Slicing. For B view centers $\mathbf { R } = \{ \mathbf { r } _ { i } \} _ { i = 1 } ^ { B } \subset \mathbb { R } ^ { d }$ , the centered empirical covariance has rank at most $B - 1$ . Thus, when $d \ge B$ , the full log-determinant is degenerate. At each step, we draw a random orthonormal projection $\mathbf { U } \in \mathbb { R } ^ { d \times d ^ { \prime } }$ using a thin QR factorization of a Gaussian matrix and evaluate SACReg on the projected centers $\{ \mathbf { U } ^ { \top } \mathbf { r } _ { i } \} _ { i = 1 } ^ { B }$ . The same projection is shared across GPUs and resampled every step. Within the slice, we compute the projected mean and covariance and use $\varepsilon = 1 0 ^ { - 4 }$ . We divide the regularizer by $d ^ { \prime }$ , making its scale approximately independent of the slice width.

We use $d ^ { \prime } = 1 2 8$ after the projector $( d _ { z } = 2 5 6 )$ . At the backbone, we use one third of the feature width: $d ^ { \prime } = 1 2 8$ for ViT-S $( \bar { d } = 3 8 4 )$ and $d ^ { \prime } = 2 5 6$ for ViT-B $( d = 7 6 8 )$ . The backbone and projector regularizers use independent random slices.

## B.1 RELATION TO THE FULL-COVARIANCE REGULARIZER

For clarity, consider the idealized case $\varepsilon ~ = ~ 0$ . Let $\pmb { \mu } \in \mathbb { R } ^ { d }$ and $\Sigma \succ 0$ denote the mean and covariance of the view centers. The full regularizer per dimension is

$$
\mathcal { R } _ { \mathrm { f u l l } } ( \mu , \Sigma ) = \frac { 1 } { 2 d } \left( \| \pmb { \mu } \| _ { 2 } ^ { 2 } + \operatorname { t r } \Sigma - d - \log \operatorname* { d e t } \Sigma \right) .
$$

For an orthonormal frame $\textbf { U } \in \mathbb { R } ^ { d \times d ^ { \prime } }$ , the projected centers have mean $\mathbf { U } ^ { \top } \pmb { \mu }$ and covariance $\mathbf { U } ^ { \top } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { U }$ , giving the sliced regularizer

$$
{ \mathcal { R } } _ { \mathbf { U } } ( { \boldsymbol \mu } , { \boldsymbol { \Sigma } } ) = \frac { 1 } { 2 d ^ { \prime } } \left( \| \mathbf { U } ^ { \top } { \boldsymbol { \mu } } \| _ { 2 } ^ { 2 } + \operatorname { t r } ( \mathbf { U } ^ { \top } { \boldsymbol { \Sigma } } \mathbf { U } ) - d ^ { \prime } - \log \operatorname* { d e t } ( \mathbf { U } ^ { \top } { \boldsymbol { \Sigma } } \mathbf { U } ) \right) .
$$

Since a fresh Haar-random frame is sampled at every step, the corresponding expected slice objective is

$$
\bar { \mathcal { R } } _ { d ^ { \prime } } ( \pmb { \mu } , \pmb { \Sigma } ) = \mathbb { E } _ { \mathbf { U } } [ \mathcal { R } _ { \mathbf { U } } ( \pmb { \mu } , \pmb { \Sigma } ) ] .
$$

Proposition B.1 (Random slicing). For every $1 \leq d ^ { \prime } \leq d \colon$

(i) $\bar { \mathcal { R } } _ { d ^ { \prime } } \geq 0 ,$ , with equality if and only $i f \pmb { \mu } = \mathbf { 0 }$ and $\Sigma = \mathbf { I } _ { d } .$

(ii) The normalized mean and total variance are preserved in expectation:

$$
\mathbb { E } _ { \mathbf { U } } \frac { \| \mathbf { U } ^ { \top } \pmb { \mu } \| _ { 2 } ^ { 2 } } { d ^ { \prime } } = \frac { \| \pmb { \mu } \| _ { 2 } ^ { 2 } } { d } , \qquad \mathbb { E } _ { \mathbf { U } } \frac { \mathrm { t r } ( \mathbf { U } ^ { \top } \pmb { \Sigma } \mathbf { U } ) } { d ^ { \prime } } = \frac { \mathrm { t r } \pmb { \Sigma } } { d } .
$$

(iii) The expected sliced objective approaches the full one monotonically:

$$
\bar { \mathcal { R } } _ { d ^ { \prime } } \leq \bar { \mathcal { R } } _ { d ^ { \prime } + 1 } \leq \mathcal { R } _ { \mathrm { f u l l } } , \qquad \bar { \mathcal { R } } _ { d } = \mathcal { R } _ { \mathrm { f u l l } } .
$$

Thus, slicing does not change the preferred representation: averaging over random subspaces still uniquely favors zero mean and isotropic unit covariance. The mean and total variance are preserved exactly in expectation, while the log-determinant observes only the spectrum inside the sampled subspace. Wider slices therefore capture progressively more of the full spectral penalty, recovering the original regularizer when $d ^ { \prime } = \bar { d } .$ In practice, we choose $d ^ { \prime } < d \mathrm { s o }$ that the projected covariance can be estimated from substantially more centers than dimensions.

Proof. Each ${ \mathcal { R } } _ { \mathbf { U } }$ is nonnegative and is minimized when $\mathbf { U } ^ { \top } \pmb { \mu } = \mathbf { 0 }$ and $\mathbf { U } ^ { \top } \pmb { \Sigma } \mathbf { U } = \mathbf { I } _ { d ^ { \prime } }$ . If this holds for all random subspaces, then every one-dimensional direction has zero projected mean and unit variance, implying ${ \pmb \mu } = { \bf 0 }$ and $\pmb { \Sigma } = \mathbf { I } _ { d }$

Haar invariance gives

$$
\mathbb { E } _ { \mathbf { U } } [ \mathbf { U U } ^ { \top } ] = \frac { d ^ { \prime } } { d } \mathbf { I } _ { d } ,
$$

which directly yields (ii). Finally, concavity of the logarithm gives

$$
\frac { 1 } { d ^ { \prime } } \mathbb { E } _ { \mathbf { U } } \log \operatorname* { d e t } ( \mathbf { U } ^ { \top } \Sigma \mathbf { U } ) \geq \frac { 1 } { d } \log \operatorname* { d e t } \Sigma .
$$

Together with (ii), this gives $\bar { \mathcal { R } } _ { d ^ { \prime } } \leq \mathcal { R } _ { \mathrm { f u l l } }$ . Applying the same argument to a random d<sup>′</sup>-dimensional subspace inside a random $( d ^ { \prime } + 1 )$ -dimensional subspace gives $\bar { \mathcal { R } } _ { d ^ { \prime } } \leq \bar { \mathcal { R } } _ { d ^ { \prime } + 1 }$ . For $d ^ { \prime } = d ,$ , U is orthogonal and the two objectives are identical.

Ring buffer. Reliable covariance estimation requires more centers than the slice dimension. For each regularized space, we therefore maintain a FIFO buffer containing detached view centers from the previous q steps and concatenate them with the current batch before computing the regularizer. This gives

$$
B _ { \mathrm { e f f } } = ( q + 1 ) B ,
$$

where only the current B centers carry gradients. Under data parallelism, the buffer contains centers from the full global batch.

We keep $B _ { \mathrm { e f f } } / d ^ { \prime } = 4$ throughout image training. Thus, $q = 3$ after the projector and at the ViT-S backbone $( B _ { \mathrm { e f f } } = 5 1 2 , d ^ { \prime } \stackrel { . } { = } 1 2 8 )$ , while $q = 7$ at the ViT-B backbone $\dot { ( B _ { \mathrm { e f f } } } = 1 0 2 4 , d ^ { \prime } = 2 5 6 )$ On video, the larger batch sizes reduce the need for buffering: no buffer is used for ViT-S, while ViT-B uses $q = 1$ at the backbone. We observed no measurable effect from the short parameter lag introduced by the buffered centers.

Averaged invariance target. For ImageNet-1k and video, we follow LeJEPA and compute the invariance target using an exponential moving average of the backbone and projector. The target $\tilde { \mathbf { z } } _ { i }$ is the mean projected representation of the V views under the averaged network, and we use

$$
\mathcal { L } _ { \mathrm { S S L } } = \frac { 2 V } { V - 1 } \frac { 1 } { B V d _ { z } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { V } \left. \mathbf { z } _ { i } ^ { ( j ) } - \tilde { \mathbf { z } } _ { i } \right. _ { 2 } ^ { 2 } .
$$

ImageNet-100 uses the pairwise objective directly. In all cases, SACReg is computed from view centers of the current network.

## C EXPERIMENTAL DETAILS

## C.1 DATASETS

ImageNet-1k (Russakovsky et al., 2015) is the standard split with 1.28M training and 50k validation images. ImageNet-100 is the 100-class subset introduced by CMC (Tian et al., 2020a), with 126,689 training and 5,000 validation images. For video we follow LeVJEPA (Kuhn et al., 2026): we take the union of the Kinetics-400, -600 and -700 (Li et al., 2023) training sets, remove duplicates and every clip that appears in a validation set, and keep 20% of the clips of each class (120,100 clips after decoding). LeVJEPA does not release its subset, so ours is a different draw of the same size. Transfer (Ericsson et al., 2021) uses DTD (Cimpoi et al., 2014), Aircraft (Maji et al., 2013), Cars (Krause et al., 2013), CIFAR10, CIFAR100 (Krizhevsky, 2009), Flowers (Nilsback & Zisserman, 2008), Food (Bossard et al., 2014) and Pets (Parkhi et al., 2012) with the splits of the VISReg protocol (Wu et al., 2026); video encoders are evaluated on ImageNet-1k, Something-Something-v2 (Goyal et al., 2017) and Kinetics-400 (Kay et al., 2017).

## C.2 MODELS AND TRAINING

Large-scale image runs. We train ViT-S/16 and ViT-B/16 at $2 2 4 ^ { 2 }$ with a two-layer projector (hidden width 2048, output 256, BatchNorm), batch size 128 split over two GPUs, for 100 and 400 epochs. Each image gives $V = 6$ global views from the LeJEPA augmentation family: random resized crops covering 30–100% of the image, horizontal flips, color jitter, random grayscale, Gaussian blur, and solarization on one of the views. We do not use local crops. The slices and buffers are as described in Appendix B. Optimization uses AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay 0.05, 10 warm-up epochs followed by a cosine schedule, gradient clipping at $1 . 0 ,$ and bf16 precision. The loss weights are (31.07, 225.50, 1.671) for ViT-S and (22.81, 222.26, 4.054) for ViT-B, for the invariance term, $\beta _ { z }$ and $\beta _ { h }$ respectively; they are set as described below and shared by the 100- and 400-epoch runs.

Controlled experiments on ImageNet-100. All controlled experiments use ViT-S/16 at $2 2 4 ^ { 2 }$ , are trained for 100 epochs with batch size 128, and are repeated over three independent random seeds. We re-implement LeJEPA, VICReg, SimCLR, DINO, BYOL, and VISReg within a common train ing codebase while preserving each method’s released loss, projector, and augmentation settings. For each method, we train a baseline and a matched variant with $\beta _ { h } \mathrm { S A C R e g } ( \bar { \mathbf { H } } )$ added; all other settings are kept fixed, so the two variants differ only by the backbone regularization term. We report the mean across the three seeds. The regularizer is applied to the representation provided as input to each method’s projector: the CLS token for SimCLR, BYOL, VICReg, and DINO, and the 512- dimensional embedding immediately preceding the projector for LeJEPA and VISReg. Our own objective uses $V = 4$ views and the pairwise invariance term defined in Appendix B, with weights 32.8 for invariance, 157.8 for $\beta _ { z }$ , and 1.894 for $\beta _ { h }$ . The corresponding baseline sets $\beta _ { h } = 0$ , with all other settings unchanged.

Video. We use LeVJEPA’s training pipeline and released configuration unchanged except for the loss: their video ViT-S/16 and ViT-B/16 (16 frames at stride 2, tubelet size 1, 95% token drop), their projector, and their optimizer (AdamW, learning rate $4 \times 1 0 ^ { - 4 }$ , weight decay 0.04, warm-up over 12% of training followed by a constant learning rate, bf16). Our loss takes $V \stackrel { } { = } 6$ global $2 2 4 ^ { 2 }$ views of the same clip. The batch size is 768 for ViT-S and 512 for ViT-B, the loss weights are those of the image run of the same model size, and we evaluate the exponential-moving-average encoder, as LeVJEPA does.

## C.3 SETTING THE LOSS WEIGHTS

The loss terms have different forms and scales, so we do not balance them by their values but by their realized pull on the backbone. For a term with weight $w ,$ let $g$ be the norm of the gradient of the unweighted term with respect to the backbone parameters, measured on a fixed batch at a given checkpoint. The product w g is the pull of the term, and its share of the total pull is w $g / \sum _ { k } \bar { w } _ { k } g _ { k }$

Adding the backbone term to an existing method. When adding $\beta _ { h } \mathrm { S A C R e g } ( \bar { \mathbf { H } } )$ to an existing SSL method, we choose $\beta _ { h }$ so that the backbone regularizer contributes a small fraction $T = 0 . 0 6$ of the sum of weighted backbone-gradient norms. Let

$$
G _ { \mathrm { S S L } } = \sum _ { k } w _ { k } g _ { k }
$$

denote the contribution of the method’s original loss terms, measured at its epoch-25 checkpoint. We set

$$
\beta _ { h } = \frac { T } { 1 - T } \frac { G _ { \mathrm { S S L } } } { g _ { R } } ,
$$

using a fixed reference $g _ { R } = 0 . 3 0$ for the gradient norm of $\operatorname { S A C R e g }$ . We use a fixed reference because this gradient can be unusually large at a collapsed backbone, which would otherwise yield an artificially small weight. The target share $T = 0 . 0 6$ was chosen from an earlier dose study on LeJEPA. Thus, $T$ and $g _ { R }$ are fixed across methods, and $\beta _ { h }$ is determined from the scale of each method’s existing objective rather than tuned separately.

This gives $\beta _ { h } ~ = ~ 0 . 0 9 8$ (SimCLR), 0.0205 (BYOL), 1.548 (VICReg), 0.258 (DINO), 0.0146 (LeJEPA), and 0.0123 (VISReg; measured from a one-epoch pilot). For DINO, this weight reduced both linear and kNN accuracy, so we tested {0.115, 0.02, 0.01} and use $\beta _ { h } = 0 . 0 1$ , selected by the online probe.

Our standalone objective. For our standalone objective, we set the three loss weights jointly. After a one-epoch pilot, we measure the backbone gradient norm $g _ { k }$ of each term and choose

$$
w _ { k } = 5 7 \frac { s _ { k } } { g _ { k } } ,
$$

with target shares $( s _ { \mathrm { { i n v } } } , s _ { \mathrm { { p r o j } } } , s _ { \mathrm { { b a c k } } } ) = ( 0 . 4 4 , 0 . 5 4 , 0 . 0 2 )$ . These shares and the total magnitude 57 come from an earlier ViT-S run, and we use the same calibration rule for ViT-S and ViT-B without further tuning.

## C.4 EVALUATION

ImageNet-1k linear probe and kNN. We follow the linear-evaluation recipe of the Lightly benchmark (the MAE linear-probe recipe): a BatchNorm layer without affine parameters followed by a linear classifier on the frozen CLS token, trained for 90 epochs with random resized crops and flips, SGD with momentum 0.9 (equivalent to LARS at zero weight decay), learning rate 0.1×batch/256 with a cosine schedule and 10 warm-up epochs; we report the best validation top-1 over the 90 epochs. The kNN classifier uses $k = 2 0 0$ neighbors with cosine similarity weighted at temperature 0.07, on the CLS features of the training set.

Transfer. We use the VISReg transfer protocol: the CLS tokens of the last four blocks are concatenated, a linear classifier is trained for 10 epochs over a grid of 13 learning rates, and the best test accuracy is reported for each dataset; the table reports the average over the eight datasets. Our implementation reproduces the published DINO ViT-B/16 row within 0.4 points on every dataset.

ImageNet-100. The linear probe is a linear classifier on the raw CLS features, trained with AdamW (learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 7 } )$ until validation accuracy stops improving; the kNN classifier uses $k = 2 0 0$ and temperature 0.1. Both are fit on a fixed subset of 500 training images per class and evaluated on the 5,000 validation images.

Video. We follow the frozen-encoder protocols of V-JEPA as used by LeVJEPA. ImageNet-1k and Something-Something-v2 use an attentive probe: one cross-attention block with a learnable query over all output tokens followed by a linear classifier, trained for 20 epochs with AdamW (learning rate $1 0 ^ { - 3 }$ , cosine schedule). For ImageNet-1k each image is repeated over the 16 input frames; for Something-Something- $- \mathbf { v } 2$ a video is sampled as 2 clips of 16 frames with 3 spatial crops. Kinetics 400 uses a linear classifier. LeVJEPA does not release the settings of this linear classifier. So we report the results with L-BFGS (multinomial logistic regression on standardized features, the best of three ridge strengths $\{ 1 0 ^ { - 5 } , 1 \dot { 0 } ^ { - 4 } , 1 0 ^ { - 3 } \} ,$ ) using one center crop per clip and the full training and validation sets.

Representation statistics. All statistics are computed on frozen features. RankMe (Garrido et al., 2023) is the exponential of the entropy of the normalized singular values reported as a fraction of the feature dimension. Positive and negative-pair cosines are the mean cosine similarity between two views of the same image and between views of two different images.

Augmentation thickness. For every model we store V = 8 augmented views of 10,000 training images (100 per class), drawn from the model’s own training augmentations. For the per-class analysis of Fig. 8, we split the 10,000 images once at random into two equal halves. On the fitting half, we estimate ${ \bf A } _ { h }$ as the average within-image covariance of the views and $\mathbf { B } _ { h }$ as the covariance of the view means minus $\mathbf { A } _ { h } / V$ , which corrects for estimating each image center from finitely many views. We use $\mathbf { B } _ { h }$ to define the whitening map and compute $\mathbf { \bar { \Theta } } \mathbf { \Theta } \mathbf { \Theta } \mathbf { \Theta } \Theta _ { h } = \mathbf { B } _ { h } ^ { \dagger / 2 } \mathbf { A } _ { h } \mathbf { B } _ { h } ^ { \dagger / 2 }$

For each class c, we define the target as the centered indicator scaled to unit variance,

$$
y _ { c } = { \frac { \mathbf { 1 } [ y = c ] - p _ { c } } { \sqrt { p _ { c } ( 1 - p _ { c } ) } } } ,
$$

where $p _ { c }$ is the class frequency in the fitting split. Using the fitting half, we fit the least-squares coefficient $\mathbf { a } _ { c }$ for predicting $y _ { c }$ from the whitened image centers. We then define $d _ { h } ( c ) ^ { 2 }$ as the mean squared residual on the held-out half. Since this is a held-out error, it can exceed 1, so the class-center separability $1 - d _ { h } ( c ) ^ { 2 }$ can be negative for some classes. The view-sensitivity term $\mathbf { a } _ { c } ^ { \top } \Theta _ { h } ( \mathbf { I } _ { r _ { h } } + \bar { \Theta } _ { h } ) ^ { - 1 } \mathbf { a } _ { c }$ uses the same $\mathbf { a } _ { c }$ and the $\Theta _ { h }$ estimated on the fitting half.

## C.5 BASELINES

OK-AI releases LeJEPA, DINO and iBOT trained under one modernized code base at ViT-S and ViT-B for 100 and 300 epochs; we evaluate these checkpoints under the protocols above. Their model card lists 1.45M training images, which is more than the standard ImageNet-1k training split; we did not correct for this. We also trained VISReg ViT-B/16 for 100 epochs with the authors’ code and their released view configuration (four global and six local crops), and evaluated the public MoCo v3, DINO, iBOT and VISReg checkpoints (300–400 epochs) the same way. Transfer numbers marked <sup>‡</sup> are taken from the VISReg paper. Video baselines are taken from the LeVJEPA paper: the 240-epoch numbers from its Fig. 2 and the remaining rows from its equal-compute table.

## C.6 DETAILED RESULTS

## D A CLOSER LOOK AT LEJEPA

Our results suggest that anti-collapse should act on the representation retained for downstream tasks. A natural alternative explanation is that the particular regularizer is unimportant: perhaps simply applying an existing anti-collapse objective at the backbone is sufficient. We test this with LeJEPA, one of the explicitly regularized methods in our study.

LeJEPA encourages the projected representation to match an isotropic Gaussian using SIGReg, a sketched Gaussianity test over random one-dimensional projections. However, its retained backbone representation is far from this target (Fig. 1). Starting from vanilla LeJEPA, with the original SIGReg and invariance losses after the projector unchanged, we therefore add a second SIGReg term directly to the 512-dimensional encoder output, using the weighting heuristic of Appendix C.3, and train under the ImageNet-100 setup of Section 4.4.

Table 4: Per-dataset linear-probe transfer accuracy (%) under the VISReg protocol; the last column is the average reported in Table 1a and 1b. <sup>‡</sup>Numbers reported by the VISReg paper; all other rows are evaluated by us under the matched protocol.
<table><tr><td>Method</td><td>Backbone</td><td> $\mathrm { E p . }$ </td><td>DTD</td><td>Aircraft</td><td>Cars</td><td>CIFAR10</td><td>CIFAR100</td><td>Flowers</td><td>Food</td><td>Pets</td><td>Avg.</td></tr><tr><td>LeJEPA</td><td>ViT-S/16</td><td>100</td><td>69.4</td><td>43.8</td><td>43.4</td><td>89.3</td><td>71.1</td><td>84.0</td><td>73.6</td><td>75.3</td><td>68.7</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>100</td><td>69.3</td><td>55.2</td><td>57.0</td><td>93.8</td><td>78.6</td><td>89.3</td><td>76.7</td><td>85.9</td><td>75.7</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>100</td><td>69.3</td><td>55.3</td><td>56.1</td><td>93.3</td><td>77.4</td><td>90.3</td><td>77.9</td><td>87.3</td><td>75.9</td></tr><tr><td>λ-JEPA</td><td>ViT-S/16</td><td>100</td><td>71.1</td><td>55.5</td><td>62.5</td><td>95.3</td><td>81.6</td><td>90.4</td><td>75.3</td><td>89.1</td><td>77.6</td></tr><tr><td>LeJEPA</td><td>ViT-S/16</td><td>300</td><td>70.2</td><td>47.6</td><td>50.3</td><td>92.1</td><td>74.4</td><td>85.8</td><td>75.7</td><td>79.1</td><td>71.9</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>300</td><td>71.5</td><td>59.8</td><td>66.2</td><td>95.0</td><td>81.1</td><td>92.0</td><td>79.6</td><td>89.6</td><td>79.4</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>300</td><td>71.6</td><td>59.7</td><td>63.3</td><td>96.0</td><td>82.4</td><td>91.2</td><td>80.9</td><td>90.6</td><td>79.5</td></tr><tr><td>λ-JEPA</td><td>ViT-S/16</td><td>400</td><td>71.8</td><td>57.9</td><td>67.9</td><td>95.5</td><td>81.8</td><td>90.3</td><td>78.1</td><td>90.7</td><td>79.3</td></tr><tr><td>LeJEPA</td><td>ViT-B/16</td><td>100</td><td>72.3</td><td>51.5</td><td>54.4</td><td>93.0</td><td>76.4</td><td>86.9</td><td>78.9</td><td>80.1</td><td>74.2</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>100</td><td>70.0</td><td>58.1</td><td>61.9</td><td>95.8</td><td>82.0</td><td>89.9</td><td>79.7</td><td>88.7</td><td>78.3</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>100</td><td>71.9</td><td>61.1</td><td>66.8</td><td>97.0</td><td>84.2</td><td>91.5</td><td>82.8</td><td>90.1</td><td>80.7</td></tr><tr><td>VISReg</td><td>ViT-B/16</td><td>100</td><td>73.3</td><td>53.1</td><td>58.0</td><td>94.3</td><td>79.6</td><td>87.9</td><td>79.1</td><td>85.5</td><td>76.4</td></tr><tr><td>λ-JEPA</td><td>ViT-B/16</td><td>100</td><td>71.5</td><td>59.8</td><td>71.1</td><td>96.7</td><td>84.9</td><td>92.2</td><td>79.4</td><td>91.7</td><td>80.9</td></tr><tr><td>LeJEPA</td><td>ViT-B/16</td><td>300</td><td>73.7</td><td>52.0</td><td>55.0</td><td>93.5</td><td>77.6</td><td>87.7</td><td>80.5</td><td>81.8</td><td>75.2</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>300</td><td>71.8</td><td>59.1</td><td>63.8</td><td>96.0</td><td>83.4</td><td>90.0</td><td>80.5</td><td>89.9</td><td>79.3</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>300</td><td>74.5</td><td>62.0</td><td>68.8</td><td>97.4</td><td>86.2</td><td>93.0</td><td>84.2</td><td>92.2</td><td>82.3</td></tr><tr><td> $\mathbf { M o C o } \mathbf { v } 3 ^ { \ddagger }$ </td><td>ViT-B/16</td><td>300</td><td>73.7</td><td>57.9</td><td>67.5</td><td>96.9</td><td>85.2</td><td>91.5</td><td>81.8</td><td>89.8</td><td>80.5</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>400</td><td>74.3</td><td>63.6</td><td>73.9</td><td>96.5</td><td>85.0</td><td>94.6</td><td>83.1</td><td>93.6</td><td>83.1</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>400</td><td>74.1</td><td>63.5</td><td>73.8</td><td>97.1</td><td>85.9</td><td>93.7</td><td>84.2</td><td>93.6</td><td>83.2</td></tr><tr><td>VISReg</td><td>ViT-B/16</td><td>400</td><td>75.7</td><td>57.1</td><td>64.8</td><td>94.6</td><td>78.8</td><td>90.4</td><td>82.9</td><td>88.3</td><td>79.1</td></tr><tr><td>λ-JEPA</td><td>ViT-B/16</td><td>400</td><td>73.2</td><td>61.1</td><td>74.0</td><td>96.9</td><td>85.4</td><td>92.0</td><td>81.0</td><td>92.2</td><td>82.0</td></tr></table>

representation statistics  
downstream evaluation  
![](images/a203d6d1312bdbbb363c05d24ee3592abf34bcf666608af0de981388220f9f49.jpg)  
Figure 7: Effect of applying SACReg to the backbones across SSL objectives on ImageNet-100. We report representation statistics at the backbone h and projection space z, together with downstream linear probe and kNN accuracy. Open circles denote the baseline method and green circles the same method with SACReg added at the backbone. Each point is the mean over three seeds.

Table 5 shows that the added SIGReg substantially improves several Gaussianity statistics at the backbone. Negative-pair cosine drops from 0.58 to 0.01, the diagonal Gaussian KL from 0.96 to 0.07, and the Epps–Pulley statistic also decreases. However, the covariance spectrum remains concentrated: RankMe/d changes only from 0.07 to 0.09, while the full-covariance KL remains high $( 3 . 8 9  3 . 0 3 )$ . At the same time, positive-pair cosine decreases and downstream performance worsens by 1.4 linear-probe points and 4.3 kNN points. Thus, the gains from SACReg are not explained simply by moving LeJEPA’s existing anti-collapse constraint to the retained representation. In this comparison, directly controlling the covariance spectrum is important: SIGReg substantially improves its Gaussianity diagnostics but leaves the backbone spectrum concentrated, whereas $\mathcal { \bar { R } } _ { \mathrm { S A C } }$ acts explicitly on that spectrum through its log-determinant term.

![](images/d70827d21b6500aa17e5f46f72244a7b3faff7ea0317061816600e3ee273921b.jpg)

Figure 8: Effect of SACReg on view sensitivity and class-center separability. Left: mean class-level changes for six SSL methods on ImageNet-100, with arrows from the baseline representation to the same method with SACReg added at the backbone. Right: class-wise changes for λ-JEPA; 96 of 100 classes improve in class-center separability.
<table><tr><td>Readout at encoder output (h)</td><td>LeJEPA</td><td>+ SIGReg at h</td></tr><tr><td>Positive-pair cosine</td><td>0.96</td><td>0.73</td></tr><tr><td>Negative-pair cosine</td><td>0.58</td><td>0.01</td></tr><tr><td>KL to  $\mathcal { N } ( 0 , 1 )$  (diagonal)</td><td>0.96</td><td>0.07</td></tr><tr><td>KL to N(0, I) (full covariance)</td><td>3.89</td><td>3.03</td></tr><tr><td>Epps-Pulley</td><td>652</td><td>504</td></tr><tr><td>RankMe/d</td><td>0.07</td><td>0.09</td></tr><tr><td>Linear probe (%)</td><td>60.4</td><td>58.9</td></tr><tr><td>kNN, k = 200 (%)</td><td>52.4</td><td>48.1</td></tr></table>

Table 5: Adding SIGReg directly to the LeJEPA encoder output on ImageNet-100. The original projector losses are unchanged. All representation statistics are measured at the 512-dimensional encoder output.

## E AUGMENTATION THICKNESS: DEFINITION AND INTERPRETATION

This section gives the formal motivation for the augmentation-thickness quantities used in Section 4.4. In particular, we show that augmentation thickness has two opposing effects: too little thickness can suppress meaningful augmentation-dependent variation, whereas too much allows within-image variation to dominate the separation between images.

## E.1 DEFINITION

Let Q denote a source image and let h be its representation under a randomly sampled augmented view. Define the image center

$$
{ \bf m } _ { h } ( Q ) = \mathbb { E } [ { \bf h } \mid Q ]\tag{39}
$$

and the residual view-dependent component

$$
\pmb { \xi } _ { h } = \mathbf h - \mathbf m _ { h } ( Q ) .\tag{40}
$$

By the law of total covariance,

$$
\mathbf { C } _ { h } = \operatorname { C o v } ( \mathbf { h } ) = \mathbf { B } _ { h } + \mathbf { A } _ { h } , \qquad \mathbf { B } _ { h } = \operatorname { C o v } ( \mathbf { m } _ { h } ( Q ) ) , \qquad \mathbf { A } _ { h } = \mathbb { E } [ \operatorname { C o v } ( \mathbf { h } \mid Q ) ] .\tag{41}
$$

Thus, $\mathbf { B } _ { h }$ measures variation between image centers with the rank $r _ { h } \ = \ \mathrm { r a n k } ( { \bf B } _ { h } )$ , while ${ \bf A } _ { h }$ measures variation across augmented views of the same image.

The same amount of within-image variation can have very different consequences depending on the scale of the between-image variation. We therefore normalize ${ \bf A } _ { h }$ by $\mathbf { B } _ { h }$ and define

$$
\begin{array} { r } { \Theta _ { h } = \mathbf { B } _ { h } ^ { \dagger / 2 } \mathbf { A } _ { h } \mathbf { B } _ { h } ^ { \dagger / 2 } . } \end{array}\tag{42}
$$

We call $\Theta _ { h }$ the augmentation thickness of the representation. Along any direction a with nonzero between-image variance,

$$
\theta _ { h } ( \mathbf { a } ) = \frac { \mathbf { a } ^ { \top } \mathbf { A } _ { h } \mathbf { a } } { \mathbf { a } ^ { \top } \mathbf { B } _ { h } \mathbf { a } }\tag{43}
$$

is the ratio between within-image and between-image variation in that direction.

We next formalize why augmentation thickness should not simply be minimized. Let $y = y ( Q )$ be a centered image-level target with $\| y \| _ { L _ { 2 } } \leq 1$ , and define

$$
E _ { A } ( y ) = \operatorname* { m i n } _ { f } \mathbb { E } \Big [ ( y ( Q ) - f ( \mathbf { v } ) ) ^ { 2 } \Big ] ,\tag{44}
$$

where $\mathbf { v }$ is a single augmented view of $Q .$ . This is the irreducible error of predicting y from one augmented view.

Let $\mathcal { M } _ { h }$ denote the set of affine functions of the image centers and define

$$
d _ { h } ( y ) = \operatorname* { m i n } _ { g \in \mathcal { M } _ { h } } \| y - g \| _ { L { 2 } } .\tag{45}
$$

Thus, $d _ { h } ( y ) ^ { 2 }$ is the error of the best affine prediction of y from the image centers.

Theorem E.1 (Two sides of augmentation thickness). Restrict to the support of $\mathbf { B } _ { h }$ and work in centered, whitened coordinates, so that

$$
\mathbb { E } [ \mathbf { m } _ { h } ( Q ) ] = \mathbf { 0 } , \qquad \mathrm { C o v } ( \mathbf { m } _ { h } ( Q ) ) = \mathbf { I } _ { r _ { h } } , \qquad \mathrm { C o v } ( \xi _ { h } ) = \Theta _ { h } .\tag{46}
$$

Then

$$
d _ { h } ( y ) \geq \left( \sqrt { E _ { A } ( y ) } - \sqrt { \| \Theta _ { h } \| _ { \mathrm { o p } } } \right) _ { + } .\tag{47}
$$

Moreover, let

$$
g ( Q ) = \mathbf { a } ^ { \top } \mathbf { m } _ { h } ( Q )\tag{48}
$$

be the best linear prediction of y from the image centers. Then the best linear prediction error from a single-view representation h is

$$
\operatorname* { i n f } _ { \mathbf { u } } \mathbb { E } \big [ ( y - \mathbf { u } ^ { \top } \mathbf { h } ) ^ { 2 } \big ] = d _ { h } ( y ) ^ { 2 } + \mathbf { a } ^ { \top } \Theta _ { h } ( \mathbf { I } _ { r _ { h } } + \Theta _ { h } ) ^ { - 1 } \mathbf { a } .\tag{49}
$$

Forfixed image-center organization andfixed task direction a, the second term is monotone increas ing in $\Theta _ { h }$ in the positive-semidefinite order.

The two statements capture complementary failure modes. Equation 47 says that if a target cannot be recovered perfectly from an augmented view, then a representation with very little augmentation thickness cannot nevertheless organize its image centers arbitrarily well for that target. Some augmentation-dependent variation may therefore need to remain in the representation. In the opposite direction, Eq. 49 shows that, once the image-center organization is fixed, additional withinimage variation makes prediction from a single view harder. Augmentation thickness therefore describes a trade-off rather than a quantity that should be uniformly minimized.

The quantity

$$
\mathbf { a } ^ { \top } \Theta _ { h } ( \mathbf { I } _ { r _ { h } } + \Theta _ { h } ) ^ { - 1 } \mathbf { a }\tag{50}
$$

is the view-sensitivity term reported in Fig. 8. We use this quantity in Fig. 8 rather than the raw directional thickness $\theta _ { h } ( \mathbf { a } )$ because it captures the contribution of augmentation variation after optimizing the linear prediction. It isolates the part of the optimal single-view linear prediction error attributable to variation across augmentations, separately from the image-center organization term $d _ { h } ( y ) ^ { 2 }$

## E.2 PROOF OF THEOREM E.1

Throughout the proof we work on the support of $\mathbf { B } _ { h }$ in centered, whitened coordinates and reuse h, ${ \bf m } _ { h }$ , and $\xi _ { h }$ for the transformed variables. Hence

$$
\mathbb { E } [ \mathbf { m } _ { h } ( Q ) ] = \mathbf { 0 } , \qquad \operatorname { C o v } ( \mathbf { m } _ { h } ( Q ) ) = \mathbf { I } _ { r _ { h } } , \qquad \mathbb { E } [ \pmb { \xi } _ { h } \mid Q ] = \mathbf { 0 } , \qquad \operatorname { C o v } ( \pmb { \xi } _ { h } ) = \Theta _ { h } .\tag{51}
$$

Too little thickness. Consider any linear function of the image centers,

$$
g ( Q ) = \mathbf { a } ^ { \top } \mathbf { m } _ { h } ( Q ) .\tag{52}
$$

Since $\mathbf { a } ^ { \top } \mathbf { h }$ is computable from a single augmented view, it is one admissible predictor of $g .$ Therefore,

$$
E _ { A } ( g ) \leq \mathbb { E } \Big [ \big ( \mathbf { a } ^ { \top } \mathbf { m } _ { h } ( Q ) - \mathbf { a } ^ { \top } \mathbf { h } \big ) ^ { 2 } \Big ]\tag{53}
$$

$$
\mathbf { \theta } = \mathbb { E } \big [ ( \mathbf { a } ^ { \top } \pmb { \xi } _ { h } ) ^ { 2 } \big ]\tag{54}
$$

$$
{ \mathbf { \theta } } = { \mathbf { a } } ^ { \top } \mathbf { \Theta } \Theta _ { h } { \mathbf { a } } .\tag{55}
$$

Now let $g$ be the best linear prediction of $y$ from the image centers. Since the image-center covariance is whitened, we may write

$$
g ( Q ) = \mathbf { a } _ { y } ^ { \top } \mathbf { m } _ { h } ( Q ) .\tag{56}
$$

Orthogonal projection in $L _ { 2 }$ gives

$$
\| \mathbf { a } _ { y } \| = \| g \| _ { L _ { 2 } } \leq \| y \| _ { L _ { 2 } } \leq 1 .\tag{57}
$$

Using the triangle inequality,

$$
\sqrt { E _ { A } ( y ) } \leq \| y - g \| _ { L _ { 2 } } + \sqrt { E _ { A } ( g ) }\tag{58}
$$

$$
\leq d _ { h } ( y ) + \sqrt { \mathbf { a } _ { y } ^ { \top } \Theta _ { h } \mathbf { a } _ { y } }\tag{59}
$$

$$
\leq d _ { h } ( y ) + \sqrt { \| \Theta _ { h } \| _ { \mathrm { o p } } } ,\tag{60}
$$

where the last inequality uses $\| \mathbf { a } _ { y } \| \leq 1$ . Rearranging yields

$$
d _ { h } ( y ) \geq \left( \sqrt { E _ { A } ( y ) } - \sqrt { \| \Theta _ { h } \| _ { \mathrm { o p } } } \right) _ { + } ,\tag{61}
$$

which proves Eq. 47.

Too much thickness. Let

$$
g ( Q ) = \mathbf { a } ^ { \top } \mathbf { m } _ { h } ( Q )\tag{62}
$$

be the best linear prediction of $y$ from the image centers, and write

$$
y = g + r .\tag{63}
$$

By the orthogonality property of linear regression,

$$
\mathbb { E } [ r \mathbf { m } _ { h } ( Q ) ] = \mathbf { 0 } , \qquad \mathbb { E } [ r ^ { 2 } ] = d _ { h } ( y ) ^ { 2 } .\tag{64}
$$

Moreover, since $\mathbb { E } [ \pmb { \xi } _ { h } \ | \ Q ] = \mathbf { 0 }$ and r is a function of $Q _ { \ l }$

$$
\operatorname { \mathbb { E } } [ r \pmb { \xi } _ { h } ] = \operatorname { \mathbb { E } } [ r \operatorname { \mathbb { E } } [ \pmb { \xi } _ { h } \mid Q ] ] = \mathbf { 0 } .\tag{65}
$$

Hence $r$ is uncorrelated with

$$
\mathbf { h } = \mathbf { m } _ { h } ( Q ) + \pmb { \xi } _ { h }\tag{66}
$$

and contributes exactly $d _ { h } ( y ) ^ { 2 }$ to the optimal affine prediction error from h.

For the component $g = \mathbf { a } ^ { \top } \mathbf { m } _ { h } ( Q )$

$$
\operatorname { C o v } ( \mathbf { h } ) = \mathbf { I } _ { r _ { h } } + \Theta _ { h } , \qquad \operatorname { C o v } ( \mathbf { h } , \mathbf { m } _ { h } ( Q ) ) = \mathbf { I } _ { r _ { h } } .\tag{67}
$$

The best linear prediction error of $g$ from h is therefore

$$
\mathbb { E } [ g ^ { 2 } ] - \mathrm { C o v } ( g , \mathbf { h } ) \mathrm { C o v } ( \mathbf { h } ) ^ { - 1 } \mathrm { C o v } ( \mathbf { h } , g )\tag{68}
$$

$$
\mathbf { \Omega } = \mathbf { a } ^ { \top } \left[ \mathbf { I } _ { r _ { h } } - ( \mathbf { I } _ { r _ { h } } + \boldsymbol { \Theta } _ { h } ) ^ { - 1 } \right] \mathbf { a }\tag{69}
$$

$$
\mathbf { \Theta } = \mathbf { a } ^ { \top } \Theta _ { h } ( \mathbf { I } _ { r _ { h } } + \Theta _ { h } ) ^ { - 1 } \mathbf { a } .\tag{70}
$$

Adding the residual contribution gives

$$
\operatorname* { i n f } _ { \mathbf { u } } \mathbb { E } \big [ ( y - \mathbf { u } ^ { \top } \mathbf { h } ) ^ { 2 } \big ] = d _ { h } ( y ) ^ { 2 } + \mathbf { a } ^ { \top } \Theta _ { h } ( \mathbf { I } _ { r _ { h } } + \Theta _ { h } ) ^ { - 1 } \mathbf { a } ,\tag{71}
$$

which proves Eq. 49.

Finally,

$$
\begin{array} { r } { \Theta _ { h } ( \mathbf { I } _ { r _ { h } } + \Theta _ { h } ) ^ { - 1 } = \mathbf { I } _ { r _ { h } } - ( \mathbf { I } _ { r _ { h } } + \Theta _ { h } ) ^ { - 1 } . } \end{array}\tag{72}
$$

If $\Theta _ { h } ^ { \prime } \succeq \Theta _ { h }$ , then

$$
( \mathbf { I } _ { r _ { h } } + \mathbf { \Theta } _ { h } ^ { \prime } ) ^ { - 1 } \preceq ( \mathbf { I } _ { r _ { h } } + \mathbf { \Theta } _ { h } ) ^ { - 1 } ,\tag{73}
$$

and therefore

$$
\begin{array} { r } { \boldsymbol { \Theta } _ { h } ^ { \prime } ( \mathbf { I } _ { r _ { h } } + \boldsymbol { \Theta } _ { h } ^ { \prime } ) ^ { - 1 } \succeq \boldsymbol { \Theta } _ { h } ( \mathbf { I } _ { r _ { h } } + \boldsymbol { \Theta } _ { h } ) ^ { - 1 } . } \end{array}\tag{74}
$$

Thus, for fixed a, increasing augmentation thickness can only increase the view-sensitivity term.   
This completes the proof.