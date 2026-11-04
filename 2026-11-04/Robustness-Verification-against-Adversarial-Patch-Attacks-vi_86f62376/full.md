# Robustness Verification against Adversarial Patch Attacks via Group-Lasso Semidefinite Relaxation

Yukina Nakagawa

Institute of Science Tokyo

2-12-1 Ookayama, Meguro-ku, Tokyo 152-8550, Japan

nakagawa.y.367d@m.isct.ac.jp

Ken Kobayashi

Institute of Science Tokyo

2-12-1 Ookayama, Meguro-ku, Tokyo 152-8550, Japan kobayashi.k@iee.eng.isct.ac.jp

Takami Sato

Keio University

2-15-45 Mita, Minato-ku, Tokyo 108-8345, Japan

takami.sato@yos.elec.keio.ac.jp

Kazuhide Nakata

Institute of Science Tokyo

2-12-1 Ookayama, Meguro-ku, Tokyo 152-8550, Japan

nakata.k@iee.eng.isct.ac.jp

## Abstract

Robustness verification is essential for ensuring the safety and security of neural networks, but current methods focus heavily on $L _ { p }$ norm-ball attacks, which do not adequately model typical real-world threats that involve physical perturbations, such as adversarial patch attacks. A naive approach requires computationally prohibitive enumeration of all possible patch positions. To address this challenge, we propose a novel neural network verification method specialized for patch-attack threats. Instead of enumerating patch locations, we first model the patch attack with group-wise sparsity and formulate verification as a mixed-integer semidefinite program (MISDP) by applying semidefinite program (SDP) relaxation. To further relax the problem to a continuous optimization problem, we propose three relaxations, including Group-Lasso relaxation, that remove the integer variables while preserving the sparse, group-wise structure of the original attack, enabling substantially more efficient verification. Our empirical analysis demonstrates that the proposed relaxations require less than 3% of the computation time of the original MISDP, while Group-Lasso relaxation incurs only a 0.34% average increase in the objective value over the original formulation. Furthermore, Group-Lasso achieves a verification rate of 91%, compared with ≤70% for prior methods.

## 1 Introduction

Deep Neural Networks (DNNs) have achieved remarkable success in many fields and have been widely deployed in safety-critical vision applications such as autonomous driving [4, 24] and medical image analysis [11, 33]. However, DNNs are widely known to be vulnerable to adversarial attacks [8, 14, 29] that intentionally modify an input with subtle malicious perturbations to induce an incorrect prediction. The vulnerability to adversarial attacks has raised critical concerns about their reliability and security, spurring extensive research into DNN verification [10, 15, 30, 32].

DNN verification has traditionally focused on $L _ { p }$ normball attacks. For an input image $\pmb { x } ^ { * }$ , the verifier considers all possible perturbed images x under a perturbation radius $\varepsilon , \mathrm { i . e . , } | | x - x ^ { * } | | _ { p } \leq \varepsilon .$ , and determines whether the model prediction remains unchanged. This verification problem is generally solved as an optimization problem. While the $L _ { p }$ norm-ball assumption can cover broader types of perturbations, it does not adequately represent the threat model of practical physical-world attacks [6, 28] in which an attacker modifies only a localized region, for example by attaching a sticker or placing a patterned object on part of an image. A naive approach to patch attacks enumerates possible patch locations and solves a verification problem for each candidate location, but the computational cost grows exponentially with the number of possible attack locations.

To address this limitation, we propose a novel DNN verification method specialized for patch attacks. Instead of enumerating patch locations, we model the patch attack with group-wise sparsity [12]. We then incorporate the patch position as a variable within a single mixed-integer program (MIP), which is an exact model but is not practical due to its high combinatorial complexity. We thus relax the problem with an SDP relaxation [16] and three proposed relaxations for the group-wise patch attack. We propose a full-group naive relaxation that permits perturbations in every group, an L<sub>2</sub>-norm relaxation that replaces the group-wise constraints with a global $L _ { 2 }$ constraint, and a Group-Lasso relaxation that constrains the sum of the $L _ { 2 }$ norms of the group-wise perturbations. All three relaxed problems provide a valid guarantee for the original attack model.

We evaluate the proposed formulations from both computational and verification perspectives. We found that all three relaxations require less than 3% of the computation time of the original problem in the evaluated settings. Group-Lasso relaxation achieves an average objective increase of only 0.34% in the evaluated setting, substantially outperforming the Full-Group and $L _ { 2 } .$ -norm relaxations. These results indicate that preserving the group structure of the attack is important for obtaining a tight relaxation. Group-Lasso relaxation also demonstrated high verification capability; while existing methods achieve verification rates of ≤70%, it was able to verify 91% of the samples on the adversarially trained model. We conduct qualitative analysis and confirm that the perturbations considered in Group-Lasso relaxation are locally grouped and concentrated around the digit character, as shown in Fig. 1.

![](images/0a4af0858fe74244f079ef2f76149355848f68791aa93fd0b22070ce3ba7c815.jpg)  
Figure 1: Adversarial perturbations generated during verification. Red and blue mean positive- and negative-direction perturbations, respectively. Compared to existing baselines (MILP and IBP) that spread perturbations across the entire image, our proposed Group-Lasso relaxation restricts perturbations to localized foreground regions, closely matching the localized patch attack threat model.

Our main contributions are summarized as follows:

• We are the first to formulate DNN verification specialized for patch attacks with group-wise sparsity modeling. We first formulate it as an exact MIP problem and design an effective relaxation enabling efficient verification.

• We design three relaxations for the group-wise modeling with theoretical soundness guarantees. In particular, Group-Lasso relaxation explicitly preserves the groupsparse structure and provides a substantially tighter approximation than the simpler relaxations.

• We demonstrate the high efficiency and effectiveness of our proposed methods, which require at most 3% of the computation time of the original MISDP. Group-Lasso relaxation differs from the exact formulation by only 0.34% in objective value and achieves a verification rate of up to 91%, outperforming existing methods, whose verification rates are at most 70%.

## 2 Related Work

## 2.1 Adversarial Attacks

Deep Neural Networks (DNNs) are widely known to be vulnerable to adversarial attacks [8, 14, 29], where carefully crafted perturbations alter model predictions while remaining imperceptible to humans. To bound perturbation magnitude, traditional attacks are typically formulated within an $L _ { p }$ -norm ball, such as $L _ { \infty }$ [21] or $L _ { 2 }$ [8] balls.

Beyond global perturbation, recent work has demonstrated that effective attacks can be realized by perturbing only a localized region of the input image [19, 27, 28]. We refer to this threat model as patch attacks. Attacks based on $L _ { p }$ norm balls perturb the entire image, which would require physically perturbing everything within the camera’s field of view, an unrealistic requirement in the physical world. An attacker would therefore need to compromise the system itself. In contrast, patch attacks allow an adversary to introduce localized physical perturbations by placing patches within the camera’s field of view [6, 26] or attaching stickers to target objects [28], without compromising the system. Owing to this practicality in real-world scenarios, patch attacks have drawn significant attention.

This emerging threat model has motivated the development of certified defenses against patch attacks [9, 17, 22, 34, 35]. Chiang et al. [9] proposed a certified defense that enumerates possible patch locations and performs verification for each location. Levine et al. [17] developed a deterministic certification approach that restricts the set of possible patch regions and verifies the resulting cases. BagCert [22] avoids explicit enumeration of all patch locations by partitioning the input image into multiple regions and performing certified defense for CNNs. However, existing approaches focus on developing certified defenses, not verification for all possible attacks. None of the prior studies has studied DNN verification methods specialized for patch attacks. This gap motivates our study.

## 2.2 DNN Verification Problem

DNN verification aims to determine whether a model preserves its prediction for all inputs within a specified perturbation set. Given an input image x<sup>∗</sup> with correct class label $y ^ { * }$ , the model $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { C }$ is certified as robust against the attack of interest if its predicted class remains unchanged throughout the perturbation set $\mathcal { X }$ centered at $\pmb { x } ^ { * }$ . We formulate the DNN verification problem as the following decision problem:

$$
\forall { \pmb x } \in { \mathcal X } , \ \underset { i \in \{ 1 , 2 , . . . , C \} } { \arg \operatorname* { m a x } } \ f ( { \pmb x } ) = \underset { i \in \{ 1 , 2 , . . . , C \} } { \arg \operatorname* { m a x } } \ f ( { \pmb x } ^ { * } ) = { \pmb y } ^ { * } .\tag{1}
$$

For an $L _ { p }$ norm ball, the perturbation set can be defined as follows:

$$
\begin{array} { r } { \mathcal { X } _ { L _ { p } } = \left\{ { \pmb x } | | { \pmb x } - { \pmb x } ^ { * } | | _ { p } \leq { \pmb \varepsilon } \right\} . } \end{array}\tag{2}
$$

DNN verification has been studied through both complete and incomplete verification approaches. Complete verification [2, 5, 10, 15, 20, 30] seeks to solve Problem Eq. (1) exactly, typically using mixed-integer programming (MIP) [30, 32] or satisfiability modulo theories (SMT) [10, 15]. Although these methods can provide exact verification, their computational cost is generally unaffordable for practical use.

Incomplete verification instead replaces the original verification problem with a relaxed problem that is computationally more tractable. The relaxed problem is constructed as a necessary condition for the original verification problem: consequently, if the relaxed problem is verified, then the original DNN verification problem is also verified. However, this relaxation can introduce false negatives, meaning that a model may be certifiably robust under complete verification even when the incomplete verifier fails to establish robustness.

## 3 Methodology

We first formalize the patch-attack threat model using groupwise sparsity (Sec. 3.1) and derive an exact mixed-integer program that jointly optimizes the perturbation and patch location (Sec. 3.2). We then apply an SDP relaxation to obtain a more tractable MISDP (Sec. 3.3), and finally propose three relaxations that remove the remaining combinatorial structure (Sec. 3.4).

## 3.1 Patch Attack Modeling

We model the patch attack with group-wise sparsity [12]. Let the pixel index set of an input image be $\mathcal { P } = \{ 1 , 2 , \ldots , n _ { 0 } \}$ We partition $\mathcal { P }$ into m non-overlapping groups:

$$
{ \mathcal { G } } = \{ 1 , 2 , \ldots , m \} ,\tag{3a}
$$

$$
\begin{array} { r } { \mathcal { P } _ { i } \cap \mathcal { P } _ { j } = \emptyset \quad ( i , j \in \mathcal { G } , i \neq j ) , } \end{array}\tag{3b}
$$

$$
\begin{array} { r } { \bigcup \mathcal { P } _ { g } = \mathcal { P } . } \end{array}
$$

$$
g \in \mathcal { G }\tag{3c}
$$

where $\mathcal { P } _ { g }$ denotes the set of pixels belonging to the group g. Eq. (3b) and Eq. (3c) respectively guarantee the non-overlap and the coverage of the partition.

The attacker can select at most k groups to perturb. The verifier cannot know which groups are selected. The perturbation applied to each selected group is bounded by ε. Let $\pmb { G } \in \{ 0 , 1 \} ^ { m }$ denote the group-selection vector, where $G ^ { ( g ) } = 1$ indicates that group g is attacked. For an original input $\pmb { x } _ { 0 } ^ { * }$ , the corresponding perturbation set is defined as

$$
\| \boldsymbol { x } _ { 0 } ^ { ( g ) } - \boldsymbol { x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \le \varepsilon G ^ { ( g ) } \quad ( g \in \mathcal { G } ) ,
$$

$$
\mathbf { 1 } ^ { \top } \pmb { G } = \mathbf { k } ,\tag{4a}
$$

$$
\pmb { G } \in \{ 0 , 1 \} ^ { m } .\tag{4b}
$$

(4c)

The vectors $\pmb { x } _ { 0 } ^ { ( g ) }$ and $\pmb { x } _ { 0 } ^ { * ( g ) }$ denote the sub-vectors of the perturbed and original images restricted to the index set $\mathcal { P } _ { g }$ of group g. The first constraint allows a perturbation of radius at most ε only for an attacked group, while the second constraint restricts the attacker to at most k groups. The binary vector $\pmb { G } \in \{ 0 , 1 \} ^ { m }$ indicates which groups are attacked, where $G ^ { ( g ) } = 1$ if group g is attacked and $G ^ { ( g ) } = 0$ otherwise.

## 3.2 Patch Attack Complete Verification Modeling

Given the perturbation set defined in Eq. (4), robustness verification can be formulated as an optimization problem that jointly searches for the adversarial perturbation and its attack location. For a given target class, we maximize the difference between the target and true class scores:

$$
\operatorname* { m a x } _ { \{ \pmb { x } _ { i } \} _ { i = 0 } ^ { L } , \pmb { G } } \pmb { x } _ { L } [ j ] - \pmb { x } _ { L } [ j ^ { \ast } ]\tag{5a}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { { } \pmb { x } _ { L } = f ( \pmb { x } _ { 0 } ) , } \end{array}\tag{5b}
$$

$$
\| \pmb { x } _ { 0 } ^ { ( g ) } - \pmb { x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \le \varepsilon G ^ { ( g ) } , \quad \forall g \in \mathcal { G } ,\tag{5c}
$$

$$
\mathbf { 1 } ^ { \top } \pmb { G } = \ d _ { k } ,\tag{5d}
$$

$$
\pmb { G } \in \{ 0 , 1 \} ^ { m } ,\tag{5e}
$$

where $j ^ { * }$ and $j \left( j \neq j ^ { * } \right)$ denote the true and target labels, and $\pmb { x } _ { L } [ j ]$ represents the j-th entry of the vector $\pmb { x } _ { L } .$ . x denotes the perturbed input and G determines the attacked groups. If the optimal value is non-positive, the model is verified to be robust against the specified target class; otherwise, the existence of a patch attack cannot be ruled out.

For an L-layer ReLU network, Eq. (5) can be formulated as an extension of the existing mixed-integer program (MIP)- based complete verification [15]:

$$
\operatorname* { m a x } _ { \{ \pmb { x } _ { i } \} _ { i = 0 } ^ { L } , \pmb { G } } { \pmb { c } } ^ { \top } { \pmb { x } } _ { L } + c _ { 0 }\tag{6a}
$$

$$
\mathrm { s . t . } \quad \pmb { x } _ { i + 1 } \geq \mathbf { 0 } , \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) ,\tag{6b}
$$

$$
\pmb { x } _ { i + 1 } \geq W _ { i } \pmb { x } _ { i } + \pmb { b } _ { i } \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) ,\tag{6c}
$$

$$
\pmb { x } _ { i + 1 } \leq ( 1 - \pmb { z } _ { i } ) M + W _ { i } \pmb { x } _ { i } + \pmb { b } _ { i } \ ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) ,\tag{6d}
$$

$$
\pmb { x } _ { i + 1 } \leq M \pmb { z } _ { i } \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) ,\tag{6e}
$$

$$
\begin{array} { r } { 0 \leq x _ { 0 } \leq 1 , } \end{array}\tag{6f}
$$

$$
\begin{array} { r } { z _ { i } \in \{ 0 , 1 \} ^ { n _ { i } } \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) , } \end{array}\tag{6g}
$$

$$
\| \pmb { x } _ { 0 } ^ { ( g ) } - \pmb { x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \le \varepsilon G ^ { ( g ) } \quad ( g \in \mathcal { G } ) ,\tag{6h}
$$

$$
\mathbf { 1 } ^ { \top } \pmb { G } = k , \quad \pmb { G } \in \{ 0 , 1 \} ^ { m } .\tag{6i}
$$

where $\pmb { c } ^ { \top } = W _ { L } ( j , \ ; ) - W _ { L } ( j ^ { * } , \ ; )$ and $c _ { 0 } = { b } _ { L } [ j ] - { b } _ { L } [ j ^ { * } ]$ $\mathbb { I } _ { [ 0 , L - 1 ] } = \{ 0 , 1 , \dots , L - 1 \}$ , and M represents a sufficiently large constant derived from the bounds on the pre-activation values. The first four constraints encode the ReLU activations using binary variables $z _ { i } ,$ while Eq. (6f) restricts the image pixels to their valid range. The remaining constraints define the group-wise patch attack. Since Eq. (6) introduces a binary variable for every ReLU unit, its computational cost grows rapidly with the network size, and directly applying this approach to large-scale neural network models is impractical.

## 3.3 Incomplete Verification via SDP Relaxation

We introduce our incomplete verification for the patch attack in this section. Firstly, we replace the ReLU representations, Eq. (6d) and Eq. (6e), with quadratic constraints to enable SDP relaxation [25]. By retaining the binary group-selection variable ${ \pmb G } ,$ the resulting mixed-integer semidefinite program (MISDP) preserves the exact group-wise patch attack model. Compared to Eq. (6), this MISDP eliminates binary variables for individual neurons, drastically reducing the total number of binary decision variables from $\textstyle \sum _ { i = 0 } ^ { L - 1 } n _ { i }$ to the number of groups m.

Following [25], we introduce a vector

$$
\pmb { \tilde { x } } = \bigl ( 1 , \pmb { x } _ { 0 } ^ { \top } , \pmb { x } _ { 1 } ^ { \top } , \ldots , \pmb { x } _ { L } ^ { \top } \bigr ) ^ { \top } \in \mathbb { R } ^ { 1 + \sum _ { i = 0 } ^ { L } n _ { i } }\tag{7}
$$

that concatenates 1 and the activation vectors of all layers, and a symmetric matrix

$$
P = \tilde { \pmb { x } } \tilde { \pmb { x } } ^ { \top } \in \mathcal { S } ^ { 1 + \sum _ { i = 0 } ^ { L } n _ { i } } .\tag{8}
$$

Then, Eq. (5) is equivalently reformulated with respect to the variable $P$ with a rank constraint rank(P) = 1. By removing

the rank constraint, we obtain the following MISDP:

$$
\operatorname* { m a x } _ { P , G } \quad c ^ { \top } P [ \pmb { x } _ { L } ] + c _ { 0 }\tag{9a}
$$

$$
\mathrm { s . t . } \quad P [ { \pmb x } _ { i + 1 } ] \geq \mathbf { 0 } \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) ,\tag{9b}
$$

$$
\begin{array} { r } { P [ \pmb { x } _ { i + 1 } ] \geq W _ { i } P [ \pmb { x } _ { i } ] + \pmb { b } _ { i } \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) , } \end{array}\tag{9c}
$$

$$
\mathrm { d i a g } ( P [ \pmb { x } _ { i + 1 } \pmb { x } _ { i + 1 } ^ { \top } ] - W _ { i } P [ \pmb { x } _ { i } \pmb { x } _ { i + 1 } ^ { \top } ] )
$$

$$
- \pmb { b } _ { i } \odot P [ \pmb { x } _ { i + 1 } ] = \pmb { 0 } \quad ( i \in \mathbb { I } _ { [ 0 , L - 1 ] } ) ,\tag{9d}
$$

$$
\mathrm { d i a g } \big ( P [ { \pmb x } _ { i } { \pmb x } _ { i } ^ { \top } ] \big )
$$

$$
- \left( l _ { i } + u _ { i } \right) \odot P [ { \pmb x } _ { i } ] + l _ { i } \odot \pmb { u } _ { i } \le \mathbf { 0 } \quad ( i \in \mathbb { I } _ { [ 0 , L ] } ) ,\tag{9e}
$$

$$
\begin{array} { r } { \pmb { 0 } \le P [ \pmb { x } _ { 0 } ] \le \mathbf { 1 } , } \end{array}\tag{9f}
$$

$$
\| P [ \pmb { x } _ { 0 } ^ { ( g ) } ] - \pmb { x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \le \varepsilon G ^ { ( g ) } \quad ( g \in \mathcal { G } ) ,\tag{9g}
$$

$$
\mathbf { 1 } ^ { \top } \pmb { G } = k , \quad \pmb { G } \in \{ 0 , 1 \} ^ { m } ,\tag{9h}
$$

$$
P [ 1 ] = 1 , \quad P \succeq 0 ,\tag{9i}
$$

where we use the same indexing notation $P [ \cdot ]$ as in [25], and $\mathbf { \xi } _ { l _ { i } }$ and $\pmb { u } _ { i }$ denote the lower and upper bounds of the pre-activation values of layer $i ,$ obtained by a lightweight method such as bound propagation [31, 36]. In addition, for the group-wise constraints we write $P [ \pmb { x } _ { 0 } ^ { ( g ) } ]$ for the sub-block of $P [ \pmb { x } _ { 0 } ]$ restricted to the index set $\mathcal { P } _ { g }$ of group g.

## 3.4 Proposed Relaxation for Group-Wise Sparsity

The MISDP (9) eliminates the binary variables associated with the ReLU activations, but retains the binary attackselection variable G. This remaining combinatorial structure still limits the scalability of the verification problem. To address these limitations, we further relax the group-wise patch attack by removing the binary variable $\pmb { G } .$ Specifically, we construct a convex perturbation set $\mathcal { \widetilde X }$ satisfying

$$
{ \mathcal { X } } \subseteq { \mathcal { \widetilde { X } } } ,\tag{10}
$$

where $\mathcal { X }$ is the original perturbation set defined in Eq. (4). Replacing $\mathcal { X }$ with $\mathcal { \widetilde X }$ yields a relaxation of the original verification problem. Since the relaxed set contains the original attack set, a successful verification result remains sound for the original threat model. We propose three relaxations with different levels of approximation: Full-Group, L<sub>2</sub>-norm, and Group-Lasso relaxations. Specifically, we replace the constraints Eq. (9g) and Eq. (9h) with the following proposed constraints Eq. (11) - Eq. (13).

## 3.4.1 Full-Group Naive Relaxation

The full-group relaxation simply removes the attackselection constraint and allows a perturbation of radius ε in every group:

$$
\mathcal { X } _ { \mathrm { F u l l } } = \left\{ \pmb { x } _ { 0 } \biggm | \| \pmb { x } _ { 0 } ^ { ( g ) } - \pmb { x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \leq \pmb { \varepsilon } , \quad \forall g \in \mathcal { G } \right\} .\tag{11}
$$

This relaxation corresponds to fixing $G ^ { ( g ) } = 1$ for all groups. Consequently, this completely removes the sparsity constraint, yielding the simplest convex approximation for the original attack set.

## 3.4.2 L<sub>2</sub>-Norm Relaxation

The $L _ { 2 }$ -norm relaxation replaces the group-wise perturbation constraints with a single constraint on the entire image:

$$
\mathcal { X } _ { \mathrm { L } _ { 2 } } = \left\{ { \pmb x } _ { 0 } \vert \Vert { \pmb x } _ { 0 } - { \pmb x } _ { 0 } ^ { * } \Vert _ { 2 } \leq k { \pmb \varepsilon } \right\} .\tag{12}
$$

Although this relaxation method ignores the group structure of the attacks, it offers the advantage of reducing the problem to the standard $L _ { 2 } \mathrm { - n o r m }$ ball formulation; consequently, existing SDP (semidefinite programming)-based verification frameworks for $L _ { p }$ -norm ball attacks can be applied directly.

## 3.4.3 Group-Lasso Relaxation

Group-Lasso relaxation preserves the group structure of the original attack while removing the binary selection variables. Specifically, we replace the individual group-wise constraints with a single group-sparse constraint:

$$
\mathcal { X } _ { \mathrm { G L } } = \left\{ \pmb { x } _ { 0 } \Bigg \vert \sum _ { g \in \mathcal { G } } \Vert \pmb { x } _ { 0 } ^ { ( g ) } - \pmb { x } _ { 0 } ^ { * ( g ) } \Vert _ { 2 } \leq k \pmb { \varepsilon } \right\} .\tag{13}
$$

This constraint bounds the total group-wise perturbation by the maximum perturbation that can be introduced into k groups, while allowing the optimization to determine how the perturbation is distributed across groups. Among the three relaxations, Group-Lasso most directly reflects the spatial sparsity of the original group-wise patch attack.

## 3.4.4 Soundness and Inclusion Relations

The relationships among the original attack set and the proposed convex relaxations are formalized by the following propositions.

Proposition 1 (Soundness). All three relaxed perturbation sets contain the original attack set $\mathcal { X }$ . Specifically, we have thefollowing relationships:

$$
\mathcal { X } \subseteq \mathcal { X } _ { \mathrm { F u l l } } , \quad \mathcal { X } \subseteq \mathcal { X } _ { \mathrm { G L } } , \quad \mathcal { X } \subseteq \mathcal { X } _ { L _ { 2 } } .\tag{14}
$$

Proof. Let $\pmb { x } _ { 0 } \in \mathcal { X }$ be an arbitrary input satisfying Eq. (4). For each group g,

$$
\| \pmb { x } _ { 0 } ^ { ( g ) } - \pmb { x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \le \varepsilon G ^ { ( g ) } \le \varepsilon ,
$$

which immediately gives $\pmb { x } _ { 0 } \in \mathcal { X } _ { \mathrm { F u l l } }$ . Furthermore,

$$
\sum _ { g } \| x _ { 0 } ^ { ( g ) } - x _ { 0 } ^ { * ( g ) } \| _ { 2 } = \sum _ { g : G ^ { ( g ) } = 1 } \| x _ { 0 } ^ { ( g ) } - x _ { 0 } ^ { * ( g ) } \| _ { 2 } \leq k \varepsilon ,
$$

where the last inequality follows from $\mathbf { 1 } ^ { \top } \pmb { G } = k$ . Therefore, $\pmb { x } _ { 0 } \in \mathcal { X } _ { \mathrm { G I } }$ . Finally, because the groups form a nonoverlapping partition of the image,

$$
\| { \pmb x } _ { 0 } - { \pmb x } _ { 0 } ^ { * } \| _ { 2 } \leq \sum _ { g } \| { \pmb x } _ { 0 } ^ { ( g ) } - { \pmb x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \leq k { \pmb \varepsilon } .\tag{15}
$$

Hence, $\pmb { x } _ { 0 } \in \mathcal { X } _ { L _ { 2 } }$

Proposition 2 (Inclusion Relations). Group-Lasso relaxation is contained within the $L _ { 2 }$ -norm relaxation $\begin{array} { r l } {  { ( \mathcal { X } _ { \mathrm { G L } } \subseteq C } } & { { } } \\ { \operatorname { \mathcal { X } } _ { \mathrm { G L } } \subseteq } \end{array}$ $\mathcal { X } _ { L _ { 2 } } )$ . Combined with the soundness properties, the resulting relationships can be summarized as:

$$
\mathcal { X } \subseteq \mathcal { X } _ { \mathrm { G L } } \subseteq \mathcal { X } _ { L _ { 2 } } , \quad \mathcal { X } \subseteq \mathcal { X } _ { \mathrm { F u l l } } .\tag{16}
$$

Proof. We can establish the inclusion relation between Group-Lasso and $L _ { 2 } .$ -norm relaxations by using the triangle inequality. For any $\pmb { x } _ { 0 } \in \mathcal { X } _ { \mathrm { G L } }$ , we have:

$$
\| { \pmb x } _ { 0 } - { \pmb x } _ { 0 } ^ { * } \| _ { 2 } \leq \sum _ { g \in { \mathcal { G } } } \| { \pmb x } _ { 0 } ^ { ( g ) } - { \pmb x } _ { 0 } ^ { * ( g ) } \| _ { 2 } \leq k { \pmb \varepsilon } .\tag{17}
$$

Thus, $\mathcal { X } _ { \mathrm { G L } } \subseteq \mathcal { X } _ { L _ { 2 } }$ . In contrast, no general inclusion relation exists between ${ \mathcal { X } } _ { \mathrm { F u l l } }$ and the other two relaxed sets, and their relative tightness depends on the number of groups m and the number of attacked groups k. □

Thus, Group-Lasso provides a tighter relaxation than the L<sub>2</sub>-norm relaxation while explicitly retaining the group structure of the original attack. After replacing the original perturbation set $\mathcal { X }$ with any of the three convex sets above, the binary variable G is no longer required. The resulting verification problem is therefore a pure SDP with continuous variables only. This removes the combinatorial optimization over attack locations and enables substantially more scalable verification of larger neural networks.

## 4 Numerical Experiments

We evaluate the proposed relaxation methods from three perspectives: computational efficiency, relaxation tightness, and verification performance.

## 4.1 Common Setup

All experiments were conducted on a node with two AMD EPYC 9654 CPUs (96 cores per socket, up to 3707 MHz) and 768 GiB of memory. To solve the MISDP, we used SCIP-SDP [3] with SoPlex [13] to handle the branch-andbound framework required for solving the auxiliary LPs. For our methods, we used MOSEK [23] to solve the relaxed SDPs. The image is partitioned into non-overlapping square groups using a sliding window, where the side length of each square group specifies the group size. When the side length of the image is not evenly divisible by the group size, the remaining pixels are grouped together as a single group.

Table 1: Model settings used for the scalability evaluation
<table><tr><td></td><td>Hidden layer</td><td>P matrix size</td><td>Accuracy</td></tr><tr><td>Tiny</td><td>5 nodes × 2 layers</td><td> $4 6 \times 4 6$ </td><td>44.37%</td></tr><tr><td>Small</td><td>10 nodes × 2 layers</td><td> $5 6 \times 5 6$ </td><td>51.81%</td></tr><tr><td>Medium</td><td>15 nodes × 2 layers</td><td> $6 6 \times 6 6$ </td><td>52.74%</td></tr><tr><td>Large</td><td>20 nodes × 2 layers</td><td> $7 6 \times 7 6$ </td><td>53.34%</td></tr></table>

Table 2: Execution time comparison between MISDP (9) and three proposed relaxations, across different network sizes. All proposed relaxations achieve <3% of MISDP runtime.
<table><tr><td>Method</td><td>Tiny [s]</td><td>Small [s]</td><td>Medium [s]</td><td>Large [s]</td></tr><tr><td>MISDP (9)</td><td>6.64</td><td>22.95</td><td>77.88</td><td>115.84</td></tr><tr><td>Full-Group</td><td>0.161</td><td>0.11</td><td>0.057</td><td>0.080</td></tr><tr><td> $L _ { 2 }$  norm</td><td>0.032</td><td>0.06</td><td>0.055</td><td>0.074</td></tr><tr><td>Group-Lasso</td><td>0.036</td><td>0.06</td><td>0.059</td><td>0.081</td></tr></table>

## 4.2 Evaluation of Computational Efficiency

We evaluate the computational efficiency of the original MISDP and the three proposed relaxations. We measure both wall-clock computation time and memory consumption as the size of the underlying semidefinite matrix increases. This experiment is intended to quantify the computational benefit of eliminating the binary variables associated with the attack location.

Experimental Setup. We evaluate four network architectures, denoted Tiny, Small, Medium, and Large, as in Tab. 1. We vary the width of the hidden layers, which is the primary factor governing the dimension of the semidefinite matrix. Because the original 28 × 28 MNIST resolution is computationally prohibitive for complete verification baselines, we train and evaluate our models on downsampled 5 × 5 images (25 dimensions) of the MNIST training data. For the verification setup, we set the perturbation radius to ε = 0.1, the target class to 5, and attack cardinality to k = 1, partitioning the input into 4-pixel groups.

Results. The computation time results are reported in Tab. 2. Across all four evaluated network architectures, the proposed relaxations drastically accelerate verification compared to the exact MISDP formulation (Eq. (9)), requiring at most 3% of the computation time of the original MISDP. Specifically, the proposed relaxation methods achieve speedup factors of at least 41×, 209×, 1320×, and 1430× for the Tiny, Small, Medium, and Large networks, respectively. This significant reduction in runtime stems from removing the binary group-selection vector G.

Table 3: Average increase in the objective value relative to the original MISDP for different perturbation radii, averaged over 50 randomly sampled test images.
<table><tr><td>Method</td><td> $\varepsilon = 0 . 1$ </td><td> $\varepsilon = 1 0 ^ { - 4 }$ </td></tr><tr><td>Full-Group</td><td>92.82%</td><td>1.07%</td></tr><tr><td>L2-norm</td><td>20.76%</td><td>0.19%</td></tr><tr><td>Group-Lasso</td><td>0.34%</td><td>0.00%</td></tr></table>

## 4.3 Evaluation of Relaxation Tightness

We investigate how closely the proposed relaxations approximate the original MISDP. We compare the objective values obtained by the three relaxations with those of the original MISDP. Because each relaxation enlarges the feasible perturbation set, its optimal objective value provides a measure of the relaxation tightness.

Experimental Setup. We use the Small network configuration from Tab. 1. The input consists of MNIST images compressed to 25 dimensions. We set the target class to 5, use the ground-truth class as the true class, set $k = 1$ and use groups containing four elements. We evaluate two perturbation radii, $\varepsilon = 0 .$ 1 and $\varepsilon = 1 0 ^ { - 4 }$ . For each setting, we compare the objective value of each relaxation with that of the original MISDP. The reported increase is computed relative to the original MISDP objective value.

Results. Tab. 3 shows the average increase in the objective value relative to the original MISDP for different perturbation radii. For each perturbation radius, we randomly sampled 50 images from the MNIST test set, computed the percentage increase in the objective value relative to the original MISDP solution for each sample, and averaged the results. As shown, Group-Lasso relaxation provides the tightest approximation to the original MISDP among the three relaxations. Even at the large attack radius ε = 0.1, the objective value changes by only 0.34% relative to the exact solution. This tight approximation is enabled by Group-Lasso relaxation, which can retain the group-wise structure of the original attack, whereas the other relaxations discard an increasing amount of this structure. This trend is also evident in Fig. 2. Group-Lasso solutions lie closest to the diagonal, while the $L _ { 2 } .$ -norm and Full-Group relaxations exhibit progressively larger deviations. This ordering is consistent with the inclusion relations established in Sec. 3.4.4.

## 4.4 Verification Rate Comparison

We evaluate the verification rates of the proposed relaxations, comparing them with prior verification methods in addition to the relaxation tightness comparison in Sec. 4.3.

![](images/fd45d6a8044a4451470c98d2a5b1da0107a56b5b3d34a8dd9a97179a291a9726.jpg)  
Objective function value of original MISDP  
Figure 2: Comparison of the objective values obtained by the original MISDP and the proposed relaxations for ε = 0.1. Each point corresponds to one evaluation sample. Group-Lasso is almost on the diagonal, demonstrating its very tight relaxation of the MISDP.

Experimental Setup. We follow the evaluation methodology adopted in prior work [18] to facilitate comparison with existing verification methods. We evaluate two fully connected architectures, FCNNa and FCNNb. FCNNa consists of two hidden layers with width 20, while FCNNb consists of three hidden layers with width 100. For both FCNNa and FCNNb, the input layer dimension matches the total number of pixels in the input image, which is 784 for MNIST in our case. For each architecture, we consider both normally trained (Clean) and PGD-adversarially trained (Adv.) models. The adversarially trained models are trained using PGD with the perturbation radius specified by the corresponding setting. For each setting, we evaluate 100 test samples, selected at indices 0, 100,.., 9900. The verification rate is the fraction of samples for which the verification optimization certifies that the model remains robust against the specified target class under the corresponding perturbation set. We compare the proposed Full-Group, L -norm, and Group-Lasso relaxations against bounded-MILP [30], CROWN-IBP [37], and CROWN [36]. The attack cardinality k is 1, partitioning the input into 4-pixel groups. These baselines are representative $L _ { p } .$ -norm-based verification methods and include the state-of-the-art methods reported in [18].

Results. The verification rates across models and defense settings are summarized in Tab. 4. For the FCNNa architecture, our proposed Group-Lasso relaxation achieves high verification rates of 86% on the standard (Clean) model and

![](images/3c808e7e81bb1491a278b39987c015759deab747ee051acf0f40054e0a9a3b31.jpg)  
(a) Ours: Full-Group, L , & Group- (b) IBP (c) MILP Lasso

Figure 3: Visualization of perturbations of successfully verified samples when changing the predicted label from 7 to 0. Red and blue mean positive- and negative-direction perturbations, respectively. Only Group-Lasso can assume the localized perturbation.  
![](images/acdfde7396f281789334ee49ad1d9512ddb2b512658988c80c7cd05019dcc40a.jpg)

![](images/8731b7b2f47adbf8dae71d3fdb4f672b2f8e4327c264175c62f116a5a2b5842a.jpg)

![](images/479b31e1300368693d9c297350e3e1239dceb7da20eb2dfda1021edb96d61e6b.jpg)  
(a) Ours: Full-Group, L , & Group- (b) IBP (c) MILP Lasso  
Figure 4: Visualization of perturbations for an unverified sample when changing the predicted label from 2 to 0. Red and blue mean positive- and negative-direction perturbations, respectively. Group-Lasso’s perturbation is localized, but it attempts to draw 0.

91% on the adversarially trained (Adv.) model. Under the Adv. setting, Group-Lasso substantially outperforms existing baselines, including bounded-MILP (70%), CROWN (47%), and CROWN-IBP (2%). Similarly, for the deeper FCNNb network, Group-Lasso achieves the highest verification rate of 69% in the Adv. setting, whereas prior baselines fail almost completely (bounded-MILP at 0%, CROWN-IBP at 0%, and CROWN at 4%). The observation is consistent when k is increased to 10, as shown in Tab. 5. Although bounded-MILP and CROWN yield higher certified rates on the Clean FCNNb model (84%), we consider that our relaxations are particularly effective on challenging samples that are difficult to verify.

To quantify this, we examine the margin of the optimal objective value $f ( { \pmb x } ) [ j ] - f ( { \pmb x } ) [ j ^ { * } ]$ , where negative values closer to zero indicate narrower margins. Among the evaluated inputs, the average optimal value for samples certified exclusively by Group-Lasso was −1.69, compared to −10.94 for samples certified by all methods. This significant gap indicates that conventional verifiers suffer from loose relaxation bounds on challenging inputs with small margins, whereas preserving the group-sparse attack structure allows Group-Lasso to certify such instances that remain unprovable by existing baselines.

## 4.5 Qualitative Analysis

We qualitatively examine the perturbations obtained from the optimization solutions. This analysis provides an intuitive comparison between the perturbations produced by the proposed group-wise threat model and those produced by conventional $L _ { p } .$ -based verification methods.

Table 4: Verification results on standard (Clean) and adversarially trained (Adv.) models. We compare our proposed SDP relaxations against prior verification baselines, showing state-of-the-art performance [18].
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="2">FCNNa</td><td colspan="2">FCNNb</td></tr><tr><td>Clean (ε = 0.02)</td><td>Adv. (ε = 0.10)</td><td>Clean (ε = 0.02)</td><td>Adv. (ε = 0.30)</td></tr><tr><td rowspan="3">Ours</td><td>Full-Group</td><td>83%</td><td>70%</td><td>59%</td><td>21%</td></tr><tr><td>L2-norm</td><td>86%</td><td>70%</td><td>59%</td><td>69%</td></tr><tr><td>Group-Lasso</td><td>86%</td><td>91%</td><td>69%</td><td>69%</td></tr><tr><td rowspan="3">Baselines</td><td>Bounded-MILP</td><td>80%</td><td>70%</td><td>84%</td><td>0%</td></tr><tr><td>CROWN-IBP</td><td>77%</td><td>2%</td><td>0%</td><td>0%</td></tr><tr><td>CROWN</td><td>79%</td><td>47%</td><td>84%</td><td>4%</td></tr></table>

Table 5: Verification performance with k = 10. Group-Lasso shows higher verification rates than other unstructured relaxations.
<table><tr><td>Method</td><td>FCNNa Clean</td><td>FCNNa Adv.</td></tr><tr><td>Full-Group</td><td>83%</td><td>70%</td></tr><tr><td>L2-norm</td><td>83%</td><td>66%</td></tr><tr><td>Group-Lasso</td><td>83%</td><td>80%</td></tr></table>

![](images/7c67675c42fea7f5695b5c2e0f2d66092b94c2c961f0143d1fada607b7b0a55e.jpg)

![](images/972cedf851373bf22c3776a93c5e700474389d06aac65acda6c12a5fd1d17a9a.jpg)

![](images/719d1b3d17a67cf3dcc7b7f8adda3f7a9dafed39ccfbe3af30e65ae2159b6010.jpg)  
(a) Ours: Full-Group, $L _ { 2 } ,$ & Group- (b) IBP (c) MILP Lasso

![](images/aa2c07cb283cf8a9d447e216af72c57797bce10129f647ccfc6831d7785cee78.jpg)

![](images/efdcea7a47a3b8f16d2bfe8f8a19d0ce898568ef0402ae2f9771c141d27c7352.jpg)  
Figure 5: Reconstructed attacked images from Fig. 4, attempting to change the prediction from 2 to 0.

For bounded MILP, we directly use the optimal solution, as it is complete verification. For incomplete verification, which only tests the upper bound of the objective value and does not return attack perturbations, we construct approximated attack perturbations using the following procedure. For IBP perturbations, we determine the perturbation direction for each pixel based on its sensitivity to the true and target label scores, computed via forward propagation, and obtain an approximate perturbation by moving as far as possible in that direction within the input perturbation bound. For SDP-based methods including ours, we extract the subvector corresponding to the original image and reshape it back into the original image’s 2D array. Specifically, the sub-vector extracted corresponds to entries [0,1] through [0,256] of the SDP solution.

Perturbations of a Successfully Verified Sample. Figure 3 shows the perturbations obtained for an input image whose true label is 7, shown in Fig. 1, with target class 0, in a setting where Group-Lasso successfully verifies the model.

All methods attempt to suppress or modify the digit 7. However, Group-Lasso relaxation concentrates the perturbation almost entirely on the digit itself, leaving the surrounding background unchanged. This behavior is consistent with the assumed localized patch threat model. In contrast, the perturbations obtained by the other methods extend into the background region. In particular, the perturbations produced by IBP and MILP distribute non-negligible changes outside the digit region. The proposed Full-Group and $L _ { 2 } \cdot$ -norm relaxations reduce this effect compared with these baselines, but still permit perturbations outside the localized region. These observations illustrate the benefit of explicitly retaining group sparsity in the perturbation set.

Perturbations of an Unverified Sample. Figure 4 shows an instance with true label 2, shown in Fig. 1, and target class 0 for which Group-Lasso does not certify robustness. Fig. 5 shows the corresponding reconstructed images. In this case, the perturbations produced by most methods modify both the digit and the surrounding background. Group-Lasso not only concentrates its perturbation on the digit region, but also tries to draw the target digit, 0. This supports the high capability of Group-Lasso to handle the patch attack, which can only inject localized perturbations. When subtle perturbations are insufficient, it is natural that the attacker instead draws the target digit.

In summary, the qualitative results support the motivation for the proposed relaxation: Group-Lasso not only provides a substantially tighter relaxation than the simpler alternatives, but also produces optimization solutions whose perturbations are spatially concentrated in a manner consistent with the assumed patch attack.

## 5 Discussion and Limitations

The results highlight the importance of preserving the spatial structure of patch threats during verification. Among the proposed relaxations, Group-Lasso consistently provides the tightest approximation to the original MISDP and effectively concentrates perturbations within targeted regions, leading to stronger robustness certification against patch attacks. However, there are several limitations in this study. We evaluated only fully connected networks and small-scale inputs, leaving high-resolution images and modern architectures for future work. Our primary focus is the theoretical formulation of patch attacks (Sec. 3) and the tightness of the proposed relaxations relative to the original MISDP, which is itself unsuitable for large-scale problems. As the first formal verification framework specialized for patch attacks, we leave the development of scalable solvers, such as firstorder methods [1, 7], to future work. Finally, our fixed, non-overlapping image partition may not effectively capture arbitrary physical patch geometries. Nevertheless, it can flexibly incorporate realistic deployable regions, which are naturally constrained in physical settings (e.g., stickers cannot be placed on the sky).

## 6 Conclusion

In this work, we introduced a one-shot framework for robustness verification against patch attacks. Instead of enumerating possible attack locations, we formulated the attack location jointly with the adversarial perturbation and derived an exact mixed-integer semidefinite programming formulation. We then eliminated the discrete attack-selection variables through three convex relaxations: Full-Group, L -norm, and Group-Lasso. Our analyses demonstrate that explicitly preserving the spatial group-sparsity of real-world threats is critical for effective verification. Group-Lasso relaxation provides a tight and efficient approximation, incurring only a 0.34% average increase in the objective value compared to the original MISDP while requiring less than 3% of the computation time. Furthermore, by achieving a verification rate of 91% compared $t 0 \leq 7 0 \%$ for prior baselines, our approach offers a practical, scalable solution for certifying neural network robustness. Future work will explore extending this framework to broader classes of physical attacks and larger network architectures.

## References

[1] Amir Ali Ahmadi and Anirudha Majumdar. DSOS and SD-SOS Optimization: More Tractable Alternatives to Sum of Squares and Semidefinite Optimization. SIAM Journal on Applied Algebra and Geometry, 3(2):193–230, 2019. doi: 10.1137/18M118935X.

[2] Osbert Bastani, Yani Ioannou, Leonidas Lampropoulos, Dimitrios Vytiniotis, Aditya V. Nori, and Antonio Criminisi. Measuring Neural Net Robustness with Constraints. In NeurIPS, volume 29, pages 2613–2621, 2016.

[3] Ksenia Bestuzheva, Leon Eifler, Ambros Gleixner, Gerald Gamrath, Katrin Halbig, Christopher Hojny, Thorsten Koch, Antonia Chmiela, Marc Pfetsch, Franziska Schlösser, Felipe Serrano, Yuji Shinano, Stefan Vigerske, and Dieter Weninger.

The SCIP Optimization Suite 9.0. Technical Report ZR-24- 06, Zuse Institute Berlin, 2024. URL https://arxiv.org/ abs/2402.17702.

[4] Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, and Karol Zieba. End to End Learning for Self-Driving Cars. arXiv preprint arXiv:1604.07316, 2016.

[5] Elena Botoeva, Panagiotis Kouvaros, Jan Kronqvist, Alessio Lomuscio, and Ruth Misener. Efficient Verification of ReLU-Based Neural Networks via Dependency Analysis. In AAAI, pages 3291–3299, 2020.

[6] Tom B Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. Adversarial Patch. In NIPS 2017 Workshop on Machine Learning and Computer Security, 2017. URL https://arxiv.org/abs/1712.09665.

[7] Samuel Burer and Renato D. C. Monteiro. A Nonlinear Programming Algorithm for Solving Semidefinite Programs via Low-Rank Factorization. Mathematical Programming, 95 (2):329–357, 2003. doi: 10.1007/s10107-002-0352-8.

[8] Nicholas Carlini and David Wagner. Towards Evaluating the Robustness of Neural Networks. In SP, pages 39–57, 2017.

[9] Ping-Yeh Chiang, Renkun Ni, Ahmed Abdelkader, Chen Zhu, Christoph Studer, and Tom Goldstein. Certified Defenses for Adversarial Patches. In ICLR, 2020. URL https://arxiv. org/abs/2003.06693.

[10] Rüdiger Ehlers. Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks. In ATVA, 2017. URL https: //api.semanticscholar.org/CorpusID:1931807.

[11] Andre Esteva, Brett Kuprel, Roberto A. Novoa, Justin Ko, Susan M. Swetter, Helen M. Blau, and Sebastian Thrun. Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks. Nature, 542(7639):115–118, 2017. doi: 10.1038/nature21056.

[12] Yanbo Fan, Baoyuan Wu, Tuanhui Li, Yong Zhang, Mingyang Li, Zhifeng Li, and Yujiu Yang. Sparse Adversarial Attack via Perturbation Factorization. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, ECCV, pages 35–50, Cham, 2020. Springer International Publishing.

[13] Ambros Gleixner and Daniel E Steffy. Linear Programming Using Limited-Precision Oracles. Mathematical Programming, 183(1):525–554, 2020.

[14] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and Harnessing Adversarial Examples. In ICLR, 2015.

[15] Guy Katz, Clark W. Barrett, David L. Dill, Kyle D. Julian, and Mykel J. Kochenderfer. Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. In CAV, pages 97–117. Springer, 2017.

[16] Jianglin Lan, Benedikt Brückner, and Alessio Lomuscio. A Semidefinite Relaxation Based Branch-and-Bound Method for Tight Neural Network Verification. In AAAI, pages 14946– 14954, 06 2023. doi: 10.1609/aaai.v37i12.26745.

[17] Alexander Levine and Soheil Feizi. (De)Randomized Smoothing for Certifiable Defense against Patch Attacks. In NeurIPS, volume 33, 2020. URL https://arxiv.org/abs/2002. 10733.

[18] Linyi Li, Tao Xie, and Bo Li. SoK: Certified Robustness for Deep Neural Networks. In SP, San Francisco, CA, USA, 2023. IEEE.

[19] Xin Liu, Huanrui Yang, Ziwei Liu, Linghao Song, Hai Li, and Yiran Chen. DPatch: An Adversarial Patch Attack on Object Detectors. In Proceedings of the 2019 AAAI Workshop on Artificial Intelligence Safety (SafeAI 2019), 2019. URL https://ceur-ws.org/Vol-2301/paper\_5.pdf.

[20] Alessio Lomuscio and Lalit Maganti. An Approach to Reachability Analysis for Feed-Forward ReLU Neural Networks. arXiv preprint arXiv:1706.07351, 2017.

[21] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards Deep Learning Models Resistant to Adversarial Attacks. In ICLR, 2018. arXiv:1706.06083.

[22] Jan Hendrik Metzen and Maksym Yatsura. Efficient Certified Defenses against Patch Attacks on Image Classifiers. In ICLR, 2021. URL https://openreview.net/forum?id= hr-3PMvDpil.

[23] MOSEK ApS. The MOSEK Python Fusion API Manual. Technical report, 2026. URL https://docs.mosek.com/ latest/pythonfusion/index.html. Version 11.2.

[24] Dean A. Pomerleau. ALVINN: An Autonomous Land Vehicle in a Neural Network. In David S. Touretzky, editor, Advances in Neural Information Processing Systems 1, pages 305–313. Morgan Kaufmann, San Francisco, CA, 1989.

[25] Aditi Raghunathan, Jacob Steinhardt, and Percy S Liang. Semidefinite Relaxations for Certifying Robustness to Adversarial Examples. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, NeurIPS, volume 31, pages 10900–10910, 2018. URL https://proceedings. neurips.cc/paper\_files/paper/2018/file/ 29c0605a3bab4229e46723f89cf59d83-Paper.pdf.

[26] Takami Sato, Junjie Shen, Ningfei Wang, Yunhan Jia, Xue Lin, and Qi Alfred Chen. Dirty Road Can Attack: Security of Deep Learning Based Automated Lane Centering under Physical-World Attack. In 30th USENIX Security Symposium (USENIX Security 21), pages 3309–3326. USENIX Association, August 2021. ISBN 978-1-939133- 24-3. URL https://www.usenix.org/conference/ usenixsecurity21/presentation/sato.

[27] Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K Reiter. Accessorize to a Crime: Real-World Fooling of Face Recognition. In CCS, pages 1528–1540. ACM, 2016. doi: 10.1145/2976749.2978392. URL https: //dl.acm.org/doi/10.1145/2976749.2978392.

[28] Dawn Song, Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Florian Tramèr, Atul Prakash, and Tadayoshi Kohno. Physical Adversarial Examples for Object Detectors. In 12th USENIX Workshop on Offensive Technologies (WOOT 18), Baltimore, MD, August 2018. USENIX Association. URL https://www.usenix.org/conference/ woot18/presentation/eykholt.

[29] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In ICLR, 2014.

[30] Vincent Tjeng, Kai S. Xiao, and Russ R. Salakhutdinov. Evaluating Robustness of Neural Networks with Mixed Integer Programming. In ICLR, 2019. URL https://arxiv.org/ abs/1711.07356.

[31] Shiqi Wang, Huan Zhang, Kaidi Xu, Suman Jana, Xue Lin, Cho-Jui Hsieh, and J. Zico Kolter. Beta-CROWN: Efficient Bound Propagation with Per-Neuron Split Constraints for Neural Network Verification. In NeurIPS, volume 34, pages 29909–29921, 2021.

[32] Eric Wong and J. Zico Kolter. Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope. In ICML, pages 5283–5292, 2018.

[33] Yuzheng Wu, Maryellen L. Giger, Kunio Doi, Carl J. Vyborny, Robert A. Schmidt, and Charles E. Metz. Artificial Neural Networks in Mammography: Application to Decision Making in the Diagnosis of Breast Cancer. Radiology, 187(1):81–87, 1993. doi: 10.1148/radiology.187.1.8451441.

[34] Chong Xiang, Arjun Nitin Bhagoji, Vikash Sehwag, and Prateek Mittal. PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking. In 30th USENIX Security Symposium (USENIX Security 21), pages 2237–2254, 2021.

[35] Chong Xiang, Saeed Mahloujifar, and Prateek Mittal. PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier. In 31st USENIX Security Symposium (USENIX Security 22), pages 2065–2082, Boston, MA, August 2022. USENIX Association. ISBN 978-1-939133-31- 1. URL https://www.usenix.org/conference/ usenixsecurity22/presentation/xiang.

[36] Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient Neural Network Robustness Certification with General Activation Functions. In NeurIPS, volume 31, 2018.

[37] Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li, Duane Boning, and Cho-Jui Hsieh. Towards Stable and Efficient Training of Verifiably Robust Neural Networks. In ICLR, 2020.