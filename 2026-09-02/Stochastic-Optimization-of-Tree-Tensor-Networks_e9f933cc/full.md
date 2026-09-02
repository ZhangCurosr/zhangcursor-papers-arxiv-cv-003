# Stochastic Optimization of Tree Tensor Networks

Marius Willner<sup>12‹</sup>, Maximilian Scharf<sup>23:</sup>, André Uschmajew<sup>14</sup>, Timo Felser<sup>2</sup> and Marco Trenti<sup>2</sup>

1 Institute of Mathematics, University of Augsburg, 86159 Augsburg, Germany 2 Tensor AI Solutions GmbH, 89284 Pfaffenhofen an der Roth, Germany   
3 Institute for Complex Quantum Systems, Ulm University, 89081 Ulm, Germany   
4 Centre for Advanced Analytics and Predictive Sciences, University of Augsburg, 86159 Augsburg, Germany

‹: These authors contributed equally to this work. ‹ marius.willner@uni-a.de , : maximilian.scharf@tensor-solutions.com

## Abstract

Tensor networks, originally developed for quantum many-body physics, are promising models for machine learning. We derive stochastic Riemannian optimizers for tree tensor networks (TTNs) on both their parameter and quotient manifolds, including adaptive and learning-rate-free schemes suitable for minibatch training. Using a hybrid CNN–TTN architecture, we evaluate the methods on Fashion-MNIST, CIFAR10, and Imagenette. The proposed optimizers achieve predictive performance comparable to unconstrained optimization while enabling numerically stable downstream compression.

## Contents

1 Introduction 2   
2 Preliminaries 3   
2.1 Optimization on TTNs 3   
2.2 Adaptive Riemannian optimization 8   
2.3 Riemannian distance over gradients 9   
3 Tools 10   
3.1 Adaptive optimization of TTNs 10   
3.2 Distance over gradients on TTN-manifolds 11   
4 Algorithms 11   
4.1 RADAM 11   
4.2 RDOG 12   
4.3 RMUON 13   
5 Numerical Experiments 13   
5.1 Fashion MNIST 14   
5.2 CIFAR10 & Imagenette 15   
6 Conclusion 18   
A Geodesics of the TTN quotient 19   
B Convergence of RADAGRAD 20   
C Additional numerical results 23

## 1 Introduction

Modern machine learning relies overwhelmingly on first-order optimization, and in practice this means stochastic gradient descent (SGD) rather than full-batch gradient descent. Given a loss that decomposes as an expectation over data, $\begin{array} { r } { f ( \theta ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m ^ { - } } \ell ( \theta ; x ^ { k } , y ^ { k } ) } \end{array}$ , the exact gradient ∇f costs $O ( m )$ per step, which is prohibitive for large m. SGD replaces it with an unbiased estimate computed on a minibatch B,

$$
\nabla f ( \theta ) \approx \frac { 1 } { | \boldsymbol { \mathcal { B } } | } \sum _ { k \in \boldsymbol { B } } \nabla _ { \theta } \ell \big ( \theta ; x ^ { k } , y ^ { k } \big ) ,
$$

trading gradient accuracy for a large gain in per-step cost [1, 2]. Beyond efficiency, the injected gradient noise has genuine optimization value: it helps escape saddle points and sharp minima, and is widely associated with improved generalization in overparameterized models [3, 4]. Adaptive variants such as ADAM [5] or MUON [6] and distance-over-gradient schemes [7] refine the basic method, and its convergence behavior for convex and nonconvex objectives is by now well characterized [8]. Together these properties have made stochastic first-order optimization the default training paradigm across essentially all of large-scale machine learning.

Tensor networks—matrix product states (MPS), tree tensor networks (TTN), and related structures developed for quantum many-body physics [9, 10]—have enabled simulations of challenging higher-dimensional many-body and lattice-gauge systems [11, 12, 13], and have subsequently been repurposed as trainable models for supervised classification and generative modeling [14, 15]. Applications of TN-based learning already include high-energy physics data analysis [16, 17] and, more recently, robust and compressible classification of syntheticaperture-radar data [18].

The canonical optimizer inherited from physics is the density-matrix renormalization group (DMRG), a deterministic algorithm that sweeps across the network, locally optimizing one or two tensors at a time to near-optimality [9]. DMRG is extraordinarily effective for computing ground states, but its central assumption—a fixed objective whose local environment can be formed and factorized exactly at each step—sits uneasily with the machine-learning setting, where the objective is an empirical average over a large, shuffled dataset. Evaluating or differentiating that average exactly reintroduces the same $O ( m )$ cost that motivates stochastic methods in the first place [1, 2], while the nonconvex, multilinear loss landscapes that tensor networks induce are precisely the regime in which gradient noise aids optimization and generalization [8, 4, 3].

Stochastic gradient methods address both issues directly, and a growing body of work trains tensor-network models this way. The dominant approach obtains gradients by automatic differentiation: the gradient of the loss with respect to each core tensor is assembled by backpropagation through the network contraction and passed to a generic stochastic optimizer, as in [19] and in recent training libraries built on deep-learning frameworks. This route is convenient and integrates tensor-network layers end-to-end with conventional neural components [14, 19, 20], but it treats the network as a generic parameterized function and ignores the geometric structure of the underlying decomposition. A second, smaller line of work instead respects that structure, performing Riemannian optimization on the fixed-rank manifold of tensor-train models [21] and TTNs [22, 23, 24]; such methods, however, have largely been developed in the full-batch context. Recent advances in the closely related field of orthogonal optimization [25, 26, 27], such as the POGO algorithm [26], serve as a basis to extend on those methods.

In this paper, we work out the forms of the stochastic algorithms on the manifolds of TTNs. Rather than indiscriminately using stochastic optimizers originally meant for Euclidean optimization, we derive the Riemannian gradient of the loss in terms of the network’s tensors and its canonical gauge structure, yielding explicit, minibatch-computable update rules that respect the TTN (quotient) geometry. Crucially, we use that the TTN parameter space factors as a Cartesian product of manifolds, enabling adaptive Riemannian optimization. Even though the TTN quotient does not exhibit a product structure, we show that it shares its geodesic spray with a Cartesian product of Grassmann manifolds. We use this to theoretically back up our considerations and to conclude about the exponential map and Riemannian distances, which are central components of our stochastic optimizers. Furthermore, we devise a novel hybrid CNN-TTN architecture, that allows the numerical evaluation of the resulting algorithms on large-scale datasets, such as CIFAR10 and Imagenette: they perform competitively with unconstrained adaptive optimizers, but produce final iterates that conform significantly better to the hierarchical structure of the TTN. When used for downstream compression tasks, deep networks optimized unconstrainedly break down completely, whereas deep TTNs trained with our methods react as expected.

The work is organized as follows: in Sec. 2, we formally introduce the TTN, and the geometric properties of its parameter and quotient space, as well as adaptive and distance-overgradients schemes for general manifolds; in Sec. 3 we discuss the application of those to TTNs as our main contribution; in Sec. 4 we provide pseudocode and protocols for the implementation of the different SGD flavours; in Sec. 5 we test the developed algorithms on computer vision benchmarks by using a CNN feature extractor and the TTN classifier; finally, in Sec. 6 we discuss the overall SGD picture for TTNs and their possible applications, open problems, and extensions for ML and beyond.

## 2 Preliminaries

## 2.1 Optimization on TTNs

In this section, we will introduce general tree tensor networks, orthogonal TTN, and some important geometrical properties of TTN that are needed to develop Riemannian optimization algorithms. For a more detailed derivation and proofs see [28, 24].

## 2.1.1 Tree tensor networks

We will start with a definition of TTNs and of the notation that will be used throughout this work. Let $N = \{ 1 , . . . , n \}$ be a set of labels for the physical indices $i _ { 1 } , . . . , i _ { n }$ with dimensions $d _ { 1 } , . . . , d _ { n }$ , that can be interpreted as the input of the tensor networks considered here. We will assume that n is even for simplicity, but the definition can also be extended to an odd number of inputs. The tensor networks we consider here have an additional index j with dimension $d _ { \mathrm { o u t : } }$ , which corresponds to the output.

A tree tensor network (TTN) is a tuple $\theta = \left( B ^ { ( t ) } \right) _ { t \in T }$ of order three core tensors $B ^ { ( t ) }$ , that are assembled through a rooted binary tree graph and labeled with elements $t \in T$ . The set of labels T is a subset of the power set $T \subset { \mathcal { P } } ( N )$ . Each tensor label is the set of all labels of those physical indices that are children of the respective tensor. Due to the binary tree graph structure, each node $t \in T$ has two direct child nodes $t _ { L } , t _ { R } \in T \cup N$ with $t = t _ { L } \cup t _ { R }$ Each core tensor connects three indices $\nu _ { t _ { L } } , \nu _ { t _ { R } }$ and $\nu _ { t }$ of dimension $\chi _ { t _ { L } } , \chi _ { t _ { R } }$ and $\chi _ { T }$ , and it is canonnically identified with its matricization, meaning $B ^ { ( t ) } \in \mathbb { R } ^ { \chi _ { t _ { L } } \chi _ { t _ { R } } \times \chi _ { t } }$ . At the lowest layer of the TTN, each core tensor joins two consecutive physical indices and a single virtual index $\nu _ { t }$ It holds that $\nu _ { t _ { L } } = i _ { t _ { l } } , \nu _ { t _ { R } } = i _ { t _ { R } } , \chi _ { t _ { L } } = d _ { t _ { l } } , \chi _ { t _ { R } } = d _ { t _ { R } }$ . All other nodes, except for the root, join three virtual indices $\nu _ { t _ { L } } , \nu _ { t _ { R } }$ and $\nu _ { t }$ . The root node $t = N$ joins the two final links, producing the outgoing index $\nu _ { t } ~ = ~ j$ , with dimension $\chi _ { t } = d _ { \mathrm { o u t } }$ . It will be explained in Section 2.1.5 how TTNs may be employed as a machine learning model. Here we first focus on manifold properties of the TTN itself.

![](images/5e599734cbcec433c3c6e750c2c758be9201a7618275811d1933cab179d96bb8.jpg)  
Figure 1: Graphical representation of a tree tensor network with $n \ = \ 8$ physical indices $i _ { 1 } , . . . , i _ { 8 }$ and one "outgoing" index j. The lowest layer of the TTN contains the nodes t1, 2u, t3, 4u, t5, 6u, t7, 8u, that act on two consecutive physical indices each. The next layer are intermediate nodes t1, 2, 3, 4u, t5, 6, 7, 8u, followed by the root node with label $N = \{ 1 , . . . , 8 \}$

## 2.1.2 Manifold structure

The core tensors $\theta = \big ( B ^ { ( t ) } \big ) _ { t \in T }$ of the TTN parameterize the Euclidean space [28]

$$
\mathcal { E } : = \bigvee _ { t \in T } \mathbb { R } ^ { \chi _ { t _ { L } } \chi _ { t _ { R } } \times \chi _ { t } } .
$$

Furthermore, a TTN is always associated with a single tensor, which is obtained through the parameter map $\tau : \mathcal { E }  \dot { \mathbb { R } } ^ { d _ { 1 } \times \dots \times d _ { n } \times d _ { \mathrm { o u t } } }$ t given by the contraction of all virtual indices $\nu _ { t }$ $t \in T ^ { * } = T \backslash N$ . The dimensions $( \chi _ { t } ) _ { t \in T ^ { * } }$ of these virtual indices are referred to as the bond dimensions of the TTN. They are subject to $\chi _ { t } \leqslant \chi _ { t _ { L } } \chi _ { t _ { R } }$ [28], but may be chosen freely otherwise. Importantly, they govern the expressivity and computational complexity of the TTN as a model, and we assume them fixed throughout our considerations. A good initial guess for the bond dimensions can be found through the unsupervised construction algorithm [29], they can be reduced using TTN compression algorithms [18] after training, or they may be chosen adaptively during optimization (see e.g. [30]) in more advanced algorithms not covered in this work.

In practice, it is usually preferable to work with orthogonal (also called isometrized) TTN, which define an important submanifold $\tau$ of general TTN. An orthogonal TTN $\theta \in \mathcal T$ is defined in the same way as general TTN, with the additional restriction that all nodes $t \in T ^ { * }$ except for the root node are isometries, i.e., elements of the Stiefel manifold $B ^ { ( t ) } \in \mathrm { S t } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } )$ satisfying

$$
( B ^ { ( t ) } ) ^ { T } B ^ { ( t ) } = \mathbb { 1 } .
$$

The parameter space of orthogonal TTN [24] is given by

$$
\mathcal { T } = \left( \bigcup _ { t \in T ^ { * } } \operatorname { S t } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } ) \right) \times \mathbb { R } ^ { \chi _ { N _ { L } } \times \chi _ { N _ { R } } \times d _ { \operatorname { o u t } } } ,\tag{1}
$$

and its tangent space reads

$$
T _ { \theta } \mathcal { T } = \left( \bigcup _ { t \in T ^ { \ast } } T _ { B } \mathsf { S t } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } ) \right) \times \mathbb { R } ^ { \chi _ { N _ { L } } \chi _ { N _ { R } } \times d _ { \mathsf { o u t } } } .
$$

It is the Cartesian product of the individual tangent spaces of the core tensors, where the tangent space of the Stiefel manifold $S \mathrm { t } ( n , k )$ is given by

$$
T _ { X } { \mathrm { S t } } ( n , k ) = \left\{ V \in \mathbb { R } ^ { n \times k } : X ^ { T } V + V ^ { T } X = 0 \right\}
$$

and the tangent space of the root tensor is again a Euclidean space.

The space $\tau$ is an embedded submanifold of the Euclidean space $\mathcal { E } .$ . From the perspective of optimization, working with $\tau$ is preferable over $\mathcal { E }$ as it is on the one hand numerically more stable and on the other hand, we can exploit results of differential geometry to derive more sophisticated optimization algorithms. In the following, we will thus work exclusively with orthogonal TTN. Note that because of the gauge freedom<sup>1</sup>, the parameter map $\boldsymbol { \tau }$ is not injective. Each TTN θ can be mapped to an orthogonal TTN $\theta ^ { \prime } \in \mathcal T$ through a process called isometrization , such that $\tau ( \theta ) = \tau ( \theta ^ { \prime } )$ . Therefore, considering only orthogonal TTN is not a restriction.

## 2.1.3 Quotient structure

Note that even if we restrict the domain of τ to $\tau _ { \ast }$ , it is still not injective. This is due to the fact that at each virtual link $\nu _ { t }$ of the TTN, one can insert an orthogonal matrix $A ^ { ( t ) } \in O ( \chi _ { t } )$ and its inverse $( A ^ { ( t ) } ) ^ { T }$ , which can be contracted with the two core tensors connected by this link, respectively. After this contraction, the parameters of the two tensors changed, without breaking the isometry condition or changing the result of $\boldsymbol { \tau }$ (see Fig. 2a for a graphical representation). Formally, this remaining gauge freedom is characterized by an action of the Lie group [28]

$$
\mathcal { G } = \{ \boldsymbol { A } = ( \boldsymbol { A } ^ { ( t ) } ) _ { t \in T ^ { * } } : \boldsymbol { A } _ { t } \in \mathrm { O } ( \chi _ { t } ) \} .\tag{2}
$$

We can take the quotient of $\tau$ with respect to this Lie group to obtain the quotient manifold $\tau / \mathcal { G }$ with associated quotient map $\pi : \mathcal T \to \mathcal T / \mathcal G$ that maps all parameters $\theta \in \mathcal { T }$ that are equivalent under $\mathcal { G }$ to the same representative $[ \theta ] \in { \mathcal { T } } / { \mathcal { G } }$ . Strictly speaking, in order for $\tau / \mathcal { G }$ to be a formal quotient manifold one has to restrict $\tau$ to only include tensors of full multilinear rank towards virtual links [28, 24]. As this does not affect our discussion, to lighten notation, we omit this technical detail. Moreover, in the following, whenever we write $\theta \in \mathcal T$ , we silently assume that $\theta$ fulfills those rank constraints. If we push the parameter map to the quotient space $\hat { \tau } : \mathcal { T } / \mathcal { G } \to \mathbb { R } ^ { d _ { 1 } \times \cdots \times d _ { n } \times d _ { \mathrm { o u t } } }$ t , satisfying $\tau = \hat { \tau } \circ \pi$ , it finally becomes injective (see Fig. 2b) [24]. This is desirable for optimization, however, it is important to note that the quotient space is an abstract space that is only useful in theory and the elements rθ s cannot directly be represented numerically. Nevertheless, it is still useful to study optimization on ${ \mathcal { T } } / { \mathcal { G } }$ , which ultimately produces practical iteration rules in $\tau$

Most importantly, we want to identify those directions in $T _ { \theta } \mathcal { T }$ that also lie in the tangent space of $T _ { \lceil \theta \rceil } { \mathcal { T } } / { \mathcal { G } } ,$ , as those directions are invariant under the action of $\mathcal { G }$ and therefore influence $\Theta = { \dot { \tau } } { \dot { ( \theta ) } }$ The subspace containing the irrelevant directions is called the vertical space $V _ { \theta } { \mathcal { T } } \subset T _ { \theta } { \mathcal { T } }$ , whereas any complementary subspace in $T _ { \theta } \mathcal { T }$ is called the horizontal space $H _ { \theta } \mathcal { T } \subset T _ { \theta } \mathcal { T }$ . There are multiple valid choices for horizontal spaces of $\tau$ , but we will only consider the simplest variant throughout, the Cartesian horizontal space

$$
\begin{array} { r } { H _ { \theta } \mathcal { T } = \biggr ( \underset { t \in T ^ { * } } { \times } H _ { B ^ { ( t ) } } \mathrm { S t } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } ) \biggr ) \times \mathbb { R } ^ { \chi _ { N _ { L } } \chi _ { N _ { R } } \times d _ { \mathrm { o u t } } } , } \end{array}\tag{3}
$$

that is again given by the Cartesian product of the individual horizontal spaces, where the canonical horizontal space of $S \mathrm { t } ( n , k )$ is given by

$$
H _ { X } { \mathrm { S t } } ( n , k ) = \left\{ V \in \mathbb { R } ^ { n \times k } : X ^ { T } V = 0 \right\} .
$$

We can relate tangent vectors of the quotient space to vectors in the horizontal space through the horizontal lift lift $: T _ { [ \theta ] } T / \mathcal { G } \to H _ { \theta } \mathcal { T }$ . Furthermore, we equip $\tau$ with the metric

$$
\rho _ { \theta } : T _ { \theta } \mathcal { T } \times T _ { \theta } \mathcal { T }  \mathbb { R } , \quad \rho _ { \theta } ( \xi , \eta ) = \sum _ { t \in T } \langle \delta B ^ { ( t ) } | \delta C ^ { ( t ) } \rangle = \sum _ { t \in T } \delta B _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } \delta C _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } ,\tag{4}
$$

![](images/a2a8a70eb56c7aa25e37165bcc9d6a3242e9d1d8d221387c15dca482548776c3.jpg)  
Figure 2: Illustration of the gauge freedom of orthogonal TTN and the corresponding quotient space. a) A given orthogonal TTN θ can be altered via the action of $\mathcal { A } \in \mathcal { G }$ by inserting pairs of orthogonal matrices $\bar { A } ^ { ( t ) } , ( A ^ { ( t ) } ) ^ { T }$ (red and orange, respectively) at each virtual link $\nu _ { t } ,$ that change the original core tensors (blue) when contracted, without breaking the isometry constraint or changing the encoded full tensor $\Theta = \tau ( \theta )$ (dark blue). b) All possible representatives $\theta \in \mathcal T$ that are mapped to the same full tensor by τ are represented by a single abstract element $[ \theta ] \in { \mathcal { T } } / { \mathcal { G } }$ in the quotient space.

where $\xi = ( \delta B ^ { ( t ) } ) _ { t \in T }$ and $\eta = ( \delta C ^ { ( t ) } ) _ { t \in T }$ , respectively. Importantly, this also induces a Riemannian metric $\hat { \rho }$ on $T _ { [ \theta ] } \mathcal { T } / \mathcal { G }$ through horizontal lifts:

$$
\begin{array} { r } { \hat { \rho } _ { [ \theta ] } : T _ { [ \theta ] } \mathcal { T } / \mathcal { G } \times T _ { [ \theta ] } \mathcal { T } / \mathcal { G } \to \mathbb { R } , \quad \hat { \rho } _ { [ \theta ] } ( \hat { \xi } , \hat { \eta } ) = \rho _ { \theta } ( \operatorname* { l i f t } _ { \theta } ( \hat { \xi } ) , \operatorname* { l i f t } _ { \theta } ( \hat { \eta } ) ) . } \end{array}\tag{5}
$$

Thus both $( \tau , \rho )$ and $( \mathcal { T } / \mathcal { G } , \hat { \rho } )$ are Riemannian manifolds.

## 2.1.4 Projectors and retractions

General vectors in Euclidean space $\mathcal { E }$ can be projected onto the tangent space and the horizontal space of $\tau$ with the orthogonal projectors ${ \mathrm { P r o j } } _ { \theta } : { \mathcal { E } }  T _ { \theta } { \mathcal { T } }$ and $\operatorname { P r o j } _ { \theta } ^ { H } : \mathcal { E }  H _ { \theta } \mathcal { T } _ { : }$ respectively. Both projectors leave the root tensor invariant and project the other core tensors to the respective spaces of the Stiefel manifold, with $P _ { X } : \mathbb { R } ^ { n \times k }  T _ { X } { \mathrm { S t } } ( n , k )$ given by

$$
P _ { X } ( V ) = V - \frac { 1 } { 2 } X \left( X ^ { T } V + V ^ { T } X \right)
$$

and $P _ { X } ^ { H } : \mathbb { R } ^ { n \times k } \longrightarrow H _ { X } S { \mathrm { t } } ( n , k )$ , given by

$$
P _ { X } ^ { H } ( V ) = ( \mathbb { 1 } - X X ^ { T } ) V .
$$

Furthermore, we need a formalism to move from a point $\theta \in \mathcal T$ to a new point $\theta ^ { \prime } \in \mathcal { T }$ along the direction of a tangent vector $\delta \theta \in T _ { \theta } \mathcal { T }$ . This operation is called a retraction $\mathrm { R } _ { \theta } : T _ { \theta } { \mathcal { T } }  T$ $( \theta , \delta \theta ) \mapsto \mathrm { R } _ { \theta } ( \delta \theta ) = \theta ^ { \prime }$ , and will be an important part of every iterative optimization algorithm. One possible retraction for $\tau$ we have already mentioned, namely the isometrization of TTN. After adding δθ to $\theta$ componentwise, the result is simply re-isometrized to project it back to $\tau$ . Apart from this, retracting each tensor except for the root individually is also a retraction on T since it is a Cartesian product of manifolds. Thus, we can use all retractions for the Stiefel manifold such as the Polar retraction $\mathrm { R } _ { X } ^ { \mathrm { p o l a r } } : T _ { X } \mathrm { S t } ( n , k ) \to \mathrm { S t } ( n , k ) , ( X , \delta X ) \mapsto U V ^ { T }$ with singular value decomposition of $\boldsymbol { X } + \delta \boldsymbol { X } = \boldsymbol { U } \boldsymbol { \Sigma } \boldsymbol { V } ^ { T }$ or the QR-retraction $\mathrm { R } _ { X } ^ { \mathrm { Q R } } : T _ { X } \mathrm { S t } ( n , k ) \to \mathrm { S t } ( n , k )$ ， $( X , \delta X ) \mapsto Q$ with QR-decomposition of $X + \delta X = Q R$

A computationally viable tool to move tangent vectors from point $\theta \in \mathcal { T }$ to a new point $\theta ^ { \prime } \in \mathcal { T }$ is vector transport. A vector transport for tangents $\xi \in T _ { \theta } \mathcal { T }$ is given by

$$
\begin{array} { r } { \nabla _ { \theta  \theta ^ { \prime } } ^ { \mathcal { T } } : T _ { \theta } \mathcal { T }  T _ { \theta ^ { \prime } } \mathcal { T } , \quad \nabla _ { \theta  \theta ^ { \prime } } ( \xi ) = \mathrm { P r o j } _ { \theta ^ { \prime } } ( \xi ) , } \end{array}\tag{6}
$$

and a suitable vector transport for $\xi \in H _ { \theta } \mathcal { T }$ reads

$$
\nabla _ { \theta  \theta ^ { \prime } } ^ { \mathcal { T } / \mathcal { G } } : H _ { \theta } \mathcal { T }  H _ { \theta ^ { \prime } } \mathcal { T } , \quad \nabla _ { \theta  \theta ^ { \prime } } ^ { H } ( \xi ) = \mathrm { P r o j } _ { \theta ^ { \prime } } ^ { H } ( \xi ) .\tag{7}
$$

This second vector transport is also compatible with the quotient, i.e. $\operatorname { D } \pi ( \theta ^ { \prime } ) \circ \mathbf { V } _ { \theta  \theta ^ { \prime } } ^ { T / \mathcal { G } } \circ \operatorname { l i f t } _ { \theta }$ is a vector transport on $\tau / \mathcal { G }$

## 2.1.5 Optimization function and Riemannian gradient

The last ingredient we need is the Euclidean gradient and a mapping to the Riemannian gradient. In this work, we consider optimization functions $f : \mathcal T \to$ R of the form $f = \mathcal { L } \circ \tau ,$ with a loss function $\mathcal { L } : \mathbb { R } ^ { d _ { 1 } \times \cdots \times d _ { n } \times \bar { d } _ { \mathrm { o u t } } } \longrightarrow \mathbb { R }$ . More specifically, we will do supervised learning following [14, 31], where a sample $x \in \mathbb { R } ^ { n }$ is encoded in a higher dimensional space through a feature map

$$
\phi \left( x \right) = \phi ^ { ( 1 ) } ( x _ { 1 } ) \otimes \cdots \otimes \phi ^ { ( n ) } ( x _ { n } ) ,
$$

that is composed of local feature maps $\phi ^ { ( i ) } : \mathbb { R }  \mathbb { R } ^ { d _ { i } }$ . The model response is defined as

$$
{ \phi } _ { j } ^ { ( N ) } = \tau ( \theta ) _ { i _ { 1 } , . . . , i _ { n } , j } \phi ( x ) _ { i _ { 1 } , . . . , i _ { n } } .
$$

For a labeled data set $( x ^ { k } , y ^ { k } ) _ { k = } ^ { m }$ and the squared error loss function, the optimization function takes the form $\begin{array} { r } { f ( \theta ) = \mathcal { L } ( \tilde { \mathbf { \Gamma } } ( \theta ) ) = \frac { 1 } { m } \sum _ { k } ^ { m } \| \Phi ( x ^ { k } , \theta ) - y ^ { k } \| ^ { 2 } } \end{array}$ , where $\Phi ( x , \theta ) = \phi ^ { ( N ) } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ To compute the Euclidean gradient, we can use the backpropagation algorithm that stores intermediate results in theforward propagation, i.e., the computation of the loss, that are then reused in the gradient calculation which uses the chain rule to propagate the gradient back. In the forward propagation, we want to compute the response $\phi ^ { ( N ) }$ for a given TTN θ without actually constructing $\tau ( \theta )$ . To this end, we compute

$$
\phi _ { \nu _ { t } } ^ { ( t ) } = B _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } \phi _ { \nu _ { t _ { L } } } ^ { ( t _ { L } ) } \phi _ { \nu _ { t _ { R } } } ^ { ( t _ { R } ) }\tag{8}
$$

recursively $\forall t \in T$ , starting from the leaf nodes (see Fig. 3a). All intermediate results $\phi ^ { ( t ) }$ $t \in T ^ { * }$ are stored as they are needed later for the backpropagation. The final result at the root node is equal to the response $\phi ^ { ( N ) }$ . With this we can also compute the derivative of the loss with respect to the response

$$
\bar { \phi } _ { j } ^ { ( N ) } : = \frac { \partial \mathcal { L } } { \partial \phi _ { j } ^ { ( N ) } } .
$$

For the backpropagation, we can apply the chain rule to Eq. 8, which yields

$$
\bar { B } _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } : = \frac { \partial \mathcal { L } } { \partial B _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } } = \bar { \phi } _ { \nu _ { t } } ^ { ( t ) } \phi _ { \nu _ { t _ { R } } } ^ { ( t _ { R } ) } \phi _ { \nu _ { t _ { L } } } ^ { ( t _ { L } ) }\tag{9}
$$

for the TTN tensors and

$$
\bar { \phi } _ { \nu _ { t _ { L } } } ^ { ( t _ { L } ) } : = \frac { \partial \mathcal { L } } { \partial \phi _ { \nu _ { t _ { L } } } ^ { ( t _ { L } ) } } = B _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } \bar { \phi } _ { \nu _ { t } } ^ { ( t ) } \phi _ { \nu _ { t _ { R } } } ^ { ( t _ { R } ) }\tag{10}
$$

$$
\bar { \phi } _ { \nu _ { t _ { R } } } ^ { ( t _ { R } ) } : = \frac { \partial \mathcal { L } } { \partial \phi _ { \nu _ { t _ { R } } } ^ { ( t _ { R } ) } } = B _ { \nu _ { t _ { L } } \nu _ { t _ { R } } \nu _ { t } } ^ { ( t ) } \bar { \phi } _ { \nu _ { t } } ^ { ( t ) } \phi _ { \nu _ { t _ { L } } } ^ { ( t _ { L } ) } .\tag{11}
$$

for the intermediate results, which now need to be computed in reverse order starting from the root node (see Fig. 3b). The derivatives of the input $\bar { \phi } ^ { ( i ) } , i = 1 , . . . , n$ are not needed and can be skipped. The full Euclidean gradient is then given by assembling the individual derivatives

$$
\begin{array} { r } { \nabla f ( \theta ) = ( \bar { B } ^ { ( t ) } ) _ { t \in T } \in \mathcal { E } . } \end{array}
$$

![](images/8edfad9230de083edc8e0d14e44b4e62e83e9bc5ec2b7c01708e2b8a83b0fb9f.jpg)  
Figure 3: a) Forward propagation. At each node t, the corresponding features $\phi ^ { ( t _ { L } ) }$ and $\phi ^ { ( t _ { R } ) }$ (here in red) are contracted with the tensor $B ^ { ( t ) }$ (in blue), which results in a new feature vector $\phi ^ { ( t ) } \left( \mathrm { E q . ~ } 8 \right)$ . This feature vector serves as one of the inputs for the parent node in the next layer, indicated by the gray arrow. The final result of the root node $\phi ^ { ( \bar { N } ) }$ (the response) is used to compute the loss ${ \mathcal { L } } .$ b) Backpropagation. The derivative of the loss w.r.t. the response $\bar { \phi } ^ { ( N ) }$ (depicted in dark blue) is inserted into Eq. 9-11 to compute $\bar { \phi } ^ { ( t _ { L } ) }$ (left), $\bar { B } ^ { ( t ) }$ (center, resulting in the orange tensor) and $\bar { \phi } ^ { ( t _ { R } ) }$ (right), where the intermediate results (red) are reused from the forward propagation. $\dot { \bar { \phi } } ( t _ { L } )$ and $\bar { \phi } ^ { ( t _ { R } ) }$ are propagated to the nodes in the lower layer where they are needed for the computation of $\bar { B } ^ { ( t ) }$ . The three orange tensors constitute the Euclidean gradient $\nabla f$

Note, that as an alternative to Equations (9–11), one can also use automatic differentiation to compute the Euclidean gradient.

Finally, the Riemannian gradient on $\tau$ is given by the projection of the Euclidean gradient $\nabla f$ to the tangent space, since $\tau$ is a Riemannian submanifold ofE: grad $f \left( \theta \right) = \mathrm { P r o j } _ { \theta } \left( \nabla f \left( \theta \right) \right)$ The Riemannian gradient grad $f ( \theta )$ is already the right update direction for Riemannian gradient descent formulated on $\tau$ , however it does not adequately represent the Riemannian gradient of the quotient space $\tau / \mathcal { G }$ with respect to metric $\hat { \rho }$ . This quotient gradient lifted to $\tau$ is actually equal to the projection of grad f to the horizontal space

$$
\operatorname { l i f t } _ { \boldsymbol { \theta } } ( \operatorname { g r a d } \hat { f } ( [ \boldsymbol { \theta } ] ) ) = \operatorname { P r o j } _ { \boldsymbol { \theta } } ^ { H } ( \operatorname { g r a d } f ( \boldsymbol { \theta } ) ) = \operatorname { P r o j } _ { \boldsymbol { \theta } } ^ { H } ( \nabla f ( \boldsymbol { \theta } ) ) ,\tag{12}
$$

with $\hat { f } : \mathcal { T } / \mathcal { G } \to \mathbb { R } , \hat { f } = \mathcal { L } \circ \hat { \tau }$ . The second identity holds since $H _ { \theta } \mathcal { T } \subset T _ { \theta } \mathcal { T }$ . Optimizing according to update directions (12) on T implicitly represents an optimization in $\mathcal { T } / \mathcal { G } \left[ 2 4 \right]$

## 2.2 Adaptive Riemannian optimization

Adaptive schemes in optimization on Euclidean spaces work by accumulating information of past iterates to coordinate-wisely rescale the gradient of the current iterate. Given an optimization function $f : \mathbb { R } ^ { n } $ R and an initial iterate $w _ { 0 } = ( w _ { 0 } ^ { ( 1 ) } , \dots , w _ { 0 } ^ { ( n ) } ) \in \mathbb { R } ^ { n }$ , adaptive schemes iterate according to

$$
w _ { k + 1 } ^ { ( i ) } = w _ { k } ^ { ( i ) } - \alpha h ( g _ { 0 } ^ { ( i ) } , \ldots , g _ { k } ^ { ( i ) } ) \cdot g _ { k } ^ { ( i ) }
$$

where ${ { g } _ { k } } = \nabla f \left( { { w } _ { k } } \right)$ and α is some step-size. ${ \mathsf { W h i l e } } - \alpha g _ { k } ^ { ( i ) }$ represents the plain standard gradient descent update direction, the real-valued function h introduces a coordinate-wise rescaling over past gradients. More advanced adaptive algorithms such as ADAM [5] also make use of stabilized update directions, replacing $- \alpha g _ { k } ^ { ( i ) }$ with $- a m _ { k } ^ { ( i ) }$ , where $m _ { k } = \beta m _ { k - 1 } + ( 1 - \beta ) g _ { k }$ is the momentum for some decay parameter $\ddot { \beta } \in ( 0 , 1 )$ . Those two strategies can yield enormous improvements over standard gradient descent, in particular in stochastic optimization settings where $g _ { k }$ are gradient estimates that are sparse but possibly instable.

Extending this mechanism to Riemannian optimization is no trivial feat, since manifolds do not come with an intrinsic coordinate representation. Here we focus on the techniques presented in [32], which apply to the case of a Riemannian manifold $( \mathcal { M } , \rho )$ , that factors into a Cartesian product of Riemannian manifolds $( \mathcal { M } ^ { ( i ) } , \rho ^ { ( i ) } )$

$$
\mathcal { M } = \bigvee _ { i = 1 } ^ { n } \mathcal { M } ^ { ( i ) } , \quad \rho = \sum _ { i = 1 } ^ { n } \rho ^ { ( i ) } .\tag{13}
$$

Points $w \in \mathcal { M }$ may be written as $w = ( w ^ { ( 1 ) } , \dots , w ^ { ( n ) } )$ , where $\boldsymbol { w } ^ { ( i ) } \in \mathcal { M } ^ { ( i ) }$ and tangent vectors $\delta w \in T _ { w } \mathcal { M }$ separate as $\delta w = ( \delta w ^ { ( 1 ) } , \dots , \delta w ^ { ( n ) } )$ , where $\delta w ^ { ( i ) } \in T _ { w ^ { ( i ) } } \mathcal { M } ^ { ( i ) }$ . The individual components serve as block coordinates in adaptive Riemannian schemes, so one iterates

$$
w _ { k + 1 } = \mathrm { R } _ { w _ { k } } \left( ( - \alpha h ( g _ { 0 } ^ { ( i ) } , \ldots , g _ { k } ^ { ( i ) } ) \cdot g _ { k } ^ { ( i ) } ) _ { i = 1 , \ldots , n } \right) ,
$$

where $g _ { k } = ( g _ { k } ^ { ( 1 ) } , . . . , g _ { k } ^ { ( n ) } ) = \mathrm { g r a d } f ( w _ { k } )$ . To ensure compatibility with the manifold structure, the update direction goes through a retraction R on $\mathcal { M } _ { : }$ , and of course h has to be adapted to act on block-coordinates instead. This framework additionally allows the incorporation of momentum-based updates by employing a vector transport V on $\mathcal { M }$ via

$$
m _ { k } = \beta \nabla _ { w _ { k - 1 }  w _ { k } } ( m _ { k - 1 } ) + ( 1 - \beta ) g _ { k } ,
$$

which is commonly used in Riemannian versions of ADAM [32].

Notably, the above block-coordinate formulas also captures the MUON optimizer [6], which is an adaptive optimizer for the hidden layers of neural networks. It comprehends each weight matrix $\bar { W } ^ { ( i ) }$ and corresponding gradient $\dot { \boldsymbol { G } } ^ { ( i ) }$ as a separate block-coordinate, effectively imposing a Cartesian product of matrix spaces $\mathcal { M } ^ { ( i ) } = \mathbb { R } ^ { \bar { \chi } _ { i } ^ { \mathrm { { i n } } } \times \chi _ { i } ^ { \mathrm { o u t } } }$ , where $( \chi _ { i } ^ { \mathrm { i n } } , \chi _ { i } ^ { \mathrm { o u t } } )$ is the dimension of the layer i. MUON introduces adaptiveness by orthogonalizing each update matrix $G ^ { ( i ) }$ individually using the polar decomposition. In its most basic form it iterates according to

$$
\boldsymbol { W } _ { k + 1 } ^ { ( i ) } = \boldsymbol { W } _ { k } ^ { ( i ) } - \alpha \boldsymbol { \mathrm { q f } } ( \boldsymbol { G } _ { k } ^ { ( i ) } ) ,
$$

where $\operatorname { q f } ( \cdot )$ refers to the orthogonal factor of the polar decomposition. This update formula manages without a retraction because the considered matrix spaces are Euclidean.

## 2.3 Riemannian distance over gradients

Tuning the learning rate α for stochastic optimizers to a given machine learning task can represent a considerable overhead, both when developing and deploying models. Distance over gradients (DOG) [7] provides a tuning-free dynamic step-size formula for stochastic gradient descent and it was extended to optimization on manifolds in [33]. Concretely, on a Riemannian manifold $( \mathcal { M } , \rho )$ with sectional curvature $\kappa ,$ a step-size schedule is given by

$$
\alpha _ { k } = \frac { r _ { k } } { \sqrt { \zeta _ { \kappa } ( r _ { k } ) \sum _ { k ^ { \prime } = 0 } ^ { k } | | g _ { k ^ { \prime } } | | _ { \rho } ^ { 2 } } } , \quad r _ { k } = \operatorname* { m a x } _ { k ^ { \prime } < k } d \bigl ( w _ { 0 } , w _ { k ^ { \prime } } \bigr )\tag{14}
$$

where $g _ { k } = \operatorname { g r a d } f \left( w _ { k } \right)$ . The expression $d ( x _ { 0 } , x _ { k } )$ is the geodesic distance between the initial and k-th iterate. The geometric curvature function $\zeta _ { \kappa }$ is included to accommodate for manifolds of negative sectional curvature and it reads

$$
\zeta _ { \kappa } ( r ) = { \left\{ \begin{array} { l l } { \displaystyle { \frac { r { \sqrt { | \kappa | } } } { \operatorname { t a n h } ( r { \sqrt { | \kappa | } } ) } } } & { { \mathrm { i f ~ } } \kappa < 0 , } \\ { 1 } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Formula (14) is applicable to stochastic Riemannian gradient descent $x _ { k + 1 } = \mathrm { R } _ { x _ { k } } ( - \alpha _ { k } g _ { k } )$ of a geodesically convex function $f$ . In practical application, the numerator of (14) is taken at least $ { \varepsilon } \in ( 0 , 1 )$ , which represents an initial estimate of the step-size and is used to avoid stepsizes of magnitude zero. While this may seem like an additional hyperparameter, numerical experiments show that DOG-like algorithms react very insensitively to the choice of ϵ [33]. Importantly, DOG requires the evaluation of the geodesic distance defined as

$$
d ( x , y ) = \operatorname* { m i n } _ { \gamma } L ( \gamma ) = \operatorname* { m i n } _ { \gamma } \int _ { a } ^ { b } \rho _ { \gamma ( s ) } ( \gamma ^ { \prime } ( s ) , \gamma ^ { \prime } ( s ) ) ^ { \frac { 1 } { 2 } } \mathrm { d } s ,
$$

which is the minimum length of all curves $\gamma : [ a , b ]  \mathcal { M }$ connecting $\gamma ( a ) = x$ and $\gamma ( b ) = y$ and the minimizer will be a geodesic on $\mathcal { M }$ . We quickly recapitulate results that exists for the Stiefel and Grassmann manifolds. A closed form solution for geodesic distances on $S \mathrm { t } ( n , k )$ does not exist, but [25] provides a lower bound. For matrices X, $Y \in S { \mathfrak { t } } ( n , k )$ it holds that

$$
d ^ { \mathrm { S t } } ( X , Y ) \geqslant q _ { k } ( \| X - Y \| _ { F } ) : = 2 \sqrt { k } \arcsin \left( \frac { \| X - Y \| _ { F } } { 2 \sqrt { k } } \right) .\tag{15}
$$

The sectional curvature of the Stiefel manifold is bounded from below by $- { \frac { 1 } { 2 } } \left[ 3 4 \right.$ , Thm. 10].

The Grassmann manifold forms as the quotient $\mathrm { G r } ( n , k ) = \mathrm { S t } ( n , k ) / \mathrm { O } ( k )$ of the Stiefel manifold with the orthogonal group. It hence contains equivalence classes $[ X ] = \{ X Q \colon Q \in \mathbf { O } ( k ) \}$ of matrices $X \in S \mathrm { t } ( n , k )$ whose columns form an orthonormal basis for the same k-dimensional subspace of $\mathbb { R } ^ { n }$ . An expression for the geodesic distance [35, Section 5.1] in terms of Stiefel representatives X, $Y \in S { \mathfrak { t } } ( n , k )$ of $[ X ] , [ Y ] \in { \mathrm { G r } } ( n , k )$ reads

$$
d ^ { \mathrm { G r } } ( [ X ] , [ Y ] ) = \sqrt { \sum _ { i } \operatorname { a r c c o s } ( \sigma _ { i } ( X ^ { T } Y ) ) ^ { 2 } } .
$$

Here, $\sigma _ { i }$ refers to the i-th singular value. The sectional curvature of the Grassmann manifold is non-negative [35, Prop. 12].

## 3 Tools

## 3.1 Adaptive optimization of TTNs

The developments of Section 2.2 directly apply to optimization on the TTN-manifold $\tau .$ Equations (1) and (4) together attest that $( \tau , \rho )$ fits into the framework (13), where the individual core tensors $B ^ { ( t ) }$ of the TTN $\theta = \left( B ^ { ( t ) } \right) _ { t \in T }$ are comprehended as block coordinates for adaptive Riemannian schemes. Tangent vectors $\xi = ( \delta B ^ { ( t ) } ) _ { t \in T } , \eta = ( \delta C ^ { ( t ) } ) _ { t \in T }$ in $T _ { \theta } \mathcal { T }$ also represent block-coordinate variations and the metric $\rho$ acts separably on those variations.

Optimization on the quotient $\tau / \mathcal { G }$ however does not immediately comply with the proposed mechanism of adaptivity, as $\tau / \mathcal { G }$ intrinsically couples all nodes of the network and does not factor into a Cartesian product of manifolds [24]. Still any element $[ \theta ] \in T _ { \theta } { \mathcal { T } } / { \mathcal { G } }$ can be identified with a representative $\theta = ( B ^ { ( t ) } ) _ { t \in T }$ , that reads as a tuple of block coordinates. Whats more, both the quotient tangents and the quotient metric can be lifted uniquely to the total space, where they decompose as tensor-wise expressions:

$$
\mathrm { l i f t } _ { \theta } \big ( \hat { \xi } \big ) = \big ( \delta B ^ { ( t ) } \big ) _ { t \in T } \in H _ { \theta } \mathcal { T } , \quad \hat { \rho } _ { [ \theta ] } \big ( \hat { \xi } , \hat { \eta } \big ) = \rho _ { \theta } \big ( \mathrm { l i f t } _ { \theta } \big ( \hat { \xi } \big ) , \mathrm { l i f t } _ { \theta } \big ( \hat { \eta } \big ) \big ) = \sum _ { t \in T } \bigl \langle \delta B ^ { ( t ) } \big | \delta C ^ { ( t ) } \big \rangle .
$$

Thus, the practical application of adaptive schemes on $\tau / \mathcal { G }$ is just as straightforward as it is on $\tau$ , since optimization on $\tau / \mathcal { G }$ happens only implicitly anyway through representatives $\theta \in \mathcal T$ and horizontal lifts of the quotient gradient (12).

In the next section we will present several algorithms, like Riemannian ADAM or Riemannian MUON, that employ adaptive schemes in the quotient optimization setting, to which theoretical convergence guarantees like [32, Theorem 1 & 2] can be extended. In order to highlight how this may be done, we show convergence of Riemannian ADAGRAD in Appendix B. Being one of the simplest adaptive algorithms, this allows us to put particular focus on the peculiarities involving the TTN quotient.

## 3.2 Distance over gradients on TTN-manifolds

Both for optimization on $\tau$ and optimization $\tau / \mathcal { G }$ , the main obstacle to achieve DOG-like algorithmic is the computation of the geodesic distance.

We begin our considerations for $\tau _ { \ast }$ , where we again use that it factorizes as a Cartesian product of manifolds equipped with a product metric. For two points $\theta \ = \ ( B ^ { ( t ) } ) _ { t \in T }$ and $\theta ^ { \prime } = ( C ^ { ( t ) } ) _ { t \in T }$ it holds that

$$
d ^ { T } ( \theta , \theta ^ { \prime } ) ^ { 2 } = \sum _ { t \in T ^ { * } } d ^ { S t } ( B ^ { ( t ) } , C ^ { ( t ) } ) ^ { 2 } + \vert \vert B ^ { ( N ) } - C ^ { ( N ) } \vert \vert _ { F } ^ { 2 } ,\tag{16}
$$

so using (15), the geodesic distance on $\tau$ can be bounded by

$$
d ^ { T } ( \theta , \theta ^ { \prime } ) \geqslant \sqrt { \sum _ { t \in T ^ { * } } q _ { k _ { t } } ( | | B ^ { ( t ) } - C ^ { ( t ) } | | _ { F } ) ^ { 2 } + | | B ^ { ( N ) } - C ^ { ( N ) } | | _ { F } ^ { 2 } } ,
$$

which we will use as a proxy for DOG-like algorithms. Compared to standard DOG, this will yield smaller, more conservative step-sizes, because we are underestimating $r _ { k }$ in (14).

Concerning geodesic distances on $\mathcal { T } / \mathcal { G }$ , for two points $[ \theta ] , [ \theta ^ { \prime } ] \in \mathcal { T } / \mathcal { G }$ we consider a connecting geodesic ${ \hat { \gamma } } : [ 0 , 1 ]  { \mathcal { T } } / { \mathcal { G } } _ { : }$ , as well as its horizontal lift γ to $\tau$ , that satisfies $\gamma ^ { \prime } ( s ) \in H _ { \gamma ( s ) } \mathcal { T }$ for all $s \in [ 0 , 1 ]$ . By writing $\boldsymbol { \gamma } ( 0 ) = \boldsymbol { \theta } = ( B ^ { ( t ) } ) _ { t \in T } , \boldsymbol { \gamma } ( 1 ) = \boldsymbol { \theta } ^ { \prime } = \big ( C ^ { ( t ) } \big ) _ { t \in T }$ it holds that

$$
d ^ { T / \mathcal { G } } ( [ \theta ] , [ \theta ^ { \prime } ] ) ^ { 2 } = \sum _ { t \in T ^ { \ast } } d ^ { \mathrm { G r } } ( [ B ^ { ( t ) } ] , [ C ^ { ( t ) } ] ) ^ { 2 } + | | B ^ { ( N ) } - C ^ { ( N ) } | | _ { F } ^ { 2 } .\tag{17}
$$

For a detailed derivation, see Appendix $\mathsf { A } ,$ where we use the fact that $\tau / \mathcal { G }$ shares its geodesic spray with a Cartesian product of Grassmann manifolds

$$
\mathcal { Z } = \Bigg ( \bigcup _ { t \in T ^ { * } } \mathrm { G r } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } ) \Bigg ) \times \mathbb { R } ^ { \chi _ { N _ { L } } \chi _ { N _ { R } } \times d _ { \mathrm { o u t } } } .\tag{18}
$$

There is still a caveat for the practical application of (17). It only holds for a specific representation $\theta ^ { \prime } \in \mathcal T$ of $[ \theta ^ { \prime } ] \in \mathcal { T } / \mathcal { G }$ , concretely for $\theta ^ { \prime } = \gamma ( 1 )$ , the endpoint of the horizontal lift of the geodesic connecting rθs and $[ \theta ^ { \prime } ]$ , to which we do not have access in general. Indeed, picking a different representative $\tilde { \theta } ^ { \prime }$ with $\pi ( \tilde { \theta } ^ { \prime } ) = [ \theta ^ { \prime } ]$ will in general produce a different result when evaluating (17). Note however that the application of DOG algorithms (14) only requires the geodesic distances between iterates, which transform into each other through repeated application of retractions. Using the exponential map (A.1) on $\tau / \mathcal { G }$ as a retraction ensures that iterates remain in compatible representations. On the other hand, any other retraction produces first-order approximations of geodesics on the quotient, so it is safe to assume that the representations we obtain do not deviate far from their "correct" geodesic representation. Thus, equation (17) still applies as an approximation for arbitrary retractions.

## 4 Algorithms

## 4.1 RADAM

The Riemannian ADAM algorithm (RADAM) for TTNs is presented as Algorithm 1 and obtained as an application of [32, Alg. 1] to TTNs. Importantly, it covers both optimization on the total space $\tau ,$ as well as optimization on the quotient $\tau / \mathcal { G }$ . Optimization on the quotient happens only implicitly by using representatives and update directions from the total space and is achieved by picking the tuple of projector and vector transport as $( \mathrm { P } , \mathrm { V } ) = ( \mathrm { P r o j } ^ { H } , \mathrm { V } ^ { \overline { { T } } / \mathcal { G } } )$ according to (7). By instead picking $( \mathsf { P } , \mathsf { V } ) = ( \mathsf { P r o j } , \mathsf { V } ^ { \mathcal { T } } )$ as in (6), the quotient structure is ignored and the iteration happens entirely in the total space. Here, $f _ { k }$ are function oracles of function $f$ given by evaluating the loss only on a random batch of training data.

![](images/8ee7f4c86ecb801f2bf086b4cb074af13cd9771e592719d8611f783efa0243a1.jpg)  
Figure 4: Even though $\tau / \mathcal { G }$ does not factor, iterates $[ \theta ]$ and update directions $\hat { \xi }$ can be lifted to a point $\boldsymbol { \theta } = \big ( \boldsymbol { B } ^ { ( t ) } \big ) _ { t \in T } \in \mathcal { T }$ and a direction $\xi = ( B ^ { ( t ) } ) _ { t \in T } \in H _ { \theta } \mathcal { T }$ that factors. Furthermore, geodesics $\hat { \gamma } \in \mathcal { T } / \mathcal { G }$ ascend to horizontal paths in $( \gamma _ { t } ) _ { t \in T } \in \mathcal { T }$ which, at non-root nodes, descend to geodesics $\tilde { \gamma } _ { t } \in \mathrm { G r } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } )$ . No quotient is taken at the root node, which resides in a Euclidean space, and where geodesics simplify to straight lines anyway.

Algorithm 1 RADAM for TTNs Algorithm 2 RDOG for TTNs   
Require: $\theta _ { 1 } = ( B _ { 1 } ^ { ( t ) } ) _ { t \in T } \in \mathcal { T } ;$ , learning rate $\alpha ,$ Require: $\boldsymbol { \theta _ { 1 } } = ( \boldsymbol { B } _ { 1 } ^ { ( t ) } ) _ { t \in T } \in \mathcal { T } ;$ step-size esti  
decay parameters $\beta _ { 1 } , \beta _ { 2 } \in ( 0 , 1 )$ , projector mate $\varepsilon ,$ projector $\mathrm { \Delta P _ { \perp } }$ , retraction $\mathrm { R , }$ geodesic   
${ \mathrm { ~ P ~ } } ,$ retraction ${ \mathrm { R } } ,$ vector transport V distances $( d ^ { ( t ) } ) _ { t \in T }$   
Set $m _ { 0 } = ( 0 ) _ { t \in T } , \nu _ { 0 } = ( 0 ) _ { t \in T }$ Set $r _ { 0 } = ( \varepsilon ) _ { t \in T } , n _ { 0 } = ( 0 ) _ { t \in T } ,$   
for $k = 1 , 2 , \dots$ do for $k = 1 , 2 , \dots$ do   
$( G _ { k } ^ { ( t ) } ) _ { t \in T } = \mathrm { P } ( \nabla f _ { k } ( \theta _ { k } ) )$ $( G _ { k } ^ { ( t ) } ) _ { t \in T } = \mathrm { P } ( \nabla f _ { k } ( \theta _ { k } ) )$   
for $t \in T$ do for $t \in T$ do   
$\begin{array} { r l } {  { \overline { { m _ { k } ^ { ( t ) } } } = \beta _ { 1 } m _ { k - 1 } ^ { ( t ) } + ( 1 - \beta _ { 1 } ) G _ { k } ^ { ( t ) } } ~ } & { { } } \end{array}$ $\bar { n _ { k + 1 } ^ { ( t ) } } = n _ { k } ^ { ( t ) } + \langle G _ { k } ^ { ( t ) } \mid G _ { k } ^ { ( t ) } \rangle$   
$\nu _ { k } ^ { ( t ) } = \beta _ { 2 } \nu _ { k - 1 } ^ { ( t ) } + ( 1 - \underline { { { \beta _ { 1 } } } } ) \langle G _ { k } ^ { ( t ) } \mid G _ { k } ^ { ( t ) } \rangle$ $\begin{array} { r } { \delta B _ { k } ^ { ( t ) } = - r _ { k } ^ { ( t ) } G _ { k } ^ { ( t ) } / \sqrt { \zeta _ { \kappa } ( r _ { k } ^ { ( t ) } ) n _ { k } ^ { ( t ) } } } \end{array}$   
$\delta B _ { k } ^ { ( t ) } = - { \alpha } m _ { k } ^ { ( t ) } / { \sqrt { \nu _ { k } ^ { ( t ) } } }$ end for   
end for $\theta _ { k + 1 } = \mathrm { R } _ { \theta _ { k } } \big ( \big ( \delta B _ { k } ^ { ( t ) } \big ) _ { t \in T } \big )$   
$\theta _ { k + 1 } = \mathrm { R } _ { \theta _ { k } } \big ( \big ( \delta B _ { k } ^ { ( t ) } \big ) _ { t \in T } \big )$ $m _ { k + 1 } = \mathrm { V } _ { \theta _ { k } \to \theta _ { k + 1 } } ( m _ { k } )$ for $r _ { k + 1 } ^ { ( t ) } = \operatorname* { m a x } \{ r _ { k } ^ { ( t ) } , d ^ { ( t ) } ( B _ { 0 } ^ { ( t ) } , B _ { k } ^ { ( t ) } ) \}$ $t \in T$ do   
end for end for   
end for

## 4.2 RDOG

Algorithm 2 presents an adaptive version of Riemannian distance over gradients (RDOG) for TTNs. It is similar to layer-wise DOG [7, Section $4 ]$ , in that it keeps a separate tracker of distance $r _ { k } ^ { ( t ) }$ for each core tensor $t \in T$ , as opposed to a single distance for the whole tensor network. Since both for (16) and (17), the total geodesic distances form as the norm of node-wise distances, adaptive RDOG employs exactly those node-wise distances denoted by $( d ^ { ( t ) } ) _ { t \in T }$ . For our experiments, we pick $d ^ { ( N ) } ( B ^ { ( t ) } , C ^ { ( t ) } ) = \| B ^ { ( t ) } - C ^ { ( t ) } \| _ { F }$ at the root tensor, but differ in

$$
\begin{array} { r } { d ^ { ( t ) } ( B ^ { ( t ) } , C ^ { ( t ) } ) = \left\{ \begin{array} { l l } { d ^ { \mathrm { G r } } ( \pi ^ { \mathrm { G r } } ( B ^ { ( t ) } ) , \pi ^ { \mathrm { G r } } ( C ^ { ( t ) } ) ) } & { \mathrm { o n ~ } T / \mathcal { G } , } \\ { q _ { k _ { t } } ( \| B ^ { ( t ) } - C ^ { ( t ) } \| _ { F } ) } & { \mathrm { o n ~ } T , } \end{array} \right. \quad \mathrm { f o r ~ a l l ~ } t \in T ^ { * } . } \end{array}\tag{19}
$$

Like for RADAM, the projector P is chosen accordingly, depending on whether we optimize on $\tau$ or $\tau / \mathcal { G }$

## 4.3 RMUON

We present two Riemannian versions of the MUON optimizer for TTNs. Algorithm 3 combines MUON updates with momentum, whereas Algorithm 4 feeds MUON updates into a DOG-like scheme.

Recall that MUON takes steps according to orthogonal parts of gradient components. Since T forms as a Cartesian product of matrix manifolds, that is, all tensors in a tree tensor network can canonically be reshaped into a matrices, node-wise gradients may be orthogonalized according to the polar decomposition to mimic MUON behavior. These orthogonalized directions are even compatible with the quotient $\tau / \mathcal { G } \colon$ As seen in [27] for any $\delta X \in H _ { X } { \cal S } \mathrm { t } ( n , k )$ , it holds that $\mathrm { q f } ( \delta X ) \in H _ { X } \mathrm { S t } ( n , k )$ as well. Thus, when picking $( \mathrm { P } , \mathrm { V } ) = ( \mathrm { P r o j } ^ { H } , \mathrm { V } ^ { T / \mathcal { G } } )$ according to (7), all update steps of Algorithms $3 \ \& \ 4$ will reside in $H _ { \theta _ { k } } \mathcal { T }$ and descend to well-formed updates in $T _ { [ \theta _ { k } ] } \mathcal { T } / \mathcal { G }$ . Again, by employing $( \mathrm { P } , \mathrm { V } ) = ( \mathrm { P r o j } , \mathrm { V } ^ { \mathcal { T } } )$ as in (6) instead, the iteration happens entirely in $\tau _ { \ast }$ , ignoring the quotient.

Algorithm 3 RMUON for TTNs Algorithm 4 RMUONDOG for TTNs   
Require: $\theta _ { 1 } = ( B _ { 1 } ^ { ( t ) } ) _ { t \in T } \in \mathcal { T }$ , learning rate α, Require: $\theta _ { 1 } = ( B _ { 1 } ^ { ( t ) } ) _ { t \in T } \in \mathcal { T } .$ , step-size esti  
decay parameter $\beta _ { 1 } \in ( 0 , 1 )$ , projector P , re- mate $\varepsilon ,$ projector P , retraction R, geodesic   
traction R, vector transport V distances $( d ^ { ( t ) } ) _ { t \in T }$   
Set $m _ { 0 } = { \left( 0 \right) } _ { t \in T }$ Set $r _ { 0 } = ( \varepsilon ) _ { t \in T } , n _ { 0 } = ( 0 ) _ { t \in T }$   
for $k = 1 , 2 , \ldots$ . do for $k = 1 , 2 , \dots$ do   
$( G _ { k } ^ { ( t ) } ) _ { t \in T } = \mathrm { P } ( \nabla f _ { k } ( \theta _ { k } ) )$ $( G _ { k } ^ { ( t ) } ) _ { t \in T } = \mathrm { P } ( \nabla f _ { k } ( \theta _ { k } ) )$   
for $t \in T$ do for $t \in T$ do   
$\begin{array} { r l } {  { \overline { { m _ { k } ^ { ( t ) } } } = \beta _ { 1 } m _ { k - 1 } ^ { ( t ) } + ( 1 - \beta _ { 1 } ) G _ { k } ^ { ( t ) } } ~ } & { { } } \end{array}$ $\overset { \vartriangle } { \boldsymbol { n } _ { k + 1 } ^ { ( t ) } } = \boldsymbol { n } _ { k } ^ { ( t ) } + \langle \mathbf { q } \mathbf { f } ( \boldsymbol { G } _ { k } ^ { ( t ) } ) \vert \mathbf { q } \mathbf { f } ( \boldsymbol { G } _ { k } ^ { ( t ) } ) \rangle$   
$\delta B _ { k } ^ { ( t ) } = - \alpha \mathsf { q f } ( m _ { k } ^ { ( t ) } )$ $\delta B _ { k } ^ { ( t ) } = - r _ { k } ^ { ( t ) } \mathsf { q f } ( G _ { k } ^ { ( t ) } ) / \sqrt { \zeta _ { \kappa } ( r _ { k } ^ { ( t ) } ) } n _ { k } ^ { ( t ) }$   
end for end for   
$\theta _ { k + 1 } = \mathrm { R } _ { \theta _ { k } } \big ( \big ( \delta B _ { k } ^ { ( t ) } \big ) _ { t \in T } \big )$ $\theta _ { k + 1 } = \mathrm { R } _ { \theta _ { k } } \big ( \big ( \delta B _ { k } ^ { ( t ) } \big ) _ { t \in T } \big )$   
$m _ { k } = \mathrm { V } _ { \theta _ { k }  \theta _ { k + 1 } } ( m _ { k } )$ for $t \in T$ do   
end for $\grave { r } _ { k + 1 } ^ { ( \bar { t } ) } = \operatorname* { m a x } \{ r _ { k } ^ { ( t ) } , d ^ { ( t ) } ( B _ { 0 } ^ { ( t ) } , B _ { k } ^ { ( t ) } ) \}$   
end for   
end for

## 5 Numerical Experiments

To numerically evaluate the proposed methods, they are implemented using the pytorch framework, which allows for automatic differentiation to calculate Euclidean gradients $\nabla f _ { k } ( \theta )$ of the objective function on random mini-batches. The objective $f = \mathcal { L } \circ \tau$ is given by the squared error loss and the batch size is fixed to 256 throughout our considerations. Initial parameters $\theta _ { 1 } \in \mathcal { T }$ were found using the unsupervised construction algorithm introduced in [29]. On a given architecture, the same initial iterate was used for all optimizers. Learning

![](images/7b09226337d69fd215d3c09c2d3dc6752c2fd364a72b944d4da46f82b4e6fd41.jpg)

![](images/9d1a0ad675a231b8f4f3b101ca5ae9eb3a1d60aaa99a25c07d97abece6cb2c4c.jpg)  
Figure 5: Comparison of the different Riemannian optimizers and the conventional ADAM optimizer through loss (solid) and accuracy (dashed) over epochs (left) and time (right) for the Fashion MNIST dataset.

rates/ step-size estimates were found by conducting a simple grid-search as summarized in the following table, whereas momentum-related hyperparameters remain fixed at $\beta _ { 1 } = 0 . 9 _ { \mathrm { : } }$ $\beta _ { 2 } = 0 . 9 9 9$
<table><tr><td>dataset</td><td> $\alpha ^ { \mathrm { A D A M } }$ </td><td> $\alpha ^ { \mathrm { R A D A M } }$ </td><td> $\alpha ^ { \mathrm { R M U O N } }$ </td><td> $\varepsilon ^ { \mathrm { R D O G } }$ </td><td> $\varepsilon ^ { \mathrm { R M U O N D O G } }$ </td></tr><tr><td>Fashion MNIST</td><td>0.001</td><td>0.03</td><td>0.007</td><td>0.1</td><td>0.1</td></tr><tr><td>CIFAR10</td><td>0.001</td><td>0.02</td><td>0.005</td><td>0.1</td><td>0.05</td></tr><tr><td>Imagenette</td><td>0.001</td><td>0.02</td><td>0.005</td><td>0.02</td><td>0.05</td></tr></table>

For Riemannian optimizers, we componentwisely use the recently proposed POGO algorithm [26] as retraction. It can be understood as an approximation to the polar retraction, that does not enforce strict orthogonality every step, but instead remains close to the orthogonal manifold at all times, in favor of cheaper and more GPU-friendly iteration rules. Furthermore, we omit a detailed discussion of Riemannian algorithms that ignore the quotient structure: As seen in Fig. 11 in the appendix, they tend to perform very similarly but slightly worse than their quotient counterparts. All experiments were conducted on a NVIDIA A30 GPU in single-precision.

## 5.1 Fashion MNIST

As a first test, we use the Fashion MNIST dataset [36] that contains 60000 training- and 10000 test images with $2 8 \times 2 8$ pixels and one channel each. As in [20, 37], we compress all images to $1 6 \times 1 6$ pixels to reduce the number of features of the TTN model, as we will use each pixel directly as an input feature. The feature map we use to encode the pixels is given by [37]

$$
\phi ^ { ( i ) } ( x _ { i } ) = \frac { 1 } { \sqrt { 1 + x _ { i } ^ { 2 } } } \binom { 1 } { x _ { i } } ,
$$

meaning that $n = 2 5 6$ and $d = d _ { 1 } , . . . , d _ { n } = 2$ . For the Fashion MNIST dataset, we choose a maximum bond dimension of $\chi _ { \mathrm { m a x } } = 1 6$

Fig. 5 shows the results of the different Riemannian optimizers and the conventional ADAM optimizer over 30 training epochs. Note that ADAM does not take the quotient structure into account and simply works with the unconstrained tensors of E. After 30 epochs, all optimizers achieve approximately the same test accuracy of „ 89%. The lowest training loss was achieved by RMUONDOG, followed closely by RADAM, ADAM & RMUON. RDOG shows the slowest convergence behavior, but the final test accuracy is still comparable. In terms of runtime, ADAM is clearly in the lead, although the Riemannian optimizers perform competitively.

![](images/b81fc5fd26da0f46ff09f2446e8128bc09083ac7966132f36ea3ba92991e3ab7.jpg)  
(a) After three epochs.

![](images/5e588f3e24cbe4ad4743694653d14318796cdbe37d310c222cecec5b98ec8e9f.jpg)  
(b) After four epochs.  
Figure 6: Test accuracy for the Fashion MNIST dataset using compressed TTN models that were trained with different optimizers. The retention ratio of the compression is defined as the ratio of kept parameters to the original number of parameters. While the models that were trained with Riemannian optimizers in $\tau$ can be heavily compressed without a significant drop of the test accuracy, this is not the case for the model trained with ADAM. Furthermore, one can see that the problem gets worse with each epoch, as the total norm of the TTN continues to grow and the TTN moves further away from the orthogonal initial parametrization.

At first glance, it seems as if optimizing on $\tau$ and taking the quotient structure into account does not bring a significant advantage, as ADAM outperforms most of the Riemannian optimizers without the additional overhead of projections and retractions. However, for many downstream tasks such as model compression or the computation of explainability measures the TTN must be orthogonal [18]. Thus, the TTN optimized by ADAM must first be orthogonalized, which accumulates the norm of all non-isometric parts of the tensors in the root tensor. Indeed Riemannian optimizers keep the total norm<sup>2</sup> $| | \tau ( \theta ) | | _ { F }$ of the TTN model $\theta$ stable during training, while the norm explodes when optimizing with ADAM, exceeding the numerical range of float32 after just six epochs (see Fig. 12 in appendix). For inference this is not a problem since one can simply work with the numerically stable version $\theta \in { \mathcal { E } }$ that was learned. However, orthogonalizing this TTN can lead to numerical instabilities, which incur problems in downstream tasks. Fig. 6 demonstrates this using the example of model compression [18], which results in significantly worse compression accuracies for ADAM when compared to the models that were optimized in $\tau$ . An explanation for this behavior is the fact that the compression algorithm is based on singular value decompositions of the orthogonality center, which for ADAM is numerically unstable and possibly ill-conditioned due to the exploding norm. Another example for a downstream task would be the computation of feature entropies (see e.g. [18, 16]). It is algorithmically closely related to model compression and applying it to final iterates obtained by ADAM can be expected to yield unreliable values.

## 5.2 CIFAR10 & Imagenette

For the CIFAR10 [38] and Imagenette [39] datasets, we use a hybrid neural network - tensor network architecture, where the neural network acts as a feature extractor and the tensor network as the classification head. The main reason for this is that a full tensor network is not able to generalize well as it lacks the translational invariance that is necessary for this dataset. To preserve the explainability of the tensor network classifier, we need to make sure that the extracted features are meaningful and interpretable, which is not the case for standard neural network backbones. Thus, we divide the input image into $p \times p$ patches, that are processed separately by the same convolutional neural network (CNN). This produces an embedding vector of dimension $d = 6 4$ for each patch of the image, that will serve as the input for the tensor network classifier (see Fig. 7). The important distinction to conventional CNN feature extractors is that no information is exchanged between the individual patches, such that we can clearly trace back each feature vector to a well-defined region of the image. The exchange of information between the patches happens entirely in the tensor network, where one can analyze the learned patterns in a rigorous and precise manner (see e.g. [18, 16]). The exact architecture used for both datasets is summarized in the following table.

![](images/d3955330f51c1679642fc4c378dcd69a01161ec07fd4a15d552d1d1d91164aa1.jpg)  
Figure 7: CNN-TTN hybrid architecture as employed for CIFAR10 and Imagenette datasets. The CNN is applied to each patch separately and acts as a feature map for the TTN head.

<table><tr><td rowspan=1 colspan=1>dataset</td><td rowspan=1 colspan=1>input res.</td><td rowspan=1 colspan=1>patches p</td><td rowspan=1 colspan=1>conv. layers</td><td rowspan=1 colspan=1>sites n</td><td rowspan=1 colspan=1>d</td><td rowspan=1 colspan=1> $\chi _ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>tree-depth</td></tr><tr><td rowspan=3 colspan=1>Fashion MNISTCIFAR10Imagenette</td><td rowspan=3 colspan=1> $1 6 \times 1 6$  $3 2 \times 3 2$  $1 6 0 \times 1 6 0$ </td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>None</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=2 colspan=1> $4 \times 4$  $8 \times 8$ </td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>12</td><td rowspan=2 colspan=1>46</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>32</td></tr></table>

In all experiments, the CNN feature extractor is optimized simultaneously with the TTN. For the CNN itself, we employ the ADAM optimizer with standard parameters $( \alpha = 0 . 0 0 1 , \beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9 )$ . In order to enrich the empirical data distribution, we augmented training images through random cropping, random erasing, and random horizontal flips.

Inspecting Figs. 8 & 9 yields similar conclusions as for the Fashion MNIST dataset: Riemannian optimizers perform on par with ADAM in terms of test accuracy, with every optimizer reaching just above „ 80% for CIFAR10 and just below „ 80% for Imagenette, but ADAM is faster as it does not require projections or retractions. The adverse effect of ADAM on downstream compression is considerably less severe in the case of CIFAR10, as seen in Fig. 10. We attribute this to the fact, that we used a smaller tensor network architecture for CIFAR10 than for Imagenette and Fashion-MNIST. In fact, the tree for the latter is twice as deep as for the former, which possibly amplifies numerical problems for algorithms that heavily rely on hierarchical structure of the network. This argument is supported by compression accuracies recovered from the model trained for Imagenette. Its larger depth renders the final iterate from ADAM incompressible.

![](images/6e787e9dae0ac288cfeee8817ffea02b09d17417d9df4fe5a8132c522aef34b4.jpg)

![](images/a17f06c4b5f8297ae19c33d56ec7a25155fbbc79ed71b6932bad71d0ba75d54a.jpg)  
Training Time (s)

Figure 8: Comparison of optimizers through loss (solid) and accuracy (dashed) over epochs (left) and time (right) on the CIFAR10 dataset.  
![](images/0388d7f11d870615bb4ee66ec93c88bc61123adb489347309555323e8a9f6c51.jpg)

![](images/3c8a23921a7a7612ae17b8312279a33315e81a7843ea0f490594e3138f00e56e.jpg)  
Figure 9: Comparison of optimizers through loss (solid) and accuracy (dashed) over epochs (left) and time (right) on the Imagenette dataset.

![](images/dcb5b2510749ea632dfa14c51e8187f29b18d4fad8b25b4271731a31a718bc60.jpg)

![](images/8c20ab113b5993526b2692d190324c9003a8fcb719bd53293fb35da172794991.jpg)  
Figure 10: Test accuracy for the CIFAR10 (left) and Imagenette (right) using compressed models that were trained for 30 epochs using different optimizers. Compressing the iterate obtained from ADAM is not possible: The model is rendered useless through orthogonalization alone.

## 6 Conclusion

In this work, we developed the theoretical framework for stochastic optimization on the manifold of orthogonal TTN for machine learning applications. Furthermore, we introduced a novel hybrid CNN-TTN framework that preserves the key advantages of tensor networks such as explainability and compressibility, while enabling the efficient processing of large-scale images. We compared different stochastic Riemannian optimizers against the conventional unconstrained Adam optimizer on three different image classification tasks. While the Riemannian optimizers are comparable to Adam in terms of convergence and generalization, the Riemannian optimizers learn a more numerically stable version for downstream tasks.

The main problem with unconstrained optimization, especially for deep TTN, is that the final iterates can no longer be orthogonalized correctly. Most tensor network algorithms, including the computation of explainability measures and model compression, however, are based on orthogonalization. Since these methods are the key advantages of tensor networks over neural networks, the computational and algorithmic overhead of Riemannian optimization is usually justified. Furthermore, by exploiting a fast approximate POGO retraction for the Stiefel manifold that profits from GPU acceleration, we were able to significantly reduce the computational overhead. In general, there is still room for improvement in the computational efficiency of the Riemannian optimizers, and it remains to be seen whether the gap to unconstrained optimization can be further narrowed so that the additional cost becomes negligible. Further analysis is also necessary in order to understand the origins of the numerical problems that arise from unconstrained optimization, and whether they can be remedied. If, for example, the instabilities of the orthogonalization can be traced back exclusively to the exploding norm of the root tensor, weight decay or other regularization methods could greatly improve stability. Another possibility would be to develop alternative algorithms for downstream tasks that do not rely on orthogonalization.

Another interesting research direction would be to develop stochastic optimization schemes for TN with adaptive bond dimensions, similar to two-site DMRG methods that are widely used in quantum simulation. Not only would this essentially combine optimization and compression into one process, but it also further reduces the number of hyperparameters that need to be tuned for the TTN and is shown to help escape local minima in ground state search[40]. It is likely that this would aid generalization in the machine learning setting, as the model is incentivized to only use the truly relevant information. The main theoretical obstacle is that adapting the bond dimensions during optimization changes the manifold structure, so conventional Riemannian optimization methods do not apply anymore.

Finally, combining the expressive power of neural networks with the interpretability and flexibility of tensor networks has great potential, well beyond image classification. Tensornetwork methods have already been explored in settings ranging from high-energy data analysis [16, 17] and SAR classification [18] to optimization problems in radiotherapy [41]. Expanding this approach to other applications and architectures could be a big step towards powerful yet transparent AI technologies.

## Acknowledgements

The authors would like to thank Tensor AI Solutions GmbH for supporting this work and providing the code framework that allowed the numerical evaluation of our findings.

Author contributions M.S. and M.W. contributed in equal parts to this work. M.W. is responsible for the theoretical developments; M.S. devised the hybrid CNN-TTN architecture and conducted the numerical experiments. A.U. supervised the mathematical formulation and the theoretical foundations. T.F. and M.T. coordinated the overall work and provided the technical background and software on TN-machine learning. All authors contributed to the conceptualization and revision of the work.

Reproducibility statement To support reproducibility and verification, the full code used to evaluate our numerical findings will be released upon journal acceptance.

Funding information The work of M.S., M.T., M.W. and T.F. was supported by Tensor AI Solutions GmbH. The work of A.U. was supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Projektnummer 506561557.

## A Geodesics of the TTN quotient

The TTN quotient is closely related to the Cartesian product $\mathcal { Z }$ of Grassmannians defined in (18). Indeed, $\mathcal { Z }$ is a group quotient to $\tau$ , with the same Lie group (2) as employed for $\tau / \mathcal { G }$ , but to a different Lie group action: one that acts tensorwise instead of coupling all tensors. We define the quotient map

$$
\pi ^ { \mathcal { Z } } : \mathcal { T } \to \mathcal { Z } , \qquad ( B ^ { ( t ) } ) _ { t \in T } \mapsto \left\{ \begin{array} { l l } { \pi ^ { \mathrm { G r } } ( B ^ { ( t ) } ) } & { \mathrm { f o r } t \in T ^ { \ast } , } \\ { B ^ { ( t ) } } & { \mathrm { e l s e } , } \end{array} \right.
$$

where $\pi ^ { \mathrm { G r } } : { \mathrm { S t } } ( n , k ) \to { \mathrm { G r } } ( n , k )$ is the Grassmann quotient map. Importantly, this means that the vertical spaces corresponding to $\mathcal { Z }$ and $\tau / \mathcal { G }$ differ, but their horizontal distributions can be defined to coincide by taking the Cartesian horizontal space (3) in both cases. Based on this, we can derive an important relation between $\tau / \mathcal { G }$ and $\mathcal { Z } _ { : }$ , as seen in the following theorem.

Theorem 1. Any geodesic $\hat { \gamma } : [ 0 , 1 ] \to \mathcal { T } / \mathcal { G }$ can be mapped to a geodesic $\tilde { \gamma } : [ 0 , 1 ] \to \mathcal { Z }$ of the same length, in such a way that both geodesics share the same representation in $\tau$ , meaning there exists a horizontal curve $\gamma : [ 0 , 1 ]  \mathcal { T }$ that satisfies $\pi ( \gamma ( s ) ) = \hat { \gamma } ( s )$ and $\pi ^ { \mathcal Z } ( \gamma ( s ) ) = \tilde { \gamma } ( s )$ for all $s \in [ 0 , 1 ]$

Proof. Let ${ \hat { \gamma } } ( 0 ) = [ \theta ]$ and $\hat { \gamma } ( 1 ) = [ \theta ^ { \prime } ]$ . Since geodesics are characterized as curves that locally minimize length, we assume that $L ( \hat { \gamma } ) = d ^ { \mathcal { T } / \mathcal { G } } ( [ \theta ] , [ \theta ^ { \prime } ] )$ . If this assumption is not met, subdivide $\hat { \gamma }$ into sufficiently short segments that each fulfill $\mathrm { i t } ,$ , and apply the upcoming argument to all segments. According to [42, Prop. II.3.1] for any representative $\theta \in \mathcal { T }$ of $[ \theta ]$ , there exists a unique horizontal lift $\gamma \ { \mathrm { o f } } \ \hat { \gamma }$ which satisfies ${ \hat { \gamma } } ( s ) = \pi ( \gamma ( s ) ) , \gamma ( 0 ) = \theta$ and $\bar { \gamma } ^ { \prime } ( s ) = \operatorname* { l i f t } _ { \gamma ( s ) } ( \hat { \gamma } ^ { \prime } ( s ) ) \in H _ { \gamma ( s ) } \mathcal { T }$ . Since horizontal lifts of curves preserve lengths [43, Lem. 26.11.1], the curve $\gamma$ will be the solution of $\begin{array} { r } { \operatorname* { m i n } _ { \gamma \mathrm { h o r i z o n t a l } } L ( \gamma ) } \end{array}$ . By the form of the horizontal space (3), for $\gamma = \left( \gamma _ { ( t ) } \right) _ { t \in T }$ , each individual component except for the root is horizontal, meaning $\gamma _ { ( t ) } ^ { \prime } ( s ) \in H _ { \gamma _ { ( t ) } ( s ) } S \mathrm { t } ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } )$ . As $\tau$ forms as a Cartesian product (meaning that in particular the Riemannian metric decouples) and since there are no coupling constraints between components of $\gamma _ { \ast }$ each non-root component will be the solution of min $\dot { \gamma } _ { ( t ) }$ horizontal $L ( \gamma _ { ( t ) } )$ with the corresponding endpoints and the root component of $\boldsymbol { \gamma }$ will solve min ${ \ L } _ { Y ( N ) } { \ L } L ( Y _ { ( N ) } )$ . Thus, the $\boldsymbol { \gamma } _ { ( t ) }$ of each non-root node is a horizontal geodesic in $\mathbf { S t } ( \boldsymbol { \chi } _ { t _ { L } } \boldsymbol { \chi } _ { t _ { R } } , \boldsymbol { \chi } _ { t } )$ . Since $H _ { \gamma _ { ( t ) } ( s ) } \mathrm { S t }$ is orthogonal to the vertical space associated with the Grassmannian, $\pi ^ { \mathrm { G r } } ( \gamma _ { ( t ) } )$ is a geodesic in $\mathrm { G r } \big ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } \big )$ [43, Lemma 26.11.4]. Applying this logic to all non-root components, shows that $\tilde { \gamma } ( s )$ is a geodesic in ${ \mathcal { Z } } ,$ and its length evaluates as $L ( \tilde { \gamma } ) = L ( \gamma )$ □

At this point, we remind of the nuance concerning the full multilinear rank constraints of $\tau / \mathcal { G }$ (see Section 2.1.3): Not all geodesics in $\mathcal { Z }$ lift to geodesics in $\tau$ that satisfy it, meaning they have no correspondence in $\tau / \mathcal { G }$ . Thus the above argument cannot be reversed. Still, the equivalence of geodesics can also be constructed with the following high-level argument. If ∇ is the Levi-Civita connection on $\tau$ , then the Levi-Civita connection $\nabla ^ { \mathcal { Z } }$ on $\mathcal { Z }$ is given by

$$
\operatorname* { l i f t } ^ { { \mathcal { Z } } } \bigl ( \nabla _ { U } ^ { { \mathcal { Z } } } V \bigr ) = \mathrm { P r o j } ^ { H } \bigl ( \nabla _ { \mathrm { l i f t } } { \mathcal { z } } _ { ( U ) } \mathrm { l i f t } ^ { { \mathcal { Z } } } \bigl ( V \bigr ) \bigr )
$$

for $U , V$ vector fields on $\mathcal { Z }$ . This follows since $H _ { \theta } \mathcal { T }$ is orthogonal to the vertical spaces corresponding to $\mathcal { Z } \ [ 4 4$ , Thm. 9.34]. The same formula also yields a connection $\nabla ^ { \bar { T } / \bar { g } }$ on $\tau / \mathcal { G }$ given by

$$
\operatorname { l i f t } ( \boldsymbol { \nabla } _ { U } ^ { \mathcal { T } / \mathcal { G } } V ) = \operatorname { P r o j } ^ { H } \bigl ( \boldsymbol { \nabla } _ { \mathrm { l i f t } ( U ) } \mathrm { l i f t } ( V ) \bigr )
$$

for $U , V$ vector fields on $\mathcal { T } / \mathcal { G } \left[ 2 4 \right.$ , Thm. 3]. Even though this connection has torsion, because geodesics only depend on the symmetric part of a connection, the geodesic spray of this connection will coincide with the Levi-Civita one of $\tau / \mathcal { G }$ . Thus it is clear that $\tau / \mathcal { G }$ and $\mathcal { Z }$ share the same geodesics, given through the same horizontal parametrization in $\tau$ .

This argument also implies that geodesic distances and exponential maps of $\tau / \mathcal { G }$ and $\mathcal { Z }$ share representations in $\tau .$ . Indeed, an important corollary of the above theorem is equation (17). It can be found by employing that

$$
L \left( \hat { \gamma } \right) ^ { 2 } = L \left( \gamma \right) ^ { 2 } = \sum _ { t \in T ^ { * } } L \left( \gamma _ { \left( t \right) } \right) ^ { 2 } + L \left( \gamma _ { \left( N \right) } \right) ^ { 2 } ,
$$

and plugging in the respective expressions for geodesic distances. Furthermore, we can conclude that the exponential map on $\tau / \mathcal { G }$ is related to the Grassmann exponential map, which in term of Stiefel representatives reads

$$
\mathrm { E x p } _ { X } ^ { \mathrm { G r } } ( \delta X ) = X V \cos ( \Sigma ) V ^ { T } + U \sin ( \Sigma ) V ^ { T } ,
$$

where $\delta X = U \Sigma V ^ { T }$ is the compact SVD of $\delta X \in T _ { X } S \mathrm { t } ( n , k )$

Corollary 2. Let $\theta = ( B ^ { ( t ) } ) _ { t \in T }$ be a representative of $[ \theta ] \in { \mathcal { T } } / { \mathcal { G } }$ and let the horizontal lift of $\hat { \xi } \in T _ { [ \theta ] } \mathcal { T } / \mathcal { G }$ be given by lif $\mathsf { \bar { t } } _ { \theta } \big ( \hat { \xi } \big ) = \big ( \delta B ^ { ( t ) } \big ) _ { t \in T }$ . The exponential map on $\tau / \mathcal { G }$ reads

$$
\begin{array} { r } { \mathrm { E x p } _ { [ \theta ] } = \pi \circ \mathrm { R } _ { \theta } ^ { \mathrm { E x p } } \circ \mathrm { l i f t } _ { \theta } , } \end{array}\tag{A.1}
$$

where

$$
\begin{array} { r } { \mathbf { R } _ { \theta } ^ { \mathrm { E x p } } ( \operatorname* { l i f t } _ { \theta } ( \hat { \xi } ) ) = \left\{ \begin{array} { l l } { \mathrm { E x p } _ { B ^ { ( t ) } } ^ { \mathrm { G r } } ( \delta B ^ { ( t ) } ) } & { \mathrm { i f ~ } t \in T ^ { * } , } \\ { B ^ { ( t ) } + \delta B ^ { ( t ) } } & { \mathrm { e l s e } . } \end{array} \right. } \end{array}
$$

Proof. Theorem 1 shows that when fixing a base point $\theta _ { i }$ , any geodesic $\hat { \gamma } : [ 0 , 1 ] \to \mathcal { T } / \mathcal { G }$ can be mapped to a geodesic $\tilde { \gamma } : [ 0 , 1 ] \to \mathcal { Z } _ { : }$ , such that both geodesics share the same representative $\gamma : [ 0 , 1 ]  T$ at all points, meaning $\pi ( \gamma ( s ) ) = \hat { \gamma } ( s )$ and $\pi ^ { \mathrm { G r } } ( \gamma _ { \left( t \right) } ( s ) ) = \tilde { \gamma } _ { \left( t \right) } ( s )$ for all $s \in [ 0 , 1 ]$ Furthermore if $\hat { \gamma } ^ { \prime } ( 0 ) = \hat { \xi }$ , then $\gamma ^ { \prime } ( 0 ) = \mathrm { l i f t } _ { \theta } ( \hat { \xi } ) = ( \delta B ^ { ( t ) } ) _ { t \in T }$ and $\tilde { \gamma } _ { ( t ) } ^ { \prime } ( 0 ) = \mathrm { D } \pi ^ { \mathrm { G r } } ( B ^ { ( t ) } ) [ \delta B ^ { ( t ) } ]$ for all $t \in T ^ { * }$ . Because $\tilde { \gamma }$ is a geodesic, it holds that $\tilde { \gamma } _ { ( t ) } ( 1 ) = \mathrm { E x p } _ { B ^ { ( t ) } } \big ( \mathrm { D } \pi ^ { \mathrm { G r } } ( B ^ { ( t ) } ) [ \delta B ^ { ( t ) } ] \big )$ By lifting all components, we see that $\gamma ( 1 ) \ = \ \mathsf { R } _ { \theta } ^ { \mathrm { E x p } } ( ( \delta B ^ { ( t ) } ) _ { t \in T } ) \ = \ \mathsf { R } _ { \theta } ^ { \mathrm { E x p } } ( \operatorname* { l i f t } _ { \theta } ( \hat { \xi } ) )$ . Finally, going back to the TTN quotient, it holds that $\hat { \gamma } ( 1 ) \ : = \ : \pi ( \mathrm { R } _ { \theta } ^ { \mathrm { E x p } } ( \mathrm { l i f t } _ { \theta } ( \hat { \xi } ) ) )$ . Thus by definition $\pi \circ \mathsf { R } _ { \theta } ^ { \mathrm { E x p } } \circ \mathsf { l i f t } _ { \theta }$ is the exponential map on $\tau / \mathcal { G }$ □

## B Convergence of RADAGRAD

Let $\hat { f } : \mathcal { T } / \mathcal { G } \to \mathbb { R } , \hat { f } = \mathcal { L } \circ \hat { \tau }$ that is accessible through function oracles $\hat { f } _ { k }$ by evaluating the loss only on a random batch of training data. In this section we present a proof of convergence for RADAGRAD on the TTN quotient manifold, by bounding the regret

$$
R _ { K } = \sum _ { k = 1 } ^ { K } \hat { f } _ { k } ( [ \theta _ { k } ] ) - \hat { f } _ { k } ( [ \theta _ { * } ] ) ,
$$

where $\begin{array} { r } { [ \theta _ { * } ] = \mathrm { a r g m i n } _ { [ \theta ] \in \mathcal { T } / \mathcal { G } } \sum _ { k = 1 } ^ { K } \hat { f } _ { k } ( [ \theta ] ) } \end{array}$ is an oracle minimizer . The proof can be extended to cover more advanced algorithms like RADAMNC or RAMSGRAD by going along the lines of [32]. Here we focus on the less involved RADAGRAD to more clearly state the peculiarities concerning quotient optimization. We make the following assumptions:

Algorithm 5 RADAGRAD for TTNs   
Require: $\theta _ { 1 } = ( B _ { 1 } ^ { ( t ) } ) _ { t \in T } \in \mathcal { T } .$ , learning rate $\alpha ,$ projector $\mathrm { P }$ , retraction R   
Set $\nu _ { 0 } = ( 0 ) _ { t \in T }$   
for $k = 1 , 2 , \dots$ do   
$( G _ { k } ^ { ( t ) } ) _ { t \in T } = \mathrm { P } ( \nabla f _ { k } ( \theta _ { k } ) )$   
for $t \in T$ do   
$\nu _ { k } ^ { ( t ) } = \bar { \nu _ { k - 1 } ^ { ( t ) } } + \langle G _ { k } ^ { ( t ) } \mid G _ { k } ^ { ( t ) } \rangle$   
$\delta B _ { k } ^ { ( t ) } = - \alpha G _ { k } ^ { ( t ) } / \sqrt { \nu _ { k } ^ { ( t ) } }$   
end for   
$\theta _ { k + 1 } = \mathrm { R } _ { \theta _ { k } } \big ( \big ( \delta B _ { k } ^ { ( t ) } \big ) _ { t \in T } \big )$   
end for

(A1) The step-size α of Algorithm 5 satisfies $\alpha \| G _ { 1 } ^ { ( t ) } \| / \sqrt { \nu _ { 1 } ^ { ( t ) } } \leqslant \frac { \pi } { 2 }$ for all $t \in T ^ { * }$ and $G _ { k } ^ { ( t ) }$ remains bounded for $t \in T$

(A2) Algorithm 5 runs employing the projector $\mathrm { \sf P } = \mathrm { \sf P r o j } ^ { \sf H }$ and the retraction ${ \mathrm { R } } = { \mathrm { R } } ^ { \mathrm { E x p } }$ that descends to the exponential map on $\tau / \mathcal { G }$

(A3) $[ \theta _ { * } ]$ and all iterates $[ \theta _ { k } ]$ of Algorithm 5 reside in a compact and geodesically convex subset of $\tau / \mathcal { G }$ with diameter bounded by $D _ { \infty }$ , and $\hat { f }$ is geodesically convex on that subset.

A few comments about the validity of the assumptions are in order. Assumption (A1) essentially requires that all gradients remain bounded, which is a standard assumption for regretbased convergence proofs. In this case, a step size $\alpha > 0$ can be found that satisfies the first condition. Assumption (A2) considers the exponential map as a retraction. Since other retractions on $\tau / \mathcal { G }$ actually match with the exponential map up to second order, Assumption (A2) is still satisfied in approximation for those. Finally, the existence of geodesically convex subsets on $\tau / \mathcal { G }$ on which $\hat { f }$ is geodesically convex as required in (A3) can be guaranteed at least locally around points in which the Riemannian Hessian Hess $\hat { f }$ is positive definite (see [44] for a definition of the Riemannian Hessian). In the case of strongly convex loss functions $\mathcal { L }$ like the $\mathcal { L } _ { 2 }$ loss, an interesting case arises if the oracle minimizer $[ \theta _ { * } ]$ provides a critical point of ${ \mathcal { L } } .$ The following result is in analogy to [45, Theorem 2.11].

Proposition 3. Let $\hat { f } = \mathcal { L } \circ \hat { \tau } : \mathcal { T } / \mathcal { G } \to$ R and let $\mathcal { L } : \mathbb { R } ^ { d _ { 1 } \times \cdots \times d _ { n } \times d _ { \mathrm { o u t } } } \longrightarrow$ R be a twice differentiable strongly convex function. Then Hess $\hat { f } ( [ \theta _ { * } ] )$ is positive definite at any point $[ \theta _ { * } ] \in \mathcal { T } / \mathcal { G }$ satisfying $\nabla \mathcal { L } ( \hat { \tau } ( [ \theta _ { * } ] ) ) = 0$

Proof. Define the space $\mathcal { H } : = \hat { \tau } ( \mathcal { T } / \mathcal { G } ) \subset \mathbb { R } ^ { d _ { 1 } \times \dots \times d _ { n } \times d _ { \mathrm { o u t } } }$ as the manifold of tensors with fixed hierarchical rank $\left( \chi _ { t } \right) _ { t \in T ^ { * } } \left[ 2 8 \right]$ . Let ${ \hat { \gamma } } : [ 0 , 1 ] \to \mathcal { T } / \mathcal { G }$ be a geodesic on the quotient with $\hat { \gamma } ( 0 ) = [ \theta _ { * } ]$ , and define $\Gamma = \hat { \tau } \circ \hat { \gamma }$ the respective curve in low-rank tensor manifold ${ \mathcal { H } } ,$ so we have the equality $\hat { f } \circ \hat { \gamma } = \mathcal { L } \circ \Gamma$ . Then according to Taylor’s theorem on Riemannian manifolds [44, Eq. 5.26], it holds that

$$
\begin{array} { r } { ( \hat { f } \circ \hat { \gamma } ) ^ { \prime \prime } ( 0 ) = \hat { \rho } _ { \hat { \gamma } ( 0 ) } ( \mathrm { H e s s } \hat { f } ( \hat { \gamma } ( 0 ) ) [ \hat { \gamma } ^ { \prime } ( 0 ) ] , \hat { \gamma } ^ { \prime } ( 0 ) ) + \hat { \rho } _ { \hat { \gamma } ( 0 ) } ( \mathrm { g r a d } \hat { f } ( \hat { \gamma } ( 0 ) ) , \hat { \gamma } ^ { \prime \prime } ( 0 ) ) , } \end{array}
$$

where $\hat { \rho }$ denotes the quotient metric (5). Because $\hat { \gamma }$ is a geodesic, its velocity is constant, and the second term vanishes. We apply Taylor’s theorem again to find

$$
\begin{array} { r } { ( \mathcal { L } \circ \Gamma ) ^ { \prime \prime } ( 0 ) = \langle \Gamma ^ { \prime } ( 0 ) | \mathrm { H e s s } \mathcal { L } ( \Gamma ( 0 ) ) | \Gamma ^ { \prime } ( 0 ) \rangle + \langle \mathrm { g r a d } \mathcal { L } ( \Gamma ( 0 ) ) | \Gamma ^ { \prime \prime } ( 0 ) \rangle . } \end{array}
$$

Here, in slight abuse of notation Hess $\mathcal { L }$ (resp. grad L) denotes the Riemannian Hessian (resp. gradient) of $\mathcal { L }$ restricted to $\mathcal { H } .$ . Recall that $\nabla \mathcal { L } ( \hat { \tau } ( [ \theta _ { * } ] ) ) = 0$ . Since $\mathcal { H }$ is an embedded submanifold of $\mathbb { R } ^ { d _ { 1 } \times \dots \times d _ { n } \times d _ { \mathrm { o u t } } }$ equipped with the Euclidean metric [28], we can conclude from [44,

Prop. 3.61 that grad $\mathcal { L } ( \Gamma ( 0 ) ) = 0$ as well and from [44, Eq. 5.34] that Hess $\mathcal { L }$ coincides with the Euclidean Hessian $\mathrm { H } _ { \mathit { L } }$ on the respective tangent space of H at $\Gamma ( 0 )$ . By assumption, $\mathrm { H } _ { \mathit { L } }$ is positive definite, therefore Hess ${ \hat { f } } ( { \hat { \gamma } } ( 0 ) )$ is positive definite as well. □

Theorem 4. Let the iterates $\theta _ { 1 } , \theta _ { 2 } , \ldots , \theta _ { K } \in \mathcal { T }$ generated by Algorithm 5 be representatives of $[ \theta _ { 1 } ] , [ \theta _ { 2 } ] , \ldots , [ \theta _ { K } ] \in { \mathcal { T } } / { \mathcal { G } }$ . Under assumptions (A1-A3), the regret on the quotient is bounded by

$$
R _ { K } = \sum _ { k = 1 } ^ { K } \hat { f } _ { k } ( [ \theta _ { k } ] ) - \hat { f } _ { k } ( [ \theta _ { * } ] ) \leqslant \left( \frac { D _ { \infty } ^ { 2 } } { 2 \alpha } + \alpha \right) \sum _ { t \in { T } } \sqrt { \sum _ { k = 1 } ^ { K } \| { G } _ { k } ^ { ( t ) } \| ^ { 2 } } .
$$

For $K  \infty$ , this implies convergence of objective values in expectation.

Proof. Consider the iterates $[ \theta _ { k } ] , [ \theta _ { k + 1 } ]$ and the oracle minimizer $[ \theta _ { * } ]$ Their representatives are denoted by $\theta _ { k } = { \big ( } B _ { k } ^ { ( t ) } { \big ) } _ { t \in T } , \theta _ { k + 1 } = { \big ( } B _ { k + 1 } ^ { ( t ) } { \big ) } _ { t \in T }$ and $\theta _ { * } = ( B _ { * } ^ { ( t ) } ) _ { t \in T }$ . It holds that $\theta _ { k + 1 } = \mathrm { R } _ { \theta _ { k } } \big ( \big ( - \alpha G _ { k } ^ { ( t ) } / ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } \big ) _ { t \in T } \big )$ , where $( G _ { k } ^ { ( t ) } ) _ { t \in T } = \operatorname { l i f t } _ { \theta _ { k } }$ pgrad $\hat { f } ( [ \theta _ { k } ] ) )$ . Let $\hat { \gamma } : [ 0 , 1 ]  \mathcal { T } / \mathcal { G }$ be a geodesic connecting $[ \theta _ { k } ]$ and $[ \theta _ { * } ]$ . We denote $\gamma$ the horizontal lift of $\hat { \gamma }$ , and we set $( \Delta _ { k } ^ { ( t ) } ) _ { t \in T } = \gamma ^ { \prime } ( 0 ) = \operatorname* { l i f t } _ { \theta _ { k } } \big ( \hat { \gamma } ^ { \prime } ( 0 ) \big ) \in H _ { \theta _ { k } } \mathcal { T }$ . By employing the geodesic convexity of $\hat { f }$ and lifting the metric we find that

$$
\sum _ { k = 1 } ^ { K } \hat { f } _ { k } ( [ \theta _ { k } ] ) - \hat { f } _ { k } ( [ \theta _ { * } ] ) \leqslant \sum _ { k = 1 } ^ { K } \hat { \rho } _ { [ \theta _ { k } ] } ( - \mathrm { g r a d } \hat { f } ( [ \theta _ { k } ] ) , \hat { \gamma } ^ { \prime } ( 0 ) ) = \sum _ { k = 1 } ^ { K } \sum _ { t \in T } \bigl \langle - G _ { k } ^ { ( t ) } | \Delta _ { k } ^ { ( t ) } \rangle .\tag{B.1}
$$

We now let $[ \boldsymbol { B } _ { k } ^ { ( t ) } ] = \pi ^ { \mathrm { G r } } ( \boldsymbol { B } _ { k } ^ { ( t ) } ) , [ \boldsymbol { B } _ { k + 1 } ^ { ( t ) } ] = \pi ^ { \mathrm { G r } } ( \boldsymbol { B } _ { k + 1 } ^ { ( t ) } )$ and $[ B _ { * } ^ { ( t ) } ] = \pi ^ { \mathrm { G r } } ( B _ { * } ^ { ( t ) } )$ . As seen in Theorem $1 _ { - }$ , any non-root component of $\boldsymbol { \gamma }$ descends to a geodesic $\pi ^ { \mathrm { G r } } ( \gamma _ { ( t ) } )$ on the Grassmannian $\mathrm { G r } \big ( \chi _ { t _ { L } } \chi _ { t _ { R } } , \chi _ { t } \big )$ , therefore

$$
\begin{array} { r l } & { [ B _ { * } ^ { ( t ) } ] = \mathrm { E x p } _ { [ B _ { k } ^ { ( t ) } ] } ( \mathrm { D } \pi ^ { \mathrm { G r } } ( B _ { k } ^ { ( t ) } ) [ \Delta _ { k } ^ { ( t ) } ] ) , } \\ & { [ B _ { k + 1 } ^ { ( t ) } ] = \mathrm { E x p } _ { [ B _ { k } ^ { ( t ) } ] } ( - \alpha / ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } \mathrm { D } \pi ^ { \mathrm { G r } } ( B _ { k } ^ { ( t ) } ) [ G _ { k } ^ { ( t ) } ] ) . } \end{array}
$$

The second equality follows from the special choice of our retraction (A2). The geometric interpretation of the metric then yields

$$
\begin{array} { r l } & { \langle - \alpha / ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } G _ { k } ^ { ( t ) } | \Delta _ { k } ^ { ( t ) } \rangle = \rho _ { [ B _ { k } ^ { ( t ) } ] } ^ { \mathrm { G r } } ( - \alpha / ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } \mathrm { D } \pi ^ { \mathrm { G r } } ( B _ { k } ^ { ( t ) } ) [ G _ { k } ^ { ( t ) } ] , \mathrm { D } \pi ^ { \mathrm { G r } } ( B _ { k } ^ { ( t ) } ) [ \Delta _ { k } ^ { ( t ) } ] ) } \\ & { \qquad = d ^ { \mathrm { G r } } ( [ B _ { k } ^ { ( t ) } ] , [ B _ { k + 1 } ^ { ( t ) } ] ) d ^ { \mathrm { G r } } ( [ B _ { k } ^ { ( t ) } ] , [ B _ { * } ^ { ( t ) } ] ) \cos \Bigl ( \angle G _ { k } ^ { ( t ) } \Delta _ { k } ^ { ( t ) } \Bigr ) . } \end{array}
$$

The Grassmann manifold is of non-negative sectional curvature [35, Prop. 4.1]. Thus, the cosine law known from Euclidean spaces directly extends, and we can write

$$
\begin{array} { r } { 2 \langle - \alpha / ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } G _ { k } ^ { ( t ) } | \Delta _ { k } ^ { ( t ) } \rangle \leqslant d ^ { \mathrm { G r } } ( [ B _ { k } ^ { ( t ) } ] , [ B _ { * } ^ { ( t ) } ] ) ^ { 2 } - d ^ { \mathrm { G r } } ( [ B _ { k + 1 } ^ { ( t ) } ] , [ B _ { * } ^ { ( t ) } ] ) ^ { 2 } + d ^ { \mathrm { G r } } ( [ B _ { k } ^ { ( t ) } ] , [ B _ { k + 1 } ^ { ( t ) } ] ) ^ { 2 } } \\ { = d ^ { \mathrm { G r } } ( [ B _ { k } ^ { ( t ) } ] , [ B _ { * } ^ { ( t ) } ] ) ^ { 2 } - d ^ { \mathrm { G r } } ( [ B _ { k + 1 } ^ { ( t ) } ] , [ B _ { * } ^ { ( t ) } ] ) ^ { 2 } + \alpha ^ { 2 } / \nu _ { k } ^ { ( t ) } \langle G _ { k } ^ { ( t ) } | G _ { k } ^ { ( t ) } \rangle . } \end{array}
$$

In the last equality, we used Assumption (A1), and that the injectivity radius of the Grassmann manifold is $\textstyle { \frac { \pi } { 2 } } . ( \mathrm { A 1 } )$ applies to all iterates instead of only the first one, as $\| G _ { k } ^ { ( t ) } \| / \sqrt { \nu _ { k } ^ { ( t ) } }$ is a decreasing sequence. At the root node, which resides in a linear space, a similar consideration holds. Using the notation from (19), we can subsume

$$
\begin{array} { r l } & { \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { t \in T } \langle - G _ { k } ^ { ( t ) } | \Delta _ { k } ^ { ( t ) } \rangle \leqslant \displaystyle \sum _ { k = 1 } ^ { K } \sum _ { t \in T } \frac { ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } } { 2 \alpha } ( d ^ { ( t ) } ( B _ { k } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } - d ^ { ( t ) } ( B _ { k + 1 } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad + \frac { \alpha } { 2 ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } } \| G _ { k } ^ { ( t ) } \| ^ { 2 } . } \end{array}\tag{B.2}
$$

Recall that $\begin{array} { r } { d ^ { ( t ) } ( B _ { k } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } \leqslant \sum _ { t \in T } d ^ { ( t ) } ( B _ { k } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } = d ^ { 7 / \mathcal { G } } ( [ \theta _ { k } ] , [ \theta _ { * } ] ) ^ { 2 } \leqslant D _ { \infty } ^ { 2 } } \end{array}$ by (A3), which we can use to further bound the first term on the right side of (B.2). Since $\nu _ { 0 } ^ { ( t ) } = 0$ and $\nu _ { k } ^ { ( t ) }$ is monotonically increasing, we can conclude that

$$
\begin{array} { r l } & { \displaystyle \sum _ { t \in T } \displaystyle \sum _ { k = 1 } ^ { K } \frac { ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } } { 2 \alpha } ( d ^ { ( t ) } ( B _ { k } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } - d ^ { ( t ) } ( B _ { k + 1 } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } ) } \\ & { = \displaystyle \frac { 1 } { 2 \alpha } \sum _ { t \in T } \displaystyle \sum _ { k = 1 } ^ { K } ( ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } - ( \nu _ { k - 1 } ^ { ( t ) } ) ^ { 1 / 2 } ) d ^ { ( t ) } ( B _ { k } ^ { ( t ) } , B _ { * } ^ { ( t ) } ) ^ { 2 } } \\ & { \leqslant \displaystyle \frac { 1 } { 2 \alpha } \sum _ { t \in T } \displaystyle \sum _ { k = 1 } ^ { K } ( ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } - ( \nu _ { k - 1 } ^ { ( t ) } ) ^ { 1 / 2 } ) D _ { \infty } ^ { 2 } } \\ & { = \displaystyle \frac { D _ { \infty } ^ { 2 } } { 2 \alpha } \sum _ { t \in T } ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } = \frac { D _ { \infty } ^ { 2 } } { 2 \alpha } \sum _ { t \in T } \displaystyle \sqrt { \sum _ { k = 1 } ^ { K } \| G _ { k } ^ { ( t ) } \| ^ { 2 } } . } \end{array}\tag{B.3}
$$

For the second term in (B.2), Auer’s lemma [46] yields that

$$
\sum _ { t \in T } \sum _ { k = 1 } ^ { K } \frac { \alpha } { 2 ( \nu _ { k } ^ { ( t ) } ) ^ { 1 / 2 } } \| G _ { k } ^ { ( t ) } \| ^ { 2 } = \frac { \alpha } { 2 } \sum _ { t \in T } \sum _ { k = 1 } ^ { K } \frac { \| G _ { k } ^ { ( t ) } \| ^ { 2 } } { \sqrt { \sum _ { j = 0 } ^ { k } \| G _ { j } ^ { ( t ) } \| ^ { 2 } } } \leqslant \alpha \sum _ { t \in T } \sqrt { \sum _ { k = 1 } ^ { K } \| G _ { k } ^ { ( t ) } \| ^ { 2 } } .\tag{B.4}
$$

Plugging (B.2), (B.3) & (B.4) into (B.1) yields the required result.

## C Additional numerical results

![](images/830faa1868085eb5b550c32b250a06720dc6a051d0677cae3270776c025176a4.jpg)

![](images/9824d54968545118b97a607d7e70a4d7b0bfc6857056082d2c94e1608ca975e7.jpg)  
Figure 11: Comparison of optimizers iterating on $\tau$ (dashed) versus optimizers respecting the quotient $\tau / \mathcal { G }$ (solid), exemplified on the CIFAR10 dataset. Algorithms on $\tau$ perform slightly worse in terms of final loss and test accuracy (not displayed). RADAM and RMUON are slightly faster on $\tau / \mathcal { G }$ because $\mathrm { P r o j } ^ { H }$ is cheaper than Proj, DOG-like algorithm are faster on $\tau$ because the evaluation of $q _ { k }$ is cheaper than $d ^ { \mathrm { G r } }$

![](images/f96a74cc677b58a567fe5e116cf108e995d8333ea7809ecb56e5b0627bfff1f0.jpg)  
Figure 12: Development of the tensor norm $\| \tau ( \theta _ { k } ) \| _ { f }$ for ADAM (solid) and RADAM (dashed). Exploding norms hint at numerical problems for algorithms that act on τpθq, like compression. Riemannian optimizers (expemplified by RADAM) keep this norm stable.

## References

[1] Herbert Robbins and Sutton Monro. “A Stochastic Approximation Method”. In: The Annals of Mathematical Statistics 22.3 (1951), pp. 400–407. DOI: 10 . 1214 / aoms / 1177729586.

[2] Léon Bottou, Frank E. Curtis, and Jorge Nocedal. “Optimization Methods for Large-Scale Machine Learning”. In: SIAM Review 60.2 (2018), pp. 223–311. DOI: 10.1137/ 16M1080173.

[3] Nitish Shirish Keskar et al. “On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima”. In: International Conference on Learning Representations (ICLR). 2017. arXiv: 1609.04836.

[4] Chi Jin et al. “How to Escape Saddle Points Efficiently”. In: International Conference on Machine Learning (ICML). 2017. DOI: 10.5555/3305381.3305559.

[5] Diederik P. Kingma and Jimmy Ba. “Adam: A Method for Stochastic Optimization”. In: International Conference on Learning Representations (ICLR). 2015. DOI: 10.48550/ arXiv.1412.6980.

[6] Jordan Keller. Muon: An optimizer for hidden layers in neural networks. https : / / kellerjordan.github.io/posts/muon. 2024.

[7] Maor Ivgi, Oliver Hinder, and Yair Carmon. DoG is SGD’s Best Friend: A Parameter-Free Dynamic Step Size Schedule. 2023. arXiv: 2302.12022 [cs.LG].

[8] Saeed Ghadimi and Guanghui Lan. “Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming”. In: SIAM Journal on Optimization 23.4 (2013), pp. 2341–2368. DOI: 10.1137/120880811.

[9] Steven R. White. “Density Matrix Formulation for Quantum Renormalization Groups”. In: Physical Review Letters 69.19 (1992), pp. 2863–2866. DOI: 10.1103/PhysRevLett. 69.2863.

[10] Timo Felser and Simone Montangero. Introduction to Tensor Network Methods: From Many-Body Quantum Systems to Machine Learning. Cham: Springer, 2026. DOI: 10. 1007/978-3-032-17635-6.

[11] Timo Felser et al. “Two-Dimensional Quantum-Link Lattice Quantum Electrodynamics at Finite Density”. In: Physical Review X 10.4 (2020), p. 041040. DOI: 10.1103/ PhysRevX.10.041040.

[12] Timo Felser, Simone Notarnicola, and Simone Montangero. “Efficient Tensor Network Ansatz for High-Dimensional Quantum Many-Body Problems”. In: Physical Review Letters 126.17 (2021), p. 170603. DOI: 10.1103/PhysRevLett.126.170603.

[13] Giuseppe Magnifico et al. “Lattice quantum electrodynamics in (3+1)-dimensions at finite density with tensor networks”. In: Nature Communications 12 (2021), p. 3600. DOI: 10.1038/s41467-021-23646-3.

[14] E. M. Stoudenmire and David J. Schwab. “Supervised Learning with Tensor Networks”. In: Advances in Neural Information Processing Systems (NeurIPS). 2016, pp. 4799–4807. DOI: 10.5555/3157382.3157634.

[15] Zhao-Yu Han et al. “Unsupervised Generative Modeling Using Matrix Product States”. In: Physical Review X 8.3 (2018), p. 031012. DOI: 10.1103/PhysRevX.8.031012.

[16] Timo Felser et al. “Quantum-inspired machine learning on high-energy physics data”. In: npj Quantum Information 7.1 (2021), pp. 1–8. DOI: 10.1038/s41534-021-00443-w.

[17] Lorenzo Borella et al. “Ultra-low latency quantum-inspired machine learning predictors implemented on FPGA”. In: Machine Learning: Science and Technology 6.4 (Dec. 2025), p. 045062. DOI: 10.1088/2632-2153/ae25b5.

[18] Maximilian Scharf et al. Quantum-Inspired Robust and Scalable SAR Object Classification. 2026. arXiv: 2604.25755 [quant-ph].

[19] Stavros Efthymiou, Jack Hidary, and Stefan Leichenauer. TensorNetwork for Machine Learning. 2019. arXiv: 1906.06329.

[20] Hao Chen and Thomas Barthel. “Machine Learning With Tree Tensor Networks, CP Rank Constraints, and Tensor Dropout”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 46.12 (2024), pp. 7825–7832. DOI: 10.1109/TPAMI.2024.3396386.

[21] Alexander Novikov et al. “Tensorizing Neural Networks”. In: Advances in Neural Information Processing Systems (NeurIPS). 2015, pp. 442–450. arXiv: 1509.06569.

[22] Curt Da Silva and Felix J. Herrmann. “Optimization on the Hierarchical Tucker manifold – applications to tensor completion”. In: Linear Algebra and its Applications 481 (2015), pp. 131–173. DOI: 10.1016/j.laa.2015.04.015.

[23] Markus Hauru, Maarten Van Damme, and Jutho Haegeman. “Riemannian optimization of isometric tensor networks”. In: SciPost Physics 10 (2021), p. 040. DOI: 10.21468/ SciPostPhys.10.2.040.

[24] Marius Willner, Marco Trenti, and Dirk Lebiedz. Riemannian Optimization on Tree Tensor Networks with Application in Machine Learning. 2025. arXiv: 2507.21726 [math.OC].

[25] Simon Mataigne, P.-A. Absil, and Nina Miolane. “Bounds on the geodesic distances on the Stiefel manifold for a family of Riemannian metrics”. In: Linear Algebra and its Applications 730 (2026), pp. 1–34. DOI: 10.1016/j.laa.2025.10.003.

[26] Adrián Javaloy and Antonio Vergari. An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale. 2026. arXiv: 2602.14656 [cs.LG].

[27] Yibang Li et al. Intrinsic Muon: Spectral Optimization on Riemannian Matrix Manifolds. 2026. arXiv: 2605.09238 [cs.LG].

[28] André Uschmajew and Bart Vandereycken. “The geometry of algorithms using hierarchical tensors”. In: Linear Algebra and its Applications 439.1 (2013), pp. 133–166. DOI: 10.1016/j.laa.2013.03.016.

[29] E Miles Stoudenmire. “Learning relevant features of data with multi-scale tensor networks”. In: Quantum Science and Technology 3.3 (Apr. 2018), p. 034003. DOI: 10.1088/ 2058-9565/aaba1a.

[30] Gianluca Ceruti, Christian Lubich, and Dominik Sulz. “Rank-Adaptive Time Integration of Tree Tensor Networks”. In: SIAM Journal on NumericalAnalysis 61.1 (2023), pp. 194– 222. DOI: 10.1137/22M1473790.

[31] Alexander Novikov, Mikhail Trofimov, and Ivan Oseledets. “Exponential Machines”. In: Bulletin of the Polish Academy of Sciences Technical Sciences (2017), pp. 789–797. DOI: 10.24425/bpas.2018.125926.

[32] Gary Bécigneul and Octavian-Eugen Ganea. Riemannian Adaptive Optimization Methods. 2018. arXiv: 1810.00760.

[33] Daniel Dodd, Louis Sharrock, and Christopher Nemeth. “Learning-rate-free stochastic optimization over Riemannian manifolds”. In: International Conference on Machine Learning (ICML). Vienna, Austria, 2024. DOI: 10.5555/3692070.3692513.

[34] Ralf Zimmermann and Jakob Stoye. “High Curvature Means Low Rank: On the Sectional Curvature of Grassmann and Stiefel Manifolds and the Underlying Matrix Trace Inequalities”. In: SIAM Journal on Matrix Analysis and Applications 46.1 (2025), pp. 748–779. DOI: 10.1137/24M1655755.

[35] Thomas Bendokat, Ralf Zimmermann, and P.-A. Absil. “A Grassmann manifold handbook: basic geometry and computational aspects”. In: Advances in Computational Mathematics 50.1 (Jan. 2024). DOI: 10.1007/s10444-023-10090-8.

[36] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a Novel Image Datasetfor Benchmarking Machine Learning Algorithms. 2017. arXiv: 1708.07747.

[37] Nikolas Klug et al. Natural Riemannian gradient for learning functional tensor networks. 2026. arXiv: 2604.09263 [math.OC].

[38] Alex Krizhevsky. “Learning Multiple Layers of Features from Tiny Images”. In: University of Toronto (May 2012). DOI: 10.24432/C5889J.

[39] Jeremy Howard. Imagenette. https://github.com/fastai/imagenette. 2019.

[40] G. Catarina and Bruno Murta. “Density-matrix renormalization group: a pedagogical introduction”. In: The European Physical Journal B 96.8 (Aug. 2023). DOI: 10.1140/ epjb/s10051-023-00575-2.

[41] Samuele Cavinato et al. “Optimizing radiotherapy plans for cancer treatment with Tensor Networks”. In: Physics in Medicine & Biology 66.12 (2021), p. 125015. DOI: 10. 1088/1361-6560/ac01f2.

[42] Shoshichi Kobayashi and Katsumi Nomizu. Foundations of Differential Geometry. Vol. I. New York-London: Interscience Publishers, a division of John Wiley and Sons, 1963, p. 329. DOI: 10.1126/science.143.3603.235.b.

[43] Peter Michor. Topics in Differential Geometry. American Mathematical Society, July 2008. DOI: 10.1090/gsm/093.

[44] Nicolas Boumal. An introduction to optimization on smooth manifolds. Cambridge University Press, 2023. DOI: 10.1017/9781009166164.

[45] Thorsten Rohwedder and André Uschmajew. “On Local Convergence of Alternating Schemes for Optimization of Convex Problems in the Tensor Train Format”. In: SIAM Journal on NumericalAnalysis 51.2 (2013), pp. 1134–1162. DOI: 10.1137/110857520.

[46] Peter Auer, Nicolò Cesa-Bianchi, and Claudio Gentile. “Adaptive and Self-Confident On-Line Learning Algorithms”. In: Journal of Computer and System Sciences 64.1 (2002), pp. 48–75. DOI: 10.1006/jcss.2001.1795.