# Playing with Kruskal: a state-of-the-art report on watershed cuts

Jean Cousty<sup>1</sup>, Laurent Najman<sup>3,1\*</sup>, Benjamin Perret<sup>1</sup> and Deise Santana Maia<sup>2</sup>

<sup>1</sup>Univ Gustave Eifel, CNRS, LIGM, F-77454 Marne-la-Vallée, France. <sup>2</sup>Univ Lille, CNRS, UMR 9189 CRIStAL, F-59000 Lille, France. <sup>3</sup>Math. Department, Khalifa University, Abu Dhabi, UAE.

\*Corresponding author(s). E-mail(s): laurent.najman@esiee.fr; Contributing authors: jean.cousty@esiee.fr; benjamin.perret@esiee.fr; deise.santanamaia@univ-lille.fr;

## Abstract

In the framework of edge-weighted graphs, watersheds have proven to be linked to well-known optimization problems, as Minimum Spanning Tree, which allowed the design of eficient algorithms for computing (hierarchical) watershed segmentations. In the present article, after reviewing the literature related to watershed segmentation, we present a detailed end-to-end pipeline of algorithms to compute (hierarchical) watershed segmentations, starting from the computation of graph-based image representations, up to the computation of connected components of the final (hierarchical) segmentation. We consider the several variations of watersheds, including their supervised and unsupervised versions, and the various ways of computing seeds, to name a few. For the first time, we bring together all these watershed notions and algorithms in a compact and understandable way. We aim at providing a reference for those interested in employing and reimplementing the watershed segmentation framework for their task at hand.

Keywords: Watershed segmentation, graph, hierarchy, binary partition tree

## 1 Introduction

The watershed transform was first proposed as a powerful tool for the segmentation of grayscale digital images [1, 2]. Since then, numerous definitions and algorithms have been developed to implement the watershed transform in both 2D and 3D image domains, as well as within the framework of graphs [3–9]. In all cases, the underlying idea of the watershed transform is to interpret an image (or a weighted graph) as a topographic surface, where a set of connected pixels with equal gray values surrounded by pixels of strictly higher gray values (or a set of adjacent vertices/edges surrounded by vertices/edges of strictly greater weights) forms a regional minimum of the surface. Each regional minimum corresponds to a zone of influence, known as a catchment basin. From any point (pixel or vertex) within the catchment basin of a regional minimum, there exists a descending path to that minimum. A descending path is defined as either a sequence of connected pixels with non-increasing gray levels or a sequence of vertices connected by edges with non-increasing weights.

A watershed segmentation is, therefore, a partition of an image into its catchment basins. Its hierarchical version consists of a sequence of nested partitions obtained by progressively merging the regions of an initial watershed segmentation [10–12]. When formalized within the framework of edge-weighted graphs, (hierarchical) watersheds are closely related to the optimization problem of finding minimum spanning trees (MST), as proved in [9, 13, 14].

In this article, we present an overview of stateof-the-art watershed definitions for edge-weighted graphs, covering flat, hierarchical, supervised, and unsupervised approaches. Then, following the trend of Najman et al. [15], we revisit the watershed paradigm through the lens of Kruskal’s algorithm for computing MSTs [16]. By doing so, we reveal a simple, modular, and powerful framework for computing watershed segmentations—flat or hierarchical, supervised or unsupervised—using a single underlying data structure: the Binary Partition Tree by Altitude Ordering (BPTAO) [14]. This tree encodes the structure of the MST, supports eficient seed propagation, and enables the direct extraction of segmentation boundaries or hierarchical saliency maps [17]. By systematically organizing and extending previous results in the literature, we describe an end-to-end pipeline that goes from an input image to the final watershed segmentation, passing through graph construction, tree computation, seed selection, and cut extraction.

Our goal is to provide a comprehensive and versatile algorithmic state-of-the-art reference for researchers and practitioners working with watershed-based segmentation. We emphasize the interplay between combinatorial optimality, morphological filtering, and hierarchy construction, and demonstrate that all major variants of watershed cuts—basic, seeded, and hierarchical—can be derived and computed within our Kruskal-inspired framework. In doing so, we hope to foster better understanding, facilitate implementation, and open new avenues for integrating watershed cuts with modern learning-based methods.

Section 2 briefly reviews watershed approaches for image segmentation. Rather than aiming at an exhaustive survey, we position watershed cuts within a broader context, emphasizing their connections to combinatorial optimization and computational geometry. Section 3 revisits the theoretical foundations of the watershed-cut framework. Building upon these principles, Section 4 introduces the generic Playing-with-Kruskal algorithmic scheme, designed to unify the computation of the various instances of watershed cuts. The construction of the central data structure of this framework, namely the Binary Partition Tree, is detailed in Section 5. Section 6 addresses seed computation, while Sections 7 and 8 explain how watershed-cut segmentations are obtained from the binary partition tree and the selected seeds. Finally, Section 9 presents illustrative use cases implemented using the Higra library [18], demonstrating the practical relevance and flexibility of the proposed framework.

## 2 Related work

Since the first definition of watershed segmentation in the late 1970s [1], several algorithms and paradigms have been proposed to compute watershed segmentations, notably in the work of Vincent and Soille [3]. Oversegmentation being a common aspect of the original watershed definitions, where markers are defined as the set of regional minima, various approaches have been proposed to alleviate this issue, either by filtering the initial regions, computing a hierarchical segmentation, improving the selection of markers, or modifying the computation of the image gradient, such as the techniques proposed in [10, 11, 19–21]. Based on constant developments over the years, watershed segmentation attracted a lot of attention in many applicative fields such as remote sensing [22], medical imaging [23, 24], biological imaging [25], and material science [26, 27]. In this section, we review the main steps of watershed developments for image analysis and discuss its possible or established connections to other important research fields, mainly combinatorial optimization and geometry.

## 2.1 Early flooding-based methods on vertex-weighted graphs

In the early work of Vincent and Soille [3] and of Meyer [4], the proposed flooding algorithm relies on vertex-weighted graphs whose vertices correspond to image pixels and whose edges are given by an adjacency relation on the pixels. It starts with a set of marked seed pixels. Each seed pixel is given a unique label. Then, the marked pixels are iteratively augmented by incorporating neighboring pixels that are not yet marked, following an increasing order of weights. The final segmentation is a pixel labeling that indicates the catchment basins of the pixels and possibly the label “watershed divide" for some pixels. Other flooding-based approaches are also described in [28–31].

## 2.2 Hierarchical and stochastic watersheds

Hierarchical watershed approaches [10–14, 32, 33] were introduced to extend the classical watershed to multiple scales. By organizing the image segmentation into nested regions forming a hierarchy [17, 34–36], it becomes possible to analyze structures at diferent levels of detail. One widely used approach relies on geodesic saliency (also known as ultrametric contour map [37, 38]), which ranks region contours based on their persistence across scales. This allows the extraction of a multi-scale hierarchy of regions that reduces spurious basins while preserving relevant structures.

Angulo and Jeulin [20] proposed the stochastic watershed segmentation, in which the final segmentation is preceded by the computation of a probability density function (PDF) of contours. These contours are obtained by combining watershed segmentations computed from diferent random sets of markers. The idea is that important contours persist across multiple random marker sets, increasing their value in the probabil ity density function, while small and low-contrast contours fade, leading to reduced oversegmentation. An exact calculation of the PDF is proposed by Stawiaski and Meyer [39], while an eficient algorithm for its computation is proposed by Malmberg and Hendriks [40]. In [41], an improvement to the original method introduces noise at every iteration and uses a randomly placed grid to distribute the markers.

## 2.3 Edge-weighted graphs and combinatorial optimization

Since the early 2000s, edge-weighted graphs have established themselves as a powerful framework for image segmentation. As presented in the following subsection, it became the reference framework for watershed segmentation allowing establishing connections between watersheds and important combinatorial optimization problems. The most notable seminal work of Felzenszwalb and Huttenlocher [42] proposes a nearly lineartime segmentation algorithm in which an input image is represented as a graph (pixels as nodes and edge weights encoding dissimilarity between pixels) whose clustering strategy is closely related to the Minimum Spanning Tree (MST) optimization problem. Since then, this simple and eficient algorithm has been extended and adapted to a wide range of segmentation frameworks, such as [43]. In the framework of edge-weighted graphs, Boykov et al. [44] consider the min-cut/max-flow optimization problem to solve various image analysis problems including segmentation. In [45], the authors propose the Image Foresting Transform (IFT), which implements an extended version of Dijkstra’s shortest paths forests algorithm on edge-weighted graphs, where each vertex is clustered with the initial seed (or subgraph) to which its path is shortest among all initial seeds.

## 2.4 Watershed cuts and unifying frameworks

Following this trend, watershed cuts were introduced in the framework of edge-weighted graphs in the work of [9, 46, 47]. They are defined following the intuitive idea of drops of water flowing on a topographic surface. In contrast to watersheds in previously considered discrete settings, the consistency of watershed cuts has been established: they can be defined equivalently by their “catchment basins" through the steepest descent property or by the “dividing lines" that separate these basins through the drop of water principle. Watershed cuts are also closely related to MSTs through an equivalence theorem, and hence to Kruskal’s MST algorithm, which, as detailed in the later sections, provides a key ingredient for eficient and versatile computations for the regular flat case, the marker-based versions, and the hierarchical extensions.

Additionally, this strong connection to MSTs opened the doors to establishing links between watershed cuts and other optimization approaches on graphs [48]. Especially, it is shown that mincuts coincide with watersheds for some particular weight functions. These links were further studied in [49, 50] that introduce a new generic combinatorial optimization problem on graphs parameterized by two exponent values $p$ and q and allows several known methods to be unified. Depending on the values of $p$ and q, the solution is a minimal cut, a random walker [51], a shortest paths forest, or finally, when $p$ tends to infinity, a new combinatorial object called a power watershed. The power watershed is a watershed cut for which ties are discriminated by a secondary optimality criterion. This result links several methods that are useful in image analysis and combines their advantages.

## 2.5 Extension, learning and large-scale processing

More recently, Mutex Watersheds [21], which can be seen as an application of the power-watershed framework, extended the multicut formulation to a “seed-free” watershed approach with time complexity comparable to MSTs. The method incorporates both attractive interactions, which encourage merging of similar pixels, and repulsive longrange edges, allowing the graph to account for both short and long-range relationships between pixels. Its main advantages are that it does not require seeds or a predefined number of regions.

In the current deep learning era, watershed segmentation remains an important component of many image analysis pipelines and is still widely used in conjunction with modern machine learning methods, often as a pre- or post-processing step (see, e.g., [38, 52–56]).

In [57], the authors proposed, in the spirit of random forest [58], to ensemble many “weak" watershed cuts, each one using only a random subset of the features for computing the edge-weights. In combination with a triplet loss, this has been proved particularly eficient for segmenting hyperspectral images [59].

Other approaches include interactive, outof-core, and deep learning based segmentation approaches [60–62]. In [60], the authors introduce an eficient interactive implementation where seeds can be added or removed without recomputing the entire segmentation, achieving a speed-up of up to 95 times, with some multi-core parallelizations, when compared to the previous stateof-the-art high-performance library. In [61], an out-of-core watersheds algorithm is proposed for data which cannot fit the main memory, enabling parallelized computation without compromising the correctness of the watershed-cut framework. Da Fonseca et al. [62] leverages a deep network to provide user-based markers optimized for connected, smooth regions, while Lapertot et al. [63] uses deep learning to generate hierarchical contours, producing semantically coherent segmentation levels.

## 2.6 Beyond images: surfaces, geometry, and terrains

The notion of a watershed has also been applied to the partitioning of 3D surface meshes, notably following the work of Mangan and Whitaker [64]. The watershed cut approach is, for instance, applied in [65] to tasks such as artwork 3D model segmentation prior to classification and indexation.

To partition surfaces, notions closely related to those involved in watershed-based image segmentation have been developed in the computational geometry community, in particular through ascending, descending, and Morse-Smale complexes [66–68], defined in the setting of simplicial complexes and often relying on discrete Morse functions [69]. Each ascending complex is associated with one minimum and comprises the locations which can be reached from this minimum by an ascending line. From this perspective, ascending complexes can be thought of as analogous to the watershed regions (the catchment basins). The descending complex is a dual notion. It can be interpreted as the regions delineated by the thalwegs, a thalweg (the dual of a watershed) being made of “closed contours" that pass through the bottoms of the valleys and separating the regional maxima. The Morse-Smale complex is obtained by intersecting the ascending and descending complexes. It is tightly linked to the critical points of the altitude function, namely minima, maxima and saddles.

For Morse-Smale functions on 2D manifold surfaces, the crossing of ascending and descending complexes is necessarily at a saddle. The strong relation between watersheds and saddles (also referred to as pass points) is well established in mathematical morphology notably within the topological watershed framework [70]. The notion of watershed from image analysis has further been studied in [71] on simplicial complexes pseudomanifolds and, later, for discrete Morse functions defined on such spaces [72]. Conversely, the watershed has inspired approaches based on discrete Morse theory [69].

Finally, it is worth noting that approaches from both image analysis and computational geometry to partition surfaces are widely applied to digital elevation models in hydrography and geoscience [73–79]. In this context, the term ridge is also frequently used to denote watershed-like structures.

## 2.7 Implementations

The literature review presented in this section is not intended to be exhaustive, but rather to provide an overview of a selection of methods derived from the notion of watersheds and related concepts. Watershed algorithms for 2D and sometimes 3D image processing are widely implemented in both open-source libraries, such as scikit-image, ITK, and OpenCV, and commercial or domain-specific platforms including MATLAB, ImageJ, and Amira/Avizo. For a broad review on open-source watershed algorithm implementations, we refer the reader to [80, 81] and to [18], the latter being the hierarchical graph analysis library used for the illustrations in this article.

## 3 Watershed cuts and hierarchies

In this section, we provide an overview of the watershed cuts family. More precisely, we present three variations of watershed cuts:

• basic watershed cuts: the method is fully unsupervised, and the input function is automatically partitioned into regions associated to the minima of the input function;

• seeded watershed cuts: the method is semisupervised and the user, or an automated procedure, must additionally provide markers located inside each region which is to be segmented. As a result, the algorithm partitions the images into regions associated to the markers.

• hierarchical watershed cuts: the method is either fully unsupervised or semi-supervised and considers a hierarchical image representation where the watershed regions of the finest scale (associated to all minima of the input function) are progressively merged to form watershed regions of higher scales representing object of higher semantic level.

## 3.1 Watershed cuts

The watershed transform introduced by Digabel and Lantuejoul [1] for morphological segmentation and later popularized by Vincent and Soille [3] is used as a fundamental step in many image segmentation procedures. A grayscale image, or more generally a function, is seen as a topographic surface: the gray values become the elevations, the basins, and valleys correspond to dark areas whereas the mountains and crest lines correspond to light areas. Intuitively, the watershed is a subset of the domain, located on the ridges of the topographic surface, that delineates its catchment basins. It may be thought of as a separating lineset from which a drop of water can flow down towards several minima. For applications to image segmentation, the watershed is often computed from the gradient magnitude of an image. Therefore, the resulting contours are located on high gradient contours of the image, which often correspond to the borders of the objects of interest.

Following the intuitive drop of water principle presented in the previous paragraph, the watershed cuts, a notion of a watershed in edgeweighted graphs, were introduced in [47]. A watershed cut is indeed a graph cut: it is only made of edges, and it partitions the vertex set of the underlying graph. The chosen graph cut satisfies two principles: (i) the induced partition associates exactly one region to each regional minimum of the graph; and (ii) each edge of the graph cut satisfies the intuitive drop of water principle presented in the previous paragraph (i.e., a “drop of water" falling down on this location can flow down with a descending path towards two distinct regional minima of the graph). In application to image analysis, the considered edge-weighted graph $( V , E , w )$ is often chosen such that:

• each vertex in V is a pixel or voxel of the image which is to be segmented

• each edge in E is a pair of pixels which are neighbor for a usual adjacency relation such as the 4- or 8- adjacency relation or the 6-, 18-, or 26- adjacency in 3D

• the weight $w ( \{ x , y \} )$ of an edge linking two vertices x and y measures the dissimilarity between the two pixels linked by the edge. A simple and often used case consists of taking any distance between the brightness, color, or features associated to the pixels x and y. Recently, it has become common to use contour maps or edge detection produced by deep learning models to weight the edges, as these models are able to account for higherorder semantic information (see [82–84]). A good choice is often a (linear) combination of learned gradients and local dissimilarities in order to capture multiscale information.

The notions introduced in this section are illustrated in Figure 1.

![](images/b0bfb44d36dbc70ac3fd8e083e084fd6d5a44aa46b82b0d4b81cbfcc207d2421.jpg)  
Fig. 1 Left: A digital image (bottom) and an edgeweighted graph representing this image (top): vertices correspond to pixels and edges represent the adjacency between pixels. The edge weights measure the dissimilarity between the values of adjacent pixels. The three minima of the edge-weighted graph are represented in bold red. Right: a watershed cut of the edge-weighted graph, which is represented by dashed edges in the graph (top) and by a color label in the image domain (bottom). The minimum spanning forest rooted in the minima of the graph associated to the watershed is depicted in bold red.

Having such a weighted graph $G = ( V , E , w )$ ， let us now formalize the three main notions necessary to define watershed cuts.

• Regional minimum. Intuitively, a regional minimum of G is a subgraph X of constant weight from which it is not possible to escape without climbing. Formally, the subgraph X is a minimum of G if every two edges of X have the same weight and every edge adjacent to an edge of X has a weight strictly higher than the weight of an edge of X.

• Graph cut. Any partition of the vertex set of a graph which is made of connected regions is called a graph cut and any graph cut C induces a unique cut set composed of every edge of G linking two end-vertices in two distinct regions of the partition C.

• Drop of water principle. Intuitively, a graph cut satisfies the drop of water principle if from every edge in the cut set it is possible to reach two distinct minima of G by a path of descending weights, more precisely, if for any edge x, y in the cut set, there exists two paths $\pi _ { x } = ( x _ { 0 } = x , \ldots , x _ { k } )$ and $\pi _ { y } = ( y _ { 0 } = y , \ldots , y _ { \ell } )$ such that:

$\pi _ { x }$ and $\pi _ { y }$ are paths of descending weights that do not cross the cut set;

– the weight of the first edge of $\pi _ { x }$ (resp. $\pi _ { y } )$ is not greater than the weight of x, y ;

$x _ { k }$ and y belong to two distinct minima of G.

Definition 1 (watershed cut) A watershed cut of G is a graph cut C which is such that each region of C contains exactly (the vertex set of) one regional minimum of G, and which satisfies the drop of water principle.

Watershed cuts have been deeply studied in the literature and are known to satisfy several properties establishing their consistency or combinatorial optimality [9, 49, 85] and drawing links with topology-preserving transforms [47, 71], shortest paths or fuzzy connectedness [47]. In our context, it is important to recall a characterization of watershed cuts based on the minimum spanning forest/tree problems, as the algorithms that we are going to propose in further sections are based on the famous Kruskal’s minimum spanning tree algorithm.

Theorem 2 (optimality [9]) A cut C is a watershed cut ofG ifand only ifthere exists a minimum spanning forest F of G which is rooted in the regional minima of G and whose connected component partition is equal to C, more precisely, if and only if there exists a subgraph F of G such that:

1. each connected component of F includes exactly (the vertex set of) one regional minimum of $G ;$

2. the connected component partition of $F$ is $C ;$ and

3. the sum of the weights of the edges in F is minimum among the sums of the weights of the subgraphs of G that satisfy the above two conditions.

## 3.2 Seeded watershed cuts

In practice, it is often desirable to supervise the segmentation process by providing a set of markers or seeds such that the resulting partition contains exactly one region associated with each marker that is provided. In this case, each marker is considered a clue to the location of the object of interest to be segmented.

Intuitively, as early described by Beucher and Meyer in [86], such seeded segmentation can be obtained by considering the watershed cut of a modified version of the graph G. The idea is then to consider a modified weight map $w ^ { \prime }$ such that the minima of $w ^ { \prime }$ correspond exactly to the provided markers and such that $w ^ { \prime }$ is “as close as possible" to w. This filtering $w ^ { \prime }$ can be obtained in two steps. First, the process begins by “digging" (i.e., setting to a minimal value) the weights of the edges that belong to one of the markers. Then, a closing by reconstruction is performed. The result of this closing by reconstruction, a filtering process studied in mathematical morphology, can be obtained using a data structure called a component tree. It consists of raising the weights of the undesired minima until they vanish. The watershed cut of the modified version $w ^ { \prime }$ of w is called a watershed cut of w seeded in the provided marker set . It indeed contains one and only one region for each of the markers. The notion of a seeded watershed cut (and also Theorem 3) is illustrated in Figure 2.

![](images/07b76684151ce39576b4b48fb1a1b62562a23aa3906da607ca8e31564e6e6f29.jpg)  
Fig. 2 Left: an initial edge-weighted graph with three minima in bold red and two seeds overlaid in blue. Middle: a closing by reconstruction from the initial graph and the seeds (changed weights in red). Edge weights within seeds are set to -1, forcing them to become minima. The closing by reconstruction then fills the minima of the original graph. Finally, the associated watershed cut is depicted by dashed edges. Right: a minimum spanning forest (bold red) of the initial graph rooted in the seeds, together with its induced graph cut (dashed edges), which coincides with the watershed of the closing by reconstruction in the middle figure, as stated in Theorem 3.

A formal definition of seeded watershed cut can be found in [85]. It is possible to obtain the result of a seeded watershed cut without actually computing the modified version $w ^ { \prime }$ of w. Instead of composing filtering followed by watershed cuts, it is suficient to consider the cut induced by a minimum spanning forest rooted in the provided markers. More precisely, the following property has been proved in [85].

Theorem 3 (seeded MSF cuts, [85]) Let M be a marker set. If C is the connected component partition of a minimum spanning forest of G rooted in the marker set M, then C is a watershed cut of G for the marker set M.

For the sake of completeness, we recall that a minimum spanning forest of G rooted in is a subgraph F of G such that:

1. each connected component of F includes exactly one of the elements of ;

2. $F$ spans $G , i . e .$ , every vertex of G is a vertex of $F ;$ and

3. the sum of the weights of the edges in F is minimal, that is to say, the sum of the edge weights in any subgraph of G for which the two above conditions hold true is greater than or equal to the sum of the weights of the edges in $F .$

## 3.3 Hierarchical watershed cuts

In some cases, the objects of interest do not all appear at the same scale, and some of them can be nested into each other. This is, for instance, the case of the nose, the face, and the silhouette of a person in a picture. In such cases, a multiscale or hierarchical image segmentation is desirable. Early studies of this approach date back to [10] and [11] in mathematical morphology and to [36], under the name of scale-set theory, from an optimization perspective. Such a hierarchical representation of an image can be envisioned thanks to hierarchical watershed cuts [12–14, 87] as presented in this section.

From a high-level topographical point of view, a hierarchical watershed can be obtained by:

1. considering a series $\begin{array} { r l r } { \Phi } & { { } = } & { ( \Phi _ { 0 } ( G ) \quad = } \end{array}$ $G , \ldots , \Phi _ { N } ( G ) )$ of progressive filtrations (described below) of the input weighted graph $G$ seen as a relief, for any $N \geq 1$ ordered from the lightest to the most aggressive; and

2. considering an associated series $\left( C _ { 0 } , \dots , C _ { N } \right)$ of watershed cuts of these filtrations where $C _ { \lambda }$ is a watershed of $\Phi _ { \lambda } ( G )$

The obtained series $\left( C _ { 0 } , \dots , C _ { N } \right)$ of partitions is called a hierarchical watershed if the following conditions are met:

• Hierarchy principle ([34–36, 88]). At every scale, every region can be obtained as the merging of regions at the previous scale. More formally, for every λ in $\{ 1 , \ldots , N \}$ , the cuts $C _ { \lambda - 1 }$ and $C _ { \lambda }$ are nested, meaning that, each region $R$ of $C _ { \lambda - 1 }$ is included in a region $R ^ { \prime }$ of $C _ { \lambda }$ , hence, conversely, every region $R ^ { \prime }$ in $C _ { \lambda }$ is the union of the regions of $C _ { \lambda - 1 }$ which are included in $R ^ { \prime } ~ ( R ^ { \prime } = \cup \{ R ~ | ~ R ~ \in ~$ $C _ { \lambda - 1 } , R \subseteq R ^ { \prime } \} )$ ).

• Extensive connected operators. Intuitively, the filtration process involved in hierarchical watershed acts by progressively removing (or “filling in") the minima while leaving unchanged the other weights of the graphs corresponding to the highest mountains and crest lines. The preservation/removal of a minimum in a filtration is in general driven by a measure of importance of that minimum like its area, depth or more frequently the area, depth or volume of its drainage region. Such filters are known as connected extensive operators in mathematical morphology [89–92]. More formally, given an increasing measure $\mu \ ( e . g .$ , the size) mapping a positive real value to any subgraph and a scale parameter $\lambda ,$ the filtering result is obtained by:

1. considering the set of connected components $C ( G , w , k )$ of the lower-threshold of the weights w at every possible value k $( i . e . ,$ the graph obtained by removing from G the edges of weight greater than k);

2. removing from the set $C ( G , w , k )$ every connected component for which the measure $\mu$ is less than the scale parameter $\lambda ,$ yielding the set $C ^ { \prime } ( G , w , k )$ ;

3. building the filtered map $w ^ { \prime }$ by stacking the connected components remaining in the sets $C ^ { \prime } ( G , w , k ) , i . e .$ , setting, for every edge u of G, $w ^ { \prime } ( u ) \ =$ min k u is an edge of a connected component in $C ^ { \prime } ( G , w , k ) \}$ . In the following, any considered filtration series results from an extensive connected operator.

The notion of a hierarchical watershed is illustrated in Figure 3.

As in the single-scale case, the hierarchical watershed segmentation problem is deeply related to the minimum spanning tree combinatorial optimization problem. This link relates to the extinction value of the minima. More precisely, for each minimum of the graph, we can associate an extinction value $\epsilon ( M )$ which is the highest scale parameter k such that the minimum $M$ “remains" $( i . e . , M$ is included in a minimum) in the filtration $\Phi _ { k } ( G )$ . Let us state the main result of the hierarchical watershed theory which establishes the multiscale optimality of the approach and suggests the algorithmic framework developed in this article.

![](images/0ee3585e28cef168ad79e27fbdfe065c70a176903ff6c2d21f09d8f7ad61d7f3.jpg)  
(e) Hierarchical watersheds  
Fig. 3 (a-d): Four filtrations of a graph obtained with a connected closing by area removing successively the components with less than 0, 3, 6, and 8 vertices. The edge weights that changed between two levels of filtrations are highlighted in red. The minima are in bold red. The watershed cuts of each filtration are represented by dashed edges. Observe that the watershed cut obtained at one filtration level is included in the watershed cut obtained at a lower level. (e): The watershed cuts of the series of filtrations form a hierarchical watershed. It can be represented as a saliency map (bottom), i.e., an edge-weighted graph, where the weight of an edge indicates at which filtration level it ceases to be in the cut or 0 if it does not belong to any watershed cut. Note that every watershed cut from the series of filtration can be recovered by thresholding the saliency map. The four regional minima of the original graph are the four minima, all at level 0, of the saliency map (each depicted in a diferent color). The hierarchical watershed can be equivalently represented as a tree (top) representing the fusion of regions along the filtration process.

Theorem 4 (multi-scale optimality) Let $\Phi \quad = \quad$ $( \Phi _ { 0 } ( G ) = G , \dots , \Phi _ { N } ( G ) )$ be a filtration series of $G$ and let $\mathcal { H } = ( C _ { 0 } , \ldots , C _ { N } )$ be a series of partitions which satisfies the hierarchy principle. If,

(i) for any value $\lambda \in \{ 0 , \ldots , N \}$ , there exists a minimum spanning forest $F _ { \lambda }$ of G which is rooted in the minima of G of extinction value (with respect to the filtration series Φ) greater than or equal to λ and whose connected component partition is $C _ { \lambda }$

then

(ii) H is a hierarchical watershed with respect to the filtration Φ.

Furthermore, if Φ is a one-by-one minima filtration series (i.e., the number of minima decreases by one between any two successive filtrations), the converse is also true.

The proof of Th. 4 is provided in Appendix A. Besides establishing the optimality of hierarchical watershed cuts, the previous theorem provides us with the following important observations:

• it is not necessary to actually compute the successive filtrations of G and to extract the watershed of these filtrations, wishing them to satisfy the hierarchy principle,

• instead, one only needs to know the extinction values of every minimum and to compute nested minimum spanning forests of the original graph G.

In the following, we will use the latter approach, showing how to

• compute extinction values of minima (or more generally seeds); and

• compute successive minimum spanning forests based on Kruskal’s minimum spanning tree algorithm and on the associated BPTAO.

## 3.4 Watershed cuts from a hierarchy

In the previous subsection, we have seen that the watershed cuts may be used to define multiscale image representation. On the other hand, in Section 3.2, we have shown how watershed can be used to extract a segmentation directly from an image based on user (or machine learning) supervision given in the form of seed vertices indicating clues on objects of interest location. In this subsection, we explain how watershed cuts can be used to extract a segmentation from a hierarchy and a seed-based supervision. The core idea is to reduce the latter extraction to the former one. This can be done with saliency maps ([11, 17], a notion also known as ultrametric contour map [37]), which is an equivalent representation of a hierarchy in the form of an edge-weighted graph that can be seen as the characteristic function of the hierarchy.

Intuitively, the saliency map of a hierarchical watershed cut is a new weight function for the edges of the graph G such that the weight of each edge indicates the highest scale for which the two extremities of that edge belong to distinct regions. In other words, at any scale below the saliency weight of an edge, its two extremities lie in distinct regions of the hierarchy, whereas at larger scales they belong to the same region. In Figure 3(e), one can see the saliency map of the presented hierarchical watershed. It has to be noted that the saliency map is an equivalent representation of a hierarchical watershed. Using notions from the classification community, the saliency weight of an edge is given by the distance between its extremities, where the distance is the ultrametric [35] associated to the hierarchical watershed.

In the remaining part of this article, seeded hierarchical watershed cuts refer to the results of the following supervised segmentation scheme:

1. compute a hierarchical watershed cut $\left( C _ { 0 } , \dots , C _ { N } \right)$ of G;

2. compute the saliency map $w _ { s }$ of $\left( C _ { 0 } , \dots , C _ { N } \right)$ ;

3. compute the watershed cut of the saliencyweighed graph $( V , E , w _ { s } )$ for the given marker set.

## 4 Algorithmic scheme overview

In this section, we present an overview of the proposed framework to produce watershed-cut segmentations from a given image, which comprises three main steps: (i) construction of the Binary Partition Tree by Altitude Ordering (BPTAO), that allows one to eficiently browse the minimum spanning tree edges of the graph; (ii) detection of the edges of the MST where to cut in order to obtain the segmentation, with the possible scale at which the corresponding cut occurs; these edges are detected by searching particular nodes of the BPTAO; (iii) computation of the segmentation regions given by the cuts.

Our framework is based on the link between BPTAO and MSTs, which have already been studied in [14], with eficient algorithms to compute BPTAO proposed in [15] and implemented in [18]. In simple terms, the BPTAO is a tree-based hierarchical data structure used to represent an MST and can be seen as a by-product of the eficient Kruskal’s MST algorithm. Figure 4 shows the BPTAO of an edge-weighted graph. In particular, the leaves of the BPTAO are associated to the vertices of the MST and each non-leaf node is associated to an edge of the MST. Intuitively, we can say that every non-leaf node of the BPTAO represents the addition of an edge in a non-decreasing order of their weights (indicated by the red boxes in Figure 4) to the MST during Kruskal’s algorithm, where the added edge is used to merge the two connected regions that contain the edge extremities.

The overall steps of our algorithm are illustrated in Figure 5 and described below.

1. Computation of an edge-weighted graph $G = ( V , E , w )$ for an input image I. As shown in Section 5, BPTAOs are computed from (connected edge-)weighted graphs, commonly obtained by setting V as the set of pixels of I and E as edges between the 4- or 8-connected pixels.

2. Construction of a BPTAO from $G \ =$ $( V , E , w )$ (Section 5). In simple terms, the nodes of the BPTAO of a given weighted graph G correspond to the connected components produced during Kruskal’s minimumspanning-tree algorithm, the leaf nodes being the vertices of G. Given a set $V$ of vertices (leaf nodes) and a total ordering on E, we compute a sequence of increasingly coarser partitions of V by incrementally merging the regions linked by the edges $e _ { 1 } , e _ { 2 } , . . . , e _ { \parallel E \parallel }$ (ordered according to ) until a single region has been obtained.

![](images/6b7bb404b74c0409898480e8b2c33a84f762fd5a008c9c27632d9abb51d44594.jpg)  
Fig. 4 A weighted graph $G = ( V , E , w )$ , a MST of G with a non-decreasing ordering on its edge weights indicated in red, and the BPTAO of this MST for this given ordering.

3. Selection of seeds/markers (Section 6). The construction of (hierarchical) watershed segmentations is guided by seeds indicating where and how many regions (or hierarchical levels) should be present in the final segmentation. Those seeds can be obtained through unsupervised or supervised methods, with the set of (ordered) regional minima of G and seeds manually input by the user (drawn from the original image) being among the common choices. Namely, in the case of hierarchical watersheds, the set of regional minima is often ordered according to their “importance level” for a given regional attribute, such as the area (proportional to the number of pixels in a region) or volume (proportional to the number and intensity of pixels in a region), which is commonly measured by extinction values [90]. In turn, extinction values can also be obtained in linear time from the BPTAO, as described later in Section 6.2.

## 4. Computation of watershed nodes of BPTAO or saliency values of the MST edges (Section 7)

• Non-hierarchical case: once the BPTAO has been constructed, the watershed-cut edges of a MST of G can be obtained in a single pass from the leaf nodes to the root of BPTAO. Those watershed-cut edges establish a partition on the vertices of G.

• Hierarchical case: in the hierarchical version of watersheds, where weighted or ordered markers are given as input, we also need to obtain the saliency values of watershed-cut edges, from which a nested sequence of partitions of V can be derived. In this case, the saliency values play a role in specifying at which level of the hierarchical watershed the nodes connected by those edges should be merged.

5. Computation of connected components corresponding to a (hierarchical) watershed of G for the given seeds S. (Section 8) Once the watershed nodes and, possibly, their saliency values, have been computed, the following step computes the (hierarchical) watershed segmentation of S, which consists of a (hierarchical set of nested) partition(s) of V. In the non-hierarchical case, if we are rather interested in the contours separating those connected components, a cut of G can be later computed in linear time in a single pass on the set E. In the hierarchical case, the saliency map of the MST can be obtained in linear time with respect to E using Algorithm 7 introduced later in Section 7.2.

In the remaining part of this article, we assume that seeds are composed of a single pixel. Moreover, in our algorithms, we will not consider cases where a single region in the watershed segmentation contains multiple non-connected seeds. Though those cases may happen in practice, they can be handled as a post-processing step by merging neighboring regions labeled by the same seeds.

![](images/27a93d1cf7d256ed73b09b355ab3fa2633c76e5c649a6b0c602abf3e65a53f5a.jpg)  
Fig. 5 Algorithmic overview of (hierarchical) watershed-cut segmentation.

## 5 Building a Binary Partition Tree

The Binary Partition Tree by Altitude Ordering (BPTAO) of an edge-weighted graph $G \ =$ $( V , E , w )$ can be eficiently built by a modified version of Kruskal’s minimum spanning tree (MST) algorithm. Indeed, Kruskal’s MST algorithm proceeds as follows:

1. At the start, the vertices of the graph are partitioned into singletons and the (incomplete) MST is empty;

2. Then, the edges of the graph are browsed in increasing weight order. Each time we find an edge $\{ x , y \}$ linking two vertices x and y belonging to two diferent regions, this edge is added to the MST and the two regions containing x and y are merged.

3. When all the vertices have been merged into a single region, the MST is complete, and the algorithm stops.

Thus, each step of the algorithm is associated with a partition of the vertices of the graph and the partition of step n + 1 is obtained by merging two regions of the partition of the step n (through an edge of the MST): in other words, the partition in step n is a refinement of the partition at step $n + 1$ . The BPTAO is indeed exactly equal to this sequence of partitions [14] and one way to obtain it is to record the region merges during the execution of Kruskal’s algorithm. This process is illustrated in Figure 6.

Kruskal’s algorithm relies on a union-find data structure to represent the current partition of the graph vertices, and, in order to be eficient, one must use union-find with path compression and rank optimization which collapses region merging sequences: one must thus record the merge sequence in a separate data-structure.

This is done in Algorithm 1, which implements the construction of the BPTAO based on Kruskal’s algorithm. In this algorithm, we assume that the graph vertices V are represented by integers from 0 to $| V | { - } 1$ . The nodes of the constructed BPTAO will also be represented by integers from 0 to $2 | V | - 1 \mathrm { : }$ : the V first nodes of this tree are indeed exactly the V vertices of the graph. The algorithm relies on a standard Tarjan’s union-find data structure with path compression and unionby-rank; a possible implementation is given in Algorithm 2 for reference. We can recognize the global structure of Kruskal’s algorithm with two new data-structures:

![](images/a86646f0ec6a1dbdfe93271df1b81e507fbd22469a869964c5fbef70d4e226a0.jpg)  
(a) Edge-weighted graph

![](images/ad392e701338467f0ac32e1478be1218449c1791a0ea812c4e8cf776daf5c979.jpg)

![](images/b992d0bc2c7415a6d30817257aa5b1d08474136a16b54680e08ce5be157e9272.jpg)

![](images/defce559446c9a0ac9ce04b27ca158bb56397e7462c169c2237458698ea24075.jpg)  
(c) Adding a node to the BPTAO  
(d) Final BPTAO  
Fig. 6 Some steps of the construction of the MST of an edge-weighted graph with Kruskal’s algorithm and the corresponding $\mathrm { B P T A O }$ . Top left: initial edge-weighted graph: each vertex of the graph represents a region. Top right: the algorithm first processes the edges {a, e}, {g, h} and $\{ c , d \}$ of weight 0 in any arbitrary order. The vertices corresponding to those edges are merged: this is represented by the red trees whose root nodes indicate the edge of the MST where the merge occurred. Bottom left: then the edge $\{ e , f \}$ is processed. The region $\{ a , e \}$ is merged with the singleton {f}. Bottom right: the process continues until the edge $\{ b , c \}$ is processed, completing the MST and leading to a partition with a single region containing all the vertices.

• An array parents of size 2 V 1: This array records the region merging operations, represented as a parent relation. For any node i of the tree, parents[i] is the parent node of i in the tree. During the execution of the algorithm, this array will indeed represent a forest of trees.

• An array roots of size V that maps the canonical elements of the regions in the union-find data structure to the roots of the trees in the forest represented by the array parents. For any vertex v of the graph, uf.find(v) is the canonical element representing the region containing v and roots[uf.find(v)] is the root of the tree representing the merging sequence that leads to this region.

Algorithm 1: Binary partition tree by Algorithm 2: Union-Find   
altitude ordering (BPTAO) Data: parents and ranks are two mappings   
Data: An edge-weighted graph G = (V, E, w) shared by the 3 functions.   
Result: Relation parents and MST MSTEdges 1 Function makeSet(x)   
numMSTEdges := 0; // Create a new set containing x   
2 MSTEdges := array of size |V| − 1; 2 parents[x] := x; ranks[x] := 0;   
3 UnionFind uf; 3 end   
4 parents := array of size 2|V| − 1;   
5 roots := array of size $| V | ;$ 4 Function find(x)   
6 foreach vertex $v \in V$ do // Finds the canonical element of   
the set containing x   
7 uf.makeSet(v);   
8 roots[v] := v; 5 if parents[x] ̸= x then   
6 parents[x] := find(parents[x]) ;   
9 end   
7 end   
// Modified Kruskal’s MST algorithm   
8 return parents[x];   
10 foreach edge $\{ i , j \} \in E$ by increasing weight   
$w ( \{ i , j \} )$ do 9 end   
11 $c _ { i } =$ uf.find(i); c = uf.find(j) ; Function $\left( c _ { x } , c _ { y } \right)$   
12 if $c _ { i } \neq c _ { j }$ then // Merges the sets represented by   
13 MSTEdges[numMSTEdges] := {i, j}; the canonical elements $c _ { x }$ and $c _ { y }$   
14 newRoot := uf.link(c , c ); 11 if ranks[c ] > ranks $: \left. c _ { y } \right.$ then   
15 numNode := |V| + numMSTEdges; 12 swap(c<sub>x</sub>, $c _ { y } )$   
16 parents[roots[c ]] := numNode; 13 end   
17 parents[roots[c ]] := numNode; 14 if ranks[c<sub>x</sub>] = rank $s / c _ { y } ]$ then   
18 roots[newRoot] := numNode; 15 ranks[c<sub>y</sub>] += 1;   
19 numMSTEdges += 1; 16 end   
20 end 17 parents[c<sub>x</sub>] := c<sub>y</sub>;   
21 end 18 return $c _ { y } ;$   
22 parents[2|V| − 2] = 2|V| − 2; // convention: 19 end   
parent of root is itself

Note that a more detailed description of this algorithm is given in [15] with an incremental transformation of Kruskal’s algorithm.

After the declaration of local variables (lines 1–5), the loop over each graph vertex v (lines 6– 9) creates a singleton v and defines that v is the root of the sub-tree associated with this singleton. Then, we have the main loop of Kruskal’s algorithm, where edges are considered in increasing order. For each edge i, j , we first identify the two regions $c _ { i }$ and $c _ { j }$ containing i and j (line 11). If those regions are diferent (line 12), i, j is part of the MST (line 13) and we merge the 2 regions $c _ { i }$ and $c _ { j }$ (line 14). We then record the merge operation by creating a new node in the forest (line 15). This new node becomes the parent of the sub-trees associated with the regions $c _ { i }$ and $c _ { j }$ (lines 16–17) and the root of the sub-tree associated with the merged region is the new node (line 18). Finally, the number of edges in the current MST is incremented.

Similar to Kruskal’s algorithm, Algorithm 1 runs in quasi-linear time $O ( | E | \alpha ( | V | ) )$ (where α is the inverse of the single-valued Ackermann function [93]) plus the time needed to sort the edge of the graph: O( E log( E )) in the general case or O( E ) if linear sorting is possible.

With the BPT already computed, all remaining algorithms in this paper require a fixed number of passes over this data structure, running in linear time relative to the number of nodes of the input BPTAO.

Note that at the end of Algorithm 1, the nodes in the parents array representing the tree are sorted in topological order from the leaves to the root. This means that browsing the tree nodes in (reverse) topological order is as simple as a forloop over the array. In the following, given a binary partition tree T and a node n of T we will use the following notations:

1. n.par is the parent of n (if n = root);

2. n.mst\_edge is the MST edge associated to n (for non-leaf nodes)

3. n.alt is the altitude of n, i.e. the weight of n.mst\_edge, the MST edge associated to this node for internal nodes and 0 otherwise;

4. n.c1 is the first child of n (for non-leaf nodes);

5. n.c2 is the second child of n (for non-leaf nodes).

## 6 Selection of seeds

Once the BPTAO of the input graph has been computed, the selection of seeds, which will determine the nature (flat or hierarchical) and the number of regions of the final segmentation, can be performed through various methods.

This section describes the most common methods for extracting seeds. For the sake of clarity, we consider the non-hierarchical and hierarchical cases separately.

## 6.1 Seeds for non-hierarchical watershed cuts

• Regional minima. Placing the seeds on the graph’s regional minima stems from the original definition of watershed, with the resulting graph-cut being compatible with the drop of water principle [9].

An algorithm to compute the regional minima of a graph is presented in Algorithm 3. This algorithm processes a BPTAO to identify nodes that represent minima, outputting a binary array indicating these nodes. It starts by initializing leaf nodes with no minima. Then, in a bottom-up traversal from leaves to the root (excluding the root), it calculates the number of minima in each node’s subtree by summing those of its children. A node is marked as a minimum if its subtree has no minima and its altitude value difers from its parent’s. Finally, the root is marked as a minimum if the entire tree has no minima. Algorithm 4 (MinimaOne-Leaf) processes a BPTAO and a minima array (precomputed by Algorithm 3) to produce a binary array minVertex, which marks exactly one leaf node as True for each minimum node in the tree. Starting from the root, it propagates the minima status of each minimum node to its first child in a top-down traversal, continuing until a leaf is reached. Then, it assigns minVertex[n] = minima[n] for each leaf, ensuring only the leaf reached by this propagation is marked True for each minimum. This approach selects a single representative leaf for each significant region (minimum) in the tree, useful for applications like image segmentation.

Algorithm 3: Minima   
Data: A BPTAO T   
Result: A binary array minima indicating   
which nodes of T are minima.   
1 foreach leaf node n ∈ T do   
2 numMin[n] :=0;   
3 minima[n] := False;   
4 end   
5 foreach node n of T from leaves (excluded)   
to root (excluded) do   
6 numMin[n] = numMin[n.c1] +   
numMin[n.c2];   
7 if numMin[n] = 0 and n.alt ̸= n.par.alt   
then   
8 minima[n] := True;   
9 numMin[n] := 1;   
10 end   
11 end   
12 if numMin[T.root] = 0 then   
13 minima[T.root] := True;   
14 end

Algorithm 4: MinimaOneLeaf   
Data: A BPTAO T   
Data: A binary array minima indicating   
which nodes of T are minima.   
Result: A binary array minVertex which is   
True for one and only one leaf node   
in each minimum.   
1 foreach node n of T from root to leaves   
(excluded) do   
2 if minima[n] then   
3 minima[n.c1] := True;   
4 end   
5 end   
6 foreach leaf node n of T do   
7 minVertex[n] = minima[n];   
8 end

• User scribbles. For certain applications, users can actively guide the segmentation algorithm by roughly drawing scribbles inside the objects of interest. However, this technique can quickly become cumbersome as the image complexity and size increase. To alleviate this problem, hybrid methods often couple user’s input with automatic segmentation, such as the one presented in [94], in which the users focus on the locations of “high uncertainty” and can interfere in case of segmentation errors.

• Supervised methods. Taking advantage of the deep learning developments in the last decade, Song et al. [95] and Da Fonseca et al. [96] propose supervised models for predicting user’s interactive seed or scribble inputs. Given a first point in the object of interest and a point in the background, both provided by the user, Song et al. [95] proposes a reinforcement learning model for generating a sequence of artificial user inputs in order to improve the precision of the resulting segmentation. In [96], the authors trained a model to predict what an average user’s scribble would look like and, then, input those learned scribbles into the watershed segmentation pipeline. Both methods can save user’s time and could be used in conjunction with interactive segmentation methods to fix eventual segmentation errors. More recently, Yang and Gong [97] employed the pre-trained Contrastive Language-Image Pretraining and Segment Anything Model to generate high-quality seeds for weakly supervised semantic segmentation. Finally, another way of tackling seed based segmentation with the help of supervised learning methods is to, instead of learning where seeds should be placed, learn high level image contours such that the local minima are placed in the regions of interest. The idea is to leverage the pre-trained deep learning methods trained on large semantic segmentation datasets (e.g.

[98]) to extract semantic knowledge from new input images in the form of fuzzy contours.

## 6.2 Weighted seeds for hierarchical watershed cuts

So far, we have considered non-weighted seeds with the aim of computing a flat watershed-cut segmentation, with the only parameters being the number of seeds and their location. By introducing weighted (or totally ordered) seeds, we can obtain a sequence of nested segmentations seeded in the i most important seeds, for i ranging from one (the coarsest segmentation level composed of a single region) to n (the finest segmentation level where each seed belongs to a distinct region). The seed weights can be interpreted as their “importance level”, where seeds with larger weights belong to the “most important image regions” which should be preserved at the coarsest segmentation levels. The main motivation for considering weighted seeds is that regions of interest are rarely present at a single segmentation level, and post-processing is often required to filter out spurious noise.

Extinction values, as already mentioned in Section 3.3, are a common measure for ordering the regional minima of a graph. Informally, the idea is that each regional minimum belongs to a catchment basin from which some simple properties can be extracted, including their area (number of pixels/vertices), volume (sum of pixels’ intensities) and dynamics (measure of contrast between the minima).

In order to obtain weighted seeds, we first compute increasing regional attributes (e.g. area, volume or depth) on the BPTAO nodes, which are then propagated from the root to the leaf nodes, resulting in the extinction value of each minimum (as well as their ancestors in the BPTAO).

Algorithm 5 describes the computation of extinction values given a BPTAO and an increasing attribute as input. Algorithm 5 is linear w.r.t. V and can be summarized as:

• First we need to correct the attribute values because some nodes of the BPTAO, those which correspond to sub-parts of a plateau, do not really represent any real component: they are artifacts of the binary tree structures. This situation is detected by checking if the altitude of a node n is equal to the altitude of its parent. In this case, its attribute value is replaced by the maximum of the attribute values of its children, i.e. in a filtering process, this component would disappear as soon as both its children are filtered out.

• Then comes the extinction value computation. We start by assigning (or any large enough value) as the extinction value of the root node. Next, for each node n from the root to the leaf-nodes (excluded), let maxA be the maximum of the attribute values of the children of n. Then, for each child c of n, the extinction value of c is equal to the extinction value of n if its attribute value is equal to maxA, and the attribute value of c otherwise.

Finally, to obtain weighted seeds from extinction values, one can simply first extract binary seeds representing the minima with Algorithms 3 and 4 and then replace the seeded minima vertices by their extinction values computed with Algorithm 5 (and non-seeded vertices are assigned a null weight).

## 7 Extracting watershed-cut edges from the BPTAO

In this section, we show how to compute a (weighted) watershed cut from a BPTAO (see Section 5) and a set of (weighted) seeds (see Section 6).

## 7.1 Watershed-cut from seeds

We now present Algorithm 6, that computes a watershed from an arbitrary set of seeds. The main idea is to run through all the nodes of the BPTAO, and determine, for each node, if there is a seed below it. An edge belongs to the watershed cut if both children of its corresponding node in the BPTAO contain a seed.

Apart from the BPTAO, Algorithm 6 requires a binary array seeded, which indicates whether each leaf node n of the tree contains a seed. At the start of the algorithm, the array value is True for seeded leaves of the tree, and False elsewhere. Algorithm 6 then browses the tree from leaves to root to determine which internal nodes have at least one seeded descendant. During a second pass, for any internal node n of the tree (in any order), the MST edge corresponding to n is marked as a watershed edge if both children of n contain at least one seed, i.e. if several marked regions merge at the node n. The algorithm returns a watershed cut represented by a binary array mapping each edge of the MST to True if the edge belongs to the cut and False otherwise.

Algorithm 5: extinctionValue   
Data: A BPTAO   
Data: An array attr representing an   
increasing attribute of T.   
Result: An array ext representing the   
extinction values of the nodes of T   
for attr.   
// Correct attribute values for   
non-canonical nodes of T   
1 foreach node n of T from leaves (excluded)   
to root (excluded) do   
2 if n.alt = n.par.alt then   
3 attr[n] := max(attr[n.c1], attr[n.c2]);   
4 end   
5 end   
// Compute extinction values   
6 ext[T.root] = ∞;   
7 foreach node n of T from root to leaves   
(excluded) do   
8 maxA := max(attr[n.c1], attr[n.c2]);   
9 if attr[n.c1] = maxA then   
10 ext[n.c1] = ext[n];   
11 else   
12 ext[n.c1] = attr[n.c1];   
13 end   
14 if attr[n.c2] = maxA then   
15 ext[n.c2] = ext[n];   
16 else   
17 ext[n.c2] = attr[n.c2];   
18 end   
19 end

Fig. 7 shows the result obtained with Algorithm 6 on a small example when one vertex of each minimum is seeded (see Section 6). In this particular case, the cut obtained at the end of the algorithm is what is usually referred to as an (unsupervised) watershed cut of the underlying graph [9].

Algorithm 6: watershedFromSeeds   
Data: A BPTAO T   
Data: A Boolean array seeded indicating   
which leaves of T are seeds.   
Result: A Boolean array ws indicating which   
MST edges belongs to the watershed   
cut.   
1 foreach node n of T from leaves (excluded)   
to root do   
2 seeded[n] := seeded[n.c1] or seeded[n.c2];   
3 end   
4 foreach non leaf node n of T do   
5 ws[n.mst\_edge] = seeded[n.c1] and   
seeded[n.c2];   
6 end

![](images/ade30ede7a9459488e802f06772a39ad1f7be0a13900419b04f17e84f06dd6c9.jpg)  
Fig. 7 Unsupervised watershed cut. The underlying graph has 3 minima: edges M (green), M (red) and M (blue). By applying Algorithm 6 to the BPTAO where one vertex of each minimum is seeded, we can identify the watershed nodes (blue points) and their associated MST edges {b, c} and {d, h}.

## 7.2 Hierarchical watershed cuts from weighted seeds

In the previous subsection, we have seen how watershed cuts can be obtained from a set of arbitrary binary seeds, which generalizes the fully unsupervised watershed cuts rooted in the minima of an input function. In the general case described above, all arbitrary seeds have the same “importance level”, meaning that we are only interested in the final pixels/vertices partition derived from those given seeds. We can further generalize seeded watershed cuts by considering that the input seeds are weighted (or ordered) according to a given criterion, which leads to not only a single partition rooted in this set of seeds, but instead to a sequence of partitions rooted in any subset of the n most important seeds (n being less or equal to the total number of seeds). As stated in [14], in the framework of edge-weighted graphs, such sequence of partitions satisfying the causality principle can always be defined through minimum spanning forests rooted in each subset of n most important input seeds.

As introduced in [15], hierarchical watersheds cuts from seeds can be computed in quasi-linear time (given that the input edges are already sorted) thanks to the same framework described in the previous subsection, by post-processing the BPTAO. As shown in previous sections, watershed-cut edges can be detected in a single pass on the non-leaf nodes of the tree, resulting in a binary array indicating which MST edges are watershed-cut edges. In other words, it provides us with a binary weighted MST from which a watershed-cut segmentation can be obtained by simply computing its connected components. Now, given that an ordered set of seeds is provided, we aim at computing not only the set of MST watershed-cut edges, but also their saliency value, which indicates their level of disappearance in the hierarchy. This can be achieved through a very simple adaptation of Algorithm 6.

In Algorithm 7, given a BPTAO and a set of weighted seeds (i.e., an array indicating for each leaf node n, the weight of the seed associated to n or 0 if n is not seeded), the result is an array hws indicating the saliency values of the MST edges. The algorithm itself is a direct generalization of Algorithm 6 for binary seeds: as classically done in mathematical morphology, binary operators or and and generalize to real numbers with the operators max and min. The first loop of the algorithm then computes the maximum seed weight value among the descendants of each node. Then, the second loop over internal nodes computes the minimum of the maximum weights of its two children, giving the saliency value of the corresponding MST edge.

Similar to the binary case, the levels of the hierarchical watershed associated to hws can be computed by extracting the connected components of the diferent levels sets of hws as described in the next section.

Algorithm 7: hierarchicalWatershed-  
FromSeeds   
Data: A BPTAO   
Data: An array of weights, such   
that for each leaf n of ${ \textsc { T } } , { \bar { \boldsymbol { w } } } .$ \_seeds[n] $\neq$   
0 indicates that n is a seed and the   
positive value w\_seeds[n] provides its   
weight.   
Result: An array hWS indicating the saliency   
of MST edges.   
1 foreach node n of T from leaves (excluded)   
to root do   
2 w\_seeds[n] := max(w\_seeds[n.c1],   
w\_seeds[n.c2]);   
3 end   
4 foreach non leaf node n of T do   
5 hWS[n.mst\_edge] = min(w\_seeds[n.c1],   
w\_seeds[n.c2]);   
6 end

## 8 Extraction of connected components

In this section, we describe how connected components of (hierarchical) watershed segmentations can be labeled once the watershed-cut edges have been extracted from the BPTAO. We can consider three cases:

1. Single segmentation from non-weighted seeds: given the BPTAO and the watershed-cut edges already computed in the previous step (Section 7.1), we aim to label the connected components of the resulting watershed segmentation. This can be done by simply employing any connected component labeling algorithm on the graph $G ^ { \prime } = ( V , E ^ { \prime } )$ , where $E ^ { \prime }$ is the set of building edges of the BPTAO that are not watershed-cut edges.

2. Single segmentation from weighted seeds: similarly, given the BPTAO and the weighted watershed-cut edges computed in the previous step (Section 7.2), let us say that we aim to extract a single segmentation level from the associated hierarchical watershed. The target segmentation level may be defined by (a) a threshold λ on the saliency value of watershed-cut edges; or (b) the number of regions it contains. In the first case, it sufices to discard the watershed-cut edges whose saliency values are lower than λ and proceed with a connected component labeling algorithm on the graph $G ^ { \prime } = ( V , E ^ { \prime } )$ , where $E ^ { \prime }$ is the set of building edges of the BPTAO that are not watershed-cut edges or whose saliency value is lower than λ. In the second case, if a precise number r of regions is required, we can keep only the $r - 1$ watershed-cut edges with the highest saliency values, and proceed in the same manner.

3. Tree representation of a hierarchical watershed: in order to keep all hierarchical segmentation levels, we may consider a tree representation of the resulting hierarchy. This can be achieved by computing the BPTAO from the MST $G ^ { \prime } = ( V , E , w ^ { \prime } )$ of $G ,$ whose edge set is the set of building edges of the BPTAO and $w ^ { \prime }$ assigns the saliency value to the watershed-cut edges and zero to the remaining ones.

## 9 Use cases of the watershed-cut framework

In this section, we present the detailed pipeline for some specific use cases of the framework corresponding to diferent routes in the global algorithmic scheme presented in Figure 5. The Python scripts for computing those use cases are presented in a Python Notebook relying on the Higra Hierarchical Graph Analysis – library.

More specifically, the algorithms described previously for computing BPTAO, watershed-cut edges, (weighted) seeds, and connected components are building blocks for several specific use cases, whose aims range from a single unsupervised watershed-cut segmentation to a tree representation of a seeded hierarchical watershed. In the following, we present the pipeline for handling the most important cases. We consider that the input is an RGB or gray-scale image whose contours are extracted using any given contour detection tool, and whose edge-weighted graph is computed directly from the extracted contours. In the demonstration Notebook, contours are extracted using the Structured Edge Detection [99].

## 9.1 Unsupervised watershed-cut

We start with the completely unsupervised (nonhierarchical) watershed-cut seeded in the set of minima of the input graph, as described in Section 3.1. Given an edge-weighted graph, the method relies on the following main steps, illustrated in Figure 8:

1. Construction of the BPTAO and associated MST (Section 5, Algorithm 1);

2. Identification of the minima seeds and their weights (Section 6.1, Algorithms 3 and 4);

3. Identification of watershed-cut nodes (Section 7.1, Algorithm 6);

4. Connected component labeling of the watershed-cut in the MST (Section 8).

Figure 9 demonstrates an application of this pipeline.

## 9.2 Seeded watershed cut

In this scenario, we consider a (non-hierarchical) watershed-cut seeded on a set of seeds which are not necessarily the regional minima of the input graph. As discussed in Section 6, we may consider several approaches for extracting seeds to handle specific applications (e.g. manual, or deep learning methods). Let us say that we are given an edgeweighted graph and a set of seeds represented as a set of vertices. We then proceed as follows (see Figure 10):

1. Construction of the BPTAO and associated MST (Section 5, Algorithm 1);

2. Identification of watershed-cut edges (Section 7.1, Algorithm 6);

3. Connected component labeling of the watershed-cut in the MST (Section 8).

This case is indeed a direct generalization of the previous one, where seeds are considered as an input. Figure 11 demonstrates an application of this pipeline.

## 9.3 Tree representation of an area-based hierarchical watershed

Let us now consider a tree representation of a hierarchical watershed by area as our final goal, as illustrated in Figure 12. Similarly to the first scenario, the seeds are the regional minima of the input graph. However, we will now weigh the seeds with area-based extinction values of their associated minima:

1. Construction of the BPTAO and associated MST (Section 5, Algorithm 1);

2. Computation of area-based extinction values on the nodes of BPTAO (Section 6.2, Algorithm 5);

3. Identification of the minima seeds and their weights (Section 6.1, Algorithms 3 and 4);

4. Computation of the saliency values of watershed-edges in the MST (Section 7.2, Algorithm 7);

5. Construction of the BPTAO of the MST weighted by the saliency values (Section 5, Algorithm 1).

The reader may note that the BPTAO algorithm (Algorithm 1) is employed twice, once on the input weighted graph G, and, in the end, on the reevaluated MST of G, which results in the tree representation of the area-based hierarchical watershed of G.

The left subfigure in Figure 13 demonstrates an application of this pipeline.

## 9.4 Watershed cut from a hierarchical watershed

Once a hierarchical watershed is computed, there are several ways of exploring this representation. Given that the input graph has n regional minima, we may want to extract a segmentation level composed of the k most important regions according to the attribute used during the hierarchy construction. In this case, we can reuse the first four steps described in the previous section. However, instead of computing the tree representation of the reweighted MST, we apply a threshold to obtain a cut, and then labelise the result.

![](images/715f1768ae084d9f549191c00416f3e274de47346c29a5056623a7538b514c89.jpg)  
Fig. 8 Unsupervised watershed-cut pipeline.

![](images/3404e34eef3fca7ac56d36851c486f11d22990fdebb7961ab73f97233cc63a4b.jpg)  
Fig. 9 Demonstration of an unsupervised watershed-cut pipeline. From left to right, we have the original image (source BSDS500 [38]), its gradient (computed using SED [99]), and its unsupervised watershed cut. In the cut, each region is filled with its average color, and the contours are drawn in red. Each region contains a single yellow point depicting the location of its associated minimum.

5. Threshold the reevaluated MST weights by keeping only the watershed-cut edges with the k largest weights; and

6. Connected component labeling of the watershed cut in the MST (Section 8).

Alternatively, we can give a specific threshold value on the watershed-cut nodes instead of the desired number of regions.

The right subfigure in Figure 13 demonstrates an application of this pipeline.

## 9.5 Seeded hierarchical watershed cut

In the previous use case, we assumed that the desired segmentation can be found by a single threshold on the saliency map. As this is not always the case, we can consider a seed based segmentation extraction, in which user input seeds S are used after a hierarchical watershed has been obtained, as explained in Section 3.4. Given the first four steps described in Section 9.3, we proceed as follows:

5. Construction of the BPTAO of the MST weighted by the saliency values obtained in step 4 of Section 9.3;

6. Identification of watershed-cut edges (Section 7.1, Algorithm 6);

7. Connected component labeling of the watershed-cut in the MST (Section 8).

Once the hierarchical watershed has been computed, we can see that the method is exactly the same as the seeded watershed method presented in Section 9.2, but applied to the BPTAO of the reweighted MST rather than the original graph.

Figure 14 demonstrates an application of this pipeline.

## 10 Conclusion

In this article, we have given a comprehensive review of a general framework for computing (hierarchical) watersheds in edge-weighted graphs.

![](images/b2b57bfd4e2ec5f558746ebcb40207c8d26e45fa5387eec328620485d9dee0bd.jpg)  
Fig. 10 Seeded watershed-cut pipeline.

![](images/93703d20fb3d7d1f56472673dd3612b009701d6eeba750da7205a1160222d4e2.jpg)  
Fig. 11 The image and gradient considered are the same as in Figure 9. The user provides two sets of pixels as seeds: a set of foreground seeds and a set of background seeds. On the left is the supervised watershed cut. Each region is filled with its average color, and the contours are drawn in red. Each region contains a single yellow point depicting the location of its associated seed. Right: the final segmentation obtained by grouping the regions associated with the foreground and background seeds. It can be seen that the contours are well placed overall with a small number of seeds; however, a gradient leak on the person’s shoulder causes a large error.

This framework revolves around three main ideas: 1) the binary partition tree by altitude ordering encodes the topology of the minima of the graph, 2) a subset of the graph vertices called seeds induces a set of watershed nodes in the tree and 3) those watershed nodes are in bijection with the watershed cut of the graph induced by the seeds. Moreover, if the seeds are weighted (for example according to an attribute value), it is then possible to associate a persistence value to the induced watershed nodes which then translates as a saliency map, i.e., a weighted graph-cut or a hierarchy, in the graph space. We provide simple and eficient algorithms for each step of this framework, and we demonstrate the modularity of the framework with several use cases. All the algorithms and use cases are demonstrated in the Python Notebook relying on the Higra Hierarchical Graph Analysis – library.

The usages and challenges around watersheds and especially watershed hierarchies of course go beyond the use cases presented in the article. We can cite among others: segmentation hierarchy filtering [100, 101], energy based optimization and non horizontal-cuts [36, 102] and combination with machine learning techniques [59, 103].

In future work we plan to adapt and develop new parallel, distributed and massively parallel algorithms (targeted for GPU), building on our existing works on out-of-core (hierarchical) watersheds [104].

## Declarations

• Funding: The work of L. Najman was partially supported by Khalifa University of Science and Technology under Award No. KU-INT-FSU-2026-004-8471000027.

• Conflict of interest: The authors declare no conflict of interest.

• Data availability: No datasets were generated or analysed during the current study.

• Author contributions: All authors contributed equally to the writing and revision of the manuscript.

## References

[1] Digabel, H., Lantuéjoul, C.: Iterative algorithms. In: Proc. 2nd European Symp. Quantitative Analysis of Microstructures in

![](images/7dd4df1f8cd1fba5d5469412a895f8408381e0dcccf1f230712275a79c9193ef.jpg)  
Fig. 12 Pipeline for computing a tree representation of a hierarchical watershed.

![](images/0be3daf85e7eb01a062ef0072ea979bb03df89ad2882348eeb5cfadbd27d2097.jpg)  
Fig. 13 Demonstration of tree-based representation and watershed cut from a hierarchical watershed. The image and gradient considered here are the same as in Figure 9. On the left is a saliency map of a hierarchical watershed by area. Each upper threshold of the saliency map represents a segmentation of the image. The right-hand image shows the watershed cut obtained by thresholding the saliency map at a level containing 20 regions.

Material Science, Biology and Medicine, vol. 19, p. 8 (1978). Riederer Verlag

[2] Beucher, S., Lantuéjoul, C.: Use of watersheds in contour detection. In: Proc. Int. Workshop on Image Processing, Sept. 1979, pp. 17–21 (1979)

[3] Vincent, L., Soille, P.: Watersheds in digital spaces: an eficient algorithm based on immersion simulations. IEEE Transactions on Pattern Analysis & Machine Intelligence 13(06), 583–598 (1991)

[4] Meyer, F.: Un algorithme optimal de ligne de partage des eaux. Procs. of 8e Congrès AFCET, Lyon-Villeurbanne 2, 847–859 (1991)

![](images/5c201762a21824a99ba0e9cbace64b1b2e5ed9e4850747418c940a0b54ee39d1.jpg)  
Fig. 14 Demonstration of seeded hierarchical watershedcut pipeline. The image and gradient considered here are the same as in Figure 9. The user provides two sets of seeds: a set of foreground seeds and a set of background seeds. On the left is a seeded hierarchical watershed cut based on area. Each region is filled with its average color, and the contours are drawn in red. Each region contains a single yellow point depicting the location of its associated seed. Right: the final segmentation obtained by grouping regions associated with the foreground and background seeds. Compared to Figure 11, the area-based hierarchical watershed helped reduce leakage in the raw gradient, leading to correct boundary placement around the person’s shoulder.

[5] Najman, L., Schmitt, M.: Watershed of a continuous function. Signal Processing 38(1), 99–112 (1994)

[6] Meyer, F.: Topographic distance and watershed lines. Signal Processing 38(1), 113–125 (1994)

[7] Couprie, M., Bertrand, G.: Topological gray-scale watershed transformation. In: Vision Geometry VI, vol. 3168, pp. 136–146 (1997). SPIE

[8] Roerdink, J.B., Meijster, A.: The watershed transform: Definitions, algorithms and

parallelization strategies. Fundamenta informaticae 41, 187–228 (2000)

[9] Cousty, J., Bertrand, G., Najman, L., Couprie, M.: Watershed cuts: Minimum spanning forests and the drop of water principle. IEEE Transactions on Pattern Analysis and Machine Intelligence 31(8), 1362–1374 (2008)

[10] Beucher, S.: Watershed, hierarchical segmentation and waterfall algorithm. In: Mathematical Morphology and Its Applications to Image Processing, pp. 69–76. Springer, Fontainebleau, France (1994)

[11] Najman, L., Schmitt, M.: Geodesic saliency of watershed contours and hierarchical segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 18(12), 1163–1173 (1996)

[12] Santana Maia, D., Cousty, J., Najman, L., Perret, B.: Characterization of graph-based hierarchical watersheds: theory and algorithms. Journal of Mathematical Imaging and Vision 62(5), 627–658 (2020)

[13] Cousty, J., Najman, L.: Incremental algorithm for hierarchical minimum spanning forests and saliency of watershed cuts. In: International Symposium on Mathematical Morphology and Its Applications to Signal and Image Processing, pp. 272–283 (2011). Springer

[14] Cousty, J., Najman, L., Perret, B.: Constructive links between some morphological hierarchies on edge-weighted graphs. In: ISMM, pp. 86–97 (2013). Springer

[15] Najman, L., Cousty, J., Perret, B.: Playing with Kruskal: algorithms for morphological trees in edge-weighted graphs. In: ISMM, pp. 135–146 (2013). Springer

[16] Kruskal, J.B.: On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the American Mathematical society 7(1), 48–50 (1956)

[17] Cousty, J., Najman, L., Kenmochi, Y.,

Guimarães, S.: Hierarchical segmentations with graphs: quasi-flat zones, minimum spanning trees, and saliency maps. Journal of Mathematical Imaging and Vision 60(4), 479–502 (2018)

[18] Perret, B., Chierchia, G., Cousty, J., Guimaraes, S.J.F., Kenmochi, Y., Najman, L.: Higra: Hierarchical graph analysis. SoftwareX 10, 100335 (2019)

[19] Levner, I., Zhang, H.: Classification-driven watershed segmentation. IEEE Transactions on Image Processing 16(5), 1437–1445 (2007)

[20] Angulo, J., Jeulin, D.: Stochastic watershed segmentation. In: Proc. of the 8th International Symposium on Mathematical Morphology, pp. 265–276 (2007)

[21] Wolf, S., Pape, C., Bailoni, A., Rahaman, N., Kreshuk, A., Kothe, U., Hamprecht, F.: The mutex watershed: eficient, parameterfree image partitioning. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 546–562 (2018)

[22] Tarabalka, Y., Chanussot, J., Benediktsson, J.A.: Segmentation and classification of hyperspectral images using watershed transformation. Pattern Recognition 43(7), 2367–2379 (2010)

[23] Grau, V., Mewes, A., Alcaniz, M., Kikinis, R., Warfield, S.K.: Improved watershed transform for medical image segmentation using prior information. IEEE Transactions on Medical Imaging 23(4), 447–458 (2004)

[24] Cousty, J., Najman, L., Couprie, M., Clément-Guinaudeau, S., Goissen, T., Garot, J.: Segmentation of 4D cardiac MRI: Automated method based on spatiotemporal watershed cuts. Image and Vision Computing 28(8), 1229–1243 (2010)

[25] Lin, G., Adiga, U., Olson, K., Guzowski, J.F., Barnes, C.A., Roysam, B.: A hybrid 3D watershed algorithm incorporating gradient cues and object models for automatic segmentation of nuclei in confocal image stacks.

Cytometry Part A: the journal of the International Society for Analytical Cytology 56(1), 23–36 (2003)

[26] Faessel, M., Jeulin, D.: Segmentation of 3D microtomographic images of granular materials with the stochastic watershed. Journal of Microscopy 239(1), 17–31 (2010)

[27] Fotos, G., Campbell, A., Murray, P., Yakushina, E.: Deep learning enhanced watershed for microstructural analysis using a boundary class semantic segmentation. Journal of Materials Science 58(36), 14390– 14410 (2023)

[28] Soille, P., Vincent, L.M.: Determining watersheds in digital pictures via flooding simulations. In: Visual Communications and Image Processing’90: Fifth in a Series, vol. 1360, pp. 240–250 (1990). SPIE

[29] Meyer, F., Maragos, P.: Multiscale morphological segmentations based on watershed, flooding, and eikonal PDE. In: International Conference on Scale-Space Theories in Computer Vision, pp. 351–362 (1999). Springer

[30] Sofou, A., Maragos, P.: Generalized flooding and multicue PDE-based image segmentation. IEEE Transactions on Image Processing 17(3), 364–376 (2008)

[31] Meyer, F.: Topographical Tools for Filtering and Segmentation 1: Watersheds on Node-or Edge-weighted Graphs. John Wiley & Sons, USA (2019)

[32] Najman, L., Schmitt, M.: A dynamic hierarchical segmentation algorithm. In: Mathematical Morphology and Its Applications to Signal Processing II, p. 10 (1994)

[33] Meyer, F.: The dynamics of minima and contours. In: Mathematical Morphology and Its Applications to Image and Signal Processing, pp. 329–336. Springer, Atlanta, GA, USA (1996)

[34] Benzécri, J.P., Blaise, S., Benier, B., Bellier, L.: L’analyse des Données: La Taxinomie.

Leçons sur l’analyse factorielle et la reconnaissance des formes et travaux du laboratoire de statistique de l’Université de Paris VI, vol. vol. 1. Dunod, Paris (1973)

[35] Leclerc, B.: Description combinatoire des ultramétriques. Mathématiques et Sciences humaines 73, 5–37 (1981)

[36] Guigues, L., Cocquerez, J.P., Le Men, H.: Scale-sets image analysis. International Journal of Computer Vision 68, 289–317 (2006)

[37] Arbelaez, P.: Boundary extraction in natural images using ultrametric contour maps. In: 2006 Conference on Computer Vision and Pattern Recognition Workshop (CVPRW’06), pp. 182–182 (2006). IEEE

[38] Arbelaez, P., Maire, M., Fowlkes, C., Malik, J.: Contour detection and hierarchical image segmentation. IEEE Transactions on Pattern Analysis & Machine Intelligence 33(5), 898–916 (2011)

[39] Stawiaski, J., Meyer, F.: Stochastic watershed on graphs and hierarchical segmentation. In: Proc Eur Conf Math Indust. Wuppertal, Germany (2010)

[40] Malmberg, F., Hendriks, C.L.L.: An eficient algorithm for exact evaluation of stochastic watersheds. Pattern Recognition Letters 47, 80–84 (2014)

[41] Bernander, K.B., Gustavsson, K., Selig, B., Sintorn, I.-M., Hendriks, C.L.L.: Improving the stochastic watershed. Pattern Recognition Letters 34(9), 993–1000 (2013)

[42] Felzenszwalb, P.F., Huttenlocher, D.P.: Eficient graph-based image segmentation. International Journal of Computer Vision 59(2), 167–181 (2004)

[43] Cayllahua-Cahuina, E., Cousty, J., Guimarães, S.J.F., Kenmochi, Y., Cámara-Chávez, G., de Albuquerque Araújo, A.: Hierarchical segmentation from a nonincreasing edge observation attribute. Pattern Recognition Letters 131, 105–112

(2020)

[44] Boykov, Y., Veksler, O., Zabih, R.: Fast approximate energy minimization via graph cuts. IEEE Transactions on Pattern Analysis and Machine Intelligence 23(11), 1222– 1239 (2002)

[45] Falcao, A., Stolfi, J., de Alencar Lotufo, R.: The image foresting transform: Theory, algorithms, and applications. IEEE transactions on pattern analysis and machine intelligence 26(1), 19–29 (2004)

[46] Cousty, J., Bertrand, G., Najman, L., Couprie, M.: Watershed cuts. In: Francis, B.G.J., Junior, B., de Mendonça, B.-N.U., Tomita, H.N.S. (eds.) International Symposium on Mathematical Morphology - International Symposium on Mathematical Morphology’07, 8th International Symposium, Proceedings, vol. 1, pp. 301–312. INPE, Rio de Janeiro, Brazil (2007)

[47] Cousty, J., Bertrand, G., Najman, L., Couprie, M.: Watershed cuts: Thinnings, shortest path forests, and topological watersheds. IEEE Transactions on Pattern Analysis and Machine Intelligence 32(5), 925–939 (2009)

[48] Allène, C., Audibert, J.-Y., Couprie, M., Cousty, J., Keriven, R.: Some links between min-cuts, optimal spanning forests and watersheds. In: Francis, B.G.J., Junior, B., de Mendonça, B.-N.U., Tomita, H.N.S. (eds.) International Symposium on Mathematical Morphology - International Symposium on Mathematical Morphology’07, 8th International Symposium, Proceedings, vol. 1, pp. 253–264 (2007)

[49] Couprie, C., Grady, L., Najman, L., Talbot, H.: Power watershed: A unifying graphbased optimization framework. IEEE Transactions on Pattern Analysis and Machine Intelligence 33(7), 1384–1399 (2010)

[50] Najman, L.: Extending the power watershed framework thanks to γ-convergence. SIAM Journal on Imaging Sciences 10(4), 2275– 2292 (2017)

[51] Grady, L.: Random walks for image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 28(11), 1768–1783 (2006)

[52] Machairas, V., Faessel, M., Cárdenas-Peña, D., Chabardes, T., Walter, T., Decenciere, E.: Waterpixels. IEEE Transactions on Image Processing 24(11), 3707–3716 (2015)

[53] Eschweiler, D., Spina, T.V., Choudhury, R.C., Meyerowitz, E., Cunha, A., Stegmaier, J.: CNN-based preprocessing to optimize watershed-based cell segmentation in 3D confocal microscopy images. In: 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), pp. 223–227 (2019). IEEE

[54] Wolf, S., Schott, L., Kothe, U., Hamprecht, F.: Learned watershed: End-to-end learning of seeded segmentation. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2011–2019 (2017)

[55] Lux, F., Matula, P.: DIC image segmentation of dense cell populations by combining deep learning and watershed. In: 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), pp. 236–239 (2019). IEEE

[56] Santana Maia, D., Pham, M.-T., Lefèvre, S.: Watershed-based attribute profiles with semantic prior knowledge for remote sensing image analysis. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 15, 2574–2591 (2022)

[57] Challa, A., Danda, S., Sagar, B.D., Najman, L.: Watersheds for semi-supervised classification. IEEE Signal Processing Letters 26(5), 720–724 (2019)

[58] Breiman, L.: Random forests. Machine Learning 45(1), 5–32 (2001)

[59] Challa, A., Danda, S., Sagar, B.S.D., Najman, L.: Triplet-watershed for hyperspectral image classification. IEEE Transactions on Geoscience and Remote Sensing 60, 1–14 (2022)

[60] Lebon, Q., Lefèvre, J., Cousty, J., Perret, B.: Interactive segmentation with incremental watershed cuts. In: Iberoamerican Congress on Pattern Recognition, pp. 189–200 (2023). Springer

[61] Lefèvre, J., Cousty, J., Perret, B., Phelippeau, H.: Out-of-core attribute algorithms for binary partition hierarchies. In: International Conference on Discrete Geometry and Mathematical Morphology, pp. 298–311 (2024). Springer

[62] Barbosa da Fonseca, G., Negrel, R., Perret, B., Cousty, J., Jamil F. Guimarães, S.: Hierarchy-based fuzzy segmentation and marker learning layer: theory and algorithms. Journal of Mathematical Imaging and Vision 67(4), 41 (2025)

[63] Lapertot, R., Chierchia, G., Perret, B.: Endto-end ultrametric learning for hierarchical segmentation. In: International Conference on Discrete Geometry and Mathematical Morphology, pp. 286–297 (2024). Springer

[64] Mangan, A.P., Whitaker, R.T.: Partitioning 3D surface meshes using watershed segmentation. IEEE Transactions on Visualization and Computer Graphics 5(4), 308–321 (1999)

[65] Philipp-Foliguet, S., Jordan, M., Najman, L., Cousty, J.: Artwork 3D model database indexing and classification. Pattern Recognition 44(3), 588–597 (2011)

[66] Edelsbrunner, H., Harer, J., Natarajan, V., Pascucci, V.: Morse-Smale complexes for piecewise linear 3-manifolds. In: Proceedings of the Nineteenth Annual Symposium on Computational Geometry, pp. 361–370 (2003)

[67] Edelsbrunner, H., Harer, J., Zomorodian, A.: Hierarchical Morse—Smale complexes for piecewise linear 2-manifolds. Discrete & Computational Geometry 30(1), 87–107 (2003)

[68] Edelsbrunner, H., Harer, J.: The persistent Morse complex segmentation of a

3-manifold. In: 3D Physiological Human Workshop, pp. 36–50 (2009). Springer

[69] Delgado-Friedrichs, O., Robins, V., Sheppard, A.: Skeletonization and partitioning of digital images using discrete Morse theory. IEEE Transactions on Pattern Analysis and Machine Intelligence 37(3), 654–666 (2014)

[70] Bertrand, G.: On topological watersheds. Journal of Mathematical Imaging and Vision 22(2), 217–230 (2005)

[71] Cousty, J., Bertrand, G., Couprie, M., Najman, L.: Collapses and watersheds in pseudomanifolds of arbitrary dimension. Journal of Mathematical Imaging and Vision 50(3), 261–285 (2014)

[72] Bertrand, G., Boutry, N., Najman, L.: Discrete Morse functions and watersheds. Journal of Mathematical Imaging and Vision 65(5), 787–801 (2023)

[73] Wu, T., Li, J., Li, T., Sivakumar, B., Zhang, G., Wang, G.: High-eficient extraction of drainage networks from digital elevation models constrained by enhanced flow enforcement from known river maps. Geomorphology 340, 184–201 (2019)

[74] Guilbert, E.: Surface network extraction from high resolution digital terrain models. Journal of spatial information science 2021(22), 33–59 (2021)

[75] Soille, P., Vogt, J., Colombo, R.: Carving and adaptive drainage enforcement of grid digital elevation models. Water resources research 39(12) (2003)

[76] Čomić, L., De Floriani, L., Papaleo, L.: Morse-Smale Decompositions for Modeling Terrain Knowledge. In: Cohn, A.G., Mark, D.M. (eds.) Spatial Information Theory, pp. 426–444. Springer, Berlin, Heidelberg (2005)

[77] Hiatt, M., Sonke, W., Addink, E.A., van Dijk, W.M., van Kreveld, M., Ophelders, T., Verbeek, K., Vlaming, J., Speckmann, B., Kleinhans, M.G.: Geometry and topology of

estuary and braided river channel networks automatically extracted from topographic data. Journal of Geophysical Research: Earth Surface 125(1), 2019–005206 (2020)

[78] Kleinhans, M.G., van Kreveld, M.J., Ophelders, T.A., Sonke, W.M., Speckmann, B., Verbeek, K.A.: Computing representative networks for braided rivers. Journal of Computational Geometry 10(1), 423–443 (2019)

[79] Ophelders, T., Schenfisch, A., Sonke, W., Speckmann, B.: Computing geomorphologically salient networks via discrete morse theory. In: 41st International Symposium on Computational Geometry (SoCG 2025), pp. 70–1 (2025). Schloss Dagstuhl–Leibniz-Zentrum für Informatik

[80] Kornilov, A.S., Safonov, I.V.: An overview of watershed algorithm implementations in open source libraries. Journal of Imaging 4(10), 123 (2018)

[81] Kornilov, A., Safonov, I., Yakimchuk, I.: A review of watershed implementations for segmentation of volumetric images. Journal of Imaging 8(5), 127 (2022)

[82] Liu, J.-J., Hou, Q., Cheng, M.-M., Feng, J., Jiang, J.: A simple pooling-based design for real-time salient object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3917–3926 (2019)

[83] Soria, X., Sappa, A., Humanante, P., Akbarinia, A.: Dense extreme inception network for edge detection. Pattern Recognition 139, 109461 (2023)

[84] von Gioi, R.G., Randall, G.: A brief analysis of the dense extreme inception network for edge detection. Image Processing On Line 12, 389–403 (2022)

[85] Allène, C., Audibert, J.-Y., Couprie, M., Keriven, R.: Some links between extremum spanning forests, watersheds and min-cuts. Image and Vision Computing 28(10), 1460– 1471 (2010)

[86] Beucher, S., Meyer, F.: The morphological approach to segmentation: the watershed transformation. In: Mathematical Morphology in Image Processing, pp. 433–481. CRC Press, Boca Raton, FL, USA (1992)

[87] Santana Maia, D., Cousty, J., Najman, L., Perret, B.: Properties of combinations of hierarchical watersheds. Pattern Recognition Letters 128, 513–520 (2019)

[88] Ronse, C.: Ordering partial partitions for image segmentation and filtering: Merging, creating and inflating blocks. Journal of Mathematical Imaging and Vision 49(1), 202–233 (2014)

[89] Salembier, P., Serra, J.: Flat zones filtering, connected operators, and filters by reconstruction. IEEE Transactions on Image Processing 4(8), 1153–1160 (1995)

[90] Vachier, C., Meyer, F.: Extinction value: a new measurement of persistence. In: IEEE Workshop on Nonlinear Signal and Image Processing, vol. 1, pp. 254–257 (1995). Neos Marmaras Greece

[91] Salembier, P., Wilkinson, M.H.: Connected operators. IEEE Signal Processing Magazine 26(6), 136–157 (2009)

[92] Breen, E.J., Jones, R.: Attribute openings, thinnings, and granulometries. Computer Vision and Image Understanding 64(3), 377–389 (1996)

[93] Tarjan, R.E.: Eficiency of a good but not linear set union algorithm. Journal of the ACM (JACM) 22(2), 215–225 (1975)

[94] Straehle, C.-N., Koethe, U., Knott, G., Briggman, K., Denk, W., Hamprecht, F.A.: Seeded watershed cut uncertainty estimators for guided interactive segmentation. In: 2012 IEEE Conference on Computer Vision and Pattern Recognition, pp. 765– 772 (2012). IEEE

[95] Song, G., Myeong, H., Lee, K.M.: Seednet: Automatic seed generation with deep reinforcement learning for robust interactive

segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1760–1768 (2018)

[96] da Fonseca, G.B., Negrel, R., Perret, B., Cousty, J., Guimaraes, S.J.F.: New hierarchy-based segmentation layer: towards automatic marker proposal. In: 2021 34th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), pp. 354–361 (2021). IEEE

[97] Yang, X., Gong, X.: Foundation model assisted weakly supervised semantic segmentation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 523–532 (2024)

[98] Mottaghi, R., Chen, X., Liu, X., Cho, N.-G., Lee, S.-W., Fidler, S., Urtasun, R., Yuille, A.: The role of context for object detection and semantic segmentation in the wild. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 891–898 (2014)

[99] Dollár, P., Zitnick, C.L.: Structured forests for fast edge detection. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 1841–1848 (2013)

[100] Soille, P.: Constrained connectivity for hierarchical image partitioning and simplification. IEEE Transactions on Pattern Analysis and Machine Intelligence 30(7), 1132– 1145 (2008)

[101] Perret, B., Cousty, J., Guimarães, S.J.F., Kenmochi, Y., Najman, L.: Removing nonsignificant regions in hierarchical clustering and segmentation. Pattern Recognition Letters 128, 433–439 (2019)

[102] Xu, Y., Carlinet, E., Géraud, T., Najman, L.: Hierarchical segmentation using treebased shape spaces. IEEE Transactions on Pattern Analysis and Machine Intelligence 39(3), 457–469 (2016)

[103] Chierchia, G., Perret, B.: Ultrametric fitting by gradient descent. Advances in Neural

Information Processing Systems 32 (2019)

[104] Lefèvre, J., Cousty, J., Perret, B., Phelippeau, H.: Out-of-core algorithms for binary partition hierarchies. Journal of Mathematical Imaging and Vision 67(2), 18 (2025)

## A Appendix: Proof of Theorem 4

Let us start this section with two results which can be straightforwardly derived from classical results on trees and on minimum spanning trees. To deduce these lemmas, the construction presented in Sec. III-B of [9] linking minimum spanning trees and forests can be considered.

Lemma 5 Let $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ be two sets of minima of G such that $\mathcal { M } _ { 2 } \subseteq \mathcal { M } _ { 1 }$ , then any minimum spanning forest relative to $\mathcal { M } _ { 1 }$ is included in a minimum spanning forest relative to $\mathcal { M } _ { 2 }$

Let X be a subgraph of (V, E). We denote by $V ( X )$ (resp. $E ( X ) )$ the set of vertices (resp. edges) of X. Let $u \in E ( X )$ .We write $X \ \backslash$ u for the graph $( V ( X ) , E ( X ) \setminus \{ u \} )$ . If $v ~ = ~ \{ x , y \} ~ \in$ $E \setminus E ( X )$ , we write $X \cup v$ for the graph $( V ( X ) \cup$ $\{ x , y \} , E ( X ) \cup \{ v \} )$

Lemma 6 (lemma 17 in [9]) Let M be a marker set and let Y be a spanning forest rooted in M. If, for any $u \in E ( Y )$ and any $v \in E$ such that $( Y \setminus u ) \cup v$ is a spanning forest rooted in M, we have $w ( u ) \leq w ( v )$ then Y is an MSF rooted in M.

Let us state a first result on hierarchical watersheds obtained from one-by-one filtrations. This result is a consequence of the fact that the attribute used to select the connected components which are kept in the resulting map is increasing and that the filtration series is one-by-one. This result is the key ingredient in the subsequent proof of Theorem 4

Let $\Phi = ( \Phi _ { 0 } ( G ) = G , \dots , \Phi _ { N } ( G ) )$ be a filtration series of G. We say that Φ is one-by-one if, for any $\lambda \in \{ 1 , \ldots , N \}$ , we have $n _ { \lambda } = n _ { \lambda - 1 } - 1$ where n and $n _ { \lambda - 1 }$ are the numbers of minima of $\Phi _ { \lambda } ( G )$ and $\Phi _ { \lambda - 1 } ( G )$ , respectively.

Lemma 7 Let $\begin{array} { l c l } { \Phi } & { = } & { ( \Phi _ { 0 } ( G ) } & { = } & { G , \dots , \Phi _ { N } ( G ) ) } \end{array}$ be a one-by-one filtration series of G. Let $\mathcal { H } =$ $( C _ { 0 } , \dots , C _ { N } )$ be a hierarchical watershed with respect to Φ. Let $i \in \{ 0 , \ldots , N \}$ . Then, for any edge u in the cut set induced by $C _ { i } , ~ f o r$ any $j \in \{ 0 , \ldots , i \}$ , we have $\Phi _ { i } ( G ) ( u ) = \Phi _ { j } ( G ) ( u )$

Proof In this proof, to simplify the notations, we write $\Phi _ { i }$ instead of $\Phi _ { i } ( G )$ when no confusion may occur. By abuse of notation, we also write $\Phi _ { i }$ to denote the weight map of the edge-weighted graph $\Phi _ { i } ( G )$ . The proof is done by induction.

(1) Induction base. If $i ~ = ~ 0$ , the property is trivial.

(2) Induction hypothesis. Let $i \geq 1$ . Assume that for any $\lambda \in \{ 0 , \ldots , i - 1 \}$ , for any edge u in the cut set induced by $C _ { \lambda } ,$ , and for any $j \in \{ 0 , \ldots , \lambda \}$ , we have $\Phi _ { \lambda } ( G ) ( u ) = \Phi _ { j } ( G ) ( u )$

(3) Induction. In order to complete the proof of Lemma 7 by induction, we are going to show that if the induction hypothesis holds true, then, for any edge $u \ = \ \{ x , y \}$ in the cut set induced by $C _ { i } ,$ and for any $j ~ \in ~ \{ 0 , \ldots , i \}$ we have $\Phi _ { i } ( u ) ~ = ~ \Phi _ { j } ( u )$ Let $u = \{ x , y \}$ be any edge in the cut set of $C _ { i } . \operatorname { L e t } R _ { x }$ (resp. $R _ { y } )$ be the region of $C _ { i }$ that contains x (resp. y). $\mathrm { A s } \ C _ { i }$ is a watershed cut of $\Phi _ { i }$ , there exists a unique minimum $M _ { x } \ ( \mathrm { r e s p . } \ M _ { y } )$ of $\Phi _ { i }$ that is included in $R _ { x }$ $\left( \mathrm { r e s p . ~ } R _ { y } \right)$ . We denote by $C ^ { \prime }$ the connected component in $\bar { C } ( G , \Phi _ { i - 1 } , \Phi _ { i - 1 } ( u ) )$ that contains u. Let $R _ { x } ^ { \prime }$ (resp. $R _ { y } ^ { \prime } )$ be the region of $C _ { i - 1 }$ that contains x (resp. y). As $C _ { i - 1 }$ is a watershed cut of $\Phi _ { i - 1 , \mathrm { ~ \tiny ~ ( ~ i ~ ) ~ } }$ there exists a unique minimum $M _ { x } ^ { \prime }$ (resp. $M _ { y } ^ { \prime } )$ of $\Phi _ { i - 1 }$ that is included in $R _ { x } ^ { \prime } ~ ( \mathrm { r e s p . } ~ R _ { y } ^ { \prime } )$ , and (ii) the minima $M _ { x } ^ { \prime }$ and $M _ { y } ^ { \prime }$ are distinct. Furthermore, by the drop of water principle, there exists a descending path from x (resp. y) to $\hat { M } _ { x } ^ { \prime }$ (resp. $M _ { y } ^ { \prime } )$ . Thus, $M _ { x } ^ { \prime }$ (resp. $M _ { y } ^ { \prime } )$ is included in $C ^ { \prime }$ . As H is a hierarchy, we have $R _ { x } ^ { \prime } \subseteq R _ { x }$ (resp. $R _ { y } ^ { \prime } \subseteq R _ { y } )$ . Hence, we deduce that $M _ { x } ^ { \prime } \subseteq R _ { x }$ (resp. $M _ { y } ^ { \prime } \subseteq R _ { y } )$

As Φ is one-by-one, at least one of the following statements holds true: (i) $M _ { x } ^ { \prime } \subseteq M _ { x } { \mathrm { ~ o r ~ ( i i ) ~ } } M _ { y } ^ { \prime } \subseteq M _ { y } ,$ otherwise the number of minima of $\Phi _ { i }$ would decrease by at least two with respect to the one of $\Phi _ { i - 1 }$ . This implies that we either have $M _ { x } \cap C ^ { \prime } \neq \emptyset$ or $M _ { y } \cap C ^ { \prime } \neq$ ∅. The graphs $M _ { x } , M _ { y }$ , and $C ^ { \prime }$ are connected components of $\Phi _ { i - 1 }$ (more precisely for each of them, there exists k such that it belongs to $C ( G , \Phi _ { i - 1 } , k ) )$ . It is well known that any two connected components of a weighted graph are either nested or disjoint. Thus, we have $M _ { x } \subseteq C ^ { \prime }$ or $M _ { y } \subseteq C ^ { \prime }$ . The minima $M _ { x }$ and $M _ { y }$ are two connected components of $\Phi _ { i } .$ Thus, the measure $\mu$ of $M _ { x }$ and $M _ { y }$ involved in the filtration is greater than the scale parameter $\lambda _ { i }$ associated to $\Phi _ { i }$ . As $\mu$ is increasing, we deduce that the measure $\mu$ of the region $C ^ { \prime }$ is also greater than the scale parameter $\lambda _ { i } .$ . Thus, $C ^ { \prime }$ is preserved as a connected component of $\Phi _ { i }$ at the same level as in $\Phi _ { i - 1 }$ $( i . e . , \ C ^ { \prime } \in C ( G , \Phi _ { i } , \Phi _ { i - 1 } ( u ) ) )$ . As u belongs to $C ^ { \prime } { \mathrm { . } }$ we deduce that $\Phi _ { i } ( u ) ~ = ~ \Phi _ { i - 1 } ( u )$ . As H is a hierarchy, since u belongs to the cut set of $C _ { i } { . }$ , it also belongs to the cut set of $C _ { i - 1 }$ . Thus, from the induction hypothesis, we deduce that $\Phi _ { i } ( u ) = \Phi _ { j } ( u )$ , for any $j \in \{ 0 , \ldots , i \}$ , which completes the proof. □

We first prove the forward implication of Theorem 4. It is a straightforward consequence of Theorem 3 or, more precisely, of the following theorem established by Allène et al.

Theorem 8 (from Theorem 5.2 in $\boldsymbol { \it 1 8 5 7 } )$ Let M be a marker set and let $( \Phi _ { 0 } , \Phi _ { 1 } )$ be a filtration series. Any minimum spanning forest rooted in M for Φ is a minimum spanning forest rooted in $\mathcal { M } \ f o r \ \Phi _ { 1 }$

Proof of the forward implication of Theorem 4 Let $\lambda \in \{ 0 , \ldots , N \}$ , and let us denote by $\mathcal { M } _ { \lambda }$ the set which contains every minimum M of G whose extinction value (for the filtration series Φ) is greater than or equal to λ $( i . e . , \epsilon ( M ) \ge \lambda )$ . Assume that condition (i) of Theorem 4 holds true. Then $F _ { \lambda }$ is a minimum spanning forest of $\Phi _ { 0 } ( G ) ~ = ~ G$ rooted in $\mathcal { M } _ { \lambda }$ . By applying the previous theorem along the filtration from $\Phi _ { 0 } ( G )$ to $\Phi _ { \lambda } ( G )$ , we obtain that $F _ { \lambda }$ is also a minimum spanning forest of $\Phi _ { \lambda } ( G )$ rooted in $\mathcal { M } _ { \lambda } .$ . Since $\mathcal { M } _ { \lambda }$ is the set of regional minima of $\Phi _ { \lambda } ( G )$ , Theorem 2 implies that $C _ { \lambda }$ is a watershed cut of $\Phi _ { \lambda } ( G )$ . This holds for every $\lambda ,$ and H satisfies the hierarchy principle by assumption. Therefore H is a hierarchical watershed with respect to $\Phi$ □

We will now focus on the backward implication of Theorem 4, namely on the following property.

Property 9 (hierarchical watershed optimality) Let $\Phi = ( \Phi _ { 0 } ( G ) = G , \ldots , \Phi _ { N } ( G ) )$ be a one-by-one filtration series of G. Let $\mathcal { H } = ( C _ { 0 } , \dots , C _ { N } )$ be a hierarchical watershed with respect to Φ. Then, for any value λ in $\{ 0 , \ldots , N \}$ , there exists a minimum spanning forest $F _ { \lambda }$ of G which is rooted in the set that contains every minimum of $G$ whose extinction value is greater than or equal to $\lambda$ and whose connected component partition is $C _ { \lambda }$

Proof In order to simplify the writing of the proof, for any $i \in \{ 0 , \ldots , N \}$ , let us denote by $\mathcal { M } _ { i }$ the set which contains every minimum M of G whose extinction value (for the filtration series Φ) is greater than or equal to i $( i . e . , \epsilon ( M ) \geq i )$ . In this proof, when no confusion may occur, we also write $\Phi _ { i }$ instead of $\Phi _ { i } ( G )$ Furthermore, by abuse of notation, we write $\Phi _ { i }$ for the weight map of the edge-weighted graph $\Phi _ { i } ( G )$ . The proof of Property 9 is made by induction.

(1) Induction base. As $\mathcal { M } _ { \mathrm { 0 } }$ is the set which contains every minimum of $G ,$ by Theorem 2, we can afirm that there exists a minimum spanning forest $F _ { 0 }$ of G which is rooted in $\mathcal { M } _ { 0 }$ and whose connected component partition is $C _ { 0 }$

(2) Induction hypothesis. Let $i \geq 1$ . Assume that for any $\lambda \in \{ 0 , \ldots , i - 1 \}$ , there exists a minimum spanning forest $F _ { \lambda }$ of G which is rooted in $\mathcal { M } _ { \lambda }$ and whose connected component partition is $C _ { \lambda }$

(3) Induction. In order to complete the proof of Property 9 by induction, we are going to show that if the induction hypothesis holds true, then, there also exists a minimum spannin forest $F _ { i }$ of $G$ which is rooted in $\mathcal { M } _ { i }$ and whose connected component partition is $C _ { i }$ . By induction hypothesis, there exists a minimum spanning forest $F _ { i - 1 }$ of $G$ which is rooted in $\mathcal { M } _ { i - 1 }$ and whose connected component partition is $C _ { i - 1 }$ . Since Φ is a one-by-one filtration, there exists a unique minimum $M ^ { \star } \in { \mathcal { M } } _ { i - 1 } \backslash { \mathcal { M } } _ { i }$ . Hence, as $C _ { i - 1 }$ is a watershed cut of $\Phi _ { i - 1 }$ , there exists a region $A _ { 1 }$ of $C _ { i - 1 }$ which includes $M ^ { \star }$ . Since H is a hierarchy, there exists a region B in $C _ { i }$ that includes $A _ { 1 }$ . Again, since H is a hierarchy, B is a union of regions of $C _ { i - 1 }$ Since Φ is one-by-one, the number of regions decreases by exactly one from $C _ { i - 1 }$ to $C _ { i }$ . Therefore, there exists a unique region A of $C _ { i - 1 }$ such that $B = A _ { 1 } \cup A _ { 2 }$ Let u be an edge of minimum weight (for the weight map $\Phi _ { 0 } )$ in the set that contains every edge of G which contains a vertex in $A _ { 1 }$ and a vertex in $A _ { 2 }$ . The existence of u follows from the fact that B is connected. We are going to establish that $F _ { i - 1 } \cup \{ u \}$ is an MSF of G rooted in $\mathcal { M } _ { i }$ whose connected component partition is $C _ { i }$ , which will complete the proof of Property 9. The connected component partition of $F _ { i - 1 } \cup \{ u \}$ is $C _ { i } .$ since adding u only "merges" the two components $A _ { 1 }$ and A of $F _ { i - 1 }$ . Moreover, it can be observed that $F _ { i - 1 } \cup \{ u \}$ is rooted in $\mathcal { M } _ { i }$ as the only minimum removed from $\mathcal { M } _ { i - 1 }$ to obtain $\mathcal { M } _ { i }$ is the one included in $A _ { 1 }$ , while A includes the unique minimum of $\Phi _ { i }$ included in B. By Lemma $5 ,$ as $\mathcal { M } _ { i } \subseteq \mathcal { M } _ { i - 1 }$ , the forest $F _ { i - 1 }$ is included in at least one MSF rooted in $\mathcal { M } _ { i }$ Any such forest is obtained from $F _ { i - 1 }$ by adding a single edge outgoing from $A _ { 1 }$ . Hence, by Lemma $6 ,$ in order to establish that $F _ { i - 1 } \cup \{ u \}$ is an MSF of G rooted in $\mathcal { M } _ { i }$ , it is suficient to prove that the weight for G of any edge v outgoing from $A _ { 1 }$ is greater than or equal to the weight of u. Let v be any edge outgoing from $A _ { 1 }$ . We distinguish two cases. First, if v contains a vertex in $A _ { 2 }$ , by definition of $u ,$ we have $\Phi _ { 0 } ( v ) \geq $ $\Phi _ { 0 } ( u )$ . Secondly, let us now assume that v does not contain a vertex in $A _ { 2 }$ . Thus, the edge v is outgoing from B and therefore belongs to the cut set of $C _ { i } .$ Let x and $_ y$ be such that $v \ = \ \{ x , y \}$ and $x \ \in \ A _ { 1 }$ Since Φ is one-by-one and $M ^ { \star } \subseteq A _ { 1 }$ is not included in any minimum of $\Phi _ { i } ,$ the unique minimum $M _ { x }$ of $\Phi _ { i }$ included in $B$ is included in $A _ { 2 }$ . As $C _ { i }$ is a watershed cut of $\Phi _ { i } ,$ and since v belongs to the cut set of $C _ { i } .$ there exists a path $\pi _ { x } = ( x _ { 0 } = x , \ldots , \ x _ { k } )$ which is descending for $\Phi _ { i }$ , which does not cross the cut set of $C _ { i }$ and which is such that $x _ { k }$ belongs to $M _ { x }$ . Thus, the path $\pi _ { x }$ remains inside B. Moreover, by definition of a watershed cut, the weight of any edge $v ^ { \prime }$ appearing in $\pi _ { x }$ satisfies $\Phi _ { i } ( v ^ { \prime } ) \leq \Phi _ { i } ( v )$ . Thus, since $x \in A _ { 1 }$ and $x _ { k } \in A _ { 2 }$ , there exists an edge $\boldsymbol { u } ^ { \prime } = \{ x _ { j } , x _ { j + 1 } \}$ along $\pi _ { x }$ such that $x _ { j } ~ \in ~ A _ { 1 }$ and $x _ { j + 1 } ~ \in ~ A _ { 2 }$ . By definition of $u ,$ we have

$$
\Phi _ { 0 } ( u ^ { \prime } ) \geq \Phi _ { 0 } ( u )\tag{1}
$$

As previously observed, we also have:

$$
\Phi _ { i } ( v ) \geq \Phi _ { i } ( u ^ { \prime } ) .\tag{2}
$$

Moreover, by definition of the filtration sequence (made from extensive connected operators), we also have:

$$
\Phi _ { i } ( u ^ { \prime } ) \geq \Phi _ { 0 } ( u ^ { \prime } ) .\tag{3}
$$

As v belongs to the cut set induced by $C _ { i }$ , by Lemma 7, we deduce that $\Phi _ { i } ( v ) = \Phi _ { 0 } ( v )$ . Thus, from Equation 2, we deduce that $\Phi _ { 0 } ( v ) \geq \Phi _ { i } ( u ^ { \prime } )$ . Thus, from Equation 3, we also have $\Phi _ { 0 } ( v ) \geq \Phi _ { 0 } ( u ^ { \prime } )$ , which, from Equation 1 implies that $\Phi _ { 0 } ( v ) \geq \Phi _ { 0 } ( u )$ , which completes the proof of Property 9. □