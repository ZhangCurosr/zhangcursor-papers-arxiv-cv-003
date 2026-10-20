# RxGS: Relightable Gaussian Stylization via BRDF Decomposition

Wanrong Hong<sup>1</sup>, Liu Jiang<sup>2</sup>, Mukesh Prasad<sup>3</sup>, Jun Li<sup>1\*</sup>

<sup>1</sup>School of Computer Science, University of Technology Sydney, Sydney, Australia.

<sup>2</sup>Department of Civil and Environmental Engineering, Stanford University, Stanford, United State.

<sup>3</sup>Australian Artificial Intelligence Institute, University of Technology Sydney, Sydney, 2007, Australia.

\*Corresponding author(s). E-mail(s): jun.li@uts.edu.au;

## Abstract

Reference-based 3D scene stylization transfers the appearance of a reference image onto a reconstructed 3D scene while preserving scene geometry. Recent methods perform this transfer by optimizing the appearance representation of neural scene models including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). These methods optimize an RGB appearance representation, in which a single per-point colour combines the efects of material reflectance, environment illumination, together with the shading it produces, and the viewing direction. When stylization modifies this combined colour, the illumination efects present during stylization are baked into the stored appearance, so the stylized scene cannot be rendered under diferent illumination. To address this limitation, we present RxGS, a relightable 3D scene stylization framework built on a Bidirectional Reflectance Distribution Function (BRDF)-aware 3D Gaussian representation. Rather than modifying a per-point RGB colour, RxGS optimizes only the difuse albedo field recovered by inverse rendering, while leaving roughness, metallicity, surface normals, and the estimated environment lighting unchanged. As a result, the stylized scene keeps all the components a physicallybased renderer needs other than the edited albedo, and can therefore still be relit. RxGS optimizes the albedo field using a nearest-neighbour feature matching loss together with a spatial smoothness prior over Gaussian centres. We evaluate the method using a Geometry Consistency Cost (GCC) based on optical-flow warping error. RxGS achieves the lowest GCC on all evaluated NeRF-Synthetic

![](images/e4bc4d6d549bddab0a56c89ded10363f89ecd4e9a76f0b92d65f1173d96e9c91.jpg)  
Fig. 1 Comparison between stylization on an RGB appearance representation and stylization on a BRDF-aware scene representation. Existing methods transfer appearance onto a per-point RGB colour, which already combines material reflectance and illumination. The illumination present during stylization is therefore written into the stored colour, and the result no longer changes when the lighting changes. RxGS instead modifies only the difuse albedo and leaves the estimated lighting representation unchanged, so the stylized albedo can be rendered under diferent illumination conditions, producing relightable stylized scenes without re-running the stylization.

scenes and remains competitive on real-captured scenes, while additionally supporting relightable stylization, which existing reference-based 3D scene stylization methods do not provide.

Keywords: Relightable rendering, 3D scene stylization, BRDF-aware representation, Neural rendering, 3D Gaussian Splatting

## 1 Introduction

Reference-based 3D scene stylization transfers the appearance of an artistic reference image onto a reconstructed 3D scene while keeping the rendered result geometrically consistent across viewpoints. Recent neural rendering methods, in particular Neural Radiance Fields (NeRF) [1] and 3D Gaussian Splatting (3DGS) [2], produce highquality novel views and have prompted a growing body of 3D stylization work. These methods stylize a scene by modifying the RGB appearance representation that the neural scene model has learned. In many applications a scene must be stylized once and then rendered under several diferent lighting conditions without repeating the stylization. This requires the stylized scene to keep a representation that can be relit after the appearance transfer, a requirement the RGB representation does not meet, as the next paragraph argues.

An RGB appearance representation stores, at each point, one colour that already combines material reflectance, the environment illumination together with the shading it produces, and view-dependent efects. Because these factors are not stored separately, a stylization process that edits this colour necessarily edits the illumination contribution together with the intrinsic material. The illumination present during stylization is thereby written into the stored appearance: when the scene is later rendered under a diferent illumination, the stored colour does not change, so no new shading or shadows appear. As illustrated in Fig. 1, stylization applied to an RGB appearance representation fixes the artistic appearance and the illumination contribution together, which prevents the result from responding to novel lighting.

To remove this restriction we build on recent inverse rendering methods for 3D Gaussian representations. Such methods [3] recover separate scene attributes, namely difuse albedo, material properties, and environment illumination, and so support relightable rendering and material editing. These separated representations have not previously been used for reference-based 3D scene stylization.

We present RxGS, a relightable 3D scene stylization framework built on a Bidirectional Reflectance Distribution Function (BRDF)-aware Gaussian scene representation. Given a reconstructed scene, RxGS first applies inverse rendering decomposition to recover, per Gaussian, the difuse albedo ${ \bf a } _ { i } ,$ the roughness $r _ { i } ,$ , the metallicity $m _ { i } ,$ , the surface normal $\mathbf { n } _ { i } ,$ and one global environment lighting $L .$ Stylization then optimizes only the field of difuse albedos $\left\{ \mathbf { a } _ { i } \right\}$ and leaves $\{ r _ { i } , m _ { i } , \mathbf { n } _ { i } \}$ and L and all geometry unchanged. Relightability follows from this bookkeeping rather than from any claim that the albedo alone fixes the rendered colour. A physically-based renderer consumes the geometry, the per-Gaussian material attributes $\{ \mathbf { a } _ { i } , r _ { i } , m _ { i } , \mathbf { n } _ { i } \}$ , and the environment lighting $L .$ RxGS changes only the albedo coeficients $\{ \mathbf { a } _ { i } ^ { \star } \}$ and holds every other input fixed, so all non-albedo inputs the renderer needs remain intact, and the same renderer can be re-run under any chosen L to produce a relit image of the stylized scene. To optimize the albedo field, RxGS uses a nearest-neighbour feature matching (NNFM) loss in the VGG feature space together with a spatial smoothness term over Gaussian centres.

We evaluate RxGS on synthetic and real-captured scenes, measuring crossview geometric consistency with the Geometry Consistency Cost (GCC) defined in Section 4. RxGS attains the lowest GCC on every evaluated NeRF-Synthetic scene and remains competitive on real-captured scenes, while keeping a relightable scene representation after stylization. Our contributions are the following.

• We present RxGS, a relightable 3D scene stylization framework based on a BRDFaware Gaussian scene representation.

• We formulate stylization as optimization of the difuse albedo field alone in a logit parameterization, regularized by a smoothness term on a k-nearest-neighbour graph over Gaussian centres, so that artistic appearance transfer leaves the material and illumination parameters required for relighting unchanged.

• We evaluate on synthetic and real-captured datasets, reporting cross-view geometric consistency (GCC) that is competitive with prior stylization methods together with relightable stylization that prior methods do not provide.

## 2 Related work

## 2.1 3D Scene Representation and Rendering

A 3D scene representation encodes scene geometry and appearance in a compact, learnable form that supports downstream tasks such as novel view synthesis, scene editing, relighting, and scene understanding. NeRF [1] introduced an implicit neural representation that parameterizes volumetric density and radiance with multilayer perceptrons, achieving novel view synthesis from multi-view observations. NeRF requires dense volumetric sampling during both training and rendering, which is computationally expensive. To reduce this cost, later work introduced voxel grids, tensor factorizations, and multi-resolution encodings, including DVGO [4], Plenoxels [5], TensoRF [6], and Instant-NGP [7]. As scene editing and stylization tasks developed, several methods adopted surface-based representations that give a more explicit geometric correspondence than a purely volumetric field, including VolSDF [8], NeuS [9], NeuMesh [10], and NeRF2Mesh [11].

Among explicit representations, 3D Gaussian Splatting (3DGS) [2] represents a scene with a set of anisotropic Gaussian primitives. Unlike volumetric methods that rely on dense ray sampling, 3DGS renders through diferentiable Gaussian rasterization, which is fast enough for real-time rendering while still reconstructing scene geometry and appearance. Following 3DGS, many extensions adapt the representation to downstream tasks, including scene editing, relighting, and inverse rendering. The inverse rendering line, which recovers separate material and lighting attributes for each Gaussian, is the one RxGS builds upon, and we review it next.

## 2.2 Inverse Rendering with 3D Gaussian Representations

Inverse rendering recovers physically meaningful scene properties, namely geometry, material attributes, and illumination, from image observations. Several methods built this decomposition into neural rendering frameworks. NeRFactor [12] factorizes scene appearance into geometry, reflectance, visibility, and illumination for relightable rendering. PhySG [13] uses spherical Gaussian representations of the BRDF and of the illumination to put physically-based shading into neural rendering. TensorIR [14] introduces a tensor-based inverse rendering framework that makes scene decomposition and relightable rendering more eficient.

After 3DGS, inverse rendering was extended to Gaussian representations by attaching material, normal, and illumination attributes to the Gaussian primitives. GaussianShader [15] adds a shading function to Gaussian rendering and estimates geometry-aware normals to improve reflective surfaces. Relightable 3D Gaussian [16] attaches BRDF attributes and incident lighting to each Gaussian and renders through diferentiable physically-based rendering. GS-IR [3] formulates inverse rendering on a BRDF-aware Gaussian representation by jointly estimating scene geometry, material properties, and environment illumination from multi-view observations; RxGS takes the decomposition produced by GS-IR as its input.

## 2.3 3D Scene Stylization

Neural style transfer was first developed for images [17–20], where the goal is to synthesize an image that keeps the scene content of one input while matching the appearance of a reference style image. As 3D scene representations matured, stylization was extended from 2D images to 3D scenes. Before neural scene representations, 3D stylization operated on explicit representations such as meshes and point clouds [21–23], manipulating surface textures, geometric attributes, or rendered appearance directly.

Neural scene representations let stylization act on a learned appearance representation, so that style is propagated consistently across viewpoints. Early NeRF-based methods modify the appearance field of a reconstructed scene while keeping the geometry fixed [24–29]. ARF [25] optimizes the appearance field of a NeRF with a neural style transfer objective, giving multi-view consistent stylization, and ARF-plus [30] improves stylization quality and training stability through stronger appearance representations and optimization strategies. Ref-NPR [31] improves controllability by transferring appearance from reference images through geometry-aware feature correspondence, which enables localized stylization. LAE-NeRF [32] lets a user mark target regions in a neural scene representation and propagates the edit through neighbouring volumetric regions. SNeRF [27] alternates between image-space stylization and scene reconstruction, transferring artistic appearance into the neural scene representation across iterations.

Because 3DGS associates appearance attributes directly with individual primitives, it makes appearance manipulation explicit. StylizedGS [33] supports region-specific style transfer through semantic masks. ReGS [34] adapts 3DGS to reference-based stylization with a texture-guided mechanism that adjusts local Gaussian arrangements to reproduce texture detail from the reference. StyleSplat segments individual objects in a Gaussian scene and fine-tunes the selected Gaussians for object-level stylization. Further methods match segmented regions across views before transfer to keep the style consistent over the whole scene [35], geometry-aware texture transfer [36], difusion-based stylization and distillation [37], and feed-forward or optimization-free stylization of Gaussian splats [38, 39]. All of these methods, whether NeRF-based or Gaussian-based, edit a per-point RGB appearance, and so produce a stylized scene that stores one fixed colour per point and cannot be relit.

A smaller line of work stylizes a decomposed appearance representation rather than RGB colour. TANGO [40] performs text-driven stylization by editing BRDF attributes on a relightable mesh, which allows appearance editing under varying illumination. RxGS difers from this mesh-based, text-driven setting: it performs reference-image stylization directly on a BRDF-aware Gaussian representation, editing only the difuse albedo field of the Gaussians and keeping the rest of the decomposition fixed.

## 3 Method

The task is to modify the appearance of a 3D scene, reconstructed from posed multiview images $\{ I _ { k } , P _ { k } \} _ { k = 1 } ^ { K } ,$ so that it matches the style of a reference image $^ { J , }$ while

![](images/f940e6ee12d34b5e17231f8a218aa15cb46acba13278751411187b531e2ddc00.jpg)  
Fig. 2 Overview of RxGS. Given posed multi-view images and a reference style image, a BRDFaware Gaussian scene representation is first reconstructed with GS-IR, which separates the scene into geometry, difuse albedo, material attributes (roughness and metallicity), surface normals, and one global environment lighting. RxGS stylizes by optimizing only the difuse albedo field, with a nearestneighbour feature matching (NNFM) loss and a spatial smoothness term, while leaving roughness, metallicity, surface normals, and the lighting unchanged. The stylized BRDF-aware representation therefore keeps its relightable property and can be rendered under novel illumination without further optimization.

keeping both the scene geometry and the physical rendering components that are needed to render the scene under novel illumination.

## 3.1 Gaussian Representation

## 3.1.1 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [2] represents a scene with a set of anisotropic 3D Gaussian primitives placed in 3D space. Each Gaussian is parameterized by a centre $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ , a scaling vector $\mathbf { s } _ { i } \in \mathbb { R } ^ { 3 }$ , a rotation quaternion $\mathbf { q } _ { i } \in \mathbb { R } ^ { 4 }$ , an opacity $\alpha _ { i } \in [ 0 , 1 ]$ and appearance-related attributes. The scaling vector and rotation quaternion together define the Gaussian covariance. The trainable parameters of the i-th Gaussian are

$$
G _ { i } = \{ \mathbf { x } _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } , \alpha _ { i } , \phi _ { i } \} ,\tag{1}
$$

where $\phi _ { i }$ denotes the appearance representation of the Gaussian.

Because the Gaussian primitives are defined in 3D space, the appearance attributes of a Gaussian are shared by every viewpoint that observes the same spatial region. This gives a fixed correspondence between scene geometry and appearance during rendering, so a change to a Gaussian’s appearance attributes afects all views of that region in the same way, which is useful for keeping a stylization consistent across viewpoints.

## 3.1.2 BRDF-aware Gaussian Representation

GS-IR [3] refines the 3DGS appearance attribute $\phi _ { i }$ into a BRDF-aware decomposition consisting of difuse albedo, roughness, metallicity, and surface normal, while jointly estimating a global environment lighting $L ,$ represented as an environment map, that is, a function on the unit sphere of incident directions giving the incoming radiance from each direction. Specifically,

$$
\phi _ { i } = \{ { \bf a } _ { i } , r _ { i } , m _ { i } , { \bf n } _ { i } \} ,\tag{2}
$$

where $\mathbf { a } _ { i } \in [ 0 , 1 ] ^ { 3 }$ is the difuse albedo of the i-th Gaussian, $r _ { i } \in [ 0 , 1 ]$ is the roughness, $m _ { i } \in [ 0 , 1 ]$ is the metallicity, and $\mathbf { n } _ { i } \in \mathbb { R } ^ { 3 }$ is the surface normal.

In an RGB appearance representation a single per-point colour already combines material reflectance, the illumination together with the shading it produces, and viewdependent efects. The decomposition above instead stores the albedo $\mathbf { a } _ { i }$ as a separate per-Gaussian coeficient, alongside $r _ { i } , m _ { i } , \mathbf { n } _ { i }$ , and the global lighting L. Although the difuse albedo is represented separately, it still influences the colour of specular reflections under the metallic-roughness BRDF when metallicity is non-zero. Therefore, modifying the albedo changes both the difuse appearance and the tint of specular highlights. However, the remaining BRDF attributes, including roughness, metallicity, surface normals, and environment lighting, remain unchanged throughout stylization. Consequently, the scene retains the complete set of rendering parameters required for relightable rendering.

## 3.2 RxGS: Relightable Gaussian Stylization

RxGS takes as input posed multi-view images $\{ I _ { k } , P _ { k } \} _ { k = 1 } ^ { K }$ and a reference style image $J .$ Let the BRDF-aware scene representation be

$$
{ \cal S } = \{ { \mathcal { G } } , L \} ,
$$

where $\mathcal { G } = \{ G _ { i } \} _ { i = 1 } ^ { N }$ is the set of N Gaussian primitives and $L$ is the estimated environment lighting. RxGS optimizes the difuse albedo field while keeping all remaining BRDF attributes, geometry, and environment lighting fixed.

We separate the parameters of each Gaussian into geometry-related and appearance-related parts,

$$
G _ { i } = \{ G _ { i } ^ { \mathrm { g e o } } , \phi _ { i } \} ,\tag{3}
$$

where $G _ { i } ^ { \mathrm { g e o } } = \{ \mathbf { x } _ { i } , \mathbf { s } _ { i } , \mathbf { q } _ { i } , \alpha _ { i } \}$ . The stylized appearance decomposition is

$$
\mathbf { \nabla } \phi _ { i } ^ { \star } = \{ \mathbf { a } _ { i } ^ { \star } , r _ { i } , m _ { i } , \mathbf { n } _ { i } \} ,\tag{4}
$$

where only the difuse albedo $\mathbf { a } _ { i } ^ { \star }$ is optimized; the geometry attributes $G _ { i } ^ { \mathrm { g e o } }$ , the roughness $r _ { i } ,$ , the metallicity $m _ { i } .$ , the surface normal $\mathbf { n } _ { i } ,$ and the environment lighting L are all held fixed. Fig. 2 gives an overview.

Existing reference-based 3D scene stylization methods perform style transfer on rendered RGB images. Given a rendered image I, the stylized result is

$$
{ \tilde { I } } = h ^ { J } ( I ) ,\tag{5}
$$

where $h ^ { J } ( \cdot )$ is a style transfer operator conditioned on the reference style image $J .$ Because the rendered RGB image already combines material reflectance, shading, illumination, and view-dependent efects at each pixel, $h ^ { J }$ modifies all of these together. The illumination present in I is thereby encoded in <sup>˜</sup>I. When the scene is later rendered under a diferent illumination, the stored stylized colour does not change, so the result shows no new illumination-dependent variation.

RxGS instead modifies only the difuse albedo field $\{ \mathbf { a } _ { i } ^ { \star } \}$ and leaves $\{ r _ { i } , m _ { i } , \mathbf { n } _ { i } \}$ and $L$ unchanged. Since only the albedo coeficient is changed and $r _ { i } , m _ { i } , \mathbf { n } _ { i } , L$ , and the geometry are held fixed, every rendering input required by the physically-based renderer other than the albedo is intact. The same renderer can therefore be run again under any chosen L, and it produces illumination-dependent variation when relit.

## 3.2.1 Albedo Parameterization

Because only the difuse albedo is optimized, the optimization must keep its value in the valid reflectance range $[ 0 , 1 ] ^ { 3 }$ . Optimizing $\mathbf { a } _ { i } ^ { \star }$ directly could leave this range. RxGS therefore parameterizes the stylized albedo $\mathbf { a } _ { i } ^ { \star } \in [ 0 , \bar { 1 } ] ^ { 3 }$ through an unconstrained latent variable $\mathbf { z } _ { i } \in \mathbb { R } ^ { 3 }$ in logit space. Let $\mathbf { a } _ { i , \mathrm { o r i g } }$ denote the original albedo of Gaussian $G _ { i }$ from the BRDF-aware decomposition. The component-wise inverse sigmoid $\sigma ^ { - 1 }$ diverges at 0 and 1, so before applying it we clamp each channel of $\mathbf { a } _ { i , \mathrm { o r i g } }$ to $[ \varepsilon , 1 - \varepsilon ]$ with $\varepsilon = 1 0 ^ { - 6 }$ . We initialize

$$
\mathbf z _ { i , 0 } = \sigma ^ { - 1 } ( \mathrm { c l a m p } ( \mathbf a _ { i , \mathrm { o r i g } } ) ) ,\tag{6}
$$

and recover the stylized albedo as

$$
\mathbf { a } _ { i } ^ { \star } = \sigma ( \mathbf { z } _ { i } ) ,\tag{7}
$$

where σ is the component-wise sigmoid function and $\sigma ^ { - 1 }$ its inverse. With this parameterization $\mathbf { a } _ { i } ^ { \star }$ stays in $[ 0 , 1 ] ^ { 3 }$ for every value of $\mathbf { z } _ { i }$ , and at initialization $\mathbf { a } _ { i } ^ { \star }$ equals the clamped recovered albedo.

## 3.2.2 Rendering Strategy

The style supervision is computed on images rendered from the stylized scene. Let

$$
\mathcal { G } ^ { \star } = \{ G _ { i } ^ { \star } \} _ { i = 1 } ^ { N } , \qquad G _ { i } ^ { \star } = \{ G _ { i } ^ { \mathrm { g e o } } , \phi _ { i } ^ { \star } \} ,
$$

denote the stylized Gaussian scene. The direct strategy renders the stylized albedo field without illumination,

$$
\hat { I } _ { k } = R _ { a } ( \mathcal { G } ^ { \star } , P _ { k } ) ,\tag{8}
$$

where $R _ { a } ( \cdot )$ is albedo rendering, which composites the per-Gaussian albedo $\mathbf { a } _ { i } ^ { \star }$ along each ray without applying any shading. Style supervision is then applied directly in the albedo domain.

We additionally consider a randomized-illumination strategy. Here, at optimization step t, a lighting $\dot { L } ^ { ( t ) }$ is randomly sampled and the stylized scene is rendered under it with a Lambertian, that is difuse-only, shading model,

$$
\hat { I } _ { k } = R ( \mathcal { G } ^ { \star } , P _ { k } , L ^ { ( t ) } ) ,\tag{9}
$$

where $R ( \cdot )$ is a renderer that applies the sampled lighting. Section 4.4 compares the two strategies.

## 3.2.3 Style Loss

Following ARF [25], we use the nearest-neighbour feature matching (NNFM) loss as the style objective. Given a rendered image $\hat { I } _ { k }$ and the reference style image $^ { J , }$ the style loss is $\mathcal { L } _ { \mathrm { n n f m } } ( \hat { I } _ { k } , J )$ . It extracts feature maps from a pre-trained VGG-16 network for both images, and for each feature vector of $\hat { I } _ { k }$ it finds the most similar feature vector of J by cosine similarity, then penalizes the cosine distance to that match. Following ARF, the features are taken from the conv3 block of the VGG-16 network.

To control the trade-of between content preservation and stylization strength, RxGS optionally incorporates a content loss. The stylization objective is

$$
\mathcal { L } _ { \mathrm { s t y l e } } = \mathcal { L } _ { \mathrm { n n f m } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c o n t e n t } } ,\tag{10}
$$

where $\lambda _ { c }$ weights the content term. Unless stated otherwise we set $\lambda _ { c } = 0$ , so stylization is driven by the NNFM objective alone.

## 3.2.4 Spatial Smoothness Prior on the Gaussian Neighbourhood Graph

To make the stylized albedo vary smoothly over the scene, we add a smoothness term on the optimized albedo field. The albedo is stored on the Gaussian primitives rather than on a regular image grid, so the smoothness prior is defined on a k-nearestneighbour graph built from the Gaussian centres. For a Gaussian $G _ { i } .$ , its neighbourhood $\mathcal { N } _ { k } ( i )$ is the set of the k Gaussians whose centres have the smallest Euclidean distance to $\mathbf { x } _ { i } .$ The spatial gradient of the stylized albedo field at $G _ { i }$ is the average, over the neighbourhood, of the albedo diference weighted by the inverse Euclidean distance between Gaussian centres,

$$
\nabla _ { \mathrm { s p a t i a l } } \mathbf { a } _ { i } ^ { \star } = \frac { 1 } { | \mathcal { N } _ { k } ( i ) | } \sum _ { j \in \mathcal { N } _ { k } ( i ) } \frac { \mathbf { a } _ { i } ^ { \star } - \mathbf { a } _ { j } ^ { \star } } { \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } + \varepsilon } ,\tag{11}
$$

where $\varepsilon = 1 0 ^ { - 6 }$ avoids division by zero and $k = 1 6$ in all experiments. The smoothness loss sums the $L _ { 1 }$ norm of this gradient over all Gaussians,

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { i } \lVert \nabla _ { \mathrm { s p a t i a l } } \mathbf { a } _ { i } ^ { \star } \rVert _ { 1 } .\tag{12}
$$

The final objective combines the style and smoothness terms,

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s t y l e } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s m o o t h } } ,\tag{13}
$$

where $\lambda _ { s } = 1 0 ^ { - 3 }$

## 4 Experiments

We evaluate RxGS from two angles: cross-view geometric consistency and relightable stylization. We first compare RxGS with existing reference-based 3D scene stylization methods using a geometric consistency metric, which measures how well scene structure is preserved across viewpoints after stylization. Because the existing methods do not support relighting, we then evaluate the relightable stylization of RxGS by rendering stylized scenes under diferent illumination conditions and examining how the appearance responds.

## 4.1 Experimental Setup

## Datasets.

We use NeRF-Synthetic [1] and NeRF-LLFF [41] for the geometric consistency comparison, and add TensorIR [14] and Mip-NeRF 360 [42] for the relightable stylization experiments. NeRF-Synthetic [1] is a synthetic multi-view dataset of eight objectcentric scenes, each with 100 training images and 200 test images rendered at 800×800, with camera poses and foreground masks. NeRF-LLFF [41] is a real-world forwardfacing multi-view dataset captured by handheld cameras with calibrated poses; it contains more complex geometry, occlusions, and appearance variation than NeRF-Synthetic, and we downsample the input images by a factor of four, following prior work [5, 11, 31]. TensorIR [14] is a multi-view dataset for inverse rendering and relighting, with diverse materials and illumination and with ground-truth geometry, material, and lighting; we use its Lego and Hotdog scenes for relightable stylization. Mip-NeRF 360 [42] is a real-world multi-view dataset of unbounded indoor and outdoor scenes captured over 360-degree viewpoints, with larger extents and more complex structure than NeRF-Synthetic and LLFF; we use its Garden, Kitchen, and Bicycle scenes.

## Implementation Details.

We take the BRDF-aware Gaussian representation reconstructed by GS-IR [3] as the input to RxGS. Stylization optimizes the difuse albedo while keeping the estimated geometry, roughness, metallicity, surface normals, and environment lighting fixed. Following ARF [25], we use the NNFM loss with a pre-trained VGG-16 network and take features from the conv3 block. Each scene is optimized for 1000 iterations.

Table 1 Geometry Consistency Cost (GCC) for stylization of the NeRF-Synthetic dataset [1]. Lower is better. The best result in each column is highlighted in orange. StylizedGS [33] does not report results on NeRF-Synthetic and is therefore omitted from this comparison.
<table><tr><td>Method</td><td>Lego</td><td>Ship</td><td>Chair</td><td>Mic</td><td>Drums</td><td>Ficus</td><td>Hotdog</td><td>Materials</td></tr><tr><td>ARF [25]</td><td>0.0038</td><td>0.017</td><td>0.0061</td><td>0.0024</td><td>0.0028</td><td>0.0024</td><td>0.0043</td><td>0.0036</td></tr><tr><td>Ref-NPR [31]</td><td>0.0045</td><td>0.0085</td><td>0.0034</td><td>0.0015</td><td>0.0035</td><td>0.0015</td><td>0.0047</td><td>0.0042</td></tr><tr><td>RxGS</td><td>0.0017</td><td>0.0045</td><td>0.0028</td><td>0.0004</td><td>0.0017</td><td>0.0005</td><td>0.0029</td><td>0.0011</td></tr></table>

Table 2 Geometry Consistency Cost (GCC) for stylization of the LLFF dataset [41]. Lower is better. The best and second-best results are highlighted in orange and peach, respectively.
<table><tr><td>Method</td><td>Fern</td><td>Fortress</td><td>Horns</td><td>Leaves</td><td>Orchids</td><td>Room</td><td>Trex</td><td>Flower</td></tr><tr><td>ARF [25]</td><td>0.0084</td><td>0.0168</td><td>0.0129</td><td>0.0111</td><td>0.0083</td><td>0.0093</td><td>0.0117</td><td>0.0126</td></tr><tr><td>Ref-NPR [31]</td><td>0.0064</td><td>0.0111</td><td>0.0086</td><td>0.0071</td><td>0.0077</td><td>0.0071</td><td>0.0085</td><td>0.0082</td></tr><tr><td>StylizedGŠ [33]</td><td>0.0056</td><td>0.0054</td><td>0.0074</td><td>0.0052</td><td>0.0051</td><td>0.0068</td><td>0.0064</td><td>0.0067</td></tr><tr><td>RxGS</td><td>0.0078</td><td>0.0105</td><td>0.0085</td><td>0.0084</td><td>0.0084</td><td>0.0102</td><td>0.0080</td><td>0.0113</td></tr></table>

## Comparison Methods.

We compare RxGS with ARF [25], Ref-NPR [31], and StylizedGS [33]. Ref-NPR requires a reference image rather than a style image, so we generate its reference images with the AdaIN style transfer model [18] from the same style images the other methods use. StylizedGS is included only on the LLFF scenes, because the original paper does not report NeRF-Synthetic results.

## Evaluation Protocol.

To reduce bias toward a particular artistic appearance, we evaluate each scene with three diferent reference styles, and for the geometric consistency comparison we average the reported results over the three styles. For relighting, NeRF-Synthetic and LLFF scenes are rendered under several illumination conditions by rotating the estimated environment map while keeping all stylization parameters fixed. We additionally evaluate relightable stylization on TensorIR and Mip-NeRF 360 scenes to examine the behaviour of stylized scenes under varying illumination across diferent scene types.

## Metric.

To measure cross-view geometric consistency we use the Geometry Consistency Cost (GCC), a per-pixel optical-flow warping error used in earlier video and 3D stylization work [31, 43]. Given two stylized renderings $\hat { I } _ { k }$ and ˆI $\hat { I } _ { k - 1 }$ from nearby viewpoints, we estimate the optical flow from $\hat { I } _ { k - 1 }$ to $\hat { I } _ { k }$ with the pre-trained RAFT model [44], warp $\hat { I } _ { k - 1 }$ to the viewpoint of $\hat { I } _ { k }$ using this flow, and compute the mean per-pixel error between $\hat { I } _ { k }$ and the warped frame over the pixels that the warp marks as visible. Averaging this error over consecutive frame pairs of a scene gives its GCC; a lower GCC indicates higher cross-view geometric consistency.

![](images/00f0c455f428a8752cf6cff7af2170a06499b6264db16eff937db7a33674f924.jpg)  
Fig. 3 Qualitative comparison of stylization on LLFF scenes. Each row shows a content scene and its style reference image (left), then the results of ARF, StylizedGS, Ref-NPR, and RxGS. For RxGS the stylized albedo field is shown by flat-albedo rendering. All methods receive the same content and style inputs.

![](images/0e51a76d386b65cd916ef3999904d53a973866843f053923b67fb19781fc1ec3.jpg)  
Fig. 4 Qualitative comparison of stylization on NeRF-Synthetic scenes. Each row shows a content scene and its style reference image (left), then the results of ARF, Ref-NPR, and RxGS. For RxGS the stylized albedo field is shown by flat-albedo rendering.

## 4.2 Cross-view Geometry Consistency

## 4.2.1 Quantitative Comparison

Using the GCC metric above, we evaluate the cross-view geometric consistency of the scenes stylized by RxGS and compare with ARF, Ref-NPR, and StylizedGS. Table 1 and Table 2 report the GCC values on NeRF-Synthetic and LLFF.

On NeRF-Synthetic, RxGS attains the lowest GCC in every scene. The margin over ARF and Ref-NPR is largest on Lego, Mic, Ficus, and Ship. On LLFF, RxGS has a lower GCC than ARF and is comparable with Ref-NPR and StylizedGS on most

![](images/ec2442c84484108496aee8aecaec662504058677ab56aeeebcbc75beca1d2e8b.jpg)  
Fig. 5 Relightable stylization on LLFF scenes. The same stylized representation is rendered under diferent illumination. The first row is albedo rendering without illumination; the remaining rows are renderings under the original environment lighting and under the environment map rotated by 90 and 180 degrees. Changing the illumination produces diferent shading and shadows while the stylized appearance stays consistent across viewpoints.

scenes. Despite the greater complexity of the real-captured scenes, RxGS stays on par with these methods while also keeping a relightable scene representation, which they do not.

## 4.2.2 Qualitative Comparison

Fig. 3 and Fig. 4 show qualitative stylization results on NeRF-Synthetic and LLFF. On both datasets, RxGS transfers the colour distribution and texture pattern of the reference style image while keeping the scene structure; these results hold on synthetic scenes (Fig. 4) and on real-captured scenes (Fig. 3) alike. The RxGS results are rendered from the optimized albedo field without illumination, which lets the transferred appearance be examined independently of scene lighting. Relightable renderings under varying illumination follow in the next section.

![](images/19234f05415da5699a878d958edcf026dad45bd4abf4744cfc21d355a6068245.jpg)  
Fig. 6 Relightable stylization on Mip-NeRF 360 under diferent environment lighting. For each scene the leftmost columns show the style image, the content image, and the stylized albedo rendering. The remaining columns render the same stylized scene under four environment maps (City, Fireplace, Forest, and Sunset). All results come from the same stylized representation without further optimization. The stylized scene responds to the illumination change, producing diferent lighting-dependent appearance while the stylized texture and the scene structure stay fixed.

## 4.3 Relightable Stylized Rendering

Fig. 5 shows relighting results on stylized LLFF scenes under diferent illumination. As the illumination changes, the rendered appearance shows the corresponding shading while keeping the stylized texture and colour distribution. In the Flower scene, rotating the dominant light source moves the shadows across the petals: under the 90-degree rotated lighting, shadows appear on the petals that face away from the incoming light. The Horns scene behaves similarly, with shadow locations shifting as the lighting direction changes; when the illumination rotates toward the right of the scene, shadows appear on the left of the horns, and under back-facing illumination larger shadowed regions appear in the foreground and the overall appearance is darker. The stylized scene therefore remains responsive to illumination changes after stylization, producing diferent shading under diferent lighting while preserving the stylized texture and scene structure.

Fig. 6 shows relighting results on stylized Mip-NeRF 360 scenes under four environment maps. The stylized appearance stays the same, but the four maps produce distinct illumination-dependent appearance. The City environment gives brighter renderings with stronger overall illumination, the Fireplace environment adds warmer colour tones, the Forest environment adds a green tint, and the Sunset environment gives a softer appearance with more moderate lighting transitions. These efects appear across the Bicycle, Garden, and Kitchen scenes: the stylized texture and structural detail stay fixed while the rendered appearance follows the illumination of each environment map. RxGS therefore keeps a relightable scene representation after stylization, so the stylized scene responds to diferent lighting environments without re-running the stylization.

## 4.4 Ablation Study

## 4.4.1 Efect of the Style Supervision Strategy

We compare two supervision strategies. The first renders the stylized albedo without illumination, so style supervision acts in the albedo domain. The second renders the stylized albedo under randomly sampled Lambertian illumination, which adds illumination-dependent shading during optimization.

The comparison uses eight NeRF-Synthetic scenes and two reference styles, giving 16 scene-style pairs. For one optimization run we define its loss reduction as $( \mathcal { L } _ { \mathrm { i n i t } } ~ - ~ \mathcal { L } _ { \mathrm { f i n a l } } ) / \mathcal { L } _ { \mathrm { i n i t } }$ , where ${ \mathcal { L } } _ { \mathrm { i n i t } }$ and ${ \mathcal { L } } _ { \mathrm { f i n a l } }$ are the stylization loss $\mathcal { L } _ { \mathrm { s t y l e } }$ at the first and the last iteration. Direct albedo supervision gives a lower final stylization loss than randomized-illumination supervision on all 16 pairs. Averaged over the 16 pairs, the loss reduction is 32.2% for direct albedo supervision and 20.1% for randomized-illumination supervision.

We attribute this gap to the extra appearance variation that randomized illumination introduces. Because the lighting direction changes during optimization, the rendered appearance contains illumination-dependent shading that is unrelated to the target style. The style supervision signal is then less stable and feature matching in the VGG space is harder. Direct albedo supervision removes the illumination contribution, so the optimization works only on the intrinsic appearance transfer. We therefore use direct albedo supervision in all experiments unless stated otherwise.

## 5 Conclusion

We presented RxGS, a relightable 3D scene stylization framework built on a BRDFaware Gaussian scene representation. RxGS optimizes only the difuse albedo field and leaves the roughness, metallicity, surface normals, and environment lighting fixed, so artistic appearance transfer does not alter the material or illumination parameters and the stylized scene keeps a representation that can be relit. RxGS preserves cross-view geometric consistency and produces illumination-dependent appearance under novel lighting.

## 6 Statements and Declarations

## 6.1 Competing Interests

The authors declare that they have no competing interests.

## References

[1] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021) https://doi.org/10.1145/ 3503250

[2] Kerbl, B., Kopanas, G., Leimk¨uhler, T., Drettakis, G.: 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4), 1–14 (2023) https://doi.org/10.1145/3592433

[3] Liang, Z., Zhang, Q., Feng, Y., Shan, Y., Jia, K.: GS-IR: 3d gaussian splatting for inverse rendering. In: Proc. CVPR, pp. 21644–21653. IEEE, New York, USA (2024). https://doi.org/10.1109/CVPR52733.2024.02045

[4] Sun, C., Sun, M., Chen, H.: Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In: Proc. CVPR, pp. 5449–5459. IEEE, New York, USA (2022). https://doi.org/10.1109/CVPR52688.2022.00538

[5] Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: Proc. CVPR, pp. 5491–5500. IEEE, New York, USA (2022). https://doi.org/10.1109/CVPR52688.2022.00542

[6] Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: TensoRF: Tensorial radiance fields. In: Proc. ECCV, pp. 333–350. Springer, Berlin, Heidelberg (2022). https://doi. org/10.1007/978-3-031-19824-3 20

[7] M¨uller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics 41(4), 1–15 (2022) https://doi.org/10.1145/3528223.3530127

[8] Yariv, L., Gu, J., Kasten, Y., Lipman, Y.: Volume rendering of neural implicit surfaces. In: Proc. NeurIPS, vol. 34, pp. 4805–4815. Curran Associates Inc., Red Hook, NY, USA (2021). https://doi.org/10.5555/3540261.3540628

[9] Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In: Proc. NeurIPS, pp. 27171–27183. Curran Associates Inc., Red Hook, NY, USA (2021). https://doi.org/10.5555/3540261.3542342

[10] Yang, B., Bao, C., Zeng, J., Bao, H., Zhang, Y., Cui, Z., Zhang, G.: NeuMesh: Learning disentangled neural mesh-based implicit field for geometry and texture editing. In: Proc. ECCV, pp. 597–614. Springer, Berlin, Heidelberg (2022). https: //doi.org/10.1007/978-3-031-19787-1 34

[11] Tang, J., Zhou, H., Chen, X., Hu, T., Ding, E., Wang, J., Zeng, G.: Delicate textured mesh recovery from NeRF via adaptive surface refinement. In: Proc.

ICCV, pp. 17693–17703. IEEE, New York, USA (2023). https://doi.org/10.1109/ ICCV51070.2023.01626

[12] Zhang, X., Srinivasan, P.P., Deng, B., Debevec, P.E., Freeman, W.T., Barron, J.T.: Nerfactor: neural factorization of shape and reflectance under an unknown illumination. ACM Transactions on Graphics 40(6), 237–123718 (2021) https: //doi.org/10.1145/3478513.3480496

[13] Zhang, K., Luan, F., Wang, Q., Bala, K., Snavely, N.: Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In: Proc. CVPR, pp. 5453–5462. IEEE, New York, USA (2021). https://doi.org/10. 1109/CVPR46437.2021.00541

[14] Jin, H., Liu, I., Xu, P., Zhang, X., Han, S., Bi, S., Zhou, X., Xu, Z., Su, H.: Tensoir: Tensorial inverse rendering. In: Proc. CVPR, pp. 165–174. IEEE, New York, USA (2023). https://doi.org/10.1109/CVPR52729.2023.00024

[15] Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., Ma, Y.: Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In: Proc. CVPR, pp. 5322–5332. IEEE, New York, USA (2024). https://doi.org/10.1109/ CVPR52733.2024.00509

[16] Gao, J., Gu, C., Lin, Y., Li, Z., Zhu, H., Cao, X., Zhang, L., Yao, Y.: Relightable 3d gaussians: Realistic point cloud relighting with BRDF decomposition and ray tracing. In: Proc. ECCV, pp. 73–89. Springer, Cham, Switzerland (2024). https: //doi.org/10.1007/978-3-031-72995-9 5

[17] Gatys, L.A., Ecker, A.S., Bethge, M.: Image style transfer using convolutional neural networks. In: Proc. CVPR, pp. 2414–2423. IEEE, New York, USA (2016). https://doi.org/10.1109/CVPR.2016.265

[18] Huang, X., Belongie, S.J.: Arbitrary style transfer in real-time with adaptive instance normalization. In: Proc. ICCV, pp. 1510–1519. IEEE, New York, USA (2017). https://doi.org/10.1109/ICCV.2017.167

[19] Doyle, L., Mould, D.: Augmenting photographs with textures using the laplacian pyramid. The Visual Computer 35, 1489–1500 (2019) https://doi.org/10.1007/ s00371-018-1513-y

[20] Xu, L., Yuan, Q., Sun, Y., Gao, Q.: Image neural style transfer combining global and local optimization. The Visual Computer 40, 8397–8411 (2024) https://doi. org/10.1007/s00371-023-03244-8

[21] H¨ollein, L., Johnson, J., Nießner, M.: StyleMesh: Style transfer for indoor 3D scene reconstructions. In: Proc. CVPR, pp. 6198–6208. IEEE, New York, USA (2022). https://doi.org/10.1109/CVPR52688.2022.00610

[22] Cao, X., Wang, W., Nagao, K., Nakamura, R.: PSNet: A style transfer network for point cloud stylization on geometry and color. In: Proc. WACV, pp. 3337– 3345. IEEE, New York, USA (2020). https://doi.org/10.1109/WACV45572.2020. 9093513

[23] Liu, H.-T.D., Tao, M., Jacobson, A.: Paparazzi: surface editing by way of multiview image processing. ACM Transactions on Graphics 37(6), 1–11 (2018) https: //doi.org/10.1145/3272127.3275047

[24] Chiang, P.-Z., Tsai, M.-S., Tseng, H.-Y., Lai, W.-S., Chiu, W.-C.: Stylizing 3D scene via implicit representation and hypernetwork. In: Proc. WACV, pp. 1475– 1484. IEEE, New York, USA (2022). https://doi.org/10.1109/WACV51458.2022. 00029

[25] Zhang, K., Kolkin, N., Bi, S., Luan, F., Xu, Z., Shechtman, E., Snavely, N.: ARF: Artistic radiance fields. In: Proc. ECCV, pp. 717–733. Springer, Berlin, Heidelberg (2022). https://doi.org/10.1007/978-3-031-19821-2 41

[26] Chen, Y., Yuan, Q., Li, Z., Liu, Y., Wang, W., Xie, C., Wen, X., Yu, Q.: UPST-NeRF: Universal photorealistic style transfer of neural radiance fields for 3D scene. IEEE Transactions on Visualization and Computer Graphics 31, 2045–2057 (2024) https://doi.org/10.1109/TVCG.2024.3378692

[27] Nguyen-Phuoc, T., Liu, F., Xiao, L.: SNeRF: stylized neural implicit representations for 3D scenes. ACM Transactions on Graphics 41(4), 1–11 (2022) https://doi.org/10.1145/3528223.3530107

[28] Goel, R., Dhawal, S., Saini, S., Narayanan, P.: StyleTRF: Stylizing tensorial radiance fields. In: Proc. ICVGIP, pp. 1–9. Association for Computing Machinery, New York, USA (2022). https://doi.org/10.1145/3571600.3571643

[29] Huang, Y., He, Y., Yuan, Y., Lai, Y., Gao, L.: StylizedNeRF: Consistent 3D scene stylization as stylized nerf via 2D-3D mutual learning. In: Proc. CVPR, pp. 18342–18352. IEEE, New York, USA (2022). https://doi.org/10.1109/ CVPR52688.2022.01780

[30] Li, W., Wu, T., Zhong, F., Oztireli, C.: Arf-plus: Controlling perceptual factors in<sup>¨</sup> artistic radiance fields for 3d scene stylization. In: Proc. WACV, pp. 2301–2310. IEEE, New York, USA (2025). https://doi.org/10.1109/WACV61041.2025.00230

[31] Zhang, Y., He, Z., Xing, J., Yao, X., Jia, J.: Ref-NPR: Reference-based non-photorealistic radiance fields for controllable scene stylization. In: Proc. CVPR, pp. 4242–4251. IEEE, New York, USA (2023). https://doi.org/10.1109/ CVPR52729.2023.00413

[32] Radl, L., Steiner, M., Kurz, A., Steinberger, M.: LAENeRF: Local appearance editing for neural radiance fields. In: Proc. CVPR, pp. 4969–4978. IEEE, New

[33] Zhang, D., Yuan, Y., Chen, Z., Zhang, F., He, Z., Shan, S., Gao, L.: Stylizedgs: Controllable stylization for 3d gaussian splatting. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(12), 11961–11973 (2025) https://doi.org/ 10.1109/TPAMI.2025.3604010

[34] Mei, Y., Xu, J., Patel, V.M.: Regs: Reference-based controllable scene stylization with gaussian splatting. In: Proc. NeurIPS. Curran Associates Inc., Red Hook, NY, USA (2024). https://doi.org/10.52202/079017-0132

[35] Liu, W., Liu, Z., Yang, X., Sha, M., Li, Y.: ABC-GS: alignment-based controllable style transfer for 3d gaussian splatting. In: Proc. ICME, pp. 1–6. IEEE, New York, USA (2025). https://doi.org/10.48550/arXiv.2503.22218

[36] Liu, W., Liu, Z., Shu, J., Wang, C., Li, Y.: GT2-GS: geometry-aware texture transfer for gaussian splatting. In: Proc. AAAI, pp. 7296–7304. AAAI Press, Palo Alto, CA, USA (2026). https://doi.org/10.1609/AAAI.V40I9.37667

[37] Yang, Y., Wang, Y., Wang, C., Wang, H., He, S.: Fantasystyle: Controllable stylized distillation for 3D gaussian splatting. In: Proc. AAAI, pp. 11784–11792. AAAI Press, Palo Alto, CA, USA (2026). https://doi.org/10.1609/AAAI.V40I14. 38164

[38] Liu, H., Huang, J., Lu, M., Saripalli, S., Jiang, P.: Stylos: Multi-view 3d stylization with single-forward gaussian splatting. In: Proc. ICLR (2026)

[39] Sablon, R.D., Hart, D.: Optimization-free style transfer for 3d gaussian splats. In: Proc. WACV, pp. 8041–8051. IEEE, New York, USA (2026). https://doi.org/ 10.1109/WACV61042.2026.00776

[40] Chen, Y., Chen, R., Lei, J., Zhang, Y., Jia, K.: TANGO: text-driven photorealistic and robust 3d stylization via lighting decomposition. In: Proc. NeurIPS. Curran Associates Inc., Red Hook, NY, USA (2022). https://doi.org/10.52202/ 068431-2242

[41] Mildenhall, B., Srinivasan, P.P., Ortiz-Cayon, R., Kalantari, N.K., Ramamoorthi, R., Ng, R., Kar, A.: Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics 38(4), 1–14 (2019) https://doi.org/10.1145/3306346.3322980

[42] Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-NeRF 360: Unbounded anti-aliased neural radiance fields. In: Proc. CVPR, pp. 5460– 5469. IEEE, New York, USA (2022). https://doi.org/10.1109/CVPR52688.2022. 00539

[43] Ruder, M., Dosovitskiy, A., Brox, T.: Artistic style transfer for videos and spherical images. International Journal of Computer Vision 126, 1199–1219 (2018) https://doi.org/10.1007/s11263-018-1089-z

[44] Teed, Z., Deng, J.: RAFT: Recurrent all-pairs field transforms for optical flow. In: Proc. ECCV, pp. 402–419. Springer, Berlin, Heidelberg (2020). https://doi. org/10.48550/arXiv.2003.12039