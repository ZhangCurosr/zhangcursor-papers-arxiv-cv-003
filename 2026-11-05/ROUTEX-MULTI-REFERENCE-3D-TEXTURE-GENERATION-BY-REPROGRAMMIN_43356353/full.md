# ROUTEX: MULTI-REFERENCE 3D TEXTURE GENERATION BY REPROGRAMMING ROTARY POSITION ENCODING

Chengyang Li, Yang Liu, Jiale Zhang, Qiang Rao, Wei Zhang, Taicheng Huang

JD.com, Beijing, China

## ABSTRACT

While single-image 3D texture generation has advanced significantly, conditioning texture synthesis on multi-view reference images remains largely under-explored, despite its necessity for resolving surface ambiguity in unobserved regions. In this paper, we propose RouTex, a novel multi-reference 3D texture generation framework based on in-context Diffusion Transformers (DiT). Our key insight is that native 3D Rotary Position Encoding (RoPE) inherently possesses spatial addressing capabilities. We introduce Positional Routing (PR) to reprogram 3D RoPE along spatial and semantic dimensions, establishing explicit reference-to-view correspondence without introducing additional parameters. To execute this routing efficiently, Dual-Granularity Attention (DGA) combines per-view local attention for fine texture fidelity with global joint attention over deduplicated references for crossview consistency. Extensive experiments demonstrate that RouTex achieves new state-of-the-art performance in multireference 3D texture generation.

Index Terms— Multi-Reference Texture Generation, 3D Modeling

## 1. INTRODUCTION

High-quality 3D assets with rich surface textures are essential for e-commerce, gaming, and augmented reality, typically acquired via reconstruction or generation. While reconstruction methods [1, 2, 3] require dense, precisely calibrated multi-view captures, generative methods [4, 5, 6] synthesize 3D content from a text prompt or a single image. However, such inputs of generation methods provide insufficient guidance for texture synthesis: text lacks fine-grained appearance control, while a single image leaves most surface areas unobserved. Worse, because the mapping from visible to occluded regions is one-to-many (e.g., a mug with a bird on the front may have a plain or a floral back), synthesized back-view textures must rely entirely on learned priors, often producing hallucinated or inaccurate details. Conditioning texture generation on multiple reference images is therefore crucial, as each additional view provides explicit appearance supervision for different surfaces of the object.

A straightforward approach to multi-reference conditioning is the cyclic injection scheme from TRELLIS [6], which applies one reference per denoising step. However, due to weak 3D priors and limited texture representation capabilities, TRELLIS recovers only coarse base textures, leading to severe distortion on fine details like text and logos. We thus turn to the prevailing Multi-View Diffusion (MVD) framework [7, 8, 9]. Directly transplanting cyclic injection into a MVD framework fails, as it lacks 3D structural priors to autoalign mesh surfaces with reference images via cross-attention. Consequently, the central challenge becomes: how to establish explicit correspondences between reference images and target views within a pure 2D framework.

To address this challenge, our key observation is that modern in-context Diffusion Transformers (DiT) [10, 11], such as Flux-Kontext [12], inherently possess spatial addressing capabilities via 3D Rotary Position Encoding (RoPE), where axis-0 of the position triplet distinguishes different images, while the two spatial axes encode 2D layout. We therefore propose to reprogram this native mechanism, without adding parameters, to inject explicit reference-to-view correspondence into generation.

Concretely, we introduce two complementary strategies to realize this design. First, Positional Routing (PR) partitions camera yaws into canonical azimuth regions to group target views and reference images, and reprograms 3D RoPE to align each reference’s spatial coordinates with its corresponding target view, enabling self-attention to automatically route appearance guidance (Sec. 2.2). Second, Dual-Granularity Attention (DGA) leverages the native double-stream and single-stream blocks of Flux-Kontext to execute this routing efficiently, combining per-view local attention for fine texture fidelity with global joint attention over deduplicated references for cross-view consistency (Sec. 2.3). Crucially, because correspondence is governed purely by position encodings across discrete regions, RouTex requires only coarse azimuth categories rather than precise camera calibration, substantially relaxing capture requirements while achieving state-of-the-art performance.

## 2. METHOD

## 2.1. Preliminaries

Flux-Kontext is an in-context image-editing Diffusion Transformer (DiT) that concatenates reference and target tokens into a unified sequence, processed by $N _ { d }$ double-stream blocks followed by $N _ { s }$ single-stream blocks. It encodes positions with 3D Rotary Position Encoding (RoPE) using triplets $( d , y , x )$ , where the semantic axis d distinguishes different input images and the spatial axes $( y , x )$ specify the 2D layout.

![](images/a3fcc40da1ca19f5d9c163880392d9476a67ad4e20f421f4fccbc53e451767bc.jpg)  
Fig. 1. Overview of RouTex. Given multi reference images from different azimuth regions and geometry conditions, Positional Routing assigns fractional semantic codes and aligns each reference to its target view coordinates. Double-stream blocks perform per-view local attention, then single-stream blocks perform global joint attention with deduplicated references.

Problem Setup. Following multi-view diffusion protocols [5, 7], we arrange six orthographic target views into a $3 \times 2$ grid, as illustrated in Fig. 1. Given a 3D mesh, text prompt t, multi-view geometry conditions, and K reference images ref $\dot { \mathbf { \eta } } = \left\{ \mathbf { r e f } _ { k } \right\} _ { k = 0 } ^ { K - 1 } \left( 1 \leq \overline { { K } } \leq 4 \right)$ , RouTex aims to generate high-fidelity textures consistent across all views. For each target view v, we denote its latent representation and geometry condition as latent and $\mathrm { g e o } _ { v }$ respectively.

## 2.2. Positional Routing

To establish explicit reference-to-view correspondence within a pure 2D framework, we propose Positional Routing (PR), which reprograms the native 3D RoPE triplets $( d , y , x )$ along both spatial and semantic dimensions without adding parameters or modifying the architecture.

Coordinate-Space Routing. We divide the 360<sup>◦</sup> camera yaw into four canonical azimuth regions $r \in \{ 0 , 1 , 2 , 3 \}$ (front, right, back, and left). Each reference and target view is assigned to a region according to its camera yaw, except the two polar views (top and bottom with pitch $\pm 9 0 ^ { \circ } )$ , which are assigned to region 0 by default.

Let a single VAE latent have spatial size $h \times w$ . For a token with local coordinate $( y , x )$ in the view at row i and column j of the $3 \times 2$ canvas, its global RoPE coordinate is

$$
y ^ { \prime } = y + i h , \qquad x ^ { \prime } = x + j w ,\tag{1}
$$

where $i \in \{ 0 , 1 , 2 \}$ and $j \in \{ 0 , 1 \}$

For a reference in region $r \neq 0$ , we align its spatial coordinates with those of the first target view $v ^ { * }$ in the same region:

$$
( y _ { \mathrm { r e f } _ { r } } ^ { \prime } , x _ { \mathrm { r e f } _ { r } } ^ { \prime } ) = ( y _ { v ^ { * } } ^ { \prime } , x _ { v ^ { * } } ^ { \prime } ) .\tag{2}
$$

The region-0 reference is instead positioned at the center of the $3 \times 2$ canvas to broadcast its front-view appearance to all target views, as shown in Fig. 1. Thus, relevant reference

and target tokens share aligned spatial positions in the RoPE space, enabling parameter-free appearance routing.

Fractional Semantic Addressing. To further distinguish token types, views, and reference regions, we reprogram the semantic axis d as follows:

$$
d = \left\{ \begin{array} { l l } { { 1 + v / V , } } & { { \mathrm { f o r ~ t a r g e t ~ l a t e n t ~ o f ~ v i e w } v , } } \\ { { 2 + v / V , } } & { { \mathrm { f o r ~ g e o m e t r y ~ c o n d i t i o n ~ o f ~ v i e w } v , } } \\ { { 3 + r / R , } } & { { \mathrm { f o r ~ r e f e r e n c e ~ i m a g e ~ i n ~ r e g i o n } r , } } \end{array} \right.\tag{3}
$$

where $V = 3 0$ is the candidate view pool size and $R = 4$ is the number of azimuth regions. Six target views are sampled per iteration while retaining their intrinsic indices $v \in$ $\{ 0 , \ldots , V - 1 \}$ , allowing the fractional offsets $v / V$ and $r / R$ to distinguish views and reference regions.

## 2.3. Dual-Granularity Attention

Positional Routing specifies which reference is associated with each target view. However, six target views outnumber the available references $( K \ \leq \ 4 )$ and directly duplicating references is computationally inefficient, we therefore introduce Dual-Granularity Attention (DGA) to exploit the native two-stage architecture of Flux-Kontext for local texture transfer and global cross-view consistency.

Double-Stream Blocks (Per-View Local Attention). During the double-stream stage, we isolate each target view into an independent sequence: [latent $\mathbf { \Sigma } _ { v } ; \mathbf { g e o } _ { v } ; \mathbf { r e f } _ { m ( v ) } ]$ , where $m ( v )$ denotes the matched reference for view v. By reshaping the batch dimension from B to $B \times 6 ,$ attention is restricted to each view-reference pair, reducing sequence length and enabling efficient fine-grained texture transfer.

Single-Stream Blocks (Global Joint Attention). Before the single-stream stage, reference features from different view-reference pairs are merged into K unique representations ${ \bf h } _ { 0 : K - 1 }$ via scatter-averaging. The input sequence is then reconstructed as [t; latents; geos; $\mathbf { h } _ { 0 : K - 1 } ]$ , allowing joint attention across all target views and unique references. This global interaction promotes cross-view appearance consistency while avoiding redundant reference computation.

Together, the double-stream blocks preserve local texture details at lower compute cost, while the single-stream blocks enforce global cross-view coherence to get high-fidelity texture.

## 2.4. Training

We train the entire model using the vanilla flow-matching [13] loss. The number of reference images K is randomly sampled from {1, 2, 3, 4}, strictly requiring at least one reference from region 0. Our view pool consists of 30 candidate viewpoints, comprising six canonical orthographic views (front, back, left, right, top, and bottom) and 24 surrounding views. At each iteration, six target views are randomly sampled from this pool while ensuring full coverage of all four azimuth regions. For geometry conditioning, the surface normal map and canonical coordinate map (CCM) are averaged into a single composite condition map per view.

## 3. EXPERIMENTS

## 3.1. Implementation Details

Our training dataset comprises approximately 200K 3D assets curated from public datasets (Objaverse [14], ABO [15], 3D-FUTURE [16], TexVerse [17]) alongside proprietary highquality handcrafted models. For each asset, we render 24 reference views at 2048×2048 resolution following the camera setup of TRELLIS [6], and 30 orthographic target views at 1024×1024 as ground truth. We train RouTex on 8 NVIDIA B200 GPUs with a batch size of 64 for 5.5K iterations (∼32 hours) at 512×512 resolution. During inference, we scale the resolution to 768×768, generating six target views in approximately 40 seconds over 15 denoising steps without classifierfree guidance. FlashAttention V4 [18] is employed across both training and inference to optimize runtime efficiency.

## 3.2. Experimental Settings

Baselines and Metrics. We compare RouTex against five state-of-the-art baselines: two multi-reference methods (TREL LIS [6] and TRELLIS.2 [19]) and three single-reference methods (Hunyuan3D-2.1 [20], MV-Adapter [7], and Lumi-Tex [5]). Specifically, multi-image input is natively supported in the official codebase of TRELLIS, whereas for TREL-LIS.2, we adapt it using the same cyclic injection mechanism as TRELLIS for fair evaluation. Besides, all other baselines run under their default single-view setups. To isolate texture generation quality from geometric variations, all methods texture the same meshes reconstructed by Hunyuan3D-2.1 and decimated to 50K faces. Outputs are evaluated by baking synthesized textures back onto these shared meshes, measuring quality via FID [21], KID [22], CLIP-I [23], and LPIPS [24].

Evaluation Dataset. We curate an independent evaluation set of 475 high-quality 3D assets, disjoint from the training data. For each asset, we render 12 evaluation views at a $1 5 ^ { \circ }$ elevation angle, uniformly sampled every $3 0 ^ { \circ }$ in azimuth starting from $0 ^ { \circ }$ . To eliminate illumination bias arising from differences between PBR-based and shaded texture pipelines among baselines, all evaluation renderings are performed under uniform ambient lighting. RouTex and other multi-reference baselines take four canonical orthogonal views (front, back, left, right) as condition images, whereas single-reference baselines use the front view.

<table><tr><td>Method</td><td>FID↓</td><td>KID↓</td><td>CLIP-I↑</td><td>LPIPS↓</td></tr><tr><td>TRELLIS</td><td>18.83</td><td>28.49</td><td>0.9149</td><td>0.1727</td></tr><tr><td>TRELLIS.2</td><td>12.93</td><td>4.53</td><td>0.9246</td><td>0.1656</td></tr><tr><td>Hunyuan3D-2.1</td><td>15.52</td><td>10.34</td><td>0.9242</td><td>0.1703</td></tr><tr><td>MV-Adapter</td><td>15.34</td><td>12.04</td><td>0.9217</td><td>0.1710</td></tr><tr><td>LumiTex</td><td>17.64</td><td>23.82</td><td>0.9258</td><td>0.1666</td></tr><tr><td>Ours</td><td>12.06</td><td>8.66</td><td>0.9423</td><td>0.1519</td></tr></table>

Table 1. Quantitative comparison with state-of-the-art methods. Red and blue highlight the best and second-best results, respectively.

## 3.3. Quantitative Results

As shown in Table 1, RouTex outperforms all baselines. Single-reference methods see only the front view, failing to hypothesize plausible textures for unobserved surfaces. Multi-reference methods suffer from weak 3D priors and limited texture representation capacity, while their iterative injection scheme lacks explicit reference-to-view correspondence, resulting in severe texture distortion. In contrast, RouTex explicitly routes multi-reference texture within a strong generative framework, producing high-quality textures that remain plausible and consistent across all views.

## 3.4. Qualitative Results

Open-Source Methods. Fig. 2 compares RouTex with opensource state-of-the-art baselines using 2 to 4 reference images, where RouTex faithfully reproduces complex textures across all input views. Single-reference methods receive no visual guidance for unobserved regions, often generating flat, uninformative surfaces. Even within observed areas, they fail to reconstruct fine-grained appearance details. While multireference methods capture basic color distributions, their limited 3D texture representations and iterative injection scheme lack explicit reference-to-view correspondence, leaving fine patterns and text noticeably blurred or garbled.

Commercial Methods. We further compare against two commercial models that support multi-image 3D generation. To reduce the resolution gap for a fairer comparison, we retrain our model at 1024×1024 resolution. As shown in Fig. 3, Hunyuan3D v3.1 produces chaotic textures with severe crossview inconsistencies. Tripo3D v3.1 generates cleaner textures overall but struggles with spatial precision and fine details. As highlighted in the zoomed-in insets, the painting in the orange box (column 4, row 2) is misaligned relative to the reference image, and the text in the red box is less crisp than ours.

## 3.5. Ablation Study

We validate the necessity of each key component through four ablation variants: (1) CSIC (Cyclic Single-Image Conditioning) adopts TRELLIS’s injection strategy with the single reference MVD framework; (2) w/o PR removes Positional Routing, assigning identical RoPE to all reference images in single-stream blocks while retaining per-view matching in double-stream blocks; (3) w/o DGA processes all target views and references jointly across both double-stream and singlestream stages while preserving Positional Routing; (4) w/o PR & DGA disables both components.

![](images/9c4eef7ef945e38231efb5df8ca303d13f1d73cb8bf52c090f6ac8c8105a7374.jpg)  
Input Images

![](images/47b0734cb0a4a78b67332bfafb80043c4925af362e522b2fc0f5c937bc025ed2.jpg)  
RouTex (Ours)

![](images/334ce885581ca6a19c642c22d4bbe17ec6bbc4e8be9cbdeb33d3550dc0b5bf06.jpg)  
TRELLIS

![](images/e87c231bd3999423282a3aaa69a5fc12792677444e3d57ff6e6e70203fd5f5f9.jpg)  
TRELLIS.2

![](images/49718d07c15254daa99ccb71dd66294d5e5d3cb8c7b53232a4979857be3f1826.jpg)  
Hunyuan3D 2.1

![](images/119decd1bc4f1196ba11eb88b6f92611ff46db9d3c1ad2bc9f2fb1a0934488e5.jpg)  
MV-Adapter

![](images/1cb3e8dcfceccba20cb9d6b82bcb157162354ea267875a5eccec3f4daceb7dd1.jpg)  
LumiTex

Fig. 2. Qualitative comparison with open-source methods. Orange and red boxes highlight zoomed-in regions for detail inspection. In the input column, red-bordered images indicate the input of single-image methods.  
![](images/1b8ba395887a8eccfe40032ccc5ecea34a4ee3e3d7adf9b67b502ecdae6611dc.jpg)  
Input Images  
RouTex (Ours)  
Hunyuan3D v3.1  
Tripo3D v3.1

As reported in Table 2 and Fig. 4, removing any core component or replacing our multi-reference injection with TREL-LIS’s strategy leads to a performance degradation. Furthermore, although removing DGA leads to only a slight quantitative drop, it remains essential for preventing localized texture artifacts and enabling a ∼17% inference speedup.

Fig. 3. Qualitative comparison with commercial methods. Orange and red boxes highlight zoomed-in regions for detail inspection.
<table><tr><td>Method</td><td>FID↓</td><td>KID↓</td><td>CLIP-I↑ LPIPS↓</td></tr><tr><td>CSIC</td><td>20.08</td><td>27.49</td><td>0.9122 0.1700</td></tr><tr><td>w/o PR &amp; DGA</td><td>14.61</td><td>13.13</td><td>0.9280 0.1619</td></tr><tr><td>w/o PR</td><td>12.49</td><td>9.63</td><td>0.9395 0.1541</td></tr><tr><td>w/o DGA</td><td>12.11</td><td>8.36</td><td>0.9413 0.1523</td></tr><tr><td>Full Model</td><td>12.06</td><td>8.66</td><td>0.9423 0.1519</td></tr></table>

Table 2. Quantitative results of ablation study. Red and blue highlight the best and second-best results, respectively.

![](images/4f8d289eba3dfa26c9b630bc4e699090b0c290728a378961913a0e0f3606aff8.jpg)  
Fig. 4. Qualitative ablation results. Green and red boxes highlight successful recoveries by the Full Model and failures of the ablated variants, respectively.

## 4. CONCLUSION

In this work, we present a multi-reference framework for high-quality 3D texture generation within a 2D multi-view diffusion framework. We introduce Positional Routing (PR) to establish explicit reference-to-view correspondence through reprogrammed 3D RoPE, and Dual-Granularity Attention (DGA) to efficiently combine local texture transfer with global cross-view consistency. With these components, RouTex achieves state-of-the-art performance in multi-reference 3D texture generation.

## 5. REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis,¨ “3D Gaussian Splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, pp. 139:1–139:14, 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, et al., “NeRF: Representing scenes as neural radiance fields for view synthesis,” in European Conference on Computer Vision (ECCV), 2020, pp. 405–421.

[3] L. Pan, D. Barath, M. Pollefeys, and J. L. Schonberger,¨ “Global structure-from-motion revisited,” in European Conference on Computer Vision (ECCV), 2024, pp. 54– 73.

[4] Z. Lai, Y. Zhao, Z. Zhao, et al., “NaTex: Seamless texture generation as latent color diffusion,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 18619–18629.

[5] J. Bao, H. Chen, L. Zhu, et al., “LumiTex: Towards high-fidelity PBR texture generation with illumination context,” in International Conference on Learning Representations (ICLR), 2026.

[6] J. Xiang, Z. Lv, S. Xu, et al., “TRELLIS: Structured 3D latents for scalable and versatile 3D generation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 21469–21480.

[7] Z. Huang, Y.-C. Guo, H. Wang, et al., “MV-Adapter: Multi-view consistent image generation made easy,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 16377–16387.

[8] ByteDance Seed, “Seed3D 1.0: From images to highfidelity simulation-ready 3D assets,” arXiv preprint arXiv:2501.00000, 2025.

[9] Y. Shi, P. Wang, J. Ye, et al., “MVDream: Multiview diffusion for 3D generation,” arXiv preprint arXiv:2308.16512, 2023.

[10] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 6840–6851, 2020.

[11] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4195–4205.

[12] Black Forest Labs, S. Batifol, A. Blattmann, et al., “FLUX.1 Kontext: Flow matching for in-context image generation and editing in latent space,” arXiv preprint arXiv:2506.15742, 2025.

[13] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, et al., “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[14] M. Deitke, D. Schwenk, J. Salvador, et al., “Objaverse: A universe of annotated 3D objects,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13142–13153.

[15] J. Collins, S. Goel, K. Deng, et al., “ABO: Dataset and benchmarks for real-world 3D object understanding,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 21094–21104.

[16] H. Fu, R. Jia, L. Gao, et al., “3D-FUTURE: 3D furniture shape with texture,” International Journal ofComputer Vision, vol. 129, no. 12, pp. 3313–3337, 2021.

[17] Y. Zhang, L. Zhang, R. Ma, and N. Cao, “TexVerse: A universe of 3D objects with high-resolution textures,” arXiv preprint arXiv:2508.10868, 2025.

[18] T. Zadouri, M. Hoehnerbach, J. Shah, et al., “FlashAttention-4: Algorithm and kernel pipelining codesign for asymmetric hardware scaling,” Proceedings of Machine Learning and Systems, vol. 8, pp. 912–926, 2026.

[19] J. Xiang, X. Chen, S. Xu, et al., “Native and compact structured latents for 3D generation,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 14419–14429.

[20] Tencent Hunyuan3D Team, “Hunyuan3D 2.1: From images to high-fidelity 3D assets with production-ready PBR material,” arXiv preprint arXiv:2506.15442, 2025.

[21] M. Heusel, H. Ramsauer, T. Unterthiner, et al., “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 6626–6637, 2017.

[22] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gret-´ ton, “Demystifying MMD GANs,” in International Conference on Learning Representations (ICLR), 2018.

[23] A. Radford, J. W. Kim, C. Hallacy, et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning (ICML), 2021, pp. 8748–8763.

[24] R. Zhang, P. Isola, A. A. Efros, et al., “The unreasonable effectiveness of deep features as a perceptual metric,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.