# Operator-Aware Posterior Sampling for Multivariate Weather Downscaling

Carlo Saccardi Rudolf Saathof Geethu Joseph Justin Dauwels

Delft University of Technology, The Netherlands {C.Saccardi, R.Saathof, G.Joseph, J.H.G.Dauwels}@tudelft.nl

## Abstract

Statistical downscaling is the task of reconstructing high-resolution meteorologi cal fields from coarse inputs. A principled approach formulates downscaling as a Bayesian inverse problem, using a diffusion prior over high-resolution fields and incorporating the coarse field only at inference through a likelihood. This decouples the prior from the coarse source, allowing one pretrained model to adapt to different sources. However, two challenges limit this flexibility: (1) when coarse and target fields come from different reanalysis systems, no fixed coarsening operator can capture their relationship; and (2) guidance gradients can differ by orders of magnitude across variables, making a single guidance strength ineffective. This paper addresses both challenges within the guidance loop, leaving the prior unchanged. We learn the forward operator from paired data and normalize the gradient per channel. We show that this normalization corresponds to steepest descent under a block-max norm, allowing a single guidance strength across variables. Our code is publicly available at https://github.com/CarloSaccardi/ operator-aware-weather-downscaling.

## 1 Introduction

Kilometer-scale meteorological fields are essential for energy system planning [Qiu et al., 2024], laser-satellite communications [Frehlich et al., 2010], and agriculture [Gutowski et al., 2020]. Global reanalyses such as ERA5 [Hersbach et al., 2020], however, have a resolution of only ∼25 km, leaving local terrain, coastlines, and other fine-scale features unresolved. Machine-learning weather models trained on these data inherit the same limitation [Bodnar et al., 2025, Price et al., 2025, Bi et al., 2023]. Dynamical downscaling recovers these fine scales by running a regional model on a nested grid [Salathé Jr et al., 2010], at a computational cost that scales rapidly with the domain size [Bader, 2009]. Statistical downscaling instead learns the coarse-to-fine mapping from paired data [Wilby et al., 1998], increasingly using diffusion models [Mardani et al., 2025, Merizzi et al., 2024].

A complementary line of work treats downscaling as an inverse problem: a diffusion model trained on high-resolution fields as a prior and incorporating the coarse field at inference through a likelihood term, without retraining the prior [Chung et al., 2023, Rozet and Louppe, 2023, Tie et al., 2026, Aich et al., 2026, Schmidt et al., 2025]. This separates the expensive component, the prior over high-resolution atmospheric states, allowing a single pretrained prior to be adapted to different sources without retraining. Existing inverse problem formulations, however, assume a forward model y = A(x<sub>0</sub>) + n with A known, obtained by synthetically coarsening the high-resolution data [Harder et al., 2023, Yang et al., 2024]. This assumption fails when the coarse and fine fields come from different systems. For example, two reanalyses, or a climate model and a reanalysis, introduce systematic, terrain-dependent differences that cannot be captured by a coarsening operator (Figure 1).

We address this limitation, together with a second challenge arising in multivariate guidance, entirely within the guidance loop of diffusion posterior sampling (DPS) [Chung et al., 2023]. First, we replace the assumed operator with a lightweight network $\mathbf { A } _ { \phi }$ that learns the coarse-to-fine relationship from data. Mapping high-resolution to low-resolution is far easier than the reverse, so this network is cheap to train, and conditioning it on orography lets it capture the terrain-dependent bias between the two systems. Second, jointly downscaled variables have guidance gradients that differ by orders of magnitude, so one guidance strength overcorrects the rough variables and barely moves the smooth ones. We normalize the gradient per channel, which we show is steepest descent on the DPS objective under a block-max norm. Both modifications leave the prior unchanged. Adapting a new source requires only a two-layer network and one validated scalar [Aich et al., 2026, Tie et al., 2026], which we verify by refitting only $\mathbf { A } _ { \phi }$ to a coarser source (App. E). On ERA5-to-CERRA, this trails conditional models trained end-to-end on a fixed coarse source, but does so without retraining the high-resolution prior. The framework therefore trades source-specific optimization for reuse of a single pretrained prior across heterogeneous coarse products.

![](images/9ad197f7840cebc752a245bf2688837ff962f88cd9da89c9894fd504c7c9030b.jpg)  
(a) ERA5 input - sshf

![](images/fe96db814da2bbce8dce4f73d7172af512e454e364ec247a1cd7cc5ccd491765.jpg)  
(b) CERRA target - sshf

![](images/f06101968c397ca0cc99fd8ce75a1fcb3c47d833851a86ca3cc3086219bf51ee.jpg)  
(c) ERA5 input - zust

![](images/3069bb8dff813ff0538fe5667597f8287bdf0fe54dc4a3980686704bde4e95af.jpg)  
(d) CERRA target - zust  
Figure 1: Comparison of surface sensible heat flux (sshf) and friction velocity (zust) between the coarse ERA5 and the high-resolution CERRA. Beyond the resolution gap, the fields differ in structure and large-scale pattern, implying an unknown and system-dependent coarse-to-fine map.

## 2 Related Work

Statistical downscaling learns a coarse-to-fine mapping from paired data, with deterministic [Harilal et al., 2022, Höhlein et al., 2020, Saccardi et al., 2025] and, more recently, diffusion-based [Fotiadis et al., 2025, Brenowitz et al., 2025, Mardani et al., 2025] models; much of this work synthetically coarsens the target to form its inputs, making the operator a known degradation [Harder et al., 2023, Yang et al., 2024]. A complementary line treats downscaling as an inverse problem, using a diffusion prior and incorporating the coarse field at inference through a likelihood [Tie et al., 2026, Aich et al., 2026, Schmidt et al., 2025]. This requires a forward operator, assumed known and typically a fixed downsampler, valid only when the coarse field is a degraded target. When the two come from different systems no such operator exists. Prior work debiases the coarse field and then trains a conditional model on the corrected pairs [Wan et al., 2023], which handles the same problem but tie the model to the coarse source it was trained on. We instead keep the inverse-problem setup and learn the operator inside the guidance loop. Finally, DPS scales the guidance gradient by the residual norm with one scalar per sample [Chung et al., 2023], which sets the step size but treats every variable alike; our per-channel normalization rescales each variable independently.

## 3 Background

An inverse problem recovers a high-resolution signal $\mathbf { x } _ { \mathrm { 0 } }$ from a measurement y related to it by a forward operator A, as $\mathbf { y } = \mathbf { A } ( \mathbf { x } _ { 0 } ) + \mathbf { n }$ with noise n $\sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ . When $\mathbf { y }$ is lower-dimensional than $\mathbf { x } _ { \mathrm { 0 } }$ the problem is ill-posed, so rather than a point estimate a Bayesian formulation targets the posterior $p ( \mathbf { x } _ { 0 } \mid \mathbf { y } ) \propto p ( \mathbf { y } \mid \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { 0 } )$ , where the likelihood $p ( \mathbf { y } \mid \mathbf { x } _ { 0 } )$ is fixed by the forward model and a diffusion model [Ho et al., 2020, Song et al., 2020] supplies the prior $p ( \mathbf { x } _ { 0 } )$ .

To sample the posterior, DPS [Chung et al., 2023] adds a guidance correction to each step of the diffusion sampler, given by the gradient of the likelihood evaluated at the denoised estimate $\hat { \mathbf { x } } _ { 0 }$ of $\mathbf { x } _ { \mathrm { 0 } }$ ,

$$
\zeta _ { i } \nabla _ { \mathbf { x } _ { t } } \big \| \mathbf { y } - \mathbf { A } ( \hat { \mathbf { x } } _ { 0 } ) \big \| _ { 2 } ^ { 2 } , \qquad \zeta _ { i } = \zeta / \big \| \mathbf { y } - \mathbf { A } ( \hat { \mathbf { x } } _ { 0 } ) \big \| _ { 2 } ,\tag{1}
$$

which steers the current estimate toward consistency with $\mathbf { y }$ . The operator A maps a high-resolution field to its coarse counterpart and thus determines how the coarse observation influences the sampling

process. The scalar $\zeta _ { i }$ normalizes the update by the residual norm, controlling its overall magnitude while applying one common scaling factor to all components of the gradient.

## 4 Method

In this section, we introduce our two contributions to the guidance loop: a learned forward operator that replaces the assumed downsampler with one fitted from data (Section 4.1), and a per-channel normalization of the guidance gradient that lets a single guidance strength serve all variables at once (Section 4.2). Algorithm 1 summarizes the full sampling procedure.

## 4.1 Learning the forward operator

The guidance correction in (1) requires a forward operator, and the standard choice is a fixed linear downsampler A. Writing the residual it produces at the denoised estimate $\hat { \mathbf { x } } _ { 0 }$ reveals why this is inadequate between different systems,

$$
\begin{array} { r } { { \bf y } - { \bf A } \hat { \bf x } _ { 0 } = \underbrace { { \bf y } - { \bf A } { \bf x } _ { 0 } } _ { \mathrm { i n t e r - s y s t e m b i a s } \delta } + \underbrace { { \bf A } ( { \bf x } _ { 0 } - \hat { \bf x } _ { 0 } ) } _ { \mathrm { r e c o n s t r u c t i o n ~ e r r o r } } . } \end{array}\tag{2}
$$

The reconstruction error vanishes as $\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } \to \mathbf { x } _ { 0 } , } \end{array}$ , but the bias $\pmb { \delta } = \mathbf { y } - \mathbf { A x } _ { 0 }$ is independent of $\hat { \mathbf { x } } _ { 0 }$ and is nonzero whenever $\mathbf { A x } _ { 0 } \neq \mathbf { y }$ , precisely the case when y and $\mathbf { x } _ { \mathrm { 0 } }$ come from different systems. The guidance gradient thus does not vanish at the target, steering the sample away from $\mathbf { x } _ { \mathrm { 0 } }$ toward a field whose coarsening matches $\mathbf { y } .$ We remove this bias by replacing A with a small network $\mathbf { A } _ { \phi }$ fitted directly to reproduce the coarse field,

$$
\phi ^ { \star } = \arg \operatorname* { m i n } _ { \phi } \mathbb { E } _ { \mathbf { x } _ { 0 } } \left\| \mathbf { y } - \mathbf { A } _ { \phi } ( \mathbf { x } _ { 0 } ) \right\| _ { 2 } ^ { 2 } ,\tag{3}
$$

which minimizes exactly the bias $\pmb { \delta }$ a fixed operator leaves behind. At inference $\phi$ is frozen.

## 4.2 Per-channel gradient normalization

The second component concerns how the guidance is scaled across variables. DPS was designed for natural images, whose channels are statistically alike, so one scale $\zeta$ suits all of them. At noise level σ it forms a single gradient from the residual pooled over all $C$ channels and adds it to the drift,

$$
\begin{array} { r } { \mathbf { g } = \nabla _ { \mathbf { x } _ { \sigma } } \bigl \| \mathbf { y } - \mathbf { A } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ) \bigr \| _ { 2 } , \qquad \mathbf { d } = \frac { \mathbf { x } _ { \sigma } - \hat { \mathbf { x } } _ { 0 } } { \sigma } + \zeta \mathbf { g } , \qquad \hat { \mathbf { x } } _ { 0 } = D _ { \theta } ( \mathbf { x } _ { \sigma } ; \sigma ) . } \end{array}\tag{4}
$$

Weather variables are not alike: a smooth field such as surface pressure and a rough, turbulent one such as friction velocity produce guidance gradients of very different magnitude. The per-channel gradient norm $\left\| \mathbf { g } _ { c } \right\| _ { F }$ scales with the roughness of variable $c ,$ through the denoiser Jacobian $\mathbf { J } = \partial \bar { \hat { \mathbf { x } } } _ { 0 } / \partial \mathbf { x } _ { \sigma }$ (App. B), and can differ by orders across the $C$ variables. This imbalance persists despite per-variable standardization, since it originates in the Jacobian rather than the data scale. Thus, a single $\zeta$ cannot fit all of them at once: it is too large for the chaotic variables, and too small for the smooth ones. The resulting update is then governed by whichever variables are loudest.

We remove this coupling by normalizing each channel to unit norm before scaling,

$$
\hat { \bf g } _ { c } = \frac { { \bf g } _ { c } } { \| { \bf g } _ { c } \| _ { F } + \varepsilon } , \qquad { \bf d } = \frac { { \bf x } _ { \sigma } - \hat { \bf x } _ { 0 } } { \sigma } + \zeta \hat { \bf g } ,\tag{5}
$$

reducing over the spatial dimensions only, so every variable contributes a unit-norm step and $\zeta$ moves each by the same amount regardless of its raw gradient magnitude. This is not an ad-hoc rescaling: it is exactly steepest descent under the block-max norm $\left\| \mathbf { \bar { v } } \right\| _ { \infty , F } = \operatorname* { m a x } _ { c } \left\| \mathbf { v } _ { c } \right\| _ { F } .$ , whose geometry treats each channel independently, so the optimal step normalizes each channel on its own, $\bar { \mathbf { v } } _ { c } ^ { \star } = - \bar { \mathbf { g } } _ { c } / \| \mathbf { g } _ { c } \| _ { F } ( \mathrm { A p p . ~ B } )$ .

## 5 Experiments

We downscale ERA5 [Hersbach et al., 2020] (∼25 km) to CERRA [Schimanke et al., 2021] $( \mathord { \sim } 5 . 5$ km) over two European domains, Central Europe and Iberia, using six surface variables: 10 m winds $( u _ { 1 0 } .$ $v _ { 1 0 } )$ , 2 m temperature $( t _ { 2 m } )$ , surface sensible heat flux (sshf), friction velocity (zust), and surface pressure (sp), with orography as a static channel. The prior $p _ { \theta } ( \mathbf { x } _ { 0 } \mid \mathbf { h } )$ and operator ${ \bf A } _ { \phi } \left( 3 \right)$ are trained separately and frozen at inference. We isolate each contribution with one ablation: the operator ablation compares the learned $\mathbf { A } _ { \phi }$ against the bicubic downsampler, and the normalization ablation compares our update (5) against DPS (1), tuning ζ separately per arm so neither reduces to a mistuned baseline. We report per-variable RMSE, CRPS, and power-spectral-density (PSD) plots. Full data, training, and baseline details are in Appendix C.

Table 1: Operator ablation. RMSE and CRPS per variable, both regions. Both rows use per-channel normalization and the same ζ; only the likelihood operator differs. The learned operator improves every variable. Best per column in bold.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Operator</td><td colspan="2"> $_ { u _ { 1 0 } }$ </td><td colspan="2"> $v _ { 1 0 }$ </td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td rowspan="2">C. Europe</td><td>Bicubic A</td><td>1.199</td><td>0.739</td><td>1.174</td><td>0.718</td><td>1.338</td><td>0.806</td><td>33.83</td><td>15.48</td><td>0.199</td><td>0.119</td><td>131.2</td><td>75.46</td></tr><tr><td>Ours</td><td>0.986</td><td>0.553</td><td>0.982</td><td>0.547</td><td>1.021</td><td>0.571</td><td>32.70</td><td>13.33</td><td>0.090</td><td>0.047</td><td>92.05</td><td>55.96</td></tr><tr><td rowspan="2">Iberia</td><td>Bicubic A</td><td>1.362</td><td>0.827</td><td>1.372</td><td>0.828</td><td>1.809</td><td>1.152</td><td>41.18</td><td>20.43</td><td>0.174</td><td>0.095</td><td>156.1</td><td>88.86</td></tr><tr><td>Ours</td><td>1.197</td><td>0.678</td><td>1.217</td><td>0.684</td><td>1.377</td><td>0.824</td><td>40.23</td><td>18.28</td><td>0.096</td><td>0.052</td><td>106.7</td><td>63.28</td></tr></table>

Table 2: Normalization ablation. RMSE and CRPS per variable, both regions. Both rows use the learned operator; only the guidance scaling differs, each arm at its own tuned ζ. A single per-sample scale (DPS) leaves the high-magnitude variable sp badly under-served (up to +67% RMSE), whereas ours keeps every variable controlled. Best per column in bold.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Guidance</td><td colspan="2">u10</td><td colspan="2">v10</td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td rowspan="2">C. Europe</td><td>DPS [Chung et al., 2023] Ours</td><td>0.969</td><td>0.539</td><td>0.966</td><td>0.535</td><td>1.092</td><td>0.608</td><td>31.27</td><td>12.79</td><td>0.096</td><td>0.049</td><td>153.7</td><td>93.17</td></tr><tr><td></td><td>0.986</td><td>0.553</td><td>0.982</td><td>0.547</td><td>1.021</td><td>0.571</td><td>32.70</td><td>13.33</td><td>0.090</td><td>0.047</td><td>92.05</td><td>55.96</td></tr><tr><td rowspan="2">Iberia</td><td>DPS [Chung et al., 2023]</td><td>1.172</td><td>0.661</td><td>1.186</td><td>0.662</td><td>1.413</td><td>0.842</td><td>38.22</td><td>17.28</td><td>0.096</td><td>0.051</td><td>161.6</td><td>91.88</td></tr><tr><td>Ours</td><td>1.197</td><td>0.678</td><td>1.217</td><td>0.684</td><td>1.377</td><td>0.824</td><td>40.23</td><td>18.28</td><td>0.096</td><td>0.052</td><td>106.7</td><td>63.28</td></tr></table>

Results. Tables 1 and 2 isolate the two contributions on both regions. The learned operator improves RMSE and CRPS on every variable over the bicubic operator, with the largest gains on the variables a fixed operator mishandles most (see Figure 6): on Central Europe, zust RMSE falls from 0.199 to 0.090 and sp from 131.2 to 92.05, and Iberia shows the same pattern. For the normalization ablation, per-channel normalization keeps every variable controlled under a single ζ, whereas the DPS scaling leaves the highest-magnitude variable, sp, badly under-served: its RMSE rises to 153.7 on Central Europe and 161.6 on Iberia, against 92.05 and 106.7 for ours. All reported numbers are point estimates whose 95% bootstrap intervals are given in Appendix D. The improvements we highlight, the learned operator on the rough variables and per-channel normalization on sp, exceed those intervals and are statistically significant, whereas the near-ties are within them.

## 6 Conclusion

We revisited diffusion posterior sampling for weather downscaling and addressed two limitations of its guidance loop that arise when the coarse and fine fields come from different systems. One, a fixed downsampler cannot express the bias between systems, so we learned the operator from data. Two, a single guidance strength cannot serve variables whose gradients differ by orders of magnitude, so we normalized the gradient per channel, which we show is steepest descent under a block-max norm. On ERA5-to-CERRA each contribution improves accuracy, with the largest gains on the rough variables a fixed operator mishandles and on the high-magnitude field one guidance scale under-serves. Richer operator architectures and more variables are natural next steps.

## Acknowledgments

This publication is part of the project DAILSCOM: Atmospheric Turbulence Informed Machine Learning for Laser Satellite Communications with file number 20617 of the research programme Open Technology Program 2023-2, which is financed by the Dutch Research Council (NWO). This work is also part of the project Delft AI4WF: Delft Artificial Intelligence for Weather Forecast with file number 2024.023 of the research programme Computing time of national computer systems, which is financed by NWO under the grant https://doi.org/10.61686/VYGRS56933.

## References

Michael Aich, Andreas Fürst, Florian Sestak, Carlos Ruiz-Gonzalez, Niklas Boers, and Johannes Brandstetter. WIND: Weather inverse diffusion for zero-shot atmospheric modeling. In Proceedings ofthe International Conference on Machine Learning, 2026.

David C. Bader. Climate models: An assessment of strengths and limitations. U.S. Department of Energy, 2009.

Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate mediumrange global weather forecasting with 3D neural networks. Nature, 2023.

Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A. Weyn, Haiyu Dong, Jayesh K. Gupta, Kit Thambiratnam, Alexander T. Archibald, Chun-Chieh Wu, Elizabeth Heider, Max Welling, Richard E. Turner, and Paris Perdikaris. A foundation model for the Earth system. Nature, 2025.

Noah D. Brenowitz, Tao Ge, Akshay Subramaniam, Peter Manshausen, Aayush Gupta, David M. Hall, Morteza Mardani, Arash Vahdat, Karthik Kashinath, and Michael S. Pritchard. Climate in a bottle: Towards a generative foundation model for the kilometer-scale global atmosphere. arXiv preprint arXiv:2505.06474, 2025.

Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In Proceedings of the International Conference on Learning Representations, 2023.

Stathi Fotiadis, Noah D Brenowitz, Tomas Geffner, Yair Cohen, Michael Pritchard, Arash Vahdat, and Morteza Mardani. Adaptive flow matching for resolving small-scale physics. In Proceedings ofthe International Conference on Machine Learning, 2025.

Rod Frehlich, Robert Sharman, Francois Vandenberghe, Wei Yu, Yubao Liu, Jason Knievel, and George Jumper. Estimates of $C _ { n } ^ { 2 }$ from numerical weather prediction model output and comparison with thermosonde data. Journal of Applied Meteorology and Climatology, 2010.

W. J. Gutowski, P. A. Ullrich, A. Hall, L. R. Leung, T. A. O’Brien, C. M. Patricola, R. W. Arritt, M. S. Bukovsky, K. V. Calvin, Z. Feng, A. D. Jones, G. J. Kooperman, E. Monier, M. S. Pritchard, S. C. Pryor, Y. Qian, A. M. Rhoades, A. F. Roberts, K. Sakaguchi, N. Urban, and C. Zarzycki. The ongoing need for high-resolution regional climate models: Process understanding and stakeholder information. Bulletin ofthe American Meteorological Society, 2020.

Paula Harder, Alex Hernandez-Garcia, Venkatesh Ramesh, Qidong Yang, Prasanna Sattegeri, Daniela Szwarcman, Campbell Watson, and David Rolnick. Hard-constrained deep learning for climate downscaling. Journal of Machine Learning Research, 2023.

Nidhin Harilal, Bri-Mathias S Hodge, Claire Monteleoni, and Aneesh Subramanian. EnhancedSD: Downscaling solar irradiance from climate model projections. In NeurIPS 2022 Workshop on Tackling Climate Change with Machine Learning, 2022.

Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, András Horányi, Joaquín Muñoz-Sabater, Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, Adrian Simmons, Cornel Soci, Saleh Abdalla, Xavier Abellan, Gianpaolo Balsamo, Peter Bechtold, Gionata Biavati, Jean Bidlot, Massimo Bonavita, Giovanna Chiara, Per Dahlgren, Dick Dee, Michail Diamantakis, Rossana Dragani, Johannes Flemming, Richard Forbes, Manuel Fuentes, Alan Geer, Leo Haimberger, Sean

Healy, Robin J. Hogan, Elías Hólm, Marta Janisková, Sarah Keeley, Patrick Laloyaux, Philippe Lopez, Cristina Lupu, Gabor Radnoti, Patricia Rosnay, Iryna Rozum, Freja Vamborg, Sebastien Villaume, and Jean-Noël Thépaut. The ERA5 global reanalysis. Quarterly Journal ofthe Royal Meteorological Society, 2020.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.

Kevin Höhlein, Michael Kern, Timothy Hewson, and Rüdiger Westermann. A comparative study of convolutional neural network models for wind field downscaling. Meteorological Applications, 2020.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, 2022.

Morteza Mardani, Noah Brenowitz, Yair Cohen, Jaideep Pathak, Chieh-Yu Chen, Cheng-Chin Liu, Arash Vahdat, Mohammad Amin Nabian, Tao Ge, and Akshay Subramaniam. Residual corrective diffusion modeling for km-scale atmospheric downscaling. Communications Earth & Environment, 2025.

Fabio Merizzi, Andrea Asperti, and Stefano Colamonaco. Wind speed super-resolution and validation: from ERA5 to CERRA via diffusion models. Neural Computing and Applications, 2024.

PhysicsNeMo Contributors. NVIDIA PhysicsNeMo: An open-source framework for physics-based deep learning in science and engineering. NVIDIA PhysicsNeMo GitHub repository, 2023.

Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, and Peter Battaglia. Probabilistic weather forecasting with machine learning. Nature, 2025.

Liying Qiu, Rahman Khorramfar, Saurabh Amin, and Michael F Howland. Decarbonized energy system planning with high-resolution spatial representation of renewables lowers cost. Cell Reports Sustainability, 2024.

François Rozet and Gilles Louppe. Score-based data assimilation. In Advances in Neural Information Processing Systems, 2023.

Carlo Saccardi, Maximilian Pierzyna, Haitz Sáez de Ocáriz Borde, Simone Monaco, Cristian Meo, Pietro Liò, Rudolf Saathof, Geethu Joseph, and Justin Dauwels. Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling. In NeurIPS 2025 Workshop on AIfor Science, 2025.

Eric P Salathé Jr, L Ruby Leung, Yun Qian, and Yongxin Zhang. Regional climate model projections for the state of Washington. Climatic Change, 2010.

S. Schimanke, M. Ridal, P. Le Moigne, L. Berggren, P. Undén, R. Randriamampianina, U. Andrea, E. Bazile, A. Bertelsen, P. Brousseau, P. Dahlgren, L. Edvinsson, A. El Said, M. Glinton, S. Hopsch, L. Isaksson, R. Mladek, E. Olsson, A. Verrelle, and Z.Q. Wang. CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), 2021.

Jonathan Schmidt, Luca Schmidt, Felix M Strnad, Nicole Ludwig, and Philipp Hennig. A generative framework for probabilistic, spatiotemporally coherent downscaling of climate simulation. npj Climate and Atmospheric Science, 2025.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Ruian Tie, Wenbo Xiong, Zhengyu Shi, Xinyu Su, Chenyu Jiang, Libo Wu, and Hao Li. Zero-shot statistical downscaling via diffusion posterior sampling. arXiv preprint arXiv:2601.21760, 2026.

Zhong Yi Wan, Ricardo Baptista, Anudhyan Boral, Yi-Fan Chen, John Anderson, Fei Sha, and Leonardo Zepeda-Núñez. Debias coarsely, sample conditionally: Statistical downscaling through optimal transport and probabilistic diffusion models. In Advances in Neural Information Processing Systems, 2023.

Robert L Wilby, TML Wigley, D Conway, PD Jones, BC Hewitson, J Main, and DS Wilks. Statistical downscaling of general circulation model output: A comparison of methods. Water Resources Research, 1998.

Qidong Yang, Alex Hernandez-Garcia, Paula Harder, Venkatesh Ramesh, Prasanna Sattigeri, Daniela Szwarcman, Campbell D Watson, and David Rolnick. Fourier neural operators for arbitrary resolution climate data downscaling. Journal ofMachine Learning Research, 2024.

## A DPS-EDM algorithm

Algorithm 1 gives the full sampling procedure: an EDM stochastic Heun solver with the learned operator in the likelihood and per-channel normalization of the guidance gradient (line 6). The EDM parameterization is convenient here because its denoiser $D _ { \theta }$ predicts $\hat { \mathbf { x } } _ { 0 }$ directly from the noisy state, so the guidance operator $\mathbf { A } _ { \phi }$ can be applied to $\hat { \mathbf { x } } _ { 0 }$ without a separate Tweedie conversion. The correction is applied at both the predictor and corrector stages.

Algorithm 1 EDM sampling with a learned operator and per-channel guidance normalization   
Require: N, y, h, D<sub>θ</sub>, A<sub>ϕ</sub>, ζ, ρ   
1: $\begin{array} { r } { \sigma _ { i }  \big ( \sigma _ { \operatorname* { m a x } } ^ { 1 / \rho } + \frac { i } { N - 1 } ( \sigma _ { \operatorname* { m i n } } ^ { 1 / \rho } - \sigma _ { \operatorname* { m a x } } ^ { 1 / \rho } ) \big ) ^ { \rho } , \quad \sigma _ { N }  0 } \end{array}$   
2: $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { 0 } ^ { 2 } \mathbf { I } )$   
3: for $i = 0$ to $\dot { N } - 1$ do   
4: xˆ<sub>0</sub> $ D _ { \theta } ( \mathbf { x } _ { i } ; \sigma _ { i } , \mathbf { h } )$   
5: $\mathbf { g }  \nabla _ { \mathbf { x } _ { i } } \| \mathbf { y } - \mathbf { A } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } , \mathbf { h } ) \| _ { 2 }$   
6: $\hat { \mathbf { g } } _ { c } \gets \mathbf { g } _ { c } / ( \lVert \mathbf { g } _ { c } \rVert _ { F } + \varepsilon )$ for each channel c ▷ per-channel, ours   
7: $\begin{array} { r } { \mathbf { d } _ { i }  \frac { \mathbf { x } _ { i } - \hat { \mathbf { x } } _ { 0 } } { \sigma _ { i } } + \zeta \hat { \mathbf { g } } } \end{array}$   
8: $\mathbf { x } _ { \mathrm { t e n t } }  \dot { \mathbf { x } } _ { i } + ( \sigma _ { i + 1 } - \sigma _ { i } ) \mathbf { d } _ { i }$   
9: if $i < N - 1$ then   
10: $\hat { \mathbf { x } } _ { 0 } ^ { \prime } \gets D _ { \theta } \big ( \mathbf { x } _ { \mathrm { t e n t } } ; \boldsymbol { \sigma } _ { i + 1 } , \mathbf { h } \big )$   
11: $\mathbf { g } ^ { \prime }  \nabla _ { \mathbf { x } _ { \mathrm { t e n t } } } \| \mathbf { y } - \dot { \mathbf { A } } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ^ { \prime } , \mathbf { h } ) \| _ { 2 } , \quad \hat { \mathbf { g } } _ { c } ^ { \prime }  \mathbf { g } _ { c } ^ { \prime } / ( \| \mathbf { g } _ { c } ^ { \prime } \| _ { F } + \varepsilon )$   
12: $\begin{array} { r } { \mathbf { d } _ { i } ^ { \prime } \gets \frac { \mathbf { x } _ { \mathrm { t e n t } } - \hat { \mathbf { x } } _ { 0 } ^ { \prime } } { \sigma _ { i + 1 } } + \zeta \hat { \mathbf { g } } ^ { \prime } } \end{array}$   
13: $\begin{array} { r } { \mathbf { x } _ { i + 1 }  \mathbf { x } _ { i } + ( \sigma _ { i + 1 } - \sigma _ { i } ) \frac { \mathbf { d } _ { i } + \mathbf { d } _ { i } ^ { \prime } } { 2 } } \end{array}$   
14: else   
15: $\mathbf { x } _ { i + 1 }  \mathbf { x _ { \mathrm { t } } }$ ent   
16: end if   
17: end for   
18: return $\mathbf { x } _ { N }$

## B Per-channel gradient normalization: derivation

We give the full derivation summarized in Section 4.2. We first factor the per-channel gradient magnitude to expose why it is imbalanced across variables (App. B.1), then state and prove that per-channel normalization is normalized steepest descent under the block-max norm, with DPS the corresponding statement under the Euclidean norm (App. B.2).

## B.1 Factorization of the per-channel gradient

Let $\mathbf { x } _ { \sigma }$ be the noisy state at level σ and $\hat { \mathbf { x } } _ { 0 } = D _ { \theta } ( \mathbf { x } _ { \sigma } ; \sigma )$ the denoiser output. The DPS guidance gradient is $\mathbf { g } = \nabla _ { \mathbf { x } _ { \sigma } } \| \mathbf { y } - \mathbf { A } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ) \| _ { 2 }$ , and by the chain rule through $\hat { \mathbf { x } } _ { 0 }$

$$
\mathbf { g } = - \mathbf { J } ^ { \top } \mathbf { A } _ { \phi } ^ { \prime } ( \hat { \mathbf { x } } _ { 0 } ) ^ { \top } \mathbf { r } , \qquad \mathbf { J } = \frac { \partial \hat { \mathbf { x } } _ { 0 } } { \partial \mathbf { x } _ { \sigma } } , \qquad \mathbf { r } = \frac { \mathbf { y } - \mathbf { A } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ) } { \| \mathbf { y } - \mathbf { A } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ) \| _ { 2 } } ,\tag{6}
$$

where J is the denoiser Jacobian and r the unit residual. The magnitude of g in a given channel is therefore governed by J: the residual r is a unit vector, and the operator $\mathbf { A } _ { \phi } ^ { \prime }$ is a fixed low-pass map, so the term that varies sharply across variables is the denoiser Jacobian. For a smooth variable such as surface pressure, $\hat { \mathbf { x } } _ { 0 }$ barely changes as $\mathbf { x } _ { \sigma }$ is perturbed, so $\| \mathbf { J } _ { c } \|$ is small; for a rough, high-frequency variable such as friction velocity, small perturbations of $\mathbf { x } _ { \sigma }$ produce large changes in $\hat { \mathbf { x } } _ { 0 } , \mathbf { s o } \parallel \mathbf { J } _ { c } \parallel$ is large. The per-channel gradient magnitudes $\| \mathbf { g } _ { c } \| _ { F }$ inherit this spread and differ by orders across the $C$ variables. Since the DPS drift adds $\zeta \mathbf { g }$ with a single scalar $\zeta ,$ the displacement of each variable is proportional to $\| \mathbf { g } _ { c } \| _ { F } \colon$ the roughest variables dominate the update while the smoothest are barely moved.

## B.2 Per-channel normalization is block-max steepest descent

Per-channel normalization is not an ad-hoc rescaling: it is the normalized steepest-descent direction on the DPS objective under the block-max norm. We contrast it below with the Euclidean-normalized step, and with DPS itself, which applies no normalization across channels at all. Normalized steepest descent on $L$ under a norm $\| \cdot \|$ takes the unit-ball direction that best aligns with the negative gradient,

$$
\mathbf { v } ^ { \star } \ = \ \arg \operatorname* { m i n } _ { \| \mathbf { v } \| \leq 1 } \ \langle \mathbf { g } , \mathbf { v } \rangle ,\tag{7}
$$

and different norms yield different steps. Partition v and $\mathbf { g }$ into the $C$ channel blocks, so that $\begin{array} { r } { \langle \mathbf { g } , \mathbf { v } \rangle = \sum _ { c = 1 } ^ { C } \langle \mathbf { g } _ { c } , \mathbf { v } _ { c } \rangle } \end{array}$ . Under the block-max norm $\left\| \mathbf { v } \right\| _ { \infty , F } : = \operatorname* { m a x } _ { c } \left\| \mathbf { v } _ { c } \right\| _ { F } .$ , the unit ball is the product set $\smash { \big \{ \mathbf { v } : \| \mathbf { v } _ { c } \| _ { F } \leq 1 \forall c \big \} }$ , so the constraint separates across channels. The objective is separable too, so $( 7 )$ splits into $\bar { C }$ independent problems min $\mathbf { \nabla } \cdot \| \mathbf { v } _ { c } \| _ { F } \leq 1  \big \langle \mathbf { g } _ { c } , \mathbf { v } _ { c } \big \rangle$ . Each is solved by Cauchy–Schwarz, giving

$$
\mathbf { v } _ { c } ^ { \star } = - \frac { \mathbf { g } _ { c } } { \| \mathbf { g } _ { c } \| _ { F } } , \qquad c = 1 , \ldots , C ,\tag{8}
$$

which is exactly the per-channel normalized direction of (5): every variable contributes a unit-norm step, independent of its raw gradient magnitude. Under the Euclidean norm, by contrast, the unit ball $\| \mathbf { v } \| _ { 2 } \leq 1$ does not separate across channels, and the steepest-descent direction is the joint normalization

$$
\mathbf { v } ^ { \star } = - \frac { \mathbf { g } } { \| \mathbf { g } \| _ { 2 } } , \qquad \| \mathbf { g } \| _ { 2 } = \Big ( \sum _ { c } \| \mathbf { g } _ { c } \| _ { F } ^ { 2 } \Big ) ^ { 1 / 2 } .\tag{9}
$$

Here every block is divided by the same scalar $1 / \lVert \mathbf { g } \rVert _ { 2 } ,$ , so its step retains the raw per-channel magnitude $\| \mathbf { g } _ { c } \| _ { F }$ and the imbalance of the previous subsection survives. DPS is the unnormalized limit of the same picture. Substituting $\zeta _ { i } \ : = \ : \zeta / \| \mathbf { y } - \mathbf { A } _ { \phi } ( \hat { \mathbf { x } } _ { 0 } ) \| _ { 2 }$ into (1) and using $\nabla \| \mathbf { r } \| _ { 2 } ^ { 2 } =$ $2 \| \mathbf { r } \| _ { 2 } \nabla \| \mathbf { r } \| _ { 2 }$ , the residual norm cancels and the update is a fixed step $2 \zeta$ along g itself, as written in (4): the per-sample scalar sets the overall step size but performs no rescaling across channels. Whether the common factor is $1 / \Vert \mathbf { g } \Vert _ { 2 }$ or absent entirely, the ratio between channels is unchanged and the loudest variables dominate the update. Block-max normalization coincides with these steps only when all $\left\| \mathbf { g } _ { c } \right\| _ { F }$ are equal; whenever they differ, it gives each variable its own unit-norm step.

Consequence for $\zeta .$ Under the Euclidean step (9) the displacement of variable c is $\zeta \| \mathbf { g } _ { c } \| _ { F } / \| \mathbf { g } \| _ { 2 }$ which depends on the channel and on σ through the Jacobian imbalance of (6), so no single $\zeta$ is simultaneously appropriate for all variables. Under the block-max step (8) the displacement of every variable is exactly $\zeta ,$ independent of c and of σ. The guidance strength therefore acts as one scale-free displacement budget shared across all channels, which is what allows a single validated $\zeta$ to serve the full multivariate field.

## C Experimental details

Data. CERRA [Schimanke et al., 2021] is a pan-European regional reanalysis at ${ \sim } 5 . 5$ km, to our knowledge the only kilometer-scale reanalysis with continental coverage. We pair it with ERA5 [Hersbach et al., 2020] over two domains, Central Europe and Iberia, on six surface variables: 10 m winds $( u _ { 1 0 } , v _ { 1 0 } )$ , 2 m temperature $\left( t _ { 2 m } \right)$ , surface sensible heat flux (sshf), friction velocity (zust), and surface pressure (sp). ERA5 (∼25 km, $7 7 \times 7 7 )$ is regridded to the CERRA projection (∼5.5 km, $3 5 0 \times 3 5 0 )$ , a ∼4.55× factor; the grid ratio $3 5 0 / 7 7$ is not an integer, so no exact area-average stencil exists. A CERRA-derived orography field is included as a static channel and all variables are standardized with per-variable ERA5 statistics. We train on 2013–2019 and the odd months of 2020, validate on the even months of 2020, and test on 2021. Unlike joint multi-region training, the prior and operator are trained separately per domain: one model on Central Europe and one on Iberia, each evaluated on its own held-out test year.

Backbone and training. The prior is an EDM denoiser [Karras et al., 2022] on a SongUNet backbone (as implemented in NVIDIA PhysicsNeMo [PhysicsNeMo Contributors, 2023]) with base channel count 32, channel multipliers (1, 2, 2), and 4 residual blocks per resolution. Its input concatenates the noised target (6 channels) and the orography (1 channel); it never sees ERA5. We train with Adam $( \beta _ { 1 } { = } 0 . 9 , \bar { \beta _ { 2 } } { = } 0 . 9 9 9 )$ at a constant learning rate $8 \times 1 0 ^ { - 4 }$ in mixed precision. During training the noise level is drawn log-normally $( P _ { \mathrm { { m e a n } } } { = } 0 . { \dot { 0 } } , P _ { \mathrm { { s t d } } } { = } 1 . 2 , \sigma _ { \mathrm { { d a t a } } } { = } 1 . 0 )$ , with the network supported over $\sigma _ { \mathrm { m i n } } { = } 0 . 0 0 2 \ \mathrm { t o } \ \sigma _ { \mathrm { m a x } } { = } 8 0 0$ . Sampling uses the stochastic Heun solver with 18 steps, and all reported quantities use an 8-member ensemble. $\mathbf { A } _ { \phi }$ corrects an area-interpolation of $\mathbf { x } _ { \mathrm { 0 } }$ with a two-layer convolutional network $( 3 \times 3$ kernels, 16 hidden channels, GELU) that also sees the interpolated orography, fitted by (3) on the same paired data and frozen at inference.

Guidance weight. The scalar ζ is selected by grid search on the validation split, minimizing the mean per-variable RMSE across the six standardized variables, and tuned separately for the per-channel and DPS arms because the two place ζ on different scales (a unit-norm direction versus a raw gradient). We sweep $\zeta \in \{ 0 . 0 1 , 0 . 1 , 0 . 5 , 1 , 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 0 0 0 \}$ for the per-channel arm and DPS, retaining in each case the value with the lowest validation error. The selected weights are ζ=10 for the per-channel arm and ζ=200 for DPS, identical on both regions.

Metrics. We report per-variable root mean square error (RMSE) and continuous ranked probability score (CRPS) of the 8-member ensemble, both in physical units, as pointwise and probabilistic measures of accuracy. Neither penalizes a method for producing an over-smooth field that scores well on average yet misses the fine-scale structure downscaling is meant to recover, so we complement them with two distributional diagnostics. Isotropic power spectral density (PSD) plots (Figures 2 and 4) show how each method distributes energy across wavenumbers, revealing whether guidance restores the small-scale variability of the CERRA target or flattens it. Per-variable marginal distribution (Figures 3 and 5), plotted as $\log _ { 1 0 }$ densities so the tails remain visible, show whether the full range of values, including rare extremes, is reproduced rather than concentrated near the mean.

Baselines. For context we also train three models that condition on the coarse field at training time rather than using it at inference: a conditional diffusion model (CondDiff), CorrDiff [Mardani et al., 2025], and a deterministic U-Net, all sharing the backbone above. Their results and per-variable marginal distributions are reported in Appendix D.

## D Comparison with conditional baselines

For context we compare against three models that condition on the coarse field at training time rather than using it at inference: a conditional diffusion model (CondDiff), CorrDiff [Mardani et al., 2025], and a deterministic U-Net, all sharing the backbone of Appendix C. Tables 3 and 5 report RMSE and CRPS on both regions. We quantify the sampling uncertainty of every entry by bootstrapping the test fields (2000 resamples in blocks of 24); the resulting 95% interval half-widths are given in Tables 4 and $^ { 6 , }$ and cover the test draw only, not seed-to-seed variability. On sp, where the fixed-scale DPS baseline is worst, the gap to ours is far larger than either interval $( 1 5 3 . 7 \dot { \pm } 1 3 . 9 \mathrm { v s . 9 2 . 0 5 } \pm 3 . 6 $ on Central Europe, $1 6 1 . 6 \pm \mathrm { { \bar { 1 } 1 . 1 ~ v s . ~ 1 0 6 . 7 \pm \mathrm { { \bar { 5 } } . 2 } } }$ on Iberia), while on the remaining variables the methods overlap.

The conditional models are the more accurate here, which is expected: they are trained end-to-end to map the coarse field directly to the high-resolution target, so the coarse-to-fine relationship is learned by the full diffusion model rather than mediated by a small operator inside the likelihood. Our posterior-sampling approach remains competitive across variables, though less accurate than the conditional models. What it retains instead is the modularity of the inverse-problem framing: because the coarse field enters only through the learned operator $\mathbf { A } _ { \phi } ,$ adapting the method to a different coarse source, for instance downscaling a climate model instead of ERA5, requires retraining only $\mathbf { A } _ { \phi } ,$ a two-layer network, while the diffusion prior is reused unchanged (App. E). A conditional model, by contrast, is tied to the coarse source it was trained on and must be retrained in full when that source changes. Figures 2 and 4 report the power spectral density of every model against the CERRA target, and Figures 3 and 5 their marginal distributions, on both regions and all six variables.

Table 3: Central Europe: full comparison. Conditional baselines (top) condition on the coarse field at training time; the inference-time methods (bottom) use it through the likelihood, with Ours our full method. Best per column in bold (lower is better). <sup>†</sup>The U-Net is deterministic, so its CRPS is a single-prediction MAE, not comparable to the 8-member ensemble CRPS; it is excluded from the CRPS comparison.
<table><tr><td rowspan="2">Model</td><td colspan="2">u10</td><td colspan="2">v10</td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td>CondDiff CorrDiff [Mardani et al., 2025]</td><td>0.904 0.884</td><td>0.489 0.477</td><td>0.903</td><td>0.486</td><td>0.843</td><td>0.452</td><td>27.08</td><td>10.22</td><td>0.075</td><td>0.039</td><td>59.98</td><td>35.56</td></tr><tr><td>U-Net†</td><td>0.856</td><td>0.616</td><td>0.877 0.852</td><td>0.471 0.610</td><td>0.831 0.831</td><td>0.449 0.641</td><td>25.76 25.42</td><td>9.782 12.76</td><td>0.072 0.071</td><td>0.038 0.049</td><td>72.01 116.0</td><td>41.82 87.82</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bicubic A</td><td>1.199</td><td>0.739</td><td>1.174</td><td>0.718</td><td>1.338</td><td>0.806</td><td>33.83</td><td>15.48</td><td>0.199</td><td>0.119</td><td>131.2</td><td>75.46</td></tr><tr><td>DPS [Chung et al., 2023]</td><td>0.969</td><td>0.539</td><td>0.966</td><td>0.535</td><td>1.092</td><td>0.608</td><td>31.27</td><td>12.79</td><td>0.096</td><td>0.049</td><td>153.7</td><td>93.17</td></tr><tr><td>Ours</td><td>0.986</td><td>0.553</td><td>0.982</td><td>0.547</td><td>1.021</td><td>0.571</td><td>32.70</td><td>13.33</td><td>0.090</td><td>0.047</td><td>92.05</td><td>55.96</td></tr></table>

Table 4: Central Europe: bootstrap intervals. Each entry is the half-width (±) of the 95% interval for the matching entry in Table 3; the full interval is that table’s value ± the half-width here. From 2000 resamples of the test fields in blocks of 24. The interval covers the test draw only; with a single training seed it says nothing about seed-to-seed variability. <sup>†</sup>U-Net CRPS is a single-member MAE (see Table 3).
<table><tr><td rowspan="2">Model</td><td colspan="2">u10</td><td colspan="2"> $v _ { 1 0 }$ </td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td>CondDiff CorrDiff [Mardani et al., 2025]</td><td>0.020 0.020</td><td>0.010 0.010</td><td>0.020 0.020</td><td>0.010 0.009</td><td>0.029 0.028</td><td>0.015 0.015</td><td>3.29 3.21</td><td>0.841 0.867</td><td>0.0018 0.0016</td><td>0.0010 0.0009</td><td>3.76 2.38</td><td>2.41 1.34</td></tr><tr><td>U-Net†</td><td>0.019</td><td>0.012</td><td>0.019</td><td>0.012</td><td>0.026</td><td>0.018</td><td>3.11</td><td>1.06</td><td>0.0015</td><td>0.0011</td><td>3.09</td><td>2.40</td></tr><tr><td>Bicubic A</td><td>0.035</td><td>0.023</td><td>0.031</td><td>0.019</td><td>0.045</td><td>0.031</td><td>3.32</td><td>1.06</td><td>0.0084</td><td>0.0054</td><td>9.20</td><td>5.15</td></tr><tr><td>DPS [Chung et al., 2023]</td><td>0.021</td><td>0.010</td><td>0.021</td><td>0.009</td><td>0.035</td><td>0.021</td><td>3.39</td><td>1.00</td><td>0.0036</td><td>0.0018</td><td>13.85</td><td>8.96</td></tr><tr><td>Ours</td><td>0.022</td><td>0.011</td><td>0.021</td><td>0.010</td><td>0.032</td><td></td><td>3.65</td><td>1.09</td><td></td><td></td><td>3.62</td><td>2.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.019</td><td></td><td></td><td>0.0023</td><td>0.0011</td><td></td><td></td></tr></table>

Table 5: Iberia: full comparison. Layout and conventions as in Table 3. <sup>†</sup>The U-Net CRPS is a single-member MAE and is excluded from the CRPS comparison.
<table><tr><td rowspan="2">Model</td><td colspan="2"> $u _ { 1 0 }$ </td><td colspan="2"> $v _ { 1 0 }$ </td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td>CondDiff CorrDiff [Mardani et al., 2025]</td><td>1.039 1.020</td><td>0.552 0.541</td><td>1.070 1.036</td><td>0.571 0.550</td><td>0.916 0.917</td><td>0.500 0.507</td><td>26.47 25.28</td><td>9.923 9.548</td><td>0.077 0.074</td><td>0.040 0.039</td><td>74.09 79.20</td><td>45.00 46.70</td></tr><tr><td>U-Net†</td><td>0.998</td><td>0.702</td><td>1.015</td><td>0.714</td><td>0.916</td><td>0.733</td><td>25.08</td><td>12.30</td><td>0.073</td><td>0.050</td><td>127.4</td><td>98.46</td></tr><tr><td>Bicubic A</td><td>1.362</td><td>0.827</td><td>1.372</td><td>0.828</td><td>1.809</td><td>1.152</td><td>41.18</td><td>20.43</td><td>0.174</td><td>0.095</td><td>156.1</td><td>88.86</td></tr><tr><td>DPS [Chung et al., 2023]</td><td>1.172</td><td>0.661</td><td>1.186</td><td>0.662</td><td>1.413</td><td>0.842</td><td>38.22</td><td>17.28</td><td>0.096</td><td>0.051</td><td>161.6</td><td>91.88</td></tr><tr><td>Ours</td><td>1.197</td><td>0.678</td><td>1.217</td><td>0.684</td><td>1.377</td><td>0.824</td><td>40.23</td><td>18.28</td><td>0.096</td><td>0.052</td><td>106.7</td><td>63.28</td></tr></table>

Table 6: Iberia: bootstrap intervals. Each entry is the half-width (±) of the 95% interval for the matching entry in Table 5; the full interval is that table’s value ± the half-width here. Resampling as in Table 4.
<table><tr><td rowspan="2">Model</td><td colspan="2"> $_ { u _ { 1 0 } }$ </td><td colspan="2"> $v _ { 1 0 }$ </td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td>CondDiff</td><td>0.025</td><td>0.012</td><td>0.027</td><td>0.014</td><td>0.022</td><td>0.011</td><td>1.99</td><td>0.592</td><td>0.0017</td><td>0.0008</td><td>6.34</td><td>4.34</td></tr><tr><td>CorrDiff [Mardani et al., 2025] U-Net†</td><td>0.027 0.025</td><td>0.013 0.016</td><td>0.027 0.027</td><td>0.014 0.017</td><td>0.023 0.023</td><td>0.013 0.016</td><td>2.10 2.00</td><td>0.629 0.767</td><td>0.0017 0.0017</td><td>0.0009 0.0011</td><td>2.95 4.96</td><td>1.85 4.08</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bicubic A</td><td>0.031</td><td>0.020</td><td>0.028</td><td>0.017</td><td>0.039</td><td>0.029</td><td>2.54</td><td>1.11</td><td>0.0078</td><td>0.0034</td><td>10.99</td><td>5.70</td></tr><tr><td>DPS [Chung et al., 2023]</td><td>0.024</td><td>0.013</td><td>0.026</td><td>0.013</td><td>0.043</td><td>0.029</td><td>2.60</td><td>1.08</td><td>0.0024</td><td>0.0012</td><td>11.05</td><td>5.99</td></tr><tr><td>Ours</td><td>0.026</td><td>0.014</td><td>0.028</td><td>0.014</td><td>0.031</td><td>0.020</td><td>2.74</td><td>1.18</td><td>0.0023</td><td>0.0012</td><td>5.24</td><td>3.43</td></tr></table>

![](images/c36ff175f9d155cc7671bfffe523aadd09e04945e6b5dc664af36f878ff62589.jpg)

The spectra and marginals in Figures 2 and 3 tell a consistent story. Across the resolved scales every method overlies the CERRA target, so the differences appear only at the highest wavenumbers, and only for the rough variables. For sshf and zust, the fixed-operator and deterministic baselines (bicubic and U-Net) lose high-frequency energy and their marginals contract toward the mean, undershooting the target tails. Ours retains the most small-scale energy and reproduces these tails most closely, with DPS close behind. For the smooth variables $( u _ { 1 0 } , v _ { 1 0 } , t _ { 2 m } , \mathrm { s p } )$ all methods track both the spectrum and the full marginal, with negligible separation. The large sp error gap between our proposed per-channel gradient normalization and DPS seen in the tables leaves no visible trace here, since it reflects a bias in the mean field rather than a mismatch in the spectrum or marginal distribution.

wavenumber k [cycles cell<sup>−1</sup>]  
Figure 2: Power spectral density, Central Europe. Test split, 8-member ensembles. The inset magnifies the highest 0.15 decades of wavenumber, where the curves separate.  
![](images/fcf66af522a42c8175e5fc5a788624d8be547e459c3ebdb90ee154ec6a36e2af.jpg)

![](images/3c655b8b287909d67a5fd70690b0a23315dbbd3c4bcfdf94ea1e6bb55e9b1033.jpg)

![](images/a6ceb74de1202970e67f6b1a70afcb05a905ece38bf11a2fbdee015c7f36ff0f.jpg)

![](images/5c2bf8157a853053190b6983dcec2741475ba122bfbaa5f867a559842c1f6091.jpg)

![](images/7284a22b4e5d22187b4738ff2ffca51c16c209c1c8c8e0bb9012bd56f0b85c31.jpg)

![](images/2a63da652f9a44920ce130c1d7be03f6fc6f5ec458f309d120b406a2874a0465.jpg)  
Figure 3: Marginal distributions, Central Europe. $\log _ { 1 0 }$ of the probability density over the test split, so the tails stay visible.

Figures 4 and 5 show the same pattern on Iberia. The methods again agree on the smooth variables and separate only on the rough ones (sshf, zust), at the highest wavenumbers and in the tails: bicubic and U-Net lose small-scale energy and contract their marginals, while ours stays closest to the CERRA target, here slightly overshooting its high-frequency energy rather than falling short. As on Central Europe, the large sp error gap between ours and DPS is not visible in either diagnostic, consistent with its being a mean-field bias rather than a spectral or distributional one.

![](images/fc24de09c521a7d7f4056b969e1c4d10c8d7812b36cebde25e78d07f1b8e8e82.jpg)  
wavenumber k [cycles cell<sup>−1</sup>]

Figure 4: Power spectral density, Iberia. Test split, 8-member ensembles. The inset magnifies the highest 0.15 decades of wavenumber, where the curves separate.  
![](images/7bc855eb6eb663c1d7eac5a1be75459234f3e6b6b855d4ce015c7fd7b8c81fb7.jpg)

![](images/5de21f196e0d8d165df6923e353910228fd68732b83063e2b90df2cfc8136367.jpg)

![](images/fcded0057ce4f6531eaec1c436a8fdd74a74e62b944bb09714c5f032e805f0ce.jpg)

![](images/539d275bb3b1868d1143f7eab6d4d46b7d0e7566959438eacb06ff08f7c5aade.jpg)

![](images/78f93c478007efb0f25e9340539b37a9003f7155a08aa138c664e299b5ac8c6d.jpg)

![](images/d3f07f57c3f5265b6fde994db60ff66602129ef427c8d0d7a32b886f9fdf87a4.jpg)  
Figure 5: Marginal distributions, Iberia. l $\mathrm { { 0 g _ { 1 0 } } }$ of the probability density over the test split, so the tails stay visible.

## E Transfer to a coarser source

Adapting to a new coarse source requires retraining only $\mathbf { A } _ { \phi } ,$ , and we test this directly: we halve the resolution of the coarse source to ∼50 km, refit $\mathbf { A } _ { \phi }$ on the new pairs, and leave the diffusion prior frozen and unchanged. Everything else in the pipeline is identical.

Table 7 reports the result next to our ${ \sim } 2 5$ km numbers, and Table 8 the corresponding bootstrap intervals. Accuracy is essentially unchanged on both regions: every difference is smaller than the combined intervals of the two estimates. Even with half the input resolution, the method keeps a large margin on sp over both the bicubic-operator and DPS arms at full resolution (95.68 here, against 131.2 and 153.7 on Central Europe). A conditional model cannot be run on this source at all without changing its input dimensions and retraining in full.  
Table 7: Transfer to $\mathbf { a } \sim 5 0$ km coarse source. RMSE and CRPS per variable, both regions. Only $\mathbf { A } _ { \phi }$ is refit; the prior is unchanged. The ∼25 km rows repeat Ours from Tables 3 and 5.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Coarse source</td><td colspan="2"> $u _ { 1 0 }$ </td><td colspan="2"> $v _ { 1 0 }$ </td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2">sshf</td><td colspan="2">zust</td><td colspan="2">sp</td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td rowspan="2">C. Europe</td><td>ERA5 ~25 km ERA5~50 km</td><td>0.986 0.994</td><td>0.553 0.557</td><td>0.982 0.989</td><td>0.547</td><td>1.021</td><td>0.571</td><td>32.70</td><td>13.33</td><td>0.090</td><td>0.047</td><td>92.05</td><td>55.96</td></tr><tr><td></td><td></td><td></td><td></td><td>0.551</td><td>1.055</td><td>0.590</td><td>36.79</td><td>14.80</td><td>0.091</td><td>0.047</td><td>95.68</td><td>58.85</td></tr><tr><td rowspan="2">Iberia</td><td>ERA5 ~25 km</td><td>1.197</td><td>0.678</td><td>1.217</td><td>0.684</td><td>1.377</td><td>0.824</td><td>40.23</td><td>18.28</td><td>0.096</td><td>0.052</td><td>106.7</td><td>63.28</td></tr><tr><td>ERA5~50 km</td><td>1.207</td><td>0.683</td><td>1.224</td><td>0.686</td><td>1.403</td><td>0.842</td><td>42.49</td><td>19.31</td><td>0.098</td><td>0.053</td><td>107.3</td><td>63.73</td></tr></table>

Table 8: Transfer to a ∼50 km coarse source: bootstrap intervals. Each entry is the half-width (±) of the 95% interval for the matching entry in Table 7. Resampling as in Table 4. The intervals for the ${ \sim } 2 5$ km rows are the Ours rows of Tables 4 and 6.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Coarse source</td><td colspan="2"> $_ { u _ { 1 0 } }$ </td><td colspan="2"> $v _ { 1 0 }$ </td><td colspan="2"> $t _ { 2 m }$ </td><td colspan="2"> $\mathrm { s s h f }$ </td><td colspan="2">zust</td><td colspan="2"> $\mathrm { s p }$ </td></tr><tr><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td><td>RMSE</td><td>CRPS</td></tr><tr><td rowspan="2">C. Europe</td><td>ERA5~25 km ERA5~50 km</td><td>0.022</td><td>0.011</td><td>0.021</td><td>0.010</td><td>0.032</td><td>0.019</td><td>3.65</td><td>1.09</td><td>0.0023</td><td>0.0011</td><td>3.62</td><td>2.46</td></tr><tr><td></td><td>0.025</td><td>0.013</td><td>0.022</td><td>0.010</td><td>0.041</td><td>0.024</td><td>4.29</td><td>1.25</td><td>0.0027</td><td>0.0013</td><td>5.08</td><td>3.57</td></tr><tr><td rowspan="2">Iberia</td><td>ERA5 ~25 km</td><td>0.026</td><td>0.014</td><td>0.028</td><td>0.014</td><td>0.031</td><td>0.020</td><td>2.74</td><td>1.18</td><td>0.0023</td><td>0.0012</td><td>5.24</td><td>3.43</td></tr><tr><td>ERA5 ~50 km</td><td>0.032</td><td>0.016</td><td>0.033</td><td>0.017</td><td>0.036</td><td>0.024</td><td>3.06</td><td>1.40</td><td>0.0025</td><td>0.0013</td><td>7.18</td><td>4.70</td></tr></table>

## F Additional figures

Figure 6: guided by a fixed linear operator, DPS (c) conflates the reconstruction residual with the ERA5-to-CERRA system bias and leaves visible artefacts, whereas our learned operator (d) recovers a cleaner field close to the CERRA target (b).

![](images/00b35cb99ffe08fd16aa3806416fd5c6aa1e4a9f4218ac7cf371d7e466008c75.jpg)  
(a) ERA5 Input

![](images/a409f75b027ed371198025c1da214b3fde8284703bbdd093f6526331ca9768fa.jpg)  
(b) CERRA Target

![](images/e84ef51be778c80f1e7bbe0d29727597d0f2b1f30cc3baa2c22dc3a2ed793c47.jpg)  
(c) Fixed Operator A

![](images/16dba6f75fa3e07feb0e7fdc9817e3b386442c62419771e5ea7396ecc488b880.jpg)  
(d) Learned operator (ours)  
Figure 6: sshf on a representative sample, Central Europe. (a) coarse ERA5 input, (b) CERRA target, (c) DPS with a fixed linear operator, (d) our learned operator.

Figure 7: ours and DPS share the same prior and learned operator and differ only in gradient scaling. The bicubic residual (d) is an order of magnitude larger than either model error (1028 Pa RMSE, vs. 78.6 ours and 288.2 DPS) and locked to terrain. The DPS error (f) is a broad one-sign offset (+250.4 Pa domain mean, vs. +54.4 ours): the field sits at the wrong pressure level rather than the wrong pattern.

(a) CERRA (target)  
![](images/a9a0a5459e8493c36ba5fcff42b62e7e5d9ca563e8d521acc15252afb7ffb23d.jpg)

(b) Ours  
![](images/605fc1925c6b195ac2580931f8dcec32fcf0790f148eb3052180fed3517b814b.jpg)

(c) DPS  
![](images/56f56b9eeb3d2a0905395fc2b9c7ca167a94d88e05811ca87d2de02a0b53ce6a.jpg)

(d) bicubic ERA5 CERRA  
![](images/3ff1de34c1f79e33b2b642c3e2d7aca749bb03bb3402410556970261eec45cc1.jpg)

(e) Ours CERRA  
![](images/cb999bb4c0c3b99b4cd687b1517b1c69b064298c9a7eb4ec10e4ce506ae69385.jpg)

![](images/158bc5840038ed86a8c6d9b2fc0a32112a034b70268ddb9bf269726b5fbb0bc8.jpg)  
Figure 7: Surface pressure on a single test timestep, Central Europe. Top: CERRA target and the 8-member ensemble mean from each method. Bottom: error fields on a common scale; (d) bicubic ERA5 residual, (e) ours, (f) DPS.