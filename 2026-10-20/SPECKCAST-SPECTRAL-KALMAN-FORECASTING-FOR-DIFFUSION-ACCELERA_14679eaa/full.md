# SPECKCAST: SPECTRAL KALMAN FORECASTING FOR DIFFUSION ACCELERATION

Javad Rajabi<sup>1,2,3∗</sup>, Hakki Can Karaimer<sup>3</sup>, David B. Lindell<sup>1,2</sup>, Babak Taati<sup>1,2</sup>, Michael S. Brown<sup>3</sup>

<sup>1</sup>University of Toronto <sup>2</sup>Vector Institute <sup>3</sup>AI Center–Toronto, Samsung Electronics rajabi@cs.toronto.edu

## ABSTRACT

Diffusion Transformers (DiTs) achieve high-fidelity image and video generation, but they suffer from substantial inference costs due to repeated transformer evaluations across many denoising steps. Feature caching reduces this cost by reusing features across denoising steps. Recent forecasting methods extend this idea by extrapolating features to skipped steps instead of reusing them. However, existing forecasting methods extrapolate the features they predict uniformly, even though the spatial-frequency components of these features change most at different stages of denoising. They also overlook that denoising progresses nonuniformly, so equal step intervals can correspond to very different amounts of change. These mismatches amplify forecast errors as more steps are skipped, limiting the achievable acceleration. We propose SpecKCast, a training-free method that forecasts the model output in the frequency domain. SpecKCast applies a 2D discrete cosine transform (DCT) to the output and tracks each coefficient with a lightweight Kalman filter along a band-specific time coordinate derived from the noise schedule. The filter tracks each coefficient’s value and rate of change, and the transformer is evaluated only when the filter’s forecast uncertainty exceeds a threshold. Experiments on FLUX.1-dev, Qwen-Image, Wan2.1-14B, and HunyuanVideo show that SpecKCast achieves a better trade-off between acceleration and fidelity than existing caching and forecasting methods, reaching up to 6× speedup with less than 0.05 GiB of additional memory.

TaylorSeer

Spectrum SpecKCast (ours)

![](images/6b143ceb4ad0b10098d1df43a9468c7fd5487e287170e0a91fdbf15e49f237b9.jpg)  
Figure 1: Qualitative comparison on FLUX at different acceleration ratios.

## 1 INTRODUCTION

Diffusion models (Ho et al., 2020) have become the dominant paradigm for high-fidelity visual generation, gaining significant attention from recent advances in image and video generation. Diffusion Transformers (DiTs) (Peebles & Xie, 2023) further improved generation quality by enabling greater model capacity. However, their costly forward passes throughout many denoising steps lead to substantially higher inference latency, limiting their use in real-time applications. To address this bottleneck, a wide range of acceleration techniques have been proposed to reduce either the step count or the per-step cost, including efficient attention (Xia et al., 2025), step distillation (Salimans & Ho, 2022), and quantization (Li et al., 2023).

Among these, feature caching has emerged as a practical, training-free approach. Motivated by the observation that model outputs are highly similar across adjacent timesteps, these methods cache features from previous steps and reuse them in subsequent ones, skipping substantial computation (Ma et al., 2024; Selvaraju et al., 2024). However, this similarity decays as the gap between timesteps grows, so naive reuse over longer spans significantly degrades generation quality, limiting caching to modest acceleration ratios. To address this, recent work shifts from reusing features to forecasting them (Liu et al., 2025b; Han et al., 2026). Rather than assuming that a cached feature remains valid at a later timestep, forecasting methods exploit temporal continuity along the denoising trajectory, using features from previous steps to estimate their evolution and extrapolate their values at the target timestep. By modeling temporal drift, forecasting remains effective across wider timestep gaps, enabling higher acceleration ratios at comparable generation quality.

Despite their promise, existing forecasting approaches impose uniformity on both what they forecast and how they model the signal over time (Liu et al., 2025b; Han et al., 2026). First, they extrapolate the model output or feature representation as a whole, or apply the same forecasting rule uniformly to all of its components (e.g., channels), even though the spatial-frequency components of the signal evolve very differently. As Figure 2 shows, low-frequency content changes sharply in the first few steps and then settles, while high-frequency content changes slightly at first but evolves rapidly toward the end. Because these bands are active at different stages, a single uniform forecaster may not serve them all. Second, existing methods use the sampler’s step index or raw diffusion timestep as their temporal coordinate, even though the amount of denoising performed at each step is nonuniform and depends on the scheduler. Therefore, equal intervals in the chosen coordinate can represent very different amounts of denoising progress. Treating these intervals as equivalent makes the resulting trajectory harder to model and forecast. Together, these limitations restrict achievable acceleration, amplify forecast errors, and degrade generation quality as the skip length increases.

Motivated by this insight, we propose SpecKCast, a training-free method that rethinks both what we forecast and how we extrapolate. The key idea is to forecast the model’s temporal evolution separately in the frequency domain, rather than uniformly in feature or latent space. SpecKCast transforms the model output into the spatial-frequency domain with a 2D discrete cosine transform (DCT), and forecasts each coefficient during denoising. Moreover, instead of indexing the trajectory by the sampler’s timestep, SpecKCast exploits a time coordinate derived from the noise scheduler and applies a nonlinear time warping that improves the accuracy of local extrapolation within each band. To track each coefficient as it moves along its trajectory, SpecKCast uses a lightweight Kalman filter, a recursive state-space estimator that maintains the coefficient’s value and rate of change, projecting this state across a skip and correcting it after a full denoising step. The filter also estimates the uncertainty of each forecast. As more steps are skipped, this uncertainty increases, while the rate of change decays to keep the prediction stable. Furthermore, SpecKCast partitions the coefficients into frequency bands that share a common temporal coordinate and forecast uncertainty, while each coefficient keeps its own value and rate of change. Finally, at skipped steps, the predicted coefficients are transformed back to the spatial domain and passed directly to the sampler.

The main contributions of this work are summarized as follows.

• Frequency-Decoupled and Scheduler-Aligned Forecasting. We introduce SpecKCast, a training-free method that decomposes the model output in the spatial-frequency domain and forecasts its components along a nonlinear time coordinate derived from the noise scheduler.

• Uncertainty-Aware Kalman Forecasting. SpecKCast uses a lightweight Kalman filter to track frequency coefficient trajectories and leverage forecast uncertainty to maintain stability over skipped denoising steps.

Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
![](images/11804502f28b14933a5a84c5d20fc57ca11e7dad27d18ca4adcae92efdb3488a.jpg)  
Figure 2: Spatial-frequency bands of the denoising velocity follow different trajectories. PCA trajectories of the 50 velocity predictions from a single FLUX.1-dev generation, projected onto the first two principal components. Larger gaps between consecutive points indicate larger changes. (a) Sampled entries of the full velocity. (b–d) Radial DCT bands: $B _ { 0 }$ moves most in the first step, $B _ { 3 }$ changes over a broader interval, and $B _ { 6 }$ moves most after step 40. Each panel has its own PCA basis and normalization. Additional views, prompts, and Qwen-Image results are in Section G.

• State-of-the-Art Performance. SpecKCast consistently improves the acceleration and generation quality trade-off over existing caching and forecasting methods while introducing negligible computational and memory overhead, requiring about 5 MiB of additional memory for FLUX.1 (Labs, 2024).

## 2 RELATED WORK

Diffusion models (Ho et al., 2020) have become a leading approach for image and video generation. Since the introduction of the Diffusion Transformer (DiT) (Peebles & Xie, 2023), many works have built on this architecture, achieving strong results across diverse domains (Chen et al., 2024a; Yang et al., 2025; Huang et al., 2025; Rajabi et al., 2026; Zhang et al., 2026). This progress, however, comes at a cost. Sampling requires many sequential forward passes through a large network, making inference slow and expensive. This limits practical applications and has made diffusion model acceleration an important research direction. A large body of work has therefore focused on reducing sampling steps, mostly by distilling a multi-step teacher into a few-step student (Meng et al., 2023). Examples include consistency models (Song & Dhariwal, 2024; Song et al., 2023) and DMD (Yin et al., 2024a;b). Although effective, these approaches require additional training and are prone to reduced sample diversity.

A more recent line of work explores feature caching, which accelerates inference without any additional training. The core idea is to reuse intermediate features across timesteps to avoid redundant computation (Zou et al., 2024; Gao et al., 2026; Xu et al., 2026; Peng et al., 2026). Early methods such as DeepCache (Ma et al., 2024) accelerated the U-Net backbone using handcrafted caching rules, and ∆-DiT (Chen et al., 2024b) later extended this idea to DiT architectures. Subsequent methods further refine what and when to cache. FORA (Selvaraju et al., 2024) reuses attention and MLP outputs across timesteps, while TeaCache (Liu et al., 2025a) exploits the correlation between timestep embeddings and model outputs to estimate how much the output changes between steps and decide when to reuse cached features. HiCache (Feng et al., 2026) forecasts features with scaled Hermite polynomials to capture their non-monotonic dynamics in DiTs. SeaCache (Chung et al., 2026) instead improves the caching decision by measuring feature changes after a spectral filter that preserves content-relevant components and suppresses noise. However, feature similarity decreases as the timestep gap increases. As more steps are skipped, the cached features deviate further from the actual ones, and these errors accumulate during sampling. This causes the trajectory to drift and degrades generation quality, limiting the achievable speedup.

To address this, recent work shifts from reusing features to forecasting them (Liu et al., 2025b; Han et al., 2026). Instead of assuming that cached features remain valid at later timesteps, forecasting methods use features from previous steps to predict how they evolve and estimate their values at the target timestep. TaylorSeer (Liu et al., 2025b) does this locally with a Taylor expansion, using finite differences of the most recent cached features, while Spectrum (Han et al., 2026) fits a Chebyshev expansion over the full feature history to better capture long-range dynamics. By modeling how features evolve, these methods remain effective over wider timestep gaps, enabling higher acceleration at comparable quality. However, both forecast uniformly in feature space and along the sampler’s timestep, ignoring that denoising progress is nonuniform across steps. In contrast, SpecKCast forecasts each frequency band separately along a scheduler-derived time axis, using a lightweight Kalman filter that adapts its forecasts as the trajectory evolves.

## 3 BACKGROUND

## 3.1 DIFFUSION MODELS AND FLOW MATCHING

Diffusion models (Ho et al., 2020) define a generative process that gradually transforms samples from a simple prior (e.g., Gaussian) into data via a learned reverse-time process. Flow matching (Lipman et al., 2022; Liu et al., 2022) instead learns a velocity field that transports noise to data along near-straight paths. Both are instances of stochastic interpolants (Albergo et al., 2025), with $x _ { t } = \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon$ , where rectified flow sets $\alpha _ { t } = 1 - t$ and $\sigma _ { t } = t$ and learns the velocity $v = \epsilon - x _ { 0 }$ Sampling integrates the reverse flow over decreasing noise levels using Euler updates:

$$
x _ { j + 1 } = x _ { j } + ( \sigma _ { j + 1 } - \sigma _ { j } ) Y _ { j } , \qquad Y _ { j } = v _ { \theta } ( x _ { j } , \sigma _ { j } , c ) ,\tag{1}
$$

where c denotes the text condition. We measure denoising cost by the number of function evaluations (NFE), defined as the number of denoising steps at which the transformer is evaluated; for models with classifier-free guidance (CFG) (Ho & Salimans, 2022), each such step includes both passes.

## 3.2 FORECASTING FOR DIFFUSION ACCELERATION

Adjacent denoising steps produce highly correlated features and outputs, allowing redundant computation to be avoided (Ma et al., 2024; Selvaraju et al., 2024; Liu et al., 2025b; Han et al., 2026). Forecasting exploits this temporal redundancy by predicting skipped signals from past observations. Let $\mathcal { U } \subseteq \{ \bar { 0 } , \dotsc , N - 1 \}$ denote the steps that evaluate the denoiser. For output-level forecasting,

$$
\begin{array} { r } { \widehat { Y } _ { j } = f ( h _ { j } , j ) , \qquad j \notin { \mathcal U } , } \end{array}\tag{2}
$$

where $h _ { j }$ encodes past outputs and $\widehat { Y } _ { j }$ replaces $Y _ { j }$ in the sampler. Forecasting may instead target intermediate features, making the choice of forecast target and representation a key design decision.

## 3.3 KALMAN FILTER

The Kalman filter (Kalman, 1960) is a classical method that recursively estimates the state of a dynamic signal from intermittent observations. It models the state transition as

$$
h _ { j } = A _ { j } h _ { j - 1 } + w _ { j } , \qquad w _ { j } \sim { \mathcal N } ( 0 , Q _ { j } ) ,\tag{3}
$$

$$
y _ { j } = H _ { j } h _ { j } + \eta _ { j } , \qquad \eta _ { j } \sim { \cal N } ( 0 , R _ { j } ) ,\tag{4}
$$

where $A _ { j }$ is the state-transition matrix, $H _ { j }$ maps the state to the observation, and $Q _ { j }$ and $R _ { j }$ are the process- and observation-noise covariances, respectively. The filter alternates between prediction and update, with updates only when observations are available. Given the previous estimate $\widehat { h } _ { j - 1 }$ and covariance $P _ { j - 1 }$ , it predicts

$$
\begin{array} { r } { \overline { { h } } _ { j } = A _ { j } \widehat { h } _ { j - 1 } , \qquad \overline { { P } } _ { j } = A _ { j } P _ { j - 1 } A _ { j } ^ { \top } + Q _ { j } . } \end{array}\tag{5}
$$

When an observation $y _ { j }$ is available, the prediction is corrected using

$$
\begin{array} { r } { K _ { j } = \overline { { P } } _ { j } H _ { j } ^ { \top } \left( H _ { j } \overline { { P } } _ { j } H _ { j } ^ { \top } + R _ { j } \right) ^ { - 1 } , } \end{array}\tag{6}
$$

$$
\widehat { h } _ { j } = \overline { { h } } _ { j } + K _ { j } ( y _ { j } - H _ { j } \overline { { h } } _ { j } ) , \qquad P _ { j } = ( I - K _ { j } H _ { j } ) \overline { { P } } _ { j } .\tag{7}
$$

Here, $K _ { j }$ controls the balance between the new observation and the model prediction. An important property of the Kalman filter is that it is fully online, requiring a constant number of operations and no window of past observations to store or refit, so a large number of signals can be tracked in parallel at negligible cost. Between observations, the filter advances the state, while the covariance grows to reflect increasing forecast uncertainty, making the filter suitable for forecasting over extended gaps between observations.

## 4 METHOD

Spectral Kalman ForeCasting (SpecKCast) is an inference-time acceleration method for DiTs that reduces transformer evaluations during denoising. Unlike feature-caching methods that approximate an intermediate activation and still execute the remaining network layers, SpecKCast models the final velocity consumed by the sampler. SpecKCast applies a 2D DCT to the model output and forecasts each frequency coefficient individually, while partitioning coefficients into bands that share forecast uncertainty and a noise-schedule-derived time coordinate, warped nonlinearly to better align with each band’s temporal evolution. SpecKCast uses a Kalman filter to track each DCT coefficient’s value and temporal derivative, while sharing covariance statistics within each frequency band. The predicted coefficients are transformed back to the spatial domain and passed directly to the sampler.

## 4.1 FREQUENCY-AWARE FORECASTING

## 4.1.1 SPATIAL-FREQUENCY DECOMPOSITION

Forecasting the entire transformer output with a single temporal model ignores an important property of diffusion predictions: different spatial frequencies exhibit different temporal dynamics, as shown in Figure 2. Low-frequency content settles early and drifts slowly, while high-frequency content remains active until the end of sampling. To account for these distinct dynamics, SpecKCast transforms each output into the spatial-frequency domain and forecasts its coefficients separately.

Given the denoiser output $Y _ { i } \in \mathbb { R } ^ { C \times H \times W }$ at timestep j, we apply an orthonormal 2D DCT independently to each channel, $Z _ { j } \doteq \mathrm { D C T } _ { \mathrm { 2 D } } ( Y _ { j } )$ ), and partition the frequency plane into L non-overlapping radial bands. Each frequency coefficient is treated as a separate temporal signal and forecast by its own predictor, while the band determines how each coefficient’s dynamics are modeled. On steps where the transformer computes the full denoising step, the observed $Z _ { j }$ updates the predictors; on skipped steps, the forecasters extrapolate each coefficient’s trajectory and together form ${ \widehat { Z } } _ { j }$ . The forecast is then transformed back to the spatial domain, $\widehat { Y } _ { j } = \mathrm { I D C T } _ { 2 \mathrm { D } } ( \widehat { Z } _ { j } )$ , and passed directly to the diffusion sampler.

## 4.1.2 BAND-SPECIFIC DENOISING TIME

Each coefficient in SpecKCast is forecast as a function of time, which raises the question of what to use as the time variable. A natural choice is the step index $j ,$ but it does not capture the nonuniform progression of the denoising process. As a result, equal increments of $j$ can correspond to very different amounts of change, making the trajectory harder to forecast. We instead derive the time variable from the noise schedule, expressed as the log signal-to-noise ratio $\lambda _ { j } = 2 \log \bigl ( ( 1 - \sigma _ { j } ) / \sigma _ { j } \bigr )$ However, individual frequency bands still change most during different phases of denoising. A band’s trajectory therefore bends where the band is most active and stays flat where it is not, making it nonlinear in $\lambda _ { j }$ . What the forecaster needs is a time variable whose increments reflect how much each band is expected to evolve. We therefore warp $\lambda _ { j }$ separately for each band $\ell ,$

$$
t _ { \ell } ( j ) = \mathrm { s i g m o i d } \Big ( { \frac { \lambda _ { j } + \log p _ { \ell } } { T _ { w } } } \Big ) , \qquad s _ { \ell } ( j ) = \mathrm { r e s c a l e } _ { [ - 1 , 1 ] } \big ( t _ { \ell } ( j ) \big ) ,\tag{8}
$$

where $p _ { \ell }$ is the calibrated signal power of band ℓ (Section B.4) and $T _ { w }$ sets the width of each band’s active window. The rescaling maps the first and last values of $t _ { \ell } ~ \mathrm { t o } ~ - 1$ and 1, so all bands share the same range. The resulting coordinate advances more when the band is expected to evolve and less when it is stable. This transformation only changes the forecasting coordinate and leaves the diffusion process unchanged.

## 4.2 SPECTRAL KALMAN FORECASTING

SpecKCast uses a Kalman filter to track each frequency coefficient with a stepwise discrete dynamics model. The state of coefficient i at step j is $h _ { i , j } = ( m _ { i , j } , \nu _ { i , j } )$ , where $m _ { i , j }$ is its estimated value and $\nu _ { i , j }$ is its temporal derivative. Each coefficient maintains its own state, while the time coordinate $s _ { \ell } .$ covariance, and process-noise statistics are shared within each band. Over each adjacent time increment $\Delta _ { \ell , j } = s _ { \ell } ( j ) - s _ { \ell } ( j - 1 )$ , the state evolves

$$
m _ { i , j } = m _ { i , j - 1 } + \Delta _ { \ell , j } \nu _ { i , j - 1 } , \qquad \nu _ { i , j } = \rho _ { \ell , j } \nu _ { i , j - 1 } , \qquad \rho _ { \ell , j } = e ^ { - \Delta _ { \ell , j } / \tau _ { \ell } } ,\tag{9}
$$

where $\tau _ { \ell }$ controls how quickly the temporal derivative decays. This prediction is applied at every sampler step, including skipped ones, while the Kalman update is applied only at computed steps.

The value $m _ { i , j }$ advances according to the temporal derivative $\nu _ { i , j }$ , which gradually decays. This damping reduces the influence of outdated temporal changes and limits overshooting over extended skips. To incorporate these dynamics into the Kalman filter, we define the transition matrix $A _ { \ell , j }$ and process-noise covariance $Q _ { \ell , j }$ as

$$
A _ { \ell , j } = \left[ \mathbb { 1 } \quad \Delta _ { \ell , j } \right] , \qquad Q _ { \ell , j } = q _ { \ell , j } \left[ \Delta _ { \ell , j } ^ { 3 } / 3 \quad \Delta _ { \ell , j } ^ { 2 } / 2 \right] ,\tag{10}
$$

Here, $q _ { \ell , j }$ is the process-noise intensity, calibrated from the temporal activity of band ℓ and adapted during inference using observed forecast errors. The construction of $A _ { \ell , j }$ and $Q _ { \ell , j }$ is detailed in Section C. The covariance shared by coefficients in band ℓ evolves according to

$$
P _ { \ell , j } = A _ { \ell , j } P _ { \ell , j - 1 } A _ { \ell , j } ^ { \top } + Q _ { \ell , j } .\tag{11}
$$

Initially, we set the state as $m _ { i , 0 } = Z _ { i , 0 }$ and $\nu _ { i , 0 } = 0$ . On compute steps, the observed coefficient $Z _ { i , j }$ corrects the predicted state through the Kalman update. On skipped steps, the predicted coefficient values $m _ { i , j }$ form the spectral forecast $\widehat { Z } _ { j }$ , which is transformed back to the spatial domain and passed to the sampler. Additionally, a separate covariance tracker estimates how each band’s uncertainty grows across skipped steps. A compute step is triggered when the combined uncertainty exceeds a threshold.

## 4.3 UNCERTAINTY-AWARE SCHEDULING

The remaining question is which steps to compute. A fixed schedule is the simplest choice, but it cannot respond to changes in forecasting difficulty across generations and denoising phases. SpecKCast instead schedules adaptively, computing a step when the uncertainty accumulated by the Kalman filter indicates that the forecast is becoming unreliable. This requires combining the per-band covariances into a single decision quantity. Coefficients within the same band share the covariance $P _ { \ell , j }$ because they use the same time coordinate and process-noise model. This avoids maintaining a separate covariance for every coefficient and reduces uncertainty tracking to only L matrices. We therefore aggregate the per-band uncertainties into a scalar risk,

$$
R ( j ) = \omega ( j ) \sum _ { \ell = 1 } ^ { L } d _ { \ell } P _ { 1 1 } ^ { ( \ell ) } ( j ) , \qquad \omega ( j ) = \exp \left( - \beta \frac { j } { N - 1 } \right) ,\tag{12}
$$

where $d _ { \ell }$ is the number of coefficients in band ℓ and $\beta$ controls how strongly late-stage risk is attenuated. Here, $P _ { 1 1 } ^ { ( \ell ) } ( j )$ is the top-left entry of $P _ { \ell , j } { \mathrm { : } }$ , tracked by the separate recursion described in Section D.1. Since the rows and columns of $P _ { \ell , j }$ follow the state ordering $h _ { i , j } = ( m _ { i , j } , \nu _ { i , j } )$ this entry is the predicted variance of the error in the coefficient estimate $m _ { i , j }$ . Multiplying it by $d _ { \ell }$ approximates the total forecast uncertainty of band ℓ. Accordingly, scheduling follows a simple threshold rule. After each compute step, we propagate the covariance forward to predict the risk at future steps. SpecKCast uses forecasted outputs until the first step k satisfying $R ( k ) > \varepsilon ,$ , at which point the transformer is evaluated. A smaller ε triggers more frequent evaluations, while a larger value permits longer skips. We calibrate ε for the desired compute budget, while the covariance adapts to the observed forecast errors during generation.

## 5 EXPERIMENTS

Setup. We evaluate SpecKCast on two state-of-the-art text-to-image models, FLUX.1-dev (Labs, 2024) at 1024 × 1024 and Qwen-Image (Wu et al., 2025) at 1328 × 1328, and further on two textto-video models, Wan2.1-14B (Wan et al., 2025) and HunyuanVideo (Kong et al., 2024). Following common practice, we use 50-step sampling as the full-computation reference for all models. We compare against FORA (Selvaraju et al., 2024), TeaCache (Liu et al., 2025a), HiCache (Feng et al., 2026), SeaCache (Chung et al., 2026), TaylorSeer (Liu et al., 2025b), and Spectrum (Han et al., 2026). Among them, TaylorSeer and Spectrum are the most closely related to our method, as they also forecast future features across timesteps instead of directly reusing cached ones. All baselines are evaluated using their official implementations. We decompose each prediction into $L \ = \ 7$ frequency bands and, unless otherwise stated, set $\tau = 0 . 1 , T _ { w } = 2 . 0$ , and $\beta = 0 . 6$ . For models that use classifier-free guidance (Ho & Salimans, 2022), SpecKCast forecasts the guided velocity (Section E). All experiments are conducted on NVIDIA H100 GPUs.

Table 1: Quantitative comparison in text-to-image generation on FLUX.1-dev and Qwen-Image.
<table><tr><td rowspan="3" colspan="2"></td><td colspan="8">FLUX.1-dev (1024×1024) Image</td><td colspan="8">Qwen-Image (1328×1328)</td><td rowspan="3"></td></tr><tr><td colspan="2">Acceleration</td><td rowspan="2">Mem. ∆ (GiB)↓</td><td rowspan="2"></td><td colspan="3">Quality</td><td rowspan="2">CLIP↑</td><td colspan="2">Acceleration</td><td rowspan="2">Mem. ∆ (GiB)↓</td><td colspan="3">Quality</td><td rowspan="2" colspan="2">Image Reward↑</td></tr><tr><td colspan="2">Latency(s)↓ Speedup↑</td><td colspan="2">PSNR↑ SSIM↑ LPIPS↓</td><td colspan="2">Reward↑</td><td colspan="2">|Latency(s)↓ Speedup↑</td><td colspan="2">PSNR↑ SSIM↑ LPIPS↓</td></tr><tr><td colspan="2">50 Reference</td><td>11.68</td><td>1.00</td><td></td><td></td><td></td><td></td><td>0.98</td><td>27.20</td><td>52.22</td><td>1.00</td><td>一</td><td>一</td><td></td><td></td><td>1.24</td><td>29.14</td></tr><tr><td rowspan="8">≈ 14</td><td>Base</td><td>3.35</td><td>3.48</td><td>0.00</td><td>15.30</td><td>0.633</td><td>0.423</td><td>0.98</td><td>27.51</td><td>14.83</td><td>3.52</td><td>0.00</td><td>15.18</td><td>0.640</td><td>0.393</td><td>1.08</td><td>29.19</td></tr><tr><td>FORA</td><td>3.98</td><td>2.93</td><td>2.01</td><td>15.03</td><td>0.628</td><td>0.433</td><td>0.96</td><td>27.48</td><td>16.11</td><td>3.24</td><td>9.46</td><td>14.03</td><td>0.571</td><td>0.456</td><td>1.02</td><td>28.97</td></tr><tr><td>TeaCache</td><td>3.55</td><td>3.29</td><td>0.07</td><td>16.86</td><td>0.684</td><td>0.355</td><td>1.01</td><td>27.57</td><td>15.68</td><td>3.33</td><td>0.20</td><td>19.42</td><td>0.781</td><td>0.229</td><td>1.19</td><td>29.36</td></tr><tr><td>HiCache</td><td>4.42</td><td>2.64</td><td>4.02</td><td>18.02</td><td>0.708</td><td>0.292</td><td>1.00</td><td>27.50</td><td>19.16</td><td>2.72</td><td>18.97</td><td>16.04</td><td>0.647</td><td>0.335</td><td>1.16</td><td>29.08</td></tr><tr><td>SeaCache</td><td>3.59</td><td>3.25</td><td>0.09</td><td>22.30</td><td>0.813</td><td>0.184</td><td>0.98</td><td>27.40</td><td>15.78</td><td>3.31</td><td>0.24</td><td>21.43</td><td>0.821</td><td>0.190</td><td>1.20</td><td>29.11</td></tr><tr><td>TaylorSeer</td><td>4.59</td><td>2.55</td><td>4.02</td><td>17.74</td><td>0.700</td><td>0.305</td><td>1.00</td><td>27.32</td><td>18.37</td><td>2.84</td><td>18.97</td><td>14.22</td><td>0.598</td><td>0.423</td><td>1.05</td><td>28.58</td></tr><tr><td>Spectrum</td><td>3.49</td><td>3.36</td><td>1.15</td><td>23.60</td><td>0.814</td><td>0.165</td><td>0.98</td><td>27.33</td><td>15.72</td><td>3.32</td><td>2.84</td><td>23.66</td><td>0.851</td><td>0.135</td><td>1.21</td><td>29.16</td></tr><tr><td>SpecKCast</td><td>3.45</td><td>3.38</td><td>0.005</td><td>24.28</td><td>0.836</td><td>0.146</td><td>0.98</td><td>27.32</td><td>15.01</td><td>3.48</td><td>0.01</td><td>24.16</td><td>0.852</td><td>0.134</td><td>1.22</td><td>29.17</td></tr><tr><td rowspan="9">≈ 10</td><td>Base</td><td>2.48</td><td>4.71</td><td>0.00</td><td>14.55</td><td>0.599</td><td>0.483</td><td>0.87</td><td>27.31</td><td>10.60</td><td>4.93</td><td>0.00</td><td>14.03</td><td>0.586</td><td>0.480</td><td>0.95</td><td>28.89</td></tr><tr><td>FORA</td><td>3.16</td><td>3.70</td><td>2.01</td><td>14.32</td><td>0.591</td><td>0.502</td><td>0.84</td><td>27.40</td><td>12.16</td><td>4.29</td><td>9.46</td><td>12.73</td><td>0.502</td><td>0.561</td><td>0.65</td><td>28.15</td></tr><tr><td>TeaCache</td><td>2.56</td><td>4.57</td><td>0.07</td><td>15.66</td><td>0.631</td><td>0.440</td><td>0.97</td><td>27.50</td><td>11.46</td><td>4.56</td><td>0.20</td><td>16.11</td><td>0.656</td><td>0.390</td><td>1.06</td><td>28.84</td></tr><tr><td>HiCache</td><td>3.62</td><td>3.23</td><td>4.02</td><td>16.30</td><td>0.635</td><td>0.390</td><td>0.99</td><td>27.60</td><td>15.41</td><td>3.39</td><td>18.97</td><td>12.77</td><td>0.488</td><td>0.519</td><td>0.88</td><td>28.02</td></tr><tr><td>SeaCache</td><td>2.55</td><td>4.59</td><td>0.09</td><td>19.29</td><td>0.731</td><td>0.304</td><td>0.97</td><td>27.52</td><td>11.75</td><td>4.45</td><td>0.24</td><td>19.61</td><td>0.767</td><td>0.257</td><td>1.15</td><td>29.08</td></tr><tr><td>TaylorSeer</td><td>3.55</td><td>3.29</td><td>4.02</td><td>15.28</td><td>0.607</td><td>0.436</td><td>0.97</td><td>27.37</td><td>14.40</td><td>3.63</td><td>18.97</td><td>10.79</td><td>0.405</td><td>0.623</td><td>0.62</td><td>26.85</td></tr><tr><td>Spectrum</td><td>2.55</td><td>4.59</td><td>0.87</td><td>21.27</td><td>0.729</td><td>0.261</td><td>0.98</td><td>27.62</td><td>11.53</td><td>4.53</td><td>2.21</td><td>20.85</td><td>0.746</td><td>0.318</td><td>1.17</td><td>29.80</td></tr><tr><td>SpecKCast</td><td>2.46</td><td>4.74</td><td>0.005</td><td>21.77</td><td>0.749</td><td>0.258</td><td>0.98</td><td>27.57</td><td>10.80</td><td>4.83</td><td>0.01</td><td>22.28</td><td>0.790</td><td>0.201</td><td>1.19</td><td>29.01</td></tr><tr><td>Base FORA</td><td>1.98</td><td>5.89</td><td>0.00</td><td>14.29</td><td>0.578</td><td>0.518</td><td>0.81</td><td>27.39</td><td>8.54</td><td>6.11</td><td>0.00</td><td>13.53</td><td>0.561</td><td>0.531</td><td></td><td>28.78</td></tr><tr><td rowspan="8">≈8 TaylorSeer Spectrum</td><td></td><td>2.70</td><td>4.33</td><td>2.01</td><td>14.12</td><td>0.574</td><td>0.529</td><td>0.74</td><td>27.07</td><td>10.15</td><td>5.15</td><td>9.46</td><td>12.39</td><td>0.475</td><td>0.589</td><td>0.80 0.40</td><td>27.76</td></tr><tr><td></td><td>2.13</td><td>5.75</td><td>0.07</td><td>15.07</td><td>0.600</td><td>0.506</td><td>0.82</td><td>27.48</td><td></td><td>4.54</td><td>0.20</td><td>16.00</td><td>0.648</td><td></td><td>1.04</td><td>28.83</td></tr><tr><td>TeaCache</td><td>3.23</td><td></td><td>4.02</td><td>15.13</td><td>0.582</td><td>0.466</td><td>0.98</td><td>27.53</td><td>11.50 13.50</td><td>3.87</td><td>18.97</td><td>10.86</td><td>0.391</td><td>0.397 0.639</td><td>0.56</td><td>27.21</td></tr><tr><td>HiCache SeaCache</td><td>2.13</td><td>3.48 5.48</td></table>

![](images/a805643db3b86263e19b9ec815b0c2e8d5b4c66e2ba28a6aedec0d8f0f78d212.jpg)  
Figure 3: Qualitative comparison on text-to-image generation using FLUX.1-dev.

Evaluation. For text-to-image generation, we use the DrawBench (Saharia et al., 2022) prompts and report ImageReward (Xu et al., 2023) and CLIP Score (Hessel et al., 2021), which measures text–image alignment. For text-to-video generation, we follow the standard prompts and protocol of VBench (Huang et al., 2024) and report its Quality Score, which aggregates dimensions such as temporal consistency, motion smoothness, and per-frame aesthetic and imaging quality. Since an accelerated sampler should reproduce the output of the full model rather than merely produce a plausible sample, we also compare each accelerated sample with its 50-step reference generated from the same prompt and seed. We report PSNR for pixel-level error, SSIM (Wang et al., 2004) for structural similarity, and LPIPS (Zhang et al., 2018) for perceptual distance. For efficiency, we report sampling latency, speedup over the 50-step reference, and memory overhead (Mem. ∆), defined as the peak allocated GPU memory above that of Base at the same NFE.

## 5.1 TEXT-TO-IMAGE GENERATION

Table 1 reports results on FLUX.1-dev and Qwen-Image, where Base runs the original sampler with fewer steps to match the NFE. SpecKCast achieves the highest speedup among acceleration methods in all but one setting, along with the best PSNR and LPIPS in all settings and the best SSIM in all but one. At the highest acceleration, it outperforms the strongest baseline by 1.1 dB PSNR on FLUX.1- dev (5.54×) and 2.0 dB on Qwen-Image (6.00×). Spectrum, the closest competitor at 14 and 10 NFE, trails SpecKCast by 0.5–1.4 dB PSNR and degrades sharply on Qwen-Image at 8 NFE, where its LPIPS rises to 0.527 compared with 0.267 for SpecKCast. Compared with Base, which runs at a similar speed, SpecKCast improves PSNR by 5.5–9.0 dB, showing that it tracks the 50-step trajectory far more closely than simply taking fewer steps. These gains cost at most 0.01 GiB of memory, whereas TaylorSeer, HiCache, and Spectrum add 0.73–18.97 GiB. Figures 3 and 5 show qualitative results on FLUX.1-dev and Qwen-Image, respectively. TaylorSeer and HiCache blur the images, TeaCache and SeaCache alter textures and lighting, and every baseline misspells or distorts the text on the sign. SpecKCast stays closest to the 50-step reference while running fastest. More samples are provided in Section F.

Table 2: Quantitative comparison in text-to-video generation on Wan2.1-14B and HunyuanVideo.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="6">Wan2.1-14B Mem. ∆</td><td rowspan="2"></td><td colspan="7">HunyuanVideo</td><td rowspan="2">VBench</td></tr><tr><td>Acceleration</td><td colspan="2"></td><td></td><td colspan="2">Quality</td><td>VBench</td><td colspan="2">Acceleration</td><td>Mem. ∆</td><td></td><td colspan="2">Quality</td></tr><tr><td colspan="2">NFE</td><td>Method</td><td>Latency(s)↓ Speedup↑</td><td>(GiB)↓</td><td>PSNR↑</td><td></td><td>SSIM↑</td><td>LPIPS↓</td><td>Quality↑</td><td>| Latency(s).↓</td><td>Speedup↑</td><td>(GiB)↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Quality↑</td></tr><tr><td rowspan="8">50</td><td>Reference</td><td>370</td><td>1.00</td><td>一</td><td>一</td><td>一</td><td>一</td><td>81.11</td><td>183 58.3</td><td>1.00</td><td>一</td><td></td><td></td><td>一</td><td>一</td><td>80.26</td></tr><tr><td>Base</td><td>116</td><td>3.20</td><td>0.00</td><td>13.95</td><td>0.471</td><td>0.518</td><td>79.81</td><td></td><td></td><td>3.14</td><td>0.00</td><td>17.33</td><td>0.641</td><td>0.417</td><td>78.38</td></tr><tr><td>FORA</td><td>108.3</td><td>3.42</td><td>0.62</td><td>21.62</td><td>0.753</td><td>0.192</td><td>80.63</td><td>56.8</td><td></td><td>3.22</td><td>0.19</td><td>23.46</td><td>0.775</td><td>0.216</td><td>78.90</td></tr><tr><td>TeaCache</td><td>105.2</td><td>3.52</td><td>0.94</td><td>22.13</td><td>0.772</td><td>0.191</td><td>80.24</td><td></td><td>61.2 56.8</td><td>2.99</td><td>0.37</td><td>20.56</td><td>0.709</td><td>0.279</td><td>79.35</td></tr><tr><td>HiCache</td><td>111.9</td><td>3.31</td><td>3.75</td><td>18.42</td><td>0.675</td><td>0.267</td><td>80.12</td><td></td><td></td><td>3.22</td><td>1.12</td><td>23.43</td><td>0.785</td><td>0.198</td><td>79.54</td></tr><tr><td>SeaCache</td><td>113.2</td><td>3.27</td><td>4.37</td><td>23.10</td><td>0.795</td><td>0.157</td><td>80.38</td><td></td><td>71.3</td><td>2.57</td><td>0.37</td><td>27.59</td><td>0.864</td><td>0.121</td><td>79.53</td></tr><tr><td>TaylorSeer</td><td>109.4</td><td>3.38</td><td>4.69</td><td>17.94</td><td>0.667</td><td>0.264</td><td>80.12</td><td>57.2</td><td></td><td>3.20</td><td>1.49</td><td>22.29</td><td>0.767</td><td>0.217</td><td>79.98</td></tr><tr><td>Spectrum</td><td>110.3 105.2</td><td>3.35 3.52</td><td>13.12</td><td>22.93</td><td>0.789 0.813</td><td>0.136</td><td>80.76</td><td>58.0</td><td>3.15</td><td></td><td>3.56 0.03</td><td>26.95</td><td>0.831 0.884</td><td>0.133 0.107</td><td>79.32</td></tr><tr><td rowspan="9">≈ 10</td><td>SpecKCast</td><td></td><td></td><td>0.05</td><td>24.47</td><td></td><td>0.140</td><td>80.86</td><td>55.0</td><td>3.33</td><td></td><td></td><td>29.72</td><td></td><td></td><td>79.68</td></tr><tr><td>Base</td><td>80.50</td><td>4.59</td><td>0.00</td><td>13.66</td><td>0.452</td><td>0.578</td><td>78.05</td><td>42.7</td><td>4.28</td><td>0.00</td><td></td><td>16.73</td><td>0.621</td><td>0.469</td><td>76.59</td></tr><tr><td>FORA</td><td>79.6</td><td>4.65</td><td>0.62</td><td>20.68</td><td>0.709</td><td>0.270</td><td>79.76</td><td>43.0</td><td>4.26</td><td>0.19</td><td></td><td>22.03</td><td>0.726</td><td>0.306</td><td>77.12</td></tr><tr><td>TeaCache</td><td>80.3</td><td>4.61</td><td>0.94</td><td>21.11</td><td>0.732</td><td>0.245</td><td>79.05</td><td>47.2</td><td>3.88</td><td></td><td>0.37</td><td>19.65</td><td>0.672</td><td>0.329</td><td>78.19</td></tr><tr><td>HiCache</td><td>80.3</td><td>4.61</td><td>3.75</td><td>14.04</td><td>0.519</td><td>0.474</td><td>74.96</td><td>43.0</td><td>4.26</td><td></td><td>1.12</td><td>21.17</td><td>0.718</td><td>0.293</td><td>78.06</td></tr><tr><td>SeaCache</td><td>89.5</td><td>4.13</td><td>4.37</td><td>19.53</td><td>0.686</td><td>0.271</td><td>79.87</td><td>45.4</td><td>4.03</td><td></td><td>0.37</td><td>19.34</td><td>0.698</td><td>0.338</td><td>78.08</td></tr><tr><td>TaylorSeer</td><td>80.2</td><td>4.61</td><td>4.69</td><td>14.57</td><td>0.549</td><td>0.404</td><td>76.20</td><td>43.4</td><td>4.22</td><td></td><td>1.49</td><td>19.54</td><td>0.689</td><td>0.318</td><td>78.19</td></tr><tr><td>Spectrum</td><td>80.8 79.1</td><td>4.58</td><td>9.38</td><td>21.41</td><td>0.733</td><td>0.175</td><td>80.42</td><td>43.8</td><td>4.17</td><td></td><td>2.81</td><td>24.60</td><td>0.767</td><td>0.191</td><td>78.80</td></tr><tr><td>SpecKCast</td><td></td><td>4.68</td><td>0.05</td><td>21.62</td><td>0.750</td><td>0.181</td><td>80.48</td><td>42.7</td><td>4.29</td><td></td><td>0.03</td><td>25.62</td><td>0.821</td><td>0.172</td><td>79.25</td></tr><tr><td rowspan="9">≈8</td><td>Base</td><td>80.2</td><td>4.61</td><td>0.00</td><td>13.56</td><td>0.449</td><td>0.614</td><td>76.98</td><td>38.2</td><td>4.79</td><td>0.00</td><td></td><td>16.35</td><td>0.612</td><td>0.510</td><td>75.01</td></tr><tr><td>FORA</td><td>65.5</td><td>5.65</td><td>0.62</td><td>20.19</td><td>0.669</td><td>0.336</td><td>78.38</td><td>36.2</td><td>5.06</td><td>0.19</td><td></td><td>21.22</td><td>0.655</td><td>0.346</td><td>75.00</td></tr><tr><td>TeaCache</td><td>65.8</td><td>5.62</td><td>0.94</td><td>19.61</td><td>0.656</td><td>0.335</td><td>76.52</td><td>36.9</td><td>4.96</td><td>0.37</td><td></td><td>18.37</td><td>0.617</td><td>0.415</td><td>76.00</td></tr><tr><td>HiCache</td><td>66.3</td><td>5.58</td><td>3.75</td><td>12.06</td><td>0.430</td><td>0.578</td><td>72.01</td><td>36.2</td><td>5.06</td><td>1.12</td><td></td><td>19.83</td><td>0.643</td><td>0.346</td><td>75.42</td></tr><tr><td>SeaCache</td><td>65.0</td><td>5.69</td><td>4.37</td><td>15.27</td><td>0.519</td><td>0.482</td><td>76.97</td><td>37.0</td><td>4.95</td><td></td><td>0.37</td><td>18.98</td><td>0.681</td><td>0.380</td><td>76.81</td></tr><tr><td>TaylorSeer</td><td>65.7</td><td>5.63</td><td>4.69</td><td>13.02 17.81</td><td>0.480 0.638</td><td>0.475 0.303</td><td>73.57 78.74</td><td>36.5 37.9</td><td>5.01 4.82</td><td></td><td>1.49 2.43</td><td>18.15 21.74</td><td>0.620 0.674</td><td>0.387 0.297</td><td>74.90 75.77</td></tr><tr><td>Spectrum SpecKCast</td><td>66.0 65.3</td><td>5.61 5.67</td><td>7.50 0.05</td></table>

![](images/7066cfcdef6a314c91f8aee549eb517edcea40b54547ac951537141a640872ab.jpg)  
Figure 4: Qualitative comparison on text-to-video generation using HunyuanVideo.

## 5.2 TEXT-TO-VIDEO GENERATION

Table 2 reports results on Wan2.1-14B and HunyuanVideo. SpecKCast achieves the highest or nearhighest speedup in every setting, along with the best SSIM in all settings and the best PSNR and LPIPS in most. On HunyuanVideo at the 14-NFE budget, it improves PSNR by 2.1 dB over Sea-Cache, the strongest baseline, while running at 3.33× instead of 2.57×. The advantage holds at the 8-NFE budget, where SpecKCast keeps the best SSIM and LPIPS on both models, while HiCache and TaylorSeer drop to 12.06 and 13.02 dB PSNR on Wan2.1-14B. SpecKCast also achieves the best VBench Quality on Wan2.1-14B at every budget, staying within 0.25 points of the reference at the 14-NFE budget. Spectrum obtains slightly lower LPIPS on Wan2.1-14B at the 14- and 10-NFE budgets, but requires up to 13.12 GiB of extra memory, compared with 0.05 GiB for SpecKCast. Figures 4 and 6 show qualitative results on HunyuanVideo and Wan2.1-14B, respectively. On HunyuanVideo, most baselines replace the puffin in the first frame with a blurred or different bird, and SeaCache adds a second cat in later frames. On Wan2.1-14B, the baselines introduce local artifacts (red boxes), mostly in the window background, and HiCache also shifts the lighting. SpecKCast stays closest to the 50-step reference across frames on both models.

Table 3: Ablation of SpecKCast components on FLUX.1-dev at 14 NFE.
<table><tr><td>Forecaster</td><td>Scheduler Time variable</td><td></td><td>Frequency decomposition</td><td>Latency(s)↓ PSNR↑</td><td></td><td>SSIM↑</td><td>LPIPS↓</td><td>Image Reward↑</td><td>CLIP↑</td></tr><tr><td>Taylor</td><td>fixed</td><td>warped-logSNR</td><td>√</td><td>3.46</td><td>21.90</td><td>0.811</td><td>0.171</td><td>0.978</td><td>27.24</td></tr><tr><td>Chebyshev</td><td>fixed</td><td>warped-logSNR</td><td>√</td><td>3.97</td><td>21.10</td><td>0.761</td><td>0.252</td><td>0.975</td><td>27.66</td></tr><tr><td>Kalman</td><td>fixed</td><td>warped-logSNR</td><td>V</td><td>3.43</td><td>22.09</td><td>0.810</td><td>0.178</td><td>0.973</td><td>27.27</td></tr><tr><td>Kalman</td><td>adaptive</td><td>raw-logSNR</td><td>√</td><td>3.52</td><td>21.03</td><td>0.775</td><td>0.210</td><td>0.977</td><td>27.24</td></tr><tr><td>Kalman</td><td>adaptive</td><td>uniform step index</td><td>√</td><td>3.51</td><td>19.54</td><td>0.740</td><td>0.256</td><td>0.969</td><td>27.28</td></tr><tr><td>Kalman</td><td>adaptive</td><td>warped-logSNR</td><td>X</td><td>3.48</td><td>21.19</td><td>0.785</td><td>0.201</td><td>0.966</td><td>27.13</td></tr><tr><td>Kalman</td><td>adaptive</td><td>warped-logSNR</td><td>√</td><td>3.45</td><td>24.28</td><td>0.836</td><td>0.146</td><td>0.980</td><td>27.32</td></tr></table>

![](images/0ba2098276c7dcf5f3d6d50ec15218b2b3be87e0bf9da52d03ca3fc6d95bcd86.jpg)  
Figure 5: Qualitative comparison on text-to-image generation using Qwen-Image.

![](images/edae89f1c2e4e0275f42b3e8da96344b221285dd74dd0d09b7dfa093a01fdc6c.jpg)  
Figure 6: Qualitative comparison on text-to-video generation using Wan2.1-14B.

## 5.3 ABLATION STUDIES

Table 3 ablates the components of SpecKCast on FLUX.1-dev at 14 NFE. Since uncertainty-aware scheduling relies on the Kalman covariance, we first compare forecasters under a fixed schedule. Kalman and Taylor perform similarly, with Kalman slightly faster and higher in PSNR, while Chebyshev is both slower and less accurate. With our frequency decomposition and time warping, the Taylor forecaster also outperforms TaylorSeer (Table 1) by 4.2 dB PSNR while reducing latency from 4.59 to 3.46 s. Replacing the fixed schedule with uncertainty-aware scheduling (Section 4.3) improves PSNR by 2.2 dB. Band-specific time warping (Section 4.1.2) outperforms the raw log-SNR $\lambda _ { j }$ by 3.3 dB and the step index $j$ by 4.7 dB, and removing the frequency decomposition lowers PSNR by 3.1 dB. Overall, the full model achieves the best PSNR, SSIM, LPIPS, and ImageReward. Since all variants except Chebyshev run within 0.1 s of each other, these gains come at almost no extra latency.

## 6 CONCLUSION

We presented SpecKCast, a training-free method for accelerating Diffusion Transformers by forecasting the denoiser output at skipped sampling steps. Rather than modeling the output uniformly, SpecKCast decomposes it into spatial-frequency components and forecasts their distinct temporal dynamics along band-specific, noise-schedule-aligned time coordinates. A lightweight Kalman filter tracks the value and rate of change of each DCT coefficient, while its uncertainty provides a principled signal for adaptively determining when a full transformer evaluation is needed. Across image and video diffusion models, SpecKCast consistently improves the acceleration–quality trade off over existing caching and forecasting methods, achieving the best fidelity to the 50-step reference in most settings with negligible latency and memory overhead. These results highlight the benefit of explicitly accounting for the distinct temporal evolution of frequency components when forecasting diffusion trajectories. SpecKCast, however, targets only temporal redundancy across denoising steps; each computed step still processes the full token grid and therefore retains the spatial redundancy within a transformer evaluation. Jointly exploiting temporal and spatial redundancy could enable further acceleration and represents a promising direction for future work.

## REFERENCES

Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal ofMachine Learning Research, 26(209):1–80, 2025.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pp. 74–91. Springer, 2024a.

Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. ∆-DiT: A training-free acceleration method tailored for diffusion transformers. arXiv preprint arXiv:2406.01125, 2024b.

Jiwoo Chung, Sangeek Hyun, MinKyu Lee, Byeongju Han, Geonho Cha, Dongyoon Wee, Youngjun Hong, and Jae-Pil Heo. Seacache: Spectral-evolution-aware cache for accelerating diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14283–14294, 2026.

Liang Feng, Shikang Zheng, Jiacheng Liu, Yuqi Lin, Qinming Zhou, Peiliang Cai, Xinyu Wang, Junjie Chen, Chang Zou, Yue Ma, et al. Hicache: A plug-in scaled-hermite upgrade for taylor-style cache-then-forecast diffusion acceleration. In International Conference on Learning Representations, volume 2026, pp. 112361–112386, 2026.

Huanlin Gao, Ping Chen, Fuyuan Shi, Ruijia Wu, Li YanTao, Qiang Hui, Yuren You, Ting Lu, Chao Tan, Shaoan Zhao, Zhaoxiang Liu, Fang Zhao, Kai Wang, and Shiguo Lian. Meancache: From instantaneous to average velocity for accelerating flow matching inference. In International Conference on Learning Representations, volume 2026, pp. 60466–60485, 2026.

Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, and Stefano Ermon. Adaptive spectral feature forecasting for diffusion sampling acceleration. arXiv preprint arXiv:2603.01623, 2026.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Longfei Huang, Yu Liang, Hao Zhang, Jinwei Chen, Wei Dong, Lunde Chen, Wanyu Liu, Bo Li, and Peng-Tao Jiang. Sdmatte: Grafting diffusion models for interactive matting. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15229–15239. IEEE, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21807–21818. IEEE, 2024.

R. E. Kalman. A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1):35–45, 1960. doi: 10.1115/1.3662552.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17489–17499. IEEE, 2023.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7353–7363. IEEE, 2025a.

Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15853–15863. IEEE, 2025b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deepcache: Accelerating diffusion models for free. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15762–15772. IEEE, 2024.

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14297–14306. IEEE, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. IEEE, 2023.

Xurui Peng, Chenqian Yan, Hong Liu, Rui Ma, Fangmin Chen, Xing Wang, Zhihua Wu, Songwei Liu, and Mingbao Lin. Ertacache: Error rectification and timesteps adjustment for efficient diffu sion. In International Conference on Learning Representations, volume 2026, pp. 82104–82120, 2026.

Javad Rajabi, Kimia Shaban, Koorosh Roohi, David B Lindell, and Babak Taati. Sega: Spectralenergy guided attention for resolution extrapolation in diffusion transformers. arXiv preprint arXiv:2605.22668, 2026.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Pratheba Selvaraju, Tianyu Ding, Tianyi Chen, Ilya Zharkov, and Luming Liang. Fora: Fast-forward caching in diffusion transformer acceleration. arXiv preprint arXiv:2407.01425, 2024.

Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In International Conference on Learning Representations, volume 2024, pp. 15078–15097, 2024.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, and Bin Cui. Trainingfree and adaptive sparse attention for efficient long video generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15982–15993. IEEE, 2025.

Chenyang Xu, Dezhen Wang, Lin Chen, Kepeng Lin, and Hao Wang. Eigencache: Rethinking diffusion acceleration as covariance-optimal forecasting and submodular information allocation. In Forty-third International Conference on Machine Learning, 2026.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pp. 83048–83077, 2025.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and¨ William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, ¨ and Taesung Park. One-step diffusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6613–6623. IEEE, 2024b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Ruihang Zhang, Felix Taubner, Pooja Ravi, Kiriakos N Kutulakos, and David B Lindell. Proxypose: 6-dof pose tracking via video-to-video translation. arXiv preprint arXiv:2607.06555, 2026.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. arXiv preprint arXiv:2410.05317, 2024.

## APPENDIX

## A KALMAN FILTER

The Kalman filter (Kalman, 1960) is the exact Bayesian estimator for a linear-Gaussian state-space model. A hidden state $h _ { j }$ evolves linearly and is observed with noise,

$$
h _ { j } = A _ { j } h _ { j - 1 } + w _ { j } , \quad w _ { j } \sim \mathcal { N } ( 0 , Q _ { j } ) , \qquad y _ { j } = H _ { j } h _ { j } + \eta _ { j } , \quad \eta _ { j } \sim \mathcal { N } ( 0 , R _ { j } ) .\tag{13}
$$

The filter represents its estimate of $h _ { j }$ as a Gaussian with mean $\hat { h } _ { j }$ and covariance $P _ { j }$ , and alternates between two operations. Between observations, it predicts the next state, $\hat { h } _ { j }  A _ { j } \hat { h } _ { j - 1 }$ and $P _ { j } $ $A _ { j } P _ { j - 1 } A _ { j } ^ { \top } + Q _ { j }$ , so the estimate moves forward and its uncertainty grows. When an observation is available, it updates the prediction using the gain $K _ { j } = P _ { j } H _ { j } ^ { \top } ( H _ { j } P _ { j } H _ { j } ^ { \top } + R _ { j } ) ^ { - 1 }$

$$
\hat { h } _ { j }  \hat { h } _ { j } + K _ { j } \underbrace { ( y _ { j } - H _ { j } \hat { h } _ { j } ) } _ { \mathrm { p r e d i c t i o n \ : r e s i d u a l } } , \qquad P _ { j }  ( I - K _ { j } H _ { j } ) P _ { j } .\tag{14}
$$

The prediction residual is the part of the observation that the prediction did not account for. Under the model, its variance is $H _ { j } { \bf \dot { P } } _ { j } H _ { j } ^ { \top } + R _ { j }$ . Comparing the observed residuals with this variance shows whether the assumed noise levels are too low or too high, which allows the filter to adjust them during inference.

For a smooth signal whose rate of change is unknown and varies slowly, a common choice of state is the local linear trend, $h = ( m , \nu ) ^ { \top }$ , where m is the value and ν is its rate of change. The value integrates the rate, and the rate is perturbed by white noise. Both the transition and the process noise then depend only on the time $\Delta$ elapsed between steps,

$$
A ( \Delta ) = \left( { \begin{array} { c c } { 1 } & { \Delta } \\ { 0 } & { 1 } \end{array} } \right) , \qquad Q ( \Delta ) = q \left( { \begin{array} { c c } { \Delta ^ { 3 } / 3 } & { \Delta ^ { 2 } / 2 } \\ { \Delta ^ { 2 } / 2 } & { \Delta } \end{array} } \right) ,\tag{15}
$$

where $Q ( \Delta )$ is the covariance accumulated when the rate is perturbed by white noise of intensity $q .$ SpecKCast uses this model with an additional damping factor on the rate (Section 4).

This model has three properties that make it suitable for the forecasting problem in Eq. (2). First, the predicted value is the minimum-mean-squared-error estimate under the model and depends on all past observations. When the rate is poorly determined, the gain reduces its influence on the forecast. A finite-difference forecaster instead relies only on the most recent observations and amplifies their noise. Second, the filter handles irregularly spaced observations directly, because $\Delta$ enters $A$ and $Q .$ Fixed-order difference formulas assume uniform spacing, which an adaptive compute schedule does not provide. Third, the predicted covariance $P$ is available before the next observation, so the filter estimates its own forecast error. This lets us choose the compute steps U by thresholding the predicted error rather than with a hand-designed schedule.

The Kalman filter is related to the Wiener filter, which is the optimal linear filter for a stationary $\mathrm { s i g . }$ nal in stationary noise and is defined by a fixed frequency response. The Kalman filter is its recursive counterpart for non-stationary signals: it tracks an evolving state together with its uncertainty. The filter itself is standard. Our contribution is the state-space model we place on the denoiser output and the way its covariance is calibrated and used for scheduling (Section 4).

## B DETAILS OF THE BAND-SPECIFIC TIME WARP

This appendix expands on the time variable introduced in Section 4.1.2. We derive the interpretation of $\lambda _ { j } + \log p _ { \ell }$ as the band’s own signal-to-noise ratio, describe the mechanics of the warp, and collect the implementation details omitted from the main text.

## B.1 THE WARPED QUANTITY IS THE BAND’S OWN LOG-SNR

Recall that a noisy latent is a linear interpolation between data and noise, $x _ { \sigma } = \left( 1 - \sigma \right) x _ { 0 } + \sigma \epsilon$ ϵ with $\epsilon \sim \mathcal { N } ( 0 , I )$ . Because the DCT is orthonormal, transforming white Gaussian noise yields white Gaussian noise, so every frequency coefficient of the noise term has unit variance and the noise contributes power $\sigma ^ { 2 }$ uniformly to every band. The data term is not uniform. Latents of natural images concentrate energy at low frequencies, so the coefficients of band ℓ carry an average signal power $p _ { \ell }$ that decreases with frequency. The signal-to-noise ratio of band ℓ at noise level $\sigma _ { j }$ is therefore approximately

$$
\begin{array} { r } { \mathrm { S N R } _ { \ell } ( \sigma _ { j } ) = \frac { ( 1 - \sigma _ { j } ) ^ { 2 } p _ { \ell } } { \sigma _ { j } ^ { 2 } } \qquad \Longrightarrow \qquad \log \mathrm { S N R } _ { \ell } ( \sigma _ { j } ) = \underbrace { 2 \log \bigl ( \frac { 1 - \sigma _ { j } } { \sigma _ { j } } \bigr ) } _ { \lambda _ { j } } + \log p _ { \ell } . } \end{array}\tag{16}
$$

The sigmoid in Eq. (8) is therefore applied to the band’s own log-SNR, scaled by $1 / T _ { w } .$ . Its argument crosses zero when $\mathrm { S N R } _ { \ell } = 1$ , the point at which the band’s signal emerges from the noise. Since $p _ { \ell }$ is large for low bands and small for high bands, low bands cross this threshold early in denoising (at large $\sigma _ { j } )$ and high bands cross it late, which explains why different bands do most of their changing in different phases.

## B.2 MECHANICS OF THE WARP

The sigmoid maps the band’s scaled log-SNR to (0, 1), and its derivative is largest at the zero crossing and decays symmetrically away from it. Consequently, most of the increase in the warped time t<sub>ℓ</sub>(j) = sigmoid $( ( \lambda _ { j } + \log { \dot { p _ { \ell } } } ) / T _ { w } ^ { \bullet } )$ occurs during the steps around the band’s emergence, and $t _ { \ell }$ barely moves while the band is still dominated by noise or has already settled. The temperature $T _ { w }$ sets the width of this active window in log-SNR units; a small $T _ { w }$ concentrates the time variable tightly around the crossing, while a large $T _ { w }$ approaches a plain rescaling of $\lambda _ { j }$ . We use $T _ { w } = 2$ for all bands and all models.

Since $\sigma _ { j }$ decreases monotonically over sampling, $\lambda _ { j }$ increases monotonically, and both the sigmoid and the final rescaling are monotone maps. Each $s _ { \ell }$ is therefore a strictly increasing function of the step index, a requirement for a valid time variable. The operator resca $\mathrm { l e } _ { [ - 1 , 1 ] }$ is the affine map taking the smallest and largest values of $t _ { \ell }$ over the N steps of the run $\mathbf { t o } - 1$ and 1, respectively. Since $t _ { \ell }$ is increasing, these are its first and last values, so

$$
s _ { \ell } ( j ) = 2 \frac { t _ { \ell } ( j ) - t _ { \ell } ( 0 ) } { t _ { \ell } ( N - 1 ) - t _ { \ell } ( 0 ) } - 1 .\tag{17}
$$

Every band’s time variable therefore spans the same interval regardless of where its active window lies.

The warp interacts with the forecaster through the increments $\Delta _ { \ell , j } = s _ { \ell } ( j ) - s _ { \ell } ( j - 1 )$ , which serve as the effective step sizes of the per-band Kalman filters. During a band’s active window the increments are large, so the filter expects more change per step, injects more process noise, and damps its trend estimate faster; outside the window the increments are small and the filter treats the band as nearly static. The warp thus lets a single, simple dynamics model adapt its behavior to each band and phase without any per-step logic.

## B.3 IMPLEMENTATION DETAILS

The following details matter for reproducing the coordinate exactly. First, rectified-flow pipelines typically apply a shift to the noise schedule, which for some models depends on the resolution, so the realized $\sigma _ { j }$ differ from the nominal linearly spaced values. SpecKCast rebuilds the exact schedule the pipeline will use before sampling begins and computes all $s _ { \ell } ( j )$ from it, so the forecaster’s time coordinates line up with the transformer calls at any resolution and step count. Second, the endpoints of the schedule are clamped away from $\sigma = 0$ and $\sigma = 1$ before evaluating $\lambda _ { j }$ to avoid infinities. The clamping changes only the first and last values of $\lambda _ { j }$ , where the sigmoid is nearly flat, so its effect on $s _ { \ell }$ is small. Third, the band powers $p _ { \ell }$ are measured once during calibration (Section B.4) from the implied clean latent $\hat { x } _ { 0 } = x _ { j } - \sigma _ { j } Y _ { j }$ over the final steps of the calibration trajectories. Since the warp is a fixed elementwise function of the schedule, the entire coordinate system costs a few scalar operations per run and is computed once before sampling.

## B.4 BAND-POWER CALIBRATION

SpecKCast uses $p _ { \ell }$ to account for differences in signal magnitude across frequency bands. We estimate these values once per model from a small calibration set C by running the unaccelerated model and recording the latent $x _ { j }$ , model output $Y _ { j }$ , and noise level $\sigma _ { j }$ at every denoising step.

Under the rectified-flow interpolation,

$$
x _ { j } = ( 1 - \sigma _ { j } ) x _ { 0 } + \sigma _ { j } \epsilon = x _ { 0 } + \sigma _ { j } ( \epsilon - x _ { 0 } ) .\tag{18}
$$

Because the model predicts the velocity $Y _ { j } \approx \epsilon - x _ { 0 } .$ , it provides the clean-sample estimate

$$
\widehat { x } _ { 0 , j } = x _ { j } - \sigma _ { j } Y _ { j } .\tag{19}
$$

We estimate power using the later denoising steps, where $\sigma _ { j }$ is smaller. In particular, if

$$
\begin{array} { r } { Y _ { j } = ( \epsilon - x _ { 0 } ) + \delta _ { j } , } \end{array}\tag{20}
$$

then

$$
\widehat { x } _ { 0 , j } = x _ { 0 } - \sigma _ { j } \delta _ { j } ,\tag{21}
$$

so velocity-prediction errors have less influence on $\widehat { x } _ { 0 , j }$ near the end of denoising.

We apply the same channel-wise orthonormal DCT used by SpecKCast,

$$
X _ { 0 , j } = \mathrm { D C T } _ { 2 \mathrm { D } } ( \widehat { x } _ { 0 , j } ) .\tag{22}
$$

Let $\Omega _ { \ell }$ contain the spatial-frequency indices belonging to band ℓ. We estimate its signal power by averaging the squared DCT coefficients over calibration samples, selected denoising steps, channels, and frequencies:

$$
p _ { \ell } = \frac { 1 } { | \mathcal { C } | | \mathcal { I } | C | \Omega _ { \ell } | } \sum _ { n \in \mathcal { C } } \sum _ { j \in \mathcal { I } } \sum _ { c = 1 } ^ { C } \sum _ { ( u , v ) \in \Omega _ { \ell } } \left| X _ { 0 , j , c , u , v } ^ { ( n ) } \right| ^ { 2 } ,\tag{23}
$$

where $\mathcal { I }$ denotes the selected denoising steps.

As shown in Section B.1, $\lambda _ { j } + \log p \ell$ is the log-SNR of band ℓ. Thus, $p \ell$ shifts the global log-SNR according to the signal strength of each band, so the resulting coordinate reflects when that band transitions from noise-dominated to signal-dominated.

In our implementation, the calibration uses 12 prompts with 50 denoising steps each, and J contains the final 25 steps of each trajectory. The resulting $p _ { \ell }$ are fixed across prompts during inference. Calibration modifies no model parameters and introduces no additional model evaluations at inference time.

Calibration-free alternatives. Offline calibration is not required to apply SpecKCast. The simplest alternative is to disable the band-specific warp and use rescaled log-SNR for every band,

$$
\begin{array} { r } { s _ { \ell } ( j ) = \mathrm { r e s c a l e } _ { [ - 1 , 1 ] } ( \lambda _ { j } ) . } \end{array}\tag{24}
$$

Forecasting remains valid, but all bands then share the same temporal coordinate. Consequently, elapsed time no longer reflects their different signal-to-noise transitions, removing the band-specific control of extrapolation, damping, and uncertainty. This variant corresponds to the raw-logSNR row of Table 3, where it lowers PSNR by 3.3 dB relative to the band-specific warp.

Alternatively, $p \ell$ can be estimated online from the first K computed outputs,

$$
p _ { \ell } = \frac { 1 } { K d _ { \ell } } \sum _ { j = 0 } ^ { K - 1 } \sum _ { i \in B _ { \ell } } Z _ { i , j } ^ { 2 } ,\tag{25}
$$

where $\boldsymbol { B _ { \ell } }$ contains the coefficients of band ℓ across all channels and $d _ { \ell } = | \boldsymbol { B } _ { \ell } | = C | \Omega _ { \ell } |$ . When these outputs come from the warm-up steps (Section D.2), this approach requires no additional transformer evaluations. However, the warm-up steps occur at the start of sampling, when the input is almost pure noise. The model output at these steps is therefore close to the noise itself, which has the same power in every band. As a result, the estimated $p \ell$ is nearly the same for all bands and does not capture how signal power decreases with frequency. For this reason, all our experiments use the offline calibration described above.

## B.5 RATIONALE FOR BAND-SPECIFIC TIME REPARAMETERIZATION

The denoising process follows one global sequence of steps j and noise levels $\sigma _ { j }$ . SpecKCast does not assign different physical denoising times to different frequency bands. Instead, it represents the same global step using a different increasing coordinate for each band:

$$
s _ { \ell } ( j ) = g _ { \ell } ( \lambda _ { j } ) ,\tag{26}
$$

where $g _ { \ell }$ is the warp associated with band ℓ. Because each $g _ { \ell }$ is strictly increasing, the ordering of the denoising steps is preserved:

$$
j _ { 1 } < j _ { 2 } \quad \Longrightarrow \quad s _ { \ell } ( j _ { 1 } ) < s _ { \ell } ( j _ { 2 } ) .\tag{27}
$$

Thus, every band follows the same denoising direction and reaches the same target step, although the effective distance between two steps can differ across bands. For a skip from step j to step $k ,$ band ℓ experiences the effective time interval

$$
\Delta _ { \ell } ( j , k ) = s _ { \ell } ( k ) - s _ { \ell } ( j ) .\tag{28}
$$

This interval can be large for a band expected to change substantially during the skip and small for a band expected to remain stable. Consequently, the forecaster can make a larger update for an active band and a more conservative prediction for a stable one. This principle is not specific to the Kalman filter. Any forecasting model that uses time as an input, including linear extrapolation and polynomial fitting, can operate in a band-specific coordinate. The warp used by SpecKCast is

$$
g _ { \ell } ( \lambda ) = \mathrm { r e s c a l e } _ { [ - 1 , 1 ] } \left[ \mathrm { s i g m o i d } \left( \frac { \lambda + \log p _ { \ell } } { T _ { w } } \right) \right] .\tag{29}
$$

For $p _ { \ell } > 0$ and $T _ { w } > 0$ , the sigmoid term is strictly increasing in $\lambda ,$ and the affine rescaling preserves this. The rescaling also gives all bands the same endpoints −1 and 1, while their progression between these endpoints can differ.

## C DETAILS OF THE KALMAN FORECASTER

The transition matrix $A _ { \ell , j }$ and process-noise covariance $Q _ { \ell , j }$ serve different purposes. The former describes the expected evolution of a coefficient, while the latter represents deviations from that evolution. For coefficient i in band ℓ, the state is

$$
h _ { i , j } = ( m _ { i , j } , \nu _ { i , j } ) ,\tag{30}
$$

where $m _ { i , j }$ is the coefficient value and $\nu _ { i , j }$ is its temporal derivative. Over the interval $\Delta _ { \ell , j }$ , we advance the value using its derivative and attenuate the derivative by

$$
\rho _ { \ell , j } = \exp ( - \Delta _ { \ell , j } / \tau _ { \ell } ) .\tag{31}
$$

These operations give

$$
A _ { \ell , j } = \left[ { \begin{array} { c c } { 1 } & { \Delta _ { \ell , j } } \\ { 0 } & { \rho _ { \ell , j } } \end{array} } \right] .\tag{32}
$$

The upper-right entry integrates the derivative over the interval, while the lower-right entry prevents old changes from being extrapolated indefinitely. As the interval grows, $\rho _ { \ell , j }$ decreases and the prediction relies less on the previous derivative.

The deterministic transition cannot capture all changes in the coefficient trajectory. We model these deviations as continuous random perturbations to the temporal derivative. If their intensity is $q _ { \ell , j }$ and the damping is ignored, the accumulated state perturbation over an interval $\Delta _ { \ell , j }$ has covariance

$$
Q _ { \ell , j } = q _ { \ell , j } \int _ { 0 } ^ { \Delta _ { \ell , j } } \left[ ( \Delta _ { \ell , j } - u ) ^ { 2 } \quad \Delta _ { \ell , j } - u \right] d u ,\tag{33}
$$

which evaluates to

$$
Q _ { \ell , j } = q _ { \ell , j } \left[ \begin{array} { c c } { { \Delta _ { \ell , j } ^ { 3 } / 3 } } & { { \Delta _ { \ell , j } ^ { 2 } / 2 } } \\ { { \Delta _ { \ell , j } ^ { 2 } / 2 } } & { { \Delta _ { \ell , j } } } \end{array} \right] .\tag{34}
$$

The lower-right term describes uncertainty accumulated in the temporal derivative. Integrating that uncertainty produces the cross-covariance terms and the cubic growth of uncertainty in the coefficient value. Consequently, longer forecast intervals result in larger covariance.

The scalar $q _ { \ell , j }$ controls the magnitude of this uncertainty. Bands with greater temporal activity receive larger process intensity, allowing their observations to produce stronger corrections. Since coefficients in the same band share $\Delta _ { \ell , j } , \rho _ { \ell , j } , q _ { \ell , j }$ , and the observation noise, they also share $A _ { \ell , j }$ $Q _ { \ell , j }$ , and the resulting Kalman covariance, while retaining separate state estimates.

This construction has intuitive limiting behavior. As $\Delta _ { \ell , j } \to 0 , A _ { \ell , j } \to I$ and $Q _ { \ell , j } \to 0 ;$ , since little evolution or uncertainty is expected over a negligible interval. For longer intervals, the derivative is increasingly attenuated while process uncertainty grows.

This transition is a stepwise discrete approximation rather than the exact discretization of a continuous damped system. It first advances the coefficient using the derivative at the beginning of each interval and then attenuates that derivative. Because the prediction is applied successively over all intervening denoising steps, the damping compounds across a multi-step skip.

Bounded accumulation across skipped steps. The damping acts through repeated prediction over the intervening denoising steps, rather than within a single large prediction. To illustrate this, consider K skipped steps with a constant increment $\Delta _ { \ell }$ , so that $\rho _ { \ell }$ is also constant. Indexing steps from the start of the skip, the temporal derivative after k steps is

$$
\nu _ { i , k } = \rho _ { \ell } ^ { k } \nu _ { i , 0 } .\tag{35}
$$

The total change in the predicted coefficient value is therefore

$$
m _ { i , K } - m _ { i , 0 } = \sum _ { k = 0 } ^ { K - 1 } \Delta _ { \ell } \rho _ { \ell } ^ { k } \nu _ { i , 0 } = \Delta _ { \ell } \frac { 1 - \rho _ { \ell } ^ { K } } { 1 - \rho _ { \ell } } \nu _ { i , 0 } .\tag{36}
$$

Consequently,

$$
\left| m _ { i , K } - m _ { i , 0 } \right| \leq \frac { \Delta _ { \ell } } { 1 - \rho _ { \ell } } \left| \nu _ { i , 0 } \right| .\tag{37}
$$

Thus, the contribution of the initial temporal derivative remains bounded as the skip length increases. In contrast, without damping, $\rho _ { \ell } = 1$ and the displacement grows linearly as $K \Delta _ { \ell } \nu _ { i , 0 }$ , limited only by the length of the coordinate interval. Moreover,

$$
\operatorname* { l i m } _ { \Delta _ { \ell } \to 0 } \frac { \Delta _ { \ell } } { 1 - \mathrm { e x p } ( - \Delta _ { \ell } / \tau _ { \ell } ) } = \tau _ { \ell } ,\tag{38}
$$

so the bound approaches $\tau _ { \ell } | \nu _ { i , 0 } |$ as the discretization is refined. Without damping, the displacement can reach $2 | \nu _ { i , 0 } |$ , since $s _ { \ell }$ spans $[ - 1 , 1 ]$

For nonuniform increments, the derivative attenuation still composes exactly:

$$
\prod _ { r = 1 } ^ { K } \rho _ { \ell , r } = \exp \left( - \frac { \sum _ { r = 1 } ^ { K } \Delta _ { \ell , r } } { \tau _ { \ell } } \right) .\tag{39}
$$

The accumulated change in value depends on the underlying denoising grid, but the derivative continues to decay with the total elapsed band time. This bound concerns the deterministic extrapolation induced by the current derivative; it does not by itself bound the total forecasting error.

Process-noise covariance. Eq. (34) corresponds to adding white noise to the temporal derivative while ignoring damping,

$$
d m = \nu d s , d \nu = \sqrt { q _ { \ell , j } } d W _ { s } ,\tag{40}
$$

where $W _ { s }$ is a Wiener process. This is the standard integrated-Wiener covariance (Section $\mathbf { A } )$ . The resulting matrix is positive definite for $q _ { \ell , j } > 0$ and $\Delta _ { \ell , j } > 0$ , since

$$
\operatorname * { d e t } ( Q _ { \ell , j } ) = \frac { q _ { \ell , j } ^ { 2 } \Delta _ { \ell , j } ^ { 4 } } { 1 2 } > 0 .\tag{41}
$$

We deliberately combine this undamped covariance with the stepwise damped transition as a simple uncertainty approximation. To justify this choice, consider the exact continuous damped model with $\rho _ { \ell , j } = \exp ( - \Delta _ { \ell , j } / \tau _ { \ell } )$ . Its diagonal process variances are

$$
Q _ { 1 1 } ^ { \mathrm { d a m p } } = q _ { \ell , j } \tau _ { \ell } ^ { 2 } \left[ \Delta _ { \ell , j } - 2 \tau _ { \ell } ( 1 - \rho _ { \ell , j } ) + \frac { \tau _ { \ell } } { 2 } ( 1 - \rho _ { \ell , j } ^ { 2 } ) \right] ,\tag{42}
$$

$$
Q _ { 2 2 } ^ { \mathrm { d a m p } } = \frac { q _ { \ell , j } \tau _ { \ell } } { 2 } ( 1 - \rho _ { \ell , j } ^ { 2 } ) .\tag{43}
$$

Applying

$$
\tau _ { \ell } ( 1 - e ^ { - x / \tau _ { \ell } } ) \leq x , \qquad e ^ { - x / \tau _ { \ell } } \leq 1\tag{44}
$$

inside the integrals that define these variances gives

$$
Q _ { 1 1 } ^ { \mathrm { d a m p } } \leq q _ { \ell , j } \frac { \Delta _ { \ell , j } ^ { 3 } } { 3 } , \qquad Q _ { 2 2 } ^ { \mathrm { d a m p } } \leq q _ { \ell , j } \Delta _ { \ell , j } .\tag{45}
$$

Our choice therefore overestimates the marginal variance growth relative to the exact damped model. This is the conservative direction for uncertainty-based scheduling, since it favors earlier model evaluation rather than overconfident forecasting. For small adjacent intervals, the damped and undamped forms agree to leading order. Any remaining mismatch is absorbed by the calibrated and online-adapted process intensity $q _ { \ell , j }$

## D DETAILS OF THE UNCERTAINTY-AWARE ADAPTIVE SCHEDULER

This appendix specifies the details behind Section 4.3: the covariance recursion used to evaluate the risk, its calibrated noise and online correction, and the exact decision rule with all safeguards.

## D.1 THE RISK SIDECAR

The risk of Eq. (12) is evaluated by a dedicated covariance recursion, which we call the sidecar, that runs alongside the mean filter of Section 4.2 but is decoupled from it in two ways. First, it operates in relative units, tracking forecast error as a fraction of each band’s power, which makes its outputs comparable across bands and generations. Second, its process noise is not the dynamics noise of the mean filter but a table $Q _ { \ell , w } ^ { \mathrm { s c } }$ calibrated directly against the forecaster’s realized errors, so the sidecar estimates how wrong the forecast tends to be after a given number of skipped steps. The index w ranges over four step windows partitioning the schedule, allowing the noise to differ between early, middle, and late denoising.

The sidecar maintains one $2 \times 2$ covariance per band with entries $\left( \tilde { p } _ { 1 1 } , \tilde { p } _ { 1 2 } , \tilde { p } _ { 2 2 } \right)$ and propagates them with the same integrated-noise structure as

$$
\begin{array} { r l } & { \tilde { p } _ { 1 1 }  \tilde { p } _ { 1 1 } + 2 \Delta _ { \ell } \tilde { p } _ { 1 2 } + \Delta _ { \ell } ^ { 2 } \tilde { p } _ { 2 2 } + Q _ { \ell , w } ^ { \mathrm { s c } } \Delta _ { \ell } ^ { 3 } / 3 , } \\ & { \tilde { p } _ { 1 2 }  \rho _ { \mathrm { s c } } \big ( \tilde { p } _ { 1 2 } + \Delta _ { \ell } \tilde { p } _ { 2 2 } \big ) + Q _ { \ell , w } ^ { \mathrm { s c } } \Delta _ { \ell } ^ { 2 } / 2 , } \\ & { \tilde { p } _ { 2 2 }  \rho _ { \mathrm { s c } } ^ { 2 } \tilde { p } _ { 2 2 } + Q _ { \ell , w } ^ { \mathrm { s c } } \Delta _ { \ell } , } \end{array}\tag{46}
$$

with a fixed memory factor $\rho _ { \mathrm { s c } } = 0 . 9$ per step. At every computed step, the sidecar conditions on the fact that the value has just been observed exactly. The trend uncertainty is reduced to its posterior, and the value uncertainty and cross term collapse to zero,

$$
\tilde { p } _ { 2 2 }  \tilde { p } _ { 2 2 } - \tilde { p } _ { 1 2 } ^ { 2 } / \tilde { p } _ { 1 1 } , \qquad \tilde { p } _ { 1 1 }  0 , \qquad \tilde { p } _ { 1 2 }  0 ,\tag{47}
$$

so the risk restarts from zero after every transformer call and re-accumulates over the following skips. The quantity $P _ { 1 1 } ^ { \left( \ell \right) }$ in Eq. (12) refers to $\tilde { p } _ { 1 1 }$ propagated by Eq. (46).

Like the mean filter, the sidecar corrects itself online. At each computed step j, we compare the observed forecast error of band ℓ with the sidecar’s predicted variance,

$$
r _ { \ell , j } = \frac { 1 } { d _ { \ell } p _ { \ell } } \sum _ { i \in \cal { B } _ { \ell } } \big ( Z _ { i , j } - \bar { m } _ { i , j } \big ) ^ { 2 } , \qquad \tilde { \kappa } _ { \ell } \gets \mathrm { c l i p } _ { [ 0 . 1 , 1 0 ] } \Big ( 0 . 8 \tilde { \kappa } _ { \ell } + 0 . 2 \frac { r _ { \ell , j } } { \tilde { p } _ { 1 1 } ^ { ( \ell ) } ( j ) } \Big ) ,\tag{48}
$$

where $\bar { m } _ { i , j }$ is the predicted coefficient value before the Kalman update, $\tilde { p } _ { 1 1 } ^ { ( \ell ) } ( j )$ is the sidecar variance before it is reset by Eq. (47), and $\tilde { \kappa } _ { \ell }$ is initialized to 1. The scale $\tilde { \kappa } _ { \ell }$ multiplies the band’s contribution to the risk. If a generation’s observed errors are larger than the sidecar predicted, the ratio exceeds 1 and $\tilde { \kappa } _ { \ell }$ increases. The risk then reaches the threshold sooner, so the transformer is evaluated more often and skips become shorter. If the observed errors are smaller than predicted, $\tilde { \kappa } _ { \ell }$ decreases and the scheduler allows longer skips. The full risk is

$$
R ( j ) = \omega ( j ) \sum _ { \ell } \tilde { \kappa } _ { \ell } \gamma _ { \ell , w ( j ) } \tilde { p } _ { 1 1 } ^ { ( \ell ) } ( j ) ,\tag{49}
$$

which reduces to Eq. (12) when $\tilde { \kappa } _ { \ell } = 1$ and $\gamma _ { \ell , w } = d _ { \ell }$ . The perceptual weights $\gamma _ { \ell , w }$ are calibrated by perturbation, injecting band-limited errors into the output at different phases of sampling and measuring the LPIPS change of the final sample. If these weights have not been calibrated, we use $\gamma _ { \ell , w } = d _ { \ell }$ , the number of coefficients in band ℓ. The risk then measures the total expected error summed over all coefficients, giving every coefficient the same importance.

## D.2 DECISION RULE

After each computed step $j ,$ the sidecar is rolled forward predictively, applying Eq. (46) over a horizon of up to 10 future steps without changing its state, and the resulting risks $R ( j + 1 ) , R ( j + 2 ) , . . .$ are compared against the threshold. The next computed step is placed at the first predicted crossing $R > \varepsilon ;$ all steps before it are forecast. Three safeguards apply. The first 3 steps of sampling are always computed, giving the filters real observations to establish values and trends; the final step is always computed, so the sample is finished by a true model evaluation; and the number of consecutive skips is capped at 10 regardless of the predicted risk. The threshold ε is set once for each compute budget. Starting from a range of candidate values, we run the scheduler on a set of calibration prompts that are not used for evaluation and measure the average number of transformer calls. If the average is above the target, we raise $\varepsilon ;$ if it is below, we lower it. We repeat this, halving the range each time, until the average matches the target.

At inference, ε is kept fixed, but the number of transformer calls is not. The risk depends on the per-band scale $\tilde { \kappa } _ { \ell } ,$ , which is updated from the errors observed during each generation. A generation whose trajectory is easy to forecast accumulates risk slowly and uses fewer calls than the target, while a harder one uses more. The target is therefore met on average rather than for every generation, and the number of calls varies around the target from one generation to the next.

## D.3 FIXED-SCHEDULE VARIANT

As a deterministic alternative, we support a fixed schedule with linearly growing gaps, computing steps

$$
j _ { r } = \big \lfloor \alpha r ( r + 1 ) / 2 \big \rfloor , \qquad r = 1 , 2 , \ldots , \quad j _ { r } < N - 1 ,\tag{50}
$$

together with the warm-up steps {0, 1, 2} and the final step, where α is chosen by a simple search so that the total number of computed steps matches the target budget. The spacing is dense early in sampling, where the output changes fastest, and sparse late, similar to the schedules the adaptive scheduler produces for typical generations. The fixed variant is preferable when a deterministic compute cost is required, while the adaptive variant reallocates calls between easy and hard generations at the same average cost. In Table 3, the adaptive variant improves PSNR by 2.2 dB over the fixed variant at the same budget.

## E MODELS WITH CLASSIFIER-FREE GUIDANCE

Some of the evaluated models use classifier-free guidance (CFG) (Ho & Salimans, 2022) at inference. For these models, each denoising step runs the transformer twice, once with the prompt and once with the negative prompt, and combines the conditional output $Y _ { j } ^ { c }$ and the unconditional output $Y _ { j } ^ { u }$ into a guided velocity

$$
Y _ { j } = \mathcal { G } \big ( Y _ { j } ^ { c } , Y _ { j } ^ { u } \big ) , \qquad \mathcal { G } ( c , u ) = u + w ( c - u ) ,\tag{51}
$$

where w is the guidance scale. Some models modify $\mathcal { G } ;$ Qwen-Image, for example, rescales each token of the guided velocity to the norm of the conditional prediction (Eq. (55)).

Forecasting the guided velocity. SpecKCast forecasts the transformer output rather than its internal features, so it can be placed outside the network. For models with CFG, we place it after the guidance combination and forecast the guided velocity $Y _ { j }$ , which is the quantity the sampler uses, with a single forecaster. On a computed step, both passes run, $Y _ { j }$ is formed with the model’s own ${ \mathcal { G } } ,$ and the forecaster is updated with it. On a skipped step, neither pass runs and the forecast $\widehat { Y } _ { j }$ is passed to the sampler. The frequency bands, time coordinates, Kalman filters, and scheduler are the same as for models without guidance, which correspond to $\mathcal { G } ( c , u ) = c$ . With n computed steps out of $N$ , SpecKCast therefore uses 2n transformer passes instead of 2N.

Relation to forecasting each pass separately. An alternative is to forecast $Y _ { j } ^ { c }$ and $Y _ { j } ^ { u }$ separately on the same schedule and combine the two forecasts with G. Suppose the forecast at a skipped step k is a linear combination of the outputs observed at earlier computed steps,

$$
\widehat { Y } _ { k } = \sum _ { j \in \mathcal { U } , j < k } a _ { k , j } Y _ { j } ,\tag{52}
$$

where the weights $a _ { k , j }$ do not depend on the observed values. If, in addition, $\mathcal { G } ( c , u ) = w c + ( 1 -$ w) u with the same w at every step, then

$$
\sum _ { j } a _ { k , j } \mathcal { G } \big ( Y _ { j } ^ { c } , Y _ { j } ^ { u } \big ) = w \sum _ { j } a _ { k , j } Y _ { j } ^ { c } + ( 1 - w ) \sum _ { j } a _ { k , j } Y _ { j } ^ { u } = \mathcal { G } \big ( \widehat { Y } _ { k } ^ { c } , \widehat { Y } _ { k } ^ { u } \big ) ,\tag{53}
$$

so forecasting the guided velocity gives the same output as forecasting each pass and combining the forecasts. Taylor extrapolation and the Chebyshev fit used by Spectrum have the form of Eq. (52), and so does a Kalman filter with fixed noise parameters, since its gains do not depend on the observed values. In SpecKCast, the equivalence is no longer exact because the process noise is adapted online from the observed errors, which makes the weights $a _ { k , j }$ depend on the observations. It is also not exact when $\mathcal { G }$ includes a nonlinear correction such as the norm rescaling of Qwen-Image. We forecast the guided velocity because it is the quantity the sampler uses, so the forecast error and the scheduling risk are measured directly on what affects the sample, and because it requires a single set of filter states instead of two.

Variants. We also implement two variants, which are not used in the reported experiments. The per-branch variant runs one forecaster for each branch on a shared schedule, set by the conditional branch, and combines the two forecasts with ${ \mathcal { G } } .$ The asymmetric variant gives each branch its own schedule and compute budget, so a step can run neither pass, only one, or both. Writing Eq. (51) as $Y _ { j } = w Y _ { i } ^ { c } - ( w - 1 ) Y _ { j } ^ { u }$ shows that an error in the unconditional forecast is scaled by $w - 1$ , while an error in the conditional forecast is scaled by w. This motivates spending fewer evaluations on the unconditional branch.

## F EXTENDED QUALITATIVE COMPARISONS

We extend the comparisons in Sections 5.1 and 5.2 with additional Qwen-Image, HunyuanVideo, and Wan2.1-14B examples. Each panel reports its evaluation count and measured speedup relative to the 50-step reference. These examples illustrate the visual effects of acceleration; aggregate results are reported in Tables 1 and 2.

Why SpecKCast preserves the reference more closely. As discussed in Section 2, cached features become outdated over long skips, while feature-space extrapolation does not distinguish the temporal dynamics of different spatial frequencies. SpecKCast forecasts the velocity passed to the sampler one DCT coefficient at a time, using band-specific time coordinates to account for these differences. Damped trends limit overshooting, fresh evaluations correct the forecast, and uncertaintyaware scheduling allocates computation when the predicted error grows. The ablations in Table 3 support the contributions of frequency decomposition, time warping, and adaptive scheduling; the examples below illustrate the combined method.

Qwen-Image: color, texture, and object structure. In Figure 7, TaylorSeer and HiCache give the strawberry scene a strong blue cast and darken the fox scene. TeaCache changes the basket’s contents and introduces a second astronaut, while SeaCache adds an extra horse leg and smooths the fox’s fur. Spectrum introduces grain and distorts the rider. SpecKCast more closely retains the reference lighting, object arrangement, and recognizable details, with 6.19–6.21× speedup at 8 NFE and 4.97× at 10 NFE. SpecKCast’s closer match to the reference is visible in both overall appearance and local structure, at lower latency than the displayed baselines.

![](images/13185a24ad9f73a3ea18876d67d3c7b5f47f11d57922eeeec0622c5cedeca990.jpg)  
Figure 7: Additional Qwen-Image comparisons. Rows show strawberries in a basket, an astronaut riding a horse, and a fox in snow. The leftmost column is the 50-NFE reference; all accelerated methods use 8 NFE in the first two rows and 10 NFE in the last, with the measured speedup shown below each image. SpecKCast more closely preserves the reference colors and object structure, while the baselines exhibit color shifts, altered content, or degraded texture.

HunyuanVideo: subject detail and geometry. In Figure 8, FORA and TaylorSeer blur the face, TeaCache changes the action and produces overlapping hand and face structure, and HiCache changes the subject’s appearance. SpecKCast produces a clearer face and retains the bottle in hand across the displayed frames at 10 NFE and 4.27× speedup, although the subject’s appearance still differs from the reference. In Figure 9, TaylorSeer and Spectrum introduce an extra wheel-like structure between the bicycle’s two wheels. SpecKCast preserves the two-wheel geometry with fewer evaluations (13 NFE compared with 14 and 16), though it omits the reference’s front basket.

![](images/d8fdfb075c0bb1cde36552dd8795c4b4ed68ea3cf86975a2cfc8bd91d0d7676c.jpg)  
Figure 8: HunyuanVideo: “a bottle and a chair.” Frames 0, 40, and 80 for each method show differences in facial detail, pose, and bottle placement. SpecKCast uses 10 NFE (4.27×), compared with 10–11 NFE (3.85–4.24×) for the baselines. It avoids the severe facial blur visible in FORA and TaylorSeer and keeps the bottle in hand, although the subject’s appearance still differs from the reference.

Wan2.1-14B: appearance and scene structure at 8 NFE. In Figure 10, HiCache and TaylorSeer produce strong saturation and blurred boundaries. FORA and TeaCache retain the layout but soften

HunyuanVideo “a bicycle leaning against a tree”  
![](images/96c8cb7e90fb9abff9bd07cc77e7f39ce7e52289a6c1c6c7ac9d2724c81a431b.jpg)  
Figure 9: HunyuanVideo: “a bicycle leaning against a tree.” TaylorSeer (16 NFE, 2.82×) and Spectrum (14 NFE, 3.13×) introduce an additional wheel-like structure between the bicycle’s two wheels. SpecKCast (13 NFE, 3.41×) better preserves the main bicycle geometry across the displayed frames, although the front basket is missing.

Wan2.1-14B

the panda’s detail, while SeaCache changes its appearance and the tableware at 7 NFE. SpecKCast retains the reference composition and drinking poses at 5.61× speedup. In Figure 11, TaylorSeer severely distorts the boat and skyline, and Spectrum alters the Eiffel Tower’s shape. SpecKCast retains a more faithful landmark silhouette and river layout at the same 8 NFE and 5.65× speedup as Spectrum.

![](images/c37fe66808e2255dd350926c343726ee66752c5d2b099e4cb07183a94c0b0be4.jpg)  
Figure 10: Wan2.1-14B: a panda drinking coffee with a zoom-in. Frames 0, 40, and 80 show subject appearance and scene layout. All accelerated methods use 8 NFE except SeaCache (7 NFE). SpecKCast achieves 5.61× speedup while avoiding the strong color and boundary artifacts of Hi-Cache and TaylorSeer. TeaCache (5.64×) and SeaCache (6.05×) are faster, but TeaCache softens the panda’s detail and SeaCache changes the scene more substantially.

Wan2.1-14B  
![](images/76624a85966e4f791340fcb1530b2b4f38ca3922c6a13e2a21463c7966335d33.jpg)  
Figure 11: Wan2.1-14B: a boat on the Seine with the Eiffel Tower in the background. All three accelerated methods use 8 NFE, with speedups of 5.62× (TaylorSeer) and 5.65× (Spectrum and SpecKCast). SpecKCast more closely preserves the tower’s silhouette and scene layout across the displayed frames.

## G ADDITIONAL SPATIAL-FREQUENCY TRAJECTORY ANALYSIS

We extend the single-generation FLUX.1-dev trajectories in Figure 2 in three ways: joint views computed over 12 prompts, individual generations for six of the prompts, and the same analysis for Qwen-Image. Both collections use the same 12 prompts, with one generation per prompt and model. Figure 12 shows the joint FLUX.1-dev trajectories in the PC1–PC2 plane and as scores against denoising step, and Figure 13 shows all three joint views for Qwen-Image. The remaining figures show individual generations for six of the prompts. Together, these visualizations examine how temporal evolution differs across spatial-frequency bands and how the patterns vary across prompts and models.

## G.1 MEASUREMENT SETUP

For both models, each generation uses a resolution of 1024 × 1024, 50 denoising steps indexed from 0 to 49, and full transformer computation at every step. We use zero-based prompt IDs: prompt $p \in \{ 0 , \ldots , 1 1 \}$ uses generation seed $p .$ The texts of prompts 0–5, which are shown individually, are given in the figure captions.

In both models, the recorded velocity is unpacked to $Y _ { j } \in \mathbb { R } ^ { 1 6 \times 1 2 8 \times 1 2 8 }$ , and we apply an orthonormal two-dimensional DCT-II independently to each channel. For spatial-frequency indices (a, b), the normalized radius is

$$
r ( a , b ) = { \frac { \sqrt { ( a / 1 2 8 ) ^ { 2 } + ( b / 1 2 8 ) ^ { 2 } } } { \sqrt { 2 } } } , \qquad a , b \in \{ 0 , \dots , 1 2 7 \} .\tag{54}
$$

The seven radial bands have edges $[ 0 , 1 / 6 4 , 1 / 3 2 , 1 / 1 6 , 1 / 8 , 1 / 4 , 1 / 2 , 1 ]$ . Each panel set visualizes 512 fixed sampled entries of the full velocity, all 128 coefficients of $B _ { 0 }$ , and 512 fixed sampled coefficients each from $B _ { 3 }$ and $B _ { 6 } .$ . Coefficient sampling uses seed 1234, and the sampled indices are shared across steps and prompts. The full-velocity panel summarizes sampled entries of the spatialdomain output, while the band panels summarize DCT coefficients. The joint analysis combines the measurements from all 12 generations, while each individual analysis uses only its own generation.

FLUX.1-dev. The FLUX collection uses guidance scale 3.5, and we record the velocity produced by the transformer at each step.

Qwen-Image. We record the velocity passed to the sampler after classifier-free guidance at scale 4 and the pipeline’s norm rescaling. For conditional and unconditional predictions $c _ { j , t }$ and $u _ { j , t }$ at packed token $t ,$ the recorded output is

$$
\widetilde { v } _ { j , t } = u _ { j , t } + 4 ( c _ { j , t } - u _ { j , t } ) , \qquad v _ { j , t } = \widetilde { v } _ { j , t } \frac { \lVert c _ { j , t } \rVert _ { 2 } } { \lVert \widetilde { v } _ { j , t } \rVert _ { 2 } } ,\tag{55}
$$

where the negative prompt is empty and each norm is taken over the packed token’s 64 channels.

## G.2 PCA CONSTRUCTION AND DISPLAY

Both models use the same procedure for joint views and normalization, applied independently to each model. For each prompt and panel, we arrange the recorded values into a coefficient-bystep matrix and divide it by its Frobenius norm. For a joint view, the 12 normalized matrices are concatenated along the coefficient axis and transposed, giving a matrix with 50 observations and 12d features, where d is the number of retained entries in that panel. We center each feature across steps and project onto its first two principal components. Thus, each joint curve summarizes temporal variation across all 12 generations. For the individual views, the same procedure is applied to each prompt separately, giving a distinct PCA basis for each prompt and panel.

Within each panel, both score sequences are divided by their joint maximum absolute value. This shared scaling preserves the relative geometry of the two displayed components. For both models, the PCA is computed by singular value decomposition, PC1 is oriented so that its final score is at least its initial score, and PC2 is oriented so that its score at step 25 is nonnegative. Within each joint figure, all rows use exactly the same normalized scores, as do the paired three-dimensional and planar views for each individual prompt. Planar views use equal axis scaling and shared square limits. Circles mark measured states, blue changes from dark to light with denoising progress, and stars mark the final state (PC1 in the score-versus-step view). Straight segments connect successive measurements without smoothing.

## G.3 TEMPORAL PATTERNS ACROSS PROMPTS

FLUX.1-dev. Figure 12 shows that the pattern of the single generation in Figure 2 holds across all 12 prompts. In $B _ { 0 }$ , the displayed scores change sharply during the first few steps and remain near their final values over most of the schedule. The middle band $B _ { 3 }$ evolves over a broader interval. The high band $B _ { 6 }$ changes slowly at first, with larger changes near the end. The planar projections show the path of the measured states, while the score-versus-step curves show when each component evolves.

Qwen-Image. In Figure 13, $B _ { 0 }$ also undergoes a sharp initial transient and then changes little in the displayed components. The middle band $B _ { 3 }$ evolves over a broader portion of the schedule, while $B _ { 6 }$ exhibits an early transient followed by a pronounced change near the end. Across both models, the different timing across bands supports the motivation in Section 4.1.1 for separate temporal models and band-specific forecasting coordinates.

Each panel has its own PCA basis and normalization, so absolute distances, directions, and amplitudes cannot be compared across models, bands, prompts, or joint and individual views. A nearly vertical trajectory in the three-dimensional plots indicates little change in PC1 and PC2; variation can still occur in the omitted components. These plots therefore provide qualitative evidence about temporal structure in the retained subspaces, rather than a quantitative measure of full-velocity motion or forecasting error.

## G.4 INDIVIDUAL PROMPT TRAJECTORIES

Figures 14 to 16 show individual FLUX.1-dev generations for prompts 0–5, and Figures 17 to 19 show the same prompts for Qwen-Image, with two prompts per figure. For each prompt, the upper row displays the trajectory with denoising step increasing downward, and the lower row displays the identical normalized PC pairs in the plane. Columns show full velocity, $B _ { 0 } , B _ { 3 } ,$ , and $B _ { 6 }$ . In the planar views, numbers next to points mark denoising steps: steps 0, 25, and 49 for the full velocity, and steps 0 and 49 together with step 1 for $B _ { 0 }$ , step 10 for $B _ { 3 }$ , and step 40 for $B _ { 6 }$ , marking where each band changes most. These separate fits show differences between generations that are averaged out in the joint views. Their coordinates are specific to each model, prompt, and panel.

![](images/477b7f6a2dd467855355ba776824d89b5721746ca66e74f68b2fb4420458b4a7.jpg)  
(a) Full velocity

![](images/aa5a4b3dad1b0d13f691837ed9659594ab5bd7fa91a9a38307d528be2c0939c3.jpg)

Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
![](images/a379a22e983ba05fc1e2eefdf6eac91a9ddf3c03d86fad61295817f3cacaf247.jpg)  
(b) Low frequency B<sub>0</sub>

(c) Mid frequency B<sub>3</sub>  
(d) High frequency B<sub>6</sub>  
![](images/0b286e0cde89f554c485b8b7fd047838c9cecebe438723ef7740f96f5f946b9d.jpg)  
Denoising trajectories (dark → light: denoising progress)  
Figure 12: Joint FLUX.1-dev trajectories across 12 prompts. Columns show the sampled full velocity and DCT bands $B _ { 0 } , B _ { 3 }$ , and $B _ { 6 }$ . Both rows use the same normalized PCA scores, computed jointly over 12 prompts with 50 measured velocities per prompt, guidance scale 3.5, and full transformer computation. Unlike Figure 2, which shows a single generation, the PCA here is fitted to all 12 generations together. Top: PC1–PC2 projections with equal axis scaling; labels identify selected steps. Bottom: PC1 (solid blue) and PC2 (dashed gray) against step index. Circles mark measurements, dark to light blue indicates denoising progress, and stars mark the final state (PC1 in the bottom row). Each column has its own PCA basis and scale, constructed as described in Section G.2.

Denoising trajectories (dark → light: denoising progress; ★ final state)

(a) Full velocity

Denoising step  
![](images/da8311f655220c30c3568a8c0fd59f22e359ef3823da0479dfba857a41775bbe.jpg)  
(b) Low frequency B<sub>0</sub>

![](images/4b84ddcd9203fa26f9320aab1c4c74632ab6bbfa9f09ae978b7c08d0a3aacd82.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/4f883f111c6816522e5e77bd9dcd1af31b8011460c468eecbb78822c9c13f4e8.jpg)  
(d) High frequency B<sub>6</sub>

![](images/3205224ef629fb720779adefe7f87f19c02f69e3c36d1f130d7dc53126e8008e.jpg)

Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
(a) Full velocity  
![](images/d88e9ecb3ae022b3da54f3e525642c36144d33b0849fb95e93a7f0742da2e81e.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/8d14c17c7e918184f3b97d2ee2d822724401e121459bf4d2c32b00145bff16c7.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/f25cde3f588fe61e0c56dffc910aacac986fb43f431cfb94110ededae24a8fe8.jpg)

(d) High frequency B<sub>6</sub>  
![](images/7a77876f7de96c85248e12c656f04f60e51182b412a2118fee0c37571425f8dd.jpg)

Denoising trajectories (dark → light: denoising progress)  
(a) Full velocity  
![](images/bb70d700a56b4e6783c0cedbb7014418e87a53e060ce73aec9c2efdfd771b75b.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/75751ffecf3ec824a7569a70d42aa3f49504fe1fd64119851d6be74595e3ed28.jpg)

★ Final PC1 state  
(c) Mid frequency B<sub>3</sub>  
![](images/e4208c021ad900463fc3ca1e30dc9145b9c8e8894e695f70f7c236eee92cc498.jpg)

(d) High frequency B<sub>6</sub>  
![](images/486492dbf68cc445d6249c29837fabac5e0ae68767c9a3dbf1259d3ad8ba680a.jpg)  
Figure 13: Spatial-frequency trajectories of Qwen-Image across 12 prompts. Columns show the sampled full velocity and DCT bands $B _ { 0 } , \ B _ { 3 } ,$ , and $B _ { 6 } .$ . All rows use the same normalized PCA scores, computed jointly over 12 prompts with 50 velocities passed to the sampler per prompt, guidance scale 4, and full transformer computation. Top: Denoising step increases downward. Middle: The same PC1–PC2 pairs with equal axis scaling; labels identify selected steps. Bottom: PC1 (solid blue) and PC2 (dashed gray) against step index. The low band changes mainly at the start, the middle band evolves over a broader interval, and the high band changes sharply near the end after a small early transient. Bases and scales are fitted independently for each column; see Section G.2 for the construction.

(a) Full velocity

![](images/fb713c9f35bb8aa2eb67fc21ab48f43c6d05250d50b08f72d57280299c8fd766.jpg)  
Prompt 000 (seed 0)  
(b) Low frequency B<sub>0</sub>  
Denoising trajectories (dark → light: denoising progress; ★ final state)

![](images/95f14bd9e1af435accd99157a3a787763c24d9b2acff0e15611ecc62dc67efd2.jpg)  
(c) Mid frequency B<sub>3</sub>

![](images/390b676e3c51b597f848eae542dcf5348c33665e30a67e102b9317a627089c9e.jpg)  
(d) High frequency B<sub>6</sub>

![](images/4ddf7d63e77ef4ca9260e29e9ed0519465daad1779f1cb148164a5916a9cef0a.jpg)  
Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/249facbcfa230d086951a8aff85baf9cb6ccb48e198d8d4ce8853893c13377b3.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/ac38b5fb2dc602a84de64e01be737129d04c8f70e3db9d005924b96009cab856.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/ed29dd5cfc68575d7303617759aa68665dbad6b55cfa91a126fc3c8ae3f9aebc.jpg)

(d) High frequency B<sub>6</sub>  
![](images/395a48268377e62928296ad3ac8d95b0e6bb4ad9d48c3d73302bc27af68623e0.jpg)  
Prompt 001 (seed 1)  
Denoising trajectories (dark → light: denoising progress; ★ final state)  
(a) Full velocity

![](images/c6c412341e9de975993d8518ccaaa5c744e71899615fe447156721302559ee82.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/1bc1a6fad52007fb1476b69f0814d6d0d992773e1ce5c51447e920a84e27d602.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/a7740c568970097118f00de87ec0c24b6bf7e2b4c9ef8573e549dc254b3f056d.jpg)

(d) High frequency B<sub>6</sub>  
![](images/0afea8a57ea085e739012ae50c7b2382129c68fbd0d473111f0acb2ef5ad1ad6.jpg)

Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
(a) Full velocity  
![](images/940a3d8c9887966708c437dd64e18614f46d46258a4c994d2a36de3e4e08fc7b.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/0ccad2963ca7890a41fecefc1186459e039c865fe9b89174bb280024348bf7fd.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/24e8d4c30b8f1a7e109c95d217ca6c7feab3be842ae105bff9a4fca782561ab4.jpg)

(d) High frequency B<sub>6</sub>  
![](images/774674ccdc6b2eb0b0d2564f00f405d83698f352eba1fa7427c6941442d32ed2.jpg)

Figure 14: Individual FLUX.1-dev trajectories: prompts 000 and 001. Top two rows: $\mathbf { \ddot { a } }$ cat holding a red apple” (seed 0). Bottom two rows: “a photograph of an astronaut riding a horse on the moon” (seed 1). For each prompt, the same 50 states are shown in 3D with the denoising step as the vertical axis and in the PC1–PC2 plane. Columns show full velocity, $B _ { 0 }$ , $B _ { 3 }$ , and $B _ { 6 } ;$ dark to light blue indicates denoising progress and stars mark the final state. Each panel uses its own PCA basis and scaling, so positions and sizes cannot be compared across panels or prompts.

(a) Full velocity

Prompt 002 (seed 2)

Denoising trajectories (dark → light: denoising progress; ★ final state)

![](images/9ec3d47e9ad74e563b273d15568e07bfc4f77074582bd90deeeaff9d05b087fc.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/b25819dd3b32fce2e03344a9db06d7a5f37804b56ffacc50116ddc38d0c94336.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/e236b2614f15de0ec68700ea9a20483780c5e8099cb985b60a2e37760e4ce25a.jpg)

(d) High frequency B<sub>6</sub>  
![](images/8b4de8415e03cb2053974d98114f074ec9632443f94c522682b1cf846b51b63e.jpg)  
Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/c4fa943970b240c276db7acd15279657377b82d91080754974fa53732e456fa7.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/9fcaa6456f9a9c1e0134128aa163978c3d988af67f48632315841c3f77b3fe1f.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/ebd8db9510a5c10bd1b779a533e3eb113fcdca66bae575bf2f88c9cee6d4ddb0.jpg)

(d) High frequency B<sub>6</sub>  
![](images/065a84533089d411411c77f24008950f174a46a9017232b3c3d33be603ef95f7.jpg)

## Prompt 003 (seed 3)

Denoising trajectories (dark → light: denoising progress; ★ final state)  
(a) Full velocity  
![](images/823c40ba63113f9429b7474b3e5b1f17935780166b926ab2ef2f23d5ddd4e5ed.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/bdbb4da5a93cbf5dcd1ddb3a33dff3a683a1ff6c5842eaa3302ae41d4375e910.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/68ab1a7a97c601b4781868a99304a1e8b7d71158c787d9bc2e8cc1e882a3dce5.jpg)

(d) High frequency B<sub>6</sub>  
![](images/0ffeffe64552985e7f53b263bdae29267d53a909b48608639b9419a74558e001.jpg)

(a) Full velocity  
Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
![](images/d6b5a86578b2ea33d5c6a6c00cfb4385b554a46c332fe2f0576e7cc5c1661185.jpg)

![](images/45eb02984de12a50e6296566c4982b2b5395ff1db7efb3bad126b8e020726574.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/f8491765daea0d8b6adda7cc254230c4fbbcc62c30e15208a7b06a6ea11f737b.jpg)

(d) High frequency B<sub>6</sub>  
![](images/8c46bd21dd0a0f37221366dcf097a2f878563c0dea66cc81f820731ff91b180a.jpg)

Figure 15: Individual FLUX.1-dev trajectories: prompts 002 and 003. Top two rows: “a bowl of ramen with chopsticks, steam rising, cinematic lighting” (seed 2). Bottom two rows: “a red panda reading a book in a cozy library” (seed 3). For each prompt, the same 50 states are shown in 3D with the denoising step as the vertical axis and in the PC1–PC2 plane. Columns show full velocity, $B _ { 0 } , B _ { 3 }$ , and $B _ { 6 } ;$ dark to light blue indicates denoising progress and stars mark the final state. Each panel uses its own PCA basis and scaling, so positions and sizes cannot be compared across panels or prompts.

(a) Full velocity

Prompt 004 (seed 4)

Denoising trajectories (dark → light: denoising progress; ★ final state)

![](images/8389d7eaaa44b0fd64b726e937445bd103bd81a5758606ee6725a653b2855020.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/1081b2f1d64e16ed1c7b41b8ba55563efd37da739953b330fe2330ec6da55da0.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/cc28fbfee0d258146f331447356dde94843fc51a3e26777063f20359785b5697.jpg)

(d) High frequency B<sub>6</sub>  
![](images/fd71cdedfd41620fe1b9bea91abd20422f206b8b78ce2f6e7f05013db2cbde29.jpg)  
Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/4ee5d21bdecc05a76f1701b7a9d80a2c5728100c37897026cc84d231f46885c7.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/5e07f69546c3d9b29373abd1625de96893a8afb177576f954e40b22b4b092e3f.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/87b350ab34b7fce254e9bb10d19852d766ddb8743eb62ec38934512bddb15891.jpg)

(d) High frequency B<sub>6</sub>  
![](images/42ef83b563144a039f4b8d9e608c2b3b14c339cd0eaca607f66b0a823b36f094.jpg)

## Prompt 005 (seed 5)

Denoising trajectories (dark → light: denoising progress; ★ final state)  
(a) Full velocity  
![](images/e85f5a42cb0348c928d15105d4572618b13fc53d1583a233f133e8c65a764f38.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/e728cc717171e841a9c6158621e0a9565692a1547e46b61158b7c3696171b4b6.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/c46ba3b0f1cacb64f37d1680d20c633920dd81f7db5b6fea8dd71363dfd92e00.jpg)

(d) High frequency B<sub>6</sub>  
![](images/eec08e35338484af26613c0b7f1a64c85bd0f2f824bbc93bfc1c7dd5197e0abd.jpg)

(a) Full velocity  
Flux.1[dev]: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
![](images/71e7ad358325ef353bad55996e1bd702389f523005f72697a1082123b077cc58.jpg)

![](images/4c5f5b25eaae20fe898d7268d83207084c4b682281894ea5fbd2714b4724d9bb.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/167198c0f31e9808f999fd5735e22c43c382baecd0d97ab41e7c6cabff6c48b1.jpg)

(d) High frequency B<sub>6</sub>  
![](images/70ef7f84688f826396e380589c425a86b58ff581ff410d0992fb6418caab844a.jpg)

Figure 16: Individual FLUX.1-dev trajectories: prompts 004 and 005. Top two rows: “a futuristic city skyline at sunset with flying cars” (seed 4). Bottom two rows: “a close-up portrait of an old fisherman with a weathered face” (seed 5). For each prompt, the same 50 states are shown in 3D with the denoising step as the vertical axis and in the PC1–PC2 plane. Columns show full velocity, B , B , and $B _ { 6 } ;$ dark to light blue indicates denoising progress and stars mark the final state. Each panel uses its own PCA basis and scaling, so positions and sizes cannot be compared across panels or prompts.

(a) Full velocity

![](images/6d7cc60217c3e00700e5b6fd6893f203bc10af866e0e1acf852a6c0479d4f217.jpg)  
Prompt 000 (seed 0)  
Denoising trajectories (dark → light: denoising progress; ★ final state)  
(b) Low frequency B<sub>0</sub>

![](images/eb5c1245ba25116d34d156edeb2e523c915fc1ee58923f7462bc782838994ed2.jpg)  
(c) Mid frequency B<sub>3</sub>

![](images/4bc9f34b9edc6527b9737cc82b30c2c8d88e65b8d7389c6dab153b4cb8339a5a.jpg)

(d) High frequency B<sub>6</sub>  
![](images/b6d317ab577fedfe03a3e26a1ced1cd91e3443cd2999c30a2247e9c62396ebb6.jpg)  
Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/8f56689be7986053d2b74797b6bbe7715f5cfd50810fead952f6cfe4429ac8df.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/22b756956ed2d553561db6188b4145801805542d3bfb2a5aad8a96689e61073f.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/46f2e32d31026f0d98f25285546d91008e4f0f4fe3409a7419a6f4947b10792c.jpg)

(d) High frequency B<sub>6</sub>  
![](images/952b9bb2a4dd9044928143e1621c5d22465cdf12cfa21b4f262ac3d1e89976f2.jpg)  
Prompt 001 (seed 1)  
Denoising trajectories (dark → light: denoising progress; ★ final state)  
(a) Full velocity

![](images/b363a9db350579dd673c8601f030cc63c5b3b5cd3702b46078b8f5c8c0f127e1.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/38ea04b613f67612139043f826e5e8709cbf505db165b5e388d7dd91317457f0.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/cb3759d2afe5df2240e26db1984e0f63119ec4929d0b021911661f365e8fe488.jpg)

(d) High frequency B<sub>6</sub>  
![](images/6d0791028f6a2d040f76a864b3fea7593eb3edd1b88422b2910a45f1a649a8dc.jpg)  
Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/ec9d793cda4036cd37f80b2b538776ac23321fe49400116107ace05d9b406e53.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/e8d2c56555085d55896fa0f34dab5c8a7e0b1adff3b486aca95f01f3b6a43584.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/b8ef73acbca1e734019a8612fae0770200f58753d1e0b2fb7e059da0718d9768.jpg)

(d) High frequency B<sub>6</sub>  
![](images/08b96f2832f52ba867d7a97c3de20e041189113bc0416671e5e46e57943fe1a3.jpg)

Figure 17: Individual Qwen-Image trajectories: prompts 000 and 001. Top two rows: $\mathbf { \ddot { a } }$ cat holding a red apple” (seed 0). Bottom two rows: “a photograph of an astronaut riding a horse on the moon” (seed 1). For each prompt, the same 50 states are shown in 3D with the denoising step as the vertical axis and in the PC1–PC2 plane. Columns show full velocity, $B _ { 0 }$ , $B _ { 3 }$ , and $B _ { 6 } ;$ dark to light blue indicates denoising progress and stars mark the final state. Each panel uses its own PCA basis and scaling, so positions and sizes cannot be compared across panels or prompts.

(a) Full velocity

Prompt 002 (seed 2)

Denoising trajectories (dark → light: denoising progress; ★ final state)

![](images/08f3dfa9f0650600bf3e6dd297520f4b40adeba47bb16223134cfcce32668356.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/4595c9afd5804bb604764cef70e70733a6be44e611dfa1b8b06e48567c9d8206.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/eddc46d86aa4b3d87cc094039509d84624b086a7bc1233ff7d01f26d6953b4f3.jpg)

(d) High frequency B<sub>6</sub>  
![](images/1e0ca5a50330115d0d7e76ee3a3f4181f0f560b1a05244e90f6aff34a29b3169.jpg)  
Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/ac8262546976907262edc86b336565bb256862a733c9a76babcde2c131411753.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/a0e63751ee2c8fdd25defb91faf4a525d52bab2c21385b7b8fa64d936b0e2a6d.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/84f653ec1f1ad5f47b3ff35634b52e171e6c8aee7bd3aead8952a72976f4a1f3.jpg)

(d) High frequency B<sub>6</sub>  
![](images/75ec75ddf3ba90eea03501330701a74bf48f783cd7d839eaeb848eceb9675f4c.jpg)

## Prompt 003 (seed 3)

Denoising trajectories (dark → light: denoising progress; ★ final state)  
(a) Full velocity  
![](images/b6d6c30b50ad8b7672564f1e0b97e7f91621f04a190e59675e0ebeeaed69d2b1.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/37e816a3c90797ba8be77945369639e2d96b42ed51d12fc7ea5b8b6078bc957c.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/60743fffcaec1a3eb9de19faf01af4518587f6d7e792d9e48f035e560c9409d5.jpg)

(d) High frequency B<sub>6</sub>  
![](images/0ddc4e98b3c9f874c49b8b33f7c31650b6a46c0cdeccd804afae7541bd06098b.jpg)

(a) Full velocity  
Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
![](images/b8660df7c5315fe74b6c344407aeb23d14128e464ecd77d5e17c3b1c44d2bf7b.jpg)

![](images/a776a55621dff5598d90da39111cd6fc2417311530a506bf84fd3adfc2cee3b3.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/86df43b4a4865b1619b749cf67ac97cecd1d599a45353a70c85042b63ad1805a.jpg)

(d) High frequency B<sub>6</sub>  
![](images/73d136f1602af940eb4ef49a36fd1ac00334eb9edd980ece27d95c5faea3ad12.jpg)

Figure 18: Individual Qwen-Image trajectories: prompts 002 and 003. Top two rows: “a bowl of ramen with chopsticks, steam rising, cinematic lighting” (seed 2). Bottom two rows: “a red panda reading a book in a cozy library” (seed 3). For each prompt, the same 50 states are shown in 3D with the denoising step as the vertical axis and in the PC1–PC2 plane. Columns show full velocity, B<sub>0</sub>, B<sub>3</sub>, and $B _ { 6 } ;$ dark to light blue indicates denoising progress and stars mark the final state. Each panel uses its own PCA basis and scaling, so positions and sizes cannot be compared across panels or prompts.

(a) Full velocity

Prompt 004 (seed 4)

Denoising trajectories (dark → light: denoising progress; ★ final state)

![](images/175f9c86e6aa4b13433b71d7a1a04ff49d9a233d22418138435a3c86fa39f942.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/b8526786bf6f998826376c36a80532bef155e649274860c8d61338fd96048074.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/2d25f96e20a947e3c04c42e8c1c323bb0382d281a1b67affca1cac10e76ad8fd.jpg)

(d) High frequency B<sub>6</sub>  
![](images/a29e1501c3af541c49497ce6077858e1aab17c53457ecb04204605e062e655d2.jpg)  
Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)

(a) Full velocity  
![](images/15ab07ea4d31fd4c8fb20c59afe92cf9ffcc298d2872b55d69c406ec8d3a2118.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/cc77770dafe6de5f0f467aac1211584da9f082b6dedfe3ec469d17ffdf76e845.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/48b07a9c1e4e9d4a7d907be332677f505055cdeeafd34d45a9a35d9ea6837263.jpg)

(d) High frequency B<sub>6</sub>  
![](images/f05d6410cfc825cef913a214bfe5504eec18c3f3f290b8b08c9d0d811cb59f8b.jpg)

## Prompt 005 (seed 5)

Denoising trajectories (dark → light: denoising progress; ★ final state)  
(a) Full velocity  
![](images/5ba3255ac291c05eeedbb1ef9a78eab13136650085a29989fb871257319211eb.jpg)

(b) Low frequency B<sub>0</sub>  
![](images/b0c19688aaa6ce6bababb0a6aad76dd5a7a4b88bcf04148b5b8d24fe5341f112.jpg)

(c) Mid frequency B<sub>3</sub>  
![](images/47ebf51dac846103f560c4cfec624e3659f5fe7624472335d48aaeceaa67ef7d.jpg)

(d) High frequency B<sub>6</sub>  
![](images/31891873543184b8c47e631dcde685564f963ff43272e83973c0318d8d54a0a1.jpg)

(a) Full velocity  
Qwen-Image: PC1–PC2 trajectories (dark → light: denoising progress; ★ final state; labels: steps)  
![](images/5674cc6b1d8638dff97eae22297357e53ff8af18440af707b0286f08d81d00ee.jpg)

![](images/49ee2a8f69b5fc71999313066fbea3ccc6f3babf36b673243d615da94e912e0e.jpg)  
Normalized PC1

(c) Mid frequency B<sub>3</sub>  
![](images/a16b0b1f8bb5b4b4dcf0746a54e3bcd4f50cf1e962a3ca4f200e447a65cdf892.jpg)

(d) High frequency B<sub>6</sub>  
![](images/709d47a85092988edb2afdc14b047c4554212de100ddf34b524588ba73aa7154.jpg)

Figure 19: Individual Qwen-Image trajectories: prompts 004 and 005. Top two rows: “a futuristic city skyline at sunset with flying cars” (seed 4). Bottom two rows: “a close-up portrait of an old fisherman with a weathered face” (seed 5). For each prompt, the same 50 states are shown in 3D with the denoising step as the vertical axis and in the PC1–PC2 plane. Columns show full velocity, $B _ { 0 } , B _ { 3 }$ , and $B _ { 6 } ;$ dark to light blue indicates denoising progress and stars mark the final state. Each panel uses its own PCA basis and scaling, so positions and sizes cannot be compared across panels or prompts.