# Residual Kalman Dynamics for Event-Based UAV Forecasting

Per Nyblom , Hannes Ovrén , and David Gustafsson

Swedish Defence Research Agency (FOI), Linköping, Sweden {per.nyblom, hannes.ovren, david.gustafsson}@foi.se

Abstract. We study short- and mid-horizon UAV bounding-box forecasting on the FRED event-camera dataset. We use a constant-velocity Kalman filter over a full center-size box state as a strong physical baseline, and train a residual model to predict acceleration-like corrections from recent box history, filtered state features, and local event representations. This simple residual formulation consistently improves over the Kalman baseline, with event-conditioned models giving the strongest results among the evaluated methods. We further show that part of the residual target is predictable from anchor position and velocity alone, indicating that canonical FRED results can reflect both visual evidence and dataset-specific motion priors. To analyze this efect, we introduce decorrelated subsets as a diagnostic stress test, showing that eventconditioned residual models retain useful predictive signal even when measured position- and velocity-based shortcuts are weakened.

Keywords: Event-based vision · UAV trajectory forecasting · Residual Kalman filtering · Shortcut learning

## 1 Introduction

Unmanned aerial vehicles (UAVs) are increasingly used in civilian, industrial, and security-critical settings, making reliable perception of small, fast-moving aerial objects important for detection, tracking, and motion anticipation [10]. Recent incidents and conflicts have further underscored the need for robust UAV and counter-UAV perception systems [1, 2, 13]. Short-horizon UAV trajectory forecasting is therefore relevant for systems that must anticipate rapid aerial maneuvers under low latency.

Using traditional RGB cameras for UAV trajectory forecasting [3, 14, 15] is problematic due to the high agility of the targets together with the relatively low frame rate of conventional cameras. Event cameras [5] are a compelling alternative and have previously been used with success for e.g. UAV tracking [8]. Unlike traditional cameras, event cameras detect changes in light intensity and output these asynchronously at microsecond temporal resolution with low latency and high dynamic range. This means that an event camera can detect and track UAVs under low contrast, during aggressive maneuvers, and even measure the rotation speed of their propellers [7, 11, 12].

![](images/eedbeb5e563cbb6b0cdcb44b71290aea21e8b8492f8d4134456a4a2e93ce93a7.jpg)

![](images/cf48f06743ae1f078cde3d7cfa110a832555839111ec64845ca343b898e2f7c2.jpg)

![](images/22f71e91a9c44be6126d3f7dd38e2e4082e94ed092aeeadc44124aa8de8d8cc6.jpg)

![](images/91691c56d2d7fb74b8bdc19d41d42caba5b1ee546589a3179c1e3546e67ccc2c.jpg)  
Fig. 1: Qualitative examples from the FRED dataset showing the efect of accelerationaware residual forecasting. Observed history boxes are shown in blue, future groundtruth boxes in green, constant-velocity Kalman predictions in cyan, and residual-model predictions in yellow. The constant-velocity model extrapolates the recent motion direction and can deviate substantially when the UAV accelerates or turns, whereas the residual model better follows the future trajectory by learning acceleration-like corrections to the Kalman prediction.

Forecasting UAV trajectories using event cameras is, however, a relatively new research direction. Liang et al. [6] developed a method that fused RGB images with events, however, their events were simulated from low frame rate RGB images that cannot represent the temporal characteristics of a UAV observed by a real event camera. To the best of our knowledge, there are only two examples of previous work that use real event camera data for UAV trajectory forecasting [7,9], and we choose to focus on these works here. For an overview on tracking and detection of drones using event cameras we instead refer the reader to a recently published survey article [8].

An enabling factor for event-based UAV trajectory forecasting was the release of The Florence RGB-Event Drone Dataset (FRED) [9] in 2025, which contains seven hours of recorded UAV flights with diferent platforms. The dataset is annotated with bounding boxes, and is geared towards UAV detection, tracking, and trajectory forecasting. As part of their work on the FRED dataset, Magrini et al. [9] also developed and benchmarked five baseline forecasting methods with diferent architectures and input modalities (RGB images, events, or both). In their experiments they show that using events gives a slight edge over using RGB images, however, the improvement is marginal compared to the Transformer that is trained on only the bounding boxes.

Hari Prasanth et al. [7] propose a method that does not involve any machine learning. Instead, they use a Kalman filter with a constant velocity motion model which is updated using position and velocity measurements extracted from the previous bounding box centers, and the forecasted trajectory is predicted by the filter (omitting measurements). They incorporate event camera data by using it to estimate the RPM of the UAV propellers, and then use this to modulate the process noise covariance matrix of the Kalman filter. The intuition is that a high RPM indicates aggressive flight patterns, which means that the filter update should place less weight on the motion model, and more on the measurements. Evaluation on the FRED dataset shows a significant improvement over the previous state of the art [9], and that the addition of event-based RPM-modulated process noise gives a small improvement over a standard Kalman filter. Interestingly, they show that simple linear extrapolation of bounding-box centers is almost as good as using the Kalman-filter approach.

Following Hari Prasanth et al. [7] , we use a constant-velocity Kalman model as a simple and directly comparable physical prior. Rather than increasing the order of the dynamics alone, we learn residual acceleration corrections from box history, filtered state features, and event evidence. In many applications, being able to estimate target size is also important, and because of this we predict full bounding boxes instead of only box centers. We also do not modulate the process noise using RPM estimation. Instead, our approach uses the Kalman filter as a reliable physical prior, but improves the prediction by also training a machinelearning model to predict acceleration-like corrections to the predicted filter states. This formulation separates a reliable physical prior from the remaining predictable structure in the data.

The benefits of this approach are illustrated in Figure 1, which shows four FRED sequences where a constant-velocity Kalman filter fails to anticipate changes in the UAV motion. In these examples, the predicted boxes continue along the recent velocity direction, whereas the ground-truth future trajectory bends due to acceleration. A residual model that predicts acceleration-like corrections can therefore better follow the future trajectory while retaining the stability of the Kalman prior.

A central dificulty in evaluating short-horizon UAV forecasting is that good performance need not imply that a model has learned to use target appearance or event evidence. Depending on how the dataset was recorded in terms of e.g. camera viewpoint and UAV flight patterns, the future motion of a target can be partially predictable from its current image location and velocity alone. For example, targets near the image boundary may be more likely to turn back toward the center of the field of view, and high-speed trajectories may be more likely to decelerate. Such regularities are useful for in-distribution forecasting, but they can also act as motion-prior shortcuts that obscure whether an eventconditioned model is actually exploiting visual evidence in contrast to simply learning dataset-specific trajectory statistics.

This motivates a diagnostic question: how much of the residual forecasting problem can be explained without looking at the event data? We address this by measuring how well estimated future accelerations can be predicted from simple anchor features such as image position and velocity, and propose a diagnostic that checks whether the same modeling choices remain useful when these simple motion priors are weakened.

To construct this diagnostic, we introduce a sample-level decorrelation procedure that greedily removes forecast windows contributing most to the predictability of future acceleration from anchor position and velocity. We use the resulting decorrelated subsets as a stress test for motion-prior dependence. Comparing models before and after this procedure helps identify which gains persist when position- and velocity-based shortcuts are reduced.

In summary, our contributions are

– a residual Kalman forecasting model (see Sec. 3.3) that predicts acceleration corrections on top of a full center-size box state, improving both short- and mid-horizon UAV bounding-box forecasts on FRED.

– an empirical diagnostic (see Sec. 4.3) showing that future acceleration in FRED is partially predictable from simple motion-prior features such as anchor position and velocity.

– a greedy sample decorrelation procedure (see Sec. 3.6) used as a stress test for motion-prior dependence, enabling a more cautious analysis of whether learned gains persist when position- and velocity-based shortcuts are weakened.

## 2 Problem Setup

## 2.1 State and Forecast Target

Each sample is anchored at label time $t _ { 0 }$ . The observed history contains boxes up to and including $t _ { 0 } .$ , and the forecasting target is the sequence of future boxes after $t _ { 0 }$ . We represent a bounding box

$$
\mathbf b _ { t } = [ x _ { t } , y _ { t } , w _ { t } , h _ { t } ] ^ { \top } \in [ 0 , 1 ] ^ { 4 } ,\tag{1}
$$

at time t in normalized center-size coordinates where $\left( x _ { t } , y _ { t } \right)$ is the box center and $\left( w _ { t } , h _ { t } \right)$ are its width and height. All coordinates are normalized by the image width and height.

The Kalman state at time $t ,$

$$
\mathbf { x } _ { t } = [ x _ { t } , y _ { t } , w _ { t } , h _ { t } , v _ { t } ^ { x } , v _ { t } ^ { y } , v _ { t } ^ { w } , v _ { t } ^ { h } ] ^ { \top } ,\tag{2}
$$

augments the box with first-order velocities for all four box channels,

This full-center state allows the baseline and residual models to forecast bounding boxes rather than only center trajectories. We use a 400 ms history window and evaluate both short-horizon 400 ms and mid-horizon 800 ms forecasts, following the stated forecast protocol in Magrini et al. [9] .

![](images/fb300bbf2a5c2ee98b883513e78685b449ddd032d2c02936478883fd0a694b03.jpg)  
Fig. 2: Event-derived input representations used by the image-conditioned residual models. Left: the event-frame representation provided with FRED. Right: the same event sequence rendered as a compact spatio-temporal representation (CSTR) [4].

## 2.2 Inputs

We evaluate three families of input features. First, we use event-derived image representations extracted from the event stream immediately preceding the anchor time, $t _ { 0 }$ . Shown in Fig. 2 are the two selected spatio-temporal representations: CSTR [4], and event frames provided by the FRED dataset. CSTR provides a compact three-channel rendering of recent event activity, while event frames use a per-polarity binary representation. Second, we use recent box history, represented as normalized center-size boxes with relative timestamps. Third, we use features derived from the optimized Kalman filter state, including filtered position and velocity estimates.

To reduce direct leakage of global image position into image-conditioned models, the main experiments use fixed spatial cutouts centered on the final observed box. Inputs are resized to (640 × 360), and (64 × 64) cutouts are extracted around the anchor box, corresponding to (128×128) pixels at the original FRED resolution. The history encoder uses relative box-history features in the main setting, where positions are expressed with respect to the final observed box. Absolute position is therefore available only through explicitly controlled filter-state features, making it possible to separate local event evidence from global motion-prior information.

## 2.3 Metrics

Adhering to the evaluation protocol of the FRED dataset, the predictions are evaluated against future boxes using the Average Displacement Error

$$
\mathrm { A D E } _ { \mathrm { c e n t e r } } = \frac { 1 } { T } \sum _ { k = 1 } ^ { T } \left. \hat { \mathbf { c } } _ { t _ { k } } - \mathbf { c } _ { t _ { k } } \right. _ { 2 } ,\tag{3}
$$

and the Final Displacement Error

$$
\mathrm { F D E } _ { \mathrm { c e n t e r } } = \left. \hat { \mathbf { c } } _ { t _ { T } } - \mathbf { c } _ { t _ { T } } \right. _ { 2 } ,\tag{4}
$$

reported in pixels. Here $\mathbf { c } _ { t } = [ x _ { t } , y _ { t } ]$ and cˆ is the box center, and its estimate, respectively, at time t. The time $t _ { T }$ refers to the time of the last future prediction. We also report bounding-box ADE/FDE in pixels and mean IoU over the forecast horizon. Center metrics measure trajectory accuracy, whereas box metrics and mIoU additionally capture changes in apparent target size.

## 2.4 Training and Test Split Policy

We use two diferent splits of the FRED dataset for evaluation. The first is the original, canonical split, and the second is a decorrelated subset.

For the canonical-split experiments, models are trained on the FRED training split and evaluated on the canonical FRED test split. During development, we reserve 20% of the training recordings at the folder level for checkpoint selection and training evaluation. This avoids mixing windows from the same recording across training and validation. To keep experiments comparable and computationally manageable, training samples are capped at 80k and train-evaluation samples at 20k unless otherwise stated. The selected checkpoint is then evaluated on the full test split.

The decorrelated experiments use the same train/test protocol, but apply the sample-level decorrelation procedure introduced in Sec. 3.6 separately to the relevant sample pools. First we select 80k samples from the canonical training split, and decorrelation then removes 50% of these, resulting in 40k samples. $\mathrm { A }$ second set of 40k samples are then sampled uniformly from the first 80k samples which gives a dataset of the same size but with the same statistics as the canonical dataset split.

## 3 Method

This section describes our residual Kalman filter, two baselines that we compare against, and the decorrelation procedure required for motion prior diagnostics.

## 3.1 Optimized Kalman Baselines

We use a constant-velocity Kalman filter as the physical forecasting baseline and as the starting point for the residual models. The filter operates on the full center-size box state in Eq. (1) where the first four components, $[ x _ { t } , y _ { t } , w _ { t } , h _ { t } ]$ are the normalized bounding-box coordinates and the last four, $[ v _ { t } ^ { x } , v _ { t } ^ { y } , v _ { t } ^ { w } , v _ { t } ^ { h } ] .$ , are their corresponding velocities. This can be viewed as four independent constantvelocity models, one for each box channel $( x , y , w , h )$

The filter directly observes the normalized box position and size, while the corresponding velocities are inferred from the observed history.

The filter is initialized fresh for each sample, filters only the observed history window, and then propagates forward without using future measurements. Initial, process, and measurement standard deviations are optimized on the training split. The primary optimized baseline uses diferentiable backpropagation through the filter with log-space standard-deviation parameters and a weighted objective

$$
J = \mathrm { F D E } _ { \mathrm { c e n t e r } } + \lambda _ { \mathrm { a d e } } \mathrm { A D E } _ { \mathrm { c e n t e r } } - \lambda _ { \mathrm { i o u } } \mathrm { m I o U } ,\tag{5}
$$

where we used $\lambda _ { \mathrm { a d e } } = 0 . 2 5$ and $\lambda _ { \mathrm { i o u } } = 1 0 0$ during training. The resulting baseline is therefore not intended as a weak reference model, but as a carefully tuned constant-velocity prior. This is important for the residual formulation: learned models should improve over a competitive motion model rather than over an under-optimized extrapolator.

As an additional baseline, we also optimize an otherwise matched constantacceleration Kalman filter.

## 3.2 Last-Four Linear Extrapolator

For comparison, we also report a linear extrapolator, intended to replicate the one used by Hari Prasanth et al. [7] but extended to support full bounding boxes. It estimates velocity for each box channel by a least-squares line fit over the last four observed boxes and then propagates forward with constant velocity:

$$
\hat { \mathbf { b } } _ { t _ { k } } = \mathbf { b } _ { t _ { 0 } } + \hat { \mathbf { v } } ( t _ { k } - t _ { 0 } ) .\tag{6}
$$

This baseline is very simple, but useful for interpreting whether the optimized Kalman filter mainly improves center extrapolation, box-size extrapolation, or both.

## 3.3 Residual Kalman Forecaster

The learned model predicts acceleration residuals rather than boxes directly. Starting at the final filtered Kalman state the residual head predicts the acceleration at forecast step $k _ { ; }$

$$
\begin{array} { r } { \mathbf { a } _ { k } = f _ { \theta } ( \phi _ { \mathcal { R } } , \phi _ { h } , \mathbf { x } _ { k } , \varDelta t _ { k } ) , } \end{array}\tag{7}
$$

where $\phi _ { \mathcal { R } }$ is the event representation feature and $\phi _ { h }$ encodes recent box history. The rollout is

$$
\mathbf b _ { k + 1 } = \mathbf b _ { k } + \mathbf v _ { k } \varDelta t _ { k } + \frac 1 2 \mathbf a _ { k } \varDelta t _ { k } ^ { 2 } ,\tag{8}
$$

$$
\mathbf { v } _ { k + 1 } = \mathbf { v } _ { k } + \mathbf { a } _ { k } \varDelta t _ { k } .\tag{9}
$$

The predicted box channels are clamped to [0, 1].

This formulation separates a stable physical prior from learned residual structure. The Kalman filter explains the dominant constant-velocity component, while the learned model is responsible only for the remaining acceleration-like correction. This also makes the diagnostic analysis more direct: if future acceleration can be predicted from position and velocity alone, then part of the residual task can be solved without event evidence.

## 3.4 Network Structure

The image encoder is a ResNet-18-style convolutional backbone with three input channels and a 128-dimensional pooled output. The history encoder receives the final $N _ { h } { = } 1 2$ history samples. Each sample contributes four normalized box values and one relative timestamp, giving a 12×5 input. In the leakage-reduced setting, box positions are expressed relative to the final history box. The box input sequence is flattened and mapped by an MLP to a fixed-dimensional history feature $\phi _ { h }$

The residual head, which is a two-layer MLP with hidden dimension 256, receives the concatenation of the image feature, history feature, current 8-D Kalman rollout state, and scalar time step. It outputs four acceleration channels corresponding to the center and size components of the box. Models without image input use the same residual-rollout structure but omit the image feature, allowing us to isolate the contribution of event-derived evidence.

## 3.5 Motion-Prior Residual Baselines and Correlation Analysis

To estimate how much of the residual forecasting problem can be solved without image evidence and box history, we include a linear residual baseline that uses the same acceleration-rollout formulation as the learned residual forecaster, but replaces the neural residual head with a single linear map from filtered center position and velocity to acceleration. Strong performance from this baseline indicates that part of the future acceleration signal is predictable from simple motion-prior features rather than from event evidence.

We further quantify this efect by fitting a constant-acceleration curve to the center positions over each history-plus-future sample window,

$$
\mathbf { c } ( t ) \approx \mathbf { c } _ { 0 } + \mathbf { v } _ { 0 } ( t - t _ { 0 } ) + \frac { 1 } { 2 } \mathbf { a } ( t - t _ { 0 } ) ^ { 2 } .\tag{10}
$$

This $_ \mathrm { y }$ ields an anchor position $\mathbf { c } _ { 0 } ,$ anchor velocity v0, and fitted future acceleration $\mathbf { a } = [ a _ { x } , a _ { y } ] ^ { \top }$ . We then measure how predictable a is from motion-prior features,

$$
\begin{array} { r } { { \bf x } _ { p r i o r } = [ c _ { 0 } ^ { x } - 0 . 5 , c _ { 0 } ^ { y } - 0 . 5 , v _ { 0 } ^ { x } , v _ { 0 } ^ { y } , | { \bf c } _ { 0 } - [ 0 . 5 , 0 . 5 ] | _ { 2 } , | { \bf v } _ { 0 } | _ { 2 } ] . } \end{array}\tag{11}
$$

We report mean absolute Pearson correlation between these prior features and fitted acceleration, as well as ridge-linear $R ^ { 2 }$ . High predictability indicates that the residual target contains structure that can be explained from anchor position and velocity alone. Such structure may be useful for in-distribution forecasting, but it can also mask whether event-conditioned models are exploiting visual evidence.

## 3.6 Greedy Decorrelation

The decorrelation method drops samples. For a candidate subset S, suficient statistics are accumulated for standardized prior features $X _ { S }$ and fitted accelerations $Y _ { S }$ . The score combines mean absolute Pearson correlation and ridge-linear

predictability:

$$
\operatorname { s c o r e } ( S ) = \alpha \cdot \operatorname { m e a n } \left( \left| \operatorname { c o r r } ( X _ { S } , Y _ { S } ) \right| \right) + \beta \cdot \operatorname { m e a n } \left( \operatorname { m a x } ( R ^ { 2 } , 0 ) \right)\tag{12}
$$

At each step, the algorithm samples a set of removable candidates and removes the example whose removal most reduces the score. The process stops when the target fraction is reached. In our decorrelated experiments, we keep 50% of the samples.

The procedure removes only the measured linear relationship between selected prior features and fitted acceleration, and it does so by dropping samples rather than by reweighting the data-generating process. Nonlinear priors, unmeasured scene statistics, and other forms of shortcut information may remain. We therefore use the decorrelated subsets as a stress test: a model whose gains persist after decorrelation is less dependent on these measured position- and velocity-based shortcuts.

## 4 Experiments

## 4.1 Protocol

All learned models use the optimized Kalman filter as the starting state for forecasting. We compare the Kalman baselines, the last-four linear extrapolator, linear residual baselines, non-image residual models, and image-conditioned residual models. Unless otherwise stated, models use a 400 ms history window, either a 400 ms or 800 ms forecast horizon, normalized boxes, AdamW optimization with learning rate $1 0 ^ { - 3 }$ , batch size 128, weight decay $1 0 ^ { - 4 }$ , Smooth L1 loss, and gradient clipping at norm 10.

The main image-conditioned experiments use fixed target-centered cutouts, relative box-history features, filtered state features, and ResNet-18 encoders. We evaluate event frames and CSTR to compare two event-derived representations, and we include non-image variants to separate the contribution of event evidence from the contribution of box history and Kalman state features.

## 4.2 Canonical Split Results

Table 1 reports results on the canonical FRED split. The optimized constantvelocity and constant-acceleration Kalman filters are strong baselines, especially for bounding-box prediction. The constant-acceleration filter performs slightly better across all reported metrics. Compared with the last-four linear extrapolator, the Kalman filters give similar center performance but substantially better bounding-box ADE/FDE and mIoU, indicating that filtering the full center-size state is important for stable box forecasts.

Residual acceleration models improve over the Kalman baselines across both forecast horizons. The linear residual model using filtered center position and velocity already improves center accuracy, showing that some future acceleration is predictable from simple non-image motion features. The MLP residual model using boxes and filter-state features improves further, and adding event-derived image representations gives the best results. This supports the residual formulation: the constant-velocity Kalman filter captures the dominant constant-velocity component, while learned residuals capture systematic acceleration corrections.

<table><tr><td></td><td></td><td colspan="3">Short-term, 0.4s</td><td colspan="3">Mid-term, 0.8s</td></tr><tr><td>Method</td><td>Reference</td><td>Inputs</td><td>(ADE/FDE)BB</td><td>(ADE/FDE)c mIoU (ADE/FDE)BB (ADE/FDE)c mIoU</td><td></td><td></td><td></td><td></td></tr><tr><td>CV Opt. Kalman</td><td>Our</td><td>Boxes</td><td>22.24/39.89</td><td>19.12/37.25</td><td>0.475</td><td>45.81/96.80</td><td>43.10/94.67</td><td>0.315</td></tr><tr><td>CA Opt. Kalman</td><td>Our</td><td>Boxes</td><td>22.07/39.37</td><td>18.91/36.69</td><td>0.480</td><td>45.25/95.61</td><td>42.52/93.51</td><td>0.316</td></tr><tr><td>CV Lin. Approx.</td><td>Our</td><td>Last 4 Boxes</td><td>31.00/54.35</td><td>20.43/39.38</td><td>0.408</td><td>59.32/117.81</td><td>45.13/97.80</td><td>0.264</td></tr><tr><td>Lin. Regr.→Acc</td><td>Our</td><td>FS Center Pos+Vel</td><td>21.14/37.00</td><td>17.96/34.26</td><td>0.468</td><td>41.47/84.16</td><td>38.70/81.93</td><td>0.301</td></tr><tr><td>MLP→Acc</td><td>Our</td><td>Boxes+FS</td><td>18.20/30.96</td><td>14.94/28.08</td><td>0.499</td><td>34.77/70.47</td><td>31.89/68.06</td><td>0.336</td></tr><tr><td>CNN→Acc</td><td>Our</td><td>EF+Boxes+FS</td><td>17.06/28.31</td><td>13.61/25.21</td><td>0.522</td><td>31.75/64.61</td><td>28.62/61.96</td><td>0.361</td></tr><tr><td>CNN→Acc</td><td>Our</td><td>CSTR+Boxes+FS</td><td>16.70/27.92</td><td>13.22/24.72</td><td>0.531</td><td>31.92/65.03</td><td>28.84/62.50</td><td>0.356</td></tr><tr><td>CNN+Transformer</td><td>[9]</td><td>EF+Boxes</td><td>48.46/66.67</td><td>124.00/45.921</td><td>0.284</td><td>72.85/119.60</td><td>280.90/83.861</td><td>0.206</td></tr><tr><td>CNN+Transformer</td><td>[9]</td><td>EF+RGB+Boxes</td><td>47.49/65.89</td><td>121.30/45.23 1</td><td>0.279</td><td>72.81/122.50</td><td>281.50/85.78</td><td>0.201</td></tr><tr><td>CV Kalman</td><td>[7]</td><td>Box Centers+RPM</td><td>-/-</td><td>15.35/34.85</td><td></td><td>-1-</td><td>31.16/77.76</td><td></td></tr></table>

Table 1: Canonical FRED split results. Proposed methods use the optimized full-box constant-velocity Kalman filter as the forecasting prior and, when applicable, learn acceleration residuals on top of it. Metrics are reported for 400 ms and 800 ms horizons; lower ADE/FDE and higher mIoU are better. Best values are highlighted. Comparisons to previously reported methods are included for context but difer in evaluation details or training splits. FS and EF refers to Kalman filter state, and event frames, respectively.

The diference between event frames and CSTR is small on the canonical split. CSTR gives the best short-horizon results, while the two event representations are close at the mid horizon. This suggests that, under the canonical evaluation, the main gain comes from combining residual learning with event-conditioned local evidence rather than from a decisive advantage of one event representation.

The comparison to previously reported FRED and RPM-modulated Kalman results should be interpreted cautiously because of diferences in evaluation details, splits, and forecast targets. Nevertheless, the results show that a carefully optimized full-box Kalman baseline plus learned acceleration residuals is highly competitive and substantially stronger than direct learned forecasting baselines reported for FRED.

## 4.3 Motion-Prior Correlation

The canonical results show that position- and velocity-only residual models perform surprisingly well. We therefore analyze whether fitted future acceleration is predictable from anchor position and velocity. Table 2 reports mean absolute Pearson correlation and ridge-linear $R ^ { 2 }$ from motion-prior features to fitted center acceleration before and after decorrelation using the method in Sec. 3.6.

Before decorrelation, both the training and test splits show non-negligible predictability. This confirms that the residual target is not purely visual: part of the future acceleration field can be explained from where the UAV is in the image and how it is moving at the anchor time. Figure 3 visualizes this effect. The position-to-acceleration field shows edge-to-center tendencies, while the velocity-to-acceleration field shows velocity-dependent deceleration patterns. These efects are consistent with motion-prior shortcuts: a model may learn that objects near certain image regions or moving at certain speeds tend to accelerate in predictable directions.

<table><tr><td>Before Split Mean  $| \rho |$  Mean  $R ^ { 2 }$ </td><td>After Mean  $| \rho |$  Mean  $R ^ { 2 }$ </td></tr><tr><td>Train 0.128 0.244</td><td>0.008 0.003</td></tr><tr><td>Test 0.131 0.252</td><td>0.007 0.003</td></tr></table>

Table 2: Motion-prior predictability of fitted future acceleration before and after sample decorrelation.

After greedy decorrelation, the measured correlations and linear predictability are greatly reduced. This supports the use of the decorrelated subsets as a stricter diagnostic setting. The decorrelated subsets do not remove all possible priors, but they substantially weaken the measured position- and velocity-based shortcuts targeted by the procedure.

## 4.4 Decorrelated-Subset Results

Tables 3 and 4 evaluate how models behave when the motion-prior shortcuts measured in Section 4.3 are weakened. Table 3 shows results when models are trained on decorrelated training samples and evaluated on the canonical test split. Performance generally degrades relative to training on the full, canonical, training split, which is expected: the decorrelated training data removes part of the structure that remains present in the canonical test data. This result should not be interpreted as a failure of decorrelation, but as evidence that the canonical split contains motion-prior regularities that are useful when evaluated on the canonical test split.

Table 4 instead shows results when models are trained on the canonical training split and evaluated on the decorrelated test subset. In this setting, models trained on the canonical data often perform worse than models trained on decorrelated data, especially for the 800 ms horizon. This suggests a train-test mismatch: models trained on the canonical split can benefit from motion-prior correlations that are weakened in the decorrelated test set. Training on decorrelated data reduces this dependence and can improve relative robustness under the diagnostic stress test.

Across decorrelated settings, event-conditioned models remain stronger than non-image residual models. The gap is not eliminated by decorrelation, indicating that local event evidence and box history still provide useful information beyond the measured position- and velocity-based priors. CSTR seems to be more stable than event frames and consistently yields better results under non-matching train/test decorrelation conditions, although the diference is modest. Overall, the decorrelated experiments support the central claim: canonical-split gains should be interpreted cautiously, but residual event-conditioned models retain useful predictive signal even when simple motion-prior shortcuts are reduced.

Center velocity to acceleration field, n=394395, grid=8x8  
![](images/c30d9c6572f1c606d5bfac2945a75b2d03df80584060a4ad4575eeb30e6c0675.jpg)

(a) Box center position to acceleration.  
![](images/eef352bd95b95fdb44e746247509c024ca6231b55dbcdebc87167d63dbf17e30.jpg)  
(b) Estimated velocity to acceleration.  
Fig. 3: Motion-prior structure in FRED for the canonical train split. Samples are binned by anchor box-center position in (a) and estimated anchor velocity in (b). Arrows show the mean fitted future center acceleration in each bin, with direction encoding acceleration direction and length/color encoding magnitude. Cell shading indicates sample count. The fields reveal edge-to-center and velocity-dependent deceleration patterns, showing that future acceleration is partly predictable from position and velocity alone.

<table><tr><td>Horizon</td><td>Inputs</td><td>ADEBB</td><td>FDEBB</td><td>ADEc</td><td>FDEc</td><td>mIoU</td></tr><tr><td rowspan="4"></td><td>Short (400 ms) FS Center Pos+Vel 21.50 (+1.7%)</td><td></td><td>37.92 (+2.5%)</td><td>18.28 (+1.8%)</td><td>35.03 (+2.2%)</td><td>0.475 (+1.5%)</td></tr><tr><td>Boxes+FS</td><td>19.15 (+5.2%)</td><td>33.50 (+8.2%)</td><td>15.92 (+6.6%)</td><td>30.59 (+8.9%)</td><td>0.485 (-2.8%)</td></tr><tr><td>EF+Boxes+FS</td><td>17.30 (+1.4%)</td><td>29.42 (+3.9%)</td><td>13.81 (+1.5%)</td><td>26.23 (+4.0%)</td><td>0.523 (+0.2%)</td></tr><tr><td>CSTR+Boxes+FS</td><td>17.00 (+1.8%)</td><td>28.67 (+2.7%)</td><td>13.51 (+2.2%)</td><td>25.44 (+2.9%)</td><td>0.529 (-0.4%)</td></tr><tr><td rowspan="4">Mid (800 ms)</td><td>FS Center Pos+Vel 43.20 (+4.2%)</td><td></td><td>89.32 (+6.1%)</td><td>40.36 (+4.3%)</td><td>86.90 (+6.1%)</td><td>0.320 (+6.3%)</td></tr><tr><td>Boxes+FS</td><td>38.89 (+11.8%)</td><td>83.20 (+18.1%)</td><td>36.04 (+13.0%)</td><td>80.73 (+18.6%)</td><td>0.321 (-4.5%)</td></tr><tr><td>EF+Boxes+FS</td><td>33.50 (+5.5%)</td><td>70.21 (+8.7%)</td><td>30.35 (+6.0%)</td><td>67.44 (+8.8%)</td><td>0.356 (-1.4%)</td></tr><tr><td>CSTR+Boxes+FS</td><td>33.31 (+4.4%)</td><td>70.07 (+7.8%)</td><td>30.16 (+4.6%)</td><td>67.40 (+7.8%)</td><td>0.358 (+0.6%)</td></tr></table>

Table 3: Efect of training-set decorrelation when evaluating on the canonical FRED test split. All models are trained on 50% of the available training samples. The decorrelated setting uses the greedy decorrelation procedure, while the reference setting uses a random 50% subset. Percentages show relative change with respect to the corresponding model trained on the random subset. Positive values indicate higher error for ADE/FDE and lower IoU for mIoU. FS and EF refer to Kalman filter state, and event frames, respectively.

<table><tr><td>Horizon</td><td>Inputs</td><td>ADEBB</td><td>FDEBB</td><td>ADEc</td><td>FDEc</td><td>mIoU</td></tr><tr><td rowspan="4"></td><td>Short (400 ms) FS Center Pos+Vel 24.64 (+3.1%)</td><td></td><td>43.76 (+4.7%)</td><td>21.40 (+4.3%)</td><td>41.03 (+5.9%)</td><td>0.434 (-5.4%)</td></tr><tr><td>Boxes+FS</td><td>21.01 (+0.8%)</td><td>36.08 (+0.8%)</td><td>17.62 (+1.1%)</td><td>33.08 (+1.2%)</td><td>0.473 (-0.8%)</td></tr><tr><td>EF+Boxes+FS</td><td>19.33 (-1.0%)</td><td>32.38 (-1.3%)</td><td>15.73 (-0.9%)</td><td>29.16 (-0.7%)</td><td>0.502 (+0.6%)</td></tr><tr><td>CSTR+Boxes+FS</td><td>19.15 (-0.3%)</td><td>32.07 (-0.1%)</td><td>15.49 (-0.4%)</td><td>28.77 (+0.1%)</td><td>0.507 (+0.0%)</td></tr><tr><td rowspan="4">Mid (800 ms)</td><td>FS Center Pos+Vel 45.23 (+5.8%)</td><td></td><td>91.55 (+7.1%)</td><td>42.56 (+6.9%)</td><td>89.50 (+8.0%)</td><td>0.283 (-12.4%)</td></tr><tr><td>Boxes+FS</td><td>37.99 (+2.4%)</td><td>76.92 (+1.4%)</td><td>35.16 (+2.9%)</td><td>74.56 (+1.9%)</td><td>0.322 (-4.5%)</td></tr><tr><td>EF+Boxes+FS</td><td>33.91 (+1.4%)</td><td>68.97 (+2.7%)</td><td>30.85 (+1.8%)</td><td>66.44 (+3.2%)</td><td>0.353 (-2.2%)</td></tr><tr><td>CSTR+Boxes+FS</td><td>33.83 (+1.4%)</td><td>68.51 (+0.5%)</td><td>30.72 (+1.7%)</td><td>65.87 (+0.6%)</td><td>0.355 (-2.2%)</td></tr></table>

Table 4: Efect of test-set decorrelation when evaluating on the decorrelated FRED test subset. Models trained on the canonical training subset are compared with the corresponding models trained on the decorrelated training subset. Percentages show relative change with respect to evaluation on the canonical test subset for the same model and horizon. Positive values indicate higher error for ADE/FDE and lower IoU for mIoU. FS and EF refer to Kalman filter state, and event frames, respectively.

## 5 Discussion

The experiments highlight both the strength and the risk of residual forecasting on FRED. The CV and CA Kalman filters over the full center-size state are strong, interpretable baselines, and learned acceleration residuals consistently improve over them. This supports the idea that short-horizon UAV forecasting benefits from combining a stable physical prior with learned corrections.

At the same time, the linear residual baselines and correlation analysis show that future acceleration is partially predictable from anchor position and velocity, meaning that canonical-split improvements may reflect both event-conditioned evidence and dataset-specific motion priors.

The decorrelation procedure is therefore best understood as a diagnostic stress test rather than a new benchmark or evidence of out-of-distribution robustness. It weakens selected linear relationships between motion-prior features and fitted acceleration by removing samples, but it does not change the underlying data collection process or eliminate nonlinear priors, scene-dependent cues, target-scale efects, or other shortcuts. Within these limitations, the stress test helps separate gains that depend strongly on measured position- and velocitybased shortcuts from gains that persist when those shortcuts are reduced.

Even so, the stress test is informative. Event-conditioned residual models still perform well after decorrelation, while models relying more directly on position and velocity are more afected. This suggests that event evidence contributes useful information, but also that part of the canonical-split improvement comes from motion-prior structure. Reporting both canonical and decorrelated-subset results gives a more cautious and transparent picture of model behavior.

There are several limitations. The experiments are restricted to short- and mid-horizon box forecasting on FRED, and the decorrelation analysis focuses on fitted center acceleration rather than full box dynamics. The residual model produces a single deterministic trajectory and does not model forecast uncertainty, such as multiple distinct plausible maneuvers. Finally, the event representations are evaluated through relatively compact encoders and target-centered cutouts; more specialized event architectures may extract stronger visual evidence.

## 6 Conclusion

We presented a residual Kalman approach for short-horizon UAV boundingbox forecasting that achieves strong results on FRED and improves over our optimized full-box Kalman baselines, while comparing favorably to previously reported methods under non-identical protocols.

The method extends center-only trajectory forecasting to a full center-size Kalman state and learns acceleration-like residuals from event representations, box history, and filtered state features. This combination gives a strong forecasting model: the Kalman filter provides a stable first-order physical prior, while the residual model learns data-dependent acceleration and turning corrections beyond fixed parametric dynamics.

We also introduced a diagnostic analysis for motion-prior dependence. By measuring how well fitted future acceleration can be predicted from anchor position and velocity, we showed that part of the residual forecasting target is explainable without event evidence. A greedy sample decorrelation procedure reduces this predictability and provides a stricter stress test for learned residual models. The resulting experiments suggest that event-conditioned residuals remain useful when measured position- and velocity-based shortcuts are weakened, but they also show that canonical-split gains should be interpreted with caution.

## References

1. Heathrow airport: Drone sighting halts departures. BBC (2019), https://bbc. com/news/uk-46803713

2. “Army of Drones Bonus” program delivers results: nearly 820,000 russian targets hit in 2025, says Mykhailo Fedorov. Ministry of Defence Ukraine (2026), https: //mod.gov.ua/en/news/army-of-drones-bonus-program-delivers-resultsnearly-820-000-russian-targets-hit-in-2025-says-mykhailo-fedorov

3. Becker, S., Hug, R., Huebner, W., Arens, M., Morris, B.T.: Generating synthetic training data for deep learning-based uav trajectory prediction. In: Proceedings of the 2nd International Conference on Robotics, Computer Vision and Intelligent Systems - ROBOVIS. pp. 13–21. INSTICC, SciTePress (2021). https://doi.org/ 10.5220/0010621400003061

4. El Shair, Z.A., Hassani, A., Rawashdeh, S.A.: CSTR: A Compact Spatio-Temporal Representation for Event-Based Vision. IEEE Access 11, 102899–102916 (2023). https://doi.org/10.1109/ACCESS.2023.3316143

5. Gallego, G., Delbruck, T., Orchard, G., Bartolozzi, C., Taba, B., Censi, A., Leutenegger, S., Davison, A.J., Conradt, J., Daniilidis, K., Scaramuzza, D.: Event-Based Vision: A Survey. IEEE transactions on pattern analysis and machine intelligence 44, 154–180 (2022). https://doi.org/10.1109/TPAMI.2020.3008413, https://ieeexplore.ieee.org/document/9138762/

6. Liang, H., Yuan, S., Liu, F., Yang, Y., Wang, B., Huang, Z., Shi, C., Jin, J.: Labelfree long-horizon 3d uav trajectory prediction via motion-aligned rgb and event cues (2025), https://arxiv.org/abs/2507.03365

7. M., H.P.S., Habibiroudkenar, P., Alamikkotervo, E., Bouzoulas, D., Ojala, R.: Event-only drone trajectory forecasting with RPM-modulated Kalman filtering. arXiv [cs.CV] (2026). https://doi.org/10.48550/arXiv.2603.01997, http: //dx.doi.org/10.48550/arXiv.2603.01997

8. Magrini, G., Berlincioni, L., Becattini, F., Cultrera, L., Pala, P.: Drone detection with event cameras. In: 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). pp. 4762–4773. IEEE (2025). https://doi.org/10. 1109/iccvw69036.2025.00494, http://dx.doi.org/10.1109/iccvw69036.2025. 00494

9. Magrini, G., Marini, N., Becattini, F., Berlincioni, L., Biondi, N., Pala, P., Del Bimbo, A.: FRED: The Florence RGB-event drone dataset. In: Proceedings of the 33rd ACM International conference on multimedia (2025)

10. Sivakumar, M., TYJ, N.M.: A literature survey of unmanned aerial vehicle usage for civil applications. Journal of Aerospace Technology and Management 13, 1–8 (2021). https://doi.org/10.1590/jatm.v13.1233

11. Spetlik, R., Uhrová, T., Matas, J.: Eficient real-time quadcopter propeller detection and attribute estimation with high-resolution event camera, pp. 217–230. Lecture notes in computer science, Springer Nature Switzerland (2025). https: //doi.org/10.1007/978-3-031-95911-0\_16, http://dx.doi.org/10.1007/978- 3-031-95911-0\_16

12. Stewart, T., Drouin, M.A., Picard, M., Billy Djupkep, F., Orth, A., Gagné, G.: Using neuromorphic cameras to track quadcopters. In: Proceedings of the 2023 International Conference on Neuromorphic Systems. pp. 1–5. ACM (2023). https: //doi.org/10.1145/3589737.3605987, http://dx.doi.org/10.1145/3589737. 3605987

13. Wilson, T.: Copenhagen and oslo airports forced to close temporarily due to drone sightings. BBC (2025), https://bbc.com/news/articles/cn4lj1yvgvgo

14. Zhou, S., Yang, L., Liu, X., Wang, L.: Learning short-term spatial–temporal dependency for uav 2-d trajectory forecasting. IEEE Sensors Journal 24(22), 38256– 38269 (2024). https://doi.org/10.1109/JSEN.2024.3466516

15. Zou, L., Wang, J., Liang, R., Wu, H., Chen, K., Wang, Y.: UAV-MM3D: A largescale synthetic benchmark for 3d perception of unmanned aerial vehicles with multi-modal data (2025), https://arxiv.org/abs/2511.22404