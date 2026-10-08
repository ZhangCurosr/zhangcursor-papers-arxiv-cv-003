# SANet: Selective Attention Network for Infrared Small Target Detection

Yingmei Zhang, Wangtao Bao, Qin Xiao, Yong Yang, Senior Member, IEEE, Weiguo Wan, Member, IEEE, Yitao Luo, Xueting Zou, and Lei Zhang

Abstract—Infrared small target detection aims to accurately identify and locate dim targets in complex backgrounds and supports applications such as maritime surveillance and military search and rescue. However, the small size and weak contrast of infrared targets make it difficult to balance detection accuracy and false alarms. This paper proposes a selective attention network (SANet) for infrared small target detection. A dual-path semantic-aware module combines standard and pinwheel-shaped convolutions to preserve local spatial consistency and capture broader contextual information. Spatial and channel attention further refine the features and improve target-background discrimination. To address the limitations of static skip connections in U-Net, a selective attention fusion module adaptively integrates features across scales using spatially varying weights. It selectively enhances salient regions and improves discrimination between true targets and false alarms. Experiments on three public benchmarks, NUAA-SIRST, IRSTD-1K, and NUDT-SIRST, show that SANet achieves competitive performance in intersection over union (IoU), normalized IoU, detection probability, and false alarm rate. Its IoU exceeds that of the second-best method by 1.93, 4.32, and 2.21 percentage points, respectively. These results support the effectiveness of SANet in dim-target perception, discriminative feature representation, and background suppression. The source code is available at https://anonymous.4open.science/r/SANet-E808/ and https://gitcode.com/m0\_61988291/SANet.

Index Terms—Infrared small target detection, dual-path semantic-aware module, pinwheel-shaped convolution, selective attention fusion, spatially varying weights.

## I. INTRODUCTION

Infrared small target detection (IRSTD) aims to accurately identify small, low-contrast targets in complex backgrounds [1]–[3]. It has broad practical value [4] and supports critical applications, including maritime surveillance, search and rescue, and early warning. In these applications, detection accuracy directly affects the speed and reliability of a system's response to potential threats and, consequently, mission outcomes and personnel safety. Improving the accuracy and robustness of IRSTD in complex scenes remains a central objective for practical deployment.

Traditional methods generally treat infrared small target detection as a filtering and target-enhancement problem, relying on handcrafted feature extraction and carefully tuned parameters. Their flexibility and robustness are often limited when target size, shape, and background interference vary substantially in real scenes. In recent years, deep learning techniques, including convolutional neural networks (CNNs), have substantially improved IRSTD accuracy. Dai et al. [5] propose a segmentation-based detector with asymmetric contextual modulation for feature fusion in a U-Net framework [6]. Li et al. [7] develop a dense nested attention network for multiscale feature fusion. Zhang et al. [8] introduce an infrared shape network that performs well on multiple datasets. Yang et al. [9] propose a progressive background-aware transformer that refines background context across multiple stages to mitigate feature degradation and error accumulation, improving detection and suppressing clutter. Wu et al. [10] replace conventional cross-layer feature fusion in U-Net with learnable local saliency kernels to guide the extraction of salient infrared target features.

Despite these advances, information bottlenecks in target feature extraction continue to limit detection performance. Many existing designs address individual deficiencies, while the underlying difficulties of infrared small target detection remain. Two limitations in network architecture and feature modeling are particularly relevant.

First, early convolutional layers in existing architectures often lack sensitivity to fine-grained information from very small targets [11]. This reduces feature contrast between targets and complex backgrounds, limiting the discrimination of weak targets and reducing detection accuracy and reliability.

Second, the static feature connections widely used in current networks lack the ability to adjust feature transfer dynamically in salient regions. Fixed fusion rules make it difficult to adaptively distinguish true infrared targets from regions that generate false alarms, restricting performance in complex backgrounds.

To address these limitations, we propose a selective attention network for infrared small target detection, termed SANet. We first develop a dual-path semantic-aware module (DSM) as the basic feature extraction unit to improve sensitivity to weak targets. DSM combines standard and pinwheel-shaped convolutions in two parallel paths. The standard convolution path preserves basic discriminative features, whereas the pinwheel-shaped convolution path [12] expands the receptive field to enhance responses to target regions. The fused features are then refined by the spatial and channel attention mechanisms of the convolutional block attention module (CBAM) [13], which emphasize useful information and suppress background noise.

Distinguishing true targets from false alarms requires an analysis of the surrounding background beyond local image regions. We therefore introduce a selective attention fusion module (SAFM) to replace static skip connections in conventional U-Net. SAFM combines selective attention with a broad receptive field and dynamically integrates shallow and deep features according to the context. This improves discrimination between true small targets and background interference in complex scenes. The main contributions are as follows.

1) We propose an infrared small target detection method based on selective attention. It improves the distinction between true targets and false detections in complex backgrounds, increasing detection accuracy and robustness.

2) We construct DSM by combining standard and pinwheel-shaped convolutions. Its broader receptive field helps suppress background noise, while spatial and channel attention improve the representation of fine-grained features.

3) We design SAFM to replace static skip connections in conventional U-Net. Dynamic feature fusion selectively guides information transfer and improves false alarm suppression.

4) Extensive experiments on three public datasets validate the proposed method. SANet compares favorably with existing state-of-the-art detectors across multiple performance metrics.

## II. RELATED WORK

## A. Traditional Methods for IRSTD

Traditional methods form the foundation of research on infrared small target detection. Early approaches rely primarily on handcrafted image processing and statistical models and offer some robustness under low-resolution and low-signal-to-noise conditions. Filter-based methods [14] suppress background interference through spatial- or frequency-domain filtering. They perform well on relatively uniform backgrounds, but their effectiveness is limited when target and background frequency components overlap. Lowrank methods [15], [16] decompose an image into a low-rank background and a sparse target component, enhancing targets at low signal-to-noise ratios. However, separating sparse noise from true targets remains difficult and can lead to high false alarm rates. Local contrast methods [17], [18] enhance target saliency by measuring intensity differences in local regions. Although effective for small targets, their reliance on fixed window sizes limits adaptation to targets at different scales.

Most traditional approaches depend on fixed handcrafted features and parameter settings. They cannot fully model the diversity and uncertainty of targets in complex backgrounds, which limits their accuracy and robustness in practical applications. These limitations motivate data-driven methods.

## B. Deep Learning Methods for IRSTD

Compared with traditional approaches, data-driven deep learning has substantially advanced infrared small target detection. MDvsFA [19] introduces adversarial learning to alleviate the trade-off between missed detections and false alarms. ACM [5] uses asymmetric contextual modulation within U-Net to combine high-level semantics with finegrained low-level details. ALCNet [20] embeds a parameterfree local contrast measurement module in the network to overcome the limited receptive field of small convolutional kernels. UIUNet [21] embeds small U-Net structures in a larger U-Net to learn multilevel and multiscale target representations. GCI-Net [22] uses Gaussian curvature information to enhance the geometric features of target regions. ABC [23] combines convolutional and linear structures with transformer modules to improve feature extraction and fusion while suppressing background noise. SCTransNet [24] uses a spatial-channel cross transformer to increase target-clutter separability across semantic levels. CFD-Net [25] draws on computational fluid dynamics, modeling infrared feature evolution as the flow of pixel particles to improve target representation. HDNet [26] combines multiscale target perception in the spatial domain with low-frequency suppression in the frequency domain to improve accuracy and robustness.

HAFNet [27] combines dual-branch semantic perception with hierarchical feature fusion in the encoder and decoder. MPCNet [28] integrates multiscale perception with crossattention feature fusion. FSGNet [29] uses frequency-aware skip connections and global semantic guidance to suppress background interference and preserve semantic consistency. APTNet [30] combines double residual attention and adaptive partial transformer blocks to strengthen target representation and context modeling.

General-purpose segmentation models have also been introduced into IRSTD. IRSAM [31] adapts the Segment Anything Model [32] to infrared small target detection and incorporates a wavelet-based Perona–Malik diffusion module to enhance edge feature extraction. State space models (SSMs) [33] offer effective long-sequence modeling and provide an alternative for capturing long-range dependencies in vision tasks. Mamba [34], a representative SSM architecture, has attracted considerable attention. MiM-ISTD [35] is the first work to apply Mamba to infrared small target detection. Building on this direction, IRMamba [36] models pixel differences in intensity and direction between scanning positions and their central neighborhoods. These differences are incorporated into the state update equations to strengthen the representation of target details.

## A. Overall Network Architecture

Fig. 1 illustrates the overall architecture of SANet. The network is built on the conventional U-Net framework and incorporates two key components, DSM and SAFM, to improve detection performance.

![](images/6e728dccb4ecffb9900637227bd4869ef5fca577c80cead84f2a72ad55b2d77d.jpg)  
Fig. 1. Overall Framework of the SANet.

![](images/244bfe0e866f618e4770c602b42757c61c39ba7173a6d89b7922f0f572059710.jpg)  
Fig. 2. Dual-path semantic perception module structure.

The encoder contains five encoding units, each comprising a DSM and a max-pooling operation to progressively reduce spatial resolution. DSM integrates contextual information over a broader receptive field and uses CBAM [13] to enhance target features and suppress background interference. This improves the network's sensitivity to dim, small targets.

The decoder contains four SAFMs with corresponding upsampling convolutional layers. SAFM dynamically fuses features from the encoder and decoder across scales, adaptively strengthening responses in salient regions and improving discrimination between true targets and false alarms. The upsampling convolutional layers retain the standard convolution design of U-Net and progressively restore spatial resolution.

Finally, the decoder features are passed to a detection head comprising a 3 × 3 convolution and a sigmoid activation function to generate the target prediction.

## B. Dual-path Semantic-aware Module

Conventional convolutional layers have fixed, limited receptive fields and cannot adequately model long-range feature dependencies. This restricts their sensitivity to dim targets, including small infrared targets. We propose DSM to improve basic feature extraction. As shown in Fig. 2, DSM consists of two parallel branches that jointly enhance feature representation. The first branch uses standard $3 \quad \times \quad 3$ convolutions to maintain local spatial consistency and preserve fine-grained details. The second branch uses pinwheel-shaped convolutions with successive kernel sizes of $3 \times 3 , 5 \times 5$ , and 3 × 3 to expand the receptive field and capture broader context.

![](images/fb7f81b468e2337390a432efae1ee0eb10ff3daddf541f88638ad141003e9550.jpg)  
Fig. 3. Pinwheel-shaped convolution structure.

![](images/4e3c4369ee90070b637b50f897d2725d8addeb2e55ea7b309506a6436df06300.jpg)  
Fig. 4. Convolutional block attention module structure.

Fig. 3 shows the pinwheel-shaped convolution. Unlike conventional convolution, asymmetric padding creates direction-sensitive horizontal and vertical kernels, allowing the network to capture target structures in different directions and adapt to shape variations. Compared with other receptivefield expansion techniques, such as standard dilated convolution, pinwheel-shaped convolution requires fewer parameters and less computation. This multiscale mechanism enables DSM to model contextual semantics at different spatial scales. The features from the two branches are subsequently fused using a $3 \ \times \ 3$ convolution to obtain a unified representation.

![](images/4ff6078689e1d4a8b1610d2e42fbca8cf2543e82f706a6d925883f1b7d32a3b0.jpg)  
Fig. 5. Selective attention fusion module structure.

CBAM further refines the fused features. As shown in Fig. 4, CBAM cascades channel and spatial attention. Spatial attention enhances spatial feature responses, while channel attention models interchannel relationships from a global perspective and emphasizes semantic channels relevant to small target detection. Together, they improve feature discrimination.

## C. Selective Attention Fusion Module

We propose SAFM to improve discrimination between true and false targets. By introducing dynamic spatial selection into the U-Net structure, SAFM uses multiscale features to emphasize spatial regions containing informative, targetrelated responses, as shown in Fig. 5.

At stage i, SAFM fuses the encoder and decoder feature maps, which satisfy $E _ { i } , D _ { i } \in \mathbb { R } ^ { C _ { i } \times \frac { H } { 2 ^ { i } } \times \frac { W } { 2 ^ { i } } }$ . Here, $C _ { i }$ is the number of channels, and H and W are the height and width of the original infrared image. The two feature maps are first concatenated along the channel dimension:

$$
X _ { i } = C o n c a t ( E _ { i } , D _ { i } ) ,\tag{1}
$$

To reduce computational complexity, channel-wise average pooling $\mathcal { P } _ { a v g }$ and max pooling $\mathcal { P } _ { m a x }$ are applied separately to the fused features $X _ { i } { \mathrm { : } }$

$$
S F _ { i } ^ { a v g } = \mathcal { P } _ { a v g } ( X _ { i } ) , S F _ { i } ^ { m a x } = \mathcal { P } _ { m a x } ( X _ { i } ) ,\tag{2}
$$

Here, $S F _ { i } ^ { a v g }$ and $S F _ { i } ^ { m a x }$ are the resulting spatial feature representations. They are concatenated to form a joint spatial representation, enabling complementary spatial attention information to interact:

$$
S A F _ { i } = f _ { 7 \times 7 } \left( C o n c a t \bigl ( S F _ { i } ^ { a v g } , S F _ { i } ^ { m a x } \bigr ) \right) ,\tag{3}
$$

The resulting spatial attention feature map $S A F _ { i }$ then weights the encoder and decoder features through elementwise multiplication to obtain the fused representation $Y _ { i } \mathbf { { : } }$

$$
Y _ { i } = E _ { i } \odot \sigma ( S A F _ { i } ) + D _ { i } \odot \sigma ( 1 - S A F _ { i } ) ,\tag{4}
$$

Here, � denotes the sigmoid activation function, and ⨀ denotes element-wise multiplication.

Finally, $Y _ { i }$ is passed through a $1 \times 1$ convolution to expand its channel representation. The result is multiplied element-

wise with the original fused features $X _ { i } ,$ , and a learnable scalar weight � is introduced to obtain the output:

$$
F _ { S A F M } = \lambda \odot \left( X _ { i } \odot f _ { 1 \times 1 } ( Y _ { i } ) \right) + X _ { i } ,\tag{5}
$$

The parameter λ is a single global learnable scalar shared throughout SAFM and is updated during network training. It controls the fusion strength between the attention-enhanced features and the residual information, preserving the original feature structure while strengthening discriminative responses in key regions.

## IV. EXPERIMENTS AND ANALYSIS

## A. Datasets and Experimental Settings

We evaluate SANet on three public benchmarks: NUAA-SIRST [5], IRSTD-1K [8], and NUDT-SIRST [7]. They contain 427, 1001, and 1327 annotated infrared images, respectively, covering diverse representative infrared scenes. The training-to-test ratios are 1:1, 4:1, and 1:1 for NUAA-SIRST, IRSTD-1K, and NUDT-SIRST, respectively. Training images are resized to 320 × 320 for NUAA-SIRST and IRSTD-1K and to $2 5 6 \times 2 5 6$ for NUDT-SIRST. The corresponding batch sizes are 4, 12, and 16.

The network is trained for 600 epochs using the Soft-IoU loss [37] and the AdamW optimizer. The initial learning rate is 0.001, and the weight decay is $1 \times 1 0 ^ { - 2 } .$ . A cosine annealing schedule gradually reduces the learning rate to $1 \times 1 0 ^ { - 5 }$ to support stable convergence. The segmentation threshold for the saliency maps is set to 0.5 in all experiments.

## B. Evaluation Metrics

We use four common metrics to evaluate detection performance: intersection over union (IoU), normalized intersection over union (nIoU), detection probability $\left( P _ { d } \right)$ , and false alarm rate (� ). They are defined as follows:

$$
I o U = \frac { T P } { T + P - T P } ,\tag{6}
$$

$$
n I o U = \frac { 1 } { N } \sum _ { i } ^ { N } \frac { T P ( i ) } { T ( i ) + P ( i ) - T P ( i ) } ,\tag{7}
$$

$$
P _ { d } = \frac { T P } { T P + F N } ,\tag{8}
$$

$$
F _ { a } = \frac { F P } { F P + T N } .\tag{9}
$$

Here, N is the total number of samples. T and P denote the numbers of ground-truth positive pixels and predicted positive pixels, respectively. TP, FP, and FN denote the numbers of true positive, false positive, and false negative pixels, respectively; TN denotes the number of true negative pixels.

## C. Quantitative Evaluation

We compare SANet with 14 existing IRSTD methods, comprising five traditional approaches and nine deep learning methods. The traditional methods are facet kernel and random walker (FKRW) [38], tri-layer local contrast method (TLLCM) [17], new top-hat transformation (NWMTH) [39], nonconvex optimization with an l<sub>p</sub>-norm constraint (NOLC) [40], and partial sum of the tensor nuclear norm (PSTNN) [41]. The deep learning methods are asymmetric contextual modulation (ACM) [5], receptive-field and direction induced attention network (RDIAN) [42], attention-guided pyramid context network (AGPCNet) [43], dense nested attention network (DNANet) [7], U-Net in U-Net (UIUNet) [21], robust principal component analysis network (RPCANet) [44], multiscale head network (MSHNet) [45], spatial-channel cross transformer network (SCTransNet) [24], and learnable local saliency kernel network (L<sup>2</sup>SKNet) [10]. The compared ACM and L<sup>2</sup>SKNet models use U-Net as the backbone. For a fair comparison, all deep learning models are retrained on the same training benchmark, following the settings described in their original papers. Table I summarizes the results, from which we make the following observations.

TABLE I  
COMPARISON WITH STATE-OF-THE-ARTS METHODS ON THREE DATASETS. IN IOU(%)↑, nIoU(%)↑, �<sub>�</sub>(%)↑, �<sub>�</sub>(10<sup>-6</sup>)↓
<table><tr><td>Method</td><td colspan="4">NUAA-SIRST</td><td colspan="4">IRSTD-1K</td><td colspan="4">NUDT-SIRST</td></tr><tr><td></td><td>IoU (%)</td><td>nIoU (%)</td><td> $P _ { d } ( \% )$ </td><td> $F _ { a } \left( 1 0 ^ { - 6 } \right)$ </td><td>IoU (%)</td><td>nIoU (%)</td><td> $P _ { d } ( \% )$ </td><td> $F _ { a } \left( 1 0 ^ { - 6 } \right)$ </td><td>IoU (%)</td><td>nIoU (%)</td><td>Pd(%)</td><td> $F _ { a } \left( 1 0 ^ { - 6 } \right)$ </td></tr><tr><td>FKRW</td><td>22.06</td><td>28.08</td><td>81.77</td><td>16.32</td><td>10.39</td><td>16.25</td><td>69.54</td><td>24.37</td><td>12.67</td><td>21.73</td><td>79.51</td><td>67.13</td></tr><tr><td>TLLCM</td><td>18.66</td><td>27.71</td><td>82.24</td><td>16.95</td><td>10.26</td><td>18.24</td><td>68.55</td><td>24.48</td><td>11.08</td><td>23.82</td><td>77.97</td><td>67.26</td></tr><tr><td>NWMTH</td><td>15.77</td><td>17.28</td><td>71.12</td><td>55.61</td><td>18.94</td><td>16.91</td><td>49.26</td><td>21.72</td><td>11.72</td><td>10.26</td><td>52.71</td><td>46.81</td></tr><tr><td>NOLC</td><td>26.64</td><td>35.68</td><td>81.62</td><td>17.44</td><td>12.39</td><td>22.18</td><td>75.38</td><td>21.94</td><td>23.87</td><td>34.90</td><td>85.47</td><td>58.20</td></tr><tr><td>PSTNN</td><td>22.40</td><td>29.59</td><td>77.95</td><td>29.11</td><td>24.57</td><td>28.71</td><td>71.99</td><td>35.26</td><td>27.72</td><td>39.80</td><td>66.13</td><td>44.17</td></tr><tr><td>ACM</td><td>65.28</td><td>65.67</td><td>90.49</td><td>47.13</td><td>58.84</td><td>58.23</td><td>91.92</td><td>27.59</td><td>60.63</td><td>60.94</td><td>91.75</td><td>20.06</td></tr><tr><td>RDIAN</td><td>70.66</td><td>74.04</td><td>93.92</td><td>43.08</td><td>63.37</td><td>62.63</td><td>88.55</td><td>30.18</td><td>77.77</td><td>79.46</td><td>95.66</td><td>26.36</td></tr><tr><td>AGPCNet</td><td>75.69</td><td>76.60</td><td>96.48</td><td>14.99</td><td>66.29</td><td>65.23</td><td>92.83</td><td>13.12</td><td>88.87</td><td>90.64</td><td>97.20</td><td>10.02</td></tr><tr><td>DNANet</td><td>76.34</td><td>79.19</td><td>95.82</td><td>15.23</td><td>66.50</td><td>66.13</td><td>91.58</td><td>18.31</td><td>91.96</td><td>92.88</td><td>98.94</td><td>9.28</td></tr><tr><td>UIUNet</td><td>77.17</td><td>78.86</td><td>95.44</td><td>14.34</td><td>65.34</td><td>65.55</td><td>90.91</td><td>15.01</td><td>90.43</td><td>89.87</td><td>98.94</td><td>7.72</td></tr><tr><td>RPCANet</td><td>53.93</td><td>61.01</td><td>95.44</td><td>149.2</td><td>62.09</td><td>61.48</td><td>89.35</td><td>48.61</td><td>87.79</td><td>89.80</td><td>95.98</td><td>32.03</td></tr><tr><td>MSHNet</td><td>73.88</td><td>72.56</td><td>95.82</td><td>19.25</td><td>67.17</td><td>60.38</td><td>92.52</td><td>12.60</td><td>76.95</td><td>79.72</td><td>95.77</td><td>20.52</td></tr><tr><td>SCTransNet</td><td>75.46</td><td>78.06</td><td>95.82</td><td>18.80</td><td>66.54</td><td>66.36</td><td>93.27</td><td>13.68</td><td>93.01</td><td>93.50</td><td>98.73</td><td>4.80</td></tr><tr><td>L2SKNet</td><td>72.90</td><td>73.89</td><td>93.92</td><td>40.34</td><td>67.38</td><td>66.36</td><td>91.58</td><td>14.78</td><td>93.52</td><td>92.86</td><td>97.57</td><td>5.29</td></tr><tr><td>SANet</td><td>79.10</td><td>80.01</td><td>96.58</td><td>11.52</td><td>71.70</td><td>66.59</td><td>93.27</td><td>11.88</td><td>95.73</td><td>95.43</td><td>99.26</td><td>2.80</td></tr></table>

![](images/675375b5765fb44612157aba5c06cc5009a912e68090a79817a7d2b11919dcad.jpg)  
Fig. 6. Visualization of different detection methods on the NUAA-SIRST, IRSTD-1K, and NUDT-SIRST datasets. The red and yellow boxes represent correctly detected targets and false alarms, respectively.

![](images/e24841a7deb02909201ce391a5932047ca463316ede4ed45dde177d48f9b2680.jpg)  
Fig. 7. 3D visualization of different IRSTD methods on the IRSTD-1K dataset. The red, blue, and yellow boxes represent correctly detected targets, missed detections, and false alarms, respectively.

TABLE II  
QUANTITATIVE ABLATION EXPERIMENTS EVALUATE THE IMPACT OF THE CORE COMPONENTS DSM AND SAFM IN SANET
<table><tr><td>Dataset</td><td>DSM</td><td>SAFM</td><td>IoU(%)</td><td>nIoU(%)</td><td>Pd(%)</td><td>Fa(10-6)</td></tr><tr><td rowspan="3">NUAA- SIRST</td><td>x</td><td>x</td><td>74.91</td><td>79.01</td><td>95.44</td><td>28.33</td></tr><tr><td>√</td><td>x</td><td>78.23</td><td>79.51</td><td>95.06</td><td>13.86</td></tr><tr><td>√</td><td>√</td><td>79.10</td><td>80.01</td><td>96.58</td><td>11.52</td></tr><tr><td rowspan="3">IRSTD- 1K</td><td>x</td><td>x</td><td>64.75</td><td>63.24</td><td>91.58</td><td>18.77</td></tr><tr><td>V</td><td>x</td><td>71.46</td><td>67.08</td><td>90.57</td><td>16.25</td></tr><tr><td>√</td><td>√</td><td>71.70</td><td>66.59</td><td>93.27</td><td>11.88</td></tr><tr><td rowspan="3">NUDT- SIRST</td><td>x</td><td>x</td><td>85.42</td><td>85.58</td><td>98.10</td><td>6.99</td></tr><tr><td>√</td><td>x</td><td>95.05</td><td>94.94</td><td>98.84</td><td>2.85</td></tr><tr><td>√</td><td>√</td><td>95.73</td><td>95.43</td><td>99.26</td><td>2.80</td></tr></table>

First, SANet achieves strong performance across all three datasets. It obtains the best IoU, nIoU, and false alarm rate and the best or joint-best detection probability among the compared methods. This supports the ability of DSM and SAFM to distinguish true targets from background interference. On IRSTD-1K, its detection probability of 93.27% is equal to that of SCTransNet.

Second, deep learning methods substantially outperform traditional approaches. Handcrafted features and prior assumptions limit the ability of traditional methods to model image semantics. In contrast, deep learning methods extract semantic information from data and reduce reliance on manually tuned parameters, improving robustness to scene variations.

Third, SANet outperforms the recent L2SKNet detector on all three datasets. L2SKNet captures infrared target saliency through learnable local saliency kernels. SANet combines the proposed basic feature extraction module with selective attention fusion, producing more accurate small target detections and better overall performance.

## D. Qualitative Evaluation

Fig. 6 compares qualitative results on NUAA-SIRST in Figs. 6(a) and 6(b), IRSTD-1K in Figs. 6(c) and 6(d), and NUDT-SIRST in Figs. 6(e) and 6(f). Compared with traditional approaches, deep learning methods reduce false alarms in most cases through data-driven feature extraction, as illustrated in Figs. 6(b)–6(d). Although the other mainstream methods are competitive, they are generally less reliable than SANet in the illustrated scenes. In the complex scene in Fig. 6(c), all compared deep learning methods except UIUNet and SANet produce false alarms. However, UIUNet generates many false detections in the similarly complex scene in Fig. 6(d). SANet maintains accurate and robust detection across these scenes.

The three-dimensional visualizations in Fig. 7 further illustrate the performance of the deep learning methods. When multiple targets occur in complex backgrounds, competing methods have difficulty maintaining stable detection. UIUNet, in particular, generates many false alarms in the second scene. These results further support the robustness of SANet in complex environments.

## E. Ablation Study

## 1) Effectiveness of the Core Components of SANet

We use conventional U-Net as the baseline and progressively introduce the two proposed modules. This stepwise integration allows us to assess the contribution of each component and the effect of combining them.

Table II shows that incorporating DSM and SAFM improves overall detection performance. Adding DSM alone increases IoU by 3.32, 6.71, and 9.63 percentage points on

TABLE III  
QUANTITATIVE ABLATION EXPERIMENTS EVALUATE THE IMPACT OF CORE COMPONENTS IN DSM
<table><tr><td>Dataset</td><td colspan="6">Conv PConv CBAM IoU(%) nIoU(%)  $P _ { d } ( ^ { 0 } / _ { 0 } ) \ F _ { a } ( 1 0 ^ { - 6 } )$ </td></tr><tr><td rowspan="3">NUAA-SIRST</td><td>√ x</td><td>x</td><td>74.90</td><td></td><td>77.53 93.16</td><td>35.74</td></tr><tr><td>√</td><td>√</td><td>x</td><td>75.26</td><td>78.84 93.92</td><td>25.57</td></tr><tr><td>√</td><td>√</td><td>√</td><td>79.10 80.01</td><td>96.58</td><td>11.52</td></tr><tr><td rowspan="3">IRSTD-1K</td><td>√</td><td>x</td><td>x</td><td>66.99</td><td>64.03</td><td>91.25 34.56</td></tr><tr><td>√</td><td>√</td><td>x</td><td>67.84</td><td>64.41 93.27</td><td>28.45</td></tr><tr><td>√</td><td>√</td><td>√</td><td>71.70</td><td>66.59 93.27</td><td>11.88</td></tr><tr><td rowspan="3">NUDT-SIRST</td><td>√</td><td>x</td><td>x</td><td>91.28</td><td>91.09</td><td>98.41 4.71</td></tr><tr><td>√</td><td>√</td><td>x</td><td>92.95</td><td>92.56 99.05</td><td>6.96</td></tr><tr><td>√</td><td>√</td><td>√</td><td>95.73</td><td>95.43</td><td>99.26 2.80</td></tr></table>

TABLE IV

QUANTITATIVE ABLATION EXPERIMENTS EVALUATE THE IMPACT OFCORE COMPONENTS IN SAFM
<table><tr><td>Dataset</td><td>Residual</td><td>λ</td><td>IoU(%)</td><td>nIoU(%)</td><td>Pd(%)</td><td>Fa(10-6)</td></tr><tr><td rowspan="3">NUAA- SIRST</td><td>x</td><td>√</td><td>78.80</td><td>80.70</td><td>96.58</td><td>13.93</td></tr><tr><td>√</td><td>x</td><td>78.13</td><td>79.00</td><td>95.82</td><td>13.10</td></tr><tr><td>√</td><td>√</td><td>79.10</td><td>80.01</td><td>96.58</td><td>11.52</td></tr><tr><td rowspan="3">IRSTD- 1K</td><td>x</td><td>√</td><td>70.99</td><td>65.58</td><td>89.56</td><td>13.49</td></tr><tr><td>√</td><td>x</td><td>70.04</td><td>67.13</td><td>91.92</td><td>26.85</td></tr><tr><td>V</td><td>√</td><td>71.70</td><td>66.59</td><td>93.27</td><td>11.88</td></tr><tr><td rowspan="3">NUDT- SIRST</td><td>x</td><td>√</td><td>95.23</td><td>94.94</td><td>99.05</td><td>2.71</td></tr><tr><td>√</td><td>x</td><td>95.11</td><td>94.72</td><td>98.94</td><td>1.54</td></tr><tr><td>√</td><td>√</td><td>95.73</td><td>95.43</td><td>99.26</td><td>2.80</td></tr></table>

NUAA-SIRST, IRSTD-1K, and NUDT-SIRST, respectively. These gains indicate that DSM improves the extraction and modeling of small infrared target details during basic feature extraction.

Adding SAFM further improves IoU, detection probability, and false alarm rate on all three datasets. It also improves nIoU on NUAA-SIRST and NUDT-SIRST, although nIoU on IRSTD-1K decreases from 67.08% to 66.59%. Through dynamic spatial selection, SAFM adaptively fuses multiscale features and improves discrimination between true targets and false alarms, yielding better overall detection performance.

2) Effectiveness of the Dual-path Semantic-aware Module

DSM improves conventional feature extraction by integrating pinwheel-shaped convolution with spatial and channel attention to generate more robust, discriminative representations. We ablate its components to assess their contributions, as shown in Table III. Pinwheel-shaped convolution improves detection by expanding the receptive field and increasing sensitivity to dim targets, particularly in terms of missed detections. However, a broader receptive field can also introduce additional background interference. For example, on NUDT-SIRST, adding pinwheel-shaped convolution increases $F _ { a }$ from $4 . 7 1 ~ \times ~ 1 0 ^ { - 6 } ~ \mathrm { t o } ~ 6 . 9 6 ~ \times ~ 1 0 ^ { - 6 }$ before attention is introduced. This suggests that broad contextual information requires careful refinement to avoid

![](images/c7a30421c6ebd898b87e4eb7b7bf2a7edd69c3144731dbd2442d465c2d11be8b.jpg)  
Fig.8 Performance evaluation of five algorithms across three metrics, where the IoU metric represents the average results on the three datasets.

redundant or noisy responses.

Integrating spatial and channel attention yields larger improvements. By adaptively adjusting responses across spatial locations and channels, the attention mechanisms focus the network on potential target regions, strengthen discriminative target features, and suppress background noise. 3) Effectiveness of the Selective Attention Fusion Module

SAFM adaptively fuses features across scales through spatially varying weights and selectively enhances salient regions. We ablate its components to assess their individual contributions, as shown in Table IV. Removing either the residual connection or the learnable scalar λ reduces IoU. The complete model achieves IoU values of 79.10%, 71.70%, and 95.73% on the three datasets, respectively, exceeding the results obtained after removing either component. The two components therefore make complementary contributions to segmentation accuracy.

The learnable scalar improves IoU on all three datasets. On NUAA-SIRST, introducing it reduces $F _ { a }$ from $1 3 . 1 0 \times 1 0 ^ { - 6 }$ to $1 1 . 5 2 \times 1 0 ^ { - 6 }$ , supporting the effectiveness of selective attention fusion. The residual connection helps preserve target information and improves detection probability on multiple datasets. The improvement is particularly clear on IRSTD-1K, where $P _ { d }$ increases from 89.56% to 93.27%. However, the full model does not minimize every metric in Table IV: some ablated variants achieve a lower false alarm rate on NUDT-SIRST or a higher nIoU on individual datasets.

## F. Model Complexity Analysis

Fig. 8 compares SANet with representative recent detectors, including DNANet, UIUNet, SCTransNet, and L2SKNet, in terms of complexity and detection performance. SANet achieves the highest mean IoU across the three benchmarks, 82.18%, exceeding the other methods by 3.84–4.56 percentage points. SANet contains 12.72 million parameters and requires 19.75 GFLOPs. Although it is not the smallest of the five models, its computational cost is moderate relative to UIUNet and SCTransNet.

These results indicate a favorable trade-off between detection accuracy and model complexity. While the architecture is somewhat more complex than the lightest alternatives, its detection performance supports its potential for practical infrared imaging applications.

## V. CONCLUSION

This paper proposes SANet, a selective attention network for infrared small target detection, to address dim targets, complex backgrounds, and false alarms. DSM and SAFM improve small target perception, discriminative feature representation, and background suppression.

DSM exploits the complementary properties of standard and pinwheel-shaped convolutions to expand the receptive field while preserving basic discriminative features, improving responses to small targets in complex backgrounds. SAFM replaces static skip connections in conventional U-Net with dynamic feature fusion, strengthening discrimination between true targets and false alarms.

Experiments on multiple public datasets show that SANet compares favorably with mainstream methods in detection accuracy and robustness across datasets, supporting its practical value. Future work will focus on reducing model complexity and improving detection in challenging scenes to extend its applicability to different infrared imaging tasks.

## REFERENCES

[1] R. Kou, C. Wang, Y. Luo, Y. Zhang, Z. Xu, Z. Peng, C. Wu, and Q. Fu, “Multiscale small-target detection techniques in single-frame infrared images: A review,” J. Image Graph., vol. 29, no. 9, pp. 2625–2649, 2024, doi: 10.11834/jig.230788.

[2] X. Pan, N. Jia, Y. Mu, and X. Gao, “Survey of small object detection,” J. Image Graph., vol. 28, no. 9, pp. 2587–2615, 2023, doi: 10.11834/jig.220455.

[3] Z. Liu, Y. Wu, and Y. Zou, “Multiscale infrared superpixel-image model for small-target detection,” J. Image Graph., vol. 24, no. 12, pp. 2159–2173, 2019, doi: 10.11834/jig.190068.

[4] D. Yang, T. Han, H. Hu, et al., “Research status and prospects of single-frame infrared small target detection technology,” Signal Process., vol. 40, no. 5, pp. 887–906, 2024, doi: 10.16798/j.issn.1003- 0530.2024.05.008.

[5] Y. Dai, Y. Wu, F. Zhou, and K. Barnard, “Asymmetric Contextual Modulation for Infrared Small Target Detection,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pp. 950–959, 2021, doi: 10.1109/WACV48630.2021.00099.

[6] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), pp. 234– 241, 2015, doi: 10.1007/978-3-319-24574-4\_28.

[7] B. Li, C. Xiao, L. Wang, Y. Wang, Z. Lin, M. Li, W. An, and Y. Guo, “Dense Nested Attention Network for Infrared Small Target Detection,” IEEE Trans. Image Process., vol. 32, pp. 1745–1758, 2023, doi: 10.1109/TIP.2022.3199107.

[8] M. Zhang, R. Zhang, Y. Yang, H. Bai, J. Zhang, and J. Guo, “ISNet: Shape Matters for Infrared Small Target Detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 877– 886, 2022, doi: 10.1109/CVPR52688.2022.00095.

[9] H. Yang, T. Mu, Z. Dong, Z. Zhang, B. Wang, and W. Ke, “PBT: Progressive background-aware transformer for infrared small target detection,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–13, 2024.

[10] F. Wu, A. Liu, T. Zhang, L. Zhang, J. Luo, and Z. Peng, “Saliency at the Helm: Steering Infrared Small Target Detection With Learnable Kernels,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–14, 2025, doi: 10.1109/TGRS.2024.3521947.

[11] Z. Yu, J. Wan, Y. Qin, X. Li, S. Z. Li, and G. Zhao, “NAS-FAS: Static-Dynamic Central Difference Network Search for Face Anti-Spoofing,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 9, pp. 3005–3023, 2021, doi: 10.1109/TPAMI.2020.3036338.

[12] J. Yang, S. Liu, J. Wu, X. Su, N. Hai, and X. Huang, “Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 9, pp. 9202– 9210, 2025, doi: 10.1609/aaai.v39i9.32996.

[13] S. Woo, J. Park, J. Y. Lee, and I. S. Kweon, “CBAM: Convolutional Block Attention Module,” in Proc. Eur. Conf. Comput. Vis. (ECCV), pp. 3–19, 2018, doi: 10.1007/978-3-030-01234-2\_1.

[14] L. Deng, J. Zhang, G. Xu, and H. Zhu, “Infrared small target detection via adaptive M-estimator ring top-hat transformation,” Pattern Recognit., vol. 112, Art. no. 107729, 2021, doi: 10.1016/j.patcog.2020.107729.

[15] Y. Dai and Y. Wu, “Reweighted Infrared Patch-Tensor Model With Both Nonlocal and Local Priors for Single-Frame Small Target Detection,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 10, no. 8, pp. 3752–3767, 2017, doi: 10.1109/JSTARS.2017.2700023.

[16] Y. Sun, J. Yang, and W. An, “Infrared Dim and Small Target Detection via Multiple Subspace Learning and Spatial-Temporal Patch-Tensor Model,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 5, pp. 3737– 3752, 2021, doi: 10.1109/TGRS.2020.3022069.

[17] J. Han, S. Moradi, I. Faramarzi, C. Liu, H. Zhang, and Q. Zhao, “A Local Contrast Method for Infrared Small-Target Detection Utilizing a Tri-Layer Window,” IEEE Geosci. Remote Sens. Lett., vol. 17, no. 10, pp. 1822–1826, 2020, doi: 10.1109/LGRS.2019.2954578.

[18] J. Han, S. Moradi, I. Faramarzi, H. Zhang, Q. Zhao, X. Zhang, and N. Li, “Infrared Small Target Detection Based on the Weighted Strengthened Local Contrast Measure,” IEEE Geosci. Remote Sens. Lett., vol. 18, no. 9, pp. 1670–1674, 2021, doi: 10.1109/LGRS.2020.3004978.

[19] H. Wang, L. Zhou, and L. Wang, “Miss Detection vs. False Alarm: Adversarial Learning for Small Object Segmentation in Infrared Images,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 8509–8518, 2019, doi: 10.1109/ICCV.2019.00860.

[20] Y. Dai, Y. Wu, F. Zhou, and K. Barnard, “Attentional Local Contrast Networks for Infrared Small Target Detection,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 11, pp. 9813–9824, 2021, doi: 10.1109/TGRS.2020.3044958.

[21] X. Wu, D. Hong, and J. Chanussot, “UIU-Net: U-Net in U-Net for Infrared Small Object Detection,” IEEE Trans. Image Process., vol. 32, pp. 364–376, 2023, doi: 10.1109/TIP.2022.3228497.

[22] M. Zhang, K. Yue, B. Li, J. Guo, Y. Li, and X. Gao, “Single-Frame Infrared Small Target Detection via Gaussian Curvature Inspired Network,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–13, 2024, doi: 10.1109/TGRS.2024.3423492.

[23] P. Pan, H. Wang, C. Wang, and C. Nie, “ABC: Attention with Bilinear Correlation for Infrared Small Target Detection,” in Proc. IEEE Int. Conf. Multimedia Expo (ICME), pp. 2381–2386, 2023, doi: 10.1109/ICME55011.2023.00406.

[24] S. Yuan, H. Qin, X. Yan, N. Akhtar, and A. Mian, “SCTransNet: Spatial-Channel Cross Transformer Network for Infrared Small Target Detection,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1–15, 2024, doi: 10.1109/TGRS.2024.3383649.

[25] M. Zhang, K. Yue, J. Guo, Q. Zhang, J. Zhang, and X. Gao, “Computational fluid dynamic network for infrared small target detection,” IEEE Trans. Neural Netw. Learn. Syst., early access, 2025, doi: 10.1109/TNNLS.2025.3548984.

[26] M. Xu, C. Yu, Z. Li, H. Tang, Y. Hu, and L. Nie, “HDNet: A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–15, 2025, doi: 10.1109/TGRS.2025.3574962.

[27] Y. Zhang, W. Bao, Y. Yang, W. Wan, Q. Xiao, and X. Zou, “HAFNet: Hierarchical Attention Fusion Network for Infrared Small Target Detection,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025, doi: 10.1109/tgrs.2025.3607732.

[28] Y. Zhang, W. Bao, Y. Yang, W. Wan, Q. Xiao, and X. Zou, “MPCNet: Multiscale Perception and Cross-Attention Feature Fusion Network for Infrared Small Target Detection,” IEEE Trans. Geosci. Remote Sens., vol. 64, pp. 1–15, 2026, doi: 10.1109/tgrs.2026.3653023.

[29] Y. Zhang et al., “FSGNet: A frequency-aware and semantic guidance network for infrared small target detection,” Expert Syst. Appl., vol. 328, Art. no. 132925, 2026, doi: 10.1016/j.eswa.2026.132925.

[30] Y. Zhang et al., “APTNet: Adaptive Partial Transformer Network for Infrared Small Target Detection,” IEEE Sensors J., vol. 25, no. 10, pp. 17960–17974, 2025, doi: 10.1109/jsen.2025.3559093.

[31] M. Zhang, Y. Wang, J. Guo, Y. Li, X. Gao, and J. Zhang, “IRSAM: Advancing Segment Anything Model for Infrared Small Target Detection,” in Proc. Eur. Conf. Comput. Vis. (ECCV), pp. 233–249, 2024, doi: 10.1007/978-3-031-72855-6\_14.

[32] A. Kirillov, et al., “Segment Anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), pp. 4015–4026, 2023, doi: 10.1109/ICCV51070.2023.00371.

[33] A. Gu, K. Goel, and C. Ré, “Efficiently modeling long sequences with structured state spaces,” arXiv:2111.00396, 2021. [Online]. Available: https://arxiv.org/abs/2111.00396.

[34] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv:2312.00752, 2023. [Online]. Available: https://arxiv.org/abs/2312.00752.

[35] T. Chen, Z. Ye, Z. Tan, T. Gong, Y. Wu, Q. Chu, B. Liu, N. Yu, and J. Ye, “MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small-Target Detection,” IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1– 13, 2024, doi: 10.1109/TGRS.2024.3485721.

[36] M. Zhang, X. Li, F. Gao, and J. Guo, “IRMamba: Pixel Difference Mamba with Layer Restoration for Infrared Small Target Detection,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 9, pp. 10003–10011, 2025, doi: 10.1609/aaai.v39i9.33085.

[37] M. A. Rahman and Y. Wang, “Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation,” in Proc. 12th Int. Symp. Vis. Comput., pp. 234–244, 2016, doi: 10.1007/978-3-319- 50835-1\_22.

[38] Y. Qin, L. Bruzzone, C. Gao, and B. Li, “Infrared Small Target Detection Based on Facet Kernel and Random Walker,” IEEE Trans. Geosci. Remote Sens., vol. 57, no. 9, pp. 7104–7118, 2019, doi: 10.1109/TGRS.2019.2911513.

[39] X. Bai and F. Zhou, “Analysis of new top-hat transformation and the application for infrared dim small target detection,” Pattern Recognit., vol. 43, no. 6, pp. 2145–2156, 2010, doi: 10.1016/j.patcog.2009.12.023.

[40] T. Zhang, H. Wu, Y. Liu, L. Peng, C. Yang, and Z. Peng, “Infrared Small Target Detection Based on Non-Convex Optimization with Lp-Norm Constraint,” Remote Sens., vol. 11, no. 5, Art. no. 559, 2019, doi: 10.3390/rs11050559.

[41] L. Zhang and Z. Peng, “Infrared Small Target Detection Based on Partial Sum of the Tensor Nuclear Norm,” Remote Sens., vol. 11, no. 4, Art. no. 382, 2019, doi: 10.3390/rs11040382.

[42] H. Sun, J. Bai, F. Yang, and X. Bai, “Receptive-Field and Direction Induced Attention Network for Infrared Dim Small Target Detection With a Large-Scale Dataset IRDST,” IEEE Trans. Geosci. Remote Sens., vol. 61, pp. 1–13, 2023, doi: 10.1109/TGRS.2023.3235150.

[43] T. Zhang, L. Li, S. Cao, T. Pu, and Z. Peng, “Attention-Guided Pyramid Context Networks for Detecting Infrared Small Target Under Complex Background,” IEEE Trans. Aerosp. Electron. Syst., vol. 59, no. 4, pp. 4250–4261, 2023, doi: 10.1109/TAES.2023.3238703.

[44] F. Wu, T. Zhang, L. Li, Y. Huang, and Z. Peng, “RPCANet: Deep Unfolding RPCA Based Infrared Small Target Detection,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), pp. 4809–4818, 2024, doi: 10.1109/WACV57701.2024.00474.

[45] Q. Liu, R. Liu, B. Zheng, H. Wang, and Y. FU, “Infrared Small Target Detection with Scale and Location Sensitivity,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 17490–17499, 2024, doi: 10.1109/CVPR52733.2024.01656.