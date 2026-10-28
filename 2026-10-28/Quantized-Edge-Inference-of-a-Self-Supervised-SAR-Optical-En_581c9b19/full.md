# Quantized Edge Inference of a Self-Supervised SAR–Optical Encoder on Commodity x86 Hardware

Christian Sanchez-Leon and Dongming Chen

Abstract—Self-supervised vision transformers are increasingly used as reusable encoders for Earth observation, but their inference cost on the hardware available in the field is rarely characterized. We deploy a pretrained SAR–optical maskedautoencoder encoder on three commodity x86 nodes, from a lowpower CPU without AVX2 to an AVX2+VNNI reference, at FP32, FP16 and INT8 on the CPU and, where present, the integrated GPU, and report latency, energy (RAPL and wall plug) and linear-probe accuracy. Quantization is accuracy-neutral: INT8 changes micro-mAP by only +0.0006, within the seed spread, so precision becomes a pure cost decision. INT8 lowers CPU latency on every node, by 1.45× even on the CPU without AVX2, and CPU energy wherever it is measured, and the VNNI reference (1.53×) suggests a further gain beyond AVX2. The Pareto-optimal device shifts from the integrated GPU, which is 4.38× faster than the CPU on the weakest node, to the CPU as the CPU class improves.

Index Terms—BigEarthNet, edge inference, energy efficiency, model quantization, OpenVINO, self-supervised learning, Sentinel-1/2, vision transformer.

## I. INTRODUCTION

Self-supervised pretraining is now a standard way to build reusable encoders for Earth observation, where unlabeled Sentinel imagery is abundant but labels are scarce. Masked autoencoders [1] over the vision transformer [2] are a common choice, evaluated on archives such as BigEarthNet [3] and its refined release reBEN [4]; the encoder studied here [5] follows this recipe on a joint SAR–optical corpus.

Most of this work is evaluated on accuracy alone, yet an encoder is only useful if one inference is cheap enough for the hardware actually available, often a small, passively cooled x86 box rather than a server GPU. The cost of that inference, and how it responds to post-training quantization, depends on details that are seldom reported: whether the CPU offers wide SIMD (AVX2) or integer dot-product instructions (VNNI), and whether a usable integrated GPU (iGPU) is present. These features vary widely across commodity hardware, so a single benchmark number does not tell a practitioner what to deploy.

Running learned models close to the sensor is already an active topic in Earth observation, but most of the work targets dedicated accelerators on board. Φ-Sat-1 flew a convolutional cloud detector on an Intel Myriad 2 vision processing unit [6], RaVÆn sizes a variational autoencoder for change detection on resource-limited flight hardware [7], and Ziaja et al. benchmark deep models for on-board use [8]. On the model side, post-training quantization of vision transformers is known to be delicate, since post-softmax, post-GELU and LayerNorm activations are hard to quantize, and dedicated schemes have been proposed for it [9], [10]. Much less is known about the general-purpose x86 hardware available on the ground, in small, passively cooled boxes, some of which still lack AVX2, and about how a self-supervised SAR–optical transformer behaves there when latency, energy and accuracy are measured together.

This letter characterizes the deployment of this encoder across a gradient of commodity x86 classes. Our contributions are: (i) we show that quantization is accuracy-neutral for this SAR–optical encoder, so precision becomes a pure cost decision; (ii) we characterize that cost across an ISA-capability gradient and show that INT8 pays off even on a CPU without AVX2, with a present-day VNNI reference indicating a further gain beyond AVX2; and (iii) using a reproducible RAPL plus wall-plug protocol, we show that the Pareto-optimal device is not fixed but moves from the iGPU to the CPU as the CPU class improves.

## II. MATERIALS AND METHODS

## A. Encoder, Intermediate Representation and Task

We study the encoder released with [5], a ViT-S/16 masked autoencoder [1], [2] pretrained in a self-supervised manner on the STRATA-S12 SAR–optical corpus and archived with it [11]. The input stacks six co-registered optical and SAR channels and the model returns a 384-D class token; we deploy only the encoder, with no decoder and no masking. The encoder is rebuilt from timm [12] transformer blocks, exported to ONNX and compiled to OpenVINO [13] intermediate representations (IR) at FP32, FP16 and INT8. INT8 uses posttraining quantization [14] with NNCF [15] on a real calibration subset of 512 reBEN patches, disjoint from train and test. The IRs are built once with the development toolchain and shipped unchanged, so a given precision yields identical outputs on every host: accuracy is a property of the IR and is measured once. Representation quality is assessed on reBEN [4] as a 19- class multi-label task through a linear probe on frozen features (20000 training and 10000 test patches, three seeds), following the protocol of [5] as implemented in its public evaluation harness [16]; we report micro- and macro-mAP and the change in micro-mAP relative to FP32.

## B. Hardware Gradient

We deploy on three commodity x86 nodes chosen to span a gradient of SIMD capability rather than of price (Table I): a floor node without AVX2, a mainstream AVX2 step node, and an AVX2+VNNI reference node, names used throughout the letter. The first two also carry an Intel integrated GPU (iGPU) and act as edge targets; the reference is used for CPU latency only, as a present-day point of comparison rather than as an edge target. The OpenVINO runtime is pinned to a single release on all three nodes so that the CPU and iGPU paths stay comparable.

## C. Measurement Protocol

For each node, device and precision the IR is compiled and run for $W = 2 0$ discarded warm-up passes, followed by $R = 5$ repetitions of $N = 2 0 0$ forward passes at batch size one (single stream). Latency is the wall-clock time of a repetition divided by N, reported as the mean with a 95% confidence interval (Student’s t) over the repetitions. Batch size one matches the edge use case, where tiles arrive one at a time and have to be processed with low delay; throughput under batching is outside the scope of this letter. As the network is dense and the input shape is fixed, latency does not depend on the pixel values. Energy is sampled on the host clock nominally every 0.5 s from two synchronized sources, the CPU-package counter of RAPL [17] and the active power of a wall-plug meter, and is reported per inference after subtracting an idle baseline measured over 60 s:

$$
e = \frac { E _ { \mathrm { w i n d o w } } - \bar { P } _ { \mathrm { i d l e } } \Delta t } { N } ,\tag{1}
$$

where $E _ { \mathrm { w i n d o w } }$ is the energy over a repetition of duration $\Delta t$ and $\bar { P } _ { \mathrm { i d l e } }$ the idle power. The CPU governor is set to performance. All parameters are defined in the released code<sup>1</sup> and recorded with every result (Section III-E).

TABLE I  
COMMODITY X86 NODES, ORDERED BY ISA CLASS. THE ROLE NAMES ARE USED IN THE TEXT, TABLES AND FIGURES.
<table><tr><td>Role</td><td>Machine</td><td>CPU</td><td>AVX2</td><td>VNNI</td><td>iGPU</td></tr><tr><td></td><td>Floor Moort M1</td><td>Celeron N5105 (Jasper Lake)</td><td>No</td><td>No</td><td>UHD Gen11</td></tr><tr><td>Step</td><td>Lenovo M720q i3-9100T</td><td>(Coffee Lake)</td><td>Yes</td><td>No</td><td>UHD 630 Gen9.5</td></tr><tr><td>Ref.</td><td>Dev. laptop</td><td>i7-14700HX (Raptor Lake)</td><td>Yes</td><td>Yes</td><td></td></tr></table>

## III. RESULTS

## A. Quantization is Accuracy-Neutral

Table II reports the linear-probe accuracy. FP16 matches FP32 exactly (micro-mAP 0.5494), and INT8 differs by only +0.0006 in micro-mAP, well inside the ±0.0022 seed spread, so precision is a pure cost decision.

TABLE II  
ACCURACY (LINEAR PROBE, 3 SEEDS); QUANTIZATION IS ACCURACY-NEUTRAL.
<table><tr><td>Precision</td><td>micro-mAP</td><td>macro-mAP</td><td>∆ micro vs FP32</td></tr><tr><td>FP32</td><td> $0 . 5 4 9 4 \pm 0 . 0 0 2 2$ </td><td> $0 . 3 3 5 2 \pm 0 . 0 0 0 6$ </td><td>0.0000</td></tr><tr><td>FP16</td><td> $0 . 5 4 9 4 \pm 0 . 0 0 2 2$ </td><td> $0 . 3 3 5 2 \pm 0 . 0 0 0 6$ </td><td>0.0000</td></tr><tr><td>INT8</td><td> $0 . 5 5 0 0 \pm 0 . 0 0 2 2$ </td><td> $0 . 3 3 5 8 \pm 0 . 0 0 0 7$ </td><td>+0.0006</td></tr></table>

## B. INT8 Lowers Cost on the CPU; FP16 Does Not

On the CPU, FP16 brings no benefit: it is essentially tied with FP32, and on the step node it is even slightly slower (35.74 vs 33.80 ms). INT8, in contrast, is faster on every node: 1.45× over FP32 on the floor node, 1.27× on the step node and 1.53× on the reference (Fig. 1). Energy tracks latency both at the CPU package and at the wall plug (Table III), so INT8 is also the most efficient CPU precision on the two nodes with energy measurements.

<table><tr><td>Floor CPU</td><td>207.32</td><td>220.11</td><td>143.33</td><td rowspan="5">200 (g9) 100 &#x27;s) atncy 50 30</td></tr><tr><td>Floor iGPU</td><td>50.34</td><td>50.30</td><td>62.10</td></tr><tr><td>Step CPU</td><td>33.80</td><td>35.74</td><td>26.56</td></tr><tr><td>Step iGPU</td><td>32.50</td><td>32.51</td><td>37.90</td></tr><tr><td>Ref. CPU</td><td>31.73</td><td>29.89</td><td>20.75</td></tr><tr><td></td><td>FP32</td><td>FP16</td><td>INT8</td></tr></table>

Fig. 1. Per-inference latency (ms, mean over $R = 5$ repetitions; log color scale) by node, device and precision. On the floor node the iGPU is the fastest device; on the step and reference nodes INT8 on the CPU is.

## C. INT8 Pays Off Without AVX2; VNNI Adds More

The floor node, which has neither AVX2 nor VNNI, gains 1.45× from INT8, more than the step node (1.27×). This is consistent with a memory-bound forward pass at batch size one, where the fourfold smaller INT8 weights help even without wide integer instructions. Against the reference, the step node has almost the same FP32 latency (33.80 vs 31.73 ms, the latter with a wider ±3.81 ms interval) and also carries AVX2, but its INT8 latency is clearly higher (26.56 vs 20.75 ms). With the runtime and the IR held identical, the additional INT8 acceleration on the reference points to VNNI rather than to AVX2.

## D. The Best Device Depends on the Hardware Class

On the floor node the iGPU is far faster and more efficient than the CPU (4.38× faster and 5.46× more efficient at the package at FP16; at the wall plug, 0.544 vs 3.014 J per inference), so the best configuration there is the iGPU, at FP32 or FP16 alike. On the step node, whose CPU is much stronger, the same iGPU barely helps (1.04× at FP32), and INT8 on the CPU becomes the fastest configuration (26.56 ms). Its energy matches FP16 on the iGPU at the package (0.859 vs 0.861 J) and is slightly lower at the wall plug (1.230 vs 1.263 J). On both nodes INT8 is the slowest precision on the iGPU, since these integrated GPUs run FP16 natively and gain nothing from integer weights. The Pareto-optimal deployment is thus not fixed: it shifts from the iGPU to the CPU as the CPU class improves (Fig. 1, Table III).

TABLE III  
PER-INFERENCE ENERGY (J), NET OF THE IDLE BASELINE, FOR THE CPU PACKAGE (RAPL) AND THE WHOLE SYSTEM (WALL PLUG). THE LOWEST VALUE PER NODE AND METER IS IN BOLD. THE REFERENCE NODE HAS NO ENERGY MEASUREMENT.
<table><tr><td></td><td></td><td colspan="2">RAPL (J)</td><td colspan="2">Wall plug (J)</td></tr><tr><td>Node</td><td>Device</td><td>FP32</td><td>FP16 INT8</td><td>FP32</td><td>FP16 INT8</td></tr><tr><td>Floor</td><td>CPU</td><td>2.117</td><td>2.187 1.249</td><td>2.938</td><td>3.014 1.650</td></tr><tr><td></td><td>iGPU</td><td>0.393</td><td>0.400 0.497</td><td>0.541</td><td>0.544 0.668</td></tr><tr><td>Step</td><td>CPU</td><td>1.260</td><td>1.250 0.859</td><td>1.723</td><td>1.830 1.230</td></tr><tr><td></td><td>iGPU</td><td>0.865</td><td>0.861 1.009</td><td>1.268</td><td>1.263 1.480</td></tr></table>

## E. Implementation and Reproducibility

Two safeguards support reproducibility. First, every artifact along the chain (checkpoint → ONNX → per-precision IR) is recorded in a SHA-256 manifest, and a provenance gate re-checks this chain on disk before each run, so every result traces back to the exact inputs that produced it. Second, the integrated-GPU runtime is not assumed to work: we sweep OpenVINO and driver combinations on the floor node and pin the first release (2024.6) whose kernels the Gen9/11 iGPU can compile, as newer releases regressed a normalization kernel to OpenCL 2.0 work-group functions that these devices (driver 23.43.027642, OpenCL C 1.2) cannot build. A version-control check confirms the pin does not bias the CPU numbers: FP32 and FP16 latency are unchanged across releases (+1.0 % and +0.1 %) while INT8 shifts by +6.8 %. On the reference node the same check moves FP32, FP16 and INT8 by −5.1, −8.2 and +12.2 %; all nodes are therefore measured on the pinned release. IRs, calibration data and code are at the URL above.

## IV. DISCUSSION AND CONCLUSION

Because INT8 does not cost accuracy, precision can be chosen on cost alone, and the right choice depends on the node. On a weak CPU without AVX2 the integrated GPU is the clear option, with FP32 and FP16 equivalent there, whereas on a mainstream AVX2 CPU the integrated GPU adds little and INT8 on the CPU is the fastest configuration and at least as efficient as the integrated GPU. INT8 is faster on the CPU of every node, and on AVX2 CPUs VNNI appears to add a further gain, so CPUs that expose it are the better choice when INT8 is planned.

In practical terms, at batch size one the floor node processes about 71,600 patches of 224×224 pixels per hour on its iGPU, at 0.544 J each at the wall plug, and the step node about 135,500 per hour on its CPU in INT8, at 1.230 J each.

We study a single ViT-S/16 encoder, so the accuracy-neutral result and the magnitude of the speedups may differ for other architectures or sizes. The reference node is a current laptop CPU, included to place the edge nodes against presentday hardware rather than as an edge target. As it is also the only node with VNNI, the VNNI observation rests on two non-identical machines with a similar FP32 latency, not on a single controlled variable. Device temperature was not recorded, so thermal throttling cannot be ruled out directly, although the narrow confidence intervals across repetitions on the edge nodes show no drift within a run. Extending this characterization to a set of encoders, and to newer integrated GPUs and low-power CPUs that expose VNNI, is left to future work.

## REFERENCES

[1] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proc. IEEE/CVF CVPR, 2022.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. ICLR, 2021.

[3] G. Sumbul, M. Charfuelan, B. Demir, and V. Markl, “BigEarthNet: A large-scale benchmark archive for remote sensing image understanding,” in Proc. IGARSS, 2019.

[4] K. N. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, and V. Markl, “reBEN: Refined BigEarthNet dataset for remote sensing image analysis,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), 2025.

[5] C. Sanchez-Leon, D. Chen, and D. Wang, “Deficit-driven stratified acquisition for building STRATA-S12: An open SAR–optical corpus for self-supervised learning,” 2026, arXiv:2610.17680. [Online]. Available: https://doi.org/10.48550/arXiv.2610.17680

[6] G. Giuffrida, L. Fanucci, G. Meoni, M. Batic, L. Buckley, A. Dunne, C. van Dijk, M. Esposito, J. Hefele, N. Vercruyssen, G. Furano, M. Pastena, and J. Aschbacher, “The Φ-Sat-1 mission: The first on-board deep neural network demonstrator for satellite Earth observation,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–14, 2022.

[7] V. R ˚užicka, A. Vaughan, D. De Martini, J. Fulton, V. Salvatelli,ˇ C. Bridges, G. Mateo-Garcia, and V. Zantedeschi, “RaVÆn: Unsupervised change detection of extreme events using ML on-board satellites,” Scientific Reports, vol. 12, p. 16939, 2022.

[8] M. Ziaja, P. Bosowski, M. Myller, G. Gajoch, M. Gumiela, J. Protich, K. Borda, D. Jayaraman, R. Dividino, and J. Nalepa, “Benchmarking deep learning for on-board space applications,” Remote Sensing, vol. 13, no. 19, p. 3981, 2021.

[9] Z. Yuan, C. Xue, Y. Chen, Q. Wu, and G. Sun, “PTQ4ViT: Post-training quantization for vision transformers with twin uniform quantization,” in Proc. ECCV, 2022.

[10] Y. Lin, T. Zhang, P. Sun, Z. Li, and S. Zhou, “FQ-ViT: Post-training quantization for fully quantized vision transformer,” in Proc. IJCAI, 2022, pp. 1173–1179.

[11] C. Sanchez-Leon, D. Chen, and D. Wang, “STRATA-S12: A stratified, quality-controlled corpus of co-registered Sentinel-1/Sentinel-2 pairs for self-supervised Earth observation,” Zenodo, dataset, 2026. [Online]. Available: https://doi.org/10.5281/zenodo.22699046

[12] R. Wightman, “PyTorch image models (timm),” https://github.com/ huggingface/pytorch-image-models, 2019, version 1.0.11.

[13] Intel Corporation, “OpenVINO toolkit,” https://github.com/ openvinotoolkit/openvino, version 2024.6.0.

[14] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in Proc. IEEE/CVF CVPR, 2018.

[15] A. Kozlov, I. Lazarevich, V. Shamporov, N. Lyalyushkin, and Y. Gorbachev, “Neural network compression framework for fast model inference,” arXiv preprint arXiv:2002.08679, 2020.

[16] C. Sanchez-Leon, D. Chen, and D. Wang, “STRATA-S12 evaluation harness,” Zenodo, software, 2026, version v1.0.1. [Online]. Available: https://doi.org/10.5281/zenodo.22892250

[17] K. N. Khan, M. Hirki, T. Niemi, J. K. Nurminen, and Z. Ou, “RAPL in action: Experiences in using RAPL for power measurements,” ACM Trans. Model. Perform. Eval. Comput. Syst., vol. 3, no. 2, pp. 1–26, 2018.