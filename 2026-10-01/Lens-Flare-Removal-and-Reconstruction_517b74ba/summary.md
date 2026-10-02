---
title: "Lens-Flare-Removal-and-Reconstruction"
source: https://arxiv.org/pdf/2609.39527v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:45:28"
field: "计算摄影与3D视觉"
keywords: ["lens flare removal", "3D Gaussian splatting", "diffusion model", "flare representation", "multi-view reconstruction", "artefact decomposition"]
innovations: ["LoRA微调单步扩散模型实现大反射光晕移除（20小时收敛）", "相机平面锚定1D高斯表示首次实现跨视角一致的3D光晕显式重建", "场景-光晕解耦联合损失支持可迁移编辑与Transfer"]
benchmarks: ["Flare7K++", "VFX large-reflective-flare benchmark"]
---

# 论文速读：Lens-Flare-Removal-and-Reconstruction

## 一句话总结
本文提出了首个镜头光晕移除与三维重建联合框架：利用LoRA微调单步扩散模型实现复杂大光晕移除，并提出基于相机平面锚定高斯的3D光晕表示，可将含光晕的多视角图像解耦为无光晕场景与可编辑重建的光晕组件。

## 研究问题与动机
- 镜头光晕是相机成像系统的 artifacts，非场景本身内容，会严重破坏下游任务（如3D场景重建、NeRF/3DGS）的画质；现有方法仅处理小散射光晕，面对覆盖整幅图像的反射光晕时失效。
- 已有光晕模拟（程序化、物理射线追踪）依赖精确镜头参数与波长尺度建模，计算昂贵且无法复现特定镜头/光源的真实模式；而每图独立2D移除模型又不具备跨视角一致性与可移植性。
- 真实拍摄无法同时获得有/无光晕配对数据；现有benchmark（Flare7K++）仅覆盖散射与小型反射光晕，缺乏全帧大面积反射光晕评测。
- 缺乏可在任意视角实时渲染、并与3DGS栅格化管线兼容的光晕3D表示，导致光晕信息在多视角重建中要么被烘焙进场景、要么被完全忽略。

## 核心贡献（创新点）
- **LoRA单步扩散光晕移除模型**：在Difix3D+基础上以rank-16 LoRA微调VAE编码器/解码器与UNet，20小时即可收敛；与基线4天训练/从头训练相比效率极高。
- **相机平面锚定1D高斯光晕表示**：将每条光晕分量参数化为沿"主点↔光源投影"线的1D高斯集合，经MLP形变后反投影至近裁剪面+Δ深度处作为标准3DGS原语；首次实现镜头光晕跨视角3D一致的显式重建。
- **场景-光晕联合解耦重建管线**：分别渲染纯场景与场景+光晕合成图，分别以移除模型输出与原始输入作监督（λ=0.9），实现可分量的Scene Gaussians + Flare Gaussians双分支优化。
- **光晕到新图像/新场景的跨域迁移**：保留源镜头特征的形状与颜色，通过查询源相机-光源条件化的MLP并将高斯锚定至目标相机/光源，支持实时可见性测试与后处理艺术缩放（2×等）。
- **开源大型反射光晕数据集与benchmark（VFX）**：从公共VFX素材提取1866张全帧反射光晕图像（中位覆盖78%，与Flare7K++的0.39%不相交），并合成程序化环形光晕；构建新benchmark填补大反射光晕评测空白。

## 方法详解
### 3.1 光晕移除模型
- **成像模型**：采用加法线性辐射量合成 $I = (I_{\text{scene}}^{\gamma} + I_{\text{flare}}^{\gamma})^{1/\gamma}$，$\gamma \in [1.8, 2.2]$近似逆CRF；仅用于合成训练数据。
- **扩散主干**：复用Difix3D+的单步latent diffusion（VAE + text-conditioned UNet）；引入LoRA适配器（rank=16），并反常地同时微调VAE Encoder/Decoder（因光晕污染图像对VAE属OOD）。
- **训练损失**：$\mathcal{L}_{\text{diff}} = \|I_g - I_o\|_2^2 + \text{LPIPS}(I_g, I_o)$；监督目标$I_g$额外叠加光源区域（仅Flare7K++散射benchmark使用），提升光源恢复质量。
- **训练数据**：Flare7K++散射/反射 + 自建VFX大反射（1866张，90段视频留18段作test） + 程序化环形光晕 + Flickr24K背景；文本提示："remove flares and keep the original image. High quality. Retain original colors"。

### 3.2 光晕表示与重建
- **1D高斯约束**：每分量由标量均值$\mu \in \mathbb{R}$编码沿主点-光源线的径向偏移；其他属性$\{s, \Sigma, \sigma, sh\}$仍为3D。
- **形变MLP**：基于Instant-NGP哈希网格（16层，2特征/层）学习$\delta_x = f_x(\mu, c, l;\theta_{hx})$（Eq.3）与$\delta_f = f_f(\mu, c, l;\theta_{hf})$（Eq.5）；光源位置修正$\delta_l = f_c(c, l;\theta_{hl})$（Eq.6）。
- **投影与反投影**：$\mu' \mapsto \mu'((l+\delta_l)-p)+p$得2D像点，再反投影至近裁剪面+0.21深度处发射为标准3DGS原语。
- **多光源**：每个光源实例化独立的一套flare Gaussian + 形变网络；只需3D光源位置（不强依赖可见性）。
- **光源定位**：Grounded-SAM2检测"light source"框→HSV均值>190过曝门控→重叠框去重（保留最小面积）；跨视角2D位置用TRASE聚类聚合为3D光心。
- **联合损失**：
  $$
  \mathcal{L} = \lambda(\|I_s-I_o\|_1 + (1-\text{SSIM}(I_s,I_o))) + (1-\lambda)(\|I-I_{\text{input}}\|_1 + (1-\text{SSIM}(I,I_{\text{input}}))), \quad \lambda=0.9
  $$
  其中$I_o$为移除模型输出（scene supervision），$I$为场景+光晕联合渲染（composite supervision）。
- **渲染开销**：单光源10,965个flare高斯+162,283个scene高斯，RTX 4090 1080p下总帧时10.79 ms（92.7 FPS），形变占主导，3光源11.1 ms；与vanilla 3DGS 6.74 ms相比仅+1.8%栅格化成本。

## 实验与结果
- **移除benchmark（Flare7K++散射）**：Ours-wGlare PSNR 26.68 / SSIM 0.8993 / LPIPS 0.0777，超越Flare7K++（26.42）、ACL-FR（25.26）、FR（24.49）等；Ours无glare版26.96/0.885/0.0767略有取舍。
- **新增VFX大反射benchmark（128张）**：Ours PSNR 27.06 / SSIM 0.9409 / LPIPS 0.0659；Ours-wGlare 25.93/0.9366/0.0671；均大幅领先FlareX（24.17）、Flare7K++（21.37）等。
- **多视角3D重建一致性（5场景，Tab.3）**：以各移除模型输出训练3DGS，在真实无光晕test view上评估；Ours平均PSNR 23.93 / SSIM 0.837 / LPIPS 0.220，Ours-wGlare 23.24 / 0.834 / 0.226，显著优于vanilla 3DGS（18.68/0.741/0.318）、BilateralGrid（18.96/0.787/0.268）、Flare7K++（22.40/0.815/0.252）等。
- **解耦重建（9场景，Tab.4）**：Ours在Scene分解上取得平均PSNR 31.79 / SSIM 0.922 / LPIPS 0.182；Flare分解平均PSNR 33.27 / SSIM 0.874 / LPIPS 0.270；综合Scene+Flare PSNR 33.06 / SSIM 0.937 / LPIPS 0.170，最优于所有ablation。
- **光晕迁移（Fig.1,14）**：将源场景光晕Transfer至新2D图像与新3D场景（含太阳光源），保持in-distribution形状与颜色；Gaussian缩放2×可后处理控制表观强度/范围。
- **光源位置鲁棒性（Sec.5.5）**：给3D光位加$\sigma \in \{0, 0.1, 0.5, 1.0\}$高斯偏移（最大~262 px像素误差）；保留$\delta_l$或$\delta_f$任一即可 graceful degradation（composite仅-1.5 dB，flare-1.4 dB）；两者全删则崩溃（-4+ dB）。无G-SAM2初始化（改用场景质心+噪声）仍能恢复1D直线结构。
- **训练效率**：移除模型20小时（A100 ×1 GPU，仅更新≈19.6M参数/~2%模型）；基线需>4天。

## 相关工作脉络
- **Flare7K/Flare7K++**（Dai et al. 2022/2023）：奠基性散射/小型反射光晕数据集与UFormer基线；本文在其benchmark上验证并与之对比，但指出其仅覆盖小面积反射、忽略环状大反射。
- **DifFlare**（Zhou et al. 2024）、**ACL-FR**（Zhou et al. 2025）、**LightsOut**（Tsai et al. 2025）、**FlareX**（Lishen et al. 2025）：各自聚焦散射或小反射，未针对全帧大反射设计；本文用LoRA单步diffusion+自建VFX数据集突破该局限。
- **BRacketFlare**（Dai et al. 2023 CVPR）：利用多曝光对称性去除单个小型反射光晕；本文推广至任意视角连续视频与全帧大反射。
- **GN-FR**（Matta et al. 2024 BMVC）：针对3D场景小散射光晕做mask-based剔除；本文指出其无法处理大散射/反射导致信息大量丢失，并提出跨视角一致移除与显式光晕表示。
- **Koreban & Schechner**（2009）：首次证明旋转对称镜头的鬼影落在"主点-光源投影"连线上；本文将此几何先验嵌入可微3DGS管线作为1D约束。
- **3DGS/Def-3DGS**（Kerbl et al. 2023; Yang et al. 2023）：标准静态/动态高斯溅射均假设3D一致性；本文揭示光晕本质3D不一致，提出camera-anchored替代表示。

## 局限性与未来方向
- **环形光晕无法表达**：高斯中心不透明、径向衰减的核函数天然难以建模中空环形profile（Fig.17）；仅靠程序化训练数据辅助2D移除，3D表示仍是开放问题。
- **遮挡光源隐式处理**：被场景几何遮挡的光源仅靠表示模型隐式拟合，缺乏显式occlusion-aware机制；离屏光源可投影处理，但在视内被遮挡时边界不清晰。
- **分布外光晕移除受限**：移除模型受训练集光晕类型（散射+反射+程序化环）约束，分布外形态难 convincingly 移除。
- **合成-真实gap**：训练数据均为合成复合（Clean Image + Flare Layer），缺乏真正光学隔离的配对采集；关闭该gap需专用capture rig。
- **外推视角迁移风险**：transfer时将source camera/query light代入MLP可保持in-distribution形状，但若目标相机深度/角度远离训练分布则可能失真。

## 研究启发与可借鉴点
- **"OOD-aware 骨干微调"范式**：当输入分布偏离VAE预训练域时（光晕污染图像），对Encoder也施加LoRA微调比仅调Decoder/UNet更有效；该策略可迁移至其他缺陷/artifact去除任务（雪花、摩尔纹、镜头畸变）。
- **1D先验约束降低3D优化自由度**：利用物理几何（主点-光源直线）把3D高斯降为1D标量$\mu$，配合fine deformation $\delta_f$吸收不对称残差；此类"强几何先验+弱残差修正"范式可复用于其他具有对称性的成像伪影表示（如光斑bokeh、衍射星芒）。
- **双重监督解耦训练**：scene branch以移除模型输出为参考（clean target），flare branch以输入-合成差值为残差target，形成"soft ground truth"而非硬阈值mask；可迁移至任何需要分离"内容 vs artifact"的3D表征任务。
- **检测噪声鲁棒性设计**：$\delta_l$（光源位置修正）与$\delta_f$（ finer平面形变）互为冗余防御，单一组件失效时仍有 graceful degradation；该"多层柔化前端的可学习容错"策略适用于依赖易错detectors的end-to-end管线。
- **训练效率量化**：LoRA rank=16 + VAE encoder解冻使参数量仅~2%可训，20小时对4天基线形成数量级优势；提示在3D视觉任务中优先探索低秩适配+部分解冻而非全量微调。

## 关键术语表
- **Scattering flare（散射光晕）**：由镜片灰尘、划痕、缺陷引起的耀斑、 shimmer与 streak，通常面积较小。
- **Reflective flare（反射光晕）**：由镜片组内部反射产生的鬼影/光环，可覆盖整幅图像，形状随镜头与光源相对位置变化。
- **Principal point（主点）**：相机光轴与像平面的交点，是反射光晕对称中心的几何参考。
- **3D Gaussian Splatting（3DGS）**：用各向异性3D高斯原语显式表示场景辐射场，支持实时可微栅格化与增量优化。
- **LoRA（Low-Rank Adaptation）**：在预训练模型权重上叠加低秩矩阵适配器，以极少参数量实现任务适配。
- **Deformation MLP**：以相机位姿与光源位置为条件、以Instant-NGP哈希网格编码输入的MLP，输出高斯几何/外观参数的偏移量。
- **VFX benchmark**：本文新建的大反射光晕评测集，基于ActionVFX公开素材的1866张图像，中位覆盖面积达78%。
- **Decomposed reconstruction**：将含光晕输入同时分解为scene-only渲染与flare-only残差两部分的可分离3D重建。

## 可复现要素
- **数据集**：
  - Flare7K++（公开，Dai et al. 2023）：散射/小反射基准。
  - Flickr24K（公开，Zhang et al. 2018）：无光晕背景源。
  - VFX dataset（本文自建）：1866张全帧大反射光晕图，源自动作VFX Stock Footage [1]；18段视频独立held-out为test。
  - 9场景多视角视频（本文采集，作者主页含补充视频）；5场景移除benchmark（采集数据，test view特意选无光晕位）。
  - 程序化环形光晕：随机中心+半径+径向渐变遮罩合成。
- **代码/权重**：论文正文与附录未明确声明开源链接；模型权重与采采集视频在作者主页/supplementary webpage展示（需访问arXiv对应项目页确认）。
- **关键超参**：
  - LoRA rank=16；学习率$1\times10^{-4}$，batch=1，200k iter，≈20 h（A100×1）。
  - 扩散时间步$t=199$（满分200）。
  - 监督叠加光源仅Flare7K++用；VFX benchmark不加。
  - 损失权$\lambda=0.9$。
  - 形变MLP：Instant-NGP 16层/2特征，基础分辨率16，表大小$2^{17}/2^{2^{15}}$；MLP 3层×128 ReLU。
  - 光源深度安置：近裁剪面+0.21 camera units。
  - Adam lr=$10^{-4}$；flare position lr $1.6\times10^{-4}\to1.6\times10^{-6}$（指数衰减）；scale/rotation/opacity/sh学习率分别为0.05/0.1/$10^{-3}$/$2.5\times10^{-3}$。
  - 30k iter训练；每500 iter clone/prune；clone ratio=0.2，opacity阈值0.005/0.002。
  - Grounded-SAM2 box阈0.35；HSV均值>190过曝门控；重叠框保留最小面积。
