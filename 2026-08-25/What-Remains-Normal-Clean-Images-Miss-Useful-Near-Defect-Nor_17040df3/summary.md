---
title: "What-Remains-Normal-Clean-Images-Miss-Useful-Near-Defect-Nor"
source: https://arxiv.org/pdf/2608.23299v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:53:24"
field: "工业视觉异常检测"
keywords: ["industrial anomaly detection", "normal-only training", "memory-based detection", "reconstruction-based detection", "context intervention", "pixel localization"]
innovations: ["固定预算诊断揭示清洁内存遗漏近缺陷正常参考", "BOUNDARYSUPPORT 通过上下文扰动暴露像素保留邻域特征", "匹配对照证明 altered-context 特征优于 clean-view 对应特征"]
benchmarks: ["MVTec AD", "VisA", "Real-IAD"]
---

# 论文速读：What-Remains-Normal-Clean-Images-Miss-Useful-Near-Defect-Nor

## 一句话总结
论文发现纯清洁训练图像构建的异常检测器会遗漏真实缺陷图像中紧邻缺陷的正常 patch，提出 BOUNDARYSUPPORT 方法通过在清洁图像上施加程序化上下文干预并排除受修改 token，仅利用像素保留的邻域特征作为额外参考，在内存库与重建两条主流范式下均显著提升像素级定位性能。

## 研究问题与动机
- 工业视觉异常检测通常仅使用无缺陷训练图像，假设清洁 patch 足以覆盖测试时遇到的所有正常区域，但该假设未被直接验证。
- 真实缺陷图像中大部分 patch 仍为正常标注，尤其是紧邻缺陷边缘的正常 patch，其局部外观虽正常但上下文与清洁图像不同，容易被纯清洁内存库遗漏。
- 现有正常参考/重建目标几乎全部来自清洁训练图，若测试时正常 patch 在候选集中代表性不足，则在最终打分前就已完成“失败”。
- 已有方法（如 SoftPatch、InReaCh、FUN-AD）关注如何处理含污染的清洁池，但未解决“清洁池本身缺少某些有用正常表征”的问题。

## 核心贡献（创新点）
- **固定预算诊断揭示清洁池不足**：在 MVTec AD 上，仅将 GT 标注的真实缺陷图像正常 patch 引入候选池即可将 P-AP 从 73.34 提升至 76.95，且两-cell 近缺陷带恢复 94.70% 增益，证明问题在于“参考身份”而非“数量”。
- **BOUNDARYSUPPORT 纯清洁训练干预**：在不使用任何真实缺陷图像的前提下，通过程序化合成改变周边上下文，排除被修改 token，仅用像素保留邻域的 altered-context 特征作为额外正常参考或重建目标。
- **跨范式与对照证据锁定有效因子**：在内存与重建两侧均证明 altered-context 特征显著优于 clean-view 对应特征；距离解析显示最终得分变化集中在真实缺陷邻域正常 patch。
- **重新定义正常证据构成对定位的影响**：定位增益并非来自单调提升缺陷响应，而是通过重塑正常/缺陷响应相对排序实现，同时给出失败案例与指标权衡分析。
- **可复用实验范式**：提供 matched fixed-budget diagnostic、feature-identity control、distance-resolved score analysis 等可直接迁移的归因评估框架。

## 方法详解
- **合成干预**：对清洁图像 $x$ 应用三种固定程序变换生成 $\tilde{x}$ 与名义插入掩码 $A$，包括 CutPaste-Scar、同类 Poisson paste（NSA-style）、同图局部 warp+Poisson paste；Real-IAD 限制 donor 与同类别、同相机视图。
- **像素保留集构造**：除名义掩码外还检测实际 RGB 变化 $D(u)=\mathbf{1}[\frac{1}{3}\sum_c|\tilde{x}_c(u)-x_c(u)|>1]$，经 max-pool 到 token 网格后得排除集 $C=P(A)\lor P(D)$，保留 ring $R=\mathrm{Dilate}(P(A),r)\setminus C$，主配置 $r=2$。
- **内存分支**：冻结 DINOv2 ViT-B/14，提取层 $[-3,-6,-8,-9]$，投影至 512 维；候选发现采用 $k{=}1$ 欧氏距离与四层平均分 $s_{\mathrm{cand}}$，筛选条件为 $p\in R$、$s_{\mathrm{cand}}(x,p)\le Q_{0.50}$、$\Delta_p\ge Q_{0.90}$、$\Delta_p>0$；选中 altered-context 特征与清洁候选合并后仍按相同 k-center 挑选出恰好 $B$ 个参考，保持原始银行预算不变。
- **重建分支**：基于冻结 encoder 的 Dinomaly 架构，保留原始 hard-mined 目标 $\mathcal{L}_{\mathrm{HM}}(x)$，新增 $\mathcal{L}_R=\frac{1}{2}\sum_j\mathrm{mean}_{p\in R}[1-\cos(\mathrm{sg}(E_j(\tilde{x})_p),G_j(\tilde{x})_p)]$，总目标 $\mathcal{L}=\mathcal{L}_{\mathrm{HM}}(x)+0.05\mathcal{L}_R$；合成图整体送入 encoder，C 内 token 不参与辅助损失但仍参与上下文注意力。
- **测试时不变**：不引入合成输入或掩码，沿用基线读路（内存取最高 0.5% patch 均值，重建取最高 1% pixel 均值）。

## 实验与结果
- **诊断实验（MVTec，seed 0）**：Clean 73.34 → Fixed-budget GT-normal oracle 76.95（+3.61 P-AP），同候选并扩 5.3% 预算仅达 76.96；近缺陷（$0<d{\le}2$）达 76.76，回收 94.70%。Inpainting 后仍有 +11.72/6.11，跨缺陷类型 transfer 亦为正。
- **BOUNDARYSUPPORT 主结果（三 seed 配对均值）**：
  - 内存：MVTec +2.21±0.04、VisA +1.82±0.03、Real-IAD +1.76±0.04 P-AP，覆盖 171/171 类别-seed。
  - 重建：MVTec +5.01±0.07、VisA +2.27±0.38、Real-IAD +4.20±0.12 P-AP，覆盖 158/171 类别-seed。
- **特征身份对照（seed 0）**：重建中将 altered 目标换为 clean 目标使 P-AP 分别回落 -7.13（MVTec）/ -6.36（VisA），而 altered 目标分别提升 +4.94 / +2.09；内存中同位置替换特征使 altered 在 26/27 类别胜出。
- **空间分布**：MVTec 内存近缺陷正常 patch 得分下降最多（$100\Delta s=-0.37$），中远与清洁正常轻微上升，表明提升来自排序优化而非全局压制。
- **最强结果**：重建 MVTec P-AP 提升 +5.01±0.07，对应 base 68.96→73.98，为全设定最大增益。

## 相关工作脉络
- **PatchCore（Roth et al., 2022）** 与 **ProCon（Chae et al., 2026）**：以清洁 patch 构建非参数内存；本文指出其候选集合本身可能缺乏关键 near-defect 正常表征。
- **Dinomaly（Guo et al., 2025）**：冻结 encoder 重建 normal 特征；本文在其主干上增加仅作用于像素保留 ring 的辅助目标，不改读路。
- **CutPaste（Li et al., 2021）**、**DRAEM（Zavrtanik et al., 2021）**、**NSA（Schluter et al., 2022）**：将合成异常作为异常监督或还原目标；本文把合成当作上下文扰动工具，学习目标仍是像素保留邻域的 altered-context 正常特征。
- **SoftPatch（Jiang et al., 2022）**、**InReaCh（McIntosh & Branzan Albu, 2023）**、**FUN-AD（Im et al., 2025）**：从含污染池中剔除不可靠 patch；本文研究的是“本就清洁但覆盖不全”的补充问题。
- **INP-Former（Luo et al., 2025）**：从含异常图中提取内在正常原型；本文通过诊断澄清这类有用原型在纯清洁池中的缺失程度，并给出无需真实缺陷的暴露方式。

## 局限性与未来方向
- 仅验证于 DINOv2 ViT-B 系列编码器，未扩展至 register-ViT 或其他 backbone。
- 仅覆盖单一内存分支（ProCon）与单一重建分支（Dinomaly），跨其他架构的外推性待验证。
- 定位提升并非单调改善所有指标；Real-IAD 重建 I-AUROC 下降 0.52，部分类别出现 P-AP 下滑。
- 合成干预依赖预定义程序变换，未探索直接根据部署数据识别缺失正常 patch 的策略。
- 近缺陷 band 的距离分层仅在皮革与瓷砖两类中详细展开，全类别逐距分析未完整呈现。

## 研究启发与可借鉴点
- **固定预算 matched diagnostic**：保持编码器、评分、候选数不变，仅更换候选身份即可量化“正常证据完整性”的贡献，便于消融对比。
- **像素保留 + 上下文变化**：用 RGB 差检验替代纯几何环，避免插值/融合外溢导致的伪保留，设计可直接移植到其他上下文增强范式。
- **feature-identity control**：固定位置与预算仅换特征来源，能清晰分离“空间挖掘”与“表征价值”，建议纳入同类工作的标准对照组。
- **距离-得分解析**：将 $\Delta s$ 按到缺陷格点距离分层统计，定位解释更扎实；可在本文基础扩展到更多类别与多尺度邻居定义。
- **指标权衡坦诚呈现**：承认并解释 P-AP 提升与图像级指标波动的共存，为后续工作设定更合理的评测预期。

## 关键术语表
- **P-AP**：基于像素排序的平均精度，本文首要定位评测指标。
- **I-AUROC**：图像级 ROC 曲线下面积，反映图片级异常/正常判别。
- **Fixed-budget GT-normal oracle**：保持内存大小不变、仅将真实缺陷图中 GT 正常 patch 并入候选的对照实验。
- **BoundarySupport**：本文提出的方法，通过程序合成改变上下文并仅利用像素保留邻域特征。
- **Altered-context feature**：在同一保留位置从上下文被扰动的合成视图中提取的特征。
- **Near-defect normal patch**：位于标注缺陷格点两格以内的正常 patch，是增益的主要来源。
- **ProCon readout**： Projection-Consistency 测试时读取器，基于多近邻软重构与温度缩放。
- **Reconstruction residual**：Dinomaly 类方法中 decoder 输出与 frozen encoder 目标的残差，用作异常证据。

## 可复现要素
- **数据集**：MVTec AD、VisA、Real-IAD，均为公开数据集。
- **代码**：https://github.com/jw-chae/boundary_support（论文声明开源）。
- **权重**：使用预训练 DINOv2 ViT-B/14 与 dinov2reg vit base 14，论文未提及额外微调权重发布。
- **关键超参**：内存比例 MVTec/VisA 为 5%、Real-IAD 为 1%；ring 半径 $r=2$；重建辅助权重 0.05；重建训练 5,000 次、batch size 16、StableAdamW、初始 lr $2{\times}10^{-3}$、余弦衰减至 $2{\times}10^{-4}$、gradient clipping 0.1、warmup 100 步。
- **输入几何**：MVTec 保持宽高比缩放至 448 后中心裁至 392×392；VisA/Real-IAD 方形 448 裁至 392×392；patch 14×14，token 网格 28×28。
- **精度**：内存候选/银行距离 FP32，缓存 FP16；重建训练 FP32。
- **未提及项**：论文未提供独立 checkpoint 下载链接，仅声明将附结果文件与脚本。
