---
title: "Training-Free-Inpainting-Across-Domains-with-a-Frozen-Text-t"
source: https://arxiv.org/pdf/2609.00862v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:52:14"
---

# 论文速读：Training-Free-Inpainting-Across-Domains-with-a-Frozen-Text-t

## 一句话总结
本文证明只需冻结的通用SD1.5文本到图像扩散模型，配合一套固定的隐空间闭环控制器Step-PI，即可在AFHQ、CelebA-HQ与Places2三个域上完成高质量条件图像修复，全程无需权重微调、数据集适配或学习的修复专用条件通道。

## 研究问题与动机
- 现有扩散修复器依赖专项训练权重、双分支架构或可学习的掩码条件接口，通用T2I模型本身不具备此类能力，直接复用存在显著空白。
- 实际部署中重新微调高分辨率扩散模型成本高昂，希望探索仅通过推理时隐空间调控来激活冻结先验的可行性。
- 已知区域的投影仅能锚定坐标，无法保证掩码边界的局部缝合质量与生成区域内部的长程结构连贯性，需在反向轨迹中引入持续反馈。
- 边界修复侧重局部连续，内部续写侧重全局语义与纹理匹配，两者在反向过程不同阶段的控制优先级应动态变化，提示需要时变调制机制。

## 核心贡献（创新点）
- **配置锁定的跨域免训练修复：** 提出仅依赖冻结SD1.5与固定控制器即可在三个自然图像域完成条件修复的方案，与依赖Learned masking interface、BrushNet双分支或PixelHacker专项微调的现有工作本质不同。
- **Step-PI边界-内部PI结构控制器：** 将经典PID分解为当前法化反馈与指数折扣历史状态的组合，与PILOT周期性潜变量优化或LanPaint结合Langevin动力学的做法在控制结构与可解释性上有本质区别。
- **预定义四字段释放调度：** 设计随噪声调度系数$\bar{\alpha}_{t_k}$与反向进度$p_k$变化的$r_B/q_k/h_k/r_I$调制向量，区别于传统均匀施加强度或依赖后验不确定性估计的动态调整机制。
- **严格的字段等同消融与统计验证：** 构建P-Guidance→PI与PI→Step-PI两组仅改变单一组件的对照实验，证明持久状态与完整调度分别在全部15个数据集-指标单元上带来显著正向增益。

## 方法详解
- **闭环节点接口：** 在确定DDIM反向轨迹（$t_N \to t_0$）的每步$k$，先执行投影$\Pi_{t_k}(z^k) = (1-M_z)\odot q_t^{\text{known}} + M_z \odot z$锚定已知区，经冻结U-Net+CFG得到干净估计$\hat{z}_0^k$与基底隐变量$z_{\text{base}}^{k-1}$，控制器$\mathcal{C}_k$基于$\hat{z}_0^k$、原图$y$、掩码$M$及历史状态输出动作$a_k$，作用于$z_{\text{base}}^{k-1}$后再投影进入下一步。
- **边界-内部双目标反馈：** 边界目标$\mathcal{L}_B = \sum \lambda_j^B \mathcal{L}_j^B$包含已知区RGB保真、接缝配对、支持掩码TV平滑与RGB梯度匹配；内部目标$\mathcal{L}_I = \sum \lambda_j^I \mathcal{L}_j^I$包含多尺度低频延续、深度加权内部一致性、半径32环统计匹配与高频频谱对齐。梯度在投影隐变量处求取并全张量$L_2$法化：$g_j^k = \text{Normalize}_2[-M_z \odot \nabla_{\tilde{z}^k}\mathcal{L}_j]$。
- **持久PI状态更新：** 借鉴经典PI分解，当前反馈与历史状态分离：$\xi_j^k = \mathcal{P}_j[m_j \odot \xi_j^{k+1} + (1-m_j) \odot (w_j \odot g_j^k)]$。边界状态采用空间变化保留系数$\beta_B$与norm裁剪$\nu_B=1$，内部状态采用标量折扣$\gamma_I=0.9$。该递推等价于每步在当前加权方向与保留历史之间的有界折中。
- **预定义释放调度与动作装配：** $\rho_k=(r_B(t_k), q_k, h_k, r_I(t_k))$，其中$r_B$与$r_I$依赖$\bar{\alpha}_{t_k}$，$q_k$与$h_k$依赖归一化步进展$p_k$并通过smoothstep分段。比例/积分/融合项经CapNorm与全局clamp后仅作用在未知区掩码：$a_k = \text{Clamp}[-a_{\max}, a_{\max}][M_z \odot (u_B^k + u_I^k)]$。最终RGB合成时精确回填已知像素，保证已知区零失真。
- **零参数更新：** 全程不修改U-Net与VAE权重，不引入可学习条件通道，仅依靠前向推理与在线梯度计算完成隐空间闭环调控。

## 实验与结果
- **数据集与基准：** 构建Main35（3,500配对样本），覆盖AFHQ v2（1,300）、CelebA-HQ（1,400）、Places365子集（800），含35种固定100样本掩码协议；评估指标为Masked L1、Boundary L1、Masked LPIPS、Boundary LPIPS、CLIP-Q。
- **消融结果：** P-Guidance→PI（加入持久状态）在全部15个单元格提升，95% Bootstrap区间均不含零；PI→Step-PI（加入完整调度）同样全提升，其中Boundary L1改善达+24.4%，Masked L1改善+7.8%，Masked LPIPS改善+4.7%。
- **跨域迁移：** 控制器仅用Main35不重叠的CelebA-HQ pilot开发并冻结，直接迁移至AFHQ与Places2均单调优于前序版本；Fixed prompt-presence干预显示三域对文本条件响应显著。
- **外部对比：** Step-PI在LanPaint与PILOT（同为vanilla SD1.5免训练基线）上包揽全部15个等数据集宏指标第一；SD-Inpaint、BrushNet、PixelHacker绝对分数更高，但依赖数万至数十万步离线优化与专项数据。
- **计算成本：** 50步DDIM下Step-PI在A100耗时约30.48 s/img，峰值显存15.55 GiB，显著慢于LanPaint（11.38 s）与已训练方法，代价为每步两次反馈梯度计算；释放调度本身未引入可分辨的额外开销。

## 相关工作脉络
- **训练类修复（SD-Inpaint, BrushNet, PixelHacker）：** 依赖专项权重与Learned masking/ dual-branch接口提供强质量参照，与本文“冻结通用模型+零训练”的定位形成对照而非直接竞争。
- **免训练基线（LanPaint, PILOT）：** 同为vanilla SD1.5免训练路线，LanPaint结合Langevin动力学与Euler/Karras采样，PILOT通过周期性潜变量梯度优化，两者均未引入本文明确定义的PI持久状态与四字段时变释放机制。
- **扩散逆问题/后验采样（RePaint, DING, GradPaint）：** 侧重不同逆问题设定或依赖VJP/高斯后验近似，未直接针对“通用T2I跨域修复+测试时隐控”这一特定接口展开。
- **测试时控制综述：** 涵盖后验梯度、可微引导、轨迹优化等广泛框架，本文聚焦于在不修改模型权重的硬约束下，将经典控制理论落地到扩散反向轨迹的隐空间调控，提供可解释的模块化实现。

## 局限性与未来方向
- 每步两次反馈梯度计算导致推理开销较大，难以直接扩展至高分辨率或实时交互场景。
- 控制器超参与调度曲线仅在CelebA-HQ小规模pilot上开发并冻结，未系统评估不同scheduler、步数、分辨率或backbone的泛化边界。
- 文本条件仅使用粗粒度类别prompt（如“一只猫”、“室内浴缸”），未检验细粒度指令遵循、多对象控制或 paraphrase 鲁棒性。
- 释放调度的四个字段存在串行耦合（$q_k$与$r_I$共同缩放内部输出），独立性未严格分解，可能存在更低维的等效表达。

## 研究启发与可借鉴点
- **控制理论视角的扩散采样：** 将经典
