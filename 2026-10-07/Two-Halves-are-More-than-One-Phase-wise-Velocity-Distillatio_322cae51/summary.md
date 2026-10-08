---
title: "Two-Halves-are-More-than-One-Phase-wise-Velocity-Distillatio"
source: https://arxiv.org/pdf/2610.08070v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 09:46:59"
field: "扩散模型快速生成"
keywords: ["diffusion distillation", "one-step image generation", "phase-wise expert", "velocity distillation", "text-to-image", "flow matching"]
innovations: ["将扩散轨迹分解为时序阶段并分配半大小专属专家学习局部平均速度，在单次完整前向预算下实现高质量生成", "引入相位特异性判别器对抗 L2 速度回归的过平滑倾向", "证明两半大小相位专家优于单全尺寸学生模型"]
benchmarks: ["ImageNet 256x256", "GenEval", "TIIF-Bench", "Qwen-Image-Bench", "DPG-Bench", "WISE"]
---

# 论文速读：Two-Halves-are-More-than-One-Phase-wise-Velocity-Distillation

## 一句话总结
PVD 将扩散生成轨迹分解为粗结构（semantic drafting）和细细节（detail refinement）两个阶段，每阶段分配一个半大小专用专家学习局部平均速度，以等价于一次教师完整前向传播的累计计算预算，实现高质量单步/少步图像生成，显著优于单片全尺寸学生模型。

## 研究问题与动机
1. **单片学生模型的容量瓶颈**：现有单步蒸馏方法将全部生成预算分配给单一全尺寸学生，难以同时建模粗到细的异构传输（early 建立全局低频结构，late 补充高频细节），导致输出过平滑、细节丢失甚至生成崩溃。
2. **回归条件平均的固有缺陷**：噪声状态可能对应多个合理干净目标，单一模型为最小化训练损失倾向回归条件平均，尤其在复杂 T2I 任务上表现更差。
3. **现有少数步骤方法无法突破单完整前向预算**：如 Phased DMD、Hierarchical Distillation 等方法虽引入阶段划分，但推理仍需多次生成器评估，不满足严格单次 teacher 前向预算。
4. **扩散过程的相位结构化本质未被充分建模**：扩散过程本身具有"先导航后细化"的两阶段动力学（参考 [31, 32]），但多数蒸馏方法对此缺乏显式时序分解。

## 核心贡献（创新点）
1. **提出 PVD 轨迹分解蒸馏框架**：将扩散概率流 ODE 分解为时序分段的平均速度传输，在单次完整前向计算预算约束下运行，区别于 MeanFlow/FACM 等单片平均速度学习。
2. **两半大小相位专家优于单全尺寸学生**：通过局部均速度目标训练早/晚期专家，使模型容量与扩散粗到细动力学对齐；即使累计计算减半，仍显著超越同等预算下单片模型。
3. **T2I 任务引入相位特异性判别器**：针对 L2 速度回归的过平滑倾向，为每个阶段配备专注全局结构一致性与细粒度细节的判别器，提供对抗性补充监督。
4. **高效参数部署**：专家以轻量级 LoRA 模块实例化共享半大小骨干，在 SD3.5-Medium/FLUX.1-dev/Qwen-Image 上分别减少激活参数 50.89%/49.75%/49.10%，峰值显存减少 46.49%/45.76%/48.36%。
5. **系统性实验验证**：在 C2I（ImageNet 256×256 FID=1.48）和多 T2I 骨干（SD3.5/FLUX/Qwen-Image）上均达当前最优蒸馏效果，并在 Unsplash 真实数据上微调后展现出更强的摄影级写实能力。

## 方法详解
1. **骨架提取与初始化**：从教师 DiT 中保留前 3 个与后 3 个 block，均匀采样中间层使深度减半，得到半大小骨干 θ；在连续时间域 t~U[0,1] 上以流匹配损失 L_FM = E[||v_θ(x_t,t) - v_φ(x_t,t)||²] 微调，初始化共享基础。

2. **时序分割**：将 [0,1] 分为早期 Semantic Drafting 阶段 T₁=[0, 0.4] 与晚期 Detail Refinement 阶段 T₂=[0.4, 1]；消融表明不对称分割 [0.4, 0.6] 最优（早阶段需更短区间建立结构，晚阶段需更长区间细化细节）。

3. **平均速度学习目标**：每个专家 u_{θ_i} 预测区间 [t, t'] 内教师平均速度 u_φ(x_t, t, t') = (1/(t'-t)) ∫_t^{t'} v_φ(x_s, s) ds；基于 MeanFlow 推导，利用恒等式 v_φ = u_φ - (t'-t)·d/dt u_φ，构造自一致性 stop-gradient 目标：
   L_vel^(i) = E[||u_{θ_i}(x_t, t, t') - sg(v_φ(x_t, t) + (t'-t)·d/dt u_{θ_i}(x_t, t, t'))||²]。

4. **阶段专属判别器对抗细化**：对 T2I 任务，引入 D_{ω_i} 在每个阶段内工作，生成 x_fake = x_t + (t'-t)·u_{θ_i}，以 hinge loss 形式计算 L_adv^(i) = -E[D_{ω_i}(x_fake)]，作为辅助边缘正则项（teacher-anchored velocity loss 为主目标）。

5. **推理流程**：按顺序依次执行 Phase 1（expert E_early 在 [0, 0.4] 内多步积分）→ Phase 2（expert E_late 在 [0.4, 1] 内多步积分），累计 FLOPs ≈ 1×teacher 单次前向。专家以冻结骨干 θ + 阶段专属 LoRA ψ_i 形式实例化。

6. **理论支撑**：Theorem 3.1 给出骨干平均速度与教师平均速度的误差上界（由瞬时预测误差的时间平均界定）；Theorem 3.2 证明相邻边际分布的 W₂ 距离随区间长度线性有界， motivating 分段轻量化专家。

## 实验与结果
- **数据集与骨干**：C2I 用 ImageNet 256×256；T2I 蒸馏自 SD3.5-Medium、FLUX.1-dev、Qwen-Image；训练集主要为 BLIP3o-60k/Echo-4o，额外使用 Unsplash 做后续训练。
- **评估基准**：GenEval、DPG-Bench、WISE、TIIF-Bench、Qwen-Image-Bench、Aesthetic Score、PickScore、ImageReward；效率指标包括 N_flops、Active Params、Peak VRAM。
- **C2I 结果**：PVD 在 ImageNet 256×256 上 FID=**1.48**，IS=**295.89**，超越 iMF（FID 1.72）和 FACM（FID 1.76），接近多步参考方法（LightningDiT FID 1.35）。
- **T2I 最强结果**：在 SD3.5-Medium 上 GenEval=0.6948（教师 0.6829），Aesthetic=5.46；在 FLUX.1-dev 上 GenEval=0.6532（教师 0.6675）；在 Qwen-Image 上 GenEval=**0.8846**（超越教师 0.8720）。PVD 在所有三个 T2I 骨干上均取得 distilled methods 中的最佳或次佳分数。
- **效率提升**：相对教师骨干，PVD 分别减少激活参数 **49.10%–50.89%**、峰值显存 **45.76%–48.36%**，累计计算量维持在 N_flops≈1.0。
- **Unsplash 后续训练**：Aesthetic 与 DreamSim 多样性显著提升，生成图像呈现更自然的照明、阴影与材质纹理，但 GenEval 略有下降（因数据侧重摄影美学而非精确属性绑定）。

## 相关工作脉络
1. **MeanFlow / iMF / α-Flow / FACM**：单片平均速度学习方法，共享模型覆盖全时间域，推理仍需多次调用；PVD 进一步将平均速度目标绑定到时序分区与独立专家，且推理预算等价单次前向。
2. **一致性模型（CM / iCM / sCM / Hyper-SD）**：通过自洽约束压缩步数，但不涉及相位分解；PVD 属于 trajectory-based 路线但引入阶段专业化。
3. **AD / LADD / DMD2 / SenseFlow**：分布匹配/对抗蒸馏方法，以提升感知锐度为目标，牺牲轨迹保真度；PVD 以 teacher-anchored velocity loss 为主、discriminator 为辅，兼顾轨迹 fidelity 与感知质量。
4. **Phased DMD / Hierarchical Distillation**：虽有阶段划分思想，但推理仍为 few-step 多步模式，不满足单次完整前向预算约束；PVD 在相同 budget 下显式因子化 transport。
5. **TimeStep Master / Glance / eDiff-I / Wan2.2**：MoE/时步路由架构，保留 backbone 全量计算；PVD 通过 LoRA 专家共享骨干并减半激活参数，实现真正的低资源部署。

## 局限性与未来方向
1. **专家数量与分割策略依赖经验**：当前采用固定两阶段 [0.4, 0.6] 分割，最优专家数量与分割方案未系统探索；不同数据分布可能需要自适应划分。
2. **精确指令遵循能力有限**：在密集计数（如"35 个马卡龙 5×7 排列"）和非常见空间关系任务上仍会失败，与 teacher 骨干存在同类短板。
3. **合成训练数据的质量瓶颈**：主要使用 BLIP3o/Echo-4o 合成数据，可能限制生成多样性与真实性上限（Unsplash 微调实验佐证此点）。
4. **潜在方向**：探索更多阶段划分策略（>2 阶段）、结合偏好优化/RLHF 增强指令遵循、与 diffusion NFT 等时序扰动方法结合以提升骨干初始化质量。

## 研究启发与可借鉴点
1. **"分解而非放大"的容量分配范式**：将单一长程映射因子化为有序阶段短程映射 + 专属轻量化专家，可在不增加总计算量的前提下突破单片模型容量瓶颈；该思路可迁移至视频生成、3D 生成等更长轨迹任务。
2. **平均速度的 self-consistent stop-gradient 目标**：Eq.(7) 的推导（基于 d/dt u_φ 恒等式）优雅地将平均速度回归转化为瞬时速度匹配形式，避免直接积分 Teacher trajectory，可直接复用到其他 flow-matching 蒸馏场景。
3. **相位特异性判别器设计**：不同阶段配备功能不同的判别器（结构 vs 细节），而非全局单一判别器，更贴合扩散过程的异质动力学；可推广至多模态或少步视频蒸馏。
4. **Unsplash 真实数据微调范式**：蒸馏模型可通过少量高质量真实数据二次训练显著提升感知质量，为"蒸馏 + 后训练"组合管线提供实证支持。
5. **LoRA 专家 + 共享骨干的部署模式**：rank-32/64 LoRA 专家的参数开销极小（FLUX 仅 0.16B/专家），适合边缘部署；可为其他大型扩散模型的轻量化提供即插即用方案。

## 关键术语表
**Phase-wise Velocity Distillation (PVD)**：将扩散生成轨迹分解为时序阶段，各阶段由专属半大小专家学习局部平均速度的蒸馏框架。
**Average Velocity**：专家预测的在时间区间 [t, t'] 内教师瞬时速度的积分均值，作为长程传输的代理目标。
**Semantic Drafting Phase**：生成早期的阶段（T₁=[0,0.4]），专注于建立全局低频结构与语义布局。
**Detail Refinement Phase**：生成晚期的阶段（T₂=[0.4,1]），专注于高频细节与纹理的精细化。
**Normalized Sampling FLOPs (N_flops)**：蒸馏模型累计推理 FLOPs 与教师单次前向 FLOPs 的比值，用于标准化计算开销比较。
**Stop-gradient Target**：在蒸馏损失中对 Teacher 端预测施加 sg(·) 操作，阻断梯度回传以稳定训练。
**Wasserstein-2 Distance (W₂)**：衡量两个概率分布之间最优传输成本的度量，本文用于界定相位划分的理论动机。
**LoRA (Low-Rank Adaptation)**：低秩适配模块，PVD 中用于以极少量参数量实例化相位专属专家。

## 可复现要素
- **数据集**：ImageNet-256（公开）；BLIP3o-60k、Echo-4o（论文声明来源，需确认公开状态）；Unsplash（公开）。
- **代码**：https://github.com/PolyU-VCLab/PVD（已开源）。
- **模型权重**：蒸馏后的 SD3.5-Medium、FLUX.1-dev、Qwen-Image 模型及 C2I 模型均已提供下载。
- **关键超参**：
  - 时间分割：[0.4, 0.6]，overlap=0.0（最优消融结果）
  - FLUX.1-dev LoRA rank=64；Qwen-Image LoRA rank=32
  - C2I 训练步数：140k（骨干）+ 39k（专家），batch=1024，LR=1e-4
  - T2I 骨干训练步数：35k；专家训练步数：4k-30k 不等
  - 优化器：AdamW，gradient clipping=1.0，weight decay=0
