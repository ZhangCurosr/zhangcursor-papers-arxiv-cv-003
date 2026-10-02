---
title: "LOCI-SPATIAL-LINEAR-MEMORY-FOR-STREAMING-WORLD-MODELS"
source: https://arxiv.org/pdf/2609.40222v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:44:51"
field: "视频世界模型与长程记忆"
keywords: ["world models", "video generation", "memory architecture", "streaming generation", "spatial consistency", "linear attention", "camera conditioning"]
innovations: ["混合空间记忆架构耦合投影条件化循环状态与显式KV缓存", "分块级保留机制使遗忘频率与视频时间对齐而非token顺序", "循环读出直接调制后续softmax层查询无需独立路由器"]
benchmarks: ["MIND memory benchmark", "WBench-Navi gated camera-return consistency", "held-out Unreal Engine trajectories (Set A/B)"]
---

# 论文速读：LOCI-SPATIAL-LINEAR-MEMORY-FOR-STREAMING-WORLD-MODELS

## 一句话总结
LOCI 提出了一种混合空间记忆架构，通过投影相机几何条件化的循环线性注意力记忆与显式历史 KV 缓存的结合，使视频世界模型在相机重新访问已观察区域时能够高保真地还原之前的场景内容，同时在受限内存预算下支持恒定内存的长视频流式生成。

## 研究问题与动机
- **空间持久性难题**：相机控制视频世界模型在重新访问之前观察过的区域时，需要保持场景结构和外观的一致性，但当前方法难以同时兼顾历史细节保留与高效检索。
- **KV 缓存的存储成本问题**：显式记忆方法通过全历史 softmax 注意力保留可独立访问的 KV 特征，但存储和访问成本随轨迹长度线性增长，难以支持长视频流式生成。
- **循环记忆的压缩损失**：固定大小循环状态能压缩历史，但会将多个观察合并为单一状态表示，丢失场景特定细节，导致重访时重建质量下降。
- **时空相关性不匹配**：在流式生成中，空间相关性并不遵循时间邻近性——一个近期未包含在当前上下文中的观察可能对重建当前视角至关重要，传统方法容易遗漏这类关键证据。

## 核心贡献（创新点）
- **混合空间记忆架构**：将投影几何条件化的循环集成与直接历史 KV 访问耦合，使循环上下文能够指导历史查询的同时保留观察级细节，与 SANA-WM 等方法仅通过压缩状态间接访问远处内容的本质区别在于本文保留了显式 KV 的可直接访问性。
- **分块同步记忆动力学设计**：在每个 chunk 的首个 token 处应用一次学习对角保留矩阵，使显式遗忘遵循经过的视频时间而非 token 顺序，与 KDA 语言模型的逐 token 保留机制相比避免了同帧不同步衰减的问题。
- **受控的循环-显式交互机制**：循环读出口经学习门控缩放后直接加到 token 特征中，后续历史注意力块的查询由此获得累积场景上下文，无需额外的状态-查询适配器或独立目标，与 LayerRecall 等需要单独路由器的方法形成对比。
- **有界稀疏历史访问策略**：采用全景银行（20帧）+ 最近窗口（8帧）+ 当前 chunk（5帧）的固定容量 bank，配合视野覆盖度准则选择多样性视图，使模型在恒定 23.6 GiB 内存下可持续生成 300 秒视频。

## 方法详解
- **架构布局**：基于 Wan2.2-TI2V-5B 的 30 层 transformer，奇偶交错布置 15 个混合块（含 intra-chunk softmax + KDA 循环记忆）和 15 个历史 softmax 块；所有块均保留独立的 UCPE 射线条件相机注意力分支，相机 KV 在全部 30 层存储。
- **PRoPE 投影相机编码**：对循环分支的 query、key、value 施加投影变换，世界到射线变换 $E_i$ 结合归一化内参 $\overline{K}_i$ 得到投影 $P_i = \mathrm{lift}(\overline{K}_i)E_i$，循环特征经 $\hat{q}_i = \mathcal{N}(A_i q_i)$、$\hat{k}_i = \mathcal{N}(B_i k_i)$、$\tilde{v}_i = C_i v_i$ 映射，其中 $A_i$ 应用 $P_i^\top$、$B_i/C_i$ 应用 $P_i^{-1}$ 及 patch 旋转，使循环记忆读写依赖于视角。
- **分块同步动力学**：每个 chunk 开始时，所有 token 读取前一时段的固定状态 $S_{c-1}$ 得到循环读出 $r_i = C_i^{-1}(d^{-1/2} S_{c-1}^\top \hat{q}_i)$，经门控 $\eta\sigma(g_i)$ 缩放后与 intra-chunk softmax 输出相加；token 级 delta 修正更新状态 $S^{(i)} = \overline{S}^{(i)} + \beta_i \hat{k}_i(\tilde{v}_i - \overline{S}^{(i)\top}\hat{k}_i)^\top$，但保留矩阵 $D_i$ 仅在 chunk 首 token 计算（从 chunk 均值表征学习），其余 token 使用单位矩阵。
- **稠密与有界稀疏访问**：稠密模式下历史 softmax 块 attend 整个已评估前缀；稀疏模式下 bank $\mathcal{B}_c$（≤20 帧）、最近窗口 $\mathcal{R}_c$（≤8 帧）、当前 chunk $\mathcal{U}_c$（5 帧）共享，总 retained frames ≤34；bank 更新基于视场覆盖度准则 $1 - \mathrm{cov}(f, \{0\} \cup \mathcal{B}) \geq 0.3$ 选择添加互补覆盖的更早观察，丢弃的观察不存档。
- **训练与推理调度**：采用 chunk-wise diffusion forcing，仅 conditioning latent 干净，无噪声 chunks 从零状态开始前向扫描；推理时每轮去噪迭代读取已提交状态但不修改，当前 chunk 采样结束后单独执行一次 forward（full history 在 t=0，bounded sparse 在 t=100）commit 状态。

## 实验与结果
- **数据集与基准**：MIND 记忆基准（50 segments）、自有 Unreal Engine 记录轨迹（Set A/B，共 23 clips）、WBench-Navi 门控相机返回一致性测试。训练数据含 UE 渲染场景（15,397 windows）、CARLA 城镇（965 windows）及 Sekai 真实行走视频，不使用 MIND 数据。
- **基线模型**：GIM-World、SSM、FramePack、Context-as-Memory、HY-WorldPlay（8B）、Matrix-Game 3.0（5B）、AlayaWorld（15B）、LingBot-World（28B）、CaR（5B）、SANA-WM（2.6B）、SolarWM（5B）等。
- **MIND 主结果（全段）**：LOCI（5B，未训练于 MIND）取得 MSE 0.0455 / PSNR 14.36 / SSIM 0.464 / LPIPS 0.643，优于 GIM-World（训练于 MIND）的 0.0614 / 13.40 / 0.414 / 0.630；在同等有界 KV 预算下 LOCI-bounded 达 PSNR 13.02，较 same-recipe full softmax 提升 +0.89 dB（95% CI [+0.64, +1.15]），44/50 segments 更优。
- **重访轨迹结果**：在 Set A/B 的 20 clips 重访点，LOCI（full history）PSNR 10.64 / LPIPS 0.561，较 full softmax（10.02 / 0.579）提升 +0.62 dB；LOCI（bounded sparse）PSNR 11.21 / LPIPS 0.547，显著优于 6/7 外部模型。
- **内存与速度**：full history 模式下 LOCI 将最大生成长度从 157s 延伸至 225s，同等长度下峰值显存降低约 30%（45.5 vs 63.6 GiB at 60s）；bounded sparse 模式下 300s 生成恒定于 23.6 GiB（较 full softmax 同设置低 15%），推理速度 5.4 s/v-s 与 full softmax 持平。

## 相关工作脉络
- **显式记忆世界模型（Context-as-Memory、ReWorld、WorldMem）**：通过 FOV 重叠、surfel 索引或 query-key 相似度检索历史观察，LOCI 不依赖单独检索器，而是让循环上下文自然引导 softmax 查询聚焦共视内容。
- **混合循环视频模型（Hybrid Forcing、SANA-WM、Video SSM）**：将线性注意力/状态空间与局部 softmax 结合，但远处内容仅通过压缩状态间接可用；LOCI 保留观察级 KV 直接可访问，循环读出口仅补充上下文而非替代。
- **LayerRecall（Ding et al., 2026）**：训练状态条件路由器检索历史 KV 并注入固定层；LOCI 无需独立路由或额外目标，循环读出直接参与 token 特征更新并通过自然层间路径传播。
- **相机条件化方法（UCPE、PRoPE、ViewRope、MeRoPE）**：UCPE 为每块独立相机注意力分支，PRoPE 将相机投影应用于 softmax 注意力；LOCI 将 PRoPE 扩展至 KDA 循环分支，使状态写入/读取均受视角约束。
- **ARL²（Li et al., 2026a）**：提供层布局与 read-then-commit 调度范式，但为文本到视频转换方法无相机控制；LOCI 直接以 chunk-wise diffusion forcing 从头训练混合架构，并与相机条件化深度集成。

## 局限性与未来方向
- **真实世界数据缺失**： Held-out 轨迹均在 Unreal Engine 中渲染生成，未在真实世界捕捉数据上评估，绝对保真度对所有模型均偏低（benchmark 平均 PSNR <15 dB）。
- **单一骨干网络与短训练预算**：仅测试 5B Wan 骨干，5,000 次更新可能不足以充分探索架构潜力；per-token retention 对比实验使用了探索性指标。
- **相机 KV 未减半**：30 层均保留独立相机注意力分支，循环路径仅减半 main-attention KV，全部历史存储未完全消除。
- **训练-推理状态不一致**：循环状态在训练中从无噪声 chunk 特征构建，推理时从生成 chunk commit，可能存在分布偏移。
- **速度优势依赖优化实现**：bounded sparse 模式下与 full softmax 的速度持平依赖于 CUDA graph 捕获与 projective transform 复用等工程优化，通用场景下的开销需进一步验证。

## 研究启发与可借鉴点
- **循环-显式记忆的互补性设计**：将固定大小循环状态作为下游查询的上下文输入而非替代显式 KV，为混合记忆架构提供了简洁有效的交互范式，可迁移至语言-视觉多模态长上下文任务。
- **分块级保留替代 token 级保留**：在视频生成中将遗忘频率与时间跨度对齐（per-chunk retention）而非空间 token 顺序，避免同帧内不同步衰减，适用于任意基于 chunk 的序列建模场景。
- **投影几何条件化循环记忆**：将 PRoPE 类相机编码引入 KDA 状态更新，使循环记忆具备视角可解码性（yaw $R^2=0.946$），为 3D 一致性的隐式建模提供了轻量方案。
- **有界 bank 的多样性选择准则**：基于 FOV 覆盖度互补性而非时间邻近性选择保留帧，可在恒定存储下最大化历史信息的空间覆盖，适用于需要长期记忆的导航/探索任务。
- **本地一致性评估指标**：使用 DINOv2 worst 5% patches 距离衡量重访局部错误，比全局 LPIPS 更能揭示模型在短间隔重访时的细节保持能力，可作为后续工作标准评估手段。

## 关键术语表
**LOCI**：论文提出的混合空间记忆架构，结合投影条件化循环线性注意力与显式历史 KV 缓存，用于视频世界模型的重访一致性生成。
**Kimi Delta Attention (KDA)**：一种固定大小状态的线性注意力变体，通过 channel-wise 保留与 token-level delta 修正维护 $S \in \mathbb{R}^{d_k \times d_v}$ 状态矩阵。
**PRoPE (Projected Rotary Position Embedding)**：将相机投影几何引入位置编码，使 attention 的 query-key 交互依赖于相对射影变换 $P_i P_j^{-1}$，实现视角不变性。
**UCPE (Unified Camera Positional Encoding)**：为每层独立维护的射线条件相机注意力分支，将相机 pose 直接融入 attention 计算。
**Chunk-wise Diffusion Forcing**：仅 conditioning latent 干净、后续 chunks 独立加噪的训练策略，使模型学习从部分噪声状态恢复完整序列。
**Bounded Sparse Access**：将历史访问限制为 conditioning frame + panorama bank（≤20）+ recent window（≤8）+ current chunk（5）的固定容量集合。
**Co-visible Attention Enrichment**：衡量模型在历史 attention 中对几何共视 token 的集中程度，定义为共视注意力占比除以共视 token 占比的比值。
**Memory Commit**：推理时当前 chunk 生成完成后执行的额外 forward pass，将最终 latent 特征写入循环状态供后续 chunk 使用。

## 可复现要素
- **数据集**：MIND benchmark（公开）、Unreal Engine 场景轨迹（自行渲染，约 92.8h）、CARLA 城镇（自行渲染，5.4h）、Sekai-Walking 真实视频；训练数据不包含 MIND。
- **代码开源**：项目页面 https://xiaji2021.github.io/LOCI/（论文声明）；权重开源情况论文未明确提及。
- **关键超参**：骨干 Wan2.2-TI2V-5B（30 层，24 heads，head dim 128）；训练 5,000 updates，batch size 64，learning rate $10^{-5}$（backbone）/$5\times10^{-5}$（camera branch）/$10^{-4}$（recurrent branch）；chunk size 5 latent frames；recurrent state $128\times128$；bounded bank 20 + recent 8 + current 5；denoising steps 50，CFG 1.0。
