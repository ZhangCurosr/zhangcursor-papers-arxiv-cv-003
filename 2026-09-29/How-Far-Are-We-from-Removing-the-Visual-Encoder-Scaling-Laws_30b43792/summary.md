---
title: "How-Far-Are-We-from-Removing-the-Visual-Encoder-Scaling-Laws"
source: https://arxiv.org/pdf/2609.35457v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:25:50"
field: "多模态大模型缩放定律"
keywords: ["encoder-free MLLM", "scaling laws", "multimodal pretraining", "mixture-of-experts", "visual encoder", "compute-optimal allocation", "loss-compute frontier"]
innovations: ["系统比较encoder-free与encoder-based MLLM的scaling laws并预测效率crossover点", "揭示decoder通过双向注意力、浅层处理和专家路由集中接管视觉编码的机制", "按主题细分的crossover分析，发现感知密集型任务追赶更晚"]
benchmarks: ["CV-Bench", "POPE", "MME", "ChartQA", "DocVQA", "AI2D", "TextVQA", "RealWorldQA", "MMStar", "MMBench-EN", "ScienceQA-IMG"]
---

# 论文速读：How-Far-Are-We-from-Removing-the-Visual-Encoder-Scaling-Laws

## 一句话总结
本文通过受控的 Scaling Law 实验，系统比较了无视觉编码器（encoder-free）与有编码器（encoder-based）多模态大语言模型的缩放行为，发现移除视觉编码器后模型在文本目标上的缩放规律几乎不变，但在多模态目标上需要更大模型规模，且预测在约 $10^{22}$ FLOPs 处会出现效率反超。

## 研究问题与动机
- **核心问题**：当前大多数多模态大语言模型（MLLMs）依赖预训练视觉编码器（如 SigLIP ViT）提供强视觉先验，而 encoder-free 架构直接喂入原始像素，其 scaling 行为尚未被系统刻画。
- **现有方法不足**：先前 encoder-free 工作（Fuyu, EVE, SOLO, SAIL 等）仅验证了可行性，未进行控制变量下的缩放定律对比，无法判断未来是否值得采用更简单的 unified 架构。
- **实际需求**：若 encoder-free 在大预算下更高效，可降低架构复杂度，避免编码器与解码器的联合缩放优化难题。

## 核心贡献（创新点）
1. **首次系统比较 encoder-free 与 encoder-based MLLM 的 scaling laws**：在共享稀疏 MoE 解码器阶梯、数据混合和优化设置的严格控制下，分别拟合文本与多模态目标的损失–计算标度律。
2. **发现计算最优分配的架构差异**：移除视觉编码器后，多模态目标的模型分配指数从 $a=0.464$ 增至 $a=0.570$，表明 encoder-free 训练更倾向于更大模型规模。
3. **预测 $10^{22}$ FLOPs 级别出现效率 crossover**：encoder-free 模型在多模态目标上损失下降更快，外推预测在约 $6.1 \times 10^{21}$ FLOPs（计算最优）和 $1.2 \times 10^{22}$ FLOPs（5× overtraining）处实现效率反超。
4. **揭示解码器接管视觉编码的机制**：通过内部探针发现三阶段适配——视觉 token 间的双向注意力增强、浅层解码器层执行隐式视觉编码、MoE expert 路由对视觉 token 更加集中。
5. **按主题细粒度的 scaling 分析**：发现 crossover 时间因主题而异——依赖语言的主题（STEM）较早追上，感知密集型主题（Caption、GUI、OCR）则明显滞后。

## 方法详解
- **受控模型阶梯**：构建 11 档稀疏 MoE 解码器（总参数 1.1B–44B，激活参数 71M–2.4B），两种架构共享相同数据混合、优化器和视觉 token 粒度。
- **Encoder-based 前端**：预训练 SigLIP 2 ViT（27层、宽度 1152）+ ConvPool adapter + projector，ViT 参数量固定约 400M。
- **Encoder-free 前端**：原始图像 patch 经 2×2 合并（每个 token 覆盖 32×32 像素）→ LayerNorm → Linear → LayerNorm → 可学习的 2D 位置编码 → RMSNorm → Linear 投影到解码器宽度。
- **注意力模式**：默认在视觉 token 间使用双向注意力，同时额外评估因果注意力变体以验证归因。
- **Scaling law 拟合流程**：
  - 在每个计算预算 $C$ 下，通过 IsoFLOP profile（固定 $C$，变化 $M$，拟合 $D=C/M$）找到最优 $M_{\text{opt}}(C)$。
  - 拟合分配律：$M_{\text{opt}} \propto C^a$，$D_{\text{opt}} \propto C^b$，$a+b=1$。
  - 拟合损失–计算前沿：$\mathcal{L}^*(C) = E + K C^{-\gamma}$，其中 $E$ 为共享的熵下界。
- **Overtraining 建模**：在计算最优预算 $C_{\text{base}}$ 上保持模型规模固定，将 token 数乘以 $k$（$k \in \{2,3,4,5\}$），推导得 $\mathcal{L}(C_{\text{base}}, k) = E + g(k) K C_{\text{base}}^{-\gamma}$，仅需拟合 $g(k)$。
- **效率增益度量**：
  - $\text{EG}^C = C_{\text{ref}}(\lambda) / C_{\text{tar}}(\lambda)$（训练计算效率）
  - $\text{EG}^M = M_{\text{ref}}(\lambda) / M_{\text{tar}}(\lambda)$（模型效率）
- **内部探针**：
  - 各解码器层视觉 token 表示与输入的余弦相似度，追踪表示演化。
  - 双向 vs 因果注意力对比。
  - MoE expert 负载不均衡（MaxVio）按视觉/文本 token 分别统计。

## 实验与结果
- **数据集**：训练集为 1:1 文本/多模态混合，多模态涵盖 captioning、charts、grounding、GUI、OCR、STEM 等主题；验证集与训练集不重叠，按主题分组评估。
- **基线架构**：Encoder-based 使用 SigLIP 2 ViT；Encoder-free 使用 patch projection + 双向注意力。
- **文本目标结果**：
  - 两者 loss–compute 曲线几乎重合，分配指数 $a \approx 0.427$（free）vs $0.422$（based），$\gamma \approx 0.097$。
  - $\text{EG}^C \approx 0.98$，$\text{EG}^M \approx 0.99$，差异可忽略。
- **多模态目标结果**：
  - Encoder-free 分配指数 $a = 0.570$ vs encoder-based $a = 0.464$，bootstrap 80% 区间不重叠。
  - Loss–compute 指数 $\gamma_{\text{free}} = 0.378$ vs $\gamma_{\text{based}} = 0.300$，encoder-free 下降更快。
  - Compute-optimal crossover 预测：点估计 $6.1 \times 10^{21}$ FLOPs，80% 区间 $[4.2, 10.0] \times 10^{21}$。
  - 5× overtraining crossover 点估计 $1.2 \times 10^{22}$ FLOPs，80% 区间 $[8.4, 20.0] \times 10^{21}$。
  - 近期旗舰模型（如 Kimi K2.5）预训练计算约 $10^{25}$ FLOPs，crossover 在其下方三个数量级。
- **按主题结果**：STEM → Charts → GUI/OCR/Caption 依次追上，感知依赖越强 crossover 越晚。
- **Downstream 验证**：在 11 个多模态基准（CV-Bench、POPE、MME、ChartQA、DocVQA、AI2D、TextVQA、RealWorldQA、MMStar、MMBench、ScienceQA）上，3-shot 平均分数随计算量增加差距缩小，与 loss 趋势一致。
- **最强结果**：最大规模 encoder-free 模型（33B-A2.2B）在 MMBench-EN 上得分 69.1 vs encoder-based 67.1，在某些主题上已反超。

## 相关工作脉络
1. **Encoder-free MLLM 先驱**：Fuyu [3]、EVE [14]、SOLO [9] 首次证明 encoder-free 可行性；SAIL [30] 系统研究单 Transformer 内的视觉–语言学习；Gemma 4 Unified [23]、Inkling [54]、TUNA-2 [39] 进一步探索原生多模态输入。本文定位为补充性 scaling 对比而非架构设计。
2. **Scaling Laws 方法论**：Kaplan et al. [27]、Hoffmann et al. [25] 建立语言模型缩放定律基础；Chinchilla 优化原则；本文将其扩展至多模态 pretraining，并提出计算最优与 overtraining 两种实用场景的对比框架。
3. **Native Multimodal Scaling**：Tong et al. [57]、Shukor et al. [49]、Wu et al. [63]、Tian et al. [55] 分别研究了 native 多模态预训练的数据缩放、耦合 scaling、数据混合影响；本文与之不同在于使用**匹配解码器阶梯 + 固定编码器**的控制实验设计。
4. **视觉编码器作用分析**：Fan et al. [18] 分析 MLLM 中视觉 token 的稀疏性与冗余性；本文从 decoder 内部探针角度揭示其如何"接管"编码器角色。
5. **跨模态信息流**：Lei et al. [30] 研究单 Transformer 内的信息流；本文通过注意力质量、表征演化、expert 路由三维度量化解码器承担的视觉编码功能。
6. **MoE 负载均衡**：MaxVio [60] 用于衡量 expert 负载不均衡，本文将其应用于按模态细分的 expert 路由分析。

## 局限性与未来方向
- **ViT 规模固定**：实验中将 ViT 固定为 ~400M 参数量，未探索 encoder-based 的联合缩放（encoder + decoder 同时增大），可能低估 encoder-based 在大预算下的潜力。
- **外推不确定性**：crossover 点基于外推预测，实际训练中可能出现拟合失效（如熵下界 $E$ 并非严格共享）。
- **单注意力模式**：主要使用双向注意力，因果注意力变体仅作为对照，未系统探索最优注意力设计。
- **仅限解码器架构**：未探索专为 encoder-free 设计的新型解码器结构，当前结论基于"直接继承语言架构"的 baseline。
- **训练资源限制**：最大预算仅到 $2 \times 10^{21}$ FLOPs，crossover 需外推约一个数量级。

## 研究启发与可借鉴点
1. **受控对比实验设计范式**：固定编码器规模 + 变化解码器规模，隔离 decoder scaling 效应，避免联合缩放混淆变量，此范式可迁移至其他架构消融研究。
2. **多探针联合诊断方法**：同时观测注意力质量（层间视觉 token 注意力）、表征演化（余弦相似度轨迹）、专家路由（MaxVio），三维度交叉验证机制假说，值得借鉴。
3. **Overtraining 的标度律建模**：将 overtraining 建模为 loss–compute 律的 prefactor 变化（$g(k)$），仅需少量额外实验即可预测部署场景效率，方法论可直接复用。
4. **按主题细分的 scaling 分析**：发现不同主题的 crossover 差异，提示未来工作可按任务类型定制架构策略，而非追求统一最优解。
5. **视觉–语言分离视角**：文本目标上两架构几乎等价，差异完全来自视觉侧，为理解多模态模型各组件的贡献提供了清晰的分母控制。

## 关键术语表
- **Encoder-free MLLM**：直接喂入原始图像 patch 到解码器的多模态大语言模型，无需独立视觉编码器。
- **Compute-optimal allocation**：在给定总计算预算下，最优分配给模型规模（FLOPs/token）和训练 token 数的比例关系。
- **IsoFLOP profile**：固定总计算量，变化每 token FLOPs，拟合验证损失二次函数以确定该预算下的最优模型规模。
- **Loss–compute frontier**：描述最小化验证损失所需训练计算量的曲线，形式为 $\mathcal{L}^*(C) = E + K C^{-\gamma}$。
- **Efficiency gain ($\text{EG}^C$, $\text{EG}^M$)**：达到相同损失时，目标系统相对于参考系统的计算效率或模型效率比值。
- **Vision-specific adaptation**：decoder 为补偿缺失的视觉编码器而发展的三阶段适配机制（双向注意力、浅层处理、专家路由集中）。
- **MaxVio**：衡量 MoE expert 负载不均衡程度的指标，定义为最重载 expert 的相对超额负载。
- **Crossover point**：两条 loss–compute 曲线相交处的计算预算，即两种架构效率持平的点。

## 可复现要素
- **数据集**：训练数据为内部 1:1 文本/多模态混合（具体数据集名称论文未公开），验证集包含标准 benchmark（CV-Bench、POPE、MME 等）和自建主题划分。
- **代码/权重**：论文未声明开源代码或模型权重，主要结果为内部实验。
- **关键超参**：序列长度 4096、Muon optimizer、2000 warmup steps、MoE 256 experts 激活 top-8、共享 expert 占比 1/256。
- **模型范围**：11 档模型，总参数 1.1B–44B，激活参数 71M–2.4B。
- **预算范围**：文本目标 $2 \times 10^{19}$–$2 \times 10^{20}$ FLOPs；多模态目标 $1 \times 10^{20}$–$1 \times 10^{21}$ FLOPs。
- **ViT 规模**：SigLIP 2，27 层，宽度 1152，patch size 16，约 400M 参数。
