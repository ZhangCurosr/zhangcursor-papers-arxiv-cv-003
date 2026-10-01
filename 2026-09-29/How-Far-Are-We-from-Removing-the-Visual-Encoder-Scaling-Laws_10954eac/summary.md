---
title: "How-Far-Are-We-from-Removing-the-Visual-Encoder-Scaling-Laws"
source: https://arxiv.org/pdf/2609.35457v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 09:52:33"
field: "多模态大模型缩放规律"
keywords: ["encoder-free MLLM", "scaling laws", "multimodal pretraining", "visual encoder", "compute-optimal allocation", "vision-specific adaptation"]
innovations: ["首次系统比较encoder-free与encoder-based MLLMs的缩放律，量化效率差距", "发现移除视觉编码器使多模态目标计算最优分配偏向更大模型", "预测encoder-free模型在多模态目标上将于约10^22 FLOPs计算量追上encoder-based模型"]
benchmarks: ["SigLIP 2 ViT", "Kimi K2.5", "CV-Bench", "POPE", "MME", "ChartQA", "DocVQA", "AI2D", "TextVQA", "RealWorldQA", "MMStar", "MMBench-EN", "ScienceQA-IMG"]
---

# 论文速读：How-Far-Are-We-from-Removing-the-Visual-Encoder-Scaling-Laws

## 一句话总结
该论文系统比较了移除预训练视觉编码器前后多模态大语言模型（MLLM）的缩放规律，发现encoder‑free模型在多模态目标上需要更多训练计算，但其损失随计算下降更快，外推预测约10^22 FLOPs计算量时可追上encoder‑based模型，且decoder会通过视觉特定适应（双向注意力、浅层重写、专家集中）接管视觉编码功能。

## 研究问题与动机
- 现代MLLM普遍依赖预训练视觉编码器提供强视觉先验，而encoder‑free架构直接学习原始像素表示，虽更简单统一，但其缩放行为尚未被系统表征。
- 现有encoder‑free工作仅验证初步可行性，缺乏对计算最优分配、效率增益及内部适应机制的全面分析。
- 需要量化移除视觉编码器如何影响模型容量与训练数据的计算最优配比，并理解decoder如何补偿缺失的视觉编码。

## 核心贡献（创新点）
1. **首次系统比较encoder‑free与encoder‑based MLLMs的缩放律**：通过受控实验分别拟合文本与多模态目标的缩放律，量化两架构效率差距随规模的演变。
2. **揭示计算最优分配的架构差异**：发现移除视觉编码器使多模态目标的模型分配指数从0.464增至0.570，显著偏向更大模型规模；文本目标则基本不变。
3. **预测encoder‑free模型将在实际预训练预算内追上**：多模态目标上encoder‑free损失下降更快，外推交叉点约6.1×10^21 FLOPs（80%区间[4.2×10^21, 1.0×10^22]），远低于近期旗舰模型（如Kimi K2.5）的10^25 FLOPs。
4. **探伤decoder接管视觉编码的机制**：发现视觉特定适应行为——视觉token间双向注意力增强、浅层decoder重写视觉表示、MoE专家对视觉token的路由更集中，为decoder架构设计提供依据。

## 方法详解
- **缩放律估计**：采用IsoFLOP profile方法，在每个计算预算C下变化模型FLOPs per token（M），固定D=C/M，拟合验证损失随log M的二次函数，获得最优M_opt与D_opt。跨预算拟合分配律M_opt∝C^a，D_opt∝C^b（a+b=1），进而得损失‑计算前沿L*(C)=E+KC^{-γ}。
- **效率增益度量**：定义计算效率增益EG^C(λ)=C_ref(λ)/C_tar(λ)与模型效率增益EG^M(λ)=M_ref(λ)/M_tar(λ)，值大于1表示目标系统更高效。
- **模型梯架**：使用11个稀疏MoE语言模型（总参数1.1B–44B，激活参数71M–2.4B），共享相同数据混合、优化设置与视觉token粒度。encoder‑based基线采用预训练SigLIP 2 ViT（27层，宽度1152）加ConvPool适配器；encoder‑free模型直接将像素块投影入decoder，视觉token间采用双向注意力。
- **计算核算**：以decoder FLOPs per token为主，分解为基础矩阵乘法项与注意力项；固定专家激活比8/256，使M大致正比于激活参数。
- **过训练分析**：基于可分离损失模型推导过训练因子k对损失‑计算律的影响，仅重拟合乘数g(k)而非完整缩放律，降低实验成本。
- **探伤方法**：通过注意力质量、层间余弦相似度、专家负载不均衡度（MaxVio）等指标，分析decoder内部如何适应视觉编码任务。

## 实验与结果
- **数据集与基线**：训练数据为1:1混合的文本与多模态数据，多模态涵盖captioning、charts、grounding、GUI、OCR、STEM等主题；验证集与训练集不重叠。encoder‑based基线使用SigLIP 2 ViT（~400M参数）。
- **主要结果**：
  - **计算最优分配**：文本目标两架构模型分配指数相近（encoder‑free a=0.427，encoder‑based a=0.422）；多模态目标encoder‑free a=0.570，encoder‑based a=0.464，证实移除编码器需更大decoder容量。
  - **效率增益**：文本目标两架构几乎重叠（EG^C≈0.98）；多模态目标encoder‑free需更多计算，但EG^C随规模增大而提升，交叉点估计为6.1×10^21 FLOPs（80%区间[4.2×10^21, 1.0×10^22]）。
  - **过训练影响**：5×过训练使交叉点移至1.2×10^22 FLOPs，且encoder‑free因偏好更大模型，过训练收益较低。
  - **主题差异**：STEM等语言主导主题交叉较早，Caption、GUI、OCR等感知密集型主题交叉较晚。
  - **探伤发现**：encoder‑free模型在浅层decoder即发生视觉表示重写（余弦相似度早期下降），视觉token间双向注意力重要性随计算增加，专家对视觉token的路由更集中（MaxVio更高）。
- **最强结果**：encoder‑free模型在多模态目标上损失‑计算指数γ=0.378，高于encoder‑based的0.300，表明其损失下降更快；跨主题分析显示在STEM等主题上encoder‑free已接近parity。

## 相关工作脉络
1. **Encoder‑Free MLLMs**：早期工作如Fuyu、EVE、SOLO证明可行性；SAIL系统研究缩放与表示学习；Gemma 4 Unified、Inkling等后续探索原生多模态输入。本文定位：首次系统比较两类架构的缩放律，填补理论空白。
2. **缩放律研究**：Chinchilla提出计算最优分配；后续扩展至数据缩放、超参优化；多模态领域有针对视觉编码器和语言模型耦合缩放的探索。本文贡献：比较encoder‑free与encoder‑based在共享MoE梯架下的缩放行为。
3. **视觉编码器作用**：SigLIP、CLIP等提供强视觉先验；有研究探讨视觉token稀疏性。本文发现：encoder‑free模型通过decoder内部适应部分接管视觉编码功能。
4. **高效训练策略**：过训练分析源于MAI‑Thinking‑1；本文推广至多模态场景，量化不同架构的过训练增益差异。
5. **解释性分析**：通过注意力质量、层表示演化、专家路由等探伤，揭示架构选择对内部计算过程的影响。

## 局限性与未来方向
- **局限性**：
  1. 实验固定视觉编码器大小（400M），未联合缩放视觉前端与decoder，可能高估encoder‑free优势。
  2. 缩放律外推存在不确定性，尤其对于感知密集型主题（如OCR、GUI），交叉点可能延迟至更高计算量。
  3. 仅评估下一词预测损失，下游任务性能与缩放律的关联需进一步验证。
  4. 未探索decoder架构设计（如专为视觉表示学习优化）的潜力。
- **未来方向**：
  1. 研究decoder架构如何适配原生视觉输入，例如引入视觉特定注意力模式或层设计。
  2. 联合缩放视觉前端与decoder，分析更全面的计算分配。
  3. 探索encoder‑free模型在长视频、3D等多模态任务上的缩放行为。
  4. 结合数据混合优化，进一步提升encoder‑free效率。

## 研究启发与可借鉴点
1. **控制变量缩放实验设计**：固定视觉token粒度、数据混合、优化设置，仅改变视觉前端架构，可清晰隔离架构对缩放规律的影响。
2. **效率增益的多维度量**：同时报告EG^C和EG^M，能全面反映系统在计算和模型规模上的效率差异。
3. **主题细粒度分析**：按多模态主题分别评估缩放律，可识别架构优势的具体适用场景，指导模型选择。
4. **内部探伤方法**：通过注意力质量、层表示演化、专家负载等指标，揭示模型内部适应机制，为架构设计提供依据。
5. **过训练建模简化**：复用计算最优拟合参数，仅重拟合过训练乘数，降低实验成本，适用于快速迭代研究。

## 关键术语表
- **Encoder‑free MLLM**：移除预训练视觉编码器，直接学习原始像素表示的多模态大语言模型。
- **Scaling law**：描述模型性能（损失）随计算量、模型规模、数据量变化的数学规律。
- **IsoFLOP profile**：在固定计算预算下，通过变化模型FLOPs per token拟合验证损失曲线，以确定计算最优分配。
- **Compute‑optimal allocation**：给定计算预算，使损失最小化的模型规模与数据量的最优配比。
- **Loss–compute exponent (γ)**：缩放律L*(C)=E+KC^{-γ}中衡量损失随计算下降快慢的指数。
- **Vision‑specific adaptation**：encoder‑free模型中decoder为补偿缺失视觉编码器而发生的内部调整（如双向注意力、浅层重写）。
- **Efficiency gain (EG^C, EG^M)**：比较两系统在相等损失下所需计算量或FLOPs per token的比值。
- **MaxVio**：衡量MoE专家负载不均衡度的指标，计算最重专家负载相对于理想均匀负载的超额比例。

## 可复现要素
- **数据集**：训练数据混合1:1文本与多模态数据，多模态涵盖captioning、charts、grounding、GUI、OCR、STEM等；验证集与训练集不重叠。论文未公开具体数据集名称，但提及内部数据。
- **代码/权重**：论文未开源代码与模型权重，仅公开论文版本。
- **关键超参**：序列长度4096，Muon优化器，2000 warmup steps；MoE顶层8/256专家激活，共享专家1个；视觉token粒度1 token per 32×32像素区域。
- **环境**：基于内部基础设施，具体硬件配置未披露。
