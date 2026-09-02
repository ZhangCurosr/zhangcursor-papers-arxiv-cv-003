---
title: "S-sup-2-sup-Prune-Spatially-Structured-Visual-Token-Pruning"
source: https://arxiv.org/pdf/2609.01224v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:21:39"
field: "多模态大模型高效推理"
keywords: ["Visual Token Pruning", "Multimodal LLM", "Spatial Allocation", "Training-free", "Laplacian Structure", "Early Representation Change"]
innovations: ["提出解耦的空间分配+局部选择两阶段免训练剪枝框架", "用Laplacian方差自适应分配区域Token预算而非均匀分布", "揭示不同打分标准产生稳定空间偏差并验证Uniform Grid价值"]
benchmarks: ["GQA", "MMMU", "POPE", "MME", "MMB EN", "MMB CN", "VQAv2", "VizWiz", "SQAMG", "MMVet"]
---

# 论文速读：S²Prune: Spatially Structured Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
本文提出了一种免训练的视觉Token剪枝方法S²Prune，通过"先按图像局部结构分配区域Token预算、再用Early Representation Change(ERC)在单元格内选择代表Token"的两阶段策略，在保持广泛空间覆盖的同时自适应调整局部采样密度，在Qwen2.5-VL-7B等主流MLLM上显著优于现有方法。

## 研究问题与动机
1. **现有剪枝方法存在稳定空间偏差**：重要性或冗余度驱动的Token打分标准会产生跨输入的固定空间偏好（如FastV、FEATHER偏好图像边界），无法保证Broad spatial coverage。
2. **均匀网格采样的价值被低估**：简单的Uniform Grid采样因能维持更广泛的空間覆盖，在激进剪枝下仍能保持竞争力，提示空间组织本身的重要性。
3. **固定密度假设的局限性**：Uniform Grid对所有区域分配相同密度，导致平滑区域过度表征而结构丰富区域（文本、边界、细结构）表征不足，尤其在低Token预算下性能下降明显。
4. **需要解耦"在哪里剪"与"剪哪些"**：现有方法未明确区分区域预算分配（spatial scaffold）与局部Token选择（local selection）两个层次，难以兼顾全局覆盖与局部细节。

## 核心贡献（创新点）
1. **从空间分配视角重新审视视觉Token剪枝**：通过系统分析揭示了不同打分标准产生的稳定空间偏差，并验证了Broad coverage对激进剪枝的鲁棒性，提出了"保持广泛空间覆盖+自适应局部密度"的设计原则，区别于现有仅关注Token重要性的工作。
2. **提出两阶段免训练剪枝框架S²Prune**：第一阶段用Laplacian方差测量区域结构复杂度并分配Token预算，第二阶段用First decoder block的Early Representation Change(ERC)在每个cell内选择代表Token，将"预算分配"与"局部选择"解耦，与GridPrune等仅做区域划分的方法形成本质差异。
3. **在十项基准和两种架构上验证了广泛适用性**：在Qwen2.5-VL-7B和LLaVA-OneVision-7B上均取得最优结果，32 Token时保留79.3%全模型性能且超越最强基线0.9点，证明结构自适应的空间分配在低预算下优势显著。

## 方法详解
**整体流程**：S²Prune分为Adaptive Spatial Scaffold（自适应空间支架）和Local Selection with Early Decoder Responses（基于早期解码器响应的局部选择）两个阶段，剪枝发生在Decoder Layer 0之后、Layer 1之前。

**阶段一：自适应空间支架（Adaptive Spatial Scaffold）**
1. 将视觉Token网格划分为G个不相交粗区域$\{\mathcal{R}_g\}_{g=1}^G$（$G \leq B$），例如$B=64$时用$5\times5$网格得25个区域。
2. 计算结构复杂度：将图像转灰度后计算四邻域Laplacian响应图$L$（外边界用replicate padding），对每个区域$\mathcal{R}_g$映射到图像域支持$\Omega_g$并裁剪响应图，原始结构分数为$c_g^{\text{raw}} = \text{Var}(L|_{\Omega_g})$。
3. 归一化：对单张图像内各区域$c_g^{\text{raw}}$做min-max归一化得到$c_g$，公式为$c_g = \frac{c_g^{\text{raw}} - \min_j c_j^{\text{raw}}}{\max(\max_j c_j^{\text{raw}} - \min_j c_j^{\text{raw}}, \epsilon_{\text{norm}})}$。
4. 预算分配：每个区域先分配1个Token保证空间覆盖，剩余$B-G$个Token按$c_g$比例分配，约束$1 \leq B_g \leq |\mathcal{R}_g|$且$\sum B_g = B$，采用iterative largest-remainder procedure处理饱和区域。
5. 递归划分单元格：每个区域$\mathcal{R}_g$被划分为$B_g$个近似等面积的非重叠矩形cell $\{\mathcal{C}_{g,k}\}_{k=1}^{B_g}$，优先沿较长边对折分割。

**阶段二：局部代表Token选择（Local Selection with ERC）**
1. 计算Early Representation Change(ERC)：视觉Token经过Decoder Layer 0后，对每个Token $i$计算$ r_i = \|h_i^1 - h_i^0\|_2$，其中$h_i^0, h_i^1$分别为Layer 0前后hidden representation。
2. 单元内最大ERC选择：在每个cell $\mathcal{C}_{g,k}$内保留ERC最大的Token，即$i_{g,k}^* = \arg\max_{i \in \mathcal{C}_{g,k}} r_i$。
3. 保持顺序与位置信息：选出的Token按原始decoder-visible序列索引排序，保留原有M-RoPE位置ID，物理移除未选中Token（非mask），从Layer 1继续推理。

**设计哲学**：Laplacian variation决定"预算花在哪里"（where），ERC只决定"在每个allocated cell里选谁"（which），二者解耦避免全局打分带来的空间偏差。

## 实验与结果
**实验设置**：在Qwen2.5-VL-7B-Instruct（主实验）和LLaVA-OneVision-7B（跨架构验证）上测试，使用10项多模态基准（GQA、$\text{SQ A}^{\text{IMG}}$、$\text{VQA}^{\text{Text}}$、VizWiz、MMMU、POPE、MME、$\text{MMB}^{\text{EN}}$、$\text{MMB}^{\text{CN}}$、MMVet），Token预算$B \in \{128, 64, 32\}$对应原始576个Token的77.8%、88.9%、94.4%缩减率。

**主要结果（Qwen2.5-VL-7B）**：
- **$B=128$**：S²Prune取得最高Acc. 64.0，相对性能92.1%，超越最强基线HoloV 0.1点；在GQA和$\text{MMB}^{\text{EN}}$上最优。
- **$B=64$**：Acc. 60.6，相对性能87.0%，超越VisPruner/GridPrune 0.3点；$\text{MMB}^{\text{CN}}$第一。
- **$B=32$**：Acc. 55.9，相对性能79.3%，领先最强基线0.9点，在6/10基准上第一（含VizWiz、$\text{MMB}^{\text{CN}}$）。

**跨架构验证（LLaVA-OneVision-7B）**：在$B \in \{160, 80, 40\}$下均取得最高Acc.和Rel.，证明方法迁移性。

**效率分析**（$B=64$，POPE任务，RTX 5090）：
- FLOPs从8.181T降至1.596T（5.1×减少）
- Prefill latency从56.97ms降至28.66ms（~2.0×加速）
- KV cache从31.50MB降至4.50MB（7.0×减少）
- 在保持最高Acc.（62.3 vs FastV 57.7、VisPruner 60.4）的同时实现显著效率提升。

**消融实验**：
- **结构先验**：Laplacian在MMMU/VQA$\textsuperscript{Text}$/VizWiz上均最优（46.3/55.2/31.0），优于FFT、Effective Rank、Random、Uniform。
- **局部选择器**：ERC在三项基准上均第一（84.6/81.0/55.2），优于Random、Hidden Norm、Query Attn.（All-Q/Last Token）。

## 相关工作脉络
1. **FastV (ECCV'24)**：基于attention score剪枝，偏好图像边界（~58% edge tokens），本文揭示其空间偏差并指出Uniform Grid可超越部分精心设计的打分方法。
2. **FEATHER (ICCV'25)**：用uniform sampling维持空间覆盖，本文继承其"coverage优先"理念但进一步引入结构自适应密度，避免均匀分配的结构浪费。
3. **VisionZip (CVPR'25)**：结合visual attention与similarity merging，本文发现其edge-center分布较平衡但仍受score-driven spatial bias影响。
4. **GridPrune (arXiv'25)**：query-conditioned regional budget allocation + local selection，本文与其关键区别在于：GridPrune用语义相关性指导预算分配，S²Prune用图像内在结构复杂度（Laplacian variance）分配容量。
5. **VisPruner (ICCV'25)**：visual attention + diversity，本文证明在激进剪枝（32 tokens）下结构自适应方法显著优于多样性导向方法。
6. **DART (EMNLP'25)**：去除重复Token，本文与之互补——DART解决"冗余"问题，S²Prune解决"空间覆盖+结构适配"问题。

## 局限性与未来方向
1. **计算Laplacian的额外开销**：需对全图做一次灰度转换和Laplacian滤波，虽轻量但与纯attention-based方法相比仍有固定计算成本。
2. **粗网格划分依赖经验设定**：$K\times K$网格大小（如$B=64$用$5\times5$）依赖实验调参，未提供自动化选择策略。
3. **仅验证两类架构**：仅在Qwen2.5-VL和LLaVA-OneVision上测试，对InternVL等其他MLLM的适用性待验证。
4. **未考虑query语义**：预算分配完全基于图像结构，未融合问题条件；未来可探索结构先验与语义相关性的联合建模。
5. **ERC仅限Layer 0**：仅用第一次解码器响应，更深层的representation change可能包含更丰富的判别信息，值得探索多层集成。

## 研究启发与可借鉴点
1. **"解耦分配与选择"的设计范式**：将"where to spend budget"与"which token to keep"分离，可迁移至其他稀疏表示场景（如视觉 grounding、文档理解中的region selection）。
2. **用图像低层结构信号辅助高层任务**：Laplacian variance作为结构复杂度的代理指标，低成本且免训练，可推广至其他需要空间感知的视觉-语言任务（如VQA中的区域聚焦、OCR增强）。
3. **重新评估简单baseline的价值**：本文系统证明Uniform Grid在激进剪枝下的竞争力，启发团队在提出新方法前充分验证空间覆盖类baseline，避免"reinventing the wheel"。
4. **稳定性分析优于单一指标优化**：通过跨图像、跨budget的空间偏差相关性分析（Pearson/Spearman>0.94）揭示方法本质缺陷，这种诊断性实验设计值得借鉴。
5. **位置信息保留的工程细节**：剪枝后保留原始M-RoPE position ID而非重编号，避免positional encoding错位，对开发免训练加速插件有直接参考价值。

## 关键术语表
**Visual Token Pruning**：通过移除冗余视觉Token减少MLLM推理开销的技术，核心挑战是在压缩率与性能之间取得平衡。
**Early Representation Change (ERC)**：视觉Token经过Decoder Layer 0后的hidden state变化量（$\|h_i^1 - h_i^0\|_2$），用于单元内局部Token选择。
**Laplacian Variation**：图像灰度Laplacian响应图在区域上的方差，量化局部结构复杂度（边界/纹理密集区域方差高）。
**Adaptive Spatial Scaffold**：S²Prune的第一阶段，用Laplacian方差将Token预算分配到不同空间区域，形成"结构自适应的空间支架"。
**Uniform Grid Sampling**：简单均匀网格采样策略，在图像各区域等密度放置Token，作为本文的重要baseline。
**Largest-Remainder Procedure**：预算分配算法，先按比例分配整数部分，再将剩余Token按小数部分大小依次分配，确保$\sum B_g = B$。
**M-RoPE**：Multi-dimensional RoPE，Qwen2.5-VL使用的多维位置编码，剪枝后需保留原始position ID避免错位。
**Training-free Pruning**：无需微调或额外训练的剪枝方法，直接利用模型前向传播中的中间信号决策。

## 可复现要素
- **数据集**：10项公开多模态基准（GQA、$\text{SQ A}^{\text{IMG}}$、$\text{VQA}^{\text{Text}}$、VizWiz、MMMU、POPE、MME、$\text{MMB}^{\text{EN}}$、$\text{MMB}^{\text{CN}}$、MMVet），均可公开获取。
- **代码**：已开源，地址github.com/yuanyuanjia71-spec/S2Prune。
- **权重**：使用官方Qwen2.5-VL-7B-Instruct和LLaVA-OneVision-7B checkpoint，无需额外训练。
- **关键超参**：
  - Token预算$B \in \{128, 64, 32\}$
  - 粗网格尺寸：$B=32 \to 4\times4$，$B=64 \to 5\times5$，$B=128 \to 8\times8$，$B=192 \to 9\times9$
  - 归一化epsilon：$\epsilon_{\text{norm}} = \text{float32.eps} \approx 1.19\times10^{-7}$
  - 分配epsilon：$\epsilon_{\text{alloc}} = \text{float64.eps} \approx 2.22\times10^{-16}$
- **实现细节**：图像resize到$672\times672$，patch size=14，spatial merge factor=2，最终$24\times24$ grid（576 tokens）；Laplacian用四邻域差分+replicate padding。
