---
title: "S-sup-2-sup-Prune-Spatially-Structured-Visual-Token-Pruning"
source: https://arxiv.org/pdf/2609.01224v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 11:51:21"
field: "多模态大模型高效推理"
keywords: ["visual token pruning", "multimodal large language models", "spatial allocation", "training-free acceleration", "Laplacian structure", "Early Representation Change"]
innovations: ["将视觉token剪枝重新定义为空间速率分配问题，解耦'预算分布'与'局部选择'", "用Laplacian响应方差无训练分配区域token预算并自适应结构复杂度", "提出ERC作为第一解码器层的轻量代表性选择信号"]
benchmarks: ["Qwen2.5-VL-7B-Instruct", "LLaVA-OneVision-7B", "MMB^EN", "POPE", "GQA", "MMMU", "MME", "VQA^Text", "MMVet"]
---

# 论文速读：S²Prune: Spatially Structured Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
本文提出S²Prune，一种无训练的视觉token剪枝方法，通过将剪枝视为"空间速率分配"问题，在保持广泛空间覆盖的前提下，根据Laplacian结构响应自适应调整各区域的token密度，并用第一解码器块的Early Representation Change（ERC）在局部单元内选择代表性token。在Qwen2.5-VL-7B-Instruct上，以32个token（原576个）达到55.9平均准确率，保留全模型79.3%性能，为同条件无训练方法中最高。

## 研究问题与动机
- **核心问题**：现有视觉token剪枝方法依赖重要性/相关性打分+全局Top-B选择，但不同打分标准会产生稳定的空间偏差，且未必优于简单Uniform Grid采样，说明"在哪放token"与"放哪些token"是两个需要解耦的问题。
- **现有方法的不足**：
  1. 基于attention/ relevancy的打分方法（如FastV、VisionZip）显著偏向图像边缘（约58%保留token落在边缘区域），对图像内部结构差异不敏感；
  2. Uniform Grid采样能维持广泛空间覆盖，但各区域token密度固定，无法适应图像内部结构异质性（平滑背景 vs. 边界/文字/细结构）；
  3. 已有空间感知方法（如GridPrune、FEATHER）将区域预算用于定位"任务相关证据位置"，而非估计"该区域所需表示容量"，语义相关性与表示需求并不对齐。

## 核心贡献（创新点）
1. **将视觉token剪枝重新定义为空间速率分配问题**：指出全局token打分不仅决定"保留哪些token"，还隐式决定了"token预算落在何处"，进而提出分离"空间支架搭建"与"局部代表选择"的设计原则。
2. **提出无训练的S²Prune框架，用Laplacian响应方差衡量区域结构复杂度来分配token预算**：与GridPrune等"查询条件化预算分配"不同，本文用图像本征结构变化（与问题无关）决定每区域token数，语义相关性与表示容量需求不再混淆。
3. **引入Early Representation Change（ERC）作为局部代表性选择信号**：仅用第一解码器块的表示变化（$r_i = \|h_i^1 - h_i^0\|_2$）在已分配的局部单元格内选representative，避免深层计算开销，且在多项消融中优于Hidden Norm、Query Attention等替代方案。

## 方法详解
S²Prune是无训练的两阶段框架，在decoder Layer 0之后、Layer 1之前执行剪枝：

**阶段一：自适应空间支架搭建（Laplacian-based regional budget allocation）**
- 将图像划分为$G=K^2$个不相交粗区域$\{\mathcal{R}_g\}$（$G \le B$），如$B=32/64/128$分别对应$4\times4/5\times5/8\times8$网格；
- 对输入图像转灰度、边缘replicate padding后，计算全图四邻域Laplacian响应图$L$：
  $L(u,v) = I(u{-}1,v) + I(u{+}1,v) + I(u,v{-}1) + I(u,v{+}1) - 4I(u,v)$；
- 每个区域的结构原始分数$c_g^{\text{raw}} = \text{Var}(L|_{\Omega_g})$，再在整图内做min–max归一化：
  $\tilde{c}_g = \frac{c_g^{\text{raw}} - \min_j c_j^{\text{raw}}}{\max(\max_j c_j^{\text{raw}} - \min_j c_j^{\text{raw}}, \epsilon_{\text{norm}})}$；
- 每个区域至少分配1个token以保覆盖，剩余$B-G$个token按$\tilde{c}_g$比例用最大余数法分配，上限为区域token总数：$1 \le B_g \le |\mathcal{R}_g|$，$\sum_g B_g = B$；
- 每个区域$\mathcal{R}_g$递归划分为$B_g$个面积近似相等的矩形单元格$\{\mathcal{C}_{g,k}\}$（优先沿长边二分）。

**阶段二：局部代表性选择（ERC-based local token selection）**
- 令所有视觉token通过decoder Layer 0，计算早期表示变化：$r_i = \|h_i^1 - h_i^0\|_2$（即第一个residual块带来的更新范数）；
- 在每个单元格$\mathcal{C}_{g,k}$内独立选取ERC最大者：$i_{g,k}^\star = \arg\max_{i \in \mathcal{C}_{g,k}} r_i$；
- 保留的$B$个token按原始decoder-visible顺序排列（维持M-RoPE位置ID不变），物理删除未选中token后继续Layer 1~N解码。

## 实验与结果
- **模型与设置**：主实验使用Qwen2.5-VL-7B-Instruct（576 token → 128/64/32）；跨架构验证使用LLaVA-OneVision-7B（729 token → 160/80/40）；10个多模态基准（GQA、$\text{SQ A}^{\text{IMG}}$、$\text{VQA}^{\text{Text}}$、VizWiz、MMMU、POPE、MME、$\text{MMB}^{\text{EN}}$、$\text{MMB}^{\text{CN}}$、MMVet）。
- **基线**：FastV、DART、GPrune、VisionZip、VisPruner、GridPrune、HoloV（均为无训练方法）。
- **主要结果（Qwen2.5-VL-7B）**：
  - $B=128$：Acc=64.0，Rel=92.1%，超越最强基线0.1点；
  - $B=64$：Acc=60.6，Rel=87.0%，超越最强基线0.3点；
  - $B=32$：Acc=55.9，Rel=79.3%，超越最强基线（GridPrune 54.4）0.9点，并在GQA/POPE/$\text{MMB}^{\text{EN}}$等关键指标上领先2.2~3.7点；
  - 在128/64/32三档预算下均为所评无训练方法中平均准确率最高。
- **跨架构**：在LLaVA-OneVision-7B上，160/80/40三档均达最好Acc/Rel（如160 token时Acc=64.5，Rel=94.6%），验证可迁移性。
- **效率（$B=64$，POPE）**：FLOPs从8.181T降至1.596T（5.1×缩减），prefill延迟从56.97ms降至28.66ms（~2×加速），KV cache从31.50MB降至4.50MB（7.0×缩减），在精度-效率权衡上最优。

## 相关工作脉络
1. **FastV (ECCV'24)**：基于attention score在Layer 2后剪枝至1/2 token；S²Prune指出其存在明显边界偏好，且未考虑空间结构异质性。
2. **FEATHER (ICCV'25)**：提出uniform sampling维持广泛覆盖；本文肯定其空间覆盖价值，但进一步用Laplacian方差将"均匀覆盖"升级为"结构自适应密度"。
3. **GridPrune (arXiv'25)**：查询条件化的区域预算分配（先决定"看哪里"，再决定"选什么"）；本文认为语义相关≠表示容量需求，改用图像本征结构信号分配预算。
4. **VisionZip / VisPruner (CVPR/ICCV '25)**：结合visual attention与相似性/多样性选择；本文指出这类全局打分仍会产生稳定空间偏差，主张将空间规划与局部选择解耦。
5. **HoloV (NeurIPS'25)**：通过空间裁片分配预算保留holistic context；本文与之区别在于不使用语义裁片，而用轻量Laplacian信号直接估计结构复杂度。
6. **DART / GPrune (EMNLP/AAAI '25)**：分别基于去重与图传播选择token；本文消融显示Laplacian结构先验与ERC组合在同等空间分配下显著优于这些信号。

## 局限性与未来方向
- **结构先验的语义盲区**：Laplacian方差只捕获局部梯度变化，无法直接建模高层语义重要性（如"文字区域"可能比"复杂纹理"更关键），极端情况下可能出现结构丰富但语义低价值的区域被过度分配。
- **静态空间划分**：采用规则网格粗分区，未考虑对象尺度与形状的不规则性；对于细长物体或极小目标，固定矩形分区可能造成局部覆盖不均。
- **仅测试两层架构**：跨架构实验仅在LLaVA-OneVision-7B上做验证，尚未在更强或更大参数规模模型（如70B级）上检验泛化。
- **未来方向**：可探索结构与语义信号的联合分配（如融合text-guided attention与Laplacian）；引入对象级超像素/分割先验替代规则网格；扩展到视频/3D等多模态序列剪枝。

## 研究启发与可借鉴点
1. **"空间支架 + 局部代表"的解耦设计**：将token剪枝拆分为"预算如何分布"和"每个位置上选谁"两个子问题，可有效避免全局打分带来的空间偏差；该思路可迁移到视觉编码器压缩、特征图稀疏化等场景。
2. **用低频级视觉统计（Laplacian方差）替代高成本语义打分做结构估计**：仅需一次前向灰度变换与4-邻域滤波即可得到高质量结构复杂度信号，计算开销几乎可忽略，适合部署在资源受限推理流水线中。
3. **ERC（第一层表征变化范数）作为轻量代表性信号**：相比query attention或hidden norm，ERC直接度量"该token对下游表征的实际扰动"，在同等空间分配下表现最稳定；可作为通用early-layer selection heuristic被复用。
4. **可控实验范式**：固定分区/预算/局部选择器，只替换区域分配策略（Uniform/Random/FFT/Effective Rank/Laplacian），可清晰隔离各组件贡献，该方法论对后续剪枝工作具有示范价值。

## 关键术语表
- **S²Prune**：Spatially Structured Visual Token Prune的缩写，本文提出的无训练视觉token剪枝框架。
- **ERC (Early Representation Change)**：第一decoder block前后隐藏状态的$\ell_2$范数变化，用作局部单元格内代表性token的选择信号。
- **Laplacian响应方差**：对图像灰度做四邻域Laplacian滤波后，各粗区域响应图的方差，用于量化区域结构复杂度。
- **Uniform Grid采样**：按规则网格等密度抽取token，作为本文对比的强空间覆盖基线。
- **最大余数分配法**：按权重比例取整数部分后，按小数余数依次加1，直到总配额用完；用于将连续权重转为整数token预算。
- **Rel. (Relative Retention)**：各剪枝方法在单个基准上的分数与全token模型分数的比值，再对10个基准取均值，用于衡量"性能损失率"。
- **coarse grid**：将视觉token网格划分为若干不相交区域的基础划分（如$5\times5$），每个区域得到至少1个token以保证空间覆盖。
- **M-RoPE**：Multi-dimensional RoPE，Qwen2.5-VL使用的多维旋转位置编码，剪枝后保留原始位置ID不被重编号。

## 可复现要素
- **数据集**：GQA、SQ A^IMG、VQA^Text、VizWiz、MMMU、POPE、MME、MMB^EN、MMB^CN、MMVet（公开数据集，官方评估协议）。
- **代码/权重**：代码已开源（github.com/yuanyuanjia71-spec/S2Prune）；模型权重使用官方Qwen2.5-VL-7B-Instruct与LLaVA-OneVision-7B。
- **关键超参**：目标token预算$B \in \{32, 64, 128\}$，对应粗网格$K \in \{4, 5, 8\}$；Laplacian使用四邻域差分与replicate padding；归一化$\epsilon_{\text{norm}} = \text{float32 eps} \approx 1.19\times10^{-7}$；分配阈值$\epsilon_{\text{alloc}} = \text{float64 eps} \approx 2.22\times10^{-16}$；ERC范数以FP32计算。
