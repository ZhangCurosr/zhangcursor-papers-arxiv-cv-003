---
title: "METEOVERSE-UNIFIED-WEATHER-CONTROLLABLE-VIDEO-WORLD-MODEL"
source: https://arxiv.org/pdf/2609.36810v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:43:11"
field: "可控视频生成与天气建模"
keywords: ["Video World Model", "Weather Control", "Diffusion Video Generation", "Mixture of Experts", "Camera-controllable Generation", "Weather Dataset"]
innovations: ["显式天气转换建模 s_trans，将天气保持/引入/移除统一为残差天气控制", "MeteoMoE：类别特定雨/雪/雾专家按转换强度调制后融合注入 DiT backbone"]
benchmarks: ["VBench-I2V", "Weather Alignment", "VLM Evaluation", "User Study"]
---

# 论文速读：METEOVERSE-UNIFIED-WEATHER-CONTROLLABLE-VIDEO-WORLD-MODEL

## 一句话总结
MeteoVerse 提出一种统一的可控天气视频世界模型，能从晴天或恶劣天气的单图出发，在给定天气无关场景描述、目标天气指令和相机轨迹的条件下生成未来视频，显式建模天气转换（保持、引入、移除）并通过 MeteoMoE 实现细粒度天气强度控制。

## 研究问题与动机
1. 现有视频世界模型将天气转换隐式处理，需同时推断场景动态、相机运动与天气演化，导致天气控制不精确。
2. 重建类方法（如 ClimateNeRF、WeatherCity）需多视角观测和昂贵的逐场景重建，难以直接应用于单图生成。
3. 视频天气编辑方法（如 WeatherWeaver、AutoAWG）假设完整源视频已存在，无法从单张输入预测未来。
4. 近期工作 Holo-World 虽将天气控制引入单图视频世界建模，但仅支持从晴天引入恶劣天气，天气移除未被探索。

## 核心贡献（创新点）
1. **统一天气可控视频世界模型**：MeteoVerse 支持从晴天或恶劣天气单图出发，统一实现天气保持、引入与移除，覆盖更完整的天气变换场景。
2. **显式天气转换建模 + MeteoMoE**：通过天气状态预测器得到 s_trans（目标与观察天气之差），并由类别特定的雨/雪/雾专家分别提取残差天气特征，相比仅用目标天气或文本指令，能精确表征"需要修改什么天气"。
3. **构建 MeteoVerse 数据集**：包含超 50K 真实天气视频片段、生成的晴天对应版本、解耦的场景/天气描述、天气强度标注和相机轨迹，为多类型天气转换提供监督信号。

## 方法详解
- **问题形式化**：将总条件 p 解耦为天气无关场景描述 p_s 与目标天气指令 p_w；天气状态预测器 P_φ 估计输入图像与目标天气的连续三维强度向量 s_src、s_tar，并计算 s_trans = s_tar − s_src（[rain, snow, fog] ∈ [0,1]^3）。
- **MeteoMoE**：在每个 DiT 块中，雨/雪/雾三类专家 token T_k 分别通过交叉注意力从 H_txt^ℓ 检索场景自适应的天气特征 E_k^ℓ；再由对应转换分量 δ_k 调制后通过轻量自注意力融合为 E_trans^ℓ，最后残差注入 backbone：H̃_txt^ℓ = H_txt^ℓ + E_trans^ℓ。
- **天气状态预测器**：基于 Qwen3-VL-2B 进行 LoRA 微调（rank=64, α=128），推理时冻结；对晴天真阴性表示为 s=[0,0,0]。
- **训练对构建**：利用 N 个伪配对晴天片段构造四组 (I, V) 对，分别监督晴天保留、恶劣天气保留、天气引入（晴天→恶劣）和天气移除（恶劣→晴天）。
- **网络与优化**：以 LingBot-World 为基础（含高噪粗生成与低噪精炼两阶段），MeteoMoE 插入两阶段每个 DiT 块的文本交叉注意力之后；冻结 backbone，训练 MeteoMoE + 自注意力/FFN 中的 LoRA（rank=64, α=128）。损失函数为 flow-matching 损失 L_fm 加帧级 L1 重建损失与 VGG 感知损失，权重 λ_l1=0.1、λ_vgg=0.05。

## 实验与结果
- **数据集与设置**：在 100 个未见真实场景上评测四种设置（晴天→晴天、晴天→恶劣、恶劣→晴天、恶劣→恶劣），共 400 用例；分辨率 480×832，81 帧，40 步去噪，CFG=5.0。
- **评估基线**：Uni3C、GEN3C、VerseCrafter、NeoVerse、LingBot-World。
- **天气保持**：MeteoVerse 取得最高 Overall Score（86.70）、Weather Alignment（93.00）和 User Study（80.90）。
- **天气引入**：Weather Alignment 从 29.00 提升至 61.00，VLM Evaluation 从 33.70 提升至 54.15，User Study 从 38.20 提升至 66.70。
- **天气移除**：Weather Alignment 从 38.00 提升至 89.00，VLM Evaluation 从 46.45 提升至 77.92，User Study 从 51.40 提升至 82.40。
- **天气预测器**：transition 分类准确率 99.00%，强度 MAE 为 0.0342；用预测状态替代 GT 状态仅使 Weather Alignment 下降 1.00（82.00→81.00），影响较小。
- **消融**：s_trans 优于 s_tar 和 p_w；类别特定专家优于全局条件（Weather Alignment 58.33）与共享专家（65.00）；去除 backbone LoRA 导致 Weather Alignment 降至 52.02。

## 相关工作脉络
1. **Holo-World**：首个将天气控制引入单图视频世界模型的参考工作，但仅支持晴天→恶劣天气的"天气引入"，本文扩展至天气保持与移除的统一框架。
2. **Video weather editing（WeatherWeaver、AutoAWG）**：需完整源视频，不能从单图预测未来，本文在单图+相机轨迹条件下实现天气控制。
3. **重建类天气合成（ClimateNeRF、WeatherCity、WeatherEdit）**：依赖多视角/几何重建，计算成本高；本文使用生成范式直接输出视频。
4. **Camera-controllable world models（Uni3C、GEN3C、VerseCrafter、NeoVerse、LingBot-World）**：这些基线侧重场景/相机控制，对天气转换缺乏显式建模，本文在同类架构基础上加入天气转换分支。
5. **天气数据集（ACDC、SHIFT、Weather-Stream 等）**：多面向恶劣天气感知/去污，本文数据集提供天气强度标注、相机轨迹与伪配对晴天，支撑天气可控生成任务。

## 局限性与未来方向
1. 晴天伪配对由生成模型构建（Nano Banana 2 + Qwen3-VL-30B），可能引入生成偏差，影响天气保留/引入的真实一致性。
2. 天气状态预测器基于 LLM-VL，推理时需额外前向计算，且对极端复杂天气（如雨雪雾共存）的精细分解仍有提升空间。
3. 当前强度表示为 3 维连续向量，未考虑空间不均匀性（如局部雾/全局雨），难以建模场景内非均匀天气分布。
4. 训练依赖 8×A800，未给出更低资源下的适配方案，限制了在消费级硬件上的部署。
5. 数据集仅覆盖雨/雪/雾三类，对沙尘、冰雹等其他恶劣天气尚未涉及。

## 研究启发与可借鉴点
1. **"状态差"而非"目标态"作为控制信号**：显式计算 s_trans 将"需要改什么"解耦出来，适用于其他需要"变换量"控制的任务（如光照、季节、风格迁移）。
2. **类别特定专家 + 转换调制**：MeteoMoE 的思路可推广到多模态/多风格混合生成中，每个专家专注一类效应并按需求强度加权融合。
3. **伪配对数据构建策略**：利用生成模型构建"对照"样本（晴天→恶劣/恶劣→晴天）以扩展监督类型，是一种低成本扩展训练数据的有效范式。
4. **冻结 backbone + 轻量适配器（LoRA + MoE）**：在预训练视频世界模型基础上仅训练天气子模块，保持场景/相机生成能力不被破坏，值得在细粒度控制任务中复用。
5. **VLM-based 评估与用户研究结合**：除传统 VBench 外，引入 Weather Alignment 和 VLM Evaluation 多粒度指标，更贴合天气控制任务的真实需求。

## 关键术语表
**Video World Model**：从单帧观测出发，在给定相机轨迹等条件下预测未来视频生成的模型。
**Weather-state predictor**：基于 Qwen3-VL-2B（LoRA 微调）估计输入图像与目标天气的连续强度向量的模块。
**MeteoMoE**：Transition-aware Mixture of weather Experts，雨/雪/雾三类专家分别提取场景自适应天气特征并按转换量调制后融合注入 DiT backbone。
**Weather transition (s_trans)**：目标天气与观察天气之差，正值表示引入该类天气，负值表示移除，零表示保持。
**Pseudo-paired sunny counterpart**：利用生成模型从恶劣天气帧合成对应的晴天版本，用于构造天气移除/引入监督对。
**Flow-matching loss**：视频扩散模型优化的目标，引导模型预测潜变量在流时间 t 处的目标速度。
**VBench-I2V**：用于图像到视频生成的全面评测基准，包含主体一致性、背景一致性、动态程度等指标。
**Weather Alignment**：基于规则阈值判断生成视频是否符合目标天气指令的百分比指标。

## 可复现要素
- **数据集**：MeteoVerse 数据集，包含超 50K 真实天气视频片段及伪配对晴天视频；项目页面 https://meteoverse.github.io/（论文未明确说明是否完全开源）。
- **代码/权重**：论文未明确声明开源代码或模型权重是否公开。
- **关键超参**：
  - 天气状态预测器（Qwen3-VL-2B LoRA）：rank=64，α=128，lr=1e-4，batch=4。
  - 视频 backbone LoRA：rank=64，α=128，lr=1e-5（LoRA）/ 5e-5（其他），AdamW + cosine decay，gradient clipping=1.0。
  - 损失权重：λ_l1=0.1，λ_vgg=0.05。
  - CFG=5.0，40 步去噪，渐进训练：clip 21→41→81 帧，分辨率 240×416→480×832。
  - 硬件：8× NVIDIA A800-SXM4，BF16 精度。
