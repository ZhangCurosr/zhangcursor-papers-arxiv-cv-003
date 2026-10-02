---
title: "METEOVERSE-UNIFIED-WEATHER-CONTROLLABLE-VIDEO-WORLD-MODEL"
source: https://arxiv.org/pdf/2609.36810v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:43:02"
field: "视频生成与天气控制"
keywords: ["视频世界模型", "天气可控生成", "天气过渡建模", "MeteoMoE", "相机控制视频生成", "天气移除与引入"]
innovations: ["显式天气过渡建模：通过观测与目标天气之差推导所需修改，替代仅目标条件化", "MeteoMoE 类别特定专家混合：雨/雪/雾独立专家并通过过渡分量调制，实现精细强度控制", "统一天气保留/引入/移除：支持晴天和恶劣天气双向输入，覆盖完整天气转移类型"]
benchmarks: ["VBench-I2V", "Weather Alignment", "VLM Evaluation", "User Study", "VGGT-Omega 相机误差"]
---

# 论文速读：METEOVERSE-UNIFIED-WEATHER-CONTROLLABLE-VIDEO-WORLD-MODEL

## 一句话总结
MeteoVerse 提出了一种统一的天气可控视频世界模型，通过显式估计观测天气与目标天气状态并计算所需过渡，结合天气专家混合模块（MeteoMoE），实现从单张晴天或恶劣天气图像出发，在指定相机轨迹下生成未来视频，支持天气保留、引入和移除三种模式及精细强度控制。

## 研究问题与动机
1. **现有视频世界模型忽略天气演化**：真实世界场景演化受天气（雨、雪、雾）影响显著，但现有视频世界模型大多仅关注场景内容与视角变化，未对天气转移进行显式建模与控制。
2. **天气转移具有双向依赖关系**：同样的目标天气指令对应不同的操作——晴天→雨天需"引入"，雨天→雨天需"保留"，雨天→晴天需"移除"，仅靠目标天气无法明确所需转换。
3. **已有天气控制方法存在局限**：重建类方法需要多视图和昂贵重建；视频天气编辑需完整源视频；Holo-World 虽进入单图视频世界建模，但仅支持晴天输入和天气引入，天气移除未被探索。
4. **隐含建模导致控制不精确**：若让视频骨干网络同时推断观测天气、解析目标指令、确定修改策略并建模场景演化与相机运动，会导致天气控制精度不足。

## 核心贡献（创新点）
1. **统一天气可控视频世界模型**：提出 MeteoVerse，支持从晴天或恶劣天气单图出发，统一处理天气保留、引入和移除，区别于 Holo-World 仅支持晴天输入的单一模式。
2. **显式天气过渡建模 + MeteoMoE**：通过天气状态预测器估计观测/目标天气并推导过渡向量，再将过渡转化为类别特定的残差天气特征注入 DiT 骨干，实现引入强度的精细控制；与直接条件化目标天气的本质区别在于显式编码了"需要做什么修改"。
3. **大规模天气视频世界建模数据集**：构建 MeteoVerse 数据集，包含超过 50K 真实雨/雪/雾视频片段、生成的晴天对应版本、解耦场景与天气描述、天气强度标注和相机轨迹，填补了该方向的监督数据空白。
4. **兼顾场景一致性与天气可控性**：冻结预训练骨干并仅微调 MeteoMoE 与 LoRA 适配器，使得场景演化与相机运动保持原有质量，同时在天气对齐指标上获得大幅提升。

## 方法详解
1. **问题形式化**：给定输入图像 $I$、天气无关场景描述 $p_s$、相机轨迹 $\mathcal{C}$ 和过渡向量 $\mathbf{s}_{trans}$，生成未来视频 $V_{1:T}$：
   $$p_{\theta,\psi}(V_{1:T} | I, p_s, \mathcal{C}, \mathbf{s}_{trans})$$
   其中 $\theta$ 为冻结的骨干参数，$\psi$ 为可训练的 MeteoMoE 与 LoRA 参数。

2. **天气状态表示**：每种天气状态用三维连续强度向量表示 $\mathbf{s} = [s_{rain}, s_{snow}, s_{fog}] \in [0,1]^3$，晴天为 $\mathbf{s}=[0,0,0]$；过渡向量 $\mathbf{s}_{trans} = \mathbf{s}_{tar} - \mathbf{s}_{src}$，各分量正/负/零分别对应引入/移除/保留对应天气。

3. **天气状态预测器**：基于 LoRA 微调 Qwen3-VL-2B，秩和缩放因子均为 64/128；训练时直接使用标注状态计算过渡，推理时冻结预测器。

4. **MeteoMoE 架构**：每个 DiT 层中，以文本交叉注意力后的特征 $H_{txt}^\ell$ 为查询，从三类专家 token $T_k$（$k \in \{rain, snow, fog\}$）中提取类别自适应特征：
   $$E_k^\ell = CA_k^\ell(Norm(H_{txt}^\ell), T_k, T_k)$$
   随后按过渡分量 $\delta_k$ 调制后通过轻量自注意力融合：
   $$E_{trans}^\ell = SA_w^\ell(\{\delta_k E_k^\ell\}_{k})$$
   最终以残差方式注入骨干：$\tilde{H}_{txt}^\ell = H_{txt}^\ell + E_{trans}^\ell$。

5. **训练损失**：基于 flow-matching 目标 $\mathcal{L}_{fm}$，辅以帧级 $\ell_1$ 重建损失（$\lambda_{\ell_1}=0.1$）和 VGG 感知损失（$\lambda_{vgg}=0.05$）。

6. **数据集构建**：收集 50K+ 81帧真实天气视频，用 Qwen3-VL-8B/30B 生成场景描述和天气描述，用 VGGT-Omega 估计相机轨迹，采用粗→细强度标注策略并经 GPT-5.6 验证；通过 Nano Banana 2 生成伪配对晴天帧并用 LingBot-World 扩展为晴天视频。

7. **训练对构造**：构建四类样本对 $(I_{sun}, V_{sun})$、$(I_{adv}, V_{adv})$、$(I_{sun}, V_{adv})$、$(I_{adv}, V_{sun})$，分别对应晴天保留、恶劣天气保留、天气引入和天气移除的监督。

## 实验与结果
- **数据集与测试设置**：100 个未见真实场景，四个天气设置（晴→晴、晴→恶劣、恶劣→晴、恶劣→恶劣），共 400 个测试案例；生成 81 帧、480×832、40 步去噪、CFG=5.0。
- **基线方法**：Uni3C、GEN3C、VerseCrafter、NeoVerse、LingBot-World。
- **天气保留（Weather Preservation）**：MeteoVerse 获得最高 Overall Score（86.70）、Weather Alignment（93.00）和用户研究（80.90），同时保持有竞争力的 VBench-I2V 和相机控制指标。
- **天气引入（Weather Introduction）**：Weather Alignment 从基线最高 29.00 提升至 61.00（+32.00），VLM Evaluation 从 33.70 提升至 54.15（+20.45），用户研究从 38.20 提升至 66.70（+28.50）。
- **天气移除（Weather Removal）**：Weather Alignment 从基线最高 38.00 提升至 89.00（+51.00），VLM Evaluation 从 46.45 提升至 77.92（+31.47），用户研究从 51.40 提升至 82.40（+31.00）。
- **消融结论**：显式过渡状态 $\mathbf{s}_{trans}$ 优于目标天气 prompt 或目标状态；类别特定专家优于全局条件化或共享专家；骨干 LoRA 适配器对维持天气对齐至关重要。

## 相关工作脉络
1. **Video World Models**（GAIA-1、Genie、Matrix-Game、LingBot-World）：通用未来预测框架，但不显式建模天气转移；MeteoVerse 在此基础上加入天气控制维度。
2. **Camera-controlled Video Generation**（MotionCtrl、CameraCtrl、CamI2V、Voyager）：关注相机轨迹控制与场景探索，未涉及天气演化；MeteoVerse 复用其相机控制范式并将其与天气控制统一。
3. **Holo-World**（Yin et al., 2026）：首个将天气控制引入单图视频世界模型的 work，但仅支持晴天输入和天气引入；MeteoVerse 扩展至晴天/恶劣天气双向输入并统一保留/引入/移除。
4. **Weather Synthesis / Editing**（ClimateNeRF、WeatherEdit、AutoAWG、IntrinsicWeather）：重建类需要多视图，视频编辑类需要完整源视频；MeteoVerse 从单图出发同时预测未来视角和天气变化。
5. **Adverse Weather Datasets**（ACDC、SHIFT、Weather-Stream）：主要面向感知/去污任务；MeteoVerse 数据集面向生成任务，提供天气强度标注、解耦描述和相机轨迹。

## 局限性与未来方向
1. **天气状态预测器存在误差**：虽然预测精度高（状态 MAE 0.0119），但在极端强度或模糊边界下仍可能有误判，进而影响下游生成。
2. **伪配对晴天视频的生成质量依赖上游模型**：Nano Banana 2 和 LingBot-World 的质量直接影响训练数据可信度，可能存在场景内容漂移。
3. **仅支持雨/雪/雾三类天气**：未覆盖其他气象现象（如沙尘暴、雷电、冰雹等）。
4. **计算开销**：需要额外的天气状态预测器和三类专家模块，推理时略高于原始 LingBot-World。
5. **长期一致性未充分验证**：当前聚焦 81 帧中等长度，更长序列的天气演化一致性仍需探索。

## 研究启发与可借鉴点
1. **过渡状态建模思想可迁移**：将"目标 - 观测"之差作为条件信号，适用于其他需要状态转移控制的生成任务（如昼夜交替、季节变换、场景老化等）。
2. **类别特定专家混合架构**：MeteoMoE 按天气类别分离专家并通过过渡分量调制的策略，可推广至多模态风格迁移或多属性联合控制场景。
3. **伪配对数据构建 pipeline**：利用现有生成模型构建配对数据的流程（Nano Banana 2 去污 → Qwen 筛选 → LingBot-World 扩展）可作为数据受限任务的有效解决方案。
4. **冻结骨干 + 轻量适配器的训练范式**：保持预训练世界模型的场景一致性和相机控制能力，仅以少量参数增量实现新能力，适合资源受限的科研团队复现。
5. **多维度评估体系设计**：同时使用 VBench-I2V、相机误差、Weather Alignment、VLM Evaluation 和用户研究，避免单一指标被优化而实际能力缺失，值得借鉴。

## 关键术语表
- **Video World Model**：从单帧观测出发，结合相机轨迹和文本条件，预测未来多帧视频的基础模型。
- **Weather Transition**：观测天气状态与目标天气状态之间的差异向量，显式编码需要执行的天气修改操作。
- **MeteoMoE**：过渡感知的天气专家混合模块，通过三类类别特定专家（雨/雪/雾）将过渡向量转换为残差天气特征并注入 DiT 骨干。
- **Flow-matching**：一种扩散模型训练目标，通过优化 latent 在噪声分布与数据分布之间的流速度来实现生成。
- **VBench-I2V**：面向图像到视频生成的综合评测基准，包含主体一致性、背景一致性、运动平滑度、动态度、美学质量和成像质量等子指标。
- **Weather Alignment**：基于 VLM 判定的天气对齐指标，衡量生成视频中请求天气条件的出现比例。
- **Pseudo-paired Sunny Counterpart**：通过去污模型将恶劣天气帧转换为晴天帧，再利用预训练世界模型生成对应晴天视频，构成伪配对训练数据。
- **LoRA**：低秩自适应微调技术，通过低秩矩阵分解对预训练模型进行参数高效微调。

## 可复现要素
- **数据集**：MeteoVerse 数据集（50K+ 视频片段），论文未明确说明是否开源，项目页面为 https://meteoverse.github.io/。
- **代码**：论文未提及代码是否开源。
- **权重**：骨干基于 LingBot-World，天气状态预测器基于 Qwen3-VL-2B/30B，伪晴天生成使用 Nano Banana 2 和 VGGT-Omega；论文未说明是否提供官方微调权重。
- **关键超参**：
  - LoRA rank=64，scaling factor $\alpha$=128（预测器和骨干共用）。
  - 天气状态学习率：LoRA 部分 $1\times10^{-4}$。
  - 骨干 LoRA 学习率 $1\times10^{-5}$，其余可训练参数 $5\times10^{-5}$。
  - 训练 GPU：8× NVIDIA A800-SXM4，batch size=1/GPU，BF16。
  - Flow-matching 辅助损失权重：$\lambda_{\ell_1}=0.1$，$\lambda_{vgg}=0.05$。
  - 推理：81帧、480×832、40步去噪、CFG=5.0。
