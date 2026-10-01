---
title: "Long-Time-No-See-Benchmarking-VLMs-for-Out-of-Sight-Spatiote"
source: https://arxiv.org/pdf/2609.34630v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:26:43"
field: "具身视觉-语言模型评测"
keywords: ["VLM", "egocentric video", "out-of-sight reasoning", "spatiotemporal reasoning", "VQA benchmark", "HD-EPIC", "3D spatial perception", "object permanence"]
innovations: ["提出首个要求目标在查询时刻几何验证为不可见且已重定位的VQA基准BEYOND3D", "设计四级诊断问题链（视觉/时序/场景/3D）与三阶段可见性轨迹推断管线", "揭示时间证据检索是视外推理的上游瓶颈，3D预测存在前置/近距离强偏向"]
benchmarks: ["BEYOND3D"]
---

# 论文速读：Long-Time-No-See-Benchmarking-VLMs-for-Out-of-Sight-Spatiote

## 一句话总结
论文提出 **BEYOND3D**，首个面向第一人称视频中"视外时空推理"（out-of-sight spatiotemporal reasoning）的VQA基准，要求模型在物体被移动且已离开视野后，仍能跟踪其空间状态并回答定位与时序问题；9个主流VLM最佳成绩仅42.2%，远超29.7%的随机猜测，表明视外推理仍是当前模型的显著瓶颈。

## 研究问题与动机
- **核心问题**：具身AI/AR系统需要推理"当前看不到的物体"——不仅记住它上次出现的位置，还要在物体被移动后更新其状态，并在它离开视野后仍保留该更新状态。
- **现有基准不足**：已有基准（如EgoTempo、VSI-Bench、EOC-Bench、Ego4D-VQ3D等）要么不要求目标物体在查询时不可见，要么不要求物体曾被主动移动更新状态，模型可通过"只看当前帧"短路作答，无法隔离真正的视外推理能力。
- **第一人称场景的特殊挑战**：佩戴者运动不断改变可见性，同时物体被频繁抓取和重新放置，物体状态随时间持续演化，使得"上次可见何时""最后放在哪"难以靠单帧恢复。
- **研究缺口**：缺乏一个严格保证目标在查询时刻几何验证为"不可见"、且每个问题均可拆解为四级诊断链路的基准。

## 核心贡献（创新点）
1. **提出"视外时空推理"问题定义**：建模物体可见性状态 $v_o(t)$、3D位置 $\ell_o(t)$、支撑装置 $f_o(t)$ 及视外地平线 $h(o, T_q)$，为后续系统评测提供形式化基础；区别于已有工作仅在静态或当前可见条件下评测3D推理。
2. **构建 BEYOND3D 基准（9,000问/135视频/8类问题）**：从 HD-EPIC 出发，用三阶段几何+检测管线推断每物体1fps可见性轨迹，并只在目标明确"不可见"的时刻生成问题；这与 SCP-Bench / Ego4D-VQ3D 仅利用可见片段形成本质差异。
3. **四级诊断型问题设计**：将任务分解为 Visual Grounding → Temporal Grounding → Scene Localization → 3D Spatial Perception 四条链路，使错误来源可归因；此前基准（如 EgoMemReason、SpaMEM）多为端到端打分，不可诊断。
4. **首次系统评测9个VLM在严格视外条件下的表现**：发现时间证据检索是最上游瓶颈，3D空间预测存在强烈的"前置+近距离"偏向；这一发现揭示了当前VLM并非缺3D能力，而是上游记忆维护不足。

## 方法详解
- **问题形式化**：给定第一人称视频 $[0, T_q]$ 和目标物体 $o$，需在其于 $T_q$ 时刻满足 $\ell_o(T_q) \neq \bot$ 且 $v_o(T_q)=0$（已移动且不可见）的条件下，回答关于其"最后已知位置与状态"的8类问题。视外地平线 $h(o,T_q) = T_q - \max\{t \le T_q \mid v_o(t)=1\}$ 刻画"多久没看见"。
- **可见性轨迹推断（三阶段）**：
  1. **视场投影**：把物体端点9个支撑点（中心+四角+四边中点）反投影到3D，再经 Project Aria 的鱼眼模型 $FISHEYE624$ 重投影到当前帧；>50% 点落在有效圆形区域内判为"in view"，否则"out of view"。
  2. **几何遮挡**：对in-view点射线求与静态厨房mesh交点，首次交点比目标点近 $\ge \delta=10$ cm 即阻断；≥50% 射线被阻断判为"occluded"；若阻断主要来自可动fixture（抽屉/柜门）则标"fixture ambiguous"。
  3. **OWLv2检测确认**：对通过前两阶段帧以1fps检测，匹配框放大20px后包含投影锚点才算确认；检测分数≥50%判为"detected visible"，否则若上一状态为fixture ambiguous则标"occluded"，否则"visually unconfirmed"。
- **可见性轨迹验证**：人工审核344帧/4,102个标记，准确率83.5%（precision 81.2%, recall 62.7%），主要误差来自 OWLv2 漏检导致的"visually unconfirmed"。
- **Q&A构造**：
  - 候选锚点要求目标在 $T_q$ 明确为 out-of-view 或 occluded，排除 transient 的 visually unconfirmed。
  - 8类问题模板用占位符填充，答案由轨迹/标注/相机位姿推导，干扰项按时间距离分近/中/远/极远4档采样，或按fixture/3D方向/距离类别构造。
  - 最终评测集1,000个视外锚点×8类=8,000问 + 1,000个可见正对照锚点（仅Visibility Check）= 共9,000问。
- **3D空间答案推导公式**：以相机姿态 $(c_q, R_q)$ 为基准，目标世界坐标 $\ell_o(T_q)$ 转换到相机坐标系 $p_o^{cam}=R_q(\ell_o(T_q)-c_q)$，由此得方向与距离；对象-对象相对问题则将原点移至可见参考物 $\ell_r(T_q)$，轴向保持 $R_q$，即 $p_{o|r}^{cam}=R_q(\ell_o(T_q)-\ell_r(T_q))$。

## 实验与结果
- **数据集**：基于 HD-EPIC，135个视频/9名参与者/9个厨房/581个目标实例/561个参考对象；1,000个视外锚点按 $T_q$ 三分位×视外地平线三分位分层均衡。
- **评测基线**：9个VLM（Qwen-3.6-27B/A3B、Qwen-3.5-9B、Qwen-3-VL-8B、InternVL-3.5-8B、VLM-3R-7B、Spatial-MLLM-6B、Cambrian-P-7B、SenseNova-SI-8B），零样本、无微调，三种输入设置（Text Only / Last Frame Only + Text / Video + Text）。
- **主要结果（Video+Text，Table 2）**：
  - 最佳模型 **Qwen-3.6-27B** 宏观平均 **42.2%**，显著高于文本基线（29.7%）和随机猜测（各类20%–50%）。
  - 各题型相对随机提升：Nearest Fixture +22.9pp（最高）、Visibility Check +9.6pp；Last Visible Time 仅 +3.6pp、Last Placement Time +5.4pp、3D Direction/Distance 仅 +1–3pp，说明时间和3D推理仍是严重短板。
  - 通用模型整体优于空间专用模型（Qwen-3.5-9B 38.6% > 任何同量级专用模型31.6–37.2%）。
  - Last Frame Only 使 Visibility Check 大幅提升至平均 ~70%，但其余题型仍接近随机，证明"单帧可见性"与"跨视频历史的状态恢复"可以被该基准解耦。
- **诊断发现**：
  - 视外地平线越长，准确率从 40.4%（short）降至 31.9%（long）；查询时刻非单调（early 34.9% / middle 38.4% / late 36.6%）。
  - 给模型显式 temporal cue（运动时间段列表）可整体提升，证实"时间证据检索"是上游瓶颈。
  - 再加 visibility cue 仅让 Nearest Fixture 再+8%，3D任务几乎无改善（0–2.4pp）。
  - 模型3D方向预测强烈偏向"前置"（最高 Spatial-MLLM 达98.7% front），距离预测集中最短档（Cambrian-P 接近100% 选最近档）。

## 相关工作脉络
- **EgoTempo / EgoMemReason**：侧重事件时序与跨段记忆，但不 grounding 到 3D 位置，也不要求查询时目标不可见；BEYOND3D 在此基础上加入 3D spatial + geometry-verified out-of-sight。
- **VSI-Bench**：静态3D场景理解，无时序/对象重定位；本文在动态、可交互的第一人称场景上推进。
- **EOC-Bench / HD-EPIC / EgoDynamic4D**：动态空间状态追踪，但允许目标在查询帧仍然可见，模型可走"当前帧 shortcut"；BEYOND3D 强制要求 object-out-of-sight。
- **Ego4D-VQ3D**：检索"之前看到的静止物体位置"，不涉及物体被主动移动后的状态更新；本文强调 relocation 引发的 spatial-state revision。
- **SpaMEM**：合成环境下的 spatial-state revision 评测；本文用真实 unscripted 第一人称视频 + 几何可见性验证，生态效度更高。
- **SCP-Bench / UCS-Bench**：前者推断 unseen 过去/未来状态，后者用户中心关系变化；两者均不要求 target 在查询时明确 out-of-view 并由几何确认。

## 局限性与未来方向
- **OWLv2 漏检带来的 recall 偏差**：pipeline recall 仅 62.7%，大量 in-view 物体被标成"visually unconfirmed"，使可见性轨迹存在系统性低估；当前只从 out-of-view / occluded 抽锚点缓解了该问题，但对 detector 本身能力有依赖。
- **fixture 几何的静态假设**：数字孪生不记录抽屉/柜门的开合状态，导致 fixture-ambiguous 区间较多；真实场景的动态夹具未被建模。
- **仅厨房场景**：135个视频全部来自 HD-EPIC 厨房任务，泛化到其他场景（客厅、户外、工业）尚待验证。
- **仅多项选择题**：无法捕捉模型生成式推理的质量，开放问答更能暴露幻觉。
- **未来方向（作者建议）**：推动模型从"当前视图感知"走向"显式更新并长期维持动态场景的潜在表征"，即建立持久的、可被事件触发的 world-state memory。

## 研究启发与可借鉴点
1. **"几何验证可见性"的三阶段管线**（视场投影→几何遮挡→检测确认）可作为通用 pipeline 复用，为其他基于 HD-EPIC / Project Aria 的工作提供高质量的可见性标注替代方案。
2. **四级诊断式问题链设计**（Visibility → Temporal → Scene → 3D）值得迁移到其他时序/空间基准构建中，帮助精准归因模型失败。
3. **"视外地平线"作为难度假指标**：$h(o,T_q)$ 比视频总时长更直接刻画难度，未来评测可按 horizon 分层报告以消除长度混杂。
4. **干扰项按"时间距离档"与"场景 plausible 替代"分层构造**的方法论（near/medium/far/very-far 时间档、counter-area 细化、方向/距离类别严格均衡）可直接复用于新 VQA 基准的数据生成。
5. **Temporal cue ablation 揭示上游瓶颈**的思路：通过给模型注入中间事实（运动时间段、可见性声明）观察下游提升幅度，是一种低成本但信息量大的诊断工具，可推广到其他模型能力评测。

## 关键术语表
- **Out-of-sight spatiotemporal reasoning**：在物体离开当前视野后，仍能在时间和空间维度上维持并推理其状态的能力。
- **BEYOND3D**：本文提出的首个专门评测第一人称视频视外时空推理的VQA基准，含9,000问、8类问题。
- **Visibility track**：对每个动态物体按1fps推断的 $(\ell_o(t), v_o(t))$ 序列，标记可见/视外/遮挡/未确认/运动中状态。
- **Out-of-sight horizon $h(o,T_q)$**：从物体最后一次可见到查询时刻 $T_q$ 的时间间隔，是衡量问题难度的关键指标。
- **Fixture**：支撑或容纳物体的场景固定部件（如 counter、drawer、cupboard），用于 scene localization 的答案空间。
- **Geo-verified out-of-sight anchor**：经几何投影与检测双重验证、确实在查询时刻不可见且已移动的目标-时刻对。
- **Diagnostic question chain**：将任务拆成四个递进问题类型，用于逐层定位模型失败来源。
- **Last Frame Only vs. Video + Text**：两种输入设置对比揭示模型是依赖单帧可见性还是真正回溯历史状态。

## 可复现要素
- **数据集**：基于 HD-EPIC（公开），自行构建的 BEYOND3D 评测集1,000个锚点/9,000问题，论文 project page 提供下载链接（链接在原文中给出）。
- **代码**：论文标注了 § Code 入口（具体仓库见原文 project page）。
- **权重**：使用各模型官方公开 checkpoint，无微调。
- **关键超参**：1 fps 采样、448×448 分辨率、鱼眼有效区域 mask、遮挡判定 $\delta=10$ cm、检测匹配框扩展 20 px、检测阈值 50%、时间干扰档 ±1-2s/±3-4s/±5-6s/±7-30s。
- **硬件**：单卡 NVIDIA A100 80GB 或 RTX PRO 6000 96GB。
- **限制**：OWLv2 召回偏低（pipeline recall 62.7%）；厨房单一场景；仅多项选择题型。
