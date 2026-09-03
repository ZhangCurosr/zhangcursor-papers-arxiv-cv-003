---
title: "YesTrack-Referring-Multi-Object-Tracking-via-MLLM-based-Yes"
source: https://arxiv.org/pdf/2609.02318v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:13:40"
field: "多模态视频理解与跟踪"
keywords: ["referring multi-object tracking", "multimodal large language models", "discriminative paradigm", "temporal consistency", "binary verification"]
innovations: ["将MLLM从生成caption转为直接判别Yes/No logit，消除自回归开销", "提出TCP（保守时序置信度先验）与TRP（降频参照传播）两个轻量时序一致性约束"]
benchmarks: ["Refer-KITTI", "Refer-KITTI-V2", "KITTI tracking (car/person)"]
---

# 论文速读：YesTrack-Referring-Multi-Object-Tracking-via-MLLM-based-Yes

## 一句话总结
论文提出 YesTrack，将 MLLM 从"自动生成描述 → 额外模块做决策"的生成范式，改为直接对候选目标做 **Yes/No 判别**，配合两个轻量时序一致性约束（TCP、TRP），在 Refer-KITTI 上以最小的 Qwen3-VL-2B 取得 HOTA 54.00 的 SOTA；该判别范式进一步推广为 YesTrack-MOT，用于通用 MOT 的数据关联也取得显著收益。

## 研究问题与动机
- **现有 RMOT 滥用 MLLM 生成能力**：主流做法（如 ReferGPT [6]）把 MLLM 当作 caption 生成器，再依赖外部文本相似度模块完成最终指代决策，链路冗长且引入额外延迟。
- **自回归解码成为实时跟踪瓶颈**：逐 token 生成大量文本造成显著推理开销，难以满足跟踪所需的低延迟要求。
- **MLLM 的视觉–语言对齐能力未被充分利用**：现有工作仅取其生成端，未直接利用其判别侧的 logits 信息。
- **端到端方法泛化受限**：TransRMOT、DKG-Track、TenRMOT 等依赖特定词汇空间与任务模块，表达多样性弱（Refer-KITTI 仅 49 词、Refer-KITTI-V2 扩展到 617 词，差异显著），在复杂/噪声表达下鲁棒性不足。

## 核心贡献（创新点）
1. **判别式 MLLM 指代范式**：将 referring 重构为二值图文匹配问题，只取 Yes/No token 的 logits 做 softmax 得到连续置信度，避免自回归，每个候选仅需一次前向。与 ReferGPT 等生成式方法的本质区别在于决策由离散生成改为直接判别。
2. **TCP（Temporal Confidence Prior）**：用历史 K 帧最低置信度作保守先验，仅在身份连续 K 帧都高置信时才偏移当前预测，显著提升遮挡/模糊下的稳定性。与简单滑动平均式平滑的本质区别在于“全过去窗内都高才偏”的严格时序一致判定。
3. **TRP（Temporal Reference Propagation）**：按固定间隔 ∆ 或新身份出现时触发 MLLM 验证，非关键帧直接继承最近关键帧的 referring 分数，将 MLLM 调用次数降至 1/∆，以微小精度代价换取大幅加速。与逐帧验证范式的本质区别在于显式利用"语义指代比轨迹更稳定"这一观察。
4. **YesTrack-MOT 通用 MOT 扩展**：将同一判别逻辑移植到数据关联，用 MLLM 替换传统 ReID embedding 做 image–image 二值验证，配合距离门控 + Hungarian 构成极简 MOT 流水线。与强运动/外观建模方案（BoT-SORT、OC-SORT）的本质区别在于完全依赖 MLLM 判别信号而非手工特征。

## 方法详解
- **框架（两阶段）**：先用现成 tracker 抽每帧候选 crop、ID 与归一化 bbox；按 TRP 将帧分为 key / non-key。关键帧走完整判别，非关键帧直接继承历史 referring 分数。
- **输入格式**：crop 图像 + 归一化空间坐标 + 指代表达 E，坐标显式嵌入文本 prompt 提供位置线索。强制模型输出 Yes/No 二选一以收紧输出空间。
- **二值概率提取（公式 1）**：解码/投影后直接取 $\ell_i^{\mathrm{yes}}, \ell_i^{\mathrm{no}}$，$p_i = \exp(\ell_i^{\mathrm{yes}})/(\exp(\ell_i^{\mathrm{yes}})+\exp(\ell_i^{\mathrm{no}}))$，避免自由文本的不稳定性，同时保留连续置信度。
- **训练损失（公式 2）**：标准 BCE $\mathcal{L}_i = -[y_i \log p_i + (1-y_i)\log(1-p_i)]$，softmax 限定在两个决策 token 上。
- **Stage 1（frame mode）**：单帧两路判别得 $p_i$；按 $[p_l, p_h]=[0.2, 0.8]$ 路由——高于/低于阈值直接保留/丢弃，区间内进入 Stage 2。
- **Stage 2（video mode）**：从 memory bank 召回前 K=4 帧 crop+ 分数，MLLM 联合评估 temporal context 后以阈值 $\gamma=0.4$ 给出终判。
- **TCP（公式 3）**：$\tilde{p}_i^{(t)} = \min\{1, p_i^{(t)} + \lambda \cdot \mathbb{I}(\min_{k \in [t-K,t-1]} p_i^{(k)} \ge \alpha)\}$，$\lambda=0.3, \alpha=0.4$，外层 clip 保证概率合法性，内层 min 保证"过去 K 帧全高才偏"。
- **TRP 触发规则**：固定间隔 ∆（Refer-KITTI 取 5、Refer-KITTI-V2 取 10）、新身份出现、表达变化时重触发；非关键帧只跑 tracker、referring 分数直接继承。
- **YesTrack-MOT 关联（公式 4–6）**：距离门控 $g(\tau_j,d_k)=\mathbb{I}(\|c(\tau_j)-c(d_k)\|_2 \le \delta), \delta=200$；image–image 验证得 $p_{jk}$，构造代价 $c_{jk}=1-p_{jk}$，Hungarian 求解最优指派；丢失 track 保留 L=10 帧用于短暂重关联。

## 实验与结果
- **数据集**：Refer-KITTI（818 条表达、215 不同、49 词表，21 序列中取 18）、Refer-KITTI-V2（9758 条、7193 不同、617 词表，含模糊/无目标 case）。
- **主结果 Refer-KITTI（Table 1）**：YesTrack (TempRMOT*) 取得 **HOTA 54.00、DetA 43.91、AssA 66.57**，整体 SOTA；YesTrack-MOT 取得 **DetA 46.84、DetRe 62.72**；均超最强端到端 CDRMT (HOTA 49.35)、DKG-Track (52.08)。
- **主结果 Refer-KITTI-V2（Table 2）**：YesTrack-MOT **HOTA 43.75、DetA 37.04、DetRe 48.78** SOTA；TempRMOT* 变体 HOTA 41.78 仍大幅领先。
- **MOT 对比（Table 5，相同 RF-DETR 检测）**：YesTrack-MOT **HOTA 45.10、DetA 40.96、DetRe 47.59**，显著优于 ByteTrack (40.83)、OC-SORT (41.34)。
- **消融（Table 3）**：仅 frame mode 即 HOTA 52.64；TCP 贡献最大 HOTA/DetA；TRP 单独加速至 22min（最省）；TCP+TRP 取 AssA 最优 66.71 且仅 25min。
- **不同 base tracker 通用性（Table 4）**：YesTrack 在 ByteTrack/TempRMOT*/YesTrack-MOT/GT 四类 tracker 上 referential metrics 均超过 iKUN (SOTA 两阶段基线)。
- **噪声表达鲁棒性（Fig.4）**：面对拼写错误/口语化表达（如 "emmm.... black car is in the left side"），YesTrack 明显优于 DKG-Track 与 ReferGPT。

## 相关工作脉络
- **TransRMOT [38]**：首提 RMOT 任务与 Refer-KITTI，端到端 joint formulation；本文取其作为 two-stage 参考骨架但抛弃其生成式指代模块。
- **TempRMOT [44]**：引入 Refer-KITTI-V2 与 temporal enhancement；本文剥离其语言组件得到 TempRMOT* 作为公平纯 MOT 对比。
- **DKG-Track [18]**：分解 static/motion 线索做细粒度对齐的端到端方法；本文与之对比验证判别范式在复杂表达上更鲁棒。
- **iKUN [9]**：两阶段 SOTA，Knowledge Unification Module + Neural Kalman；本文在 referential metrics 上全面超越，体现 MLLM 判别 vs. 知识统一模块的范式差异。
- **ReferGPT [6]**：首开 MLLM+RMOT 的生成路线；本文核心对比对象，证明"直接判别 > 生成再匹配"。
- **CLIP-SCGI [11]、LVLM-ReID [35]、LLaVA-ReID [22]**：将 MLLM 用于 ReID 均以生成/属性提取为主；本文把 MLLM 角色从"描述器"转为"二值裁判"，路径迥异。
- **ByteTrack [45]、BoT-SORT [1]、OC-SORT [5]**：MOT 强基线；YesTrack-MOT 以极简结构实现更强 DetA/DetRe，证明 MLLM 判别可作为通用数据关联组件。

## 局限性与未来方向
- **两阶段设计依赖现成 tracker 质量**：tracker 误差会直接传导至 referring 阶段，无法自修复。
- **TRP 关键帧错误会传播**：错误 MLLM 决策要到下一触发间隔才修正，期间可能出现临时丢目标或误跟干扰物。
- **TCP 是启发式而非可学习**：手动设定 λ、α、K，未与 MLLM 联合优化，难以自适应不同场景动态。
- **未来方向**：端到端紧耦合框架、将 TCP 替换为可学习时序模块、扩展至更复杂的视频理解任务。

## 研究启发与可借鉴点
1. **判别式 MLLM 复用范式可迁移**：不仅 RMOT/MOT，凡需"候选 × 查询"二值判定的任务（如 VQA 式筛选、跨模态检索过滤）均可套用 logit-only 提取思路，避免生成开销。
2. **TRP 思想可迁移到任何两阶段 pipeline**：凡"语义相关度比底层状态更稳定"的场景（如 video grounding、dialogue state tracking）都可引入 key/non-key 触发机制降频验证。
3. **TCP 的保守先验机制（全窗内最小才偏）**：适用于任何需防短时跳变的时序置信度校准，可替代简单的 EMA 平滑并降低过拟合短期噪声风险。
4. **与团队结合点**：若本团队涉及多模态视频理解/导航中的目标指代，可把 YesTrack 判别头嵌入现有两阶段系统作 plug-in，预期在低资源/噪声表达下带来显著提升。

## 关键术语表
- **Referring Multi-Object Tracking (RMOT)**：在视频序列中，依据自然语言指代表达持续定位并关联匹配的目标轨迹。
- **MLLM (Multimodal Large Language Model)**：融合视觉与语言的大规模预训练语言模型，如本文所用的 Qwen3-VL。
- **Temporal Confidence Prior (TCP)**：利用过去 K 帧最低置信度作保守先验，仅在连续高置信时才偏移当前预测的时序正则项。
- **Temporal Reference Propagation (TRP)**：按触发条件只在关键帧调用 MLLM 验证、非关键帧继承最近分数的降频推理机制。
- **HOTA / DetA / AssA**：Higher Order Tracking Accuracy 及其分解的检测精度 DetA 与关联精度 AssA，RMOT 主指标。
- **Refer-KITTI / Refer-KITTI-V2**：首版与升级版 RMOT 基准，词表从 49 扩充至 617，表达式复杂度与语义多样性大幅提升。
- **Binary referential metrics (Acc / Prec / Recall)**：以 GT tracklets 为候选、独立于 tracker 方差的语言指代识别指标。
- **YesTrack-MOT**：把同一判别范式移植到通用 MOT 数据关联的极简实现（距离门控 + MLLM 验证 + Hungarian）。

## 可复现要素
- **代码**：已开源，https://github.com/ggbondrighthere24/YesTrack。
- **数据集**：Refer-KITTI、Refer-KITTI-V2 公开；KITTI tracking split 同 Refer-KITTI-V2 验证集。
- **MLLM**：Qwen3-VL-2B-Instruct；crop 分辨率 320×320。
- **关键超参**：$p_l=0.2, p_h=0.8, \gamma=0.4, K=3, \alpha=0.4, \lambda=0.3, \Delta=\{5,10\},$ memory bank 大小 4, 距离阈值 200 px, lost 保留 10 帧。
- **硬件**：2×NVIDIA RTX 4090 (24GB)。
- **损失**：BCE，softmax 限定于 Yes/No 两 token。
