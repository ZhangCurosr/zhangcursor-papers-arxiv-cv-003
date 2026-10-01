---
title: "HyperDAM-Hyperspectral-Distractor-Aware-Memory-with-Amodal-E"
source: https://arxiv.org/pdf/2609.34396v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 04:07:22"
field: "高光谱单目标跟踪"
keywords: ["hyperspectral tracking", "SAM 3", "distractor-aware memory", "amodal tracking", "video object segmentation"]
innovations: ["无监督 HSI 记忆门控拒绝光谱不一致更新", "外挂因果 3D 扩压头实现 modal-to-amodal 外推", "HOTC2026-Modal 密集监督标注扩展"]
benchmarks: ["HOTC 2026 Public-LB75", "HOTC 2026 Private Leaderboard"]
---

# 论文速读：HyperDAM-Hyperspectral-Distractor-Aware-Memory-with-Amodal-E

## 一句话总结
本文提出 HyperDAM，一个基于 DAM4SAM3 的高光谱单目标跟踪系统，通过高光谱记忆门控拒绝干扰物污染、因果时空扩压器恢复遮挡目标幅度边界、以及 RTS 平滑补全完全遮挡帧，最终在 HOTC 2026 竞赛中获得第二名（公开榜 AUC 最佳）。

## 研究问题与动机
1. **记忆污染问题**：现有高光谱跟踪器记忆更新主要依赖空间与外观证据，错误的高置信度掩码可能进入记忆库，污染后续预测。
2. **Modal-Amodal Gap**：SAM 系列预测可见（modal）掩码，而 HOTC 等基准要求评估幅度（amodal）边界框，遮挡时二者存在显著差异。
3. **完全遮挡处理**：SAM 在完全遮挡时输出空掩码，传统方法缺乏有效的轨迹补全机制。
4. **跨域泛化优先**：竞赛场景下应避免针对榜单过拟合，需追求跨域鲁棒性。

## 核心贡献（创新点）
1. **HSI 记忆门控**：基于帧零参考光谱的无监督一致性检验，可在 DRM 更新边界拒绝光谱不一致的候选，本质区别于特征注入式方法，不改变当前预测也不修改基础跟踪器权重。
2. **幅度扩压器**：外挂 9.4M 参数的因果六帧 3D 头，从 SAM3 掩码解码器特征预测非负向外残差，严格保持"只扩不缩"合同，区别于直接训练 amodal segmentation 的方法。
3. **HOTC2026-Modal 标注扩展**：为全部 481 个视频补充人工校验的 modal 掩码与 mask-tight box，填补了密集监督信号空白，区别于引入新基准的方式。
4. **静态场景恢复 + RTS 平滑**：通过角点追踪分类静态/动态场景，并结合离线 RTS 平滑补全空掩码帧，属于系统级辅助组件而非核心创新。

## 方法详解
**基线**：冻结 DAM4SAM3（基于 SAM3），使用官方第一帧 box 初始化，PVS-only 模式运行。

**HSI 门控（§4.2）**：
- 输入：候选更新掩码 $M_t$ 及其膨胀区域 $E_t = \text{Dilate}(M_t) \setminus M_t$
- 光谱解码：$X_t = \mathcal{D}(I_t^{\text{raw}})$，VIS/NIR/RedNIR 分别对应 B=16/25/15 波段
- 参考光谱：$p_0 = \text{median}_{x \in M_0} X_0(x)$（帧零目标）
- 相似度计算：$s_t^M = \cos(\hat{p}_t^M, \hat{p}_0)$，$s_t^E = \cos(\hat{p}_t^E, \hat{p}_0)$
- 接受条件：$A_t^{\text{HSI}} = \mathbb{1}[s_t^M \geq \tau_{\text{id}} \land s_t^M - s_t^E \geq \tau_{\text{exp}}]$
- 特性：仅能拒绝请求，不能发起更新；阈值由帧零标定；不可靠时弃权

**幅度扩压器（§4.3）**：
- 从 SAM3 掩码解码器提取 32 通道特征，裁剪 $32 \times 24 \times 24$ 对齐 box，堆叠最近 6 帧
- 输入还包括：对齐 mask logits、256 维 object query、box 几何、帧零 anchor
- 6 个残差 3D Conv 块聚合时空信息，4 个头预测 $\Delta_t = (\delta_L, \delta_T, \delta_R, \delta_B)$
- Amodal box：$x - \delta_L, y - \delta_T, w + \delta_L + \delta_R, h + \delta_T + \delta_B$
- 训练目标：冻结 SAM3，仅训练扩压头；侧面仅在有真实遮挡时正监督；预训练于合成 MOVi-MC-AC 数据
- 完全遮挡帧排除在训练外

**静态/动态分类（§4.5）**：
- 前 10 帧使用 Shi-Tomasi 角点 + 金字塔 Lucas-Kanade 光流估计背景运动
- 静态场景中突发的空间不对称 box 增长视为跟踪错误，触发恢复

## 实验与结果
**数据集**：HOTC 2026，406 训练视频 + 75 视频 Public-LB75（26,860 帧），含 VIS/NIR/RedNIR 三种高光谱模态。

**消融实验（Table 1，Public-LB75）**：
| 配置 | Public AUC |
|------|-----------|
| B0: DAM4SAM3 native | 0.69039 |
| B0 + HSI gate | 0.69214 |
| B1 + static-scene recovery | 0.69513 |
| B2 + amodal expander | 0.69732 |
| B3 + RTS smoothing | 0.69768 |

**基线对比（Table 2）**：HyperDAM 以 0.69768 AUC 位居第一，超越 SAM3（0.67393）和 DAM4SAM3（0.69039）；峰值 VRAM 4685 MiB，参数量 836.73M。

**竞赛成绩**： organizer 私有榜 AUC 68.0093%，DP@20 87.7703%，排名第二。

## 相关工作脉络
1. **Hyperspectral tracking（[1]-[15]）**：从材料描述子到学习式光谱-空间表征，本文继承其光谱价值验证思路，但聚焦记忆更新决策而非端到端特征学习。
2. **SAM/SAM2/SAM3 系列（[21]-[23]）**：基础分割-跟踪范式，本文基于 DAM4SAM3 [24] 扩展，区别于直接微调冻结模型的工作（如 [31]-[34]）。
3. **Amodal tracking（[41]-[44]）**：TAO-Amodal 建立 benchmark，Amodal SAM 扩展分割能力，本文聚焦 box 级外推且保持 SAM 冻结。
4. **WHISPERS 高光谱跟踪工作**：PGSR-Track [39]、SAM2Local [40] 等采用光谱提示或检测驱动，本文的 HSI 门控为轻量无监督方案，不注入学习特征。
5. **LOMO/MOVi-MC-AC 合成数据（[43]）**：用于扩压头预训练，本文首次将合成 amodal 数据迁移至真实高光谱跟踪任务。

## 局限性与未来方向
1. HSI 门控增益有限，因其仅能拒绝请求而不能主动发起更新；未来可探索基于检测器的多候选光谱原型关联。
2. 数据集规模小且多样性不足，LoRA 适配未带来泛化提升，限制了 learned 组件的潜力释放。
3. 静态场景分类依赖前 10 帧，对动态场景中的静止目标可能存在误判。
4. 完全遮挡恢复依赖离线 RTS，无法处理长时遮挡后的目标重识别。

## 研究启发与可借鉴点
1. **无监督门控设计**：HSI 门控的"只拒不启"约束是一种简洁的记忆安全策略，可迁移至其他 foundation model tracker 的防污染场景。
2. **Modal-Amodal 外推范式**：冻结预训练模型 + 外挂轻量扩压头的思路，避免了从头训练 amodal segmentation 的高成本，适合资源受限场景。
3. **合成到真实迁移**：在 MOVi-MC-AC 上预训练扩压头后直接部署到真实高光谱数据，验证了合成 amodal 数据的跨域有效性。
4. **离线平滑补充在线缺失**：RTS 平滑仅作用于空掩码帧且不影响观测帧，可作为通用后处理模块接入现有跟踪系统。
5. **竞赛导向的跨域优先**：明确以跨域鲁棒性替代榜单优化，避免了 overfitting 风险，值得后续 benchmark 研究借鉴。

## 关键术语表
**HSI Gate**：基于高光谱影像的光谱一致性门控，通过余弦相似度检验候选掩码与帧零参考的光谱相似性，拒绝干扰物写入记忆。

**DAM4SAM3**：基于 SAM3 的 distractor-aware memory 跟踪框架，维护近期外观记忆与干扰物分辨记忆（DRM）。

**Amodal Expander**：外挂于 SAM3 掩码解码器的 3D 卷积头，从 6 帧时空特征预测向外 box 残差，将 modal box 扩展至 amodal  extent。

**Modal Mask**：目标可见部分的二值掩码，与 amodal mask（含被遮挡部分）相对。

**RTS Smoothing**：Rauch-Tung-Striebel 平滑，一种离线 smoother，利用前后观测重构完全遮挡帧的目标 box。

**Public-LB75**：HOTC 2026 组织者提供的 75 视频公开榜单评估集，ground truth  withheld，仅返回 AUC 分数。

**DRM (Distractor-Resolving Memory)**：DAM4SAM3 中专门存储干扰物表征的记忆模块，用于避免目标被相似外观的 distractor 劫持。

**PVS-only Mode**：SAM3 的 Promptable Video Segmentation 模式，禁用检测器推理，仅依赖视觉 prompt 进行跟踪。

## 可复现要素
- **数据集**：HOTC 2026（406 训练视频 + 75 视频 Public-LB75），高光谱影像开源；HOTC2026-Modal 标注已发布至 Hugging Face（https://huggingface.co/datasets/ryo818/HOTC2026-Modal）
- **代码**：已开源（https://github.com/RyogaYuzawa/hotc2026-hyperdam）
- **权重**：DAM4SAM3 使用 SAM3-TrackBench 提供版本（revision d37e4a975e48），扩压头 9.4M 参数预训练于 MOVi-MC-AC
- **关键超参**：HSI 门控阈值 $\tau_{\text{id}}, \tau_{\text{exp}}$ 由帧零标定；扩压头输入 6 帧序列，特征 crop 尺寸 $24 \times 24$，波段数 B=16/25/15
- **评测硬件**：单卡 NVIDIA L4 GPU，FP32
