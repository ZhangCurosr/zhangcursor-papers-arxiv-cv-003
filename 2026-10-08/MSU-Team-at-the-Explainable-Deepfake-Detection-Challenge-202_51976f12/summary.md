---
title: "MSU-Team-at-the-Explainable-Deepfake-Detection-Challenge-202"
source: https://arxiv.org/pdf/2610.09952v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 22:50:33"
field: "可解释AI与深度伪造检测"
keywords: ["deepfake detection", "explainable AI", "contrastive learning", "vision-language models", "pseudo-mask supervision", "image forensics"]
innovations: ["解释驱动的伪掩码弱监督生成管线", "局部补丁级对比学习状态划分与pull/push规则", "类条件双VLM生成器防类别泄漏架构"]
benchmarks: ["XPlainVerse Challenge 2026", "Detection Accuracy", "Explanation Score", "Final Challenge Score"]
---

# 论文速读：MSU-Team-at-the-Explainable-Deepfake-Detection-Challenge-2026

## 一句话总结
本文提出了一个模块化可解释深度伪造检测系统，结合多骨干DINOv3+Mesorch检测器、解释驱动的伪掩码弱监督、局部补丁级对比学习，以及类条件Qwen3-VL生成器，在XPlainVerse挑战赛上取得0.9349检测准确率和0.7456综合得分。

## 研究问题与动机
1. **现有检测器缺乏可解释性**：生成模型使伪造图像日益逼真，但传统二分类检测器无法提供视觉证据支持预测，难以审计模型决策和理解失败案例。
2. **挑战赛要求双重输出**：Explainable Deepfake Detection Challenge 2026同时要求真实/伪造分类和基于可见取证线索的复杂/简单解释，现有方法难以兼顾。
3. **解释与检测的语义鸿沟**：数据集提供的文本解释描述了伪造痕迹，但如何将这些语言证据转化为检测器的空间监督信号尚未被充分探索。
4. **类条件幻觉问题**：若使用单一VLM生成解释，可能因类别泄漏而在真实图像上 hallucinate 伪造痕迹，需要分离的生成器避免此类错误。

## 核心贡献（创新点）
1. **解释驱动伪掩码生成管线**：将复杂解释中的局部伪造描述通过Grounding-DINO转换为弱补丁级监督掩码，首次将自然语言证据直接转化为检测器的空间正则化信号。
2. **统一 forensic 特征融合架构**：结合DINOv3（视觉表示）与Mesorch（DCT感知多尺度取证特征），通过Unified Forensic Feature Map实现跨模态空间对齐，本质区别于仅用单骨干的检测器。
3. **局部补丁级对比学习目标**：基于Artifact Evidence Map预测的置信度划分CF/UF/WF状态，设计pull/push/ignore规则组织补丁嵌入空间，无需配对图像或像素级操作掩码即可提升伪造证据的可分性。
4. **类条件双生成器VLM pipeline**：按检测预测路由选择G_fake或G_real生成复杂解释，再由GRPO优化的文本简化器S生成简单解释，从架构上杜绝类别泄漏导致的假阳性痕迹生成。

## 方法详解
### 4.1 伪掩码生成管线（离线）
- **词汇构建**：用Qwen3-VL-32B-Instruct从训练解释中提取短伪造短语，聚类去重得到词汇$\mathcal{V}=\{c_1,...,c_K\}$，划分为局部$\mathcal{V}_{loc}$和全局$\mathcal{V}_{glob}$子集。
- **图像级定位**：对每张假图$I_i$，模型接收图像、解释$e_i$和属性列表，输出可视觉定位的短语集合$\mathcal{A}_i=\{a_{i1},...,a_{in_i}\}$，每个短语$p_{ik}=g(a_{ik},I_i,e_i)$为具体视觉描述。
- **边界框生成**：Grounding DINO返回候选框$\mathcal{B}_i$，阈值过滤后合并为证据支持区域$U_i=\bigcup_{b\in\mathcal{B}_i}b$。
- **补丁目标**：对检测器网格$H_p\times W_p$，计算$t_{iuv}=|R_{uv}\cap U_i|/|R_{uv}|$，得到$T_i\in[0,1]^{H_p\times W_p}$，重叠框只计一次。

### 4.2 检测器架构
- **多骨干特征提取**：
  - 3个Transformer型DINOv3：全局嵌入$f_i^{(k)}$+密集特征图$H_i^{(k)}$
  - 3个CNN型DINOv3：同上
  - Mesorch：$H_i^{Mes}=F_{Mes}(I_i)\in\mathbb{R}^{H_p\times W_p\times D}$
- **DINO特征融合**：
  - $H_i^{DINO}=F_{map}^{DINO}(H_i^{(1)},...,H_i^{(K)})\in\mathbb{R}^{H_p\times W_p\times D}$
  - $f_i^{DINO}=F_{glob}^{DINO}(f_i^{(1)},...,f_i^{(K)})\in\mathbb{R}^{D}$
- **统一取证特征图（UFFM）**：
  - $U_i[u,v]=F_{UFFM}(H_i^{DINO}[u,v], H_i^{Mes}[u,v])$
- **图像级分类分支**：
  - 最大池化$u_i^{pool}=\max_{u,v}U_i[u,v]$（允许局部证据影响全局决策）
  - $g_i=[u_i^{pool};f_i^{DINO}]$，经MLP得$logit\ z_i$，$p_i=\sigma(z_i)$
  - 损失：Focal Loss($\alpha=0.5,\gamma=2.0$)
- **伪造证据图分支**：
  - $a_i[u,v]=h_{art}(U_i[u,v])$，$q_i[u,v]=\sigma(a_i[u,v])$
  - 损失：$\mathcal{L}_{art}=\frac{1}{H_pW_p}\sum\mathrm{BCE}(q_i[u,v],T_i[u,v])$
- **局部对比学习目标**：
  - 补丁匹配：$(u^*,v^*)=\arg\max\cos(U_a[u,v],U_b[u',v'])$，双向匹配
  - 状态划分（不确定性带$\delta$）：
    - $|q_i[u,v]-0.5|\leq\delta$→UF(假)/UR(真)
    - $q_i[u,v]>0.5+\delta$→CF(假)/CR(真)
    - $q_i[u,v]<0.5-\delta$→WF(假)/WR(真)
  - 匹配规则表（Tables 1-3）：CF-CF pull、CF-CR push、UF-UR pull等，含stop-gradient
  - 损失：$\mathcal{L}_{lcl}=\sum_{(a,b)}\ell_{ab}$，$\ell_{pull}=1-\cos(a,b)$，$\ell_{push}=\max(0,m-[1-\cos(a,b)])$
- **总检测损失**：$\mathcal{L}_{det}=\lambda_{img}\mathcal{L}_{img}+\lambda_{art}\mathcal{L}_{art}+\lambda_{lcl}\mathcal{L}_{lcl}$

### 4.3 类条件视觉语言解释生成
- **复杂解释模型**：$G_{fake}:I_i\mapsto\hat{e}_{i,fake}^c$，$G_{real}:I_i\mapsto\hat{e}_{i,real}^c$，LoRA适配全线性层，视觉塔冻结
- **简化模型**：$S:\hat{e}_i^c\mapsto\hat{e}_i^s$，纯文本操作，GRPO优化
- **推理流程**：检测器预测$\hat{y}_i$→路由到$G_{fake}$或$G_{real}$→$S$简化→输出$(\hat{y}_i,\hat{e}_i^c,\hat{e}_i^s)$

## 实验与结果
### 数据集
- **XPlainVerse**：百万级可解释深度伪造检测数据集，挑战赛子集76万图（27万真实+49万伪造）
- 公开划分：训练45万、验证11万、测试20万（含9万真实+11万伪造）

### 检测器消融（验证集AUC）
| 变体 | $\lambda_{img}$ | $\lambda_{art}$ | $\lambda_{lcl}$ | Val. AUC |
|------|----------------|-----------------|-----------------|----------|
| 仅图像级检测器 | 15 | 0 | 0 | 0.8975 |
| +伪掩码监督 | 15 | 1 | 0 | 0.9139 |
| +局部对比学习 | 15 | 1 | 1 | 0.9417 |

### 解释生成评估（内部验证集10K）
| 方法 | $B_c$ | $F_{ent}$ | $F_{evd}$ | $B_s$ | $L_s$ | $S_s$ | $M_{exp}$ |
|------|-------|-----------|-----------|-------|-------|-------|-----------|
| Zero-shot Qwen3-VL | 0.6023 | 0.5310 | 0.3934 | 0.4495 | 0.3418 | 0.4172 | 0.4812 |
| 组织者基线 | 0.7109 | 0.5536 | 0.4620 | 0.4578 | 0.4264 | 0.4484 | 0.5365 |
| 本文 | 0.7253 | 0.6235 | 0.5447 | 0.4991 | 0.9703 | 0.6405 | 0.6236 |

**关键提升**：归一化SLE从0.4264→0.9703（+127.7%），实体F1从0.5536→0.6235（+12.6%），证据F1从0.4620→0.5447（+17.9%）

### 最终测试集结果
- 检测准确率：**0.9349**
- 检测macro F1：**0.9340**（假F1=0.9418，真F1=0.9261）
- 复杂解释BERTScore-F1：0.7004
- 简单解释BERTScore-F1：0.6509
- 解释分数：**0.5571**
- **最终挑战赛得分：0.7456**

### 训练配置
- 检测器：10 epochs，batch size=4×16累积，lr=$5\times10^{-4}$，cosine decay，warmup=0.03
- 增强：随机旋转、翻转、裁切、JPEG压缩、噪声、亮度/颜色扰动
- VLM：LoRA rank=16, alpha=32，4 epochs，lr=$2\times10^{-4}$，batch=128
- GRPO：1 epoch，lr=$10^{-4}$，batch=32，每prompt采样32生成，$\beta=0.01$

## 相关工作脉络
1. **GenImage/WildFake/Community Forensics**：大规模合成图像检测数据集，但缺乏解释标注，仅支持二分类任务，无法用于可解释检测。
2. **MultiFakeVerse**：86K真实+758K人 centric操作数据集，覆盖人物/物体/场景/动作编辑，与XPlainVerse场景更接近，但仍缺文本解释。
3. **DD-VQA/FakeClue**：将检测reformulate为VQA或引入自然语言解释，但DD-VQA基于FaceForensics++仅限人脸，FakeVLM的100K数据规模远小于XPlainVerse。
4. **DRCT**：图像级对比学习分离真实/生成/扩散重建样本，本文受其启发但扩展到**补丁级**对比，无需配对图像。
5. **HiDA-Net**：强调空间敏感性和局部编辑检测，本文与其共享"局部预测迫使关注细粒度取证细节"的理念，但通过伪掩码引入语义-grounded监督。
6. **FakeVLM/ForenDeX/FakeXplainer**：均使用VLM进行可解释检测，但FakeVLM微调视觉塔，ForenDeX注入取证特征，FakeXplainer联合SFT+GRPO；本文选择**分离架构**（专用检测器+类条件VLM）避免类别泄漏。

## 局限性与未来方向
1. **计算成本高**：多DINOv3骨干+Mesorch+多个Qwen3-VL模型，适合挑战赛但不适合实际部署。
2. **VLM仅作"附件"**：检测器独立决策，VLM不参与真实性判断或假设比较，未充分利用多模态推理潜力。
3. **GRPO优化未必提升实用性**：虽改善normalized SLE，但不保证简单解释对普通用户更易理解。
4. **伪掩码不完备**：基于Grounding-DINO的弱监督可能遗漏部分伪造区域，且依赖词汇构建质量。
5. **未来方向**：探索VLM参与的端到端可解释检测、改进简化器的人类可用性评估、降低多骨干计算开销。

## 研究启发与可借鉴点
1. **弱监督信号转换策略**：将自然语言解释转化为空间伪掩码的思路可迁移至其他多模态 grounding 任务，如医学图像报告引导的区域定位。
2. **补丁级对比学习的状态划分设计**：CF/UF/WF三态分类及pull/push/ignore规则值得借鉴，尤其适用于无配对标注的异常检测领域。
3. **类条件生成器防泄漏架构**：按预测类别路由到专用VLM的策略可推广至任何存在类别混淆风险的多模态生成任务。
4. **Max Pooling融合局部证据**：对全局决策的影响机制简洁有效，可在其他需要"局部证据驱动全局判断"的视觉任务中复用。
5. **挑战赛解耦设计哲学**：检测与解释分离、简化器独立GRPO优化，为多目标竞赛提供工程化参考范式。

## 关键术语表
- **XPlainVerse**：百万级可解释深度伪造检测数据集，含真实/伪造图像配对复杂和简单自然语言解释。
- **Artifact Evidence Map**：检测器输出的补丁级伪造证据图，估计每个补丁包含伪造线索的概率。
- **Grounding-DINO**：开放词汇目标检测模型，将文本短语与图像区域对齐，用于伪掩码生成。
- **Mesorch**：多尺度混合架构取证骨干，提供DCT感知、多分辨率压缩痕迹特征。
- **Local Contrastive Learning (LCL)**：补丁级对比目标，通过pull/push规则组织伪造/真实证据嵌入空间。
- **Qwen3-VL**：阿里巴巴开源的多模态大模型，本文使用8B和32B版本分别作为生成器和词汇构建器。
- **GRPO (Group Relative Policy Optimization)**：强化学习优化方法，用于简化器直接优化官方评估指标。
- **Focal Loss**：处理类别不平衡的分类损失，$\mathcal{L}=-\alpha_t(1-p_t)^\gamma\log(p_t)$，本文$\alpha=0.5,\gamma=2.0$。

## 可复现要素
- **数据集**：XPlainVerse挑战赛子集，760K图像（270K真实+490K伪造），论文未声明完全开源，但提供划分统计。
- **代码**：论文未提供开源代码链接。
- **权重**：未公开提供预训练权重。
- **关键超参**：检测器lr=$5\times10^{-4}$，$\lambda_{img}=15,\lambda_{art}=1,\lambda_{lcl}=1$；VLM lr=$2\times10^{-4}$，LoRA rank=16 alpha=32；GRPO lr=$10^{-4}$，$\beta=0.01$，采样32。
- **硬件**：使用MSU-270超算。
