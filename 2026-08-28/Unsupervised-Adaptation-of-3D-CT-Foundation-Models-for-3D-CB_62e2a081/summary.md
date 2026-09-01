---
title: "Unsupervised-Adaptation-of-3D-CT-Foundation-Models-for-3D-CB"
source: https://arxiv.org/pdf/2608.27190v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 00:54:16"
field: "医学图像分割与域适应"
keywords: ["Unsupervised Domain Adaptation", "CBCT Segmentation", "Foundation Models", "3D Medical Imaging", "Feature Alignment", "Redundancy Reduction", "Barlow Twins"]
innovations: ["提出基于冗余降低的无监督特征对齐框架，无需adversary预测头", "架构无关设计，同时适配CNN和ViT基础模型，显著提升CT到CBCT迁移性能", "公开Pancreatic-3D CBCT肝脏分割标注，促进公平基准研究"]
benchmarks: ["Pancreatic-CT-CBCT-SEG (DR)", "LiTS + Private DI dataset"]
---

# 论文速读：Unsupervised-Adaptation-of-3D-CT-Foundation-Models-for-3D-CB

## 一句话总结
本文提出了一种无监督域适应框架，通过冗余降低的特征对齐策略，将预训练的3D CT基础模型迁移到3D CBCT肝脏分割任务，无需目标域标注数据且推理时无需额外适配。

## 研究问题与动机
- CBCT（锥形束CT）在介入治疗和放射治疗中应用广泛，但其图像质量受限于视野、散射、束硬化效应及碘对比剂导致的高强度区域，与诊断CT存在显著模态差异
- 现有3D CT基础模型直接零样本迁移到CBCT时性能严重下降，因为默认预处理（如min-max归一化）无法处理CBCT中的高强度异常值
- 获取CBCT标注成本极高，全监督微调不可行；已有生成式域适应方法（如CycleGAN、SIFA-3D）假设源/目标域视野相似，不适用于CT-CBCT场景
- 需要一种轻量级、架构无关、无需目标域标注的域适应方法

## 核心贡献（创新点）
- 提出3D特征级无监督域适应框架，将预训练基础模型表示对齐到CT-CBCT分割任务，同时支持CNN和ViT架构
- 设计基于冗余降低的对齐策略，通过Barlow Twins机制实现源域与目标域特征的无监督对齐，无需 adversary 的任务预测头
- 在两个互补的CT-CBCT肝脏分割基准（介入血管和放射治疗场景）上验证，超越现有零样本基础模型和无监督域适应方法
- 公开Pancreatic-3D CBCT数据集的肝脏分割标注，促进后续公平对比研究

## 方法详解
- **网络分解**：将预训练网络$h(x) = g(f(\psi(x)))$分解为特征提取器$\psi$、表示头$f$、任务预测头$g$，以及对抗头$f'$（初始化为$f$的副本）
- **源域标签监督**：用任务损失$\mathcal{L}_{task}$训练$f$和$g$，使预测在源域CT上准确
- **对抗特征对齐**：引入两个损失函数：
  - 对齐损失$\mathcal{L}_{align}$：促使源域和目标域的$f(z)$与$f'(z)$在对应特征维度完全相关（主对角线元素趋近1）
  - 分离损失$\mathcal{L}_{sep}$：促使目标域的$f(z)$与$f'(z)$完全不相关（所有相关系数趋近0）
- **冗余降低机制**：基于Barlow Twins，对批内特征进行中心化、L2归一化后计算交叉相关矩阵$\mathcal{C}$，$\mathcal{L}_{align} = \sum(1-\mathcal{C}[i,i])^2 + \frac{1}{D}\sum_{i\neq j}\mathcal{C}[i,j]^2$，$\mathcal{L}_{sep} = \sum\mathcal{C}[i,i]^2 + \frac{1}{D}\sum_{i\neq j}\mathcal{C}[i,j]^2$
- **优化策略**：交替优化源域任务损失、对抗头的对齐+分离损失、特征提取器的任务损失+对齐损失
- **推理阶段**：丢弃对抗头$f'$，仅使用原始路径$\psi \rightarrow f \rightarrow g$

## 实验与结果
- **数据集**：
  - DR数据集（放射治疗CBCT）：130个CT（LiTS）+ 39个CBCT（Pancreatic-CT-CBCT-SEG）
  - DI数据集（介入CBCT）：678个CT + 573个CBCT
- **评估指标**：F1分数
- **最强结果**：
  - nnUNet + Ours：DR 78.0%，DI 86.0%（新增参数仅0.08M）
  - VISTA-3D + Ours（10点prompt）：DR 90.0%，DI 81.5%（新增参数0.99M）
  - SAM-Med3D + Ours（10点prompt）：DR 77.6%，DI 74.0%
- **提升幅度**：相比Source Only nnUNet（DR 66.5%，DI 80.1%），nnUNet+Ours分别提升11.5和5.9个百分点；VISTA-3D从DI 48.9%提升至81.5%
- **鲁棒性**：超参数$\alpha$和$\gamma$在宽范围内性能稳定

## 相关工作脉络
- **DA-nnUNet** [4]：基于对抗判别的特征对齐，本文方法无需adversary预测头，更轻量
- **MDD-UNet** [5]：基于margin disparity discrepancy，使用任务特定损失（如交叉熵），本文方法与任务无关
- **SIFA-3D** [6]：联合图像-特征对齐，依赖GAN且假设源/目标视野相似，不适用于CT-CBCT
- **MAPSeg** [7]：自训练方法，依赖masked autoencoder预训练和伪标签，对初始化敏感
- **VISTA-3D** [13]、**SAM-Med3D** [11]、**MedSAM2** [10]：3D医学基础模型，直接零样本迁移到CBCT性能不佳
- **Barlow Twins** [16]：自监督学习的冗余降低机制，本文将其引入无监督域适应

## 局限性与未来方向
- 仅在肝脏分割任务上验证，虽声称可扩展到多器官任务但未充分验证
- 对CT-CBCT的特定模态差异（如视野不一致、高强度区域）有隐含假设
- 训练阶段需源域标注数据，仅适用于有CT标注的场景
- 未来可探索其他干预成像应用（去噪、姿态估计）及泛化到其他模态迁移

## 研究启发与可借鉴点
- 冗余降低的Barlow Twins机制可用于无监督域适应的特征对齐，避免引入额外的对抗判别头
- 网络分解策略（$\psi \rightarrow f \rightarrow g$）使预训练基础模型适配更灵活，不影响原始预测头
- 架构无关设计使方法可同时适配CNN（如nnUNet、VISTA-3D）和ViT（如SAM-Med3D）
- 公开标注数据集促进公平对比，值得借鉴的数据共享实践

## 关键术语表
**CBCT（Cone-Beam CT）**：锥形束计算机断层扫描，常用于介入治疗和放射治疗的实时影像引导
**UDA（Unsupervised Domain Adaptation）**：无监督域适应，指在无目标域标注情况下将源域模型迁移到目标域
**Foundation Models**：基础模型，指在大规模数据上预训练的通用模型，可进行零样本或少样本迁移
**Barlow Twins**：自监督学习框架，通过最小化特征表示的冗余（交叉相关矩阵趋近单位矩阵）学习不变表示
**Redundancy Reduction**：冗余降低，指最小化特征维度间的相关性以提高表示独立性
**IoF（Interventional CBCT）**：介入型锥形束CT，使用碘对比剂的血管成像，存在高强度区域
**DR（Radiation Therapy CBCT）**：放射治疗型CBCT，用于放疗定位和剂量验证

## 可复现要素
- **数据集**：Pancreatic-CT-CBCT-SEG公开数据集，肝脏分割标注由作者生成并公开；LiTS公开数据集
- **代码**：开源，位于作者仓库（论文提供了链接）
- **权重**：提供预训练模型权重和适配后权重
- **关键超参**：$\alpha$、$\gamma$（冗余降低损失的权重），论文显示在宽范围内性能稳定
- **分辨率**：1.8mm各向同性体素间距
