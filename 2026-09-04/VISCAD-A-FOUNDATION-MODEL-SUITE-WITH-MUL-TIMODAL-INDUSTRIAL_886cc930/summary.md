---
title: "VISCAD-A-FOUNDATION-MODEL-SUITE-WITH-MUL-TIMODAL-INDUSTRIAL"
source: https://arxiv.org/pdf/2609.03811v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 18:43:10"
field: "工业AI设计"
keywords: ["工业CAD", "多模态基础模型", "程序生成", "装配设计", "测试时缩放"]
innovations: ["27B多模态零件生成器VisCAD-M1统一覆盖文本/图纸/照片/渲染图", "并行测试时缩放(parallel-tts)使模型自rerank提升约5%相对SOTA", "DSL-无关中间表示IR支持跨FreeCAD/SolidWorks/Fusion的装配迁移"]
benchmarks: ["PubCADBench", "RealCADBench"]
---

# 论文速读：VISCAD: A FOUNDATION MODEL SUITE WITH MULTI-MODAL INDUSTRIAL CAD INTELLIGENCE

## 一句话总结
VisCAD 是一个面向工业产品的多模态 CAD 基础模型套件，核心模型 VisCAD-M1（27B 参数）可将文本、工程图纸、真实产品照片、渲染图等多种设计意图映射为可执行的 FreeCAD Python 程序；配套装配级 harness VisCAD-H1 以 DSL-无关中间表示（IR）支持复杂装配体生成，整体在零件级和装配级评测上均超越当前最前沿通用模型。

## 研究问题与动机
1. **单模态局限**：现有专业 CAD 模型多只接受渲染图或文本等单模态输入，工程图纸与真实产品照片这两类常见工业意图严重缺乏覆盖。
2. **泛化能力不足**：基于 ABC、DeepCAD、Fusion 360 Gallery 等合成/原生日益窄化数据集训练的模型在异构真实工业输入上泛化差，往往不及通用前沿模型。
3. **装配级智能缺失**：现有方法几乎全部聚焦单零件生成，装配级设计涉及 BoM 拆解、配合关系规划、位姿估计等，通用 code-assistant 循环无法显式处理这些结构化领域知识。
4. **数据与监督稀缺**：多样化工业设计数据分散且残缺，缺少高质量"金标准"程序，且模型训练难以充分利用不同质量和规模的数据源。

## 核心贡献（创新点）
1. **VisCAD-M1 多模态零件生成器**：27B 模型统一支持文本、2D 工程图、真实照片、渲染图到可执行 FreeCAD Python 程序的映射；与已有工作的本质区别在于首次在同一接口下覆盖从工程图到真实照片的完整工业意图谱系，而非仅处理 CAD 原生渲染。
2. **并行测试时缩放（parallel-tts）**：将训练好的 27B 模型本身用作 reranker，从多路 rollout 中判别式挑选最优程序；本质区别在于"自我重排"而非"自我迭代生成"，实现约 5% 相对 SOTA 提升（0.5797 vs 0.5496）。
3. **DSL-无关装配中间表示（IR）与 VisCAD-H1 harness**：将装配推理与后端 DSL 语法解耦，同一 IR 可跨 FreeCAD、SolidWorks、Autodesk Fusion 等平台复用；区别于通用 coding harness 仅做执行-改写的循环，H1 显式引入 BoM、per-part 约束、全局放置审查等结构化工具。
4. **综合数据治理管道**：利用现成图像编辑/图像到 3D 模型填补缺失意图-程序-形状三元组，并结合 RSI-based 程序搜索获取更精确监督；与直接使用原始公开代码的做法相比，显著扩大可训练意图多样性。
5. **三阶段累积训练管道**：mid-training（语法 + API）→ robust SFT（视觉域桥接）→ continual high-quality SFT（几何保真恢复）；区别于传统单阶段微调，逐步递进最大化不同质量和规模数据的利用率。

## 方法详解
- **共享抽象**：将全部数据资源建模为三元组 $t = (i, p, s)$，其中 $i$ 为多模态意图空间、$p$ 为程序空间（可按 CAD DSL 索引）、$s$ 为 3D 形状空间；生成过程即意图到程序的映射，并伴随几何（形状）与视觉校验。
- **数据治理管道**：
  - 四个步骤类别过滤器保留刚性机械产品，最终保留 21 个主类、128 个子类；
  - 正向路径 $i \rightarrow p \rightarrow s$：标准化意图 → 图像编辑降复杂度 → 图像到 3D 重建形状 → RSI 程序搜索；
  - 反向路径 $s \rightarrow p \rightarrow i$：从已有模型/程序合成多视角投影意图，并做数据增强与递归自改进。
- **三阶段训练**：
  - **Mid-training**：约 100 万条公开 CadQuery 程序，执行导出实体并按多视角投影生成渲染图作为输入，建立可逆的"黄金意图-程序"对，教授 CAD 语法与 API；
  - **Robust SFT**：约 15 万张电商/爬取产品图，经清洗标注后由教师模型蒸馏同一意图下的多个候选程序，降低对单一噪声轨迹的敏感性；
  - **Continual SFT**：约 1 万条精选样本，通过 generate-verify-feedback 循环恢复细粒度几何保真，三阶段累积不互相覆盖。
- **测试时缩放（TTS）**：
  - 尝试两种形式：顺序细化（sequential-tts）无收益；并行重排（parallel-tts）有效。
  - 对同一输入生成 $k$ 路 rollout（程序 + 六视图导出 3D 渲染），由模型 listwise reranking，$k$ 从 8 扩展到 16 分数持续提升。
- **VisCAD-H1 装配工作流（四阶段门控）**：
  1. **Multimodal understanding & BoM**：将多源证据转化为零件清单、装配顺序、配合/边界提示；
  2. **Parallel part IR**：为每个零件分配独立于具体 CAD 后端的中间表示（局部几何、接口、约束），做语法与局部有效性门控；
  3. **Global IR review**：参考视图对齐检查跨部件几何、接口、位姿，发射带 poses 的已审查 IR；
  4. **CAD realization**：将 IR 翻译为后端建模调用，并行构建零件后装配。
- **评估指标**：沿用 RealCADBench 四元组——可执行性 Success Rate、Solid IoU、Surface IoU、Judge Score（基于 Kimi-K2.6 的 GT-free 视觉语义评审），零件级 profile average 为四者等权均值：$P_c = \frac{1}{4}(E_c + G_c^{\text{solid}} + G_c^{\text{surface}} + J_c/100)$。

## 实验与结果
- **数据集**：
  - PubCADBench：聚合 BenchCAD(200)、CADBench(300)、Orthographic Reconstruction(200)、P3D-Text(200)、P3D-Image(200)，共 1,100 任务；
  - RealCADBench：Text(568)、2D Drawing(236)、Real Picture(568)、Rendered Image(373)，共 1,745 任务。
- **基线**：Claude-Opus-4.8、Gemini-3.1-Pro、GPT-5.4、GPT-5.5、Kimi-K3、Doubao-Seed-2.0-pro、Qwen3-VL-8B/32B、Qwen3.8-27B 等前沿与开源模型；装配侧对比 Codex、Claude Code。
- **零件级最强结果**：
  - VisCAD-M1 平均 profile 0.5540，超越 Gemini-3.1-Pro 的 0.5496（PubCADBench 0.5596 vs 0.5533，RealCADBench 0.5485 vs 0.5459）；
  - VisCAD-M1 + parallel-tts（k=16）进一步提升至 0.5797（PubCADBench 0.5833，RealCADBench 0.5761），相对 SOTA 提升约 5%。
- **多切片表现**：
  - BenchCAD Solid IoU 0.4422、CADBench Solid IoU 0.5484 均为最优；
  - RealCADBench Text 切片可执行性 0.9331（远超 Gemini 0.8451）、Real Picture 可执行性 0.9877、Rendered Image Solid IoU 0.5858 均领先；
  - 2D Drawing 切片 Judge 分数仍落后 Gemini（69.13 vs 79.08）。
- **装配级结果**（50 实例）：VisCAD-H1 装配 Judge 85.0，显著优于 Codex 68.0 与 Claude Code 52.0；有效导出率 49/50、49/50、50/50。
- **消融**：
  - Mid-training 在各切片带来 IoU 约 +1pt、Judge 约 +5pt；
  - Robust SFT 中每样本 k=3 蒸馏多候选优于 k=1（图 5 显示训练末期 Solid/Surface IoU 提升）；
  - TTS 中 k=16 比 k=8 继续增益。

## 相关工作脉络
1. **程序原生 CAD 生成**（DeepCAD、SketchGraphs、CAD-Recode 等）：以单零件、单一模态（文本/点云/草图）为核心，VisCAD-M1 扩展到多模态并强调工业异构输入覆盖。
2. **B-Rep / 大模型直接生成**（BRepNet、BrepGen、CAD-Llama、ParaCAD-RL）：直接生成几何拓扑或依赖 RL，VisCAD 采用 DSL 可执行程序路线，保留人机可审计、可编辑特性。
3. **多模态输入扩展**（OpenECAD、CadVLM、ChatCAD、PICASSO 等）：多数仍聚焦渲染图/草图，VisCAD 将工程图纸与真实产品照片纳入统一接口。
4. **基准数据集**（ABC、Fusion 360 Gallery → PubCADBench、RealCADBench）：VisCAD 推动从合成原生日益向工业真实意图（工厂自动化 SKU、工程图纸）的评测迁移。
5. **执行时修订 harness**（Gencad-Selfrepairing、Gencad-3d 等）：通用 code-assistant 只做 execute-observe-rewrite，VisCAD-H1 显式引入 BoM/IR/Placement 等装配领域先验。
6. **定位差异**：VisCAD 不是"更大通用模型在 CAD 上的 fine-tune"，而是"领域特化训练 + DSL-无关装配 harness"的组合，强调工业场景下的可执行性与跨后端一致性。

## 局限性与未来方向
1. **真实照片/工程图生成质量仍有差距**：RealCADBench 2D Drawing 切片 Judge 显著落后 Gemini，真实照片下的视觉语义匹配也有提升空间。
2. **装配生成验证规模有限**：H1 仅在 50 实例上进行定量评估，长尾复杂装配场景尚未充分验证。
3. **图像到 3D 模型的噪声传导**：数据治理中依赖 off-the-shelf image-to-3D 补全形状，误差可能沿管道传递。
4. **TTS 的算力开销**：parallel-tts 需要 $k$ 路 rollout 与 reranking，实际应用需权衡延迟与收益。
5. **未来方向**：将 VisCAD-M1 与 H1 整合为统一的 model-agent 闭环，实现零件生成、装配推理、执行验证、反馈修正的一体化；探索更多意图模态（如触觉/音频）与跨 CAD 后端自动适配。

## 研究启发与可借鉴点
1. **三阶段累积训练范式**：mid-training（语法先验）→ robust SFT（域桥接）→ continual SFT（保真恢复）的顺序可迁移至其他"代码+多模态"生成任务，避免单一阶段训练导致语法与语义失衡。
2. **并行 TTS reranking 而非顺序细化**：对于强自一致性模型，判别式重排比生成式迭代更能稳健提升最终质量，可推广至代码生成、程序合成等场景。
3. **DSL-无关中间表示（IR）设计**：将领域推理层与后端实现层解耦的思路，适用于任何"多后端、多方言"的程序生成系统（如不同数据库 SQL dialect、不同前端框架）。
4. **RSI-based 数据补全 + 递归自改进**：用现有模型/工具补全缺失数据三元组并通过程序搜索提升监督质量，可复用到其他数据稀疏的领域（如科学计算、仿真脚本生成）。
5. **GT-free 视觉语义 Judge 与几何指标联合评估**：结合 Solid/Surface IoU 与 GPT/VLM-based judge 形成互补，避免纯几何指标的"可 hack"弱点，值得在 3D 生成评测中推广。

## 关键术语表
**VisCAD-M1**：27B 多模态基础模型，将文本/图纸/照片/渲染图映射为可执行 FreeCAD Python 程序。
**VisCAD-H1**：面向装配级生成的领域专用 harness，基于四阶段门控工作流与 DSL-无关 IR。
**DSL-agnostic IR**：平台无关的装配中间表示，解耦推理层与后端语法，支持跨 FreeCAD/SolidWorks/Fusion 迁移。
**parallel-tts**：测试时缩放的一种，从同一输入生成 $k$ 路 rollout 并由模型 listwise reranking 挑选最优解。
**RealCADBench**：面向工业产品真实设计意图的多模态 CAD 基准，包含 Text、2D Drawing、Real Picture、Rendered Image 四类切片。
**Profile Average**：零件级综合得分，为可执行性、Solid IoU、Surface IoU、Judge/100 四者的等权均值。
**Recursive Self-Improvement (RSI)**：利用模型自身生成/搜索改进监督数据的循环训练范式。
**BoM（Bill of Materials）**：装配体的零件清单，包含零件数量、类型及装配顺序等信息。

## 可复现要素
- **数据集**：PubCADBench（由 BenchCAD、CADBench、Orthographic Reconstruction、P3D-Text、P3D-Image 聚合）与 RealCADBench；论文未声明是否独立开源，仅引用 RealCADBench 作为评测合同。
- **代码/权重**：论文未明确开源声明（代码仓库与模型权重未列出）。
- **关键超参**：模型参数量 27B；训练规模约 mid-training 1M 程序、robust SFT 150k 图片、continual SFT 10k 精选样本；TTS rollout 数 $k=8$ 与 $k=16$；评估 voxelization 分辨率 R=96，2% padding；对齐使用 PCA-signed alignment 消除位姿与均匀缩放。
