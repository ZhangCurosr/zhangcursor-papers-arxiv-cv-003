---
title: "VISCAD-A-FOUNDATION-MODEL-SUITE-WITH-MUL-TIMODAL-INDUSTRIAL"
source: https://arxiv.org/pdf/2609.03811v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 11:55:58"
field: "多模态程序生成"
keywords: ["CAD生成", "多模态基础模型", "工业AI", "程序生成", "测试时扩展", "装配规划"]
innovations: ["27B多模态零件级生成器VisCAD-M1统一四种意图模态输出可执行FreeCAD程序", "并行测试时扩展（parallel-TTS）利用模型自身作为reranker实现约5%相对SOTA提升", "DSL无关中间表示IR实现装配推理与后端语法解耦并支持FreeCAD/Fusion/SolidWorks跨平台移植"]
benchmarks: ["PubCADBench", "RealCADBench"]
---

# 论文速读：VISCAD: A FOUNDATION MODEL SUITE WITH MULTI-MODAL INDUSTRIAL CAD INTELLIGENCE

## 一句话总结
VisCAD提出了一个工业CAD基础模型套件，包含27B参数多模态生成器VisCAD-M1（将文本、工程图纸、真实产品照片、渲染图统一映射为可执行FreeCAD程序）和领域专用装配harness VisCAD-H1；VisCAD-M1在PubCADBench和RealCADBench上达到平均分数0.5540，超越最强前沿模型，配合并行测试时扩展（parallel-TTS）可进一步提升至0.5797（相对SOTA约5%提升）。

## 研究问题与动机
1. 现有专用CAD模型训练于狭窄输入域（仅CAD渲染图或纯文本），难以泛化到工业场景常见的工程图纸和真实产品照片等多模态意图。
2. 通用前沿多模态大模型（如Gemini、GPT-5.5）虽覆盖更广输入，但在不同CAD域上表现不稳定，且在复杂装配级任务上缺乏领域结构化推理能力。
3. 装配级CAD生成涉及零件分解、BOM规划、配合关系推断、全局位姿估计等多步骤复杂任务，普通code-assistant循环无法暴露BoM、逐零件计划或全局放置审查等原子能力。
4. 现有基准（ABC、Fusion 360 Gallery等）以CAD原生合成数据为主，缺乏对工业制造场景真实输入的有效覆盖和评估。

## 核心贡献（创新点）
1. **VisCAD-M1 27B多模态零件级生成器**：首次通过单一模型接口统一处理文本、工程图纸、真实产品照片、渲染图四种意图模态，输出可执行FreeCAD Python程序；与Gemini-3.1-Pro等SOTA模型相比，在PubCADBench+RealCADBench平均分数上领先约0.8%。
2. **并行测试时扩展方法（parallel-TTS）**：将自训练模型作为reranker，通过listwise策略从k次rollout中选择最优输出；k=16时相对SOTA实现约5%相对提升（0.5797 vs 0.5496），而序列式TTS无效，揭示模型判别能力优于生成式自改进。
3. **综合数据清洗流程（含RSI程序搜索）**：利用现成图像编辑/图像到3D模型工具完成逆向数据补全（i→p→s和s→p→i双向路径），并通过递归自改进程序搜索获取更精确的监督信号，解决了工业场景多样意图与高质量程序对的稀缺问题。
4. **多阶段训练流水线**：mid-training（~1M CADQuery程序）建立语法根基→robust SFT（~150K产品图+教师蒸馏多候选程序）桥接视觉域差距→continual SFT（~10K高质量样本生成-验证-反馈循环）恢复几何保真度，三阶段累加最大化不同质量/规模数据效用。
5. **DSL无关中间表示（IR）与装配harness VisCAD-H1**：将装配推理（BoM分解、逐零件IR、配合关系、全局放置）与后端DSL语法解耦，同一IR可移植到FreeCAD、Fusion、SolidWorks三种CAD环境；50实例装配任务中评分85.0，显著优于Codex（68.0）和Claude Code（52.0）。

## 方法详解
**数据清洗管线**：将异构数据（公开CAD语料、工厂自动化SKU目录、采购工业STEP/STL资产）统一组织为三元组 $t=(i,p,s)$（intent-program-shape），包含正向路径 $i \to p \to s$ 和反向路径 $s \to p \to i$；通过四步类别过滤器保留刚性机械产品（21大类、128小类），移除软件/电子/易耗品等弱CAD类别。

**多阶段训练（Figure 3）**：
- **Mid-training**：执行~1M CadQuery程序导出实体与多视图渲染图，构建"golden intent-program pairs"，教授CAD语法与API对应关系。
- **Robust SFT**：对~150K产品图进行清洗/预处理/标注，教师模型蒸馏出每个意图的多个候选程序（sample_k=3），正则化监督训练降低对单一噪声轨迹的敏感性。
- **Continual SFT**：在generate-verify-feedback循环中使用~10K精选调优样本，恢复细粒度几何保真。

**VisCAD-H1四阶段门控工作流（Figure 4）**：
1. **Multimodal Understanding & BoM**：从产品图像/多视图图纸/文本中提取零件清单、数量、装配顺序、配合/边界提示。
2. **Parallel Part IR**：为每个零件分配独立于具体DSL的中间表示（局部几何、接口、约束），在转为实体前通过语法与局部有效性门控。
3. **Global IR Review**：联合所有零件IR与参考视图检查跨零件几何、接口与放置，发射带姿态的审查后IR；防止"曲轴无轴销、五个活塞缩为一个轮毂"等拼装错误。
4. **CAD Realization**：将审查后IR翻译为后端建模调用，并行构建零件并组装。

**评估指标**：可执行性（Success Rate）、Solid IoU（体素体积重叠，分辨率R=96，PCA符号对齐去除位姿与均匀缩放）、Surface IoU（边界敏感性）、Judge Score（Kimi-K2.6免GT视觉判别模型）。零件级profile平均：$P_c = \frac{1}{4}(E_c + G_c^{\text{solid}} + G_c^{\text{surface}} + J_c/100)$。

## 实验与结果
- **数据集**：PubCADBench（1,100任务：BenchCAD 200、CADBench 300、Orthographic Reconstruction 200、P3D-Text 200、P3D-Image 200）；RealCADBench（1,745任务：Text 568、2D Drawing 236、Real Picture 568、Rendered Image 373）。
- **基线**：Gemini-3.1-Pro、Claude-Opus-4.8、GPT-5.4/5.5、Kimi-K3、Doubao-Seed-2.0-pro、Qwen3-VL-8B/32B、Qwen3.8-27B（零件级）；Codex、Claude Code（装配级）。
- **零件级最强结果**：VisCAD-M1在PubCADBench达0.5596（Solid IoU在CADBench 0.5484最优；BenchCAD 0.4422最优），在RealCADBench达0.5485（Real Picture可执行性0.9877最优；Rendered Image Solid IoU 0.5858最优）；总体平均0.5540，超越Gemini-3.1-Pro（0.5496）。
- **测试时扩展提升**：parallel-TTS k=16使PubCADBench达0.5833、RealCADBench达0.5761，总体0.5797，相对SOTA约+5%。
- **装配级**：VisCAD-H1在50实例子集上Assembly Judge得85.0，Codex 68.0，Claude Code 52.0；有效样本率均接近100%。

## 相关工作脉络
1. **DeepCAD/Text2CAD/CAD-Recode**：程序原生CAD生成，但仅支持单模态意图（文本/草图/点云）且限于单个零件；VisCAD-M1扩展至四种多模态输入并面向工业场景。
2. **BRepNet/BrepGen/CAD-Llama/ParaCAD-RL**：直接学习B-Rep或RL训练；VisCAD-M1坚持FreeCAD Python DSL输出以保障可执行性与编辑性。
3. **OpenECAD/CadVLM/ChatCAD/PICASSO**：扩展多模态输入但仍未充分覆盖工程图纸和真实产品照片，且输出多为单零件；VisCAD-M1统一四种意图模态接口。
4. **BenchCAD/CADBench/P3D-Bench**：公开程序生成基准；本文将其聚合为PubCADBench并与RealCADBench（工业真实意图）配合形成完整评估。
5. **General code-assistant harnesses（Codex/Claude Code）**：通用执行-重写循环，缺乏BoM分解、逐零件IR门控、配合审查等结构化装配推理；VisCAD-H1通过领域专用工作流和DSL无关IR实现跨CAD平台可移植。

## 局限性与未来方向
1. 装配级数据相对稀缺，当前50实例装配评估规模有限，需更大范围真实工业装配数据验证。
2. 零件级与装配级目前仍为独立组件（M1与H1），尚未完全整合为单一holistic agent-native CAD智能体；未来需统一零件生成、装配推理、执行验证与反馈闭环。
3. Judge Score与几何IoU存在不一致（如CADBench上VisCAD-M1几何重叠最优但Judge低于GPT-5.5/Gemini），表明视觉判别模型仍需与几何指标更好对齐。
4. parallel-TTS提升显著但线性扩展受rollout数k限制，未来需探索更高效的多样性-质量权衡策略。

## 研究启发与可借鉴点
1. **多阶段训练的"质量-规模"协同设计**：mid-training用大规模精确数据建立语法根基，robust SFT用中等规模噪声数据+教师蒸馏增强泛化，continual SFT用小规模高质量数据恢复细节——该三段式对任何领域VLM训练均有参考价值。
2. **DSL无关中间表示（IR）思想**：将领域推理（装配/配合/放置）与后端实现语法解耦，使同一计划可移植至多种CAD环境；类似思路可迁移到其他领域（如芯片布局、电路设计）的领域智能体构建。
3. **测试时扩展的"判别优于生成"发现**：模型自身无法通过序列式refinement改善输出，但可通过并行reranking显著提升——这对后续研究中的推理阶段设计有重要启示：应优先开发强判别器而非盲目增加生成步数。
4. **双向数据构建管线（i→p→s与s→p→i）**：从缺失模态出发，利用现成图像编辑/图像到3D工具补全意图或程序，再经RSI搜索优化，为多模态训练数据构建提供了可复用的数据增强范式。
5. **多维度评估体系**：结合可执行性、体素IoU、表面IoU与免GT视觉Judge的综合评分，兼顾工程可用性与视觉保真度，可作为多模态程序生成任务的评估模板。

## 关键术语表
- **VisCAD-M1**：27B参数多模态基础模型，将文本/工程图纸/真实照片/渲染图映射为可执行FreeCAD Python程序，用于零件级CAD生成。
- **VisCAD-H1**：CAD原生装配级harness，基于前沿VLM通过四阶段门控工作流完成BoM分解、逐零件IR构建、全局放置审查与后端 realization。
- **parallel-TTS**：Test-Time Scaling方法，将模型自身作为listwise reranker，从k次rollout中选择最优输出；k=16时实现约5%相对SOTA提升。
- **DSL-agnostic IR（中间表示）**：平台无关的装配结构表示，包含BOM、零件局部几何、接口约束、配合关系与全局放置，可在FreeCAD/Fusion/SolidWorks间移植。
- **PubCADBench**：聚合BenchCAD、CADBench、正交重建、P3D-Text、P3D-Image五个公开基准的1,100任务零件级评测集。
- **RealCADBench**：面向工业真实设计意图（文本、2D图纸、真实产品照片、渲染图）的1,745任务评测集，包含可执行性、IoU与免GT视觉Judge多维指标。
- **Solid IoU / Surface IoU**：经PCA符号对齐后，体素填充重叠率（体积敏感）与边界体素重叠率（薄结构敏感）的评估指标。
- **RSI（Recursive Self-Improvement）程序搜索**：利用递归自改进思想，在给定意图与近似形状下搜索更优CAD程序以实现更精确监督。

## 可复现要素
- **数据集**：PubCADBench由公开基准聚合；RealCADBench由JoyIndustrial团队发布（Team, 2026，unpublished benchmark draft）；公开CAD语料（CadQuery等）、工厂自动化SKU目录、采购STEP/STL资产来源见Appendix B。论文未明确说明数据是否对外公开。
- **代码/权重**：论文未明确声明VisCAD-M1权重或VisCAD-H1代码是否开源（论文发表年份为2026年，可能尚未开源）。
- **关键超参**：mid-training ~1M程序；robust SFT ~150K产品图，sample_k=3；continual SFT ~10K样本；parallel-TTS中k∈{8,16}；评估voxel分辨率R=96、padding 2%；Judge模型为Kimi-K2.6。
