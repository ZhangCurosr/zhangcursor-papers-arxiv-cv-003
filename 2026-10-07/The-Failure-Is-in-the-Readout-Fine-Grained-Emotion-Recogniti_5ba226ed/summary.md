---
title: "The-Failure-Is-in-the-Readout-Fine-Grained-Emotion-Recogniti"
source: https://arxiv.org/pdf/2610.08162v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:40:06"
---

# 论文速读：The-Failure-Is-in-the-Readout-Fine-Grained-Emotion-Recogniti

## 一句话总结
本文指出 EmoNet-Face-HQ 细粒度情感识别基准的评测失败源于“生成式回答解析协议”而非 VLM 本身的感知能力；通过将答案读取方式改为逐类别二元查询并直接从首 token logits 读取 $P(\text{yes})$，11 款现成开源 VLM 的二次加权一致性 $\kappa_w$ 均显著超过人类专家共识锚点（$0.468$），并超越该基准专为弥补此差距而训练的专用微调模型 EIF。

## 研究问题与动机
- **生成式提取协议的结构性缺陷**：基准要求模型输出恰好 25 个非零情感类别的 JSON，固定计数强制模型为未见类别虚构强度值，导致评分被提示词约束而非人脸内容决定，parser 失效或复述选项
