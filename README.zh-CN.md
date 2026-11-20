# Cantonese-LLM-Lab — 粤语大模型

[粵語](README.md) · **简体中文** · [English](README.en.md)

面向粤语的问答大模型，在**昇腾 910C 国产算力**上完成评测与部署。

**第一署名单位：粤语语料库建设与大模型评测重点实验室**
（Key Laboratory for Cantonese Corpus Construction and Large Language Model Evaluation）
**资助：** 该实验室 2025 年度研究课题，**课题编号 25ZD03**。

---

## 亮点

- 🏆 **HKMMLU 开源粤语模型第 1 名**：0.6467，领先此前最强的开源粤语模型 **+5.69pp**
  （95%CI [+5.07, +6.38]，p=2.5e-63），全量 26,368 题、官方口径。
- 🥇 **HKMMLU 论文榜单上所有 ≤34B 模型中排第 1**：超过 Qwen2.5-14B、Yi-1.5-34B，
  以及 Mistral-Large，而每个 token 只激活 **3.3B** 参数。
- 🥈 **8B 版本同样超过全部开源粤语模型**（0.6030）。
- ⚡ **答得准，也答得短**：解析失败率仅 **0.03%**，平均输出 **5.4 token**；
  同基座需要约 495 token，**短约 90 倍**，适合低延迟和按 token 计费的场景。
- 🇨🇳 **国产算力原生支持**：bf16 权重在昇腾 910C 上免转换直接运行，
  另提供昇腾原生 **W8A8 量化版**（11.1 GiB），`vllm-ascend` 可直接加载。

## HKMMLU（官方 zero-shot，全量 26,368 题）

| 排名 | 模型 | 准确率 |
|---|---|---|
| **1** | **本项目 30B-A3B** | **0.6467** |
| **2** | **本项目 8B** | **0.6030** |
| 3 | `CantoneseLLMChat-v1.0-7B` | 0.5897 |
| 4 | `Qwen2-Cantonese-7B-Instruct` | 0.5883 |
| 5 | `CantoneseLLM-v2.0-8B-Thinking` | 0.5566 |
| 6 | `Llama-3-Cantonese-8B-Instruct` | 0.5240 |
| 7 | `Qwen/Qwen3-8B` | 0.4697 |
| 8 | `CantoneseLLM-v2.0-30B-A3B-Thinking` | 0.4492 |
| 9 | `Qwen/Qwen3-30B-A3B` | 0.4487 |

所有模型使用同一套评测、同一硬件，逐题配对检验。
该评测与 [HKMMLU 论文](https://arxiv.org/abs/2505.02177) 的分数已对齐
（三个公开模型均在 ±1pp 以内），因此可与论文榜单直接比较：

| 模型 | HKMMLU |
|---|---|
| **本项目 30B-A3B（激活 3.3B）** | **64.67** |
| Qwen2.5-14B-Instruct | 62.3 |
| Yi-1.5-34B-Chat | 60.6 |
| Mistral-Large-Instruct | 60.0 |

## 模型

门禁仓库，需申请；授权 CC BY-NC-ND 4.0。

- **30B-A3B**：https://huggingface.co/Zeteng/yue-qa-qwen3-30b-a3b
- **8B**（含 bf16、昇腾 W8A8、CUDA int4）：https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4
- **数据集**：https://huggingface.co/datasets/Zeteng/cantonese-llm-data

## 授权

本仓库内容采用 MIT 授权。模型权重与数据集的授权以各自的 Hugging Face 页面为准。
