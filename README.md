# Cantonese-LLM-Lab — 粵語大模型

**粵語** · [简体中文](README.zh-CN.md) · [English](README.en.md)

面向粵語嘅問答大模型，喺**昇騰 910C 國產算力**上完成評測同部署。

**第一署名單位：粵語語料庫建設與大模型評測重點實驗室**
（Key Laboratory for Cantonese Corpus Construction and Large Language Model Evaluation）
**資助：** 該實驗室 2025 年度研究課題，**課題編號 25ZD03**。

---

## 亮點

- 🏆 **HKMMLU 開源粵語模型第 1 名**：0.6467，比之前最強嘅開源粵語模型高 **+5.69pp**
  （95%CI [+5.07, +6.38]，p=2.5e-63），全量 26,368 題、官方口徑。
- 🥇 **HKMMLU 論文榜單上所有 ≤34B 模型入面排第 1**：超過 Qwen2.5-14B、Yi-1.5-34B，
  同埋 Mistral-Large，而每個 token 只激活 **3.3B** 參數。
- 🥈 **8B 版本一樣超過全部開源粵語模型**（0.6030）。
- ⚡ **答得準，亦答得短**：解析失敗率只有 **0.03%**，平均輸出 **5.4 token**；
  同一基座要大約 495 token，**短大約 90 倍**，啱晒低延遲同按 token 計費嘅場景。
- 🇨🇳 **國產算力原生支援**：bf16 權重喺昇騰 910C 上免轉換直接運行，
  另外提供昇騰原生 **W8A8 量化版**（11.1 GiB），`vllm-ascend` 可以直接載入。

## HKMMLU（官方 zero-shot，全量 26,368 題）

| 排名 | 模型 | 準確率 |
|---|---|---|
| **1** | **本項目 30B-A3B** | **0.6467** |
| **2** | **本項目 8B** | **0.6030** |
| 3 | `CantoneseLLMChat-v1.0-7B` | 0.5897 |
| 4 | `Qwen2-Cantonese-7B-Instruct` | 0.5883 |
| 5 | `CantoneseLLM-v2.0-8B-Thinking` | 0.5566 |
| 6 | `Llama-3-Cantonese-8B-Instruct` | 0.5240 |
| 7 | `Qwen/Qwen3-8B` | 0.4697 |
| 8 | `CantoneseLLM-v2.0-30B-A3B-Thinking` | 0.4492 |
| 9 | `Qwen/Qwen3-30B-A3B` | 0.4487 |

所有模型用同一套評測、同一硬件，逐題配對檢驗。
呢套評測同 [HKMMLU 論文](https://arxiv.org/abs/2505.02177) 嘅分數已經對齊
（三個公開模型都喺 ±1pp 之內），所以可以同論文榜單直接比較：

| 模型 | HKMMLU |
|---|---|
| **本項目 30B-A3B（激活 3.3B）** | **64.67** |
| Qwen2.5-14B-Instruct | 62.3 |
| Yi-1.5-34B-Chat | 60.6 |
| Mistral-Large-Instruct | 60.0 |

## 模型

門禁倉庫，需要申請；授權 CC BY-NC-ND 4.0。

- **30B-A3B**：https://huggingface.co/Zeteng/yue-qa-qwen3-30b-a3b
- **8B**（含 bf16、昇騰 W8A8、CUDA int4）：https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4
- **數據集**：https://huggingface.co/datasets/Zeteng/cantonese-llm-data

## 授權

本倉庫內容採用 MIT 授權。模型權重同數據集嘅授權以各自嘅 Hugging Face 頁面為準。
