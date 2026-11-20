# Cantonese-LLM-Lab — Cantonese Large Language Models

[粵語](README.md) · [简体中文](README.zh-CN.md) · **English**

Cantonese question-answering LLMs, evaluated and deployed on **Ascend 910C** domestic NPUs.

**First affiliation: Key Laboratory for Cantonese Corpus Construction and Large Language Model Evaluation**
**Funding:** 2025 research project of the Laboratory, **grant no. 25ZD03**.

---

## Highlights

- 🏆 **#1 open Cantonese model on HKMMLU**: 0.6467, **+5.69pp** over the previous best open
  Cantonese model (95% CI [+5.07, +6.38], p=2.5e-63), full 26,368 questions, official protocol.
- 🥇 **#1 among all ≤34B models on the HKMMLU paper leaderboard**: ahead of Qwen2.5-14B,
  Yi-1.5-34B and Mistral-Large, with only **3.3B** parameters active per token.
- 🥈 **The 8B model also beats every open Cantonese model** (0.6030).
- ⚡ **Accurate and concise**: parse failures **0.03%**, average output **5.4 tokens**,
  versus ~495 for the same base model (**~90× shorter**), well suited to low-latency and
  per-token-billed deployments.
- 🇨🇳 **Native domestic-NPU support**: bf16 weights run on Ascend 910C with no conversion,
  plus an Ascend-native **W8A8** build (11.1 GiB) that loads directly in `vllm-ascend`.

## HKMMLU (official zero-shot, full 26,368 questions)

| Rank | Model | Accuracy |
|---|---|---|
| **1** | **Ours 30B-A3B** | **0.6467** |
| **2** | **Ours 8B** | **0.6030** |
| 3 | `CantoneseLLMChat-v1.0-7B` | 0.5897 |
| 4 | `Qwen2-Cantonese-7B-Instruct` | 0.5883 |
| 5 | `CantoneseLLM-v2.0-8B-Thinking` | 0.5566 |
| 6 | `Llama-3-Cantonese-8B-Instruct` | 0.5240 |
| 7 | `Qwen/Qwen3-8B` | 0.4697 |
| 8 | `CantoneseLLM-v2.0-30B-A3B-Thinking` | 0.4492 |
| 9 | `Qwen/Qwen3-30B-A3B` | 0.4487 |

All models were run with the same evaluation on the same hardware, with paired per-question tests.
Our evaluation matches the [HKMMLU paper](https://arxiv.org/abs/2505.02177) scores (three
public models all within ±1pp), so results are directly comparable to the paper's leaderboard:

| Model | HKMMLU |
|---|---|
| **Ours 30B-A3B (3.3B active)** | **64.67** |
| Qwen2.5-14B-Instruct | 62.3 |
| Yi-1.5-34B-Chat | 60.6 |
| Mistral-Large-Instruct | 60.0 |

## Models

Gated repositories (request access); licence CC BY-NC-ND 4.0.

- **30B-A3B**: https://huggingface.co/Zeteng/yue-qa-qwen3-30b-a3b
- **8B** (bf16, Ascend W8A8, CUDA int4): https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4
- **Dataset**: https://huggingface.co/datasets/Zeteng/cantonese-llm-data

## Licence

Repository content is MIT-licensed. Model weights and datasets follow the licences on their Hugging Face pages.
