# CUDA 部署（NVIDIA）

CUDA 上有三個選擇：

| 權重 | 大細 | 說明 |
|---|---|---|
| `v2-qwen3-8b/`（bf16） | 15.3 GiB | ✅ 首選，單卡 ≥24 GB 可跑 |
| `v2-qwen3-8b-int4/` | **5.7 GiB** | NF4 + double quant，一張 8 GB 卡跑得；**只限 CUDA** |
| `v2-clean-data/` | 15.3 GiB | **只用公開數據訓嘅消融版**，完全可以自己複現；HKMMLU 官方口徑下仲高過完整版 1.65pp |

（昇騰 910C 請睇 [`ascend.md`](ascend.md)——有原生 W8A8，**但 int4 嗰份
bitsandbytes 格式 `vllm-ascend` 唔支持**。）

## transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
mid, sub = "Zeteng/qwen_yue_qa_finetuned_int4", "v2-qwen3-8b"
tok = AutoTokenizer.from_pretrained(mid, subfolder=sub)
model = AutoModelForCausalLM.from_pretrained(mid, subfolder=sub,
                                             dtype="bfloat16", device_map="auto")
```

問答場景在 `apply_chat_template` 裡加 `enable_thinking=False`。

## vLLM（OpenAI 兼容服務）

先把子目錄拉到本地，再指本地路徑（vLLM 不直接支持 `subfolder`）：

```bash
hf download Zeteng/qwen_yue_qa_finetuned_int4 --include "v2-qwen3-8b/*" --local-dir ./yue-v2
python -m vllm.entrypoints.openai.api_server \
  --model ./yue-v2/v2-qwen3-8b --served-model-name yue-v2-8b \
  --port 8000 --tensor-parallel-size 1 --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

## 實測環境

評測跑在 **H100 80GB**（`torch 2.6.0+cu124` / `transformers 5.6.2`）。
HKMMLU 3300 題約 116 秒；30 條生成提問（`max_new_tokens=128`）約 60 秒。
評測腳本見 https://github.com/Lam810/cantonese-llm-lab → `eval/eval_yue.py`。

## int4（NF4，5.7 GiB）

```python
mid, sub = "Zeteng/qwen_yue_qa_finetuned_int4", "v2-qwen3-8b-int4"
tok = AutoTokenizer.from_pretrained(mid, subfolder=sub)
model = AutoModelForCausalLM.from_pretrained(mid, subfolder=sub, device_map="auto")
```

要裝 `bitsandbytes`；量化配置已經寫喺 `config.json` 裡面，唔需要自己傳 `BitsAndBytesConfig`。

## 注意

- **生成用 `temperature>0`。** 貪心解碼在列舉類問題上會重複同一句。
- **選擇題類評測，用官方口徑先。** 我哋主表用嘅係 HKMMLU **官方 zero-shot prompting**
  （生成式、要模型直接輸出字母、解析唔出嚟嘅單獨記 `unparsed_rate` 並按答錯計），
  腳本係 `eval/eval_hkmmlu_official.py`。**只有呢個口徑同官方榜單可比。**
- 如果你要用「比較第一個位置上 A/B/C/D 的 logprob」呢種打分法（我哋嘅輔助口徑），
  **務必先關思維鏈**：Qwen3 默認開 thinking、第一個 token 係 `<think>`，
  量到嘅係「它想先想一想」而唔係知識。基座因此一度只得 0.2888（近隨機），
  關咗之後係 0.5700——**我一度將增益寫成 +31.85pp**。
  而且**呢個口徑同官方榜單唔可以直接比**，官方標準係 prompting。
