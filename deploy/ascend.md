# 昇騰部署（Ascend 910C）

兩個選擇，都已經喺 `Ascend910`（2 die/卡，64 GB HBM/die）上用 `vllm-ascend`
起 OpenAI 兼容服務並驗收通過：

| 權重 | 大細 | 說明 |
|---|---|---|
| `v2-qwen3-8b/`（bf16） | 15.3 GiB | **唔需要任何轉換或量化**，載落嚟就用得 |
| **`v2-ascend-w8a8-dynamic/`** | **11.1 GiB** | **昇騰原生 W8A8（動態）**，`--quantization ascend`，已過非劣效閘門 |

> ⚠️ 倉庫裡嗰個 `v2-qwen3-8b-int4/` 係 **bitsandbytes NF4 格式，`vllm-ascend` 唔支持**，
> 昇騰上唔好用佢。

---

## 一、bf16（免轉換）

| | |
|---|---|
| 硬件 | Ascend 910C，單 die，TP=1 |
| 軟件 | `vllm 0.11.0` + `vllm-ascend 0.11.0` + `torch 2.7.1` / `torch_npu 2.7.1`，CANN 8.5.0 |
| 服務就緒 | **70 秒**（冷啟動到 `/health` 通過） |
| 吞吐 | 8 並發 **286 tok/s**（⚠️ 8 個請求用同一個 prompt，有前綴緩存加成，非冷啟動數字） |
| 對話 | 正常，輸出自然書面粵語 |

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <NNAL安裝路徑>/nnal/atb/set_env.sh        # 這一行不能少，見下面坑①

python -m vllm.entrypoints.openai.api_server \
  --model <本地權重目錄> --served-model-name yue-v2-8b \
  --port 8000 --tensor-parallel-size 1 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --trust-remote-code
```

---

## 二、昇騰原生 W8A8_DYNAMIC（11.1 GiB）

用 **msmodelslim**（昇騰自家工具）造嘅**動態** W8A8——按 token 動態定標。
校準集係 `v2-clean` 嘅 test split（粵語、訓練冇見過）：量化工具默認用英文語料校準，
而呢個模型唯一嘅賣點就係粵語語域，**校準集揀錯就會將佢量化甩咗**。

```bash
# 3 個分片 + index，vllm-ascend 會自己讀 quant_model_description.json
python -m vllm.entrypoints.openai.api_server \
  --model <本地路徑>/v2-ascend-w8a8-dynamic --served-model-name yue-v2-8b-w8a8 \
  --quantization ascend \
  --port 8000 --tensor-parallel-size 1 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --trust-remote-code
```

### 發佈閘門（全量 HKMMLU 26,368 題，同口徑 bf16 對照，逐題配對）

```
對照 bf16 0.5394   W8A8_DYNAMIC 0.5348   Δ −0.46pp
McNemar 精確 p=0.0524   95%CI [−0.93, +0.00]pp
損失置信上界 0.93pp ≤ 容差 δ=2.00pp   ✅ 過閘門
書面粵語純度 0.9102（bf16 0.8791）   退化率 0.0427
解析失敗率 2.72%（bf16 4.42%）   平均輸出 11.3 tok（bf16 13.0）
```

用嘅係**非劣效檢驗**而唔係顯著性檢驗——n 越大越容易顯著，拿顯著性當容差門檻係錯嘅。
損失上界 0.93pp **低過 1~3pp 呢個通行容差區間嘅下沿，所以 δ 取區間內任何值結論都一樣**。

### ⚠️ 靜態 W8A8 我哋造過，決定唔發——而且吞吐更高

```
W8A8_DYNAMIC   8 並發   182 tok / 0.6s = 312.4 tok/s     69 token 自然收尾
W8A8 靜態      8 並發  1024 tok / 2.0s = 511.6 tok/s    128 token 撞上限
```

靜態嗰版吞吐高 **64%**，兩邊 `serve` 都 `rc=0`——但佢嘅輸出係 **token 沙拉**，
**因為佢從來唔吐 EOS、一路生成到 `max_tokens` 上限，所以吞吐反而更高**。
**只睇返回碼同 tok/s 會把報廢嘅模型報成「更好」。**

### 分片

單文件 11,936,951,344 字節超咗 HF 嘅 5 GB 上限，所以切成 3 片。
分片係本地重做嘅產物，同原產物只係**假定**等價，所以驗過：
張量集合 **903/903 一致**、抽檢 **7/7 逐字節相同**，並且喺昇騰上重新起服務復測，
純度／準確率／解析失敗率／輸出長度同未分片版**小數第 16 位一致**。

---

## 打包嗰陣踩過嘅五層坑（自己造量化權重就會遇到）

1. **缺 `config.json` / `generation_config.json`** —— msmodelslim 唔會幫你複製。
2. **描述文件名要精確。** `vllm-ascend` 嘅 `get_config_filenames()` 只認
   `quant_model_description.json`，帶後綴嘅 `..._w8a8_dynamic.json` 讀唔到。
3. **`config.json` 裡面唔可以有 `quantization_config`。** vLLM 嘅
   `get_quant_config()` **先**查 `hf_config.quantization_config`，有就拿佢當量化描述、
   **唔再讀** 那份 903 條嘅描述文件，然後查 `model.embed_tokens.weight` 直接 `KeyError`。
   「加一個字段讓它更明確，結果讓它更盲目。」
4. **分詞器要用上游基座嗰套完整嘅**（9,732 字節、內嵌 4,168 字符模板），
   唔好用 msmodelslim 存嘅 5,404 字節精簡版。
5. **`vllm-ascend` 只支持 W8A8 / W8A8_DYNAMIC / W4A8_DYNAMIC / W4A4_FLATQUANT_DYNAMIC**，
   **冇 weight-only**。而且 NPU 上 `override_quantization_method()` 無條件返回 `"ascend"`，
   所以 `llmcompressor` 出嘅 compressed-tensors 格式喺昇騰上直接
   `KeyError: 'model.embed_tokens.weight'`——**格式唔匹配，唔係配置問題**。

---

## 三個報錯指不到真因的坑

**① `libatb.so: cannot open shared object file` → 缺的是 NNAL，不是 vLLM。**
`vllm-ascend` 依賴 NNAL / ATB，它**不在 CANN 裡**，`ascend-toolkit/set_env.sh` 也不會把它
加進庫路徑，必須額外 `source nnal/atb/set_env.sh`。排查時注意 `libatb.so` 埋在
`.../nnal/atb/<ver>/atb/cxx_abi_{0,1}/lib/` 第 9 層，`find -maxdepth 6` 掃不到，
容易誤判成「沒裝」。

**② `set -u` 會讓作業腳本在 source 昇騰環境時靜默退出。**
昇騰的 `set_env.sh` 引用了未定義的 `LD_LIBRARY_PATH` / `PYTHONPATH` /
`CMAKE_PREFIX_PATH` / `ZSH_VERSION`。而且只寫 `set -x` 不寫 `set -e` 時，
中間命令失敗也會跑到最後的 `echo DONE`，Slurm 記成 `COMPLETED 0:0`。

**③ `AttributeError: 'list' object has no attribute 'keys'` → 分詞器的 transformers 版本錯配。**
`transformers` 5.x 把 `extra_special_tokens` 存成 list，4.x 期望 dict；且 5.x 只存
`tokenizer.json`，不再存 `vocab.json` / `merges.txt`。本倉庫發佈的是**上游基座的完整分詞器
文件**（LoRA 不動分詞器），所以新舊 transformers 都能讀。

---

## 圖模式優化：試過，係負結果

舊版嘅呢一節寫「還沒做：圖模式 / 融合算子（`torch_npu` 的 `torchair`），
這是主要優化空間」。**已經試過，起唔到**——所以呢條路唔係「未做」，係**走唔通**：

- `--additional-config '{"torchair_graph_config":{"enabled":true}}'`
  → `IndexError: index 0 is out of bounds for dimension 0 with size 0`，服務起唔到。
- 換 MoE 模型（`Qwen3-30B-A3B`，torchair 對 MoE 有專門實現）**同一個錯**。
- `enforce_eager=False` 走 ACL Graph（PIECEWISE）**可以**起，但係 vllm-ascend 默認行為，
  唔係 torchair。

當前仍然係 eager 逐算子下發 + ACL Graph，同一套腳本／數據／模型喺 910C 上比 H100
慢約 **3.6 倍**。呢個差距真實存在，但**唔係靠開 torchair 就能收窄**——
呢個版本組合（`vllm-ascend 0.11.0` + `torch_npu 2.7.1`）上開唔起。
完整記錄見 https://github.com/Lam810/cantonese-llm-lab → `docs/ascend-notes.md`。

---

完整腳本（含健康檢查、對話驗收、吞吐壓測）同全部評測腳本見
https://github.com/Lam810/cantonese-llm-lab
（`deploy/npu_serve_test.sh`、`deploy/quant_msmodelslim.py`、`eval/eval_hkmmlu_official.py`）。
