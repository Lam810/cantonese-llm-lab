# cantonese-llm-lab — 粵語大模型微調、評測同昇騰國產卡適配

**粵語** · [简体中文](README.zh-CN.md) · [English](README.en.md)

粵語大模型嘅**微調、評測同昇騰適配**工程筆記，附埋可以直接複用嘅腳本。

呢度包括咗四樣嘢——一套唔靠 LLM 判官嘅評測、一個 LoRA 體檢工具、
一份多源粵語 SFT 數據重建腳本，同埋將同一套推理腳本搬上昇騰 910C 嘅記錄。

---

## 第一署名單位同資助

**第一署名單位：粵語語料庫建設與大模型評測重點實驗室**
（Key Laboratory for Cantonese Corpus Construction and Large Language Model Evaluation）

**資助：** 粵語語料庫建設與大模型評測重點實驗室 2025 年度研究課題，**課題編號 25ZD03**。

---

## 最新結果（2026-09-16）

**一、同基座橫向對照:我們排第一,但有一項輸。**
拉了 6 個現有粵語模型 + 基座,全量 HKMMLU(26,368 題)、117 條生成提問、兩種口徑
（關/開思維鏈,每個模型取它自己最優那個）、McNemar 配對檢驗:

| 每個模型取自己最優口徑 | HKMMLU |
|---|---|
| **本項目** | **0.6227** |
| `CantoneseLLMChat-v1.0-7B` | 0.6045 |
| `Qwen2-Cantonese-7B-Instruct` | 0.5932 |
| `Qwen3-8B`（基座） | 0.5789 |
| `CantoneseLLM-v2.0-8B-Thinking` | 0.5485 |
| `Llama-3-Cantonese-8B-Instruct` | 0.5294 |
| `v2.0-8B-Chat-Vector-Merged` | 0.4612 |

書面粵語純度、輸出長度也都第一（中位 **51** token,對手最高到 583）。
**但粵語維基困惑度輸給 `CantoneseLLMChat-v1.0-7B`（7.77 vs 13.45）**——
他們做的是繼續預訓練,我們只做 LoRA SFT。誠實講:他們的是「粵語語言模型」,
我們的是「識講粵語嘅問答模型」。詳見 [`results/sota_comparison.md`](results/sota_comparison.md)。

**二、公開出去嘅數據子集,比完整語料訓得更好。**
授權原因,71,202 條裡面只有 27,207 條再分發得。原本以為公開嗰份係「閹割版」——
**實測係相反**:只用公開嗰 27,207 條訓,HKMMLU **高 1.19pp（p=6.0e-7）**、純度亦更高。
那 62% 唔可以再分發嘅數據,不單止冇幫手,而且喺拖後腿。
詳見 [`results/ablation_clean_data.md`](results/ablation_clean_data.md)。

**三、兩個已發布嘅數字要更正。** 口徑改嚴之後:HKMMLU 增益由 +3.94pp（抽 3300 題）
變成 **+2.29pp**（全量 26,368 題）;粵語純度由 0.9591（30 條提問）變成
**0.8975**（117 條提問）。增益本身極顯著（p=1.3e-14）,只係幅度細過原先報嘅。
見 [`results/v2_results.md`](results/v2_results.md) 嘅口徑更正一節。

---

## 主要結論

**1. 冇驗證集嘅 SFT，訓練 loss 曲線證明唔到任何嘢。**
出事嗰次訓練，loss 喺 200 步之內由 1.59 跌到 0.25，然後 6400 步走平——睇落好似收斂，
實際上係塌咗入輸出重複串嘅退化解。`eval_strategy="no"` 令呢兩種情況喺日誌上面長得一模一樣。

**2. 權重差 ‖ΔW‖/‖W‖ 係最快嘅 LoRA 體檢。**
310 個張量裡面有 254 個完全冇動過（同 `target_modules` 吻合），但最後 5 層嘅 `q_proj`
相對差達到 **0.50 ~ 1.06**——r=8/alpha=16 嘅 LoRA 正常應該喺 0.01–0.05。
發散係結構化嘅：只打喺最貼近輸出嘅幾層 query 投影上面。

**3. 將未合併嘅 adapter 掛返原始基座，一步就分得開「訓練壞咗」同「合併壞咗」。**
兩邊都退化 ⇒ 合併唔係根因。

**4. 一套 CUDA 評測腳本搬去昇騰，代碼只需要改 4 行；準確率對得上（3300 題差 6 題），吞吐差 3.6 倍。**
`vllm-ascend` 起 OpenAI 兼容服務都通（0.6B / TP=1，**65 秒就緒**），
但有兩個「報錯指唔到真因」嘅坑：缺 **NNAL/ATB**（`libatb.so`，唔喺 CANN 裡面、要單獨 source），
同埋 **`set -u` 會令腳本喺 source 昇騰環境嗰陣靜默退出**。

**5. 跨簡繁嘅字符串指標，兩種字形都一定要寫。**
只寫簡體嘅「普通話標記詞」喺繁體語料上面整張表靜默失配，將粵語純度虛高成 0.964（真值 0.824）。

**6. 對帶思維鏈嘅模型做「第一個 token」類打分，先確認第一個 token 係唔係答案位。**
選擇題比較 A/B/C/D 嘅 logprob，而 Qwen3 默認開思維鏈、第一個 token 係 `<think>`，
量到嘅係「佢想先想一想」。基座因此只得 0.2888（近隨機），熄咗之後係 0.5700——
**我一度將 v2 嘅增益寫成 +31.85pp，真實值係 +3.94pp。**
詳見 [`results/v2_results.md`](results/v2_results.md)。

**7.「本地重新合併嘅權重」同「實際被評測嘅權重」只係假定等價——要驗。**
搬 16GB 唔現實嗰陣，只搬 349MB 嘅 adapter、本地再合併係對嘅做法；但一定要用帶鑑權嘅
HTTP range 請求逐個字節比對（抽 3 個張量各 32,768 個 bf16 元素，100% 相同）。
同一個代理可能下載 8.7 MB/s、上傳 50 KB/s——**兩個方向要分開測**，
而且**估工期要用「已傳字節 ÷ 已耗時」，唔可以用秒級瞬時速率**（我因此兩次估錯）。
HF 嘅 Xet 後端對大文件會報 `xorb not found` 整批失敗，`HF_HUB_DISABLE_XET=1` 繞得過。
詳見 [`docs/publishing-logistics.md`](docs/publishing-logistics.md)。

詳見 [`docs/v1-postmortem.md`](docs/v1-postmortem.md) 同 [`docs/ascend-notes.md`](docs/ascend-notes.md)。

---

## 倉庫結構

```
eval/eval_yue.py            三個客觀任務：HKMMLU 選擇題（logprob 打分）／書面粵語純度
                            ＋退化檢測／分佈內外困惑度（含跨分詞器可比嘅 bits-per-char）
eval/diag_lora_merge.py     LoRA 體檢：base / base+adapter / merged 三路對照
                            ＋逐模組 ‖ΔW‖/‖W‖
data/build_yue_sft.py       多源粵語 SFT 數據重建：合併→opencc 簡繁歸一→兩級去重
                            →按來源分層切 train/dev/test（先切再訓，杜絕洩漏）
train/train_yue_lora.py     LoRA SFT：帶驗證集＋早停＋只對 assistant 段算 loss
train/merge_lora.py         單進程合併 LoRA，並自動做 ‖ΔW‖/‖W‖ 體檢
train/merge_lora_stream.py  逐分片流式合併，峰值內存約 5 GB（低內存機用）
deploy/npu_serve_test.sh    昇騰上面起 vllm-ascend OpenAI 兼容服務並驗收（健康檢查／對話／吞吐）
docs/v1-postmortem.md       v1 點樣發散、點樣查出根因
docs/ascend-notes.md        昇騰 910C 適配筆記，連圖模式優化嘅負結果
docs/data-licensing.md      點解合併語料唔可以再分發（四個來源裡面有兩個唔得），
                            以及對權重授權嘅連帶影響
docs/publishing-logistics.md 將 16GB 權重由隔離集群搬去公開托管站：各段實測帶寬、
                            只搬 adapter 嘅做法、逐字節驗證
results/v1_vs_base.md       v1 同基座嘅對照
results/v2_results.md       v2（Qwen3-8B + LoRA）嘅訓練、合併體檢同評測，
                            連一個被自己修返嘅錯結論
results/sota_comparison.md  同 6 個現有粵語模型嘅橫向對照（兩種口徑＋配對檢驗）
results/ablation_clean_data.md
                            消融：只用公開發布嘅 27,207 條，訓出嚟反而更好
results/quantization.md     fp16 / int4 / int8 變體實測，以及「邊幾欄係噪聲、唔可以當結論」
```

腳本裡面嘅路徑全部寫成 `$LAB_ROOT` / `$HOME`，用之前照自己嘅環境改返。

---

## 評測任務講解

三個任務**全部唔靠 LLM 判官**，可以重算：

| 任務 | 做法 | 點解要咁做 |
|---|---|---|
| **HKMMLU** | 比較 `A/B/C/D` 四個字母 token 嘅 logprob | 細模型 0-shot 好多時唔肯輸出字母；比 logprob 可以繞開「指令遵循能力」呢個混淆變量 |
| **書面粵語純度** | 粵語虛詞（嘅係唔咗喺佢哋…）同普通話虛詞（的是不了在他們…）嘅計數比 | 直接量「佢到底喺講粵語定係講普通話」；**一定要同退化率一齊睇**，唔係一串亂碼裡面出現一個「嘅」都會係 1.0 |
| **退化檢測** | 貪心解碼下 `(.{6,})\1{2,}` 嘅命中率 | 一票否決項：模型仲識唔識講人話 |
| **困惑度** | 同時報 PPL 同 **bits-per-char** | 唔同模型分詞器唔同，PPL 直接比唔得，bits-per-char 就比得 |

---

## 數據重建

合併四個公開來源（作者自建嘅粵語指令集、粵語 CoT、粵語對話、粵中平行語料），
統一 opencc `s2hk` 字形，兩級去重（整條 + 同 instruction），按來源分層切分，
固定 `seed=42`：**71,202 條**（train 68,202 / dev 1,000 / test 2,000）。
確定性驗過：兩台唔同機器重跑，各來源條數、去重數、切分大細逐項一致。

⚠️ **合併語料唔再分發**——四個來源裡面有一個**冇任何授權聲明**、一個係 **AGPL-3.0**，
佔咗 62%。公開嘅係腳本；可以再分發嘅嗰 27,207 條（自有 + CC0）單獨發喺
[`Zeteng/cantonese-llm-data`](https://huggingface.co/datasets/Zeteng/cantonese-llm-data) 嘅
`v2-clean/`。詳見 [`docs/data-licensing.md`](docs/data-licensing.md)。

兩個值得記落嚟嘅數字：
- **50,606 條**記錄喺建庫嗰陣被簡繁歸一化過——公開粵語語料嘅簡繁混排問題係大面積嘅；
- 粵語純度中位數 1.0、均值 0.824。**純度只度量、唔過濾**：清洗閾值最容易將最應該研究嘅樣本丟咗，
  要做消融就用 `--min-yue-ratio`。

---

## 模型權重

微調出嚟嘅權重（v2 完整 bf16、LoRA adapter、int4）喺
**https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4** 。

---

## 授權

代碼 MIT。評測用到嘅 HKMMLU 等數據集跟各自嘅授權。
