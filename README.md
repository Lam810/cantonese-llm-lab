# cantonese-llm-lab

粵語大模型的**微調、評測與昇騰適配**工程筆記，附可直接複用的腳本。

本倉庫的起點是一次失敗：一個已經公開發佈的粵語 LoRA 微調模型，補做評測時才發現**生成完全退化**。
把根因查清楚之後，這裡沉澱了四件東西——一套不依賴 LLM 判官的評測、一個 LoRA 體檢工具、
一份多源粵語 SFT 數據重建腳本，以及把同一套推理腳本搬上昇騰 910C 的記錄。

---

## 第一署名單位與資助

**第一署名單位：粵語語料庫建設與大模型評測重點實驗室**
（Key Laboratory for Cantonese Corpus Construction and Large Language Model Evaluation）

**資助：** 粵語語料庫建設與大模型評測重點實驗室 2025 年度研究課題，**課題編號 25ZD03**。

---

## 主要結論

**1. 沒有驗證集的 SFT，訓練 loss 曲線不能證明任何事。**
出事的那次訓練，loss 在 200 步內從 1.59 掉到 0.25 然後 6400 步走平——看起來像收斂，
實際是塌進了輸出重複串的退化解。`eval_strategy="no"` 讓這兩種情況在日誌上長得一模一樣。

**2. 權重差 ‖ΔW‖/‖W‖ 是最快的 LoRA 體檢。**
310 個張量裡 254 個完全沒動（和 `target_modules` 吻合），但最後 5 層的 `q_proj`
相對差達到 **0.50 ~ 1.06**——r=8/alpha=16 的 LoRA 正常應在 0.01–0.05。
發散是結構化的：只打在最靠近輸出的幾層 query 投影上。

**3. 把未合併的 adapter 掛回原始基座，能一步分開「訓練壞了」和「合併壞了」。**
兩者都退化 ⇒ 合併不是根因。

**4. 一套 CUDA 評測腳本搬到昇騰，代碼只需改 4 行；準確率對得上（3300 題差 6 題），吞吐差 3.6 倍。**
`vllm-ascend` 起 OpenAI 兼容服務也是通的（0.6B / TP=1，**65 秒就緒**），
但有兩個報錯指不到真因的坑：缺 **NNAL/ATB**（`libatb.so`，不在 CANN 裡、要單獨 source），
以及 **`set -u` 會讓腳本在 source 昇騰環境時靜默退出**。

**5. 跨簡繁的字符串指標必須兩種字形都寫。**
只寫簡體的「普通話標記詞」在繁體語料上整張表靜默失配，把粵語純度虛高成 0.964（真值 0.824）。

詳見 [`docs/v1-postmortem.md`](docs/v1-postmortem.md) 和 [`docs/ascend-notes.md`](docs/ascend-notes.md)。

---

## 倉庫結構

```
eval/eval_yue.py            三個客觀任務：HKMMLU 選擇題（logprob 打分）／書面粵語純度
                            ＋退化檢測／分佈內外困惑度（含跨分詞器可比的 bits-per-char）
eval/diag_lora_merge.py     LoRA 體檢：base / base+adapter / merged 三路對照
                            ＋逐模組 ‖ΔW‖/‖W‖
data/build_yue_sft.py       多源粵語 SFT 數據重建：合併→opencc 簡繁歸一→兩級去重
                            →按來源分層切 train/dev/test（先切再訓，杜絕洩漏）
train/train_yue_lora.py     LoRA SFT：帶驗證集＋早停＋只對 assistant 段算 loss
deploy/npu_serve_test.sh    昇騰上起 vllm-ascend OpenAI 兼容服務並驗收（健康檢查／對話／吞吐）
train/merge_lora.py         單進程合併 LoRA 並自動做 ‖ΔW‖/‖W‖ 體檢
docs/data-licensing.md      為什麼合併語料不能再分發（四個來源裡兩個不能），以及對權重授權的連帶影響
docs/                       復盤與昇騰筆記
results/                    評測數字
```

腳本裡的路徑都寫成 `$LAB_ROOT` / `$HOME`，用之前按自己的環境替換。

---

## 評測任務說明

三個任務**都不依賴 LLM 判官**，可複算：

| 任務 | 做法 | 為什麼這樣做 |
|---|---|---|
| **HKMMLU** | 比較 `A/B/C/D` 四個字母 token 的 logprob | 小模型 0-shot 常常不肯輸出字母；比 logprob 繞開了指令遵循能力這個混淆變量 |
| **書面粵語純度** | 粵語虛詞（嘅係唔咗喺佢哋…）與普通話虛詞（的是不了在他們…）的計數比 | 直接量「它到底在說粵語還是普通話」；**必須和退化率一起看**，否則一串亂碼裡出現一個「嘅」也會是 1.0 |
| **退化檢測** | 貪心解碼下 `(.{6,})\1{2,}` 命中率 | 一票否決項：模型還會不會說人話 |
| **困惑度** | 同時報 PPL 和 **bits-per-char** | 不同模型分詞器不同，PPL 不可直接比，bits-per-char 可以 |

---

## 數據重建

合併四個公開來源（作者自建的粵語指令集、粵語 CoT、粵語對話、粵中平行語料），
統一 opencc `s2hk` 字形，兩級去重（整條 + 同 instruction），按來源分層切分，
固定 `seed=42`：**71,202 條**（train 68,202 / dev 1,000 / test 2,000）。
確定性已驗證：兩台不同機器重跑，各來源條數、去重數、切分大小逐項一致。

⚠️ **合併語料不再分發**——四個來源裡有一個**沒有任何授權聲明**、一個是 **AGPL-3.0**，
佔了 62%。公開的是腳本，可再分發的那 27,207 條（自有 + CC0）單獨發在
[`Zeteng/cantonese-llm-data`](https://huggingface.co/datasets/Zeteng/cantonese-llm-data) 的
`v2-clean/`。詳見 [`docs/data-licensing.md`](docs/data-licensing.md)。

兩個值得記下的數字：
- **50,606 條**記錄在建庫時被簡繁歸一化過——公開粵語語料的簡繁混排問題是大面積的；
- 粵語純度中位數 1.0、均值 0.824。**純度只度量不過濾**：清洗閾值最容易把最該研究的樣本丟掉，
  要做消融再用 `--min-yue-ratio`。

---

## 授權

代碼 MIT。評測用到的 HKMMLU 等數據集遵循各自的授權。
