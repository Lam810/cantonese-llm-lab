# Cantonese-LLM-Lab — Cantonese LLM fine-tuning, evaluation, and Ascend NPU porting

[粵語](README.md) · [简体中文](README.zh-CN.md) · **English**

This project covers **fine-tuning, evaluating, and porting a Cantonese LLM to Ascend NPUs**,
with reusable scripts.

The repository contains six components:
1. **An evaluation suite without an LLM judge** — HKMMLU under the official protocol on the
   full set, the official yue↔zh translation tasks, written-Cantonese purity, degeneration
   detection, and perplexity, all paired per question with exact McNemar tests;
2. **A prompt-ablation design** — a 2×2 (language × format scaffold) for testing whether a
   ranking is attributable to the prompt;
3. **A release gate** — a non-inferiority test (rather than a significance test) plus a
   δ-sensitivity check;
4. **A LoRA diagnostic tool** — per-module ‖ΔW‖/‖W‖;
5. **A multi-source Cantonese SFT data rebuild script** — script normalisation, two-level
   deduplication, split-before-train;
6. **A record of porting to Ascend 910C** — four lines of code changed coming from CUDA, plus
   native W8A8 quantization, a **negative result** on graph-mode optimisation, and several cases
   where the error message does not identify the underlying cause.

---

## Primary affiliation and funding

**Primary affiliation: Key Laboratory for Cantonese Corpus Construction and Large Language
Model Evaluation** (粵語語料庫建設與大模型評測重點實驗室)

**Funding:** 2025 annual research project of the Key Laboratory for Cantonese Corpus
Construction and Large Language Model Evaluation, **Grant No. 25ZD03**.

---

## Latest results (2026-09-18)

This project maintains **two parallel model lines with no designated flagship** — their
strengths differ:

| | HKMMLU (knowledge) | Man→Yue (producing Cantonese) | Yue→Man (understanding Cantonese) | Unparsed |
|---|---|---|---|---|
| **30B-A3B (MoE, attention-only LoRA)** | **0.6467 (1st of 11)** | 44.51 | **57.30** | **0.0003** |
| **8B (dense, seven-module LoRA)** | 0.5394 (6th of 11) | **52.36** | 52.81 | 0.0442 |

**Use the 30B for knowledge QA and format compliance; the 8B remains preferable for
generating idiomatic written Cantonese.** Full results for the 30B run, the attribution
analysis (base versus recipe; net of the format component) and the known trade-off are in
[`results/moe_30b.md`](results/moe_30b.md).

Sections 1 and 2 below compare the **eight models at the 8B scale**, under exactly the same
protocol as the 30B run.

**1. HKMMLU, official protocol, full 26,368 questions: the 8B release is 4th of 8.**

| Rank | Model | Accuracy | 95% CI | Unparsed | Mean out tok |
|---|---|---|---|---|---|
| 1 | `CantoneseLLMChat-v1.0-7B` | 0.5897 | ±0.59pp | 0.0000 | 11.7 |
| 2 | `Qwen2-Cantonese-7B-Instruct` | 0.5883 | ±0.59pp | 0.0000 | 8.8 |
| 3 | **This project — ablation (public data only)** | 0.5560 | ±0.60pp | 0.0000 | 9.1 |
| 4 | **This project — v2** | 0.5394 | ±0.60pp | **0.0442** | 13.0 |
| 5 | `Llama-3-Cantonese-8B-Instruct` | 0.5240 | ±0.60pp | 0.0005 | 4.4 |
| 6 | `Qwen3-8B` (base) | 0.4697 | ±0.60pp | 0.0281 | 471 |
| 7 | `CantoneseLLM-v2.0-8B-Thinking` | 0.3993 | ±0.59pp | 0.0825 | 451 |
| 8 | `v2.0-8B-Thinking-Chat-Vector-Merged` | 0.3968 | ±0.59pp | 0.0736 | 493 |

Paired McNemar, 8B release against the leader of this table: **−5.03pp**
(95% CI [−5.68, −4.37], p = 1.6e-51). The difference is statistically significant.

> This table excludes the 30B run. Under the same protocol: this project's 30B-A3B scores
> **0.6467**, ahead of the leader of this table by **+5.69pp** (95% CI [+5.07, +6.38],
> p = 2.5e-63); `Qwen/Qwen3-30B-A3B` (base) scores 0.4487. Inserted into this table, our
> 30B would rank 1st and its base 10th.

**The ranking remains stable under prompt ablation.** We ran a full 2×2 over the prompt (language × format
scaffold) plus two sample sizes = **8 combinations, and the rival wins every one of them**.
Even letting each model be scored at its own best arm: `chat_7b` 0.6018 > `qwen2` 0.5889 >
this project 0.5767. See [`results/prompt_ablation.md`](results/prompt_ablation.md).

**2. Official translation tasks: the two directions rank in opposite orders. This result
distinguishes the capabilities measured in the two directions.**

| | yue→zh chrF | zh→yue chrF |
|---|---|---|
| **This project — v2** | 23.95 (**52.81** after normalising script) | **51.57** (1st) |
| `CantoneseLLMChat-v1.0-7B` | **59.51** (60.09) | 46.06 |
| `Llama-3-Cantonese-8B-Instruct` | 16.74 (31.94) | 48.18 |
| `Qwen2-Cantonese-7B-Instruct` | 13.94 (26.27) | 37.94 |

**We are 1st at *producing* Cantonese** (51.57 vs 46.06). At *understanding* it the raw
number is 4th, but the official references are **Simplified** while our output is
**Traditional**, and chrF is character-level — after separating out script we are **3rd and
7.28 points behind**, not the 35.6 the raw figure suggests. We found sentences whose
register was converted correctly throughout (喺→在, 同→和, 係→是, 嘅→的) yet scored near
zero purely on orthography.

**3. The primary limitation is instruction following — not knowledge or comprehension.**

| Evidence | Figure |
|---|---|
| HKMMLU unparsed rate | **4.42%** (both strong rivals: 0.0000) |
| Share of the gain from one added format-scaffold sentence that is purely recovered parse failures | **32%**, highest of all 8 models (base 4%, Thinking 12%) |
| Traditional-script output rate on yue→zh | **99.7%** (the prompt explicitly asks for Simplified; rival: 5.0%) |

Three independent sources point in the same direction. **The corresponding intervention also
localizes the primary limitation**: adding the scaffold drops our unparsed rate
from 4.42% to 0.62% (−86%), while switching to a Cantonese prompt only reaches 4.01%
(essentially unchanged). **The next training round needs format-following data, not more
Cantonese text.**

Performance is lower on two axes: `hk_*` (Hong Kong local knowledge) 0.5334, second-worst of
the eight (leader 0.6243); and Cantonese-Wikipedia perplexity 13.45 against 7.77 for
`CantoneseLLMChat-v1.0-7B` — they did continued pre-training (CPT), we only did LoRA SFT.
This reflects the different training objectives: **the rival is a Cantonese *language model*,
whereas this project is a question-answering model with Cantonese generation capability**. The
two translation directions are consistent with this distinction.
See [`results/standard_eval.md`](results/standard_eval.md).

**4. Non-chain-of-thought deployment.**
Mean output **13 tokens**, against 471 for the base and 451/493 for the two Thinking
variants — **roughly 35×**. Where chain-of-thought is not an option (short answers, low
latency, per-token billing, edge devices) those models are unsuitable.

**5. The public-data subset outperforms the full corpus under both protocols.**
Licensing leaves only 27,207 of 71,202 examples redistributable. Although the public subset is
smaller, **the measured performance is higher**: +1.19pp on the
logprob protocol (p = 6.0e-7) and **+1.65pp on the official one** (p = 5.8e-10).
The gap is larger under the second protocol, indicating that it is not specific to one
evaluation protocol. Both sets of weights are published; the ablation is reproducible from the
published weights and evaluation scripts. See [`results/ablation_clean_data.md`](results/ablation_clean_data.md).

**6. An Ascend-native W8A8_DYNAMIC build is published.**
Dynamic W8A8 via msmodelslim (11.1 GiB), loads straight into `vllm-ascend`. Full 26,368
questions, same-protocol bf16 control, paired per question: Δ **−0.46pp**, 95% CI
[−0.93, +0.00], upper confidence bound on the loss 0.93pp ≤ tolerance δ = 2.0pp.
**The static W8A8 variant is not published**: although it is 64% faster, its output is severely
degraded because it does not emit EOS and continues generating until the token limit. See
[`results/quantization.md`](results/quantization.md).

## Repository layout

```
eval/eval_hkmmlu_official.py  HKMMLU under the **official protocol** (zero-shot prompting,
                            generative) on the full set, plus the official yue↔zh translation
                            tasks (chrF/BLEU, also reported after script normalisation) and
                            written-Cantonese purity; `--mc-style` drives the prompt ablation
eval/eval_yue.py            Secondary protocol: HKMMLU first-token logprob scoring, purity,
                            degeneration detection, in/out-of-distribution perplexity
                            (with tokenizer-comparable bits-per-char). The 117 purity
                            prompts live in this file
eval/compare_models.py      Paired per-question testing (exact McNemar; the binomial is
                            evaluated in log space)
eval/prompt_ablation.py     Prompt ablation: strict vs scaffold, paired per question, and
                            how much of the gain is merely recovered parse failures
eval/prompt_grid.py         Prompt 2x2 (language x format scaffold): main effects,
                            interaction, unparsed rate across all four arms
eval/gate_check.py          Release gate: **non-inferiority test** against a same-protocol
                            control, paired per question, plus a delta-sensitivity self-check
eval/summarize_std.py       Standard-eval summary (multiple choice / translation / purity)
eval/summarize_trans.py     Translation summary: raw chrF alongside script-normalised chrF
data/build_yue_sft.py       Multi-source Cantonese SFT rebuild: merge -> opencc script
                            normalisation -> two-level dedup -> source-stratified
                            train/dev/test split (split before training, no leakage)
train/train_yue_lora.py     LoRA SFT with a validation set, early stopping, and loss on
                            assistant spans only
train/merge_lora.py         Single-process LoRA merge with an automatic ||dW||/||W|| check
deploy/ascend.md            Ascend deployment guide (**also published under deploy/ in the HF
                            model card**): bf16 conversion-free and native W8A8, five classes
                            of packaging problem, and the torchair negative result
deploy/cuda.md              CUDA deployment guide (also mirrored to HF): bf16 / int4 /
                            the public-data ablation
docs/v1-postmortem.md       How v1 diverged and how the root cause was found
docs/ascend-notes.md        Ascend 910C porting notes, including the negative result on
                            graph-mode optimisation
docs/data-licensing.md      Why the merged corpus cannot be redistributed (two of four
                            sources block it) and the knock-on effect on weight licensing
docs/publishing-logistics.md Moving 16 GB of weights from an isolated cluster to a public
                            host: measured per-hop bandwidth, the adapter-only approach,
                            byte-level verification
results/standard_eval.md    **The main results document**: eight models at the 8B scale
                            x 3 tasks, full-set paired tests
results/moe_30b.md          **Changing the base**: the same recipe applied to
                            Qwen3-30B-A3B (MoE) -- results, attribution (base versus
                            recipe; net of the format component), known trade-off
results/prompt_ablation.md  Full prompt 2x2: ranking robustness, the language main effect,
                            and what the old 0.6227 figure means
results/sota_comparison.md  Comparison against six existing Cantonese models (two protocols
                            plus paired tests)
results/ablation_clean_data.md
                            Ablation: training on only the 27,207 public examples does better
results/quantization.md     Measured fp16 / int4 / int8 / Ascend-native W8A8 variants, and
                            how to read the noise-dominated columns
results/v2_results.md       v2 (Qwen3-8B + LoRA): training, merge health check, evaluation
results/v1_vs_base.md       v1 against the base model
```

Paths in the scripts are written as `$LAB_ROOT` / `$HOME` — replace them with paths for the
local environment before running the scripts.

---

## About the evaluation tasks

| Task | How | Why this way |
|---|---|---|
| **HKMMLU (main protocol)** | **Official zero-shot prompting**: ask for the letter directly, greedy decoding, full 26,368 questions | Prompting *is* the official standard (see the dataset README), and **only this protocol is comparable to the official leaderboard**. Unparseable outputs are **recorded separately as `unparsed_rate` and scored as wrong, never silently dropped** |
| HKMMLU (secondary) | Compare the logprobs of the four letter tokens `A/B/C/D` | Removes instruction-following ability as a confounder and measures knowledge alone. **But it is not directly comparable to the official leaderboard**, so it is secondary only |
| **Official translation tasks** | yue→zh and zh→yue, 2,000 items each, chrF + BLEU(zh), **also reported after script normalisation** | Multiple choice cannot measure "can it produce Cantonese"; this can. **Read the two directions separately** — producing and understanding are different abilities |
| **Written-Cantonese purity** | Ratio of Cantonese function words (嘅係唔咗喺佢哋…) to Mandarin ones (的是不了在他們…) | Measures whether the output is oriented toward Cantonese or Mandarin. **Interpret together with degeneracy** to avoid over-interpreting a single function word in a degenerate output |
| **Degeneracy detection** | Hit rate of `(.{6,})\1{2,}` under greedy decoding | A veto criterion: is the model still producing language at all |
| **Perplexity** | Report both PPL and **bits-per-char** | Tokenizers differ across models, so PPL is not directly comparable; bits-per-char is |

---

## Data rebuild

Four public sources are merged (the authors' own Cantonese instruction set, Cantonese CoT,
Cantonese dialogue, and a Cantonese–Mandarin parallel corpus), normalized to one script with
opencc `s2hk`, deduplicated at two levels (whole record + same instruction), then split
stratified by source with a fixed `seed=42`: **71,202 examples**
(train 68,202 / dev 1,000 / test 2,000).
Determinism is verified: re-running on two different machines gives identical per-source counts,
dedup counts, and split sizes.

⚠️ **The merged corpus is not redistributed.** One of the four sources has **no license
statement at all** and another is **AGPL-3.0**, together 62% of the data. What is public is the
script; the 27,207 redistributable examples (own data + CC0) are published separately as
`v2-clean/` in
[`Zeteng/cantonese-llm-data`](https://huggingface.co/datasets/Zeteng/cantonese-llm-data).
See [`docs/data-licensing.md`](docs/data-licensing.md).

Two relevant statistics are:
- **50,606 records** were script-normalized during the build — Simplified/Traditional mixing in
  public Cantonese corpora is widespread, not marginal;
- Cantonese purity: median 1.0, mean 0.824. **Purity is measured but never used as a filter** —
  an overly aggressive cleaning threshold may remove samples of research value. The threshold
  can be varied with `--min-yue-ratio` for ablation.

---

## Model weights

All at **https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4**
(**gated repository, access by application**; licensed CC BY-NC-ND 4.0):

| Subfolder | Size | Notes |
|---|---|---|
| `v2-qwen3-8b/` | 15.3 GiB | ✅ Recommended version — bf16, already merged |
| `v2-ascend-w8a8-dynamic/` | **11.1 GiB** | ✨ **Ascend 910C native W8A8**, loads with `vllm-ascend --quantization ascend`, passed the non-inferiority gate |
| `v2-qwen3-8b-int4/` | 5.7 GiB | NF4, fits one 8 GB GPU, **CUDA only** |
| `v2-clean-data/` | 15.3 GiB | **The public-data-only ablation**, reproducible from the published data and scripts; under the official protocol it beats the full-data model |
| `v2-qwen3-8b-lora/` | 349 MB | LoRA adapter |

**The 30B-A3B run lives in a separate repository:** **https://huggingface.co/Zeteng/yue-qa-qwen3-30b-a3b** (gated, CC BY-NC-ND 4.0):

| Path | Size | Notes |
|---|---|---|
| `adapter/` | 102 MB | LoRA adapter (attention-only, r=32). The base is public weights, so the merge is reproducible — the criteria are 18867 / 192 / 0.15516 in `merge_check.json` |
| Merged bf16 weights | 57 GiB | To be released |

The dataset is at **https://huggingface.co/datasets/Zeteng/cantonese-llm-data** (also gated).

---

## License

Code is MIT. Evaluation datasets such as HKMMLU remain under their own licenses.
