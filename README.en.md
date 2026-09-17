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

## Latest results (2026-09-17)

**1. HKMMLU, official protocol, full 26,368 questions: we are 4th of 8.**

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

Paired McNemar against the leader: **−5.03pp** (95% CI [−5.68, −4.37], p = 1.6e-51).
The difference is statistically significant.

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

**6. An Ascend-native W8A8_DYNAMIC build is published and passed a non-inferiority gate.**
Dynamic W8A8 via msmodelslim (11.1 GiB), loads straight into `vllm-ascend`. Full 26,368
questions, same-protocol bf16 control, paired per question: Δ **−0.46pp**, 95% CI
[−0.93, +0.00], upper confidence bound on the loss 0.93pp ≤ tolerance δ = 2.0pp.
**The static W8A8 variant is not published**: although it is 64% faster, its output is severely
degraded because it does not emit EOS and continues generating until the token limit. See
[`results/quantization.md`](results/quantization.md).

---

## Main findings

**1. Without a validation set, an SFT training-loss curve proves nothing.**
In the run that went wrong, loss fell from 1.59 to 0.25 within 200 steps and then flattened for
6400 more — it looks like convergence, but it had actually collapsed into a degenerate solution
that emits repeated strings. `eval_strategy="no"` makes those two cases look identical in the log.

**2. The weight delta `‖ΔW‖/‖W‖` is the fastest LoRA health check.**
254 of 310 tensors were untouched (consistent with `target_modules`), but `q_proj` in the last
5 layers reached a relative difference of **0.50–1.06** — a healthy r=8/alpha=16 LoRA should sit
at 0.01–0.05. The divergence was structured: confined to the query projections in the layers
closest to the output.

**3. Re-attaching the unmerged adapter to the pristine base separates "training broke" from
"merging broke" in one step.** Both degenerated ⇒ merging was not the root cause.

**4. Moving one CUDA evaluation script to Ascend took 4 lines of code changes; accuracy matched
(6 questions out of 3300) and throughput differed by 3.6×.**
Bringing up an OpenAI-compatible server with `vllm-ascend` also works (0.6B / TP=1,
**ready in 65 s**), but two issues produce error messages that do not identify the underlying cause:
a missing **NNAL/ATB** (`libatb.so`, which is not in CANN and must be sourced separately), and
**`set -u`, which makes the script exit silently while sourcing the Ascend environment**.

**5. A string metric that spans Simplified and Traditional must list both character forms.**
A Mandarin-marker list written only in Simplified silently failed to match anything in a
Traditional corpus, inflating Cantonese purity to 0.964 (true value 0.824).

**6. Before scoring a thinking-capable model on its "first token", confirm the first token is
actually the answer slot.** Multiple choice is scored by comparing the logprobs of A/B/C/D, but
Qwen3 has thinking on by default and its first token is `<think>` — so what gets measured is
"it would like to think first". The base therefore scored only 0.2888 (near chance); with
thinking off it is 0.5700. See [`results/v2_results.md`](results/v2_results.md).

**7. "Weights re-merged locally" and "the weights that were actually evaluated" are only
assumed equivalent — verify it.** When moving 16 GB is impractical, moving just the 349 MB
adapter and re-merging locally is the right call, but it has to be verified byte-for-byte with
authenticated HTTP range requests (3 tensors sampled, 32,768 bf16 elements each, 100%
identical). The same proxy may download at 8.7 MB/s and upload at 50 KB/s — **measure each
direction separately** — and **estimate remaining time from bytes-transferred ÷ elapsed, never
from an instantaneous per-second rate**. HF's Xet backend
fails whole batches on large files with `xorb not found`; `HF_HUB_DISABLE_XET=1` works around it.
See [`docs/publishing-logistics.md`](docs/publishing-logistics.md).

**8. A sentence added to a prompt "so the parser works" becomes an experimental variable.**
To make the answer regex fire, the prompt appended "write 答案：X on the last line".
The motivation was purely engineering, but it also changed the model's action space (it may
now reason before answering), which made it an **undeclared experimental variable**. Measured
on the full set with pairing: base +5.51pp, Thinking +3.98pp, this project +3.00pp — while
the three rivals that already had zero parse failures went **−0.14 to −0.93pp**.
**Whether a protocol is usable depends on whether it is equally neutral toward every model
being compared, not on whether it appears fair.** The relevant reporting target is
**ranking stability across prompts**.
See [`results/prompt_ablation.md`](results/prompt_ablation.md).

**9. Significance is the wrong thing to use as a tolerance threshold — larger n always makes
differences significant.**
The same quantized variant measured −1.48pp on 3,300 questions (p = 0.025, significant) and
−0.46pp on the full 26,368 (p = 0.052, not significant). A rule of "publish only if not
significant" would make the publication decision depend counterintuitively on sample size. The correct form
is a **non-inferiority test**: fix an acceptable loss ceiling δ, then require the **95% upper
confidence bound on the loss to be ≤ δ**. And remember both models answered the *same*
questions, so **pairing is mandatory** — using each accuracy's own standard error inflates the
variance (0.87pp vs 0.65pp paired) and reports a real regression as noise.
**Also check whether the verdict is sensitive to δ rather than trying to argue δ perfectly**
(`eval/gate_check.py` prints this).

**10. Drawing a universal conclusion ("only X behaves this way") from M of N rows will bite
you.**
Once all 8 models were included, the base model and the Thinking variant both gained more than
we did. The three initially missing rows happened to be the three most extreme behaviours (two
chain-of-thought models plus the base), and those are exactly the ones that need format hints
the most.

**11. Character-level metrics (chrF/BLEU) on cross-script tasks must also be reported after
normalisation.**
The official yue→zh references are Simplified and our output is Traditional; chrF is
character-level, so sentences whose register was fully converted (喺→在, 同→和, 係→是, 嘅→的)
scored near zero. Raw 23.95, normalised **52.81** — a **28.86-point** difference. Report both:
the raw figure measures "did you follow the instruction", the normalised one measures "did you
convert the register", and **the gap between them is what orthographic compliance is worth.**
Normalise with character-only `zh-hans`, not `zh-cn` which also substitutes vocabulary.

More detail in [`docs/v1-postmortem.md`](docs/v1-postmortem.md) and
[`docs/ascend-notes.md`](docs/ascend-notes.md).

---

## Repository layout

```
eval/eval_yue.py            Three objective tasks: HKMMLU multiple choice (logprob scoring) /
                            written-Cantonese purity + degeneracy detection / in- and
                            out-of-distribution perplexity (with cross-tokenizer bits-per-char)
eval/diag_lora_merge.py     LoRA health check: base / base+adapter / merged three-way
                            comparison, plus per-module ‖ΔW‖/‖W‖
data/build_yue_sft.py       Multi-source Cantonese SFT rebuild: merge → opencc script
                            normalization → two-level dedup → source-stratified train/dev/test
                            split (split before training, so no leakage)
train/train_yue_lora.py     LoRA SFT: validation set + early stopping + loss on the assistant
                            span only
train/merge_lora.py         Single-process LoRA merge with an automatic ‖ΔW‖/‖W‖ health check
train/merge_lora_stream.py  Shard-by-shard streaming merge, ~5 GB peak RAM (for low-memory machines)
deploy/npu_serve_test.sh    Bring up and validate a vllm-ascend OpenAI-compatible server on
                            Ascend (health check / chat / throughput)
deploy/ascend.md            Ascend deployment guide (**also published under deploy/ in the HF
                            model card**): bf16 conversion-free and native W8A8, five packaging
                            issues, and the torchair negative result
deploy/cuda.md              CUDA deployment guide (also mirrored to HF): bf16 / int4 /
                            the public-data ablation
docs/v1-postmortem.md       How v1 diverged and how the root cause was found
docs/ascend-notes.md        Ascend 910C porting notes, including the negative result on
                            graph-mode optimization
docs/data-licensing.md      Why the merged corpus cannot be redistributed (two of four sources
                            cannot), and what that implies for the weights' license
docs/publishing-logistics.md Moving 16 GB of weights from an isolated cluster to public
                            hosting: measured bandwidth per hop, the adapter-only approach,
                            byte-level verification
results/v1_vs_base.md       v1 against its base
results/v2_results.md       v2 (Qwen3-8B + LoRA) training, merge health check, and evaluation
results/sota_comparison.md  Head-to-head against 6 existing Cantonese models
                            (two protocols + paired tests)
results/ablation_clean_data.md
                            Ablation: training on only the published 27,207 examples
                            gives a *better* model
results/quantization.md     Measured fp16 / int4 / int8 / Ascend-native W8A8 variants,
                            and how to interpret noisy columns
eval/eval_hkmmlu_official.py  HKMMLU under the **official protocol** (zero-shot prompting,
                            generative) on the full set, plus the official yue↔zh translation
                            tasks (chrF/BLEU, also reported after script normalisation) and
                            written-Cantonese purity; `--mc-style` drives the prompt ablation
eval/prompt_ablation.py     Prompt ablation: strict vs scaffold, paired per question, and
                            how much of the gain is merely recovered parse failures
eval/prompt_grid.py         Prompt 2×2 (language × format scaffold): main effects,
                            interaction, unparsed rate across all four arms
eval/gate_check.py          Release gate: **non-inferiority test** against a same-protocol
                            control, paired per question, plus a δ-sensitivity self-check
eval/summarize_trans.py     Translation summary: raw chrF alongside script-normalised chrF
deploy/quant_msmodelslim.py W8A8 via Ascend's own msmodelslim (dynamic/static, anti-outlier
                            as a separate pass)
deploy/shard_safetensors.py Split one large safetensors into <5 GB shards + index
                            (works around HF's git-lfs limit)
deploy/verify_shards.py     Verify the shards are byte-identical to the original, tensor by tensor
results/standard_eval.md    **The main results document**: 8 models × 3 tasks, full-set paired
                            tests
results/prompt_ablation.md  Full prompt 2×2: ranking robustness, the language main effect,
                            and clarification that the earlier 0.6227 value is not a result
```

Paths in the scripts are written as `$LAB_ROOT` / `$HOME` — replace them with paths for the
local environment before running the scripts.

---

## About the evaluation tasks

All three tasks are **judge-free** and recomputable:

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

The dataset is at **https://huggingface.co/datasets/Zeteng/cantonese-llm-data** (also gated).

---

## License

Code is MIT. Evaluation datasets such as HKMMLU remain under their own licenses.
