# cantonese-llm-lab — Cantonese LLM fine-tuning, evaluation, and Ascend NPU porting

[粵語](README.md) · [简体中文](README.zh-CN.md) · **English**

Engineering notes on **fine-tuning, evaluating, and porting a Cantonese LLM to Ascend NPUs**,
with directly reusable scripts.

Four things are collected here: an evaluation suite that needs no LLM judge, a LoRA health-check
tool, a multi-source Cantonese SFT data rebuild script, and a record of moving the same
inference scripts onto an Ascend 910C.

---

## Primary affiliation and funding

**Primary affiliation: Key Laboratory for Cantonese Corpus Construction and Large Language
Model Evaluation** (粵語語料庫建設與大模型評測重點實驗室)

**Funding:** 2025 annual research project of the Key Laboratory for Cantonese Corpus
Construction and Large Language Model Evaluation, **Grant No. 25ZD03**.

---

## Latest results (2026-09-16)

**1. Head-to-head against existing Cantonese LLMs: we rank first, with one loss.**
Six existing Cantonese models plus the base, on the full HKMMLU (26,368 questions),
117 generation prompts, two protocols (thinking off/on, each model scored at its own best),
with McNemar paired tests:

| Each model at its own best protocol | HKMMLU |
|---|---|
| **This project** | **0.6227** |
| `CantoneseLLMChat-v1.0-7B` | 0.6045 |
| `Qwen2-Cantonese-7B-Instruct` | 0.5932 |
| `Qwen3-8B` (base) | 0.5789 |
| `CantoneseLLM-v2.0-8B-Thinking` | 0.5485 |
| `Llama-3-Cantonese-8B-Instruct` | 0.5294 |
| `v2.0-8B-Chat-Vector-Merged` | 0.4612 |

Written-Cantonese purity and output length also come first (median **51** tokens; the
longest competitor runs to 583). **But we lose on Cantonese-Wikipedia perplexity to
`CantoneseLLMChat-v1.0-7B` (7.77 vs 13.45)** — they did continued pre-training, we only
did LoRA SFT. Put plainly: theirs is a *Cantonese language model*, ours is a
*question-answering model that writes Cantonese*.
See [`results/sota_comparison.md`](results/sota_comparison.md).

**2. The published data subset trains a better model than the full corpus.**
For licensing reasons, only 27,207 of the 71,202 examples can be redistributed. The
assumption was that the public release is a cut-down version — **the measurement says the
opposite**: training on only those 27,207 gives **+1.19pp on HKMMLU (p=6.0e-7)** and higher
purity. The 62% that cannot be redistributed was not helping; it was holding the model back.
See [`results/ablation_clean_data.md`](results/ablation_clean_data.md).

**3. Two previously published numbers are corrected.** Under the tighter protocol, the
HKMMLU gain goes from +3.94pp (3300-question sample) to **+2.29pp** (full 26,368), and
Cantonese purity from 0.9591 (30 prompts) to **0.8975** (117 prompts). The gain itself is
still highly significant (p=1.3e-14); it is simply smaller than first reported.
See the correction section in [`results/v2_results.md`](results/v2_results.md).

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
**ready in 65 s**), but two traps produce error messages that point nowhere near the real cause:
a missing **NNAL/ATB** (`libatb.so`, which is not in CANN and must be sourced separately), and
**`set -u`, which makes the script exit silently while sourcing the Ascend environment**.

**5. A string metric that spans Simplified and Traditional must list both character forms.**
A Mandarin-marker list written only in Simplified silently failed to match anything in a
Traditional corpus, inflating Cantonese purity to 0.964 (true value 0.824).

**6. Before scoring a thinking-capable model on its "first token", confirm the first token is
actually the answer slot.** Multiple choice is scored by comparing the logprobs of A/B/C/D, but
Qwen3 has thinking on by default and its first token is `<think>` — so what gets measured is
"it would like to think first". The base therefore scored only 0.2888 (near chance); with
thinking off it is 0.5700. **I briefly reported v2's gain as +31.85pp; the real value is
+3.94pp.** See [`results/v2_results.md`](results/v2_results.md).

**7. "Weights re-merged locally" and "the weights that were actually evaluated" are only
assumed equivalent — verify it.** When moving 16 GB is impractical, moving just the 349 MB
adapter and re-merging locally is the right call, but it has to be verified byte-for-byte with
authenticated HTTP range requests (3 tensors sampled, 32,768 bf16 elements each, 100%
identical). The same proxy may download at 8.7 MB/s and upload at 50 KB/s — **measure each
direction separately** — and **estimate remaining time from bytes-transferred ÷ elapsed, never
from an instantaneous per-second rate** (I gave two wrong ETAs that way). HF's Xet backend
fails whole batches on large files with `xorb not found`; `HF_HUB_DISABLE_XET=1` works around it.
See [`docs/publishing-logistics.md`](docs/publishing-logistics.md).

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
docs/v1-postmortem.md       How v1 diverged and how the root cause was found
docs/ascend-notes.md        Ascend 910C porting notes, including the negative result on
                            graph-mode optimization
docs/data-licensing.md      Why the merged corpus cannot be redistributed (two of four sources
                            cannot), and what that implies for the weights' license
docs/publishing-logistics.md Moving 16 GB of weights from an isolated cluster to public
                            hosting: measured bandwidth per hop, the adapter-only approach,
                            byte-level verification
results/v1_vs_base.md       v1 against its base
results/v2_results.md       v2 (Qwen3-8B + LoRA) training, merge health check, and evaluation,
                            including one wrong conclusion that was later corrected
results/sota_comparison.md  Head-to-head against 6 existing Cantonese models
                            (two protocols + paired tests)
results/ablation_clean_data.md
                            Ablation: training on only the published 27,207 examples
                            gives a *better* model
results/quantization.md     Measured fp16 / int4 / int8 variants, and which columns are noise
                            and must not be read as results
```

Paths in the scripts are written as `$LAB_ROOT` / `$HOME` — substitute your own environment
before running them.

---

## About the evaluation tasks

All three tasks are **judge-free** and recomputable:

| Task | How | Why this way |
|---|---|---|
| **HKMMLU** | Compare the logprobs of the four letter tokens `A/B/C/D` | Small models often refuse to emit a bare letter in 0-shot; comparing logprobs removes instruction-following ability as a confounder |
| **Written-Cantonese purity** | Ratio of Cantonese function words (嘅係唔咗喺佢哋…) to Mandarin ones (的是不了在他們…) | Measures directly whether it is writing Cantonese or Mandarin. **Must be read together with degeneracy** — otherwise a single 嘅 inside a stream of garbage also scores 1.0 |
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

Two numbers worth recording:
- **50,606 records** were script-normalized during the build — Simplified/Traditional mixing in
  public Cantonese corpora is widespread, not marginal;
- Cantonese purity: median 1.0, mean 0.824. **Purity is measured but never used as a filter** —
  a cleaning threshold is the easiest way to throw away exactly the samples worth studying. Use
  `--min-yue-ratio` if you want to ablate it.

---

## Model weights

The fine-tuned weights (full v2 bf16, LoRA adapter, int4) are at
**https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4** .

---

## License

Code is MIT. Evaluation datasets such as HKMMLU remain under their own licenses.
