# cantonese-llm-lab — 粤语大模型微调、评测与昇腾国产卡适配

[粵語](README.md) · **简体中文** · [English](README.en.md)

粤语大模型的**微调、评测与昇腾适配**工程笔记，附可直接复用的脚本。

这里沉淀了四件东西——一套不依赖 LLM 判官的评测、一个 LoRA 体检工具、
一份多源粤语 SFT 数据重建脚本，以及把同一套推理脚本搬上昇腾 910C 的记录。

---

## 第一署名单位与资助

**第一署名单位：粤语语料库建设与大模型评测重点实验室**
（Key Laboratory for Cantonese Corpus Construction and Large Language Model Evaluation）

**资助：** 粤语语料库建设与大模型评测重点实验室 2025 年度研究课题，**课题编号 25ZD03**。

---

## 主要结论

**1. 没有验证集的 SFT，训练 loss 曲线不能证明任何事。**
出事的那次训练，loss 在 200 步内从 1.59 掉到 0.25 然后 6400 步走平——看起来像收敛，
实际是塌进了输出重复串的退化解。`eval_strategy="no"` 让这两种情况在日志上长得一模一样。

**2. 权重差 ‖ΔW‖/‖W‖ 是最快的 LoRA 体检。**
310 个张量里 254 个完全没动（和 `target_modules` 吻合），但最后 5 层的 `q_proj`
相对差达到 **0.50 ~ 1.06**——r=8/alpha=16 的 LoRA 正常应在 0.01–0.05。
发散是结构化的：只打在最靠近输出的几层 query 投影上。

**3. 把未合并的 adapter 挂回原始基座，能一步分开「训练坏了」和「合并坏了」。**
两者都退化 ⇒ 合并不是根因。

**4. 一套 CUDA 评测脚本搬到昇腾，代码只需改 4 行；准确率对得上（3300 题差 6 题），吞吐差 3.6 倍。**
`vllm-ascend` 起 OpenAI 兼容服务也是通的（0.6B / TP=1，**65 秒就绪**），
但有两个报错指不到真因的坑：缺 **NNAL/ATB**（`libatb.so`，不在 CANN 里、要单独 source），
以及 **`set -u` 会让脚本在 source 昇腾环境时静默退出**。

**5. 跨简繁的字符串指标必须两种字形都写。**
只写简体的「普通话标记词」在繁体语料上整张表静默失配，把粤语纯度虚高成 0.964（真值 0.824）。

**6. 对带思维链的模型做「第一个 token」类打分，先确认第一个 token 是不是答案位。**
选择题比较 A/B/C/D 的 logprob，而 Qwen3 默认开思维链、第一个 token 是 `<think>`，
量到的是「它想先想一想」。基座因此只有 0.2888（近随机），关掉后是 0.5700——
**我一度把 v2 的增益写成 +31.85pp，真实值是 +3.94pp。**
详见 [`results/v2_results.md`](results/v2_results.md)。

**7.「本地重新合并的权重」和「实际被评测的权重」只是假定等价——要验。**
搬 16GB 不现实时，只搬 349MB 的 adapter、本地合并是对的；但必须用带鉴权的 HTTP range
请求逐字节比对（抽 3 个张量各 32,768 个 bf16 元素，100% 相同）。
同一个代理可能下载 8.7 MB/s、上传 50 KB/s——**两个方向要分别测**，
而且**估工期要用「已传字节 ÷ 已耗时」，不能用秒级瞬时速率**（我因此两次给出错误预估）。
HF 的 Xet 后端对大文件会报 `xorb not found` 整批失败，`HF_HUB_DISABLE_XET=1` 可绕过。
详见 [`docs/publishing-logistics.md`](docs/publishing-logistics.md)。

详见 [`docs/v1-postmortem.md`](docs/v1-postmortem.md) 和 [`docs/ascend-notes.md`](docs/ascend-notes.md)。

---

## 仓库结构

```
eval/eval_yue.py            三个客观任务：HKMMLU 选择题（logprob 打分）／书面粤语纯度
                            ＋退化检测／分布内外困惑度（含跨分词器可比的 bits-per-char）
eval/diag_lora_merge.py     LoRA 体检：base / base+adapter / merged 三路对照
                            ＋逐模块 ‖ΔW‖/‖W‖
data/build_yue_sft.py       多源粤语 SFT 数据重建：合并→opencc 简繁归一→两级去重
                            →按来源分层切 train/dev/test（先切再训，杜绝泄漏）
train/train_yue_lora.py     LoRA SFT：带验证集＋早停＋只对 assistant 段算 loss
train/merge_lora.py         单进程合并 LoRA，并自动做 ‖ΔW‖/‖W‖ 体检
train/merge_lora_stream.py  逐分片流式合并，峰值内存约 5 GB（低内存机用）
deploy/npu_serve_test.sh    昇腾上起 vllm-ascend OpenAI 兼容服务并验收（健康检查／对话／吞吐）
docs/v1-postmortem.md       v1 如何发散、如何查出根因
docs/ascend-notes.md        昇腾 910C 适配笔记，含图模式优化的负结果
docs/data-licensing.md      为什么合并语料不能再分发（四个来源里两个不能），
                            以及对权重授权的连带影响
docs/publishing-logistics.md 把 16GB 权重从隔离集群搬到公开托管站：各段实测带宽、
                            只搬 adapter 的做法、逐字节验证
results/v1_vs_base.md       v1 与基座的对照
results/v2_results.md       v2（Qwen3-8B + LoRA）的训练、合并体检与评测，
                            含一个被自己修掉的错误结论
results/quantization.md     fp16 / int4 / int8 变体实测，以及「哪几栏是噪声不能当结论」
```

脚本里的路径都写成 `$LAB_ROOT` / `$HOME`，用之前按自己的环境替换。

---

## 评测任务说明

三个任务**都不依赖 LLM 判官**，可复算：

| 任务 | 做法 | 为什么这样做 |
|---|---|---|
| **HKMMLU** | 比较 `A/B/C/D` 四个字母 token 的 logprob | 小模型 0-shot 常常不肯输出字母；比 logprob 绕开了指令遵循能力这个混淆变量 |
| **书面粤语纯度** | 粤语虚词（嘅係唔咗喺佢哋…）与普通话虚词（的是不了在他们…）的计数比 | 直接量「它到底在说粤语还是普通话」；**必须和退化率一起看**，否则一串乱码里出现一个「嘅」也会是 1.0 |
| **退化检测** | 贪心解码下 `(.{6,})\1{2,}` 命中率 | 一票否决项：模型还会不会说人话 |
| **困惑度** | 同时报 PPL 和 **bits-per-char** | 不同模型分词器不同，PPL 不可直接比，bits-per-char 可以 |

---

## 数据重建

合并四个公开来源（作者自建的粤语指令集、粤语 CoT、粤语对话、粤中平行语料），
统一 opencc `s2hk` 字形，两级去重（整条 + 同 instruction），按来源分层切分，
固定 `seed=42`：**71,202 条**（train 68,202 / dev 1,000 / test 2,000）。
确定性已验证：两台不同机器重跑，各来源条数、去重数、切分大小逐项一致。

⚠️ **合并语料不再分发**——四个来源里有一个**没有任何授权声明**、一个是 **AGPL-3.0**，
占了 62%。公开的是脚本；可再分发的那 27,207 条（自有 + CC0）单独发在
[`Zeteng/cantonese-llm-data`](https://huggingface.co/datasets/Zeteng/cantonese-llm-data) 的
`v2-clean/`。详见 [`docs/data-licensing.md`](docs/data-licensing.md)。

两个值得记下的数字：
- **50,606 条**记录在建库时被简繁归一化过——公开粤语语料的简繁混排问题是大面积的；
- 粤语纯度中位数 1.0、均值 0.824。**纯度只度量不过滤**：清洗阈值最容易把最该研究的样本丢掉，
  要做消融再用 `--min-yue-ratio`。

---

## 模型权重

微调出的权重（v2 完整 bf16、LoRA adapter、int4）在
**https://huggingface.co/Zeteng/qwen_yue_qa_finetuned_int4** 。

---

## 授权

代码 MIT。评测用到的 HKMMLU 等数据集遵循各自的授权。
