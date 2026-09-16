# 量化变体实测（fp16 / int4 / int8）

同一份 bf16 权重出发，**全部关闭思维链、与 bf16 基线同口径**。
HKMMLU 取 66 个 config 各 50 题（n=3300，标准误约 0.85pp）；生成类 30 条粤语提问、贪心解码。

| 变体 | 体积 | HKMMLU | HKDSE | 书面粤语纯度 | 退化率 | 中位长度 | PPL 留出 | PPL 粤语维基 |
|---|---|---|---|---|---|---|---|---|
| **bf16（基线）** | 15.3 GiB | 0.6094 | 0.6000 | 0.9591 | 0.000 | 109 | **9.97** | **13.57** |
| fp16 | 16.0 GiB | 0.5945 | 0.5867 | 0.9562 | 0.067 | 147 | **9.97** | **13.57** |
| **int4（NF4+double quant）** | **5.7 GiB** | 0.5909 | 0.5867 | 0.9473 | 0.000 | 87 | 10.50 | 14.49 |
| int8（LLM.int8()） | 8.9 GiB | 0.6064 | 0.6067 | 0.9256 | 0.033 | 100 | 10.06 | 13.79 |

int4/int8 由 `bitsandbytes` 产出（**免校准**），fp16 是直接 cast。

## 怎么读这张表

**只有困惑度那两栏值得当结论。** 它是在 300 段文本上逐 token 算的，样本量足够：
int4 代价最大（留出 +0.53、维基 +0.93），int8 很小（+0.09 / +0.23），**fp16 为零**。

**退化率那一栏是噪声。** 只有 30 条提问，0.067 = 2 条、0.033 = 1 条。
不能据此说「fp16 比 int4 更容易退化」——这是我差点写进报告的一个过度解读。

**HKMMLU 的 1~2pp 差异也在噪声边缘。** n=3300、标准误 0.85pp，fp16 的 −1.49pp 约 1.75σ。
配合 fp16 的困惑度与 bf16 **完全相同**（溢出扫描显示安全余量 1927×，对这份权重 fp16 基本无损），
合理解释是：选择题用 argmax 比较 A/B/C/D 的 logprob，接近平手时会被数值噪声翻转。
**同一个模型在「无损精度转换」下 HKMMLU 掉 1.5pp，正是这个指标的噪声下限的实测值。**

## fp16 必须先扫溢出

bf16 的指数位和 fp32 一样宽（±3.4e38），fp16 只到 ±65504。直接 cast 可能把大值变成 `inf`，
模型直接坏掉，而且**坏得很隐蔽**（权重看起来转好了）。所以 `train/cast_fp16.py` 先扫一遍
全模型绝对值上限再决定：本例上限给出 **1927× 安全余量**，通过；超了就中止、不产出。

## bitsandbytes 还是 llmcompressor

原计划用 `llmcompressor` 产出 compressed-tensors 的 W8A8 / W4A16——那是 `vllm-ascend`
原生认的格式。但**它全部版本都要 `torch>=2.10`**，而目标训练机是 torch 2.6，
`pip` 在 `torch<=2.6` 约束下把 33 个版本全试了一遍给出 `ResolutionImpossible`。

所以退到 `bitsandbytes`（只依赖 torch+numpy，且 NF4/int8 **免校准**）：

| | llmcompressor | bitsandbytes |
|---|---|---|
| 依赖 | torch≥2.10 | torch+numpy |
| 校准集 | int8 需 512 条 | 不需要 |
| 升腾 (`vllm-ascend`) | ✅ w8a8 / w4a8 | ❌ 仅 CUDA |

免校准其实消掉了一个真实风险：量化工具默认用英文 wikitext 校准，
而这个模型唯一的卖点就是粤语语域——**用错校准集会把它量化掉**。
需要升腾原生格式时，只能在一台 torch≥2.7 的机器上做（见 `docs/ascend-notes.md`）。

## 升腾原生 W8A8：做了，不发——两重失败

用 `llmcompressor` 的 `SmoothQuantModifier` + `GPTQModifier(scheme="W8A8")`，
256 条**粤语**校准样本（取自训练完全没用过的 test 切分），在升腾节点上跑了 **6 小时 43 分**。
产物结构完好（651 张量、索引与实际一致、两个分片都没截断、252 个 `weight_scale`
恰好等于 36 层 × 7 投影）。但两件事都不成立：

### 失败一：vllm-ascend 不吃这个格式

```
KeyError: 'model.embed_tokens.weight'
  vllm_ascend/quantization/quant_config.py:170
  is_skipped = self.quant_description[prefix + '.weight'] == "FLOAT"
```

真因不是配置写错，是**格式体系不同**：

```python
# vllm_ascend/quantization/quant_config.py
@classmethod
def override_quantization_method(cls, hf_quant_cfg, user_quant):
    if torch.npu.is_available():
        return ASCEND_QUANTIZATION_METHOD     # 只要有 NPU 就无条件劫持
...
@classmethod
def get_config_filenames(cls): return ["quant_model_description.json"]
```

**只要检测到 NPU，vllm-ascend 就无条件把任何量化方法换成它自己的**，而它只认
`quant_model_description.json`（升腾 ModelSlim 格式，要求**每个**权重前缀都有一条记录，
不量化的写 `"FLOAT"`）。`llmcompressor` 出的 compressed-tensors 把信息写在 `config.json`
的 `quantization_config` 里，没有这个文件，于是查 `model.embed_tokens.weight` 直接 KeyError。

**补一个描述文件也不行**——张量布局就不一样：compressed-tensors 是
`weight` + `weight_scale`，升腾 W8A8 要 `deq_scale` / `quant_bias`。

### 失败二：质量掉到基座以下

| | HKMMLU | HKDSE | 纯度 | 退化 | 中位长度 | PPL留出 | PPL维基 | 体积 |
|---|---|---|---|---|---|---|---|---|
| bf16 | **0.6094** | **0.6000** | **0.9591** | 0.000 | 109 | **9.97** | **13.57** | 15.3 GiB |
| int4 (NF4) | 0.5909 | 0.5867 | 0.9473 | 0.000 | 87 | 10.50 | 14.49 | 5.7 GiB |
| **W8A8** | **0.5427** | **0.5022** | 0.9154 | 0.067 | 67 | 10.85 | 15.56 | 8.8 GiB |
| 基座 Qwen3-8B | 0.5700 | 0.5400 | 0.0924 | 0.000 | 194 | 26.72 | 21.69 | 15.3 GiB |

> 这一组是抽样口径（n=3300）下测的，与上表可比。全量口径的数字见
> [`v2_results.md`](v2_results.md) 的口径更正一节。

**W8A8 的 HKMMLU 0.5427 低于基座 0.5700**，比 bf16 掉 6.67pp——标准误 0.85pp，
**7.8σ，远不是噪声**。HKDSE 掉 9.78pp。按发布闸门（纯度≥0.85 且 退化≤0.10 且
HKMMLU≥0.55）**不通过，不发**。

### 反直觉的地方，和它的解释

**int4 只掉 1.85pp，W8A8 掉 6.67pp——尽管 W8A8 位宽更高、体积还大 55%。**

差别不在位宽，在**动没动激活**：int4 (NF4) 是权重-only，激活保持 bf16；
W8A8 权重和激活都量化成 int8。旁证：bnb 的 int8（也是权重为主）PPL 只 +0.09/+0.23，
这个 W8A8 PPL +0.88/+2.0，差一个数量级。**激活量化才是难的那部分，SmoothQuant 也没救回来。**

### 这直接决定了升腾侧该用哪个权重

`vllm_ascend/quantization/` 里支持的全部方案：

```
w8a8.py   w8a8_dynamic.py   w4a8_dynamic.py   w4a4_flatquant_dynamic.py
→ W8A8 / W8A8_DYNAMIC / W4A8_DYNAMIC / W4A4_FLATQUANT_DYNAMIC
```

**没有任何 weight-only 选项。** 没有 W8A16，没有 W4A16。
**升腾上每一种能用的量化方案都要量化激活**——都踩在实测最伤的那一刀上。

所以「升腾侧用哪个权重」的答案是**已经验证过的 bf16**。
这不是没做出来，是做了、量出来不划算。

