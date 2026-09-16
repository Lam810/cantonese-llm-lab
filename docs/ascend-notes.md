# 把一套 CUDA 评测脚本搬到昇腾 910C 上：实际要改的东西

> 硬件：Ascend 910C（`Ascend910` 系列，2 die/卡，**64 GB HBM / die**），aarch64 openEuler，CANN 8.5.0。
> 软件：`torch 2.7.1` + `torch_npu 2.7.1`（+ `vllm-ascend 0.11.0`，本文没用到）。

## 结论

同一份脚本、同一份数据、同一个模型（Qwen3-0.6B），**代码只改了 4 行**：

| | HKMMLU (n=3300) | 粤语纯度 | 耗时 |
|---|---|---|---|
| H100 80GB | 0.31061 | 0.014502 | 116 s |
| **Ascend 910C** | **0.30879** | **0.014543** | **412 s** |

准确率差 0.18pp（3300 题里 6 题），是 bf16 在不同硬件上的数值差异，不是实现错误。
**吞吐差 3.6 倍**——这是 eager 模式逐算子下发的代价，没有走图模式/融合算子。

## 代码改动：就这 4 行

```python
import torch
try:                      # 关键：import torch_npu 之后 torch.npu 才存在
    import torch_npu      # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except Exception:
    _HAS_NPU = False

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if _HAS_NPU: return "npu"
    return "cpu"
```

然后把 `device_map="cuda"` 换成 `device_map=pick_device()`。`transformers` 的
`from_pretrained` / `generate` / 前向全部照常工作，不需要改模型代码。

## 起 OpenAI 兼容服务（vllm-ascend）

`vllm-ascend 0.11.0` + `vllm 0.11.0`（源码 editable 安装）在 910C 上能直接起标准的
`vllm.entrypoints.openai.api_server`，`/health`、`/v1/models`、`/v1/chat/completions` 全部照常。
Qwen3-0.6B、TP=1、`--max-model-len 4096`，**从进程启动到 /health 通过约 65 秒**。

但有两个坑会让它在引擎初始化阶段直接死掉，而且报错都指不到真正的原因：

### 坑 A：`libatb.so: cannot open shared object file` —— 缺的是 NNAL，不是 vLLM

```
ERROR [patch_core.py:58] OSError: libatb.so: cannot open shared object file: No such file or directory
ERROR [patch_core.py:58] OSError: [Errno None] Please check that the nnal package is installed.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

vllm-ascend 依赖 **NNAL / ATB**（Ascend Transformer Boost），它**不在 CANN 里**，是单独的包，
而且 **`ascend-toolkit/set_env.sh` 不会把它加进库路径**。装过之后还得单独 source：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <nnal-install-path>/nnal/atb/set_env.sh     # 这一行才是 libatb.so 的来源
```

排查时注意：`libatb.so` 的实际路径埋得很深
（`.../nnal/atb/<ver>/atb/cxx_abi_{0,1}/lib/libatb.so`），
**`find -maxdepth 6` 扫不到**——我第一次就是这样误判成"没装"，实际早就装好了。
`set_env.sh` 会按环境自动选 `cxx_abi_0/1` 并导出 `ATB_HOME_PATH`，确认一下这个变量非空即可。

### 坑 B：`set -u` 会让脚本在 source 环境时直接退出

昇腾的 `set_env.sh` 引用了一批可能未定义的变量：

```
set_env.sh: line 48: LD_LIBRARY_PATH: unbound variable
set_env.sh: line 48: PYTHONPATH: unbound variable
set_env.sh: line 31: CMAKE_PREFIX_PATH: unbound variable
atb/set_env.sh: line 43: ZSH_VERSION: unbound variable
```

在 `set -eu` 的作业脚本里，这会让整个脚本**在第一行 source 处静默退出**，
Slurm 还会记成正常结束。要么别用 `set -u`，要么先把这几个变量预设成空：

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}
```

同理，作业脚本里只写 `set -x` 不写 `set -e` 时，中间命令失败也会一路跑到最后的 `echo DONE`，
**Slurm 记成 `COMPLETED 0:0`**——看状态码会以为跑成功了。

## 四个实际踩到的坑

**① 登录节点上 `import torch` 直接抛异常。**

```
RuntimeError: Failed to load the backend extension: torch_npu.
You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
```

因为登录节点没有 NPU 设备和驱动，而 `torch_npu` 装了之后会被 torch 自动加载。
**这不是环境坏了**，进到分配了 NPU 的计算节点里就正常。想在登录节点做纯 CPU 的检查，
就设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。同理 `npu-smi` 也只在计算节点上有。

**② 缺 `accelerate` 的报错信息具有误导性。**

```
ValueError: Using a `device_map` ... requires `accelerate`. You can install it with `pip install accelerate`
```

照着装会出事：`pip install accelerate` 在 aarch64 上会去解析 `torch` 依赖，
**开始下载 542 MB 的 `nvidia_cublas` 和 `cuda_bindings` 轮子**——在一台昇腾机器上。
正确做法是 `pip install --no-deps --target=<dir> accelerate`（它真正需要的
`psutil`/`safetensors`/`huggingface_hub` 环境里本来就有），装完 7 MB。

**③ 不要往现成的 vLLM 环境里装东西。**
用 `--target=<自己的目录>` + `PYTHONPATH` 追加，别污染别人调好的 `torch_npu`/`vllm-ascend` 组合。

**④ CANN 目录的 owner 警告可以忽略。**
每次 `import torch_npu` 都会刷一串
`UserWarning: The /usr/local/Ascend/cann-x.y.z ... owner does not match the current owner.`，
这是共享集群上 CANN 装在 root 下的正常现象，不影响运行，但会把 stderr 淹掉——
调试时记得 `grep -v "does not match the current owner"`。

## 作业脚本骨架

```bash
#SBATCH -p <npu-partition>
#SBATCH --gres=npu:2          # 一张卡 = 两个 die，最小申请单位通常是 2

source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info                   # 只在计算节点上有
PYTHONPATH=$LAB_ROOT/pylibs $VENV/bin/python $LAB_ROOT/code/eval_yue.py --model ... 
```

上机前的 30 秒自检：

```python
import torch, torch_npu
print(torch.npu.is_available(), torch.npu.device_count())
x = torch.randn(512, 512).npu(); print((x @ x).sum().item())
```

## 8B 模型的实测（v2，bf16 未量化）

`Qwen3-8B` 基座 + LoRA 合并后的 16.4 GB bf16 权重，910C 单 die、TP=1、`--max-model-len 4096`：

| | |
|---|---|
| 服务就绪 | **70 秒** |
| 对话 | 正常，输出自然书面粤语 |
| 吞吐 | 8 并发 **286 tok/s**（⚠️ 8 个请求用的是同一个 prompt，有前缀缓存加成，不是冷启动数字） |

一个要写进 card 的观察：**贪心解码（temperature=0）在列举类问题上会重复**
（「維多利亞港係香港嘅一個著名景點」连出 7 次），同一份权重在 CUDA 上用
`transformers` 跑 30 条提问时退化率是 0。生成建议用 `temperature>0`。

### 又一个版本兼容坑：transformers 5.x 存的分词器，4.x 读不了

合并是在 transformers **5.6.2** 上做的，部署环境是 **4.57.1**，服务启动直接崩：

```
AttributeError: 'list' object has no attribute 'keys'
  ... _set_model_specific_special_tokens(special_tokens=self.extra_special_tokens)
```

5.x 把 `extra_special_tokens` 写成 list，4.x 期望 dict；而且 5.x **只存 `tokenizer.json`**，
不再存 `vocab.json` / `merges.txt` / `special_tokens_map.json`，老工具链会缺文件。

**正确修法不是打补丁，而是把上游基座的完整分词器文件覆盖进合并产物**——LoRA 根本没动分词器。
顺带注意 chat template 的存放位置：上游 Qwen3 把它嵌在 `tokenizer_config.json` 里
（4168 字符），而 transformers 5.x 改成额外写一个 `chat_template.jinja`。
**真正的失败模式不是"两者并存"，是"两者都没有"**——我按"删掉 .jinja"这条做过一次，
而当时的 `tokenizer_config.json` 恰好是 5.x 写的精简版（693 字节、不含模板），
结果模型彻底没有对话模板。正确做法是先确认 `tokenizer_config.json` 里有 `chat_template` 字段，
没有就从上游基座补齐；两者并存且内容一致是安全的（≥4.49 优先读 `.jinja`）。

### 一个纯属自找的坑

我给这个 serving 脚本打过 NNAL 和 `set -u` 两个补丁，**但只打在集群那份上**；
后来改了本机副本再 scp 覆盖过去，**把两个修复都冲掉了**，于是 `libatb.so` 的错误原封不动地复现。
**脚本必须有唯一源头**，远端热修完要立刻同步回本地，否则下一次同步就是一次回退。

## 图模式优化：三个配置的实测，结果是负的

同一个 8B 模型（v2 bf16 合并权重）、24 条**互不相同**的 prompt 并发。
用同一条 prompt 会吃到前缀缓存、吞吐虚高——上面那个 286 tok/s 就是这个毛病，不要跨表比。

| 配置 | 结果 |
|---|---|
| **A 默认**（ACL Graph piecewise） | 24/24 成功，1298 tok / 2.8 s = **462.9 tok/s** |
| **B `+DENSE_OPTIMIZE` `+MLP_OPTIMIZE`** | 24/24 成功，1287 tok / 2.9 s = **447.8 tok/s** |
| **C torchair 图模式** | **起不来**，EngineCore 初始化失败 |

**B 没有增益**（−3.3%，这个样本量下是噪声）。`VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1`
和 `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=1` 对 Qwen3-8B 这种稠密模型没起作用。

**C 的报错指不到真因：**

```
TypeError: RotaryEmbedding.forward_native() takes from 3 to 4 positional arguments but 5 were given
RuntimeError: Engine core initialization failed.
```

真因在 `vllm_ascend/ascend_config.py`：

```python
TORCHAIR_MODEL_LIST = ["deepseek", "pangu", "kimi_k2", "qwen"]

def _check_torchair_supported(model_type: str):
    for supported_model in TORCHAIR_MODEL_LIST:
        if supported_model in model_type.lower():   # 子串匹配
            return True
    return False
```

Qwen3 dense 的 `model_type` 是 `qwen3`，`"qwen" in "qwen3"` 成立，**这道门把它放过去了**；
但 `vllm_ascend/torchair/models/` 里只有 `qwen2.py` 和 `qwen3_moe.py`——**没有稠密 Qwen3 的实现**，
于是落到通用路径上撞签名。这是"支持列表用子串匹配"造成的假阳性放行：
白名单说支持，代码里没有。查这类问题要去看白名单旁边那个目录里到底有没有对应文件，
而不是信白名单。

**结论：对 Qwen3-8B 稠密模型，昇腾这边目前没有可用的图模式加速手段，默认的
ACL Graph piecewise 已经是最好的配置。** 3.6 倍那个差距没有被压下来。

## 昇腾原生量化：llmcompressor 根本没用上 NPU

做 compressed-tensors 的 W8A8 时，日志第一行就写着：

```
dispatch_for_sequential | WARNING - CUDA/XPU is not available! Compressing model on CPU instead
```

`llmcompressor` 只认 CUDA/XPU，不认昇腾，于是整个 SmoothQuant + GPTQ 全在 CPU 上跑：
**8B 模型、256 条校准样本，耗时 6 小时 43 分**（SmoothQuant 约 2 小时 10 分，
GPTQ 约 4 小时 30 分）。NPU 在量化阶段完全空转，只有最后起服务验收时才用得上。
要排这种作业，按 CPU 核数而不是按卡数估时间。

## 昇腾侧量化：能用的方案全都要量化激活

`vllm_ascend/quantization/` 里支持的全部方案：

```
w8a8.py   w8a8_dynamic.py   w4a8_dynamic.py   w4a4_flatquant_dynamic.py
→ W8A8 / W8A8_DYNAMIC / W4A8_DYNAMIC / W4A4_FLATQUANT_DYNAMIC
```

**没有任何 weight-only 选项**（没有 W8A16、没有 W4A16）。而实测下来，
**动激活是伤得最重的那一刀**：int4（权重-only，NF4）HKMMLU 只掉 1.85pp，
W8A8（权重+激活）掉 **6.67pp**，直接低于基座——尽管位宽更高、体积还大 55%。

还有一件事：`llmcompressor` 出的 compressed-tensors **vllm-ascend 根本不认**。
只要检测到 NPU，`override_quantization_method` 就无条件劫持成昇腾自己的方法，
而它只读 `quant_model_description.json`（ModelSlim 格式，每个权重前缀都要有记录）。
补描述文件也不行——张量布局都不一样。

**结论：昇腾侧目前该用的是已经验证过的 bf16。** 详见
[`../results/quantization.md`](../results/quantization.md)。

### 顺带：5.x 存的分词器坑，在别人的模型上也一样

上面那条「transformers 5.x 存的分词器 4.x 读不了」不是我们 merge 的特例。
从 HF 拉的 6 个粤语对照模型里有 **2 个**是同一个毛病：`tokenizer_config.json`
只有 664 字节、`extra_special_tokens` 是 list、chat template 在独立的 `.jinja` 里。

**先确认影响范围再动手**：评测集群的 transformers 是 5.6.2，读得了，完全不受影响；
只有昇腾侧 vllm 环境（4.57.1）会崩。修的时候必须用模型**自己那份** chat template
——模板会改变输出行为，拿别的模型的顶上去，横向对比就不成立了。

## 还没做的

- **训练侧**。本文只验证了推理与量化。
- **msmodelslim（昇腾自家量化工具）的实测**。它出的才是 vllm-ascend 认的格式，
  而且它认 NPU（`dev_type='npu'`），不像 llmcompressor 只能跑 CPU。环境已铺好、
  `Calibrator`/`QuantConfig` 已验证可导入，但按上面的证据，预期它也逃不过
  "量化激活就掉点"这一条。
- 一个真正能用的 torchair 稠密 Qwen3 路径——需要上游补 `torchair/models/qwen3.py`，
  不是配置能绕开的。
