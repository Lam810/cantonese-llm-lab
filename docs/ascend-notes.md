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

## 五个实际踩到的坑

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

**⑤ `ASCEND_RT_VISIBLE_DEVICES` 不能写死 die 0..7。**

在一个作业里把 N 个任务铺到 N 个 die 上，很自然会写成：

```bash
run_one $((i % 8)) "$N" "$M" &      # ← 错：假定本作业拿到的是 die 0..7
...
ASCEND_RT_VISIBLE_DEVICES=$G python ...
```

**只要同一节点上还有别的作业占着 die 0（哪怕是你自己提交的另一个作业），
Slurm 分给你的就是 die 2..9**，写死的 0..7 会落到 cgroup 外面。报错完全看不出
和别的作业有关：

```
RuntimeError: Engine core initialization failed. Failed core proc(s): {}
[ERROR] ... (PID:364433, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

**而且是八个任务全挂**，不是只挂占用冲突的那两个。作业 137921 就是这么死的：
同一个脚本前三次都跑对，第四次错，唯一的变化是 npu1-6 上还有我自己的 137919
（它只要 2 个 die）。**「同脚本跑过三次都对」不能当成脚本正确的证据——
它只说明前三次的环境恰好满足了脚本里那个没写出来的假设。**

正确做法是从 Slurm 给的列表里取，别自己编号：

```bash
DEVS=(${ASCEND_RT_VISIBLE_DEVICES//,/ })
[ ${#DEVS[@]} -eq 0 ] && DEVS=(${ASCEND_VISIBLE_DEVICES//,/ })
[ ${#DEVS[@]} -eq 0 ] && DEVS=(0 1 2 3 4 5 6 7)
NDEV=${#DEVS[@]}
echo "Slurm 给的 die: ${DEVS[*]}  (共 $NDEV)"    # 这行一定要打，出事时省半小时
run_one "${DEVS[$((i % NDEV))]}" "$N" "$M" &
```

顺带：`#SBATCH --gres=npu:N` 的 **N 只能是偶数**（一卡双芯，2 张卡按 1 张物理卡计费），
写 `npu:1` 会被直接拒：`错误: NPU 卡数只能是 2、4、6、8、...`。

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

**没有任何 weight-only 选项**（没有 W8A16、没有 W4A16）。

`llmcompressor` 出的 compressed-tensors **vllm-ascend 根本不认**：只要检测到 NPU，
`override_quantization_method` 就无条件劫持成昇腾自己的方法，而它只读
`quant_model_description.json`（ModelSlim 格式，每个权重前缀都要有记录）。
补描述文件也不行——张量布局都不一样。

### ⚠️ 这一节曾经的结论是错的

原本写的是「动激活是伤得最重的那一刀，所以昇腾侧该用 bf16」，依据是
`llmcompressor` 的 W8A8 掉 6.67pp、低于基座。**后来用 msmodelslim 重做，
`W8A8_DYNAMIC` 和 bf16 统计上无法区分（−1.48pp，标准误 0.87pp）。**

所以「掉 6.67pp」不是「量化激活」的固有代价，而是**那一条工具链的代价**——
llmcompressor 的 GPTQ + SmoothQuant 组合在这个模型上失手了，而
msmodelslim 的动态定标（按 token 算 scale）没有。

**教训：把某个工具的失败归因成某类方法的固有限制，是过度概括。**
我当时有三条相互印证的证据（int4 只掉 1.85pp、bnb int8 PPL 只 +0.09、
llmcompressor W8A8 掉 6.67pp），看起来很硬——但它们都只覆盖了一个实现。
换实现之后结论就翻了。详见下面 msmodelslim 那节和
[`../results/quantization.md`](../results/quantization.md)。

### 顺带：5.x 存的分词器坑，在别人的模型上也一样

**先确认影响范围再动手**：评测集群的 transformers 是 5.6.2，读得了，完全不受影响；
只有昇腾侧 vllm 环境（4.57.1）会崩。细节见下面 MoE 那节。

## MoE 图模式：torchair 在有实现的模型上也起不来

前面那节说稠密 Qwen3 走 torchair 会挂，因为 `vllm_ascend/torchair/models/` 里只有
`qwen2.py` 和 `qwen3_moe.py`。那么**有实现的 MoE 呢？** 用现成的
`hon9kon9ize/CantoneseLLM-v2.0-30B-A3B-Thinking`（56.9 GiB，TP=2）直接验，
不用先花 5 小时训一个：

| 配置 | 结果 |
|---|---|
| **A 默认**（ACL Graph piecewise） | ✅ 111 秒就绪，8 并发 1014 tok / 5.2s = **195.1 tok/s** |
| **C torchair** | ❌ `RuntimeError: index 0 is out of bounds for dimension 0 with size 0` |

**所以昇腾图模式这条路是完整的负结果：稠密 Qwen3 白名单放行但目录里没实现；
Qwen3-MoE 目录里有实现但仍然起不来。** 3.6 倍那个差距在这个版本上没有可用手段去压。

### 顺带两条

**5.x 分词器坑在别人的模型上一样出现。** 6 个从 HF 拉的粤语对照模型里有 **2 个**
（两个 `-Thinking`）的 `tokenizer_config.json` 只有 664 字节、`extra_special_tokens`
是 list、chat template 在独立的 `.jinja` 里。修的时候必须用模型**自己那份**模板
——模板会改变输出行为，拿别的模型的顶上去，横向对比就不成立了。

**判断昇腾作业死活不能看 `squeue`。** 一个作业在 `npu1-3` 上显示 RUNNING 跑了
1 小时 53 分，**`StdOut` 指定的日志文件从来没被创建过**。这是坏节点的典型症状，
而 Slurm 会一直显示正常运行到墙钟到期。之前在 `npu1-19` 上遇过一次。
**要看日志文件有没有被创建**，并且在 sbatch 里 `--exclude` 掉已知的坏节点。
同一个作业 `scancel` 之后还会卡在 `CG` 状态好几小时，继续占着 16 个 die。

## 昇腾原生量化：msmodelslim 走通了，但量化本身只占 20 分钟

**结论先说：`W8A8_DYNAMIC` 在昇腾上跑得起来，而且质量和 bf16 统计上无法区分
（HKMMLU −1.48pp，标准误 0.87pp）。静态 `W8A8` 完全报废。**

| | 激活量化 | anti-outlier | 服务 | 输出 |
|---|---|---|---|---|
| **W8A8_DYNAMIC** | 按 token 动态定标 | 跳过 | ✅ 100 秒就绪 | 通顺粤语 |
| W8A8 静态 | 离线校准固定 scale | 跳过 | ✅ 起得来 | **token 沙拉** |

动态定标在运行时按每个 token 算 scale，天然抗离群值，所以缺了 anti-outlier 也活得下来；
静态那版的 scale 离线定死，没有 anti-outlier 就没有任何防线。

### ⚠️ 吞吐更高的那个是坏的

```
W8A8_DYNAMIC   8 并发   182 tok / 0.6s = 312.4 tok/s     69 token 自然收尾
W8A8 静态      8 并发  1024 tok / 2.0s = 511.6 tok/s    128 token 撞上限
```

静态那版吞吐高 64%，**恰恰因为它坏了**——它从不吐 EOS，一路生成到 `max_tokens`。
两边 `serve rc=0`。**只看返回码和 tok/s 会把一个彻底报废的模型报成「更好」。必须看输出。**

### 为什么必须用 msmodelslim 而不是 llmcompressor

`vllm-ascend` 只要检测到 NPU 就无条件把量化方法劫持成自己的
（`override_quantization_method` 无条件返回 `ASCEND_QUANTIZATION_METHOD`，
即 `vllm_ascend/utils.py:43` 里的 `"ascend"`），而它只读
`quant_model_description.json`。msmodelslim 出的描述文件有 904 条，
第二条就是 `model.embed_tokens.weight = FLOAT`
——**正是 llmcompressor 那版报 `KeyError: 'model.embed_tokens.weight'` 缺的那个键。**

而且 msmodelslim 认 NPU（`dev_type='npu'`）：8B + 256 条校准 **20 分钟**。
llmcompressor 只能跑 CPU，同样的活花了 **6 小时 43 分**。

### 量化 20 分钟，环境和打包花了五层

| 层 | 症状 | 真因 |
|---|---|---|
| 1 | `rollback_names_process` 挂 | 校准张量在 CPU、`dev_type='npu'` 把模型挪去了 NPU |
| 2 | `split_module_name_get_info` IndexError | `do_smooth=True` 的 DAG 自动识别在 Qwen3 上返回**空 attn 列表** |
| 3 | `os_ln_fcs` AttributeError | anti-outlier 的 m1/m2/m3 都不支持 Qwen3 → 只能跳过 |
| 4 | 「Invalid repository ID」 | msmodelslim **不写 `config.json`**，vLLM 连架构都认不出 |
| 5 | `KeyError: embed_tokens` | **我给 `config.json` 加了 `quantization_config`，反而让 vLLM 跳过描述文件** |

**第 2 层最难查：四个列表等长、校验通过，但内容是空的。**

```python
if not (len(attn_list) == len(mhsa_ln_list) == len(ffn_list) == len(ffn_ln_list)):
    raise ValueError("Failed to get network pattern by DAG")   # ← 这道校验过了
...
item0 = module_names[0][0] if isinstance(module_names[0], list) else module_names[0]
                   # ↑ IndexError: list index out of range
```

正规做法是把 anti-outlier 从 `QuantConfig(do_smooth=True)` 里拿出来，用
`AntiOutlier(model, cfg, norm_class_name=...)` 单独跑，并**显式给 norm 层类名**
（从模型里动态取：`type(model.model.layers[0].input_layernorm).__name__`，不硬编码）。

**第 5 层最反直觉：加一个字段让它「更明确」，结果让它更盲目。**

```python
# vllm/model_executor/model_loader/weight_utils.py: get_quant_config()
hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
if hf_quant_config is not None:
    return quant_cls.from_config(hf_quant_config)          # 有就走这条
possible_config_filenames = quant_cls.get_config_filenames()  # 只有 None 才到这
```

`config.json` 里**不能**有 `quantization_config`，否则 vLLM 拿那个 2 键的 dict 当
`quant_description`，再也不读那 904 条的文件。要靠命令行 `--quantization ascend` 触发。
只看报错（`KeyError` 指向描述文件里明明存在的键）是查不出来的，得读 vLLM 源码。

### 打包清单（缺一个就起不来）

```
config.json                      ← 从基座拷，msmodelslim 不写
generation_config.json           ← 同上
quant_model_description.json     ← msmodelslim 默认写 ..._<scheme>.json，
                                   而 get_config_filenames() 只认精确名
quant_model_weight_*.safetensors ← vLLM 按 *.safetensors glob，名字无所谓
config.json 里【不要】有 quantization_config，改用 --quantization ascend
```

### 又一次覆盖掉自己的修复

评测作业一直 `ModuleNotFoundError: No module named 'acl'`：

```bash
export PYTHONPATH=$L/pylibs                  # 我写的（覆盖）
export PYTHONPATH=${PYTHONPATH:-}:$L/pylibs  # 跑通的 serve 脚本（追加）
```

`acl` 是 CANN 由 `set_env.sh` 加进 `PYTHONPATH` 的，**覆盖式赋值把它冲掉了**。
本文前面「脚本必须有唯一源头」那条讲的是同一类错误，这是第三次。
**凡是 source 过昇腾环境之后再动 `PYTHONPATH`/`LD_LIBRARY_PATH`，只能追加。**

## 还没做的

- **训练侧**。本文只验证了推理与量化。
- **W4A8_DYNAMIC / W4A4_FLATQUANT_DYNAMIC**。vllm-ascend 还支持这两种，没试。
- **anti-outlier 对 Qwen3 的支持**。m1/m2/m3 全挂，只能跳过。有 anti-outlier 的
  静态 W8A8 会不会活？不知道——这是唯一还能捞回静态方案的路。
- 一个真正能用的 torchair 稠密 Qwen3 路径——需要上游补 `torchair/models/qwen3.py`，
  不是配置能绕开的。
