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

## 还没做的

- 图模式 / 融合算子（`torch_npu` 的 `torchair`），预期是把 3.6 倍的差距压下来的主要手段
- `vllm-ascend` 起 OpenAI 兼容服务（环境已就绪，未验证）
- 训练侧（本文只验证了推理）
