#!/bin/bash
#SBATCH -J yue_bf16ctl
#SBATCH -p a320m2tn910cu
#SBATCH --exclude=npu1-3,npu1-11,npu1-19
#SBATCH --gres=npu:2
#SBATCH -c 24
#SBATCH -t 06:00:00
#SBATCH -o /data/user/zlin810/yue-lab/logs/%x_%j.out
#SBATCH -e /data/user/zlin810/yue-lab/logs/%x_%j.err
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}
L=/data/user/zlin810/yue-lab
E=/data/user/zlin810/nf-vllm-env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/user/zlin810/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=$L/pylibs:${PYTHONPATH:-}   # 必须追加：覆盖会冲掉 CANN 加的 acl
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
mkdir -p $L/results/ms
hostname; date

# 这一跑是为了修掉我自己的口径错误。
# 昇腾 W8A8_DYNAMIC 量出 HKMMLU 0.5479，我拿它去比闸门 0.55 —— 但那个闸门阈值
# 是从 **logprob 口径** 的数字定的（bf16 0.6094），而这次用的是官方生成口径。
# 两个口径量的不是一回事，不能跨口径比阈值。
# 所以要在**完全相同的脚本、引擎（vLLM on NPU）、抽样、max_tokens** 下跑一遍 bf16，
# 才知道 0.5479 是量化的代价，还是口径本身的差异。
D=$L/models/merged-clean

echo "######## 1/2 bf16 纯度（117 条，与 W8A8_DYNAMIC 完全同设置）########"
$E/bin/python $L/code/eval_hkmmlu_official.py --model "$D" --name bf16_npu_ctl \
   --task purity --purity-max-new 128 \
   --out $L/results/ms/bf16_npu_purity.json
echo "purity rc=$?"

echo "######## 2/2 bf16 HKMMLU 官方口径（每 config 50 题，与量化版同抽样）########"
$E/bin/python $L/code/eval_hkmmlu_official.py --model "$D" --name bf16_npu_ctl \
   --task mc --hkmmlu-dir $L/data/HKMMLU --limit-per-cfg 50 --max-tokens 512 \
   --out $L/results/ms/bf16_npu_mc.json
echo "mc rc=$?"

echo "######## 同口径对照 ########"
$E/bin/python - <<'PYX'
import json
L = "/data/user/zlin810/yue-lab/results/ms"
def load(tag):
    p = json.load(open(f"{L}/{tag}_purity.json"))["purity"]
    m = json.load(open(f"{L}/{tag}_mc.json"))["mc"]
    return p, m
rows = []
for tag, name in (("bf16_npu", "bf16（对照）"), ("ms_w8a8_dynamic", "W8A8_DYNAMIC")):
    try:
        p, m = load(tag)
        rows.append((name, m["overall_acc"], m["macro_acc"], m["unparsed_rate"],
                     p["full"]["mean_yue_ratio"], p["full"]["degenerate_rate"],
                     p["full"]["median_out_tokens"]))
    except Exception as e:
        print(f"  {name}: 读不到 ({e})")
h = "%-16s %9s %9s %10s %9s %8s %9s" % ("", "HKMMLU", "macro", "解析失败", "纯度117", "退化", "中位tok")
print(h); print("-" * len(h))
for r in rows:
    print("%-16s %9.4f %9.4f %10.4f %9.4f %8.4f %9s" % r)
if len(rows) == 2:
    d = rows[1][1] - rows[0][1]
    print("\n  同口径下 W8A8_DYNAMIC 相对 bf16：%+.2fpp" % (d * 100))
    print("  （标准误约 0.87pp @ n=3300，所以 |差| < 1.7pp 都在噪声内）")
PYX
date; echo BF16_CTL_DONE
