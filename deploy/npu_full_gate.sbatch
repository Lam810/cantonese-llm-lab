#!/bin/bash
#SBATCH -J yue_fullgate
#SBATCH -p a320m2tn910cu
#SBATCH --exclude=npu1-3,npu1-11,npu1-19
#SBATCH --gres=npu:2
#SBATCH -c 12
#SBATCH -t 02:00:00
#SBATCH -o /data/user/zlin810/yue-lab/logs/%x_%j.out
#SBATCH -e /data/user/zlin810/yue-lab/logs/%x_%j.err
# 不能用 set -u：昇腾 set_env.sh 引用未定义变量
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}
L=/data/user/zlin810/yue-lab
E=/data/user/zlin810/nf-vllm-env
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/user/zlin810/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=${PYTHONPATH:-}:$L/pylibs   # 追加，覆盖会冲掉 CANN 的 acl
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
hostname; date

# 为什么要跑全量：n=3300 的 95%CI 是 [-2.76,-0.21]pp，损失上界 2.76pp 压不进
# δ=2.0pp 的容差，判不干净。全量 26,368 题把 CI 收窄到 ~±0.6pp 才能定论。
# 对照直接用主表里那份 ours_mc.json——它就是同一个 bf16 模型、同一个脚本、
# 同一个 strict 提示词、同样的全量题集，逐题可配对。
V=$L/stage/v2-ascend-w8a8-dynamic-sharded
OUT=$L/results/ms/ms_w8a8_dyn_sharded_mc_full.json

echo "######## W8A8_DYNAMIC（分片版）全量 HKMMLU ########"
$E/bin/python $L/code/eval_hkmmlu_official.py --model "$V" --name ms_w8a8_dyn_sharded_full \
   --task mc --quantization ascend --hkmmlu-dir $L/data/HKMMLU \
   --limit-per-cfg 0 --max-tokens 512 --mc-style strict --gpu-util 0.85 --out "$OUT"
RC=$?; echo "mc_full rc=$RC"

echo "######## 非劣效闸门（对照 = 同口径 bf16 全量）########"
$E/bin/python $L/code/gate_check.py \
   $L/results/ms/ms_w8a8_dyn_sharded_purity.json "$OUT" \
   --control $L/results/npustd/ours_mc.json --delta 0.02
echo "gate rc=$?"
date; echo FULLGATE_DONE
