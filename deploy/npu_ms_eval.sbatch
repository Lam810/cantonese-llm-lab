#!/bin/bash
#SBATCH -J yue_mseval
#SBATCH -p a320m2tn910cu
#SBATCH --exclude=npu1-3,npu1-11,npu1-19
#SBATCH --gres=npu:2
#SBATCH -c 24
#SBATCH -t 06:00:00
#SBATCH -o /data/user/zlin810/yue-lab/logs/%x_%j.out
#SBATCH -e /data/user/zlin810/yue-lab/logs/%x_%j.err
# 不能用 set -u：昇腾 set_env.sh 引用未定义变量，开了会在 source 处静默退出
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
# 只评 W8A8_DYNAMIC。静态那版输出是 token 沙拉，已判废——注意它的吞吐反而更高
# （511 vs 312 tok/s），因为它从不吐 EOS、一路生成到 max_tokens 上限。
# 只看 tok/s 会把报废的模型报成更好的，必须看输出。
D=$L/variants_ms/v2-qwen3-8b-ms-w8a8_dynamic-noanti

echo "######## 1/2 纯度 + 退化（117 条，与主表同提问集）########"
$E/bin/python $L/code/eval_hkmmlu_official.py --model "$D" --name ms_w8a8_dynamic \
   --task purity --quantization ascend --purity-max-new 128 \
   --out $L/results/ms/ms_w8a8_dynamic_purity.json
echo "purity rc=$?"

echo "######## 2/2 HKMMLU 官方口径（每 config 50 题）########"
$E/bin/python $L/code/eval_hkmmlu_official.py --model "$D" --name ms_w8a8_dynamic \
   --task mc --quantization ascend --hkmmlu-dir $L/data/HKMMLU \
   --limit-per-cfg 50 --max-tokens 512 \
   --out $L/results/ms/ms_w8a8_dynamic_mc.json
echo "mc rc=$?"

echo "######## 闸门判定 ########"
$E/bin/python $L/code/gate_check.py $L/results/ms/ms_w8a8_dynamic_purity.json \
                                    $L/results/ms/ms_w8a8_dynamic_mc.json
date; echo MS_EVAL_DONE
