#!/bin/bash
#SBATCH -J yue_npustd
#SBATCH -p a320m2tn910cu
#SBATCH --exclude=npu1-3,npu1-11,npu1-19
#SBATCH --gres=npu:8
#SBATCH -c 64
#SBATCH -t 08:00:00
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
# PYTHONPATH 两条纪律：
#   1) 必须追加，覆盖会冲掉 CANN 由 set_env.sh 加的 acl 模块
#   2) 自己 unzip 的 pylibs_* 要放在**最后**。放前面会遮蔽环境里本来能用的包——
#      上一轮 pylibs_bleu 里那个 x86_64 的 regex 就把 vllm 依赖的 regex 顶掉了，
#      症状是 "cannot import name '_regex' ... circular import"，24 个任务全挂。
export PYTHONPATH=${PYTHONPATH:-}:$L/pylibs:$L/pylibs_bleu
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
mkdir -p $L/results/npustd $L/logs/npustd
hostname; date; npu-smi info 2>/dev/null | grep -c Ascend910

# HPC3 的 H100 被别的作业占满（还剩 13 小时），标准评测搬到昇腾跑。
# 依据：bf16 对照那次，3300 题的生成式 MC 在 NPU 上只花了 86 秒
# （vLLM 连续批处理 + 平均输出只有 15 token），所以全量 26,368 题约 12 分钟/模型。
# 口径：HKMMLU **官方 zero-shot prompting（生成）**，全量；外加官方翻译任务。
MODELS=(
  "ours|$L/models/merged-clean"
  "ours_cleandata|$L/stage/v2-clean-data"
  "rival_v2_8b_thinking|$L/rivals/CantoneseLLM-v2.0-8B-Thinking"
  "rival_v2_8b_cvmerged|$L/rivals/CantoneseLLM-v2.0-8B-Thinking-Chat-Vector-Merged"
  "rival_chat_7b|$L/rivals/CantoneseLLMChat-v1.0-7B"
  "rival_qwen2_canto_7b|$L/rivals/Qwen2-Cantonese-7B-Instruct"
  "rival_llama3_canto_8b|$L/rivals/Llama-3-Cantonese-8B-Instruct"
  "base_qwen3_8b|$L/models/Qwen3-8B"
)

run_one () {   # $1=die  $2=name  $3=path
  local G=$1 N=$2 M=$3
  [ -f "$M/config.json" ] || { echo "!! 跳过 $N：$M 没有 config.json"; return 1; }
  for T in purity mc trans; do
    local LOG=$L/logs/npustd/${N}_${T}.log
    local OUT=$L/results/npustd/${N}_${T}.json
    [ -f "$OUT" ] && { echo "· $N/$T 已有结果，跳过"; continue; }
    case $T in
      purity) ARGS="--task purity --purity-max-new 128";;
      mc)     ARGS="--task mc --hkmmlu-dir $L/data/HKMMLU --limit-per-cfg 0 --max-tokens 512";;
      trans)  ARGS="--task trans --trans-dir $L/data/hkmmlu_translate --max-tokens 256";;
    esac
    ASCEND_RT_VISIBLE_DEVICES=$G $E/bin/python $L/code/eval_hkmmlu_official.py \
       --model "$M" --name "$N" $ARGS --gpu-util 0.85 --out "$OUT" > "$LOG" 2>&1
    RC=$?; echo "[$(date +%H:%M:%S)] $N/$T rc=$RC (die$G)"
  done
}

# 8 个 die，每个 die 一条串行队列
# 不能写死 die 0..7 这些绝对编号。同一节点上只要还有别的作业（哪怕是我自己
# 的另一个作业）占着 die 0，Slurm 分给本作业的就是 die 2..9，写死 0..7 会落到
# cgroup 外面，报 "Device:-1, RankID:-1 / Engine core initialization failed"，
# 而且**八个任务全挂**、看不出和别的作业有关。作业 137921 就是这么死的：
# 同脚本前三次都对，第四次错，唯一变化是 npu1-6 上还有我自己的 137919。
DEVS=(${ASCEND_RT_VISIBLE_DEVICES//,/ })
[ ${#DEVS[@]} -eq 0 ] && DEVS=(${ASCEND_VISIBLE_DEVICES//,/ })
[ ${#DEVS[@]} -eq 0 ] && DEVS=(0 1 2 3 4 5 6 7)
NDEV=${#DEVS[@]}
echo "Slurm 给的 die: ${DEVS[*]}  (共 $NDEV)"

i=0; PIDS=""
for spec in "${MODELS[@]}"; do
  N="${spec%%|*}"; M="${spec#*|}"
  run_one "${DEVS[$((i % NDEV))]}" "$N" "$M" &
  PIDS="$PIDS $!"; i=$((i+1)); sleep 15
done
for p in $PIDS; do wait $p; done      # 裸 wait 会等到常驻进程，逐个等
echo "######## 全部完成 $(date) ########"

echo "######## 汇总 ########"
$E/bin/python $L/code/summarize_std.py $L/results/npustd
echo "######## 配对检验（官方生成口径，基准=ours）########"
cd $L/results/npustd
$E/bin/python $L/code/compare_models.py --key mc ours_mc.json \
   ours_cleandata_mc.json rival_v2_8b_thinking_mc.json rival_v2_8b_cvmerged_mc.json \
   rival_chat_7b_mc.json rival_qwen2_canto_7b_mc.json rival_llama3_canto_8b_mc.json \
   base_qwen3_8b_mc.json 2>&1 | tee mcnemar_npustd.txt
date; echo NPUSTD_DONE
