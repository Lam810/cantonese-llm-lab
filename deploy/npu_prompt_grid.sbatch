#!/bin/bash
#SBATCH -J yue_grid
#SBATCH -p a320m2tn910cu
#SBATCH --exclude=npu1-3,npu1-11,npu1-19
#SBATCH --gres=npu:8
#SBATCH -c 64
#SBATCH -t 06:00:00
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
export PYTHONPATH=${PYTHONPATH:-}:$L/pylibs:$L/pylibs_bleu:$L/pylibs_zh
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
mkdir -p $L/results/legacy $L/results/yue_strict $L/logs/grid
hostname; date

# 提示词 2×2：语言（标准书面中文 / 书面粤语）× 格式脚手架（无 / 有）
#
#                    标准书面中文主体      粤语主体
#   无脚手架         strict（主表）        yue_strict
#   有脚手架         scaffold              legacy（= 旧表用的那个）
#
# 为什么要做满 2×2：旧表给 ours 0.6227、官方 strict 口径只有 0.5394，差 8.33pp。
# 已知 scaffold 那句话值 +3.00pp、每科前 10 题的子集偏差值 +0.91pp，还剩约 4.4pp
# 没交代。旧提示词的**主体本身也不一样、而且是粤语写的**，这一项必须单独量——
# 给粤语专化的模型喂粤语提示词，可能是真实增益，不是作弊。不做满这个格子，
# 就只能说「旧数字复现不出来」，不能说清它由什么构成。
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
STYLES=(legacy yue_strict)

run_one () {
  local G=$1 N=$2 M=$3 S=$4
  [ -f "$M/config.json" ] || { echo "!! 跳过 $N：$M 没有 config.json"; return 1; }
  local OUT=$L/results/$S/${N}_mc.json
  [ -f "$OUT" ] && { echo "· $N/$S 已有结果，跳过"; return 0; }
  ASCEND_RT_VISIBLE_DEVICES=$G $E/bin/python $L/code/eval_hkmmlu_official.py \
     --model "$M" --name "$N" --task mc --hkmmlu-dir $L/data/HKMMLU \
     --limit-per-cfg 0 --max-tokens 512 --mc-style "$S" --gpu-util 0.85 --out "$OUT" \
     > $L/logs/grid/${N}_${S}.log 2>&1
  RC=$?; echo "[$(date +%H:%M:%S)] $N/$S rc=$RC (die$G)"   # RC 先存：$(date) 会冲掉 $?
}

# 不能写死 die 0..7：同节点上有别的作业（包括自己的）占着 die 0 时，Slurm 分给
# 本作业的是 die 2..9，绝对编号会落到 cgroup 外，报 Device:-1 且**全部任务**都挂。
DEVS=(${ASCEND_RT_VISIBLE_DEVICES//,/ })
[ ${#DEVS[@]} -eq 0 ] && DEVS=(${ASCEND_VISIBLE_DEVICES//,/ })
[ ${#DEVS[@]} -eq 0 ] && DEVS=(0 1 2 3 4 5 6 7)
NDEV=${#DEVS[@]}
echo "Slurm 给的 die: ${DEVS[*]}  (共 $NDEV)"

# 16 个 (模型,提示词) 任务铺在 NDEV 个 die 上。把慢模型（思维链那三个，
# 平均输出 450+ tok）的两个臂分到不同 die，别让它们在同一条队列里串起来。
TASKS=()
for S in "${STYLES[@]}"; do
  for spec in "${MODELS[@]}"; do TASKS+=("$S|$spec"); done
done

i=0; PIDS=""
for t in "${TASKS[@]}"; do
  S="${t%%|*}"; rest="${t#*|}"; N="${rest%%|*}"; M="${rest#*|}"
  run_one "${DEVS[$((i % NDEV))]}" "$N" "$M" "$S" &
  PIDS="$PIDS $!"; i=$((i+1)); sleep 12
done
for p in $PIDS; do wait $p; done      # 裸 wait 会等到常驻进程，逐个等

echo "######## 提示词 2×2 汇总 ########"
$E/bin/python $L/code/prompt_grid.py \
   --strict $L/results/npustd --scaffold $L/results/scaffold \
   --legacy $L/results/legacy --yue-strict $L/results/yue_strict \
   2>&1 | tee $L/results/legacy/prompt_grid.txt
date; echo GRID_DONE
