#!/bin/bash
#SBATCH -J yue_scaffold
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
mkdir -p $L/results/scaffold $L/logs/scaffold
hostname; date; npu-smi info 2>/dev/null | grep -c Ascend910

# 提示词消融：strict（主表，已跑完在 results/npustd）vs scaffold（多一句
# 「請喺最後一行寫「答案：X」」）。为什么要跑：我们早期那张把自己排第一的表
# 用的就是 scaffold，而 scaffold 对 ours 值 +8.3pp、对最强对手只值 +1.5pp。
# 光看两个数字说不清是「脚手架帮了所有人、只是帮我们更多」还是「换了别的东西」，
# 所以这里把提示词做成唯一变量，全量重跑一遍，逐题配对。
# 预期：如果 ours 的增量显著大于对手，那就坐实了「我们的短板是指令遵循，
# 不是知识」——这是一条对我们不利但必须写进报告的结论。
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
  for T in mc; do
    local LOG=$L/logs/scaffold/${N}_${T}.log
    local OUT=$L/results/scaffold/${N}_${T}.json
    [ -f "$OUT" ] && { echo "· $N/$T 已有结果，跳过"; continue; }
    # 唯一变量是 --mc-style scaffold：题目、题量、解析器、采样参数、max-tokens
    # 全部和 strict 那轮逐字相同，这样两轮可以逐题配对做 McNemar。
    ARGS="--task mc --hkmmlu-dir $L/data/HKMMLU --limit-per-cfg 0 --max-tokens 512 --mc-style scaffold"
    ASCEND_RT_VISIBLE_DEVICES=$G $E/bin/python $L/code/eval_hkmmlu_official.py \
       --model "$M" --name "$N" $ARGS --gpu-util 0.85 --out "$OUT" > "$LOG" 2>&1
    RC=$?; echo "[$(date +%H:%M:%S)] $N/$T rc=$RC (die$G)"
  done
}

# 8 个 die，每个 die 一条串行队列
i=0; PIDS=""
for spec in "${MODELS[@]}"; do
  N="${spec%%|*}"; M="${spec#*|}"
  run_one $((i % 8)) "$N" "$M" &
  PIDS="$PIDS $!"; i=$((i+1)); sleep 15
done
for p in $PIDS; do wait $p; done      # 裸 wait 会等到常驻进程，逐个等
echo "######## 全部完成 $(date) ########"

echo "######## 汇总（scaffold 口径）########"
$E/bin/python $L/code/summarize_std.py $L/results/scaffold
echo "######## strict vs scaffold 逐模型对照 ########"
$E/bin/python $L/code/prompt_ablation.py $L/results/npustd $L/results/scaffold \
   2>&1 | tee $L/results/scaffold/prompt_ablation.txt
date; echo SCAFFOLD_DONE
