#!/bin/bash
#SBATCH -J yue_transres
#SBATCH -p a320m2tn910cu
#SBATCH --exclude=npu1-3,npu1-11,npu1-19
#SBATCH --gres=npu:8
#SBATCH -c 64
#SBATCH -t 03:00:00
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
# PYTHONPATH 两条纪律：必须追加（覆盖会冲掉 CANN 的 acl）；自备目录放最后
# （pylibs_bleu 里那个 x86_64 的 regex 曾经遮蔽掉环境的 regex，24 个任务全挂）
export PYTHONPATH=${PYTHONPATH:-}:$L/pylibs:$L/pylibs_bleu:$L/pylibs_zh
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
mkdir -p $L/results/trans2 $L/logs/trans2
hostname; date

# 为什么重跑翻译：官方 can2man 的参考译文是**简体**，我们的模型输出**繁体**。
# chrF 是字符级的，繁简差异会把语域已经转对的句子判成几乎全错。实测样例：
#   src 鐳射影碟喺香港同澳門地區仍然係一種常見嘅稱呼。
#   ref 镭射影碟在香港和澳门地区仍然是一种常见的称呼。
#   hyp 鐳射影碟在香港和澳門地區仍然是一種常見的稱呼。   ← 语域全对，只差字形
# 这一轮同时报 chrF（照做不误）和 chrF_t2s（归一到简体，只量语域转换），
# 两者的差就是「没按要求出简体」值多少分。顺带把 2000 条译文全部落盘，
# 以后换打分口径不用再占 NPU。
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

run_one () {
  local G=$1 N=$2 M=$3
  [ -f "$M/config.json" ] || { echo "!! 跳过 $N：$M 没有 config.json"; return 1; }
  local OUT=$L/results/trans2/${N}_trans.json
  [ -f "$OUT" ] && { echo "· $N 已有结果，跳过"; return 0; }
  ASCEND_RT_VISIBLE_DEVICES=$G $E/bin/python $L/code/eval_hkmmlu_official.py \
     --model "$M" --name "$N" --task trans --trans-dir $L/data/hkmmlu_translate \
     --max-tokens 256 --save-hyps --gpu-util 0.85 --out "$OUT" \
     > $L/logs/trans2/${N}.log 2>&1
  RC=$?; echo "[$(date +%H:%M:%S)] $N rc=$RC (die$G)"   # RC 先存：$(date) 会冲掉 $?
}

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

echo "######## 汇总：raw vs 繁简归一 ########"
$E/bin/python $L/code/summarize_trans.py $L/results/trans2 \
   2>&1 | tee $L/results/trans2/summary.txt
date; echo TRANSRES_DONE
