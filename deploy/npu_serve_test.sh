#!/bin/bash
# 昇腾 910C 上用 vllm-ascend 起 OpenAI 兼容服务并验收。
# 设计目标：一次作业内把「小模型验通路 → 目标模型上线 → 功能与吞吐验收」全做完，
# 不做交互式探索——NPU 按机时计费。
set -u
L=$LAB_ROOT
E=$VENV
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_USE_MODELSCOPE=false
export HF_HUB_OFFLINE=1
export PYTHONPATH=${PYTHONPATH:-}:$L/pylibs

MODEL=$1; NAME=$2; TP=${3:-1}; PORT=${4:-8000}; MAXLEN=${5:-4096}
LOG=$L/logs/serve_${NAME}.log
echo "=== [$NAME] 启动 vllm serve  model=$MODEL tp=$TP port=$PORT ==="
$E/bin/python -m vllm.entrypoints.openai.api_server \
   --model "$MODEL" --served-model-name "$NAME" \
   --port $PORT --tensor-parallel-size $TP \
   --max-model-len $MAXLEN --gpu-memory-utilization 0.85 \
   --trust-remote-code > "$LOG" 2>&1 &
SRV=$!
# 等健康检查，最多 15 分钟（昇腾首次编译很慢）
OK=0
for i in $(seq 1 180); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then OK=1; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "!! 服务进程已退出"; break; fi
  sleep 5
done
if [ "$OK" != "1" ]; then
  echo "=== [$NAME] 服务未就绪，日志尾 ==="; tail -40 "$LOG"; kill $SRV 2>/dev/null; return 1 2>/dev/null || exit 1
fi
echo "=== [$NAME] 就绪，用时约 $((i*5))s ==="
curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo

echo "=== [$NAME] 粤语对话验收 ==="
for q in "香港有咩好玩嘅地方？" "點樣由旺角去中環最快？" "你可唔可以解釋下咩係強積金？"; do
  curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":128,\"temperature\":0}" \
  | $E/bin/python -c "
import json,sys
d=json.load(sys.stdin)
c=d['choices'][0]['message']['content'].replace(chr(10),' / ')
u=d.get('usage',{})
print('  Q: $q')
print('  A:', c[:200])
print('     tokens: prompt=%s completion=%s' % (u.get('prompt_tokens'), u.get('completion_tokens')))
" 2>&1 | tail -4
done

echo "=== [$NAME] 吞吐（8 并发 × 128 token）==="
START=$(date +%s.%N)
# 只等这 8 个 curl —— 裸 wait 会连常驻的 vllm 服务进程一起等，直接把作业挂死
PIDS=""
for i in $(seq 1 8); do
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"介紹下香港嘅公共交通。\"}],\"max_tokens\":128,\"temperature\":0}" \
    -o /tmp/bench_$i.json &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
END=$(date +%s.%N)
$E/bin/python -c "
import json, glob
tot=sum(json.load(open(f)).get('usage',{}).get('completion_tokens',0) for f in glob.glob('/tmp/bench_*.json'))
el=$END-$START
print('  8 并发合计 %d completion tokens / %.1fs = %.1f tok/s' % (tot, el, tot/el))
"
rm -f /tmp/bench_*.json
kill $SRV 2>/dev/null; sleep 8; kill -9 $SRV 2>/dev/null
echo "=== [$NAME] 完成 ==="
