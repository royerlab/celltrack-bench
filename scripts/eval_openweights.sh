#!/usr/bin/env bash
# Serve an open-weight VLM with vLLM and run it through the frozen CellTrack-Bench protocol.
#
#   MODEL=gemma-4-31b-it MODEL_PATH=google/gemma-4-31b-it scripts/eval_openweights.sh
#   MODEL=qwen3.5-35b-a3b MODEL_PATH=/models/Qwen3.5-35B-A3B SPLIT=validation SAMPLE=4 scripts/eval_openweights.sh
#
# Environment:
#   MODEL        key of the serving table below (required); also the served model name
#   MODEL_PATH   Hugging Face id or local checkpoint directory (required)
#   SERVER_PY    python interpreter with vLLM installed (default: python3)
#   CLIENT_PY    python interpreter with this package installed (default: python3)
#   SPLIT        test | validation (default: test); BENCHMARK  data/hard | data/full (default: data/hard)
#   CONDITIONS   harness conditions (default: "1", the frozen protocol); SAMPLE  N items for a smoke run
#   CONC         concurrent conversations (default: 8); PORT (default: random 8000-8999)
#
# The paper's open-weight rows used vLLM 0.26 (MiniMax-M3: 0.27.1), the per-model settings below, an
# 8192-token total output budget with 2048 reserved for the answer, temperature 1.0 and top-p 0.95.
# Kimi-K3 (1.5 TB in mxfp4) does not fit one 8xH200 node and was served across two nodes with
# tensor-parallel 8 x pipeline-parallel 2 over Ray; that launcher is cluster-specific and not included.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
MODEL="${MODEL:?set MODEL to a key of the serving table}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a Hugging Face id or a local checkpoint directory}"
SERVER_PY="${SERVER_PY:-python3}"
CLIENT_PY="${CLIENT_PY:-python3}"
SPLIT="${SPLIT:-test}"
BENCHMARK="${BENCHMARK:-$REPO/data/hard}"
CONDITIONS="${CONDITIONS:-1}"
TASKS="${TASKS:-cell_linking track_repair track_inspection}"
CONC="${CONC:-8}"
BUDGET="${BUDGET:-8192}"
RESERVE="${RESERVE:-2048}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
LOG_DIR="${LOG_DIR:-$REPO/runs/logs}"
EXTRA_VLLM=""
BLOCK="${BLOCK:-}"

# Per-model serving config. The tool-call parser must match the family: a wrong parser does not error,
# it silently yields zero tool calls and makes every tool condition look like a model failure.
case "$MODEL" in
  gemma-4-31b-it)            TP=2; PARSER=gemma4;     MAXLEN=65536 ;;
  mistral-medium-3.5-128b)   TP=4; PARSER=mistral;    MAXLEN=65536 ;;
  inkling-small)             TP=8; PARSER=inkling;    MAXLEN=49152 ;;
  qwen3.5-397b-a17b)         TP=8; PARSER=qwen3_xml;  MAXLEN=40960 ;;
  qwen3.5-122b-a10b)         TP=8; PARSER=qwen3_xml;  MAXLEN=65536 ;;
  qwen3.5-35b-a3b)           TP=2; PARSER=qwen3_xml;  MAXLEN=65536 ;;
  qwen3.6-27b)               TP=4; PARSER=qwen3_xml;  MAXLEN=65536 ;;
  # MiniMax-M3's sparse layers need a KV block size of exactly 128 (needs vLLM >= 0.27.1); prefix
  # caching off. The ~800 GB checkpoint can take close to an hour to load from network storage.
  minimax-m3)                TP=8; PARSER=minimax_m3; MAXLEN=40960; BLOCK="${BLOCK:-128}"
                             EXTRA_VLLM="--no-enable-prefix-caching" ;;
  # InternVL3.5-38B's own chat template renders no tools, although the tokenizer has <tool_call>
  # tokens: serve the Qwen3/Hermes-style template shipped next to this script with the hermes parser.
  internvl3.5-38b)           TP=4; PARSER=hermes;     MAXLEN=40960
                             EXTRA_VLLM="--chat-template $HERE/internvl35_tool_chat_template.jinja" ;;
  # Molmo2 has no tool-call syntax; the hermes parser only keeps tool-enabled requests from failing,
  # so expect zero tool calls. fp32 weights are served in bf16, and the whole worst-case multimodal
  # item (~32k tokens) must fit one batch because Molmo2 does not support chunked multimodal input.
  molmo2-8b)                 TP=1; PARSER=hermes;     MAXLEN=36864
                             EXTRA_VLLM="--dtype bfloat16 --max-num-batched-tokens 32768" ;;
  # GLM-4.6V uses the GLM XML tool format; TP must divide its 12 vision attention heads.
  glm-4.6v)                  TP=4; PARSER=glm45;      MAXLEN=65536 ;;
  # Llama-4 Scout emits pythonic tool calls ("[f(x=1)]"); native 10M context capped to 65536.
  llama-4-scout-17b-16e)     TP=4; PARSER=llama4_pythonic; MAXLEN=65536 ;;
  *) echo "unknown MODEL=$MODEL; add it to the serving table in $0" >&2; exit 1 ;;
esac
TP="${TP_OVERRIDE:-$TP}"
block_args=(); [[ -n "$BLOCK" ]] && block_args=(--block-size "$BLOCK")

"$CLIENT_PY" -c "import celltrack_bench, cv2, tifffile, openai" \
  || { echo "client interpreter lacks celltrack_bench/cv2/tifffile/openai (pip install -e '.[run]')" >&2; exit 1; }
mkdir -p "$LOG_DIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false

echo "=== $MODEL | TP=$TP parser=$PARSER maxlen=$MAXLEN port=$PORT ==="
# --limit-mm-per-prompt image=26: the crop-based conditions send one image per cell (~20 images).
"$SERVER_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" ${EXTRA_VLLM:-} \
  --max-model-len "$MAXLEN" \
  --limit-mm-per-prompt '{"image": 26}' \
  --gpu-memory-utilization "${GPUMEM:-0.92}" \
  "${block_args[@]}" \
  --enable-auto-tool-choice --tool-call-parser "$PARSER" \
  --port "$PORT" --host 127.0.0.1 \
  --trust-remote-code \
  > "$LOG_DIR/${MODEL}_vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

HEALTH_TRIES="${HEALTH_TRIES:-360}"
echo "waiting for vLLM (pid $SERVER_PID, up to $((HEALTH_TRIES * 10))s) ..."
for _ in $(seq 1 "$HEALTH_TRIES"); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "vLLM died:"; tail -n 30 "$LOG_DIR/${MODEL}_vllm.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "vLLM never became healthy"; tail -n 30 "$LOG_DIR/${MODEL}_vllm.log"; exit 1; }

cd "$REPO" || exit 1
"$CLIENT_PY" -u -m celltrack_bench.runner \
  --split "$SPLIT" --split-dir "$BENCHMARK" ${SAMPLE:+--sample "$SAMPLE"} \
  --conditions $CONDITIONS --tasks $TASKS \
  --backend chat --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" --label "${LABEL:-ow-$MODEL}" \
  --concurrency "$CONC" --output-budget "$BUDGET" --answer-reserve "$RESERVE" \
  --temperature 1.0 --top-p 0.95 2>&1 | tee "$LOG_DIR/${MODEL}_runner.log"
rc=${PIPESTATUS[0]}

RES=$(grep -oE "^  -> .*\.jsonl$" "$LOG_DIR/${MODEL}_runner.log" | tail -1 | sed "s/^  -> //")
if [[ -f "$RES" && -z "${SAMPLE:-}" && "$CONDITIONS" == "1" ]]; then
  "$CLIENT_PY" -m celltrack_bench.leaderboard --benchmark "$BENCHMARK" --split "$SPLIT" --results "$RES" --system "$MODEL"
elif [[ -f "$RES" ]]; then
  "$CLIENT_PY" -m celltrack_bench.score --results "$RES" --split "$SPLIT" --split-dir "$BENCHMARK"
fi
exit "$rc"
