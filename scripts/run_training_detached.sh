#!/bin/bash
set -euo pipefail

ROOT="/Users/aman/Sturnus"
SESSION_NAME="${STURNUS_SESSION_NAME:-sturnus-train}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
TMUX_DIR="$ROOT/.tmux"
SOCKET_PATH="$TMUX_DIR/${SESSION_NAME}.sock"
RUN_FINETUNE="${STURNUS_RUN_FINETUNE:-1}"
RUN_PHASE1="${STURNUS_RUN_PHASE1:-0}"
RUN_BENCHMARK="${STURNUS_RUN_BENCHMARK:-0}"
MAX_TOKENS="${STURNUS_MAX_TOKENS:-1000000}"
BATCH_SIZE="${STURNUS_BATCH_SIZE:-256}"
CHECKPOINT_INTERVAL="${STURNUS_CHECKPOINT_INTERVAL:-180}"
SEED="${STURNUS_SEED:-42}"
CLEAN="${STURNUS_CLEAN:-0}"

mkdir -p "$LOG_DIR"
mkdir -p "$TMUX_DIR"

if [ -f "$ROOT/.env.local" ]; then
  set -a
  source "$ROOT/.env.local"
  set +a
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux is not installed."
  echo "Install it with: brew install tmux"
  exit 1
fi

if ! command -v caffeinate >/dev/null 2>&1; then
  echo "[error] caffeinate is not available on this system."
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[error] HF_TOKEN is not set."
  echo "Run: export HF_TOKEN='your_token_here'"
  exit 1
fi

if tmux -S "$SOCKET_PATH" has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux -S $SOCKET_PATH attach -t $SESSION_NAME"
  exit 1
fi

read -r -d '' RUN_CMD <<'EOF' || true
cd /Users/aman/Sturnus
source sturnus_env/bin/activate
export PYTHONUNBUFFERED=1
echo "[boot] session started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[boot] logging to logs/${STURNUS_SESSION_NAME:-sturnus-train}.log"
echo "[boot] run_finetune=${STURNUS_RUN_FINETUNE:-1} run_phase1=${STURNUS_RUN_PHASE1:-0} run_benchmark=${STURNUS_RUN_BENCHMARK:-0}"
echo "[boot] max_tokens=${STURNUS_MAX_TOKENS:-1000000} batch_size=${STURNUS_BATCH_SIZE:-256} checkpoint_interval=${STURNUS_CHECKPOINT_INTERVAL:-180} seed=${STURNUS_SEED:-42} clean=${STURNUS_CLEAN:-0}"
if [ "${STURNUS_RUN_FINETUNE:-1}" = "1" ]; then
  clean_args=()
  if [ "${STURNUS_CLEAN:-0}" = "1" ]; then
    clean_args+=(--clean)
  fi
  python3 finetune.py --max-tokens "${STURNUS_MAX_TOKENS:-1000000}" --batch-size "${STURNUS_BATCH_SIZE:-256}" --checkpoint-interval "${STURNUS_CHECKPOINT_INTERVAL:-180}" --seed "${STURNUS_SEED:-42}" "${clean_args[@]}"
  status=$?
  echo "$status" > "logs/${STURNUS_SESSION_NAME:-sturnus-train}.exitcode"
  echo "[done] finetune exit=$status at $(date '+%Y-%m-%d %H:%M:%S')"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
fi
if [ "${STURNUS_RUN_PHASE1:-0}" = "1" ]; then
  STURNUS_TRAIN_STEPS=12 STURNUS_SAVE_EVERY=60 python3 scripts/train_phase1.py
fi
if [ "${STURNUS_RUN_BENCHMARK:-0}" = "1" ]; then
  python3 scripts/benchmark.py
  benchmark_status=$?
  echo "$benchmark_status" > "logs/${STURNUS_SESSION_NAME:-sturnus-train}.benchmark.exitcode"
  echo "[done] benchmark exit=$benchmark_status at $(date '+%Y-%m-%d %H:%M:%S')"
  exit "$benchmark_status"
fi
echo "[done] session finished at $(date '+%Y-%m-%d %H:%M:%S')"
EOF

tmux -S "$SOCKET_PATH" new-session -d -s "$SESSION_NAME" \
  "export STURNUS_SESSION_NAME='$SESSION_NAME'; export HF_TOKEN='$HF_TOKEN'; export STURNUS_RUN_FINETUNE='$RUN_FINETUNE'; export STURNUS_RUN_PHASE1='$RUN_PHASE1'; export STURNUS_RUN_BENCHMARK='$RUN_BENCHMARK'; export STURNUS_MAX_TOKENS='$MAX_TOKENS'; export STURNUS_BATCH_SIZE='$BATCH_SIZE'; export STURNUS_CHECKPOINT_INTERVAL='$CHECKPOINT_INTERVAL'; export STURNUS_SEED='$SEED'; export STURNUS_CLEAN='$CLEAN'; caffeinate -dimsu bash -lc \"$RUN_CMD\" 2>&1 | tee -a '$LOG_FILE'"

if ! tmux -S "$SOCKET_PATH" has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] failed to start tmux session '$SESSION_NAME'."
  exit 1
fi

echo "[ok] Started detached session: $SESSION_NAME"
echo "[ok] Log file: $LOG_FILE"
echo "[tip] Attach: tmux -S $SOCKET_PATH attach -t $SESSION_NAME"
echo "[tip] Live log: tail -f $LOG_FILE"
echo "[note] Closing the Terminal app is fine. Closing the MacBook lid usually still sleeps the machine unless clamshell mode is active."
