#!/bin/bash
set -euo pipefail

ROOT="/Users/aman/Sturnus"
PYTHON_BIN="$ROOT/sturnus-env312/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$ROOT/sturnus_env/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$ROOT/sturnus-env/bin/python"
fi
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/sturnus-fresh-10m.log"
PID_FILE="$LOG_DIR/sturnus-fresh-10m.pid"
MAX_TOKENS="${STURNUS_MAX_TOKENS:-10000000}"
BATCH_SIZE="${STURNUS_BATCH_SIZE:-256}"
PRINT_EVERY_BATCHES="${STURNUS_PRINT_EVERY_BATCHES:-10}"
CHECKPOINT_EVERY_BATCHES="${STURNUS_CHECKPOINT_EVERY_BATCHES:-100}"
SEED="${STURNUS_SEED:-42}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

if [ -f "$ROOT/.env.local" ]; then
  set -a
  source "$ROOT/.env.local"
  set +a
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] missing python: $PYTHON_BIN"
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.11|3.12|3.13) ;;
  *)
    echo "[error] unsupported python: $PYTHON_VERSION"
    echo "[error] use a 3.11 or 3.12 environment for MLX stability"
    exit 1
    ;;
esac

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[error] HF_TOKEN is not set"
  exit 1
fi

pkill -INT -f 'scripts/finetune.py' >/dev/null 2>&1 || true
sleep 3
pkill -TERM -f 'scripts/finetune.py' >/dev/null 2>&1 || true
sleep 1

rm -rf "$ROOT/state"
rm -f \
  "$ROOT/logs/finetune_metrics.json" \
  "$ROOT/logs/proof_metrics.jsonl" \
  "$ROOT/logs/benchmark_runs.jsonl" \
  "$ROOT/logs/benchmark_summary.json" \
  "$LOG_FILE" \
  "$PID_FILE"

nohup env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_TOKEN="${HF_TOKEN}" HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error caffeinate -dimsu \
  "$PYTHON_BIN" "$ROOT/scripts/finetune.py" \
  --clean \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --print-every-batches "$PRINT_EVERY_BATCHES" \
  --checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
  --seed "$SEED" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "[ok] fresh run started"
echo "[ok] pid=$PID"
echo "[ok] python=$PYTHON_BIN ($PYTHON_VERSION)"
echo "[ok] log=$LOG_FILE"
echo "[ok] pidfile=$PID_FILE"
echo "[tip] tail -f $LOG_FILE"
