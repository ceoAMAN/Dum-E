#!/bin/bash
# Dum-E resume loop. Runs finetune to the token target; if a transient Metal/GPU
# error or any non-graceful exit kills it before the target, it resumes from the
# last checkpoint. A graceful SIGINT (exit 130) stops cleanly. Invoked by
# run_caffeinated_1m.sh under caffeinate + nohup — not meant to be called directly.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="$1"; MAX_TOKENS="$2"; PRINT_EVERY="$3"; CKPT_EVERY="$4"; SEED="$5"; CLEAN="$6"
set -a; [ -f .env.local ] && source .env.local; set +a
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error

attempt=1
while true; do
  echo "[loop] ── attempt ${attempt} ($([ -n "$CLEAN" ] && echo fresh || echo resume)) ──"
  "$PY" scripts/finetune.py $CLEAN \
      --max-tokens "$MAX_TOKENS" \
      --print-every-batches "$PRINT_EVERY" \
      --checkpoint-every-batches "$CKPT_EVERY" \
      --seed "$SEED"
  code=$?
  toks=$("$PY" -c "import json;print(json.load(open('logs/marathon_metrics.json')).get('total_tokens',0))" 2>/dev/null || echo 0)
  echo "[loop] finetune exited code=${code} at tok=${toks}/${MAX_TOKENS}"
  if [ "$code" -eq 130 ]; then echo "[loop] graceful stop (SIGINT) — not resuming."; break; fi
  if [ "$toks" -ge "$MAX_TOKENS" ] 2>/dev/null; then echo "[loop] target reached — done."; break; fi
  attempt=$((attempt + 1))
  echo "[loop] exited before target (likely transient GPU error) — resuming from last checkpoint in 10s..."
  sleep 10
  CLEAN=""   # every attempt after the first resumes (no --clean)
done
echo "[loop] Dum-E finished after ${attempt} attempt(s)."
