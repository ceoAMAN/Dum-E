#!/bin/bash
# =============================================================================
#  STURNUS — Full Protocol
#  Loop 1: training_b_full    (100% token, Timeline B, weight updates)
#  Loop 2: deployment_half    (50%  token, natural routing, A or B)
#  Loop 3: timeline_a_centile (1%   token, forced A, fast-path probe)
#
#  Usage:
#    bash scripts/run_full_protocol.sh              # resume from state
#    bash scripts/run_full_protocol.sh --clean      # wipe state, fresh start
#    bash scripts/run_full_protocol.sh --skip-warmup --clean  # skip phase 1-3
#
#  Env overrides (all optional):
#    STURNUS_MAX_TOKENS=10000000
#    STURNUS_BATCH_SIZE=256
#    STURNUS_PRINT_EVERY_BATCHES=10
#    STURNUS_CHECKPOINT_EVERY_BATCHES=100
#    STURNUS_SEED=42
#    STURNUS_WARMUP_STEPS=12      # steps for phase 1/2/3 warmup
# =============================================================================
set -euo pipefail

# ── resolve paths ─────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT/sturnus-env312/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$ROOT/sturnus-env/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] No venv python found at sturnus-env312 or sturnus-env"
  echo "[error] Run your setup script first"
  exit 1
fi

LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/sturnus-full-protocol.log"
SUMMARY_FILE="$LOG_DIR/protocol_summary.txt"
mkdir -p "$LOG_DIR"

# ── args ──────────────────────────────────────────────────────────────────────
DO_CLEAN=0
SKIP_WARMUP=0
for arg in "$@"; do
  case "$arg" in
    --clean)        DO_CLEAN=1 ;;
    --skip-warmup)  SKIP_WARMUP=1 ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^#  \{0,2\}//'
      exit 0 ;;
  esac
done

# ── env ───────────────────────────────────────────────────────────────────────
if [ -f "$ROOT/.env.local" ]; then
  set -a; source "$ROOT/.env.local"; set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[error] HF_TOKEN is not set"
  echo "[error] export HF_TOKEN=hf_... and re-run"
  exit 1
fi

MAX_TOKENS="${STURNUS_MAX_TOKENS:-10000000}"
BATCH_SIZE="${STURNUS_BATCH_SIZE:-256}"
PRINT_EVERY="${STURNUS_PRINT_EVERY_BATCHES:-10}"
CHECKPOINT_EVERY="${STURNUS_CHECKPOINT_EVERY_BATCHES:-100}"
SEED="${STURNUS_SEED:-42}"
WARMUP_STEPS="${STURNUS_WARMUP_STEPS:-12}"

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# ── helpers ───────────────────────────────────────────────────────────────────
PROTOCOL_START=$(date +%s)
STAGE_START=$PROTOCOL_START

_banner() {
  local title="$1"
  echo "" | tee -a "$LOG_FILE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
  printf "  %-66s\n" "$title" | tee -a "$LOG_FILE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
  STAGE_START=$(date +%s)
}

_done() {
  local label="$1"
  local secs=$(( $(date +%s) - STAGE_START ))
  local h=$((secs/3600)) m=$(( (secs%3600)/60 )) s=$((secs%60))
  echo "[done] $label — elapsed $(printf '%02d:%02d:%02d' $h $m $s)" | tee -a "$LOG_FILE"
  echo "  $label: $(printf '%02d:%02d:%02d' $h $m $s)" >> "$SUMMARY_FILE"
}

_run() {
  # run with HF_TOKEN + PYTHONUNBUFFERED, tee to log
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 HF_TOKEN="$HF_TOKEN" \
    HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
    "$PYTHON_BIN" "$@" 2>&1 | tee -a "$LOG_FILE"
  # preserve exit code through pipe
  return "${PIPESTATUS[0]}"
}

# ── preamble ──────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee "$LOG_FILE"
echo "  STURNUS — Full 3-Loop Protocol" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "  Python   : $PYTHON_BIN ($PYTHON_VERSION)" | tee -a "$LOG_FILE"
echo "  Max tok  : $MAX_TOKENS" | tee -a "$LOG_FILE"
echo "  Batch    : $BATCH_SIZE tokens" | tee -a "$LOG_FILE"
echo "  Seed     : $SEED" | tee -a "$LOG_FILE"
echo "  Clean    : $DO_CLEAN" | tee -a "$LOG_FILE"
echo "  Warmup   : $( [ "$SKIP_WARMUP" = 1 ] && echo 'skipped' || echo "${WARMUP_STEPS} steps/phase" )" | tee -a "$LOG_FILE"
echo "  Log      : $LOG_FILE" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# write summary header
{
  echo "STURNUS Protocol Summary — $(date)"
  echo "Timings:"
} > "$SUMMARY_FILE"

cd "$ROOT"

# ── clean ─────────────────────────────────────────────────────────────────────
if [ "$DO_CLEAN" = 1 ]; then
  _banner "CLEAN — Wiping state and logs"
  rm -rf "$ROOT/state"
  rm -f "$LOG_DIR/finetune_metrics.json" \
        "$LOG_DIR/proof_metrics.jsonl" \
        "$LOG_DIR/benchmark_runs.jsonl" \
        "$LOG_DIR/benchmark_summary.json"
  echo "[clean] Done" | tee -a "$LOG_FILE"
fi

# ── kill any existing finetune ────────────────────────────────────────────────
pkill -INT -f 'scripts/finetune.py' > /dev/null 2>&1 || true
sleep 2
pkill -TERM -f 'scripts/finetune.py' > /dev/null 2>&1 || true

# ── warmup phases 1, 2, 3 ────────────────────────────────────────────────────
if [ "$SKIP_WARMUP" = 0 ]; then

  _banner "PHASE 1 — Central model warmup (LoRA, ${WARMUP_STEPS} steps)"
  STURNUS_TRAIN_STEPS="$WARMUP_STEPS" STURNUS_SAVE_EVERY=60 \
    _run scripts/train_phase1.py
  _done "Phase 1 (Central warmup)"

  _banner "PHASE 2 — Gate model warmup (4-loss, ${WARMUP_STEPS} steps)"
  STURNUS_TRAIN_STEPS="$WARMUP_STEPS" \
    _run scripts/train_phase2.py
  _done "Phase 2 (Gate warmup)"

  _banner "PHASE 3 — Expert group warmup (3-loss, ${WARMUP_STEPS} steps/group)"
  STURNUS_TRAIN_STEPS="$WARMUP_STEPS" STURNUS_EXPERT_GROUP_LIMIT=5 \
    _run scripts/train_phase3.py
  _done "Phase 3 (Expert warmup)"

else
  echo "[skip] Warmup phases skipped (--skip-warmup)" | tee -a "$LOG_FILE"
fi

# ── loop 1: full finetune ─────────────────────────────────────────────────────
_banner "LOOP 1 — training_b_full  (100% token, Timeline B, ${MAX_TOKENS} tokens)"
CLEAN_FLAG=""
[ "$DO_CLEAN" = 1 ] && CLEAN_FLAG="--clean"
# caffeinate prevents sleep during the long training run; run it as a
# background daemon so it doesn't try to exec the shell function _run
caffeinate -dimsu &
CAF_PID=$!
_run scripts/finetune.py \
    $CLEAN_FLAG \
    --max-tokens               "$MAX_TOKENS" \
    --batch-size               "$BATCH_SIZE" \
    --print-every-batches      "$PRINT_EVERY" \
    --checkpoint-every-batches "$CHECKPOINT_EVERY" \
    --seed                     "$SEED"
kill "$CAF_PID" 2>/dev/null || true
_done "Loop 1 (training_b_full)"

# ── loop 2 + 3: benchmark ────────────────────────────────────────────────────
_banner "LOOP 2+3 — benchmark  (deployment_half + timeline_a_centile)"
_run scripts/benchmark.py
_done "Loop 2+3 (benchmark)"

# ── validate ──────────────────────────────────────────────────────────────────
_banner "VALIDATE — End-to-end inference check"
_run scripts/validate.py --samples 20
_done "Validate"

# ── final summary ─────────────────────────────────────────────────────────────
TOTAL_SECS=$(( $(date +%s) - PROTOCOL_START ))
TOTAL_H=$((TOTAL_SECS/3600))
TOTAL_M=$(( (TOTAL_SECS%3600)/60 ))
TOTAL_S=$((TOTAL_SECS%60))

echo "" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "  PROTOCOL COMPLETE" | tee -a "$LOG_FILE"
echo "  Total time : $(printf '%02d:%02d:%02d' $TOTAL_H $TOTAL_M $TOTAL_S)" | tee -a "$LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# print benchmark summary if it exists
if [ -f "$LOG_DIR/benchmark_summary.json" ]; then
  echo "Benchmark summary:" | tee -a "$LOG_FILE"
  cat "$LOG_DIR/benchmark_summary.json" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "[tip] Full log: $LOG_FILE"
echo "[tip] Benchmark runs: $LOG_DIR/benchmark_runs.jsonl"
echo "[tip] Proof metrics: $LOG_DIR/proof_metrics.jsonl"
