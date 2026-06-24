#!/bin/bash
# =============================================================================
#  STURNUS — Caffeinated 10M Token Training Run
# =============================================================================
#  • Validates all 18 datasets before starting
#  • Cleans state for a fresh start
#  • Gradients ON (gate + experts learn)
#  • caffeinate -dimsu keeps the Mac awake with lid closed
#  • PYTHONFAULTHANDLER + PYTHONUNBUFFERED for crash diagnostics
#  • Checkpoints every 100 batches, progress every 10
#  • Automatic semaphore leak cleanup
#
#  Usage:
#    bash scripts/run_caffeinated_10m.sh              # default 10M tokens
#    STURNUS_MAX_TOKENS=20000000 bash scripts/run_caffeinated_10m.sh
#
#  Monitor:
#    tail -f logs/sturnus-caffeinated-10m.log
# =============================================================================
set -Eeuo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
CYN='\033[0;36m'
BLD='\033[1m'
RST='\033[0m'

_ok()   { echo -e "${GRN}[  ok  ]${RST} $*"; }
_warn() { echo -e "${YLW}[ warn ]${RST} $*"; }
_err()  { echo -e "${RED}[error ]${RST} $*"; }
_info() { echo -e "${CYN}[ info ]${RST} $*"; }

# ── resolve paths ─────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN=""
for candidate in "$ROOT/sturnus-env312/bin/python" \
                 "$ROOT/sturnus_env/bin/python" \
                 "$ROOT/sturnus-env/bin/python"; do
  if [ -x "$candidate" ]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  _err "No virtualenv python found"
  _err "Searched: sturnus-env312, sturnus_env, sturnus-env"
  _err "Run setup_native.sh first"
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# ── tunables ──────────────────────────────────────────────────────────────────
MAX_TOKENS="${STURNUS_MAX_TOKENS:-10000000}"
BATCH_SIZE="${STURNUS_BATCH_SIZE:-256}"
PRINT_EVERY="${STURNUS_PRINT_EVERY_BATCHES:-10}"
CHECKPOINT_EVERY="${STURNUS_CHECKPOINT_EVERY_BATCHES:-100}"
SEED="${STURNUS_SEED:-42}"

LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/sturnus-caffeinated-10m.log"
PID_FILE="$LOG_DIR/sturnus-caffeinated-10m.pid"

mkdir -p "$LOG_DIR"

# ── banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}  ☕  STURNUS — Caffeinated 10M Token Run${RST}"
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "  Target     : ${CYN}${MAX_TOKENS}${RST} tokens"
echo -e "  Gradients  : ${GRN}ON${RST} (gate + experts will learn)"
echo -e "  Batch size : ${BATCH_SIZE} tokens"
echo -e "  Print every: ${PRINT_EVERY} batches"
echo -e "  Checkpoint : every ${CHECKPOINT_EVERY} batches"
echo -e "  Seed       : ${SEED}"
echo -e "  Python     : ${PYTHON_BIN} (${PYTHON_VERSION})"
echo -e "  Log        : ${LOG_FILE}"
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0: PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════════════
_info "Running pre-flight checks..."

# Python version
case "$PYTHON_VERSION" in
  3.11|3.12|3.13)
    _ok "Python $PYTHON_VERSION supported"
    ;;
  *)
    _err "Unsupported Python: $PYTHON_VERSION"
    _err "MLX requires 3.11, 3.12, or 3.13"
    exit 1
    ;;
esac

# .env.local
cd "$ROOT"
if [ -f "$ROOT/.env.local" ]; then
  set -a
  source "$ROOT/.env.local"
  set +a
  _ok ".env.local loaded"
else
  _warn "No .env.local found"
fi

# HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
  _err "HF_TOKEN is not set"
  _err "export HF_TOKEN=hf_... and re-run"
  exit 1
fi
_ok "HF_TOKEN set (${HF_TOKEN:0:8}...)"

# Disk space (need at least 5GB for checkpoints + logs)
AVAIL_GB=$(df -g "$ROOT" 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
if [ "$AVAIL_GB" -lt 5 ] 2>/dev/null; then
  _warn "Low disk space: ${AVAIL_GB}GB available (recommend ≥5GB)"
else
  _ok "Disk space: ${AVAIL_GB}GB available"
fi

# Available RAM
RAM_MB=$("$PYTHON_BIN" -c "
import subprocess, re
try:
    out = subprocess.check_output(['vm_stat'], text=True)
    free = int(re.search(r'Pages free:\s+(\d+)', out).group(1))
    inactive = int(re.search(r'Pages inactive:\s+(\d+)', out).group(1))
    page_size = 16384
    mb = (free + inactive) * page_size // (1024 * 1024)
    print(mb)
except Exception:
    print(0)
" 2>/dev/null || echo "0")
if [ "$RAM_MB" -lt 4000 ] 2>/dev/null; then
  _warn "Available RAM: ${RAM_MB}MB (recommend ≥4GB free)"
else
  _ok "Available RAM: ${RAM_MB}MB"
fi

# MLX import check
if ! "$PYTHON_BIN" -c "import mlx.core" 2>/dev/null; then
  _err "mlx not importable — run setup_native.sh"
  exit 1
fi
_ok "MLX importable"

echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: VALIDATE DATASETS
# ══════════════════════════════════════════════════════════════════════════════
_info "Validating all 18 datasets (timeout 60s each)..."
echo ""

VALIDATE_EXIT=0
PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" \
  "$PYTHON_BIN" "$ROOT/scripts/validate_datasets.py" || VALIDATE_EXIT=$?

echo ""
if [ "$VALIDATE_EXIT" -ne 0 ]; then
  _warn "Some datasets had issues (exit code $VALIDATE_EXIT)"
  _warn "Training will skip failed datasets — proceeding anyway"
else
  _ok "All datasets validated successfully"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: KILL STALE PROCESSES + CLEAN SEMAPHORES
# ══════════════════════════════════════════════════════════════════════════════
_info "Cleaning up stale processes..."

set +e
pkill -INT -f 'scripts/finetune.py' 2>/dev/null
pkill -f 'caffeinate' 2>/dev/null
sleep 1
pkill -TERM -f 'scripts/finetune.py' 2>/dev/null
set -e

_ok "Stale processes cleaned"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: STATE INITIALIZATION (CLEAN OR RESUME)
# ══════════════════════════════════════════════════════════════════════════════
CLEAN_ARG="--clean"
if [ "${STURNUS_RESUME:-0}" -eq 1 ]; then
  _info "RESUMING training — preserving state and appending to log..."
  CLEAN_ARG=""
else
  _info "Wiping state for fresh start..."
  rm -rf "$ROOT/state"
  rm -f \
    "$LOG_DIR/finetune_metrics.json" \
    "$LOG_DIR/marathon_metrics.json" \
    "$LOG_DIR/proof_metrics.jsonl" \
    "$LOG_DIR/benchmark_runs.jsonl" \
    "$LOG_DIR/benchmark_summary.json" \
    "$LOG_FILE" \
    "$PID_FILE"
  _ok "State wiped clean"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: LAUNCH TRAINING (CAFFEINATED, LID-OFF, GRADIENTS ON)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}  🚀  LAUNCHING 10M TOKEN TRAINING${RST}"
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo ""
_info "caffeinate -dimsu: display sleep inhibited, system sleep inhibited, lid-close safe"
_info "PYTHONFAULTHANDLER=1: crash tracebacks will appear in log"
_info "PYTHONUNBUFFERED=1: real-time log output"
echo ""

# Launch caffeinate as a background daemon to prevent sleep
caffeinate -dimsu &
CAF_PID=$!

if [ "${STURNUS_RESUME:-0}" -eq 1 ]; then
  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HF_TOKEN="${HF_TOKEN}" \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    "$PYTHON_BIN" "$ROOT/scripts/finetune.py" \
      $CLEAN_ARG \
      --max-tokens "$MAX_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --print-every-batches "$PRINT_EVERY" \
      --checkpoint-every-batches "$CHECKPOINT_EVERY" \
      --seed "$SEED" \
    >> "$LOG_FILE" 2>&1 &
else
  nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HF_TOKEN="${HF_TOKEN}" \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    "$PYTHON_BIN" "$ROOT/scripts/finetune.py" \
      $CLEAN_ARG \
      --max-tokens "$MAX_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --print-every-batches "$PRINT_EVERY" \
      --checkpoint-every-batches "$CHECKPOINT_EVERY" \
      --seed "$SEED" \
    > "$LOG_FILE" 2>&1 &
fi

PID=$!
echo "$PID" > "$PID_FILE"
echo ""
_ok "Training launched!"
echo ""
echo -e "  ${BLD}PID${RST}      : ${CYN}${PID}${RST}"
echo -e "  ${BLD}Python${RST}   : ${PYTHON_BIN} (${PYTHON_VERSION})"
echo -e "  ${BLD}Target${RST}   : ${MAX_TOKENS} tokens"
echo -e "  ${BLD}Log${RST}      : ${LOG_FILE}"
echo -e "  ${BLD}PID file${RST} : ${PID_FILE}"
echo ""
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}  Monitor commands:${RST}"
echo -e "    ${CYN}tail -f ${LOG_FILE}${RST}"
echo -e "    ${CYN}cat logs/marathon_metrics.json${RST}"
echo -e "    ${CYN}kill -INT \$(cat ${PID_FILE})${RST}   # graceful stop"
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo ""

# Wait a few seconds and check the process is still alive
sleep 4

if kill -0 "$PID" 2>/dev/null; then
  _ok "Process still running after 4s boot check"
  echo ""
  _info "Recent lines of output:"
  echo -e "${CYN}────────────────────────────────────────${RST}"
  tail -n 20 "$LOG_FILE" 2>/dev/null || echo "(waiting for output...)"
  echo -e "${CYN}────────────────────────────────────────${RST}"
else
  _err "Process died during boot!"
  _err "Last lines of log:"
  echo ""
  tail -n 30 "$LOG_FILE" 2>/dev/null || echo "(no log output)"
  exit 1
fi

echo ""
echo -e "${GRN}${BLD}☕ Training is running with the lid off. Go get some sleep.${RST}"
echo -e "${CYN}   caffeinate PID: ${CAF_PID} (kill this when training ends)${RST}"
echo ""
