#!/bin/bash
# =============================================================================
#  Dum-E — Caffeinated 1M-Token Training Run
# =============================================================================
#  Launches the real marathon DETACHED, with the Mac kept awake (lid-close safe),
#  so a multi-hour run survives sleep. The training loop skips any dataset that
#  fails to stream, so no heavy pre-validation is needed.
#
#  Usage:
#    bash scripts/run_caffeinated_1m.sh                 # 1,000,000 tokens
#    DUME_MAX_TOKENS=2000000 bash scripts/run_caffeinated_1m.sh
#    DUME_RESUME=1 bash scripts/run_caffeinated_1m.sh   # continue, don't wipe state
#
#  Monitor:
#    tail -f logs/dume-1m.log
#    cat logs/benchmarks_summary.json        # latest paper metrics
#    kill -INT $(cat logs/dume-1m.pid)       # graceful stop
# =============================================================================
set -Eeuo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'
_ok()   { echo -e "${GRN}[  ok  ]${RST} $*"; }
_warn() { echo -e "${YLW}[ warn ]${RST} $*"; }
_err()  { echo -e "${RED}[error ]${RST} $*"; }
_info() { echo -e "${CYN}[ info ]${RST} $*"; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── locate the venv python (brian-env is the real one) ───────────────────────
PYTHON_BIN=""
for c in "/Users/aman/brian-env/bin/python" \
         "$ROOT/brian-env/bin/python" \
         "$ROOT/sturnus-env312/bin/python" \
         "$ROOT/sturnus_env/bin/python"; do
  if [ -x "$c" ]; then PYTHON_BIN="$c"; break; fi
done
[ -z "$PYTHON_BIN" ] && { _err "No venv python found (looked for brian-env first)"; exit 1; }
_ok "python: $PYTHON_BIN"

# ── tunables ─────────────────────────────────────────────────────────────────
MAX_TOKENS="${DUME_MAX_TOKENS:-1000000}"
PRINT_EVERY="${DUME_PRINT_EVERY:-10}"
CHECKPOINT_EVERY="${DUME_CHECKPOINT_EVERY:-100}"
SEED="${DUME_SEED:-42}"
LOG_DIR="$ROOT/logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dume-1m.log"
PID_FILE="$LOG_DIR/dume-1m.pid"

echo ""
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "${BLD}  🤖☕  Dum-E — Caffeinated 1M-Token Run${RST}"
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo -e "  Target     : ${CYN}${MAX_TOKENS}${RST} tokens"
echo -e "  Checkpoint : every ${CHECKPOINT_EVERY} batches   Print: every ${PRINT_EVERY}"
echo -e "  Log        : ${LOG_FILE}"
echo -e "${BLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}\n"

# ── preflight ────────────────────────────────────────────────────────────────
if [ -f "$ROOT/.env.local" ]; then set -a; source "$ROOT/.env.local"; set +a; _ok ".env.local loaded"; else _warn "no .env.local"; fi
[ -z "${HF_TOKEN:-}" ] && { _err "HF_TOKEN not set (put it in .env.local)"; exit 1; }
_ok "HF_TOKEN set (${HF_TOKEN:0:8}...)"
"$PYTHON_BIN" -c "import mlx.core" 2>/dev/null && _ok "mlx importable" || { _err "mlx not importable"; exit 1; }

# ── clear stale processes ────────────────────────────────────────────────────
set +e
pkill -INT -f 'scripts/finetune.py' 2>/dev/null
pkill -f 'caffeinate' 2>/dev/null
sleep 1
set -e

# ── clean vs resume ──────────────────────────────────────────────────────────
CLEAN_ARG="--clean"
if [ "${DUME_RESUME:-0}" -eq 1 ]; then
  CLEAN_ARG=""; _info "RESUME mode — keeping state, appending to log"
else
  _info "Fresh run — finetune --clean wipes state + per-run logs"
  rm -f "$LOG_FILE" "$PID_FILE"
fi

# ── launch caffeinated + detached (with auto-resume loop) ────────────────────
caffeinate -dimsu & CAF_PID=$!
nohup bash "$ROOT/scripts/_dume_loop.sh" \
  "$PYTHON_BIN" "$MAX_TOKENS" "$PRINT_EVERY" "$CHECKPOINT_EVERY" "$SEED" "$CLEAN_ARG" \
  > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
# Detach from this shell's job control so the run survives the launcher exiting.
disown "$PID" 2>/dev/null || true
disown "$CAF_PID" 2>/dev/null || true

_ok "Dum-E launched — PID ${PID}, caffeinate PID ${CAF_PID}"

# ── boot check ───────────────────────────────────────────────────────────────
sleep 6
if kill -0 "$PID" 2>/dev/null; then
  _ok "alive after boot check"
  echo -e "${CYN}──── recent log ────${RST}"; tail -n 15 "$LOG_FILE" 2>/dev/null
else
  _err "process died during boot — last log:"; tail -n 30 "$LOG_FILE" 2>/dev/null; exit 1
fi
echo ""
echo -e "${GRN}${BLD}☕ Dum-E is training (auto-resumes on transient GPU errors). Monitor: tail -f ${LOG_FILE}${RST}"
echo -e "${CYN}   stop: pkill -INT -f finetune.py   (graceful; loop won't resume)  then  pkill caffeinate${RST}\n"
