#!/bin/bash
# Keep training until the clock reaches TARGET tokens, one run after another.
#
# SEQUENTIAL ONLY. Two model processes will not fit in 16 GB, so this waits for
# each run to exit before starting the next and refuses to start if anything is
# already training.
set -u
REPO=/Users/aman/Sturnus
PY=/Users/aman/brian-env/bin/python
TARGET=${TARGET:-500000}
MAX_CYCLES=${MAX_CYCLES:-10}
TOK_PER_BATCH_GUESS=${TOK_PER_BATCH_GUESS:-146}
SUP=$REPO/logs/supervisor.log

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUP"; }

stat_of() {   # prints "clock batch"
  "$PY" - <<PYEOF
import sys; sys.path.insert(0, "$REPO")
from dume import state
b = state.load()
print(int(b["clock"]) if b else 0, int(b["batch"]) if b else 0)
PYEOF
}

# wait for whatever is training right now
if [ -f "$REPO/logs/dume-fresh-500k.pid" ]; then
  P=$(cat "$REPO/logs/dume-fresh-500k.pid")
  if ps -p "$P" >/dev/null 2>&1; then
    say "waiting for the running job (pid $P) to finish"
    while ps -p "$P" >/dev/null 2>&1; do sleep 30; done
    say "pid $P exited"
  fi
fi

for cycle in $(seq 1 "$MAX_CYCLES"); do
  read -r CLOCK BATCH <<< "$(stat_of)"
  say "cycle $cycle: clock=$CLOCK batch=$BATCH target=$TARGET"

  if [ "$CLOCK" -ge "$TARGET" ]; then
    say "TARGET REACHED at $CLOCK tokens. done."
    exit 0
  fi

  # size the next run from the rate this state actually achieved
  if [ "$BATCH" -gt 0 ]; then
    RATE=$(( CLOCK / BATCH )); [ "$RATE" -lt 1 ] && RATE=$TOK_PER_BATCH_GUESS
  else
    RATE=$TOK_PER_BATCH_GUESS
  fi
  NEED=$(( (TARGET - CLOCK + RATE - 1) / RATE ))
  NEED=$(( NEED * 110 / 100 ))                 # 10% headroom: the rate drifts
  [ "$NEED" -lt 50 ]   && NEED=50
  [ "$NEED" -gt 4000 ] && NEED=4000
  say "cycle $cycle: ${RATE} tok/batch -> launching $NEED batches"

  LOG=$REPO/logs/dume-cycle$cycle.log
  caffeinate -i "$PY" -u -m dume.main train --batches "$NEED" > "$LOG" 2>&1
  RC=$?
  read -r NEWCLOCK NEWBATCH <<< "$(stat_of)"
  say "cycle $cycle ended rc=$RC, clock $CLOCK -> $NEWCLOCK ($(( NEWCLOCK - CLOCK )) tokens)"

  if [ "$NEWCLOCK" -le "$CLOCK" ]; then
    say "ABORT: a whole cycle advanced the clock by nothing. see $LOG"
    exit 1
  fi
  if [ "$RC" -ne 0 ]; then
    say "ABORT: train exited $RC. see $LOG"
    exit "$RC"
  fi
done
say "stopped after $MAX_CYCLES cycles without reaching $TARGET"
exit 1
