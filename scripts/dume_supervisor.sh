#!/bin/bash
# Dum-E self-healing training supervisor.
#
# Keeps the marathon alive until TARGET tokens, resuming from the last checkpoint
# after ANYTHING kills it — Metal OOM, session teardown, logout, even a reboot.
# Designed to be run every minute by cron: it is idempotent (does nothing if a run
# is already alive or the target is reached) and only ever launches ONE run.
#
#   install:  (crontab -l 2>/dev/null; echo "* * * * * /bin/bash /Users/aman/Sturnus/scripts/dume_supervisor.sh") | crontab -
#   stop:     crontab -r         (or remove the dume line);  then pkill -f finetune.py
#
set -uo pipefail
ROOT="/Users/aman/Sturnus"
PY="/Users/aman/brian-env/bin/python"
TARGET=1000000
LOG="$ROOT/logs/dume-1m.log"
MARKER="$ROOT/.dume_run_started"      # lives OUTSIDE state/ so --clean can't wipe it
cd "$ROOT" || exit 0

# 1. Done already? Stop launching (and let cron keep no-op'ing).
tok=$("$PY" -c "import json;print(int(json.load(open('logs/marathon_metrics.json')).get('total_tokens',0)))" 2>/dev/null || echo 0)
if [ "$tok" -ge "$TARGET" ] 2>/dev/null; then
  exit 0
fi

# 2. Already running? Leave it be.
if pgrep -f "scripts/finetune.py" >/dev/null 2>&1; then
  exit 0
fi

# 3. Not running and not done -> (re)launch a RESUME. Wipe (--clean) only on the very
#    first ever start; every restart after that resumes from the last checkpoint.
clean=""
if [ ! -f "$MARKER" ]; then
  clean="--clean"
  touch "$MARKER"
fi
mkdir -p "$ROOT/logs"
echo "[supervisor] $(date '+%F %T') (re)launch tok=${tok} clean='${clean}'" >> "$LOG"
nohup caffeinate -dimsu "$PY" scripts/finetune.py $clean \
  --max-tokens "$TARGET" --print-every-batches 10 --checkpoint-every-batches 50 --seed 42 \
  >> "$LOG" 2>&1 &
exit 0
