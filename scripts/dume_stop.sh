#!/bin/bash
# One-shot clean stop for the Dum-E marathon, fired by cron at the cutoff time.
# 1) stops the supervisor from relaunching, 2) kills training + caffeinate (so the
# laptop can sleep), 3) snapshots the paper logs, 4) removes its own + the
# supervisor cron entries so nothing runs again.
set -uo pipefail
ROOT="/Users/aman/Sturnus"
cd "$ROOT" || exit 0

# 1+4. remove BOTH dume cron lines first so the supervisor can't respawn training.
crontab -l 2>/dev/null | grep -v 'dume_supervisor\|dume_stop' | crontab - 2>/dev/null || true

# 2. kill training + caffeinate; re-kill for a few seconds to catch a last respawn.
for _ in $(seq 1 8); do
  pkill -f 'scripts/finetune.py' 2>/dev/null
  pkill -x caffeinate 2>/dev/null
  sleep 1
done

# 3. snapshot the paper artefacts (small logs) into analysis/ with a timestamp.
SNAP="$ROOT/analysis/run_$(date '+%Y%m%d_%H%M')"
mkdir -p "$SNAP"
for f in benchmarks.csv benchmarks_summary.json trajectory.csv eval.csv \
         lambda_trajectory.jsonl per_domain_k.json marathon_metrics.json; do
  cp -f "$ROOT/logs/$f" "$SNAP/" 2>/dev/null || true
done
tok=$(python3 -c "import json;print(json.load(open('$ROOT/logs/marathon_metrics.json'))['total_tokens'])" 2>/dev/null || echo '?')
echo "[stop] $(date '+%F %T') marathon stopped at ${tok} tok; snapshot -> ${SNAP}; trained weights remain in state/checkpoints/" >> "$ROOT/logs/dume-1m.log"
