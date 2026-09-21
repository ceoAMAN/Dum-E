#!/bin/bash
# Start a run from NOTHING.
#
# A run inherits more than it looks like it does: the adapters under
# state/dume/ckpt, the clock and every regulator's history in state.pkl, and
# the geometry those were formed against. Half-cleaning gives a run that is
# neither fresh nor a continuation, and no way to tell afterwards which it was.
# So: archive the logs, move the whole of state/dume aside, re-form, launch.
#
# Nothing is deleted. 3.5 GB is cheap next to a run that cannot be reproduced.
#
#   TARGET=500000 bash scripts/fresh_run.sh
set -eu
REPO=/Users/aman/Sturnus
PY=/Users/aman/brian-env/bin/python
TARGET=${TARGET:-500000}
SAMPLES=${SAMPLES:-200}
TAG=${TAG:-$(date '+%Y%m%d_%H%M')}
cd "$REPO"

# SEQUENTIAL ONLY: two model processes do not fit in 16 GB.
if pgrep -f "dume.main" >/dev/null; then
  echo "REFUSING: something is still training —"
  pgrep -fl "dume.main"
  exit 1
fi

DEST="logs/archive/$TAG"
mkdir -p "$DEST"
# ...except this script's own stdout, which is being written to right now:
# archiving it leaves the transcript of the launch inside the PREVIOUS run's
# archive, following a moved inode. Redirect to exactly this name.
find logs -maxdepth 1 -type f \( -name '*.log' -o -name '*.pid' -o -name '*.out' \) \
     -not -name 'fresh_run.out' -exec mv {} "$DEST/" \;
echo "logs        -> $DEST"

if [ -d state/dume ]; then
  mv state/dume "state/dume.$TAG"
  echo "state+ckpt  -> state/dume.$TAG"
fi

# what this run is, recorded where the run's own logs will be
git rev-parse HEAD > logs/COMMIT
git status --porcelain > logs/DIRTY
[ -s logs/DIRTY ] && echo "WARNING: tree is dirty; this run is not reproducible from a commit"

# the geometry lived in the state just moved aside, so it has to be re-formed
caffeinate -i "$PY" -u -m dume.main form --samples "$SAMPLES" 2>&1 | tee logs/form.log

echo "launching to TARGET=$TARGET at $(git rev-parse --short HEAD)"
exec env TARGET="$TARGET" bash scripts/run_to_500k.sh
