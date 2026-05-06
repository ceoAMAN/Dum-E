#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT/sturnus-env312/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$ROOT/sturnus-env/bin/python"
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] No venv python found — run setup first"
  exit 1
fi

: "${HF_TOKEN:?Please export HF_TOKEN before running run_alternating.sh}"
echo "[boot] Using Hugging Face token for authentication..."
echo "[boot] Python: $PYTHON_BIN"

cd "$ROOT"

echo ""
echo "==========================================================="
echo " STAGE 1: CENTRAL SYNTHESIS (Phase 1 warmup)"
echo "==========================================================="
STURNUS_TRAIN_STEPS=12 STURNUS_SAVE_EVERY=60 \
  HF_TOKEN="$HF_TOKEN" "$PYTHON_BIN" scripts/train_phase1.py

echo ""
echo "==========================================================="
echo " STAGE 2: TIMELINE B (Target: 500,000 tokens)"
echo "==========================================================="
HF_TOKEN="$HF_TOKEN" "$PYTHON_BIN" scripts/finetune.py \
  --max-tokens 500000 \
  --batch-size 256 \
  --checkpoint-every-batches 60 \
  --seed 42

echo ""
echo "==========================================================="
echo " STAGE 3: BENCHMARK (all 3 loops)"
echo "==========================================================="
HF_TOKEN="$HF_TOKEN" "$PYTHON_BIN" scripts/benchmark.py
