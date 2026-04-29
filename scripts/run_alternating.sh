#!/bin/bash
set -e

: "${HF_TOKEN:?Please export HF_TOKEN before running run_alternating.sh}"
echo "[boot] Using Hugging Face token for authentication..."

echo ""
echo "==========================================================="
echo " STAGE 1: CENTRAL SYNTHESIS (Target: 50,000 tokens)"
echo "==========================================================="
STURNUS_TRAIN_STEPS=12 STURNUS_SAVE_EVERY=60 python3 scripts/train_phase1.py

echo ""
echo "==========================================================="
echo " STAGE 2: TIMELINE B (Target: 500,000 tokens)"
echo "==========================================================="
python3 finetune.py \
  --max-tokens 500000 \
  --batch-size 256 \
  --checkpoint-interval 60 \
  --seed 42

echo ""
echo "==========================================================="
echo " STAGE 3: BENCHMARK"
echo "==========================================================="
python3 scripts/benchmark.py
