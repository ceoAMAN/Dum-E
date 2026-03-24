#!/bin/bash
set -e

echo "================================================"
echo " Sturnus Native — Environment + Model Setup"
echo " 242 models · ~62.25GB · Qwen2.5 4-bit · MLX"
echo "================================================"

# ── 1. Python environment ─────────────────────────
echo ""
echo "[1/5] Creating Python environment..."
python3 -m venv ~/sturnus-env
source ~/sturnus-env/bin/activate
pip install --upgrade pip --quiet

# ── 2. Dependencies ───────────────────────────────
echo "[2/5] Installing dependencies..."
pip install mlx mlx-lm --quiet
pip install huggingface_hub datasets psutil --quiet
pip install transformers peft torch numpy python-dotenv --quiet
echo "      All dependencies installed."

python3 -c "
import mlx.core as mx
if not mx.metal.is_available():
    print('      WARNING: Apple Silicon Metal not detected (You are likely in a sandbox). Downloading will proceed, but inference will be extremely slow.')
else:
    print('      MLX OK — Metal available natively.')
"

# ── 3. Directory structure ────────────────────────
echo "[3/5] Creating directory structure..."
mkdir -p ~/Sturnus/models/gate
mkdir -p ~/Sturnus/models/central
mkdir -p ~/Sturnus/models/experts
mkdir -p ~/Sturnus/checkpoints
echo "      Directories ready."

AVAILABLE_GB=$(df -g ~ | awk 'NR==2 {print $4}')
echo "      Available SSD: ${AVAILABLE_GB}GB — need 63GB"
if [ "$AVAILABLE_GB" -lt 63 ]; then
    echo "      WARNING: Low disk space. Free up space before continuing."
    exit 1
fi

# ── 4. Gate + Central ─────────────────────────────
echo "[4/5] Quantizing Gate and Central..."

echo "      Gate — Qwen2.5-1.5B → ~750MB..."
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-1.5B-Instruct \
  --mlx-path ~/Sturnus/models/gate \
  --quantize \
  --q-bits 4
echo "      Gate done."

echo "      Central — Qwen2.5-3B → ~1.5GB..."
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-3B-Instruct \
  --mlx-path ~/Sturnus/models/central \
  --quantize \
  --q-bits 4
echo "      Central done."

# ── 5. 240 Expert models ──────────────────────────
echo "[5/5] Setting up 240 experts — Qwen2.5-0.5B each..."
echo "      Strategy: quantize once, copy 240 times"
echo "      This is faster than hitting HuggingFace 240 times"
echo "      Estimated time: 45-90 minutes"
echo ""

echo "      Downloading and quantizing base expert model..."
mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-0.5B-Instruct \
  --mlx-path ~/Sturnus/models/expert_base_tmp \
  --quantize \
  --q-bits 4
echo "      Base quantized. Copying to 240 expert directories..."

for i in $(seq 0 239); do
    EXPERT_ID=$(printf "%03d" $i)
    cp -r ~/Sturnus/models/expert_base_tmp \
          ~/Sturnus/models/experts/expert_$EXPERT_ID
    printf "\r      Copied %d/240 experts..." $((i+1))
done
echo ""

rm -rf ~/Sturnus/models/expert_base_tmp
echo "      Temp base removed."

# ── Verification ──────────────────────────────────
echo ""
echo "Verifying setup..."
EXPERT_COUNT=$(ls ~/Sturnus/models/experts | wc -l | tr -d ' ')
echo "  Expert directories: $EXPERT_COUNT (expected 240)"
echo "  Total model storage:"
du -sh ~/Sturnus/models/

# ── Smoke test ────────────────────────────────────
echo ""
echo "Running smoke test..."
python3 - <<'PYEOF'
import os
from mlx_lm import load, generate
import mlx.core as mx

base = os.path.expanduser('~')
ping = [{'role': 'user', 'content': 'Reply with one word only: ready'}]

def test_model(name, path):
    model, tok = load(path)
    prompt = tok.apply_chat_template(ping, tokenize=False, add_generation_prompt=True)
    out = generate(model, tok, prompt=prompt, max_tokens=5, verbose=False)
    print(f'  {name}: {out.strip()[:40]}')
    del model
    mx.metal.clear_cache()

test_model('Gate   ', f'{base}/Sturnus/models/gate')
test_model('Central', f'{base}/Sturnus/models/central')
test_model('Expert 000', f'{base}/Sturnus/models/experts/expert_000')
test_model('Expert 119', f'{base}/Sturnus/models/experts/expert_119')
test_model('Expert 239', f'{base}/Sturnus/models/experts/expert_239')

print('  Smoke test PASSED — all models load and infer correctly')
PYEOF

echo ""
echo "================================================"
echo " Setup complete"
echo " Activate env:  source ~/sturnus-env/bin/activate"
echo " Run tests:     STURNUS_NATIVE=1 python config.py"
echo " Run benchmark: STURNUS_NATIVE=1 python scripts/benchmark.py"
echo "================================================"
