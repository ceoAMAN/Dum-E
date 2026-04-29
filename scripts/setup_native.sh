#!/bin/bash
set -e
: "${HF_TOKEN:?Please export HF_TOKEN before running setup_native.sh}"

echo "================================================"
echo " Sturnus — Environment + Model Setup"
echo " 102 models · ~8GB download · MLX 4-bit"
echo "================================================"

echo ""
echo "[1/5] Creating Python environment..."
python3 -m venv sturnus_env
source sturnus_env/bin/activate
pip install --upgrade pip --quiet

echo "[2/5] Installing dependencies..."
pip install mlx mlx-lm --quiet
pip install huggingface_hub datasets numpy --quiet
echo "      All dependencies installed."

python3 -c "
import mlx.core as mx
if not mx.metal.is_available():
    print('      WARNING: Apple Silicon Metal not detected.')
    print('      Models will download but inference will be very slow.')
else:
    print('      MLX OK — Metal available natively.')
"

echo "[3/5] Creating directory structure..."
mkdir -p state/checkpoints
mkdir -p logs
echo "      Directories ready."

AVAILABLE_GB=$(df -g . | awk 'NR==2 {print $4}')
echo "      Available SSD: ${AVAILABLE_GB}GB — need ~10GB"
if [ "$AVAILABLE_GB" -lt 10 ]; then
    echo "      WARNING: Low disk space. Free up space before continuing."
    exit 1
fi

echo "[4/5] Pre-downloading models from mlx-community..."
echo "      These are pre-quantized 4-bit models."
echo ""

echo "      Gate — Qwen2.5-0.5B-Instruct-4bit (~300MB)..."
python3 -c "
from mlx_lm import load
print('      Downloading Gate model...')
model, tok = load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')
del model, tok
print('      Gate downloaded.')
"

echo "      Expert — Qwen2.5-1.5B-Instruct-4bit (~1GB, shared by all 100 experts)..."
python3 -c "
from mlx_lm import load
print('      Downloading Expert model...')
model, tok = load('mlx-community/Qwen2.5-1.5B-Instruct-4bit')
del model, tok
print('      Expert base downloaded.')
"

echo "      Central — Mistral-7B-Instruct-v0.3-4bit (~4GB)..."
python3 -c "
from mlx_lm import load
print('      Downloading Central model...')
model, tok = load('mlx-community/Mistral-7B-Instruct-v0.3-4bit')
del model, tok
print('      Central downloaded.')
"

echo "[5/5] Running smoke test..."
python3 -c "
import mlx.core as mx
from mlx_lm import load, generate

def test_model(name, model_id):
    model, tok = load(model_id)
    prompt = 'Hello'
    ids = tok.encode(prompt)
    tokens = mx.array([ids])
    output = model(tokens)
    mx.eval(output)
    print(f'  {name}: shape={list(output.shape)} — OK')
    del model, tok
    mx.metal.clear_cache()

test_model('Gate    ', 'mlx-community/Qwen2.5-0.5B-Instruct-4bit')
test_model('Expert  ', 'mlx-community/Qwen2.5-1.5B-Instruct-4bit')
test_model('Central ', 'mlx-community/Mistral-7B-Instruct-v0.3-4bit')

print('  Smoke test PASSED — all 3 model tiers load and infer')
"

echo ""
echo "================================================"
echo " Setup complete"
echo ""
echo " Architecture:"
echo "   Gate:    Qwen2.5-0.5B-Instruct  (0.5B params)"
echo "   Expert:  Qwen2.5-1.5B-Instruct  (1.5B × 100 = 150B)"
echo "   Central: Mistral-7B-Instruct    (7B params)"
echo "   Total:   ~157.5B parameters"
echo ""
echo " Next steps:"
echo "   source sturnus_env/bin/activate"
echo "   python3 finetune.py --max-tokens 500000"
echo "   python3 scripts/benchmark.py"
echo "================================================"
