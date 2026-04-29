from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import configs
import data
from scripts.train_common import TrainConfig, load_mlx_model, train_loop, resolve_model_id

def run() -> None:
    print("[train_phase1] Central fine-tuning (MLX)")
    model_id = resolve_model_id(configs.CENTRAL_TRAIN_MODEL_ID)
    batch_size = int(os.getenv("STURNUS_TRAIN_BATCH_SIZE", "2"))
    max_steps = int(os.getenv("STURNUS_TRAIN_STEPS", "100"))
    grad_accum = int(os.getenv("STURNUS_TRAIN_ACCUM", "4"))
    data.authenticate_huggingface()
    batch_iter = data.iter_mixture_token_batches(
        model_id=model_id,
        batch_size=batch_size,
        max_length=configs.MAX_SEQ_LEN,
        seed=42,
    )
    cfg = TrainConfig(
        model_id=model_id,
        output_dir=configs.CHECKPOINT_DIR / "central",
        batch_size=batch_size,
        max_steps=max_steps,
        grad_accum_steps=grad_accum,
        learning_rate=configs.LEARNING_RATE,
        max_length=configs.MAX_SEQ_LEN,
        save_every=int(os.getenv("STURNUS_SAVE_EVERY", "0")),
    )
    model, tokenizer = load_mlx_model(model_id)
    print(f"[train_phase1] Loaded {model_id}")
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.freeze()
    lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    linear_to_lora_layers(model, num_layers, lora_config)
    weights_path = cfg.output_dir / "weights.safetensors"
    if weights_path.exists():
        model.load_weights(str(weights_path), strict=False)
    model.train()
    train_loop(model, tokenizer, batch_iter, cfg)
    print("[train_phase1] Done")

if __name__ == "__main__":
    run()
