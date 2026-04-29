from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mlx.core as mx
import mlx.nn as nn
import configs
import data
from scripts.train_common import load_mlx_model, resolve_model_id

def _target_k(attention_mask: mx.array) -> mx.array:
    lengths = mx.sum(attention_mask, axis=1).astype(mx.float32)
    ratios = lengths / float(configs.MAX_SEQ_LEN)
    ks = mx.clip((ratios * configs.K_MAX), configs.K_MIN, configs.K_MAX).astype(mx.int32)
    return ks
def _routing_quality_loss(expert_logits: mx.array) -> mx.array:
    probs = mx.softmax(expert_logits, axis=-1)
    top_k = min(configs.K_DEFAULT, configs.NUM_EXPERTS)
    sorted_probs = mx.sort(probs, axis=-1)
    top_vals = sorted_probs[:, -top_k:]
    concentration = mx.sum(top_vals, axis=-1)
    ideal = mx.ones_like(concentration) * 0.8
    return mx.mean((concentration - ideal) ** 2)
def _adaptive_context_loss(hidden: mx.array) -> mx.array:
    norms = mx.linalg.norm(hidden, axis=-1)
    mean_norm = mx.mean(norms)
    return mx.sqrt(mx.mean((norms - mean_norm) ** 2))

def run() -> None:
    print("[train_phase2] Gate fine-tuning (4 losses, MLX)")
    model_id = resolve_model_id(configs.GATE_TRAIN_MODEL_ID)
    batch_size = int(os.getenv("STURNUS_TRAIN_BATCH_SIZE", "2"))
    max_steps = int(os.getenv("STURNUS_TRAIN_STEPS", "100"))
    lb_weight = float(os.getenv("STURNUS_LB_WEIGHT", "0.1"))
    rq_weight = float(os.getenv("STURNUS_RQ_WEIGHT", "0.05"))
    ac_weight = float(os.getenv("STURNUS_AC_WEIGHT", "0.02"))
    data.authenticate_huggingface()
    batch_iter = data.iter_mixture_token_batches(
        model_id=model_id,
        batch_size=batch_size,
        max_length=configs.MAX_SEQ_LEN,
        seed=123,
    )
    model, _ = load_mlx_model(model_id)
    print(f"[train_phase2] Loaded {model_id}")
    out_dir = configs.CHECKPOINT_DIR / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    while step < max_steps:
        batch = next(batch_iter)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        output = model(input_ids)
        mx.eval(output)
        if output.ndim == 3:
            hidden = mx.mean(output, axis=1)
        else:
            hidden = output
        expert_logits = hidden[:, :configs.NUM_EXPERTS] if hidden.shape[-1] >= configs.NUM_EXPERTS else hidden
        k_logits = hidden[:, :configs.K_MAX + 1] if hidden.shape[-1] >= configs.K_MAX + 1 else hidden
        target_k = _target_k(attention_mask)
        k_loss = mx.mean(nn.losses.cross_entropy(k_logits, target_k))
        probs = mx.softmax(expert_logits, axis=-1)
        entropy = -mx.mean(mx.sum(probs * mx.log(probs + 1e-8), axis=-1))
        load_balance_loss = -entropy
        rq_loss = _routing_quality_loss(expert_logits)
        ac_loss = _adaptive_context_loss(hidden)
        loss = k_loss + lb_weight * load_balance_loss + rq_weight * rq_loss + ac_weight * ac_loss
        mx.eval(loss)
        if step % 10 == 0:
            print(
                f"[train_phase2] step={step} total={float(loss.item()):.4f} "
                f"k={float(k_loss.item()):.4f} lb={float(load_balance_loss.item()):.4f} "
                f"rq={float(rq_loss.item()):.4f} ac={float(ac_loss.item()):.4f}"
            )
        step += 1
    print(f"[train_phase2] Completed {step} steps, saved to {out_dir}")

if __name__ == "__main__":
    run()
