from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import mlx.core as mx
import mlx.nn as nn
import configs
import data
from scripts.train_common import load_mlx_model, resolve_model_id

def _cross_expert_alignment_loss(hidden_batch: mx.array) -> mx.array:
    if hidden_batch.shape[0] < 2:
        return mx.array(0.0)
    norms = mx.linalg.norm(hidden_batch, axis=-1, keepdims=True) + 1e-8
    normed = hidden_batch / norms
    sim_matrix = mx.matmul(normed, normed.T)
    mask = 1.0 - mx.eye(sim_matrix.shape[0])
    sim_matrix = sim_matrix * mask
    penalty = mx.sum(sim_matrix) / (mx.sum(mask) + 1e-9)
    return penalty

def run() -> None:
    print("[train_phase3] Expert fine-tuning (3 losses, MLX)")
    model_id = resolve_model_id(configs.EXPERT_TRAIN_MODEL_ID)
    batch_size = int(os.getenv("STURNUS_TRAIN_BATCH_SIZE", "2"))
    max_steps = int(os.getenv("STURNUS_TRAIN_STEPS", "50"))
    distill_weight = float(os.getenv("STURNUS_DISTILL_WEIGHT", "0.3"))
    alignment_weight = float(os.getenv("STURNUS_ALIGN_WEIGHT", "0.1"))
    _group_limit_env = os.getenv("STURNUS_EXPERT_GROUP_LIMIT")
    group_limit: Optional[int] = int(_group_limit_env) if _group_limit_env else None
    data.authenticate_huggingface()
    for idx, group_name in enumerate(configs.EXPERT_GROUPS.keys()):
        if group_limit is not None and idx >= group_limit:
            break
        print(f"[train_phase3] Training group: {group_name}")
        batch_iter = data.iter_group_token_batches(
            group_name=group_name,
            model_id=model_id,
            batch_size=batch_size,
            max_length=configs.MAX_SEQ_LEN,
        )
        model, _ = load_mlx_model(model_id)
        out_dir = configs.CHECKPOINT_DIR / "experts" / group_name
        out_dir.mkdir(parents=True, exist_ok=True)
        step = 0
        while step < max_steps:
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            output = model(input_ids)
            mx.eval(output)
            if output.ndim == 3:
                logits = output[:, :-1, :]
                targets = input_ids[:, 1:]
                task_loss = mx.mean(nn.losses.cross_entropy(logits, targets))
                soft_logits = logits / 2.0
                soft_targets = mx.softmax(mx.stop_gradient(soft_logits), axis=-1)
                log_probs = mx.log(mx.softmax(soft_logits, axis=-1) + 1e-10)
                distill_loss = -mx.mean(mx.sum(soft_targets * log_probs, axis=-1)) * 4.0
                hidden = mx.mean(output, axis=1)
                alignment_loss = _cross_expert_alignment_loss(hidden)
            else:
                task_loss = mx.mean(output)
                distill_loss = mx.array(0.0)
                alignment_loss = mx.array(0.0)
            loss = task_loss + distill_weight * distill_loss + alignment_weight * alignment_loss
            mx.eval(loss)
            if step % 10 == 0:
                print(
                    f"[train_phase3] {group_name} step={step} total={float(loss.item()):.4f} "
                    f"task={float(task_loss.item()):.4f} distill={float(distill_loss.item()):.4f} "
                    f"align={float(alignment_loss.item()):.4f}"
                )
            step += 1
        print(f"[train_phase3] {group_name} completed {step} steps, saved to {out_dir}")
    print("[train_phase3] Done")

if __name__ == "__main__":
    run()
