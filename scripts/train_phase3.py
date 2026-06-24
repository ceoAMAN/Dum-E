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
import mlx.optimizers as optim
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
        from mlx_lm.tuner.utils import linear_to_lora_layers
        model.freeze()
        lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
        num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
        linear_to_lora_layers(model, num_layers, lora_config)
        weights_path = out_dir / "weights.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path), strict=False)
        model.train()

        optimizer = optim.Adam(learning_rate=configs.LEARNING_RATE)

        def _expert_loss(m: nn.Module, input_ids: mx.array) -> mx.array:
            # Single backbone pass: get hidden states, then derive logits cheaply
            if hasattr(m, 'model'):
                hidden_out = m.model(input_ids)
                # Derive logits from hidden states (just the lm_head, no second pass)
                if hasattr(m, 'lm_head'):
                    output = m.lm_head(hidden_out)
                else:
                    output = m(input_ids)
            else:
                output = m(input_ids)
                hidden_out = output
            if output.ndim == 3:
                logits = output[:, :-1, :]
                targets = input_ids[:, 1:]
                task_loss = mx.mean(nn.losses.cross_entropy(logits, targets))
                soft_logits = logits / 2.0
                soft_targets = mx.softmax(mx.stop_gradient(soft_logits), axis=-1)
                log_probs = mx.log(mx.softmax(soft_logits, axis=-1) + 1e-10)
                distill_loss = -mx.mean(mx.sum(soft_targets * log_probs, axis=-1)) * 4.0
                # Use backbone hidden states for alignment (not vocab logits)
                if hidden_out.ndim == 3:
                    hidden = mx.mean(hidden_out, axis=1)
                else:
                    hidden = hidden_out
                alignment_loss = _cross_expert_alignment_loss(hidden)
            else:
                task_loss = mx.mean(output)
                distill_loss = mx.array(0.0)
                alignment_loss = mx.array(0.0)
            return task_loss + distill_weight * distill_loss + alignment_weight * alignment_loss

        loss_and_grad_fn = nn.value_and_grad(model, _expert_loss)

        step = 0
        while step < max_steps:
            batch = next(batch_iter)
            input_ids = batch["input_ids"]
            loss, grads = loss_and_grad_fn(model, input_ids)
            mx.eval(loss)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            if step % 10 == 0:
                print(f"[train_phase3] {group_name} step={step} loss={float(loss.item()):.4f}")
            step += 1

        from mlx.utils import tree_flatten
        mx.save_safetensors(str(out_dir / "weights.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        print(f"[train_phase3] {group_name} completed {step} steps, saved to {out_dir}")
    print("[train_phase3] Done")

if __name__ == "__main__":
    run()
