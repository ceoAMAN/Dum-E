from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import configs

@dataclass
class TrainConfig:
    model_id: str
    output_dir: Path
    batch_size: int
    max_steps: int
    grad_accum_steps: int
    learning_rate: float
    max_length: int
    save_every: int
def resolve_model_id(default_id: str) -> str:
    override = os.getenv("STURNUS_TRAIN_MODEL_ID")
    if override:
        return override
    if os.getenv("STURNUS_TINY", "0") == "1":
        return configs.GATE_MODEL_ID
    return default_id
def load_mlx_model(model_id: str):
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_id)
    return model, tokenizer

def train_loop(
    model: nn.Module,
    tokenizer,
    batch_iter: Iterator[Dict[str, Any]],
    cfg: TrainConfig,
    loss_fn=None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    step = 0
    accum_loss = 0.0
    start_time = __import__("time").time()

    def _loss_fn(m: nn.Module, tokens: mx.array) -> mx.array:
        if loss_fn is not None:
            return loss_fn(m, tokens)
        output = m(tokens)
        if output.ndim == 3:
            logits = output[:, :-1, :]
            targets = tokens[:, 1:]
            return mx.mean(nn.losses.cross_entropy(logits, targets))
        return mx.mean(output)

    loss_and_grad_fn = nn.value_and_grad(model, _loss_fn)

    while step < cfg.max_steps:
        batch = next(batch_iter)
        input_ids = batch["input_ids"]
        tokens = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)

        loss, grads = loss_and_grad_fn(model, tokens)
        mx.eval(loss)
        loss_value = float(loss.item())
        accum_loss += loss_value
        if (step + 1) % cfg.grad_accum_steps == 0:
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            accum_loss = 0.0
        elapsed = __import__("time").time() - start_time
        tokens_in_batch = int(tokens.size)
        tok_per_sec = tokens_in_batch / max(elapsed / max(step + 1, 1), 1e-6)
        print(
            f"[train] step={step:>4d}/{cfg.max_steps:<4d} | "
            f"loss={loss_value:.4f} | "
            f"batch_tokens={tokens_in_batch:>4d} | "
            f"tok/s={tok_per_sec:.1f}"
        )
        import time
        if not hasattr(cfg, 'last_save_time'):
            cfg.last_save_time = time.time()
        now = time.time()
        if cfg.save_every > 0 and (now - cfg.last_save_time) >= cfg.save_every:
            cfg.last_save_time = now
            print(f"[train] Saving checkpoint at step {step}...")
            from mlx.utils import tree_flatten
            flat_params = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(cfg.output_dir / "weights.safetensors"), flat_params)
        step += 1
    print(f"[train] Completed {step} steps")
