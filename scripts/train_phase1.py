# pyre-unsafe
"""Phase 1 training (Central fine-tuning)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
import data
from scripts.train_common import TrainConfig, build_model, train_loop, resolve_model_id


def run() -> None:
    print("[train_phase1] Central fine-tuning")
    model_id = resolve_model_id(configs.CENTRAL_TRAIN_MODEL_ID)

    batch_size = int(os.getenv("STURNUS_TRAIN_BATCH_SIZE", "2"))
    max_steps = int(os.getenv("STURNUS_TRAIN_STEPS", "100"))
    grad_accum = int(os.getenv("STURNUS_TRAIN_ACCUM", "4"))
    device = os.getenv("STURNUS_TRAIN_DEVICE", "cpu")

    batch_iter = data.iter_mixture_token_batches(
        model_id=model_id,
        batch_size=batch_size,
        max_length=configs.MAX_SEQ_LEN,
        seed=42,
    )

    def _to_torch(batch):
        return {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in batch.items()}

    batch_iter = (_to_torch(b) for b in batch_iter)

    cfg = TrainConfig(
        model_id=model_id,
        output_dir=configs.CHECKPOINT_DIR / "central",
        batch_size=batch_size,
        max_steps=max_steps,
        grad_accum_steps=grad_accum,
        learning_rate=configs.LEARNING_RATE,
        max_length=configs.MAX_SEQ_LEN,
        device=device,
        lora_r=configs.LORA_R,
        lora_alpha=configs.LORA_ALPHA,
        lora_dropout=configs.LORA_DROPOUT,
        save_every=int(os.getenv("STURNUS_SAVE_EVERY", "0")),
    )

    model = build_model(model_id, lora=True)
    train_loop(model, batch_iter, cfg)


if __name__ == "__main__":
    run()
