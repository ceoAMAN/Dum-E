# pyre-unsafe
"""Phase 3 training (Expert fine-tuning with 3 signals)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
import data
from scripts.train_common import build_model, resolve_model_id


def _cross_expert_alignment_loss(hidden_batch: torch.Tensor) -> torch.Tensor:
    if hidden_batch.shape[0] < 2:
        return torch.tensor(0.0, device=hidden_batch.device)

    normed = F.normalize(hidden_batch, p=2, dim=-1)
    sim_matrix = torch.mm(normed, normed.t())
    mask = 1.0 - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
    sim_matrix = sim_matrix * mask
    penalty = sim_matrix.sum() / (mask.sum() + 1e-9)  # pyre-ignore[16]
    return penalty


def run() -> None:
    print("[train_phase3] Expert fine-tuning (3 losses)")
    model_id = resolve_model_id(configs.EXPERT_TRAIN_MODEL_ID)
    batch_size = int(os.getenv("STURNUS_TRAIN_BATCH_SIZE", "2"))
    max_steps = int(os.getenv("STURNUS_TRAIN_STEPS", "50"))
    grad_accum = int(os.getenv("STURNUS_TRAIN_ACCUM", "4"))
    device = os.getenv("STURNUS_TRAIN_DEVICE", "cpu")

    distill_weight = float(os.getenv("STURNUS_DISTILL_WEIGHT", "0.3"))
    alignment_weight = float(os.getenv("STURNUS_ALIGN_WEIGHT", "0.1"))

    _group_limit_env = os.getenv("STURNUS_EXPERT_GROUP_LIMIT")
    group_limit: Optional[int] = int(_group_limit_env) if _group_limit_env else None

    for idx, group_name in enumerate(configs.EXPERT_GROUPS.keys()):
        if group_limit is not None and idx >= group_limit:
            break
        print(f"[train_phase3] training group {group_name}")
        batch_iter = data.iter_group_token_batches(
            group_name=group_name,
            model_id=model_id,
            batch_size=batch_size,
            max_length=configs.MAX_SEQ_LEN,
        )

        def _to_torch(batch):  # pyre-ignore[3]
            return {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in batch.items()}

        batch_iter = (_to_torch(b) for b in batch_iter)

        model = build_model(model_id, lora=True)
        model.to(device)  # pyre-ignore[16]
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=configs.LEARNING_RATE)
        out_dir = configs.CHECKPOINT_DIR / "experts" / group_name
        out_dir.mkdir(parents=True, exist_ok=True)

        step = 0
        optimizer.zero_grad(set_to_none=True)
        while step < max_steps:
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device)  # pyre-ignore[16]
            attention_mask = batch["attention_mask"].to(device)  # pyre-ignore[16]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True,
            )

            task_loss = outputs.loss

            logits = outputs.logits
            temperature = 2.0
            soft_logits = logits / temperature
            soft_targets = F.softmax(soft_logits.detach(), dim=-1)
            log_probs = F.log_softmax(soft_logits, dim=-1)
            distill_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)

            hidden = outputs.hidden_states[-1]
            last_hidden = hidden[:, -1, :]
            alignment_loss = _cross_expert_alignment_loss(last_hidden)

            loss = (task_loss + distill_weight * distill_loss + alignment_weight * alignment_loss) / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print(
                    f"[train_phase3] {group_name} step={step} total={loss.item():.4f} "
                    f"task={task_loss.item():.4f} distill={distill_loss.item():.4f} "
                    f"align={alignment_loss.item():.4f}"
                )

            step += 1

        model.save_pretrained(out_dir)
        print(f"[train_phase3] {group_name} saved to {out_dir}")


if __name__ == "__main__":
    run()
