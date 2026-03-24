# pyre-unsafe
"""Phase 2 training (Gate fine-tuning with 4 losses)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
import data
from scripts.train_common import resolve_model_id


class GatePolicyModel(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.base = base_model
        self.expert_head = nn.Linear(hidden_size, configs.NUM_EXPERTS)
        self.k_head = nn.Linear(hidden_size, configs.K_MAX + 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        last = hidden[:, -1, :]
        expert_logits = self.expert_head(last)
        k_logits = self.k_head(last)
        return expert_logits, k_logits, last


def _target_k(attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).float()
    ratios = lengths / float(configs.MAX_SEQ_LEN)
    ks = torch.clamp((ratios * configs.K_MAX).round(), configs.K_MIN, configs.K_MAX).long()
    return ks


def _routing_quality_loss(expert_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(expert_logits, dim=-1)
    top_vals, _ = torch.topk(probs, k=min(configs.K_DEFAULT, configs.NUM_EXPERTS), dim=-1)
    concentration = top_vals.sum(dim=-1)
    ideal = torch.ones_like(concentration) * 0.8
    return F.mse_loss(concentration, ideal)


def _adaptive_context_loss(hidden: torch.Tensor) -> torch.Tensor:
    norms = hidden.norm(dim=-1)
    return norms.std()


def run() -> None:
    print("[train_phase2] Gate fine-tuning (4 losses)")
    model_id = resolve_model_id(configs.GATE_TRAIN_MODEL_ID)
    batch_size = int(os.getenv("STURNUS_TRAIN_BATCH_SIZE", "2"))
    max_steps = int(os.getenv("STURNUS_TRAIN_STEPS", "100"))
    grad_accum = int(os.getenv("STURNUS_TRAIN_ACCUM", "4"))
    device = os.getenv("STURNUS_TRAIN_DEVICE", "cpu")

    lb_weight = float(os.getenv("STURNUS_LB_WEIGHT", "0.1"))
    rq_weight = float(os.getenv("STURNUS_RQ_WEIGHT", "0.05"))
    ac_weight = float(os.getenv("STURNUS_AC_WEIGHT", "0.02"))

    batch_iter = data.iter_mixture_token_batches(
        model_id=model_id,
        batch_size=batch_size,
        max_length=configs.MAX_SEQ_LEN,
        seed=123,
    )

    def _to_torch(batch: Dict[str, torch.Tensor]):
        return {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in batch.items()}

    batch_iter = (_to_torch(b) for b in batch_iter)

    base = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    lora_cfg = LoraConfig(
        r=configs.LORA_R,
        lora_alpha=configs.LORA_ALPHA,
        lora_dropout=configs.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_cfg)

    hidden_size = base.configs.hidden_size
    model = GatePolicyModel(base, hidden_size=hidden_size).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=configs.LEARNING_RATE)
    out_dir = configs.CHECKPOINT_DIR / "gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        batch = next(batch_iter)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        expert_logits, k_logits, hidden = model(input_ids, attention_mask)
        target_k = _target_k(attention_mask)

        k_loss = F.cross_entropy(k_logits, target_k)

        probs = F.softmax(expert_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        load_balance_loss = -entropy

        rq_loss = _routing_quality_loss(expert_logits)
        ac_loss = _adaptive_context_loss(hidden)

        loss = (k_loss + lb_weight * load_balance_loss + rq_weight * rq_loss + ac_weight * ac_loss) / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(
                f"[train_phase2] step={step} total={loss.item():.4f} "
                f"k={k_loss.item():.4f} lb={load_balance_loss.item():.4f} "
                f"rq={rq_loss.item():.4f} ac={ac_loss.item():.4f}"
            )

        step += 1

    model.base.save_pretrained(out_dir)
    torch.save(
        {"expert_head": model.expert_head.state_dict(), "k_head": model.k_head.state_dict()},
        out_dir / "heads.pt",
    )
    print(f"[train_phase2] saved to {out_dir}")


if __name__ == "__main__":
    run()
