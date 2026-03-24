# pyre-unsafe
"""Shared training utilities for Sturnus."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    device: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    save_every: int


def _device() -> str:
    return os.getenv("STURNUS_TRAIN_DEVICE", "cpu")


def resolve_model_id(default_id: str) -> str:
    override = os.getenv("STURNUS_TRAIN_MODEL_ID")
    if override:
        return override
    if os.getenv("STURNUS_TINY", "0") == "1":
        return "sshleifer/tiny-gpt2"
    return default_id


def build_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(model_id: str, lora: bool = True) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    if lora:
        lora_cfg = LoraConfig(
            r=configs.LORA_R,
            lora_alpha=configs.LORA_ALPHA,
            lora_dropout=configs.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    return model


def train_loop(
    model: torch.nn.Module,
    batch_iter: Iterator[Dict[str, torch.Tensor]],
    cfg: TrainConfig,
) -> None:
    device = torch.device(cfg.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < cfg.max_steps:
        batch = next(batch_iter)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss / cfg.grad_accum_steps
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print(f"[train] step={step} loss={loss.item():.4f}")

        if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            save_path = cfg.output_dir / f"step-{step}"
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)

        step += 1

    model.save_pretrained(cfg.output_dir)
