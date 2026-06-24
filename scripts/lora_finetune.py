#!/usr/bin/env python
"""Fast batch LoRA fine-tuning for the Sturnus Central model.

This is the *fast* path — pure batched LoRA on the central responder, separate
from the slow ~50 tok/s online MoE marathon (`finetune.py`). It uses the exact
same LoRA setup `central.load()` uses (rank/alpha from configs), so the adapter
it writes loads straight back into the live runtime.

  python scripts/lora_finetune.py --steps 400 --batch-size 4 --max-len 512
  python scripts/lora_finetune.py --smoke           # 2 steps, proves it trains

Data: data/custom_prompts.jsonl  ({"text": "..."} per line).
Output: state/checkpoints/central/weights.safetensors (overwrites; backup kept).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load as mlx_load
from mlx_lm.tuner.utils import linear_to_lora_layers

import configs

DATA_PATH = Path("data/custom_prompts.jsonl")
CENTRAL_ADAPTER = Path(configs.CHECKPOINT_DIR) / "central" / "weights.safetensors"


def iter_texts():
    """Infinite shuffle-light generator over the local prompt corpus."""
    while True:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("prompt") or ""
                if text and len(text) > 20:
                    yield text


def iter_mixture_texts():
    """Infinite generator over the full 24-dataset streaming mixture (incl. the
    action/tool-calling datasets). This is what lets Central — the model that
    actually answers in voice deployment (timeline A) — learn the Jarvis action
    layer, not just the local custom prompts."""
    from data import authenticate_huggingface, iter_mixture_samples
    authenticate_huggingface()
    for sample in iter_mixture_samples():
        text = sample.text
        if text and len(text.strip()) > 20:
            yield text


def make_batch(texts, tokenizer, max_len):
    """Tokenize + right-pad a list of texts into (inputs, targets, mask)."""
    seqs = []
    for t in texts:
        ids = tokenizer.encode(t)[:max_len + 1]
        if len(ids) >= 2:
            seqs.append(ids)
    if not seqs:
        return None
    width = max(len(s) for s in seqs)
    pad = tokenizer.eos_token_id or 0
    inputs, targets, masks = [], [], []
    for s in seqs:
        # next-token prediction: predict s[1:] from s[:-1]
        inp = s[:-1]
        tgt = s[1:]
        n = len(inp)
        pad_n = (width - 1) - n
        inputs.append(inp + [pad] * pad_n)
        targets.append(tgt + [pad] * pad_n)
        masks.append([1.0] * n + [0.0] * pad_n)
    return (
        mx.array(inputs), mx.array(targets), mx.array(masks),
    )


def loss_fn(model, inputs, targets, mask):
    logits = model(inputs).astype(mx.float32)              # [B, T, V]
    ce = nn.losses.cross_entropy(logits, targets)          # [B, T]
    denom = mask.sum()
    return (ce * mask).sum() / mx.maximum(denom, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Fast batch LoRA fine-tune for Central.")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=configs.MAX_SEQ_LEN)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--data", choices=["custom", "mixture"], default="custom",
                    help="custom = local custom_prompts.jsonl only; "
                         "mixture = full 24-dataset stream incl. action/tool-calling.")
    ap.add_argument("--smoke", action="store_true", help="2-step smoke test, no overwrite of real adapter.")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 2
        args.save_every = 10_000  # don't save during smoke

    if args.data == "custom" and not DATA_PATH.exists():
        print(f"[lora] training data not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[lora] loading {configs.CENTRAL_MODEL_ID} ...")
    model, tokenizer = mlx_load(configs.CENTRAL_MODEL_ID)
    model.freeze()
    lora_config = {"rank": configs.LORA_R, "scale": configs.LORA_ALPHA, "dropout": configs.LORA_DROPOUT}
    num_layers = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    linear_to_lora_layers(model, num_layers, lora_config)

    # Continue from the existing adapter if present.
    if CENTRAL_ADAPTER.exists() and not args.smoke:
        try:
            model.load_weights(str(CENTRAL_ADAPTER), strict=False)
            print(f"[lora] resumed from {CENTRAL_ADAPTER}")
        except Exception as e:
            print(f"[lora] could not resume ({e}); training fresh LoRA")

    model.train()
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[lora] trainable LoRA params: {trainable:,}")

    opt = optim.AdamW(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    gen = iter_mixture_texts() if args.data == "mixture" else iter_texts()
    print(f"[lora] data source: {args.data}")

    t0 = time.time()
    tokens_seen = 0
    running = 0.0
    for step in range(1, args.steps + 1):
        batch_texts = [next(gen) for _ in range(args.batch_size)]
        batch = make_batch(batch_texts, tokenizer, args.max_len)
        if batch is None:
            continue
        inputs, targets, mask = batch
        loss, grads = loss_and_grad(model, inputs, targets, mask)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        tokens_seen += int(mask.sum().item())
        running += float(loss.item())
        if step % 10 == 0 or step == args.steps or args.smoke:
            tps = tokens_seen / max(1e-6, time.time() - t0)
            print(f"[lora] step {step}/{args.steps} | loss {running/min(step,10):.4f} "
                  f"| {tps:.0f} tok/s | {tokens_seen:,} tok")
            running = 0.0
        if step % args.save_every == 0 and not args.smoke:
            _save(model)

    if not args.smoke:
        _save(model)
    print(f"[lora] done in {time.time()-t0:.1f}s ({tokens_seen:,} tokens)")


def _save(model):
    CENTRAL_ADAPTER.parent.mkdir(parents=True, exist_ok=True)
    if CENTRAL_ADAPTER.exists():
        shutil.copy2(CENTRAL_ADAPTER, str(CENTRAL_ADAPTER) + ".bak")
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(CENTRAL_ADAPTER), flat)
    print(f"[lora] saved adapter -> {CENTRAL_ADAPTER} ({len(flat)} tensors)")


if __name__ == "__main__":
    main()
