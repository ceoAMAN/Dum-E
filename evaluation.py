"""Held-out evaluation — a mixture-skew-immune training signal.

The `avg_loss` recorded in the marathon is the gate's domain-routing
cross-entropy on the *training* mixture. Because the mixture was dominated by
"general"-classified local_custom, that loss collapsed to ~0 and told us
nothing. This module evaluates the two things actually being trained, on a
fixed, balanced, held-out set (data/heldout_eval.jsonl):

  1. gate routing accuracy  — does the gate route an unseen prompt to the
     domain it belongs to? (the gate's real objective; gate-only, cheap)
  2. held-out expert quality — mean r_i (expert contribution) and mean expert
     MSE-to-synthesis on unseen prompts (the experts' real objective; heavy,
     needs experts + Central)

Both are immune to mixture skew because the eval set is balanced 10/10/10/10.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import numpy as np

import configs

DOMAINS = ["code", "reasoning", "knowledge", "general"]
DEFAULT_EVAL_PATH = "data/heldout_eval.jsonl"


def load_eval_set(path: str = DEFAULT_EVAL_PATH) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    samples = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "text" in row and "domain" in row:
            samples.append({"text": row["text"], "domain": row["domain"]})
    return samples


def gate_domain(gate, text: str) -> str:
    """Predicted domain (argmax of the gate's 4 domain logits) for one prompt."""
    gate.load()
    token_ids = gate.tokenizer.encode(text)[: configs.MAX_SEQ_LEN]
    if len(token_ids) < 1:
        return "general"
    gate_out = gate.forward(mx.array(token_ids))
    logits = gate_out.domain_logits
    if logits is None or logits.shape[0] < len(DOMAINS):
        return "general"
    return DOMAINS[int(mx.argmax(logits[: len(DOMAINS)]).item())]


def gate_routing_accuracy(gate, samples: List[Dict[str, str]]) -> Dict:
    """Gate-only, cheap. Fraction of held-out prompts routed to their true
    domain, plus per-domain accuracy and a confusion matrix."""
    if not samples:
        return {"accuracy": 0.0, "n": 0, "per_domain": {}, "confusion": {}}
    correct = 0
    per_domain_total: Dict[str, int] = {d: 0 for d in DOMAINS}
    per_domain_correct: Dict[str, int] = {d: 0 for d in DOMAINS}
    confusion: Dict[str, Dict[str, int]] = {d: {e: 0 for e in DOMAINS} for d in DOMAINS}
    for s in samples:
        true_d = s["domain"]
        pred_d = gate_domain(gate, s["text"])
        if true_d in per_domain_total:
            per_domain_total[true_d] += 1
            confusion[true_d][pred_d] = confusion[true_d].get(pred_d, 0) + 1
        if pred_d == true_d:
            correct += 1
            if true_d in per_domain_correct:
                per_domain_correct[true_d] += 1
    per_domain = {
        d: (per_domain_correct[d] / per_domain_total[d] if per_domain_total[d] else 0.0)
        for d in DOMAINS
    }
    return {
        "accuracy": correct / len(samples),
        "n": len(samples),
        "per_domain": per_domain,
        "confusion": confusion,
    }


def heldout_expert_quality(
    gate,
    expert_pool,
    central,
    triple_k,
    masking,
    session_tracker,
    samples: List[Dict[str, str]],
    k: int = 4,
) -> Dict:
    """Heavy. Mean r_i and mean expert MSE-to-synthesis on held-out prompts —
    the experts' actual training objective, measured on unseen data. Mirrors the
    Timeline-B forward in scripts/finetune.py without taking any gradient step."""
    if not samples:
        return {"mean_r_i": 0.0, "mean_expert_mse": 0.0, "n": 0}
    gate.load()
    central.load()
    r_i_all: List[float] = []
    mse_all: List[float] = []
    n_used = 0
    for s in samples:
        text = s["text"]
        token_ids = gate.tokenizer.encode(text)[: configs.MAX_SEQ_LEN]
        if len(token_ids) < configs.FRAGMENT_MIN:
            continue
        tokens = mx.array(token_ids)
        n_tokens = len(token_ids)
        gate_out = gate.forward(tokens)
        selected = triple_k.select_experts(gate_out, session_tracker, masking, 0)[:k]
        if not selected:
            continue
        ids_to_load = [se.expert_id for se in selected if se.expert_id not in expert_pool.loaded_experts]
        try:
            if ids_to_load:
                expert_pool.load_experts(ids_to_load)
        except RuntimeError:
            continue
        selected = [se for se in selected if se.expert_id in expert_pool.loaded_experts]
        if not selected:
            continue
        frag_size = max(configs.FRAGMENT_MIN, n_tokens // max(len(selected), 1))
        expert_outputs = []
        for i, sel in enumerate(selected):
            fs = i * frag_size
            fe = min(fs + frag_size, n_tokens)
            if fs >= n_tokens:
                break
            frag = tokens[fs:fe]
            if frag.shape[0] < configs.FRAGMENT_MIN:
                continue
            expert_outputs.append(expert_pool.expert_forward(sel.expert_id, frag))
        if not expert_outputs:
            continue
        expert_data = [
            {"expert_id": eo.expert_id, "output_text": eo.output_text,
             "hidden_states": eo.hidden_states, "wall_time": eo.wall_time}
            for eo in expert_outputs
        ]
        central_out = central.forward(text, expert_data, send_to_user=False)
        for eo in expert_outputs:
            r_i = central.compute_r_i(eo.hidden_states, central_out.contribution_hidden, eo.wall_time, synthesis_hidden=central_out.synthesis_hidden)
            r_i_all.append(r_i)
            # expert MSE-to-synthesis (the apply_expert_gradients objective)
            if eo.hidden_states is not None and central_out.synthesis_hidden is not None:
                eh = eo.hidden_states.reshape(-1)
                sh = central_out.synthesis_hidden.reshape(-1)
                md = min(int(eh.shape[0]), int(sh.shape[0]))
                if md > 0:
                    mse = mx.mean((eh[:md] - sh[:md]) ** 2)
                    mx.eval(mse)
                    mse_all.append(float(mse.item()))
        n_used += 1
    return {
        "mean_r_i": float(np.mean(r_i_all)) if r_i_all else 0.0,
        "mean_expert_mse": float(np.mean(mse_all)) if mse_all else 0.0,
        "n": n_used,
        "n_expert_activations": len(r_i_all),
    }
