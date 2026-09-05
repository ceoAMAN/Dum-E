"""The three models and how to run them. No policy lives here.

Gate    — backbone FROZEN. Cluster geometry is formed from its hidden states and
          stamped with its weight hash, so the backbone may not move online
          (rule 15/16). Only the route head trains, by regression onto the
          grounded delta, so it has a gradient at k=1.
Central — LoRA'd. Trains ONLY in the pretrain phase (plain CE on q|y). In the
          joint phase it is the frozen instrument.
Experts — LoRA'd. Train by reward-weighted self-imitation on their own text.

ExpertPool has NO cap and NO eviction policy of its own: the scheduler is the
single owner of residency and tells the pool what to load and drop.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from . import config as C


# ── shared helpers ──────────────────────────────────────────────────────────
def _lora(model):
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model.freeze()
    n = len(model.layers) if hasattr(model, "layers") else len(model.model.layers)
    linear_to_lora_layers(model, n, {"rank": C.LORA_R, "scale": C.LORA_ALPHA / C.LORA_R, "dropout": C.LORA_DROPOUT})
    return model


def backbone(model, ids: mx.array) -> mx.array:
    """(T, D) hidden states from the transformer body, no lm_head."""
    h = model.model(ids.reshape(1, -1)) if hasattr(model, "model") else model(ids.reshape(1, -1))
    return h[0] if h.ndim == 3 else h


def ce_per_token(model, ids: mx.array, n_ctx: int) -> mx.array:
    """Teacher-forced CE over positions [n_ctx, T) only. On-graph."""
    logits = model(ids.reshape(1, -1))
    logits = logits[0] if logits.ndim == 3 else logits
    T, s = int(logits.shape[0]), max(1, int(n_ctx))
    if s >= T:
        return mx.zeros((0,), dtype=mx.float32)
    return nn.losses.cross_entropy(logits[s - 1:T - 1, :], ids[s:T], reduction="none")


def finite(x) -> bool:
    try:
        return bool(np.isfinite(np.asarray(x, dtype=np.float64)).all())
    except Exception:   # noqa: BLE001
        return False


def tree_finite(tree) -> bool:
    for _, v in tree_flatten(tree):
        if isinstance(v, mx.array) and not bool(mx.all(mx.isfinite(v)).item()):
            return False
    return True


def save_lora(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))


# ── memory (measured facts, not constants) ──────────────────────────────────
def active_mb() -> float:
    return float(mx.get_active_memory()) / 2**20


def peak_mb() -> float:
    return float(mx.get_peak_memory()) / 2**20


def reset_peak() -> None:
    mx.reset_peak_memory()


def total_ram_mb() -> float:
    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    return float(int(out)) / 2**20


def _sampler(temp: float):
    from mlx_lm.sample_utils import make_sampler
    return make_sampler(temp=temp)


# ── Gate ────────────────────────────────────────────────────────────────────
class Gate:
    def __init__(self):
        self.model = None
        self.tok = None
        self.route_head: Optional[nn.Linear] = None
        self._opt = None

    def load(self) -> "Gate":
        if self.model is not None:
            return self
        from mlx_lm import load
        self.model, self.tok = load(C.GATE_MODEL_ID)
        self.model.freeze()
        self.model.eval()
        self.route_head = nn.Linear(C.GATE_D, C.E)
        p = self._head_path()
        if p.exists():
            self.route_head.load_weights(str(p))
        mx.eval(self.route_head.parameters())
        return self

    def _head_path(self) -> Path:
        return Path(C.CHECKPOINT_DIR) / "gate" / "route_head.safetensors"

    def weight_hash(self) -> str:
        """Identity of the frozen backbone — the version geometry is stamped with."""
        flat = tree_flatten(self.model.parameters())
        h = hashlib.sha1(C.GATE_MODEL_ID.encode())
        for name, v in flat[:4]:
            h.update(name.encode())
            h.update(np.asarray(v.astype(mx.float32)).tobytes()[:4096])
        return h.hexdigest()[:12]

    def encode(self, text: str) -> List[int]:
        return list(self.tok.encode(text))

    def hidden(self, ids: List[int]) -> np.ndarray:
        """(T, D) float32 per-token hidden states. Frozen backbone, no grad."""
        h = backbone(self.model, mx.array(ids))
        mx.eval(h)
        return np.asarray(h.astype(mx.float32))

    @staticmethod
    def _z(pooled: np.ndarray) -> mx.array:
        """Raw Qwen hidden states have magnitudes ~50+; a fresh head on them emits
        logits in the tens. Z-score so the head sees ~N(0,1) and the regression
        target (deltas, ~+-2) is reachable."""
        x = np.asarray(pooled, dtype=np.float32)
        return mx.array((x - x.mean()) / (x.std() + 1e-6))

    def route_logits(self, pooled: np.ndarray) -> np.ndarray:
        out = self.route_head(self._z(pooled))
        mx.eval(out)
        return np.asarray(out)

    @property
    def opt(self):
        if self._opt is None:
            self._opt = optim.Adam(learning_rate=C.LR * 10)
        return self._opt

    def train_step(self, pooled: np.ndarray, targets: Dict[int, float]) -> float:
        """Regress route logits onto the grounded delta of the experts that ran.
        A regression, not a softmax: it has a gradient at k=1."""
        if not targets:
            return 0.0
        x = self._z(pooled)
        ids = mx.array(sorted(targets), dtype=mx.int32)
        tv = mx.array([float(targets[i]) for i in sorted(targets)], dtype=mx.float32)

        def loss_fn(head):
            return mx.mean((head(x)[ids] - tv) ** 2)

        loss, grads = nn.value_and_grad(self.route_head, loss_fn)(self.route_head)
        if not finite(loss.item()) or not tree_finite(grads):
            return float("nan")
        grads, _ = optim.clip_grad_norm(grads, C.GRAD_CLIP)
        self.opt.update(self.route_head, grads)
        mx.eval(self.route_head.parameters(), self.opt.state)
        return float(loss.item())

    def save(self) -> None:
        p = self._head_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(p), dict(tree_flatten(self.route_head.parameters())))


# ── Central ─────────────────────────────────────────────────────────────────
class Central:
    def __init__(self):
        self.model = None
        self.tok = None
        self._limit: Optional[int] = None
        self._opt = None

    def load(self) -> "Central":
        if self.model is not None:
            return self
        from mlx_lm import load
        self.model, self.tok = load(C.CENTRAL_MODEL_ID)
        _lora(self.model)
        p = self._ckpt()
        legacy = Path(C.LEGACY_CENTRAL_CKPT)
        if p.exists():
            self.model.load_weights(str(p), strict=False)
        elif legacy.exists():
            self.model.load_weights(str(legacy), strict=False)
            print(f"[central] inherited grounded-CE checkpoint {legacy}")
        self.model.eval()
        return self

    def _ckpt(self) -> Path:
        return Path(C.CHECKPOINT_DIR) / "central" / "weights.safetensors"

    def limit(self) -> int:
        if self._limit is None:
            args = getattr(self.model, "args", None)
            v = None
            for a in ("max_position_embeddings", "max_seq_len", "context_length"):
                x = getattr(args, a, None)
                if isinstance(x, int) and x > 0:
                    v = x
                    break
            self._limit = int(v or 4096)
        return self._limit

    def encode(self, text: str) -> List[int]:
        return list(self.tok.encode(text))

    def target_ids(self, answer: str) -> List[int]:
        """y, truncated ONCE. Every pass scores exactly these tokens."""
        m = min(C.TARGET_MAX_TOKENS, max(1, self.limit() // 4))
        return self.encode(answer)[:m]

    def context_ids(self, question: str, expert_texts: List[str], n_target: int) -> List[int]:
        """[question | \\n expert_1 | \\n expert_2 ...] capped so the whole of y
        still fits. Identical construction for the baseline and every expert
        pass, so the passes differ ONLY by the expert text."""
        cap = max(1, self.limit() - n_target)
        ids = self.encode(question)
        nl = self.encode("\n")
        for t in expert_texts:
            if not t:
                continue
            ids = ids + nl + self.encode(t)
        return ids[:cap]

    def ce_vector(self, ctx: List[int], y: List[int]) -> np.ndarray:
        """Per-token CE of y given ctx. Evaluated, (M,) float32."""
        ids = mx.array(ctx + y)
        v = ce_per_token(self.model, ids, len(ctx))
        mx.eval(v)
        return np.asarray(v.astype(mx.float32))

    @property
    def opt(self):
        if self._opt is None:
            self._opt = optim.Adam(learning_rate=C.LR)
        return self._opt

    def pretrain_step(self, question: str, answer: str) -> Tuple[float, int]:
        """Plain next-token CE on the real answer. The ONLY time Central trains."""
        y = self.target_ids(answer)
        ctx = self.context_ids(question, [], len(y))
        ids = mx.array(ctx + y)
        self.model.train()

        def loss_fn(m):
            return mx.mean(ce_per_token(m, ids, len(ctx)))

        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
        self.model.eval()
        if not finite(loss.item()) or not tree_finite(grads):
            return float("nan"), 0
        grads, _ = optim.clip_grad_norm(grads, C.GRAD_CLIP)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state)
        return float(loss.item()), len(y)

    def generate(self, question: str, notes: List[str], max_tokens: int = 256) -> str:
        from mlx_lm import generate
        content = question
        clean = [n.strip() for n in notes if n and n.strip()]
        if clean:
            content += "\n\nExpert analyses to consider:\n" + "\n".join(f"- {n}" for n in clean)
            content += "\n\nUse the analyses where they help and give the best final answer."
        tmpl = getattr(self.tok, "apply_chat_template", None)
        prompt = (tmpl([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                  if tmpl and getattr(self.tok, "chat_template", None) else content)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens)

    def save(self) -> None:
        save_lora(self.model, self._ckpt())


# ── Experts ─────────────────────────────────────────────────────────────────
class ExpertPool:
    def __init__(self):
        self.resident: Dict[int, object] = {}
        self.tok = None
        self._opts: Dict[int, object] = {}
        self.last_used: Dict[int, float] = {}

    def _ckpt(self, eid: int) -> Path:
        return Path(C.CHECKPOINT_DIR) / f"expert_{eid:03d}" / "weights.safetensors"

    def load(self, eid: int) -> None:
        if eid in self.resident:
            self.last_used[eid] = time.time()
            return
        from mlx_lm import load
        model, tok = load(C.EXPERT_MODEL_ID)
        _lora(model)
        p = self._ckpt(eid)
        if p.exists():
            model.load_weights(str(p), strict=False)
        model.eval()
        self.resident[eid] = model
        self.tok = self.tok or tok
        self.last_used[eid] = time.time()

    def unload(self, eid: int) -> None:
        self.resident.pop(eid, None)
        self._opts.pop(eid, None)
        self.last_used.pop(eid, None)
        mx.clear_cache()

    def opt(self, eid: int):
        if eid not in self._opts:
            self._opts[eid] = optim.Adam(learning_rate=C.LR)
        return self._opts[eid]

    def prompt(self, span_text: str, question: str) -> str:
        system = ("You are a domain specialist. Analyse the excerpt and give the single key "
                  "insight another model should use to answer. Be concise. Do not answer as if "
                  "you were the user, and do not invent facts that are not present.")
        user = f"Full question under consideration:\n{question}\n\nExcerpt assigned to you:\n{span_text}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        tmpl = getattr(self.tok, "apply_chat_template", None)
        if tmpl and getattr(self.tok, "chat_template", None):
            return tmpl(msgs, tokenize=False, add_generation_prompt=True)
        return f"{system}\n\n{user}\n"

    def _gen(self, eid: int, prompt: str, temp: float) -> str:
        from mlx_lm import generate
        kw = {"sampler": _sampler(temp)} if temp > 0 else {}
        return generate(self.resident[eid], self.tok, prompt=prompt, max_tokens=C.EXPERT_GEN_TOKENS, **kw).strip()

    def run(self, eid: int, span_text: str, question: str) -> Tuple[str, float]:
        """Greedy analysis of the span. Returns (text, wall_seconds)."""
        t0 = time.perf_counter()
        text = self._gen(eid, self.prompt(span_text, question), 0.0)
        return text, time.perf_counter() - t0

    def sample(self, eid: int, span_text: str, question: str) -> str:
        return self._gen(eid, self.prompt(span_text, question), C.SAMPLE_TEMP)

    def update(self, eid: int, prompt: str, texts: List[str], advantages: List[float]) -> float:
        """Reward-weighted self-imitation: loss = sum_g A_g * CE(e_g | prompt).
        A>0 pulls toward the text, A<0 pushes away. No ratios, no KL, no reference."""
        model = self.resident[eid]
        p_ids = self.tok.encode(prompt)
        seqs = []
        for t in texts:
            t_ids = self.tok.encode(t) if t else []
            if not t_ids:
                return 0.0
            seqs.append(mx.array(list(p_ids) + list(t_ids)))
        n_ctx = len(p_ids)
        model.train()

        def loss_fn(m):
            total = mx.array(0.0, dtype=mx.float32)
            for a, ids in zip(advantages, seqs):
                total = total + float(a) * mx.mean(ce_per_token(m, ids, n_ctx))
            return total

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        model.eval()
        if not finite(loss.item()) or not tree_finite(grads):
            return float("nan")
        grads, _ = optim.clip_grad_norm(grads, C.GRAD_CLIP)
        self.opt(eid).update(model, grads)
        mx.eval(model.parameters(), self.opt(eid).state)
        return float(loss.item())

    def save(self, eid: int) -> None:
        if eid in self.resident:
            save_lora(self.resident[eid], self._ckpt(eid))


def measure_expert_peak_mb() -> float:
    """Peak MB of one expert = weights + one forward at SPAN-scale. Measured, not
    configured: the old constant (850) was weights-only and the OOM forensics put
    real peaks at 2.8-4.8 GB."""
    from mlx_lm import load
    mx.clear_cache()
    reset_peak()
    before = peak_mb()
    model, _ = load(C.EXPERT_MODEL_ID)
    mx.eval(model.parameters())
    probe = mx.zeros((1, 256), dtype=mx.int32)
    out = model.model(probe) if hasattr(model, "model") else model(probe)
    mx.eval(out)
    m = peak_mb() - before
    del out, probe, model
    mx.clear_cache()
    return max(1.0, m)
