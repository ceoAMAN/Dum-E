"""The three models and how to run them. No policy lives here.

Gate    — backbone FROZEN. Cluster geometry is formed from its hidden states and
          stamped with its weight hash, so the backbone may not move online
          (rule 15/16). Only the route head trains, by regression onto the
          grounded delta, so it has a gradient at k=1.
Central — LoRA'd. Plain CE on q|y, in pretrain AND in the training loop
          (last in each batch, after the experts are graded, so it is the
          frozen instrument WHILE a batch is scored and never differentiated
          through). Never on held-out or self-referent samples.
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


def _head(model, h: mx.array) -> mx.array:
    """Project hidden states through the LM head, tied or not."""
    lm = getattr(model, "lm_head", None)
    return lm(h) if lm is not None else model.model.embed_tokens.as_linear(h)


def ce_per_token(model, ids: mx.array, n_ctx: int) -> mx.array:
    """Teacher-forced CE over positions [n_ctx, T) only. On-graph.

    The head is applied to the SCORED TAIL ONLY, never the whole sequence.
    model(ids) materialises T x 151,936 floats: on a 2,400-token ultrachat row
    that is 1.5 GB per call, and score() makes k+1 of them per batch with
    Central and four experts already resident — 7.3 GB of a 12.1 GB Metal
    working set. That is what OOM-killed the trainer (SIGKILL, rc 137).
    Only M <= TARGET_MAX_TOKENS positions are ever read, so slicing the hidden
    states before the projection costs nothing and bounds the transient by M
    instead of T. Verified bit-equal against the full projection."""
    h = model.model(ids.reshape(1, -1))
    h = h[0] if h.ndim == 3 else h
    T, s = int(h.shape[0]), max(1, int(n_ctx))
    if s >= T:
        return mx.zeros((0,), dtype=mx.float32)
    logits = _head(model, h[s - 1:T - 1, :])
    return nn.losses.cross_entropy(logits, ids[s:T], reduction="none")


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


def save_lora(model, path: Path, version: str = "") -> None:
    """Write to .tmp then rename: a kill mid-save must not leave a truncated
    safetensors that crashes every later boot. The version travels WITH the
    weights so a stale checkpoint is refused, not silently inherited."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # mx.save_safetensors silently APPENDS .safetensors to any other extension,
    # so the tmp name must already end in it or the rename below finds nothing.
    tmp = path.with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), dict(tree_flatten(model.trainable_parameters())),
                        metadata={"dume_version": str(version)})
    tmp.replace(path)


def load_lora(model, path: Path, version: str = "") -> bool:
    """Load only if the stamp matches. Returns False (leaving the model at base)
    on a mismatch or an unreadable file — never a silent half-load."""
    try:
        _, meta = mx.load(str(path), return_metadata=True)
        stamp = str(meta.get("dume_version", ""))
        if version and stamp != str(version):
            print(f"[ckpt] {path} was written by {stamp or 'an unstamped run'} != {version} — refusing it")
            return False
        model.load_weights(str(path), strict=False)
        return True
    except Exception as e:      # noqa: BLE001
        print(f"[ckpt] could not read {path}: {e} — starting from base")
        return False


# ── memory (measured facts, not constants) ──────────────────────────────────
def active_mb() -> float:
    return float(mx.get_active_memory()) / 2**20


def peak_mb() -> float:
    return float(mx.get_peak_memory()) / 2**20


def reset_peak() -> None:
    mx.reset_peak_memory()


_THERMAL = None          # NSProcessInfo, resolved once; None means unavailable


def thermal_state() -> int:
    """The device's own thermal pressure, 0..3 (nominal / fair / serious /
    critical), straight from NSProcessInfo.

    This is the DEVICE's side of the tug of war over k (Aman, 2026-09-20: "the
    device wants k less so it doesn't get hot, its input is temperature; the
    system wants to do work fastest so it wants k high"). It is a real input,
    not a proxy: macOS decides these levels from the sensors we cannot read
    ourselves, and it costs 1.6 us to poll, so it can be read every batch.

    A raw temperature would be better and is not available here: `pmset -g
    therm` has never recorded a warning on this machine, no thermal sysctl or
    SMC key is exposed, and `powermetrics --samplers smc` does not exist on this
    OS. Reading degrees needs sudo on every call, which cannot run unattended
    inside the loop.

    Returns 0 when the framework is missing, which is the honest default — an
    unmeasurable device is not a hot one, and a missing sensor must not quietly
    throttle the pool."""
    global _THERMAL
    if _THERMAL is None:
        try:
            import Foundation                            # pyobjc-framework-Cocoa
            _THERMAL = Foundation.NSProcessInfo.processInfo()
        except Exception:                                # noqa: BLE001
            _THERMAL = False
    if _THERMAL is False:
        return 0
    try:
        return max(0, min(3, int(_THERMAL.thermalState())))
    except Exception:                                    # noqa: BLE001
        return 0


def total_ram_mb() -> float:
    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    return float(int(out)) / 2**20


def working_set_mb() -> float:
    """What the GPU will actually let us hold — Metal refuses past this, so it,
    not total RAM, is the ceiling k is fitted under. Measured from the device."""
    try:
        return float(mx.device_info()["max_recommended_working_set_size"]) / 2**20
    except Exception:       # noqa: BLE001
        return 0.75 * total_ram_mb()


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
        if p.exists() and not load_lora(self.route_head, p, self.weight_hash()):
            self.route_head = nn.Linear(C.GATE_D, C.E)      # mismatch: start from init
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

    def summarise(self, ids: List[int], budget: int) -> str:
        """A bounded map of the WHOLE input, for experts that only ever see a slice.

        THE GATE WRITES IT, not Central and not an expert (Aman, 2026-09-21). It
        is the only component that reads all of T already — the routing geometry
        needs the full hidden states — so the prefill is a pass it was going to
        make anyway, and it is the smallest model in the stack. It is also FROZEN
        and decoded greedily, so the summary is a pure function of (ids, budget):
        the gate does not drift as the pool learns, which a summary written by
        Central would.

        THE BUDGET IS NOT A FUNCTION OF THE INPUT ALONE, so neither is the text.
        `budget` is the smallest span the law allocated, that span is T_eff/k, and
        k falls with thermal pressure by design (the tug of war: the device wants k
        low, the system wants it high). MEASURED at k_max=4, span_max=446: a T=100
        row gives k=4, L=25 and NO map when the device is nominal, and k=2, L=50
        and a full map at thermal level 1. The same row therefore gets categorically
        different prompts on a warm machine. That is a consequence of k being
        thermally regulated, not of anything here — but the earlier claim that this
        was stationary across epochs was wrong, and nothing downstream should be
        built on it.

        THE BUDGET IS APEX-NADIR'S, not a constant. `budget` is the SMALLEST span
        the law allocated this batch, so the map can never outweigh the material
        of even the most thinly-fed expert — the same invariant the span itself
        obeys. Below EXPERT_GEN_TOKENS there is no summary at all: that is the
        floor a note has to clear to say anything, and a two-token map costs a
        full prefill to deliver nothing.

        MEASURED on this machine, 0.5B-4bit gate, real prose (2026-09-21):

            T      prefill  decode  total   wrote
            128     0.13     0.12   0.25 s   28 tok
            320     0.24     0.26   0.50 s   27
            639        -        -   0.68 s   33
            1784    1.16     0.63   1.79 s   65

        sec ~ 0.18 + 0.0009*T. Over this run's real (T, k) distribution the floor
        gates out 79% of rows, so the map fires on 21%, costs 0.53 s when it fires
        and 0.113 s amortised — 1.0% of an 11.8 s batch. Weights 268 MB, peak 959 MB
        on the longest prefill, against the scheduler's 4096 MB sqrt reserve.

        Note the gate stops on EOS far below the budget: it wrote ~30 tokens against
        a median budget of 148. The law's allocation is therefore a GATE and a hard
        ceiling here, not a length control — the typical length is the model's.

        At large T the cost is prefill, and hidden() already prefills these same ids
        for the routing geometry. The chat template wraps them differently so the
        cache is not reusable as it stands; at 1% of a batch that is headroom worth
        knowing about and not worth the surgery.

        Returns "" when it cannot help, which every caller treats as "no context"
        rather than as a failure."""
        # TWO bounds, and both are somebody else's arithmetic. The law's smallest
        # span keeps the map from outweighing the material. TARGET_MAX_TOKENS keeps
        # it inside the reserve the scheduler ALREADY spent on it: span_max is
        # computed as free/slope - 2*TARGET_MAX_TOKENS on the stated basis that the
        # backpropped sequence is "span + orientation header + the expert's own
        # generated text" (scheduler.py:64). The old header was hard-clamped at
        # TARGET_MAX_TOKENS by construction; dropping that clamp without restoring
        # it here would let the header outgrow the memory already budgeted for it.
        budget = min(int(budget), C.TARGET_MAX_TOKENS)
        if budget < C.EXPERT_GEN_TOKENS or not ids:
            return ""
        from mlx_lm import generate
        body = self.tok.decode(list(ids))
        msgs = [{"role": "system", "content": "Summarise what the document is about in one short sentence. "
                                              "No preamble, no detail, no answer."},
                {"role": "user", "content": body}]
        tmpl = getattr(self.tok, "apply_chat_template", None)
        text = (tmpl(msgs, tokenize=False, add_generation_prompt=True)
                if tmpl and getattr(self.tok, "chat_template", None)
                else f"{msgs[0]['content']}\n\n{body}\n")
        try:
            return generate(self.model, self.tok, prompt=text, max_tokens=budget).strip()
        except Exception as e:                           # noqa: BLE001
            print(f"[gate] summary failed: {e} — experts run without context")
            return ""

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
        save_lora(self.route_head, self._head_path(), self.weight_hash())


# ── Central ─────────────────────────────────────────────────────────────────
class Central:
    def __init__(self):
        self.model = None
        self.tok = None
        self._limit: Optional[int] = None
        self._opt = None
        self.version: str = ""       # set by System at boot

    def load(self) -> "Central":
        if self.model is not None:
            return self
        from mlx_lm import load
        self.model, self.tok = load(C.CENTRAL_MODEL_ID)
        _lora(self.model)
        p = self._ckpt()
        legacy = Path(C.LEGACY_CENTRAL_CKPT)
        if p.exists():
            load_lora(self.model, p, self.version)
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
        nl = self.encode("\n")
        ids = self.encode(question)[:C.WORKING_PROBE_TOKENS] + nl
        for t in expert_texts:
            if not t:
                continue
            ids = ids + self.encode(t) + nl
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
        """Plain next-token CE on the real answer, at full weight. Used by both
        `pretrain` and the training loop: whenever a REAL y exists, Central learns
        from it unscaled — admitted or refused. The reliability score is a
        DEPLOYMENT deduction (see System.answer), never a training one. The loop
        calls this last in the batch, and never on a held-out or self-referent
        sample — see the comment there for why each is excluded."""
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

    def generate(self, question: str, notes: List[str], max_tokens: int = 256,
                 base: str = "", base_label: str = "Your own draft answer") -> str:
        """`base` is the output IN CHARGE of this merge; `notes` support it.

        Deployment runs three passes (Aman, 2026-09-20: "central for producing
        output does first it's own, then centroids synthesised output then uses
        both of it create one"), and this is the third. Which output leads is
        decided by reliability, not by rank: "the comparison is done against
        which is more reliability score — if pool then synthesised output, if
        central you know it". So `base` is Central's own draft when Central
        scores higher and the experts' synthesis when the pool does, and
        `base_label` says which, because a synthesis presented as Central's own
        draft is exactly the wrong prior.

        The final answer is always emitted here. The synthesis is made from the
        expert parts and is never itself the answer."""
        from mlx_lm import generate
        content = question
        clean = [n.strip() for n in notes if n and n.strip()]
        base = base.strip()
        if base:
            content += f"\n\n{base_label}:\n" + base
        if clean:
            content += "\n\nAlso consider:\n" + "\n".join(f"- {n}" for n in clean)
        if clean and base:
            content += ("\n\nWeigh them against each other and give the best final "
                        "answer, starting from the first.")
        elif clean:
            content += "\n\nUse the analyses where they help and give the best final answer."
        elif base:
            content += "\n\nGive the best final answer."
        tmpl = getattr(self.tok, "apply_chat_template", None)
        prompt = (tmpl([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                  if tmpl and getattr(self.tok, "chat_template", None) else content)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens)

    def save(self) -> None:
        save_lora(self.model, self._ckpt(), self.version)


# ── Experts ─────────────────────────────────────────────────────────────────
class ExpertPool:
    """One base, many adapters.

    Every expert is the SAME frozen 1.5B base plus a 35 MB LoRA. Loading them as
    100 independent models meant `mlx_lm.load()` per expert — 0.56-0.76 s and a
    full 1.2 GB base copy each — so k_max=4 held four identical bases (4.8 GB) to
    serve 140 MB of actual difference, and that redundancy is what set span_max
    to 173 tokens. Swapping the adapter in place instead is 0.0035 s (170x) and
    costs 35 MB.

    Experts run SEQUENTIALLY on this machine, so exactly one adapter is live in
    the base at a time (`self.active`). `resident` is the set whose weights are
    held in RAM and can be made live for free; `_activate` moves one in, stashing
    whatever was live first — an expert's gradient step is in the base's tensors
    until it is stashed, so stashing before overwriting is not an optimisation,
    it is the difference between training and losing the step.
    """

    def __init__(self):
        self.base = None                       # the one shared, LoRA-wrapped model
        self.pristine: Dict[str, mx.array] = {}   # a fresh adapter, for experts with no checkpoint
        self.resident: Dict[int, Dict[str, mx.array]] = {}   # eid -> its adapter tensors
        self.active: Optional[int] = None      # the eid whose weights are in `base` right now
        self.tok = None
        self._opts: Dict[int, object] = {}
        self.last_used: Dict[int, float] = {}
        self.dirty: set = set()      # updated since last save; unload() must flush these

    version: str = ""            # set by System at boot; stamped into every checkpoint

    def _ckpt(self, eid: int) -> Path:
        return Path(C.CHECKPOINT_DIR) / f"expert_{eid:03d}" / "weights.safetensors"

    # ── the shared base ─────────────────────────────────────────────────────
    def _boot(self) -> None:
        if self.base is not None:
            return
        from mlx_lm import load
        model, tok = load(C.EXPERT_MODEL_ID)
        _lora(model)
        mx.eval(model.parameters())
        self.base, self.tok = model, self.tok or tok
        self.pristine = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}

    def _fresh(self) -> Dict[str, mx.array]:
        """A new adapter, initialised the way mlx does it: lora_b zero, lora_a
        uniform(-1/sqrt(fan_in)). Re-drawn per expert — one shared draw would
        give every expert the same starting direction."""
        out = {}
        for k, v in self.pristine.items():
            if k.endswith("lora_a"):
                sc = 1.0 / math.sqrt(v.shape[0])
                out[k] = mx.random.uniform(low=-sc, high=sc, shape=v.shape, dtype=v.dtype)
            else:
                out[k] = mx.zeros(v.shape, dtype=v.dtype)
        return out

    def _stash(self) -> None:
        """Pull the live adapter out of the base and back into `resident`."""
        if self.active is None or self.active not in self.resident:
            return
        self.resident[self.active] = {k: v for k, v in tree_flatten(self.base.trainable_parameters())}

    def _activate(self, eid: int):
        """Make this expert's weights the live ones and return the model."""
        self.load(eid)
        if self.active != eid:
            self._stash()
            self.base.load_weights(list(self.resident[eid].items()), strict=False)
            mx.eval(self.base.parameters())
            self.active = eid
        self.last_used[eid] = time.time()
        return self.base

    # ── residency ───────────────────────────────────────────────────────────
    def load(self, eid: int) -> None:
        self._boot()
        if eid in self.resident:
            self.last_used[eid] = time.time()
            return
        adapter = self._fresh()
        p = self._ckpt(eid)
        if p.exists():
            try:
                w, meta = mx.load(str(p), return_metadata=True)
                stamp = str(meta.get("dume_version", ""))
                if self.version and stamp != str(self.version):
                    print(f"[ckpt] {p} was written by {stamp or 'an unstamped run'} != {self.version} — refusing it")
                else:
                    adapter.update({k: v for k, v in w.items() if k in adapter})
            except Exception as e:      # noqa: BLE001
                print(f"[ckpt] could not read {p}: {e} — starting from base")
        self.resident[eid] = adapter
        self.last_used[eid] = time.time()

    def unload(self, eid: int) -> None:
        """Eviction must never lose training. An expert updated since its last
        save is flushed BEFORE it is dropped — otherwise, with k_max seats and a
        fresh trial every batch, most updates die within a few batches while
        expert_loss still prints."""
        if eid in self.dirty:
            self.save(eid)
        if self.active == eid:
            self._stash()
            self.active = None
        self.resident.pop(eid, None)
        self._opts.pop(eid, None)
        self.last_used.pop(eid, None)
        mx.clear_cache()

    def opt(self, eid: int):
        if eid not in self._opts:
            self._opts[eid] = optim.Adam(learning_rate=C.LR)
        return self._opts[eid]

    def prompt(self, span_text: str, context: str = "") -> str:
        system = ("You are a domain specialist. Analyse the excerpt and give the single key "
                  "insight another model should use to answer. Be concise. Do not answer as if "
                  "you were the user, and do not invent facts that are not present.")
        # THE EXPERT SEES ITS FRAGMENT AND NOTHING ELSE (Aman, 2026-09-21).
        #
        # The input used to ride along with every span — first in full, then bounded
        # to TARGET_MAX_TOKENS. Both made apex-nadir's allocation a label rather than
        # a budget: if every expert reads all of T, the pool costs k*T and dividing
        # the input buys nothing. 100 small experts are only cheaper than one big
        # model when the work is ACTUALLY divided.
        #
        # Measured on the live pool, T has median 51 and p90 639, so the 128-token
        # bound meant the whole input for four rows in five and a truncated head for
        # the rest: two different contracts decided by row length, which is why
        # 84-93% of the k prompts were identical and clone_frac sat at 0.30 through
        # 599 expert updates while the adapters themselves diverged (pairwise cosine
        # +0.0000 — the updates landed, the prompts gave them nothing to diverge ON).
        #
        # So the expert is extractive: it condenses the region it was given. Central
        # holds the input and assembles the notes. Prompt cost is now T across the
        # pool however large k grows, and the sequence the backward pass runs over is
        # bounded by the span, not by the input.
        user = f"{context}\n\nExcerpt:\n{span_text}" if context else f"Excerpt:\n{span_text}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        tmpl = getattr(self.tok, "apply_chat_template", None)
        if tmpl and getattr(self.tok, "chat_template", None):
            return tmpl(msgs, tokenize=False, add_generation_prompt=True)
        return f"{system}\n\n{user}\n"

    def _gen(self, eid: int, prompt: str, temp: float, budget: Optional[int] = None) -> str:
        from mlx_lm import generate
        kw = {"sampler": _sampler(temp)} if temp > 0 else {}
        # EXPERT_GEN_TOKENS is the FLOOR, not the count: apex-nadir allocates the
        # budget and an expert never writes less than the floor. This is also what
        # makes the cost curve c(t) = a + bt real — with a fixed 32 the wall time
        # would be constant in t, b would fit to ~0 and cost would stop mattering.
        n = max(C.EXPERT_GEN_TOKENS, int(budget)) if budget else C.EXPERT_GEN_TOKENS
        return generate(self._activate(eid), self.tok, prompt=prompt, max_tokens=n, **kw).strip()

    def run(self, eid: int, span_text: str, context: str = "",
            budget: Optional[int] = None) -> Tuple[str, float]:
        """Greedy analysis of the span. Returns (text, wall_seconds)."""
        t0 = time.perf_counter()
        text = self._gen(eid, self.prompt(span_text, context), 0.0, budget)
        return text, time.perf_counter() - t0

    def sample(self, eid: int, span_text: str, context: str = "",
               budget: Optional[int] = None) -> str:
        """The exploring candidate for self-imitation, at SAMPLE_TEMP."""
        return self._gen(eid, self.prompt(span_text, context), C.SAMPLE_TEMP, budget)

    def update(self, eid: int, prompt: str, texts: List[str], advantages: List[float]) -> Optional[float]:
        """Reward-weighted self-imitation: loss = sum_g A_g * CE(e_g | prompt).
        A>0 pulls toward the text, A<0 pushes away. No ratios, no KL, no reference."""
        model = self._activate(eid)
        p_ids = self.tok.encode(prompt)
        seqs = []
        for t in texts:
            t_ids = self.tok.encode(t) if t else []
            if not t_ids:
                return None
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
        self.dirty.add(eid)
        return float(loss.item())

    def save(self, eid: int) -> None:
        if eid not in self.resident:
            return
        if self.active == eid:
            self._stash()
        p = self._ckpt(eid)
        p.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(p), dict(self.resident[eid]), metadata={"dume_version": str(self.version)})
        self.dirty.discard(eid)

    def save_all(self) -> None:
        for e in list(self.dirty):
            self.save(e)


def measure_update_slope_mb(pool, eid: int) -> float:
    """MB of peak memory per PROMPT TOKEN in an expert's backward pass.

    Measured, not configured. The gradient step stores activations for every
    layer over the whole sequence, so its cost is linear in prompt length and
    it is by far the largest transient in the system: ~12 MB/token on the M4,
    i.e. 13 GB at 1,024 tokens against a 12.1 GB Metal working set. That is
    what SIGKILLed the trainer. Two SMALL probes (96 and 192 tokens, ~1.9 and
    ~3.0 GB) bracket the line; no weights are written — value_and_grad alone,
    no optimiser step."""
    import mlx.nn as nn
    model = pool._activate(eid)
    ids = pool.tok.encode("token " * 200)
    out = []
    for n in (96, 192):
        # TWO sequences, because that is what update() actually backprops: the
        # greedy and the sampled candidate are summed in one loss.
        seqs = [mx.array(list(ids[:n]) + list(pool.tok.encode(t)))
                for t in (" a short note.", " a different short note.")]
        mx.clear_cache()
        reset_peak()
        model.train()
        loss, grads = nn.value_and_grad(
            model, lambda m: sum(mx.mean(ce_per_token(m, q, n)) for q in seqs))(model)
        mx.eval(loss, grads)          # MLX is lazy: without this nothing is computed and the
        model.eval()                  # two probes read the same peak, giving slope 0
        out.append(peak_mb())
        mx.clear_cache()
    slope = (out[1] - out[0]) / 96.0
    return float(max(slope, 0.5))          # a floor: a non-positive slope is a bad measurement


def measure_slot_mb(pool) -> float:
    """MB an ADDITIONAL resident expert costs, now that the base is shared: its
    adapter plus Adam's two moments over the same tensors. The base itself is
    paid once, not once per seat — charging every seat a full 1.2 GB copy is
    what pinned k_fit at 8 and span_max at 173."""
    pool._boot()
    mb = sum(v.size * v.dtype.size for v in pool.pristine.values()) / 2**20
    return float(mb * 3.0)          # weights + Adam m + Adam v


def measure_expert_peak_mb() -> float:
    """Peak MB of one expert = weights + one forward at SPAN-scale. Measured, not
    configured: the old constant (850) was weights-only and the OOM forensics put
    real peaks at 2.8-4.8 GB."""
    from mlx_lm import load
    mx.clear_cache()
    reset_peak()
    before = active_mb()          # NOT peak_mb(): reset_peak() just set the peak to 0, so
    model, _ = load(C.EXPERT_MODEL_ID)   # peak-minus-peak would charge the expert for the
    mx.eval(model.parameters())          # gate and Central that are already resident
    # 256 tokens. Do NOT raise this: the probe runs with Central already resident,
    # and a longer prefill through a 1.5B expert spikes past the Metal working set
    # and takes the machine down. Measured on the M4 at 16 GB.
    probe = mx.zeros((1, C.EXPERT_PROBE_TOKENS), dtype=mx.int32)
    out = model.model(probe) if hasattr(model, "model") else model(probe)
    mx.eval(out)
    m = peak_mb() - before
    del out, probe, model
    mx.clear_cache()
    if m < 100.0:                 # a 1.5B 4-bit expert is ~1 GB of weights alone (rule 20)
        raise RuntimeError(f"expert peak measured at {m:.0f} MB — the measurement is wrong, refusing to size k on it")
    return m
