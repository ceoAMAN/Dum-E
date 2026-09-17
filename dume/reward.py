"""The grounded reward. Central is the INSTRUMENT, never the judge.

  b[t]   = CE_t(Central, q \\n | y)              baseline, one forward per batch
  a_i[t] = CE_t(Central, q \\n e_i \\n | y)      one forward per expert
  d_i[t] = b[t] - a_i[t]                          nats saved on real token t

Everything that varies across experts is the presence of e_i in a prompt scored
against a FIXED y that came off disk. Experts cannot move the referent. Central
appears in both b and a_i, so its systematic error cancels to first order. The
frame is identical in both passes and in pretraining (question, newline; expert
text, newline), so the paired difference is not confounded by a format change.

Reliability R[c] is a vector over token types (= centroids), fitted ONLY on a
held-out shard, so it is never fitted on the tokens it grades. In training it is
MEASURED, not applied (Aman, 2026-09-18): the delta is a plain mean over the M
target tokens. rho — the mean of R over this target's token types — is still
computed, because admission needs it (a composition Central cannot read cannot
grade an expert) and deployment reads it as the deduction factor.

Why not weight the delta by R: the paired delta already carries reliability.
Where Central is reliable, b[t] is small, so there is no headroom and d[t] is
small; where it is not, the headroom is large and a good note earns a large
d[t]. High reliability -> low gradient, low reliability -> high gradient, with
no scalar anywhere. Multiplying by R on top of that (the old "trust the
instrument" weighting) pushed the other way: it downweighted exactly the
tokens with the most headroom. Live magnitude when removed: R spanned
[0.36, 0.60] across centroids, a 1.7x thumb on the scale against the natural
effect. Writer and reader still index the SAME assign_y[t]: the caller builds
assign_y from the target ids themselves, so len == M always.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config as C


def is_heldout(sample_key: str) -> bool:
    return (zlib.crc32(sample_key.encode("utf-8")) % C.HELDOUT_MOD) == 0


class Reliability:
    """Per-centroid mean baseline CE, as a likelihood in (0, 1]. Cold clusters
    pool to the global estimate (coldness = n_observations, never a flag)."""

    def __init__(self, n_clusters: int):
        self.N = np.zeros(int(n_clusters), dtype=np.float64)
        self.S = np.zeros(int(n_clusters), dtype=np.float64)

    @property
    def C(self) -> int:
        return int(self.N.shape[0])

    def R(self, c: int) -> float:
        if self.N[c] >= C.RELIABILITY_MIN_OBS:
            return float(np.exp(-self.S[c] / self.N[c]))
        tot = self.N.sum()
        return float(np.exp(-self.S.sum() / tot)) if tot > 0 else 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.R(c) for c in range(self.C)], dtype=np.float32)

    def observe(self, b: np.ndarray, assign: np.ndarray) -> None:
        if len(b) != len(assign):
            raise RuntimeError(f"reliability: {len(b)} baseline tokens vs {len(assign)} token types")
        for t in range(len(b)):
            c = int(assign[t])
            if 0 <= c < self.C:
                self.N[c] += 1.0
                self.S[c] += float(b[t])

    def flatness(self) -> Optional[float]:
        warm = [self.R(c) for c in range(self.C) if self.N[c] >= C.RELIABILITY_MIN_OBS]
        return (max(warm) - min(warm)) if len(warm) >= 2 else None

    def total_obs(self) -> float:
        return float(self.N.sum())


@dataclass
class Scored:
    b: np.ndarray
    base_len: int = 0
    deltas: Dict[int, float] = field(default_factory=dict)
    zero_delta: Dict[int, bool] = field(default_factory=dict)
    correct: Dict[int, float] = field(default_factory=dict)   # d[t] > 0, weighted
    halluc: Dict[int, float] = field(default_factory=dict)    # d[t] < 0, weighted (positive magnitude)
    dropped: List[int] = field(default_factory=list)     # experts whose text did not fit the context
    rho: float = 1.0
    admitted: bool = True


def rho_of(reliability: Reliability, assign_y: np.ndarray, M: int) -> float:
    """Mean reliability over this target's token types — the composition's
    deduction factor, MEASURED here and applied only at deployment. Refuses a
    misaligned assignment rather than resampling it (rule 20)."""
    a = np.asarray(assign_y)
    if len(a) != M:
        raise RuntimeError(f"rho: {len(a)} token types for {M} target tokens")
    return float(np.mean([reliability.R(int(c)) for c in a])) if M else 0.0


def score(central, question: str, y: List[int], assign_y: np.ndarray,
          expert_texts: Dict[int, str], reliability: Reliability) -> Scored:
    ctx0 = central.context_ids(question, [], len(y))
    b = central.ce_vector(ctx0, y).astype(np.float64)
    M = len(b)
    base_len = len(ctx0)
    out = Scored(b=b, base_len=base_len, rho=rho_of(reliability, assign_y, M))
    # M >= TARGET_MIN_TOKENS: a mean over one token is not a measurement, and the
    # short-answer sources are a fifth of the mixture, not a rare accident.
    out.admitted = out.rho >= C.R_MIN and M >= C.TARGET_MIN_TOKENS
    for eid, text in expert_texts.items():
        d = _delta_vector(central, question, y, b, text, base_len)
        if d is None:                        # context did not grow: the text was cut, a_i == b by construction
            out.dropped.append(eid)
            continue
        out.deltas[eid], out.correct[eid], out.halluc[eid] = _split(d)
        out.zero_delta[eid] = bool(np.max(np.abs(d)) < 1e-6) if M else True
    return out


def _split(d: np.ndarray) -> tuple:
    """CORRECTNESS and HALLUCINATION are the same gradient, read on both sides
    of zero (Aman, 2026-09-10: "hallucination and correctness is give as
    gradient ... validated on training data").

        d[t] = b[t] - a_i[t]   nats the expert saved Central on REAL token t
        d[t] > 0   the expert moved Central TOWARD the token that actually came
                   off disk                                    -> correctness
        d[t] < 0   the expert moved Central AWAY from it       -> hallucination

    No judge, no second model, no extra pass: the referent is y, which came off
    disk and no expert can move. The mean alone cannot separate an expert that
    helps hugely on half the tokens and hurts on the other half from one that
    does nothing at all — both average to zero. Splitting at zero can, and
    correct - halluc is EXACTLY the old delta, so nothing downstream shifts."""
    if d.size == 0:
        return 0.0, 0.0, 0.0
    pos = float(np.maximum(d, 0.0).mean())
    neg = float(np.maximum(-d, 0.0).mean())
    return pos - neg, pos, neg


def _delta_vector(central, question: str, y: List[int], b: np.ndarray, text: str,
                  ctx0_len: int) -> Optional[np.ndarray]:
    ctx = central.context_ids(question, [text], len(y))
    if len(ctx) <= ctx0_len:
        return None
    a = central.ce_vector(ctx, y).astype(np.float64)
    if len(a) != len(b):                     # canary B: never element-wise comparable
        raise RuntimeError(f"target length mismatch: baseline {len(b)}, expert {len(a)}")
    return b - a


def score_one(central, question: str, y: List[int], b: np.ndarray, text: str,
              ctx0_len: int) -> Optional[float]:
    d = _delta_vector(central, question, y, b, text, ctx0_len)
    if d is None:
        return None
    return float(d.mean()) if d.size else 0.0


class CentralBand:
    """Timeline A or B — the ONE consumer of confidence in this system.

    Central's own apex-nadir has the same shape as the experts': quality against
    a probe variable, judged against what it costs. Central's probe variable is
    how much EXPERT CONTEXT it gets, and that is already measured — d = b - a_i
    is precisely the nats an expert adds, and the fitted envelopes are in those
    same nats. So the gate reads the curves directly:

        gain = z(t*, u) = M(t*) + (u - 0.5)(A(t*) - N(t*))    for the BEST
                                                              available expert

        gain <= 0  ->  Timeline A: the allocation buys nothing, Central alone
        gain >  0  ->  Timeline B: experts, then Central
        no curves  ->  Timeline B: it cannot be known, so do not skip them

    Confidence is inside this by construction: every delta that fitted the
    curves was weighted by Central's held-out reliability R, so a gain Central
    could not read never entered them.

    Why not a quantile of trust: a band at Q.9 of the live trust distribution
    admits ~10% of inputs FOREVER, whatever Central has learned — it measures
    "better than usual", never "good enough". This measures good enough. It is
    also self-earning with no freeze/reroll choice to make: as Central improves,
    b falls, so d = b - a shrinks, so the curves sink toward zero and Timeline A
    fires more often on its own. K -> 0 is the fixed point, not a target.
    """

    def __init__(self, window: int = C.E):
        self.window = int(window)
        self.a_count = 0
        self.seen = 0
        self.last_gain: Optional[float] = None

    def decide(self, alloc, T: int, u: float) -> str:
        gain = alloc.predicted_gain(T, u)
        self.last_gain = gain
        self.seen += 1
        if gain is None:
            return "B"
        if gain <= 0.0:
            self.a_count += 1
            return "A"
        return "B"

    def rate(self) -> float:
        return (self.a_count / self.seen) if self.seen else 0.0

    def state(self) -> Dict[str, object]:
        return {"timeline_a": round(self.rate(), 4), "seen": self.seen,
                "last_gain": (None if self.last_gain is None else round(self.last_gain, 4))}
