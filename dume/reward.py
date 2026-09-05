"""The grounded reward. Central is the INSTRUMENT, never the judge.

  b[t]   = CE_t(Central, q | y)              baseline, one forward per batch
  a_i[t] = CE_t(Central, q | e_i | y)        one forward per expert
  d_i[t] = b[t] - a_i[t]                     nats saved on real token t

Everything that varies across experts is the presence of e_i in a prompt scored
against a FIXED y that came off disk. Experts cannot move the referent. Central
appears in both b and a_i, so its systematic error cancels to first order.

Reliability R[c] is a vector over token types (= centroids), fitted ONLY on a
held-out shard, so it is never fitted on the tokens it grades. It weights the
delta per token: an expert whose gain lands where Central cannot read is
discounted. With a scalar R this weight cancels across experts exactly; the
vector is what makes it do work.

SIGN CHOICE (flagged for Aman): weight = R (trust the instrument where it is
accurate). The alternative — weight by Central's UNreliability, "reward the
headroom" — re-introduces a Goodhart target. One-line change in `weights()`.
"""
from __future__ import annotations

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
        for t in range(min(len(b), len(assign))):
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
    deltas: Dict[int, float] = field(default_factory=dict)
    zero_delta: Dict[int, bool] = field(default_factory=dict)
    rho: float = 1.0
    admitted: bool = True
    context_grew: bool = True


def weights(reliability: Reliability, assign_y: np.ndarray, M: int) -> np.ndarray:
    """Per-target-token weight. SIGN CHOICE lives here: R, not 1-R."""
    a = np.asarray(assign_y)
    if len(a) != M:
        a = a[np.minimum(len(a) - 1, (np.arange(M) * len(a)) // max(1, M))] if len(a) else np.zeros(M, dtype=int)
    return np.array([reliability.R(int(c)) for c in a], dtype=np.float64)


def score(central, question: str, y: List[int], assign_y: np.ndarray,
          expert_texts: Dict[int, str], reliability: Reliability) -> Scored:
    ctx0 = central.context_ids(question, [], len(y))
    b = central.ce_vector(ctx0, y).astype(np.float64)
    M = len(b)
    w = weights(reliability, assign_y, M)
    out = Scored(b=b, rho=float(w.mean()) if M else 0.0)
    out.admitted = out.rho >= C.R_MIN and M > 0
    for eid, text in expert_texts.items():
        ctx = central.context_ids(question, [text], len(y))
        grew = len(ctx) > len(ctx0)
        out.context_grew = out.context_grew and grew
        a = central.ce_vector(ctx, y).astype(np.float64)
        if len(a) != M:                      # canary B: never element-wise comparable
            raise RuntimeError(f"target length mismatch: baseline {M}, expert {eid} {len(a)}")
        d = b - a
        delta = float((w * d).sum() / w.sum()) if w.sum() > 0 else 0.0
        out.deltas[eid] = delta
        out.zero_delta[eid] = bool(np.max(np.abs(d)) < 1e-6) if M else True
    return out


def score_one(central, question: str, y: List[int], b: np.ndarray, w: np.ndarray, text: str) -> float:
    ctx = central.context_ids(question, [text], len(y))
    a = central.ce_vector(ctx, y).astype(np.float64)
    if len(a) != len(b):
        raise RuntimeError("target length mismatch in score_one")
    d = b - a
    return float((w * d).sum() / w.sum()) if w.sum() > 0 else 0.0
