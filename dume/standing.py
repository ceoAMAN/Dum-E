"""Expert standing, keyed by the ROUTING unit: (expert, cluster).

Four running numbers per pair — n, sum(delta), sum(delta^2), sum(tokens) — give
mean, variance and normalisation in constant space:

    S_ik = ( sum_delta - z * sqrt(n * var) ) / sum_tokens

A RATE: doubling output for the same total help halves S, so volume cannot buy
standing and the class -> allocation -> token-count -> class loop is broken at
the divisor. The z-discount means two lucky batches cannot outrank fifty
consistent ones.

Membership has exactly ONE writer: migrate(). It is an explicit decision, never
a side effect of an expert having run (rule 7). sqrt(E) experts are GENERAL —
candidates everywhere, never migrated. Class size per cluster is capped at
CENTROID_EXPERTS, so promotion means displacement.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

GENERAL = -1      # sqrt(E) experts: candidates everywhere, never migrated
SURPLUS = -2      # dormant pool: eligible only as the trial slot until they earn a seat


class Standing:
    def __init__(self, n_clusters: int, n_experts: int = C.E):
        self.C, self.E = int(n_clusters), int(n_experts)
        self.n = np.zeros((self.E, self.C), dtype=np.float64)
        self.sd = np.zeros((self.E, self.C), dtype=np.float64)
        self.sd2 = np.zeros((self.E, self.C), dtype=np.float64)
        self.sn = np.zeros((self.E, self.C), dtype=np.float64)
        self.assigned = np.full(self.E, SURPLUS, dtype=np.int64)
        # first sqrt(E) general; then CENTROID_EXPERTS per cluster; the rest are surplus (dormant)
        self.assigned[:min(C.GENERAL_EXPERTS, self.E)] = GENERAL
        e = C.GENERAL_EXPERTS
        for c in range(self.C):
            for _ in range(C.CENTROID_EXPERTS):
                if e < self.E:
                    self.assigned[e] = c
                    e += 1
        self.moves = 0

    # ── reading ─────────────────────────────────────────────────────────────
    def score(self, eid: int, cid: int) -> Optional[float]:
        n = self.n[eid, cid]
        if n <= 0 or self.sn[eid, cid] <= 0:
            return None
        mean = self.sd[eid, cid] / n
        var = max(0.0, self.sd2[eid, cid] / n - mean * mean)
        return float((self.sd[eid, cid] - C.STANDING_Z * math.sqrt(n * var)) / self.sn[eid, cid])

    def members(self, cid: int) -> List[int]:
        return [int(e) for e in np.where(self.assigned == cid)[0]]

    def generals(self) -> List[int]:
        return [int(e) for e in np.where(self.assigned == GENERAL)[0]]

    def surplus(self) -> List[int]:
        return [int(e) for e in np.where(self.assigned == SURPLUS)[0]]

    def ranked(self, cid: int) -> List[Tuple[int, float]]:
        """Members + generals with evidence in this cluster, best first."""
        out = []
        for e in self.members(cid) + self.generals():
            s = self.score(e, cid)
            if s is not None:
                out.append((e, s))
        return sorted(out, key=lambda x: -x[1])

    def trial(self, cid: int, exclude: set) -> Optional[int]:
        """The least-measured eligible expert — the dormant slot. Surplus first:
        that is the only way a dormant expert ever gets measured."""
        pool = [e for e in self.surplus() if e not in exclude]
        if not pool:
            pool = [e for e in self.members(cid) + self.generals() if e not in exclude]
        if not pool:
            pool = [e for e in range(self.E) if e not in exclude]
        if not pool:
            return None
        return int(min(pool, key=lambda e: (self.n[e, cid], e)))

    def total_n(self) -> float:
        return float(self.n.sum())

    # ── the two writers ─────────────────────────────────────────────────────
    def observe(self, eid: int, cid: int, delta: float, n_tokens: int) -> None:
        self.n[eid, cid] += 1.0
        self.sd[eid, cid] += float(delta)
        self.sd2[eid, cid] += float(delta) ** 2
        self.sn[eid, cid] += float(max(n_tokens, C.SPAN_MIN))

    def migrate(self, chains=None) -> List[Tuple[int, int, int]]:
        """Move each non-general expert to the cluster where it stands best,
        if it can OUTPERFORM THE BOTTOM of that cluster's class (or the class
        has room). Displaced experts take the mover's old seat. Returns moves."""
        moves: List[Tuple[int, int, int]] = []
        for e in range(self.E):
            cur = int(self.assigned[e])
            if cur == GENERAL:
                continue
            best, best_s = cur, (self.score(e, cur) if cur >= 0 else None)
            for c in range(self.C):
                s = self.score(e, c)
                if s is not None and (best_s is None or s > best_s):
                    best, best_s = c, s
            if best == cur or best_s is None:
                continue
            seated = [(m, self.score(m, best)) for m in self.members(best)]
            if len(seated) < C.CENTROID_EXPERTS:
                self._move(e, cur, best, moves, chains)
                continue
            scored = [(m, s) for m, s in seated if s is not None]
            unscored = [m for m, s in seated if s is None]
            bottom, bottom_s = (min(scored, key=lambda x: x[1]) if scored else (unscored[0], None))
            if bottom_s is None or best_s > bottom_s:
                self._move(e, cur, best, moves, chains)
                self._move(bottom, best, cur, moves, chains)      # displacement
        return moves

    def _move(self, e: int, frm: int, to: int, moves, chains) -> None:
        self.assigned[e] = to
        self.moves += 1
        moves.append((e, frm, to))
        if chains is not None:
            chains.record_move(e, frm, to)

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, object]:
        return {"n": self.n, "sd": self.sd, "sd2": self.sd2, "sn": self.sn,
                "assigned": self.assigned, "moves": self.moves, "C": self.C, "E": self.E}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Standing":
        s = Standing(int(d["C"]), int(d["E"]))
        for k in ("n", "sd", "sd2", "sn", "assigned"):
            setattr(s, k, np.asarray(d[k]).copy())
        s.moves = int(d.get("moves", 0))
        return s
