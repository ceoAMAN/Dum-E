"""Markov chains: expert migration between clusters, and cluster territory size.

A chain conditions on the PRESENT state and generates FORWARD. It adapts, so it
is non-stationary, and every seductive Markov result (stationary distribution,
equilibrium populations) assumes transitions hold still. They don't. This module
offers ONE-step prediction and nothing more.

Verified properties (tests in the previous session): bounded evidence (no 1/n
freeze), Dirichlet prior instead of an abstain guard, self-scoring accuracy,
seed strength in absolute pseudo-observations capped at what the pool has seen.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import config as C


class MarkovChain:
    def __init__(self, n_states: int, alpha: float = None, memory: float = None):
        self.n_states = int(n_states)
        self.alpha = float(C.CHAIN_PRIOR if alpha is None else alpha)
        self.memory = float(C.CHAIN_MEMORY if memory is None else memory)
        self.N = np.full((self.n_states, self.n_states), self.alpha, dtype=np.float64)
        self.hits = 0
        self.trials = 0

    def observe(self, i: int, j: int) -> None:
        i, j = int(i), int(j)
        if not (0 <= i < self.n_states and 0 <= j < self.n_states):
            return
        if self.evidence(i) > 0.0:
            self.trials += 1
            if self.next_state(i) == j:
                self.hits += 1
        row = self.N[i]
        total = row.sum()
        if total >= self.memory:
            row *= self.memory / (total + 1.0)
        row[j] += 1.0

    def seed_from(self, other: "MarkovChain", strength: float = 1.0) -> None:
        if other.n_states != self.n_states:
            return
        pool_evidence = other.N.sum(axis=1) - other.alpha * other.n_states
        w = np.minimum(1.0, np.maximum(0.0, pool_evidence) / max(strength, 1e-9))
        rows = other.N / other.N.sum(axis=1, keepdims=True)
        self.N = self.alpha + strength * rows * w[:, None]

    def predict(self, i: int) -> np.ndarray:
        row = self.N[int(i)]
        return row / row.sum()

    def next_state(self, i: int) -> int:
        return int(np.argmax(self.N[int(i)]))

    def evidence(self, i: int) -> float:
        return float(self.N[int(i)].sum() - self.alpha * self.n_states)

    def accuracy(self) -> Optional[float]:
        return (self.hits / self.trials) if self.trials else None


# ── mounting 1: expert migration ────────────────────────────────────────────
class MigrationChains:
    """State = cluster index, plus one extra state DORMANT (= n_clusters) for the
    surplus pool — "where do dormants go" is the trend that matters most. Pool
    chain carries the signal; per-expert chains refine it once an expert has
    actually moved enough to have an opinion."""

    def __init__(self, n_clusters: int):
        self.n_clusters = int(n_clusters)
        self.n_states = self.n_clusters + 1
        self.DORMANT = self.n_clusters
        self.pool = MarkovChain(self.n_states)
        self.per_expert: Dict[int, MarkovChain] = {}

    def _state(self, cluster: int) -> int:
        return self.DORMANT if cluster < 0 else int(cluster)

    def _chain(self, eid: int) -> MarkovChain:
        if eid not in self.per_expert:
            c = MarkovChain(self.n_states)
            c.seed_from(self.pool, strength=C.CHAIN_SEED_STRENGTH)
            self.per_expert[eid] = c
        return self.per_expert[eid]

    def record_move(self, eid: int, from_cluster: int, to_cluster: int) -> None:
        i, j = self._state(from_cluster), self._state(to_cluster)
        self.pool.observe(i, j)
        self._chain(int(eid)).observe(i, j)

    def predict_next(self, eid: int, current: int) -> Tuple[int, float]:
        current = self._state(current)
        c = self._chain(int(eid))
        own_w, pool_w = c.evidence(current), self.pool.evidence(current)
        if own_w + pool_w <= 0.0:
            return int(current), 0.0
        p = (own_w * c.predict(current) + pool_w * self.pool.predict(current)) / (own_w + pool_w)
        j = int(np.argmax(p))
        return j, float(p[j])


# ── mounting 2: cluster territory ───────────────────────────────────────────
# High traffic -> TIGHTEN (self-correcting). High traffic -> widen is a runaway.
STARVED, LIGHT, HEALTHY, HEAVY, OVERFULL = 0, 1, 2, 3, 4
LOAD_REGIMES = ("STARVED", "LIGHT", "HEALTHY", "HEAVY", "OVERFULL")
_REGIME_CUTS = (0.25, 0.75, 1.5, 3.0)
_TAU_STEP = {STARVED: -2.0, LIGHT: -1.0, HEALTHY: 0.0, HEAVY: +1.0, OVERFULL: +2.0}


def classify_load(share: float, n_clusters: int) -> int:
    rel = float(share) * float(max(1, n_clusters))
    for state, cut in enumerate(_REGIME_CUTS):
        if rel < cut:
            return state
    return OVERFULL


class SizeChains:
    def __init__(self, n_clusters: int):
        self.chains = [MarkovChain(len(LOAD_REGIMES)) for _ in range(int(n_clusters))]
        self.state = [HEALTHY] * int(n_clusters)

    def observe_load(self, c: int, share: float, n_clusters: int) -> int:
        nxt = classify_load(share, n_clusters)
        self.chains[c].observe(self.state[c], nxt)
        self.state[c] = nxt
        return nxt

    def next_tau(self, c: int, tau_now: float) -> float:
        """Move tau toward where the cluster is HEADING. Step-limited and clamped
        to [SIM_FAR, TAU_MAX] so a mispredicting chain bounds the damage."""
        ch = self.chains[c]
        cur = self.state[c]
        predicted = ch.next_state(cur) if ch.evidence(cur) > 0.0 else cur
        tau = float(tau_now) + _TAU_STEP[predicted] * C.TAU_STEP
        return max(C.SIM_FAR, min(C.TAU_MAX, tau))

    def accuracy(self) -> Optional[float]:
        hits = sum(ch.hits for ch in self.chains)
        trials = sum(ch.trials for ch in self.chains)
        return (hits / trials) if trials else None
