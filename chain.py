"""Markov chains: expert migration between clusters, and cluster territory size.

Two things in this system change slowly and need to be REMEMBERED rather than
recomputed: which cluster an expert belongs in, and how much of the input space
a cluster claims. Both were previously handled by keeping a register — an
append-only list of events, rescanned on every query (gating.migration_delta),
growing without bound and answering only "what happened on average".

A chain is a different object. It conditions on the PRESENT state (this expert,
in this cluster, right now) and it generates FORWARD (where does it go next).
A register can do neither. The accumulated counts are not a summary of the past
kept alongside it — they ARE the past, in the only form that predicts.

WHAT THIS DELIBERATELY DOES NOT DO
The chain adapts: new observations keep moving the transition probabilities, on
purpose. That makes it non-stationary, and every seductive Markov result —
stationary distribution, equilibrium populations, long-run shares — assumes the
transitions hold still. They don't here. So this module offers ONE-step
prediction and no more. A stationary distribution would always compute, always
look reasonable, and never tell you it was meaningless. That is precisely the
failure mode this system has already been burned by once.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import configs


class MarkovChain:
    """Decayed-count transition estimator over a small discrete state space.

    Storage is one S x S matrix of counts. Reading a row and normalising gives
    P(next | current). Three properties matter and each earns its complexity:

    BOUNDED EVIDENCE. With pure accumulation a new observation moves the estimate
    by 1/n: 2% at n=50, 0.01% at n=10_000. The chain would freeze — still
    running, still confident, no longer responsive, no symptom. So a row
    accumulates freely up to `memory` and is rescaled before each increment
    after that. Asymptotically this is an EMA with lambda = 1 - 1/memory, but it
    behaves better cold (early observations count fully) and the parameter is
    sayable in English: "this chain remembers `memory` transitions."

    A PRIOR INSTEAD OF A FLOOR. Every cell starts at `alpha` rather than zero, so
    a row with three observations returns near-uniform — "no opinion" — instead
    of a confident estimate off three samples. That removes the
    `if n < threshold: abstain` guard entirely. A guard can be forgotten at a
    call site; a prior cannot.

    SELF-CHECK. Every observe() scores the standing prediction against what
    actually happened before folding the new evidence in. The chain therefore
    reports its own accuracy for free. If a cluster re-forms underneath it, or a
    state definition drifts, the hit rate falls and says so. Nothing else in this
    system has had that property.
    """

    def __init__(self, n_states: int, alpha: float = None, memory: float = None):
        self.n_states = int(n_states)
        self.alpha = float(configs.CHAIN_PRIOR if alpha is None else alpha)
        self.memory = float(configs.CHAIN_MEMORY if memory is None else memory)
        self.N = np.full((self.n_states, self.n_states), self.alpha, dtype=np.float64)
        self.hits = 0
        self.trials = 0

    # ── evidence ────────────────────────────────────────────────────────────
    def observe(self, i: int, j: int) -> None:
        """Record a transition i -> j, scoring the standing prediction first."""
        i, j = int(i), int(j)
        if not (0 <= i < self.n_states and 0 <= j < self.n_states):
            return
        if self.evidence(i) > 0.0:          # only score rows that had an opinion
            self.trials += 1
            if self.next_state(i) == j:
                self.hits += 1
        row = self.N[i]
        total = row.sum()
        if total >= self.memory:
            # Rescale so the row holds `memory` units of evidence AFTER the
            # increment. Old evidence is never deleted, only diluted — which is
            # what keeps one new observation worth a constant 1/memory forever
            # instead of decaying to nothing.
            row *= self.memory / (total + 1.0)
        row[j] += 1.0

    def seed_from(self, other: "MarkovChain", strength: float = 1.0) -> None:
        """Initialise from a pool-level chain at low strength.

        A fresh (expert, cluster) pair has no past, but the pool does. Seeding
        from the pool's typical behaviour means a new or dormant expert starts
        from "this is how experts generally move here" rather than a coin flip,
        and its own evidence overwrites that as it earns some. For migration this
        is not a nicety: an individual expert migrates a handful of times ever,
        so a per-expert S x S matrix is prior-dominated for most of its life and
        the pool chain is where the signal actually lives.
        """
        if other.n_states != self.n_states:
            return
        # Never inherit more belief than the pool actually holds. Seeding at a
        # flat `strength` from a COLD pool manufactures evidence out of nothing:
        # the row would report 8 observations of support for a uniform guess,
        # and a caller gating on confidence would act on it. Per-row, transfer
        # at most what that row has really seen.
        pool_evidence = other.N.sum(axis=1) - other.alpha * other.n_states
        w = np.minimum(1.0, np.maximum(0.0, pool_evidence) / max(strength, 1e-9))
        rows = other.N / other.N.sum(axis=1, keepdims=True)
        rows = rows * w[:, None]
        # `strength` is an absolute quantity of pseudo-observations, NOT scaled
        # by the state count. Scaling it by n_states would silently make the
        # pool harder to outvote every time a cluster is added — at 20 clusters
        # a new expert would need ~40 real moves to overrule an inherited
        # opinion, and experts migrate a handful of times in their whole life.
        self.N = self.alpha + strength * rows

    # ── reading ─────────────────────────────────────────────────────────────
    def predict(self, i: int) -> np.ndarray:
        """P(next | current = i). One step. See the module note on horizons."""
        row = self.N[int(i)]
        return row / row.sum()

    def next_state(self, i: int) -> int:
        return int(np.argmax(self.N[int(i)]))

    def evidence(self, i: int) -> float:
        """Real observations backing row i, with the prior's mass removed. Zero
        means the row is speaking entirely from its prior."""
        return float(self.N[int(i)].sum() - self.alpha * self.n_states)

    def accuracy(self) -> Optional[float]:
        return (self.hits / self.trials) if self.trials else None

    def resize(self, n_states: int) -> None:
        """Grow or shrink the state space, keeping the overlapping block.

        Only safe when state INDICES keep their meaning. They do not survive a
        cluster re-formation — "cluster 7" afterwards is a different region of
        space — so re-formation must reset migration chains outright rather than
        resize them. That is a loud, discrete event and callers should say so.
        """
        n = int(n_states)
        if n == self.n_states:
            return
        fresh = np.full((n, n), self.alpha, dtype=np.float64)
        keep = min(n, self.n_states)
        fresh[:keep, :keep] = self.N[:keep, :keep]
        self.N = fresh
        self.n_states = n


# ── mounting 1: expert migration between clusters ───────────────────────────
class MigrationChains:
    """Where each expert sits, and where it is heading.

    State = cluster index. The pool chain carries the signal; per-expert chains
    refine it once an expert has actually moved enough times to have an opinion
    of its own. `evidence()` is what tells you which of the two is speaking.
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = int(n_clusters)
        self.pool = MarkovChain(self.n_clusters)
        self.per_expert: Dict[int, MarkovChain] = {}
        self.home: Dict[int, int] = {}

    def _chain(self, expert_id: int) -> MarkovChain:
        eid = int(expert_id)
        if eid not in self.per_expert:
            c = MarkovChain(self.n_clusters)
            c.seed_from(self.pool, strength=configs.CHAIN_SEED_STRENGTH)
            self.per_expert[eid] = c
        return self.per_expert[eid]

    def record_move(self, expert_id: int, from_cluster: int, to_cluster: int) -> None:
        self.pool.observe(from_cluster, to_cluster)
        self._chain(expert_id).observe(from_cluster, to_cluster)
        self.home[int(expert_id)] = int(to_cluster)

    def predict_home(self, expert_id: int) -> Tuple[int, float]:
        """The cluster this expert is most likely in next, and the confidence."""
        cur = self.home.get(int(expert_id))
        if cur is None:
            return -1, 0.0
        c = self._chain(expert_id)
        # Blend own evidence with the pool's, weighted by how much each has. An
        # expert that has never moved defers entirely to the pool.
        own_w = c.evidence(cur)
        pool_w = self.pool.evidence(cur)
        if own_w + pool_w <= 0.0:
            return int(cur), 0.0
        p = (own_w * c.predict(cur) + pool_w * self.pool.predict(cur)) / (own_w + pool_w)
        j = int(np.argmax(p))
        return j, float(p[j])

    def reset_for_reformation(self, n_clusters: int) -> None:
        """Cluster re-formation invalidates every state index. Start clean and
        say so — a silently remapped chain would keep predicting against regions
        of space that no longer exist."""
        print(f"[chain] cluster re-formation: discarding migration evidence "
              f"({len(self.per_expert)} expert chains, pool accuracy "
              f"{self.pool.accuracy()}), rebuilding at {n_clusters} clusters")
        self.__init__(int(n_clusters))


# ── mounting 2: cluster territory size ──────────────────────────────────────
# A cluster's territory in the membership-test design is a spherical cap,
# {x : v_k . x >= tau_k}, whose area is monotone in tau_k alone. So "resizing a
# cluster" is one number. The direction v_k is FROZEN at formation — every
# formation-time guarantee (max-min vector, shared-direction removal, the
# calibration done last) is a statement about direction, and letting it drift at
# runtime would void them with no symptom. The radius is the one degree of
# freedom that is safe to move online.
#
# Feedback sign is what makes this work rather than collapse. High traffic ->
# WIDEN is a runaway: more territory, more traffic, more territory, until one
# cluster eats the sphere. High traffic -> TIGHTEN is self-correcting: a cluster
# absorbing too much is too coarse, so it narrows and the overflow lands on its
# neighbours; a starved cluster widens to catch more.
STARVED, LIGHT, HEALTHY, HEAVY, OVERFULL = 0, 1, 2, 3, 4
LOAD_REGIMES = ("STARVED", "LIGHT", "HEALTHY", "HEAVY", "OVERFULL")
_REGIME_CUTS = (0.25, 0.75, 1.5, 3.0)      # multiples of an equal share
_TAU_STEP = {STARVED: -2.0, LIGHT: -1.0, HEALTHY: 0.0, HEAVY: +1.0, OVERFULL: +2.0}


def classify_load(share: float, n_clusters: int) -> int:
    """Bucket a cluster's traffic share against what an equal split would give."""
    if n_clusters <= 0:
        return HEALTHY
    rel = share * float(n_clusters)
    for state, cut in enumerate(_REGIME_CUTS):
        if rel < cut:
            return state
    return OVERFULL


class SizeChains:
    """Per-cluster load-regime chains driving tau_k.

    The chain rather than a plain control law because a reactive rule only
    tightens AFTER the cluster is already overloaded — it lags by a batch every
    time. One-step-ahead lets the adjustment land before the overload, and the
    hit rate says whether the prediction is worth acting on.
    """

    def __init__(self):
        self.chains: Dict[str, MarkovChain] = {}
        self.state: Dict[str, int] = {}
        self.tau: Dict[str, float] = {}

    def _chain(self, cluster_id: str) -> MarkovChain:
        if cluster_id not in self.chains:
            self.chains[cluster_id] = MarkovChain(len(LOAD_REGIMES))
            self.state[cluster_id] = HEALTHY
            self.tau[cluster_id] = float(configs.SIM_MEMBER)
        return self.chains[cluster_id]

    def observe_load(self, cluster_id: str, share: float, n_clusters: int) -> int:
        c = self._chain(cluster_id)
        nxt = classify_load(share, n_clusters)
        c.observe(self.state[cluster_id], nxt)
        self.state[cluster_id] = nxt
        return nxt

    def next_tau(self, cluster_id: str) -> float:
        """Adjust tau toward where this cluster is HEADING, not where it is.

        Step-limited and hard-clamped inside the empirically validated band
        structure: tau never leaves [SIM_NEIGHBOUR, TAU_MAX], so a mispredicting
        chain can make a cluster somewhat too tight or too loose but can never
        make it swallow the sphere or vanish. Clusters are coupled — tightening
        one pushes its rejects onto its neighbours — and this does not enforce
        conservation of total territory; the clamps bound the damage instead.
        Watch for inputs that match NO cluster; that is what over-tightening
        looks like from the outside.
        """
        c = self._chain(cluster_id)
        cur = self.state[cluster_id]
        predicted = c.next_state(cur) if c.evidence(cur) > 0.0 else cur
        step = _TAU_STEP[predicted] * configs.TAU_STEP
        tau = self.tau[cluster_id] + step
        tau = max(float(configs.SIM_NEIGHBOUR), min(float(configs.TAU_MAX), tau))
        self.tau[cluster_id] = tau
        return tau


# ── persistence ─────────────────────────────────────────────────────────────
def save_chains(path: str, migration: MigrationChains, sizing: SizeChains) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump({"migration": migration, "sizing": sizing}, f)


def load_chains(path: str) -> Tuple[Optional[MigrationChains], Optional[SizeChains]]:
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        return blob.get("migration"), blob.get("sizing")
    except Exception as e:
        print(f"[chain] could not load {path}: {e}")
        return None, None
