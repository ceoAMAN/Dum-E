"""Markov chains: expert migration between clusters, and cluster territory size.

A chain conditions on the PRESENT state and generates FORWARD. It adapts, so it
is non-stationary, and every seductive Markov result (stationary distribution,
equilibrium populations) assumes transitions hold still. They don't. This module
offers ONE-step prediction and nothing more.

Verified properties: bounded evidence (no 1/n freeze), Dirichlet prior instead
of an abstain guard, self-scoring accuracy.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

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
    surplus pool — "where do dormants go" is the trend that matters most. One
    pool chain; its self-scored accuracy is what health reads. (The per-expert
    refinement was deleted: it had no consumer.)"""

    def __init__(self, n_clusters: int):
        self.n_clusters = int(n_clusters)
        self.n_states = self.n_clusters + 1
        self.DORMANT = self.n_clusters
        self.pool = MarkovChain(self.n_states)

    def _state(self, cluster: int) -> int:
        return self.DORMANT if cluster < 0 else int(cluster)

    def record_move(self, eid: int, from_cluster: int, to_cluster: int) -> None:
        self.pool.observe(self._state(from_cluster), self._state(to_cluster))


# ── mounting 2: cluster territory ───────────────────────────────────────────
# High traffic -> TIGHTEN (self-correcting). High traffic -> widen is a runaway.
STARVED, LIGHT, HEALTHY, HEAVY, OVERFULL = 0, 1, 2, 3, 4
LOAD_REGIMES = ("STARVED", "LIGHT", "HEALTHY", "HEAVY", "OVERFULL")
_REGIME_CUTS = (0.25, 0.75, 1.5, 3.0)        # of the population-average presence rate
_TAU_STEP = {STARVED: -2.0, LIGHT: -1.0, HEALTHY: 0.0, HEAVY: +1.0, OVERFULL: +2.0}


def classify_load(rel: float) -> int:
    """rel = this cluster's presence rate / the average cluster's presence rate."""
    for state, cut in enumerate(_REGIME_CUTS):
        if float(rel) < cut:
            return state
    return OVERFULL


class SizeChains:
    """Load is PRESENCE: how often a cluster's territory contains the input
    (sim >= tau_c), over a window of 10*C batches. That is exactly the quantity
    tau gates — tightening reduces it — so the loop self-corrects. The regime is
    relative to the population average, so the setpoint is reachable: the
    average cluster is HEALTHY by construction, a cluster present far more
    often than average tightens, one never present widens. (The earlier version
    measured share of allocated tokens against 1/C; with <= k_max seats among C
    clusters that setpoint was unreachable and every tau ran to a rail.)"""

    def __init__(self, n_clusters: int):
        self.C = int(n_clusters)
        self.chains = [MarkovChain(len(LOAD_REGIMES)) for _ in range(self.C)]
        self.state = [HEALTHY] * self.C
        self.window: deque = deque(maxlen=C.LOAD_WINDOW_PER_CLUSTER * self.C)

    def observe(self, inside: Sequence[int]) -> Optional[np.ndarray]:
        """Record which clusters' territories contained this input. Returns the
        per-cluster presence rate once the window holds >= C batches (every
        cluster has had a turn), else None: no measurement, no move."""
        mask = np.zeros(self.C, dtype=np.float64)
        for c in inside:
            if 0 <= int(c) < self.C:
                mask[int(c)] = 1.0
        self.window.append(mask)
        if len(self.window) < self.C:
            return None
        rate = np.mean(np.stack(self.window), axis=0)
        ref = float(rate.mean())
        if ref <= 0.0:
            return None
        for c in range(self.C):
            nxt = classify_load(rate[c] / ref)
            self.chains[c].observe(self.state[c], nxt)
            self.state[c] = nxt
        return rate

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

    def regimes(self) -> List[str]:
        return [LOAD_REGIMES[s] for s in self.state]


class ThermalRegulator:
    """The device's thermal pressure, measured against where it NORMALLY sits.

    The raw level is REACTIVE (Aman, 2026-09-20: "don't you feel like it is
    reactive to all changes"). It answers a reading identically whether the
    machine has been sitting there all day or just jumped to it, so a box that
    idles warm is permanently throttled and one that spikes for a single batch
    is treated like one that has been climbing for an hour. Two measured
    components fix that:

      BASELINE - "mean temp, the temp at which the system normally works
      happily". The running mean of every level this run has seen. Pressure is
      only what sits ABOVE it, so the regulator costs nothing at the machine's
      own normal operating point, whatever that turns out to be.

      VOLATILITY - "the mean of all change in temp per run, to measure how
      radically it is changing". The running mean of |level - previous level|.
      A machine whose temperature is swinging needs a firmer hand than one
      drifting gently to the same place, so it multiplies the response.

        pressure = max(0, level - baseline) * (1 + volatility)

    Both are means over the run, not constants, and both start at zero, so a
    cold regulator applies nothing and has to EARN the right to throttle.

    The baseline was a flat lifetime mean and could not re-learn (Aman,
    2026-09-21, on the room going 20C -> 34C mid-session: "need to fix"). A
    1/n mean is frozen once n is large -- at batch 3000 one reading moves it
    by 0.0003 -- so a machine whose ROOM changes is pinned to a normal that no
    longer exists and throttles for the rest of the run. The point where a
    machine runs happily is a property of the machine AND its environment, and
    the environment is not stationary.

    So the update is weighted by RUN LENGTH, the count of consecutive reads at
    the current level, over n:

        baseline += min(1, run / n) * (level - baseline)

    A transient resets run to 1 and moves the baseline by 1/n, as before, so
    spikes are still throttled. A level that HOLDS accumulates run, and once it
    has held for a meaningful fraction of the run's history it is by definition
    the new normal and the baseline follows. Nothing is tuned: the horizon is
    the history it has to outweigh. A level that oscillates never accumulates
    run at all, and its volatility is high, so a swinging machine is throttled
    hardest -- which is what it was throttled for."""

    def __init__(self) -> None:
        self.n = 0.0
        self.baseline = 0.0       # mean level
        self.volatility = 0.0     # mean |change| between reads: HOW MUCH it moves at all
        self.last: Optional[float] = None
        self.peak = 0.0
        self.run = 0.0            # consecutive reads at the CURRENT level
        self.gap_up = 0.0         # mean reads BETWEEN upward steps   -> how fast it heats
        self.gap_down = 0.0       # mean reads BETWEEN downward steps -> how fast it cools
        self.since_up = 0.0
        self.since_down = 0.0
        self.n_up = 0.0
        self.n_down = 0.0
        self.excess = 0.0         # how far the last step beat its own mean interval
        self.k: Optional[float] = None   # the ramped k, carried between reads

    def observe(self, level: float) -> None:
        level = float(level)
        self.since_up += 1.0
        self.since_down += 1.0
        if self.last is not None:
            d = level - self.last
            self.volatility += (abs(d) - self.volatility) / max(self.n, 1.0)
            if d > 0:
                self.excess = max(0.0, self.gap_up / self.since_up - 1.0) if self.gap_up else 0.0
                self.n_up += 1.0
                self.gap_up += (self.since_up - self.gap_up) / self.n_up
                self.since_up = 0.0
            elif d < 0:
                self.excess = max(0.0, self.gap_down / self.since_down - 1.0) if self.gap_down else 0.0
                self.n_down += 1.0
                self.gap_down += (self.since_down - self.gap_down) / self.n_down
                self.since_down = 0.0
            else:
                self.excess = 0.0          # it did not move: nothing to react to
        self.n += 1.0
        self.run = self.run + 1.0 if level == self.last else 1.0
        w = min(1.0, self.run / self.n)
        self.baseline += w * (level - self.baseline)
        self.last = level
        self.peak = max(self.peak, level)

    def pressure(self, level: Optional[float] = None) -> float:
        """How hard to throttle, in the units k_thermal takes its root in. Zero
        at or below the baseline: normal operation is free.

        Two independent multipliers sit on that deviation, and they answer
        different questions:

          VOLATILITY - how much this machine moves AT ALL. A machine whose
          temperature is swinging needs a firmer hand than one drifting gently
          to the same place.

          EXCESS - whether the LAST step was abnormal for this machine, i.e.
          arrived sooner than its own mean interval in that direction. A
          machine heating at the pace it always heats at is behaving normally
          and pays nothing extra for it; the same step arriving early does.

        They are added, not multiplied, so neither can swamp the other: a
        steady machine pays 1x, a swinging one pays for the swing, and an
        early step pays for the surprise on top."""
        cur = float(self.last if level is None else level)
        return max(0.0, cur - self.baseline) * (1.0 + self.volatility + self.excess)

    def k_thermal(self, k_max: float, level: Optional[float] = None) -> float:
        """The device's k, RAMPED toward its target rather than snapped to it.

        The ramp rate is the rate this machine itself moves, per direction, and
        the two are not assumed equal: k climbs back at the pace it COOLS and
        falls at the pace it HEATS.

        The rate has to be read as an INTERVAL. macOS publishes the level as an
        ordinal 0-3, so a move is always exactly one step and "levels per move"
        is the constant 1 for every machine alive -- measured on the archived
        run it gave rate_up = rate_down = 1.000 and the ramp degenerated back
        into a snap. The interval between steps is where the speed actually
        lives: one step per 50 reads is 1/50, one per 5 reads is ten times
        that. Rate and gap are reciprocals, so this is the same quantity read
        in the units the signal exists in.

        Until a direction has been seen there is no measured interval to ramp
        at, so k snaps -- an unmeasured rate is not invented."""
        target = float(k_max) ** (1.0 / (1.0 + self.pressure(level)))
        if self.k is None:
            self.k = target
            return self.k
        d = target - self.k
        gap = self.gap_down if d > 0 else self.gap_up   # rising k <=> the machine cooled
        w = 1.0 if gap <= 0.0 else min(1.0, 1.0 / gap)
        if d < 0:
            # TIGHTENING ONLY. A radical departure is not ramped into: the
            # ramp fraction rises with excess, and excess is already the ratio
            # by which the step beat this machine's own interval, so a step ten
            # times early carries w to 0.9 and one far past that snaps outright.
            # Relaxing back up stays on the measured cooling pace regardless --
            # a machine is allowed to be quick to protect itself and slow to
            # trust that it is safe.
            w = max(w, self.excess / (1.0 + self.excess))
        self.k += w * d
        return self.k

    def state(self) -> Dict[str, float]:
        return {"n": self.n, "baseline": self.baseline, "volatility": self.volatility,
                "peak": self.peak, "last": float(self.last or 0.0), "run": self.run,
                "gap_up": self.gap_up, "gap_down": self.gap_down, "excess": self.excess,
                "k": float(self.k if self.k is not None else 0.0)}
