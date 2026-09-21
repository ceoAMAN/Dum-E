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
    """The device's thermal pressure, read in DEGREES off the silicon.

    This used to read `NSProcessInfo.thermalState()`, an ordinal 0-3, and on
    this machine that ordinal is a CONSTANT. Measured over 2900 archived
    batches and 541 live ones it returned `fair` every single time. Every
    quantity derived from it was therefore identically zero -- the baseline
    converged to the one value the level ever took, volatility sat at 0.000,
    the up/down intervals were never set because no step ever happened, and
    `k_thermal` went 2.758 -> 4.000 = k_max and stayed there for the rest of
    the run. **The device never once cast a vote in the tug of war over k.**

    The silicon is not constant. `models.die_temp()` reads 24 SoC die sensors
    and their mean moved 52 -> 56 C inside one minute of the same run, with
    max(tdie) touching 67. What was inert was the sensor, not the machine.

    With degrees the whole apparatus the ordinal needed disappears. There is no
    baseline-vs-level subtraction, no volatility, no up/down interval means, no
    `excess`, no ramp and no learned threshold. Those existed to squeeze a rate
    out of a four-valued signal that has no usable pointwise derivative. A
    continuous one does, so the regulator is just the signal and its first two
    derivatives (Aman, 2026-09-21: "f(x) = y'' + y' + c", and on the regulator
    it replaces, "it is dumb regulators it feels useless"):

        span     = peak - floor        the range this machine has actually worked over
        z        = (T - mean) / span   how far above normal it sits right now
        pressure = max(0, z - mean|z|)

    THE EXPLICIT DERIVATIVES WERE BUILT, MEASURED, AND REMOVED. Both forms
    failed on measurement rather than on argument:

      on the READING, d/dread and d2/dread2 are noise. Differencing doubles it
      each time, and on a calm fixture -- 0.6 C of ripple over a 10 C span -- a
      SETTLED machine spiked to pressure 2.06 while a real +3 C step scored
      LESS than idle. The order came out backwards.

      on the MEAN, they vanish. The mean re-learns at run/n, so at n = 260 its
      drift is ~0.001 against a z of 0.5. Audited: z carried 97-99% of the
      signal, the first derivative 0.8-1.9%, the second 0.4-0.7%. A term that
      moves the answer by half a percent is a mechanism that runs perfectly and
      does nothing.

    They were redundant, not merely weak. z is ALREADY the rate term, because
    the mean lags: measured at the same temperature, a die that has just reached
    60 C scores z = +0.504 and one that has been at 60 C for 400 reads scores
    z = 0.000. A fast reading against a slowly re-learning normal is a high-pass
    filter whose time constant is the machine's own history. That is where the
    second-order behaviour actually lives -- in the gap between the two, not in
    a difference taken on either.

    Dividing by the span is what makes the three ADDABLE without an invented
    coefficient, and that was the last place a constant could have hidden.
    Degrees cannot be summed with degrees-per-read; fractions of the span can be
    summed with fractions-per-read, because the read is the clock the regulator
    acts on.

    The subtraction is the machine's own jitter: the running mean of |z| while
    it works. Without it a settled machine pays a permanent tax for its own
    ripple, which is the whole "don't you feel like it is reactive to all
    changes" complaint the regulator exists to answer. Nothing is configured --
    a quiet machine reacts to a small move, a noisy one needs a bigger one, and
    both pay nothing to idle.

    The span, and NOT the mean absolute deviation. The deviation is the noise
    floor -- measured at 0.33 C on this machine -- so a z-score taken in it
    makes every real thermal event tens of sigma: a 3 C rise scored z = 8.7,
    pressure 26.6, and collapsed k to 1.05. A 3 C rise on a die that idles at 45
    and works at 67 is not an emergency. The span is the range the machine has
    been measured over, so pressure reaches 1 -- k halved -- when the excursion
    is the size of the machine's whole working range, and it re-scales itself if
    the machine ever works harder than it has before.

    Cooling is negative in all three terms and the max() floors it at zero, so a
    machine that is cool, or heating no faster than it normally does, pays
    exactly nothing and k sits at k_max. That is the "tendency to get f(x) = 4".

    The mean RE-LEARNS, because the point a machine runs happily is a property
    of the machine AND its room, and the room is not stationary (Aman, on 20C ->
    34C mid-session: "need to fix"). The update is weighted by how long the
    temperature has held the SAME SIDE of the mean, over n:

        mean += min(1, run / n) * (T - mean)

    A transient flips sides, resets run to 1, and moves the mean by 1/n. A
    genuine shift holds one side, accumulates run, and once it has held for a
    meaningful fraction of the run's history it IS the new normal. Nothing is
    tuned: the horizon is the history it has to outweigh.

    That re-learning is also the one real danger of degrees over an ordinal. An
    ordinal is capped at 3; a die is not capped at anything, and a mean that
    always follows would normalise its way into a cooked chip. So the OS keeps a
    veto underneath, at `serious` -- Apple's number, not one we chose, and by
    then the OS is already throttling, so adding experts worsens exactly what it
    is complaining about."""

    SERIOUS = 2.0            # NSProcessInfoThermalStateSerious

    def __init__(self) -> None:
        self.n = 0.0
        self.mean = 0.0          # this machine's normal die temperature, in C
        self.dev = 0.0           # mean |z|: this machine's own jitter, the deadband
        self.run = 0.0           # consecutive reads on the same side of the mean
        self.last: Optional[float] = None
        self.peak = 0.0
        self.floor = 0.0         # peak - floor is the SPAN: the scale z is taken in
        self.level = 0.0         # the OS ordinal, kept only for its veto
        self.z = 0.0             # heat now, as a fraction of the span above normal

    def observe(self, temp: float, level: float = 0.0) -> None:
        t = float(temp)
        self.level = float(level)
        if self.last is None:
            # the first read IS the normal. Starting the mean at zero would make
            # the first deviation the whole temperature, ~50 C of spread that
            # then takes hundreds of reads to decay back out of the scale.
            self.n, self.mean, self.run = 1.0, t, 1.0
            self.last, self.peak, self.floor = t, t, t
            return
        # deviation is measured against the mean BEFORE this read moves it, so
        # the scale is never flattered by the sample that is widening it
        self.n += 1.0
        self.run = (self.run + 1.0 if self.last is not None
                    and (t >= self.mean) == (self.last >= self.mean) else 1.0)
        self.mean += min(1.0, self.run / self.n) * (t - self.mean)
        self.peak, self.floor = max(self.peak, t), min(self.floor, t)
        span = self.peak - self.floor
        if span > 0.0:
            self.z = (t - self.mean) / span
        self.dev += (abs(self.z) - self.dev) / self.n
        self.last = t

    def out_of_hand(self) -> bool:
        """The OS's own call, underneath ours. A third party's number."""
        return self.level >= self.SERIOUS

    def pressure(self) -> float:
        """How far above its own normal this die sits, as a fraction of the range
        it works over, less the jitter it shows while doing nothing. Zero when
        cool or merely steady: normal operation is free."""
        return max(0.0, self.z - self.dev)

    def k_thermal(self, k_max: float) -> float:
        """How many experts the device is willing to run right now.

        Each unit of pressure takes another root of the RAM bound -- the same
        sqrt bracket k_tier, GENERAL_EXPERTS and capacity() are built from, so
        no new constant enters here either."""
        if self.out_of_hand():
            return 1.0
        return float(k_max) ** (1.0 / (1.0 + self.pressure()))

    def state(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "dev": self.dev, "run": self.run,
                "peak": self.peak, "floor": self.floor, "last": float(self.last or 0.0),
                "level": self.level, "z": self.z, "pressure": self.pressure()}
