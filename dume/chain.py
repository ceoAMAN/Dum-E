"""Markov chains: expert migration between clusters, and cluster territory size.

A chain conditions on the PRESENT state and generates FORWARD. It adapts, so it
is non-stationary, and every seductive Markov result (stationary distribution,
equilibrium populations) assumes transitions hold still. They don't. This module
offers ONE-step prediction and nothing more.

Verified properties: bounded evidence (no 1/n freeze), Dirichlet prior instead
of an abstain guard, self-scoring accuracy.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
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

    # NSProcessInfoThermalStateCritical. Apple's own number for "this machine
    # is in trouble", not ours -- see out_of_hand().
    CRITICAL = 3.0

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
        self.prev_gap_up = 0.0    # the previous interval, so the NEXT one can be compared to it
        self.prev_gap_down = 0.0
        self.accel_up = 0.0       # d2y/dx2, MEAN over the run: what this machine typically does,
        self.accel_down = 0.0     # folded across runs and seeded back. Not the live signal.
        self.accel_now = 0.0      # the LAST fractional change: what it is doing right now
        self.n_acc_up = 0.0
        self.n_acc_down = 0.0
        self.u_mean = 0.0         # mean urgency PER STEP: this machine's normal surprise
        self.u_dev = 0.0          # and its spread. together they are the out-of-hand line
        self.n_u = 0.0
        self.k: Optional[float] = None   # k, carried between reads
        self.v = 0.0              # ...and its velocity. k is second order.

    def observe(self, level: float) -> None:
        level = float(level)
        self.since_up += 1.0
        self.since_down += 1.0
        if self.last is not None:
            d = level - self.last
            self.volatility += (abs(d) - self.volatility) / max(self.n, 1.0)
            if d > 0:
                self._step_up()
            elif d < 0:
                self._step_down()
            else:
                self.excess = 0.0          # it did not move: nothing to react to
        self.n += 1.0
        self.run = self.run + 1.0 if level == self.last else 1.0
        w = min(1.0, self.run / self.n)
        self.baseline += w * (level - self.baseline)
        self.last = level
        self.peak = max(self.peak, level)

    def _step_up(self) -> None:
        # The FIRST step in a direction yields no interval: since_up measures
        # time since BOOT, not time between steps. Recording it invents a gap
        # out of how long the run happened to be calm, and that gap then RAMPS
        # the very reaction a cold regulator most needs.
        if self.n_up >= 1.0:
            self.excess = max(0.0, self.gap_up / self.since_up - 1.0) if self.gap_up else 0.0
            if self.prev_gap_up > 0.0:
                # d2y/dx2. excess asks whether THIS step was early; acceleration
                # asks whether the steps are converging. Intervals of 50, 40,
                # 30, 20 are each only mildly early and every one of them is a
                # machine running away.
                frac = (self.since_up - self.prev_gap_up) / self.prev_gap_up
                self.accel_now = frac
                self.n_acc_up += 1.0
                self.accel_up += (frac - self.accel_up) / self.n_acc_up
            self.prev_gap_up = self.since_up
            self.gap_up += (self.since_up - self.gap_up) / self.n_up
        else:
            self.excess = 0.0
        self.n_up += 1.0
        self.since_up = 0.0
        self._note_urgency()

    def _step_down(self) -> None:
        if self.n_down >= 1.0:
            self.excess = max(0.0, self.gap_down / self.since_down - 1.0) if self.gap_down else 0.0
            if self.prev_gap_down > 0.0:
                frac = (self.since_down - self.prev_gap_down) / self.prev_gap_down
                self.accel_now = frac
                self.n_acc_down += 1.0
                self.accel_down += (frac - self.accel_down) / self.n_acc_down
            self.prev_gap_down = self.since_down
            self.gap_down += (self.since_down - self.gap_down) / self.n_down
        else:
            self.excess = 0.0
        self.n_down += 1.0
        self.since_down = 0.0
        self._note_urgency()

    def urgency(self) -> float:
        """How far out of hand things are, from the two things that can say so.

        EXCESS -- this step arrived earlier than this machine's own mean
        interval, by that ratio.

        ACCELERATION -- the intervals are converging: this one shorter than
        the one before it, by that fraction. Only shrinking counts; a machine
        whose steps are spreading out is calming down and pays nothing.

        Both are read INSTANTANEOUSLY, not as run means. A mean acceleration
        cannot see a runaway: over [10]x10, 30, 20 the real shrink at the end
        averages with nine zeros and one jump to +0.133, the wrong sign. That
        is the same failure the flat baseline had. The means exist to be the
        cross-run prior for what this machine typically does, not to be the
        live signal.

        Added, because a step can be early without the trend being bad and the
        trend can be bad without any single step standing out."""
        return self.excess + max(0.0, -self.accel_now)

    def _note_urgency(self) -> None:
        """The machine's own out-of-hand line, built from the same means.

        Taken over STEPS, not over reads. Urgency only exists on a read where
        the machine moved; averaged over every read it would sit near zero,
        because most reads do not move, and then any step at all would look
        extreme.

        Mean plus mean-deviation -- the shape the baseline and volatility
        already use. A machine whose steps normally arrive hot has a high line
        and is not punished for being itself; a placid one has a low line and
        a small surprise is enough."""
        u = self.urgency()
        if self.n_u >= 1.0:
            self.u_dev += (abs(u - self.u_mean) - self.u_dev) / self.n_u
        self.n_u += 1.0
        self.u_mean += (u - self.u_mean) / self.n_u

    def threshold(self) -> Optional[float]:
        """The learned line, or None until there is a mean AND a spread."""
        return self.u_mean + self.u_dev if self.n_u >= 2.0 else None

    def out_of_hand(self, level: float) -> bool:
        """Two commands, and the machine's own one leads.

        LEARNED (Aman, 2026-09-21: "machine takes mean all along and use it
        craft thresholds and we progress"). Once steps have been measured, this
        machine's mean urgency plus its own spread is the line, and it moves
        with the machine.

        MANUAL, second in command ("the thresholds we put manually as third
        party ... makes operator second in command"). A number we did not pick
        and cannot learn our way out of: NSProcessInfoThermalStateCritical, the
        level at which Apple itself says the machine is in trouble. It is the
        whole of the line while the regulator is cold, before a single step has
        been measured, and it stays afterwards as the floor the learned line is
        not allowed to talk the machine out of."""
        if level >= self.CRITICAL:
            return True
        t = self.threshold()
        return t is not None and self.urgency() > t

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
        return max(0.0, cur - self.baseline) * (1.0 + self.volatility + self.urgency())

    def k_thermal(self, k_max: float, level: Optional[float] = None) -> float:
        """k as a damped second-order system (Aman, 2026-09-21: "we have second
        order differential equation on f(x) = k, and damping is temp ... as
        tendency to get f(x) = 4").

            k'' = kappa * (target - k)  -  c * k'

        A SPRING toward the target -- the standing tendency back to k_max that
        the heat is what holds k away from -- and DAMPING set by temperature.

        The first-order ramp this replaces had no velocity. It could only
        chase: a k that had been falling all run met each new target as if
        from rest, and every read paid the same fraction of a gap it had
        already been closing for an hour. With a velocity, a machine already
        on its way down carries that into the next read, and one holding still
        does not twitch.

        Neither coefficient is picked:

          KAPPA -- the measured rate, per direction. 1/gap_down going up and
          1/gap_up going down, so k climbs back at the pace this machine cools
          and falls at the pace it heats. Rate and gap are reciprocals; the
          interval is the only place the speed exists, since macOS publishes an
          ordinal 0-3 and "levels per move" is the constant 1 on every machine
          alive.

          C -- critical damping for that spring, PLUS how fast the machine is
          moving right now measured against its own line.

          The critical part is derived, not chosen. Damping is applied
          implicitly, v <- (v + kappa*d) / (1 + c), because the explicit
          v -= c*v form diverges at exactly the c this design produces; that
          discretisation is non-oscillatory for c >= kappa + 2*sqrt(kappa),
          not the textbook 2*sqrt(kappa), which leaves it underdamped and
          rings (measured: 4 turning points and a 2.8% overshoot at kappa=1).
          Below the line k oscillates around its own target forever, which on
          a machine that is behaving is the least excusable place to thrash.

          The rest is the temperature term (Aman, 2026-09-21: "damping term is
          current rate of change in comparison the mean threshold"). Not the
          absolute level -- the CURRENT rate of change over the line this
          machine learned for itself, urgency / threshold. It is 0 for a
          machine moving at its own usual pace, rises to 1 at the line, and
          past the line there is no damping question left because the equation
          is abandoned. So a machine getting jumpy stops chasing its target
          and holds still, which is the whole of what k thrash was.

        A direction with no measured interval has no equation to run, so k
        snaps. An unmeasured rate is not invented, and a regulator that has
        never seen the machine move has no business being gentle with it.

        k is clamped into [1, k_max] with the velocity zeroed at the wall: a
        spring that keeps its momentum at a limit stores energy and rebounds,
        and this one would rebound to more experts than the RAM holds."""
        lvl = float(self.last if level is None else level)
        target = float(k_max) ** (1.0 / (1.0 + self.pressure(level)))
        if self.k is None:
            self.k, self.v = target, 0.0
            return self.k
        d = target - self.k
        if d < 0.0 and self.out_of_hand(lvl):
            # OUT OF HAND: the equation does not apply. It describes a machine
            # drifting around its operating point, and this one is not. Snap,
            # and kill the velocity so the spring cannot carry the overshoot
            # back out. Tightening only -- a machine is allowed to be quick to
            # protect itself and slow to trust that it is safe.
            self.k, self.v = target, 0.0
            return self.k
        gap = self.gap_down if d > 0 else self.gap_up   # rising k <=> the machine cooled
        if gap <= 0.0:
            self.k, self.v = target, 0.0
            return self.k
        kappa = min(1.0, 1.0 / gap)
        t = self.threshold()
        heat = self.urgency() / t if t else 0.0     # 0 at its own pace, 1 at the line
        c = kappa + 2.0 * math.sqrt(kappa) + heat
        self.v = (self.v + kappa * d) / (1.0 + c)
        self.k += self.v
        if self.k > k_max:
            self.k, self.v = float(k_max), 0.0
        elif self.k < 1.0:
            self.k, self.v = 1.0, 0.0
        return self.k

    def state(self) -> Dict[str, float]:
        return {"n": self.n, "baseline": self.baseline, "volatility": self.volatility,
                "peak": self.peak, "last": float(self.last or 0.0), "run": self.run,
                "gap_up": self.gap_up, "gap_down": self.gap_down, "excess": self.excess,
                "accel_up": self.accel_up, "accel_down": self.accel_down,
                "accel_now": self.accel_now,
                "urgency": self.urgency(),
                "u_mean": self.u_mean, "u_dev": self.u_dev,
                "threshold": float(self.threshold() or 0.0),
                "v": self.v,
                "k": float(self.k if self.k is not None else 0.0)}

    def seed(self, baseline: float = 0.0, volatility: float = 0.0,
             gap_up: float = 0.0, gap_down: float = 0.0,
             accel_up: float = 0.0, accel_down: float = 0.0,
             u_mean: float = 0.0, u_dev: float = 0.0, **_) -> "ThermalRegulator":
        """A SOFT prior from what previous runs measured on this machine.

        A cold regulator knows nothing, so it spends its first reads throttling
        a machine it has not learned the normal of yet -- on the archived run
        k_thermal was 2.758 at the first health record and did not reach 3.9
        until clock 12k. That warm-up is pure loss: the operating point was
        already measured, 580 records of it, in the run before.

        Seeded with ONE observation's worth, never with the prior run's own
        count. The prior is a starting guess about a machine whose room, load
        and fan curve have all moved since; this run's first real read weighs
        as much as the whole of it, and by ten reads it is a tenth. It removes
        the cold start without being able to outvote the present.

        A direction with no measured interval is left at zero rather than
        filled in: the archived 500k run never changed thermal level once, so
        its logs carry an operating point and nothing about rates."""
        self.baseline = float(baseline)
        self.volatility = float(volatility)
        self.n = 1.0
        if gap_up > 0:
            self.gap_up, self.n_up = float(gap_up), 1.0
        if gap_down > 0:
            self.gap_down, self.n_down = float(gap_down), 1.0
        # acceleration is seeded only where an interval was: a d2y/dx2 without
        # a dy/dx under it is a trend in a rate this machine has never shown.
        if gap_up > 0:
            self.accel_up, self.n_acc_up = float(accel_up), 1.0
        if gap_down > 0:
            self.accel_down, self.n_acc_down = float(accel_down), 1.0
        # the out-of-hand line, same one observation's worth. It needs a spread
        # as well as a mean, so it comes back only once this run has taken a
        # step of its own: a threshold nothing here has confirmed does not get
        # to fire on its own authority.
        if u_dev > 0:
            self.u_mean, self.u_dev, self.n_u = float(u_mean), float(u_dev), 1.0
        return self


def thermal_prior(log: str) -> Optional[Dict[str, float]]:
    """What a previous run's log says about this machine's thermal behaviour.

    Reads the `thermal=` field of every health record. Returns None rather
    than a guess when the log carries no records."""
    import re
    lv: List[float] = []
    try:
        with open(log, errors="ignore") as fh:
            for ln in fh:
                if ln.startswith("[health b"):
                    m = re.search(r" thermal=([0-9.]+)", ln)
                    if m:
                        lv.append(float(m.group(1)))
    except OSError:
        return None
    if not lv:
        return None
    gaps_up: List[float] = []
    gaps_down: List[float] = []
    since_up = since_down = 0.0
    for a, b in zip(lv, lv[1:]):
        since_up += 1.0
        since_down += 1.0
        if b > a:
            gaps_up.append(since_up); since_up = 0.0
        elif b < a:
            gaps_down.append(since_down); since_down = 0.0
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"baseline": mean(lv),
            "volatility": mean([abs(b - a) for a, b in zip(lv, lv[1:])]),
            "gap_up": mean(gaps_up),
            "gap_down": mean(gaps_down),
            "n": float(len(lv))}


# ---------------------------------------------------------------- cross-run
# The regulator learns a machine, and the machine outlives the run. Everything
# it learned used to die with state/dume, so every run paid the cold start
# again: snapping on the first step, and k_thermal 2.758 at the archived run's
# first health record against 4.0 once settled.
#
# ONLY THE MEAN is kept. Not the runs, not their series -- a running mean per
# field with the count needed to keep it running, folded once per finished run
# so every run weighs the same however long it was. The file sits beside
# state/dume rather than inside it, because a clean start is meant to wipe the
# weights and the clock, not what the machine is.
# field -> the attribute holding how many times THIS run measured it. A field
# with no observations contributes nothing; it is not folded in as a zero.
# Acceleration needs this: 0.0 accel means "steps at a steady rate", a real
# reading, where 0.0 gap means "never stepped".
_PRIOR_FIELDS = {"baseline": "n", "volatility": "n", "gap_up": "n_up",
                 "gap_down": "n_down", "accel_up": "n_acc_up", "accel_down": "n_acc_down",
                 "u_mean": "n_u", "u_dev": "n_u"}


def _prior_path() -> Path:
    return Path(C.STATE_DIR).parent / "thermal_prior.json"


def load_prior(path: Optional[Path] = None) -> Optional[Dict[str, float]]:
    """The cross-run mean, or None if no run has contributed a field yet."""
    try:
        raw = json.loads(Path(path or _prior_path()).read_text())
    except (OSError, ValueError):
        return None
    out = {f: float(raw[f][0]) for f in _PRIOR_FIELDS if raw.get(f, [0, 0])[1] >= 1}
    out["runs"] = float(raw.get("runs", 0))
    return out if len(out) > 1 else None


def fold_prior(reg: "ThermalRegulator", path: Optional[Path] = None) -> Dict[str, float]:
    """Fold ONE finished run's regulator into the cross-run mean.

    A direction this run never stepped in contributes nothing to that field --
    no observation is not an interval of zero, and averaging it in as one would
    drag the mean toward a rate no machine has."""
    p = Path(path or _prior_path())
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        raw = {}
    out: Dict[str, object] = {"runs": float(raw.get("runs", 0)) + 1.0}
    for f, counter in _PRIOR_FIELDS.items():
        m, n = raw.get(f, [0.0, 0.0])
        if float(getattr(reg, counter, 0.0)) < 1.0:
            out[f] = [float(m), float(n)]      # not measured this run: contributes nothing
            continue
        n = float(n) + 1.0
        out[f] = [float(m) + (float(getattr(reg, f)) - float(m)) / n, n]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return {f: out[f][0] for f in _PRIOR_FIELDS}
