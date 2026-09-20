"""Apex-Nadir: the token allocation given to an expert.

The allocation law is FITTED, never set. It replaces three constants that used
to decide allocation by fiat — k_upper (how many experts), SPAN_MIN (how big a
span) and the EXPERT_GEN_TOKENS divisor floor — with curves measured off the
run itself.

    probe schedule   t_lo = T mod n^2,  t_mid = sqrt(T^2 - t_lo^2)/2,  t_hi = T
                     n = floor(sqrt(T)) decremented until n^2 < T strictly.
                     A function of T alone: zero constants.
    envelopes        A = Q.9 (apex, overfit boundary), M = Q.5 (GROUNDING BASE),
                     N = Q.1 (nadir, underfit boundary). Every expert feeds all
                     three curves.
    per-expert       M is the base of the equation; A - N is the spread, and an
                     expert's own overall rank u in [0,1] places it inside that
                     spread:

                         Q_e(t) = exp( M(t) + (u - 0.5) * (A(t) - N(t)) )

                     The envelopes are fitted on the RAW delta, in nats, where
                     the spread is naturally additive; the exponential is taken
                     last. Interpolating in likelihood-ratio space instead lets
                     M - (A-N)/2 go negative whenever the envelope is wide
                     (N ~ e^-3, A ~ e^2, M ~ 1), and clipping that at zero
                     leaves flat regions the argmax wanders into — it handed the
                     WORST expert the entire input.

                     u = 0.5 reproduces M exactly. This is what makes the
                     allocation PER EXPERT rather than one pool-wide number —
                     the old system's documented failure was an allocation()
                     that took no expert_id and so could only size the pool.
    cost             c(t) = a + b t, least squares over MEASURED latency.
                     `a` is load-in overhead and is NEVER forced to zero: doing
                     so makes g monotone decreasing and pins the argmax at t_lo.
    goldilocks       g_e(t) = Q_e(t) / c(t),  t*_e = argmax over [t_lo, t_hi].
    allocation law   log t* = log alpha + beta log T  =>  ALLOC(T) = alpha T^beta
                     k(T) = T / ALLOC(T) = T^(1-beta) / alpha

Quality is banked as the RAW delta in nats. It is signed by construction (canary
C fails the run if the negative fraction leaves 15-85%), which is exactly why
the exponential is applied to the interpolated curve rather than to each
observation: exp() of the combination is always positive, so g = Q/c needs no
clipping and has no sign hazard.

The bank is CLEARED at every refit: when a new graph is formed the old logs go
with it. The curves are therefore non-stationary by construction, which is the
only honest choice while the experts under them are still being trained — a
quality measured 500 batches ago was measured on a different expert. The FITTED
graph is what persists between refits, and a refused fit keeps the previous one.

THE INVARIANT: k is an allocation, never a gate. Scarce tokens shrink k; they
never zero an expert's share, and the nadir floor is informational — it must not
gate execution. K=0 is Timeline A's business, and Timeline A is decided by
CONFIDENCE, not by this module.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C

E_SPAN = math.e          # §7 span test: T_max/T_min >= e. Not a tunable — it is `e`.


def probe_sizes(T: int) -> Tuple[int, int, int]:
    """(t_lo, t_mid, t_hi) from T alone. t_lo = T mod n^2 SCATTERS with T
    (50->1, 53->4, 55->6, 64->15), which sweeps the low probe across the range
    over successive inputs — span diversity for the fit, for free."""
    T = max(1, int(T))
    n = int(math.isqrt(T))
    while n > 1 and n * n >= T:
        n -= 1
    t_lo = max(1, T % (n * n)) if n >= 1 else 1
    t_lo = min(t_lo, T)
    t_mid = math.sqrt(max(0.0, float(T) ** 2 - float(t_lo) ** 2)) / 2.0
    t_mid = int(round(min(max(t_mid, t_lo), T)))          # clamped: falls below t_lo for T <~ 20
    return int(t_lo), int(t_mid), int(T)


def _quadratic(ts: Sequence[float], vs: Sequence[float]) -> np.ndarray:
    """Degree 2 through three points: three points, three coefficients, exact."""
    t = np.asarray(ts, dtype=np.float64)
    v = np.asarray(vs, dtype=np.float64)
    if len(t) < 3 or len(np.unique(t)) < 3:
        return np.array([0.0, 0.0, float(np.mean(v)) if len(v) else 0.0])
    return np.polyfit(t, v, 2)


class AllocLaw:
    """Banks probes, fits the curves, serves ALLOC(T) and k(T).

    Refit cadence is E inputs (Aman: "recomputed every total number expert
    times"). Until the first ADMISSIBLE fit, ALLOC(T) = T and k = 1 — the §7
    identity fallback, which is a refusal to allocate, not a guess."""

    def __init__(self, n_experts: int = C.E):
        self.E = int(n_experts)
        self.q: List[Tuple[float, float]] = []        # (span t, quality exp(d))
        self.lat: List[Tuple[float, float]] = []      # (span t, seconds)
        self.peaks: List[Tuple[float, float]] = []    # (T, t*)
        self.A = self.M = self.N = None               # quadratic coefficients
        self.cost = None                              # (a, b)
        self.alpha, self.beta = 1.0, 1.0              # identity: ALLOC(T) = T, k = 1
        self.fitted = False
        self.n_seen = 0
        self.n_fits = 0
        self.grid_lo, self.grid_hi = 1.0, 1.0
        self.last_reject = ""

    # ── banking ─────────────────────────────────────────────────────────────
    def sample(self, rank: Dict[int, float]) -> List[int]:
        """§2: the top sqrt(E) and the bottom sqrt(E) by overall rank — 20 of 100.
        ONLY these feed the curves. Every expert still RECEIVES a per-expert
        allocation; restricting the fit to the extremes is what saves the
        compute, and the extremes are what give the envelope its spread."""
        if not rank:
            return []
        half = max(1, int(math.isqrt(self.E)))
        order = sorted(rank, key=lambda e: -rank[e])
        return list(dict.fromkeys(order[:half] + order[-half:]))

    def width_varies(self) -> Optional[float]:
        """|A' - N'| at the middle of the fitted range. sign(dt*/du) = sign(A'-N'),
        so when this is ~0 every expert gets an identical allocation and the
        per-expert mechanism is doing nothing. Health reads it."""
        if self.A is None:
            return None
        t = 0.5 * (self.grid_lo + self.grid_hi) if self.grid_hi > self.grid_lo else 1.0
        return float(abs((2 * self.A[0] * t + self.A[1]) - (2 * self.N[0] * t + self.N[1])))

    @staticmethod
    def placement(rank: Dict[int, float]) -> Dict[int, float]:
        """u in [0,1] per expert: where it sits in the pool's own rank spread.
        Percentile position, not the raw rate, so one outlier cannot drag every
        other expert to the same end of the envelope."""
        if not rank:
            return {}
        if len(rank) == 1:
            return {int(next(iter(rank))): 0.5}
        order = sorted(rank, key=lambda e: rank[e])
        last = len(order) - 1
        return {int(e): i / last for i, e in enumerate(order)}

    def observe(self, t: int, delta: float, seconds: Optional[float] = None) -> None:
        """One probe: an expert given a span of t tokens produced delta nats."""
        if not math.isfinite(delta):
            return
        self.q.append((float(max(1, int(t))), float(delta)))          # raw nats
        if seconds is not None and math.isfinite(seconds) and seconds > 0:
            self.lat.append((float(max(1, int(t))), float(seconds)))

    def input_seen(self, T: int) -> bool:
        """Called once per input. Banks this input's goldilocks peak when the
        curves exist, and returns True on the batches where a refit is due."""
        self.n_seen += 1
        if self.A is not None:
            peak = self.peak(T)
            if peak is not None:
                self.peaks.append((float(T), float(peak)))
        return (self.n_seen % self.E) == 0

    # ── the fits ────────────────────────────────────────────────────────────
    def fit(self) -> bool:
        """Refit envelopes, cost and the allocation law, then CLEAR the logs:
        a new graph retires the observations that made the old one."""
        self._fit_envelopes()
        self._fit_cost()
        ok = self._fit_law()
        self.n_fits += 1
        self.q.clear()
        self.lat.clear()
        self.peaks.clear()
        return ok

    def _fit_envelopes(self) -> None:
        """§3. Probe sizes vary across inputs (t_lo scatters), so the bank is
        split into a low/mid/high third by t and each third contributes one
        point: its median t, and the Q.9/Q.5/Q.1 of q within it."""
        if len(self.q) < 9:
            return
        arr = np.array(self.q, dtype=np.float64)
        arr = arr[np.argsort(arr[:, 0])]
        thirds = np.array_split(arr, 3)
        ts, a, m, n = [], [], [], []
        for part in thirds:
            if len(part) == 0:
                return
            ts.append(float(np.median(part[:, 0])))
            a.append(float(np.quantile(part[:, 1], 0.9)))
            m.append(float(np.quantile(part[:, 1], 0.5)))
            n.append(float(np.quantile(part[:, 1], 0.1)))
        if len(set(ts)) < 3:
            return
        self.grid_lo, self.grid_hi = float(min(ts)), float(max(ts))
        A, M, N = _quadratic(ts, a), _quadratic(ts, m), _quadratic(ts, n)
        # concavity guard: an upward-opening grounding curve has no interior peak,
        # so the argmax pins to a boundary forever and the fit can never say so —
        # exact interpolation leaves zero residual to detect it with. Refuse and
        # keep the previous graph, the same discipline §7 applies to the law.
        if M[0] >= 0.0:
            self.last_reject = f"grounding curve convex (m2={M[0]:.3g}) — probe range too narrow"
            return
        self.A, self.M, self.N = A, M, N

    def _fit_cost(self) -> None:
        """§4. c(t) = a + b t. The intercept is load-in overhead and stays
        free — clamping a negative fitted `a` to zero and refitting through the
        origin is what pinned the old argmax at t_lo forever."""
        if len(self.lat) < 3:
            return
        arr = np.array(self.lat, dtype=np.float64)
        if len(np.unique(arr[:, 0])) < 2:
            return
        b, a = np.polyfit(arr[:, 0], arr[:, 1], 1)
        self.cost = (float(a), float(b))

    def _fit_law(self) -> bool:
        """§6 + §7. Rejection keeps the previous model and falls back to the
        identity, which yields k = 1 — a refusal, not a guess."""
        if len(self.peaks) < 3:
            self.last_reject = f"only {len(self.peaks)} peaks banked"
            return self.fitted
        arr = np.array(self.peaks, dtype=np.float64)
        T, ts = arr[:, 0], arr[:, 1]
        good = (T > 0) & (ts > 0)
        T, ts = T[good], ts[good]
        if len(T) < 3:
            self.last_reject = "fewer than 3 positive peaks"
            return self.fitted
        span = float(T.max() / max(T.min(), 1e-9))
        if span < E_SPAN:                       # a power law fitted inside a decade-fraction
            self.last_reject = f"span x{span:.3f} < e"
            return self.fitted
        beta, log_alpha = np.polyfit(np.log(T), np.log(ts), 1)
        if not (0.0 < beta <= 1.0):             # beta<=1 => k non-decreasing in T
            self.last_reject = f"beta {beta:.4f} outside (0, 1]"
            return self.fitted
        self.alpha, self.beta = float(np.exp(log_alpha)), float(beta)
        self.fitted = True
        self.last_reject = ""
        return True

    # ── serving ─────────────────────────────────────────────────────────────
    def peak(self, T: int, u: float = 0.5) -> Optional[int]:
        """t*_e = argmax Q_e(t)/c(t) on a 128-point grid over [t_lo, t_hi],
        where Q_e = exp(M + (u - 0.5)(A - N)): the grounding curve is the base,
        the expert's rank places it within the apex-nadir spread, and the
        exponential is taken on the combination so Q_e is always positive."""
        if self.A is None:
            return None
        t_lo, _, t_hi = probe_sizes(T)
        if t_hi <= t_lo:
            return int(t_hi)
        grid = np.linspace(float(t_lo), float(t_hi), 128)
        a, m, n = (np.polyval(self.A, grid), np.polyval(self.M, grid), np.polyval(self.N, grid))
        z = m + (float(np.clip(u, 0.0, 1.0)) - 0.5) * (a - n)
        q = np.exp(np.clip(z, -50.0, 50.0))                # strictly positive by construction
        c = np.polyval(np.array(self.cost[::-1]), grid) if self.cost else np.ones_like(grid)
        c = np.where(c > 1e-9, c, 1e-9)         # cost must never divide by <= 0
        return int(round(float(grid[int(np.argmax(q / c))])))

    def alloc(self, T: int, u: Optional[float] = None) -> int:
        """Tokens for THIS expert. The curves answer it directly when they
        exist; the power law is the pool-wide fallback; identity when neither
        is fitted."""
        T = max(1, int(T))
        if u is not None and self.A is not None:
            pk = self.peak(T, u)
            if pk:
                return int(max(1, min(T, pk)))
        if not self.fitted:
            return T
        return int(max(1, min(T, round(self.alpha * (T ** self.beta)))))

    def z(self, t: float, u: float) -> Optional[float]:
        """The interpolated curve in NATS at span t for an expert placed at u.
        This is the predicted gain: z > 0 means the expert is expected to save
        Central nats, z <= 0 means it is expected to add nothing or hurt."""
        if self.A is None:
            return None
        uu = float(np.clip(u, 0.0, 1.0)) - 0.5
        return float(np.polyval(self.M, t) + uu * (np.polyval(self.A, t) - np.polyval(self.N, t)))

    def predicted_gain(self, T: int, u: float) -> Optional[float]:
        """Nats this expert is predicted to add at the budget it would be given."""
        if self.A is None:
            return None
        return self.z(float(self.alloc(T, u)), u)

    @staticmethod
    def budget(alloc_tokens: int, span_tokens: int, context_share: int) -> int:
        """The tokens an expert may WRITE. The allocation is a SOFT LIMIT, and
        this is the common-sense clamp that keeps it from running away:

          floor    EXPERT_GEN_TOKENS  — an expert never writes less than this
          ceiling  the span it read   — a note about an excerpt is not longer
                                        than the excerpt
          ceiling  TARGET_MAX_TOKENS — a note exists to help produce y and is
                                        never longer than y itself
          ceiling  its share of Central's context — every note has to FIT

        The TARGET_MAX_TOKENS ceiling is the one that binds. Central's context
        share does NOT: it divides a nominal 32k window, so it permits ~31k
        generated tokens at k=1 and never clamps anything. Relying on it alone
        asked an expert to generate a whole input's worth of text and crashed
        the machine.

        Without this a 1354-token span asks for 1354 generated tokens, 40x the
        floor, and expert generation is the dominant cost in the loop. Both
        ceilings are measured quantities; neither is a tuned constant."""
        cap = min(int(span_tokens), int(context_share), C.TARGET_MAX_TOKENS)
        return int(max(C.EXPERT_GEN_TOKENS, min(int(alloc_tokens), max(C.EXPERT_GEN_TOKENS, cap))))

    def k(self, T: int) -> int:
        """k = T / ALLOC(T). An ALLOCATION, never a gate: floored at 1 so a
        scarce budget shrinks k but never zeroes an expert's share."""
        return int(max(1, min(int(T), round(max(1, int(T)) / max(1, self.alloc(T))))))

    def mean_pass_seconds(self) -> Optional[float]:
        """tau: the pool's average measured pass time. Not a configured budget —
        it is what a pass HAS been costing, refit from the same cleared latency
        log as c(t), so it moves with the machine's thermal state instead of
        being a number someone once chose."""
        if not self.lat:
            return None
        return float(np.mean([s for _, s in self.lat]))

    def k_effective(self, T: int, k_ram: int, k_thermal: Optional[float] = None) -> int:
        """Aman's general equation (2026-09-10), made dimensional.

        His form was `1/experts + 1/estimated_time + tokens_per_second`. The
        three terms as written are a count, a per-second and a rate, so the sum
        is not a number of experts. What survives is the SHAPE, which is right:
        a soft minimum where the tightest constraint dominates smoothly, instead
        of the hard `min` the scheduler uses. Each constraint is converted into
        a number of experts and they are combined by HARMONIC MEAN — n/(sum 1/x),
        not 1/(sum 1/x). The latter returns 2 when both say 4; the mean returns 4
        when they agree and collapses toward the smaller when they do not.

            k_gate = T / ALLOC(T)          what apex-nadir asks for
            k_ram  = the scheduler's bound (still a HARD clamp afterwards: a soft
                                           blend can land above it, and that is
                                           an OOM, not a preference)

        THERE IS NO k_time TERM, and that is deliberate. It used to be
        tau / (a + b*alloc(T)), where tau = mean_pass_seconds() is the mean of
        the latency bank and (a, b) is the line fitted to that SAME bank — one
        curve divided by itself at two points, with no deadline, budget or
        latency target entering anywhere. Measured on the live fit it sat at
        1.08-1.14 across T = 64..2048 and did not move when the machine changed
        speed, because tau and c(t) scale together and cancel. A harmonic mean is
        dominated by its smallest term, so that ~1 alone set k: across all 1566
        batches of the last run k was 2 while k_gate asked for 11 to 114, and
        3/(1/k_gate + 1 + 1/k_ram) is below 3 for EVERY k_gate and k_ram that
        exist. A term carrying no information was outvoting both terms that do.

        A real time term needs a real budget — a deadline, or Central's own
        measured pass time to spend against. Nothing in the system measures one
        yet, so the honest form is the two constraints that are grounded and
        independent. Put the third back when there is a clock to answer to.

        THE THIRD TERM IS THE DEVICE (Aman, 2026-09-20): "it is a constant tug
        of war — the device wants k less so it doesn't get hot, its input is
        temperature; the system wants to do work fastest so it wants k high, as
        high k adds more tokens per second and processing". k_gate is the system
        pulling up, k_thermal is the device pulling down, and the harmonic mean
        is the negotiation. k_thermal comes from NSProcessInfo's thermal state
        via Scheduler.k_thermal, so the device's side has a real measured input
        rather than an assumed one.

        It is a bound as well as a blend term, exactly like k_ram: a soft mean
        over three terms can still land above what the device is asking for, and
        at `serious` the OS is already throttling, so running more experts makes
        the thing it is complaining about worse.

        Until the allocation law is fitted this degrades to k_gate, which is the
        §7 identity fallback — a refusal, not a guess.

        The device does NOT enter as a third voice. Heat and RAM are the same
        axis — both say how much this machine will do right now — so thermal
        pressure MODULATES the physical bound rather than adding a term. That
        also makes a nominal device exactly free: k_thermal == k_ram at nominal,
        so turning the sensor on cannot move k until the machine is actually
        warm. A third harmonic term would have shifted every k the moment the
        sensor was installed, which is a silent behaviour change, not a
        measurement.
        """
        k_gate = max(1.0, float(self.k(T)))
        # what the machine will allow RIGHT NOW: the physical bound, tightened
        # by whatever the device is currently asking for
        k_dev = float(k_ram) if k_thermal is None else min(float(k_ram), max(1.0, float(k_thermal)))
        k_dev = max(1.0, k_dev)
        eff = 2.0 / (1.0 / k_gate + 1.0 / k_dev)
        return max(1, min(int(round(eff)), max(1, int(round(k_dev)))))

    def state(self) -> Dict[str, object]:
        return {"fitted": self.fitted, "alpha": round(self.alpha, 5), "beta": round(self.beta, 5),
                "q": len(self.q), "lat": len(self.lat), "peaks": len(self.peaks),
                "fits": self.n_fits, "reject": self.last_reject,
                "width_var": (None if self.width_varies() is None else round(self.width_varies(), 6)),
                "cost": (None if self.cost is None else (round(self.cost[0], 5), round(self.cost[1], 7)))}
